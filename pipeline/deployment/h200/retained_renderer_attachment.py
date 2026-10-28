"""Read-only admission of the existing eight-renderer fleet for new gateways.

No renderer start/stop, native step, model inference, allocation mutation or
cached-render channel creation is implemented here. Historical readiness is
accepted only together with the same actual live process/boot identity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import socket
import struct
import subprocess

SCHEMA = 'roborender.retained-eight-renderer-attachment.v1'
OWNER = 'dagger-leo-eval-scale8-20261024'
ALLOCATION = 'leo-rr-leo-5c74c8d595d5-r1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 8 << 20:
        raise ValueError('bounded absolute regular evidence required')
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('bound evidence bytes changed')
    return json.loads(raw)


def validate_fleet(value):
    rows = value.get('workers', [])
    if (value.get('schema') != 'roborender.scaled-renderer-fleet.v1'
            or value.get('allocation_owner') != OWNER or value.get('allocation_name') != ALLOCATION
            or any(value.get(k) is not False for k in
                   ('automatic_allocation_release', 'automatic_restart', 'compute_release_authorized'))
            or not value.get('hostname') or not value.get('boot_id') or not value.get('node_id')
            or len(rows) != 8):
        raise ValueError('exact retained eight-renderer fleet and retention policy required')
    for slot, row in enumerate(rows):
        if (row.get('slot') != slot or row.get('port') != 18560 + slot or row.get('host') != '127.0.0.1'
                or type(row.get('pid')) is not int or row['pid'] <= 1
                or re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', row.get('gpu_uuid', '')) is None
                or re.fullmatch(r'[0-9a-f]{32}', row.get('server_instance', '')) is None
                or any(re.fullmatch(r'[0-9a-f]{64}', row.get(k, '')) is None for k in
                       ('source_sha256', 'generate_source_sha256', 'readiness_receipt_sha256'))
                or row.get('source_sha256') != value.get('server_source_sha256')
                or set(row.get('identity', {})) != {'boot_id', 'start_ticks', 'cmdline_sha256'}
                or row['identity']['boot_id'] != value['boot_id']):
            raise ValueError('ordered renderer process/GPU/source pins differ')
    if any(len({r[k] for r in rows}) != 8 for k in ('pid', 'gpu_uuid', 'server_instance', 'port')):
        raise ValueError('renderer fleet aliases an owner, GPU, instance or port')
    return value


def verify_release(source, manifest_path, expected_source_sha):
    source = Path(source)
    if not source.is_absolute() or source.resolve(strict=True) != source or source.is_symlink():
        raise ValueError('physical immutable source required')
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 8 << 20:
        raise ValueError('bounded source manifest required')
    value = json.loads(manifest_path.read_bytes()); rows = value.get('files', [])
    if (value.get('schema') != 'roborender.source-snapshot.v1' or not rows
            or value.get('file_count') != len(rows) or value.get('source_sha256') != expected_source_sha
            or digest(canonical(rows)) != expected_source_sha
            or [r['path'] for r in rows] != sorted({r['path'] for r in rows})):
        raise ValueError('source manifest identity differs')
    for row in rows:
        relative = Path(row['path']); path = source / relative
        if relative.is_absolute() or '..' in relative.parts or path.resolve(strict=True) != path:
            raise ValueError('source path escaped its physical release')
        raw = path.read_bytes()
        if (digest(raw) != row['sha256'] or len(raw) != row['bytes']
                or bool(path.stat().st_mode & 0o111) != row['executable']):
            raise ValueError('frozen source changed: ' + row['path'])
    return value


def process_identity(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    if fields[0] == 'Z':
        raise RuntimeError('retained renderer became a zombie')
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=fields[19], cmdline_sha256=digest((root / 'cmdline').read_bytes()))


def process_device_visibility(pid):
    return [v.split(b'=', 1)[1].decode() for v in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
            if v.startswith(b'CUDA_VISIBLE_DEVICES=')]


def process_gpu_pids(pid):
    """Only kernel-attested PID namespace aliases, never guessed host PIDs."""
    status = (Path('/proc') / str(pid) / 'status').read_text()
    row = next((line.split(':', 1)[1].split() for line in status.splitlines() if line.startswith('NSpid:')), [])
    return {pid, *(int(value) for value in row)}


def owns_listener(pid, port):
    """Match the exact loopback listening socket inode to the live process FD."""
    expected = '0100007F:' + f'{port:04X}'
    rows = [line.split() for line in Path('/proc/net/tcp').read_text().splitlines()[1:]]
    inodes = [r[9] for r in rows if r[1] == expected and r[3] == '0A']
    if len(inodes) != 1:return False
    for path in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            if os.readlink(path) == 'socket:[' + inodes[0] + ']':return True
        except FileNotFoundError:continue
    return False


def metadata(port):
    def exchange(stream, value):
        def receive(count):
            result = bytearray()
            while len(result) < count:
                block = stream.recv(count - len(result))
                if not block:raise ConnectionError('renderer metadata EOF')
                result.extend(block)
            return bytes(result)
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        stream.sendall(struct.pack('!Q', len(raw)) + raw)
        count = struct.unpack('!Q', receive(8))[0]
        if not 0 < count < 65536:raise ValueError('renderer metadata response exceeds bound')
        return pickle.loads(receive(count))
    with socket.create_connection(('127.0.0.1', port), timeout=10) as stream:
        if exchange(stream, {'cmd': 'ping'}) != {'ok': True}:
            raise RuntimeError('retained renderer ping failed')
        return exchange(stream, {'cmd': 'observation_contract'})


def observe(fleet, *, expected_gateways=None):
    """Actual H200-only read-only attestation; no process is created or stopped."""
    fleet = validate_fleet(fleet)
    if (socket.gethostname() != fleet['hostname']
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != fleet['boot_id']
            or os.environ.get('ROBORENDER_BETA_RUN_ID') != OWNER
            or os.environ.get('ROBORENDER_BETA_ROUND_ID') != '1'):
        raise RuntimeError('retained host/boot/allocation owner changed')
    node = os.environ.get('NODE_NAME') or os.environ.get('LEPTON_NODE_NAME')
    if node and node != fleet['node_id']:raise RuntimeError('retained node changed')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    devices = [[v.strip() for v in line.split(',')] for line in raw.splitlines()]
    if (len(devices) != 8 or [int(r[0]) for r in devices] != list(range(8))
            or [r[1] for r in devices] != [w['gpu_uuid'] for w in fleet['workers']]
            or any('H200' not in r[2] or float(r[3]) >= .75 * float(r[4]) for r in devices)):
        raise RuntimeError('actual eight H200s/UUID order or 25% memory headroom differs')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    clients = [[v.strip() for v in line.split(',')] for line in raw.splitlines() if line.strip()]
    allowed = {w['gpu_uuid']: process_gpu_pids(w['pid']) for w in fleet['workers']}
    if expected_gateways is not None:
        if (len(expected_gateways) != 8 or [g.get('slot') for g in expected_gateways] != list(range(8))
                or len({g.get('pid') for g in expected_gateways}) != 8):
            raise ValueError('exact ordered live gateway process receipts required')
        for gateway, worker in zip(expected_gateways, fleet['workers']):
            if (gateway.get('gpu_uuid') != worker['gpu_uuid'] or gateway.get('pid') == worker['pid']
                    or process_identity(gateway['pid']) != gateway.get('process_identity')
                    or process_device_visibility(gateway['pid']) != [worker['gpu_uuid']]
                    or not owns_listener(gateway['pid'], 18620 + gateway['slot'])):
                raise RuntimeError('actual admitted gateway process/UUID/listener changed')
            allowed[worker['gpu_uuid']].update(process_gpu_pids(gateway['pid']))
    if (any(len(r) != 2 or r[0] not in allowed or not r[1].isdigit()
            or int(r[1]) not in allowed[r[0]] for r in clients)
            or any(not any(r[0] == w['gpu_uuid'] and int(r[1]) in process_gpu_pids(w['pid']) for r in clients)
                   for w in fleet['workers'])):
        raise RuntimeError('unexpected retained GPU client population; no takeover')
    observed = []
    for worker in fleet['workers']:
        if process_identity(worker['pid']) != worker['identity']:
            raise RuntimeError('renderer PID/boot/start/cmdline identity changed')
        visible = process_device_visibility(worker['pid'])
        if visible != [worker['gpu_uuid']] or not owns_listener(worker['pid'], worker['port']):
            raise RuntimeError('renderer CUDA UUID or actual listener owner differs')
        prior = read_bound(worker['readiness_receipt_path'], worker['readiness_receipt_sha256'])
        if (prior['transport']['server_instance'] != worker['server_instance']
                or prior['server']['identity'] != worker['identity']
                or prior['server']['source_sha256'] != worker['source_sha256']):
            raise RuntimeError('live renderer no longer matches its original source/instance evidence')
        response = metadata(worker['port']); attestation = response.get('attestation', {})
        expected = dict(schema='roborender.dagger.student-observation.v1', mode='generated_only',
            renderer='Image3F', renderer_steps=5, renderer_cfg=1., teacache_threshold=.3,
            raytraced_policy_fallback=False)
        if (response.get('ok') is not True or attestation.get('contract') != expected
                or attestation.get('backend_class') != 'runtime.image3f_deploy.Image3FSession'
                or attestation.get('generate_source_sha256') != worker['generate_source_sha256']
                or process_identity(worker['pid']) != worker['identity']):
            raise RuntimeError('actual retained renderer runtime/contract changed')
        observed.append(dict(worker, observation_attestation=attestation))
    return dict(schema=SCHEMA, allocation_name=ALLOCATION, allocation_owner=OWNER,
        hostname=fleet['hostname'], node_id=fleet['node_id'], boot_id=fleet['boot_id'],
        renderers=observed, gpu_client_counts=[sum(r[0] == d[1] for r in clients) for d in devices],
        render_requests=0, renderer_restarts=0, allocation_retained=True, compute_release_authorized=False)


def gateway_identity(source, source_sha, fleet, slot, ready, *, expected_pid=None, proc_root=Path('/proc')):
    """Bind live /ready to the exact new frozen-code process and GPU mapping."""
    pid = ready.get('pid'); worker = fleet['workers'][slot]
    if (type(pid) is not int or pid <= 1 or expected_pid is not None and pid != expected_pid
            or ready.get('ok') is not True or ready.get('role') != 'render_student'
            or re.fullmatch('[0-9a-f]{64}', ready.get('gateway_instance_id', '')) is None):
        raise RuntimeError('actual gateway readiness identity differs')
    proc = proc_root / str(pid)
    argv = (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    expected = {'--port': str(18620 + slot), '--devices': 'cuda:0', '--image3f-ports': str(worker['port'])}
    if (argv[1:3] != ['-m', 'pipeline.dagger.services.render_student_server']
            or '--fake' in argv or any(argv.count(k) != 1 or argv[argv.index(k) + 1] != v for k, v in expected.items())
            or (proc / 'cwd').resolve(strict=True) != Path(source).resolve(strict=True)):
        raise RuntimeError('gateway executable/source/device/renderer route differs')
    env = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
    if (env.get(b'CUDA_VISIBLE_DEVICES') != worker['gpu_uuid'].encode()
            or env.get(b'PYTHONPATH', b'').split(b':')[0] != str(source).encode()
            or env.get(b'ROBORENDER_SERVICE_SOURCE_SHA256') != source_sha.encode()):
        raise RuntimeError('gateway live GPU/source environment differs')
    actual = process_identity(pid)
    if not owns_listener(pid, 18620 + slot):raise RuntimeError('gateway ready endpoint has another socket owner')
    return dict(slot=slot, pid=pid, gateway_instance_id=ready['gateway_instance_id'],
                gpu_uuid=worker['gpu_uuid'], source_sha256=source_sha, process_identity=actual)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fleet', required=True); p.add_argument('--fleet-sha256', required=True)
    p.add_argument('--service-ready');p.add_argument('--service-ready-sha256')
    a = p.parse_args()
    if bool(a.service_ready) != bool(a.service_ready_sha256):p.error('running attachment requires exact ready file and hash')
    gateways = None
    if a.service_ready:
        ready = read_bound(a.service_ready, a.service_ready_sha256)
        if ready.get('renderer_fleet_sha256') != a.fleet_sha256:
            raise ValueError('service ready belongs to another renderer fleet')
        gateways = ready['gateway_process_receipts']
    print(json.dumps(observe(read_bound(a.fleet, a.fleet_sha256), expected_gateways=gateways), sort_keys=True))


if __name__ == '__main__':main()
