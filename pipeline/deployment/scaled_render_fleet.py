"""Pinned eight-H200 Image3F fleet; never allocates, releases or auto-restarts.

Stage on CPU. Explicitly start only on the admitted new eight-GPU allocation.
Each GPU owns one process/session, private loopback port, lock and durable
launch intent. Lost launch acknowledgements require identity reconciliation;
an unknown process or occupied port is never stopped or replaced here.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import socket
import struct
import subprocess
import sys
import time
import uuid

SCHEMA = 'roborender.scaled-renderer-fleet.v1'
OWNER = 'dagger-leo-eval-scale8-20261024'
ALLOCATION = 'leo-rr-leo-5c74c8d595d5-r1'
CACHE_SCHEMA = 'roborender.image3f.cached-reply.v1'
SOURCE_SHA = '6b549fbbee8720e6cdecf97bb82a5ae860bf26666f90087a91972084efe92927'
REPO = Path('/workspace/project/roborender')
PYTHON = REPO / 'renderer/envs/image3f-py313-cu130-trt10133/venv/bin/python'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 << 20:
        raise ValueError('bounded regular fleet evidence required')
    if expected is not None and sha(path) != expected:
        raise ValueError('fleet evidence bytes differ')
    return json.loads(path.read_bytes())


def immutable(path, value):
    path = Path(path)
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError('immutable fleet evidence differs: ' + path.name)
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    if type(pid) is not int or pid <= 1:
        return None
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(start_ticks=fields[19], cmdline_sha256=sha(proc / 'cmdline'),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_config(config):
    if (not isinstance(config, dict) or config.get('schema') != SCHEMA
            or config.get('allocation_name') != ALLOCATION or config.get('allocation_owner') != OWNER
            or not isinstance(config.get('hostname'), str)
            or re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}', config['hostname']) is None
            or not isinstance(config.get('replica_id'), str) or not config['replica_id'].startswith(ALLOCATION + '-')
            or re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', config.get('boot_id', '')) is None
            or config.get('source_sha256') != SOURCE_SHA
            or config.get('automatic_restart') is not False
            or config.get('automatic_allocation_release') is not False
            or config.get('compute_release_authorized') is not False
            or not _sha(config.get('generate_source_sha256'))
            or not _sha(config.get('launcher_sha256'))):
        raise ValueError('exact new allocation, pinned source and retention policy required')
    root = Path(config.get('root', ''))
    if (not root.is_absolute() or '..' in root.parts or root == REPO / 'runtime'
            or not root.is_relative_to(REPO / 'runtime/evaluation-b16-20261024')):
        raise ValueError('fleet requires a separate bounded evaluation runtime')
    workers = config.get('workers')
    if not isinstance(workers, list) or len(workers) != 8:
        raise ValueError('exactly eight independent renderer workers required')
    for slot, worker in enumerate(workers):
        if (set(worker) != {'slot', 'gpu_uuid', 'port'} or worker['slot'] != slot
                or worker['port'] != 18560 + slot
                or not isinstance(worker['gpu_uuid'], str)
                or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', worker['gpu_uuid']) is None):
            raise ValueError('eight exact GPU UUIDs and loopback18560..18567 required')
    if len({row['gpu_uuid'] for row in workers}) != 8:
        raise ValueError('renderer fleet cannot alias physical GPUs')
    return config


def validate_authorization(config):
    approval = read(config['authorization']['path'], config['authorization']['raw_sha256'])
    if (approval.get('run_id') != OWNER or approval.get('gpu_count') != 8 or approval.get('replicas') != 1
            or approval.get('compute_release_authorized') is not False
            or approval.get('retain_existing_allocations') is not True
            or not approval.get('user_instruction_verbatim')):
        raise ValueError('explicit eight-H200 infrastructure authorization required')


def validate_devices(config):
    devices = read(config['devices']['path'], config['devices']['raw_sha256'])
    if (devices.get('allocation_name') != ALLOCATION or devices.get('run_id') != OWNER
            or devices.get('hostname') != config['hostname'] or devices.get('node_id') != config['node_id']
            or devices.get('replica_id') != config['replica_id']
            or devices.get('boot_id') != config['boot_id']
            or [row['uuid'] for row in devices.get('gpus', [])] != [row['gpu_uuid'] for row in config['workers']]
            or devices.get('identity_reconciliation', {}).get('replica_unchanged') is not True
            or devices.get('identity_reconciliation', {}).get('ssh_host_key_pinned') is not True):
        raise ValueError('fleet host/device binding differs from actual allocation readback')


def validate_dependencies(config):
    validate_config(config)
    validate_authorization(config)
    validate_devices(config)
    if sha(__file__) != config['launcher_sha256']:
        raise ValueError('fleet launcher source drifted')
    manifest = read(config['source_manifest'], config['source_manifest_raw_sha256'])
    if (manifest['source_sha256'] != SOURCE_SHA
            or hashlib.sha256(canonical(manifest['files']).rstrip(b'\n')).hexdigest() != SOURCE_SHA):
        raise ValueError('renderer frozen source manifest differs')
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe frozen source path')
        path = Path(config['source_root']) / relative
        if path.is_symlink() or sha(path) != row['sha256']:
            raise ValueError('renderer frozen source bytes differ')
    for ref in config['science_assets'] + config['runtime_source']:
        if sha(ref['path']) != ref['raw_sha256']:
            raise ValueError('renderer science/runtime dependency drifted')


def host_guard(config):
    validate_config(config)
    if (socket.gethostname() != config['hostname']
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != config['boot_id']
            or os.environ.get('ROBORENDER_BETA_RUN_ID') != OWNER
            or os.environ.get('ROBORENDER_BETA_ROUND_ID') != '1'):
        raise ValueError('fleet is not on the exact new eight-H200 allocation')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
        text=True, timeout=15).strip().splitlines()
    expected = [(i, row['gpu_uuid']) for i, row in enumerate(config['workers'])]
    actual = []
    for row in raw:
        index, gpu, name = (part.strip() for part in row.split(','))
        if 'H200' not in name:
            raise ValueError('non-H200 device in fleet allocation')
        actual.append((int(index), gpu))
    if actual != expected:
        raise ValueError('actual eight-device order/UUID differs from allocation readback')


def compute_clients():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True, timeout=15).strip()
    return raw.splitlines() if raw else []


def empty_workload_guard(config):
    if compute_clients():
        raise ValueError('unknown GPU compute clients; no takeover or automatic stop')
    probes = []
    try:
        for row in config['workers']:
            probe = socket.socket()
            probes.append(probe)
            probe.bind(('127.0.0.1', row['port']))
    finally:
        for probe in probes:
            probe.close()


def worker_root(config, slot):
    if type(slot) is not int or not 0 <= slot < 8:
        raise ValueError('invalid renderer slot')
    return Path(config['root']) / f'worker-{slot}'


def start(config, config_path):
    host_guard(config)
    validate_dependencies(config)
    root = Path(config['root'])
    with (root / 'start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'supervisor-launch.json').exists():
            receipt = read(root / 'supervisor-launch.json')
            if not receipt.get('identity') or identity(receipt['pid']) != receipt['identity']:
                raise ValueError('prior supervisor is not the same live owner; no automatic restart')
            return receipt
        if (root / 'start-intent.json').exists():
            raise ValueError('ambiguous fleet launch intent; reconcile without duplicate launch')
        empty_workload_guard(config)
        argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'supervise',
            '--config', str(config_path), '--config-sha256', sha(config_path)]
        immutable(root / 'start-intent.json', dict(argv=argv, config_sha256=sha(config_path),
            allocation_name=ALLOCATION, automatic_restart=False, compute_release_authorized=False))
        with (root / 'supervisor.log').open('xb') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
        receipt = dict(pid=child.pid, identity=identity(child.pid), config_sha256=sha(config_path),
            started_unix=time.time(), automatic_restart=False, compute_release_authorized=False)
        immutable(root / 'supervisor-launch.json', receipt)
        return receipt


def supervise(config, config_path):
    host_guard(config)
    validate_dependencies(config)
    root = Path(config['root'])
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        empty_workload_guard(config)
        immutable(root / 'supervisor-owner.json', dict(pid=os.getpid(), identity=identity(os.getpid()),
            config_sha256=sha(config_path), allocation_name=ALLOCATION, automatic_restart=False,
            compute_release_authorized=False))
        children = []
        for row in config['workers']:
            slot = row['slot']
            path = worker_root(config, slot)
            path.mkdir(exist_ok=False)
            argv = [str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), 'serve', '--slot', str(slot),
                '--config', str(config_path), '--config-sha256', sha(config_path)]
            immutable(path / 'launch-intent.json', dict(argv=argv, gpu_uuid=row['gpu_uuid'], port=row['port'],
                config_sha256=sha(config_path), automatic_restart=False, compute_release_authorized=False))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=row['gpu_uuid'],
                IMAGE3F_ROOT=str(REPO/'renderer/online_image3f/runtime'), IMAGE3F_QUEUE=str(path/'queue'),
                IMAGE3F_PORT=str(row['port']), ENDPOINT_FILE=str(path/'endpoint.txt'),
                ROBORENDER_MODEL_BASE=str(REPO/'renderer/models'), DIFFSYNTH_MODEL_BASE_PATH=str(REPO/'renderer/models'),
                DIFFSYNTH_SKIP_DOWNLOAD='true', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
            with (path / 'console.log').open('xb') as log:
                child = subprocess.Popen(argv, cwd=path, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            immutable(path / 'launch.json', dict(pid=child.pid, identity=identity(child.pid),
                slot=slot, gpu_uuid=row['gpu_uuid'], port=row['port'], started_unix=time.time(),
                config_sha256=sha(config_path), automatic_restart=False, compute_release_authorized=False))
        # CPU-only observation. No restart, signal, cleanup or allocation API.
        while True:
            rows = [dict(slot=i, pid=child.pid, exit_code=child.poll()) for i, child in enumerate(children)]
            value = dict(checked_unix=time.time(), children=rows,
                phase='LIVE' if all(row['exit_code'] is None for row in rows) else 'WORKER_EXITED',
                automatic_restart=False, compute_release_authorized=False)
            temporary = root / 'supervisor-status.tmp'
            with temporary.open('w') as stream:
                json.dump(value, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(root / 'supervisor-status.json')
            if all(row['exit_code'] is not None for row in rows):
                return value
            time.sleep(5)


def serve(config, config_path, slot):
    host_guard(config)
    validate_dependencies(config)
    path = worker_root(config, slot)
    row = config['workers'][slot]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != row['gpu_uuid']:
        raise ValueError('worker CUDA UUID differs from its exact slot')
    with (path / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clients = [line for line in compute_clients() if line.split(',')[0].strip() == row['gpu_uuid']]
        if clients:
            raise ValueError('assigned GPU already has a compute client; no takeover')
        owner = dict(slot=slot, pid=os.getpid(), identity=identity(os.getpid()), hostname=socket.gethostname(),
            allocation_name=ALLOCATION, allocation_owner=OWNER, source_root=config['source_root'],
            source_sha256=config['source_sha256'], gpu_uuid=row['gpu_uuid'], port=row['port'],
            config_sha256=sha(config_path), launcher_sha256=config['launcher_sha256'],
            started_unix=time.time(), transport_schema=CACHE_SCHEMA, automatic_restart=False,
            automatic_allocation_release=False, compute_release_authorized=False)
        immutable(path / 'owner.json', owner)
        source = Path(config['source_root']) / 'pipeline/dagger'
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('scaled_fleet_image3f_server', source/'image3f_server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()


def _exchange(stream, value):
    def receive(count):
        data = b''
        while len(data) < count:
            block = stream.recv(count-len(data))
            if not block:
                raise ConnectionError('renderer metadata EOF')
            data += block
        return data
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.sendall(struct.pack('!Q', len(raw)) + raw)
    count = struct.unpack('!Q', receive(8))[0]
    if not 0 < count < 65536:
        raise ValueError('renderer metadata response bound')
    return pickle.loads(receive(count))


def attest_worker(config, slot):
    path = worker_root(config, slot)
    owner = read(path / 'owner.json')
    launch = read(path / 'launch.json')
    worker = config['workers'][slot]
    if (not owner.get('identity') or owner['pid'] != launch['pid'] or owner['identity'] != launch['identity']
            or identity(owner['pid']) != owner['identity']
            or owner['source_sha256'] != config['source_sha256'] or owner['gpu_uuid'] != worker['gpu_uuid']
            or owner['port'] != worker['port'] or owner['allocation_name'] != ALLOCATION
            or owner.get('allocation_owner') != OWNER or owner.get('hostname') != config['hostname']
            or owner.get('slot') != slot or owner.get('launcher_sha256') != config['launcher_sha256']
            or owner.get('config_sha256') != sha(Path(config['root'])/'deployment.json')
            or launch.get('config_sha256') != owner['config_sha256']):
        raise ValueError('renderer live owner differs from immutable launch')
    with socket.create_connection(('127.0.0.1', worker['port']), timeout=10) as stream:
        if _exchange(stream, {'cmd': 'ping'}) != {'ok': True}:
            raise ValueError('renderer metadata ping failed')
        channel = uuid.uuid4().hex
        opened = _exchange(stream, dict(cmd='transport_open', channel=channel))
        transport = opened.get('transport', {})
        if (opened.get('ok') is not True or set(transport) != {'schema', 'server_instance', 'channel'}
                or transport['schema'] != CACHE_SCHEMA or transport['channel'] != channel
                or re.fullmatch('[0-9a-f]{32}', transport['server_instance']) is None):
            raise ValueError('cached renderer channel handshake differs')
        observed = _exchange(stream, {'cmd': 'observation_contract'})
    attestation = observed.get('attestation', {})
    expected = dict(schema='roborender.dagger.student-observation.v1', mode='generated_only',
        renderer='Image3F', renderer_steps=5, renderer_cfg=1., teacache_threshold=.3,
        raytraced_policy_fallback=False)
    if (observed.get('ok') is not True or attestation.get('contract') != expected
            or attestation.get('backend_class') != 'runtime.image3f_deploy.Image3FSession'
            or attestation.get('generate_source_sha256') != config['generate_source_sha256']
            or identity(owner['pid']) != owner['identity']):
        raise ValueError('actual renderer runtime source/observation contract differs')
    return dict(schema='roborender.scaled-renderer-worker-readiness.v1', checked_unix=time.time(),
        server=owner, endpoint=dict(host='127.0.0.1', port=worker['port']),
        renderer_identity=config['renderer_identity'], transport=transport, observation_attestation=attestation,
        render_requests=0, optimizer_steps=0, native_steps=0, inference_qualified=False,
        allocation_retained=True, compute_release_authorized=False)


def attest(config):
    host_guard(config)
    validate_dependencies(config)
    root = Path(config['root'])
    if (root / 'endpoint-pool.json').exists():
        previous = read(root / 'endpoint-pool.json')
        for worker in previous['workers']:
            if not worker.get('identity') or identity(worker['pid']) != worker['identity']:
                raise ValueError('previously attested fleet owner changed; no automatic re-admission')
        # A prior receipt is identity evidence, not a fresh endpoint health check.
        for slot in range(8):
            current = attest_worker(config, slot)
            if current['transport']['server_instance'] != previous['workers'][slot]['server_instance']:
                raise ValueError('renderer cached server instance changed')
        return previous
    observations = [attest_worker(config, slot) for slot in range(8)]
    if len({row['transport']['server_instance'] for row in observations}) != 8:
        raise ValueError('fleet endpoints alias a renderer process')
    workers, endpoints = [], []
    for slot, value in enumerate(observations):
        path = worker_root(config, slot) / 'readiness.json'
        if path.exists():
            previous = read(path)
            if (previous['server'] != value['server'] or previous['endpoint'] != value['endpoint']
                    or previous['renderer_identity'] != value['renderer_identity']
                    or previous['observation_attestation'] != value['observation_attestation']
                    or previous['transport']['server_instance'] != value['transport']['server_instance']):
                raise ValueError('persisted worker readiness differs from live process')
            value = previous
        else:
            immutable(path, value)
        owner = value['server']
        binding = dict(slot=slot, host='127.0.0.1', port=owner['port'], gpu_uuid=owner['gpu_uuid'],
            server_instance=value['transport']['server_instance'], source_sha256=config['source_sha256'],
            generate_source_sha256=config['generate_source_sha256'], readiness_receipt_sha256=sha(path))
        endpoints.append(binding)
        workers.append(dict(**binding, pid=owner['pid'], identity=owner['identity'],
            readiness_receipt_path=str(path)))
    result = dict(schema=SCHEMA, allocation_name=ALLOCATION, allocation_owner=OWNER,
        hostname=config['hostname'], node_id=config['node_id'], replica_id=config['replica_id'],
        boot_id=config['boot_id'], ssh_config=config['ssh_config'], ssh_alias=ALLOCATION,
        server_source_sha256=config['source_sha256'], workers=workers,
        renderer=dict(identity=config['renderer_identity'], endpoints=endpoints,
            routing='lane_mod_endpoint_count', max_inflight_per_endpoint=1),
        inference_qualified=False, throughput_qualified=False, automatic_restart=False,
        automatic_allocation_release=False, compute_release_authorized=False)
    immutable(root / 'endpoint-pool.json', result)
    return result


def stage(template_path, devices_path, authorization_path, reference_path, root, ssh_config):
    template, devices = read(template_path), read(devices_path)
    reference = read(reference_path)
    if (devices.get('allocation_name') != ALLOCATION or devices.get('run_id') != OWNER
            or len(devices.get('gpus', [])) != 8 or reference['server']['source_sha256'] != SOURCE_SHA
            or template['source_sha256'] != SOURCE_SHA):
        raise ValueError('actual eight-device receipt and unchanged admitted renderer release required')
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('renderer fleet staging namespace must be fresh')
    config = {key: template[key] for key in ('source_root', 'source_sha256', 'source_manifest',
        'source_manifest_raw_sha256', 'science_assets', 'runtime_source', 'renderer_identity')}
    config.update(schema=SCHEMA, root=str(root), allocation_name=ALLOCATION, allocation_owner=OWNER,
        hostname=devices['hostname'], node_id=devices['node_id'], replica_id=devices['replica_id'],
        boot_id=devices['boot_id'], ssh_config=str(Path(ssh_config).resolve()),
        authorization=dict(path=str(Path(authorization_path).resolve()), raw_sha256=sha(authorization_path)),
        devices=dict(path=str(Path(devices_path).resolve()), raw_sha256=sha(devices_path)),
        generate_source_sha256=reference['metadata_probe']['observation_attestation']['generate_source_sha256'],
        launcher_sha256=sha(__file__), automatic_restart=False, automatic_allocation_release=False,
        compute_release_authorized=False,
        workers=[dict(slot=i, gpu_uuid=row['uuid'], port=18560+i) for i, row in enumerate(devices['gpus'])])
    validate_config(config)
    validate_authorization(config)
    validate_devices(config)
    root.mkdir(parents=True)
    # Immutable executable artifact for remote use, not a mutable live checkout.
    launcher = root / 'fleet-launcher.py'
    with launcher.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    launcher.chmod(0o444)
    immutable(root / 'deployment.json', config)
    return dict(config=str(root/'deployment.json'), config_sha256=sha(root/'deployment.json'),
        launcher=str(launcher), launcher_sha256=sha(launcher), gpu_work_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'start', 'supervise', 'serve', 'attest'])
    parser.add_argument('--config')
    parser.add_argument('--config-sha256')
    parser.add_argument('--slot', type=int)
    parser.add_argument('--template')
    parser.add_argument('--devices')
    parser.add_argument('--authorization')
    parser.add_argument('--reference-endpoint')
    parser.add_argument('--root')
    parser.add_argument('--ssh-config')
    args = parser.parse_args()
    if args.action == 'stage':
        result = stage(args.template, args.devices, args.authorization, args.reference_endpoint,
            args.root, args.ssh_config)
    else:
        if not args.config_sha256 or not _sha(args.config_sha256):
            raise ValueError('exact fleet config byte hash required')
        config = read(args.config, args.config_sha256)
        validate_config(config)
        result = (start(config, args.config) if args.action == 'start' else
            supervise(config, args.config) if args.action == 'supervise' else
            serve(config, args.config, args.slot) if args.action == 'serve' else attest(config))
    if result is not None:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
