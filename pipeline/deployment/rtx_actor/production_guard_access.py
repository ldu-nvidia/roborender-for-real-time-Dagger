"""Read one nonsecret guard from an attested native container, without stepping.

Older pinned native releases wrote root-owned 0600 resource receipts. This
control-only compatibility path never changes native source or reads tokens,
checkpoints, arbitrary files, or an unattested container.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


def guard_path(path, worker):
    root = Path(worker['runtime_root']).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute owned guard path required')
    if path.is_relative_to('/dagger-runtime'):
        path = root / path.relative_to('/dagger-runtime')
    relative = path.relative_to(root)
    if (len(relative.parts) != 3 or relative.parts[0] != 'phases'
            or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,127}', relative.parts[1])
            or relative.name != 'resource-guard.json'
            or path.resolve(strict=True) != path):
        raise ValueError('only an exact owned native resource guard is readable')
    return path, Path('/dagger-runtime') / relative


def attest_container(worker, *, run=subprocess.run):
    name = worker['container_name']
    if not re.fullmatch('standing-isaac-leo-prod-[a-z0-9-]+-g[01]', name):
        raise ValueError('explicit production container identity required')
    result = run(['docker', 'inspect', name], check=True, capture_output=True, timeout=15)
    rows = json.loads(result.stdout)
    if len(rows) != 1:
        raise RuntimeError('ambiguous production container')
    row = rows[0]
    mounts = [m for m in row.get('Mounts', []) if m.get('Destination') == '/dagger-runtime']
    requests = row.get('HostConfig', {}).get('DeviceRequests', [])
    devices = [d for request in requests for d in (request.get('DeviceIDs') or [])]
    if (row.get('Name') != '/' + name or row.get('State', {}).get('Running') is not True
            or row.get('Config', {}).get('Labels', {}).get('org.roborender.qualification') != name
            or devices != [worker['gpu_uuid']] or len(mounts) != 1
            or mounts[0].get('Type') != 'bind' or mounts[0].get('RW') is not True
            or Path(mounts[0]['Source']).resolve(strict=True) != Path(worker['runtime_root']).resolve(strict=True)):
        raise RuntimeError('native container/GPU/owned runtime binding differs')
    return row['Id']


def read_owned_guard(path, worker, expected_sha256, *, repair=False, run=subprocess.run):
    path, container_path = guard_path(path, worker)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('exact native guard digest required')
    container_id = attest_container(worker, run=run)
    code = '''import hashlib,json,os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
assert p.resolve(strict=True)==p
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
with os.fdopen(fd,"rb") as stream:
 info=os.fstat(stream.fileno())
 assert stat.S_ISREG(info.st_mode) and info.st_size<=4<<20
 raw=stream.read((4<<20)+1)
 assert len(raw)<=4<<20 and hashlib.sha256(raw).hexdigest()==expected
 value=json.loads(raw)
 assert value.get("schema")=="roborender.native-phase-resource-guard.v1"
 if sys.argv[3]=="repair": os.fchmod(stream.fileno(),0o644)
sys.stdout.buffer.write(raw)
'''
    result = run(['docker', 'exec', '--user', '0', container_id, 'python3', '-c', code,
                  str(container_path), expected_sha256, 'repair' if repair else 'read'],
                 check=True, capture_output=True, timeout=15)
    raw = result.stdout
    if len(raw) > 4 << 20 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError('container returned a different resource guard')
    if json.loads(raw).get('schema') != 'roborender.native-phase-resource-guard.v1':
        raise RuntimeError('native resource guard schema differs')
    return raw
