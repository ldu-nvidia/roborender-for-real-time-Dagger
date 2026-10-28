"""CPU checks for the exact nonsecret root-container metadata compatibility read."""
from test.deployment.source_paths import PIPELINE_ROOT
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(PIPELINE_ROOT / 'deployment/rtx_actor'))
from production_guard_access import guard_path, read_owned_guard
from recover_production_resident_pool import require_idle


@pytest.fixture
def owned(tmp_path):
    runtime = tmp_path / 'gpu0'
    guard = runtime / 'phases/production-startup-g0-b2/resource-guard.json'
    guard.parent.mkdir(parents=True)
    raw = b'{"schema":"roborender.native-phase-resource-guard.v1"}\n'
    guard.write_bytes(raw)
    worker = dict(runtime_root=str(runtime), container_name='standing-isaac-leo-prod-test-g0', gpu_uuid='GPU-owned')
    inspection = dict(Id='container-exact-id', Name='/' + worker['container_name'], State=dict(Running=True),
        Config=dict(Labels={'org.roborender.qualification': worker['container_name']}),
        HostConfig=dict(DeviceRequests=[dict(DeviceIDs=['GPU-owned'])]),
        Mounts=[dict(Type='bind', Source=str(runtime), Destination='/dagger-runtime', RW=True)])
    return guard, raw, worker, inspection


def test_exact_owned_guard_read_and_repair_are_digest_bound(owned):
    guard, raw, worker, inspection = owned
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps([inspection]).encode() if argv[1] == 'inspect' else raw)
    assert read_owned_guard(guard, worker, hashlib.sha256(raw).hexdigest(), repair=True, run=run) == raw
    assert calls[1][:5] == ['docker', 'exec', '--user', '0', 'container-exact-id']
    assert calls[1][-3] == '/dagger-runtime/phases/production-startup-g0-b2/resource-guard.json'
    assert calls[1][-1] == 'repair'


@pytest.mark.parametrize('fault', ['stopped', 'label', 'mount', 'gpu', 'name'])
def test_different_owner_never_reads_or_repairs(owned, tmp_path, fault):
    guard, raw, worker, inspection = owned
    if fault == 'stopped': inspection['State']['Running'] = False
    elif fault == 'label': inspection['Config']['Labels'] = {}
    elif fault == 'mount': inspection['Mounts'][0]['Source'] = str(tmp_path)
    elif fault == 'gpu': inspection['HostConfig']['DeviceRequests'][0]['DeviceIDs'] = ['GPU-other']
    else: inspection['Name'] = '/other'
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps([inspection]).encode())
    with pytest.raises(RuntimeError):
        read_owned_guard(guard, worker, hashlib.sha256(raw).hexdigest(), repair=True, run=run)
    assert len(calls) == 1


@pytest.mark.parametrize('relative', ['resident.token', 'phases/name/checkpoint.pt',
    'phases/name/deeper/resource-guard.json', 'phases/name/../resource-guard.json'])
def test_guard_reader_cannot_read_tokens_checkpoints_or_arbitrary_metadata(owned, relative):
    guard, raw, worker, inspection = owned
    with pytest.raises((ValueError, FileNotFoundError)):
        guard_path(Path(worker['runtime_root']) / relative, worker)


def test_unexpected_bytes_cannot_be_used_as_guard_receipt(owned):
    guard, raw, worker, inspection = owned
    def run(argv, **kwargs):
        return SimpleNamespace(stdout=json.dumps([inspection]).encode() if argv[1] == 'inspect' else b'changed')
    with pytest.raises(RuntimeError):
        read_owned_guard(guard, worker, hashlib.sha256(raw).hexdigest(), run=run)


@pytest.mark.parametrize('phase,active,event', [
    ('RUNNING', 'startup', {'phase': 'worker_ready'}),
    ('IDLE', None, {'phase': 'phase_end', 'command_id': 'other-workload'}),
    ('CRASH_UNCERTAIN', None, {'phase': 'worker_ready'})])
def test_recovery_rejects_active_changed_or_uncertain_native_owner(phase, active, event):
    with pytest.raises(RuntimeError):
        require_idle(dict(phase=phase, active_command_id=active, current_event=event), 'startup')
