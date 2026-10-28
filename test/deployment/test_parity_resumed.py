"""Renewal changes time authority only, never native/checkpoint semantics."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'runtime/raven-parity-overnight-20261026-t6oMpN'
spec = importlib.util.spec_from_file_location('resumed_observer_test', ROOT / 'watchdog_resumed.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def setup(root):
    for name in ('user-authorization.json', 'user-authorization-resumed.json'):
        (root / name).write_bytes((ROOT / name).read_bytes())
    value = dict(schema='roborender.parity-overnight-progress.v1', compute_release_authorized=False,
        deadline_unix=None, resumed_authorization=m.auth.authority(root=root), updated_unix=100,
        phase='TEST', complete=False, faults=[], accepted_operations=[])
    m.m.base.save(root / 'progress.json', value)
    return value

def test_removed_horizon_does_not_remove_stall_detection(tmp_path):
    setup(tmp_path)
    assert m.observe_progress(tmp_path, 110)['health'] == 'HEALTHY'
    value = m.observe_progress(tmp_path, 10**12)
    assert value['deadline_reached'] is False
    assert value['faults'] == ['no_recorded_progress_for20minutes']

@pytest.mark.parametrize('key,value', [('compute_release_authorized', True),
    ('native_unknown_replay_authorized', True), ('retired_artifact_restore_authorized', True),
    ('production_promotion_authorized_by_this_record', True), ('deadline_unix', 10**12),
    ('user_request', 'invented approval')])
def test_scope_cannot_expand(tmp_path, key, value):
    setup(tmp_path)
    path = tmp_path / 'user-authorization-resumed.json'
    data = json.loads(path.read_bytes()); data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m.auth.authority(new_work=True, root=tmp_path)

def test_operation_faults_remain_faults(tmp_path):
    value = setup(tmp_path)
    status = tmp_path / 'operation.json'; m.m.base.save(status, {'phase': 'CRASH_UNCERTAIN'})
    value['accepted_operations'] = [{'status_file': str(status)}]
    m.m.base.save(tmp_path / 'progress.json', value)
    assert m.observe_progress(tmp_path, 110)['faults'] == ['operation_fault:' + str(status)]

def test_same_lock_native_pair_and_deduplicated_new_prompt(tmp_path):
    assert m.m.base.ROOT == ROOT and m.m.base.INTERVAL == 600
    assert m.m.probe.BINDINGS == ROOT / 'native-v6/warm-bindings.json'
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='Queued message test-double')
    value = dict(phase='TEST', health='HEALTHY', faults=[], complete=False, deadline_reached=False)
    assert m.deliver(value, tmp_path, runner, 100)['delivery'] == 'QUEUED'
    assert 'eight-hour' not in calls[0][-1]
    assert 'user-authorization-resumed.json' in calls[0][-1]
    assert calls[0][-1].count('watchdog_resumed.py') == 2
    m.deliver(value, tmp_path, runner, 200)
    assert len(calls) == 1
