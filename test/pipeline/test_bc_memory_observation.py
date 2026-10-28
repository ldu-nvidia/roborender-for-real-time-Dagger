"""CPU-only failure journal and diagnostic recovery policy; never launch Isaac."""
from copy import deepcopy
import json

import pytest

from pipeline.bc_collection import run_continuous as launcher
from pipeline.deployment.rtx_actor.bc_memory_guard import runtime_guard

GIB = 2**30


def snapshot():
    return dict(pid=123, memory_current=38 * GIB, memory_peak=43 * GIB, oom_killed=False,
                **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0))


def test_limit_event_is_not_oom_and_strict_policy_remains_default():
    value = snapshot()
    value['events']['max'] = 1
    with pytest.raises(RuntimeError, match='memory.max limit event'):
        runtime_guard({'MemAvailable': 16 * 1024**2}, value)
    warning = runtime_guard({'MemAvailable': 16 * 1024**2}, value,
                            recovered_limit_cap_bytes=43 * GIB)
    assert warning == dict(status='WARNING', reason='recovered_memory_max_event',
                           max_events=1, capacity_certified=False)


@pytest.mark.parametrize('mutation,reason', [
    ('swap', 'swap usage'), ('oom', 'OOM allocation'), ('oom_kill', 'OOM kill'),
    ('docker_oom', 'OOM kill'), ('host', '8GiB floor'), ('headroom', '2GiB'),
    ('peak', 'peak exceeded'), ('missing', 'counters'), ('negative', 'counters'),
    ('boolean', 'counters'), ('missing_current', 'current/peak'), ('cap', 'current/peak'),
])
def test_recovered_limit_mode_keeps_hard_safety_checks(mutation, reason):
    value, host, cap = snapshot(), {'MemAvailable': 16 * 1024**2}, 43 * GIB
    value['events']['max'] = 1
    if mutation == 'swap': value['memory_swap.current'] = 1
    if mutation in ('oom', 'oom_kill'): value['events'][mutation] = 1
    if mutation == 'docker_oom': value['oom_killed'] = True
    if mutation == 'host': host['MemAvailable'] = 8 * 1024**2 - 1
    if mutation == 'headroom': value['memory_current'] = 41 * GIB + 1
    if mutation == 'peak': value['memory_peak'] = cap + 1
    if mutation == 'missing': value['events'].pop('oom')
    if mutation == 'negative': value['events']['max'] = -1
    if mutation == 'boolean': value['events']['max'] = True
    if mutation == 'missing_current': value.pop('memory_current')
    if mutation == 'cap': cap = 80 * GIB
    with pytest.raises(RuntimeError, match=reason):
        runtime_guard(host, value, recovered_limit_cap_bytes=cap)


@pytest.mark.parametrize('fault', ['memory', 'gpu', 'owner', 'interrupt'])
def test_rejecting_sample_is_fsynced_before_exception_reaches_cleanup(tmp_path, monkeypatch, fault):
    path = tmp_path / 'memory-samples.jsonl'
    value = snapshot()
    if fault == 'memory': value['events']['oom'] = 1
    synced = []
    monkeypatch.setattr(launcher.os, 'fsync', lambda fd: synced.append(json.loads(path.read_text())))
    def gpu():
        if fault == 'gpu': raise RuntimeError('GPU observation unavailable')
        return dict(uuid='GPU-test', used_bytes=55 * GIB, total_bytes=96 * GIB)
    def owner():
        if fault == 'owner': raise RuntimeError('owner differs')
        if fault == 'interrupt': raise KeyboardInterrupt('stop requested')
    with path.open('x') as stream:
        with pytest.raises((RuntimeError, KeyboardInterrupt)):
            launcher.observe_resources(stream, host={'MemAvailable': 16 * 1024**2},
                sample=value, guard=runtime_guard, gpu_reader=gpu, owner_check=owner,
                recovered_limit_cap_bytes=43 * GIB)
    assert len(synced) == 1
    row = json.loads(path.read_text())
    assert row['container'] == value and row['observation_failure']['message']
    assert row['host_available_kib'] == 16 * 1024**2


def test_recovered_warning_is_persisted_with_actual_gpu_and_memory(tmp_path):
    path, value = tmp_path / 'memory-samples.jsonl', snapshot()
    value['events']['max'] = 2
    with path.open('x') as stream:
        row = launcher.observe_resources(stream, host={'MemAvailable': 16 * 1024**2},
            sample=value, guard=runtime_guard, gpu_reader=lambda: {'uuid': 'GPU-test'},
            owner_check=lambda: None, recovered_limit_cap_bytes=43 * GIB)
    assert json.loads(path.read_text()) == row
    assert row['memory_guard']['status'] == 'WARNING'
    assert 'observation_failure' not in row


def test_limit_warning_cannot_be_promoted_to_capacity():
    from test.pipeline.test_continuous_bc_launcher import fixture_capacity
    values = deepcopy(fixture_capacity())
    values['memory_samples'][0]['container']['events']['max'] = 1
    with pytest.raises(ValueError, match='resource/ownership'):
        launcher.build_capacity(**values)


def test_launcher_wires_journal_before_cleanup_and_keeps_pinned_strict_guard():
    import ast
    import inspect
    source = inspect.getsource(launcher.main)
    assert source.index('observe_resources(') < source.index('cleanup_exact(')
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    observations = [node for node in calls if ast.unparse(node.func) == 'observe_resources']
    assert len(observations) == 1
    # The active launcher no longer opts into the historical recovered-max
    # relaxation. It uses a pinned observer and explicit live headroom checks.
    assert 'recovered_limit_cap_bytes' not in {k.arg for k in observations[0].keywords}
    pinned = [node for node in calls if ast.unparse(node.func) == 'PinnedCgroupObserver']
    assert len(pinned) == 1
    assert ast.unparse(pinned[0].args[-1]) == 'args.memory_gib * GIB'
    assert "request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL" in source
    assert "runtime_guard(memory_kib(), value)" in source
    assert "value['memory_current'] + 2 * GIB > args.memory_gib * GIB" in source
    assert "memory_kib()['MemAvailable'] < 8 * 1024**2" in source


def test_startup_memory_floor_is_journaled(tmp_path):
    path = tmp_path / 'memory-samples.jsonl'
    with path.open('x') as stream:
        with pytest.raises(RuntimeError, match='floor'):
            launcher.observe_resources(stream, host={'MemAvailable': 1}, sample={},
                guard=runtime_guard, gpu_reader=lambda: {}, owner_check=lambda: None, startup=True)
    assert json.loads(path.read_text())['observation_failure']['message'].endswith('startup')
