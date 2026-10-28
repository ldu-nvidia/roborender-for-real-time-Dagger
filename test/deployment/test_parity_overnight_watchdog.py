"""Scoped observer tests; explicitly bound runtime helper, never native actions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT = Path('/workspace/project/roborender/runtime/raven-parity-overnight-20261026-t6oMpN/watchdog.py')

@pytest.fixture
def observer(tmp_path):
    if not SCRIPT.is_file():
        pytest.skip('dated operational helper is not part of the source release')
    spec = importlib.util.spec_from_file_location('scoped_parity_watchdog', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save(tmp_path / 'progress.json', dict(schema='roborender.parity-overnight-progress.v1',
        phase='CODE_FIXES', updated_unix=100, deadline_unix=30000, complete=False,
        compute_release_authorized=False, faults=[], accepted_operations=[], evidence=[]))
    return module, tmp_path

def test_fresh_then_stale_and_deadline(observer):
    m, root = observer
    assert m.observe(root, 200)['health'] == 'HEALTHY'
    assert m.observe(root, 1400)['faults'] == ['no_recorded_progress_for20minutes']
    assert m.observe(root, 30000)['deadline_reached']
    assert m.observe(root, 200)['controls_workloads'] is False

def test_one_notice_when_healthy_unchanged(observer):
    m, root = observer
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='Queued message id=1')
    value = m.observe(root, 200)
    first = m.deliver(value, root, runner, now=200)
    assert m.deliver(value, root, runner, now=5000) == first
    assert len(calls) == 1
    assert calls[0][1] == 'queue'
    assert 'Never release' in calls[0][-1] or 'never release' in calls[0][-1]

def test_ambiguous_delivery_never_replayed(observer):
    m, root = observer
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        raise TimeoutError('unknown')
    value = m.observe(root, 1400)
    first = m.deliver(value, root, runner, now=1400)
    assert first['delivery'] == 'UNCERTAIN'
    assert m.deliver(value, root, runner, now=5000, force=True) == first
    assert len(calls) == 1

def test_missing_receipt_and_native_terminal_fault(observer):
    m, root = observer
    progress = m.read(root / 'progress.json')
    progress['evidence'] = [str(root / 'absent.json')]
    progress['accepted_operations'] = [dict(status_file=str(root / 'native.json'))]
    m.save(root / 'native.json', dict(phase='CRASH_UNCERTAIN'))
    m.save(root / 'progress.json', progress)
    value = m.observe(root, 200)
    assert len(value['faults']) == 2
    assert value['controls_workloads'] is False

def test_reject_release_or_symlink_input(observer):
    m, root = observer
    progress = m.read(root / 'progress.json')
    progress['compute_release_authorized'] = True
    m.save(root / 'progress.json', progress)
    with pytest.raises(ValueError, match='authority'):
        m.observe(root, 200)
    (root / 'link.json').symlink_to(root / 'progress.json')
    with pytest.raises(ValueError, match='regular'):
        m.read(root / 'link.json')
