"""Camera stalls are terminal observations; no native mutation or replay."""
from copy import deepcopy
import pytest
from resident_isaac_client import NativeCaptureProgressGuard


def progress():
    return dict(active_command_id='command-1', phase='RUNNING', worker_instance_id='owner-1',
                native_progress=dict(phase='capture_teacher', event='start', run_id='run-1',
                                     round_id=1, vector_step=291, sequence=2587))


def test_responsive_endpoint_cannot_hide_unchanged_native_capture():
    guard = NativeCaptureProgressGuard('command-1')
    row = progress()
    guard.observe(row, 0)
    guard.observe(deepcopy(row), 899.9)
    with pytest.raises(RuntimeError, match='capture progress stalled.*replay forbidden'):
        guard.observe(row, 900)


@pytest.mark.parametrize('field,value', [('vector_step', 292), ('sequence', 2588)])
def test_actual_phase_progress_restarts_observation_budget(field, value):
    guard = NativeCaptureProgressGuard('command-1')
    row = progress(); guard.observe(row, 0)
    row['native_progress'][field] = value
    guard.observe(row, 899); guard.observe(row, 1700)
    with pytest.raises(RuntimeError): guard.observe(row, 1799)


@pytest.mark.parametrize('change', ['old_command', 'idle', 'checkpoint', 'completed_capture', 'no_progress'])
def test_other_phases_are_not_camera_stalls(change):
    guard = NativeCaptureProgressGuard('command-1')
    row = progress(); guard.observe(row, 0)
    if change == 'old_command': row['active_command_id'] = 'prior-command'
    elif change == 'idle': row['phase'] = 'IDLE'
    elif change == 'checkpoint': row['native_progress']['phase'] = 'seal_ack_barrier'
    elif change == 'completed_capture': row['native_progress']['event'] = 'end'
    else: row.pop('native_progress')
    guard.observe(row, 10000)
    assert guard.identity is None and guard.since is None
    guard.observe(progress(), 10001)
    guard.observe(progress(), 10900)


def test_remote_clock_cannot_extend_deadline():
    guard = NativeCaptureProgressGuard('command-1')
    row = progress(); row['native_progress']['updated_at_utc'] = 'future'
    guard.observe(row, 20)
    row['native_progress']['updated_at_utc'] = 'even-further-future'
    with pytest.raises(RuntimeError): guard.observe(row, 920)


def test_budget_must_be_positive():
    with pytest.raises(ValueError): NativeCaptureProgressGuard('command-1', seconds=0)
