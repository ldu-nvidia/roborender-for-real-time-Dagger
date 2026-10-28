from copy import deepcopy
from types import SimpleNamespace

import pytest

from bc_capacity_guard import GIB, CapacityRejected, guarded_save, projected_save_gate, plateau_gate
from run_guarded_bc_capacity import augment_command, admission, runtime_guard
import bc_memory_guard
import run_guarded_bc_capacity


def sample():
    return {'memory_current': 25 * GIB, 'memory_max': 43 * GIB, 'memory_peak': 29 * GIB,
        'memory_swap.current': 0, 'events': {'max': 0, 'oom': 0, 'oom_kill': 0},
        'stat': {'anon': 20 * GIB, 'file': 4 * GIB, 'shmem': GIB, 'kernel': GIB // 4}}


def trainer(b=6):
    depth = SimpleNamespace(shape=(b, 480, 832, 1))
    robot = SimpleNamespace(shape=(b, 480, 832, 3))
    views = {name: {'depth': depth, 'robot_rgb': robot, 'rgb': None}
             for name in ('head_color', 'hand_left', 'hand_right')}
    return SimpleNamespace(env=SimpleNamespace(num_envs=b), _depth_chunk=[depth] * 81,
        _robot_rgb_chunk=[robot] * 81, _views_chunk=[views] * 81, vg_mode='collect',
        vg_rollout_action_source='teacher', vg_save_raw_depth=True, vg_save_raw_depth_native=True,
        vg_save_sim_rgb=False, vg_control_res=[240, 416],
        _bc_ledger=SimpleNamespace(vector_steps=162, emitted_counts=[2] * b))


def test_guard_calls_original_once_and_keeps_all_buffers():
    t, calls, reports = trainer(), [], []
    original_lists = (t._depth_chunk, t._robot_rgb_chunk, t._views_chunk)
    result = guarded_save(t, lambda: calls.append(1) or 'exact', snapshot=sample,
        host_available=lambda: 20 * GIB, limit_bytes=43 * GIB,
        report=lambda phase, value: reports.append({'phase': phase, **value}))
    assert result == 'exact' and calls == [1]
    assert original_lists == (t._depth_chunk, t._robot_rgb_chunk, t._views_chunk)
    assert reports[-1]['completed_original_save'] is True
    assert reports[-1]['emitted_chunks_per_lane'] == [2] * 6


def test_guard_preserves_native_quota_exception():
    class _QuotaReached(Exception):
        pass
    reports = []
    def original():
        raise _QuotaReached()
    with pytest.raises(_QuotaReached):
        guarded_save(trainer(), original, snapshot=sample, host_available=lambda: 20 * GIB,
            limit_bytes=43 * GIB, report=lambda p, v: reports.append(v))
    assert reports[-1]['completed_original_save'] is True


@pytest.mark.parametrize('field,value', [('memory_current', 34 * GIB), ('memory_swap.current', 1),
    ('memory_max', 40 * GIB), ('events', {'max': 1, 'oom': 0, 'oom_kill': 0})])
def test_guard_rejects_before_original_allocations(field, value):
    s, calls = sample(), []
    s[field] = value
    with pytest.raises(CapacityRejected):
        guarded_save(trainer(), lambda: calls.append(1), snapshot=lambda: s,
            host_available=lambda: 20 * GIB, limit_bytes=43 * GIB, report=lambda *x: None)
    assert calls == []


def test_low_host_and_changed_buffers_rejected():
    calls = []
    with pytest.raises(CapacityRejected):
        guarded_save(trainer(), lambda: calls.append(1), snapshot=sample,
            host_available=lambda: 15 * GIB, limit_bytes=43 * GIB, report=lambda *x: None)
    t = trainer()
    t._views_chunk = []
    with pytest.raises(CapacityRejected):
        guarded_save(t, lambda: calls.append(1), snapshot=sample,
            host_available=lambda: 20 * GIB, limit_bytes=43 * GIB, report=lambda *x: None)
    assert calls == []


def flush(current=25, anon=20, count=2):
    s = sample()
    return {'phase': 'postsave', 'completed_original_save': True, 'memory': s,
        'vector_steps': count * 81, 'emitted_chunks_per_lane': [count] * 6,
        'memory_current_bytes': int(current * GIB), 'memory_peak_bytes': 29 * GIB,
        'anon_bytes': int(anon * GIB), 'shmem_bytes': GIB, 'kernel_bytes': GIB // 4,
        'file_bytes': 4 * GIB, 'host_available_bytes': 20 * GIB}


def test_predeclared_plateau_and_full_emission_gate():
    assert plateau_gate([flush(count=1), flush(count=2)], cap_bytes=43 * GIB)['runtime_guard_enabled']
    with pytest.raises(CapacityRejected, match='two complete'):
        plateau_gate([flush(count=1)], cap_bytes=43 * GIB)
    with pytest.raises(CapacityRejected, match='plateau'):
        plateau_gate([flush(count=1), flush(anon=24, count=2)], cap_bytes=43 * GIB)


def test_full_buffer_flush_gate_allows_staggered_episode_emission_counts():
    rows = [flush(count=i) for i in (1, 2, 3)]
    for row, counts in zip(rows, ([0, 1] * 3, [1, 2] * 3, [2, 2] * 3)):
        row['emitted_chunks_per_lane'] = counts
    assert plateau_gate(rows, cap_bytes=43 * GIB)['bounded_repeated_buffer_only']
    rows[-1]['emitted_chunks_per_lane'][1] = 1
    with pytest.raises(CapacityRejected):
        plateau_gate(rows, cap_bytes=43 * GIB)


@pytest.mark.parametrize('fault', ['decreasing', 'same_step'])
def test_repeated_writer_gate_requires_real_advancing_windows_and_monotonic_counts(fault):
    rows = [flush(count=1), flush(count=2), flush(count=3)]
    if fault == 'decreasing':
        rows[1]['emitted_chunks_per_lane'][0] = 0
    else:
        rows[1]['vector_steps'] = rows[0]['vector_steps']
    with pytest.raises(CapacityRejected):
        plateau_gate(rows, cap_bytes=43 * GIB)


def test_cap_override_is_only_exact_qualification_command():
    base = ['docker', 'run', '--memory', '40g', '--memory-swap', '40g', '--env',
        'PYTHONPATH=/bc-production-source', 'sha256:' + 'a' * 64,
        'trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer']
    value = augment_command(base, guard_path='/guard.py', guard_sha256='b' * 64,
                            memory_gib=43, image='sha256:' + 'a' * 64)
    assert value[value.index('--memory') + 1] == '43g'
    assert value[value.index('--memory-swap') + 1] == '43g'
    assert 'trainer._target_=bc_capacity_guard.GuardedBCSessionTrainer' in value
    assert base[base.index('--memory') + 1] == '40g'
    with pytest.raises(ValueError):
        augment_command(base, guard_path='/guard.py', guard_sha256='b' * 64, memory_gib=44, image=base[-2])


def test_host_admission_and_owned_runtime_safety():
    admission(51 * 1024**2, 43)
    with pytest.raises(RuntimeError):
        admission(51 * 1024**2 - 1, 43)
    s = {'pid': 12, 'memory_swap.current': 0, 'events': {'max': 0, 'oom': 0, 'oom_kill': 0}}
    runtime_guard({'MemAvailable': 8 * 1024**2}, s)
    with pytest.raises(RuntimeError):
        runtime_guard({'MemAvailable': 8 * 1024**2 - 1}, s)
    s['memory_swap.current'] = 1
    with pytest.raises(RuntimeError):
        runtime_guard({'MemAvailable': 12 * 1024**2}, s)


def test_shared_host_guard_is_not_a_qualification_wrapper_copy():
    assert run_guarded_bc_capacity.admission is bc_memory_guard.admission
    assert run_guarded_bc_capacity.runtime_guard is bc_memory_guard.runtime_guard
    assert admission.__module__ == 'bc_memory_guard'
    assert runtime_guard.__module__ == 'bc_memory_guard'
    for bad in (None, True, '53477376'):
        with pytest.raises(RuntimeError):
            admission(bad, 43)


def test_shared_native_ack_does_not_authorize_production_data():
    base = ['docker', 'run', '--memory', '40g', '--memory-swap', '40g', '--env',
        'PYTHONPATH=/bc-production-source', 'sha256:' + 'a' * 64,
        'trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer']
    value = augment_command(base, guard_path='/guard.py', guard_sha256='b' * 64,
                            memory_gib=43, image='sha256:' + 'a' * 64)
    assert 'ROBORENDER_BC_CAPACITY_ACK=FULL_BC_BUFFERS_GUARDED_RUNTIME' in value
    assert 'ROBORENDER_BC_GUARD_NUM_ENVS=6' in value
    decision = projected_save_gate(num_envs=6, current_bytes=25 * GIB,
        limit_bytes=43 * GIB, host_available_bytes=20 * GIB, swap_bytes=0,
        events={'max': 0, 'oom': 0, 'oom_kill': 0})
    assert decision['admitted'] is True and decision['production_admitted'] is False


@pytest.mark.parametrize('b', range(1, 7))
def test_exact_frozen_tail_batch_uses_same_guard_but_not_shared_capacity_claim(b):
    reports, calls = [], []
    guarded_save(trainer(b), lambda: calls.append(1), snapshot=sample,
        host_available=lambda: 20 * GIB, limit_bytes=43 * GIB,
        expected_num_envs=b, report=lambda p, v: reports.append(v))
    assert calls == [1] and reports[0]['num_envs'] == b
    assert reports[0]['projected_additional_bytes'] == (b + 2) * GIB
    assert reports[0]['production_admitted'] is False


@pytest.mark.parametrize('b,expected', [(1, 6), (6, 1), (1, 0), (1, 7), (1, True), (7, 7)])
def test_mismatched_unfrozen_or_unsupported_batch_fails_before_native_write(b, expected):
    calls = []
    with pytest.raises(CapacityRejected):
        guarded_save(trainer(b), lambda: calls.append(1), snapshot=sample,
            host_available=lambda: 20 * GIB, limit_bytes=43 * GIB,
            expected_num_envs=expected, report=lambda *x: None)
    assert calls == []


def test_tail_wrong_actual_camera_shape_fails_before_native_write():
    t, calls = trainer(2), []
    t._depth_chunk[0].shape = (6, 480, 832, 1)
    with pytest.raises(CapacityRejected):
        guarded_save(t, lambda: calls.append(1), snapshot=sample,
            host_available=lambda: 20 * GIB, limit_bytes=43 * GIB,
            expected_num_envs=2, report=lambda *x: None)
    assert calls == []


def test_native_warm_rebuild_wraps_each_fresh_original_writer_exactly_once(monkeypatch):
    import hashlib
    import sys
    from pathlib import Path
    from types import ModuleType
    import bc_capacity_guard as native

    calls = []

    class Base:
        def __init__(self, **kwargs):
            self.serial = 0
            self.install()

        def install(self):
            serial = self.serial
            self.trainer = SimpleNamespace(_save_collected_chunks=lambda: calls.append(('writer', serial)))

        def _rebuild(self, request):
            self.serial += 1
            self.install()

    stub = ModuleType('isaac6_bc_session')
    stub.Isaac6BCSessionTrainer = Base
    stub.memory_snapshot = sample
    monkeypatch.setitem(sys.modules, 'isaac6_bc_session', stub)
    monkeypatch.setenv('ROBORENDER_BC_CAPACITY_ACK', 'FULL_BC_BUFFERS_GUARDED_RUNTIME')
    monkeypatch.setenv('ROBORENDER_BC_CAPACITY_GUARD_SHA256', hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    monkeypatch.setenv('ROBORENDER_BC_CAPACITY_MEMORY_BYTES', str(43 * GIB))
    monkeypatch.setenv('ROBORENDER_BC_GUARD_NUM_ENVS', '6')

    def observed_guard(trainer, original, **kwargs):
        calls.append(('guard', kwargs['expected_num_envs']))
        return original()

    monkeypatch.setattr(native, 'guarded_save', observed_guard)
    session = native._native_class()()
    session.trainer._save_collected_chunks()
    session._rebuild({'cohort_id': 'next-independent-cohort'})
    session.trainer._save_collected_chunks()
    assert calls == [('guard', 6), ('writer', 0), ('guard', 6), ('writer', 1)]
