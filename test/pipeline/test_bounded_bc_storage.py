"""Pinned Raven differential tests; CPU-only, not live B32 certification."""
from copy import deepcopy
import ast
import hashlib
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.deployment.rtx_actor import bounded_bc_storage as storage
from pipeline.deployment.rtx_actor import bounded_bc_guard as guard
from pipeline.deployment.rtx_actor import continuous_bc_guard as legacy_guard
from pipeline.deployment.rtx_actor import isaac6_bc_production as native
from pipeline.bc_collection import continuous_plan, run_continuous
from test.pipeline.test_bc_native_same_input import owner as reference_owner
from test.pipeline.test_continuous_bc_launcher import fixture_capacity

GIB = 2**30
REFERENCE = Path(__file__).resolve().parents[2] / (
    'runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py')


def owner(output, batch):
    output.mkdir()
    result = reference_owner(output)
    result.env.num_envs = batch
    result.vg_mode, result.vg_save_sim_rgb = 'collect', False
    return result


def populate(value, window, order, *, archive=False):
    b = value.env.num_envs
    value._depth_chunk, value._robot_rgb_chunk, value._views_chunk = [], [], []
    for t in range(81):
        depth = torch.arange(b * 4 * 6, dtype=torch.float32).reshape(b, 4, 6, 1) / 37
        depth += window * 3 + t / 300
        depth[:, 0, 0] = 0
        depth[:, 0, 1] = float('inf')
        depth[:, 0, 2] = float('nan')
        depth[:, 0, 3] = -1
        rgb = torch.full((b, 4, 6, 3), (window * 81 + t) % 256, dtype=torch.uint8)
        value._depth_chunk.append(depth)
        value._robot_rgb_chunk.append(rgb)
        value._views_chunk.append({view: dict(depth=depth + i / 9, robot_rgb=rgb.clone(), rgb=None)
                                   for i, view in enumerate(order)})
        if archive:
            storage.archive_latest_capture(value, 'cpu')
    rows = torch.arange(window * 81, (window + 1) * 81).float()[:, None, None]
    value.storage = SimpleNamespace(actor_obs=rows.expand(81, b, 113).clone(),
        teacher_obs=rows.expand(81, b, 226).clone(), gt_actions=rows.expand(81, b, 26).clone(),
        dones=torch.zeros(81, b, 1, dtype=torch.bool))
    for lane in range(b):
        # Ends before emission, on emission, at the first row, and nonterminal
        # continuations produce different episode ages across one shared batch.
        if window == 0:
            value.storage.dones[(39, 80, 79, 0)[lane % 4], lane] = True
        if window == 2 and lane % 2:
            value.storage.dones[0, lane] = True
            value.storage.dones[79, lane] = True
        if window == 3:
            value.storage.dones[69, lane] = True


def assert_state_equal(left, right):
    assert left._vg_saved_chunks == right._vg_saved_chunks
    assert left._ep_seq_base == right._ep_seq_base
    for a, b in zip(left._ep_buf, right._ep_buf):
        assert list(a) == list(b)
        if not a:
            continue
        for name in ('frames', 'obs'):
            assert list(a[name]) == list(b[name])
            for key in a[name]:
                assert len(a[name][key]) == len(b[name][key])
                for expected, actual in zip(a[name][key], b[name][key]):
                    assert expected.dtype == actual.dtype
                    np.testing.assert_array_equal(expected, actual)
                    assert actual.base is None and actual.flags.owndata
        for expected, actual in zip(a['gt'], b['gt']):
            np.testing.assert_array_equal(expected, actual)
            assert actual.base is None


@pytest.mark.parametrize('batch', [1, 2, 16, 32, 40, 50])
def test_full_population_four_windows_matches_pinned_writer_bytes_and_rng(tmp_path, batch):
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == native.NATIVE_COLLECTOR_SHA256
    ref, bounded = owner(tmp_path / 'original', batch), owner(tmp_path / 'bounded', batch)
    order = ['hand_left', 'hand_right', 'head_color']
    for window in range(4):
        populate(ref, window, order)
        populate(bounded, window, order, archive=True)
        before = torch.get_rng_state().clone()
        before_np = np.random.get_state()
        with np.errstate(invalid='ignore'):
            ref._save_collected_chunks()
            storage.write_window(bounded)
        assert torch.equal(torch.get_rng_state(), before)
        after_np = np.random.get_state()
        assert before_np[0] == after_np[0] and before_np[2:] == after_np[2:]
        np.testing.assert_array_equal(before_np[1], after_np[1])
        assert_state_equal(ref, bounded)
        assert bounded._depth_chunk == bounded._views_chunk == bounded._robot_rgb_chunk == []
        paths = sorted((tmp_path / 'original').glob('*.npz'))
        assert [p.name for p in paths] == sorted(p.name for p in (tmp_path / 'bounded').glob('*.npz'))
        for path in paths:
            other = tmp_path / 'bounded' / path.name
            assert path.read_bytes() == other.read_bytes()
    assert all(len(buf['gt']) == 11 for buf in bounded._ep_buf)  # Final tails were NOT flushed.


@pytest.mark.parametrize('order', list(itertools.permutations(sorted(storage.VIEWS))))
def test_view_key_order_preserved_for_npz_bytes(tmp_path, order):
    ref, bounded = owner(tmp_path / 'original', 2), owner(tmp_path / 'bounded', 2)
    populate(ref, 0, order)
    populate(bounded, 0, order)
    with np.errstate(invalid='ignore'):
        ref._save_collected_chunks()
        storage.write_window(bounded)
    for path in (tmp_path / 'original').glob('*.npz'):
        assert path.read_bytes() == (tmp_path / 'bounded' / path.name).read_bytes()
    assert_state_equal(ref, bounded)


def test_archiving_owns_snapshot_and_rejects_dtype_changes(tmp_path):
    value = owner(tmp_path / 'bounded', 2)
    populate(value, 0, sorted(storage.VIEWS))
    original = value._depth_chunk[-1]
    storage.archive_latest_capture(value, 'cpu')
    assert original.data_ptr() != value._depth_chunk[-1].data_ptr()
    original.fill_(999)
    assert not bool((value._depth_chunk[-1] == 999).all())
    value._depth_chunk[-1] = value._depth_chunk[-1].half()
    with pytest.raises(RuntimeError, match='dtype/device'):
        storage.archive_latest_capture(value, 'cpu')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0., .001, .05, 1.5, 10., 20.])
def test_degenerate_depth_and_quantization_edges_match_original_bytes(tmp_path, value):
    ref, bounded = owner(tmp_path / 'original', 1), owner(tmp_path / 'bounded', 1)
    for item in (ref, bounded):
        populate(item, 1, sorted(storage.VIEWS))
        for frame in item._views_chunk:
            for view in frame.values():
                view['depth'].fill_(value)
        for depth in item._depth_chunk:
            depth.fill_(value)
    with np.errstate(invalid='ignore'):
        ref._save_collected_chunks()
        storage.write_window(bounded)
    original = next((tmp_path / 'original').glob('*.npz'))
    assert original.read_bytes() == (tmp_path / 'bounded' / original.name).read_bytes()


def test_conversion_is_lane_bounded_not_two_independent_populations(tmp_path, monkeypatch):
    value = owner(tmp_path / 'bounded', 32)
    populate(value, 0, sorted(storage.VIEWS))
    calls, original = [], storage._lane_arrays
    def convert(owner, lane):
        assert owner.env.num_envs == 32
        arrays = original(owner, lane)
        assert all(array.shape[:2] == (81, 1) for array in arrays.values())
        calls.append(lane)
        return arrays
    monkeypatch.setattr(storage, '_lane_arrays', convert)
    with np.errstate(invalid='ignore'):
        storage.write_window(value)
    assert calls == list(range(32))


def test_lane_barrier_releases_conversion_scratch_but_preserves_tails(tmp_path, monkeypatch):
    import weakref
    value = owner(tmp_path / 'bounded', 40)
    populate(value, 0, sorted(storage.VIEWS))
    original = storage._lane_arrays
    refs, boundaries = [], []
    def convert(actor, lane):
        arrays = original(actor, lane)
        refs[:] = [weakref.ref(array) for array in arrays.values()]
        return arrays
    def barrier(lane):
        assert all(ref() is None for ref in refs)
        assert value.env.num_envs == 40 and len(value._views_chunk) == 81
        assert len(value._ep_buf[lane].get('gt', [])) == (41, 0, 1, 80)[lane % 4]
        boundaries.append(lane)
    value._bc_writer_lane_boundary = barrier
    monkeypatch.setattr(storage, '_lane_arrays', convert)
    with np.errstate(invalid='ignore'):
        storage.write_window(value)
    assert boundaries == list(range(40))


def test_payload_cache_release_preserves_bytes_and_fsyncs_before_advice(tmp_path, monkeypatch):
    import os
    path = tmp_path / 'payload.npz'
    np.savez_compressed(path, values=np.arange(1000))
    original = path.read_bytes()
    calls = []
    fsync, advise = os.fsync, os.posix_fadvise
    def sync(fd):
        calls.append('fsync')
        return fsync(fd)
    def drop(fd, offset, count, advice):
        assert calls == ['fsync']
        assert (offset, count, advice) == (0, 0, os.POSIX_FADV_DONTNEED)
        calls.append('advice')
        return advise(fd, offset, count, advice)
    monkeypatch.setattr(os, 'fsync', sync)
    monkeypatch.setattr(os, 'posix_fadvise', drop)
    result = storage.release_payload_cache(path)
    assert result['bytes'] == len(original) and result['advisory_only']
    assert result['fsync_completed'] and path.read_bytes() == original
    assert calls == ['fsync', 'advice']


@pytest.mark.parametrize('kind', ['empty', 'directory', 'symlink'])
def test_payload_cache_release_rejects_non_payloads(tmp_path, kind):
    path = tmp_path / 'payload'
    if kind == 'empty': path.touch()
    elif kind == 'directory': path.mkdir()
    else: path.symlink_to(tmp_path / 'absent')
    with pytest.raises((RuntimeError, OSError)):
        storage.release_payload_cache(path)


def test_writer_cache_callback_receives_closed_complete_payloads(tmp_path):
    value = owner(tmp_path / 'bounded', 40)
    populate(value, 1, sorted(storage.VIEWS))
    calls = []
    def written(path):
        with np.load(path, allow_pickle=False) as archive:
            assert archive['gt_actions'].shape == (81, 26)
        storage.release_payload_cache(path)
        calls.append(path)
    value._bc_payload_written = written
    with np.errstate(invalid='ignore'):
        storage.write_window(value)
    assert len(calls) == len(set(calls)) == 40


def test_cache_failure_does_not_accept_or_retry_written_payload(tmp_path):
    value = owner(tmp_path / 'bounded', 1)
    populate(value, 1, sorted(storage.VIEWS))
    def fail(path):
        raise OSError('injected fsync failure')
    value._bc_payload_written = fail
    with pytest.raises(OSError, match='fsync failure'), np.errstate(invalid='ignore'):
        storage.write_window(value)
    assert value._vg_saved_chunks == 0
    assert len(list(Path(value.vg_collect_dir).glob('*.npz'))) == 1


@pytest.mark.parametrize('fault', ['short', 'order', 'mode', 'raw', 'debug', 'population'])
def test_incomplete_or_changed_writer_recipe_fails_closed(tmp_path, fault):
    value = owner(tmp_path / 'bounded', 2)
    populate(value, 0, sorted(storage.VIEWS))
    if fault == 'short': value._depth_chunk.pop()
    if fault == 'order': value._views_chunk[2] = dict(reversed(list(value._views_chunk[2].items())))
    if fault == 'mode': value.vg_mode = 'online'
    if fault == 'raw': value.vg_save_raw_depth_native = False
    if fault == 'debug': value.vg_save_sim_rgb = True
    if fault == 'population': value._ep_buf, value._ep_seq_base = [{}], [0]
    with pytest.raises(RuntimeError): storage.write_window(value)
    assert not list(Path(value.vg_collect_dir).glob('*.npz'))


def snapshot():
    return dict(memory_current=30 * GIB, memory_peak=31 * GIB, memory_max=43 * GIB,
                **{'memory_swap.current': 0}, events=dict(max=0, oom=0, oom_kill=0))


def test_distinct_projection_does_not_relax_original_guard():
    assert storage.raw_window_bytes(32) == 28983951360
    with pytest.raises(RuntimeError):
        legacy_guard.memory_gate(snapshot(), 30 * GIB, num_envs=32, cap_bytes=43 * GIB, saving=True)
    result = guard.memory_gate(snapshot(), 30 * GIB, num_envs=32, cap_bytes=43 * GIB,
                              saving=True, tail_growth_bytes=0)
    assert result['projected_additional_bytes'] == 8 * GIB
    assert result['projection_is_proven_upper_bound'] is False
    assert result['protocol'] == storage.PROTOCOL


@pytest.mark.parametrize('fault', ['swap', 'oom', 'oom_kill', 'cap', 'host', 'counters', 'negative_counter'])
def test_new_guard_keeps_all_resource_floor_and_event_rejections(fault):
    value, available = snapshot(), 30 * GIB
    if fault == 'swap': value['memory_swap.current'] = 1
    if fault == 'oom': value['events']['oom'] = 1
    if fault == 'oom_kill': value['events']['oom_kill'] = 1
    if fault == 'negative_counter': value['events']['max'] = -1
    if fault == 'cap': value['memory_current'] = 34 * GIB
    if fault == 'host': available = 15 * GIB
    if fault == 'counters': value.pop('memory_peak')
    with pytest.raises(RuntimeError):
        guard.memory_gate(value, available, num_envs=32, cap_bytes=43 * GIB,
                          saving=True, tail_growth_bytes=0)


def test_bounded_save_records_reclaimed_limit_event_without_waiving_headroom():
    value = snapshot()
    value['events']['max'] = 1
    result = guard.memory_gate(value, 30 * GIB, num_envs=32, cap_bytes=43 * GIB,
                               saving=True, tail_growth_bytes=0)
    assert result['recovered_limit_events'] == 1 and result['capacity_certified'] is False
    value['memory_current'] = 34 * GIB
    with pytest.raises(RuntimeError, match='headroom'):
        guard.memory_gate(value, 30 * GIB, num_envs=32, cap_bytes=43 * GIB,
                          saving=True, tail_growth_bytes=0)


def test_gpu_admits_whole_remaining_window_before_any_step_not_only_next_frame():
    good = guard.gpu_gate(64 * GIB, 96 * GIB, num_envs=32, captured_rows=0)
    assert good['remaining_raw_window_bytes'] == storage.raw_window_bytes(32)
    assert good['lane_stack_bytes'] == 81 * 480 * 832 * 4
    assert good['allocator_slack_bytes'] == 2 * GIB
    with pytest.raises(RuntimeError): guard.gpu_gate(40 * GIB, 96 * GIB, num_envs=32, captured_rows=0)
    with pytest.raises(RuntimeError): guard.gpu_gate(10 * GIB, 96 * GIB, num_envs=32, captured_rows=81)


def test_save_cannot_hide_copied_tail_growth_inside_scratch_allowance():
    with pytest.raises(RuntimeError, match='tail growth'):
        guard.memory_gate(snapshot(), 30 * GIB, num_envs=32, cap_bytes=43 * GIB, saving=True)
    with pytest.raises(RuntimeError, match='headroom'):
        guard.memory_gate(snapshot(), 40 * GIB, num_envs=32, cap_bytes=43 * GIB,
                          saving=True, tail_growth_bytes=10 * GIB)
    value = snapshot()
    value['memory_current'] = 20 * GIB
    result = guard.memory_gate(value, 40 * GIB, num_envs=32, cap_bytes=43 * GIB,
                              saving=True, tail_growth_bytes=10 * GIB)
    assert result['projected_additional_bytes'] == 18 * GIB


def test_tail_payload_high_water_includes_all_lanes_and_pre_emit_row(tmp_path):
    value = owner(tmp_path / 'bounded', 32)
    populate(value, 0, sorted(storage.VIEWS))
    # One terminal at row0 leaves80 rows in every lane, the largest retained
    # tail possible. At native resolution these image tails alone are9.9976GiB.
    value.storage.dones.zero_()
    value.storage.dones[0] = True
    budget = storage.tail_memory_budget(value)
    assert budget['current_tail_bytes'] == 0
    assert budget['peak_tail_bytes'] == budget['final_tail_bytes'] == 32 * 80 * budget['bytes_per_tail_row']
    with np.errstate(invalid='ignore'):
        storage.write_window(value)
    populate(value, 1, sorted(storage.VIEWS))
    value.storage.dones.zero_()
    next_budget = storage.tail_memory_budget(value)
    assert next_budget['current_tail_bytes'] == budget['final_tail_bytes']
    assert next_budget['additional_tail_bytes'] == budget['bytes_per_tail_row']
    assert next_budget['peak_tail_bytes'] == (32 * 80 + 1) * budget['bytes_per_tail_row']


@pytest.mark.parametrize('target', ['primary_depth', 'primary_rgb', 'view_depth', 'view_rgb'])
@pytest.mark.parametrize('fault', ['dtype', 'device'])
def test_any_foreign_device_or_dtype_is_rejected_before_save(tmp_path, target, fault):
    value = owner(tmp_path / 'bounded', 2)
    populate(value, 0, sorted(storage.VIEWS))
    storage.validate_device_buffers(value, 'cpu')
    if target.startswith('primary'):
        container = value._depth_chunk if target.endswith('depth') else value._robot_rgb_chunk
        key = 17
    else:
        container, key = value._views_chunk[17]['hand_left'], 'depth' if target.endswith('depth') else 'robot_rgb'
    container[key] = (container[key].to('meta') if fault == 'device' else
                      container[key].to(torch.float64))
    with pytest.raises(RuntimeError, match='device/dtype'):
        storage.validate_device_buffers(value, 'cpu')
    assert not list(Path(value.vg_collect_dir).glob('*.npz'))


@pytest.mark.parametrize('schema', [native.SCHEMA, native.SCHEMA_V2, 'unknown'])
def test_bounded_storage_cannot_bypass_continuous_request_schema(schema):
    request = bounded_request()
    request['schema'] = schema
    with pytest.raises(ValueError, match='schema/keys'):
        native.validate_collection_request(request)


@pytest.mark.parametrize('windows', [True, 1, 2, 5, 9, 100, 4.0])
def test_native_request_rejects_bypassing_qualification_window_budget(windows):
    request = bounded_request()
    request['max_windows'] = windows
    with pytest.raises(ValueError, match='4/8-window'):
        native.validate_collection_request(request)


def bounded_request(b=2):
    motions = [dict(motion_id=f'm{i}', path=f'/motions/m{i}.npz', sha256=f'{i:064x}') for i in range(b)]
    return continuous_plan.build_request(continuous_plan.build_plan(motions, run_id='bounded-smoke',
        output_dir='/dagger-runtime/bc_isaac', max_windows=4, num_envs=b,
        collection_storage_protocol=storage.PROTOCOL))


def test_explicit_opt_in_never_changes_default_or_admits_production():
    request = bounded_request()
    assert native.validate_collection_request(request) == request
    assert run_continuous.validate_launch(request, qualification=True, expected_num_envs=2,
        physical_gpu=0, memory_gib=43, timeout=3600) == request
    with pytest.raises(ValueError, match='qualification-only'):
        run_continuous.validate_launch(request, qualification=False, expected_num_envs=2,
            physical_gpu=0, memory_gib=43, timeout=3600)
    legacy = deepcopy(request)
    legacy.pop('collection_storage_protocol')
    assert run_continuous.capacity_contract({}, legacy, gpu_uuid='GPU-test', memory_gib=43)['recipe'] != storage.PROTOCOL
    assert run_continuous.capacity_contract({}, request, gpu_uuid='GPU-test', memory_gib=43)['recipe'] == storage.PROTOCOL
    for protocol in [None, 'unreviewed', 1]:
        invalid = deepcopy(request)
        invalid['collection_storage_protocol'] = protocol
        with pytest.raises(ValueError): native.validate_collection_request(invalid)
    invalid = deepcopy(request)
    invalid.pop('collection_rng_protocol')
    with pytest.raises(ValueError, match='RNG'): native.validate_collection_request(invalid)


def test_new_command_pins_distinct_guard_and_keeps_legacy_command_unchanged():
    image = 'sha256:' + 'a' * 64
    def base(name, runtime, overlay, code, b, **kwargs):
        return ['docker', 'run', '--name', name, '--memory', '40g', '--memory-swap', '40g',
                '--env', 'PYTHONPATH=/isaac6-port', image, 'python', 'train.py',
                'trainer._target_=old', 'algo.config.num_steps_per_env=81']
    request = bounded_request(32)
    _, command = run_continuous.build_command(request=request, request_path='/request.json',
        request_sha256='b' * 64, source=Path('/frozen'), runtime=Path('/runtime'), overlay=Path('/overlay'),
        physical_gpu=0, memory_gib=43, guard_sha256='c' * 64, base_builder=base, image_id=image)
    assert 'trainer._target_=bounded_bc_guard.BoundedContinuousBCTrainer' in command
    assert 'ROBORENDER_CONTINUOUS_BC_ACK=' + guard.ACK in command
    assert 'num_envs=32' in command and 'algo.config.num_learning_epochs=0' in command
    assert command[command.index('--memory') + 1] == command[command.index('--memory-swap') + 1] == '43g'


def test_bounded_receipts_must_name_changed_writer_and_actual_source():
    values = fixture_capacity()
    request = values['launch']['request']
    request['collection_storage_protocol'] = storage.PROTOCOL
    values['native']['request'] = request
    values['native']['native_payload_writer_unchanged'] = False
    values['native']['collection_storage'] = dict(protocol=storage.PROTOCOL, completed_windows=4,
        num_envs=4, frames_per_window=81, logical_scenes=1, cpu_conversion_lanes=1,
        capture_device='cuda:0', production_admitted=False, writer_sha256='d' * 64)
    values['launch']['collector_execution']['storage_writer_sha256'] = 'd' * 64
    values['execution_receipt']['request_canonical_sha256'] = run_continuous.digest(native.canonical(request))
    values['launch']['capacity_contract']['recipe'] = storage.PROTOCOL
    for row in values['save_events']:
        row['protocol'] = storage.PROTOCOL
        if row['phase'] == 'postsave':
            row['completed_bounded_save'] = row.pop('completed_original_save')
            row['native_payload_writer_unchanged'] = False
    result = run_continuous.build_capacity(**values)
    assert result['production_admitted'] is False
    values['native']['native_payload_writer_unchanged'] = True
    with pytest.raises(ValueError): run_continuous.build_capacity(**values)
    values['native']['native_payload_writer_unchanged'] = False
    values['native']['collection_storage']['writer_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='writer source'): run_continuous.build_capacity(**values)


def bounded_cpu_class(base, monkeypatch, events):
    """Execute the actual native hook class with explicitly CPU-only services.

    CUDA device admission, resource sampling and native image resolution are
    doubles here; their fail-closed gates have separate tests. The original
    capture/policy/rollout and new archive/write hooks are not replaced.
    """
    from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
    node = next(node for node in ast.walk(ast.parse(Path(guard.__file__).read_text()))
                if isinstance(node, ast.ClassDef) and node.name == 'BoundedContinuousBCTrainer')
    def archive(owner, device):
        assert owner.env.events[-1] == (owner.env.steps, 'policy')
        storage.archive_latest_capture(owner, device)
        events.append(('archive', owner.env.steps, len(owner._depth_chunk)))
    def check_shapes(owner, batch):
        assert batch == owner.env.num_envs == 3
        assert len(owner._depth_chunk) == len(owner._views_chunk) == 81
        assert all(tuple(row.shape) == (3, 4, 6, 1) for row in owner._depth_chunk)
    sample = dict(snapshot(), memory_current=20 * GIB)
    namespace = dict(vars(guard), torch=torch, Isaac6BCProductionTrainer=base, b=3, cap=43 * GIB, production=False,
        available=lambda: 64 * GIB, memory_snapshot=lambda: deepcopy(sample),
        validate_full_buffers=check_shapes, archive_latest_capture=archive,
        write_window=storage.write_window, validate_device_buffers=storage.validate_device_buffers,
        tail_memory_budget=storage.tail_memory_budget, wait_for_observer=wait_for_observer,
        report=lambda phase, value: events.append((phase, value)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), guard.__file__, 'exec'), namespace)
    cls = namespace['BoundedContinuousBCTrainer']
    # Host heap and /dagger-runtime telemetry are native-only boundaries.
    monkeypatch.setattr(cls, '_bc_writer_lane_boundary',
                        lambda self, lane: events.append(('lane_boundary', lane)))
    monkeypatch.setattr(cls, '_bc_payload_written',
                        lambda self, path, phase='written': events.append(('payload_cache', str(path), phase)))
    monkeypatch.setattr(cls, '_bounded_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *_: (64 * GIB, 96 * GIB))
    monkeypatch.setitem(sys.modules, 'bounded_bc_storage', storage)
    return cls


def test_native_memory_rejection_preserves_exact_snapshot(monkeypatch):
    events = []
    cls = bounded_cpu_class(object, monkeypatch, events)
    bad = snapshot()
    bad['events']['oom'] = 1
    cls._check_host_memory.__globals__['memory_snapshot'] = lambda: deepcopy(bad)
    with pytest.raises(RuntimeError):
        cls()._check_host_memory(saving=True, tail_growth_bytes=123)
    phase, value = events[-1]
    assert phase == 'memory_rejected' and value['memory'] == bad
    assert value['saving'] is True and value['tail_growth_bytes'] == 123
    assert value['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('windows', [4, 8])
def test_original_policy_archive_save_hooks_across_windows_match_original_adapter(
        tmp_path, monkeypatch, windows):
    from test.pipeline import test_bc_collection_functional_parity as fixture
    from pipeline.deployment.rtx_actor import initial_reset_contract

    _, original_adapter = fixture._classes(monkeypatch)
    events = []
    bounded_class = bounded_cpu_class(original_adapter, monkeypatch, events)
    staged = ModuleType('gr00t.rl.envs.base_task.staged_task_base')
    staged.StagedTaskBase = fixture.StagedBoundary
    monkeypatch.setitem(sys.modules, staged.__name__, staged)
    monkeypatch.setitem(sys.modules, 'initial_reset_contract', initial_reset_contract)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *_: torch.get_rng_state())
    def cpu_model_gate(owner):
        assert owner.env.device == 'cpu'  # Declared physics/architecture double.
        receipt = native.bc_collection_rng.expected_receipt(3, 0)
        receipt.pop('completed_windows')
        return receipt
    monkeypatch.setattr(native.bc_collection_rng, 'validate_native', cpu_model_gate)
    inventory = [dict(motion_id=f'm{i}', path=f'/motions/m{i}.npz', sha256=f'{i:064x}') for i in range(5)]
    template = continuous_plan.build_request(continuous_plan.build_plan(inventory,
        run_id='bounded-hook-cpu', output_dir=str(tmp_path / 'bounded'), max_windows=windows,
        num_envs=3, collection_storage_protocol=storage.PROTOCOL))
    model_file = tmp_path / 'cpu-only-model-identity'
    model_file.write_bytes(b'CPU double, not native teacher weights')
    owners, receipts, random_states = [], [], []
    saved_rng = torch.get_rng_state()
    try:
        for name, cls in [('original', original_adapter), ('bounded', bounded_class)]:
            request = dict(template, output_dir=str(tmp_path / name))
            if name == 'original':
                request.pop('collection_storage_protocol')
            request_file = tmp_path / (name + '-request.json')
            request_file.write_bytes(native.canonical(request))
            monkeypatch.setenv('ROBORENDER_BC_COLLECTION_REQUEST', str(request_file))
            monkeypatch.setenv('ROBORENDER_BC_COLLECTION_REQUEST_SHA256', native.file_sha256(request_file))
            owner = fixture._owner(cls, request, model_file, [2, 3])
            torch.manual_seed(0)
            receipts.append(owner.train())
            random_states.append(torch.get_rng_state())
            owners.append(owner)
    finally:
        torch.set_rng_state(saved_rng)
    reference, bounded = owners
    assert torch.equal(*random_states)
    assert reference.env.events == bounded.env.events
    assert reference.env.reset_calls == bounded.env.reset_calls == 1
    assert reference.env.lane_resets == bounded.env.lane_resets
    assert reference.lifecycle == bounded.lifecycle
    torch.testing.assert_close(torch.stack(reference.env.actions), torch.stack(bounded.env.actions), rtol=0, atol=0)
    assert_state_equal(reference, bounded)
    expected_files = sorted((tmp_path / 'original').glob('*.npz'))
    actual_files = sorted((tmp_path / 'bounded').glob('*.npz'))
    assert len(expected_files) == len(actual_files) > 0
    for before, after in zip(expected_files, actual_files):
        assert before.name == after.name and before.read_bytes() == after.read_bytes()
    assert [(e[1], e[2]) for e in events if e[0] == 'archive'] == [
        (step, step % 81 + 1) for step in range(windows * 81)]
    assert len([e for e in events if e[0] == 'presave']) == windows
    assert all(e[1]['completed_bounded_save'] and not e[1]['native_payload_writer_unchanged']
               for e in events if e[0] == 'postsave')
    for key in ['vector_steps', 'emitted_chunks', 'remaining_unsealed_tail_rows_per_lane',
                'discarded_reset_tail_rows_per_lane', 'collection_rng', 'initial_state_attestation']:
        assert receipts[0][key] == receipts[1][key]
    assert receipts[0]['native_payload_writer_unchanged'] is True
    assert receipts[1]['native_payload_writer_unchanged'] is False
    assert receipts[1]['collection_storage']['completed_windows'] == windows
    assert receipts[1]['collection_storage']['capture_device'] == 'cpu'  # Not live CUDA evidence.


@pytest.mark.parametrize('bounded_class_requested', [False, True])
def test_storage_request_cannot_select_the_other_native_trainer(tmp_path, monkeypatch, bounded_class_requested):
    from test.pipeline import test_bc_collection_functional_parity as fixture
    _, adapter = fixture._classes(monkeypatch)
    cls = bounded_cpu_class(adapter, monkeypatch, []) if bounded_class_requested else adapter
    request = bounded_request(3)
    if bounded_class_requested:
        request.pop('collection_storage_protocol')
    path = tmp_path / 'request.json'
    path.write_bytes(native.canonical(request))
    monkeypatch.setenv('ROBORENDER_BC_COLLECTION_REQUEST', str(path))
    monkeypatch.setenv('ROBORENDER_BC_COLLECTION_REQUEST_SHA256', native.file_sha256(path))
    # No env/model even exists: reject the storage mismatch before touching it.
    with pytest.raises(RuntimeError, match='storage implementation'):
        cls().train()
