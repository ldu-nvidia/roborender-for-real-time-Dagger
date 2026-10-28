"""CPU proof of continuous actor boundaries; not native GPU qualification."""
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import (
    CONTINUOUS_SCHEMA, CONTINUOUS_PLAN_SCHEMA, ProductionSegmentDriver, validate_segment,
)
from pipeline.dagger.rtx.code.src.dagger_rtx.qualification_driver import actor_ids
from test.pipeline.test_dagger_fixed100_native import fixed_inputs
from test.pipeline.test_dagger_coverage_production import production


def sha(index):
    return 'd' * 64 if index == 0 else hashlib.sha256(str(index).encode()).hexdigest()


def continuous_inputs(fixed_inputs):
    plan, ref, config, protocol = deepcopy(fixed_inputs)
    config['schema'] = 'roborender.cross_gpu_dagger_continuous_fixed100.leo.v1'
    plan.update(schema=CONTINUOUS_PLAN_SCHEMA, native_lifecycles_per_actor_per_round=1,
                optimizer_updates_per_round=100)
    for row in plan['rounds']:
        rid = row['round_id']
        assignments = row['epochs'][0]['cohorts'][0]['assignments']
        row['native_round'] = dict(cohort_id=f'training-r{rid}-continuous', start_iteration=0,
            window_count=100, qualification_after_updates=4, assignments=assignments,
            reset_at_window_boundary=False, lifecycle_policy='native_episode_done_only',
            model_rollout_lifecycle='per_81_step_window')
    segment = dict(schema=CONTINUOUS_SCHEMA, run_id=config['run_id'], round_id=1,
        cohort_id='training-r1-continuous', start_iteration_index=0, window_count=100,
        qualification_after_updates=4, lifecycle_policy='native_episode_done_only',
        input_checkpoint_sha256=sha(0), mse_protocol_file_sha256='e' * 64,
        coverage_plan_sha256=ref['raw_sha256'], episode_namespace='continuous-r1',
        motion_assignments=[dict(physical_gpu_index=g, path=f'/native/g{g}.json', sha256='f' * 64)
                            for g in (0, 1)])
    return plan, config, protocol, segment


def test_continuous_segment_is_one_fresh_round(fixed_inputs):
    plan, config, _, segment = continuous_inputs(fixed_inputs)
    assert validate_segment(segment, config, plan) == segment
    config['schema'] = 'roborender.cross_gpu_dagger_fixed100_full295.leo.v1'
    with pytest.raises(ValueError): validate_segment(segment, config, plan)


@pytest.mark.parametrize('field,value', [('start_iteration_index', 4), ('window_count', 2),
    ('qualification_after_updates', 5), ('lifecycle_policy', 'cohort_boundary'),
    ('input_checkpoint_sha256', 'a' * 64), ('episode_namespace', 'old-cohort')])
def test_continuous_segment_rejects_resumed_or_truncated_scene(fixed_inputs, field, value):
    plan, config, _, segment = continuous_inputs(fixed_inputs)
    segment[field] = value
    with pytest.raises(ValueError):
        validate_segment(segment, config, plan)


def fake_driver(tmp_path, *, bad_parent_at=None):
    """Exercise the actual driver loop without importing Isaac or Torch."""
    obj = object.__new__(ProductionSegmentDriver)
    obj.shared_population = False
    obj.config = dict(run_id='continuous-test', round_id=1, initial_checkpoint_sha256=sha(0),
        num_envs_per_gpu=16, physical_rtx_gpus=[0, 1])
    obj.round_id = obj.gpu = 1
    obj.gpu = 0
    obj.sha, obj.segment_sha, obj.source_sha256 = 'a' * 64, 'b' * 64, 'c' * 64
    obj.coverage_plan_document_sha256 = 'e' * 64
    obj.segment = dict(schema=CONTINUOUS_SCHEMA, start_iteration_index=0, window_count=100,
        lifecycle_policy='native_episode_done_only', qualification_after_updates=4,
        input_checkpoint_sha256=sha(0), coverage_plan_sha256='e' * 64)
    obj.state = dict(windows=[], uncertain_reason=None)
    obj.selected = None
    obj.path = tmp_path / 'state.json'
    calls, offsets, barriers, gates = [], [], [], []
    obj.runner = SimpleNamespace(phase_telemetry=SimpleNamespace(set_vector_step_offset=offsets.append),
        actor_prefix='continuous-test-g0', request_guard=None)

    def window(index, manage_lifecycle):
        assert manage_lifecycle is True
        calls.extend([('model_begin', index), ('window', index), ('model_clear', index)])
        return dict(transition_batches=81, physical_transition_rows=1296)
    obj.runner.run_window = window
    obj.status = lambda: dict(iterations=[dict(round_id=1, iteration_id=1, status='collecting',
        input_checkpoint_sha256=sha(0), candidate_checkpoint_sha256=None)],
        run=dict(status='collecting', active_checkpoint_sha256=sha(0)), barriers=[])
    obj.request = lambda method, path, body: (barriers.append(body) or dict(ok=True,
        iteration_id=len(calls) // 3, expected=len(actor_ids(obj.config))))

    def candidate(index, parent):
        assert parent == sha(index)
        if index == bad_parent_at:
            raise RuntimeError('candidate parent identity mismatch')
        if index == 99:
            obj.selected = dict(selected_checkpoint_sha256=sha(100), stop_reason='fixed_100_updates')
        return sha(index + 1)
    obj._candidate = candidate
    obj._wait_continuous_qualification = lambda checkpoint: gates.append((len(calls) // 3, checkpoint))
    return obj, calls, offsets, barriers, gates


def test_continuous_round_keeps_one_runner_for100_full_windows_and_exact_parents(tmp_path):
    obj, calls, offsets, barriers, gates = fake_driver(tmp_path)
    result = obj.run()
    assert len(calls) == 300 and offsets == [i * 81 for i in range(100)]
    assert [row['input_checkpoint_sha256'] for row in result['windows']] == [sha(i) for i in range(100)]
    assert len(barriers) == 1600 and gates == [(4, sha(4))]
    assert result['completed_windows'] == 100 and result['transition_batches'] == 8100
    assert result['physical_transition_rows'] == 129600
    assert result['native_environment_initializations'] == 1
    assert result['model_rollout_initializations'] == 100
    assert result['environment_reset_at_window_boundary'] is False
    assert result['final_candidate_checkpoint_sha256'] == sha(100)


def test_bad_checkpoint_barrier_cannot_advance_or_replay_next_window(tmp_path):
    obj, calls, _, _, gates = fake_driver(tmp_path, bad_parent_at=2)
    with pytest.raises(RuntimeError, match='parent identity'):
        obj.run()
    assert len(calls) == 9 and gates == []
    assert obj.state['phase'] == 'CRASH_UNCERTAIN'
    assert len(obj.state['windows']) == 2


def gate_driver(tmp_path):
    obj, _, _, _, _ = fake_driver(tmp_path)
    del obj._wait_continuous_qualification
    obj.qualification_gate_file = tmp_path / 'gate.json'
    obj.timeout = 2
    obj.monotonic = lambda: 0
    obj.resource_snapshot = lambda: dict(passed=True, actual_prefix=True)
    obj.assignment_snapshot = lambda: [dict(lane=i) for i in range(16)]
    proof = dict(schema='roborender.dagger-continuous-prefix-qualification.v1', purpose='qualification',
        run_id=obj.config['run_id'], round_id=1, plan_sha256=obj.segment['coverage_plan_sha256'],
        parent_checkpoint_sha256=sha(0), source_sha256=obj.source_sha256,
        optimizer_updates=4, full_batch_environments=32, allocation_retained=True)
    proof_path = tmp_path / 'proof.json'
    proof_path.write_text(json.dumps(proof))
    gate = dict(schema='roborender.dagger-continuous-gate.v1', run_id=obj.config['run_id'], round_id=1,
        source_sha256=obj.source_sha256, segment_sha256=obj.segment_sha, config_sha256=obj.sha,
        parent_checkpoint_sha256=sha(0), checkpoint_sha256=sha(4), optimizer_updates=4, approved=True,
        native_pids={'0': os.getpid(), '1': os.getpid() + 1},
        proof=dict(path=str(proof_path), raw_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest()))
    return obj, gate, proof_path


def test_gate_waits_without_new_window_and_admits_exact_owned_checkpoint(tmp_path):
    obj, gate, _ = gate_driver(tmp_path)
    waits = []
    def wait(seconds):
        assert obj.state['phase'] == 'WAITING_FOR_QUALIFICATION'
        waits.append(seconds)
        obj.qualification_gate_file.write_text(json.dumps(gate))
    obj.sleep = wait
    obj._wait_continuous_qualification(sha(4))
    assert waits == [1] and obj.state['phase'] == 'QUALIFIED'
    assert obj.state['native_pid'] == os.getpid()
    assert obj.state['native_environment_initializations'] == 1


@pytest.mark.parametrize('fault', ['source', 'checkpoint', 'pid', 'proof_bytes', 'config'])
def test_gate_rejects_stale_proof_or_different_native_owner(tmp_path, fault):
    obj, gate, proof = gate_driver(tmp_path)
    if fault == 'source': gate['source_sha256'] = 'd' * 64
    if fault == 'checkpoint': gate['checkpoint_sha256'] = sha(3)
    if fault == 'pid': gate['native_pids']['0'] += 1
    if fault == 'proof_bytes': proof.write_text('{}')
    if fault == 'config': gate['config_sha256'] = 'f' * 64
    obj.qualification_gate_file.write_text(json.dumps(gate))
    with pytest.raises(RuntimeError): obj._wait_continuous_qualification(sha(4))
    assert obj.state['phase'] == 'WAITING_FOR_QUALIFICATION'


def test_resource_prefix_snapshot_keeps_monitor_and_counters_alive(tmp_path):
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import PhaseResourceGuard
    from test.pipeline.test_native_resource_guard_v2 import actual
    calls = []
    def sample():
        calls.append(1)
        return actual(events=dict(max=31 + len(calls), oom=0, oom_kill=0))
    terminal = tmp_path / 'terminal.json'
    prefix = tmp_path / 'prefix.json'
    with PhaseResourceGuard(terminal, sample=sample) as guard:
        receipt = guard.snapshot(prefix)
        raw = prefix.read_bytes()
        assert guard.thread.is_alive() and not guard.stop.is_set()
        assert not terminal.exists() and receipt['scope'] == 'continuous_round_prefix'
        assert receipt['phase_limit_pressure_events'] == 1
        assert prefix.stat().st_mode & 0o777 == 0o644
        with pytest.raises(RuntimeError): guard.snapshot(prefix)
    assert prefix.read_bytes() == raw
    assert guard.summary['path'] == str(terminal)
    assert guard.summary['samples'] == 3
    assert guard.summary['phase_limit_pressure_events'] == 2


def test_real_new_plan_reaches_native_continuous_input_admission(fixed_inputs, tmp_path, monkeypatch):
    from pathlib import Path
    from pipeline.dagger.coverage_production import make_plan, make_first_config, make_continuous_segment, digest
    from pipeline.bc_training.handoff_common import publish
    from test.pipeline.test_dagger_coverage_production import rendering_dependencies
    from test.pipeline.test_dagger_native_production_integration import extracted, RESIDENT
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import read_config, canonical
    old, _, _, protocol = fixed_inputs
    plan = make_plan(**{key: old[key] for key in ('run_id', 'inventory', 'panel',
        'excluded_process_seeds', 'seed', 'motion_root', 'source_sha256', 'teacher_sha256',
        'policy_spec_sha256')}, environments_per_actor=16, fixed100=True, continuous_round=True)
    ref = publish(tmp_path / 'continuous-plan.json', plan)
    config = make_first_config(plan, coverage_plan=ref, checkpoint=tmp_path / 'parent.pt',
        checkpoint_sha256='d' * 64, gateway_pids=[101, 102, 103, 104],
        rendered_observation_config=rendering_dependencies(tmp_path, plan))
    bindings = []
    for gpu, assignment in enumerate(plan['rounds'][0]['native_round']['assignments']):
        path = tmp_path / f'assignment-{gpu}.json'
        path.write_bytes(canonical(assignment))
        bindings.append(dict(physical_gpu_index=gpu, path=str(path), sha256=digest(assignment)))
    proto = publish(tmp_path / 'protocol.json', protocol)
    item = make_continuous_segment(plan, round_id=1, input_checkpoint_sha256='d' * 64,
        mse_protocol_file_sha256=proto['raw_sha256'], motion_assignments=bindings,
        coverage_plan_sha256=ref['raw_sha256'])
    assert validate_segment(item, config, plan) == item
    segment_ref = publish(tmp_path / 'segment.json', item)
    config_ref = publish(tmp_path / 'config.json', config)
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', '0')
    ns = extracted(RESIDENT, ['_production_segment_inputs'], klass='ResidentRavenBackend',
                   namespace=dict(json=json, os=os, read_config=read_config))
    backend = SimpleNamespace(artifact=Path, motion_inventory_sha256=plan['rounds'][0]['native_round']['assignments'][0]['motion_inventory_sha256'])
    payload = dict(round_config=config_ref['path'], round_config_sha256=config_ref['raw_sha256'],
        coverage_plan_file=ref['path'], coverage_plan_sha256=ref['raw_sha256'],
        mse_protocol_file=proto['path'], mse_protocol_sha256=proto['raw_sha256'],
        segment_file=segment_ref['path'], segment_sha256=segment_ref['raw_sha256'],
        motion_assignment_file=bindings[0]['path'], motion_assignment_sha256=bindings[0]['sha256'])
    result = ns['_production_segment_inputs'](backend, payload)
    assert result[1]['window_count'] == 100 and result[1]['start_iteration_index'] == 0
    assert result[2] == plan['rounds'][0]['native_round']['assignments'][0]
