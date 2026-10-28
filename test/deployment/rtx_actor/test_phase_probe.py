"""CPU-only tests; these do not certify native stepping or warm GPU reuse."""
from test.deployment.source_paths import ACTOR_ROOT
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ACTOR = ACTOR_ROOT
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR))
sys.path.insert(0, str(ROOT / 'pipeline/dagger/rtx/code/src'))

import phase_probe as probe
from test.deployment.rtx_actor.test_leo_capacity_probe import Batch
from test.deployment.rtx_actor.test_resident_phase_transitions import methods
from test.deployment.rtx_actor.test_warm_camera_diagnostic import plan as warm_plan
from dagger_rtx.resident_service import ResidentOwner, SCHEMA as OWNER_SCHEMA, validate_command


def fixture():
    source = warm_plan()
    assignment = deepcopy(source['arms']['0'][0]['assignment'])
    assignment.update(run_id=probe.RUN_ID, cohort_id='phase-probe-one')
    value = dict(schema=probe.SCHEMA, purpose='warm_phase_smoke', command_id='phase-probe-one',
        source_sha256='b'*64, physical_gpu=0, native_pid=os.getpid(), worker_instance_id='a'*32,
        checkpoint_sha256='c'*64, assignment=assignment, num_envs=16, steps=81,
        acknowledgement=probe.ACK)
    state = dict(schema=OWNER_SCHEMA, native_pid=os.getpid(), worker_instance_id='a'*32,
        source_sha256='b'*64, native_initializations=1, phase='IDLE', active_command_id=None)
    return value, state, source['inventory']


def validate_owner(value, state, inventory):
    return probe.validate_owner(value, state, source_sha256='b'*64, physical_gpu=0,
                                native_pid=os.getpid(), inventory=inventory)


@pytest.mark.parametrize('mutation', [
    {'steps': 80}, {'steps': 82}, {'steps': 162}, {'steps': True},
    {'num_envs': 2}, {'num_envs': 8}, {'num_envs': 32}, {'num_envs': True},
    {'purpose': 'production'}, {'acknowledgement': 'yes'}, {'physical_gpu': True},
    {'worker_instance_id': 'x'*32}, {'source_sha256': 'bad'}, {'native_pid': False},
    {'student_inference': True}, {'command_id': '../escape'},
])
def test_exact_smoke_budget_and_owner_shape(mutation):
    value, _, _ = fixture()
    value.update(mutation)
    with pytest.raises(ValueError):
        probe.validate_plan(value)


@pytest.mark.parametrize('mutation', [
    {'purpose': 'test'}, {'purpose': 'validation'}, {'run_id': 'production'}, {'cohort_id': 'other'},
])
def test_training_only_assignment_cannot_adopt_test_or_another_command(mutation):
    value, _, _ = fixture()
    value['assignment'].update(mutation)
    with pytest.raises(ValueError):
        probe.validate_plan(value)


def test_plan_hash_regular_file_and_detached_validation(tmp_path):
    value, _, _ = fixture()
    path = tmp_path / 'plan.json'
    path.write_bytes(probe.canonical(value))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert probe.read_plan(path, sha) == value
    copied = probe.validate_plan(value)
    copied['assignment']['lanes'][0]['motion_id'] = 'changed'
    assert value['assignment']['lanes'][0]['motion_id'] == 'm0'
    with pytest.raises(ValueError, match='digest'):
        probe.read_plan(path, 'f'*64)
    link = tmp_path / 'link'; link.symlink_to(path)
    with pytest.raises(ValueError, match='regular'):
        probe.read_plan(link, sha)


@pytest.mark.parametrize('mutation', [
    {'schema': 'legacy'}, {'native_pid': 1}, {'worker_instance_id': 'e'*32},
    {'source_sha256': 'e'*64}, {'native_initializations': 2},
    {'phase': 'CRASH_UNCERTAIN'}, {'phase': 'QUEUED'},
    {'active_command_id': 'other'}, {'phase': 'RUNNING', 'active_command_id': 'other'},
])
def test_owner_mismatch_rejected_before_mutation(mutation):
    value, state, inventory = fixture()
    state.update(mutation)
    with pytest.raises(ValueError, match='identity'):
        validate_owner(value, state, inventory)


def test_admission_and_running_revalidation_are_distinct_but_same_identity():
    value, state, inventory = fixture()
    assert validate_owner(value, state, inventory) == value
    state.update(phase='RUNNING', active_command_id=value['command_id'])
    assert validate_owner(value, state, inventory) == value
    value['physical_gpu'] = 1
    with pytest.raises(ValueError):
        validate_owner(value, state, inventory)


@pytest.mark.parametrize('which', ['inventory', 'assignment_sha', 'assignment_name'])
def test_inventory_binding_cannot_be_relabelled(which):
    value, state, inventory = fixture()
    if which == 'inventory':
        inventory['motions'][0]['bytes'] = 2
    elif which == 'assignment_sha':
        value['assignment']['lanes'][0]['sha256'] = 'd'*64
    else:
        value['assignment']['lanes'][0]['path'] = '/motions/other.npz'
    with pytest.raises(ValueError):
        validate_owner(value, state, inventory)


class GeneratedBatch(Batch):
    num_envs = 16

    def capture_batch(self, keys):
        values = super().capture_batch(keys)
        for value in values:
            value.metadata['observation_source'] = 'generated_controls'
        return values


def execute_steps(batch):
    value, _, _ = fixture()
    calls, events = [], []
    lifecycle = SimpleNamespace(begin=lambda: calls.append('begin'), finish=lambda: calls.append('finish'))
    result = probe.run_probe_steps(batch, lifecycle, value, report=events.append)
    return result, calls, events


def test_exact_b16_teacher_steps_natural_resets_and_no_production_claim():
    batch = GeneratedBatch()
    result, calls, events = execute_steps(batch)
    assert calls == ['begin', 'finish'] and len(batch.steps) == 81
    assert result['environment_step_calls'] == 81 and result['physical_transition_rows'] == 1296
    assert result['optimizer_steps'] == result['chunks_published'] == 0
    assert not result['production_qualified'] and not result['H200_renderer_tested']
    assert result['observation_sources'] == {'raytraced': 0, 'generated_controls': 1296}
    assert [set(row) for row in batch.bindings] == [set(range(16)), {0}, {1}]
    assert all(k.run_id == probe.RUN_ID for row in batch.captures for k in row)
    assert batch.captures[-1][2].episode_step_index == 80
    assert len(events) == 81 * 3


def test_actual_batch_size_rejected_before_lifecycle():
    value, _, _ = fixture()
    with pytest.raises(ValueError, match='not B16'):
        probe.run_probe_steps(Batch(), None, value, report=lambda _: None)


def test_teacher_tail_corruption_rejected_before_native_step():
    batch = GeneratedBatch(bad_tail=True)
    with pytest.raises(RuntimeError, match='teacher/Homie'):
        execute_steps(batch)
    assert not batch.steps


def test_uncertain_native_step_not_retried_and_histories_cleaned():
    value, _, _ = fixture()
    batch, calls, events = GeneratedBatch(fail_step=True), [], []
    life = SimpleNamespace(begin=lambda: calls.append('begin'), finish=lambda: calls.append('finish'))
    with pytest.raises(RuntimeError, match='uncertain native'):
        probe.run_probe_steps(batch, life, value, report=events.append)
    assert len(batch.steps) == 1 and calls == ['begin', 'finish']
    assert events[-1]['phase'] == 'env_step_start'


def test_native_rgb_cannot_get_generated_control_completion_receipt():
    batch = Batch(); batch.num_envs = 16
    with pytest.raises(RuntimeError, match='generated-control'):
        execute_steps(batch)


def model_trainer():
    def model():
        value = torch.nn.Linear(2, 1)
        value.register_buffer('scalar_count', torch.tensor(0))
        return value
    return SimpleNamespace(policy_model=model(), ref_model=model(), unwrapped_model=SimpleNamespace(
        homie_walk_model=model(), homie_stand_model=model()))


def test_model_fingerprints_cover_parameters_and_normalization_state():
    trainer = model_trainer()
    before = probe.parameter_fingerprints(trainer)
    assert set(before) == {'student', 'teacher', 'homie_walk', 'homie_stand'}
    assert before == probe.parameter_fingerprints(trainer)
    trainer.policy_model.scalar_count.add_(1)
    after = probe.parameter_fingerprints(trainer)
    assert before['student'] != after['student'] and before['teacher'] == after['teacher']
    with torch.no_grad():
        trainer.ref_model.weight.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        probe.parameter_fingerprints(trainer)


def test_rollout_modes_cannot_update_normalization_statistics():
    trainer = model_trainer()
    with pytest.raises(RuntimeError, match='rollout modes'):
        probe.require_rollout_inference_modes(trainer)
    models = [trainer.policy_model, trainer.ref_model,
              trainer.unwrapped_model.homie_walk_model, trainer.unwrapped_model.homie_stand_model]
    for model in models: model.eval()
    probe.require_rollout_inference_modes(trainer)
    trainer.ref_model.train()
    with pytest.raises(RuntimeError, match='rollout modes'):
        probe.require_rollout_inference_modes(trainer)


@pytest.mark.parametrize('failure', [None, 'step', 'optimizer', 'weights', 'reset', 'training_mode'])
def test_native_flow_isolated_phase_preserves_models_optimizer_and_kit(tmp_path, monkeypatch, failure):
    value, _, _ = fixture()
    events, optimizer_calls = [], []
    trainer = model_trainer()
    models = [trainer.policy_model, trainer.ref_model,
              trainer.unwrapped_model.homie_walk_model, trainer.unwrapped_model.homie_stand_model]
    trainer.optimizer = SimpleNamespace(step=lambda: optimizer_calls.append('forbidden'))
    original_step = trainer.optimizer.step
    trainer.env = SimpleNamespace(_rr_collection_initial_reset={'completed': True},
        simulator=SimpleNamespace(sim=SimpleNamespace(
            play=lambda: events.append('play'), pause=lambda: events.append('pause'))))
    trainer._native_resource_guard = lambda: events.append('guard')
    trainer._rollout_lifecycle = lambda: 'actual-lifecycle'
    backend = SimpleNamespace(trainer=trainer, kit_identity=1234, scene_initializations=1,
        source_sha256=value['source_sha256'], phase_resets=0, profile='explicit_evaluation',
        _kit_unchanged=lambda: events.append('same-kit'))
    def switch(profile, **kw):
        assert profile == 'training' and kw['assignment'] == value['assignment'] and kw['panel'] is None
        events.append('switch-training'); backend.profile = profile; backend.scene_initializations += 1
    backend._switch_scene = switch
    def prepare(seed):
        events.append('restore-training-state'); backend.phase_resets += 1
    backend._prepare_training = prepare
    @contextmanager
    def observe(*args):
        events.append('observer-enter')
        try:
            yield SimpleNamespace(episode_assignments=[{'lane': i} for i in range(16)],
                episode_resets=[0 if failure == 'reset' else 1]*16)
        finally:
            events.append('observer-exit')
    def initialize(t):
        assert t is trainer
        events.append('native-initialize-corrected-reset')
        for model in models: model.eval()
        if failure == 'training_mode': trainer.ref_model.train()
        return SimpleNamespace(obs_dict={'actual': True})
    monkeypatch.setitem(sys.modules, 'dagger_rtx.native_entrypoints', SimpleNamespace(
        NativeRavenRuntime=SimpleNamespace(initialize=initialize),
        build_origin_attested_entrypoints=lambda *a: events.append('attest-origins') or 'actual-entrypoints'))
    monkeypatch.setitem(sys.modules, 'dagger_rtx.native_raven', SimpleNamespace(
        build_native_raven_batch=lambda *a: events.append('native-batch') or 'actual-batch'))
    monkeypatch.setitem(sys.modules, 'dagger_rtx.observation_contract', SimpleNamespace(
        GENERATED_ONLY_CONTRACT='generated-only',
        configure_native_observations=lambda t, c: events.append(('observation-contract', c))))
    monkeypatch.setitem(sys.modules, 'native_motion_assignment', SimpleNamespace(
        install_native_assignment_observer=observe))
    monkeypatch.setitem(sys.modules, 'resident_isaac_trainer', SimpleNamespace(
        reset_rollout_for_scene=lambda _: events.append('boundary-history-clear'),
        seeded=lambda _: events.append('seed'), initial_state=lambda *a: {'robot': b'actual'},
        digest=lambda data: hashlib.sha256(data).hexdigest()))
    def step(batch, lifecycle, plan, *, report):
        assert batch == 'actual-batch' and lifecycle == 'actual-lifecycle' and plan == value
        events.append('step')
        report({'phase': 'env_step_start', 'completed_vector_steps': 0, 'irreversible_step_intents': 1})
        if failure == 'step': raise RuntimeError('uncertain physical step')
        if failure == 'optimizer': trainer.optimizer.step()
        if failure == 'weights': trainer.policy_model.scalar_count.add_(1)
        return {'fixture': 'actual loop covered in separate81step test'}
    monkeypatch.setattr(probe, 'run_probe_steps', step)
    reports = []
    if failure:
        with pytest.raises(RuntimeError):
            probe.run_native(backend, value, tmp_path, lambda *a, **kw: reports.append((a, kw)))
    else:
        result = probe.run_native(backend, value, tmp_path, lambda *a, **kw: reports.append((a, kw)))
        assert result['complete'] and result['diagnostic_only'] and result['kit_identity'] == 1234
        assert result['environment_step_calls'] == 81 and result['physical_transition_rows'] == 1296
        assert result['model_state_before'] == result['model_state_after']
        assert result['scene_rebuilds'] == result['phase_resets'] == 1
        assert not result['learner_integration_tested'] and not result['student_inference_tested']
        assert result['checkpoint_promotions'] == result['optimizer_steps'] == result['chunks_published'] == 0
        assert result['checkpoint_binding_scope'].startswith('context_only_')
        assert events[-1] == 'pause'
    assert trainer.optimizer.step is original_step and not optimizer_calls
    assert 'observer-exit' in events
    assert events.index('observer-enter') < events.index('restore-training-state')
    assert events.index('restore-training-state') < events.index('native-initialize-corrected-reset')
    assert events.count('step') == (0 if failure == 'training_mode' else 1)
    if failure != 'training_mode':
        progress = json.loads((tmp_path / 'progress.json').read_text())
        assert progress['completed'] == 0 and progress['irreversible_step_intents'] == 1


def backend_fixture(tmp_path, monkeypatch):
    value, state, inventory = fixture()
    path = tmp_path / 'plan.json'; path.write_bytes(probe.canonical(value))
    (tmp_path / 'resident-state.json').write_text(json.dumps(state))
    checked = []
    scope = methods('_training_phase_probe_inputs', 'validate_command_identity',
        os=os, json=json, Path=Path, file_identity=lambda *args: checked.append(args))
    backend = SimpleNamespace(runtime_root=tmp_path, source_sha256='b'*64,
        motion_inventory=inventory, _consumed_outputs=set(), artifact=lambda name: tmp_path / name)
    backend._training_phase_probe_inputs = lambda payload: scope['_training_phase_probe_inputs'](backend, payload)
    backend.validate_command_identity = lambda command: scope['validate_command_identity'](backend, command)
    for key, val in {'RESIDENT_PRODUCTION_FULL295': '1', 'RESIDENT_PIPELINE_MODE': 'production',
        'RESIDENT_SCENE_LOCK_FILE': '/runtime/scene.lock', 'RESIDENT_RESOURCE_POLICY':
        'roborender.native-memory-policy.v3', 'RESIDENT_PHYSICAL_GPU': '0'}.items():
        monkeypatch.setenv(key, val)
    payload = dict(output_name=value['command_id'], plan_file='plan.json',
                   plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return backend, payload, value, checked


def test_resident_input_admission_is_fresh_owner_guard_and_checkpoint_bound(tmp_path, monkeypatch):
    backend, payload, value, checked = backend_fixture(tmp_path, monkeypatch)
    assert backend._training_phase_probe_inputs(payload) == value
    assert checked == [(Path('/diagnostic-assets') / ('c'*64 + '.pt'), 'c'*64)]
    backend._consumed_outputs.add(payload['output_name'])
    with pytest.raises(ValueError, match='no replay'):
        backend._training_phase_probe_inputs(payload)


@pytest.mark.parametrize('key', ['RESIDENT_PRODUCTION_FULL295', 'RESIDENT_PIPELINE_MODE',
    'RESIDENT_SCENE_LOCK_FILE', 'RESIDENT_RESOURCE_POLICY'])
def test_missing_native_guard_cannot_admit_probe(tmp_path, monkeypatch, key):
    backend, payload, _, _ = backend_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(key)
    with pytest.raises(ValueError, match='guard'):
        backend._training_phase_probe_inputs(payload)


def test_owner_envelope_cannot_bypass_plan_identity(tmp_path, monkeypatch):
    backend, payload, value, _ = backend_fixture(tmp_path, monkeypatch)
    command = dict(operation='probe_training_phase', command_id=value['command_id'],
                   worker_instance_id=value['worker_instance_id'], payload=payload)
    backend.validate_command_identity(command)
    for key, changed in [('command_id', 'other'), ('worker_instance_id', 'f'*32)]:
        with pytest.raises(ValueError, match='envelope'):
            backend.validate_command_identity({**command, key: changed})


def test_resident_operation_handler_uses_new_probe_and_preserves_no_replay(tmp_path, monkeypatch):
    backend, payload, value, _ = backend_fixture(tmp_path, monkeypatch)
    calls, writes = [], []
    monkeypatch.setattr(probe, 'run_native', lambda *args: calls.append(args) or {
        'complete': True, 'diagnostic_only': True, 'optimizer_steps': 0})
    scope = methods('_run_training_phase_probe', 'validate', '_run_operation',
        atomic_write_json=lambda path, data, **kw: writes.append((path, data, kw)),
        attest_native_thp=lambda _: {'attested': True}, NATIVE_THP_CONFIGURATION='actual')
    backend._run_training_phase_probe = lambda p, r: scope['_run_training_phase_probe'](backend, p, r)
    scope['validate'](backend, 'probe_training_phase', payload)
    result = scope['_run_operation'](backend, 'probe_training_phase', payload, lambda *a, **kw: None)
    assert result['plan_sha256'] == payload['plan_sha256'] and result['optimizer_steps'] == 0
    assert len(calls) == len(writes) == 1
    assert writes[0][0].name == 'training-phase-probe-result.json' and writes[0][2]['mode'] == 0o644
    assert payload['output_name'] in backend._consumed_outputs
    with pytest.raises(ValueError, match='no replay'):
        scope['_run_operation'](backend, 'probe_training_phase', payload, lambda *a, **kw: None)
    assert len(calls) == 1


def test_probe_not_exempted_from_actual_production_phase_resource_guard(tmp_path, monkeypatch):
    calls = []
    class Guard:
        def __init__(self, path, **kw): calls.append(('guard', path)); self.summary = {'actual': True}
        def check(self): calls.append('check')
        def __enter__(self): return self
        def __exit__(self, *_): calls.append('exit')
    monkeypatch.setitem(sys.modules, 'dagger_rtx.phase_resource_guard', SimpleNamespace(
        PhaseResourceGuard=Guard, POLICY_VERSION='v3'))
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1')
    backend = SimpleNamespace(runtime_root=tmp_path, trainer=SimpleNamespace(_native_resource_guard='prior'),
        validate=lambda op, p: calls.append(('validate', op)))
    def run(op, p, report):
        backend.trainer._native_resource_guard()
        return {'complete': True}
    backend._run_operation = run
    scope = methods('_run_guarded', os=os, atomic_write_json=lambda *a, **kw: calls.append('write'))
    result = scope['_run_guarded'](backend, 'probe_training_phase', {'output_name': 'probe'}, None)
    assert result['resource_guard'] == {'actual': True}
    assert backend.trainer._native_resource_guard == 'prior' and 'check' in calls


def test_owner_dispatch_serializes_and_deduplicates_probe(tmp_path):
    calls = []
    backend = SimpleNamespace(validate=lambda *a: None,
        validate_command_identity=lambda value: calls.append('admit'),
        run=lambda *args: calls.append('run') or {'complete': True, 'diagnostic_only': True})
    owner = ResidentOwner(tmp_path, backend, gpu_uuid='GPU-test', source_sha256='b'*64)
    command = dict(schema=OWNER_SCHEMA, operation='probe_training_phase', command_id='probe-one',
        worker_instance_id=owner.state['worker_instance_id'], payload={'output_name': 'probe-one'})
    assert validate_command(command) == command
    accepted = owner.submit(command)
    assert owner.submit(command) == accepted
    owner.run_next()
    assert owner.submit(command) == accepted and calls == ['admit', 'run']
    assert owner.state['native_initializations'] == 1 and owner.state['phase'] == 'IDLE'


def test_uncertain_probe_blocks_later_native_work(tmp_path):
    calls = []
    def fail(*args): calls.append('run'); raise RuntimeError('native uncertainty')
    owner = ResidentOwner(tmp_path, SimpleNamespace(validate=lambda *a: None, run=fail),
        gpu_uuid='GPU-test', source_sha256='b'*64)
    command = dict(schema=OWNER_SCHEMA, operation='probe_training_phase', command_id='probe-one',
        worker_instance_id=owner.state['worker_instance_id'], payload={})
    owner.submit(command)
    with pytest.raises(RuntimeError, match='uncertainty'): owner.run_next()
    with pytest.raises(ValueError, match='uncertain'): owner.submit({**command, 'command_id': 'probe-two'})
    assert calls == ['run'] and owner.state['phase'] == 'CRASH_UNCERTAIN'
