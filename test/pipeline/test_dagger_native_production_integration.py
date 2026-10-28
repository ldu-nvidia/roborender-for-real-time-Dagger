"""Execute extracted native-boundary methods with CPU dependencies only.

No Isaac, Kit, model, CUDA, renderer or environment-step implementation imports.
These tests qualify dispatch/order/identity boundaries, not native execution.
"""
from test.deployment.source_paths import REPO_ROOT
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

import pytest

from test.pipeline.test_dagger_coverage_production import production
from test.pipeline.test_dagger_production_segment import inputs as legacy_inputs, segment
from pipeline.bc_training.handoff_common import publish
from dagger_rtx.beta_round import read_config, acknowledgement, gpu_uuids
from dagger_rtx.durable import atomic_write_json

ROOT = REPO_ROOT
RESIDENT = ROOT / 'pipeline/deployment/rtx_actor/resident_isaac_trainer.py'
NATIVE = ROOT / 'pipeline/dagger/rtx/code/src/dagger_rtx/native_service_trainer.py'


@pytest.fixture
def inputs(legacy_inputs):
    """Keep legacy lifecycle fixtures, with the current generated-only final test."""
    plan, ref, config, mse = legacy_inputs
    assert config['validation']['observation_mode'] == 'generated'
    mse = dict(mse, final_test=dict(mse['final_test'], observation_mode='generated'))
    return plan, ref, config, mse


def extracted(path, methods, *, klass=None, namespace=None):
    tree = ast.parse(path.read_text())
    body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == klass) if klass else tree.body
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(nodes) == len(methods)
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def staged(inputs, tmp_path, monkeypatch, *, gpu=0):
    plan, ref, config, mse = inputs
    jobs = tmp_path / 'jobs'; jobs.mkdir()
    config_ref = publish(jobs / 'round.json', config)
    mse_ref = publish(jobs / 'mse.json', mse)
    # Keep the actual raw plan SHA, including publication's trailing newline.
    (jobs / 'coverage.json').write_bytes(Path(ref['path']).read_bytes())
    item = segment(inputs, epoch=2, index=3)
    item['mse_protocol_file_sha256'] = mse_ref['raw_sha256']
    cohort = plan['rounds'][0]['epochs'][1]['cohorts'][3]
    for g, assignment in enumerate(cohort['assignments']):
        raw = json.dumps(assignment, sort_keys=True, separators=(',', ':')).encode()
        (jobs / f'assignment-g{g}.json').write_bytes(raw)
        item['motion_assignments'][g] = dict(physical_gpu_index=g, path=str(jobs / f'assignment-g{g}.json'),
                                            sha256=hashlib.sha256(raw).hexdigest())
    segment_ref = publish(jobs / 'segment.json', item)
    (jobs / 'token').write_bytes(b'cpu-test-token'); (jobs / 'token').chmod(0o600)
    (jobs / 'connection').mkdir()
    payload = dict(round_config='round.json', round_config_sha256=config_ref['raw_sha256'],
        mse_protocol_file='mse.json', mse_protocol_sha256=mse_ref['raw_sha256'],
        segment_file='segment.json', segment_sha256=segment_ref['raw_sha256'],
        coverage_plan_file='coverage.json', coverage_plan_sha256=ref['raw_sha256'],
        motion_assignment_file=f'assignment-g{gpu}.json', motion_assignment_sha256=item['motion_assignments'][gpu]['sha256'],
        token_file='token', connection_status_directory='connection', connection_config_sha256='f' * 64,
        output_name=f'production-segment-g{gpu}')
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', str(gpu))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', config['rtx_actors'][gpu]['gpu_uuid'])
    ns = extracted(RESIDENT, ['file_identity'], namespace=dict(Path=Path, hashlib=hashlib))
    ns.update(Path=Path, json=json, os=os, time=time, read_config=read_config, nullcontext=nullcontext,
              acknowledgement=acknowledgement, gpu_uuids=gpu_uuids, atomic_write_json=atomic_write_json)
    ns = extracted(RESIDENT, ['artifact', 'validate', '_population_scope', '_production_segment_inputs', '_run_production_segment', 'run', '_run_guarded'],
                   klass='ResidentRavenBackend', namespace=ns)
    cls = type('CPUResidentBoundary', (), {name: ns[name] for name in (
        'artifact', 'validate', '_population_scope', '_production_segment_inputs', '_run_production_segment', 'run', '_run_guarded')})
    backend = cls(); backend.jobs_root = jobs; backend.runtime_root = tmp_path / 'native-runtime'
    backend.runtime_root.mkdir(); backend.source_sha256 = config['source_sha256']
    backend.motion_inventory_sha256 = cohort['assignments'][gpu]['motion_inventory_sha256']
    backend._consumed_outputs = set(); backend.active_output = None
    backend.profile='training';backend.scene_initializations=1
    backend.kit=SimpleNamespace(update=lambda:None)
    backend._kit_unchanged=lambda:None
    return backend, payload, config, item, ns


@pytest.mark.parametrize('gpu', [0, 1])
def test_production_boundary_binds_each_physical_gpu_own_B2_assignment(inputs, tmp_path, monkeypatch, gpu):
    backend, payload, config, item, _ = staged(inputs, tmp_path, monkeypatch, gpu=gpu)
    backend.validate('collect_production_segment', payload)
    actual_config, actual_segment, assignment = backend._production_segment_inputs(payload)
    assert actual_config == config and actual_segment == item
    assert len(assignment['lanes']) == 2 and assignment['purpose'] == 'training'
    assert assignment['simulator_seed'] != config['seed']
    assert actual_segment['start_iteration_index'] == 154


@pytest.mark.parametrize('fault', ['coverage_sha', 'mse_sha', 'assignment_sha', 'source', 'gpu', 'replay'])
def test_wrong_bound_artifacts_rejected_before_native_operation(inputs, tmp_path, monkeypatch, fault):
    backend, payload, config, item, _ = staged(inputs, tmp_path, monkeypatch)
    if fault == 'coverage_sha': payload['coverage_plan_sha256'] = '0' * 64
    if fault == 'mse_sha': payload['mse_protocol_sha256'] = '0' * 64
    if fault == 'assignment_sha': payload['motion_assignment_sha256'] = '0' * 64
    if fault == 'source': backend.source_sha256 = '0' * 64
    if fault == 'gpu': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', config['rtx_actors'][1]['gpu_uuid'])
    if fault == 'replay': backend._consumed_outputs.add(payload['output_name'])
    with pytest.raises(ValueError): backend.validate('collect_production_segment', payload)
    assert not list(backend.runtime_root.iterdir())


def native_prefix_namespace(monkeypatch):
    def required(key):
        if not os.environ.get(key): raise RuntimeError('missing ' + key)
        return os.environ[key]
    class NativeBoundaryReached(Exception): pass
    calls = []
    def initialize(trainer):
        # The legacy lifecycle config has no training observation override;
        # rendered final testing must not silently relabel its training source.
        assert trainer.vg_raytraced_mix_prob == .5
        assert trainer._roborender_student_observation_contract is None
        calls.append('first-real-reset-boundary')
        raise NativeBoundaryReached()
    namespace = dict(__package__='dagger_rtx', Path=Path, os=os, json=json, hashlib=hashlib,
        SERVICE_MODES={'beta-production'}, _required_env=required,
        verify_assets=lambda *a, **kw: ({}, []),
        NativeRavenRuntime=SimpleNamespace(initialize=initialize))
    ns = extracted(NATIVE, ['train'], klass='NativeDaggerRTXServiceTrainer', namespace=namespace)
    return ns['train'], NativeBoundaryReached, calls


def arm_native_environment(backend, payload, monkeypatch):
    for key in list(os.environ):
        if key.startswith('DAGGER_'): monkeypatch.delenv(key)
    values = dict(DAGGER_RTX_MODE='beta-production', DAGGER_BETA_CONFIG=str(backend.artifact(payload['round_config'])),
        DAGGER_BETA_CONFIG_SHA256=payload['round_config_sha256'],
        DAGGER_MSE_PROTOCOL_FILE=str(backend.artifact(payload['mse_protocol_file'])),
        DAGGER_MSE_PROTOCOL_SHA256=payload['mse_protocol_sha256'],
        DAGGER_PRODUCTION_SEGMENT_FILE=str(backend.artifact(payload['segment_file'])),
        DAGGER_PRODUCTION_SEGMENT_SHA256=payload['segment_sha256'],
        DAGGER_PRODUCTION_COVERAGE_PLAN_FILE=str(backend.artifact(payload['coverage_plan_file'])),
        DAGGER_RUNTIME_ROOT=str(backend.runtime_root / 'first-reset-preflight'), DAGGER_ASSET_MANIFEST='/not-opened/manifest.json')
    for key, value in values.items(): monkeypatch.setenv(key, value)


def test_native_trainer_checks_segment_mse_bytes_before_first_real_reset(inputs, tmp_path, monkeypatch):
    backend, payload, _, _, _ = staged(inputs, tmp_path, monkeypatch)
    arm_native_environment(backend, payload, monkeypatch)
    train, stop, calls = native_prefix_namespace(monkeypatch)
    trainer = SimpleNamespace(env=SimpleNamespace(num_envs=2), vg_raytraced_mix_prob=.5,
        _native_episode_assignment_observer=lambda *a: None, _native_resource_guard=lambda: None)
    monkeypatch.setenv('DAGGER_PRODUCTION_SEGMENT_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='bytes changed'): train(trainer)
    assert not calls
    assert trainer.vg_raytraced_mix_prob == .5
    monkeypatch.setenv('DAGGER_PRODUCTION_SEGMENT_SHA256', payload['segment_sha256'])
    with pytest.raises(stop): train(trainer)
    assert calls == ['first-real-reset-boundary']


def test_native_trainer_missing_production_guard_fails_before_reset(inputs, tmp_path, monkeypatch):
    backend, payload, _, _, _ = staged(inputs, tmp_path, monkeypatch)
    arm_native_environment(backend, payload, monkeypatch)
    train, _, calls = native_prefix_namespace(monkeypatch)
    trainer = SimpleNamespace(env=SimpleNamespace(num_envs=2), _native_episode_assignment_observer=lambda *a: None)
    with pytest.raises(RuntimeError, match='guard|observer'): train(trainer)
    assert not calls


def fake_guard(monkeypatch, events):
    class Guard:
        summary = {'passed': True, 'samples': 2, 'minimum_host_available_bytes': 12 << 30}
        def __init__(self, path, *, policy_version):
            assert policy_version == os.environ.get('RESIDENT_RESOURCE_POLICY','roborender.native-memory-policy.v2')
            self.path = path
        def __enter__(self): events.append('guard-enter'); return self
        def check(self): events.append('guard-check')
        def __exit__(self, *args): events.append('guard-exit')
    monkeypatch.setitem(sys.modules, 'dagger_rtx.phase_resource_guard', SimpleNamespace(
        PhaseResourceGuard=Guard,POLICY_VERSION='roborender.native-memory-policy.v2'))
    return Guard


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('policy_version', ['roborender.native-memory-policy.v2','roborender.native-memory-policy.v3'])
def test_production_phase_observer_guard_environment_order_and_cleanup(inputs, tmp_path, monkeypatch, fail, policy_version):
    monkeypatch.setenv('RESIDENT_RESOURCE_POLICY',policy_version)
    backend, payload, config, item, ns = staged(inputs, tmp_path, monkeypatch)
    events = []; fake_guard(monkeypatch, events)
    observer_active = [False]
    @contextmanager
    def observer(trainer, assignment, output, *, episode_namespace, motion_inventory=None, lane_offset=0):
        assert motion_inventory is None and lane_offset == 0  # Legacy B2 fixture, not full295 protocol.
        events.append('observer-installed'); observer_active[0] = True
        assert episode_namespace == item['episode_namespace']
        try: yield SimpleNamespace(episode_assignments=[{'episode_id': episode_namespace + '-episode-000-00000000'}])
        finally: observer_active[0] = False; events.append('observer-restored')
    import native_motion_assignment
    monkeypatch.setattr(native_motion_assignment, 'install_native_assignment_observer', observer)
    phase_ns = extracted(RESIDENT, ['phase_environment'], namespace=dict(os=os, contextmanager=contextmanager))
    ns['phase_environment'] = phase_ns['phase_environment']
    ns['seeded'] = lambda seed: events.append('seeded')
    ns['reset_rollout_for_scene'] = lambda model: events.append('rollout-reset')
    old_guard = lambda: None
    backend.trainer = SimpleNamespace(_native_resource_guard=old_guard, policy_model=object(), ref_model=object(),
        env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(
            play=lambda: events.append('sim-play'), pause=lambda: events.append('sim-pause')))))
    backend._switch_scene = lambda *a, **kw: events.append('scene-bound-B2')
    backend._kit_unchanged = lambda: events.append('same-Kit')
    backend.kit=SimpleNamespace(update=lambda:None)
    backend.scene_initializations = 2; backend.phase_resets = 0; backend.profile = 'training'
    def prepare(seed):
        assert observer_active[0] and callable(backend.trainer._native_resource_guard)
        events.append('prepare-training-with-observer'); backend.phase_resets += 1
    backend._prepare_training = prepare
    def train(trainer):
        assert observer_active[0]
        assert os.environ['DAGGER_WINDOW_INDEX'] == '154'
        assert os.environ['DAGGER_PRODUCTION_SEGMENT_SHA256'] == payload['segment_sha256']
        assert os.environ['DAGGER_MSE_PROTOCOL_SHA256'] == payload['mse_protocol_sha256']
        assert os.environ['DAGGER_HELDOUT_COLLECTION_FILE'] == ''
        trainer._native_resource_guard(); events.append('native-driver')
        if fail: raise RuntimeError('injected-native-boundary-failure')
        output = Path(os.environ['DAGGER_RUNTIME_ROOT'])
        (output / 'production-segment-state.json').write_text(json.dumps(dict(phase='COMPLETE', completed_windows=2,
            transition_batches=162, physical_transition_rows=324, history_reset_at_window_boundary=False)))
        return {'result': 'PASS'}
    ns['NativeDaggerRTXServiceTrainer'] = SimpleNamespace(train=train)
    monkeypatch.setenv('DAGGER_WINDOW_INDEX', 'old-global-offset')
    monkeypatch.setenv('DAGGER_PRODUCTION_SEGMENT_SHA256', 'old-segment-hash')
    if fail:
        with pytest.raises(RuntimeError, match='injected-native'): backend._run_production_segment(payload, lambda *a, **kw: None)
    else:
        result = backend._run_production_segment(payload, lambda *a, **kw: None)
        assert result['complete'] and result['actor_state']['completed_windows'] == 2
        assert result['resource_guard']['passed'] is True
    assert events.index('observer-installed') < events.index('prepare-training-with-observer') < events.index('native-driver')
    assert backend.trainer._native_resource_guard is old_guard and not observer_active[0]
    assert os.environ['DAGGER_WINDOW_INDEX'] == 'old-global-offset'
    assert os.environ['DAGGER_PRODUCTION_SEGMENT_SHA256'] == 'old-segment-hash'


@pytest.mark.parametrize('enabled,operation,wrapped', [
    (False, 'collect_mse_validation', False), (True, 'collect_mse_validation', True),
    (True, 'evaluate_explicit_panel', True), (True, 'shutdown', False),
    (True, 'collect_production_segment', False), (True, 'capture_initial_b2', True)])
def test_production_only_generic_phase_guard_is_opt_in(inputs, tmp_path, monkeypatch, enabled, operation, wrapped):
    backend, _, _, _, _ = staged(inputs, tmp_path, monkeypatch)
    events = []; fake_guard(monkeypatch, events)
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1' if enabled else '0')
    old = lambda: None; backend.trainer = SimpleNamespace(_native_resource_guard=old)
    backend.validate = lambda *a: events.append('validated')
    def underlying(*args):
        events.append('actual-phase')
        return {'complete': True}
    backend._run_operation = underlying
    result = backend.run(operation, {'output_name': 'phase'}, lambda *a, **kw: None)
    assert ('guard-enter' in events) is wrapped
    assert ('resource_guard' in result) is wrapped
    assert backend.trainer._native_resource_guard is old


def test_production_generic_guard_restores_callback_on_phase_failure(inputs, tmp_path, monkeypatch):
    backend, _, _, _, _ = staged(inputs, tmp_path, monkeypatch)
    events = []; fake_guard(monkeypatch, events)
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1')
    old = lambda: None; backend.trainer = SimpleNamespace(_native_resource_guard=old)
    backend.validate = lambda *a: None
    def fail(*args): raise RuntimeError('injected-heldout-phase-failure')
    backend._run_operation = fail
    with pytest.raises(RuntimeError, match='injected-heldout'):
        backend.run('collect_mse_validation', {'output_name': 'heldout'}, lambda *a, **kw: None)
    assert backend.trainer._native_resource_guard is old
    assert events == ['guard-enter', 'guard-exit']


def test_native_request_guard_checks_resources_before_transport_or_capture():
    tree = ast.parse(NATIVE.read_text())
    nested = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'guarded_native_request')
    calls = []
    ns = dict(resource_guard=lambda: calls.append('resource'), connection_guard=lambda: calls.append('connection'))
    exec(compile(ast.Module(body=[nested], type_ignores=[]), 'native-resource-request-guard', 'exec'), ns)
    ns['guarded_native_request']()
    assert calls == ['resource', 'connection']
    calls.clear()
    def failed_resource(): raise RuntimeError('latched-memory-error')
    ns['resource_guard'] = failed_resource
    with pytest.raises(RuntimeError, match='latched-memory'): ns['guarded_native_request']()
    assert not calls


def test_population_journals_do_not_write_to_readonly_jobs_mount():
    source = RESIDENT.read_text()
    assert "shared_root=self.artifact(payload['population_scope_file']).parent" not in source
    assert "RESIDENT_POPULATION_BARRIERS" in source


@pytest.mark.parametrize('enabled,batch,profile,allowed', [
    (False, 2, 'training', False), (True, 6, 'training', False),
    (True, 2, 'evaluation', False), (True, 2, 'training', True)])
def test_initial_capture_requires_explicit_production_training_B2(inputs, tmp_path, monkeypatch,
                                                                enabled, batch, profile, allowed):
    backend, _, _, _, _ = staged(inputs, tmp_path, monkeypatch)
    backend.trainer = SimpleNamespace(env=SimpleNamespace(num_envs=batch))
    backend.profile = profile
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1' if enabled else '0')
    payload = {'output_name': 'initial-b2'}
    if allowed:
        backend.validate('capture_initial_b2', payload)
        backend._consumed_outputs.add(payload['output_name'])
        with pytest.raises(ValueError, match='cannot be replayed'):
            backend.validate('capture_initial_b2', payload)
    else:
        with pytest.raises(ValueError, match='production-only'):
            backend.validate('capture_initial_b2', payload)
    assert not list(backend.runtime_root.iterdir())


def test_initial_B2_capture_uses_resident_scene_and_guard_without_step_or_optimizer(inputs, tmp_path, monkeypatch):
    backend, _, _, _, ns = staged(inputs, tmp_path, monkeypatch)
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1')
    events = []; fake_guard(monkeypatch, events)
    backend.scene_seed = 42; backend.scene_initializations = 1
    backend.phase_resets = 0; backend.profile = 'training'
    old_guard = lambda: None
    sim = SimpleNamespace(play=lambda: events.append('play'), pause=lambda: events.append('pause'))
    def forbidden(*a, **kw): raise AssertionError('capture must not step or optimize')
    def capture(batch, runtime, path):
        backend.trainer._native_resource_guard()
        events.append('capture')
        atomic_write_json(path, {'cpu_boundary_test_only': True, 'environment_step_calls': 0})
    backend.trainer = SimpleNamespace(_native_resource_guard=old_guard,
        policy_model=object(), ref_model=object(), optimizer=SimpleNamespace(step=forbidden),
        env=SimpleNamespace(num_envs=2, step=forbidden, simulator=SimpleNamespace(sim=sim)), _capture_only=capture)
    backend._prepare_training = lambda seed: events.append('prepare')
    backend._kit_unchanged = lambda: events.append('same-Kit')
    def initialize(trainer):
        assert callable(trainer._native_resource_guard)
        events.append('initialize-boundary')
        return object()
    def inspect_scene(env,*,flush_app,inspect_only):
        assert inspect_only is True and env is backend.trainer.env
        events.append('inspect-scene-only')
        return {'existing_render_products':['fixture-product'],'camera_sensors':['fixture-camera']}
    thp_configuration = object()
    thp_receipt = {'cpu_boundary_test_only': True, 'attested': True}
    def attest_thp(configuration):
        assert configuration is thp_configuration
        events.append('thp-attested')
        return thp_receipt
    namespace = dict(ns, reset_rollout_for_scene=lambda model: events.append('rollout-reset'),
        release_native_scene=inspect_scene,
        attest_native_thp=attest_thp, NATIVE_THP_CONFIGURATION=thp_configuration,
        NativeRavenRuntime=SimpleNamespace(initialize=initialize),
        build_origin_attested_entrypoints=lambda *a: object(), build_native_raven_batch=lambda *a: object())
    operation = extracted(RESIDENT, ['_run_initial_b2_capture'], klass='ResidentRavenBackend', namespace=namespace)
    backend._run_initial_b2_capture = MethodType(operation['_run_initial_b2_capture'], backend)
    backend._run_operation = lambda operation, payload, report: backend._run_initial_b2_capture(payload, report)
    result = backend.run('capture_initial_b2', {'output_name': 'initial-b2'}, lambda *a, **kw: None)
    assert result['complete'] and result['num_envs'] == 2
    assert result['environment_step_calls'] == result['optimizer_steps'] == 0
    assert result['scene_initializations'] == result['native_initializations'] == 1
    assert result['resource_guard']['passed'] is True
    assert result['native_thp_policy'] == thp_receipt and events.count('thp-attested') == 1
    assert events.index('guard-enter') < events.index('initialize-boundary') < events.index('capture')
    assert events.index('pause') < events.index('thp-attested') < events.index('guard-exit')
    assert backend.trainer._native_resource_guard is old_guard and backend.active_output is None
