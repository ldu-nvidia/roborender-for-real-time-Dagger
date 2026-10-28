"""Synthetic CPU protocol/native-routing tests, never live qualification."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from types import MethodType

import pytest
import torch

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import scaled_success_contract as scaled
import explicit_motion_panel as explicit
import initial_robot_refresh_diagnostic as refresh
import rendered_student_success_contract as rendered
import rendered_success_observations as observations
from native_artifact_io import publish_json
from test.deployment.rtx_actor.test_production_success_contract import accepted
from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config, cohort_result
from test.deployment.rtx_actor.test_rendered_endpoint_pool import pool_config, Client
from rendered_endpoint_pool import RenderedEndpointPool, TIMING_SCHEMA
from test.deployment.rtx_actor.test_rendered_student_native_routing import evaluator_function


def config_for(accepted, panel, *, sharded=False, renderer_count=None):
    config = rendering_config(accepted)
    if sharded:
        return rendered.make_sharded_config(config, panel, config['checkpoint_sha256'],
            pool_config(8 if renderer_count is None else renderer_count, config['renderer']['identity'])['endpoints'])
    config.update(schema=rendered.SCALED_CONFIG_SCHEMA, panel_sha256=explicit.panel_sha256(panel),
                  renderer=pool_config(4 if renderer_count is None else renderer_count, config['renderer']['identity']))
    return rendered.validate_config(config, panel)


def scaled_cohort(accepted, panel, config, index=0, arm='bc'):
    spec = panel['cohorts'][index]
    count = len(spec['case_ids'])
    row = cohort_result(accepted)
    row.update(arm=arm, completed=count, expected=count, original_raven_metric_completed=count,
        original_raven_metric_successes=count, native_successes=count, exported_successes=count,
        checkpoint_sha256=config['checkpoint_sha256'], seed=spec['assignment']['simulator_seed'],
        motion_assignment_sha256=scaled.digest(spec['assignment']), vector_steps=1,
        outcomes=[dict(lane=i, steps=1, native_success=True, exported_success=True, timeout=False) for i in range(count)])
    mode = refresh.SCALED_STUDENT_CLEAR_ONCE
    env = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count, dtype=torch.bool))
    row.update({refresh.FIELD: mode, 'initial_robot_refresh_control': refresh.control_contract(mode),
        'initial_robot_refresh_attestation': refresh.clear_already_applied_initial_refresh(
            env, mode=mode, arm=arm, teacher_action_mode='distill')})
    receipt = row['rendered_observation_receipt']
    receipt.update(vector_steps_rendered=1, render_requests_completed=count,
        evaluation_binding=rendered.observation_binding(panel, config, index),
        anchor_seed=config['anchors']['seed'] + index,
        anchor_selections=[dict(lane=i, selected_at_vector_step=0,
            file_sha256=config['anchors']['files'][i]['sha256'], frame_index=0, prompt_sha256='f'*64) for i in range(count)])
    first = deepcopy(receipt['bounded_first_inputs'][0]['lanes'][0])
    receipt['bounded_first_inputs'] = [dict(step=0, lanes=[dict(deepcopy(first), lane=i) for i in range(count)])]
    from rendered_endpoint_pool import expected_physical_gpu, endpoint_slot_for_lane
    physical_gpu = expected_physical_gpu(config['renderer'], index)
    clients = [Client(i, config['renderer'], endpoint_slot=endpoint_slot_for_lane(config['renderer'], i,
        cohort_index=index, physical_gpu=physical_gpu), channel_namespace=(physical_gpu or 0)*1000) for i in range(count)]
    pool = RenderedEndpointPool(clients, config['renderer'], cohort_index=index, physical_gpu=physical_gpu)
    try:
        pool.render([{} for _ in range(count)], 0)
        receipt['renderer_endpoint_pool'] = pool.receipt()
        receipt['renderer_transport_receipts'] = [client.transport_receipt() for client in pool.clients]
    finally:
        pool.close()
    receipt['phase_timing'] = dict(schema=TIMING_SCHEMA, vector_steps_completed=1,
        preprocessing_seconds=0., render_wait_seconds=0., postprocessing_seconds=0.,
        steps=[dict(step=0, preprocessing_seconds=0., render_wait_seconds=0., postprocessing_seconds=0.)])
    row['scaled_initial_state_attestation'] = dict(schema=scaled.INITIAL_SCHEMA,
        environment_step_calls_before_attestation=0,
        initialization_distribution_sha256=scaled.digest(panel['scaled_protocol']),
        native_reset=dict(schema='roborender.explicit-motion-success-reset.v1', assignment=deepcopy(spec['assignment']),
            assignment_sha256=scaled.digest(spec['assignment']), sample_indices_forced=False,
            seed_reapplied_per_episode=False, lanes=[dict(lane, sample_idx=5+i, sample_idx_in_seq=2,
                reset_rng_sha256='e'*64) for i, lane in enumerate(spec['assignment']['lanes'])]),
        initial_state={name: dict(dtype='float32', shape=[count, 3], sha256='f'*64) for name in scaled.INITIAL_ARRAYS})
    return row


def partition_result(accepted, panel, config, index=0, arm='bc'):
    value = explicit.run_partition(panel, explicit.make_partition(panel, [index]),
        execute_cohort=lambda spec, i: scaled_cohort(accepted, panel, config, i, arm),
        checkpoint_sha256=config['checkpoint_sha256'], arm=arm, **scaled.evaluation_options(panel))
    return scaled.wrap_partition(panel, value, config['checkpoint_sha256'], config)


@pytest.mark.parametrize('batch,expected', [(4, [4]*73+[3]), (8, [8]*36+[7]), (16, [16]*18+[7])])
def test_all295_motion_support_new_initialization_is_explicit(accepted, batch, expected):
    panel = scaled.build_panel(accepted, batch_size=batch)
    assert [len(row['case_ids']) for row in panel['cohorts']] == expected
    source = [lane['motion_id'] for row in accepted['cohorts'] for lane in row['assignment']['lanes']]
    assert [lane['motion_id'] for row in panel['cohorts'] for lane in row['assignment']['lanes']] == source
    assert len({case for row in panel['cohorts'] for case in row['case_ids']}) == 295
    assert not scaled.protocol_identity(panel)['historical_fixed295_equivalent']
    assert not panel['scaled_protocol']['historical_outcomes_reused']
    with pytest.raises(ValueError):
        __import__('production_success_contract').require_panel(panel)
    qualification = scaled.build_panel(accepted, batch_size=batch, qualification=True)
    assert [len(c['case_ids']) for c in qualification['cohorts']] == [batch, batch]
    assert {c['assignment']['simulator_seed'] for c in qualification['cohorts']}.isdisjoint(
        c['assignment']['simulator_seed'] for c in panel['cohorts'])


@pytest.mark.parametrize('fault', ['seed', 'motion', 'case', 'order', 'count', 'source', 'distribution'])
def test_modified_panel_cannot_silently_relabel_fixed295(accepted, fault):
    panel = scaled.build_panel(accepted)
    if fault == 'seed': panel['cohorts'][0]['assignment']['simulator_seed'] += 1
    elif fault == 'motion': panel['cohorts'][0]['assignment']['lanes'][0]['sha256'] = 'e'*64
    elif fault == 'case': panel['cohorts'][0]['case_ids'][0] = 'old-case'
    elif fault == 'order': panel['cohorts'].reverse()
    elif fault == 'count': panel['episode_count'] = 294
    elif fault == 'source': panel['scaled_protocol']['source_panel']['cohorts'][0]['case_ids'][0] = 'wrong'
    else: panel['scaled_protocol']['historical_initial_states_preserved'] = True
    with pytest.raises(ValueError): scaled.require_panel(panel)


@pytest.mark.parametrize('batch', [4, 8, 16])
def test_scaled_reset_attestation_is_distinct_old_b2_guards_unchanged(batch):
    kwargs = dict(arm='bc', teacher_action_mode='distill', num_envs=batch)
    assert refresh.require_corrected_execution(refresh.SCALED_STUDENT_CLEAR_ONCE, **kwargs)
    for old in (refresh.STUDENT_CLEAR_ONCE, refresh.SMOKE_STUDENT_CLEAR_ONCE):
        with pytest.raises(ValueError): refresh.require_corrected_execution(old, **kwargs)
    with pytest.raises(ValueError):
        refresh.require_corrected_execution(refresh.SCALED_STUDENT_CLEAR_ONCE, has_resume=True, **kwargs)
    kwargs['arm'] = 'teacher'
    with pytest.raises(ValueError): refresh.require_corrected_execution(refresh.SCALED_STUDENT_CLEAR_ONCE, **kwargs)


@pytest.mark.parametrize('arm', ['bc', 'dagger'])
def test_completed_partition_and_full_qualification_aggregate_are_truthful(accepted, arm):
    panel = scaled.build_panel(accepted, qualification=True)
    config = config_for(accepted, panel)
    config['checkpoint_sha256'] = 'c'*64 if arm == 'dagger' else config['checkpoint_sha256']
    parts = [partition_result(accepted, panel, config, index, arm) for index in range(2)]
    assert all(not part['full_panel_complete'] for part in parts)
    value = scaled.aggregate_results(panel, parts, config['checkpoint_sha256'], config, arm=arm)
    assert value['completed_episodes'] == value['successes'] == 32
    assert value['qualification_only'] and value['arm'] == arm
    for bad in (parts[:1], parts + parts[:1], [parts[0], parts[0]]):
        with pytest.raises(ValueError): scaled.aggregate_results(panel, bad, config['checkpoint_sha256'], config, arm=arm)


def test_full295_requires_exactly19_committed_partitions_including_real_b7_tail(accepted):
    panel = scaled.build_panel(accepted, batch_size=16)
    config = config_for(accepted, panel)
    parts = [partition_result(accepted, panel, config, index) for index in range(19)]
    assert parts[-1]['partition_result']['completed_episodes'] == 7
    value = scaled.aggregate_results(panel, parts, config['checkpoint_sha256'], config)
    assert value['completed_episodes'] == 295 and not value['qualification_only']
    assert len({row['case_id'] for row in value['outcomes']}) == 295
    assert [row['motion_id'] for row in value['outcomes']] == [
        lane['motion_id'] for cohort in accepted['cohorts'] for lane in cohort['assignment']['lanes']]
    with pytest.raises(ValueError):
        scaled.aggregate_results(panel, parts[:-1], config['checkpoint_sha256'], config)


@pytest.mark.parametrize('fault', ['fallback', 'reset', 'initial', 'action', 'checkpoint', 'goal', 'partial', 'wrong_cohort'])
def test_missing_or_forged_runtime_evidence_fails(accepted, fault):
    panel = scaled.build_panel(accepted, batch_size=4, qualification=True)
    config = config_for(accepted, panel)
    value = partition_result(accepted, panel, config)
    receipt = value['partition_result']['cohort_receipts'][0]
    row = receipt['result']
    if fault == 'fallback': row['rendered_observation_receipt']['raytraced_fallbacks'] = 1
    elif fault == 'reset': row['scaled_initial_state_attestation']['native_reset']['assignment']['simulator_seed'] += 1
    elif fault == 'initial': row['scaled_initial_state_attestation']['initial_state'].pop('homie_obs')
    elif fault == 'action': row['optimizer_steps'] = 1
    elif fault == 'checkpoint': row['checkpoint_sha256'] = 'a'*64
    elif fault == 'goal': row['outcomes'][0]['native_success'] = False
    elif fault == 'partial': row['completed'] -= 1
    else: receipt['cohort_index'] = 1
    # Rehashing an altered result is still insufficient; its semantic gates run.
    receipt['result_sha256'] = scaled.digest(row)
    with pytest.raises(ValueError): scaled.validate_result(panel, value, config['checkpoint_sha256'], rendered_config=config)


def test_admission_binds_fresh_native_identity_and_does_not_confuse_qualification_with_readiness(accepted):
    panel = scaled.build_panel(accepted, qualification=True)
    config = config_for(accepted, panel)
    state = dict(worker_instance_id='a'*32, source_sha256='c'*64, phase='IDLE', active_command_id=None, commands={})
    args = dict(source_sha256=state['source_sha256'], worker_id=state['worker_instance_id'],
        command_id='qualified-c0', output_name='qualified-c0', cohort_indices=[0])
    admission = scaled.build_admission(panel, config, **args)
    kwargs = dict(source_sha256=state['source_sha256'], native_state=state,
        output_name=args['output_name'], partition=explicit.make_partition(panel, [0]))
    assert scaled.validate_admission(admission, panel, config, **kwargs) == admission
    for field in ('source_sha256', 'worker_instance_id', 'panel_sha256', 'config_sha256', 'partition_sha256', 'checkpoint_sha256'):
        bad = deepcopy(admission)
        bad[field] = '0'*len(bad[field])
        with pytest.raises(ValueError): scaled.validate_admission(bad, panel, config, **kwargs)
    with pytest.raises(ValueError): scaled.build_admission(panel, config, **dict(args, cohort_indices=[0, 1]))
    test_panel = scaled.build_panel(accepted)
    with pytest.raises(ValueError, match='actual qualification'):
        scaled.build_admission(test_panel, config_for(accepted, test_panel), **args)


@pytest.mark.parametrize('batch', [4, 8, 16])
def test_real_evaluator_function_uses_one_global_cohort_and_endpoint_pool(accepted, tmp_path, monkeypatch, batch):
    panel = scaled.build_panel(accepted, batch_size=batch, qualification=True)
    config = config_for(accepted, panel)
    calls, closed = [], []
    class Adapter:
        def __init__(self, trainer, **kwargs):
            assert kwargs['endpoint_pool_config'] == config['renderer']
            assert kwargs['evaluation_binding']['cohort_index'] == 1
            for lane in range(batch):
                assert kwargs['client_factory'](lane) == (18550 + lane % 4, 75.)
        def close(self): closed.append(True)
    class Evaluator:
        def evaluate_cohort(self, assignment, **kwargs):
            assert self._require_rendered_success_observations
            assert self._scaled_initialization_distribution == scaled.digest(panel['scaled_protocol'])
            assert kwargs['diagnostic_initial_robot_refresh'] == refresh.SCALED_STUDENT_CLEAR_ONCE
            calls.append(assignment)
            return scaled_cohort(accepted, panel, config, 1)
    monkeypatch.setattr(observations, 'GeneratedSuccessObservations', Adapter)
    def client(host, port, **kwargs):
        assert kwargs['transport_recovery'] == 'roborender.image3f.cached-reply.v1'
        return port, kwargs['recovery_timeout_s']
    monkeypatch.setitem(sys.modules, 'pipeline.dagger.services.render_student', SimpleNamespace(_Image3FLoopbackClient=client))
    execute = evaluator_function(tmp_path, Evaluator)
    prepared = []
    value = execute(SimpleNamespace(), panel, root=tmp_path/'native', checkpoint_path='/selected.pt',
        checkpoint_sha256=config['checkpoint_sha256'], prepare_cohort=prepared.append,
        report=lambda row: None, rendered_observation_config=config,
        _scaled_partition=explicit.make_partition(panel, [1]), **scaled.evaluation_options(panel))
    assert calls == prepared == [panel['cohorts'][1]['assignment']] and closed == [True]
    assert value['partition_result']['completed_episodes'] == batch
    assert (tmp_path/'native'/'cohort-receipts'/'cohort-001.json').is_file()
    assert not (tmp_path/'native'/'cohort-receipts'/'cohort-000.json').exists()


def test_new_operation_without_partition_fails_before_scene(accepted, tmp_path):
    panel = scaled.build_panel(accepted, qualification=True)
    prepared = []
    with pytest.raises(ValueError, match='separate admitted partition'):
        evaluator_function(tmp_path, object)(SimpleNamespace(), panel, root=tmp_path/'unused',
            checkpoint_path='unused', checkpoint_sha256='b'*64, prepare_cohort=prepared.append,
            report=lambda row: None, **scaled.evaluation_options(panel))
    assert not prepared and not (tmp_path/'unused').exists()


def native_admission(accepted, tmp_path, monkeypatch, *, qualification=True, sharded=False):
    panel = scaled.build_panel(accepted, qualification=qualification)
    config = config_for(accepted, panel, sharded=sharded)
    state = dict(worker_instance_id='a'*32, source_sha256='c'*64, phase='IDLE', active_command_id=None, commands={})
    partition = explicit.make_partition(panel, [0])
    admission = scaled.build_admission(panel, config, source_sha256=state['source_sha256'],
        worker_id=state['worker_instance_id'], command_id='native-c0', output_name='native-c0',
        cohort_indices=[0], qualification_evidence_sha256=None if qualification else 'd'*64)
    documents = {'panel.json': panel, 'config.json': config, 'partition.json': partition,
                 'admission.json': admission, 'resident-state.json': state}
    paths = {}
    for name, value in documents.items():
        paths[name] = tmp_path/name
        paths[name].write_bytes(explicit.canonical(value))
    payload = dict(checkpoint_sha256=config['checkpoint_sha256'], arm='bc', output_name='native-c0',
        **scaled.evaluation_options(panel))
    for key, name in [('panel', 'panel.json'), ('rendered_observation_config', 'config.json'),
                      ('partition', 'partition.json'), ('scaled_admission', 'admission.json')]:
        payload[key+'_file'] = name
        payload[key+'_sha256'] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
    source = HERE/'resident_isaac_trainer.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'ResidentRavenBackend')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_scaled_success_inputs', 'validate_command_identity'}]
    checks = []
    def file_identity(path, expected, maximum=None):
        path = Path(path)
        checks.append(path)
        if path.parent != Path('/diagnostic-assets'):
            assert path.stat().st_size <= maximum
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('raw identity differs')
        return expected
    scope = dict(os=os, Path=Path, json=json, file_identity=file_identity)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), scope)
    owner = SimpleNamespace(runtime_root=tmp_path, _consumed_outputs=set(), artifact=lambda name: paths[name],
        source_sha256=state['source_sha256'], motion_inventory_sha256=panel['motion_inventory_sha256'])
    for method in methods:
        setattr(owner, method.name, MethodType(scope[method.name], owner))
    monkeypatch.setenv('RESIDENT_SCALED_SUCCESS', scaled.SCHEMA)
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295', '1')
    monkeypatch.setenv('RESIDENT_PIPELINE_MODE', 'production')
    return owner, payload, paths, documents, checks


@pytest.mark.parametrize('fault', [None, 'raw', 'source', 'worker', 'operation', 'partition', 'checkpoint', 'extra', 'resume', 'opt_in'])
def test_native_admission_rechecks_all_inputs_before_checkpoint_or_scene(accepted, tmp_path, monkeypatch, fault):
    owner, payload, paths, documents, checks = native_admission(accepted, tmp_path, monkeypatch)
    if fault == 'raw': paths['panel.json'].write_text('{}')
    elif fault in {'source', 'worker', 'operation'}:
        field = {'source': 'source_sha256', 'worker': 'worker_instance_id', 'operation': 'operation'}[fault]
        documents['admission.json'][field] = 'wrong'
        paths['admission.json'].write_bytes(explicit.canonical(documents['admission.json']))
        payload['scaled_admission_sha256'] = hashlib.sha256(paths['admission.json'].read_bytes()).hexdigest()
    elif fault == 'partition':
        paths['partition.json'].write_bytes(explicit.canonical(explicit.make_partition(documents['panel.json'], [1])))
        payload['partition_sha256'] = hashlib.sha256(paths['partition.json'].read_bytes()).hexdigest()
    elif fault == 'checkpoint': payload['checkpoint_sha256'] = 'd'*64
    elif fault in {'extra', 'resume'}: payload['completed_receipts_file' if fault == 'resume' else 'unknown'] = 'old'
    elif fault == 'opt_in': monkeypatch.delenv('RESIDENT_SCALED_SUCCESS')
    if fault:
        with pytest.raises(ValueError): owner._scaled_success_inputs(payload)
        assert not any(path.parent == Path('/diagnostic-assets') for path in checks)
    else:
        actual = owner._scaled_success_inputs(payload)
        assert actual[0] == documents['panel.json'] and actual[3] == documents['admission.json']
        assert checks[-1].parent == Path('/diagnostic-assets')


def test_native_envelope_must_match_admission_before_acceptance(accepted, tmp_path, monkeypatch):
    owner, payload, _, _, _ = native_admission(accepted, tmp_path, monkeypatch)
    command = dict(operation=scaled.OPERATION, command_id='native-c0', worker_instance_id='a'*32, payload=payload)
    owner.validate_command_identity(command)
    command['command_id'] = 'different-command'
    with pytest.raises(ValueError, match='envelope'): owner.validate_command_identity(command)


def test_native_full_test_never_launches_on_a_syntactic_qualification_hash(accepted, tmp_path, monkeypatch):
    owner, payload, _, _, checks = native_admission(accepted, tmp_path, monkeypatch, qualification=False)
    with pytest.raises(ValueError, match='genuinely validated'):
        owner._scaled_success_inputs(payload)
    assert not any(path.parent == Path('/diagnostic-assets') for path in checks)
