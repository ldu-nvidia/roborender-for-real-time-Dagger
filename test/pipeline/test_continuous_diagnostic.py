"""CPU-only bounded real-path tests; no native/H200 qualification claim."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from pipeline.dagger.rtx.code.src.dagger_rtx import continuous_diagnostic as scope_module
from pipeline.dagger.rtx.code.src.dagger_rtx.contracts import canonical_json
from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import ProductionSegmentDriver
from pipeline.dagger.services.client import ServiceError
from pipeline.deployment import continuous_production_collection as provider_module
from test.pipeline.test_dagger_continuous_actor import fake_driver, gate_driver, sha, continuous_inputs
from test.pipeline.test_dagger_fixed100_native import fixed_inputs
from test.pipeline.test_dagger_coverage_production import production
from test.pipeline.test_continuous_production_collection import provider
from test.pipeline.test_dagger_continuous_plan import continuous_plan, continuous_evidence
from pipeline.bc_training.handoff_common import binding, bound, publish
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.coverage_production import digest
from contextlib import contextmanager
import sys
import time
from test.pipeline.test_dagger_native_production_integration import extracted, RESIDENT
from dagger_rtx.durable import atomic_write_json
from dagger_rtx.beta_round import acknowledgement


def scope_for(segment, config_sha='a'*64, segment_sha='b'*64, source_sha='c'*64):
    return dict(schema=scope_module.SCHEMA, purpose='diagnostic_only',
        run_id=segment['run_id'], round_id=segment['round_id'], source_sha256=source_sha,
        config_sha256=config_sha, segment_sha256=segment_sha,
        coverage_plan_sha256=segment['coverage_plan_sha256'],
        mse_protocol_file_sha256=segment['mse_protocol_file_sha256'],
        parent_checkpoint_sha256=segment['input_checkpoint_sha256'],
        optimizer_updates=4, environments_per_actor=16, full_batch_environments=32, rollout_steps=81,
        checkpoint_selection='none_diagnostic_candidate_only',
        terminal_action='finish_diagnostic_after_genuine_prefix_qualification', acknowledgement=scope_module.ACK,
        owners=[dict(physical_gpu=g, gpu_uuid=f'GPU-{g}', worker_instance_id=str(g+1)*32,
                     native_pid=os.getpid()+g, command_id=f'probe-g{g}') for g in (0, 1)])


@pytest.mark.parametrize('mutation', [
    {'schema': 'production'}, {'purpose': 'production'}, {'optimizer_updates': 5},
    {'optimizer_updates': True}, {'environments_per_actor': 8}, {'full_batch_environments': 16},
    {'rollout_steps': 162}, {'checkpoint_selection': 'final_update_4'}, {'round_id': 2},
    {'terminal_action': 'continue'}, {'acknowledgement': 'yes'}, {'source_sha256': 'd'*64},
    {'config_sha256': 'd'*64}, {'segment_sha256': 'd'*64}, {'mse_protocol_file_sha256': 'd'*64},
])
def test_exact_diagnostic_scope_cannot_expand_or_select_production(fixed_inputs, mutation):
    _, _, _, segment = continuous_inputs(fixed_inputs)
    scope = scope_for(segment)
    scope_module.validate_scope(scope, config_sha256='a'*64, segment_sha256='b'*64,
                                segment=segment, source_sha256='c'*64)
    scope.update(mutation)
    with pytest.raises(ValueError):
        scope_module.validate_scope(scope, config_sha256='a'*64, segment_sha256='b'*64,
                                    segment=segment, source_sha256='c'*64)


def test_scope_file_and_native_identity_bound_before_steps(fixed_inputs, tmp_path):
    _, _, _, segment = continuous_inputs(fixed_inputs)
    scope = scope_for(segment)
    path = tmp_path/'scope.json'; path.write_bytes(canonical_json(scope))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert scope_module.read_scope(path, digest, config_sha256='a'*64, segment_sha256='b'*64,
                                   segment=segment, source_sha256='c'*64) == scope
    with pytest.raises(ValueError, match='bytes changed'):
        scope_module.read_scope(path, 'f'*64, config_sha256='a'*64, segment_sha256='b'*64,
                                segment=segment, source_sha256='c'*64)
    state = dict(schema='roborender.isaac-resident.v1', source_sha256='c'*64, native_initializations=1,
        phase='IDLE', active_command_id=None, **{k: scope['owners'][0][k]
        for k in ('native_pid', 'worker_instance_id', 'gpu_uuid')})
    assert scope_module.require_owner(scope, state, gpu=0, native_pid=os.getpid()) == scope['owners'][0]
    state.update(phase='RUNNING', active_command_id='probe-g0')
    scope_module.require_owner(scope, state, gpu=0, native_pid=os.getpid(), command_id='probe-g0')
    for mutation in ({'native_pid': 1}, {'phase': 'QUEUED'}, {'active_command_id': 'other'},
                     {'source_sha256': 'e'*64}, {'worker_instance_id': 'e'*32}):
        with pytest.raises(ValueError):
            scope_module.require_owner(scope, {**state, **mutation}, gpu=0, native_pid=os.getpid())


def test_real_driver_stops_after_four_windows_no_fifth_or_selection(tmp_path):
    driver, calls, offsets, barriers, gates = fake_driver(tmp_path)
    driver.diagnostic_scope = {'explicit': 'already-admitted-constructor'}
    driver.diagnostic_scope_sha256 = 'f'*64
    def qualified(checkpoint):
        gates.append(checkpoint)
        driver.state['qualification_proof'] = {'actual': 'prefix-proof-fixture'}
    driver._wait_continuous_qualification = qualified
    result = driver.run()
    assert len(calls) == 12 and offsets == [0, 81, 162, 243]
    assert len(barriers) == 64 and gates == [sha(4)]
    assert result['phase'] == 'DIAGNOSTIC_COMPLETE' and result['completed_windows'] == 4
    assert result['physical_transition_rows'] == 5184 and result['transition_batches'] == 324
    assert result['environment_initializations'] == 1 and result['model_rollout_lifecycles'] == 4
    assert result['selection'] is None and result['fifth_window_started'] is False
    assert result['production_round_complete'] is False
    assert result['final_candidate_checkpoint_sha256'] == sha(4)


def test_diagnostic_cannot_finish_without_real_prefix_gate(tmp_path):
    driver, calls, _, _, _ = fake_driver(tmp_path)
    driver.diagnostic_scope = {'explicit': True}; driver.diagnostic_scope_sha256 = 'f'*64
    with pytest.raises(RuntimeError, match='genuine'):
        driver.run()
    assert len(calls) == 12 and driver.state['phase'] == 'CRASH_UNCERTAIN'


@pytest.mark.parametrize('fault', [None, 'production_gate', 'wrong_scope', 'continue_decision'])
def test_diagnostic_gate_is_separate_from_production_continue_gate(tmp_path, fault):
    driver, gate, _ = gate_driver(tmp_path)
    driver.diagnostic_scope = {'explicit': True}; driver.diagnostic_scope_sha256 = 'f'*64
    gate.update(schema=scope_module.GATE_SCHEMA, diagnostic_scope_sha256='f'*64,
                decision='finish_diagnostic')
    if fault == 'production_gate':
        gate['schema'] = 'roborender.dagger-continuous-gate.v1'
    elif fault == 'wrong_scope': gate['diagnostic_scope_sha256'] = 'e'*64
    elif fault == 'continue_decision': gate['decision'] = 'continue'
    driver.qualification_gate_file.write_text(json.dumps(gate))
    if fault:
        with pytest.raises(RuntimeError): driver._wait_continuous_qualification(sha(4))
    else:
        driver._wait_continuous_qualification(sha(4))
        assert driver.state['phase'] == 'QUALIFIED'


def test_production_driver_cannot_accept_diagnostic_finish_gate(tmp_path):
    driver, gate, _ = gate_driver(tmp_path)
    gate.update(schema=scope_module.GATE_SCHEMA, diagnostic_scope_sha256='f'*64, decision='finish_diagnostic')
    driver.qualification_gate_file.write_text(json.dumps(gate))
    with pytest.raises(RuntimeError): driver._wait_continuous_qualification(sha(4))


def no_fifth_fixture():
    site = SimpleNamespace(run='diagnostic-run', rid=1, config={'student': {'initial_checkpoint_sha256': 'a'*64}})
    status = dict(iterations=[dict(round_id=1, iteration_id=i, status='advanced' if i < 5 else 'collecting',
        candidate_checkpoint_sha256='b'*64 if i < 5 else None) for i in range(1, 6)],
        barriers=[dict(round_id=1, iteration_id=i) for i in range(1, 5)])
    context = dict(status='active', selection=dict(history=[], best_checkpoint_sha256='a'*64))
    coordinator = SimpleNamespace(get=lambda route: {'mse_round': context} if 'mse-rounds' in route else status)
    def absent(route):
        assert route.endswith('-0005')
        raise ServiceError('404') from HTTPError('http://local'+route, 404, 'missing', {}, None)
    return site, status, context, coordinator, SimpleNamespace(get=absent)


def test_empty_iteration5_metadata_is_not_a_fifth_update():
    site, _, _, coordinator, learner = no_fifth_fixture()
    result = provider_module.require_no_fifth_update(site, learner, coordinator)
    assert result['optimizer_job5_absent'] and result['iteration5_barriers'] == 0
    assert result['next_iteration_status'] == 'collecting_empty'


@pytest.mark.parametrize('fault', ['job_exists', 'timeout', 'text404', 'barrier5', 'candidate5', 'selection', 'iteration6'])
def test_no_fifth_proof_rejects_uncertainty_and_any_extra_work(fault):
    site, status, context, coordinator, learner = no_fifth_fixture()
    if fault == 'job_exists': learner.get = lambda _: {'job': {'status': 'queued'}}
    elif fault in {'timeout', 'text404'}:
        def unavailable(_): raise ServiceError('404' if fault == 'text404' else 'transport unavailable')
        learner.get = unavailable
    elif fault == 'barrier5': status['barriers'].append(dict(round_id=1, iteration_id=5))
    elif fault == 'candidate5': status['iterations'][-1]['candidate_checkpoint_sha256'] = 'd'*64
    elif fault == 'selection': context['status'] = 'selected_pending_test'
    else: status['iterations'].append(dict(round_id=1, iteration_id=6))
    with pytest.raises((ValueError, ServiceError)):
        provider_module.require_no_fifth_update(site, learner, coordinator)


def test_stage_diagnostic_persists_scope_before_only_explicit_pair_dispatch(provider, tmp_path):
    evidence, site, old, _, _, _ = provider
    directory = tmp_path/'new-diagnostic'
    site.remote = site.jobs/'round'
    config_ref = publish(tmp_path/'config.json', site.config)
    protocol = tmp_path/'protocol.json'; publish(protocol, {'cpu_only_protocol': True})
    site.request = dict(round_config_path=config_ref['path'], mse_protocol_path=str(protocol),
        mse_protocol_file_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    site.read = lambda *args: {}
    writes, submitted = [], []
    site.publish = lambda path, raw, *args: writes.append((path, raw))
    def pool(action, commands=None):
        if action == 'status':
            return dict(workers=[dict(row, phase='IDLE', active_command_id=None) for row in old['owners']])
        assert action == 'submit'
        saved = json.loads((directory/'intent.json').read_bytes())
        assert commands == saved['commands'] and bound(saved['diagnostic_scope'])
        submitted.append(commands)
    site.pool_action = pool
    intent = provider_module._stage(site, evidence['plan'], directory, diagnostic=True)
    scope = provider_module._require_diagnostic_intent(site, intent)
    assert len(submitted) == 1 and len(submitted[0]) == 2
    assert all(c['operation'] == 'probe_continuous_training' for c in submitted[0])
    assert scope['optimizer_updates'] == 4 and intent['segment']['window_count'] == 100
    assert scope['checkpoint_selection'] == 'none_diagnostic_candidate_only'
    assert next(raw for path, raw in writes if path.name == 'diagnostic-scope.json') == canonical_json(scope)


def test_versioned_diagnostic_scope_binds_both_curriculum_requests(fixed_inputs):
    from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import bootstrap_request
    _, _, _, segment = continuous_inputs(fixed_inputs)
    scope = scope_for(segment)
    scope.update(schema=scope_module.CURRICULUM_SCHEMA,
                 training_curriculum=[bootstrap_request(), bootstrap_request()])
    kwargs = dict(config_sha256='a'*64, segment_sha256='b'*64, segment=segment, source_sha256='c'*64)
    assert scope_module.validate_scope(scope, **kwargs) == scope
    with pytest.raises(ValueError): scope_module.validate_scope({**scope, 'schema': scope_module.SCHEMA}, **kwargs)
    with pytest.raises(ValueError): scope_module.validate_scope({**scope, 'training_curriculum': [bootstrap_request()]}, **kwargs)


def diagnostic_provider(provider, tmp_path):
    evidence, site, old, prefixes, state, get = provider
    intent = deepcopy(old)
    segment = prefixes[0]['actor_state']['segment']
    scope = scope_for(segment, digest(site.config), intent['segment_sha256'], site.config['source_sha256'])
    for gpu, (command, owner) in enumerate(zip(intent['commands'], intent['owners'])):
        scope['owners'][gpu] = dict(physical_gpu=gpu, command_id=command['command_id'],
            **{k: owner[k] for k in ('worker_instance_id', 'native_pid', 'gpu_uuid')})
    directory = site.round/'continuous-diagnostic'
    immutable_json(directory/'diagnostic-scope.json', scope)
    scope_ref = binding(directory/'diagnostic-scope.json')
    for command in intent['commands']:
        command['operation'] = 'probe_continuous_training'
        command['payload'].update(diagnostic_scope_file='diagnostic/scope.json',
                                  diagnostic_scope_sha256=scope_ref['raw_sha256'])
    intent.update(schema='roborender.continuous-diagnostic-intent.v1', segment=segment,
                  diagnostic_scope=scope_ref)
    immutable_json(directory/'intent.json', intent)
    final = bound(evidence['proof']['learner_receipts'][-1])['output_checkpoint_sha256']
    terminals = []
    for prefix in prefixes:
        terminal = deepcopy(prefix)
        terminal['actor_state'].update(phase='DIAGNOSTIC_COMPLETE', final_candidate_checkpoint_sha256=final,
            environment_reset_at_window_boundary=False, selection=None, fifth_window_started=False)
        terminal.update(complete=True, diagnostic_only=True, production_round_complete=False,
                        diagnostic_scope_sha256=scope_ref['raw_sha256'], segment_sha256=intent['segment_sha256'])
        terminals.append(dict(status='COMPLETE', receipt=terminal))
    def pool(action, commands=None):
        state['actions'].append(action)
        if action == 'status':
            finished = state['polls'] >= 2
            return dict(workers=[dict(owner, phase='IDLE' if finished else 'RUNNING',
                active_command_id=None if finished else command['command_id'])
                for command, owner in zip(intent['commands'], intent['owners'])])
        assert action == 'result', 'persisted diagnostic commands must never be resubmitted'
        state['polls'] += 1
        return dict(results=terminals if state['polls'] >= 2 else [dict(status='RUNNING')]*2)
    site.pool_action = pool
    no_fifth_site, status, context, _, missing = no_fifth_fixture()
    context['selection']['best_checkpoint_sha256'] = site.config['student']['initial_checkpoint_sha256']
    def read(route):
        if '/jobs/' in route:
            return missing.get(route) if route.endswith('-0005') else get(route)
        if '/mse-rounds/' in route: return dict(mse_round=context)
        return status
    return evidence, site, intent, prefixes, state, read, terminals


def test_actual_diagnostic_provider_reconciles_four_and_never_resubmits_or_promotes(provider, tmp_path, monkeypatch):
    evidence, site, intent, prefixes, state, get, _ = diagnostic_provider(provider, tmp_path)
    monkeypatch.setattr(provider_module, 'ServiceClient', lambda *a, **kw: SimpleNamespace(get=get))
    monkeypatch.setattr(provider_module, '_read_prefix', lambda s, c: prefixes[c['gpu']])
    gates = []
    def qualify(s, plan, admitted, ps, learner, directory):
        assert admitted == intent and ps == prefixes
        immutable_json(directory/'qualification.json', evidence['proof'])
        gates.append('actual-prefix-validator-used-at-terminal')
        return binding(directory/'qualification.json')
    monkeypatch.setattr(provider_module, 'admit_prefix', qualify)
    result = provider_module.collect_diagnostic(site, sleep=lambda _: None)
    assert len(gates) == 1 and result['completed_updates'] == 4
    assert result['diagnostic_only'] and result['selected_checkpoint_sha256'] is None
    assert result['optimizer_job5_absent'] and result['production_round_complete'] is False
    assert len(result['learner_receipts']) == 4 and len(result['native_terminal_receipts']) == 2
    assert not {'submit', 'shutdown', 'abort'} & set(state['actions'])
    assert not (site.round/'production-collection-result.json').exists()


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('updates', [4, 100])
def test_shared_population_terminal_uses_exact_admitted_diagnostic_or_production_budget(
        provider, tmp_path, diagnostic, updates):
    from pipeline.dagger.rtx.code.src.dagger_rtx import population_curriculum as population
    _, site, intent, _, _, _, terminals = diagnostic_provider(provider, tmp_path)
    site.config['population_curriculum_protocol'] = population.PROTOCOL
    receipt = terminals[0]['receipt']
    state = receipt['actor_state']
    state['config_sha256'] = digest(site.config)
    scope = scope_for(state['segment'], digest(site.config), intent['segment_sha256'], site.config['source_sha256'])
    receipt['diagnostic_scope_sha256'] = digest(scope)
    receipt['population_curriculum'] = dict(schema=population.PROTOCOL, population=population.POPULATION,
        vector_reset_barriers=1+81*updates, global_reset_events=2, physical_resets_added=0,
        final_state_sha256='c'*64)
    kwargs = dict(terminal=True, diagnostic_scope=scope if diagnostic else None)
    args = (site, intent['commands'][0], intent['owners'][0], intent['segment_sha256'], receipt)
    if updates == (4 if diagnostic else 100):
        result = provider_module.native_wrapper(*args, **kwargs)
        assert result['population_curriculum']['vector_reset_barriers'] == 1+81*updates
    else:
        with pytest.raises(ValueError, match='contiguous native reset barriers'):
            provider_module.native_wrapper(*args, **kwargs)


@pytest.mark.parametrize('fault', ['scope', 'diagnostic_only', 'production_complete', 'phase', 'windows'])
def test_shared_population_four_update_terminal_cannot_weaken_production_gate(provider, tmp_path, fault):
    from pipeline.dagger.rtx.code.src.dagger_rtx import population_curriculum as population
    _, site, intent, _, _, _, terminals = diagnostic_provider(provider, tmp_path)
    site.config['population_curriculum_protocol'] = population.PROTOCOL
    receipt = terminals[0]['receipt']; state = receipt['actor_state']
    state['config_sha256'] = digest(site.config)
    scope = scope_for(state['segment'], digest(site.config), intent['segment_sha256'], site.config['source_sha256'])
    receipt['diagnostic_scope_sha256'] = digest(scope)
    receipt['population_curriculum'] = dict(schema=population.PROTOCOL, population=population.POPULATION,
        vector_reset_barriers=325, global_reset_events=2, physical_resets_added=0, final_state_sha256='c'*64)
    if fault == 'scope': receipt['diagnostic_scope_sha256'] = 'f'*64
    elif fault == 'diagnostic_only': receipt['diagnostic_only'] = False
    elif fault == 'production_complete': receipt['production_round_complete'] = True
    elif fault == 'phase': state['phase'] = 'COMPLETE'
    else: state['completed_windows'] = 3
    with pytest.raises(ValueError, match='exact admitted diagnostic scope'):
        provider_module.native_wrapper(site, intent['commands'][0], intent['owners'][0], intent['segment_sha256'],
            receipt, terminal=True, diagnostic_scope=scope)


@pytest.mark.parametrize('fault', ['fifth_step', 'selection', 'wrong_scope', 'owner', 'bad_proof'])
def test_diagnostic_terminal_cannot_hide_extra_work_or_uncertain_owner(provider, tmp_path, monkeypatch, fault):
    evidence, site, intent, prefixes, state, get, terminals = diagnostic_provider(provider, tmp_path)
    if fault == 'fifth_step': terminals[0]['receipt']['actor_state']['transition_batches'] = 325
    elif fault == 'selection': terminals[0]['receipt']['actor_state']['selection'] = {'selected': True}
    elif fault == 'wrong_scope': terminals[0]['receipt']['diagnostic_scope_sha256'] = 'e'*64
    elif fault == 'owner': intent['owners'][0]['native_pid'] += 1
    else: evidence['proof']['optimizer_updates'] = 3
    immutable_json(site.round/'continuous-diagnostic'/'qualification.json', evidence['proof'])
    state['polls'] = 2
    monkeypatch.setattr(provider_module, 'ServiceClient', lambda *a, **kw: SimpleNamespace(get=get))
    with pytest.raises(ValueError): provider_module.collect_diagnostic(site, sleep=lambda _: None)


def test_diagnostic_gate_publication_uses_distinct_decision_and_atomic_path(provider, tmp_path):
    evidence, site, intent, _, _, _, _ = diagnostic_provider(provider, tmp_path)
    proof_path = tmp_path/'proof-for-gate.json'; immutable_json(proof_path, evidence['proof'])
    writes, commands = [], []
    site.publish = lambda path, raw, *args: writes.append((path, raw))
    site.command = lambda argv, host: commands.append(argv)
    provider_module.publish_gate(site, intent, binding(proof_path))
    gate = next(json.loads(raw) for path, raw in writes if path.name.endswith('prepared.json'))
    assert gate['schema'] == scope_module.GATE_SCHEMA and gate['decision'] == 'finish_diagnostic'
    assert gate['diagnostic_scope_sha256'] == intent['diagnostic_scope']['raw_sha256']
    assert str(commands[-1][-1]).endswith('/continuous-diagnostic-gate.json')
    assert 'os.link' in commands[-1][2]


def test_real_driver_constructor_binds_exact_scope_before_running(fixed_inputs, tmp_path):
    plan, config, protocol, segment = continuous_inputs(fixed_inputs)
    config_sha, segment_sha = digest(config), digest(segment)
    scope = scope_for(segment, config_sha, segment_sha, config['source_sha256'])
    runner = SimpleNamespace(num_envs=16, actor_prefix=config['run_id']+'-g0',
                             episode_namespace=segment['episode_namespace'])
    options = dict(segment=segment, segment_file_sha256=segment_sha,
        mse_protocol=protocol, mse_protocol_file_sha256=segment['mse_protocol_file_sha256'],
        qualification_gate_file=tmp_path/'gate.json', resource_snapshot=lambda: {},
        assignment_snapshot=lambda: [], coverage_plan_document_sha256=digest(plan),
        diagnostic_scope=scope, diagnostic_scope_sha256=digest(scope))
    driver = ProductionSegmentDriver(runner, lambda *a: None, tmp_path, config, config_sha, 0, **options)
    assert driver.state['diagnostic_only'] and driver.state['production_round_complete'] is False
    options['diagnostic_scope_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='exact canonical'):
        ProductionSegmentDriver(runner, lambda *a: None, tmp_path, config, config_sha, 0, **options)


def test_native_diagnostic_admission_and_envelope_before_any_scene_mutation(fixed_inputs, tmp_path, monkeypatch):
    _, config, _, segment = continuous_inputs(fixed_inputs)
    scope = scope_for(segment, digest(config), digest(segment), config['source_sha256'])
    scope['owners'][0]['gpu_uuid'] = config['rtx_actors'][0]['gpu_uuid']
    path = tmp_path/'scope.json'; path.write_bytes(canonical_json(scope))
    state = dict(schema='roborender.isaac-resident.v1', source_sha256=config['source_sha256'],
        native_initializations=1, phase='IDLE', active_command_id=None,
        **{k: scope['owners'][0][k] for k in ('native_pid', 'worker_instance_id', 'gpu_uuid')})
    (tmp_path/'resident-state.json').write_text(json.dumps(state))
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', '0'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES', scope['owners'][0]['gpu_uuid'])
    calls = []
    backend = SimpleNamespace(runtime_root=tmp_path, source_sha256=config['source_sha256'],
        artifact=lambda name: tmp_path/name, validate=lambda op, p: calls.append((op, p)),
        _production_segment_inputs=lambda p: (config, segment, {}))
    ns = extracted(RESIDENT, ['_continuous_diagnostic_inputs', 'validate_command_identity'],
        klass='ResidentRavenBackend', namespace=dict(os=os, json=json))
    backend._continuous_diagnostic_inputs = lambda p: ns['_continuous_diagnostic_inputs'](backend, p)
    payload = dict(output_name='probe-g0', round_config_sha256=digest(config), segment_sha256=digest(segment),
        diagnostic_scope_file=path.name, diagnostic_scope_sha256=digest(scope))
    assert backend._continuous_diagnostic_inputs(payload) == scope
    assert calls[-1][0] == 'collect_production_segment'
    assert 'diagnostic_scope_file' not in calls[-1][1]
    command = dict(operation='probe_continuous_training', command_id='probe-g0',
                   worker_instance_id=scope['owners'][0]['worker_instance_id'], payload=payload)
    ns['validate_command_identity'](backend, command)
    with pytest.raises(ValueError, match='envelope'):
        ns['validate_command_identity'](backend, {**command, 'command_id': 'other'})
    payload['diagnostic_scope_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='bytes changed'): backend._continuous_diagnostic_inputs(payload)


@pytest.mark.parametrize('bad_terminal', [False, True])
@pytest.mark.parametrize('curriculum_mode', [None, 'bootstrap_from_config', 'restore_predecessor'])
def test_native_diagnostic_handler_uses_real_driver_mode_and_returns_clean_only_after4(
        fixed_inputs, tmp_path, monkeypatch, bad_terminal, curriculum_mode):
    plan, config, _, segment = continuous_inputs(fixed_inputs)
    scope = scope_for(segment, digest(config), digest(segment), config['source_sha256'])
    assignment = plan['rounds'][0]['native_round']['assignments'][0]
    payload = dict(output_name='probe-g0', round_config='config.json', round_config_sha256=digest(config),
        token_file='token', connection_status_directory='connection', connection_config_sha256='a'*64,
        mse_protocol_file='protocol.json', mse_protocol_sha256=segment['mse_protocol_file_sha256'],
        segment_file='segment.json', segment_sha256=digest(segment), coverage_plan_file='plan.json',
        motion_assignment_sha256='b'*64, diagnostic_scope_file='scope.json', diagnostic_scope_sha256=digest(scope))
    curriculum_prior = None
    if curriculum_mode is not None:
        from pipeline.dagger.rtx.code.src.dagger_rtx import training_curriculum as curriculum
        from test.pipeline.test_training_curriculum import env, trained, binding as curriculum_binding
        import torch
        native_env = env()
        native_env.reset_all = lambda: native_env._reset_tasks_callback(torch.arange(16))
        request = curriculum.bootstrap_request()
        if curriculum_mode == 'restore_predecessor':
            prior_binding = curriculum_binding()
            prior_binding.update(native_pid=os.getpid(), gpu_uuid='GPU-fixture', source_sha256=config['source_sha256'])
            curriculum_prior = curriculum.export_training_curriculum(trained(), prior_binding, phase='training')
            request = dict(schema=curriculum.REQUEST_SCHEMA, mode=curriculum_mode, predecessor=dict(
                binding=prior_binding, export_sha256=curriculum.digest(curriculum_prior), native_terminal_receipt_sha256='e'*64))
        scope.update(schema=scope_module.CURRICULUM_SCHEMA,
                     training_curriculum=[request, curriculum.bootstrap_request()])
        payload.update(training_curriculum=request, diagnostic_scope_sha256=digest(scope))
        (tmp_path/'resident-state.json').write_text(json.dumps(dict(worker_instance_id='1'*32)))
    events = []
    class Guard:
        summary = {'passed': True}
        def __init__(self, *a, **kw): pass
        def __enter__(self): events.append('guard'); return self
        def __exit__(self, *a): events.append('guard-exit')
        def check(self): events.append('guard-check')
        def snapshot(self, path): return self.summary
    @contextmanager
    def observer(*a, **kw):
        events.append('observer')
        try: yield SimpleNamespace(episode_assignments=[{'lane': i} for i in range(16)])
        finally: events.append('observer-exit')
    monkeypatch.setitem(sys.modules, 'dagger_rtx.phase_resource_guard', SimpleNamespace(
        PhaseResourceGuard=Guard, POLICY_VERSION='v3'))
    monkeypatch.setitem(sys.modules, 'native_motion_assignment', SimpleNamespace(install_native_assignment_observer=observer))
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU', '0'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture')
    monkeypatch.setenv('DAGGER_CONTINUOUS_DIAGNOSTIC_FILE', 'prior-env')
    def native_train(trainer):
        assert os.environ['DAGGER_CONTINUOUS_DIAGNOSTIC_FILE'] == str(tmp_path/'scope.json')
        assert os.environ['DAGGER_CONTINUOUS_DIAGNOSTIC_SHA256'] == digest(scope)
        assert os.environ['DAGGER_WINDOW_INDEX'] == '0'
        events.append('real-native-driver-entry')
        if curriculum_mode is not None:
            trainer.env.reset_all()
            # Simulate later natural curriculum evolution without claiming native physics.
            trainer.env.termination_level = torch.tensor(.67)
            trainer.env.average_episode_length = torch.tensor(22.)
            trainer.env.log_dict.update(termination_level=trainer.env.termination_level,
                                       average_episode_length=trainer.env.average_episode_length)
        state = dict(phase='DIAGNOSTIC_COMPLETE', completed_windows=4,
            transition_batches=325 if bad_terminal else 324, physical_transition_rows=5184,
            history_reset_at_window_boundary=False, environment_initializations=1,
            model_rollout_lifecycles=4, environment_reset_at_window_boundary=False,
            diagnostic_scope_sha256=digest(scope), selection=None, fifth_window_started=False,
            production_round_complete=False, qualification_proof={'verified': True},
            final_candidate_checkpoint_sha256='b'*64)
        atomic_write_json(Path(os.environ['DAGGER_RUNTIME_ROOT'])/'production-segment-state.json', state)
        return dict(result='PASS', diagnostic_only=True)
    phase = extracted(RESIDENT, ['phase_environment'], namespace=dict(os=os, contextmanager=contextmanager))
    ns = extracted(RESIDENT, ['_run_production_segment'], klass='ResidentRavenBackend', namespace=dict(
        os=os, json=json, Path=Path, time=time, acknowledgement=acknowledgement,
        nullcontext=__import__('contextlib').nullcontext,
        phase_environment=phase['phase_environment'], reset_rollout_for_scene=lambda _: None,
        seeded=lambda _: events.append('seed'), atomic_write_json=atomic_write_json,
        NativeDaggerRTXServiceTrainer=SimpleNamespace(train=native_train)))
    trainer = SimpleNamespace(policy_model=None, ref_model=None, _native_resource_guard='prior-guard',
        env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(play=lambda: None,
                                                         pause=lambda: events.append('pause')))))
    if curriculum_mode is not None:
        native_env.simulator = trainer.env.simulator
        trainer.env = native_env
    backend = SimpleNamespace(trainer=trainer, runtime_root=tmp_path, _consumed_outputs=set(),
        source_sha256=config['source_sha256'], _curriculum_predecessor=lambda *_: curriculum_prior,
        _population_scope=lambda *_: None,
        artifact=lambda name: tmp_path/name, _continuous_diagnostic_inputs=lambda p: scope,
        _production_segment_inputs=lambda p: (config, segment, assignment),
        _switch_scene=lambda *a, **kw: events.append('training-scene'),
        _prepare_training=lambda _: events.append('restore-training-state'),
        _kit_unchanged=lambda: events.append('same-kit'), profile='training', phase_resets=1, scene_initializations=2)
    if bad_terminal:
        with pytest.raises(RuntimeError, match='lifecycle budget'):
            ns['_run_production_segment'](backend, payload, lambda *a, **kw: None, diagnostic=True)
        assert 'pause' not in events  # Outer failure handler, not this success tail, pauses uncertainty.
    else:
        result = ns['_run_production_segment'](backend, payload, lambda *a, **kw: None, diagnostic=True)
        assert result['complete'] and result['diagnostic_only'] and result['optimizer_updates'] == 4
        assert result['selected_checkpoint_sha256'] is None and result['production_round_complete'] is False
        assert (tmp_path/'phases/probe-g0/continuous-diagnostic-native-receipt.json').is_file()
        assert not (tmp_path/'phases/probe-g0/production-segment-native-receipt.json').exists()
        assert 'pause' in events
        if curriculum_mode is not None:
            initialized = result['training_curriculum_initialization']
            assert initialized['mode'] == curriculum_mode and initialized['initial_reset_calls'] == 1
            expected = curriculum_prior['fields']['termination_level']['value'] if curriculum_prior else 1.
            assert initialized['before_initial_reset']['termination_level']['value'] == expected
            assert initialized['after_initial_reset']['termination_level']['value'] >= expected
            assert result['training_curriculum']['binding']['final_checkpoint_sha256'] == 'b'*64
            assert result['training_curriculum']['fields']['termination_level']['value'] == torch.tensor(.67).item()
    assert events.index('observer') < events.index('restore-training-state') < events.index('real-native-driver-entry')
    assert trainer._native_resource_guard == 'prior-guard'
    assert os.environ['DAGGER_CONTINUOUS_DIAGNOSTIC_FILE'] == 'prior-env'


def test_both_transport_allowlists_require_explicit_diagnostic_operation():
    from pipeline.deployment.rtx_actor.production_resident_pool import OPERATIONS
    from dagger_rtx.resident_service import validate_command, SCHEMA
    assert 'probe_continuous_training' in OPERATIONS
    command = dict(schema=SCHEMA, operation='probe_continuous_training', command_id='test-diagnostic',
                   worker_instance_id='a'*32, payload={})
    assert validate_command(command) == command
    with pytest.raises(ValueError): validate_command({**command, 'operation': 'truncate_production'})
