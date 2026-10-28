"""CPU/read-only evidence tests. Synthetic receipts never qualify a live run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.dagger.services import diagnostic_reentry as chain
from pipeline.dagger.services import bc32_diagnostic_parent as original
from test.pipeline.test_bc32_diagnostic_parent import admission, request


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'raw_sha256': hashlib.sha256(raw).hexdigest()}


def execution():
    return dict(native_source_sha256='c'*64, learner_source_sha256='f'*64,
                worker_sha256='1'*64, server_sha256='2'*64, controller_sha256='3'*64)


def unified_execution():
    return dict(execution(), schema=chain.UNIFIED_EXECUTION_SCHEMA, learner_source_sha256='c'*64)


def test_unified_execution_is_explicit_and_keeps_file_pins():
    value = unified_execution()
    assert chain.validate_execution(value, 'c'*64) == value
    assert chain.validate_execution(execution(), 'c'*64) == execution()


@pytest.mark.parametrize('mutation', [
    lambda v: v.pop('schema'),
    lambda v: v.update(schema='unversioned'),
    lambda v: v.update(learner_source_sha256='f'*64),
    lambda v: v.update(native_source_sha256='a'*64),
    lambda v: v.update(worker_sha256='unknown'),
    lambda v: v.update(extra_field=True),
])
def test_unified_source_cannot_bypass_exact_execution_contract(mutation):
    value = unified_execution(); mutation(value)
    with pytest.raises(ValueError):
        chain.validate_execution(value, 'c'*64)


def test_unified_execution_cannot_adopt_old_scope_without_curriculum():
    value, job = initial()
    value['execution'] = unified_execution()
    with pytest.raises(ValueError, match='training-curriculum scope v2'):
        chain.validate_admission(value, job)


def curriculum_terminal_case():
    from pipeline.dagger.rtx.code.src.dagger_rtx import training_curriculum as curriculum
    from test.pipeline.test_training_curriculum import exported
    value, _ = initial()
    scope = deepcopy(value['scope_admission']['scope'])
    scope.update(schema=chain.CURRICULUM_SCOPE_SCHEMA,
                 training_curriculum=[curriculum.bootstrap_request() for _ in range(2)])
    export = exported(); b = export['binding']; owner = scope['owners'][0]
    b.update(run_id=scope['run_id'], round_id=scope['round_id'],
             source_sha256=scope['source_sha256'], diagnostic_scope_sha256=original.digest(scope),
             **{k: owner[k] for k in ('gpu_uuid', 'worker_instance_id', 'native_pid', 'command_id')})
    row = dict(training_curriculum=export, training_curriculum_sha256=curriculum.digest(export),
        native_terminal_receipt_sha256='f'*64, training_curriculum_initialization=dict(
            mode='bootstrap_from_config', initial_reset_calls=1, restored_before_initial_reset=True,
            physical_or_rng_state_restored=False, before_initial_reset={}, after_initial_reset={}))
    return curriculum, row, scope


def test_curriculum_terminal_requires_exact_training_boundary():
    curriculum, row, scope = curriculum_terminal_case()
    assert chain.validate_curriculum_terminal(row, scope, 0, 'd'*64) == curriculum.build_restore_request(row)


@pytest.mark.parametrize('fault', ['owner', 'checkpoint', 'scope', 'digest', 'test_state', 'reset_twice', 'missing_observation'])
def test_curriculum_handoff_rejects_wrong_origin_or_reset(fault):
    curriculum, row, scope = curriculum_terminal_case()
    if fault in {'owner', 'checkpoint', 'scope'}:
        key = {'owner':'worker_instance_id', 'checkpoint':'final_checkpoint_sha256', 'scope':'diagnostic_scope_sha256'}[fault]
        row['training_curriculum']['binding'][key] = '9'*(32 if fault == 'owner' else 64)
        row['training_curriculum_sha256'] = curriculum.digest(row['training_curriculum'])
    elif fault == 'digest': row['training_curriculum_sha256'] = '9'*64
    elif fault == 'test_state': row['training_curriculum']['phase'] = 'evaluation'
    elif fault == 'reset_twice': row['training_curriculum_initialization']['initial_reset_calls'] = 2
    else: row['training_curriculum_initialization'].pop('after_initial_reset')
    with pytest.raises(ValueError):
        chain.validate_curriculum_terminal(row, scope, 0, 'd'*64)


def initial():
    job = request()
    value = dict(schema=chain.SCHEMA, mode='initial_bc32', scope_admission=admission()[0],
                 execution=execution(), predecessor=None)
    job['parameters'].update({chain.PARAMETER: value, chain.RNG_PARAMETER: chain.RNG_PROTOCOL})
    return value, job


def rng(step, parent, before, after):
    return dict(schema=chain.RNG_PROTOCOL, parent_checkpoint_sha256=parent,
                input_rng_sha256=before, output_rng_sha256=after, global_step=step,
                worker_file_sha256='1'*64, train_generator_usage='preserved_unused_by_dagger')


@pytest.fixture
def return_case(tmp_path, monkeypatch):
    # Resource/panel semantics have their own comprehensive canonical suite.
    # Only that independent validator is substituted; every evidence file/hash
    # and lineage check in diagnostic_reentry executes normally here.
    from pipeline.evaluation import scaled_success_qualification
    monkeypatch.setattr(scaled_success_qualification, 'validate_qualification',
                        lambda value, **kw: value if value['resource_fixture'] is True else (_ for _ in ()).throw(ValueError('resource failure')))
    initial_admission, job = initial()
    prior_scope = initial_admission['scope_admission']['scope']
    scope_ref = write(tmp_path/'prior/scope.json', prior_scope)
    receipts = []
    parent = prior_scope['parent_checkpoint_sha256']
    for i in range(1, 5):
        child = str(i+3)*64
        row = dict(schema='roborender.dagger_learner_candidate_receipt.v3', iteration_id=i,
            parent_checkpoint_sha256=parent, output_checkpoint_sha256=child,
            parent_global_step=i+1, global_step=i+2, train_chunks=32, train_frames=2592,
            batching_semantics='raven_exact_flattened_bxt', optimizer_state_preserved=True,
            optimizer_parameter_states=78, auxiliary_object_head_zero_moments=True,
            replay_manifest_sha256=None, diagnostic_only=True, production_promoted=False,
            diagnostic_scope_sha256=original.digest(prior_scope),
            output_checkpoint_schema=original.CHECKPOINT_SCHEMA, learner_execution=execution(),
            train_mse=.5, diagnostic_learner_chain_admission_sha256=original.digest(initial_admission),
            learner_rng_receipt=rng(i+2, parent, str(i)*64, str(i+1)*64))
        receipts.append(write(tmp_path/f'prior/learner-{i}.json', row))
        parent = child
    terminal = []
    for gpu, owner in enumerate(prior_scope['owners']):
        terminal.append(write(tmp_path/f'prior/terminal-{gpu}.json', dict(
            schema='roborender.continuous-diagnostic-native-terminal.v1', **owner,
            source_sha256=prior_scope['source_sha256'], actor_state=dict(phase='DIAGNOSTIC_COMPLETE',
                completed_windows=4, transition_batches=324, physical_transition_rows=5184,
                fifth_window_started=False, selection=None, final_candidate_checkpoint_sha256=parent))))
    qualification = write(tmp_path/'prior/qualification.json', dict(
        source_sha256=prior_scope['source_sha256'], parent_checkpoint_sha256=prior_scope['parent_checkpoint_sha256'],
        optimizer_updates=4, learner_receipts=receipts, online_metrics_verified=True))
    result = dict(schema='roborender.continuous-four-update-diagnostic-result.v1', complete=True,
        diagnostic_only=True, production_round_complete=False, run_id=prior_scope['run_id'], round_id=1,
        source_sha256=prior_scope['source_sha256'], completed_updates=4, full_batch_environments=32,
        frames_per_update=2592, native_lifecycles_per_actor=[1,1], model_rollout_initializations_per_actor=[4,4],
        selected_checkpoint_sha256=None, optimizer_job5_absent=True, learner_receipts=receipts,
        native_terminal_receipts=terminal, diagnostic_scope=scope_ref, qualification=qualification,
        last_diagnostic_candidate_checkpoint_sha256=parent)
    result_ref = write(tmp_path/'prior/result.json', result)
    selection = dict(arm='dagger', checkpoint_sha256=parent, diagnostic_result=result_ref)
    plan = dict(source_sha256=prior_scope['source_sha256'], checkpoint_sha256=parent,
                checkpoint_selection=selection, output=str(tmp_path/'test'))
    owners = [{k:o[k] for k in ('physical_gpu','worker_instance_id','gpu_uuid')} for o in prior_scope['owners']]
    trial = dict(panel={}, config={}, owners=owners)
    proof = dict(arm='dagger', checkpoint_sha256=parent, batch_size=16, resource_fixture=True,
                 trials=[deepcopy(trial) for _ in range(4)], tail_trial=dict(panel={},config={},owners=owners[:1]))
    tested = dict(schema='roborender.paired-scale-phase-smoke.v1.result', status='DIAGNOSTIC_COMPLETE',
        online_readback_verified=True, warm_native_owners_retained=True, optimizer_steps=0,
        diagnostic_candidate_tested=True, compute_release_authorized=False, production_campaign_completed=False,
        sampler_status=dict(state='STOPPED_IDLE'), checkpoint_selection=selection,
        qualification_proof=write(tmp_path/'test/proof.json',proof), plan=write(tmp_path/'test/plan.json',plan))
    test_ref = write(tmp_path/'test/result.json', tested)
    idle = [dict(o, source_sha256=prior_scope['source_sha256'], phase='IDLE', active_command_id=None,
                 native_initializations=1) for o in prior_scope['owners']]
    idle_ref = write(tmp_path/'test/final-idle.json',idle)
    new = deepcopy(initial_admission)
    new['mode'] = 'post_test_return'
    new['predecessor'] = dict(diagnostic_result=result_ref, scaled_test_result=test_ref, post_test_idle=idle_ref)
    old = new['scope_admission']
    old['scope'].update(run_id='diagnostic-return', parent_checkpoint_sha256=parent)
    for owner in old['scope']['owners']: owner['command_id'] += '-return'
    old['segment'].update(run_id='diagnostic-return', input_checkpoint_sha256=parent)
    old['scope']['segment_sha256'] = original.digest(old['segment'])
    old['scope_sha256'] = original.digest(old['scope'])
    job.update(run_id='diagnostic-return', input_checkpoint_sha256=parent)
    job['parameters'].update({chain.PARAMETER:new, original.PARAMETER:old})
    checkpoint = dict(schema=original.CHECKPOINT_SCHEMA, global_step=6,
        dagger_run_id=prior_scope['run_id'], dagger_iteration_id=4, dagger_round_id=1,
        production_checkpoint=False, production_promoted=False, selection=chain.SELECTION,
        learner_rng_receipt=chain.read_bound(receipts[-1])['learner_rng_receipt'],
        **{chain.PARAMETER:initial_admission})
    return SimpleNamespace(admission=new, job=job, checkpoint=checkpoint, root=tmp_path,
        initial=initial_admission, result=result, tested=tested, proof=proof, idle=idle,
        result_ref=result_ref, test_ref=test_ref, idle_ref=idle_ref, receipts=receipts)


def test_initial_is_explicit_and_keeps_old_schema():
    value, job = initial()
    assert chain.validate_admission(value, job) == value
    chain.validate_parent_metadata(dict(schema=original.BC_SCHEMA,global_step=2), job)
    assert chain.checkpoint_metadata(job)['schema'] == original.CHECKPOINT_SCHEMA
    assert chain.checkpoint_metadata(job)['production_promoted'] is False


def test_real_return_lineage_uses_new_schema_and_exact_global_steps(return_case):
    case=return_case
    chain.validate_admission(case.admission, case.job)
    chain.validate_parent_metadata(case.checkpoint,case.job)
    assert chain.checkpoint_metadata(case.job)['schema'] == chain.CHECKPOINT_SCHEMA
    for iteration in (2,3,4):
        job=deepcopy(case.job);job['parameters']['iteration_id']=iteration
        checkpoint=dict(case.checkpoint, schema=chain.CHECKPOINT_SCHEMA, global_step=5+iteration,
            dagger_run_id=job['run_id'], dagger_iteration_id=iteration-1,
            learner_rng_receipt=rng(5+iteration,'8'*64,'1'*64,'2'*64),
            **{chain.PARAMETER:case.admission})
        chain.validate_parent_metadata(checkpoint,job)


@pytest.mark.parametrize('mutate',[
    lambda v,j:v.update(schema='legacy'), lambda v,j:v.update(mode='production'),
    lambda v,j:v.update(predecessor={}), lambda v,j:v['execution'].update(native_source_sha256='f'*64),
    lambda v,j:v['execution'].update(learner_source_sha256='c'*64),
    lambda v,j:j['parameters'].pop(chain.RNG_PARAMETER),
    lambda v,j:j['parameters'].update(learner_rng_protocol='legacy'),
    lambda v,j:j['parameters'].update(iteration_id=5),lambda v,j:j.update(round_id=2),
    lambda v,j:j['parameters'].update(microbatch_environments=16),
    lambda v,j:j['parameters'].update(replay_manifest={}),
])
def test_rejects_scope_rng_source_and_budget_expansion(mutate):
    value,job=initial();mutate(value,job)
    with pytest.raises((ValueError,KeyError)):chain.validate_admission(value,job)


@pytest.mark.parametrize('key,value',[
    ('global_step',2),('schema',original.BC_SCHEMA),('production_promoted',True),
    ('dagger_iteration_id',3),('dagger_run_id','different'),('learner_rng_receipt',None),
])
def test_return_rejects_spoofed_bc_stale_rng_or_other_candidate(return_case,key,value):
    return_case.checkpoint[key]=value
    with pytest.raises((ValueError,KeyError)):chain.validate_parent_metadata(return_case.checkpoint,return_case.job)


@pytest.mark.parametrize('kind,key,value',[
    ('learner','optimizer_parameter_states',70),('learner','global_step',9),
    ('learner','diagnostic_learner_chain_admission_sha256','f'*64),
    ('learner','learner_rng_receipt',None),('learner','train_chunks',16),
    ('result','optimizer_job5_absent',False),('result','production_round_complete',True),
    ('tested','diagnostic_candidate_tested',False),('tested','optimizer_steps',1),
    ('tested','online_readback_verified',False),('tested','status','RUNNING'),
    ('idle','phase','RUNNING'),('idle','native_pid',999),('idle','native_initializations',2),
])
def test_actual_evidence_is_rederived_not_just_hash_bound(return_case,kind,key,value):
    case=return_case
    if kind=='learner':
        ref=case.receipts[1];row=chain.read_bound(ref);row[key]=value
        # Rebind every legitimate outer reference so failure is semantic.
        ref.update(write(Path(ref['path']),row))
        qref=case.result['qualification'];q=chain.read_bound(qref);q['learner_receipts']=case.receipts
        qref.update(write(Path(qref['path']),q))
        case.result_ref.update(write(Path(case.result_ref['path']),case.result))
        case.test_ref.update(write(Path(case.test_ref['path']),case.tested))
    elif kind=='result':
        case.result[key]=value;case.result_ref.update(write(Path(case.result_ref['path']),case.result))
        case.test_ref.update(write(Path(case.test_ref['path']),case.tested))
    elif kind=='tested':
        case.tested[key]=value;case.test_ref.update(write(Path(case.test_ref['path']),case.tested))
    else:
        case.idle[0][key]=value;case.idle_ref.update(write(Path(case.idle_ref['path']),case.idle))
    with pytest.raises((ValueError,KeyError)):chain.validate_admission(case.admission,case.job)


def test_real_raw_file_attestation_and_symlink_rejection(tmp_path):
    value,_=initial();file=tmp_path/'worker.py';file.write_text('actual code')
    value['execution']['worker_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    assert chain.validate_runtime_files(value,worker_file=file)['worker_sha256']==value['execution']['worker_sha256']
    alias=tmp_path/'alias.py';alias.symlink_to(file)
    with pytest.raises(ValueError):chain.validate_runtime_files(value,worker_file=alias)
    file.write_text('changed code')
    with pytest.raises(ValueError):chain.validate_runtime_files(value,worker_file=file)


def test_evidence_tampering_missing_and_symlink_fail_closed(tmp_path):
    ref=write(tmp_path/'evidence.json',{'ok':True});assert chain.read_bound(ref)=={'ok':True}
    alias=tmp_path/'alias.json';alias.symlink_to(ref['path'])
    with pytest.raises(ValueError):chain.read_bound(dict(ref,path=str(alias)))
    Path(ref['path']).write_text('{}')
    with pytest.raises(ValueError):chain.read_bound(ref)
    with pytest.raises(ValueError):chain.read_bound(dict(ref,path=str(tmp_path/'missing')))


def test_server_rejects_chain_before_ledger_or_process(tmp_path):
    from pipeline.dagger.services.server import Handler
    from pipeline.dagger.services.contracts import ContractError
    _,job=initial();job['parameters'][chain.PARAMETER]['execution']['server_sha256']='f'*64
    handler=object.__new__(Handler)
    launcher=tmp_path/'run_raven_learner_worker_h200.sh';launcher.write_text('not executed')
    handler.server=SimpleNamespace(worker=launcher)  # No store: admission must fail first.
    handler._read_json=lambda:job
    with pytest.raises(ContractError):handler._learner('POST','/v1/jobs')


def test_controller_scope_is_loaded_lazily_and_exact(tmp_path):
    value,job=initial();native,config=admission()
    scope=tmp_path/'scope/diagnostic-scope.json'
    scope.parent.mkdir();scope.write_bytes(original.canonical_json(native['scope']))
    write(scope.parent/'intent.json',dict(schema='roborender.continuous-diagnostic-intent.v1',
        segment=native['segment'],diagnostic_scope=dict(path=str(scope),raw_sha256=native['scope_sha256'])))
    plan=tmp_path/'chain.json';write(plan,{k:v for k,v in value.items() if k!='scope_admission'})
    job['parameters'].pop(chain.RNG_PARAMETER)
    actual=chain.controller_admission(plan,scope,config,job)
    assert actual==value and job['parameters'][chain.RNG_PARAMETER]==chain.RNG_PROTOCOL
