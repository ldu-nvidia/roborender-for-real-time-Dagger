"""Synthetic CPU metadata fixtures; never substituted for native GPU evidence."""
from argparse import Namespace
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.bc_training import training_extension as extension, legacy_reuse as legacy
from pipeline.deployment.rtx_actor.isaac6_bc_production import canonical, source_run_tag
from test.pipeline.test_bc_legacy_reuse import inputs, native_complete, admitted
from test.pipeline.test_bc_offline_roles_v3 import put


@pytest.fixture
def extension_inputs(admitted,tmp_path,monkeypatch):
    base_plan,base,base_binding=admitted
    monkeypatch.setattr(extension,'BASE_ADMISSION_SHA256',base_binding['raw_sha256'])
    validation=legacy.bound(base_plan['fresh_validation_plan'])
    rows=[s for c in validation['partitions']['validation']['cohorts'] for s in c['slots']]
    forbidden=set(base_plan['historical']['known_process_seeds'])|set(base['closed_loop_test']['simulator_seeds'])
    forbidden.update(c['process_seed'] for c in validation['partitions']['validation']['cohorts'])
    slots=[{**r,'target_chunks':2 if i<265 else 1,'exposure_span_chunks':9,
        'selected_chunk_ordinals':[1,7] if i<265 else [5]} for i,r in enumerate(rows)]
    cohorts=[]
    for i,start in enumerate(range(0,295,6)):
        group=slots[start:start+6]
        cohorts.append(dict(cohort_id=f'trainext-{i:04d}',split='train',process_seed=1700000000+i,
            num_envs=len(group),slots=group,target_chunks=sum(s['target_chunks'] for s in group)))
    sampling=dict(schema=extension.SAMPLING_SCHEMA,seed=2026101521,validation_plan=base_plan['fresh_validation_plan'],
        legacy_manifest=base_plan['legacy_manifest'],closed_loop_test_panel=base_plan['closed_loop_test_panel'],
        total_target_chunks=560,frames_per_chunk=81,exposure_span_chunks=9,num_envs_ceiling=6,
        selection_uses_labels_rewards_or_loss=False,partitions={'train':dict(cohorts=cohorts)},
        all_forbidden_process_seeds=sorted(forbidden),selection_policy='frozen_fixture_ordinals',
        collection_cost={'selected_chunks':560},motion_inventory=validation['motion_inventory'])
    sample_binding=put(tmp_path/'training560-sampling.json',sampling)
    monkeypatch.setattr(extension,'SAMPLING_PLAN_SHA256',sample_binding['raw_sha256'])
    return dict(run_name='2026-10-16-prod-1-BC',base_admission=base_binding,
        extension_plan=sample_binding,dataset_root=str(tmp_path))


def test_fresh_production_name_preserves_all_data_and_bootstrap_bindings(extension_inputs):
    old=extension.build_plan(**extension_inputs)
    new=extension.build_plan(**{**extension_inputs,'run_name':'10-17-prod-BC'})
    assert new=={**old,'run_name':'10-17-prod-BC'}
    assert extension.validate_plan(new)==new
    assert new['initialization']=='clean_bootstrap_only'


@pytest.mark.parametrize('name',['smoke-not-production','../10-17-prod','a/b','x'*97,''])
def test_fresh_production_name_rejects_unsafe_or_smoke_identity(extension_inputs,name):
    with pytest.raises(ValueError,match='experiment name'):
        extension.build_plan(**{**extension_inputs,'run_name':name})


def training_native_fixture(sampling,root):
    (root/'isaac').mkdir(parents=True)
    cohorts=[]; transfers=[]; sample_sha=legacy.sha(canonical(sampling))
    for ci,c in enumerate(sampling['partitions']['train']['cohorts']):
        request=dict(schema='roborender.bc-native-collection-request.v2',run_id='bc-leo-training560-20261015',
            cohort_id=c['cohort_id'],split='train',process_seed=c['process_seed'],rank=0,global_env_offset=ci*6,
            motions=[dict(motion_id=s['motion_id'],path='/motions/'+s['filename'],sha256=s['motion_sha256']) for s in c['slots']],
            chunks_per_env=[s['target_chunks'] for s in c['slots']],max_windows=31,output_dir='/dagger-runtime/bc_isaac',
            sampling_plan_sha256=sample_sha,selected_chunk_ordinals_per_env=[s['selected_chunk_ordinals'] for s in c['slots']],
            exposure_span_chunks_per_env=[9]*len(c['slots']))
        req_sha=legacy.sha(canonical(request)); rows=[]; files=[]; emitted=[]
        for lane,slot in enumerate(c['slots']):
            stop=slot['selected_chunk_ordinals'][-1]+1; emitted.append(stop)
            for ordinal in range(stop):
                name=f'chunk_fixture_{ci}_{lane}_{ordinal}.npz'; source='/dagger-runtime/bc_isaac/'+name
                raw_sha=legacy.sha(source.encode()); selected=ordinal in slot['selected_chunk_ordinals']
                row=dict(local_env_id=lane,global_env_id=ci*6+lane,source_run_tag=source_run_tag(request),rank=0,
                    seq=ordinal,source_path=source,source_sha256=raw_sha,source_bytes=1,frames=81,
                    cohort_id=c['cohort_id'],motion_id=slot['motion_id'],motion_sha256=slot['motion_sha256'],
                    simulator_seed=c['process_seed'],split='train',sampling_plan_sha256=sample_sha,
                    request_sha256=req_sha,teacher_only=True,episode_id='episode0',
                    first_episode_step_index=ordinal*81,last_episode_step_index=ordinal*81+80,
                    emitted_chunk_ordinal=ordinal,selection_policy='explicit_preselected_ordinals_v2',
                    selected_for_split=selected,overcollection=not selected)
                rows.append(row)
                if selected:
                    destination=root/'isaac'/name; destination.write_bytes(b'x')
                    files.append(dict(source_path=source,source_sha256=raw_sha,destination_sha256=raw_sha,
                        source_bytes=1,destination_bytes=1,destination_path=str(destination)))
        doc=dict(schema='roborender.bc-native-cohort.v1',status='complete',production_admitted=False,
            teacher_only=True,optimizer_steps=0,native_payload_writer_unchanged=True,
            native_collector_sha256=extension.NATIVE_COLLECTOR_SHA256,request=request,request_sha256=req_sha,
            vector_steps=max(emitted)*81,chunks=rows,selected_chunks=sum(request['chunks_per_env']),
            overcollection_chunks=len(rows)-sum(request['chunks_per_env']),emitted_per_lane=emitted,
            model_assets={k:dict(path='/models/'+k,sha256=v) for k,v in
                {'teacher':'1'*64,'homie_walk':'2'*64,'homie_stand':'3'*64}.items()})
        raw_sha=legacy.sha(canonical(doc)); cohorts.append(dict(document=doc,sha256=raw_sha))
        transfer=dict(schema='roborender.bc-cohort-transfer.v1',verified=True,cohort_id=c['cohort_id'],
            cohort_manifest_sha256=raw_sha,remote_runtime='/home/ldu/roborender/runtime/bc-sessions/fixture-'+str(ci),files=files)
        transfers.append(dict(document=transfer,sha256=legacy.sha(canonical(transfer))))
    return cohorts,transfers


@pytest.fixture
def native560(extension_inputs,tmp_path):
    sampling=legacy.bound(extension_inputs['extension_plan']); root=tmp_path/'new560'
    cohorts,transfers=training_native_fixture(sampling,root)
    pool=extension.native_pool(sampling,cohorts,transfers,dataset_root=root)
    return sampling,root,cohorts,transfers,pool


def test_distinct_plan_exact_counts_and_historical_limitations(extension_inputs):
    plan=extension.build_plan(**extension_inputs)
    assert extension.validate_plan(plan)==plan
    assert plan['counts']=={'train':2560,'validation':320} and plan['training_batches_per_epoch']==160
    assert plan['bootstrap_sha256']==legacy.BOOTSTRAP_SHA256
    assert plan['new_training_motion_count']==295 and not plan['historical_motion_frequencies_known']
    assert not plan['production_admitted'] and not plan['training_extension_admitted']


@pytest.mark.parametrize('field,value',[('training_batches_per_epoch',125),('initialization','old_BC'),
    ('production_admitted',True),('historical_motion_frequencies_known',True)])
def test_plan_cannot_change_learning_or_fabricate_provenance(extension_inputs,field,value):
    plan=extension.build_plan(**extension_inputs); plan[field]=value
    with pytest.raises(ValueError,match='differs'): extension.validate_plan(plan)


def test_unknown_prior_admission_is_rejected(extension_inputs):
    extension_inputs['base_admission']={**extension_inputs['base_admission'],'raw_sha256':'f'*64}
    with pytest.raises(ValueError,match='immutable2000'): extension.build_plan(**extension_inputs)


def test_all_native560_requests_quota_and_chronology_reconstructed(native560):
    sampling,root,cohorts,transfers,pool=native560
    assert pool['chunk_count']==560 and pool['motion_count']==295 and not pool['payloads_read']
    assert len(pool['overcollection_excluded'])>560
    assert Counter(r['split'] for r in pool['chunks'])=={'train':560}
    assert len({r['source_sha256'] for r in pool['chunks']})==560


@pytest.fixture
def native_execution_case(extension_inputs,native560,tmp_path,monkeypatch):
    from test.pipeline.test_bc_capacity_reset_equivalence import fixture as capacity_fixture
    sampling,root,cohorts,transfers,pool=native560
    plan=extension.build_plan(**extension_inputs)
    source_pool=put(root/'manifest.json',pool)
    source=dict(files=[dict(path='pipeline/deployment/rtx_actor/isaac6_bc_production.py',sha256='a'*64)])
    source['source_sha256']=legacy.sha(canonical(source['files']))
    monkeypatch.setattr(extension,'NATIVE_SOURCE_SHA256',source['source_sha256'])
    source_binding=put(root/'native-source.json',source)
    execution=dict(adapter_source_sha256='a'*64,source_release_sha256=source['source_sha256'],
        container_image_id='sha256:'+'c'*64,native_runtime_version='CPU synthetic fixture',
        lifecycle_recipe='one_kit_fresh_seeded_scene_per_declared_cohort_v1',
        memory_buffer_recipe='original_raven_capture_lists_and_writer_v1',runtime_contract_sha256='d'*64)
    protocol,_=capacity_fixture(); topology=protocol['execution_topology']
    capacities=[]
    models={k:v['sha256'] for k,v in pool['model_assets'].items()}
    for size in (1,6):
        _,receipt=capacity_fixture()
        receipt.update(num_envs=size,collector_execution=execution,model_sha256=models)
        for index,row in enumerate(receipt['per_flush_memory']): row['emitted_chunks_per_lane']=[index]*size
        capacities.append(put(root/f'capacity-{size}.json',receipt))
    session_map={}; cohort_sessions={}
    ordered=[49]+list(range(49))
    for ci in ordered:
        sid='tail-session' if ci==49 else 'wide-session-'+str(ci//4)
        cohort_sessions[f'trainext-{ci:04d}']=sid
        session_map.setdefault(sid,[]).append(ci)
    session_bindings={sid:put(root/f'{sid}.json',dict(session_id=sid,scene_initializations=len(indices),
        collector_execution=execution)) for sid,indices in session_map.items()}
    for c,t in zip(cohorts,transfers):
        cid=c['document']['request']['cohort_id']; audit=root/'manifests/cohorts'/cid; audit.mkdir(parents=True)
        # Native fixture envelopes above hash these exact canonical bytes.
        (audit/'native-cohort-manifest.json').write_bytes(canonical(c['document']))
        (audit/'transfer-receipt.json').write_bytes(canonical(t['document']))
    def chain(root,cb,*args):
        doc=extension.bound(cb); sid=cohort_sessions[doc['request']['cohort_id']]
        cap=capacities[0 if len(doc['request']['motions'])==1 else 1]
        return {'session-execution.json':session_bindings[sid]}, {
            'session-execution.json':extension.bound(session_bindings[sid]),
            'launch.json':dict(execution_topology=topology,capacity_binding=cap)}
    monkeypatch.setattr(extension,'execution_chain',chain)
    seal=put(root/'SEALED.json',dict(schema='roborender.bc-source-seal.v2',verdict='PASS',isaac_chunks=560,
        rendered_chunks=0,sampling_plan_sha256=legacy.sha(canonical(sampling)),payloads_read=False,
        qualification_scope='native provenance/exact560/verified transfer; render/admission pending',
        manifest_sha256=source_pool['raw_sha256']))
    evidence=dict(schema='roborender.training560-native-execution.v1',source_pool=source_pool,source_seal=seal,
        sampling_plan=extension_inputs['extension_plan'],native_source_manifest=source_binding,
        session_execution_receipts=list(session_bindings.values()),collector_execution=execution,
        execution_topology=topology,capacity_receipts=capacities,selected_chunks=560,model_sha256=models)
    return plan,sampling,root,pool,evidence


def test_native_execution_summary_rebuilt_with_different_session_order(native_execution_case):
    plan,sampling,root,pool,evidence=native_execution_case
    # Producer uses B1-first session order, sampling plan lists B6 cohorts first.
    assert extension.native_execution(put(root/'native-execution.json',evidence),sampling,plan)==pool


@pytest.mark.parametrize('fault',['missing_session','duplicate_capacity','wrong_model_summary','bad_source_seal'])
def test_native_execution_cannot_borrow_or_omit_evidence(native_execution_case,fault):
    plan,sampling,root,pool,evidence=native_execution_case
    if fault=='missing_session': evidence['session_execution_receipts'].pop()
    elif fault=='duplicate_capacity': evidence['capacity_receipts'].append(evidence['capacity_receipts'][0])
    elif fault=='wrong_model_summary': evidence['model_sha256']['teacher']='f'*64
    else: evidence['source_seal']=put(root/'bad-seal.json',dict(verdict='PASS'))
    with pytest.raises(ValueError):
        extension.native_execution(put(root/'native-execution-bad.json',evidence),sampling,plan)


@pytest.mark.parametrize('fault',['missing_cohort','changed_ordinal','wrong_homie','reused_source','wrong_transfer','bad_episode'])
def test_native560_changed_membership_or_labels_fail(native560,fault):
    sampling,root,cohorts,transfers,pool=native560
    if fault=='missing_cohort': cohorts.pop()
    elif fault=='changed_ordinal': cohorts[0]['document']['request']['selected_chunk_ordinals_per_env'][0]=[0,7]
    elif fault=='wrong_homie': cohorts[-1]['document']['model_assets']['homie_walk']['sha256']='f'*64
    elif fault=='reused_source': cohorts[0]['document']['chunks'][2]['source_sha256']=cohorts[0]['document']['chunks'][0]['source_sha256']
    elif fault=='wrong_transfer': transfers[0]['document']['files'][0]['destination_sha256']='f'*64
    else: cohorts[0]['document']['chunks'][1]['first_episode_step_index']=0
    with pytest.raises(ValueError): extension.native_pool(sampling,cohorts,transfers,dataset_root=root)


@pytest.fixture
def ready(extension_inputs,native560,tmp_path,monkeypatch):
    from pipeline.bc_training import render_execution
    sampling,root,cohorts,transfers,pool=native560
    plan=extension.build_plan(**extension_inputs); plan_binding=put(tmp_path/'extension-plan.json',plan)
    source=put(root/'manifest.json',pool)
    evidence=put(root/'native-execution.json',dict(source_pool=source))
    monkeypatch.setattr(extension,'native_execution',lambda *args:pool)
    folder=root/'rendered'; folder.mkdir()
    rows=[]
    for row in pool['chunks']:
        name=row['chunk_id']+'.npz'; (folder/name).write_bytes(b'r')
        rows.append({**row,'rendered_npz':'rendered/'+name,'rendered_sha256':legacy.sha(('new-'+name).encode()),
            'rendered_size_bytes':1,'render_contract':'raven_three_view_independent_v2',
            'render_chain_length':1,'render_chain_index':0,'render_inherits_previous_reference':False})
    rendered=dict(schema='roborender.bc_pool.rendered.v2',frames_per_chunk=81,
        render_contract='raven_three_view_independent_v2',render_recipe='raven_3view_50step_independent_v2',
        max_chain_chunks=1,rendered_pool=str(folder),chunks=rows)
    render_binding=put(root/'rendered.json',rendered)
    receipt=dict(source_pool_raw_sha256=source['raw_sha256'])
    rendering=dict(protocol=put(root/'render-protocol.json',dict(source_pool=source)),
        render_plan=put(root/'render-plan.json',{}),rendered_manifest=render_binding,
        execution_receipt=put(root/'render-execution.json',receipt))
    prior=render_execution.finish
    monkeypatch.setattr(render_execution,'finish',lambda p,l,m:receipt if str(m)==render_binding['path'] else prior(p,l,m))
    return plan,plan_binding,evidence,rendering,pool


def test_reuse_pipeline_revalidates_data_without_render_or_resume(ready,tmp_path):
    from pipeline.bc_training import reuse_pipeline
    plan,plan_ref,evidence,rendering,_=ready
    admitted=extension.publish_admission(plan_binding=plan_ref,native_evidence=evidence,
        rendering=rendering,output_dir=tmp_path/'fresh-admission')
    job=reuse_pipeline.plan(admission=admitted['admission'],run_name=plan['run_name'],
        source_root=str(tmp_path/'source'),asset_root=str(tmp_path/'assets'),runtime_root=str(tmp_path/'execution'),
        max_epochs=100,min_epochs=10,patience=8,seed=42)
    assert job['counts']=={'train':2560,'validation':320}
    assert job['initialization']=='clean_bootstrap_only' and job['rerender'] is False
    assert len(job['stages'])==1
    argv=job['stages'][0]['argv']
    assert '--allow-training-extension' in argv and '--resume-checkpoint' not in argv
    assert job['env']['WANDB_MODE']=='online'
    calls=[]
    result=reuse_pipeline.execute(job,runner=lambda *args,**kwargs:calls.append((args,kwargs)))
    assert result['status']=='complete' and len(calls)==1
    with pytest.raises(FileExistsError):reuse_pipeline.execute(job,runner=lambda *a,**k:None)


def test_reuse_pipeline_rejects_different_run_name(ready,tmp_path):
    from pipeline.bc_training import reuse_pipeline
    plan,plan_ref,evidence,rendering,_=ready
    admitted=extension.publish_admission(plan_binding=plan_ref,native_evidence=evidence,
        rendering=rendering,output_dir=tmp_path/'fresh-admission')
    with pytest.raises(ValueError,match='invocation differs'):
        reuse_pipeline.plan(admission=admitted['admission'],run_name='wrong-run',
            source_root=str(tmp_path/'source'),asset_root=str(tmp_path/'assets'),runtime_root=str(tmp_path/'execution'),
            max_epochs=100,min_epochs=10,patience=8,seed=42)


def test_additive_indexes_leave_base_files_unchanged_and_allow_exact_resume(ready,tmp_path):
    plan,pb,evidence,rendering,pool=ready
    base=extension.bound(plan['base_admission'])
    old={key:Path(b['path']).read_bytes() for key,b in base['split_files'].items()}
    result=extension.publish_admission(plan_binding=pb,native_evidence=evidence,rendering=rendering,
        output_dir=tmp_path/'merged-admission')
    assert result['counts']=={'train':2560,'validation':320} and result['training_batches_per_epoch']==160
    assert all(Path(base['split_files'][k]['path']).read_bytes()==v for k,v in old.items())
    report=extension.bound(result['admission'])
    assert report['training_extension_admitted'] and not report['production_admitted']
    kwargs=dict(allow_training_extension=True,run_name=plan['run_name'],bootstrap_path=plan['bootstrap_path'],
        policy_spec_path=plan['bootstrap_spec']['path'],dataset_root=plan['dataset_root'])
    assert extension.validate_training_admission(result['admission']['path'],result['split_files']['train']['path'],
        result['split_files']['validation']['path'],**kwargs)==report
    identity={**report,'report_sha256':result['admission']['raw_sha256']}
    extension.validate_resume_identity(dict(data_admission=identity),identity)
    with pytest.raises(ValueError,match='same admitted'):
        extension.validate_resume_identity(dict(data_admission={**identity,'report_sha256':'f'*64}),identity)
    with pytest.raises(FileExistsError):
        extension.publish_admission(plan_binding=pb,native_evidence=evidence,rendering=rendering,output_dir=tmp_path/'merged-admission')


@pytest.mark.parametrize('fault',['raw_leak','scoped_episode_leak','seed_leak','render_leak','chain_leak'])
def test_validation_and_test_leakage_is_rejected(ready,fault,tmp_path):
    plan,pb,evidence,rendering,pool=ready
    base=extension.bound(plan['base_admission']); val=extension.bound(base['native_evidence']['source_pool'])
    rendered=extension.bound(rendering['rendered_manifest'])
    if fault=='raw_leak': pool['chunks'][0]['source_sha256']=val['chunks'][0]['source_sha256']
    elif fault=='scoped_episode_leak':
        for k in ('source_run_tag','rank','local_env_id','episode_id'): pool['chunks'][0][k]=val['chunks'][0][k]
    elif fault=='seed_leak': pool['chunks'][0]['simulator_seed']=base['closed_loop_test']['simulator_seeds'][0]
    elif fault=='render_leak': rendered['chunks'][0]['rendered_sha256']=extension.bound(base['rendering']['validation']['rendered_manifest'])['chunks'][0]['rendered_sha256']
    else:
        v=extension.bound(base['rendering']['validation']['rendered_manifest']); v['chunks'][0]['render_chain_id']='same-chain'
        base['rendering']['validation']['rendered_manifest']=put(tmp_path/'changed-val-render.json',v)
        rendered['chunks'][0]['render_chain_id']='same-chain'
    with pytest.raises(ValueError,match='leak|duplicates'):
        extension.assert_independence(pool,base,rendered)


def test_bare_native_episode_names_are_not_global_ids(ready):
    plan,pb,evidence,rendering,pool=ready
    base=extension.bound(plan['base_admission']); rendered=extension.bound(rendering['rendered_manifest'])
    val=extension.bound(base['native_evidence']['source_pool'])
    pool['chunks'][0]['episode_id']=val['chunks'][0]['episode_id']
    extension.assert_independence(pool,base,rendered)


def test_render_weights_source_or_prompt_cannot_drift_from_base(ready,tmp_path):
    plan,pb,evidence,rendering,pool=ready
    receipt=extension.bound(rendering['execution_receipt'])
    receipt['model_sha256']={'adapter':'f'*64}
    rendering['execution_receipt']=put(tmp_path/'different-render-weights.json',receipt)
    with pytest.raises(ValueError,match='renderer model/source/prompt'):
        extension.publish_admission(plan_binding=pb,native_evidence=evidence,rendering=rendering,
            output_dir=tmp_path/'different-render-not-admitted')
    assert not (tmp_path/'different-render-not-admitted').exists()


@pytest.mark.parametrize('field,value',[('render_chain_length',6),('render_chain_index',1),
    ('render_inherits_previous_reference',True)])
def test_new_training_cannot_adopt_chained_rendering(ready,field,value,tmp_path):
    plan,pb,evidence,rendering,pool=ready
    rendered=extension.bound(rendering['rendered_manifest'])
    rendered['chunks'][0][field]=value
    rendering['rendered_manifest']=put(Path(rendering['rendered_manifest']['path']),rendered)
    with pytest.raises(ValueError,match='lineage'):
        extension.publish_admission(plan_binding=pb,native_evidence=evidence,rendering=rendering,
            output_dir=tmp_path/'not-admitted')
    assert not (tmp_path/'not-admitted').exists()


def test_explicit_training_extension_dispatch_and_no_other_schema_bypass(monkeypatch,tmp_path):
    from pipeline.bc_training import train_bc_h200 as trainer
    report=dict(schema=extension.ADMISSION_SCHEMA,training_extension_admitted=True,production_admitted=False)
    path=tmp_path/'admission.json'; put(path,report)
    args=Namespace(data_admission_json=path,require_production_data_admission=True,allow_qualification_data=False,
        allow_legacy_reuse=False,allow_training_extension=True,run_name='2026-10-16-prod-1-BC',
        train_split=tmp_path/'train',validation_split=tmp_path/'val',bootstrap=tmp_path/'bootstrap',
        policy_spec=tmp_path/'spec',resume_checkpoint=None,resume_validation_state=None,dataset_root=tmp_path)
    captured={}
    def admit(*a,**kwargs):
        captured.update(kwargs)
        if not kwargs['allow_training_extension']: raise ValueError('explicit opt-in required')
        return report
    monkeypatch.setattr(extension,'validate_training_admission',admit)
    result=trainer.load_data_admission(args)
    assert result['status']=='training_extension_admitted' and captured['allow_training_extension']
    trainer.validate_resume_data_admission(dict(data_admission=result),result)
    args.allow_legacy_reuse=True
    with pytest.raises(ValueError,match='another admission'): trainer.load_data_admission(args)
    args.allow_legacy_reuse=False; args.allow_training_extension=False
    with pytest.raises(ValueError,match='explicit opt-in'): trainer.load_data_admission(args)
    put(path,dict(schema=legacy.ADMISSION_SCHEMA)); args.allow_training_extension=True
    with pytest.raises(ValueError,match='exact distinct'): trainer.load_data_admission(args)


def test_fresh_native_admission_cannot_accept_extension_report(tmp_path):
    from pipeline.bc_training.data_admission import validate_training_admission
    path=tmp_path/'extension.json'; put(path,dict(schema=extension.ADMISSION_SCHEMA,training_extension_admitted=True))
    with pytest.raises(ValueError,match='unknown admission'):
        validate_training_admission(path,tmp_path/'train',tmp_path/'validation')
