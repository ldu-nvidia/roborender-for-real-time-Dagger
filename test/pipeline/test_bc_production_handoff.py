"""Synthetic metadata only; these tests never qualify native production assets."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from pipeline.bc_training import assemble_production_protocol as assembler
from pipeline.bc_training import plan_success_test as success
from pipeline.bc_training.handoff_common import binding, bound, canonical, digest
from test.pipeline import test_purpose_aware_admission as fixture
from test.pipeline.test_bc_offline_roles_v3 import make_panel
from pipeline.legacy.bc_collection.sampling_plan import build_offline_plan
from pipeline.bc_training import production_protocol as strict
from pipeline.bc_training import data_admission
from pipeline.deployment.rtx_actor.bc_execution_evidence import build_session_receipts

put = fixture.put

@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-production-handoff')
    with pytest.MonkeyPatch.context() as patch:
        def builder(inv, **kwargs):
            _, panel = make_panel(root, inv)
            return build_offline_plan(inv, seed=kwargs['seed'], num_envs=6, success_panel=panel)
        patch.setattr(fixture, 'build_plan', builder)
        original = fixture.build_pool
        def pool(plan, *args, **kwargs):
            return original(plan, *args, expected_total=plan['total_target_chunks'], **kwargs)
        patch.setattr(fixture, 'build_pool', pool)
        f = fixture.make_complete(root)
    p = f.protocol
    p.update(schema=strict.SCHEMA_V3, closed_loop_test_panel=f.plan['artifact_roles']['closed_loop_test']['panel'])
    source_root = root / 'expected-source'
    assets = {'artifacts': [{'role': ('privileged_teacher' if k == 'teacher' else k), 'sha256': v}
                           for k, v in p['model_sha256'].items()]}
    put(source_root / assembler.ASSETS, assets)
    adapter = source_root / assembler.ADAPTER
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_bytes(b'# synthetic adapter, never executed\n')
    files = [dict(path=name, bytes=(source_root/name).stat().st_size,
                  sha256=digest((source_root/name).read_bytes())) for name in sorted((assembler.ASSETS, assembler.ADAPTER))]
    source = {'files': files, 'source_sha256': digest(canonical(files))}
    source_binding = put(root / 'expected-source-manifest.json', source)
    execution = {**p['collector_execution'], 'source_release_sha256': source['source_sha256'],
                 'adapter_source_sha256': digest(adapter.read_bytes())}
    p['collector_execution'] = execution
    capacities = []
    for size in (6, 1):
        bounded_protocol, receipt = fixture.bounded_capacity_fixture(f,size)
        p['execution_topology']=bounded_protocol['execution_topology']
        p['capacity_equivalence']=bounded_protocol['capacity_equivalence']
        capacities.append(put(root / f'new-capacity-{size}.json', receipt))
    p['capacity_receipts'] = capacities
    cb, tb, eb, sb = [], [], [], []
    docs = [strict.bound(b) for b in p['cohort_manifests']]
    groups=[]
    for doc in docs:
        if (not groups or len(groups[-1])==4
                or len(groups[-1][0]['request']['motions'])!=len(doc['request']['motions'])):
            groups.append([])
        groups[-1].append(doc)
    for group_index,chosen in enumerate(groups):
        sid = f'synthetic-session-{group_index:03}'
        session = dict(session_id=sid, requests=[{'sha256': d['request_sha256']} for d in chosen])
        session_binding = put(root / 'native-sessions' / (sid+'.json'), session)
        launch = dict(collector_execution=execution, source_sha256=source['source_sha256'],
            source_manifest_raw_sha256=source_binding['raw_sha256'], port_manifest_raw_sha256=fixture.h('port'),
            session_sha256=session_binding['raw_sha256'], session_canonical_sha256=digest(canonical(session)),
            request_identities={d['request_sha256']:digest(canonical(d['request'])) for d in chosen},
            qualification_only=False, capacity_binding={**next(b for b in capacities
                if bound(b)['num_envs']==len(chosen[0]['request']['motions'])),
                'path':'/leo/native/capacity.json'}, execution_topology=p['execution_topology'])
        native = dict(status='complete', failure=None, session_id=sid, native_initializations=1,
            scene_initializations=len(chosen), cohorts=[dict(cohort_id=d['request']['cohort_id'],
                request_sha256=d['request_sha256'], output_dir=d['request']['output_dir'],
                vector_steps=d['vector_steps']) for d in chosen])
        host = dict(passed=True, exit_code=0, failure=None)
        session_dir = root / 'pool/manifests/sessions' / sid
        small = {name: put(session_dir/name, doc) for name, doc in
                 (('launch.json',launch),('host-result.json',host),('bc-session-result.json',native))}
        envelopes = []
        for doc in chosen:
            cid = doc['request']['cohort_id']
            audit = root / 'pool/manifests/cohorts' / cid
            old_cb = next(b for b in p['cohort_manifests'] if strict.bound(b)['request']['cohort_id']==cid)
            old_tb = next(b for b in p['transfer_receipts'] if strict.bound(b)['cohort_id']==cid)
            audit.mkdir(parents=True)
            # Byte-exact transfer, so the immutable source-pool hashes do not change.
            (audit/'native-cohort-manifest.json').write_bytes(Path(old_cb['path']).read_bytes())
            (audit/'transfer-receipt.json').write_bytes(Path(old_tb['path']).read_bytes())
            cb.append(binding(audit/'native-cohort-manifest.json'))
            tb.append(binding(audit/'transfer-receipt.json'))
            envelopes.append(dict(document=doc,raw_sha256=cb[-1]['raw_sha256']))
        session_receipt, receipts = build_session_receipts(launch=launch,
            launch_raw_sha256=small['launch.json']['raw_sha256'], session=session,result=native,
            result_raw_sha256=small['bc-session-result.json']['raw_sha256'],host_result=host,
            host_result_raw_sha256=small['host-result.json']['raw_sha256'],cohort_envelopes=envelopes)
        small['session-execution.json'] = put(session_dir/'session-execution.json',session_receipt)
        sb.append(small['session-execution.json'])
        for doc, receipt in zip(chosen,receipts):
            cid=doc['request']['cohort_id']; audit=root/'pool/manifests/cohorts'/cid
            eb.append(put(audit/'execution-receipt.json',receipt))
            entries={**small,'execution-receipt.json':eb[-1]}
            put(audit/'execution-transfer-receipt.json',dict(schema='roborender.bc-execution-metadata-transfer.v1',
                verified=True,cohort_id=cid,session_id=sid,
                cohort_manifest_raw_sha256=receipt['cohort_manifest_raw_sha256'],remote_runtime='/synthetic/'+sid,
                files={name:{**b,'bytes':Path(b['path']).stat().st_size} for name,b in entries.items()},payloads_rehashed=False))
    p.update(cohort_manifests=cb,transfer_receipts=tb,cohort_execution_receipts=eb,session_execution_receipts=sb)
    f.splits={k:v for k,v in f.splits.items() if k!='test'}
    f.args=dict(source_pool=p['source_pool'], sampling_plan=p['sampling_plan'],motion_inventory=p['motion_inventory'],
        native_motion_eligibility=p['native_motion_eligibility'],native_source_root=source_root,
        native_source_manifest=source_binding,render_bindings=put(root/'render-bindings.json',{k:p[k] for k in assembler.RENDER_KEYS}),
        capacity_receipts=capacities)
    return f

def test_assembler_independent_post_render_validator_accepts(complete,tmp_path,monkeypatch):
    real=Path.open
    def no_payload(path,*args,**kwargs):
        assert path.suffix!='.npz', 'dataset tensor opened'
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_payload)
    result=assembler.assemble(**complete.args,output=tmp_path/'protocol.json')
    assert result['metadata_only'] and not result['production_admitted']
    assert result['cohort_count']==100 and result['session_count']==28
    report=data_admission.audit_split_files(complete.manifest,complete.splits,
        require_independent_render=True,purpose='production',run_name='prod-synthetic-handoff',
        protocol_path=result['protocol']['path'],protocol_sha256=result['protocol']['raw_sha256'],
        render_receipt_path=complete.rb['path'],render_receipt_sha256=complete.rb['raw_sha256'])
    assert report['production_admitted'] and report['artifact_roles']['offline_test_chunks']==0

@pytest.mark.parametrize('change',['source','model','capacity','capacity_changed_bytes','missing_mode','changed_test','wrong_plan'])
def test_assembler_rejects_wrong_bound_inputs(complete,tmp_path,change):
    args=deepcopy(complete.args)
    if change=='source': args['native_source_manifest']['raw_sha256']='0'*64
    elif change=='model':
        manifest=bound(args['native_source_manifest']);manifest['files'][0]['sha256']='0'*64
        manifest['source_sha256']=digest(canonical(manifest['files']))
        args['native_source_manifest']=put(tmp_path/'source.json',manifest)
    elif change=='capacity': args['capacity_receipts']=args['capacity_receipts'][:1]
    elif change=='capacity_changed_bytes':
        changed=bound(args['capacity_receipts'][0]);changed['num_envs']=5
        args['capacity_receipts'][0]=put(tmp_path/'changed-capacity.json',changed)
    elif change=='missing_mode':
        inv=bound(args['motion_inventory']);inv['motions'].pop()
        args['motion_inventory']=put(tmp_path/'inventory.json',inv)
    elif change=='changed_test':
        plan=bound(args['sampling_plan']);plan['artifact_roles']['closed_loop_test']['panel']['raw_sha256']='0'*64
        args['sampling_plan']=put(tmp_path/'plan.json',plan)
    else:
        plan=bound(args['sampling_plan']);plan['partitions']['validation']['cohorts'][0]['slots'][0]['selected_chunk_ordinals']=[0]
        args['sampling_plan']=put(tmp_path/'plan.json',plan)
    with pytest.raises((ValueError,KeyError)):
        assembler.assemble(**args,output=tmp_path/'protocol.json')
    assert not (tmp_path/'protocol.json').exists()

@pytest.mark.parametrize('change',['host','session','role','cohort_receipt','missing_transfer'])
def test_assembler_rejects_changed_execution_or_missing_transfer(complete,tmp_path,change):
    root=Path(complete.args['source_pool']['path']).parent
    audit=root/'manifests/cohorts/train-0000'
    handoff=json.loads((audit/'execution-transfer-receipt.json').read_bytes())
    name={'host':'host-result.json','session':'session-execution.json','cohort_receipt':'execution-receipt.json'}.get(change)
    target=Path(handoff['files'][name]['path']) if name else audit/('native-cohort-manifest.json' if change=='role' else 'execution-transfer-receipt.json')
    raw=target.read_bytes()
    try:
        if change=='missing_transfer': target.unlink()
        else:
            doc=json.loads(raw)
            if change=='host': doc['passed']=False
            elif change=='session': doc['source_manifest_raw_sha256']='0'*64
            elif change=='role': doc['request']['split']='validation'
            else: doc['actual_vector_steps']=0
            target.write_bytes(canonical(doc))
        with pytest.raises((ValueError,FileNotFoundError)):
            assembler.assemble(**complete.args,output=tmp_path/'protocol.json')
    finally: target.write_bytes(raw)

def training_fixture(complete,tmp_path):
    result=assembler.assemble(**complete.args,output=tmp_path/'protocol.json')
    report=data_admission.audit_split_files(complete.manifest,complete.splits,
        require_independent_render=True,purpose='production',run_name='prod-synthetic-handoff',
        protocol_path=result['protocol']['path'],protocol_sha256=result['protocol']['raw_sha256'],
        render_receipt_path=complete.rb['path'],render_receipt_sha256=complete.rb['raw_sha256'])
    admission=put(tmp_path/'admission.json',report)
    run=tmp_path/'training';checkpoints=run/'validation_checkpoints';checkpoints.mkdir(parents=True)
    selected=checkpoints/'epoch_0002_step_000032.pt';selected.write_bytes(b'synthetic-checkpoint-not-a-real-policy')
    os.link(selected,run/'best_validation.pt')
    history=[dict(epoch=i,mse=m,checkpoint=str(checkpoints/f'epoch_{i:04}_step_{i*16:06}.pt'))
             for i,m in enumerate([.6,.3,.2,.25])]
    state=put(run/'validation_state.json',dict(status='complete',stop_reason='validation_saturation',
        best_epoch=2,best_mse=.2,selected_checkpoint=str(run/'best_validation.pt'),history=history,
        train_split_sha256=report['split_files']['train']['sha256'],
        validation_split_sha256=report['split_files']['validation']['sha256']))
    config=put(run/'run_config.json',dict(output=str(run),data_admission_json=admission['path'],
        train_split=str(complete.splits['train']),validation_split=str(complete.splits['validation'])))
    return dict(training_state=state,training_config=config,data_admission=admission,
        resident_source_sha256='a'*64,resident_gpu_uuid='GPU-synthetic',resident_port=18700,
        resident_source_root='/resident/frozen',
        resident_token_file='/private/token',resident_jobs_root='/host/jobs',resident_assets_root='/host/assets',
        panel_relative_path='panels/full295.json',command_id='bc-success-synthetic')

def test_selected_best_plan_never_executes_or_loads_model(complete,tmp_path,monkeypatch):
    args=training_fixture(complete,tmp_path)
    # This test isolates the synthetic training handoff. Exact real accepted
    # panel identity is covered separately by production success routing tests.
    from pipeline.deployment.rtx_actor import production_success_contract as contract
    test=bound(args['data_admission'])['artifact_roles']['closed_loop_test']['panel']
    source_panel = bound(test)
    rendered = {**source_panel, 'panel_id': contract.PANEL_ID, 'observation_mode': 'image3f_3view'}
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    import production_success_contract as native_contract
    for selected in (contract, native_contract):
        monkeypatch.setattr(selected, 'LEGACY_PANEL_ID', source_panel['panel_id'])
        monkeypatch.setattr(selected, 'LEGACY_PANEL_RAW_SHA256', test['raw_sha256'])
        monkeypatch.setattr(selected, 'LEGACY_PANEL_SHA256', digest(canonical(source_panel)))
        monkeypatch.setattr(selected, 'PANEL_RAW_SHA256', digest(canonical(rendered) + b'\n'))
        monkeypatch.setattr(selected, 'PANEL_SHA256', digest(canonical(rendered)))
    args['rendered_observation_config'] = put(tmp_path / 'renderer.json', rendering_config(rendered))
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('process launched'))
    result=success.plan(**args,output=tmp_path/'test-plan.json')
    doc=bound(result['plan'])
    assert not doc['executed'] and not doc['success_test_completed']
    assert doc['selected_epoch']==2 and doc['test_case_count']==295
    assert doc['command']['operation']=='evaluate_explicit_panel'
    assert doc['command']['payload']['checkpoint_sha256']==doc['checkpoint']['sha256']
    assert doc['optimizer_steps']==0
    assert bound(doc['command_file'])==doc['command']
    assert doc['submission_argv_on_resident_host'][-1]=='--wait'

@pytest.mark.parametrize('change',['running','not_best','split','checkpoint','wrong_report','unsafe_destination'])
def test_selected_best_plan_rejects_wrong_selection_or_identity(complete,tmp_path,change):
    args=training_fixture(complete,tmp_path)
    if change=='unsafe_destination': args['panel_relative_path']='../escape.json'
    elif change=='wrong_report': args['data_admission']['raw_sha256']='0'*64
    elif change=='checkpoint':
        state=bound(args['training_state']);path=Path(state['selected_checkpoint']);path.unlink();path.write_bytes(b'not-selected')
    else:
        state=bound(args['training_state'])
        if change=='running':state['status']='running'
        elif change=='not_best':state.update(best_epoch=3,best_mse=.25)
        else:state['train_split_sha256']='0'*64
        args['training_state']=put(Path(args['training_state']['path']),state)
    with pytest.raises(ValueError):success.plan(**args,output=tmp_path/'test-plan.json')
    assert not (tmp_path/'test-plan.json').exists()
