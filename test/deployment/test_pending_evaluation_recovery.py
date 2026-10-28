from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from pipeline.dagger.beta_campaign import Campaign
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids,canonical
from pipeline.dagger.services.contracts import file_sha256
from test.deployment.test_round_boundary_recovery import boundary,insert
from pipeline.deployment.pending_evaluation_recovery import ledger_evidence,publish,validate,validate_binding
from pipeline.deployment.round_boundary_recovery import digest


@pytest.fixture
def pending(boundary):
    b=boundary;s=b.state;rid=s['round_id'];config=s['config'];run=s['run_id']
    s['config_path']=str(b.root/f'round-{rid}/config.json');Path(s['config_path']).parent.mkdir()
    Path(s['config_path']).write_bytes(canonical(config))
    rows=[];results=[];parent=config['student']['initial_checkpoint_sha256']
    with sqlite3.connect(b.root/'services/state.sqlite3') as db:
        for i in range(1,4):
            path=b.root/f'current-{i}.pt';path.write_bytes(f'unit checkpoint {i}'.encode());sha=file_sha256(path)
            row={'run_id':run,'round_id':rid,'iteration_id':i,'input_checkpoint_sha256':parent,'candidate_checkpoint_sha256':sha,
                'status':'validating' if i==3 else 'advanced','created_at':1,'updated_at':1}
            insert(db,'round_iterations',**row);rows.append(row)
            result={'iteration_id':i,'parent_checkpoint_sha256':parent,'output_checkpoint_sha256':sha,
                'output_checkpoint_path':str(path),'train_chunks':12,'train_frames':972,'replay_manifest_sha256':None}
            results.append(result)
            insert(db,'learner_jobs',job_id=f'current-{i}',run_id=run,round_id=rid,input_checkpoint_sha256=parent,
                chunk_ids_json='[]',status='succeeded',request_json='{}',result_json=json.dumps(result),log_path='unit',created_at=1,updated_at=1)
            for a in range(12):
                insert(db,'iteration_barriers',run_id=run,round_id=rid,iteration_id=i,actor_id=f'actor{a}',
                    checkpoint_sha256=sha,chunk_ids_json='[]',created_at=1)
            parent=sha
        for i in range(2916):
            key=f'prepare-{i}'
            insert(db,'step_prepares',idempotency_key=key,request_sha256='x',run_id=run,round_id=rid,actor_id=f'a{i%12}',
                rollout_window_id=f'w{i//972}',rollout_step_index=i//12%81,episode_id='e',episode_step_index=i,
                capture_digest='x',prepare_receipt='x',metadata_json='{}',response_json='{}',artifact_path='unit',
                checkpoint_sha256=sha,status='committed',created_at=1,updated_at=1)
            insert(db,'step_commits',idempotency_key=f'commit-{i}',request_sha256='x',prepare_idempotency_key=key,
                transition_json='{}',created_at=1,updated_at=1)
        insert(db,'success_evaluations',evaluation_id='pending',run_id=run,round_id=rid,purpose='validation',
            checkpoint_sha256=sha,scenario_manifest_sha256=config['validation']['scenario_manifest_sha256'],
            scenario_ids_json='[]',observation_mode='raytraced',metric='raven_goal_reached_rate',status='collecting',created_at=1,updated_at=1)
        db.execute("UPDATE runs SET current_round=?,status='validating' WHERE run_id=?",(rid,run))
    barrier=b.evidence['native_phase_barrier']
    receipt={'actor_ids':actor_ids(config),'native_phase_barrier':barrier,'actors':[
        {'physical_gpu':g,'run_id':run,'round_id':rid,'phase':'COMPLETE','uncertain_reason':None,'lifecycle_initializations':1,
            'transition_batches':243,'physical_transition_rows':1458,'final_candidate_checkpoint_sha256':sha,'config_sha256':digest(config),
            'windows':[{'index':r['iteration_id']-1,'input_checkpoint_sha256':r['input_checkpoint_sha256'],
                'output_checkpoint_sha256':r['candidate_checkpoint_sha256']} for r in rows]} for g in (0,1)]}
    directory=b.root/f'operations/r{rid}-collect-unit';directory.mkdir(parents=True)
    (directory/'receipt.json').write_bytes(canonical(receipt))
    s.update(final_checkpoint={'path':str(path),'sha256':sha},final_learner=results[-1],operation_id='old-evaluate')
    f={'phase':'EVALUATE','native_restart_allowed':False,'error_type':'RuntimeError'}
    e={'checked_at':1000.,'allocation_absent':True,'management_owners_stopped':True,'native_phase_barrier':barrier,
        'native_evaluation_commands':0,'evaluation_artifacts_absent':True,'ledger':ledger_evidence(b.root,s)}
    (b.root/'state.json').write_bytes(canonical(s));(b.root/'failure.json').write_bytes(canonical(f))
    return SimpleNamespace(**vars(b),failure_eval=f,evidence_eval=e)


def test_evaluation_only_admission_preserves_all_training(pending,monkeypatch):
    b=pending;monkeypatch.setattr('pipeline.deployment.pending_evaluation_recovery.time.time',lambda:1001.)
    old=(b.root/'state.json').read_bytes();database=(b.root/'services/state.sqlite3').read_bytes()
    publish(b.root,b.state,b.failure_eval,b.evidence_eval,{},'e'*32)
    s=json.loads((b.root/'state.json').read_bytes())
    assert s['phase']=='EVALUATE' and s['operation_id'] is None
    assert s['final_checkpoint']==b.state['final_checkpoint'] and s['completed_rounds']==b.state['completed_rounds']
    assert (b.root/'services/state.sqlite3').read_bytes()==database
    assert (b.root/'recoveries'/('e'*32)/'prior-state.json').read_bytes().strip()==old.strip()
    campaign=Campaign(b.first,b.root,SimpleNamespace())
    assert campaign.operation_context()['evaluation_attempt']=='e'*32
    assert campaign.state['config_path']==b.state['config_path']
    changed=deepcopy(s);changed['final_checkpoint']['sha256']='0'*64
    with pytest.raises(RuntimeError):validate_binding(b.root,changed)


@pytest.mark.parametrize('bad',['native_command','artifact','owner','allocation','stale','generation','partial_training','outcome','checkpoint'])
def test_started_evaluation_or_uncertain_training_never_replayed(pending,bad):
    b=pending;s=deepcopy(b.state);e=deepcopy(b.evidence_eval)
    if bad=='native_command':e['native_evaluation_commands']=1
    if bad=='artifact':e['evaluation_artifacts_absent']=False
    if bad=='owner':e['management_owners_stopped']=False
    if bad=='allocation':e['allocation_absent']=False
    if bad=='stale':e['checked_at']=1
    if bad=='generation':e['native_phase_barrier']['workers'][0]['worker_instance_id']='f'*32
    if bad=='checkpoint':Path(s['final_checkpoint']['path']).write_bytes(b'wrong')
    if bad in ('partial_training','outcome'):
        with sqlite3.connect(b.root/'services/state.sqlite3') as db:
            if bad=='partial_training':db.execute("UPDATE step_prepares SET status='prepared' WHERE idempotency_key='prepare-0'")
            else:insert(db,'success_evaluation_outcomes',evaluation_id='pending',scenario_id='started',idempotency_key='started',
                request_json='{}',goal_reached=0,reward=0.,steps=1,termination='failure',safety_violation=0,created_at=1)
    with pytest.raises(RuntimeError):validate(b.root,s,b.failure_eval,e,now=1001.)


def test_completed_logging_replaces_the_registration_only_startup_flag(boundary,monkeypatch):
    import sys
    from pipeline.dagger.beta_campaign import online_loggers
    checks=[];logged=[]
    class Run:
        url='https://wandb.invalid/unit';summary={}
        def log(self,values):logged.append(values);self.summary.update(values)
        def finish(self):pass
    run=Run()
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **kwargs:run))
    monkeypatch.setattr('pipeline.deployment.wandb_preflight.admit_logging',lambda *args:(object(),{'entity':'nvidia'}))
    monkeypatch.setattr('pipeline.deployment.wandb_preflight.verify_remote_summary',lambda api,entity,rid,values:checks.append(values))
    logger=online_loggers('nvidia')[1]
    logger(boundary.first,{'train_mse':.5},{'goal_reached_rate':.1,'successes':1,'checkpoint_sha256':'a'*64})
    assert run.summary['orchestration/collection_started'] is True
    assert run.summary['orchestration/round_complete'] is True
    assert checks[-1]['orchestration/round_complete'] is True
    assert logged[-1]['validation/successes']==1
