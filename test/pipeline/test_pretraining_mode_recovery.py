"""CPU recovery gate: never retrain BC, reuse partial data, or replay actors."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import pytest
from pipeline.bc_training.handoff_common import binding,publish
from pipeline.deployment.pretraining_mode_recovery import (adopt_completed_bc_test,compare_plans,ZERO_TABLES,
    recovery_run_id,compact_recovery_prefix)
from pipeline.deployment.mode_execution import execute


class Adapter:
    def __init__(self):self.calls=[]
    def validate(self,phase,receipt,completed):
        assert receipt.get('fixture') is True
    def dagger(self,completed):
        self.calls.append('dagger');return {'fixture':True}


def rename_recovery_plan(path,run):
    plan=json.loads(path.read_bytes());plan['run_id']=run
    plan['settings']['run_id']=run
    plan['stage_names']={key:run+'-'+key for key in plan['stage_names']}
    coverage_path=Path(plan['settings']['coverage_plan']['path'])
    coverage=json.loads(coverage_path.read_bytes());coverage['run_id']='dagger-leo-'+run
    for row in coverage['rounds']:
        row['wandb_run_name']=run+'-r'+str(row['round_id'])
        for epoch in row['epochs']:
            for cohort in epoch['cohorts']:
                for assignment in cohort['assignments']:
                    assignment['run_id']=coverage['run_id']
    coverage_path.write_text(json.dumps(coverage));plan['settings']['coverage_plan']=binding(coverage_path)
    path.write_text(json.dumps(plan))


def test_repeated_recovery_names_remain_bounded_and_parent_specific():
    previous='10-17-prod-recovery-v13-recovery-v14'
    for index in range(100):
        current=recovery_run_id(previous,'v'+str(index))
        assert 2<=len(current)<=40 and current!=previous
        assert current==recovery_run_id(previous,'v'+str(index))
        assert current!=recovery_run_id(previous+'x' if len(previous)<40 else previous[:-1]+'x','v'+str(index))
        previous=current
    assert recovery_run_id('prod','v3')=='prod-recovery-v3'


@pytest.mark.parametrize('previous,label',[('','v1'),('a'*41,'v1'),('../prod','v1'),('prod',''),('prod','UPPER'),('prod','a'*9),('prod','../x')])
def test_recovery_names_reject_invalid_inputs(previous,label):
    with pytest.raises(ValueError):recovery_run_id(previous,label)


def test_compact_recovery_keeps_full_previous_state_binding(tmp_path):
    p,s,old,e=fixture(tmp_path)
    rename_recovery_plan(p,compact_recovery_prefix('prod')+'v16')
    previous=binding(old);result=adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=Adapter())
    assert result['recovery']['previous_state']==previous
    assert result['run_id']==compact_recovery_prefix('prod')+'v16'


@pytest.mark.parametrize('bad',['foreign','empty','too_long'])
def test_compact_recovery_rejects_wrong_parent_or_label(tmp_path,bad):
    p,s,old,e=fixture(tmp_path)
    run=compact_recovery_prefix('another' if bad=='foreign' else 'prod')+('' if bad=='empty' else 'x'*9 if bad=='too_long' else 'v16')
    rename_recovery_plan(p,run)
    with pytest.raises(ValueError):adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=Adapter())


@pytest.mark.parametrize('fault',['seed','settings_identity','coverage_identity'])
def test_compact_recovery_does_not_relax_science_or_cross_identity(tmp_path,fault):
    p,s,old,e=fixture(tmp_path);rename_recovery_plan(p,compact_recovery_prefix('prod')+'v16')
    plan=json.loads(p.read_bytes())
    if fault=='seed':plan['settings']['seed']+=1
    elif fault=='settings_identity':plan['settings']['run_id']='other'
    else:
        path=Path(plan['settings']['coverage_plan']['path']);coverage=json.loads(path.read_bytes())
        coverage['run_id']='dagger-leo-other';path.write_text(json.dumps(coverage))
        plan['settings']['coverage_plan']=binding(path)
    p.write_text(json.dumps(plan))
    with pytest.raises(ValueError):adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=Adapter())


def fixture(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    def save(path,value):return publish(path,value)
    coverage=dict(run_id='dagger-leo-prod',source_sha256='a',seed=42,
        rounds=[dict(round_id=i,wandb_run_id=f'old{i}',wandb_run_name=f'old{i}',seed=42+i,
            epochs=[{'cohorts':[{'assignments':[{'run_id':'dagger-leo-prod','seed':123}]}]}]) for i in range(1,5)])
    c0=save(old/'coverage.json',coverage)
    coverage=deepcopy(coverage);coverage.update(run_id='dagger-leo-prod-recovery-v3',source_sha256='b')
    for row in coverage['rounds']:
        row.update(wandb_run_id='new'+str(row['round_id']),wandb_run_name='new'+str(row['round_id']))
        row['epochs'][0]['cohorts'][0]['assignments'][0]['run_id']=coverage['run_id']
    c1=save(new/'coverage.json',coverage)
    panel=save(tmp_path/'panel.json',dict(cases=295))
    p=dict(schema='roborender.pipeline-mode.v1',mode='production',run_id='prod',output_root=str(old),
        source_root='/old',source_sha256='a',test_panel=panel,bc={'epochs':100},dagger={'beta':.75},
        settings=dict(run_id='prod',source_root='/old',source_sha256='a',seed=42,coverage_plan=c0),
        stage_names={'BC':'prod-BC'})
    oldplan=save(old/'plan.json',p);p=deepcopy(p)
    p.update(run_id='prod-recovery-v3',output_root=str(new),source_root='/new',source_sha256='b',
        stage_names={'BC':'prod-recovery-v3-BC'})
    p['settings'].update(run_id=p['run_id'],source_root='/new',source_sha256='b',coverage_plan=c1)
    newplan=save(new/'plan.json',p)
    settings=dict(retained={'identity':'same'},compute_release_authorized=False)
    s0=save(old/'settings.json',settings);s1=save(new/'settings.json',settings)
    parent='d'*64;completed=[]
    for phase in ('prepare_bc','train_bc','test_bc'):
        ref=save(old/(phase+'.json'),dict(fixture=True,checkpoint={'sha256':parent},checkpoint_sha256=parent))
        completed.append(dict(phase=phase,receipt=ref))
    state=dict(schema='roborender.mode-execution.v1',mode='production',run_id='prod',
        phase='dagger',status='RECONCILE_REQUIRED',completed=completed,plan=oldplan,settings=s0)
    oldstate=save(old/'state.json',state)
    campaign=save(old/'campaign.json',dict(run_id='dagger-leo-prod',round_id=1,phase='COLLECT_HELDOUT',
        completed_rounds=[],config={'student':{'initial_checkpoint_sha256':parent}}))
    workers=[]
    for gpu in (0,1):
        journal=save(old/f'worker{gpu}.json',dict(source_sha256='a',worker_instance_id=str(gpu),
            commands={'x':{'command':{'operation':'collect_mse_validation'},'status':'RUNNING'}}))
        workers.append(dict(physical_gpu=gpu,worker_instance_id=str(gpu),container_absent=True,
            unit_state='inactive',journal=journal))
    dbpath=old/'state.sqlite3'
    with sqlite3.connect(dbpath) as db:
        for table in ZERO_TABLES:db.execute('CREATE TABLE '+table+' (id TEXT)')
        db.executescript('CREATE TABLE checkpoints(sha256 TEXT); CREATE TABLE runs(run_id TEXT,status TEXT,active_checkpoint_sha256 TEXT); CREATE TABLE round_iterations(run_id TEXT,round_id INT,iteration_id INT,status TEXT,candidate_checkpoint_sha256 TEXT); CREATE TABLE validation_collections(run_id TEXT,status TEXT);')
        db.execute('INSERT INTO checkpoints VALUES(?)',(parent,))
        db.execute('INSERT INTO runs VALUES(?,?,?)',('heldout-dagger-leo-prod-r1','collecting',parent))
        db.execute('INSERT INTO round_iterations VALUES(?,?,?,?,?)',('heldout-dagger-leo-prod-r1',1,1,'collecting',None))
        db.execute('INSERT INTO validation_collections VALUES(?,?)',('heldout-dagger-leo-prod-r1','collecting'))
    e=dict(schema='roborender.pretraining-reconciliation.v1',previous_state=oldstate,campaign_state=campaign,
        compute_release_authorized=False,controller_status={'state':'FAILED'},workers=workers,
        services=[dict(host=h,label=l,status={'state':'STOPPED'}) for h,l in [('beta','tunnels'),('local','tunnels'),('gpu','gpu-services')]],
        retained_before={'allocation_retained':True},retained_after={'allocation_retained':True},
        closed_ledger=dict(database=binding(dbpath),old_population_excluded=True,
            counts={**{t:0 for t in ZERO_TABLES},'runs':1,'round_iterations':1}))
    evidence=save(old/'reconciliation.json',e)
    return Path(newplan['path']),Path(s1['path']),Path(oldstate['path']),Path(evidence['path'])


def test_completed_bc_and_test_are_preserved_not_executed(tmp_path):
    p,s,old,e=fixture(tmp_path);before=old.read_bytes();a=Adapter()
    result=adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=a)
    assert result['recovery']['heldout_population_reused'] is False
    assert result['recovery']['bc_test_episodes_repeated']==0
    state_path=p.parent/'execution/state.json';recovered_bytes=state_path.read_bytes()
    with pytest.raises(ValueError,match='MSE validation is disabled'):
        execute(p,s,adapter=a)
    assert a.calls==[] and state_path.read_bytes()==recovered_bytes
    assert execute(p,s,adapter=a,allow_legacy_validation=True)['status']=='COMPLETE'
    assert a.calls==['dagger'] and old.read_bytes()==before
    with pytest.raises(FileExistsError):adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=a)


@pytest.mark.parametrize('fault',['live','training','learner','checkpoint','seed','coverage','panel','round','receipt','retention','main_run'])
def test_unsafe_recovery_fails_closed(tmp_path,fault):
    p,s,old,e=fixture(tmp_path);evidence=json.loads(e.read_bytes());plan=json.loads(p.read_bytes())
    if fault=='live':evidence['services'][0]['status']['state']='RUNNING'
    elif fault=='training':
        path=Path(evidence['workers'][0]['journal']['path']);v=json.loads(path.read_bytes())
        v['commands']['x']['command']['operation']='collect_production_segment';path.write_text(json.dumps(v))
        evidence['workers'][0]['journal']=binding(path)
    elif fault in ('learner','checkpoint','main_run'):
        path=Path(evidence['closed_ledger']['database']['path'])
        with sqlite3.connect(path) as db:
            if fault=='learner':db.execute("INSERT INTO learner_jobs VALUES('x')")
            elif fault=='checkpoint':db.execute("INSERT INTO checkpoints VALUES('not-parent')")
            else:db.execute("UPDATE runs SET run_id='dagger-leo-prod'")
        evidence['closed_ledger']['database']=binding(path)
    elif fault=='seed':plan['settings']['seed']=43
    elif fault=='coverage':
        path=Path(plan['settings']['coverage_plan']['path']);v=json.loads(path.read_bytes());v['rounds'][0]['seed']+=1
        path.write_text(json.dumps(v));plan['settings']['coverage_plan']=binding(path)
    elif fault=='panel':plan['test_panel']['raw_sha256']='0'*64
    elif fault=='round':
        path=Path(evidence['campaign_state']['path']);v=json.loads(path.read_bytes());v['round_id']=2
        path.write_text(json.dumps(v));evidence['campaign_state']=binding(path)
    elif fault=='receipt':
        v=json.loads(old.read_bytes());Path(v['completed'][1]['receipt']['path']).write_text('{}')
    else:evidence['compute_release_authorized']=True
    e.write_text(json.dumps(evidence));p.write_text(json.dumps(plan))
    with pytest.raises((ValueError,RuntimeError,AssertionError)):
        adopt_completed_bc_test(p,s,previous_state=old,reconciliation=e,adapter=Adapter())
    assert not (p.parent/'execution/state.json').exists()
