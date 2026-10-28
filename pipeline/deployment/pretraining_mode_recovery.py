"""Preserve completed BC/test after a failed, never-trained held-out collection.

This creates a fresh execution namespace. It never restores a simulator, reuses
partial validation data, changes sampling, or mutates/releases compute.
"""
from copy import deepcopy
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from pipeline.bc_training.handoff_common import binding,bound
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.mode_execution import SCHEMA

ZERO_TABLES=('learner_jobs','learner_job_iterations',
             'promotions','success_evaluations')


def compact_recovery_prefix(previous):
    """Bound the name while keeping the actual parent in hash-bound receipts."""
    if not isinstance(previous,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,39}',previous):
        raise ValueError('bounded prior mode identity required')
    digest=hashlib.sha256(b'roborender.recovery-parent.v1\0'+previous.encode()).hexdigest()[:16]
    return previous[:10]+'-rec-'+digest+'-'


def recovery_run_id(previous,label):
    prefix=compact_recovery_prefix(previous)
    if not isinstance(label,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,7}',label):
        raise ValueError('one to eight lowercase recovery-label characters required')
    full=previous+'-recovery-'+label
    return full if len(full)<=40 else prefix+label


def compare_plans(previous,current):
    """Only source/runtime identity changes; every scientific value stays fixed."""
    run=current['run_id'];prefix=compact_recovery_prefix(previous['run_id'])
    compact=(isinstance(run,str) and run.startswith(prefix)
        and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,7}',run[len(prefix):]) is not None)
    if (not isinstance(run,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,39}',run)
            or run==previous['run_id'] or not (compact or run.startswith(previous['run_id']+'-recovery-'))):
        raise ValueError('fresh explicitly named recovery identity required')
    allowed={'run_id','source_root','source_sha256','output_root','settings','stage_names'}
    if set(previous)!=set(current) or any(previous[k]!=current[k] for k in previous.keys()-allowed):
        raise ValueError('scientific mode contract changed')
    old,new=previous['settings'],current['settings']
    if old.get('run_id')!=previous['run_id'] or new.get('run_id')!=run:
        raise ValueError('mode/settings recovery identity differs')
    identities={'run_id','source_root','source_sha256','coverage_plan','capacity_admission'}
    if set(old)!=set(new) or any(old[k]!=new[k] for k in old.keys()-identities):
        raise ValueError('scientific settings changed')
    before,after=deepcopy(bound(old['coverage_plan'])),deepcopy(bound(new['coverage_plan']))
    old_run,new_run=before['run_id'],after['run_id']
    if old_run!='dagger-leo-'+previous['run_id'] or new_run!='dagger-leo-'+run:
        raise ValueError('coverage recovery identity differs')
    for round_row in before['rounds']:
        for epoch in round_row.get('epochs',[]):
            for cohort in epoch['cohorts']:
                for assignment in cohort['assignments']:
                    if assignment['run_id']!=old_run:raise ValueError('unexpected old cohort owner')
                    assignment['run_id']=new_run
    before['run_id']=after['run_id'];before['source_sha256']=after['source_sha256']
    if len(before['rounds'])!=4 or len(after['rounds'])!=4:
        raise ValueError('four rounds required')
    for a,b in zip(before['rounds'],after['rounds']):
        for key in ('wandb_run_id','wandb_run_name'):a[key]=b[key]
    if before!=after:raise ValueError('coverage, seeds, budgets or sampling changed')
    if {k:previous['run_id']+'-'+k for k in previous['stage_names']}!=previous['stage_names']:
        raise ValueError('previous logging namespace differs')
    if {k:current['run_id']+'-'+k for k in previous['stage_names']}!=current['stage_names']:
        raise ValueError('fresh logging namespace differs')


def validate_reconciliation(old,old_ref,evidence):
    if (old.get('status')!='RECONCILE_REQUIRED' or old.get('phase')!='dagger'
            or [r['phase'] for r in old['completed']]!=['prepare_bc','train_bc','test_bc']
            or evidence.get('schema')!='roborender.pretraining-reconciliation.v1'
            or evidence.get('previous_state')!=old_ref
            or evidence.get('compute_release_authorized') is not False
            or evidence.get('controller_status',{}).get('state') not in ('FAILED','STOPPED')):
        raise ValueError('failed pretraining boundary required')
    campaign=bound(evidence['campaign_state'])
    prior=bound(old['plan'])
    if (campaign.get('run_id')!='dagger-leo-'+prior['run_id'] or campaign.get('round_id')!=1
            or campaign.get('phase')!='COLLECT_HELDOUT' or campaign.get('completed_rounds')!=[]):
        raise ValueError('only first-round pretraining collection is recoverable')
    train=bound(old['completed'][1]['receipt']);test=bound(old['completed'][2]['receipt'])
    parent=train['checkpoint']['sha256']
    if test['checkpoint_sha256']!=parent or campaign['config']['student']['initial_checkpoint_sha256']!=parent:
        raise ValueError('completed test and untrained round parent differ')
    if (evidence.get('retained_before')!=evidence.get('retained_after')
            or evidence['retained_after'].get('allocation_retained') is not True):
        raise ValueError('same retained allocation required')
    services=evidence.get('services',[])
    if len(services)!=3 or {(r['host'],r['label']) for r in services}!={('beta','tunnels'),('local','tunnels'),('gpu','gpu-services')}:
        raise ValueError('exact former service owners required')
    if any(r['status'].get('state') not in ('STOPPED','COMPLETE','FAILED','ABSENT') for r in services):
        raise ValueError('old services remain live')
    workers=evidence.get('workers',[])
    if len(workers)!=2 or {w['physical_gpu'] for w in workers}!={0,1}:
        raise ValueError('both old native owners required')
    for worker in workers:
        journal=bound(worker['journal'])
        if (worker.get('container_absent') is not True or worker.get('unit_state') not in ('inactive','failed')
                or journal['source_sha256']!=prior['source_sha256']
                or journal['worker_instance_id']!=worker['worker_instance_id']):
            raise ValueError('old native lifetime not reconciled')
        for command in journal['commands'].values():
            if command['command']['operation'] not in ('capture_initial_b2','evaluate_explicit_panel','collect_mse_validation'):
                raise ValueError('native training accepted; this recovery cannot replay it')
    ledger=evidence['closed_ledger']
    if ledger.get('old_population_excluded') is not True or ledger.get('counts')!={**{t:0 for t in ZERO_TABLES},'runs':1,'round_iterations':1}:
        raise ValueError('any learner/round registration forbids pretraining recovery')
    path=Path(ledger['database']['path'])
    if binding(path)!=ledger['database']:raise ValueError('closed ledger bytes changed')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        if any(db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ZERO_TABLES):
            raise ValueError('actual closed ledger contains training or test work')
        if db.execute('SELECT sha256 FROM checkpoints').fetchall()!=[(parent,)]:
            raise ValueError('closed ledger has non-parent checkpoint')
        validation_id='heldout-'+campaign['run_id']+'-r1'
        if (db.execute('SELECT run_id,status,active_checkpoint_sha256 FROM runs').fetchall()!=[(validation_id,'collecting',parent)]
                or db.execute('SELECT run_id,round_id,iteration_id,status,candidate_checkpoint_sha256 FROM round_iterations').fetchall()!=[(validation_id,1,1,'collecting',None)]
                or db.execute('SELECT run_id,status FROM validation_collections').fetchall()!=[(validation_id,'collecting')]):
            raise ValueError('only the excluded held-out registration may exist')
    return parent


def adopt_completed_bc_test(plan_path,settings_path,*,previous_state,reconciliation,adapter=None):
    evidence=bound(binding(reconciliation))
    campaign_root=Path(evidence['campaign_state']['path']).parent
    with ExitStack() as stack:
        for path in (Path(previous_state).parent/'controller.lock',campaign_root/'scheduler.lock',
                     campaign_root/'services/bundle-owner.lock'):
            path.parent.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context(path.open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _adopt_locked(plan_path,settings_path,previous_state=previous_state,
            reconciliation=reconciliation,adapter=adapter)


def _adopt_locked(plan_path,settings_path,*,previous_state,reconciliation,adapter):
    old_ref=binding(previous_state);old=bound(old_ref)
    evidence_ref=binding(reconciliation);evidence=bound(evidence_ref)
    plan_ref,settings_ref=binding(plan_path),binding(settings_path)
    plan,settings=bound(plan_ref),bound(settings_ref)
    compare_plans(bound(old['plan']),plan)
    validate_reconciliation(old,old_ref,evidence)
    if (settings.get('compute_release_authorized') is not False or
            bound(old['settings'])['retained']!=settings['retained']):
        raise ValueError('retained allocation binding changed')
    if adapter is None:
        from pipeline.deployment.mode_site import ModeSite
        adapter=ModeSite(plan,settings)
    for row in old['completed']:adapter.validate(row['phase'],bound(row['receipt']),old['completed'])
    root=Path(plan['output_root'])/'execution'
    if root.resolve()==Path(previous_state).parent.resolve():raise ValueError('fresh output namespace required')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=root/'state.json'
        if target.exists():raise FileExistsError('recovery already admitted')
        state=dict(schema=SCHEMA,plan=plan_ref,settings=settings_ref,run_id=plan['run_id'],mode=plan['mode'],
            status='READY',completed=old['completed'],compute_release_authorized=False,
            recovery=dict(previous_state=old_ref,reconciliation=evidence_ref,native_state_reused=False,
                heldout_population_reused=False,completed_bc_and_test_preserved=True,
                bc_optimizer_updates_repeated=0,bc_test_episodes_repeated=0))
        atomic(target,state)
    return state
