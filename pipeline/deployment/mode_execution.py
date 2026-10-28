"""Sequential mode-selected workflow with durable, non-replayable phase intents.

The site adapter uses the actual native collector, offline trainer, corrected
resident evaluator and MSE campaign. This module never allocates/releases GPUs.
An interrupted RUNNING phase is not retried: reconcile its native receipts first.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import time

from pipeline.bc_training.handoff_common import binding, bound, canonical
from pipeline.deployment.beta_operations import atomic

PHASES=('prepare_bc','train_bc','test_bc','dagger')
SCHEMA='roborender.mode-execution.v1'


def adopt_completed_bc(plan_path,settings_path,*,previous_state,stopped_evidence,adapter=None):
    """Explicit source-fix recovery before ANY test/training native command.

    Preserve actual completed BC receipts and their original source provenance.
    Never adopt partial tests, DAgger windows, or simulator state. Evidence must
    bind both stopped native owners and their complete command journals.
    """
    old_ref=binding(previous_state);old=bound(old_ref)
    evidence_ref=binding(stopped_evidence);evidence=bound(evidence_ref)
    plan_ref,settings_ref=binding(plan_path),binding(settings_path)
    plan,settings=bound(plan_ref),bound(settings_ref)
    prior=bound(old['plan']);prior_settings=bound(old['settings'])
    if (old.get('phase')!='test_bc' or old.get('status') not in ('RUNNING','RECONCILE_REQUIRED')
            or [r['phase'] for r in old.get('completed',[])]!=['prepare_bc','train_bc']):
        raise ValueError('only completed BC before native test is recoverable here')
    for key in ('schema','mode','run_id','bc','dagger','test','logging','execution','stage_names','dagger_seed'):
        if prior.get(key)!=plan.get(key):raise ValueError('scientific mode contract changed: '+key)
    for key in set(prior['settings'])-{'source_root','source_sha256'}:
        if prior['settings'][key]!=plan['settings'].get(key):raise ValueError('scientific settings changed: '+key)
    if bound(prior['test_panel'])!=bound(plan['test_panel']):raise ValueError('test panel changed')
    if (prior_settings['retained']!=settings['retained']
            or settings.get('compute_release_authorized') is not False):
        raise ValueError('retained allocation binding changed')
    if (evidence.get('schema')!='roborender.pretest-recovery.v1'
            or evidence.get('previous_state')!=old_ref
            or evidence.get('controller_status',{}).get('state') not in ('STOPPED','FAILED')
            or evidence.get('bootstrap_status',{}).get('state') not in ('STOPPED','FAILED')
            or len(evidence.get('workers',[]))!=2):
        raise ValueError('stopped controller/bootstrap and two native owners required')
    if {w['physical_gpu'] for w in evidence['workers']}!={0,1}:
        raise ValueError('two distinct native owners required')
    for worker in evidence['workers']:
        journal=bound(worker['journal'])
        if worker.get('unit_state') not in ('inactive','failed') or journal['source_sha256']!=prior['source_sha256']:
            raise ValueError('old native owner not stopped or source changed')
        for command in journal.get('commands',{}).values():
            if command.get('command',{}).get('operation')!='capture_initial_b2':
                raise ValueError('native test/training accepted; cannot use pretest recovery')
    if Path(previous_state).parent.joinpath('test_bc-receipt.json').exists():
        raise ValueError('test receipt already exists')
    if adapter is None:
        from pipeline.deployment.mode_site import ModeSite
        adapter=ModeSite(plan,settings)
    for saved in old['completed']:
        adapter.validate(saved['phase'],bound(saved['receipt']),old['completed'])
    root=Path(plan['output_root'])/'execution';root.mkdir(parents=True,exist_ok=True)
    if root.resolve()==Path(previous_state).parent.resolve():raise ValueError('fresh continuation directory required')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'state.json'
        if path.exists():raise FileExistsError('continuation already initialized')
        state=dict(schema=SCHEMA,plan=plan_ref,settings=settings_ref,run_id=plan['run_id'],mode=plan['mode'],
            status='READY',completed=old['completed'],compute_release_authorized=False,
            recovery=dict(previous_state=old_ref,stopped_evidence=evidence_ref,
                native_state_reused=False,completed_bc_receipts_preserved=True))
        atomic(path,state)
    return state


def execute(plan_path, settings_path, *, adapter=None, allow_legacy_validation=False):
    plan_ref,settings_ref=binding(plan_path),binding(settings_path)
    plan,settings=bound(plan_ref),bound(settings_ref)
    if plan.get('schema')!='roborender.pipeline-mode.v1' or plan.get('mode') not in ('smoke','production'):
        raise ValueError('an explicit immutable pipeline mode is required')
    from pipeline.deployment.training_launch_policy import require_training_launch
    require_training_launch(validation_enabled=True,
        allow_legacy_validation=allow_legacy_validation)
    if settings.get('compute_release_authorized') is not False:
        raise ValueError('execution must retain the existing allocation')
    if adapter is None:
        from pipeline.deployment.mode_site import ModeSite
        adapter=ModeSite(plan,settings)
    root=Path(plan['output_root'])/'execution';root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'state.json'
        state=(json.loads(path.read_bytes()) if path.exists() else dict(schema=SCHEMA,
            plan=plan_ref,settings=settings_ref,run_id=plan['run_id'],mode=plan['mode'],
            status='READY',completed=[],compute_release_authorized=False))
        if state.get('plan')!=plan_ref or state.get('settings')!=settings_ref:
            raise ValueError('frozen execution bindings changed')
        if state['status'] not in ('READY','COMPLETE'):
            raise RuntimeError('incomplete phase requires reconciliation; automatic replay forbidden')
        if [r['phase'] for r in state['completed']]!=list(PHASES[:len(state['completed'])]):
            raise ValueError('phase completion prefix differs')
        for saved in state['completed']:
            adapter.validate(saved['phase'],bound(saved['receipt']),state['completed'])
        for phase in PHASES[len(state['completed']):]:
            started=time.time()
            state.update(status='RUNNING',phase=phase,started_at=started)
            atomic(path,state)
            print(json.dumps(dict(phase=phase,event='start',mode=plan['mode'])),flush=True)
            try:
                receipt=getattr(adapter,phase)(state['completed'])
                adapter.validate(phase,receipt,state['completed'])
                receipt_path=root/(phase+'-receipt.json')
                if receipt_path.exists():raise FileExistsError('phase receipt already exists')
                atomic(receipt_path,receipt)
                state['completed'].append(dict(phase=phase,receipt=binding(receipt_path),
                    wall_seconds=time.time()-started))
                state.update(status='COMPLETE' if phase==PHASES[-1] else 'READY')
                atomic(path,state)
                print(json.dumps(dict(phase=phase,event='complete',wall_seconds=time.time()-started)),flush=True)
            except BaseException as exc:
                state.update(status='RECONCILE_REQUIRED',error_type=type(exc).__name__,
                    automatic_native_replay=False,compute_release_authorized=False)
                atomic(path,state)
                raise
        return state
