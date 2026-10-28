"""Admit ONLY an unstarted evaluation after fully completed round training.

No GPU allocation, native command, optimizer submission or ledger mutation is
performed by admission. A recorded/accepted evaluation command rejects recovery.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re
import sqlite3
import time

from pipeline.dagger.beta_campaign import immutable_json, validate_actor_completion
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import budgets, validate_config
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.beta_operations import CommandOperations, atomic, process_identity
from pipeline.deployment.round_boundary_recovery import digest, validate_binding as validate_boundary_binding

SCHEMA='roborender.pending-evaluation-recovery.v1'


def ledger_evidence(root,state):
    run,rid=state['run_id'],state['round_id'];windows=budgets(state['config'])[0]
    path=Path(root)/'services/state.sqlite3'
    if path.is_symlink() or not path.is_file():raise RuntimeError('existing closed ledger required')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        current=db.execute('SELECT current_round,status FROM runs WHERE run_id=?',(run,)).fetchone()
        if current is None or tuple(current)!=(rid,'validating'):raise RuntimeError('run is not at final validation barrier')
        rows=[dict(r) for r in db.execute('SELECT * FROM round_iterations WHERE run_id=? AND round_id=? ORDER BY iteration_id',(run,rid))]
        jobs=[dict(r) for r in db.execute('SELECT status,result_json FROM learner_jobs WHERE run_id=? AND round_id=?',(run,rid))]
        results=sorted([json.loads(j['result_json']) for j in jobs if j['status']=='succeeded' and j['result_json']],key=lambda r:r['iteration_id'])
        if len(jobs)!=windows or len(results)!=windows or len(rows)!=windows:raise RuntimeError('learner budget incomplete')
        for row,result in zip(rows,results):
            if (result['iteration_id']!=row['iteration_id'] or result['parent_checkpoint_sha256']!=row['input_checkpoint_sha256']
                    or result['output_checkpoint_sha256']!=row['candidate_checkpoint_sha256']
                    or result.get('train_chunks')!=12 or result.get('train_frames')!=972 or result.get('replay_manifest_sha256') is not None):
                raise RuntimeError('learner population or checkpoint chain differs')
        count=db.execute('SELECT COUNT(*) FROM step_prepares WHERE run_id=? AND round_id=?',(run,rid)).fetchone()[0]
        committed=db.execute("SELECT COUNT(*) FROM step_prepares WHERE run_id=? AND round_id=? AND status='committed'",(run,rid)).fetchone()[0]
        commits=db.execute('SELECT COUNT(*) FROM step_commits c JOIN step_prepares p ON c.prepare_idempotency_key=p.idempotency_key WHERE p.run_id=? AND p.round_id=?',(run,rid)).fetchone()[0]
        barriers=db.execute('SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? AND round_id=?',(run,rid)).fetchone()[0]
        if (count,committed,commits,barriers)!=(windows*972,windows*972,windows*972,windows*12):raise RuntimeError('round ledger is incomplete')
        evaluations=[dict(r) for r in db.execute("SELECT evaluation_id,status,checkpoint_sha256,scenario_manifest_sha256 FROM success_evaluations WHERE run_id=? AND round_id=? AND purpose='validation'",(run,rid))]
        if (len(evaluations)!=1 or evaluations[0]['status']!='collecting'
                or evaluations[0]['checkpoint_sha256']!=results[-1]['output_checkpoint_sha256']
                or evaluations[0]['scenario_manifest_sha256']!=state['config']['validation']['scenario_manifest_sha256']):
            raise RuntimeError('pending evaluation identity differs')
        if db.execute('SELECT COUNT(*) FROM success_evaluation_outcomes WHERE evaluation_id=?',(evaluations[0]['evaluation_id'],)).fetchone()[0]:
            raise RuntimeError('evaluation already has outcomes')
    return {'iterations':rows,'results':results,'counts':{'prepares':count,'commits':commits,'barriers':barriers},'evaluation':evaluations[0]}


def validate(root,state,failure,evidence,*,now=None):
    root=Path(root);config=validate_config(state['config']);rid=state['round_id']
    if (state.get('phase')!='ABANDONED' or failure.get('phase')!='EVALUATE' or failure.get('native_restart_allowed') is not False
            or state['run_id']!=config['run_id'] or rid!=config['round_id']
            or [r['round_id'] for r in state['completed_rounds']]!=list(range(1,rid))):
        raise RuntimeError('only pending evaluation after completed training is admissible')
    validate_boundary_binding(root,state)
    if digest(json.loads(Path(state['config_path']).read_bytes()))!=digest(config):raise RuntimeError('round config changed')
    abort=state.get('abort_receipt',{})
    if not (abort.get('ok') is True and abort.get('owned_actors_stopped') is True and abort.get('h200_allocation_released') is True
            and abort.get('actor_process_reconciliation_warnings')==[]):raise RuntimeError('scoped cleanup incomplete')
    if (not 0<=(time.time() if now is None else now)-evidence.get('checked_at',0)<=300
            or evidence.get('allocation_absent') is not True or evidence.get('management_owners_stopped') is not True
            or evidence.get('native_evaluation_commands')!=0 or evidence.get('evaluation_artifacts_absent') is not True):
        raise RuntimeError('fresh absent-evaluation and idle-owner evidence required')
    ledger=ledger_evidence(root,state)
    if evidence.get('ledger')!=ledger:raise RuntimeError('ledger changed during admission')
    receipts=list((root/'operations').glob(f'r{rid}-collect-*/receipt.json'))
    if len(receipts)!=1:raise RuntimeError('one complete collection receipt required')
    receipt=json.loads(receipts[0].read_bytes())
    candidate=validate_actor_completion(config,receipt,ledger['iterations'])
    if evidence.get('native_phase_barrier')!=receipt.get('native_phase_barrier'):raise RuntimeError('resident owner generation/profile changed')
    final=state['final_checkpoint']
    if final['sha256']!=candidate or file_sha256(Path(final['path']))!=candidate or state['final_learner']!=ledger['results'][-1]:
        raise RuntimeError('final checkpoint/learner differs')
    return {'schema':SCHEMA,'run_id':state['run_id'],'round_id':rid,'config_sha256':digest(config),
        'completed_history_sha256':digest(state['completed_rounds']),'final_checkpoint':final,'resume_phase':'EVALUATE',
        'native_restart_allowed':False,'completed_training_repeated':False,'completed_evaluation_repeated':False}


def validate_binding(root,state):
    binding=state.get('pending_evaluation_recovery')
    if binding is None:return
    rid=binding.get('recovery_id')
    if not isinstance(rid,str) or not re.fullmatch('[a-f0-9]{32}',rid):raise RuntimeError('invalid evaluation recovery ID')
    directory=Path(root)/'recoveries'/rid
    admission=json.loads((directory/'admission.json').read_bytes())
    prior=json.loads((directory/'prior-state.json').read_bytes())
    if (file_sha256(directory/'admission.json')!=binding.get('admission_sha256') or admission.get('schema')!=SCHEMA
            or admission['run_id']!=state['run_id'] or admission['prior_state_sha256']!=digest(prior)
            or admission.get('native_restart_allowed') is not False or admission.get('completed_training_repeated') is not False
            or admission.get('completed_evaluation_repeated') is not False or state['round_id']<admission['round_id']
            or state['completed_rounds'][:admission['round_id']-1]!=prior['completed_rounds']):
        raise RuntimeError('pending-evaluation admission changed')
    if state['round_id']==admission['round_id'] and (digest(state['config'])!=admission['config_sha256'] or state['final_checkpoint']!=admission['final_checkpoint']):
        raise RuntimeError('pending evaluation science/checkpoint changed')


def publish(root,state,failure,evidence,operations,recovery_id):
    if not re.fullmatch('[a-f0-9]{32}',recovery_id):raise ValueError('fresh recovery ID required')
    root=Path(root);admission=validate(root,state,failure,evidence)
    if json.loads((root/'state.json').read_bytes())!=state:raise RuntimeError('campaign changed')
    directory=root/'recoveries'/recovery_id
    if directory.exists():raise RuntimeError('recovery archive exists')
    admission.update(prior_state_sha256=digest(state),evidence_sha256=digest(evidence),operations_sha256=digest(operations))
    for name,value in (('prior-state.json',state),('prior-failure.json',failure),('evidence.json',evidence),('operations.json',operations),('admission.json',admission)):
        immutable_json(directory/name,value)
    updated=deepcopy(state);updated.pop('abort_receipt',None)
    updated.update(phase='EVALUATE',operation_id=None,updated_at=time.time(),operations_sha256=digest(operations),
        pending_evaluation_recovery={'recovery_id':recovery_id,'admission_sha256':file_sha256(directory/'admission.json')})
    atomic(root/'state.json',updated)
    return admission


def inspect_live(site,root,state):
    from pipeline.deployment.owned_process import status
    if site.allocation('inspect').get('present') is not False:raise RuntimeError('allocation is not absent')
    for label in ('cpu-services','tunnels','lease-watchdog'):
        if status(site.round/'processes'/label)['state'] not in {'ABSENT','COMPLETE','STOPPED','FAILED'}:raise RuntimeError('management owner still active')
    for directory in (Path(root)/'operations').glob(f"r{state['round_id']}-*"):
        saved=json.loads((directory/'process.json').read_bytes())
        if saved.get('identity') is not None and process_identity(saved['pid'])==saved['identity']:raise RuntimeError('old operation is live')
    barrier=site.barrier()['native_phase_barrier']
    code="""import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_bytes());run=sys.argv[2];rid=int(sys.argv[3]);count=0;absent=True
for w in p['workers']:
 s=json.loads((pathlib.Path(w['runtime_root'])/'resident-state.json').read_bytes())
 assert s['phase']=='IDLE' and s['worker_instance_id']==w['worker_instance_id']
 latest=max(s['commands'].values(),key=lambda c:c.get('accepted_at_ns',0))
 assert latest['command']['command_id']==f'{run}-r{rid}-collect-g{w["physical_gpu"]}'
 count+=sum(cid.startswith(f'{run}-r{rid}-') and c['command']['operation']!='collect_round' or any(cid.startswith(f'{run}-r{r}-') for r in range(rid+1,5)) for cid,c in s['commands'].items())
 absent=absent and not (pathlib.Path(w['runtime_root'])/'phases'/f'{run}-r{rid}-evaluate').exists()
print(json.dumps({'commands':count,'artifacts_absent':absent}))
"""
    native=site.json([site.site['rtx_python'],'-c',code,site.pool_file,state['run_id'],str(state['round_id'])],'beta')
    return {'checked_at':time.time(),'allocation_absent':True,'management_owners_stopped':True,'native_phase_barrier':barrier,
        'native_evaluation_commands':native['commands'],'evaluation_artifacts_absent':native['artifacts_absent'],'ledger':ledger_evidence(root,state)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('root','operations'):parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--recovery-id',required=True);parser.add_argument('--admit',action='store_true')
    args=parser.parse_args();root=args.root.resolve(strict=True)
    operations=json.loads(args.operations.read_bytes());CommandOperations(operations,root/'operations')
    with ExitStack() as stack:
        for path in (root/'scheduler.lock',root/'services/bundle-owner.lock'):
            lock=stack.enter_context(path.open('r+'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'state.json').read_bytes());failure=json.loads((root/'failure.json').read_bytes())
        from pipeline.deployment.recovering_campaign import make_site
        site=make_site(operations,root,state);site.verify_sources()
        evidence=inspect_live(site,root,state);validate(root,state,failure,evidence)
        if args.admit:publish(root,state,failure,evidence,operations,args.recovery_id)
        print(json.dumps({'admissible':True,'published':args.admit,'native_work_started':False,'allocation_performed':False}))


if __name__=='__main__':main()
