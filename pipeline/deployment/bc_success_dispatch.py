"""Queue the immutable selected BC checkpoint's actual full295 resident test.

CPU-only controller; it never provisions/releases compute or restarts Isaac.
All native work goes through the existing resident command endpoint. Repeated
submission uses the identical durable command identity, never a new episode run.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import time

from pipeline.bc_training.handoff_common import binding,bound,canonical,publish,read
from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel
from pipeline.deployment.rtx_actor.production_success_contract import (
    evaluation_options,protocol_identity,validate_production_result,
)
from pipeline.deployment.wandb_preflight import admit_logging,verify_remote_summary
from pipeline.deployment.rtx_actor.production_resident_pool import validate_pool
from pipeline.deployment.rendered_success_dispatch import artifact as rendered_artifact, configuration, payload_fields

SSH='/workspace/.ssh/roborender_leo_access_config'
HOST='roborender-leo-direct'
NETRC='/workspace/projects/attention/setup/secrets/wandb.netrc'
RUN_ID_PREFIX='20261016-prod-1-bc-success'


def protocol_suffix(panel, config_sha256=None):
    value = 'student-image3f-v1-'+protocol_identity(panel)['sha256'][:12]
    return value + ('-' + config_sha256[:12] if config_sha256 is not None else '')


class Leo:
    def command(self,argv,*,timeout=180):
        result=subprocess.run(['ssh','-F',SSH,'-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,shlex.join(list(map(str,argv)))],
            check=True,capture_output=True,timeout=timeout)
        return result.stdout

    def stage(self,source,destination,sha256):
        """Copy at most once; verify remote bytes before no-clobber publication."""
        destination=Path(destination)
        check=('import pathlib,hashlib,sys;p=pathlib.Path(sys.argv[1]);'
               'assert not p.is_symlink();assert not p.exists() or p.is_file();'
               'print("missing" if not p.exists() else hashlib.sha256(p.read_bytes()).hexdigest())')
        prior=self.command(['python3','-c',check,str(destination)]).decode().strip()
        if prior==sha256:return
        if prior!='missing':raise RuntimeError('existing staged artifact has different bytes')
        partial=str(destination)+'.part-'+str(os.getpid())
        self.command(['python3','-c','import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()',partial])
        subprocess.run(['scp','-q','-F',SSH,'-o','BatchMode=yes','-o','ConnectTimeout=12',str(source),HOST+':'+partial],
            check=True,capture_output=True,timeout=1800)
        code=('import pathlib,hashlib,os,sys;p=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);'
              'assert not p.is_symlink();assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3];'
              'os.chmod(p,0o400);os.link(p,d);p.unlink()')
        self.command(['python3','-c',code,partial,str(destination),sha256],timeout=180)


def validate_selection(selection):
    from pipeline.bc_training import final_selection
    if selection.get('schema') == final_selection.SCHEMA:
        value = final_selection.validate_selection(selection)
        return bound(value['test_panel'])
    if (selection.get('schema')!='roborender.bc-production-selected.v1'
            or not isinstance(selection.get('run_name'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', selection['run_name'])
            or selection.get('selection')!='minimum_complete_validation_mse'
            or selection.get('success_test_executed') is not False
            or selection.get('expected_test_episodes')!=295):
        raise ValueError('immutable MSE-selected production BC checkpoint required')
    panel=validate_panel(bound(selection['test_panel']))
    if panel['purpose']!='test' or panel['episode_count']!=295:
        raise ValueError('separate frozen full295 test required')
    evaluation_options(panel,raw_sha256=selection['test_panel']['raw_sha256'])
    configuration(selection.get('rendered_observation_config'), panel, selection['checkpoint']['sha256'])
    checkpoint=selection['checkpoint']; path=Path(checkpoint['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=checkpoint['bytes']:
        raise ValueError('selected checkpoint path/size changed')
    return panel


def campaign_identity(selection):
    """Keep historical receipts stable; new campaigns never share logging IDs."""
    if selection['run_name'] == '2026-10-16-prod-1-BC':
        return 'bc-prod-1', RUN_ID_PREFIX, '2026-10-16-prod-1-BC-success'
    name = selection['run_name']
    import hashlib
    short = name[:48] + '-' + hashlib.sha256(name.encode()).hexdigest()[:12]
    return short, short + '-success', name + '-success'


def command_plan(selection,pool):
    if selection.get('schema')=='roborender.bc-production-selected.v2':
        raise ValueError('paired scaled BC v2 requires its separately admitted paired executor; legacy GPU0/B2 dispatch is forbidden')
    panel=validate_selection(selection)
    validate_pool(pool)
    sha=selection['checkpoint']['sha256']
    if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('checkpoint digest required')
    config, config_raw, config_sha = rendered_artifact(selection.get('rendered_observation_config'), panel, sha)
    suffix=protocol_suffix(panel, config_sha)
    campaign, _, _ = campaign_identity(selection)
    import hashlib
    name='bc-full295-'+hashlib.sha256(campaign.encode()).hexdigest()[:12]+'-'+sha[:12]+'-'+suffix
    panel_file='production-tests/'+campaign+'-full295-'+suffix+'.json'
    config_file='production-tests/rendered-'+config_sha+'.json'
    command=dict(gpu=0,command_id=name,operation='evaluate_explicit_panel',payload=dict(
        panel_file=panel_file,panel_sha256=selection['test_panel']['raw_sha256'],
        checkpoint_sha256=sha,arm='bc',output_name=name,
        **evaluation_options(panel,raw_sha256=selection['test_panel']['raw_sha256']),
        **payload_fields(config_file, config_sha)))
    return dict(command=command,protocol_suffix=suffix,
        rendered_config=config, rendered_config_raw_sha256=config_sha,
        rendered_config_destination=str(Path(pool['jobs_root'])/config_file),
        checkpoint_destination=str(Path(pool['assets_root'])/(sha+'.pt')),
        panel_destination=str(Path(pool['jobs_root'])/panel_file),
        command_destination=str(Path(pool['jobs_root'])/'production-tests'/(name+'-command.json')))


def log_success(result,selection,output,*,wandb_module=None):
    panel=validate_selection(selection)
    config, _, config_sha = rendered_artifact(selection.get('rendered_observation_config'), panel, selection['checkpoint']['sha256'])
    result=validate_production_result(panel,result,selection['checkpoint']['sha256'],allowed_arms=('bc',),rendered_config=config)
    protocol=protocol_identity(panel)
    suffix=protocol_suffix(panel,config_sha)
    _, run_prefix, name_prefix = campaign_identity(selection)
    run_id=run_prefix+'-'+suffix
    run_name=name_prefix+'-'+suffix
    os.environ['NETRC']=NETRC
    api,identity=admit_logging('online','nvidia')
    if wandb_module is None:
        import wandb as wandb_module
    run=wandb_module.init(entity='nvidia',project='dagger',id=run_id,name=run_name,
        group='BC',job_type='success-test',mode='online',resume='allow',dir=str(output),
        config=dict(checkpoint_sha256=selection['checkpoint']['sha256'],completed_episodes=295,
            metric='native_goal_or_until_done',selection_feedback=False,optimizer_steps=0,
            observation_mode=panel['observation_mode'],rendered_observation_config_sha256=config_sha,
            test_panel_raw_sha256=selection['test_panel']['raw_sha256'],
            success_test_protocol=protocol,
            rendered_student_success_protocol=result['rendered_student_success_protocol']))
    expected={'test/successes':result['successes'],'test/completed_episodes':295,
        'test/success_rate':result['success_rate'],'test/optimizer_steps':0,
        'test/success_test_protocol':protocol,
        'test/rendered_student_success_protocol':result['rendered_student_success_protocol']}
    run.log(expected); url=run.url; run.finish()
    verify_remote_summary(api,'nvidia',run_id,expected)
    return dict(url=url,run_id=run_id,name=run_name,online_readback_verified=True,
        authenticated=identity['authenticated'])


def execute(selection_ref,pool_ref,remote_pool,remote_source,output,*,transport=None,sleep=time.sleep,deadline=None):
    selection=bound(selection_ref);pool=bound(pool_ref);panel=validate_selection(selection);plan=command_plan(selection,pool)
    # A protocol change gets its own immutable journal. Existing historical
    # intent, command and result files are never rewritten or silently adopted.
    base_output=Path(output)
    output=base_output/plan['protocol_suffix'];output.mkdir(parents=True,exist_ok=True)
    transport=Leo() if transport is None else transport
    intent=dict(schema='roborender.bc-success-dispatch.v1',selection=selection_ref,pool=pool_ref,
        remote_pool=str(remote_pool),remote_source=str(remote_source),plan=plan,optimizer_steps=0)
    ip=output/'intent.json'
    if ip.exists():
        if json.loads(read(ip))!=intent:raise RuntimeError('success dispatch identity changed; no replay')
    else:publish(ip,intent)
    helper=Path(remote_source)/'pipeline/deployment/rtx_actor/production_resident_pool.py'
    def action(name,command_path=None):
        argv=['python3',str(helper),'--pool',str(remote_pool),'--action',name]
        if command_path is not None:argv+=['--commands',str(command_path)]
        return json.loads(transport.command(argv))
    status=action('status')
    ids=[w['worker_instance_id'] for w in pool['workers']]
    if ([w['worker_instance_id'] for w in status['workers']]!=ids
            or any(w['source_sha256']!=pool['source_sha256'] for w in status['workers'])):
        raise RuntimeError('resident native lifetime differs from frozen pool')
    transport.stage(selection['checkpoint']['path'],plan['checkpoint_destination'],selection['checkpoint']['sha256'])
    transport.stage(selection['test_panel']['path'],plan['panel_destination'],selection['test_panel']['raw_sha256'])
    config_path=output/'rendered-observation-config.json'
    if config_path.exists():
        if read(config_path)!=canonical(plan['rendered_config'])+b'\n':raise RuntimeError('rendered test dependencies changed')
    else:publish(config_path,plan['rendered_config'])
    transport.stage(config_path,plan['rendered_config_destination'],plan['rendered_config_raw_sha256'])
    cp=output/'command.json';commands=[plan['command']]
    if cp.exists():
        if json.loads(read(cp))!=commands:raise RuntimeError('success command changed')
    else:publish(cp,commands)
    transport.stage(cp,plan['command_destination'],binding(cp)['raw_sha256'])
    # Same resident instance+command/payload returns its persisted acceptance,
    # including after a lost response. It cannot cause another evaluation.
    accepted=action('submit',plan['command_destination'])
    if not (output/'acceptance.json').exists():publish(output/'acceptance.json',accepted)
    while True:
        response=action('result',plan['command_destination']);rows=response.get('results',[])
        if len(rows)!=1:raise RuntimeError('success command owner cardinality differs')
        row=rows[0]
        if row.get('status')=='COMPLETE':break
        if row.get('status') not in ('QUEUED','RUNNING'):raise RuntimeError('native success evaluation failed/uncertain')
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('success observer detached; native work retained')
        sleep(30)
    receipt=row.get('receipt',{})
    if receipt.get('complete') is not True or receipt.get('optimizer_steps')!=0 or receipt.get('native_initializations')!=1:
        raise RuntimeError('completed actual native success phase required')
    result=validate_production_result(panel,receipt['result'],selection['checkpoint']['sha256'],allowed_arms=('bc',),
        rendered_config=plan['rendered_config'])
    # The helper verifies each actual resource receipt during pool qualification;
    # this evaluation phase must also expose a successful bounded guard summary.
    guard=receipt.get('resource_guard',{})
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import validate_summary
    validate_summary(guard)
    publish(output/'native-command-result.json',response) if not (output/'native-command-result.json').exists() else None
    logging=log_success(result,selection,output)
    if not (output/'wandb.json').exists():publish(output/'wandb.json',logging)
    final=base_output.parent/('success-test-result-'+plan['protocol_suffix']+'.json')
    if final.exists():
        if json.loads(read(final))!=result:raise RuntimeError('immutable success result changed')
    else:publish(final,result)
    return dict(status='BC295_COMPLETE',result=binding(final),wandb=logging,
        completed_episodes=295,successes=result['successes'],success_rate=result['success_rate'],
        optimizer_steps=0,native_workers_retained=True,dispatch_output=str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--pool-local',type=Path)
    p.add_argument('--remote-pool',type=Path);p.add_argument('--remote-source-root',type=Path)
    p.add_argument('--paired-runtime-bindings',type=Path,
        help='Explicit v4 source/paired owner/eight-renderer/sampler admission for selected.v2 only')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hours',type=float,default=48)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not 0<a.hours<=72:p.error('bounded observer hours required')
    legacy=(a.pool_local,a.remote_pool,a.remote_source_root)
    if a.paired_runtime_bindings:
        if any(legacy):p.error('paired runtime cannot be mixed with legacy pool/source overrides')
    elif not all(legacy):p.error('legacy execution requires pool-local, remote-pool and remote-source-root')
    if not a.execute:
        print(json.dumps(dict(execute=False,waits_for_actual_selection_and_pool=True,expected_cases=295)));return
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+a.hours*3600
        admission=a.paired_runtime_bindings or a.pool_local
        while not a.selection.exists() or not admission.exists():
            if time.monotonic()>=deadline:raise TimeoutError('selection/pool observer detached; compute retained')
            phase='WAITING_SELECTED_BC_AND_PAIRED_RUNTIME' if a.paired_runtime_bindings else 'WAITING_SELECTED_BC_AND_ACTUAL_B2_POOL'
            print(json.dumps(dict(phase=phase,at_unix=time.time())),flush=True);time.sleep(30)
        if a.paired_runtime_bindings:
            from pipeline.deployment.paired_bc_success import execute as execute_paired
            result=execute_paired(binding(a.selection),binding(admission),a.output,
                deadline=time.time()+max(0,deadline-time.monotonic()))
        else:
            result=execute(binding(a.selection),binding(a.pool_local),a.remote_pool,a.remote_source_root,a.output,deadline=deadline)
        result_path=Path(result['dispatch_output'])/'result.json'
        if not result_path.exists():publish(result_path,result)
        print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
