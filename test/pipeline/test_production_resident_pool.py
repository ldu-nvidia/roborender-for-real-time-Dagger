"""CPU-only new B2 profile/dispatch fixtures, not native qualification evidence."""
from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=REPO_ROOT
sys.path.insert(0,str(ROOT/'pipeline/deployment/rtx_actor'))
from production_resident_pool import SCHEMA,QUALIFICATION,validate_pool,qualification
import production_resident_pool as pool_module
from run_production_resident_isaac import production_command,resource_gate
from bootstrap_production_resident_pool import validate_spec
from leo_binding import GPU_UUIDS
from pipeline.bc_training.handoff_common import publish,binding
from pipeline.deployment import bc_success_dispatch as dispatcher
from pipeline.deployment.rtx_actor.explicit_motion_panel import summarize
from pipeline.deployment.rtx_actor.initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE, control_contract
from pipeline.deployment.rtx_actor.production_success_contract import protocol_identity

GIB=1<<30


@pytest.fixture
def pool(tmp_path):
    workers=[];results=[]
    for gpu in (0,1):
        runtime=tmp_path/f'gpu{gpu}';runtime.mkdir()
        resource=dict(schema='roborender.native-phase-resource-guard.v1',passed=True,samples=2,interval_s=.5,
            minimum_host_available_bytes=12*GIB,swap_peak_bytes=0,oom_events=0,limit_pressure_events=0)
        ref=publish(runtime/'guard.json',resource)
        guard={k:resource[k] for k in ('passed','samples','minimum_host_available_bytes','swap_peak_bytes','oom_events','limit_pressure_events')}
        guard.update(path=ref['path'],raw_sha256=ref['raw_sha256'])
        worker=dict(physical_gpu=gpu,gpu_uuid=GPU_UUIDS[gpu],port=18640+gpu,token_file=str(tmp_path/'token'),
            runtime_root=str(runtime),container_name=f'owned-prod-g{gpu}',worker_instance_id=str(gpu+1)*32,
            native_pid=100+gpu,memory_gib=21,evaluation_memory_gib=21,initial_num_envs=2)
        workers.append(worker)
        results.append(dict(gpu=gpu,receipt=dict(complete=True,source_sha256='a'*64,optimizer_steps=0,
            native_pid=worker['native_pid'],native_initializations=1,num_envs=2,
            environment_step_calls=0,resource_guard=guard,capture_receipt=dict(result='PASS',num_envs=2,
                environment_step_calls=0,rows=[{'lane':0},{'lane':1}]))))
    q=publish(tmp_path/'qualification.json',dict(schema=QUALIFICATION,complete=True,source_sha256='a'*64,
        profile='initial_training_b2_capture',native_initializations_per_worker=1,optimizer_steps=0,results=results))
    return dict(schema=SCHEMA,source_sha256='a'*64,jobs_root=str(tmp_path/'jobs'),assets_root=str(tmp_path/'assets'),
        workers=workers,qualification={'path':q['path'],'sha256':q['raw_sha256']})


def test_new_pool_requires_actual_b2_capture_and_guard(pool):
    assert validate_pool(pool)==pool
    assert qualification(pool)['complete']
    assert all(w['memory_gib']==w['evaluation_memory_gib']==21 for w in pool['workers'])


@pytest.mark.parametrize('fault',[None,'missing','wrong_pid','wrong_policy','not_disabled','global','boolean'])
def test_explicit_native_memory_policy_needs_actual_bound_startup(pool,tmp_path,fault):
    q=json.loads(Path(pool['qualification']['path']).read_bytes())
    for worker,row in zip(pool['workers'],q['results']):
        row['receipt']['native_thp_policy']=dict(schema='roborender.native-thp-policy.v1',
            policy='disable-v1',disabled_after=1,native_pid=worker['native_pid'],global_host_policy_modified=False)
    value=q['results'][0]['receipt']['native_thp_policy']
    if fault=='missing':q['results'][0]['receipt'].pop('native_thp_policy')
    elif fault=='wrong_pid':value['native_pid']+=1
    elif fault=='wrong_policy':value['policy']='inherited'
    elif fault=='not_disabled':value['disabled_after']=0
    elif fault=='global':value['global_host_policy_modified']=True
    elif fault=='boolean':value['disabled_after']=True
    ref=publish(tmp_path/'thp-qualification.json',q)
    pool['qualification']={'path':ref['path'],'sha256':ref['raw_sha256']}
    if fault:
        with pytest.raises(RuntimeError,match='startup attestation'):
            pool_module.validate_native_policy(pool,'disable-v1')
    else:pool_module.validate_native_policy(pool,'disable-v1')


@pytest.mark.parametrize('fault',['old_profile','unverified_guard','changed_capture','wrong_source','capture_source','optimizer'])
def test_legacy_or_changed_qualification_cannot_be_relabelled(pool,tmp_path,fault):
    q=json.loads(Path(pool['qualification']['path']).read_bytes())
    if fault=='old_profile': q['profile']='training_b6/evaluation_b5/training_b6'
    elif fault=='unverified_guard':q['results'][0]['receipt']['resource_guard']['passed']=False
    elif fault=='changed_capture':q['results'][0]['receipt']['capture_receipt']['num_envs']=6
    elif fault=='wrong_source':q['source_sha256']='f'*64
    elif fault=='capture_source':q['results'][0]['receipt']['source_sha256']='f'*64
    else:q['results'][0]['receipt']['optimizer_steps']=1
    b=publish(tmp_path/'bad.json',q);pool['qualification']={'path':b['path'],'sha256':b['raw_sha256']}
    with pytest.raises(RuntimeError): qualification(pool)


def test_launcher_changes_only_optin_batch_and_production_inputs(tmp_path):
    (tmp_path/'port-manifest.json').write_text('{"files":[]}')
    cmd=production_command(name='standing-isaac-leo-prod-test-g0',runtime=tmp_path,overlay=tmp_path,
        source=ROOT,jobs=tmp_path,assets=tmp_path,token=tmp_path/'token',source_sha='a'*64,gpu=0,
        memory_gib=21,motion_inventory_sha256='b'*64)
    assert 'num_envs=2' in cmd and 'num_envs=6' not in cmd
    assert 'RESIDENT_PRODUCTION_FULL295=1' in cmd
    assert 'RESIDENT_MOTION_INVENTORY=/resident/jobs/full-motion-inventory.json' in cmd
    assert 'RESIDENT_MOTION_INVENTORY_SHA256='+'b'*64 in cmd
    assert 'simulator._target_=resident_isaac_trainer.IsaacSim' in cmd
    assert 'device='+GPU_UUIDS[0] in cmd and '--restart' not in cmd
    assert 'RESIDENT_SCENE_LOCK_FILE=/resident/scene-reconfigure.lock' in cmd
    assert ('type=bind,src='+str(tmp_path.parent/'scene-reconfigure.lock')+
        ',dst=/resident/scene-reconfigure.lock,readonly') in cmd
    assert not any(arg.startswith('RESIDENT_SCALED_SUCCESS=') for arg in cmd)


def test_scaled_evaluator_capability_is_explicit_without_weakening_memory(tmp_path):
    (tmp_path/'port-manifest.json').write_text('{"files":[]}')
    values=dict(name='standing-isaac-leo-prod-scaled-test-g0',runtime=tmp_path,overlay=tmp_path,
        source=ROOT,jobs=tmp_path,assets=tmp_path,token=tmp_path/'token',source_sha='a'*64,gpu=0,
        memory_gib=21,motion_inventory_sha256='b'*64)
    default=production_command(**values)
    scaled=production_command(**values,scaled_success=True)
    marker='RESIDENT_SCALED_SUCCESS=roborender.scaled-rendered-success-protocol.v1'
    index=scaled.index(marker)
    assert scaled[index-1]=='--env'
    assert scaled[:index-1]+scaled[index+1:]==default
    with pytest.raises(ValueError):production_command(**values,scaled_success=True,mode='smoke')
    with pytest.raises(ValueError):production_command(**values,scaled_success=1)


@pytest.mark.parametrize('sample',[{'sampling_error':'unknown'},
    {'memory_swap.current':1,'events':{'max':0,'oom':0,'oom_kill':0},'memory_current':GIB},
    {'memory_swap.current':0,'events':{'max':1,'oom':0,'oom_kill':0},'memory_current':GIB}])
def test_outer_launcher_never_exempts_unknown_swap_or_pressure(sample):
    with pytest.raises(RuntimeError):resource_gate({'MemAvailable':12*1024**2},sample)


@pytest.mark.parametrize('action',['wait','result'])
@pytest.mark.parametrize('unsafe',[False,True])
def test_wait_and_result_verify_actual_resource_receipt(pool,tmp_path,monkeypatch,capsys,action,unsafe):
    q=json.loads(Path(pool['qualification']['path']).read_bytes())
    receipt=q['results'][0]['receipt']
    if unsafe:receipt['resource_guard']['passed']=False
    class Client:
        def wait(self,command_id):return dict(status='COMPLETE',receipt=receipt)
        def _request(self,route):return dict(status='COMPLETE',receipt=receipt)
    monkeypatch.setattr(pool_module,'selected_clients',lambda _,gpus: {gpu:Client() for gpu in gpus})
    p=publish(tmp_path/'pool.json',pool)
    c=publish(tmp_path/'commands.json',[dict(gpu=0,command_id='test',operation='evaluate_explicit_panel',payload={})])
    monkeypatch.setattr(sys,'argv',['pool','--pool',p['path'],'--action',action,'--commands',c['path']])
    if unsafe:
        with pytest.raises(RuntimeError):pool_module.main()
    else:
        pool_module.main();assert json.loads(capsys.readouterr().out)['results'][0]['status']=='COMPLETE'


@pytest.mark.parametrize('operation',['prepare_scene','evaluate_pair','capture_reset','collect_mse_round','shutdown'])
def test_production_pool_rejects_unqualified_legacy_operation(pool,tmp_path,monkeypatch,operation):
    monkeypatch.setattr(pool_module,'clients',lambda _: [object(),object()])
    p=publish(tmp_path/'pool.json',pool)
    c=publish(tmp_path/'commands.json',[dict(gpu=0,command_id='test',operation=operation,payload={})])
    monkeypatch.setattr(sys,'argv',['pool','--pool',p['path'],'--action','submit','--commands',c['path']])
    with pytest.raises(ValueError):pool_module.main()


@pytest.mark.parametrize('flag,batch,profile,allowed',[
    ('1',2,'training',True),('0',2,'training',False),('1',6,'training',False),('1',2,'evaluation',False)])
def test_actual_native_initial_capture_is_production_b2_only(tmp_path,monkeypatch,flag,batch,profile,allowed):
    # Execute the actual pure validation method without importing GPU/Isaac.
    source=ROOT/'pipeline/deployment/rtx_actor/resident_isaac_trainer.py'
    backend=next(n for n in ast.parse(source.read_text()).body
        if isinstance(n,ast.ClassDef) and n.name=='ResidentRavenBackend')
    method=next(n for n in backend.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]));namespace={'os':os}
    exec(compile(module,str(source),'exec'),namespace)
    obj=SimpleNamespace(_consumed_outputs=set(),runtime_root=tmp_path,profile=profile,
        trainer=SimpleNamespace(env=SimpleNamespace(num_envs=batch)))
    monkeypatch.setenv('RESIDENT_PRODUCTION_FULL295',flag)
    if allowed:namespace['validate'](obj,'capture_initial_b2',{'output_name':'cpu-initial-b2'})
    else:
        with pytest.raises(ValueError):namespace['validate'](obj,'capture_initial_b2',{'output_name':'cpu-initial-b2'})


@pytest.fixture
def selected(tmp_path):
    panel_path=ROOT/'runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json'
    from pipeline.deployment.rtx_actor.production_success_contract import build_rendered_panel
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config, complete_result
    panel=build_rendered_panel(json.loads(panel_path.read_bytes()))
    panel_path=Path(publish(tmp_path/'rendered-panel.json',panel)['path'])
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'CPU synthetic checkpoint')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config=rendering_config(panel,checkpoint_sha256=sha)
    dependencies=publish(tmp_path/'rendered-dependencies.json',config)
    selection=dict(schema='roborender.bc-production-selected.v1',run_name='2026-10-16-prod-1-BC',
        selection='minimum_complete_validation_mse',success_test_executed=False,expected_test_episodes=295,
        test_panel=binding(panel_path),rendered_observation_config=dependencies,
        checkpoint=dict(path=str(checkpoint),sha256=sha,bytes=checkpoint.stat().st_size))
    result=complete_result(panel,config=config)
    return selection,result


def test_dispatch_uses_all295_forward_only_cases_and_never_allocates(selected,pool,tmp_path,monkeypatch):
    selection,result=selected
    receipt=dict(complete=True,optimizer_steps=0,native_initializations=1,result=result,
        resource_guard=dict(passed=True,samples=2,minimum_host_available_bytes=12*GIB,swap_peak_bytes=0,oom_events=0,limit_pressure_events=0))
    class Transport:
        def __init__(self):self.staged=[];self.actions=[]
        def stage(self,*args):self.staged.append(args)
        def command(self,argv):
            action=argv[argv.index('--action')+1];self.actions.append(action)
            if action=='status':return dispatcher.canonical(dict(workers=[dict(worker_instance_id=w['worker_instance_id'],source_sha256=pool['source_sha256']) for w in pool['workers']]))
            if action=='submit':return b'{"results":[{"status":"accepted"}]}'
            return dispatcher.canonical(dict(results=[dict(status='COMPLETE',receipt=receipt)]))
    transport=Transport();monkeypatch.setattr(dispatcher,'log_success',lambda *a,**k:dict(online_readback_verified=True))
    sp=publish(tmp_path/'selection.json',selection);pp=publish(tmp_path/'pool.json',pool)
    output=tmp_path/'bc'/'success-dispatch'
    output.mkdir(parents=True)
    historical=[output/'intent.json',output/'command.json',output/'wandb.json',
        output.parent/'success-test-result.json']
    for path in historical:path.write_bytes(b'{"historical_original_reset":true}')
    out=dispatcher.execute(sp,pp,'/remote/pool.json','/remote/frozen',output,transport=transport)
    assert out['status']=='BC295_COMPLETE' and out['completed_episodes']==295 and out['optimizer_steps']==0
    assert transport.actions==['status','submit','result'] and len(transport.staged)==4
    assert json.loads(Path(out['result']['path']).read_bytes())==result
    assert Path(out['dispatch_output']).parent==output
    assert all(path.read_bytes()==b'{"historical_original_reset":true}' for path in historical)


def test_partial_or_changed_test_does_not_publish_success(selected,pool,tmp_path,monkeypatch):
    selection,result=selected
    result['completed_episodes']=294
    class Transport:
        def stage(self,*a):pass
        def command(self,argv):
            action=argv[argv.index('--action')+1]
            if action=='status':return dispatcher.canonical(dict(workers=[dict(worker_instance_id=w['worker_instance_id'],source_sha256=pool['source_sha256']) for w in pool['workers']]))
            if action=='submit':return b'{}'
            return dispatcher.canonical(dict(results=[dict(status='COMPLETE',receipt=dict(complete=True,
                optimizer_steps=0,native_initializations=1,result=result))]))
    sp=publish(tmp_path/'selection.json',selection);pp=publish(tmp_path/'pool.json',pool);output=tmp_path/'bc'/'dispatch'
    with pytest.raises(ValueError):dispatcher.execute(sp,pp,'/remote/pool','/remote/source',output,transport=Transport())
    assert not (output.parent/'success-test-result.json').exists()
