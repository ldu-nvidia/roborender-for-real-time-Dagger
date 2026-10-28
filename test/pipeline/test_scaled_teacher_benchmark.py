"""CPU contract/native-branch tests; synthetic outcomes are not live proof."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import PIPELINE_ROOT
import ast
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch

HERE = PIPELINE_ROOT / 'deployment/rtx_actor'
sys.path.insert(0, str(HERE))
import scaled_teacher_benchmark as teacher
import scaled_success_contract as scaled
import initial_robot_refresh_diagnostic as reset
import explicit_motion_panel as explicit
from test.deployment.rtx_actor.test_production_success_contract import accepted
from test.deployment.rtx_actor.test_teacher_action_diagnostic import ppo_result, FakeTeacher
from test.deployment.rtx_actor.test_rendered_student_native_routing import evaluator_function


@pytest.fixture
def material(accepted):
    panel = scaled.build_panel(accepted, batch_size=16)
    owners = [dict(physical_gpu=i, gpu_uuid=f'GPU-cpu-{i}', native_pid=os.getpid(),
                   worker_instance_id=str(i+1)*32, source_sha256='c'*64) for i in range(2)]
    return panel, owners


def benchmark(material, index=0):
    panel, owners = material
    return teacher.build_benchmark(panel, owner=owners[index%2], command_id=f'teacher-c{index}',
        output_name=f'teacher-c{index}', cohort_index=index, shadow_checkpoint_sha256='b'*64)


def native_result(material, index=0):
    panel, _ = material; spec = panel['cohorts'][index]; count = len(spec['case_ids'])
    row = ppo_result(spec)
    row.update(completed=count, expected=count, native_successes=1, exported_successes=1,
        original_raven_metric_completed=count, original_raven_metric_successes=1, vector_steps=3,
        outcomes=[dict(lane=i, steps=3, native_success=i==0, exported_success=i==0, timeout=i!=0)
                  for i in range(count)], training_curriculum_restored=False, teacher_resource_guard_checks=6,
        teacher_isolation=dict(schema='roborender.isolated-ppo-teacher-execution.v1',
            student_forward_calls=0, renderer_calls=0, distill_policy_step_calls=0,
            teacher_rollout_calls=3, shared_teacher_unchanged=True, private_teacher_used=True))
    row['teacher_parameter_attestation_at_start'] = deepcopy(row['teacher_parameter_attestation'])
    mode = reset.SCALED_TEACHER_CLEAR_ONCE
    env = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count,dtype=torch.bool))
    row.update({reset.FIELD:mode, 'initial_robot_refresh_control':reset.control_contract(mode),
        'initial_robot_refresh_attestation':reset.clear_already_applied_initial_refresh(env,mode=mode)})
    row['scaled_initial_state_attestation'] = dict(schema=scaled.INITIAL_SCHEMA,
        environment_step_calls_before_attestation=0,
        initialization_distribution_sha256=scaled.digest(panel['scaled_protocol']),
        native_reset=dict(schema='roborender.explicit-motion-success-reset.v1', assignment=deepcopy(spec['assignment']),
            assignment_sha256=scaled.digest(spec['assignment']), sample_indices_forced=False,
            seed_reapplied_per_episode=False, lanes=[dict(lane,sample_idx=i*100+5,sample_idx_in_seq=2,
                reset_rng_sha256='e'*64) for i,lane in enumerate(spec['assignment']['lanes'])]),
        initial_state={name:dict(dtype='float32',shape=[count,3],sha256='f'*64) for name in scaled.INITIAL_ARRAYS})
    return row


def wrapped(material,index=0):
    return teacher.wrap_result(material[0],benchmark(material,index),native_result(material,index))


def test_same_physical_panel_and_all295_failures_retained_no_threshold_gate(material):
    panel,_=material;before=deepcopy(panel)
    rows=[wrapped(material,i) for i in range(19)]
    value=teacher.aggregate_results(panel,rows)
    assert panel==before and [r['benchmark']['num_envs'] for r in rows]==[16]*18+[7]
    assert value['completed']==295 and value['successes']==19 and value['success_rate']==19/295
    assert len(value['outcomes'])==295 and sum(not r['native_success'] for r in value['outcomes'])==276
    assert [r['case_id'] for r in value['outcomes']]==[case for c in panel['cohorts'] for case in c['case_ids']]
    assert value['renderer_calls']==value['student_forward_calls']==value['optimizer_steps']==0
    for values in (rows[:2],rows[:-1],rows+[rows[0]],list(reversed(rows))):
        with pytest.raises(ValueError):teacher.aggregate_results(panel,values)


def test_no_qualification_substitute_or_student_gate_widening(accepted,material):
    with pytest.raises(ValueError):teacher.require_panel(scaled.build_panel(accepted,qualification=True))
    for count in (1,2,7,16):
        assert reset.require_corrected_execution(reset.SCALED_TEACHER_CLEAR_ONCE,num_envs=count,
                                               arm='teacher',teacher_action_mode='raven_ppo')
    for arm,mode,count in [('bc','raven_ppo',16),('teacher','distill',16),('teacher','raven_ppo',17)]:
        with pytest.raises(ValueError):reset.require_corrected_execution(reset.SCALED_TEACHER_CLEAR_ONCE,
            arm=arm,teacher_action_mode=mode,num_envs=count)
    with pytest.raises(ValueError):reset.require_corrected_execution(reset.CLEAR_ONCE,
        arm='teacher',teacher_action_mode='raven_ppo',num_envs=16)
    with pytest.raises(ValueError):reset.require_corrected_execution(reset.SCALED_STUDENT_CLEAR_ONCE,
        arm='teacher',teacher_action_mode='raven_ppo',num_envs=16)
    assert teacher.evaluation_options()['teacher_action_mode']=='raven_ppo'
    assert scaled.evaluation_options(material[0])['teacher_action_mode']=='distill'


def test_generic_evaluator_cannot_acquire_scaled_teacher_reset_without_bound_operation(accepted,tmp_path):
    from test.deployment.rtx_actor.test_explicit_motion_panel import panel as legacy_panel
    panel=legacy_panel()
    prepared=[]
    with pytest.raises(ValueError,match='separate admitted teacher operation'):
        evaluator_function(tmp_path,object)(SimpleNamespace(),panel,root=tmp_path/'unused',
            checkpoint_path='/unused',checkpoint_sha256='b'*64,prepare_cohort=prepared.append,
            report=lambda *_:None,**teacher.evaluation_options())
    assert prepared==[] and not (tmp_path/'unused').exists()


@pytest.mark.parametrize('fault',['reset','initial_step','teacher','homie','start_proof','student','renderer',
    'shared','private','rollouts','distill','guard','optimizer','incomplete','success','duration','timeout','arm','checkpoint'])
def test_missing_or_inconsistent_actual_execution_evidence_fails_closed(material,fault):
    value=wrapped(material);r=value['cohort_result']
    if fault=='reset':r['initial_robot_refresh_attestation']['native_steps_added']=1
    elif fault=='initial_step':r['scaled_initial_state_attestation']['environment_step_calls_before_attestation']=1
    elif fault=='teacher':r['teacher_parameter_attestation']['checkpoint_sha256']='0'*64
    elif fault=='homie':r['teacher_parameter_attestation']['homie_parameter_attestations']['walk']['strict_parameter_equality']=False
    elif fault=='start_proof':r.pop('teacher_parameter_attestation_at_start')
    elif fault in {'student','renderer','distill'}:r['teacher_isolation'][{'student':'student_forward_calls','renderer':'renderer_calls','distill':'distill_policy_step_calls'}[fault]]=1
    elif fault in {'shared','private'}:r['teacher_isolation'][{'shared':'shared_teacher_unchanged','private':'private_teacher_used'}[fault]]=False
    elif fault=='rollouts':r['teacher_isolation']['teacher_rollout_calls']=2
    elif fault=='guard':r['teacher_resource_guard_checks']=5
    elif fault=='optimizer':r['optimizer_steps']=1
    elif fault=='incomplete':r['completed']-=1
    elif fault=='success':r['original_raven_metric_successes']=16
    elif fault=='duration':r['outcomes'][0]['steps']=4
    elif fault=='timeout':r['outcomes'][0]['timeout']=1
    elif fault=='arm':r['arm']='bc'
    elif fault=='checkpoint':r['checkpoint_sha256']='0'*64
    with pytest.raises((ValueError,RuntimeError)):teacher.validate_result(material[0],value)


@pytest.mark.parametrize('fault',['command','owner','source','metric','reset','checkpoint','cohort'])
def test_descriptor_scientific_and_owner_identity_is_exact(material,fault):
    b=benchmark(material)
    if fault=='command':b['command_id']='illegal/name'
    elif fault=='owner':b['owner']['physical_gpu']=1
    elif fault=='source':b['owner']['source_sha256']='unknown'
    elif fault=='metric':b['metric']='success_only'
    elif fault=='reset':b['initial_clocks']='random'
    elif fault=='checkpoint':b['shadow_not_behavior_checkpoint_sha256']='none'
    elif fault=='cohort':b['cohort_index']=19
    with pytest.raises(ValueError):teacher.validate_benchmark(b,material[0])


def test_actual_native_admission_envelope_and_accepted_work_no_replay(material,tmp_path,monkeypatch):
    panel,owners=material;b=benchmark(material);paths={}
    for name,value in [('panel.json',panel),('benchmark.json',b),('resident-state.json',dict(owners[0],phase='IDLE',active_command_id=None,commands={}))]:
        paths[name]=tmp_path/name;paths[name].write_bytes(explicit.canonical(value))
    payload=dict(panel_file='panel.json',panel_sha256=teacher.digest(panel),benchmark_file='benchmark.json',
        benchmark_sha256=teacher.digest(b),checkpoint_sha256='b'*64,output_name=b['output_name'],**teacher.evaluation_options())
    cls=next(n for n in ast.parse((HERE/'resident_isaac_trainer.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ResidentRavenBackend')
    methods=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'_scaled_teacher_inputs','validate_command_identity'}]
    checked=[]
    def identity(path,sha,maximum=None):
        checked.append(str(path))
        if str(path).startswith('/diagnostic-assets/'):return
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('actual file digest differs')
    scope=dict(os=os,Path=Path,json=json,file_identity=identity)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<native-teacher-admission>','exec'),scope)
    owner=SimpleNamespace(runtime_root=tmp_path,_consumed_outputs=set(),artifact=lambda name:paths[name],
        source_sha256='c'*64,motion_inventory_sha256=panel['motion_inventory_sha256'],trainer=SimpleNamespace(_scaled_teacher_private=object()))
    for method in methods:setattr(owner,method.name,MethodType(scope[method.name],owner))
    for key,value in dict(RESIDENT_SCALED_SUCCESS=scaled.SCHEMA,RESIDENT_SCALED_TEACHER=teacher.SCHEMA,
        RESIDENT_PRODUCTION_FULL295='1',RESIDENT_PIPELINE_MODE='production',
        RESIDENT_PHYSICAL_GPU='0',RESIDENT_SCENE_LOCK_FILE='/cpu/scene.lock',RESIDENT_RESOURCE_POLICY='roborender.native-memory-policy.v3').items():monkeypatch.setenv(key,value)
    assert owner._scaled_teacher_inputs(payload)==(panel,b)
    command=dict(operation=teacher.OPERATION,command_id=b['command_id'],worker_instance_id=owners[0]['worker_instance_id'],payload=payload)
    owner.validate_command_identity(command)
    with pytest.raises(ValueError):owner.validate_command_identity(dict(command,command_id='other-command'))
    state=dict(owners[0],phase='RUNNING',active_command_id=b['command_id'],commands={b['command_id']:dict(status='RUNNING',command=command)})
    paths['resident-state.json'].write_bytes(explicit.canonical(state));owner._scaled_teacher_inputs(payload)
    state.update(phase='IDLE',active_command_id=None);paths['resident-state.json'].write_bytes(explicit.canonical(state))
    with pytest.raises(ValueError):owner._scaled_teacher_inputs(payload)
    assert any(v.endswith('/'+'b'*64+'.pt') for v in checked)


def test_private_teacher_creation_is_separate_explicit_startup_capability(monkeypatch):
    tree=ast.parse((HERE/'resident_isaac_trainer.py').read_text())
    block=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                       and "os.environ.get('RESIDENT_SCALED_TEACHER')" in ast.unparse(n.test)))
    import isolated_ppo_teacher
    built=[]
    monkeypatch.setattr(isolated_ppo_teacher,'build_isolated_teacher',lambda trainer:built.append(trainer) or 'private')
    trainer=SimpleNamespace();scope=dict(os=os,trainer=trainer)
    code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'<actual-private-startup>','exec')
    monkeypatch.setenv('RESIDENT_SCALED_SUCCESS',scaled.SCHEMA)
    monkeypatch.delenv('RESIDENT_SCALED_TEACHER',raising=False)
    exec(code,scope)
    assert not built and not hasattr(trainer,'_scaled_teacher_private')
    monkeypatch.setenv('RESIDENT_SCALED_TEACHER',teacher.SCHEMA)
    exec(code,scope)
    assert built==[trainer] and trainer._scaled_teacher_private=='private'


def test_launcher_preserves_student_command_and_adds_only_explicit_teacher_capability(tmp_path):
    import run_production_resident_isaac as launcher
    (tmp_path/'port-manifest.json').write_text('{"files":[]}')
    values=dict(name='standing-isaac-leo-prod-teacher-cpu-g0',runtime=tmp_path,overlay=tmp_path,
        source=HERE.parents[2],jobs=tmp_path,assets=tmp_path,token=tmp_path/'token',source_sha='a'*64,gpu=0)
    base=launcher.production_command(**values,motion_inventory_sha256='a'*64,scaled_success=True)
    explicit_false=launcher.production_command(**values,motion_inventory_sha256='a'*64,scaled_success=True,scaled_teacher=False)
    enabled=launcher.production_command(**values,motion_inventory_sha256='a'*64,scaled_success=True,scaled_teacher=True)
    assert base==explicit_false and not any('RESIDENT_SCALED_TEACHER=' in v for v in base)
    at=enabled.index('RESIDENT_SCALED_TEACHER='+teacher.SCHEMA)
    assert enabled[at-1]=='--env' and enabled[:at-1]+enabled[at+1:]==base
    for options in ({'scaled_teacher':True},{'scaled_teacher':'true','scaled_success':True},
                    {'scaled_teacher':True,'scaled_success':True,'mode':'smoke'}):
        with pytest.raises(ValueError):launcher.production_command(**values,motion_inventory_sha256='a'*64,**options)


def test_actual_evaluator_router_is_private_teacher_only_and_preserves_shared_teacher(material,tmp_path,monkeypatch):
    from isolated_ppo_teacher import build_isolated_teacher
    panel,_=material;b=benchmark(material);row=native_result(material);prepared=[]
    shared=FakeTeacher();student=torch.nn.Linear(2,2)
    trainer=SimpleNamespace(ref_model=shared,policy_model=student,unwrapped_model=student,env=object(),policy_step=lambda *_:pytest.fail('student'))
    trainer._scaled_teacher_private=build_isolated_teacher(trainer)
    class Evaluator:
        def evaluate_cohort(self,assignment,**kwargs):
            assert assignment==panel['cohorts'][0]['assignment']
            assert self.ref_model is trainer._scaled_teacher_private and self.ref_model is not shared
            assert self._scaled_initialization_distribution==scaled.digest(panel['scaled_protocol'])
            assert not self._require_rendered_success_observations and not hasattr(self,'_success_observation_transform')
            with torch.no_grad():
                for _ in range(3):self.ref_model.rollout(obs_dict={'teacher_obs':torch.zeros(16,226)})
            return deepcopy(row)
    execute=evaluator_function(tmp_path,Evaluator)
    execute.__globals__.update(deepcopy=deepcopy,os=os)
    actual=execute(trainer,panel,root=tmp_path/'teacher',checkpoint_path='/CPU/shadow.pt',checkpoint_sha256='b'*64,
        prepare_cohort=prepared.append,report=lambda *_:None,_teacher_benchmark=b,**teacher.evaluation_options())
    assert actual==wrapped(material) and prepared==[panel['cohorts'][0]['assignment']]
    assert not hasattr(shared,'_original_ppo_std_policy') and shared.calls==[]
    assert not hasattr(trainer,'_ppo_teacher_parameter_attestation')
    assert json.loads((tmp_path/'teacher/result.json').read_bytes())==actual


@pytest.mark.parametrize('trip',[0,1,2,None])
def test_actual_loop_guard_fault_stops_before_physics_and_no_student_dispatch(monkeypatch,trip):
    import rendered_success_observations as observations
    from success_diagnostic_core import FirstEpisodeAccounting
    method=next(n for n in ast.walk(ast.parse((SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text())) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    loop=deepcopy(next(n for n in ast.walk(method) if isinstance(n,ast.For) and ast.unparse(n.target)=='step'))
    events=[];captured={};guard_calls=[]
    def guard():
        guard_calls.append(1)
        if len(guard_calls)==trip:raise RuntimeError('native resource fault')
    def transform(owner,obs,dones,step):events.append('observation-no-transform');return obs
    def policy(owner,obs,dones,**kwargs):
        assert kwargs['arm']=='teacher' and kwargs['mode']=='raven_ppo'
        events.append('teacher');return {'actions':torch.zeros(2,26)},None
    def step(actions):
        events.append('physics');captured.update(current_last_stage_completed_task_buf=np.array([1,0]),reset_buf=np.array([1,1]),time_out_buf=np.array([0,1]))
        return {},torch.ones(2),torch.ones(2),{'grasp_goal_reached':torch.tensor([1,0])}
    env=SimpleNamespace(step=step,config={},_transition_goal_before_reset=torch.tensor([1,0]),
        update_eval_metrics_per_step=lambda _:None,process_eval_episode_completions=lambda *_:None,reset_eval_episode_tracking=lambda _:None)
    monkeypatch.setattr(observations,'apply_observation_transform',transform)
    scope=dict(self=SimpleNamespace(env=env,accelerator=SimpleNamespace(device='cpu'),_native_resource_guard=None if trip==0 else guard),
        pooled=None,scaled_teacher={'schema':teacher.SCHEMA},teacher_guard_checks=0,cycle_timings=None,numeric_steps=0,
        torch=torch,np=np,time=time,json=json,obs={},dones=torch.zeros(2),arm='teacher',teacher_mode='raven_ppo',
        select_policy_actions=policy,episode_reward=torch.zeros(2),episode_length=torch.zeros(2,dtype=torch.long),
        captured=captured,count=2,account=FirstEpisodeAccounting(2),trace=io.StringIO(),emit=lambda *_:None,
        started=time.monotonic(),cpu=lambda t:t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t),lane_details=lambda *_:{})
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<actual-teacher-loop>','exec')
    if trip is not None:
        with pytest.raises(RuntimeError,match='native resource'):exec(code,scope)
        assert 'physics' not in events and scope['account'].result()['completed']==0
    else:
        exec(code,scope)
        assert events==['teacher','physics'] and scope['teacher_guard_checks']==2
        assert scope['account'].result()['completed']==2 and scope['account'].result()['exported_successes']==1


def test_native_teacher_prepare_does_not_reset_shared_teacher_or_change_student_default():
    tree=ast.parse((HERE/'resident_isaac_trainer.py').read_text())
    switch=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_switch_scene_admitted')
    prepare=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare' and 'scaled_teacher' in ast.unparse(n))
    for method,flag in [(switch,'preserve_shared_teacher'),(prepare,'scaled_teacher')]:
        select=next(deepcopy(n) for n in method.body if isinstance(n,ast.Assign) and any(ast.unparse(t)=='models' for t in n.targets))
        for active in (False,True):
            scope=dict(self=SimpleNamespace(trainer=SimpleNamespace(policy_model='student',ref_model='shared')),**{flag:active})
            exec(compile(ast.fix_missing_locations(ast.Module(body=[select],type_ignores=[])),'<actual-scene-model-selection>','exec'),scope)
            assert scope['models']==(('student',) if active else ('student','shared'))
