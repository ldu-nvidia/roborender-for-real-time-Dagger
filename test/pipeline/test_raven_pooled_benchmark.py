"""Hash-pinned original differential checks; CPU only, no native certification."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import PIPELINE_ROOT
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import io
import time
import threading
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from pipeline.deployment.rtx_actor import raven_pooled_benchmark as module
from pipeline.deployment import raven_pooled_dispatch as dispatch
from test.pipeline.test_raven_evaluation_equivalence import original_metrics, reference_tree, TRAINER, REFERENCE

HERE = PIPELINE_ROOT / 'deployment/rtx_actor'
sys.path.insert(0, str(HERE))
from test.deployment.rtx_actor.test_scaled_success_contract import accepted, config_for, native_admission, scaled_cohort


@pytest.fixture
def material(accepted, monkeypatch):
    import production_success_contract as old
    from pipeline.deployment.rtx_actor import production_success_contract as new
    from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
    for key in ('PANEL_SHA256','PANEL_RAW_SHA256','LEGACY_PANEL_SHA256','LEGACY_PANEL_RAW_SHA256'):
        monkeypatch.setattr(new,key,getattr(old,key))
    panel = scaled.build_panel(accepted, batch_size=16, qualification=True)
    config = config_for(accepted, panel, sharded=True)
    owners = [dict(physical_gpu=i,gpu_uuid=f'GPU-test-{i}',native_pid=os.getpid(),
        worker_instance_id=str(i+1)*32,source_sha256='c'*64) for i in range(2)]
    return panel, config, owners


def plan(material, qualification=True):
    return dispatch.build_plan(*material, command_prefix='cpu-pooled-benchmark', qualification_only=qualification)


def completed_result(accepted, material, index=0, qualification=True):
    panel,config,_=material;spec=plan(material,qualification)['commands'][index]['artifacts'][1]['value']
    value=scaled_cohort(accepted,panel,config,index,arm='dagger')
    value.pop('scaled_initial_state_attestation')
    account=module.PooledEpisodeAccounting(16,spec['minimum_completed_episodes'])
    steps=0
    while not account.result()['complete']:
        goals=[i%2==steps%2 for i in range(16)]
        account.update(steps,goals,goals,[True]*16,[False]*16,[{}]*16);steps+=1
    clocks=SimpleNamespace(num_envs=16,max_episode_length=400,max_stage_time=[150],
        episode_length_buf=torch.zeros(16,dtype=torch.int64),time_in_stage_buf=torch.zeros(16,dtype=torch.int64),
        actual_time_in_stage_buf=torch.zeros(16,dtype=torch.int64))
    clock=module.randomize_initial_clocks(clocks)
    value.update(account.result(),schema=module.RESULT_SCHEMA,benchmark=spec,benchmark_sha256=module.digest(spec),
        vector_steps=steps,initial_clocks=clock,training_curriculum_restored=False,
        teacher_action_mode='distill',benchmark_certified=False,
        original_raven_metric_completed=len(account.outcomes),
        original_raven_metric_successes=account.result()['exported_successes'],
        native_reset_resample_every=500)
    value['native_episode_resets']=[dict(lane=lane,episode_index=episode,sample_idx=lane*100,
        sample_idx_in_seq=0,reset_rng_sha256='a'*64,motion_id=spec['motion_support'][lane]['motion_id'])
        for episode in range(steps+1) for lane in range(16)]
    receipt=value['rendered_observation_receipt'];namespace=module.digest(spec)
    receipt.update(vector_steps_rendered=steps,render_requests_completed=16*steps,
        anchor_selections=[dict(lane=lane,selected_at_vector_step=step,file_sha256=config['anchors']['files'][lane]['sha256'],
            frame_index=0,prompt_sha256='f'*64,episode_index=step,
            episode_session_id=f'{namespace}:lane{lane}:episode{step}') for step in range(steps) for lane in range(16)],
        pooled_episode_sessions=dict(benchmark_sha256=namespace,current_episode_indices=[steps-1]*16,
            natural_done_refreshes_anchor=True,generated_frame_history_carried=False,
            transport_request_sequence='monotonic_not_scientific_episode_state'))
    receipt['bounded_first_inputs']=[dict(deepcopy(receipt['bounded_first_inputs'][0]),step=i) for i in range(min(2,steps))]
    from test.deployment.rtx_actor.test_rendered_endpoint_pool import Client
    from rendered_endpoint_pool import RenderedEndpointPool, endpoint_slot_for_lane, TIMING_SCHEMA
    clients=[Client(i,config['renderer'],endpoint_slot=endpoint_slot_for_lane(config['renderer'],i,
        cohort_index=index,physical_gpu=index),channel_namespace=index*1000) for i in range(16)]
    pool=RenderedEndpointPool(clients,config['renderer'],cohort_index=index,physical_gpu=index)
    try:
        for step in range(steps):pool.render([{}]*16,step)
        receipt['renderer_endpoint_pool']=pool.receipt()
        receipt['renderer_transport_receipts']=[c.transport_receipt() for c in clients]
    finally:pool.close()
    receipt['phase_timing']=dict(schema=TIMING_SCHEMA,vector_steps_completed=steps,
        preprocessing_seconds=0.,render_wait_seconds=0.,postprocessing_seconds=0.,
        steps=[dict(step=i,preprocessing_seconds=0.,render_wait_seconds=0.,postprocessing_seconds=0.) for i in range(steps)])
    return value


def original_eval():
    cls = next(n for n in reference_tree(TRAINER).body if isinstance(n,ast.ClassDef)
        and n.name=='TRLDistillTrainerObjPredHomieAPI')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='eval')


@pytest.mark.parametrize('dtype',[torch.int64,torch.float32])
@pytest.mark.parametrize('count',[1,4,16,32])
def test_exact_original_initial_clocks_rng_order_aliases_and_stale_observations(dtype,count):
    class Staged: pass
    def env():
        value=Staged();value.num_envs=count;value.max_episode_length=400;value.max_stage_time=torch.tensor([157.,250.])
        for name in ('episode_length_buf','time_in_stage_buf','actual_time_in_stage_buf'):
            setattr(value,name,torch.zeros(count,dtype=dtype))
        return value
    clock=next(n for n in original_eval().body if isinstance(n,ast.If)
        and ast.unparse(n.test)=="self.config.get('init_at_random_ep_len', False)")
    statements=[deepcopy(n) for n in clock.body if not isinstance(n,ast.ImportFrom)]
    left,right=env(),env();obs=torch.randn(count,3);before=obs.clone()
    prior_episode=right.episode_length_buf;prior_stage=right.time_in_stage_buf
    with torch.random.fork_rng():
        torch.manual_seed(928);initial=torch.get_rng_state()
        scope=dict(torch=torch,StagedTaskBase=Staged,self=SimpleNamespace(env=left))
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),'<original-clocks>','exec'),scope)
        original_rng=torch.get_rng_state();torch.set_rng_state(initial)
        receipt=module.randomize_initial_clocks(right)
        assert torch.equal(torch.get_rng_state(),original_rng)
    for name in receipt['order']: assert torch.equal(getattr(left,name),getattr(right,name))
    assert right.episode_length_buf is not prior_episode and right.time_in_stage_buf is prior_stage
    assert torch.equal(obs,before) and receipt['observations_recomputed'] is False


@pytest.mark.parametrize('count,target',[(1,3),(4,7),(16,32),(16,150),(32,150)])
def test_original_pooled_metric_and_actual_stop_closure_match_with_overshoot(count,target):
    env=original_metrics(count);current=module.PooledEpisodeAccounting(count,target)
    stop=deepcopy(next(n for n in original_eval().body if isinstance(n,ast.FunctionDef) and n.name=='terminate_rollout'))
    scope=dict(self=SimpleNamespace(env=env),eval_num_envs_episodes=False,max_episodes=target,torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stop],type_ignores=[])),'<original-stop>','exec'),scope)
    rng=torch.Generator().manual_seed(1020+count);rewards=torch.zeros(count);lengths=torch.zeros(count,dtype=torch.int32)
    for step in range(2000):
        assert scope['terminate_rollout']()==current.result()['complete']
        if scope['terminate_rollout']():break
        goals=torch.rand(count,generator=rng)<.2;dones=torch.rand(count,generator=rng)<.17
        rewards+=1;lengths+=1;env.update_eval_metrics_per_step({'grasp_goal_reached':goals})
        new_ids=dones.nonzero(as_tuple=False)
        env.process_eval_episode_completions(new_ids,rewards,lengths)
        rows=current.update(step,goals,goals,dones,[False]*count,[{} for _ in range(count)])
        env.reset_eval_episode_tracking(new_ids);rewards[new_ids]=0;lengths[new_ids]=0
        assert [r['exported_success'] for r in current.outcomes]==env.eval_metrics['goal_reached_buffer']
        assert [r['steps'] for r in current.outcomes]==env.eval_metrics['length_buffer']
    result=current.result()
    assert target<=result['completed']<=target+count-1
    assert result['exported_successes']==sum(env.eval_metrics['goal_reached_buffer'])
    with pytest.raises(ValueError):current.update(step,[False]*count,[False]*count,[False]*count,[False]*count,[{}]*count)


def test_final_step_keeps_all_simultaneous_successes_and_failures_and_repeated_lanes():
    account=module.PooledEpisodeAccounting(4,5)
    account.update(0,[1,0,0,0],[1,0,0,0],[1,1,1,1],[0]*4,[{}]*4)
    account.update(1,[0,1,0,1],[0,1,0,1],[1]*4,[0]*4,[{}]*4)
    value=account.result()
    assert value['completed']==8 and value['overshoot']==3 and value['exported_successes']==3
    assert value['episodes_per_lane']==[2]*4
    assert [(r['lane'],r['episode_index']) for r in value['outcomes']]==[(i,j) for j in range(2) for i in range(4)]


def test_original_motion_mapping_is_lane_modulo_not_uniform_global_pool():
    path=REFERENCE/'envs/loco_manip/reset_from_dataset_wsppt.py';raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='d42185ef7535e193bb5affcda6e7ad2fbac92e1c478feb9a846ed455f476ac94'
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='ResetFromDatasetWSPPT')
    method=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_sample_idx'))
    scope={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    env=SimpleNamespace(device='cpu',reset_dataset=SimpleNamespace(sample_idx_based_on_demo_clip_number=[(i*100,(i+1)*100) for i in range(295)]))
    ids=torch.arange(16)
    for _ in range(4):
        samples=scope['_get_sample_idx'](env,ids)
        assert torch.equal(samples//100,ids)


@pytest.mark.parametrize('qualification,target,steps',[(True,32,900),(False,150,4500)])
def test_separate_protocol_dispatch_budget_and_fixed_panel_unchanged(material,qualification,target,steps):
    panel,config,_=material;before=deepcopy((panel,config));value=plan(material,qualification)
    assert dispatch.validate_plan(value)==value and (panel,config)==before
    for i,entry in enumerate(value['commands']):
        spec=entry['artifacts'][1]['value']
        assert entry['body']['operation']==module.OPERATION
        assert entry['payload_sha256']==module.digest(entry['body'])
        assert spec['minimum_completed_episodes']==target and spec['max_vector_steps']==steps
        assert spec['owner']['physical_gpu']==i and spec['motion_support']==panel['cohorts'][i]['assignment']['lanes']
        assert not spec['full295_motion_coverage'] and not spec['training_curriculum_restored']
        assert 'scaled_admission_file' not in entry['body']['payload']
    from pipeline.deployment.scaled_success_dispatch import validate_plan
    with pytest.raises(ValueError):validate_plan(value)


@pytest.mark.parametrize('key,value',[('minimum_completed_episodes',150),('max_vector_steps',450),('generated_only',False),
    ('teacher_probability',.5),('training_curriculum_restored',True),('natural_episode_clocks_randomized',True),
    ('observations_recomputed_after_clock_randomization',True),('fixed295_equivalent',True),('source_pins',{})])
def test_benchmark_mutations_fail_closed(material,key,value):
    spec=deepcopy(plan(material)['commands'][0]['artifacts'][1]['value']);spec[key]=value
    with pytest.raises(ValueError):module.validate_benchmark(spec,*material[:2])


def test_native_actual_request_admission_and_active_operation_binding(accepted,material,tmp_path,monkeypatch):
    backend,payload,paths,documents,checks=native_admission(accepted,tmp_path,monkeypatch,sharded=True)
    native=dict(physical_gpu=0,gpu_uuid='GPU-fixture',native_pid=os.getpid(),worker_instance_id='a'*32,source_sha256='c'*64)
    documents['resident-state.json'].update(native)
    spec=module.build_benchmark(documents['panel.json'],documents['config.json'],owner=native,
        command_id='native-c0',output_name='native-c0',cohort_index=0,qualification_only=True,arm=payload['arm'])
    paths['benchmark.json']=tmp_path/'benchmark.json';paths['benchmark.json'].write_bytes(module.canonical(spec))
    paths['resident-state.json'].write_bytes(module.canonical(documents['resident-state.json']))
    payload.pop('scaled_admission_file');payload.pop('scaled_admission_sha256')
    payload.update(benchmark_file='benchmark.json',benchmark_sha256=module.digest(spec))
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU','0')
    assert backend._scaled_success_inputs(payload,raven_pooled=True)[3]==spec
    command=dict(operation=module.OPERATION,command_id='native-c0',worker_instance_id='a'*32,payload=payload)
    backend.validate_command_identity(command)
    state=documents['resident-state.json'];state.update(phase='RUNNING',active_command_id='native-c0',
        commands={'native-c0':dict(status='RUNNING',command=command)})
    paths['resident-state.json'].write_bytes(module.canonical(state))
    backend._scaled_success_inputs(payload,raven_pooled=True)
    state['commands']['native-c0']['command']['operation']='evaluate_scaled_rendered_panel'
    paths['resident-state.json'].write_bytes(module.canonical(state))
    with pytest.raises(ValueError):backend._scaled_success_inputs(payload,raven_pooled=True)


def test_fixed_evaluation_clock_and_first_episode_branches_remain_explicit():
    source=(SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    assert 'account = FirstEpisodeAccounting(count)' in source
    assert 'range(450 if pooled is None else pooled[\'max_vector_steps\'])' in source
    tree=ast.parse(source);method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    statements=[ast.unparse(n) for n in method.body]
    reset=next(i for i,s in enumerate(statements) if 'obs = self.env.reset_panel()' in s)
    random=next(i for i,s in enumerate(statements) if 'initial_clocks = randomize_initial_clocks(self.env)' in s)
    metrics=next(i for i,s in enumerate(statements) if 'self.env.init_eval_metrics_tracking' in s)
    assert reset<random<metrics


@pytest.mark.parametrize('qualification',[True,False])
def test_complete_result_validates_native_render_and_separate_weighted_denominator(accepted,material,qualification):
    value=plan(material,qualification);results=[];guards=[]
    tree=ast.parse((HERE.parents[1]/'dagger/rtx/code/src/dagger_rtx/resident_service.py').read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ResidentOwner')
    method=deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='result'))
    actual_api={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<actual-resident-result-api>','exec'),actual_api)
    for i,entry in enumerate(value['commands']):
        result=completed_result(accepted,material,i,qualification)
        row=dict(status='COMPLETE',payload_sha256=entry['payload_sha256'],receipt=dict(complete=True,
            source_sha256='c'*64,native_pid=os.getpid(),native_initializations=1,optimizer_steps=0,
            command_id=entry['body']['command_id'],worker_instance_id=entry['body']['worker_instance_id'],
            kit_identity=123,same_kit=True,physics_pause_requested=True,scene_initializations=2,phase_resets=1,
            purpose='test',profile='explicit_evaluation',result=result))
        cid=entry['body']['command_id']
        row=actual_api['result'](SimpleNamespace(lock=threading.Lock(),state=dict(commands={cid:
            dict(row,command=entry['body'],acceptance={})})),cid)
        assert 'command' not in row and 'acceptance' not in row
        results.append(dispatch.validate_complete(value,entry,row,lambda *args:guards.append(args)))
        for key,invalid in [('command_id','another-command'),('worker_instance_id','e'*32),
                ('kit_identity',None),('same_kit',False),('physics_pause_requested',False),
                ('scene_initializations',0),('phase_resets',0)]:
            wrong=deepcopy(row);wrong['receipt'][key]=invalid
            with pytest.raises(ValueError):dispatch.validate_complete(value,entry,wrong,lambda *args:None)
        wrong=deepcopy(row);wrong['payload_sha256']='f'*64
        with pytest.raises(ValueError):dispatch.validate_complete(value,entry,wrong,lambda *args:None)
    summary=dispatch.aggregate(value,results)
    assert len(guards)==2 and summary['completed']==(64 if qualification else 320)
    assert summary['weighted_success_rate']==.5 and summary['benchmark_complete'] is not qualification
    assert [r['overshoot'] for r in summary['actor_results']]==([0,0] if qualification else [10,10])


@pytest.mark.parametrize('fault',['truncate','duplicate','clock','stage','anchor','reset','fallback','teacher','training','wrong_schema','old_goal','missing_resetfix','arm','duration','timeout'])
def test_result_rejects_unsupported_or_incomplete_evidence(accepted,material,fault):
    value=completed_result(accepted,material);spec=value['benchmark']
    if fault=='truncate':value['outcomes'].pop()
    elif fault=='duplicate':value['outcomes'][-1]=deepcopy(value['outcomes'][-2])
    elif fault=='clock':value['initial_clocks']['after']['episode_length_buf'][0]=400
    elif fault=='stage':value['initial_clocks']['after']['actual_time_in_stage_buf'][0]=-1
    elif fault=='anchor':value['rendered_observation_receipt']['anchor_selections'].pop()
    elif fault=='reset':value['native_episode_resets'].pop()
    elif fault=='fallback':value['rendered_observation_receipt']['raytraced_fallbacks']=1
    elif fault=='teacher':value['teacher_action_mode']='raven_ppo'
    elif fault=='training':value['training_curriculum_restored']=True
    elif fault=='wrong_schema':value['schema']='roborender.scaled-rendered-success-partition.v1'
    elif fault=='old_goal':value['goal_accounting_version']='old'
    elif fault=='arm':value['arm']='bc'
    elif fault=='duration':value['outcomes'][-1]['steps']+=1
    elif fault=='timeout':value['outcomes'][-1]['timeout']=1
    else:value.pop('initial_robot_refresh_attestation')
    with pytest.raises(ValueError):module.validate_result(value,spec)


def test_actual_pooled_render_adapter_refreshes_every_done_lane_without_suppressing_later_episodes(tmp_path,monkeypatch):
    import rendered_success_observations as observations
    from test.deployment.rtx_actor.test_rendered_success_observations import fixture
    monkeypatch.setattr(observations,'GeneratedSuccessObservations',observations.GeneratedPooledSuccessObservations)
    original=observations.GeneratedPooledSuccessObservations.__init__
    monkeypatch.setattr(observations.GeneratedPooledSuccessObservations,'__init__',
        lambda self,*a,**k:original(self,*a,benchmark_sha256='b'*64,**k))
    trainer,adapter,obs,clients,*_=fixture(tmp_path,count=2)
    try:
        for step,done in enumerate([[0,0],[1,0],[1,1],[0,1]]):
            output=adapter(obs,torch.tensor(done),step)
            assert output['vision_obs'] is not obs['vision_obs']
        receipt=adapter.receipt()
        assert receipt['render_requests_completed']==8 and all(len(c.calls)==4 for c in clients)
        assert [(r['lane'],r['episode_index']) for r in receipt['anchor_selections']]==[(0,0),(1,0),(0,1),(0,2),(1,1),(1,2)]
        assert receipt['pooled_episode_sessions']['current_episode_indices']==[2,2]
    finally:adapter.close()


@pytest.mark.parametrize('pooled,trip_at',[(False,None),(True,None),(True,0),(True,1),(True,2),(True,3),(True,4)])
def test_actual_native_loop_executes_repeated_episode_metric_callbacks_and_stops_exactly(monkeypatch,pooled,trip_at):
    import rendered_success_observations as observations
    from pipeline.deployment.rtx_actor.success_diagnostic_core import FirstEpisodeAccounting
    tree=ast.parse((SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text())
    trainer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SuccessDiagnosticTrainer')
    method=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    loop=deepcopy(next(n for n in ast.walk(method) if isinstance(n,ast.For) and ast.unparse(n.target)=='step'))
    env=original_metrics(16);env.config={};captured={};events=[];step_counter=[0]
    def render(trainer,obs,dones,step):
        events.append(('render',step));return {**obs,'vision_obs':obs['vision_obs'].clone()}
    monkeypatch.setattr(observations,'apply_observation_transform',render)
    def policy(trainer,obs,dones,**kwargs):
        events.append(('policy',step_counter[0]));return {'actions':torch.ones(16,26)},None
    def step(actions):
        index=step_counter[0];events.append(('step',index));step_counter[0]+=1
        assert actions['actions'].eq(1).all()
        goals=torch.tensor([(index+i)%3==0 for i in range(16)])
        done=torch.full((16,),index%2==1,dtype=torch.bool)
        captured.update(current_last_stage_completed_task_buf=goals.numpy(),reset_buf=done.numpy(),time_out_buf=np.zeros(16,dtype=bool))
        env._transition_goal_before_reset=goals
        return {'vision_obs':torch.zeros(16,3)},torch.ones(16),done,{'grasp_goal_reached':goals}
    env.step=step
    account=module.PooledEpisodeAccounting(16,32) if pooled else FirstEpisodeAccounting(16)
    guard_calls=[]
    def resource_guard():
        guard_calls.append(len(guard_calls)+1)
        if guard_calls[-1]==trip_at:
            raise RuntimeError('injected actual native resource fault')
    scope=dict(self=SimpleNamespace(env=env,accelerator=SimpleNamespace(device='cpu'),
            _native_resource_guard=None if trip_at==0 else resource_guard),
        pooled={'max_vector_steps':900} if pooled else None,scaled_teacher=None,teacher_guard_checks=0,
        torch=torch,np=np,time=time,json=json,
        cycle_timings=None,numeric_steps=0,numeric_active=False,arm='dagger',teacher_mode='distill',count=16,
        obs={'vision_obs':torch.zeros(16,3)},dones=torch.zeros(16),select_policy_actions=policy,
        episode_reward=torch.zeros(16),episode_length=torch.zeros(16,dtype=torch.int64),captured=captured,
        account=account,trace=io.StringIO(),emit=lambda *a:None,started=time.monotonic(),
        cpu=lambda t:t.detach().cpu().numpy() if isinstance(t,torch.Tensor) else np.asarray(t),
        lane_details=lambda *a:{})
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<actual-native-loop>','exec')
    if trip_at is not None:
        with pytest.raises(RuntimeError,match='native resource'):
            exec(code,scope)
        assert events=={0:[],1:[],2:[('render',0)],3:[('render',0),('policy',0)],
            4:[('render',0),('policy',0),('step',0)]}[trip_at]
        assert step_counter[0]==(1 if trip_at==4 else 0)
        assert account.result()['completed']==0
        return
    exec(code,scope)
    total_steps=4 if pooled else 2
    assert events==[(event,step) for step in range(total_steps) for event in ('render','policy','step')]
    assert account.result()['completed']==(32 if pooled else 16)
    assert len(env.eval_metrics['goal_reached_buffer'])==account.result()['completed']
    assert sum(env.eval_metrics['goal_reached_buffer'])==account.result()['exported_successes']
    assert len(guard_calls)==(3*total_steps if pooled else 0)


def test_all_original_source_proof_files_are_hash_pinned():
    for name,digest in module.SOURCE_PINS.items():
        assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()==digest


def test_actual_native_pooled_terminal_binds_command_owner_and_pause_before_same_kit_attestation():
    tree=ast.parse((HERE/'resident_isaac_trainer.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_run_explicit_test')
    block=deepcopy(next(n for n in method.body if isinstance(n,ast.If) and ast.unparse(n.test)=='raven_pooled'
        and 'same_kit=True' in ast.unparse(n)))
    events=[]
    native=SimpleNamespace(kit_identity=123,_kit_unchanged=lambda:events.append('same-kit'),
        trainer=SimpleNamespace(env=SimpleNamespace(simulator=SimpleNamespace(sim=SimpleNamespace(pause=lambda:events.append('pause'))))))
    scope=dict(self=native,raven_pooled=True,receipt={},scaled_inputs=[None,None,None,
        dict(command_id='actual-pooled',owner=dict(worker_instance_id='a'*32))])
    code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),'<actual-pooled-terminal>','exec')
    exec(code,scope)
    assert events==['pause','same-kit']
    assert scope['receipt']==dict(command_id='actual-pooled',worker_instance_id='a'*32,kit_identity=123,
        same_kit=True,physics_pause_requested=True)
    def failed_kit():raise RuntimeError('actual Kit identity changed')
    native._kit_unchanged=failed_kit;scope['receipt']={}
    with pytest.raises(RuntimeError,match='Kit identity changed'):exec(code,scope)
    assert scope['receipt']=={}


def test_native_reset_observer_counts_episodes_not_periodic_or_nested_task_resets(monkeypatch):
    tree=ast.parse((HERE/'isaac6_motion_evaluator.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExplicitMotionSuccessEvaluator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_cohort')
    funcs=[deepcopy(n) for n in method.body if isinstance(n,ast.FunctionDef)
        and n.name in {'observed_reset','observed_episode_reset'}]
    assignment={'lanes':[{'motion_id':f'motion-{i}'} for i in range(3)]}
    env=SimpleNamespace(device='cpu',reset_dataset=SimpleNamespace(sample_idx_based_on_demo_clip_number=[(i*10,(i+1)*10) for i in range(3)]))
    owner=SimpleNamespace(env=env,_raven_pooled_benchmark={'test':True},_first_native_resets={},
        _pooled_native_resets=[],_pooled_reset_counts=[0]*3,_pooled_latest_task_reset=[None]*3)
    def task(lanes):
        env.sample_idx=lanes*10+2;env.sample_idx_in_seq=torch.zeros(len(lanes),dtype=torch.long)
    scope=dict(self=owner,original=task,assignment=assignment,torch=torch,hashlib=hashlib)
    def episode(lanes):
        scope['observed_reset'](lanes);scope['observed_reset'](lanes)
    scope['original_episode_reset']=episode
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda *_:torch.get_rng_state())
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),'<actual-native-reset-observers>','exec'),scope)
    scope['observed_episode_reset'](torch.arange(3))
    scope['observed_reset'](torch.tensor([2]))  # periodic task reset, not an episode completion
    assert len(owner._pooled_native_resets)==3
    scope['observed_episode_reset'](torch.tensor([2,0]))
    assert [(r['lane'],r['episode_index']) for r in owner._pooled_native_resets]==[(0,0),(1,0),(2,0),(2,1),(0,1)]
    assert owner._pooled_reset_counts==[2,1,2]


def test_actual_native_router_uses_separate_pooled_receipt_not_fixed_panel_summary(accepted,material,tmp_path,monkeypatch):
    import rendered_success_observations as observations
    import explicit_motion_panel as explicit
    from test.deployment.rtx_actor.test_rendered_student_native_routing import evaluator_function
    from initial_robot_refresh_diagnostic import SCALED_STUDENT_CLEAR_ONCE
    panel,config,_=material;value=plan(material);spec=value['commands'][0]['artifacts'][1]['value']
    result=completed_result(accepted,material);prepared=[];adapters=[]
    class Adapter:
        def __init__(self,owner,**kwargs):
            assert kwargs['benchmark_sha256']==module.digest(spec)
            assert kwargs['evaluation_binding']['checkpoint_sha256']==config['checkpoint_sha256']
            assert kwargs['endpoint_pool_physical_gpu']==0
            self.closed=False;adapters.append(self)
        def close(self):self.closed=True
    class Evaluator:
        def evaluate_cohort(self,assignment,**kwargs):
            assert self._raven_pooled_benchmark==spec
            assert not hasattr(self,'_scaled_initialization_distribution')
            assert assignment==panel['cohorts'][0]['assignment']
            assert self._success_observation_transform is adapters[-1]
            return deepcopy(result)
    monkeypatch.setenv('RESIDENT_PHYSICAL_GPU','0')
    monkeypatch.setattr(observations,'GeneratedPooledSuccessObservations',Adapter)
    monkeypatch.setattr(explicit,'run_partition',lambda *a,**k:pytest.fail('pooled episodes cannot become fixed-panel outcomes'))
    execute=evaluator_function(tmp_path,Evaluator)
    execute.__globals__.update(deepcopy=deepcopy,hashlib=hashlib,canonical=module.canonical,os=os)
    actual=execute(SimpleNamespace(env=object()),panel,root=tmp_path/'native-output',
        checkpoint_path='/CPU/actual-checkpoint.pt',checkpoint_sha256=config['checkpoint_sha256'],
        prepare_cohort=lambda assignment:prepared.append(assignment),report=lambda *_:None,arm='dagger',
        teacher_action_mode='distill',diagnostic_initial_robot_refresh=SCALED_STUDENT_CLEAR_ONCE,
        rendered_observation_config=config,_scaled_partition=explicit.make_partition(panel,[0]),_raven_benchmark=spec)
    assert actual==result and len(prepared)==1 and adapters[0].closed
    assert json.loads((tmp_path/'native-output/result.json').read_bytes())['schema']==module.RESULT_SCHEMA
    assert not list((tmp_path/'native-output/cohort-receipts').iterdir())
