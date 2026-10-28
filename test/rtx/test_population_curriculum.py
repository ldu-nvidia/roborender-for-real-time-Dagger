"""Differential reset/curriculum tests using pinned Raven callback bodies.

Tiny physical reset double and thread transport; not a live Isaac certificate.
"""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest
import torch

from dagger_rtx.population_curriculum import shared_population, snapshot, merge, LOGS, BUFFERS, PairedExchange, PROTOCOL
from dagger_rtx.training_curriculum import current_values

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs'


def methods(path,pin,names):
    raw=(ROOT/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pin
    result=[deepcopy(node) for cls in ast.parse(raw).body if isinstance(cls,ast.ClassDef)
            for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in result}==set(names)
    for node in result:node.decorator_list=[]
    return result


def environment(count, all_curricula=False):
    base=methods('legged_base_task/legged_robot_base.py','78d0a736125d09c9ebe650e871e87338bdcb463941dca949d4a347edb70dde5c',
                 ['_reset_tasks_callback','_update_average_episode_length',
                  '_update_reward_penalty_curriculum','_update_reward_limits_curriculum',
                  '_update_obs_noise_curriculum'])
    staged=methods('base_task/staged_task_base.py','c67503aad0a9ebf9c66c6dcc629f5ea0872b58451fe3d3a098b83faf1d949e26',
                   ['_reset_tasks_callback','_update_termination_curriculum','_update_reward_penalty_curriculum'])
    tree=ast.Module(body=[
        ast.ClassDef(name='Base',bases=[],keywords=[],decorator_list=[],body=base),
        ast.ClassDef(name='Staged',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],decorator_list=[],body=staged)],type_ignores=[])
    namespace=dict(torch=torch)
    exec(compile(ast.fix_missing_locations(tree),'<pinned Raven curriculum>','exec'),namespace)
    class Config(dict):
        __getattr__=dict.__getitem__
    class Env(namespace['Staged']):
        def reset_envs_idx(self,ids):
            self.reset_history.append(ids.tolist())
            if not len(ids):return
            self._update_average_episode_length(ids)
            self._reset_tasks_callback(ids)
            # Stand-in for physical reset, AFTER the original callbacks.
            self.physical_levels.append(float(self.termination_level))
        def _episodic_domain_randomization(self,ids):pass
    env=Env()
    env.config=Config(termination_curriculum=SimpleNamespace(
        level_down_ave_goal_reached_rate=.6,level_up_ave_goal_reached_rate=.96,
        degree=-.00008,min_level=.1,max_level=1.))
    env.device='cpu'; env.num_envs=count;env.common_step_counter=0;env.max_episode_length=500
    env.is_evaluating=False;env.average_episode_length=0.;env.num_compute_average_epl=torch.tensor(1000.)
    env.termination_level=torch.tensor(.5)
    for flag in ('use_reward_penalty_curriculum','use_reward_limits_dof_pos_curriculum',
                 'use_reward_limits_dof_vel_curriculum','use_reward_limits_torque_curriculum','add_noise_currculum'):
        setattr(env,flag,all_curricula)
    if all_curricula:
        limits=Config()
        for prefix in ('soft_dof_pos','soft_dof_vel','soft_torque'):
            limits.update({prefix+'_curriculum_level_down_threshold':10.,prefix+'_curriculum_level_up_threshold':20.,
                prefix+'_curriculum_degree':.01,prefix+'_min_limit':.1,prefix+'_max_limit':1.})
            setattr(env,prefix+'_curriculum_value',torch.tensor(.5))
        env.config['rewards']=Config(reward_limit=Config(reward_limits_curriculum=limits),
            reward_penalty_level_down_threshold=10.,reward_penalty_level_up_threshold=20.,
            reward_penalty_degree=.01,reward_min_penalty_scale=.1,reward_max_penalty_scale=1.)
        env.config['obs']=Config(soft_dof_pos_curriculum_level_down_threshold=10.,
            soft_dof_pos_curriculum_degree=.01,noise_value_min=.1,noise_value_max=1.)
        env.reward_penalty_scale=torch.tensor(.5);env.current_noise_curriculum_value=torch.tensor(.5)
    env.last_episode_length_buf=torch.arange(count)+20
    for key in BUFFERS:setattr(env,key,torch.zeros(count,dtype=torch.long if 'stage_buf' in key else torch.bool))
    env.log_dict={key:torch.tensor(0.) for key in LOGS}
    env.reset_history=[];env.physical_levels=[]
    return env


class Exchange:
    def __init__(self):self.barrier=threading.Barrier(2,timeout=5);self.values={}
    def at(self,gpu):
        def exchange(index,phase,value):
            self.values[index,phase,gpu]=deepcopy(value)
            self.barrier.wait()
            rows=[self.values[index,phase,g] for g in (0,1)]
            self.barrier.wait()
            return rows
        return exchange


@pytest.mark.parametrize('all_curricula',[False,True])
def test_two_b16_curricula_match_one_b32_across_asymmetric_and_empty_resets(all_curricula):
    reference=environment(32,all_curricula); actors=[environment(16,all_curricula),environment(16,all_curricula)]
    for gpu,env in enumerate(actors):env.last_episode_length_buf=reference.last_episode_length_buf[gpu*16:(gpu+1)*16].clone()
    #31/32 successes crosses .96 globally;15/16 does NOT locally.
    reference.current_completed_task_buf[:31]=True
    reference.current_last_stage_completed_task_buf[:30]=True
    reference.current_max_stage_buf[:]=4
    for gpu,env in enumerate(actors):
        for key in BUFFERS:setattr(env,key,getattr(reference,key)[gpu*16:(gpu+1)*16].clone())
    exchange=Exchange()
    with shared_population(actors[0],exchange.at(0)) as first, shared_population(actors[1],exchange.at(1)) as second:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for clock,ids in enumerate((list(range(32)),[0,2],[],[16,21],[1,18])):
                reference.common_step_counter=clock
                reference.reset_envs_idx(torch.tensor(ids,dtype=torch.long))
                futures=[]
                for gpu,env in enumerate(actors):
                    env.common_step_counter=clock
                    local=torch.tensor([i-gpu*16 for i in ids if gpu*16<=i<(gpu+1)*16],dtype=torch.long)
                    futures.append(pool.submit(env.reset_envs_idx,local))
                for future in futures:future.result(timeout=10)
                for env in actors:
                    assert current_values(env)==current_values(reference)
                    for name in LOGS:assert torch.equal(env.log_dict[name],reference.log_dict[name])
                    if env.reset_history[-1]:assert env.physical_levels[-1]==reference.physical_levels[-1]
    assert first==second and first['vector_reset_barriers']==5
    assert first['global_reset_events']==4 and first['physical_resets_added']==0
    assert actors[1].reset_history[1]==[] and actors[0].reset_history[3]==[]
    assert all('reset_envs_idx' not in vars(env) for env in actors)


def test_negative_control_independent_curriculum_differs_at_global_threshold():
    global_env=environment(32);local_env=environment(16)
    global_env.current_completed_task_buf[:31]=True;local_env.current_completed_task_buf[:15]=True
    global_env.reset_envs_idx(torch.arange(32));local_env.reset_envs_idx(torch.arange(16))
    assert float(global_env.termination_level)<float(local_env.termination_level)


@pytest.mark.parametrize('fault',['clock','curriculum','ids','count'])
def test_inconsistent_population_fails_closed(fault):
    env=environment(16);rows=[snapshot(env,torch.arange(16)),snapshot(env,torch.arange(16))]
    if fault=='clock':rows[1]['clock']+=1
    elif fault=='curriculum':rows[1]['curriculum']['termination_level']['value']=.6
    elif fault=='ids':rows[1]['ids']=[0,0]
    else:rows[1]['count']=8
    with pytest.raises((ValueError,RuntimeError)):merge(rows,'cpu')


def test_durable_pair_exchange_same_pid_namespaces_and_no_replay(tmp_path):
    scope=dict(schema=PROTOCOL,source_sha256='a'*64,config_sha256='b'*64,segment_sha256='c'*64,
        owners=[dict(physical_gpu=g,native_pid=os.getpid(),worker_instance_id=str(g)*32,
            gpu_uuid=f'GPU-fixture{g}',command_id=f'cpu-fixture{g}',source_sha256='a'*64) for g in (0,1)])
    peers=[PairedExchange(tmp_path,scope,g,timeout_s=1) for g in (0,1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(peers[g],0,'before',dict(gpu=g)) for g in (0,1)]
        assert [f.result(timeout=3) for f in futures]==[[dict(gpu=0),dict(gpu=1)]]*2
    with pytest.raises(RuntimeError,match='no replay'):peers[0](0,'before',dict(gpu=0))
    with pytest.raises(RuntimeError,match='no native replay'):PairedExchange(tmp_path,scope,0)
