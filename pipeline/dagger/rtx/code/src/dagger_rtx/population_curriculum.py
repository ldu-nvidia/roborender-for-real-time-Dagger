"""Pair native actors at reset boundaries for one Raven curriculum population.

Only statistics/curriculum are shared. Physical resets remain lane-local and
use the existing corrected reset functions. Incomplete barriers are never
replayed. This is a fresh-lifecycle training protocol, not native recovery.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy
import json
import os
from pathlib import Path
import time

from .durable import atomic_write_json
from .training_curriculum import current_values, digest, _get
from .contracts import require_sha256

PROTOCOL = 'roborender.raven-paired-population-curriculum.v1'
CONFIG_FIELD = 'population_curriculum_protocol'
POPULATION = 'shared_paired_native_population_v1'
BUFFERS = ('last_max_stage_buf', 'current_max_stage_buf',
           'last_completed_task_buf', 'current_completed_task_buf',
           'last_last_stage_completed_task_buf', 'current_last_stage_completed_task_buf')
LOGS = ('average_stage_reached', 'average_goal_reached', 'average_last_stage_goal_reached')


def validate_progress(value, updates):
    if (not isinstance(value,dict) or value.get('schema')!=PROTOCOL
            or value.get('population')!=POPULATION
            or value.get('vector_reset_barriers')!=1+81*updates
            or type(value.get('global_reset_events')) is not int
            or not 1<=value['global_reset_events']<=1+81*updates
            or value.get('physical_resets_added')!=0):
        raise ValueError('shared curriculum lacks actual contiguous native reset barriers')
    require_sha256(value.get('final_state_sha256'),'shared curriculum final state')
    return value


class PairedExchange:
    def __init__(self, root, scope, gpu, *, timeout_s=1200, guard=lambda: None):
        if (scope.get('schema') != PROTOCOL or gpu not in (0, 1)
                or len(scope.get('owners', [])) != 2
                or scope['owners'][gpu]['native_pid'] != os.getpid()):
            raise ValueError('global curriculum requires the exact two admitted native owners')
        owners=scope['owners']
        if ([r.get('physical_gpu') for r in owners]!=[0,1]
                or any(type(r.get('native_pid')) is not int or r['native_pid']<=0 for r in owners)
                # Native PIDs can both be1 in separate container namespaces.
                or any(len({r.get(k) for r in owners})!=2 for k in ('worker_instance_id','gpu_uuid','command_id'))
                or any(r.get('source_sha256')!=scope.get('source_sha256') for r in owners)):
            raise ValueError('distinct current-source native lifecycles required')
        for field in ('source_sha256','config_sha256','segment_sha256'):
            require_sha256(scope.get(field),'population '+field)
        self.scope, self.gpu = scope, gpu
        self.root = Path(root) / digest(scope)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.guard = timeout_s, guard
        if list(self.root.glob(f'*-g{gpu}.json')):
            raise RuntimeError('population barrier already consumed; no native replay')

    def __call__(self, index, phase, value):
        if phase not in ('before', 'after'):
            raise ValueError('unknown native population barrier phase')
        path = self.root / f'{index:07d}-{phase}-g{self.gpu}.json'
        row = dict(scope_sha256=digest(self.scope), owner=self.scope['owners'][self.gpu], value=value)
        if path.exists():
            raise RuntimeError('native population step already published; no replay')
        atomic_write_json(path, row)
        peer = self.root / f'{index:07d}-{phase}-g{1-self.gpu}.json'
        start = time.monotonic()
        while not peer.exists():
            self.guard()
            if time.monotonic()-start > self.timeout_s:
                raise RuntimeError('native population peer missing; state uncertain, no replay')
            time.sleep(.05)
        other = json.loads(peer.read_bytes())
        if (other.get('scope_sha256') != digest(self.scope)
                or other.get('owner') != self.scope['owners'][1-self.gpu]):
            raise RuntimeError('native population peer identity changed')
        return [row['value'], other['value']] if self.gpu == 0 else [other['value'], row['value']]


def snapshot(env, ids):
    return dict(count=int(env.num_envs), clock=int(env.common_step_counter),
        ids=ids.cpu().tolist(), lengths=env.last_episode_length_buf.cpu().tolist(),
        buffers={name:getattr(env,name).cpu().tolist() for name in BUFFERS},
        curriculum=current_values(env),
        prior_logs={name:float(env.log_dict[name]) for name in LOGS},
        num_compute_average_epl=float(env.num_compute_average_epl))


def merge(rows, device):
    import torch

    if len(rows) != 2:
        raise ValueError('exactly two native population contributions required')
    first, second = rows
    for name in ('clock', 'curriculum', 'prior_logs', 'num_compute_average_epl'):
        if first[name] != second[name]:
            raise RuntimeError('native population before-reset state differs: '+name)
    if first['count'] != second['count'] or first['count'] < 1:
        raise ValueError('equal native lane populations required')
    all_ids, lengths, offset = [], [], 0
    after = {name:[] for name in LOGS}
    for row in rows:
        count, ids = row['count'], row['ids']
        if (len(set(ids)) != len(ids) or any(type(i) is not int or not 0<=i<count for i in ids)
                or len(row['lengths']) != count or any(len(row['buffers'][k])!=count for k in BUFFERS)):
            raise ValueError('invalid native population lane snapshot')
        all_ids.extend(offset+i for i in ids); lengths.extend(row['lengths']); offset+=count
        for log, last, current in zip(LOGS,BUFFERS[::2],BUFFERS[1::2]):
            values=list(row['buffers'][last])
            for i in ids: values[i]=row['buffers'][current][i]
            after[log].extend(values)
    return dict(ids=torch.tensor(all_ids,device=device,dtype=torch.long),
        lengths=torch.tensor(lengths,device=device,dtype=torch.long),
        logs={name:torch.tensor(values,device=device,dtype=torch.float32).mean()
              for name,values in after.items()}, count=offset)


@contextmanager
def shared_population(env, exchange):
    """Run original curriculum methods once per GLOBAL reset, even on idle lanes.

Both actors enter reset_envs_idx every vector step, including empty local ids.
No reset is added and no episode is truncated at a chunk/optimizer boundary.
"""
    import torch

    if bool(env.is_evaluating) or _get(env.config,'reset_from_dataset.reset_from_teleop_curriculum',False):
        raise ValueError('shared population only supports the admitted training reset profile')
    if _get(env.config,'enable_staged_reset',False):
        raise ValueError('shared population does not support staged partial resets')
    names=('reset_envs_idx','_update_average_episode_length','_update_termination_curriculum')
    originals={name:getattr(env,name) for name in names}
    owned={name:name in vars(env) for name in names}
    active=None
    sequence=0
    receipt=dict(schema=PROTOCOL, population=POPULATION, vector_reset_barriers=0,
                 global_reset_events=0, physical_resets_added=0)

    def average(_ids):
        # Invoke the ACTUAL original method (some LocoManip profiles override
        # the base EMA with a no-op). Never substitute a guessed formula.
        shadow=copy(env)
        shadow.num_envs=active['count']
        shadow.last_episode_length_buf=active['lengths']
        if isinstance(env.average_episode_length,torch.Tensor):
            shadow.average_episode_length=env.average_episode_length.clone()
        originals['_update_average_episode_length'].__func__(shadow,active['ids'])
        env.average_episode_length=shadow.average_episode_length

    def termination():
        env.log_dict.update(active['logs'])
        return originals['_update_termination_curriculum']()

    def reset(ids,*args,**kwargs):
        nonlocal active,sequence
        if active is not None:
            raise RuntimeError('nested population reset is not admitted')
        rows=exchange(sequence,'before',snapshot(env,ids))
        active=merge(rows,env.device)
        try:
            result=originals['reset_envs_idx'](ids,*args,**kwargs)
            if active['ids'].numel() and ids.numel()==0:
                # The other RTX reset lanes. A single B32 Raven env updates
                # these shared scalars once; do so here WITHOUT resetting any
                # of this actor's still-running lanes or randomizing its scene.
                average(ids)
                if env.use_reward_penalty_curriculum: env._update_reward_penalty_curriculum()
                if any(getattr(env,k) for k in ('use_reward_limits_dof_pos_curriculum',
                        'use_reward_limits_dof_vel_curriculum','use_reward_limits_torque_curriculum')):
                    env._update_reward_limits_curriculum()
                if env.add_noise_currculum: env._update_obs_noise_curriculum()
                termination()
            post=dict(curriculum=current_values(env),
                      logs={name:float(env.log_dict[name]) for name in LOGS})
            observed=exchange(sequence,'after',post)
            if observed[0]!=observed[1]:
                raise RuntimeError('native population curriculum diverged after reset; no continuation')
            receipt['vector_reset_barriers']+=1
            receipt['global_reset_events']+=int(bool(active['ids'].numel()))
            receipt['final_state_sha256']=digest(post)
            sequence+=1
            return result
        finally:
            active=None

    env.reset_envs_idx=reset
    env._update_average_episode_length=average
    env._update_termination_curriculum=termination
    try:
        yield receipt
    finally:
        for name in names:
            if owned[name]:setattr(env,name,originals[name])
            else:delattr(env,name)
