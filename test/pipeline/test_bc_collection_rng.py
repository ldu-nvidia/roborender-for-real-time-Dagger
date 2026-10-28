"""Pinned original discarded draw + real AdamW versus collection-only draws.

CPU differential evidence only; no renderer/Isaac numerical parity claim.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.distributions import Normal

from pipeline.deployment.rtx_actor import bc_collection_rng as rng
from pipeline.deployment.rtx_actor import isaac6_bc_production as native
from pipeline.bc_collection import continuous_plan

REF=Path(__file__).resolve().parents[2]/'runtime/raven_gr00t_collection_94d7985/gr00t/rl'

def original_action_mean():
    path=REF/'trl/modules/vision_actor_critic_modules_obj_pred.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rng.POLICY_SHA256
    cls=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='action_mean_with_obj_pred')
    ns=dict(Normal=Normal,torch=torch)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    return ns['action_mean_with_obj_pred']

class TinyPolicy(torch.nn.Module):
    # Deterministic trainable forward substitutes for the ResNet, not its RNG.
    def __init__(self,batch):
        super().__init__();self.weight=torch.nn.Parameter(torch.arange(11,dtype=torch.float32)/10)
        self.std=torch.nn.Parameter(torch.ones(11),requires_grad=False)
        self.batch=batch;self.clamp_noise_std=False
    def forward(self,obs_dict,**kwargs):
        return dict(actions=self.weight.view(1,1,11).expand(self.batch,81,11).contiguous(),obj_pred=torch.zeros(self.batch,81,3))
    @property
    def action_mean(self):return self.distribution.mean
    @property
    def action_std(self):return self.distribution.stddev

@pytest.mark.parametrize('batch',[1,2,16,32,40,50])
def test_original_one_epoch_draws_and_real_optimizer_match_rng_continuation(batch):
    saved=torch.get_rng_state();policy=TinyPolicy(batch);optimizer=torch.optim.AdamW(policy.parameters(),lr=.0002)
    method=original_action_mean();states=[];next_reset=[]
    try:
        torch.manual_seed(31)
        for window in range(5):
            torch.randperm(batch,device='cpu')  # Actual pinned PPO train-loop operation.
            result=method(policy,{})
            loss=result['action_mean'].square().mean();loss.backward();optimizer.step();optimizer.zero_grad()
            states.append(torch.get_rng_state().clone());next_reset.append(torch.randint(10000,(batch,)))
        torch.manual_seed(31)
        for window in range(5):
            rng.consume_window(batch,device='cpu')
            assert torch.equal(torch.get_rng_state(),states[window])
            assert torch.equal(torch.randint(10000,(batch,)),next_reset[window])
        assert len(optimizer.state)>0  # Reference really optimized, unlike collection.
    finally:torch.set_rng_state(saved)

def motions():return [dict(motion_id='a',path='/motions/a.npz',sha256='a'*64)]

def test_new_plan_protocol_explicit_legacy_remains_unchanged():
    new=continuous_plan.build_plan(motions(),run_id='rng',max_windows=4,output_dir='/runs/new',num_envs=1)
    assert new['collection_rng_protocol']==rng.PROTOCOL
    request=continuous_plan.build_request(new)
    assert native.validate_collection_request(request)==request
    old=continuous_plan.build_plan(motions(),run_id='rng',max_windows=4,output_dir='/runs/old',num_envs=1,collection_rng_protocol=None,collection_loop_protocol=None)
    assert 'collection_rng_protocol' not in old
    assert 'collection_rng_protocol' not in continuous_plan.build_request(old)
    for invalid in ('unknown',None,1):
        bad=deepcopy(request);bad['collection_rng_protocol']=invalid
        with pytest.raises(ValueError,match='RNG protocol'):native.validate_collection_request(bad)

@pytest.mark.parametrize('batch',[0,51,-1,True,1.5])
def test_bad_population_rejected_without_random_draws(batch):
    state=torch.get_rng_state()
    with pytest.raises(ValueError):rng.consume_window(batch,device='cpu')
    assert torch.equal(state,torch.get_rng_state())

def test_architecture_guard_rejects_unknown_model():
    with pytest.raises(ValueError,match='pinned nonrecurrent'):
        rng.validate_native(SimpleNamespace(policy_model=TinyPolicy(2),args=None))

def test_raven_network_and_train_loop_randomness_assumptions_are_pinned():
    for relative,pin in [('agents/modules/modules.py',rng.MODULE_SHA256),('trl/trainer/ppo_trainer.py',rng.PPO_SHA256)]:
        assert hashlib.sha256((REF/relative).read_bytes()).hexdigest()==pin
    raw=(REF/'agents/modules/modules.py').read_text()
    assert 'Dropout(' not in raw and 'torch.rand(' not in raw and 'torch.randn(' not in raw
    train=ast.parse((REF/'trl/trainer/ppo_trainer.py').read_bytes())
    loop=next(n for n in ast.walk(train) if isinstance(n,ast.FunctionDef) and n.name=='train')
    calls=[ast.unparse(n.func) for n in ast.walk(loop) if isinstance(n,ast.Call)]
    assert calls.count('torch.randperm')==1
