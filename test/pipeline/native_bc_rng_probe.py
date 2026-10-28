"""Bounded CPU/CUDA RNG differential; no Isaac or renderer entry point.

Run only outside native measurement intervals. A separate process executes the
pinned original Normal method with a declared tiny deterministic policy/AdamW;
it does not read or alter the resident simulator's RNG or policy state.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def load_helper(path):
    spec=importlib.util.spec_from_file_location('bc_collection_draws',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def architecture_guard(helper,path):
    """Real pinned policy class; collector arguments are explicitly declared doubles."""
    from types import SimpleNamespace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    expected='3e903c55218ca2aae4c970fab6aeb820de0a35d8965a34262b8206fd62f74dee'
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('canonical policy specification changed')
    spec=json.loads(raw)
    env=OmegaConf.create(dict(robot=spec['env_config']['robot'],simulator=dict(config=dict(cameras=dict(
        camera_resolutions=[108,192],camera_types=[dict(rgb=True)])))))
    policy=instantiate(OmegaConf.create(spec['actor_config']),env_config=env,
        algo_config=OmegaConf.create(spec['algo_config']),module_dim_dict=spec['module_dim_dict'],_recursive_=False)
    owner=SimpleNamespace(policy_model=policy,env=SimpleNamespace(num_envs=16),num_steps_per_env=81,
        vg_mode='collect',vg_rollout_action_source='teacher',args=SimpleNamespace(num_ppo_epochs=0,
            num_mini_batches=1,local_batch_size=16,local_mini_batch_size=16,
            per_device_train_batch_size=16,num_micro_batches=1))
    receipt=helper.validate_native(owner)
    return dict(receipt=receipt,policy_spec_sha256=expected,real_policy_class=True,
        declared_collector_arguments=True,resident_owner_used=False)

def compare(helper, original_path, *, device, batches=(1,2,16,32,40), seeds=(31,42,901), windows=5):
    import torch
    from torch.distributions import Normal
    raw=Path(original_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=helper.POLICY_SHA256:raise ValueError('pinned original policy method changed')
    cls=next(node for node in ast.parse(raw).body if isinstance(node,ast.ClassDef))
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='action_mean_with_obj_pred')
    namespace=dict(torch=torch,Normal=Normal)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(original_path),'exec'),namespace)
    original=namespace['action_mean_with_obj_pred']
    class TinyPolicy(torch.nn.Module):
        def __init__(self,batch):
            super().__init__();self.weight=torch.nn.Parameter(torch.arange(11,device=device,dtype=torch.float32)/10)
            self.std=torch.nn.Parameter(torch.full((11,),.001,device=device),requires_grad=False)
            self.clamp_noise_std=False;self.batch=batch
        def forward(self,obs_dict,**kwargs):
            return dict(actions=self.weight.view(1,1,11).expand(self.batch,81,11).contiguous(),
                obj_pred=torch.zeros(self.batch,81,3,device=device))
        @property
        def action_mean(self):return self.distribution.mean
        @property
        def action_std(self):return self.distribution.stddev
    def state():
        return [torch.get_rng_state().clone()]+([torch.cuda.get_rng_state(device).clone()] if str(device).startswith('cuda') else [])
    def equal(left,right):return len(left)==len(right) and all(torch.equal(a,b) for a,b in zip(left,right))
    rows=[]
    for batch in batches:
        for seed in seeds:
            model=TinyPolicy(batch);optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=0.)
            torch.manual_seed(seed);expected=[]
            for _ in range(windows):
                torch.randperm(batch,device=device);value=original(model,{})
                loss=value['action_mean'].square().mean();loss.backward();optimizer.step();optimizer.zero_grad()
                after=state();reset=torch.randint(10000,(batch,),device=device);expected.append((after,reset.clone()))
            torch.manual_seed(seed)
            for index,(after,reset) in enumerate(expected):
                helper.consume_window(batch,device=device)
                if not equal(state(),after):raise ValueError(f'RNG state differs at B{batch}/seed{seed}/window{index}')
                if not torch.equal(torch.randint(10000,(batch,),device=device),reset):raise ValueError('next reset draw differs')
            if not optimizer.state:raise ValueError('reference did not execute a real optimizer')
            rows.append(dict(batch=batch,seed=seed,windows=windows,reference_optimizer_updates=windows,
                adapter_optimizer_updates=0,rng_state_exact=True,next_reset_draws_exact=True,
                final_state_sha256=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in state()]))
    return dict(schema='roborender.bc-collection-rng-device-differential.v1',protocol=helper.PROTOCOL,
        device=str(device),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        cuda_device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        original_policy_sha256=helper.POLICY_SHA256,comparisons=rows,tiny_deterministic_policy=True,
        original_normal_method_executed=True,physics_tested=False,discarded_policy_weights_reproduced=False,
        resident_state_touched=False,compute_release_authorized=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--helper',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda:0'],required=True)
    parser.add_argument('--policy-spec',type=Path)
    args=parser.parse_args();helper=load_helper(args.helper)
    # Native model constructors may print diagnostics. Keep the one structured
    # result on stdout; logs cannot corrupt the controller's JSON receipt.
    with redirect_stdout(sys.stderr):
        guard=architecture_guard(helper,args.policy_spec) if args.policy_spec else None
        result=compare(helper,args.original,device=args.device)
    result['architecture_guard']=guard
    result.update(helper_sha256=hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result,sort_keys=True))
