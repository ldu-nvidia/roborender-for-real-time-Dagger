"""Actual CUDA BC wrap/resume probe; diagnostic-only, never a production parent.

Uses63 freshly rendered chunks, the production cyclic sampler and learner, and
the saved initial student. Four uninterrupted updates are compared against two
updates resumed from the actual update2 checkpoint. No simulator operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def bound(ref):
    if sha(ref['path'])!=ref['raw_sha256']:raise ValueError('bound input changed')
    return read(ref['path'])

def validate_plan(p):
    if (p.get('schema')!='roborender.bc-cyclic-device-probe.v1' or p.get('chunk_count')!=63
            or p.get('batch_size')!=32 or p.get('reference_updates')!=4 or p.get('resume_from_update')!=2
            or p.get('actual_optimizer_operations')!=6 or p.get('validation_enabled') is not False
            or p.get('diagnostic_only') is not True or p.get('production_promoted') is not False
            or p.get('compute_release_authorized') is not False):raise ValueError('bounded diagnostic scope changed')
    rows=p.get('selected_chunks',[])
    if (len(rows)!=63 or len({r['path'] for r in rows})!=63
            or [r['path'] for r in rows]!=sorted(r['path'] for r in rows)):
        raise ValueError('exact ordered nondivisible fresh pool required')
    return p

def execute(plan_path):
    p=validate_plan(read(plan_path));source=Path(p['source_root']).resolve(strict=True)
    if sha(__file__)!=p['probe_sha256']:raise ValueError('test helper changed')
    manifest=bound(p['source_manifest'])
    if manifest['source_sha256']!=p['source_sha256']:raise ValueError('source identity differs')
    sys.path.insert(0,str(source))
    from pipeline.bc_training import train_bc_h200 as bc,cyclic_sampling as cyclic
    from pipeline.bc_training import qualify_raven_bc32_lifecycle as q32,qualify_raven_bc_lifecycle as base
    from pipeline.dagger.services import learner_memory
    import numpy as np
    import random
    import torch
    from torch.utils.data import DataLoader
    for name,module in list(sys.modules.items()):
        if name=='pipeline' or name.startswith('pipeline.'):
            locations=([module.__file__] if getattr(module,'__file__',None) else [])+list(getattr(module,'__path__',[]))
            if any(not Path(x).resolve().is_relative_to(source) for x in locations):raise ValueError('source jail escaped')
    fresh=bound(p['fresh_training_plan']);proof=bound(p['fresh_parent_evidence'])
    if (proof.get('fresh_input_protocol',{}).get('retired_data_used') is not False
            or fresh.get('fresh_input_protocol')!=proof['fresh_input_protocol']
            or p['selected_chunks']!=sorted(fresh['selected_chunks'],key=lambda r:r['path'])[:63]):
        raise ValueError('fresh diagnostic lineage changed')
    for row in p['selected_chunks']:
        if sha(row['path'])!=fresh['selected_payload_sha256'][row['path']]:raise ValueError('fresh rendered payload changed')
    if sha(fresh['bootstrap'])!=q32.BOOTSTRAP_SHA or sha(fresh['policy_spec'])!=q32.SPEC_SHA:
        raise ValueError('saved initial policy/spec changed')
    output=Path(p['output']);output.mkdir(exist_ok=False)
    identity=base.cuda_identity(120.);memory=q32.memory_admission(identity)
    # The still-running joint observer reserves25% on every H200. The old
    # standalone BC guard reserved16GiB only; use the stricter admitted cap
    # here, not a weakened observer or eviction of the retained renderers.
    paired_memory=learner_memory.begin(torch.cuda)
    memory['allocator_cap_bytes']=min(memory['allocator_cap_bytes'],paired_memory['policy']['allocator_cap_bytes'])
    memory['allocator_fraction']=memory['allocator_cap_bytes']/identity['gpu_total_bytes']
    memory['paired_headroom_policy']=paired_memory['policy']
    torch.cuda.set_per_process_memory_fraction(memory['allocator_fraction'],0)
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
    generator=torch.Generator().manual_seed(42)
    paths=[Path(r['path']) for r in p['selected_chunks']]
    spec=read(fresh['policy_spec'])
    class Tracked(bc.BCChunks):
        def __getitem__(self,index):return dict(super().__getitem__(index),probe_sample_index=index)
    dataset=Tracked(paths,int(spec['input_contract']['actor_obs_dim']),11,include_object_target=True)
    sampler=cyclic.CyclicBatchSampler(63,32)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,generator=generator)
    mean=torch.tensor(spec['input_contract']['rgb_mean'],device=device).view(1,1,1,1,3)
    std=torch.tensor(spec['input_contract']['rgb_std'],device=device).view(1,1,1,1,3)
    sampling=bc.training_sampling_contract(paths,Path(p['rendered_root']),shuffle=False,cyclic_batch_size=32)
    batch_contract=dict(bc.student_lifecycle_batch_fields(bc.REPLAY_STUDENT_LIFECYCLE,32),
        chunks_per_update=32,training_chunk_count=63,updates_per_epoch=2,drop_last=False,
        partial_batch='wrap_sorted_pool',epoch_is_exact_dataset_pass=False)
    metadata=dict(training_sampling_contract=sampling,training_batch_contract=batch_contract,
        diagnostic_only=True,production_promoted=False,policy_spec=spec)
    def build():
        policy,_=bc.build_policy(Path(fresh['policy_spec']),Path(fresh['bootstrap']),device)
        q32.verify_initial_policy(policy,Path(fresh['bootstrap']));bc.configure_raven_offline_student(policy)
        bc.install_index_safe_max_pool2d(policy)
        optimizer=torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad],lr=2e-4,weight_decay=0.)
        return policy,optimizer
    def memory_check():
        learner_memory.observe(paired_memory,torch.cuda,'bc_cyclic_operation_boundary')
        free,total=torch.cuda.mem_get_info()
        if free<16<<30 or torch.cuda.max_memory_reserved()>memory['allocator_cap_bytes']:
            raise RuntimeError('unchanged B32 headroom gate failed; no retry or microbatch fallback')
        return dict(free_bytes=free,total_bytes=total,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    import wandb
    run=wandb.init(entity='nvidia',project='dagger',name=output.name,job_type='bc-cyclic-diagnostic',
        mode='online',resume='never',dir=str(output),config=p)
    bc.atomic_json(output/'wandb.json',dict(id=run.id,url=run.url))
    metrics=[]
    def epochs(policy,optimizer,accepted,branch,number):
        policy.train()
        for _ in range(number):
            sampler.start_after(accepted)
            for batch in loader:
                memory_check();indices=batch['probe_sample_index'].tolist()
                if indices!=[(accepted*32+i)%63 for i in range(32)]:raise ValueError('actual DataLoader cursor reset/dropped rows')
                vision,actor,target=bc.prepare_batch(batch,device,mean,std)
                obj=batch['object_target'].to(device=device,dtype=torch.float32)
                before={name:int(x.num_batches_tracked) for name,x in policy.named_modules()
                    if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)}
                optimizer.zero_grad(set_to_none=True)
                loss=bc.backward_raven_offline_bc_mse(policy,vision,actor,target,obj,chunk_batch_size=32)
                torch.nn.utils.clip_grad_norm_([x for x in policy.parameters() if x.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();accepted+=1;torch.cuda.synchronize()
                if len(optimizer.state)!=78 or any(int(x['step'])!=accepted for x in optimizer.state.values()):
                    raise ValueError('actual AdamW state/update count differs')
                if any(int(x.num_batches_tracked)-before[name]!=82 for name,x in policy.named_modules() if name in before):
                    raise ValueError('full replay/learning BatchNorm population changed')
                row=dict(branch=branch,lineage_step=accepted,train_mse=loss,sample_indices=indices,
                    cursor=cyclic.cursor_state(63,32,accepted),memory=memory_check())
                metrics.append(row);bc.atomic_json(output/'progress.json',dict(metrics=metrics))
                run.log(dict(branch=branch,lineage_global_step=accepted,train_mse=loss,
                    next_chunk_index=row['cursor']['next_index']))
                del vision,actor,target,obj
            path=output/(branch+'-update'+str(accepted)+'.pt')
            bc.save_checkpoint(path,policy,optimizer,accepted//2,accepted,None,None,metadata,generator)
        return accepted
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        policy,optimizer=build();epochs(policy,optimizer,0,'reference',2)
        del policy,optimizer;torch.cuda.empty_cache()
        policy,optimizer=build();saved=torch.load(output/'reference-update2.pt',map_location='cpu',weights_only=False)
        cyclic.validate_checkpoint_sampling(saved['training_sampling_contract'],saved['training_batch_contract'],saved['training_cursor'],2)
        policy.load_state_dict(saved['policy_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict'])
        rng=saved['rng_state'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda']);generator.set_state(rng['train_generator'])
        del saved;epochs(policy,optimizer,2,'resumed',1)
        reference=torch.load(output/'reference-update4.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(output/'resumed-update4.pt',map_location='cpu',weights_only=False)
        for key in ('policy_state_dict','optimizer_state_dict','rng_state','training_cursor'):
            base.assert_equal(reference[key],resumed[key])
        result=dict(schema=p['schema']+'.result',result='PASS',actual_optimizer_operations=6,
            policy_optimizer_rng_resume_exact=True,actual_loader_indices_exact=True,
            memory=memory_check(),memory_policy=memory,device=identity,metrics=metrics,source_sha256=p['source_sha256'],
            diagnostic_only=True,production_promoted=False,compute_release_authorized=False)
        bc.atomic_json(output/'result.json',result)
        run.summary.update(dict(result='PASS',actual_optimizer_operations=6,policy_optimizer_rng_resume_exact=True,
            actual_loader_indices_exact=True,production_promoted=False));run.finish();return result
    except BaseException:
        run.finish(exit_code=1);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);a=p.parse_args()
    print(json.dumps(execute(a.plan),sort_keys=True))
