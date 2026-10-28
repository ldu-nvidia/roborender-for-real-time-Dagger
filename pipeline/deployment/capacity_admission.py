"""Small immutable capacity evidence; no arrays, GPU commands or allocations.

Native teacher stepping is necessary but not sufficient: production still
requires the real two-cohort render/student/learner qualification afterwards.
"""
import hashlib
import json
import math
from pathlib import Path
from pipeline.bc_training.handoff_common import binding,bound

SCHEMA='roborender.leo-paired-capacity-admission.v1'
GIB=1<<30
LEARNER_SCHEMA='roborender.fullbatch-capacity-probe.v1'
LEARNER_SOURCE_PATHS={'pipeline/bc_training/train_bc_h200.py',
    'pipeline/dagger/services/raven_learner_worker.py'}


def derive(evidence,*,source_sha256):
    if set(evidence)!={'pair_result','resources','native_receipts'}:
        raise ValueError('exact paired native capacity evidence required')
    result=bound(evidence['pair_result'])
    batch=result.get('num_envs_per_gpu')
    memory_gib=result.get('memory_gib_per_actor')
    if (result.get('passed') is not True or result.get('failure') is not None
            or result.get('mode')!='teacher-capacity' or result.get('physical_gpus')!=[0,1]
            or result.get('exit_codes')!=[0,0] or result.get('deployment_host')!='leo-rtx-6k'
            or result.get('capacity_steps')!=10 or result.get('conservative_capacity_guard') is not True
            or result.get('source_sha256')!=source_sha256 or type(memory_gib)is not int or memory_gib not in (20,21)
            or batch not in (2,4,8,16,32,64) or type(batch)is not int):
        raise ValueError('capacity probe failed, source changed or safety profile differs')
    if len(evidence['native_receipts'])!=2:raise ValueError('two native completions required')
    for ref in evidence['native_receipts']:
        native=bound(ref)
        if (native.get('result')!='PASS' or native.get('mode')!='teacher-capacity'
                or native.get('num_envs')!=batch or native.get('environment_step_calls')!=10
                or native.get('physical_transition_rows')!=10*batch
                or native.get('optimizer_steps')!=0 or native.get('chunks_published')!=0
                or native.get('production_qualified')is not False):
            raise ValueError('native completion is incomplete or not diagnostic-only')
    ref=evidence['resources'];path=Path(ref['path'])
    if binding(path)!=ref or path.stat().st_size>32<<20:raise ValueError('bounded immutable resource telemetry required')
    samples=[json.loads(line) for line in path.read_text().splitlines()]
    minimum_host=1<<63;max_private=[0,0];max_gpu=[0.,0.];observed=[0,0]
    for row in samples:
        if type(row.get('mem_available_kib'))is not int or row['mem_available_kib']<0:
            raise ValueError('invalid host memory observation')
        host=row['mem_available_kib']*1024;minimum_host=min(minimum_host,host)
        if host<12*GIB:raise ValueError('host reserve failed')
        if 'container_memory' not in row:continue
        if len(row['container_memory'])!=2:raise ValueError('paired memory observations required')
        for gpu,sample in enumerate(row['container_memory']):
            if sample.get('lifecycle')=='completed_attested':continue
            if 'stat' not in sample and row['elapsed_s']<=15:continue
            if (sample.get('sampling_error') or sample.get('oom_killed')
                    or sample.get('memory_swap.current')!=0
                    or sample.get('events',{}).get('oom',0) or sample.get('events',{}).get('oom_kill',0)):
                raise ValueError('owned memory observation unsafe')
            stat=sample['stat']
            if any(type(stat.get(k))is not int or stat[k]<0 for k in ('anon','shmem','kernel')):
                raise ValueError('invalid private memory counters')
            private=sum(stat[k] for k in ('anon','shmem','kernel'))
            if private>=(memory_gib-4)*GIB:raise ValueError('private memory margin failed')
            max_private[gpu]=max(max_private[gpu],private);observed[gpu]+=1
        gpu_rows=row.get('gpu_samples',[])
        if len(gpu_rows)!=2:raise ValueError('paired GPU memory observations required')
        for gpu,line in enumerate(gpu_rows):
            index,used,total,util,power=map(float,line.split(','))
            if (not all(math.isfinite(v) for v in (index,used,total,util,power))
                    or index!=gpu or total<=0 or not 0<=used<.75*total):raise ValueError('GPU headroom failed')
            max_gpu[gpu]=max(max_gpu[gpu],used/total)
    if min(observed)<2:raise ValueError('insufficient actual owned samples')
    return dict(schema=SCHEMA,source_sha256=source_sha256,environments_per_actor=batch,
        physical_gpus=[0,1],probe_memory_gib_per_actor=memory_gib,
        evidence=evidence,minimum_host_available_bytes=minimum_host,
        maximum_private_bytes=max_private,maximum_vram_fraction=max_gpu,
        teacher_capacity_passed=True,full_h200_qualification_required=True)


def validate(admission,*,source_sha256,environments_per_actor):
    expected=derive(admission['evidence'],source_sha256=source_sha256)
    if admission!=expected or admission['environments_per_actor']!=environments_per_actor:
        raise ValueError('selected production capacity differs from actual paired probe')
    return admission


def validate_production_memory_policy(admission):
    """New activations must match production THP and creation-time swap caps.

    Historical inherited-policy probes remain readable through derive/validate;
    they are not evidence for a new disable-v1 runtime.
    """
    result=bound(admission['evidence']['pair_result'])
    scopes=result.get('systemd_scope_admissions',{})
    if (result.get('native_thp_policy')!='disable-v1'
            or result.get('production_memory_policy_matched')is not True
            or result.get('memory_gib_per_actor')!=21 or len(scopes)!=2):
        raise ValueError('paired current production THP/systemd policy evidence required')
    names=[]
    for proof in scopes.values():
        properties=proof.get('properties',{})
        name=proof.get('scope','')
        if (proof.get('schema')!='roborender.native-systemd-scope-admission.v1'
                or proof.get('passed')is not True or proof.get('returncode')!=0
                or proof.get('compute_release_authorized')is not False
                or proof.get('global_host_policy_modified')is not False
                or not name.startswith('docker-') or not name.endswith('.scope')
                or proof.get('cgroup_path')!='/sys/fs/cgroup/system.slice/'+name
                or properties!=dict(ControlGroup='/system.slice/'+name,
                    MemoryMax=str(21*GIB),MemorySwapMax='0',CPUQuotaPerSecUSec='12s')):
            raise ValueError('actual paired systemd zero-swap policy differs')
        names.append(name)
    if len(set(names))!=2:
        raise ValueError('two independent native scopes required')
    for ref in admission['evidence']['native_receipts']:
        row=bound(ref);thp=row.get('native_thp_policy',{});memory=row.get('cgroup_memory',{})
        if (thp.get('schema')!='roborender.native-thp-policy.v1'
                or thp.get('policy')!='disable-v1' or thp.get('disabled_after')!=1
                or type(thp.get('native_pid'))is not int or thp['native_pid']<=0
                or thp.get('global_host_policy_modified')is not False
                or memory.get('memory.max')!=str(21*GIB)
                or memory.get('memory.swap.max')!='0' or memory.get('memory.swap.current')!='0'):
            raise ValueError('actual native THP/zero-swap receipt differs')
    return admission


def validate_learner(admission,*,environments_per_actor,checkpoint_sha256,source_manifest):
    """Full current-policy learner capacity, separate from RTX-only evidence.

    This is a component gate, never a substitute for the existing two-cohort
    renderer/learner/native qualification. Reserve 25 GiB for the colocated
    renderer/gateway and at least 20% device headroom; no unsafe OOM ladder.
    """
    total=2*environments_per_actor
    if (admission.get('schema')!=LEARNER_SCHEMA or admission.get('passed')is not True
            or admission.get('batch')!=total or admission.get('frames')!=81*total
            or admission.get('checkpoint_sha256')!=checkpoint_sha256
            or admission.get('optimizer_steps')!=2 or admission.get('full_batch')is not True
            or admission.get('microbatching')is not False
            or admission.get('activation_checkpointing')is not False
            or admission.get('production_checkpoint_saved')is not False
            or admission.get('production_promoted')is not False
            or admission.get('compute_released')is not False
            or admission.get('current_source_paired_native_and_end_to_end_qualification_still_required')is not True):
        raise ValueError('actual selected-parent full-batch learner probe required')
    files={row['path']:row['sha256'] for row in source_manifest['files']}
    actual=admission.get('source_files',{})
    if set(actual)!=LEARNER_SOURCE_PATHS or any(files.get(k)!=v for k,v in actual.items()):
        raise ValueError('learner capacity source differs from the deployment release')
    keys=('peak_reserved_bytes','peak_allocated_bytes','device_total_bytes','renderer_gateway_allowance_bytes')
    if any(type(admission.get(k))is not int or admission[k]<=0 for k in keys):
        raise ValueError('finite actual device-memory counters required')
    reserved,allocated,device,allowance=(admission[k] for k in keys)
    fraction=(reserved+allowance)/device
    if (allocated>reserved or allowance<25*GIB or fraction>.80
            or admission.get('minimum_joint_headroom_fraction')!=.20
            or admission.get('projected_joint_fraction')!=fraction):
        raise ValueError('learner plus renderer requires at least 20% H200 headroom')
    metrics=admission.get('metrics',[])
    if len(metrics)!=2 or [row.get('step') for row in metrics]!=[0,1]:
        raise ValueError('two actual full-batch probe updates required')
    for row in metrics:
        for key in ('seconds','mse','gradient_norm'):
            value=row.get(key)
            if type(value)not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('invalid learner probe metric')
        if row['seconds']<=0 or row['gradient_norm']<=0:
            raise ValueError('learner probe did not perform a real backward update')
    return admission
