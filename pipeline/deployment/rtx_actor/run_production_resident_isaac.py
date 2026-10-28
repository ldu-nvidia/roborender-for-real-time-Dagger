"""Opt-in B2 standing owner using the qualified Isaac6 launcher and fixed21GiB."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

from run_resident_isaac import build_resident_command
from materialize_isaac6 import materialize
from qualify_native_isaac6 import IMAGE,REPO,RAVEN
from run_success_diagnostic import gpu_gate
from run_throughput_pair import memory_kib,container_memory_sample
from native_thp_policy import INHERITED, DISABLED, ENV as THP_ENV, validate_policy

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import (own_cgroup_root,sample_resources,
    classify_sample,validate_transition,SUSTAINED_SCHEMA as RESOURCE_SCHEMA,
    SUSTAINED_POLICY as POLICY,SUSTAINED_VERSION,PressureWindow,POLICY_VERSION)

GIB=1<<30
SYSTEMD_SWAP_ANNOTATION='org.systemd.property.MemorySwapMax=uint64 0'


def prepare_population_barriers(pool_root):
    """Private shared receipts, separate from read-only jobs/configs and assets."""
    root=Path(pool_root)
    if not root.is_absolute() or root.resolve(strict=True)!=root:
        raise ValueError('physical native pool root required')
    path=root/'population-barriers'
    if path.is_symlink():
        raise ValueError('native population barrier directory cannot be a symlink')
    path.mkdir(mode=0o700,exist_ok=True)
    if not path.is_dir() or path.resolve(strict=True)!=path:
        raise ValueError('physical population barrier directory required')
    return path


def production_command(*,motion_inventory_sha256,mode='production',native_thp_policy=INHERITED,
                       scaled_success=False,scaled_teacher=False,**values):
    validate_policy(native_thp_policy)
    if mode not in ('smoke','production'): raise ValueError('explicit native mode required')
    if type(scaled_success) is not bool or (scaled_success and mode != 'production'):
        raise ValueError('scaled success is a separate explicit production-owner capability')
    if type(scaled_teacher) is not bool or (scaled_teacher and (not scaled_success or mode != 'production')):
        raise ValueError('scaled teacher requires a separate opt-in production scaled owner')
    if not re.fullmatch('[0-9a-f]{64}',motion_inventory_sha256): raise ValueError('full motion inventory binding required')
    if values.get('memory_gib',21)!=21: raise ValueError('production B2 owner requires21GiB')
    values['memory_gib']=21
    command=build_resident_command(**values)
    if command.count('num_envs=6')!=1: raise RuntimeError('qualified launcher batch argument changed')
    command[command.index('num_envs=6')]='num_envs=2'
    command=['RESIDENT_MOTION_INVENTORY=/resident/jobs/full-motion-inventory.json'
             if item.startswith('RESIDENT_MOTION_INVENTORY=') else item for item in command]
    if any(item.startswith('org.systemd.property.MemorySwapMax=') for item in command):
        raise RuntimeError('unexpected inherited systemd swap annotation')
    # Docker's equal memory/swap flags set the kernel file to zero, but some
    # runc/systemd versions omit the explicit zero property. A later host
    # daemon-reload can then restore infinity. Bind zero to THIS scope during
    # creation; never change global host policy or repair a running native owner.
    command[command.index(IMAGE):command.index(IMAGE)]=['--annotation',SYSTEMD_SWAP_ANNOTATION,
        '--mount','type=bind,src='+str(Path(values['jobs']).parent/'scene-reconfigure.lock')+',dst=/resident/scene-reconfigure.lock,readonly',
        '--mount','type=bind,src='+str(Path(values['jobs']).parent/'population-barriers')+',dst=/resident/population-barriers',
        '--env','RESIDENT_POPULATION_BARRIERS=/resident/population-barriers',
        '--env','RESIDENT_SCENE_LOCK_FILE=/resident/scene-reconfigure.lock',
        '--env','RESIDENT_PIPELINE_MODE='+mode,
        '--env','RESIDENT_PRODUCTION_FULL295=1',
        '--env','RESIDENT_RESOURCE_POLICY='+SUSTAINED_VERSION,
        '--env','RESIDENT_MOTION_INVENTORY_SHA256='+motion_inventory_sha256,
        '--env','RESIDENT_PRODUCTION_MOTION_ROOT=/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775']
    if native_thp_policy != INHERITED:
        command[command.index(IMAGE):command.index(IMAGE)] = ['--env', THP_ENV+'='+native_thp_policy]
    if scaled_success:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            '--env', 'RESIDENT_SCALED_SUCCESS=roborender.scaled-rendered-success-protocol.v1']
    if scaled_teacher:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            '--env', 'RESIDENT_SCALED_TEACHER=roborender.scaled-teacher-benchmark.v1']
    return command


def resource_gate(host,sample,*,policy_version=POLICY_VERSION,pressure=None,now=None):
    if host.get('MemAvailable',0)*1024<8*GIB: raise RuntimeError('production host headroom below8GiB')
    if (sample.get('sampling_error') or sample.get('memory_swap.current')!=0 or sample.get('oom_killed')
            or type(sample.get('pid')) is not int or sample['pid']<=0
            or sample.get('resource_sample_v2',{}).get('process_id')!=sample['pid']):
        raise RuntimeError('production native cgroup resource counters unsafe or unavailable')
    result=classify_sample(sample['resource_sample_v2'],policy_version=policy_version)
    if policy_version==SUSTAINED_VERSION:
        if pressure is None:raise RuntimeError('stateful v3 pressure history required')
        pressure.observe(sample['resource_sample_v2'],time.monotonic() if now is None else now)
        sample['pressure_window']=pressure.summary()
    return result


def enrich_memory_sample(sample,*,policy_version=POLICY_VERSION):
    """Read only the inspected owned native process's actual cgroup/PSI."""
    if type(sample.get('pid')) is not int or sample['pid']<=0:
        raise RuntimeError('actual owned native host PID required')
    root=own_cgroup_root(membership=Path(f'/proc/{sample["pid"]}/cgroup'))
    actual=sample_resources(cgroup_root=root,policy_version=policy_version)
    actual['process_id']=sample['pid']
    sample['resource_sample_v2']=actual
    return sample


class NativeScopeAdmissionError(RuntimeError):
    """Bounded nonsecret evidence of an unsafe/unobservable owned scope."""
    def __init__(self, diagnostic):
        self.diagnostic=diagnostic
        super().__init__('owned systemd scope resource admission failed')


def attest_systemd_scope(cgroup_root,*,run=None):
    """Read the exact Docker scope property, not merely its current kernel cap.

    Creation-time annotation is applied by runc before the container entrypoint.
    This host-side admission precedes native capture/evaluation qualification.
    Periodic kernel counter guards remain unchanged and mandatory.
    """
    root=Path(cgroup_root)
    if (not root.is_absolute() or root.parent!=Path('/sys/fs/cgroup/system.slice')
            or not re.fullmatch(r'docker-[0-9a-f]{64}\.scope',root.name)):
        raise NativeScopeAdmissionError(dict(schema='roborender.native-systemd-scope-admission.v1',
            reason='exact_owned_docker_scope_required'))
    invoke=subprocess.run if run is None else run
    diagnostic=dict(schema='roborender.native-systemd-scope-admission.v1',scope=root.name,
        cgroup_path=str(root),compute_release_authorized=False)
    try:
        result=invoke(['systemctl','show',root.name,
            '--property=ControlGroup,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec'],
            capture_output=True,text=True,timeout=3)
    except (OSError,subprocess.SubprocessError) as exc:
        raise NativeScopeAdmissionError(diagnostic|dict(reason='systemd_scope_observation_unavailable',
            error_type=type(exc).__name__)) from None
    stderr=result.stderr.encode()
    diagnostic.update(returncode=result.returncode,stderr_bytes=len(stderr),
        stderr_sha256=hashlib.sha256(stderr).hexdigest())
    raw=result.stdout.encode()
    if result.returncode or len(raw)>4096:
        raise NativeScopeAdmissionError(diagnostic|dict(reason='systemd_scope_observation_unavailable',
            stdout_bytes=len(raw),stdout_sha256=hashlib.sha256(raw).hexdigest()))
    observed={};valid=True
    patterns={'ControlGroup':r'/system.slice/docker-[0-9a-f]{64}\.scope',
        'MemoryMax':r'(?:[0-9]+|infinity)','MemorySwapMax':r'(?:[0-9]+|infinity)',
        'CPUQuotaPerSecUSec':r'(?:[0-9]+(?:\.[0-9]+)?(?:us|ms|s|min|h)|infinity)'}
    for line in result.stdout.splitlines():
        key,separator,value=line.partition('=')
        if not separator or key not in patterns or key in observed or not re.fullmatch(patterns.get(key,''),value):
            valid=False;continue
        observed[key]=value
    expected=dict(ControlGroup=str(root).removeprefix('/sys/fs/cgroup'),
        MemoryMax=str(21*GIB),MemorySwapMax='0',CPUQuotaPerSecUSec='12s')
    if not valid or observed!=expected:
        raise NativeScopeAdmissionError(diagnostic|dict(reason='systemd_scope_resource_properties_differ',
            observed_properties=observed))
    return diagnostic|dict(passed=True,properties=observed,
        property_output_sha256=hashlib.sha256(raw).hexdigest(),global_host_policy_modified=False)


def sampling_diagnostic(exc):
    """Keep only the resource reader's bounded, nonsecret structured evidence.

    A bare exception name is not enough to diagnose which procfs counter
    failed. Do not serialize arbitrary exception messages, commands or locals.
    """
    value=getattr(exc,'diagnostic',None)
    result=dict(error_type=type(exc).__name__)
    if isinstance(value,dict):
        try:
            raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        except (TypeError,ValueError):
            result['resource_read_diagnostic_unavailable']='invalid_structured_value'
        else:
            result['resource_read_diagnostic_sha256']=hashlib.sha256(raw).hexdigest()
            if len(raw)<=32<<10:
                result['resource_read']=json.loads(raw)
            else:
                result['resource_read_diagnostic_unavailable']='bounded_size_exceeded'
    return result


def write_failure_evidence(runtime,exc,*,container,source_sha256,last_observation,
                           native_docker_pid,native_process_alive):
    """Persist cause BEFORE cleanup; never confuse cleanup with native crash.

    This records only public resource counters and owned identifiers. It has no
    restart, simulator, allocation, credential or model-data operation.
    """
    value=dict(schema='roborender.native-launcher-failure.v1',at_ns=time.time_ns(),
        container=container,source_sha256=source_sha256,
        native_docker_pid=native_docker_pid,native_process_alive=native_process_alive,
        cleanup_requested_by_launcher=True,compute_release_authorized=False,
        partial_native_state_replay_allowed=False,error=sampling_diagnostic(exc),
        last_resource_observation=last_observation)
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(raw)>64<<10:raise RuntimeError('bounded native failure evidence required')
    path=Path(runtime)/'launcher-failure.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-root','source-manifest','jobs-root','assets-root','token-file'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run-id',required=True); p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True)
    p.add_argument('--motion-inventory-sha256',required=True)
    p.add_argument('--mode',choices=('smoke','production'),default='production')
    p.add_argument('--native-thp-policy',choices=(INHERITED,DISABLED),default=INHERITED)
    p.add_argument('--scaled-success',action='store_true',
        help='Enable separately admitted scaled rendered evaluation; does not certify capacity')
    p.add_argument('--scaled-teacher',action='store_true',
        help='Opt in to a private PPO teacher at startup; requires --scaled-success and fresh admission')
    p.add_argument('--execute',action='store_true'); a=p.parse_args()
    if a.scaled_teacher and (not a.scaled_success or a.mode != 'production'):
        raise ValueError('scaled teacher requires a separate opt-in production scaled owner')
    if not re.fullmatch('standing-isaac-leo-prod-[a-z0-9-]{1,42}',a.run_id):
        raise ValueError('fresh explicit production standing identity required')
    if (a.mode=='smoke') != a.run_id.startswith('standing-isaac-leo-prod-smoke-'):
        raise ValueError('smoke native owners require a separate smoke run namespace')
    source=resolve_source_release(a.source_root)
    manifest=json.loads(a.source_manifest.read_bytes())
    if hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=manifest['source_sha256']:
        raise ValueError('source manifest identity changed')
    for row in manifest['files']:
        path=(source/row['path']).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('production standing source changed')
    runtime=REPO/'runtime/standing-isaac'/a.run_id/f'gpu{a.physical_gpu}'
    if runtime.exists(): raise ValueError('existing native lifetime cannot be replayed')
    token=a.token_file.resolve(strict=True)
    if token.stat().st_mode&0o077: raise ValueError('owner-only resident token required')
    inventory=(a.jobs_root/'full-motion-inventory.json').read_bytes()
    if hashlib.sha256(inventory).hexdigest()!=a.motion_inventory_sha256 or json.loads(inventory).get('motion_count')!=295:
        raise ValueError('actual full295 inventory bytes differ')
    if not a.execute:
        print(json.dumps(dict(execute=False,initial_num_envs=2,memory_gib=21,minimum_host_gib=8,
            production_phase_guards=True,resource_guard_schema=RESOURCE_SCHEMA,
            resource_policy=POLICY,native_thp_policy=a.native_thp_policy,
            source_sha256=manifest['source_sha256'],scaled_success=a.scaled_success,
            **({'scaled_teacher':True} if a.scaled_teacher else {})))); return
    if socket.gethostname()!='leo-rtx-6k': raise RuntimeError('dedicated Leo host required')
    gpu_gate(a.physical_gpu,deployment_host='leo-rtx-6k')
    with socket.socket() as listener: listener.bind(('127.0.0.1',18640+a.physical_gpu))
    if memory_kib()['MemAvailable']*1024<29*GIB: raise RuntimeError('21GiB owner plus8GiB host reserve required')
    runtime.mkdir(parents=True); (runtime/'home').mkdir()
    from scene_reconfiguration_lock import prepare_scene_lock
    if a.jobs_root.resolve(strict=True).parent!=runtime.parent:
        raise ValueError('scene lock and native jobs must share the owned pool root')
    prepare_scene_lock(runtime.parent)
    prepare_population_barriers(runtime.parent)
    materialize(RAVEN,runtime/'overlay',source/'pipeline/dagger/rtx/config/assets.manifest.json')
    name=a.run_id+f'-g{a.physical_gpu}'
    cmd=production_command(name=name,runtime=runtime,overlay=runtime/'overlay',source=source,
        jobs=a.jobs_root.resolve(strict=True),assets=a.assets_root.resolve(strict=True),token=token,
        source_sha=manifest['source_sha256'],gpu=a.physical_gpu,memory_gib=21,
        motion_inventory_sha256=a.motion_inventory_sha256,mode=a.mode,native_thp_policy=a.native_thp_policy,
        scaled_success=a.scaled_success,scaled_teacher=a.scaled_teacher)
    (runtime/'launch.json').write_text(json.dumps(dict(command=cmd,source_sha256=manifest['source_sha256'],
        initial_num_envs=2,memory_gib=21,minimum_host_reserve_bytes=8*GIB,sample_interval_seconds=.5,
        resource_guard_schema=RESOURCE_SCHEMA,resource_policy=POLICY,scaled_success=a.scaled_success,
        **({'scaled_teacher':True} if a.scaled_teacher else {}))))
    def interrupted(*_): raise KeyboardInterrupt('owned production resident stop requested')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    process=None
    last_observation=None
    systemd_admission=None
    try:
        with (runtime/'console.log').open('xb') as log,(runtime/'resources.jsonl').open('x') as resources:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            (runtime/'launcher.json').write_text(json.dumps(dict(container=name,docker_pid=process.pid,initial_num_envs=2)))
            start=time.monotonic()
            previous=None
            pressure=PressureWindow()
            while process.poll() is None:
                host=memory_kib(); sample=container_memory_sample(name)
                observed='memory_swap.current' in sample and not sample.get('sampling_error')
                if observed:
                    try:
                        enrich_memory_sample(sample,policy_version=SUSTAINED_VERSION)
                        if systemd_admission is None:
                            proof=attest_systemd_scope(sample['resource_sample_v2']['cgroup_path'])
                            raw=json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
                            path=runtime/'systemd-scope-admission.json'
                            with path.open('xb') as stream:
                                stream.write(raw);stream.flush();os.fsync(stream.fileno())
                            systemd_admission=dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest())
                        sample['systemd_scope_admission']=systemd_admission
                    except NativeScopeAdmissionError:
                        raise
                    except Exception as exc:
                        sample['sampling_error']='native v2 sampling failed: '+type(exc).__name__
                        sample['sampling_diagnostic']=sampling_diagnostic(exc)
                        observed=False
                last_observation=dict(at_ns=time.time_ns(),host_available_bytes=host['MemAvailable']*1024,
                    container=sample,counters_observed=observed)
                if host['MemAvailable']*1024<8*GIB: raise RuntimeError('production host headroom below8GiB')
                if observed:
                    resource_gate(host,sample,policy_version=SUSTAINED_VERSION,pressure=pressure)
                    validate_transition(previous,sample['resource_sample_v2'])
                    previous=sample['resource_sample_v2']
                elif previous is not None or time.monotonic()-start>30:
                    raise RuntimeError('owned cgroup counters unavailable')
                resources.write(json.dumps(last_observation)+'\n'); resources.flush()
                if not (runtime/'resident-state.json').exists() and time.monotonic()-start>1500:
                    raise TimeoutError('production native startup deadline; no automatic restart')
                time.sleep(.5)
            if process.returncode: raise RuntimeError('production native owner exited unsuccessfully')
    except BaseException as exc:
        try:
            proof=write_failure_evidence(runtime,exc,container=name,
                source_sha256=manifest['source_sha256'],last_observation=last_observation,
                native_docker_pid=process.pid if process is not None else None,
                native_process_alive=process is not None and process.poll() is None)
            print(json.dumps(dict(native_launcher_failure=proof,compute_release_authorized=False)),
                file=sys.stderr,flush=True)
        except Exception as evidence_error:
            # Failure to write evidence must not suppress safe owned cleanup,
            # replace the actual failure, or trigger a retry/restart.
            print(json.dumps(dict(native_launcher_failure_evidence_error=type(evidence_error).__name__,
                compute_release_authorized=False)),file=sys.stderr,flush=True)
        if process is not None and process.poll() is None:
            from qualify_graphics import cleanup_owned
            cleanup_owned(name); process.wait(timeout=30)
        raise


if __name__=='__main__': main()
