"""Explicit production B2 resident pool; legacy B6/B5 qualification is separate."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dagger/rtx/code/src'))
from dagger_rtx.phase_resource_guard import (validate_summary,classify_sample,
    validate_transition,POLICY,SCHEMA as GUARD_SCHEMA,SUMMARY_FIELDS,
    SUSTAINED_SCHEMA,policy_for)

try:
    from .leo_binding import GPU_UUIDS
    from .resident_isaac_client import ResidentClient
except ImportError:
    from leo_binding import GPU_UUIDS
    from resident_isaac_client import ResidentClient

SCHEMA='roborender.leo-production-resident-pool.v1'
SMOKE_SCHEMA='roborender.leo-corrected-smoke-resident-pool.v1'
QUALIFICATION='roborender.leo-production-b2-startup.v1'
OPERATIONS=frozenset(('capture_initial_b2','collect_production_segment','probe_continuous_training',
    'collect_mse_validation','evaluate_explicit_panel'))
GIB=1<<30


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4<<20:
        raise ValueError('bounded regular production pool metadata required')
    raw=path.read_bytes()
    if len(raw)>4<<20: raise ValueError('production pool metadata grew')
    return raw


def validate_pool(pool):
    if (set(pool)!={'schema','source_sha256','jobs_root','assets_root','workers','qualification'}
            or pool['schema'] not in (SCHEMA,SMOKE_SCHEMA) or not re.fullmatch('[0-9a-f]{64}',pool['source_sha256'])
            or len(pool['workers'])!=2):
        raise ValueError('distinct actual production B2 pool required')
    for name in ('jobs_root','assets_root'):
        if not Path(pool[name]).is_absolute() or '..' in Path(pool[name]).parts:
            raise ValueError('absolute production pool paths required')
    for gpu,row in enumerate(pool['workers']):
        if (set(row)!={'physical_gpu','gpu_uuid','port','token_file','runtime_root','container_name',
                'worker_instance_id','native_pid','memory_gib','evaluation_memory_gib','initial_num_envs'}
                or row['physical_gpu']!=gpu or row['gpu_uuid']!=GPU_UUIDS[gpu] or row['port']!=18640+gpu
                or row['memory_gib']!=21 or row['evaluation_memory_gib']!=21 or row['initial_num_envs']!=2
                or type(row['native_pid']) is not int or row['native_pid']<=0
                or not isinstance(row['worker_instance_id'],str) or not row['worker_instance_id']):
            raise ValueError('production B2 native/GPU/cap identity differs')
        for name in ('token_file','runtime_root'):
            if not Path(row[name]).is_absolute() or '..' in Path(row[name]).parts:
                raise ValueError('absolute worker paths required')
    if (set(pool['qualification'])!={'path','sha256'} or not Path(pool['qualification']['path']).is_absolute()
            or not re.fullmatch('[0-9a-f]{64}',pool['qualification']['sha256'])):
        raise ValueError('actual production startup qualification binding required')
    return pool


def validate_guard(summary,worker):
    validate_summary(summary)
    if (not isinstance(summary.get('path'),str)
            or not isinstance(summary.get('raw_sha256'),str)
            or not re.fullmatch('[0-9a-f]{64}',summary['raw_sha256'])):
        raise RuntimeError('hash-bound actual native resource guard required')
    path=Path(summary['path']); root=Path(worker['runtime_root']).resolve()
    if path.is_relative_to('/dagger-runtime'):
        path=root/path.relative_to('/dagger-runtime')
    if not path.is_absolute() or not path.resolve(strict=True).is_relative_to(root):
        raise RuntimeError('native guard receipt escaped its owned runtime')
    try:
        raw=read(path)
    except PermissionError:
        # Preserve the running native release. Older containers wrote this
        # nonsecret metadata root:root 0600; read only its attested guard.
        try:
            from .production_guard_access import read_owned_guard
        except ImportError:
            from production_guard_access import read_owned_guard
        raw=read_owned_guard(path,worker,summary['raw_sha256'])
    if hashlib.sha256(raw).hexdigest()!=summary['raw_sha256']:
        raise RuntimeError('native guard receipt changed')
    value=json.loads(raw)
    interval=value.get('interval_s')
    if (value.get('schema') not in ('roborender.native-phase-resource-guard.v1',GUARD_SCHEMA,SUSTAINED_SCHEMA)
            or type(interval) not in (int,float) or not math.isfinite(interval) or not 0<interval<=.5
            or value.get('passed') is not True or value.get('samples')!=summary['samples']
            or any(value.get(k)!=summary[k] for k in ('minimum_host_available_bytes','swap_peak_bytes',
                'oom_events','limit_pressure_events'))):
        raise RuntimeError('native guard summary does not reproduce actual receipt')
    if value['schema'] in (GUARD_SCHEMA,SUSTAINED_SCHEMA):
        if (summary.get('schema')!=value['schema'] or value.get('policy')!=policy_for(value.get('policy_version'))
                or value.get('sampling_error') is not None
                or any(value.get(k)!=summary.get(k) for k in SUMMARY_FIELDS)):
            raise RuntimeError('native guard policy/summary differs from actual receipt')
        if value['schema']==SUSTAINED_SCHEMA and value.get('pressure_window')!=summary.get('pressure_window'):
            raise RuntimeError('native v3 pressure history differs from actual receipt')
        for key in ('counter_read_attempts','counter_read_recoveries'):
            if key in summary and value.get(key)!=summary[key]:
                raise RuntimeError('native counter-read audit differs from actual receipt')
        validate_summary(value)
        first=value.get('first_sample');last=value.get('last_sample')
        classify_sample(first,policy_version=value['policy_version'])
        classify_sample(last,policy_version=value['policy_version'])
        validate_transition(first,last)
        if (first['events']['max']!=value['initial_limit_pressure_events']
                or last['events']['max']!=value['limit_pressure_events']
                or value['minimum_host_available_bytes']>min(first['host_available_bytes'],last['host_available_bytes'])
                or not isinstance(value.get('memory_current_peak_bytes'),int)
                or not max(first['memory_current_bytes'],last['memory_current_bytes'])<=value['memory_current_peak_bytes']<=21*GIB
                or value['maximum_working_set_bytes']<max(first['working_set_bytes'],last['working_set_bytes'])
                or value['maximum_some_avg10']<max(first['some_avg10'],last['some_avg10'])
                or value['maximum_full_avg10']<max(first['full_avg10'],last['full_avg10'])):
            raise RuntimeError('native v2 guard extrema/counters differ from actual samples')
    elif summary.get('schema') not in (None,'roborender.native-phase-resource-guard.v1'):
        raise RuntimeError('native guard version relabelled')
    return value


def qualification(pool):
    validate_pool(pool); raw=read(pool['qualification']['path'])
    if hashlib.sha256(raw).hexdigest()!=pool['qualification']['sha256']:
        raise RuntimeError('production startup qualification changed')
    value=json.loads(raw)
    if (value.get('schema')!=QUALIFICATION or value.get('complete') is not True
            or value.get('source_sha256')!=pool['source_sha256']
            or value.get('profile')!='initial_training_b2_capture'
            or value.get('native_initializations_per_worker')!=1 or value.get('optimizer_steps')!=0
            or len(value.get('results',[]))!=2):
        raise RuntimeError('actual B2 startup qualification required; old profiles cannot be relabeled')
    for worker,result in zip(pool['workers'],value['results']):
        receipt=result.get('receipt',{}); capture=receipt.get('capture_receipt',{})
        if (result.get('gpu')!=worker['physical_gpu'] or receipt.get('native_pid')!=worker['native_pid']
                or receipt.get('complete') is not True or receipt.get('source_sha256')!=pool['source_sha256']
                or receipt.get('optimizer_steps')!=0
                or receipt.get('native_initializations')!=1 or receipt.get('num_envs')!=2
                or receipt.get('environment_step_calls')!=0
                or capture.get('result')!='PASS' or capture.get('num_envs')!=2
                or capture.get('environment_step_calls')!=0 or len(capture.get('rows',[]))!=2):
            raise RuntimeError('B2 native capture result differs')
        validate_guard(receipt.get('resource_guard'),worker)
    return value


def validate_native_policy(pool, policy):
    """An explicit memory opt-in needs actual hash-bound native startup proof."""
    try:
        from .native_thp_policy import INHERITED, validate_policy
    except ImportError:
        from native_thp_policy import INHERITED, validate_policy
    validate_policy(policy)
    if policy == INHERITED:
        return
    value = qualification(pool)
    for worker, result in zip(pool['workers'], value['results']):
        receipt = result['receipt'].get('native_thp_policy', {})
        if (receipt.get('schema') != 'roborender.native-thp-policy.v1'
                or receipt.get('policy') != policy or type(receipt.get('disabled_after')) is not int
                or receipt.get('disabled_after') != 1
                or receipt.get('native_pid') != worker['native_pid']
                or receipt.get('global_host_policy_modified') is not False):
            raise RuntimeError('explicit native memory policy lacks startup attestation')


def attest_live_owner(worker,source_sha256,*,run=None,mode=None):
    """Fail immediately for an absent/stopped/replaced owned native container.

    Read-only Docker metadata precedes any potentially retrying localhost RPC.
    Reuse the guard reader's exact name/label/GPU/runtime-mount attestation,
    then bind the container's source/GPU/physical-index/port environment. No
    Docker exec, evaluator, restart, stop, reset, or allocation action exists.
    Native PID and lifetime are additionally checked by the following /ready.
    """
    try:
        from .production_guard_access import attest_container
    except ImportError:
        from production_guard_access import attest_container
    run=subprocess.run if run is None else run
    inspected=[]
    def bounded_inspect(argv,**kwargs):
        if argv!=['docker','inspect',worker['container_name']]:
            raise RuntimeError('only exact owned-container metadata is readable')
        result=run(argv,**kwargs)
        if len(result.stdout)>4<<20:raise RuntimeError('bounded owned-container metadata required')
        inspected.extend(json.loads(result.stdout))
        return result
    try:
        container_id=attest_container(worker,run=bounded_inspect)
        if len(inspected)!=1:raise ValueError('one exact native container required')
        value=inspected[0];pid=value.get('State',{}).get('Pid')
        if type(pid) is not int or pid<=0:raise ValueError('live container host PID required')
        expected={'RESIDENT_SOURCE_SHA256':source_sha256,'CUDA_VISIBLE_DEVICES':worker['gpu_uuid'],
                  'RESIDENT_PHYSICAL_GPU':str(worker['physical_gpu']),'RESIDENT_PORT':str(worker['port'])}
        if mode is not None:
            if mode not in ('smoke','production'):raise ValueError('unknown native mode')
            expected['RESIDENT_PIPELINE_MODE']=mode
        actual={}
        for item in value.get('Config',{}).get('Env',[]):
            key,separator,content=item.partition('=')
            if key in expected:
                if not separator or key in actual:raise ValueError('ambiguous owned environment binding')
                actual[key]=content
        if actual!=expected:raise ValueError('native source/GPU/port environment differs')
        return dict(container_id=container_id,host_pid=pid)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError):
        # This precise terminal class/message is consumed by the safe CPU
        # observer. Never disguise dead native lifetime as retryable SSH loss;
        # never include inspect environments or subprocess stderr in logs.
        raise RuntimeError('native owner changed or stopped') from None


def selected_clients(pool, gpus):
    """Attest full immutable qualification, but contact only selected owners.

    An unrelated owner's missing live endpoint must not prevent polling the
    durable result of this command. Pool-wide status/memory admission still
    explicitly selects both owners through clients(). No native state is
    reconstructed or replayed by this control-only isolation.
    """
    if (not isinstance(gpus,(list,tuple)) or not gpus or len(set(gpus))!=len(gpus)
            or any(type(gpu) is not int or gpu not in (0,1) for gpu in gpus)):
        raise ValueError('unique selected owned GPUs required')
    qualification(pool)
    class NativeBoundClient(ResidentClient):
        def ready(self):
            attest_live_owner(self.expected_worker,self.expected_source_sha256,
                             mode='smoke' if pool['schema']==SMOKE_SCHEMA else None)
            ready=super().ready()
            if ready.get('native_pid')!=self.expected_native_pid:
                raise RuntimeError('production native owner changed')
            return ready
    output={}
    for gpu in gpus:
        row=pool['workers'][gpu]
        client=NativeBoundClient(row['port'],row['token_file'],source_sha256=pool['source_sha256'],gpu_uuid=row['gpu_uuid'])
        client.instance=row['worker_instance_id']; client.expected_native_pid=row['native_pid']
        client.expected_worker=row;client.expected_source_sha256=pool['source_sha256']
        client.ready()
        output[gpu]=client
    return output


def clients(pool):
    values=selected_clients(pool,[0,1])
    return [values[gpu] for gpu in (0,1)]


def quiescent(pool):
    return all(c.ready()['phase']=='IDLE' for c in clients(pool))


def fixed_memory_admission(pool):
    """B2/B1 tests retain21GiB caps; no legacy B5 cap increase is performed."""
    from run_throughput_pair import container_memory_sample,memory_kib
    from run_production_resident_isaac import enrich_memory_sample,resource_gate
    if not quiescent(pool): raise RuntimeError('all native owners must be idle at phase boundary')
    host=memory_kib()['MemAvailable']*1024
    if host<8*GIB: raise RuntimeError('host headroom below8GiB')
    samples=[container_memory_sample(w['container_name']) for w in pool['workers']]
    for row in samples:
        if row.get('sampling_error'):raise RuntimeError('actual owned memory counters unavailable')
        enrich_memory_sample(row)
        row['resource_classification_v2']=resource_gate({'MemAvailable':host//1024},row)
    return dict(admitted=True,host_available_bytes=host,cgroup_limits_gib=[21,21],samples=samples,
        cap_changed=False,resource_guard_schema=GUARD_SCHEMA,resource_policy=POLICY)


def observe_commands(commands, values, action):
    """Expose either peer's failure promptly, without changing native work.

    Cancellation only detaches read-only polling. In-flight bounded HTTP reads
    finish normally; no submit, stop, restart or allocation action is performed.
    Successful output retains the caller's lane ordering.
    """
    cancel = Event()
    for client in values.values():
        client.observer_cancel = cancel
    try:
        with ThreadPoolExecutor(len(commands)) as threads:
            futures = {threads.submit(action, row): i for i, row in enumerate(commands)}
            results = [None] * len(commands)
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                cancel.set()
                raise
        return results
    finally:
        for client in values.values():
            client.observer_cancel = None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--action',choices=('status','submit','wait','result','evaluation-memory','training-memory'),required=True)
    p.add_argument('--commands',type=Path)
    args=p.parse_args(); pool=validate_pool(json.loads(read(args.pool)))
    if args.action=='status': output=dict(workers=[c.ready() for c in clients(pool)],quiescent=quiescent(pool))
    elif args.action in ('evaluation-memory','training-memory'):
        output=dict(complete=True,memory_admission=fixed_memory_admission(pool),evaluation_memory=False)
    else:
        commands=json.loads(read(args.commands))
        if (not commands or len({c['gpu'] for c in commands})!=len(commands)
                or any(type(c['gpu']) is not int or c['gpu'] not in (0,1)
                    or c.get('operation') not in (OPERATIONS|{'collect_mse_round'} if pool['schema']==SMOKE_SCHEMA else OPERATIONS) for c in commands)):
            raise ValueError('one command per selected owned GPU required')
        values=selected_clients(pool,[row['gpu'] for row in commands])
        def action(row):
            c=values[row['gpu']]
            if args.action in ('result','wait'):
                result=(c.wait(row['command_id']) if args.action=='wait'
                    else c._request('/v1/commands/'+row['command_id']))
                if result.get('status')=='COMPLETE':
                    validate_guard(result.get('receipt',{}).get('resource_guard'),pool['workers'][row['gpu']])
                return result
            return c.submit(row['command_id'],row['operation'],row['payload'])
        if args.action == 'wait':
            output=dict(results=observe_commands(commands,values,action))
        else:
            with ThreadPoolExecutor(len(commands)) as pool_threads:
                output=dict(results=list(pool_threads.map(action,commands)))
    print(json.dumps(output,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__': main()
