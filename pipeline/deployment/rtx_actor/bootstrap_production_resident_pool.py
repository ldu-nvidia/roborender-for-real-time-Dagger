"""Create a new guarded B2 standing pool and attest real zero-step captures.

No H200 allocation API is used. An interrupted native bootstrap is preserved,
never restarted or relabeled as the historical B6/B5 qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import time

from leo_binding import GPU_UUIDS
from resident_isaac_client import ResidentClient,canonical
from production_resident_pool import SCHEMA,SMOKE_SCHEMA,QUALIFICATION,validate_pool,validate_guard
from run_success_diagnostic import gpu_gate
from run_throughput_pair import memory_kib
from native_thp_policy import INHERITED, DISABLED, validate_policy


class StartupClient(ResidentClient):
    """Do not spend the network recovery budget waiting on a dead local owner."""
    def __init__(self,*args,unit,**kwargs):
        super().__init__(*args,**kwargs)
        self.unit=unit

    def ready(self):
        if subprocess.run(['systemctl','--user','is-active','--quiet',self.unit],timeout=10).returncode:
            raise RuntimeError('native startup owner stopped; reconcile without replay')
        return super().ready()


def publish(path,value):
    path=Path(path); raw=canonical(value)+b'\n'
    with path.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def validate_spec(spec):
    fields={'schema','source_root','source_manifest','source_sha256','run_id','root','python','motion_inventory'}
    if (not fields <= set(spec) or set(spec)-fields-{'mode','native_thp_policy'}
            or spec['schema']!='roborender.production-b2-bootstrap.v1'
            or not re.fullmatch('standing-isaac-leo-prod-[a-z0-9-]{1,42}',spec['run_id'])
            or not re.fullmatch('[0-9a-f]{64}',spec['source_sha256'])):
        raise ValueError('explicit fresh production B2 bootstrap required')
    validate_policy(spec.get('native_thp_policy',INHERITED))
    mode=spec.get('mode','production')
    if mode not in ('smoke','production') or (mode=='smoke') != spec['run_id'].startswith('standing-isaac-leo-prod-smoke-'):
        raise ValueError('explicit native mode and isolated run namespace required')
    if Path(spec['root'])!=Path('/home/ldu/roborender/runtime/standing-isaac')/spec['run_id']:
        raise ValueError('exact fresh owned pool root required')
    for name in ('source_root','source_manifest','root','python'):
        if not Path(spec[name]).is_absolute() or '..' in Path(spec[name]).parts:
            raise ValueError('absolute bootstrap paths required')
    ref=spec['motion_inventory']
    if set(ref)!={'path','raw_sha256'} or not Path(ref['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}',ref['raw_sha256']):
        raise ValueError('pinned full295 inventory required')
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    raw=a.spec.read_bytes()
    if len(raw)>1<<20 or hashlib.sha256(raw).hexdigest()!=a.spec_sha256: raise ValueError('bootstrap spec changed')
    spec=validate_spec(json.loads(raw));source=Path(spec['source_root'])
    manifest=json.loads(Path(spec['source_manifest']).read_bytes())
    if manifest.get('source_sha256')!=spec['source_sha256']: raise ValueError('bootstrap source differs')
    inv=Path(spec['motion_inventory']['path']).read_bytes()
    if len(inv)>4<<20 or hashlib.sha256(inv).hexdigest()!=spec['motion_inventory']['raw_sha256'] or json.loads(inv).get('motion_count')!=295:
        raise ValueError('full295 inventory bytes changed')
    if not a.execute:
        print(json.dumps(dict(execute=False,initial_num_envs_per_worker=2,memory_gib_per_worker=21,
            host_reserve_gib=8,profile='initial_training_b2_capture',qualified=False,source_sha256=spec['source_sha256'])));return
    if socket.gethostname()!='leo-rtx-6k': raise RuntimeError('dedicated Leo host required')
    for gpu in (0,1): gpu_gate(gpu,deployment_host='leo-rtx-6k')
    if memory_kib()['MemAvailable']*1024<50*(1<<30): raise RuntimeError('42GiB pair caps plus8GiB host cushion required')
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=False)
    jobs=root/'jobs';assets=root/'assets';jobs.mkdir();assets.mkdir()
    (jobs/'full-motion-inventory.json').write_bytes(inv)
    token=root/'resident.token';fd=os.open(token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream: stream.write(secrets.token_hex(32)+'\n')
    workers=[];clients=[];units=[]
    publish(root/'bootstrap-intent.json',dict(spec_sha256=a.spec_sha256,source_sha256=spec['source_sha256'],
        profile='initial_training_b2_capture',native_restart_allowed=False))
    try:
        for gpu in (0,1):
            unit='roborender-'+spec['run_id']+f'-g{gpu}'
            command=[spec['python'],str(source/'pipeline/deployment/rtx_actor/run_production_resident_isaac.py'),
                '--source-root',str(source),'--source-manifest',spec['source_manifest'],'--run-id',spec['run_id'],
                '--physical-gpu',str(gpu),'--jobs-root',str(jobs),'--assets-root',str(assets),'--token-file',str(token),
                '--motion-inventory-sha256',spec['motion_inventory']['raw_sha256'],
                '--mode',spec.get('mode','production'),'--execute']
            if spec.get('native_thp_policy',INHERITED) != INHERITED:
                command += ['--native-thp-policy',spec['native_thp_policy']]
            subprocess.run(['systemd-run','--user','--unit',unit,'--property=Restart=no','--property=TimeoutStopSec=45',*command],
                check=True,stdout=subprocess.DEVNULL,timeout=20)
            units.append(unit)
            workers.append(dict(physical_gpu=gpu,gpu_uuid=GPU_UUIDS[gpu],port=18640+gpu,token_file=str(token),
                runtime_root=str(root/f'gpu{gpu}'),container_name=spec['run_id']+f'-g{gpu}',
                memory_gib=21,evaluation_memory_gib=21,initial_num_envs=2))
            clients.append(StartupClient(18640+gpu,token,source_sha256=spec['source_sha256'],gpu_uuid=GPU_UUIDS[gpu],timeout=3,unit=unit))
        deadline=time.monotonic()+1500
        for worker,client,unit in zip(workers,clients,units):
            while True:
                if subprocess.run(['systemctl','--user','is-active','--quiet',unit],timeout=10).returncode:
                    raise RuntimeError('new native startup failed; replay forbidden')
                try: ready=client.ready();break
                except OSError:
                    if time.monotonic()>=deadline: raise TimeoutError('new native startup not ready')
                    time.sleep(2)
            if ready['phase']!='IDLE': raise RuntimeError('new native worker is not idle')
            worker.update(worker_instance_id=ready['worker_instance_id'],native_pid=ready['native_pid'])
        results=[]
        for worker,client in zip(workers,clients):
            name=f'production-startup-g{worker["physical_gpu"]}-b2'
            client.submit(name,'capture_initial_b2',dict(output_name=name))
            result=client.wait(name,wall_budget=600);receipt=result['receipt'];ready=client.ready()
            if (ready['phase']!='IDLE' or ready['native_pid']!=worker['native_pid']
                    or receipt.get('source_sha256')!=spec['source_sha256'] or receipt.get('native_initializations')!=1
                    or receipt.get('num_envs')!=2 or receipt.get('environment_step_calls')!=0):
                raise RuntimeError('actual initial B2 capture/owner differs')
            validate_guard(receipt.get('resource_guard'),worker)
            if spec.get('native_thp_policy',INHERITED) == DISABLED:
                thp=receipt.get('native_thp_policy',{})
                if (thp.get('policy')!=DISABLED or thp.get('disabled_after')!=1
                        or thp.get('native_pid')!=worker['native_pid']
                        or thp.get('global_host_policy_modified') is not False):
                    raise RuntimeError('native process THP opt-in was not attested')
            results.append(dict(gpu=worker['physical_gpu'],receipt=receipt))
        q=publish(root/'qualification.json',dict(schema=QUALIFICATION,complete=True,source_sha256=spec['source_sha256'],
            profile='initial_training_b2_capture',native_initializations_per_worker=1,optimizer_steps=0,
            results=results,policy_quality_certified=False,legacy_b6_b5_qualification_claimed=False))
        pool=dict(schema=SMOKE_SCHEMA if spec.get('mode')=='smoke' else SCHEMA,source_sha256=spec['source_sha256'],jobs_root=str(jobs),assets_root=str(assets),
            workers=workers,qualification=q)
        validate_pool(pool);ref=publish(root/'pool.json',pool)
        print(json.dumps(dict(status='B2_NATIVE_READY',pool=ref,source_sha256=spec['source_sha256'],
            native_initializations_per_worker=1,workers_retained=True)),flush=True)
    except BaseException as exc:
        publish(root/'bootstrap-failure.json',dict(error_type=type(exc).__name__,native_restart_allowed=False,
            already_started_units=units,source_sha256=spec['source_sha256'],workers_not_automatically_restarted=True))
        raise


if __name__=='__main__':main()
