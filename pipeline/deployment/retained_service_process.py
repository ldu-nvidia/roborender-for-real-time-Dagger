"""Explicit, identity-owned persistent service process; never a compute lease.

Only the genuine service bundle, SSH tunnel supervisor and exact production
mode controller are accepted. Arbitrary native/optimizer commands are rejected.
The controller's scientific budgets remain in its immutable mode/coverage plans;
an unrelated 12-hour supervisor timeout must not interrupt a multi-round run.
It does not allocate, delete, release, scale, or expire cloud resources.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.deployment.owned_process import atomic, identity, status, stop

SCHEMA = "roborender.retained-service-process.v1"

def read_spec(path):
    value=json.loads(Path(path).read_bytes())
    if set(value)!={"schema","argv","cwd","env","owner","role","release_compute"}:
        raise ValueError("exact persistent service specification required")
    argv=value["argv"];role=value["role"]
    entry={"gpu-services":"pipeline/deployment/h200/beta_service_host.py",
           "tunnels":"pipeline/deployment/rtx_actor/tunnel_supervisor.py"}.get(role)
    valid_argv=isinstance(argv,list) and len(argv)>=2 and all(isinstance(a,str) and a for a in argv)
    service_entry=bool(valid_argv and entry and Path(argv[1]).is_absolute() and argv[1].endswith('/'+entry))
    controller_entry=bool(valid_argv and role=='mode-controller' and len(argv)==12
        and argv[1:6]==['-m','pipeline.deployment.run_mode','--mode','production','--settings']
        and argv[7]=='--output' and argv[9]=='--execution-settings' and argv[11]=='--execute'
        and all(Path(argv[i]).is_absolute() and 'runtime' in Path(argv[i]).parts
            and '..' not in Path(argv[i]).parts for i in (6,8,10)))
    campaign_entry=bool(valid_argv and role=='campaign-controller' and len(argv)==12
        and argv[1:4]==['-m','pipeline.dagger.mse_campaign','--first-round-config']
        and argv[5]=='--mse-plan' and argv[7]=='--root' and argv[9]=='--operations'
        and argv[11]=='--execute' and all(Path(argv[i]).is_absolute()
            and 'runtime' in Path(argv[i]).parts and '..' not in Path(argv[i]).parts
            for i in (4,6,8,10)))
    if (value["schema"]!=SCHEMA or value["release_compute"] is not False
        or not (service_entry or controller_entry or campaign_entry)
        or not isinstance(argv,list) or len(argv)<2 or not all(isinstance(a,str) and a for a in argv)
        or not Path(argv[0]).is_absolute() or not Path(value["cwd"]).is_absolute()
        or not isinstance(value["owner"],str) or not value["owner"].startswith('dagger-leo-')
        or not isinstance(value["env"],dict)
        or any(not isinstance(k,str) or not isinstance(v,str) for k,v in value['env'].items())
        or any(k in value['env'] for k in ('HOME','CODEX_HOME','LD_PRELOAD','WANDB_API_KEY'))):
        raise ValueError("persistent boundary accepts only explicitly owned services")
    if role=='gpu-services' and ('--continuous-services' not in argv or '--retained-owner' not in argv):
        raise ValueError("retained service bundle must attest its existing allocation owner")
    if role=='mode-controller':
        settings=json.loads(Path(argv[6]).read_bytes())
        execution=json.loads(Path(argv[10]).read_bytes())
        if (value['owner']!='dagger-leo-'+settings.get('run_id','')
                or settings.get('run_id','').startswith('smoke-')
                or execution.get('compute_release_authorized') is not False):
            raise ValueError('production controller identity/retention binding differs')
    if role=='campaign-controller':
        config=json.loads(Path(argv[4]).read_bytes())
        operations=json.loads(Path(argv[10]).read_bytes())
        state=json.loads((Path(argv[8])/'state.json').read_bytes())
        release=state.get('control_only_release',{})
        import hashlib
        if (config.get('schema')!='roborender.cross_gpu_dagger_mse_full295.leo.v2'
                or value['owner']!=config.get('run_id') or state.get('run_id')!=value['owner']
                or operations.get('compute_release_permitted') is not False
                or set(release)!={'path','sha256'}
                or hashlib.sha256(Path(release['path']).read_bytes()).hexdigest()!=release['sha256']):
            raise ValueError('reconciled production campaign identity/retention binding differs')
    return value

def start(directory,spec_path):
    spec=read_spec(spec_path)
    if directory.exists():
        if json.loads((directory/'spec.json').read_bytes())!=spec:raise RuntimeError('service identity changed')
        return status(directory)
    directory.mkdir(parents=True);atomic(directory/'spec.json',spec)
    with (directory/'supervisor.log').open('xb') as log:
        subprocess.Popen([sys.executable,'-m','pipeline.deployment.retained_service_process','supervise',
            '--directory',str(directory)],cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    return dict(state='STARTING',owner=spec['owner'])

def supervise(directory):
    spec=read_spec(directory/'spec.json')
    if (directory/'owner.json').exists():raise RuntimeError('service owner cannot be silently restarted')
    stopping=False
    def requested(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    atomic(directory/'owner.json',dict(pid=os.getpid(),identity=identity(os.getpid()),owner=spec['owner']))
    child=None;result=dict(state='FAILED',owner=spec['owner'],restart_allowed=False,allocation_release_requested=False)
    try:
        with (directory/'console.log').open('xb') as log:
            child=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=dict(os.environ,**spec['env']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None and not stopping:time.sleep(.5)
            result.update(state='STOPPED' if stopping else 'COMPLETE' if child.returncode==0 else 'FAILED',exit_code=child.returncode)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        atomic(directory/'result.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop','supervise'])
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--spec',type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    if 'runtime' not in directory.parts or len(directory.parts)<5:raise ValueError('scoped runtime directory required')
    if args.action=='supervise':supervise(directory);return
    if args.action=='start' and args.spec is None:parser.error('start requires --spec')
    result=start(directory,args.spec) if args.action=='start' else status(directory) if args.action=='status' else stop(directory)
    print(json.dumps(result))
if __name__=='__main__':main()
