"""Read-only, host-local diagnostic sampling; transport cannot erase samples.

Executes only the reviewed resource probe constant from the pinned sampler.
Never invokes a simulator operation, allocator or workload lifecycle command.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
SCHEMA='roborender.host-local-resource-journal.v1'
PROBE_SHA='ba75aa2e4474f86ecf642efc91349bc542067643ca59b8db7c4840983b0bf2a1'
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity():
    proc=Path('/proc/self');fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=fields[19],cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
def publish(path,value):
    temporary=path.with_suffix('.tmp');raw=canonical(value)
    with temporary.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
def probe_code(source,kind):
    if digest(source)!=PROBE_SHA:raise ValueError('reviewed resource probe changed')
    wanted={'native':'NATIVE','renderer':'RENDERER'}[kind]
    tree=ast.parse(Path(source).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==wanted for t in n.targets))
def validate_plan(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('kind') not in ('native','renderer')
        or plan.get('period_seconds')!=5 or plan.get('compute_release_authorized') is not False
        or plan.get('native_operations')!=0 or plan.get('lifetime_seconds')!=21600):
        raise ValueError('read-only exact five-second observation plan required')
    for key in ('output','probe_source','source_root'):
        p=Path(plan[key])
        if not p.is_absolute() or '..' in p.parts:raise ValueError('explicit observation paths required')
    return plan
def run(path):
    plan=validate_plan(json.loads(Path(path).read_bytes()));out=Path(plan['output'])
    if socket.gethostname()!=plan['hostname'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=plan['boot_id']:
        raise ValueError('observation host identity changed')
    if digest(__file__)!=plan['helper_sha256']:raise ValueError('observer source changed')
    sys.path.insert(0,plan['source_root']);code=compile(probe_code(plan['probe_source'],plan['kind']),'reviewed_resource_probe','exec')
    out.mkdir(parents=True,exist_ok=False);owner=identity();publish(out/'owner.json',dict(identity=owner,plan_sha256=digest(path)))
    sys.argv=['reviewed-resource-probe',json.dumps(plan['parameters'],separators=(',',':'))]
    deadline=time.monotonic()+plan['lifetime_seconds'];slot=math.ceil(time.time()/5)
    count=0
    try:
        with (out/'resources.jsonl').open('xb') as stream:
            while time.monotonic()<deadline and not (out/'stop').exists():
                time.sleep(max(0,slot*5-time.time()));start=time.time();buffer=io.StringIO()
                with contextlib.redirect_stdout(buffer):exec(code,{})
                end=time.time();value=json.loads(buffer.getvalue())
                if end-start>8 or abs(value['at_unix']-end)>2:raise RuntimeError('host sample timing invalid')
                record=dict(schema=SCHEMA,kind=plan['kind'],slot=slot,started_unix=start,finished_unix=end,
                    owner=owner,value=value)
                stream.write(canonical(record));stream.flush();os.fsync(stream.fileno());count+=1
                publish(out/'status.json',dict(state='RUNNING',identity=owner,samples=count,slot=slot,at_unix=end))
                slot+=1
                if time.time()>slot*5+3:raise RuntimeError('host sample cadence lost')
        publish(out/'status.json',dict(state='STOPPED',identity=owner,samples=count,at_unix=time.time()))
    except BaseException as error:
        publish(out/'status.json',dict(state='FAILED',identity=owner,samples=count,at_unix=time.time(),error_type=type(error).__name__,message=str(error)[-500:]))
        raise
def merge_rows(native,renderer):
    if native['schema']!=SCHEMA or renderer['schema']!=SCHEMA or native['kind']!='native' or renderer['kind']!='renderer' or native['slot']!=renderer['slot']:
        raise ValueError('actual same-slot host samples required')
    for row in (native,renderer):
        if type(row['slot']) is not int or row['slot']<0:raise ValueError('integer sample slot required')
        stamps=[row['started_unix'],row['finished_unix'],row['value']['at_unix']]
        if any(type(t) not in (int,float) or not math.isfinite(t) for t in stamps):raise ValueError('finite sample times required')
        if not 0<=stamps[1]-stamps[0]<=8 or abs(stamps[0]-row['slot']*5)>3:raise ValueError('actual sample timing invalid')
    start=min(native['started_unix'],renderer['started_unix']);end=max(native['finished_unix'],renderer['finished_unix'])
    if not 0<=end-start<=8 or any(not start-2<=r['value']['at_unix']<=end+2 for r in (native,renderer)):
        raise ValueError('host clocks/capture times differ')
    return dict(schema='roborender.scaled-success-resource-sample.v1',at_unix=end,sample_started_unix=start,sample_finished_unix=end,
        host_available_bytes=native['value']['host_available_bytes'],native=native['value']['native'],renderers=renderer['value']['renderers'])
def start(path):
    plan=validate_plan(json.loads(Path(path).read_bytes()));out=Path(plan['output']);control=out.with_name(out.name+'-control')
    if control.exists() or out.exists():raise ValueError('observer launch already accepted; reconcile only')
    if digest(__file__)!=plan['helper_sha256']:raise ValueError('observer source changed')
    control.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'run','--plan',str(Path(path).resolve())]
    with (control/'intent.json').open('xb') as stream:
        stream.write(canonical(dict(plan_sha256=digest(path),argv=argv,native_operations=0,compute_release_authorized=False)));stream.flush();os.fsync(stream.fileno())
    with (control/'stdout.log').open('xb') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    publish(control/'popen.json',dict(pid=child.pid,accepted=True,native_operations=0))
    return dict(pid=child.pid,accepted=True,native_operations=0)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','start']);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    if args.action=='run':run(args.plan)
    else:print(json.dumps(start(args.plan),sort_keys=True))
