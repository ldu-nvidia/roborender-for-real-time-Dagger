"""Isolated Leo qualification of the standing workers; not production training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import is_source_release_location, resolve_source_release
from resident_memory_admission import GIB, admit_future_caps



def validate_spec(spec):
    """Source-owned parameterization of the previously exercised warm probe."""
    import re
    keys = {'schema', 'source_root', 'source_manifest', 'source_sha256', 'run_id',
            'root', 'old_round_root', 'cold_evaluation_root', 'panel_sha256', 'checkpoints', 'python'}
    if (not isinstance(spec, dict) or set(spec) != keys
            or spec['schema'] != 'roborender.resident-qualification-request.v1'
            or not re.fullmatch(r'standing-isaac-leo-[a-z0-9-]{1,48}', spec['run_id'])
            or set(spec['checkpoints']) != {'bc', 'dagger'}):
        raise ValueError('explicit fresh native warm qualification identity required')
    base = Path('/home/ldu/roborender')
    if Path(spec['root']) != base / 'runtime/standing-isaac' / spec['run_id']:
        raise ValueError('qualification output must be the exact owned resident runtime')
    for key in ('source_root', 'source_manifest', 'root', 'old_round_root', 'cold_evaluation_root', 'python'):
        value = Path(spec[key])
        if not value.is_absolute() or '..' in value.parts:
            raise ValueError('qualification paths must be absolute and explicit')
    if not is_source_release_location(spec['source_root']):
        raise ValueError('qualification requires immutable native source')
    for value in (spec['source_sha256'], spec['panel_sha256'], *spec['checkpoints'].values()):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('qualified source/panel/checkpoint hashes required')
    return spec


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('scoped standing qualification stopped')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    raw_spec = args.spec.read_bytes()
    if hashlib.sha256(raw_spec).hexdigest() != args.spec_sha256:
        raise ValueError('immutable native qualification spec changed')
    spec = validate_spec(json.loads(raw_spec))
    if not args.execute:
        print(json.dumps({'execute': False, 'source_sha256': spec['source_sha256'],
                          'run_id': spec['run_id'], 'gpu_count': 2, 'h200_count': 0,
                          'required_host_headroom_gib': 50}))
        return
    source = resolve_source_release(spec['source_root'], error='qualification requires immutable native source')
    sys.path.insert(0, str(source / 'pipeline/deployment/rtx_actor'))
    from resident_isaac_client import ResidentClient, canonical
    from leo_binding import GPU_UUIDS
    from run_throughput_pair import container_memory_sample, memory_kib, memory_safety
    from run_success_diagnostic import gpu_gate
    root = Path(spec['root'])
    root.mkdir(parents=True, exist_ok=False)
    jobs, assets = root / 'jobs', root / 'assets'
    jobs.mkdir(); assets.mkdir()
    token = root / 'resident.token'
    fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(secrets.token_hex(32)+'\n')
    def save(name, value):
        p = root / name
        temporary = p.with_suffix(p.suffix+'.tmp')
        temporary.write_bytes(canonical(value))
        temporary.replace(p)
    def report(phase, **fields):
        value = {'phase':phase,'at_ns':time.time_ns(), **fields}
        save('qualification-progress.json', value)
        print(json.dumps(value), flush=True)
    panel = Path(spec['old_round_root']) / 'panel.json'
    raw = panel.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['panel_sha256']:
        raise RuntimeError('cold reference panel identity differs')
    (jobs / 'panel.json').write_bytes(raw)
    for arm, digest in spec['checkpoints'].items():
        old = Path(spec['old_round_root']) / 'assets' / (digest+'.pt')
        if hashlib.sha256(old.read_bytes()).hexdigest() != digest:
            raise RuntimeError('existing checkpoint identity differs')
        destination = assets / (digest+'.pt')
        if not destination.exists():
            os.link(old, destination)
    request = {'schema':'roborender.warm-paired-panel.v1','panel_sha256':spec['panel_sha256'],
               'cohort_limit':len(json.loads(raw)['cohorts']),'checkpoint_sha256':spec['checkpoints']}
    (jobs / 'paired-request.json').write_bytes(canonical(request))
    units, workers, clients = [], [], []
    launches, startup_seconds = {}, {}
    try:
        for gpu in (0,1):
            gpu_gate(gpu, deployment_host='leo-rtx-6k')
        if memory_kib()['MemAvailable'] < 50*1024**2:
            raise RuntimeError('resident pair needs 42 GiB caps plus 8 GiB host cushion')
        for gpu in (0,1):
            unit = f"roborender-{spec['run_id']}-g{gpu}"
            launches[gpu] = time.monotonic()
            command = [spec['python'], str(source / 'pipeline/deployment/rtx_actor/run_resident_isaac.py'),
                       '--source-root',str(source),'--source-manifest',spec['source_manifest'],
                       '--run-id',spec['run_id'],'--physical-gpu',str(gpu),'--jobs-root',str(jobs),
                       '--assets-root',str(assets),'--token-file',str(token),'--memory-gib','21','--execute']
            subprocess.run(['systemd-run','--user','--unit',unit,'--property=Restart=no',
                            '--property=TimeoutStopSec=45',*command], check=True,
                           stdout=subprocess.DEVNULL, timeout=20)
            units.append(unit)
            workers.append({'physical_gpu':gpu,'gpu_uuid':GPU_UUIDS[gpu],'port':18640+gpu,
                'token_file':str(token),'runtime_root':str(root / f'gpu{gpu}'),
                'container_name':spec['run_id']+f'-g{gpu}', 'memory_gib':21,'evaluation_memory_gib':28})
            clients.append(ResidentClient(18640+gpu, token, source_sha256=spec['source_sha256'],gpu_uuid=GPU_UUIDS[gpu]))
        report('initializing_standing_workers', native_initializations_target=1)
        deadline = time.monotonic()+1500
        for gpu, client in enumerate(clients):
            while True:
                if subprocess.run(['systemctl','--user','is-active','--quiet',units[gpu]],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode:
                    raise RuntimeError('standing worker exited before native readiness')
                try:
                    ready = client.ready()
                    break
                except OSError:
                    if subprocess.run(['systemctl','--user','is-active','--quiet',units[gpu]],
                                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode:
                        raise RuntimeError('standing worker exited before native readiness')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('standing native startup did not finish')
                    report('waiting_for_native_ready', physical_gpu=gpu)
                    time.sleep(2)
            workers[gpu]['worker_instance_id'] = ready['worker_instance_id']
            workers[gpu]['native_pid'] = ready['native_pid']
            startup_seconds[str(gpu)] = time.monotonic()-launches[gpu]
        def phase(gpu, name, operation, payload):
            report('native_phase_start', physical_gpu=gpu, command=name, operation=operation)
            client = clients[gpu]
            client.submit(name, operation, payload)
            result = client.wait(name, wall_budget=3600)
            if result['receipt'].get('native_initializations') != 1:
                raise RuntimeError('Kit reinitialized during a resident phase')
            ready = client.ready()
            if ready['native_pid'] != workers[gpu]['native_pid'] or ready['phase'] != 'IDLE':
                raise RuntimeError('resident owner changed or is not idle')
            report('native_phase_complete', physical_gpu=gpu, command=name,
                   seconds=result['seconds'], profile=result['receipt'].get('profile'))
            return result
        def cap(gpu, limit):
            if any(client.ready()['phase']!='IDLE' for client in clients):
                raise RuntimeError('memory cap change outside all-worker barrier')
            samples = [container_memory_sample(w['container_name']) for w in workers]
            current = memory_kib()
            memory_safety(current,current,samples)
            target_caps = [21 * GIB, 21 * GIB]
            target_caps[gpu] = limit * GIB
            admission = admit_future_caps(current['MemAvailable'] * 1024, samples, target_caps)
            if not admission['admitted']:
                raise RuntimeError('resident evaluation host budget insufficient')
            report('memory_cap_admitted', physical_gpu=gpu, admission=admission)
            subprocess.run(['docker','update','--memory',f'{limit}g','--memory-swap',f'{limit}g',
                            workers[gpu]['container_name']],check=True,stdout=subprocess.DEVNULL,timeout=20)
        results = []
        cold_root = Path(spec['cold_evaluation_root']) / 'bc'
        for gpu in (0,1):
            def capture(name, profile, seed):
                return phase(gpu,name,'capture_reset',{'profile':profile,'seed':seed,'output_name':name,
                    'panel_file':'panel.json','panel_sha256':spec['panel_sha256']})
            a = capture(f'g{gpu}-training-first','training',42+gpu)
            cap(gpu,28)
            b = capture(f'g{gpu}-evaluation-reset','evaluation',1103)
            cold = json.loads((cold_root / 'cohort-000/initial-state.json').read_bytes())
            if b['receipt']['initial_state'] != cold:
                save(f'cold-state-mismatch-g{gpu}.json',{'cold':cold,'warm':b['receipt']['initial_state']})
                raise RuntimeError('B5 warm/cold reset differs; panel must not be relabeled')
            paired = phase(gpu, f'g{gpu}-paired-success','evaluate_pair',{
                'panel_file':'panel.json','panel_sha256':spec['panel_sha256'],
                'request_file':'paired-request.json','request_sha256':hashlib.sha256(canonical(request)).hexdigest(),
                'output_name':f'g{gpu}-paired-success'})
            output = Path(workers[gpu]['runtime_root']) / 'phases' / f'g{gpu}-paired-success'
            cold_result = json.loads((cold_root/'result.json').read_bytes())
            if json.loads((output/'bc/result.json').read_bytes())['runtime_physics_sha256'] != cold_result['runtime_physics_sha256']:
                raise RuntimeError('warm/cold runtime physics or collider identity differs')
            for cohort in json.loads(raw)['cohorts']:
                name = f"cohort-{cohort['index']:03d}"
                if json.loads((output/'bc'/name/'initial-state.json').read_bytes()) != json.loads((cold_root/name/'initial-state.json').read_bytes()):
                    raise RuntimeError('a later warm/cold cohort reset differs')
            c = capture(f'g{gpu}-training-return','training',42+gpu)
            for row in (a,c):
                if row['receipt']['environment_step_calls'] != 0 or row['receipt']['num_envs'] != 6:
                    raise RuntimeError('profile reset probe stepped or changed B6')
            cap(gpu,21)
            results.append({'gpu':gpu,'first':a,'evaluation_reset':b,'return':c,'paired':paired})
        qualified = {'complete':True,'source_sha256':spec['source_sha256'],
            'profiles':['training_b6','evaluation_b5','training_b6'],
            'native_initializations_per_worker':1,'paired_initial_states':True,
            'cold_initial_states_equal':True,'optimizer_steps':0,'live_collection_learner_validated':False,
            'startup_seconds':startup_seconds,'results':results}
        save('qualification.json',qualified)
        for worker in workers:
            worker.pop('native_pid')
        pool = {'schema':'roborender.leo-resident-pool.v1','source_sha256':spec['source_sha256'],
                'jobs_root':str(jobs),'assets_root':str(assets),'workers':workers,
                'qualification':{'path':str(root/'qualification.json'),
                                 'sha256':hashlib.sha256((root/'qualification.json').read_bytes()).hexdigest()}}
        save('pool.json',pool)
        report('resident_pool_qualified_and_idle', pool=str(root/'pool.json'),
               production_admitted=False, native_initializations_per_worker=1)
    except BaseException as exc:
        report('qualification_failed',error_type=type(exc).__name__,native_replay_allowed=False)
        for unit in units:
            subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
        raise


if __name__=='__main__':
    main()
