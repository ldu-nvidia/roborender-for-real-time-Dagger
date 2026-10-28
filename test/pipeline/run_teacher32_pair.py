"""Detached, one-shot paired teacher smoke. No allocation lifecycle operations."""
import fcntl
import hashlib
import json
import socket
import secrets
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    release, output = map(Path, sys.argv[1:3])
    mixed = len(sys.argv) == 6 and sys.argv[3] == '--mixed'
    mode_based = mixed or sys.argv[3:] == ['--mode-three']
    if sys.argv[3:] and not mode_based:
        raise ValueError('only the explicit --mode-three extension is supported')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    host_lock = Path('/data1/workspaces/roborender/runs/teacher32-pair.lock').open('a')
    fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path[:0] = [str(release), str(release/'pipeline/deployment/rtx_actor')]
    from pipeline.success_rate.teacher_episodes import make_spec, digest
    from pipeline.deployment.rtx_actor.resident_isaac_client import ResidentClient
    from pipeline.deployment.rtx_actor.native_artifact_io import publish_json
    from pipeline.deployment.rtx_actor.bc_resource_observer import PinnedCgroupObserver
    from pipeline.deployment.rtx_actor.materialize_isaac6 import materialize
    from pipeline.deployment.rtx_actor.qualify_native_isaac6 import command, IMAGE
    from pipeline.bc_collection.run_continuous import verify_source
    from verify_teacher_episode_smoke import verify
    if mixed:
        from pipeline.success_rate.mixed_episodes import make_spec as mixed_spec
        inputs_path, checkpoint_path = map(Path,sys.argv[4:6])

    root = Path('/data1/workspaces/roborender')
    uuids = ['GPU-4d58bc2b-43c8-6ff8-e7fd-2348575f0cff',
             'GPU-b92877a7-a5e0-9a83-3bc0-d8ede481f35d']
    def available():
        lines = dict(s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())
        return int(lines['MemAvailable'].split()[0])*1024
    def gpu_read(rank):
        row = subprocess.check_output(['nvidia-smi','-i',uuids[rank],
            '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
        row = [s.strip() for s in row]
        assert row[:2] == [str(rank),uuids[rank]], row
        return dict(uuid=uuids[rank],used_mib=int(row[2]),total_mib=int(row[3]),utilization=int(row[4]))
    actors = []
    try:
        assert socket.gethostname() == 'leo-rtx-6k'
        assert not (output/'plan.json').exists(), 'existing intent: reconcile, never duplicate'
        manifest = json.loads((release.parent/'source-manifest.json').read_text())
        verify_source(release, manifest)
        for _ in range(3):
            assert available() >= 52*2**30, 'paired startup host RAM admission'
            for rank in (0,1):
                sample = gpu_read(rank)
                assert sample['used_mib'] < 1024 and sample['utilization'] < 10
            time.sleep(1)
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader'],text=True)
        assert all(g not in apps for g in uuids), apps
        bc = root/'runs/bc-production10000-20261027-074500/request.json'
        bc_raw = bc.read_bytes()
        if mode_based:
            for port in (18640,18641):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1',port))
        expected = 128 if mixed else 192 if mode_based else 128
        plan = dict(schema='roborender.mixed32-pair.v1' if mixed else 'roborender.teacher32-pair.v2' if mode_based else 'roborender.teacher32-pair.v1',
            seeds=[20261102,20261103] if mixed else [20261031,20261101] if mode_based else [20261029,20261030],
            bc_request_sha256=hashlib.sha256(bc_raw).hexdigest(),expected_episodes=expected,
            source_sha256=manifest['source_sha256'],gpu_uuids=uuids,
            host_reserve_bytes=8*2**30,compute_release_authorized=False)
        command_id = 'mixed-two' if mixed else 'teacher-three'
        publish_json(output/'plan.json',plan)
        pair_hash = hashlib.sha256((output/'plan.json').read_bytes()).hexdigest()
        started = time.monotonic()
        # Prepare BOTH immutable commands before any native acceptance.
        for rank in (0,1):
            run = output/f'gpu{rank}'
            run.mkdir(); (run/'home').mkdir()
            spec = make_spec(json.loads(bc_raw),plan['bc_request_sha256'],seed=plan['seeds'][rank],
                             mode_based=mode_based)
            if mixed:
                spec = mixed_spec(json.loads(bc_raw),plan['bc_request_sha256'],seed=plan['seeds'][rank],
                    inputs_sha256=hashlib.sha256(inputs_path.read_bytes()).hexdigest())
            publish_json(run/'teacher-episodes.json',spec)
            if mode_based:
                fd = os.open(run/'mode-token', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
                with os.fdopen(fd,'w') as stream:
                    stream.write(secrets.token_urlsafe(48)+'\n')
            mounts = dict(raven=root/'simulation_assets/rtx_BC_transfer',
                          anchors=root/'simulation_assets/anchor_pool_2048',
                          cache=root/f'caches/isaac/leo-gpu{rank}')
            assert all(p.is_dir() for p in mounts.values())
            code = release/'pipeline/dagger/rtx/code'
            overlay = run/'overlay'
            materialize(mounts['raven'],overlay,code.parent/'config/assets.manifest.json')
            name = f'rr-{output.name}-g{rank}'
            cmd = command(name,run,overlay,code,32,mode='mixed-episodes' if mixed else 'teacher-episodes',
                deployment_host='leo-rtx-6k',physical_gpu=rank,memory_gib=32,
                cpu_count=12,compact_host_heap=True,seed=spec['seed'],storage_paths=mounts,
                teacher_episode_spec=run/'teacher-episodes.json',teacher_bc_request=bc,
                teacher_pair_dir=output, **(dict(mixed_inputs=inputs_path,mixed_checkpoint=checkpoint_path) if mixed else {}))
            cmd.remove('--rm')
            cmd[2:2] = ['--detach','--cidfile',str(run/'container.cid')]
            publish_json(run/'launch-intent.json',dict(command=cmd,source_sha256=manifest['source_sha256'],
                spec=spec,gpu_uuid=uuids[rank],pair_sha256=pair_hash,compute_release_authorized=False))
            client = ResidentClient(18640+rank,run/'mode-token',source_sha256=manifest['source_sha256'],
                                    gpu_uuid=uuids[rank]) if mode_based else None
            actors.append(dict(run=run,name=name,command=cmd,rank=rank,observer=None,
                               client=client,submitted=False,spec=spec))
        for actor in actors:
            reply = subprocess.run(actor['command'],capture_output=True,text=True,timeout=90)
            publish_json(actor['run']/'launch-response.json',dict(returncode=reply.returncode,
                stdout=reply.stdout,stderr=reply.stderr))
            assert reply.returncode == 0, reply.stderr
        released = False
        last_progress, progress_at = [None,None], [time.monotonic()]*2
        while True:
            assert time.monotonic()-started < (43200 if mixed else 3600), 'paired smoke safety deadline'
            statuses = []
            for actor in actors:
                run = actor['run']
                if actor['observer'] is None:
                    actor['observer'] = PinnedCgroupObserver(run/'container.cid',actor['name'],IMAGE,32*2**30)
                observer = actor['observer']
                sample = observer.sample()
                host_available = available()
                row = dict(time_ns=time.time_ns(),container=sample,host_available_bytes=host_available,
                           gpu=gpu_read(actor['rank']))
                with (run/'memory-samples.jsonl').open('a') as log:
                    log.write(json.dumps(row)+'\n'); log.flush()
                assert host_available >= 8*2**30, 'paired host reserve reached'
                assert sample['memory_swap.current'] == 0
                assert all(sample['events'][k] == 0 for k in ['max','oom','oom_kill'])
                if mode_based and not actor['submitted'] and (run/'mode-ready.json').exists():
                    client = actor['client']
                    ready = client.ready()
                    assert ready['phase']=='IDLE' and ready['mode']=='IDLE'
                    publish_json(run/'mode-initial-owner.json',ready)
                    idle = dict(schema='roborender.isaac-phase.v1',mode='IDLE',operation='idle',parameters={})
                    client.submit('idle-before-test','run_phase',idle)
                    publish_json(run/'mode-initial-idle.json',client.wait('idle-before-test',wall_budget=30))
                    payload = dict(schema='roborender.isaac-phase.v1',mode='STUDENT_TEST' if mixed else 'TEACHER_TEST',
                        operation='evaluate_collection_mixed' if mixed else 'evaluate_collection_teacher',
                        parameters=dict(arm='mixed' if mixed else 'teacher',spec_sha256=digest(actor['spec'])))
                    publish_json(run/'mode-command-intent.json',dict(command_id=command_id,
                        worker_instance_id=ready['worker_instance_id'],operation='run_phase',payload=payload))
                    acceptance = client.submit(command_id,'run_phase',payload)
                    publish_json(run/'mode-acceptance.json',acceptance)
                    actor['submitted']=True
                status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
                assert status.get('phase') != 'FAILED', status
                statuses.append(status)
            if mixed:
                assert released or time.monotonic()-started < 1800, 'native startup/barrier stalled'
                for rank,s in enumerate(statuses):
                    progress = (s.get('phase'),s.get('vector_steps'),s.get('completed'))
                    if progress != last_progress[rank]:
                        last_progress[rank], progress_at[rank] = progress, time.monotonic()
                    assert (not released or s.get('phase')=='COMPLETE_IDLE'
                            or time.monotonic()-progress_at[rank] < 900), f'mixed GPU{rank} vector progress stalled'
            if not released and all(s.get('phase') == 'READY_PAIR' for s in statuses):
                assert all(s['pair_sha256'] == pair_hash for s in statuses)
                publish_json(output/'start.json',dict(pair_sha256=pair_hash,ranks=[0,1]))
                released = True
                print('BOTH_READY: admitted simultaneous evaluation',flush=True)
            if all(s.get('phase') == 'COMPLETE_IDLE' for s in statuses):
                if mode_based:
                    terminal = [a['client']._request('/v1/commands/'+command_id) for a in actors]
                    if any(r['status'] in ('QUEUED','RUNNING') for r in terminal):
                        time.sleep(1)
                        continue
                    assert all(r['status']=='COMPLETE' for r in terminal), terminal
                    for actor, receipt in zip(actors,terminal):
                        client, run = actor['client'], actor['run']
                        original = json.loads((run/'mode-acceptance.json').read_text())
                        payload = json.loads((run/'mode-command-intent.json').read_text())['payload']
                        assert client.submit(command_id,'run_phase',payload)==original
                        assert client._request('/v1/commands/'+command_id)==receipt
                        idle = dict(schema='roborender.isaac-phase.v1',mode='IDLE',operation='idle',parameters={})
                        client.submit('idle-after-test','run_phase',idle)
                        idle_receipt=client.wait('idle-after-test',wall_budget=30)
                        before=json.loads((run/'mode-initial-idle.json').read_text())
                        assert before['receipt']['kit_identity']==receipt['receipt']['kit_identity']==idle_receipt['receipt']['kit_identity']
                        ready=client.ready()
                        assert ready['phase']=='IDLE' and ready['mode']=='IDLE'
                        publish_json(run/'mode-completion.json',dict(teacher=receipt,idle_after=idle_receipt,
                            owner=ready,duplicate_returned_same_receipt=True))
                reports = []
                starts, ends = [], []
                for actor, status in zip(actors,statuses):
                    run, observer = actor['run'], actor['observer']
                    publish_json(run/'host-result.json',dict(verdict='PASS',container_id=observer.cid,
                        pid=observer.pid,source_sha256=manifest['source_sha256'],warm_owner_retained=True,
                        elapsed_seconds=time.monotonic()-started,result_sha256=hashlib.sha256(
                            (run/'result.json').read_bytes()).hexdigest(),compute_released=False))
                    reports.append(verify(run))
                    starts.append(json.loads((run/'evaluation-start.json').read_text())['time_ns'])
                    ends.append(status['time_ns'])
                assert released and max(starts) < min(ends), 'no actual concurrent evaluation'
                successes = sum(r['successes'] for r in reports)
                report = dict(verdict='PASS',expected=expected,completed=sum(r['completed'] for r in reports),successes=successes,
                    success_rate=successes/expected,actors=reports,overlap_seconds=(min(ends)-max(starts))/1e9,
                    mode_invocation=mode_based,
                    compute_released=False,warm_owners_retained=True,student_rendering_certified=mixed,
                    policy='mixed_teacher_student' if mixed else 'teacher')
                publish_json(output/'paired-result.json',report)
                print(json.dumps(report),flush=True)
                return
            time.sleep(3)
    except BaseException as exc:
        for actor in actors:
            if (actor['run']/'container.cid').exists() and not (actor['run']/'observer-hold.json').exists():
                publish_json(actor['run']/'observer-hold.json',dict(error=str(exc)))
        if not (output/'host-fault.json').exists():
            publish_json(output/'host-fault.json',dict(error=str(exc),type=type(exc).__name__,
                native_replay_allowed=False,compute_released=False))
        raise


if __name__ == '__main__':
    main()
