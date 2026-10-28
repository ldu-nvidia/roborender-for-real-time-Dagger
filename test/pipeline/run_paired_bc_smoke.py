"""Live B32x2/four-window BC smoke; detached, bounded, no retries or production.

Run only from a source-pinned release on Leo. Test orchestration is separate
from production collection. All compute is retained, including on failures.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.bc_collection import run_continuous as single
from pipeline.bc_collection.continuous_plan import build_plan, build_request
from pipeline.bc_collection.storage_guard import storage_snapshot, observe_storage
from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL
from pipeline.deployment.rtx_actor.paired_bc_guard import SCHEMA, validate_config
from pipeline.deployment.rtx_actor.bc_resource_observer import (
    PinnedCgroupObserver, recover_observation, observe_command,
)

GIB = 2**30
GUARD = 'pipeline/deployment/rtx_actor/paired_bc_guard.py'
SELF = 'test/pipeline/run_paired_bc_smoke.py'


def publish(path, value):
    single.publish_json(Path(path), value)


def host_gate(host):
    if type(host.get('MemAvailable')) is not int or host['MemAvailable'] * 1024 < 4 * GIB:
        raise RuntimeError('paired host below user-authorized4GiB reserve')


def actor_gate(sample):
    reason = None
    if sample.get('sampling_error') or type(sample.get('pid')) is not int or sample['pid'] <= 0:
        reason = 'observation_unavailable'
    elif sample.get('oom_killed') is not False:
        reason = 'oom_killed_or_unknown'
    elif any(type(sample.get(k)) is not int or sample[k] < 0
             for k in ('memory_current', 'memory_peak', 'memory_swap.current')):
        reason = 'memory_counters_invalid'
    elif any(type(sample.get('events', {}).get(k)) is not int or sample['events'][k] < 0
             for k in ('max', 'oom', 'oom_kill')):
        reason = 'memory_events_invalid'
    elif sample['memory_swap.current'] != 0:
        reason = 'owned_swap_detected'
    elif any(sample['events'][k] for k in ('max', 'oom', 'oom_kill')):
        reason = 'memory_limit_or_oom_event'
    elif sample['memory_current'] + 2 * GIB > 43 * GIB:
        reason = 'actor_headroom_below2GiB'
    elif sample['memory_peak'] > 43 * GIB:
        reason = 'actor_peak_exceeds43GiB'
    if reason:
        raise RuntimeError(reason + ': ' + json.dumps(sample, sort_keys=True))


def record_actor_sample(row, name, sample, gpu, *, enforce_gate):
    # Preserve the rejecting observation BEFORE raising. Otherwise the last
    # row silently omits the failing actor and cannot distinguish pressure
    # from an unavailable read after the exact workload has been stopped.
    row['actors'].append(dict(name=name, container=sample, gpu=gpu))
    if enforce_gate:
        actor_gate(sample)


def paired_command(command, *, actor, stage, config_sha, image):
    cmd = list(command)
    target = 'trainer._target_=bounded_bc_guard.BoundedContinuousBCTrainer'
    if cmd.count(target) != 1 or actor not in (0, 1):
        raise ValueError('expected single bounded collector target')
    cmd[cmd.index(target)] = 'trainer._target_=paired_bc_guard.PairedBCTrainer'
    at = cmd.index(image)
    cmd[at:at] = ['--mount', f'type=bind,src={stage},dst=/bc-pair',
                 '--env', 'ROBORENDER_PAIR_CONFIG_SHA256=' + config_sha,
                 '--env', 'ROBORENDER_PAIR_ACTOR=' + str(actor)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--inventory-request', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    source = args.source_root.resolve(strict=True)
    manifest, manifest_sha = single.read_json(args.source_manifest)
    hashes = single.verify_source(source, manifest)
    if not {GUARD, SELF, single.ACTOR + '/bc_resource_observer.py'} <= hashes.keys():
        raise ValueError('paired source must be frozen and attested')
    paths = single.load_paths(source)
    stage = args.stage
    if (stage.parent != paths.get('run_root') or stage.resolve(strict=False) != stage
            or not stage.name.startswith('bc-pair32-') or len(stage.name) > 65):
        raise ValueError('fresh canonical bc-pair32- stage required')
    prior, _ = single.read_json(args.inventory_request)
    inventory = prior['motion_inventory']  # Motion assets only; never reuse collected data.
    requests = [build_request(build_plan(inventory, run_id=stage.name + f'-g{i}',
        max_windows=4, num_envs=32, process_seed=i, output_dir='/dagger-runtime/bc_isaac',
        collection_storage_protocol=PROTOCOL)) for i in (0, 1)]
    if not args.execute and not args.detach:
        print(json.dumps(dict(staged_only=True, schema=SCHEMA, num_envs=[32, 32], windows=[4, 4],
            seeds=[0, 1], host_reserve_gib=4, serialized_saves=True,
            source_sha256=manifest['source_sha256'], production_admitted=False)))
        return
    if socket.gethostname() != 'leo-rtx-6k' or Path(__file__).resolve() != source / SELF:
        raise RuntimeError('execute paired smoke only from exact pinned Leo release')
    from roborender.infrastructure.source_locations import resolve_source_release
    if resolve_source_release(source) != source:
        raise ValueError('immutable source root required')
    if args.detach:
        if args.execute:
            raise ValueError('choose detach or execute, not both')
        stage.mkdir(exist_ok=False)
        command = [sys.executable, str(source / SELF), '--source-root', str(source),
            '--source-manifest', str(args.source_manifest), '--inventory-request', str(args.inventory_request),
            '--stage', str(stage), '--execute']
        publish(stage / 'launcher-intent.json', dict(command=command, replay_allowed=False, retain_compute=True))
        with (stage / 'launcher.log').open('xb') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, cwd=source)
        publish(stage / 'launcher-owner.json', dict(pid=proc.pid,
            proc_stat=Path(f'/proc/{proc.pid}/stat').read_text(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), command=command))
        print(json.dumps(dict(submitted_once=True, pid=proc.pid, stage=str(stage))))
        return
    stage.mkdir(exist_ok=True)
    if (stage / 'pair-intent.json').exists():
        raise RuntimeError('prior accepted/preparing pair intent; reconcile, never replay')
    publish(stage / 'pair-intent.json', dict(schema=SCHEMA, source_sha256=manifest['source_sha256'],
        source_manifest_sha256=manifest_sha, host_reserve_bytes=4 * GIB,
        replay_allowed=False, retain_compute=True, production_admitted=False))
    sys.path.insert(0, str(source / single.ACTOR))
    from materialize_isaac6 import materialize
    from qualify_native_isaac6 import IMAGE
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib
    from leo_binding import checked_gpu
    from bc_memory_guard import exclusive_native_owner

    def owners():
        return subprocess.check_output(['docker', 'ps', '--filter', 'ancestor=' + IMAGE,
            '--format', '{{.Names}}'], text=True, timeout=15).splitlines()

    actors, failure, samples, started = [], None, [], time.monotonic()
    logs = []
    lock = (paths.get('run_root') / '.continuous-bc-owner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exclusive_native_owner(owners())
        initial = memory_kib()
        host_gate(initial)
        # Diagnostic startup allowance, NOT a proven pair capacity bound. Later
        # step/save checks use actual shared MemAvailable and nonoverlap leases.
        if initial['MemAvailable'] * 1024 < 52 * GIB:
            raise RuntimeError('paired smoke requires52GiB initially available (48GiB startup allowance +4 reserve)')
        publish(stage / 'initial-host-memory.json', initial)
        for i in (0, 1):
            paths.check_assets(i)
            gpu_gate(i, deployment_host='leo-rtx-6k')
            single.gpu_sample(checked_gpu(i, 'bc-collect'))
            publish(stage / f'request{i}.json', requests[i])
        config = validate_config(dict(schema=SCHEMA, num_envs_per_actor=32, windows_per_actor=4,
            host_reserve_bytes=4 * GIB, actor_cap_bytes=43 * GIB, seeds=[0, 1],
            request_sha256=[single.digest((stage / f'request{i}.json').read_bytes()) for i in (0, 1)],
            source_sha256=manifest['source_sha256'], guard_sha256=hashes[GUARD],
            production_admitted=False, retain_compute=True, replay_allowed=False))
        publish(stage / 'config.json', config)
        (stage / 'save.lock').touch(exist_ok=False)
        config_sha = single.digest((stage / 'config.json').read_bytes())
        for i in (0, 1):
            runtime = paths.validate_run(paths.get('bc_collection_runs') / requests[i]['run_id'])
            policy = dict(schema=single.STORAGE_SCHEMA, root=str(paths.get('run_root').parent),
                reserve_bytes=64 * GIB, bytes_per_emission_budget=512 << 20,
                session_metadata_budget_bytes=GIB, poll_seconds=1)
            budget = single.storage_budget(policy, requests[i])
            report = {}
            if observe_storage(storage_snapshot(policy, runtime, runtime), policy, report) < 2 * budget['budget_bytes'] + policy['reserve_bytes']:
                raise RuntimeError('shared disk cannot cover both complete smoke budgets and reserve')
            runtime.mkdir()
            paths.freeze(runtime)
            for name in ('home', 'bc_isaac'):
                (runtime / name).mkdir()
            paths.mounts(i)['cache'].mkdir(parents=True, exist_ok=True)
            publish(runtime / 'intent.json', dict(pair=str(stage), request_sha256=config['request_sha256'][i],
                source_sha256=manifest['source_sha256'], replay_allowed=False, production_admitted=False))
            materialize(paths.get('bc_runtime_bundle'), runtime / 'overlay',
                source / 'pipeline/dagger/rtx/config/assets.manifest.json')
            name, command = single.build_command(request=requests[i], request_path=stage / f'request{i}.json',
                request_sha256=config['request_sha256'][i], source=source, runtime=runtime,
                overlay=runtime / 'overlay', physical_gpu=i, memory_gib=43,
                guard_sha256=hashes[single.ACTOR + '/bounded_bc_guard.py'], runtime_paths=paths)
            command = paired_command(command, actor=i, stage=stage, config_sha=config_sha, image=IMAGE)
            publish(runtime / 'launch.json', dict(schema=SCHEMA, command=command, request=requests[i],
                pair_config_sha256=config_sha, source_sha256=manifest['source_sha256'],
                source_manifest_sha256=manifest_sha, physical_gpu=i, concurrent_collectors=2,
                production_admitted=False, host_reserve_bytes=4 * GIB))
            actors.append(dict(name=name, runtime=runtime, command=command, policy=policy, storage=report,
                gpu_uuid=checked_gpu(i, 'bc-collect'), process=None, observer=None,
                request_sha256=config['request_sha256'][i]))
        existing = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        if any(a['name'] in existing for a in actors):
            raise RuntimeError('paired container namespace already exists')
        single.verify_source(source, manifest)
        exclusive_native_owner(owners())
        if memory_kib()['MemAvailable'] * 1024 < 52 * GIB:
            raise RuntimeError('paired startup admission changed')
        def interrupted(*_):
            raise InterruptedError('stop only the exact paired smoke workloads')
        signal.signal(signal.SIGTERM, interrupted)
        for actor in actors:
            log = (actor['runtime'] / 'console.log').open('xb')
            logs.append(log)
            actor['process'] = subprocess.Popen(actor['command'], stdout=log, stderr=subprocess.STDOUT)
            actor['submitted'] = time.monotonic()
        with (stage / 'memory-samples.jsonl').open('x') as stream:
            while True:
                row = dict(time=time.monotonic(), actors=[])
                try:
                    host = memory_kib()
                    row['host'] = host
                    host_gate(host)
                    observed = owners()
                    if not set(observed) <= {a['name'] for a in actors}:
                        raise RuntimeError('foreign native owner appeared; never stop it')
                    for actor in actors:
                        runtime, proc = actor['runtime'], actor['process']
                        observe_storage(storage_snapshot(actor['policy'], runtime, runtime), actor['policy'], actor['storage'])
                        gpu = single.gpu_sample(actor['gpu_uuid'])
                        def read_actor():
                            terminal = runtime / 'bc_isaac/cohort-manifest.json'
                            def read_terminal():
                                native, receipt_sha = single.read_json(terminal)
                                single.validate_completion(native, requests[len(row['actors'])], actor['request_sha256'])
                                return dict(completion_receipt_sha256=receipt_sha, native_pass_receipt=True)
                            def read_live():
                                if actor['observer'] is None:
                                    actor['observer'] = PinnedCgroupObserver(runtime / 'container.cid',
                                        actor['name'], IMAGE, 43 * GIB)
                                return actor['observer'].sample()
                            return observe_command(proc, read_live, read_terminal, terminal.is_file)

                        def journal(event):
                            event = dict(time=time.monotonic(), actor=actor['name'], **event)
                            with (stage / 'observation-events.jsonl').open('a') as log:
                                log.write(json.dumps(event) + '\n')
                                log.flush()
                                os.fsync(log.fileno())

                        def validate_actor(sample):
                            if sample.get('lifecycle') != 'completed_attested':
                                actor_gate(sample)

                        sample = recover_observation(read_actor, validate_actor, journal=journal,
                            hold=lambda: publish(stage / 'observation-hold.json', dict(actor=actor['name'])),
                            release=lambda: (stage / 'observation-hold.json').unlink(),
                            host_check=lambda: host_gate(memory_kib()),
                            attempts=20 if actor['observer'] is None else 3)
                        record_actor_sample(row, actor['name'], sample, gpu, enforce_gate=False)
                    if time.monotonic() - started > 3600:
                        raise TimeoutError('paired four-window smoke safety timeout')
                except BaseException as exc:
                    row['failure'] = dict(type=type(exc).__name__, message=str(exc))
                    raise
                finally:
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    samples.append(row)
                if all(a['process'].poll() == 0 for a in actors):
                    break
                time.sleep(1)
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        publish(stage / 'abort.json', failure)
    finally:
        for actor in actors:
            if actor['process'] is None:
                continue
            try:
                actor['container_id'] = single.cleanup_exact(actor['name'], actor['runtime'] / 'container.cid', IMAGE)
                actor['process'].wait(timeout=30)
            except Exception as exc:
                failure = dict(type=type(exc).__name__, message='cleanup uncertain: ' + str(exc))
        for log in logs:
            log.close()
        lock.close()
    summaries = []
    for i, actor in enumerate(actors):
        native_path = actor['runtime'] / 'bc_isaac/cohort-manifest.json'
        native = single.read_json(native_path)[0] if native_path.exists() else {}
        live = [r['actors'][i]['container'] for r in samples
                if len(r['actors']) > i and r['actors'][i]['container'].get('pid')]
        summaries.append(dict(actor=i, runtime=str(actor['runtime']), container_id=actor.get('container_id'),
            launcher_exit_code=actor['process'].returncode if actor['process'] is not None else None,
            accepted_windows=native.get('accepted_windows'), emitted_chunks=native.get('emitted_chunks'),
            per_lane=native.get('emitted_per_lane'), initial_reset_all_calls=native.get('initial_reset_all_calls'),
            optimizer_steps=native.get('optimizer_steps'),
            cgroup_peak_bytes=max([r['memory_peak'] for r in live if type(r.get('memory_peak')) is int] or [0]),
            gpu_peak_bytes=max([r['actors'][i]['gpu']['used_bytes'] for r in samples if len(r['actors']) > i] or [0])))
    result = dict(schema=SCHEMA, verdict='PASS' if failure is None else 'FAIL', failure=failure,
        actors=summaries, source_sha256=manifest['source_sha256'], seconds=time.monotonic() - started,
        minimum_host_available_bytes=min([r['host']['MemAvailable'] * 1024 for r in samples if 'host' in r] or [0]),
        host_reserve_bytes=4 * GIB, serialized_saves=True, production_admitted=False,
        replay_allowed=False, retain_compute=True, requires_independent_payload_audit=True)
    publish(stage / 'pair-result.json', result)
    print(json.dumps(result), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
