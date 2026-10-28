"""Isolated B1/B4/B6 full-buffer qualification. Plan-only unless --execute is given."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import resolve_source_release

from bc_memory_guard import GIB, admission, runtime_guard, augment_command, exclusive_native_owner


def prepare_output_directories(runtime, requests):
    """Host-owned output directories permit small post-Docker evidence writes.

    Native payload files may still be root-owned/readable. Never chown payloads,
    adopt an existing output, or change the original writer.
    """
    runtime = Path(runtime).resolve(strict=True)
    paths = []
    for request in requests:
        raw = Path(request['output_dir'])
        if '..' in raw.parts or not raw.is_relative_to('/dagger-runtime'):
            raise ValueError('output directory must stay inside the fresh owned runtime')
        relative = raw.relative_to('/dagger-runtime')
        target = runtime / relative
        resolved = target.resolve(strict=False)
        if (not relative.parts or target.exists() or target.is_symlink()
                or resolved != target or not resolved.is_relative_to(runtime)
                or any(target == prior or target.is_relative_to(prior) or prior.is_relative_to(target)
                       for prior in paths)):
            raise ValueError('output directories must be fresh and distinct')
        paths.append(target)
    for target in paths:
        target.mkdir(parents=True, exist_ok=False)
    return paths


def validate_requests(requests, *, qualification_only, qualification_num_envs=6,
                      qualification_max_windows=4):
    if not requests or len(requests) > (1 if qualification_only else 4):
        raise ValueError('qualification requires one cohort; guarded production sessions are bounded to four cohorts')
    b = len(requests[0]['motions'])
    if not 1 <= b <= 6 or any(len(row['motions']) != b for row in requests):
        raise ValueError('guarded warm sessions require one exact B1..B6; short tails are separate sessions')
    if qualification_only:
        r = requests[0]
        if (type(qualification_num_envs) is not int or qualification_num_envs not in (1, 4, 6)
                or type(qualification_max_windows) is not int or qualification_max_windows not in (4, 8)
                or b != qualification_num_envs
                or r['schema'] != 'roborender.bc-native-collection-request.v2'
                or type(r['max_windows']) is not int or r['max_windows'] != qualification_max_windows
                or r['chunks_per_env'] != [1] * b or r['selected_chunk_ordinals_per_env'] != [[1]] * b
                or r['exposure_span_chunks_per_env'] != [2] * b or r['split'] != 'validation'):
            raise ValueError('only explicitly chosen B1, B4 or B6 later-ordinal full-buffer qualification is allowed')
    elif any(r['schema'] != 'roborender.bc-native-collection-request.v2' for r in requests):
        raise ValueError('new guarded production collection requires v2 selected-ordinal requests')
    return b


def validate_prior_boundary(path, expected_sha, requests, execution, *, _owned_root='/home/ldu/roborender/runtime/bc-sessions'):
    """Resume only after an entire declared session completed, never a prefix."""
    if path is None or expected_sha is None:
        if path is not None or expected_sha is not None:
            raise ValueError('prior session requires an explicit execution receipt SHA')
        return None
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(_owned_root):
        raise ValueError('prior session must be an owned Leo BC runtime')
    raw = (path / 'session-execution.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('prior completed-session receipt changed')
    receipt = json.loads(raw)
    result = json.loads((path / 'bc-session-result.json').read_bytes())
    host = json.loads((path / 'host-result.json').read_bytes())
    if (receipt.get('schema') != 'roborender.bc-session-execution.v1' or receipt.get('verdict') != 'PASS'
            or receipt.get('collector_execution', {}).get('runtime_contract_sha256') != execution['runtime_contract_sha256']
            or result.get('status') != 'complete' or result.get('failure') is not None
            or receipt.get('session_id') != result.get('session_id')
            or receipt.get('native_initializations') != 1 or result.get('native_initializations') != 1
            or host.get('passed') is not True or host.get('exit_code') != 0 or host.get('failure') is not None
            or receipt.get('scene_initializations') != len(result.get('cohorts', []))
            or receipt.get('native_result_raw_sha256') != hashlib.sha256((path / 'bc-session-result.json').read_bytes()).hexdigest()
            or receipt.get('host_result_raw_sha256') != hashlib.sha256((path / 'host-result.json').read_bytes()).hexdigest()):
        raise ValueError('partial, changed or failed prior session cannot be resumed')
    previous = []
    for row in result['cohorts']:
        relative = Path(row['output_dir']).relative_to('/dagger-runtime')
        child = (path / relative / 'cohort-manifest.json').resolve(strict=True)
        if not child.is_relative_to(path):
            raise ValueError('prior cohort escaped its owned runtime')
        child_raw = child.read_bytes()
        doc = json.loads(child_raw)
        cohort_receipt = json.loads(child.with_name('execution-receipt.json').read_bytes())
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if (doc.get('status') != 'complete' or doc.get('optimizer_steps') != 0
                or doc.get('request_sha256') != row['request_sha256']
                or receipt.get('request_identities', {}).get(doc['request_sha256']) != hashlib.sha256(canonical(doc['request'])).hexdigest()
                or cohort_receipt.get('schema') != 'roborender.bc-cohort-execution.v1'
                or cohort_receipt.get('verdict') != 'PASS'
                or cohort_receipt.get('cohort_manifest_raw_sha256') != hashlib.sha256(child_raw).hexdigest()
                or cohort_receipt.get('session_execution_canonical_sha256') != hashlib.sha256(canonical(receipt)).hexdigest()
                or cohort_receipt.get('collector_execution') != receipt['collector_execution']
                or cohort_receipt.get('actual_vector_steps') != doc['vector_steps']
                or cohort_receipt.get('actual_num_envs') != len(doc['request']['motions'])):
            raise ValueError('prior cohort is incomplete')
        previous.append(doc['request'])
    if (not previous or {r['cohort_id'] for r in previous} & {r['cohort_id'] for r in requests}
            or any(r[key] != requests[0][key] for r in previous for key in ('run_id', 'rank', 'sampling_plan_sha256'))):
        raise ValueError('continuation repeats a cohort or changes its collection plan/actor')
    return {'runtime': str(path), 'execution_receipt_raw_sha256': expected_sha,
            'completed_cohorts': [r['cohort_id'] for r in previous], 'native_state_resumed': False}


def main(argv=None, *, qualification_only=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--session-sha256', required=True)
    p.add_argument('--requests-dir', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--guard-sha256', required=True)
    p.add_argument('--host-guard-sha256', required=True)
    p.add_argument('--launcher-sha256', required=True)
    p.add_argument('--evidence-producer-sha256', required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, choices=(0,), default=0)
    p.add_argument('--memory-gib', type=int, choices=(40, 43), default=40)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--qualification-num-envs', type=int, choices=(1, 4, 6), default=6,
                   help='Exact width; B1 tail/B4 fallback need separate fresh requests and approval, never automatic')
    p.add_argument('--qualification-max-windows', type=int, choices=(4, 8), default=4,
                   help='Explicit fresh probe bound; never extends an existing native session')
    p.add_argument('--capacity-receipt', type=Path)
    p.add_argument('--capacity-receipt-sha256')
    p.add_argument('--previous-session', type=Path)
    p.add_argument('--previous-session-execution-sha256')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    host_guard_path = Path(__file__).resolve().with_name('bc_memory_guard.py')
    evidence_path = Path(__file__).resolve().with_name('bc_capacity_evidence.py')
    for path, expected in ((a.session, a.session_sha256), (a.guard, a.guard_sha256),
                           (host_guard_path, a.host_guard_sha256), (evidence_path, a.evidence_producer_sha256),
                           (Path(__file__), a.launcher_sha256)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('immutable launch/guard/session bytes changed')
    if a.guard.resolve() != Path(__file__).resolve().with_name('bc_capacity_guard.py'):
        raise ValueError('launcher and native guard must belong to one immutable qualification bundle')
    source = resolve_source_release(a.source_root, error='source must be a frozen Leo release')
    manifest = json.loads(a.source_manifest.read_bytes())
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical(manifest['files'])).hexdigest() != manifest['source_sha256']:
        raise ValueError('source manifest identity differs')
    for row in manifest['files']:
        path = (source / row['path']).resolve(strict=True)
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('frozen source differs')
    sys.path.insert(0, str(source / 'pipeline/deployment/rtx_actor'))
    sys.path.insert(0, str(source))
    from run_bc_session import read_requests, build_command
    from isaac6_bc_production import publish_json, SCHEMA_V2
    from bc_execution_evidence import collector_identity, publish_completed_session
    from materialize_isaac6 import materialize
    from qualify_native_isaac6 import IMAGE, RAVEN
    from qualify_graphics import cleanup_owned
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib, container_memory_sample, owned_memory_sample
    def native_owners():
        return subprocess.check_output(['docker', 'ps', '--filter', 'ancestor=' + IMAGE,
            '--format', '{{.Names}}'], text=True, timeout=15).splitlines()
    session, requests = read_requests(json.loads(a.session.read_bytes()), a.requests_dir)
    b = validate_requests(requests, qualification_only=qualification_only,
                          qualification_num_envs=a.qualification_num_envs,
                          qualification_max_windows=a.qualification_max_windows)
    if qualification_only and any(x is not None for x in (a.capacity_receipt, a.capacity_receipt_sha256,
                                                          a.previous_session, a.previous_session_execution_sha256)):
        raise ValueError('qualification cannot borrow prior production/session evidence')
    if not qualification_only and (a.capacity_receipt is None or a.capacity_receipt_sha256 is None):
        raise ValueError('guarded production collection requires an exact actual-B capacity receipt')
    runtime = a.runtime.resolve()
    if (runtime.exists() or not runtime.is_relative_to('/home/ldu/roborender/runtime/bc-sessions')
            or not 60 <= a.timeout <= (1800 if qualification_only else 12 * 3600)):
        raise ValueError('fresh scoped runtime and bounded timeout required')

    def command(overlay):
        name, base = build_command(session=session, requests=requests, session_path=a.session.resolve(),
            session_sha256=a.session_sha256, requests_dir=a.requests_dir.resolve(), runtime=runtime,
            overlay=overlay, source=source, physical_gpu=0, memory_gib=40)
        return name, augment_command(base, guard_path=a.guard.resolve(), guard_sha256=a.guard_sha256,
                                     memory_gib=a.memory_gib, image=IMAGE, num_envs=b)

    def execution_and_admission(overlay):
        port_raw = (overlay / 'port-manifest.json').read_bytes()
        execution = collector_identity(manifest, json.loads(port_raw), IMAGE,
            runtime_overlays={'native_pre_save_guard': a.guard_sha256, 'host_runtime_guard': a.host_guard_sha256})
        capacity_binding = None
        if not qualification_only:
            from pipeline.bc_training.production_protocol import bound, validate_bounded_capacity
            capacity_binding = {'path': str(a.capacity_receipt.resolve()), 'raw_sha256': a.capacity_receipt_sha256}
            receipt = bound(capacity_binding)
            if receipt.get('schema') != 'roborender.native-bc-capacity.v2':
                raise ValueError('guarded sessions require an actual repeated-full-buffer v2 capacity receipt')
            model_names = {'privileged_teacher': 'teacher', 'homie_walk': 'homie_walk', 'homie_stand': 'homie_stand'}
            assets = json.loads((source / 'pipeline/dagger/rtx/config/assets.manifest.json').read_bytes())
            models = {model_names[row['role']]: row['sha256'] for row in assets['artifacts'] if row['role'] in model_names}
            if set(models) != {'teacher', 'homie_walk', 'homie_stand'}:
                raise ValueError('actual native teacher/Homie model identities are incomplete')
            protocol = {'collector_execution': execution, 'model_sha256': models,
                'execution_topology': {'host_id': 'leo-rtx-6k', 'concurrent_collectors': 1,
                    'cgroup_limit_bytes_per_collector': a.memory_gib * GIB, 'minimum_host_reserve_bytes': 8 * GIB},
                'capacity_equivalence': {'minimum_full_flushes_per_lane': 2, 'maximum_growth_fraction': .10,
                    'absolute_growth_slack_bytes': 256 << 20}}
            validate_bounded_capacity(receipt, protocol)
            if receipt['num_envs'] != b:
                raise ValueError('capacity receipt does not qualify actual session B, including short tails')
        previous = validate_prior_boundary(a.previous_session, a.previous_session_execution_sha256, requests, execution)
        return port_raw, execution, capacity_binding, previous

    if not a.execute:
        with tempfile.TemporaryDirectory(prefix='bc-capacity-plan-') as temporary:
            overlay = Path(temporary) / 'overlay'
            materialize(RAVEN, overlay, source / 'pipeline/dagger/rtx/config/assets.manifest.json')
            _, execution, capacity_binding, previous = execution_and_admission(overlay)
            name, cmd = command(overlay)
            print(json.dumps({'execute': False, 'name': name, 'maximum_vector_steps': sum(r['max_windows'] * 81 for r in requests),
                'minimum_initial_available_gib': a.memory_gib + 8, 'production_admitted': False,
                'qualification_only': qualification_only, 'collector_execution': execution,
                'capacity_binding': capacity_binding, 'previous_completed_session': previous,
                'command': [s.replace(str(overlay), str(runtime / 'overlay')) for s in cmd]}))
        return
    if socket.gethostname() != 'leo-rtx-6k':
        raise RuntimeError('qualification is Leo-only')
    gpu_gate(0, deployment_host='leo-rtx-6k')
    exclusive_native_owner(native_owners())
    initial = memory_kib()
    admission(initial['MemAvailable'], a.memory_gib)
    # Admission precedes any native process. No cache reclaim or other-job stop.
    runtime.mkdir(parents=True)
    (runtime / 'home').mkdir()
    prepare_output_directories(runtime, requests)
    materialize(RAVEN, runtime / 'overlay', source / 'pipeline/dagger/rtx/config/assets.manifest.json')
    name, cmd = command(runtime / 'overlay')
    if subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('exact container identity already exists; not ours to stop')
    # Recheck immediately before launch, after source/overlay preparation.
    admission(memory_kib()['MemAvailable'], a.memory_gib)
    exclusive_native_owner(native_owners())
    port_raw, execution, capacity_binding, previous = execution_and_admission(runtime / 'overlay')
    publish_json(runtime / 'launch.json', {'command': cmd, 'source_sha256': manifest['source_sha256'],
        'source_manifest_raw_sha256': hashlib.sha256(a.source_manifest.read_bytes()).hexdigest(),
        'port_manifest_raw_sha256': hashlib.sha256(port_raw).hexdigest(),
        'collector_execution': execution,
        'session_canonical_sha256': hashlib.sha256(canonical(session)).hexdigest(),
        'request_identities': {row['sha256']: hashlib.sha256(canonical(request)).hexdigest()
                               for row, request in zip(session['requests'], requests)},
        'guard_sha256': a.guard_sha256, 'launcher_sha256': a.launcher_sha256,
        'host_guard_sha256': a.host_guard_sha256,
        'evidence_producer_sha256': a.evidence_producer_sha256,
        'execution_topology': {'host_id': 'leo-rtx-6k', 'concurrent_collectors': 1,
            'cgroup_limit_bytes_per_collector': a.memory_gib * GIB, 'minimum_host_reserve_bytes': 8 * GIB},
        'qualification_only': qualification_only,
        'qualification_max_windows': a.qualification_max_windows if qualification_only else None,
        'capacity_binding': capacity_binding,
        'previous_completed_session': previous,
        'session_sha256': a.session_sha256, 'production_admitted': False})
    process, failure, launched = None, None, False
    started, min_available, max_peak = time.monotonic(), initial['MemAvailable'], 0
    last_sample = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt('scoped guarded BC stop requested')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (runtime / 'console.log').open('xb') as log, (runtime / 'memory-samples.jsonl').open('x') as samples:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            launched = True
            while process.poll() is None:
                if time.monotonic() - started > a.timeout:
                    raise TimeoutError('bounded guarded BC timeout')
                host = memory_kib()
                min_available = min(min_available, host['MemAvailable'])
                sample = owned_memory_sample(name, process, runtime / 'bc-session-result.json',
                                             completion_field='status', completion_value='complete')
                if sample.get('lifecycle') == 'completed_attested':
                    break
                if not sample.get('pid') and time.monotonic() - started < 15:
                    if host['MemAvailable'] < 8 * 1024**2:
                        raise RuntimeError('host floor violated during Docker startup')
                else:
                    runtime_guard(host, sample)
                    exclusive_native_owner(native_owners(), name)
                    last_sample = sample
                    max_peak = max(max_peak, sample['memory_peak'])
                samples.write(json.dumps({'monotonic_seconds': time.monotonic(), 'container': sample,
                    'simultaneous_native_image_owners': 1 if sample.get('pid') else 0,
                    'host_available_kib': host['MemAvailable']}, sort_keys=True) + '\n')
                samples.flush()
                time.sleep(1)
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if launched:
            cleanup_owned(name)
        if process is not None and process.poll() is None:
            process.wait(timeout=30)
    result_path = runtime / 'bc-session-result.json'
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else {}
    passed = (failure is None and process is not None and process.returncode == 0
        and result.get('status') == 'complete' and result.get('native_initializations') == 1
        and len(result.get('cohorts', [])) == len(requests)
        and all(row['selected_chunks'] == sum(request['chunks_per_env']) for row, request in zip(result['cohorts'], requests)))
    plateau = None
    if passed and qualification_only:
        try:
            from bc_capacity_guard import plateau_gate
            events = [json.loads(line) for line in (runtime / 'capacity-save-events.jsonl').read_text().splitlines()]
            plateau = plateau_gate(events, num_envs=b, cap_bytes=a.memory_gib * GIB)
        except Exception as exc:
            passed = False
            failure = {'type': type(exc).__name__, 'message': str(exc)}
    report = {'passed': passed, 'failure': failure, 'exit_code': process.returncode if process else None,
        'minimum_host_available_kib': min_available, 'sampled_cgroup_peak_bytes': max_peak,
        'last_live_sample': last_sample, 'elapsed_seconds': time.monotonic() - started,
        'production_admitted': False, 'full_b8_qualification': False, 'plateau': plateau}
    publish_json(runtime / 'host-result.json', report)
    if passed:
        report['execution_evidence'] = publish_completed_session(runtime, session, report, publish_json)
        if qualification_only:
            from bc_capacity_evidence import publish_capacity_receipt
            report['capacity_receipt'] = publish_capacity_receipt(runtime, publish_json)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
