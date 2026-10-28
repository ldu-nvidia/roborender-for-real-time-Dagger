"""Serial, metadata-only orchestration of already-qualified BC sessions.

No native runtime, physics, collection or guard implementation is modified.
An interrupted ACTIVE intent is never replayed, even when files look complete.

Future campaigns use v2 storage admission. Historical v1 documents remain
readable for plan-only inspection, but this controller cannot execute them or
adopt their states. Frozen running controllers are not modified by this module.
Storage budgets are conservative declared allocations, NOT empirical compressed
file-size guarantees. Sampling/termination cannot guarantee against an external
writer exhausting the disk between checks; a substantial reserve is mandatory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from roborender.infrastructure.source_locations import LEGACY_RELEASE_ROOT, resolve_source_release

LEGACY_SCHEMA = 'roborender.guarded-bc-campaign.v1'
SCHEMA = 'roborender.guarded-bc-campaign.v2'
STATE_SCHEMA = 'roborender.guarded-bc-campaign-state.v2'
from pipeline.bc_collection.storage_guard import (
    STORAGE_SCHEMA, GIB, MIN_CHUNK_BUDGET, StorageRejected,
    validate_storage_policy, storage_snapshot, observe_storage,
)
MAX_BYTES = 32 << 20
ACTOR = 'pipeline/deployment/rtx_actor'
MODULES = ('bc_memory_guard.py', 'bc_capacity_guard.py', 'bc_capacity_evidence.py',
           'run_guarded_bc_capacity.py', 'run_guarded_bc_session.py', 'bc_execution_evidence.py')


def session_storage_budget(policy, requests):
    validate_storage_policy(policy)
    emissions = 0
    width = 0
    for request in requests:
        windows, b = request.get('max_windows'), len(request.get('motions', []))
        if type(windows) is not int or not 1 <= windows <= 1000000 or not 1 <= b <= 6:
            raise ValueError('storage admission requires each actual B and declared native window ceiling')
        emissions += b * windows
        width = max(width, b)
    if not emissions:
        raise ValueError('storage admission requires a nonempty session')
    # Each full81-row original writer call emits at most one chunk per lane,
    # including any carried partial tail. Count ALL emissions, not just selected.
    in_flight = width * policy['bytes_per_emission_budget'] + policy['session_metadata_budget_bytes']
    if policy['reserve_bytes'] < in_flight:
        raise ValueError('storage reserve does not cover a full in-flight writer batch and metadata')
    budget = emissions * policy['bytes_per_emission_budget'] + policy['session_metadata_budget_bytes']
    return dict(maximum_emissions=emissions, budget_bytes=budget,
                required_available_bytes=budget + policy['reserve_bytes'],
                budget_is_guaranteed_payload_bound=False,
                basis='declared maximum B*windows, not observed compressed file sizes')


def wait_guarded_storage(process, *, timeout, policy, snapshot, report, clock):
    deadline = clock() + timeout
    while True:
        observe_storage(snapshot(), policy, report)
        remaining = deadline - clock()
        if remaining <= 0:
            raise subprocess.TimeoutExpired('owned guarded BC session', timeout)
        try:
            code = process.wait(timeout=min(policy['poll_seconds'], remaining))
        except subprocess.TimeoutExpired:
            continue
        observe_storage(snapshot(), policy, report)
        return code


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_raw(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('metadata must be a bounded regular file: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('metadata grew beyond bound')
    return raw


def bound(binding):
    if (set(binding) != {'path', 'raw_sha256'} or not Path(binding['path']).is_absolute()
            or not re.fullmatch(r'[0-9a-f]{64}', binding['raw_sha256'])):
        raise ValueError('exact absolute metadata binding required')
    raw = read_raw(binding['path'])
    if digest(raw) != binding['raw_sha256']:
        raise ValueError('immutable metadata changed: ' + binding['path'])
    return json.loads(raw)


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), raw_sha256=digest(read_raw(path)))


def publish(path, value, *, replace=False):
    path = Path(path)
    raw = canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.campaign-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            if path.is_symlink():
                raise ValueError('state target may not be a symlink')
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_campaign(doc, *, source_prefix='/home/ldu/roborender/runtime/source-releases',
                      runtime_prefix='/home/ldu/roborender/runtime/bc-sessions'):
    required = {'schema', 'campaign_id', 'source_root', 'source_manifest', 'physical_gpu',
                'memory_gib', 'sessions'}
    if doc.get('schema') == SCHEMA:
        required.add('storage_guard')
    if (set(doc) != required or doc['schema'] not in (SCHEMA, LEGACY_SCHEMA)
            or not re.fullmatch(r'bc-campaign-[a-z0-9-]{1,60}', doc['campaign_id'])
            or doc['physical_gpu'] != 0 or type(doc['physical_gpu']) is not int
            or doc['memory_gib'] not in (40, 43) or type(doc['memory_gib']) is not int
            or not isinstance(doc['sessions'], list) or not 1 <= len(doc['sessions']) <= 300):
        raise ValueError('invalid single-collector bounded campaign')
    policy = validate_storage_policy(doc['storage_guard']) if doc['schema'] == SCHEMA else None
    source = Path(doc['source_root']).resolve(strict=True)
    if Path(source_prefix) == LEGACY_RELEASE_ROOT:
        source = resolve_source_release(source, error='campaign source is not an owned immutable release')
    elif not source.is_relative_to(source_prefix):
        raise ValueError('campaign source is not an owned immutable release')
    manifest = bound(doc['source_manifest'])
    if (manifest.get('schema') != 'roborender.source-snapshot.v1'
            or digest(canonical(manifest['files'])) != manifest['source_sha256']):
        raise ValueError('source release manifest identity differs')
    files = {row['path']: row['sha256'] for row in manifest['files']}
    for name in MODULES:
        path = source / ACTOR / name
        if digest(read_raw(path)) != files[f'{ACTOR}/{name}']:
            raise ValueError('executed host/guard/evidence source changed')
    seen_sessions, seen_runtime, seen_cohorts = set(), set(), set()
    requests_by_session = []
    for row in doc['sessions']:
        if set(row) != {'session_id', 'session', 'requests_dir', 'runtime', 'capacity_receipt', 'timeout_seconds'}:
            raise ValueError('unexpected session launch fields')
        session = bound(row['session'])
        runtime = Path(row['runtime'])
        if (session.get('schema') != 'roborender.bc-native-session.v1' or row['session_id'] != session.get('session_id')
                or row['session_id'] in seen_sessions or str(runtime) in seen_runtime
                or not runtime.is_absolute() or '..' in runtime.parts or not runtime.is_relative_to(runtime_prefix)
                or runtime == Path(runtime_prefix) or runtime.is_symlink()
                or not 60 <= row['timeout_seconds'] <= 12 * 3600 or type(row['timeout_seconds']) is not int
                or len(session.get('requests', [])) != 1):
            raise ValueError('session identity, runtime or bounds differ')
        seen_sessions.add(row['session_id']); seen_runtime.add(str(runtime))
        request_root = Path(row['requests_dir']).resolve(strict=True)
        requests = []
        for request_binding in session['requests']:
            path = Path(request_binding['path'])
            if path.parent != Path('/bc-session-requests') or path.name in {'.', '..'}:
                raise ValueError('invalid request mount path')
            raw = read_raw(request_root / path.name)
            if digest(raw) != request_binding['sha256']:
                raise ValueError('request raw file changed')
            request = json.loads(raw)
            if (request.get('schema') != 'roborender.bc-native-collection-request.v2'
                    or request['cohort_id'] in seen_cohorts or request.get('rank') != 0):
                raise ValueError('cohort replay, legacy request or wrong rank')
            seen_cohorts.add(request['cohort_id']); requests.append(request)
        size = len(requests[0]['motions'])
        if not 1 <= size <= 6 or any(len(r['motions']) != size for r in requests):
            raise ValueError('a session must have one exact qualified native width')
        if policy is not None:
            if not runtime.is_relative_to(policy['root']) or runtime == Path(policy['root']):
                raise ValueError('session runtime must lie below the declared storage root')
            session_storage_budget(policy, requests)
        receipt = bound(row['capacity_receipt'])
        if (receipt.get('schema') != 'roborender.native-bc-capacity.v2' or receipt.get('verdict') != 'PASS'
                or receipt.get('native_runtime_tested') is not True or receipt.get('num_envs') != size
                or receipt.get('concurrent_collectors') != 1
                or receipt.get('cgroup_limit_bytes') != doc['memory_gib'] * 2**30):
            raise ValueError('genuine exact-width/capacity evidence required; guarded runner revalidates contract')
        requests_by_session.append(requests)
    all_requests = [r for group in requests_by_session for r in group]
    if any(r[k] != all_requests[0][k] for r in all_requests for k in ('run_id', 'sampling_plan_sha256')):
        raise ValueError('campaign mixes independent sampling plans/runs')
    return manifest, files, requests_by_session


def command(doc, row, files):
    source = Path(doc['source_root']); actor = source / ACTOR
    return ['/home/ldu/roborender/.venv/bin/python', str(actor / 'run_guarded_bc_session.py'),
        '--session', row['session']['path'], '--session-sha256', row['session']['raw_sha256'],
        '--requests-dir', row['requests_dir'], '--source-root', str(source),
        '--source-manifest', doc['source_manifest']['path'], '--guard', str(actor / 'bc_capacity_guard.py'),
        '--guard-sha256', files[f'{ACTOR}/bc_capacity_guard.py'],
        '--host-guard-sha256', files[f'{ACTOR}/bc_memory_guard.py'],
        '--launcher-sha256', files[f'{ACTOR}/run_guarded_bc_capacity.py'],
        '--evidence-producer-sha256', files[f'{ACTOR}/bc_capacity_evidence.py'],
        '--runtime', row['runtime'], '--physical-gpu', '0', '--memory-gib', str(doc['memory_gib']),
        '--timeout', str(row['timeout_seconds']), '--capacity-receipt', row['capacity_receipt']['path'],
        '--capacity-receipt-sha256', row['capacity_receipt']['raw_sha256'], '--execute']


def verify_completed(doc, row, manifest):
    """Recompute the existing small producer receipts, not payload hashes."""
    runtime = Path(row['runtime'])
    paths = {name: runtime / name for name in
             ('launch.json', 'host-result.json', 'bc-session-result.json', 'session-execution.json')}
    raw = {name: read_raw(path) for name, path in paths.items()}
    values = {name: json.loads(data) for name, data in raw.items()}
    launch, host, native = (values[name] for name in ('launch.json', 'host-result.json', 'bc-session-result.json'))
    files = {r['path']: r['sha256'] for r in manifest['files']}
    if (launch.get('qualification_only') is not False
            or launch.get('session_sha256') != row['session']['raw_sha256']
            or launch.get('source_sha256') != manifest['source_sha256']
            or launch.get('source_manifest_raw_sha256') != doc['source_manifest']['raw_sha256']
            or launch.get('capacity_binding') != row['capacity_receipt']
            or launch.get('guard_sha256') != files[f'{ACTOR}/bc_capacity_guard.py']
            or launch.get('host_guard_sha256') != files[f'{ACTOR}/bc_memory_guard.py']
            or launch.get('launcher_sha256') != files[f'{ACTOR}/run_guarded_bc_capacity.py']
            or launch.get('evidence_producer_sha256') != files[f'{ACTOR}/bc_capacity_evidence.py']
            or launch.get('execution_topology') != dict(host_id='leo-rtx-6k', concurrent_collectors=1,
                cgroup_limit_bytes_per_collector=doc['memory_gib'] * 2**30, minimum_host_reserve_bytes=8 * 2**30)):
        raise ValueError('completed launch does not match this exact campaign intent')
    envelopes, cohort_receipts = [], []
    for item in native.get('cohorts', []):
        relative = Path(item['output_dir']).relative_to('/dagger-runtime')
        directory = (runtime / relative).resolve(strict=True)
        if not directory.is_relative_to(runtime.resolve(strict=True)):
            raise ValueError('cohort completion escaped owned runtime')
        cohort_raw = read_raw(directory / 'cohort-manifest.json')
        envelopes.append(dict(document=json.loads(cohort_raw), raw_sha256=digest(cohort_raw)))
        cohort_receipts.append(json.loads(read_raw(directory / 'execution-receipt.json')))
    module_path = Path(doc['source_root']) / ACTOR / 'bc_execution_evidence.py'
    spec = importlib.util.spec_from_file_location('campaign_frozen_execution_producer', module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected_session, expected_cohorts = module.build_session_receipts(launch=launch,
        launch_raw_sha256=digest(raw['launch.json']), session=bound(row['session']), result=native,
        result_raw_sha256=digest(raw['bc-session-result.json']), host_result=host,
        host_result_raw_sha256=digest(raw['host-result.json']), cohort_envelopes=envelopes)
    if expected_session != values['session-execution.json'] or expected_cohorts != cohort_receipts:
        raise ValueError('completed native receipts do not reconstruct exactly')
    return dict(session_id=row['session_id'], runtime=row['runtime'],
        session_execution=binding(paths['session-execution.json']),
        host_result=binding(paths['host-result.json']), native_result=binding(paths['bc-session-result.json']))


def run(doc, campaign_binding, state_dir, *, max_sessions, wall_budget, execute=False,
        _popen=subprocess.Popen, _clock=time.monotonic, _verify=verify_completed,
        _validate=validate_campaign, _hostname=socket.gethostname,
        _storage_snapshot=storage_snapshot,
        _state_prefix='/home/ldu/roborender/runtime/bc-campaigns'):
    if bound(campaign_binding) != doc:
        raise ValueError('campaign document differs from its pinned raw file')
    manifest, files, requests_by_session = _validate(doc)
    if type(max_sessions) is not int or not 1 <= max_sessions <= 300 or not 60 <= wall_budget <= 48 * 3600:
        raise ValueError('explicit bounded session count/wall budget required')
    commands = [command(doc, row, files) for row in doc['sessions']]
    if not execute:
        return dict(execute=False, sessions=len(commands), commands=commands, production_admitted=False,
            execution_supported=doc['schema'] == SCHEMA,
            storage_budgets=[session_storage_budget(doc['storage_guard'], r) for r in requests_by_session]
                if doc['schema'] == SCHEMA else None)
    if doc['schema'] != SCHEMA:
        raise ValueError('historical campaign v1 is read-only; future execution requires explicit v2 storage admission')
    policy = doc['storage_guard']
    if _hostname() != 'leo-rtx-6k':
        raise ValueError('native BC campaign execution is Leo-only')
    state_dir = Path(state_dir)
    if (not state_dir.is_absolute() or '..' in state_dir.parts
            or not state_dir.is_relative_to(_state_prefix) or state_dir == Path(_state_prefix)
            or state_dir.is_symlink() or state_dir.resolve(strict=False) != state_dir
            or not state_dir.is_relative_to(policy['root']) or state_dir == Path(policy['root'])):
        raise ValueError('campaign state must be a scoped owned non-symlink directory')
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = state_dir / 'state.json'
        if state_path.exists():
            state = json.loads(read_raw(state_path))
            if (state.get('schema') != STATE_SCHEMA or state.get('campaign') != campaign_binding
                    or state.get('controller_sha256') != digest(read_raw(__file__))
                    or state.get('phase') not in {'READY', 'COMPLETE'}):
                raise ValueError('changed or ACTIVE/UNKNOWN/BLOCKED campaign cannot be replayed')
        else:
            state = dict(schema=STATE_SCHEMA, campaign=campaign_binding,
                controller_sha256=digest(read_raw(__file__)), phase='READY', next_index=0, completed=[])
            publish(state_path, state)
        if state['next_index'] != len(state['completed']) or not 0 <= state['next_index'] <= len(commands):
            raise ValueError('completion prefix is inconsistent')
        if (state['phase'] == 'COMPLETE') != (state['next_index'] == len(commands)):
            raise ValueError('completion phase differs from the exact completed prefix')
        for index, saved in enumerate(state['completed']):
            intent = bound(saved['intent'])
            child = bound(saved['child'])
            exit_record = bound(saved['exit'])
            if (intent.get('schema') != 'roborender.bc-campaign-intent.v1'
                    or child.get('schema') != 'roborender.bc-campaign-child.v1' or child.get('observer_owned') is not True
                    or exit_record.get('schema') != 'roborender.bc-campaign-exit.v1'
                    or exit_record.get('pid') != child.get('pid')
                    or intent.get('campaign') != campaign_binding or intent.get('index') != index
                    or intent.get('command') != commands[index] or intent.get('session_id') != doc['sessions'][index]['session_id']
                    or child.get('intent_raw_sha256') != saved['intent']['raw_sha256']
                    or exit_record.get('child_raw_sha256') != saved['child']['raw_sha256']
                    or exit_record.get('returncode') != 0
                    or bound(saved['storage']).get('passed') is not True
                    or bound(saved['storage']).get('policy') != policy
                    or bound(saved['storage']).get('filesystem') != state.get('storage_filesystem')
                    or exit_record.get('storage_raw_sha256') != saved['storage']['raw_sha256']
                    or _verify(doc, doc['sessions'][index], manifest) != saved['completion']):
                raise ValueError('completed-session receipt chain changed; do not resume')
        started, launched = _clock(), 0
        while state['next_index'] < len(commands) and launched < max_sessions:
            index = state['next_index']; row = doc['sessions'][index]
            # Reserve the native runner's whole timeout plus cleanup; never begin
            # a session we already know exceeds this observer's bounded budget.
            if _clock() - started + row['timeout_seconds'] + 60 > wall_budget:
                print(json.dumps(dict(phase='bc_campaign_budget_boundary', next_session=row['session_id'],
                                      completed_sessions=state['next_index'])), flush=True)
                break
            if Path(row['runtime']).exists():
                raise ValueError('next runtime already exists; partial native work is never replayed')
            budget = session_storage_budget(policy, requests_by_session[index])
            storage_report = dict(schema='roborender.bc-campaign-storage-observation.v1',
                policy=policy, budget=budget, session_id=row['session_id'], passed=False,
                samples=0, guarantee_against_external_writers=False)
            available = observe_storage(_storage_snapshot(policy, row['runtime'], state_dir), policy, storage_report)
            if ('storage_filesystem' in state
                    and state['storage_filesystem'] != storage_report['filesystem']):
                raise StorageRejected('guarded filesystem identity changed between campaign sessions')
            if available < budget['required_available_bytes']:
                raise StorageRejected('insufficient free storage for the next declared maximum session budget plus reserve')
            prefix = state_dir / f'session-{index:04d}'
            intent_path = prefix.with_suffix('.intent.json')
            publish(intent_path, dict(schema='roborender.bc-campaign-intent.v1', campaign=campaign_binding,
                index=index, session_id=row['session_id'], command=commands[index], source_sha256=manifest['source_sha256'],
                storage_admission=dict(budget=budget, available_bytes=available, filesystem=storage_report['filesystem'])))
            state.update(phase='ACTIVE', active_intent=binding(intent_path),
                         storage_filesystem=storage_report['filesystem'])
            publish(state_path, state, replace=True)
            print(json.dumps(dict(phase='bc_campaign_session_start', index=index,
                                  session_id=row['session_id'], runtime=row['runtime'])), flush=True)
            process = None
            try:
                with prefix.with_suffix('.console.log').open('xb') as log:
                    process = _popen(commands[index], stdout=log, stderr=subprocess.STDOUT)
                    child_path = prefix.with_suffix('.child.json')
                    publish(child_path, dict(schema='roborender.bc-campaign-child.v1', pid=process.pid,
                        intent_raw_sha256=state['active_intent']['raw_sha256'], observer_owned=True))
                    code = wait_guarded_storage(process, timeout=row['timeout_seconds'] + 60,
                        policy=policy, snapshot=lambda: _storage_snapshot(policy, row['runtime'], state_dir),
                        report=storage_report, clock=_clock)
                storage_report['passed'] = code == 0
                storage_path = prefix.with_suffix('.storage.json')
                publish(storage_path, storage_report)
                exit_path = prefix.with_suffix('.exit.json')
                publish(exit_path, dict(schema='roborender.bc-campaign-exit.v1', pid=process.pid,
                    child_raw_sha256=binding(child_path)['raw_sha256'], returncode=code,
                    storage_raw_sha256=binding(storage_path)['raw_sha256']))
                if code != 0:
                    raise RuntimeError('guarded child failed; native session is not replayable')
                completion = _verify(doc, row, manifest)
                saved = dict(intent=state.pop('active_intent'), child=binding(child_path),
                    exit=binding(exit_path), storage=binding(storage_path), completion=completion)
                state['completed'].append(saved)
                state['next_index'] += 1; launched += 1
                state['phase'] = 'COMPLETE' if state['next_index'] == len(commands) else 'READY'
                publish(state_path, state, replace=True)
                print(json.dumps(dict(phase='bc_campaign_session_complete', session_id=row['session_id'],
                                      completed_sessions=state['next_index'], total_sessions=len(commands))), flush=True)
            except BaseException as exc:
                # Terminate only our just-spawned host runner. Its SIGTERM handler
                # owns cleanup. If it will not exit, leave BLOCKED, never replay.
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass
                storage_path = prefix.with_suffix('.storage-failure.json')
                storage_report.update(passed=False, failure=dict(type=type(exc).__name__, message=str(exc)),
                                      owned_child_still_running=process is not None and process.poll() is None)
                publish(storage_path, storage_report)
                state.update(phase='BLOCKED', failure=dict(type=type(exc).__name__, message=str(exc)))
                publish(state_path, state, replace=True)
                print(json.dumps(dict(phase='bc_campaign_blocked', session_id=row['session_id'],
                                      error_type=type(exc).__name__, replay_allowed=False)), flush=True)
                raise
        return dict(phase=state['phase'], completed_sessions=state['next_index'],
            total_sessions=len(commands), launched_this_invocation=launched, production_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--campaign-sha256', required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--max-sessions', type=int, default=1)
    parser.add_argument('--wall-budget', type=int, default=14400)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    ref = dict(path=str(args.campaign.resolve(strict=True)), raw_sha256=args.campaign_sha256)
    doc = bound(ref)
    if args.execute:
        signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('controller stop requested')))
    print(json.dumps(run(doc, ref, args.state_dir, max_sessions=args.max_sessions,
                         wall_budget=args.wall_budget, execute=args.execute), sort_keys=True))


if __name__ == '__main__':
    main()
