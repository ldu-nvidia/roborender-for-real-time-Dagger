"""Guarded single-scene Raven-compatible BC collection, never an implicit launch.

Default invocation is CPU-only staging. --execute must run from the pinned Leo
release. --qualification executes only a bounded 4/8-window full-buffer probe;
production execution additionally requires genuine current-source capacity and
natural-drain receipts. Both retain every native complete chunk and never replay
a runtime. The explicit B32/10000 protocol uses per-window disk admission, not
a claim that the entire campaign is guaranteed to fit on disk.
"""
from __future__ import annotations

import argparse
import fcntl
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.bc_collection.storage_guard import (
    STORAGE_SCHEMA, validate_storage_policy, storage_snapshot, observe_storage,
)
from pipeline.deployment.rtx_actor.isaac6_bc_production import canonical, publish_json, validate_collection_request
from pipeline.deployment.source_snapshot import file_record
from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL as BOUNDED_STORAGE_PROTOCOL
from pipeline.deployment.rtx_actor.bc_runtime import load_paths

GIB = 2**30
SCHEMA = 'roborender.bc-continuous-execution.v1'
CAPACITY_SCHEMA = 'roborender.native-bc-continuous-capacity.v1'
REQUEST_SCHEMA = 'roborender.bc-native-continuous-request.v1'
ACTOR = 'pipeline/deployment/rtx_actor'
SELF = 'pipeline/bc_collection/run_continuous.py'
REQUIRED_SOURCE = {SELF, 'pipeline/bc_collection/continuous_plan.py',
    'pipeline/bc_collection/episode_drain.py',
    'pipeline/bc_collection/streaming_storage.py',
    'pipeline/bc_collection/production_gate.py',
    'pipeline/bc_collection/collector.py', 'pipeline/bc_collection/chunk_writer.py',
    'pipeline/bc_collection/continuous_ingest.py', 'pipeline/bc_collection/storage_guard.py',
    'roborender/infrastructure/source_locations.py', 'roborender/infrastructure/paths.py',
    'configs/storage_paths.json', 'pipeline/deployment/source_snapshot.py',
    'pipeline/dagger/rtx/config/assets.manifest.json'} | {f'{ACTOR}/{name}.py' for name in (
        'continuous_bc_guard', 'bounded_bc_guard', 'bounded_bc_storage', 'isaac6_bc_production',
        'bc_resource_observer',
        'bc_collection_rng', 'bc_runtime', 'native_motion_assignment',
        'initial_reset_contract', 'materialize_isaac6', 'roborender_isaac6', 'qualify_native_isaac6',
        'isaac6_bc_session', 'bc_capacity_guard', 'bc_memory_guard', 'qualify_graphics',
        'run_success_diagnostic', 'run_throughput_pair', 'leo_binding', 'native_thp_policy',
        'isaac6_bc_trainer', 'isaac6_capture_trainer')}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError('bounded regular metadata required')
    raw = path.read_bytes()
    if expected is not None and (not re.fullmatch('[0-9a-f]{64}', expected) or digest(raw) != expected):
        raise ValueError('immutable metadata bytes changed')
    return json.loads(raw), digest(raw)


def verify_source(source, manifest):
    rows = manifest.get('files')
    if (manifest.get('schema') != 'roborender.source-snapshot.v1' or not isinstance(rows, list)
            or manifest.get('file_count') != len(rows) or digest(canonical(rows)) != manifest.get('source_sha256')):
        raise ValueError('source manifest identity differs')
    names = [row['path'] for row in rows]
    if names != sorted(set(names)) or not REQUIRED_SOURCE <= set(names):
        raise ValueError('continuous launcher/native/guard/plan/ingest source missing from snapshot')
    for row in rows:
        if file_record(source, row['path']) != row:
            raise ValueError('frozen continuous source bytes or permissions differ: ' + row['path'])
    return {row['path']: row['sha256'] for row in rows}


def validate_launch(request, *, qualification, expected_num_envs, physical_gpu, memory_gib, timeout):
    from pipeline.bc_collection.episode_drain import is_bounded_qualification
    from pipeline.bc_collection.production_gate import is_production
    request = validate_collection_request(request)
    production = is_production(request)
    if production and qualification:
        raise ValueError('production cannot be relabeled as qualification')
    smoke = 'diagnostic_stop' in request
    if request.get('collection_storage_protocol') is not None and not qualification and not production:
        raise ValueError('bounded storage is qualification-only until separately live certified')
    if (request['schema'] != REQUEST_SCHEMA or request['split'] != 'train'
            or request['rank'] != 0 or request['global_env_offset'] != 0
            or request['output_dir'] != '/dagger-runtime/bc_isaac'
            or type(expected_num_envs) is not int or not 1 <= expected_num_envs <= (50 if smoke else 40)
            or len(request['motions']) != expected_num_envs
            or type(physical_gpu) is not int or physical_gpu not in (0, 1)
            or type(memory_gib) is not int or memory_gib not in (40, 43)
            or type(timeout) is not int or not 60 <= timeout <= (7 * 86400 if production else
                3600 if qualification and not smoke else 12 * 3600)
            or (qualification and request['max_windows'] not in (4, 8) and not smoke
                and not is_bounded_qualification(request))):
        raise ValueError('single logical scene, exact B, full-buffer cap or finite launch budget differs')
    return request


def storage_budget(policy, request):
    validate_storage_policy(policy)
    b, windows = len(request['motions']), request['max_windows']
    reserve_needed = b * policy['bytes_per_emission_budget'] + policy['session_metadata_budget_bytes']
    if policy['reserve_bytes'] < reserve_needed:
        raise ValueError('reserve must cover one full writer flush and metadata')
    from pipeline.bc_collection.production_gate import is_production
    if is_production(request):
        from pipeline.bc_collection.streaming_storage import policy as streaming_policy
        rolling = streaming_policy(b)
        if (request.get('streaming_storage') != rolling
                or policy['reserve_bytes'] != rolling['reserve_bytes']
                or policy['bytes_per_emission_budget'] != rolling['bytes_per_emission_ceiling']
                or policy['session_metadata_budget_bytes'] != rolling['metadata_bytes']):
            raise ValueError('production rolling disk policy differs from native guarded writes')
        return dict(maximum_emissions=b * windows, budget_bytes=None,
            required_available_bytes=rolling['window_admission_bytes'],
            budget_is_guaranteed_payload_bound=False, admission='per_window_and_actual_archive_bound',
            whole_campaign_fit_guaranteed=False, shortage_is_success=False)
    budget = b * windows * policy['bytes_per_emission_budget'] + policy['session_metadata_budget_bytes']
    return dict(maximum_emissions=b * windows, budget_bytes=budget,
                required_available_bytes=budget + policy['reserve_bytes'],
                budget_is_guaranteed_payload_bound=False)


def capacity_contract(execution, request, *, gpu_uuid, memory_gib):
    # Run/output IDs, seed and budget are deliberately excluded. Capacity is
    # source/recipe/width/geometry/hardware bound, not one request instance.
    return dict(collector_execution=execution, num_envs=len(request['motions']),
        motion_inventory_sha256=digest(canonical(request['motion_inventory'])),
        host_id='leo-rtx-6k', gpu_uuid=gpu_uuid, concurrent_collectors=1,
        logical_scenes=1, cgroup_limit_bytes=memory_gib * GIB,
        minimum_host_reserve_bytes=8 * GIB,
        recipe=request.get('collection_storage_protocol',
                           'original_full_bc_buffers_teacher_only_random_clocks_continuous_v1'))


def gpu_sample(gpu_uuid):
    raw = subprocess.check_output(['nvidia-smi', '-i', gpu_uuid,
        '--query-gpu=uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
    fields = [field.strip() for field in raw.strip().split(',')]
    if len(fields) != 3 or fields[0] != gpu_uuid:
        raise RuntimeError('exact collector GPU identity/counters unavailable')
    result = dict(uuid=fields[0], total_bytes=int(fields[1]) * 2**20, used_bytes=int(fields[2]) * 2**20)
    validate_gpu_sample(result, gpu_uuid)
    return result


def validate_gpu_sample(value, gpu_uuid):
    total, used = value.get('total_bytes'), value.get('used_bytes')
    if (value.get('uuid') != gpu_uuid or type(total) is not int or type(used) is not int
            or not 0 <= used < total or total - used < max(8 * GIB, .15 * total)):
        raise RuntimeError('collector GPU headroom below8GiB/15percent or counters unavailable')


def observe_resources(stream, *, host, sample, guard, gpu_reader, owner_check,
                      startup=False, recovered_limit_cap_bytes=None):
    """Persist the rejecting sample before unwinding into exact-owner cleanup.

    Guard/telemetry failures must not discard the most important observation.
    fsync failure also fails closed; no workload recovery or replay occurs here.
    """
    row = dict(monotonic_seconds=time.monotonic(), container=sample,
               simultaneous_native_image_owners=1 if sample.get('pid') else 0,
               host_available_kib=host.get('MemAvailable'))
    try:
        if startup and not sample.get('pid'):
            if type(host.get('MemAvailable')) is not int or host['MemAvailable'] < 8 * 1024**2:
                raise RuntimeError('host memory floor violated during startup')
            row['memory_guard'] = dict(status='STARTING')
        else:
            row['memory_guard'] = guard(host, sample, recovered_limit_cap_bytes=recovered_limit_cap_bytes)
            owner_check()
        row['gpu'] = gpu_reader()
    except BaseException as exc:
        row['observation_failure'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row


def validate_capacity(receipt, contract):
    if (receipt.get('schema') != CAPACITY_SCHEMA or receipt.get('verdict') != 'PASS'
            or receipt.get('native_runtime_tested') is not True or receipt.get('contract') != contract
            or receipt.get('bounded_repeated_buffer_only') is not True
            or receipt.get('production_admitted') is not False
            or receipt.get('long_campaign_execution_proven') is not False
            or receipt.get('minimum_full_emissions_per_lane', 0) < 2
            or receipt.get('accepted_windows') not in (4, 8)
            or receipt.get('oom_events') != 0 or receipt.get('swap_bytes_peak') != 0
            or type(receipt.get('evidence')) is not dict
            or not all(re.fullmatch('[0-9a-f]{64}', str(receipt['evidence'].get(key, '')))
                       for key in ('launch', 'host', 'native', 'save_events', 'memory_samples', 'execution'))):
        raise ValueError('actual current-source continuous full-buffer capacity receipt required')
    return receipt


def build_command(*, request, request_path, request_sha256, source, runtime, overlay,
                  physical_gpu, memory_gib, guard_sha256, base_builder=None, image_id=None,
                  runtime_paths=None):
    from pipeline.deployment.rtx_actor.bc_runtime import build_command as native_command
    if runtime_paths is None and base_builder is None:
        runtime_paths = load_paths(source)
    bounded = request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL
    if bounded:
        from pipeline.deployment.rtx_actor.bounded_bc_guard import ACK
        from pipeline.bc_collection.production_gate import is_production
        if is_production(request):
            from pipeline.deployment.rtx_actor.bounded_bc_guard import PRODUCTION_ACK as ACK
    else:
        from pipeline.deployment.rtx_actor.continuous_bc_guard import ACK
    name = 'bc-cohort-leo-cont-' + digest(request['run_id'].encode())[:24]
    cmd = native_command(request=request, request_path=request_path, request_sha256=request_sha256,
        name=name, runtime=runtime, overlay=overlay, source=source, physical_gpu=physical_gpu,
        memory_gib=40, base_builder=base_builder, image_id=image_id, runtime_paths=runtime_paths)
    if image_id is None:
        from qualify_native_isaac6 import IMAGE
        image_id = IMAGE
    for flag in ('--memory', '--memory-swap'):
        if cmd.count(flag) != 1 or cmd[cmd.index(flag) + 1] != '40g':
            raise ValueError('base native memory/swap binding differs')
        cmd[cmd.index(flag) + 1] = f'{memory_gib}g'
    # Persist the zero-swap property in this exact systemd scope. Equal Docker
    # flags alone can leave MemorySwapMax=infinity in systemd; a daemon reload
    # can then replace the kernel's initial zero. Never mutate global host policy.
    swap_annotation = 'org.systemd.property.MemorySwapMax=uint64 0'
    inherited = [item for item in cmd if item.startswith('org.systemd.property.MemorySwapMax=')]
    if inherited and inherited != [swap_annotation]:
        raise ValueError('unexpected inherited systemd no-swap property')
    if not inherited:
        cmd[cmd.index(image_id):cmd.index(image_id)] = ['--annotation', swap_annotation]
    targets = [i for i, value in enumerate(cmd) if value.startswith('trainer._target_=')]
    if len(targets) != 1 or cmd[targets[0]] != 'trainer._target_=isaac6_bc_production.Isaac6BCProductionTrainer':
        raise ValueError('original collection adapter is not the sole trainer')
    cmd[targets[0]] = ('trainer._target_=bounded_bc_guard.BoundedContinuousBCTrainer' if bounded
                      else 'trainer._target_=continuous_bc_guard.GuardedContinuousBCTrainer')
    at = cmd.index(image_id)
    # Import only this stage from the pinned source; never the editable checkout.
    cmd[at:at] = ['--mount',
        f'type=bind,src={Path(source) / "pipeline/bc_collection"},dst=/bc-stage/pipeline/bc_collection,readonly']
    for index, value in enumerate(cmd):
        if value.startswith('PYTHONPATH='):
            cmd[index] = value.replace('PYTHONPATH=', 'PYTHONPATH=/bc-stage:', 1)
    at = cmd.index(image_id)
    cmd[at:at] = ['--cidfile', str(Path(runtime) / 'container.cid'),
        '--mount', 'type=bind,src=/proc/meminfo,dst=/run/host-meminfo,readonly',
        '--env', 'ROBORENDER_CONTINUOUS_BC_ACK=' + ACK,
        '--env', 'ROBORENDER_CONTINUOUS_BC_GUARD_SHA256=' + guard_sha256,
        '--env', 'ROBORENDER_CONTINUOUS_BC_MEMORY_BYTES=' + str(memory_gib * GIB)]
    return name, cmd


def cleanup_exact(name, cid_path, image_id):
    """Stop only a Docker-accepted exact local workload ID, never an allocation."""
    from pipeline.deployment.rtx_actor.qualify_graphics import LABEL
    cid_path = Path(cid_path)
    if cid_path.is_symlink() or not cid_path.is_file():
        raise RuntimeError('Docker acceptance is unknown: no exact cidfile; no blind cleanup')
    cid = cid_path.read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}', cid):
        raise RuntimeError('Docker acceptance cidfile is malformed; no cleanup')
    live = subprocess.check_output(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id=' + cid],
                                   text=True, timeout=15).splitlines()
    if not live:
        return cid  # Exact local Docker inventory confirms --rm completed.
    if live != [cid]:
        raise RuntimeError('exact Docker identity inventory changed; no cleanup')
    info = json.loads(subprocess.check_output(['docker', 'inspect', '--type', 'container', cid],
                                              text=True, timeout=15))[0]
    if (info.get('Id') != cid or info.get('Name') != '/' + name or info.get('Image') != image_id
            or info.get('Config', {}).get('Labels', {}).get(LABEL) != name):
        raise RuntimeError('exact Docker owner/image identity changed; no cleanup')
    if info['State']['Running']:
        subprocess.run(['docker', 'stop', '--time', '10', cid], check=True, timeout=25)
    return cid


def validate_completion(native, request, request_sha256):
    if request.get('collection_loop_protocol') is not None:
        from pipeline.bc_collection.collector import implementation_receipt
        if native.get('collection_implementation') != implementation_receipt():
            raise ValueError('standalone collector/writer source receipt differs')
    b, windows = len(request['motions']), request['max_windows']
    bounded = request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL
    target = request.get('diagnostic_stop', {}).get('chunks_per_lane')
    drain = request.get('episode_drain')
    if drain is not None:
        from pipeline.bc_collection.episode_drain import validate_completion as validate_drain
        actual = native.get('accepted_windows')
        if type(actual) is not int or not 1 <= actual <= windows:
            raise ValueError('total-target drain did not finish within safety ceiling')
        validate_drain(native.get('episode_drain'), num_envs=b, target_chunks=drain['target_chunks'],
            windows=actual, chunks=native.get('emitted_chunks'),
            unrecorded_rows=native.get('unrecorded_rows_per_lane'),
            tail_rows=native.get('remaining_unsealed_tail_rows_per_lane'))
        disk = native.get('streaming_storage', {})
        if (any(disk.get(k) != v for k, v in request['streaming_storage'].items())
                or disk.get('output_root') != request['output_dir']
                or disk.get('windows_admitted') != actual
                or disk.get('emissions_verified') != native.get('emitted_chunks')
                or type(disk.get('minimum_available_bytes')) is not int
                or disk['minimum_available_bytes'] < request['streaming_storage']['reserve_bytes']):
            raise ValueError('streaming disk window/archive/reserve evidence differs')
        windows = actual
    if target is not None:
        actual = native.get('accepted_windows')
        if (type(actual) is not int or not 1 <= actual <= windows
                or len(native.get('emitted_per_lane', [])) != b
                or any(type(x) is not int or x < target for x in native['emitted_per_lane'])):
            raise ValueError('diagnostic chunk target not reached on every lane within safety budget')
        windows = actual
    if (native.get('schema') != 'roborender.bc-native-continuous-manifest.v1'
            or native.get('status') != 'complete' or native.get('request') != request
            or native.get('request_sha256') != request_sha256
            or native.get('completion_reason') != ('total_chunks_and_episodes_drained' if drain is not None else
                'all_lanes_chunk_target' if target is not None else 'vector_window_budget')
            or native.get('accepted_windows') != windows or native.get('vector_steps') != windows * 81
            or native.get('initial_reset_all_calls') != 1 or native.get('optimizer_steps') != 0
            or native.get('teacher_only') is not True
            or native.get('native_payload_writer_unchanged') is not (not bounded)
            or native.get('init_at_random_ep_len') is not True
            or len(native.get('emitted_per_lane', [])) != b
            or any(type(x) is not int or x < 0 for x in native['emitted_per_lane'])
            or native.get('emitted_chunks') != sum(native['emitted_per_lane'])
            or native.get('selected_chunks') != sum(native['emitted_per_lane'])
            or native.get('overcollection_chunks') != 0):
        raise ValueError('continuous collection lacks exact completed windows/all-emission/reset evidence')
    if bounded:
        storage = native.get('collection_storage', {})
        if (storage.get('protocol') != BOUNDED_STORAGE_PROTOCOL
                or storage.get('completed_windows') != windows or storage.get('num_envs') != b
                or storage.get('frames_per_window') != 81 or storage.get('logical_scenes') != 1
                or storage.get('cpu_conversion_lanes') != 1 or storage.get('capture_device') != 'cuda:0'
                or storage.get('production_admitted') is not False
                or not re.fullmatch('[0-9a-f]{64}', str(storage.get('writer_sha256', '')))):
            raise ValueError('bounded collection lacks explicit actual storage/writer evidence')
    reset = native.get('initial_state_attestation', {}).get('initial_reset_completion', {})
    if reset != dict(schema='roborender.initial-reset-completion.v1', generation=1, num_envs=b,
            boundary='after_reset_all_writes_and_observations', cleared_lanes=b, clear_calls=1,
            native_steps_added=0, ordinary_done_resets_modified=False):
        raise ValueError('corrected once-only initial robot/object reset is not attested')
    return native


def build_capacity(*, launch, host, native, save_events, memory_samples, execution_receipt, evidence):
    from pipeline.deployment.rtx_actor.bc_capacity_guard import plateau_gate
    contract = launch['capacity_contract']
    request = launch['request']
    validate_completion(native, request, launch['request_sha256'])
    if (launch.get('qualification_only') is not True or host.get('passed') is not True
            or host.get('exit_code') != 0 or host.get('failure') is not None
            or execution_receipt.get('verdict') != 'PASS'
            or request['max_windows'] not in (4, 8) or 'episode_drain' in request):
        raise ValueError('only an actually completed bounded qualification can produce capacity')
    if (execution_receipt.get('schema') != SCHEMA or execution_receipt.get('native_runtime_tested') is not True
            or execution_receipt.get('qualification_only') is not True
            or execution_receipt.get('request_sha256') != launch['request_sha256']
            or execution_receipt.get('request_canonical_sha256') != digest(canonical(request))
            or execution_receipt.get('collector_execution') != launch['collector_execution']
            or contract['collector_execution'] != launch['collector_execution']
            or execution_receipt.get('actual_vector_steps') != native['vector_steps']
            or execution_receipt.get('actual_num_envs') != len(request['motions'])
            or execution_receipt.get('initial_reset_all_calls') != 1
            or execution_receipt.get('native_manifest_sha256') != evidence['native']
            or execution_receipt.get('host_result_sha256') != evidence['host']
            or execution_receipt.get('launch_sha256') != evidence['launch']
            or not re.fullmatch('[0-9a-f]{64}', str(host.get('container_id', '')))
            or execution_receipt.get('container_id') != host['container_id']
            or {k: row['sha256'] for k, row in native.get('model_assets', {}).items()}
                != launch['collector_execution']['model_sha256']):
        raise ValueError('qualification request/source/assets/receipt chain differs')
    b, cap = contract['num_envs'], contract['cgroup_limit_bytes']
    bounded = request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL
    if bounded:
        if (native['collection_storage']['writer_sha256'] !=
                launch['collector_execution'].get('storage_writer_sha256')
                or contract.get('recipe') != BOUNDED_STORAGE_PROTOCOL
                or any(row.get('protocol') != BOUNDED_STORAGE_PROTOCOL for row in save_events)
                or any(row.get('native_payload_writer_unchanged') is not False or
                       'completed_original_save' in row for row in save_events[1::2])):
            raise ValueError('bounded writer source/profile differs or falsely claims unchanged original writer')
    plateau = plateau_gate(save_events, num_envs=b, cap_bytes=cap,
        completion_field='completed_bounded_save' if bounded else 'completed_original_save')
    posts = plateau['per_flush_memory']
    if (len(save_events) != 2 * request['max_windows'] or len(posts) != request['max_windows']
            or posts[-1]['vector_steps'] != native['vector_steps']
            or posts[-1]['emitted_chunks_per_lane'] != native['emitted_per_lane']):
        raise ValueError('full native save barriers do not cover the exact qualification lifetime')
    for i, row in enumerate(save_events):
        if (row.get('phase') != ('presave' if i % 2 == 0 else 'postsave')
                or row.get('vector_steps') != (i // 2 + 1) * 81
                or row.get('memory', {}).get('memory_max') != cap
                or row.get('memory', {}).get('memory_swap.current') != 0
                or any(row.get('memory', {}).get('events', {}).get(key) != 0
                       for key in ('max', 'oom', 'oom_kill'))
                or (i % 2 == 0 and (row.get('admitted') is not True or row.get('num_envs') != b))):
            raise ValueError('full-buffer pre/post-save evidence changed')
    live = [row for row in memory_samples if row.get('container', {}).get('pid')]
    if not live or any(row.get('host_available_kib', 0) < 8 * 1024**2 for row in memory_samples):
        raise ValueError('whole-run live host resource evidence is missing or unsafe')
    for row in live:
        c = row['container']
        validate_gpu_sample(row.get('gpu', {}), contract['gpu_uuid'])
        if (row.get('simultaneous_native_image_owners') != 1 or c.get('sampling_error')
                or c.get('memory_swap.current') != 0 or c.get('oom_killed') is not False
                or any(c.get('events', {}).get(k) != 0 for k in ('max', 'oom', 'oom_kill'))
                or type(c.get('memory_peak')) is not int or not 0 < c['memory_peak'] <= cap):
            raise ValueError('live continuous resource/ownership evidence failed')
    times = [row.get('monotonic_seconds') for row in memory_samples]
    start, end = host.get('started_monotonic'), host.get('finished_monotonic')
    if (not times or any(type(x) not in (int, float) for x in times + [start, end])
            or times[0] < start or end < times[-1] or times[0] - start > 15
            or end - times[-1] > 45 or any(not 0 < b - a <= 15 for a, b in zip(times, times[1:]))):
        raise ValueError('qualification resource observer lacks whole-run coverage')
    receipt = dict(schema=CAPACITY_SCHEMA, verdict='PASS', native_runtime_tested=True, contract=contract,
        minimum_full_emissions_per_lane=min(native['emitted_per_lane']), accepted_windows=request['max_windows'],
        oom_events=0, swap_bytes_peak=0, bounded_repeated_buffer_only=True,
        cgroup_memory_peak_bytes=max(row['container']['memory_peak'] for row in live),
        minimum_gpu_free_bytes=min(row['gpu']['total_bytes'] - row['gpu']['used_bytes'] for row in live),
        minimum_host_available_bytes=min(row['host_available_kib'] * 1024 for row in memory_samples),
        plateau=plateau, evidence=evidence, production_admitted=False, long_campaign_execution_proven=False)
    return validate_capacity(receipt, contract)


def read_capacity(path, expected_sha):
    """Rebuild PASS from its bound native/host evidence, not just a JSON verdict."""
    path = Path(path)
    value, _ = read_json(path, expected_sha)
    root = path.parent
    paths = dict(launch=root / 'launch.json', host=root / 'host-result.json',
                 native=root / 'bc_isaac/cohort-manifest.json', execution=root / 'continuous-execution.json')
    documents, evidence = {}, {}
    for key, child in paths.items():
        documents[key], evidence[key] = read_json(child)
    for key, name in (('save_events', 'capacity-save-events.jsonl'), ('memory_samples', 'memory-samples.jsonl')):
        child = root / name
        if child.is_symlink() or child.stat().st_size > 64 << 20:
            raise ValueError('bounded physical qualification resource evidence required')
        raw = child.read_bytes()
        documents[key] = [json.loads(line) for line in raw.splitlines()]
        evidence[key] = digest(raw)
    actual = build_capacity(launch=documents['launch'], host=documents['host'], native=documents['native'],
        save_events=documents['save_events'], memory_samples=documents['memory_samples'],
        execution_receipt=documents['execution'], evidence=evidence)
    if actual != value:
        raise ValueError('capacity PASS differs from its actual bound native evidence')
    return value


def read_drain_proof(path, expected_sha, contract):
    """Reconcile real functional completion, not a caller-supplied PASS flag.

    Qualification stays excluded from the production dataset. Capacity remains
    a separate full-buffer proof; neither substitutes for the other.
    """
    from pipeline.bc_collection.episode_drain import is_bounded_qualification
    path = Path(path)
    execution, execution_sha = read_json(path, expected_sha)
    root = path.parent
    launch, launch_sha = read_json(root / 'launch.json')
    host, host_sha = read_json(root / 'host-result.json')
    native, native_sha = read_json(root / 'bc_isaac/cohort-manifest.json')
    request = validate_collection_request(launch['request'])
    validate_completion(native, request, launch['request_sha256'])
    drain = native.get('episode_drain', {})
    if (not any(drain.get('triggered', {}).get('episode_ids', []))
            or not any(drain.get('recorded_drain_rows', []))
            or not any(drain.get('unrecorded_after_drain_rows', []))):
        raise ValueError('drain proof must actually cross a recorded terminal and exclude new-episode rows')
    if (not is_bounded_qualification(request) or len(request['motions']) != 32
            or launch.get('qualification_only') is not True
            or launch.get('capacity_contract') != contract
            or launch.get('collector_execution') != contract['collector_execution']
            or execution.get('schema') != SCHEMA or execution.get('verdict') != 'PASS'
            or execution.get('native_runtime_tested') is not True
            or execution.get('qualification_only') is not True
            or execution.get('production_admitted') is not False
            or execution.get('collector_execution') != contract['collector_execution']
            or execution.get('request_sha256') != launch['request_sha256']
            or execution.get('request_canonical_sha256') != digest(canonical(request))
            or execution.get('actual_num_envs') != 32
            or execution.get('actual_vector_steps') != native['vector_steps']
            or execution.get('initial_reset_all_calls') != 1
            or execution.get('launch_sha256') != launch_sha
            or execution.get('host_result_sha256') != host_sha
            or execution.get('native_manifest_sha256') != native_sha
            or host.get('passed') is not True or host.get('failure') is not None
            or type(host.get('exit_code')) is not int or host['exit_code'] != 0
            or host.get('replay_allowed') is not False
            or not re.fullmatch('[0-9a-f]{64}', str(host.get('container_id', '')))
            or execution.get('container_id') != host['container_id']
            or {k: row['sha256'] for k, row in native.get('model_assets', {}).items()}
                != contract['collector_execution']['model_sha256']):
        raise ValueError('current-source B32 natural-drain execution/owner/evidence proof differs')
    return dict(path=str(path), raw_sha256=execution_sha, launch_sha256=launch_sha,
                host_sha256=host_sha, native_sha256=native_sha,
                capacity_contract_sha256=digest(canonical(contract)))


def detach_once(command, directory, *, popen=subprocess.Popen):
    """Durable one-shot launcher, not a retry or supervision controller.

    SSH disconnect cannot signal this new session. Once intent exists, reconcile
    it: even failed/uncertain submission consumes this namespace.
    """
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    publish_json(directory / 'intent.json', dict(command=command, replay_allowed=False,
        retain_compute=True, state='SUBMITTING_ONCE'))
    with (directory / 'console.log').open('xb') as log:
        process = popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
    # Missing process identity never authorizes resubmitting the command.
    publish_json(directory / 'owner.json', dict(pid=process.pid,
        proc_stat=Path(f'/proc/{process.pid}/stat').read_text(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), command=command))
    return dict(submitted_once=True, pid=process.pid, launcher=str(directory), replay_allowed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(0, 1), default=0)
    parser.add_argument('--num-envs', type=int, default=32, help='One logical scene, never two independent B16 scenes')
    parser.add_argument('--memory-gib', type=int, choices=(40, 43), default=43)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--storage-reserve-gib', type=int, default=64)
    parser.add_argument('--capacity-receipt', type=Path)
    parser.add_argument('--capacity-receipt-sha256')
    parser.add_argument('--drain-execution', type=Path)
    parser.add_argument('--drain-execution-sha256')
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true', help='one-shot SSH-independent launch; never retry')
    args = parser.parse_args(argv)
    if args.execute and args.detach:
        raise ValueError('choose execute or detach, not both')
    raw_request, request_sha = read_json(args.request, args.request_sha256)
    request = validate_launch(raw_request, qualification=args.qualification, expected_num_envs=args.num_envs,
        physical_gpu=args.physical_gpu, memory_gib=args.memory_gib, timeout=args.timeout)
    from pipeline.bc_collection.production_gate import is_production
    production = is_production(request)
    manifest, manifest_sha = read_json(args.source_manifest)
    source = args.source_root.resolve(strict=True)
    source_files = verify_source(source, manifest)
    paths = load_paths(source)
    runtime = paths.validate_run(args.runtime)
    policy = validate_storage_policy(dict(schema=STORAGE_SCHEMA, root=str(args.storage_root),
        reserve_bytes=args.storage_reserve_gib * GIB, bytes_per_emission_budget=512 << 20,
        session_metadata_budget_bytes=GIB, poll_seconds=1))
    budget = storage_budget(policy, request)
    if args.qualification and (args.capacity_receipt is not None or args.capacity_receipt_sha256 is not None):
        raise ValueError('qualification cannot borrow another run capacity')
    capacity = None
    if (args.capacity_receipt is None) != (args.capacity_receipt_sha256 is None):
        raise ValueError('capacity path and raw hash must be supplied together')
    if args.capacity_receipt is not None:
        capacity = read_capacity(args.capacity_receipt, args.capacity_receipt_sha256)
    if (args.drain_execution is None) != (args.drain_execution_sha256 is None):
        raise ValueError('drain execution path and raw hash must be supplied together')
    if args.drain_execution is not None and not production:
        raise ValueError('only explicit production may consume a drain proof')
    drain_binding = None
    if args.drain_execution is not None:
        if capacity is None:
            raise ValueError('drain proof requires independent current-source capacity')
        drain_binding = read_drain_proof(args.drain_execution, args.drain_execution_sha256, capacity['contract'])
    if not args.execute and not args.detach:
        print(json.dumps(dict(execute=False, production_admitted=False, qualification_only=args.qualification,
            num_envs=len(request['motions']), logical_scenes=1, physical_gpu=args.physical_gpu,
            request_sha256=request_sha, source_sha256=manifest['source_sha256'],
            required_vector_steps=request['max_windows'] * 81, storage_budget=budget,
            capacity_supplied=capacity is not None, source_files_verified=True,
            next_gate='fresh Leo owner/GPU/RAM/storage and actual current-source full-buffer capacity')))
        return
    if socket.gethostname() != 'leo-rtx-6k':
        raise RuntimeError('execution is Leo-only; staging does not launch a remote workload')
    from roborender.infrastructure.source_locations import resolve_source_release
    if resolve_source_release(source) != source or Path(__file__).resolve() != source / SELF:
        raise ValueError('execute the launcher from the same immutable source release')
    if not args.qualification and capacity is None:
        raise ValueError('production execution requires completed current-source full-buffer qualification')
    if production and drain_binding is None:
        raise ValueError('production execution requires actual current-source B32 natural-drain proof')
    if args.detach:
        arguments = list(sys.argv[1:] if argv is None else argv)
        arguments.remove('--detach')
        arguments.append('--execute')
        directory = paths.get('run_root') / (runtime.name + '-launcher')
        print(json.dumps(detach_once([sys.executable, str(source / SELF), *arguments], directory)))
        return
    actor = source / ACTOR
    sys.path.insert(0, str(actor))
    from materialize_isaac6 import materialize
    from qualify_native_isaac6 import IMAGE
    from run_success_diagnostic import gpu_gate
    from run_throughput_pair import memory_kib, owned_memory_sample
    from bc_memory_guard import admission, runtime_guard, exclusive_native_owner
    from leo_binding import checked_gpu

    def owners():
        return subprocess.check_output(['docker', 'ps', '--filter', 'ancestor=' + IMAGE,
            '--format', '{{.Names}}'], text=True, timeout=15).splitlines()

    # A single host lock covers both GPU choices. It never adopts an old run;
    # durable fresh-runtime creation and the owner gate prevent replay/races.
    paths.check_assets(args.physical_gpu)
    lock_root = paths.get('run_root')
    if lock_root.resolve(strict=True) != lock_root:
        raise ValueError('canonical lock root changed')
    with (lock_root / '.continuous-bc-owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exclusive_native_owner(owners())  # Never stop/replace existing warm Isaac owners.
        gpu_gate(args.physical_gpu, deployment_host='leo-rtx-6k')
        admission(memory_kib()['MemAvailable'], args.memory_gib)
        storage_report = {}
        available = observe_storage(storage_snapshot(policy, runtime, runtime), policy, storage_report)
        if available < budget['required_available_bytes']:
            raise RuntimeError('full declared emission budget plus storage reserve is unavailable')
        runtime.mkdir(parents=True, exist_ok=False)
        paths.freeze(runtime)
        cache = paths.mounts(args.physical_gpu)['cache']
        cache.mkdir(parents=True, exist_ok=True)
        paths.check_assets(args.physical_gpu)
        (runtime / 'home').mkdir()
        (runtime / 'bc_isaac').mkdir()
        # Failure after this intent consumes the namespace, even if launch
        # acceptance is uncertain; no automatic retries are implemented.
        publish_json(runtime / 'intent.json', dict(schema='roborender.bc-continuous-intent.v1',
            request_sha256=request_sha, source_sha256=manifest['source_sha256'],
            state='PREPARING', replay_allowed=False, retain_all_h200_compute=True))
        materialize(paths.get('bc_runtime_bundle'), runtime / 'overlay', source / 'pipeline/dagger/rtx/config/assets.manifest.json')
        port, port_sha = read_json(runtime / 'overlay/port-manifest.json')
        guard_name = ('bounded_bc_guard' if request.get('collection_storage_protocol')
                      == BOUNDED_STORAGE_PROTOCOL else 'continuous_bc_guard')
        guard_sha = source_files[f'{ACTOR}/{guard_name}.py']
        execution = dict(source_sha256=manifest['source_sha256'], source_manifest_sha256=manifest_sha,
            port_manifest_sha256=port_sha, image_id=IMAGE, guard_sha256=guard_sha,
            storage_registry_sha256=paths.registry.raw_sha256,
            native_runtime_version=dict(isaac_sim=port['isaac_sim'], isaaclab=port['isaaclab_commit']),
            lifecycle='one_continuous_logical_scene_exact_window_budget_v1')
        if request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL:
            execution['storage_writer_sha256'] = source_files[f'{ACTOR}/bounded_bc_storage.py']
        assets, assets_sha = read_json(runtime / 'overlay/assets.manifest.json')
        names = {'privileged_teacher': 'teacher', 'homie_walk': 'homie_walk', 'homie_stand': 'homie_stand'}
        models = {names[row['role']]: row['sha256'] for row in assets['artifacts'] if row['role'] in names}
        if set(models) != set(names.values()):
            raise ValueError('native teacher/Homie identity incomplete')
        execution.update(assets_manifest_sha256=assets_sha, model_sha256=models)
        contract = capacity_contract(execution, request, gpu_uuid=checked_gpu(args.physical_gpu, 'bc-collect'),
                                     memory_gib=args.memory_gib)
        if capacity is not None:
            validate_capacity(capacity, contract)
        if production:
            drain_binding = read_drain_proof(args.drain_execution, args.drain_execution_sha256, contract)
        name, cmd = build_command(request=request, request_path=args.request.resolve(), request_sha256=request_sha,
            source=source, runtime=runtime, overlay=runtime / 'overlay', physical_gpu=args.physical_gpu,
            memory_gib=args.memory_gib, guard_sha256=guard_sha, runtime_paths=paths)
        named = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                                        text=True, timeout=15).splitlines()
        if name in named:
            raise RuntimeError('container identity already exists; do not adopt or stop it')
        verify_source(source, manifest)
        read_json(args.request, request_sha)
        exclusive_native_owner(owners())
        admission(memory_kib()['MemAvailable'], args.memory_gib)
        available = observe_storage(storage_snapshot(policy, runtime, runtime), policy, storage_report)
        if available < budget['required_available_bytes']:
            raise RuntimeError('storage admission changed before launch')
        launch = dict(schema='roborender.bc-continuous-launch.v1', request=request,
            request_sha256=request_sha, command=cmd, collector_execution=execution, capacity_contract=contract,
            qualification_only=args.qualification, production_admitted=False, storage_policy=policy,
            storage_budget=budget, capacity_binding=None if capacity is None else
            dict(path=str(args.capacity_receipt), raw_sha256=args.capacity_receipt_sha256))
        if production:
            launch['production_admission'] = dict(schema='roborender.bc-production-launch-admission.v1',
                ready=True, request_sha256=request_sha, source_sha256=manifest['source_sha256'],
                capacity=launch['capacity_binding'], natural_drain=drain_binding,
                rolling_storage=budget, diagnostic_data_included=False)
        publish_json(runtime / 'launch.json', launch)
        process, failure, accepted, container_id = None, None, False, None
        observer = None
        started = time.monotonic()
        native_path = runtime / 'bc_isaac/cohort-manifest.json'
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise InterruptedError('stop only this exact newly launched collection workload')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (runtime / 'console.log').open('xb') as log, (runtime / 'memory-samples.jsonl').open('x') as samples:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                accepted = True
                while process.poll() is None:
                    if time.monotonic() - started > args.timeout:
                        raise TimeoutError('continuous collection safety timeout; partial data excluded')
                    observe_storage(storage_snapshot(policy, runtime, runtime), policy, storage_report)
                    host = memory_kib()
                    if request.get('collection_storage_protocol') == BOUNDED_STORAGE_PROTOCOL:
                        from bc_resource_observer import PinnedCgroupObserver, recover_observation, observe_command
                        def read_actor():
                            nonlocal observer
                            def read_terminal():
                                completed, receipt_sha = read_json(native_path)
                                validate_completion(completed, request, request_sha)
                                return dict(completion_receipt_sha256=receipt_sha, native_pass_receipt=True)
                            def read_live():
                                nonlocal observer
                                if observer is None:
                                    observer = PinnedCgroupObserver(runtime / 'container.cid', name, IMAGE,
                                                                   args.memory_gib * GIB)
                                return observer.sample()
                            return observe_command(process, read_live, read_terminal, native_path.is_file)

                        def journal(event):
                            with (runtime / 'observation-events.jsonl').open('a') as events:
                                events.write(json.dumps(dict(monotonic_seconds=time.monotonic(), **event)) + '\n')
                                events.flush()
                                os.fsync(events.fileno())

                        def check_sample(value):
                            if value.get('lifecycle') != 'completed_attested':
                                runtime_guard(memory_kib(), value)
                                if value['memory_current'] + 2 * GIB > args.memory_gib * GIB:
                                    raise RuntimeError('owned collector below2GiB current memory headroom')

                        def check_host():
                            if memory_kib()['MemAvailable'] < 8 * 1024**2:
                                raise RuntimeError('single collector below8GiB host reserve')

                        sample = recover_observation(read_actor, check_sample, journal=journal,
                            hold=lambda: publish_json(runtime / 'observer-hold.json', dict(owner=name)),
                            release=lambda: (runtime / 'observer-hold.json').unlink(),
                            host_check=check_host, attempts=20 if observer is None else 3)
                    else:
                        sample = owned_memory_sample(name, process, native_path,
                            completion_field='status', completion_value='complete', receipt_max_bytes=32 << 20)
                    if sample.get('lifecycle') == 'completed_attested':
                        break
                    observe_resources(samples, host=host, sample=sample, guard=runtime_guard,
                        gpu_reader=lambda: gpu_sample(contract['gpu_uuid']),
                        owner_check=lambda: exclusive_native_owner(owners(), name),
                        startup=time.monotonic() - started < 15)
                    time.sleep(1)
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
        finally:
            # Popen is not proof of Docker acceptance. Require cidfile and
            # exact ID/image/label; a missing response never authorizes replay.
            if accepted:
                try:
                    container_id = cleanup_exact(name, runtime / 'container.cid', IMAGE)
                    if process.poll() is None:
                        process.wait(timeout=30)
                except Exception as exc:
                    failure = dict(type=type(exc).__name__, message='owned cleanup uncertain: ' + str(exc))
            signal.signal(signal.SIGTERM, old_handler)
        native, native_sha = {}, None
        if failure is None:
            try:
                if process is None or process.returncode != 0:
                    raise RuntimeError('exact collection process did not exit successfully')
                native, native_sha = read_json(native_path)
                validate_completion(native, request, request_sha)
            except Exception as exc:
                failure = dict(type=type(exc).__name__, message=str(exc))
        host_result = dict(passed=failure is None, failure=failure,
            exit_code=process.returncode if process is not None else None,
            elapsed_seconds=time.monotonic() - started, storage=storage_report,
            container_id=container_id,
            started_monotonic=started, finished_monotonic=time.monotonic(),
            production_admitted=False, replay_allowed=False)
        publish_json(runtime / 'host-result.json', host_result)
        if failure is not None:
            print(json.dumps(host_result))
            raise SystemExit(1)
        receipt = dict(schema=SCHEMA, verdict='PASS', native_runtime_tested=True,
            request_sha256=request_sha, request_canonical_sha256=digest(canonical(request)),
            native_manifest_sha256=native_sha,
            host_result_sha256=digest((runtime / 'host-result.json').read_bytes()),
            launch_sha256=digest((runtime / 'launch.json').read_bytes()), collector_execution=execution,
            actual_num_envs=len(request['motions']), actual_vector_steps=native['vector_steps'],
            container_id=container_id,
            initial_reset_all_calls=1, qualification_only=args.qualification, production_admitted=False)
        publish_json(runtime / 'continuous-execution.json', receipt)
        capacity_probe = args.qualification and 'diagnostic_stop' not in request and 'episode_drain' not in request
        if capacity_probe:
            try:
                events_raw = (runtime / 'capacity-save-events.jsonl').read_bytes()
                samples_raw = (runtime / 'memory-samples.jsonl').read_bytes()
                capacity = build_capacity(launch=launch, host=host_result, native=native,
                    save_events=[json.loads(line) for line in events_raw.splitlines()],
                    memory_samples=[json.loads(line) for line in samples_raw.splitlines()], execution_receipt=receipt,
                    evidence=dict(launch=receipt['launch_sha256'], host=receipt['host_result_sha256'], native=native_sha,
                        save_events=digest(events_raw), memory_samples=digest(samples_raw),
                        execution=digest((runtime / 'continuous-execution.json').read_bytes())))
                publish_json(runtime / 'capacity-receipt.json', capacity)
            except Exception as exc:
                failure = dict(schema='roborender.bc-continuous-qualification-failure.v1',
                    native_collection_completed=True, capacity_qualified=False, production_admitted=False,
                    failure=dict(type=type(exc).__name__, message=str(exc)), replay_allowed=False)
                publish_json(runtime / 'qualification-failure.json', failure)
                print(json.dumps(failure))
                raise SystemExit(1)
        print(json.dumps(dict(execution=str(runtime / 'continuous-execution.json'),
            capacity_receipt=str(runtime / 'capacity-receipt.json') if capacity_probe else None,
            functional_episode_drain_test='episode_drain' in request,
            production_admitted=False, actual_vector_steps=native['vector_steps'])))


if __name__ == '__main__':
    main()
