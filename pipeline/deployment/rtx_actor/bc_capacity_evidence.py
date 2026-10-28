"""Publish actual bounded full-buffer evidence from completed small receipts.

No plan can create PASS. Reads JSON/JSONL only, never NPZ/model/image payloads.
This producer is management provenance, not part of the native runtime contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from bc_capacity_guard import plateau_gate

GIB = 2**30
RULE = dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=.10,
            absolute_growth_slack_bytes=256 << 20)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build_capacity_receipt(*, launch, host, native, cohort, session_execution,
                           cohort_execution, save_events, memory_samples, bindings):
    from pipeline.bc_training.production_protocol import validate_bounded_capacity
    from pipeline.deployment.rtx_actor.isaac6_bc_production import NATIVE_COLLECTOR_SHA256
    execution = launch['collector_execution']
    topology = launch.get('execution_topology', {})
    request = cohort['request']
    b = len(request['motions'])
    maximum_windows = launch.get('qualification_max_windows', 4)
    if (launch.get('qualification_only') is not True or launch.get('production_admitted') is not False
            or launch.get('evidence_producer_sha256') != digest(Path(__file__).read_bytes())
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in bindings.values())
            or b not in (1, 4, 6) or request['schema'] != 'roborender.bc-native-collection-request.v2'
            or request['chunks_per_env'] != [1] * b or request['selected_chunk_ordinals_per_env'] != [[1]] * b
            or type(maximum_windows) is not int or maximum_windows not in (4, 8)
            or request['exposure_span_chunks_per_env'] != [2] * b
            or type(request['max_windows']) is not int or request['max_windows'] != maximum_windows
            or request['split'] != 'validation'
            or topology.get('host_id') != 'leo-rtx-6k' or topology.get('concurrent_collectors') != 1
            or topology.get('cgroup_limit_bytes_per_collector') not in (40 * GIB, 43 * GIB)
            or topology.get('minimum_host_reserve_bytes') != 8 * GIB
            or host.get('passed') is not True or host.get('failure') is not None or host.get('exit_code') != 0
            or native.get('status') != 'complete' or native.get('failure') is not None
            or native.get('native_initializations') != 1 or native.get('scene_initializations') != 1
            or len(native.get('cohorts', [])) != 1
            or cohort.get('status') != 'complete' or cohort.get('optimizer_steps') != 0
            or cohort.get('teacher_only') is not True or cohort.get('native_payload_writer_unchanged') is not True
            or cohort.get('native_collector_sha256') != NATIVE_COLLECTOR_SHA256
            or cohort.get('selected_chunks') != b or type(cohort.get('vector_steps')) is not int
            or not 162 <= cohort['vector_steps'] <= 81 * maximum_windows or cohort['vector_steps'] % 81):
        raise ValueError('real bounded single-cohort qualification did not complete')
    if (session_execution.get('schema') != 'roborender.bc-session-execution.v1'
            or session_execution.get('verdict') != 'PASS' or session_execution.get('collector_execution') != execution
            or session_execution.get('session_id') != native.get('session_id')
            or session_execution.get('launch_raw_sha256') != bindings['launch.json']
            or session_execution.get('native_result_raw_sha256') != bindings['bc-session-result.json']
            or session_execution.get('host_result_raw_sha256') != bindings['host-result.json']
            or session_execution.get('source_manifest_raw_sha256') != launch['source_manifest_raw_sha256']
            or session_execution.get('port_manifest_raw_sha256') != launch['port_manifest_raw_sha256']
            or session_execution.get('session_sha256') != launch['session_sha256']
            or session_execution.get('session_canonical_sha256') != launch['session_canonical_sha256']
            or session_execution.get('request_identities') != launch['request_identities']
            or cohort_execution.get('schema') != 'roborender.bc-cohort-execution.v1'
            or cohort_execution.get('verdict') != 'PASS' or cohort_execution.get('collector_execution') != execution
            or cohort_execution.get('cohort_manifest_raw_sha256') != bindings['cohort-manifest.json']
            or cohort_execution.get('session_execution_canonical_sha256') != digest(canonical(session_execution))
            or cohort_execution.get('request_sha256') != cohort['request_sha256']
            or cohort_execution.get('request_canonical_sha256') != digest(canonical(request))
            or cohort_execution.get('actual_vector_steps') != cohort['vector_steps']
            or cohort_execution.get('actual_num_envs') != b
            or session_execution.get('request_identities', {}).get(cohort['request_sha256']) != digest(canonical(request))):
        raise ValueError('capacity evidence changed or is not bound to completed native execution')
    cap = topology['cgroup_limit_bytes_per_collector']
    plateau = plateau_gate(save_events, num_envs=b, cap_bytes=cap)
    posts = plateau['per_flush_memory']
    if (posts[-1]['vector_steps'] != cohort['vector_steps']
            or posts[-1]['emitted_chunks_per_lane'] != cohort['emitted_per_lane']
            or native['cohorts'][0]['vector_steps'] != cohort['vector_steps']
            or native['cohorts'][0]['request_sha256'] != cohort['request_sha256']):
        raise ValueError('last actual save does not match native completion')
    pres = [r for r in save_events if r.get('phase') == 'presave']
    if (len(pres) != len(posts) or len(save_events) != 2 * len(posts)
            or any(r.get('phase') != ('presave' if i % 2 == 0 else 'postsave') for i, r in enumerate(save_events))
            or any(r.get('admitted') is not True or r.get('num_envs') != b
                   or r.get('memory_max_bytes') != cap for r in pres)):
        raise ValueError('every original save needs its admitted pre-save and completed post-save')
    live = [row for row in memory_samples if row.get('container', {}).get('pid')]
    if not live or any(type(row.get('host_available_kib')) is not int or row['host_available_kib'] < 8 * 1024**2
                       for row in memory_samples):
        raise ValueError('missing or unsafe live host memory evidence')
    for row in live:
        c = row['container']
        if (row.get('simultaneous_native_image_owners') != 1 or c.get('sampling_error')
                or c.get('memory_swap.current') != 0 or c.get('oom_killed') is not False
                or any(c.get('events', {}).get(k) != 0 for k in ('max', 'oom', 'oom_kill'))
                or type(c.get('memory_peak')) is not int or not 0 < c['memory_peak'] <= cap):
            raise ValueError('observed live ownership/cgroup safety differs')
    for row in save_events:
        mem = row['memory']
        if (mem.get('memory_max') != cap or mem.get('memory_swap.current') != 0
                or any(mem.get('events', {}).get(k) != 0 for k in ('max', 'oom', 'oom_kill'))):
            raise ValueError('native flush reported pressure, swap or a different cap')
    models = {k: row['sha256'] for k, row in cohort.get('model_assets', {}).items()}
    if (set(models) != {'teacher', 'homie_walk', 'homie_stand'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in models.values())):
        raise ValueError('native model identities missing')
    receipt = dict(schema='roborender.native-bc-capacity.v2', purpose='full_bc_capture_and_save', verdict='PASS',
        native_runtime_tested=True, collector_execution=execution, native_collector_sha256=NATIVE_COLLECTOR_SHA256,
        model_sha256=models, num_envs=b, host_id=topology['host_id'], concurrent_collectors=1,
        simultaneous_collectors_observed=1, cgroup_limit_bytes=cap, runtime_guard_enabled=True,
        active_guard_minimum_host_available_bytes=8 * GIB, bounded_repeated_buffer_only=True,
        oom_events_delta=0, swap_bytes_peak=0, cgroup_max_events=0, plateau_rule=RULE,
        per_flush_memory=posts, vector_steps=cohort['vector_steps'], full_chunks_saved=sum(cohort['emitted_per_lane']),
        emitted_chunks_per_lane=cohort['emitted_per_lane'], maximum_completed_ordinal_tested=1,
        maximum_vector_steps_declared=81 * maximum_windows,
        cgroup_memory_peak_bytes=max([r['container']['memory_peak'] for r in live] + [r['memory_peak_bytes'] for r in posts]),
        runtime_min_mem_available_bytes=min([r['host_available_kib'] * 1024 for r in memory_samples]
                                           + [r['host_available_bytes'] for r in save_events]),
        evidence_raw_sha256=bindings, producer_sha256=digest(Path(__file__).read_bytes()),
        production_admitted=False, long_campaign_execution_proven=False)
    protocol = dict(execution_topology=topology, collector_execution=execution, model_sha256=models,
                    capacity_equivalence=RULE)
    validate_bounded_capacity(receipt, protocol)  # Same real checker used by production admission.
    return receipt


def publish_capacity_receipt(runtime, publish):
    runtime = Path(runtime).resolve(strict=True)
    names = ('launch.json', 'host-result.json', 'bc-session-result.json', 'session-execution.json',
             'capacity-save-events.jsonl', 'memory-samples.jsonl')
    raw, values, bindings = {}, {}, {}
    for name in names:
        path = runtime / name
        if path.stat().st_size > 32 << 20:
            raise ValueError('capacity metadata exceeds bounded size')
        raw[name] = path.read_bytes()
        bindings[name] = digest(raw[name])
        values[name] = [json.loads(line) for line in raw[name].splitlines()] if name.endswith('.jsonl') else json.loads(raw[name])
    native = values['bc-session-result.json']
    if len(native.get('cohorts', [])) != 1:
        raise ValueError('capacity producer supports one actual qualification cohort only')
    child = (runtime / Path(native['cohorts'][0]['output_dir']).relative_to('/dagger-runtime')).resolve(strict=True)
    if not child.is_relative_to(runtime):
        raise ValueError('native cohort escaped owned runtime')
    for name in ('cohort-manifest.json', 'execution-receipt.json'):
        path = child / name
        if path.stat().st_size > 32 << 20:
            raise ValueError('cohort metadata exceeds bounded size')
        value = path.read_bytes()
        bindings[name] = digest(value)
        values[name] = json.loads(value)
    receipt = build_capacity_receipt(launch=values['launch.json'], host=values['host-result.json'], native=native,
        cohort=values['cohort-manifest.json'], session_execution=values['session-execution.json'],
        cohort_execution=values['execution-receipt.json'], save_events=values['capacity-save-events.jsonl'],
        memory_samples=values['memory-samples.jsonl'], bindings=bindings)
    path = runtime / 'capacity-receipt.json'
    publish(path, receipt)
    return {'path': str(path), 'raw_sha256': digest(path.read_bytes())}
