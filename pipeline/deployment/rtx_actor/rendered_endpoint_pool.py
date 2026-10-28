"""Bounded, deterministic Image3F endpoint dispatch; no native-state ownership.

One process/session per physical GPU is admitted externally. Source/GPU pins
are tied to a deployment readiness receipt, while the live cached-channel and
generation-source handshakes are checked here before any image generation.
There is deliberately no alternate-worker retry or native-RGB fallback.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import pickle
from pathlib import Path
import re
import threading
import time

CACHE_SCHEMA = 'roborender.image3f.cached-reply.v1'
POOL_SCHEMA = 'roborender.rendered-success-endpoint-pool.v1'
SHARDED_POOL_SCHEMA = 'roborender.rendered-success-endpoint-pool.v2'
TIMING_SCHEMA = 'roborender.rendered-success-timing.v1'
ROUTING = 'lane_mod_endpoint_count'
SHARDED_ROUTING = 'cohort_parity_actor_shards_v1'
ACTOR_SHARDS = [dict(physical_gpu=0, endpoint_slots=[0, 1, 2, 3]),
                dict(physical_gpu=1, endpoint_slots=[4, 5, 6, 7])]
ENDPOINT_FIELDS = {'slot', 'host', 'port', 'gpu_uuid', 'server_instance',
                   'source_sha256', 'generate_source_sha256', 'readiness_receipt_sha256'}
CLIENT_TIMING_FIELDS = ('worker_total_ms', 'loopback_total_ms')
CLIENT_TIMING_SEMANTICS = {
    'schema': 'roborender.rendered-success-client-timing.v1',
    'worker_total_ms': 'server_host_wall_decode_session_lock_wait_generate_encode',
    'loopback_total_ms': 'client_host_wall_encode_rpc_cached_recovery_response_decode',
    'renderer_runtime_ms': 'server_reported_pipeline_ms_with_server_total_fallback_not_gpu_kernel_time',
    'queue_seconds': 'lane_start_minus_vector_start_includes_prior_same_endpoint_requests',
    'elapsed_seconds': 'client_host_wall_render_call_and_completion_validation',
    'interpretation': 'nested_not_additive_differences_are_not_pure_network_time_no_cross_host_clock_subtraction',
}


def _client_timings(result):
    """Keep optional original client durations; missing evidence is not zero.

    These are nested host-wall scopes, not additive GPU/kernel measurements.
    The server total includes its session-lock wait. The client total includes
    serialization and cached-reply recovery, so subtracting the server total
    does not isolate network latency. No cross-host timestamps are compared.
    """
    missing = object()
    values = {}
    for key in CLIENT_TIMING_FIELDS:
        value = getattr(result, key, missing)
        if value is missing:
            continue
        if not _nonnegative(value):
            raise RuntimeError('renderer completion has invalid optional client timing')
        values[key] = value
    return values


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_pool_config(renderer):
    sharded = isinstance(renderer, dict) and renderer.get('routing') == SHARDED_ROUTING
    keys = {'identity', 'endpoints', 'routing', 'max_inflight_per_endpoint'} | ({'actor_shards'} if sharded else set())
    if (not isinstance(renderer, dict) or set(renderer) != keys
            or renderer['routing'] not in (ROUTING, SHARDED_ROUTING)
            or type(renderer['max_inflight_per_endpoint']) is not int
            or renderer['max_inflight_per_endpoint'] != 1
            or not isinstance(renderer['endpoints'], list)
            or not 1 <= len(renderer['endpoints']) <= 8):
        raise ValueError('scaled renderer requires a bounded exclusive endpoint pool')
    if sharded and (len(renderer['endpoints']) != 8 or
            json.dumps(renderer['actor_shards'], sort_keys=True) != json.dumps(ACTOR_SHARDS, sort_keys=True)):
        raise ValueError('sharded renderer requires eight endpoints and the exact two disjoint physical-GPU shards')
    endpoints = renderer['endpoints']
    for slot, row in enumerate(endpoints):
        if (not isinstance(row, dict) or set(row) != ENDPOINT_FIELDS
                or type(row['slot']) is not int or row['slot'] != slot
                or row['host'] != '127.0.0.1' or type(row['port']) is not int
                or not 1024 <= row['port'] <= 65535
                or not isinstance(row['gpu_uuid'], str)
                or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', row['gpu_uuid']) is None
                or not isinstance(row['server_instance'], str)
                or re.fullmatch('[0-9a-f]{32}', row['server_instance']) is None
                or any(not _sha(row[key]) for key in (
                    'source_sha256', 'generate_source_sha256', 'readiness_receipt_sha256'))):
            raise ValueError('endpoint requires exact live worker, source, GPU and readiness identity')
    for key in ('port', 'gpu_uuid', 'server_instance'):
        if len({row[key] for row in endpoints}) != len(endpoints):
            raise ValueError('endpoint pool must use distinct ports, processes and physical GPUs')
    # A different code release per device is not a throughput optimization.
    for key in ('source_sha256', 'generate_source_sha256'):
        if len({row[key] for row in endpoints}) != 1:
            raise ValueError('endpoint workers must use the same pinned renderer source')
    return deepcopy(renderer)


def expected_physical_gpu(renderer, cohort_index):
    """Full-panel cohort identity fixes actor placement, independent of timing."""
    validate_pool_config(renderer)
    if renderer['routing'] != SHARDED_ROUTING:
        return None
    if type(cohort_index) is not int or not 0 <= cohort_index < 300:
        raise ValueError('sharded rendering requires the original full-panel cohort index')
    return cohort_index % 2


def execution_shard(renderer, *, cohort_index=None, physical_gpu=None):
    expected = expected_physical_gpu(renderer, cohort_index)
    if expected is None:
        return None
    if type(physical_gpu) is not int or physical_gpu != expected:
        raise ValueError('actual RTX physical GPU differs from the cohort renderer shard')
    return dict(cohort_index=cohort_index, physical_gpu=physical_gpu,
                endpoint_slots=list(ACTOR_SHARDS[physical_gpu]['endpoint_slots']))


def endpoint_slot_for_lane(renderer, lane, *, cohort_index=None, physical_gpu=None):
    if type(lane) is not int or not 0 <= lane < 40:
        raise ValueError('explicit physical lane index required')
    shard = execution_shard(renderer, cohort_index=cohort_index, physical_gpu=physical_gpu)
    slots = shard['endpoint_slots'] if shard else list(range(len(renderer['endpoints'])))
    return slots[lane % len(slots)]


def _open_pinned_channel(client, endpoint, identity):
    if (getattr(client, 'host', None) != endpoint['host']
            or getattr(client, 'port', None) != endpoint['port']
            or getattr(client, '_recovery', None) != CACHE_SCHEMA
            or getattr(client, '_recovery_timeout_s', None) != 75.
            or not _nonnegative(getattr(client, 'timeout', None))
            or not 0 < client.timeout <= 120
            or getattr(client, '_sequence', None) != 0
            or getattr(client, '_server_instance', None) is not None):
        raise ValueError('scaled endpoint requires a fresh exact cached-reply client')
    # This is metadata only. The same socket/channel is retained for rendering;
    # no unseen logical request is replayed if this connection is subsequently lost.
    attestation = client.observation_contract()
    expected = dict(schema='roborender.dagger.student-observation.v1', mode='generated_only',
        renderer='Image3F', renderer_steps=identity['steps'], renderer_cfg=identity['cfg'],
        teacache_threshold=identity['teacache_threshold'], raytraced_policy_fallback=False)
    if (not isinstance(attestation, dict)
            or attestation.get('schema') != 'roborender.image3f.observation-attestation.v1'
            or attestation.get('contract') != expected
            or attestation.get('backend_class') != 'runtime.image3f_deploy.Image3FSession'
            or attestation.get('generate_source_sha256') != endpoint['generate_source_sha256']):
        raise RuntimeError('live renderer generation contract/source differs from endpoint admission')
    with client._request_lock:
        response = client._exchange(pickle.dumps({'cmd': 'transport_open', 'channel': client._channel},
            protocol=pickle.HIGHEST_PROTOCOL), max_response_bytes=4096)
        expected_transport = dict(schema=CACHE_SCHEMA, server_instance=endpoint['server_instance'],
            channel=client._channel)
        if (not isinstance(response, dict) or response.get('ok') is not True
                or response.get('transport') != expected_transport):
            raise RuntimeError('live renderer server instance differs from endpoint admission')
        client._server_instance = endpoint['server_instance']
    return dict(lane=None, endpoint_slot=endpoint['slot'], channel=client._channel,
        server_instance=endpoint['server_instance'])


class RenderedEndpointPool:
    """One worker future per endpoint and one bounded vector request at a time.

Each worker consumes its deterministic lane list in order. A failed lane
poisons the whole pool; unstarted lanes are skipped and already-started lanes
are drained before raising. Partial COMPLETE results remain diagnostic only.
The underlying server lock also serializes distinct actors sharing a GPU.
"""
    def __init__(self, clients, renderer, *, cohort_index=None, physical_gpu=None):
        self.config = validate_pool_config(renderer)
        self.clients = tuple(clients)
        if not 1 <= len(self.clients) <= 40 or len({id(c) for c in self.clients}) != len(self.clients):
            raise ValueError('one exclusive transport client per physical lane required')
        self.endpoints = self.config['endpoints']
        self.shard = execution_shard(self.config, cohort_index=cohort_index, physical_gpu=physical_gpu)
        self.host_boot_id = None
        if self.shard:
            self.host_boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', self.host_boot_id) is None:
                raise ValueError('sharded concurrency proof requires actual host boot identity')
        self.slots = self.shard['endpoint_slots'] if self.shard else list(range(len(self.endpoints)))
        self.routes = [self.slots[lane % len(self.slots)] for lane in range(len(self.clients))]
        self.bindings = []
        for lane, client in enumerate(self.clients):
            row = _open_pinned_channel(client, self.endpoints[self.routes[lane]], self.config['identity'])
            self.bindings.append({**row, 'lane': lane})
        if len({row['channel'] for row in self.bindings}) != len(self.bindings):
            raise ValueError('duplicate renderer transport channel')
        self.executor = ThreadPoolExecutor(max_workers=len(self.slots), thread_name_prefix='success-endpoint')
        self._busy = threading.Lock()
        self.failed = self.closed = False
        self.steps = []
        self.partial = None
        self.trace = hashlib.sha256()

    def render(self, requests, step):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError('endpoint pool permits only one bounded vector step in flight')
        try:
            if self.failed or self.closed or step != len(self.steps) or len(requests) != len(self.clients):
                raise RuntimeError('endpoint pool cannot retry/replay a partial or mismatched step')
            started = time.perf_counter()
            started_monotonic = time.monotonic()
            stop = threading.Event()
            results = [None] * len(self.clients)
            lane_rows = [None] * len(self.clients)
            errors = []
            error_lock = threading.Lock()

            def worker(slot):
                for lane, route in enumerate(self.routes):
                    if route != slot:
                        continue
                    if stop.is_set():
                        lane_rows[lane] = dict(lane=lane, endpoint_slot=slot, state='not_started')
                        continue
                    lane_started = time.perf_counter()
                    request_started = time.monotonic()
                    try:
                        result = self.clients[lane].render(**requests[lane])
                        transport = self.clients[lane].transport_receipt()
                        expected = self.bindings[lane]
                        if (transport.get('schema') != CACHE_SCHEMA or transport.get('failed') is not False
                                or transport.get('server_instance') != expected['server_instance']
                                or transport.get('channel') != expected['channel']
                                or transport.get('logical_requests_completed') != step + 1):
                            raise RuntimeError('renderer completion differs from pinned lane/channel/sequence')
                        runtime_ms = float(result.runtime_ms)
                        if not math.isfinite(runtime_ms) or runtime_ms < 0:
                            raise RuntimeError('renderer completion lacks finite runtime timing')
                        client_timings = _client_timings(result)
                        results[lane] = result
                        lane_rows[lane] = dict(lane=lane, endpoint_slot=slot, state='complete',
                            sequence=step, channel=expected['channel'], queue_seconds=lane_started-started,
                            elapsed_seconds=time.perf_counter()-lane_started, renderer_runtime_ms=runtime_ms,
                            **client_timings)
                        if self.shard:
                            lane_rows[lane].update(request_started_monotonic_s=request_started,
                                request_finished_monotonic_s=time.monotonic())
                    except Exception as exc:
                        lane_rows[lane] = dict(lane=lane, endpoint_slot=slot, state='failed',
                            error_type=type(exc).__name__)
                        with error_lock:
                            errors.append(exc)
                        stop.set()

            futures = [self.executor.submit(worker, slot) for slot in self.slots]
            for future in futures:
                future.result()  # drain in-flight work before exposing failure
            if errors:
                self.failed = True
                self.partial = dict(step=step, lanes=lane_rows, accepted=False)
                raise RuntimeError('rendered endpoint partial failure; no fallback or lane replay') from errors[0]
            rows = []
            for slot in range(len(self.endpoints)):
                lanes = [row for row in lane_rows if row['endpoint_slot'] == slot]
                rows.append(dict(endpoint_slot=slot, lanes=[row['lane'] for row in lanes],
                    completed=len(lanes), queue_seconds=sum(row['queue_seconds'] for row in lanes),
                    elapsed_seconds=sum(row['elapsed_seconds'] for row in lanes),
                    renderer_runtime_ms=sum(row['renderer_runtime_ms'] for row in lanes)))
            record = dict(step=step, completed=len(results), wall_seconds=time.perf_counter()-started,
                endpoints=rows, lanes=lane_rows)
            if self.shard:
                record.update(started_monotonic_s=started_monotonic, finished_monotonic_s=time.monotonic())
            self.steps.append(record)
            self.trace.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            return results
        except Exception:
            self.failed = True
            raise
        finally:
            self._busy.release()

    def receipt(self):
        value = dict(schema=SHARDED_POOL_SCHEMA if self.shard else POOL_SCHEMA,
            routing=self.config['routing'], max_inflight_per_endpoint=1,
            endpoints=deepcopy(self.endpoints), lane_bindings=deepcopy(self.bindings),
            vector_steps_completed=len(self.steps), render_requests_completed=len(self.steps)*len(self.clients),
            failed=self.failed, trace_sha256=self.trace.hexdigest(), steps=deepcopy(self.steps),
            partial_step=deepcopy(self.partial), client_timing_semantics=deepcopy(CLIENT_TIMING_SEMANTICS))
        if self.shard:
            value.update(actor_shards=deepcopy(self.config['actor_shards']), execution_shard=deepcopy(self.shard),
                host_boot_id=self.host_boot_id, clock='CLOCK_MONOTONIC',
                concurrency_metric='overlapping_successful_render_requests_not_gpu_kernel_overlap')
        return value

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True)
            self.closed = True


def validate_pool_receipt(value, *, count, steps, renderer=None, cohort_index=None):
    sharded = isinstance(value, dict) and value.get('routing') == SHARDED_ROUTING
    if (not isinstance(value, dict) or value.get('schema') != (SHARDED_POOL_SCHEMA if sharded else POOL_SCHEMA)
            or value.get('routing') not in (ROUTING, SHARDED_ROUTING) or value.get('max_inflight_per_endpoint') != 1
            or value.get('failed') is not False or value.get('partial_step') is not None
            or value.get('vector_steps_completed') != steps
            or value.get('render_requests_completed') != count*steps
            or not _sha(value.get('trace_sha256'))):
        raise ValueError('scaled observation lacks a complete exact endpoint-pool receipt')
    if ('client_timing_semantics' in value
            and value['client_timing_semantics'] != CLIENT_TIMING_SEMANTICS):
        raise ValueError('endpoint optional client timing semantics differ')
    config = dict(identity={}, endpoints=value.get('endpoints'), routing=value['routing'], max_inflight_per_endpoint=1)
    if sharded:
        config['actor_shards'] = value.get('actor_shards')
    validate_pool_config(config)
    if renderer is not None and any(config[key] != renderer.get(key) for key in config if key != 'identity'):
        raise ValueError('endpoint receipt differs from frozen endpoint admission')
    if sharded:
        shard = value.get('execution_shard')
        if not isinstance(shard, dict):
            raise ValueError('endpoint receipt is missing exact actor shard evidence')
        expected = execution_shard(config, cohort_index=shard.get('cohort_index'), physical_gpu=shard.get('physical_gpu'))
        if (shard != expected or any(type(slot) is not int for slot in shard.get('endpoint_slots', []))
                or (cohort_index is not None and shard['cohort_index'] != cohort_index)):
            raise ValueError('endpoint receipt actor shard differs from the original cohort')
        if (not isinstance(value.get('host_boot_id'), str)
                or re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value['host_boot_id']) is None
                or value.get('clock') != 'CLOCK_MONOTONIC'
                or value.get('concurrency_metric') != 'overlapping_successful_render_requests_not_gpu_kernel_overlap'):
            raise ValueError('sharded request-concurrency evidence lacks host/clock identity')
        slots = shard['endpoint_slots']
    else:
        if {'actor_shards', 'execution_shard'} & set(value):
            raise ValueError('unsharded receipt cannot carry actor shard evidence')
        slots = list(range(len(config['endpoints'])))
    endpoints = value['endpoints']
    bindings = value.get('lane_bindings')
    if not isinstance(bindings, list) or len(bindings) != count:
        raise ValueError('endpoint lane coverage differs')
    for lane, row in enumerate(bindings):
        slot = slots[lane % len(slots)]
        if (not isinstance(row, dict) or set(row) != {'lane', 'endpoint_slot', 'channel', 'server_instance'}
                or row['lane'] != lane or row['endpoint_slot'] != slot
                or row['server_instance'] != endpoints[slot]['server_instance']
                or not isinstance(row['channel'], str) or re.fullmatch('[0-9a-f]{32}', row['channel']) is None):
            raise ValueError('endpoint deterministic route/live channel differs')
    if len({row['channel'] for row in bindings}) != count:
        raise ValueError('duplicate endpoint lane channel')
    records = value.get('steps')
    if not isinstance(records, list) or len(records) != steps:
        raise ValueError('endpoint per-step evidence missing')
    trace = hashlib.sha256()
    for step, record in enumerate(records):
        if (not isinstance(record, dict) or record.get('step') != step
                or record.get('completed') != count or not _nonnegative(record.get('wall_seconds'))
                or not isinstance(record.get('lanes'), list) or len(record['lanes']) != count
                or not isinstance(record.get('endpoints'), list) or len(record['endpoints']) != len(endpoints)):
            raise ValueError('endpoint step cardinality/timing differs')
        if sharded and (any(not _nonnegative(record.get(key)) for key in ('started_monotonic_s', 'finished_monotonic_s'))
                or record['finished_monotonic_s'] < record['started_monotonic_s']):
            raise ValueError('sharded vector request interval differs')
        for lane, row in enumerate(record['lanes']):
            if (not isinstance(row, dict) or row.get('lane') != lane or row.get('state') != 'complete'
                    or row.get('endpoint_slot') != bindings[lane]['endpoint_slot']
                    or row.get('channel') != bindings[lane]['channel'] or row.get('sequence') != step
                    or any(not _nonnegative(row.get(key)) for key in (
                        'queue_seconds', 'elapsed_seconds', 'renderer_runtime_ms'))):
                raise ValueError('endpoint completed-lane evidence differs')
            optional_timings = set(CLIENT_TIMING_FIELDS) & set(row)
            if optional_timings and (value.get('client_timing_semantics') != CLIENT_TIMING_SEMANTICS
                    or any(not _nonnegative(row[key]) for key in optional_timings)):
                raise ValueError('endpoint optional client timing evidence differs')
            if sharded and (any(not _nonnegative(row.get(key)) for key in
                    ('request_started_monotonic_s', 'request_finished_monotonic_s'))
                    or not record['started_monotonic_s'] <= row['request_started_monotonic_s'] <=
                        row['request_finished_monotonic_s'] <= record['finished_monotonic_s']):
                raise ValueError('sharded lane request interval differs')
        for slot, row in enumerate(record['endpoints']):
            lanes = [lane for lane in range(count) if slots[lane % len(slots)] == slot]
            if (not isinstance(row, dict) or row.get('endpoint_slot') != slot
                    or row.get('lanes') != lanes or row.get('completed') != len(lanes)
                    or any(not _nonnegative(row.get(key)) for key in (
                        'queue_seconds', 'elapsed_seconds', 'renderer_runtime_ms'))):
                raise ValueError('endpoint per-step counts/timing differ')
            for key in ('queue_seconds', 'elapsed_seconds', 'renderer_runtime_ms'):
                if not math.isclose(row[key], sum(record['lanes'][lane][key] for lane in lanes), abs_tol=1e-9):
                    raise ValueError('endpoint aggregate timing differs from lanes')
            if sharded and any(record['lanes'][a]['request_finished_monotonic_s'] >
                    record['lanes'][b]['request_started_monotonic_s'] for a, b in zip(lanes, lanes[1:])):
                raise ValueError('one batch-one renderer cannot have overlapping local requests')
        trace.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    if trace.hexdigest() != value['trace_sha256']:
        raise ValueError('endpoint timing/sequence trace differs')
    return deepcopy(value)


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_timing_receipt(value, *, steps):
    keys = ('preprocessing_seconds', 'render_wait_seconds', 'postprocessing_seconds')
    if (not isinstance(value, dict) or value.get('schema') != TIMING_SCHEMA
            or value.get('vector_steps_completed') != steps
            or any(not _nonnegative(value.get(key)) for key in keys)
            or not isinstance(value.get('steps'), list) or len(value['steps']) != steps):
        raise ValueError('scaled observation lacks phase timing receipt')
    for step, row in enumerate(value['steps']):
        if not isinstance(row, dict) or row.get('step') != step or any(not _nonnegative(row.get(k)) for k in keys):
            raise ValueError('scaled observation phase timing differs')
    for key in keys:
        if not math.isclose(value[key], sum(row[key] for row in value['steps']), abs_tol=1e-9):
            raise ValueError('scaled observation phase timing sum differs')
    return deepcopy(value)
