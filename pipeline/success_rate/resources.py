"""Consolidated success evaluation resources; versioned protocols remain distinct."""
from __future__ import annotations


# ---- scaled success resources ----

import math as math

from pipeline.dagger.rtx.code.src.dagger_rtx import phase_resource_guard as memory_guard

GIB = 1 << 30
SCHEMA = "roborender.scaled-success-resource-sample.v1"
SUMMARY_SCHEMA = "roborender.scaled-success-resource-summary.v1"


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def summarize(samples, *, native_owners, renderer_owners, started_unix, ended_unix):
    """Require continuously attributed RTX/H200 headroom; reject missing data.

    Owner descriptors are frozen before the probe. Renderer descriptors bind
    separate GPUs, not just different ports pointing at the same GPU/session.
    Reclaimable page cache is not private memory or added to MemAvailable.
    """
    for name, owners in (("native", native_owners), ("renderer", renderer_owners)):
        if not isinstance(owners, list) or not owners:
            raise ValueError(f"explicit {name} owners required")
        for owner in owners:
            if (not isinstance(owner, dict)
                    or set(owner) != {"owner_id", "gpu_uuid", "source_sha256"}
                    or any(not isinstance(v, str) or not v for v in owner.values())):
                raise ValueError("owner identity is incomplete")
            sha = owner["source_sha256"]
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("source digest is invalid")
        for field in ("owner_id", "gpu_uuid"):
            if len({owner[field] for owner in owners}) != len(owners):
                raise ValueError("duplicate GPU/process owners cannot certify parallel capacity")
    if len(native_owners) > 2 or len(renderer_owners) > 8:
        raise ValueError("outside the authorized two-RTX/eight-H200 topology")
    if {x['gpu_uuid'] for x in native_owners} & {x['gpu_uuid'] for x in renderer_owners}:
        raise ValueError("RTX and H200 GPU attribution overlaps")
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (started_unix, ended_unix))
            or ended_unix <= started_unix or not isinstance(samples, list) or len(samples) < 3):
        raise ValueError("full probe interval and at least three real samples required")
    if any(not isinstance(x, dict) for x in samples):
        raise ValueError("resource samples must be actual structured observations")
    times = [x.get("at_unix") for x in samples]
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in times)
            or any(b <= a or b-a > 10 for a, b in zip(times, times[1:]))
            or times[0] > started_unix or times[-1] < ended_unix
            or started_unix-times[0] > 10 or times[-1]-ended_unix > 10):
        raise ValueError("resource samples must bracket the whole probe without gaps over10s")
    native_expected = {x["owner_id"]: x for x in native_owners}
    renderer_expected = {x["owner_id"]: x for x in renderer_owners}
    peaks = {x["owner_id"]: 0.0 for x in native_owners + renderer_owners}
    private_peaks = {x["owner_id"]: 0 for x in native_owners}
    pressure_windows = {x["owner_id"]: memory_guard.PressureWindow() for x in native_owners}
    previous_native = {}
    initial_limit_events = {}
    min_host = 1 << 63
    for sample in samples:
        if sample.get("schema") != SCHEMA or not _integer(sample.get("host_available_bytes")):
            raise ValueError("actual host RAM telemetry required")
        min_host = min(min_host, sample["host_available_bytes"])
        if min_host < 12 * GIB:
            raise ValueError("qualification requires at least12GiB host headroom")
        for kind, expected in (("native", native_expected), ("renderers", renderer_expected)):
            rows = sample.get(kind)
            if (not isinstance(rows, list) or len(rows) != len(expected)
                    or any(not isinstance(x, dict) for x in rows)
                    or {x.get("owner_id") for x in rows} != set(expected)):
                raise ValueError("all exact owners must be sampled together")
            for row in rows:
                owner = expected[row["owner_id"]]
                if (any(row.get(k) != v for k, v in owner.items())
                        or row.get("alive") is not True or row.get("sampling_error")):
                    raise ValueError("owner changed, disappeared or cannot be observed")
                used, total = row.get("vram_used_bytes"), row.get("vram_total_bytes")
                if not _integer(used) or not _integer(total, 1) or used >= .75 * total:
                    raise ValueError("at least25% device-memory headroom required")
                peaks[row["owner_id"]] = max(peaks[row["owner_id"]], used / total)
                if kind == "native":
                    counters = [row.get(k) for k in ("anon_bytes", "shmem_bytes", "kernel_bytes")]
                    if (any(not _integer(x) for x in counters)
                            or row.get("memory_limit_bytes") != 21 * GIB
                            or row.get("swap_bytes") != 0 or row.get("swap_limit_bytes") != 0
                            or row.get("oom_events") != 0 or row.get("oom_kill_events") != 0
                            or not _integer(row.get("limit_hit_events"))
                            or row.get("thp_policy") != "disable-v1"):
                        raise ValueError("native memory, swap, OOM or THP policy differs")
                    raw = row.get('safety_sample')
                    if not isinstance(raw, dict):
                        raise ValueError('actual v3 memory/PSI/cgroup sample required')
                    if (any(raw.get(k) != row[k] for k in ('memory_limit_bytes', 'swap_bytes', 'swap_limit_bytes'))
                            or raw.get('events') != dict(max=row['limit_hit_events'],
                                oom=row['oom_events'], oom_kill=row['oom_kill_events'])
                            or raw.get('host_available_bytes', 0) < 12*GIB):
                        raise ValueError('flat resource telemetry differs from actual v3 counters')
                    try:
                        memory_guard.classify_sample(raw, policy_version=memory_guard.SUSTAINED_VERSION)
                        memory_guard.validate_transition(previous_native.get(row['owner_id']), raw)
                        pressure_windows[row['owner_id']].observe(raw, sample['at_unix'])
                    except RuntimeError as exc:
                        raise ValueError('actual v3 memory/PSI safety failed') from exc
                    initial_limit_events.setdefault(row['owner_id'], raw['events']['max'])
                    previous_native[row['owner_id']] = raw
                    private = sum(counters)
                    if private >= 17 * GIB:
                        raise ValueError("at least4GiB cgroup private-memory margin required")
                    private_peaks[row["owner_id"]] = max(private_peaks[row["owner_id"]], private)
    return dict(schema=SUMMARY_SCHEMA, resources_passed=True,
                production_ready=False, full_native_evaluation_receipts_required=True,
                native_owners=native_owners, renderer_owners=renderer_owners,
                sample_count=len(samples), started_unix=started_unix, ended_unix=ended_unix,
                minimum_host_available_bytes=min_host, maximum_vram_fraction=peaks,
                maximum_native_private_bytes=private_peaks,
                native_memory_policy=memory_guard.SUSTAINED_VERSION,
                initial_limit_events=initial_limit_events,
                final_limit_events={key: value['events']['max'] for key, value in previous_native.items()},
                interval_limit_events={key: value['events']['max']-initial_limit_events[key]
                                       for key, value in previous_native.items()},
                pressure_windows={key: window.summary() for key, window in pressure_windows.items()},
                compute_release_authorized=False, artificial_gpu_activity=False)


# ---- scaled success qualification ----

import math as qualification_math

from pipeline.success_rate.scaled_success_resources import summarize as qualification_summarize

qualification_SCHEMA = 'roborender.scaled-success-qualification.v1'


def qualification_request_concurrency(pools):
    """Measured overlapping successful RPCs, not GPU-kernel concurrency.

    Both native actors live on one attested Leo boot. Their monotonic timestamps
    are comparable; unrelated-host clocks or wall timestamps are not.
    """
    boots = {pool.get('host_boot_id') for pool in pools}
    if len(boots) != 1 or not next(iter(boots), None):
        raise ValueError('paired renderer timing requires one actual Leo boot')
    events = []
    for pool in pools:
        if pool.get('clock') != 'CLOCK_MONOTONIC':
            raise ValueError('paired renderer timing must use the same monotonic clock')
        for step in pool['steps']:
            for lane in step['lanes']:
                begin, end = lane.get('request_started_monotonic_s'), lane.get('request_finished_monotonic_s')
                if (any(type(t) not in (int, float) or not qualification_math.isfinite(t) for t in (begin, end))
                        or begin < 0 or end <= begin or lane.get('state') != 'complete'):
                    raise ValueError('actual successful render-request intervals required')
                events.extend(((begin, 1, lane['endpoint_slot']), (end, -1, lane['endpoint_slot'])))
    active, maximum = set(), 0
    for _, direction, slot in sorted(events):  # ends before starts at equal time
        if direction == 1:
            if slot in active:
                raise ValueError('one renderer received overlapping requests despite exclusive shards')
            active.add(slot)
            maximum = max(maximum, len(active))
        else:
            if slot not in active:
                raise ValueError('renderer request interval is inconsistent')
            active.remove(slot)
    return maximum


def qualification_validate_qualification(proof, *, source_sha256, panel, config):
    from pipeline.success_rate import scaled_success_contract as contract
    from pipeline.success_rate.rendered_student_success_contract import validate_config
    contract.require_panel(panel)
    validate_config(config, panel)
    batch = panel['scaled_protocol']['batch_size']
    arm = proof.get('arm') if isinstance(proof, dict) else None
    if (not isinstance(proof, dict) or proof.get('schema') != qualification_SCHEMA
            or proof.get('source_sha256') != source_sha256 or proof.get('batch_size') != batch
            or proof.get('checkpoint_sha256') != config['checkpoint_sha256']
            or proof.get('compute_release_authorized') is not False
            or proof.get('cpu_tests_passed') is not True or arm not in ('bc', 'dagger')):
        raise ValueError('actual source/checkpoint-bound native qualification required')
    if (len(config['renderer'].get('endpoints', [])) != 8
            or config['renderer'].get('routing') != 'cohort_parity_actor_shards_v1'):
        raise ValueError('production qualification requires the requested eight-H200 sharded topology')
    trials = proof.get('trials')
    sizes = [n for n in (4, 8, 16) if n <= batch] + [batch]
    if not isinstance(trials, list) or len(trials) != len(sizes):
        raise ValueError('measured lower-batch ladder and repeated target-batch probe required')
    repeated_states = []
    tail = proof.get('tail_trial')
    if batch == 16 and not isinstance(tail, dict):
        raise ValueError('real seven-case terminal cohort qualification required')
    entries = [(trial, size, False) for trial, size in zip(trials, sizes)]
    if batch == 16:
        entries.append((tail, 16, True))
    for trial_index, (trial, size, is_tail) in enumerate(entries):
        qpanel, qconfig = trial['panel'], trial['config']
        contract.require_panel(qpanel)
        validate_config(qconfig, qpanel)
        if (qpanel['scaled_protocol']['purpose'] != 'qualification'
                or qpanel['scaled_protocol']['batch_size'] != size
                or qpanel['episode_count'] != (7 if is_tail else 2*size)
                or qpanel['scaled_protocol'].get('qualification_variant', 'paired') != ('tail7' if is_tail else 'paired')
                or qpanel['scaled_protocol']['source_panel_sha256'] != panel['scaled_protocol']['source_panel_sha256']
                or qconfig['checkpoint_sha256'] != config['checkpoint_sha256']
                or qconfig['renderer'] != config['renderer']
                or qconfig['anchors'] != config['anchors']):
            raise ValueError('qualification changed checkpoint, renderer, anchors or prescribed batch')
        owners = trial.get('owners')
        gpu_indices = [0] if is_tail else [0, 1]
        if not isinstance(owners, list) or [r.get('physical_gpu') for r in owners] != gpu_indices:
            raise ValueError('both physical RTX GPUs must complete each qualification size')
        if len({r.get('worker_instance_id') for r in owners}) != len(gpu_indices):
            raise ValueError('two independent native owners required')
        values = []
        native_owners = []
        states = []
        used_endpoints = set()
        pools = []
        for gpu, row in enumerate(owners):
            receipt = row['native_receipt']
            if (receipt.get('complete') is not True or receipt.get('source_sha256') != source_sha256
                    or receipt.get('optimizer_steps') != 0 or receipt.get('native_initializations') != 1
                    or receipt.get('resource_guard', {}).get('passed') is not True):
                raise ValueError('native evaluation receipt is incomplete or source/resource identity differs')
            result = receipt['result']
            contract.validate_result(qpanel, result, config['checkpoint_sha256'], rendered_config=qconfig)
            values.append(result)
            native_owners.append(dict(owner_id=row['worker_instance_id'],
                gpu_uuid=row['gpu_uuid'], source_sha256=source_sha256))
            # Exactly one original full qualification cohort per owner.
            partition = result['partition_result']
            if partition['partition']['cohort_indices'] != [gpu] or partition['arm'] != arm:
                raise ValueError('native qualification omitted or duplicated a cohort')
            cohort = partition['cohort_receipts'][0]['result']
            states.append(cohort['scaled_initial_state_attestation'])
            pool = cohort['rendered_observation_receipt']['renderer_endpoint_pool']
            pools.append(pool)
            for step in pool['steps']:
                active = {row['endpoint_slot'] for row in step['endpoints'] if row['completed'] > 0}
                expected = set(range(4*gpu, 4*gpu+4))
                if active != expected:
                    raise ValueError('every paired step must genuinely render on its four dedicated H200s')
                used_endpoints |= active
        if used_endpoints != set(range(4 if is_tail else 8)):
            raise ValueError('all eight H200s must perform genuine qualification renders')
        # Cold scene rebuilds are intentionally serialized on Leo. Require
        # full eight-renderer overlap on the repeated, already-warm target;
        # every earlier size must still execute all eight real endpoints.
        required_concurrency = 8 if trial_index == len(sizes)-1 else 4
        if qualification_request_concurrency(pools) < required_concurrency:
            raise ValueError('requested renderer RPC concurrency has not been observed')
        contract.aggregate_results(qpanel, values, config['checkpoint_sha256'], rendered_config=qconfig, arm=arm)
        resource = trial['resources']
        actual = qualification_summarize(resource['samples'], native_owners=native_owners,
            renderer_owners=resource['renderer_owners'], started_unix=trial['started_unix'],
            ended_unix=trial['ended_unix'])
        endpoints = qconfig['renderer'].get('endpoints', [])
        pinned = {(e['gpu_uuid'], e['source_sha256'], e['server_instance']) for e in endpoints}
        sampled = {(e['gpu_uuid'], e['source_sha256'], e['owner_id']) for e in resource['renderer_owners']}
        if pinned != sampled or len(pinned) != 8:
            raise ValueError('eight actual independent H200 renderers must be bound and sampled')
        if resource.get('summary') != actual:
            raise ValueError('stored headroom summary differs from measured full-probe samples')
        if size == batch and not is_tail:
            repeated_states.append(states)
    if len(repeated_states) != 2 or repeated_states[0] != repeated_states[1]:
        raise ValueError('target-batch repeated initialization evidence differs')
    return proof
