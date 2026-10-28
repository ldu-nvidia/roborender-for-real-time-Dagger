"""Fail-closed memory admission for the explicitly scoped B32 learner diagnostic.

No batch splitting, precision change, activation-policy substitution, retry, or
renderer eviction. The cap reserves headroom; it is not a capacity certificate.
Actual paired resource sampling must still pass throughout the workload.
"""
from __future__ import annotations

import math

SCHEMA = 'roborender.diagnostic-learner-memory.v1'
GIB = 1 << 30
MAX_DEVICE_FRACTION = .75
EXTRA_MARGIN_BYTES = 2 * GIB


def admission(*, free_bytes, total_bytes, reserved_bytes):
    if (any(type(x) is not int for x in (free_bytes, total_bytes, reserved_bytes))
            or not 0 < free_bytes <= total_bytes or not 0 <= reserved_bytes <= total_bytes - free_bytes):
        raise ValueError('actual device free/total/current allocator counters required')
    external = total_bytes - free_bytes - reserved_bytes
    cap = math.floor(MAX_DEVICE_FRACTION * total_bytes) - external - EXTRA_MARGIN_BYTES
    # B32 has an observed ~77GB allocated peak in the separate BC diagnostic.
    # This floor only avoids a predictably inadequate attempt; it does not
    # assert that the un-recomputed DAgger graph fits the cap.
    if cap < 72 * GIB or cap <= reserved_bytes:
        raise RuntimeError('insufficient guarded B32 learner budget; no GPU attempt or fallback')
    return dict(schema=SCHEMA, total_bytes=total_bytes, free_before_bytes=free_bytes,
        reserved_before_bytes=reserved_bytes, external_before_bytes=external,
        allocator_cap_bytes=cap, allocator_fraction=cap / total_bytes,
        maximum_device_fraction=MAX_DEVICE_FRACTION,
        extra_margin_bytes=EXTRA_MARGIN_BYTES,
        batch_substitution_allowed=False, activation_policy_substitution_allowed=False,
        renderer_eviction_allowed=False, retry_allowed=False, compute_release_authorized=False,
        estimate_is_capacity_proof=False)


def begin(cuda):
    if cuda.device_count() != 1:
        raise RuntimeError('exact single-device learner placement required')
    free, total = cuda.mem_get_info(0)
    policy = admission(free_bytes=free, total_bytes=total, reserved_bytes=cuda.memory_reserved(0))
    cuda.set_per_process_memory_fraction(policy['allocator_fraction'], 0)
    cuda.reset_peak_memory_stats(0)
    return dict(policy=policy, observations=[])


def observe(context, cuda, phase):
    if context is None:
        return None
    free, total = cuda.mem_get_info(0)
    row = dict(phase=phase, free_bytes=free, total_bytes=total,
        allocated_bytes=cuda.memory_allocated(0), reserved_bytes=cuda.memory_reserved(0),
        peak_allocated_bytes=cuda.max_memory_allocated(0), peak_reserved_bytes=cuda.max_memory_reserved(0))
    policy = context['policy']
    if (any(type(row[k]) is not int or row[k] < 0 for k in row if k != 'phase')
            or total != policy['total_bytes'] or not 0 < free <= total
            or total - free >= MAX_DEVICE_FRACTION * total
            or row['allocated_bytes'] > row['reserved_bytes']
            or row['reserved_bytes'] > row['peak_reserved_bytes']
            or row['peak_allocated_bytes'] > row['peak_reserved_bytes']
            or row['peak_reserved_bytes'] > policy['allocator_cap_bytes']):
        raise RuntimeError('actual learner device/allocator headroom failed; no retry or substitution')
    context['observations'].append(row)
    return row
