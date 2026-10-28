"""Shared B1..B6 host RAM guards for qualification and eventual production.

This module performs no process launch, source staging, collection or mutation.
Both callers must bind these exact bytes in their executed runtime contract.
Qualification-only quotas/time limits remain in their separate wrapper.
"""
from __future__ import annotations

GIB = 2**30
HOST_FLOOR_BYTES = 8 * GIB
ALLOWED_MEMORY_GIB = (40, 43)


def exclusive_native_owner(names, expected_name=None):
    """A single-collector receipt must observe a single native-image owner.

    The host launcher supplies Docker's read-only ancestor-image inventory.
    Existing unrelated owners are rejected, never stopped.
    """
    if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))
            or set(names) != (set() if expected_name is None else {expected_name})):
        raise RuntimeError('single native-image ownership differs; do not stop other owners')
    return len(names)


def augment_command(command, *, guard_path, guard_sha256, memory_gib, image, num_envs=6):
    """The same command transformation is used by both launch purposes."""
    if (memory_gib not in ALLOWED_MEMORY_GIB or type(num_envs) is not int
            or not 1 <= num_envs <= 6):
        raise ValueError('guarded runtime requires explicit B1..B6 and40/43GiB cap')
    cmd = list(command)
    for flag in ('--memory', '--memory-swap'):
        if cmd.count(flag) != 1 or cmd[cmd.index(flag) + 1] != '40g':
            raise ValueError('base command cap differs')
        cmd[cmd.index(flag) + 1] = f'{memory_gib}g'
    targets = [i for i, v in enumerate(cmd) if v.startswith('trainer._target_=')]
    paths = [i for i, v in enumerate(cmd) if v.startswith('PYTHONPATH=')]
    if len(targets) != 1 or cmd[targets[0]] != 'trainer._target_=isaac6_bc_session.Isaac6BCSessionTrainer' or len(paths) != 1:
        raise ValueError('unexpected native entrypoint')
    cmd[targets[0]] = 'trainer._target_=bc_capacity_guard.GuardedBCSessionTrainer'
    cmd[paths[0]] = cmd[paths[0]].replace('PYTHONPATH=', 'PYTHONPATH=/capacity-guard:', 1)
    at = cmd.index(image)
    cmd[at:at] = ['--mount', f'type=bind,src={guard_path},dst=/capacity-guard/bc_capacity_guard.py,readonly',
        '--mount', 'type=bind,src=/proc/meminfo,dst=/run/host-meminfo,readonly',
        '--env', 'ROBORENDER_BC_CAPACITY_ACK=FULL_BC_BUFFERS_GUARDED_RUNTIME',
        '--env', 'ROBORENDER_BC_GUARD_NUM_ENVS=' + str(num_envs),
        '--env', 'ROBORENDER_BC_CAPACITY_GUARD_SHA256=' + guard_sha256,
        '--env', 'ROBORENDER_BC_CAPACITY_MEMORY_BYTES=' + str(memory_gib * GIB)]
    return cmd


def admission(available_kib, memory_gib):
    if (type(available_kib) is not int or type(memory_gib) is not int
            or memory_gib not in ALLOWED_MEMORY_GIB
            or available_kib < (memory_gib + 8) * 1024**2):
        raise RuntimeError('BC cap exceeds actual MemAvailable minus8GiB; no launch')


def runtime_guard(host, sample, *, recovered_limit_cap_bytes=None):
    """Strict by default; a bounded diagnostic may observe recovered max events.

    A cumulative memory.events.max count is not an OOM. The opt-in only permits
    it with measured current headroom; it does not grant capacity certification.
    """
    if type(host.get('MemAvailable')) is not int or host['MemAvailable'] < 8 * 1024**2:
        raise RuntimeError('host memory fell below8GiB floor')
    if (sample.get('sampling_error') or not sample.get('pid') or 'memory_swap.current' not in sample
            or type(sample['memory_swap.current']) is not int or sample['memory_swap.current'] < 0
            or any(type(sample.get('events', {}).get(k)) is not int or sample['events'][k] < 0
                   for k in ('max', 'oom', 'oom_kill'))):
        raise RuntimeError('owned live cgroup counters unavailable')
    if sample.get('oom_killed') or sample['events']['oom_kill']:
        raise RuntimeError('owned cgroup OOM kill observed')
    if sample['events']['oom']:
        raise RuntimeError('owned cgroup OOM allocation event observed')
    if sample['memory_swap.current']:
        raise RuntimeError('owned cgroup swap usage observed: ' + str(sample['memory_swap.current']) + ' bytes')
    if recovered_limit_cap_bytes is not None:
        if (type(recovered_limit_cap_bytes) is not int
                or recovered_limit_cap_bytes not in tuple(v * GIB for v in ALLOWED_MEMORY_GIB)
                or any(type(sample.get(k)) is not int or sample[k] < 0
                       for k in ('memory_current', 'memory_peak'))):
            raise RuntimeError('recovered-limit diagnostic requires valid cap/current/peak counters')
        if (sample['memory_current'] + 2 * GIB > recovered_limit_cap_bytes
                or sample['memory_peak'] > recovered_limit_cap_bytes):
            raise RuntimeError('owned cgroup below2GiB current headroom or peak exceeded cap')
    if sample['events']['max']:
        if recovered_limit_cap_bytes is None:
            raise RuntimeError('owned cgroup memory.max limit event observed; strict capacity policy')
        return dict(status='WARNING', reason='recovered_memory_max_event',
                    max_events=sample['events']['max'], capacity_certified=False)
    return dict(status='OK')
