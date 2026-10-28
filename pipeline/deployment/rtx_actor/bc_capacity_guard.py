"""Shared B1..B6 pre-save RAM gate; native capture/writer are unchanged.

The caller's separate protocol declares qualification or production purpose.
This safety gate alone never grants production data admission.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

GIB = 2**30


class CapacityRejected(RuntimeError):
    pass


def projected_save_gate(*, num_envs, current_bytes, limit_bytes, host_available_bytes,
                        swap_bytes, events):
    if (type(num_envs) is not int or not 1 <= num_envs <= 6 or limit_bytes not in (40 * GIB, 43 * GIB)
            or any(type(v) is not int or v < 0 for v in (current_bytes, host_available_bytes, swap_bytes))
            or not isinstance(events, dict) or not {'max', 'oom', 'oom_kill'} <= set(events)):
        raise CapacityRejected('missing or unexpected bounded qualification memory counters')
    extra = (num_envs + 2) * GIB
    decision = {'schema': 'roborender.bc-capacity-presave.v1', 'num_envs': num_envs,
        'memory_current_bytes': current_bytes, 'memory_max_bytes': limit_bytes,
        'host_available_bytes': host_available_bytes, 'owned_swap_bytes': swap_bytes,
        'events': events, 'projected_additional_bytes': extra, 'cgroup_margin_bytes': 2 * GIB,
        'host_floor_bytes': 8 * GIB, 'production_admitted': False,
        'projection_is_proven_upper_bound': False}
    reasons = []
    if swap_bytes or any(events[k] for k in ('max', 'oom', 'oom_kill')):
        reasons.append('owned swap or cgroup pressure/OOM was observed')
    if current_bytes + extra + 2 * GIB > limit_bytes:
        reasons.append('insufficient cgroup headroom for original save allocations')
    if host_available_bytes < extra + 8 * GIB:
        reasons.append('insufficient host headroom for original save allocations')
    decision['admitted'] = not reasons
    decision['reasons'] = reasons
    return decision


def validate_buffers(trainer, expected_num_envs):
    b = int(trainer.env.num_envs)
    if (type(expected_num_envs) is not int or not 1 <= expected_num_envs <= 6
            or b != expected_num_envs or len(trainer._depth_chunk) != 81 or len(trainer._robot_rgb_chunk) != 81
            or len(trainer._views_chunk) != 81 or trainer.vg_mode != 'collect'
            or trainer.vg_rollout_action_source != 'teacher'
            or not trainer.vg_save_raw_depth or not trainer.vg_save_raw_depth_native
            or trainer.vg_save_sim_rgb or list(trainer.vg_control_res) != [240, 416]):
        raise CapacityRejected('required original full BC buffers or recipe changed')
    for depth, robot, views in zip(trainer._depth_chunk, trainer._robot_rgb_chunk, trainer._views_chunk):
        if tuple(depth.shape) != (b, 480, 832, 1) or tuple(robot.shape) != (b, 480, 832, 3):
            raise CapacityRejected('unexpected primary camera buffers')
        if set(views) != {'head_color', 'hand_left', 'hand_right'}:
            raise CapacityRejected('unexpected conditioning view set')
        for row in views.values():
            if (tuple(row['depth'].shape) != (b, 480, 832, 1)
                    or tuple(row['robot_rgb'].shape) != (b, 480, 832, 3)
                    or row.get('rgb') is not None):
                raise CapacityRejected('unexpected native conditioning buffers')
    return b


def guarded_save(trainer, original, *, snapshot, host_available, limit_bytes, report, expected_num_envs=6):
    b = validate_buffers(trainer, expected_num_envs)
    before = snapshot()
    decision = projected_save_gate(num_envs=b, current_bytes=before['memory_current'],
        limit_bytes=before['memory_max'], host_available_bytes=host_available(),
        swap_bytes=before['memory_swap.current'], events=before['events'])
    if before['memory_max'] != limit_bytes:
        raise CapacityRejected('running cgroup cap differs from frozen launch')
    report('presave', {**decision, 'memory': before,
        'vector_steps': trainer._bc_ledger.vector_steps,
        'emitted_chunks_per_lane': list(trainer._bc_ledger.emitted_counts)})
    if not decision['admitted']:
        raise CapacityRejected('; '.join(decision['reasons']))
    completed = False
    try:
        result = original()  # Original method, including its native quota exception.
        completed = True
        return result
    except BaseException as exc:
        completed = type(exc).__name__ == '_QuotaReached'
        raise
    finally:
        after = snapshot()
        stat = after.get('stat', {})
        report('postsave', {'completed_original_save': completed, 'memory': after,
            'vector_steps': trainer._bc_ledger.vector_steps,
            'emitted_chunks_per_lane': list(trainer._bc_ledger.emitted_counts),
            'memory_current_bytes': after.get('memory_current'), 'memory_peak_bytes': after.get('memory_peak'),
            'anon_bytes': stat.get('anon'), 'shmem_bytes': stat.get('shmem'),
            'kernel_bytes': stat.get('kernel'), 'file_bytes': stat.get('file'),
            'host_available_bytes': host_available()})


def plateau_gate(events, *, num_envs=6, cap_bytes, completion_field='completed_original_save'):
    """Predeclared repeated-buffer gate, not a long-campaign memory guarantee."""
    if completion_field not in ('completed_original_save', 'completed_bounded_save'):
        raise ValueError('unknown complete writer event field')
    rows = [r for r in events if r.get('phase') == 'postsave' and r.get(completion_field) is True]
    if len(rows) < 2 or any(v < 2 for v in rows[-1]['emitted_chunks_per_lane']):
        raise CapacityRejected('need two complete native emissions per lane and two full-window saves')
    for row in rows:
        if (len(row['emitted_chunks_per_lane']) != num_envs or row['vector_steps'] % 81
                or row['host_available_bytes'] < 8 * GIB
                or row['memory_current_bytes'] + 2 * GIB > cap_bytes
                or row['memory_peak_bytes'] > cap_bytes
                or row['memory']['memory_swap.current']
                or any(row['memory']['events'][k] for k in ('max', 'oom', 'oom_kill'))):
            raise CapacityRejected('flush memory/host/OOM/swap safety evidence differs')
    for prior, last in zip(rows, rows[1:]):
        if (last['vector_steps'] - prior['vector_steps'] < 81
                or any(b < a for a, b in zip(prior['emitted_chunks_per_lane'], last['emitted_chunks_per_lane']))):
            raise CapacityRejected('full writer calls must advance81 steps and preserve monotonic emission counters')
    prior, last = rows[-2:]
    private = lambda r: sum(r[k] for k in ('anon_bytes', 'shmem_bytes', 'kernel_bytes'))
    measurements = {'private_bytes': (private(prior), private(last)),
                    'current_bytes': (prior['memory_current_bytes'], last['memory_current_bytes'])}
    for name, (old, new) in measurements.items():
        if new - old > 256 * 1024**2 + old * .10:
            raise CapacityRejected('predeclared last-two-flush plateau exceeded: ' + name)
    return {'rule': 'growth<=256MiB+10percent_prior for private(anon+shmem+kernel) and total current',
            'per_flush_memory': rows, 'runtime_guard_enabled': True,
            'active_guard_minimum_host_available_bytes': 8 * GIB,
            'bounded_repeated_buffer_only': True}


def _native_class():
    # Hydra resolves this only after Kit initialization; CPU imports stay light.
    from isaac6_bc_session import Isaac6BCSessionTrainer, memory_snapshot
    if (os.environ.get('ROBORENDER_BC_CAPACITY_ACK') != 'FULL_BC_BUFFERS_GUARDED_RUNTIME'
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != os.environ.get('ROBORENDER_BC_CAPACITY_GUARD_SHA256')):
        raise CapacityRejected('qualification guard source/ack differs')
    limit = int(os.environ['ROBORENDER_BC_CAPACITY_MEMORY_BYTES'])
    expected_num_envs = int(os.environ['ROBORENDER_BC_GUARD_NUM_ENVS'])
    if limit not in (40 * GIB, 43 * GIB):
        raise CapacityRejected('unsupported qualification memory cap')
    if not 1 <= expected_num_envs <= 6:
        raise CapacityRejected('native guard requires explicit B1..B6; capacity is qualified per actual B')

    def host_available():
        values = dict(line.split(':', 1) for line in Path('/run/host-meminfo').read_text().splitlines())
        return int(values['MemAvailable'].split()[0]) * 1024

    def report(phase, value):
        row = {'phase': phase, 'monotonic_seconds': time.monotonic(), **value}
        with Path('/dagger-runtime/capacity-save-events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps({'phase': 'bc_capacity_' + phase, **value}), flush=True)

    class GuardedBCSessionTrainer(Isaac6BCSessionTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._guard_current_trainer()

        def _guard_current_trainer(self):
            trainer = self.trainer
            original = trainer._save_collected_chunks
            trainer._save_collected_chunks = lambda: guarded_save(trainer, original,
                snapshot=memory_snapshot, host_available=host_available,
                limit_bytes=limit, report=report, expected_num_envs=expected_num_envs)

        def _rebuild(self, request):
            super()._rebuild(request)
            self._guard_current_trainer()

    return GuardedBCSessionTrainer


def __getattr__(name):
    if name != 'GuardedBCSessionTrainer':
        raise AttributeError(name)
    value = _native_class()
    globals()[name] = value
    return value
