"""Continuous-collection-only full-buffer safety guard; no physics changes.

The original collector remains the capture/writer. This distinct profile does
not widen the historical B1..B6 guard or attest capacity without native evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

GIB = 2**30
ACK = "CONTINUOUS_TEACHER_FULL_BC_BUFFERS_NO_REPLAY_V1"


def memory_gate(snapshot, available_bytes, *, num_envs, cap_bytes, saving=False):
    if (type(num_envs) is not int or not 1 <= num_envs <= 40
            or cap_bytes not in (40 * GIB, 43 * GIB)
            or type(available_bytes) is not int or available_bytes < 0
            or snapshot.get('sampling_error') or snapshot.get('memory_max') != cap_bytes
            or any(type(snapshot.get(k)) is not int or snapshot[k] < 0
                   for k in ('memory_current', 'memory_peak', 'memory_swap.current'))
            or any(type(snapshot.get('events', {}).get(k)) is not int
                   for k in ('max', 'oom', 'oom_kill'))):
        raise RuntimeError('continuous BC memory counters/cap are unavailable or changed')
    # Capture can exhaust RAM before the first save. Guard each native capture,
    # in addition to the host's one-second observer and the full save barrier.
    # Four original depth/RGB buffers (primary plus three conditioning views),
    # with a conservative 3x transient allowance for the next capture.
    extra = ((num_envs + 2) * GIB if saving else
             max(GIB, 3 * num_envs * 4 * 480 * 832 * (4 + 3)))
    if (snapshot['memory_swap.current'] or any(snapshot['events'][k] for k in ('max', 'oom', 'oom_kill'))
            or snapshot['memory_current'] + extra + 2 * GIB > cap_bytes
            or snapshot['memory_peak'] > cap_bytes or available_bytes < extra + 8 * GIB):
        raise RuntimeError('continuous BC capture/save memory headroom, swap or OOM guard rejected')
    return dict(num_envs=num_envs, memory_max_bytes=cap_bytes,
                projected_additional_bytes=extra, host_available_bytes=available_bytes,
                admitted=True, projection_is_proven_upper_bound=False)


def validate_full_buffers(trainer, num_envs):
    if (int(trainer.env.num_envs) != num_envs or trainer.vg_mode != 'collect'
            or trainer.vg_rollout_action_source != 'teacher'
            or not trainer.vg_save_raw_depth or not trainer.vg_save_raw_depth_native
            or trainer.vg_save_sim_rgb or list(trainer.vg_control_res) != [240, 416]
            or any(len(getattr(trainer, key)) != 81
                   for key in ('_depth_chunk', '_robot_rgb_chunk', '_views_chunk'))):
        raise RuntimeError('continuous BC original full-buffer capture recipe differs')
    for depth, rgb, views in zip(trainer._depth_chunk, trainer._robot_rgb_chunk, trainer._views_chunk):
        if (tuple(depth.shape) != (num_envs, 480, 832, 1)
                or tuple(rgb.shape) != (num_envs, 480, 832, 3)
                or set(views) != {'head_color', 'hand_left', 'hand_right'}):
            raise RuntimeError('continuous BC original primary/view buffer shapes differ')
        for row in views.values():
            if (tuple(row['depth'].shape) != (num_envs, 480, 832, 1)
                    or tuple(row['robot_rgb'].shape) != (num_envs, 480, 832, 3)
                    or row.get('rgb') is not None):
                raise RuntimeError('continuous BC conditioning buffer shapes differ')


def guarded_save(trainer, original, *, snapshot, host_available, num_envs, cap_bytes, report):
    validate_full_buffers(trainer, num_envs)
    before = snapshot()
    decision = memory_gate(before, host_available(), num_envs=num_envs, cap_bytes=cap_bytes, saving=True)
    report('presave', dict(**decision, memory=before, vector_steps=trainer._bc_ledger.vector_steps,
                          emitted_chunks_per_lane=list(trainer._bc_ledger.emitted_counts)))
    complete = False
    try:
        value = original()
        complete = True
        return value
    finally:
        after = snapshot()
        stat = after.get('stat', {})
        report('postsave', dict(completed_original_save=complete, memory=after,
            vector_steps=trainer._bc_ledger.vector_steps,
            emitted_chunks_per_lane=list(trainer._bc_ledger.emitted_counts),
            memory_current_bytes=after.get('memory_current'), memory_peak_bytes=after.get('memory_peak'),
            anon_bytes=stat.get('anon'), shmem_bytes=stat.get('shmem'), kernel_bytes=stat.get('kernel'),
            file_bytes=stat.get('file'), host_available_bytes=host_available()))


def __getattr__(name):
    if name != 'GuardedContinuousBCTrainer':
        raise AttributeError(name)
    from isaac6_bc_production import Isaac6BCProductionTrainer, load_request
    from isaac6_bc_session import memory_snapshot
    request = load_request()
    if (request['schema'] != 'roborender.bc-native-continuous-request.v1'
            or os.environ.get('ROBORENDER_CONTINUOUS_BC_ACK') != ACK
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != os.environ.get('ROBORENDER_CONTINUOUS_BC_GUARD_SHA256')):
        raise RuntimeError('continuous native guard source/request/ack differs')
    cap = int(os.environ['ROBORENDER_CONTINUOUS_BC_MEMORY_BYTES'])
    b = len(request['motions'])

    def host_available():
        rows = dict(line.split(':', 1) for line in Path('/run/host-meminfo').read_text().splitlines())
        return int(rows['MemAvailable'].split()[0]) * 1024

    def report(phase, value):
        with Path('/dagger-runtime/capacity-save-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=phase, monotonic_seconds=time.monotonic(), **value),
                                    sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    class GuardedContinuousBCTrainer(Isaac6BCProductionTrainer):
        def policy_step(self, *args, **kwargs):
            if getattr(self, '_in_train_rollout', False):
                memory_gate(memory_snapshot(), host_available(), num_envs=b, cap_bytes=cap)
            return super().policy_step(*args, **kwargs)

        def _save_collected_chunks(self):
            return guarded_save(self, super()._save_collected_chunks, snapshot=memory_snapshot,
                                host_available=host_available, num_envs=b, cap_bytes=cap, report=report)

    globals()[name] = GuardedContinuousBCTrainer
    return GuardedContinuousBCTrainer
