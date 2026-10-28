"""Bounded teacher-only native capacity probe; not DAgger or a dataset producer.

This CPU-importable loop calls the same origin-attested batch capture/step
callbacks as production. It never retries an irreversible step, changes the
native reset sampler, opens an H200 connection, or publishes learner chunks.
"""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from dagger_rtx.contracts import StepKey, RAVEN_VIEW_ORDER


def step_acknowledgement(steps):
    if type(steps) is not int or steps not in (3, 10):
        raise ValueError("capacity qualification permits exactly 3 or 10 native steps")
    return f"I_ACKNOWLEDGE_{steps}_NATIVE_TEACHER_VECTOR_STEPS"


def run_teacher_capacity(batch, lifecycle, *, steps, acknowledgement, report):
    if acknowledgement != step_acknowledgement(steps):
        raise ValueError("explicit bounded native teacher-step acknowledgement required")
    return _run_teacher_steps(batch, lifecycle, steps=steps, report=report,
                              mode="teacher-capacity")


def run_teacher_camera_soak(batch, lifecycle, *, steps, acknowledgement, report):
    """Fresh diagnostic lifetime, not production replay or capacity admission.

    Exercise four window lengths without resetting at the 81-step boundaries.
    No H200/student/learner is involved, so a PASS cannot certify mixed-policy
    production. Episode resets remain exclusively native done/timeout resets.
    """
    if (type(steps) is not int or steps != 324 or batch.num_envs != 16
            or acknowledgement != "I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS"):
        raise ValueError("camera soak requires exactly B16 and 324 acknowledged teacher steps")
    return _run_teacher_steps(batch, lifecycle, steps=steps, report=report,
                              mode="teacher-camera-soak")


def run_recorded_action_camera_soak(batch, lifecycle, *, steps, acknowledgement, report, reference):
    from recorded_action_camera import ACK, MODE, RecordedActions
    if (type(steps) is not int or steps != 324 or batch.num_envs != 16
            or acknowledgement != ACK or not isinstance(reference, RecordedActions)):
        raise ValueError("recorded-action diagnostic requires fresh B16/324 and bound reference")
    return _run_teacher_steps(batch,lifecycle,steps=steps,report=report,mode=MODE,reference=reference)


def _run_teacher_steps(batch, lifecycle, *, steps, report, mode, reference=None):
    count = batch.num_envs
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("capacity probe batch size is outside its bounded scope")
    keys = [StepKey("leo-" + mode, 1, f"capacity-lane-{lane:03d}",
                    "window-000000", 0, f"episode-{lane:03d}-00000000", 0)
            for lane in range(count)]
    episode_counts = [0] * count
    bindings = dict(enumerate(keys))
    timings = []
    calls = 0
    done_count = timeout_count = 0
    sources = {"raytraced": 0, "generated_controls": 0}
    lifecycle.begin()
    try:
        for step in range(steps):
            started = time.monotonic()
            report({"phase": "capture_teacher_start", "step": step,
                    "completed_vector_steps": calls, "num_envs": count})
            if bindings:
                batch.bind_episodes(bindings)
            captures = list(batch.capture_batch(keys))
            if len(captures) != count or [c.key for c in captures] != keys:
                raise RuntimeError("native capacity capture lane/step identity mismatch")
            teachers = []
            for capture in captures:
                arrays = capture.arrays
                teacher = arrays["teacher_action"]
                homie = arrays["homie_action"]
                if (teacher.shape != (26,) or teacher.dtype != np.float32
                        or homie.shape != (15,) or homie.dtype != np.float32
                        or not np.isfinite(teacher).all()
                        or not np.array_equal(teacher[11:], homie)):
                    raise RuntimeError("native capacity teacher/Homie labels mismatch")
                actor = arrays["actor_obs"]
                if actor.shape != (113,) or actor.dtype != np.float32 or not np.isfinite(actor).all():
                    raise RuntimeError("native capacity actor observation invalid")
                for view in RAVEN_VIEW_ORDER:
                    depth, rgb = arrays[f"depth_ctrl__{view}"], arrays[f"rgb_robot__{view}"]
                    if (depth.shape != (240, 416) or depth.dtype != np.uint8
                            or rgb.shape != (240, 416, 3) or rgb.dtype != np.uint8):
                        raise RuntimeError("native capacity conditioning view shape/dtype invalid")
                source = capture.metadata["observation_source"]
                if source not in sources:
                    raise RuntimeError("native capacity observation source invalid")
                sources[source] += 1
                teachers.append(teacher)
            teachers = np.ascontiguousarray(np.stack(teachers), dtype=np.float32)
            actions = teachers if reference is None else reference.select(step,captures,teachers)
            if reference is not None and step < 129:
                report({"phase":"recorded_action_comparison", "completed_vector_steps":calls,
                        **reference.metrics[-1]})
            captured = time.monotonic()
            # A crash after this marker is uncertain. There is no resume/replay.
            report({"phase": "env_step_start", "step": step,
                    "completed_vector_steps": calls, "irreversible_step_intents": calls + 1})
            results = tuple(batch.step_batch(actions, teachers))
            calls += 1
            if len(results) != count:
                raise RuntimeError("native capacity transition count mismatch")
            if reference is not None:
                reference.observe(step,results)
            finished = time.monotonic()
            bindings = {}
            for lane, result in enumerate(results):
                if (type(result.done) is not bool or type(result.timeout) is not bool
                        or not np.isfinite(result.reward)):
                    raise RuntimeError("native capacity transition outcome invalid")
                done_count += int(result.done)
                timeout_count += int(result.timeout)
                key = keys[lane]
                next_identity = dict(rollout_window_id=f"window-{(step + 1) // 81:06d}",
                                     rollout_step_index=(step + 1) % 81)
                if result.done or result.timeout:
                    episode_counts[lane] += 1
                    key = replace(key, episode_id=f"episode-{lane:03d}-{episode_counts[lane]:08d}",
                                  episode_step_index=0, **next_identity)
                    bindings[lane] = key
                else:
                    key = replace(key, episode_step_index=key.episode_step_index + 1,
                                  **next_identity)
                keys[lane] = key
            row = {"step": step, "capture_teacher_seconds": captured - started,
                   "native_step_seconds": finished - captured, "total_seconds": finished - started,
                   "completed_vector_steps": calls, "completed_transitions": calls * count}
            timings.append(row)
            report({"phase": "vector_step_complete", **row})
        return {"schema": "roborender.leo." + mode + ".v1", "result": "PASS",
                "mode": mode, "num_envs": count,
                "environment_step_calls": calls, "physical_transition_rows": calls * count,
                "teacher_inference_tested": True, "student_inference_tested": False,
                "H200_renderer_tested": False, "optimizer_steps": 0,
                "chunks_published": 0, "production_qualified": False,
                "done_count": done_count, "timeout_count": timeout_count,
                "observation_sources": sources, "timings": timings,
                **({"recorded_action_diagnostic":reference.summary()} if reference is not None else {})}
    finally:
        lifecycle.finish()
