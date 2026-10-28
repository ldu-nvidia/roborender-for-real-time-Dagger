"""Numerical action-reference experiment, never production simulator replay.

Fresh native owners use a pinned scene/seed. Recorded actions stress the camera
trajectory; native teacher inference still runs. Numerical drift is measured,
not hidden. No checkpoint, rollout artifact or production receipt is published.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

MODE = "recorded-action-camera-soak"
ACK = "I_ACKNOWLEDGE_FRESH_DIAGNOSTIC_129_RECORDED_THEN_195_TEACHER_STEPS"
SCHEMA = "roborender.recorded-action-camera-reference.v1"


def decode(value, shape):
    if (not isinstance(value, dict) or set(value) != {"dtype", "shape", "sha256", "data_b64"}
            or value["dtype"] != "float32" or value["shape"] != list(shape)):
        raise ValueError("exact finite float32 reference envelope required")
    raw = base64.b64decode(value["data_b64"], validate=True)
    if len(raw) != int(np.prod(shape))*4 or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("action reference array digest/length mismatch")
    result = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite action reference")
    return result


def validate_reference(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "reference_run_id", "num_envs",
            "reference_steps", "reference_assignment_sha256", "rows", "production_replay_allowed",
            "simulator_restore_claimed"} or value["schema"] != SCHEMA
            or value["num_envs"] != 16 or value["reference_steps"] != 129
            or value["production_replay_allowed"] is not False
            or value["simulator_restore_claimed"] is not False
            or not isinstance(value["reference_run_id"], str)
            or len(value["reference_assignment_sha256"]) != 64
            or len(value["rows"]) != 129):
        raise ValueError("bounded fresh diagnostic reference required")
    decoded = []
    for index, row in enumerate(value["rows"]):
        if (set(row) != {"step", "selected_actions", "teacher_actions", "actor_obs", "outcomes", "selected_sources"}
                or type(row["step"]) is not int or row["step"] != index
                or len(row["outcomes"]) != 16 or len(row["selected_sources"]) != 16):
            raise ValueError("reference ordering/cardinality mismatch")
        actions, teacher, obs = (decode(row[name], shape) for name, shape in (
            ("selected_actions", (16,26)), ("teacher_actions", (16,26)), ("actor_obs", (16,113))))
        if not np.array_equal(actions[:, 11:], teacher[:, 11:]):
            raise ValueError("recorded Homie tail differs from recorded teacher")
        for lane, (source, outcome) in enumerate(zip(row["selected_sources"], row["outcomes"])):
            if (source not in {"teacher", "student"} or set(outcome) != {"done", "timeout"}
                    or any(type(outcome[k]) is not bool for k in outcome)):
                raise ValueError("invalid recorded selection/outcome")
            if source == "teacher" and not np.array_equal(actions[lane], teacher[lane]):
                raise ValueError("recorded teacher selection differs")
        decoded.append(dict(actions=actions, teacher=teacher, obs=obs, outcomes=row["outcomes"]))
    return decoded


def read_reference(path, expected_sha, assignment_sha):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 4 << 20:
        raise ValueError("bounded regular numerical reference required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("numerical reference digest mismatch")
    value = json.loads(raw)
    if value.get("reference_assignment_sha256") != assignment_sha:
        raise ValueError("reference is bound to a different target scene")
    return RecordedActions(validate_reference(value), expected_sha)


class RecordedActions:
    def __init__(self, rows, digest):
        self.rows, self.digest = rows, digest
        self.metrics = []
        self.outcome_mismatches = 0

    def select(self, step, captures, teachers):
        if step >= 129:
            return teachers
        row = self.rows[step]
        current = np.stack([capture.arrays["actor_obs"] for capture in captures])
        metric = dict(step=step, actor_max_abs_error=float(np.max(np.abs(current-row["obs"]))),
            teacher_max_abs_error=float(np.max(np.abs(teachers-row["teacher"]))),
            homie_max_abs_error=float(np.max(np.abs(teachers[:,11:]-row["teacher"][:,11:]))),
            actor_exact=bool(np.array_equal(current,row["obs"])),
            teacher_exact=bool(np.array_equal(teachers,row["teacher"])))
        # Do not apply an archived sequence to an unrelated initial state.
        # Later numerical drift is explicitly measured; this is not a claim
        # of native restore or of an exactly reproduced production trajectory.
        if step == 0 and not np.allclose(current,row["obs"],rtol=1e-5,atol=1e-5):
            raise RuntimeError("recorded-action diagnostic initial state mismatch: " + json.dumps(metric))
        self.metrics.append(metric)
        return row["actions"].copy()

    def observe(self, step, results):
        if step < 129:
            self.outcome_mismatches += sum((result.done,result.timeout) != (ref["done"],ref["timeout"])
                for result,ref in zip(results,self.rows[step]["outcomes"]))

    def summary(self):
        return dict(reference_sha256=self.digest, recorded_vector_steps=len(self.metrics),
            recorded_transition_rows=16*len(self.metrics), numerical_comparisons=self.metrics,
            outcome_mismatches=self.outcome_mismatches, production_replay_allowed=False,
            simulator_restore_claimed=False, live_student_inference=False)
