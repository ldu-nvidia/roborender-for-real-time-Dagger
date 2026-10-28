"""Consolidated success evaluation episode_metrics; versioned protocols remain distinct."""
from __future__ import annotations


# ---- episode metrics ----


import math as math
import time as time
from collections.abc import Mapping as Mapping
from copy import deepcopy as deepcopy
from dataclasses import dataclass as dataclass
from typing import Any as Any, Iterable as Iterable


RAVEN_SUCCESS_METRIC = "raven_goal_reached_rate"


@dataclass
class EpisodeGoalLatch:
    """Accumulate Raven's per-step goal signal for one live episode."""

    goal_reached: bool = False

    def update(self, grasp_goal_reached: bool) -> None:
        if not isinstance(grasp_goal_reached, bool):
            raise TypeError("grasp_goal_reached must be boolean")
        self.goal_reached = self.goal_reached or grasp_goal_reached

    def finish(self) -> bool:
        result = self.goal_reached
        self.goal_reached = False
        return result


def wilson_95(successes: int, total: int) -> list[float]:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    rate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def success_summary(outcomes: Iterable[bool]) -> dict[str, Any]:
    values = list(outcomes)
    if not values or any(not isinstance(value, bool) for value in values):
        raise ValueError("outcomes must be a non-empty Boolean sequence")
    successes = sum(values)
    total = len(values)
    return {
        "metric": RAVEN_SUCCESS_METRIC,
        "successes": successes,
        "failures": total - successes,
        "completed_episodes": total,
        "goal_reached_rate": successes / total,
        "wilson_95_interval": wilson_95(successes, total),
    }


def expand_validation_panel(panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sealed 64-case x four-seed panel into 256 episodes."""

    if panel.get("schema") != "fast_roborender.round_dagger.validation_panel_registry.v1":
        raise ValueError("unexpected validation panel schema")
    seeds = panel.get("paired_seeds")
    case_ids = panel.get("case_ids")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("paired_seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired_seeds must be unique")
    if not isinstance(case_ids, list) or not case_ids or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError("case_ids must be a non-empty string list")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_ids must be unique")
    episodes = [
        {
            "scenario_id": f"seed{seed}.{case_id}",
            "seed": seed,
            "case_id": case_id,
            "motion_group_id": case_id.rsplit("__reset_", 1)[0],
            "reset_index": int(case_id.rsplit("__reset_", 1)[1]),
        }
        for seed in seeds
        for case_id in case_ids
    ]
    if len(episodes) != panel.get("case_count"):
        raise ValueError("case_count does not equal paired_seeds x case_ids")
    return episodes


def require_student_only_evaluation(config, *, arm):
    """Reject teacher-takeover settings before a student evaluation starts.

    Raven's enforce_teacher_rollout branch also replaces action_mean in eval
    mode. Selecting that mean later does not undo the takeover. Teacher-only
    diagnostics deliberately retain their separate control convention.
    """
    if arm not in {"teacher", "bc", "dagger"}:
        raise ValueError("unsupported success evaluation arm")
    if arm == "teacher":
        return
    if not isinstance(config, Mapping):
        raise ValueError("student-only evaluation requires explicit control settings")
    ratio = config.get("ratio_teacher_rollout")
    warmup = config.get("rollout_with_teacher_num_steps")
    if (config.get("enforce_teacher_rollout") is not False
            or type(ratio) not in (int, float) or ratio != 0
            or type(warmup) is not int or warmup != 0):
        raise ValueError(
            "student-only evaluation requires enforce_teacher_rollout=false, "
            "ratio_teacher_rollout=0 and rollout_with_teacher_num_steps=0"
        )


class FirstEpisodeAccounting:
    def __init__(self, count: int):
        self.native = [False] * count
        self.exported = [False] * count
        self.outcomes = [None] * count

    def update(self, step, native, exported, dones, timeouts, details):
        count = len(self.outcomes)
        if any(len(values) != count for values in (native, exported, dones, timeouts, details)):
            raise ValueError("lane cardinality mismatch")
        completed = []
        for lane in range(count):
            if self.outcomes[lane] is not None:
                continue
            self.native[lane] |= bool(native[lane])
            self.exported[lane] |= bool(exported[lane])
            if dones[lane]:
                row = dict(details[lane], lane=lane, steps=step + 1,
                           native_success=self.native[lane],
                           exported_success=self.exported[lane], timeout=bool(timeouts[lane]))
                self.outcomes[lane] = row
                completed.append(row)
        return completed

    def result(self):
        rows = [row for row in self.outcomes if row is not None]
        complete = len(rows) == len(self.outcomes)
        return {"completed": len(rows), "expected": len(self.outcomes), "complete": complete,
                "native_successes": sum(row["native_success"] for row in rows),
                "exported_successes": sum(row["exported_success"] for row in rows),
                "accounting_disagreements": sum(row["native_success"] != row["exported_success"] for row in rows),
                "outcomes": rows}


def require_paired_states(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("initial-state fields differ")
    different = [name for name in reference if reference[name] != candidate[name]]
    if different:
        raise ValueError("initial states do not match: " + ", ".join(different))


class PooledEpisodeAccounting:
    def __init__(self, count, minimum):
        if type(count) is not int or not 1 <= count <= 32 or type(minimum) is not int or minimum <= 0:
            raise ValueError('positive explicit pooled population and target required')
        self.count, self.minimum = count, minimum
        self.native, self.exported = [False] * count, [False] * count
        self.episode_index, self.episode_steps = [0] * count, [0] * count
        self.outcomes, self.next_step = [], 0

    def update(self, step, native, exported, dones, timeouts, details):
        if step != self.next_step or self.result()['complete']:
            raise ValueError('pooled accounting cannot replay or run beyond its stop boundary')
        if any(len(values) != self.count for values in (native, exported, dones, timeouts, details)):
            raise ValueError('pooled lane cardinality differs')
        completed = []
        for lane in range(self.count):
            self.native[lane] |= bool(native[lane]); self.exported[lane] |= bool(exported[lane])
            self.episode_steps[lane] += 1
            if dones[lane]:
                row = dict(details[lane], lane=lane, episode_index=self.episode_index[lane],
                    completion_index=len(self.outcomes), steps=self.episode_steps[lane], terminal_vector_step=step,
                    native_success=self.native[lane], exported_success=self.exported[lane], timeout=bool(timeouts[lane]))
                self.outcomes.append(row); completed.append(row)
                self.native[lane] = self.exported[lane] = False
                self.episode_steps[lane] = 0; self.episode_index[lane] += 1
        self.next_step += 1
        return completed

    def result(self):
        rows = self.outcomes
        return dict(completed=len(rows), expected=self.minimum, complete=len(rows) >= self.minimum,
            overshoot=max(0, len(rows) - self.minimum),
            native_successes=sum(r['native_success'] for r in rows),
            exported_successes=sum(r['exported_success'] for r in rows),
            accounting_disagreements=sum(r['native_success'] != r['exported_success'] for r in rows),
            outcomes=deepcopy(rows), episodes_per_lane=list(self.episode_index))




SCHEMA = 'roborender.scaled-success-cycle-timing.v1'
PHASES = ('capture_control_seconds', 'rendered_observation_seconds',
          'student_homie_seconds', 'env_step_seconds', 'bookkeeping_seconds')
METADATA = dict(schema=SCHEMA, clock='host_monotonic_perf_counter',
    timing_kind='exclusive_host_wall_not_gpu_kernel_time',
    cuda_synchronization_added=False,
    asynchronous_cuda_caveat='Asynchronous CUDA work may be charged to a later phase containing an existing wait.',
    capture_control_scope='Existing observation tensor device transfer/preparation; native camera work may occur inside rendered_observation or env_step.',
    student_homie_scope='Existing diagnostic_step, including student, Homie and frozen reference-teacher computation/checks.',
    measurement_boundary='Vector-cycle entry through existing accounting, trace and progress publication; excludes timing telemetry overhead.',
    policy_reset_rng_or_step_semantics_changed=False)


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 450:
        raise ValueError('bounded completed vector-cycle timing rows required')
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {'step', 'total_seconds', *PHASES}
                or type(row['step']) is not int or row['step'] != index
                or any(not _nonnegative(row[key]) for key in (*PHASES, 'total_seconds'))
                or not math.isclose(sum(row[key] for key in PHASES), row['total_seconds'],
                                    rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError('exclusive vector-cycle phase order, duration or total differs')


def validate_receipt(value, *, expected_steps):
    if (type(expected_steps) is not int or not 1 <= expected_steps <= 450
            or not isinstance(value, dict)
            or set(value) != set(METADATA) | {'vector_steps_completed', 'steps', 'totals'}
            or any(value.get(key) != item for key, item in METADATA.items())
            or value.get('vector_steps_completed') != expected_steps
            or not isinstance(value.get('steps'), list) or len(value['steps']) != expected_steps):
        raise ValueError('scaled cycle timing receipt scope differs')
    _validate_rows(value['steps'])
    totals = {key: sum(row[key] for row in value['steps']) for key in (*PHASES, 'total_seconds')}
    if value.get('totals') != totals:
        raise ValueError('scaled cycle timing aggregate differs')
    return deepcopy(value)


class VectorCycleTiming:
    def __init__(self, step, clock):
        self.step, self.clock = step, clock
        self.started = self.last = clock()
        if not _nonnegative(self.started):
            raise RuntimeError('host monotonic timing unavailable')
        self.phases = {}

    def mark(self, phase):
        if len(self.phases) >= len(PHASES) or phase != PHASES[len(self.phases)]:
            raise RuntimeError('vector-cycle phases must be exclusive and ordered')
        now = self.clock()
        if not _nonnegative(now) or now < self.last:
            raise RuntimeError('host monotonic timing regressed')
        self.phases[phase] = now - self.last
        self.last = now


class SuccessCycleTimings:
    def __init__(self, *, clock=None):
        self.clock = time.perf_counter if clock is None else clock
        self.rows, self.active = [], None

    def start(self, step):
        if self.active is not None or type(step) is not int or step != len(self.rows) or step >= 450:
            raise RuntimeError('exact sequential vector-cycle timing required')
        self.active = VectorCycleTiming(step, self.clock)
        return self.active

    def complete(self, cycle):
        if cycle is not self.active or cycle is None or len(cycle.phases) != len(PHASES)-1:
            raise RuntimeError('only this completed vector cycle can be recorded')
        cycle.mark('bookkeeping_seconds')
        row = dict(step=cycle.step, **cycle.phases, total_seconds=cycle.last-cycle.started)
        self.rows.append(row)
        self.active = None
        return deepcopy(row)

    def receipt(self, *, expected_steps):
        if self.active is not None:
            raise RuntimeError('partial vector-cycle timing cannot certify completion')
        value = dict(METADATA, vector_steps_completed=len(self.rows), steps=deepcopy(self.rows),
            totals={key: sum(row[key] for row in self.rows) for key in (*PHASES, 'total_seconds')})
        return validate_receipt(value, expected_steps=expected_steps)


# ---- compare paired success ----
#!/usr/bin/env python3


import argparse as comparison_argparse
from collections.abc import Mapping as comparison_Mapping
import json as comparison_json
import math as comparison_math
from pathlib import Path as comparison_Path
from typing import Any as comparison_Any


class comparison_PairedEvaluationError(ValueError):
    """Raised when two evaluation reports are not scientifically comparable."""


def comparison__evaluation(value: comparison_Any, label: str) -> comparison_Mapping[str, comparison_Any]:
    if not isinstance(value, comparison_Mapping):
        raise comparison_PairedEvaluationError(f"{label} must contain a JSON object")
    result = value.get("evaluation", value)
    if not isinstance(result, comparison_Mapping):
        raise comparison_PairedEvaluationError(f"{label}.evaluation must be an object")
    return result


def comparison__outcomes(report: comparison_Mapping[str, comparison_Any], label: str) -> dict[str, bool]:
    rows = report.get("outcomes")
    if not isinstance(rows, list):
        raise comparison_PairedEvaluationError(f"{label} has no per-scenario outcomes")
    result: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, comparison_Mapping):
            raise comparison_PairedEvaluationError(f"{label} outcome is not an object")
        scenario = row.get("scenario_id")
        success = row.get("goal_reached")
        if not isinstance(scenario, str) or not scenario:
            raise comparison_PairedEvaluationError(f"{label} outcome has invalid scenario_id")
        if not isinstance(success, (bool, int)) or success not in (False, True, 0, 1):
            raise comparison_PairedEvaluationError(f"{label} outcome has invalid goal_reached")
        if scenario in result:
            raise comparison_PairedEvaluationError(f"{label} repeats scenario {scenario}")
        result[scenario] = bool(success)
    return result


def comparison__exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comparison_math.comb(discordant, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def comparison_compare_reports(
    baseline_value: comparison_Any,
    candidate_value: comparison_Any,
    *,
    expected_cases: int = 256,
) -> dict[str, comparison_Any]:
    baseline = comparison__evaluation(baseline_value, "baseline")
    candidate = comparison__evaluation(candidate_value, "candidate")
    identity_fields = ("scenario_manifest_sha256", "observation_mode", "metric")
    # Retain legacy report readability, but never discard stronger identities
    # supplied by a corrected evaluator. A missing identity on just one side
    # is a mismatch, not permission to compare old and corrected scores.
    runtime_fields = ("goal_accounting_version", "runtime_physics_sha256",
                      "reset_identity_sha256", "camera_calibration_sha256")
    for field in identity_fields:
        if not baseline.get(field) or not candidate.get(field):
            raise comparison_PairedEvaluationError(f"evaluation {field} is missing")
        if baseline.get(field) != candidate.get(field):
            raise comparison_PairedEvaluationError(f"evaluation {field} differs")
    for field in runtime_fields:
        if baseline.get(field) != candidate.get(field):
            raise comparison_PairedEvaluationError(f"evaluation {field} differs")
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("status") != "complete":
            raise comparison_PairedEvaluationError(f"{label} evaluation is not complete")
        if report.get("completed") != expected_cases or report.get("total") != expected_cases:
            raise comparison_PairedEvaluationError(
                f"{label} must contain exactly {expected_cases} completed cases"
            )
    left = comparison__outcomes(baseline, "baseline")
    right = comparison__outcomes(candidate, "candidate")
    if set(left) != set(right) or len(left) != expected_cases:
        raise comparison_PairedEvaluationError("evaluations do not contain the identical scenario set")

    fail_fail = fail_success = success_fail = success_success = 0
    for scenario in sorted(left):
        pair = (left[scenario], right[scenario])
        if pair == (False, False):
            fail_fail += 1
        elif pair == (False, True):
            fail_success += 1
        elif pair == (True, False):
            success_fail += 1
        else:
            success_success += 1
    baseline_successes = success_fail + success_success
    candidate_successes = fail_success + success_success
    return {
        "schema": "roborender.paired_success_comparison.v1",
        "scenario_manifest_sha256": baseline["scenario_manifest_sha256"],
        "observation_mode": baseline["observation_mode"],
        "metric": baseline["metric"],
        "goal_accounting_version": baseline.get("goal_accounting_version", "legacy_unversioned"),
        "runtime_identity_fields_complete": all(bool(baseline.get(field)) for field in runtime_fields),
        "runtime_identity_fields": {field: baseline[field] for field in runtime_fields if field in baseline},
        "completed_cases": expected_cases,
        "baseline": {
            "evaluation_id": baseline.get("evaluation_id"),
            "checkpoint_sha256": baseline.get("checkpoint_sha256"),
            "successes": baseline_successes,
            "goal_reached_rate": baseline_successes / expected_cases,
        },
        "candidate": {
            "evaluation_id": candidate.get("evaluation_id"),
            "checkpoint_sha256": candidate.get("checkpoint_sha256"),
            "successes": candidate_successes,
            "goal_reached_rate": candidate_successes / expected_cases,
        },
        "paired_cells": {
            "both_failed": fail_fail,
            "baseline_failed_candidate_succeeded": fail_success,
            "baseline_succeeded_candidate_failed": success_fail,
            "both_succeeded": success_success,
        },
        "absolute_rate_change": (candidate_successes - baseline_successes) / expected_cases,
        "exact_two_sided_mcnemar_p": comparison__exact_mcnemar(fail_success, success_fail),
    }


def comparison_main() -> int:
    parser = comparison_argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=comparison_Path, required=True)
    parser.add_argument("--candidate", type=comparison_Path, required=True)
    parser.add_argument("--output", type=comparison_Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=256)
    args = parser.parse_args()
    if args.expected_cases < 1:
        parser.error("--expected-cases must be positive")
    result = comparison_compare_reports(
        comparison_json.loads(args.baseline.read_text(encoding="utf-8")),
        comparison_json.loads(args.candidate.read_text(encoding="utf-8")),
        expected_cases=args.expected_cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(comparison_json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(comparison_json.dumps(result, sort_keys=True))
    return 0
