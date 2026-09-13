"""Fail-closed contracts for Raven fixed-panel success validation.

The H200 coordinator registers a sealed 64-case x 4-seed assignment.  RTX
executes one native 64-environment Raven cohort per seed.  A seed is never
split across freshly initialized processes: doing so restarts Raven's RNG and
does not reproduce lanes 32..63 of the original cohort.  This module is
intentionally independent of Isaac/Torch; the native adapter lives in
:mod:`dagger_rtx.fixed_panel_raven`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .contracts import canonical_json, require_identifier, require_sha256
from .durable import atomic_write_json


ASSIGNMENT_SCHEMA = "roborender.raven_success_validation_assignment.v1"
OUTCOME_JOURNAL_SCHEMA = "roborender.raven_success_validation_outcome.v1"
SUMMARY_SCHEMA = "roborender.raven_success_validation_local_summary.v1"
ACTION_POLICY = "deterministic_student_action_mean_beta_0"
OBSERVATION_MODE = "raytraced"
METRIC = "raven_goal_reached_rate"
EXPECTED_EPISODES = 256
EXPECTED_CASES_PER_SEED = 64
COHORT_ENVIRONMENTS = 64
EXPECTED_SEED_COUNT = 4
EXPECTED_MOTION_GROUPS = 5
EXPECTED_VIEWS = ("head_color", "hand_left", "hand_right")
EXPECTED_PAIRED_SEEDS = (1103, 2207, 3301, 4409)
EXPECTED_SCENARIO_MANIFEST_SHA256 = (
    "ad1c3a57896e01f19e260ecb0daead8b508031326e217e134793627df5af8ace"
)
EXPECTED_MOTION_GROUP_NAMES = (
    "record_2025-10-14_21-29-22_0.675m.npz",
    "record_2025-10-14_21-46-10_0.700m.npz",
    "record_2025-10-14_21-52-25_0.725m.npz",
    "record_2025-10-14_22-00-53_0.750m.npz",
    "record_2025-10-14_22-18-36_0.775m.npz",
)
_CASE = re.compile(r"^(?P<group>[^/]+\.npz)__reset_(?P<reset>[0-9]{2})$")
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_OUTCOME_ACK_KEYS = {
    "ok",
    "evaluation_id",
    "scenario_id",
    "status",
    "completed",
    "total",
}
_OUTCOME_IDEMPOTENT_ACK_KEYS = {
    "ok",
    "evaluation_id",
    "scenario_id",
    "status",
    "idempotent",
}


class FixedPanelValidationError(RuntimeError):
    """A validation identity, simulator result, or durable receipt diverged."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FixedPanelValidationError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(value)}"
        )


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FixedPanelValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_outcome_ack(
    response: Mapping[str, Any],
    *,
    evaluation_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise FixedPanelValidationError("outcome acknowledgement must be an object")
    if set(response) == _OUTCOME_IDEMPOTENT_ACK_KEYS:
        if (
            response["ok"] is not True
            or response["evaluation_id"] != evaluation_id
            or response["scenario_id"] != scenario_id
            or response["status"] not in {"collecting", "complete"}
            or response["idempotent"] is not True
        ):
            raise FixedPanelValidationError(
                "idempotent outcome acknowledgement identity/status mismatch"
            )
        return dict(response)
    _strict_keys(response, _OUTCOME_ACK_KEYS, "outcome acknowledgement")
    completed = _strict_int(response["completed"], "acknowledgement completed", minimum=1)
    total = _strict_int(response["total"], "acknowledgement total", minimum=1)
    status = response["status"]
    if (
        response["ok"] is not True
        or response["evaluation_id"] != evaluation_id
        or response["scenario_id"] != scenario_id
        or total != EXPECTED_EPISODES
        or completed > total
        or status not in {"collecting", "complete"}
        or (status == "collecting") != (completed < total)
    ):
        raise FixedPanelValidationError("outcome acknowledgement identity/status mismatch")
    return dict(response)


@dataclass(frozen=True)
class FixedPanelCase:
    scenario_id: str
    seed: int
    case_id: str
    motion_group_id: str
    reset_index: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixedPanelCase":
        if not isinstance(value, Mapping):
            raise FixedPanelValidationError("assignment episode must be an object")
        _strict_keys(
            value,
            {"scenario_id", "seed", "case_id", "motion_group_id", "reset_index"},
            "assignment episode",
        )
        scenario_id = require_identifier(value["scenario_id"], "scenario_id")
        seed = _strict_int(value["seed"], "seed")
        case_id = require_identifier(value["case_id"], "case_id")
        motion_group_id = require_identifier(value["motion_group_id"], "motion_group_id")
        reset_index = _strict_int(value["reset_index"], "reset_index")
        match = _CASE.fullmatch(case_id)
        if (
            match is None
            or match.group("group") != motion_group_id
            or int(match.group("reset")) != reset_index
            or scenario_id != f"seed{seed}.{case_id}"
        ):
            raise FixedPanelValidationError(
                "scenario/case/motion/reset identity is not the sealed panel convention"
            )
        return cls(scenario_id, seed, case_id, motion_group_id, reset_index)


@dataclass(frozen=True)
class FixedPanelCohort:
    cohort_index: int
    seed_index: int
    paired_seed: int
    cases: tuple[FixedPanelCase, ...]

    @property
    def cohort_id(self) -> str:
        return f"seed-{self.paired_seed}-cohort-64"


@dataclass(frozen=True)
class FixedPanelAssignment:
    path: Path
    assignment_sha256: str
    evaluation_id: str
    run_id: str
    round_id: int
    checkpoint_sha256: str
    scenario_manifest: str
    scenario_manifest_sha256: str
    episodes: tuple[FixedPanelCase, ...]
    seeds: tuple[int, ...]
    motion_groups: tuple[str, ...]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_assignment_sha256: str,
        expected_evaluation_id: str,
        expected_run_id: str,
        expected_round_id: int,
        expected_checkpoint_sha256: str,
        expected_scenario_manifest_sha256: str,
    ) -> "FixedPanelAssignment":
        path = Path(path).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise FixedPanelValidationError("assignment must be a regular non-symlink file")
        actual_assignment_sha = file_sha256(path)
        if actual_assignment_sha != require_sha256(
            expected_assignment_sha256, "expected_assignment_sha256"
        ):
            raise FixedPanelValidationError("assignment file SHA-256 mismatch")
        try:
            value = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FixedPanelValidationError("assignment is not valid UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise FixedPanelValidationError("assignment must contain a JSON object")
        _strict_keys(
            value,
            {
                "schema", "evaluation_id", "run_id", "round_id",
                "checkpoint_sha256", "scenario_manifest",
                "scenario_manifest_sha256", "observation_mode", "action_policy",
                "metric", "episode_count", "episodes",
            },
            "assignment",
        )
        if value["schema"] != ASSIGNMENT_SCHEMA:
            raise FixedPanelValidationError("assignment schema mismatch")
        evaluation_id = require_identifier(value["evaluation_id"], "evaluation_id")
        run_id = require_identifier(value["run_id"], "run_id")
        round_id = _strict_int(value["round_id"], "round_id")
        checkpoint = require_sha256(value["checkpoint_sha256"], "checkpoint_sha256")
        panel_sha = require_sha256(
            value["scenario_manifest_sha256"], "scenario_manifest_sha256"
        )
        if (
            evaluation_id != require_identifier(expected_evaluation_id, "expected_evaluation_id")
            or run_id != require_identifier(expected_run_id, "expected_run_id")
            or round_id != expected_round_id
            or checkpoint != require_sha256(
                expected_checkpoint_sha256, "expected_checkpoint_sha256"
            )
            or panel_sha != require_sha256(
                expected_scenario_manifest_sha256,
                "expected_scenario_manifest_sha256",
            )
            or panel_sha != EXPECTED_SCENARIO_MANIFEST_SHA256
        ):
            raise FixedPanelValidationError("assignment immutable identity mismatch")
        if (
            value["observation_mode"] != OBSERVATION_MODE
            or value["action_policy"] != ACTION_POLICY
            or value["metric"] != METRIC
            or value["episode_count"] != EXPECTED_EPISODES
            or not isinstance(value["scenario_manifest"], str)
        ):
            raise FixedPanelValidationError("assignment evaluation contract mismatch")
        raw_episodes = value["episodes"]
        if not isinstance(raw_episodes, list) or len(raw_episodes) != EXPECTED_EPISODES:
            raise FixedPanelValidationError("assignment must contain exactly 256 episodes")
        episodes = tuple(FixedPanelCase.from_mapping(item) for item in raw_episodes)
        if len({item.scenario_id for item in episodes}) != EXPECTED_EPISODES:
            raise FixedPanelValidationError("assignment scenario IDs are not unique")
        seeds = tuple(dict.fromkeys(item.seed for item in episodes))
        if seeds != EXPECTED_PAIRED_SEEDS:
            raise FixedPanelValidationError(
                f"assignment paired seeds must be exactly {EXPECTED_PAIRED_SEEDS}"
            )
        per_seed: dict[int, list[FixedPanelCase]] = {seed: [] for seed in seeds}
        for item in episodes:
            per_seed[item.seed].append(item)
        if any(len(rows) != EXPECTED_CASES_PER_SEED for rows in per_seed.values()):
            raise FixedPanelValidationError("each paired seed must contain exactly 64 cases")
        canonical_cases = tuple(item.case_id for item in per_seed[seeds[0]])
        if len(set(canonical_cases)) != EXPECTED_CASES_PER_SEED:
            raise FixedPanelValidationError("the base panel does not contain 64 unique cases")
        if any(tuple(item.case_id for item in per_seed[seed]) != canonical_cases for seed in seeds[1:]):
            raise FixedPanelValidationError("paired seeds do not share the same ordered 64 cases")
        motion_groups = tuple(dict.fromkeys(item.motion_group_id for item in per_seed[seeds[0]]))
        if motion_groups != EXPECTED_MOTION_GROUP_NAMES:
            raise FixedPanelValidationError(
                "fixed panel motion groups differ from the sealed Raven inventory"
            )
        for seed in seeds:
            grouped = {group: [] for group in motion_groups}
            for item in per_seed[seed]:
                grouped[item.motion_group_id].append(item.reset_index)
            if sorted(len(values) for values in grouped.values()) != [12, 13, 13, 13, 13]:
                raise FixedPanelValidationError("motion groups must be balanced 13/13/13/13/12")
            if any(sorted(values) != list(range(len(values))) for values in grouped.values()):
                raise FixedPanelValidationError("reset indices must be contiguous from zero")
        result = cls(
            path=path,
            assignment_sha256=actual_assignment_sha,
            evaluation_id=evaluation_id,
            run_id=run_id,
            round_id=round_id,
            checkpoint_sha256=checkpoint,
            scenario_manifest=value["scenario_manifest"],
            scenario_manifest_sha256=panel_sha,
            episodes=episodes,
            seeds=seeds,
            motion_groups=motion_groups,
        )
        result.cohorts()  # prove the native B=64 schedule covers the assignment
        return result

    def cohorts(self) -> tuple[FixedPanelCohort, ...]:
        lookup = {
            (item.seed, item.motion_group_id, item.reset_index): item
            for item in self.episodes
        }
        cohorts: list[FixedPanelCohort] = []
        for seed_index, seed in enumerate(self.seeds):
            consumed = {group: 0 for group in self.motion_groups}
            cohort: list[FixedPanelCase] = []
            for environment_index in range(EXPECTED_CASES_PER_SEED):
                group = self.motion_groups[environment_index % len(self.motion_groups)]
                reset_index = consumed[group]
                try:
                    cohort.append(lookup[(seed, group, reset_index)])
                except KeyError as exc:
                    raise FixedPanelValidationError(
                        "assignment cannot reproduce Raven cyclic motion/reset mapping"
                    ) from exc
                consumed[group] += 1
            cohorts.append(
                FixedPanelCohort(
                    cohort_index=len(cohorts),
                    seed_index=seed_index,
                    paired_seed=seed,
                    cases=tuple(cohort),
                )
            )
        if (
            len(cohorts) != EXPECTED_SEED_COUNT
            or any(len(cohort.cases) != COHORT_ENVIRONMENTS for cohort in cohorts)
            or {case.scenario_id for cohort in cohorts for case in cohort.cases}
            != {case.scenario_id for case in self.episodes}
        ):
            raise FixedPanelValidationError("fixed-panel cohort plan is not an exact partition")
        return tuple(cohorts)


def attest_runtime_reset(
    cohort: FixedPanelCohort,
    *,
    motion_groups: Sequence[str],
    motion_sample_bounds: Sequence[Sequence[int]],
    sample_indices: Sequence[int],
    sample_indices_in_sequence: Sequence[int],
    samples_per_motion: int,
) -> dict[str, Any]:
    """Bind sealed case ordinals to one real native B=64 reset.

    ``reset_index`` in the sealed panel is the case ordinal within a motion
    group, not a literal row in Raven's motion dataset.  The actual sampled
    dataset row and within-sequence index are RNG outcomes.  This gate proves
    that all 64 RNG draws happened together in one process, each lane used the
    expected motion interval, and every sealed ordinal maps to exactly one
    native lane.  The concrete draws are returned for durable attestation.
    """

    if len(cohort.cases) != COHORT_ENVIRONMENTS:
        raise FixedPanelValidationError("runtime reset requires one complete B=64 cohort")
    groups = tuple(str(value) for value in motion_groups)
    if len(groups) != EXPECTED_MOTION_GROUPS or len(set(groups)) != len(groups):
        raise FixedPanelValidationError("runtime reset motion-group inventory differs")
    if len(motion_sample_bounds) != len(groups):
        raise FixedPanelValidationError("runtime reset motion bounds width differs")
    bounds: list[tuple[int, int]] = []
    for index, value in enumerate(motion_sample_bounds):
        if not isinstance(value, Sequence) or len(value) != 2:
            raise FixedPanelValidationError(f"motion bound {index} must be [lower, upper]")
        lower = _strict_int(value[0], f"motion bound {index} lower")
        upper = _strict_int(value[1], f"motion bound {index} upper", minimum=1)
        if lower >= upper:
            raise FixedPanelValidationError(f"motion bound {index} must be non-empty")
        bounds.append((lower, upper))
    samples_per_motion = _strict_int(
        samples_per_motion, "samples_per_motion", minimum=1
    )
    if (
        len(sample_indices) != COHORT_ENVIRONMENTS
        or len(sample_indices_in_sequence) != COHORT_ENVIRONMENTS
    ):
        raise FixedPanelValidationError("runtime reset sample vectors must have width 64")

    rows: list[dict[str, Any]] = []
    for lane, case in enumerate(cohort.cases):
        group_index = lane % len(groups)
        expected_group = groups[group_index]
        expected_ordinal = lane // len(groups)
        if (
            case.seed != cohort.paired_seed
            or case.motion_group_id != expected_group
            or case.reset_index != expected_ordinal
        ):
            raise FixedPanelValidationError(
                f"sealed case identity does not match native lane {lane}"
            )
        sample_index = _strict_int(sample_indices[lane], f"lane {lane} sample_idx")
        sample_in_sequence = _strict_int(
            sample_indices_in_sequence[lane], f"lane {lane} sample_idx_in_seq"
        )
        lower, upper = bounds[group_index]
        if not lower <= sample_index < upper:
            raise FixedPanelValidationError(
                f"lane {lane} sampled outside its sealed motion interval"
            )
        if sample_in_sequence >= samples_per_motion:
            raise FixedPanelValidationError(
                f"lane {lane} sample_idx_in_seq exceeds Raven sequence width"
            )
        rows.append(
            {
                "lane": lane,
                "scenario_id": case.scenario_id,
                "case_id": case.case_id,
                "paired_seed": case.seed,
                "motion_group_id": case.motion_group_id,
                "reset_ordinal": case.reset_index,
                "sample_interval": [lower, upper],
                "actual_sample_idx": sample_index,
                "actual_sample_idx_in_seq": sample_in_sequence,
            }
        )
    preimage = {
        "semantics": "reset_index_is_within_motion_case_ordinal",
        "cohort_id": cohort.cohort_id,
        "paired_seed": cohort.paired_seed,
        "environment_count": COHORT_ENVIRONMENTS,
        "samples_per_motion": samples_per_motion,
        "lanes": rows,
    }
    return {
        **preimage,
        "reset_identity_sha256": hashlib.sha256(canonical_json(preimage)).hexdigest(),
    }


@dataclass(frozen=True)
class EpisodeOutcome:
    case: FixedPanelCase
    goal_reached: bool
    reward: float
    steps: int
    termination: str
    safety_violation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.goal_reached, bool) or not isinstance(self.safety_violation, bool):
            raise FixedPanelValidationError("outcome booleans must be bool")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward):
            raise FixedPanelValidationError("outcome reward must be finite")
        _strict_int(self.steps, "outcome steps", minimum=1)
        if self.termination not in {"success", "failure", "timeout", "safety_violation"}:
            raise FixedPanelValidationError("invalid outcome termination")
        if self.termination == "success" and not self.goal_reached:
            raise FixedPanelValidationError("success termination requires goal_reached")
        if self.safety_violation != (self.termination == "safety_violation"):
            raise FixedPanelValidationError(
                "safety_violation termination must equal safety_violation"
            )


class EpisodeLatch:
    """Independent Raven metric latch for one exactly-once episode per lane."""

    def __init__(self, cohort: FixedPanelCohort):
        self.cohort = cohort
        self.goal = [False] * len(cohort.cases)
        self.reward = [0.0] * len(cohort.cases)
        self.steps = [0] * len(cohort.cases)
        self.finished = [False] * len(cohort.cases)

    def observe(
        self,
        *,
        rewards: Sequence[float],
        dones: Sequence[bool],
        grasp_goal_reached: Sequence[bool],
        timeouts: Sequence[bool] | None = None,
        safety_violations: Sequence[bool] | None = None,
    ) -> tuple[EpisodeOutcome, ...]:
        batch = len(self.cohort.cases)
        values = (rewards, dones, grasp_goal_reached)
        if any(len(item) != batch for item in values):
            raise FixedPanelValidationError("episode latch vector width mismatch")
        if timeouts is None:
            timeouts = [False] * batch
        if safety_violations is None:
            safety_violations = [False] * batch
        if len(timeouts) != batch or len(safety_violations) != batch:
            raise FixedPanelValidationError("episode latch status vector width mismatch")
        completed: list[EpisodeOutcome] = []
        for lane in range(batch):
            if self.finished[lane]:
                # Raven keeps stepping lanes while waiting for the slowest first
                # episode.  Their reset episodes are deliberately excluded.
                continue
            reward = rewards[lane]
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise FixedPanelValidationError("episode reward vector contains a non-finite value")
            if any(not isinstance(item[lane], bool) for item in (dones, grasp_goal_reached, timeouts, safety_violations)):
                raise FixedPanelValidationError("episode status vectors must contain bool")
            self.reward[lane] += float(reward)
            self.steps[lane] += 1
            self.goal[lane] = self.goal[lane] or grasp_goal_reached[lane]
            if dones[lane]:
                self.finished[lane] = True
                safety = safety_violations[lane]
                termination = (
                    "safety_violation" if safety
                    else "success" if self.goal[lane]
                    else "timeout" if timeouts[lane]
                    else "failure"
                )
                completed.append(
                    EpisodeOutcome(
                        case=self.cohort.cases[lane],
                        goal_reached=self.goal[lane],
                        reward=self.reward[lane],
                        steps=self.steps[lane],
                        termination=termination,
                        safety_violation=safety,
                    )
                )
        return tuple(completed)

    @property
    def completed(self) -> int:
        return sum(self.finished)


class EvaluationService(Protocol):
    def post_outcome(self, evaluation_id: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_evaluation(self, evaluation_id: str) -> Mapping[str, Any]: ...


class AuthenticatedEvaluationClient:
    """Bounded bearer client restricted to the loopback evaluation service."""

    def __init__(
        self,
        base_url: str,
        token_file: Path,
        *,
        timeout_s: float = 30.0,
        attempts: int = 3,
        retry_delay_s: float = 1.0,
        _allow_test_port: bool = False,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.path not in ("", "/")
            or (parsed.port != 18614 and not _allow_test_port)
        ):
            raise FixedPanelValidationError(
                "evaluation service must be http://127.0.0.1:18614"
            )
        token_file = Path(token_file)
        if token_file.stat().st_mode & 0o077:
            raise PermissionError("evaluation bearer token must be owner-only")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise PermissionError("invalid evaluation bearer token")
        if timeout_s <= 0 or attempts < 1 or retry_delay_s < 0:
            raise ValueError("invalid evaluation client retry settings")
        self.host = parsed.hostname
        self.port = parsed.port
        self.token = token
        self.timeout_s = timeout_s
        self.attempts = attempts
        self.retry_delay_s = retry_delay_s

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise FixedPanelValidationError("unsafe evaluation service path")
        body = None if payload is None else canonical_json(payload)
        if body is not None and len(body) > 1 << 20:
            raise FixedPanelValidationError("evaluation request exceeds 1 MiB")
        for attempt in range(1, self.attempts + 1):
            connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout_s
            )
            try:
                headers = {"Authorization": f"Bearer {self.token}"}
                if body is not None:
                    headers.update(
                        {"Content-Type": "application/json", "Content-Length": str(len(body))}
                    )
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read((8 << 20) + 1)
                if len(raw) > 8 << 20:
                    raise FixedPanelValidationError("evaluation response exceeds 8 MiB")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise FixedPanelValidationError("evaluation service returned invalid JSON") from exc
                if not isinstance(value, dict):
                    raise FixedPanelValidationError("evaluation response must be an object")
                if 200 <= response.status < 300:
                    if value.get("ok") is not True:
                        raise FixedPanelValidationError("evaluation response is not ok")
                    return value
                if response.status not in _TRANSIENT_STATUSES or attempt == self.attempts:
                    raise FixedPanelValidationError(
                        f"evaluation service HTTP {response.status}: {str(value)[:256]}"
                    )
            except (TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
                if attempt == self.attempts:
                    raise FixedPanelValidationError(
                        f"evaluation transport failed after {attempt} attempts: {exc}"
                    ) from exc
            finally:
                connection.close()
            time.sleep(self.retry_delay_s)
        raise AssertionError("unreachable")

    def post_outcome(self, evaluation_id: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        evaluation_id = require_identifier(evaluation_id, "evaluation_id")
        return self._request("POST", f"/v2/evaluations/{evaluation_id}/outcomes", payload)

    def get_evaluation(self, evaluation_id: str) -> Mapping[str, Any]:
        evaluation_id = require_identifier(evaluation_id, "evaluation_id")
        return self._request("GET", f"/v2/evaluations/{evaluation_id}")


def _outcome_identity(assignment: FixedPanelAssignment, case: FixedPanelCase) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": OUTCOME_JOURNAL_SCHEMA,
                "evaluation_id": assignment.evaluation_id,
                "scenario_id": case.scenario_id,
                "checkpoint_sha256": assignment.checkpoint_sha256,
                "scenario_manifest_sha256": assignment.scenario_manifest_sha256,
            }
        )
    ).hexdigest()


class DurableOutcomeJournal:
    """Persist-before-send outcome ledger with exact idempotent replay."""

    def __init__(self, root: Path, assignment: FixedPanelAssignment):
        self.root = Path(root)
        self.assignment = assignment
        (self.root / "pending").mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "acked").mkdir(parents=True, exist_ok=True, mode=0o700)
        attestation = {
            "schema": "roborender.raven_success_validation_attestation.v2",
            "assignment_sha256": assignment.assignment_sha256,
            "evaluation_id": assignment.evaluation_id,
            "run_id": assignment.run_id,
            "round_id": assignment.round_id,
            "checkpoint_sha256": assignment.checkpoint_sha256,
            "scenario_manifest_sha256": assignment.scenario_manifest_sha256,
            "episode_count": len(assignment.episodes),
            "observation_mode": OBSERVATION_MODE,
            "action_policy": ACTION_POLICY,
            "metric": METRIC,
            "views": list(EXPECTED_VIEWS),
        }
        path = self.root / "assignment-attestation.json"
        if path.exists():
            if json.loads(path.read_bytes()) != attestation:
                raise FixedPanelValidationError("outcome journal assignment identity changed")
        else:
            atomic_write_json(path, attestation)

    def payload(self, outcome: EpisodeOutcome) -> dict[str, Any]:
        identity = _outcome_identity(self.assignment, outcome.case)
        return {
            "idempotency_key": f"fixed-panel-{identity}",
            "scenario_id": outcome.case.scenario_id,
            "episode_id": f"episode-{identity}",
            "goal_reached": outcome.goal_reached,
            "reward": float(outcome.reward),
            "steps": outcome.steps,
            "termination": outcome.termination,
            "safety_violation": outcome.safety_violation,
            "checkpoint_sha256": self.assignment.checkpoint_sha256,
            "scenario_manifest_sha256": self.assignment.scenario_manifest_sha256,
        }

    def _paths(self, case: FixedPanelCase) -> tuple[Path, Path]:
        identity = _outcome_identity(self.assignment, case)
        return self.root / "pending" / f"{identity}.json", self.root / "acked" / f"{identity}.json"

    def stage(self, outcome: EpisodeOutcome) -> dict[str, Any]:
        payload = self.payload(outcome)
        pending, acked = self._paths(outcome.case)
        for path in (pending, acked):
            if path.exists():
                existing = json.loads(path.read_bytes())
                existing_payload = existing.get("payload", existing)
                if existing_payload != payload:
                    raise FixedPanelValidationError(
                        "sealed scenario produced a different outcome on replay"
                    )
        if not pending.exists():
            atomic_write_json(
                pending,
                {
                    "schema": OUTCOME_JOURNAL_SCHEMA,
                    "payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
                    "payload": payload,
                },
            )
        return payload

    def post_staged(
        self,
        case: FixedPanelCase,
        service: EvaluationService,
    ) -> Mapping[str, Any]:
        pending, acked = self._paths(case)
        if not pending.is_file():
            raise FixedPanelValidationError("outcome must be durably staged before posting")
        staged = json.loads(pending.read_bytes())
        payload = staged["payload"]
        if acked.exists():
            prior = json.loads(acked.read_bytes())
            if prior.get("payload") != payload:
                raise FixedPanelValidationError("outcome acknowledgement payload changed")
            return _validate_outcome_ack(
                prior.get("response"),
                evaluation_id=self.assignment.evaluation_id,
                scenario_id=case.scenario_id,
            )
        response = _validate_outcome_ack(
            service.post_outcome(self.assignment.evaluation_id, payload),
            evaluation_id=self.assignment.evaluation_id,
            scenario_id=case.scenario_id,
        )
        atomic_write_json(
            acked,
            {
                "schema": OUTCOME_JOURNAL_SCHEMA,
                "payload_sha256": staged["payload_sha256"],
                "payload": payload,
                "response": response,
            },
        )
        return response

    def submit(self, outcome: EpisodeOutcome, service: EvaluationService) -> Mapping[str, Any]:
        self.stage(outcome)
        return self.post_staged(outcome.case, service)

    def acked_scenarios(self) -> set[str]:
        result = set()
        for path in sorted((self.root / "acked").glob("*.json")):
            value = json.loads(path.read_bytes())
            result.add(value["payload"]["scenario_id"])
        return result

    def verify_remote(self, service: EvaluationService, *, require_complete: bool) -> dict[str, Any]:
        response = service.get_evaluation(self.assignment.evaluation_id)
        report = response.get("evaluation")
        if not isinstance(report, Mapping):
            raise FixedPanelValidationError("evaluation status response is malformed")
        if (
            report.get("evaluation_id") != self.assignment.evaluation_id
            or report.get("run_id") != self.assignment.run_id
            or report.get("round_id") != self.assignment.round_id
            or report.get("checkpoint_sha256") != self.assignment.checkpoint_sha256
            or report.get("scenario_manifest_sha256") != self.assignment.scenario_manifest_sha256
            or report.get("observation_mode") != OBSERVATION_MODE
            or report.get("metric") != METRIC
            or report.get("total") != EXPECTED_EPISODES
            or (
                self.assignment.round_id > 0
                and report.get("purpose") not in {"validation", "test"}
            )
        ):
            raise FixedPanelValidationError("remote evaluation immutable identity mismatch")
        local = self.acked_scenarios()
        expected = {item.scenario_id for item in self.assignment.episodes}
        if not local.issubset(expected):
            raise FixedPanelValidationError("local acknowledgement includes an unknown scenario")
        if require_complete and (
            local != expected
            or report.get("status") != "complete"
            or report.get("completed") != EXPECTED_EPISODES
        ):
            raise FixedPanelValidationError("fixed-panel evaluation is not exactly complete")
        if report.get("completed", 0) < len(local):
            raise FixedPanelValidationError("remote service lost an acknowledged outcome")
        return dict(report)

    def write_summary(self, service: EvaluationService) -> Path:
        report = self.verify_remote(service, require_complete=True)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "assignment_sha256": self.assignment.assignment_sha256,
            "evaluation_id": self.assignment.evaluation_id,
            "run_id": self.assignment.run_id,
            "round_id": self.assignment.round_id,
            "checkpoint_sha256": self.assignment.checkpoint_sha256,
            "scenario_manifest_sha256": self.assignment.scenario_manifest_sha256,
            "observation_mode": OBSERVATION_MODE,
            "action_policy": ACTION_POLICY,
            "metric": METRIC,
            "views": list(EXPECTED_VIEWS),
            "completed_episodes": report["completed"],
            "successes": report["successes"],
            "goal_reached_rate": report["goal_reached_rate"],
            "wilson_95_interval": report["wilson_95_interval"],
            "mean_reward": report["mean_reward"],
            "mean_steps": report["mean_steps"],
            "safety_violations": report["safety_violations"],
        }
        path = self.root / "summary.json"
        atomic_write_json(path, summary)
        return path


def assignment_from_environment() -> FixedPanelAssignment:
    def required(name: str) -> str:
        value = os.environ.get(name, "")
        if not value:
            raise FixedPanelValidationError(f"required environment variable {name} is unset")
        return value

    return FixedPanelAssignment.load(
        Path(required("DAGGER_FIXED_PANEL_ASSIGNMENT")),
        expected_assignment_sha256=required("DAGGER_FIXED_PANEL_ASSIGNMENT_SHA256"),
        expected_evaluation_id=required("DAGGER_FIXED_PANEL_EVALUATION_ID"),
        expected_run_id=required("DAGGER_FIXED_PANEL_RUN_ID"),
        expected_round_id=int(required("DAGGER_FIXED_PANEL_ROUND_ID")),
        expected_checkpoint_sha256=required("DAGGER_FIXED_PANEL_CHECKPOINT_SHA256"),
        expected_scenario_manifest_sha256=required(
            "DAGGER_FIXED_PANEL_SCENARIO_MANIFEST_SHA256"
        ),
    )
