"""Fail-closed shadow-runtime scale ladder for one frozen round actor.

The scale ladder is deliberately narrower than a production campaign.  It
reuses every actor/model/renderer/source identity from an immutable, exact
legacy-versus-shadow runtime A/B result and changes only the local Isaac batch
size.  Each rung is one rank, two train-purpose steps, metadata-only shadow
observation, and one of 2/4/8/16/32 local environments.

This module owns artifacts and authorization; Docker and Isaac are launched by
``scripts/round_dagger/run_runtime_adapter_shadow_scale_docker.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .actor_pair import ActorPairError, inspect_actor_rank
from .artifacts import (
    ArtifactError,
    create_cohort,
    discover_rollout_shards,
    load_strict_json,
    publish_immutable_json,
    seal_cohort,
    sha256_file,
    verify_cohort,
    verify_rollout_shard,
)
from .runtime import (
    RUNTIME_ADAPTER_CUBLAS_WORKSPACE_CONFIG,
    RUNTIME_ADAPTER_EVIDENCE_ENV,
    RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA,
    RUNTIME_ADAPTER_INPUT_PROFILE_ENV,
    RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
    RUNTIME_ADAPTER_MODE_ENV,
    RUNTIME_ADAPTER_SCALE_VALIDATION_ACK,
    RUNTIME_ADAPTER_SHADOW_ACK,
    RUNTIME_ADAPTER_SHADOW_ACK_ENV,
    RUNTIME_ADAPTER_VALIDATION_ACK_ENV,
)
from .runtime_ab import RESULT_SCHEMA as RUNTIME_AB_RESULT_SCHEMA
from .runtime_ab import compare_runtime_ab


PLAN_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_scale_plan.v1"
CONFIG_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_scale_config.v1"
RUNG_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_scale_rung.v1"
EVIDENCE_SCHEMA = "fast_roborender.round_runtime_scale_validation.v1"
VALIDATION_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_scale_validation.v1"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_scale_telemetry.v1"
AUTHORIZATION_SCHEMA = (
    "fast_roborender.round_dagger.runtime_adapter_scale_authorization.v1"
)

SCALE_ACK = "bounded-runtime-adapter-shadow-scale-v1"
CAMPAIGN_ID = "rtx-runtime-adapter-shadow-scale-v1"
RUNTIME_SCALE_VALIDATION_ACK = RUNTIME_ADAPTER_SCALE_VALIDATION_ACK
RUNTIME_SCALE_PLAN_ENV = "ROUND_DAGGER_RUNTIME_SCALE_PLAN"
RUNTIME_SCALE_PLAN_SHA256_ENV = "ROUND_DAGGER_RUNTIME_SCALE_PLAN_SHA256"
RUNTIME_SCALE_RUNG_CONTRACT_ENV = "ROUND_DAGGER_RUNTIME_SCALE_RUNG_CONTRACT"
RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV = (
    "ROUND_DAGGER_RUNTIME_SCALE_RUNG_CONTRACT_SHA256"
)
RUNTIME_SCALE_EVIDENCE_ENV = "ROUND_DAGGER_RUNTIME_SCALE_EVIDENCE"

RANKS = 1
STEPS_PER_ENVIRONMENT = 2
PURPOSE = "train"
MACRO_ROUND = 0
RUNGS = (2, 4, 8, 16, 32)
DEFAULT_TORCH_HOME = Path(
    "/mnt/data2/ldu/roborender_round_dagger_validation/torch_cache"
)
RESNET18_FILENAME = "resnet18-f37072fd.pth"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
REFERENCE_LAG = 256
SIMULATOR_SEED = 0
IMAGE3F_SEED = 42
VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [26]}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PLAN_KEYS = {
    "acknowledgement",
    "campaign_id",
    "config_path",
    "config_sha256",
    "input_profile",
    "macro_round",
    "prerequisite_plan",
    "prerequisite_plan_sha256",
    "prerequisite_result",
    "prerequisite_result_sha256",
    "purpose",
    "ranks",
    "repository_root",
    "run_id",
    "rung_contracts",
    "rungs",
    "schema",
    "steps_per_environment",
    "torch_home",
    "torch_resnet18_sha256",
}
_RUNG_REFERENCE_KEYS = {"environments_per_rank", "path", "sha256"}
_RUNG_KEYS = {
    "campaign_id",
    "cohort_contract_sha256",
    "cohort_dir",
    "cohort_index",
    "config_sha256",
    "environments_per_rank",
    "evidence_path",
    "global_env_offset",
    "image3f_run_id",
    "macro_round",
    "purpose",
    "ranks",
    "run_dir",
    "schema",
    "steps_per_environment",
}
_AB_RESULT_KEYS = {
    "arms",
    "canonical_rollout_sha256",
    "exact_array_match",
    "exact_payload_match",
    "exact_raw_policy_input_match",
    "global_env_id",
    "input_profile",
    "performance",
    "plan",
    "plan_sha256",
    "schema",
    "shared_config_sha256",
    "source",
    "status",
    "transition_count_per_arm",
    "warmup",
}
_ALLOWED_INPUT_PROFILES = frozenset(
    {
        RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
        RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA,
    }
)


class RuntimeScaleError(RuntimeError):
    """A scale plan, prerequisite, environment, or result is unsafe."""


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        missing = sorted(keys - set(value)) if isinstance(value, Mapping) else sorted(keys)
        extra = sorted(set(value) - keys) if isinstance(value, Mapping) else []
        raise RuntimeScaleError(
            f"{label} key-set mismatch: missing={missing}, extra={extra}"
        )
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeScaleError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise RuntimeScaleError(f"{label} must be a safe non-empty token")
    return value


def _canonical_path(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or os.path.normpath(str(candidate)) != str(candidate):
        raise RuntimeScaleError(f"{label} must be an absolute canonical path")
    return candidate


def _absolute_file(path: str | Path, label: str, *, immutable: bool = False) -> Path:
    candidate = _canonical_path(path, label)
    try:
        info = os.lstat(candidate)
    except OSError as exc:
        raise RuntimeScaleError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or candidate.resolve(strict=True) != candidate:
        raise RuntimeScaleError(f"{label} must be a regular non-symlink file")
    if immutable and stat.S_IMODE(info.st_mode) != 0o444:
        raise RuntimeScaleError(f"{label} mode must be exactly 0444")
    return candidate


def _absolute_directory(
    path: str | Path, label: str, *, sealed: bool = False
) -> Path:
    candidate = _canonical_path(path, label)
    try:
        info = os.lstat(candidate)
    except OSError as exc:
        raise RuntimeScaleError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or candidate.resolve(strict=True) != candidate:
        raise RuntimeScaleError(f"{label} must be a non-symlink directory")
    if sealed and stat.S_IMODE(info.st_mode) != 0o555:
        raise RuntimeScaleError(f"{label} mode must be exactly 0555")
    return candidate


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeScaleError(f"{label} must be numeric")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0.0):
        raise RuntimeScaleError(f"{label} must be finite and positive")
    return number


def _require_metric(
    metrics: Mapping[str, Any], name: str, expected: int, *, step: int
) -> None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeScaleError(
            f"scale telemetry step {step} metric {name} is not numeric"
        )
    if float(value) != float(expected):
        raise RuntimeScaleError(
            f"scale telemetry step {step} metric {name} must equal {expected}"
        )


def _capacity_scope(environments_per_rank: int | None = None) -> dict[str, Any]:
    """Describe exactly what a rung PASS proves, without implying production."""

    scope: dict[str, Any] = {
        "authoritative_rollout": False,
        "classification": "bounded-shadow-execution-capacity-only",
        "excludes": [
            "two-rank-actor-concurrency",
            "three-hundred-step-duration",
            "two-hundred-fifty-six-simultaneous-environments",
            "distributed-learner-training",
            "production-promotion",
        ],
        "processes": RANKS,
        "promotion_gate": False,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
    }
    if environments_per_rank is not None:
        scope["local_environments"] = environments_per_rank
    return scope


def _prepare_root(path: str | Path) -> Path:
    target = _canonical_path(path, "scale root")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise RuntimeScaleError("scale root must be a non-symlink directory")
        if (target / "plan.json").exists():
            return target
        if any(target.iterdir()):
            raise RuntimeScaleError("scale root contains an incomplete immutable plan")
    else:
        target.mkdir(parents=True)
    return _absolute_directory(target, "scale root")


def _validate_prerequisite(
    result_path: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Recompute the exact A/B comparison and return its immutable bindings."""

    result_file = _absolute_file(
        result_path, "runtime A/B result", immutable=True
    )
    result_sha = sha256_file(result_file)
    result = _exact_mapping(
        load_strict_json(result_file), _AB_RESULT_KEYS, "runtime A/B result"
    )
    if result["schema"] != RUNTIME_AB_RESULT_SCHEMA or result["status"] != "PASS":
        raise RuntimeScaleError("runtime A/B prerequisite is not an exact PASS result")
    for key in (
        "exact_array_match",
        "exact_payload_match",
        "exact_raw_policy_input_match",
    ):
        if result[key] is not True:
            raise RuntimeScaleError(f"runtime A/B prerequisite {key} is not true")
    if result["transition_count_per_arm"] != 2 or result["global_env_id"] != 0:
        raise RuntimeScaleError("runtime A/B prerequisite is not the exact 1x1x2 gate")
    profile = result["input_profile"]
    if profile not in _ALLOWED_INPUT_PROFILES:
        raise RuntimeScaleError(
            "runtime A/B prerequisite must use live Isaac or deterministic CUDA input"
        )
    _sha(result["canonical_rollout_sha256"], "canonical rollout")
    _sha(result["shared_config_sha256"], "runtime A/B shared config")
    plan_path = _absolute_file(result["plan"], "runtime A/B plan", immutable=True)
    if sha256_file(plan_path) != result["plan_sha256"]:
        raise RuntimeScaleError("runtime A/B prerequisite plan digest mismatch")

    # This is intentionally stronger than trusting the PASS JSON: it reloads
    # both sealed cohorts, their runtime evidence and scalar telemetry, repeats
    # the exact array/policy-input comparison, and rejects any changed byte.
    recomputed = compare_runtime_ab(plan_path, output=result_file)
    if recomputed.get("result_sha256") != result_sha:
        raise RuntimeScaleError("runtime A/B prerequisite changed during validation")
    plan = load_strict_json(plan_path)
    if plan.get("input_profile") != profile:
        raise RuntimeScaleError("runtime A/B result and plan input profiles differ")
    repository = _absolute_directory(
        plan.get("repository_root", ""), "sealed repository root", sealed=True
    )
    if result.get("source", {}).get("repository_root") != str(repository):
        raise RuntimeScaleError("runtime A/B source root identity mismatch")
    return result_file, result, plan_path, dict(plan)


def _expected_config(
    *,
    prerequisite_result_sha256: str,
    prerequisite_plan_sha256: str,
    input_profile: str,
    torch_resnet18_sha256: str,
) -> dict[str, Any]:
    return {
        "acknowledgement": SCALE_ACK,
        "environments_per_rank": list(RUNGS),
        "input_profile": input_profile,
        "prerequisite_plan_sha256": prerequisite_plan_sha256,
        "prerequisite_result_sha256": prerequisite_result_sha256,
        "purpose": PURPOSE,
        "ranks": RANKS,
        "runtime_adapter_mode": "shadow",
        "runtime_validation_ack": RUNTIME_SCALE_VALIDATION_ACK,
        "schema": CONFIG_SCHEMA,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "torch_resnet18_sha256": torch_resnet18_sha256,
        "validation_only": True,
    }


def _expected_rung(
    root: Path,
    *,
    run_id: str,
    environments_per_rank: int,
    cohort_index: int,
    config_sha256: str,
    cohort_contract_sha256: str,
) -> dict[str, Any]:
    rung_root = root / f"envs_{environments_per_rank:03d}"
    run_dir = rung_root / "run"
    return {
        "campaign_id": CAMPAIGN_ID,
        "cohort_contract_sha256": cohort_contract_sha256,
        "cohort_dir": str(rung_root / "cohort"),
        "cohort_index": cohort_index,
        "config_sha256": config_sha256,
        "environments_per_rank": environments_per_rank,
        "evidence_path": str(
            run_dir / "ranks" / "rank_0000" / "runtime_adapter_evidence.json"
        ),
        "global_env_offset": 0,
        "image3f_run_id": f"{run_id}-e{environments_per_rank:03d}",
        "macro_round": MACRO_ROUND,
        "purpose": PURPOSE,
        "ranks": RANKS,
        "run_dir": str(run_dir),
        "schema": RUNG_SCHEMA,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
    }


def prepare_runtime_scale(
    root: str | Path,
    *,
    run_id: str,
    runtime_ab_result: str | Path,
    torch_home: str | Path = DEFAULT_TORCH_HOME,
) -> dict[str, Any]:
    """Create the immutable five-rung scale plan from one exact A/B PASS."""

    target = _prepare_root(root)
    run_token = _token(run_id, "scale run ID")
    result_path, result, ab_plan_path, ab_plan = _validate_prerequisite(
        runtime_ab_result
    )
    result_sha = sha256_file(result_path)
    ab_plan_sha = sha256_file(ab_plan_path)
    cache_root = _absolute_directory(torch_home, "scale TORCH_HOME")
    resnet = _absolute_file(
        cache_root / "hub" / "checkpoints" / RESNET18_FILENAME,
        "cached ResNet18 checkpoint",
    )
    resnet_sha = sha256_file(resnet)
    if resnet_sha != RESNET18_SHA256:
        raise RuntimeScaleError("cached ResNet18 checkpoint digest mismatch")
    existing = target / "plan.json"
    if existing.exists() or existing.is_symlink():
        report = validate_runtime_scale(existing)
        plan = load_strict_json(existing)
        if (
            plan["run_id"] != run_token
            or plan["prerequisite_result"] != str(result_path)
            or plan["prerequisite_result_sha256"] != result_sha
            or plan["torch_home"] != str(cache_root)
        ):
            raise RuntimeScaleError("existing immutable scale plan identity differs")
        return report

    config = _expected_config(
        prerequisite_result_sha256=result_sha,
        prerequisite_plan_sha256=ab_plan_sha,
        input_profile=result["input_profile"],
        torch_resnet18_sha256=resnet_sha,
    )
    config_path = target / "config.json"
    config_sha = publish_immutable_json(config_path, config)
    references = []
    for cohort_index, environments in enumerate(RUNGS):
        rung_root = target / f"envs_{environments:03d}"
        cohort_dir = rung_root / "cohort"
        create_cohort(
            cohort_dir,
            campaign_id=CAMPAIGN_ID,
            macro_round=MACRO_ROUND,
            cohort_index=cohort_index,
            purpose=PURPOSE,
            expected_ranks=[0],
            expected_global_env_ids=list(range(environments)),
            expected_transitions=environments * STEPS_PER_ENVIRONMENT,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            policy_sha256=ab_plan["policy_sha256"],
            runtime_policy_sha256=ab_plan["runtime_policy_sha256"],
            teacher_sha256=ab_plan["teacher_sha256"],
            renderer_sha256=ab_plan["renderer_sha256"],
            beta_sha256=ab_plan["beta_sha256"],
            config_sha256=config_sha,
        )
        rung = _expected_rung(
            target,
            run_id=run_token,
            environments_per_rank=environments,
            cohort_index=cohort_index,
            config_sha256=config_sha,
            cohort_contract_sha256=sha256_file(cohort_dir / "contract.json"),
        )
        rung_path = rung_root / "rung.json"
        rung_sha = publish_immutable_json(rung_path, rung)
        references.append(
            {
                "environments_per_rank": environments,
                "path": str(rung_path),
                "sha256": rung_sha,
            }
        )
    plan = {
        "acknowledgement": SCALE_ACK,
        "campaign_id": CAMPAIGN_ID,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "input_profile": result["input_profile"],
        "macro_round": MACRO_ROUND,
        "prerequisite_plan": str(ab_plan_path),
        "prerequisite_plan_sha256": ab_plan_sha,
        "prerequisite_result": str(result_path),
        "prerequisite_result_sha256": result_sha,
        "purpose": PURPOSE,
        "ranks": RANKS,
        "repository_root": ab_plan["repository_root"],
        "run_id": run_token,
        "rung_contracts": references,
        "rungs": list(RUNGS),
        "schema": PLAN_SCHEMA,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "torch_home": str(cache_root),
        "torch_resnet18_sha256": resnet_sha,
    }
    plan_path = target / "plan.json"
    publish_immutable_json(plan_path, plan)
    return validate_runtime_scale(plan_path)


def _load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = _absolute_file(path, "runtime scale plan", immutable=True)
    plan = _exact_mapping(
        load_strict_json(plan_path), _PLAN_KEYS, "runtime scale plan"
    )
    if plan["schema"] != PLAN_SCHEMA or plan["acknowledgement"] != SCALE_ACK:
        raise RuntimeScaleError("runtime scale plan schema or acknowledgement mismatch")
    if (
        plan["campaign_id"] != CAMPAIGN_ID
        or plan["macro_round"] != MACRO_ROUND
        or plan["purpose"] != PURPOSE
        or plan["ranks"] != RANKS
        or plan["steps_per_environment"] != STEPS_PER_ENVIRONMENT
        or plan["rungs"] != list(RUNGS)
    ):
        raise RuntimeScaleError("runtime scale plan is outside the bounded contract")
    _token(plan["run_id"], "runtime scale run ID")
    if plan["input_profile"] not in _ALLOWED_INPUT_PROFILES:
        raise RuntimeScaleError("runtime scale plan input profile is not allowed")
    return plan_path, plan


def _load_rungs(plan: Mapping[str, Any]) -> dict[int, tuple[Path, dict[str, Any]]]:
    references = plan["rung_contracts"]
    if not isinstance(references, list) or len(references) != len(RUNGS):
        raise RuntimeScaleError("runtime scale plan must contain exactly five rungs")
    result = {}
    for index, expected_environments in enumerate(RUNGS):
        reference = _exact_mapping(
            references[index], _RUNG_REFERENCE_KEYS, "runtime scale rung reference"
        )
        if reference["environments_per_rank"] != expected_environments:
            raise RuntimeScaleError("runtime scale rung order differs from contract")
        path = _absolute_file(
            reference["path"], f"runtime scale rung {expected_environments}", immutable=True
        )
        if sha256_file(path) != reference["sha256"]:
            raise RuntimeScaleError("runtime scale rung digest mismatch")
        document = _exact_mapping(
            load_strict_json(path), _RUNG_KEYS, "runtime scale rung"
        )
        result[expected_environments] = (path, document)
    return result


def _renderer_telemetry(
    document: Mapping[str, Any], *, environments_per_rank: int
) -> dict[str, Any]:
    """Require two immutable, batch-bound renderer records for one rung."""

    scalar_dir = _absolute_directory(
        Path(document["run_dir"]) / "telemetry" / "scalars",
        "scale scalar telemetry directory",
    )
    expected_paths = tuple(
        scalar_dir / f"step_{step:09d}.json"
        for step in range(STEPS_PER_ENVIRONMENT)
    )
    actual_paths = tuple(sorted(scalar_dir.glob("step_*.json")))
    if actual_paths != expected_paths:
        raise RuntimeScaleError(
            "scale scalar telemetry must contain exactly steps 0 and 1"
        )

    records = []
    inference_values = []
    for step, candidate in enumerate(expected_paths):
        path = _absolute_file(
            candidate, f"scale scalar telemetry step {step}", immutable=True
        )
        digest = sha256_file(path)
        scalar = _exact_mapping(
            load_strict_json(path),
            {"metrics", "step"},
            f"scale scalar telemetry step {step}",
        )
        if sha256_file(path) != digest:
            raise RuntimeScaleError(
                f"scale scalar telemetry step {step} changed while loading"
            )
        if type(scalar["step"]) is not int or scalar["step"] != step:
            raise RuntimeScaleError(
                f"scale scalar telemetry document step must equal {step}"
            )
        if not isinstance(scalar["metrics"], Mapping):
            raise RuntimeScaleError(
                f"scale scalar telemetry step {step} metrics must be a mapping"
            )
        metrics = dict(scalar["metrics"])
        for name, expected in (
            ("renderer/parallel_batch_size", environments_per_rank),
            ("renderer/reference_provenance/configured_lag", REFERENCE_LAG),
            ("renderer/reference_provenance/request_step", step),
            ("renderer/reference_provenance/source_step", 0),
            ("renderer/reference_provenance_valid", 1),
            ("renderer/reference_source_anchor", 1),
            ("renderer/reference_source_client", 0),
            ("renderer/reference_source_generated_lag", 0),
            ("renderer/reference_source_idempotent_replay", 0),
            ("renderer/reference_source_unknown", 0),
            ("renderer/stream_step", step),
            ("rollout_frame_step", step),
            ("telemetry/view_count", len(VIEW_CONTRACT["views"])),
        ):
            _require_metric(metrics, name, expected, step=step)

        finite_metrics = {}
        for name in (
            "renderer/client_wall_ms",
            "renderer/inference_ms",
            "renderer/ms",
            "renderer/parallel_batch_wall_ms",
        ):
            finite_metrics[name] = _finite_number(
                metrics.get(name),
                f"scale telemetry step {step} {name}",
                positive=True,
            )
        queue_ms = _finite_number(
            metrics.get("renderer/queue_ms"),
            f"scale telemetry step {step} renderer/queue_ms",
        )
        if queue_ms < 0.0:
            raise RuntimeScaleError(
                f"scale telemetry step {step} renderer/queue_ms must be non-negative"
            )
        finite_metrics["renderer/queue_ms"] = queue_ms
        inference_values.append(finite_metrics["renderer/inference_ms"])
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "step": step,
                "parallel_batch_size": environments_per_rank,
                "reference_provenance_valid": 1,
                **finite_metrics,
            }
        )
    return {
        "schema": TELEMETRY_SCHEMA,
        "records": records,
        "renderer_inference_ms": {
            "count": len(inference_values),
            "mean": float(sum(inference_values) / len(inference_values)),
            "values": inference_values,
        },
    }


def _policy_rgb_content(
    document: Mapping[str, Any], *, environments_per_rank: int
) -> dict[str, Any]:
    """Prove the batched replay contains nondegenerate, per-env RGB outputs."""

    cohort_dir = _absolute_directory(document["cohort_dir"], "scale cohort")
    sidecars = discover_rollout_shards(cohort_dir)
    if len(sidecars) != 1:
        raise RuntimeScaleError(
            "scale content validation requires exactly one rank-0 rollout shard"
        )
    shard = verify_rollout_shard(sidecars[0])
    if (
        shard["rank"] != 0
        or shard["rollout_id"] != 0
        or shard["part_index"] != 0
        or shard["global_env_ids"] != list(range(environments_per_rank))
    ):
        raise RuntimeScaleError("scale content shard identity or coverage mismatch")
    payload = _absolute_file(
        cohort_dir / shard["payload_path"],
        "scale rollout payload",
        immutable=True,
    )
    payload_sha = sha256_file(payload)
    if payload_sha != shard["payload_sha256"]:
        raise RuntimeScaleError("scale content payload digest mismatch")
    try:
        with np.load(payload, allow_pickle=False) as archive:
            if "policy_rgb" not in archive.files:
                raise RuntimeScaleError("scale rollout payload has no policy_rgb")
            policy_rgb = np.ascontiguousarray(archive["policy_rgb"])
    except RuntimeScaleError:
        raise
    except Exception as exc:
        raise RuntimeScaleError(f"cannot load scale policy_rgb: {exc}") from exc
    if sha256_file(payload) != payload_sha:
        raise RuntimeScaleError("scale rollout payload changed while loading")
    expected_shape = (
        STEPS_PER_ENVIRONMENT,
        environments_per_rank,
        *VIEW_CONTRACT["shape"],
    )
    if policy_rgb.dtype != np.uint8 or policy_rgb.shape != expected_shape:
        raise RuntimeScaleError(
            "scale policy_rgb dtype or shape mismatch: "
            f"got {policy_rgb.dtype} {policy_rgb.shape}, expected uint8 {expected_shape}"
        )
    if not np.isfinite(policy_rgb).all():
        raise RuntimeScaleError("scale policy_rgb contains non-finite values")

    step_summaries = []
    for step in range(STEPS_PER_ENVIRONMENT):
        batch = policy_rgb[step]
        minimum = int(batch.min())
        maximum = int(batch.max())
        nonzero_pixels = int(np.count_nonzero(batch))
        if minimum == maximum or nonzero_pixels == 0:
            raise RuntimeScaleError(
                f"scale policy_rgb step {step} is zero or has degenerate range"
            )
        environment_hashes = [
            hashlib.sha256(np.ascontiguousarray(batch[index]).tobytes(order="C")).hexdigest()
            for index in range(environments_per_rank)
        ]
        if len(set(environment_hashes)) != environments_per_rank:
            raise RuntimeScaleError(
                f"scale policy_rgb step {step} is not unique per environment"
            )
        step_summaries.append(
            {
                "step": step,
                "minimum": minimum,
                "maximum": maximum,
                "nonzero_pixels": nonzero_pixels,
                "unique_environment_count": len(set(environment_hashes)),
                "environment_sha256": environment_hashes,
            }
        )
    return {
        "dtype": "uint8",
        "shape": list(expected_shape),
        "payload_path": str(payload),
        "payload_sha256": payload_sha,
        "steps": step_summaries,
    }


def _validate_runtime_evidence(
    document: Mapping[str, Any], *, environments_per_rank: int, input_profile: str
) -> dict[str, Any]:
    path = _absolute_file(document["evidence_path"], "scale runtime evidence", immutable=True)
    if path.parent != Path(document["run_dir"]) / "ranks" / "rank_0000":
        raise RuntimeScaleError("scale runtime evidence is not rank-specific")
    digest = sha256_file(path)
    raw = _exact_mapping(
        load_strict_json(path),
        {
            "adapter_arm",
            "audit",
            "cuda_determinism",
            "dimensions",
            "input_profile",
            "policy_inputs",
            "promotion_gate",
            "rank",
            "replay_export",
            "rollout",
            "schema",
            "status",
            "validation_ack",
            "world_size",
        },
        "scale runtime evidence",
    )
    if sha256_file(path) != digest:
        raise RuntimeScaleError("scale runtime evidence changed while loading")
    expected_scalars = {
        "schema": EVIDENCE_SCHEMA,
        "status": "informational_only",
        "promotion_gate": False,
        "validation_ack": RUNTIME_SCALE_VALIDATION_ACK,
        "adapter_arm": "shadow",
        "input_profile": input_profile,
        "rank": 0,
        "world_size": 1,
    }
    mismatch = [key for key, value in expected_scalars.items() if raw[key] != value]
    if mismatch:
        raise RuntimeScaleError(
            "scale runtime evidence identity mismatch: " + ", ".join(mismatch)
        )
    if raw["dimensions"] != {
        "local_num_envs": environments_per_rank,
        "rollout_steps": STEPS_PER_ENVIRONMENT,
    }:
        raise RuntimeScaleError("scale runtime evidence dimensions mismatch")
    audit = _exact_mapping(
        raw["audit"],
        {
            "batch_dimensions_valid",
            "callback_count",
            "callback_order_valid",
            "decision_count",
            "enabled",
            "expected_steps",
            "num_envs",
            "sequential_steps_valid",
            "transition_count",
        },
        "scale shadow audit",
    )
    expected_audit = {
        "batch_dimensions_valid": True,
        "callback_count": 3 * STEPS_PER_ENVIRONMENT,
        "callback_order_valid": True,
        "decision_count": STEPS_PER_ENVIRONMENT,
        "enabled": True,
        "expected_steps": STEPS_PER_ENVIRONMENT,
        "num_envs": environments_per_rank,
        "sequential_steps_valid": True,
        # The audit counts batched callbacks; replay coverage below counts
        # individual environment transitions.
        "transition_count": STEPS_PER_ENVIRONMENT,
    }
    if audit != expected_audit:
        raise RuntimeScaleError("scale shadow audit counters or invariants mismatch")
    policy_inputs = _exact_mapping(
        raw["policy_inputs"], {"key", "records"}, "scale policy inputs"
    )
    if policy_inputs["key"] != "vision_obs":
        raise RuntimeScaleError("scale policy input key mismatch")
    records = policy_inputs["records"]
    if not isinstance(records, list) or len(records) != STEPS_PER_ENVIRONMENT:
        raise RuntimeScaleError("scale policy input record count mismatch")
    for step, record in enumerate(records):
        item = _exact_mapping(
            record, {"dtype", "sha256", "shape", "step"}, "scale policy input"
        )
        if (
            item["step"] != step
            or item["dtype"] != "torch.float32"
            or item["shape"] != [environments_per_rank, *VIEW_CONTRACT["shape"]]
        ):
            raise RuntimeScaleError("scale policy input metadata mismatch")
        _sha(item["sha256"], "scale policy input")
    replay = _exact_mapping(
        raw["replay_export"], {"published_shard_parts", "successful"}, "scale replay"
    )
    if replay != {"published_shard_parts": 1, "successful": True}:
        raise RuntimeScaleError("scale replay export was not exactly successful")
    rollout = _exact_mapping(
        raw["rollout"], {"cuda", "synchronized", "wall_seconds"}, "scale rollout"
    )
    if rollout["synchronized"] is not True:
        raise RuntimeScaleError("scale CUDA rollout was not synchronized")
    wall_seconds = _finite_number(
        rollout["wall_seconds"], "scale rollout wall_seconds", positive=True
    )
    cuda = _exact_mapping(
        rollout["cuda"],
        {
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "device",
            "enabled",
            "peak_allocated_bytes",
            "peak_delta_allocated_bytes",
            "peak_delta_reserved_bytes",
            "peak_reserved_bytes",
        },
        "scale CUDA metrics",
    )
    if cuda["enabled"] is not True or not str(cuda["device"]).startswith("cuda"):
        raise RuntimeScaleError("scale runtime did not execute on CUDA")
    for key in (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_delta_allocated_bytes",
        "peak_delta_reserved_bytes",
        "peak_reserved_bytes",
    ):
        if type(cuda[key]) is not int or cuda[key] < 0:
            raise RuntimeScaleError(f"scale CUDA metric {key} is malformed")
    if (
        cuda["peak_allocated_bytes"] - cuda["baseline_allocated_bytes"]
        != cuda["peak_delta_allocated_bytes"]
        or cuda["peak_reserved_bytes"] - cuda["baseline_reserved_bytes"]
        != cuda["peak_delta_reserved_bytes"]
    ):
        raise RuntimeScaleError("scale CUDA peak deltas are inconsistent")
    deterministic = _exact_mapping(
        raw["cuda_determinism"],
        {
            "cublas_workspace_config",
            "cudnn_allow_tf32",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
            "deterministic_warn_only",
            "enabled",
            "matmul_allow_tf32",
            "matmul_precision",
        },
        "scale CUDA determinism",
    )
    deterministic_profile = (
        input_profile == RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA
    )
    if bool(deterministic["enabled"]) != deterministic_profile:
        raise RuntimeScaleError("scale CUDA determinism/profile mismatch")
    if deterministic_profile:
        expected = {
            "cublas_workspace_config": RUNTIME_ADAPTER_CUBLAS_WORKSPACE_CONFIG,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "enabled": True,
            "matmul_allow_tf32": False,
            "matmul_precision": "highest",
        }
        if deterministic != expected:
            raise RuntimeScaleError("scale deterministic CUDA settings mismatch")
    telemetry = _renderer_telemetry(
        document, environments_per_rank=environments_per_rank
    )
    content = _policy_rgb_content(
        document, environments_per_rank=environments_per_rank
    )
    return {
        "path": str(path),
        "sha256": digest,
        "wall_seconds": wall_seconds,
        "cuda": cuda,
        "telemetry": telemetry,
        "policy_rgb_content": content,
    }


def validate_runtime_scale(path: str | Path) -> dict[str, Any]:
    """Revalidate the prerequisite, plan, contracts, and current rung states."""

    plan_path, plan = _load_plan(path)
    result_path, result, ab_plan_path, ab_plan = _validate_prerequisite(
        plan["prerequisite_result"]
    )
    bindings = {
        "prerequisite_result": str(result_path),
        "prerequisite_result_sha256": sha256_file(result_path),
        "prerequisite_plan": str(ab_plan_path),
        "prerequisite_plan_sha256": sha256_file(ab_plan_path),
        "repository_root": ab_plan["repository_root"],
        "input_profile": result["input_profile"],
    }
    mismatch = [key for key, value in bindings.items() if plan[key] != value]
    if mismatch:
        raise RuntimeScaleError(
            "scale plan differs from exact A/B prerequisite: " + ", ".join(mismatch)
        )
    _absolute_directory(plan["repository_root"], "sealed repository root", sealed=True)
    torch_home = _absolute_directory(plan["torch_home"], "scale TORCH_HOME")
    resnet = _absolute_file(
        torch_home / "hub" / "checkpoints" / RESNET18_FILENAME,
        "cached ResNet18 checkpoint",
    )
    if (
        plan["torch_resnet18_sha256"] != RESNET18_SHA256
        or sha256_file(resnet) != plan["torch_resnet18_sha256"]
    ):
        raise RuntimeScaleError("plan-bound ResNet18 cache identity mismatch")
    config_path = _absolute_file(plan["config_path"], "scale config", immutable=True)
    if sha256_file(config_path) != plan["config_sha256"]:
        raise RuntimeScaleError("scale config digest mismatch")
    expected_config = _expected_config(
        prerequisite_result_sha256=plan["prerequisite_result_sha256"],
        prerequisite_plan_sha256=plan["prerequisite_plan_sha256"],
        input_profile=plan["input_profile"],
        torch_resnet18_sha256=plan["torch_resnet18_sha256"],
    )
    if load_strict_json(config_path) != expected_config:
        raise RuntimeScaleError("scale config content mismatch")

    reports = {}
    for cohort_index, environments in enumerate(RUNGS):
        rung_path, document = _load_rungs(plan)[environments]
        cohort_dir = _absolute_directory(document["cohort_dir"], "scale cohort")
        expected = _expected_rung(
            plan_path.parent,
            run_id=plan["run_id"],
            environments_per_rank=environments,
            cohort_index=cohort_index,
            config_sha256=plan["config_sha256"],
            cohort_contract_sha256=sha256_file(cohort_dir / "contract.json"),
        )
        if document != expected:
            raise RuntimeScaleError(f"scale rung {environments} content mismatch")
        contract = load_strict_json(cohort_dir / "contract.json")
        expected_contract = {
            "campaign_id": CAMPAIGN_ID,
            "macro_round": MACRO_ROUND,
            "cohort_index": cohort_index,
            "purpose": PURPOSE,
            "expected_ranks": [0],
            "expected_global_env_ids": list(range(environments)),
            "expected_transitions": environments * STEPS_PER_ENVIRONMENT,
            "expected_steps_per_env": STEPS_PER_ENVIRONMENT,
            "view_contract": VIEW_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "policy_sha256": ab_plan["policy_sha256"],
            "runtime_policy_sha256": ab_plan["runtime_policy_sha256"],
            "teacher_sha256": ab_plan["teacher_sha256"],
            "renderer_sha256": ab_plan["renderer_sha256"],
            "beta_sha256": ab_plan["beta_sha256"],
            "config_sha256": plan["config_sha256"],
        }
        mismatches = [
            key for key, value in expected_contract.items() if contract.get(key) != value
        ]
        if mismatches:
            raise RuntimeScaleError(
                f"scale rung {environments} cohort mismatch: {mismatches}"
            )
        evidence = None
        seal_path = cohort_dir / "seal.json"
        rank = inspect_actor_rank(cohort_dir, 0)
        evidence_path = Path(document["evidence_path"])
        if seal_path.exists() or seal_path.is_symlink():
            sealed = verify_cohort(cohort_dir)
            if (
                sealed["ranks"] != [0]
                or sealed["global_env_ids"] != list(range(environments))
                or sealed["transition_count"] != environments * STEPS_PER_ENVIRONMENT
            ):
                raise RuntimeScaleError("sealed scale rung coverage mismatch")
            evidence = _validate_runtime_evidence(
                document,
                environments_per_rank=environments,
                input_profile=plan["input_profile"],
            )
            state = "sealed"
        elif rank["state"] == "complete" and evidence_path.exists():
            evidence = _validate_runtime_evidence(
                document,
                environments_per_rank=environments,
                input_profile=plan["input_profile"],
            )
            state = "complete"
        elif rank["state"] == "empty" and not evidence_path.exists() and not Path(
            document["run_dir"]
        ).exists():
            state = "empty"
        else:
            state = "partial"
        reports[str(environments)] = {
            "cohort_dir": str(cohort_dir),
            "evidence": evidence,
            "expected_transitions": environments * STEPS_PER_ENVIRONMENT,
            "rung_contract": str(rung_path),
            "rung_contract_sha256": sha256_file(rung_path),
            "state": state,
        }
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "valid",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "acknowledgement": SCALE_ACK,
        "prerequisite": {
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "plan": str(ab_plan_path),
            "plan_sha256": sha256_file(ab_plan_path),
            "status": result["status"],
        },
        "source": result["source"],
        "actor_image": ab_plan["actor_image"],
        "actor_image_id_sha256": ab_plan["actor_image_id_sha256"],
        "actor_gpu_id": ab_plan["actor_gpu_id"],
        "input_profile": plan["input_profile"],
        "scope": _capacity_scope(),
        "rungs": reports,
    }


def _require_sealed_predecessors(
    report: Mapping[str, Any], environments_per_rank: int
) -> None:
    index = RUNGS.index(environments_per_rank)
    incomplete = [
        rung
        for rung in RUNGS[:index]
        if report["rungs"][str(rung)]["state"] != "sealed"
    ]
    if incomplete:
        raise RuntimeScaleError(
            f"runtime scale rung {environments_per_rank} requires sealed "
            "predecessors: " + ", ".join(map(str, incomplete))
        )


def runtime_scale_environment(path: str | Path, environments_per_rank: int) -> dict[str, str]:
    """Return the exact process environment for one still-empty scale rung."""

    if environments_per_rank not in RUNGS:
        raise RuntimeScaleError("runtime scale rung must be one of 2,4,8,16,32")
    report = validate_runtime_scale(path)
    _require_sealed_predecessors(report, environments_per_rank)
    rung_report = report["rungs"][str(environments_per_rank)]
    if rung_report["state"] != "empty":
        raise RuntimeScaleError(
            f"runtime scale rung {environments_per_rank} is "
            f"{rung_report['state']}; refusing overwrite"
        )
    plan_path, plan = _load_plan(path)
    _, result, _, ab_plan = _validate_prerequisite(plan["prerequisite_result"])
    rung_path, rung = _load_rungs(plan)[environments_per_rank]
    evidence_path = rung["evidence_path"]
    environment = {
        "ACCELERATE": ab_plan["accelerate"],
        "ANCHORS": ab_plan["anchors"],
        "DRY_RUN": "0",
        "GR00T_PYTHON": ab_plan["gr00t_python"],
        "GR00T_ROOT": ab_plan["gr00t_root"],
        "IMAGE3F_ENDPOINT_MANIFEST": ab_plan["selected_endpoint_manifest"],
        "IMAGE3F_RUN_ID": rung["image3f_run_id"],
        "ROBORENDER_TELEMETRY_DIR": f"{rung['run_dir']}/telemetry",
        "ROLLOUT_STEPS": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_ACTOR_DOCKER_GPU_ID": str(ab_plan["actor_gpu_id"]),
        "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": ab_plan[
            "anchor_inventory_sha256"
        ],
        "ROUND_DAGGER_BETA": "0.5",
        "ROUND_DAGGER_BETA_SHA256": ab_plan["beta_sha256"],
        "ROUND_DAGGER_CAMPAIGN_ID": CAMPAIGN_ID,
        "ROUND_DAGGER_COHORT_DIR": rung["cohort_dir"],
        "ROUND_DAGGER_COHORT_INDEX": str(rung["cohort_index"]),
        "ROUND_DAGGER_CONFIG_SHA256": plan["config_sha256"],
        "ROUND_DAGGER_ENVS_PER_SHARD": str(environments_per_rank),
        "ROUND_DAGGER_EXPECTED_RENDERERS": "1",
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": "0",
        "ROUND_DAGGER_IMAGE3F_SEED": str(IMAGE3F_SEED),
        "ROUND_DAGGER_MACRO_ROUND": str(MACRO_ROUND),
        "ROUND_DAGGER_MIN_ANCHORS": str(ab_plan["anchor_count"]),
        "ROUND_DAGGER_POLICY_CHECKPOINT": ab_plan["policy_checkpoint"],
        "ROUND_DAGGER_POLICY_MANIFEST": ab_plan["policy_manifest"],
        "ROUND_DAGGER_POLICY_SHA256": ab_plan["policy_sha256"],
        "ROUND_DAGGER_POLICY_SPEC": ab_plan["policy_spec"],
        "ROUND_DAGGER_RENDERER_SHA256": ab_plan["renderer_sha256"],
        "ROUND_DAGGER_REPOSITORY_ROOT": ab_plan["repository_root"],
        "ROUND_DAGGER_ROLLOUT_ID": "0",
        "ROUND_DAGGER_RTX_IMAGE": ab_plan["actor_image"],
        "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": ab_plan["actor_image_id_sha256"],
        RUNTIME_SCALE_PLAN_ENV: str(plan_path),
        RUNTIME_SCALE_PLAN_SHA256_ENV: report["plan_sha256"],
        RUNTIME_SCALE_RUNG_CONTRACT_ENV: str(rung_path),
        RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV: sha256_file(rung_path),
        RUNTIME_SCALE_EVIDENCE_ENV: evidence_path,
        RUNTIME_ADAPTER_EVIDENCE_ENV: evidence_path,
        RUNTIME_ADAPTER_INPUT_PROFILE_ENV: result["input_profile"],
        RUNTIME_ADAPTER_MODE_ENV: "shadow",
        RUNTIME_ADAPTER_SHADOW_ACK_ENV: RUNTIME_ADAPTER_SHADOW_ACK,
        RUNTIME_ADAPTER_VALIDATION_ACK_ENV: RUNTIME_SCALE_VALIDATION_ACK,
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": ab_plan["runtime_policy_sha256"],
        "ROUND_DAGGER_SIMULATOR_SEED": str(SIMULATOR_SEED),
        "ROUND_DAGGER_SOURCE_INVENTORY_SHA256": ab_plan[
            "source_inventory_sha256"
        ],
        "ROUND_DAGGER_TEACHER_CHECKPOINT": ab_plan["teacher_checkpoint"],
        "ROUND_DAGGER_TEACHER_SHA256": ab_plan["teacher_sha256"],
        "ROUND_DAGGER_VALIDATION_ACTOR_RANK": "0",
        "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": str(environments_per_rank),
        "ROUND_DAGGER_VALIDATION_PURPOSE": PURPOSE,
        "ROUND_DAGGER_VALIDATION_RANKS": str(RANKS),
        "ROUND_DAGGER_VALIDATION_STEPS": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_VALIDATION_TRAIN_ACK": "bounded-train-replay",
        "TRAIN_RUN_DIR": rung["run_dir"],
        "TORCH_HOME": plan["torch_home"],
    }
    if result["input_profile"] == RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_DETERMINISTIC_CUDA:
        environment["CUBLAS_WORKSPACE_CONFIG"] = RUNTIME_ADAPTER_CUBLAS_WORKSPACE_CONFIG
    return environment


def _environment_rung(
    plan: Mapping[str, Any], environment: Mapping[str, str]
) -> tuple[int, str]:
    raw_path = environment.get(RUNTIME_SCALE_RUNG_CONTRACT_ENV, "")
    raw_sha = environment.get(RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV, "")
    for environments, (path, _) in _load_rungs(plan).items():
        if raw_path == str(path):
            return environments, raw_sha
    raise RuntimeScaleError("process rung contract is absent from immutable scale plan")


def authorize_runtime_scale(
    path: str | Path,
    expected_plan_sha256: str,
    *,
    expected_rung_contract_sha256: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail unless the process environment exactly matches one empty rung."""

    expected_plan = _sha(expected_plan_sha256, "expected runtime scale plan")
    _, plan = _load_plan(path)
    observed = os.environ if environment is None else environment
    environments, observed_rung_sha = _environment_rung(plan, observed)
    expected_environment = runtime_scale_environment(path, environments)
    if expected_environment[RUNTIME_SCALE_PLAN_SHA256_ENV] != expected_plan:
        raise RuntimeScaleError("runtime scale plan SHA-256 differs from authorization")
    expected_rung_sha = expected_environment[RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV]
    if expected_rung_contract_sha256 is not None:
        supplied = _sha(
            expected_rung_contract_sha256, "expected runtime scale rung contract"
        )
        if supplied != expected_rung_sha:
            raise RuntimeScaleError("runtime scale rung SHA-256 differs from authorization")
    if observed_rung_sha != expected_rung_sha:
        raise RuntimeScaleError("process runtime scale rung digest mismatch")
    mismatches = [
        name for name, value in expected_environment.items() if observed.get(name) != value
    ]
    if mismatches:
        raise RuntimeScaleError(
            "runtime scale process environment differs from immutable plan: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "plan": expected_environment[RUNTIME_SCALE_PLAN_ENV],
        "plan_sha256": expected_plan,
        "rung_contract": expected_environment[RUNTIME_SCALE_RUNG_CONTRACT_ENV],
        "rung_contract_sha256": expected_rung_sha,
        "environments_per_rank": environments,
        "expected_transitions": environments * STEPS_PER_ENVIRONMENT,
        "runtime_adapter_mode": "shadow",
        "input_profile": expected_environment[RUNTIME_ADAPTER_INPUT_PROFILE_ENV],
    }


def seal_runtime_scale(path: str | Path, environments_per_rank: int) -> dict[str, Any]:
    """Validate completed rank/evidence, seal once, and reverify the rung."""

    if environments_per_rank not in RUNGS:
        raise RuntimeScaleError("runtime scale rung must be one of 2,4,8,16,32")
    report = validate_runtime_scale(path)
    _require_sealed_predecessors(report, environments_per_rank)
    state = report["rungs"][str(environments_per_rank)]["state"]
    if state == "sealed":
        return verify_runtime_scale(path, environments_per_rank)
    if state != "complete":
        raise RuntimeScaleError(
            f"runtime scale rung {environments_per_rank} is {state}; refusing seal"
        )
    _, plan = _load_plan(path)
    _, rung = _load_rungs(plan)[environments_per_rank]
    sealed = seal_cohort(rung["cohort_dir"])
    if sealed["transition_count"] != environments_per_rank * STEPS_PER_ENVIRONMENT:
        raise RuntimeScaleError("sealed runtime scale transition count mismatch")
    return verify_runtime_scale(path, environments_per_rank)


def verify_runtime_scale(path: str | Path, environments_per_rank: int) -> dict[str, Any]:
    """Require one rung's exact sealed cohort and immutable runtime evidence."""

    if environments_per_rank not in RUNGS:
        raise RuntimeScaleError("runtime scale rung must be one of 2,4,8,16,32")
    report = validate_runtime_scale(path)
    _require_sealed_predecessors(report, environments_per_rank)
    rung = report["rungs"][str(environments_per_rank)]
    if rung["state"] != "sealed":
        raise RuntimeScaleError(
            f"runtime scale rung {environments_per_rank} is not sealed"
        )
    return {
        "schema": "fast_roborender.round_dagger.runtime_adapter_scale_result.v1",
        "status": "PASS",
        "plan": report["plan"],
        "plan_sha256": report["plan_sha256"],
        "environments_per_rank": environments_per_rank,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "transition_count": environments_per_rank * STEPS_PER_ENVIRONMENT,
        "rung_contract": rung["rung_contract"],
        "rung_contract_sha256": rung["rung_contract_sha256"],
        "evidence": rung["evidence"],
        "scope": _capacity_scope(environments_per_rank),
        "source": report["source"],
        "prerequisite": report["prerequisite"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    prepare = operations.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--runtime-ab-result", type=Path, required=True)
    prepare.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    validate = operations.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    environment = operations.add_parser("environment")
    environment.add_argument("--plan", type=Path, required=True)
    environment.add_argument("--rung", type=int, choices=RUNGS, required=True)
    environment.add_argument("--null", action="store_true")
    authorize = operations.add_parser("authorize")
    authorize.add_argument("--plan", type=Path, required=True)
    authorize.add_argument("--expected-plan-sha256", required=True)
    authorize.add_argument("--expected-rung-contract-sha256")
    for name in ("seal", "verify"):
        operation = operations.add_parser(name)
        operation.add_argument("--plan", type=Path, required=True)
        operation.add_argument("--rung", type=int, choices=RUNGS, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "prepare":
        result = prepare_runtime_scale(
            arguments.root,
            run_id=arguments.run_id,
            runtime_ab_result=arguments.runtime_ab_result,
            torch_home=arguments.torch_home,
        )
    elif arguments.operation == "validate":
        result = validate_runtime_scale(arguments.plan)
    elif arguments.operation == "environment":
        result = runtime_scale_environment(arguments.plan, arguments.rung)
        if arguments.null:
            for key in sorted(result):
                sys.stdout.buffer.write(key.encode("ascii") + b"\0")
                sys.stdout.buffer.write(result[key].encode("utf-8") + b"\0")
            return 0
    elif arguments.operation == "authorize":
        result = authorize_runtime_scale(
            arguments.plan,
            arguments.expected_plan_sha256,
            expected_rung_contract_sha256=arguments.expected_rung_contract_sha256,
        )
    elif arguments.operation == "seal":
        result = seal_runtime_scale(arguments.plan, arguments.rung)
    else:
        result = verify_runtime_scale(arguments.plan, arguments.rung)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeScaleError, ActorPairError, ArtifactError, KeyError) as error:
        print(f"runtime scale gate failed: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "CAMPAIGN_ID",
    "PLAN_SCHEMA",
    "RUNGS",
    "RUNTIME_SCALE_EVIDENCE_ENV",
    "RUNTIME_SCALE_PLAN_ENV",
    "RUNTIME_SCALE_PLAN_SHA256_ENV",
    "RUNTIME_SCALE_RUNG_CONTRACT_ENV",
    "RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV",
    "RUNTIME_SCALE_VALIDATION_ACK",
    "RuntimeScaleError",
    "SCALE_ACK",
    "authorize_runtime_scale",
    "prepare_runtime_scale",
    "runtime_scale_environment",
    "seal_runtime_scale",
    "validate_runtime_scale",
    "verify_runtime_scale",
]
