#!/usr/bin/env python3
"""Fail-closed evidence gates for the fixed-256, step-100 H100 smoke.

This profile is deliberately separate from the historical two-update H100
gate.  It proves the production held-out cadence by evaluating the exact
fixed-256 snapshot at optimizer frontiers 0 and 100, logging those values to a
single authenticated W&B run, and comparing a stop-at-50/resume execution with
an uninterrupted 100-update execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.learner import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    HELDOUT_TELEMETRY_SCHEMA,
    RESULT_SCHEMA,
    TELEMETRY_SCHEMA,
)
from round_dagger.learner_identity import learner_wandb_run_id  # noqa: E402
from round_dagger.replay import load_heldout_snapshot  # noqa: E402
from scripts.round_dagger.h100_validation_wandb_readback import (  # noqa: E402
    _load_checkpoint,
    _semantic_identity,
)
from scripts.scaling.wandb_preflight import _load_wandb, query_identity  # noqa: E402


CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.heldout100_validation_contract.v1"
)
READBACK_SCHEMA = (
    "fast_roborender.round_dagger.h100_heldout100_wandb_readback.v1"
)
RESUME_SCHEMA = "fast_roborender.round_dagger.h100_heldout100_strict_resume.v1"
ATTESTATION_SCHEMA = (
    "fast_roborender.round_dagger.oci_nrt_h100_heldout100_validation.v1"
)
CONTRACT_RELATIVE_PATH = Path(
    "configs/round_dagger/validation_learner_1x8h100_heldout100.json"
)
BINDING_SCHEMA = "fast_roborender.round_dagger.heldout100_runtime_binding.v1"
TRAIN_STEPS = 100
STOP_STEP = 50
HELDOUT_STEPS = [0, 100]
WORLD_SIZE = 8
HELDOUT_CASES = 256
LEARNED_ACTION_INDICES = list(range(11))
MINIMUM_RELATIVE_ACTION_MSE_REDUCTION = 0.001
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

TRAIN_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "optimizer_step",
    "loss",
    "replay_transitions_rank",
    "replay_transitions_per_rank",
    "replay_transitions_global",
    "world_size",
    "gradient_reduction",
}
HELDOUT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "optimizer_step",
    "heldout_snapshot_sha256",
    "heldout_snapshot_file_sha256",
    "heldout_transitions_rank",
    "heldout_transitions_per_rank",
    "heldout_transitions_global",
    "world_size",
    "learned_action_indices",
    "action_mse",
    "action_mae",
    "action_rmse",
    "action_cosine",
    "per_action",
}
QUALITY_GLOBAL_KEYS = {
    "quality/heldout_action_mse": "action_mse",
    "quality/heldout_action_mae": "action_mae",
    "quality/heldout_action_rmse": "action_rmse",
    "quality/heldout_action_cosine": "action_cosine",
    "quality/heldout_transitions": "heldout_transitions_global",
}


class Heldout100ValidationError(RuntimeError):
    """The dedicated step-100 validation contract was not satisfied."""


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Heldout100ValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _regular(path: Path, label: str) -> Path:
    target = Path(os.path.abspath(path.expanduser()))
    if target.is_symlink() or not target.is_file():
        raise Heldout100ValidationError(
            f"{label} must be a regular non-symlink file"
        )
    return target.resolve(strict=True)


def _canonical_directory(path: Path, label: str) -> Path:
    target = Path(os.path.abspath(path.expanduser()))
    if target.is_symlink() or not target.is_dir():
        raise Heldout100ValidationError(
            f"{label} must be a real non-symlink directory"
        )
    resolved = target.resolve(strict=True)
    if resolved != target:
        raise Heldout100ValidationError(f"{label} must be a canonical realpath")
    return target


def _tree_disjoint(left: Path, right: Path) -> bool:
    return left != right and left not in right.parents and right not in left.parents


def validate_remote_layout(
    *,
    campaign: Path,
    output: Path,
    job: Path,
    log: Path,
    output_relative_path: str,
) -> dict[str, str]:
    """Validate the nested output overlay and isolated scratch/log roots."""

    campaign = _canonical_directory(campaign, "remote campaign root")
    output = _canonical_directory(output, "remote learner output root")
    job = _canonical_directory(job, "remote job scratch root")
    log = _canonical_directory(log, "remote scheduler log root")
    relative = Path(output_relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Heldout100ValidationError("remote output relative path is unsafe")
    expected_output = campaign.joinpath(relative).resolve(strict=True)
    if output != expected_output or output == campaign:
        raise Heldout100ValidationError(
            "remote output is not the exact canonical nested campaign target"
        )
    isolated = ((output, job), (output, log), (job, log))
    if any(not _tree_disjoint(left, right) for left, right in isolated):
        raise Heldout100ValidationError(
            "remote output, job, and log roots must be pairwise tree-disjoint"
        )
    if not _tree_disjoint(campaign, job) or not _tree_disjoint(campaign, log):
        raise Heldout100ValidationError(
            "remote job/log roots must not alias or overlap the campaign"
        )
    if job.parent != log.parent:
        raise Heldout100ValidationError(
            "remote job and log roots must be distinct siblings"
        )
    return {
        "campaign": str(campaign),
        "output": str(output),
        "job": str(job),
        "log": str(log),
        "output_relative_path": relative.as_posix(),
    }


def _strict_line(line: str, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise Heldout100ValidationError(
                    f"{label} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    def constant(value: str) -> None:
        raise Heldout100ValidationError(
            f"{label} contains non-finite constant {value!r}"
        )

    try:
        value = json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
    except Heldout100ValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise Heldout100ValidationError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise Heldout100ValidationError(f"{label} must be a JSON object")
    return value


def _jsonl(path: Path, label: str) -> tuple[Path, list[dict[str, Any]]]:
    path = _regular(path, label)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise Heldout100ValidationError(f"{label} has a torn final record")
    rows = [
        _strict_line(line, f"{label} line {index}")
        for index, line in enumerate(raw.decode("utf-8").splitlines(), 1)
        if line.strip()
    ]
    return path, rows


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        raise Heldout100ValidationError(f"{label} is invalid")
    return float(value)


def load_contract(
    path: Path,
    *,
    expected_sha256: str | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Load the exact dedicated contract without relaxing the two-step gate."""

    path = _regular(path, "heldout100 validation contract")
    if source_root is not None:
        canonical = Path(source_root).resolve(strict=True) / CONTRACT_RELATIVE_PATH
        if path != canonical:
            raise Heldout100ValidationError(
                "heldout100 validation contract path is non-canonical"
            )
    if expected_sha256 is not None and sha256_file(path) != _digest(
        expected_sha256, "validation contract SHA-256"
    ):
        raise Heldout100ValidationError("validation contract digest mismatch")
    document = load_strict_json(path)
    expected_fields = {
        "schema",
        "purpose",
        "validation_target",
        "allocation",
        "optimization",
        "heldout_evaluation",
        "quality_gate",
        "restart_validation",
        "cluster",
        "container",
        "wandb",
        "production_target",
    }
    if not isinstance(document, Mapping) or set(document) != expected_fields:
        raise Heldout100ValidationError("validation contract field set mismatch")
    exact = {
        "schema": CONTRACT_SCHEMA,
        "purpose": "live_h100_fixed256_step100_wandb_restart_gate",
        "validation_target": {
            "campaign_id_binding": "runtime_exact",
            "macro_round": 0,
        },
        "allocation": {
            "nodes": 1,
            "gpus_per_node": 8,
            "gpu_architecture": "H100-SM90",
            "backend": "nccl",
            "batch_size_per_rank": 1,
            "global_batch_size": 8,
        },
        "optimization": {
            "optimizer_steps": TRAIN_STEPS,
            "save_every_steps": 5,
            "intentional_stop_step": STOP_STEP,
            "learning_rate": 2.5e-5,
            "weight_decay": 0.01,
            "seed": 17,
            "partition_seed": 23,
        },
        "heldout_evaluation": {
            "interval_steps": 100,
            "required_optimizer_steps": HELDOUT_STEPS,
            "case_count": HELDOUT_CASES,
            "cohort_count": 8,
            "snapshot_relative_path": "heldout_snapshot.json",
            "snapshot_binding": "runtime_exact_file_and_semantic_sha256",
        },
        "quality_gate": {
            "minimum_relative_action_mse_reduction": (
                MINIMUM_RELATIVE_ACTION_MSE_REDUCTION
            ),
            "require_final_loss_below_initial": True,
        },
        "restart_validation": {
            "uninterrupted_comparator": "required",
            "wandb_lineage": "single_canonical_run",
            "first_invocation_resume": "never",
            "second_invocation_resume": "must",
            "stale_history_policy": "fail_closed_no_adoption",
        },
        "cluster": {
            "host": "oci-nrt",
            "account": "general_cs_infra__ngcsre",
            "partition": "batch_singlenode",
            "qos": "normal",
            "time_limit_minutes": 225,
        },
        "container": {
            "path": (
                "/lustre/fs1/portfolios/general/projects/"
                "general_cs_infra__ngcsre/users/ldu/roborender_h100_build/"
                "containers/roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh"
            ),
            "sha256": (
                "f3391b76a66182618f9e71b67e89b0af2d758ffe16afdb164760fcbb4860ebe6"
            ),
            "python": "/opt/viral-h100/bin/python",
        },
        "wandb": {
            "mode": "online",
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "project": "roborender-round-dagger-validation",
        },
        "production_target": {
            "campaign_contract_binding": "campaign_json_policy_config_sha256",
        },
    }
    if dict(document) != exact:
        raise Heldout100ValidationError("validation contract semantics mismatch")
    return dict(document)


def verify_heldout_snapshot(
    path: Path,
    *,
    campaign_root: Path,
    campaign_id: str,
    expected_file_sha256: str,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    root = _canonical_directory(campaign_root, "heldout100 campaign root")
    path = _regular(path, "fixed-256 held-out snapshot")
    if path != root / "heldout_snapshot.json":
        raise Heldout100ValidationError(
            "fixed-256 held-out snapshot is not at the campaign root"
        )
    file_sha256 = _digest(
        expected_file_sha256, "held-out snapshot file SHA-256"
    )
    semantic_sha256 = _digest(
        expected_snapshot_sha256, "held-out snapshot semantic SHA-256"
    )
    try:
        snapshot = load_heldout_snapshot(
            path, expected_file_sha256=file_sha256
        )
    except Exception as exc:
        raise Heldout100ValidationError(
            f"fixed-256 held-out snapshot validation failed: {exc}"
        ) from exc
    expected = {
        "campaign_id": campaign_id,
        "purpose": "eval",
        "cohort_count": 8,
        "transition_count": HELDOUT_CASES,
        "snapshot_sha256": semantic_sha256,
        "file_sha256": file_sha256,
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise Heldout100ValidationError("fixed-256 held-out identity mismatch")
    if len(snapshot.get("resolved_shards", ())) != 8:
        raise Heldout100ValidationError("fixed-256 snapshot must contain eight shards")
    if any(row.get("macro_round") != 0 for row in snapshot.get("cohorts", ())):
        raise Heldout100ValidationError("fixed-256 snapshot is not bound to round zero")
    return snapshot


def runtime_binding(
    *,
    contract: Path,
    contract_sha256: str,
    source_root: Path,
    campaign_root: Path,
    campaign_id: str,
    heldout_snapshot: Path,
    heldout_snapshot_file_sha256: str,
    heldout_snapshot_sha256: str,
    production_contract_sha256: str,
) -> dict[str, Any]:
    """Bind the reusable source profile to one exact fresh campaign."""

    load_contract(
        contract,
        expected_sha256=contract_sha256,
        source_root=source_root,
    )
    root = _canonical_directory(campaign_root, "heldout100 campaign root")
    snapshot = verify_heldout_snapshot(
        heldout_snapshot,
        campaign_root=root,
        campaign_id=campaign_id,
        expected_file_sha256=heldout_snapshot_file_sha256,
        expected_snapshot_sha256=heldout_snapshot_sha256,
    )
    campaign_path = _regular(root / "campaign.json", "campaign contract")
    campaign = load_strict_json(campaign_path)
    production_sha256 = _digest(
        production_contract_sha256, "production learner contract SHA-256"
    )
    if (
        not isinstance(campaign, Mapping)
        or campaign.get("campaign_id") != campaign_id
        or campaign.get("policy_config_sha256") != production_sha256
    ):
        raise Heldout100ValidationError(
            "campaign does not own the supplied production learner contract"
        )
    values = {
        "schema": BINDING_SCHEMA,
        "campaign_root": str(root),
        "campaign_id": campaign_id,
        "campaign_contract_file_sha256": sha256_file(campaign_path),
        "macro_round": 0,
        "heldout_snapshot_path": str(Path(heldout_snapshot).resolve(strict=True)),
        "heldout_snapshot_file_sha256": snapshot["file_sha256"],
        "heldout_snapshot_sha256": snapshot["snapshot_sha256"],
        "production_contract_sha256": production_sha256,
        "validation_contract_sha256": _digest(
            contract_sha256, "validation contract SHA-256"
        ),
    }
    payload = ("\n".join(f"{key}={values[key]}" for key in values) + "\n").encode()
    values["binding_sha256"] = hashlib.sha256(payload).hexdigest()
    return values


def training_rows(
    path: Path, *, campaign_id: str, macro_round: int
) -> tuple[Path, list[dict[str, Any]]]:
    if not campaign_id or macro_round != 0:
        raise Heldout100ValidationError(
            "heldout100 learner telemetry must be bound to campaign round 0"
        )
    path, rows = _jsonl(path, "learner telemetry")
    if len(rows) != TRAIN_STEPS:
        raise Heldout100ValidationError(
            "learner telemetry must contain exactly optimizer steps 1..100"
        )
    partition: tuple[int, ...] | None = None
    for step, row in enumerate(rows, 1):
        if set(row) != TRAIN_KEYS or row.get("schema") != TELEMETRY_SCHEMA:
            raise Heldout100ValidationError("learner telemetry fields differ")
        if (
            row.get("campaign_id") != campaign_id
            or row.get("macro_round") != macro_round
            or row.get("optimizer_step") != step
            or row.get("world_size") != WORLD_SIZE
            or row.get("gradient_reduction")
            != "ordered_allgather_rank_sum_v1"
        ):
            raise Heldout100ValidationError(
                "learner telemetry optimizer frontier or lineage mismatch"
            )
        _finite(row.get("loss"), "learner loss", nonnegative=True)
        counts = row.get("replay_transitions_per_rank")
        if (
            not isinstance(counts, list)
            or len(counts) != WORLD_SIZE
            or any(type(item) is not int or item < 1 for item in counts)
            or row.get("replay_transitions_rank") != counts[0]
            or row.get("replay_transitions_global") != sum(counts)
        ):
            raise Heldout100ValidationError(
                "learner replay transition partition is invalid"
            )
        current = tuple(counts)
        if partition is None:
            partition = current
        elif current != partition:
            raise Heldout100ValidationError(
                "learner replay partition changed during validation"
            )
    return path, rows


def heldout_rows(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    heldout_snapshot_file_sha256: str,
    heldout_snapshot_sha256: str,
) -> tuple[Path, list[dict[str, Any]]]:
    file_sha256 = _digest(
        heldout_snapshot_file_sha256, "held-out snapshot file SHA-256"
    )
    semantic_sha256 = _digest(
        heldout_snapshot_sha256, "held-out snapshot semantic SHA-256"
    )
    if not campaign_id or macro_round != 0:
        raise Heldout100ValidationError(
            "heldout100 quality telemetry must be bound to campaign round 0"
        )
    path, rows = _jsonl(path, "held-out quality telemetry")
    if len(rows) != 2 or [row.get("optimizer_step") for row in rows] != HELDOUT_STEPS:
        raise Heldout100ValidationError(
            "held-out quality telemetry must contain only steps 0 and 100"
        )
    for row in rows:
        if set(row) != HELDOUT_KEYS or row.get("schema") != HELDOUT_TELEMETRY_SCHEMA:
            raise Heldout100ValidationError("held-out telemetry fields differ")
        if (
            row.get("campaign_id") != campaign_id
            or row.get("macro_round") != macro_round
            or row.get("heldout_snapshot_file_sha256")
            != file_sha256
            or row.get("heldout_snapshot_sha256") != semantic_sha256
            or row.get("heldout_transitions_rank") != 32
            or row.get("heldout_transitions_per_rank") != [32] * WORLD_SIZE
            or row.get("heldout_transitions_global") != HELDOUT_CASES
            or row.get("world_size") != WORLD_SIZE
            or row.get("learned_action_indices") != LEARNED_ACTION_INDICES
        ):
            raise Heldout100ValidationError("held-out telemetry identity mismatch")
        mse = _finite(row.get("action_mse"), "held-out action MSE", nonnegative=True)
        _finite(row.get("action_mae"), "held-out action MAE", nonnegative=True)
        rmse = _finite(
            row.get("action_rmse"), "held-out action RMSE", nonnegative=True
        )
        cosine = _finite(row.get("action_cosine"), "held-out action cosine")
        if not math.isclose(rmse * rmse, mse, rel_tol=1e-10, abs_tol=1e-12):
            raise Heldout100ValidationError("held-out RMSE is inconsistent with MSE")
        if not -1.0 <= cosine <= 1.0:
            raise Heldout100ValidationError("held-out cosine is outside [-1, 1]")
        per_action = row.get("per_action")
        if not isinstance(per_action, Mapping) or set(per_action) != {
            str(index) for index in LEARNED_ACTION_INDICES
        }:
            raise Heldout100ValidationError("held-out per-action coverage differs")
        for action, metrics in per_action.items():
            if not isinstance(metrics, Mapping) or set(metrics) != {
                "mse",
                "mae",
                "rmse",
                "cosine",
            }:
                raise Heldout100ValidationError(
                    f"held-out action {action} metric set differs"
                )
            action_mse = _finite(
                metrics["mse"], f"held-out action {action} MSE", nonnegative=True
            )
            _finite(
                metrics["mae"], f"held-out action {action} MAE", nonnegative=True
            )
            action_rmse = _finite(
                metrics["rmse"],
                f"held-out action {action} RMSE",
                nonnegative=True,
            )
            action_cosine = _finite(
                metrics["cosine"], f"held-out action {action} cosine"
            )
            if not math.isclose(
                action_rmse * action_rmse,
                action_mse,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ) or not -1.0 <= action_cosine <= 1.0:
                raise Heldout100ValidationError(
                    f"held-out action {action} metrics are inconsistent"
                )
    return path, rows


def quality_improvement(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Require a measurable fixed-256 MSE improvement at optimizer step 100."""

    if len(rows) != 2 or [row.get("optimizer_step") for row in rows] != HELDOUT_STEPS:
        raise Heldout100ValidationError(
            "quality improvement requires exact held-out steps 0 and 100"
        )
    initial_mse = _finite(
        rows[0].get("action_mse"), "step-0 held-out action MSE", nonnegative=True
    )
    final_mse = _finite(
        rows[1].get("action_mse"), "step-100 held-out action MSE", nonnegative=True
    )
    if initial_mse <= 0.0:
        raise Heldout100ValidationError(
            "step-0 held-out action MSE must be positive for an improvement gate"
        )
    absolute_reduction = initial_mse - final_mse
    relative_reduction = absolute_reduction / initial_mse
    if relative_reduction < MINIMUM_RELATIVE_ACTION_MSE_REDUCTION:
        raise Heldout100ValidationError(
            "step-100 held-out action MSE did not improve by at least 0.1%"
        )
    return {
        "initial_action_mse": initial_mse,
        "final_action_mse": final_mse,
        "absolute_action_mse_reduction": absolute_reduction,
        "relative_action_mse_reduction": relative_reduction,
        "action_mae_delta": float(rows[1]["action_mae"])
        - float(rows[0]["action_mae"]),
        "action_rmse_delta": float(rows[1]["action_rmse"])
        - float(rows[0]["action_rmse"]),
        "action_cosine_delta": float(rows[1]["action_cosine"])
        - float(rows[0]["action_cosine"]),
    }


def quality_payload(row: Mapping[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {
        key: row[source] for key, source in QUALITY_GLOBAL_KEYS.items()
    }
    per_action = row["per_action"]
    for index in LEARNED_ACTION_INDICES:
        metrics = per_action[str(index)]
        for metric in ("mse", "mae", "rmse", "cosine"):
            result[f"quality/action_{index}/{metric}"] = metrics[metric]
    return result


def _result(
    path: Path,
    *,
    semantic: Mapping[str, Any],
    expected_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, "learner result")
    document = load_strict_json(path)
    expected = {
        "schema": RESULT_SCHEMA,
        "campaign_id": semantic["campaign_id"],
        "macro_round": semantic["macro_round"],
        "code_sha256": semantic["learner_code_sha256"],
        "learner_config_sha256": semantic["learner_config_sha256"],
        "model_lineage_sha256": semantic["model_lineage_sha256"],
        "replay_snapshot_sha256": semantic["replay_snapshot_sha256"],
        "optimizer_steps": TRAIN_STEPS,
        "checkpoint_sha256": expected_checkpoint_sha256,
    }
    if not isinstance(document, Mapping) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise Heldout100ValidationError("learner result lineage mismatch")
    for key in ("initial_loss", "final_loss"):
        _finite(document.get(key), f"learner result {key}", nonnegative=True)
    return path, dict(document)


def _config_value(config: object, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    try:
        return config[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _remote_training(remote: object) -> list[dict[str, Any]]:
    try:
        raw = list(remote.scan_history(keys=sorted(TRAIN_KEYS), page_size=1000))
    except Exception as exc:
        raise Heldout100ValidationError(
            f"remote W&B training-history query failed ({type(exc).__name__})"
        ) from None
    return [
        {key: row.get(key) for key in TRAIN_KEYS}
        for row in raw
        if isinstance(row, Mapping) and row.get("schema") == TELEMETRY_SCHEMA
    ]


def _remote_quality(remote: object) -> list[dict[str, Any]]:
    quality_keys = sorted(quality_payload({
        **{source: 0 for source in QUALITY_GLOBAL_KEYS.values()},
        "per_action": {
            str(index): {metric: 0 for metric in ("mse", "mae", "rmse", "cosine")}
            for index in LEARNED_ACTION_INDICES
        },
    }))
    try:
        raw = list(
            remote.scan_history(
                keys=["optimizer_step", *quality_keys], page_size=1000
            )
        )
    except Exception as exc:
        raise Heldout100ValidationError(
            f"remote W&B quality-history query failed ({type(exc).__name__})"
        ) from None
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        if row.get("quality/heldout_action_mse") is None:
            continue
        result = {"optimizer_step": row.get("optimizer_step")}
        result.update({key: row.get(key) for key in quality_keys})
        rows.append(result)
    return rows


def _same_numeric(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def readback(
    *,
    run_id: str,
    uninterrupted_telemetry: Path,
    resumed_telemetry: Path,
    uninterrupted_heldout: Path,
    resumed_heldout: Path,
    uninterrupted_result: Path,
    resumed_result: Path,
    uninterrupted_checkpoint: Path,
    stopped_checkpoint: Path,
    resumed_checkpoint: Path,
    resume_evidence: Path,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    model_lineage_sha256: str,
    replay_snapshot_file_sha256: str,
    replay_snapshot_sha256: str,
    parent_runtime_policy_sha256: str,
    learner_identity_sha256: str,
    runtime_container_sha256: str,
    validation_contract_sha256: str,
    validation_binding_sha256: str,
    production_contract_sha256: str,
    heldout_snapshot_file_sha256: str,
    heldout_snapshot_sha256: str,
    project: str,
    expected_email: str,
    entity: str,
    authorization_scope: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 5.0,
    wandb_module: object | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise Heldout100ValidationError("W&B readback timing must be positive")
    binding_sha256 = _digest(
        validation_binding_sha256, "heldout100 runtime binding SHA-256"
    )
    production_sha256 = _digest(
        production_contract_sha256, "production learner contract SHA-256"
    )
    heldout_file_sha256 = _digest(
        heldout_snapshot_file_sha256, "held-out snapshot file SHA-256"
    )
    heldout_semantic_sha256 = _digest(
        heldout_snapshot_sha256, "held-out snapshot semantic SHA-256"
    )
    if not campaign_id or macro_round != 0:
        raise Heldout100ValidationError(
            "heldout100 readback must be bound to campaign round 0"
        )
    canonical_run_id = learner_wandb_run_id(
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_identity_sha256=learner_identity_sha256,
    )
    expected_run_id = f"{canonical_run_id}-b{binding_sha256[:32]}"
    if run_id != expected_run_id:
        raise Heldout100ValidationError(
            "W&B run ID is not bound to the fresh campaign validation binding"
        )
    try:
        semantic = _semantic_identity(
            run_id=canonical_run_id,
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_code_sha256=learner_code_sha256,
            learner_config_sha256=learner_config_sha256,
            model_lineage_sha256=model_lineage_sha256,
            replay_snapshot_file_sha256=replay_snapshot_file_sha256,
            replay_snapshot_sha256=replay_snapshot_sha256,
            parent_runtime_policy_sha256=parent_runtime_policy_sha256,
            learner_identity_sha256=learner_identity_sha256,
            runtime_container_sha256=runtime_container_sha256,
            validation_contract_sha256=validation_contract_sha256,
        )
    except Exception as exc:
        raise Heldout100ValidationError(f"learner semantic identity failed: {exc}") from exc

    full_train_path, full_train = training_rows(
        uninterrupted_telemetry, campaign_id=campaign_id, macro_round=macro_round
    )
    resumed_train_path, resumed_train = training_rows(
        resumed_telemetry, campaign_id=campaign_id, macro_round=macro_round
    )
    if full_train != resumed_train:
        raise Heldout100ValidationError(
            "uninterrupted and resumed learner histories differ"
        )
    full_quality_path, full_quality = heldout_rows(
        uninterrupted_heldout,
        campaign_id=campaign_id,
        macro_round=macro_round,
        heldout_snapshot_file_sha256=heldout_file_sha256,
        heldout_snapshot_sha256=heldout_semantic_sha256,
    )
    resumed_quality_path, resumed_quality = heldout_rows(
        resumed_heldout,
        campaign_id=campaign_id,
        macro_round=macro_round,
        heldout_snapshot_file_sha256=heldout_file_sha256,
        heldout_snapshot_sha256=heldout_semantic_sha256,
    )
    if full_quality != resumed_quality:
        raise Heldout100ValidationError(
            "uninterrupted and resumed held-out quality histories differ"
        )
    quality_delta = quality_improvement(resumed_quality)
    try:
        full_checkpoint_path, full_checkpoint, full_checkpoint_sha = _load_checkpoint(
            uninterrupted_checkpoint,
            label="uninterrupted checkpoint",
            completed_steps=TRAIN_STEPS,
            semantic=semantic,
        )
        stop_checkpoint_path, stop_checkpoint, stop_checkpoint_sha = _load_checkpoint(
            stopped_checkpoint,
            label="intentional-stop checkpoint",
            completed_steps=STOP_STEP,
            semantic=semantic,
        )
        resumed_checkpoint_path, resumed_checkpoint_doc, resumed_checkpoint_sha = (
            _load_checkpoint(
                resumed_checkpoint,
                label="resumed checkpoint",
                completed_steps=TRAIN_STEPS,
                semantic=semantic,
            )
        )
    except Exception as exc:
        raise Heldout100ValidationError(f"restart checkpoint validation failed: {exc}") from exc
    if stop_checkpoint_path.stat().st_mode & 0o222:
        raise Heldout100ValidationError(
            "intentional-stop checkpoint must be read-only"
        )
    full_result_path, full_result = _result(
        uninterrupted_result,
        semantic=semantic,
        expected_checkpoint_sha256=full_checkpoint_sha,
    )
    resumed_result_path, resumed_result_doc = _result(
        resumed_result,
        semantic=semantic,
        expected_checkpoint_sha256=resumed_checkpoint_sha,
    )
    if (
        full_result["initial_loss"] != resumed_result_doc["initial_loss"]
        or full_result["final_loss"] != resumed_result_doc["final_loss"]
        or full_checkpoint["initial_loss"] != full_result["initial_loss"]
        or stop_checkpoint["initial_loss"] != resumed_result_doc["initial_loss"]
        or resumed_checkpoint_doc["initial_loss"]
        != resumed_result_doc["initial_loss"]
    ):
        raise Heldout100ValidationError("result/checkpoint loss lineage differs")
    initial_loss = _finite(
        full_result["initial_loss"], "initial deterministic-probe loss", nonnegative=True
    )
    final_loss = _finite(
        full_result["final_loss"], "final deterministic-probe loss", nonnegative=True
    )
    if not final_loss < initial_loss:
        raise Heldout100ValidationError(
            "final deterministic-probe loss must be strictly below initial loss"
        )
    loss_delta = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "absolute_loss_reduction": initial_loss - final_loss,
        "relative_loss_reduction": (initial_loss - final_loss) / initial_loss
        if initial_loss > 0.0
        else 0.0,
    }
    resume_path = _regular(resume_evidence, "strict resume evidence")
    if resume_path.stat().st_mode & 0o222:
        raise Heldout100ValidationError("strict resume evidence must be read-only")
    resume_doc = load_strict_json(resume_path)
    expected_resume = {
        "schema": RESUME_SCHEMA,
        "status": "PASS",
        "run_id": run_id,
        "stop_step": STOP_STEP,
        "target_step": TRAIN_STEPS,
        "first_invocation_resume_mode": "never",
        "second_invocation_resume_mode": "must",
        "learner_identity_sha256": learner_identity_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_code_sha256": learner_code_sha256,
        "replay_snapshot_file_sha256": replay_snapshot_file_sha256,
        "replay_snapshot_sha256": replay_snapshot_sha256,
        "heldout_snapshot_file_sha256": heldout_file_sha256,
        "heldout_snapshot_sha256": heldout_semantic_sha256,
        "validation_binding_sha256": binding_sha256,
        "production_contract_sha256": production_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": validation_contract_sha256,
        "stopped_checkpoint_sha256": stop_checkpoint_sha,
        "resumed_checkpoint_sha256": resumed_checkpoint_sha,
        "canonical_wandb_lineage": True,
    }
    if not isinstance(resume_doc, Mapping) or dict(resume_doc) != expected_resume:
        raise Heldout100ValidationError("strict resume evidence differs")

    wandb_module = _load_wandb() if wandb_module is None else wandb_module
    identity = query_identity(
        wandb_module,
        expected_email=expected_email,
        requested_entity=entity,
        authorization_scope=authorization_scope,
    )
    try:
        api = wandb_module.Api(timeout=30)
    except Exception as exc:
        raise Heldout100ValidationError(
            f"remote W&B API initialization failed ({type(exc).__name__})"
        ) from None
    deadline = time.monotonic() + timeout_s
    last_error = ""
    remote_report: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            remote = api.run(f"{entity}/{project}/{run_id}")
            if getattr(remote, "id", run_id) != run_id:
                raise Heldout100ValidationError("remote W&B run ID mismatch")
            if getattr(remote, "state", "finished") != "finished":
                raise Heldout100ValidationError("remote W&B run is not finished")
            config = getattr(remote, "config", None)
            expected_config = {
                **semantic,
                "heldout_snapshot_file_sha256": heldout_file_sha256,
                "heldout_snapshot_sha256": heldout_semantic_sha256,
                "heldout_eval_interval": 100,
                "heldout_transitions": HELDOUT_CASES,
            }
            if any(
                _config_value(config, key) != value
                for key, value in expected_config.items()
            ):
                raise Heldout100ValidationError("remote W&B config mismatch")
            remote_train = _remote_training(remote)
            if remote_train != resumed_train:
                raise Heldout100ValidationError(
                    "remote W&B training history differs from local history"
                )
            remote_quality = _remote_quality(remote)
            expected_quality = [
                {"optimizer_step": row["optimizer_step"], **quality_payload(row)}
                for row in resumed_quality
            ]
            if len(remote_quality) != len(expected_quality):
                raise Heldout100ValidationError(
                    "remote W&B quality history does not contain exactly steps 0 and 100"
                )
            for observed, expected in zip(remote_quality, expected_quality, strict=True):
                if set(observed) != set(expected) or any(
                    not _same_numeric(observed[key], expected[key]) for key in expected
                ):
                    raise Heldout100ValidationError(
                        "remote W&B quality metrics differ from local held-out telemetry"
                    )
            remote_report = {
                "state": "finished",
                "training_history_steps": list(range(1, TRAIN_STEPS + 1)),
                "quality_history_steps": HELDOUT_STEPS,
                "quality_key_count": len(quality_payload(resumed_quality[0])),
                "config_verified": True,
                "quality_improvement_verified": True,
            }
            break
        except Heldout100ValidationError as exc:
            last_error = str(exc)
            remote_report = None
            time.sleep(poll_interval_s)
    if remote_report is None:
        raise Heldout100ValidationError(
            f"remote W&B heldout100 readback timed out: {last_error or 'no evidence'}"
        )
    return {
        "schema": READBACK_SCHEMA,
        "status": "PASS",
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": project,
        "wandb_run_id": run_id,
        **semantic,
        # The validation execution contract is intentionally also the learner
        # contract for this smoke.  Retain both names so generic checkpoint
        # identity code and validation-specific attestation code bind the same
        # immutable bytes without an implicit alias.
        "validation_contract_sha256": semantic["learner_contract_sha256"],
        "validation_binding_sha256": binding_sha256,
        "production_contract_sha256": production_sha256,
        "heldout_snapshot_file_sha256": heldout_file_sha256,
        "heldout_snapshot_sha256": heldout_semantic_sha256,
        "heldout_case_count": HELDOUT_CASES,
        "heldout_optimizer_steps": HELDOUT_STEPS,
        "quality_improvement": quality_delta,
        "loss_improvement": loss_delta,
        "remote": remote_report,
        "uninterrupted_telemetry_sha256": sha256_file(full_train_path),
        "resumed_telemetry_sha256": sha256_file(resumed_train_path),
        "uninterrupted_heldout_sha256": sha256_file(full_quality_path),
        "resumed_heldout_sha256": sha256_file(resumed_quality_path),
        "uninterrupted_result_sha256": sha256_file(full_result_path),
        "resumed_result_sha256": sha256_file(resumed_result_path),
        "uninterrupted_checkpoint_sha256": full_checkpoint_sha,
        "stopped_checkpoint_sha256": stop_checkpoint_sha,
        "resumed_checkpoint_sha256": resumed_checkpoint_sha,
        "strict_resume_evidence_sha256": sha256_file(resume_path),
        "histories_identical": True,
        "quality_histories_identical": True,
        "quality_improvement_verified": True,
        "loss_improvement_verified": True,
        "strict_resume_verified": True,
        "single_canonical_wandb_run": True,
        "stale_wandb_history_policy": "fail_closed_no_adoption",
    }


def build_attestation(
    *,
    contract: Path,
    contract_sha256: str,
    source_root: Path,
    campaign_root: Path,
    heldout_snapshot: Path,
    heldout_snapshot_file_sha256: str,
    heldout_snapshot_sha256: str,
    production_contract_sha256: str,
    validation_binding_sha256: str,
    campaign_id: str,
    readback_path: Path,
    restart_report: Path,
    output: Path,
) -> dict[str, Any]:
    binding = runtime_binding(
        contract=contract,
        contract_sha256=contract_sha256,
        source_root=source_root,
        campaign_root=campaign_root,
        campaign_id=campaign_id,
        heldout_snapshot=heldout_snapshot,
        heldout_snapshot_file_sha256=heldout_snapshot_file_sha256,
        heldout_snapshot_sha256=heldout_snapshot_sha256,
        production_contract_sha256=production_contract_sha256,
    )
    if binding["binding_sha256"] != _digest(
        validation_binding_sha256, "heldout100 runtime binding SHA-256"
    ):
        raise Heldout100ValidationError("heldout100 runtime binding digest mismatch")
    contract_doc = load_contract(
        contract, expected_sha256=contract_sha256, source_root=source_root
    )
    readback_path = _regular(readback_path, "heldout100 W&B readback")
    readback_doc = load_strict_json(readback_path)
    if (
        not isinstance(readback_doc, Mapping)
        or readback_doc.get("schema") != READBACK_SCHEMA
        or readback_doc.get("status") != "PASS"
        or readback_doc.get("campaign_id") != campaign_id
        or readback_doc.get("validation_contract_sha256") != contract_sha256
        or readback_doc.get("learner_contract_sha256") != contract_sha256
        or readback_doc.get("validation_binding_sha256")
        != binding["binding_sha256"]
        or readback_doc.get("production_contract_sha256")
        != binding["production_contract_sha256"]
        or readback_doc.get("heldout_snapshot_file_sha256")
        != binding["heldout_snapshot_file_sha256"]
        or readback_doc.get("heldout_snapshot_sha256")
        != binding["heldout_snapshot_sha256"]
        or readback_doc.get("heldout_case_count") != HELDOUT_CASES
        or readback_doc.get("heldout_optimizer_steps") != HELDOUT_STEPS
        or readback_doc.get("account_email") != "ldu@nvidia.com"
        or readback_doc.get("entity") != "nvidia"
        or readback_doc.get("authorization_scope") != "team"
        or readback_doc.get("team_member") is not True
        or readback_doc.get("entity_is_team") is not True
        or readback_doc.get("personal") is not False
        or readback_doc.get("project")
        != "roborender-round-dagger-validation"
        or readback_doc.get("remote", {}).get("training_history_steps")
        != list(range(1, TRAIN_STEPS + 1))
        or readback_doc.get("remote", {}).get("quality_history_steps")
        != HELDOUT_STEPS
        or readback_doc.get("remote", {}).get("quality_key_count") != 49
        or readback_doc.get("remote", {}).get("quality_improvement_verified")
        is not True
        or readback_doc.get("quality_improvement_verified") is not True
        or readback_doc.get("loss_improvement_verified") is not True
        or readback_doc.get("strict_resume_verified") is not True
        or readback_doc.get("single_canonical_wandb_run") is not True
        or readback_doc.get("stale_wandb_history_policy")
        != "fail_closed_no_adoption"
    ):
        raise Heldout100ValidationError("heldout100 W&B readback did not pass")
    for key in (
        "learner_code_sha256",
        "learner_config_sha256",
        "learner_identity_sha256",
        "runtime_container_sha256",
        "replay_snapshot_file_sha256",
        "replay_snapshot_sha256",
        "model_lineage_sha256",
        "parent_runtime_policy_sha256",
    ):
        _digest(readback_doc.get(key), f"readback {key}")
    restart_path = _regular(restart_report, "restart-equivalence report")
    restart_doc = load_strict_json(restart_path)
    if (
        restart_doc.get("semantic_equivalent") is not True
        or restart_doc.get("results_equivalent") is not True
        or restart_doc.get("expected_completed_steps") != TRAIN_STEPS
    ):
        raise Heldout100ValidationError("100-step restart equivalence did not pass")
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS",
        "purpose": contract_doc["purpose"],
        "campaign_id": campaign_id,
        "allocation": contract_doc["allocation"],
        "optimizer_steps": TRAIN_STEPS,
        "intentional_stop_step": STOP_STEP,
        "heldout_optimizer_steps": HELDOUT_STEPS,
        "heldout_case_count": HELDOUT_CASES,
        "heldout_snapshot_path": str(heldout_snapshot),
        "heldout_snapshot_file_sha256": binding["heldout_snapshot_file_sha256"],
        "heldout_snapshot_sha256": binding["heldout_snapshot_sha256"],
        "validation_contract_sha256": contract_sha256,
        "validation_binding_sha256": binding["binding_sha256"],
        "production_target_contract_sha256": binding[
            "production_contract_sha256"
        ],
        "learner_code_sha256": readback_doc["learner_code_sha256"],
        "learner_config_sha256": readback_doc["learner_config_sha256"],
        "learner_identity_sha256": readback_doc["learner_identity_sha256"],
        "runtime_container_sha256": readback_doc["runtime_container_sha256"],
        "training_replay_snapshot_file_sha256": readback_doc[
            "replay_snapshot_file_sha256"
        ],
        "training_replay_snapshot_sha256": readback_doc[
            "replay_snapshot_sha256"
        ],
        "parent_checkpoint_sha256": readback_doc["model_lineage_sha256"],
        "parent_runtime_policy_sha256": readback_doc[
            "parent_runtime_policy_sha256"
        ],
        "wandb_run_id": readback_doc["wandb_run_id"],
        "wandb_quality_key_count": readback_doc["remote"]["quality_key_count"],
        "quality_improvement": readback_doc["quality_improvement"],
        "loss_improvement": readback_doc["loss_improvement"],
        "wandb_readback_sha256": sha256_file(readback_path),
        "restart_report_sha256": sha256_file(restart_path),
        "strict_resume_verified": True,
        "uninterrupted_equivalence_verified": True,
        "stale_wandb_history_policy": "fail_closed_no_adoption",
        "quality_improvement_verified": True,
        "loss_improvement_verified": True,
    }


def _add_readback_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--uninterrupted-telemetry", required=True, type=Path)
    parser.add_argument("--resumed-telemetry", required=True, type=Path)
    parser.add_argument("--uninterrupted-heldout", required=True, type=Path)
    parser.add_argument("--resumed-heldout", required=True, type=Path)
    parser.add_argument("--uninterrupted-result", required=True, type=Path)
    parser.add_argument("--resumed-result", required=True, type=Path)
    parser.add_argument("--uninterrupted-checkpoint", required=True, type=Path)
    parser.add_argument("--stopped-checkpoint", required=True, type=Path)
    parser.add_argument("--resumed-checkpoint", required=True, type=Path)
    parser.add_argument("--resume-evidence", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--learner-code-sha256", required=True)
    parser.add_argument("--learner-config-sha256", required=True)
    parser.add_argument("--model-lineage-sha256", required=True)
    parser.add_argument("--replay-snapshot-file-sha256", required=True)
    parser.add_argument("--replay-snapshot-sha256", required=True)
    parser.add_argument("--parent-runtime-policy-sha256", required=True)
    parser.add_argument("--learner-identity-sha256", required=True)
    parser.add_argument("--runtime-container-sha256", required=True)
    parser.add_argument("--validation-contract-sha256", required=True)
    parser.add_argument("--validation-binding-sha256", required=True)
    parser.add_argument("--production-contract-sha256", required=True)
    parser.add_argument("--heldout-snapshot-file-sha256", required=True)
    parser.add_argument("--heldout-snapshot-sha256", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--authorization-scope", required=True, choices=("team",))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract")
    contract.add_argument("--path", required=True, type=Path)
    contract.add_argument("--sha256")
    contract.add_argument("--source-root", type=Path)
    binding = commands.add_parser("binding")
    binding.add_argument("--contract", required=True, type=Path)
    binding.add_argument("--contract-sha256", required=True)
    binding.add_argument("--source-root", required=True, type=Path)
    binding.add_argument("--campaign-root", required=True, type=Path)
    binding.add_argument("--campaign-id", required=True)
    binding.add_argument("--heldout-snapshot", required=True, type=Path)
    binding.add_argument("--heldout-snapshot-file-sha256", required=True)
    binding.add_argument("--heldout-snapshot-sha256", required=True)
    binding.add_argument("--production-contract-sha256", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--path", required=True, type=Path)
    snapshot.add_argument("--campaign-root", required=True, type=Path)
    snapshot.add_argument("--campaign-id", required=True)
    snapshot.add_argument("--file-sha256", required=True)
    snapshot.add_argument("--semantic-sha256", required=True)
    local = commands.add_parser("local")
    local.add_argument("--telemetry", required=True, type=Path)
    local.add_argument("--heldout", required=True, type=Path)
    local.add_argument("--campaign-id", required=True)
    local.add_argument("--macro-round", required=True, type=int)
    local.add_argument("--heldout-snapshot-file-sha256", required=True)
    local.add_argument("--heldout-snapshot-sha256", required=True)
    layout = commands.add_parser("remote-layout")
    layout.add_argument("--campaign", required=True, type=Path)
    layout.add_argument("--output", required=True, type=Path)
    layout.add_argument("--job", required=True, type=Path)
    layout.add_argument("--log", required=True, type=Path)
    layout.add_argument("--output-relative-path", required=True)
    readback_parser = commands.add_parser("readback")
    _add_readback_arguments(readback_parser)
    attest = commands.add_parser("attest")
    attest.add_argument("--contract", required=True, type=Path)
    attest.add_argument("--contract-sha256", required=True)
    attest.add_argument("--source-root", required=True, type=Path)
    attest.add_argument("--campaign-root", required=True, type=Path)
    attest.add_argument("--heldout-snapshot", required=True, type=Path)
    attest.add_argument("--heldout-snapshot-file-sha256", required=True)
    attest.add_argument("--heldout-snapshot-sha256", required=True)
    attest.add_argument("--production-contract-sha256", required=True)
    attest.add_argument("--validation-binding-sha256", required=True)
    attest.add_argument("--campaign-id", required=True)
    attest.add_argument("--readback", required=True, type=Path)
    attest.add_argument("--restart-report", required=True, type=Path)
    attest.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "contract":
        report = load_contract(
            arguments.path,
            expected_sha256=arguments.sha256,
            source_root=arguments.source_root,
        )
    elif arguments.command == "binding":
        report = runtime_binding(
            contract=arguments.contract,
            contract_sha256=arguments.contract_sha256,
            source_root=arguments.source_root,
            campaign_root=arguments.campaign_root,
            campaign_id=arguments.campaign_id,
            heldout_snapshot=arguments.heldout_snapshot,
            heldout_snapshot_file_sha256=(
                arguments.heldout_snapshot_file_sha256
            ),
            heldout_snapshot_sha256=arguments.heldout_snapshot_sha256,
            production_contract_sha256=arguments.production_contract_sha256,
        )
    elif arguments.command == "snapshot":
        snapshot = verify_heldout_snapshot(
            arguments.path,
            campaign_root=arguments.campaign_root,
            campaign_id=arguments.campaign_id,
            expected_file_sha256=arguments.file_sha256,
            expected_snapshot_sha256=arguments.semantic_sha256,
        )
        report = {
            "status": "PASS",
            "file_sha256": snapshot["file_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "transition_count": snapshot["transition_count"],
            "cohort_count": snapshot["cohort_count"],
        }
    elif arguments.command == "local":
        _, train = training_rows(
            arguments.telemetry,
            campaign_id=arguments.campaign_id,
            macro_round=arguments.macro_round,
        )
        _, quality = heldout_rows(
            arguments.heldout,
            campaign_id=arguments.campaign_id,
            macro_round=arguments.macro_round,
            heldout_snapshot_file_sha256=(
                arguments.heldout_snapshot_file_sha256
            ),
            heldout_snapshot_sha256=arguments.heldout_snapshot_sha256,
        )
        improvement = quality_improvement(quality)
        report = {
            "status": "PASS",
            "training_steps": [train[0]["optimizer_step"], train[-1]["optimizer_step"]],
            "heldout_steps": [row["optimizer_step"] for row in quality],
            "quality_key_count": len(quality_payload(quality[0])),
            "quality_improvement": improvement,
        }
    elif arguments.command == "remote-layout":
        report = validate_remote_layout(
            campaign=arguments.campaign,
            output=arguments.output,
            job=arguments.job,
            log=arguments.log,
            output_relative_path=arguments.output_relative_path,
        )
    elif arguments.command == "readback":
        kwargs = vars(arguments).copy()
        kwargs.pop("command")
        output = kwargs.pop("output")
        report = readback(**kwargs)
        if output.exists() or output.is_symlink():
            raise Heldout100ValidationError("readback output already exists")
        publish_immutable_json(output, report)
        os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    else:
        report = build_attestation(
            contract=arguments.contract,
            contract_sha256=arguments.contract_sha256,
            source_root=arguments.source_root,
            campaign_root=arguments.campaign_root,
            heldout_snapshot=arguments.heldout_snapshot,
            heldout_snapshot_file_sha256=(
                arguments.heldout_snapshot_file_sha256
            ),
            heldout_snapshot_sha256=arguments.heldout_snapshot_sha256,
            production_contract_sha256=arguments.production_contract_sha256,
            validation_binding_sha256=arguments.validation_binding_sha256,
            campaign_id=arguments.campaign_id,
            readback_path=arguments.readback,
            restart_report=arguments.restart_report,
            output=arguments.output,
        )
        if arguments.output.exists() or arguments.output.is_symlink():
            raise Heldout100ValidationError("attestation output already exists")
        publish_immutable_json(arguments.output, report)
        os.chmod(arguments.output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Heldout100ValidationError as exc:
        print(f"heldout100 validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
