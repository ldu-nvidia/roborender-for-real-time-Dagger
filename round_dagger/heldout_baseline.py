"""Standalone fixed-256 action-quality baseline for the active policy.

The production learner already evaluates its immutable held-out replay at
optimizer step zero and every 100 updates.  Collection used to happen before
that learner existed, however, leaving the dashboard blank during the longest
upstream phase.  This module runs the *same* ``evaluate_heldout_actions`` path
against policy-v0 without opening, advancing, or otherwise mutating the DAgger
coordinator.

The result is an immutable, checkpoint- and snapshot-bound artifact.  W&B is a
separate delivery frontier: an interrupted upload can safely replay the exact
same result at optimizer step zero under a deterministic run ID.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from round_dagger import artifacts, state


BASELINE_RESULT_SCHEMA = "fast_roborender.round_dagger.heldout_baseline.v1"
BASELINE_RUN_ID_SCHEMA = "fast_roborender.round_dagger.heldout_baseline_run.v1"
BASELINE_WANDB_RECEIPT_SCHEMA = (
    "fast_roborender.round_dagger.heldout_baseline_wandb_receipt.v1"
)
CANONICAL_RESULT_RELATIVE = Path("quality/policy_v0_fixed256.json")
CANONICAL_WANDB_RECEIPT_RELATIVE = Path(
    "quality/policy_v0_fixed256.wandb.json"
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class HeldoutBaselineError(RuntimeError):
    """The active policy, held-out replay, or baseline evidence is invalid."""


# Lazy compatibility shims keep the module mockable/testable while avoiding a
# Torch import in the production control interpreter.
def validate_policy_bundle_identity(**kwargs: Any) -> Any:
    from round_dagger.policy_bundle import validate_policy_bundle_identity as function

    return function(**kwargs)


def load_heldout_snapshot(*args: Any, **kwargs: Any) -> Any:
    from round_dagger.replay import load_heldout_snapshot as function

    return function(*args, **kwargs)


def load_checkpoint_strict(*args: Any, **kwargs: Any) -> Any:
    from round_dagger.gr00t_policy import load_checkpoint_strict as function

    return function(*args, **kwargs)


def load_heldout_partition(*args: Any, **kwargs: Any) -> Any:
    from round_dagger.learner import load_heldout_partition as function

    return function(*args, **kwargs)


def evaluate_heldout_actions(*args: Any, **kwargs: Any) -> Any:
    from round_dagger.learner import evaluate_heldout_actions as function

    return function(*args, **kwargs)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HeldoutBaselineError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not value.is_absolute():
        raise HeldoutBaselineError(f"{label} must be absolute")
    return value


def _symbol(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise HeldoutBaselineError("factory must use module.path:symbol syntax")
    module_name, symbol_name = specification.rsplit(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), symbol_name)
    except (ImportError, AttributeError) as exc:
        raise HeldoutBaselineError(
            f"cannot import baseline factory {specification}: {type(exc).__name__}"
        ) from None
    if not callable(value):
        raise HeldoutBaselineError("baseline factory symbol is not callable")
    return value


def baseline_wandb_run_id(
    *,
    campaign_id: str,
    campaign_sha256: str,
    checkpoint_sha256: str,
    heldout_snapshot_sha256: str,
) -> str:
    payload = {
        "schema": BASELINE_RUN_ID_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_sha256": _sha(campaign_sha256, "campaign SHA-256"),
        "checkpoint_sha256": _sha(checkpoint_sha256, "checkpoint SHA-256"),
        "heldout_snapshot_sha256": _sha(
            heldout_snapshot_sha256, "held-out snapshot SHA-256"
        ),
    }
    return "rrd-baseline-" + artifacts.canonical_json_sha256(payload)


def _active_policy_v0(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = state.load_campaign(campaign_root)
    coordinator = state.load_state(campaign_root)
    if coordinator.get("phase") not in {"READY", "COLLECTING"}:
        raise HeldoutBaselineError(
            "policy-v0 baseline must run before collection completes"
        )
    if coordinator.get("next_macro_round") != 0 or coordinator.get(
        "open_macro_round"
    ) not in {None, 0}:
        raise HeldoutBaselineError("policy-v0 baseline is no longer at round zero")
    active = coordinator.get("active_policy")
    if not isinstance(active, Mapping) or active.get("version") != 0:
        raise HeldoutBaselineError("active policy is not policy-v0")
    if coordinator.get("candidate_policy") is not None:
        raise HeldoutBaselineError("a trained candidate already exists")
    return dict(campaign), dict(coordinator)


def _policy_v0_for_verification(
    campaign_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load policy-v0 anywhere in its still-open first macro-round."""

    campaign = state.load_campaign(campaign_root)
    coordinator = state.load_state(campaign_root)
    if coordinator.get("next_macro_round") != 0:
        raise HeldoutBaselineError("policy-v0 baseline is no longer at round zero")
    phase = coordinator.get("phase")
    expected_open = None if phase == "READY" else 0
    if coordinator.get("open_macro_round") != expected_open:
        raise HeldoutBaselineError("policy-v0 coordinator frontier is malformed")
    active = coordinator.get("active_policy")
    if not isinstance(active, Mapping) or active.get("version") != 0:
        raise HeldoutBaselineError("active policy is not policy-v0")
    return dict(campaign), dict(coordinator)


def _evaluation_bundle(
    *,
    campaign: Mapping[str, Any],
    active: Mapping[str, Any],
    policy_spec_path: Path,
    heldout_snapshot: Mapping[str, Any],
    code_sha256: str,
    batch_size: int,
) -> Any:
    # Keep this control-plane module importable on the lightweight RTX
    # interpreter.  Torch and the GR00T runtime are required only by the
    # explicitly invoked evaluation path.
    from round_dagger.gr00t_policy import load_policy_spec
    from round_dagger.learner import BUNDLE_SCHEMA, LearnerBundleConfig, LearnerError

    spec = load_policy_spec(policy_spec_path)
    if spec["checkpoint_sha256"] != active["checkpoint_sha256"]:
        raise HeldoutBaselineError("policy spec checkpoint differs from active policy")
    if spec["runtime_policy_sha256"] != active["runtime_policy_sha256"]:
        raise HeldoutBaselineError("policy spec runtime identity differs from active policy")
    input_contract = spec["input_contract"]
    output_contract = spec["output_contract"]
    try:
        return LearnerBundleConfig.from_mapping(
            {
                "schema": BUNDLE_SCHEMA,
                "campaign_id": campaign["campaign_id"],
                "macro_round": 0,
                "model_lineage_sha256": active["checkpoint_sha256"],
                "code_sha256": _sha(code_sha256, "baseline code SHA-256"),
                "rgb_mean": input_contract["rgb_mean"],
                "rgb_std": input_contract["rgb_std"],
                "rgb_input_key": "vision_obs",
                "rgb_layout": "NHWC",
                "output_key": "actions",
                "observation_key_map": {},
                "train_action_slice": output_contract["train_action_slice"],
                "batch_size": batch_size,
                "max_steps": 1,
                "save_every": 1,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "seed": 0,
                "partition_seed": 0,
                "initial_checkpoint_sha256": active["checkpoint_sha256"],
                "heldout_snapshot_sha256": heldout_snapshot["snapshot_sha256"],
                "heldout_snapshot_file_sha256": heldout_snapshot["file_sha256"],
                "heldout_eval_interval": 100,
                "model_config": spec,
            }
        )
    except LearnerError as exc:
        raise HeldoutBaselineError(f"cannot construct baseline bundle: {exc}") from exc


def _publish_idempotent(path: Path, document: Mapping[str, Any]) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise HeldoutBaselineError("baseline result path is unsafe")
        existing = artifacts.load_strict_json(path)
        if existing != document:
            raise HeldoutBaselineError("baseline result conflicts with immutable evidence")
        return artifacts.sha256_file(path)
    return artifacts.publish_immutable_json(path, document)


def evaluate_active_policy_v0(
    *,
    campaign_root: str | os.PathLike[str],
    heldout_snapshot_path: str | os.PathLike[str],
    heldout_snapshot_file_sha256: str,
    code_sha256: str,
    output: str | os.PathLike[str] | None = None,
    device: Any = "cpu",
    batch_size: int = 32,
    model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    adapter_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate active policy-v0 on all fixed-256 held-out transitions.

    The coordinator is read before and after evaluation and must remain
    byte-for-byte equivalent.  The only campaign write is the immutable result
    artifact under ``quality/``.
    """

    try:
        import torch

        from round_dagger.gr00t_policy import (
            create_vision_actor_obj_pred,
            vision_actor_obj_pred_adapter_factory,
        )
    except ImportError as exc:
        raise HeldoutBaselineError(
            "policy-v0 baseline requires the Torch/GR00T runtime"
        ) from exc

    if model_factory is None:
        model_factory = create_vision_actor_obj_pred
    if adapter_factory is None:
        adapter_factory = vision_actor_obj_pred_adapter_factory

    root = _absolute(campaign_root, "campaign root")
    if root.is_symlink() or not root.is_dir():
        raise HeldoutBaselineError("campaign root is missing or unsafe")
    campaign, before = _active_policy_v0(root)
    active = dict(before["active_policy"])
    checkpoint = root / active["checkpoint_path"]
    spec_path = Path(f"{checkpoint}.policy_spec.json")
    manifest_path = Path(f"{checkpoint}.manifest.json")
    if any(path.is_symlink() or not path.is_file() for path in (checkpoint, spec_path, manifest_path)):
        raise HeldoutBaselineError("active policy bundle is incomplete or unsafe")
    if artifacts.sha256_file(checkpoint) != active["checkpoint_sha256"]:
        raise HeldoutBaselineError("active checkpoint SHA-256 mismatch")
    if artifacts.sha256_file(spec_path) != active["policy_spec_sha256"]:
        raise HeldoutBaselineError("active policy spec SHA-256 mismatch")
    if artifacts.sha256_file(manifest_path) != active[
        "policy_bundle_manifest_sha256"
    ]:
        raise HeldoutBaselineError("active policy manifest SHA-256 mismatch")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=spec_path,
        manifest=manifest_path,
        expected_checkpoint_sha256=active["checkpoint_sha256"],
        expected_runtime_policy_sha256=active["runtime_policy_sha256"],
    )

    heldout_path = _absolute(heldout_snapshot_path, "held-out snapshot")
    expected_heldout_file_sha = _sha(
        heldout_snapshot_file_sha256, "held-out snapshot file SHA-256"
    )
    heldout = load_heldout_snapshot(
        heldout_path, expected_file_sha256=expected_heldout_file_sha
    )
    if heldout["campaign_id"] != campaign["campaign_id"]:
        raise HeldoutBaselineError("held-out snapshot belongs to another campaign")
    if heldout["transition_count"] != 256:
        raise HeldoutBaselineError("policy-v0 baseline requires exactly 256 cases")
    config = _evaluation_bundle(
        campaign=campaign,
        active=active,
        policy_spec_path=spec_path,
        heldout_snapshot=heldout,
        code_sha256=code_sha256,
        batch_size=batch_size,
    )
    model = model_factory(config.to_mapping())
    if not isinstance(model, torch.nn.Module):
        raise HeldoutBaselineError("model factory did not return torch.nn.Module")
    load_checkpoint_strict(
        model,
        checkpoint,
        expected_sha256=active["checkpoint_sha256"],
        map_location="cpu",
    )
    adapter = adapter_factory(config.to_mapping())
    if not hasattr(adapter, "actions"):
        raise HeldoutBaselineError("adapter factory result lacks actions()")
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    partition = load_heldout_partition(heldout_path, config, rank=0, world_size=1)
    record = evaluate_heldout_actions(
        model,
        partition=partition,
        adapter=adapter,
        device=resolved_device,
        optimizer_step=0,
    )
    result_path = (
        root / CANONICAL_RESULT_RELATIVE
        if output is None
        else _absolute(output, "baseline output")
    )
    if result_path != root / CANONICAL_RESULT_RELATIVE:
        raise HeldoutBaselineError(
            f"baseline output must be {root / CANONICAL_RESULT_RELATIVE}"
        )
    document = {
        "schema": BASELINE_RESULT_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": artifacts.canonical_json_sha256(campaign),
        "policy_version": 0,
        "checkpoint_path": active["checkpoint_path"],
        "checkpoint_sha256": active["checkpoint_sha256"],
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "policy_spec_sha256": active["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": active[
            "policy_bundle_manifest_sha256"
        ],
        "evaluation_bundle_sha256": config.sha256,
        "heldout_snapshot_path": str(heldout_path),
        "heldout_snapshot_file_sha256": heldout["file_sha256"],
        "heldout_snapshot_sha256": heldout["snapshot_sha256"],
        "case_count": 256,
        "optimizer_step": 0,
        "metrics": record,
    }
    # The evaluation is observational.  Re-read the coordinator *before* the
    # immutable publication frontier so a concurrent collection/promotion can
    # never leave a result that claims to describe a different active policy.
    after = state.load_state(root)
    if after != before:
        raise HeldoutBaselineError("baseline evaluation mutated coordinator state")
    digest = _publish_idempotent(result_path, document)
    return {
        **document,
        "result_path": str(result_path),
        "result_sha256": digest,
    }


def wandb_metrics(result: Mapping[str, Any]) -> dict[str, int | float]:
    if result.get("schema") != BASELINE_RESULT_SCHEMA:
        raise HeldoutBaselineError("baseline result schema mismatch")
    record = result.get("metrics")
    if not isinstance(record, Mapping):
        raise HeldoutBaselineError("baseline result metrics are malformed")
    required = {
        "action_mse",
        "action_mae",
        "action_rmse",
        "action_cosine",
        "heldout_transitions_global",
        "per_action",
    }
    if not required.issubset(record):
        raise HeldoutBaselineError("baseline result lacks action-quality metrics")
    payload: dict[str, int | float] = {
        "baseline/optimizer_step": 0,
        "baseline/evaluation_complete": 1,
        "baseline/case_count": int(result["case_count"]),
        "baseline/action_mse": float(record["action_mse"]),
        "baseline/action_mae": float(record["action_mae"]),
        "baseline/action_rmse": float(record["action_rmse"]),
        "baseline/action_cosine": float(record["action_cosine"]),
        # Mirror the learner's established names so dashboard panels can plot
        # baseline and later step-100 samples without translating semantics.
        "quality/heldout_action_mse": float(record["action_mse"]),
        "quality/heldout_action_mae": float(record["action_mae"]),
        "quality/heldout_action_rmse": float(record["action_rmse"]),
        "quality/heldout_action_cosine": float(record["action_cosine"]),
        "quality/heldout_transitions": int(record["heldout_transitions_global"]),
    }
    per_action = record["per_action"]
    if not isinstance(per_action, Mapping):
        raise HeldoutBaselineError("baseline per-action metrics are malformed")
    for action_index, metrics in per_action.items():
        if not isinstance(metrics, Mapping):
            raise HeldoutBaselineError("baseline per-action row is malformed")
        for name in ("mse", "mae", "rmse", "cosine"):
            value = float(metrics[name])
            if not math.isfinite(value):
                raise HeldoutBaselineError("baseline W&B metric is non-finite")
            payload[f"quality/action_{action_index}/{name}"] = value
    if any(isinstance(value, float) and not math.isfinite(value) for value in payload.values()):
        raise HeldoutBaselineError("baseline W&B payload is non-finite")
    return payload


def load_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    result_path = _absolute(path, "baseline result")
    document = artifacts.load_strict_json(result_path)
    if not isinstance(document, Mapping) or document.get("schema") != BASELINE_RESULT_SCHEMA:
        raise HeldoutBaselineError("baseline result is malformed")
    # Exercise the complete scalar mapper as a bounded structural verifier.
    wandb_metrics(document)
    return dict(document)


def verify_active_policy_v0_result(
    campaign_root: str | os.PathLike[str],
    *,
    require_wandb_receipt: bool = False,
) -> dict[str, Any]:
    """Verify the immutable baseline still binds the active policy-v0 state."""

    root = _absolute(campaign_root, "campaign root")
    campaign, coordinator = _policy_v0_for_verification(root)
    result_path = root / CANONICAL_RESULT_RELATIVE
    result = load_result(result_path)
    active = coordinator["active_policy"]
    expected = {
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": artifacts.canonical_json_sha256(campaign),
        "policy_version": 0,
        "checkpoint_path": active["checkpoint_path"],
        "checkpoint_sha256": active["checkpoint_sha256"],
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "policy_spec_sha256": active["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": active[
            "policy_bundle_manifest_sha256"
        ],
        "case_count": 256,
        "optimizer_step": 0,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise HeldoutBaselineError(
            f"policy-v0 baseline active-policy binding differs: {mismatches}"
        )
    result["result_path"] = str(result_path)
    result["result_sha256"] = artifacts.sha256_file(result_path)
    if require_wandb_receipt:
        receipt_path = root / CANONICAL_WANDB_RECEIPT_RELATIVE
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise HeldoutBaselineError("policy-v0 baseline W&B receipt is missing")
        receipt = artifacts.load_strict_json(receipt_path)
        receipt_keys = {
            "schema",
            "campaign_id",
            "checkpoint_sha256",
            "heldout_snapshot_sha256",
            "result_sha256",
            "run_id",
            "entity",
            "project",
            "optimizer_step",
        }
        expected_receipt = {
            "schema": BASELINE_WANDB_RECEIPT_SCHEMA,
            "campaign_id": result["campaign_id"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "heldout_snapshot_sha256": result["heldout_snapshot_sha256"],
            "result_sha256": result["result_sha256"],
            "run_id": baseline_wandb_run_id(
                campaign_id=result["campaign_id"],
                campaign_sha256=result["campaign_sha256"],
                checkpoint_sha256=result["checkpoint_sha256"],
                heldout_snapshot_sha256=result["heldout_snapshot_sha256"],
            ),
            "optimizer_step": 0,
        }
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != receipt_keys
            or any(receipt.get(key) != value for key, value in expected_receipt.items())
        ):
            raise HeldoutBaselineError("policy-v0 baseline W&B receipt differs")
        if not all(
            isinstance(receipt.get(key), str) and bool(receipt[key])
            for key in ("entity", "project")
        ):
            raise HeldoutBaselineError("policy-v0 baseline W&B destination is missing")
        result["wandb_receipt_path"] = str(receipt_path)
        result["wandb_receipt_sha256"] = artifacts.sha256_file(receipt_path)
    return result


__all__ = [
    "BASELINE_RESULT_SCHEMA",
    "BASELINE_WANDB_RECEIPT_SCHEMA",
    "CANONICAL_RESULT_RELATIVE",
    "CANONICAL_WANDB_RECEIPT_RELATIVE",
    "HeldoutBaselineError",
    "baseline_wandb_run_id",
    "evaluate_active_policy_v0",
    "load_result",
    "verify_active_policy_v0_result",
    "wandb_metrics",
]
