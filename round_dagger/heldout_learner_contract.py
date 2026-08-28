"""Materialize an exact production learner contract for fixed held-out eval.

The checked-in learner contract intentionally remains unbound while the live
8x32 expert capture is incomplete.  This module is the only publication path
from a recursively verified capture-conversion attestation to the four fields
consumed by the production launcher.  It never accepts caller-supplied snapshot
digests, so a path or digest placeholder cannot become launch eligible.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from round_dagger import artifacts, replay
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM, VISION_SHAPE
from round_dagger.heldout_conversion import verify_conversion_bundle


EVALUATION_INTERVAL_STEPS = 100
EXPECTED_CASE_COUNT = 256
EXPECTED_COHORT_COUNT = 8
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class HeldoutLearnerContractError(RuntimeError):
    """A verified fixed-256 bundle could not be bound to a learner contract."""


def _regular(path: str | os.PathLike[str], label: str) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        metadata = os.lstat(target)
    except OSError as exc:
        raise HeldoutLearnerContractError(
            f"cannot inspect {label} {target}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise HeldoutLearnerContractError(
            f"{label} must be a regular non-symlink file"
        )
    return target


def _directory(path: str | os.PathLike[str], label: str) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        metadata = os.lstat(target)
    except OSError as exc:
        raise HeldoutLearnerContractError(
            f"cannot inspect {label} {target}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise HeldoutLearnerContractError(
            f"{label} must be a real non-symlink directory"
        )
    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - lstat/resolve race defense
        raise HeldoutLearnerContractError(
            f"cannot resolve {label} {target}: {exc}"
        ) from exc
    if resolved != target:
        raise HeldoutLearnerContractError(f"{label} must be canonical")
    return target


def _safe_output(path: str | os.PathLike[str]) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if target.exists() or target.is_symlink():
        raise HeldoutLearnerContractError(
            "output learner contract already exists; rebinding is forbidden"
        )
    parent = _directory(target.parent, "output directory")
    return parent / target.name


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HeldoutLearnerContractError(f"{label} must be a lowercase SHA-256")
    return value


def _fixed_bundle(attestation: Mapping[str, Any]) -> dict[str, str]:
    fixed = attestation.get("fixed_bundle")
    expected = {
        "snapshot_path",
        "snapshot_file_sha256",
        "snapshot_sha256",
        "case_manifest_path",
        "case_manifest_file_sha256",
        "case_manifest_sha256",
        "verified_snapshot_sha256",
    }
    if not isinstance(fixed, Mapping) or set(fixed) != expected:
        raise HeldoutLearnerContractError(
            "conversion attestation has no exact fixed-bundle identity"
        )
    result = dict(fixed)
    for key in expected - {"snapshot_path", "case_manifest_path"}:
        _sha(result[key], f"fixed bundle {key}")
    if result["verified_snapshot_sha256"] != result["snapshot_sha256"]:
        raise HeldoutLearnerContractError(
            "conversion attestation did not verify the snapshot semantic identity"
        )
    return result


def materialize_heldout_learner_contract(
    *,
    base_contract_path: str | os.PathLike[str],
    conversion_attestation_path: str | os.PathLike[str],
    campaign_root: str | os.PathLike[str],
    campaign_id: str,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Publish an immutable learner contract bound to one verified snapshot.

    The base must be the ordinary schema-valid production contract without a
    ``heldout_evaluation`` member.  The new member is derived exclusively from
    the recursively verified conversion attestation.  Publication is
    create-only and an identical retry must use a new output path.
    """

    base_path = _regular(base_contract_path, "base learner contract")
    attestation_path = _regular(
        conversion_attestation_path, "held-out conversion attestation"
    )
    root = _directory(campaign_root, "campaign root")
    output = _safe_output(output_path)
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or len(campaign_id) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign_id) is None
    ):
        raise HeldoutLearnerContractError("campaign_id is not a safe identifier")

    # Import here to avoid coupling ordinary replay/conversion imports to the
    # deployment CLI module.
    from scripts.round_dagger import production_launch

    base_file_sha = artifacts.sha256_file(base_path)
    try:
        contract = production_launch.load_learner_contract(base_path)
    except Exception as exc:
        raise HeldoutLearnerContractError(
            f"base production learner contract is invalid: {exc}"
        ) from exc
    if "heldout_evaluation" in contract:
        raise HeldoutLearnerContractError(
            "base learner contract is already held-out-bound; rebinding is forbidden"
        )
    if artifacts.sha256_file(base_path) != base_file_sha:
        raise HeldoutLearnerContractError(
            "base learner contract changed while it was validated"
        )

    try:
        verified = verify_conversion_bundle(attestation_path)
    except Exception as exc:
        raise HeldoutLearnerContractError(
            f"held-out conversion bundle is invalid: {exc}"
        ) from exc
    if (
        verified.get("status") != "verified"
        or verified.get("campaign_id") != campaign_id
        or verified.get("case_count") != EXPECTED_CASE_COUNT
        or verified.get("cohort_count") != EXPECTED_COHORT_COUNT
    ):
        raise HeldoutLearnerContractError(
            "conversion is not the exact verified fixed-256 campaign bundle"
        )
    if attestation_path.parent != root:
        raise HeldoutLearnerContractError(
            "conversion attestation must be at the canonical campaign root"
        )

    attestation = artifacts.load_strict_json(attestation_path)
    fixed = _fixed_bundle(attestation)
    snapshot_path = _regular(fixed["snapshot_path"], "held-out replay snapshot")
    if snapshot_path.parent != root or snapshot_path.name != "heldout_snapshot.json":
        raise HeldoutLearnerContractError(
            "held-out replay snapshot is not at its canonical campaign path"
        )
    if verified.get("snapshot_path") != str(snapshot_path):
        raise HeldoutLearnerContractError(
            "verified conversion and attestation snapshot paths differ"
        )
    snapshot_file_sha = _sha(
        fixed["snapshot_file_sha256"], "held-out snapshot file SHA-256"
    )
    snapshot_semantic_sha = _sha(
        fixed["snapshot_sha256"], "held-out snapshot semantic SHA-256"
    )
    try:
        snapshot = replay.load_heldout_snapshot(
            snapshot_path, expected_file_sha256=snapshot_file_sha
        )
    except Exception as exc:
        raise HeldoutLearnerContractError(
            f"held-out replay snapshot is invalid: {exc}"
        ) from exc
    expected_view = {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": list(VISION_SHAPE),
        "views": ["head_color", "hand_left", "hand_right"],
    }
    expected_action = {"dtype": "float32", "shape": [REPLAY_ACTION_DIM]}
    if (
        snapshot.get("campaign_id") != campaign_id
        or snapshot.get("cohort_count") != EXPECTED_COHORT_COUNT
        or snapshot.get("transition_count") != EXPECTED_CASE_COUNT
        or snapshot.get("snapshot_sha256") != snapshot_semantic_sha
        or verified.get("snapshot_sha256") != snapshot_semantic_sha
        or snapshot.get("view_contract") != expected_view
        or snapshot.get("action_contract") != expected_action
    ):
        raise HeldoutLearnerContractError(
            "held-out snapshot is not exact 256-case three-view 26D evaluation data"
        )

    heldout = {
        "snapshot_path": str(snapshot_path),
        "snapshot_file_sha256": snapshot_file_sha,
        "snapshot_sha256": snapshot_semantic_sha,
        "interval_steps": EVALUATION_INTERVAL_STEPS,
    }
    materialized = dict(contract)
    materialized["heldout_evaluation"] = heldout
    try:
        output_file_sha = artifacts.publish_immutable_json(output, materialized)
        loaded = production_launch.load_learner_contract(output)
    except Exception as exc:
        raise HeldoutLearnerContractError(
            f"materialized learner contract failed production validation: {exc}"
        ) from exc
    if loaded != materialized or artifacts.sha256_file(output) != output_file_sha:
        raise HeldoutLearnerContractError(
            "published learner contract identity changed during validation"
        )

    # Re-run the recursive conversion verifier after publication.  Exact
    # digests in the new contract make any subsequent artifact mutation fail
    # production preflight as well.
    try:
        post = verify_conversion_bundle(attestation_path)
    except Exception as exc:
        raise HeldoutLearnerContractError(
            f"held-out conversion bundle failed post-publication verification: {exc}"
        ) from exc
    if post != verified:
        raise HeldoutLearnerContractError(
            "held-out conversion bundle changed while the contract was published"
        )
    return {
        "status": "materialized",
        "output_path": str(output),
        "output_file_sha256": output_file_sha,
        "base_contract_path": str(base_path),
        "base_contract_file_sha256": base_file_sha,
        "conversion_attestation_path": str(attestation_path),
        "conversion_attestation_file_sha256": verified["file_sha256"],
        "campaign_id": campaign_id,
        "case_count": EXPECTED_CASE_COUNT,
        "cohort_count": EXPECTED_COHORT_COUNT,
        "heldout_evaluation": heldout,
    }


__all__ = [
    "EVALUATION_INTERVAL_STEPS",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_COHORT_COUNT",
    "HeldoutLearnerContractError",
    "materialize_heldout_learner_contract",
]
