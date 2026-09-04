"""Create one fresh, backend-portable production DAgger campaign.

This module is deliberately narrower than the production launcher.  It turns
already verified capture/runtime inputs into a *new* campaign root and never
adopts, repairs, or rewrites an existing campaign.  A failed attempt leaves its
new root quarantined; retrying therefore requires another previously absent
path.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from round_dagger import artifacts, state
from round_dagger.heldout_capture import verify_capture_aggregate
from round_dagger.heldout_conversion import (
    convert_capture_aggregate,
    verify_conversion_bundle,
)
from round_dagger.heldout_learner_contract import (
    materialize_heldout_learner_contract,
)
from round_dagger.orchestration import LEARNER_BACKEND_LEPTON_JOB_V1
from round_dagger.promotion import stage_bootstrap_policy
from scripts.round_dagger import production_launch


REPORT_SCHEMA = "fast_roborender.round_dagger.fresh_lepton_campaign.v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class FreshLeptonCampaignError(RuntimeError):
    """Fresh campaign inputs or outputs failed an identity boundary."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise FreshLeptonCampaignError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(value)
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    if not path.is_absolute():  # pragma: no cover - abspath is defensive
        raise FreshLeptonCampaignError(f"{label} must be absolute")
    return path


def _real_directory(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise FreshLeptonCampaignError(f"cannot inspect {label}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise FreshLeptonCampaignError(f"{label} must be a real directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - lstat/resolve race defense
        raise FreshLeptonCampaignError(f"cannot resolve {label}") from exc
    if resolved != path:
        raise FreshLeptonCampaignError(f"{label} must be canonical")
    return path


def _regular(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise FreshLeptonCampaignError(f"cannot inspect {label}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FreshLeptonCampaignError(
            f"{label} must be a regular non-symlink file"
        )
    return path


def _pinned_file(
    path_value: str | os.PathLike[str], expected_sha256: str, label: str
) -> Path:
    path = _regular(_absolute(path_value, label), label)
    expected = _sha(expected_sha256, f"{label} SHA-256")
    if artifacts.sha256_file(path) != expected:
        raise FreshLeptonCampaignError(f"{label} SHA-256 mismatch")
    return path


def _new_root(path_value: str | os.PathLike[str], protected_root: Path) -> Path:
    path = _absolute(path_value, "new campaign root")
    if os.path.lexists(path):
        raise FreshLeptonCampaignError(
            "new campaign root already exists; adoption or replacement is forbidden"
        )
    parent = _real_directory(path.parent, "new campaign parent")
    path = parent / path.name
    if (
        path == protected_root
        or protected_root in path.parents
        or path in protected_root.parents
    ):
        raise FreshLeptonCampaignError(
            "new campaign root overlaps the protected OCI campaign"
        )
    for ancestor in path.parents:
        if (ancestor / "campaign.json").exists() or (
            ancestor / "coordinator_state.json"
        ).exists():
            raise FreshLeptonCampaignError(
                "new campaign root would be nested inside an existing campaign"
            )
    return path


def _remote_root(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "//" in value
        or any(part in {".", ".."} for part in Path(value).parts)
    ):
        raise FreshLeptonCampaignError(f"{label} must be a canonical absolute path")
    return value


def _same_path_and_sha(
    *,
    path: Path,
    expected_path: Any,
    expected_sha256: Any,
    actual_sha256: str,
    label: str,
) -> None:
    if not isinstance(expected_path, str) or _absolute(expected_path, label) != path:
        raise FreshLeptonCampaignError(f"{label} path differs from runtime contract")
    if expected_sha256 != actual_sha256:
        raise FreshLeptonCampaignError(f"{label} digest differs from runtime contract")


def _validate_materialized_contract(contract: Mapping[str, Any]) -> None:
    topology = contract.get("topology")
    expected_topology = {
        "target_environments": 256,
        "active_environments": 64,
        "cohort_count": 4,
        "environments_per_actor_rank": 32,
        "actor_ranks_per_cohort": 2,
        "macro_round_count": 5,
        "steps_per_environment": 512,
        "transitions_per_cohort": 32768,
        "transitions_per_macro_round": 131072,
        "transitions_per_campaign": 655360,
    }
    if not isinstance(topology, Mapping) or dict(topology) != expected_topology:
        raise FreshLeptonCampaignError("runtime topology is not exact production 256")
    observation = contract.get("observation_contract")
    action = contract.get("action_contract")
    if (
        not isinstance(observation, Mapping)
        or observation.get("policy_views")
        != ["head_color", "hand_left", "hand_right"]
        or observation.get("policy_image_shape") != [324, 192, 3]
        or observation.get("policy_view_layout") != "vertical"
        or observation.get("reference_lag_frames") != 256
        or not isinstance(action, Mapping)
        or action.get("replay_action_dim") != 26
    ):
        raise FreshLeptonCampaignError(
            "runtime observation/action contract is not exact three-view production"
        )


def _capture_inventory_sha(runtime: Mapping[str, Any]) -> str:
    provenance = runtime.get("capture_provenance")
    value = (
        provenance.get("evaluation_source_inventory_sha256")
        if isinstance(provenance, Mapping)
        else None
    )
    return _sha(value, "runtime evaluation source inventory")


def _input_inventory(
    *,
    invariant_profile: Path,
    invariant_profile_sha256: str,
    runtime_contract: Path,
    runtime_contract_sha256: str,
    capture_aggregate: Path,
    capture_aggregate_sha256: str,
    heldout_plan: Path,
    heldout_plan_sha256: str,
    motion_split: Path,
    motion_split_sha256: str,
    scenarios: Path,
    scenarios_sha256: str,
    base_learner_contract: Path,
    base_learner_contract_sha256: str,
) -> dict[str, dict[str, str]]:
    return {
        "invariant_profile": {
            "path": str(invariant_profile),
            "sha256": invariant_profile_sha256,
        },
        "runtime_contract": {
            "path": str(runtime_contract),
            "sha256": runtime_contract_sha256,
        },
        "capture_aggregate": {
            "path": str(capture_aggregate),
            "sha256": capture_aggregate_sha256,
        },
        "heldout_plan": {"path": str(heldout_plan), "sha256": heldout_plan_sha256},
        "motion_split": {"path": str(motion_split), "sha256": motion_split_sha256},
        "scenarios": {"path": str(scenarios), "sha256": scenarios_sha256},
        "base_learner_contract": {
            "path": str(base_learner_contract),
            "sha256": base_learner_contract_sha256,
        },
    }


def materialize_fresh_lepton_campaign(
    *,
    campaign_root: str | os.PathLike[str],
    protected_oci_campaign_root: str | os.PathLike[str],
    invariant_profile_path: str | os.PathLike[str],
    invariant_profile_sha256: str,
    runtime_contract_path: str | os.PathLike[str],
    runtime_contract_sha256: str,
    capture_aggregate_path: str | os.PathLike[str],
    capture_aggregate_sha256: str,
    heldout_plan_path: str | os.PathLike[str],
    heldout_plan_sha256: str,
    motion_split_path: str | os.PathLike[str],
    motion_split_sha256: str,
    scenarios_path: str | os.PathLike[str],
    scenarios_sha256: str,
    base_learner_contract_path: str | os.PathLike[str],
    base_learner_contract_sha256: str,
    learner_sealed_source_root: str,
    learner_source_inventory_sha256: str,
    evaluation_sealed_source_root: str | os.PathLike[str],
    evaluation_source_inventory_sha256: str,
) -> dict[str, Any]:
    """Materialize and recursively validate a fresh Lepton campaign root."""

    protected = _real_directory(
        _absolute(protected_oci_campaign_root, "protected OCI campaign root"),
        "protected OCI campaign root",
    )
    root = _new_root(campaign_root, protected)
    pinned = {
        "invariant_profile": _pinned_file(
            invariant_profile_path, invariant_profile_sha256, "invariant profile"
        ),
        "runtime_contract": _pinned_file(
            runtime_contract_path, runtime_contract_sha256, "runtime contract"
        ),
        "capture_aggregate": _pinned_file(
            capture_aggregate_path, capture_aggregate_sha256, "capture aggregate"
        ),
        "heldout_plan": _pinned_file(
            heldout_plan_path, heldout_plan_sha256, "held-out plan"
        ),
        "motion_split": _pinned_file(
            motion_split_path, motion_split_sha256, "motion split"
        ),
        "scenarios": _pinned_file(scenarios_path, scenarios_sha256, "scenarios"),
        "base_learner_contract": _pinned_file(
            base_learner_contract_path,
            base_learner_contract_sha256,
            "base learner contract",
        ),
    }
    input_shas = {
        "invariant_profile": _sha(invariant_profile_sha256, "invariant profile SHA-256"),
        "runtime_contract": _sha(runtime_contract_sha256, "runtime contract SHA-256"),
        "capture_aggregate": _sha(capture_aggregate_sha256, "capture aggregate SHA-256"),
        "heldout_plan": _sha(heldout_plan_sha256, "held-out plan SHA-256"),
        "motion_split": _sha(motion_split_sha256, "motion split SHA-256"),
        "scenarios": _sha(scenarios_sha256, "scenarios SHA-256"),
        "base_learner_contract": _sha(
            base_learner_contract_sha256, "base learner contract SHA-256"
        ),
    }

    try:
        materialized, runtime, lineage = production_launch.materialize_campaign_contract(
            pinned["invariant_profile"],
            runtime_contract_path=pinned["runtime_contract"],
            runtime_contract_sha256=input_shas["runtime_contract"],
        )
    except Exception as exc:
        raise FreshLeptonCampaignError(
            f"runtime campaign contract verification failed: {type(exc).__name__}"
        ) from exc
    _validate_materialized_contract(materialized)
    if materialized.get("campaign_id") is None or _TOKEN_RE.fullmatch(
        str(materialized.get("campaign_id"))
    ) is None:
        raise FreshLeptonCampaignError("materialized campaign ID is malformed")
    campaign_id = str(materialized["campaign_id"])
    if runtime.get("contract_sha256") != input_shas["runtime_contract"]:
        raise FreshLeptonCampaignError("runtime contract digest normalization differs")
    expected_inventory = _sha(
        evaluation_source_inventory_sha256, "evaluation source inventory SHA-256"
    )
    learner_inventory = _sha(
        learner_source_inventory_sha256, "learner source inventory SHA-256"
    )
    learner_root = _remote_root(
        learner_sealed_source_root, "learner sealed source root"
    )
    evaluation_root = _real_directory(
        _absolute(evaluation_sealed_source_root, "evaluation sealed source root"),
        "evaluation sealed source root",
    )
    if learner_inventory == expected_inventory:
        raise FreshLeptonCampaignError(
            "learner and accepted evaluation source inventories must be independent"
        )
    if _capture_inventory_sha(runtime) != expected_inventory:
        raise FreshLeptonCampaignError(
            "evaluation source inventory differs from verified runtime capture"
        )
    try:
        evaluation_seal = production_launch._verify_local_evaluation_seal(
            evaluation_root,
            expected_inventory,
            campaign_contract=materialized,
        )
    except Exception as exc:
        raise FreshLeptonCampaignError(
            f"accepted evaluation source seal verification failed: {type(exc).__name__}"
        ) from exc
    if (
        evaluation_seal.get("root") != str(evaluation_root)
        or evaluation_seal.get("inventory_sha256") != expected_inventory
    ):
        raise FreshLeptonCampaignError("accepted evaluation source seal identity differs")

    _same_path_and_sha(
        path=pinned["heldout_plan"],
        expected_path=runtime.get("heldout_plan_path"),
        expected_sha256=materialized["motion_split"].get("heldout_plan_sha256"),
        actual_sha256=input_shas["heldout_plan"],
        label="held-out plan",
    )
    _same_path_and_sha(
        path=pinned["motion_split"],
        expected_path=runtime.get("split_manifest_path"),
        expected_sha256=materialized["motion_split"].get("manifest_sha256"),
        actual_sha256=input_shas["motion_split"],
        label="motion split",
    )
    _same_path_and_sha(
        path=pinned["scenarios"],
        expected_path=runtime.get("scenario_path"),
        expected_sha256=materialized["evaluation"].get("scenario_contract_sha256"),
        actual_sha256=input_shas["scenarios"],
        label="scenarios",
    )
    try:
        aggregate = verify_capture_aggregate(pinned["capture_aggregate"])
        base_contract = production_launch.load_learner_contract(
            pinned["base_learner_contract"]
        )
    except Exception as exc:
        raise FreshLeptonCampaignError(
            f"capture/base learner verification failed: {type(exc).__name__}"
        ) from exc
    if aggregate.get("file_sha256") != input_shas["capture_aggregate"]:
        raise FreshLeptonCampaignError("verified capture aggregate digest differs")
    if aggregate.get("plan_sha256") != input_shas["heldout_plan"]:
        raise FreshLeptonCampaignError("capture aggregate uses another held-out plan")
    if (
        base_contract.get("schema")
        != production_launch.LEARNER_CONTRACT_SCHEMA
        or production_launch.learner_contract_backend(base_contract) is not None
        or "heldout_evaluation" in base_contract
    ):
        raise FreshLeptonCampaignError(
            "base learner contract must be unbound backend-portable v2"
        )

    bootstrap_contract = materialized["artifacts"]["bootstrap_bundle"]
    bootstrap_source = _pinned_file(
        bootstrap_contract["checkpoint_path"],
        bootstrap_contract["checkpoint_sha256"],
        "bootstrap checkpoint",
    )
    identity = {
        "policy_sha256": bootstrap_contract["checkpoint_sha256"],
        "runtime_policy_sha256": bootstrap_contract["runtime_policy_sha256"],
        "teacher_sha256": materialized["artifacts"]["teacher"]["checkpoint_sha256"],
        "renderer_sha256": materialized["artifacts"]["renderer"][
            "cohort_identity_sha256"
        ],
        "beta_sha256": artifacts.canonical_json_sha256(
            {"beta": float(materialized["beta_schedule"]["start"])}
        ),
        "config_sha256": materialized["evaluation"]["hydra_config_sha256"],
    }

    # Rehash every caller-selected file immediately before the sole mkdir.
    for name, path in pinned.items():
        if artifacts.sha256_file(path) != input_shas[name]:
            raise FreshLeptonCampaignError(f"{name} changed before publication")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:  # concurrent create
        raise FreshLeptonCampaignError(
            "new campaign root appeared concurrently; publication refused"
        ) from exc

    try:
        conversion = convert_capture_aggregate(
            pinned["capture_aggregate"],
            campaign_root=root,
            campaign_id=campaign_id,
            identity=identity,
            expected_beta=float(materialized["beta_schedule"]["start"]),
            plan=pinned["heldout_plan"],
            motion_split=pinned["motion_split"],
            scenarios=pinned["scenarios"],
        )
        if conversion.get("case_count") != 256 or conversion.get("cohort_count") != 8:
            raise FreshLeptonCampaignError("held-out conversion is not exact 8x32")
        attestation = root / "heldout_conversion_attestation.json"
        verified_conversion = verify_conversion_bundle(attestation)
        if verified_conversion.get("file_sha256") != conversion.get("file_sha256"):
            raise FreshLeptonCampaignError("held-out conversion identity changed")

        learner_report = materialize_heldout_learner_contract(
            base_contract_path=pinned["base_learner_contract"],
            conversion_attestation_path=attestation,
            campaign_root=root,
            campaign_id=campaign_id,
            output_path=root / "learner_contract.json",
        )
        learner_path = root / "learner_contract.json"
        learner_sha = str(learner_report["output_file_sha256"])
        learner_contract, observed_learner_sha = (
            production_launch.load_campaign_owned_learner_contract(
                learner_path,
                campaign_root=root,
                expected_sha256=learner_sha,
                expected_backend=LEARNER_BACKEND_LEPTON_JOB_V1,
            )
        )
        if (
            observed_learner_sha != learner_sha
            or learner_contract.get("schema")
            != production_launch.LEARNER_CONTRACT_SCHEMA
            or production_launch.learner_contract_backend(learner_contract) is not None
        ):
            raise FreshLeptonCampaignError(
                "campaign learner contract is not backend-portable v2"
            )

        bootstrap = stage_bootstrap_policy(
            root,
            checkpoint_source=bootstrap_source,
            code_sha256=expected_inventory,
            policy_config_sha256=learner_sha,
        )
        initialized = state.initialize_campaign(
            root,
            campaign_id=campaign_id,
            bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
            bootstrap_policy_path=bootstrap["checkpoint_path"],
            target_environments=materialized["topology"]["target_environments"],
            active_environments=materialized["topology"]["active_environments"],
            steps_per_environment=materialized["topology"]["steps_per_environment"],
            view_names=tuple(materialized["observation_contract"]["policy_views"]),
            policy_image_shape=tuple(
                materialized["observation_contract"]["policy_image_shape"]
            ),
            action_dim=materialized["action_contract"]["replay_action_dim"],
            teacher_sha256=identity["teacher_sha256"],
            renderer_sha256=identity["renderer_sha256"],
            camera_mapping_sha256=materialized["artifacts"]["camera_acceptance"][
                "sha256"
            ],
            training_config_sha256=identity["config_sha256"],
            beta_schedule=materialized["beta_schedule"],
            macro_round_count=materialized["topology"]["macro_round_count"],
            data_lineage=lineage,
        )
        campaign = state.load_campaign(root)
        coordinator = state.load_state(root)
    except FreshLeptonCampaignError:
        raise
    except Exception as exc:
        raise FreshLeptonCampaignError(
            f"fresh campaign publication failed: {type(exc).__name__}"
        ) from exc

    expected_campaign = {
        "campaign_id": campaign_id,
        "target_environments": 256,
        "active_environments": 64,
        "cohort_count": 4,
        "steps_per_environment": 512,
        "macro_round_count": 5,
        "view_names": ["head_color", "hand_left", "hand_right"],
        "policy_image_shape": [324, 192, 3],
        "action_dim": 26,
        "teacher_sha256": identity["teacher_sha256"],
        "renderer_sha256": identity["renderer_sha256"],
        "camera_mapping_sha256": materialized["artifacts"]["camera_acceptance"][
            "sha256"
        ],
        "training_config_sha256": identity["config_sha256"],
        "code_sha256": expected_inventory,
        "policy_config_sha256": learner_sha,
        "beta_schedule": materialized["beta_schedule"],
        "data_lineage": lineage,
        "data_lineage_sha256": materialized["data_lineage_sha256"],
    }
    mismatches = [
        key for key, value in expected_campaign.items() if campaign.get(key) != value
    ]
    if mismatches:
        raise FreshLeptonCampaignError(
            f"fresh campaign differs from verified production inputs: {mismatches}"
        )
    expected_state = {
        "revision": 0,
        "phase": "READY",
        "next_macro_round": 0,
        "open_macro_round": None,
        "last_collected_macro_round": -1,
        "last_ingested_macro_round": -1,
        "last_trained_macro_round": -1,
        "last_evaluated_macro_round": -1,
        "candidate_policy": None,
    }
    state_mismatches = [
        key for key, value in expected_state.items() if coordinator.get(key) != value
    ]
    active = coordinator.get("active_policy", {})
    if (
        state_mismatches
        or active.get("version") != 0
        or active.get("code_sha256") != expected_inventory
        or active.get("policy_config_sha256") != learner_sha
    ):
        raise FreshLeptonCampaignError(
            "fresh coordinator state is not pristine production version zero"
        )

    inputs = _input_inventory(
        invariant_profile=pinned["invariant_profile"],
        invariant_profile_sha256=input_shas["invariant_profile"],
        runtime_contract=pinned["runtime_contract"],
        runtime_contract_sha256=input_shas["runtime_contract"],
        capture_aggregate=pinned["capture_aggregate"],
        capture_aggregate_sha256=input_shas["capture_aggregate"],
        heldout_plan=pinned["heldout_plan"],
        heldout_plan_sha256=input_shas["heldout_plan"],
        motion_split=pinned["motion_split"],
        motion_split_sha256=input_shas["motion_split"],
        scenarios=pinned["scenarios"],
        scenarios_sha256=input_shas["scenarios"],
        base_learner_contract=pinned["base_learner_contract"],
        base_learner_contract_sha256=input_shas["base_learner_contract"],
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "campaign_root": str(root),
        "protected_oci_campaign_root": str(protected),
        "campaign_id": campaign_id,
        "inputs": inputs,
        "runtime_contract_sha256": input_shas["runtime_contract"],
        "source_seals": {
            "learner": {
                "root": learner_root,
                "inventory_sha256": learner_inventory,
            },
            "evaluation": {
                "root": str(evaluation_root),
                "inventory_sha256": expected_inventory,
                "snapshot_name": evaluation_seal.get("snapshot_name"),
            },
        },
        "heldout_conversion": {
            "attestation_path": str(attestation),
            "attestation_file_sha256": conversion["file_sha256"],
            "snapshot_path": verified_conversion["snapshot_path"],
            "snapshot_sha256": verified_conversion["snapshot_sha256"],
            "case_count": verified_conversion["case_count"],
            "cohort_count": verified_conversion["cohort_count"],
        },
        "learner_contract": {
            "path": str(learner_path),
            "sha256": learner_sha,
            "schema": learner_contract["schema"],
            "backend": None,
        },
        "bootstrap": {
            "checkpoint_path": active["checkpoint_path"],
            "checkpoint_sha256": active["checkpoint_sha256"],
            "runtime_policy_sha256": active["runtime_policy_sha256"],
            "manifest_sha256": active["manifest_sha256"],
            "code_sha256": active["code_sha256"],
            "policy_config_sha256": active["policy_config_sha256"],
        },
        "campaign_sha256": coordinator["campaign_sha256"],
        "data_lineage_sha256": campaign["data_lineage_sha256"],
        "coordinator": {
            "revision": coordinator["revision"],
            "phase": coordinator["phase"],
            "next_macro_round": coordinator["next_macro_round"],
            "active_policy_version": active["version"],
        },
    }
    evidence_path = root / "fresh_lepton_campaign_materialization.json"
    report["evidence_path"] = str(evidence_path)
    report["evidence_file_sha256"] = artifacts.publish_immutable_json(
        evidence_path, report
    )
    # The evidence file intentionally cannot self-bind its own digest.  Return
    # the observed file digest beside the immutable semantic document.
    return report


__all__ = [
    "FreshLeptonCampaignError",
    "REPORT_SCHEMA",
    "materialize_fresh_lepton_campaign",
]
