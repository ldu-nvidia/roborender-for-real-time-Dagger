"""Bounded recovery for a source-unbound legacy paired evaluation.

The normal macro-round owns ``rounds/.../evaluation``.  This module never
renames, removes, or overwrites that directory.  It inventories the completed
legacy evaluation, prepares a source-bound contract in a deterministic
``evaluation_recovery/hardened-<digest>`` namespace, and only after a complete
deep verification publishes the candidate evaluation and performs the normal
CAS-backed EVALUATED -> READY promotion.

It is intentionally not a general state migration mechanism.  Recovery is
accepted only for one open round produced by the bounded INGESTED learner
recovery, with an already-staged candidate and an attested dead legacy
coordinator.  A legacy evaluation may be absent or retained as diagnostic-only
evidence; it is never treated as promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from round_dagger import evaluation, orchestration, promotion, state
from round_dagger.artifacts import (
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)

PLAN_SCHEMA = "fast_roborender.round_dagger.hardened_evaluation_recovery.v2"
INVENTORY_SCHEMA = (
    "fast_roborender.round_dagger.legacy_evaluation_inventory.v1"
)
SOURCE_SCHEMA = "fast_roborender.round_dagger.recovery_source_bundle.v2"
COMPLETION_SCHEMA = (
    "fast_roborender.round_dagger.hardened_evaluation_recovery_completion.v1"
)
HOLD_SCHEMA = "fast_roborender.round_dagger.promotion_hold_attestation.v1"
LEARNER_RECOVERY_RECEIPT_SCHEMA = (
    "fast_roborender.round_dagger.ingested_learner_recovery_receipt.v1"
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_RECOVERY_ID_RE = re.compile(r"hardened-[0-9a-f]{20}")
_LEGACY_RESULT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "candidate_manifest_sha256",
    "evaluator_config_sha256",
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "passed",
    "metrics",
}
_PLAN_KEYS = {
    "schema",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "legacy_evaluation_root",
    "legacy_inventory_path",
    "legacy_inventory_sha256",
    "legacy_contract_sha256",
    "legacy_result_sha256",
    "promotion_hold_attestation_path",
    "promotion_hold_attestation_sha256",
    "learner_recovery_receipt_path",
    "learner_recovery_receipt_sha256",
    "learner_source_inventory_sha256",
    "evaluation_source_inventory_sha256",
    "anchor_pool_path",
    "anchor_count",
    "anchor_index",
    "anchor_inventory_sha256",
    "recovery_id",
    "recovery_root",
    "evaluator_contract_path",
    "evaluator_contract_sha256",
    "evaluator_result_path",
    "evaluator_source_bundle_path",
    "evaluator_source_bundle_sha256",
    "recovery_source_bundle_path",
    "recovery_source_bundle_sha256",
}
_LEARNER_RECEIPT_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "ingested_revision",
    "trained_revision",
    "replay_snapshot_file_sha256",
    "replay_snapshot_sha256",
    "sealed_source_inventory_sha256",
    "learner_config_sha256",
    "learner_result_path",
    "learner_result_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "candidate_checkpoint_sha256",
    "plan_sha256",
}
_HOLD_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "candidate_manifest_sha256",
    "policy_directory",
    "held_mode_octal",
    "previous_mode_octal",
    "legacy_orchestrator_pid",
    "legacy_evaluation_presence",
    "legacy_result_path",
    "legacy_result_sha256",
}
_RECOVERY_REPOSITORY_SOURCES = (
    "round_dagger/evaluation_recovery.py",
    "round_dagger/promotion.py",
    "round_dagger/state.py",
    "round_dagger/orchestration.py",
    "round_dagger/learner_recovery.py",
    "scripts/round_dagger/run_hardened_evaluation_recovery.sh",
)
RECOVERY_EVALUATION_TIMEOUT_SECONDS = 28_800


class EvaluationRecoveryError(ArtifactError):
    """Recovery evidence or coordinator preconditions are incomplete."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise EvaluationRecoveryError(
            f"{label} has missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise EvaluationRecoveryError(f"{label} must be a lowercase SHA-256")
    return value


def _root(value: str | Path) -> Path:
    try:
        return state._campaign_root(value)
    except (ArtifactError, OSError, ValueError) as exc:
        raise EvaluationRecoveryError("campaign root is missing or unsafe") from exc


def _relative(root: Path, path: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise EvaluationRecoveryError(f"{label} escaped campaign root") from exc


def _inventory_tree(root: Path, directory: Path) -> dict[str, Any]:
    if not directory.exists():
        return {
            "schema": INVENTORY_SCHEMA,
            "root": _relative(root, directory, "legacy evaluation root"),
            "presence": "absent",
            "files": [],
        }
    if directory.is_symlink() or not directory.is_dir():
        raise EvaluationRecoveryError("legacy evaluation directory is unsafe")
    files: list[dict[str, Any]] = []
    for current, directories, names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise EvaluationRecoveryError(
                    "legacy evaluation contains an unsafe directory"
                )
        for name in names:
            child = current_path / name
            if child.is_symlink() or not child.is_file():
                raise EvaluationRecoveryError(
                    "legacy evaluation contains a non-regular artifact"
                )
            files.append(
                {
                    "path": _relative(root, child, "legacy evaluation artifact"),
                    "size": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    files.sort(key=lambda item: item["path"])
    return {
        "schema": INVENTORY_SCHEMA,
        "root": _relative(root, directory, "legacy evaluation root"),
        "presence": "present",
        "files": files,
    }


def _require_source_root(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise EvaluationRecoveryError(f"{label} must be absolute")
    path = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_dir():
        raise EvaluationRecoveryError(f"{label} is missing or unsafe")
    return path


def recovery_source_bundle(
    repository_root: str | Path,
    gr00t_root: str | Path,
    *,
    learner_source_inventory_sha256: str,
    evaluation_source_inventory_sha256: str,
) -> dict[str, Any]:
    repository = _require_source_root(repository_root, "repository root")
    gr00t = _require_source_root(gr00t_root, "GR00T root")
    learner_inventory_sha = _sha(
        learner_source_inventory_sha256,
        "learner source inventory SHA-256",
    )
    evaluation_inventory_sha = _sha(
        evaluation_source_inventory_sha256,
        "evaluation source inventory SHA-256",
    )
    evaluator = evaluation.evaluator_source_bundle(
        repository,
        gr00t,
        evaluation_inventory_sha,
    )
    entries = list(evaluator["sources"])
    known = {entry["source"] for entry in entries}
    for relative in _RECOVERY_REPOSITORY_SOURCES:
        name = f"repository/{relative}"
        if name in known:
            continue
        source = repository / relative
        if source.is_symlink() or not source.is_file():
            raise EvaluationRecoveryError(
                f"recovery source is missing or unsafe: {name}"
            )
        entries.append({"source": name, "sha256": sha256_file(source)})
    entries.sort(key=lambda item: item["source"])
    return {
        "schema": SOURCE_SCHEMA,
        "learner_source_inventory_sha256": learner_inventory_sha,
        "evaluation_source_inventory_sha256": evaluation_inventory_sha,
        "sources": entries,
    }


def _legacy_result(
    root: Path,
    *,
    legacy_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = legacy_root / "result.json"
    result = _exact(
        load_strict_json(result_path), _LEGACY_RESULT_KEYS, "legacy evaluator result"
    )
    expected = {
        "schema": evaluation.EVALUATOR_RESULT_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "macro_round": contract["macro_round"],
        "candidate_manifest_sha256": contract["candidate_manifest_sha256"],
        "evaluator_config_sha256": contract["evaluation_config_sha256"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise EvaluationRecoveryError("legacy evaluator result lineage mismatch")
    if type(result["passed"]) is not bool or not isinstance(result["metrics"], Mapping):
        raise EvaluationRecoveryError("legacy evaluator result payload is malformed")
    bundle, relative = state._internal_path(
        root, result["evidence_bundle_path"], "legacy evidence bundle"
    )
    expected_bundle = legacy_root / "results" / "evaluation_bundle.json"
    if bundle != expected_bundle or result["evidence_bundle_path"] != relative:
        raise EvaluationRecoveryError("legacy evidence bundle path is non-canonical")
    if sha256_file(bundle) != _sha(
        result["evidence_bundle_sha256"], "legacy evidence bundle SHA"
    ):
        raise EvaluationRecoveryError("legacy evidence bundle SHA mismatch")
    return result


def _pid_exists(pid: int) -> bool:
    if type(pid) is not int or pid <= 0:
        raise EvaluationRecoveryError("legacy orchestrator PID must be positive")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _candidate_policy_directory(
    root: Path, candidate: Mapping[str, Any]
) -> Path:
    expected = root / "policies" / f"policy_{candidate['policy_version']:06d}"
    candidate_path, _ = state._internal_path(
        root, candidate["manifest_path"], "candidate manifest"
    )
    if candidate_path.parent != expected or expected.is_symlink() or not expected.is_dir():
        raise EvaluationRecoveryError("candidate policy directory is non-canonical")
    return expected


def _allowed_previous_mode(mode: int) -> int:
    if (
        type(mode) is not int
        or not 0 <= mode <= 0o777
        or mode & 0o700 != 0o700
        or mode & 0o022
    ):
        raise EvaluationRecoveryError(
            "previous candidate mode must grant owner rwx and no group/other write"
        )
    return mode


def _parse_octal_mode(value: Any, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0[0-7]{3}", value) is None:
        raise EvaluationRecoveryError(f"{label} must be a four-digit octal mode")
    return int(value, 8)


def attest_promotion_hold(
    root: str | Path,
    *,
    legacy_orchestrator_pid: int,
    previous_mode_octal: str | None = None,
) -> dict[str, Any]:
    """Apply or attest a 0555 hold after the legacy coordinator has exited."""

    root = _root(root)
    current = state.load_state(root)
    if current["phase"] != "TRAINED" or current.get("candidate_policy") is None:
        raise EvaluationRecoveryError("promotion hold requires one TRAINED candidate")
    if _pid_exists(legacy_orchestrator_pid):
        raise EvaluationRecoveryError("legacy orchestrator PID is still alive")
    macro_round = current["open_macro_round"]
    candidate = promotion.load_candidate_manifest(
        root,
        current["candidate_policy"]["manifest_path"],
        expected_macro_round=macro_round,
        bind_state=True,
    )
    policy_directory = _candidate_policy_directory(root, candidate)
    observed_mode = policy_directory.stat().st_mode & 0o777
    if observed_mode == 0o555:
        if previous_mode_octal is None:
            raise EvaluationRecoveryError(
                "an already-held directory requires --previous-mode-octal"
            )
        previous_mode = _allowed_previous_mode(
            _parse_octal_mode(previous_mode_octal, "previous mode")
        )
    else:
        if previous_mode_octal is not None:
            raise EvaluationRecoveryError(
                "--previous-mode-octal is only valid for an already-held directory"
            )
        previous_mode = _allowed_previous_mode(observed_mode)
        os.chmod(policy_directory, 0o555)
        if policy_directory.stat().st_mode & 0o777 != 0o555:
            raise EvaluationRecoveryError("failed to apply exact candidate mode 0555")
    legacy_result = (
        state.round_directory(root, macro_round) / "evaluation" / "result.json"
    )
    legacy_root = legacy_result.parent
    if legacy_root.exists() and (legacy_root.is_symlink() or not legacy_root.is_dir()):
        raise EvaluationRecoveryError("legacy evaluation path is unsafe")
    if legacy_result.exists() and (
        legacy_result.is_symlink() or not legacy_result.is_file()
    ):
        raise EvaluationRecoveryError("legacy evaluator result is unsafe")
    legacy_presence = "present" if legacy_root.exists() else "absent"
    document = {
        "schema": HOLD_SCHEMA,
        "status": "legacy_coordinator_exited_candidate_held",
        "campaign_id": state.load_campaign(root)["campaign_id"],
        "campaign_sha256": current["campaign_sha256"],
        "macro_round": macro_round,
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "policy_directory": _relative(root, policy_directory, "policy directory"),
        "held_mode_octal": "0555",
        "previous_mode_octal": f"{previous_mode:04o}",
        "legacy_orchestrator_pid": legacy_orchestrator_pid,
        "legacy_evaluation_presence": legacy_presence,
        "legacy_result_path": (
            _relative(root, legacy_result, "legacy result")
            if legacy_result.exists()
            else None
        ),
        "legacy_result_sha256": (
            sha256_file(legacy_result) if legacy_result.exists() else None
        ),
    }
    output = root / "operator" / f"promotion_hold_macro_round_{macro_round:06d}.json"
    digest = publish_immutable_json(output, document)
    return {
        **document,
        "attestation_path": _relative(root, output, "promotion hold attestation"),
        "attestation_sha256": digest,
    }


def _load_hold(
    root: Path,
    *,
    path: str | Path,
    expected_sha256: str,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    require_held_mode: bool | None,
    verify_mutable_lineage: bool = True,
) -> dict[str, Any]:
    hold_path, relative = state._internal_path(
        root, path, "promotion hold attestation"
    )
    expected_path = (
        root
        / "operator"
        / f"promotion_hold_macro_round_{candidate['macro_round']:06d}.json"
    )
    if hold_path != expected_path or relative != hold_path.relative_to(root).as_posix():
        raise EvaluationRecoveryError("promotion hold attestation path is non-canonical")
    if sha256_file(hold_path) != _sha(
        expected_sha256, "promotion hold attestation SHA"
    ):
        raise EvaluationRecoveryError("promotion hold attestation SHA mismatch")
    hold = _exact(load_strict_json(hold_path), _HOLD_KEYS, "promotion hold attestation")
    policy_directory = _candidate_policy_directory(root, candidate)
    previous_mode = _allowed_previous_mode(
        _parse_octal_mode(hold["previous_mode_octal"], "previous mode")
    )
    expected = {
        "schema": HOLD_SCHEMA,
        "status": "legacy_coordinator_exited_candidate_held",
        "campaign_id": state.load_campaign(root)["campaign_id"],
        "campaign_sha256": current["campaign_sha256"],
        "macro_round": candidate["macro_round"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "policy_directory": _relative(root, policy_directory, "policy directory"),
        "held_mode_octal": "0555",
        "previous_mode_octal": f"{previous_mode:04o}",
    }
    if verify_mutable_lineage:
        legacy_result = (
            state.round_directory(root, candidate["macro_round"])
            / "evaluation"
            / "result.json"
        )
        legacy_root = legacy_result.parent
        legacy_presence = "present" if legacy_root.exists() else "absent"
        expected.update(
            {
                "legacy_evaluation_presence": legacy_presence,
                "legacy_result_path": (
                    _relative(root, legacy_result, "legacy result")
                    if legacy_result.exists()
                    else None
                ),
                "legacy_result_sha256": (
                    sha256_file(legacy_result) if legacy_result.exists() else None
                ),
            }
        )
    if any(hold.get(key) != value for key, value in expected.items()):
        raise EvaluationRecoveryError("promotion hold attestation lineage mismatch")
    pid = hold["legacy_orchestrator_pid"]
    if verify_mutable_lineage and _pid_exists(pid):
        raise EvaluationRecoveryError("attested legacy orchestrator PID is alive")
    mode = policy_directory.stat().st_mode & 0o777
    if require_held_mode is None:
        if mode not in {0o555, previous_mode}:
            raise EvaluationRecoveryError(
                "candidate policy directory mode is neither held nor its "
                "attested previous mode"
            )
    else:
        expected_mode = 0o555 if require_held_mode else previous_mode
        if mode != expected_mode:
            raise EvaluationRecoveryError(
                f"candidate policy directory mode is not {expected_mode:04o}"
            )
    return hold


def _repair_interrupted_hold(root: Path, plan: Mapping[str, Any]) -> None:
    """Reapply 0555 after a crash in either bounded writable window.

    The only accepted writable mode is the exact mode recorded before the
    promotion hold.  Re-holding is safe because the immutable attestation also
    proves the legacy coordinator PID remains dead and binds the exact staged
    candidate.  READY is the completed terminal state and intentionally keeps
    the restored pre-hold mode.
    """

    current = state.load_state(root)
    if current["phase"] == "READY":
        return
    if current["phase"] not in {"TRAINED", "EVALUATED"}:
        raise EvaluationRecoveryError(
            "interrupted hold repair requires TRAINED, EVALUATED, or READY"
        )
    candidate = promotion.load_candidate_manifest(
        root,
        plan["candidate_manifest_path"],
        expected_macro_round=plan["macro_round"],
        bind_state=True,
    )
    hold = _load_hold(
        root,
        path=plan["promotion_hold_attestation_path"],
        expected_sha256=plan["promotion_hold_attestation_sha256"],
        current=current,
        candidate=candidate,
        require_held_mode=None,
        verify_mutable_lineage=False,
    )
    policy_directory = _candidate_policy_directory(root, candidate)
    previous_mode = _allowed_previous_mode(
        _parse_octal_mode(hold["previous_mode_octal"], "previous mode")
    )
    if policy_directory.stat().st_mode & 0o777 == previous_mode:
        os.chmod(policy_directory, 0o555)
        if policy_directory.stat().st_mode & 0o777 != 0o555:
            raise EvaluationRecoveryError(
                "failed to reapply candidate hold after interrupted finalization"
            )


def _load_learner_recovery_receipt(
    root: Path,
    *,
    path: str | Path,
    expected_sha256: str,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path, relative = state._internal_path(
        root, path, "learner recovery receipt"
    )
    expected_path = (
        state.round_directory(root, candidate["macro_round"])
        / "learner_recovery"
        / "trained_receipt.json"
    )
    if receipt_path != expected_path or relative != receipt_path.relative_to(root).as_posix():
        raise EvaluationRecoveryError("learner recovery receipt path is non-canonical")
    if sha256_file(receipt_path) != _sha(
        expected_sha256, "learner recovery receipt SHA"
    ):
        raise EvaluationRecoveryError("learner recovery receipt SHA mismatch")
    receipt = _exact(
        load_strict_json(receipt_path),
        _LEARNER_RECEIPT_KEYS,
        "learner recovery receipt",
    )
    campaign = state.load_campaign(root)
    expected = {
        "schema": LEARNER_RECOVERY_RECEIPT_SCHEMA,
        "status": "TRAINED",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": current["campaign_sha256"],
        "macro_round": candidate["macro_round"],
        "candidate_manifest_path": candidate["manifest_path"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise EvaluationRecoveryError("learner recovery receipt lineage mismatch")
    if (
        type(receipt["ingested_revision"]) is not int
        or receipt["ingested_revision"] < 0
        or receipt["trained_revision"] != receipt["ingested_revision"] + 1
    ):
        raise EvaluationRecoveryError("learner recovery revisions are non-canonical")
    expected_revision_offset = {"TRAINED": 0, "EVALUATED": 1, "READY": 2}.get(
        current["phase"]
    )
    if (
        expected_revision_offset is None
        or current["revision"]
        != receipt["trained_revision"] + expected_revision_offset
    ):
        raise EvaluationRecoveryError(
            "coordinator revision differs from learner recovery lineage"
        )
    if (
        current["phase"] != "READY"
        and receipt["replay_snapshot_sha256"]
        != current["replay_snapshot_sha256"]
    ):
        raise EvaluationRecoveryError("replay snapshot differs from learner recovery")
    for key in (
        "replay_snapshot_file_sha256",
        "replay_snapshot_sha256",
        "sealed_source_inventory_sha256",
        "learner_config_sha256",
        "learner_result_sha256",
        "candidate_manifest_sha256",
        "candidate_checkpoint_sha256",
        "plan_sha256",
    ):
        _sha(receipt[key], f"learner recovery {key}")
    learner_result, learner_relative = state._internal_path(
        root, receipt["learner_result_path"], "learner recovery result"
    )
    if (
        learner_relative != receipt["learner_result_path"]
        or sha256_file(learner_result) != receipt["learner_result_sha256"]
    ):
        raise EvaluationRecoveryError("learner recovery result changed")
    snapshot_path = (
        root
        / "replay"
        / "snapshots"
        / f"through_macro_round_{candidate['macro_round']:06d}.json"
    )
    if sha256_file(snapshot_path) != receipt["replay_snapshot_file_sha256"]:
        raise EvaluationRecoveryError("learner recovery replay snapshot file changed")
    return receipt


def prepare_recovery(
    root: str | Path,
    *,
    repository_root: str | Path,
    gr00t_root: str | Path,
    hold_attestation_path: str | Path,
    expected_hold_attestation_sha256: str,
    scenario_cases_path: str | Path,
    thresholds_path: str | Path,
    evaluation_config_path: str | Path,
    renderer_runtime_sha256: str,
    expected_renderer_count: int,
    learner_recovery_receipt_path: str | Path,
    expected_learner_recovery_receipt_sha256: str,
    evaluation_source_inventory_sha256: str,
    anchor_pool_path: str | Path,
    anchor_count: int,
    anchor_index: int,
    anchor_inventory_sha256: str,
) -> dict[str, Any]:
    """Inventory the legacy result and publish an isolated hardened contract."""

    root = _root(root)
    current = state.load_state(root)
    if current["phase"] != "TRAINED" or current.get("candidate_policy") is None:
        raise EvaluationRecoveryError(
            "hardened recovery preparation requires one open TRAINED candidate"
        )
    macro_round = current["open_macro_round"]
    round_root = state.round_directory(root, macro_round)
    legacy_root = round_root / "evaluation"
    legacy_contract_path = legacy_root / "contract.json"
    inventory = _inventory_tree(root, legacy_root)
    inventory_sha = canonical_json_sha256(inventory)
    legacy_contract_sha = (
        sha256_file(legacy_contract_path) if legacy_contract_path.is_file() else None
    )
    legacy_result_path = legacy_root / "result.json"
    legacy_result_sha = (
        sha256_file(legacy_result_path) if legacy_result_path.is_file() else None
    )
    if legacy_result_sha is not None:
        if legacy_contract_sha is None:
            raise EvaluationRecoveryError(
                "legacy result exists without its evaluation contract"
            )
        legacy_contract = evaluation.load_contract(root, legacy_contract_path)
        _legacy_result(root, legacy_root=legacy_root, contract=legacy_contract)

    candidate = promotion.load_candidate_manifest(
        root,
        current["candidate_policy"]["manifest_path"],
        expected_macro_round=macro_round,
        bind_state=True,
    )
    hold = _load_hold(
        root,
        path=hold_attestation_path,
        expected_sha256=expected_hold_attestation_sha256,
        current=current,
        candidate=candidate,
        require_held_mode=True,
    )
    learner_receipt = _load_learner_recovery_receipt(
        root,
        path=learner_recovery_receipt_path,
        expected_sha256=expected_learner_recovery_receipt_sha256,
        current=current,
        candidate=candidate,
    )
    learner_inventory_sha = _sha(
        learner_receipt["sealed_source_inventory_sha256"],
        "learner source inventory SHA-256",
    )
    evaluation_inventory_sha = _sha(
        evaluation_source_inventory_sha256,
        "evaluation source inventory SHA-256",
    )
    anchor_path = Path(anchor_pool_path)
    if (
        not anchor_path.is_absolute()
        or any(part in {".", ".."} for part in anchor_path.parts)
    ):
        raise EvaluationRecoveryError(
            "anchor pool path must be normalized and absolute"
        )
    if type(anchor_count) is not int or anchor_count <= 0:
        raise EvaluationRecoveryError("anchor count must be positive")
    if type(anchor_index) is not int or anchor_index < 0:
        raise EvaluationRecoveryError("anchor index must be non-negative")
    if anchor_index >= anchor_count:
        raise EvaluationRecoveryError("anchor index must be smaller than anchor count")
    anchor_inventory_sha = _sha(
        anchor_inventory_sha256,
        "anchor inventory SHA-256",
    )
    repository = _require_source_root(repository_root, "repository root")
    gr00t = _require_source_root(gr00t_root, "GR00T root")
    evaluator_bundle = evaluation.evaluator_source_bundle(
        repository,
        gr00t,
        evaluation_inventory_sha,
    )
    evaluator_sha = canonical_json_sha256(evaluator_bundle)
    recovery_bundle = recovery_source_bundle(
        repository,
        gr00t,
        learner_source_inventory_sha256=learner_inventory_sha,
        evaluation_source_inventory_sha256=evaluation_inventory_sha,
    )
    recovery_sha = canonical_json_sha256(recovery_bundle)
    external_inputs: list[tuple[Path, str]] = []
    for raw, label in (
        (scenario_cases_path, "scenario cases"),
        (thresholds_path, "evaluation thresholds"),
        (evaluation_config_path, "evaluation config"),
    ):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise EvaluationRecoveryError(f"{label} path must be absolute")
        path = Path(os.path.abspath(path))
        if path.is_symlink() or not path.is_file():
            raise EvaluationRecoveryError(f"{label} is missing or unsafe")
        external_inputs.append((path, label))
    scenario_path, _ = external_inputs[0]
    threshold_path, _ = external_inputs[1]
    config_path, _ = external_inputs[2]
    scenario_cases = evaluation._validate_cases(load_strict_json(scenario_path))
    thresholds = evaluation._validate_thresholds(load_strict_json(threshold_path))
    if type(expected_renderer_count) is not int or expected_renderer_count < 1:
        raise EvaluationRecoveryError("expected renderer count must be positive")
    recovery_identity = canonical_json_sha256(
        {
            "campaign_sha256": current["campaign_sha256"],
            "macro_round": macro_round,
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "legacy_inventory_sha256": inventory_sha,
            "evaluator_source_bundle_sha256": evaluator_sha,
            "recovery_source_bundle_sha256": recovery_sha,
            "promotion_hold_attestation_sha256": expected_hold_attestation_sha256,
            "learner_recovery_receipt_sha256": (
                expected_learner_recovery_receipt_sha256
            ),
            "learner_source_inventory_sha256": learner_inventory_sha,
            "evaluation_source_inventory_sha256": evaluation_inventory_sha,
            "anchor_pool_path": str(anchor_path),
            "anchor_count": anchor_count,
            "anchor_index": anchor_index,
            "anchor_inventory_sha256": anchor_inventory_sha,
            "scenario_cases_sha256": sha256_file(scenario_path),
            "thresholds_sha256": sha256_file(threshold_path),
            "evaluation_config_sha256": sha256_file(config_path),
            "renderer_runtime_sha256": _sha(
                renderer_runtime_sha256, "renderer runtime SHA"
            ),
            "expected_renderer_count": expected_renderer_count,
        }
    )
    recovery_id = f"hardened-{recovery_identity[:20]}"
    recovery_root = round_root / "evaluation_recovery" / recovery_id
    evaluator_source_path = recovery_root / "evaluator_source_bundle.json"
    recovery_source_path = recovery_root / "recovery_source_bundle.json"
    inventory_path = recovery_root / "legacy_evaluation_inventory.json"
    publish_immutable_json(evaluator_source_path, evaluator_bundle)
    publish_immutable_json(recovery_source_path, recovery_bundle)
    publish_immutable_json(inventory_path, inventory)

    policies = {
        "parent": current["active_policy"],
        "candidate": candidate,
    }
    hardened_contract = evaluation.create_contract(
        root,
        candidate_manifest=candidate["manifest_path"],
        scenario_cases=scenario_cases,
        parent_runtime_policy_sha256=policies["parent"][
            "runtime_policy_sha256"
        ],
        candidate_runtime_policy_sha256=policies["candidate"][
            "runtime_policy_sha256"
        ],
        renderer_runtime_sha256=_sha(
            renderer_runtime_sha256, "renderer runtime SHA"
        ),
        evaluation_config_sha256=sha256_file(config_path),
        evaluator_code_sha256=evaluator_sha,
        expected_renderer_count=expected_renderer_count,
        anchor_pool_path=anchor_path,
        anchor_count=anchor_count,
        anchor_index=anchor_index,
        anchor_inventory_sha256=anchor_inventory_sha,
        output=recovery_root / "contract.json",
        thresholds=thresholds,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": state.load_campaign(root)["campaign_id"],
        "campaign_sha256": current["campaign_sha256"],
        "macro_round": macro_round,
        "candidate_manifest_path": candidate["manifest_path"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "legacy_evaluation_root": _relative(
            root, legacy_root, "legacy evaluation root"
        ),
        "legacy_inventory_path": _relative(root, inventory_path, "inventory"),
        "legacy_inventory_sha256": sha256_file(inventory_path),
        "legacy_contract_sha256": legacy_contract_sha,
        "legacy_result_sha256": legacy_result_sha,
        "promotion_hold_attestation_path": state._internal_path(
            root, hold_attestation_path, "promotion hold attestation"
        )[1],
        "promotion_hold_attestation_sha256": _sha(
            expected_hold_attestation_sha256,
            "promotion hold attestation SHA",
        ),
        "learner_recovery_receipt_path": state._internal_path(
            root, learner_recovery_receipt_path, "learner recovery receipt"
        )[1],
        "learner_recovery_receipt_sha256": _sha(
            expected_learner_recovery_receipt_sha256,
            "learner recovery receipt SHA",
        ),
        "learner_source_inventory_sha256": learner_inventory_sha,
        "evaluation_source_inventory_sha256": evaluation_inventory_sha,
        "anchor_pool_path": str(anchor_path),
        "anchor_count": anchor_count,
        "anchor_index": anchor_index,
        "anchor_inventory_sha256": anchor_inventory_sha,
        "recovery_id": recovery_id,
        "recovery_root": _relative(root, recovery_root, "recovery root"),
        "evaluator_contract_path": hardened_contract["contract_path"],
        "evaluator_contract_sha256": hardened_contract["contract_sha256"],
        "evaluator_result_path": _relative(
            root, recovery_root / "result.json", "hardened result"
        ),
        "evaluator_source_bundle_path": _relative(
            root, evaluator_source_path, "evaluator source bundle"
        ),
        "evaluator_source_bundle_sha256": sha256_file(evaluator_source_path),
        "recovery_source_bundle_path": _relative(
            root, recovery_source_path, "recovery source bundle"
        ),
        "recovery_source_bundle_sha256": sha256_file(recovery_source_path),
    }
    plan_path = recovery_root / "recovery_plan.json"
    plan_sha = publish_immutable_json(plan_path, plan)
    return {
        **plan,
        "plan_path": _relative(root, plan_path, "recovery plan"),
        "plan_sha256": plan_sha,
    }


def _load_plan(
    root: Path, path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path]:
    plan_path, relative = state._internal_path(root, path, "recovery plan")
    if sha256_file(plan_path) != _sha(expected_sha256, "expected recovery plan SHA"):
        raise EvaluationRecoveryError("recovery plan SHA mismatch")
    plan = _exact(load_strict_json(plan_path), _PLAN_KEYS, "recovery plan")
    if plan["schema"] != PLAN_SCHEMA:
        raise EvaluationRecoveryError("recovery plan schema mismatch")
    if _RECOVERY_ID_RE.fullmatch(str(plan["recovery_id"])) is None:
        raise EvaluationRecoveryError("recovery ID is non-canonical")
    expected_root = (
        root
        / "rounds"
        / f"macro_round_{plan['macro_round']:06d}"
        / "evaluation_recovery"
        / plan["recovery_id"]
    )
    if plan_path != expected_root / "recovery_plan.json":
        raise EvaluationRecoveryError("recovery plan path is non-canonical")
    if plan["recovery_root"] != _relative(root, expected_root, "recovery root"):
        raise EvaluationRecoveryError("recovery root differs from plan path")
    if relative != plan_path.relative_to(root).as_posix():
        raise EvaluationRecoveryError("recovery plan relative path is ambiguous")
    return plan, plan_path


def _verify_plan_evidence(
    root: Path,
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
    gr00t_root: str | Path,
) -> dict[str, Any]:
    campaign = state.load_campaign(root)
    current = state.load_state(root)
    expected_round = (
        current["next_macro_round"] - 1
        if current["phase"] == "READY"
        else current["open_macro_round"]
    )
    expected = {
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": canonical_json_sha256(campaign),
        "macro_round": expected_round,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise EvaluationRecoveryError("recovery plan campaign/round lineage mismatch")
    for path_key, sha_key, label in (
        ("legacy_inventory_path", "legacy_inventory_sha256", "legacy inventory"),
        (
            "evaluator_source_bundle_path",
            "evaluator_source_bundle_sha256",
            "evaluator source bundle",
        ),
        (
            "recovery_source_bundle_path",
            "recovery_source_bundle_sha256",
            "recovery source bundle",
        ),
        (
            "evaluator_contract_path",
            "evaluator_contract_sha256",
            "hardened evaluator contract",
        ),
        (
            "promotion_hold_attestation_path",
            "promotion_hold_attestation_sha256",
            "promotion hold attestation",
        ),
        (
            "learner_recovery_receipt_path",
            "learner_recovery_receipt_sha256",
            "learner recovery receipt",
        ),
    ):
        artifact, relative = state._internal_path(root, plan[path_key], label)
        if relative != plan[path_key] or sha256_file(artifact) != _sha(
            plan[sha_key], f"{label} SHA"
        ):
            raise EvaluationRecoveryError(f"{label} changed")
    hardened_contract = evaluation.load_contract(
        root,
        root / plan["evaluator_contract_path"],
        bind_state=current["phase"] != "READY",
    )
    expected_anchor_pool = {
        "path": plan["anchor_pool_path"],
        "anchor_count": plan["anchor_count"],
        "anchor_index": plan["anchor_index"],
        "inventory_sha256": plan["anchor_inventory_sha256"],
    }
    if hardened_contract.get("anchor_pool") != expected_anchor_pool:
        raise EvaluationRecoveryError(
            "hardened evaluator anchor contract differs from recovery plan"
        )
    inventory_path = root / plan["legacy_inventory_path"]
    expected_inventory = load_strict_json(inventory_path)
    legacy_root = root / plan["legacy_evaluation_root"]
    if _inventory_tree(root, legacy_root) != expected_inventory:
        raise EvaluationRecoveryError("legacy evaluation changed after inventory")
    for name, expected_sha, label in (
        ("contract.json", plan["legacy_contract_sha256"], "legacy evaluation contract"),
        ("result.json", plan["legacy_result_sha256"], "legacy evaluator result"),
    ):
        artifact = legacy_root / name
        if expected_sha is None:
            if artifact.exists():
                raise EvaluationRecoveryError(f"{label} appeared after preparation")
        elif sha256_file(artifact) != _sha(expected_sha, f"{label} SHA"):
            raise EvaluationRecoveryError(f"{label} changed")
    candidate = promotion.load_candidate_manifest(
        root,
        plan["candidate_manifest_path"],
        expected_macro_round=plan["macro_round"],
        bind_state=current["phase"] != "READY",
    )
    _load_hold(
        root,
        path=plan["promotion_hold_attestation_path"],
        expected_sha256=plan["promotion_hold_attestation_sha256"],
        current=current,
        candidate=candidate,
        require_held_mode=current["phase"] in {"TRAINED", "EVALUATED"},
    )
    learner_receipt = _load_learner_recovery_receipt(
        root,
        path=plan["learner_recovery_receipt_path"],
        expected_sha256=plan["learner_recovery_receipt_sha256"],
        current=current,
        candidate=candidate,
    )
    learner_inventory_sha = _sha(
        plan["learner_source_inventory_sha256"],
        "planned learner source inventory SHA-256",
    )
    if learner_receipt["sealed_source_inventory_sha256"] != learner_inventory_sha:
        raise EvaluationRecoveryError(
            "learner source inventory differs from recovery receipt"
        )
    evaluation_inventory_sha = _sha(
        plan["evaluation_source_inventory_sha256"],
        "planned evaluation source inventory SHA-256",
    )
    actual_evaluator = evaluation.evaluator_source_bundle(
        repository_root,
        gr00t_root,
        evaluation_inventory_sha,
    )
    actual_recovery = recovery_source_bundle(
        repository_root,
        gr00t_root,
        learner_source_inventory_sha256=learner_inventory_sha,
        evaluation_source_inventory_sha256=evaluation_inventory_sha,
    )
    evaluator_path = root / plan["evaluator_source_bundle_path"]
    recovery_path = root / plan["recovery_source_bundle_path"]
    if load_strict_json(evaluator_path) != actual_evaluator:
        raise EvaluationRecoveryError("evaluator source bytes changed after preparation")
    if load_strict_json(recovery_path) != actual_recovery:
        raise EvaluationRecoveryError("recovery source bytes changed after preparation")
    return current


def finalize_recovery(
    root: str | Path,
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    repository_root: str | Path,
    gr00t_root: str | Path,
) -> dict[str, Any]:
    """Verify hardened evidence, publish evaluation, and atomically promote."""

    root = _root(root)
    plan, _ = _load_plan(root, plan_path, expected_plan_sha256)
    _repair_interrupted_hold(root, plan)
    current = _verify_plan_evidence(
        root,
        plan,
        repository_root=repository_root,
        gr00t_root=gr00t_root,
    )
    if current["phase"] not in {"TRAINED", "EVALUATED", "READY"}:
        raise EvaluationRecoveryError(
            "recovery finalization requires TRAINED, EVALUATED, or its READY retry"
    )
    candidate = promotion.load_candidate_manifest(
        root,
        plan["candidate_manifest_path"],
        expected_macro_round=plan["macro_round"],
        bind_state=current["phase"] != "READY",
    )
    if candidate["manifest_sha256"] != plan["candidate_manifest_sha256"]:
        raise EvaluationRecoveryError("staged candidate differs from recovery plan")
    result_path = root / plan["evaluator_result_path"]
    results_dir = result_path.parent / "results"
    hold = load_strict_json(root / plan["promotion_hold_attestation_path"])
    previous_mode = _allowed_previous_mode(
        _parse_octal_mode(hold["previous_mode_octal"], "previous mode")
    )
    policy_directory = _candidate_policy_directory(root, candidate)

    if current["phase"] != "READY":
        verified = evaluation.verify_evaluator_result_evidence(
            root,
            contract_path=root / plan["evaluator_contract_path"],
            results_dir=results_dir,
            result_path=result_path,
        )
        if verified["passed"] is not True:
            raise EvaluationRecoveryError(
                "hardened paired evaluation failed; candidate was not published"
            )
        if current["phase"] == "TRAINED":
            os.chmod(policy_directory, previous_mode)
            if policy_directory.stat().st_mode & 0o777 != previous_mode:
                os.chmod(policy_directory, 0o555)
                raise EvaluationRecoveryError(
                    "failed to restore exact pre-hold candidate mode"
                )
            try:
                evidence = promotion.write_evaluation(
                    root,
                    candidate_manifest=candidate["manifest_path"],
                    passed=True,
                    metrics=verified["metrics"],
                    evaluator_config_sha256=verified["evaluator_config_sha256"],
                    evidence_bundle_path=verified["evidence_bundle_path"],
                    evidence_bundle_sha256=verified["evidence_bundle_sha256"],
                )
            except BaseException:
                if state.load_state(root)["phase"] == "TRAINED":
                    os.chmod(policy_directory, 0o555)
                raise
            os.chmod(policy_directory, 0o555)
            if policy_directory.stat().st_mode & 0o777 != 0o555:
                raise EvaluationRecoveryError(
                    "failed to reapply candidate hold before evaluation CAS"
                )
            latest = state.load_state(root)
            state.transition(
                root,
                expected_phase="TRAINED",
                next_phase="EVALUATED",
                macro_round=plan["macro_round"],
                evidence_path=evidence["manifest_path"],
                evidence_sha256=evidence["evidence_sha256"],
                expected_revision=latest["revision"],
            )
        else:
            evidence = promotion.load_evaluation_manifest(
                root,
                root
                / "policies"
                / f"policy_{candidate['policy_version']:06d}"
                / "evaluation.json",
                candidate=candidate,
            )
    else:
        evidence = promotion.load_evaluation_manifest(
            root,
            root
            / "policies"
            / f"policy_{candidate['policy_version']:06d}"
            / "evaluation.json",
            candidate=candidate,
        )
    if (
        evidence["passed"] is not True
        or evidence["evidence_bundle_path"]
        != f"{plan['recovery_root']}/results/evaluation_bundle.json"
    ):
        raise EvaluationRecoveryError(
            "published evaluation is not the planned hardened evidence"
        )
    before_promotion = state.load_state(root)
    if before_promotion["phase"] == "EVALUATED":
        os.chmod(policy_directory, previous_mode)
        if policy_directory.stat().st_mode & 0o777 != previous_mode:
            os.chmod(policy_directory, 0o555)
            raise EvaluationRecoveryError(
                "failed to restore exact pre-hold mode for promotion CAS"
            )
    try:
        promoted = promotion.promote_candidate(
            root,
            macro_round=plan["macro_round"],
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evidence["manifest_path"],
        )
    except BaseException:
        if state.load_state(root)["phase"] == "EVALUATED":
            os.chmod(policy_directory, 0o555)
        raise
    final_state = state.load_state(root)
    if (
        final_state["phase"] != "READY"
        or final_state["active_policy"]["version"] != candidate["policy_version"]
        or final_state["active_policy"]["checkpoint_sha256"]
        != candidate["checkpoint_sha256"]
    ):
        raise EvaluationRecoveryError("hardened recovery did not atomically promote")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "macro_round": plan["macro_round"],
        "recovery_id": plan["recovery_id"],
        "plan_sha256": _sha(expected_plan_sha256, "recovery plan SHA"),
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "evaluation_sha256": evidence["evidence_sha256"],
        "promotion_sha256": promoted["promotion_sha256"],
        "active_policy_version": candidate["policy_version"],
        "active_checkpoint_sha256": candidate["checkpoint_sha256"],
    }
    completion_path = root / plan["recovery_root"] / "recovery_completion.json"
    completion_sha = publish_immutable_json(completion_path, completion)
    return {
        **completion,
        "completion_path": _relative(root, completion_path, "recovery completion"),
        "completion_sha256": completion_sha,
    }


def run_trusted_recovery(
    root: str | Path,
    *,
    repository_root: str | Path,
    gr00t_root: str | Path,
    hold_attestation_path: str | Path,
    expected_hold_attestation_sha256: str,
    scenario_cases_path: str | Path,
    thresholds_path: str | Path,
    evaluation_config_path: str | Path,
    renderer_runtime_sha256: str,
    expected_renderer_count: int,
    learner_recovery_receipt_path: str | Path,
    expected_learner_recovery_receipt_sha256: str,
    evaluation_source_inventory_sha256: str,
    anchor_pool_path: str | Path,
    anchor_count: int,
    anchor_index: int,
    anchor_inventory_sha256: str,
    environment: Mapping[str, str] | None = None,
    hook_timeout_seconds: int = RECOVERY_EVALUATION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one held-candidate recovery through a pre-latched evaluator hook.

    Hook verification and the sealed anonymous-memory snapshot deliberately
    happen before :func:`prepare_recovery`, the first operation that may
    publish a recovery artifact.  The descriptor returned by the trusted
    coordinator verifier is the only hook body executed and remains open until
    evaluation exits.  Finalization is unreachable after a failed hook.
    """

    campaign_root = _root(root)
    repository = _require_source_root(repository_root, "repository root")
    gr00t = _require_source_root(gr00t_root, "GR00T root")
    if gr00t != repository / "GR00T-VisualSim2Real":
        raise EvaluationRecoveryError(
            "GR00T root must be derived from the sealed evaluation source"
        )
    inventory_sha = _sha(
        evaluation_source_inventory_sha256,
        "evaluation source inventory SHA-256",
    )
    if (
        type(hook_timeout_seconds) is not int
        or hook_timeout_seconds < 1
    ):
        raise EvaluationRecoveryError("evaluation hook timeout must be positive")

    child_environment = dict(os.environ if environment is None else environment)
    child_environment.pop("ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH", None)
    child_environment["ROUND_DAGGER_EVALUATION_SOURCE_ROOT"] = str(repository)
    child_environment[
        "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256"
    ] = inventory_sha
    hook_path = repository / orchestration._EVALUATION_HOOK_RELATIVE  # noqa: SLF001
    hook = orchestration.HookSpec((str(hook_path),), hook_timeout_seconds)
    descriptor: int | None = None
    try:
        # This is intentionally first: no contract, plan, result, or state
        # publication is permitted until the exact hook bytes are immutable.
        descriptor, argv = orchestration._verified_evaluation_hook(  # noqa: SLF001
            hook,
            child_environment,
        )
        prepared = prepare_recovery(
            campaign_root,
            repository_root=repository,
            gr00t_root=gr00t,
            hold_attestation_path=hold_attestation_path,
            expected_hold_attestation_sha256=expected_hold_attestation_sha256,
            scenario_cases_path=scenario_cases_path,
            thresholds_path=thresholds_path,
            evaluation_config_path=evaluation_config_path,
            renderer_runtime_sha256=renderer_runtime_sha256,
            expected_renderer_count=expected_renderer_count,
            learner_recovery_receipt_path=learner_recovery_receipt_path,
            expected_learner_recovery_receipt_sha256=(
                expected_learner_recovery_receipt_sha256
            ),
            evaluation_source_inventory_sha256=inventory_sha,
            anchor_pool_path=anchor_pool_path,
            anchor_count=anchor_count,
            anchor_index=anchor_index,
            anchor_inventory_sha256=anchor_inventory_sha256,
        )

        current = state.load_state(campaign_root)
        campaign = state.load_campaign(campaign_root)
        candidate = promotion.load_candidate_manifest(
            campaign_root,
            prepared["candidate_manifest_path"],
            expected_macro_round=prepared["macro_round"],
            bind_state=True,
        )
        active = current["active_policy"]
        result_path, result_relative = state._internal_path(
            campaign_root,
            prepared["evaluator_result_path"],
            "hardened evaluator result",
        )
        if result_relative != prepared["evaluator_result_path"]:
            raise EvaluationRecoveryError(
                "hardened evaluator result path is non-canonical"
            )
        scenario_path = Path(os.path.abspath(scenario_cases_path))
        threshold_path = Path(os.path.abspath(thresholds_path))
        config_path = Path(os.path.abspath(evaluation_config_path))
        child_environment.update(
            {
                "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH": "1",
                "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign_root),
                "ROUND_DAGGER_CAMPAIGN_ID": campaign["campaign_id"],
                "ROUND_DAGGER_MACRO_ROUND": str(prepared["macro_round"]),
                "ROUND_DAGGER_POLICY_CHECKPOINT": str(
                    campaign_root / active["checkpoint_path"]
                ),
                "ROUND_DAGGER_POLICY_SHA256": active["checkpoint_sha256"],
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": active[
                    "runtime_policy_sha256"
                ],
                "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                    campaign_root / candidate["manifest_path"]
                ),
                "ROUND_DAGGER_CANDIDATE_SHA256": candidate["manifest_sha256"],
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": sha256_file(config_path),
                "ROUND_DAGGER_EVALUATOR_RESULT": str(result_path),
                "ROUND_DAGGER_EVALUATION_RECOVERY_ID": prepared["recovery_id"],
                "ROUND_DAGGER_EVALUATION_CONFIG": str(config_path),
                "ROUND_DAGGER_EVALUATION_SCENARIOS": str(scenario_path),
                "ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256": sha256_file(
                    scenario_path
                ),
                "ROUND_DAGGER_EVALUATION_THRESHOLDS": str(threshold_path),
                "ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256": sha256_file(
                    threshold_path
                ),
                "ROUND_DAGGER_RENDERER_RUNTIME_SHA256": _sha(
                    renderer_runtime_sha256, "renderer runtime SHA"
                ),
                "ROUND_DAGGER_EXPECTED_RENDERERS": str(expected_renderer_count),
                "ANCHORS": prepared["anchor_pool_path"],
                "ROUND_DAGGER_MIN_ANCHORS": str(prepared["anchor_count"]),
                "ROUND_DAGGER_ANCHOR_INDEX": str(prepared["anchor_index"]),
                "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": prepared[
                    "anchor_inventory_sha256"
                ],
            }
        )
        try:
            completed = subprocess.run(
                argv,
                cwd=repository,
                env=child_environment,
                check=False,
                timeout=hook_timeout_seconds,
                pass_fds=(descriptor,),
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationRecoveryError(
                "hardened paired evaluation hook timed out"
            ) from exc
        except OSError as exc:
            raise EvaluationRecoveryError(
                f"hardened paired evaluation hook could not start ({type(exc).__name__})"
            ) from exc
        if completed.returncode != 0:
            raise EvaluationRecoveryError(
                "hardened paired evaluation hook failed with exit status "
                f"{completed.returncode}"
            )
        return finalize_recovery(
            campaign_root,
            plan_path=prepared["plan_path"],
            expected_plan_sha256=prepared["plan_sha256"],
            repository_root=repository,
            gr00t_root=gr00t,
        )
    except orchestration.OrchestrationError as exc:
        raise EvaluationRecoveryError(
            "evaluation hook failed trusted coordinator verification"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _required_environment(
    environment: Mapping[str, str], name: str
) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise EvaluationRecoveryError(f"missing {name}")
    return value


def _environment_integer(
    environment: Mapping[str, str], name: str, *, minimum: int
) -> int:
    value = _required_environment(environment, name)
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise EvaluationRecoveryError(f"{name} must be a canonical integer")
    parsed = int(value)
    if parsed < minimum:
        raise EvaluationRecoveryError(f"{name} must be at least {minimum}")
    return parsed


def run_recovery_from_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Parse the operator handoff and enter the trusted recovery runner."""

    values = dict(os.environ if environment is None else environment)
    if values.get("ROUND_DAGGER_RECOVERY_OPERATOR_ACK") != (
        "legacy-orchestrator-stopped"
    ):
        raise EvaluationRecoveryError(
            "ROUND_DAGGER_RECOVERY_OPERATOR_ACK must be "
            "legacy-orchestrator-stopped"
        )
    repository = Path(
        _required_environment(values, "ROUND_DAGGER_EVALUATION_SOURCE_ROOT")
    )
    return run_trusted_recovery(
        _required_environment(values, "ROUND_DAGGER_CAMPAIGN_ROOT"),
        repository_root=repository,
        gr00t_root=repository / "GR00T-VisualSim2Real",
        hold_attestation_path=_required_environment(
            values, "ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION"
        ),
        expected_hold_attestation_sha256=_required_environment(
            values, "ROUND_DAGGER_PROMOTION_HOLD_ATTESTATION_SHA256"
        ),
        scenario_cases_path=_required_environment(
            values, "ROUND_DAGGER_EVALUATION_SCENARIOS"
        ),
        thresholds_path=_required_environment(
            values, "ROUND_DAGGER_EVALUATION_THRESHOLDS"
        ),
        evaluation_config_path=_required_environment(
            values, "ROUND_DAGGER_EVALUATION_CONFIG"
        ),
        renderer_runtime_sha256=_required_environment(
            values, "ROUND_DAGGER_RENDERER_RUNTIME_SHA256"
        ),
        expected_renderer_count=_environment_integer(
            values, "ROUND_DAGGER_EXPECTED_RENDERERS", minimum=1
        ),
        learner_recovery_receipt_path=_required_environment(
            values, "ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT"
        ),
        expected_learner_recovery_receipt_sha256=_required_environment(
            values, "ROUND_DAGGER_LEARNER_RECOVERY_RECEIPT_SHA256"
        ),
        evaluation_source_inventory_sha256=_required_environment(
            values, "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256"
        ),
        anchor_pool_path=_required_environment(values, "ANCHORS"),
        anchor_count=_environment_integer(
            values, "ROUND_DAGGER_MIN_ANCHORS", minimum=1
        ),
        anchor_index=_environment_integer(
            values, "ROUND_DAGGER_ANCHOR_INDEX", minimum=0
        ),
        anchor_inventory_sha256=_required_environment(
            values, "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256"
        ),
        environment=values,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hold = subparsers.add_parser("attest-hold")
    hold.add_argument("--root", required=True)
    hold.add_argument("--legacy-orchestrator-pid", required=True, type=int)
    hold.add_argument("--previous-mode-octal")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--repository-root", required=True)
    prepare.add_argument("--gr00t-root", required=True)
    prepare.add_argument("--hold-attestation", required=True)
    prepare.add_argument("--expected-hold-attestation-sha256", required=True)
    prepare.add_argument("--scenario-cases", required=True)
    prepare.add_argument("--thresholds", required=True)
    prepare.add_argument("--evaluation-config", required=True)
    prepare.add_argument("--renderer-runtime-sha256", required=True)
    prepare.add_argument("--expected-renderer-count", required=True, type=int)
    prepare.add_argument("--learner-recovery-receipt", required=True)
    prepare.add_argument(
        "--expected-learner-recovery-receipt-sha256", required=True
    )
    prepare.add_argument(
        "--evaluation-source-inventory-sha256", required=True
    )
    prepare.add_argument("--anchor-pool-path", required=True)
    prepare.add_argument("--anchor-count", type=int, required=True)
    prepare.add_argument("--anchor-index", type=int, required=True)
    prepare.add_argument("--anchor-inventory-sha256", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--root", required=True)
    finalize.add_argument("--plan", required=True)
    finalize.add_argument("--expected-plan-sha256", required=True)
    finalize.add_argument("--repository-root", required=True)
    finalize.add_argument("--gr00t-root", required=True)
    subparsers.add_parser("run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "attest-hold":
            result = attest_promotion_hold(
                args.root,
                legacy_orchestrator_pid=args.legacy_orchestrator_pid,
                previous_mode_octal=args.previous_mode_octal,
            )
        elif args.command == "prepare":
            result = prepare_recovery(
                args.root,
                repository_root=args.repository_root,
                gr00t_root=args.gr00t_root,
                hold_attestation_path=args.hold_attestation,
                expected_hold_attestation_sha256=(
                    args.expected_hold_attestation_sha256
                ),
                scenario_cases_path=args.scenario_cases,
                thresholds_path=args.thresholds,
                evaluation_config_path=args.evaluation_config,
                renderer_runtime_sha256=args.renderer_runtime_sha256,
                expected_renderer_count=args.expected_renderer_count,
                learner_recovery_receipt_path=args.learner_recovery_receipt,
                expected_learner_recovery_receipt_sha256=(
                    args.expected_learner_recovery_receipt_sha256
                ),
                evaluation_source_inventory_sha256=(
                    args.evaluation_source_inventory_sha256
                ),
                anchor_pool_path=args.anchor_pool_path,
                anchor_count=args.anchor_count,
                anchor_index=args.anchor_index,
                anchor_inventory_sha256=args.anchor_inventory_sha256,
            )
        elif args.command == "finalize":
            result = finalize_recovery(
                args.root,
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                repository_root=args.repository_root,
                gr00t_root=args.gr00t_root,
            )
        else:
            result = run_recovery_from_environment()
    except (
        EvaluationRecoveryError,
        evaluation.EvaluationError,
        promotion.PromotionError,
        ArtifactError,
        OSError,
        ValueError,
    ) as exc:
        print(f"hardened evaluation recovery failed: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
