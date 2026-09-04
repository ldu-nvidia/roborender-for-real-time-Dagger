"""Audited acceptance of one returned Lepton learner transaction.

The remote completion marker is necessary, but it is deliberately not the
coordinator's commit record: it lives outside the campaign and it describes a
remote archive.  This module revalidates that entire boundary after return and
publishes one create-only, campaign-local receipt.  The receipt is the sole
Lepton-specific prerequisite for ``INGESTED -> TRAINED``.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from round_dagger import artifacts, replay
from round_dagger.lepton_job import (
    LeptonJobError,
    load_contract,
    load_request,
    validate_completion_marker,
)


SCHEMA = "fast_roborender.round_dagger.learner_acceptance.v1"
BACKEND = "lepton_job_v1"
REMOTE_MARKER_BASENAME = "lepton_learner_success.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_EXPECTED_ENVIRONMENT = {
    "ROUND_DAGGER_CAMPAIGN_ID",
    "ROUND_DAGGER_MACRO_ROUND",
    "ROUND_DAGGER_LEARNER_IDENTITY_SHA256",
    "ROUND_DAGGER_LEARNER_CODE_SHA256",
    "ROUND_DAGGER_LEARNER_CONFIG_SHA256",
    "ROUND_DAGGER_LEARNER_CONTRACT_SHA256",
    "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256",
    "ROUND_DAGGER_REMOTE_CONTAINER_SHA256",
    "ROUND_DAGGER_RUNTIME_POLICY_SHA256",
    "ROUND_DAGGER_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256",
    "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256",
    "ROUND_DAGGER_POLICY_CHECKPOINT",
    "ROUND_DAGGER_REMOTE_HANDOFF_ROOT",
}
_KEYS = {
    "schema",
    "status",
    "backend",
    "campaign_id",
    "macro_round",
    "learner_identity_sha256",
    "backend_contract_path",
    "backend_contract_sha256",
    "request_path",
    "request_sha256",
    "completion_marker_path",
    "completion_marker_sha256",
    "completion_checksum_path",
    "completion_checksum_sha256",
    "completion_checksum_source_basename",
    "returned_output_path",
    "returned_archive_path",
    "returned_archive_sha256",
    "returned_inventory_path",
    "returned_inventory_sha256",
    "returned_manifest_path",
    "returned_manifest_sha256",
    "returned_tree_sha256",
    "learner_result_path",
    "learner_result_sha256",
    "candidate_checkpoint_path",
    "candidate_checkpoint_sha256",
    "candidate_runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_manifest_path",
    "policy_manifest_sha256",
    "completion_attestation_path",
    "completion_attestation_sha256",
    "wandb_readback_path",
    "wandb_readback_sha256",
    "replay_snapshot_path",
    "replay_snapshot_file_sha256",
    "replay_snapshot_sha256",
    "parent_checkpoint_path",
    "parent_checkpoint_sha256",
    "parent_runtime_policy_sha256",
    "learner_code_sha256",
    "learner_config_sha256",
    "learner_contract_sha256",
    "runtime_container_sha256",
    "content_sha256",
}


class LearnerAcceptanceError(RuntimeError):
    """Returned learner evidence is absent, conflicting, or incomplete."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LearnerAcceptanceError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise LearnerAcceptanceError(f"{label} is malformed")
    return value


def _absolute(path: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(path)
    candidate = Path(raw)
    if (
        not candidate.is_absolute()
        or os.path.abspath(raw) != raw
        or "\x00" in raw
    ):
        raise LearnerAcceptanceError(f"{label} must be canonical and absolute")
    return candidate


def _regular(path: Path, label: str, *, immutable: bool = False) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LearnerAcceptanceError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise LearnerAcceptanceError(f"{label} must be a regular file")
    if immutable and info.st_mode & 0o222:
        raise LearnerAcceptanceError(f"{label} must be read-only")
    return info


def _directory(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LearnerAcceptanceError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LearnerAcceptanceError(f"{label} must be a real directory")
    return path


def _campaign_relative(campaign_root: Path, path: Path, label: str) -> str:
    try:
        relative = path.relative_to(campaign_root)
    except ValueError as exc:
        raise LearnerAcceptanceError(f"{label} lies outside the campaign") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise LearnerAcceptanceError(f"{label} has an unsafe campaign path")
    return relative.as_posix()


def acceptance_path(campaign_root: Path, macro_round: int) -> Path:
    if type(macro_round) is not int or macro_round < 0:
        raise LearnerAcceptanceError("macro round must be a non-negative integer")
    root = _absolute(campaign_root, "campaign root")
    return (
        root
        / "rounds"
        / f"macro_round_{macro_round:06d}"
        / f"learner_acceptance_round_{macro_round:06d}.json"
    )


def _expected_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise LearnerAcceptanceError("expected learner environment is malformed")
    missing = _EXPECTED_ENVIRONMENT - set(value)
    if missing:
        raise LearnerAcceptanceError(
            f"expected learner environment is incomplete: {sorted(missing)}"
        )
    result = {key: value[key] for key in _EXPECTED_ENVIRONMENT}
    for key, raw in result.items():
        if not isinstance(raw, str) or not raw or any(c in raw for c in "\x00\r\n"):
            raise LearnerAcceptanceError(f"expected environment {key} is unsafe")
    return result


def _collect(
    *,
    campaign_root: Path,
    macro_round: int,
    backend_contract_path: Path,
    request_path: Path,
    marker_path: Path,
    checksum_path: Path,
    return_bundle_root: Path,
    learner_output: Path,
    expected_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Revalidate every remote-to-local learner boundary and collect identities."""

    env = _expected_environment(expected_environment)
    root = _directory(_absolute(campaign_root, "campaign root"), "campaign root")
    if env["ROUND_DAGGER_CAMPAIGN_ID"] != _token(
        env["ROUND_DAGGER_CAMPAIGN_ID"], "campaign ID"
    ):
        raise LearnerAcceptanceError("campaign ID is malformed")
    if not env["ROUND_DAGGER_MACRO_ROUND"].isdigit() or int(
        env["ROUND_DAGGER_MACRO_ROUND"]
    ) != macro_round:
        raise LearnerAcceptanceError("macro-round environment mismatch")
    for name in (
        "ROUND_DAGGER_LEARNER_IDENTITY_SHA256",
        "ROUND_DAGGER_LEARNER_CODE_SHA256",
        "ROUND_DAGGER_LEARNER_CONFIG_SHA256",
        "ROUND_DAGGER_LEARNER_CONTRACT_SHA256",
        "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256",
        "ROUND_DAGGER_REMOTE_CONTAINER_SHA256",
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256",
        "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256",
        "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256",
    ):
        _sha(env[name], name)

    contract_path = _absolute(backend_contract_path, "backend contract")
    request_file = _absolute(request_path, "learner request")
    marker = _absolute(marker_path, "completion marker")
    checksum = _absolute(checksum_path, "completion checksum")
    bundle_root = _directory(
        _absolute(return_bundle_root, "returned bundle root"),
        "returned bundle root",
    )
    output = _directory(_absolute(learner_output, "learner output"), "learner output")
    expected_handoff = (
        _absolute(env["ROUND_DAGGER_REMOTE_HANDOFF_ROOT"], "remote handoff root")
        / "lepton_job_v1"
        / env["ROUND_DAGGER_CAMPAIGN_ID"]
        / f"macro_round_{macro_round:06d}"
        / f"identity_{env['ROUND_DAGGER_LEARNER_IDENTITY_SHA256']}"
    )
    if (
        request_file != expected_handoff / "lepton_learner_request.json"
        or marker != expected_handoff / "remote_learner_success.json"
        or checksum != expected_handoff / "remote_learner_success.json.sha256"
        or bundle_root != expected_handoff / "returned_bundle"
    ):
        raise LearnerAcceptanceError(
            "returned evidence is outside the canonical identity-scoped handoff"
        )
    expected_output = (
        root / "rounds" / f"macro_round_{macro_round:06d}" / "learner"
    )
    if output != expected_output:
        raise LearnerAcceptanceError("learner output is not coordinator-canonical")

    for path, label in (
        (contract_path, "backend contract"),
        (request_file, "learner request"),
        (marker, "completion marker"),
        (checksum, "completion checksum"),
    ):
        _regular(path, label, immutable=True)
    try:
        contract = load_contract(contract_path, immutable=True)
        request = load_request(request_file, contract=contract, immutable=True)
    except LeptonJobError as exc:
        raise LearnerAcceptanceError(f"Lepton request identity is invalid: {exc}") from exc
    if (
        request.campaign_id != env["ROUND_DAGGER_CAMPAIGN_ID"]
        or request.macro_round != macro_round
        or request.identity_sha256
        != env["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
        or request.environment["ROUND_DAGGER_LEARNER_CODE_SHA256"]
        != env["ROUND_DAGGER_LEARNER_CODE_SHA256"]
        or request.environment["ROUND_DAGGER_LEARNER_CONFIG_SHA256"]
        != env["ROUND_DAGGER_LEARNER_CONFIG_SHA256"]
        or request.environment["ROUND_DAGGER_LEARNER_CONTRACT_SHA256"]
        != env["ROUND_DAGGER_LEARNER_CONTRACT_SHA256"]
        or request.environment["ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256"]
        != env["ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256"]
        or request.environment["ROUND_DAGGER_RUNTIME_POLICY_SHA256"]
        != env["ROUND_DAGGER_RUNTIME_POLICY_SHA256"]
        or request.environment["ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"]
        != env["ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"]
        or request.environment["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"]
        != env["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"]
        or contract.image_digest != env["ROUND_DAGGER_REMOTE_CONTAINER_SHA256"]
    ):
        raise LearnerAcceptanceError("Lepton request differs from coordinator lineage")

    marker_sha = artifacts.sha256_file(marker)
    expected_checksum = f"{marker_sha}  {REMOTE_MARKER_BASENAME}\n".encode("ascii")
    try:
        checksum_payload = checksum.read_bytes()
    except OSError as exc:
        raise LearnerAcceptanceError("cannot read completion checksum") from exc
    if checksum_payload != expected_checksum:
        raise LearnerAcceptanceError(
            "completion checksum does not bind the original remote marker basename"
        )
    try:
        completion_marker = validate_completion_marker(
            marker, request=request, contract=contract
        )
    except LeptonJobError as exc:
        raise LearnerAcceptanceError(f"completion marker is invalid: {exc}") from exc

    # Revalidate the downloaded immutable archive triple and the extracted tree.
    from scripts.round_dagger import lepton_campaign_archive_transfer as archive_transfer
    from scripts.round_dagger import remote_campaign_transfer as tree_transfer

    returned_paths = archive_transfer._local_bundle_paths(
        bundle_root, completion_marker["learner_output_manifest_sha256"]
    )
    for path, label in (
        (returned_paths["archive"], "returned learner archive"),
        (returned_paths["inventory"], "returned learner inventory"),
        (returned_paths["manifest"], "returned learner manifest"),
    ):
        _regular(path, label, immutable=True)
    try:
        bundle_manifest, _inventory = archive_transfer._validate_bundle(
            returned_paths["archive"],
            returned_paths["inventory"],
            returned_paths["manifest"],
            expected_manifest_sha256=completion_marker[
                "learner_output_manifest_sha256"
            ],
        )
        observed_tree = tree_transfer.verify_inventory(
            output, returned_paths["inventory"]
        )
    except (
        archive_transfer.CampaignArchiveError,
        tree_transfer.TransferInventoryError,
    ) as exc:
        raise LearnerAcceptanceError(
            f"returned learner archive/tree is invalid: {exc}"
        ) from exc
    if (
        bundle_manifest.get("tree_kind") != "learner_output"
        or bundle_manifest["archive"]["sha256"]
        != completion_marker["learner_output_archive_sha256"]
        or bundle_manifest["inventory"]["sha256"]
        != completion_marker["learner_output_inventory_sha256"]
        or observed_tree["tree_sha256"]
        != completion_marker["learner_output_tree_sha256"]
    ):
        raise LearnerAcceptanceError("returned learner bundle differs from marker")

    snapshot_path = _absolute(env["ROUND_DAGGER_REPLAY_SNAPSHOT"], "replay snapshot")
    parent_checkpoint = _absolute(
        env["ROUND_DAGGER_POLICY_CHECKPOINT"], "parent checkpoint"
    )
    _regular(snapshot_path, "replay snapshot")
    _regular(parent_checkpoint, "parent checkpoint")
    if artifacts.sha256_file(parent_checkpoint) != env[
        "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256"
    ]:
        raise LearnerAcceptanceError("parent checkpoint SHA-256 mismatch")
    try:
        snapshot = replay.load_snapshot(
            snapshot_path,
            expected_file_sha256=env[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ],
            artifact_root=root,
        )
    except Exception as exc:
        raise LearnerAcceptanceError(f"replay snapshot is invalid: {exc}") from exc
    if (
        snapshot["campaign_id"] != request.campaign_id
        or snapshot["through_macro_round"] != macro_round
        or snapshot["snapshot_sha256"]
        != env["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"]
    ):
        raise LearnerAcceptanceError("replay snapshot lineage mismatch")

    result_path = output / request.paths["learner_result"].name
    attestation_path = output / request.paths["completion_attestation"].name
    _regular(result_path, "learner result")
    _regular(attestation_path, "learner completion attestation", immutable=True)
    if (
        artifacts.sha256_file(result_path) != completion_marker["result_sha256"]
        or artifacts.sha256_file(attestation_path)
        != completion_marker["completion_attestation_sha256"]
    ):
        raise LearnerAcceptanceError(
            "result or completion attestation differs from completion marker"
        )

    from scripts.round_dagger import learner_completion_attestation as completion

    try:
        attestation = completion.verify_attestation(
            attestation_path,
            result=result_path,
            campaign_id=request.campaign_id,
            macro_round=macro_round,
            code_sha256=env["ROUND_DAGGER_LEARNER_CODE_SHA256"],
            config_sha256=env["ROUND_DAGGER_LEARNER_CONFIG_SHA256"],
            parent_sha256=env["ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256"],
            snapshot_file_sha256=env[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ],
            snapshot_sha256=env["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"],
            runtime_container_sha256=env[
                "ROUND_DAGGER_REMOTE_CONTAINER_SHA256"
            ],
            learner_contract_sha256=env[
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
            ],
            learner_identity_sha256=env[
                "ROUND_DAGGER_LEARNER_IDENTITY_SHA256"
            ],
        )
    except completion.CompletionAttestationError as exc:
        raise LearnerAcceptanceError(
            f"completion attestation/readback is invalid: {exc}"
        ) from exc

    result = artifacts.load_strict_json(result_path)
    candidate = output / result["checkpoint_path"]
    policy_spec = output / result["policy_spec_path"]
    policy_manifest = output / result["policy_bundle_manifest_path"]
    readback = output / attestation["artifacts"]["wandb_readback"]["path"]
    for path, label in (
        (candidate, "candidate checkpoint"),
        (policy_spec, "candidate policy spec"),
        (policy_manifest, "candidate policy manifest"),
        (readback, "W&B completion readback"),
    ):
        _regular(path, label)
    expected_result_hashes = {
        candidate: result["checkpoint_sha256"],
        policy_spec: result["policy_spec_sha256"],
        policy_manifest: result["policy_bundle_manifest_sha256"],
        readback: attestation["artifacts"]["wandb_readback"]["sha256"],
    }
    if any(artifacts.sha256_file(path) != digest for path, digest in expected_result_hashes.items()):
        raise LearnerAcceptanceError("learner result/policy/readback artifact changed")

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "backend": BACKEND,
        "campaign_id": request.campaign_id,
        "macro_round": macro_round,
        "learner_identity_sha256": request.identity_sha256,
        "backend_contract_path": str(contract_path),
        "backend_contract_sha256": contract.sha256,
        "request_path": str(request_file),
        "request_sha256": request.sha256,
        "completion_marker_path": str(marker),
        "completion_marker_sha256": marker_sha,
        "completion_checksum_path": str(checksum),
        "completion_checksum_sha256": artifacts.sha256_file(checksum),
        "completion_checksum_source_basename": REMOTE_MARKER_BASENAME,
        "returned_output_path": _campaign_relative(root, output, "learner output"),
        "returned_archive_path": str(returned_paths["archive"]),
        "returned_archive_sha256": artifacts.sha256_file(returned_paths["archive"]),
        "returned_inventory_path": str(returned_paths["inventory"]),
        "returned_inventory_sha256": artifacts.sha256_file(returned_paths["inventory"]),
        "returned_manifest_path": str(returned_paths["manifest"]),
        "returned_manifest_sha256": artifacts.sha256_file(returned_paths["manifest"]),
        "returned_tree_sha256": observed_tree["tree_sha256"],
        "learner_result_path": _campaign_relative(root, result_path, "learner result"),
        "learner_result_sha256": artifacts.sha256_file(result_path),
        "candidate_checkpoint_path": _campaign_relative(root, candidate, "candidate checkpoint"),
        "candidate_checkpoint_sha256": artifacts.sha256_file(candidate),
        "candidate_runtime_policy_sha256": result["runtime_policy_sha256"],
        "policy_spec_path": _campaign_relative(root, policy_spec, "policy spec"),
        "policy_spec_sha256": artifacts.sha256_file(policy_spec),
        "policy_manifest_path": _campaign_relative(root, policy_manifest, "policy manifest"),
        "policy_manifest_sha256": artifacts.sha256_file(policy_manifest),
        "completion_attestation_path": _campaign_relative(root, attestation_path, "completion attestation"),
        "completion_attestation_sha256": artifacts.sha256_file(attestation_path),
        "wandb_readback_path": _campaign_relative(root, readback, "W&B readback"),
        "wandb_readback_sha256": artifacts.sha256_file(readback),
        "replay_snapshot_path": _campaign_relative(root, snapshot_path, "replay snapshot"),
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "parent_checkpoint_path": _campaign_relative(root, parent_checkpoint, "parent checkpoint"),
        "parent_checkpoint_sha256": artifacts.sha256_file(parent_checkpoint),
        "parent_runtime_policy_sha256": env["ROUND_DAGGER_RUNTIME_POLICY_SHA256"],
        "learner_code_sha256": env["ROUND_DAGGER_LEARNER_CODE_SHA256"],
        "learner_config_sha256": env["ROUND_DAGGER_LEARNER_CONFIG_SHA256"],
        "learner_contract_sha256": env["ROUND_DAGGER_LEARNER_CONTRACT_SHA256"],
        "runtime_container_sha256": env["ROUND_DAGGER_REMOTE_CONTAINER_SHA256"],
    }
    document["content_sha256"] = artifacts.canonical_json_sha256(document)
    return document


def publish_acceptance(
    *,
    receipt_path: Path,
    campaign_root: Path,
    macro_round: int,
    backend_contract_path: Path,
    request_path: Path,
    marker_path: Path,
    checksum_path: Path,
    return_bundle_root: Path,
    learner_output: Path,
    expected_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Publish or adopt one byte-identical acceptance receipt."""

    canonical = acceptance_path(campaign_root, macro_round)
    if _absolute(receipt_path, "acceptance receipt") != canonical:
        raise LearnerAcceptanceError("acceptance receipt path is non-canonical")
    document = _collect(
        campaign_root=campaign_root,
        macro_round=macro_round,
        backend_contract_path=backend_contract_path,
        request_path=request_path,
        marker_path=marker_path,
        checksum_path=checksum_path,
        return_bundle_root=return_bundle_root,
        learner_output=learner_output,
        expected_environment=expected_environment,
    )
    try:
        artifacts.publish_immutable_json(canonical, document)
    except (artifacts.ArtifactError, OSError) as exc:
        raise LearnerAcceptanceError(
            "acceptance receipt conflicts with an existing artifact"
        ) from exc
    return validate_acceptance(
        canonical,
        campaign_root=campaign_root,
        macro_round=macro_round,
        backend_contract_path=backend_contract_path,
        expected_environment=expected_environment,
    )


def validate_acceptance(
    receipt_path: Path,
    *,
    campaign_root: Path,
    macro_round: int,
    backend_contract_path: Path,
    expected_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Deeply revalidate a receipt and every artifact it commits."""

    canonical = acceptance_path(campaign_root, macro_round)
    path = _absolute(receipt_path, "acceptance receipt")
    if path != canonical:
        raise LearnerAcceptanceError("acceptance receipt path is non-canonical")
    _regular(path, "acceptance receipt", immutable=True)
    try:
        document = artifacts.load_strict_json(path)
    except Exception as exc:
        raise LearnerAcceptanceError("acceptance receipt is unreadable") from exc
    if not isinstance(document, Mapping) or set(document) != _KEYS:
        raise LearnerAcceptanceError("acceptance receipt fields mismatch")
    content = dict(document)
    claimed = content.pop("content_sha256")
    if _sha(claimed, "acceptance content SHA-256") != artifacts.canonical_json_sha256(content):
        raise LearnerAcceptanceError("acceptance receipt content digest mismatch")
    expected = _collect(
        campaign_root=campaign_root,
        macro_round=macro_round,
        backend_contract_path=backend_contract_path,
        request_path=_absolute(document["request_path"], "receipt request"),
        marker_path=_absolute(document["completion_marker_path"], "receipt marker"),
        checksum_path=_absolute(document["completion_checksum_path"], "receipt checksum"),
        return_bundle_root=_absolute(document["returned_manifest_path"], "receipt return manifest").parent,
        learner_output=_absolute(campaign_root, "campaign root") / document["returned_output_path"],
        expected_environment=expected_environment,
    )
    if dict(document) != expected:
        raise LearnerAcceptanceError("acceptance receipt conflicts with validated evidence")
    return dict(document)


__all__ = [
    "BACKEND",
    "LearnerAcceptanceError",
    "SCHEMA",
    "acceptance_path",
    "publish_acceptance",
    "validate_acceptance",
]
