"""Immutable learner handoff, evaluation gate, and atomic policy promotion."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from round_dagger.artifacts import (
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.policy_bundle import (
    BOOTSTRAP_MANIFEST_SCHEMA as POLICY_BUNDLE_MANIFEST_SCHEMA,
    PolicyBundleError,
    validate_policy_bundle_identity,
)
from round_dagger.state import (
    _campaign_root,
    _check_path_chain,
    _internal_path,
    _safe_relative,
    complete_promotion,
    load_campaign,
    load_state,
)

CANDIDATE_SCHEMA = "fast_roborender.round_dagger.policy_candidate.v2"
EVALUATION_SCHEMA = "fast_roborender.round_dagger.policy_evaluation.v2"
PROMOTION_SCHEMA = "fast_roborender.round_dagger.policy_promotion.v1"
BOOTSTRAP_SCHEMA = "fast_roborender.round_dagger.bootstrap_policy.v2"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class PromotionError(ArtifactError):
    """A learner handoff, evaluation, or promotion artifact is invalid."""


def _require_exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PromotionError(
            f"{label} has missing={sorted(keys - actual)} unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PromotionError(f"{label} must be an integer >= {minimum}")
    return value


def _new_root(root: str | Path) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    _check_path_chain(target)
    target.mkdir(parents=True, exist_ok=True)
    _check_path_chain(target)
    return target


def _open_regular(path: Path) -> BinaryIO:
    _check_path_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PromotionError(f"checkpoint source is not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _copy_immutable(source: Path, destination: Path) -> str:
    """Copy a checkpoint, publishing only a complete verified regular file."""

    source = Path(os.path.abspath(os.path.expanduser(os.fspath(source))))
    destination = Path(os.path.abspath(os.fspath(destination)))
    _check_path_chain(source)
    _check_path_chain(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _check_path_chain(destination)

    # Hash source through a no-follow descriptor.  The second pass must match,
    # rejecting a concurrently replaced source rather than staging mixed bytes.
    source_sha = hashlib.sha256()
    try:
        with _open_regular(source) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                source_sha.update(chunk)
    except OSError as exc:
        raise PromotionError(
            f"checkpoint source is missing or unsafe: {source}"
        ) from exc
    expected_sha = source_sha.hexdigest()
    if destination.exists():
        if sha256_file(destination) != expected_sha:
            raise PromotionError(
                "checkpoint destination already exists with different bytes"
            )
        return expected_sha

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    copied_sha = hashlib.sha256()
    try:
        with (
            _open_regular(source) as input_handle,
            os.fdopen(descriptor, "wb") as output,
        ):
            for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                copied_sha.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if (
            copied_sha.hexdigest() != expected_sha
            or sha256_file(temporary) != expected_sha
        ):
            raise PromotionError("checkpoint source changed during immutable copy")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if sha256_file(destination) != expected_sha:
                raise PromotionError("competing checkpoint publication changed bytes")
        os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return expected_sha


_BOOTSTRAP_KEYS = {
    "schema",
    "policy_version",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256",
    "code_sha256",
    "policy_config_sha256",
}
_CANDIDATE_KEYS = {
    "schema",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "policy_version",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256",
    "parent_policy_version",
    "parent_checkpoint_sha256",
    "replay_snapshot_sha256",
    "code_sha256",
    "learner_config_sha256",
    "campaign_training_config_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "optimizer_steps",
}


def _sidecar_paths(checkpoint: Path) -> tuple[Path, Path]:
    return (
        Path(f"{checkpoint}.policy_spec.json"),
        Path(f"{checkpoint}.manifest.json"),
    )


def _stage_policy_bundle(
    *, source_checkpoint: Path, destination_checkpoint: Path
) -> dict[str, str]:
    """Validate and immutably stage one actor-consumable policy bundle.

    The source manifest is validated before any publication.  Its basename
    fields cannot be copied verbatim because campaign staging canonicalizes the
    checkpoint to ``checkpoint.pt``; a semantically identical destination
    manifest is therefore generated only after the checkpoint and policy spec
    have both been copied and verified.  The caller publishes its coordinator
    manifest last, which is the bundle's commit marker.
    """

    source_checkpoint = Path(
        os.path.abspath(os.path.expanduser(os.fspath(source_checkpoint)))
    )
    source_spec, source_manifest = _sidecar_paths(source_checkpoint)
    for path in (source_checkpoint, source_spec, source_manifest):
        _check_path_chain(path)
    source_document = load_strict_json(source_manifest)
    if not isinstance(source_document, Mapping):
        raise PromotionError("source policy bundle manifest must be a mapping")
    runtime_sha = _require_sha256(
        source_document.get("runtime_policy_sha256"),
        "source policy runtime_policy_sha256",
    )
    expected_checkpoint_sha = _require_sha256(
        source_document.get("checkpoint_sha256"),
        "source policy checkpoint_sha256",
    )
    try:
        source_report = validate_policy_bundle_identity(
            checkpoint=source_checkpoint,
            policy_spec=source_spec,
            manifest=source_manifest,
            expected_checkpoint_sha256=expected_checkpoint_sha,
            expected_runtime_policy_sha256=runtime_sha,
        )
    except PolicyBundleError as exc:
        raise PromotionError(f"source policy bundle is invalid: {exc}") from exc

    destination_spec, destination_manifest = _sidecar_paths(destination_checkpoint)
    checkpoint_sha = _copy_immutable(source_checkpoint, destination_checkpoint)
    if checkpoint_sha != expected_checkpoint_sha:
        raise PromotionError("policy checkpoint changed after source bundle validation")
    policy_spec_sha = _copy_immutable(source_spec, destination_spec)
    if policy_spec_sha != source_report["policy_spec_sha256"]:
        raise PromotionError("policy spec changed after source bundle validation")
    destination_bundle_manifest = {
        "schema": POLICY_BUNDLE_MANIFEST_SCHEMA,
        "checkpoint_path": destination_checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "runtime_policy_sha256": runtime_sha,
        "policy_spec_path": destination_spec.name,
        "policy_spec_sha256": policy_spec_sha,
    }
    bundle_manifest_sha = publish_immutable_json(
        destination_manifest, destination_bundle_manifest
    )
    try:
        validate_policy_bundle_identity(
            checkpoint=destination_checkpoint,
            policy_spec=destination_spec,
            manifest=destination_manifest,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_runtime_policy_sha256=runtime_sha,
        )
    except PolicyBundleError as exc:
        raise PromotionError(f"staged policy bundle is invalid: {exc}") from exc
    return {
        "checkpoint_sha256": checkpoint_sha,
        "runtime_policy_sha256": runtime_sha,
        "policy_spec_path": destination_spec.name,
        "policy_spec_sha256": policy_spec_sha,
        "policy_bundle_manifest_path": destination_manifest.name,
        "policy_bundle_manifest_sha256": bundle_manifest_sha,
    }
_EVALUATION_KEYS = {
    "schema",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "candidate_manifest_sha256",
    "policy_version",
    "checkpoint_sha256",
    "replay_snapshot_sha256",
    "evaluator_config_sha256",
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "passed",
    "metrics",
}
_PROMOTION_KEYS = {
    "schema",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "parent_policy_version",
    "parent_checkpoint_sha256",
    "candidate_manifest_sha256",
    "evaluation_sha256",
    "checkpoint_sha256",
    "policy_version",
}


def _policy_paths(root: Path, version: int) -> tuple[Path, str, Path, str]:
    version = _require_int(version, "policy_version")
    directory = root / "policies" / f"policy_{version:06d}"
    checkpoint = directory / "checkpoint.pt"
    candidate = directory / "candidate.json"
    return (
        checkpoint,
        checkpoint.relative_to(root).as_posix(),
        candidate,
        candidate.relative_to(root).as_posix(),
    )


def stage_bootstrap_policy(
    root: str | Path,
    *,
    checkpoint_source: str | Path,
    code_sha256: str,
    policy_config_sha256: str,
) -> dict[str, Any]:
    """Create the immutable version-zero policy bundle before campaign init."""

    root = _new_root(root)
    checkpoint_path, relative, _, _ = _policy_paths(root, 0)
    bundle = _stage_policy_bundle(
        source_checkpoint=Path(checkpoint_source),
        destination_checkpoint=checkpoint_path,
    )
    manifest = {
        "schema": BOOTSTRAP_SCHEMA,
        "policy_version": 0,
        "checkpoint_path": relative,
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "runtime_policy_sha256": bundle["runtime_policy_sha256"],
        "policy_spec_path": f"{checkpoint_path.parent.relative_to(root).as_posix()}/{bundle['policy_spec_path']}",
        "policy_spec_sha256": bundle["policy_spec_sha256"],
        "policy_bundle_manifest_path": f"{checkpoint_path.parent.relative_to(root).as_posix()}/{bundle['policy_bundle_manifest_path']}",
        "policy_bundle_manifest_sha256": bundle["policy_bundle_manifest_sha256"],
        "code_sha256": _require_sha256(code_sha256, "code_sha256"),
        "policy_config_sha256": _require_sha256(
            policy_config_sha256, "policy_config_sha256"
        ),
    }
    _require_exact(manifest, _BOOTSTRAP_KEYS, "bootstrap manifest")
    manifest_path = checkpoint_path.parent / "bootstrap.json"
    manifest_sha = publish_immutable_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
    }


def _validate_candidate(
    root: Path,
    document: Any,
    *,
    state: Mapping[str, Any] | None = None,
    campaign: Mapping[str, Any] | None = None,
    expected_macro_round: int | None = None,
) -> dict[str, Any]:
    candidate = _require_exact(document, _CANDIDATE_KEYS, "candidate manifest")
    if candidate["schema"] != CANDIDATE_SCHEMA:
        raise PromotionError("candidate manifest schema mismatch")
    _require_int(candidate["macro_round"], "candidate macro_round")
    version = _require_int(
        candidate["policy_version"], "candidate policy_version", minimum=1
    )
    _require_int(candidate["parent_policy_version"], "candidate parent_policy_version")
    _require_int(candidate["optimizer_steps"], "candidate optimizer_steps", minimum=1)
    for key in (
        "campaign_sha256",
        "checkpoint_sha256",
        "runtime_policy_sha256",
        "policy_spec_sha256",
        "policy_bundle_manifest_sha256",
        "parent_checkpoint_sha256",
        "replay_snapshot_sha256",
        "code_sha256",
        "learner_config_sha256",
        "campaign_training_config_sha256",
        "teacher_sha256",
        "renderer_sha256",
    ):
        _require_sha256(candidate[key], f"candidate {key}")
    _require_exact(
        {"campaign_id": candidate["campaign_id"]}, {"campaign_id"}, "candidate identity"
    )
    if (
        not isinstance(candidate["campaign_id"], str)
        or _TOKEN_RE.fullmatch(candidate["campaign_id"]) is None
    ):
        raise PromotionError("candidate campaign_id is unsafe")
    checkpoint, expected_checkpoint, _, _ = _policy_paths(root, version)
    if (
        _safe_relative(candidate["checkpoint_path"], "candidate checkpoint_path")
        != expected_checkpoint
    ):
        raise PromotionError("candidate checkpoint path is non-canonical")
    if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
        raise PromotionError("candidate checkpoint is missing or corrupt")
    expected_spec, expected_bundle_manifest = _sidecar_paths(checkpoint)
    expected_spec_relative = expected_spec.relative_to(root).as_posix()
    expected_bundle_relative = expected_bundle_manifest.relative_to(root).as_posix()
    if (
        _safe_relative(candidate["policy_spec_path"], "candidate policy_spec_path")
        != expected_spec_relative
        or _safe_relative(
            candidate["policy_bundle_manifest_path"],
            "candidate policy_bundle_manifest_path",
        )
        != expected_bundle_relative
    ):
        raise PromotionError("candidate policy sidecar path is non-canonical")
    if (
        sha256_file(expected_spec) != candidate["policy_spec_sha256"]
        or sha256_file(expected_bundle_manifest)
        != candidate["policy_bundle_manifest_sha256"]
    ):
        raise PromotionError("candidate policy sidecar is missing or corrupt")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=expected_spec,
        manifest=expected_bundle_manifest,
        expected_checkpoint_sha256=candidate["checkpoint_sha256"],
        expected_runtime_policy_sha256=candidate["runtime_policy_sha256"],
    )
    if (
        expected_macro_round is not None
        and candidate["macro_round"] != expected_macro_round
    ):
        raise PromotionError("candidate macro_round mismatch")
    if campaign is not None:
        expected_campaign = {
            "campaign_id": campaign["campaign_id"],
            "campaign_sha256": canonical_json_sha256(campaign),
            "campaign_training_config_sha256": campaign["training_config_sha256"],
            "teacher_sha256": campaign["teacher_sha256"],
            "renderer_sha256": campaign["renderer_sha256"],
        }
        if any(candidate[key] != value for key, value in expected_campaign.items()):
            raise PromotionError(
                "candidate campaign/config/teacher/renderer lineage mismatch"
            )
    if state is not None:
        active = state["active_policy"]
        expected_state = {
            "macro_round": state["open_macro_round"],
            "policy_version": active["version"] + 1,
            "parent_policy_version": active["version"],
            "parent_checkpoint_sha256": active["checkpoint_sha256"],
            "replay_snapshot_sha256": state["replay_snapshot_sha256"],
        }
        if any(candidate[key] != value for key, value in expected_state.items()):
            raise PromotionError("candidate parent/replay/macro-round lineage mismatch")
    return candidate


def load_candidate_manifest(
    root: str | Path,
    path: str | Path,
    *,
    expected_macro_round: int | None = None,
    bind_state: bool = True,
) -> dict[str, Any]:
    root = _campaign_root(root)
    candidate_path, relative = _internal_path(root, path, "candidate manifest")
    candidate = _validate_candidate(
        root,
        load_strict_json(candidate_path),
        state=load_state(root) if bind_state else None,
        campaign=load_campaign(root),
        expected_macro_round=expected_macro_round,
    )
    _, _, expected_path, expected_relative = _policy_paths(
        root, candidate["policy_version"]
    )
    if candidate_path != expected_path or relative != expected_relative:
        raise PromotionError("candidate manifest path is non-canonical")
    return {
        **candidate,
        "manifest_path": relative,
        "manifest_sha256": sha256_file(candidate_path),
    }


def stage_candidate(
    root: str | Path,
    *,
    macro_round: int,
    checkpoint_source: str | Path,
    replay_snapshot_sha256: str,
    code_sha256: str,
    learner_config_sha256: str,
    optimizer_steps: int,
) -> dict[str, Any]:
    """Stage and verify the H100 learner result without changing active policy."""

    root = _campaign_root(root)
    state = load_state(root)
    campaign = load_campaign(root)
    macro_round = _require_int(macro_round, "macro_round")
    if state["phase"] != "INGESTED" or state["open_macro_round"] != macro_round:
        raise PromotionError("candidate staging requires the open INGESTED macro-round")
    replay_sha = _require_sha256(replay_snapshot_sha256, "replay_snapshot_sha256")
    if replay_sha != state["replay_snapshot_sha256"]:
        raise PromotionError(
            "candidate replay snapshot does not match coordinator state"
        )
    version = int(state["active_policy"]["version"]) + 1
    checkpoint_path, relative, manifest_path, _ = _policy_paths(root, version)
    bundle = _stage_policy_bundle(
        source_checkpoint=Path(checkpoint_source),
        destination_checkpoint=checkpoint_path,
    )
    manifest = {
        "schema": CANDIDATE_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": canonical_json_sha256(campaign),
        "macro_round": macro_round,
        "policy_version": version,
        "checkpoint_path": relative,
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "runtime_policy_sha256": bundle["runtime_policy_sha256"],
        "policy_spec_path": f"{checkpoint_path.parent.relative_to(root).as_posix()}/{bundle['policy_spec_path']}",
        "policy_spec_sha256": bundle["policy_spec_sha256"],
        "policy_bundle_manifest_path": f"{checkpoint_path.parent.relative_to(root).as_posix()}/{bundle['policy_bundle_manifest_path']}",
        "policy_bundle_manifest_sha256": bundle["policy_bundle_manifest_sha256"],
        "parent_policy_version": int(state["active_policy"]["version"]),
        "parent_checkpoint_sha256": state["active_policy"]["checkpoint_sha256"],
        "replay_snapshot_sha256": replay_sha,
        "code_sha256": _require_sha256(code_sha256, "code_sha256"),
        "learner_config_sha256": _require_sha256(
            learner_config_sha256, "learner_config_sha256"
        ),
        "campaign_training_config_sha256": campaign["training_config_sha256"],
        "teacher_sha256": campaign["teacher_sha256"],
        "renderer_sha256": campaign["renderer_sha256"],
        "optimizer_steps": _require_int(optimizer_steps, "optimizer_steps", minimum=1),
    }
    _validate_candidate(
        root, manifest, state=state, campaign=campaign, expected_macro_round=macro_round
    )
    manifest_sha = publish_immutable_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha,
    }


def _validate_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise PromotionError("evaluation metrics must be a mapping")
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or _TOKEN_RE.fullmatch(key) is None:
            raise PromotionError("evaluation metric names must be safe tokens")
        if isinstance(value, bool):
            raise PromotionError("evaluation metric values must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PromotionError("evaluation metric values must be numeric") from exc
        if not math.isfinite(number):
            raise PromotionError("evaluation metrics must be finite")
        clean[key] = number
    return clean


def _validate_evaluation(
    root: Path,
    document: Any,
    *,
    candidate: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _require_exact(document, _EVALUATION_KEYS, "evaluation manifest")
    if evidence["schema"] != EVALUATION_SCHEMA or type(evidence["passed"]) is not bool:
        raise PromotionError("evaluation schema/passed flag is invalid")
    for key in (
        "campaign_sha256",
        "candidate_manifest_sha256",
        "checkpoint_sha256",
        "replay_snapshot_sha256",
        "evaluator_config_sha256",
        "evidence_bundle_sha256",
    ):
        _require_sha256(evidence[key], f"evaluation {key}")
    _require_int(evidence["macro_round"], "evaluation macro_round")
    _require_int(evidence["policy_version"], "evaluation policy_version", minimum=1)
    expected = {
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": canonical_json_sha256(campaign),
        "macro_round": candidate["macro_round"],
        "policy_version": candidate["policy_version"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
    }
    if any(evidence[key] != value for key, value in expected.items()):
        raise PromotionError("evaluation belongs to another campaign/candidate/replay")
    if not isinstance(evidence["evidence_bundle_path"], str):
        raise PromotionError("evaluation evidence bundle path is unsafe")
    bundle_path, bundle_relative = _internal_path(
        root, evidence["evidence_bundle_path"], "evaluation evidence bundle"
    )
    expected_bundle = (
        root
        / "rounds"
        / f"macro_round_{candidate['macro_round']:06d}"
        / "evaluation"
        / "results"
        / "evaluation_bundle.json"
    )
    if (
        bundle_path != expected_bundle
        or evidence["evidence_bundle_path"] != bundle_relative
    ):
        raise PromotionError("evaluation evidence bundle path is non-canonical")
    if sha256_file(bundle_path) != evidence["evidence_bundle_sha256"]:
        raise PromotionError("evaluation evidence bundle SHA mismatch")
    evidence["metrics"] = _validate_metrics(evidence["metrics"])
    return evidence


def write_evaluation(
    root: str | Path,
    *,
    candidate_manifest: str | Path,
    passed: bool,
    metrics: Mapping[str, float],
    evaluator_config_sha256: str,
    evidence_bundle_path: str | Path,
    evidence_bundle_sha256: str,
) -> dict[str, Any]:
    """Publish fixed-seed evaluation evidence for one exact staged candidate."""

    root = _campaign_root(root)
    state = load_state(root)
    if state["phase"] != "TRAINED":
        raise PromotionError("evaluation publication requires a TRAINED candidate")
    candidate = load_candidate_manifest(
        root,
        candidate_manifest,
        expected_macro_round=state["open_macro_round"],
        bind_state=True,
    )
    staged = state["candidate_policy"]
    if staged["manifest_sha256"] != candidate["manifest_sha256"]:
        raise PromotionError("candidate manifest does not match coordinator state")
    evidence = {
        "schema": EVALUATION_SCHEMA,
        "campaign_id": load_campaign(root)["campaign_id"],
        "campaign_sha256": state["campaign_sha256"],
        "macro_round": candidate["macro_round"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "policy_version": candidate["policy_version"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
        "evaluator_config_sha256": _require_sha256(
            evaluator_config_sha256, "evaluator_config_sha256"
        ),
        "evidence_bundle_path": _internal_path(
            root, evidence_bundle_path, "evaluation evidence bundle"
        )[1],
        "evidence_bundle_sha256": _require_sha256(
            evidence_bundle_sha256, "evidence_bundle_sha256"
        ),
        "passed": passed,
        "metrics": _validate_metrics(metrics),
    }
    if type(passed) is not bool:
        raise PromotionError("evaluation passed flag must be boolean")
    campaign = load_campaign(root)
    _validate_evaluation(root, evidence, candidate=candidate, campaign=campaign)
    candidate_path, _ = _internal_path(
        root, candidate["manifest_path"], "candidate manifest"
    )
    evidence_path = candidate_path.parent / "evaluation.json"
    evidence_sha = publish_immutable_json(evidence_path, evidence)
    return {
        **evidence,
        "manifest_path": evidence_path.relative_to(root).as_posix(),
        "evidence_sha256": evidence_sha,
    }


def load_evaluation_manifest(
    root: str | Path,
    path: str | Path,
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    root = _campaign_root(root)
    evidence_path, relative = _internal_path(root, path, "evaluation manifest")
    expected_path = (
        root
        / "policies"
        / f"policy_{candidate['policy_version']:06d}"
        / "evaluation.json"
    )
    if evidence_path != expected_path:
        raise PromotionError("evaluation manifest path is non-canonical")
    evidence = _validate_evaluation(
        root,
        load_strict_json(evidence_path),
        candidate=candidate,
        campaign=load_campaign(root),
    )
    return {
        **evidence,
        "manifest_path": relative,
        "evidence_sha256": sha256_file(evidence_path),
    }


def promote_candidate(
    root: str | Path,
    *,
    macro_round: int,
    candidate_manifest: str | Path,
    evaluation_manifest: str | Path,
) -> dict[str, Any]:
    """Promote only a passed, non-stale, byte-verified candidate."""

    root = _campaign_root(root)
    state = load_state(root)
    macro_round = _require_int(macro_round, "macro_round")
    # Bind to campaign even on an idempotent retry, but do not bind to current
    # parent after it has already advanced.
    candidate = load_candidate_manifest(
        root,
        candidate_manifest,
        expected_macro_round=macro_round,
        bind_state=state["phase"] != "READY",
    )
    evaluation = load_evaluation_manifest(
        root, evaluation_manifest, candidate=candidate
    )
    if evaluation["passed"] is not True:
        raise PromotionError("candidate failed the fixed-seed evaluation gate")
    candidate_sha = candidate["manifest_sha256"]
    evaluation_sha = evaluation["evidence_sha256"]
    promotion = {
        "schema": PROMOTION_SCHEMA,
        "campaign_id": candidate["campaign_id"],
        "campaign_sha256": candidate["campaign_sha256"],
        "macro_round": macro_round,
        "parent_policy_version": candidate["parent_policy_version"],
        "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
        "candidate_manifest_sha256": candidate_sha,
        "evaluation_sha256": evaluation_sha,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "policy_version": candidate["policy_version"],
    }
    _require_exact(promotion, _PROMOTION_KEYS, "promotion manifest")
    promotion_path = (
        root
        / "policies"
        / f"policy_{candidate['policy_version']:06d}"
        / "promotion.json"
    )

    if state["phase"] == "READY":
        active = state["active_policy"]
        if (
            active["version"] != candidate["policy_version"]
            or active["checkpoint_sha256"] != candidate["checkpoint_sha256"]
        ):
            raise PromotionError("active policy changed since candidate training")
        if not promotion_path.exists():
            raise PromotionError("READY retry has no immutable promotion evidence")
        existing = _require_exact(
            load_strict_json(promotion_path), _PROMOTION_KEYS, "promotion manifest"
        )
        if (
            existing != promotion
            or sha256_file(promotion_path) != state["last_promotion_sha256"]
        ):
            raise PromotionError("promotion retry evidence does not match active state")
        return {
            **promotion,
            "manifest_path": promotion_path.relative_to(root).as_posix(),
            "promotion_sha256": sha256_file(promotion_path),
        }

    if state["phase"] != "EVALUATED" or state["open_macro_round"] != macro_round:
        raise PromotionError("promotion requires the open EVALUATED macro-round")
    active = state["active_policy"]
    if (
        candidate["parent_policy_version"] != active["version"]
        or candidate["parent_checkpoint_sha256"] != active["checkpoint_sha256"]
    ):
        raise PromotionError("candidate parent is no longer the active policy")
    if state["evaluation_sha256"] != evaluation_sha:
        raise PromotionError("coordinator evaluation digest does not match evidence")
    staged = state["candidate_policy"]
    if (
        staged["manifest_sha256"] != candidate_sha
        or staged["checkpoint_sha256"] != candidate["checkpoint_sha256"]
    ):
        raise PromotionError("coordinator candidate does not match learner handoff")
    promotion_sha = publish_immutable_json(promotion_path, promotion)
    complete_promotion(
        root,
        macro_round=macro_round,
        checkpoint_path=candidate["checkpoint_path"],
        promotion_manifest_path=promotion_path,
        promotion_manifest_sha256=promotion_sha,
    )
    return {
        **promotion,
        "manifest_path": promotion_path.relative_to(root).as_posix(),
        "promotion_sha256": promotion_sha,
    }


def load_active_policy(root: str | Path) -> dict[str, Any]:
    """Return and fully verify the exact checkpoint an actor must freeze."""

    root = _campaign_root(root)
    state = load_state(root)
    active = dict(state["active_policy"])
    checkpoint, _ = _internal_path(root, active["checkpoint_path"], "active checkpoint")
    if sha256_file(checkpoint) != active["checkpoint_sha256"]:
        raise PromotionError("active policy checkpoint failed SHA-256 verification")
    policy_dir = checkpoint.parent
    manifest_path = policy_dir / (
        "bootstrap.json" if active["version"] == 0 else "candidate.json"
    )
    manifest = load_strict_json(manifest_path)
    if sha256_file(manifest_path) != active["manifest_sha256"]:
        raise PromotionError("active policy manifest failed SHA-256 verification")
    expected = {
        "checkpoint_path": active["checkpoint_path"],
        "checkpoint_sha256": active["checkpoint_sha256"],
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "policy_spec_sha256": active["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": active[
            "policy_bundle_manifest_sha256"
        ],
        "code_sha256": active["code_sha256"],
    }
    config_field = (
        "policy_config_sha256" if active["version"] == 0 else "learner_config_sha256"
    )
    expected[config_field] = active["policy_config_sha256"]
    if not isinstance(manifest, Mapping) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise PromotionError("active policy manifest lineage mismatch")
    policy_spec, policy_bundle_manifest = _sidecar_paths(checkpoint)
    expected_spec_path = policy_spec.relative_to(root).as_posix()
    expected_bundle_path = policy_bundle_manifest.relative_to(root).as_posix()
    if (
        manifest.get("policy_spec_path") != expected_spec_path
        or manifest.get("policy_bundle_manifest_path") != expected_bundle_path
        or sha256_file(policy_spec) != active["policy_spec_sha256"]
        or sha256_file(policy_bundle_manifest)
        != active["policy_bundle_manifest_sha256"]
    ):
        raise PromotionError("active policy sidecar lineage mismatch")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=policy_spec,
        manifest=policy_bundle_manifest,
        expected_checkpoint_sha256=active["checkpoint_sha256"],
        expected_runtime_policy_sha256=active["runtime_policy_sha256"],
    )
    active["absolute_checkpoint_path"] = str(checkpoint)
    active["absolute_policy_spec_path"] = str(policy_spec)
    active["absolute_policy_bundle_manifest_path"] = str(policy_bundle_manifest)
    active["manifest_path"] = manifest_path.relative_to(root).as_posix()
    return active


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "CANDIDATE_SCHEMA",
    "EVALUATION_SCHEMA",
    "PROMOTION_SCHEMA",
    "PromotionError",
    "load_active_policy",
    "load_candidate_manifest",
    "load_evaluation_manifest",
    "promote_candidate",
    "stage_bootstrap_policy",
    "stage_candidate",
    "write_evaluation",
]
