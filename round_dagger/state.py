"""Durable, evidence-bound state for round-based DAgger.

A macro-round freezes one student checkpoint and one DAgger behavior-mixture
``beta`` across every cohort.  Mutable coordinator state is updated with a
revision compare-and-swap; the contracts and evidence it points at are
immutable, checksum-verified artifacts.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from round_dagger.artifacts import (
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
    verify_cohort,
)
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM
from round_dagger.policy_bundle import validate_policy_bundle_identity
from round_dagger.production_data_lineage import (
    ProductionDataLineageError,
    build_cohort_assignment,
    data_lineage_sha256,
    validate_cohort_assignment,
    validate_data_lineage,
)

# V1 campaigns predate a durable execution horizon and remain readable for
# smoke tests and non-production experiments.  A bounded campaign is V2: the
# immutable campaign manifest carries the exact number of macro-rounds and the
# coordinator refuses to open anything beyond that frontier.
CAMPAIGN_SCHEMA = "fast_roborender.round_dagger.campaign.v1"
BOUNDED_CAMPAIGN_SCHEMA = "fast_roborender.round_dagger.campaign.v2"
PRODUCTION_CAMPAIGN_SCHEMA = "fast_roborender.round_dagger.campaign.v3"
STATE_SCHEMA = "fast_roborender.round_dagger.coordinator_state.v2"
ROUND_CONTRACT_SCHEMA = "fast_roborender.round_dagger.round_contract.v2"
PRODUCTION_ROUND_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.round_contract.v3"
)
COLLECTION_SCHEMA = "fast_roborender.round_dagger.macro_round_collection.v1"
BOOTSTRAP_SCHEMA = "fast_roborender.round_dagger.bootstrap_policy.v2"
CANDIDATE_SCHEMA = "fast_roborender.round_dagger.policy_candidate.v2"
EVALUATION_SCHEMA = "fast_roborender.round_dagger.policy_evaluation.v2"
PROMOTION_SCHEMA = "fast_roborender.round_dagger.policy_promotion.v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EVALUATION_RECOVERY_ID_RE = re.compile(r"hardened-[0-9a-f]{20}")
PHASES = {
    "READY",
    "COLLECTING",
    "COLLECTED",
    "INGESTED",
    "TRAINED",
    "EVALUATED",
}
_CANDIDATE_MANIFEST_KEYS = {
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
_EVALUATION_MANIFEST_KEYS = {
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
_PROMOTION_MANIFEST_KEYS = {
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


class StateError(ArtifactError):
    """A coordinator artifact or state transition violated its contract."""


def _require_exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise StateError(
            f"{label} has missing={sorted(keys - actual)} unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StateError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise StateError(f"{name} must be a lowercase SHA-256")
    return value


def _require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise StateError(f"{name} is not a safe token")
    return value


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_path_chain(path: Path) -> None:
    """Reject symlinks and non-directory ancestors without resolving them."""

    path = _absolute(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise StateError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise StateError(f"path ancestor is not a directory: {current}")


def _campaign_root(root: str | Path, *, create: bool = False) -> Path:
    target = _absolute(root)
    _check_path_chain(target)
    if create:
        target.mkdir(parents=True, exist_ok=True)
        _check_path_chain(target)
    try:
        info = os.lstat(target)
    except FileNotFoundError as exc:
        raise StateError(f"campaign root does not exist: {target}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise StateError(f"campaign root is not a directory: {target}")
    return target


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StateError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise StateError(f"{label} must be a canonical relative POSIX path")
    if any(
        part in {"", ".", ".."} or TOKEN_RE.fullmatch(part) is None
        for part in path.parts
    ):
        raise StateError(f"{label} is unsafe")
    return value


def _internal_path(root: Path, value: str | Path, label: str) -> tuple[Path, str]:
    supplied = Path(value).expanduser()
    absolute = _absolute(supplied if supplied.is_absolute() else root / supplied)
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise StateError(f"{label} must be inside campaign root") from exc
    _safe_relative(relative, label)
    _check_path_chain(absolute)
    return absolute, relative


def _validate_beta_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schedule, Mapping):
        raise StateError("beta_schedule must be a mapping")
    allowed = {"kind", "start", "end", "decay_rounds", "rate"}
    unknown = set(schedule) - allowed
    if unknown:
        raise StateError(f"unknown beta_schedule fields: {sorted(unknown)}")
    kind = schedule.get("kind")
    if kind not in {"constant", "linear", "exponential"}:
        raise StateError("beta_schedule.kind must be constant, linear, or exponential")
    try:
        start = float(schedule["start"])
        end = float(schedule.get("end", start))
        decay_rounds = int(schedule.get("decay_rounds", 1))
        rate = float(schedule.get("rate", 0.99))
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError("beta_schedule contains invalid numeric values") from exc
    if not all(math.isfinite(item) for item in (start, end, rate)):
        raise StateError("beta_schedule values must be finite")
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise StateError("beta_schedule start/end must be in [0,1]")
    if decay_rounds <= 0 or not 0.0 < rate <= 1.0:
        raise StateError("beta_schedule decay/rate is invalid")
    return {
        "kind": kind,
        "start": start,
        "end": end,
        "decay_rounds": decay_rounds,
        "rate": rate,
    }


def beta_for_macro_round(schedule: Mapping[str, Any], macro_round: int) -> float:
    """Return deterministic beta indexed by durable macro-round."""

    parsed = _validate_beta_schedule(schedule)
    index = _require_int(macro_round, "macro_round")
    if parsed["kind"] == "constant":
        beta = parsed["start"]
    elif parsed["kind"] == "linear":
        fraction = min(1.0, index / float(parsed["decay_rounds"]))
        beta = parsed["start"] + (parsed["end"] - parsed["start"]) * fraction
    else:
        beta = max(parsed["end"], parsed["start"] * parsed["rate"] ** index)
    return float(min(1.0, max(0.0, beta)))


# Compatibility spelling for early callers; all persisted fields are canonical.
beta_for_round = beta_for_macro_round


def _atomic_replace_json(path: Path, document: Mapping[str, Any]) -> None:
    _check_path_chain(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_path_chain(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _state_lock(root: Path) -> Iterator[None]:
    path = root / ".coordinator_state.lock"
    _check_path_chain(path)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StateError("coordinator lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_campaign(document: Mapping[str, Any]) -> dict[str, Any]:
    base_keys = {
        "schema",
        "campaign_id",
        "target_environments",
        "active_environments",
        "cohort_count",
        "steps_per_environment",
        "view_names",
        "policy_image_shape",
        "action_dim",
        "teacher_sha256",
        "renderer_sha256",
        "camera_mapping_sha256",
        "training_config_sha256",
        "bootstrap_policy_manifest_sha256",
        "code_sha256",
        "policy_config_sha256",
        "beta_schedule",
        "beta_schedule_sha256",
    }
    schema = document.get("schema") if isinstance(document, Mapping) else None
    if schema == CAMPAIGN_SCHEMA:
        keys = base_keys
    elif schema == BOUNDED_CAMPAIGN_SCHEMA:
        keys = base_keys | {"macro_round_count"}
    elif schema == PRODUCTION_CAMPAIGN_SCHEMA:
        keys = base_keys | {
            "macro_round_count",
            "data_lineage",
            "data_lineage_sha256",
        }
    else:
        raise StateError("campaign schema mismatch")
    campaign = _require_exact(document, keys, "campaign manifest")
    _require_token(campaign["campaign_id"], "campaign_id")
    target = _require_int(
        campaign["target_environments"], "target_environments", minimum=1
    )
    active = _require_int(
        campaign["active_environments"], "active_environments", minimum=1
    )
    cohorts = _require_int(campaign["cohort_count"], "cohort_count", minimum=1)
    if target != active * cohorts:
        raise StateError(
            "target_environments must equal active_environments * cohort_count"
        )
    _require_int(campaign["steps_per_environment"], "steps_per_environment", minimum=1)
    _require_int(campaign["action_dim"], "action_dim", minimum=1)
    if schema in {BOUNDED_CAMPAIGN_SCHEMA, PRODUCTION_CAMPAIGN_SCHEMA}:
        _require_int(
            campaign["macro_round_count"], "macro_round_count", minimum=1
        )
    views = campaign["view_names"]
    if not isinstance(views, list) or len(views) != 3 or len(set(views)) != 3:
        raise StateError("view_names must contain exactly three unique views")
    for view in views:
        _require_token(view, "view name")
    shape = campaign["policy_image_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[2] != 3
        or shape[0] % 3
    ):
        raise StateError("policy_image_shape must be [3*H,W,3]")
    for name in (
        "teacher_sha256",
        "renderer_sha256",
        "camera_mapping_sha256",
        "training_config_sha256",
        "bootstrap_policy_manifest_sha256",
        "code_sha256",
        "policy_config_sha256",
        "beta_schedule_sha256",
    ):
        _require_sha(campaign[name], name)
    schedule = _validate_beta_schedule(campaign["beta_schedule"])
    if canonical_json_sha256(schedule) != campaign["beta_schedule_sha256"]:
        raise StateError("beta schedule digest mismatch")
    if schema == PRODUCTION_CAMPAIGN_SCHEMA:
        try:
            lineage = validate_data_lineage(campaign["data_lineage"])
            lineage_sha = data_lineage_sha256(lineage)
        except ProductionDataLineageError as exc:
            raise StateError(f"production data lineage is invalid: {exc}") from exc
        if campaign["data_lineage_sha256"] != lineage_sha:
            raise StateError("production data lineage digest mismatch")
        if (
            target != 256
            or active != 64
            or cohorts != 4
            or campaign["steps_per_environment"] != 512
            or campaign["macro_round_count"] != 5
        ):
            raise StateError("production data lineage requires exact 4x64x512x5")
    return campaign


def _validate_bootstrap(
    root: Path, checkpoint_path: str, checkpoint_sha: str
) -> dict[str, Any]:
    checkpoint_relative = _safe_relative(checkpoint_path, "bootstrap_policy_path")
    checkpoint, _ = _internal_path(root, checkpoint_relative, "bootstrap_policy_path")
    if sha256_file(checkpoint) != _require_sha(
        checkpoint_sha, "bootstrap_policy_sha256"
    ):
        raise StateError("bootstrap policy checkpoint SHA-256 mismatch")
    manifest_path = checkpoint.parent / "bootstrap.json"
    manifest = _require_exact(
        load_strict_json(manifest_path),
        {
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
        },
        "bootstrap manifest",
    )
    if manifest["schema"] != BOOTSTRAP_SCHEMA or manifest["policy_version"] != 0:
        raise StateError("bootstrap manifest schema/version mismatch")
    if (
        manifest["checkpoint_path"] != checkpoint_relative
        or manifest["checkpoint_sha256"] != checkpoint_sha
    ):
        raise StateError("bootstrap manifest does not bind the supplied checkpoint")
    _require_sha(manifest["code_sha256"], "bootstrap code_sha256")
    _require_sha(manifest["policy_config_sha256"], "bootstrap policy_config_sha256")
    runtime_sha = _require_sha(
        manifest["runtime_policy_sha256"], "bootstrap runtime_policy_sha256"
    )
    spec_relative = _safe_relative(
        manifest["policy_spec_path"], "bootstrap policy_spec_path"
    )
    bundle_relative = _safe_relative(
        manifest["policy_bundle_manifest_path"],
        "bootstrap policy_bundle_manifest_path",
    )
    expected_spec = Path(f"{checkpoint}.policy_spec.json")
    expected_bundle = Path(f"{checkpoint}.manifest.json")
    spec, verified_spec_relative = _internal_path(
        root, spec_relative, "bootstrap policy spec"
    )
    bundle_manifest, verified_bundle_relative = _internal_path(
        root, bundle_relative, "bootstrap policy bundle manifest"
    )
    if (
        spec != expected_spec
        or bundle_manifest != expected_bundle
        or verified_spec_relative != spec_relative
        or verified_bundle_relative != bundle_relative
    ):
        raise StateError("bootstrap policy sidecar path is non-canonical")
    if sha256_file(spec) != _require_sha(
        manifest["policy_spec_sha256"], "bootstrap policy_spec_sha256"
    ):
        raise StateError("bootstrap policy spec SHA-256 mismatch")
    if sha256_file(bundle_manifest) != _require_sha(
        manifest["policy_bundle_manifest_sha256"],
        "bootstrap policy_bundle_manifest_sha256",
    ):
        raise StateError("bootstrap policy bundle manifest SHA-256 mismatch")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=spec,
        manifest=bundle_manifest,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_runtime_policy_sha256=runtime_sha,
    )
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def initialize_campaign(
    root: str | Path,
    *,
    campaign_id: str,
    bootstrap_policy_sha256: str,
    bootstrap_policy_path: str,
    target_environments: int = 256,
    active_environments: int = 64,
    steps_per_environment: int = 512,
    view_names: tuple[str, str, str] = ("head_color", "hand_left", "hand_right"),
    policy_image_shape: tuple[int, int, int] = (324, 192, 3),
    # 11 learned controls + 15 fixed Homie controls.  Isaac's robot.actions_dim
    # is 31 only after HomieBase maps this composite rollout action into the
    # physical whole-body command.
    action_dim: int = REPLAY_ACTION_DIM,
    teacher_sha256: str,
    renderer_sha256: str,
    camera_mapping_sha256: str,
    training_config_sha256: str,
    beta_schedule: Mapping[str, Any] | None = None,
    macro_round_count: int | None = None,
    data_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a campaign and its checksum-bound version-zero policy pointer."""

    root = _campaign_root(root, create=True)
    target = _require_int(target_environments, "target_environments", minimum=1)
    active = _require_int(active_environments, "active_environments", minimum=1)
    if target % active:
        raise StateError("target_environments must be divisible by active_environments")
    bootstrap = _validate_bootstrap(
        root, bootstrap_policy_path, bootstrap_policy_sha256
    )
    schedule = _validate_beta_schedule(
        beta_schedule
        or {"kind": "linear", "start": 1.0, "end": 0.0, "decay_rounds": 4}
    )
    if macro_round_count is not None:
        macro_round_count = _require_int(
            macro_round_count, "macro_round_count", minimum=1
        )
    normalized_lineage: dict[str, Any] | None = None
    if data_lineage is not None:
        if macro_round_count is None:
            raise StateError(
                "production data lineage requires a bounded macro-round count"
            )
        try:
            normalized_lineage = validate_data_lineage(data_lineage)
        except ProductionDataLineageError as exc:
            raise StateError(f"production data lineage is invalid: {exc}") from exc
    campaign = {
        "schema": (
            PRODUCTION_CAMPAIGN_SCHEMA
            if normalized_lineage is not None
            else (
                BOUNDED_CAMPAIGN_SCHEMA
                if macro_round_count is not None
                else CAMPAIGN_SCHEMA
            )
        ),
        "campaign_id": _require_token(campaign_id, "campaign_id"),
        "target_environments": target,
        "active_environments": active,
        "cohort_count": target // active,
        "steps_per_environment": _require_int(
            steps_per_environment, "steps_per_environment", minimum=1
        ),
        "view_names": list(view_names),
        "policy_image_shape": list(policy_image_shape),
        "action_dim": _require_int(action_dim, "action_dim", minimum=1),
        "teacher_sha256": _require_sha(teacher_sha256, "teacher_sha256"),
        "renderer_sha256": _require_sha(renderer_sha256, "renderer_sha256"),
        "camera_mapping_sha256": _require_sha(
            camera_mapping_sha256, "camera_mapping_sha256"
        ),
        "training_config_sha256": _require_sha(
            training_config_sha256, "training_config_sha256"
        ),
        "bootstrap_policy_manifest_sha256": bootstrap["manifest_sha256"],
        "code_sha256": bootstrap["code_sha256"],
        "policy_config_sha256": bootstrap["policy_config_sha256"],
        "beta_schedule": schedule,
        "beta_schedule_sha256": canonical_json_sha256(schedule),
    }
    if macro_round_count is not None:
        campaign["macro_round_count"] = macro_round_count
    if normalized_lineage is not None:
        campaign["data_lineage"] = normalized_lineage
        campaign["data_lineage_sha256"] = data_lineage_sha256(
            normalized_lineage
        )
    _validate_campaign(campaign)
    state = {
        "schema": STATE_SCHEMA,
        "campaign_sha256": canonical_json_sha256(campaign),
        "revision": 0,
        "phase": "READY",
        "next_macro_round": 0,
        "open_macro_round": None,
        "last_collected_macro_round": -1,
        "last_ingested_macro_round": -1,
        "last_trained_macro_round": -1,
        "last_evaluated_macro_round": -1,
        "active_policy": {
            "version": 0,
            "checkpoint_sha256": bootstrap["checkpoint_sha256"],
            "checkpoint_path": bootstrap["checkpoint_path"],
            "runtime_policy_sha256": bootstrap["runtime_policy_sha256"],
            "policy_spec_sha256": bootstrap["policy_spec_sha256"],
            "policy_bundle_manifest_sha256": bootstrap[
                "policy_bundle_manifest_sha256"
            ],
            "manifest_sha256": bootstrap["manifest_sha256"],
            "code_sha256": bootstrap["code_sha256"],
            "policy_config_sha256": bootstrap["policy_config_sha256"],
        },
        "collection_manifest_sha256": None,
        "replay_snapshot_sha256": None,
        "candidate_policy": None,
        "evaluation_sha256": None,
        "last_promotion_sha256": None,
    }
    _validate_state(state, campaign)
    with _state_lock(root):
        publish_immutable_json(root / "campaign.json", campaign)
        state_path = root / "coordinator_state.json"
        if state_path.exists():
            existing = _load_state_unlocked(root)
            if existing != state:
                raise StateError("campaign state already exists with different content")
        else:
            _atomic_replace_json(state_path, state)
    return {"campaign": campaign, "state": state}


def load_campaign(root: str | Path) -> dict[str, Any]:
    root = _campaign_root(root)
    return _validate_campaign(load_strict_json(root / "campaign.json"))


_ACTIVE_KEYS = {
    "version",
    "checkpoint_sha256",
    "checkpoint_path",
    "runtime_policy_sha256",
    "policy_spec_sha256",
    "policy_bundle_manifest_sha256",
    "manifest_sha256",
    "code_sha256",
    "policy_config_sha256",
}
_CANDIDATE_KEYS = _ACTIVE_KEYS | {
    "macro_round",
    "manifest_path",
    "parent_policy_version",
    "parent_checkpoint_sha256",
    "replay_snapshot_sha256",
}


def _validate_policy(
    value: Any, label: str, *, candidate: bool = False
) -> dict[str, Any]:
    policy = _require_exact(
        value, _CANDIDATE_KEYS if candidate else _ACTIVE_KEYS, label
    )
    _require_int(policy["version"], f"{label}.version")
    for key in (
        "checkpoint_sha256",
        "runtime_policy_sha256",
        "policy_spec_sha256",
        "policy_bundle_manifest_sha256",
        "manifest_sha256",
        "code_sha256",
        "policy_config_sha256",
    ):
        _require_sha(policy[key], f"{label}.{key}")
    _safe_relative(policy["checkpoint_path"], f"{label}.checkpoint_path")
    if candidate:
        _require_int(policy["macro_round"], f"{label}.macro_round")
        _safe_relative(policy["manifest_path"], f"{label}.manifest_path")
        _require_int(policy["parent_policy_version"], f"{label}.parent_policy_version")
        _require_sha(
            policy["parent_checkpoint_sha256"], f"{label}.parent_checkpoint_sha256"
        )
        _require_sha(
            policy["replay_snapshot_sha256"], f"{label}.replay_snapshot_sha256"
        )
    return policy


def _validate_state(
    document: Mapping[str, Any], campaign: Mapping[str, Any] | str
) -> dict[str, Any]:
    if isinstance(campaign, Mapping):
        campaign_sha256 = canonical_json_sha256(campaign)
        macro_round_count = campaign.get("macro_round_count")
    else:
        # Compatibility for internal callers that validate a legacy V1 state
        # against only its digest.  Bounded campaign loads always pass the
        # complete immutable campaign document.
        campaign_sha256 = campaign
        macro_round_count = None
    keys = {
        "schema",
        "campaign_sha256",
        "revision",
        "phase",
        "next_macro_round",
        "open_macro_round",
        "last_collected_macro_round",
        "last_ingested_macro_round",
        "last_trained_macro_round",
        "last_evaluated_macro_round",
        "active_policy",
        "collection_manifest_sha256",
        "replay_snapshot_sha256",
        "candidate_policy",
        "evaluation_sha256",
        "last_promotion_sha256",
    }
    state = _require_exact(document, keys, "coordinator state")
    if state["schema"] != STATE_SCHEMA or state["campaign_sha256"] != campaign_sha256:
        raise StateError("coordinator state schema/campaign mismatch")
    _require_int(state["revision"], "state revision")
    if state["phase"] not in PHASES:
        raise StateError("invalid coordinator phase")
    next_round = _require_int(state["next_macro_round"], "next_macro_round")
    open_round = state["open_macro_round"]
    if open_round is not None:
        _require_int(open_round, "open_macro_round")
    if macro_round_count is not None:
        limit = _require_int(
            macro_round_count, "campaign.macro_round_count", minimum=1
        )
        if next_round > limit:
            raise StateError("next_macro_round exceeds the bounded campaign horizon")
        if open_round is not None and open_round >= limit:
            raise StateError("open_macro_round exceeds the bounded campaign horizon")
    for key in (
        "last_collected_macro_round",
        "last_ingested_macro_round",
        "last_trained_macro_round",
        "last_evaluated_macro_round",
    ):
        if type(state[key]) is not int or state[key] < -1:
            raise StateError(f"{key} must be an integer >= -1")
    active = _validate_policy(state["active_policy"], "active_policy")
    if active["version"] != next_round:
        raise StateError("active policy version must equal next_macro_round")
    for key in (
        "collection_manifest_sha256",
        "replay_snapshot_sha256",
        "evaluation_sha256",
        "last_promotion_sha256",
    ):
        if state[key] is not None:
            _require_sha(state[key], key)
    if state["candidate_policy"] is not None:
        _validate_policy(state["candidate_policy"], "candidate_policy", candidate=True)

    phase = state["phase"]
    if phase == "READY":
        if open_round is not None:
            raise StateError("READY state cannot have an open macro-round")
    else:
        if open_round != next_round:
            raise StateError(
                "open_macro_round must equal next_macro_round until promotion"
            )
    baseline = next_round - 1
    reached = {
        "last_collected_macro_round": phase
        in {"COLLECTED", "INGESTED", "TRAINED", "EVALUATED"},
        "last_ingested_macro_round": phase in {"INGESTED", "TRAINED", "EVALUATED"},
        "last_trained_macro_round": phase in {"TRAINED", "EVALUATED"},
        "last_evaluated_macro_round": phase == "EVALUATED",
    }
    for key, current_round_reached in reached.items():
        expected = next_round if current_round_reached else baseline
        if state[key] != expected:
            raise StateError(f"{key} is inconsistent with coordinator phase")
    expected_evidence = {
        "collection_manifest_sha256": reached["last_collected_macro_round"],
        "replay_snapshot_sha256": reached["last_ingested_macro_round"],
        "candidate_policy": reached["last_trained_macro_round"],
        "evaluation_sha256": reached["last_evaluated_macro_round"],
    }
    if phase == "READY" and next_round > 0:
        expected_evidence = {key: True for key in expected_evidence}
    for key, required in expected_evidence.items():
        if required != (state[key] is not None):
            raise StateError(f"{key} presence is inconsistent with coordinator phase")
    if (next_round == 0) != (state["last_promotion_sha256"] is None):
        raise StateError("last_promotion_sha256 is inconsistent with policy version")
    return state


def _load_state_unlocked(root: Path) -> dict[str, Any]:
    campaign = _validate_campaign(load_strict_json(root / "campaign.json"))
    return _validate_state(
        load_strict_json(root / "coordinator_state.json"),
        campaign,
    )


def load_state(root: str | Path) -> dict[str, Any]:
    return _load_state_unlocked(_campaign_root(root))


def _replace_state(
    root: Path, state: dict[str, Any], expected_revision: int
) -> dict[str, Any]:
    with _state_lock(root):
        current = _load_state_unlocked(root)
        if current["revision"] != expected_revision:
            raise StateError(
                f"coordinator compare-and-swap failed: expected revision {expected_revision}, "
                f"found {current['revision']}"
            )
        updated = dict(state)
        updated["revision"] = expected_revision + 1
        campaign = _validate_campaign(load_strict_json(root / "campaign.json"))
        _validate_state(updated, campaign)
        _atomic_replace_json(root / "coordinator_state.json", updated)
        return updated


_ROUND_KEYS = {
    "schema",
    "campaign_id",
    "campaign_sha256",
    "macro_round",
    "policy_version",
    "policy_checkpoint_sha256",
    "policy_checkpoint_path",
    "policy_runtime_sha256",
    "policy_spec_sha256",
    "policy_bundle_manifest_sha256",
    "policy_manifest_sha256",
    "policy_code_sha256",
    "policy_config_sha256",
    "beta",
    "beta_sha256",
    "beta_schedule_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "camera_mapping_sha256",
    "training_config_sha256",
    "steps_per_environment",
    "view_names",
    "policy_image_shape",
    "action_dim",
    "cohorts",
}
_PRODUCTION_ROUND_KEYS = _ROUND_KEYS | {
    "data_lineage",
    "data_lineage_sha256",
}


def _validate_round_contract(
    document: Any, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    production = campaign["schema"] == PRODUCTION_CAMPAIGN_SCHEMA
    expected_schema = (
        PRODUCTION_ROUND_CONTRACT_SCHEMA if production else ROUND_CONTRACT_SCHEMA
    )
    contract = _require_exact(
        document,
        _PRODUCTION_ROUND_KEYS if production else _ROUND_KEYS,
        "macro-round contract",
    )
    if contract["schema"] != expected_schema:
        raise StateError("macro-round contract schema mismatch")
    if contract["campaign_id"] != campaign["campaign_id"] or contract[
        "campaign_sha256"
    ] != canonical_json_sha256(campaign):
        raise StateError("macro-round contract belongs to another campaign")
    macro_round = _require_int(contract["macro_round"], "macro_round")
    _require_int(contract["policy_version"], "policy_version")
    for key in (
        "policy_checkpoint_sha256",
        "policy_runtime_sha256",
        "policy_spec_sha256",
        "policy_bundle_manifest_sha256",
        "policy_manifest_sha256",
        "policy_code_sha256",
        "policy_config_sha256",
        "beta_sha256",
        "beta_schedule_sha256",
        "teacher_sha256",
        "renderer_sha256",
        "camera_mapping_sha256",
        "training_config_sha256",
    ):
        _require_sha(contract[key], key)
    _safe_relative(contract["policy_checkpoint_path"], "policy_checkpoint_path")
    beta = float(contract["beta"])
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise StateError("round beta must be finite and in [0,1]")
    if canonical_json_sha256({"beta": beta}) != contract["beta_sha256"]:
        raise StateError("round beta digest mismatch")
    for key in (
        "beta_schedule_sha256",
        "teacher_sha256",
        "renderer_sha256",
        "camera_mapping_sha256",
        "training_config_sha256",
    ):
        campaign_key = key
        if contract[key] != campaign[campaign_key]:
            raise StateError(f"macro-round {key} does not match campaign")
    if contract["steps_per_environment"] != campaign["steps_per_environment"]:
        raise StateError("macro-round step count does not match campaign")
    if (
        contract["view_names"] != campaign["view_names"]
        or contract["policy_image_shape"] != campaign["policy_image_shape"]
        or contract["action_dim"] != campaign["action_dim"]
    ):
        raise StateError("macro-round observation/action contract mismatch")
    if production:
        try:
            lineage = validate_data_lineage(contract["data_lineage"])
            lineage_sha = data_lineage_sha256(lineage)
        except ProductionDataLineageError as exc:
            raise StateError(f"macro-round data lineage is invalid: {exc}") from exc
        if (
            lineage != campaign["data_lineage"]
            or contract["data_lineage_sha256"] != lineage_sha
            or lineage_sha != campaign["data_lineage_sha256"]
        ):
            raise StateError("macro-round data lineage differs from campaign")
    cohorts = contract["cohorts"]
    if not isinstance(cohorts, list) or len(cohorts) != campaign["cohort_count"]:
        raise StateError("macro-round cohort count mismatch")
    for index, raw in enumerate(cohorts):
        base_cohort = {
            "cohort_index": index,
            "environment_start": index * campaign["active_environments"],
            "environment_count": campaign["active_environments"],
        }
        cohort_keys = set(base_cohort)
        if production:
            cohort_keys |= {"motion_assignment", "motion_assignment_sha256"}
        cohort = _require_exact(raw, cohort_keys, f"cohorts[{index}]")
        if {key: cohort[key] for key in base_cohort} != base_cohort:
            raise StateError(
                "macro-round cohorts do not exactly partition environments"
            )
        if production:
            try:
                assignment = validate_cohort_assignment(
                    cohort["motion_assignment"],
                    data_lineage=campaign["data_lineage"],
                    cohort_index=index,
                    expected_ranks=[0, 1],
                )
            except ProductionDataLineageError as exc:
                raise StateError(
                    f"macro-round cohort assignment is invalid: {exc}"
                ) from exc
            if cohort["motion_assignment_sha256"] != canonical_json_sha256(
                assignment
            ):
                raise StateError("macro-round cohort assignment digest mismatch")
    if macro_round < 0:  # pragma: no cover - _require_int already rejects
        raise StateError("invalid macro-round")
    return contract


def round_directory(root: str | Path, macro_round: int) -> Path:
    return (
        _campaign_root(root)
        / "rounds"
        / f"macro_round_{_require_int(macro_round, 'macro_round'):06d}"
    )


def load_round_contract(root: str | Path, macro_round: int) -> dict[str, Any]:
    root = _campaign_root(root)
    campaign = load_campaign(root)
    return _validate_round_contract(
        load_strict_json(round_directory(root, macro_round) / "contract.json"), campaign
    )


def open_macro_round(
    root: str | Path, *, expected_revision: int | None = None
) -> dict[str, Any]:
    """Freeze the active policy and beta across every cohort in one macro-round."""

    root = _campaign_root(root)
    campaign = load_campaign(root)
    state = load_state(root)
    if expected_revision is not None and state["revision"] != expected_revision:
        raise StateError("coordinator revision changed before opening macro-round")
    if state["phase"] != "READY" or state["open_macro_round"] is not None:
        raise StateError("a macro-round can only open from READY")
    macro_round = state["next_macro_round"]
    macro_round_count = campaign.get("macro_round_count")
    if macro_round_count is not None and macro_round >= macro_round_count:
        raise StateError(
            "bounded campaign is complete; no additional macro-round may open"
        )
    active = dict(state["active_policy"])
    checkpoint, _ = _internal_path(root, active["checkpoint_path"], "active checkpoint")
    if sha256_file(checkpoint) != active["checkpoint_sha256"]:
        raise StateError("active checkpoint is missing or corrupt")
    policy_spec = Path(f"{checkpoint}.policy_spec.json")
    policy_bundle_manifest = Path(f"{checkpoint}.manifest.json")
    if (
        sha256_file(policy_spec) != active["policy_spec_sha256"]
        or sha256_file(policy_bundle_manifest)
        != active["policy_bundle_manifest_sha256"]
    ):
        raise StateError("active policy sidecar is missing or corrupt")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=policy_spec,
        manifest=policy_bundle_manifest,
        expected_checkpoint_sha256=active["checkpoint_sha256"],
        expected_runtime_policy_sha256=active["runtime_policy_sha256"],
    )
    beta = beta_for_macro_round(campaign["beta_schedule"], macro_round)
    production = campaign["schema"] == PRODUCTION_CAMPAIGN_SCHEMA
    contract = {
        "schema": (
            PRODUCTION_ROUND_CONTRACT_SCHEMA
            if production
            else ROUND_CONTRACT_SCHEMA
        ),
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": canonical_json_sha256(campaign),
        "macro_round": macro_round,
        "policy_version": active["version"],
        "policy_checkpoint_sha256": active["checkpoint_sha256"],
        "policy_checkpoint_path": active["checkpoint_path"],
        "policy_runtime_sha256": active["runtime_policy_sha256"],
        "policy_spec_sha256": active["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": active[
            "policy_bundle_manifest_sha256"
        ],
        "policy_manifest_sha256": active["manifest_sha256"],
        "policy_code_sha256": active["code_sha256"],
        "policy_config_sha256": active["policy_config_sha256"],
        "beta": beta,
        "beta_sha256": canonical_json_sha256({"beta": beta}),
        "beta_schedule_sha256": campaign["beta_schedule_sha256"],
        "teacher_sha256": campaign["teacher_sha256"],
        "renderer_sha256": campaign["renderer_sha256"],
        "camera_mapping_sha256": campaign["camera_mapping_sha256"],
        "training_config_sha256": campaign["training_config_sha256"],
        "steps_per_environment": campaign["steps_per_environment"],
        "view_names": campaign["view_names"],
        "policy_image_shape": campaign["policy_image_shape"],
        "action_dim": campaign["action_dim"],
        "cohorts": [],
    }
    if production:
        contract["data_lineage"] = campaign["data_lineage"]
        contract["data_lineage_sha256"] = campaign["data_lineage_sha256"]
    for index in range(campaign["cohort_count"]):
        cohort = {
                "cohort_index": index,
                "environment_start": index * campaign["active_environments"],
                "environment_count": campaign["active_environments"],
        }
        if production:
            assignment = build_cohort_assignment(
                campaign["data_lineage"],
                cohort_index=index,
                expected_ranks=[0, 1],
            )
            cohort["motion_assignment"] = assignment
            cohort["motion_assignment_sha256"] = canonical_json_sha256(
                assignment
            )
        contract["cohorts"].append(cohort)
    _validate_round_contract(contract, campaign)
    publish_immutable_json(
        round_directory(root, macro_round) / "contract.json", contract
    )
    updated = dict(state)
    updated.update(
        phase="COLLECTING",
        open_macro_round=macro_round,
        collection_manifest_sha256=None,
        replay_snapshot_sha256=None,
        candidate_policy=None,
        evaluation_sha256=None,
    )
    _replace_state(root, updated, state["revision"])
    return contract


# Compatibility spelling; persisted fields and contract use ``macro_round``.
open_round = open_macro_round


_COLLECTION_ENTRY_KEYS = {
    "cohort_index",
    "cohort_manifest_path",
    "cohort_manifest_sha256",
    "transition_count",
}
_COLLECTION_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "round_contract_path",
    "round_contract_sha256",
    "runtime_policy_sha256",
    "cohort_count",
    "transition_count",
    "cohorts",
}


def _expected_view_contract(campaign: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": campaign["policy_image_shape"],
        "views": campaign["view_names"],
    }


def _expected_action_contract(campaign: Mapping[str, Any]) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [campaign["action_dim"]]}


def _verified_collection_cohorts(
    root: Path,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], str]:
    if len(paths) != campaign["cohort_count"]:
        raise StateError(
            "collection must contain exactly campaign.cohort_count manifests"
        )
    reports: dict[int, tuple[dict[str, Any], str]] = {}
    runtime_sha: str | None = None
    for raw_path in paths:
        path, relative = _internal_path(root, raw_path, "cohort manifest")
        report = verify_cohort(path)
        index = _require_int(report.get("cohort_index"), "cohort_index")
        if index in reports:
            raise StateError("collection contains a duplicate cohort_index")
        if (
            report.get("campaign_id") != campaign["campaign_id"]
            or report.get("macro_round") != contract["macro_round"]
            or report.get("purpose") != "train"
        ):
            raise StateError("cohort campaign/macro_round/purpose mismatch")
        expected = (
            contract["cohorts"][index] if index < len(contract["cohorts"]) else None
        )
        if expected is None:
            raise StateError("cohort_index is outside macro-round contract")
        expected_envs = list(
            range(
                expected["environment_start"],
                expected["environment_start"] + expected["environment_count"],
            )
        )
        expected_transitions = (
            expected["environment_count"] * contract["steps_per_environment"]
        )
        if (
            report.get("expected_global_env_ids") != expected_envs
            or report.get("global_env_ids") != expected_envs
        ):
            raise StateError(
                "cohort does not cover its exact logical environment partition"
            )
        if (
            report.get("expected_transitions") != expected_transitions
            or report.get("transition_count") != expected_transitions
        ):
            raise StateError("cohort transition count mismatch")
        expected_digests = {
            "policy_sha256": contract["policy_checkpoint_sha256"],
            "runtime_policy_sha256": contract["policy_runtime_sha256"],
            "teacher_sha256": contract["teacher_sha256"],
            "renderer_sha256": contract["renderer_sha256"],
            "beta_sha256": contract["beta_sha256"],
            "config_sha256": contract["training_config_sha256"],
        }
        if any(report.get(key) != value for key, value in expected_digests.items()):
            raise StateError(
                "cohort policy/teacher/renderer/beta/config provenance mismatch"
            )
        if report.get("view_contract") != _expected_view_contract(
            campaign
        ) or report.get("action_contract") != _expected_action_contract(campaign):
            raise StateError("cohort view/action contract mismatch")
        current_runtime = _require_sha(
            report.get("runtime_policy_sha256"), "runtime_policy_sha256"
        )
        if runtime_sha is None:
            runtime_sha = current_runtime
        elif current_runtime != runtime_sha:
            raise StateError("cohorts used different runtime policy parameters")
        reports[index] = (report, relative)
    if set(reports) != set(range(campaign["cohort_count"])):
        raise StateError("collection cohort indices are not exactly 0..cohort_count-1")
    seed_provenance = [
        "actor_seed_contract" in reports[index][0]
        for index in range(campaign["cohort_count"])
    ]
    if any(seed_provenance) and not all(seed_provenance):
        raise StateError(
            "collection mixes cohorts with and without actor seed provenance"
        )
    if all(seed_provenance):
        simulator_seeds: dict[int, tuple[int, int]] = {}
        generation_seeds: dict[int, tuple[int, int]] = {}
        trajectory_digests: dict[str, tuple[int, str]] = {}
        for index in range(campaign["cohort_count"]):
            report = reports[index][0]
            seed_contract = report["actor_seed_contract"]
            for assignment in seed_contract["assignments"]:
                rank = int(assignment["rank"])
                simulator_seed = int(assignment["simulator_seed"])
                generation_seed = int(assignment["generation_seed"])
                if simulator_seed in simulator_seeds:
                    previous = simulator_seeds[simulator_seed]
                    raise StateError(
                        "collection reuses simulator seed "
                        f"{simulator_seed} across cohort/rank {previous} and {(index, rank)}"
                    )
                if generation_seed in generation_seeds:
                    previous = generation_seeds[generation_seed]
                    raise StateError(
                        "collection reuses Image3F generation seed "
                        f"{generation_seed} across cohort/rank {previous} and {(index, rank)}"
                    )
                simulator_seeds[simulator_seed] = (index, rank)
                generation_seeds[generation_seed] = (index, rank)
            fingerprints = report[
                "physical_teacher_trajectory_sha256_by_global_env"
            ]
            for global_env_id, digest in fingerprints.items():
                if digest in trajectory_digests:
                    previous = trajectory_digests[digest]
                    raise StateError(
                        "collection contains a repeated physical/teacher trajectory "
                        f"for cohort/env {previous} and {(index, global_env_id)}"
                    )
                trajectory_digests[digest] = (index, global_env_id)
    entries = [
        {
            "cohort_index": index,
            "cohort_manifest_path": reports[index][1],
            "cohort_manifest_sha256": reports[index][0]["cohort_sha256"],
            "transition_count": reports[index][0]["transition_count"],
        }
        for index in range(campaign["cohort_count"])
    ]
    assert runtime_sha is not None
    return entries, runtime_sha


def seal_collection(
    root: str | Path, *, macro_round: int, cohort_manifests: Sequence[str | Path]
) -> dict[str, Any]:
    """Verify all cohorts and immutably seal one complete macro-round."""

    root = _campaign_root(root)
    campaign = load_campaign(root)
    state = load_state(root)
    macro_round = _require_int(macro_round, "macro_round")
    if state["phase"] != "COLLECTING" or state["open_macro_round"] != macro_round:
        raise StateError("collection sealing requires the open COLLECTING macro-round")
    contract_path = round_directory(root, macro_round) / "contract.json"
    contract = load_round_contract(root, macro_round)
    entries, runtime_sha = _verified_collection_cohorts(
        root, campaign, contract, cohort_manifests
    )
    document = {
        "schema": COLLECTION_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "macro_round": macro_round,
        "round_contract_path": contract_path.relative_to(root).as_posix(),
        "round_contract_sha256": sha256_file(contract_path),
        "runtime_policy_sha256": runtime_sha,
        "cohort_count": len(entries),
        "transition_count": sum(item["transition_count"] for item in entries),
        "cohorts": entries,
    }
    path = round_directory(root, macro_round) / "collection.json"
    file_sha = publish_immutable_json(path, document)
    return {
        **document,
        "manifest_path": path.relative_to(root).as_posix(),
        "manifest_sha256": file_sha,
    }


def load_collection_manifest(
    root: str | Path, path: str | Path, *, expected_macro_round: int | None = None
) -> dict[str, Any]:
    """Load a collection seal and reverify its contract and every cohort."""

    root = _campaign_root(root)
    manifest_path, relative = _internal_path(root, path, "collection manifest")
    document = _require_exact(
        load_strict_json(manifest_path), _COLLECTION_KEYS, "collection manifest"
    )
    if document["schema"] != COLLECTION_SCHEMA:
        raise StateError("collection manifest schema mismatch")
    campaign = load_campaign(root)
    if document["campaign_id"] != campaign["campaign_id"]:
        raise StateError("collection manifest belongs to another campaign")
    macro_round = _require_int(document["macro_round"], "macro_round")
    if expected_macro_round is not None and macro_round != expected_macro_round:
        raise StateError("collection manifest macro_round mismatch")
    contract = load_round_contract(root, macro_round)
    contract_path, contract_relative = _internal_path(
        root, document["round_contract_path"], "round contract path"
    )
    if (
        contract_relative
        != round_directory(root, macro_round)
        .joinpath("contract.json")
        .relative_to(root)
        .as_posix()
    ):
        raise StateError("collection points at a non-canonical round contract")
    if sha256_file(contract_path) != _require_sha(
        document["round_contract_sha256"], "round_contract_sha256"
    ):
        raise StateError("collection round contract SHA mismatch")
    cohorts = document["cohorts"]
    if not isinstance(cohorts, list):
        raise StateError("collection cohorts must be a list")
    paths: list[str] = []
    for index, raw in enumerate(cohorts):
        entry = _require_exact(
            raw, _COLLECTION_ENTRY_KEYS, f"collection.cohorts[{index}]"
        )
        if entry["cohort_index"] != index:
            raise StateError(
                "collection cohorts must be sorted exactly by cohort_index"
            )
        _require_sha(entry["cohort_manifest_sha256"], "cohort_manifest_sha256")
        _require_int(entry["transition_count"], "transition_count", minimum=1)
        paths.append(
            _safe_relative(entry["cohort_manifest_path"], "cohort_manifest_path")
        )
    verified, runtime_sha = _verified_collection_cohorts(
        root, campaign, contract, paths
    )
    if cohorts != verified:
        raise StateError("collection entries differ from reverified cohort artifacts")
    if document["runtime_policy_sha256"] != runtime_sha:
        raise StateError("collection runtime policy digest mismatch")
    if document["cohort_count"] != campaign["cohort_count"] or document[
        "cohort_count"
    ] != len(cohorts):
        raise StateError("collection cohort_count mismatch")
    transitions = sum(item["transition_count"] for item in cohorts)
    if document["transition_count"] != transitions:
        raise StateError("collection transition_count mismatch")
    return {
        **document,
        "manifest_path": relative,
        "manifest_sha256": sha256_file(manifest_path),
    }


def transition(
    root: str | Path,
    *,
    expected_phase: str,
    next_phase: str,
    macro_round: int,
    evidence_path: str | Path,
    evidence_sha256: str | None = None,
    candidate_policy: Mapping[str, Any] | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Advance one phase after revalidating the exact immutable evidence."""

    allowed = {
        ("COLLECTING", "COLLECTED"),
        ("COLLECTED", "INGESTED"),
        ("INGESTED", "TRAINED"),
        ("TRAINED", "EVALUATED"),
    }
    if (expected_phase, next_phase) not in allowed:
        raise StateError(
            f"unsupported coordinator transition: {expected_phase} -> {next_phase}"
        )
    root = _campaign_root(root)
    state = load_state(root)
    macro_round = _require_int(macro_round, "macro_round")
    if expected_revision is not None and state["revision"] != expected_revision:
        raise StateError("coordinator revision changed before transition")
    if state["phase"] != expected_phase or state["open_macro_round"] != macro_round:
        raise StateError(
            f"cannot transition macro-round {macro_round}: phase/open macro-round is "
            f"{state['phase']}/{state['open_macro_round']}"
        )
    artifact_path, relative = _internal_path(root, evidence_path, "transition evidence")
    actual_sha = sha256_file(artifact_path)
    if evidence_sha256 is not None and actual_sha != _require_sha(
        evidence_sha256, "evidence_sha256"
    ):
        raise StateError("transition evidence SHA mismatch")
    updated = dict(state)
    updated["phase"] = next_phase

    if next_phase == "COLLECTED":
        report = load_collection_manifest(
            root, artifact_path, expected_macro_round=macro_round
        )
        if report["manifest_sha256"] != actual_sha:
            raise StateError("collection evidence digest mismatch")
        updated["last_collected_macro_round"] = macro_round
        updated["collection_manifest_sha256"] = actual_sha
    elif next_phase == "INGESTED":
        if state["last_collected_macro_round"] != macro_round:
            raise StateError("cannot ingest before collection")
        from round_dagger.replay import load_snapshot

        snapshot = load_snapshot(artifact_path, expected_file_sha256=actual_sha)
        campaign = load_campaign(root)
        if (
            snapshot.get("campaign_id") != campaign["campaign_id"]
            or snapshot.get("through_macro_round") != macro_round
        ):
            raise StateError("replay snapshot campaign/macro_round mismatch")
        if snapshot.get("expected_train_cohorts") != campaign["cohort_count"]:
            raise StateError("replay snapshot cohort contract mismatch")
        updated["last_ingested_macro_round"] = macro_round
        updated["replay_snapshot_sha256"] = actual_sha
    elif next_phase == "TRAINED":
        if state["last_ingested_macro_round"] != macro_round:
            raise StateError("cannot train before replay ingestion")
        candidate = _validate_policy(
            candidate_policy, "candidate_policy", candidate=True
        )
        manifest = _require_exact(
            load_strict_json(artifact_path),
            _CANDIDATE_MANIFEST_KEYS,
            "candidate manifest",
        )
        if manifest["schema"] != CANDIDATE_SCHEMA:
            raise StateError("training evidence is not a candidate manifest")
        if (
            canonical_json_sha256(manifest) != actual_sha
            or candidate["manifest_sha256"] != actual_sha
        ):
            raise StateError("candidate manifest digest mismatch")
        expected_values = {
            "campaign_id": load_campaign(root)["campaign_id"],
            "campaign_sha256": state["campaign_sha256"],
            "macro_round": macro_round,
            "policy_version": state["active_policy"]["version"] + 1,
            "checkpoint_path": candidate["checkpoint_path"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "runtime_policy_sha256": candidate["runtime_policy_sha256"],
            "policy_spec_sha256": candidate["policy_spec_sha256"],
            "policy_bundle_manifest_sha256": candidate[
                "policy_bundle_manifest_sha256"
            ],
            "parent_policy_version": state["active_policy"]["version"],
            "parent_checkpoint_sha256": state["active_policy"]["checkpoint_sha256"],
            "replay_snapshot_sha256": state["replay_snapshot_sha256"],
            "code_sha256": candidate["code_sha256"],
            "learner_config_sha256": candidate["policy_config_sha256"],
            "campaign_training_config_sha256": load_campaign(root)[
                "training_config_sha256"
            ],
            "teacher_sha256": load_campaign(root)["teacher_sha256"],
            "renderer_sha256": load_campaign(root)["renderer_sha256"],
        }
        if any(manifest.get(key) != value for key, value in expected_values.items()):
            raise StateError("candidate manifest/state lineage mismatch")
        checkpoint, _ = _internal_path(
            root, candidate["checkpoint_path"], "candidate checkpoint"
        )
        spec = Path(f"{checkpoint}.policy_spec.json")
        bundle_manifest = Path(f"{checkpoint}.manifest.json")
        if (
            manifest["policy_spec_path"] != spec.relative_to(root).as_posix()
            or manifest["policy_bundle_manifest_path"]
            != bundle_manifest.relative_to(root).as_posix()
            or sha256_file(checkpoint) != candidate["checkpoint_sha256"]
            or sha256_file(spec) != candidate["policy_spec_sha256"]
            or sha256_file(bundle_manifest)
            != candidate["policy_bundle_manifest_sha256"]
        ):
            raise StateError("candidate policy bundle is missing or non-canonical")
        validate_policy_bundle_identity(
            checkpoint=checkpoint,
            policy_spec=spec,
            manifest=bundle_manifest,
            expected_checkpoint_sha256=candidate["checkpoint_sha256"],
            expected_runtime_policy_sha256=candidate["runtime_policy_sha256"],
        )
        _require_int(manifest["optimizer_steps"], "optimizer_steps", minimum=1)
        if (
            candidate["macro_round"] != macro_round
            or candidate["manifest_path"] != relative
        ):
            raise StateError("candidate state summary path/macro_round mismatch")
        updated["last_trained_macro_round"] = macro_round
        updated["candidate_policy"] = candidate
    else:  # EVALUATED
        if state["last_trained_macro_round"] != macro_round:
            raise StateError("cannot evaluate before training")
        evidence = _require_exact(
            load_strict_json(artifact_path),
            _EVALUATION_MANIFEST_KEYS,
            "evaluation manifest",
        )
        if evidence["schema"] != EVALUATION_SCHEMA:
            raise StateError("evaluation evidence schema mismatch")
        candidate = state["candidate_policy"]
        expected = {
            "campaign_id": load_campaign(root)["campaign_id"],
            "campaign_sha256": state["campaign_sha256"],
            "macro_round": macro_round,
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "policy_version": candidate["version"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise StateError("evaluation/candidate lineage mismatch")
        _require_sha(evidence["evaluator_config_sha256"], "evaluator_config_sha256")
        _require_sha(evidence["evidence_bundle_sha256"], "evidence_bundle_sha256")
        if not isinstance(evidence["evidence_bundle_path"], str):
            raise StateError("evaluation evidence bundle path is unsafe")
        bundle_path, bundle_relative = _internal_path(
            root, evidence["evidence_bundle_path"], "evaluation evidence bundle"
        )
        canonical_bundle = (
            root
            / "rounds"
            / f"macro_round_{macro_round:06d}"
            / "evaluation"
            / "results"
            / "evaluation_bundle.json"
        )
        recovery_parent = bundle_path.parent.parent
        recovery_bundle = (
            recovery_parent.parent.name == "evaluation_recovery"
            and bool(_EVALUATION_RECOVERY_ID_RE.fullmatch(recovery_parent.name))
            and bundle_path
            == recovery_parent / "results" / "evaluation_bundle.json"
            and recovery_parent.parent.parent
            == root / "rounds" / f"macro_round_{macro_round:06d}"
        )
        if (
            (bundle_path != canonical_bundle and not recovery_bundle)
            or evidence["evidence_bundle_path"] != bundle_relative
        ):
            raise StateError("evaluation evidence bundle path is non-canonical")
        if sha256_file(bundle_path) != evidence["evidence_bundle_sha256"]:
            raise StateError("evaluation evidence bundle SHA mismatch")
        if type(evidence["passed"]) is not bool or not isinstance(
            evidence["metrics"], Mapping
        ):
            raise StateError("evaluation passed/metrics fields are malformed")
        for name, value in evidence["metrics"].items():
            _require_token(name, "evaluation metric name")
            if isinstance(value, bool):
                raise StateError("evaluation metrics must be finite numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise StateError("evaluation metrics must be finite numbers") from exc
            if not math.isfinite(number):
                raise StateError("evaluation metrics must be finite numbers")
        updated["last_evaluated_macro_round"] = macro_round
        updated["evaluation_sha256"] = actual_sha
    return _replace_state(root, updated, state["revision"])


def complete_promotion(
    root: str | Path,
    *,
    macro_round: int,
    checkpoint_path: str,
    promotion_manifest_path: str | Path,
    promotion_manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically advance the active policy after passed evaluation evidence."""

    root = _campaign_root(root)
    state = load_state(root)
    macro_round = _require_int(macro_round, "macro_round")
    if state["phase"] != "EVALUATED" or state["open_macro_round"] != macro_round:
        raise StateError("policy promotion requires the open EVALUATED macro-round")
    candidate = _validate_policy(
        state["candidate_policy"], "candidate_policy", candidate=True
    )
    relative_checkpoint = _safe_relative(checkpoint_path, "promoted checkpoint path")
    if relative_checkpoint != candidate["checkpoint_path"]:
        raise StateError("promoted checkpoint path does not match candidate")
    checkpoint, _ = _internal_path(root, relative_checkpoint, "promoted checkpoint")
    if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
        raise StateError("promoted checkpoint is missing or corrupt")
    policy_spec = Path(f"{checkpoint}.policy_spec.json")
    policy_bundle_manifest = Path(f"{checkpoint}.manifest.json")
    if (
        sha256_file(policy_spec) != candidate["policy_spec_sha256"]
        or sha256_file(policy_bundle_manifest)
        != candidate["policy_bundle_manifest_sha256"]
    ):
        raise StateError("promoted policy sidecar is missing or corrupt")
    validate_policy_bundle_identity(
        checkpoint=checkpoint,
        policy_spec=policy_spec,
        manifest=policy_bundle_manifest,
        expected_checkpoint_sha256=candidate["checkpoint_sha256"],
        expected_runtime_policy_sha256=candidate["runtime_policy_sha256"],
    )
    promotion_path, _ = _internal_path(
        root, promotion_manifest_path, "promotion manifest"
    )
    actual_promotion_sha = sha256_file(promotion_path)
    if actual_promotion_sha != _require_sha(
        promotion_manifest_sha256, "promotion manifest sha256"
    ):
        raise StateError("promotion manifest SHA mismatch")
    promotion = _require_exact(
        load_strict_json(promotion_path),
        _PROMOTION_MANIFEST_KEYS,
        "promotion manifest",
    )
    campaign = load_campaign(root)
    expected_promotion = {
        "schema": PROMOTION_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": state["campaign_sha256"],
        "macro_round": macro_round,
        "parent_policy_version": candidate["parent_policy_version"],
        "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "evaluation_sha256": state["evaluation_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "policy_version": candidate["version"],
    }
    if promotion != expected_promotion:
        raise StateError("promotion manifest does not bind coordinator evidence")
    updated = dict(state)
    updated["active_policy"] = {
        "version": candidate["version"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_path": candidate["checkpoint_path"],
        "runtime_policy_sha256": candidate["runtime_policy_sha256"],
        "policy_spec_sha256": candidate["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": candidate[
            "policy_bundle_manifest_sha256"
        ],
        "manifest_sha256": candidate["manifest_sha256"],
        "code_sha256": candidate["code_sha256"],
        "policy_config_sha256": candidate["policy_config_sha256"],
    }
    updated["phase"] = "READY"
    updated["next_macro_round"] = macro_round + 1
    updated["open_macro_round"] = None
    updated["last_promotion_sha256"] = actual_promotion_sha
    return _replace_state(root, updated, state["revision"])


__all__ = [
    "BOUNDED_CAMPAIGN_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "PRODUCTION_CAMPAIGN_SCHEMA",
    "COLLECTION_SCHEMA",
    "ROUND_CONTRACT_SCHEMA",
    "PRODUCTION_ROUND_CONTRACT_SCHEMA",
    "STATE_SCHEMA",
    "StateError",
    "beta_for_macro_round",
    "beta_for_round",
    "complete_promotion",
    "initialize_campaign",
    "load_campaign",
    "load_collection_manifest",
    "load_round_contract",
    "load_state",
    "open_macro_round",
    "open_round",
    "round_directory",
    "seal_collection",
    "transition",
]
