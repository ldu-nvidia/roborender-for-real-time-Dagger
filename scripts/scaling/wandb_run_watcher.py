#!/usr/bin/env python3
"""Upload DAgger scalar/media artifacts to a rank-zero W&B sidecar run.

Producers write atomically into this layout beneath a training run directory::

    telemetry/scalars/step_000123.json
    telemetry/images/step_000123_generated_triplet.png
    telemetry/videos/step_000123_rollout.mp4

Scalar JSON uses ``{"step": 123, "metrics": {"loss/name": 0.1}}``.  Flat
numeric mappings with a ``step`` field are also accepted.  This watcher never
reads arbitrary files outside those directories, ignores symlinks and partial
files, and records fingerprints so a restart does not upload unchanged media.

The sidecar uses ``trainer_step`` as a custom W&B x-axis rather than the W&B
internal step.  Late-arriving media can therefore be uploaded safely without
colliding with the trainer's scalar-history step.  Artifact fingerprints are
committed only after the W&B sender finishes successfully, so a crash cannot
turn a merely queued upload into a permanently skipped artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    from scripts.scaling.wandb_preflight import (
        WandbPreflightError,
        _load_wandb,
        query_identity,
    )
except ModuleNotFoundError:  # Support direct execution by absolute script path.
    from wandb_preflight import WandbPreflightError, _load_wandb, query_identity


STEP_RE = re.compile(r"(?:^|[_-])step[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
STATE_SCHEMA_VERSION = 2
INITIALIZED_MARKER_SCHEMA = "fast_roborender.wandb_sidecar_initialized.v2"
DELIVERY_RECEIPT_SCHEMA = "fast_roborender.wandb_sidecar_delivery_receipt.v1"
PUBLIC_CONFIG_SCHEMA = "fast_roborender.round_dagger.wandb_public_config.v2"
PRODUCTION_PUBLIC_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.wandb_public_config.v3"
)
PUBLIC_CONFIG_KEYS = {
    "round_dagger/campaign_id",
    "round_dagger/macro_round",
    "round_dagger/cohort_index",
    "round_dagger/cohort_count",
    "round_dagger/environments_per_cohort",
    "round_dagger/target_environments",
    "round_dagger/steps_per_environment",
    "round_dagger/transitions_per_cohort",
    "round_dagger/transitions_per_macro_round",
    "round_dagger/macro_round_count",
    "round_dagger/transitions_per_campaign",
    "round_dagger/policy_version",
    "round_dagger/policy_checkpoint_sha256",
    "round_dagger/runtime_policy_sha256",
    "round_dagger/teacher_sha256",
    "round_dagger/renderer_sha256",
    "round_dagger/camera_mapping_sha256",
    "round_dagger/training_config_sha256",
    "round_dagger/beta",
    "round_dagger/beta_sha256",
    "round_dagger/orchestration_config_sha256",
}
PRODUCTION_PUBLIC_CONFIG_KEYS = PUBLIC_CONFIG_KEYS | {
    "round_dagger/campaign_contract_sha256",
    "round_dagger/data_lineage_sha256",
}
PUBLIC_CONFIG_SHA_KEYS = {
    key for key in PRODUCTION_PUBLIC_CONFIG_KEYS if key.endswith("_sha256")
}
PUBLIC_CONFIG_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_CONFIG_EXACT_INTS = {
    "round_dagger/cohort_count": 4,
    "round_dagger/environments_per_cohort": 64,
    "round_dagger/target_environments": 256,
    "round_dagger/steps_per_environment": 512,
    "round_dagger/transitions_per_cohort": 64 * 512,
    "round_dagger/transitions_per_macro_round": 256 * 512,
    "round_dagger/macro_round_count": 5,
    "round_dagger/transitions_per_campaign": 5 * 256 * 512,
}
MEDIA_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".webm", ".mov"},
}
TELEMETRY_SCOPES = {
    "production": "dagger-observability-sidecar",
    "bounded-validation": "dagger-bounded-validation-sidecar",
}


class ArtifactError(RuntimeError):
    """An artifact did not satisfy the telemetry contract."""


def _strict_public_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate public W&B config key {key!r}")
        result[key] = value
    return result


def load_public_config(path: Path) -> dict[str, str | int | float]:
    """Load the explicit non-secret config projection emitted by orchestration.

    Arbitrary user mappings are intentionally unsupported: accepting only the
    round contract's fixed allowlist prevents credentials, hook argv, paths, or
    inherited environment values from being copied into W&B config.
    """

    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactError(
                    "public W&B config must be a regular non-symlink file"
                )
            if info.st_size > 64 * 1024:
                raise ArtifactError("public W&B config exceeds 64 KiB")
            payload = handle.read(64 * 1024 + 1)
        document = json.loads(
            payload,
            object_pairs_hook=_strict_public_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactError(f"non-finite JSON constant {value}")
            ),
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"cannot load public W&B config ({type(exc).__name__})"
        ) from None
    if not isinstance(document, dict) or set(document) != {"schema", "config"}:
        raise ArtifactError("public W&B config wrapper has unexpected fields")
    schema = document["schema"]
    if schema not in {PUBLIC_CONFIG_SCHEMA, PRODUCTION_PUBLIC_CONFIG_SCHEMA}:
        raise ArtifactError("public W&B config schema mismatch")
    expected_keys = (
        PRODUCTION_PUBLIC_CONFIG_KEYS
        if schema == PRODUCTION_PUBLIC_CONFIG_SCHEMA
        else PUBLIC_CONFIG_KEYS
    )
    values = document["config"]
    if not isinstance(values, dict) or set(values) != expected_keys:
        actual = set(values) if isinstance(values, dict) else set()
        raise ArtifactError(
            "public W&B config keys differ from the non-secret allowlist: "
            f"missing={sorted(expected_keys - actual)} "
            f"unknown={sorted(actual - expected_keys)}"
        )

    clean: dict[str, str | int | float] = {}
    for key, value in values.items():
        if key in PUBLIC_CONFIG_SHA_KEYS:
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise ArtifactError(f"public W&B config {key} is not a SHA-256")
            clean[key] = value
        elif key == "round_dagger/campaign_id":
            if not isinstance(value, str) or PUBLIC_CONFIG_TOKEN_RE.fullmatch(value) is None:
                raise ArtifactError("public W&B campaign_id is not a safe token")
            clean[key] = value
        elif key == "round_dagger/beta":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ArtifactError("public W&B beta must be finite and in [0,1]")
            clean[key] = float(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ArtifactError(f"public W&B config {key} must be an integer")
            if key in PUBLIC_CONFIG_EXACT_INTS and value != PUBLIC_CONFIG_EXACT_INTS[key]:
                raise ArtifactError(f"public W&B config {key} violates production topology")
            if key in {"round_dagger/macro_round", "round_dagger/policy_version"} and value < 0:
                raise ArtifactError(f"public W&B config {key} must be non-negative")
            if key == "round_dagger/cohort_index" and value not in {-1, 0, 1, 2, 3}:
                raise ArtifactError("public W&B cohort_index must be -1 or 0..3")
            clean[key] = value
    return clean


@dataclass(frozen=True)
class Artifact:
    path: Path
    relative_path: str
    kind: str
    step: int
    key: str
    fingerprint: str


def process_rank(environment: Mapping[str, str] | None = None) -> int:
    environment = os.environ if environment is None else environment
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = environment.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                raise ArtifactError(f"{key} must be an integer") from None
    return 0


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _step_from_name(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    if match is None:
        raise ArtifactError(f"media filename has no step_<integer>: {path.name}")
    return int(match.group(1))


def _metric_key(kind: str, path: Path, root: Path) -> str:
    directory = (root / "telemetry" / f"{kind}s").resolve()
    relative = path.resolve().relative_to(directory)
    components = list(relative.with_suffix("").parts)
    leaf = STEP_RE.sub("_", components[-1]).strip("_-") or kind
    components[-1] = leaf
    return "/".join(("media", kind, *components))


def _is_stable(path: Path, now: float, min_age_s: float) -> bool:
    try:
        return not path.is_symlink() and now - path.stat().st_mtime >= min_age_s
    except FileNotFoundError:
        return False


def discover_artifacts(
    run_dir: Path,
    *,
    min_age_s: float = 5.0,
    max_image_bytes: int = 25 * 1024 * 1024,
    max_video_bytes: int = 250 * 1024 * 1024,
    now: float | None = None,
    errors: list[str] | None = None,
) -> list[Artifact]:
    """Discover stable, in-tree image/video artifacts in deterministic order.

    Discovery is strict by default.  A long-running watcher can pass ``errors``
    to quarantine malformed files while continuing to upload healthy telemetry.
    Error messages intentionally contain only the artifact kind, basename, and
    exception type so third-party or filesystem errors cannot leak credentials.
    """

    run_dir = run_dir.resolve()
    telemetry = run_dir / "telemetry"
    now = time.time() if now is None else now
    found: list[Artifact] = []
    for kind, suffixes in MEDIA_SUFFIXES.items():
        directory = telemetry / f"{kind}s"
        if not directory.is_dir():
            continue
        directory_resolved = directory.resolve()
        try:
            directory_resolved.relative_to(run_dir)
        except ValueError:
            # Do not follow a telemetry-directory symlink out of the run tree.
            continue
        size_limit = max_image_bytes if kind == "image" else max_video_bytes
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if not _is_stable(path, now, min_age_s):
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(directory_resolved)
                resolved.relative_to(run_dir)
                if resolved.stat().st_size > size_limit:
                    raise ArtifactError(
                        f"{kind} artifact exceeds the {size_limit}-byte upload limit: "
                        f"{resolved.name}"
                    )
                step = _step_from_name(resolved)
                relative_path = str(resolved.relative_to(run_dir))
                found.append(
                    Artifact(
                        path=resolved,
                        relative_path=relative_path,
                        kind=kind,
                        step=step,
                        key=_metric_key(kind, resolved, run_dir),
                        fingerprint=_fingerprint(resolved),
                    )
                )
            except Exception as exc:
                if errors is None:
                    raise
                errors.append(
                    f"invalid {kind} artifact {path.name} ({type(exc).__name__})"
                )
                continue
    return sorted(found, key=lambda item: (item.step, item.kind, item.relative_path))


def load_scalar_file(path: Path) -> tuple[int, dict[str, int | float]]:
    """Read one scalar artifact and reject non-numeric or non-finite values."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid scalar JSON {path.name}: {type(exc).__name__}") from None
    if not isinstance(raw, dict):
        raise ArtifactError(f"scalar JSON must be an object: {path.name}")
    step_value = raw.get("step", raw.get("trainer_step"))
    if isinstance(step_value, bool) or not isinstance(step_value, int) or step_value < 0:
        raise ArtifactError(f"scalar JSON requires a non-negative integer step: {path.name}")
    metrics = raw.get("metrics")
    if metrics is None:
        metrics = {k: v for k, v in raw.items() if k not in {"step", "trainer_step"}}
    if not isinstance(metrics, dict):
        raise ArtifactError(f"metrics must be an object: {path.name}")

    clean: dict[str, int | float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ArtifactError(f"metric names must be non-empty strings: {path.name}")
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            clean[key] = value
        elif value is not None:
            raise ArtifactError(f"metric {key!r} is not finite numeric data: {path.name}")
    if not clean:
        raise ArtifactError(f"scalar JSON contains no numeric metrics: {path.name}")
    return step_value, clean


def discover_scalars(
    run_dir: Path,
    *,
    min_age_s: float = 1.0,
    now: float | None = None,
    errors: list[str] | None = None,
) -> list[Artifact]:
    run_dir = run_dir.resolve()
    directory = run_dir / "telemetry" / "scalars"
    if not directory.is_dir():
        return []
    directory_resolved = directory.resolve()
    try:
        directory_resolved.relative_to(run_dir)
    except ValueError:
        return []
    now = time.time() if now is None else now
    found = []
    for path in directory.rglob("*.json"):
        if not path.is_file() or not _is_stable(path, now, min_age_s):
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(directory_resolved)
            resolved.relative_to(run_dir)
            step, _ = load_scalar_file(resolved)
            relative_path = str(resolved.relative_to(run_dir))
            found.append(
                Artifact(
                    path=resolved,
                    relative_path=relative_path,
                    kind="scalar",
                    step=step,
                    key="",
                    fingerprint=_fingerprint(resolved),
                )
            )
        except Exception as exc:
            if errors is None:
                raise
            errors.append(
                f"invalid scalar artifact {path.name} ({type(exc).__name__})"
            )
            continue
    return sorted(found, key=lambda item: (item.step, item.relative_path))


def update_final_drain(
    *,
    marker_present: bool,
    deadline: float | None,
    now: float,
    min_age_s: float,
) -> tuple[float | None, bool]:
    """Return the final-drain deadline and whether a post-deadline scan finished.

    The caller invokes this immediately after a scan.  On first observing the
    completion marker, it waits one complete stability window and scans again;
    therefore artifacts written just before the marker cannot be skipped merely
    because they were younger than ``min_age_s`` during the first scan.
    """

    if deadline is None:
        if not marker_present:
            return None, False
        deadline = now + min_age_s
    return deadline, now >= deadline


def load_finish_exit_code(path: Path) -> int:
    """Read the supervisor's terminal result from a safe finish marker.

    Legacy/standalone callers create an empty marker, which remains a success.
    Production actor pairs publish ``0`` or ``2`` atomically after writing
    their immutable terminal evidence and final scalar.  Propagating ``2`` to
    ``run.finish`` makes a failed collection visually terminal in W&B instead
    of leaving a green or apparently live run behind.
    """

    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError("W&B finish marker must be a regular non-symlink file")
        if info.st_size > 8:
            raise ArtifactError("W&B finish marker is oversized")
        payload = path.read_text(encoding="ascii").strip()
    except ArtifactError:
        raise
    except (OSError, UnicodeError):
        raise ArtifactError("W&B finish marker is unreadable") from None
    if payload == "":
        return 0
    if payload not in {"0", "2"}:
        raise ArtifactError("W&B finish marker exit code is invalid")
    return int(payload)


def _validate_run_id(value: str) -> str:
    value = value.strip()
    if not RUN_ID_RE.fullmatch(value):
        raise ArtifactError("W&B sidecar run ID is malformed")
    return value


def load_or_create_run_id(path: Path) -> str:
    """Return one crash-safe, create-only durable W&B run ID.

    The final pathname is never opened for writing.  A deterministic hidden
    staging inode is fsynced and hard-linked into the final name while the
    parent directory lock is held.  SIGKILL can therefore strand only the
    staging name, which a later caller removes without interpreting its bytes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise ArtifactError(
            f"cannot open W&B sidecar run ID directory ({type(exc).__name__})"
        ) from None
    staging_name = f".{path.name}.staging"
    staging_created = False
    locked = False

    def read_existing() -> str | None:
        try:
            info = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ArtifactError(
                f"cannot inspect W&B sidecar run ID file ({type(exc).__name__})"
            ) from None
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactError("W&B sidecar run ID file must not be a symlink")
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError("W&B sidecar run ID must be a regular file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size > 256:
                    raise ArtifactError("W&B sidecar run ID file is unsafe")
                payload = handle.read(257)
            return _validate_run_id(payload.decode("utf-8"))
        except ArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ArtifactError(
                f"cannot read W&B sidecar run ID file ({type(exc).__name__})"
            ) from None

    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        locked = True
        try:
            staging_info = os.stat(
                staging_name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            staging_info = None
        except OSError as exc:
            raise ArtifactError(
                f"cannot inspect W&B run ID staging file ({type(exc).__name__})"
            ) from None
        if staging_info is not None:
            if not stat.S_ISREG(staging_info.st_mode):
                raise ArtifactError("W&B run ID staging artifact is unsafe")
            os.unlink(staging_name, dir_fd=directory_fd)
            os.fsync(directory_fd)

        existing = read_existing()
        if existing is not None:
            return existing

        candidate = _validate_run_id(secrets.token_urlsafe(18))
        payload = (candidate + "\n").encode("utf-8")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                staging_name, flags, 0o600, dir_fd=directory_fd
            )
        except OSError as exc:
            raise ArtifactError(
                f"cannot create W&B run ID staging file ({type(exc).__name__})"
            ) from None
        staging_created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                staging_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            winner = read_existing()
            if winner is None:
                raise ArtifactError("W&B run ID publication race is ambiguous")
            return winner
        os.fsync(directory_fd)
        return candidate
    finally:
        if staging_created:
            try:
                os.unlink(staging_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        if locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _decode_state(path: Path) -> tuple[str | None, dict[str, str], bool]:
    if not path.is_file():
        return None, {}, False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}, False
    if not isinstance(raw, dict):
        return None, {}, False
    if raw.get("schema_version") == STATE_SCHEMA_VERSION:
        run_id = raw.get("wandb_run_id")
        artifacts = raw.get("artifacts")
        if not isinstance(run_id, str) or not isinstance(artifacts, dict):
            raise ArtifactError("W&B upload state schema is malformed")
        return _validate_run_id(run_id), {
            str(key): str(value) for key, value in artifacts.items()
        }, True
    # Version-one state was an unbound path -> fingerprint mapping.  It cannot
    # safely suppress uploads into a newly assigned sidecar run.
    return None, {str(key): str(value) for key, value in raw.items()}, True


def load_state(path: Path, *, expected_run_id: str | None = None) -> dict[str, str]:
    run_id, artifacts, _ = _decode_state(path)
    if expected_run_id is not None:
        expected_run_id = _validate_run_id(expected_run_id)
        if run_id is not None and run_id != expected_run_id:
            raise ArtifactError("W&B upload state belongs to a different sidecar run")
        if run_id is None:
            # Re-upload legacy/unreadable state into the new stable run rather
            # than creating an incomplete run whose history silently skips it.
            return {}
    return artifacts


def save_state(
    path: Path, state: Mapping[str, str], *, run_id: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            payload: Mapping[str, object]
            if run_id is None:
                payload = dict(sorted(state.items()))
            else:
                payload = {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "wandb_run_id": _validate_run_id(run_id),
                    "artifacts": dict(sorted(state.items())),
                }
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bind_state_to_run(path: Path, run_id: str) -> None:
    """Bind persistent artifact fingerprints to exactly one W&B run."""

    state_run_id, _, _ = _decode_state(path)
    if state_run_id is not None and state_run_id != run_id:
        raise ArtifactError("W&B upload state belongs to a different sidecar run")
    if state_run_id is None:
        # Missing, corrupt, and legacy state all restart from an empty upload
        # set.  This can duplicate media, but cannot produce an incomplete run.
        save_state(path, {}, run_id=run_id)


def load_initialized_marker(
    path: Path,
    *,
    expected_run_id: str,
    telemetry_scope: str = "production",
    project: str = "dagger",
    entity: str = "nvidia",
) -> bool:
    """Return whether remote initialization was durably acknowledged locally."""

    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("W&B initialized marker must be a regular non-symlink file")
    try:
        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            payload: dict[str, object] = {}
            for key, value in pairs:
                if key in payload:
                    raise ArtifactError("W&B initialized marker has duplicate fields")
                payload[key] = value
            return payload

        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=strict_object
        )
    except (OSError, json.JSONDecodeError):
        raise ArtifactError("W&B initialized marker is malformed") from None
    if payload != {
        "schema": INITIALIZED_MARKER_SCHEMA,
        "wandb_run_id": _validate_run_id(expected_run_id),
        "telemetry_scope": telemetry_scope,
        "project": project,
        "entity": entity,
    }:
        raise ArtifactError("W&B initialized marker does not match the sidecar run")
    return True


def write_initialized_marker(
    path: Path,
    *,
    run_id: str,
    telemetry_scope: str = "production",
    project: str = "dagger",
    entity: str = "nvidia",
) -> None:
    """Atomically record that W&B accepted initialization of this exact run ID."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": INITIALIZED_MARKER_SCHEMA,
                    "wandb_run_id": _validate_run_id(run_id),
                    "telemetry_scope": telemetry_scope,
                    "project": project,
                    "entity": entity,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class WandbArtifactWatcher:
    """Incrementally upload stable telemetry files to an initialized W&B run."""

    def __init__(
        self,
        *,
        run_dir: Path,
        wandb_module: object,
        wandb_run: object,
        state_file: Path,
        run_id: str | None = None,
        min_age_s: float = 5.0,
        max_image_bytes: int = 25 * 1024 * 1024,
        max_video_bytes: int = 250 * 1024 * 1024,
    ):
        self.run_dir = run_dir.resolve()
        self.wandb = wandb_module
        self.run = wandb_run
        self.state_file = state_file
        self.run_id = run_id
        self.min_age_s = min_age_s
        self.max_image_bytes = max_image_bytes
        self.max_video_bytes = max_video_bytes
        # ``durable_state`` is a delivery acknowledgement: every fingerprint
        # in it followed a successful ``run.finish``.  ``state`` additionally
        # includes artifacts queued during this process, preventing duplicate
        # log calls while deliberately remaining replayable after a crash.
        self.durable_state = load_state(state_file, expected_run_id=run_id)
        self.state = dict(self.durable_state)
        self.last_scan_error_count = 0
        self.last_scan_pending_count = 0

    def scan_once(self, *, now: float | None = None) -> int:
        discovery_errors: list[str] = []
        artifacts = discover_scalars(
            self.run_dir,
            min_age_s=self.min_age_s,
            now=now,
            errors=discovery_errors,
        ) + discover_artifacts(
            self.run_dir,
            min_age_s=self.min_age_s,
            max_image_bytes=self.max_image_bytes,
            max_video_bytes=self.max_video_bytes,
            now=now,
            errors=discovery_errors,
        )
        for error in discovery_errors:
            print(f"W&B artifact watcher skipped: {error}", file=sys.stderr)
        artifacts.sort(key=lambda item: (item.step, item.kind, item.relative_path))
        uploaded = 0
        pending = 0
        upload_errors = 0
        for artifact in artifacts:
            if self.state.get(artifact.relative_path) == artifact.fingerprint:
                continue
            pending += 1
            try:
                payload: dict[str, object] = {"trainer_step": artifact.step}
                if artifact.kind == "scalar":
                    _, metrics = load_scalar_file(artifact.path)
                    payload.update(metrics)
                elif artifact.kind == "image":
                    payload[artifact.key] = self.wandb.Image(str(artifact.path))
                elif artifact.kind == "video":
                    video_format = artifact.path.suffix.lower().lstrip(".")
                    payload[artifact.key] = self.wandb.Video(
                        str(artifact.path), format=video_format
                    )
                else:  # Defensive: discovery only emits the three known kinds.
                    raise ArtifactError(f"unknown artifact kind {artifact.kind!r}")
                self.run.log(payload)
                self.state[artifact.relative_path] = artifact.fingerprint
                uploaded += 1
            except Exception as exc:
                # A malformed media file or transient W&B failure must not stop
                # future uploads.  Keep state unchanged so this artifact retries.
                print(
                    "W&B artifact watcher upload deferred: "
                    f"{artifact.relative_path} ({type(exc).__name__})",
                    file=sys.stderr,
                )
                upload_errors += 1
        self.last_scan_error_count = len(discovery_errors) + upload_errors
        self.last_scan_pending_count = pending - uploaded
        return uploaded

    def commit_state(self) -> None:
        """Persist queued fingerprints after the remote sender is drained.

        Callers must invoke this only after ``run.finish`` succeeds.  If the
        process dies or finish fails first, the previous durable cursor stays
        intact and a restarted watcher replays the unacknowledged artifacts.
        """

        save_state(self.state_file, self.state, run_id=self.run_id)
        self.durable_state = dict(self.state)

    def terminal_projection(self, *, exit_code: int) -> dict[str, int] | None:
        """Return the final dashboard projection for actor terminal telemetry."""

        found = False
        for relative_path in self.state:
            if "actor_pair_terminal_" not in relative_path or not relative_path.endswith(".json"):
                continue
            path = self.run_dir / relative_path
            try:
                _, metrics = load_scalar_file(path)
            except ArtifactError:
                continue
            found = found or metrics.get("collection/terminal_pending") == 1
        if not found:
            return None
        succeeded = int(exit_code == 0)
        return {
            "collection/delivery_confirmed": succeeded,
            "collection/terminal_pending": 0,
            "collection/succeeded": succeeded,
            "collection/failed": 1 - succeeded,
        }

    def write_delivery_receipt(self, *, exit_code: int) -> None:
        """Bind a successful sender drain to the exact durable cursor bytes."""

        projection = self.terminal_projection(exit_code=exit_code)
        if projection is None or exit_code != 0:
            return
        state_sha256 = hashlib.sha256(self.state_file.read_bytes()).hexdigest()
        path = self.run_dir / "telemetry" / "WANDB_DELIVERY_RECEIPT.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "schema": DELIVERY_RECEIPT_SCHEMA,
                        "wandb_run_id": self.run_id,
                        "upload_state_sha256": state_sha256,
                        "outcome": "SUCCEEDED",
                    },
                    output,
                    sort_keys=True,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def write_ready_marker(
    path: Path,
    *,
    account_email: str,
    entity: str,
    project: str,
    authorization_scope: str,
    team_member: bool,
    entity_is_team: bool,
    personal: bool,
    run_id: str | None = None,
    telemetry_scope: str = "production",
) -> None:
    """Publish watcher readiness atomically after W&B initialization succeeds."""

    if telemetry_scope not in TELEMETRY_SCOPES:
        raise ArtifactError("W&B telemetry scope is invalid")

    payload = {
        "status": "ready",
        "account_email": account_email,
        "entity": entity,
        "authorization_scope": authorization_scope,
        "team_member": team_member,
        "entity_is_team": entity_is_team,
        "personal": personal,
        "project": project,
        "telemetry_scope": telemetry_scope,
        "pid": os.getpid(),
        "ready_unix_s": time.time(),
    }
    if run_id is not None:
        payload["wandb_run_id"] = _validate_run_id(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--telemetry-scope",
        choices=tuple(TELEMETRY_SCOPES),
        default="production",
        help=(
            "label the remote run as production observability or a bounded "
            "validation gate; bounded validation never imports production config"
        ),
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--expected-email", default=os.environ.get("WANDB_EXPECTED_EMAIL"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--authorization-scope", required=True, choices=("team",)
    )
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--max-runtime-s", type=float, default=0.0)
    parser.add_argument("--min-age-s", type=float, default=5.0)
    parser.add_argument("--max-image-mib", type=float, default=25.0)
    parser.add_argument("--max-video-mib", type=float, default=250.0)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--run-id-file", type=Path, default=None)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help=(
            "optional strict round_dagger W&B public config; defaults to "
            "RUN_DIR/telemetry/WANDB_PUBLIC_CONFIG.json when present"
        ),
    )
    parser.add_argument("--ready-marker", type=Path, default=None)
    parser.add_argument("--finish-marker", default="RUN_DONE")
    parser.add_argument("--max-consecutive-error-scans", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if process_rank() != 0:
        print("W&B artifact watcher: nonzero rank; no-op")
        return 0
    if not args.expected_email:
        print("W&B artifact watcher: expected email is required", file=sys.stderr)
        return 2
    if not args.entity:
        print("W&B artifact watcher: entity is required", file=sys.stderr)
        return 2
    if (
        args.poll_interval_s <= 0
        or args.min_age_s < 0
        or args.max_runtime_s < 0
        or args.max_consecutive_error_scans < 1
    ):
        print("W&B artifact watcher: invalid timing argument", file=sys.stderr)
        return 2

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or run_dir / "telemetry" / ".wandb_upload_state.json"
    run_id_file = args.run_id_file or run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID"
    initialized_marker = run_dir / "telemetry" / "WANDB_SIDECAR_INITIALIZED.json"
    default_public_config = run_dir / "telemetry" / "WANDB_PUBLIC_CONFIG.json"
    public_config_path = args.config_json
    if args.telemetry_scope == "bounded-validation":
        if public_config_path is not None:
            print(
                "W&B artifact watcher: bounded validation cannot use production "
                "public config",
                file=sys.stderr,
            )
            return 2
        public_config_path = None
    elif public_config_path is None and default_public_config.exists():
        public_config_path = default_public_config
    managed_paths = [
        ("state file", state_file),
        ("run ID file", run_id_file),
        ("initialized marker", initialized_marker),
    ]
    if public_config_path is not None:
        managed_paths.append(("public config", public_config_path))
    for label, path in managed_paths:
        try:
            path.resolve().relative_to(run_dir)
        except ValueError:
            print(f"W&B artifact watcher: {label} must be inside run-dir", file=sys.stderr)
            return 2
    try:
        public_config = (
            load_public_config(public_config_path)
            if public_config_path is not None
            else {}
        )
    except ArtifactError as exc:
        print(f"W&B artifact watcher: {exc}", file=sys.stderr)
        return 2
    try:
        wandb = _load_wandb()
        identity = query_identity(
            wandb,
            expected_email=args.expected_email,
            requested_entity=args.entity,
            authorization_scope=args.authorization_scope,
        )
    except WandbPreflightError as exc:
        print(f"W&B artifact watcher: {exc}", file=sys.stderr)
        return 2

    try:
        run_id = load_or_create_run_id(run_id_file)
        bind_state_to_run(state_file, run_id)
        remote_initialized = load_initialized_marker(
            initialized_marker,
            expected_run_id=run_id,
            telemetry_scope=args.telemetry_scope,
            project=args.project,
            entity=identity.entity,
        )
        if not remote_initialized:
            # ``allow`` can safely recover a crash between local ID creation
            # and remote initialization.  Re-uploading all artifacts prevents
            # a legacy/local state file from suppressing missing remote data.
            save_state(state_file, {}, run_id=run_id)
    except ArtifactError as exc:
        print(f"W&B artifact watcher: {exc}", file=sys.stderr)
        return 2

    wandb_dir = run_dir / ".wandb_media"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = None
    try:
        run = wandb.init(
            project=args.project,
            entity=identity.entity,
            name=args.name or f"{run_dir.name}-media",
            group=args.group,
            job_type=TELEMETRY_SCOPES[args.telemetry_scope],
            id=run_id,
            # Once initialization is durably acknowledged, ``must`` prevents
            # a deleted remote run from being silently recreated while local
            # upload state suppresses its prior history.
            resume="must" if remote_initialized else "allow",
            dir=str(wandb_dir),
            config={
                "source_run_dir": str(run_dir),
                "artifact_schema": STATE_SCHEMA_VERSION,
                "telemetry_scope": args.telemetry_scope,
                "expected_account": identity.email,
                "account_email": identity.email,
                "entity": identity.entity,
                "authorization_scope": identity.authorization_scope,
                "team_member": identity.team_member,
                "entity_is_team": identity.entity_is_team,
                "personal": identity.personal,
                **public_config,
            },
        )
        run.define_metric("trainer_step")
        run.define_metric("*", step_metric="trainer_step")
        write_initialized_marker(
            initialized_marker,
            run_id=run_id,
            telemetry_scope=args.telemetry_scope,
            project=args.project,
            entity=identity.entity,
        )
    except Exception as exc:
        if run is not None:
            try:
                run.finish(exit_code=2)
            except Exception:
                pass
        print(
            f"W&B artifact watcher: remote initialization failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2
    ready_marker = args.ready_marker or run_dir / "telemetry" / "WANDB_WATCHER_READY.json"
    ready_marker = ready_marker.resolve()
    try:
        ready_marker.relative_to(run_dir)
    except ValueError:
        print("W&B artifact watcher: ready marker must be inside run-dir", file=sys.stderr)
        run.finish(exit_code=2)
        return 2
    write_ready_marker(
        ready_marker,
        account_email=identity.email,
        entity=identity.entity,
        project=args.project,
        authorization_scope=identity.authorization_scope,
        team_member=identity.team_member,
        entity_is_team=identity.entity_is_team,
        personal=identity.personal,
        run_id=run_id,
        telemetry_scope=args.telemetry_scope,
    )
    watcher = WandbArtifactWatcher(
        run_dir=run_dir,
        wandb_module=wandb,
        wandb_run=run,
        state_file=state_file,
        run_id=run_id,
        min_age_s=args.min_age_s,
        max_image_bytes=int(args.max_image_mib * 1024 * 1024),
        max_video_bytes=int(args.max_video_mib * 1024 * 1024),
    )

    started = time.monotonic()
    final_drain_deadline: float | None = None
    consecutive_error_scans = 0
    exit_code = 0
    try:
        while True:
            watcher.scan_once()
            if watcher.last_scan_error_count:
                consecutive_error_scans += 1
            else:
                consecutive_error_scans = 0
            if consecutive_error_scans >= args.max_consecutive_error_scans:
                raise ArtifactError(
                    "telemetry artifacts failed discovery/upload for "
                    f"{consecutive_error_scans} consecutive scans"
                )
            now = time.monotonic()
            if args.once:
                if (
                    watcher.last_scan_pending_count
                    or watcher.last_scan_error_count
                ):
                    exit_code = 2
                break
            final_drain_deadline, drain_complete = update_final_drain(
                marker_present=(run_dir / args.finish_marker).exists(),
                deadline=final_drain_deadline,
                now=now,
                min_age_s=args.min_age_s,
            )
            if drain_complete:
                if watcher.last_scan_pending_count or watcher.last_scan_error_count:
                    raise ArtifactError(
                        "final telemetry drain left pending or invalid artifacts"
                    )
                exit_code = load_finish_exit_code(run_dir / args.finish_marker)
                break
            if (
                final_drain_deadline is None
                and args.max_runtime_s
                and now - started >= args.max_runtime_s
            ):
                # A bounded watcher that expires without its supervisor's
                # terminal marker has lost lifecycle ownership.  Marking that
                # run successful would hide an orphaned actor/supervisor.
                print(
                    "W&B artifact watcher: runtime expired before terminal marker",
                    file=sys.stderr,
                )
                exit_code = 2
                break
            sleep_s = min(args.poll_interval_s, 60.0)
            if final_drain_deadline is not None:
                sleep_s = min(sleep_s, max(0.01, final_drain_deadline - now))
            time.sleep(sleep_s)
    except (ArtifactError, OSError) as exc:
        print(f"W&B artifact watcher failed: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        try:
            projection = watcher.terminal_projection(exit_code=exit_code)
            if projection is not None:
                # The immutable history scalar intentionally says pending.  A
                # successful sender drain makes this W&B summary the final
                # operator-facing projection; a failed drain never commits the
                # local cursor/receipt below.
                run.summary.update(projection)
            run.finish(exit_code=exit_code)
        except Exception as exc:
            try:
                run.finish(exit_code=2)
            except Exception:
                pass
            print(
                f"W&B artifact watcher: remote finish failed ({type(exc).__name__})",
                file=sys.stderr,
            )
            exit_code = 2
        else:
            try:
                watcher.commit_state()
                watcher.write_delivery_receipt(exit_code=exit_code)
            except (ArtifactError, OSError) as exc:
                # Remote delivery succeeded but the acknowledgement was not
                # durable.  Fail closed; restart will replay rather than lose
                # telemetry.  Do not expose exception text because paths may
                # contain operator-specific information.
                print(
                    "W&B artifact watcher: upload acknowledgement failed "
                    f"({type(exc).__name__})",
                    file=sys.stderr,
                )
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
