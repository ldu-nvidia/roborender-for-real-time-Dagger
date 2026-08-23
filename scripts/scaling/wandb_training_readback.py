#!/usr/bin/env python3
"""Fail closed until trainer scalars and sidecar scalar/image/video exist remotely."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import yaml

try:
    from scripts.scaling.wandb_preflight import _load_wandb, query_identity
    from scripts.scaling.wandb_run_watcher import (
        discover_artifacts,
        discover_scalars,
        load_scalar_file,
    )
except ModuleNotFoundError:  # Direct execution from scripts/scaling.
    from wandb_preflight import _load_wandb, query_identity
    from wandb_run_watcher import discover_artifacts, discover_scalars, load_scalar_file


RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
REPORT_SCHEMA = "fast_roborender.wandb_training_readback.v3"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_REMOTE_MEDIA_BYTES = {
    "image": 25 * 1024 * 1024,
    "video": 250 * 1024 * 1024,
}
HISTORY_PAGE_SIZE = 500


class ReadbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScalarPoint:
    key: str
    step: int
    value: int | float
    relative_path: str


@dataclass(frozen=True)
class LocalMediaEvidence:
    """A local media object with independently calculated fingerprints."""

    key: str
    kind: str
    step: int
    relative_path: str
    size: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class RemoteMediaEvidence:
    name: str
    kind: str
    size: int
    md5: str
    file_object: object = field(compare=False, repr=False)


@dataclass(frozen=True)
class DownloadedMediaEvidence:
    name: str
    kind: str
    size: int
    md5: str
    sha256: str


def _run_id(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ReadbackError(f"{label} is missing or unsafe")
    value = path.read_text(encoding="utf-8").strip()
    if RUN_ID_RE.fullmatch(value) is None:
        raise ReadbackError(f"{label} is malformed")
    return value


def _trainer_run_id(run_dir: Path) -> str:
    path = run_dir / "meta.yaml"
    if not path.is_file() or path.is_symlink():
        raise ReadbackError("trainer W&B metadata is missing or unsafe")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = payload.get("wandb_run") if isinstance(payload, dict) else None
    if not isinstance(value, str) or RUN_ID_RE.fullmatch(value) is None:
        raise ReadbackError("trainer W&B run ID is missing or malformed")
    return value


def _media_fingerprints(path: Path) -> tuple[int, str, str]:
    before = path.stat()
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5_digest.update(chunk)
            sha256_digest.update(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise ReadbackError("local media changed while it was fingerprinted")
    return (
        after.st_size,
        base64.b64encode(md5_digest.digest()).decode("ascii"),
        sha256_digest.hexdigest(),
    )


def _expected_sidecar_evidence(
    run_dir: Path,
) -> tuple[tuple[ScalarPoint, ...], tuple[LocalMediaEvidence, ...], dict[str, int]]:
    scalar_artifacts = discover_scalars(run_dir, min_age_s=0.0)
    media_artifacts = discover_artifacts(run_dir, min_age_s=0.0)
    by_kind = {
        "scalar": len(scalar_artifacts),
        "image": sum(item.kind == "image" for item in media_artifacts),
        "video": sum(item.kind == "video" for item in media_artifacts),
    }
    if any(count < 1 for count in by_kind.values()):
        raise ReadbackError(f"local scalar/image/video telemetry is incomplete: {by_kind}")
    scalar_points: list[ScalarPoint] = []
    scalar_steps: set[int] = set()
    for artifact in scalar_artifacts:
        step, metrics = load_scalar_file(artifact.path)
        if step in scalar_steps:
            raise ReadbackError("local scalar telemetry has a duplicate trainer step")
        scalar_steps.add(step)
        scalar_points.append(
            ScalarPoint(
                key="trainer_step",
                step=step,
                value=step,
                relative_path=artifact.relative_path,
            )
        )
        scalar_points.extend(
            ScalarPoint(
                key=key,
                step=step,
                value=value,
                relative_path=artifact.relative_path,
            )
            for key, value in metrics.items()
        )
    media_evidence = []
    for artifact in media_artifacts:
        size, md5, sha256 = _media_fingerprints(artifact.path)
        media_evidence.append(
            LocalMediaEvidence(
                key=artifact.key,
                kind=artifact.kind,
                step=artifact.step,
                relative_path=artifact.relative_path,
                size=size,
                md5=md5,
                sha256=sha256,
            )
        )
    scalar_points.sort(key=lambda item: (item.key, item.step, item.relative_path))
    return tuple(scalar_points), tuple(media_evidence), by_kind


def _expected_trainer_evidence(run_dir: Path) -> tuple[ScalarPoint, ...]:
    marker_re = re.compile(r"dagger_bc_loss_step_([0-9]{9})\.json\Z")
    points = []
    updates = []
    for path in sorted((run_dir / "validation").glob("dagger_bc_loss_step_*.json")):
        if not path.is_file() or path.is_symlink():
            raise ReadbackError("trainer loss marker is missing or unsafe")
        match = marker_re.fullmatch(path.name)
        if match is None:
            raise ReadbackError("trainer loss marker name is malformed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReadbackError(
                f"trainer loss marker is invalid ({type(exc).__name__})"
            ) from None
        update = int(match.group(1))
        loss = payload.get("loss") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "fast_roborender.dagger_bc_loss.v1"
            or payload.get("update") != update
            or payload.get("finite_verified") is not True
            or payload.get("nonnegative_verified") is not True
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0
        ):
            raise ReadbackError("trainer loss marker contract is invalid")
        updates.append(update)
        points.append(
            ScalarPoint(
                key="production/dagger_bc_loss",
                # WandbCallback logs with state.global_step after the update;
                # W&B's zero-based internal _step is therefore update - 1.
                step=update - 1,
                value=loss,
                relative_path=str(path.relative_to(run_dir)),
            )
        )
    if not points:
        raise ReadbackError("trainer loss evidence is missing")
    if updates != list(range(1, updates[-1] + 1)):
        raise ReadbackError("trainer loss evidence is not contiguous from update one")
    return tuple(points)


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _history_step(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or float(number) < 0 or not float(number).is_integer():
        return None
    return int(number)


def _history_media_size(value: object, *, kind: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if (
        not math.isfinite(number)
        or number <= 0
        or not number.is_integer()
        or number > MAX_REMOTE_MEDIA_BYTES[kind]
    ):
        return None
    return int(number)


def _numbers_match(expected: int | float, observed: int | float) -> bool:
    return math.isclose(
        float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12
    )


def _scan_scalar_points(
    remote_run: object,
    expected: tuple[ScalarPoint, ...],
    *,
    axis: str,
    page_size: int,
) -> tuple[ScalarPoint, ...]:
    """Bind local values from one complete, unprojected W&B history scan.

    W&B's explicit ``keys=`` projection can silently return null for scalar
    fields that are present in the underlying history rows.  Scanning the
    complete rows once is both faster than per-key queries and avoids that
    projection defect.  Each accepted point is still bound to its exact local
    axis step and numeric value.  Every point originating in one local scalar
    record must coexist in one remote history row; sparse or split rows cannot
    collectively satisfy that record.
    """

    expected_by_step: dict[int, dict[str, list[ScalarPoint]]] = {}
    for point in expected:
        expected_by_step.setdefault(point.step, {}).setdefault(
            point.relative_path, []
        ).append(point)
    verified_identities: set[tuple[str, int, str]] = set()
    for row in remote_run.scan_history(page_size=page_size, use_cache=False):
        if not isinstance(row, dict):
            continue
        step = _history_step(row.get(axis))
        if step is None or step not in expected_by_step:
            continue
        for record_points in expected_by_step[step].values():
            if all(
                (value := _finite_number(row.get(point.key))) is not None
                and _numbers_match(point.value, value)
                for point in record_points
            ):
                verified_identities.update(
                    (point.key, point.step, point.relative_path)
                    for point in record_points
                )
    verified = [
        point
        for point in expected
        if (point.key, point.step, point.relative_path) in verified_identities
    ]
    return tuple(verified)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_evidence_report(
    required: tuple[ScalarPoint, ...],
    verified: tuple[ScalarPoint, ...],
    *,
    axis: str,
    resume_training: bool,
    resume_checkpoint_step: int,
    rollout_steps: int,
) -> dict[str, object]:
    def records(points: tuple[ScalarPoint, ...]) -> list[dict[str, object]]:
        return [
            {"key": item.key, "step": item.step, "value": item.value}
            for item in points
        ]

    required_records = records(required)
    verified_records = records(verified)
    steps = sorted({item.step for item in required})
    verified_identities = {
        (item.key, item.step, float(item.value)) for item in verified
    }
    boundary: dict[str, object] = {
        "resume_training": resume_training,
        "resume_checkpoint_step": resume_checkpoint_step,
        "rollout_steps": rollout_steps,
        "boundary_pre_step": None,
        "boundary_post_step": None,
        "boundary_pre_verified": not resume_training,
        "boundary_post_verified": not resume_training,
    }
    if resume_training:
        if axis == "trainer_step":
            boundary_value = resume_checkpoint_step * rollout_steps
            pre = max((step for step in steps if step < boundary_value), default=None)
            post = min((step for step in steps if step >= boundary_value), default=None)
        else:
            pre_value = resume_checkpoint_step - 1
            pre = pre_value if pre_value in steps else None
            post_value = resume_checkpoint_step
            post = post_value if post_value in steps else None
        boundary["boundary_pre_step"] = pre
        boundary["boundary_post_step"] = post
        boundary["boundary_pre_verified"] = pre is not None and all(
            (item.key, item.step, float(item.value)) in verified_identities
            for item in required
            if item.step == pre
        )
        boundary["boundary_post_verified"] = post is not None and all(
            (item.key, item.step, float(item.value)) in verified_identities
            for item in required
            if item.step == post
        )
    return {
        "axis": axis,
        "required_point_count": len(required),
        "verified_point_count": len(verified),
        "required_points_sha256": _canonical_sha256(required_records),
        "verified_points_sha256": _canonical_sha256(verified_records),
        "required_step_count": len(steps),
        "required_steps_sha256": _canonical_sha256(steps),
        "min_step": steps[0] if steps else None,
        "max_step": steps[-1] if steps else None,
        "all_points_verified": required_records == verified_records,
        **boundary,
    }


def _media_kind_from_remote_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    if len(path.parts) >= 3 and path.parts[:2] == ("media", "images"):
        return "image"
    if len(path.parts) >= 3 and path.parts[:2] == ("media", "videos"):
        return "video"
    return None


def _structured_item(value: object, key: str) -> object:
    """Read mappings and W&B SummarySubDict objects without private payloads."""

    try:
        return value[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _remote_media_files(remote_run: object) -> tuple[RemoteMediaEvidence, ...]:
    files = remote_run.files()  # type: ignore[attr-defined]
    found = []
    for item in files:
        name = getattr(item, "name", None)
        kind = _media_kind_from_remote_path(name)
        size = getattr(item, "size", None)
        md5 = getattr(item, "md5", None)
        if (
            kind is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= MAX_REMOTE_MEDIA_BYTES[kind]
            or not isinstance(md5, str)
            or not md5
        ):
            continue
        found.append(
            RemoteMediaEvidence(
                name=name,
                kind=kind,
                size=size,
                md5=md5,
                file_object=item,
            )
        )
    names = [item.name for item in found]
    if len(names) != len(set(names)):
        raise ReadbackError("remote W&B media file names are not unique")
    return tuple(found)


def _download_remote_media(
    remote: RemoteMediaEvidence, download_root: Path
) -> DownloadedMediaEvidence:
    expected_path = download_root / PurePosixPath(remote.name)
    expected_resolved = expected_path.resolve(strict=False)
    try:
        expected_resolved.relative_to(download_root.resolve(strict=True))
    except ValueError:
        raise ReadbackError("remote media download path escaped temporary storage") from None
    result = remote.file_object.download(  # type: ignore[attr-defined]
        root=str(download_root), replace=True
    )
    close = getattr(result, "close", None)
    if callable(close):
        close()
    if not expected_path.is_file() or expected_path.is_symlink():
        raise ReadbackError("remote media download is missing or unsafe")
    current = expected_path
    while current != download_root:
        if current.is_symlink():
            raise ReadbackError("remote media download contains a symlink")
        current = current.parent
    size, md5, sha256 = _media_fingerprints(expected_path)
    if size != remote.size or md5 != remote.md5:
        raise ReadbackError("downloaded media does not match W&B file metadata")
    return DownloadedMediaEvidence(
        name=remote.name,
        kind=remote.kind,
        size=size,
        md5=md5,
        sha256=sha256,
    )


def _verify_remote_media(
    remote_run: object,
    expected: tuple[LocalMediaEvidence, ...],
    *,
    download_root: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Bind distinct W&B media history events to downloaded remote blobs.

    W&B may content-deduplicate multiple uploads into one remote path.  The
    logical evidence is therefore injective in remote history ``_step`` rather
    than file path: every local artifact needs a distinct event at its exact
    ``trainer_step``.  Referenced paths are downloaded and hashed once.
    """

    download_root.mkdir(parents=True, exist_ok=False)
    remote_files = _remote_media_files(remote_run)
    remote_by_name = {item.name: item for item in remote_files}
    summary = getattr(remote_run, "summary", None)
    expected_by_key: dict[str, list[LocalMediaEvidence]] = {}
    for item in expected:
        expected_by_key.setdefault(item.key, []).append(item)
    expected_types = {"image": "image-file", "video": "video-file"}
    events_by_key: dict[str, list[dict[str, object]]] = {
        key: [] for key in expected_by_key
    }
    for row in remote_run.scan_history(  # type: ignore[attr-defined]
        page_size=HISTORY_PAGE_SIZE,
        use_cache=False,
    ):
        if not isinstance(row, dict):
            continue
        history_step = _history_step(row.get("_step"))
        trainer_step = _history_step(row.get("trainer_step"))
        if history_step is None or trainer_step is None:
            continue
        for key, local_items in expected_by_key.items():
            kind = local_items[0].kind
            metadata = row.get(key)
            media_type = _structured_item(metadata, "_type")
            path = _structured_item(metadata, "path")
            size = _history_media_size(
                _structured_item(metadata, "size"), kind=kind
            )
            sha256 = _structured_item(metadata, "sha256")
            if (
                media_type != expected_types[kind]
                or _media_kind_from_remote_path(path) != kind
                or size is None
                or not isinstance(sha256, str)
                or SHA256_RE.fullmatch(sha256) is None
            ):
                continue
            remote_file = remote_by_name.get(path)
            if (
                remote_file is None
                or remote_file.kind != kind
                or remote_file.size != size
            ):
                continue
            events_by_key[key].append(
                {
                    "history_step": history_step,
                    "trainer_step": trainer_step,
                    "path": path,
                    "size": size,
                    "sha256": sha256,
                    "md5": remote_file.md5,
                }
            )

    assignments: list[tuple[LocalMediaEvidence, dict[str, object]]] = []
    used_history_steps: set[int] = set()
    summary_events: dict[str, int] = {}
    for key, local_items in sorted(expected_by_key.items()):
        kinds = {item.kind for item in local_items}
        if len(kinds) != 1:
            raise ReadbackError("one media key maps to multiple local media kinds")
        kind = next(iter(kinds))
        events = events_by_key[key]
        events.sort(key=lambda item: int(item["history_step"]))
        local_assignments = []
        ordered_local_items = sorted(
            local_items, key=lambda item: (item.step, item.relative_path)
        )
        for local in ordered_local_items:
            event = next(
                (
                    item
                    for item in events
                    if int(item["history_step"]) not in used_history_steps
                    and item["trainer_step"] == local.step
                    and item["size"] == local.size
                    and item["sha256"] == local.sha256
                    and item["md5"] == local.md5
                ),
                None,
            )
            if event is None:
                raise ReadbackError("local media has no distinct exact W&B history event")
            history_step = int(event["history_step"])
            used_history_steps.add(history_step)
            assignments.append((local, event))
            local_assignments.append((local, event))

        summary_value = _structured_item(summary, key)
        summary_type = _structured_item(summary_value, "_type")
        summary_path = _structured_item(summary_value, "path")
        summary_size = _structured_item(summary_value, "size")
        summary_sha256 = _structured_item(summary_value, "sha256")
        newest_local, newest_event = max(
            local_assignments, key=lambda item: int(item[1]["history_step"])
        )
        if newest_local != ordered_local_items[-1]:
            raise ReadbackError("remote media history order differs from local order")
        if not (
            summary_type == expected_types[kind]
            and newest_event["path"] == summary_path
            and newest_local.size == summary_size
            and newest_local.sha256 == summary_sha256
        ):
            raise ReadbackError("remote media summary is not the newest exact event")
        summary_events[key] = int(newest_event["history_step"])

    referenced_paths = sorted({str(event["path"]) for _, event in assignments})
    downloaded = {
        path: _download_remote_media(remote_by_name[path], download_root)
        for path in referenced_paths
    }
    verified_files = []
    for local, event in assignments:
        remote = downloaded[str(event["path"])]
        if (
            (local.kind, local.size, local.md5, local.sha256)
            != (remote.kind, remote.size, remote.md5, remote.sha256)
        ):
            raise ReadbackError("downloaded remote media does not match local artifact")
        history_step = int(event["history_step"])
        verified_files.append(
            {
                "key": local.key,
                "kind": local.kind,
                "trainer_step": local.step,
                "remote_history_step": history_step,
                "local_relative_path": local.relative_path,
                "remote_path": remote.name,
                "size": local.size,
                "local_md5": local.md5,
                "remote_md5": remote.md5,
                "local_sha256": local.sha256,
                "remote_sha256": remote.sha256,
                "summary_current": history_step == summary_events[local.key],
            }
        )
    verified_files.sort(
        key=lambda item: (str(item["key"]), str(item["local_relative_path"]))
    )
    return set(expected_by_key), verified_files


def _verify_remote_media_attempt(
    remote_run: object,
    expected: tuple[LocalMediaEvidence, ...],
    *,
    validation_dir: Path,
    attempt: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Verify one retry without retaining partial downloads for later retries."""

    with tempfile.TemporaryDirectory(
        prefix=f".wandb-readback-{attempt:04d}-", dir=validation_dir
    ) as temporary:
        return _verify_remote_media(
            remote_run,
            expected,
            download_root=Path(temporary) / "media",
        )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument(
        "--authorization-scope", required=True, choices=("team",)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--rollout-steps", type=int, required=True)
    parser.add_argument("--resume-training", type=int, choices=(0, 1), required=True)
    parser.add_argument("--resume-checkpoint-step", type=int, required=True)
    return parser


def _fully_verified_keys(
    required: tuple[ScalarPoint, ...], verified: tuple[ScalarPoint, ...]
) -> set[str]:
    verified_set = {
        (item.key, item.step, float(item.value), item.relative_path)
        for item in verified
    }
    keys = {item.key for item in required}
    return {
        key
        for key in keys
        if all(
            (item.key, item.step, float(item.value), item.relative_path)
            in verified_set
            for item in required
            if item.key == key
        )
    }


def main() -> int:
    args = _parser().parse_args()
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive")
    if args.resume_checkpoint_step < 0:
        raise SystemExit("--resume-checkpoint-step must be non-negative")
    if args.resume_training == 0 and args.resume_checkpoint_step != 0:
        raise SystemExit("non-resume readback requires checkpoint step zero")
    if args.resume_training == 1 and args.resume_checkpoint_step <= 0:
        raise SystemExit("resume readback requires a positive checkpoint step")

    run_dir = args.run_dir.resolve(strict=True)
    trainer_run_id = _trainer_run_id(run_dir)
    sidecar_run_id = _run_id(
        run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID", "sidecar W&B run ID"
    )
    local_sidecar_scalars, local_media, local_counts = _expected_sidecar_evidence(
        run_dir
    )
    local_trainer_scalars = _expected_trainer_evidence(run_dir)
    sidecar_scalar_keys = {item.key for item in local_sidecar_scalars}
    sidecar_media_keys = {item.key for item in local_media}
    sidecar_keys = sidecar_scalar_keys | sidecar_media_keys
    trainer_keys = {item.key for item in local_trainer_scalars}

    wandb = _load_wandb()
    identity = query_identity(
        wandb,
        expected_email=args.expected_email,
        requested_entity=args.entity,
        authorization_scope=args.authorization_scope,
    )
    api = wandb.Api(timeout=30)
    verified_trainer_scalars: tuple[ScalarPoint, ...] = ()
    verified_sidecar_scalars: tuple[ScalarPoint, ...] = ()
    observed_sidecar_media: set[str] = set()
    verified_remote_media_events: list[dict[str, Any]] = []
    last_error = ""
    deadline = time.monotonic() + args.timeout_s
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            trainer = api.run(
                f"{identity.entity}/{args.project}/{trainer_run_id}"
            )
            verified_trainer_scalars = _scan_scalar_points(
                trainer,
                local_trainer_scalars,
                axis="_step",
                page_size=HISTORY_PAGE_SIZE,
            )
            sidecar = api.run(
                f"{identity.entity}/{args.project}/{sidecar_run_id}"
            )
            verified_sidecar_scalars = _scan_scalar_points(
                sidecar,
                local_sidecar_scalars,
                axis="trainer_step",
                page_size=HISTORY_PAGE_SIZE,
            )
            if (
                len(verified_trainer_scalars) == len(local_trainer_scalars)
                and len(verified_sidecar_scalars) == len(local_sidecar_scalars)
                and sidecar_media_keys != observed_sidecar_media
            ):
                (
                    observed_sidecar_media,
                    verified_remote_media_events,
                ) = _verify_remote_media_attempt(
                    sidecar,
                    local_media,
                    validation_dir=validation_dir,
                    attempt=attempt,
                )
            observed_trainer = _fully_verified_keys(
                local_trainer_scalars, verified_trainer_scalars
            )
            observed_sidecar_scalars = _fully_verified_keys(
                local_sidecar_scalars, verified_sidecar_scalars
            )
            observed_sidecar = observed_sidecar_scalars | observed_sidecar_media
            if (
                trainer_keys <= observed_trainer
                and sidecar_keys <= observed_sidecar
            ):
                break
        except Exception as exc:  # Never persist third-party error text.
            last_error = type(exc).__name__
        time.sleep(5)

    observed_trainer = _fully_verified_keys(
        local_trainer_scalars, verified_trainer_scalars
    )
    observed_sidecar_scalars = _fully_verified_keys(
        local_sidecar_scalars, verified_sidecar_scalars
    )
    observed_sidecar = observed_sidecar_scalars | observed_sidecar_media
    trainer_scalar_report = _scalar_evidence_report(
        local_trainer_scalars,
        verified_trainer_scalars,
        axis="_step",
        resume_training=bool(args.resume_training),
        resume_checkpoint_step=args.resume_checkpoint_step,
        rollout_steps=args.rollout_steps,
    )
    sidecar_scalar_report = _scalar_evidence_report(
        local_sidecar_scalars,
        verified_sidecar_scalars,
        axis="trainer_step",
        resume_training=bool(args.resume_training),
        resume_checkpoint_step=args.resume_checkpoint_step,
        rollout_steps=args.rollout_steps,
    )
    status = (
        "PASS"
        if trainer_keys <= observed_trainer and sidecar_keys <= observed_sidecar
        and trainer_scalar_report["all_points_verified"] is True
        and sidecar_scalar_report["all_points_verified"] is True
        and trainer_scalar_report["boundary_pre_verified"] is True
        and trainer_scalar_report["boundary_post_verified"] is True
        and sidecar_scalar_report["boundary_pre_verified"] is True
        and sidecar_scalar_report["boundary_post_verified"] is True
        and len(verified_remote_media_events) == len(local_media)
        and len(
            {item["remote_history_step"] for item in verified_remote_media_events}
        )
        == len(local_media)
        else "FAIL"
    )
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": args.project,
        "trainer_run_id": trainer_run_id,
        "sidecar_run_id": sidecar_run_id,
        "required_trainer_keys": sorted(trainer_keys),
        "observed_trainer_keys": sorted(observed_trainer),
        "trainer_scalar_evidence": trainer_scalar_report,
        "required_sidecar_keys": sorted(sidecar_keys),
        "observed_sidecar_keys": sorted(observed_sidecar),
        "required_sidecar_scalar_keys": sorted(sidecar_scalar_keys),
        "observed_sidecar_scalar_keys": sorted(observed_sidecar_scalars),
        "sidecar_scalar_evidence": sidecar_scalar_report,
        "required_sidecar_media_keys": sorted(sidecar_media_keys),
        "observed_sidecar_media_keys": sorted(observed_sidecar_media),
        "verified_remote_media_events": verified_remote_media_events,
        "required_remote_media_event_count": len(local_media),
        "verified_remote_media_event_count": len(verified_remote_media_events),
        "downloaded_remote_media_file_count": len(
            {item["remote_path"] for item in verified_remote_media_events}
        ),
        "remote_media_history_steps_unique": len(
            {item["remote_history_step"] for item in verified_remote_media_events}
        )
        == len(verified_remote_media_events),
        "remote_media_paths_unique": len(
            {item["remote_path"] for item in verified_remote_media_events}
        )
        == len(verified_remote_media_events),
        "local_artifact_counts": local_counts,
        "last_api_error_type": last_error,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
