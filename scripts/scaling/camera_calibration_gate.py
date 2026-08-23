#!/usr/bin/env python3
"""Create and verify explicit human acceptance of three-view camera calibration.

The technical calibration run produces two diagnostic PNG panels and a JSON
manifest.  This utility turns an explicit human review into a content-addressed
acceptance record.  It intentionally fails closed: acceptance is bound to the
exact manifest bytes, panel bytes, ordered camera contract, and both repository
revisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import stat
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


TECHNICAL_SCHEMA = "fast_roborender.camera_calibration.v2"
ACCEPTANCE_SCHEMA = "fast_roborender.camera_calibration_acceptance.v2"
VERIFICATION_SCHEMA = "fast_roborender.camera_calibration_verification.v2"
TECHNICAL_STATUS = "awaiting_visual_acceptance"
ACCEPTANCE_STATUS = "accepted"
EXPECTED_VIEWS = ("head_color", "hand_left", "hand_right")
EXPECTED_ROWS = ("depth_control", "simulator_rgb", "generated_rgb")
EXPECTED_TILE_SHAPE_HWC = (240, 416, 3)
EXPECTED_PANEL_SHAPE_HWC = (720, 1248, 3)
EXPECTED_PANEL_NAMES = ("step000_env0.png", "step001_env0.png")
CONFIRMATION = (
    "I reviewed both three-view calibration panels and accept the camera mapping."
)
_HEX_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class GateError(RuntimeError):
    """Raised when an acceptance or calibration artifact fails validation."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise GateError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def load_json(path: Path) -> dict[str, Any]:
    absolute = _absolute_path(path)
    payload = _read_regular_nofollow(absolute, "JSON document")
    return load_json_bytes(payload, str(absolute))


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except GateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON document {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"JSON document {label} must contain one object")
    return document


def sha256_file(path: Path) -> str:
    absolute = _absolute_path(path)
    return hashlib.sha256(_read_regular_nofollow(absolute, "file to hash")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _require_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_REVISION.fullmatch(value) is None:
        raise GateError(f"{label} must be a full lowercase Git revision")
    return value


def _require_expected_revision(actual: Any, expected: str, label: str) -> str:
    actual_revision = _require_revision(actual, label)
    expected_revision = _require_revision(expected, f"expected {label}")
    if actual_revision != expected_revision:
        raise GateError(
            f"{label} mismatch: manifest has {actual_revision}, expected "
            f"{expected_revision}"
        )
    return actual_revision


def _require_run_id(value: Any, expected: str | None = None) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise GateError("run_id must match [A-Za-z0-9_-]{1,128}")
    if expected is not None:
        if not isinstance(expected, str) or _RUN_ID.fullmatch(expected) is None:
            raise GateError("expected run_id must match [A-Za-z0-9_-]{1,128}")
        if value != expected:
            raise GateError(f"run_id mismatch: manifest has {value!r}, expected {expected!r}")
    return value


def _require_exact_sequence(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise GateError(f"{label} must equal {list(expected)!r} in that exact order")


def _read_exact(handle: Any, length: int, path: Path, label: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise GateError(f"truncated PNG {path}: missing {label}")
    return value


def _read_png_shape(handle: Any, path: Path) -> tuple[int, int, int]:
    if _read_exact(handle, 8, path, "signature") != b"\x89PNG\r\n\x1a\n":
        raise GateError(f"panel {path} is not a PNG")
    saw_ihdr = False
    saw_idat = False
    saw_iend = False
    shape: tuple[int, int, int] | None = None
    while not saw_iend:
        length = struct.unpack(">I", _read_exact(handle, 4, path, "chunk length"))[0]
        chunk_type = _read_exact(handle, 4, path, "chunk type")
        if length > 256 * 1024 * 1024:
            raise GateError(f"PNG chunk in {path} exceeds the safety limit")
        payload = _read_exact(handle, length, path, "chunk payload")
        expected_crc = struct.unpack(">I", _read_exact(handle, 4, path, "chunk CRC"))[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise GateError(f"PNG chunk CRC mismatch in {path}")

        if not saw_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise GateError(f"PNG {path} does not start with a valid IHDR")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            metadata = (bit_depth, color_type, compression, filtering, interlace)
            if metadata != (8, 2, 0, 0, 0):
                raise GateError(
                    f"panel {path} must be non-interlaced 8-bit RGB PNG; "
                    f"metadata is {metadata}"
                )
            shape = (height, width, 3)
            saw_ihdr = True
        elif chunk_type == b"IHDR":
            raise GateError(f"PNG {path} contains multiple IHDR chunks")

        if chunk_type == b"IDAT":
            saw_idat = True
        elif chunk_type == b"IEND":
            if length != 0:
                raise GateError(f"PNG {path} has a non-empty IEND chunk")
            saw_iend = True

    if handle.read(1):
        raise GateError(f"PNG {path} has trailing bytes after IEND")
    if not saw_idat or shape is None:
        raise GateError(f"PNG {path} is missing required image chunks")
    return shape


def read_png_shape(path: Path) -> tuple[int, int, int]:
    """Validate PNG framing/chunk CRCs and return its HWC shape."""

    absolute = _absolute_path(path)
    payload = _read_regular_nofollow(absolute, "panel PNG")
    return read_png_shape_bytes(payload, absolute)


def read_png_shape_bytes(payload: bytes, path: Path) -> tuple[int, int, int]:
    return _read_png_shape(io.BytesIO(payload), path)


def _resolve_artifact(path_value: Any, manifest_path: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise GateError(f"{label} path must be a non-empty string")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return _absolute_path(candidate)


def _validate_technical_manifest_bytes(
    manifest_path: Path,
    manifest_bytes: bytes,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_source_snapshot_attestation_sha256: str | None = None,
    expected_source_snapshot_inventory_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate one held technical-manifest byte snapshot and its panels."""

    manifest = load_json_bytes(manifest_bytes, str(manifest_path))
    if manifest.get("schema") != TECHNICAL_SCHEMA:
        raise GateError(f"technical manifest schema must be {TECHNICAL_SCHEMA!r}")
    if manifest.get("status") != TECHNICAL_STATUS:
        raise GateError(f"technical manifest status must be {TECHNICAL_STATUS!r}")
    run_id = _require_run_id(manifest.get("run_id"), expected_run_id)

    fast_revision = _require_expected_revision(
        manifest.get("fast_roborender_revision"),
        expected_fast_roborender_revision,
        "fast_roborender_revision",
    )
    gr00t_revision = _require_expected_revision(
        manifest.get("gr00t_revision"), expected_gr00t_revision, "gr00t_revision"
    )
    expected_hashes = {
        "fast_roborender_source_tree_sha256": expected_fast_source_tree_sha256,
        "gr00t_source_tree_sha256": expected_gr00t_source_tree_sha256,
        "source_snapshot_attestation_sha256": (
            expected_source_snapshot_attestation_sha256
        ),
        "source_snapshot_inventory_sha256": expected_source_snapshot_inventory_sha256,
        "immutable_asset_contract_sha256": expected_asset_contract_sha256,
    }
    if any(value is not None for value in expected_hashes.values()) and any(
        value is None for value in expected_hashes.values()
    ):
        raise GateError(
            "source-tree, snapshot, and immutable-asset expectations must be supplied together"
        )
    normalized_hashes: dict[str, str] = {}
    for key, expected in expected_hashes.items():
        actual = manifest.get(key)
        if expected is not None:
            actual_hash = _require_sha256(actual, key)
            expected_hash = _require_sha256(expected, f"expected {key}")
            if actual_hash != expected_hash:
                raise GateError(
                    f"{key} mismatch: manifest has {actual_hash}, expected {expected_hash}"
                )
            normalized_hashes[key] = actual_hash
        elif actual is not None:
            normalized_hashes[key] = _require_sha256(actual, key)
    _require_exact_sequence(manifest.get("policy_views"), EXPECTED_VIEWS, "policy_views")
    _require_exact_sequence(manifest.get("columns"), EXPECTED_VIEWS, "columns")
    _require_exact_sequence(manifest.get("rows"), EXPECTED_ROWS, "rows")
    _require_exact_sequence(
        manifest.get("tile_shape_hwc"), EXPECTED_TILE_SHAPE_HWC, "tile_shape_hwc"
    )
    _require_exact_sequence(
        manifest.get("panel_shape_hwc"), EXPECTED_PANEL_SHAPE_HWC, "panel_shape_hwc"
    )

    panel_entries = manifest.get("panels")
    if not isinstance(panel_entries, list) or len(panel_entries) != len(
        EXPECTED_PANEL_NAMES
    ):
        raise GateError("technical manifest must contain exactly two panel entries")
    normalized_panels: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, (entry, expected_name) in enumerate(
        zip(panel_entries, EXPECTED_PANEL_NAMES, strict=True)
    ):
        if not isinstance(entry, dict):
            raise GateError(f"panels[{index}] must be an object with path and sha256")
        panel_path = _resolve_artifact(entry.get("path"), manifest_path, f"panels[{index}]")
        if panel_path.name != expected_name:
            raise GateError(
                f"panels[{index}] must name {expected_name!r}, got {panel_path.name!r}"
            )
        if panel_path in seen_paths:
            raise GateError("technical manifest contains a duplicate panel path")
        seen_paths.add(panel_path)
        declared_hash = _require_sha256(entry.get("sha256"), f"panels[{index}].sha256")
        panel_bytes = _read_regular_nofollow(panel_path, f"panels[{index}]")
        actual_hash = hashlib.sha256(panel_bytes).hexdigest()
        if actual_hash != declared_hash:
            raise GateError(
                f"panels[{index}] SHA-256 mismatch: manifest has {declared_hash}, "
                f"file has {actual_hash}"
            )
        actual_shape = read_png_shape_bytes(panel_bytes, panel_path)
        if actual_shape != EXPECTED_PANEL_SHAPE_HWC:
            raise GateError(
                f"panels[{index}] shape must be {EXPECTED_PANEL_SHAPE_HWC}, "
                f"got {actual_shape}"
            )
        normalized_panels.append(
            {
                "path": str(panel_path),
                "sha256": actual_hash,
                "shape_hwc": list(actual_shape),
            }
        )

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "run_id": run_id,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        **normalized_hashes,
        "policy_views": list(EXPECTED_VIEWS),
        "columns": list(EXPECTED_VIEWS),
        "rows": list(EXPECTED_ROWS),
        "tile_shape_hwc": list(EXPECTED_TILE_SHAPE_HWC),
        "panel_shape_hwc": list(EXPECTED_PANEL_SHAPE_HWC),
        "panels": normalized_panels,
    }


def validate_technical_manifest(
    manifest_path: Path,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_source_snapshot_attestation_sha256: str | None = None,
    expected_source_snapshot_inventory_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate the complete technical calibration artifact set."""

    manifest_path = _absolute_path(manifest_path)
    manifest_bytes = _read_regular_nofollow(
        manifest_path, "technical calibration manifest"
    )
    return _validate_technical_manifest_bytes(
        manifest_path,
        manifest_bytes,
        expected_fast_roborender_revision,
        expected_gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_source_snapshot_attestation_sha256,
        expected_source_snapshot_inventory_sha256,
        expected_asset_contract_sha256,
        expected_run_id,
    )


def _validate_reviewer(reviewer: Any) -> str:
    if not isinstance(reviewer, str):
        raise GateError("reviewer must be explicitly provided as text")
    reviewer = reviewer.strip()
    if not reviewer:
        raise GateError("reviewer must be explicitly provided")
    if len(reviewer) > 256 or any(ord(character) < 32 for character in reviewer):
        raise GateError("reviewer contains invalid control characters or is too long")
    return reviewer


def _absolute_path(path: Path) -> Path:
    """Normalize lexical components without following the final path entry."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_canonical_directory(path: Path, label: str) -> tuple[Path, int]:
    absolute = _absolute_path(path)
    try:
        metadata = os.lstat(absolute)
    except OSError as exc:
        raise GateError(f"cannot inspect {label} {absolute}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise GateError(f"{label} must be a real directory, not a symlink: {absolute}")
    try:
        canonical = absolute.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} {absolute}: {exc}") from exc
    if canonical != absolute:
        raise GateError(f"{label} must be supplied as its canonical path: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise GateError(f"cannot safely open {label} {absolute}: {exc}") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise GateError(f"{label} changed while it was opened: {absolute}")
    return absolute, descriptor


def _verify_directory_anchor(path: Path, descriptor: int, label: str) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.lstat(path)
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"{label} changed after it was opened: {path}: {exc}") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or canonical != path
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise GateError(f"{label} changed after it was opened: {path}")


def _verify_visible_entry(
    directory: Path,
    directory_descriptor: int,
    entry_name: str,
    expected_identity: tuple[int, int],
    *,
    require_directory: bool = False,
) -> None:
    _verify_directory_anchor(
        directory, directory_descriptor, "calibration publication directory"
    )
    try:
        held = os.stat(
            entry_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        visible = os.lstat(directory / entry_name)
    except OSError as exc:
        raise GateError(f"published entry is not visible at {directory / entry_name}: {exc}") from exc
    expected_type = stat.S_ISDIR if require_directory else stat.S_ISREG
    if (
        not expected_type(held.st_mode)
        or not expected_type(visible.st_mode)
        or (held.st_dev, held.st_ino) != expected_identity
        or (visible.st_dev, visible.st_ino) != expected_identity
    ):
        raise GateError(f"published entry changed or is not visible at {directory / entry_name}")


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular_nofollow(path: Path, label: str) -> bytes:
    absolute = _absolute_path(path)
    try:
        visible_before = os.lstat(absolute)
        canonical = absolute.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot safely inspect {label} {absolute}: {exc}") from exc
    if not stat.S_ISREG(visible_before.st_mode):
        raise GateError(f"{label} is not a regular non-symlink file: {absolute}")
    if canonical != absolute:
        raise GateError(f"{label} must be supplied as its canonical path: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise GateError(f"cannot safely open {label} {absolute}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (visible_before.st_dev, visible_before.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise GateError(f"{label} is not a regular non-symlink file: {absolute}")
        payload = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        try:
            visible_after = os.lstat(absolute)
        except OSError as exc:
            raise GateError(f"{label} changed while it was read: {absolute}") from exc
        stable_fields = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if (
            stable_fields
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            or not stat.S_ISREG(visible_after.st_mode)
            or (visible_after.st_dev, visible_after.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise GateError(f"{label} changed while it was read: {absolute}")
        return payload
    finally:
        os.close(descriptor)


def _read_existing_publication(
    directory_descriptor: int, destination_name: str
) -> tuple[bytes, int, tuple[int, int], int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(destination_name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise GateError(
            "existing calibration manifest is not a safe regular file: "
            f"{destination_name}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise GateError(
                "existing calibration manifest is not a regular non-symlink file: "
                f"{destination_name}"
            )
        if stat.S_IMODE(opened.st_mode) != 0o444:
            raise GateError(
                "existing calibration manifest mode must be exactly 0444: "
                f"{destination_name}"
            )
        payload = _read_descriptor(descriptor)
        opened_after = os.fstat(descriptor)
        try:
            current = os.stat(
                destination_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise GateError(
                f"existing calibration manifest changed while read: {destination_name}"
            ) from exc
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
        ) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise GateError(
                f"existing calibration manifest changed while read: {destination_name}"
            )
        return (
            payload,
            descriptor,
            (opened.st_dev, opened.st_ino),
            opened.st_mtime_ns,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_held_publication_inode(
    descriptor: int,
    payload: bytes,
    expected_identity: tuple[int, int],
    expected_mtime_ns: int,
) -> None:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or (before.st_dev, before.st_ino) != expected_identity
        or before.st_size != len(payload)
        or before.st_mtime_ns != expected_mtime_ns
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise GateError("held calibration publication inode changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    held_payload = _read_descriptor(descriptor)
    after = os.fstat(descriptor)
    if (
        held_payload != payload
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_mode,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
        )
    ):
        raise GateError("held calibration publication inode changed")


def atomic_publish_bytes_once(
    payload: bytes, destination: Path, validated_directory: Path
) -> str:
    """Publish complete bytes once without following or replacing any final entry."""

    directory, directory_descriptor = _open_canonical_directory(
        validated_directory, "calibration validation directory"
    )
    destination = _absolute_path(destination)
    if destination.parent != directory:
        os.close(directory_descriptor)
        raise GateError("calibration manifest destination must be a direct validation child")
    temporary_name: str | None = None
    temporary_descriptor: int | None = None
    existing_descriptor: int | None = None
    existing_mtime_ns: int | None = None
    final_identity: tuple[int, int] | None = None
    try:
        _verify_directory_anchor(
            directory, directory_descriptor, "calibration validation directory"
        )
        for _ in range(64):
            candidate = f".{destination.name}.tmp.{os.getpid()}.{secrets.token_hex(12)}"
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                temporary_descriptor = os.open(
                    candidate, flags, 0o600, dir_fd=directory_descriptor
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_descriptor is None or temporary_name is None:
            raise GateError("cannot allocate a private calibration publication temp file")

        view = memoryview(payload)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise GateError("short write to calibration publication temp file")
            view = view[written:]
        os.fchmod(temporary_descriptor, 0o444)
        os.fsync(temporary_descriptor)
        temporary_metadata = os.fstat(temporary_descriptor)
        temporary_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        temporary_mtime_ns = temporary_metadata.st_mtime_ns
        _verify_held_publication_inode(
            temporary_descriptor,
            payload,
            temporary_identity,
            temporary_mtime_ns,
        )

        _verify_directory_anchor(
            directory, directory_descriptor, "calibration validation directory"
        )
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            outcome = "published"
            final_identity = temporary_identity
        except FileExistsError:
            (
                existing,
                existing_descriptor,
                final_identity,
                existing_mtime_ns,
            ) = _read_existing_publication(directory_descriptor, destination.name)
            if existing != payload:
                raise GateError(
                    "refusing to replace different calibration manifest: "
                    f"{destination}"
                )
            outcome = "already_present"

        _verify_visible_entry(
            directory,
            directory_descriptor,
            destination.name,
            final_identity,
        )
        _verify_held_publication_inode(
            temporary_descriptor,
            payload,
            temporary_identity,
            temporary_mtime_ns,
        )
        if existing_descriptor is not None and existing_mtime_ns is not None:
            _verify_held_publication_inode(
                existing_descriptor,
                payload,
                final_identity,
                existing_mtime_ns,
            )

        # The final name is a hard link to a fully written and fsynced inode.
        # Removing the private name cannot expose a partially written final.
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_name = None
        os.fsync(directory_descriptor)
        _verify_visible_entry(
            directory,
            directory_descriptor,
            destination.name,
            final_identity,
        )
        _verify_held_publication_inode(
            temporary_descriptor,
            payload,
            temporary_identity,
            temporary_mtime_ns,
        )
        if existing_descriptor is not None and existing_mtime_ns is not None:
            _verify_held_publication_inode(
                existing_descriptor,
                payload,
                final_identity,
                existing_mtime_ns,
            )
            os.close(existing_descriptor)
            existing_descriptor = None
        os.close(temporary_descriptor)
        temporary_descriptor = None
        return outcome
    finally:
        try:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            if existing_descriptor is not None:
                os.close(existing_descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory_descriptor)


def _capture_technical_snapshot(
    manifest_path: Path,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str,
    expected_gr00t_source_tree_sha256: str,
    expected_source_snapshot_attestation_sha256: str,
    expected_source_snapshot_inventory_sha256: str,
    expected_asset_contract_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    manifest_path = _absolute_path(manifest_path)
    manifest_bytes = _read_regular_nofollow(
        manifest_path, "source technical calibration manifest"
    )
    technical = _validate_technical_manifest_bytes(
        manifest_path,
        manifest_bytes,
        expected_fast_roborender_revision,
        expected_gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_source_snapshot_attestation_sha256,
        expected_source_snapshot_inventory_sha256,
        expected_asset_contract_sha256,
        expected_run_id,
    )
    panel_digests: dict[str, str] = {}
    for index, panel in enumerate(technical["panels"]):
        panel_path = Path(panel["path"])
        panel_bytes = _read_regular_nofollow(panel_path, f"source calibration panel {index}")
        digest = hashlib.sha256(panel_bytes).hexdigest()
        if digest != panel["sha256"]:
            raise GateError(f"source calibration panel {index} changed during validation")
        panel_digests[str(panel_path)] = digest
    return {
        "technical": technical,
        "manifest_path": manifest_path,
        "manifest_bytes": manifest_bytes,
        "panel_digests": panel_digests,
    }


def _verify_technical_snapshot_unchanged(snapshot: dict[str, Any]) -> None:
    current_manifest = _read_regular_nofollow(
        snapshot["manifest_path"], "source technical calibration manifest"
    )
    if current_manifest != snapshot["manifest_bytes"]:
        raise GateError("source technical calibration manifest changed before publication")
    for index, (path_value, expected_digest) in enumerate(
        snapshot["panel_digests"].items()
    ):
        current_panel = _read_regular_nofollow(
            Path(path_value), f"source calibration panel {index}"
        )
        if hashlib.sha256(current_panel).hexdigest() != expected_digest:
            raise GateError(f"source calibration panel {index} changed before publication")


def publish_technical_manifest_once(
    manifest_path: Path,
    destination: Path,
    validation_directory: Path,
    expected_run_id: str,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str,
    expected_gr00t_source_tree_sha256: str,
    expected_source_snapshot_attestation_sha256: str,
    expected_source_snapshot_inventory_sha256: str,
    expected_asset_contract_sha256: str,
) -> dict[str, Any]:
    run_id = _require_run_id(expected_run_id, expected_run_id)
    validation_directory = _absolute_path(validation_directory)
    expected_destination = (
        validation_directory / f"camera_calibration_manifest_{run_id}.json"
    )
    if _absolute_path(destination) != expected_destination:
        raise GateError(f"calibration manifest destination does not exactly bind RUN_ID={run_id}")
    snapshot = _capture_technical_snapshot(
        manifest_path,
        expected_fast_roborender_revision,
        expected_gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_source_snapshot_attestation_sha256,
        expected_source_snapshot_inventory_sha256,
        expected_asset_contract_sha256,
        run_id,
    )
    # Re-read the exact manifest and every panel after validation and directly
    # before the no-replace link.  The final manifest bytes themselves come
    # from the held validated snapshot, never from a later mutable source read.
    _verify_technical_snapshot_unchanged(snapshot)
    outcome = atomic_publish_bytes_once(
        snapshot["manifest_bytes"], expected_destination, validation_directory
    )
    return {
        **snapshot["technical"],
        "publication": outcome,
        "published_path": str(expected_destination),
    }


def reserve_calibration_run_root(
    runtime_directory: Path, run_root: Path, run_id: str
) -> Path:
    run_id = _require_run_id(run_id, run_id)
    runtime_directory, descriptor = _open_canonical_directory(
        runtime_directory, "runtime directory"
    )
    expected = runtime_directory / run_id
    if _absolute_path(run_root) != expected:
        os.close(descriptor)
        raise GateError(f"calibration RUN_ROOT must be exactly {expected}")
    created_identity: tuple[int, int] | None = None
    try:
        _verify_directory_anchor(runtime_directory, descriptor, "runtime directory")
        try:
            os.mkdir(run_id, 0o755, dir_fd=descriptor)
        except FileExistsError as exc:
            raise GateError(
                f"fresh calibration refuses to reuse existing RUN_ROOT={expected}"
            ) from exc
        created = os.stat(run_id, dir_fd=descriptor, follow_symlinks=False)
        created_identity = (created.st_dev, created.st_ino)
        os.fsync(descriptor)
        _verify_visible_entry(
            runtime_directory,
            descriptor,
            run_id,
            created_identity,
            require_directory=True,
        )
        created_identity = None
    finally:
        try:
            if created_identity is not None:
                try:
                    current = os.stat(
                        run_id, dir_fd=descriptor, follow_symlinks=False
                    )
                    if (current.st_dev, current.st_ino) == created_identity:
                        os.rmdir(run_id, dir_fd=descriptor)
                        os.fsync(descriptor)
                except FileNotFoundError:
                    pass
        finally:
            os.close(descriptor)
    return expected


def atomic_json_create(path: Path, document: dict[str, Any]) -> None:
    """Atomically create a JSON record without replacing prior acceptance."""

    path = _absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise GateError(f"cannot encode acceptance record: {exc}") from exc
    outcome = atomic_publish_bytes_once(payload, path, path.parent)
    if outcome != "published":
        raise GateError(f"refusing to replace existing acceptance record {path}")


def accept_manifest(
    manifest_path: Path,
    output_path: Path,
    reviewer: str,
    confirmation: str,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_source_snapshot_attestation_sha256: str | None = None,
    expected_source_snapshot_inventory_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise GateError(
            "explicit confirmation does not match the required phrase: " + CONFIRMATION
        )
    technical = validate_technical_manifest(
        manifest_path,
        expected_fast_roborender_revision,
        expected_gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_source_snapshot_attestation_sha256,
        expected_source_snapshot_inventory_sha256,
        expected_asset_contract_sha256,
        expected_run_id,
    )
    acceptance = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": ACCEPTANCE_STATUS,
        "accepted_at_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "reviewer": _validate_reviewer(reviewer),
        "confirmation": CONFIRMATION,
        "run_id": technical["run_id"],
        "source_manifest": {
            "path": technical["manifest_path"],
            "sha256": technical["manifest_sha256"],
        },
        "fast_roborender_revision": technical["fast_roborender_revision"],
        "gr00t_revision": technical["gr00t_revision"],
        "policy_views": technical["policy_views"],
        "columns": technical["columns"],
        "rows": technical["rows"],
        "tile_shape_hwc": technical["tile_shape_hwc"],
        "panel_shape_hwc": technical["panel_shape_hwc"],
        "panels": technical["panels"],
    }
    for key in (
        "fast_roborender_source_tree_sha256",
        "gr00t_source_tree_sha256",
        "source_snapshot_attestation_sha256",
        "source_snapshot_inventory_sha256",
        "immutable_asset_contract_sha256",
    ):
        if key in technical:
            acceptance[key] = technical[key]
    atomic_json_create(output_path, acceptance)
    return acceptance


def _validate_acceptance_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GateError("accepted_at_utc must be an ISO-8601 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise GateError("accepted_at_utc must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise GateError("accepted_at_utc must use UTC")


def verify_acceptance(
    acceptance_path: Path,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    *,
    manifest_path: Path | None = None,
    expected_acceptance_sha256: str | None = None,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_source_snapshot_attestation_sha256: str | None = None,
    expected_source_snapshot_inventory_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    if expected_run_id is None:
        raise GateError("expected run_id is required for acceptance verification")
    acceptance_path = _absolute_path(acceptance_path)
    acceptance_bytes = _read_regular_nofollow(
        acceptance_path, "camera calibration acceptance"
    )
    acceptance_hash = hashlib.sha256(acceptance_bytes).hexdigest()
    if expected_acceptance_sha256 is not None:
        expected_hash = _require_sha256(
            expected_acceptance_sha256, "expected acceptance SHA-256"
        )
        if acceptance_hash != expected_hash:
            raise GateError(
                f"acceptance SHA-256 mismatch: file has {acceptance_hash}, "
                f"expected {expected_hash}"
            )

    acceptance = load_json_bytes(acceptance_bytes, str(acceptance_path))
    if acceptance.get("schema") != ACCEPTANCE_SCHEMA:
        raise GateError(f"acceptance schema must be {ACCEPTANCE_SCHEMA!r}")
    if acceptance.get("status") != ACCEPTANCE_STATUS:
        raise GateError(f"acceptance status must be {ACCEPTANCE_STATUS!r}")
    _validate_reviewer(acceptance.get("reviewer", ""))
    if acceptance.get("confirmation") != CONFIRMATION:
        raise GateError("acceptance is missing the exact explicit confirmation")
    _validate_acceptance_timestamp(acceptance.get("accepted_at_utc"))
    fast_revision = _require_expected_revision(
        acceptance.get("fast_roborender_revision"),
        expected_fast_roborender_revision,
        "fast_roborender_revision",
    )
    gr00t_revision = _require_expected_revision(
        acceptance.get("gr00t_revision"), expected_gr00t_revision, "gr00t_revision"
    )
    run_id = _require_run_id(acceptance.get("run_id"), expected_run_id)

    source = acceptance.get("source_manifest")
    if not isinstance(source, dict):
        raise GateError("acceptance source_manifest must be an object")
    recorded_manifest = _resolve_artifact(
        source.get("path"), acceptance_path, "source_manifest"
    )
    if manifest_path is not None:
        explicit_manifest = _absolute_path(manifest_path)
        if explicit_manifest != recorded_manifest:
            raise GateError(
                "explicit technical manifest path does not match the accepted manifest path"
            )
    declared_manifest_hash = _require_sha256(
        source.get("sha256"), "source_manifest.sha256"
    )
    manifest_bytes = _read_regular_nofollow(
        recorded_manifest, "accepted technical calibration manifest"
    )
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_hash != declared_manifest_hash:
        raise GateError(
            f"technical manifest SHA-256 mismatch: acceptance has "
            f"{declared_manifest_hash}, file has {actual_manifest_hash}"
        )
    technical = _validate_technical_manifest_bytes(
        recorded_manifest,
        manifest_bytes,
        fast_revision,
        gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_source_snapshot_attestation_sha256,
        expected_source_snapshot_inventory_sha256,
        expected_asset_contract_sha256,
        run_id,
    )

    if technical["run_id"] != run_id:
        raise GateError("acceptance run_id does not match its technical manifest")

    for key in (
        "policy_views",
        "columns",
        "rows",
        "tile_shape_hwc",
        "panel_shape_hwc",
        "panels",
    ):
        if acceptance.get(key) != technical[key]:
            raise GateError(f"acceptance {key} does not match its technical manifest")
    for key in (
        "fast_roborender_source_tree_sha256",
        "gr00t_source_tree_sha256",
        "source_snapshot_attestation_sha256",
        "source_snapshot_inventory_sha256",
        "immutable_asset_contract_sha256",
    ):
        if key in technical and acceptance.get(key) != technical[key]:
            raise GateError(f"acceptance {key} does not match its technical manifest")

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "verified",
        "acceptance_path": str(acceptance_path),
        "acceptance_sha256": acceptance_hash,
        "source_manifest_path": technical["manifest_path"],
        "source_manifest_sha256": technical["manifest_sha256"],
        "run_id": run_id,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        **{
            key: technical[key]
            for key in (
                "fast_roborender_source_tree_sha256",
                "gr00t_source_tree_sha256",
                "source_snapshot_attestation_sha256",
                "source_snapshot_inventory_sha256",
                "immutable_asset_contract_sha256",
            )
            if key in technical
        },
        "policy_views": list(EXPECTED_VIEWS),
        "panels_verified": len(technical["panels"]),
        "reviewer": acceptance["reviewer"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    accept = subparsers.add_parser("accept", help="record explicit human acceptance")
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    accept.add_argument("--reviewer", required=True)
    accept.add_argument("--confirm", required=True, help=f"must equal: {CONFIRMATION}")
    accept.add_argument(
        "--expected-fast-roborender-revision", "--expected-fast-revision", required=True
    )
    accept.add_argument("--expected-gr00t-revision", required=True)
    accept.add_argument("--expected-fast-source-tree-sha256")
    accept.add_argument("--expected-gr00t-source-tree-sha256")
    accept.add_argument("--expected-source-snapshot-attestation-sha256")
    accept.add_argument("--expected-source-snapshot-inventory-sha256")
    accept.add_argument("--expected-asset-contract-sha256")
    accept.add_argument("--expected-run-id", required=True)

    verify = subparsers.add_parser("verify", help="verify an acceptance and all inputs")
    verify.add_argument("--acceptance", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument(
        "--expected-fast-roborender-revision", "--expected-fast-revision", required=True
    )
    verify.add_argument("--expected-gr00t-revision", required=True)
    verify.add_argument("--expected-acceptance-sha256")
    verify.add_argument("--expected-fast-source-tree-sha256")
    verify.add_argument("--expected-gr00t-source-tree-sha256")
    verify.add_argument("--expected-source-snapshot-attestation-sha256")
    verify.add_argument("--expected-source-snapshot-inventory-sha256")
    verify.add_argument("--expected-asset-contract-sha256")
    verify.add_argument("--expected-run-id", required=True)

    publish = subparsers.add_parser(
        "publish", help="atomically publish a validated technical manifest once"
    )
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--destination", type=Path, required=True)
    publish.add_argument("--validation-directory", type=Path, required=True)
    publish.add_argument("--expected-run-id", required=True)
    publish.add_argument(
        "--expected-fast-roborender-revision", "--expected-fast-revision", required=True
    )
    publish.add_argument("--expected-gr00t-revision", required=True)
    publish.add_argument("--expected-fast-source-tree-sha256", required=True)
    publish.add_argument("--expected-gr00t-source-tree-sha256", required=True)
    publish.add_argument(
        "--expected-source-snapshot-attestation-sha256", required=True
    )
    publish.add_argument("--expected-source-snapshot-inventory-sha256", required=True)
    publish.add_argument("--expected-asset-contract-sha256", required=True)

    reserve = subparsers.add_parser(
        "reserve-run-root", help="exclusively reserve a fresh calibration RUN_ROOT"
    )
    reserve.add_argument("--runtime-directory", type=Path, required=True)
    reserve.add_argument("--run-root", type=Path, required=True)
    reserve.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "accept":
            accept_manifest(
                args.manifest,
                args.output,
                args.reviewer,
                args.confirm,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                args.expected_fast_source_tree_sha256,
                args.expected_gr00t_source_tree_sha256,
                args.expected_source_snapshot_attestation_sha256,
                args.expected_source_snapshot_inventory_sha256,
                args.expected_asset_contract_sha256,
                args.expected_run_id,
            )
            report = verify_acceptance(
                args.output,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                manifest_path=args.manifest,
                expected_fast_source_tree_sha256=args.expected_fast_source_tree_sha256,
                expected_gr00t_source_tree_sha256=args.expected_gr00t_source_tree_sha256,
                expected_source_snapshot_attestation_sha256=(
                    args.expected_source_snapshot_attestation_sha256
                ),
                expected_source_snapshot_inventory_sha256=(
                    args.expected_source_snapshot_inventory_sha256
                ),
                expected_asset_contract_sha256=args.expected_asset_contract_sha256,
                expected_run_id=args.expected_run_id,
            )
        elif args.command == "verify":
            report = verify_acceptance(
                args.acceptance,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                manifest_path=args.manifest,
                expected_acceptance_sha256=args.expected_acceptance_sha256,
                expected_fast_source_tree_sha256=args.expected_fast_source_tree_sha256,
                expected_gr00t_source_tree_sha256=args.expected_gr00t_source_tree_sha256,
                expected_source_snapshot_attestation_sha256=(
                    args.expected_source_snapshot_attestation_sha256
                ),
                expected_source_snapshot_inventory_sha256=(
                    args.expected_source_snapshot_inventory_sha256
                ),
                expected_asset_contract_sha256=args.expected_asset_contract_sha256,
                expected_run_id=args.expected_run_id,
            )
        elif args.command == "publish":
            report = publish_technical_manifest_once(
                args.manifest,
                args.destination,
                args.validation_directory,
                args.expected_run_id,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                args.expected_fast_source_tree_sha256,
                args.expected_gr00t_source_tree_sha256,
                args.expected_source_snapshot_attestation_sha256,
                args.expected_source_snapshot_inventory_sha256,
                args.expected_asset_contract_sha256,
            )
        else:
            reserved = reserve_calibration_run_root(
                args.runtime_directory, args.run_root, args.run_id
            )
            report = {
                "status": "reserved",
                "run_id": args.run_id,
                "run_root": str(reserved),
            }
    except (GateError, OSError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
