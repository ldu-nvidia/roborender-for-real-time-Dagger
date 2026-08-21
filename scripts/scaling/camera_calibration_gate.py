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
import json
import os
import re
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any


TECHNICAL_SCHEMA = "fast_roborender.camera_calibration.v1"
ACCEPTANCE_SCHEMA = "fast_roborender.camera_calibration_acceptance.v1"
VERIFICATION_SCHEMA = "fast_roborender.camera_calibration_verification.v1"
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
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"JSON document {path} must contain one object")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


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


def _require_exact_sequence(value: Any, expected: tuple[Any, ...], label: str) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise GateError(f"{label} must equal {list(expected)!r} in that exact order")


def _read_exact(handle: Any, length: int, path: Path, label: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise GateError(f"truncated PNG {path}: missing {label}")
    return value


def read_png_shape(path: Path) -> tuple[int, int, int]:
    """Validate PNG framing/chunk CRCs and return its HWC shape."""

    try:
        with path.open("rb") as handle:
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
                expected_crc = struct.unpack(
                    ">I", _read_exact(handle, 4, path, "chunk CRC")
                )[0]
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
    except GateError:
        raise
    except OSError as exc:
        raise GateError(f"cannot read panel PNG {path}: {exc}") from exc

    if not saw_idat or shape is None:
        raise GateError(f"PNG {path} is missing required image chunks")
    return shape


def _resolve_artifact(path_value: Any, manifest_path: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise GateError(f"{label} path must be a non-empty string")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} path {candidate}: {exc}") from exc
    if not resolved.is_file():
        raise GateError(f"{label} path is not a regular file: {resolved}")
    return resolved


def validate_technical_manifest(
    manifest_path: Path,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete technical calibration artifact set."""

    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve technical manifest {manifest_path}: {exc}") from exc
    manifest = load_json(manifest_path)
    if manifest.get("schema") != TECHNICAL_SCHEMA:
        raise GateError(f"technical manifest schema must be {TECHNICAL_SCHEMA!r}")
    if manifest.get("status") != TECHNICAL_STATUS:
        raise GateError(f"technical manifest status must be {TECHNICAL_STATUS!r}")

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
        "immutable_asset_contract_sha256": expected_asset_contract_sha256,
    }
    if any(value is not None for value in expected_hashes.values()) and any(
        value is None for value in expected_hashes.values()
    ):
        raise GateError(
            "source-tree and immutable-asset expectations must be supplied together"
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
        actual_hash = sha256_file(panel_path)
        if actual_hash != declared_hash:
            raise GateError(
                f"panels[{index}] SHA-256 mismatch: manifest has {declared_hash}, "
                f"file has {actual_hash}"
            )
        actual_shape = read_png_shape(panel_path)
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
        "manifest_sha256": sha256_file(manifest_path),
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


def _validate_reviewer(reviewer: Any) -> str:
    if not isinstance(reviewer, str):
        raise GateError("reviewer must be explicitly provided as text")
    reviewer = reviewer.strip()
    if not reviewer:
        raise GateError("reviewer must be explicitly provided")
    if len(reviewer) > 256 or any(ord(character) < 32 for character in reviewer):
        raise GateError("reviewer contains invalid control characters or is too long")
    return reviewer


def atomic_json_create(path: Path, document: dict[str, Any]) -> None:
    """Atomically create a JSON record without replacing prior acceptance."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise GateError(f"refusing to replace existing acceptance record {path}") from exc
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Some network filesystems do not permit directory fsync.  The
            # atomic hard-link publication above remains the required barrier.
            pass
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def accept_manifest(
    manifest_path: Path,
    output_path: Path,
    reviewer: str,
    confirmation: str,
    expected_fast_roborender_revision: str,
    expected_gr00t_revision: str,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
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
        expected_asset_contract_sha256,
    )
    acceptance = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": ACCEPTANCE_STATUS,
        "accepted_at_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "reviewer": _validate_reviewer(reviewer),
        "confirmation": CONFIRMATION,
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
    expected_asset_contract_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        acceptance_path = acceptance_path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve acceptance record {acceptance_path}: {exc}") from exc
    acceptance_hash = sha256_file(acceptance_path)
    if expected_acceptance_sha256 is not None:
        expected_hash = _require_sha256(
            expected_acceptance_sha256, "expected acceptance SHA-256"
        )
        if acceptance_hash != expected_hash:
            raise GateError(
                f"acceptance SHA-256 mismatch: file has {acceptance_hash}, "
                f"expected {expected_hash}"
            )

    acceptance = load_json(acceptance_path)
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

    source = acceptance.get("source_manifest")
    if not isinstance(source, dict):
        raise GateError("acceptance source_manifest must be an object")
    recorded_manifest = _resolve_artifact(
        source.get("path"), acceptance_path, "source_manifest"
    )
    if manifest_path is not None:
        try:
            explicit_manifest = manifest_path.resolve(strict=True)
        except OSError as exc:
            raise GateError(f"cannot resolve expected technical manifest: {exc}") from exc
        if explicit_manifest != recorded_manifest:
            raise GateError(
                "explicit technical manifest path does not match the accepted manifest path"
            )
    declared_manifest_hash = _require_sha256(
        source.get("sha256"), "source_manifest.sha256"
    )
    actual_manifest_hash = sha256_file(recorded_manifest)
    if actual_manifest_hash != declared_manifest_hash:
        raise GateError(
            f"technical manifest SHA-256 mismatch: acceptance has "
            f"{declared_manifest_hash}, file has {actual_manifest_hash}"
        )
    technical = validate_technical_manifest(
        recorded_manifest,
        fast_revision,
        gr00t_revision,
        expected_fast_source_tree_sha256,
        expected_gr00t_source_tree_sha256,
        expected_asset_contract_sha256,
    )

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
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        **{
            key: technical[key]
            for key in (
                "fast_roborender_source_tree_sha256",
                "gr00t_source_tree_sha256",
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
    accept.add_argument("--expected-asset-contract-sha256")

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
    verify.add_argument("--expected-asset-contract-sha256")
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
                args.expected_asset_contract_sha256,
            )
            report = verify_acceptance(
                args.output,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                manifest_path=args.manifest,
                expected_fast_source_tree_sha256=args.expected_fast_source_tree_sha256,
                expected_gr00t_source_tree_sha256=args.expected_gr00t_source_tree_sha256,
                expected_asset_contract_sha256=args.expected_asset_contract_sha256,
            )
        else:
            report = verify_acceptance(
                args.acceptance,
                args.expected_fast_roborender_revision,
                args.expected_gr00t_revision,
                manifest_path=args.manifest,
                expected_acceptance_sha256=args.expected_acceptance_sha256,
                expected_fast_source_tree_sha256=args.expected_fast_source_tree_sha256,
                expected_gr00t_source_tree_sha256=args.expected_gr00t_source_tree_sha256,
                expected_asset_contract_sha256=args.expected_asset_contract_sha256,
            )
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
