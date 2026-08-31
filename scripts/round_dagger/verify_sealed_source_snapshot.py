#!/usr/bin/env python3
"""Fail-closed verification for a stage_hsg_source_snapshot.sh source tree.

The verifier is intentionally independent of Git and of the snapshot builder.  A
caller must pin the external inventory SHA-256.  The inventory is then checked
against every regular file in the sealed tree, including bytes and permission
bits, before the internal manifest and external attestation are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
INVENTORY_SCHEMA = "fast_roborender.hsg_source_inventory.v1"
MANIFEST_SCHEMA = "fast_roborender.hsg_source_snapshot.v2"
ATTESTATION_SCHEMA = "fast_roborender.hsg_source_snapshot_attestation.v2"
REPORT_SCHEMA = "fast_roborender.sealed_source_verification.v1"
CANONICAL_SOURCE_ROOT_NAME = "fast_roborender"


class VerificationError(RuntimeError):
    """The source snapshot does not satisfy the sealed-source contract."""


def _sealed_directory_mode(mode: int) -> bool:
    """Accept read/execute-only directories plus filesystem-forced setgid.

    Some managed parallel filesystems (including the production OCI Lustre
    mount) force the setgid bit on every directory so children inherit the
    allocation group.  Setgid does not grant write access.  Continue to reject
    every write bit, setuid, sticky bit, and any ordinary permission other than
    ``0555``.
    """

    permissions = stat.S_IMODE(mode)
    return permissions in {0o555, stat.S_ISGID | 0o555}


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        raise VerificationError(
            f"{label} field set mismatch: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON value {value!r} is forbidden")


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is not valid canonical JSON input") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must contain exactly one JSON object")
    return value


def _regular_file_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    """Read one non-writable regular file without following a final symlink."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"cannot safely open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError(f"{label} is not a regular file: {path}")
        if before.st_mode & 0o222:
            raise VerificationError(f"{label} is writable: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != before.st_size:
            raise VerificationError(f"{label} changed while it was read: {path}")
        return payload, before
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_inventory_path(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise VerificationError(f"inventory record {index} path is unsafe")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise VerificationError(f"inventory record {index} path is unsafe: {value!r}")
    return value


def _validate_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    _exact_keys(
        payload,
        {"schema", "file_count", "total_file_bytes", "files"},
        "source inventory",
    )
    if payload["schema"] != INVENTORY_SCHEMA:
        raise VerificationError("source inventory schema mismatch")
    records = payload["files"]
    if not isinstance(records, list):
        raise VerificationError("source inventory files must be a list")
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise VerificationError(f"inventory record {index} must be an object")
        _exact_keys(
            record,
            {"path", "mode", "bytes", "sha256"},
            f"inventory record {index}",
        )
        relative = _safe_inventory_path(record["path"], index)
        mode = record["mode"]
        if mode not in {"0o444", "0o555"}:
            raise VerificationError(f"inventory mode is invalid for {relative!r}")
        size = record["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise VerificationError(f"inventory size is invalid for {relative!r}")
        digest = record["sha256"]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise VerificationError(f"inventory SHA-256 is invalid for {relative!r}")
        validated.append(record)
    paths = [record["path"] for record in validated]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise VerificationError("source inventory paths are not sorted and unique")
    count = payload["file_count"]
    total = payload["total_file_bytes"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(validated):
        raise VerificationError("source inventory file_count mismatch")
    expected_total = sum(record["bytes"] for record in validated)
    if isinstance(total, bool) or not isinstance(total, int) or total != expected_total:
        raise VerificationError("source inventory total_file_bytes mismatch")
    return validated


def _tree_records(root: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Return exact records plus builder-compatible tree hashes.

    File payloads are streamed into the exact-file and tree digests.  A sealed
    production snapshot can be several gigabytes, so retaining each payload in
    a dictionary would turn a provenance check into a login-node memory hazard.
    """

    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise VerificationError(f"sealed source root is unavailable: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise VerificationError(f"sealed source root must be a real directory: {root}")
    if not _sealed_directory_mode(root_stat.st_mode):
        raise VerificationError(
            "sealed source root mode must be 0o555 with only optional "
            f"filesystem-forced setgid: {root}"
        )

    files: list[tuple[str, Path, int]] = []
    directories = [root]
    while directories:
        directory = directories.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise VerificationError(f"cannot enumerate sealed source directory: {directory}") from exc
        if directory != root and not entries:
            relative_directory = directory.relative_to(root).as_posix()
            raise VerificationError(
                f"sealed source contains an untracked empty directory: {relative_directory}"
            )
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise VerificationError(f"cannot inspect sealed source entry: {path}") from exc
            if stat.S_ISLNK(mode):
                raise VerificationError(f"sealed source contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                if entry.name == "__pycache__":
                    raise VerificationError(
                        f"sealed source contains executable bytecode directory: {relative}"
                    )
                if not _sealed_directory_mode(mode):
                    raise VerificationError(
                        "sealed source directory mode must be 0o555 with only "
                        f"optional filesystem-forced setgid: {relative}"
                    )
                directories.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise VerificationError(f"sealed source contains a special file: {relative}")
            if entry.name.endswith((".pyc", ".pyo")):
                raise VerificationError(
                    f"sealed source contains executable bytecode: {relative}"
                )
            files.append((relative, path, mode))

    records: list[dict[str, Any]] = []
    fast_tree = hashlib.sha256()
    gr00t_tree = hashlib.sha256()
    for relative, path, observed_mode in sorted(files, key=lambda item: item[0]):
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise VerificationError(f"cannot safely open sealed source file: {path}") from exc
        try:
            opened_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_mode != observed_mode
                or opened_stat.st_mode & 0o222
                or stat.S_IMODE(opened_stat.st_mode) not in {0o444, 0o555}
            ):
                raise VerificationError(f"sealed source file mode is invalid: {relative}")

            tree_targets: list[hashlib._Hash] = []  # type: ignore[name-defined]
            if ".git" not in PurePosixPath(relative).parts:
                encoded = relative.encode("utf-8")
                fast_tree.update(len(encoded).to_bytes(8, "big"))
                fast_tree.update(encoded)
                fast_tree.update(stat.S_IMODE(opened_stat.st_mode).to_bytes(2, "big"))
                fast_tree.update(b"F")
                tree_targets.append(fast_tree)
            marker = "GR00T-VisualSim2Real/"
            if relative.startswith(marker):
                nested_relative = relative[len(marker) :]
                if ".git" not in PurePosixPath(nested_relative).parts:
                    encoded = nested_relative.encode("utf-8")
                    gr00t_tree.update(len(encoded).to_bytes(8, "big"))
                    gr00t_tree.update(encoded)
                    gr00t_tree.update(
                        stat.S_IMODE(opened_stat.st_mode).to_bytes(2, "big")
                    )
                    gr00t_tree.update(b"F")
                    tree_targets.append(gr00t_tree)

            file_digest = hashlib.sha256()
            observed_bytes = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    observed_bytes += len(chunk)
                    file_digest.update(chunk)
                    for target in tree_targets:
                        target.update(chunk)
            after = os.fstat(descriptor)
            identity_before = (
                opened_stat.st_dev,
                opened_stat.st_ino,
                opened_stat.st_mode,
                opened_stat.st_size,
                opened_stat.st_mtime_ns,
                opened_stat.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after or observed_bytes != opened_stat.st_size:
                raise VerificationError(f"sealed source file changed while read: {relative}")
        finally:
            os.close(descriptor)
        records.append(
            {
                "path": relative,
                "mode": oct(stat.S_IMODE(opened_stat.st_mode)),
                "bytes": observed_bytes,
                "sha256": file_digest.hexdigest(),
            }
        )
    return records, fast_tree.hexdigest(), gr00t_tree.hexdigest()


def _validate_manifest(payload: dict[str, Any], root_name: str) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "snapshot_name",
            "fast_roborender_revision",
            "gr00t_revision",
            "fast_roborender_index_tree",
            "gr00t_index_tree",
            "export_method",
            "lfs_file_count",
            "lfs_total_bytes",
            "tracked_only",
            "lfs_pointer_oids_verified",
            "symlinks_rejected",
            "bytecode_rejected",
        },
        "source snapshot internal manifest",
    )
    required = {
        "schema": MANIFEST_SCHEMA,
        "status": "sealed_tracked_hydrated_sources",
        "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
        "tracked_only": True,
        "lfs_pointer_oids_verified": True,
        "symlinks_rejected": True,
        "bytecode_rejected": True,
    }
    for key, expected in required.items():
        if payload[key] != expected or type(payload[key]) is not type(expected):
            raise VerificationError(f"source snapshot internal manifest {key} mismatch")
    for key in (
        "fast_roborender_revision",
        "gr00t_revision",
        "fast_roborender_index_tree",
        "gr00t_index_tree",
    ):
        if not isinstance(payload[key], str) or REVISION_RE.fullmatch(payload[key]) is None:
            raise VerificationError(f"source snapshot internal manifest {key} is malformed")
    for key in ("lfs_file_count", "lfs_total_bytes"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VerificationError(f"source snapshot internal manifest {key} is invalid")
    expected_name = (
        f"fast_roborender_{payload['fast_roborender_revision'][:7]}_"
        f"{payload['gr00t_revision'][:7]}_sealed"
    )
    if payload["snapshot_name"] != expected_name:
        raise VerificationError(
            "sealed source manifest snapshot name mismatch: "
            f"{payload['snapshot_name']!r} != {expected_name!r}"
        )
    if root_name not in {expected_name, CANONICAL_SOURCE_ROOT_NAME}:
        raise VerificationError(
            "sealed source root name must be its content-addressed snapshot name "
            f"or {CANONICAL_SOURCE_ROOT_NAME!r}: {root_name!r}"
        )
    return payload


def _validate_revision_marker(payload: bytes, expected: str, label: str) -> None:
    try:
        actual = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{label} revision marker is not ASCII") from exc
    if actual != expected:
        raise VerificationError(
            f"{label} revision marker mismatch: {actual!r} != {expected!r}"
        )


def _validate_attestation(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    inventory_sha256: str,
    records: list[dict[str, Any]],
    fast_tree_sha256: str,
    gr00t_tree_sha256: str,
) -> None:
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "snapshot_name",
            "fast_roborender_revision",
            "gr00t_revision",
            "fast_roborender_index_tree",
            "gr00t_index_tree",
            "fast_roborender_source_tree_sha256",
            "gr00t_source_tree_sha256",
            "inventory_sha256",
            "deterministic_archive_sha256",
            "file_count",
            "total_file_bytes",
        },
        "source snapshot attestation",
    )
    expected = {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS",
        "snapshot_name": manifest["snapshot_name"],
        "fast_roborender_revision": manifest["fast_roborender_revision"],
        "gr00t_revision": manifest["gr00t_revision"],
        "fast_roborender_index_tree": manifest["fast_roborender_index_tree"],
        "gr00t_index_tree": manifest["gr00t_index_tree"],
        "fast_roborender_source_tree_sha256": fast_tree_sha256,
        "gr00t_source_tree_sha256": gr00t_tree_sha256,
        "inventory_sha256": inventory_sha256,
        "file_count": len(records),
        "total_file_bytes": sum(record["bytes"] for record in records),
    }
    for key, value in expected.items():
        if payload[key] != value or type(payload[key]) is not type(value):
            raise VerificationError(f"source snapshot attestation {key} mismatch")
    archive_sha = payload["deterministic_archive_sha256"]
    if not isinstance(archive_sha, str) or SHA256_RE.fullmatch(archive_sha) is None:
        raise VerificationError("source snapshot deterministic archive SHA-256 is malformed")


def verify(root: Path, expected_inventory_sha256: str) -> dict[str, Any]:
    """Verify a sealed snapshot and return content-bound evidence."""

    if SHA256_RE.fullmatch(expected_inventory_sha256) is None:
        raise VerificationError("expected inventory SHA-256 is malformed")
    root = Path(os.path.abspath(root))
    inventory_path = root.parent / f"{root.name}.inventory.json"
    attestation_path = root.parent / f"{root.name}.snapshot_attestation.json"

    inventory_bytes, inventory_stat = _regular_file_bytes(
        inventory_path, "source snapshot inventory"
    )
    if stat.S_IMODE(inventory_stat.st_mode) != 0o444:
        raise VerificationError("source snapshot inventory mode must be exactly 0o444")
    inventory_sha256 = _sha256(inventory_bytes)
    if inventory_sha256 != expected_inventory_sha256:
        raise VerificationError(
            "source snapshot inventory SHA-256 mismatch: "
            f"{inventory_sha256} != {expected_inventory_sha256}"
        )
    inventory = _parse_json(inventory_bytes, "source snapshot inventory")
    inventory_records = _validate_inventory(inventory)

    actual_records, fast_tree_sha256, gr00t_tree_sha256 = _tree_records(root)
    if actual_records != inventory_records:
        raise VerificationError(
            "source snapshot inventory does not exactly match sealed source files"
        )

    record_by_path = {record["path"]: record for record in actual_records}
    manifest_path = root / ".source_snapshot_manifest.json"
    manifest_bytes, _ = _regular_file_bytes(
        manifest_path, "source snapshot internal manifest"
    )
    manifest_record = record_by_path.get(".source_snapshot_manifest.json")
    if (
        manifest_record is None
        or manifest_record["bytes"] != len(manifest_bytes)
        or manifest_record["sha256"] != _sha256(manifest_bytes)
    ):
        raise VerificationError("source snapshot internal manifest is not inventory-bound")
    manifest = _validate_manifest(
        _parse_json(manifest_bytes, "source snapshot internal manifest"), root.name
    )

    marker_paths = {
        ".source_revision": (manifest["fast_roborender_revision"], "fast_roborender"),
        "GR00T-VisualSim2Real/.source_revision": (
            manifest["gr00t_revision"],
            "GR00T",
        ),
    }
    for relative, (expected, label) in marker_paths.items():
        if relative not in record_by_path:
            raise VerificationError(f"{label} revision marker is missing from inventory")
        marker_bytes, _ = _regular_file_bytes(root / relative, f"{label} revision marker")
        if _sha256(marker_bytes) != record_by_path[relative]["sha256"]:
            raise VerificationError(f"{label} revision marker is not inventory-bound")
        _validate_revision_marker(marker_bytes, expected, label)

    attestation_bytes, attestation_stat = _regular_file_bytes(
        attestation_path, "source snapshot attestation"
    )
    if stat.S_IMODE(attestation_stat.st_mode) != 0o444:
        raise VerificationError("source snapshot attestation mode must be exactly 0o444")
    attestation = _parse_json(attestation_bytes, "source snapshot attestation")
    _validate_attestation(
        attestation,
        manifest,
        inventory_sha256,
        actual_records,
        fast_tree_sha256,
        gr00t_tree_sha256,
    )

    # Close ordinary rename/write races: all source bytes and both external
    # evidence documents must still be identical after semantic validation.
    final_records, final_fast_tree, final_gr00t_tree = _tree_records(root)
    if (
        final_records != actual_records
        or final_fast_tree != fast_tree_sha256
        or final_gr00t_tree != gr00t_tree_sha256
    ):
        raise VerificationError("sealed source changed during verification")
    final_inventory_bytes, _ = _regular_file_bytes(
        inventory_path, "source snapshot inventory"
    )
    final_attestation_bytes, _ = _regular_file_bytes(
        attestation_path, "source snapshot attestation"
    )
    if final_inventory_bytes != inventory_bytes or final_attestation_bytes != attestation_bytes:
        raise VerificationError("external source evidence changed during verification")

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "root": str(root),
        "snapshot_name": manifest["snapshot_name"],
        "fast_roborender_revision": manifest["fast_roborender_revision"],
        "gr00t_revision": manifest["gr00t_revision"],
        "file_count": inventory["file_count"],
        "total_file_bytes": inventory["total_file_bytes"],
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "attestation_path": str(attestation_path),
        "attestation_sha256": _sha256(attestation_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "fast_roborender_source_tree_sha256": fast_tree_sha256,
        "gr00t_source_tree_sha256": gr00t_tree_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = verify(arguments.root, arguments.expected_inventory_sha256)
    except VerificationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
