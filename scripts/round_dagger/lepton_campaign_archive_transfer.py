#!/usr/bin/env python3
"""Pack and publish inventory-bound DAgger trees through Lepton.

Direct SSH is not part of this transport.  A projection is first encoded as a
deterministic ``tar+gzip`` archive, then the existing authenticated Lepton Web
terminal streamer uploads the archive and its two small evidence files.  The
remote side validates the archive, extracts only regular files/directories,
reuses :mod:`remote_campaign_transfer` for byte-exact inventory verification,
and atomically renames the verified tree into place.  The reverse path packages
a learner output on H200 and raw-streams it back to an RTX result directory.

The three commands intentionally separate artifact creation, remote-local
extraction, and the live transport.  ``extract-publish`` is suitable for unit
and in-container validation without contacting Lepton.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import posixpath
import secrets
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round_dagger import lepton_web_terminal_transfer as webtransfer  # noqa: E402
from scripts.round_dagger import remote_campaign_transfer as tree_transfer  # noqa: E402


SCHEMA = "fast_roborender.round_dagger.inventory_tree_archive.v1"
REPORT_SCHEMA = "fast_roborender.round_dagger.inventory_tree_publish.v1"
ARCHIVE_FORMAT = "tar+gzip"
COMPRESSION_LEVEL = 1
GZIP_MTIME = 0
_SHA256 = frozenset("0123456789abcdef")
_STAGE_PREFIX = ".incoming-inventory-tree-"
TREE_KINDS = frozenset({"campaign_projection", "learner_output"})


class CampaignArchiveError(RuntimeError):
    """A bundle was unsafe, inconsistent, or could not be published."""


def _canonical_json(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    else:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
    return text.encode("utf-8")


def _sha256_handle(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _regular_identity(path: str | os.PathLike[str]) -> tuple[int, str]:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CampaignArchiveError(f"cannot open regular file: {source}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CampaignArchiveError(f"path is not a regular file: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return _sha256_handle(handle)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CampaignArchiveError("cannot open regular evidence file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CampaignArchiveError("evidence path is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _strict_parent(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    parent = absolute.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CampaignArchiveError("artifact parent must be an existing real directory")
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current /= component
        if current.is_symlink():
            raise CampaignArchiveError("artifact parent path contains a symlink")
    return absolute


def _publish_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> str:
    destination = _strict_parent(path)
    expected = hashlib.sha256(payload).hexdigest()
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise CampaignArchiveError("refusing an unsafe existing artifact")
        observed = _regular_identity(destination)
        if observed != (len(payload), expected):
            raise CampaignArchiveError("refusing to replace a different artifact")
        return "adopted"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".incoming", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if destination.is_symlink() or _regular_identity(destination) != (
                len(payload),
                expected,
            ):
                raise CampaignArchiveError("competing artifact publication differs")
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return "published"
    finally:
        temporary.unlink(missing_ok=True)


def _publish_temporary(path: Path, temporary: Path, expected: tuple[int, str]) -> str:
    destination = _strict_parent(path)
    if _regular_identity(temporary) != expected:
        raise CampaignArchiveError("temporary archive identity changed")
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise CampaignArchiveError("refusing an unsafe existing archive")
        if _regular_identity(destination) != expected:
            raise CampaignArchiveError("refusing to replace a different archive")
        return "adopted"
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError:
        if destination.is_symlink() or _regular_identity(destination) != expected:
            raise CampaignArchiveError("competing archive publication differs")
    directory = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return "published"


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CampaignArchiveError("bundle contains an invalid relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise CampaignArchiveError("bundle contains an unsafe relative path")
    return value


def _projection_directories(root: Path) -> list[str]:
    directories: list[str] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        del filenames
        directory_path = Path(directory)
        for name in sorted(dirnames):
            child = directory_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CampaignArchiveError("projection contains an unsafe directory")
            directories.append(child.relative_to(root).as_posix())
    return sorted(directories)


def _tar_info(name: str, *, directory: bool, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_deterministic_archive(
    root: Path,
    inventory: Mapping[str, Any],
    directories: Sequence[str],
    output: Path,
) -> tuple[int, str]:
    descriptor = os.open(
        output,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=COMPRESSION_LEVEL,
                fileobj=raw,
                mtime=GZIP_MTIME,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT
                ) as archive:
                    for relative in directories:
                        archive.addfile(_tar_info(relative, directory=True))
                    for record in inventory["files"]:
                        relative = record["path"]
                        source = root / relative
                        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                            os, "O_NOFOLLOW", 0
                        )
                        source_descriptor = os.open(source, flags)
                        try:
                            info = os.fstat(source_descriptor)
                            if not stat.S_ISREG(info.st_mode) or info.st_size != record["size"]:
                                raise CampaignArchiveError(
                                    "projection changed while its archive was created"
                                )
                            with os.fdopen(
                                source_descriptor, "rb", closefd=False
                            ) as handle:
                                archive.addfile(
                                    _tar_info(
                                        relative,
                                        directory=False,
                                        size=record["size"],
                                    ),
                                    handle,
                                )
                        finally:
                            os.close(source_descriptor)
            raw.flush()
            os.fsync(raw.fileno())
    finally:
        os.close(descriptor)
    return _regular_identity(output)


def _manifest_document(
    *,
    archive_identity: tuple[int, str],
    inventory_identity: tuple[int, str],
    inventory: Mapping[str, Any],
    directories: Sequence[str],
    tree_kind: str,
) -> dict[str, Any]:
    if tree_kind not in TREE_KINDS:
        raise CampaignArchiveError("tree kind is unsupported")
    return {
        "schema": SCHEMA,
        "tree_kind": tree_kind,
        "archive": {
            "format": ARCHIVE_FORMAT,
            "compression_level": COMPRESSION_LEVEL,
            "gzip_mtime": GZIP_MTIME,
            "size": archive_identity[0],
            "sha256": archive_identity[1],
        },
        "inventory": {
            "schema": tree_transfer.SCHEMA,
            "size": inventory_identity[0],
            "sha256": inventory_identity[1],
            "tree_sha256": inventory["tree_sha256"],
            "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
        },
        "members": {
            "directories": list(directories),
            "files": [record["path"] for record in inventory["files"]],
        },
    }


def _hex_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise CampaignArchiveError(f"{label} is not a lowercase SHA-256")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = _read_regular_bytes(path)
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignArchiveError("cannot read campaign archive manifest") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "schema",
        "tree_kind",
        "archive",
        "inventory",
        "members",
    }:
        raise CampaignArchiveError("campaign archive manifest key-set mismatch")
    result = dict(document)
    if result["schema"] != SCHEMA:
        raise CampaignArchiveError("campaign archive manifest schema mismatch")
    if result["tree_kind"] not in TREE_KINDS:
        raise CampaignArchiveError("archive tree kind is unsupported")
    archive = result["archive"]
    if not isinstance(archive, Mapping) or set(archive) != {
        "format",
        "compression_level",
        "gzip_mtime",
        "size",
        "sha256",
    }:
        raise CampaignArchiveError("archive identity key-set mismatch")
    if (
        archive["format"] != ARCHIVE_FORMAT
        or archive["compression_level"] != COMPRESSION_LEVEL
        or archive["gzip_mtime"] != GZIP_MTIME
        or type(archive["size"]) is not int
        or archive["size"] < 0
    ):
        raise CampaignArchiveError("archive encoding contract mismatch")
    _hex_digest(archive["sha256"], "archive digest")
    inventory = result["inventory"]
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "schema",
        "size",
        "sha256",
        "tree_sha256",
        "file_count",
        "total_bytes",
    }:
        raise CampaignArchiveError("inventory identity key-set mismatch")
    if (
        inventory["schema"] != tree_transfer.SCHEMA
        or type(inventory["size"]) is not int
        or inventory["size"] < 0
        or type(inventory["file_count"]) is not int
        or inventory["file_count"] < 0
        or type(inventory["total_bytes"]) is not int
        or inventory["total_bytes"] < 0
    ):
        raise CampaignArchiveError("inventory identity is malformed")
    _hex_digest(inventory["sha256"], "inventory file digest")
    _hex_digest(inventory["tree_sha256"], "inventory tree digest")
    members = result["members"]
    if not isinstance(members, Mapping) or set(members) != {"directories", "files"}:
        raise CampaignArchiveError("archive member key-set mismatch")
    for key in ("directories", "files"):
        values = members[key]
        if not isinstance(values, list):
            raise CampaignArchiveError("archive member list is malformed")
        safe = [_safe_relative(value) for value in values]
        if safe != sorted(safe) or len(safe) != len(set(safe)):
            raise CampaignArchiveError("archive members are not unique and sorted")
    if set(members["directories"]) & set(members["files"]):
        raise CampaignArchiveError("archive directory/file members overlap")
    return result


def _validate_bundle(
    archive: Path,
    inventory_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_manifest_sha256 = _hex_digest(
        expected_manifest_sha256, "caller-pinned manifest digest"
    )
    manifest_identity = _regular_identity(manifest_path)
    if manifest_identity[1] != expected_manifest_sha256:
        raise CampaignArchiveError("manifest differs from the caller-pinned digest")
    manifest = _load_manifest(manifest_path)
    if _regular_identity(archive) != (
        manifest["archive"]["size"],
        manifest["archive"]["sha256"],
    ):
        raise CampaignArchiveError("archive differs from its manifest identity")
    if _regular_identity(inventory_path) != (
        manifest["inventory"]["size"],
        manifest["inventory"]["sha256"],
    ):
        raise CampaignArchiveError("inventory file differs from its manifest identity")
    try:
        inventory = tree_transfer.load_inventory(inventory_path)
    except tree_transfer.TransferInventoryError as exc:
        raise CampaignArchiveError("campaign inventory is invalid") from exc
    expected_inventory = manifest["inventory"]
    if (
        inventory["tree_sha256"] != expected_inventory["tree_sha256"]
        or inventory["file_count"] != expected_inventory["file_count"]
        or inventory["total_bytes"] != expected_inventory["total_bytes"]
        or [record["path"] for record in inventory["files"]]
        != manifest["members"]["files"]
    ):
        raise CampaignArchiveError("manifest and campaign inventory disagree")
    directories = set(manifest["members"]["directories"])
    for relative in manifest["members"]["directories"]:
        parent = PurePosixPath(relative).parent
        if parent != PurePosixPath(".") and parent.as_posix() not in directories:
            raise CampaignArchiveError("manifest omits a directory parent")
    for relative in manifest["members"]["files"]:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            if parent.as_posix() not in directories:
                raise CampaignArchiveError("manifest omits a file parent directory")
            parent = parent.parent
    if manifest["tree_kind"] == "campaign_projection":
        for required in ("policies", "replay", "rounds"):
            if required not in directories:
                raise CampaignArchiveError("manifest omits a required projection directory")
    # Detect ordinary mutation across the multi-file validation window.  The
    # remote copies are mode-0444, but publication still fails closed if an
    # owner changes one between the first identity check and parsing.
    if _regular_identity(manifest_path)[1] != expected_manifest_sha256:
        raise CampaignArchiveError("manifest changed while it was validated")
    if _regular_identity(archive) != (
        manifest["archive"]["size"],
        manifest["archive"]["sha256"],
    ) or _regular_identity(inventory_path) != (
        manifest["inventory"]["size"],
        manifest["inventory"]["sha256"],
    ):
        raise CampaignArchiveError("bundle changed while it was validated")
    return manifest, inventory


def pack_projection(
    projection: Path,
    archive: Path,
    inventory_path: Path,
    manifest_path: Path,
    *,
    tree_kind: str = "campaign_projection",
) -> dict[str, Any]:
    if tree_kind not in TREE_KINDS:
        raise CampaignArchiveError("tree kind is unsupported")
    root = Path(os.path.abspath(projection.expanduser()))
    try:
        if tree_kind == "campaign_projection":
            tree_transfer.validate_campaign_projection(root)
        else:
            tree_transfer.validate_tree_structure(root)
        inventory = tree_transfer.build_inventory(root)
    except tree_transfer.TransferInventoryError as exc:
        raise CampaignArchiveError("campaign projection validation failed") from exc
    outputs = [_strict_parent(path) for path in (archive, inventory_path, manifest_path)]
    if len(set(outputs)) != 3:
        raise CampaignArchiveError("archive, inventory, and manifest paths must differ")
    for output in outputs:
        if output == root or root in output.parents:
            raise CampaignArchiveError("bundle artifacts must be outside the projection")

    directories = _projection_directories(root)
    inventory_payload = _canonical_json(inventory, pretty=True)
    inventory_status = _publish_bytes(inventory_path, inventory_payload)
    inventory_identity = _regular_identity(inventory_path)

    temporary = archive.parent / f".{archive.name}.{secrets.token_hex(16)}.incoming"
    try:
        archive_identity = _write_deterministic_archive(
            root, inventory, directories, temporary
        )
        try:
            observed = tree_transfer.build_inventory(root)
        except tree_transfer.TransferInventoryError as exc:
            raise CampaignArchiveError("projection changed during archive creation") from exc
        if observed != inventory:
            raise CampaignArchiveError("projection changed during archive creation")
        if _projection_directories(root) != directories:
            raise CampaignArchiveError("projection directories changed during archive creation")
        try:
            if tree_kind == "campaign_projection":
                tree_transfer.validate_campaign_projection(root)
            else:
                tree_transfer.validate_tree_structure(root)
        except tree_transfer.TransferInventoryError as exc:
            raise CampaignArchiveError("projection changed during archive creation") from exc
        archive_status = _publish_temporary(archive, temporary, archive_identity)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = _manifest_document(
        archive_identity=archive_identity,
        inventory_identity=inventory_identity,
        inventory=inventory,
        directories=directories,
        tree_kind=tree_kind,
    )
    manifest_status = _publish_bytes(
        manifest_path, _canonical_json(manifest, pretty=True)
    )
    manifest_identity = _regular_identity(manifest_path)
    _validate_bundle(
        archive,
        inventory_path,
        manifest_path,
        expected_manifest_sha256=manifest_identity[1],
    )
    return {
        "schema": REPORT_SCHEMA,
        "operation": "pack",
        "status": "PASS",
        "tree_kind": tree_kind,
        "archive": {
            "path": str(archive),
            "status": archive_status,
            "size": archive_identity[0],
            "sha256": archive_identity[1],
        },
        "inventory": {
            "path": str(inventory_path),
            "status": inventory_status,
            "size": inventory_identity[0],
            "sha256": inventory_identity[1],
            "tree_sha256": inventory["tree_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "status": manifest_status,
            "size": manifest_identity[0],
            "sha256": manifest_identity[1],
        },
    }


def _validate_tar_members(
    archive: tarfile.TarFile,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> tuple[list[tarfile.TarInfo], list[tarfile.TarInfo]]:
    members = archive.getmembers()
    expected_directories = manifest["members"]["directories"]
    expected_files = manifest["members"]["files"]
    if [member.name for member in members] != expected_directories + expected_files:
        raise CampaignArchiveError("archive member sequence differs from the manifest")
    directories = members[: len(expected_directories)]
    files = members[len(expected_directories) :]
    for member in directories:
        _safe_relative(member.name)
        if (
            not member.isdir()
            or member.size != 0
            or member.mode != 0o755
            or member.uid != 0
            or member.gid != 0
            or member.mtime != 0
            or member.uname
            or member.gname
        ):
            raise CampaignArchiveError("archive directory metadata is unsafe")
    expected_records = {record["path"]: record for record in inventory["files"]}
    for member in files:
        _safe_relative(member.name)
        record = expected_records[member.name]
        if (
            not member.isreg()
            or member.size != record["size"]
            or member.mode != 0o644
            or member.uid != 0
            or member.gid != 0
            or member.mtime != 0
            or member.uname
            or member.gname
        ):
            raise CampaignArchiveError("archive file metadata is unsafe")
    return directories, files


def _safe_stage(destination: Path, manifest_sha256: str) -> Path:
    destination = Path(os.path.abspath(destination.expanduser()))
    parent = _strict_parent(destination).parent
    destination_id = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:16]
    stage = parent / f"{_STAGE_PREFIX}{destination_id}-{manifest_sha256[:24]}"
    if stage.parent != parent or not stage.name.startswith(_STAGE_PREFIX):
        raise CampaignArchiveError("derived extraction stage is unsafe")
    if destination == stage or destination in stage.parents or stage in destination.parents:
        raise CampaignArchiveError("campaign stage and destination overlap")
    return stage


def _remove_private_stage(stage: Path) -> None:
    if not stage.name.startswith(_STAGE_PREFIX) or stage.parent.is_symlink():
        raise CampaignArchiveError("refusing to remove a non-private stage")
    if os.path.lexists(stage):
        if stage.is_symlink() or not stage.is_dir():
            raise CampaignArchiveError("private stage path is unsafe")
        shutil.rmtree(stage)


def _extract_archive(
    archive_path: Path,
    stage: Path,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            directories, files = _validate_tar_members(archive, manifest, inventory)
            stage.mkdir(mode=0o700)
            for member in directories:
                target = stage / member.name
                target.mkdir(mode=0o755)
                os.chmod(target, 0o755)
            records = {record["path"]: record for record in inventory["files"]}
            for member in files:
                target = stage / member.name
                source = archive.extractfile(member)
                if source is None:
                    raise CampaignArchiveError("archive regular file has no payload")
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(target, flags, 0o600)
                digest = hashlib.sha256()
                written = 0
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as output:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            output.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    record = records[member.name]
                    if written != record["size"] or digest.hexdigest() != record["sha256"]:
                        raise CampaignArchiveError("extracted file differs from its inventory")
                    # Returned learner artifacts are immutable evidence.  Make
                    # the inode read-only before the extraction tree can be
                    # atomically published at its final path.
                    os.fchmod(descriptor, 0o444)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            # Persist all directory entries before the stage is renamed into
            # the canonical destination.  File fsync alone does not make the
            # containing directory entries crash durable.
            for directory in sorted(
                (Path(root) for root, _dirs, _files in os.walk(stage)),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                directory_fd = os.open(
                    directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            os.chmod(stage, 0o755)
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise CampaignArchiveError("cannot safely extract campaign archive") from exc


def extract_publish(
    archive: Path,
    inventory_path: Path,
    manifest_path: Path,
    destination: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest, inventory = _validate_bundle(
        archive,
        inventory_path,
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    destination = Path(os.path.abspath(destination.expanduser()))
    stage = _safe_stage(destination, expected_manifest_sha256)
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise CampaignArchiveError("campaign destination is unsafe")
        try:
            observed = tree_transfer.verify_inventory(destination, inventory_path)
            if manifest["tree_kind"] == "campaign_projection":
                tree_transfer.validate_campaign_projection(destination)
            else:
                tree_transfer.validate_tree_structure(destination)
        except tree_transfer.TransferInventoryError as exc:
            raise CampaignArchiveError("existing campaign destination conflicts") from exc
        _remove_private_stage(stage)
        status = "adopted"
    else:
        _remove_private_stage(stage)
        try:
            _extract_archive(archive, stage, manifest, inventory)
            if manifest["tree_kind"] == "campaign_projection":
                tree_transfer.validate_campaign_projection(stage)
            else:
                tree_transfer.validate_tree_structure(stage)
            tree_transfer.verify_inventory(stage, inventory_path)
            observed = tree_transfer.publish_tree(stage, destination, inventory_path)
            if manifest["tree_kind"] == "campaign_projection":
                tree_transfer.validate_campaign_projection(destination)
            else:
                tree_transfer.validate_tree_structure(destination)
        except (CampaignArchiveError, tree_transfer.TransferInventoryError):
            _remove_private_stage(stage)
            raise
        status = "published"
    return {
        "schema": REPORT_SCHEMA,
        "operation": "extract-publish",
        "status": "PASS",
        "publication": status,
        "tree_kind": manifest["tree_kind"],
        "destination": str(destination),
        "manifest_sha256": expected_manifest_sha256,
        "archive_sha256": manifest["archive"]["sha256"],
        "inventory_sha256": manifest["inventory"]["sha256"],
        "tree_sha256": observed["tree_sha256"],
        "file_count": observed["file_count"],
        "total_bytes": observed["total_bytes"],
    }


def _remote_bundle_paths(bundle_root: str, manifest_sha256: str) -> dict[str, str]:
    manifest_sha256 = _hex_digest(manifest_sha256, "caller-pinned manifest digest")
    probe = webtransfer.validate_remote_path(posixpath.join(bundle_root, "probe"))
    root = posixpath.dirname(probe)
    token = manifest_sha256
    return {
        "archive": webtransfer.validate_remote_path(
            posixpath.join(root, f"campaign-projection-{token}.tar.gz")
        ),
        "inventory": webtransfer.validate_remote_path(
            posixpath.join(root, f"campaign-projection-{token}.inventory.json")
        ),
        "manifest": webtransfer.validate_remote_path(
            posixpath.join(root, f"campaign-projection-{token}.manifest.json")
        ),
    }


def upload_publish(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest_sha256 = _hex_digest(
        arguments.manifest_sha256, "caller-pinned manifest digest"
    )
    manifest, _ = _validate_bundle(
        Path(arguments.archive),
        Path(arguments.inventory),
        Path(arguments.manifest),
        expected_manifest_sha256=manifest_sha256,
    )
    if manifest["tree_kind"] != "campaign_projection":
        raise CampaignArchiveError("forward handoff is not a campaign projection")
    remote_paths = _remote_bundle_paths(arguments.remote_bundle_root, manifest_sha256)
    remote_destination = webtransfer.validate_remote_path(arguments.remote_destination)
    remote_python = webtransfer.validate_remote_path(arguments.remote_python)
    remote_tool = webtransfer.validate_remote_path(arguments.remote_tool)
    transport: webtransfer.NodeWebTerminalTransport | None = None
    try:
        connection = webtransfer.discover_connection(
            arguments.deployment,
            arguments.replica,
            sdk_python=getattr(arguments, "sdk_python", None),
        )
        transport = webtransfer.NodeWebTerminalTransport(
            connection,
            heartbeat_seconds=arguments.heartbeat_seconds,
            heartbeat_timeout_seconds=arguments.heartbeat_timeout_seconds,
        )
        shell = webtransfer.RemoteShell(
            transport, command_timeout_seconds=arguments.command_timeout_seconds
        )
        engine = webtransfer.TransferEngine(
            webtransfer.ShellRemoteFileStore(shell), chunk_bytes=arguments.chunk_bytes
        )
        streamer = webtransfer.StreamingUploader(
            transport,
            chunk_bytes=arguments.chunk_bytes,
            completion_timeout_seconds=arguments.command_timeout_seconds,
        )
        uploaded: dict[str, Any] = {}
        # The manifest is the commit record and is deliberately uploaded last.
        for label, local in (
            ("archive", arguments.archive),
            ("inventory", arguments.inventory),
            ("manifest", arguments.manifest),
        ):
            uploaded[label] = engine.upload_streaming(
                local,
                remote_paths[label],
                streamer,
                adopt_existing=True,
                mode=0o444,
            )
        command = shlex.join(
            [
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                remote_python,
                remote_tool,
                "extract-publish",
                "--archive",
                remote_paths["archive"],
                "--inventory",
                remote_paths["inventory"],
                "--manifest",
                remote_paths["manifest"],
                "--manifest-sha256",
                manifest_sha256,
                "--destination",
                remote_destination,
            ]
        )
        result = shell.run(
            f"__lxft_result=$({command}); __lxft_emit \"$__lxft_result\"",
            timeout_seconds=arguments.extract_timeout_seconds,
        )
        if len(result.data) != 1:
            raise CampaignArchiveError("remote publication returned an invalid record count")
        try:
            remote_report = json.loads(result.data[0].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignArchiveError("remote publication report is malformed") from exc
        if (
            not isinstance(remote_report, Mapping)
            or remote_report.get("schema") != REPORT_SCHEMA
            or remote_report.get("status") != "PASS"
            or remote_report.get("tree_kind") != "campaign_projection"
            or remote_report.get("manifest_sha256") != manifest_sha256
            or remote_report.get("archive_sha256") != manifest["archive"]["sha256"]
            or remote_report.get("inventory_sha256")
            != manifest["inventory"]["sha256"]
            or remote_report.get("tree_sha256")
            != manifest["inventory"]["tree_sha256"]
        ):
            raise CampaignArchiveError("remote publication report identity mismatch")
        return {
            "schema": REPORT_SCHEMA,
            "operation": "upload-publish",
            "status": "PASS",
            "manifest_sha256": manifest_sha256,
            "remote_artifacts": remote_paths,
            "uploads": uploaded,
            "publication": dict(remote_report),
        }
    except webtransfer.TransferError as exc:
        raise CampaignArchiveError("Lepton Web-terminal transfer failed") from exc
    finally:
        if transport is not None:
            transport.close()


def _local_bundle_paths(bundle_root: Path, manifest_sha256: str) -> dict[str, Path]:
    manifest_sha256 = _hex_digest(manifest_sha256, "caller-pinned manifest digest")
    root = _strict_parent(bundle_root / "probe").parent
    token = manifest_sha256
    return {
        "archive": root / f"inventory-tree-{token}.tar.gz",
        "inventory": root / f"inventory-tree-{token}.inventory.json",
        "manifest": root / f"inventory-tree-{token}.manifest.json",
    }


def download_publish(arguments: argparse.Namespace) -> dict[str, Any]:
    """Stream a pinned remote result bundle back and atomically publish it."""

    manifest_sha256 = _hex_digest(
        arguments.manifest_sha256, "caller-pinned manifest digest"
    )
    remote_paths = {
        "archive": webtransfer.validate_remote_path(arguments.remote_archive),
        "inventory": webtransfer.validate_remote_path(arguments.remote_inventory),
        "manifest": webtransfer.validate_remote_path(arguments.remote_manifest),
    }
    local_paths = _local_bundle_paths(Path(arguments.local_bundle_root), manifest_sha256)
    destination = _strict_parent(Path(arguments.destination))
    for path in local_paths.values():
        if path == destination or destination in path.parents or path in destination.parents:
            raise CampaignArchiveError("local bundle and result destination overlap")
    transport: webtransfer.NodeWebTerminalTransport | None = None
    try:
        connection = webtransfer.discover_connection(
            arguments.deployment,
            arguments.replica,
            sdk_python=getattr(arguments, "sdk_python", None),
        )
        transport = webtransfer.NodeWebTerminalTransport(
            connection,
            heartbeat_seconds=arguments.heartbeat_seconds,
            heartbeat_timeout_seconds=arguments.heartbeat_timeout_seconds,
        )
        shell = webtransfer.RemoteShell(
            transport, command_timeout_seconds=arguments.command_timeout_seconds
        )
        engine = webtransfer.TransferEngine(
            webtransfer.ShellRemoteFileStore(shell), chunk_bytes=arguments.chunk_bytes
        )
        downloader = webtransfer.StreamingDownloader(
            transport,
            completion_timeout_seconds=arguments.command_timeout_seconds,
        )
        downloaded: dict[str, Any] = {}
        # Fetch and authenticate the small commit record before transferring
        # the potentially large archive.
        downloaded["manifest"] = engine.download_streaming(
            remote_paths["manifest"],
            local_paths["manifest"],
            downloader,
            adopt_existing=True,
            mode=0o444,
        )
        if _regular_identity(local_paths["manifest"])[1] != manifest_sha256:
            raise CampaignArchiveError("downloaded manifest differs from the caller pin")
        for label in ("inventory", "archive"):
            downloaded[label] = engine.download_streaming(
                remote_paths[label],
                local_paths[label],
                downloader,
                adopt_existing=True,
                mode=0o444,
            )
        manifest, _ = _validate_bundle(
            local_paths["archive"],
            local_paths["inventory"],
            local_paths["manifest"],
            expected_manifest_sha256=manifest_sha256,
        )
        if manifest["tree_kind"] != "learner_output":
            raise CampaignArchiveError("result return bundle is not a learner output")
        publication = extract_publish(
            local_paths["archive"],
            local_paths["inventory"],
            local_paths["manifest"],
            destination,
            expected_manifest_sha256=manifest_sha256,
        )
        return {
            "schema": REPORT_SCHEMA,
            "operation": "download-publish",
            "status": "PASS",
            "manifest_sha256": manifest_sha256,
            "remote_artifacts": remote_paths,
            "local_artifacts": {key: str(value) for key, value in local_paths.items()},
            "downloads": downloaded,
            "publication": publication,
        }
    except webtransfer.TransferError as exc:
        raise CampaignArchiveError("Lepton Web-terminal result transfer failed") from exc
    finally:
        if transport is not None:
            transport.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--projection", required=True)
    pack.add_argument("--archive", required=True)
    pack.add_argument("--inventory", required=True)
    pack.add_argument("--manifest", required=True)
    pack.add_argument(
        "--tree-kind",
        choices=sorted(TREE_KINDS),
        default="campaign_projection",
    )

    extract = subparsers.add_parser("extract-publish")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--inventory", required=True)
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--manifest-sha256", required=True)
    extract.add_argument("--destination", required=True)

    upload = subparsers.add_parser("upload-publish")
    upload.add_argument("--archive", required=True)
    upload.add_argument("--inventory", required=True)
    upload.add_argument("--manifest", required=True)
    upload.add_argument("--manifest-sha256", required=True)
    upload.add_argument("--remote-bundle-root", required=True)
    upload.add_argument("--remote-destination", required=True)
    upload.add_argument("--remote-python", required=True)
    upload.add_argument("--remote-tool", required=True)
    upload.add_argument("--deployment", default=webtransfer.DEFAULT_DEPLOYMENT)
    upload.add_argument("--replica")
    upload.add_argument("--sdk-python")
    upload.add_argument("--chunk-bytes", type=int, default=128 * 1024)
    upload.add_argument("--command-timeout-seconds", type=float, default=300.0)
    upload.add_argument("--extract-timeout-seconds", type=float, default=900.0)
    upload.add_argument("--heartbeat-seconds", type=float, default=15.0)
    upload.add_argument("--heartbeat-timeout-seconds", type=float, default=45.0)

    download = subparsers.add_parser("download-publish")
    download.add_argument("--remote-archive", required=True)
    download.add_argument("--remote-inventory", required=True)
    download.add_argument("--remote-manifest", required=True)
    download.add_argument("--manifest-sha256", required=True)
    download.add_argument("--local-bundle-root", required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--deployment", default=webtransfer.DEFAULT_DEPLOYMENT)
    download.add_argument("--replica")
    download.add_argument("--sdk-python")
    download.add_argument("--chunk-bytes", type=int, default=128 * 1024)
    download.add_argument("--command-timeout-seconds", type=float, default=900.0)
    download.add_argument("--heartbeat-seconds", type=float, default=15.0)
    download.add_argument("--heartbeat-timeout-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "pack":
            report = pack_projection(
                Path(arguments.projection),
                Path(arguments.archive),
                Path(arguments.inventory),
                Path(arguments.manifest),
                tree_kind=arguments.tree_kind,
            )
        elif arguments.command == "extract-publish":
            report = extract_publish(
                Path(arguments.archive),
                Path(arguments.inventory),
                Path(arguments.manifest),
                Path(arguments.destination),
                expected_manifest_sha256=arguments.manifest_sha256,
            )
        elif arguments.command == "upload-publish":
            report = upload_publish(arguments)
        else:
            report = download_publish(arguments)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except CampaignArchiveError as exc:
        print(f"campaign archive transfer failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
