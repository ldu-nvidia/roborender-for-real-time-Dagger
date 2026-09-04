#!/usr/bin/env python3
"""Run one production Lepton learner handoff as an identity-bound transaction.

The RTX coordinator owns this boundary.  It creates a learner-only immutable
projection, publishes it to the reserved H200 node, materializes and publishes
the exact Lepton request, invokes the journaled exclusive-node lifecycle, and
then atomically returns the inventory-bound learner output.  Every resumable
publication is create-only: an existing object is accepted only when its bytes
match the caller-pinned identity.

Credentials are never read or projected here.  Lepton authentication remains
inside the explicitly pinned control Python and the W&B credential remains a
Lepton Secret injected only into the Job.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger import (  # noqa: E402
    artifacts,
    learner_acceptance,
    orchestration,
    replay,
    state,
)
from round_dagger.lepton_job import (  # noqa: E402
    LeptonBackendContract,
    LeptonJobError,
    LeptonRunRequest,
    load_contract,
    load_request,
    materialize_request,
    require_finalized_supply_chain,
    validate_completion_marker,
)
from scripts.round_dagger import lepton_campaign_archive_transfer as archive_transfer  # noqa: E402
from scripts.round_dagger import remote_campaign_transfer as tree_transfer  # noqa: E402


SUBMITTER = ROOT / "scripts/round_dagger/submit_lepton_round_learner_and_wait.py"
ARCHIVE_TRANSFER = (
    ROOT / "scripts/round_dagger/lepton_campaign_archive_transfer.py"
)
WEB_TRANSFER = ROOT / "scripts/round_dagger/lepton_web_terminal_transfer.py"
TRANSFER_WRAPPER = ROOT / "scripts/round_dagger/run_lepton_web_terminal_transfer.sh"
PINNED_SDK_PYTHON = Path(
    "/mnt/data2/ldu/roborender_lepton_control/env/bin/python"
)
PINNED_SDK_FREEZE = Path(
    "/mnt/data2/ldu/roborender_lepton_control/control.freeze.txt"
)
PINNED_SDK_FREEZE_SHA256 = (
    "aa3cadd8d69359f84997ffc76f71bcfeadbdd37f2aac249fb6a2ee9e04c58104"
)
BRIDGE_REPORT_SCHEMA = "fast_roborender.round_dagger.production_lepton_bridge.v1"
RETIREMENT_INTENT_SCHEMA = (
    "fast_roborender.round_dagger.local_handoff_retirement_intent.v1"
)
RETIREMENT_COMPLETE_SCHEMA = (
    "fast_roborender.round_dagger.local_handoff_retirement_complete.v1"
)
_RETIREMENT_INTENT_FIELDS = {
    "schema",
    "status",
    "campaign_id",
    "macro_round",
    "learner_identity_sha256",
    "request_sha256",
    "result_sha256",
    "candidate_checkpoint_sha256",
    "completion_marker_sha256",
    "campaign_manifest_sha256",
    "projection_tree_sha256",
    "projection_total_bytes",
    "archive_sha256",
    "archive_size_bytes",
    "retirement_targets",
    "content_sha256",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_PROJECTION_STAGE_PREFIX = ".incoming-campaign-"
_PROJECTION_STAGE_RE = re.compile(r"\.incoming-campaign-[a-z0-9_]{8}\Z")
_DOWNLOAD_STAGE_PREFIX = ".incoming."
_DOWNLOAD_STAGE_RE = re.compile(r"\.incoming\.[0-9a-f]{32}\Z")
_TREE_STAGE_PREFIX = ".incoming-inventory-tree-"
_TREE_STAGE_RE = re.compile(
    r"\.incoming-inventory-tree-[0-9a-f]{16}-[0-9a-f]{24}\Z"
)
_EXCLUDED_TOP_LEVEL = frozenset({"actors", "operator", ".coordinator_state.lock"})
_REMOTE_SUCCESS_MARKER_BASENAME = "lepton_learner_success.json"
_CAMPAIGN_INPUTS = (
    "ROUND_DAGGER_POLICY_CHECKPOINT",
    "ROUND_DAGGER_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_BASE_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_MANIFEST",
    "ROUND_DAGGER_LEARNER_CONTRACT",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_SCENE_SAMPLE_MANIFEST",
)


class ProductionLeptonBridgeError(RuntimeError):
    """The production Lepton handoff was unsafe or incomplete."""


@dataclass(frozen=True)
class BridgePaths:
    local_handoff: Path
    local_projection: Path
    local_archive: Path
    local_inventory: Path
    local_manifest: Path
    local_request: Path
    local_lifecycle: Path
    local_marker: Path
    local_checksum: Path
    local_return_bundle: Path
    local_output: Path
    local_result: Path
    remote_bundle_root: PurePosixPath
    remote_campaign: PurePosixPath
    remote_output: PurePosixPath
    remote_request: PurePosixPath


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ProductionLeptonBridgeError(f"missing or unsafe environment: {name}")
    return value


def _sha_environment(name: str) -> str:
    value = _environment(name)
    if _SHA256.fullmatch(value) is None:
        raise ProductionLeptonBridgeError(f"{name} must be lowercase SHA-256")
    return value


def _token_environment(name: str) -> str:
    value = _environment(name)
    if _TOKEN.fullmatch(value) is None:
        raise ProductionLeptonBridgeError(f"{name} is malformed")
    return value


def _absolute_environment(name: str, *, require_regular: bool = False) -> Path:
    raw = _environment(name)
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw or os.path.abspath(raw) != raw:
        raise ProductionLeptonBridgeError(f"{name} must be canonical and absolute")
    if require_regular:
        _regular(path, name)
    return path


def _regular(path: Path, label: str, *, immutable: bool = False) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ProductionLeptonBridgeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProductionLeptonBridgeError(f"{label} must be a regular file")
    if immutable and info.st_mode & 0o222:
        raise ProductionLeptonBridgeError(f"{label} must be read-only")
    return info


def _real_directory(path: Path, label: str, *, create: bool = False) -> Path:
    path = Path(os.path.abspath(path))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ProductionLeptonBridgeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ProductionLeptonBridgeError(f"{label} must be a real directory")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ProductionLeptonBridgeError(f"{label} path contains a symlink")
    return path


def _sha256_file(path: Path) -> str:
    _regular(path, str(path))
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _verify_transferred_completion_checksum(paths: BridgePaths) -> str:
    """Verify an exact remote checksum after the marker is locally renamed.

    The remote sidecar binds the source basename
    ``lepton_learner_success.json``.  The RTX audit copy intentionally uses
    ``remote_learner_success.json``; preserving the transferred sidecar bytes
    means it must be checked against the source basename, not recomputed from
    the local audit filename.
    """

    _regular(paths.local_marker, "transferred completion marker", immutable=True)
    checksum_before = _regular(
        paths.local_checksum,
        "transferred completion checksum",
        immutable=True,
    )
    if paths.local_checksum != Path(f"{paths.local_marker}.sha256"):
        raise ProductionLeptonBridgeError(
            "transferred completion checksum path is non-canonical"
        )
    payload = paths.local_checksum.read_bytes()
    checksum_after = _regular(
        paths.local_checksum,
        "transferred completion checksum",
        immutable=True,
    )
    identity_before = (
        checksum_before.st_dev,
        checksum_before.st_ino,
        checksum_before.st_mode,
        checksum_before.st_size,
        checksum_before.st_mtime_ns,
        checksum_before.st_ctime_ns,
    )
    identity_after = (
        checksum_after.st_dev,
        checksum_after.st_ino,
        checksum_after.st_mode,
        checksum_after.st_size,
        checksum_after.st_mtime_ns,
        checksum_after.st_ctime_ns,
    )
    marker_sha256 = _sha256_file(paths.local_marker)
    expected = (
        f"{marker_sha256}  {_REMOTE_SUCCESS_MARKER_BASENAME}\n".encode("ascii")
    )
    if identity_before != identity_after or payload != expected:
        raise ProductionLeptonBridgeError(
            "transferred completion checksum does not bind the remote marker"
        )
    return marker_sha256


def _run(
    command: Sequence[str],
    label: str,
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionLeptonBridgeError(f"{label} could not execute") from exc
    if completed.returncode != 0:
        raise ProductionLeptonBridgeError(
            f"{label} exited {completed.returncode}"
        )
    return completed


def _run_json(command: Sequence[str], label: str) -> dict[str, Any]:
    completed = _run(command, label, capture_output=True)
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProductionLeptonBridgeError(f"{label} returned malformed evidence") from exc
    if not isinstance(value, Mapping):
        raise ProductionLeptonBridgeError(f"{label} returned non-object evidence")
    return dict(value)


def _verify_control_python(sdk_python: Path) -> None:
    if sdk_python != PINNED_SDK_PYTHON:
        raise ProductionLeptonBridgeError("Lepton SDK Python path differs from pin")
    try:
        resolved = sdk_python.resolve(strict=True)
    except OSError as exc:
        raise ProductionLeptonBridgeError("Lepton SDK Python cannot be resolved") from exc
    if not resolved.is_file() or not os.access(sdk_python, os.X_OK):
        raise ProductionLeptonBridgeError("Lepton SDK Python is not executable")
    _run(
        [
            str(sdk_python),
            "-c",
            (
                "import leptonai,numpy;"
                "assert getattr(leptonai,'__version__',None)=='0.27.3';"
                "assert numpy.__version__=='2.2.6'"
            ),
        ],
        "pinned Lepton SDK import probe",
    )
    freeze_info = _regular(
        PINNED_SDK_FREEZE, "pinned Lepton control environment freeze", immutable=True
    )
    del freeze_info
    freeze_bytes = PINNED_SDK_FREEZE.read_bytes()
    if _sha256_file(PINNED_SDK_FREEZE) != PINNED_SDK_FREEZE_SHA256:
        raise ProductionLeptonBridgeError(
            "pinned Lepton control environment freeze identity mismatch"
        )
    completed = _run(
        [str(sdk_python), "-m", "pip", "freeze", "--all"],
        "pinned Lepton control environment freeze probe",
        capture_output=True,
    )
    if completed.stdout.encode("utf-8").replace(b"\r\n", b"\n") != freeze_bytes:
        raise ProductionLeptonBridgeError(
            "Lepton control Python differs from its pinned freeze"
        )


def _excluded(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if not parts:
        return False
    for part in parts:
        if not part.startswith(_TREE_STAGE_PREFIX):
            continue
        if _TREE_STAGE_RE.fullmatch(part) is None:
            raise ProductionLeptonBridgeError(
                "campaign contains a malformed private extraction stage"
            )
        # A SIGKILL can strand the archive extractor's private tree beside the
        # canonical learner output.  It is never learner input and must not be
        # copied into the next immutable campaign projection.
        return True
    if parts[0] in _EXCLUDED_TOP_LEVEL:
        return True
    return len(parts) >= 3 and parts[0] == "rounds" and parts[2] == "orchestration_run"


def _copy_stable_regular(source: Path, destination: Path) -> None:
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, source_flags)
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProductionLeptonBridgeError(
                f"campaign projection source is not regular: {source}"
            )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            digest = hashlib.sha256()
            written = 0
            with os.fdopen(source_descriptor, "rb", closefd=False) as source_handle:
                with os.fdopen(
                    destination_descriptor, "wb", closefd=False
                ) as destination_handle:
                    for chunk in iter(
                        lambda: source_handle.read(8 * 1024 * 1024), b""
                    ):
                        destination_handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
            after = os.fstat(source_descriptor)
            identity = lambda info: (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            if identity(before) != identity(after) or written != before.st_size:
                raise ProductionLeptonBridgeError(
                    f"campaign input changed while copied: {source}"
                )
            os.chmod(destination, 0o444)
            if _sha256_file(destination) != digest.hexdigest():
                raise ProductionLeptonBridgeError(
                    f"campaign projection copy verification failed: {source}"
                )
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _inspect_directory_stage(
    stage: Path,
    parent: Path,
    *,
    pattern: re.Pattern[str],
    label: str,
) -> dict[str, Any]:
    """Inventory one exact private stage without following any links."""

    if stage.parent != parent or pattern.fullmatch(stage.name) is None:
        raise ProductionLeptonBridgeError("refusing to inspect a non-private stage")
    try:
        root_info = os.lstat(stage)
    except OSError as exc:
        raise ProductionLeptonBridgeError(
            f"{label} disappeared during inspection"
        ) from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ProductionLeptonBridgeError(f"{label} is unsafe")

    directories: list[tuple[Path, os.stat_result]] = []
    files: list[tuple[Path, os.stat_result]] = []

    def walk_error(error: OSError) -> None:
        raise ProductionLeptonBridgeError(f"cannot inspect {label}") from error

    for directory, dirnames, filenames in os.walk(
        stage, topdown=True, followlinks=False, onerror=walk_error
    ):
        base = Path(directory)
        try:
            base_info = os.lstat(base)
        except OSError as exc:
            raise ProductionLeptonBridgeError(
                f"{label} changed during inspection"
            ) from exc
        if stat.S_ISLNK(base_info.st_mode) or not stat.S_ISDIR(base_info.st_mode):
            raise ProductionLeptonBridgeError(
                f"{label} contains an unsafe directory"
            )
        directories.append((base, base_info))
        for name in sorted(dirnames):
            child = base / name
            try:
                child_info = os.lstat(child)
            except OSError as exc:
                raise ProductionLeptonBridgeError(
                    f"{label} changed during inspection"
                ) from exc
            if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISDIR(
                child_info.st_mode
            ):
                raise ProductionLeptonBridgeError(
                    f"{label} contains a symlink or special entry"
                )
        for name in sorted(filenames):
            child = base / name
            try:
                child_info = os.lstat(child)
            except OSError as exc:
                raise ProductionLeptonBridgeError(
                    f"{label} changed during inspection"
                ) from exc
            if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISREG(
                child_info.st_mode
            ):
                raise ProductionLeptonBridgeError(
                    f"{label} contains a symlink or special entry"
                )
            files.append((child, child_info))
    return {
        "stage": stage,
        "directories": directories,
        "files": files,
        "total_bytes": sum(info.st_size for _path, info in files),
    }


def _inspect_private_stage(stage: Path, parent: Path) -> dict[str, Any]:
    return _inspect_directory_stage(
        stage,
        parent,
        pattern=_PROJECTION_STAGE_RE,
        label="campaign projection stage",
    )


def _inspect_private_regular(
    path: Path, parent: Path, *, pattern: re.Pattern[str], label: str
) -> tuple[Path, os.stat_result, Path]:
    if path.parent != parent or pattern.fullmatch(path.name) is None:
        raise ProductionLeptonBridgeError(
            "refusing to inspect a non-private incoming file"
        )
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ProductionLeptonBridgeError(
            f"{label} disappeared during inspection"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProductionLeptonBridgeError(
            f"{label} must be a no-follow regular file"
        )
    return path, info, parent


def _remove_inspected_private_stages(
    inventories: Sequence[Mapping[str, Any]], parent: Path
) -> None:
    """Remove only byte/object-identical stages after every stage was inspected."""

    parents = {parent}
    for inventory in inventories:
        parents.add(Path(inventory["stage"]).parent)
        directories = list(inventory["directories"])
        files = list(inventory["files"])
        # Make directories removable only after the complete no-follow pass for
        # every candidate has succeeded.  Recheck object identity before each
        # mutation so an externally replaced path fails closed.
        for path, expected in sorted(
            directories, key=lambda item: len(item[0].parts)
        ):
            try:
                observed = os.lstat(path)
            except OSError as exc:
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before retirement"
                ) from exc
            if _stat_identity(observed) != _stat_identity(expected):
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before retirement"
                )
            os.chmod(path, 0o700, follow_symlinks=False)
        for path, expected in files:
            try:
                observed = os.lstat(path)
            except OSError as exc:
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before retirement"
                ) from exc
            if (
                _stat_identity(observed) != _stat_identity(expected)
                or stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
            ):
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before retirement"
                )
            path.unlink()
        for path, expected in sorted(
            directories, key=lambda item: len(item[0].parts), reverse=True
        ):
            try:
                observed = os.lstat(path)
            except OSError as exc:
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before directory retirement"
                ) from exc
            if (
                observed.st_dev != expected.st_dev
                or observed.st_ino != expected.st_ino
                or stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISDIR(observed.st_mode)
            ):
                raise ProductionLeptonBridgeError(
                    "campaign projection stage changed before directory retirement"
                )
            path.rmdir()
    for stage_parent in sorted(parents):
        descriptor = os.open(
            stage_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _remove_inspected_private_regulars(
    files: Sequence[tuple[Path, os.stat_result, Path]],
) -> None:
    """Retire only identity-stable files after all candidates were inspected."""

    parents: set[Path] = set()
    for path, expected, parent in files:
        try:
            observed = os.lstat(path)
        except OSError as exc:
            raise ProductionLeptonBridgeError(
                "private incoming file changed before retirement"
            ) from exc
        if (
            _stat_identity(observed) != _stat_identity(expected)
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
        ):
            raise ProductionLeptonBridgeError(
                "private incoming file changed before retirement"
            )
        path.unlink()
        parents.add(parent)
    for parent in sorted(parents):
        descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def prune_orphan_projection_stages(paths: BridgePaths) -> dict[str, int]:
    """Retire exact SIGKILL orphans under one locked learner identity.

    Every directory and regular-file candidate is validated without following
    links before the first mutation.  This covers projection copies, archive
    construction, marker/checksum downloads, returned-bundle downloads, and
    the destination-bound output extraction tree.
    """

    parent = _real_directory(paths.local_handoff, "identity-scoped local handoff")
    artifact_temporary_specs: list[
        tuple[str, re.Pattern[str], str]
    ] = []
    for artifact, random_name_pattern, label in (
        (paths.local_archive, r"[0-9a-f]{32}", "campaign archive"),
        (
            paths.local_inventory,
            r"[a-z0-9_]{8}",
            "campaign inventory",
        ),
        (
            paths.local_manifest,
            r"[a-z0-9_]{8}",
            "campaign manifest",
        ),
    ):
        artifact = Path(os.path.abspath(artifact))
        if artifact.parent != parent:
            raise ProductionLeptonBridgeError(
                f"{label} is outside the identity-scoped handoff"
            )
        prefix = f".{artifact.name}."
        artifact_temporary_specs.append(
            (
                prefix,
                re.compile(
                    rf"\.{re.escape(artifact.name)}\."
                    rf"{random_name_pattern}\.incoming\Z"
                ),
                label,
            )
        )
    directory_candidates: list[
        tuple[Path, Path, re.Pattern[str], str]
    ] = []
    regular_candidates: list[
        tuple[Path, Path, re.Pattern[str], str]
    ] = []
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.name.startswith(_PROJECTION_STAGE_PREFIX):
                    if _PROJECTION_STAGE_RE.fullmatch(entry.name) is None:
                        raise ProductionLeptonBridgeError(
                            "identity handoff contains a malformed private stage name"
                        )
                    directory_candidates.append(
                        (
                            parent / entry.name,
                            parent,
                            _PROJECTION_STAGE_RE,
                            "campaign projection stage",
                        )
                    )
                else:
                    matching_artifact = next(
                        (
                            (pattern, label)
                            for prefix, pattern, label in artifact_temporary_specs
                            if entry.name.startswith(prefix)
                        ),
                        None,
                    )
                    if matching_artifact is not None:
                        pattern, label = matching_artifact
                        if pattern.fullmatch(entry.name) is None:
                            raise ProductionLeptonBridgeError(
                                "identity handoff contains a malformed "
                                f"{label} incoming file"
                            )
                        regular_candidates.append(
                            (
                                parent / entry.name,
                                parent,
                                pattern,
                                f"{label} incoming file",
                            )
                        )
                        continue
                if entry.name.startswith(_DOWNLOAD_STAGE_PREFIX):
                    if _DOWNLOAD_STAGE_RE.fullmatch(entry.name) is None:
                        raise ProductionLeptonBridgeError(
                            "identity handoff contains a malformed download incoming file"
                        )
                    regular_candidates.append(
                        (
                            parent / entry.name,
                            parent,
                            _DOWNLOAD_STAGE_RE,
                            "handoff download incoming file",
                        )
                    )
    except OSError as exc:
        raise ProductionLeptonBridgeError(
            "cannot enumerate identity-scoped private temporaries"
        ) from exc

    return_bundle = Path(os.path.abspath(paths.local_return_bundle))
    if return_bundle.parent != parent:
        raise ProductionLeptonBridgeError(
            "local return bundle is outside the identity-scoped handoff"
        )
    if os.path.lexists(return_bundle):
        return_bundle = _real_directory(return_bundle, "local result bundle")
        try:
            with os.scandir(return_bundle) as entries:
                for entry in entries:
                    if not entry.name.startswith(_DOWNLOAD_STAGE_PREFIX):
                        continue
                    if _DOWNLOAD_STAGE_RE.fullmatch(entry.name) is None:
                        raise ProductionLeptonBridgeError(
                            "result bundle contains a malformed download incoming file"
                        )
                    regular_candidates.append(
                        (
                            return_bundle / entry.name,
                            return_bundle,
                            _DOWNLOAD_STAGE_RE,
                            "result-bundle download incoming file",
                        )
                    )
        except OSError as exc:
            raise ProductionLeptonBridgeError(
                "cannot enumerate local result-bundle temporaries"
            ) from exc

    local_output = Path(os.path.abspath(paths.local_output))
    output_parent = local_output.parent
    if os.path.lexists(output_parent):
        output_parent = _real_directory(
            output_parent, "local learner output parent"
        )
        destination_id = hashlib.sha256(
            str(local_output).encode("utf-8")
        ).hexdigest()[:16]
        output_prefix = f"{_TREE_STAGE_PREFIX}{destination_id}-"
        output_pattern = re.compile(
            rf"{re.escape(output_prefix)}[0-9a-f]{{24}}\Z"
        )
        try:
            with os.scandir(output_parent) as entries:
                for entry in entries:
                    if not entry.name.startswith(output_prefix):
                        continue
                    if output_pattern.fullmatch(entry.name) is None:
                        raise ProductionLeptonBridgeError(
                            "learner output parent contains a malformed extraction stage"
                        )
                    directory_candidates.append(
                        (
                            output_parent / entry.name,
                            output_parent,
                            output_pattern,
                            "learner output extraction stage",
                        )
                    )
        except OSError as exc:
            raise ProductionLeptonBridgeError(
                "cannot enumerate learner output extraction stages"
            ) from exc

    # Complete the no-follow validation pass for every orphan class before
    # deleting even one candidate.  A malicious/special late candidate leaves
    # all earlier safe candidates untouched.
    inventories = [
        _inspect_directory_stage(
            stage, stage_parent, pattern=pattern, label=label
        )
        for stage, stage_parent, pattern, label in sorted(
            directory_candidates, key=lambda item: str(item[0])
        )
    ]
    inspected_regulars = [
        _inspect_private_regular(
            path, file_parent, pattern=pattern, label=label
        )
        for path, file_parent, pattern, label in sorted(
            regular_candidates, key=lambda item: str(item[0])
        )
    ]
    _remove_inspected_private_stages(inventories, parent)
    _remove_inspected_private_regulars(inspected_regulars)
    return {
        "stage_count": len(inventories),
        "regular_file_count": len(inspected_regulars),
        "total_bytes": (
            sum(int(item["total_bytes"]) for item in inventories)
            + sum(info.st_size for _path, info, _parent in inspected_regulars)
        ),
    }


def _remove_private_stage(stage: Path, parent: Path) -> None:
    if not os.path.lexists(stage):
        return
    inventory = _inspect_private_stage(stage, parent)
    _remove_inspected_private_stages([inventory], parent)


def build_campaign_projection(campaign_root: Path, projection: Path) -> str:
    """Create or adopt the immutable learner-facing campaign projection."""

    campaign_root = _real_directory(campaign_root, "campaign root")
    parent = _real_directory(projection.parent, "local handoff", create=True)
    projection = Path(os.path.abspath(projection))
    if projection == campaign_root or projection in campaign_root.parents:
        raise ProductionLeptonBridgeError("projection overlaps the campaign root")
    if campaign_root in projection.parents:
        raise ProductionLeptonBridgeError("projection must be outside the campaign root")
    if os.path.lexists(projection):
        if projection.is_symlink() or not projection.is_dir():
            raise ProductionLeptonBridgeError("existing projection is unsafe")
        tree_transfer.validate_campaign_projection(projection)
        return "adopted"

    stage = Path(
        tempfile.mkdtemp(prefix=_PROJECTION_STAGE_PREFIX, dir=parent)
    )
    try:
        for directory, dirnames, filenames in os.walk(
            campaign_root, topdown=True, followlinks=False
        ):
            source_directory = Path(directory)
            relative_directory = source_directory.relative_to(campaign_root)
            target_directory = stage / relative_directory
            target_directory.mkdir(parents=True, exist_ok=True)
            included_directories: list[str] = []
            for name in sorted(dirnames):
                relative = PurePosixPath(
                    *(relative_directory.parts + (name,))
                )
                if _excluded(relative):
                    continue
                child = source_directory / name
                info = os.lstat(child)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ProductionLeptonBridgeError(
                        f"campaign projection contains unsafe directory: {child}"
                    )
                included_directories.append(name)
                (target_directory / name).mkdir(exist_ok=True)
            dirnames[:] = included_directories
            for name in sorted(filenames):
                relative = PurePosixPath(
                    *(relative_directory.parts + (name,))
                )
                if _excluded(relative):
                    continue
                _copy_stable_regular(
                    source_directory / name, target_directory / name
                )
        tree_transfer.validate_campaign_projection(stage)
        try:
            os.rename(stage, projection)
        except FileExistsError:
            tree_transfer.validate_campaign_projection(projection)
            _remove_private_stage(stage, parent)
            return "adopted"
        directory_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        tree_transfer.validate_campaign_projection(projection)
        return "published"
    except Exception:
        _remove_private_stage(stage, parent)
        raise


def _projected_path(campaign_root: Path, projection: Path, source: Path) -> Path:
    source = Path(os.path.abspath(source))
    try:
        relative = source.relative_to(campaign_root)
    except ValueError as exc:
        raise ProductionLeptonBridgeError(
            f"learner input lies outside campaign: {source}"
        ) from exc
    if not relative.parts or _excluded(PurePosixPath(*relative.parts)):
        raise ProductionLeptonBridgeError(
            f"learner input is excluded from projection: {source}"
        )
    projected = projection / relative
    _regular(source, f"canonical learner input {source}")
    _regular(projected, f"projected learner input {relative}", immutable=True)
    if _sha256_file(source) != _sha256_file(projected):
        raise ProductionLeptonBridgeError(
            f"projected learner input differs from canonical bytes: {relative}"
        )
    return projected


def validate_projected_inputs(campaign_root: Path, projection: Path) -> None:
    for name in _CAMPAIGN_INPUTS:
        value = os.environ.get(name)
        if value is not None:
            _projected_path(campaign_root, projection, Path(value))
    expected_digests = {
        "ROUND_DAGGER_POLICY_CHECKPOINT": "ROUND_DAGGER_POLICY_SHA256",
        "ROUND_DAGGER_REPLAY_SNAPSHOT": "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256",
        "ROUND_DAGGER_POLICY_SPEC": "ROUND_DAGGER_POLICY_SPEC_SHA256",
        "ROUND_DAGGER_POLICY_MANIFEST": "ROUND_DAGGER_POLICY_MANIFEST_SHA256",
        "ROUND_DAGGER_LEARNER_CONTRACT": "ROUND_DAGGER_LEARNER_CONTRACT_SHA256",
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": (
            "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256"
        ),
    }
    for path_name, digest_name in expected_digests.items():
        if path_name in os.environ:
            if _sha256_file(Path(os.environ[path_name])) != _sha_environment(digest_name):
                raise ProductionLeptonBridgeError(
                    f"canonical learner input differs from {digest_name}"
                )


def validate_projected_replay_snapshots(
    campaign_root: Path, projection: Path
) -> dict[str, Any]:
    """Recursively validate replay from the learner's relocated filesystem.

    Snapshot bytes remain identical to the coordinator inputs, and therefore
    retain the exact learner identity.  Cohort discovery is instead rooted in
    the immutable projection so no RTX absolute path can be opened by the H200
    learner.  This gate runs before archive upload and covers the production
    four-cohort cumulative replay plus all eight fixed held-out cohorts.
    """

    train_source = _absolute_environment(
        "ROUND_DAGGER_REPLAY_SNAPSHOT", require_regular=True
    )
    heldout_source = _absolute_environment(
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT", require_regular=True
    )
    train_path = _projected_path(
        campaign_root, projection, train_source
    )
    heldout_path = _projected_path(
        campaign_root, projection, heldout_source
    )
    try:
        training = replay.load_snapshot(
            train_path,
            expected_file_sha256=_sha_environment(
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ),
            artifact_root=projection,
        )
        heldout = replay.load_heldout_snapshot(
            heldout_path,
            expected_file_sha256=_sha_environment(
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256"
            ),
            artifact_root=projection,
        )
    except replay.ReplayError as exc:
        raise ProductionLeptonBridgeError(
            f"projected replay is not portable: {exc}"
        ) from exc

    campaign_id = _token_environment("ROUND_DAGGER_CAMPAIGN_ID")
    macro_round_text = _environment("ROUND_DAGGER_MACRO_ROUND")
    cohort_count_text = _environment("ROUND_DAGGER_COHORT_COUNT")
    if not macro_round_text.isdigit() or not cohort_count_text.isdigit():
        raise ProductionLeptonBridgeError(
            "projected replay production dimensions are malformed"
        )
    expected_cohorts = int(cohort_count_text)
    if (
        training["campaign_id"] != campaign_id
        or training["through_macro_round"] != int(macro_round_text)
        or training["expected_train_cohorts"] != expected_cohorts
        or training["snapshot_sha256"]
        != _sha_environment("ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256")
    ):
        raise ProductionLeptonBridgeError(
            "projected training replay lineage mismatch"
        )
    if (
        heldout["campaign_id"] != campaign_id
        or heldout["cohort_count"] != 8
        or heldout["transition_count"] != 256
        or heldout["snapshot_sha256"]
        != _sha_environment("ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256")
    ):
        raise ProductionLeptonBridgeError(
            "projected held-out replay lineage or fixed-256 coverage mismatch"
        )

    projection = projection.resolve(strict=True)
    for label, snapshot in (("training", training), ("held-out", heldout)):
        for cohort in snapshot["cohorts"]:
            try:
                Path(cohort["cohort_manifest_path"]).relative_to(projection)
            except ValueError as exc:
                raise ProductionLeptonBridgeError(
                    f"projected {label} cohort resolved outside projection"
                ) from exc
        for shard in snapshot["resolved_shards"]:
            for field in ("shard_manifest_path", "payload_path"):
                candidate = Path(shard[field])
                try:
                    candidate.relative_to(projection)
                except ValueError as exc:
                    raise ProductionLeptonBridgeError(
                        f"projected {label} replay resolved outside projection"
                    ) from exc
    return {"training": training, "heldout": heldout}


def derive_paths(contract: LeptonBackendContract) -> BridgePaths:
    campaign_root = _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT")
    campaign_id = _token_environment("ROUND_DAGGER_CAMPAIGN_ID")
    macro_text = _environment("ROUND_DAGGER_MACRO_ROUND")
    if not macro_text.isdigit():
        raise ProductionLeptonBridgeError("ROUND_DAGGER_MACRO_ROUND is malformed")
    macro_round = int(macro_text)
    identity = _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
    handoff_root = _absolute_environment("ROUND_DAGGER_REMOTE_HANDOFF_ROOT")
    local_handoff = (
        handoff_root
        / "lepton_job_v1"
        / campaign_id
        / f"macro_round_{macro_round:06d}"
        / f"identity_{identity}"
    )
    if local_handoff == campaign_root or campaign_root in local_handoff.parents:
        raise ProductionLeptonBridgeError("Lepton handoff root must be outside campaign")
    _real_directory(local_handoff, "identity-scoped local handoff", create=True)

    local_output = _absolute_environment("ROUND_DAGGER_LEARNER_OUTPUT_DIR")
    local_result = _absolute_environment("ROUND_DAGGER_LEARNER_RESULT")
    expected_output = (
        campaign_root / "rounds" / f"macro_round_{macro_round:06d}" / "learner"
    )
    expected_result = expected_output / f"learner_result_round_{macro_round:06d}.json"
    if local_output != expected_output or local_result != expected_result:
        raise ProductionLeptonBridgeError(
            "local learner output/result path is not coordinator-canonical"
        )
    try:
        output_relative = local_output.relative_to(campaign_root)
    except ValueError as exc:
        raise ProductionLeptonBridgeError("learner output lies outside campaign") from exc

    remote_identity_root = (
        contract.project_root
        / "campaigns"
        / campaign_id
        / f"macro_round_{macro_round:06d}"
        / f"identity_{identity}"
    )
    remote_campaign = remote_identity_root / "campaign_input"
    remote_output = remote_campaign / PurePosixPath(*output_relative.parts)
    remote_handoff = contract.project_root / "handoff" / identity
    return BridgePaths(
        local_handoff=local_handoff,
        local_projection=local_handoff / "campaign_input",
        local_archive=local_handoff / "campaign_input.tar.gz",
        local_inventory=local_handoff / "campaign_input.inventory.json",
        local_manifest=local_handoff / "campaign_input.manifest.json",
        local_request=local_handoff / "lepton_learner_request.json",
        local_lifecycle=local_handoff / "exclusive_lifecycle.json",
        local_marker=local_handoff / "remote_learner_success.json",
        local_checksum=local_handoff / "remote_learner_success.json.sha256",
        local_return_bundle=local_handoff / "returned_bundle",
        local_output=local_output,
        local_result=local_result,
        remote_bundle_root=remote_handoff / "campaign_bundle",
        remote_campaign=remote_campaign,
        remote_output=remote_output,
        remote_request=remote_handoff / "lepton_learner_request.json",
    )


def _ensure_remote_directories(
    contract: LeptonBackendContract,
    paths: BridgePaths,
    *,
    sdk_python: Path,
    control_exec: Path,
) -> None:
    directories = sorted(
        {
            str(paths.remote_bundle_root),
            str(paths.remote_campaign.parent),
            str(paths.remote_request.parent),
        }
    )
    remote_code = (
        "import os,stat,sys;"
        "root=os.path.abspath(sys.argv[1]);"
        "ri=os.lstat(root);"
        "assert stat.S_ISDIR(ri.st_mode) and not stat.S_ISLNK(ri.st_mode);"
        "paths=sys.argv[2:];"
        "\nfor path in paths:\n"
        " path=os.path.abspath(path);"
        " assert os.path.commonpath((root,path))==root and path!=root;"
        " current=root;"
        "\n for part in os.path.relpath(path,root).split(os.sep):\n"
        "  assert part not in ('','.','..'); current=os.path.join(current,part);"
        "\n  try: os.mkdir(current,0o755)\n"
        "  except FileExistsError: pass\n"
        "  info=os.lstat(current);"
        "  assert stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)\n"
    )
    remote_command = shlex.join(
        [str(contract.python), "-c", remote_code, str(contract.project_root), *directories]
    )
    deployment = str(contract.document["orchestration"]["control_deployment"])
    _run(
        [str(sdk_python), str(control_exec), deployment, remote_command],
        "identity-scoped remote directory preparation",
    )


def _upload_projection(
    contract: LeptonBackendContract,
    paths: BridgePaths,
    *,
    sdk_python: Path,
) -> tuple[str, dict[str, Any]]:
    try:
        projection_status = build_campaign_projection(
            _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
            paths.local_projection,
        )
        validate_projected_inputs(
            _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
            paths.local_projection,
        )
        validate_projected_replay_snapshots(
            _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
            paths.local_projection,
        )
        pack = archive_transfer.pack_projection(
            paths.local_projection,
            paths.local_archive,
            paths.local_inventory,
            paths.local_manifest,
        )
    except (
        archive_transfer.CampaignArchiveError,
        tree_transfer.TransferInventoryError,
    ) as exc:
        raise ProductionLeptonBridgeError(
            f"learner campaign projection is invalid: {exc}"
        ) from exc
    manifest_sha256 = str(pack["manifest"]["sha256"])
    deployment = str(contract.document["orchestration"]["control_deployment"])
    remote_tool = (
        contract.source_root
        / "scripts/round_dagger/lepton_campaign_archive_transfer.py"
    )
    report = _run_json(
        [
            str(sdk_python),
            str(ARCHIVE_TRANSFER),
            "upload-publish",
            "--archive",
            str(paths.local_archive),
            "--inventory",
            str(paths.local_inventory),
            "--manifest",
            str(paths.local_manifest),
            "--manifest-sha256",
            manifest_sha256,
            "--remote-bundle-root",
            str(paths.remote_bundle_root),
            "--remote-destination",
            str(paths.remote_campaign),
            "--remote-python",
            str(contract.python),
            "--remote-tool",
            str(remote_tool),
            "--deployment",
            deployment,
            "--sdk-python",
            str(sdk_python),
            "--extract-timeout-seconds",
            "3600",
        ],
        "immutable campaign projection upload",
    )
    if (
        report.get("schema") != archive_transfer.REPORT_SCHEMA
        or report.get("operation") != "upload-publish"
        or report.get("status") != "PASS"
        or report.get("manifest_sha256") != manifest_sha256
    ):
        raise ProductionLeptonBridgeError("campaign upload report identity mismatch")
    return projection_status, report


def _materialize_request(
    contract: LeptonBackendContract, paths: BridgePaths
) -> LeptonRunRequest:
    source_environment = dict(os.environ)
    dashboard_environment = {
        name: _environment(name)
        for name in (
            "ROUND_DAGGER_BETA",
            "ROUND_DAGGER_BETA_SHA256",
            "ROUND_DAGGER_SCENE_SAMPLE_MANIFEST",
        )
    }
    campaign_root = _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT")
    scene_manifest = _absolute_environment(
        "ROUND_DAGGER_SCENE_SAMPLE_MANIFEST", require_regular=True
    )
    if (
        scene_manifest.name != "manifest.json"
        or scene_manifest.parent.name != "scene_samples"
    ):
        raise ProductionLeptonBridgeError(
            "scene sample manifest path is non-canonical"
        )
    try:
        scene_relative = scene_manifest.relative_to(campaign_root)
    except ValueError as exc:
        raise ProductionLeptonBridgeError(
            "scene sample manifest lies outside campaign"
        ) from exc
    expected_remote_scene = str(
        paths.remote_campaign / PurePosixPath(*scene_relative.parts)
    )
    if os.path.lexists(paths.local_request):
        request = load_request(paths.local_request, contract=contract, immutable=True)
    else:
        expected_run_id = (
            f"rrd-m{int(_environment('ROUND_DAGGER_MACRO_ROUND')):06d}"
            f"-id{_sha_environment('ROUND_DAGGER_LEARNER_IDENTITY_SHA256')}"
        )
        supplied_run_id = source_environment.get("ROUND_DAGGER_WANDB_RUN_ID")
        if supplied_run_id is not None and supplied_run_id != expected_run_id:
            raise ProductionLeptonBridgeError(
                "ambient W&B run ID differs from exact learner identity"
            )
        source_environment["ROUND_DAGGER_WANDB_RUN_ID"] = expected_run_id
        request = materialize_request(
            paths.local_request,
            contract=contract,
            local_campaign_root=campaign_root,
            remote_campaign_root=str(paths.remote_campaign),
            remote_learner_output=str(paths.remote_output),
            source_environment=source_environment,
        )
    if (
        request.identity_sha256
        != _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
        or request.paths["campaign_root"] != paths.remote_campaign
        or request.paths["learner_output"] != paths.remote_output
    ):
        raise ProductionLeptonBridgeError("materialized request path/identity mismatch")
    for name in ("ROUND_DAGGER_BETA", "ROUND_DAGGER_BETA_SHA256"):
        if request.environment[name] != dashboard_environment[name]:
            raise ProductionLeptonBridgeError(
                f"materialized request {name} differs from coordinator environment"
            )
    if (
        request.environment["ROUND_DAGGER_SCENE_SAMPLE_MANIFEST"]
        != expected_remote_scene
    ):
        raise ProductionLeptonBridgeError(
            "materialized request scene sample manifest differs from coordinator input"
        )
    return request


def _upload_regular(
    local: Path,
    remote: PurePosixPath,
    *,
    deployment: str,
    sdk_python: Path,
) -> None:
    _regular(local, f"immutable upload {local}", immutable=True)
    report = _run_json(
        [
            str(sdk_python),
            str(WEB_TRANSFER),
            "--deployment",
            deployment,
            "--sdk-python",
            str(sdk_python),
            "upload",
            "--local",
            str(local),
            "--remote",
            str(remote),
            "--adopt-existing",
            "--mode",
            "0444",
        ],
        f"immutable upload to {remote}",
    )
    if report.get("operation") != "upload" or report.get("status") not in {
        "published",
        "adopted",
    }:
        raise ProductionLeptonBridgeError("immutable upload report is invalid")


def build_lifecycle_command(
    contract_path: Path,
    request: LeptonRunRequest,
    paths: BridgePaths,
    *,
    sdk_python: Path,
    control_exec: Path,
) -> list[str]:
    return [
        str(sdk_python),
        str(SUBMITTER),
        "submit",
        "--apply",
        "--exclusive-node-lifecycle",
        "--contract",
        str(contract_path),
        "--contract-sha256",
        _sha_environment("ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256"),
        "--request",
        str(paths.local_request),
        "--request-sha256",
        request.sha256,
        "--remote-contract",
        _environment("ROUND_DAGGER_LEPTON_REMOTE_BACKEND_CONTRACT"),
        "--remote-request",
        str(paths.remote_request),
        "--lifecycle-state",
        str(paths.local_lifecycle),
        "--control-exec-program",
        str(control_exec),
        "--transfer-program",
        str(TRANSFER_WRAPPER),
        "--wait-timeout-seconds",
        "172800",
    ]


def _download_regular(
    remote: PurePosixPath,
    local: Path,
    *,
    deployment: str,
    sdk_python: Path,
) -> None:
    _real_directory(local.parent, "completion evidence directory", create=True)
    report = _run_json(
        [
            str(sdk_python),
            str(WEB_TRANSFER),
            "--deployment",
            deployment,
            "--sdk-python",
            str(sdk_python),
            "download",
            "--remote",
            str(remote),
            "--local",
            str(local),
            "--adopt-existing",
            "--mode",
            "0444",
        ],
        f"immutable download from {remote}",
    )
    if report.get("operation") != "download" or report.get("status") not in {
        "published",
        "adopted",
    }:
        raise ProductionLeptonBridgeError("immutable download report is invalid")


def _fetch_and_validate_marker(
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    paths: BridgePaths,
    *,
    sdk_python: Path,
) -> dict[str, Any]:
    expected_remote_marker = (
        paths.remote_output / _REMOTE_SUCCESS_MARKER_BASENAME
    )
    expected_remote_checksum = PurePosixPath(f"{expected_remote_marker}.sha256")
    if (
        request.paths["success_marker"] != expected_remote_marker
        or request.paths["success_checksum"] != expected_remote_checksum
    ):
        raise ProductionLeptonBridgeError(
            "remote completion evidence path is non-canonical"
        )
    deployment = str(contract.document["orchestration"]["control_deployment"])
    _download_regular(
        request.paths["success_marker"],
        paths.local_marker,
        deployment=deployment,
        sdk_python=sdk_python,
    )
    _download_regular(
        request.paths["success_checksum"],
        paths.local_checksum,
        deployment=deployment,
        sdk_python=sdk_python,
    )
    marker_sha = _verify_transferred_completion_checksum(paths)
    evidence = validate_completion_marker(
        paths.local_marker, request=request, contract=contract
    )
    if marker_sha != _sha256_file(paths.local_marker):
        raise ProductionLeptonBridgeError("completion marker checksum changed")
    return evidence


def _download_output(
    contract: LeptonBackendContract,
    evidence: Mapping[str, Any],
    paths: BridgePaths,
    *,
    sdk_python: Path,
) -> dict[str, Any]:
    manifest = PurePosixPath(str(evidence["learner_output_manifest_path"]))
    expected_manifest = (
        contract.project_root
        / "results"
        / _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
        / "learner_output.manifest.json"
    )
    if manifest != expected_manifest:
        raise ProductionLeptonBridgeError("remote learner bundle path is non-canonical")
    _real_directory(paths.local_return_bundle, "local result bundle", create=True)
    _real_directory(paths.local_output.parent, "local learner output parent", create=True)
    deployment = str(contract.document["orchestration"]["control_deployment"])
    report = _run_json(
        [
            str(sdk_python),
            str(ARCHIVE_TRANSFER),
            "download-publish",
            "--remote-archive",
            str(manifest.parent / "learner_output.tar.gz"),
            "--remote-inventory",
            str(manifest.parent / "learner_output.inventory.json"),
            "--remote-manifest",
            str(manifest),
            "--manifest-sha256",
            str(evidence["learner_output_manifest_sha256"]),
            "--local-bundle-root",
            str(paths.local_return_bundle),
            "--destination",
            str(paths.local_output),
            "--deployment",
            deployment,
            "--sdk-python",
            str(sdk_python),
        ],
        "inventory-bound learner output return",
    )
    publication = report.get("publication")
    if (
        report.get("schema") != archive_transfer.REPORT_SCHEMA
        or report.get("operation") != "download-publish"
        or report.get("status") != "PASS"
        or report.get("manifest_sha256")
        != evidence["learner_output_manifest_sha256"]
        or not isinstance(publication, Mapping)
        or publication.get("archive_sha256")
        != evidence["learner_output_archive_sha256"]
        or publication.get("inventory_sha256")
        != evidence["learner_output_inventory_sha256"]
        or publication.get("tree_sha256")
        != evidence["learner_output_tree_sha256"]
    ):
        raise ProductionLeptonBridgeError("returned learner bundle identity mismatch")
    return report


def validate_local_result(
    paths: BridgePaths, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if _sha256_file(paths.local_result) != evidence["result_sha256"]:
        raise ProductionLeptonBridgeError("returned learner result differs from marker")
    macro_round = int(_environment("ROUND_DAGGER_MACRO_ROUND"))
    config = SimpleNamespace(
        campaign_root=_absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
        macro_round=macro_round,
        learner_code_sha256=_sha_environment("ROUND_DAGGER_LEARNER_CODE_SHA256"),
        learner_config_sha256=_sha_environment("ROUND_DAGGER_LEARNER_CONFIG_SHA256"),
        learner_model_lineage_sha256=_sha_environment(
            "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256"
        ),
    )
    backend = orchestration.ProductionBackend(config)
    try:
        campaign = state.load_campaign(config.campaign_root)
        snapshot = replay.load_snapshot(
            _absolute_environment("ROUND_DAGGER_REPLAY_SNAPSHOT"),
            expected_file_sha256=_sha_environment(
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ),
        )
        result = backend.load_learner_result(campaign=campaign, snapshot=snapshot)
    except Exception as exc:
        raise ProductionLeptonBridgeError(
            f"returned standard learner result/candidate is invalid: {exc}"
        ) from exc
    if Path(result["manifest_path"]) != paths.local_result:
        raise ProductionLeptonBridgeError("validated learner result path changed")
    return result


def publish_local_acceptance(
    *,
    contract_path: Path,
    paths: BridgePaths,
) -> dict[str, Any]:
    """Commit the fully revalidated remote return inside the campaign."""

    macro_round = int(_environment("ROUND_DAGGER_MACRO_ROUND"))
    campaign_root = _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT")
    receipt = learner_acceptance.acceptance_path(campaign_root, macro_round)
    try:
        return learner_acceptance.publish_acceptance(
            receipt_path=receipt,
            campaign_root=campaign_root,
            macro_round=macro_round,
            backend_contract_path=contract_path,
            request_path=paths.local_request,
            marker_path=paths.local_marker,
            checksum_path=paths.local_checksum,
            return_bundle_root=paths.local_return_bundle,
            learner_output=paths.local_output,
            expected_environment=os.environ,
        )
    except learner_acceptance.LearnerAcceptanceError as exc:
        raise ProductionLeptonBridgeError(
            f"campaign-local learner acceptance failed: {exc}"
        ) from exc


def _retirement_path_guard(paths: BridgePaths) -> Path:
    """Prove that cleanup targets only one identity-scoped handoff copy."""

    handoff = _real_directory(paths.local_handoff, "identity-scoped local handoff")
    campaign_id = _token_environment("ROUND_DAGGER_CAMPAIGN_ID")
    identity = _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
    macro_text = _environment("ROUND_DAGGER_MACRO_ROUND")
    if not macro_text.isdigit():
        raise ProductionLeptonBridgeError("retirement macro round is malformed")
    expected_tail = (
        "lepton_job_v1",
        campaign_id,
        f"macro_round_{int(macro_text):06d}",
        f"identity_{identity}",
    )
    if tuple(handoff.parts[-4:]) != expected_tail:
        raise ProductionLeptonBridgeError(
            "local handoff is not the exact learner identity scope"
        )
    expected_children = {
        paths.local_projection: "campaign_input",
        paths.local_archive: "campaign_input.tar.gz",
        paths.local_inventory: "campaign_input.inventory.json",
        paths.local_manifest: "campaign_input.manifest.json",
    }
    for candidate, basename in expected_children.items():
        if candidate.parent != handoff or candidate.name != basename:
            raise ProductionLeptonBridgeError(
                "retirement artifact is outside the identity-scoped handoff"
            )
    campaign_root = _real_directory(
        _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"), "campaign root"
    )
    if (
        handoff == campaign_root
        or handoff in campaign_root.parents
        or campaign_root in handoff.parents
    ):
        raise ProductionLeptonBridgeError(
            "identity-scoped handoff overlaps canonical campaign replay"
        )
    return handoff


def _publish_retirement_document(path: Path, body: Mapping[str, Any]) -> str:
    document = dict(body)
    document["content_sha256"] = artifacts.canonical_json_sha256(document)
    try:
        digest = artifacts.publish_immutable_json(path, document)
        loaded = artifacts.load_strict_json(path)
    except Exception as exc:
        raise ProductionLeptonBridgeError(
            "cannot publish immutable handoff-retirement evidence"
        ) from exc
    if loaded != document or _sha256_file(path) != digest:
        raise ProductionLeptonBridgeError(
            "handoff-retirement evidence failed immutable round-trip"
        )
    return digest


def _retire_projection_subset(
    projection: Path,
    *,
    inventory: Mapping[str, Any],
    allowed_directories: Sequence[str],
) -> None:
    """Remove an exact remaining subset of an inventory-bound projection."""

    if not os.path.lexists(projection):
        return
    if projection.is_symlink() or not projection.is_dir():
        raise ProductionLeptonBridgeError("retirement projection is unsafe")
    records = {str(item["path"]): dict(item) for item in inventory["files"]}
    allowed = set(allowed_directories)
    observed_files: list[Path] = []
    observed_directories: list[Path] = []
    for directory, dirnames, filenames in os.walk(
        projection, topdown=True, followlinks=False
    ):
        base = Path(directory)
        for name in sorted(dirnames):
            child = base / name
            relative = child.relative_to(projection).as_posix()
            if child.is_symlink() or not child.is_dir() or relative not in allowed:
                raise ProductionLeptonBridgeError(
                    "retirement projection contains an unbound directory"
                )
            observed_directories.append(child)
        for name in sorted(filenames):
            child = base / name
            relative = child.relative_to(projection).as_posix()
            record = records.get(relative)
            if child.is_symlink() or not child.is_file() or record is None:
                raise ProductionLeptonBridgeError(
                    "retirement projection contains an unbound file"
                )
            info = os.lstat(child)
            if info.st_size != record["size"] or _sha256_file(child) != record["sha256"]:
                raise ProductionLeptonBridgeError(
                    "retirement projection file differs from immutable inventory"
                )
            observed_files.append(child)
    for path in observed_files:
        path.unlink()
    for path in sorted(
        observed_directories,
        key=lambda item: len(item.relative_to(projection).parts),
        reverse=True,
    ):
        path.rmdir()
    projection.rmdir()


def retire_local_handoff_inputs(
    paths: BridgePaths,
    *,
    upload_report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Retire only redundant local learner inputs after verified output return.

    The canonical campaign, replay ledger/snapshots, accepted cohorts, returned
    output, request, completion marker/checksum and transfer inventories are
    retained.  An immutable intent is published before unlinking, so a retry
    can safely finish a partially completed cleanup using the retained exact
    inventory as its allowlist.
    """

    handoff = _retirement_path_guard(paths)
    if _verify_transferred_completion_checksum(paths) != _sha256_file(
        paths.local_marker
    ):
        raise ProductionLeptonBridgeError("completion marker checksum changed")
    marker = artifacts.load_strict_json(paths.local_marker)
    if marker != dict(evidence):
        raise ProductionLeptonBridgeError(
            "retirement evidence differs from validated completion marker"
        )
    if (
        evidence.get("status") != "PASS"
        or evidence.get("campaign_id") != _token_environment("ROUND_DAGGER_CAMPAIGN_ID")
        or evidence.get("macro_round") != int(_environment("ROUND_DAGGER_MACRO_ROUND"))
        or evidence.get("learner_identity_sha256")
        != _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
        or _sha256_file(paths.local_result) != evidence.get("result_sha256")
        or Path(str(result.get("manifest_path"))) != paths.local_result
        or result.get("checkpoint_sha256") is None
    ):
        raise ProductionLeptonBridgeError(
            "local learner output is not fully validated for input retirement"
        )
    _regular(paths.local_request, "immutable learner request", immutable=True)
    if (
        upload_report.get("schema") != archive_transfer.REPORT_SCHEMA
        or upload_report.get("operation") != "upload-publish"
        or upload_report.get("status") != "PASS"
        or not isinstance(upload_report.get("publication"), Mapping)
    ):
        raise ProductionLeptonBridgeError(
            "campaign upload evidence is invalid for input retirement"
        )
    publication = dict(upload_report["publication"])
    try:
        manifest = archive_transfer._load_manifest(paths.local_manifest)
        inventory = tree_transfer.load_inventory(paths.local_inventory)
    except (
        archive_transfer.CampaignArchiveError,
        tree_transfer.TransferInventoryError,
    ) as exc:
        raise ProductionLeptonBridgeError(
            "retained campaign transfer evidence is invalid"
        ) from exc
    manifest_sha256 = _sha256_file(paths.local_manifest)
    if (
        upload_report.get("manifest_sha256") != manifest_sha256
        or publication.get("manifest_sha256") != manifest_sha256
        or publication.get("archive_sha256") != manifest["archive"]["sha256"]
        or publication.get("inventory_sha256") != manifest["inventory"]["sha256"]
        or publication.get("tree_sha256") != manifest["inventory"]["tree_sha256"]
        or inventory["tree_sha256"] != manifest["inventory"]["tree_sha256"]
        or _sha256_file(paths.local_inventory) != manifest["inventory"]["sha256"]
    ):
        raise ProductionLeptonBridgeError(
            "campaign upload and retained transfer identities disagree"
        )

    intent_body = {
        "schema": RETIREMENT_INTENT_SCHEMA,
        "status": "AUTHORIZED",
        "campaign_id": evidence["campaign_id"],
        "macro_round": evidence["macro_round"],
        "learner_identity_sha256": evidence["learner_identity_sha256"],
        "request_sha256": _sha256_file(paths.local_request),
        "result_sha256": evidence["result_sha256"],
        "candidate_checkpoint_sha256": result["checkpoint_sha256"],
        "completion_marker_sha256": _sha256_file(paths.local_marker),
        "campaign_manifest_sha256": manifest_sha256,
        "projection_tree_sha256": inventory["tree_sha256"],
        "projection_total_bytes": inventory["total_bytes"],
        "archive_sha256": manifest["archive"]["sha256"],
        "archive_size_bytes": manifest["archive"]["size"],
        "retirement_targets": ["campaign_input", "campaign_input.tar.gz"],
    }
    intent_path = handoff / "local_input_retirement_intent.json"
    intent_sha256 = _publish_retirement_document(intent_path, intent_body)

    if os.path.lexists(paths.local_archive):
        archive_info = _regular(
            paths.local_archive, "identity-scoped campaign archive", immutable=True
        )
        if (
            archive_info.st_size != manifest["archive"]["size"]
            or _sha256_file(paths.local_archive) != manifest["archive"]["sha256"]
        ):
            raise ProductionLeptonBridgeError(
                "campaign archive differs from retirement intent"
            )
        paths.local_archive.unlink()
    _retire_projection_subset(
        paths.local_projection,
        inventory=inventory,
        allowed_directories=manifest["members"]["directories"],
    )
    directory_descriptor = os.open(
        handoff, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)

    complete_path = handoff / "local_input_retirement_complete.json"
    already_complete = complete_path.exists()
    complete_body = {
        "schema": RETIREMENT_COMPLETE_SCHEMA,
        "status": "PASS",
        "campaign_id": evidence["campaign_id"],
        "macro_round": evidence["macro_round"],
        "learner_identity_sha256": evidence["learner_identity_sha256"],
        "intent_sha256": intent_sha256,
        "retired_projection_bytes": inventory["total_bytes"],
        "retired_archive_bytes": manifest["archive"]["size"],
        "retained_evidence": [
            "campaign_input.inventory.json",
            "campaign_input.manifest.json",
            "lepton_learner_request.json",
            "remote_learner_success.json",
            "remote_learner_success.json.sha256",
            "returned_bundle",
        ],
    }
    completion_sha256 = _publish_retirement_document(complete_path, complete_body)
    if os.path.lexists(paths.local_projection) or os.path.lexists(paths.local_archive):
        raise ProductionLeptonBridgeError(
            "identity-scoped local learner inputs remain after retirement"
        )
    return {
        "status": "adopted" if already_complete else "published",
        "completion_path": str(complete_path),
        "completion_sha256": completion_sha256,
        "retired_bytes": inventory["total_bytes"] + manifest["archive"]["size"],
    }


def _load_exact_retirement_intent(paths: BridgePaths) -> dict[str, Any] | None:
    """Load the durable cleanup commit record, or prove that none exists."""

    intent_path = paths.local_handoff / "local_input_retirement_intent.json"
    complete_path = paths.local_handoff / "local_input_retirement_complete.json"
    if not os.path.lexists(intent_path):
        if os.path.lexists(complete_path):
            raise ProductionLeptonBridgeError(
                "retirement completion exists without its immutable intent"
            )
        return None
    handoff = _retirement_path_guard(paths)
    if intent_path.parent != handoff or complete_path.parent != handoff:
        raise ProductionLeptonBridgeError(
            "retirement recovery evidence is outside the identity handoff"
        )
    _regular(intent_path, "local input retirement intent", immutable=True)
    try:
        document = artifacts.load_strict_json(intent_path)
    except Exception as exc:
        raise ProductionLeptonBridgeError(
            "cannot load immutable local input retirement intent"
        ) from exc
    if not isinstance(document, Mapping) or set(document) != _RETIREMENT_INTENT_FIELDS:
        raise ProductionLeptonBridgeError(
            "local input retirement intent fields are not exact"
        )
    intent = dict(document)
    content = dict(intent)
    claimed_content_sha256 = content.pop("content_sha256")
    if (
        not isinstance(claimed_content_sha256, str)
        or _SHA256.fullmatch(claimed_content_sha256) is None
        or artifacts.canonical_json_sha256(content) != claimed_content_sha256
    ):
        raise ProductionLeptonBridgeError(
            "local input retirement intent content digest mismatch"
        )
    if (
        intent["schema"] != RETIREMENT_INTENT_SCHEMA
        or intent["status"] != "AUTHORIZED"
        or intent["campaign_id"] != _token_environment("ROUND_DAGGER_CAMPAIGN_ID")
        or type(intent["macro_round"]) is not int
        or intent["macro_round"] < 0
        or intent["macro_round"] != int(_environment("ROUND_DAGGER_MACRO_ROUND"))
        or intent["learner_identity_sha256"]
        != _sha_environment("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
        or intent["retirement_targets"]
        != ["campaign_input", "campaign_input.tar.gz"]
    ):
        raise ProductionLeptonBridgeError(
            "local input retirement intent lineage mismatch"
        )
    for key in (
        "learner_identity_sha256",
        "request_sha256",
        "result_sha256",
        "candidate_checkpoint_sha256",
        "completion_marker_sha256",
        "campaign_manifest_sha256",
        "projection_tree_sha256",
        "archive_sha256",
    ):
        if not isinstance(intent[key], str) or _SHA256.fullmatch(intent[key]) is None:
            raise ProductionLeptonBridgeError(
                f"local input retirement intent {key} is malformed"
            )
    for key in ("projection_total_bytes", "archive_size_bytes"):
        if type(intent[key]) is not int or intent[key] < 0:
            raise ProductionLeptonBridgeError(
                f"local input retirement intent {key} is malformed"
            )
    return intent


def recover_local_handoff_retirement(
    contract: LeptonBackendContract, paths: BridgePaths
) -> dict[str, Any] | None:
    """Finish an intent-committed cleanup without any remote side effect."""

    intent = _load_exact_retirement_intent(paths)
    if intent is None:
        return None
    try:
        request = load_request(paths.local_request, contract=contract, immutable=True)
    except LeptonJobError as exc:
        raise ProductionLeptonBridgeError(
            "retirement recovery request is invalid"
        ) from exc
    if (
        request.sha256 != intent["request_sha256"]
        or _sha256_file(paths.local_request) != intent["request_sha256"]
        or request.campaign_id != intent["campaign_id"]
        or request.macro_round != intent["macro_round"]
        or request.identity_sha256 != intent["learner_identity_sha256"]
        or request.paths["campaign_root"] != paths.remote_campaign
        or request.paths["learner_output"] != paths.remote_output
    ):
        raise ProductionLeptonBridgeError(
            "retirement recovery request differs from the immutable intent"
        )
    try:
        marker_sha256 = _verify_transferred_completion_checksum(paths)
        evidence = validate_completion_marker(
            paths.local_marker, request=request, contract=contract
        )
    except LeptonJobError as exc:
        raise ProductionLeptonBridgeError(
            "retirement recovery completion evidence is invalid"
        ) from exc
    if (
        marker_sha256 != intent["completion_marker_sha256"]
        or _sha256_file(paths.local_marker) != marker_sha256
        or evidence.get("result_sha256") != intent["result_sha256"]
    ):
        raise ProductionLeptonBridgeError(
            "retirement recovery completion evidence differs from the intent"
        )
    result = validate_local_result(paths, evidence)
    if result.get("checkpoint_sha256") != intent["candidate_checkpoint_sha256"]:
        raise ProductionLeptonBridgeError(
            "retirement recovery candidate differs from the intent"
        )

    _regular(paths.local_manifest, "retained campaign manifest", immutable=True)
    _regular(paths.local_inventory, "retained campaign inventory", immutable=True)
    try:
        manifest = archive_transfer._load_manifest(paths.local_manifest)
        inventory = tree_transfer.load_inventory(paths.local_inventory)
    except (
        archive_transfer.CampaignArchiveError,
        tree_transfer.TransferInventoryError,
    ) as exc:
        raise ProductionLeptonBridgeError(
            "retirement recovery transfer evidence is invalid"
        ) from exc
    manifest_sha256 = _sha256_file(paths.local_manifest)
    if (
        manifest_sha256 != intent["campaign_manifest_sha256"]
        or _sha256_file(paths.local_inventory) != manifest["inventory"]["sha256"]
        or inventory["tree_sha256"] != intent["projection_tree_sha256"]
        or inventory["total_bytes"] != intent["projection_total_bytes"]
        or manifest["inventory"]["tree_sha256"] != intent["projection_tree_sha256"]
        or manifest["archive"]["sha256"] != intent["archive_sha256"]
        or manifest["archive"]["size"] != intent["archive_size_bytes"]
    ):
        raise ProductionLeptonBridgeError(
            "retirement recovery transfer evidence differs from the intent"
        )
    upload_report = {
        "schema": archive_transfer.REPORT_SCHEMA,
        "operation": "upload-publish",
        "status": "PASS",
        "manifest_sha256": manifest_sha256,
        "publication": {
            "manifest_sha256": manifest_sha256,
            "archive_sha256": manifest["archive"]["sha256"],
            "inventory_sha256": manifest["inventory"]["sha256"],
            "tree_sha256": manifest["inventory"]["tree_sha256"],
        },
    }
    retirement = retire_local_handoff_inputs(
        paths,
        upload_report=upload_report,
        evidence=evidence,
        result=result,
    )
    return {
        "intent": intent,
        "request": request,
        "evidence": evidence,
        "result": result,
        "retirement": retirement,
    }


def _emit_success_report(
    *,
    request: LeptonRunRequest,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    retirement: Mapping[str, Any],
    projection_publication: str,
    campaign_manifest_sha256: str,
    learner_output_publication: str,
    acceptance: Mapping[str, Any],
) -> None:
    print(
        json.dumps(
            {
                "schema": BRIDGE_REPORT_SCHEMA,
                "status": "PASS",
                "campaign_id": request.campaign_id,
                "macro_round": request.macro_round,
                "learner_identity_sha256": request.identity_sha256,
                "projection_publication": projection_publication,
                "campaign_manifest_sha256": campaign_manifest_sha256,
                "request_sha256": request.sha256,
                "learner_output_manifest_sha256": evidence[
                    "learner_output_manifest_sha256"
                ],
                "learner_output_tree_sha256": evidence[
                    "learner_output_tree_sha256"
                ],
                "learner_output_publication": learner_output_publication,
                "result_sha256": evidence["result_sha256"],
                "candidate_checkpoint_sha256": result["checkpoint_sha256"],
                "learner_acceptance_sha256": artifacts.sha256_file(
                    learner_acceptance.acceptance_path(
                        _absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
                        request.macro_round,
                    )
                ),
                "learner_acceptance_content_sha256": acceptance[
                    "content_sha256"
                ],
                "optimizer_steps": result["optimizer_steps"],
                "local_input_retirement": retirement["status"],
                "local_input_retirement_sha256": retirement[
                    "completion_sha256"
                ],
                "local_input_retired_bytes": retirement["retired_bytes"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _load_backend() -> tuple[Path, LeptonBackendContract, Path, Path]:
    if _environment("ROUND_DAGGER_LEARNER_BACKEND") != "lepton_job_v1":
        raise ProductionLeptonBridgeError("backend is not exactly lepton_job_v1")
    contract_path = _absolute_environment(
        "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT", require_regular=True
    )
    try:
        contract = load_contract(contract_path, immutable=True)
        require_finalized_supply_chain(contract)
    except LeptonJobError as exc:
        raise ProductionLeptonBridgeError(
            f"finalized Lepton backend contract is invalid: {exc}"
        ) from exc
    if contract.sha256 != _sha_environment(
        "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256"
    ):
        raise ProductionLeptonBridgeError("Lepton backend contract SHA-256 mismatch")
    sdk_python = _absolute_environment("ROUND_DAGGER_LEPTON_SDK_PYTHON")
    _verify_control_python(sdk_python)
    control_exec = _absolute_environment(
        "ROUND_DAGGER_LEPTON_CONTROL_EXEC_PROGRAM", require_regular=True
    )
    if not os.access(control_exec, os.X_OK):
        raise ProductionLeptonBridgeError("Lepton control exec program is not executable")
    remote_contract = _environment("ROUND_DAGGER_LEPTON_REMOTE_BACKEND_CONTRACT")
    try:
        remote_contract_path = PurePosixPath(remote_contract)
        remote_contract_path.relative_to(contract.project_root)
    except (TypeError, ValueError) as exc:
        raise ProductionLeptonBridgeError(
            "remote Lepton backend contract is outside project root"
        ) from exc
    if (
        not remote_contract_path.is_absolute()
        or remote_contract_path.as_posix() != remote_contract
        or any(part in {"", ".", ".."} for part in remote_contract_path.parts)
    ):
        raise ProductionLeptonBridgeError(
            "remote Lepton backend contract is non-canonical"
        )
    return contract_path, contract, sdk_python, control_exec


def main() -> int:
    contract_path, contract, sdk_python, control_exec = _load_backend()
    paths = derive_paths(contract)
    lock_path = paths.local_handoff / ".bridge.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionLeptonBridgeError(
                "another bridge owns this learner identity"
            ) from exc
        prune_orphan_projection_stages(paths)
        recovered = recover_local_handoff_retirement(contract, paths)
        if recovered is not None:
            acceptance = publish_local_acceptance(
                contract_path=contract_path,
                paths=paths,
            )
            _emit_success_report(
                request=recovered["request"],
                evidence=recovered["evidence"],
                result=recovered["result"],
                retirement=recovered["retirement"],
                projection_publication="retired-recovery",
                campaign_manifest_sha256=recovered["intent"][
                    "campaign_manifest_sha256"
                ],
                learner_output_publication="adopted-local",
                acceptance=acceptance,
            )
            return 0
        request = _materialize_request(contract, paths)
        _ensure_remote_directories(
            contract, paths, sdk_python=sdk_python, control_exec=control_exec
        )
        projection_status, upload_report = _upload_projection(
            contract, paths, sdk_python=sdk_python
        )
        deployment = str(contract.document["orchestration"]["control_deployment"])
        _upload_regular(
            paths.local_request,
            paths.remote_request,
            deployment=deployment,
            sdk_python=sdk_python,
        )
        _run(
            build_lifecycle_command(
                contract_path,
                request,
                paths,
                sdk_python=sdk_python,
                control_exec=control_exec,
            ),
            "exclusive stop-Job-restore learner lifecycle",
        )
        evidence = _fetch_and_validate_marker(
            contract, request, paths, sdk_python=sdk_python
        )
        return_report = _download_output(
            contract, evidence, paths, sdk_python=sdk_python
        )
        result = validate_local_result(paths, evidence)
        acceptance = publish_local_acceptance(
            contract_path=contract_path,
            paths=paths,
        )
        retirement = retire_local_handoff_inputs(
            paths,
            upload_report=upload_report,
            evidence=evidence,
            result=result,
        )
        _emit_success_report(
            request=request,
            evidence=evidence,
            result=result,
            retirement=retirement,
            projection_publication=projection_status,
            campaign_manifest_sha256=upload_report["manifest_sha256"],
            learner_output_publication=return_report["publication"]["publication"],
            acceptance=acceptance,
        )
        return 0
    finally:
        os.close(lock_descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ProductionLeptonBridgeError,
        LeptonJobError,
        archive_transfer.CampaignArchiveError,
        tree_transfer.TransferInventoryError,
        OSError,
    ) as exc:
        print(f"production Lepton learner bridge failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
