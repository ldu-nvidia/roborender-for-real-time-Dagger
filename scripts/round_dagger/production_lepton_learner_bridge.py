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
import shutil
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

from round_dagger import orchestration, replay, state  # noqa: E402
from round_dagger.lepton_job import (  # noqa: E402
    LeptonBackendContract,
    LeptonJobError,
    LeptonRunRequest,
    load_contract,
    load_request,
    materialize_request,
    require_finalized_supply_chain,
    validate_completion_marker,
    verify_checksum,
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

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_EXCLUDED_TOP_LEVEL = frozenset({"actors", "operator", ".coordinator_state.lock"})
_CAMPAIGN_INPUTS = (
    "ROUND_DAGGER_POLICY_CHECKPOINT",
    "ROUND_DAGGER_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_BASE_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_MANIFEST",
    "ROUND_DAGGER_LEARNER_CONTRACT",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT",
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


def _remove_private_stage(stage: Path, parent: Path) -> None:
    if stage.parent != parent or not stage.name.startswith(".incoming-campaign-"):
        raise ProductionLeptonBridgeError("refusing to remove a non-private stage")
    if not os.path.lexists(stage):
        return
    if stage.is_symlink() or not stage.is_dir():
        raise ProductionLeptonBridgeError("campaign projection stage is unsafe")
    for directory, dirnames, _filenames in os.walk(stage, topdown=False):
        for name in dirnames:
            os.chmod(Path(directory) / name, 0o700)
        os.chmod(directory, 0o700)
    shutil.rmtree(stage)


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
        tempfile.mkdtemp(prefix=".incoming-campaign-", dir=parent)
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
    if os.path.lexists(paths.local_request):
        request = load_request(paths.local_request, contract=contract, immutable=True)
    else:
        source_environment = dict(os.environ)
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
            local_campaign_root=_absolute_environment("ROUND_DAGGER_CAMPAIGN_ROOT"),
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
    marker_sha = verify_checksum(paths.local_marker, paths.local_checksum)
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
        _ensure_remote_directories(
            contract, paths, sdk_python=sdk_python, control_exec=control_exec
        )
        projection_status, upload_report = _upload_projection(
            contract, paths, sdk_python=sdk_python
        )
        request = _materialize_request(contract, paths)
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
        print(
            json.dumps(
                {
                    "schema": BRIDGE_REPORT_SCHEMA,
                    "status": "PASS",
                    "campaign_id": request.campaign_id,
                    "macro_round": request.macro_round,
                    "learner_identity_sha256": request.identity_sha256,
                    "projection_publication": projection_status,
                    "campaign_manifest_sha256": upload_report["manifest_sha256"],
                    "request_sha256": request.sha256,
                    "learner_output_manifest_sha256": evidence[
                        "learner_output_manifest_sha256"
                    ],
                    "learner_output_tree_sha256": evidence[
                        "learner_output_tree_sha256"
                    ],
                    "learner_output_publication": return_report["publication"][
                        "publication"
                    ],
                    "result_sha256": evidence["result_sha256"],
                    "candidate_checkpoint_sha256": result["checkpoint_sha256"],
                    "optimizer_steps": result["optimizer_steps"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
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
