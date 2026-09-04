"""Fail-closed unattended driver for the bounded five-round campaign.

The one-round orchestrator remains the owner of every durable campaign state
transition.  This module only selects the current durable round, publishes a
fresh immutable orchestration configuration for that exact active policy, and
invokes the already validated production launcher.  A restart therefore
resumes the phase recorded by the coordinator; it never guesses which work
finished and never advances state itself.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from round_dagger import artifacts, state
from round_dagger.orchestration import (
    RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
    LEARNER_BACKEND_LEPTON_JOB_V1,
    LEARNER_BACKEND_OCI_NRT,
    MacroRoundConfig,
    trusted_train_hook_relative,
    validate_learner_backend,
)


DRIVER_STATUS_SCHEMA = "fast_roborender.round_dagger.production_driver_status.v1"
DRIVER_BINDING_SCHEMA = "fast_roborender.round_dagger.production_driver_binding.v2"
COLLECTION_FAILURE_SCHEMA = (
    "fast_roborender.round_dagger.collection_attempt_failure.v1"
)
COLLECTION_ATTEMPT_SCHEMA = (
    "fast_roborender.round_dagger.collection_launcher_attempt.v1"
)
ORCHESTRATION_SCHEMA = RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA
PRODUCTION_MACRO_ROUNDS = 5
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SENSITIVE_RE = re.compile(
    r"(?i)(api[_-]?key|password|secret|access[_-]?token|identity[_-]?token)"
)
_WANDB_SECRET_RE = re.compile(r"wandb_v1_[A-Za-z0-9_-]+")
_PHASES = {"READY", "COLLECTING", "COLLECTED", "INGESTED", "TRAINED", "EVALUATED"}
_ATTEMPT_STAGING_RE = re.compile(
    r"^\.macro_round_[0-9]{6}_policy_[0-9]{6}_[0-9a-f]{12}_"
    r"attempt_[0-9]{6}\.json\.staging$"
)
_FAILURE_STAGING_RE = re.compile(
    r"^\.macro_round_[0-9]{6}_policy_[0-9]{6}_[0-9a-f]{12}_"
    r"failure_[0-9]{6}\.json\.staging$"
)
_OPERATOR_SIGNAL_RETURN_CODES = {
    128 + signal.SIGHUP,
    128 + signal.SIGINT,
    128 + signal.SIGTERM,
}


class ProductionDriverError(RuntimeError):
    """The unattended driver could not prove that continuing was safe."""


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ProductionDriverError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        raise ProductionDriverError(f"{label} must be absolute")
    return path


def _check_path_chain(path: Path, label: str) -> None:
    """Reject every existing symlink or non-directory ancestor."""

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ProductionDriverError(f"{label} contains a symbolic link: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ProductionDriverError(
                f"{label} contains a non-directory ancestor: {current}"
            )


def _lstat_regular(path: Path, label: str, *, immutable: bool = False) -> None:
    _check_path_chain(path, label)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProductionDriverError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ProductionDriverError(f"{label} must be a regular non-symlink file: {path}")
    if immutable and info.st_mode & 0o222:
        raise ProductionDriverError(f"{label} must be non-writable: {path}")


def _lstat_directory(path: Path, label: str) -> None:
    _check_path_chain(path, label)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProductionDriverError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ProductionDriverError(f"{label} must be a non-symlink directory: {path}")


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    _lstat_regular(path, label, immutable=True)
    observed = artifacts.sha256_file(path)
    if observed != _sha(expected_sha256, f"{label} SHA-256"):
        raise ProductionDriverError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )


def _reject_secrets(value: Any, label: str = "input") -> None:
    """Reject credential-shaped keys before they can enter configs/status."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or _SENSITIVE_RE.search(key):
                raise ProductionDriverError(f"{label} contains a credential-shaped key")
            _reject_secrets(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{label}[{index}]")
    elif isinstance(value, str) and _WANDB_SECRET_RE.search(value):
        raise ProductionDriverError(f"{label} contains a W&B credential")


def _atomic_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o600) -> None:
    _check_path_chain(path.parent, "atomic JSON parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_path_chain(path, "atomic JSON path")
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _publish_immutable(path: Path, document: Mapping[str, Any]) -> str:
    """Publish complete JSON create-only without exposing a partial final name.

    The deterministic hidden staging name is protected by an advisory lock on
    the parent directory.  A SIGKILL can therefore strand only that staging
    inode; the next publisher removes it without interpreting its bytes.  The
    final path is installed with ``link(2)``, which never overwrites an
    existing artifact.
    """

    _check_path_chain(path.parent, "immutable publication parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_path_chain(path, "immutable publication path")
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.parent, directory_flags)
    staging_name = f".{path.name}.staging"
    staging_created = False

    def existing_bytes() -> bytes:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ProductionDriverError(
                f"existing immutable artifact is unsafe: {path}"
            ) from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
                raise ProductionDriverError(
                    f"existing immutable artifact is unsafe: {path}"
                )
            if info.st_size != len(payload):
                raise ProductionDriverError(
                    f"existing immutable artifact conflicts: {path}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)

    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        try:
            staging_info = os.stat(
                staging_name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            staging_info = None
        if staging_info is not None:
            if not stat.S_ISREG(staging_info.st_mode):
                raise ProductionDriverError(
                    f"immutable publication staging artifact is unsafe: {path}"
                )
            os.unlink(staging_name, dir_fd=directory_fd)
            os.fsync(directory_fd)

        try:
            current = existing_bytes()
        except FileNotFoundError:
            current = None
        if current is not None:
            if current != payload:
                raise ProductionDriverError(
                    f"existing immutable artifact conflicts: {path}"
                )
            return expected_sha256

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(staging_name, flags, 0o400, dir_fd=directory_fd)
        staging_created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fchmod(output.fileno(), 0o444)
                os.fsync(output.fileno())
            try:
                os.link(
                    staging_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if existing_bytes() != payload:
                    raise ProductionDriverError(
                        f"existing immutable artifact conflicts: {path}"
                    ) from None
            os.fsync(directory_fd)
            return expected_sha256
        finally:
            if staging_created:
                try:
                    os.unlink(staging_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except FileNotFoundError:
                    pass
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


@dataclass(frozen=True)
class DriverInputs:
    campaign_root: Path
    campaign_contract: Path
    campaign_contract_sha256: str
    learner_contract: Path
    learner_contract_sha256: str
    sealed_source_root: Path
    sealed_source_inventory_sha256: str
    evaluation_sealed_source_root: Path
    evaluation_sealed_source_inventory_sha256: str
    launcher: Path
    launcher_sha256: str
    control_root: Path
    learner_backend: str = LEARNER_BACKEND_LEPTON_JOB_V1
    lepton_backend_contract: Path | None = None
    lepton_backend_contract_sha256: str | None = None
    lepton_remote_backend_contract: str | None = None
    lepton_control_exec_program: Path | None = None
    lepton_sdk_python: Path | None = None
    heartbeat_seconds: float = 60.0
    collection_retry_limit: int = 2
    collection_retry_backoff_seconds: float = 30.0

    def validate(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        try:
            validate_learner_backend(self.learner_backend)
        except Exception as exc:
            raise ProductionDriverError(f"invalid learner backend: {exc}") from exc
        lepton_values = (
            self.lepton_backend_contract,
            self.lepton_backend_contract_sha256,
            self.lepton_remote_backend_contract,
            self.lepton_control_exec_program,
            self.lepton_sdk_python,
        )
        if self.learner_backend == LEARNER_BACKEND_OCI_NRT:
            if any(value is not None for value in lepton_values):
                raise ProductionDriverError(
                    "OCI learner backend forbids Lepton-specific inputs"
                )
        elif any(value is None for value in lepton_values):
            raise ProductionDriverError(
                "Lepton learner backend requires its exact contract, remote "
                "contract path, control helper, and SDK Python"
            )
        if self.heartbeat_seconds <= 0:
            raise ProductionDriverError("heartbeat_seconds must be positive")
        if (
            type(self.collection_retry_limit) is not int
            or self.collection_retry_limit < 0
        ):
            raise ProductionDriverError("collection_retry_limit must be a non-negative integer")
        if self.collection_retry_backoff_seconds < 0:
            raise ProductionDriverError(
                "collection_retry_backoff_seconds must be non-negative"
            )
        _lstat_directory(self.campaign_root, "campaign root")
        if not self.evaluation_sealed_source_root.is_absolute():
            raise ProductionDriverError(
                "evaluation sealed source root must be absolute"
            )
        _lstat_directory(self.evaluation_sealed_source_root, "evaluation sealed source")
        if not self.sealed_source_root.is_absolute():
            raise ProductionDriverError("sealed source root must be absolute")
        for value, label in (
            (self.sealed_source_inventory_sha256, "sealed source inventory"),
            (
                self.evaluation_sealed_source_inventory_sha256,
                "evaluation sealed source inventory",
            ),
        ):
            _sha(value, label)
        _verify_file(
            self.campaign_contract,
            self.campaign_contract_sha256,
            "post-seal runtime campaign contract",
        )
        _verify_file(self.learner_contract, self.learner_contract_sha256, "learner contract")
        if self.learner_backend != LEARNER_BACKEND_OCI_NRT:
            assert self.lepton_backend_contract is not None
            assert self.lepton_backend_contract_sha256 is not None
            assert self.lepton_remote_backend_contract is not None
            assert self.lepton_control_exec_program is not None
            assert self.lepton_sdk_python is not None
            _verify_file(
                self.lepton_backend_contract,
                self.lepton_backend_contract_sha256,
                "Lepton backend contract",
            )
            remote_contract = self.lepton_remote_backend_contract
            if (
                not remote_contract.startswith(
                    "/workspace/projects/roborender/"
                )
                or "//" in remote_contract
                or any(part in {".", ".."} for part in Path(remote_contract).parts)
            ):
                raise ProductionDriverError(
                    "Lepton remote backend contract path is unsafe"
                )
            for path, label in (
                (self.lepton_control_exec_program, "Lepton control helper"),
                (self.lepton_sdk_python, "Lepton SDK Python"),
            ):
                if not path.is_absolute() or not os.access(path, os.X_OK):
                    raise ProductionDriverError(f"{label} is not executable")
        _verify_file(self.launcher, self.launcher_sha256, "production launcher")
        repository_root = Path(__file__).resolve().parents[1]
        expected_launcher = (
            repository_root / "scripts/round_dagger/launch_production_macro_round.sh"
        )
        expected_learner = self.campaign_root / "learner_contract.json"
        # The driver and actor hooks deliberately execute from the live RTX
        # checkout whose revisions are matched to the remote learner seal by
        # production_launch.  Evaluation consumes a separate immutable local
        # seal, which production_launch verifies by inventory before any
        # orchestration begins.  Requiring these roots to be identical makes
        # that intended three-root deployment impossible: a source tree cannot
        # be both the live Git checkout and the read-only evaluation seal.
        if self.launcher != expected_launcher:
            raise ProductionDriverError("launcher must belong to the driver source root")
        if self.learner_contract != expected_learner:
            raise ProductionDriverError(
                "learner contract must be the canonical campaign-owned "
                "learner_contract.json"
            )
        if stat.S_IMODE(os.lstat(self.learner_contract).st_mode) != 0o444:
            raise ProductionDriverError(
                "campaign-owned learner contract mode must be exactly 0444"
            )
        try:
            from scripts.round_dagger import production_launch

            learner_contract = production_launch.load_learner_contract(
                self.learner_contract
            )
            learner_contract_backend = production_launch.learner_contract_backend(
                learner_contract
            )
            campaign = state.load_campaign(self.campaign_root)
        except Exception as exc:
            raise ProductionDriverError(f"cannot validate immutable inputs: {exc}") from exc
        _reject_secrets(learner_contract, "learner contract")
        if (
            self.learner_backend == LEARNER_BACKEND_LEPTON_JOB_V1
            and learner_contract_backend is not None
        ) or (
            self.learner_backend == LEARNER_BACKEND_OCI_NRT
            and learner_contract_backend != LEARNER_BACKEND_OCI_NRT
        ):
            raise ProductionDriverError(
                "campaign learner contract backend differs from the selected "
                "learner backend"
            )
        if campaign.get("policy_config_sha256") != self.learner_contract_sha256:
            raise ProductionDriverError(
                "durable campaign policy_config_sha256 differs from the "
                "campaign-owned learner contract"
            )
        invariant_profile = (
            repository_root / "configs/round_dagger/production_campaign_256x512.json"
        )
        try:
            materialized_contract, runtime_evidence, _lineage = (
                production_launch.materialize_campaign_contract(
                    invariant_profile,
                    runtime_contract_path=self.campaign_contract,
                    runtime_contract_sha256=self.campaign_contract_sha256,
                )
            )
            contract_campaign_id = materialized_contract["campaign_id"]
            contract_round_count = materialized_contract["topology"][
                "macro_round_count"
            ]
            evaluator_sha = materialized_contract["evaluation"][
                "hydra_config_sha256"
            ]
        except Exception as exc:
            raise ProductionDriverError(
                f"cannot materialize the post-seal runtime campaign contract: {exc}"
            ) from exc
        _reject_secrets(runtime_evidence, "runtime campaign evidence")
        _reject_secrets(materialized_contract, "materialized campaign contract")
        if contract_campaign_id != campaign["campaign_id"]:
            raise ProductionDriverError("runtime contract belongs to another campaign")
        if contract_round_count != PRODUCTION_MACRO_ROUNDS:
            raise ProductionDriverError("runtime contract must bind exactly five macro-rounds")
        if campaign.get("macro_round_count") != PRODUCTION_MACRO_ROUNDS:
            raise ProductionDriverError("durable campaign must bind exactly five macro-rounds")
        _sha(evaluator_sha, "runtime evaluator config")
        try:
            state._check_path_chain(self.control_root)
        except Exception as exc:
            raise ProductionDriverError(f"unsafe driver control root: {exc}") from exc
        self.control_root.mkdir(parents=True, exist_ok=True)
        try:
            state._check_path_chain(self.control_root)
        except Exception as exc:
            raise ProductionDriverError(f"unsafe driver control root: {exc}") from exc
        _lstat_directory(self.control_root, "driver control root")
        _verify_file(
            self.learner_contract,
            self.learner_contract_sha256,
            "campaign-owned learner contract",
        )
        if stat.S_IMODE(os.lstat(self.learner_contract).st_mode) != 0o444:
            raise ProductionDriverError(
                "campaign-owned learner contract mode changed during validation"
            )
        return dict(materialized_contract), dict(learner_contract), campaign


class ExclusiveCampaignLock:
    """Process-lifetime non-blocking exclusive lock shared by every driver."""

    def __init__(self, campaign_root: Path):
        self.path = campaign_root / ".production_five_round_driver.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "ExclusiveCampaignLock":
        _check_path_chain(self.path, "driver lock path")
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ProductionDriverError(f"unsafe driver lock path: {self.path}")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ProductionDriverError(
                "another production driver holds the campaign lock"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


class ExclusiveLauncherExecutionLock:
    """Lock inherited by the launcher so it survives a driver SIGKILL.

    The campaign lock protects one live driver, but the launcher is a separate
    process and intentionally continues if its supervising driver disappears.
    Passing this locked descriptor through ``exec`` lets a replacement driver
    distinguish an actually live, not-yet-heartbeated orchestrator from an
    abandoned immutable attempt without relying on a mutable PID record.
    """

    def __init__(self, campaign_root: Path):
        self.path = campaign_root / ".production_five_round_launcher.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "ExclusiveLauncherExecutionLock":
        _check_path_chain(self.path, "launcher execution lock path")
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            info = None
        if info is not None and not stat.S_ISREG(info.st_mode):
            raise ProductionDriverError(
                f"unsafe launcher execution lock path: {self.path}"
            )
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ProductionDriverError(
                "a prior launcher/orchestrator is still alive; refusing duplicate "
                "orchestration"
            ) from exc
        self._descriptor = descriptor
        return self

    @property
    def descriptor(self) -> int:
        if self._descriptor is None:
            raise ProductionDriverError("launcher execution lock is not held")
        return self._descriptor

    def __exit__(self, *_: object) -> None:
        if self._descriptor is not None:
            # Do not issue LOCK_UN: the launcher inherits this same open-file
            # description. Closing the driver's copy releases the lock only
            # after the launcher has also exited; an explicit unlock would
            # incorrectly drop protection while an orphan child is still live.
            os.close(self._descriptor)
            self._descriptor = None


def _proc_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _refuse_live_orphan(status_path: Path) -> None:
    if not os.path.lexists(status_path):
        return
    try:
        previous = artifacts.load_strict_json(status_path)
    except Exception as exc:
        raise ProductionDriverError(f"cannot inspect previous driver status: {exc}") from exc
    if not isinstance(previous, Mapping) or previous.get("state") != "LAUNCHING":
        return
    pid = previous.get("launcher_pid")
    ticks = previous.get("launcher_start_ticks")
    if type(pid) is int and type(ticks) is int and _proc_start_ticks(pid) == ticks:
        raise ProductionDriverError(
            f"prior launcher process {pid} is still alive; refusing duplicate orchestration"
        )


def _snapshot(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        campaign = state.load_campaign(campaign_root)
        coordinator = state.load_state(campaign_root)
    except Exception as exc:
        raise ProductionDriverError(f"durable campaign state is invalid: {exc}") from exc
    if campaign.get("macro_round_count") != PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("durable campaign horizon is not five rounds")
    phase = coordinator.get("phase")
    next_round = coordinator.get("next_macro_round")
    if phase not in _PHASES or type(next_round) is not int:
        raise ProductionDriverError("coordinator phase/frontier is invalid")
    if not 0 <= next_round <= PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("coordinator frontier lies outside rounds 0..4")
    if next_round == PRODUCTION_MACRO_ROUNDS and phase != "READY":
        raise ProductionDriverError("completed campaign must be READY")
    expected_open = None if phase == "READY" else next_round
    if coordinator.get("open_macro_round") != expected_open:
        raise ProductionDriverError("coordinator open round is inconsistent with its phase")
    return campaign, coordinator


def _derive_learner_bundle_sha(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    macro_round: int,
    source_inventory_sha256: str,
    learner_contract_path: Path,
) -> str:
    # Import lazily so status inspection and unit tests do not import the heavy
    # production preflight module unless a round config is being built.
    from scripts.round_dagger import production_launch

    active = coordinator["active_policy"]
    checkpoint = campaign_root / active["checkpoint_path"]
    spec = Path(f"{checkpoint}.policy_spec.json")
    learner_contract = production_launch.load_learner_contract(learner_contract_path)
    try:
        return production_launch.derive_learner_bundle_sha256(
            base_policy_spec=spec,
            parent_checkpoint=checkpoint,
            expected_parent_checkpoint_sha256=active["checkpoint_sha256"],
            runtime_policy_sha256=active["runtime_policy_sha256"],
            campaign_id=campaign["campaign_id"],
            macro_round=macro_round,
            code_sha256=source_inventory_sha256,
            learner_contract=learner_contract,
        )
    except Exception as exc:
        raise ProductionDriverError(f"cannot derive per-round learner identity: {exc}") from exc


def _orchestration_document(
    inputs: DriverInputs,
    runtime_contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> dict[str, Any]:
    macro_round = coordinator["next_macro_round"]
    if not 0 <= macro_round < PRODUCTION_MACRO_ROUNDS:
        raise ProductionDriverError("cannot materialize a config outside rounds 0..4")
    active = coordinator["active_policy"]
    repository_root = Path(__file__).resolve().parents[1]
    backend_contract_sha256 = (
        inputs.learner_contract_sha256
        if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
        else inputs.lepton_backend_contract_sha256
    )
    if backend_contract_sha256 is None:
        raise ProductionDriverError("learner backend contract identity is missing")
    document = {
        "schema": ORCHESTRATION_SCHEMA,
        "advance_after_rejected_evaluation": True,
        "learner_backend": inputs.learner_backend,
        "learner_backend_contract_sha256": backend_contract_sha256,
        "campaign_root": str(inputs.campaign_root),
        "actor_root": str(inputs.campaign_root / "actors"),
        "macro_round": macro_round,
        "macro_round_count": PRODUCTION_MACRO_ROUNDS,
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "learner_code_sha256": inputs.sealed_source_inventory_sha256,
        "learner_config_sha256": _derive_learner_bundle_sha(
            campaign_root=inputs.campaign_root,
            campaign=campaign,
            coordinator=coordinator,
            macro_round=macro_round,
            source_inventory_sha256=inputs.sealed_source_inventory_sha256,
            learner_contract_path=inputs.learner_contract,
        ),
        "learner_model_lineage_sha256": active["checkpoint_sha256"],
        "evaluator_config_sha256": runtime_contract["evaluation"]["hydra_config_sha256"],
        "hooks": {
            "collect": {
                "argv": [
                    str(
                        repository_root
                        / "scripts/round_dagger/run_production_actor_cohort_hook.sh"
                    )
                ],
                "timeout_seconds": 14400,
            },
            "transfer": None,
            "ingest": None,
            "train": {
                "argv": [
                    str(
                        repository_root
                        / trusted_train_hook_relative(inputs.learner_backend)
                    )
                ],
                "timeout_seconds": 172800,
            },
            "evaluate": {
                "argv": [
                    str(
                        inputs.evaluation_sealed_source_root
                        / "scripts/round_dagger/run_production_paired_evaluation_hook.sh"
                    )
                ],
                "timeout_seconds": 28800,
            },
            "promote": None,
        },
    }
    _reject_secrets(document, "orchestration config")
    return document


def _next_attempt_path(control_root: Path, coordinator: Mapping[str, Any]) -> Path:
    active = coordinator["active_policy"]
    stem = (
        f"macro_round_{coordinator['next_macro_round']:06d}"
        f"_policy_{active['version']:06d}"
        f"_{active['checkpoint_sha256'][:12]}"
    )
    directory = control_root / "orchestration_configs"
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1_000_000):
        candidate = directory / f"{stem}_attempt_{attempt:06d}.json"
        if not os.path.lexists(candidate):
            return candidate
    raise ProductionDriverError("orchestration attempt namespace is exhausted")


def materialize_orchestration_config(
    inputs: DriverInputs,
    runtime_contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> tuple[Path, str, Path]:
    document = _orchestration_document(inputs, runtime_contract, campaign, coordinator)
    config_path = _next_attempt_path(inputs.control_root, coordinator)
    config_sha = _publish_immutable(config_path, document)
    try:
        loaded = MacroRoundConfig.load(config_path)
    except Exception as exc:
        raise ProductionDriverError(f"published orchestration config is invalid: {exc}") from exc
    if loaded.sha256 != artifacts.canonical_json_sha256(document):
        raise ProductionDriverError("orchestration canonical identity changed after publication")
    binding_path = config_path.with_suffix(".binding.json")
    binding = {
        "schema": DRIVER_BINDING_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_contract_path": str(inputs.campaign_contract),
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "learner_contract_path": str(inputs.learner_contract),
        "learner_contract_sha256": inputs.learner_contract_sha256,
        "learner_backend": inputs.learner_backend,
        "learner_backend_contract_path": str(
            inputs.learner_contract
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract
        ),
        "learner_backend_contract_sha256": (
            inputs.learner_contract_sha256
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract_sha256
        ),
        "lepton_remote_backend_contract": inputs.lepton_remote_backend_contract,
        "lepton_control_exec_program": (
            None
            if inputs.lepton_control_exec_program is None
            else str(inputs.lepton_control_exec_program)
        ),
        "lepton_sdk_python": (
            None if inputs.lepton_sdk_python is None else str(inputs.lepton_sdk_python)
        ),
        "launcher_path": str(inputs.launcher),
        "launcher_sha256": inputs.launcher_sha256,
        "macro_round": coordinator["next_macro_round"],
        "active_policy_version": coordinator["active_policy"]["version"],
        "active_policy_checkpoint_sha256": coordinator["active_policy"]["checkpoint_sha256"],
        "orchestration_config_path": str(config_path),
        "orchestration_config_sha256": config_sha,
    }
    _publish_immutable(binding_path, binding)
    return config_path, config_sha, binding_path


def _launcher_argv(
    inputs: DriverInputs,
    config_path: Path,
    *,
    dry_run: bool,
    stop_after_collected: bool = False,
) -> list[str]:
    if type(stop_after_collected) is not bool:
        raise ProductionDriverError("stop_after_collected must be a boolean")
    argv = [
        str(inputs.launcher),
        "--orchestration-config",
        str(config_path),
        "--campaign-contract",
        str(inputs.campaign_contract),
        "--campaign-contract-sha256",
        inputs.campaign_contract_sha256,
        "--learner-contract",
        str(inputs.learner_contract),
        "--learner-contract-sha256",
        inputs.learner_contract_sha256,
        "--learner-backend",
        inputs.learner_backend,
        "--learner-backend-contract",
        str(
            inputs.learner_contract
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract
        ),
        "--learner-backend-contract-sha256",
        (
            inputs.learner_contract_sha256
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else str(inputs.lepton_backend_contract_sha256)
        ),
        "--sealed-source-root",
        str(inputs.sealed_source_root),
        "--sealed-source-inventory-sha256",
        inputs.sealed_source_inventory_sha256,
        "--evaluation-sealed-source-root",
        str(inputs.evaluation_sealed_source_root),
        "--evaluation-sealed-source-inventory-sha256",
        inputs.evaluation_sealed_source_inventory_sha256,
    ]
    if inputs.learner_backend != LEARNER_BACKEND_OCI_NRT:
        assert inputs.lepton_remote_backend_contract is not None
        assert inputs.lepton_control_exec_program is not None
        assert inputs.lepton_sdk_python is not None
        argv.extend(
            [
                "--lepton-remote-backend-contract",
                inputs.lepton_remote_backend_contract,
                "--lepton-control-exec-program",
                str(inputs.lepton_control_exec_program),
                "--lepton-sdk-python",
                str(inputs.lepton_sdk_python),
            ]
        )
    if dry_run:
        argv.append("--dry-run")
    if stop_after_collected:
        argv.append("--stop-after-collected")
    if any(_SENSITIVE_RE.search(item) for item in argv):
        raise ProductionDriverError(
            "launcher argv unexpectedly contains a credential-shaped value"
        )
    return argv


def _collection_only_receipt(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the only successful terminal state for collection-only mode.

    A collection-only invocation deliberately does *not* advance the campaign
    to ``READY``: it stops at the durable, immutable ``COLLECTED`` boundary so
    the normal driver can later ingest/train/evaluate the exact same rollout.
    Treating that boundary as an ordinary successful round resolution would
    either falsely report a failure or risk continuing into H200 work that the
    operator explicitly deferred.
    """

    allowed_before_phases = {"READY", "COLLECTING", "COLLECTED"}
    if before.get("phase") not in allowed_before_phases:
        raise ProductionDriverError(
            "collection-only mode may start only from READY, COLLECTING, or "
            "COLLECTED"
        )
    macro_round = before.get("next_macro_round")
    if type(macro_round) is not int:
        raise ProductionDriverError("collection-only before-state macro round is invalid")
    if (
        after.get("phase") != "COLLECTED"
        or after.get("next_macro_round") != macro_round
        or after.get("open_macro_round") != macro_round
        or after.get("last_collected_macro_round") != macro_round
    ):
        raise ProductionDriverError(
            "collection-only launcher did not finish at the expected durable "
            "COLLECTED boundary"
        )
    if after.get("active_policy") != before.get("active_policy"):
        raise ProductionDriverError(
            "collection-only launcher unexpectedly changed the active policy"
        )
    if after.get("candidate_policy") != before.get("candidate_policy"):
        raise ProductionDriverError(
            "collection-only launcher unexpectedly changed the candidate policy"
        )

    collection_path = state.round_directory(campaign_root, macro_round) / "collection.json"
    try:
        collection = state.load_collection_manifest(
            campaign_root,
            collection_path,
            expected_macro_round=macro_round,
        )
    except Exception as exc:
        raise ProductionDriverError(
            "collection-only launcher did not leave a valid immutable collection"
        ) from exc
    expected_cohorts = campaign.get("cohort_count")
    expected_transitions = (
        campaign.get("target_environments") * campaign.get("steps_per_environment")
        if type(campaign.get("target_environments")) is int
        and type(campaign.get("steps_per_environment")) is int
        else None
    )
    if (
        collection.get("cohort_count") != expected_cohorts
        or collection.get("transition_count") != expected_transitions
    ):
        raise ProductionDriverError(
            "collection-only launcher did not seal the exact campaign topology"
        )
    active = after.get("active_policy")
    if not isinstance(active, Mapping) or collection.get("runtime_policy_sha256") != active.get(
        "runtime_policy_sha256"
    ):
        raise ProductionDriverError(
            "collection-only aggregate is not bound to the active policy runtime"
        )
    return {
        "collection_manifest_path": collection["manifest_path"],
        "collection_manifest_sha256": collection["manifest_sha256"],
        "cohort_count": collection["cohort_count"],
        "transition_count": collection["transition_count"],
    }


def _status_document(
    *,
    inputs: DriverInputs,
    coordinator: Mapping[str, Any],
    driver_state: str,
    config_path: Path | None = None,
    config_sha256: str | None = None,
    binding_path: Path | None = None,
    launcher_pid: int | None = None,
    launcher_start_ticks: int | None = None,
    detail: str | None = None,
    collection_retry_count: int = 0,
    collection_recovery: Mapping[str, Any] | None = None,
    collection_retry_ledger_path: Path | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": DRIVER_STATUS_SCHEMA,
        "updated_unix_ns": time.time_ns(),
        "driver_pid": os.getpid(),
        "state": driver_state,
        "campaign_root": str(inputs.campaign_root),
        "campaign_contract_path": str(inputs.campaign_contract),
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "learner_backend": inputs.learner_backend,
        "learner_backend_contract_path": str(
            inputs.learner_contract
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract
        ),
        "learner_backend_contract_sha256": (
            inputs.learner_contract_sha256
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract_sha256
        ),
        "lepton_remote_backend_contract": inputs.lepton_remote_backend_contract,
        "lepton_control_exec_program": (
            None
            if inputs.lepton_control_exec_program is None
            else str(inputs.lepton_control_exec_program)
        ),
        "lepton_sdk_python": (
            None if inputs.lepton_sdk_python is None else str(inputs.lepton_sdk_python)
        ),
        "coordinator_phase": coordinator["phase"],
        "next_macro_round": coordinator["next_macro_round"],
        "macro_round_count": PRODUCTION_MACRO_ROUNDS,
        "completed": (
            coordinator["phase"] == "READY"
            and coordinator["next_macro_round"] == PRODUCTION_MACRO_ROUNDS
        ),
        "active_policy_version": coordinator["active_policy"]["version"],
        "active_policy_checkpoint_sha256": coordinator["active_policy"]["checkpoint_sha256"],
        "orchestration_config_path": None if config_path is None else str(config_path),
        "orchestration_config_sha256": config_sha256,
        "binding_path": None if binding_path is None else str(binding_path),
        "launcher_pid": launcher_pid,
        "launcher_start_ticks": launcher_start_ticks,
        "collection_retry_count": collection_retry_count,
        "collection_retry_limit": inputs.collection_retry_limit,
        "collection_recovery": collection_recovery,
        "collection_retry_ledger_path": (
            None
            if collection_retry_ledger_path is None
            else str(collection_retry_ledger_path)
        ),
        "detail": detail,
    }
    _reject_secrets(document, "driver status")
    return document


def _retry_frontier(
    campaign: Mapping[str, Any], coordinator: Mapping[str, Any]
) -> dict[str, Any]:
    active = coordinator.get("active_policy")
    if not isinstance(active, Mapping):
        raise ProductionDriverError("retry frontier has no active policy")
    frontier = {
        "campaign_id": campaign.get("campaign_id"),
        "macro_round": coordinator.get("next_macro_round"),
        "policy_version": active.get("version"),
        "policy_checkpoint_sha256": active.get("checkpoint_sha256"),
    }
    if not isinstance(frontier["campaign_id"], str):
        raise ProductionDriverError("retry frontier campaign ID is invalid")
    if type(frontier["macro_round"]) is not int or type(frontier["policy_version"]) is not int:
        raise ProductionDriverError("retry frontier indices are invalid")
    _sha(frontier["policy_checkpoint_sha256"], "retry frontier policy checkpoint")
    return frontier


def _retry_ledger_directory(inputs: DriverInputs, *, create: bool) -> Path:
    directory = inputs.control_root / "collection_attempt_failures"
    _check_path_chain(directory, "collection retry ledger")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        _check_path_chain(directory, "collection retry ledger")
    if directory.exists():
        _lstat_directory(directory, "collection retry ledger")
    return directory


def _attempt_ledger_directory(inputs: DriverInputs, *, create: bool) -> Path:
    directory = inputs.control_root / "collection_launcher_attempts"
    _check_path_chain(directory, "collection attempt ledger")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        _check_path_chain(directory, "collection attempt ledger")
    if directory.exists():
        _lstat_directory(directory, "collection attempt ledger")
    return directory


def _prune_ledger_staging(
    directory: Path,
    *,
    pattern: re.Pattern[str],
    label: str,
) -> list[str]:
    """Remove only exact regular staging inodes left by our publisher.

    The immutable publisher and this recovery path take the same advisory lock
    on the ledger directory. Hidden names that do not match the complete
    canonical final-record grammar are not removed and remain visible to the
    strict loader. A matching symlink or non-regular inode is ambiguous and
    therefore rejected rather than followed or deleted.
    """

    _check_path_chain(directory, label)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(directory, directory_flags)
    removed: list[str] = []
    locked = False
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        locked = True
        for name in sorted(os.listdir(directory_fd)):
            if pattern.fullmatch(name) is None:
                continue
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ProductionDriverError(
                    f"cannot inspect {label} staging artifact"
                ) from exc
            if not stat.S_ISREG(info.st_mode):
                raise ProductionDriverError(f"{label} staging artifact is unsafe")
            os.unlink(name, dir_fd=directory_fd)
            removed.append(name)
        if removed:
            os.fsync(directory_fd)
    finally:
        if locked:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
    return removed


def _load_collection_attempts(
    *,
    inputs: DriverInputs,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directory = _attempt_ledger_directory(inputs, create=False)
    if not directory.exists():
        return []
    _prune_ledger_staging(
        directory,
        pattern=_ATTEMPT_STAGING_RE,
        label="collection attempt ledger",
    )
    frontier = _retry_frontier(campaign, coordinator)
    matching: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        _lstat_regular(path, "collection launcher attempt", immutable=True)
        try:
            document = artifacts.load_strict_json(path)
        except Exception as exc:
            raise ProductionDriverError(
                "cannot load immutable collection launcher attempt"
            ) from exc
        expected_keys = {
            "schema",
            "campaign_id",
            "campaign_contract_sha256",
            "macro_round",
            "policy_version",
            "policy_checkpoint_sha256",
            "attempt_index",
            "orchestration_config_path",
            "orchestration_config_sha256",
        }
        if (
            not isinstance(document, Mapping)
            or set(document) != expected_keys
            or document.get("schema") != COLLECTION_ATTEMPT_SCHEMA
        ):
            raise ProductionDriverError("collection launcher attempt schema is invalid")
        _reject_secrets(document, "collection launcher attempt")
        for field in (
            "campaign_contract_sha256",
            "policy_checkpoint_sha256",
            "orchestration_config_sha256",
        ):
            _sha(document.get(field), f"collection launcher attempt {field}")
        if document["campaign_contract_sha256"] != inputs.campaign_contract_sha256:
            raise ProductionDriverError(
                "collection launcher attempt belongs to another campaign contract"
            )
        attempt_index = document.get("attempt_index")
        if type(attempt_index) is not int or attempt_index < 0:
            raise ProductionDriverError("collection launcher attempt index is invalid")
        if (
            not isinstance(document.get("campaign_id"), str)
            or type(document.get("macro_round")) is not int
            or document["macro_round"] < 0
            or type(document.get("policy_version")) is not int
            or document["policy_version"] < 0
        ):
            raise ProductionDriverError("collection launcher attempt frontier is invalid")
        config_path = document.get("orchestration_config_path")
        if not isinstance(config_path, str) or not Path(config_path).is_absolute():
            raise ProductionDriverError("collection launcher config path is invalid")
        expected_name = (
            f"macro_round_{document['macro_round']:06d}"
            f"_policy_{document['policy_version']:06d}"
            f"_{document['policy_checkpoint_sha256'][:12]}"
            f"_attempt_{attempt_index:06d}.json"
        )
        if path.name != expected_name:
            raise ProductionDriverError("collection launcher attempt filename is not canonical")
        if all(document.get(key) == value for key, value in frontier.items()):
            matching.append(dict(document))
    indices = [document["attempt_index"] for document in matching]
    if indices != list(range(len(indices))):
        raise ProductionDriverError(
            "collection launcher attempts are not a contiguous immutable ledger"
        )
    return matching


def _publish_collection_attempt(
    *,
    inputs: DriverInputs,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    attempt_index: int,
    config_path: Path,
    config_sha256: str,
) -> Path:
    if type(attempt_index) is not int or attempt_index < 0:
        raise ProductionDriverError("collection launcher attempt index is invalid")
    frontier = _retry_frontier(campaign, coordinator)
    directory = _attempt_ledger_directory(inputs, create=True)
    path = directory / (
        f"macro_round_{frontier['macro_round']:06d}"
        f"_policy_{frontier['policy_version']:06d}"
        f"_{frontier['policy_checkpoint_sha256'][:12]}"
        f"_attempt_{attempt_index:06d}.json"
    )
    document = {
        "schema": COLLECTION_ATTEMPT_SCHEMA,
        **frontier,
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "attempt_index": attempt_index,
        "orchestration_config_path": str(config_path),
        "orchestration_config_sha256": config_sha256,
    }
    _reject_secrets(document, "collection launcher attempt")
    _publish_immutable(path, document)
    return path


def _load_collection_failures(
    *,
    inputs: DriverInputs,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load immutable failed launcher attempts at this exact frontier."""

    directory = _retry_ledger_directory(inputs, create=False)
    if not directory.exists():
        return []
    _prune_ledger_staging(
        directory,
        pattern=_FAILURE_STAGING_RE,
        label="collection failure ledger",
    )
    frontier = _retry_frontier(campaign, coordinator)
    matching: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        _lstat_regular(path, "collection attempt failure", immutable=True)
        try:
            document = artifacts.load_strict_json(path)
        except Exception as exc:
            raise ProductionDriverError(
                "cannot load immutable collection attempt failure"
            ) from exc
        if not isinstance(document, Mapping):
            raise ProductionDriverError("collection attempt failure is not an object")
        expected_keys = {
            "schema",
            "campaign_id",
            "campaign_contract_sha256",
            "macro_round",
            "policy_version",
            "policy_checkpoint_sha256",
            "failure_index",
            "retry_attempt_number",
            "failed_orchestration_config_path",
            "failed_orchestration_config_sha256",
            "launcher_return_code",
            "operator_cancelled",
            "recovery",
        }
        if (
            set(document) != expected_keys
            or document.get("schema") != COLLECTION_FAILURE_SCHEMA
        ):
            raise ProductionDriverError("collection attempt failure schema is invalid")
        _reject_secrets(document, "collection attempt failure")
        _sha(
            document.get("campaign_contract_sha256"),
            "retry authorization campaign contract",
        )
        _sha(
            document.get("policy_checkpoint_sha256"),
            "retry authorization policy checkpoint",
        )
        _sha(
            document.get("failed_orchestration_config_sha256"),
            "retry authorization orchestration config",
        )
        if document.get("campaign_contract_sha256") != inputs.campaign_contract_sha256:
            raise ProductionDriverError(
                "collection failure belongs to another campaign contract"
            )
        failure_index = document.get("failure_index")
        retry_attempt_number = document.get("retry_attempt_number")
        return_code = document.get("launcher_return_code")
        if (
            not isinstance(document.get("campaign_id"), str)
            or type(document.get("macro_round")) is not int
            or document["macro_round"] < 0
            or type(document.get("policy_version")) is not int
            or document["policy_version"] < 0
        ):
            raise ProductionDriverError("collection failure frontier is invalid")
        if type(failure_index) is not int or failure_index < 1:
            raise ProductionDriverError("collection failure index is invalid")
        if (
            type(retry_attempt_number) is not int
            or retry_attempt_number != failure_index - 1
        ):
            raise ProductionDriverError("collection retry attempt number is invalid")
        if type(return_code) is not int or return_code == 0:
            raise ProductionDriverError("collection failure return code is invalid")
        operator_cancelled = document.get("operator_cancelled")
        recovery = document.get("recovery")
        if type(operator_cancelled) is not bool:
            raise ProductionDriverError("collection cancellation flag is invalid")
        if recovery is not None and (
            not isinstance(recovery, Mapping)
            or recovery.get("safe_to_retry") is not True
        ):
            raise ProductionDriverError("collection recovery proof is invalid")
        if operator_cancelled and recovery is not None:
            raise ProductionDriverError("operator cancellation cannot carry retry proof")
        expected_name = (
            f"macro_round_{document['macro_round']:06d}"
            f"_policy_{document['policy_version']:06d}"
            f"_{document['policy_checkpoint_sha256'][:12]}"
            f"_failure_{failure_index:06d}.json"
        )
        if path.name != expected_name:
            raise ProductionDriverError("collection failure filename is not canonical")
        config_path = document.get("failed_orchestration_config_path")
        if not isinstance(config_path, str) or not Path(config_path).is_absolute():
            raise ProductionDriverError("collection failure config path is invalid")
        if all(document.get(key) == value for key, value in frontier.items()):
            matching.append(dict(document))
    indices = [document["failure_index"] for document in matching]
    if indices != list(range(1, len(indices) + 1)):
        raise ProductionDriverError(
            "collection attempt failures are not a contiguous immutable ledger"
        )
    return matching


def _collection_progress_signature(
    recovery: Mapping[str, Any] | None,
) -> tuple[int, ...] | None:
    """Project recovery evidence onto a monotonic durable-data frontier.

    Retry limits protect against repeatedly failing at the *same* collection
    frontier.  They must not permanently consume the whole round's budget when
    an attempt has meanwhile sealed a cohort (or durably completed an actor
    rank).  Older/minimal recovery documents do not carry cohort detail; those
    conservatively return ``None`` and retain the historical bounded behavior.
    """

    if not isinstance(recovery, Mapping) or recovery.get("safe_to_retry") is not True:
        return None
    cohorts = recovery.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        return None
    signature: list[int] = []
    for expected_index, cohort in enumerate(cohorts):
        if (
            not isinstance(cohort, Mapping)
            or cohort.get("cohort_index") != expected_index
        ):
            return None
        state = cohort.get("state")
        if state == "absent":
            signature.append(0)
            continue
        if state == "sealed":
            signature.append(3)
            continue
        if state != "unsealed":
            return None
        rank_states = cohort.get("rank_states")
        if not isinstance(rank_states, Mapping) or set(rank_states) != {"0", "1"}:
            return None
        if any(value not in {"empty", "resumable", "complete"} for value in rank_states.values()):
            return None
        # Partial/resumable shards remain useful for the actor's own restart,
        # but only an immutable rank completion advances the global retry
        # budget.  The joint seal is the final stage above.
        signature.append(sum(value == "complete" for value in rank_states.values()))
    return tuple(signature)


def _collection_retry_streak(
    failures: Sequence[Mapping[str, Any]],
) -> int:
    """Count consecutive failures since the last durable frontier advance."""

    streak = 0
    previous_signature: tuple[int, ...] | None = None
    for failure in failures:
        signature = _collection_progress_signature(failure.get("recovery"))
        progressed = (
            signature is not None
            and previous_signature is not None
            and len(signature) == len(previous_signature)
            and all(new >= old for new, old in zip(signature, previous_signature))
            and any(new > old for new, old in zip(signature, previous_signature))
        )
        streak = 1 if progressed else streak + 1
        if signature is not None:
            previous_signature = signature
    return streak


def _publish_collection_failure(
    *,
    inputs: DriverInputs,
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    failure_index: int,
    config_path: Path,
    config_sha256: str,
    launcher_return_code: int,
    operator_cancelled: bool,
    recovery: Mapping[str, Any] | None,
) -> Path:
    if type(failure_index) is not int or failure_index < 1:
        raise ProductionDriverError("collection failure index must be positive")
    if type(launcher_return_code) is not int or launcher_return_code == 0:
        raise ProductionDriverError("collection failure return code must be nonzero")
    if operator_cancelled and recovery is not None:
        raise ProductionDriverError("operator cancellation cannot be retryable")
    if recovery is not None and recovery.get("safe_to_retry") is not True:
        raise ProductionDriverError("collection failure lacks a safe recovery proof")
    frontier = _retry_frontier(campaign, coordinator)
    directory = _retry_ledger_directory(inputs, create=True)
    stem = (
        f"macro_round_{frontier['macro_round']:06d}"
        f"_policy_{frontier['policy_version']:06d}"
        f"_{frontier['policy_checkpoint_sha256'][:12]}"
        f"_failure_{failure_index:06d}.json"
    )
    path = directory / stem
    document = {
        "schema": COLLECTION_FAILURE_SCHEMA,
        **frontier,
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "failure_index": failure_index,
        "retry_attempt_number": failure_index - 1,
        "failed_orchestration_config_path": str(config_path),
        "failed_orchestration_config_sha256": config_sha256,
        "launcher_return_code": launcher_return_code,
        "operator_cancelled": operator_cancelled,
        "recovery": None if recovery is None else dict(recovery),
    }
    _reject_secrets(document, "collection attempt failure")
    _publish_immutable(path, document)
    return path


def _is_operator_cancellation(return_code: int) -> bool:
    """Conservatively identify outcomes that must never be auto-retried."""

    return (
        return_code < 0
        or return_code in _OPERATOR_SIGNAL_RETURN_CODES
        or 129 <= return_code <= 192
    )


def _recoverable_collection_evidence(
    *,
    inputs: DriverInputs,
    campaign: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Prove that rerunning the idempotent orchestrator is phase-safe.

    READY at the same frontier means the round was never opened.  COLLECTING
    additionally requires a strict actor-filesystem proof because collection
    may contain unsealed work.  COLLECTED through EVALUATED are durable state
    machine checkpoints whose loaders revalidate immutable evidence before the
    orchestrator continues.  Forward progress within the same round is safe;
    policy/frontier changes and phase regression fail closed.
    """

    before_phase = before.get("phase")
    after_phase = after.get("phase")
    phase_order = {
        "READY": 0,
        "COLLECTING": 1,
        "COLLECTED": 2,
        "INGESTED": 3,
        "TRAINED": 4,
        "EVALUATED": 5,
    }
    if (
        before_phase not in phase_order
        or after_phase not in phase_order
        or after.get("next_macro_round") != before.get("next_macro_round")
        or after.get("active_policy") != before.get("active_policy")
        or phase_order[after_phase] < phase_order[before_phase]
        or (before_phase != "READY" and after_phase == "READY")
    ):
        return None

    macro_round = before.get("next_macro_round")
    if type(macro_round) is not int:
        return None
    expected_before_open = None if before_phase == "READY" else macro_round
    expected_after_open = None if after_phase == "READY" else macro_round
    if (
        before.get("open_macro_round") != expected_before_open
        or after.get("open_macro_round") != expected_after_open
    ):
        return None

    base = {
        "schema": "fast_roborender.round_dagger.launcher_recovery.v1",
        "macro_round": macro_round,
        "from_phase": before_phase,
        "resume_phase": after_phase,
        "safe_to_retry": True,
    }
    if after_phase == "READY":
        return {**base, "resume_mode": "round_not_opened", "cohorts": []}
    if after_phase in {"COLLECTED", "INGESTED", "TRAINED", "EVALUATED"}:
        return {**base, "resume_mode": "durable_phase", "cohorts": []}

    # Imported lazily so the driver remains lightweight for status-only paths.
    from round_dagger.actor_pair import (
        ActorPairError,
        inspect_actor_pair_for_recovery,
    )

    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        return None
    actor_round = (
        inputs.campaign_root
        / "actors"
        / campaign_id
        / f"macro_round_{macro_round:06d}"
    )
    cohorts: list[dict[str, Any]] = []
    encountered_gap = False
    unsealed_count = 0
    for cohort_index in range(4):
        cohort = actor_round / f"cohort_{cohort_index:02d}"
        if not os.path.lexists(cohort):
            encountered_gap = True
            cohorts.append({"cohort_index": cohort_index, "state": "absent"})
            continue
        try:
            report = inspect_actor_pair_for_recovery(cohort)
        except (ActorPairError, OSError, ValueError):
            return None
        if report["sealed"]:
            # Collection is strictly sequential.  A seal after an absent or
            # unfinished cohort means the filesystem frontier is ambiguous.
            if encountered_gap or unsealed_count:
                return None
            cohorts.append({"cohort_index": cohort_index, "state": "sealed"})
            continue
        states = report.get("rank_states")
        if not isinstance(states, Mapping) or set(states.values()) - {
            "empty",
            "complete",
            "resumable",
        }:
            return None
        if encountered_gap or unsealed_count:
            return None
        unsealed_count += 1
        encountered_gap = True
        cohorts.append(
            {
                "cohort_index": cohort_index,
                "state": "unsealed",
                "rank_states": dict(states),
                "requires_temp_cleanup": bool(
                    report.get("requires_temp_cleanup", False)
                ),
            }
        )

    return {
        **base,
        "resume_mode": "collection_frontier",
        "cohorts": cohorts,
    }


def _reconcile_unmatched_attempt(
    *,
    inputs: DriverInputs,
    runtime_contract: Mapping[str, Any],
    campaign: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> tuple[Path, str, Path, dict[str, Any]]:
    """Validate and resume one attempt whose launcher result was not recorded."""

    frontier = _retry_frontier(campaign, coordinator)
    if any(attempt.get(key) != value for key, value in frontier.items()):
        raise ProductionDriverError(
            "unmatched launcher attempt is not bound to the active frontier"
        )
    recovery = _recoverable_collection_evidence(
        inputs=inputs,
        campaign=campaign,
        before=coordinator,
        after=coordinator,
    )
    if recovery is None:
        raise ProductionDriverError(
            "unmatched launcher attempt has no phase-safe resume proof"
        )

    raw_config_path = attempt.get("orchestration_config_path")
    if not isinstance(raw_config_path, str) or not Path(raw_config_path).is_absolute():
        raise ProductionDriverError("unmatched launcher config path is invalid")
    config_path = Path(os.path.abspath(raw_config_path))
    if str(config_path) != raw_config_path:
        raise ProductionDriverError("unmatched launcher config path is not canonical")
    expected_directory = Path(
        os.path.abspath(inputs.control_root / "orchestration_configs")
    )
    if config_path.parent != expected_directory:
        raise ProductionDriverError(
            "unmatched launcher config lies outside the immutable config namespace"
        )
    expected_stem = (
        f"macro_round_{frontier['macro_round']:06d}"
        f"_policy_{frontier['policy_version']:06d}"
        f"_{frontier['policy_checkpoint_sha256'][:12]}"
    )
    match = re.fullmatch(
        rf"{re.escape(expected_stem)}_attempt_([0-9]{{6}})\.json",
        config_path.name,
    )
    if match is None or int(match.group(1)) < 1:
        raise ProductionDriverError(
            "unmatched launcher config filename is not canonical"
        )

    config_sha = attempt.get("orchestration_config_sha256")
    _sha(config_sha, "unmatched launcher orchestration config")
    _verify_file(config_path, config_sha, "unmatched launcher orchestration config")
    try:
        observed_document = artifacts.load_strict_json(config_path)
        MacroRoundConfig.load(config_path)
    except Exception as exc:
        raise ProductionDriverError(
            "unmatched launcher orchestration config is invalid"
        ) from exc
    expected_document = _orchestration_document(
        inputs, runtime_contract, campaign, coordinator
    )
    if observed_document != expected_document:
        raise ProductionDriverError(
            "unmatched launcher config no longer matches the exact active frontier"
        )

    binding_path = config_path.with_suffix(".binding.json")
    _lstat_regular(binding_path, "unmatched launcher binding", immutable=True)
    try:
        binding = artifacts.load_strict_json(binding_path)
    except Exception as exc:
        raise ProductionDriverError("unmatched launcher binding is invalid") from exc
    expected_binding = {
        "schema": DRIVER_BINDING_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_contract_path": str(inputs.campaign_contract),
        "campaign_contract_sha256": inputs.campaign_contract_sha256,
        "learner_contract_path": str(inputs.learner_contract),
        "learner_contract_sha256": inputs.learner_contract_sha256,
        "learner_backend": inputs.learner_backend,
        "learner_backend_contract_path": str(
            inputs.learner_contract
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract
        ),
        "learner_backend_contract_sha256": (
            inputs.learner_contract_sha256
            if inputs.learner_backend == LEARNER_BACKEND_OCI_NRT
            else inputs.lepton_backend_contract_sha256
        ),
        "lepton_remote_backend_contract": inputs.lepton_remote_backend_contract,
        "lepton_control_exec_program": (
            None
            if inputs.lepton_control_exec_program is None
            else str(inputs.lepton_control_exec_program)
        ),
        "lepton_sdk_python": (
            None if inputs.lepton_sdk_python is None else str(inputs.lepton_sdk_python)
        ),
        "launcher_path": str(inputs.launcher),
        "launcher_sha256": inputs.launcher_sha256,
        "macro_round": frontier["macro_round"],
        "active_policy_version": frontier["policy_version"],
        "active_policy_checkpoint_sha256": frontier[
            "policy_checkpoint_sha256"
        ],
        "orchestration_config_path": str(config_path),
        "orchestration_config_sha256": config_sha,
    }
    if binding != expected_binding:
        raise ProductionDriverError(
            "unmatched launcher binding no longer matches the exact active frontier"
        )
    return config_path, config_sha, binding_path, recovery


def _round_was_resolved(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    before_policy = before.get("active_policy")
    after_policy = after.get("active_policy")
    before_round = before.get("next_macro_round")
    after_round = after.get("next_macro_round")
    before_version = (
        before_policy.get("version") if isinstance(before_policy, Mapping) else None
    )
    after_version = (
        after_policy.get("version") if isinstance(after_policy, Mapping) else None
    )
    advanced = bool(
        isinstance(before_policy, Mapping)
        and isinstance(after_policy, Mapping)
        and type(before_round) is int
        and type(after_round) is int
        and type(before_version) is int
        and type(after_version) is int
        and after.get("phase") == "READY"
        and after.get("open_macro_round") is None
        and after_round == before_round + 1
    )
    if not advanced:
        return False
    resolution = after.get("last_resolution")
    if resolution is None:
        # Compatibility for a V2/V4 campaign completed before terminal
        # resolution receipts were introduced.
        return after_version == before_version + 1
    if not isinstance(resolution, Mapping):
        return False
    decision = resolution.get("decision")
    if (
        resolution.get("macro_round") != before_round
        or not isinstance(resolution.get("candidate_policy_version"), int)
        or not isinstance(resolution.get("candidate_manifest_sha256"), str)
        or not isinstance(resolution.get("evaluation_sha256"), str)
    ):
        return False
    candidate = after.get("candidate_policy")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("version") != resolution["candidate_policy_version"]
        or candidate.get("manifest_sha256")
        != resolution["candidate_manifest_sha256"]
    ):
        return False
    if decision == "promoted":
        return bool(
            after_version == resolution["candidate_policy_version"]
            and after_policy.get("checkpoint_sha256")
            == candidate.get("checkpoint_sha256")
        )
    if decision == "rejected":
        return dict(after_policy) == dict(before_policy)
    return False


# Kept as a private compatibility spelling for older diagnostics.  New driver
# paths intentionally accept either a promoted candidate or an attested parent
# retention as the successful terminal state of one macro-round.
_round_was_promoted = _round_was_resolved


def _invoke_launcher(
    argv: Sequence[str],
    *,
    log_path: Path,
    heartbeat_seconds: float,
    on_heartbeat: Callable[[int, int | None], None],
    execution_lock_fd: int | None = None,
) -> int:
    forwarded_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded_signals)
    previous_handlers: dict[int, Any] = {}
    stop_signal: list[int] = []
    child: subprocess.Popen[bytes] | None = None
    signals_unblocked = False

    def forward(signum: int, _frame: object) -> None:
        stop_signal.append(signum)
        if child is None:
            return
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    def restore_child_signal_mask() -> None:
        # The pre-fork block closes the parent race, but a forked child inherits
        # that mask. Restore the caller's original mask before exec so forwarded
        # termination signals are actionable in the launcher process group.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for signum in forwarded_signals:
            previous_handlers[signum] = signal.signal(signum, forward)
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                preexec_fn=restore_child_signal_mask,
                pass_fds=(
                    () if execution_lock_fd is None else (execution_lock_fd,)
                ),
            )
            start_ticks = _proc_start_ticks(child.pid)
            # Handlers are active and the child process group exists before any
            # pending operator signal is unblocked. This closes the Popen ->
            # first-heartbeat window in which a driver-only SIGTERM could
            # otherwise orphan a still-running orchestrator.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            signals_unblocked = True
            on_heartbeat(child.pid, start_ticks)
            while True:
                try:
                    return_code = child.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    on_heartbeat(child.pid, start_ticks)
            if stop_signal:
                return 128 + stop_signal[-1]
            return return_code
    finally:
        if signals_unblocked:
            cleanup_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, forwarded_signals
            )
        else:
            cleanup_mask = previous_mask
        try:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)


def run_driver(
    inputs: DriverInputs,
    *,
    dry_run: bool = False,
    stop_after_collected: bool = False,
    launcher_runner: Callable[..., int] = _invoke_launcher,
) -> dict[str, Any]:
    """Run/resume production rounds, or seal only the current collection.

    ``stop_after_collected`` is intentionally narrow: it may finish the
    currently open macro-round only at its immutable collection boundary.  It
    never trains, evaluates, promotes, or advances to another macro-round.
    """

    if type(stop_after_collected) is not bool:
        raise ProductionDriverError("stop_after_collected must be a boolean")

    runtime_contract, _learner_contract, _campaign = inputs.validate()
    status_path = inputs.control_root / "driver_status.json"
    with (
        ExclusiveCampaignLock(inputs.campaign_root),
        ExclusiveLauncherExecutionLock(inputs.campaign_root) as execution_lock,
    ):
        _refuse_live_orphan(status_path)
        while True:
            campaign, coordinator = _snapshot(inputs.campaign_root)
            if coordinator["next_macro_round"] == PRODUCTION_MACRO_ROUNDS:
                if stop_after_collected:
                    raise ProductionDriverError(
                        "collection-only mode cannot run after all macro-rounds complete"
                    )
                result = _status_document(
                    inputs=inputs,
                    coordinator=coordinator,
                    driver_state="COMPLETE",
                    detail="all five macro-rounds are durably promoted",
                )
                _atomic_json(status_path, result)
                return result

            if stop_after_collected and coordinator["phase"] == "COLLECTED":
                receipt = _collection_only_receipt(
                    campaign_root=inputs.campaign_root,
                    campaign=campaign,
                    before=coordinator,
                    after=coordinator,
                )
                completed = _status_document(
                    inputs=inputs,
                    coordinator=coordinator,
                    driver_state="COLLECTION_ONLY_COMPLETE",
                    detail=(
                        "the exact immutable collection is already sealed; "
                        "no learner, evaluation, or promotion hook was launched"
                    ),
                )
                completed.update(receipt)
                _atomic_json(status_path, completed)
                return completed
            if stop_after_collected and coordinator["phase"] not in {
                "READY",
                "COLLECTING",
            }:
                raise ProductionDriverError(
                    "collection-only mode refuses a macro-round that has already "
                    "advanced beyond collection"
                )

            collection_failures = _load_collection_failures(
                inputs=inputs,
                campaign=campaign,
                coordinator=coordinator,
            )
            collection_attempts = _load_collection_attempts(
                inputs=inputs,
                campaign=campaign,
                coordinator=coordinator,
            )
            if len(collection_attempts) < len(collection_failures) or len(
                collection_attempts
            ) > len(collection_failures) + 1:
                raise ProductionDriverError(
                    "collection attempt/failure ledgers have an impossible frontier"
                )
            for index, failure in enumerate(collection_failures):
                attempt = collection_attempts[index]
                if (
                    failure["retry_attempt_number"] != attempt["attempt_index"]
                    or failure["failed_orchestration_config_path"]
                    != attempt["orchestration_config_path"]
                    or failure["failed_orchestration_config_sha256"]
                    != attempt["orchestration_config_sha256"]
                ):
                    raise ProductionDriverError(
                        "collection attempt/failure ledgers disagree"
                    )
            unmatched_attempt = (
                collection_attempts[-1]
                if len(collection_attempts) == len(collection_failures) + 1
                else None
            )
            failure_count = len(collection_failures)
            retry_streak = _collection_retry_streak(collection_failures)
            retry_count = min(retry_streak, inputs.collection_retry_limit)
            latest_failure_path = None
            if collection_failures:
                frontier = _retry_frontier(campaign, coordinator)
                latest_failure_path = _retry_ledger_directory(
                    inputs, create=False
                ) / (
                    f"macro_round_{frontier['macro_round']:06d}"
                    f"_policy_{frontier['policy_version']:06d}"
                    f"_{frontier['policy_checkpoint_sha256'][:12]}"
                    f"_failure_{failure_count:06d}.json"
                )
                last_failure = collection_failures[-1]
                fresh_recovery = None
                if (
                    not last_failure["operator_cancelled"]
                    and last_failure["recovery"] is not None
                ):
                    fresh_recovery = _recoverable_collection_evidence(
                        inputs=inputs,
                        campaign=campaign,
                        before=coordinator,
                        after=coordinator,
                    )
                # An unmatched later attempt is already a durable record that
                # this bounded retry was authorized before the driver died.
                # Reconcile that exact attempt instead of re-authorizing it
                # from an older failure at a now-more-advanced phase.
                if unmatched_attempt is None and (
                    last_failure["operator_cancelled"]
                    or last_failure["recovery"] is None
                    or fresh_recovery is None
                    or retry_streak > inputs.collection_retry_limit
                ):
                    failed = _status_document(
                        inputs=inputs,
                        coordinator=coordinator,
                        driver_state="FAILED",
                        collection_retry_count=retry_count,
                        collection_recovery=last_failure["recovery"],
                        collection_retry_ledger_path=latest_failure_path,
                        detail=(
                            "previous launcher attempt was operator-cancelled; "
                            "automatic restart is forbidden"
                            if last_failure["operator_cancelled"]
                            else "previous launcher failure had no safe recovery proof"
                            if last_failure["recovery"] is None
                            else "current collection frontier no longer satisfies the "
                            "recorded safe-recovery contract"
                            if fresh_recovery is None
                            else f"bounded collection retry limit "
                            f"{inputs.collection_retry_limit} was already exhausted"
                        ),
                    )
                    _atomic_json(status_path, failed)
                    raise ProductionDriverError(
                        "immutable collection failure ledger forbids another launch"
                    )

            resume_recovery = None
            if unmatched_attempt is not None:
                (
                    config_path,
                    config_sha,
                    binding_path,
                    resume_recovery,
                ) = _reconcile_unmatched_attempt(
                    inputs=inputs,
                    runtime_contract=runtime_contract,
                    campaign=campaign,
                    coordinator=coordinator,
                    attempt=unmatched_attempt,
                )
            else:
                config_path, config_sha, binding_path = (
                    materialize_orchestration_config(
                        inputs, runtime_contract, campaign, coordinator
                    )
                )
            if not dry_run and unmatched_attempt is None:
                _publish_collection_attempt(
                    inputs=inputs,
                    campaign=campaign,
                    coordinator=coordinator,
                    attempt_index=failure_count,
                    config_path=config_path,
                    config_sha256=config_sha,
                )
            log_path = config_path.with_suffix(".launcher.log")

            def heartbeat(pid: int, ticks: int | None) -> None:
                # Re-read only for reporting.  The launcher/orchestrator owns all
                # state transitions while it is alive.
                _, current = _snapshot(inputs.campaign_root)
                _atomic_json(
                    status_path,
                    _status_document(
                        inputs=inputs,
                        coordinator=current,
                        driver_state="LAUNCHING",
                        config_path=config_path,
                        config_sha256=config_sha,
                        binding_path=binding_path,
                        launcher_pid=pid,
                        launcher_start_ticks=ticks,
                        collection_retry_count=retry_count,
                        collection_retry_ledger_path=latest_failure_path,
                        collection_recovery=resume_recovery,
                        detail=(
                            f"resuming unmatched launcher attempt from durable "
                            f"{current['phase']} phase"
                            if unmatched_attempt is not None
                            else "production launcher is running"
                            if retry_count == 0
                            else f"production launcher retry {retry_count} is running"
                        ),
                    ),
                )

            argv = _launcher_argv(
                inputs,
                config_path,
                dry_run=dry_run,
                stop_after_collected=stop_after_collected,
            )
            return_code = launcher_runner(
                argv,
                log_path=log_path,
                heartbeat_seconds=inputs.heartbeat_seconds,
                on_heartbeat=heartbeat,
                execution_lock_fd=execution_lock.descriptor,
            )
            _, after = _snapshot(inputs.campaign_root)
            operator_cancelled = return_code != 0 and _is_operator_cancellation(
                return_code
            )
            if return_code != 0:
                recovery = (
                    None
                    if dry_run or operator_cancelled
                    else _recoverable_collection_evidence(
                        inputs=inputs,
                        campaign=campaign,
                        before=coordinator,
                        after=after,
                    )
                )
                failure_ledger_path = None
                next_failure_count = failure_count
                next_retry_streak = retry_streak
                if not dry_run:
                    next_failure_count = failure_count + 1
                    failure_ledger_path = _publish_collection_failure(
                        inputs=inputs,
                        campaign=campaign,
                        coordinator=coordinator,
                        failure_index=next_failure_count,
                        config_path=config_path,
                        config_sha256=config_sha,
                        launcher_return_code=return_code,
                        operator_cancelled=operator_cancelled,
                        recovery=recovery,
                    )
                    next_retry_streak = _collection_retry_streak(
                        [
                            *collection_failures,
                            {
                                "recovery": recovery,
                                "operator_cancelled": operator_cancelled,
                            },
                        ]
                    )
                if (
                    recovery is not None
                    and next_retry_streak <= inputs.collection_retry_limit
                ):
                    next_retry = next_retry_streak
                    retrying = _status_document(
                        inputs=inputs,
                        coordinator=after,
                        driver_state="RETRYING",
                        config_path=config_path,
                        config_sha256=config_sha,
                        binding_path=binding_path,
                        collection_retry_count=next_retry,
                        collection_recovery=recovery,
                        collection_retry_ledger_path=failure_ledger_path,
                        detail=(
                            f"launcher exited nonzero ({return_code}); verified immutable "
                            f"durable phase frontier permits bounded retry {next_retry}/"
                            f"{inputs.collection_retry_limit}"
                        ),
                    )
                    _atomic_json(status_path, retrying)
                    if inputs.collection_retry_backoff_seconds:
                        time.sleep(inputs.collection_retry_backoff_seconds)
                    # Re-snapshot and publish a fresh immutable launcher config.
                    # Completed actor ranks are skipped by the actor-pair gate.
                    continue
                failed = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="FAILED",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    collection_retry_count=retry_count,
                    collection_recovery=recovery,
                    collection_retry_ledger_path=(
                        failure_ledger_path or latest_failure_path
                    ),
                    detail=(
                        f"launcher was cancelled by an operator signal ({return_code}); "
                        "automatic retry is forbidden"
                        if operator_cancelled
                        else f"launcher exited nonzero ({return_code}); fail-closed stop"
                        if recovery is None
                        else f"launcher exited nonzero ({return_code}); bounded collection "
                        f"retry limit {inputs.collection_retry_limit} exhausted"
                    ),
                )
                _atomic_json(status_path, failed)
                raise ProductionDriverError(f"production launcher exited {return_code}")
            if dry_run:
                if after != coordinator:
                    raise ProductionDriverError("dry-run unexpectedly changed coordinator state")
                validated = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state=(
                        "COLLECTION_ONLY_DRY_RUN_VALIDATED"
                        if stop_after_collected
                        else "DRY_RUN_VALIDATED"
                    ),
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail=(
                        "current durable collection passed launcher preflight "
                        "without submission"
                        if stop_after_collected
                        else "current durable round passed launcher preflight "
                        "without submission"
                    ),
                )
                _atomic_json(status_path, validated)
                return validated
            if stop_after_collected:
                receipt = _collection_only_receipt(
                    campaign_root=inputs.campaign_root,
                    campaign=campaign,
                    before=coordinator,
                    after=after,
                )
                completed = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="COLLECTION_ONLY_COMPLETE",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail=(
                        "the exact four-cohort collection is sealed; no learner, "
                        "evaluation, or promotion hook was launched"
                    ),
                )
                completed.update(receipt)
                _atomic_json(status_path, completed)
                return completed
            if not _round_was_resolved(coordinator, after):
                failed = _status_document(
                    inputs=inputs,
                    coordinator=after,
                    driver_state="FAILED",
                    config_path=config_path,
                    config_sha256=config_sha,
                    binding_path=binding_path,
                    detail="launcher returned success without exactly one durable terminal resolution",
                )
                _atomic_json(status_path, failed)
                raise ProductionDriverError(
                    "launcher success did not advance exactly one round to READY"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--campaign-contract-sha256", required=True)
    parser.add_argument("--learner-contract", type=Path, required=True)
    parser.add_argument("--learner-contract-sha256", required=True)
    parser.add_argument("--sealed-source-root", type=Path, required=True)
    parser.add_argument("--sealed-source-inventory-sha256", required=True)
    parser.add_argument("--evaluation-sealed-source-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-sealed-source-inventory-sha256", required=True
    )
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--learner-backend",
        choices=("oci_nrt", "lepton_job_v1"),
        default="lepton_job_v1",
    )
    parser.add_argument("--lepton-backend-contract", type=Path)
    parser.add_argument("--lepton-backend-contract-sha256")
    parser.add_argument("--lepton-remote-backend-contract")
    parser.add_argument("--lepton-control-exec-program", type=Path)
    parser.add_argument("--lepton-sdk-python", type=Path)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--collection-retry-limit", type=int, default=2)
    parser.add_argument(
        "--collection-retry-backoff-seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--stop-after-collected",
        action="store_true",
        help=(
            "seal the current round's four cohorts and stop before learner "
            "ingestion, training, evaluation, or promotion"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        inputs = DriverInputs(
            campaign_root=_absolute(arguments.campaign_root, "campaign root"),
            campaign_contract=_absolute(arguments.campaign_contract, "campaign contract"),
            campaign_contract_sha256=arguments.campaign_contract_sha256,
            learner_contract=_absolute(arguments.learner_contract, "learner contract"),
            learner_contract_sha256=arguments.learner_contract_sha256,
            sealed_source_root=_absolute(arguments.sealed_source_root, "sealed source root"),
            sealed_source_inventory_sha256=arguments.sealed_source_inventory_sha256,
            evaluation_sealed_source_root=_absolute(
                arguments.evaluation_sealed_source_root,
                "evaluation sealed source root",
            ),
            evaluation_sealed_source_inventory_sha256=(
                arguments.evaluation_sealed_source_inventory_sha256
            ),
            launcher=_absolute(arguments.launcher, "launcher"),
            launcher_sha256=arguments.launcher_sha256,
            control_root=_absolute(arguments.control_root, "control root"),
            learner_backend=arguments.learner_backend,
            lepton_backend_contract=(
                None
                if arguments.lepton_backend_contract is None
                else _absolute(
                    arguments.lepton_backend_contract,
                    "Lepton backend contract",
                )
            ),
            lepton_backend_contract_sha256=(
                arguments.lepton_backend_contract_sha256
            ),
            lepton_remote_backend_contract=(
                arguments.lepton_remote_backend_contract
            ),
            lepton_control_exec_program=(
                None
                if arguments.lepton_control_exec_program is None
                else _absolute(
                    arguments.lepton_control_exec_program,
                    "Lepton control exec program",
                )
            ),
            lepton_sdk_python=(
                None
                if arguments.lepton_sdk_python is None
                else _absolute(arguments.lepton_sdk_python, "Lepton SDK Python")
            ),
            heartbeat_seconds=arguments.heartbeat_seconds,
            collection_retry_limit=arguments.collection_retry_limit,
            collection_retry_backoff_seconds=(
                arguments.collection_retry_backoff_seconds
            ),
        )
        result = run_driver(
            inputs,
            dry_run=arguments.dry_run,
            stop_after_collected=arguments.stop_after_collected,
        )
    except ProductionDriverError as exc:
        print(f"production five-round driver failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "DriverInputs",
    "ExclusiveCampaignLock",
    "ProductionDriverError",
    "materialize_orchestration_config",
    "run_driver",
]
