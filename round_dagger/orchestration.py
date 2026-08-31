"""Fail-closed production orchestration for one round-based DAgger macro-round.

The actor, transfer, learner, and evaluator are deliberately external hooks:
their deployment details differ between RTX and the remote learner backend. Hooks only
produce artifacts.  This module owns every coordinator transition and verifies
the immutable artifact contract after each hook returns.

The production topology is intentionally not configurable: four synchronous,
sequential cohorts cover 64 logical environments each for 512 steps, yielding
exactly 256 environments and 131,072 transitions per macro-round.
"""

from __future__ import annotations

import ctypes
import contextlib
import fcntl
import functools
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from round_dagger import (
    actor_host_access,
    artifacts,
    evaluation,
    handoff,
    image3f_lifecycle,
    learner_identity,
    policy_bundle,
    promotion,
    replay,
    seed_policy,
    state,
    wandb_delivery,
)
from scripts.round_dagger import verify_sealed_source_snapshot as source_seal

# The internal remote-learner acceptance adapter contains deployment-specific
# allocation bindings and is intentionally not part of this portable release.
# Keep the coordinator importable for local/fake backends and fail explicitly
# if someone selects that omitted adapter at runtime.
try:  # pragma: no cover - exercised only by a separately deployed adapter.
    from round_dagger import learner_acceptance
except ImportError:  # pragma: no cover - expected in this portable distribution.
    learner_acceptance = None


ORCHESTRATION_CONFIG_SCHEMA = "fast_roborender.round_dagger.orchestration_config.v1"
BOUNDED_ORCHESTRATION_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.orchestration_config.v2"
)
PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.orchestration_config.v3"
)
BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.orchestration_config.v4"
)
RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.orchestration_config.v5"
)
EVALUATOR_RESULT_SCHEMA = "fast_roborender.round_dagger.evaluator_result.v2"
WANDB_PUBLIC_CONFIG_SCHEMA = "fast_roborender.round_dagger.wandb_public_config.v2"
PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA = (
    "fast_roborender.round_dagger.wandb_public_config.v3"
)
WANDB_PUBLIC_CONFIG_RECOVERY_SCHEMA = (
    "fast_roborender.round_dagger.wandb_public_config_recovery.v1"
)

COHORT_COUNT = 4
ENVIRONMENTS_PER_COHORT = 64
TARGET_ENVIRONMENTS = COHORT_COUNT * ENVIRONMENTS_PER_COHORT
STEPS_PER_ENVIRONMENT = 512
TRANSITIONS_PER_COHORT = ENVIRONMENTS_PER_COHORT * STEPS_PER_ENVIRONMENT
TRANSITIONS_PER_MACRO_ROUND = TARGET_ENVIRONMENTS * STEPS_PER_ENVIRONMENT
PRODUCTION_MACRO_ROUND_COUNT = 5
TRANSITIONS_PER_CAMPAIGN = (
    PRODUCTION_MACRO_ROUND_COUNT * TRANSITIONS_PER_MACRO_ROUND
)
EXPECTED_ACTOR_RANKS = (0, 1)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HOOK_NAMES = ("collect", "transfer", "ingest", "train", "evaluate", "promote")
_REQUIRED_HOOKS = {"collect", "train", "evaluate"}
_COORDINATOR_SOURCE_ROOT = Path(__file__).resolve().parents[1]
LEARNER_BACKEND_OCI_NRT = "oci_nrt"
LEARNER_BACKEND_LEPTON_JOB_V1 = "lepton_job_v1"
LEARNER_BACKENDS = (
    LEARNER_BACKEND_OCI_NRT,
    LEARNER_BACKEND_LEPTON_JOB_V1,
)
_TRUSTED_HOOK_RELATIVES = MappingProxyType(
    {
        "collect": Path(
            "scripts/round_dagger/run_production_actor_cohort_hook.sh"
        ),
        "evaluate": Path(
            "scripts/round_dagger/run_production_paired_evaluation_hook.sh"
        ),
    }
)
_TRUSTED_TRAIN_HOOK_RELATIVES = MappingProxyType(
    {
        LEARNER_BACKEND_OCI_NRT: Path(
            "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
        ),
        LEARNER_BACKEND_LEPTON_JOB_V1: Path(
            "scripts/round_dagger/run_production_lepton_learner_hook.sh"
        ),
    }
)
_EVALUATION_HOOK_RELATIVE = _TRUSTED_HOOK_RELATIVES["evaluate"]
# Linux UAPI values are stable across supported architectures.  Python builds
# predating their stdlib exposure still run on kernels that provide memfd and
# file seals, so use the libc entry point plus audited UAPI constants rather
# than weakening dispatch to a mutable temporary file.
_MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 0x0001)
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_SENSITIVE_ARGUMENT_RE = re.compile(
    r"(?i)(?:^--?(?:api[_-]?key|authorization|credential|password|secret|token)"
    r"(?:$|[=:])|(?:api[_-]?key|authorization|credential|password|secret|token)"
    r"\s*(?:=|:))"
)
_TRUSTED_HOST_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_LEARNER_RESULT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256",
    "model_lineage_sha256",
    "replay_snapshot_sha256",
    "code_sha256",
    "learner_config_sha256",
    "optimizer_steps",
    "initial_loss",
    "final_loss",
}
_EVALUATOR_RESULT_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "candidate_manifest_sha256",
    "evaluator_config_sha256",
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "passed",
    "metrics",
}
_PUBLIC_CONFIG_KEYS = {
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
_PRODUCTION_PUBLIC_CONFIG_KEYS = _PUBLIC_CONFIG_KEYS | {
    "round_dagger/campaign_contract_sha256",
    "round_dagger/data_lineage_sha256",
}
_PUBLIC_SHA_KEYS = {key for key in _PUBLIC_CONFIG_KEYS if key.endswith("_sha256")}
_PHASE_CODES = {
    "COLLECTING": 1,
    "COLLECTED": 2,
    "INGESTED": 3,
    "TRAINED": 4,
    "EVALUATED": 5,
    "READY": 6,
}


class OrchestrationError(RuntimeError):
    """A macro-round configuration, hook, or postcondition failed closed."""


class CampaignMutationLockError(RuntimeError):
    """The campaign-wide mutator lease could not be acquired safely."""


class CampaignMutationBusyError(CampaignMutationLockError):
    """Another process or thread currently owns the campaign mutator lease."""


_MUTATION_LOCK_SCHEMA = "fast_roborender.round_dagger.mutation_lease.v1"
_MUTATION_LOCK_DIRECTORY = "operator"
_MUTATION_LOCK_NAME = ".campaign_mutation.lock"
_MUTATION_REGISTRY_GUARD = threading.RLock()


@dataclass
class _HeldCampaignMutationLease:
    key: tuple[int, int]
    root: Path
    descriptor: int
    directory_descriptor: int
    owner_thread: int
    owner_pid: int
    depth: int
    metadata: dict[str, Any]


_HELD_MUTATION_LEASES: dict[tuple[int, int], _HeldCampaignMutationLease] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks for diagnostics, never lock authority."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        # The executable name in field two may contain spaces and parentheses;
        # split only after its final closing parenthesis.  starttime is field 22,
        # hence index 19 in the suffix beginning with field three.
        suffix = raw[raw.rfind(")") + 2 :].split()
        return int(suffix[19])
    except (OSError, ValueError, IndexError):
        return None


def _boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError:
        return None
    return value or None


def _write_lease_metadata(descriptor: int, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise CampaignMutationLockError("could not write mutation lease metadata")
        offset += written
    os.fsync(descriptor)


def _read_lease_metadata(descriptor: int) -> dict[str, Any] | None:
    try:
        size = os.fstat(descriptor).st_size
        if size <= 0 or size > 65_536:
            return None
        payload = os.pread(descriptor, size, 0)
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _owner_summary(metadata: Mapping[str, Any] | None) -> str:
    if metadata is None:
        return "owner metadata unavailable"
    fields = []
    for key in ("operation", "hostname", "pid", "pid_start_ticks", "acquired_at"):
        value = metadata.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return ", ".join(fields) if fields else "owner metadata unavailable"


def _open_mutation_lock(
    root: str | os.PathLike[str],
) -> tuple[Path, int, int, tuple[int, int]]:
    try:
        # Validate the complete path chain, not only the final component later
        # protected by O_NOFOLLOW.
        campaign_root = state._campaign_root(root)  # noqa: SLF001
    except (artifacts.ArtifactError, OSError, TypeError, ValueError) as exc:
        raise CampaignMutationLockError(
            "campaign root is missing, non-directory, or contains a symlink"
        ) from exc
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(campaign_root, root_flags)
    except OSError as exc:
        raise CampaignMutationLockError(
            "campaign root is missing, non-directory, or a symlink"
        ) from exc
    operator_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        root_stat = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise CampaignMutationLockError("campaign root is not a directory")
        try:
            os.mkdir(_MUTATION_LOCK_DIRECTORY, 0o700, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        except FileExistsError:
            pass
        operator_descriptor = os.open(
            _MUTATION_LOCK_DIRECTORY,
            root_flags,
            dir_fd=root_descriptor,
        )
        operator_stat = os.fstat(operator_descriptor)
        if not stat.S_ISDIR(operator_stat.st_mode):
            raise CampaignMutationLockError("campaign operator path is unsafe")
        lock_descriptor = os.open(
            _MUTATION_LOCK_NAME,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=operator_descriptor,
        )
        lock_stat = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise CampaignMutationLockError(
                "campaign mutation lock is not a singly-linked regular file"
            )
        if lock_stat.st_uid != os.geteuid():
            raise CampaignMutationLockError(
                "campaign mutation lock is not owned by the current user"
            )
        return (
            campaign_root,
            operator_descriptor,
            lock_descriptor,
            (root_stat.st_dev, root_stat.st_ino),
        )
    except OSError as exc:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if operator_descriptor is not None:
            os.close(operator_descriptor)
        raise CampaignMutationLockError(
            "campaign mutation lock path is missing, unsafe, or a symlink"
        ) from exc
    except BaseException:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if operator_descriptor is not None:
            os.close(operator_descriptor)
        raise
    finally:
        os.close(root_descriptor)


def _release_mutation_lease(held: _HeldCampaignMutationLease) -> None:
    with _MUTATION_REGISTRY_GUARD:
        current = _HELD_MUTATION_LEASES.get(held.key)
        if current is not held:
            return
        current.depth -= 1
        if current.depth > 0:
            return
        _HELD_MUTATION_LEASES.pop(held.key, None)
    try:
        released = {
            **held.metadata,
            "state": "released",
            "released_at": _utc_now(),
        }
        try:
            _write_lease_metadata(held.descriptor, released)
        except (OSError, CampaignMutationLockError):
            # Closing the descriptor is the authoritative release.  Diagnostic
            # metadata must never strand a kernel lease or mask a body error.
            pass
        try:
            fcntl.flock(held.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(held.descriptor)
    finally:
        os.close(held.directory_descriptor)


@contextlib.contextmanager
def campaign_mutation_lease(
    root: str | os.PathLike[str], operation: str
) -> Iterator[Mapping[str, Any]]:
    """Acquire one nonblocking campaign-wide mutation lease.

    Kernel ``flock`` ownership is the sole liveness authority, so an exited
    process cannot leave a stale PID-based lock.  The JSON is durable diagnostic
    provenance and includes both boot ID and process-start ticks to disambiguate
    PID reuse.  Same-thread nesting is reentrant for the hardened recovery's
    outer runner -> prepare/finalize call graph; another thread still fails
    closed instead of overlapping side effects.
    """

    if (
        not isinstance(operation, str)
        or re.fullmatch(r"[a-z0-9_.-]{1,96}", operation) is None
    ):
        raise CampaignMutationLockError("mutation operation name is invalid")
    campaign_root, directory_descriptor, descriptor, key = _open_mutation_lock(root)
    pid = os.getpid()
    thread_id = threading.get_ident()
    held: _HeldCampaignMutationLease | None = None
    with _MUTATION_REGISTRY_GUARD:
        inherited = _HELD_MUTATION_LEASES.get(key)
        if inherited is not None and inherited.owner_pid != pid:
            # After fork, discard the child's duplicate descriptors before a
            # fresh open/lock attempt.  The parent's open file description keeps
            # its lease, while the child cannot mistake inherited metadata for
            # reentrant ownership.
            _HELD_MUTATION_LEASES.pop(key, None)
            os.close(inherited.descriptor)
            os.close(inherited.directory_descriptor)
            inherited = None
        if inherited is not None:
            os.close(descriptor)
            os.close(directory_descriptor)
            if inherited.owner_thread != thread_id:
                raise CampaignMutationBusyError(
                    "campaign mutation lease is busy ("
                    + _owner_summary(inherited.metadata)
                    + ")"
                )
            inherited.depth += 1
            held = inherited
        else:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = _read_lease_metadata(descriptor)
                os.close(descriptor)
                os.close(directory_descriptor)
                raise CampaignMutationBusyError(
                    "campaign mutation lease is busy ("
                    + _owner_summary(owner)
                    + ")"
                ) from exc
            try:
                os.fchmod(descriptor, 0o600)
                metadata = {
                    "schema": _MUTATION_LOCK_SCHEMA,
                    "state": "held",
                    "lease_id": secrets.token_hex(16),
                    "operation": operation,
                    "hostname": socket.gethostname(),
                    "pid": pid,
                    "pid_start_ticks": _process_start_ticks(pid),
                    "boot_id": _boot_id(),
                    "thread_id": thread_id,
                    "acquired_at": _utc_now(),
                    "liveness_authority": "kernel_flock",
                }
                _write_lease_metadata(descriptor, metadata)
            except BaseException:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
                    os.close(directory_descriptor)
                raise
            held = _HeldCampaignMutationLease(
                key=key,
                root=campaign_root,
                descriptor=descriptor,
                directory_descriptor=directory_descriptor,
                owner_thread=thread_id,
                owner_pid=pid,
                depth=1,
                metadata=metadata,
            )
            _HELD_MUTATION_LEASES[key] = held
    try:
        yield MappingProxyType(dict(held.metadata))
    finally:
        _release_mutation_lease(held)


def campaign_mutation_guard(
    *,
    operation: str,
    root_parameter: str,
    error_type: type[Exception],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a public mutating entrypoint with the shared campaign lease."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            try:
                with campaign_mutation_lease(
                    bound.arguments[root_parameter], operation
                ):
                    return function(*args, **kwargs)
            except CampaignMutationLockError as exc:
                raise error_type(str(exc)) from exc

        return guarded

    return decorate


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise OrchestrationError(
            f"{label} has missing={sorted(keys - actual)} unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OrchestrationError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OrchestrationError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OrchestrationError(f"{label} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise OrchestrationError(f"{label} must be an absolute path")
    return Path(os.path.abspath(path))


def _safe_relative(value: str, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OrchestrationError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OrchestrationError(f"{label} is unsafe")
    return path


def validate_learner_backend(value: Any) -> str:
    """Return one immutable learner backend enum value or fail closed."""

    if not isinstance(value, str) or value not in LEARNER_BACKENDS:
        raise OrchestrationError(
            "learner_backend must be exactly one of "
            f"{list(LEARNER_BACKENDS)}"
        )
    return value


def trusted_train_hook_relative(learner_backend: str) -> Path:
    """Resolve a train primitive only from the explicit backend binding."""

    backend = validate_learner_backend(learner_backend)
    return _TRUSTED_TRAIN_HOOK_RELATIVES[backend]


@dataclass(frozen=True)
class HookSpec:
    """One argv-only hook. Credentials must be supplied through the environment."""

    argv: tuple[str, ...]
    timeout_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple) or not self.argv:
            raise OrchestrationError("hook argv must be a non-empty tuple")
        for index, raw in enumerate(self.argv):
            if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
                raise OrchestrationError(f"hook argv[{index}] is invalid")
            if _SENSITIVE_ARGUMENT_RE.search(raw):
                raise OrchestrationError(
                    "hook argv appears to contain a credential; pass secrets "
                    "through the inherited environment"
                )
        if not Path(self.argv[0]).is_absolute():
            raise OrchestrationError("hook argv[0] must be an absolute executable path")
        _true_int(self.timeout_seconds, "hook timeout_seconds", minimum=1)

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "HookSpec":
        document = _exact_mapping(value, {"argv", "timeout_seconds"}, label)
        raw_argv = document["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise OrchestrationError(f"{label}.argv must be a non-empty JSON list")
        argv: list[str] = []
        for index, raw in enumerate(raw_argv):
            if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw:
                raise OrchestrationError(f"{label}.argv[{index}] is invalid")
            if _SENSITIVE_ARGUMENT_RE.search(raw):
                raise OrchestrationError(
                    f"{label}.argv[{index}] appears to contain a credential; "
                    "pass secrets through the inherited environment"
                )
            argv.append(raw)
        if not Path(argv[0]).is_absolute():
            raise OrchestrationError(f"{label}.argv[0] must be an absolute executable path")
        timeout = _true_int(document["timeout_seconds"], f"{label}.timeout_seconds", minimum=1)
        return cls(tuple(argv), timeout)

    def public_digest(self) -> str:
        return artifacts.canonical_json_sha256(
            {"argv": list(self.argv), "timeout_seconds": self.timeout_seconds}
        )


@dataclass(frozen=True)
class MacroRoundConfig:
    """Strict operator-supplied inputs for one expected macro-round."""

    campaign_root: Path
    actor_root: Path
    macro_round: int
    runtime_policy_sha256: str
    learner_code_sha256: str
    learner_config_sha256: str
    learner_model_lineage_sha256: str
    evaluator_config_sha256: str
    hooks: Mapping[str, HookSpec | None]
    sha256: str
    macro_round_count: int | None = None
    campaign_contract_sha256: str | None = None
    learner_backend: str = LEARNER_BACKEND_LEPTON_JOB_V1
    learner_backend_contract_sha256: str | None = None
    advance_after_rejected_evaluation: bool = False

    def __post_init__(self) -> None:
        for path, label in (
            (self.campaign_root, "campaign_root"),
            (self.actor_root, "actor_root"),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise OrchestrationError(f"{label} must be an absolute Path")
        _true_int(self.macro_round, "macro_round")
        if self.macro_round_count is not None:
            _true_int(self.macro_round_count, "macro_round_count", minimum=1)
            if self.macro_round >= self.macro_round_count:
                raise OrchestrationError(
                    "macro_round must be below the bounded campaign horizon"
                )
        for value, label in (
            (self.runtime_policy_sha256, "runtime_policy_sha256"),
            (self.learner_code_sha256, "learner_code_sha256"),
            (self.learner_config_sha256, "learner_config_sha256"),
            (self.learner_model_lineage_sha256, "learner_model_lineage_sha256"),
            (self.evaluator_config_sha256, "evaluator_config_sha256"),
            (self.sha256, "orchestration config sha256"),
        ):
            _sha256(value, label)
        if self.campaign_contract_sha256 is not None:
            _sha256(
                self.campaign_contract_sha256,
                "production campaign contract sha256",
            )
        validate_learner_backend(self.learner_backend)
        if self.learner_backend_contract_sha256 is not None:
            _sha256(
                self.learner_backend_contract_sha256,
                "learner backend contract sha256",
            )
        if type(self.advance_after_rejected_evaluation) is not bool:
            raise OrchestrationError(
                "advance_after_rejected_evaluation must be boolean"
            )
        if not isinstance(self.hooks, Mapping) or set(self.hooks) != set(_HOOK_NAMES):
            raise OrchestrationError("hooks must contain exactly the six orchestration hooks")
        frozen_hooks: dict[str, HookSpec | None] = {}
        for name in _HOOK_NAMES:
            hook = self.hooks[name]
            if hook is None and name in _REQUIRED_HOOKS:
                raise OrchestrationError(f"{name} hook is required")
            if hook is not None and not isinstance(hook, HookSpec):
                raise OrchestrationError(f"{name} hook must be HookSpec or null")
            frozen_hooks[name] = hook
        object.__setattr__(self, "hooks", MappingProxyType(frozen_hooks))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MacroRoundConfig":
        source = Path(path).expanduser()
        try:
            raw = artifacts.load_strict_json(source)
        except Exception as exc:
            raise OrchestrationError(
                f"cannot load strict orchestration config ({type(exc).__name__})"
            ) from exc
        base_keys = {
            "schema",
            "campaign_root",
            "actor_root",
            "macro_round",
            "runtime_policy_sha256",
            "learner_code_sha256",
            "learner_config_sha256",
            "learner_model_lineage_sha256",
            "evaluator_config_sha256",
            "hooks",
        }
        if not isinstance(raw, Mapping):
            raise OrchestrationError("orchestration config must be a mapping")
        schema = raw.get("schema")
        if schema == ORCHESTRATION_CONFIG_SCHEMA:
            keys = base_keys
        elif schema == BOUNDED_ORCHESTRATION_CONFIG_SCHEMA:
            keys = base_keys | {"macro_round_count"}
        elif schema == PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA:
            keys = base_keys | {
                "macro_round_count",
                "campaign_contract_sha256",
            }
        elif schema == BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA:
            keys = base_keys | {
                "macro_round_count",
                "campaign_contract_sha256",
                "learner_backend",
                "learner_backend_contract_sha256",
            }
        elif schema == RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA:
            keys = base_keys | {
                "macro_round_count",
                "campaign_contract_sha256",
                "learner_backend",
                "learner_backend_contract_sha256",
                "advance_after_rejected_evaluation",
            }
        else:
            raise OrchestrationError("orchestration config schema mismatch")
        document = _exact_mapping(raw, keys, "orchestration config")
        raw_hooks = _exact_mapping(document["hooks"], set(_HOOK_NAMES), "hooks")
        hooks: dict[str, HookSpec | None] = {}
        for name in _HOOK_NAMES:
            value = raw_hooks[name]
            if value is None:
                if name in _REQUIRED_HOOKS:
                    raise OrchestrationError(f"{name} hook is required")
                hooks[name] = None
            else:
                hooks[name] = HookSpec.from_mapping(value, label=f"hooks.{name}")
        return cls(
            campaign_root=_absolute_path(document["campaign_root"], "campaign_root"),
            actor_root=_absolute_path(document["actor_root"], "actor_root"),
            macro_round=_true_int(document["macro_round"], "macro_round"),
            runtime_policy_sha256=_sha256(
                document["runtime_policy_sha256"], "runtime_policy_sha256"
            ),
            learner_code_sha256=_sha256(
                document["learner_code_sha256"], "learner_code_sha256"
            ),
            learner_config_sha256=_sha256(
                document["learner_config_sha256"], "learner_config_sha256"
            ),
            learner_model_lineage_sha256=_sha256(
                document["learner_model_lineage_sha256"],
                "learner_model_lineage_sha256",
            ),
            evaluator_config_sha256=_sha256(
                document["evaluator_config_sha256"], "evaluator_config_sha256"
            ),
            hooks=hooks,
            sha256=artifacts.canonical_json_sha256(document),
            macro_round_count=(
                None
                if schema == ORCHESTRATION_CONFIG_SCHEMA
                else _true_int(
                    document["macro_round_count"],
                    "macro_round_count",
                    minimum=1,
                )
            ),
            campaign_contract_sha256=(
                _sha256(
                    document["campaign_contract_sha256"],
                    "campaign_contract_sha256",
                )
                if schema
                in {
                    PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                    BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                    RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                }
                else None
            ),
            learner_backend=(
                validate_learner_backend(document["learner_backend"])
                if schema
                in {
                    BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                    RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                }
                # Pre-v4 files existed only for the named OCI compatibility
                # path. Never reinterpret their absent backend as Lepton.
                else LEARNER_BACKEND_OCI_NRT
            ),
            learner_backend_contract_sha256=(
                _sha256(
                    document["learner_backend_contract_sha256"],
                    "learner_backend_contract_sha256",
                )
                if schema
                in {
                    BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                    RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
                }
                else None
            ),
            advance_after_rejected_evaluation=(
                document["advance_after_rejected_evaluation"]
                if schema == RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA
                else False
            ),
        )


class HookRunner(Protocol):
    def run(
        self,
        name: str,
        hook: HookSpec,
        *,
        environment: Mapping[str, str],
        workdir: Path,
    ) -> None: ...


def _create_sealable_memfd(name: str) -> int:
    """Create a CLOEXEC, sealable Linux memfd on old and new CPython builds."""

    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING
    native = getattr(os, "memfd_create", None)
    if callable(native):
        return native(name, flags)
    if not sys.platform.startswith("linux"):
        raise OrchestrationError("sealed in-memory dispatch requires Linux")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        memfd_create = libc.memfd_create
    except (AttributeError, OSError) as exc:
        raise OrchestrationError(
            "sealed in-memory dispatch has no memfd_create entry point"
        ) from exc
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = int(memfd_create(name.encode("ascii"), flags))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OrchestrationError(
            f"sealed in-memory dispatch memfd_create failed (errno={error_number})"
        )
    return descriptor


def _verified_source_hook(
    name: str,
    hook: HookSpec,
    environment: Mapping[str, str],
) -> tuple[int, list[str], Path, str]:
    """Verify and snapshot one production hook before any seal code executes.

    The verifier is imported when the coordinator starts, from the
    coordinator's own contracted source.  It is intentionally not imported
    from the later source seal.  Bash consumes a sealed anonymous-memory
    snapshot, never the still-mutable source inode, so neither replacement nor
    in-place mutation can change the bytes after verification.

    The local RTX evaluation seal is also the coordinator hook seal.  This
    makes collection, learner submission, and evaluation use one verified
    source inventory instead of letting the first two execute mutable checkout
    bytes. The learner independently verifies the selected backend's remote
    source seal.
    """

    if name == "train":
        raw_backend = environment.get("ROUND_DAGGER_LEARNER_BACKEND")
        try:
            relative_hook = trusted_train_hook_relative(raw_backend)
        except OrchestrationError as exc:
            raise OrchestrationError(
                "train hook requires an explicit valid learner backend binding"
            ) from exc
    else:
        try:
            relative_hook = _TRUSTED_HOOK_RELATIVES[name]
        except KeyError as exc:
            raise OrchestrationError(
                f"{name} hook has no trusted sealed-dispatch contract"
            ) from exc

    raw_root = environment.get("ROUND_DAGGER_EVALUATION_SOURCE_ROOT")
    inventory_sha256 = environment.get(
        "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256"
    )
    if not raw_root or not inventory_sha256:
        raise OrchestrationError(
            f"{name} hook requires an explicit sealed coordinator source contract"
        )
    root = Path(os.path.abspath(raw_root))
    try:
        resolved = root.resolve(strict=True)
        root_info = root.lstat()
        parent_info = root.parent.lstat()
    except OSError as exc:
        raise OrchestrationError("sealed evaluation source is unavailable") from exc
    if (
        resolved != root
        or stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != 0o555
    ):
        raise OrchestrationError("sealed evaluation source root is unsafe")
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o555
    ):
        raise OrchestrationError(
            "sealed evaluation source parent must be an immutable 0555 directory"
        )
    expected_hook = root / relative_hook
    # Evaluation is configured directly from the later seal because its hook
    # is not needed until after training.  Collection and learner submission
    # retain their coordinator-checkout argv identity for launch-contract
    # compatibility, but those path bytes are never opened or executed here.
    configured_hook = (
        expected_hook
        if name == "evaluate"
        else _COORDINATOR_SOURCE_ROOT / relative_hook
    )
    if hook.argv != (str(configured_hook),):
        raise OrchestrationError(
            f"{name} hook path differs from its production primitive"
        )
    try:
        report = source_seal.verify(root, inventory_sha256)
        inventory_path = Path(report["inventory_path"])
        inventory_bytes, _ = source_seal._regular_file_bytes(  # noqa: SLF001
            inventory_path, "evaluation source inventory"
        )
        if hashlib.sha256(inventory_bytes).hexdigest() != inventory_sha256:
            raise source_seal.VerificationError(
                "evaluation source inventory changed after verification"
            )
        inventory = source_seal._parse_json(  # noqa: SLF001
            inventory_bytes, "evaluation source inventory"
        )
        records = source_seal._validate_inventory(inventory)  # noqa: SLF001
    except (KeyError, OSError, source_seal.VerificationError) as exc:
        raise OrchestrationError(
            "sealed coordinator source failed coordinator-side verification"
        ) from exc
    expected_record = next(
        (
            record
            for record in records
            if record["path"] == relative_hook.as_posix()
        ),
        None,
    )
    if expected_record is None or expected_record["mode"] != "0o555":
        raise OrchestrationError(
            f"{name} hook is absent from the sealed source inventory"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(expected_hook, flags)
    except OSError as exc:
        raise OrchestrationError(f"cannot safely open sealed {name} hook") from exc
    snapshot_descriptor: int | None = None
    try:
        snapshot_descriptor = _create_sealable_memfd(
            f"round-dagger-{name}-hook"
        )
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o555
            or before.st_nlink != 1
        ):
            raise OrchestrationError(f"sealed {name} hook metadata is unsafe")
        digest = hashlib.sha256()
        observed = 0
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            digest.update(block)
            offset = 0
            while offset < len(block):
                written = os.write(snapshot_descriptor, block[offset:])
                if written <= 0:
                    raise OrchestrationError(
                        "cannot snapshot sealed evaluation hook bytes"
                    )
                offset += written
        after = os.fstat(source_descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or observed != before.st_size:
            raise OrchestrationError(f"sealed {name} hook changed while opened")
        if (
            digest.hexdigest() != expected_record["sha256"]
            or observed != expected_record["bytes"]
        ):
            raise OrchestrationError(
                f"sealed {name} hook bytes differ from verified inventory"
            )
        seals = _F_SEAL_WRITE | _F_SEAL_GROW | _F_SEAL_SHRINK | _F_SEAL_SEAL
        fcntl.fcntl(snapshot_descriptor, _F_ADD_SEALS, seals)
        observed_seals = fcntl.fcntl(snapshot_descriptor, _F_GET_SEALS)
        if observed_seals & seals != seals:
            raise OrchestrationError(
                f"{name} hook snapshot could not be made immutable"
            )
        os.lseek(snapshot_descriptor, 0, os.SEEK_SET)
        os.close(source_descriptor)
        source_descriptor = -1
        return (
            snapshot_descriptor,
            ["/bin/bash", f"/proc/self/fd/{snapshot_descriptor}"],
            root,
            inventory_sha256,
        )
    except Exception:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)
        raise


def _verified_evaluation_hook(
    hook: HookSpec,
    environment: Mapping[str, str],
) -> tuple[int, list[str]]:
    """Compatibility wrapper for the original evaluation-only helper."""

    descriptor, argv, _, _ = _verified_source_hook(
        "evaluate", hook, environment
    )
    return descriptor, argv


def _actor_host_access_helper(environment: Mapping[str, str]) -> None:
    """Run the narrow root helper from the coordinator source, never actor code."""

    inventory_sha256 = environment.get(
        "ROUND_DAGGER_LEARNER_SOURCE_INVENTORY_SHA256"
    )
    if not inventory_sha256 or _SHA256_RE.fullmatch(inventory_sha256) is None:
        raise OrchestrationError(
            "actor host-access recovery lacks coordinator source identity"
        )
    helper = (
        _COORDINATOR_SOURCE_ROOT
        / "scripts/round_dagger/run_actor_host_access_docker.sh"
    )
    try:
        info = os.lstat(helper)
    except OSError as exc:
        raise OrchestrationError(
            "actor host-access recovery helper is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not os.access(helper, os.X_OK)
    ):
        raise OrchestrationError("actor host-access recovery helper is unsafe")
    helper_environment = dict(environment)
    helper_environment.update(
        ROUND_DAGGER_TRUSTED_COORDINATOR_HOST_ACCESS="1",
        ROUND_DAGGER_COORDINATOR_SOURCE_ROOT=str(_COORDINATOR_SOURCE_ROOT),
        ROUND_DAGGER_COORDINATOR_SOURCE_INVENTORY_SHA256=inventory_sha256,
    )
    try:
        completed = subprocess.run(
            [str(helper)],
            cwd=_COORDINATOR_SOURCE_ROOT,
            env=helper_environment,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OrchestrationError(
            "actor host-access recovery helper could not complete"
        ) from exc
    if completed.returncode != 0:
        raise OrchestrationError(
            "actor host-access recovery helper failed closed"
        )


def _repair_actor_host_access(environment: Mapping[str, str]) -> bool:
    """Repair inaccessible legacy output and prove host access; return if repaired."""

    raw_run_dir = environment.get("TRAIN_RUN_DIR")
    if not raw_run_dir:
        raise OrchestrationError("collect hook lacks actor run directory")
    run_dir = Path(os.path.abspath(raw_run_dir))
    try:
        actor_host_access.verify_host_access(run_dir)
        return False
    except actor_host_access.ActorHostAccessError:
        _actor_host_access_helper(environment)
    try:
        actor_host_access.verify_host_access(run_dir)
    except actor_host_access.ActorHostAccessError as exc:
        raise OrchestrationError(
            "actor run remained inaccessible after ownership recovery"
        ) from exc
    return True


def _prepare_actor_host_access(environment: Mapping[str, str]) -> None:
    raw_run_dir = environment.get("TRAIN_RUN_DIR")
    if not raw_run_dir:
        raise OrchestrationError("collect hook lacks actor run directory")
    run_dir = Path(os.path.abspath(raw_run_dir))
    try:
        actor_host_access.prepare_host_layout(run_dir)
    except actor_host_access.ActorHostAccessError:
        _actor_host_access_helper(environment)
        try:
            actor_host_access.verify_host_access(run_dir)
        except actor_host_access.ActorHostAccessError as exc:
            raise OrchestrationError(
                "actor run is not host-accessible before collection"
            ) from exc


def _collect_cohort_is_sealed(environment: Mapping[str, str]) -> bool:
    raw_cohort = environment.get("ROUND_DAGGER_COHORT_DIR")
    if not raw_cohort:
        raise OrchestrationError("collect hook lacks cohort directory")
    try:
        return bool(artifacts.verify_cohort(Path(raw_cohort))["sealed"])
    except (artifacts.ArtifactError, OSError, ValueError):
        return False


def _run_collect_with_host_access(
    invoke: Callable[[], subprocess.CompletedProcess[Any]],
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[Any]:
    """Run legacy collection with one sealed-only ownership recovery retry.

    The accepted actor/evaluation source remains unchanged.  A fresh cohort's
    shared layout is created by the host before the legacy root container runs.
    If that legacy hook seals the cohort but then fails its host-side telemetry
    finalization, the coordinator normalizes only telemetry/W&B output and
    invokes the same sealed hook one more time.  Its existing sealed branch
    cannot launch rollout ranks, so cohort data are adopted without rerollout.
    """

    _prepare_actor_host_access(environment)
    completed = invoke()
    if completed.returncode == 0:
        _repair_actor_host_access(environment)
        return completed
    if not _collect_cohort_is_sealed(environment):
        return completed
    if not _repair_actor_host_access(environment):
        return completed
    retried = invoke()
    if retried.returncode == 0:
        _repair_actor_host_access(environment)
    return retried


def _isolate_trusted_hook_python(
    child_environment: dict[str, str],
) -> None:
    """Pin bare ``python3`` calls in trusted host hooks to the control venv.

    The reviewed shell hooks still contain bare ``python3`` invocations.  They
    must not resolve through an operator's PATH or user site-packages.  The
    production contract already binds an absolute control-venv entry point;
    require its sibling ``python3`` to resolve to the same interpreter and
    publish a small deterministic host PATH.  Actor-container PATH is not
    forwarded by the Docker executor, so this affects coordinator-side tools
    only.
    """

    raw_control_python = child_environment.get("ROUND_DAGGER_CONTROL_PYTHON")
    if not raw_control_python or not os.path.isabs(raw_control_python):
        raise OrchestrationError(
            "trusted hook requires an absolute control Python"
        )
    control_python = Path(raw_control_python)
    sibling_python = control_python.parent / "python3"
    virtual_environment = control_python.parent.parent
    venv_configuration = virtual_environment / "pyvenv.cfg"
    try:
        control_target = control_python.resolve(strict=True)
        sibling_target = sibling_python.resolve(strict=True)
        control_info = control_target.stat()
        sibling_info = sibling_target.stat()
        venv_info = venv_configuration.lstat()
    except OSError as exc:
        raise OrchestrationError(
            "trusted hook control Python is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(control_info.st_mode)
        or not stat.S_ISREG(sibling_info.st_mode)
        or stat.S_ISLNK(venv_info.st_mode)
        or not stat.S_ISREG(venv_info.st_mode)
        or control_python.parent.name != "bin"
        or not os.access(control_python, os.X_OK)
        or not os.access(sibling_python, os.X_OK)
        or not os.path.samefile(control_target, sibling_target)
    ):
        raise OrchestrationError(
            "trusted hook python3 does not match the control Python"
        )
    # Non-interactive bash reads BASH_ENV before it evaluates the verified
    # memfd-backed hook.  Do not let an operator login shell inject commands
    # ahead of the sealed source.  ENV is the corresponding POSIX-shell knob;
    # the remaining entries can otherwise alter or hang short ``python3 -c``
    # control probes used by the reviewed hooks.
    for name in (
        "BASH_ENV",
        "ENV",
        "PYTHONHOME",
        "PYTHONBREAKPOINT",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        child_environment.pop(name, None)
    child_environment.update(
        {
            "PATH": f"{control_python.parent}:{_TRUSTED_HOST_SYSTEM_PATH}",
            "VIRTUAL_ENV": str(virtual_environment),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )


class SubprocessHookRunner:
    """Run hooks without a shell and without echoing argv or environment."""

    def run(
        self,
        name: str,
        hook: HookSpec,
        *,
        environment: Mapping[str, str],
        workdir: Path,
    ) -> None:
        child_environment = dict(os.environ)
        child_environment.update(environment)
        verified_descriptor: int | None = None
        collect_sealed_teacher: Path | None = None
        trusted_hook = name in _TRUSTED_HOOK_RELATIVES or name == "train"
        if trusted_hook:
            _isolate_trusted_hook_python(child_environment)
        if name == "collect":
            raw_source_root = child_environment.get(
                "ROUND_DAGGER_EVALUATION_SOURCE_ROOT"
            )
            raw_sealed_teacher = child_environment.get(
                "ROUND_DAGGER_EVALUATION_SEALED_TEACHER"
            )
            if not raw_source_root or not raw_sealed_teacher:
                raise OrchestrationError(
                    "collect hook requires the inventory-bound sealed teacher"
                )
            source_root = Path(os.path.abspath(raw_source_root))
            collect_sealed_teacher = Path(os.path.abspath(raw_sealed_teacher))
            try:
                collect_sealed_teacher.relative_to(source_root)
            except ValueError as exc:
                raise OrchestrationError(
                    "sealed teacher is outside the trusted actor source"
                ) from exc
        if trusted_hook:
            (
                verified_descriptor,
                argv,
                trusted_source_root,
                trusted_source_inventory_sha256,
            ) = _verified_source_hook(name, hook, child_environment)
            child_environment["ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT"] = str(
                trusted_source_root
            )
            child_environment[
                "ROUND_DAGGER_TRUSTED_HOOK_SOURCE_INVENTORY_SHA256"
            ] = trusted_source_inventory_sha256
            if name == "collect":
                child_environment["ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH"] = "1"
                child_environment["ROUND_DAGGER_ACTOR_SOURCE_ROOT"] = str(
                    trusted_source_root
                )
                child_environment[
                    "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256"
                ] = trusted_source_inventory_sha256
                child_environment["GR00T_ROOT"] = str(
                    trusted_source_root / "GR00T-VisualSim2Real"
                )
                child_environment["ROUND_DAGGER_ACTOR_PAIR_EXECUTOR"] = str(
                    trusted_source_root
                    / "scripts/round_dagger/run_actor_rank_docker.sh"
                )
                child_environment["ROUND_DAGGER_ACTOR_PAIR_RANK_SCRIPT"] = str(
                    trusted_source_root
                    / "scripts/round_dagger/run_actor_cohort_with_wandb.sh"
                )
                assert collect_sealed_teacher is not None
                child_environment["ROUND_DAGGER_TEACHER_CHECKPOINT"] = str(
                    collect_sealed_teacher
                )
            elif name == "train":
                child_environment["ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH"] = "1"
                child_environment["ROUND_DAGGER_COORDINATOR_SOURCE_ROOT"] = str(
                    trusted_source_root
                )
            else:
                child_environment[
                    "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH"
                ] = "1"
        else:
            executable = Path(hook.argv[0])
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise OrchestrationError(
                    f"{name} hook executable is missing or not executable"
                )
            argv = list(hook.argv)
        try:
            options: dict[str, Any] = {
                "cwd": workdir,
                "env": child_environment,
                "check": False,
                "timeout": hook.timeout_seconds,
            }
            if verified_descriptor is not None:
                options["pass_fds"] = (verified_descriptor,)

            def invoke() -> subprocess.CompletedProcess[Any]:
                if verified_descriptor is not None:
                    os.lseek(verified_descriptor, 0, os.SEEK_SET)
                return subprocess.run(argv, **options)

            completed = (
                _run_collect_with_host_access(invoke, child_environment)
                if name == "collect"
                and child_environment.get("ROUND_DAGGER_CAMPAIGN_CONTRACT")
                else invoke()
            )
        except subprocess.TimeoutExpired as exc:
            raise OrchestrationError(f"{name} hook timed out") from exc
        except OSError as exc:
            raise OrchestrationError(
                f"{name} hook could not start ({type(exc).__name__})"
            ) from exc
        finally:
            if verified_descriptor is not None:
                os.close(verified_descriptor)
        if completed.returncode != 0:
            raise OrchestrationError(
                f"{name} hook failed with exit status {completed.returncode}"
            )


def validate_production_topology(
    campaign: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> None:
    """Reject any campaign that is not exactly 4 x 64 x 512."""

    expected = {
        "target_environments": TARGET_ENVIRONMENTS,
        "active_environments": ENVIRONMENTS_PER_COHORT,
        "cohort_count": COHORT_COUNT,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
    }
    for key, value in expected.items():
        if campaign.get(key) != value:
            raise OrchestrationError(
                f"production topology requires campaign.{key}={value}"
            )
    if contract is None:
        return
    if contract.get("steps_per_environment") != STEPS_PER_ENVIRONMENT:
        raise OrchestrationError("macro-round step count is not exactly 512")
    cohorts = contract.get("cohorts")
    expected_cohorts = [
        {
            "cohort_index": index,
            "environment_start": index * ENVIRONMENTS_PER_COHORT,
            "environment_count": ENVIRONMENTS_PER_COHORT,
        }
        for index in range(COHORT_COUNT)
    ]
    if not isinstance(cohorts, list) or len(cohorts) != len(expected_cohorts):
        raise OrchestrationError(
            "macro-round cohorts do not exactly cover logical environments 0..255"
        )
    for observed, expected_cohort in zip(cohorts, expected_cohorts, strict=True):
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value for key, value in expected_cohort.items()
        ):
            raise OrchestrationError(
                "macro-round cohorts do not exactly cover logical environments 0..255"
            )


def _public_wandb_config(
    *,
    config: MacroRoundConfig,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    cohort_index: int | None,
) -> dict[str, Any]:
    """Return the only orchestration values permitted in W&B run config.

    Paths, hook argv, inherited environment values, account details, and
    credentials are intentionally absent.
    """

    public = {
        "round_dagger/campaign_id": campaign["campaign_id"],
        "round_dagger/macro_round": config.macro_round,
        "round_dagger/cohort_index": -1 if cohort_index is None else cohort_index,
        "round_dagger/cohort_count": COHORT_COUNT,
        "round_dagger/environments_per_cohort": ENVIRONMENTS_PER_COHORT,
        "round_dagger/target_environments": TARGET_ENVIRONMENTS,
        "round_dagger/steps_per_environment": STEPS_PER_ENVIRONMENT,
        "round_dagger/transitions_per_cohort": TRANSITIONS_PER_COHORT,
        "round_dagger/transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
        "round_dagger/macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
        "round_dagger/transitions_per_campaign": TRANSITIONS_PER_CAMPAIGN,
        "round_dagger/policy_version": contract["policy_version"],
        "round_dagger/policy_checkpoint_sha256": contract[
            "policy_checkpoint_sha256"
        ],
        "round_dagger/runtime_policy_sha256": contract["policy_runtime_sha256"],
        "round_dagger/teacher_sha256": contract["teacher_sha256"],
        "round_dagger/renderer_sha256": contract["renderer_sha256"],
        "round_dagger/camera_mapping_sha256": contract["camera_mapping_sha256"],
        "round_dagger/training_config_sha256": contract["training_config_sha256"],
        "round_dagger/beta": float(contract["beta"]),
        "round_dagger/beta_sha256": contract["beta_sha256"],
        "round_dagger/orchestration_config_sha256": config.sha256,
    }
    schema = WANDB_PUBLIC_CONFIG_SCHEMA
    if config.campaign_contract_sha256 is not None:
        if "data_lineage_sha256" not in contract:
            raise OrchestrationError(
                "production telemetry requires frozen data lineage"
            )
        public["round_dagger/campaign_contract_sha256"] = (
            config.campaign_contract_sha256
        )
        public["round_dagger/data_lineage_sha256"] = contract[
            "data_lineage_sha256"
        ]
        schema = PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA
    return {"schema": schema, "config": public}


def _validate_public_wandb_config(document: Mapping[str, Any]) -> None:
    wrapper = _exact_mapping(
        document, {"schema", "config"}, "public W&B config wrapper"
    )
    schema = wrapper["schema"]
    if schema not in {
        WANDB_PUBLIC_CONFIG_SCHEMA,
        PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA,
    }:
        raise OrchestrationError("public W&B config schema mismatch")
    values = _exact_mapping(
        wrapper["config"],
        (
            _PRODUCTION_PUBLIC_CONFIG_KEYS
            if schema == PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA
            else _PUBLIC_CONFIG_KEYS
        ),
        "public W&B config",
    )
    campaign_id = values["round_dagger/campaign_id"]
    if (
        not isinstance(campaign_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id) is None
    ):
        raise OrchestrationError("public W&B campaign_id is not a safe token")
    for key in _PUBLIC_SHA_KEYS:
        _sha256(values[key], key)
    if schema == PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA:
        for key in (
            "round_dagger/campaign_contract_sha256",
            "round_dagger/data_lineage_sha256",
        ):
            _sha256(values[key], key)
    exact_integers = {
        "round_dagger/cohort_count": COHORT_COUNT,
        "round_dagger/environments_per_cohort": ENVIRONMENTS_PER_COHORT,
        "round_dagger/target_environments": TARGET_ENVIRONMENTS,
        "round_dagger/steps_per_environment": STEPS_PER_ENVIRONMENT,
        "round_dagger/transitions_per_cohort": TRANSITIONS_PER_COHORT,
        "round_dagger/transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
        "round_dagger/macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
        "round_dagger/transitions_per_campaign": TRANSITIONS_PER_CAMPAIGN,
    }
    for key, expected in exact_integers.items():
        if _true_int(values[key], key, minimum=1) != expected:
            raise OrchestrationError(f"public W&B config {key} violates topology")
    for key in ("round_dagger/macro_round", "round_dagger/policy_version"):
        _true_int(values[key], key)
    cohort_index = values["round_dagger/cohort_index"]
    if type(cohort_index) is not int or cohort_index not in {-1, 0, 1, 2, 3}:
        raise OrchestrationError("public W&B cohort_index must be -1 or 0..3")
    beta = values["round_dagger/beta"]
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta))
        or not 0.0 <= float(beta) <= 1.0
    ):
        raise OrchestrationError("public W&B beta must be finite and in [0,1]")


def _canonical_public_config_bytes(
    document: Mapping[str, Any], *, stable: bool
) -> bytes:
    """Canonical bytes for a validated public W&B config.

    The stable projection excludes only the orchestration source digest.  Comparing
    these bytes (rather than Python values) preserves JSON type distinctions such
    as ``0`` versus ``0.0`` during a recovery decision.
    """

    values = dict(document["config"])
    if stable:
        del values["round_dagger/orchestration_config_sha256"]
    return json.dumps(
        {"schema": document["schema"], "config": values},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _recover_production_public_wandb_config(
    *, path: Path, requested: Mapping[str, Any]
) -> Path:
    """Accept the one safe immutable-config drift allowed on a production retry."""

    try:
        before_sha = artifacts.sha256_file(path)
        existing = artifacts.load_strict_json(path)
        after_load_sha = artifacts.sha256_file(path)
    except (artifacts.ArtifactError, OSError) as exc:
        raise OrchestrationError(
            "existing public W&B config is missing, corrupt, or unsafe"
        ) from exc
    if before_sha != after_load_sha:
        raise OrchestrationError("existing public W&B config changed while loading")
    if not isinstance(existing, Mapping):
        raise OrchestrationError("existing public W&B config must be a mapping")

    # Both documents must independently satisfy the complete production-v3
    # allowlist before any recovery comparison is made.
    _validate_public_wandb_config(existing)
    _validate_public_wandb_config(requested)
    if (
        existing["schema"] != PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA
        or requested["schema"] != PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA
    ):
        raise OrchestrationError(
            "public W&B config recovery requires production v3 documents"
        )

    existing_bytes = _canonical_public_config_bytes(existing, stable=False)
    requested_bytes = _canonical_public_config_bytes(requested, stable=False)
    if hashlib.sha256(existing_bytes).hexdigest() != before_sha:
        raise OrchestrationError(
            "existing public W&B config is not a canonical immutable artifact"
        )
    existing_stable = _canonical_public_config_bytes(existing, stable=True)
    requested_stable = _canonical_public_config_bytes(requested, stable=True)
    if existing_stable != requested_stable:
        raise OrchestrationError(
            "public W&B config differs beyond orchestration_config_sha256"
        )

    old_orchestration_sha = existing["config"][
        "round_dagger/orchestration_config_sha256"
    ]
    new_orchestration_sha = requested["config"][
        "round_dagger/orchestration_config_sha256"
    ]
    if old_orchestration_sha == new_orchestration_sha:
        # The immutable publisher reaches this function only after a byte
        # conflict.  Equal source digests therefore indicate non-semantic byte
        # drift or corruption, neither of which is recoverable.
        raise OrchestrationError(
            "public W&B config conflicts without orchestration config drift"
        )

    normalized_stable_sha = hashlib.sha256(existing_stable).hexdigest()
    requested_sha = hashlib.sha256(requested_bytes).hexdigest()
    recovery = {
        "schema": WANDB_PUBLIC_CONFIG_RECOVERY_SCHEMA,
        "old_public_config_sha256": before_sha,
        "new_public_config_sha256": requested_sha,
        "old_orchestration_config_sha256": old_orchestration_sha,
        "new_orchestration_config_sha256": new_orchestration_sha,
        "normalized_stable_config_sha256": normalized_stable_sha,
    }
    recovery_path = (
        path.parent
        / "WANDB_PUBLIC_CONFIG_RECOVERY"
        / f"{new_orchestration_sha}.json"
    )
    try:
        artifacts.publish_immutable_json(recovery_path, recovery)
        final_sha = artifacts.sha256_file(path)
    except (artifacts.ArtifactError, OSError) as exc:
        raise OrchestrationError(
            "cannot publish immutable public W&B config recovery provenance"
        ) from exc
    if final_sha != before_sha:
        raise OrchestrationError(
            "existing public W&B config changed during recovery publication"
        )
    return path


class RoundTelemetry:
    """Deterministic scalar/config artifacts consumed by the W&B sidecar."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.telemetry_dir = self.run_dir / "telemetry"

    def publish_config(self, document: Mapping[str, Any]) -> Path:
        _validate_public_wandb_config(document)
        path = self.telemetry_dir / "WANDB_PUBLIC_CONFIG.json"
        try:
            artifacts.publish_immutable_json(path, document)
        except artifacts.ArtifactError:
            # Production restarts may intentionally use a newly sealed
            # orchestration source while continuing the same W&B run.  Keep the
            # original public config immutable and record the sole permitted
            # source-digest drift separately.  Every other conflict fails closed.
            return _recover_production_public_wandb_config(
                path=path, requested=document
            )
        return path

    def record(self, *, step: int, metrics: Mapping[str, int | float]) -> Path:
        clean: dict[str, int | float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise OrchestrationError("telemetry metric keys must be non-empty strings")
            if isinstance(value, bool):
                clean[key] = int(value)
            elif isinstance(value, int):
                clean[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                clean[key] = value
            else:
                raise OrchestrationError(f"telemetry metric {key!r} is not finite numeric")
        payload = {"step": _true_int(step, "telemetry step"), "metrics": clean}
        path = self.telemetry_dir / "scalars" / f"step_{step:09d}_macro_round.json"
        artifacts.publish_immutable_json(path, payload)
        return path


class ProductionBackend:
    """Adapter that composes the existing immutable round_dagger primitives."""

    def __init__(self, config: MacroRoundConfig):
        self.config = config
        self.root = config.campaign_root

    def load_campaign(self) -> dict[str, Any]:
        return state.load_campaign(self.root)

    def load_state(self) -> dict[str, Any]:
        return state.load_state(self.root)

    def open_round(self) -> dict[str, Any]:
        current = self.load_state()
        return state.open_macro_round(self.root, expected_revision=current["revision"])

    def load_contract(self) -> dict[str, Any]:
        return state.load_round_contract(self.root, self.config.macro_round)

    def ensure_cohort_contract(
        self,
        path: Path,
        *,
        campaign: Mapping[str, Any],
        contract: Mapping[str, Any],
        cohort_index: int,
    ) -> None:
        start = cohort_index * ENVIRONMENTS_PER_COHORT
        expected_global_env_ids = list(
            range(start, start + ENVIRONMENTS_PER_COHORT)
        )
        actor_seed_contract = seed_policy.derive_actor_seed_contract(
            campaign_id=campaign["campaign_id"],
            macro_round=self.config.macro_round,
            cohort_index=cohort_index,
            expected_ranks=EXPECTED_ACTOR_RANKS,
            expected_global_env_ids=expected_global_env_ids,
        )
        artifacts.create_cohort(
            path,
            campaign_id=campaign["campaign_id"],
            macro_round=self.config.macro_round,
            cohort_index=cohort_index,
            purpose="train",
            expected_ranks=EXPECTED_ACTOR_RANKS,
            expected_global_env_ids=expected_global_env_ids,
            expected_transitions=TRANSITIONS_PER_COHORT,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": contract["policy_image_shape"],
                "views": contract["view_names"],
            },
            action_contract={"dtype": "float32", "shape": [contract["action_dim"]]},
            policy_sha256=contract["policy_checkpoint_sha256"],
            runtime_policy_sha256=contract["policy_runtime_sha256"],
            teacher_sha256=contract["teacher_sha256"],
            renderer_sha256=contract["renderer_sha256"],
            beta_sha256=contract["beta_sha256"],
            config_sha256=contract["training_config_sha256"],
            actor_seed_contract=actor_seed_contract,
            data_lineage=contract.get("data_lineage"),
            motion_assignment=(
                contract["cohorts"][cohort_index].get("motion_assignment")
                if contract.get("data_lineage") is not None
                else None
            ),
        )

    def _actor_run_dir(self, path: Path) -> Path | None:
        """Map an exact actor-source cohort to its lifecycle run directory."""

        try:
            relative = path.relative_to(self.config.actor_root)
        except ValueError:
            return None
        expected_round = f"macro_round_{self.config.macro_round:06d}"
        if (
            len(relative.parts) != 3
            or relative.parts[1] != expected_round
            or re.fullmatch(r"cohort_[0-9]{2}", relative.parts[2]) is None
        ):
            return None
        return path.with_name(f"{path.name}_run")

    def _require_actor_cleanup(self, path: Path) -> dict[str, Any] | None:
        run_dir = self._actor_run_dir(path)
        if run_dir is None:
            return None
        report = run_dir / "image3f_stream_cleanup" / "cleanup_complete.json"
        if not report.exists():
            return None
        try:
            return image3f_lifecycle.verify_cleanup_report(
                run_dir, cohort_dir=path
            )
        except image3f_lifecycle.Image3FLifecycleError as exc:
            raise OrchestrationError(
                f"actor cohort has invalid Image3F cleanup evidence: {exc}"
            ) from exc

    def _require_actor_wandb_delivery(self, path: Path) -> dict[str, Any] | None:
        """Require sender-drained actor telemetry for production cohorts."""

        run_dir = self._actor_run_dir(path)
        if run_dir is None or self.config.campaign_contract_sha256 is None:
            return None
        try:
            return wandb_delivery.verify_delivery(run_dir)
        except wandb_delivery.WandbDeliveryError as exc:
            # Legacy actor containers may have completed and sealed their
            # cohort before leaving telemetry owned by another runtime UID.
            # That is a recoverable *access* boundary: return ``None`` so the
            # orchestrator re-enters the idempotent collect hook, whose
            # host-access wrapper normalizes only the actor run telemetry and
            # then executes the sealed (no-rerollout) recovery branch.  If the
            # host can already inspect the run, the evidence is genuinely
            # malformed or contradictory and must continue to fail closed.
            try:
                actor_host_access.verify_host_access(run_dir)
            except actor_host_access.ActorHostAccessError:
                return None
            raise OrchestrationError(
                f"actor cohort has invalid W&B delivery evidence: {exc}"
            ) from exc

    def complete_cohort(self, path: Path) -> dict[str, Any] | None:
        if not (path / artifacts.SEAL_FILENAME).exists():
            return None
        if self._actor_run_dir(path) is not None and self._require_actor_cleanup(path) is None:
            # A pre-hotfix or interrupted seal is not collection completion.
            # Returning None makes the orchestrator invoke the idempotent
            # collection hook, whose sealed recovery path performs cleanup.
            return None
        if (
            self._actor_run_dir(path) is not None
            and self.config.campaign_contract_sha256 is not None
            and self._require_actor_wandb_delivery(path) is None
        ):
            # A sealed cohort with an interrupted sidecar remains replayable.
            # Returning None invokes the collect hook's sealed recovery branch.
            return None
        return artifacts.verify_cohort(path)

    def seal_cohort(self, path: Path) -> dict[str, Any]:
        if self._actor_run_dir(path) is not None and self._require_actor_cleanup(path) is None:
            raise OrchestrationError(
                "actor cohort cannot be sealed without verified Image3F cleanup"
            )
        if (
            self._actor_run_dir(path) is not None
            and self.config.campaign_contract_sha256 is not None
            and self._require_actor_wandb_delivery(path) is None
        ):
            raise OrchestrationError(
                "production actor cohort cannot be accepted without durable W&B delivery"
            )
        return artifacts.seal_cohort(path)

    def transfer_local(self, source: Path, destination: Path) -> dict[str, Any]:
        return handoff.transfer_local(source, destination)

    def seal_collection(self, cohort_paths: Sequence[Path]) -> dict[str, Any]:
        return state.seal_collection(
            self.root,
            macro_round=self.config.macro_round,
            cohort_manifests=list(cohort_paths),
        )

    def mark_collected(self, collection: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="COLLECTING",
            next_phase="COLLECTED",
            macro_round=self.config.macro_round,
            evidence_path=collection["manifest_path"],
            evidence_sha256=collection["manifest_sha256"],
            expected_revision=current["revision"],
        )

    def load_collection(self) -> dict[str, Any]:
        path = state.round_directory(self.root, self.config.macro_round) / "collection.json"
        return state.load_collection_manifest(
            self.root, path, expected_macro_round=self.config.macro_round
        )

    def ingest_collection(self, collection: Mapping[str, Any]) -> dict[str, Any]:
        campaign = self.load_campaign()
        ledger_path = self.root / "replay" / "replay.sqlite3"
        with replay.ReplayLedger(ledger_path) as ledger:
            for entry in collection["cohorts"]:
                cohort_path = self.root / entry["cohort_manifest_path"]
                report = artifacts.verify_cohort(cohort_path)
                ledger.ingest_cohort(report["manifest_path"])
            output = (
                self.root
                / "replay"
                / "snapshots"
                / f"through_macro_round_{self.config.macro_round:06d}.json"
            )
            return ledger.build_snapshot(
                campaign["campaign_id"],
                through_macro_round=self.config.macro_round,
                output=output,
                expected_train_cohorts=COHORT_COUNT,
            )

    def load_snapshot(self, *, expected_file_sha256: str | None = None) -> dict[str, Any]:
        path = (
            self.root
            / "replay"
            / "snapshots"
            / f"through_macro_round_{self.config.macro_round:06d}.json"
        )
        return replay.load_snapshot(path, expected_file_sha256=expected_file_sha256)

    def mark_ingested(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="COLLECTED",
            next_phase="INGESTED",
            macro_round=self.config.macro_round,
            evidence_path=snapshot["path"],
            evidence_sha256=snapshot["file_sha256"],
            expected_revision=current["revision"],
        )

    def learner_output_dir(self) -> Path:
        return state.round_directory(self.root, self.config.macro_round) / "learner"

    def learner_result_path(self) -> Path:
        return self.learner_output_dir() / (
            f"learner_result_round_{self.config.macro_round:06d}.json"
        )

    def requires_learner_acceptance(self) -> bool:
        """Whether this concrete backend crosses the audited Lepton boundary."""

        return self.config.learner_backend == LEARNER_BACKEND_LEPTON_JOB_V1

    def learner_acceptance_path(self) -> Path:
        if learner_acceptance is None:
            raise OrchestrationError(
                "the deployment-specific remote learner acceptance adapter is "
                "not included in the portable release"
            )
        return learner_acceptance.acceptance_path(
            self.root, self.config.macro_round
        )

    def load_learner_acceptance(
        self, *, expected_environment: Mapping[str, str]
    ) -> dict[str, Any]:
        if not self.requires_learner_acceptance():
            raise OrchestrationError(
                "learner acceptance is defined only for the Lepton backend"
            )
        if learner_acceptance is None:
            raise OrchestrationError(
                "the deployment-specific remote learner acceptance adapter is "
                "not included in the portable release"
            )
        contract_raw = os.environ.get("ROUND_DAGGER_LEPTON_BACKEND_CONTRACT")
        contract_sha = os.environ.get(
            "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256"
        )
        if (
            not contract_raw
            or not Path(contract_raw).is_absolute()
            or self.config.learner_backend_contract_sha256 is None
            or contract_sha != self.config.learner_backend_contract_sha256
        ):
            raise OrchestrationError(
                "Lepton acceptance backend-contract identity is unavailable"
            )
        try:
            validation_environment = dict(os.environ)
            validation_environment.update(expected_environment)
            return learner_acceptance.validate_acceptance(
                self.learner_acceptance_path(),
                campaign_root=self.root,
                macro_round=self.config.macro_round,
                backend_contract_path=Path(os.path.abspath(contract_raw)),
                expected_environment=validation_environment,
            )
        except learner_acceptance.LearnerAcceptanceError as exc:
            raise OrchestrationError(
                f"Lepton learner acceptance is invalid: {exc}"
            ) from exc

    def load_learner_result(
        self,
        *,
        campaign: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self.learner_result_path()
        document = _exact_mapping(
            artifacts.load_strict_json(path), _LEARNER_RESULT_KEYS, "learner result"
        )
        if document["schema"] != "fast_roborender.round_dagger.learner_result.v2":
            raise OrchestrationError("learner result schema mismatch")
        expected = {
            "campaign_id": campaign["campaign_id"],
            "macro_round": self.config.macro_round,
            # The learner records the snapshot semantic digest.  Coordinator
            # state separately binds the immutable snapshot file digest.
            "replay_snapshot_sha256": snapshot["snapshot_sha256"],
            "code_sha256": self.config.learner_code_sha256,
            "learner_config_sha256": self.config.learner_config_sha256,
            "model_lineage_sha256": self.config.learner_model_lineage_sha256,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise OrchestrationError("learner result lineage mismatch")
        checkpoint_relative = _safe_relative(
            document["checkpoint_path"], "learner checkpoint_path"
        )
        if len(checkpoint_relative.parts) != 1:
            raise OrchestrationError("learner checkpoint_path must be one basename")
        checkpoint = path.parent / checkpoint_relative.as_posix()
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise OrchestrationError("learner checkpoint is missing or unsafe")
        if artifacts.sha256_file(checkpoint) != _sha256(
            document["checkpoint_sha256"], "learner checkpoint_sha256"
        ):
            raise OrchestrationError("learner checkpoint SHA-256 mismatch")
        spec_relative = _safe_relative(
            document["policy_spec_path"], "learner policy_spec_path"
        )
        bundle_relative = _safe_relative(
            document["policy_bundle_manifest_path"],
            "learner policy_bundle_manifest_path",
        )
        if len(spec_relative.parts) != 1 or len(bundle_relative.parts) != 1:
            raise OrchestrationError("learner policy sidecars must be one basename")
        policy_spec = path.parent / spec_relative.as_posix()
        policy_bundle_manifest = path.parent / bundle_relative.as_posix()
        if (
            policy_spec != Path(f"{checkpoint}.policy_spec.json")
            or policy_bundle_manifest != Path(f"{checkpoint}.manifest.json")
            or policy_spec.is_symlink()
            or not policy_spec.is_file()
            or policy_bundle_manifest.is_symlink()
            or not policy_bundle_manifest.is_file()
        ):
            raise OrchestrationError(
                "learner policy sidecar is missing, unsafe, or non-canonical"
            )
        if artifacts.sha256_file(policy_spec) != _sha256(
            document["policy_spec_sha256"], "learner policy_spec_sha256"
        ) or artifacts.sha256_file(policy_bundle_manifest) != _sha256(
            document["policy_bundle_manifest_sha256"],
            "learner policy_bundle_manifest_sha256",
        ):
            raise OrchestrationError("learner policy sidecar SHA-256 mismatch")
        runtime_policy_sha = _sha256(
            document["runtime_policy_sha256"], "learner runtime_policy_sha256"
        )
        try:
            policy_bundle.validate_policy_bundle_identity(
                checkpoint=checkpoint,
                policy_spec=policy_spec,
                manifest=policy_bundle_manifest,
                expected_checkpoint_sha256=document["checkpoint_sha256"],
                expected_runtime_policy_sha256=runtime_policy_sha,
            )
        except policy_bundle.PolicyBundleError as exc:
            raise OrchestrationError(
                f"learner policy bundle identity mismatch: {exc}"
            ) from exc
        _true_int(document["optimizer_steps"], "optimizer_steps", minimum=1)
        for key in ("initial_loss", "final_loss"):
            value = document[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OrchestrationError(f"learner {key} must be finite numeric")
            if not math.isfinite(float(value)):
                raise OrchestrationError(f"learner {key} must be finite numeric")
        environment = {
            **document,
            "manifest_path": str(path),
            "manifest_sha256": artifacts.sha256_file(path),
            "absolute_checkpoint_path": str(checkpoint),
            "absolute_policy_spec_path": str(policy_spec),
            "absolute_policy_bundle_manifest_path": str(policy_bundle_manifest),
        }
        return environment

    def stage_candidate(
        self,
        learner_result: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        *,
        learner_acceptance_sha256: str | None = None,
    ) -> dict[str, Any]:
        return promotion.stage_candidate(
            self.root,
            macro_round=self.config.macro_round,
            checkpoint_source=learner_result["absolute_checkpoint_path"],
            replay_snapshot_sha256=snapshot["file_sha256"],
            code_sha256=self.config.learner_code_sha256,
            learner_config_sha256=self.config.learner_config_sha256,
            optimizer_steps=learner_result["optimizer_steps"],
            learner_acceptance_sha256=learner_acceptance_sha256,
        )

    def mark_trained(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        summary = {
            "version": candidate["policy_version"],
            "macro_round": candidate["macro_round"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "checkpoint_path": candidate["checkpoint_path"],
            "runtime_policy_sha256": candidate["runtime_policy_sha256"],
            "policy_spec_sha256": candidate["policy_spec_sha256"],
            "policy_bundle_manifest_sha256": candidate[
                "policy_bundle_manifest_sha256"
            ],
            "manifest_sha256": candidate["manifest_sha256"],
            "manifest_path": candidate["manifest_path"],
            "parent_policy_version": candidate["parent_policy_version"],
            "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
            "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
            "code_sha256": candidate["code_sha256"],
            "policy_config_sha256": candidate["learner_config_sha256"],
        }
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="INGESTED",
            next_phase="TRAINED",
            macro_round=self.config.macro_round,
            evidence_path=candidate["manifest_path"],
            evidence_sha256=candidate["manifest_sha256"],
            candidate_policy=summary,
            expected_revision=current["revision"],
        )

    def load_candidate(self) -> dict[str, Any]:
        current = self.load_state()
        candidate = current.get("candidate_policy")
        if not isinstance(candidate, Mapping):
            raise OrchestrationError("coordinator has no staged candidate")
        return promotion.load_candidate_manifest(
            self.root,
            candidate["manifest_path"],
            expected_macro_round=self.config.macro_round,
            bind_state=True,
        )

    def evaluator_result_path(self) -> Path:
        return (
            state.round_directory(self.root, self.config.macro_round)
            / "evaluation"
            / "result.json"
        )

    def load_evaluator_result(
        self, *, campaign: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> dict[str, Any]:
        path = self.evaluator_result_path()
        result = _exact_mapping(
            artifacts.load_strict_json(path), _EVALUATOR_RESULT_KEYS, "evaluator result"
        )
        expected = {
            "schema": EVALUATOR_RESULT_SCHEMA,
            "campaign_id": campaign["campaign_id"],
            "macro_round": self.config.macro_round,
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "evaluator_config_sha256": self.config.evaluator_config_sha256,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise OrchestrationError("evaluator result lineage mismatch")
        if not isinstance(result["evidence_bundle_path"], str):
            raise OrchestrationError("evaluation evidence bundle path is unsafe")
        expected_bundle = path.parent / "results" / "evaluation_bundle.json"
        try:
            bundle_path, bundle_relative = state._internal_path(
                self.root, result["evidence_bundle_path"], "evaluation evidence bundle"
            )
        except (artifacts.ArtifactError, OSError, ValueError) as exc:
            raise OrchestrationError("evaluation evidence bundle path is unsafe") from exc
        if (
            bundle_path != expected_bundle
            or result["evidence_bundle_path"] != bundle_relative
        ):
            raise OrchestrationError("evaluation evidence bundle path is non-canonical")
        if (
            not isinstance(result["evidence_bundle_sha256"], str)
            or _SHA256_RE.fullmatch(result["evidence_bundle_sha256"]) is None
        ):
            raise OrchestrationError("evaluation evidence bundle SHA is invalid")
        try:
            actual_bundle_sha = artifacts.sha256_file(bundle_path)
        except (artifacts.ArtifactError, OSError) as exc:
            raise OrchestrationError("evaluation evidence bundle is missing or unsafe") from exc
        if actual_bundle_sha != result["evidence_bundle_sha256"]:
            raise OrchestrationError("evaluation evidence bundle SHA mismatch")
        if type(result["passed"]) is not bool:
            raise OrchestrationError("evaluator result passed must be boolean")
        metrics = result["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise OrchestrationError("evaluator result metrics must be non-empty")
        clean_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise OrchestrationError("evaluation metric keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OrchestrationError("evaluation metrics must be finite numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise OrchestrationError("evaluation metrics must be finite numeric")
            clean_metrics[key] = numeric
        # Recompute the paired gate from every canonical outcome and its raw
        # source artifacts.  This check intentionally runs even when a result
        # existed before the evaluator hook, preventing a digest-matching but
        # semantically opaque bundle from advancing coordinator state.
        try:
            verified = evaluation.verify_evaluator_result_evidence(
                self.root,
                contract_path=path.parent / "contract.json",
                results_dir=path.parent / "results",
                result_path=path,
            )
        except (evaluation.EvaluationError, artifacts.ArtifactError, OSError) as exc:
            raise OrchestrationError(
                f"evaluator result failed deep evidence validation: {exc}"
            ) from exc
        if verified["metrics"] != clean_metrics or verified["passed"] != result["passed"]:
            raise OrchestrationError("deep evaluator result differs from loaded result")
        return {**result, "metrics": clean_metrics, "path": str(path)}

    def publish_evaluation(
        self,
        candidate: Mapping[str, Any],
        evaluator_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return promotion.write_evaluation(
            self.root,
            candidate_manifest=candidate["manifest_path"],
            passed=evaluator_result["passed"],
            metrics=evaluator_result["metrics"],
            evaluator_config_sha256=self.config.evaluator_config_sha256,
            evidence_bundle_path=evaluator_result["evidence_bundle_path"],
            evidence_bundle_sha256=evaluator_result["evidence_bundle_sha256"],
        )

    def mark_evaluated(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        current = self.load_state()
        return state.transition(
            self.root,
            expected_phase="TRAINED",
            next_phase="EVALUATED",
            macro_round=self.config.macro_round,
            evidence_path=evidence["manifest_path"],
            evidence_sha256=evidence["evidence_sha256"],
            expected_revision=current["revision"],
        )

    def load_evaluation(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        path = (
            self.root
            / "policies"
            / f"policy_{candidate['policy_version']:06d}"
            / "evaluation.json"
        )
        return promotion.load_evaluation_manifest(
            self.root, path, candidate=candidate
        )

    def promote(
        self, candidate: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> dict[str, Any]:
        return promotion.promote_candidate(
            self.root,
            macro_round=self.config.macro_round,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )

    def reject(
        self, candidate: Mapping[str, Any], evaluation: Mapping[str, Any]
    ) -> dict[str, Any]:
        return promotion.reject_candidate(
            self.root,
            macro_round=self.config.macro_round,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )


def _cohort_environment(
    *,
    config: MacroRoundConfig,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    cohort_index: int,
    source: Path,
    destination: Path,
    actor_run_dir: Path,
) -> dict[str, str]:
    offset = cohort_index * ENVIRONMENTS_PER_COHORT
    checkpoint = config.campaign_root / contract["policy_checkpoint_path"]
    return {
        "ROUND_DAGGER_CAMPAIGN_ROOT": str(config.campaign_root),
        "ROUND_DAGGER_CAMPAIGN_ID": str(campaign["campaign_id"]),
        "ROUND_DAGGER_MACRO_ROUND": str(config.macro_round),
        "ROUND_DAGGER_COHORT_INDEX": str(cohort_index),
        "ROUND_DAGGER_COHORT_DIR": str(source),
        "ROUND_DAGGER_SOURCE_COHORT": str(source),
        "ROUND_DAGGER_DESTINATION_COHORT": str(destination),
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(offset),
        "ROUND_DAGGER_LOGICAL_ENV_COUNT": str(ENVIRONMENTS_PER_COHORT),
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_EXPECTED_TRANSITIONS": str(TRANSITIONS_PER_COHORT),
        "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
        "ROUND_DAGGER_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
        "ROUND_DAGGER_POLICY_MANIFEST": f"{checkpoint}.manifest.json",
        "ROUND_DAGGER_POLICY_SPEC_SHA256": contract["policy_spec_sha256"],
        "ROUND_DAGGER_POLICY_MANIFEST_SHA256": contract[
            "policy_bundle_manifest_sha256"
        ],
        "ROUND_DAGGER_POLICY_SHA256": str(contract["policy_checkpoint_sha256"]),
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": contract["policy_runtime_sha256"],
        "ROUND_DAGGER_TEACHER_SHA256": str(contract["teacher_sha256"]),
        "ROUND_DAGGER_RENDERER_SHA256": str(contract["renderer_sha256"]),
        "ROUND_DAGGER_BETA": repr(float(contract["beta"])),
        "ROUND_DAGGER_BETA_SHA256": str(contract["beta_sha256"]),
        "ROUND_DAGGER_CONFIG_SHA256": str(contract["training_config_sha256"]),
        "ROUND_DAGGER_ACTOR_RUN_DIR": str(actor_run_dir),
        "TRAIN_RUN_DIR": str(actor_run_dir),
        "ROBORENDER_TELEMETRY_DIR": str(actor_run_dir / "telemetry"),
        "ROLLOUT_STEPS": str(STEPS_PER_ENVIRONMENT),
    }


def _validate_cohort_report(
    report: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    contract: Mapping[str, Any],
    config: MacroRoundConfig,
    cohort_index: int,
) -> None:
    start = cohort_index * ENVIRONMENTS_PER_COHORT
    expected_global_env_ids = list(
        range(start, start + ENVIRONMENTS_PER_COHORT)
    )
    expected_seed_contract = seed_policy.derive_actor_seed_contract(
        campaign_id=campaign["campaign_id"],
        macro_round=config.macro_round,
        cohort_index=cohort_index,
        expected_ranks=EXPECTED_ACTOR_RANKS,
        expected_global_env_ids=expected_global_env_ids,
    )
    expected = {
        "campaign_id": campaign["campaign_id"],
        "macro_round": config.macro_round,
        "cohort_index": cohort_index,
        "purpose": "train",
        "sealed": True,
        "expected_ranks": list(EXPECTED_ACTOR_RANKS),
        "ranks": list(EXPECTED_ACTOR_RANKS),
        "expected_global_env_ids": expected_global_env_ids,
        "global_env_ids": expected_global_env_ids,
        "expected_transitions": TRANSITIONS_PER_COHORT,
        "transition_count": TRANSITIONS_PER_COHORT,
        "policy_sha256": contract["policy_checkpoint_sha256"],
        "runtime_policy_sha256": contract["policy_runtime_sha256"],
        "teacher_sha256": contract["teacher_sha256"],
        "renderer_sha256": contract["renderer_sha256"],
        "beta_sha256": contract["beta_sha256"],
        "config_sha256": contract["training_config_sha256"],
        "actor_seed_contract": expected_seed_contract,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise OrchestrationError(
            f"cohort {cohort_index} failed exact production postconditions: {mismatches}"
        )
    fingerprints = report.get(
        "physical_teacher_trajectory_sha256_by_global_env"
    )
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints) != {str(item) for item in expected_global_env_ids}
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in fingerprints.values()
        )
    ):
        raise OrchestrationError(
            f"cohort {cohort_index} lacks exact physical/teacher trajectory provenance"
        )


def _ensure_scene_sample_manifest(
    *,
    actor_run_dir: Path,
    round_dir: Path,
    campaign_id: str,
) -> Path:
    """Package one immutable, bounded actor contact-sheet set for the learner.

    Actor telemetry lives below the top-level ``actors`` tree, which is
    deliberately excluded from the H200 campaign projection.  Copying only a
    SHA-pinned sample set into the durable round directory gives the learner
    its W&B scene panel without transferring the full rollout-media tree.
    Publication is idempotent, so an INGESTED-phase restart can rebuild or
    revalidate this handoff without touching replay or rerunning an actor.
    """

    try:
        from scripts.round_dagger import build_scene_sample_manifest

        destination = round_dir / "scene_samples"
        expected = build_scene_sample_manifest.package_scene_samples(
            actor_telemetry_dir=actor_run_dir / "telemetry",
            destination=destination,
            campaign_id=campaign_id,
            actor_rank=0,
            rollout_steps=STEPS_PER_ENVIRONMENT,
        )
        manifest_path = destination / build_scene_sample_manifest.MANIFEST_NAME
        observed = artifacts.load_strict_json(manifest_path)
    except Exception as exc:
        raise OrchestrationError(
            "cannot package the immutable actor scene samples required by the "
            "learner dashboard"
        ) from exc
    if observed != expected:
        raise OrchestrationError("actor scene sample manifest changed after publication")
    return manifest_path


class MacroRoundOrchestrator:
    """Run or resume one explicitly numbered production macro-round."""

    def __init__(
        self,
        config: MacroRoundConfig,
        *,
        backend: ProductionBackend | None = None,
        hook_runner: HookRunner | None = None,
    ):
        self.config = config
        self.backend = backend or ProductionBackend(config)
        self.hook_runner = hook_runner or SubprocessHookRunner()

    def _run_hook(self, name: str, environment: Mapping[str, str]) -> None:
        hook = self.config.hooks[name]
        if hook is None:
            return
        before = self.backend.load_state()
        self.hook_runner.run(
            name,
            hook,
            environment=environment,
            workdir=self.config.campaign_root,
        )
        after = self.backend.load_state()
        if after != before:
            raise OrchestrationError(
                f"{name} hook mutated coordinator state; hooks may only produce artifacts"
            )

    def _base_environment(
        self,
        *,
        campaign: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> dict[str, str]:
        checkpoint = self.config.campaign_root / contract["policy_checkpoint_path"]
        environment = {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.config.campaign_root),
            "ROUND_DAGGER_CAMPAIGN_ID": str(campaign["campaign_id"]),
            "ROUND_DAGGER_MACRO_ROUND": str(self.config.macro_round),
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
            "ROUND_DAGGER_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
            "ROUND_DAGGER_POLICY_MANIFEST": f"{checkpoint}.manifest.json",
            "ROUND_DAGGER_BASE_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
            "ROUND_DAGGER_POLICY_SPEC_SHA256": contract["policy_spec_sha256"],
            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": contract[
                "policy_bundle_manifest_sha256"
            ],
            "ROUND_DAGGER_POLICY_SHA256": str(contract["policy_checkpoint_sha256"]),
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": contract[
                "policy_runtime_sha256"
            ],
            "ROUND_DAGGER_BETA": repr(float(contract["beta"])),
            "ROUND_DAGGER_BETA_SHA256": str(contract["beta_sha256"]),
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": str(TARGET_ENVIRONMENTS),
            "ROUND_DAGGER_COHORT_COUNT": str(COHORT_COUNT),
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": str(ENVIRONMENTS_PER_COHORT),
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": str(
                TRANSITIONS_PER_MACRO_ROUND
            ),
            "ROUND_DAGGER_MACRO_ROUND_COUNT": str(
                PRODUCTION_MACRO_ROUND_COUNT
            ),
            "ROUND_DAGGER_TRANSITIONS_PER_CAMPAIGN": str(
                TRANSITIONS_PER_CAMPAIGN
            ),
            "ROUND_DAGGER_LEARNER_BACKEND": self.config.learner_backend,
        }
        if self.config.campaign_contract_sha256 is not None:
            runtime_contract_path = os.environ.get(
                "ROUND_DAGGER_CAMPAIGN_CONTRACT"
            )
            if not runtime_contract_path:
                raise OrchestrationError(
                    "external campaign-contract path is missing"
                )
            environment.update(
                ROUND_DAGGER_CAMPAIGN_CONTRACT=runtime_contract_path,
                ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256=(
                    self.config.campaign_contract_sha256
                ),
                ROUND_DAGGER_DATA_LINEAGE_SHA256=contract[
                    "data_lineage_sha256"
                ],
            )
        return environment

    def _completed_retry(
        self, current: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if current.get("phase") != "READY":
            return None
        next_round = current.get("next_macro_round")
        if next_round == self.config.macro_round:
            return None
        if next_round != self.config.macro_round + 1:
            raise OrchestrationError(
                "configured macro_round is stale or skips coordinator state"
            )
        active = current["active_policy"]
        resolution = current.get("last_resolution")
        if (
            isinstance(resolution, Mapping)
            and resolution.get("macro_round") == self.config.macro_round
        ):
            version = resolution.get("candidate_policy_version")
            if type(version) is not int or version < 1:
                raise OrchestrationError("completed round has invalid resolution version")
        else:
            # Legacy V2 campaigns recorded a successful promotion only; the
            # active policy directory therefore remains the compatibility
            # source for its candidate manifest.
            version = active["version"]
        candidate_path = (
            self.config.campaign_root
            / "policies"
            / f"policy_{version:06d}"
            / "candidate.json"
        )
        candidate = promotion.load_candidate_manifest(
            self.config.campaign_root,
            candidate_path,
            expected_macro_round=self.config.macro_round,
            bind_state=False,
        )
        evaluation = self.backend.load_evaluation(candidate)
        if evaluation.get("passed") is True:
            return self.backend.promote(candidate, evaluation)
        if evaluation.get("passed") is False:
            return self.backend.reject(candidate, evaluation)
        raise OrchestrationError("completed round evaluation has an invalid passed flag")

    def run(self) -> dict[str, Any]:
        try:
            with campaign_mutation_lease(
                self.config.campaign_root,
                f"macro_round_{self.config.macro_round:06d}",
            ):
                return self._run_with_mutation_lease()
        except CampaignMutationLockError as exc:
            raise OrchestrationError(str(exc)) from exc

    def _run_with_mutation_lease(self) -> dict[str, Any]:
        campaign = self.backend.load_campaign()
        validate_production_topology(campaign)
        if campaign.get("schema") == state.PRODUCTION_CAMPAIGN_SCHEMA:
            if self.config.campaign_contract_sha256 is None:
                raise OrchestrationError(
                    "production orchestration requires an external campaign-contract SHA-256"
                )
            lineage = campaign.get("data_lineage")
            if (
                not isinstance(lineage, Mapping)
                or lineage.get("runtime_contract_sha256")
                != self.config.campaign_contract_sha256
            ):
                raise OrchestrationError(
                    "external campaign contract differs from durable campaign lineage"
                )
            runtime_contract_path = os.environ.get(
                "ROUND_DAGGER_CAMPAIGN_CONTRACT"
            )
            runtime_contract_sha = os.environ.get(
                "ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256"
            )
            if (
                not isinstance(runtime_contract_path, str)
                or not Path(runtime_contract_path).is_absolute()
                or runtime_contract_sha != self.config.campaign_contract_sha256
            ):
                raise OrchestrationError(
                    "validated external campaign-contract handoff is missing"
                )
            try:
                observed_runtime_contract_sha = artifacts.sha256_file(
                    runtime_contract_path
                )
            except Exception as exc:
                raise OrchestrationError(
                    "external campaign contract is unavailable"
                ) from exc
            if observed_runtime_contract_sha != self.config.campaign_contract_sha256:
                raise OrchestrationError(
                    "external campaign contract changed after launch preflight"
                )
        if self.config.macro_round_count != PRODUCTION_MACRO_ROUND_COUNT:
            raise OrchestrationError(
                "production orchestration config must bind exactly five macro-rounds"
            )
        current = self.backend.load_state()
        completed = self._completed_retry(current)
        if completed is not None:
            return completed

        if current["phase"] == "READY":
            if current["next_macro_round"] != self.config.macro_round:
                raise OrchestrationError("configured macro_round is not the next durable round")
            contract = self.backend.open_round()
        else:
            if current.get("open_macro_round") != self.config.macro_round:
                raise OrchestrationError("another macro-round is already open")
            contract = self.backend.load_contract()
        validate_production_topology(campaign, contract)
        if self.config.runtime_policy_sha256 != contract["policy_runtime_sha256"]:
            raise OrchestrationError(
                "orchestration runtime policy SHA-256 does not match the frozen round contract"
            )
        if (
            self.config.learner_model_lineage_sha256
            != contract["policy_checkpoint_sha256"]
        ):
            raise OrchestrationError(
                "learner model lineage does not match the frozen parent checkpoint"
            )

        round_dir = state.round_directory(
            self.config.campaign_root, self.config.macro_round
        )
        telemetry = RoundTelemetry(round_dir / "orchestration_run")
        telemetry.publish_config(
            _public_wandb_config(
                config=self.config,
                campaign=campaign,
                contract=contract,
                cohort_index=None,
            )
        )
        phase = self.backend.load_state()["phase"]
        telemetry.record(
            step=self.config.macro_round * 100,
            metrics={
                "round/macro_round": self.config.macro_round,
                # This is the immutable round-open baseline.  Keeping its
                # payload independent of the resume phase makes retries
                # byte-identical after a later phase has already committed.
                "round/phase_code": _PHASE_CODES["COLLECTING"],
                "round/cohorts_complete": 0,
                "round/logical_environments_complete": 0,
                "round/transitions_complete": 0,
                "round/beta": float(contract["beta"]),
                "round/policy_version": int(contract["policy_version"]),
            },
        )

        base_environment = self._base_environment(campaign=campaign, contract=contract)
        if phase == "COLLECTING":
            accepted_paths: list[Path] = []
            actor_round = (
                self.config.actor_root
                / str(campaign["campaign_id"])
                / f"macro_round_{self.config.macro_round:06d}"
            )
            accepted_round = round_dir / "cohorts"
            for cohort_index in range(COHORT_COUNT):
                source = actor_round / f"cohort_{cohort_index:02d}"
                actor_run_dir = actor_round / f"cohort_{cohort_index:02d}_run"
                destination = accepted_round / f"cohort_{cohort_index:02d}"
                self.backend.ensure_cohort_contract(
                    source,
                    campaign=campaign,
                    contract=contract,
                    cohort_index=cohort_index,
                )
                RoundTelemetry(actor_run_dir).publish_config(
                    _public_wandb_config(
                        config=self.config,
                        campaign=campaign,
                        contract=contract,
                        cohort_index=cohort_index,
                    )
                )
                environment = {
                    **base_environment,
                    **_cohort_environment(
                        config=self.config,
                        campaign=campaign,
                        contract=contract,
                        cohort_index=cohort_index,
                        source=source,
                        destination=destination,
                        actor_run_dir=actor_run_dir,
                    ),
                }
                source_report = self.backend.complete_cohort(source)
                if source_report is None:
                    self._run_hook("collect", environment)
                    source_report = self.backend.seal_cohort(source)
                _validate_cohort_report(
                    source_report,
                    campaign=campaign,
                    contract=contract,
                    config=self.config,
                    cohort_index=cohort_index,
                )

                accepted_report = self.backend.complete_cohort(destination)
                if accepted_report is None:
                    if self.config.hooks["transfer"] is None:
                        accepted_report = self.backend.transfer_local(source, destination)
                    else:
                        self._run_hook("transfer", environment)
                        accepted_report = self.backend.complete_cohort(destination)
                        if accepted_report is None:
                            raise OrchestrationError(
                                "transfer hook returned without publishing a sealed destination"
                            )
                _validate_cohort_report(
                    accepted_report,
                    campaign=campaign,
                    contract=contract,
                    config=self.config,
                    cohort_index=cohort_index,
                )
                if accepted_report["cohort_sha256"] != source_report["cohort_sha256"]:
                    raise OrchestrationError("transferred cohort digest differs from source")
                accepted_paths.append(destination)
                completed_cohorts = cohort_index + 1
                telemetry.record(
                    step=self.config.macro_round * 100 + completed_cohorts,
                    metrics={
                        "round/macro_round": self.config.macro_round,
                        "round/phase_code": _PHASE_CODES["COLLECTING"],
                        "round/cohort_index": cohort_index,
                        "round/cohorts_complete": completed_cohorts,
                        "round/logical_environments_complete": completed_cohorts
                        * ENVIRONMENTS_PER_COHORT,
                        "round/transitions_complete": completed_cohorts
                        * TRANSITIONS_PER_COHORT,
                    },
                )

            collection = self.backend.seal_collection(accepted_paths)
            if collection.get("cohort_count") != COHORT_COUNT or collection.get(
                "transition_count"
            ) != TRANSITIONS_PER_MACRO_ROUND:
                raise OrchestrationError("immutable collection aggregate is not exactly 256x512")
            self.backend.mark_collected(collection)
            telemetry.record(
                step=self.config.macro_round * 100 + 10,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["COLLECTED"],
                    "round/cohorts_complete": COHORT_COUNT,
                    "round/logical_environments_complete": TARGET_ENVIRONMENTS,
                    "round/transitions_complete": TRANSITIONS_PER_MACRO_ROUND,
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "COLLECTED":
            collection = self.backend.load_collection()
            ingest_environment = {
                **base_environment,
                "ROUND_DAGGER_COLLECTION_MANIFEST": str(
                    self.config.campaign_root / collection["manifest_path"]
                ),
                "ROUND_DAGGER_REPLAY_LEDGER": str(
                    self.config.campaign_root / "replay" / "replay.sqlite3"
                ),
            }
            self._run_hook("ingest", ingest_environment)
            snapshot = self.backend.ingest_collection(collection)
            expected_cumulative = (self.config.macro_round + 1) * TRANSITIONS_PER_MACRO_ROUND
            if snapshot.get("expected_train_cohorts") != COHORT_COUNT or snapshot.get(
                "transition_count"
            ) != expected_cumulative:
                raise OrchestrationError("cumulative replay snapshot coverage is incomplete")
            self.backend.mark_ingested(snapshot)
            telemetry.record(
                step=self.config.macro_round * 100 + 20,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["INGESTED"],
                    "round/replay_transitions": int(snapshot["transition_count"]),
                    "round/replay_cohorts": int(snapshot["cohort_count"]),
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "INGESTED":
            snapshot = self.backend.load_snapshot(
                expected_file_sha256=current["replay_snapshot_sha256"]
            )
            learner_output = self.backend.learner_output_dir()
            # Production launch exports the immutable runtime/container
            # contract.  Generic local hooks do not have a container concept,
            # so only attach the production learner identity when both inputs
            # are present; the OCI bridge itself requires them and fails closed.
            container_sha = os.environ.get(
                "ROUND_DAGGER_REMOTE_CONTAINER_SHA256"
            )
            learner_contract_sha = os.environ.get(
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
            )
            identity_environment: dict[str, str] = {}
            if container_sha is not None or learner_contract_sha is not None:
                if container_sha is None or learner_contract_sha is None:
                    raise OrchestrationError(
                        "production learner identity inputs are incomplete"
                    )
                container_sha = _sha256(container_sha, "runtime container SHA-256")
                learner_contract_sha = _sha256(
                    learner_contract_sha, "learner contract SHA-256"
                )
                if learner_contract_sha != campaign["policy_config_sha256"]:
                    raise OrchestrationError(
                        "learner contract identity differs from the campaign"
                    )
                learner_identity_sha256 = learner_identity.learner_identity_sha256(
                    campaign_id=str(campaign["campaign_id"]),
                    macro_round=self.config.macro_round,
                    learner_code_sha256=self.config.learner_code_sha256,
                    learner_config_sha256=self.config.learner_config_sha256,
                    replay_snapshot_file_sha256=str(snapshot["file_sha256"]),
                    replay_snapshot_sha256=str(snapshot["snapshot_sha256"]),
                    parent_checkpoint_sha256=self.config.learner_model_lineage_sha256,
                    parent_runtime_policy_sha256=str(
                        contract["policy_runtime_sha256"]
                    ),
                    runtime_container_sha256=container_sha,
                    learner_contract_sha256=learner_contract_sha,
                )
                identity_environment = {
                    "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": container_sha,
                    "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": learner_contract_sha,
                    "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": learner_identity_sha256,
                }
            train_environment = {
                **base_environment,
                **identity_environment,
                "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot["path"]),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": str(
                    snapshot["file_sha256"]
                ),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": str(
                    snapshot["snapshot_sha256"]
                ),
                "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(learner_output),
                "ROUND_DAGGER_LEARNER_RESULT": str(
                    self.backend.learner_result_path()
                ),
                "ROUND_DAGGER_LEARNER_CODE_SHA256": self.config.learner_code_sha256,
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256": self.config.learner_config_sha256,
                "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": self.config.learner_model_lineage_sha256,
            }
            if os.environ.get("ROUND_DAGGER_WANDB_PROJECT"):
                actor_round = (
                    self.config.actor_root
                    / str(campaign["campaign_id"])
                    / f"macro_round_{self.config.macro_round:06d}"
                )
                scene_sample_manifest = _ensure_scene_sample_manifest(
                    actor_run_dir=actor_round / "cohort_00_run",
                    round_dir=round_dir,
                    campaign_id=str(campaign["campaign_id"]),
                )
                train_environment["ROUND_DAGGER_SCENE_SAMPLE_MANIFEST"] = str(
                    scene_sample_manifest
                )
            requires_acceptance = bool(
                getattr(
                    self.backend,
                    "requires_learner_acceptance",
                    lambda: False,
                )()
            )
            acceptance_sha256: str | None = None
            if requires_acceptance:
                acceptance_path = self.backend.learner_acceptance_path()
                # A result alone is not a transaction commit.  Re-enter the
                # idempotent bridge whenever its campaign-local acceptance
                # receipt is absent, then deeply validate the receipt before
                # any candidate bytes can be staged.
                if not os.path.lexists(acceptance_path):
                    self._run_hook("train", train_environment)
                self.backend.load_learner_acceptance(
                    expected_environment=train_environment
                )
                acceptance_sha256 = artifacts.sha256_file(acceptance_path)
            elif not self.backend.learner_result_path().exists():
                # Generic/local and legacy OCI backends retain their existing
                # result-gated compatibility contract.
                self._run_hook("train", train_environment)
            learner_result = self.backend.load_learner_result(
                campaign=campaign, snapshot=snapshot
            )
            if acceptance_sha256 is None:
                candidate = self.backend.stage_candidate(learner_result, snapshot)
            else:
                candidate = self.backend.stage_candidate(
                    learner_result,
                    snapshot,
                    learner_acceptance_sha256=acceptance_sha256,
                )
            self.backend.mark_trained(candidate)
            telemetry.record(
                step=self.config.macro_round * 100 + 30,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["TRAINED"],
                    "round/policy_version": int(candidate["policy_version"]),
                    "round/optimizer_steps": int(candidate["optimizer_steps"]),
                    "round/initial_loss": float(learner_result["initial_loss"]),
                    "round/final_loss": float(learner_result["final_loss"]),
                },
            )

        current = self.backend.load_state()
        if current["phase"] == "TRAINED":
            candidate = self.backend.load_candidate()
            evaluator_result_path = self.backend.evaluator_result_path()
            evaluate_environment = {
                **base_environment,
                "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                    self.config.campaign_root / candidate["manifest_path"]
                ),
                "ROUND_DAGGER_CANDIDATE_SHA256": str(candidate["manifest_sha256"]),
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.config.evaluator_config_sha256,
                "ROUND_DAGGER_EVALUATOR_RESULT": str(evaluator_result_path),
            }
            if not evaluator_result_path.exists():
                self._run_hook("evaluate", evaluate_environment)
            evaluator_result = self.backend.load_evaluator_result(
                campaign=campaign, candidate=candidate
            )
            evidence = self.backend.publish_evaluation(candidate, evaluator_result)
            self.backend.mark_evaluated(evidence)
            telemetry.record(
                step=self.config.macro_round * 100 + 40,
                metrics={
                    "round/macro_round": self.config.macro_round,
                    "round/phase_code": _PHASE_CODES["EVALUATED"],
                    "round/evaluation_passed": int(evaluator_result["passed"]),
                    **{
                        f"evaluation/{key}": value
                        for key, value in evaluator_result["metrics"].items()
                    },
                },
            )

        current = self.backend.load_state()
        if current["phase"] != "EVALUATED":
            raise OrchestrationError(
                f"macro-round stopped in unexpected phase {current['phase']}"
            )
        candidate = self.backend.load_candidate()
        evaluation = self.backend.load_evaluation(candidate)
        passed = evaluation.get("passed")
        if type(passed) is not bool:
            raise OrchestrationError("candidate evaluation passed flag is invalid")
        parent = dict(current["active_policy"])
        if passed:
            promote_environment = {
                **base_environment,
                "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                    self.config.campaign_root / candidate["manifest_path"]
                ),
                "ROUND_DAGGER_EVALUATION_MANIFEST": str(
                    self.config.campaign_root / evaluation["manifest_path"]
                ),
                "ROUND_DAGGER_EVALUATION_SHA256": str(
                    evaluation["evidence_sha256"]
                ),
            }
            self._run_hook("promote", promote_environment)
            result = self.backend.promote(candidate, evaluation)
            expected_active = {
                "version": candidate["policy_version"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
            }
            decision = "promoted"
        else:
            # No external promotion hook is allowed for a rejected candidate.
            # Legacy configurations remain fail-closed at EVALUATED.  The
            # unattended production schema must opt in explicitly before it
            # can publish a rejection receipt and advance with the parent.
            if not self.config.advance_after_rejected_evaluation:
                raise OrchestrationError(
                    "candidate failed paired evaluation; promotion hook was not invoked"
                )
            # The coordinator publishes a terminal rejection receipt and
            # deterministically advances with the exact verified parent.
            result = self.backend.reject(candidate, evaluation)
            expected_active = {
                "version": parent["version"],
                "checkpoint_sha256": parent["checkpoint_sha256"],
            }
            decision = "rejected"
        final_state = self.backend.load_state()
        if (
            final_state.get("phase") != "READY"
            or final_state.get("next_macro_round") != self.config.macro_round + 1
            or final_state.get("active_policy", {}).get("version")
            != expected_active["version"]
            or final_state.get("active_policy", {}).get("checkpoint_sha256")
            != expected_active["checkpoint_sha256"]
        ):
            raise OrchestrationError("terminal resolution did not atomically advance")
        resolution = final_state.get("last_resolution")
        if self.config.advance_after_rejected_evaluation:
            if not isinstance(resolution, Mapping):
                raise OrchestrationError("terminal resolution lacks an immutable receipt")
            if (
                resolution.get("decision") != decision
                or resolution.get("candidate_manifest_sha256")
                != candidate["manifest_sha256"]
                or resolution.get("evaluation_sha256")
                != evaluation["evidence_sha256"]
            ):
                raise OrchestrationError(
                    "terminal resolution receipt does not match evidence"
                )
        telemetry.record(
            step=self.config.macro_round * 100 + 50,
            metrics={
                "round/macro_round": self.config.macro_round,
                "round/phase_code": _PHASE_CODES["READY"],
                "round/evaluation_passed": int(passed),
                "round/candidate_promoted": int(passed),
                "round/candidate_rejected": int(not passed),
                "round/policy_version": int(expected_active["version"]),
                "round/candidate_policy_version": int(candidate["policy_version"]),
                "round/cohorts_complete": COHORT_COUNT,
                "round/logical_environments_complete": TARGET_ENVIRONMENTS,
                "round/transitions_complete": TRANSITIONS_PER_MACRO_ROUND,
            },
        )
        return result


__all__ = [
    "ORCHESTRATION_CONFIG_SCHEMA",
    "BOUNDED_ORCHESTRATION_CONFIG_SCHEMA",
    "PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA",
    "BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA",
    "RESILIENT_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA",
    "LEARNER_BACKEND_OCI_NRT",
    "LEARNER_BACKEND_LEPTON_JOB_V1",
    "LEARNER_BACKENDS",
    "EVALUATOR_RESULT_SCHEMA",
    "WANDB_PUBLIC_CONFIG_SCHEMA",
    "PRODUCTION_WANDB_PUBLIC_CONFIG_SCHEMA",
    "WANDB_PUBLIC_CONFIG_RECOVERY_SCHEMA",
    "COHORT_COUNT",
    "ENVIRONMENTS_PER_COHORT",
    "TARGET_ENVIRONMENTS",
    "STEPS_PER_ENVIRONMENT",
    "TRANSITIONS_PER_COHORT",
    "TRANSITIONS_PER_MACRO_ROUND",
    "PRODUCTION_MACRO_ROUND_COUNT",
    "TRANSITIONS_PER_CAMPAIGN",
    "EXPECTED_ACTOR_RANKS",
    "OrchestrationError",
    "CampaignMutationLockError",
    "CampaignMutationBusyError",
    "campaign_mutation_lease",
    "campaign_mutation_guard",
    "HookSpec",
    "MacroRoundConfig",
    "validate_learner_backend",
    "trusted_train_hook_relative",
    "HookRunner",
    "SubprocessHookRunner",
    "RoundTelemetry",
    "ProductionBackend",
    "validate_production_topology",
    "MacroRoundOrchestrator",
]
