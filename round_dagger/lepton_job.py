"""Fail-closed Lepton Job backend for the eight-H200 DAgger learner.

The actor/renderer remains on the RTX workstation.  This module owns only the
camera-free learner allocation on Lepton.  It deliberately keeps the existing
OCI/Slurm bridge independent: no OCI symbols or defaults are imported here.

Two immutable JSON documents are used at the boundary:

* a deployment contract pins the reservation, node group, image digest,
  storage mount, W&B secret *reference*, and retry policy;
* a run request pins the exact learner identity and the public environment
  consumed by ``run_h100_learner.sh``.

Lepton/Kubernetes retry counters are zero.  Semantic retry/adoption is owned
by the coordinator bridge so every retry retains one learner identity, one
restart frontier, and one W&B run ID.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import tempfile
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import fcntl


BACKEND = "lepton_job_v1"
CONTRACT_SCHEMA = "fast_roborender.round_dagger.lepton_job_contract.v2"
REQUEST_SCHEMA = "fast_roborender.round_dagger.lepton_learner_request.v1"
CHECKSUM_SCHEMA = "sha256sum-v1"
PROJECT_ROOT = PurePosixPath("/workspace/projects/roborender")
WORKSPACE_MOUNT = PurePosixPath("/workspace")
WORKSPACE_SOURCE = PurePosixPath("/users/ldu")
RESOURCE_SHAPE = "my.gpu.8xh200"
RESERVATION_ID = "cosmos-solution-architect-xcwtffiq"
NODE_GROUP = "neb-cdg-lepton-001-trrjqlvo"
WANDB_SECRET_NAME = "LEO_ROBORENDER_WANDB_API_KEY.ldu"
WANDB_SECRET_ENV = "WANDB_API_KEY"
CONTROL_DEPLOYMENT = "leo-cosmos-h200-dev"
GPU_COUNT = 8
WORKERS = 1
SEMANTIC_MAX_ATTEMPTS = 3
EXCLUSIVE_LIFECYCLE_SCHEMA = (
    "fast_roborender.round_dagger.lepton_exclusive_node_lifecycle.v1"
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_JOB_RE = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")
_ENV_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_IMAGE_RE = re.compile(
    r"[A-Za-z0-9._/-]+(?:\:[A-Za-z0-9._-]+)?@sha256:([0-9a-f]{64})"
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:api[_-]?key|authorization|credential|password|secret|token)"
)
_WANDB_VALUE_RE = re.compile(r"wandb_v1_[A-Za-z0-9_-]+")

_CONTRACT_KEYS = {
    "schema",
    "backend",
    "cluster",
    "allocation",
    "container",
    "supply_chain",
    "storage",
    "wandb",
    "orchestration",
}
_CLUSTER_KEYS = {"workspace", "reservation", "node_group", "resource_shape"}
_ALLOCATION_KEYS = {"workers", "gpus_per_worker", "shared_memory_mib"}
_CONTAINER_KEYS = {"image", "python", "entrypoint"}
_SUPPLY_CHAIN_KEYS = {
    "source_root",
    "source_inventory",
    "source_inventory_sha256",
    "source_verifier",
    "python_freeze",
    "python_freeze_sha256",
}
_STORAGE_KEYS = {"source", "mount", "volume", "project_root"}
_WANDB_KEYS = {
    "secret_environment",
    "secret_name",
    "entity",
    "project",
    "expected_email",
    "authorization_scope",
}
_ORCHESTRATION_KEYS = {
    "visibility",
    "control_deployment",
    "poll_seconds",
    "semantic_max_attempts",
    "platform_worker_retries",
    "platform_job_retries",
    "ttl_seconds_after_finished",
}
_REQUEST_KEYS = {
    "schema",
    "campaign_id",
    "macro_round",
    "learner_identity_sha256",
    "backend_contract_sha256",
    "environment",
    "paths",
}
_PATH_KEYS = {
    "source_root",
    "campaign_root",
    "learner_output",
    "learner_result",
    "completion_attestation",
    "success_marker",
    "success_checksum",
}

# run_h100_learner.sh owns semantic validation of these values.  The backend
# additionally freezes the exact public key-set so credentials or unreviewed
# scheduler settings cannot hitch a ride in a job spec.
_REQUIRED_ENV = {
    "ROUND_DAGGER_CAMPAIGN_ROOT",
    "ROUND_DAGGER_CAMPAIGN_ID",
    "ROUND_DAGGER_MACRO_ROUND",
    "ROUND_DAGGER_POLICY_CHECKPOINT",
    "ROUND_DAGGER_POLICY_SHA256",
    "ROUND_DAGGER_RUNTIME_POLICY_SHA256",
    "ROUND_DAGGER_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256",
    "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256",
    "ROUND_DAGGER_LEARNER_OUTPUT_DIR",
    "ROUND_DAGGER_LEARNER_RESULT",
    "ROUND_DAGGER_LEARNER_CODE_SHA256",
    "ROUND_DAGGER_LEARNER_CONFIG_SHA256",
    "ROUND_DAGGER_LEARNER_CONTRACT_SHA256",
    "ROUND_DAGGER_LEARNER_IDENTITY_SHA256",
    "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256",
    "ROUND_DAGGER_BASE_POLICY_SPEC",
    "ROUND_DAGGER_LEARNER_BATCH_SIZE",
    "ROUND_DAGGER_LEARNER_MAX_STEPS",
    "ROUND_DAGGER_LEARNER_SAVE_EVERY",
    "ROUND_DAGGER_LEARNER_LEARNING_RATE",
    "ROUND_DAGGER_LEARNER_WEIGHT_DECAY",
    "ROUND_DAGGER_LEARNER_SEED",
    "ROUND_DAGGER_LEARNER_PARTITION_SEED",
    "ROUND_DAGGER_TARGET_ENVIRONMENTS",
    "ROUND_DAGGER_COHORT_COUNT",
    "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT",
    "ROUND_DAGGER_STEPS_PER_ENVIRONMENT",
    "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256",
    "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL",
    "ROUND_DAGGER_WANDB_PROJECT",
    "ROUND_DAGGER_WANDB_ENTITY",
    "ROUND_DAGGER_WANDB_RUN_ID",
    "ROUND_DAGGER_WANDB_EXPECTED_EMAIL",
    "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE",
}
_OPTIONAL_ENV = {
    "ROUND_DAGGER_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_MANIFEST",
    "ROUND_DAGGER_POLICY_SPEC_SHA256",
    "ROUND_DAGGER_POLICY_MANIFEST_SHA256",
    "ROUND_DAGGER_LEARNER_CONTRACT",
    "ROUND_DAGGER_WANDB_READBACK_TIMEOUT_SECONDS",
}
_CAMPAIGN_PATH_ENV = {
    "ROUND_DAGGER_POLICY_CHECKPOINT",
    "ROUND_DAGGER_REPLAY_SNAPSHOT",
    "ROUND_DAGGER_BASE_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_SPEC",
    "ROUND_DAGGER_POLICY_MANIFEST",
    "ROUND_DAGGER_LEARNER_CONTRACT",
    "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT",
}

_ACTIVE_STATES = {
    "Starting",
    "Running",
    "Queueing",
    "Awaiting",
    "PendingRetry",
    "Restarting",
}
_FAILURE_STATES = {"Failed", "Stopped", "Stopping", "Deleting", "Deleted"}
_SUCCESS_STATES = {"Completed", "Archived"}
_CONTROL_READY = "Ready"
_CONTROL_STOPPED = "Stopped"
_CONTROL_STOPPING = "Stopping"
_CONTROL_RESTORING = {"Starting", "Updating", "Scaling", "Not Ready"}
_LIFECYCLE_PHASES = {
    "control_ready",
    "control_stop_requested",
    "control_stopped",
    "control_restore_requested",
}
_SUCCESS_MARKER_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "macro_round",
    "learner_code_sha256",
    "learner_config_sha256",
    "learner_identity_sha256",
    "result_path",
    "result_sha256",
    "completion_attestation_path",
    "completion_attestation_sha256",
    "learner_output_manifest_path",
    "learner_output_manifest_sha256",
    "learner_output_archive_sha256",
    "learner_output_inventory_sha256",
    "learner_output_tree_sha256",
}

_LEARNER_OUTPUT_SUCCESS_SCHEMA = (
    "fast_roborender.round_dagger.remote_learner_success.v4"
)
_LEARNER_OUTPUT_ARCHIVE_NAME = "learner_output.tar.gz"
_LEARNER_OUTPUT_INVENTORY_NAME = "learner_output.inventory.json"
_LEARNER_OUTPUT_MANIFEST_NAME = "learner_output.manifest.json"


class LeptonJobError(RuntimeError):
    """A Lepton deployment contract or lifecycle proof failed closed."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise LeptonJobError(
            f"{label} has missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise LeptonJobError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=pairs)
    except LeptonJobError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeptonJobError(
            f"cannot read JSON document ({type(exc).__name__})"
        ) from exc
    if not isinstance(value, Mapping):
        raise LeptonJobError("top-level JSON value must be an object")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _load_regular(path: str | os.PathLike[str], *, immutable: bool) -> tuple[Path, dict[str, Any], str]:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        info = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise LeptonJobError(f"JSON document does not exist: {absolute}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LeptonJobError(f"JSON document must be a regular non-symlink file: {absolute}")
    if immutable and info.st_mode & 0o222:
        raise LeptonJobError(f"JSON document must be immutable: {absolute}")
    return absolute, _strict_json(absolute), sha256_file(absolute)


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LeptonJobError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise LeptonJobError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise LeptonJobError(f"{label} is malformed")
    return value


def _project_path(value: Any, label: str, *, root: PurePosixPath = PROJECT_ROOT) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise LeptonJobError(f"{label} must be a safe absolute POSIX path")
    raw_parts = value.split("/")
    if any(part in {".", ".."} for part in raw_parts):
        raise LeptonJobError(f"{label} must be a normalized absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LeptonJobError(f"{label} must be a normalized absolute POSIX path")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LeptonJobError(f"{label} must be under {root}") from exc
    return path


@dataclass(frozen=True)
class LeptonBackendContract:
    sha256: str
    document: Mapping[str, Any]
    image_digest: str
    python: PurePosixPath
    entrypoint: PurePosixPath
    source_root: PurePosixPath
    source_inventory: PurePosixPath
    source_inventory_sha256: str
    source_verifier: PurePosixPath
    python_freeze: PurePosixPath
    python_freeze_sha256: str
    project_root: PurePosixPath
    poll_seconds: int
    semantic_max_attempts: int
    ttl_seconds_after_finished: int


def load_contract(
    path: str | os.PathLike[str], *, immutable: bool = False
) -> LeptonBackendContract:
    _path, raw, digest = _load_regular(path, immutable=immutable)
    document = _exact(raw, _CONTRACT_KEYS, "Lepton backend contract")
    if document["schema"] != CONTRACT_SCHEMA or document["backend"] != BACKEND:
        raise LeptonJobError("Lepton backend schema/name mismatch")

    cluster = _exact(document["cluster"], _CLUSTER_KEYS, "cluster")
    expected_cluster = {
        "workspace": "b5k2m9x7",
        "reservation": RESERVATION_ID,
        "node_group": NODE_GROUP,
        "resource_shape": RESOURCE_SHAPE,
    }
    if cluster != expected_cluster:
        raise LeptonJobError("Lepton cluster allocation differs from the reserved H200 node")

    allocation = _exact(document["allocation"], _ALLOCATION_KEYS, "allocation")
    if allocation["workers"] != WORKERS or allocation["gpus_per_worker"] != GPU_COUNT:
        raise LeptonJobError("Lepton learner must use one worker with eight H200 GPUs")
    shared_memory = _true_int(
        allocation["shared_memory_mib"], "allocation.shared_memory_mib", minimum=1
    )
    if shared_memory != 524288:
        raise LeptonJobError("Lepton shared memory must be exactly 512 GiB")

    container = _exact(document["container"], _CONTAINER_KEYS, "container")
    image = container["image"]
    match = _IMAGE_RE.fullmatch(image) if isinstance(image, str) else None
    if match is None:
        raise LeptonJobError("container image must be pinned by an OCI SHA-256 digest")
    image_digest = match.group(1)

    storage = _exact(document["storage"], _STORAGE_KEYS, "storage")
    expected_storage = {
        "source": str(WORKSPACE_SOURCE),
        "mount": str(WORKSPACE_MOUNT),
        "volume": "node-nfs:nfs",
        "project_root": str(PROJECT_ROOT),
    }
    if storage != expected_storage:
        raise LeptonJobError("Lepton persistent storage mapping is not canonical")
    project_root = PurePosixPath(storage["project_root"])
    python = _project_path(container["python"], "container.python", root=project_root)
    entrypoint = _project_path(
        container["entrypoint"], "container.entrypoint", root=project_root
    )

    supply_chain = _exact(
        document["supply_chain"], _SUPPLY_CHAIN_KEYS, "supply_chain"
    )
    source_root = _project_path(
        supply_chain["source_root"], "supply_chain.source_root", root=project_root
    )
    source_inventory = _project_path(
        supply_chain["source_inventory"],
        "supply_chain.source_inventory",
        root=project_root,
    )
    source_verifier = _project_path(
        supply_chain["source_verifier"],
        "supply_chain.source_verifier",
        root=project_root,
    )
    python_freeze = _project_path(
        supply_chain["python_freeze"],
        "supply_chain.python_freeze",
        root=project_root,
    )
    if source_root != entrypoint.parents[2]:
        raise LeptonJobError("supply-chain source root differs from learner entrypoint")
    if source_inventory != source_root.parent / f"{source_root.name}.inventory.json":
        raise LeptonJobError("source inventory path is non-canonical")
    expected_verifier = (
        source_root / "scripts/round_dagger/verify_sealed_source_snapshot.py"
    )
    if source_verifier != expected_verifier:
        raise LeptonJobError("sealed-source verifier path is non-canonical")
    expected_freeze = (
        python.parent.parent.parent / f"{python.parent.parent.name}.freeze.txt"
    )
    if python_freeze != expected_freeze:
        raise LeptonJobError("Python environment freeze path is non-canonical")
    source_inventory_sha256 = _sha(
        supply_chain["source_inventory_sha256"],
        "supply_chain.source_inventory_sha256",
    )
    python_freeze_sha256 = _sha(
        supply_chain["python_freeze_sha256"],
        "supply_chain.python_freeze_sha256",
    )

    wandb = _exact(document["wandb"], _WANDB_KEYS, "wandb")
    expected_wandb = {
        "secret_environment": WANDB_SECRET_ENV,
        "secret_name": WANDB_SECRET_NAME,
        "entity": "nvidia",
        "project": "roborender-round-dagger",
        "expected_email": "ldu@nvidia.com",
        "authorization_scope": "team",
    }
    if wandb != expected_wandb:
        raise LeptonJobError("Lepton W&B authorization contract differs from production")

    orchestration = _exact(
        document["orchestration"], _ORCHESTRATION_KEYS, "orchestration"
    )
    if (
        orchestration["visibility"] != "private"
        or orchestration["control_deployment"] != CONTROL_DEPLOYMENT
        or orchestration["semantic_max_attempts"] != SEMANTIC_MAX_ATTEMPTS
        or orchestration["platform_worker_retries"] != 0
        or orchestration["platform_job_retries"] != 0
    ):
        raise LeptonJobError("Lepton privacy/retry contract changed")
    poll_seconds = _true_int(
        orchestration["poll_seconds"], "orchestration.poll_seconds", minimum=1
    )
    ttl = _true_int(
        orchestration["ttl_seconds_after_finished"],
        "orchestration.ttl_seconds_after_finished",
        minimum=3600,
    )
    return LeptonBackendContract(
        sha256=digest,
        document=document,
        image_digest=image_digest,
        python=python,
        entrypoint=entrypoint,
        source_root=source_root,
        source_inventory=source_inventory,
        source_inventory_sha256=source_inventory_sha256,
        source_verifier=source_verifier,
        python_freeze=python_freeze,
        python_freeze_sha256=python_freeze_sha256,
        project_root=project_root,
        poll_seconds=poll_seconds,
        semantic_max_attempts=SEMANTIC_MAX_ATTEMPTS,
        ttl_seconds_after_finished=ttl,
    )


def require_finalized_supply_chain(contract: LeptonBackendContract) -> None:
    """Reject planning placeholders before any live Lepton client is opened.

    The checked-in contract intentionally carries all-zero digests until the
    final source tree and Python environment are sealed.  A remote preflight
    would reject those values as well, but a live submitter should fail locally
    and unambiguously before it performs even a read-only provider API call.
    """

    if (
        contract.source_inventory_sha256 == "0" * 64
        or contract.python_freeze_sha256 == "0" * 64
    ):
        raise LeptonJobError(
            "Lepton supply-chain identities are unfinalized placeholders"
        )


def verify_pinned_immutable_file(
    path: str | os.PathLike[str], expected_sha256: str, label: str
) -> bytes:
    """Read one exact mode-0444 file and reject mutation during hashing."""

    expected = _sha(expected_sha256, f"{label} SHA-256")
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        before = os.lstat(absolute)
    except OSError as exc:
        raise LeptonJobError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise LeptonJobError(f"{label} must be an immutable mode-0444 regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise LeptonJobError(f"cannot safely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o444
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise LeptonJobError(f"{label} changed before it was opened")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if identity(opened) != identity(after):
            raise LeptonJobError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected:
        raise LeptonJobError(f"{label} SHA-256 mismatch")
    return b"".join(chunks)


def verify_sealed_contract_source(
    contract: LeptonBackendContract,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    """Run the repository's independent sealed-tree verifier."""

    # Verify the externally pinned evidence before executing any source file.
    inventory_bytes = verify_pinned_immutable_file(
        str(contract.source_inventory),
        contract.source_inventory_sha256,
        "sealed source inventory",
    )
    try:
        inventory = json.loads(inventory_bytes.decode("utf-8"))
        records = inventory["files"]
        relative_verifier = contract.source_verifier.relative_to(
            contract.source_root
        ).as_posix()
        matches = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("path") == relative_verifier
        ]
        if len(matches) != 1:
            raise ValueError
        verifier_sha256 = _sha(matches[0].get("sha256"), "verifier inventory SHA-256")
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeptonJobError("sealed source inventory has no unique verifier record") from exc
    verifier = Path(str(contract.source_verifier))
    if verifier.is_symlink() or not verifier.is_file() or verifier.stat().st_mode & 0o222:
        raise LeptonJobError("sealed-source verifier is missing or unsafe")
    if sha256_file(verifier) != verifier_sha256:
        raise LeptonJobError("sealed-source verifier is not inventory-bound")
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    try:
        completed = runner(
            [
                str(contract.python),
                str(verifier),
                "--root",
                str(contract.source_root),
                "--expected-inventory-sha256",
                contract.source_inventory_sha256,
            ],
            cwd=str(contract.source_root),
            env=environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeptonJobError("sealed-source verifier could not execute") from exc
    if completed.returncode != 0:
        raise LeptonJobError("sealed source verification failed")


def verify_python_environment_freeze(
    contract: LeptonBackendContract,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    """Bind the executable environment to exact canonical ``pip freeze --all`` bytes."""

    expected = verify_pinned_immutable_file(
        str(contract.python_freeze),
        contract.python_freeze_sha256,
        "Python environment freeze",
    )
    if not expected or not expected.endswith(b"\n") or b"\r" in expected:
        raise LeptonJobError("Python environment freeze is not canonical text")
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        completed = runner(
            [str(contract.python), "-m", "pip", "freeze", "--all"],
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeptonJobError("Python environment freeze probe could not execute") from exc
    observed = completed.stdout
    if isinstance(observed, str):
        observed = observed.encode("utf-8")
    if completed.returncode != 0 or not isinstance(observed, bytes):
        raise LeptonJobError("Python environment freeze probe failed")
    if observed.replace(b"\r\n", b"\n") != expected:
        raise LeptonJobError("live Python environment differs from its pinned freeze")


@dataclass(frozen=True)
class LeptonRunRequest:
    sha256: str
    document: Mapping[str, Any]
    campaign_id: str
    macro_round: int
    identity_sha256: str
    environment: Mapping[str, str]
    paths: Mapping[str, PurePosixPath]

    @property
    def wandb_run_id(self) -> str:
        return self.environment["ROUND_DAGGER_WANDB_RUN_ID"]


def _validate_public_environment(
    value: Any, *, contract: LeptonBackendContract, identity: str, campaign_id: str,
    macro_round: int,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise LeptonJobError("request.environment must be an object")
    for key, raw in value.items():
        if (
            isinstance(key, str)
            and key != "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE"
            and (_SENSITIVE_KEY_RE.search(key) or key == WANDB_SECRET_ENV)
        ) or (isinstance(raw, str) and _WANDB_VALUE_RE.search(raw)):
            raise LeptonJobError("request.environment must not contain credentials")
    actual = set(value)
    missing = _REQUIRED_ENV - actual
    unknown = actual - (_REQUIRED_ENV | _OPTIONAL_ENV)
    if missing or unknown:
        raise LeptonJobError(
            f"request.environment has missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    environment: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or _ENV_RE.fullmatch(key) is None:
            raise LeptonJobError("request.environment contains a malformed name")
        if (
            (_SENSITIVE_KEY_RE.search(key) and key != "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE")
            or key == WANDB_SECRET_ENV
        ):
            raise LeptonJobError("request.environment must not contain credentials")
        if not isinstance(raw, str) or not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
            raise LeptonJobError(f"request.environment[{key}] is malformed")
        if _WANDB_VALUE_RE.search(raw):
            raise LeptonJobError("request.environment contains a W&B credential")
        environment[key] = raw

    expected = {
        "ROUND_DAGGER_CAMPAIGN_ID": campaign_id,
        "ROUND_DAGGER_MACRO_ROUND": str(macro_round),
        "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": identity,
        "ROUND_DAGGER_WANDB_PROJECT": contract.document["wandb"]["project"],
        "ROUND_DAGGER_WANDB_ENTITY": contract.document["wandb"]["entity"],
        "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": contract.document["wandb"]["expected_email"],
        "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE": contract.document["wandb"]["authorization_scope"],
        "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
        "ROUND_DAGGER_COHORT_COUNT": "4",
        "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
        "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "131072",
        "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL": "100",
        "ROUND_DAGGER_LEARNER_BATCH_SIZE": "8",
        "ROUND_DAGGER_LEARNER_MAX_STEPS": "1200",
        "ROUND_DAGGER_LEARNER_SAVE_EVERY": "5",
        "ROUND_DAGGER_LEARNER_LEARNING_RATE": "2.5e-5",
        "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": "0.01",
        "ROUND_DAGGER_LEARNER_SEED": "17",
        "ROUND_DAGGER_LEARNER_PARTITION_SEED": "23",
        "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": contract.image_digest,
    }
    # The remote-container digest is injected by the backend and is therefore
    # intentionally absent from the public request mapping.
    for key, wanted in expected.items():
        if key == "ROUND_DAGGER_REMOTE_CONTAINER_SHA256":
            continue
        if environment.get(key) != wanted:
            raise LeptonJobError(f"request.environment[{key}] differs from the backend contract")
    expected_wandb_id = f"rrd-m{macro_round:06d}-id{identity}"
    if environment["ROUND_DAGGER_WANDB_RUN_ID"] != expected_wandb_id:
        raise LeptonJobError("W&B run ID is not bound to the learner identity")
    for key in (
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256",
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256",
    ):
        _sha(environment[key], f"request.environment[{key}]")
    from round_dagger.learner_identity import (
        LearnerIdentityError,
        learner_identity_sha256,
    )

    try:
        expected_identity = learner_identity_sha256(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_code_sha256=environment[
                "ROUND_DAGGER_LEARNER_CODE_SHA256"
            ],
            learner_config_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
            ],
            replay_snapshot_file_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ],
            replay_snapshot_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"
            ],
            parent_checkpoint_sha256=environment[
                "ROUND_DAGGER_POLICY_SHA256"
            ],
            parent_runtime_policy_sha256=environment[
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256"
            ],
            runtime_container_sha256=contract.image_digest,
            learner_contract_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
            ],
        )
    except LearnerIdentityError as exc:
        raise LeptonJobError("request learner lineage is malformed") from exc
    if identity != expected_identity:
        raise LeptonJobError("learner identity does not bind the exact request semantics")
    return environment


def load_request(
    path: str | os.PathLike[str], *, contract: LeptonBackendContract,
    immutable: bool = False,
) -> LeptonRunRequest:
    _path, raw, digest = _load_regular(path, immutable=immutable)
    document = _exact(raw, _REQUEST_KEYS, "Lepton run request")
    if document["schema"] != REQUEST_SCHEMA:
        raise LeptonJobError("Lepton run request schema mismatch")
    if document["backend_contract_sha256"] != contract.sha256:
        raise LeptonJobError("run request is not bound to this backend contract")
    campaign_id = _token(document["campaign_id"], "campaign_id")
    macro_round = _true_int(document["macro_round"], "macro_round")
    identity = _sha(document["learner_identity_sha256"], "learner identity")
    environment = _validate_public_environment(
        document["environment"], contract=contract, identity=identity,
        campaign_id=campaign_id, macro_round=macro_round,
    )
    paths_raw = _exact(document["paths"], _PATH_KEYS, "request.paths")
    paths = {
        key: _project_path(value, f"request.paths.{key}", root=contract.project_root)
        for key, value in paths_raw.items()
    }
    for name in _CAMPAIGN_PATH_ENV & set(environment):
        candidate = _project_path(
            environment[name], f"request.environment.{name}", root=contract.project_root
        )
        try:
            candidate.relative_to(paths["campaign_root"])
        except ValueError as exc:
            raise LeptonJobError(
                f"request.environment.{name} must be inside remote campaign"
            ) from exc
    if paths["source_root"] != contract.source_root:
        raise LeptonJobError("request source root differs from the pinned sealed source")
    if PurePosixPath(environment["ROUND_DAGGER_CAMPAIGN_ROOT"]) != paths["campaign_root"]:
        raise LeptonJobError("campaign root path is inconsistent")
    if PurePosixPath(environment["ROUND_DAGGER_LEARNER_OUTPUT_DIR"]) != paths["learner_output"]:
        raise LeptonJobError("learner output path is inconsistent")
    if PurePosixPath(environment["ROUND_DAGGER_LEARNER_RESULT"]) != paths["learner_result"]:
        raise LeptonJobError("learner result path is inconsistent")
    if paths["learner_result"].parent != paths["learner_output"]:
        raise LeptonJobError("learner result must be directly inside learner output")
    round_padded = f"{macro_round:06d}"
    expected_attestation = paths["learner_output"] / f"learner_completion_round_{round_padded}.json"
    if paths["completion_attestation"] != expected_attestation:
        raise LeptonJobError("completion attestation path is non-canonical")
    expected_marker = paths["learner_output"] / "lepton_learner_success.json"
    if paths["success_marker"] != expected_marker:
        raise LeptonJobError("Lepton success marker path is non-canonical")
    if paths["success_checksum"] != PurePosixPath(f"{expected_marker}.sha256"):
        raise LeptonJobError("Lepton success checksum path is non-canonical")
    return LeptonRunRequest(
        sha256=digest,
        document=document,
        campaign_id=campaign_id,
        macro_round=macro_round,
        identity_sha256=identity,
        environment=environment,
        paths=paths,
    )


def _local_campaign_relative(
    value: str, *, local_campaign_root: Path, label: str
) -> PurePosixPath:
    path = Path(os.path.abspath(Path(value).expanduser()))
    try:
        relative = path.relative_to(local_campaign_root)
    except ValueError as exc:
        raise LeptonJobError(f"{label} must be inside the local campaign") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise LeptonJobError(f"{label} is non-canonical")
    return PurePosixPath(*relative.parts)


def materialize_request(
    output: str | os.PathLike[str],
    *,
    contract: LeptonBackendContract,
    local_campaign_root: str | os.PathLike[str],
    remote_campaign_root: str,
    remote_learner_output: str,
    source_environment: Mapping[str, str],
) -> LeptonRunRequest:
    """Create one immutable request by rebasing trusted coordinator inputs.

    Only the reviewed learner environment allowlist is projected.  Local
    campaign paths are rebased into the identity-scoped NFS campaign tree;
    host paths and unrelated ambient variables never reach the Job.
    """

    local_root = Path(os.path.abspath(Path(local_campaign_root).expanduser()))
    if local_root.is_symlink() or not local_root.is_dir():
        raise LeptonJobError("local campaign root is missing or unsafe")
    remote_campaign = _project_path(
        remote_campaign_root, "remote campaign root", root=contract.project_root
    )
    remote_output = _project_path(
        remote_learner_output, "remote learner output", root=contract.project_root
    )
    try:
        remote_output.relative_to(remote_campaign)
    except ValueError as exc:
        raise LeptonJobError("remote learner output must be inside remote campaign") from exc

    missing = _REQUIRED_ENV - set(source_environment)
    if missing:
        raise LeptonJobError(f"coordinator environment is missing {sorted(missing)}")
    environment = {
        key: str(source_environment[key])
        for key in sorted(_REQUIRED_ENV | (_OPTIONAL_ENV & set(source_environment)))
    }
    if Path(os.path.abspath(environment["ROUND_DAGGER_CAMPAIGN_ROOT"])) != local_root:
        raise LeptonJobError("coordinator campaign root differs from materialization input")
    for key in _CAMPAIGN_PATH_ENV & set(environment):
        relative = _local_campaign_relative(
            environment[key], local_campaign_root=local_root, label=key
        )
        environment[key] = str(remote_campaign / relative)
    environment["ROUND_DAGGER_CAMPAIGN_ROOT"] = str(remote_campaign)
    macro_text = environment["ROUND_DAGGER_MACRO_ROUND"]
    if not macro_text.isdigit():
        raise LeptonJobError("coordinator macro round is malformed")
    macro_round = int(macro_text)
    environment["ROUND_DAGGER_LEARNER_OUTPUT_DIR"] = str(remote_output)
    environment["ROUND_DAGGER_LEARNER_RESULT"] = str(
        remote_output / f"learner_result_round_{macro_round:06d}.json"
    )
    identity = environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
    document = {
        "schema": REQUEST_SCHEMA,
        "campaign_id": environment["ROUND_DAGGER_CAMPAIGN_ID"],
        "macro_round": macro_round,
        "learner_identity_sha256": identity,
        "backend_contract_sha256": contract.sha256,
        "environment": environment,
        "paths": {
            "source_root": str(contract.entrypoint.parents[2]),
            "campaign_root": str(remote_campaign),
            "learner_output": str(remote_output),
            "learner_result": environment["ROUND_DAGGER_LEARNER_RESULT"],
            "completion_attestation": str(
                remote_output
                / f"learner_completion_round_{macro_round:06d}.json"
            ),
            "success_marker": str(remote_output / "lepton_learner_success.json"),
            "success_checksum": str(
                remote_output / "lepton_learner_success.json.sha256"
            ),
        },
    }
    destination = Path(os.path.abspath(Path(output).expanduser()))
    if destination.is_symlink() or destination.exists():
        raise LeptonJobError("run request destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise LeptonJobError("run request destination appeared concurrently") from exc
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return load_request(destination, contract=contract, immutable=True)


def job_name(identity_sha256: str, attempt: int) -> str:
    identity = _sha(identity_sha256, "learner identity")
    attempt = _true_int(attempt, "semantic attempt", minimum=1)
    if attempt > SEMANTIC_MAX_ATTEMPTS:
        raise LeptonJobError("semantic attempt exceeds the backend limit")
    name = f"rrd-h200-{identity[:24]}-a{attempt}"
    if _JOB_RE.fullmatch(name) is None:
        raise LeptonJobError("derived Lepton Job name is malformed")
    return name


def _job_labels(
    *, contract: LeptonBackendContract, request: LeptonRunRequest, attempt: int
) -> dict[str, str]:
    """Return Kubernetes-label-safe halves that retain every digest bit."""

    return {
        "roborender-dagger-backend": BACKEND,
        "roborender-dagger-identity-hi": request.identity_sha256[:32],
        "roborender-dagger-identity-lo": request.identity_sha256[32:],
        "roborender-dagger-contract-hi": contract.sha256[:32],
        "roborender-dagger-contract-lo": contract.sha256[32:],
        "roborender-dagger-request-hi": request.sha256[:32],
        "roborender-dagger-request-lo": request.sha256[32:],
        "roborender-dagger-attempt": str(attempt),
    }


def build_job_document(
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    *,
    request_path: str | PurePosixPath,
    contract_path: str | PurePosixPath,
    attempt: int,
) -> dict[str, Any]:
    request_remote = _project_path(request_path, "remote request", root=contract.project_root)
    contract_remote = _project_path(contract_path, "remote contract", root=contract.project_root)
    name = job_name(request.identity_sha256, attempt)
    labels = _job_labels(contract=contract, request=request, attempt=attempt)
    public_env = [
        {"name": "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT", "value": str(contract_remote)},
        {"name": "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256", "value": contract.sha256},
        {"name": "ROUND_DAGGER_LEPTON_RUN_REQUEST", "value": str(request_remote)},
        {"name": "ROUND_DAGGER_LEPTON_RUN_REQUEST_SHA256", "value": request.sha256},
        {"name": "ROUND_DAGGER_LEPTON_SEMANTIC_ATTEMPT", "value": str(attempt)},
    ]
    secret_env = {
        "name": WANDB_SECRET_ENV,
        "value_from": {"secret_name_ref": WANDB_SECRET_NAME},
    }
    command = [
        str(contract.entrypoint),
        "--contract",
        str(contract_remote),
        "--contract-sha256",
        contract.sha256,
        "--request",
        str(request_remote),
        "--request-sha256",
        request.sha256,
        "--semantic-attempt",
        str(attempt),
    ]
    document = {
        "metadata": {"name": name, "labels": labels, "visibility": "private"},
        "spec": {
            "resource_shape": RESOURCE_SHAPE,
            "affinity": {"allowed_dedicated_node_groups": [NODE_GROUP]},
            "container": {
                "image": contract.document["container"]["image"],
                "command": command,
            },
            "shared_memory_size": contract.document["allocation"]["shared_memory_mib"],
            "completions": WORKERS,
            "parallelism": WORKERS,
            "max_failure_retry": 0,
            "max_job_failure_retry": 0,
            "envs": [*public_env, secret_env],
            "mounts": [
                {
                    "path": str(WORKSPACE_SOURCE),
                    "from": contract.document["storage"]["volume"],
                    "mount_path": str(WORKSPACE_MOUNT),
                }
            ],
            "ttl_seconds_after_finished": contract.ttl_seconds_after_finished,
            "intra_job_communication": False,
            "reservation_config": {
                "reservation_id": RESERVATION_ID,
                "allow_burst_to_other_reservations": False,
            },
            "log": {"enable_collection": True},
        },
    }
    # Defense in depth: a credential value can never enter a printed plan.
    serialized = json.dumps(document, sort_keys=True)
    if _WANDB_VALUE_RE.search(serialized):
        raise LeptonJobError("job document unexpectedly contains a W&B credential")
    return document


def to_sdk_job(document: Mapping[str, Any]) -> Any:
    """Convert a validated public document to the installed Lepton SDK type."""

    try:
        from leptonai.api.v1.types.job import LeptonJob
    except ImportError as exc:  # pragma: no cover - deployment-only dependency
        raise LeptonJobError("Lepton SDK is not installed") from exc
    try:
        return LeptonJob.model_validate(document)
    except Exception as exc:  # pydantic error types differ across SDK releases
        raise LeptonJobError("Lepton SDK rejected the pinned Job document") from exc


def _job_projection(job: Any) -> tuple[str, str, Mapping[str, str]]:
    metadata = getattr(job, "metadata", None)
    name = getattr(metadata, "name", None)
    labels = getattr(metadata, "labels", None)
    status = getattr(job, "status", None)
    raw_state = getattr(status, "state", None)
    state = getattr(raw_state, "value", raw_state)
    if not isinstance(name, str) or not isinstance(labels, Mapping) or not isinstance(state, str):
        raise LeptonJobError("Lepton API returned a malformed Job")
    return name, state, {str(key): str(value) for key, value in labels.items()}


def validate_job_identity(
    job: Any, *, contract: LeptonBackendContract, request: LeptonRunRequest,
    attempt: int,
) -> str:
    name, state, labels = _job_projection(job)
    expected_labels = _job_labels(
        contract=contract, request=request, attempt=attempt
    )
    if name != job_name(request.identity_sha256, attempt) or labels != expected_labels:
        raise LeptonJobError("existing Lepton Job does not match the exact learner identity")
    if state not in _ACTIVE_STATES | _FAILURE_STATES | _SUCCESS_STATES:
        raise LeptonJobError(f"Lepton Job returned an unsupported state: {state}")
    return state


class JobClient(Protocol):
    def list_all(self, **kwargs: Any) -> Sequence[Any]: ...
    def create(self, spec: Any) -> Any: ...
    def get(self, id_or_job: Any, **kwargs: Any) -> Any: ...


class ControlDeploymentClient(Protocol):
    def get(self, name_or_deployment: Any) -> Any: ...
    def scale(self, name_or_deployment: Any, replicas: int) -> Any: ...


class LeptonControlDeploymentAdapter:
    """Expose explicit zero/one Pod scaling over the deployment SDK.

    Lepton's deployment ``stop`` helper only patches ``min_replicas`` and its
    ``restart`` endpoint cannot revive a Pod with zero replicas.  The provider
    normalizes a stopped Pod to ``min_replicas=max_replicas=0``.  Use one
    symmetric, partial public update for both directions so crash recovery can
    deterministically invert the mutation without replaying the full Pod spec.
    """

    def __init__(
        self,
        client: Any,
        *,
        partial_builder: Callable[[int], Any] | None = None,
    ):
        self.client = client
        self.partial_builder = partial_builder

    def get(self, name_or_deployment: Any) -> Any:
        return self.client.get(name_or_deployment)

    def scale(self, name_or_deployment: Any, replicas: int) -> Any:
        if name_or_deployment != CONTROL_DEPLOYMENT:
            raise LeptonJobError("control deployment name mismatch")
        count = _true_int(replicas, "control replica count")
        if count not in {0, 1}:
            raise LeptonJobError("control replica count must be zero or one")
        if self.partial_builder is not None:
            partial = self.partial_builder(count)
        else:
            try:
                from leptonai.api.v1.types.deployment import (
                    LeptonDeployment,
                    LeptonDeploymentUserSpec,
                    ResourceRequirement,
                )
            except ImportError as exc:  # pragma: no cover - deployment-only dependency
                raise LeptonJobError("Lepton SDK is not installed") from exc
            partial = LeptonDeployment(
                spec=LeptonDeploymentUserSpec(
                    resource_requirement=ResourceRequirement(
                        min_replicas=count,
                        max_replicas=count,
                    )
                )
            )
        return self.client.update(name_or_deployment, partial)


def _member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def validate_control_deployment(
    deployment: Any, *, contract: LeptonBackendContract
) -> str:
    """Validate the immutable allocation projection of the control Pod.

    The Pod image and environment intentionally differ from the learner Job,
    but both must resolve to the same one-node reservation.  Validation is
    repeated after every lifecycle API read so a renamed/recreated deployment
    cannot be adopted accidentally while the coordinator is recovering.
    """

    metadata = _member(deployment, "metadata")
    spec = _member(deployment, "spec")
    status = _member(deployment, "status")
    name = _member(metadata, "name")
    raw_state = _member(status, "state")
    state = getattr(raw_state, "value", raw_state)
    if name != CONTROL_DEPLOYMENT or not isinstance(state, str):
        raise LeptonJobError("Lepton control deployment identity is malformed")
    allowed_states = {
        _CONTROL_READY,
        _CONTROL_STOPPED,
        _CONTROL_STOPPING,
        *_CONTROL_RESTORING,
    }
    if state not in allowed_states:
        raise LeptonJobError(f"unsupported Lepton control deployment state: {state}")

    requirement = _member(spec, "resource_requirement")
    if (
        _member(requirement, "resource_shape") != RESOURCE_SHAPE
        or _member(requirement, "shared_memory_size") != 524288
    ):
        raise LeptonJobError("control deployment is not the canonical eight-H200 shape")
    minimum = _member(requirement, "min_replicas")
    maximum = _member(requirement, "max_replicas")
    replicas = (minimum, maximum)
    if replicas not in {(0, 0), (1, 1)}:
        raise LeptonJobError("control deployment replica contract changed")

    affinity = _member(requirement, "affinity")
    allowed_groups = _member(affinity, "allowed_dedicated_node_groups")
    if list(allowed_groups or ()) != [NODE_GROUP]:
        raise LeptonJobError("control deployment node-group affinity changed")
    reservation = _member(spec, "reservation_config")
    if (
        _member(reservation, "reservation_id") != RESERVATION_ID
        # The Pod API represents its non-burst default as null, while the Job
        # API requires an explicit false.  Neither value authorizes bursting.
        or _member(reservation, "allow_burst_to_other_reservations") not in {None, False}
    ):
        raise LeptonJobError("control deployment reservation contract changed")
    if _member(spec, "is_pod") is not True:
        raise LeptonJobError("control deployment is no longer a Lepton Pod")

    mounts = _member(spec, "mounts")
    if not isinstance(mounts, Sequence):
        raise LeptonJobError("control deployment mount list is malformed")
    canonical_mounts = [
        mount
        for mount in mounts
        if _member(mount, "path") == str(WORKSPACE_SOURCE)
        and _member(mount, "from_") in {None, contract.document["storage"]["volume"]}
        and (
            _member(mount, "from") == contract.document["storage"]["volume"]
            or _member(mount, "from_") == contract.document["storage"]["volume"]
        )
        and _member(mount, "mount_path") == str(WORKSPACE_MOUNT)
    ]
    if len(canonical_mounts) != 1:
        raise LeptonJobError("control deployment persistent NFS mount changed")

    # Pod scale-to-zero is represented by Lepton as (0, 0, "Not Ready"), not
    # the deployment-oriented (0, 1, "Stopped") projection.  Normalize status
    # only after every immutable allocation field above has been checked.
    if replicas == (0, 0):
        if state in {_CONTROL_STOPPED, "Not Ready"}:
            return _CONTROL_STOPPED
        return _CONTROL_STOPPING
    if state == _CONTROL_READY:
        return _CONTROL_READY
    if state in {_CONTROL_STOPPED, _CONTROL_STOPPING}:
        return "Starting"
    return state


def wait_for_control_state(
    client: ControlDeploymentClient,
    *,
    contract: LeptonBackendContract,
    wanted: str,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    timeout = _true_int(timeout_seconds, "control wait timeout", minimum=1)
    if wanted not in {_CONTROL_READY, _CONTROL_STOPPED}:
        raise LeptonJobError("unsupported control deployment wait target")
    deadline = time.monotonic() + timeout
    while True:
        try:
            deployment = client.get(CONTROL_DEPLOYMENT)
        except Exception as exc:
            raise LeptonJobError("cannot read Lepton control deployment") from exc
        state = validate_control_deployment(deployment, contract=contract)
        if state == wanted:
            return deployment
        if wanted == _CONTROL_STOPPED and state not in {_CONTROL_STOPPING}:
            raise LeptonJobError("control deployment diverged while stopping")
        if wanted == _CONTROL_READY and state not in _CONTROL_RESTORING:
            raise LeptonJobError("control deployment diverged while restoring")
        if time.monotonic() >= deadline:
            raise LeptonJobError(f"timed out waiting for control deployment {wanted}")
        sleep(contract.poll_seconds)


class ExclusiveNodeLifecycle:
    """Crash-recoverable ownership journal for the sole reserved H200 node.

    The local journal is written *before* every mutating API operation.  A
    restarted coordinator may therefore finish an exact stop/restore, but a
    different learner identity or an already-stopped unowned Pod fails closed.
    The advisory lock prevents two coordinators on this workstation from
    driving the allocation concurrently; Lepton Job create-by-name remains the
    cross-host serialization boundary.
    """

    _KEYS = {
        "schema",
        "backend",
        "control_deployment",
        "contract_sha256",
        "request_sha256",
        "learner_identity_sha256",
        "phase",
        "semantic_attempt",
    }

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        contract: LeptonBackendContract,
        request: LeptonRunRequest,
        client: ControlDeploymentClient,
        timeout_seconds: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self.contract = contract
        self.request = request
        self.client = client
        self.timeout_seconds = _true_int(
            timeout_seconds, "control lifecycle timeout", minimum=1
        )
        self.sleep = sleep
        self._lock_descriptor: int | None = None
        self._document: dict[str, Any] | None = None

    @property
    def phase(self) -> str:
        if self._document is None:
            raise LeptonJobError("exclusive-node lifecycle is not open")
        return str(self._document["phase"])

    def __enter__(self) -> "ExclusiveNodeLifecycle":
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or parent.resolve() != parent:
            raise LeptonJobError("lifecycle state parent must not traverse a symlink")
        lock = Path(f"{self.path}.lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock, flags, 0o600)
        except OSError as exc:
            raise LeptonJobError("cannot open exclusive-node lifecycle lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise LeptonJobError("exclusive-node lifecycle lock is not regular")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LeptonJobError("another exclusive-node coordinator is active") from exc
            self._lock_descriptor = descriptor
            self._load_or_create()
            return self
        except Exception:
            os.close(descriptor)
            self._lock_descriptor = None
            raise

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def _expected(self) -> dict[str, Any]:
        return {
            "schema": EXCLUSIVE_LIFECYCLE_SCHEMA,
            "backend": BACKEND,
            "control_deployment": CONTROL_DEPLOYMENT,
            "contract_sha256": self.contract.sha256,
            "request_sha256": self.request.sha256,
            "learner_identity_sha256": self.request.identity_sha256,
        }

    def _load_or_create(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink() or not self.path.is_file():
                raise LeptonJobError("exclusive-node lifecycle state is unsafe")
            if stat.S_IMODE(self.path.stat().st_mode) != 0o600:
                raise LeptonJobError("exclusive-node lifecycle state mode is unsafe")
            document = _strict_json(self.path)
            if set(document) != self._KEYS:
                raise LeptonJobError("exclusive-node lifecycle key-set mismatch")
            if any(document.get(key) != value for key, value in self._expected().items()):
                raise LeptonJobError("exclusive-node lifecycle identity mismatch")
            phase = document.get("phase")
            attempt = document.get("semantic_attempt")
            if phase not in _LIFECYCLE_PHASES or type(attempt) is not int:
                raise LeptonJobError("exclusive-node lifecycle state is malformed")
            if attempt < 0 or attempt > self.contract.semantic_max_attempts:
                raise LeptonJobError("exclusive-node lifecycle attempt is invalid")
            self._document = document
            return
        self._document = {
            **self._expected(),
            "phase": "control_ready",
            "semantic_attempt": 0,
        }
        self._publish()

    def _publish(self) -> None:
        if self._document is None:
            raise LeptonJobError("exclusive-node lifecycle is not open")
        payload = (
            json.dumps(self._document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _transition(self, phase: str, attempt: int) -> None:
        if phase not in _LIFECYCLE_PHASES:
            raise LeptonJobError("invalid exclusive-node lifecycle transition")
        attempt = _true_int(attempt, "semantic attempt")
        if attempt > self.contract.semantic_max_attempts:
            raise LeptonJobError("semantic attempt exceeds the backend limit")
        assert self._document is not None
        self._document = {
            **self._document,
            "phase": phase,
            "semantic_attempt": attempt,
        }
        self._publish()

    def _get(self) -> tuple[Any, str]:
        try:
            deployment = self.client.get(CONTROL_DEPLOYMENT)
        except Exception as exc:
            raise LeptonJobError("cannot read Lepton control deployment") from exc
        return deployment, validate_control_deployment(
            deployment, contract=self.contract
        )

    def ensure_stopped(self, attempt: int, *, preflight: Callable[[], None]) -> None:
        attempt = _true_int(attempt, "semantic attempt", minimum=1)
        if attempt > self.contract.semantic_max_attempts:
            raise LeptonJobError("semantic attempt exceeds the backend limit")
        if self.phase == "control_restore_requested":
            self.ensure_ready()
        assert self._document is not None
        journal_attempt = int(self._document["semantic_attempt"])
        if self.phase in {"control_stop_requested", "control_stopped"}:
            if journal_attempt != attempt:
                raise LeptonJobError("cannot change attempt while control Pod is stopped")
        elif self.phase == "control_ready":
            _deployment, state = self._get()
            if state != _CONTROL_READY:
                raise LeptonJobError(
                    "control Pod is not Ready and is not owned by this lifecycle"
                )
            self._transition("control_stop_requested", attempt)
        else:  # pragma: no cover - guarded by strict journal parser
            raise LeptonJobError("unsupported exclusive-node lifecycle phase")

        _deployment, state = self._get()
        if self.phase == "control_stop_requested":
            if state == _CONTROL_READY:
                # The journal records intent before mutation.  If the
                # coordinator crashed before scale-down, the recovered call
                # must obtain a fresh preflight rather than reuse stale node
                # idleness or staged-byte observations.
                preflight()
                try:
                    self.client.scale(CONTROL_DEPLOYMENT, 0)
                except Exception as exc:
                    raise LeptonJobError(
                        "could not scale down Lepton control deployment"
                    ) from exc
                state = validate_control_deployment(
                    wait_for_control_state(
                        self.client,
                        contract=self.contract,
                        wanted=_CONTROL_STOPPED,
                        timeout_seconds=self.timeout_seconds,
                        sleep=self.sleep,
                    ),
                    contract=self.contract,
                )
            elif state == _CONTROL_STOPPING:
                state = validate_control_deployment(
                    wait_for_control_state(
                        self.client,
                        contract=self.contract,
                        wanted=_CONTROL_STOPPED,
                        timeout_seconds=self.timeout_seconds,
                        sleep=self.sleep,
                    ),
                    contract=self.contract,
                )
            elif state != _CONTROL_STOPPED:
                raise LeptonJobError("control deployment diverged after stop intent")
            if state != _CONTROL_STOPPED:
                raise LeptonJobError("control deployment did not stop")
            self._transition("control_stopped", attempt)
        elif state != _CONTROL_STOPPED:
            raise LeptonJobError("journal says stopped but control deployment is not Stopped")

    def ensure_ready(self) -> None:
        assert self._document is not None
        attempt = int(self._document["semantic_attempt"])
        _deployment, state = self._get()
        if self.phase == "control_ready":
            if state != _CONTROL_READY:
                raise LeptonJobError(
                    "control Pod is not Ready and is not owned by this lifecycle"
                )
            return
        if self.phase in {"control_stop_requested", "control_stopped"}:
            # Publish restoration intent before the first mutating API call.
            self._transition("control_restore_requested", attempt)
        if self.phase != "control_restore_requested":  # pragma: no cover
            raise LeptonJobError("unsupported exclusive-node lifecycle phase")

        _deployment, state = self._get()
        if state == _CONTROL_READY:
            self._transition("control_ready", attempt)
            return
        if state == _CONTROL_STOPPING:
            state = validate_control_deployment(
                wait_for_control_state(
                    self.client,
                    contract=self.contract,
                    wanted=_CONTROL_STOPPED,
                    timeout_seconds=self.timeout_seconds,
                    sleep=self.sleep,
                ),
                contract=self.contract,
            )
        if state == _CONTROL_STOPPED:
            try:
                self.client.scale(CONTROL_DEPLOYMENT, 1)
            except Exception as exc:
                raise LeptonJobError(
                    "could not scale up Lepton control deployment"
                ) from exc
        elif state not in _CONTROL_RESTORING:
            raise LeptonJobError("control deployment diverged during restoration")
        ready = wait_for_control_state(
            self.client,
            contract=self.contract,
            wanted=_CONTROL_READY,
            timeout_seconds=self.timeout_seconds,
            sleep=self.sleep,
        )
        if validate_control_deployment(ready, contract=self.contract) != _CONTROL_READY:
            raise LeptonJobError("control deployment restoration was not Ready")
        self._transition("control_ready", attempt)


def verify_control_preflight(
    *,
    control_exec_program: Path,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    remote_contract: str | PurePosixPath,
    remote_request: str | PurePosixPath,
    command_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    """Verify staged NFS bytes and an idle control Pod immediately before stop.

    The helper authenticates from the local Lepton SDK profile.  The remote
    command receives only public paths and digests, emits no process command
    lines, and rejects any GPU compute process or learner process.  Provider
    output is suppressed because it is not needed for the lifecycle proof and
    could otherwise disclose unrelated environment details.
    """

    program = Path(os.path.abspath(control_exec_program.expanduser()))
    if program.is_symlink() or not program.is_file() or not os.access(program, os.X_OK):
        raise LeptonJobError("control deployment exec program is missing or unsafe")
    script = r'''import hashlib, json, os, pathlib, stat, subprocess, sys
(contract_path, contract_sha, request_path, request_sha) = sys.argv[1:]
def regular_immutable(path_text, expected_sha):
    path = pathlib.Path(path_text)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o222:
        raise SystemExit(31)
    digest = hashlib.sha256()
    chunks = []
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            chunks.append(block)
    if digest.hexdigest() != expected_sha:
        raise SystemExit(33)
    return b"".join(chunks)
def executable(path_text, allow_external_target=False):
    path = pathlib.Path(path_text)
    project_text = "/workspace/projects/roborender/"
    if not str(path).startswith(project_text):
        raise SystemExit(32)
    resolved = path.resolve(strict=True)
    project = pathlib.Path("/workspace/projects/roborender").resolve(strict=True)
    if (
        (not allow_external_target and project not in resolved.parents)
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
    ):
        raise SystemExit(32)
contract_bytes = regular_immutable(contract_path, contract_sha)
request_bytes = regular_immutable(request_path, request_sha)
try:
    contract_doc = json.loads(contract_bytes.decode("utf-8"))
    request_doc = json.loads(request_bytes.decode("utf-8"))
    supply = contract_doc["supply_chain"]
    container = contract_doc["container"]
    source_root = supply["source_root"]
    source_inventory = supply["source_inventory"]
    source_inventory_sha = supply["source_inventory_sha256"]
    source_verifier = supply["source_verifier"]
    python_freeze = supply["python_freeze"]
    python_freeze_sha = supply["python_freeze_sha256"]
    python_path = container["python"]
    entrypoint = container["entrypoint"]
    if request_doc["schema"] == "fast_roborender.round_dagger.lepton_bounded_smoke_request.v1":
        entrypoint = source_root + "/scripts/round_dagger/run_lepton_h200_smoke_job.sh"
    output = request_doc["paths"]["learner_output"]
except Exception:
    raise SystemExit(41)
inventory_bytes = regular_immutable(source_inventory, source_inventory_sha)
freeze_bytes = regular_immutable(python_freeze, python_freeze_sha)
executable(entrypoint)
executable(python_path, allow_external_target=True)
try:
    records = json.loads(inventory_bytes.decode("utf-8"))["files"]
    relative_verifier = pathlib.Path(source_verifier).relative_to(source_root).as_posix()
    matches = [record for record in records if record.get("path") == relative_verifier]
    if len(matches) != 1:
        raise ValueError
    verifier_sha = matches[0]["sha256"]
    if not isinstance(verifier_sha, str) or len(verifier_sha) != 64:
        raise ValueError
except Exception:
    raise SystemExit(39)
regular_immutable(source_verifier, verifier_sha)
source_check = subprocess.run(
    [python_path, source_verifier, "--root", source_root,
     "--expected-inventory-sha256", source_inventory_sha],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"},
)
if source_check.returncode != 0:
    raise SystemExit(39)
freeze_check = subprocess.run(
    [python_path, "-m", "pip", "freeze", "--all"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
         "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"},
)
if (
    freeze_check.returncode != 0
    or freeze_check.stdout.replace(b"\r\n", b"\n") != freeze_bytes
):
    raise SystemExit(40)
out = pathlib.Path(output)
if out.is_symlink():
    raise SystemExit(34)
marker = out / "lepton_learner_success.json"
checksum = out / "lepton_learner_success.json.sha256"
if marker.exists() != checksum.exists():
    raise SystemExit(35)
for completed in (marker, checksum):
    if completed.exists():
        info = completed.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o222:
            raise SystemExit(35)
for active in out.glob(".lepton_learner_active*") if out.exists() else ():
    raise SystemExit(36)
ancestors = set()
pid = os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
        pid = int(fields[3])
    except Exception:
        break
needles = {
    "run_h100_learner.sh",
    "train_learner.py",
    "lepton_h200_learner_smoke.py",
    "torchrun",
}
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name) in ancestors:
        continue
    try:
        argv = [part.decode("utf-8", "replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(pathlib.Path(arg).name in needles for arg in argv):
        raise SystemExit(37)
gpu = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
)
if gpu.returncode != 0 or any(line.strip() for line in gpu.stdout.splitlines()):
    raise SystemExit(38)
'''
    contract_path = _project_path(
        str(remote_contract), "remote contract", root=contract.project_root
    )
    request_path = _project_path(
        str(remote_request), "remote request", root=contract.project_root
    )
    arguments = [
        str(contract_path),
        contract.sha256,
        str(request_path),
        request.sha256,
    ]
    # The control helper writes through a Kubernetes TTY. Linux canonical TTY
    # input truncates a single line above 4095 bytes; the readable source plus
    # its identity arguments exceeds that boundary. Compress only this fixed,
    # public program (never credentials), keep every reviewed path and digest
    # as a separately quoted argv item, and fail closed if future growth would
    # cross the terminal limit again.
    encoded_script = base64.b85encode(
        zlib.compress(script.encode("utf-8"), level=9)
    ).decode("ascii")
    bootstrap = (
        "import base64,zlib;"
        "exec(zlib.decompress(base64.b85decode("
        f"{encoded_script!r})))"
    )
    command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote(bootstrap),
            *(shlex.quote(item) for item in arguments),
        ]
    )
    # The helper base64-encodes this command into its own TTY line. Keep the
    # raw command below 2.8 KiB so the expanded wrapper remains below 4095.
    command_bytes = len(command.encode("utf-8"))
    if command_bytes > 2800:
        raise LeptonJobError(
            "control deployment preflight exceeds the Kubernetes TTY line limit "
            f"({command_bytes} raw bytes)"
        )
    try:
        completed = command_runner(
            [str(program), CONTROL_DEPLOYMENT, command],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeptonJobError("control deployment preflight could not execute") from exc
    if completed.returncode != 0:
        raise LeptonJobError(
            "control deployment preflight rejected staged bytes or node idleness"
        )


def find_attempt_jobs(
    client: JobClient, *, contract: LeptonBackendContract, request: LeptonRunRequest
) -> dict[int, Any]:
    observed: dict[int, Any] = {}
    active: list[int] = []
    for attempt in range(1, contract.semantic_max_attempts + 1):
        name = job_name(request.identity_sha256, attempt)
        matches = [
            job
            for job in client.list_all(job_query_mode="alive_and_archive", q=name)
            if getattr(getattr(job, "metadata", None), "name", None) == name
        ]
        if len(matches) > 1:
            raise LeptonJobError(f"multiple Lepton Jobs share exact name {name}")
        if matches:
            state = validate_job_identity(
                matches[0], contract=contract, request=request, attempt=attempt
            )
            observed[attempt] = matches[0]
            if state in _ACTIVE_STATES:
                active.append(attempt)
    if observed:
        newest = max(observed)
        expected = set(range(1, newest + 1))
        if set(observed) != expected:
            raise LeptonJobError("semantic attempt history contains a gap")
    if len(active) > 1:
        raise LeptonJobError("multiple semantic attempts are active concurrently")
    if active and any(attempt > active[0] for attempt in observed):
        raise LeptonJobError("an older attempt is active after a newer attempt exists")
    return observed


def verify_reserved_node_job_exclusivity(
    client: JobClient,
    *,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
) -> None:
    """Reject any active Job competing for the exact reserved allocation.

    This check runs only while a journal-owned lifecycle still observes the
    control Pod as Ready, before stop intent is published.  The Lepton node
    group is shared by many unrelated reservations, so node-group affinity
    alone is deliberately insufficient: only an active Job that explicitly
    names our pinned reservation can compete for this sole reserved node.
    Exact active learner attempts are adopted through the stopped-journal
    recovery path and never reach this preflight.  This bridge never cancels,
    deletes, or relabels a conflicting Job.
    """

    try:
        jobs = client.list_all(
            job_query_mode="alive_only",
            node_groups=[NODE_GROUP],
            status=sorted(_ACTIVE_STATES),
        )
    except Exception as exc:
        raise LeptonJobError("cannot prove reserved-node Job exclusivity") from exc
    for job in jobs:
        spec = _member(job, "spec")
        reservation = _member(spec, "reservation_config")
        if _member(reservation, "reservation_id") != RESERVATION_ID:
            continue
        affinity = _member(spec, "affinity")
        allowed_groups = _member(affinity, "allowed_dedicated_node_groups")
        status = _member(job, "status")
        raw_state = _member(status, "state")
        job_state = getattr(raw_state, "value", raw_state)
        if (
            list(allowed_groups or ()) != [NODE_GROUP]
            or _member(reservation, "allow_burst_to_other_reservations") is not False
            or job_state not in _ACTIVE_STATES
        ):
            raise LeptonJobError(
                "a malformed active Job claims the reserved H200 allocation"
            )
        raise LeptonJobError(
            "an active Job already claims the reserved H200 allocation"
        )


def submit_or_adopt_attempt(
    client: JobClient,
    *,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    job_document: Mapping[str, Any],
    attempt: int,
    existing: Mapping[int, Any] | None = None,
) -> tuple[Any, bool]:
    jobs = dict(existing) if existing is not None else find_attempt_jobs(
        client, contract=contract, request=request
    )
    if attempt in jobs:
        validate_job_identity(jobs[attempt], contract=contract, request=request, attempt=attempt)
        return jobs[attempt], True
    if any(
        validate_job_identity(job, contract=contract, request=request, attempt=index)
        in _ACTIVE_STATES
        for index, job in jobs.items()
    ):
        raise LeptonJobError("cannot submit while another semantic attempt is active")
    try:
        created = client.create(to_sdk_job(job_document))
    except Exception as exc:
        # A second coordinator may have won the create-by-name race.  Re-read
        # and adopt only the exact identity; never surface provider exception
        # text because SDK errors can contain request headers.
        refreshed = find_attempt_jobs(client, contract=contract, request=request)
        if attempt not in refreshed:
            raise LeptonJobError("Lepton Job creation failed without an adoptable Job") from exc
        validate_job_identity(
            refreshed[attempt], contract=contract, request=request, attempt=attempt
        )
        return refreshed[attempt], True
    validate_job_identity(created, contract=contract, request=request, attempt=attempt)
    return created, False


def wait_for_terminal(
    client: JobClient,
    job: Any,
    *,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    attempt: int,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[str], None] | None = None,
) -> tuple[Any, str]:
    timeout = _true_int(timeout_seconds, "wait timeout", minimum=1)
    deadline = time.monotonic() + timeout
    current = job
    while True:
        state = validate_job_identity(
            current, contract=contract, request=request, attempt=attempt
        )
        if heartbeat is not None:
            heartbeat(state)
        if state in _SUCCESS_STATES | _FAILURE_STATES:
            return current, state
        if time.monotonic() >= deadline:
            raise LeptonJobError("timed out waiting for the Lepton learner Job")
        sleep(contract.poll_seconds)
        current = client.get(current, job_query_mode="alive_and_archive")


def learner_output_bundle_paths(
    contract: LeptonBackendContract, request: LeptonRunRequest
) -> dict[str, PurePosixPath]:
    """Return the sole canonical, identity-bound H200 result-bundle paths."""

    root = contract.project_root / "results" / request.identity_sha256
    return {
        "root": root,
        "archive": root / _LEARNER_OUTPUT_ARCHIVE_NAME,
        "inventory": root / _LEARNER_OUTPUT_INVENTORY_NAME,
        "manifest": root / _LEARNER_OUTPUT_MANIFEST_NAME,
    }


def _ensure_real_directory(path: Path, *, root: Path) -> None:
    """Create descendants without accepting a symlink at any path component."""

    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LeptonJobError("result bundle directory is outside the project root") from exc
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise LeptonJobError("persistent project root is unavailable") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise LeptonJobError("persistent project root is unsafe")
    current = root
    for component in relative.parts:
        current /= component
        try:
            os.mkdir(current, 0o755)
        except FileExistsError:
            pass
        except OSError as exc:
            raise LeptonJobError("cannot create result bundle directory") from exc
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise LeptonJobError("result bundle directory disappeared") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LeptonJobError("result bundle path contains an unsafe component")


def _immutable_regular_identity(path: Path, label: str) -> tuple[int, str]:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LeptonJobError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise LeptonJobError(f"{label} is unsafe")
    if info.st_mode & 0o222:
        raise LeptonJobError(f"{label} must be read-only")
    return info.st_size, sha256_file(path)


def _validate_learner_output_bundle(
    *,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    manifest_sha256: str | None = None,
    archive_sha256: str | None = None,
    inventory_sha256: str | None = None,
    tree_sha256: str | None = None,
) -> dict[str, str]:
    """Independently bind the three immutable bundle files and semantic tree."""

    posix_paths = learner_output_bundle_paths(contract, request)
    paths = {
        key: Path(str(value))
        for key, value in posix_paths.items()
        if key != "root"
    }
    archive_size, observed_archive_sha = _immutable_regular_identity(
        paths["archive"], "learner-output archive"
    )
    inventory_size, observed_inventory_sha = _immutable_regular_identity(
        paths["inventory"], "learner-output inventory"
    )
    _manifest_size, observed_manifest_sha = _immutable_regular_identity(
        paths["manifest"], "learner-output manifest"
    )
    manifest = _strict_json(paths["manifest"])
    if set(manifest) != {"schema", "tree_kind", "archive", "inventory", "members"}:
        raise LeptonJobError("learner-output manifest key-set mismatch")
    if (
        manifest["schema"]
        != "fast_roborender.round_dagger.inventory_tree_archive.v1"
        or manifest["tree_kind"] != "learner_output"
    ):
        raise LeptonJobError("learner-output manifest identity mismatch")
    archive = _exact(
        manifest["archive"],
        {"format", "compression_level", "gzip_mtime", "size", "sha256"},
        "learner-output archive identity",
    )
    inventory = _exact(
        manifest["inventory"],
        {
            "schema",
            "size",
            "sha256",
            "tree_sha256",
            "file_count",
            "total_bytes",
        },
        "learner-output inventory identity",
    )
    if (
        archive["format"] != "tar+gzip"
        or archive["compression_level"] != 1
        or archive["gzip_mtime"] != 0
        or archive["size"] != archive_size
        or _sha(archive["sha256"], "learner-output archive SHA-256")
        != observed_archive_sha
    ):
        raise LeptonJobError("learner-output archive differs from its manifest")
    if (
        inventory["schema"]
        != "fast_roborender.round_dagger.remote_tree_inventory.v1"
        or inventory["size"] != inventory_size
        or _sha(inventory["sha256"], "learner-output inventory SHA-256")
        != observed_inventory_sha
    ):
        raise LeptonJobError("learner-output inventory differs from its manifest")
    observed_tree_sha = _sha(
        inventory["tree_sha256"], "learner-output tree SHA-256"
    )
    expected = {
        "manifest_sha256": manifest_sha256,
        "archive_sha256": archive_sha256,
        "inventory_sha256": inventory_sha256,
        "tree_sha256": tree_sha256,
    }
    observed = {
        "manifest_sha256": observed_manifest_sha,
        "archive_sha256": observed_archive_sha,
        "inventory_sha256": observed_inventory_sha,
        "tree_sha256": observed_tree_sha,
    }
    for key, value in expected.items():
        if value is not None and _sha(value, f"expected {key}") != observed[key]:
            raise LeptonJobError(f"learner-output {key} mismatch")
    return observed


def _pack_learner_output(
    *,
    contract: LeptonBackendContract,
    request: LeptonRunRequest,
    environment: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, str]:
    """Invoke the inventory-bound packer from the already-verified sealed source."""

    posix_paths = learner_output_bundle_paths(contract, request)
    paths = {key: Path(str(value)) for key, value in posix_paths.items()}
    _ensure_real_directory(paths["root"], root=Path(str(contract.project_root)))
    tool = (
        Path(str(request.paths["source_root"]))
        / "scripts/round_dagger/lepton_campaign_archive_transfer.py"
    )
    if tool.is_symlink() or not tool.is_file():
        raise LeptonJobError("sealed learner-output packer is missing or unsafe")
    command = [
        str(contract.python),
        str(tool),
        "pack",
        "--tree-kind",
        "learner_output",
        "--projection",
        str(request.paths["learner_output"]),
        "--archive",
        str(paths["archive"]),
        "--inventory",
        str(paths["inventory"]),
        "--manifest",
        str(paths["manifest"]),
    ]
    try:
        completed = runner(
            command,
            cwd=str(request.paths["source_root"]),
            env=dict(environment),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeptonJobError(
            "sealed learner-output packer could not execute"
        ) from exc
    if completed.returncode != 0:
        raise LeptonJobError(
            f"sealed learner-output packer failed with status {completed.returncode}"
        )
    output = completed.stdout
    if isinstance(output, bytes):
        try:
            output = output.decode("utf-8")
        except UnicodeError as exc:
            raise LeptonJobError(
                "sealed learner-output packer emitted invalid output"
            ) from exc
    try:
        report = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LeptonJobError(
            "sealed learner-output packer emitted invalid evidence"
        ) from exc
    if (
        not isinstance(report, Mapping)
        or report.get("schema")
        != "fast_roborender.round_dagger.inventory_tree_publish.v1"
        or report.get("operation") != "pack"
        or report.get("status") != "PASS"
        or report.get("tree_kind") != "learner_output"
    ):
        raise LeptonJobError(
            "sealed learner-output packer did not report exact success"
        )
    return _validate_learner_output_bundle(contract=contract, request=request)


def _publish_immutable_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise LeptonJobError("completion marker is a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise LeptonJobError("different completion marker already exists")
        if path.stat().st_mode & 0o222:
            raise LeptonJobError("existing completion marker must be read-only")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".incoming", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise LeptonJobError("competing completion marker differs")
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def checksum_payload(marker: Path) -> bytes:
    return f"{sha256_file(marker)}  {marker.name}\n".encode("ascii")


def publish_checksum(marker: Path, checksum: Path) -> str:
    marker = Path(os.path.abspath(marker))
    checksum = Path(os.path.abspath(checksum))
    if marker.is_symlink() or not marker.is_file():
        raise LeptonJobError("completion marker is missing or unsafe")
    if checksum != Path(f"{marker}.sha256"):
        raise LeptonJobError("completion checksum path is non-canonical")
    payload = checksum_payload(marker)
    if checksum.exists() or checksum.is_symlink():
        if checksum.is_symlink() or not checksum.is_file() or checksum.read_bytes() != payload:
            raise LeptonJobError("existing completion checksum conflicts")
        if checksum.stat().st_mode & 0o222:
            raise LeptonJobError("existing completion checksum must be read-only")
        return payload[:64].decode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{checksum.name}.", dir=checksum.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, checksum, follow_symlinks=False)
        except FileExistsError:
            if checksum.is_symlink() or checksum.read_bytes() != payload:
                raise LeptonJobError("competing completion checksum differs")
        os.chmod(checksum, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return payload[:64].decode("ascii")


def verify_checksum(marker: Path, checksum: Path) -> str:
    marker = Path(os.path.abspath(marker))
    checksum = Path(os.path.abspath(checksum))
    for path, label in ((marker, "completion marker"), (checksum, "completion checksum")):
        if path.is_symlink() or not path.is_file():
            raise LeptonJobError(f"{label} is missing or unsafe")
    if checksum != Path(f"{marker}.sha256") or checksum.read_bytes() != checksum_payload(marker):
        raise LeptonJobError("completion checksum does not bind the exact marker")
    return sha256_file(marker)


def validate_completion_marker(
    marker: Path,
    *,
    request: LeptonRunRequest,
    contract: LeptonBackendContract | None = None,
) -> dict[str, Any]:
    """Validate the identity projection available to the remote coordinator.

    The in-job publisher performs the stronger verification against the result
    and completion-attestation bytes before making this marker visible.  The
    coordinator independently verifies the marker checksum and its complete
    semantic identity after transfer from NFS.
    """

    document = _strict_json(marker)
    if set(document) != _SUCCESS_MARKER_KEYS:
        raise LeptonJobError("completion marker key-set mismatch")
    bundle_manifest = (
        (contract.project_root if contract is not None else PROJECT_ROOT)
        / "results"
        / request.identity_sha256
        / _LEARNER_OUTPUT_MANIFEST_NAME
    )
    expected = {
        "schema": _LEARNER_OUTPUT_SUCCESS_SCHEMA,
        "status": "PASS",
        "campaign_id": request.campaign_id,
        "macro_round": request.macro_round,
        "learner_code_sha256": request.environment[
            "ROUND_DAGGER_LEARNER_CODE_SHA256"
        ],
        "learner_config_sha256": request.environment[
            "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
        ],
        "learner_identity_sha256": request.identity_sha256,
        "result_path": request.paths["learner_result"].name,
        "completion_attestation_path": request.paths[
            "completion_attestation"
        ].name,
        "learner_output_manifest_path": str(bundle_manifest),
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise LeptonJobError(f"completion marker {key} mismatch")
    for key in (
        "result_sha256",
        "completion_attestation_sha256",
        "learner_output_manifest_sha256",
        "learner_output_archive_sha256",
        "learner_output_inventory_sha256",
        "learner_output_tree_sha256",
    ):
        _sha(document[key], f"completion marker {key}")
    return document


def execute_request(
    *,
    contract_path: str | os.PathLike[str],
    contract_sha256: str,
    request_path: str | os.PathLike[str],
    request_sha256: str,
    semantic_attempt: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Execute one already-staged request from inside the Lepton Job."""

    contract = load_contract(contract_path, immutable=True)
    if contract.sha256 != _sha(contract_sha256, "backend contract SHA-256"):
        raise LeptonJobError("backend contract file SHA-256 mismatch")
    request = load_request(request_path, contract=contract, immutable=True)
    if request.sha256 != _sha(request_sha256, "run request SHA-256"):
        raise LeptonJobError("run request file SHA-256 mismatch")
    attempt = _true_int(semantic_attempt, "semantic attempt", minimum=1)
    if attempt > contract.semantic_max_attempts:
        raise LeptonJobError("semantic attempt exceeds the backend limit")
    secret = os.environ.get(WANDB_SECRET_ENV)
    if not secret:
        raise LeptonJobError("W&B credential was not injected by Lepton Secret")

    source_root = Path(str(request.paths["source_root"]))
    entrypoint = Path(str(contract.entrypoint))
    python = Path(str(contract.python))
    if source_root.is_symlink() or not source_root.is_dir():
        raise LeptonJobError("sealed learner source is missing or unsafe")
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise LeptonJobError("Lepton learner entrypoint is missing or unsafe")
    if not python.exists() or not os.access(python, os.X_OK):
        raise LeptonJobError("pinned Lepton learner Python is not executable")

    # These supply-chain gates run before runtime directories are created and
    # before any learner/W&B process is launched.  A semantic retry therefore
    # cannot train from drifted source or a drifted site-packages environment.
    verify_sealed_contract_source(contract, runner=runner)
    verify_python_environment_freeze(contract, runner=runner)

    marker = Path(str(request.paths["success_marker"]))
    checksum = Path(str(request.paths["success_checksum"]))
    result = Path(str(request.paths["learner_result"]))
    attestation = Path(str(request.paths["completion_attestation"]))
    if os.path.lexists(checksum) and not os.path.lexists(marker):
        raise LeptonJobError("completion checksum exists without its marker")
    if os.path.lexists(marker):
        evidence = validate_completion_marker(
            marker, request=request, contract=contract
        )
        for path, key, label in (
            (result, "result_sha256", "learner result"),
            (
                attestation,
                "completion_attestation_sha256",
                "learner completion attestation",
            ),
        ):
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise LeptonJobError(f"{label} is unavailable") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise LeptonJobError(f"{label} is unsafe")
            if sha256_file(path) != evidence[key]:
                raise LeptonJobError(f"{label} differs from completion marker")
        _validate_learner_output_bundle(
            contract=contract,
            request=request,
            manifest_sha256=evidence["learner_output_manifest_sha256"],
            archive_sha256=evidence["learner_output_archive_sha256"],
            inventory_sha256=evidence["learner_output_inventory_sha256"],
            tree_sha256=evidence["learner_output_tree_sha256"],
        )
        marker_sha = publish_checksum(marker, checksum)
        return {
            "schema": "fast_roborender.round_dagger.lepton_job_execution.v2",
            "status": "PASS",
            "publication": "adopted",
            "learner_identity_sha256": request.identity_sha256,
            "semantic_attempt": attempt,
            "success_marker_sha256": marker_sha,
            "result_sha256": evidence["result_sha256"],
            "learner_output_manifest_path": evidence[
                "learner_output_manifest_path"
            ],
            "learner_output_manifest_sha256": evidence[
                "learner_output_manifest_sha256"
            ],
            "learner_output_archive_sha256": evidence[
                "learner_output_archive_sha256"
            ],
            "learner_output_inventory_sha256": evidence[
                "learner_output_inventory_sha256"
            ],
            "learner_output_tree_sha256": evidence[
                "learner_output_tree_sha256"
            ],
        }

    runtime_root = (
        Path(str(contract.project_root)) / "runtime" / request.identity_sha256
    )
    runtime_paths = {
        "WANDB_DIR": Path(str(contract.project_root)) / "wandb",
        "WANDB_CACHE_DIR": Path(str(contract.project_root)) / ".cache" / "wandb",
        "WANDB_CONFIG_DIR": Path(str(contract.project_root)) / ".config" / "wandb",
        "HF_HOME": Path(str(contract.project_root)) / ".cache" / "huggingface",
        "TORCH_HOME": Path(str(contract.project_root)) / ".cache" / "torch",
        "XDG_CACHE_HOME": Path(str(contract.project_root)) / ".cache",
        "TMPDIR": runtime_root / "tmp",
    }
    for path in runtime_paths.values():
        if path.is_symlink():
            raise LeptonJobError("persistent runtime path is a symlink")
        path.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(request.environment)
    environment.update(
        {
            "ROUND_DAGGER_H100_PYTHON": str(python),
            "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": contract.image_digest,
            "ROUND_DAGGER_LEPTON_SEMANTIC_ATTEMPT": str(attempt),
            "GR00T_ROOT": str(source_root / "GR00T-VisualSim2Real"),
            "WANDB_MODE": "online",
            **{name: str(path) for name, path in runtime_paths.items()},
        }
    )
    completed = runner(
        [str(source_root / "scripts/round_dagger/run_h100_learner.sh")],
        env=environment,
        cwd=source_root,
        check=False,
    )
    if completed.returncode != 0:
        raise LeptonJobError(
            f"learner process failed with status {completed.returncode}"
        )

    from scripts.round_dagger import remote_learner_success

    # The marker and checksum are deliberately published *after* the exact
    # learner-output tree is packed.  This prevents a self-referential archive:
    # the separately fetched marker binds the immutable return bundle.
    if os.path.lexists(marker) or os.path.lexists(checksum):
        raise LeptonJobError("completion evidence appeared during learner execution")
    with tempfile.TemporaryDirectory(
        prefix="learner-result-proof-", dir=runtime_root
    ) as proof:
        document = remote_learner_success.publish(
            Path(proof) / "result_and_attestation.json",
            result,
            attestation,
            campaign_id=request.campaign_id,
            macro_round=request.macro_round,
            learner_code_sha256=request.environment[
                "ROUND_DAGGER_LEARNER_CODE_SHA256"
            ],
            learner_config_sha256=request.environment[
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
            ],
            learner_identity_sha256=request.identity_sha256,
        )
    bundle = _pack_learner_output(
        contract=contract,
        request=request,
        environment=environment,
        runner=runner,
    )
    for path, key, label in (
        (result, "result_sha256", "learner result"),
        (
            attestation,
            "completion_attestation_sha256",
            "learner completion attestation",
        ),
    ):
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise LeptonJobError(
                f"{label} disappeared after bundle publication"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or sha256_file(path) != document[key]
        ):
            raise LeptonJobError(f"{label} changed during bundle publication")
    bundle_paths = learner_output_bundle_paths(contract, request)
    document = {
        **document,
        "schema": _LEARNER_OUTPUT_SUCCESS_SCHEMA,
        "learner_output_manifest_path": str(bundle_paths["manifest"]),
        "learner_output_manifest_sha256": bundle["manifest_sha256"],
        "learner_output_archive_sha256": bundle["archive_sha256"],
        "learner_output_inventory_sha256": bundle["inventory_sha256"],
        "learner_output_tree_sha256": bundle["tree_sha256"],
    }
    _publish_immutable_json(marker, document)
    marker_sha = publish_checksum(marker, checksum)
    return {
        "schema": "fast_roborender.round_dagger.lepton_job_execution.v2",
        "status": "PASS",
        "publication": "published",
        "learner_identity_sha256": request.identity_sha256,
        "semantic_attempt": attempt,
        "success_marker_sha256": marker_sha,
        "result_sha256": document["result_sha256"],
        "learner_output_manifest_path": document["learner_output_manifest_path"],
        "learner_output_manifest_sha256": document[
            "learner_output_manifest_sha256"
        ],
        "learner_output_archive_sha256": document[
            "learner_output_archive_sha256"
        ],
        "learner_output_inventory_sha256": document[
            "learner_output_inventory_sha256"
        ],
        "learner_output_tree_sha256": document["learner_output_tree_sha256"],
    }


def fetch_completion_pair(
    *,
    transfer_program: Path,
    deployment: str,
    remote_marker: PurePosixPath,
    command_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> tuple[Path, Path, tempfile.TemporaryDirectory[str]]:
    """Fetch marker and checksum through the authenticated control deployment.

    The returned TemporaryDirectory owner must remain alive while the paths are
    used.  Authentication is delegated to the transfer program's SDK profile;
    no token is accepted by this API or placed in argv.
    """

    if deployment != CONTROL_DEPLOYMENT:
        raise LeptonJobError("completion fetch deployment is not canonical")
    remote = _project_path(str(remote_marker), "remote completion marker")
    temporary = tempfile.TemporaryDirectory(prefix="roborender-lepton-completion-")
    marker = Path(temporary.name) / remote.name
    checksum = Path(f"{marker}.sha256")
    for remote_path, local_path in (
        (remote, marker),
        (PurePosixPath(f"{remote}.sha256"), checksum),
    ):
        completed = command_runner(
            [
                str(transfer_program),
                "--deployment",
                deployment,
                "download",
                "--remote",
                str(remote_path),
                "--local",
                str(local_path),
                "--mode",
                "0444",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            temporary.cleanup()
            raise LeptonJobError("could not fetch exact Lepton completion evidence")
    verify_checksum(marker, checksum)
    return marker, checksum, temporary
