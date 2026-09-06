"""Rank identity for independent, frozen round-actor processes.

Actor collection performs no gradient synchronization.  Keeping its logical
rank separate from Torch's distributed environment lets each Isaac process run
on one isolated GPU without creating a CUDA/NCCL process group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_INTEGER = re.compile(r"0|[1-9][0-9]*")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_DEVICE_NAMESPACE_SCHEMA = (
    "fast_roborender.round_dagger.native_actor_gpu_device_namespace.v1"
)


class ActorIdentityError(RuntimeError):
    """The independent actor rank environment is incomplete or invalid."""


@dataclass(frozen=True)
class ActorProcessIdentity:
    rank: int
    world_size: int


def actor_process_identity(
    environment: Mapping[str, str] | None = None,
) -> ActorProcessIdentity | None:
    """Return an explicitly configured non-DDP actor identity, if any."""

    values = os.environ if environment is None else environment
    rank_raw = values.get("ROUND_DAGGER_ACTOR_RANK")
    world_raw = values.get("ROUND_DAGGER_ACTOR_WORLD_SIZE")
    if rank_raw is None and world_raw is None:
        return None
    if rank_raw is None or world_raw is None:
        raise ActorIdentityError(
            "ROUND_DAGGER_ACTOR_RANK and ROUND_DAGGER_ACTOR_WORLD_SIZE "
            "must be set together"
        )
    if not _INTEGER.fullmatch(rank_raw) or not _INTEGER.fullmatch(world_raw):
        raise ActorIdentityError("independent actor rank fields must be canonical integers")
    rank = int(rank_raw)
    world_size = int(world_raw)
    if world_size < 1:
        raise ActorIdentityError("independent actor world size must be positive")
    if rank >= world_size:
        raise ActorIdentityError("independent actor rank must be smaller than world size")
    return ActorProcessIdentity(rank=rank, world_size=world_size)


def native_actor_cuda_device(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Return the rank-local CUDA device for an independent native actor.

    Each Isaac/Kit process receives a private device mount containing only its
    assigned physical H200.  ``CUDA_VISIBLE_DEVICES`` must be absent because Kit
    warns that CUDA and Omniverse enumeration differ whenever that remapping is
    active.  With one NVIDIA/DRM device exposed, every subsystem safely uses
    process-local index zero.
    """

    values = os.environ if environment is None else environment
    identity = actor_process_identity(values)
    if identity is None or values.get("ROUND_DAGGER_OBSERVATION_SOURCE") != "native":
        return None
    assigned_uuid = values.get("ROUND_DAGGER_ASSIGNED_GPU_UUID")
    assigned_physical_index_raw = values.get(
        "ROUND_DAGGER_ASSIGNED_PHYSICAL_INDEX"
    )
    expected_raw = values.get("ROUND_DAGGER_EXPECTED_PHYSICAL_GPU_JSON")
    namespace_raw = values.get("ROUND_DAGGER_GPU_DEVICE_NAMESPACE_JSON")
    namespace_sha256 = values.get("ROUND_DAGGER_GPU_DEVICE_NAMESPACE_SHA256")
    if "CUDA_VISIBLE_DEVICES" in values or "NVIDIA_VISIBLE_DEVICES" in values:
        raise ActorIdentityError(
            "native actor private device namespace forbids CUDA/NVIDIA visibility remapping"
        )
    if not assigned_uuid or _GPU_UUID.fullmatch(assigned_uuid) is None:
        raise ActorIdentityError("native actor assigned H200 UUID is malformed")
    if not namespace_sha256 or _SHA256.fullmatch(namespace_sha256) is None:
        raise ActorIdentityError("native actor GPU namespace SHA-256 is malformed")
    if (
        not isinstance(assigned_physical_index_raw, str)
        or _INTEGER.fullmatch(assigned_physical_index_raw) is None
        or int(assigned_physical_index_raw) not in range(8)
    ):
        raise ActorIdentityError("native actor physical GPU index is malformed")
    assigned_physical_index = int(assigned_physical_index_raw)
    try:
        expected = json.loads(expected_raw or "")
        gpu_namespace = json.loads(namespace_raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ActorIdentityError(
            "native actor physical GPU identity is malformed"
        ) from exc
    canonical_namespace = json.dumps(
        gpu_namespace, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if (
        not isinstance(expected, dict)
        or expected.get("rank") != assigned_physical_index
        or expected.get("physical_index") != assigned_physical_index
        or expected.get("uuid") != assigned_uuid
        or not isinstance(gpu_namespace, dict)
        or gpu_namespace.get("schema") != _GPU_DEVICE_NAMESPACE_SCHEMA
        or gpu_namespace.get("rank") != assigned_physical_index
        or gpu_namespace.get("physical_index") != assigned_physical_index
        or gpu_namespace.get("uuid") != assigned_uuid
        or gpu_namespace.get("pci_bus_id") != expected.get("pci_bus_id")
        or hashlib.sha256(canonical_namespace).hexdigest() != namespace_sha256
    ):
        raise ActorIdentityError(
            "private GPU device namespace differs from the attested physical H200"
        )
    return "cuda:0"


def wait_for_native_actor_fleet_release(rank: int) -> None:
    """Hold a runtime-ready actor until the controller releases all ranks."""

    raw_release = os.environ.get("ROUND_DAGGER_FLEET_RELEASE_FILE", "")
    raw_generation = os.environ.get("ROUND_DAGGER_GENERATION_ROOT", "")
    release = Path(raw_release)
    generation = Path(raw_generation)
    if (
        not release.is_absolute()
        or not generation.is_absolute()
        or Path(os.path.normpath(raw_release)) != release
        or Path(os.path.normpath(raw_generation)) != generation
        or release.name != "kit-ready.release"
    ):
        raise ActorIdentityError("native actor fleet release path is malformed")
    try:
        release.relative_to(generation / "logs")
    except ValueError as exc:
        raise ActorIdentityError(
            "native actor fleet release path leaves generation logs"
        ) from exc
    while True:
        try:
            if release.is_symlink():
                raise ActorIdentityError(
                    "native actor fleet release path is a symlink"
                )
            payload = release.read_bytes()
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if payload != b"release\n":
            raise ActorIdentityError("native actor fleet release token differs")
        break
    print(f"NATIVE_ACTOR_FLEET_RELEASED rank={rank}", flush=True)


def renderer_slot(
    *, rank: int, local_env_count: int, local_env_index: int, renderer_count: int
) -> int:
    """Map a rank-local environment onto the ordered persistent renderer fleet."""

    values = (rank, local_env_count, local_env_index, renderer_count)
    if any(type(value) is not int for value in values):
        raise ActorIdentityError("renderer slot inputs must be true integers")
    if rank < 0 or local_env_count < 1 or renderer_count < 1:
        raise ActorIdentityError("renderer slot dimensions are invalid")
    if local_env_index < 0 or local_env_index >= local_env_count:
        raise ActorIdentityError("local environment index is out of range")
    return (rank * local_env_count + local_env_index) % renderer_count


__all__ = [
    "ActorIdentityError",
    "ActorProcessIdentity",
    "actor_process_identity",
    "native_actor_cuda_device",
    "wait_for_native_actor_fleet_release",
    "renderer_slot",
]
