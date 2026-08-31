"""Rank identity for independent, frozen round-actor processes.

Actor collection performs no gradient synchronization.  Keeping its logical
rank separate from Torch's distributed environment lets each Isaac process run
on one isolated GPU without creating a CUDA/NCCL process group.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping


_INTEGER = re.compile(r"0|[1-9][0-9]*")


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
    "renderer_slot",
]
