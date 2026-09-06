"""Deterministic, provenance-bound actor seeds for production round DAgger.

Isaac/GR00T starts one independent process per actor rank.  The trainer adds the
rank to Hydra's top-level ``seed`` before constructing the simulator, while the
Image3F generation seed is consumed directly.  This module captures that exact
runtime behavior in a small immutable contract so sequential cohorts cannot
silently reuse the same physical RNG streams.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


ACTOR_SEED_CONTRACT_SCHEMA = "fast_roborender.round_dagger.actor_seed_contract.v1"
ACTOR_SEED_ALGORITHM = "sha256-canonical-json-u31-v1"
SIMULATOR_RANK_OFFSET = "train_agent_trl.actor_process_identity.rank"

_DOMAIN = b"fast_roborender.round_dagger.actor_seed.v1\x00"
_MAX_RUNTIME_SEED = (1 << 31) - 1
_CONTRACT_KEYS = {
    "schema",
    "algorithm",
    "simulator_rank_offset",
    "campaign_id",
    "macro_round",
    "cohort_index",
    "simulator_base_seed",
    "assignments",
}
_ASSIGNMENT_KEYS = {
    "rank",
    "simulator_seed",
    "generation_seed",
    "global_env_ids",
}


class ActorSeedError(ValueError):
    """An actor seed contract is malformed or does not match its cohort."""


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ActorSeedError(f"{label} must be an integer >= {minimum}")
    return value


def _campaign_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ActorSeedError("campaign_id must be a non-empty string of at most 128 bytes")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ActorSeedError("campaign_id must be ASCII") from exc
    return value


def _sorted_unique(values: Sequence[int], label: str) -> list[int]:
    parsed = [_integer(item, f"{label}[]") for item in values]
    if parsed != sorted(set(parsed)):
        raise ActorSeedError(f"{label} must be sorted and unique")
    if not parsed:
        raise ActorSeedError(f"{label} must not be empty")
    return parsed


def _seed(identity: Mapping[str, Any], role: str, rank: int, *, modulus: int) -> int:
    payload = json.dumps(
        {**dict(identity), "role": role, "rank": rank},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    # Reserve zero, which several simulator libraries treat as a special/default
    # seed, and constrain the value to the signed 31-bit range shared by NumPy,
    # Torch, Hydra, and the Isaac configuration path.
    return 1 + int.from_bytes(hashlib.sha256(_DOMAIN + payload).digest()[:8], "big") % modulus


def derive_actor_seed_contract(
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    expected_ranks: Sequence[int],
    expected_global_env_ids: Sequence[int],
) -> dict[str, Any]:
    """Derive the exact simulator and Image3F seed assignment for one cohort."""

    campaign_id = _campaign_id(campaign_id)
    macro_round = _integer(macro_round, "macro_round")
    cohort_index = _integer(cohort_index, "cohort_index")
    ranks = _sorted_unique(expected_ranks, "expected_ranks")
    envs = _sorted_unique(expected_global_env_ids, "expected_global_env_ids")
    if len(envs) % len(ranks):
        raise ActorSeedError(
            "expected_global_env_ids must divide evenly across actor ranks"
        )
    identity = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
    }
    maximum_rank = max(ranks)
    base_modulus = _MAX_RUNTIME_SEED - maximum_rank
    if base_modulus < 1:
        raise ActorSeedError("actor rank is too large for the simulator seed range")
    simulator_base_seed = _seed(
        identity, "simulator_base", 0, modulus=base_modulus
    )
    environments_per_rank = len(envs) // len(ranks)
    assignments: list[dict[str, Any]] = []
    for offset, rank in enumerate(ranks):
        start = offset * environments_per_rank
        assigned_envs = envs[start : start + environments_per_rank]
        assignments.append(
            {
                "rank": rank,
                # train_agent_trl adds the independent actor rank exactly once.
                "simulator_seed": simulator_base_seed + rank,
                "generation_seed": _seed(
                    identity,
                    "image3f_generation",
                    rank,
                    modulus=_MAX_RUNTIME_SEED,
                ),
                "global_env_ids": assigned_envs,
            }
        )
    return {
        "schema": ACTOR_SEED_CONTRACT_SCHEMA,
        "algorithm": ACTOR_SEED_ALGORITHM,
        "simulator_rank_offset": SIMULATOR_RANK_OFFSET,
        **identity,
        "simulator_base_seed": simulator_base_seed,
        "assignments": assignments,
    }


def validate_actor_seed_contract(
    value: Any,
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    expected_ranks: Sequence[int],
    expected_global_env_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate and recompute a seed contract instead of trusting its values."""

    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        raise ActorSeedError("actor seed contract key-set mismatch")
    document = dict(value)
    if document.get("schema") != ACTOR_SEED_CONTRACT_SCHEMA:
        raise ActorSeedError("actor seed contract schema mismatch")
    if document.get("algorithm") != ACTOR_SEED_ALGORITHM:
        raise ActorSeedError("actor seed algorithm mismatch")
    if document.get("simulator_rank_offset") != SIMULATOR_RANK_OFFSET:
        raise ActorSeedError("actor simulator rank-offset contract mismatch")
    assignments = document.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ActorSeedError("actor seed assignments must be a non-empty list")
    for index, raw in enumerate(assignments):
        if not isinstance(raw, Mapping) or set(raw) != _ASSIGNMENT_KEYS:
            raise ActorSeedError(f"actor seed assignment {index} key-set mismatch")
        _integer(raw["rank"], f"assignments[{index}].rank")
        _integer(raw["simulator_seed"], f"assignments[{index}].simulator_seed", minimum=1)
        _integer(raw["generation_seed"], f"assignments[{index}].generation_seed", minimum=1)
        if raw["simulator_seed"] > _MAX_RUNTIME_SEED or raw["generation_seed"] > _MAX_RUNTIME_SEED:
            raise ActorSeedError("actor seed exceeds the signed 31-bit runtime range")
        if not isinstance(raw["global_env_ids"], list):
            raise ActorSeedError("actor seed assignment global_env_ids must be a list")
        _sorted_unique(raw["global_env_ids"], f"assignments[{index}].global_env_ids")
    expected = derive_actor_seed_contract(
        campaign_id=campaign_id,
        macro_round=macro_round,
        cohort_index=cohort_index,
        expected_ranks=expected_ranks,
        expected_global_env_ids=expected_global_env_ids,
    )
    if document != expected:
        raise ActorSeedError("actor seed contract differs from deterministic derivation")
    simulator = [item["simulator_seed"] for item in assignments]
    generation = [item["generation_seed"] for item in assignments]
    if len(simulator) != len(set(simulator)) or len(generation) != len(set(generation)):
        raise ActorSeedError("actor seed assignments collide within the cohort")
    return document


def rank_seed_assignment(contract: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """Return one rank assignment from an already validated contract."""

    rank = _integer(rank, "rank")
    assignments = contract.get("assignments")
    if not isinstance(assignments, list):
        raise ActorSeedError("actor seed contract lacks assignments")
    matches = [dict(item) for item in assignments if isinstance(item, Mapping) and item.get("rank") == rank]
    if len(matches) != 1:
        raise ActorSeedError(f"actor seed contract does not contain exactly one rank {rank}")
    return matches[0]


__all__ = [
    "ACTOR_SEED_ALGORITHM",
    "ACTOR_SEED_CONTRACT_SCHEMA",
    "ActorSeedError",
    "derive_actor_seed_contract",
    "rank_seed_assignment",
    "validate_actor_seed_contract",
]
