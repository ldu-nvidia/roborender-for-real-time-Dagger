"""Exact collection identities for the native-H200 BC/DAgger protocol."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NativeActorGeneration:
    """One immutable collection generation in the native-H200 protocol."""

    generation_id: str
    macro_round: int
    stage: str
    actor_role: str
    beta: float
    beta_text: str
    parent_generation_id: str | None
    train_round: str


NATIVE_ACTOR_GENERATIONS = (
    NativeActorGeneration(
        "gen-000-bc", 0, "bc", "bc_collect", 1.0, "1.0", None, "BC/isaac"
    ),
    NativeActorGeneration(
        "gen-001-dagger-beta-0p75",
        1,
        "dagger",
        "dagger_collect",
        0.75,
        "0.75",
        "gen-000-bc",
        "dagger_1",
    ),
    NativeActorGeneration(
        "gen-002-dagger-beta-0p50",
        2,
        "dagger",
        "dagger_collect",
        0.5,
        "0.50",
        "gen-001-dagger-beta-0p75",
        "dagger_2",
    ),
    NativeActorGeneration(
        "gen-003-dagger-beta-0p25",
        3,
        "dagger",
        "dagger_collect",
        0.25,
        "0.25",
        "gen-002-dagger-beta-0p50",
        "dagger_3",
    ),
    NativeActorGeneration(
        "gen-004-dagger-beta-0p00",
        4,
        "dagger",
        "dagger_collect",
        0.0,
        "0.00",
        "gen-003-dagger-beta-0p25",
        "dagger_4",
    ),
)
GENERATION_IDS = tuple(item.generation_id for item in NATIVE_ACTOR_GENERATIONS)
_BY_ID = {item.generation_id: item for item in NATIVE_ACTOR_GENERATIONS}
TRAIN_ROUND_RELATIVE_ROOTS = {
    item.generation_id: item.train_round
    for item in NATIVE_ACTOR_GENERATIONS
}
NATIVE_SHARD_ID = re.compile(
    rf"^(?P<generation>{'|'.join(map(re.escape, GENERATION_IDS))})"
    r"__wave-(?P<wave>[0-9]{3})__gpu-(?P<gpu>[0-9]{2})"
    r"__shard-(?P<shard>[0-9]{6})$"
)


def native_actor_generation(generation_id: str) -> NativeActorGeneration:
    """Resolve only one of the five protocol generation IDs."""

    try:
        return _BY_ID[generation_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "native actor generation must be one of: " + ", ".join(GENERATION_IDS)
        ) from exc


def train_round_relative_root(generation_id: str) -> str:
    """Return the one canonical train-round root for a generation identity."""

    native_actor_generation(generation_id)
    return TRAIN_ROUND_RELATIVE_ROOTS[generation_id]


__all__ = (
    "GENERATION_IDS",
    "NATIVE_ACTOR_GENERATIONS",
    "NATIVE_SHARD_ID",
    "TRAIN_ROUND_RELATIVE_ROOTS",
    "NativeActorGeneration",
    "native_actor_generation",
    "train_round_relative_root",
)
