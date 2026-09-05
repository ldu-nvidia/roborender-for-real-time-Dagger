#!/usr/bin/env python3
"""Camera-free cumulative-replay learner for round-based DAgger.

The learner deliberately has no Isaac, PhysX, camera, or RoboRender imports.  It
consumes an immutable replay snapshot, assigns whole rollout shards to torchrun
ranks deterministically, reconstructs the exact policy observation dictionary,
and minimizes supervised action error against the expert labels.

The public adapter boundary keeps the data/restart machinery independent from
the concrete GR00T policy class.  Production code can therefore instantiate the
policy from its resolved bundle while unit tests use a tiny CPU module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
import zipfile
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from round_dagger.artifacts import (
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.actor import ActorExportError, runtime_state_dict_sha256
from round_dagger.gr00t_policy import GR00TPolicyError, validate_policy_spec
from round_dagger.learner_identity import (
    LearnerIdentityError,
    learner_identity_sha256 as compute_learner_identity_sha256,
)
from round_dagger.policy_bundle import (
    BOOTSTRAP_MANIFEST_SCHEMA as POLICY_BUNDLE_MANIFEST_SCHEMA,
    PolicyBundleError,
    validate_policy_bundle_identity,
)
from round_dagger.policy_identity import (
    PolicyIdentityError,
    runtime_policy_identity_sha256,
    validate_homie_controller_sha256,
)
from round_dagger.replay import load_snapshot
from round_dagger.checkpoint_compat import (
    CheckpointCompatibilityError,
    load_restricted_restart_document,
)
from round_dagger.native_actor_protocol import GENERATION_IDS, train_round_relative_root


BUNDLE_SCHEMA = "fast_roborender.round_dagger.train_only_learner_bundle.v2"
CANONICAL_CAMPAIGN_ID = "dagger-redo-v001"
CANONICAL_CAMPAIGN_ROOT = Path(
    "/workspace/project/roborender/data"
)
CHECKPOINT_SCHEMA = "fast_roborender.round_dagger.learner_checkpoint.v4"
LEGACY_RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v4"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v3"
HELDOUT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_heldout_telemetry.v1"
)
TRAINING_QUALITY_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_training_quality.v2"
)
TRAINING_CONTEXT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_training_context.v1"
)
TRAINING_QUALITY_PROBE_POLICY = (
    "lineage_stratified_current_historical_replay.v1"
)
TRAINING_QUALITY_PROBE_BUDGET = 256
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_GENERATION_RE = re.compile(
    rf"(?:{'|'.join(map(re.escape, GENERATION_IDS))})\Z"
)
CUDA_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SINGLE_PROCESS_GRADIENT_REDUCTION = "single_process_v1"
ORDERED_ALLGATHER_GRADIENT_REDUCTION = "ordered_allgather_rank_sum_v1"
_PROBE_SEED_DOMAIN = b"fast_roborender.round_dagger.loss_probe.v1"
_TRAINING_QUALITY_PROBE_DOMAIN = (
    b"fast_roborender.round_dagger.training_quality_probe.v1"
)


class LearnerError(RuntimeError):
    """The replay, learner bundle, or restart contract is invalid."""


def configure_deterministic_runtime(device: torch.device | str) -> None:
    """Enforce the bit-exact CUDA restart contract before any learner kernels.

    RNG restoration alone is insufficient for a convolutional CUDA learner:
    cuDNN and cuBLAS may otherwise select/reduce with nondeterministic kernels.
    ``CUBLAS_WORKSPACE_CONFIG`` must be present in the process environment
    before CUDA initializes, so this function validates rather than silently
    installing it after the fact.  The production launcher exports the one
    pinned value before starting Python.
    """

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_workspace != CUDA_CUBLAS_WORKSPACE_CONFIG:
        raise LearnerError(
            "CUDA learner requires CUBLAS_WORKSPACE_CONFIG="
            f"{CUDA_CUBLAS_WORKSPACE_CONFIG!r} before Python starts; "
            f"got {actual_workspace!r}"
        )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception as exc:
        raise LearnerError(
            f"cannot configure deterministic CUDA learner runtime: {exc}"
        ) from exc
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cuda.matmul.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise LearnerError("deterministic CUDA learner runtime did not take effect")


def _ordered_allgather_gradient_hook(
    world_size: int, bucket
):
    """Average one DDP bucket without a floating-point reduction collective.

    NCCL all-reduce is numerically valid but its summation topology is not a
    restart-stable semantic contract: recreating an eight-rank communicator
    can change the order of floating-point additions.  That was sufficient to
    change an Adam second moment on H200 even when the final model weights
    happened to round to the same bytes.  All-gather transports each rank's
    unreduced bytes, after which every rank performs the same elementwise adds
    in explicit rank order.  The result is therefore independent of NCCL ring
    construction while retaining DDP's averaged-gradient semantics.
    """

    if world_size <= 1:
        raise LearnerError("ordered DDP reduction requires multiple ranks")
    reduced = _ordered_allgather_sum(bucket.buffer(), world_size)
    reduced.div_(world_size)
    # Leave the public hook signature unannotated for ``bucket`` and the
    # return value.  This module uses postponed annotations, while DDP's hook
    # validator compares those two annotations to live Python types instead
    # of resolving them first (PyTorch 2.13 included).  An otherwise-correct
    # annotation is therefore seen as a string and rejected at registration.
    future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    future.set_result(reduced)
    return future


def _ordered_allgather_sum(source: torch.Tensor, world_size: int) -> torch.Tensor:
    """Sum identically-shaped tensors in explicit rank order on every rank."""

    if world_size <= 1:
        raise LearnerError("ordered distributed sum requires multiple ranks")
    gathered = [torch.empty_like(source) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, source)
    reduced = gathered[0].clone()
    for rank in range(1, world_size):
        reduced.add_(gathered[rank])
    return reduced


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LearnerError(f"{label} must be a lowercase SHA-256")
    return value


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LearnerError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise LearnerError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LearnerError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise LearnerError(f"{label} must be a finite {qualifier}number")
    return result


def _require_campaign_storage_path(
    value: str | os.PathLike[str],
    label: str,
    *,
    generation_id: str | None = None,
) -> Path:
    """Confine canonical learner inputs and outputs to the local H200 data root."""

    root = CANONICAL_CAMPAIGN_ROOT.resolve(strict=True)
    required_root = (
        root
        if generation_id is None
        else (root / train_round_relative_root(generation_id)).resolve(strict=True)
    )
    path = Path(value).expanduser()
    resolved = path.resolve(strict=path.exists())
    try:
        resolved.relative_to(required_root)
    except ValueError as exc:
        raise LearnerError(f"{label} lies outside canonical campaign storage") from exc
    return resolved


@dataclass(frozen=True)
class LearnerBundleConfig:
    """Immutable policy/input/optimizer contract shared by every learner rank."""

    campaign_id: str
    generation_id: str
    macro_round: int
    model_lineage_sha256: str
    code_sha256: str
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    rgb_input_key: str = "vision_obs"
    rgb_layout: str = "NHWC"
    output_key: str = "actions"
    observation_key_map: tuple[tuple[str, str], ...] = ()
    train_action_slice: tuple[int, int] | None = None
    batch_size: int = 64
    target_optimizer_examples: int = 4096
    checkpoint_every_optimizer_examples: int = 1024
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    seed: int = 0
    partition_seed: int = 0
    initial_checkpoint_sha256: str | None = None
    model_config_json: str = "{}"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LearnerBundleConfig":
        if not isinstance(raw, Mapping):
            raise LearnerError("learner bundle must be a mapping")
        forbidden = sorted(
            key
            for key in raw
            if isinstance(key, str)
            and any(
                token in key.lower()
                for token in ("validation", "heldout", "held_out", "test", "eval")
            )
        )
        if forbidden:
            raise LearnerError(
                "train-only learner bundle forbids validation/test fields: "
                + ", ".join(forbidden)
            )
        allowed = {
            "schema",
            "campaign_id",
            "generation_id",
            "macro_round",
            "model_lineage_sha256",
            "code_sha256",
            "rgb_mean",
            "rgb_std",
            "rgb_input_key",
            "rgb_layout",
            "output_key",
            "observation_key_map",
            "train_action_slice",
            "batch_size",
            "target_optimizer_examples",
            "checkpoint_every_optimizer_examples",
            "learning_rate",
            "weight_decay",
            "seed",
            "partition_seed",
            "initial_checkpoint_sha256",
            "model_config",
        }
        unknown = set(raw) - allowed
        required = {
            "schema",
            "campaign_id",
            "generation_id",
            "macro_round",
            "model_lineage_sha256",
            "code_sha256",
            "rgb_mean",
            "rgb_std",
        }
        missing = required - set(raw)
        if missing or unknown:
            raise LearnerError(
                f"learner bundle has missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        if raw["schema"] != BUNDLE_SCHEMA:
            raise LearnerError("learner bundle schema mismatch")
        campaign_id = raw["campaign_id"]
        if campaign_id != CANONICAL_CAMPAIGN_ID:
            raise LearnerError(
                f"campaign_id must be exactly {CANONICAL_CAMPAIGN_ID!r}"
            )
        generation_id = raw["generation_id"]
        if not isinstance(generation_id, str) or _GENERATION_RE.fullmatch(
            generation_id
        ) is None:
            raise LearnerError("generation_id is not a canonical native-H200 generation")
        macro_round = _true_int(raw["macro_round"], "macro_round")
        expected_generation = (
            GENERATION_IDS[macro_round]
            if macro_round < len(GENERATION_IDS)
            else None
        )
        if generation_id != expected_generation:
            raise LearnerError("generation_id does not match macro_round")

        def channel_vector(name: str, *, positive: bool) -> tuple[float, float, float]:
            values = raw[name]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise LearnerError(f"{name} must contain exactly three numbers")
            parsed = tuple(
                _finite_float(value, f"{name}[{index}]", positive=positive)
                for index, value in enumerate(values)
            )
            if len(parsed) != 3:
                raise LearnerError(f"{name} must contain exactly three numbers")
            return parsed  # type: ignore[return-value]

        key_map_raw = raw.get("observation_key_map", {})
        if not isinstance(key_map_raw, Mapping):
            raise LearnerError("observation_key_map must be a mapping")
        key_map: list[tuple[str, str]] = []
        destinations: set[str] = set()
        for source, destination in sorted(key_map_raw.items()):
            if (
                not isinstance(source, str)
                or not source.startswith("obs__")
                or _SAFE_KEY_RE.fullmatch(source) is None
                or not isinstance(destination, str)
                or _SAFE_KEY_RE.fullmatch(destination) is None
            ):
                raise LearnerError("observation_key_map contains an unsafe key")
            if destination in destinations:
                raise LearnerError("observation_key_map destinations must be unique")
            destinations.add(destination)
            key_map.append((source, destination))

        action_slice_raw = raw.get("train_action_slice")
        action_slice: tuple[int, int] | None = None
        if action_slice_raw is not None:
            if (
                not isinstance(action_slice_raw, Sequence)
                or isinstance(action_slice_raw, (str, bytes))
                or len(action_slice_raw) != 2
            ):
                raise LearnerError("train_action_slice must be [start, stop]")
            start = _true_int(action_slice_raw[0], "train_action_slice[0]")
            stop = _true_int(action_slice_raw[1], "train_action_slice[1]", minimum=1)
            if start >= stop:
                raise LearnerError("train_action_slice must have start < stop")
            action_slice = (start, stop)

        rgb_input_key = raw.get("rgb_input_key", "vision_obs")
        output_key = raw.get("output_key", "actions")
        if (
            not isinstance(rgb_input_key, str)
            or _SAFE_KEY_RE.fullmatch(rgb_input_key) is None
        ):
            raise LearnerError("rgb_input_key is unsafe")
        if (
            not isinstance(output_key, str)
            or _SAFE_KEY_RE.fullmatch(output_key) is None
        ):
            raise LearnerError("output_key is unsafe")
        rgb_layout = raw.get("rgb_layout", "NHWC")
        if rgb_layout not in {"NHWC", "NCHW"}:
            raise LearnerError("rgb_layout must be NHWC or NCHW")
        initial_sha = raw.get("initial_checkpoint_sha256")
        if initial_sha is not None:
            initial_sha = _require_sha256(initial_sha, "initial_checkpoint_sha256")
        model_config = raw.get("model_config", {})
        if not isinstance(model_config, Mapping):
            raise LearnerError("model_config must be a mapping")
        try:
            model_config_json = json.dumps(
                dict(model_config),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise LearnerError(
                "model_config must contain canonical JSON values"
            ) from exc
        weight_decay = _finite_float(raw.get("weight_decay", 0.0), "weight_decay")
        if weight_decay < 0.0:
            raise LearnerError("weight_decay must be non-negative")
        return cls(
            campaign_id=campaign_id,
            generation_id=generation_id,
            macro_round=macro_round,
            model_lineage_sha256=_require_sha256(
                raw["model_lineage_sha256"], "model_lineage_sha256"
            ),
            code_sha256=_require_sha256(raw["code_sha256"], "code_sha256"),
            rgb_mean=channel_vector("rgb_mean", positive=False),
            rgb_std=channel_vector("rgb_std", positive=True),
            rgb_input_key=rgb_input_key,
            rgb_layout=rgb_layout,
            output_key=output_key,
            observation_key_map=tuple(key_map),
            train_action_slice=action_slice,
            batch_size=_true_int(raw.get("batch_size", 64), "batch_size", minimum=1),
            target_optimizer_examples=_true_int(
                raw.get("target_optimizer_examples", 4096),
                "target_optimizer_examples",
                minimum=1,
            ),
            checkpoint_every_optimizer_examples=_true_int(
                raw.get("checkpoint_every_optimizer_examples", 1024),
                "checkpoint_every_optimizer_examples",
                minimum=1,
            ),
            learning_rate=_finite_float(
                raw.get("learning_rate", 1.0e-4), "learning_rate", positive=True
            ),
            weight_decay=weight_decay,
            seed=_true_int(raw.get("seed", 0), "seed"),
            partition_seed=_true_int(raw.get("partition_seed", 0), "partition_seed"),
            initial_checkpoint_sha256=initial_sha,
            model_config_json=model_config_json,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LearnerBundleConfig":
        return cls.from_mapping(load_strict_json(path))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": self.campaign_id,
            "generation_id": self.generation_id,
            "macro_round": self.macro_round,
            "model_lineage_sha256": self.model_lineage_sha256,
            "code_sha256": self.code_sha256,
            "rgb_mean": list(self.rgb_mean),
            "rgb_std": list(self.rgb_std),
            "rgb_input_key": self.rgb_input_key,
            "rgb_layout": self.rgb_layout,
            "output_key": self.output_key,
            "observation_key_map": dict(self.observation_key_map),
            "train_action_slice": (
                list(self.train_action_slice)
                if self.train_action_slice is not None
                else None
            ),
            "batch_size": self.batch_size,
            "target_optimizer_examples": self.target_optimizer_examples,
            "checkpoint_every_optimizer_examples": (
                self.checkpoint_every_optimizer_examples
            ),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "partition_seed": self.partition_seed,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "model_config": json.loads(self.model_config_json),
        }
        return result

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_mapping())


class PolicyAdapter(Protocol):
    """Converts a policy observation mapping into an action tensor."""

    def actions(
        self, model: nn.Module, observations: Mapping[str, torch.Tensor]
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class MappingPolicyAdapter:
    """Default adapter for policies accepting one observation mapping."""

    output_key: str = "actions"

    def actions(
        self, model: nn.Module, observations: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        output = model(observations)
        if isinstance(output, torch.Tensor):
            return output
        if not isinstance(output, Mapping) or self.output_key not in output:
            raise LearnerError(
                f"policy output must be a tensor or contain {self.output_key!r}"
            )
        actions = output[self.output_key]
        if not isinstance(actions, torch.Tensor):
            raise LearnerError("policy action output is not a torch tensor")
        return actions


def _balanced_shard_assignments(
    resolved_shards: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Balance immutable shards by transition count with deterministic LPT.

    The learner takes the same number of optimizer steps on every DDP rank and
    samples uniformly inside each rank-local replay partition.  Independent
    hash bucketing can therefore overweight transitions assigned to ranks with
    small partitions.  Longest-processing-time-first assignment minimizes that
    skew while retaining deterministic, input-order-independent ownership.
    """

    world_size = _true_int(world_size, "world_size", minimum=1)
    seed = _true_int(seed, "partition seed")
    normalized: list[tuple[int, str, str, dict[str, Any]]] = []
    identities: set[str] = set()
    for raw in resolved_shards:
        if not isinstance(raw, Mapping):
            raise LearnerError("resolved shard must be a mapping")
        logical_id = raw.get("logical_id")
        shard_id = raw.get("shard_id")
        if not isinstance(logical_id, str) or not logical_id:
            raise LearnerError("resolved shard logical_id must be a non-empty string")
        if not isinstance(shard_id, str) or not shard_id:
            raise LearnerError("resolved shard shard_id must be a non-empty string")
        identity = f"{logical_id}/{shard_id}"
        if identity in identities:
            raise LearnerError("resolved replay contains duplicate shard identity")
        identities.add(identity)
        transitions = _true_int(
            raw.get("transition_count"),
            f"transition_count for replay shard {identity}",
            minimum=1,
        )
        seeded_digest = hashlib.sha256(
            f"{seed}:{identity}".encode("utf-8")
        ).hexdigest()
        normalized.append((transitions, seeded_digest, identity, dict(raw)))

    assignments: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    frontier = max(
        _true_int(item[3].get("macro_round"), "replay shard macro_round")
        for item in normalized
    )
    strata = (
        [item for item in normalized if int(item[3]["macro_round"]) < frontier],
        [item for item in normalized if int(item[3]["macro_round"]) == frontier],
    ) if frontier > 0 else (normalized,)
    # Balance newest and cumulative-prior shards independently.  This makes
    # every DDP rank sample the same exact 50/50 source mix instead of letting
    # whole-shard placement silently erase one side of the DAgger batch.
    for stratum in strata:
        stratum.sort(key=lambda item: (-item[0], item[1], item[2]))
        transition_totals = [0 for _ in range(world_size)]
        shard_totals = [0 for _ in range(world_size)]
        for transitions, _digest, _identity, shard in stratum:
            rank = min(
                range(world_size),
                key=lambda candidate: (
                    transition_totals[candidate],
                    shard_totals[candidate],
                    candidate,
                ),
            )
            assignments[rank].append(shard)
            transition_totals[rank] += transitions
            shard_totals[rank] += 1

    for assigned in assignments:
        assigned.sort(
            key=lambda item: (
                item["macro_round"],
                item["cohort_index"],
                item["shard_id"],
            )
        )
    return assignments


def partition_shards(
    resolved_shards: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return one rank's deterministic transition-balanced shard assignment."""

    rank = _true_int(rank, "rank")
    world_size = _true_int(world_size, "world_size", minimum=1)
    if rank >= world_size:
        raise LearnerError("rank must be smaller than world_size")
    return _balanced_shard_assignments(
        resolved_shards, world_size=world_size, seed=seed
    )[rank]


_TRAINING_QUALITY_PROBE_SUMMARY_KEYS = {
    "policy",
    "budget",
    "campaign_id",
    "macro_round",
    "snapshot_file_sha256",
    "snapshot_sha256",
    "partition_seed",
    "world_size",
    "replay_transitions_per_rank",
    "replay_transitions_global",
    "transitions_per_rank",
    "transitions_global",
    "current_round_transitions_per_rank",
    "current_round_transitions_global",
    "historical_transitions_per_rank",
    "historical_transitions_global",
    "rank_selection_sha256s",
    "selection_sha256",
}


def _training_quality_probe_digest_payload(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "domain": _TRAINING_QUALITY_PROBE_DOMAIN.decode("ascii"),
        **{
            key: summary[key]
            for key in sorted(_TRAINING_QUALITY_PROBE_SUMMARY_KEYS - {"selection_sha256"})
        },
    }


def validate_training_quality_probe_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the immutable fixed-probe selection contract."""

    if not isinstance(summary, Mapping) or set(summary) != (
        _TRAINING_QUALITY_PROBE_SUMMARY_KEYS
    ):
        raise LearnerError("training-quality probe summary fields mismatch")
    clean = dict(summary)
    if clean.get("policy") != TRAINING_QUALITY_PROBE_POLICY:
        raise LearnerError("training-quality probe policy mismatch")
    budget = _true_int(clean.get("budget"), "training-quality probe budget", minimum=1)
    if budget != TRAINING_QUALITY_PROBE_BUDGET:
        raise LearnerError("training-quality probe budget mismatch")
    campaign_id = clean.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise LearnerError("training-quality probe campaign ID is invalid")
    macro_round = _true_int(
        clean.get("macro_round"), "training-quality probe macro round"
    )
    _require_sha256(
        clean.get("snapshot_file_sha256"),
        "training-quality probe snapshot file SHA-256",
    )
    _require_sha256(
        clean.get("snapshot_sha256"),
        "training-quality probe snapshot semantic SHA-256",
    )
    _true_int(clean.get("partition_seed"), "training-quality probe partition seed")
    world_size = _true_int(
        clean.get("world_size"), "training-quality probe world size", minimum=1
    )

    list_fields = (
        "replay_transitions_per_rank",
        "transitions_per_rank",
        "current_round_transitions_per_rank",
        "historical_transitions_per_rank",
    )
    normalized_lists: dict[str, list[int]] = {}
    for field in list_fields:
        values = clean.get(field)
        if not isinstance(values, list) or len(values) != world_size:
            raise LearnerError(f"training-quality probe {field} violates world size")
        minimum = 1 if field in {"replay_transitions_per_rank", "transitions_per_rank"} else 0
        normalized_lists[field] = [
            _true_int(value, f"training-quality probe {field}", minimum=minimum)
            for value in values
        ]
    replay_counts = normalized_lists["replay_transitions_per_rank"]
    counts = normalized_lists["transitions_per_rank"]
    current_counts = normalized_lists["current_round_transitions_per_rank"]
    historical_counts = normalized_lists["historical_transitions_per_rank"]
    if any(
        selected > replay
        or selected != current + historical
        for selected, replay, current, historical in zip(
            counts, replay_counts, current_counts, historical_counts
        )
    ):
        raise LearnerError("training-quality probe rank coverage is inconsistent")
    totals = {
        "replay_transitions_global": sum(replay_counts),
        "transitions_global": sum(counts),
        "current_round_transitions_global": sum(current_counts),
        "historical_transitions_global": sum(historical_counts),
    }
    if any(clean.get(key) != value for key, value in totals.items()):
        raise LearnerError("training-quality probe global counts are inconsistent")
    if clean["transitions_global"] > budget:
        raise LearnerError("training-quality probe exceeds its declared budget")
    if clean["transitions_global"] != min(
        budget, clean["replay_transitions_global"]
    ):
        raise LearnerError("training-quality probe does not cover its bounded budget")
    if (
        clean["current_round_transitions_global"]
        + clean["historical_transitions_global"]
        != clean["transitions_global"]
    ):
        raise LearnerError("training-quality probe source counts do not cover selection")
    if macro_round == 0 and clean["historical_transitions_global"] != 0:
        raise LearnerError("round-zero training-quality probe cannot be historical")
    if (
        macro_round > 0
        and clean["transitions_global"] == TRAINING_QUALITY_PROBE_BUDGET
        and clean["current_round_transitions_global"] != budget // 2
    ):
        raise LearnerError(
            "production training-quality probe must split current/historical evenly"
        )
    rank_digests = clean.get("rank_selection_sha256s")
    if (
        not isinstance(rank_digests, list)
        or len(rank_digests) != world_size
        or len(set(rank_digests)) != world_size
    ):
        raise LearnerError("training-quality probe rank digest coverage mismatch")
    for digest in rank_digests:
        _require_sha256(digest, "training-quality probe rank selection SHA-256")
    declared = _require_sha256(
        clean.get("selection_sha256"),
        "training-quality probe selection SHA-256",
    )
    expected = canonical_json_sha256(_training_quality_probe_digest_payload(clean))
    if declared != expected:
        raise LearnerError("training-quality probe selection digest mismatch")
    return clean


@dataclass(frozen=True)
class TrainingQualityProbe:
    """One fixed, lineage-bound held-in probe reused at every frontier."""

    summary_document: Mapping[str, Any]
    indices_by_rank: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        clean = validate_training_quality_probe_summary(self.summary_document)
        world_size = clean["world_size"]
        if len(self.indices_by_rank) != world_size:
            raise LearnerError("training-quality probe index rank coverage mismatch")
        for rank, indices in enumerate(self.indices_by_rank):
            if len(indices) != clean["transitions_per_rank"][rank]:
                raise LearnerError("training-quality probe rank index count mismatch")
            if len(indices) != len(set(indices)) or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            ):
                raise LearnerError("training-quality probe contains invalid/duplicate index")
        object.__setattr__(self, "summary_document", clean)

    def summary(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.summary_document, sort_keys=True))


def _probe_tie_digest(*parts: Any) -> str:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(_TRAINING_QUALITY_PROBE_DOMAIN + b":" + payload).hexdigest()


def build_training_quality_probe(
    snapshot: Mapping[str, Any],
    *,
    world_size: int,
    partition_seed: int,
    budget: int = TRAINING_QUALITY_PROBE_BUDGET,
) -> TrainingQualityProbe:
    """Select a deterministic bounded probe without enumerating replay rows.

    The selection is stratified by current versus historical replay, rank,
    macro-round, and cohort.  Local indices match :class:`ReplayPartition`
    exactly because both use the same balanced shard assignment and canonical
    per-rank shard order.
    """

    if not isinstance(snapshot, Mapping):
        raise LearnerError("training-quality probe snapshot must be a mapping")
    world_size = _true_int(world_size, "training-quality probe world size", minimum=1)
    partition_seed = _true_int(
        partition_seed, "training-quality probe partition seed"
    )
    budget = _true_int(budget, "training-quality probe budget", minimum=1)
    if budget != TRAINING_QUALITY_PROBE_BUDGET:
        raise LearnerError(
            f"training-quality probe budget must be {TRAINING_QUALITY_PROBE_BUDGET}"
        )
    campaign_id = snapshot.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise LearnerError("training-quality probe campaign ID is invalid")
    macro_round = _true_int(
        snapshot.get("through_macro_round"),
        "training-quality probe snapshot macro round",
    )
    snapshot_file_sha256 = _require_sha256(
        snapshot.get("file_sha256"), "training-quality probe snapshot file SHA-256"
    )
    snapshot_sha256 = _require_sha256(
        snapshot.get("snapshot_sha256"),
        "training-quality probe snapshot semantic SHA-256",
    )
    shards = snapshot.get("resolved_shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)) or not shards:
        raise LearnerError("training-quality probe snapshot has no resolved shards")
    assignments = _balanced_shard_assignments(
        shards, world_size=world_size, seed=partition_seed
    )
    if any(not assigned for assigned in assignments):
        raise LearnerError("training-quality probe partition leaves an empty rank")

    # A cell contains one rank/source/round/cohort stratum and one or more
    # non-contiguous shard segments expressed in rank-local replay coordinates.
    cells: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    replay_counts: list[int] = []
    current_capacity = 0
    historical_capacity = 0
    for rank, assigned in enumerate(assignments):
        local_base = 0
        for shard in assigned:
            shard_round = _true_int(
                shard.get("macro_round"), "training-quality probe shard macro round"
            )
            cohort_index = _true_int(
                shard.get("cohort_index"), "training-quality probe cohort index"
            )
            if shard_round > macro_round:
                raise LearnerError("training-quality probe snapshot contains a future shard")
            count = _true_int(
                shard.get("transition_count"),
                "training-quality probe shard transition count",
                minimum=1,
            )
            source = "current" if shard_round == macro_round else "historical"
            key = (rank, source, shard_round, cohort_index)
            cell = cells.setdefault(key, {"capacity": 0, "segments": []})
            cell["segments"].append(
                {
                    "local_start": local_base,
                    "count": count,
                    "logical_id": shard["logical_id"],
                    "shard_id": shard["shard_id"],
                    "macro_round": shard_round,
                    "cohort_index": cohort_index,
                }
            )
            cell["capacity"] += count
            if source == "current":
                current_capacity += count
            else:
                historical_capacity += count
            local_base += count
        replay_counts.append(local_base)
    replay_total = sum(replay_counts)
    if replay_total != _true_int(
        snapshot.get("transition_count"),
        "training-quality probe replay transition count",
        minimum=1,
    ):
        raise LearnerError("training-quality probe replay coverage mismatch")
    selected_total = min(budget, replay_total)
    if selected_total < world_size:
        raise LearnerError("training-quality probe budget cannot cover every rank")
    if macro_round == 0:
        if historical_capacity:
            raise LearnerError("round-zero replay unexpectedly contains historical rows")
        current_quota, historical_quota = selected_total, 0
    elif selected_total == budget:
        half = budget // 2
        if current_capacity < half or historical_capacity < half:
            raise LearnerError(
                "production probe requires at least 128 current and 128 historical rows"
            )
        current_quota, historical_quota = half, budget - half
    else:
        current_quota = min((selected_total + 1) // 2, current_capacity)
        historical_quota = selected_total - current_quota
        if historical_quota > historical_capacity:
            shift = historical_quota - historical_capacity
            historical_quota -= shift
            current_quota += shift
        if current_quota > current_capacity:
            shift = current_quota - current_capacity
            current_quota -= shift
            historical_quota += shift
    source_quotas = {"current": current_quota, "historical": historical_quota}

    allocation = {key: 0 for key in cells}
    sources_by_rank: dict[int, list[str]] = {}
    for rank in range(world_size):
        sources_by_rank[rank] = [
            source
            for source in ("current", "historical")
            if any(key[0] == rank and key[1] == source for key in cells)
        ]
        if not sources_by_rank[rank]:
            raise LearnerError("training-quality probe rank has no source rows")
    current_only = [rank for rank, sources in sources_by_rank.items() if sources == ["current"]]
    historical_only = [
        rank for rank, sources in sources_by_rank.items() if sources == ["historical"]
    ]
    dual = [rank for rank, sources in sources_by_rank.items() if len(sources) == 2]
    lower_current_dual = max(
        0, len(dual) - (historical_quota - len(historical_only))
    )
    upper_current_dual = min(len(dual), current_quota - len(current_only))
    if lower_current_dual > upper_current_dual:
        raise LearnerError("training-quality source quotas cannot cover every rank")
    desired_current_dual = round(len(dual) * current_quota / selected_total)
    current_dual_count = min(
        upper_current_dual, max(lower_current_dual, desired_current_dual)
    )
    ordered_dual = sorted(
        dual,
        key=lambda rank: _probe_tie_digest(
            snapshot_file_sha256,
            snapshot_sha256,
            campaign_id,
            macro_round,
            partition_seed,
            world_size,
            "rank-source",
            rank,
        ),
    )
    reserved_source = {rank: "current" for rank in current_only}
    reserved_source.update({rank: "historical" for rank in historical_only})
    reserved_source.update(
        {
            rank: ("current" if offset < current_dual_count else "historical")
            for offset, rank in enumerate(ordered_dual)
        }
    )
    for rank, source in reserved_source.items():
        candidates = [key for key in cells if key[0] == rank and key[1] == source]
        chosen = min(
            candidates,
            key=lambda key: _probe_tie_digest(
                snapshot_file_sha256,
                snapshot_sha256,
                campaign_id,
                macro_round,
                partition_seed,
                world_size,
                "rank-cell",
                *key,
            ),
        )
        allocation[chosen] += 1

    for source, quota in source_quotas.items():
        remaining = quota - sum(
            count for key, count in allocation.items() if key[1] == source
        )
        if remaining < 0:
            raise LearnerError("training-quality probe source reservation exceeds quota")
        for _ in range(remaining):
            candidates = [
                key
                for key, cell in cells.items()
                if key[1] == source and allocation[key] < cell["capacity"]
            ]
            if not candidates:
                raise LearnerError("training-quality probe source capacity exhausted")
            chosen = min(
                candidates,
                key=lambda key: (
                    Fraction(allocation[key] + 1, cells[key]["capacity"]),
                    _probe_tie_digest(
                        snapshot_file_sha256,
                        snapshot_sha256,
                        campaign_id,
                        macro_round,
                        partition_seed,
                        world_size,
                        "cell-allocation",
                        *key,
                    ),
                ),
            )
            allocation[chosen] += 1

    descriptors_by_rank: list[list[dict[str, Any]]] = [
        [] for _ in range(world_size)
    ]
    for key in sorted(cells):
        selected = allocation[key]
        if selected == 0:
            continue
        rank, source, shard_round, cohort_index = key
        cell = cells[key]
        capacity = int(cell["capacity"])
        segments = list(cell["segments"])
        segment_ends: list[int] = []
        segment_total = 0
        for segment in segments:
            segment_total += int(segment["count"])
            segment_ends.append(segment_total)
        if segment_total != capacity or selected > capacity:
            raise LearnerError("training-quality probe cell capacity mismatch")
        for ordinal in range(selected):
            lower = ordinal * capacity // selected
            upper = (ordinal + 1) * capacity // selected
            width = upper - lower
            if width < 1:
                raise LearnerError("training-quality probe stratum over-selected")
            phase = int(
                _probe_tie_digest(
                    snapshot_file_sha256,
                    snapshot_sha256,
                    campaign_id,
                    macro_round,
                    partition_seed,
                    world_size,
                    "sample",
                    *key,
                    ordinal,
                ),
                16,
            ) % width
            cell_offset = lower + phase
            segment_index = bisect_right(segment_ends, cell_offset)
            segment_base = 0 if segment_index == 0 else segment_ends[segment_index - 1]
            segment = segments[segment_index]
            transition_index = cell_offset - segment_base
            descriptors_by_rank[rank].append(
                {
                    "rank": rank,
                    "source": source,
                    "macro_round": shard_round,
                    "cohort_index": cohort_index,
                    "logical_id": segment["logical_id"],
                    "shard_id": segment["shard_id"],
                    "transition_index": transition_index,
                    "local_index": int(segment["local_start"]) + transition_index,
                }
            )

    indices_by_rank: list[tuple[int, ...]] = []
    counts: list[int] = []
    current_counts: list[int] = []
    historical_counts: list[int] = []
    rank_digests: list[str] = []
    for rank, descriptors in enumerate(descriptors_by_rank):
        descriptors.sort(key=lambda item: item["local_index"])
        local_indices = tuple(int(item["local_index"]) for item in descriptors)
        if not local_indices or len(local_indices) != len(set(local_indices)):
            raise LearnerError("training-quality probe rank selection overlaps/is empty")
        if any(index >= replay_counts[rank] for index in local_indices):
            raise LearnerError("training-quality probe local index exceeds replay partition")
        indices_by_rank.append(local_indices)
        counts.append(len(descriptors))
        current_counts.append(sum(item["source"] == "current" for item in descriptors))
        historical_counts.append(
            sum(item["source"] == "historical" for item in descriptors)
        )
        rank_digests.append(
            canonical_json_sha256(
                {
                    "domain": _TRAINING_QUALITY_PROBE_DOMAIN.decode("ascii"),
                    "policy": TRAINING_QUALITY_PROBE_POLICY,
                    "campaign_id": campaign_id,
                    "macro_round": macro_round,
                    "snapshot_file_sha256": snapshot_file_sha256,
                    "snapshot_sha256": snapshot_sha256,
                    "partition_seed": partition_seed,
                    "world_size": world_size,
                    "rank": rank,
                    "replay_transitions_rank": replay_counts[rank],
                    "samples": descriptors,
                }
            )
        )
    base_summary = {
        "policy": TRAINING_QUALITY_PROBE_POLICY,
        "budget": budget,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "snapshot_file_sha256": snapshot_file_sha256,
        "snapshot_sha256": snapshot_sha256,
        "partition_seed": partition_seed,
        "world_size": world_size,
        "replay_transitions_per_rank": replay_counts,
        "replay_transitions_global": replay_total,
        "transitions_per_rank": counts,
        "transitions_global": sum(counts),
        "current_round_transitions_per_rank": current_counts,
        "current_round_transitions_global": sum(current_counts),
        "historical_transitions_per_rank": historical_counts,
        "historical_transitions_global": sum(historical_counts),
        "rank_selection_sha256s": rank_digests,
    }
    summary = {
        **base_summary,
        "selection_sha256": canonical_json_sha256(
            _training_quality_probe_digest_payload(base_summary)
        ),
    }
    return TrainingQualityProbe(
        summary_document=summary,
        indices_by_rank=tuple(indices_by_rank),
    )


def _read_npz_from_verified_descriptor(
    path: Path, expected_sha256: str
) -> dict[str, np.ndarray]:
    """Hash and decode one NPZ through the same held, non-symlink descriptor."""

    expected_sha256 = _require_sha256(expected_sha256, "payload_sha256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LearnerError(f"cannot safely open replay payload {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise LearnerError(f"replay payload is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise LearnerError(
                    f"replay payload digest changed before learner load: {path}"
                )
            handle.seek(0)
            try:
                with zipfile.ZipFile(handle, "r") as archive:
                    members = [entry.filename for entry in archive.infolist()]
                    if len(members) != len(set(members)) or any(
                        not member.endswith(".npy")
                        or "/" in member
                        or _SAFE_KEY_RE.fullmatch(member[:-4]) is None
                        for member in members
                    ):
                        raise LearnerError("replay NPZ has unsafe or duplicate members")
                handle.seek(0)
                with np.load(handle, allow_pickle=False) as bundle:
                    arrays = {
                        key: np.ascontiguousarray(bundle[key]) for key in bundle.files
                    }
            except LearnerError:
                raise
            except Exception as exc:
                raise LearnerError(
                    f"cannot decode replay payload {path}: {exc}"
                ) from exc
    finally:
        os.close(descriptor)
    return arrays


@dataclass
class _LoadedShard:
    identity: str
    macro_round: int
    arrays: dict[str, np.ndarray]
    steps: int
    envs: int

    @property
    def transitions(self) -> int:
        return self.steps * self.envs


class ReplayPartition:
    """An eagerly verified rank-local view over cumulative replay shards."""

    def __init__(
        self,
        *,
        snapshot: Mapping[str, Any],
        shards: Sequence[_LoadedShard],
        config: LearnerBundleConfig,
        rank: int,
        world_size: int,
        transition_counts_by_rank: Sequence[int],
        training_frontier: int | None,
    ):
        if not shards:
            raise LearnerError(
                f"rank {rank} received no replay shards; use fewer ranks or more shard parts"
            )
        self.snapshot = dict(snapshot)
        self.shards = list(shards)
        self.config = config
        self.rank = rank
        self.world_size = world_size
        if len(transition_counts_by_rank) != world_size:
            raise LearnerError("per-rank replay transition counts violate world size")
        self.transition_counts_by_rank = tuple(
            _true_int(value, "per-rank replay transitions", minimum=1)
            for value in transition_counts_by_rank
        )
        self.global_transition_count = sum(self.transition_counts_by_rank)
        if self.global_transition_count != _true_int(
            self.snapshot.get("transition_count"),
            "snapshot transition_count",
            minimum=1,
        ):
            raise LearnerError(
                "per-rank replay transition counts disagree with snapshot total"
            )
        self._ends: list[int] = []
        indices_by_round: dict[int, list[int]] = {}
        total = 0
        obs_keys: set[str] | None = None
        for shard in self.shards:
            start = total
            total += shard.transitions
            self._ends.append(total)
            indices_by_round.setdefault(shard.macro_round, []).extend(
                range(start, total)
            )
            current = {key for key in shard.arrays if key.startswith("obs__")}
            if "obs__actor_obs" not in current:
                raise LearnerError("every replay shard must contain obs__actor_obs")
            if obs_keys is None:
                obs_keys = current
            elif current != obs_keys:
                raise LearnerError(
                    "policy observation keys differ across replay shards"
                )
        self.transition_count = total
        if self.transition_counts_by_rank[rank] != self.transition_count:
            raise LearnerError(
                "loaded replay transition count disagrees with rank assignment"
            )
        self.observation_keys = tuple(sorted(obs_keys or ()))
        if training_frontier is None:
            newest = tuple(range(self.transition_count))
            prior = ()
        else:
            if set(indices_by_round) - set(range(training_frontier + 1)):
                raise LearnerError("replay partition contains a future macro-round")
            newest = tuple(indices_by_round.get(training_frontier, ()))
            prior = tuple(
                index
                for macro_round in range(training_frontier)
                for index in indices_by_round.get(macro_round, ())
            )
            if not newest:
                raise LearnerError(
                    "replay partition has no newest sealed-round examples"
                )
            if training_frontier == 0:
                if prior or len(newest) != self.transition_count:
                    raise LearnerError(
                        "BC replay must contain only round-zero train examples"
                    )
            elif not prior:
                raise LearnerError(
                    "DAgger replay partition has no cumulative-prior train examples"
                )
        self.newest_round_indices = newest
        self.cumulative_prior_indices = prior

    def _location(self, index: int) -> tuple[_LoadedShard, int, int]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.transition_count
        ):
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        base = 0 if shard_index == 0 else self._ends[shard_index - 1]
        offset = index - base
        shard = self.shards[shard_index]
        return shard, offset // shard.envs, offset % shard.envs

    def batch(
        self, indices: Sequence[int], *, device: torch.device | str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not indices:
            raise LearnerError("cannot construct an empty learner batch")
        rgb_rows: list[np.ndarray] = []
        expert_rows: list[np.ndarray] = []
        observation_rows: dict[str, list[np.ndarray]] = {
            key: [] for key in self.observation_keys
        }
        for index in indices:
            shard, time_index, env_index = self._location(int(index))
            rgb_rows.append(shard.arrays["policy_rgb"][time_index, env_index])
            expert_rows.append(shard.arrays["expert_actions"][time_index, env_index])
            for key in self.observation_keys:
                observation_rows[key].append(shard.arrays[key][time_index, env_index])

        device = torch.device(device)
        rgb = torch.from_numpy(np.stack(rgb_rows)).to(
            device=device, dtype=torch.float32
        )
        rgb = rgb.div(255.0)
        mean = torch.tensor(self.config.rgb_mean, device=device, dtype=rgb.dtype)
        std = torch.tensor(self.config.rgb_std, device=device, dtype=rgb.dtype)
        rgb = (rgb - mean.view(1, 1, 1, 3)) / std.view(1, 1, 1, 3)
        if self.config.rgb_layout == "NCHW":
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
        observations: dict[str, torch.Tensor] = {self.config.rgb_input_key: rgb}
        explicit_map = dict(self.config.observation_key_map)
        for source, rows in observation_rows.items():
            destination = explicit_map.get(source, source[len("obs__") :])
            if destination == self.config.rgb_input_key or destination in observations:
                raise LearnerError(
                    f"policy observation destination collision: {destination}"
                )
            array = np.stack(rows)
            observations[destination] = torch.from_numpy(array).to(device=device)
        targets = torch.from_numpy(np.stack(expert_rows)).to(
            device=device, dtype=torch.float32
        )
        return observations, targets


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    view_shape: tuple[int, ...],
    action_dim: int,
    expected_transitions: int,
) -> tuple[int, int]:
    required = {"policy_rgb", "expert_actions", "obs__actor_obs"}
    if not required <= set(arrays):
        raise LearnerError(
            f"replay payload lacks learner arrays: {sorted(required - set(arrays))}"
        )
    rgb = arrays["policy_rgb"]
    expert = arrays["expert_actions"]
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or tuple(rgb.shape[2:]) != view_shape:
        raise LearnerError("policy_rgb violates the replay view contract")
    steps, envs = int(rgb.shape[0]), int(rgb.shape[1])
    if expert.shape != (steps, envs, action_dim) or not np.issubdtype(
        expert.dtype, np.floating
    ):
        raise LearnerError("expert_actions violates the replay action contract")
    if not np.isfinite(expert).all():
        raise LearnerError("expert_actions contains non-finite values")
    if steps * envs != expected_transitions:
        raise LearnerError("payload transition count disagrees with replay snapshot")
    for key, array in arrays.items():
        if key.startswith("obs__"):
            if array.ndim < 2 or tuple(array.shape[:2]) != (steps, envs):
                raise LearnerError(f"{key} must begin with [T,B]")
            if array.dtype.hasobject or (
                np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all()
            ):
                raise LearnerError(f"{key} contains unsafe values")
    return steps, envs


def load_replay_partition(
    snapshot_path: str | os.PathLike[str],
    config: LearnerBundleConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    expected_snapshot_file_sha256: str | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
) -> ReplayPartition:
    """Load only the rank's shards after replay and held-FD digest validation."""

    snapshot_path = _require_campaign_storage_path(
        snapshot_path,
        "train replay snapshot",
        generation_id=config.generation_id,
    )
    rank = _true_int(rank, "rank")
    world_size = _true_int(world_size, "world_size", minimum=1)
    if rank >= world_size:
        raise LearnerError("rank must be smaller than world_size")
    try:
        snapshot = load_snapshot(
            snapshot_path,
            expected_file_sha256=expected_snapshot_file_sha256,
            artifact_root=artifact_root,
        )
    except Exception as exc:
        raise LearnerError(f"replay snapshot validation failed: {exc}") from exc
    if snapshot["campaign_id"] != config.campaign_id:
        raise LearnerError("learner campaign does not match replay snapshot")
    if snapshot["through_macro_round"] != config.macro_round:
        raise LearnerError(
            "learner macro_round does not match replay snapshot frontier"
        )
    return _partition_loaded_snapshot(
        snapshot,
        config,
        rank=rank,
        world_size=world_size,
        training_frontier=config.macro_round,
    )


def _partition_loaded_snapshot(
    snapshot: Mapping[str, Any],
    config: LearnerBundleConfig,
    *,
    rank: int,
    world_size: int,
    training_frontier: int | None,
) -> ReplayPartition:
    view_shape = tuple(snapshot["view_contract"].get("shape", ()))
    action_shape = snapshot["action_contract"].get("shape")
    if len(view_shape) != 3 or view_shape[-1] != 3:
        raise LearnerError("snapshot view contract must be one HWC RGB stack")
    if not isinstance(action_shape, list) or len(action_shape) != 1:
        raise LearnerError("snapshot action contract must be one-dimensional")
    action_dim = _true_int(action_shape[0], "action dimension", minimum=1)
    if (
        config.train_action_slice is not None
        and config.train_action_slice[1] > action_dim
    ):
        raise LearnerError("train_action_slice exceeds replay action dimension")

    # Every rank computes the same coverage gate before any rank enters DDP.
    # This converts a potentially asymmetric "rank N has no data" hang into a
    # deterministic launch-time failure on all ranks.
    assignments = _balanced_shard_assignments(
        snapshot["resolved_shards"],
        world_size=world_size,
        seed=config.partition_seed,
    )
    empty_ranks = [
        index
        for index, assigned_shards in enumerate(assignments)
        if not assigned_shards
    ]
    if empty_ranks:
        raise LearnerError(
            "replay shard partition leaves ranks without data: "
            f"{empty_ranks}; use fewer ranks or publish more shard parts"
        )
    if training_frontier is not None:
        newest_counts = tuple(
            sum(
                int(shard["transition_count"])
                for shard in assigned_shards
                if int(shard["macro_round"]) == training_frontier
            )
            for assigned_shards in assignments
        )
        prior_counts = tuple(
            sum(
                int(shard["transition_count"])
                for shard in assigned_shards
                if int(shard["macro_round"]) < training_frontier
            )
            for assigned_shards in assignments
        )
        required = (newest_counts,) if training_frontier == 0 else (
            newest_counts,
            prior_counts,
        )
        if any(0 in counts or len(set(counts)) != 1 for counts in required):
            raise LearnerError(
                "exact uniform replay sampling requires equal non-empty source "
                "populations on every learner rank"
            )
    assigned = assignments[rank]
    transition_counts_by_rank = tuple(
        sum(int(shard["transition_count"]) for shard in assigned_shards)
        for assigned_shards in assignments
    )
    loaded: list[_LoadedShard] = []
    for shard in assigned:
        arrays = _read_npz_from_verified_descriptor(
            Path(shard["payload_path"]), shard["payload_sha256"]
        )
        steps, envs = _validate_loaded_arrays(
            arrays,
            view_shape=view_shape,
            action_dim=action_dim,
            expected_transitions=int(shard["transition_count"]),
        )
        loaded.append(
            _LoadedShard(
                identity=f"{shard['logical_id']}/{shard['shard_id']}",
                macro_round=_true_int(
                    shard.get("macro_round"), "replay shard macro_round"
                ),
                arrays=arrays,
                steps=steps,
                envs=envs,
            )
        )
    return ReplayPartition(
        snapshot=snapshot,
        shards=loaded,
        config=config,
        rank=rank,
        world_size=world_size,
        transition_counts_by_rank=transition_counts_by_rank,
        training_frontier=training_frontier,
    )


def _assert_training_and_heldout_are_disjoint(
    training: Mapping[str, Any], heldout: Mapping[str, Any]
) -> None:
    """Reject any artifact identity reused across train and evaluation replay."""

    if training["view_contract"] != heldout["view_contract"]:
        raise LearnerError("training and held-out view contracts differ")
    if training["action_contract"] != heldout["action_contract"]:
        raise LearnerError("training and held-out action contracts differ")
    train_cohorts = {item["manifest_sha256"] for item in training["cohorts"]}
    heldout_cohorts = {item["manifest_sha256"] for item in heldout["cohorts"]}
    train_shards = {
        item["manifest_sha256"] for item in training["resolved_shards"]
    }
    heldout_shards = {
        item["manifest_sha256"] for item in heldout["resolved_shards"]
    }
    train_payloads = {item["payload_sha256"] for item in training["resolved_shards"]}
    heldout_payloads = {
        item["payload_sha256"] for item in heldout["resolved_shards"]
    }
    overlaps = {
        "cohort manifest": train_cohorts & heldout_cohorts,
        "shard manifest": train_shards & heldout_shards,
        "payload": train_payloads & heldout_payloads,
    }
    reused = [label for label, digests in overlaps.items() if digests]
    if reused:
        raise LearnerError(
            "training and held-out replay reuse artifact identities: "
            + ", ".join(reused)
        )


class DeterministicBatchSchedule:
    """Step-addressable shuffled batches, identical before and after restart."""

    def __init__(self, size: int, batch_size: int, *, seed: int):
        self.size = _true_int(size, "replay partition size", minimum=1)
        self.batch_size = _true_int(batch_size, "batch_size", minimum=1)
        self.seed = _true_int(seed, "seed")
        self._epoch = -1
        self._permutation: list[int] = []

    def _permutation_for(self, epoch: int) -> list[int]:
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(self.size, generator=generator).tolist()
            self._epoch = epoch
        return self._permutation

    def indices(self, optimizer_step: int) -> list[int]:
        optimizer_step = _true_int(optimizer_step, "optimizer_step")
        start = optimizer_step * self.batch_size
        result: list[int] = []
        for position in range(start, start + self.batch_size):
            epoch, offset = divmod(position, self.size)
            result.append(self._permutation_for(epoch)[offset])
        return result


class CumulativeDaggerBatchSchedule:
    """Step-addressable BC/DAgger sampling over immutable example strata.

    Round zero is pure behavior cloning.  Every later optimizer batch contains
    exactly half examples from the newest sealed round and half from the union
    of every earlier train round.  Each half walks an independent seeded random
    permutation, so every example in that source population is sampled
    uniformly before the schedule begins its next shuffled epoch.
    """

    def __init__(
        self,
        partition: ReplayPartition,
        batch_size: int,
        *,
        seed: int,
    ):
        self.partition = partition
        self.batch_size = _true_int(batch_size, "batch_size", minimum=1)
        self.seed = _true_int(seed, "seed")
        self.is_bc = partition.config.macro_round == 0
        if self.is_bc:
            self._bc = DeterministicBatchSchedule(
                len(partition.newest_round_indices), self.batch_size, seed=self.seed
            )
            self._newest = None
            self._prior = None
            return
        if self.batch_size % 2:
            raise LearnerError(
                "DAgger batch_size must be even for the exact 50/50 source contract"
            )
        half = self.batch_size // 2
        self._bc = None
        self._newest = DeterministicBatchSchedule(
            len(partition.newest_round_indices),
            half,
            seed=self.seed + 0x4E4557,
        )
        self._prior = DeterministicBatchSchedule(
            len(partition.cumulative_prior_indices),
            half,
            seed=self.seed + 0x5052494F52,
        )

    def indices(self, optimizer_step: int) -> list[int]:
        optimizer_step = _true_int(optimizer_step, "optimizer_step")
        if self.is_bc:
            assert self._bc is not None
            return [
                self.partition.newest_round_indices[index]
                for index in self._bc.indices(optimizer_step)
            ]
        assert self._newest is not None and self._prior is not None
        newest = [
            self.partition.newest_round_indices[index]
            for index in self._newest.indices(optimizer_step)
        ]
        prior = [
            self.partition.cumulative_prior_indices[index]
            for index in self._prior.indices(optimizer_step)
        ]
        # Interleaving prevents source order from becoming an accidental model
        # input while retaining an exact count that tests and telemetry can audit.
        return [index for pair in zip(newest, prior) for index in pair]

    def source_counts(self, optimizer_steps: int) -> dict[str, int]:
        optimizer_steps = _true_int(optimizer_steps, "optimizer_steps")
        examples = optimizer_steps * self.batch_size
        if self.is_bc:
            return {
                "newest_source_examples": examples,
                "prior_source_examples": 0,
            }
        return {
            "newest_source_examples": examples // 2,
            "prior_source_examples": examples // 2,
        }


def optimizer_source_exposure(
    config: LearnerBundleConfig, *, completed_steps: int, world_size: int
) -> dict[str, int | str]:
    """Return the auditable global optimizer-example and source exposure."""

    completed_steps = _true_int(completed_steps, "completed_steps")
    world_size = _true_int(world_size, "world_size", minimum=1)
    optimizer_examples = completed_steps * config.batch_size * world_size
    if config.macro_round == 0:
        return {
            "sampling_policy": "bc_train_uniform_v1",
            "optimizer_examples": optimizer_examples,
            "newest_source_examples": optimizer_examples,
            "prior_source_examples": 0,
        }
    if config.batch_size % 2:
        raise LearnerError(
            "DAgger batch_size must be even for the exact 50/50 source contract"
        )
    return {
        "sampling_policy": "newest_half_cumulative_prior_half_uniform_v1",
        "optimizer_examples": optimizer_examples,
        "newest_source_examples": optimizer_examples // 2,
        "prior_source_examples": optimizer_examples // 2,
    }


def optimizer_exposure_summary(
    snapshot: Mapping[str, Any],
    config: LearnerBundleConfig,
    *,
    completed_steps: int,
    world_size: int,
) -> dict[str, int | float | str]:
    """Bind optimizer examples to source populations and expected passes."""

    exposure = optimizer_source_exposure(
        config, completed_steps=completed_steps, world_size=world_size
    )
    counts = snapshot.get("counts_by_round")
    if not isinstance(counts, Mapping) or set(counts) != {
        str(index) for index in range(config.macro_round + 1)
    }:
        raise LearnerError("training snapshot round populations are incomplete")
    normalized = {
        int(key): _true_int(value, f"round {key} transition count", minimum=1)
        for key, value in counts.items()
    }
    newest_population = normalized[config.macro_round]
    prior_population = sum(
        normalized[index] for index in range(config.macro_round)
    )
    if config.macro_round == 0:
        unit_examples = newest_population
        prior_passes = 0.0
    else:
        if prior_population <= 0:
            raise LearnerError("DAgger snapshot has no cumulative-prior population")
        unit_examples = 2 * max(newest_population, prior_population)
        prior_passes = int(exposure["prior_source_examples"]) / prior_population
    optimizer_examples = int(exposure["optimizer_examples"])
    return {
        **exposure,
        "newest_source_population": newest_population,
        "prior_source_population": prior_population,
        "balanced_exposure_unit_examples": unit_examples,
        "balanced_exposure_units": optimizer_examples / unit_examples,
        "newest_expected_passes": int(exposure["newest_source_examples"])
        / newest_population,
        "prior_expected_passes": prior_passes,
    }


def _aligned_learned_actions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    train_action_slice: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if predictions.ndim < 1 or targets.ndim < 1:
        raise LearnerError("policy and expert actions must have an action dimension")
    if predictions.shape[:-1] != targets.shape[:-1]:
        raise LearnerError(
            "policy/expert action batch shapes differ: "
            f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )
    if train_action_slice is None:
        if predictions.shape != targets.shape:
            raise LearnerError(
                "policy/expert action shapes differ without train_action_slice: "
                f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
            )
        action_indices = tuple(range(int(targets.shape[-1])))
    else:
        start, stop = train_action_slice
        if targets.shape[-1] < stop:
            raise LearnerError("expert actions do not cover train_action_slice")
        targets = targets[..., start:stop]
        if predictions.shape[-1] >= stop:
            predictions = predictions[..., start:stop]
        elif start == 0 and predictions.shape[-1] == stop:
            # Production contract: the trainable VisionActor emits an 11-D
            # prefix while replay stores the 26-D policy + Homie action.
            predictions = predictions[..., :stop]
        else:
            raise LearnerError(
                "policy action width is incompatible with train_action_slice; "
                "a reduced-width prediction is allowed only for a prefix slice"
            )
        if predictions.shape != targets.shape:
            raise LearnerError(
                "sliced policy/expert action shapes differ: "
                f"{tuple(predictions.shape)} vs {tuple(targets.shape)}"
            )
        action_indices = tuple(range(start, stop))
    return predictions.float(), targets.float(), action_indices


def supervised_action_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    train_action_slice: tuple[int, int] | None,
) -> torch.Tensor:
    predictions, targets, _action_indices = _aligned_learned_actions(
        predictions, targets, train_action_slice
    )
    return torch.mean(torch.square(predictions.float() - targets.float()))


def _linear_quantile(values: np.ndarray, probability: float) -> float:
    """Return a deterministic NumPy-compatible linear quantile."""

    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        raise LearnerError("held-out quantile population is empty")
    if not np.isfinite(flattened).all():
        raise LearnerError("held-out quantile population is non-finite")
    ordered = np.sort(flattened, kind="mergesort")
    position = (ordered.size - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def evaluate_heldout_actions(
    model: nn.Module,
    *,
    partition: ReplayPartition,
    adapter: PolicyAdapter,
    device: torch.device | str,
    optimizer_step: int,
    indices: Sequence[int] | None = None,
    transition_counts_by_rank: Sequence[int] | None = None,
    include_per_action_p90: bool = False,
) -> dict[str, Any]:
    """Evaluate an exact population or an explicitly bounded fixed subset.

    Held-out validation retains its exhaustive default.  Training-quality
    callers supply their immutable rank-local probe indices and exact global
    rank counts, ensuring every frontier performs bounded work.
    """

    optimizer_step = _true_int(optimizer_step, "optimizer_step")
    device = torch.device(device)
    rank, world_size = _distributed_context()
    if rank != partition.rank or world_size != partition.world_size:
        raise LearnerError("held-out partition does not match distributed context")
    if indices is None:
        local_indices = tuple(range(partition.transition_count))
        selected_counts = partition.transition_counts_by_rank
        if transition_counts_by_rank is not None:
            raise LearnerError(
                "held-out transition counts require explicit selected indices"
            )
    else:
        local_indices = tuple(indices)
        if not local_indices:
            raise LearnerError("held-out selected evaluation indices are empty")
        if len(local_indices) != len(set(local_indices)) or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < partition.transition_count
            for index in local_indices
        ):
            raise LearnerError(
                "held-out selected evaluation indices overlap or exceed partition"
            )
        if (
            not isinstance(transition_counts_by_rank, Sequence)
            or isinstance(transition_counts_by_rank, (str, bytes))
            or len(transition_counts_by_rank) != world_size
        ):
            raise LearnerError(
                "held-out selected transition counts violate world size"
            )
        selected_counts = tuple(
            _true_int(
                count,
                "held-out selected transitions per rank",
                minimum=1,
            )
            for count in transition_counts_by_rank
        )
        if selected_counts[rank] != len(local_indices):
            raise LearnerError(
                "held-out selected indices disagree with this rank's count"
            )
        if any(
            selected > replay
            for selected, replay in zip(
                selected_counts, partition.transition_counts_by_rank
            )
        ):
            raise LearnerError(
                "held-out selected counts exceed replay rank population"
            )
    preserved_rng = _rng_state()
    was_training = model.training
    model.eval()
    statistics: torch.Tensor | None = None
    action_indices: tuple[int, ...] | None = None
    absolute_error_chunks: list[np.ndarray] = []
    action_l2_chunks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(
                0, len(local_indices), partition.config.batch_size
            ):
                stop = min(start + partition.config.batch_size, len(local_indices))
                observations, targets = partition.batch(
                    local_indices[start:stop], device=device
                )
                predictions = adapter.actions(_unwrap_model(model), observations)
                predictions, targets, current_indices = _aligned_learned_actions(
                    predictions, targets, partition.config.train_action_slice
                )
                if action_indices is None:
                    action_indices = current_indices
                    # Global scalars: squared error, absolute error, element
                    # count, row cosine sum, row count.  Six per-dimension
                    # accumulators follow: squared error, absolute error,
                    # sample count, prediction/target dot product, prediction
                    # norm squared, and target norm squared.
                    statistics = torch.zeros(
                        5 + 6 * len(action_indices),
                        dtype=torch.float64,
                        device=device,
                    )
                elif current_indices != action_indices:
                    raise LearnerError(
                        "held-out learned-action contract changed between batches"
                    )
                assert statistics is not None
                flat_predictions = predictions.reshape(-1, predictions.shape[-1]).to(
                    dtype=torch.float64
                )
                flat_targets = targets.reshape(-1, targets.shape[-1]).to(
                    dtype=torch.float64
                )
                difference = flat_predictions - flat_targets
                squared = torch.square(difference)
                absolute = torch.abs(difference)
                absolute_error_chunks.append(
                    absolute.detach().cpu().numpy().reshape(
                        -1, absolute.shape[-1]
                    )
                )
                action_l2_chunks.append(
                    torch.linalg.vector_norm(difference, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                prediction_norm = torch.linalg.vector_norm(
                    flat_predictions, dim=-1
                )
                target_norm = torch.linalg.vector_norm(flat_targets, dim=-1)
                denominator = prediction_norm * target_norm
                row_cosine = torch.zeros_like(denominator)
                nonzero = denominator > 0.0
                row_cosine[nonzero] = torch.sum(
                    flat_predictions[nonzero] * flat_targets[nonzero], dim=-1
                ) / denominator[nonzero]
                row_cosine[
                    (prediction_norm == 0.0) & (target_norm == 0.0)
                ] = 1.0
                row_cosine.clamp_(-1.0, 1.0)
                statistics[0] += squared.sum()
                statistics[1] += absolute.sum()
                statistics[2] += squared.numel()
                statistics[3] += row_cosine.sum()
                statistics[4] += row_cosine.numel()
                dimension = len(action_indices)
                statistics[5 : 5 + dimension] += squared.sum(dim=0)
                statistics[5 + dimension : 5 + 2 * dimension] += absolute.sum(dim=0)
                statistics[5 + 2 * dimension : 5 + 3 * dimension] += (
                    flat_predictions.shape[0]
                )
                statistics[5 + 3 * dimension : 5 + 4 * dimension] += (
                    flat_predictions * flat_targets
                ).sum(dim=0)
                statistics[5 + 4 * dimension : 5 + 5 * dimension] += torch.square(
                    flat_predictions
                ).sum(dim=0)
                statistics[5 + 5 * dimension : 5 + 6 * dimension] += torch.square(
                    flat_targets
                ).sum(dim=0)
    finally:
        model.train(was_training)
        _restore_rng_state(preserved_rng)
    if statistics is None or action_indices is None:
        raise LearnerError("held-out replay evaluation produced no samples")
    local_absolute_errors = np.concatenate(absolute_error_chunks, axis=0)
    local_action_l2 = np.concatenate(action_l2_chunks)
    if world_size > 1:
        statistics = _ordered_allgather_sum(statistics, world_size)
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            (local_absolute_errors.tolist(), local_action_l2.tolist()),
        )
        absolute_by_rank: list[np.ndarray] = []
        action_l2_by_rank: list[np.ndarray] = []
        for rank_payload in gathered:
            if (
                not isinstance(rank_payload, (tuple, list))
                or len(rank_payload) != 2
            ):
                raise LearnerError("held-out quantile gather contract mismatch")
            absolute_by_rank.append(
                np.asarray(rank_payload[0], dtype=np.float64).reshape(
                    -1, len(action_indices)
                )
            )
            action_l2_by_rank.append(
                np.asarray(rank_payload[1], dtype=np.float64).reshape(-1)
            )
        global_absolute_errors = np.concatenate(absolute_by_rank, axis=0)
        global_action_l2 = np.concatenate(action_l2_by_rank)
    else:
        global_absolute_errors = local_absolute_errors
        global_action_l2 = local_action_l2
    dimension = len(action_indices)
    element_count = float(statistics[2].item())
    row_count = float(statistics[4].item())
    if element_count <= 0.0 or row_count <= 0.0:
        raise LearnerError("held-out metric reduction produced an empty population")
    mse = float((statistics[0] / element_count).item())
    mae = float((statistics[1] / element_count).item())
    cosine = float((statistics[3] / row_count).item())
    dimension_squared = statistics[5 : 5 + dimension]
    dimension_absolute = statistics[5 + dimension : 5 + 2 * dimension]
    dimension_count = statistics[5 + 2 * dimension : 5 + 3 * dimension]
    dimension_dot = statistics[5 + 3 * dimension : 5 + 4 * dimension]
    dimension_prediction_squared = statistics[
        5 + 4 * dimension : 5 + 5 * dimension
    ]
    dimension_target_squared = statistics[
        5 + 5 * dimension : 5 + 6 * dimension
    ]
    per_action: dict[str, dict[str, float]] = {}
    for offset, action_index in enumerate(action_indices):
        count = float(dimension_count[offset].item())
        if count <= 0.0:
            raise LearnerError("held-out per-action metric has zero samples")
        action_mse = float((dimension_squared[offset] / count).item())
        norm_product = float(
            torch.sqrt(
                dimension_prediction_squared[offset]
                * dimension_target_squared[offset]
            ).item()
        )
        if norm_product > 0.0:
            action_cosine = max(
                -1.0,
                min(1.0, float(dimension_dot[offset].item()) / norm_product),
            )
        elif (
            float(dimension_prediction_squared[offset].item()) == 0.0
            and float(dimension_target_squared[offset].item()) == 0.0
        ):
            action_cosine = 1.0
        else:
            action_cosine = 0.0
        per_action[str(action_index)] = {
            "mse": action_mse,
            "mae": float((dimension_absolute[offset] / count).item()),
            "rmse": math.sqrt(action_mse),
            "cosine": action_cosine,
        }
    result = {
        "schema": HELDOUT_TELEMETRY_SCHEMA,
        "campaign_id": partition.config.campaign_id,
        "macro_round": partition.config.macro_round,
        "optimizer_step": optimizer_step,
        "heldout_snapshot_sha256": partition.snapshot["snapshot_sha256"],
        "heldout_snapshot_file_sha256": partition.snapshot["file_sha256"],
        "heldout_transitions_rank": len(local_indices),
        "heldout_transitions_per_rank": list(selected_counts),
        "heldout_transitions_global": sum(selected_counts),
        "world_size": world_size,
        "learned_action_indices": list(action_indices),
        "action_mse": mse,
        "action_mae": mae,
        "action_rmse": math.sqrt(mse),
        "action_cosine": cosine,
        "action_absolute_error_p90": _linear_quantile(
            global_absolute_errors, 0.90
        ),
        "action_l2_p90": _linear_quantile(global_action_l2, 0.90),
        "per_action": per_action,
    }
    if include_per_action_p90:
        result["per_action_absolute_error_p90"] = {
            str(action_index): _linear_quantile(
                global_absolute_errors[:, offset], 0.90
            )
            for offset, action_index in enumerate(action_indices)
        }
    return result


def _training_quality_record(
    record: Mapping[str, Any],
    *,
    probe: TrainingQualityProbe,
    rank: int,
) -> dict[str, Any]:
    """Relabel a fixed held-in probe as training-distribution evidence.

    ``evaluate_heldout_actions`` is intentionally reused because it already
    implements the deterministic, globally reduced action metrics.  This
    adapter prevents training examples from ever being published under the
    validation namespace merely because they share the same evaluator.
    """

    if record.get("schema") != HELDOUT_TELEMETRY_SCHEMA:
        raise LearnerError("training-quality source record schema mismatch")
    summary = probe.summary()
    rank = _true_int(rank, "training-quality probe record rank")
    if rank >= summary["world_size"]:
        raise LearnerError("training-quality probe record rank exceeds world size")
    expected_source = {
        "campaign_id": summary["campaign_id"],
        "macro_round": summary["macro_round"],
        "heldout_snapshot_sha256": summary["snapshot_sha256"],
        "heldout_snapshot_file_sha256": summary["snapshot_file_sha256"],
        "heldout_transitions_rank": summary["transitions_per_rank"][rank],
        "heldout_transitions_per_rank": summary["transitions_per_rank"],
        "heldout_transitions_global": summary["transitions_global"],
        "world_size": summary["world_size"],
    }
    if any(record.get(key) != value for key, value in expected_source.items()):
        raise LearnerError("training-quality probe/source identity mismatch")
    per_action_p90 = record.get("per_action_absolute_error_p90")
    if not isinstance(per_action_p90, Mapping):
        raise LearnerError("training-quality source lacks per-action p90")
    return {
        "schema": TRAINING_QUALITY_TELEMETRY_SCHEMA,
        "campaign_id": record["campaign_id"],
        "macro_round": record["macro_round"],
        "optimizer_step": record["optimizer_step"],
        "training_snapshot_sha256": record["heldout_snapshot_sha256"],
        "training_snapshot_file_sha256": record[
            "heldout_snapshot_file_sha256"
        ],
        "training_probe_policy": summary["policy"],
        "training_probe_budget": summary["budget"],
        "training_probe_partition_seed": summary["partition_seed"],
        "training_probe_selection_sha256": summary["selection_sha256"],
        "training_probe_rank_selection_sha256s": summary[
            "rank_selection_sha256s"
        ],
        "training_probe_record_rank": rank,
        "training_probe_rank_selection_sha256": summary[
            "rank_selection_sha256s"
        ][rank],
        "training_probe_transitions_rank": summary["transitions_per_rank"][rank],
        "training_probe_transitions_per_rank": summary["transitions_per_rank"],
        "training_probe_transitions_global": summary["transitions_global"],
        "training_probe_current_round_transitions_rank": summary[
            "current_round_transitions_per_rank"
        ][rank],
        "training_probe_current_round_transitions_per_rank": summary[
            "current_round_transitions_per_rank"
        ],
        "training_probe_current_round_transitions_global": summary[
            "current_round_transitions_global"
        ],
        "training_probe_historical_transitions_rank": summary[
            "historical_transitions_per_rank"
        ][rank],
        "training_probe_historical_transitions_per_rank": summary[
            "historical_transitions_per_rank"
        ],
        "training_probe_historical_transitions_global": summary[
            "historical_transitions_global"
        ],
        "training_replay_transitions_rank": summary[
            "replay_transitions_per_rank"
        ][rank],
        "training_replay_transitions_per_rank": summary[
            "replay_transitions_per_rank"
        ],
        "training_replay_transitions_global": summary[
            "replay_transitions_global"
        ],
        "world_size": record["world_size"],
        "learned_action_indices": record["learned_action_indices"],
        "action_mse": record["action_mse"],
        "action_mae": record["action_mae"],
        "action_rmse": record["action_rmse"],
        "action_cosine": record["action_cosine"],
        "action_absolute_error_p90": record["action_absolute_error_p90"],
        "action_l2_p90": record["action_l2_p90"],
        "per_action": record["per_action"],
        "per_action_absolute_error_p90": dict(per_action_p90),
    }


def summarize_training_rollout_context(
    partition: ReplayPartition,
    train_action_slice: tuple[int, int] | None,
    *,
    scheduled_beta: float,
    probe: TrainingQualityProbe | None = None,
) -> dict[str, float | int]:
    """Summarize the sealed DAgger collection that produced the replay.

    These are data-collection diagnostics, not learner evaluation.  Keeping
    them in the training section makes the expert intervention mix and replay
    coverage visible without contaminating the OOD validation section.
    """

    rank, world_size = _distributed_context()
    if rank != partition.rank or world_size != partition.world_size:
        raise LearnerError("training context partition does not match distributed context")
    scheduled_beta = _finite_float(scheduled_beta, "scheduled DAgger beta")
    if not 0.0 <= scheduled_beta <= 1.0:
        raise LearnerError("scheduled DAgger beta must be in [0,1]")
    probe_indices: tuple[int, ...] | None = None
    if probe is not None:
        summary = probe.summary()
        if (
            summary["campaign_id"] != partition.config.campaign_id
            or summary["macro_round"] != partition.config.macro_round
            or summary["world_size"] != world_size
            or summary["snapshot_file_sha256"] != partition.snapshot["file_sha256"]
            or summary["snapshot_sha256"] != partition.snapshot["snapshot_sha256"]
            or summary["replay_transitions_per_rank"]
            != list(partition.transition_counts_by_rank)
        ):
            raise LearnerError("training context probe identity mismatch")
        probe_indices = probe.indices_by_rank[rank]
    scalar = torch.zeros(9, dtype=torch.float64)
    action_l2_chunks: list[np.ndarray] = []
    environment_ids: set[int] = set()
    for shard in partition.shards:
        arrays = shard.arrays
        required = {
            "student_actions",
            "expert_actions",
            "behavior_actions",
            "expert_mask",
            "dones",
            "global_env_ids",
        }
        if not required <= set(arrays):
            raise LearnerError(
                "training replay lacks DAgger context arrays: "
                + ", ".join(sorted(required - set(arrays)))
            )
        student = torch.from_numpy(arrays["student_actions"]).float()
        expert = torch.from_numpy(arrays["expert_actions"]).float()
        behavior = torch.from_numpy(arrays["behavior_actions"]).float()
        student, expert, _ = _aligned_learned_actions(
            student, expert, train_action_slice
        )
        behavior, _, _ = _aligned_learned_actions(
            behavior, torch.from_numpy(arrays["expert_actions"]).float(), train_action_slice
        )
        difference = student.double() - expert.double()
        flat_difference = difference.reshape(-1, difference.shape[-1])
        flat_student = student.double().reshape(-1, student.shape[-1])
        flat_expert = expert.double().reshape(-1, expert.shape[-1])
        denominator = torch.linalg.vector_norm(flat_student, dim=-1) * torch.linalg.vector_norm(
            flat_expert, dim=-1
        )
        cosine = torch.zeros_like(denominator)
        nonzero = denominator > 0
        cosine[nonzero] = torch.sum(
            flat_student[nonzero] * flat_expert[nonzero], dim=-1
        ) / denominator[nonzero]
        cosine[
            (torch.linalg.vector_norm(flat_student, dim=-1) == 0)
            & (torch.linalg.vector_norm(flat_expert, dim=-1) == 0)
        ] = 1.0
        scalar[0] += torch.square(flat_difference).sum()
        scalar[1] += torch.abs(flat_difference).sum()
        scalar[2] += flat_difference.numel()
        scalar[3] += cosine.sum()
        scalar[4] += cosine.numel()
        scalar[5] += int(np.count_nonzero(arrays["expert_mask"]))
        scalar[6] += int(
            np.count_nonzero(
                np.all(
                    behavior.detach().cpu().numpy()
                    == expert.detach().cpu().numpy(),
                    axis=-1,
                )
            )
        )
        scalar[7] += int(np.count_nonzero(arrays["dones"]))
        scalar[8] += flat_difference.shape[0]
        if probe_indices is None:
            action_l2_chunks.append(
                torch.linalg.vector_norm(flat_difference, dim=-1).cpu().numpy()
            )
        ids = arrays["global_env_ids"]
        environment_ids.update(int(value) for value in ids.reshape(-1).tolist())
    if probe_indices is not None:
        student_rows: list[np.ndarray] = []
        expert_rows: list[np.ndarray] = []
        for index in probe_indices:
            shard, time_index, env_index = partition._location(index)
            student_rows.append(shard.arrays["student_actions"][time_index, env_index])
            expert_rows.append(shard.arrays["expert_actions"][time_index, env_index])
        student_probe = torch.from_numpy(np.stack(student_rows)).float()
        expert_probe = torch.from_numpy(np.stack(expert_rows)).float()
        student_probe, expert_probe, _ = _aligned_learned_actions(
            student_probe, expert_probe, train_action_slice
        )
        local_l2 = (
            torch.linalg.vector_norm(
                student_probe.double() - expert_probe.double(), dim=-1
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )
    else:
        if not action_l2_chunks:
            raise LearnerError("training context replay is empty")
        local_l2 = np.concatenate(action_l2_chunks).reshape(-1)
    if world_size > 1:
        gathered_scalars: list[Any] = [None] * world_size
        gathered_l2: list[Any] = [None] * world_size
        gathered_envs: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered_scalars, scalar.tolist())
        torch.distributed.all_gather_object(gathered_l2, local_l2.tolist())
        torch.distributed.all_gather_object(gathered_envs, sorted(environment_ids))
        scalar = torch.as_tensor(
            np.sum(np.asarray(gathered_scalars, dtype=np.float64), axis=0),
            dtype=torch.float64,
        )
        global_l2 = np.concatenate(
            [np.asarray(values, dtype=np.float64) for values in gathered_l2]
        )
        global_environment_ids = {
            int(value) for values in gathered_envs for value in values
        }
    else:
        global_l2 = local_l2
        global_environment_ids = environment_ids
    element_count = float(scalar[2])
    transition_count = int(scalar[8])
    if element_count <= 0 or transition_count <= 0:
        raise LearnerError("training context aggregation is empty")
    mse = float(scalar[0] / element_count)
    return {
        "parent_rollout_action_mse": mse,
        "parent_rollout_action_mae": float(scalar[1] / element_count),
        "parent_rollout_action_rmse": math.sqrt(mse),
        "parent_rollout_action_cosine": float(scalar[3] / scalar[4]),
        "parent_rollout_action_l2_p90": _linear_quantile(global_l2, 0.90),
        "expert_intervention_rate": float(scalar[5] / transition_count),
        "behavior_expert_match_rate": float(scalar[6] / transition_count),
        "expert_label_coverage": 1.0,
        "replay_transitions": transition_count,
        "replay_environments": len(global_environment_ids),
        "done_rate": float(scalar[7] / transition_count),
        "scheduled_beta": scheduled_beta,
    }


def _distributed_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _probe_seed(seed: int, rank: int) -> int:
    """Derive a stable RNG/sampling seed outside the optimizer schedule."""

    digest = hashlib.sha256(
        _PROBE_SEED_DOMAIN
        + b":"
        + str(_true_int(seed, "seed")).encode("ascii")
        + b":"
        + str(_true_int(rank, "rank")).encode("ascii")
    ).digest()
    # Keep the seed in the range accepted by every supported Torch backend.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _evaluate_loss_probe(
    model: nn.Module,
    *,
    partition: ReplayPartition,
    indices: Sequence[int],
    adapter: PolicyAdapter,
    device: torch.device,
    train_action_slice: tuple[int, int] | None,
    seed: int,
    rank: int,
    world_size: int,
) -> float:
    """Evaluate one immutable replay probe without perturbing training state.

    The same rank-local examples, eval mode, and domain-separated RNG state are
    used before and after optimization.  Saving and restoring all RNG streams
    makes the probe observational: stochastic policy code cannot change the
    subsequent optimizer trajectory or restart frontier.
    """

    prior_training = model.training
    prior_rng_state = _rng_state()
    try:
        probe_seed = _probe_seed(seed, rank)
        random.seed(probe_seed)
        np.random.seed(probe_seed % (2**32))
        torch.manual_seed(probe_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(probe_seed)
        model.eval()
        observations, targets = partition.batch(indices, device=device)
        with torch.no_grad():
            loss = supervised_action_loss(
                adapter.actions(model, observations),
                targets,
                train_action_slice,
            )
        if not torch.isfinite(loss):
            raise LearnerError("non-finite deterministic loss probe")
        scalar = loss.detach()
    finally:
        model.train(prior_training)
        _restore_rng_state(prior_rng_state)
    if world_size > 1:
        scalar = _ordered_allgather_sum(scalar, world_size) / world_size
    return float(scalar.cpu())


def _atomic_torch_save(path: Path, document: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(document), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Local H200 ranks may run as root while the campaign is owned by the
        # submitting user. Keep restart state owner-writable and readable by
        # later local stages so its digest can be verified.
        os.chmod(
            path,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        )
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _checkpoint_document(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: LearnerBundleConfig,
    snapshot: Mapping[str, Any],
    completed_steps: int,
    initial_loss: float,
    rng_states: Sequence[Mapping[str, Any]],
    world_size: int,
    gradient_reduction: str,
    learner_identity_sha256: str | None,
    runtime_container_sha256: str | None,
    learner_contract_sha256: str | None,
) -> dict[str, Any]:
    policy = _unwrap_model(model)
    policy_state = policy.state_dict()
    model_config = config.to_mapping()["model_config"]
    raw_controller_sha256 = model_config.get("homie_controller_sha256")
    try:
        controller_sha256 = (
            None
            if raw_controller_sha256 is None
            else validate_homie_controller_sha256(raw_controller_sha256)
        )
    except PolicyIdentityError as exc:
        raise LearnerError(str(exc)) from exc
    exposure = optimizer_exposure_summary(
        snapshot,
        config,
        completed_steps=completed_steps,
        world_size=world_size,
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        # These two fields make the promoted payload directly consumable by
        # the existing GR00T actor checkpoint loader, while ``schema`` and the
        # lineage fields below retain the stronger round-learner contract.
        "checkpoint_schema": "gr00t.trl.full_state.v2",
        "policy_state_dict": policy_state,
        "policy_parameter_names": sorted(name for name, _ in policy.named_parameters()),
        "homie_controller_sha256": controller_sha256,
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "training_counts_by_round": dict(snapshot["counts_by_round"]),
        "completed_steps": completed_steps,
        **exposure,
        # This is the eval-mode loss on the deterministic replay probe before
        # the first update.  Keeping it in the restart frontier prevents a
        # resumed run from replacing the comparable experiment baseline.
        "initial_loss": initial_loss,
        "world_size": world_size,
        "gradient_reduction": gradient_reduction,
        "model_state_dict": policy_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None,
        "value_state_dict": None,
        "rng_states": list(rng_states),
    }


def _load_restart(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: LearnerBundleConfig,
    snapshot: Mapping[str, Any],
    rank: int,
    world_size: int,
    gradient_reduction: str,
    learner_identity_sha256: str | None,
    runtime_container_sha256: str | None,
    learner_contract_sha256: str | None,
) -> tuple[int, float]:
    if path.is_symlink() or not path.is_file():
        raise LearnerError("restart checkpoint must be a regular non-symlink file")
    try:
        checkpoint = load_restricted_restart_document(path, map_location="cpu")
    except CheckpointCompatibilityError as exc:
        raise LearnerError(f"cannot load restart checkpoint: {exc}") from exc
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise LearnerError("restart checkpoint schema mismatch")
    expected = {
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "training_counts_by_round": dict(snapshot["counts_by_round"]),
        "world_size": world_size,
        "gradient_reduction": gradient_reduction,
        "homie_controller_sha256": config.to_mapping()["model_config"].get(
            "homie_controller_sha256"
        ),
    }
    mismatches = [
        key for key, value in expected.items() if checkpoint.get(key) != value
    ]
    if mismatches:
        raise LearnerError(f"restart lineage mismatch: {', '.join(mismatches)}")
    completed = _true_int(checkpoint.get("completed_steps"), "completed_steps")
    exposure = optimizer_exposure_summary(
        snapshot,
        config,
        completed_steps=completed,
        world_size=world_size,
    )
    if any(checkpoint.get(key) != value for key, value in exposure.items()):
        raise LearnerError("restart optimizer-example exposure mismatch")
    initial_loss = _finite_float(checkpoint.get("initial_loss"), "initial_loss")
    if initial_loss < 0.0:
        raise LearnerError("initial_loss must be non-negative")
    try:
        _unwrap_model(model).load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        rng_states = checkpoint["rng_states"]
        if not isinstance(rng_states, Sequence) or len(rng_states) != world_size:
            raise LearnerError("restart does not contain one RNG state per rank")
        _restore_rng_state(rng_states[rank])
    except Exception as exc:
        raise LearnerError(f"restart state restoration failed: {exc}") from exc
    return completed, initial_loss


class JsonlTelemetry:
    """Append rank-zero learner metrics to one local JSONL file."""

    def __init__(
        self, path: str | os.PathLike[str] | None, *, enabled: bool = True
    ) -> None:
        self.path = (
            None
            if path is None or not enabled
            else Path(os.path.abspath(Path(path).expanduser()))
        )
        if self.path is not None:
            if self.path.is_symlink():
                raise LearnerError("telemetry path must not be a symbolic link")
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        try:
            payload = (
                json.dumps(
                    dict(record), sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        except (TypeError, ValueError) as exc:
            raise LearnerError("telemetry record is not canonical JSON") from exc
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.path, flags, 0o644)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def close(self, *, exit_code: int = 0) -> None:
        if type(exit_code) is not int:
            raise LearnerError("telemetry exit code must be an integer")


@dataclass(frozen=True)
class TrainResult:
    completed_steps: int
    target_steps: int
    exposure: Mapping[str, int | float | str]
    initial_loss: float
    final_loss: float
    restart_checkpoint: str
    replay_snapshot_sha256: str
    learner_config_sha256: str


def train_from_replay(
    model: nn.Module,
    *,
    snapshot_path: str | os.PathLike[str],
    config: LearnerBundleConfig,
    output_dir: str | os.PathLike[str],
    target_optimizer_examples: int,
    adapter: PolicyAdapter | None = None,
    device: torch.device | str = "cpu",
    resume_from: str | os.PathLike[str] | None = None,
    expected_snapshot_file_sha256: str | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    checkpoint_every_optimizer_examples: int,
    telemetry: JsonlTelemetry | None = None,
    learner_identity_sha256: str | None = None,
    runtime_container_sha256: str | None = None,
    learner_contract_sha256: str | None = None,
) -> TrainResult:
    """Run supervised DAgger updates over one immutable cumulative snapshot."""

    target_optimizer_examples = _true_int(
        target_optimizer_examples, "target_optimizer_examples", minimum=1
    )
    snapshot_path = _require_campaign_storage_path(
        snapshot_path,
        "train replay snapshot",
        generation_id=config.generation_id,
    )
    output_dir = _require_campaign_storage_path(
        output_dir,
        "learner output",
        generation_id=config.generation_id,
    )
    if resume_from is not None:
        resume_from = _require_campaign_storage_path(
            resume_from, "restart checkpoint"
        )
    checkpoint_every_optimizer_examples = _true_int(
        checkpoint_every_optimizer_examples,
        "checkpoint_every_optimizer_examples",
        minimum=1,
    )
    semantic_identity = (
        learner_identity_sha256,
        runtime_container_sha256,
        learner_contract_sha256,
    )
    if any(value is None for value in semantic_identity) and any(
        value is not None for value in semantic_identity
    ):
        raise LearnerError("restart semantic identity fields are incomplete")
    if all(value is not None for value in semantic_identity):
        learner_identity_sha256 = _require_sha256(
            learner_identity_sha256, "learner identity SHA-256"
        )
        runtime_container_sha256 = _require_sha256(
            runtime_container_sha256, "runtime container SHA-256"
        )
        learner_contract_sha256 = _require_sha256(
            learner_contract_sha256, "learner contract SHA-256"
        )
    rank, world_size = _distributed_context()
    examples_per_step = config.batch_size * world_size
    max_steps, remainder = divmod(target_optimizer_examples, examples_per_step)
    if remainder or max_steps < 1:
        raise LearnerError(
            "target_optimizer_examples must be a positive multiple of "
            "batch_size * world_size"
        )
    if target_optimizer_examples > config.target_optimizer_examples:
        raise LearnerError(
            "target_optimizer_examples exceeds the immutable stage maximum"
        )
    save_every, save_remainder = divmod(
        checkpoint_every_optimizer_examples, examples_per_step
    )
    if (
        save_remainder
        or save_every < 1
        or checkpoint_every_optimizer_examples
        != config.checkpoint_every_optimizer_examples
    ):
        raise LearnerError(
            "checkpoint_every_optimizer_examples must be a positive bundle-pinned "
            "multiple of batch_size * world_size"
        )
    device = torch.device(device)
    configure_deterministic_runtime(device)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    partition = load_replay_partition(
        snapshot_path,
        config,
        rank=rank,
        world_size=world_size,
        expected_snapshot_file_sha256=expected_snapshot_file_sha256,
        artifact_root=artifact_root,
    )
    model = model.to(device)
    gradient_reduction = SINGLE_PROCESS_GRADIENT_REDUCTION
    if world_size > 1 and not isinstance(model, DistributedDataParallel):
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
        gradient_reduction = ORDERED_ALLGATHER_GRADIENT_REDUCTION
        model.register_comm_hook(world_size, _ordered_allgather_gradient_hook)
    elif world_size > 1:
        raise LearnerError(
            "caller-supplied DistributedDataParallel cannot prove the ordered "
            "gradient-reduction contract"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        # Pin the implementation rather than allowing PyTorch's device- and
        # release-dependent foreach heuristic to become part of the restart
        # state implicitly.  The convolution backward was the observed source
        # of the CUDA divergence, but an explicit optimizer path keeps the
        # complete update contract reproducible as well.
        foreach=False,
        fused=False,
    )
    restart_path = Path(output_dir).expanduser().resolve() / "restart.pt"
    completed_steps = 0
    first_loss: float | None = None
    if resume_from is not None:
        completed_steps, first_loss = _load_restart(
            Path(resume_from).expanduser().resolve(),
            model=model,
            optimizer=optimizer,
            config=config,
            snapshot=partition.snapshot,
            rank=rank,
            world_size=world_size,
            gradient_reduction=gradient_reduction,
            learner_identity_sha256=learner_identity_sha256,
            runtime_container_sha256=runtime_container_sha256,
            learner_contract_sha256=learner_contract_sha256,
        )
    if completed_steps > max_steps:
        raise LearnerError("restart completed_steps exceeds requested max_steps")
    training_frontier = max_steps
    saved_at_step = (
        completed_steps
        if resume_from is not None
        and Path(resume_from).expanduser().resolve() == restart_path
        else -1
    )
    schedule = CumulativeDaggerBatchSchedule(
        partition, config.batch_size, seed=config.seed
    )
    adapter = adapter or MappingPolicyAdapter(config.output_key)
    probe_schedule = DeterministicBatchSchedule(
        partition.transition_count,
        config.batch_size,
        seed=_probe_seed(config.seed, rank),
    )
    loss_probe_indices = probe_schedule.indices(0)
    if first_loss is None:
        first_loss = _evaluate_loss_probe(
            model,
            partition=partition,
            indices=loss_probe_indices,
            adapter=adapter,
            device=device,
            train_action_slice=config.train_action_slice,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
        )
    model.train()
    for step in range(completed_steps, training_frontier):
        observations, targets = partition.batch(schedule.indices(step), device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions = adapter.actions(model, observations)
        loss = supervised_action_loss(predictions, targets, config.train_action_slice)
        if not torch.isfinite(loss):
            raise LearnerError(f"non-finite supervised loss at optimizer step {step}")
        loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            _unwrap_model(model).parameters(),
            max_norm=float("inf"),
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )
        gradient_norm = float(gradient_norm_tensor.detach().cpu())
        if not math.isfinite(gradient_norm):
            raise LearnerError(
                f"non-finite gradient norm at optimizer step {step}"
            )
        optimizer.step()
        scalar = loss.detach()
        if world_size > 1:
            scalar = _ordered_allgather_sum(scalar, world_size) / world_size
        loss_value = float(scalar.cpu())
        completed_steps = step + 1
        if rank == 0 and telemetry is not None:
            exposure = optimizer_exposure_summary(
                partition.snapshot,
                config,
                completed_steps=completed_steps,
                world_size=world_size,
            )
            telemetry.log(
                {
                    "schema": TELEMETRY_SCHEMA,
                    "campaign_id": config.campaign_id,
                    "generation_id": config.generation_id,
                    "macro_round": config.macro_round,
                    "optimizer_step": completed_steps,
                    "loss": loss_value,
                    "gradient_l2_norm": gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **exposure,
                    "replay_transitions_rank": partition.transition_count,
                    "replay_transitions_per_rank": list(
                        partition.transition_counts_by_rank
                    ),
                    "replay_transitions_global": partition.global_transition_count,
                    "world_size": world_size,
                    "gradient_reduction": gradient_reduction,
                },
            )
        if (
            completed_steps % save_every == 0
            or completed_steps == training_frontier
        ):
            local_rng_state = _rng_state()
            if world_size > 1:
                gathered_rng_states: list[Any] = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(
                    gathered_rng_states, local_rng_state
                )
            else:
                gathered_rng_states = [local_rng_state]
            if rank == 0:
                _atomic_torch_save(
                    restart_path,
                    _checkpoint_document(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        snapshot=partition.snapshot,
                        completed_steps=completed_steps,
                        initial_loss=first_loss,
                        rng_states=gathered_rng_states,
                        world_size=world_size,
                        gradient_reduction=gradient_reduction,
                        learner_identity_sha256=learner_identity_sha256,
                        runtime_container_sha256=runtime_container_sha256,
                        learner_contract_sha256=learner_contract_sha256,
                    ),
                )
            saved_at_step = completed_steps
            if world_size > 1:
                torch.distributed.barrier()
    if saved_at_step != completed_steps:
        local_rng_state = _rng_state()
        if world_size > 1:
            gathered_rng_states = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_rng_states, local_rng_state)
        else:
            gathered_rng_states = [local_rng_state]
        if rank == 0:
            _atomic_torch_save(
                restart_path,
                _checkpoint_document(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    snapshot=partition.snapshot,
                    completed_steps=completed_steps,
                    initial_loss=first_loss,
                    rng_states=gathered_rng_states,
                    world_size=world_size,
                    gradient_reduction=gradient_reduction,
                    learner_identity_sha256=learner_identity_sha256,
                    runtime_container_sha256=runtime_container_sha256,
                    learner_contract_sha256=learner_contract_sha256,
                ),
            )
        if world_size > 1:
            torch.distributed.barrier()

    # Compare the updated policy on exactly the same examples, eval mode, and
    # deterministic RNG stream as the pre-update baseline.  This probe is
    # deliberately observational and is not serialized into the restart RNG
    # frontier.
    final_loss = _evaluate_loss_probe(
        model,
        partition=partition,
        indices=loss_probe_indices,
        adapter=adapter,
        device=device,
        train_action_slice=config.train_action_slice,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )
    exposure = optimizer_exposure_summary(
        partition.snapshot,
        config,
        completed_steps=completed_steps,
        world_size=world_size,
    )
    return TrainResult(
        completed_steps=completed_steps,
        target_steps=max_steps,
        exposure=exposure,
        initial_loss=first_loss,
        final_loss=final_loss,
        restart_checkpoint=str(restart_path),
        replay_snapshot_sha256=partition.snapshot["snapshot_sha256"],
        learner_config_sha256=config.sha256,
    )


def _private_binary_snapshot(
    source: Path, destination: Path
) -> tuple[Path, str, bool]:
    """Copy a possibly mutable source once into a private same-filesystem inode."""

    source = Path(os.path.abspath(source.expanduser()))
    source_descriptor: int | None = None
    snapshot: Path | None = None
    snapshot_descriptor: int | None = None
    snapshot_ready = False
    destination = destination.expanduser().resolve()
    parent_preexisted = destination.parent.exists()
    try:
        source_descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise LearnerError("candidate source checkpoint is missing or unsafe")
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot_descriptor, snapshot_name = tempfile.mkstemp(
            prefix=f".{destination.name}.snapshot.", dir=destination.parent
        )
        snapshot = Path(snapshot_name)
        with os.fdopen(source_descriptor, "rb") as incoming, os.fdopen(
            snapshot_descriptor, "wb"
        ) as outgoing:
            source_descriptor = None
            snapshot_descriptor = None
            for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.chmod(snapshot, stat.S_IRUSR)
        snapshot_sha256 = sha256_file(snapshot)
        snapshot_ready = True
        return snapshot, snapshot_sha256, parent_preexisted
    except OSError as exc:
        raise LearnerError("candidate source checkpoint is missing or unsafe") from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)
        if not snapshot_ready:
            if snapshot is not None:
                try:
                    snapshot.unlink()
                except FileNotFoundError:
                    pass
            if not parent_preexisted:
                try:
                    destination.parent.rmdir()
                except OSError:
                    pass


def _publish_immutable_binary(
    snapshot: Path,
    destination: Path,
    *,
    expected_sha256: str,
    parent_preexisted: bool,
) -> str:
    """Link one already-validated private snapshot into its canonical basename."""

    snapshot = Path(os.path.abspath(snapshot.expanduser()))
    destination = destination.expanduser().resolve()
    expected = _require_sha256(
        expected_sha256, "candidate source checkpoint SHA-256"
    )
    published = False
    created_destination = False
    try:
        if (
            snapshot.is_symlink()
            or not snapshot.is_file()
            or snapshot.parent != destination.parent
            or sha256_file(snapshot) != expected
        ):
            raise LearnerError(
                "private candidate checkpoint changed before publication"
            )
        if destination.exists():
            if destination.is_symlink() or sha256_file(destination) != expected:
                raise LearnerError(
                    "immutable candidate path already contains different bytes"
                )
            published = True
            return expected
        try:
            os.link(snapshot, destination)
            created_destination = True
        except FileExistsError:
            if destination.is_symlink() or sha256_file(destination) != expected:
                raise LearnerError("competing candidate publication changed bytes")
        try:
            os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            directory = os.open(
                destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            if created_destination:
                destination.unlink(missing_ok=True)
                created_destination = False
            raise
        published = True
        return expected
    finally:
        if not published and created_destination:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        try:
            snapshot.unlink()
        except FileNotFoundError:
            pass
        if not published and not parent_preexisted:
            try:
                destination.parent.rmdir()
            except OSError:
                pass


def _discard_private_binary_snapshot(
    snapshot: Path, destination: Path, *, parent_preexisted: bool
) -> None:
    try:
        snapshot.unlink()
    except FileNotFoundError:
        pass
    if not parent_preexisted:
        try:
            destination.parent.rmdir()
        except OSError:
            pass


def _validate_candidate_attestation(
    *,
    checkpoint: Mapping[str, Any],
    result: TrainResult,
    config: LearnerBundleConfig,
    learner_identity_sha256: str | None,
    runtime_container_sha256: str | None,
    learner_contract_sha256: str | None,
) -> dict[str, str] | None:
    """Verify the exact execution identity before publishing a candidate.

    The library keeps an all-null development mode so existing local restart
    workflows and legacy checkpoints remain readable.  Supplying any identity
    field, however, selects the production path: all three values must be
    present, must match the saved restart, and the learner identity must bind
    the complete immutable training semantics again at publication time.
    """

    supplied = (
        learner_identity_sha256,
        runtime_container_sha256,
        learner_contract_sha256,
    )
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise LearnerError("candidate attestation fields are incomplete")
    identity_sha = _require_sha256(
        learner_identity_sha256, "learner identity SHA-256"
    )
    container_sha = _require_sha256(
        runtime_container_sha256, "runtime container SHA-256"
    )
    contract_sha = _require_sha256(
        learner_contract_sha256, "learner contract SHA-256"
    )
    checkpoint_expected = {
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "learner_identity_sha256": identity_sha,
        "runtime_container_sha256": container_sha,
        "learner_contract_sha256": contract_sha,
        "replay_snapshot_sha256": result.replay_snapshot_sha256,
        **dict(result.exposure),
    }
    mismatches = [
        key
        for key, expected in checkpoint_expected.items()
        if checkpoint.get(key) != expected
    ]
    if mismatches:
        raise LearnerError(
            "candidate checkpoint attestation mismatch: " + ", ".join(mismatches)
        )
    if result.learner_config_sha256 != config.sha256:
        raise LearnerError("learner result/config identity mismatch")
    replay_file_sha = _require_sha256(
        checkpoint.get("replay_snapshot_file_sha256"),
        "replay snapshot file SHA-256",
    )
    parent_runtime_sha = _require_sha256(
        config.to_mapping()["model_config"].get("runtime_policy_sha256"),
        "parent runtime policy SHA-256",
    )
    try:
        computed_identity_sha = compute_learner_identity_sha256(
            campaign_id=config.campaign_id,
            macro_round=config.macro_round,
            learner_code_sha256=config.code_sha256,
            learner_config_sha256=config.sha256,
            replay_snapshot_file_sha256=replay_file_sha,
            replay_snapshot_sha256=result.replay_snapshot_sha256,
            parent_checkpoint_sha256=config.model_lineage_sha256,
            parent_runtime_policy_sha256=parent_runtime_sha,
            runtime_container_sha256=container_sha,
            learner_contract_sha256=contract_sha,
        )
    except LearnerIdentityError as exc:
        raise LearnerError(f"candidate attestation is invalid: {exc}") from exc
    if computed_identity_sha != identity_sha:
        raise LearnerError(
            "candidate learner identity does not bind the exact training semantics"
        )
    return {
        "learner_identity_sha256": identity_sha,
        "runtime_container_sha256": container_sha,
        "learner_contract_sha256": contract_sha,
    }


def publish_learner_result(
    *,
    output_dir: str | os.PathLike[str],
    result: TrainResult,
    config: LearnerBundleConfig,
    learner_identity_sha256: str | None = None,
    runtime_container_sha256: str | None = None,
    learner_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal a learner payload and a manifest consumable by policy staging."""

    if result.completed_steps != result.target_steps:
        raise LearnerError("incomplete learner result cannot be published")

    output = _require_campaign_storage_path(
        output_dir,
        "learner output",
        generation_id=config.generation_id,
    )
    checkpoint_name = f"candidate_round_{config.macro_round:06d}.pt"
    checkpoint_path = output / checkpoint_name
    restart_path = Path(result.restart_checkpoint)
    private_checkpoint, checkpoint_sha, parent_preexisted = _private_binary_snapshot(
        restart_path, checkpoint_path
    )
    try:
        checkpoint_document = load_restricted_restart_document(
            private_checkpoint, map_location="cpu"
        )
        if not isinstance(checkpoint_document, Mapping):
            raise LearnerError("restart checkpoint root must be a mapping")
        attestation = _validate_candidate_attestation(
            checkpoint=checkpoint_document,
            result=result,
            config=config,
            learner_identity_sha256=learner_identity_sha256,
            runtime_container_sha256=runtime_container_sha256,
            learner_contract_sha256=learner_contract_sha256,
        )
        policy_state = checkpoint_document.get("policy_state_dict")
        if not isinstance(policy_state, Mapping) or not policy_state:
            raise LearnerError(
                "candidate checkpoint has no non-empty policy_state_dict"
            )
        policy_state_sha = runtime_state_dict_sha256(policy_state)
        policy_spec = config.to_mapping()["model_config"]
        policy_spec = validate_policy_spec(policy_spec)
        controller_sha256 = validate_homie_controller_sha256(
            policy_spec["homie_controller_sha256"]
        )
        checkpoint_controller_sha256 = validate_homie_controller_sha256(
            checkpoint_document.get("homie_controller_sha256")
        )
        if checkpoint_controller_sha256 != controller_sha256:
            raise LearnerError(
                "candidate checkpoint Homie controllers differ from learner policy spec"
            )
        runtime_policy_sha = runtime_policy_identity_sha256(
            policy_state_sha, controller_sha256
        )
        policy_spec["checkpoint_sha256"] = checkpoint_sha
        policy_spec["policy_state_sha256"] = policy_state_sha
        policy_spec["runtime_policy_sha256"] = runtime_policy_sha
        policy_spec = validate_policy_spec(policy_spec)
    except (
        ActorExportError,
        GR00TPolicyError,
        PolicyIdentityError,
        CheckpointCompatibilityError,
        KeyError,
        TypeError,
    ) as exc:
        _discard_private_binary_snapshot(
            private_checkpoint,
            checkpoint_path,
            parent_preexisted=parent_preexisted,
        )
        raise LearnerError(f"cannot publish candidate policy sidecars: {exc}") from exc
    except BaseException:
        _discard_private_binary_snapshot(
            private_checkpoint,
            checkpoint_path,
            parent_preexisted=parent_preexisted,
        )
        raise

    # Hashing, restricted loading, identity validation, and policy derivation
    # all use one private snapshot.  Publication links that exact inode, so a
    # concurrent replacement or mutation of the original restart cannot split
    # candidate bytes from the semantics that were attested above.
    published_checkpoint_sha = _publish_immutable_binary(
        private_checkpoint,
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        parent_preexisted=parent_preexisted,
    )
    if published_checkpoint_sha != checkpoint_sha:
        raise LearnerError("candidate checkpoint changed after attestation")

    policy_spec_name = f"{checkpoint_name}.policy_spec.json"
    policy_spec_path = output / policy_spec_name
    policy_spec_sha = publish_immutable_json(policy_spec_path, policy_spec)
    policy_bundle_manifest_name = f"{checkpoint_name}.manifest.json"
    policy_bundle_manifest_path = output / policy_bundle_manifest_name
    policy_bundle_manifest = {
        "schema": POLICY_BUNDLE_MANIFEST_SCHEMA,
        "checkpoint_path": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "runtime_policy_sha256": runtime_policy_sha,
        "policy_spec_path": policy_spec_name,
        "policy_spec_sha256": policy_spec_sha,
    }
    policy_bundle_manifest_sha = publish_immutable_json(
        policy_bundle_manifest_path, policy_bundle_manifest
    )
    try:
        validate_policy_bundle_identity(
            checkpoint=checkpoint_path,
            policy_spec=policy_spec_path,
            manifest=policy_bundle_manifest_path,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_runtime_policy_sha256=runtime_policy_sha,
        )
    except PolicyBundleError as exc:
        raise LearnerError(f"published candidate policy bundle is invalid: {exc}") from exc
    document = {
        "schema": RESULT_SCHEMA if attestation is not None else LEGACY_RESULT_SCHEMA,
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "checkpoint_path": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "runtime_policy_sha256": runtime_policy_sha,
        "policy_spec_path": policy_spec_name,
        "policy_spec_sha256": policy_spec_sha,
        "policy_bundle_manifest_path": policy_bundle_manifest_name,
        "policy_bundle_manifest_sha256": policy_bundle_manifest_sha,
        "model_lineage_sha256": config.model_lineage_sha256,
        "replay_snapshot_sha256": result.replay_snapshot_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": result.learner_config_sha256,
        "optimizer_steps": result.completed_steps,
        **dict(result.exposure),
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        **({} if attestation is None else attestation),
    }
    manifest_path = output / f"learner_result_round_{config.macro_round:06d}.json"
    publish_immutable_json(manifest_path, document)
    return {
        **document,
        "manifest_path": str(manifest_path),
        "manifest_sha256": canonical_json_sha256(document),
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "LEGACY_RESULT_SCHEMA",
    "RESULT_SCHEMA",
    "HELDOUT_TELEMETRY_SCHEMA",
    "TRAINING_QUALITY_TELEMETRY_SCHEMA",
    "TRAINING_CONTEXT_TELEMETRY_SCHEMA",
    "TRAINING_QUALITY_PROBE_POLICY",
    "TRAINING_QUALITY_PROBE_BUDGET",
    "LearnerError",
    "LearnerBundleConfig",
    "PolicyAdapter",
    "MappingPolicyAdapter",
    "ReplayPartition",
    "TrainingQualityProbe",
    "DeterministicBatchSchedule",
    "CumulativeDaggerBatchSchedule",
    "JsonlTelemetry",
    "TrainResult",
    "partition_shards",
    "build_training_quality_probe",
    "validate_training_quality_probe_summary",
    "load_replay_partition",
    "optimizer_source_exposure",
    "optimizer_exposure_summary",
    "supervised_action_loss",
    "evaluate_heldout_actions",
    "summarize_training_rollout_context",
    "CUDA_CUBLAS_WORKSPACE_CONFIG",
    "SINGLE_PROCESS_GRADIENT_REDUCTION",
    "ORDERED_ALLGATHER_GRADIENT_REDUCTION",
    "configure_deterministic_runtime",
    "train_from_replay",
    "publish_learner_result",
]
