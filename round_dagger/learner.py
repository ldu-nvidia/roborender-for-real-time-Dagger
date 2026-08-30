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
from round_dagger.replay import load_heldout_snapshot, load_snapshot


BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
CHECKPOINT_SCHEMA = "fast_roborender.round_dagger.learner_checkpoint.v3"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v2"
HELDOUT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_heldout_telemetry.v1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
CUDA_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_PROBE_SEED_DOMAIN = b"fast_roborender.round_dagger.loss_probe.v1"


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


@dataclass(frozen=True)
class LearnerBundleConfig:
    """Immutable policy/input/optimizer contract shared by every learner rank."""

    campaign_id: str
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
    # Production supplies its 1200-step target explicitly through the sealed
    # learner contract.  Keep the generic bundle default stable for smoke and
    # library callers that intentionally omit a production target.
    max_steps: int = 500
    save_every: int = 100
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    seed: int = 0
    partition_seed: int = 0
    initial_checkpoint_sha256: str | None = None
    heldout_snapshot_sha256: str | None = None
    heldout_snapshot_file_sha256: str | None = None
    heldout_eval_interval: int | None = None
    model_config_json: str = "{}"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LearnerBundleConfig":
        if not isinstance(raw, Mapping):
            raise LearnerError("learner bundle must be a mapping")
        allowed = {
            "schema",
            "campaign_id",
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
            "max_steps",
            "save_every",
            "learning_rate",
            "weight_decay",
            "seed",
            "partition_seed",
            "initial_checkpoint_sha256",
            "heldout_snapshot_sha256",
            "heldout_snapshot_file_sha256",
            "heldout_eval_interval",
            "model_config",
        }
        unknown = set(raw) - allowed
        required = {
            "schema",
            "campaign_id",
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
        if not isinstance(campaign_id, str) or not campaign_id:
            raise LearnerError("campaign_id must be a non-empty string")

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
        heldout_values = (
            raw.get("heldout_snapshot_sha256"),
            raw.get("heldout_snapshot_file_sha256"),
            raw.get("heldout_eval_interval"),
        )
        if any(value is not None for value in heldout_values) and any(
            value is None for value in heldout_values
        ):
            raise LearnerError(
                "held-out evaluation fields must be supplied together"
            )
        heldout_snapshot_sha256 = None
        heldout_snapshot_file_sha256 = None
        heldout_eval_interval = None
        if all(value is not None for value in heldout_values):
            heldout_snapshot_sha256 = _require_sha256(
                heldout_values[0], "heldout_snapshot_sha256"
            )
            heldout_snapshot_file_sha256 = _require_sha256(
                heldout_values[1], "heldout_snapshot_file_sha256"
            )
            heldout_eval_interval = _true_int(
                heldout_values[2], "heldout_eval_interval", minimum=1
            )
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
            macro_round=_true_int(raw["macro_round"], "macro_round"),
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
            max_steps=_true_int(raw.get("max_steps", 500), "max_steps", minimum=1),
            save_every=_true_int(raw.get("save_every", 100), "save_every", minimum=1),
            learning_rate=_finite_float(
                raw.get("learning_rate", 1.0e-4), "learning_rate", positive=True
            ),
            weight_decay=weight_decay,
            seed=_true_int(raw.get("seed", 0), "seed"),
            partition_seed=_true_int(raw.get("partition_seed", 0), "partition_seed"),
            initial_checkpoint_sha256=initial_sha,
            heldout_snapshot_sha256=heldout_snapshot_sha256,
            heldout_snapshot_file_sha256=heldout_snapshot_file_sha256,
            heldout_eval_interval=heldout_eval_interval,
            model_config_json=model_config_json,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LearnerBundleConfig":
        return cls.from_mapping(load_strict_json(path))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": self.campaign_id,
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
            "max_steps": self.max_steps,
            "save_every": self.save_every,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "partition_seed": self.partition_seed,
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "model_config": json.loads(self.model_config_json),
        }
        if self.heldout_snapshot_sha256 is not None:
            result.update(
                heldout_snapshot_sha256=self.heldout_snapshot_sha256,
                heldout_snapshot_file_sha256=self.heldout_snapshot_file_sha256,
                heldout_eval_interval=self.heldout_eval_interval,
            )
        return result

    @property
    def heldout_evaluation_enabled(self) -> bool:
        return self.heldout_snapshot_sha256 is not None

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

    # Largest shards are placed first.  The seeded digest makes ties stable
    # without depending on manifest order or Python hash randomization.
    normalized.sort(key=lambda item: (-item[0], item[1], item[2]))
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    transition_totals = [0 for _ in range(world_size)]
    shard_totals = [0 for _ in range(world_size)]
    for transitions, _digest, _identity, shard in normalized:
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
        total = 0
        obs_keys: set[str] | None = None
        for shard in self.shards:
            total += shard.transitions
            self._ends.append(total)
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
) -> ReplayPartition:
    """Load only the rank's shards after replay and held-FD digest validation."""

    rank = _true_int(rank, "rank")
    world_size = _true_int(world_size, "world_size", minimum=1)
    if rank >= world_size:
        raise LearnerError("rank must be smaller than world_size")
    try:
        snapshot = load_snapshot(
            snapshot_path, expected_file_sha256=expected_snapshot_file_sha256
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
        snapshot, config, rank=rank, world_size=world_size
    )


def _partition_loaded_snapshot(
    snapshot: Mapping[str, Any],
    config: LearnerBundleConfig,
    *,
    rank: int,
    world_size: int,
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
    )


def load_heldout_partition(
    snapshot_path: str | os.PathLike[str],
    config: LearnerBundleConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> ReplayPartition:
    """Load one rank's immutable evaluation-only replay partition."""

    if not config.heldout_evaluation_enabled:
        raise LearnerError("learner bundle does not enable held-out evaluation")
    rank = _true_int(rank, "rank")
    world_size = _true_int(world_size, "world_size", minimum=1)
    if rank >= world_size:
        raise LearnerError("rank must be smaller than world_size")
    try:
        snapshot = load_heldout_snapshot(
            snapshot_path,
            expected_file_sha256=config.heldout_snapshot_file_sha256,
        )
    except Exception as exc:
        raise LearnerError(
            f"held-out replay snapshot validation failed: {exc}"
        ) from exc
    if snapshot["snapshot_sha256"] != config.heldout_snapshot_sha256:
        raise LearnerError("held-out replay snapshot semantic digest mismatch")
    if snapshot["campaign_id"] != config.campaign_id:
        raise LearnerError("learner campaign does not match held-out replay")
    return _partition_loaded_snapshot(
        snapshot, config, rank=rank, world_size=world_size
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


def evaluate_heldout_actions(
    model: nn.Module,
    *,
    partition: ReplayPartition,
    adapter: PolicyAdapter,
    device: torch.device | str,
    optimizer_step: int,
) -> dict[str, Any]:
    """Evaluate every held-out transition and globally reduce exact sums."""

    optimizer_step = _true_int(optimizer_step, "optimizer_step")
    device = torch.device(device)
    rank, world_size = _distributed_context()
    if rank != partition.rank or world_size != partition.world_size:
        raise LearnerError("held-out partition does not match distributed context")
    preserved_rng = _rng_state()
    was_training = model.training
    model.eval()
    statistics: torch.Tensor | None = None
    action_indices: tuple[int, ...] | None = None
    try:
        with torch.no_grad():
            for start in range(0, partition.transition_count, partition.config.batch_size):
                stop = min(
                    start + partition.config.batch_size,
                    partition.transition_count,
                )
                observations, targets = partition.batch(
                    list(range(start, stop)), device=device
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
    if world_size > 1:
        torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
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
    return {
        "schema": HELDOUT_TELEMETRY_SCHEMA,
        "campaign_id": partition.config.campaign_id,
        "macro_round": partition.config.macro_round,
        "optimizer_step": optimizer_step,
        "heldout_snapshot_sha256": partition.snapshot["snapshot_sha256"],
        "heldout_snapshot_file_sha256": partition.snapshot["file_sha256"],
        "heldout_transitions_rank": partition.transition_count,
        "heldout_transitions_per_rank": list(partition.transition_counts_by_rank),
        "heldout_transitions_global": partition.global_transition_count,
        "world_size": world_size,
        "learned_action_indices": list(action_indices),
        "action_mse": mse,
        "action_mae": mae,
        "action_rmse": math.sqrt(mse),
        "action_cosine": cosine,
        "per_action": per_action,
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
        torch.distributed.all_reduce(scalar, op=torch.distributed.ReduceOp.SUM)
        scalar = scalar / world_size
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
        # RTX learner validation runs in the root-based Isaac image while the
        # host coordinator runs as the workspace user.  A mkstemp-derived 0600
        # restart is resumable by a later root container but cannot be hashed,
        # copied, or attested by that coordinator.  Restart state contains no
        # credentials; keep it owner-writable and make it read-only to group/
        # other so cross-container persistence remains verifiable.
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
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "completed_steps": completed_steps,
        # This is the eval-mode loss on the deterministic replay probe before
        # the first update.  Keeping it in the restart frontier prevents a
        # resumed run from replacing the comparable experiment baseline.
        "initial_loss": initial_loss,
        "world_size": world_size,
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
    learner_identity_sha256: str | None,
    runtime_container_sha256: str | None,
    learner_contract_sha256: str | None,
) -> tuple[int, float]:
    if path.is_symlink() or not path.is_file():
        raise LearnerError("restart checkpoint must be a regular non-symlink file")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise LearnerError(f"cannot load restart checkpoint: {exc}") from exc
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise LearnerError("restart checkpoint schema mismatch")
    expected = {
        "campaign_id": config.campaign_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "world_size": world_size,
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
    """Rank-zero durable JSONL telemetry with optional authenticated W&B logging."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None,
        *,
        enabled: bool,
        heldout_path: str | os.PathLike[str] | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_id: str | None = None,
        wandb_resume_must: bool = False,
        wandb_resume_never: bool = False,
        wandb_config: Mapping[str, Any] | None = None,
    ):
        self.path = (
            Path(os.path.abspath(Path(path).expanduser()))
            if path is not None and enabled
            else None
        )
        self.wandb_run = None
        if heldout_path is not None and not enabled:
            self.heldout_path = None
        elif heldout_path is not None:
            self.heldout_path = Path(
                os.path.abspath(Path(heldout_path).expanduser())
            )
        elif self.path is not None:
            self.heldout_path = self.path.with_name(
                f"{self.path.stem}.heldout{self.path.suffix}"
            )
        else:
            self.heldout_path = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.heldout_path is not None:
            self.heldout_path.parent.mkdir(parents=True, exist_ok=True)
        if enabled and wandb_project:
            if wandb_resume_must and wandb_resume_never:
                raise LearnerError(
                    "W&B resume='must' and resume='never' are mutually exclusive"
                )
            if wandb_resume_must and not wandb_run_id:
                raise LearnerError(
                    "strict W&B resume requires a durable prior run ID"
                )
            if wandb_resume_never and not wandb_run_id:
                raise LearnerError(
                    "W&B resume='never' requires a canonical run ID"
                )
            try:
                import wandb  # type: ignore

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    id=wandb_run_id,
                    resume=(
                        "must"
                        if wandb_resume_must
                        else (
                            "never"
                            if wandb_resume_never
                            else ("allow" if wandb_run_id else None)
                        )
                    ),
                    config=dict(wandb_config or {}),
                )
                # Bind every model-quality series to the actual optimizer
                # frontier.  Without this declaration W&B's generated panels
                # can collapse sparse held-out evaluations into one-value bar
                # summaries, which obscures whether the student improves.
                self.wandb_run.define_metric("optimizer_step")
                self.wandb_run.define_metric("loss", step_metric="optimizer_step")
                self.wandb_run.define_metric(
                    "quality/*", step_metric="optimizer_step"
                )
            except Exception as exc:
                raise LearnerError(f"W&B initialization failed: {exc}") from exc

    @staticmethod
    def _persist(path: Path | None, clean: Mapping[str, Any]) -> None:
        if path is None:
            return
        clean = dict(clean)
        payload = (
            json.dumps(
                clean, sort_keys=True, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        # A scheduler can deliver SIGKILL while telemetry is being persisted.
        # Appending directly to JSONL can then leave a torn final line,
        # permanently wedging completion attestation even though the last
        # durable restart is valid.  The production log is small (1200 scalar
        # rows), so rewrite it through a same-directory temporary and
        # atomically replace the old inode.  A crash exposes either the
        # complete previous log or the complete new log.
        if path.is_symlink():
            raise LearnerError("telemetry path must not be a symlink")
        existing = b""
        if path.exists():
            if not path.is_file():
                raise LearnerError("telemetry path must be a regular file")
            existing = path.read_bytes()
            if existing and not existing.endswith(b"\n"):
                raise LearnerError("telemetry log has a torn final record")
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(existing)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(
                path,
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IRGRP
                | stat.S_IROTH,
            )
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _heldout_wandb_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "optimizer_step": record["optimizer_step"],
            "quality/heldout_action_mse": record["action_mse"],
            "quality/heldout_action_mae": record["action_mae"],
            "quality/heldout_action_rmse": record["action_rmse"],
            "quality/heldout_action_cosine": record["action_cosine"],
            "quality/heldout_transitions": record["heldout_transitions_global"],
        }
        per_action = record.get("per_action")
        if not isinstance(per_action, Mapping):
            raise LearnerError("held-out telemetry per_action must be a mapping")
        for action_index, metrics in per_action.items():
            if not isinstance(metrics, Mapping):
                raise LearnerError("held-out per-action metrics must be mappings")
            for metric in ("mse", "mae", "rmse", "cosine"):
                payload[f"quality/action_{action_index}/{metric}"] = metrics[metric]
        return payload

    @staticmethod
    def _heldout_record_already_persisted(
        path: Path | None, record: Mapping[str, Any]
    ) -> bool:
        """Make replayed evaluation frontiers idempotent after restart."""

        if path is None or not path.exists():
            return False
        if path.is_symlink() or not path.is_file():
            raise LearnerError("held-out telemetry path is unsafe")
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LearnerError("held-out telemetry log has a torn final record")
        target_step = _true_int(record.get("optimizer_step"), "optimizer_step")
        target_bytes = json.dumps(
            dict(record),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        seen: dict[int, bytes] = {}
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                existing = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LearnerError(
                    f"held-out telemetry line {line_number} is invalid"
                ) from exc
            if (
                not isinstance(existing, Mapping)
                or existing.get("schema") != HELDOUT_TELEMETRY_SCHEMA
            ):
                raise LearnerError("held-out telemetry history schema mismatch")
            step = _true_int(
                existing.get("optimizer_step"),
                f"held-out telemetry line {line_number} optimizer_step",
            )
            canonical = json.dumps(
                dict(existing),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if step in seen and seen[step] != canonical:
                raise LearnerError(
                    "held-out telemetry contains conflicting duplicate steps"
                )
            seen[step] = canonical
        if target_step in seen:
            if seen[target_step] != target_bytes:
                raise LearnerError(
                    "held-out telemetry changed across deterministic restart"
                )
            return True
        if seen and target_step < max(seen):
            raise LearnerError(
                "held-out telemetry restart would create a missing historical step"
            )
        return False

    def log(
        self,
        record: Mapping[str, Any],
        *,
        wandb_extra: Mapping[str, Any] | None = None,
    ) -> None:
        clean = dict(record)
        self._persist(self.path, clean)
        if self.wandb_run is not None:
            wandb_record = dict(clean)
            if wandb_extra is not None:
                extension = dict(wandb_extra)
                duplicate = set(wandb_record) & set(extension)
                conflicting = {
                    key
                    for key in duplicate
                    if wandb_record[key] != extension[key]
                }
                if conflicting:
                    raise LearnerError(
                        "W&B telemetry extension conflicts with base fields: "
                        + ", ".join(sorted(conflicting))
                    )
                for key in duplicate:
                    extension.pop(key)
                wandb_record.update(extension)
            # W&B's internal step remains append-only across restarts.  The
            # explicit optimizer_step field is the declared custom x-axis, so
            # replaying an equal/older durable frontier cannot be discarded as
            # an out-of-order global-step update.
            self.wandb_run.log(wandb_record)

    def log_heldout(
        self, record: Mapping[str, Any], *, log_wandb: bool = True
    ) -> dict[str, Any]:
        clean = dict(record)
        if clean.get("schema") != HELDOUT_TELEMETRY_SCHEMA:
            raise LearnerError("held-out telemetry schema mismatch")
        already_persisted = self._heldout_record_already_persisted(
            self.heldout_path, clean
        )
        if not already_persisted:
            self._persist(self.heldout_path, clean)
        wandb_payload = self._heldout_wandb_payload(clean)
        # Local JSONL idempotence and remote delivery are separate frontiers.
        # A process can be killed after the fsynced record is durable but
        # before the W&B sender acknowledges it.  Re-log an identical due
        # frontier on restart; the explicit optimizer step makes the update
        # deterministic, while the local history still contains exactly one
        # canonical row per step.
        if log_wandb and self.wandb_run is not None:
            self.wandb_run.log(wandb_payload)
        return wandb_payload

    def close(self, *, exit_code: int = 0) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)


@dataclass(frozen=True)
class TrainResult:
    completed_steps: int
    target_steps: int
    validation_stopped: bool
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
    max_steps: int,
    adapter: PolicyAdapter | None = None,
    device: torch.device | str = "cpu",
    resume_from: str | os.PathLike[str] | None = None,
    expected_snapshot_file_sha256: str | None = None,
    heldout_snapshot_path: str | os.PathLike[str] | None = None,
    save_every: int = 100,
    telemetry: JsonlTelemetry | None = None,
    validation_stop_at_step: int | None = None,
    learner_identity_sha256: str | None = None,
    runtime_container_sha256: str | None = None,
    learner_contract_sha256: str | None = None,
) -> TrainResult:
    """Run supervised DAgger updates over one immutable cumulative snapshot."""

    max_steps = _true_int(max_steps, "max_steps", minimum=1)
    save_every = _true_int(save_every, "save_every", minimum=1)
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
    if validation_stop_at_step is not None:
        validation_stop_at_step = _true_int(
            validation_stop_at_step, "validation_stop_at_step", minimum=1
        )
        if validation_stop_at_step >= max_steps:
            raise LearnerError(
                "validation_stop_at_step must be below the final max_steps frontier"
            )
    rank, world_size = _distributed_context()
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
    )
    if config.heldout_evaluation_enabled != (heldout_snapshot_path is not None):
        raise LearnerError(
            "held-out snapshot path and immutable bundle configuration must "
            "be supplied together"
        )
    heldout_partition = None
    if heldout_snapshot_path is not None:
        heldout_partition = load_heldout_partition(
            heldout_snapshot_path,
            config,
            rank=rank,
            world_size=world_size,
        )
        _assert_training_and_heldout_are_disjoint(
            partition.snapshot, heldout_partition.snapshot
        )
    model = model.to(device)
    if world_size > 1 and not isinstance(model, DistributedDataParallel):
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
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
            learner_identity_sha256=learner_identity_sha256,
            runtime_container_sha256=runtime_container_sha256,
            learner_contract_sha256=learner_contract_sha256,
        )
    if completed_steps > max_steps:
        raise LearnerError("restart completed_steps exceeds requested max_steps")
    if (
        validation_stop_at_step is not None
        and completed_steps >= validation_stop_at_step
    ):
        raise LearnerError(
            "validation_stop_at_step must be beyond the restart completed_steps"
        )
    training_frontier = (
        max_steps
        if validation_stop_at_step is None
        else validation_stop_at_step
    )
    saved_at_step = (
        completed_steps
        if resume_from is not None
        and Path(resume_from).expanduser().resolve() == restart_path
        else -1
    )
    schedule = DeterministicBatchSchedule(
        partition.transition_count, config.batch_size, seed=config.seed
    )
    adapter = adapter or MappingPolicyAdapter(config.output_key)
    probe_schedule = DeterministicBatchSchedule(
        partition.transition_count,
        config.batch_size,
        seed=_probe_seed(config.seed, rank),
    )
    probe_indices = probe_schedule.indices(0)
    if first_loss is None:
        first_loss = _evaluate_loss_probe(
            model,
            partition=partition,
            indices=probe_indices,
            adapter=adapter,
            device=device,
            train_action_slice=config.train_action_slice,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
        )
    model.train()
    # Re-evaluate a due restart frontier before taking another update.  This
    # closes the crash window in which the checkpoint and local quality row are
    # durable but the remote telemetry sender has not synchronized yet.  The
    # JSONL writer verifies that a repeated frontier is byte-identical.
    heldout_frontier_due = (
        heldout_partition is not None
        and config.heldout_eval_interval is not None
        and (
            completed_steps == 0
            or completed_steps % config.heldout_eval_interval == 0
            or completed_steps == max_steps
        )
    )
    if heldout_frontier_due:
        assert heldout_partition is not None
        heldout_record = evaluate_heldout_actions(
            model,
            partition=heldout_partition,
            adapter=adapter,
            device=device,
            optimizer_step=completed_steps,
        )
        if rank == 0 and telemetry is not None:
            telemetry.log_heldout(heldout_record, log_wandb=True)
    for step in range(completed_steps, training_frontier):
        observations, targets = partition.batch(schedule.indices(step), device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions = adapter.actions(model, observations)
        loss = supervised_action_loss(predictions, targets, config.train_action_slice)
        if not torch.isfinite(loss):
            raise LearnerError(f"non-finite supervised loss at optimizer step {step}")
        loss.backward()
        optimizer.step()
        scalar = loss.detach()
        if world_size > 1:
            torch.distributed.all_reduce(scalar, op=torch.distributed.ReduceOp.SUM)
            scalar = scalar / world_size
        loss_value = float(scalar.cpu())
        completed_steps = step + 1
        heldout_wandb: Mapping[str, Any] | None = None
        heldout_due = (
            heldout_partition is not None
            and config.heldout_eval_interval is not None
            and (
                completed_steps % config.heldout_eval_interval == 0
                or completed_steps == max_steps
            )
        )
        if heldout_due:
            assert heldout_partition is not None
            heldout_record = evaluate_heldout_actions(
                model,
                partition=heldout_partition,
                adapter=adapter,
                device=device,
                optimizer_step=completed_steps,
            )
            if rank == 0 and telemetry is not None:
                heldout_wandb = telemetry.log_heldout(
                    heldout_record, log_wandb=False
                )
        if rank == 0 and telemetry is not None:
            telemetry.log(
                {
                    "schema": TELEMETRY_SCHEMA,
                    "campaign_id": config.campaign_id,
                    "macro_round": config.macro_round,
                    "optimizer_step": completed_steps,
                    "loss": loss_value,
                    "replay_transitions_rank": partition.transition_count,
                    "replay_transitions_per_rank": list(
                        partition.transition_counts_by_rank
                    ),
                    "replay_transitions_global": partition.global_transition_count,
                    "world_size": world_size,
                },
                wandb_extra=heldout_wandb,
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
        indices=probe_indices,
        adapter=adapter,
        device=device,
        train_action_slice=config.train_action_slice,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )
    return TrainResult(
        completed_steps=completed_steps,
        target_steps=max_steps,
        validation_stopped=validation_stop_at_step is not None,
        initial_loss=first_loss,
        final_loss=final_loss,
        restart_checkpoint=str(restart_path),
        replay_snapshot_sha256=partition.snapshot["snapshot_sha256"],
        learner_config_sha256=config.sha256,
    )


def _publish_immutable_binary(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise LearnerError("candidate source checkpoint is missing or unsafe")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != expected:
            raise LearnerError(
                "immutable candidate path already contains different bytes"
            )
        return expected
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if sha256_file(temporary) != expected:
            raise LearnerError("candidate checkpoint copy digest mismatch")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256_file(destination) != expected:
                raise LearnerError("competing candidate publication changed bytes")
        os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return expected


def publish_learner_result(
    *,
    output_dir: str | os.PathLike[str],
    result: TrainResult,
    config: LearnerBundleConfig,
) -> dict[str, Any]:
    """Seal a learner payload and a manifest consumable by policy staging."""

    if result.validation_stopped or result.completed_steps != result.target_steps:
        raise LearnerError(
            "validation-stopped or incomplete learner result cannot be published"
        )

    output = Path(output_dir).expanduser().resolve()
    checkpoint_name = f"candidate_round_{config.macro_round:06d}.pt"
    checkpoint_path = output / checkpoint_name
    checkpoint_sha = _publish_immutable_binary(
        Path(result.restart_checkpoint), checkpoint_path
    )
    try:
        checkpoint_document = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_document, Mapping):
            raise LearnerError("candidate checkpoint root must be a mapping")
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
        KeyError,
        TypeError,
    ) as exc:
        raise LearnerError(f"cannot publish candidate policy sidecars: {exc}") from exc

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
        "schema": RESULT_SCHEMA,
        "campaign_id": config.campaign_id,
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
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
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
    "RESULT_SCHEMA",
    "HELDOUT_TELEMETRY_SCHEMA",
    "LearnerError",
    "LearnerBundleConfig",
    "PolicyAdapter",
    "MappingPolicyAdapter",
    "ReplayPartition",
    "DeterministicBatchSchedule",
    "JsonlTelemetry",
    "TrainResult",
    "partition_shards",
    "load_replay_partition",
    "load_heldout_partition",
    "supervised_action_loss",
    "evaluate_heldout_actions",
    "CUDA_CUBLAS_WORKSPACE_CONFIG",
    "configure_deterministic_runtime",
    "train_from_replay",
    "publish_learner_result",
]
