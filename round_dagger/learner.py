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
from round_dagger.replay import deterministic_partition, load_snapshot


BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
CHECKPOINT_SCHEMA = "fast_roborender.round_dagger.learner_checkpoint.v1"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
CUDA_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


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
    max_steps: int = 500
    save_every: int = 100
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


def partition_shards(
    resolved_shards: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Assign immutable shards to ranks without Python hash randomization."""

    rank = _true_int(rank, "rank")
    world_size = _true_int(world_size, "world_size", minimum=1)
    seed = _true_int(seed, "partition seed")
    if rank >= world_size:
        raise LearnerError("rank must be smaller than world_size")
    selected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for raw in resolved_shards:
        if not isinstance(raw, Mapping):
            raise LearnerError("resolved shard must be a mapping")
        identity = f"{raw.get('logical_id')}/{raw.get('shard_id')}"
        if identity in identities:
            raise LearnerError("resolved replay contains duplicate shard identity")
        identities.add(identity)
        if deterministic_partition(identity, world_size, seed=seed) == rank:
            selected.append(dict(raw))
    selected.sort(
        key=lambda item: (item["macro_round"], item["cohort_index"], item["shard_id"])
    )
    return selected


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
    assignments = [
        partition_shards(
            snapshot["resolved_shards"],
            rank=candidate_rank,
            world_size=world_size,
            seed=config.partition_seed,
        )
        for candidate_rank in range(world_size)
    ]
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


def supervised_action_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    train_action_slice: tuple[int, int] | None,
) -> torch.Tensor:
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
    return torch.mean(torch.square(predictions.float() - targets.float()))


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
) -> dict[str, Any]:
    policy = _unwrap_model(model)
    policy_state = policy.state_dict()
    return {
        "schema": CHECKPOINT_SCHEMA,
        # These two fields make the promoted payload directly consumable by
        # the existing GR00T actor checkpoint loader, while ``schema`` and the
        # lineage fields below retain the stronger round-learner contract.
        "checkpoint_schema": "gr00t.trl.full_state.v2",
        "policy_state_dict": policy_state,
        "policy_parameter_names": sorted(name for name, _ in policy.named_parameters()),
        "campaign_id": config.campaign_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "completed_steps": completed_steps,
        # This is the loss observed at optimizer step zero, before the first
        # update.  Keeping it in the restart frontier prevents a resumed run
        # from silently replacing the experiment baseline with the first loss
        # seen after the interruption.
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
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "world_size": world_size,
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
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_id: str | None = None,
        wandb_resume_must: bool = False,
        wandb_config: Mapping[str, Any] | None = None,
    ):
        self.path = (
            Path(path).expanduser().resolve() if path is not None and enabled else None
        )
        self.wandb_run = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if enabled and wandb_project:
            if wandb_resume_must and not wandb_run_id:
                raise LearnerError(
                    "strict W&B resume requires a durable prior run ID"
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
                        else ("allow" if wandb_run_id else None)
                    ),
                    config=dict(wandb_config or {}),
                )
            except Exception as exc:
                raise LearnerError(f"W&B initialization failed: {exc}") from exc

    def log(self, record: Mapping[str, Any]) -> None:
        clean = dict(record)
        payload = json.dumps(
            clean, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if self.wandb_run is not None:
            step = int(clean.get("optimizer_step", 0))
            self.wandb_run.log(clean, step=step)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


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
    save_every: int = 100,
    telemetry: JsonlTelemetry | None = None,
    validation_stop_at_step: int | None = None,
) -> TrainResult:
    """Run supervised DAgger updates over one immutable cumulative snapshot."""

    max_steps = _true_int(max_steps, "max_steps", minimum=1)
    save_every = _true_int(save_every, "save_every", minimum=1)
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
    final_loss: float | None = None
    model.train()
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
        if first_loss is None:
            first_loss = loss_value
        final_loss = loss_value
        completed_steps = step + 1
        if rank == 0 and telemetry is not None:
            telemetry.log(
                {
                    "schema": "fast_roborender.round_dagger.learner_telemetry.v1",
                    "campaign_id": config.campaign_id,
                    "macro_round": config.macro_round,
                    "optimizer_step": completed_steps,
                    "loss": loss_value,
                    "replay_transitions_rank": partition.transition_count,
                    "world_size": world_size,
                }
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
                ),
            )
        if world_size > 1:
            torch.distributed.barrier()

    # Report a deterministic post-update loss.  This is deliberately an eval
    # pass and is not serialized into the restart RNG frontier.
    model.eval()
    observations, targets = partition.batch(
        schedule.indices(max(0, completed_steps - 1)), device=device
    )
    with torch.no_grad():
        evaluated = supervised_action_loss(
            adapter.actions(model, observations), targets, config.train_action_slice
        )
    if world_size > 1:
        torch.distributed.all_reduce(evaluated, op=torch.distributed.ReduceOp.SUM)
        evaluated = evaluated / world_size
    final_loss = float(evaluated.cpu())
    if first_loss is None:
        first_loss = final_loss
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
        runtime_policy_sha = runtime_state_dict_sha256(policy_state)
        policy_spec = config.to_mapping()["model_config"]
        policy_spec = validate_policy_spec(policy_spec)
        policy_spec["checkpoint_sha256"] = checkpoint_sha
        policy_spec["runtime_policy_sha256"] = runtime_policy_sha
        policy_spec = validate_policy_spec(policy_spec)
    except (ActorExportError, GR00TPolicyError, KeyError, TypeError) as exc:
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
    "supervised_action_loss",
    "CUDA_CUBLAS_WORKSPACE_CONFIG",
    "configure_deterministic_runtime",
    "train_from_replay",
    "publish_learner_result",
]
