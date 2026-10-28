"""Opt-in, inference-only action-MSE validation for fixed held-out windows.

This module does not activate the existing production controller. A caller must
seal validation provenance separately, compare it with every training window,
and bind the resulting manifest digest for the entire round. Closed-loop success
tests are deliberately not accepted as checkpoint-selection input.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from pipeline.bc_training.train_bc_h200 import EarlyStopping
from pipeline.dagger.services.contracts import file_sha256, sha256_digest


MANIFEST_SCHEMA = "roborender.dagger-heldout-mse.v1"
SCOPED_MANIFEST_SCHEMA = "roborender.dagger-heldout-mse.v2"
RESULT_SCHEMA = "roborender.dagger-heldout-mse-result.v1"
STATE_SCHEMA = "roborender.dagger-mse-selection.v1"
VIEWS = ("head_color", "hand_left", "hand_right")
RESULT_KEYS = {
    "schema", "purpose", "round_id", "validation_manifest_sha256",
    "checkpoint_sha256", "mse", "per_action_mse", "frames", "elements",
    "chunk_count", "inference_only", "selection_metric", "test_payloads_opened",
}


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_window_identity(metadata: dict) -> str:
    """Window indices repeat across actors/runs; identity is the full scope."""
    keys = ("run_id", "round_id", "actor_id", "rollout_window_id")
    if any(key not in metadata for key in keys):
        raise ValueError("source window lacks run/round/actor identity")
    return "window-sha256:" + canonical_digest({key: metadata[key] for key in keys})


def source_episode_identity(metadata: dict, episode_id: str) -> str:
    """Native episode counters are local to a run/round/actor, not global."""
    keys = ("run_id", "round_id", "actor_id")
    if (any(key not in metadata for key in keys) or not isinstance(episode_id, str)
            or not episode_id or type(metadata["round_id"]) is not int
            or any(not isinstance(metadata[key], str) or not metadata[key] for key in ("run_id", "actor_id"))):
        raise ValueError("source episode lacks run/round/actor/raw identity")
    return "episode-sha256:" + canonical_digest({**{key: metadata[key] for key in keys}, "episode_id": episode_id})


def _identity_set(values, name: str, *, allow_empty: bool = False) -> set[str]:
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{name} must be an identity list")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} contains an invalid identity")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate identities")
    return set(values)


def load_validation_manifest(path: Path, expected_sha256: str,
                             training_provenance: dict) -> dict:
    sha256_digest(expected_sha256, "validation_manifest_sha256")
    if file_sha256(path) != expected_sha256:
        raise ValueError("validation manifest digest changed")
    manifest = json.loads(path.read_bytes())
    required = {"schema", "purpose", "round_id", "policy_spec_sha256",
                "teacher_checkpoint_sha256", "view_order", "chunks"}
    if set(manifest) != required or manifest["schema"] not in {MANIFEST_SCHEMA, SCOPED_MANIFEST_SCHEMA}:
        raise ValueError("invalid held-out validation manifest schema/fields")
    if manifest["purpose"] != "validation":
        raise ValueError("only validation data may select a checkpoint; test data is forbidden")
    if type(manifest["round_id"]) is not int or manifest["round_id"] not in (1, 2, 3, 4):
        raise ValueError("validation round must be 1..4")
    if manifest["view_order"] != list(VIEWS):
        raise ValueError("validation requires the three-view policy order")
    for key in ("policy_spec_sha256", "teacher_checkpoint_sha256"):
        sha256_digest(manifest[key], key)
    if (set(training_provenance) != {"purpose", "round_id", "chunk_sha256s",
                                   "source_episode_ids", "source_window_ids"}
            or training_provenance["purpose"] != "train"
            or training_provenance["round_id"] != manifest["round_id"]):
        raise ValueError("complete matching-round training provenance is required")
    training = {name: _identity_set(training_provenance[name], name, allow_empty=True)
                for name in ("chunk_sha256s", "source_episode_ids", "source_window_ids")}
    if training["chunk_sha256s"] and (not training["source_episode_ids"] or not training["source_window_ids"]):
        raise ValueError("training chunk provenance lacks source episode/window identities")
    for digest in training["chunk_sha256s"]:
        sha256_digest(digest, "training chunk sha256")
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("validation manifest has no chunks")
    chunk_ids, hashes, paths = set(), set(), set()
    for row in chunks:
        scoped = manifest["schema"] == SCOPED_MANIFEST_SCHEMA
        fields = {"chunk_id", "path", "sha256", "size_bytes", "frames",
                  "source_episode_ids", "source_window_ids", "observation_sources"}
        if set(row) != fields | ({"source_scope", "raw_source_episode_ids"} if scoped else set()):
            raise ValueError("invalid validation chunk fields")
        if scoped:
            scope = row["source_scope"]
            if not isinstance(scope, dict) or set(scope) != {"run_id", "round_id", "actor_id"} or scope["round_id"] != manifest["round_id"]:
                raise ValueError("invalid validation episode source scope")
            raw_ids = _identity_set(row["raw_source_episode_ids"], "raw_source_episode_ids")
            if row["source_episode_ids"] != sorted(source_episode_identity(scope, raw) for raw in raw_ids):
                raise ValueError("scoped episode identities disagree with immutable raw IDs")
        if not isinstance(row["chunk_id"], str) or not row["chunk_id"]:
            raise ValueError("invalid validation chunk identity")
        sha256_digest(row["sha256"], "validation chunk sha256")
        chunk_path = Path(row["path"])
        if not chunk_path.is_absolute():
            raise ValueError("validation chunk paths must be absolute")
        chunk_path = chunk_path.resolve(strict=True)
        if row["chunk_id"] in chunk_ids or row["sha256"] in hashes or chunk_path in paths:
            raise ValueError("validation chunks are duplicated")
        chunk_ids.add(row["chunk_id"])
        hashes.add(row["sha256"])
        paths.add(chunk_path)
        if row["sha256"] in training["chunk_sha256s"]:
            raise ValueError("training and validation chunk content overlaps")
        for key in ("source_episode_ids", "source_window_ids"):
            if _identity_set(row[key], key) & training[key]:
                raise ValueError(f"training and validation {key} overlap")
        if not _identity_set(row["observation_sources"], "observation_sources") <= {
            "raytraced", "generated_controls",
        }:
            raise ValueError("unknown validation observation source")
        for key in ("size_bytes", "frames"):
            if type(row[key]) is not int or row[key] < 1:
                raise ValueError(f"validation chunk {key} must be positive")
        if not chunk_path.is_file() or chunk_path.stat().st_size != row["size_bytes"]:
            raise ValueError("validation chunk size differs from manifest")
    return manifest


def _load_chunk(row: dict, spec: dict, device: torch.device):
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise ValueError("validation chunk content changed")
    frames = row["frames"]
    shape = tuple(spec["input_contract"]["vision_shape"])
    if len(shape) != 3 or shape[0] % 3 or shape[2] != 3:
        raise ValueError("invalid three-view policy image shape")
    view_shape = (frames, shape[0] // 3, shape[1], 3)
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    with np.load(path, allow_pickle=False) as data:
        views = []
        for view in VIEWS:
            value = data[f"rgb_gen__{view}"]
            if value.shape != view_shape:
                raise ValueError("validation RGB shape differs from policy contract")
            if value.dtype == np.uint8:
                value = value.astype(np.float32) / 255.0
            elif value.dtype != np.float32:
                raise ValueError("validation RGB must be uint8 or float32")
            if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
                raise ValueError("validation RGB must be finite in [0,1]")
            views.append(value)
        actor = data["obs_actor_obs"]
        actions = data["gt_actions"]
        if actor.shape != (frames, actor_dim) or actions.ndim != 2 or actions.shape[0] != frames:
            raise ValueError("validation state/action shapes are not synchronized")
        if actions.shape[1] < action_dim:
            raise ValueError("validation teacher action dimension is too short")
        target = actions[:, :action_dim]
        if not np.isfinite(actor).all() or not np.isfinite(target).all():
            raise ValueError("validation state/teacher actions must be finite")
        # Lean learner chunks deliberately omit provenance arrays; their sealed
        # provenance must come from the immutable coordinator/collection ledger.
        if "episode_ids" in data and not set(map(str, data["episode_ids"])) <= set(row.get("raw_source_episode_ids", row["source_episode_ids"])):
            raise ValueError("payload episode identities disagree with manifest")
        vision = torch.from_numpy(np.concatenate(views, axis=1)).to(device).unsqueeze(0)
        actor_tensor = torch.from_numpy(actor.copy()).to(device=device, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.from_numpy(target.copy()).to(device=device, dtype=torch.float32).unsqueeze(0)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("invalid policy RGB normalization")
    return (vision - mean) / std, actor_tensor, target_tensor


@contextmanager
def _preserve_inference_state(policy, device):
    modes = [(module, module.training) for module in policy.modules()]
    versions = [(name, tensor, tensor._version) for name, tensor in policy.state_dict().items()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(0)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(0)
        try:
            policy.eval()
            with torch.inference_mode():
                yield
            if any(tensor._version != version for _, tensor, version in versions):
                raise RuntimeError("validation modified model parameters or buffers")
        finally:
            for module, training in modes:
                module.training = training


def evaluate_policy(policy, spec: dict, manifest: dict, *, checkpoint_sha256: str,
                    manifest_sha256: str, device: torch.device) -> dict:
    """Forward pass only. Caller first admits manifest and constructs pinned policy."""
    if manifest["purpose"] != "validation" or manifest["schema"] not in {MANIFEST_SCHEMA, SCOPED_MANIFEST_SCHEMA}:
        raise ValueError("only held-out validation may enter MSE selection")
    if spec["input_contract"]["views"] != list(VIEWS) or canonical_digest(spec) != manifest["policy_spec_sha256"]:
        raise ValueError("validation policy specification differs from the fixed manifest")
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    if action_dim < 1:
        raise ValueError("policy action dimension must be positive")
    squared = torch.zeros(action_dim, dtype=torch.float64, device=device)
    frames = 0
    with _preserve_inference_state(policy, device):
        for row in manifest["chunks"]:
            vision, actor, target = _load_chunk(row, spec, device)
            prediction = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
            if prediction.shape != target.shape:
                raise ValueError("validation prediction shape differs from teacher labels")
            error = (prediction.double() - target.double()).square()
            if not bool(torch.isfinite(error).all()):
                raise ValueError("non-finite validation action error")
            squared += error.sum(dim=(0, 1))
            frames += target.shape[0] * target.shape[1]
    if not frames:
        raise ValueError("validation has no frames")
    per_action = (squared / frames).cpu().tolist()
    if not all(math.isfinite(value) for value in per_action):
        raise ValueError("non-finite aggregate validation MSE")
    return {"schema": RESULT_SCHEMA, "purpose": "validation", "round_id": manifest["round_id"],
            "validation_manifest_sha256": manifest_sha256, "checkpoint_sha256": checkpoint_sha256,
            "mse": sum(per_action) / action_dim, "per_action_mse": per_action,
            "frames": frames, "elements": frames * action_dim,
            "chunk_count": len(manifest["chunks"]), "inference_only": True,
            "selection_metric": "action_mse", "test_payloads_opened": False}


def evaluate_checkpoint(checkpoint_path: Path, checkpoint_sha256: str, manifest_path: Path,
                        manifest_sha256: str, training_provenance: dict, *,
                        device: torch.device, policy_factory=None) -> dict:
    """Construct a separate checkpoint-pinned model, never a live learner model."""
    manifest = load_validation_manifest(manifest_path, manifest_sha256, training_provenance)
    sha256_digest(checkpoint_sha256, "checkpoint_sha256")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("validation checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if policy_factory is None:
        from pipeline.dagger.services.raven_learner_worker import build_policy
        policy_factory = build_policy
    # Construction can initialize temporary random weights before strict loading.
    # Preserve the control process's RNG as well as the inference loop's RNG.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        policy = policy_factory(checkpoint["policy_spec"], checkpoint["policy_state_dict"], device)
        return evaluate_policy(policy, checkpoint["policy_spec"], manifest,
                               checkpoint_sha256=checkpoint_sha256,
                               manifest_sha256=manifest_sha256, device=device)


@dataclass
class RoundMSESelection:
    """Pure round-bound stopping state; the controller owns persistence/activation."""
    round_id: int
    validation_manifest_sha256: str
    stopper: EarlyStopping = field(default_factory=EarlyStopping)
    history: list[dict] = field(default_factory=list)
    best_checkpoint_sha256: str | None = None

    def __post_init__(self):
        if type(self.round_id) is not int or self.round_id not in (1, 2, 3, 4):
            raise ValueError("selection round must be 1..4")
        sha256_digest(self.validation_manifest_sha256, "validation_manifest_sha256")

    def record(self, iteration: int, result: dict) -> dict:
        if set(result) != RESULT_KEYS or result["schema"] != RESULT_SCHEMA:
            raise ValueError("selection accepts only complete held-out MSE receipts")
        if (result["purpose"] != "validation" or result["selection_metric"] != "action_mse"
                or result["inference_only"] is not True or result["test_payloads_opened"] is not False):
            raise ValueError("test/success/training metrics cannot select checkpoints")
        if (result["round_id"] != self.round_id
                or result["validation_manifest_sha256"] != self.validation_manifest_sha256):
            raise ValueError("validation identity changed during round selection")
        sha256_digest(result["checkpoint_sha256"], "checkpoint_sha256")
        values = result["per_action_mse"]
        if not isinstance(values, list) or not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid per-action validation MSE")
        if not math.isfinite(result["mse"]) or not math.isclose(result["mse"], sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("aggregate MSE disagrees with per-action values")
        if (type(result["frames"]) is not int or result["frames"] < 1
                or result["elements"] != result["frames"] * len(values)
                or type(result["chunk_count"]) is not int or result["chunk_count"] < 1):
            raise ValueError("invalid validation population counts")
        if type(iteration) is not int or iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if self.history and iteration <= self.history[-1]["iteration"]:
            previous = next((row for row in self.history if row["iteration"] == iteration), None)
            if previous is not None and previous["result"] == result:
                return deepcopy(previous)
            raise ValueError("changed or out-of-order validation receipt")
        if not self.history and iteration != 0:
            raise ValueError("round validation requires an iteration-zero baseline")
        if self.history and self.stopper.is_saturated(self.stopper.last_epoch):
            raise ValueError("round MSE patience is already saturated")
        if self.history:
            baseline = self.history[0]["result"]
            if any(result[key] != baseline[key] for key in ("frames", "elements", "chunk_count")):
                raise ValueError("validation population changed during round")
        index = len(self.history)
        meaningful, stop = self.stopper.update(index, float(result["mse"]))
        absolute_best = self.stopper.best_epoch == index
        if absolute_best:
            self.best_checkpoint_sha256 = result["checkpoint_sha256"]
        record = {"iteration": iteration, "result": deepcopy(result), "absolute_best": absolute_best,
                  "meaningful_improvement": meaningful, "saturated": stop}
        self.history.append(record)
        return deepcopy(record)

    def state_dict(self) -> dict:
        return {"schema": STATE_SCHEMA, "round_id": self.round_id,
                "validation_manifest_sha256": self.validation_manifest_sha256,
                "early_stopping_state": self.stopper.state_dict(),
                "best_checkpoint_sha256": self.best_checkpoint_sha256,
                "history": deepcopy(self.history)}

    @classmethod
    def restore(cls, state: dict):
        if set(state) != {"schema", "round_id", "validation_manifest_sha256",
                          "early_stopping_state", "best_checkpoint_sha256", "history"} or state["schema"] != STATE_SCHEMA:
            raise ValueError("invalid round MSE selection state")
        saved = state["early_stopping_state"]
        stopper = EarlyStopping(**{key: saved[key] for key in (
            "min_epochs", "patience", "relative_improvement", "absolute_improvement",
        )})
        instance = cls(state["round_id"], state["validation_manifest_sha256"], stopper)
        for row in state["history"]:
            instance.record(row["iteration"], row["result"])
        if instance.state_dict() != state:
            raise ValueError("persisted round selection state disagrees with its receipts")
        return instance
