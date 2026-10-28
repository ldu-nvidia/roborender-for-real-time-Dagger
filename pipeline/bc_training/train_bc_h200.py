#!/usr/bin/env python3
"""Native H200 offline BC training; intentionally has no Isaac/renderer imports."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.services.raven_learner_worker import (
    _batch_norm_recompute_context,
    backward_global_mse_in_environment_microbatches,
    install_index_safe_max_pool2d,
)
from pipeline.bc_training.stopping_gate import (
    should_stop_for_plateau, training_target_met,
    fitting_stop_contract as build_fitting_stop_contract,
    validate_fitting_stop_resume, epoch_sequence,
    fixed_final_contract, validate_fixed_final_resume,
)
from pipeline.bc_training import cyclic_sampling


VIEWS = ("head_color", "hand_left", "hand_right")
RAVEN_TRAIN_CHUNKS_PER_UPDATE = 16
FIXED_FINAL_TRAIN_CHUNKS_PER_UPDATE = 32
RAVEN_CHUNK_FRAMES = 81
RAVEN_BATCHING_CONTRACT = "raven_full_bxt_16x81_activation_checkpoint_v1"
DIRECT_STUDENT_LIFECYCLE = "direct_supervised_v1"
REPLAY_STUDENT_LIFECYCLE = "raven_offline_replay_v1"
RAVEN_REPLAY_BATCHING_CONTRACT = "raven_offline_replay_16x81_activation_checkpoint_v2"
RAVEN_REPLAY_B32_BATCHING_CONTRACT = "raven_offline_replay_32x81_activation_checkpoint_v1"
LEARNER_KEYS = tuple(f"rgb_gen__{view}" for view in VIEWS) + (
    "obs_actor_obs",
    "gt_actions",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_parent(path)


def fsync_parent(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass
class EarlyStopping:
    min_epochs: int = 10
    patience: int = 8
    relative_improvement: float = 0.005
    absolute_improvement: float = 1e-6
    best_mse: float = math.inf
    bad_validations: int = 0
    best_epoch: int = -1
    patience_best_mse: float = math.inf
    last_epoch: int = -1

    def __post_init__(self) -> None:
        if self.min_epochs < 1 or self.patience < 1:
            raise ValueError("minimum epochs and patience must be positive")
        if any(not math.isfinite(value) or value < 0 for value in (
            self.relative_improvement, self.absolute_improvement,
        )):
            raise ValueError("early-stopping improvement thresholds must be finite and nonnegative")

    def state_dict(self) -> dict:
        if self.last_epoch < 0:
            raise ValueError("early-stopping state has no validation observation")
        return {"schema": "roborender.bc-early-stopping.v2", **asdict(self)}

    def is_saturated(self, epoch: int) -> bool:
        return epoch >= self.min_epochs and self.bad_validations >= self.patience

    def update(self, epoch: int, mse: float) -> tuple[bool, bool]:
        """Return meaningful improvement/patience stop; always retain absolute best.

        Small decreases still select the best checkpoint. Patience compares with
        the last meaningful decrease, allowing cumulative small gains to reset it.
        """
        if epoch < 0 or epoch <= self.last_epoch:
            raise ValueError("validation epochs must be nonnegative and strictly increasing")
        if not math.isfinite(mse) or mse < 0:
            raise ValueError(f"validation MSE is not finite and nonnegative: {mse}")
        threshold = (
            0.0
            if not math.isfinite(self.patience_best_mse)
            else max(self.absolute_improvement, self.relative_improvement * self.patience_best_mse)
        )
        improved = (not math.isfinite(self.patience_best_mse)
                    or mse < self.patience_best_mse - threshold)
        if mse < self.best_mse:
            self.best_mse = mse
            self.best_epoch = epoch
        if improved:
            self.patience_best_mse = mse
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        self.last_epoch = epoch
        return improved, self.is_saturated(epoch)


def load_split(path: Path, expected: str, dataset_root: Path) -> tuple[dict, list[Path]]:
    payload = json.loads(path.read_text())
    if payload.get("split") != expected:
        raise ValueError(f"{path}: expected split={expected!r}")
    items = payload.get("items", [])
    if payload.get("chunk_count") != len(items) or not items:
        raise ValueError(f"{path}: invalid or empty chunk index")
    paths = [(dataset_root / item["rendered_npz"]).resolve() for item in items]
    if len(set(paths)) != len(paths) or len({item["chunk_id"] for item in items}) != len(items):
        raise ValueError(f"{path}: duplicate rendered chunks or chunk identities")
    if any(not chunk.is_relative_to(dataset_root.resolve()) for chunk in paths):
        raise ValueError(f"{path}: rendered chunk escapes dataset root")
    missing = [str(chunk) for chunk in paths if not chunk.is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing rendered chunks, first={missing[0]}")
    source_manifest = (path.parent / payload["source_manifest"]).resolve()
    if sha256_file(source_manifest) != payload["source_manifest_sha256"]:
        raise ValueError(f"{path}: source manifest hash mismatch")
    return payload, paths


def validate_split_pair(train_index: dict, train_paths: list[Path],
                        validation_index: dict, validation_paths: list[Path]) -> None:
    """Metadata-only independence gate; rendering-lineage admission is additional."""
    if train_index["source_manifest_sha256"] != validation_index["source_manifest_sha256"]:
        raise ValueError("training and validation source manifest identities differ")
    if set(train_paths) & set(validation_paths):
        raise ValueError("training and validation rendered paths overlap")
    for field in ("chunk_id", "stream_id"):
        if ({item[field] for item in train_index["items"]}
                & {item[field] for item in validation_index["items"]}):
            raise ValueError(f"training and validation {field} identities overlap")


class BCChunks(Dataset):
    """One item is Raven's complete synchronized 81-frame chunk."""

    def __init__(self, paths: list[Path], actor_dim: int, action_dim: int,
                 *, include_object_target: bool = False):
        self.paths = paths
        self.actor_dim = actor_dim
        self.action_dim = action_dim
        self.include_object_target = include_object_target

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as chunk:
            missing = [key for key in LEARNER_KEYS if key not in chunk.files]
            if missing:
                raise ValueError(f"{path}: missing learner fields {missing}")
            views = [chunk[f"rgb_gen__{view}"] for view in VIEWS]
            if any(view.dtype != np.uint8 for view in views):
                raise ValueError(f"{path}: rendered RGB must be uint8")
            lengths = {array.shape[0] for array in views}
            lengths |= {chunk["obs_actor_obs"].shape[0], chunk["gt_actions"].shape[0]}
            if len(lengths) != 1:
                raise ValueError(f"{path}: unsynchronized learner arrays {sorted(lengths)}")
            vision = np.concatenate(views, axis=1)
            actor = chunk["obs_actor_obs"]
            target = chunk["gt_actions"][:, : self.action_dim]
            if vision.shape[1:] != (324, 192, 3):
                raise ValueError(f"{path}: bad three-view shape {vision.shape}")
            if actor.shape[1:] != (self.actor_dim,) or target.shape[1:] != (self.action_dim,):
                raise ValueError(
                    f"{path}: actor/target shape mismatch {actor.shape}, {target.shape}"
                )
            if not np.isfinite(actor).all() or not np.isfinite(target).all():
                raise ValueError(f"{path}: actor observations or teacher actions are non-finite")
            result = {
                "vision": torch.from_numpy(vision.copy()),
                "actor": torch.from_numpy(actor.copy()),
                "target": torch.from_numpy(target.copy()),
                "chunk": str(path),
            }
            if self.include_object_target:
                if "obs_gt_obj_pos" not in chunk.files:
                    raise ValueError(f"{path}: Raven offline replay requires obs_gt_obj_pos")
                object_observation = chunk["obs_gt_obj_pos"]
                if (object_observation.ndim != 2
                        or object_observation.shape[0] != vision.shape[0]
                        or object_observation.shape[1] < 3
                        or not np.isfinite(object_observation).all()):
                    raise ValueError(f"{path}: invalid Raven object-position observation")
                result["object_target"] = torch.from_numpy(object_observation[:, :3].copy())
            return result


def build_policy(spec_path: Path, checkpoint_path: Path, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from pipeline.dagger.services.runtime_policy_contract import attest_policy_runtime

    runtime_attestation = attest_policy_runtime()

    spec = json.loads(spec_path.read_text())
    if sha256_file(checkpoint_path) != spec["checkpoint_sha256"]:
        raise ValueError("student bootstrap hash does not match policy specification")
    env = OmegaConf.create(
        {
            "robot": spec["env_config"]["robot"],
            "simulator": {
                "config": {
                    "cameras": {
                        "camera_resolutions": [108, 192],
                        "camera_types": [{"rgb": True}],
                    }
                }
            },
        }
    )
    policy = instantiate(
        OmegaConf.create(spec["actor_config"]),
        env_config=env,
        algo_config=OmegaConf.create(spec["algo_config"]),
        module_dim_dict=spec["module_dim_dict"],
        _recursive_=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy._roborender_runtime_attestation = runtime_attestation
    for parameter in policy.obj_pred_mlp.parameters():
        parameter.requires_grad_(False)
    if hasattr(policy, "std"):
        policy.std.requires_grad_(False)
    return policy.to(device), spec


def prepare_batch(batch: dict, device: torch.device, mean: torch.Tensor, std: torch.Tensor):
    vision = batch["vision"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    vision = (vision - mean) / std
    actor = batch["actor"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
    return vision, actor, target


def backward_raven_bc_mse(
    policy: torch.nn.Module,
    vision: torch.Tensor,
    actor: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Historical direct-supervised update; does not replay Raven's BN/RNG state."""
    expected = (RAVEN_TRAIN_CHUNKS_PER_UPDATE, RAVEN_CHUNK_FRAMES)
    if not (vision.shape[:2] == actor.shape[:2] == target.shape[:2] == expected):
        raise ValueError(
            "BC training requires one complete Raven BatchNorm population "
            f"B,T={expected}; got vision={tuple(vision.shape[:2])}, "
            f"actor={tuple(actor.shape[:2])}, target={tuple(target.shape[:2])}"
        )
    return backward_global_mse_in_environment_microbatches(
        policy,
        vision,
        actor,
        target,
        microbatch_environments=RAVEN_TRAIN_CHUNKS_PER_UPDATE,
        full_environment_batch_per_timestep=True,
    )


def student_lifecycle(args: argparse.Namespace) -> str:
    """Resolve explicitly versioned behavior without changing legacy BC116."""
    protocol = args.training_protocol
    mode = getattr(args, "student_lifecycle", None) or (
        REPLAY_STUDENT_LIFECYCLE if protocol == "fixed_final" else DIRECT_STUDENT_LIFECYCLE
    )
    if mode not in (DIRECT_STUDENT_LIFECYCLE, REPLAY_STUDENT_LIFECYCLE):
        raise ValueError(f"unknown BC student lifecycle: {mode}")
    if mode == REPLAY_STUDENT_LIFECYCLE:
        if protocol != "fixed_final":
            raise ValueError("Raven offline replay requires a new fixed_final protocol identity")
        if args.experimental_shuffle:
            raise ValueError("Raven offline replay requires sorted/cyclic chunk membership")
    return mode


def student_lifecycle_batch_fields(mode: str, chunk_batch_size: int = 16) -> dict:
    if type(chunk_batch_size) is not int or chunk_batch_size not in (16, 32):
        raise ValueError("offline BC admits only explicit full batches of16 or32 chunks")
    if mode == DIRECT_STUDENT_LIFECYCLE:
        if chunk_batch_size != 16:
            raise ValueError("historical direct-supervised BC retains its B16 batch identity")
        return {"schema": RAVEN_BATCHING_CONTRACT}
    if mode != REPLAY_STUDENT_LIFECYCLE:
        raise ValueError(f"unknown BC student lifecycle: {mode}")
    return {
        "schema": RAVEN_REPLAY_BATCHING_CONTRACT if chunk_batch_size == 16 else RAVEN_REPLAY_B32_BATCHING_CONTRACT,
        "student_lifecycle": mode,
        "replay_train_mode_forwards": RAVEN_CHUNK_FRAMES,
        "student_normal_sample_calls_per_update": RAVEN_CHUNK_FRAMES + 1,
        "lane_permutation": "torch.randperm_on_training_device_after_replay",
        "object_prediction_loss_coefficient": 0.0,
        "object_head_trainable_with_zero_gradients": True,
    }


def configure_raven_offline_student(policy: torch.nn.Module) -> None:
    """Admit only the pinned feed-forward, history-one student configuration.

    The bootstrap's frozen object head is an initialization convention, not the
    original optimizer lifecycle: zero-weight auxiliary loss creates zero grads
    and AdamW state for that head. Std remains frozen but its Normal draws matter.
    """
    required = ("init_rollout", "clear_rollout", "rollout_with_obj_pred",
                "action_mean_with_obj_pred")
    if any(not callable(getattr(policy, name, None)) for name in required):
        raise ValueError("Raven offline replay requires the object-predicting student API")
    if (getattr(policy, "is_recurrent", None) is not False
            or getattr(policy, "max_rollout_history", None) != 1
            or getattr(policy, "running_mean_std", None) is not None
            or getattr(policy, "state_history_module", None) is not None
            or getattr(policy, "use_temporal_attention", False)):
        raise ValueError("Raven offline replay supports only nonrecurrent history-one/RMS-off policy")
    if any(isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p != 0
           for module in policy.modules()):
        raise ValueError("Raven offline replay requires disabled dropout")
    noise = getattr(policy, "std", None)
    if (not isinstance(noise, torch.nn.Parameter) or noise.requires_grad
            or not getattr(policy, "clamp_noise_std", False)
            or getattr(policy, "max_noise_std", None) != 0.001
            or not bool(torch.isfinite(noise).all()) or not bool((noise > 0).all())):
        raise ValueError("Raven offline replay requires frozen positive std clamped at 0.001")
    if not isinstance(getattr(policy, "obj_pred_mlp", None), torch.nn.Module):
        raise ValueError("Raven offline replay requires the auxiliary object head")
    for parameter in policy.obj_pred_mlp.parameters():
        parameter.requires_grad_(True)


def backward_raven_offline_bc_mse(
    policy: torch.nn.Module,
    vision: torch.Tensor,
    actor: torch.Tensor,
    target: torch.Tensor,
    object_target: torch.Tensor,
    *, chunk_batch_size: int = 16,
) -> float:
    """Replay 81 train/no-grad student steps, permute lanes, then one B*T update.

    Uses the pinned student's methods, including discarded Normal samples. No
    teacher/Homie recomputation is needed for saved labels. Checkpoint replay
    preserves RNG and restores BN buffers, so recomputation adds no extra state.
    This scopes parity to the student lifecycle, not an unknown original library
    environment, simulator RNG, or teacher-side state.
    """
    student_lifecycle_batch_fields(REPLAY_STUDENT_LIFECYCLE, chunk_batch_size)
    expected = (chunk_batch_size, RAVEN_CHUNK_FRAMES)
    if (vision.shape[:2] != expected or actor.shape[:2] != expected
            or target.shape != (*expected, 11) or object_target.shape != (*expected, 3)):
        raise ValueError(f"Raven offline replay requires complete B{chunk_batch_size},T81, action11/object3 tensors")
    if not policy.training or any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not module.training
        for module in policy.modules()
    ):
        raise ValueError("Raven offline replay must remain in train mode, including BatchNorm")
    if not all(bool(torch.isfinite(value).all()) for value in (vision, actor, target, object_target)):
        raise ValueError("Raven offline replay inputs must be finite")
    policy.init_rollout()
    try:
        with torch.no_grad():
            for step in range(RAVEN_CHUNK_FRAMES):
                policy.rollout_with_obj_pred({
                    "vision_obs": vision[:, step], "actor_obs": actor[:, step],
                })
    finally:
        policy.clear_rollout()
    lanes = torch.randperm(chunk_batch_size, device=vision.device)

    def full_forward(images, observations):
        result = policy.action_mean_with_obj_pred({
            "vision_obs": images, "actor_obs": observations,
        })
        return result["action_mean"], result["predicted_obj_pos"]

    def contexts():
        return contextlib.nullcontext(), _batch_norm_recompute_context(policy)

    prediction, object_prediction = checkpoint(
        full_forward, vision[lanes], actor[lanes], use_reentrant=False,
        context_fn=contexts, preserve_rng_state=True,
    )
    if prediction.shape != target.shape or object_prediction.shape != object_target.shape:
        raise ValueError("Raven offline prediction/label shapes differ")
    action_loss = torch.nn.functional.mse_loss(prediction, target[lanes])
    object_loss = torch.nn.functional.mse_loss(object_prediction, object_target[lanes])
    if not bool(torch.isfinite(action_loss)) or not bool(torch.isfinite(object_loss)):
        raise ValueError("Raven offline action/object MSE must be finite before zero weighting")
    (action_loss + 0.0 * object_loss).backward()
    return float(action_loss.detach().item())


def validate_training_batch_population(
    chunk_count: int, chunk_batch_size: int, *, cyclic: bool = False
) -> int:
    """Return updates/epoch or reject any non-Raven/partial population."""
    if cyclic:
        return cyclic_sampling.updates_per_epoch(chunk_count, chunk_batch_size)
    if type(chunk_batch_size) is not int or chunk_batch_size not in (16, 32):
        raise ValueError(
            "three-view Raven BC requires --chunk-batch-size "
            "16 or32; partial/microbatched BatchNorm "
            "populations are not permitted"
        )
    if type(chunk_count) is not int or chunk_count < 1 or chunk_count % chunk_batch_size:
        raise ValueError(
            f"training split has {chunk_count} chunks and is not divisible by "
            f"the required {chunk_batch_size}; refusing to drop or "
            "pad an incomplete BatchNorm population"
        )
    return chunk_count // chunk_batch_size


@torch.inference_mode()
def validate(policy, loader, device, mean, std, action_dim: int) -> tuple[float, list[float]]:
    policy.eval()
    squared = torch.zeros(action_dim, dtype=torch.float64, device=device)
    frames = 0
    for batch in loader:
        vision, actor, target = prepare_batch(batch, device, mean, std)
        prediction = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
        if prediction.shape != target.shape:
            raise ValueError("validation prediction/teacher action shapes differ")
        error = (prediction - target).double().square()
        if not bool(torch.isfinite(error).all()):
            raise ValueError("validation contains non-finite action errors")
        squared += error.sum(dim=(0, 1))
        frames += target.shape[0] * target.shape[1]
    if not frames:
        raise ValueError("validation split has no frames")
    per_action = squared / frames
    return float(per_action.mean().item()), [float(value) for value in per_action.cpu()]


def save_checkpoint(
    path: Path,
    policy,
    optimizer,
    epoch: int,
    global_step: int,
    mse: float | None,
    per_action_mse: list[float] | None,
    metadata: dict,
    train_generator: torch.Generator,
) -> None:
    payload = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_mse": mse,
        "validation_per_action_mse": per_action_mse,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "train_generator": train_generator.get_state(),
        },
        **metadata,
    }
    if metadata.get('training_sampling_contract', {}).get('schema') == cyclic_sampling.SCHEMA:
        batch = metadata['training_batch_contract']
        payload['training_cursor'] = cyclic_sampling.cursor_state(
            batch['training_chunk_count'], batch['chunks_per_update'], global_step)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_parent(path)


def update_checkpoint_alias(source: Path, alias: Path) -> None:
    """Atomically point an alias at an immutable completed checkpoint."""
    temporary = alias.with_suffix(alias.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
    os.replace(temporary, alias)
    fsync_parent(alias)


def parse_args(argv=None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    assets = Path(os.environ.get("ROBORENDER_ASSET_ROOT", str(root)))
    if not assets.is_absolute():
        raise ValueError("ROBORENDER_ASSET_ROOT must be absolute")
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-protocol", choices=("fixed_final", "legacy_validation"),
        default="fixed_final", help="New BC runs disable validation and select the final epoch; legacy validation is explicit compatibility only.")
    parser.add_argument("--student-lifecycle",
        choices=(DIRECT_STUDENT_LIFECYCLE, REPLAY_STUDENT_LIFECYCLE), default=None,
        help="Default: Raven offline student replay for new fixed_final runs; historical direct supervision for legacy_validation. Checkpoints cannot cross modes.")
    parser.add_argument("--finalize-completed-run", type=Path, default=None,
        help="Recover only completed fixed-final logging/receipt publication; never construct a learner or run another update.")
    parser.add_argument("--dataset-root", type=Path, default=assets / "data/BC")
    parser.add_argument(
        "--train-split",
        type=Path,
        default=assets / "data/BC/splits_independent_v2/train.json",
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=assets / "data/BC/splits_independent_v2/validation.json",
    )
    parser.add_argument("--bootstrap", type=Path, default=assets / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt")
    parser.add_argument("--policy-spec", type=Path, default=assets / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json")
    parser.add_argument("--data-admission-json", type=Path, default=assets / "data/BC/splits_independent_v2/data-admission.json")
    parser.add_argument("--require-production-data-admission", action="store_true", default=True)
    parser.add_argument("--allow-qualification-data", action="store_true",
        help="Permit a v2 qualification report only for an explicitly smoke-named run; never production.")
    parser.add_argument("--allow-legacy-reuse", action="store_true",
        help="Explicit legacy2000/fresh320 contract; clean bootstrap or exact same-admission resume only.")
    parser.add_argument("--allow-training-extension", action="store_true",
        help="Explicit admitted legacy2000+fresh560 training/fresh320 validation contract; exact resume only.")
    parser.add_argument("--diagnostic-without-production-admission", dest="require_production_data_admission",
        action="store_false", help="Explicit diagnostic-only opt-out; never qualifies production data.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, AdamW, epoch, and global step from a BC checkpoint.",
    )
    parser.add_argument(
        "--resume-validation-state",
        type=Path,
        default=None,
        help="Prior validation_state.json; inferred from --resume-checkpoint when omitted.",
    )
    parser.add_argument("--output-root", type=Path, default=assets / "checkpoints/students/BC")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--chunk-batch-size", type=int, default=None,
        help=("Complete stored chunks per update: defaults to 32 for fixed_final, "
              "16 for legacy_validation; explicit 16 preserves admitted B16 replay histories.")
    )
    parser.add_argument("--validation-chunk-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--relative-improvement", type=float, default=0.005)
    parser.add_argument("--absolute-improvement", type=float, default=1e-6)
    # The first production run reached its epoch-40 safety ceiling with only
    # four of eight patience checks consumed.  A ceiling is not a convergence
    # criterion: leave enough room for the validation plateau rule to decide.
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-epochs", type=int, default=100)
    budget.add_argument("--no-epoch-cap", dest="max_epochs", action="store_const", const=None,
        help="No epoch ceiling: continue until the configured stopping rule or an explicit stop/error.")
    fit = parser.add_mutually_exclusive_group()
    fit.add_argument("--stop-training-eval-mse-below", type=float, default=None,
        help="Opt-in: plateau stops only when full training-set eval-mode MSE is below this target; epoch cap still applies.")
    fit.add_argument("--stop-training-epoch-mse-below", type=float, default=None,
        help="Opt-in: plateau stops only when the FULL epoch training-mode average MSE is below this target.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experimental-shuffle", action="store_true",
        help="Opt out of Raven's sorted/cyclic offline chunk order; not Raven-equivalent.",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="dagger")
    parser.add_argument("--wandb-group", default="BC")
    parser.add_argument("--wandb-run-name", default="BC")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.chunk_batch_size is None:
        args.chunk_batch_size = (FIXED_FINAL_TRAIN_CHUNKS_PER_UPDATE
            if args.training_protocol == "fixed_final" else RAVEN_TRAIN_CHUNKS_PER_UPDATE)
    args.epoch_budget_explicit = any(
        token == "--max-epochs" or token.startswith("--max-epochs=") for token in argv
    )
    args.student_lifecycle = student_lifecycle(args)
    student_lifecycle_batch_fields(args.student_lifecycle, args.chunk_batch_size)
    return args


def infer_resume_validation_state(checkpoint: Path) -> Path:
    """Locate the immutable run state associated with a BC checkpoint."""
    if checkpoint.parent.name == "validation_checkpoints":
        return checkpoint.parent.parent / "validation_state.json"
    return checkpoint.parent / "validation_state.json"


def load_data_admission(args: argparse.Namespace) -> dict:
    """Check statistical provenance before any CUDA model construction."""
    if getattr(args, "allow_training_extension", False) and (
            getattr(args, "allow_qualification_data", False) or getattr(args, "allow_legacy_reuse", False)
            or not args.require_production_data_admission):
        raise ValueError("training extension cannot use another admission protocol or diagnostic bypass")
    if args.data_admission_json is None or not args.data_admission_json.is_file():
        if args.require_production_data_admission:
            raise ValueError("production BC requires --data-admission-json")
        if not isinstance(args.run_name, str) or not args.run_name.startswith("smoke-"):
            raise ValueError("historical diagnostic opt-out requires smoke- run name")
        return {"status": "historical_unqualified", "production_admitted": False}
    from pipeline.bc_training.data_admission import validate_training_admission

    path = args.data_admission_json.resolve(strict=True)
    from pipeline.bc_training import legacy_reuse
    document = json.loads(legacy_reuse.read_metadata(path))
    from pipeline.bc_training import train_union
    if document.get("schema") == train_union.SCHEMA:
        if (getattr(args, "training_protocol", None) != "fixed_final"
                or getattr(args, "allow_training_extension", False)
                or getattr(args, "allow_legacy_reuse", False)
                or getattr(args, "allow_qualification_data", False)
                or not args.require_production_data_admission):
            raise ValueError("train union requires fixed_final without legacy/qualification/bypass flags")
        report = train_union.validate_training_admission(
            path, args.train_split, args.validation_split, args.dataset_root)
        return {"status": "train_union_admitted", "report_path": str(path),
                "report_sha256": sha256_file(path), **report}
    from pipeline.bc_training import training_extension
    if document.get("schema") == training_extension.ADMISSION_SCHEMA:
        if (getattr(args, "allow_qualification_data", False) or getattr(args, "allow_legacy_reuse", False)
                or not args.require_production_data_admission):
            raise ValueError("training extension cannot use another admission protocol or diagnostic bypass")
        report = training_extension.validate_training_admission(path, args.train_split, args.validation_split,
            allow_training_extension=getattr(args, "allow_training_extension", False), run_name=args.run_name,
            bootstrap_path=args.bootstrap, policy_spec_path=args.policy_spec,
            resume_checkpoint=args.resume_checkpoint, resume_validation_state=args.resume_validation_state,
            dataset_root=args.dataset_root)
        return {"status": "training_extension_admitted", "report_path": str(path),
                "report_sha256": sha256_file(path), **report}
    if getattr(args, "allow_training_extension", False):
        raise ValueError("training extension flag requires its exact distinct admission schema")
    if document.get("schema") == legacy_reuse.ADMISSION_SCHEMA:
        if getattr(args, "allow_qualification_data", False) or not args.require_production_data_admission:
            raise ValueError("legacy reuse cannot use qualification or diagnostic admission bypass")
        report = legacy_reuse.validate_training_admission(path, args.train_split, args.validation_split,
            allow_legacy_reuse=getattr(args, "allow_legacy_reuse", False), run_name=args.run_name,
            bootstrap_path=args.bootstrap, policy_spec_path=args.policy_spec,
            resume_checkpoint=args.resume_checkpoint, resume_validation_state=args.resume_validation_state,
            dataset_root=args.dataset_root)
        return {"status": "legacy_reuse_admitted", "report_path": str(path),
                "report_sha256": sha256_file(path), **report}
    if getattr(args, "allow_legacy_reuse", False):
        raise ValueError("legacy reuse flag requires its exact distinct admission schema")
    if (getattr(args, "allow_qualification_data", False)
            and not getattr(args, "wandb_run_name", args.run_name).startswith("smoke-")):
        raise ValueError("qualification W&B run name must also start with smoke-")
    report = validate_training_admission(path, args.train_split, args.validation_split,
        allow_qualification=getattr(args, "allow_qualification_data", False), run_name=args.run_name)
    return {"status": "production_admitted" if report["production_admitted"] else "qualification_only",
            "report_path": str(path),
            "report_sha256": sha256_file(path), **report}


def validate_resume_data_admission(checkpoint, expected):
    from pipeline.bc_training import legacy_reuse
    from pipeline.bc_training import training_extension
    if (expected or {}).get("schema") == training_extension.ADMISSION_SCHEMA:
        training_extension.validate_resume_identity(checkpoint, expected)
    if (expected or {}).get("schema") == legacy_reuse.ADMISSION_SCHEMA:
        legacy_reuse.validate_resume_identity(checkpoint, expected)
    old = checkpoint.get("data_admission", {}).get("report_sha256")
    current = (expected or {}).get("report_sha256")
    if old != current:
        raise ValueError("resume checkpoint data admission differs (including qualification); cannot drop or replace its report")


def validate_resume_training_batch_contract(
    checkpoint: dict, expected_training_batch_contract: dict | None
) -> None:
    if expected_training_batch_contract is None:
        return
    if checkpoint.get("training_batch_contract") != expected_training_batch_contract:
        raise ValueError(
            "resume checkpoint training batching contract does not match; "
            "legacy B=4 and Raven B=16 optimizer histories cannot be mixed"
        )


def training_sampling_contract(paths: list[Path], dataset_root: Path, *, shuffle: bool,
                               cyclic_batch_size: int | None = None) -> dict:
    """Pin lexical order and versioned cursor semantics.

    Legacy v1 requires divisible pools. Fixed-final v2 wraps within full
    batches and never resets at reporting-epoch boundaries.
    """
    relative_paths = [str(path.relative_to(dataset_root)) for path in paths]
    if relative_paths != sorted(relative_paths):
        raise ValueError("training paths must be lexically sorted before sampling")
    result = {"schema": "roborender.bc-sampling.v1",
            "mode": "experimental_shuffle" if shuffle else "raven_sorted_cyclic",
            "ordered_paths_sha256": hashlib.sha256(json.dumps(
                relative_paths, separators=(",", ":"), ensure_ascii=True,
            ).encode()).hexdigest(), "epoch_boundary_cursor": 0}
    if cyclic_batch_size is not None:
        if shuffle:
            raise ValueError('persistent Raven cyclic loading forbids shuffled chunk membership')
        result.pop('epoch_boundary_cursor')
        result.update(schema=cyclic_sampling.SCHEMA, chunk_count=len(paths),
            batch_size=cyclic_batch_size,
            updates_per_reporting_epoch=cyclic_sampling.updates_per_epoch(len(paths), cyclic_batch_size),
            reporting_epoch='ceil_N_over_B_full_updates_cursor_never_reset',
            cursor_commit='accepted_optimizer_updates_only', partial_batch='wrap_sorted_pool')
    return result


def load_resume_state(
    checkpoint_path: Path,
    validation_state_path: Path,
    *,
    policy,
    optimizer,
    train_generator: torch.Generator,
    spec: dict,
    bootstrap_sha256: str,
    train_split_sha256: str,
    validation_split_sha256: str,
    stopper: EarlyStopping,
    seed: int,
    expected_learning_rate: float,
    expected_training_batch_contract: dict | None = None,
    expected_data_admission: dict | None = None,
    expected_sampling_contract: dict | None = None,
) -> tuple[dict, list[dict], str]:
    """Strictly restore a prior BC optimizer/checkpoint and selection history."""
    checkpoint_path = checkpoint_path.resolve()
    validation_state_path = validation_state_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_protocol", {}).get("schema") == "roborender.bc-fixed-final.v1":
        raise ValueError("fixed_final checkpoints cannot resume into legacy validation")
    required = {
        "policy_state_dict", "optimizer_state_dict", "epoch", "global_step",
        "validation_mse", "policy_spec", "bootstrap_sha256",
        "train_split_sha256", "validation_split_sha256",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {missing}")
    expected = {
        "bootstrap_sha256": bootstrap_sha256,
        "train_split_sha256": train_split_sha256,
        "validation_split_sha256": validation_split_sha256,
    }
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise ValueError(f"resume checkpoint {key} does not match this run")
    if checkpoint["policy_spec"] != spec:
        raise ValueError("resume checkpoint policy specification does not match")
    validate_resume_training_batch_contract(
        checkpoint, expected_training_batch_contract
    )
    if expected_sampling_contract is not None and checkpoint.get("training_sampling_contract") != expected_sampling_contract:
        raise ValueError("resume checkpoint sampling contract differs or is unrecorded; refusing to silently change chunk order")
    validate_resume_data_admission(checkpoint, expected_data_admission)
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    if epoch < 0 or global_step < 0:
        raise ValueError("resume checkpoint has invalid epoch/global_step")
    state = (json.loads(validation_state_path.read_text())
             if validation_state_path.is_file() else {})
    source_history = checkpoint.get("validation_history", state.get("history"))
    if not isinstance(source_history, list) or not source_history:
        raise ValueError("resume validation state has no history")
    history = [record.copy() for record in source_history if int(record["epoch"]) <= epoch]
    if not history or int(history[-1]["epoch"]) != epoch:
        raise ValueError("resume validation history does not contain checkpoint epoch")
    if int(history[-1]["global_step"]) != global_step:
        raise ValueError("resume checkpoint global step disagrees with validation history")
    if not math.isclose(
        float(history[-1]["mse"]), float(checkpoint["validation_mse"]),
        rel_tol=1e-7, abs_tol=1e-9,
    ):
        raise ValueError("resume checkpoint MSE disagrees with validation history")
    previous_step = -1
    for record in history:
        record_epoch = int(record["epoch"])
        record_step = int(record["global_step"])
        if record_step < 0 or record_step <= previous_step:
            raise ValueError("resume validation history global steps are not strictly increasing")
        meaningful, saturated = stopper.update(record_epoch, float(record["mse"]))
        # Rebuild legacy selection semantics explicitly instead of trusting stale
        # improved flags, which previously discarded small new minima.
        record.update({"improved": stopper.best_epoch == record_epoch,
                       "meaningful_improvement": meaningful,
                       "bad_validations": stopper.bad_validations,
                       "saturated": saturated})
        previous_step = record_step
    if int(history[0]["epoch"]) != 0:
        raise ValueError("resume validation history is missing its epoch-zero baseline")
    stored_stopper = checkpoint.get("early_stopping_state")
    if stored_stopper is not None and stored_stopper != stopper.state_dict():
        raise ValueError("resume early-stopping state/config disagrees with validation history")
    for key, value in expected.items():
        if key in state and state[key] != value:
            raise ValueError(f"resume validation state {key} does not match this run")

    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if restored_rates != {float(expected_learning_rate)}:
        raise ValueError(
            "resume optimizer learning rate disagrees with --learning-rate: "
            f"checkpoint={sorted(restored_rates)} requested={expected_learning_rate}"
        )

    rng = checkpoint.get("rng_state")
    if rng is None:
        # Legacy checkpoints predate resumable RNG snapshots. This is deterministic,
        # but cannot reproduce the exact future shuffle of the interrupted process.
        random.seed(seed + epoch)
        np.random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        torch.cuda.manual_seed_all(seed + epoch)
        train_generator.manual_seed(seed + epoch)
        rng_mode = "legacy_deterministic_epoch_seed"
    else:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
        train_generator.set_state(rng["train_generator"])
        rng_mode = "exact_checkpoint_restore"
    return checkpoint, history, rng_mode


def restore_fixed_final_checkpoint(path, *, metadata, policy, optimizer, generator):
    """Restore only an exact epoch-boundary continuation of this new protocol."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_fixed_final_resume(checkpoint, metadata["training_protocol"])
    for key in ("bootstrap_sha256", "policy_spec", "train_split_sha256",
                "validation_split_sha256", "training_batch_contract",
                "training_sampling_contract", "optimizer_contract", "training_source_identity"):
        if checkpoint.get(key) != metadata[key]:
            raise ValueError(f"fixed_final resume {key} differs from the checkpoint")
    validate_resume_data_admission(checkpoint, metadata["data_admission"])
    epoch, step = checkpoint["epoch"], checkpoint["global_step"]
    updates = metadata["training_batch_contract"]["updates_per_epoch"]
    if metadata['training_sampling_contract'].get('schema') == cyclic_sampling.SCHEMA:
        batch = metadata['training_batch_contract']
        cyclic_sampling.validate_cursor(checkpoint.get('training_cursor'),
            batch['training_chunk_count'], batch['chunks_per_update'], step)
    history = checkpoint.get("training_history")
    if (not isinstance(history, list) or len(history) != epoch + 1
            or step != epoch * updates):
        raise ValueError("fixed_final checkpoint is not a complete epoch boundary")
    for expected_epoch, record in enumerate(history):
        if (record.get("epoch") != expected_epoch
                or record.get("global_step") != expected_epoch * updates):
            raise ValueError("fixed_final epoch history is not contiguous")
        mse = record.get("training_epoch_mse")
        if expected_epoch and (not isinstance(mse, (int, float))
                               or not math.isfinite(mse) or mse < 0):
            raise ValueError("fixed_final history has invalid training MSE")
    if checkpoint.get("training_epoch_mse") != history[-1].get("training_epoch_mse"):
        raise ValueError("fixed_final checkpoint training MSE differs from history")
    rng = checkpoint["rng_state"]
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda", "train_generator"}:
        raise ValueError("fixed_final resume requires complete RNG state; no synthetic reseeding")
    rates = {float(group["lr"]) for group in checkpoint["optimizer_state_dict"]["param_groups"]}
    if rates != {metadata["optimizer_contract"]["learning_rate"]}:
        raise ValueError("fixed_final optimizer learning rate differs")
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    generator.set_state(rng["train_generator"])
    return checkpoint, [dict(record) for record in history]


def training_source_identity():
    return {relative: sha256_file(PROJECT_ROOT / relative) for relative in (
        "pipeline/bc_training/train_bc_h200.py", "pipeline/bc_training/stopping_gate.py",
        "pipeline/bc_training/cyclic_sampling.py",
        "pipeline/dagger/services/raven_learner_worker.py")}


def _verify_completed_fixed_final(output):
    """Verify the durable final boundary using metadata/hashes, never torch.load."""
    config_path, state_path = output / "run_config.json", output / "training_state.json"
    config, state = json.loads(config_path.read_text()), json.loads(state_path.read_text())
    protocol = config.get("training_protocol", {})
    expected = fixed_final_contract(protocol.get("max_epochs"),
        budget_explicit=config.get("epoch_budget_explicit") is True)
    if (protocol != expected or state.get("training_protocol") != expected
            or state.get("schema") != "roborender.bc-fixed-final-state.v1"
            or state.get("status") not in {"TRAINING", "READY_FOR_TEST"}
            or state.get("validation_enabled") is not False
            or Path(config["output"]).resolve() != output
            or state.get("training_config_sha256") != sha256_file(config_path)
            or state.get("training_source_identity") != training_source_identity()):
        raise ValueError("fixed-final completion config/source/protocol identity differs")
    wandb_record = output / "wandb-run.json"
    if config["wandb_mode"] != "disabled":
        if state.get("wandb_run_sha256") != sha256_file(wandb_record):
            raise ValueError("fixed-final W&B run receipt identity changed")
    elif state.get("wandb_run_sha256") is not None or wandb_record.exists():
        raise ValueError("disabled logging unexpectedly acquired a W&B run")
    batch = state.get("training_batch_contract", {})
    cyclic = state.get('training_sampling_contract', {}).get('schema') == cyclic_sampling.SCHEMA
    updates = validate_training_batch_population(batch.get("training_chunk_count", 0),
                                                 batch.get("chunks_per_update", 0), cyclic=cyclic)
    if cyclic:
        if (state['training_sampling_contract'] != config.get('training_sampling_contract')
                or batch.get('drop_last') is not False
                or batch.get('partial_batch') != 'wrap_sorted_pool'
                or batch.get('epoch_is_exact_dataset_pass') is not
                    (batch['training_chunk_count'] % batch['chunks_per_update'] == 0)):
            raise ValueError('fixed-final completion cyclic sampling identity differs')
        cyclic_sampling.validate_cursor(state.get('training_cursor'),
            batch['training_chunk_count'], batch['chunks_per_update'], state['global_step'])
    lifecycle_fields = student_lifecycle_batch_fields(
        config.get("student_lifecycle", DIRECT_STUDENT_LIFECYCLE), batch.get("chunks_per_update", 0))
    if (any(batch.get(key) != value for key, value in lifecycle_fields.items())
            or batch.get("frames_per_chunk") != RAVEN_CHUNK_FRAMES
            or batch.get("chunks_per_update") != config["chunk_batch_size"]
            or batch.get("policy_forward_population") != config["chunk_batch_size"] * RAVEN_CHUNK_FRAMES
            or batch.get("updates_per_epoch") != updates
            or type(state.get("epoch")) is not int or state["epoch"] != expected["max_epochs"]
            or type(state.get("global_step")) is not int
            or state["global_step"] != expected["max_epochs"] * updates):
        raise ValueError("fixed-final completion requires the exact fully completed epoch budget")
    for arg_name, hash_name in (("train_split", "train_split_sha256"),
            ("validation_split", "validation_split_sha256"),
            ("bootstrap", "bootstrap_sha256"), ("policy_spec", "policy_spec_sha256"),
            ("data_admission_json", "data_admission_sha256")):
        expected_hash = state.get(hash_name)
        path = Path(config[arg_name]).resolve() if config.get(arg_name) is not None else None
        if expected_hash is None:
            if path is not None and path.exists():
                raise ValueError(f"fixed-final completion unexpectedly gained {arg_name}")
            if arg_name == "data_admission_json" and config.get("require_production_data_admission"):
                raise ValueError("fixed-final completion requires production data admission")
        elif path is None or sha256_file(path) != expected_hash:
            raise ValueError(f"fixed-final completion {arg_name} identity changed")
    selected = output / "epoch_checkpoints" / f"epoch_{state['epoch']:04d}_step_{state['global_step']:06d}.pt"
    if (Path(state["selected_checkpoint"]).resolve() != selected.resolve()
            or sha256_file(selected) != state.get("selected_checkpoint_sha256")
            or sha256_file(output / "last.pt") != state.get("selected_checkpoint_sha256")):
        raise ValueError("fixed-final completion final checkpoint identity changed")
    return config, state


def _completion_operation_id(state):
    identity = {key: state[key] for key in ("training_config_sha256", "selected_checkpoint_sha256",
        "epoch", "global_step", "training_protocol", "training_source_identity", "wandb_run_sha256")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def publish_fixed_final_completion(output, *, wandb_run=None, elapsed_seconds=None, recovery=True):
    """Idempotent finalization; optional recovery uses the same W&B run, no GPU."""
    output = Path(output).resolve(strict=True)
    with (output / "completion.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("BC completion publication is already owned") from exc
        config, state = _verify_completed_fixed_final(output)
        operation_id = _completion_operation_id(state)
        final_state = {**state, "status": "READY_FOR_TEST", "stop_reason": "explicit_epoch_budget",
            "optimizer_state_saved": True, "rng_state_saved": True,
            "test_execution_started": False, "elapsed_seconds": elapsed_seconds,
            "completion_operation_id": operation_id}
        intent_path, receipt_path = output / "training-finalization-intent.json", output / "training-complete.json"
        if recovery and not intent_path.exists():
            raise ValueError("no saved finalization intent; trainer quiescence must be reconciled before publication recovery")
        if intent_path.exists():
            prior = json.loads(intent_path.read_text())
            if (prior.get("completion_operation_id") != operation_id
                    or any(prior.get(key) != final_state[key] for key in final_state if key != "elapsed_seconds")):
                raise ValueError("BC finalization intent differs; refusing duplicate/colliding publication")
            final_state = prior
        else:
            atomic_json(intent_path, final_state)
        if receipt_path.exists():
            if json.loads(receipt_path.read_text()) != final_state:
                raise ValueError("BC completion receipt differs from its finalization intent")
            return final_state
        mode = config["wandb_mode"]
        if mode != "disabled":
            import wandb
            record = json.loads((output / "wandb-run.json").read_text())
            if record.get("mode") != mode or record.get("project") != config["wandb_project"]:
                raise ValueError("BC W&B run identity differs from configuration")
            proof = {"completion_operation_id": operation_id,
                     "selected_checkpoint_sha256": state["selected_checkpoint_sha256"],
                     "global_step": state["global_step"]}
            remote = None
            if mode == "online":
                remote = wandb.Api().run(f"{record['entity']}/{record['project']}/{record['id']}")
                existing_id = remote.summary.get("completion_operation_id")
                if existing_id is not None and existing_id != operation_id:
                    raise ValueError("W&B already records a different BC completion operation")
            published = remote is not None and all(remote.summary.get(k) == v for k, v in proof.items())
            if not published:
                if wandb_run is None:
                    if mode != "online":
                        raise ValueError("offline W&B finalization requires explicit log recovery; no remote publication is claimed")
                    wandb_run = wandb.init(id=record["id"], entity=record["entity"],
                        project=record["project"], resume="must", mode="online")
                if wandb_run.id != record["id"] or wandb_run.entity != record["entity"]:
                    raise ValueError("BC finalization refused a different W&B run")
                wandb_run.summary.update(final_state)
                wandb_run.finish()
            elif wandb_run is not None:
                wandb_run.finish()
            if mode == "online":
                remote = wandb.Api().run(f"{record['entity']}/{record['project']}/{record['id']}")
                if any(remote.summary.get(k) != v for k, v in proof.items()):
                    raise RuntimeError("BC final checkpoint logging is not yet confirmed by W&B readback")
        atomic_json(output / "training_state.json", final_state)
        atomic_json(receipt_path, final_state)
        return final_state


def run_fixed_final(args, data_admission, protocol, *, device=None):
    """Train only, save the final complete epoch, then seal a testing handoff.

    No validation payload, evaluation loader, held-out forward pass, patience
    rule or test-driven selection is reachable through this entry point.
    Existing admission checks remain metadata-only provenance checks.
    """
    expected = fixed_final_contract(args.max_epochs, budget_explicit=args.epoch_budget_explicit,
        epoch_target=args.stop_training_epoch_mse_below, eval_target=args.stop_training_eval_mse_below)
    if protocol != expected:
        raise ValueError("fixed_final execution protocol differs from explicit arguments")
    lifecycle = student_lifecycle(args)
    lifecycle_fields = student_lifecycle_batch_fields(lifecycle, args.chunk_batch_size)
    replay_student = lifecycle == REPLAY_STUDENT_LIFECYCLE
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0") if device is None else device
    _, train_paths = load_split(args.train_split.resolve(), "train", args.dataset_root.resolve())
    train_paths = sorted(train_paths, key=lambda path: str(path.relative_to(args.dataset_root.resolve())))
    sampling_contract = training_sampling_contract(
        train_paths, args.dataset_root.resolve(), shuffle=args.experimental_shuffle,
        cyclic_batch_size=args.chunk_batch_size)
    updates_per_epoch = validate_training_batch_population(len(train_paths), args.chunk_batch_size, cyclic=True)
    policy, spec = build_policy(args.policy_spec.resolve(), args.bootstrap.resolve(), device)
    if replay_student:
        configure_raven_offline_student(policy)
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    train_data = BCChunks(train_paths, actor_dim, action_dim, include_object_target=replay_student)
    generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = cyclic_sampling.CyclicBatchSampler(len(train_paths), args.chunk_batch_size)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler,
        generator=generator, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    training_batch_contract = {
        **lifecycle_fields,
        "chunks_per_update": args.chunk_batch_size,
        "frames_per_chunk": RAVEN_CHUNK_FRAMES,
        "policy_forward_population": args.chunk_batch_size * RAVEN_CHUNK_FRAMES,
        "drop_last": False, "training_chunk_count": len(train_data),
        "partial_batch": "wrap_sorted_pool", "epoch_is_exact_dataset_pass": len(train_data) % args.chunk_batch_size == 0,
        "updates_per_epoch": updates_per_epoch,
        "index_safe_max_pools": install_index_safe_max_pool2d(policy),
        "activation_checkpointing": True,
    }
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    metadata = {
        "schema": "roborender.bc_student_checkpoint.v4",
        "bootstrap": str(args.bootstrap.resolve()),
        "bootstrap_sha256": sha256_file(args.bootstrap.resolve()), "policy_spec": spec,
        "train_split": str(args.train_split.resolve()),
        "train_split_sha256": sha256_file(args.train_split.resolve()),
        # Preserve admission identity without opening validation images/actions.
        "validation_split_sha256": (sha256_file(args.validation_split.resolve())
                                    if args.validation_split.is_file() else None),
        "validation_enabled": False, "selection": "last_completed_epoch",
        "training_protocol": protocol, "data_admission": data_admission,
        "training_batch_contract": training_batch_contract,
        "training_sampling_contract": sampling_contract,
        "training_source_identity": training_source_identity(),
        "optimizer_contract": {"algorithm": "AdamW", "learning_rate": args.learning_rate,
            "weight_decay": 0.0, "max_grad_norm": args.max_grad_norm},
    }
    if replay_student:
        metadata["optimizer_contract"].update({
            "implementation": "torch.optim.AdamW", "torch_version": torch.__version__,
            "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
            "trainable_parameters": [name for name, parameter in policy.named_parameters()
                                     if parameter.requires_grad],
            "original_transformers_optimizer_version": "not_certified",
        })
    start_epoch, global_step = 0, 0
    history = [{"epoch": 0, "global_step": 0, "training_epoch_mse": None}]
    if args.resume_checkpoint is not None:
        checkpoint, history = restore_fixed_final_checkpoint(args.resume_checkpoint.resolve(),
            metadata=metadata, policy=policy, optimizer=optimizer, generator=generator)
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_root / (args.run_name or f"bc_3view_{timestamp}")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "epoch_checkpoints"
    checkpoints.mkdir()
    config = {**vars(args), "output": output, "training_protocol": protocol,
        "training_sampling_contract": sampling_contract,
        "device": torch.cuda.get_device_name(0)}
    atomic_json(output / "run_config.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()})
    state_inputs = {
        "train_split_sha256": metadata["train_split_sha256"],
        "validation_split_sha256": metadata["validation_split_sha256"],
        "training_config_sha256": sha256_file(output / "run_config.json"),
        "data_admission_sha256": (sha256_file(args.data_admission_json.resolve())
            if args.data_admission_json is not None and args.data_admission_json.is_file() else None),
        "training_batch_contract": training_batch_contract,
        "training_source_identity": metadata["training_source_identity"],
        "training_sampling_contract": sampling_contract,
        "bootstrap_sha256": metadata["bootstrap_sha256"],
        "policy_spec_sha256": sha256_file(args.policy_spec.resolve()),
    }
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, group=args.wandb_group,
            name=args.wandb_run_name, mode=args.wandb_mode,
            config={key: str(value) if isinstance(value, Path) else value for key, value in config.items()})
        atomic_json(output / "wandb-run.json", {"id": wandb_run.id, "url": wandb_run.url,
            "entity": wandb_run.entity, "project": args.wandb_project,
            "name": args.wandb_run_name, "mode": args.wandb_mode})
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
    state_inputs["wandb_run_sha256"] = (sha256_file(output / "wandb-run.json")
                                        if wandb_run is not None else None)
    started = time.time()
    initial = checkpoints / f"epoch_{start_epoch:04d}_step_{global_step:06d}.pt"
    save_checkpoint(initial, policy, optimizer, start_epoch, global_step, None, None,
        {**metadata, "training_history": history,
         "training_epoch_mse": history[-1]["training_epoch_mse"]}, generator)
    update_checkpoint_alias(initial, output / "last.pt")
    for epoch in epoch_sequence(start_epoch, protocol["max_epochs"]):
        policy.train()
        batch_sampler.start_after(global_step)
        total_squared, total_elements, epoch_updates = 0.0, 0, 0
        for batch in train_loader:
            vision, actor, target = prepare_batch(batch, device, mean, std)
            optimizer.zero_grad(set_to_none=True)
            if replay_student:
                object_target = batch["object_target"].to(
                    device=device, dtype=torch.float32, non_blocking=True)
                loss_value = backward_raven_offline_bc_mse(
                    policy, vision, actor, target, object_target, chunk_batch_size=args.chunk_batch_size)
            else:
                loss_value = backward_raven_bc_mse(policy, vision, actor, target)
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite training loss at step {global_step}")
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            elements = target.numel()
            total_squared += loss_value * elements
            total_elements += elements
            global_step += 1
            epoch_updates += 1
            if wandb_run is not None:
                wandb_run.log({"train/mse_step": loss_value, "global_step": global_step})
        if epoch_updates != updates_per_epoch or total_elements == 0:
            raise RuntimeError("incomplete BC epoch; refusing final checkpoint/handoff")
        train_mse = total_squared / total_elements
        history.append({"epoch": epoch, "global_step": global_step, "training_epoch_mse": train_mse})
        selected = checkpoints / f"epoch_{epoch:04d}_step_{global_step:06d}.pt"
        save_checkpoint(selected, policy, optimizer, epoch, global_step, None, None,
            {**metadata, "training_history": history, "training_epoch_mse": train_mse}, generator)
        update_checkpoint_alias(selected, output / "last.pt")
        selected_sha256 = sha256_file(selected)
        if sha256_file(output / "last.pt") != selected_sha256:
            raise RuntimeError("last checkpoint alias differs from completed epoch checkpoint")
        state = {"schema": "roborender.bc-fixed-final-state.v1", "status": "TRAINING",
            "epoch": epoch, "global_step": global_step, "training_epoch_mse": train_mse,
            "selected_checkpoint": str(selected), "selected_checkpoint_sha256": selected_sha256,
            "training_protocol": protocol, "validation_enabled": False,
            "training_cursor": cyclic_sampling.cursor_state(len(train_data), args.chunk_batch_size, global_step),
            **state_inputs}
        atomic_json(output / "training_state.json", state)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "global_step": global_step,
                "train/mse": train_mse, "train/mse_epoch": train_mse})
        print(f"epoch={epoch} train_mse={train_mse:.8f} validation=disabled "
              f"budget={protocol['max_epochs']} selection=last_completed_epoch", flush=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    final_state = publish_fixed_final_completion(output, wandb_run=wandb_run,
        elapsed_seconds=time.time() - started, recovery=False)
    print(json.dumps({"output": str(output), **final_state}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.finalize_completed_run is not None:
        if args.resume_checkpoint is not None or args.training_protocol != "fixed_final":
            raise ValueError("finalize-only and training resume are separate operations")
        result = publish_fixed_final_completion(args.finalize_completed_run)
        print(json.dumps(result, sort_keys=True))
        return
    data_admission = load_data_admission(args)
    protocol = None
    if args.training_protocol == "fixed_final":
        protocol = fixed_final_contract(args.max_epochs,
            budget_explicit=args.epoch_budget_explicit,
            epoch_target=args.stop_training_epoch_mse_below,
            eval_target=args.stop_training_eval_mse_below)
        if args.resume_validation_state is not None:
            raise ValueError("fixed_final does not consume legacy --resume-validation-state")
    print(f"data_admission={data_admission['status']}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("BC training requires a CUDA H200")
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError(f"expected H200 SM90, got {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    if protocol is not None:
        run_fixed_final(args, data_admission, protocol)
        return
    if (args.min_epochs < 1 or args.patience < 1
            or (args.max_epochs is not None and args.max_epochs < args.min_epochs)):
        raise ValueError("invalid early-stopping epoch/patience settings")
    fitting_stop_contract = build_fitting_stop_contract(
        args.stop_training_epoch_mse_below, args.stop_training_eval_mse_below)
    fitting_target = (args.stop_training_epoch_mse_below
                      if args.stop_training_epoch_mse_below is not None
                      else args.stop_training_eval_mse_below)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    train_index, train_paths = load_split(args.train_split.resolve(), "train", args.dataset_root.resolve())
    validation_index, validation_paths = load_split(
        args.validation_split.resolve(), "validation", args.dataset_root.resolve()
    )
    validate_split_pair(train_index, train_paths, validation_index, validation_paths)
    train_paths = sorted(train_paths, key=lambda path: str(path.relative_to(args.dataset_root.resolve())))
    sampling_contract = training_sampling_contract(
        train_paths, args.dataset_root.resolve(), shuffle=args.experimental_shuffle,
    )

    policy, spec = build_policy(args.policy_spec.resolve(), args.bootstrap.resolve(), device)
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    train_data = BCChunks(train_paths, actor_dim, action_dim)
    validation_data = BCChunks(validation_paths, actor_dim, action_dim)
    updates_per_epoch = validate_training_batch_population(
        len(train_data), args.chunk_batch_size
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.chunk_batch_size,
        shuffle=args.experimental_shuffle,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        # Divisibility is checked above. This is a fail-safe against a future
        # sampler change; no training chunk is dropped in a valid run.
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.validation_chunk_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_eval_loader = (DataLoader(
        train_data, batch_size=args.validation_chunk_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 1),
    ) if args.stop_training_eval_mse_below is not None else None)
    index_safe_max_pools = install_index_safe_max_pool2d(policy)
    training_batch_contract = {
        "schema": RAVEN_BATCHING_CONTRACT,
        "chunks_per_update": RAVEN_TRAIN_CHUNKS_PER_UPDATE,
        "frames_per_chunk": RAVEN_CHUNK_FRAMES,
        "policy_forward_population": (
            RAVEN_TRAIN_CHUNKS_PER_UPDATE * RAVEN_CHUNK_FRAMES
        ),
        "drop_last": True,
        "training_chunk_count": len(train_data),
        "updates_per_epoch": updates_per_epoch,
        "index_safe_max_pools": index_safe_max_pools,
        "activation_checkpointing": True,
    }
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    stopper = EarlyStopping(
        min_epochs=args.min_epochs,
        patience=args.patience,
        relative_improvement=args.relative_improvement,
        absolute_improvement=args.absolute_improvement,
    )
    history: list[dict] = []
    start_epoch = 0
    global_step = 0
    resume_checkpoint = None
    resume_rng_mode = "fresh_run"
    resume_checkpoint_sha256 = None
    resume_state_path = None
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        resume_state_path = (
            args.resume_validation_state.resolve()
            if args.resume_validation_state is not None
            else infer_resume_validation_state(resume_path).resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {resume_path}")
        resume_checkpoint, history, resume_rng_mode = load_resume_state(
            resume_path,
            resume_state_path,
            policy=policy,
            optimizer=optimizer,
            train_generator=generator,
            spec=spec,
            bootstrap_sha256=sha256_file(args.bootstrap.resolve()),
            train_split_sha256=sha256_file(args.train_split.resolve()),
            validation_split_sha256=sha256_file(args.validation_split.resolve()),
            stopper=stopper,
            seed=args.seed,
            expected_learning_rate=args.learning_rate,
            expected_training_batch_contract=training_batch_contract,
            expected_data_admission=data_admission,
            expected_sampling_contract=sampling_contract,
        )
        start_epoch = int(resume_checkpoint["epoch"])
        global_step = int(resume_checkpoint["global_step"])
        resume_checkpoint_sha256 = sha256_file(resume_path)
        previous_fit_contract = resume_checkpoint.get("training_mse_stop_contract")
        validate_fitting_stop_resume(previous_fit_contract, fitting_stop_contract)
        if args.max_epochs is not None and args.max_epochs <= start_epoch:
            raise ValueError(
                f"--max-epochs ({args.max_epochs}) must exceed resumed epoch ({start_epoch})"
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"bc_3view_{timestamp}"
    output = (args.output_root / run_name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    validation_checkpoints = output / "validation_checkpoints"
    validation_checkpoints.mkdir()
    metadata = {
        "schema": "roborender.bc_student_checkpoint.v3",
        "bootstrap": str(args.bootstrap.resolve()),
        "bootstrap_sha256": sha256_file(args.bootstrap.resolve()),
        "policy_spec": spec,
        "train_split": str(args.train_split.resolve()),
        "train_split_sha256": sha256_file(args.train_split.resolve()),
        "validation_split": str(args.validation_split.resolve()),
        "validation_split_sha256": sha256_file(args.validation_split.resolve()),
        "selection": "minimum complete-validation MSE",
        "data_admission": data_admission,
        "training_batch_contract": training_batch_contract,
        "training_sampling_contract": sampling_contract,
        "training_mse_stop_contract": fitting_stop_contract,
        "epoch_budget": {"max_epochs": args.max_epochs, "uncapped": args.max_epochs is None},
        "resume": {
            "checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
            "checkpoint_sha256": resume_checkpoint_sha256,
            "validation_state": str(resume_state_path) if resume_state_path else None,
            "start_epoch": start_epoch,
            "start_global_step": global_step,
            "rng_restore_mode": resume_rng_mode,
        },
    }
    config = {**vars(args), "output": output, "device": torch.cuda.get_device_name(0)}
    atomic_json(output / "run_config.json", {key: str(value) if isinstance(value, Path) else value for key, value in config.items()})

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )
        atomic_json(output / "wandb-run.json", {"id": wandb_run.id, "url": wandb_run.url,
                    "project": args.wandb_project, "name": args.wandb_run_name, "mode": args.wandb_mode})
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("validation/*", step_metric="epoch")

    started = time.time()
    initial_mse, initial_per_action = validate(policy, validation_loader, device, mean, std, action_dim)
    training_eval_mse = (validate(policy, train_eval_loader, device, mean, std, action_dim)[0]
                        if train_eval_loader is not None else None)
    if resume_checkpoint is None:
        stopper.update(0, initial_mse)
    elif not math.isclose(
        initial_mse, float(resume_checkpoint["validation_mse"]),
        rel_tol=1e-5, abs_tol=1e-7,
    ):
        raise RuntimeError(
            "resumed policy validation MSE does not reproduce its checkpoint: "
            f"stored={resume_checkpoint['validation_mse']} observed={initial_mse}"
        )
    initial_checkpoint = validation_checkpoints / (
        f"epoch_{start_epoch:04d}_step_{global_step:06d}.pt"
    )
    if resume_checkpoint is None:
        history.append({
            "epoch": 0,
            "global_step": 0,
            "mse": initial_mse,
            "per_action_mse": initial_per_action,
            "improved": True,
            "meaningful_improvement": True,
            "checkpoint": str(initial_checkpoint),
        })
    # Relocate the records carried into this run, so a subsequent resume does
    # not depend on the old directory for the selected or starting checkpoint.
    history[-1]["checkpoint"] = str(initial_checkpoint)
    if training_eval_mse is not None:
        history[-1]["training_eval_mse"] = training_eval_mse
    selected_checkpoint = initial_checkpoint
    if stopper.best_epoch != start_epoch:
        best_record = next(
            record for record in history if int(record["epoch"]) == stopper.best_epoch
        )
        source_best = Path(best_record["checkpoint"]).resolve()
        if not source_best.is_file():
            raise FileNotFoundError(f"prior best validation checkpoint is missing: {source_best}")
        imported_best = validation_checkpoints / source_best.name
        update_checkpoint_alias(source_best, imported_best)
        best_record["checkpoint"] = str(imported_best)
        selected_checkpoint = imported_best
    save_checkpoint(
        initial_checkpoint, policy, optimizer, start_epoch, global_step,
        initial_mse, initial_per_action,
        {**metadata, "early_stopping_state": stopper.state_dict(),
         "validation_history": history}, generator,
    )
    update_checkpoint_alias(selected_checkpoint, output / "best_validation.pt")
    update_checkpoint_alias(initial_checkpoint, output / "last.pt")
    state_identity = {key: metadata[key] for key in (
        "bootstrap_sha256", "train_split_sha256", "validation_split_sha256",
    )}
    training_epoch_mse = history[-1].get("train_mse")
    stop_measurement = (training_epoch_mse if args.stop_training_epoch_mse_below is not None
                        else training_eval_mse)
    # Fresh epoch zero has no training average. It cannot meet a fitting goal.
    joint_stop = (False if fitting_target is not None and stop_measurement is None
                  else should_stop_for_plateau(stopper.is_saturated(start_epoch),
                                               fitting_target, stop_measurement))
    atomic_json(output / "validation_state.json", {
        "status": "saturated" if joint_stop else "running",
        "training_mse_stop_contract": fitting_stop_contract,
        "training_eval_mse": training_eval_mse,
        "training_epoch_mse": training_epoch_mse,
        "validation_plateau": stopper.is_saturated(start_epoch),
        "best_mse": stopper.best_mse, "best_epoch": stopper.best_epoch,
        "bad_validations": stopper.bad_validations,
        "early_stopping_state": stopper.state_dict(),
        "history": history, **state_identity,
    })
    if wandb_run is not None:
        baseline_metrics = {
            "epoch": start_epoch,
            "global_step": global_step,
            "validation/mse": initial_mse,
            "validation/best_mse": stopper.best_mse,
            "validation/bad_checks": stopper.bad_validations,
            "validation/improved": int(stopper.best_epoch == start_epoch),
            "validation/saturated": int(joint_stop),
            "validation/plateau": int(stopper.is_saturated(start_epoch)),
            "validation/resumed": int(resume_checkpoint is not None),
        }
        baseline_metrics.update(
            {f"validation/mse_action_{index:02d}": value for index, value in enumerate(initial_per_action)}
        )
        if training_eval_mse is not None:
            baseline_metrics.update({"train/eval_mse": training_eval_mse,
                "train/eval_target_met": int(training_target_met(args.stop_training_eval_mse_below, training_eval_mse))})
        if args.stop_training_epoch_mse_below is not None and training_epoch_mse is not None:
            baseline_metrics.update({"train/mse_epoch": training_epoch_mse,
                "train/epoch_target_met": int(training_target_met(fitting_target, training_epoch_mse))})
        wandb_run.log(baseline_metrics)
    print(
        f"validation epoch={start_epoch} mse={initial_mse:.8f} "
        f"resume={resume_checkpoint is not None} rng={resume_rng_mode} "
        f"training_eval_mse={training_eval_mse}",
        flush=True,
    )

    plateau_stop_reason = ("validation_saturation_and_training_target"
                          if fitting_target is not None else "validation_saturation")
    stop_reason = plateau_stop_reason if joint_stop else ("max_epochs" if args.max_epochs is not None else None)
    epochs = () if joint_stop else epoch_sequence(start_epoch, args.max_epochs)
    for epoch in epochs:
        policy.train()
        total_squared = 0.0
        total_elements = 0
        for batch in train_loader:
            vision, actor, target = prepare_batch(batch, device, mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss_value = backward_raven_bc_mse(policy, vision, actor, target)
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite training loss at step {global_step}")
            torch.nn.utils.clip_grad_norm_(
                trainable, args.max_grad_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            elements = target.numel()
            total_squared += loss_value * elements
            total_elements += elements
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/mse_step": loss_value, "global_step": global_step})
        train_mse = total_squared / total_elements
        validation_mse, per_action = validate(policy, validation_loader, device, mean, std, action_dim)
        training_eval_mse = (validate(policy, train_eval_loader, device, mean, std, action_dim)[0]
                            if train_eval_loader is not None else None)
        meaningful, plateau = stopper.update(epoch, validation_mse)
        training_epoch_mse = train_mse
        stop_measurement = train_mse if args.stop_training_epoch_mse_below is not None else training_eval_mse
        saturated = should_stop_for_plateau(plateau, fitting_target, stop_measurement)
        improved = stopper.best_epoch == epoch
        validation_checkpoint = validation_checkpoints / (
            f"epoch_{epoch:04d}_step_{global_step:06d}.pt"
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_mse": train_mse,
            "training_eval_mse": training_eval_mse,
            "training_epoch_mse": train_mse,
            "validation_plateau": plateau,
            "mse": validation_mse,
            "per_action_mse": per_action,
            "improved": improved,
            "meaningful_improvement": meaningful,
            "bad_validations": stopper.bad_validations,
            "saturated": saturated,
            "checkpoint": str(validation_checkpoint),
        }
        history.append(record)
        save_checkpoint(
            validation_checkpoint,
            policy,
            optimizer,
            epoch,
            global_step,
            validation_mse,
            per_action,
            {**metadata, "early_stopping_state": stopper.state_dict(),
             "validation_history": history},
            generator,
        )
        if improved:
            update_checkpoint_alias(validation_checkpoint, output / "best_validation.pt")
        update_checkpoint_alias(validation_checkpoint, output / "last.pt")
        state = {
            "status": "saturated" if saturated else "running",
            "best_mse": stopper.best_mse,
            "best_epoch": stopper.best_epoch,
            "bad_validations": stopper.bad_validations,
            "early_stopping_state": stopper.state_dict(),
            "training_mse_stop_contract": fitting_stop_contract,
            "training_eval_mse": training_eval_mse,
            "validation_plateau": plateau,
            "history": history,
            **state_identity,
        }
        atomic_json(output / "validation_state.json", state)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train/mse": train_mse,
            "train/mse_epoch": train_mse,
            "validation/mse": validation_mse,
            "validation/best_mse": stopper.best_mse,
            "validation/bad_checks": stopper.bad_validations,
            "validation/improved": int(improved),
            "validation/meaningful_improvement": int(meaningful),
            "validation/saturated": int(saturated),
            "validation/plateau": int(plateau),
        }
        if training_eval_mse is not None:
            metrics.update({"train/eval_mse": training_eval_mse,
                "train/eval_target_met": int(training_target_met(args.stop_training_eval_mse_below, training_eval_mse))})
        if args.stop_training_epoch_mse_below is not None:
            metrics["train/epoch_target_met"] = int(training_target_met(fitting_target, train_mse))
        metrics.update({f"validation/mse_action_{index:02d}": value for index, value in enumerate(per_action)})
        if wandb_run is not None:
            wandb_run.log(metrics)
        print(
            f"epoch={epoch} train_mse={train_mse:.8f} validation_mse={validation_mse:.8f} "
            f"best={stopper.best_mse:.8f} patience={stopper.bad_validations}/{stopper.patience}",
            flush=True,
        )
        if training_eval_mse is not None:
            print(f"epoch={epoch} training_eval_mse={training_eval_mse:.8f} "
                  f"target_below={args.stop_training_eval_mse_below} plateau={plateau} joint_stop={saturated}", flush=True)
        if saturated:
            stop_reason = plateau_stop_reason
            break

    final_state = {
        "status": "complete",
        "stop_reason": stop_reason,
        "best_mse": stopper.best_mse,
        "best_epoch": stopper.best_epoch,
        "bad_validations": stopper.bad_validations,
        "early_stopping_state": stopper.state_dict(),
        "global_step": global_step,
        "elapsed_seconds": time.time() - started,
        "selected_checkpoint": str(output / "best_validation.pt"),
        "training_mse_stop_contract": fitting_stop_contract,
        "training_eval_mse": training_eval_mse,
        "training_epoch_mse": training_epoch_mse,
        "history": history,
        **state_identity,
    }
    atomic_json(output / "validation_state.json", final_state)
    if wandb_run is not None:
        wandb_run.summary.update(final_state)
        wandb_run.finish()
    print(json.dumps({"output": str(output), **{key: final_state[key] for key in ("stop_reason", "best_mse", "best_epoch", "global_step")}}, sort_keys=True))


if __name__ == "__main__":
    main()
