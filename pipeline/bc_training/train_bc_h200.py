#!/usr/bin/env python3
"""Native H200 offline BC training; intentionally has no Isaac/renderer imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


VIEWS = ("head_color", "hand_left", "hand_right")
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclass
class EarlyStopping:
    min_epochs: int = 10
    patience: int = 8
    relative_improvement: float = 0.005
    absolute_improvement: float = 1e-6
    best_mse: float = math.inf
    bad_validations: int = 0
    best_epoch: int = -1

    def update(self, epoch: int, mse: float) -> tuple[bool, bool]:
        if not math.isfinite(mse):
            raise ValueError(f"validation MSE is not finite: {mse}")
        threshold = (
            0.0
            if not math.isfinite(self.best_mse)
            else max(self.absolute_improvement, self.relative_improvement * self.best_mse)
        )
        improved = not math.isfinite(self.best_mse) or mse < self.best_mse - threshold
        if improved:
            self.best_mse = mse
            self.best_epoch = epoch
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        saturated = epoch >= self.min_epochs and self.bad_validations >= self.patience
        return improved, saturated


def load_split(path: Path, expected: str, dataset_root: Path) -> tuple[dict, list[Path]]:
    payload = json.loads(path.read_text())
    if payload.get("split") != expected:
        raise ValueError(f"{path}: expected split={expected!r}")
    items = payload.get("items", [])
    if payload.get("chunk_count") != len(items) or not items:
        raise ValueError(f"{path}: invalid or empty chunk index")
    paths = [(dataset_root / item["rendered_npz"]).resolve() for item in items]
    missing = [str(chunk) for chunk in paths if not chunk.is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing rendered chunks, first={missing[0]}")
    source_manifest = (path.parent / payload["source_manifest"]).resolve()
    if sha256_file(source_manifest) != payload["source_manifest_sha256"]:
        raise ValueError(f"{path}: source manifest hash mismatch")
    return payload, paths


class BCChunks(Dataset):
    """One item is Raven's complete synchronized 81-frame chunk."""

    def __init__(self, paths: list[Path], actor_dim: int, action_dim: int):
        self.paths = paths
        self.actor_dim = actor_dim
        self.action_dim = action_dim

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as chunk:
            missing = [key for key in LEARNER_KEYS if key not in chunk.files]
            if missing:
                raise ValueError(f"{path}: missing learner fields {missing}")
            views = [chunk[f"rgb_gen__{view}"] for view in VIEWS]
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
            return {
                "vision": torch.from_numpy(vision.copy()),
                "actor": torch.from_numpy(actor.copy()),
                "target": torch.from_numpy(target.copy()),
                "chunk": str(path),
            }


def build_policy(spec_path: Path, checkpoint_path: Path, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

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


@torch.inference_mode()
def validate(policy, loader, device, mean, std, action_dim: int) -> tuple[float, list[float]]:
    policy.eval()
    squared = torch.zeros(action_dim, dtype=torch.float64, device=device)
    frames = 0
    for batch in loader:
        vision, actor, target = prepare_batch(batch, device, mean, std)
        prediction = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
        error = (prediction - target).double().square()
        squared += error.sum(dim=(0, 1))
        frames += target.shape[0] * target.shape[1]
    per_action = squared / frames
    return float(per_action.mean().item()), [float(value) for value in per_action.cpu()]


def save_checkpoint(
    path: Path,
    policy,
    optimizer,
    epoch: int,
    global_step: int,
    mse: float,
    per_action_mse: list[float],
    metadata: dict,
) -> None:
    payload = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_mse": mse,
        "validation_per_action_mse": per_action_mse,
        **metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def update_checkpoint_alias(source: Path, alias: Path) -> None:
    """Atomically point an alias at an immutable validation checkpoint."""
    temporary = alias.with_suffix(alias.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, alias)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=root / "data/BC")
    parser.add_argument("--train-split", type=Path, default=root / "data/BC/splits/train.json")
    parser.add_argument("--validation-split", type=Path, default=root / "data/BC/splits/validation.json")
    parser.add_argument("--bootstrap", type=Path, default=root / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt")
    parser.add_argument("--policy-spec", type=Path, default=root / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json")
    parser.add_argument("--output-root", type=Path, default=root / "checkpoints/students/BC")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--validation-chunk-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--relative-improvement", type=float, default=0.005)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="dagger")
    parser.add_argument("--wandb-group", default="BC")
    parser.add_argument("--wandb-run-name", default="BC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BC training requires a CUDA H200")
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError(f"expected H200 SM90, got {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
    if args.min_epochs < 1 or args.patience < 1 or args.max_epochs < args.min_epochs:
        raise ValueError("invalid early-stopping epoch/patience settings")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    train_index, train_paths = load_split(args.train_split.resolve(), "train", args.dataset_root.resolve())
    validation_index, validation_paths = load_split(
        args.validation_split.resolve(), "validation", args.dataset_root.resolve()
    )
    train_streams = {item["stream_id"] for item in train_index["items"]}
    validation_streams = {item["stream_id"] for item in validation_index["items"]}
    if train_streams & validation_streams:
        raise ValueError("training and validation streams overlap")

    policy, spec = build_policy(args.policy_spec.resolve(), args.bootstrap.resolve(), device)
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    train_data = BCChunks(train_paths, actor_dim, action_dim)
    validation_data = BCChunks(validation_paths, actor_dim, action_dim)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.chunk_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.validation_chunk_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"bc_3view_{timestamp}"
    output = (args.output_root / run_name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    validation_checkpoints = output / "validation_checkpoints"
    validation_checkpoints.mkdir()
    metadata = {
        "schema": "roborender.bc_student_checkpoint.v1",
        "bootstrap": str(args.bootstrap.resolve()),
        "bootstrap_sha256": sha256_file(args.bootstrap.resolve()),
        "policy_spec": spec,
        "train_split": str(args.train_split.resolve()),
        "train_split_sha256": sha256_file(args.train_split.resolve()),
        "validation_split": str(args.validation_split.resolve()),
        "validation_split_sha256": sha256_file(args.validation_split.resolve()),
        "selection": "minimum complete-validation MSE",
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
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("validation/*", step_metric="epoch")

    stopper = EarlyStopping(
        min_epochs=args.min_epochs,
        patience=args.patience,
        relative_improvement=args.relative_improvement,
    )
    history: list[dict] = []
    global_step = 0
    started = time.time()
    initial_mse, initial_per_action = validate(policy, validation_loader, device, mean, std, action_dim)
    stopper.update(0, initial_mse)
    initial_checkpoint = validation_checkpoints / "epoch_0000_step_000000.pt"
    initial = {
        "epoch": 0,
        "global_step": 0,
        "mse": initial_mse,
        "per_action_mse": initial_per_action,
        "improved": True,
        "checkpoint": str(initial_checkpoint),
    }
    history.append(initial)
    save_checkpoint(
        initial_checkpoint, policy, optimizer, 0, 0, initial_mse, initial_per_action, metadata
    )
    update_checkpoint_alias(initial_checkpoint, output / "best_validation.pt")
    update_checkpoint_alias(initial_checkpoint, output / "last.pt")
    if wandb_run is not None:
        baseline_metrics = {
            "epoch": 0,
            "global_step": 0,
            "validation/mse": initial_mse,
            "validation/best_mse": initial_mse,
            "validation/bad_checks": 0,
            "validation/improved": 1,
            "validation/saturated": 0,
        }
        baseline_metrics.update(
            {f"validation/mse_action_{index:02d}": value for index, value in enumerate(initial_per_action)}
        )
        wandb_run.log(baseline_metrics)
    print(f"validation epoch=0 mse={initial_mse:.8f}", flush=True)

    stop_reason = "max_epochs"
    for epoch in range(1, args.max_epochs + 1):
        policy.train()
        total_squared = 0.0
        total_elements = 0
        for batch in train_loader:
            vision, actor, target = prepare_batch(batch, device, mean, std)
            optimizer.zero_grad(set_to_none=True)
            prediction = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            elements = target.numel()
            total_squared += float(loss.detach()) * elements
            total_elements += elements
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/mse_step": float(loss.detach()), "global_step": global_step})
        train_mse = total_squared / total_elements
        validation_mse, per_action = validate(policy, validation_loader, device, mean, std, action_dim)
        improved, saturated = stopper.update(epoch, validation_mse)
        validation_checkpoint = validation_checkpoints / (
            f"epoch_{epoch:04d}_step_{global_step:06d}.pt"
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_mse": train_mse,
            "mse": validation_mse,
            "per_action_mse": per_action,
            "improved": improved,
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
            metadata,
        )
        if improved:
            update_checkpoint_alias(validation_checkpoint, output / "best_validation.pt")
        update_checkpoint_alias(validation_checkpoint, output / "last.pt")
        state = {
            "status": "saturated" if saturated else "running",
            "best_mse": stopper.best_mse,
            "best_epoch": stopper.best_epoch,
            "bad_validations": stopper.bad_validations,
            "history": history,
        }
        atomic_json(output / "validation_state.json", state)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train/mse": train_mse,
            "validation/mse": validation_mse,
            "validation/best_mse": stopper.best_mse,
            "validation/bad_checks": stopper.bad_validations,
            "validation/improved": int(improved),
            "validation/saturated": int(saturated),
        }
        metrics.update({f"validation/mse_action_{index:02d}": value for index, value in enumerate(per_action)})
        if wandb_run is not None:
            wandb_run.log(metrics)
        print(
            f"epoch={epoch} train_mse={train_mse:.8f} validation_mse={validation_mse:.8f} "
            f"best={stopper.best_mse:.8f} patience={stopper.bad_validations}/{stopper.patience}",
            flush=True,
        )
        if saturated:
            stop_reason = "validation_saturation"
            break

    final_state = {
        "status": "complete",
        "stop_reason": stop_reason,
        "best_mse": stopper.best_mse,
        "best_epoch": stopper.best_epoch,
        "bad_validations": stopper.bad_validations,
        "global_step": global_step,
        "elapsed_seconds": time.time() - started,
        "selected_checkpoint": str(output / "best_validation.pt"),
        "history": history,
    }
    atomic_json(output / "validation_state.json", final_state)
    if wandb_run is not None:
        wandb_run.summary.update(final_state)
        wandb_run.finish()
    print(json.dumps({"output": str(output), **{key: final_state[key] for key in ("stop_reason", "best_mse", "best_epoch", "global_step")}}, sort_keys=True))


if __name__ == "__main__":
    main()
