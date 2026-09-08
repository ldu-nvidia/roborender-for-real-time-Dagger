#!/usr/bin/env python3
"""One Raven-equivalent supervised update over barrier-admitted DAgger chunks."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pipeline.dagger.services.contracts import canonical_json, file_sha256, validate_dagger_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args()


def build_policy(spec: dict, state: dict, device: torch.device):
    env = OmegaConf.create({
        "robot": spec["env_config"]["robot"],
        "simulator": {"config": {"cameras": {
            "camera_resolutions": [108, 192], "camera_types": [{"rgb": True}],
        }}},
    })
    policy = instantiate(
        OmegaConf.create(spec["actor_config"]),
        env_config=env,
        algo_config=OmegaConf.create(spec["algo_config"]),
        module_dim_dict=spec["module_dim_dict"],
        _recursive_=False,
    )
    policy.load_state_dict(state, strict=True)
    for parameter in policy.obj_pred_mlp.parameters():
        parameter.requires_grad_(False)
    if hasattr(policy, "std"):
        policy.std.requires_grad_(False)
    return policy.to(device)


def chunk_paths(state_root: Path, chunk_ids: list[str]) -> list[Path]:
    database = sqlite3.connect(state_root / "state.sqlite3")
    database.row_factory = sqlite3.Row
    try:
        paths = []
        for chunk_id in chunk_ids:
            row = database.execute(
                "SELECT path,status FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            if row is None or row["status"] != "committed":
                raise RuntimeError(f"chunk is not committed: {chunk_id}")
            path = Path(row["path"]).resolve()
            validate_dagger_chunk(path)
            paths.append(path)
        return paths
    finally:
        database.close()


def load_batch(paths: list[Path], device: torch.device, spec: dict):
    views = spec["input_contract"]["views"]
    vision, actor, target = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as chunk:
            vision.append(np.concatenate([chunk[f"rgb_gen__{view}"] for view in views], axis=1))
            actor.append(chunk["obs_actor_obs"])
            target.append(chunk["gt_actions"][:, : spec["output_contract"]["policy_action_dim"]])
    vision_tensor = torch.from_numpy(np.stack(vision)).to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    vision_tensor = (vision_tensor - mean) / std
    return (
        vision_tensor,
        torch.from_numpy(np.stack(actor)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(target)).to(device=device, dtype=torch.float32),
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("DAgger learner requires one H200 (SM90)")
    request = json.loads(args.job.read_text())
    parameters = request["parameters"]
    unknown = set(parameters) - {"learning_rate", "max_grad_norm", "num_mini_batches", "seed"}
    if unknown:
        raise ValueError(f"unknown learner parameters: {sorted(unknown)}")
    learning_rate = float(parameters.get("learning_rate", 2e-4))
    max_grad_norm = float(parameters.get("max_grad_norm", 1.0))
    num_mini_batches = int(parameters.get("num_mini_batches", 1))
    seed = int(parameters.get("seed", 42))
    if learning_rate <= 0 or max_grad_norm <= 0 or num_mini_batches != 1:
        raise ValueError("Raven DAgger requires positive LR/clip and num_mini_batches=1")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")

    digest = request["input_checkpoint_sha256"]
    checkpoint_path = args.state_root.resolve() / "checkpoints" / f"{digest}.pt"
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != digest:
        raise RuntimeError("registered input checkpoint is missing or corrupt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint.get("policy_spec")
    if not isinstance(spec, dict):
        raise RuntimeError("input checkpoint lacks embedded three-view policy_spec")
    policy = build_policy(spec, checkpoint["policy_state_dict"], device)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    if isinstance(checkpoint.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    paths = chunk_paths(args.state_root.resolve(), request["chunk_ids"])
    vision, actor, target = load_batch(paths, device, spec)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
    loss = torch.nn.functional.mse_loss(prediction, target)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite DAgger training loss")
    loss.backward()
    finite_gradients = True
    nonzero_gradients = False
    for parameter in trainable:
        if parameter.grad is not None:
            finite_gradients &= bool(torch.isfinite(parameter.grad).all())
            nonzero_gradients |= bool(torch.count_nonzero(parameter.grad))
    if not finite_gradients or not nonzero_gradients:
        raise RuntimeError("DAgger gradients are non-finite or all zero")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()

    output_path = args.result.parent / "checkpoint.pt"
    output = dict(checkpoint)
    output.update({
        "schema": "roborender.dagger_student_checkpoint.v1",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(checkpoint.get("global_step", 0)) + 1,
        "parent_checkpoint_sha256": digest,
        "dagger_run_id": request["run_id"],
        "dagger_round_id": request["round_id"],
        "dagger_job_id": request["job_id"],
        "dagger_chunk_ids": list(request["chunk_ids"]),
        "train_mse": float(loss.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
    })
    atomic_torch_save(output, output_path)
    receipt = {
        "schema": "roborender.dagger_learner_receipt.v1",
        "output_checkpoint_path": str(output_path.resolve()),
        "output_checkpoint_sha256": file_sha256(output_path),
        "parent_checkpoint_sha256": digest,
        "chunks": len(paths),
        "frames": len(paths) * 81,
        "train_mse": float(loss.detach()),
        "gradient_norm_before_clip": float(gradient_norm),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(receipt, args.result)
    print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
