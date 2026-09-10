#!/usr/bin/env python3
"""One Raven-equivalent supervised update over barrier-admitted DAgger chunks."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The learner service launches this executable by absolute path under Raven's
# hardened PYTHONSAFEPATH runtime.  Add only this file's audited project root;
# do not depend on the caller's working directory or ambient PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.services.contracts import (
    CANONICAL_BETA,
    canonical_json,
    file_sha256,
    validate_dagger_chunk,
)
from pipeline.dagger.services.phase_telemetry import PhaseTelemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args()


def build_policy(spec: dict, state: dict, device: torch.device):
    # Keep the service/test control plane importable without the heavyweight
    # GR00T runtime.  The worker imports these only when it constructs a model.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

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


def chunk_paths(state_root: Path, chunk_ids: list[str], *,
                deep_validation_samples: int, sample_seed: str) -> list[Path]:
    database = sqlite3.connect(state_root / "state.sqlite3")
    database.row_factory = sqlite3.Row
    try:
        paths = []
        for chunk_id in chunk_ids:
            row = database.execute(
                "SELECT path,status,size_bytes,sha256 FROM chunk_reservations WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if row is None or row["status"] != "committed":
                raise RuntimeError(f"chunk is not committed: {chunk_id}")
            path = Path(row["path"]).resolve()
            if not path.is_file() or path.stat().st_size != row["size_bytes"]:
                raise RuntimeError(f"chunk is missing or has the wrong size: {chunk_id}")
            paths.append(path)
        sample_count = min(max(0, deep_validation_samples), len(paths))
        sampled = sorted(
            zip(chunk_ids, paths),
            key=lambda pair: hashlib.sha256(
                f"{sample_seed}:{pair[0]}".encode()
            ).hexdigest(),
        )[:sample_count]
        for chunk_id, path in sampled:
            row = database.execute(
                "SELECT sha256 FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"sampled chunk hash failed verification: {chunk_id}")
            validate_dagger_chunk(path)
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
    # Online H200 chunks store both native ray-traced and Image3F observations
    # in Raven's lossless float32 [0,1] policy-view contract.
    vision_tensor = torch.from_numpy(np.stack(vision)).to(device=device, dtype=torch.float32)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    vision_tensor = (vision_tensor - mean) / std
    return (
        vision_tensor,
        torch.from_numpy(np.stack(actor)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(target)).to(device=device, dtype=torch.float32),
    )


def backward_global_mse_in_environment_microbatches(
    policy,
    vision: torch.Tensor,
    actor: torch.Tensor,
    target: torch.Tensor,
    *,
    microbatch_environments: int,
) -> float:
    """Backpropagate one whole-window MSE without materializing one giant CNN graph.

    A Raven rollout tensor is environment-major: ``[B, T, ...]``.  The policy
    internally flattens ``B*T`` images before its CNN, so a B=28, T=81 window
    otherwise asks cuDNN to process 2,268 tall three-view frames in one call.
    Splitting only the environment dimension bounds that working set.  Every
    microbatch's squared-error *sum* is divided by the element count of the
    complete window before backward, so all microbatches together produce one
    global-mean MSE gradient and exactly one later optimizer step.
    """
    if vision.ndim != 5 or actor.ndim != 3 or target.ndim != 3:
        raise ValueError("learner tensors must be environment-major [B,T,...]")
    if not (vision.shape[:2] == actor.shape[:2] == target.shape[:2]):
        raise ValueError("vision, actor, and target B/T dimensions must match")
    if microbatch_environments < 1:
        raise ValueError("microbatch_environments must be positive")

    total_elements = target.numel()
    if total_elements < 1:
        raise ValueError("empty learner target")
    squared_error_sum = torch.zeros((), device=target.device, dtype=torch.float64)
    for start in range(0, target.shape[0], microbatch_environments):
        stop = min(start + microbatch_environments, target.shape[0])
        prediction = policy({
            "vision_obs": vision[start:stop],
            "actor_obs": actor[start:stop],
        })["actions"]
        micro_sum = torch.nn.functional.mse_loss(
            prediction, target[start:stop], reduction="sum"
        )
        if not bool(torch.isfinite(micro_sum)):
            raise RuntimeError("non-finite DAgger training loss")
        (micro_sum / total_elements).backward()
        squared_error_sum += micro_sum.detach().to(torch.float64)
        del prediction, micro_sum
    return float((squared_error_sum / total_elements).item())


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


def save_iteration_checkpoint(path: Path, base: dict, policy, optimizer, scheduler, *,
                              iteration_id: int, global_step: int, train_mse: float,
                              request: dict) -> None:
    output = dict(base)
    output.update({
        "schema": "roborender.dagger_student_checkpoint.v3",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "dagger_iteration_id": iteration_id,
        "global_step": global_step,
        "parent_checkpoint_sha256": request["input_checkpoint_sha256"],
        "dagger_run_id": request["run_id"],
        "dagger_round_id": request["round_id"],
        "dagger_job_id": request["job_id"],
        "dagger_train_chunk_ids": list(request["train_chunk_ids"]),
        "train_mse": train_mse,
        "validation_metric": "raven_goal_reached_rate",
        "validation_status": "pending_closed_loop_evaluation",
        "selection_rule": "maximum_fixed_panel_raven_goal_reached_rate",
    })
    atomic_torch_save(output, path)


def save_inference_sidecar(
    state_root: Path, full_checkpoint_sha256: str, policy
) -> Path:
    """Publish the policy-only bytes consumed by the eight rollout gateways.

    The full learner checkpoint also carries Adam state and training metadata.
    Deserializing all of it eight times merely to obtain ``policy_state_dict``
    made every checkpoint barrier unnecessarily expensive.  This sidecar is
    keyed by the already verified full-checkpoint digest and is published
    before the learner receipt can activate that digest.
    """

    path = state_root.resolve() / "inference_checkpoints" / (
        f"{full_checkpoint_sha256}.pt"
    )
    if not path.exists():
        atomic_torch_save({
            "schema": "roborender.student_inference_sidecar.v1",
            "full_checkpoint_sha256": full_checkpoint_sha256,
            "policy_state_dict": policy.state_dict(),
        }, path)
    return path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("DAgger learner requires one H200 (SM90)")
    request = json.loads(args.job.read_text())
    phase_telemetry = PhaseTelemetry(
        args.result.parent / "phase-timing.jsonl",
        latest_path=args.result.parent / "progress.json",
    )
    atexit.register(phase_telemetry.close)
    phase_identity = {
        "run_id": request["run_id"],
        "round_id": request["round_id"],
    }
    parameters = request["parameters"]
    allowed = {
        "learning_rate", "max_grad_norm", "num_mini_batches", "seed",
        "iteration_id", "wandb_mode", "wandb_project", "wandb_group", "wandb_run_name",
        "wandb_run_id", "wandb_resume", "wandb_extra_config",
        "deep_validation_samples", "microbatch_environments",
    }
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unknown learner parameters: {sorted(unknown)}")
    learning_rate = float(parameters.get("learning_rate", 2e-4))
    max_grad_norm = float(parameters.get("max_grad_norm", 1.0))
    num_mini_batches = int(parameters.get("num_mini_batches", 1))
    seed = int(parameters.get("seed", 42))
    iteration_id = int(parameters.get("iteration_id", 0))
    # Online prepare/commit already enforces the full tensor contract and
    # receipts.  Production should not rescan large immutable chunks before
    # each one-step update.
    deep_validation_samples = int(parameters.get("deep_validation_samples", 0))
    # B=28 is four environment microbatches of seven.  This mirrors Raven's
    # four-way environment minibatch sizing for the same actor while retaining
    # this service contract's single whole-window optimizer update.
    microbatch_environments = int(parameters.get("microbatch_environments", 7))
    if learning_rate <= 0 or max_grad_norm <= 0 or num_mini_batches != 1:
        raise ValueError("Raven DAgger requires positive LR/clip and num_mini_batches=1")
    if iteration_id < 1:
        raise ValueError("iteration_id must be positive")
    if deep_validation_samples < 0:
        raise ValueError("deep_validation_samples must be non-negative")
    if microbatch_environments < 1:
        raise ValueError("microbatch_environments must be positive")
    wandb_extra_config = parameters.get("wandb_extra_config", {})
    if not isinstance(wandb_extra_config, dict):
        raise ValueError("wandb_extra_config must be an object")
    allowed_wandb_extra = {
        "smoke_test", "concurrent_environments", "rollout_steps_per_environment",
        "total_transitions", "rtx_gpu_index", "rtx_gpu_uuid",
        "rtx_collection_config_sha256", "raven_commit", "render_contract_sha256",
        "image3f_adapter_sha256", "image3f_engine_sha256",
        "student_action_probability", "teacher_action_probability",
        "renderer_worker_count",
    }
    unknown_wandb_extra = set(wandb_extra_config) - allowed_wandb_extra
    if unknown_wandb_extra:
        raise ValueError(f"unknown wandb_extra_config fields: {sorted(unknown_wandb_extra)}")
    train_ids = request.get("train_chunk_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("train_chunk_ids must be non-empty")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")

    digest = request["input_checkpoint_sha256"]
    checkpoint_path = args.state_root.resolve() / "checkpoints" / f"{digest}.pt"
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != digest:
        raise RuntimeError("registered input checkpoint is missing or corrupt")
    with phase_telemetry.span(
        **phase_identity, phase="learner_checkpoint_and_model_initialization",
        details={"iteration_id": iteration_id},
    ) as initialization_timing:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        spec = checkpoint.get("policy_spec")
        if not isinstance(spec, dict):
            raise RuntimeError("input checkpoint lacks embedded three-view policy_spec")
        policy = build_policy(spec, checkpoint["policy_state_dict"], device)
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if isinstance(checkpoint.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        if isinstance(checkpoint.get("lr_scheduler_state_dict"), dict):
            scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        initialization_timing["checkpoint_sha256"] = digest

    with phase_telemetry.span(
        **phase_identity, phase="learner_dataset_sample_check_and_load",
        total=len(train_ids),
        details={"iteration_id": iteration_id, "train_chunks": len(train_ids)},
    ) as load_timing:
        train_paths = chunk_paths(
            args.state_root.resolve(), train_ids,
            deep_validation_samples=deep_validation_samples, sample_seed=f"{seed}:train",
        )
        train_vision, train_actor, train_target = load_batch(train_paths, device, spec)
        load_timing["train_frames"] = len(train_paths) * 81

    checkpoint_root = args.result.parent / "candidate_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    wandb_mode = parameters.get("wandb_mode", "disabled")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    wandb_resume = parameters.get("wandb_resume", "allow")
    if wandb_resume not in {"allow", "must", "never", "auto"}:
        raise ValueError("wandb_resume must be allow, must, never, or auto")
    if wandb_mode != "disabled":
        import wandb
        wandb_config = {
            "run_id": request["run_id"], "round_id": request["round_id"],
            "beta": CANONICAL_BETA[request["round_id"]],
            "input_checkpoint_sha256": request["input_checkpoint_sha256"],
            "train_chunks": len(train_ids),
            "frames_per_chunk": 81,
            "train_frames": len(train_ids) * 81,
            "train_chunk_ids": list(train_ids),
            "view_order": ["head_color", "hand_left", "hand_right"],
            "policy_action_dim": 11,
            "teacher_action_dim": 26,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "num_mini_batches": num_mini_batches,
            "microbatch_environments": microbatch_environments,
            "deep_validation_samples": deep_validation_samples,
            "seed": seed,
            "iteration_id": iteration_id,
            "validation_metric": "raven_goal_reached_rate",
            "validation_execution": "external_closed_loop_fixed_panel",
        }
        overlap = set(wandb_config) & set(wandb_extra_config)
        if overlap:
            raise ValueError(f"wandb_extra_config cannot replace canonical fields: {sorted(overlap)}")
        wandb_config.update(wandb_extra_config)
        wandb_run = wandb.init(
            project=parameters.get("wandb_project", "dagger"),
            group=parameters.get("wandb_group", f"dagger_{request['round_id']}"),
            name=parameters.get("wandb_run_name", request["job_id"]),
            id=parameters.get("wandb_run_id"),
            resume=wandb_resume if parameters.get("wandb_run_id") else None,
            mode=wandb_mode,
            config=wandb_config,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    global_step = int(checkpoint.get("global_step", 0))
    gradient_norm = float("nan")
    try:
        train_started = time.perf_counter()
        with phase_telemetry.span(
            **phase_identity, phase="learner_iteration_training",
            details={"iteration_id": iteration_id, "global_step_before": global_step},
        ) as train_timing:
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            train_mse = backward_global_mse_in_environment_microbatches(
                policy,
                train_vision,
                train_actor,
                train_target,
                microbatch_environments=microbatch_environments,
            )
            finite_gradients = True
            nonzero_gradients = False
            for parameter in trainable:
                if parameter.grad is not None:
                    finite_gradients &= bool(torch.isfinite(parameter.grad).all())
                    nonzero_gradients |= bool(torch.count_nonzero(parameter.grad))
            if not finite_gradients or not nonzero_gradients:
                raise RuntimeError("DAgger gradients are non-finite or all zero")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_timing.update({
                "iteration_id": iteration_id, "global_step": global_step,
                "train_mse": train_mse, "gradient_norm": gradient_norm,
                "microbatch_environments": microbatch_environments,
            })
        train_seconds = time.perf_counter() - train_started
        output_path = checkpoint_root / (
            f"iteration_{iteration_id:04d}_step_{global_step:06d}.pt"
        )
        checkpoint_started = time.perf_counter()
        with phase_telemetry.span(
            **phase_identity, phase="learner_candidate_checkpoint",
            details={"iteration_id": iteration_id, "global_step": global_step},
        ) as checkpoint_timing:
            save_iteration_checkpoint(
                output_path, checkpoint, policy, optimizer, scheduler,
                iteration_id=iteration_id, global_step=global_step,
                train_mse=train_mse, request=request,
            )
            checkpoint_timing["checkpoint_path"] = str(output_path.resolve())
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        if wandb_run is not None:
            wandb_run.log({
                "iteration_id": iteration_id, "global_step": global_step,
                "train/iteration_mse": train_mse,
                "train/mse_step": train_mse,
                "train/mse": train_mse,
                "timing/train_iteration_seconds": train_seconds,
                "timing/checkpoint_seconds": checkpoint_seconds,
            })
            wandb_run.summary.update({
                "last_iteration_id": iteration_id,
                "last_train_mse": train_mse,
                "validation_status": "pending_closed_loop_success_evaluation",
            })
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    output_sha256 = file_sha256(output_path)
    inference_sidecar = save_inference_sidecar(
        args.state_root, output_sha256, policy
    )
    receipt = {
        "schema": "roborender.dagger_learner_candidate_receipt.v3",
        "output_checkpoint_path": str(output_path.resolve()),
        "output_checkpoint_sha256": output_sha256,
        "inference_sidecar_path": str(inference_sidecar.resolve()),
        "parent_checkpoint_sha256": digest,
        "train_chunks": len(train_paths),
        "train_frames": len(train_paths) * 81,
        "iteration_id": iteration_id,
        "train_mse": train_mse,
        "validation_metric": "raven_goal_reached_rate",
        "validation_status": "pending_closed_loop_evaluation",
        "gradient_norm_before_clip": gradient_norm,
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(receipt, args.result)
    phase_telemetry.emit(
        **phase_identity,
        phase="learner_job",
        event="end",
        status="completed",
        completed=1,
        total=1,
        details={
            "iteration_id": iteration_id,
            "train_mse": train_mse,
            "validation_status": "pending_closed_loop_evaluation",
        },
    )
    phase_telemetry.close()
    print(canonical_json(receipt), flush=True)


if __name__ == "__main__":
    main()
