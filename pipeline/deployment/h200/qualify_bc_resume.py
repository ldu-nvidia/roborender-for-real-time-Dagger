"""One disposable BC resume update plus sampled validation; no production writes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from pipeline.bc_training import train_bc_h200 as bc

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "checkpoints/students/BC/bc_3view_autoregressive_v1_20260914T044707Z/best_validation.pt"
CHECKPOINT_SHA = "053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718"


def optimizer_steps(optimizer):
    return sorted(int(value["step"].item()) for value in optimizer.state.values() if "step" in value)


def save_smoke_checkpoint(path, policy, optimizer, parent, spec, generator, mse, per_action, **extra):
    """Keep the real policy/optimizer contract across BC -> DAgger smoke.

    A single update is NOT a completed BC epoch or a new best-validation model.
    Diagnostic MSE here is explicitly sampled, not the success-rate metric.
    """
    metadata = {key: parent[key] for key in (
        "bootstrap", "bootstrap_sha256", "train_split", "train_split_sha256",
        "validation_split", "validation_split_sha256", "training_batch_contract") if key in parent}
    metadata.update(schema="roborender.bc_student_checkpoint.v2", policy_spec=spec,
        source_checkpoint_sha256=CHECKPOINT_SHA, source_epoch=parent["epoch"],
        source_global_step=parent["global_step"], qualification_optimizer_steps=1,
        production_checkpoint=False, sampled_validation_only=True, completed_epoch=False,
        selection="disposable smoke update, not selected by validation", **extra)
    bc.save_checkpoint(path, policy, optimizer, int(parent["epoch"]), int(parent["global_step"]) + 1,
                       mse, per_action, metadata, generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runtime/migrations/alpha-to-beta-20261008/h200"):
        raise RuntimeError("qualification output must remain in the isolated migration directory")
    output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("this test requires exactly one allocated SM90 GPU")
    if bc.sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("BC source checkpoint differs from the pinned best-validation bytes")
    device = torch.device("cuda:0")
    dataset_root = ROOT / "data/BC"
    train_split = dataset_root / "splits_autoregressive_v1/train.json"
    val_split = dataset_root / "splits_autoregressive_v1/validation.json"
    train_index, train_paths = bc.load_split(train_split, "train", dataset_root)
    val_index, val_paths = bc.load_split(val_split, "validation", dataset_root)
    if {x["stream_id"] for x in train_index["items"]} & {x["stream_id"] for x in val_index["items"]}:
        raise RuntimeError("BC split streams overlap")
    bootstrap = ROOT / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt"
    spec_path = bootstrap.with_suffix(".policy_spec.json")
    policy, spec = bc.build_policy(spec_path, bootstrap, device)
    safe_pools = bc.install_index_safe_max_pool2d(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(42)
    contract = {
        "schema": bc.RAVEN_BATCHING_CONTRACT, "chunks_per_update": 16,
        "frames_per_chunk": 81, "policy_forward_population": 16 * 81,
        "drop_last": True, "training_chunk_count": len(train_paths),
        "updates_per_epoch": bc.validate_training_batch_population(len(train_paths), 16),
        "index_safe_max_pools": safe_pools, "activation_checkpointing": True,
    }
    restored, history, rng_mode = bc.load_resume_state(
        CHECKPOINT, CHECKPOINT.parent / "validation_state.json", policy=policy,
        optimizer=optimizer, train_generator=generator, spec=spec,
        bootstrap_sha256=bc.sha256_file(bootstrap), train_split_sha256=bc.sha256_file(train_split),
        validation_split_sha256=bc.sha256_file(val_split), stopper=bc.EarlyStopping(),
        seed=42, expected_learning_rate=2e-4, expected_training_batch_contract=contract,
    )
    restored_steps = optimizer_steps(optimizer)
    if not restored_steps or min(restored_steps) <= 0:
        raise RuntimeError("AdamW state was not restored")
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    selected = torch.randperm(len(train_paths), generator=generator)[:16].tolist()
    train_loader = DataLoader(bc.BCChunks([train_paths[i] for i in selected], actor_dim, action_dim), batch_size=16)
    val_loader = DataLoader(bc.BCChunks(val_paths[:4], actor_dim, action_dim), batch_size=4)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    before, _ = bc.validate(policy, val_loader, device, mean, std, action_dim)
    policy.train()
    batch = bc.prepare_batch(next(iter(train_loader)), device, mean, std)
    initial_parameter = trainable[-1].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    mse = bc.backward_raven_bc_mse(policy, *batch)
    gradient = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    if not math.isfinite(mse) or not math.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("invalid BC loss or gradients")
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.monotonic() - started
    if optimizer_steps(optimizer) != [x + 1 for x in restored_steps]:
        raise RuntimeError("AdamW counters did not advance exactly once")
    if torch.equal(initial_parameter, trainable[-1].detach()):
        raise RuntimeError("trained policy parameter did not change")
    del batch
    after, per_action = bc.validate(policy, val_loader, device, mean, std, action_dim)
    if not math.isfinite(before) or not math.isfinite(after):
        raise RuntimeError("invalid sampled BC validation")
    candidate = output / "disposable_candidate.pt"
    save_smoke_checkpoint(candidate, policy, optimizer, restored, spec, generator, after, per_action)
    report = {
        "schema": "roborender.migration.bc-resume-smoke.v1", "result": "PASS",
        "source_checkpoint_sha256": CHECKPOINT_SHA, "source_epoch": restored["epoch"],
        "source_global_step": restored["global_step"], "rng_restore_mode": rng_mode,
        "optimizer_steps": 1, "training_chunks": 16, "frames_per_chunk": 81,
        "train_mse": mse, "grad_norm": gradient, "update_seconds": update_seconds,
        "validation_chunks": 4, "sampled_validation_mse_before": before,
        "sampled_validation_mse_after": after, "sampled_validation_per_action": per_action,
        "full_validation_tested": False, "success_rate_tested": False,
        "candidate_sha256": bc.sha256_file(candidate), "production_promoted": False,
        "train_sample_paths": [str(train_paths[i]) for i in selected],
        "validation_sample_paths": [str(p) for p in val_paths[:4]],
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
