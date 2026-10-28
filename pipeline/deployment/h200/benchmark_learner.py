"""Disposable real-data learner throughput trial; never promotes a checkpoint.

Uses the same full B*T forward/backward as DAgger. Changing B is an explicitly
labeled optimization-semantics experiment, not a quality-equivalent shortcut.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader
from pipeline.bc_training import train_bc_h200 as bc
from pipeline.dagger.services import raven_learner_worker as learner
from pipeline.deployment.h200.migration_smoke_control import CHECKPOINT, CHECKPOINT_SHA, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-chunks", type=int, choices=[8, 16, 28, 32, 56], required=True)
    parser.add_argument("--checkpoint-activations", choices=["on", "off"], default="on")
    parser.add_argument("--repeats", type=int, choices=[3, 5], default=3)
    parser.add_argument("--co-resident-renderer", action="store_true",
                        help="record that an idle Image3F worker remains resident on this GPU")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runtime/migrations/alpha-to-beta-20261008/h200"):
        raise RuntimeError("benchmark must remain in an isolated workload directory")
    output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("scope each learner trial to exactly one H200")
    report = {"batch_chunks": args.batch_chunks, "frames_per_chunk": 81,
              "activation_checkpointing": args.checkpoint_activations,
              "checkpoint_sha256": CHECKPOINT_SHA, "data_source": "corrected_BC_train_only",
              "full_B_times_T_forward": True, "production_promoted": False,
              "quality_equivalence_claimed": False, "measurements": []}
    report["co_resident_idle_image3f"] = args.co_resident_renderer
    started = time.monotonic()
    try:
        if bc.sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
            raise RuntimeError("source checkpoint identity mismatch")
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        spec = checkpoint["policy_spec"]
        device = torch.device("cuda:0")
        torch.manual_seed(610091)
        policy = learner.build_policy(spec, checkpoint["policy_state_dict"], device)
        learner.install_index_safe_max_pool2d(policy)
        parameters = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0.0)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = 2e-4
        report["initialization_seconds"] = time.monotonic() - started
        dataset_root = ROOT / "data/BC"
        _, paths = bc.load_split(dataset_root / "splits_autoregressive_v1/train.json", "train", dataset_root)
        chosen = paths[:args.batch_chunks]
        if len(chosen) != args.batch_chunks:
            raise RuntimeError("not enough unique training chunks")
        load_started = time.monotonic()
        loader = DataLoader(bc.BCChunks(chosen, spec["input_contract"]["actor_obs_dim"],
                                      spec["output_contract"]["policy_action_dim"]), batch_size=args.batch_chunks)
        mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1,1,1,1,3)
        std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1,1,1,1,3)
        vision, actor, target = bc.prepare_batch(next(iter(loader)), device, mean, std)
        torch.cuda.synchronize()
        report["data_load_seconds"] = time.monotonic() - load_started
        report["training_paths"] = [str(p) for p in chosen]
        policy.train()
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.repeats + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_start = time.monotonic()
            if args.checkpoint_activations == "on":
                mse = learner.backward_global_mse_in_environment_microbatches(
                    policy, vision, actor, target, microbatch_environments=args.batch_chunks,
                    full_environment_batch_per_timestep=True)
            else:
                predicted = policy({"vision_obs": vision, "actor_obs": actor})["actions"]
                loss = torch.nn.functional.mse_loss(predicted, target)
                loss.backward()
                mse = float(loss.detach())
                del predicted, loss
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            if not math.isfinite(mse) or not math.isfinite(gradient) or gradient <= 0:
                raise RuntimeError("invalid loss/gradient")
            optimizer.step()
            torch.cuda.synchronize()
            row = {"index": index, "warmup": index == 0, "seconds": time.monotonic()-step_start,
                   "mse": mse, "grad_norm": gradient}
            report["measurements"].append(row)
            print(json.dumps(row), flush=True)
        save_start = time.monotonic()
        torch.save({"policy_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "source_checkpoint_sha256": CHECKPOINT_SHA, "throughput_benchmark_only": True}, output / "disposable.pt")
        report["checkpoint_seconds"] = time.monotonic()-save_start
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        measured = [r["seconds"] for r in report["measurements"] if not r["warmup"]]
        report["median_update_seconds"] = statistics.median(measured)
        report["training_frames_per_second"] = args.batch_chunks*81/report["median_update_seconds"]
        report["status"] = "PASS"
    except torch.OutOfMemoryError as exc:
        report.update(status="OOM", error=str(exc))
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
    report["total_seconds"] = time.monotonic()-started
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
