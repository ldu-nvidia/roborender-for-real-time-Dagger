"""Render one migrated BC chunk, then test the real BC loader and update.

Run render and train with their respective existing qualified Python runtimes.
The training smoke combines the fresh chunk with fifteen existing train-only
chunks, preserving the real 16x81 BatchNorm/optimizer population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def validate_artifact_paths(source, output, chain_sources=()):
    """Admit only isolated artifacts from the two explicit migration scopes.

    Both paths must remain within the same scope after resolving symlinks.
    In particular, the canonical production BC pool is never an output here.
    """
    roots = [ROOT / "runtime/migrations" / name for name in
             ("alpha-to-beta-20261008", "beta-to-leo-20261013")]
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not source.is_file() or not any(source.is_relative_to(root) and
            output.is_relative_to(root) for root in roots)):
        raise RuntimeError("fresh BC qualification is restricted to one isolated migration scope")
    chain = [Path(p).resolve() for p in chain_sources]
    if len(chain) > 2 or any(not p.is_relative_to(source.parent) or not p.is_file() for p in chain):
        raise RuntimeError("smoke chain must contain at most three adjacent collected files")
    return source, output, chain


def render(source, output, chain_sources=()):
    from pipeline.bc_rendering.render_bc_teacher_worker import load_pipeline, render_chain, sha256_file
    from pipeline.bc_rendering.render_bc_teacher_8gpu import ADAPTER_SHA256, RAVEN_RENDER_PROMPT, build_render_chains

    adapter = ROOT / "renderer/offline_teacher/checkpoints/roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors"
    if sha256_file(adapter) != ADAPTER_SHA256:
        raise RuntimeError("teacher adapter hash mismatch")
    with np.load(source, allow_pickle=False) as z:
        if z["gt_actions"].shape != (81, 26):
            raise RuntimeError("qualification needs one complete native BC chunk")
    config = {"adapter_path": str(adapter), "num_inference_steps": 50, "cfg_scale": 5.0,
              "height": 240, "width": 416, "student_height": 108, "student_width": 192}
    jobs = []
    for index, path in enumerate([source, *chain_sources]):
        with np.load(path, allow_pickle=False) as z:
            if z["gt_actions"].shape != (81, 26):
                raise RuntimeError("all smoke chain members must be complete native chunks")
            env_id, seq = int(z["env_id"].item()), int(z["seq"].item())
            done = bool(np.any(z["dones"]))
        destination = output / ("rendered.npz" if index == 0 else f"rendered-{index:03d}.npz")
        jobs.append({"source_npz": str(path), "output_npz": str(destination), "frame_count": 81,
           "prompt": RAVEN_RENDER_PROMPT.read_text().strip(), "seed": 42,
           "stream_tag": str(path.parent), "env_id": env_id, "global_env_id": env_id,
           "episode_id": seq, "sequence_id": seq, "has_done": done,
           "source_payload_sha256": sha256_file(path)})
    chains = build_render_chains(jobs)
    started = time.monotonic()
    pipe = load_pipeline(config)
    loaded = time.monotonic()
    result = [render_chain(pipe, chain, config) for chain in chains]
    # All original arrays, including control and teacher state/action, must
    # survive byte-for-byte; there is no interpolating or relabeling here.
    for job in jobs:
        with np.load(job["source_npz"], allow_pickle=False) as original, np.load(job["output_npz"], allow_pickle=False) as rendered:
            for key in original.files:
                if not np.array_equal(original[key], rendered[key]):
                    raise RuntimeError(f"source/label synchronization changed: {key}")
            shapes = {k: list(rendered[k].shape) for k in rendered.files if k.startswith("rgb_gen")}
    destination = output / "rendered.npz"
    report = {"result": "PASS", "mode": "fresh_bc_teacher_render", "source": str(source),
              "source_sha256": sha256_file(source), "output": str(destination),
              "output_sha256": sha256_file(destination), "adapter_sha256": ADAPTER_SHA256,
              "all_source_arrays_preserved": True, "renderer_config": config,
              "loading_seconds": loaded - started, "total_seconds": time.monotonic() - started,
              "render_result": result, "generated_shapes": shapes, "production_promoted": False}
    report.update(rendered_chunks=len(jobs), reference_carry_tested=any(len(c["jobs"]) > 1 for c in chains),
                  chain_lengths=[len(c["jobs"]) for c in chains])
    (output / "render-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def train(source, output):
    from torch.utils.data import DataLoader
    from pipeline.bc_training import train_bc_h200 as bc
    from pipeline.deployment.h200.qualify_bc_resume import CHECKPOINT, CHECKPOINT_SHA, optimizer_steps, save_smoke_checkpoint

    if bc.sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("BC source checkpoint changed")
    device = torch.device("cuda:0")
    spec_path = ROOT / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json"
    bootstrap = ROOT / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt"
    policy, spec = bc.build_policy(spec_path, bootstrap, device)
    bc.install_index_safe_max_pool2d(policy)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    policy.load_state_dict(saved["policy_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    steps_before = optimizer_steps(optimizer)
    generator = torch.Generator().manual_seed(610121)
    torch.manual_seed(610121)
    torch.cuda.manual_seed_all(610121)
    actor_dim = int(spec["input_contract"]["actor_obs_dim"])
    action_dim = int(spec["output_contract"]["policy_action_dim"])
    _, existing_train = bc.load_split(ROOT / "data/BC/splits_autoregressive_v1/train.json", "train", ROOT / "data/BC")
    smoke_paths = [source, *existing_train[:15]]
    loader = DataLoader(bc.BCChunks(smoke_paths, actor_dim, action_dim), batch_size=16)
    mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1, 1, 1, 1, 3)
    policy.train()
    batch = bc.prepare_batch(next(iter(loader)), device, mean, std)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    mse = bc.backward_raven_bc_mse(policy, *batch)
    grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
    if not np.isfinite(mse) or not np.isfinite(grad) or grad <= 0:
        raise RuntimeError("invalid fresh-BC loss or gradients")
    optimizer.step()
    torch.cuda.synchronize()
    if optimizer_steps(optimizer) != [v + 1 for v in steps_before]:
        raise RuntimeError("expected exactly one optimizer update")
    del batch
    _, validation_paths = bc.load_split(ROOT / "data/BC/splits_autoregressive_v1/validation.json", "validation", ROOT / "data/BC")
    validation_loader = DataLoader(bc.BCChunks(validation_paths[:4], actor_dim, action_dim), batch_size=4)
    sampled_mse, per_action = bc.validate(policy, validation_loader, device, mean, std, action_dim)
    if not np.isfinite(sampled_mse):
        raise RuntimeError("nonfinite sampled BC diagnostic MSE")
    candidate = output / "fresh-bc-disposable-candidate.pt"
    save_smoke_checkpoint(candidate, policy, optimizer, saved, spec, generator, sampled_mse, per_action,
                          fresh_training_chunk_sha256=bc.sha256_file(source))
    report = {"result": "PASS", "mode": "fresh_bc_loader_training", "train_mse": mse,
              "grad_norm": grad, "seconds": time.monotonic() - started, "optimizer_steps": 1,
              "sample_chunks": 16, "fresh_migration_chunks": 1, "frames_per_chunk": 81,
              "production_batch_size_tested": True, "training_paths": [str(p) for p in smoke_paths],
              "source_sha256": bc.sha256_file(source), "checkpoint_sha256": CHECKPOINT_SHA,
              "sampled_diagnostic_validation_mse": sampled_mse, "success_rate_tested": False,
              "candidate_sha256": bc.sha256_file(candidate), "production_promoted": False}
    (output / "train-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["render", "train"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    source, output, chain = validate_artifact_paths(args.source, args.output, args.chain_source)
    if args.mode != "render" and args.chain_source:
        raise RuntimeError("chain sources apply only to rendering")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("exactly one allocated H200 is required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"{args.mode}-result.json").exists():
        raise RuntimeError("refusing to overwrite a completed qualification")
    if args.mode == "render":
        render(source, output, chain)
    else:
        train(source, output)


if __name__ == "__main__":
    main()
