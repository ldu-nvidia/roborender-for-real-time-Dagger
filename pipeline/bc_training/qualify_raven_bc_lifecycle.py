"""Bounded, disposable BC lifecycle qualification; CPU-only planning by default.

Exactly two real B16x81 updates from the first32 sorted admitted merged chunks.
This is not a completed epoch, production checkpoint selection, success test,
allocation request, or compute cleanup. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.bc_training import train_bc_h200 as bc
from pipeline.bc_training.launcher_data import resolve_selection
from pipeline.bc_training.train_union import validate_training_admission

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "roborender.bc-lifecycle-qualification.v1"
UPDATES = 2


def source_identity():
    return {**bc.training_source_identity(), **{
        name: bc.sha256_file(ROOT / name) for name in (
            "pipeline/bc_training/qualify_raven_bc_lifecycle.py",
            "pipeline/bc_training/train_union.py",
            "pipeline/bc_training/launcher_data.py",
            "pipeline/dagger/services/runtime_policy_contract.py",
        )}}


def plan(args):
    """Metadata admission only: no CUDA calls, image decoding, or output writes."""
    selection = resolve_selection(project_root=ROOT, asset_root=args.asset_root,
        environ={"ROBORENDER_STORAGE_REGISTRY": str(args.registry)})
    root = Path(selection["DATASET_ROOT"])
    train = Path(selection["TRAIN_SPLIT"])
    retired = Path(selection["VALIDATION_SPLIT"])
    admission = Path(selection["DATA_ADMISSION_JSON"])
    report = validate_training_admission(admission, train, retired, root)
    if (report["chunk_count"] != 2880
            or report["original_chunk_counts"] != {"train": 2560, "validation": 320}
            or report["validation_enabled"] is not False):
        raise ValueError("qualification requires the admitted 2560+320 merged train-only pool")
    index, paths = bc.load_split(train, "train", root)
    paths = sorted(paths, key=lambda path: str(path.relative_to(root.resolve())))[:32]
    rows = {str((root / item["rendered_npz"]).resolve()): item for item in index["items"]}
    bootstrap = args.asset_root / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt"
    spec = bootstrap.with_suffix(".policy_spec.json")
    bootstrap_hash = bc.sha256_file(bootstrap)
    if json.loads(spec.read_bytes())["checkpoint_sha256"] != bootstrap_hash:
        raise ValueError("bootstrap/spec identity mismatch")
    return {
        "schema": SCHEMA, "execute": args.execute, "optimizer_updates": UPDATES,
        "chunks_per_update": 16, "frames_per_chunk": 81,
        "student_lifecycle": bc.REPLAY_STUDENT_LIFECYCLE,
        "validation_enabled": False, "completed_epoch": False,
        "production_checkpoint": False, "production_promoted": False,
        "success_rate_tested": False, "compute_release_authorized": False,
        "selection": "first32_lexically_sorted_merged_training_chunks",
        "full_training_chunk_count": 2880, "full_epoch_updates": 180,
        "dataset_root": str(root), "data_admission": str(admission),
        "data_admission_sha256": bc.sha256_file(admission),
        "train_split": str(train), "train_split_sha256": bc.sha256_file(train),
        "retired_validation_index": str(retired),
        "retired_validation_index_sha256": bc.sha256_file(retired),
        "bootstrap": str(bootstrap), "bootstrap_sha256": bootstrap_hash,
        "policy_spec": str(spec), "policy_spec_sha256": bc.sha256_file(spec),
        "source_identity": source_identity(), "seed": args.seed,
        "storage_registry": str(args.registry), "storage_registry_sha256": bc.sha256_file(args.registry),
        "selected_chunks": [{"path": str(path), "chunk_id": rows[str(path)]["chunk_id"],
            "original_role": report["origin_roles"][rows[str(path)]["chunk_id"]],
            "size_bytes": path.stat().st_size} for path in paths],
    }


def output_path(args):
    if args.output is None or not args.output.is_absolute():
        raise ValueError("--execute requires an absolute --output in an isolated smoke directory")
    # Planning already validates this registry through its central resolver.
    run_root = Path(json.loads(args.registry.read_bytes())["hosts"]["cosmos"]["roots"]["run_root"]).resolve()
    output = args.output.resolve()
    scopes = (run_root, (args.asset_root / "runtime").resolve())
    if (not output.name.startswith("smoke-") or output.exists()
            or not any(output != scope and output.is_relative_to(scope) for scope in scopes)):
        raise ValueError("qualification output must be new, smoke-named, and inside runtime or registered runs")
    return output


def cuda_identity(min_free_gib):
    if not math.isfinite(min_free_gib) or min_free_gib < 32:
        raise ValueError("minimum free GPU memory must be at least32 GiB")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one already-owned H200 for this qualification")
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0) != (9, 0) or "H200" not in props.name:
        raise RuntimeError("qualification requires an actual H200 SM90 device")
    uuid = str(getattr(props, "uuid", ""))
    if not uuid or uuid == "None":
        raise RuntimeError("CUDA runtime must expose the actual GPU UUID")
    free, total = torch.cuda.mem_get_info(0)
    if free < min_free_gib * (1 << 30):
        raise RuntimeError("insufficient currently free GPU memory for bounded B16 qualification")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host_available = int(meminfo["MemAvailable"].split()[0]) * 1024
    if host_available < 8 << 30:
        raise RuntimeError("less than8 GiB currently available host memory")
    cgroup_headroom = None
    maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if maximum.is_file() and current.is_file() and maximum.read_text().strip() != "max":
        cgroup_headroom = int(maximum.read_text()) - int(current.read_text())
        if cgroup_headroom < 4 << 30:
            raise RuntimeError("less than4 GiB current cgroup memory headroom")
    return {"gpu_uuid": uuid, "gpu_name": props.name, "capability": [9, 0],
        "hostname": os.uname().nodename, "pid": os.getpid(), "visible_gpu_count": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_free_bytes_before": free, "gpu_total_bytes": total,
        "host_available_bytes_before": host_available, "cgroup_headroom_bytes_before": cgroup_headroom,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()}


def tensor_digest(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def run_updates(policy, optimizer, loader, device, mean, std, *, progress=None, chunk_batch_size=16,
                pre_update=None):
    """Reusable real update loop; CPU tests substitute only the feature network."""
    trainable = [p for p in policy.parameters() if p.requires_grad]
    before = {name: tensor_digest(p) for name, p in policy.named_parameters()}
    batch_norms = {name: module for name, module in policy.named_modules()
                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    if not batch_norms or any(module.num_batches_tracked is None for module in batch_norms.values()):
        raise ValueError("qualification requires tracked BatchNorm layers")
    initial_bn = {name: int(module.num_batches_tracked) for name, module in batch_norms.items()}
    metrics = []
    policy.train()
    for update, batch in enumerate(loader, start=1):
        if update > UPDATES:
            raise ValueError("qualification loader contains more than exactly two batches")
        if pre_update is not None:
            pre_update(update)
        vision, actor, target = bc.prepare_batch(batch, device, mean, std)
        object_target = batch["object_target"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        loss = bc.backward_raven_offline_bc_mse(policy, vision, actor, target, object_target,
                                              chunk_batch_size=chunk_batch_size)
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
        if not math.isfinite(loss) or not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("BC qualification requires finite loss and nonzero finite gradients")
        for parameter in policy.obj_pred_mlp.parameters():
            if parameter.grad is None or bool(torch.count_nonzero(parameter.grad)):
                raise RuntimeError("auxiliary object head must have explicit zero gradients")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if any(int(module.num_batches_tracked) - initial_bn[name] != 82 * update
               for name, module in batch_norms.items()):
            raise RuntimeError("BatchNorm did not advance exactly82 times per update")
        for parameter in trainable:
            state = optimizer.state.get(parameter, {})
            if (not bool(torch.isfinite(parameter).all()) or "step" not in state
                    or int(state["step"]) != update
                    or not bool(torch.isfinite(state["exp_avg"]).all())
                    or not bool(torch.isfinite(state["exp_avg_sq"]).all())):
                raise RuntimeError("invalid/discontinuous AdamW parameter state")
        for parameter in policy.obj_pred_mlp.parameters():
            if any(bool(torch.count_nonzero(optimizer.state[parameter][key]))
                   for key in ("exp_avg", "exp_avg_sq")):
                raise RuntimeError("zero-weight auxiliary head acquired nonzero moments")
        row = {"update": update, "train_mse": loss, "gradient_norm_before_clip": norm,
            "update_seconds": elapsed, "bn_counter_delta": 82 * update}
        if device.type == "cuda":
            row.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        metrics.append(row)
        print(json.dumps({"phase": "bc_smoke_update", **row}), flush=True)
        if progress is not None:
            progress(row)
        del vision, actor, target, object_target
    if len(metrics) != UPDATES:
        raise ValueError("qualification requires exactly two full batches")
    after = {name: tensor_digest(p) for name, p in policy.named_parameters()}
    changed = [name for name in before if before[name] != after[name]]
    if not changed or any(name == "std" or name.startswith("obj_pred_mlp.") for name in changed):
        raise RuntimeError("action policy must update while std/object-head weights remain unchanged")
    return {"updates": metrics, "changed_parameter_names": changed,
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "batch_norm_counter_deltas": {name: int(module.num_batches_tracked) - initial_bn[name]
                                      for name, module in batch_norms.items()},
        "optimizer_parameter_states": len(optimizer.state)}


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise RuntimeError("checkpoint tensor reload mismatch")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise RuntimeError("checkpoint numpy RNG mismatch")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError("checkpoint dictionary mismatch")
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError("checkpoint sequence mismatch")
        for a, b in zip(left, right):
            assert_equal(a, b)
    elif left != right:
        raise RuntimeError("checkpoint scalar reload mismatch")


def reload_checkpoint(path, original_policy, original_optimizer, generator, spec_path, bootstrap):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    policy, _ = bc.build_policy(spec_path, bootstrap, torch.device("cpu"))
    bc.configure_raven_offline_student(policy)
    bc.install_index_safe_max_pool2d(policy)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.)
    policy.load_state_dict(saved["policy_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    assert_equal(original_policy.state_dict(), policy.state_dict())
    assert_equal(original_optimizer.state_dict(), optimizer.state_dict())
    rng = saved["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    generator.set_state(rng["train_generator"])
    assert_equal(rng, {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
        "train_generator": generator.get_state()})
    return {"policy_reload_exact": True, "optimizer_reload_exact": True, "rng_restore_exact": True}


def execute(args, admitted):
    output = output_path(args)
    identity = cuda_identity(args.min_free_gpu_gib)
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"phase": "bc_smoke_started", "output": str(output), **identity}), flush=True)
    run = None
    started = time.monotonic()
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")
        generator = torch.Generator().manual_seed(args.seed)
        paths = [Path(row["path"]) for row in admitted["selected_chunks"]]
        payload_hashes = {str(path): bc.sha256_file(path) for path in paths}
        bc.atomic_json(output / "plan.json", {**admitted, "device": identity,
                       "selected_payload_sha256": payload_hashes})
        if args.wandb_mode == "online":
            import wandb
            run = wandb.init(project=args.wandb_project, name=output.name,
                group="bc-lifecycle-qualification", mode="online", config=admitted)
        policy, spec = bc.build_policy(Path(admitted["policy_spec"]), Path(admitted["bootstrap"]), device)
        bc.configure_raven_offline_student(policy)
        pools = bc.install_index_safe_max_pool2d(policy)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.)
        dataset = bc.BCChunks(paths, int(spec["input_contract"]["actor_obs_dim"]), 11,
                              include_object_target=True)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, generator=generator)
        mean = torch.tensor(spec["input_contract"]["rgb_mean"], device=device).view(1,1,1,1,3)
        std = torch.tensor(spec["input_contract"]["rgb_std"], device=device).view(1,1,1,1,3)
        torch.cuda.reset_peak_memory_stats()
        metrics = run_updates(policy, optimizer, loader, device, mean, std,
            progress=(lambda row: run.log(row, step=row["update"])) if run is not None else None)
        checkpoint = output / "diagnostic-checkpoint.pt"
        metadata = {**admitted, "policy_spec": spec, "selection": "diagnostic_only_not_completed_epoch",
            "qualification_optimizer_steps": UPDATES,
            "training_batch_contract": bc.student_lifecycle_batch_fields(bc.REPLAY_STUDENT_LIFECYCLE),
            "training_protocol": {"schema": SCHEMA, "optimizer_updates": UPDATES}}
        bc.save_checkpoint(checkpoint, policy, optimizer, 0, UPDATES, None, None, metadata, generator)
        reloaded = reload_checkpoint(checkpoint, policy, optimizer, generator,
            Path(admitted["policy_spec"]), Path(admitted["bootstrap"]))
        if source_identity() != admitted["source_identity"] or any(
                bc.sha256_file(Path(path)) != sha for path, sha in payload_hashes.items()):
            raise RuntimeError("source or selected training data changed during qualification")
        if plan(args) != admitted:
            raise RuntimeError("admitted metadata changed during qualification")
        result = {**admitted, **metrics, **reloaded, "result": "PASS", "device": identity,
            "index_safe_max_pools": pools, "checkpoint": str(checkpoint),
            "checkpoint_sha256": bc.sha256_file(checkpoint), "selected_payload_sha256": payload_hashes,
            "gpu_free_bytes_after": torch.cuda.mem_get_info(0)[0],
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "total_seconds": time.monotonic() - started}
        if run is not None:
            result["wandb"] = {"id": run.id, "url": run.url}
            run.summary.update({"result": "PASS", "optimizer_updates": UPDATES,
                                "production_promoted": False, **reloaded})
            run.finish()
            run = None
        bc.atomic_json(output / "smoke-result.json", result)
        print(json.dumps({"result": "PASS", "output": str(output), **reloaded}), flush=True)
        return result
    except BaseException as error:
        bc.atomic_json(output / "failure.json", {"schema": SCHEMA, "result": "FAIL",
            "error_type": type(error).__name__, "error": str(error),
            "compute_release_authorized": False, "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        if run is not None:
            run.finish(exit_code=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-root", type=Path,
        default=Path(os.environ.get("ROBORENDER_ASSET_ROOT", ROOT)))
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/storage_paths.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gpu-gib", type=float, default=64.)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="dagger")
    args = parser.parse_args(argv)
    if not args.asset_root.is_absolute() or not args.registry.is_absolute():
        parser.error("asset root and registry must be absolute")
    return args


def main(argv=None):
    args = parse_args(argv)
    admitted = plan(args)
    if args.execute:
        execute(args, admitted)
    else:
        print(json.dumps(admitted, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
