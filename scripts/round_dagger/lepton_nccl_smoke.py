#!/usr/bin/env python3
"""Fail-closed eight-rank CUDA/NCCL smoke for the Lepton H200 learner node."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist


REMOTE_PROJECT_ROOT = Path("/workspace/projects/roborender")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-world-size", type=int, default=8)
    return parser


def _validate_output(path: Path) -> Path:
    resolved_parent = path.parent.resolve(strict=True)
    project_root = REMOTE_PROJECT_ROOT.resolve(strict=True)
    if resolved_parent != project_root and project_root not in resolved_parent.parents:
        raise RuntimeError("output must remain under the RoboRender project root")
    if path.exists() or path.is_symlink():
        raise RuntimeError("refusing to replace an existing smoke report")
    return resolved_parent / path.name


def _atomic_json(path: Path, document: dict) -> None:
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".incoming"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o440)
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.expected_world_size <= 0:
        raise RuntimeError("expected world size must be positive")
    output = _validate_output(arguments.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != arguments.expected_world_size:
        raise RuntimeError(
            f"world size {world_size} != expected {arguments.expected_world_size}"
        )
    if torch.cuda.device_count() < arguments.expected_world_size:
        raise RuntimeError("fewer visible CUDA devices than expected ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    started = time.monotonic()
    reduction = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    expected_sum = arguments.expected_world_size * (
        arguments.expected_world_size + 1
    ) / 2
    if reduction.item() != expected_sum:
        raise RuntimeError("NCCL all-reduce returned an incorrect result")

    torch.manual_seed(7300 + rank)
    left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    product = left @ right
    if not torch.isfinite(product).all().item():
        raise RuntimeError("BF16 tensor-core probe produced a non-finite value")
    checksum = product.float().mean()
    dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    properties = torch.cuda.get_device_properties(device)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "cuda_device_name": properties.name,
        "cuda_device_uuid": str(properties.uuid),
        "cuda_capability": list(properties.major_minor)
        if hasattr(properties, "major_minor")
        else [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    dist.barrier()

    if rank == 0:
        records = [record for record in gathered if record is not None]
        uuids = {record["cuda_device_uuid"] for record in records}
        ranks = {record["rank"] for record in records}
        local_ranks = {record["local_rank"] for record in records}
        if len(records) != world_size or len(uuids) != world_size:
            raise RuntimeError("ranks did not bind to distinct physical GPUs")
        if ranks != set(range(world_size)) or local_ranks != set(range(world_size)):
            raise RuntimeError("rank topology is incomplete")
        _atomic_json(
            output,
            {
                "schema": "fast_roborender.lepton_nccl_smoke.v1",
                "status": "PASS",
                "backend": dist.get_backend(),
                "world_size": world_size,
                "expected_all_reduce_sum": expected_sum,
                "observed_all_reduce_sum": reduction.item(),
                "bf16_checksum_sum": checksum.item(),
                "elapsed_seconds": time.monotonic() - started,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": list(torch.cuda.nccl.version()),
                "ranks": records,
            },
        )
        print(f"LEPTON_NCCL_SMOKE_PASS report={output}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
