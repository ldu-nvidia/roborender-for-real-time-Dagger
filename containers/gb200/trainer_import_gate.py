#!/usr/bin/env python3
"""Static import and CUDA gate for a declared DAgger trainer target."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path


MODULES = (
    "accelerate",
    "cv2",
    "datasets",
    "gymnasium",
    "hydra",
    "isaaclab",
    "isaaclab.app",
    "gr00t",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "pandas",
    "pyarrow",
    "ray",
    "rl_games",
    "rsl_rl",
    "scipy",
    "skrl",
    "stable_baselines3",
    "tensordict",
    "torch",
    "torchvision",
    "transformers",
    "trl",
    "wandb",
)

# These modules import Omniverse APIs and must only be loaded after AppLauncher
# has instantiated SimulationApp.  The sealed camera gate validates them in
# that supported lifecycle order.
KIT_MODULES = ("isaaclab_contrib", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")

DEFAULT_GR00T_REVISION = "b4d17449d70eb3fb7caa7ea9366ca59bb0589cdd"


def main() -> None:
    expected_machine = os.environ.get("ROBORENDER_EXPECTED_MACHINE", "aarch64")
    expected_cc = [
        int(value)
        for value in os.environ.get("ROBORENDER_EXPECTED_CC", "10,0").split(",")
    ]
    expected_revision = os.environ.get(
        "ROBORENDER_EXPECTED_GR00T_REVISION", DEFAULT_GR00T_REVISION
    )
    versions = {}
    for name in MODULES:
        print(f"importing={name}", flush=True)
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "import-ok")

    import torch
    import torchvision

    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []
    marker = Path("/opt/GR00T-VisualSim2Real/.source_revision")
    revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    report = {
        "architecture": platform.machine(),
        "python": sys.version,
        "imports": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "devices": devices,
        "gr00t_revision": revision,
        "kit_ordered_imports": list(KIT_MODULES),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if platform.machine() != expected_machine:
        raise SystemExit(
            f"expected {expected_machine} userspace, found {platform.machine()}"
        )
    if not devices or any(row["compute_capability"] != expected_cc for row in devices):
        raise SystemExit(f"expected only SM{expected_cc[0]}{expected_cc[1]} devices, found {devices}")
    if revision != expected_revision:
        raise SystemExit(f"unexpected GR00T source revision: {revision}")
    if not torch.__version__.startswith("2.7.0") or torch.version.cuda != "12.8":
        raise SystemExit(
            f"trainer must retain Isaac 5.1 torch/CUDA, found {torch.__version__}/{torch.version.cuda}"
        )
    if not torchvision.__version__.startswith("0.22.0"):
        raise SystemExit(f"unexpected torchvision runtime: {torchvision.__version__}")
    sample = torch.arange(4096, device="cuda", dtype=torch.float32)
    if sample.square().sum().item() <= 0:
        raise SystemExit("CUDA kernel validation failed")
    print("ROBORENDER_TRAINER_IMPORT_GATE_PASS")
    if expected_machine == "aarch64" and expected_cc == [10, 0]:
        print("HSG_GB200_TRAINER_IMPORT_GATE_PASS")


if __name__ == "__main__":
    main()
