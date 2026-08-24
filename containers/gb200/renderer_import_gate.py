#!/usr/bin/env python3
"""Fail closed unless the Image3F runtime matches its declared CUDA target."""

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
    "diffsynth",
    "einops",
    "ftfy",
    "imageio",
    "modelscope",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "pandas",
    "peft",
    "PIL",
    "safetensors",
    "sentencepiece",
    "tensorrt",
    "tokenizers",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
)


def main() -> None:
    expected_machine = os.environ.get("ROBORENDER_EXPECTED_MACHINE", "aarch64")
    expected_cc = [
        int(value)
        for value in os.environ.get("ROBORENDER_EXPECTED_CC", "10,0").split(",")
    ]
    expected_cuda = os.environ.get("ROBORENDER_EXPECTED_TORCH_CUDA")
    expected_tensorrt = os.environ.get("ROBORENDER_EXPECTED_TENSORRT")
    imported = {}
    for name in MODULES:
        module = importlib.import_module(name)
        imported[name] = getattr(module, "__version__", "import-ok")

    import tensorrt as trt
    import torch

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
            )
    report = {
        "architecture": platform.machine(),
        "python": sys.version,
        "imports": imported,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "devices": devices,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if platform.machine() != expected_machine:
        raise SystemExit(
            f"expected {expected_machine} userspace, found {platform.machine()}"
        )
    if not devices or any(row["compute_capability"] != expected_cc for row in devices):
        raise SystemExit(f"expected only SM{expected_cc[0]}{expected_cc[1]} devices, found {devices}")
    if expected_cuda and torch.version.cuda != expected_cuda:
        raise SystemExit(
            f"expected torch CUDA {expected_cuda}, found {torch.version.cuda}"
        )
    if expected_tensorrt and trt.__version__ != expected_tensorrt:
        raise SystemExit(
            f"expected TensorRT {expected_tensorrt}, found {trt.__version__}"
        )
    if not report["bf16_supported"]:
        raise SystemExit("BF16 is unavailable")

    # Exercise CUDA kernels and create a native TensorRT builder, rather than
    # treating import success as proof that shared objects are loadable.
    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise SystemExit("non-finite result from SM100 BF16 matmul")
    logger = trt.Logger(trt.Logger.ERROR)
    if trt.Builder(logger) is None:
        raise SystemExit("TensorRT builder creation failed")

    package_root = Path("/opt/roborender")
    sys.path.insert(0, str(package_root))
    from runtime.release_contract import adapter_path, load_config

    packaged_adapter = adapter_path(load_config())
    if not packaged_adapter.is_file():
        raise SystemExit(f"Image3F adapter is absent from the container: {packaged_adapter}")
    print("ROBORENDER_RENDERER_IMPORT_GATE_PASS")
    if expected_machine == "aarch64" and expected_cc == [10, 0]:
        print("HSG_GB200_RENDERER_IMPORT_GATE_PASS")


if __name__ == "__main__":
    main()
