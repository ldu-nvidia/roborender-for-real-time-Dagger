#!/usr/bin/env python3
"""Apply the Torch 2.7 H100 compatibility fix to the Image3F exporter."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


RELATIVE_EXPORTER = Path(
    "vendor/raven-image3f-joint/control/export_image3f_block_stack_onnx.py"
)
ORIGINAL = """\
    capture = capture_reference_block_stack(pipe, row, request, args.seed)
    wrapper = WanBlockStackWrapper(pipe.dit).eval().cuda()
"""
PATCHED = """\
    capture = capture_reference_block_stack(pipe, row, request, args.seed)
    # Hooks run under inference_mode, so their clones are inference tensors.
    # Clone once more after leaving inference_mode to create normal tensors that
    # Torch 2.7's ONNX exporter may safely save for backward while tracing.
    with torch.inference_mode(False):
        capture = {key: value.detach().clone() for key, value in capture.items()}
    inference_keys = [
        key for key, value in capture.items() if torch.is_inference(value)
    ]
    if inference_keys:
        raise RuntimeError(
            "ONNX capture normalization left inference tensors: "
            f"{sorted(inference_keys)}"
        )
    wrapper = WanBlockStackWrapper(pipe.dit).eval().cuda()
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch(package_root: Path) -> Path:
    exporter = package_root / RELATIVE_EXPORTER
    source = exporter.read_text(encoding="utf-8")
    if PATCHED in source:
        return exporter
    count = source.count(ORIGINAL)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one known exporter insertion point, found {count}: "
            f"{exporter}"
        )
    exporter.write_text(source.replace(ORIGINAL, PATCHED), encoding="utf-8")
    return exporter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    args = parser.parse_args()
    exporter = patch(args.package_root.resolve())
    print(f"patched_exporter={exporter}")
    print(f"patched_exporter_sha256={sha256(exporter)}")


if __name__ == "__main__":
    main()
