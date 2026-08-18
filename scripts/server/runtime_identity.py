#!/usr/bin/env python3
"""Extract the immutable Image3F model and accelerator identity for health checks.

The deployment runtime verifies adapter and TensorRT engine hashes while an
``Image3FSession`` is constructed.  This module exposes that already-verified
identity through the server ping without importing the heavyweight runtime,
which also makes the contract straightforward to unit test.
"""

from __future__ import annotations

import re
import math
from collections.abc import Mapping
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Image3F {label} is unavailable or malformed")
    return value


def _required_text(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Image3F {label}.{key} is unavailable or malformed")
    return value.strip()


def describe_runtime_identity(session: object) -> dict[str, object]:
    """Return a fail-closed, pickle-safe identity for a loaded session.

    ``Image3FSession`` has already authenticated the selected engine against its
    build manifest, current GPU/software fingerprint, ONNX lineage, and adapter
    hash.  Missing provenance here is therefore a server-startup error, not a
    reason to advertise a partially identified endpoint.
    """

    describe = getattr(session, "describe", None)
    if not callable(describe):
        raise RuntimeError("Image3F session does not expose describe()")
    description = _mapping(describe(), "runtime description")
    config = _mapping(getattr(session, "config", None), "deployment config")
    adapter = _mapping(config.get("adapter"), "deployment config adapter")
    engine_manifest = _mapping(
        getattr(session, "engine_manifest", None), "TensorRT engine manifest"
    )
    engine = _mapping(engine_manifest.get("engine"), "TensorRT engine record")

    release_id = _required_text(config, "release_id", "deployment config")
    adapter_sha256 = _required_text(adapter, "sha256", "deployment config adapter")
    engine_sha256 = _required_text(engine, "sha256", "TensorRT engine record")
    engine_adapter_sha256 = _required_text(
        engine_manifest, "adapter_sha256", "TensorRT engine manifest"
    )
    if not _SHA256_RE.fullmatch(adapter_sha256):
        raise RuntimeError("Image3F deployment adapter SHA-256 is malformed")
    if not _SHA256_RE.fullmatch(engine_sha256):
        raise RuntimeError("Image3F TensorRT engine SHA-256 is malformed")
    if engine_adapter_sha256 != adapter_sha256:
        raise RuntimeError("Image3F TensorRT engine was built for a different adapter")

    try:
        steps = int(description["steps"])
        cfg_scale = float(description["cfg_scale"])
        tea_cache_threshold = float(description["tea_cache_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Image3F generation identity is malformed: {exc}") from exc
    if steps < 1 or not math.isfinite(cfg_scale) or not math.isfinite(tea_cache_threshold):
        raise RuntimeError("Image3F generation identity contains invalid numeric values")
    cuda_graph = description.get("cuda_graph")
    if not isinstance(cuda_graph, bool):
        raise RuntimeError("Image3F CUDA graph identity is unavailable or malformed")
    engine_path = _required_text(description, "engine_path", "runtime description")

    runtime_gpu = _required_text(description, "gpu", "runtime description")
    runtime_capability = _required_text(
        description, "compute_capability", "runtime description"
    )
    engine_gpu = _required_text(engine_manifest, "gpu", "TensorRT engine manifest")
    engine_capability = _required_text(
        engine_manifest, "compute_capability", "TensorRT engine manifest"
    )
    if engine_gpu != runtime_gpu or engine_capability != runtime_capability:
        raise RuntimeError(
            "Image3F TensorRT engine hardware identity does not match the loaded runtime"
        )
    engine_status = _required_text(
        engine_manifest, "status", "TensorRT engine manifest"
    )
    if engine_status != "built_microbenchmarked_and_correctness_checked":
        raise RuntimeError(f"Image3F TensorRT engine has unqualified status {engine_status!r}")
    engine_precision = _required_text(
        engine_manifest, "precision", "TensorRT engine manifest"
    )
    if engine_precision != "bf16_fp16":
        raise RuntimeError(f"Image3F TensorRT engine has unsupported precision {engine_precision!r}")
    graph_manifest = _mapping(
        engine_manifest.get("cuda_graph"), "TensorRT CUDA graph record"
    )
    graph_enabled = graph_manifest.get("enabled_for_runtime")
    if not isinstance(graph_enabled, bool) or graph_enabled is not cuda_graph:
        raise RuntimeError(
            "Image3F TensorRT CUDA graph identity does not match the loaded runtime"
        )

    # Do not hash a multi-GiB engine on every ping. Image3FSession did so during
    # construction; this response carries the verified manifest identity.
    return {
        "schema_version": 1,
        "release_id": release_id,
        "adapter_sha256": adapter_sha256,
        "generation": {
            "steps": steps,
            "cfg_scale": cfg_scale,
            "tea_cache_threshold": tea_cache_threshold,
            "cuda_graph": cuda_graph,
        },
        "runtime": {
            "backend": "tensorrt",
            "engine_loaded": True,
            "engine_verified_at_load": True,
            "gpu": runtime_gpu,
            "compute_capability": runtime_capability,
            "engine_path": engine_path,
        },
        "engine": {
            "sha256": engine_sha256,
            "engine_key": _required_text(
                engine_manifest, "engine_key", "TensorRT engine manifest"
            ),
            "export_key": _required_text(
                engine_manifest, "export_key", "TensorRT engine manifest"
            ),
            "adapter_sha256": engine_adapter_sha256,
            "status": engine_status,
            "precision": engine_precision,
            "cuda_graph_enabled_for_runtime": graph_enabled,
            "gpu": engine_gpu,
            "compute_capability": engine_capability,
            "tensorrt": _required_text(
                engine_manifest, "tensorrt", "TensorRT engine manifest"
            ),
            "cuda": _required_text(engine_manifest, "cuda", "TensorRT engine manifest"),
            "torch": _required_text(
                engine_manifest, "torch", "TensorRT engine manifest"
            ),
            "driver": _required_text(
                engine_manifest, "driver", "TensorRT engine manifest"
            ),
        },
    }
