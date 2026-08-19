#!/usr/bin/env python3
"""Discover and actively health-check an arbitrary-size Image3F endpoint fleet."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("server closed connection")
        data.extend(chunk)
    return bytes(data)


def ping(path: Path, timeout_s: float) -> dict[str, object]:
    started = time.perf_counter()
    fields = path.read_text(encoding="utf-8").split()
    if len(fields) != 2:
        raise ValueError("endpoint file must contain exactly 'host port'")
    host, port_s = fields
    port = int(port_s)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        payload = pickle.dumps(
            {"cmd": "ping", "protocol_version": 2}, protocol=pickle.HIGHEST_PROTOCOL
        )
        sock.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", recv_exact(sock, 8))[0]
        response = pickle.loads(recv_exact(sock, size))  # noqa: S301 - trusted cluster
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"unhealthy response: {response!r}")
    return {
        "file": str(path.resolve()),
        "host": host,
        "port": port,
        "rtt_ms": (time.perf_counter() - started) * 1000.0,
        "response": response,
    }


def probe(paths: list[Path], timeout_s: float) -> tuple[list[dict[str, object]], list[str]]:
    healthy: list[dict[str, object]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(64, max(1, len(paths)))) as pool:
        futures = {pool.submit(ping, path, timeout_s): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                healthy.append(future.result())
            except Exception as exc:  # report every endpoint; the caller decides whether to retry
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    healthy.sort(key=lambda item: item["file"])
    errors.sort()
    return healthy, errors


def capability_error(
    response: dict[str, object],
    *,
    minimum_protocol_version: int,
    required_reference_lag: int,
    require_reference_history: bool,
    expected_release_id: str | None = None,
    expected_adapter_sha256: str | None = None,
    expected_inference_steps: int | None = None,
    expected_teacache_threshold: float | None = None,
    expected_cfg_scale: float | None = None,
    expected_cuda_graph: bool | None = None,
    require_tensorrt_engine: bool = False,
    expected_output_views: tuple[str, ...] | None = None,
    expected_output_layout: str | None = None,
    expected_output_tile_shape: tuple[int, int, int] | None = None,
    expected_output_stacked_shape: tuple[int, int, int] | None = None,
) -> str | None:
    """Return why a healthy endpoint is incompatible, or ``None``."""
    capabilities = response.get("capabilities", {})
    if not isinstance(capabilities, dict):
        return "capabilities must be a mapping"
    try:
        protocol_version = int(response.get("protocol_version", 0))
        max_lag = int(capabilities.get("reference_lag_max", 0))
    except (TypeError, ValueError) as exc:
        return f"malformed numeric capability: {exc}"
    if protocol_version < minimum_protocol_version:
        return f"protocol {protocol_version} < required {minimum_protocol_version}"
    history_required = require_reference_history or required_reference_lag > 0
    if history_required and not capabilities.get("lagged_reference_history"):
        return "lagged reference history is not supported"
    if history_required and max_lag < required_reference_lag:
        return f"maximum reference lag {max_lag} < required {required_reference_lag}"
    if expected_output_views is not None:
        output_views = capabilities.get("output_views")
        if not isinstance(output_views, list) or tuple(output_views) != expected_output_views:
            return f"output views {output_views!r} != required {list(expected_output_views)!r}"
    if expected_output_layout is not None and capabilities.get("output_layout") != expected_output_layout:
        return (
            f"output layout {capabilities.get('output_layout')!r} != required "
            f"{expected_output_layout!r}"
        )
    for field, expected in (
        ("output_tile_shape_hwc", expected_output_tile_shape),
        ("output_stacked_shape_hwc", expected_output_stacked_shape),
    ):
        if expected is None:
            continue
        actual = capabilities.get(field)
        if not isinstance(actual, list) or tuple(actual) != expected:
            return f"{field} {actual!r} != required {list(expected)!r}"

    identity_required = any(
        value is not None
        for value in (
            expected_release_id,
            expected_adapter_sha256,
            expected_inference_steps,
            expected_teacache_threshold,
            expected_cfg_scale,
            expected_cuda_graph,
        )
    ) or require_tensorrt_engine
    if not identity_required:
        return None
    identity = response.get("runtime_identity")
    if not isinstance(identity, dict):
        return "runtime_identity must be a mapping"
    if identity.get("schema_version") != 1:
        return f"runtime identity schema {identity.get('schema_version')!r} != required 1"
    if expected_release_id is not None and identity.get("release_id") != expected_release_id:
        return (
            f"release {identity.get('release_id')!r} != required "
            f"{expected_release_id!r}"
        )
    adapter_sha256 = identity.get("adapter_sha256")
    if not isinstance(adapter_sha256, str) or not _SHA256_RE.fullmatch(adapter_sha256):
        return "runtime adapter SHA-256 is missing or malformed"
    if expected_adapter_sha256 is not None and adapter_sha256 != expected_adapter_sha256:
        return f"adapter SHA-256 {adapter_sha256} != required {expected_adapter_sha256}"

    generation = identity.get("generation")
    if not isinstance(generation, dict):
        return "runtime generation identity must be a mapping"
    try:
        steps = int(generation.get("steps"))
        tea_cache_threshold = float(generation.get("tea_cache_threshold"))
        cfg_scale = float(generation.get("cfg_scale"))
    except (TypeError, ValueError) as exc:
        return f"malformed generation identity: {exc}"
    if expected_inference_steps is not None and steps != expected_inference_steps:
        return f"inference steps {steps} != required {expected_inference_steps}"
    if expected_teacache_threshold is not None and not math.isclose(
        tea_cache_threshold, expected_teacache_threshold, rel_tol=0.0, abs_tol=1e-9
    ):
        return (
            f"TeaCache threshold {tea_cache_threshold} != required "
            f"{expected_teacache_threshold}"
        )
    if expected_cfg_scale is not None and not math.isclose(
        cfg_scale, expected_cfg_scale, rel_tol=0.0, abs_tol=1e-9
    ):
        return f"CFG scale {cfg_scale} != required {expected_cfg_scale}"
    if expected_cuda_graph is not None:
        cuda_graph = generation.get("cuda_graph")
        if not isinstance(cuda_graph, bool):
            return "CUDA graph identity must be boolean"
        if cuda_graph is not expected_cuda_graph:
            return f"CUDA graph enabled={cuda_graph} != required {expected_cuda_graph}"

    if require_tensorrt_engine:
        runtime = identity.get("runtime")
        engine = identity.get("engine")
        if not isinstance(runtime, dict):
            return "runtime accelerator identity must be a mapping"
        if not isinstance(engine, dict):
            return "TensorRT engine identity must be a mapping"
        if runtime.get("backend") != "tensorrt":
            return f"runtime backend {runtime.get('backend')!r} is not TensorRT"
        if runtime.get("engine_loaded") is not True:
            return "TensorRT engine is not loaded"
        if runtime.get("engine_verified_at_load") is not True:
            return "TensorRT engine was not verified at load"
        if not isinstance(runtime.get("engine_path"), str) or not runtime["engine_path"].strip():
            return "TensorRT engine path is missing or malformed"
        engine_sha256 = engine.get("sha256")
        if not isinstance(engine_sha256, str) or not _SHA256_RE.fullmatch(engine_sha256):
            return "TensorRT engine SHA-256 is missing or malformed"
        if engine.get("adapter_sha256") != adapter_sha256:
            return "TensorRT engine adapter SHA-256 does not match the runtime adapter"
        for field in (
            "engine_key",
            "export_key",
            "status",
            "precision",
            "gpu",
            "compute_capability",
            "tensorrt",
            "cuda",
            "torch",
            "driver",
        ):
            if not isinstance(engine.get(field), str) or not engine[field].strip():
                return f"TensorRT engine {field} is missing or malformed"
        if engine.get("status") != "built_microbenchmarked_and_correctness_checked":
            return f"TensorRT engine status {engine.get('status')!r} is not qualified"
        if engine.get("precision") != "bf16_fp16":
            return f"TensorRT engine precision {engine.get('precision')!r} is unsupported"
        if engine.get("cuda_graph_enabled_for_runtime") is not generation.get(
            "cuda_graph"
        ):
            return "TensorRT CUDA graph identity does not match the loaded runtime"
        for field in ("gpu", "compute_capability"):
            if runtime.get(field) != engine.get(field):
                return f"TensorRT engine {field} does not match the loaded runtime"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--wait-timeout-s", type=float, default=3600)
    parser.add_argument("--connect-timeout-s", type=float, default=10)
    parser.add_argument("--poll-s", type=float, default=10)
    parser.add_argument("--require-reference-history", action="store_true")
    parser.add_argument("--required-reference-lag", type=int, default=0)
    parser.add_argument("--minimum-protocol-version", type=int, default=2)
    parser.add_argument("--expected-release-id")
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-inference-steps", type=int)
    parser.add_argument("--expected-teacache-threshold", type=float)
    parser.add_argument("--expected-cfg-scale", type=float)
    parser.add_argument(
        "--expected-cuda-graph", choices=("enabled", "disabled"), default=None
    )
    parser.add_argument("--require-tensorrt-engine", action="store_true")
    parser.add_argument("--expected-output-views")
    parser.add_argument("--expected-output-layout")
    parser.add_argument("--expected-output-tile-shape")
    parser.add_argument("--expected-output-stacked-shape")
    parser.add_argument("--write-manifest", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
    parser.add_argument("--hydra-list", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    def parse_csv(raw, *, cast, flag):
        if raw is None:
            return None
        try:
            values = tuple(cast(value.strip()) for value in raw.split(","))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{flag} is malformed: {exc}") from exc
        if not values or any(value == "" for value in values):
            raise SystemExit(f"{flag} must be a non-empty comma-separated value")
        return values

    expected_output_views = parse_csv(
        args.expected_output_views, cast=str, flag="--expected-output-views"
    )
    expected_output_tile_shape = parse_csv(
        args.expected_output_tile_shape,
        cast=int,
        flag="--expected-output-tile-shape",
    )
    expected_output_stacked_shape = parse_csv(
        args.expected_output_stacked_shape,
        cast=int,
        flag="--expected-output-stacked-shape",
    )
    for flag, shape in (
        ("--expected-output-tile-shape", expected_output_tile_shape),
        ("--expected-output-stacked-shape", expected_output_stacked_shape),
    ):
        if shape is not None and (len(shape) != 3 or any(value <= 0 for value in shape)):
            raise SystemExit(f"{flag} must contain exactly three positive integers")
    if args.expected < 0:
        raise SystemExit("--expected must be non-negative")
    if args.required_reference_lag < 0:
        raise SystemExit("--required-reference-lag must be non-negative")
    if args.expected_inference_steps is not None and args.expected_inference_steps < 1:
        raise SystemExit("--expected-inference-steps must be positive")
    if args.expected_adapter_sha256 is not None and not _SHA256_RE.fullmatch(
        args.expected_adapter_sha256
    ):
        raise SystemExit("--expected-adapter-sha256 must be 64 lowercase hex characters")
    if args.expected_release_id is not None and not args.expected_release_id.strip():
        raise SystemExit("--expected-release-id must be non-empty")
    for flag, value in (
        ("--expected-teacache-threshold", args.expected_teacache_threshold),
        ("--expected-cfg-scale", args.expected_cfg_scale),
    ):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise SystemExit(f"{flag} must be finite and non-negative")
    if args.expected == 0:
        print("[]" if args.hydra_list else json.dumps({"healthy": [], "count": 0}))
        return 0

    deadline = time.monotonic() + args.wait_timeout_s
    last_report: dict[str, object] = {}
    while True:
        paths = sorted(args.endpoint_dir.glob("renderer_*.txt"))
        healthy, errors = probe(paths, args.connect_timeout_s) if paths else ([], [])
        unique = {(item["host"], item["port"]) for item in healthy}
        duplicate_count = len(healthy) - len(unique)
        capable = []
        for item in healthy:
            response = item["response"]
            incompatible = capability_error(
                response,
                minimum_protocol_version=args.minimum_protocol_version,
                required_reference_lag=args.required_reference_lag,
                require_reference_history=args.require_reference_history,
                expected_release_id=args.expected_release_id,
                expected_adapter_sha256=args.expected_adapter_sha256,
                expected_inference_steps=args.expected_inference_steps,
                expected_teacache_threshold=args.expected_teacache_threshold,
                expected_cfg_scale=args.expected_cfg_scale,
                expected_cuda_graph=(
                    args.expected_cuda_graph == "enabled"
                    if args.expected_cuda_graph is not None
                    else None
                ),
                require_tensorrt_engine=args.require_tensorrt_engine,
                expected_output_views=expected_output_views,
                expected_output_layout=args.expected_output_layout,
                expected_output_tile_shape=expected_output_tile_shape,
                expected_output_stacked_shape=expected_output_stacked_shape,
            )
            if incompatible is None:
                capable.append(item)
            else:
                errors.append(f"{item['file']}: incompatible endpoint: {incompatible}")
        last_report = {
            "endpoint_dir": str(args.endpoint_dir.resolve()),
            "expected": args.expected,
            "discovered": len(paths),
            "healthy_count": len(healthy),
            "capable_count": len(capable),
            "duplicate_count": duplicate_count,
            "healthy": healthy,
            "errors": errors,
        }
        if len(capable) >= args.expected and duplicate_count == 0:
            selected = capable[: args.expected]
            last_report["selected"] = selected
            if args.write_manifest:
                args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.write_manifest.with_name(
                    f".{args.write_manifest.name}.{os.getpid()}.tmp"
                )
                tmp.write_text(
                    "".join(f"{item['file']}\n" for item in selected), encoding="utf-8"
                )
                os.replace(tmp, args.write_manifest)
            if args.write_report:
                args.write_report.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.write_report.with_name(
                    f".{args.write_report.name}.{os.getpid()}.tmp"
                )
                tmp.write_text(
                    json.dumps(last_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp, args.write_report)
            if args.hydra_list:
                print("[" + ",".join(item["file"] for item in selected) + "]")
            elif args.json:
                print(json.dumps(last_report, indent=2, sort_keys=True))
            else:
                print(
                    f"READY: {len(capable)} healthy/capable Image3F endpoints "
                    f"(required {args.expected})"
                )
                for item in selected:
                    print(
                        f"  {Path(item['file']).name}: {item['host']}:{item['port']} "
                        f"ping={item['rtt_ms']:.1f}ms"
                    )
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(last_report, indent=2, sort_keys=True))
            print(
                f"FATAL: only {len(capable)}/{args.expected} healthy endpoints became ready",
                flush=True,
            )
            return 2
        print(
            f"waiting for renderers: discovered={len(paths)} healthy={len(healthy)} "
            f"capable={len(capable)} required={args.expected}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
