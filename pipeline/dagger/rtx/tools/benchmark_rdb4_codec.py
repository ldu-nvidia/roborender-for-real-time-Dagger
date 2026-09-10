#!/usr/bin/env python3
"""CPU-only benchmark for the production RDB4 ndarray codec.

The production default is left untouched.  This script compares the exact
RDB4/zlib-1 implementation with a benchmark-only deterministic raw frame that
uses the same canonical JSON header, lexical descriptors, per-array hashes,
contiguous C-order blob, and raw-blob attestation, but does not compress it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import zlib

import numpy as np

from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    MODEL_VIEW_MAP,
    RAVEN_VIEW_ORDER,
    RavenShapeProfile,
    canonical_json,
    canonical_wire_array,
    decode_arrays,
    payload_digest,
    sha256_bytes,
)
from dagger_rtx.lean_boundary_v4 import CONTRACT_VERSION_V4
from dagger_rtx.lean_boundary_v3 import project_minimal_arrays_v3
from dagger_rtx.ndarray_batch_v4 import (
    NDARRAY_BATCH_MAGIC_V4,
    pack_ndarray_batch_v4,
    unpack_ndarray_batch_v4,
)


SCHEMA = "roborender.rdb4-codec-benchmark.v1"
RAW_CODEC = "raw-benchmark-only"
VIEWS = tuple(RAVEN_VIEW_ORDER)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(samples_ns: Sequence[int]) -> dict[str, Any]:
    values = [value / 1_000_000.0 for value in samples_ns]
    return {
        "samples": len(values),
        "mean_ms": round(sum(values) / len(values), 6),
        "p50_ms": round(_percentile(values, 0.50), 6),
        "p90_ms": round(_percentile(values, 0.90), 6),
        "min_ms": round(min(values), 6),
        "max_ms": round(max(values), 6),
    }


def _time_call(
    function: Callable[[], Any], *, warmups: int, repeats: int
) -> tuple[dict[str, Any], Any]:
    result: Any = None
    for _ in range(warmups):
        result = function()
    samples: list[int] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = function()
            samples.append(time.perf_counter_ns() - started)
    finally:
        if gc_enabled:
            gc.enable()
    return _summary_ms(samples), result


def _depth_control(lane: int, view_index: int, profile: RavenShapeProfile) -> np.ndarray:
    y, x = np.indices((profile.control_h, profile.control_w), dtype=np.int32)
    values = (
        31
        + (x * (2 + view_index)) // max(profile.control_w // 5, 1)
        + (y * (3 + lane % 4)) // max(profile.control_h // 6, 1)
        + lane * 7
        + view_index * 13
    )
    # Piecewise-constant depth regions resemble the quantized h5_v4 control.
    values += ((x // 32 + y // 24 + lane) % 5) * 3
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _robot_control(lane: int, view_index: int, profile: RavenShapeProfile) -> np.ndarray:
    image = np.zeros((profile.control_h, profile.control_w, 3), dtype=np.uint8)
    x0 = 55 + ((lane * 11 + view_index * 29) % 155)
    y0 = 34 + ((lane * 7 + view_index * 17) % 72)
    width = 70 + (lane % 5) * 7
    height = 105 + (view_index % 2) * 19
    yy, xx = np.indices((height, width), dtype=np.int32)
    texture = np.empty((height, width, 3), dtype=np.uint8)
    texture[..., 0] = (53 + xx * 3 + lane * 5) % 256
    texture[..., 1] = (91 + yy * 2 + view_index * 19) % 256
    texture[..., 2] = (137 + xx + yy + lane * 3) % 256
    image[y0 : y0 + height, x0 : x0 + width] = texture
    return np.ascontiguousarray(image)


def _raytraced_rgb(lane: int, view_index: int, profile: RavenShapeProfile) -> np.ndarray:
    y, x = np.indices((profile.policy_h, profile.policy_w), dtype=np.int32)
    # Quantized camera-like colors cast to float32 [0,1], matching the actual
    # Raven policy tensor dtype while retaining realistic spatial correlation.
    image = np.empty((profile.policy_h, profile.policy_w, 3), dtype=np.uint8)
    image[..., 0] = (x + lane * 9 + view_index * 17) % 256
    image[..., 1] = (2 * y + lane * 5 + view_index * 31) % 256
    image[..., 2] = ((x // 3) + (y // 2) + lane * 13) % 256
    return np.ascontiguousarray(image.astype(np.float32) / np.float32(255.0))


def _lean_group(lane: int, source: str, profile: RavenShapeProfile) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(10_000 + lane)
    arrays: dict[str, np.ndarray] = {
        "actor_obs": np.ascontiguousarray(
            rng.normal(0.0, 0.55, profile.actor_obs).astype(np.float32)
        ),
        "teacher_action": np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, 26).astype(np.float32)
        ),
    }
    if source == "generated_controls":
        for view_index, view in enumerate(VIEWS):
            arrays[f"depth_ctrl__{view}"] = _depth_control(lane, view_index, profile)
            arrays[f"rgb_robot__{view}"] = _robot_control(lane, view_index, profile)
    elif source == "raytraced":
        for view_index, view in enumerate(VIEWS):
            arrays[f"raytraced_rgb__{view}"] = _raytraced_rgb(
                lane, view_index, profile
            )
    else:
        raise ValueError(source)
    return arrays


def _metadata(lane: int, source: str) -> dict[str, Any]:
    anchor_sha = sha256_bytes(f"representative-anchor-{lane}".encode())
    step_key = {
        "run_id": "codec-benchmark-b28",
        "round_id": 1,
        "actor_id": f"codec-benchmark-lane-{lane:03d}",
        "rollout_window_id": "window-000000",
        "rollout_step_index": 17,
        "episode_id": f"episode-{lane:03d}-0000",
        "episode_step_index": 17,
    }
    return {
        "contract_version": CONTRACT_VERSION_V4,
        "view_order": list(VIEWS),
        "model_view_map": dict(MODEL_VIEW_MAP),
        "observation_source": source,
        "prompt": "A photorealistic robot manipulation workspace.",
        "env_id": lane,
        "seq": 17,
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
        "step_key": step_key,
        "beta": 0.75,
        "episode_assignment": {
            "observation_source": source,
            "use_raytraced": source == "raytraced",
            "anchor_sha256": anchor_sha,
        },
    }


def _full_capture_template(
    lean: Mapping[str, np.ndarray], source: str, profile: RavenShapeProfile
) -> dict[str, np.ndarray]:
    # Full local capture fields that payload_digest hashes before v4 projection.
    arrays: dict[str, np.ndarray] = {
        "actor_obs": lean["actor_obs"],
        "teacher_obs": np.linspace(-1, 1, profile.teacher_obs, dtype=np.float32),
        "critic_obs": np.linspace(1, -1, profile.critic_obs, dtype=np.float32),
        "homie_obs": np.linspace(-0.5, 0.5, profile.homie_obs, dtype=np.float32),
        "gt_obj_pos": np.asarray([0.31, -0.08, 0.72], dtype=np.float32),
        "teacher_action": lean["teacher_action"],
        "homie_action": np.ascontiguousarray(lean["teacher_action"][11:]),
    }
    for view_index, view in enumerate(VIEWS):
        depth_ctrl = (
            lean[f"depth_ctrl__{view}"]
            if source == "generated_controls"
            else _depth_control(50 + view_index, view_index, profile)
        )
        full = np.repeat(np.repeat(depth_ctrl, 2, axis=0), 2, axis=1)
        arrays[f"depth_metric__{view}"] = np.ascontiguousarray(
            0.15 + full.astype(np.float32) * np.float32(0.012)
        )
        arrays[f"depth_rawq__{view}"] = np.ascontiguousarray(
            full.astype(np.uint16) * np.uint16(127)
        )
        arrays[f"depth_raw__{view}"] = np.ascontiguousarray(
            (0.15 + depth_ctrl.astype(np.float32) * np.float32(0.012)).astype(
                np.float16
            )
        )
        arrays[f"depth_ctrl__{view}"] = depth_ctrl
        arrays[f"rgb_robot__{view}"] = _robot_control(50 + view_index, view_index, profile)
        arrays[f"anchor_rgb__{view}"] = _robot_control(80 + view_index, view_index, profile)
        if source == "raytraced":
            arrays[f"raytraced_rgb__{view}"] = lean[f"raytraced_rgb__{view}"]
    return arrays


def _raw_pack(
    header: Mapping[str, Any], groups: Sequence[Mapping[str, np.ndarray]]
) -> bytes:
    value = json.loads(canonical_json(header))
    records = value["lanes"]
    if len(records) != len(groups):
        raise ValueError("raw benchmark record/group mismatch")
    offset = 0
    chunks: list[bytes] = []
    for record, arrays in zip(records, groups, strict=True):
        descriptors: list[dict[str, Any]] = []
        for name in sorted(arrays):
            array = canonical_wire_array(arrays[name])
            raw = array.tobytes(order="C")
            descriptors.append(
                {
                    "name": name,
                    "dtype": array.dtype.name,
                    "shape": list(array.shape),
                    "sha256": sha256_bytes(raw),
                    "offset": offset,
                    "nbytes": len(raw),
                }
            )
            chunks.append(raw)
            offset += len(raw)
        record["arrays"] = descriptors
    raw_blob = b"".join(chunks)
    value["blob_codec"] = RAW_CODEC
    value["raw_nbytes"] = len(raw_blob)
    value["raw_sha256"] = sha256_bytes(raw_blob)
    encoded_header = canonical_json(value)
    return b"".join(
        (
            NDARRAY_BATCH_MAGIC_V4,
            struct.pack("<Q", len(encoded_header)),
            encoded_header,
            raw_blob,
        )
    )


def _raw_unpack(body: bytes) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    if body[:4] != NDARRAY_BATCH_MAGIC_V4 or len(body) < 12:
        raise ValueError("invalid raw benchmark frame")
    header_len = struct.unpack("<Q", body[4:12])[0]
    header_end = 12 + header_len
    raw_header = body[12:header_end]
    header = json.loads(raw_header)
    if canonical_json(header) != raw_header or header.get("blob_codec") != RAW_CODEC:
        raise ValueError("invalid raw benchmark header")
    blob = memoryview(body)[header_end:]
    if len(blob) != header["raw_nbytes"] or sha256_bytes(blob.tobytes()) != header["raw_sha256"]:
        raise ValueError("raw benchmark blob attestation mismatch")
    expected_offset = 0
    decoded: list[dict[str, np.ndarray]] = []
    for record in header["lanes"]:
        arrays: dict[str, np.ndarray] = {}
        descriptors = record["arrays"]
        if [item["name"] for item in descriptors] != sorted(item["name"] for item in descriptors):
            raise ValueError("raw descriptors are not lexical")
        for descriptor in descriptors:
            if descriptor["offset"] != expected_offset:
                raise ValueError("raw descriptor offsets are not contiguous")
            end = expected_offset + descriptor["nbytes"]
            raw = blob[expected_offset:end]
            if sha256_bytes(raw.tobytes()) != descriptor["sha256"]:
                raise ValueError("raw array digest mismatch")
            dtype = np.dtype(descriptor["dtype"])
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<")
            array = np.frombuffer(raw, dtype=dtype).reshape(descriptor["shape"]).copy()
            arrays[descriptor["name"]] = array
            expected_offset = end
        decoded.append(arrays)
    if expected_offset != len(blob):
        raise ValueError("raw frame has trailing bytes")
    return header, decoded


def _assert_roundtrip(
    expected: Sequence[Mapping[str, np.ndarray]],
    actual: Sequence[Mapping[str, np.ndarray]],
) -> None:
    if len(expected) != len(actual):
        raise AssertionError("round-trip lane count mismatch")
    for lhs, rhs in zip(expected, actual, strict=True):
        if set(lhs) != set(rhs):
            raise AssertionError("round-trip array names mismatch")
        for name in lhs:
            if lhs[name].dtype != rhs[name].dtype or not np.array_equal(lhs[name], rhs[name]):
                raise AssertionError(f"round-trip mismatch for {name}")


def _load_real_fixture(
    root: Path, profile: RavenShapeProfile
) -> tuple[
    list[str],
    list[dict[str, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    """Load 14+14 distinct real captures outside the timed region."""

    step0 = sorted(root.rglob("prepare-000.json"))
    if len(step0) != 28:
        raise RuntimeError(f"expected exactly 28 fixture step-0 receipts, found {len(step0)}")
    selected: dict[str, list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]]] = {
        "generated_controls": [],
        "raytraced": [],
    }
    for path in step0:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload["metadata"]["observation_source"]
        if source not in selected:
            raise RuntimeError(f"unexpected fixture source {source}")
        if len(selected[source]) < 14:
            selected[source].append((path, payload["metadata"], decode_arrays(payload["arrays"])))
    if len(selected["generated_controls"]) != 14:
        raise RuntimeError("fixture does not contain 14 generated-control step-0 lanes")
    if len(selected["raytraced"]) < 14:
        needed = 14 - len(selected["raytraced"])
        for step0_path, _metadata_value, _arrays in selected["raytraced"][:needed]:
            step1_path = step0_path.with_name("prepare-001.json")
            payload = json.loads(step1_path.read_text(encoding="utf-8"))
            if payload["metadata"]["observation_source"] != "raytraced":
                raise RuntimeError("selected step-1 fixture changed observation source")
            selected["raytraced"].append(
                (step1_path, payload["metadata"], decode_arrays(payload["arrays"]))
            )
    ordered = selected["generated_controls"] + selected["raytraced"]
    sources = ["generated_controls"] * 14 + ["raytraced"] * 14
    groups: list[dict[str, np.ndarray]] = []
    capture_metadata: list[dict[str, Any]] = []
    captures: list[dict[str, np.ndarray]] = []
    provenance: list[dict[str, Any]] = []
    for source, (path, metadata, arrays) in zip(sources, ordered, strict=True):
        groups.append(
            project_minimal_arrays_v3(
                {"metadata": metadata, "arrays": arrays}, shape_profile=profile
            )
        )
        capture_metadata.append(metadata)
        captures.append(arrays)
        provenance.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "observation_source": source,
            }
        )
    return sources, groups, capture_metadata, captures, provenance


def _parallel(functions: Sequence[Callable[[], Any]]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=len(functions)) as executor:
        futures = [executor.submit(function) for function in functions]
        return [future.result() for future in futures]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--digest-repeats", type=int, default=5)
    parser.add_argument("--fixture-root", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 3 or args.digest_repeats < 3:
        parser.error("require warmups>=0, repeats>=3, digest-repeats>=3")

    profile = RavenShapeProfile()
    fixture_started = time.perf_counter()
    if args.fixture_root:
        sources, groups, capture_metadata, full_captures, fixture_provenance = (
            _load_real_fixture(args.fixture_root, profile)
        )
        payload_origin = "real retained native Raven prepare receipts"
    else:
        sources = ["generated_controls"] * 14 + ["raytraced"] * 14
        groups = [_lean_group(lane, source, profile) for lane, source in enumerate(sources)]
        capture_metadata = [_metadata(lane, source) for lane, source in enumerate(sources)]
        full_captures = [
            _full_capture_template(group, source, profile)
            for group, source in zip(groups, sources, strict=True)
        ]
        fixture_provenance = []
        payload_origin = "deterministic synthetic camera-like payload"
    fixture_load_seconds = time.perf_counter() - fixture_started
    metadata = [_metadata(lane, source) for lane, source in enumerate(sources)]

    wire_digests = [payload_digest(meta, arrays) for meta, arrays in zip(metadata, groups, strict=True)]
    records = []
    for lane, (meta, digest) in enumerate(zip(metadata, wire_digests, strict=True)):
        records.append(
            {
                "contract_version": CONTRACT_VERSION_V4,
                "idempotency_key": f"codec-prepare-idempotency-{lane:03d}",
                "step_key": meta["step_key"],
                "prepare_receipt": sha256_bytes(f"prepare-{lane}".encode()),
                "capture_digest": sha256_bytes(f"capture-{lane}".encode()),
                "wire_digest": digest,
                "beta": 0.75,
                "metadata": meta,
            }
        )
    header = {
        "contract_version": CONTRACT_VERSION_V4,
        "kind": "prepare_batch",
        "batch_id": "codec-benchmark-prepare-b28",
        "gateway_instance_id": "codec-benchmark-gateway-generation-1",
        "lanes": records,
    }

    zlib_encode, zlib_body = _time_call(
        lambda: pack_ndarray_batch_v4(header, groups),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    raw_encode, raw_body = _time_call(
        lambda: _raw_pack(header, groups),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    zlib_decode, zlib_decoded = _time_call(
        lambda: unpack_ndarray_batch_v4(zlib_body),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    raw_decode, raw_decoded = _time_call(
        lambda: _raw_unpack(raw_body),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    _assert_roundtrip(groups, zlib_decoded[1])
    _assert_roundtrip(groups, raw_decoded[1])

    wire_digest_time, repeated_wire_digests = _time_call(
        lambda: [
            payload_digest(meta, arrays)
            for meta, arrays in zip(metadata, groups, strict=True)
        ],
        warmups=args.warmups,
        repeats=args.digest_repeats,
    )
    if repeated_wire_digests != wire_digests:
        raise AssertionError("wire digest benchmark was not deterministic")

    full_capture_digest_time, full_capture_digests = _time_call(
        lambda: [
            payload_digest(meta, arrays)
            for meta, arrays in zip(capture_metadata, full_captures, strict=True)
        ],
        warmups=1,
        repeats=args.digest_repeats,
    )
    if len(set(full_capture_digests)) != 28:
        raise AssertionError("metadata must make each full capture digest distinct")

    raw_header_len = struct.unpack("<Q", raw_body[4:12])[0]
    raw_blob_bytes = len(raw_body) - 12 - raw_header_len
    zlib_header_len = struct.unpack("<Q", zlib_body[4:12])[0]
    compressed_blob_bytes = len(zlib_body) - 12 - zlib_header_len
    expected_raw_array_bytes = sum(array.nbytes for group in groups for array in group.values())
    if raw_blob_bytes != expected_raw_array_bytes:
        raise AssertionError("raw payload byte accounting mismatch")

    # Isolate compression itself from descriptor/raw-hash/header assembly.
    raw_blob = raw_body[12 + raw_header_len :]
    zlib_only, compressed_only = _time_call(
        lambda: zlib.compress(raw_blob, level=1),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    zlib_decompress_only, expanded_only = _time_call(
        lambda: zlib.decompress(compressed_only),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    if expanded_only != raw_blob:
        raise AssertionError("standalone zlib round-trip mismatch")

    # Measure the actual lane-i%8 deployment topology. Threaded wall is the
    # client-visible local barrier; sequential aggregate approximates summed CPU.
    shard_indices = [[lane for lane in range(28) if lane % 8 == shard] for shard in range(8)]
    shard_groups = [[groups[lane] for lane in indices] for indices in shard_indices]
    shard_headers = []
    for shard, indices in enumerate(shard_indices):
        shard_headers.append(
            {
                **{name: value for name, value in header.items() if name not in {"batch_id", "gateway_instance_id", "lanes"}},
                "batch_id": f"codec-benchmark-shard-{shard}",
                "gateway_instance_id": f"codec-benchmark-gateway-{shard}-generation-1",
                "lanes": [records[lane] for lane in indices],
            }
        )
    zlib_shard_encode_functions = [
        (lambda shard=shard: pack_ndarray_batch_v4(shard_headers[shard], shard_groups[shard]))
        for shard in range(8)
    ]
    raw_shard_encode_functions = [
        (lambda shard=shard: _raw_pack(shard_headers[shard], shard_groups[shard]))
        for shard in range(8)
    ]
    zlib_shard_sequential_encode, zlib_shard_bodies = _time_call(
        lambda: [function() for function in zlib_shard_encode_functions],
        warmups=args.warmups,
        repeats=args.repeats,
    )
    raw_shard_sequential_encode, raw_shard_bodies = _time_call(
        lambda: [function() for function in raw_shard_encode_functions],
        warmups=args.warmups,
        repeats=args.repeats,
    )
    zlib_shard_parallel_encode, _ = _time_call(
        lambda: _parallel(zlib_shard_encode_functions), warmups=1, repeats=args.repeats
    )
    raw_shard_parallel_encode, _ = _time_call(
        lambda: _parallel(raw_shard_encode_functions), warmups=1, repeats=args.repeats
    )
    zlib_shard_decode_functions = [
        (lambda shard=shard: unpack_ndarray_batch_v4(zlib_shard_bodies[shard]))
        for shard in range(8)
    ]
    raw_shard_decode_functions = [
        (lambda shard=shard: _raw_unpack(raw_shard_bodies[shard]))
        for shard in range(8)
    ]
    zlib_shard_sequential_decode, zlib_shard_decoded = _time_call(
        lambda: [function() for function in zlib_shard_decode_functions],
        warmups=args.warmups,
        repeats=args.repeats,
    )
    raw_shard_sequential_decode, raw_shard_decoded = _time_call(
        lambda: [function() for function in raw_shard_decode_functions],
        warmups=args.warmups,
        repeats=args.repeats,
    )
    zlib_shard_parallel_decode, _ = _time_call(
        lambda: _parallel(zlib_shard_decode_functions), warmups=1, repeats=args.repeats
    )
    raw_shard_parallel_decode, _ = _time_call(
        lambda: _parallel(raw_shard_decode_functions), warmups=1, repeats=args.repeats
    )
    for shard in range(8):
        _assert_roundtrip(shard_groups[shard], zlib_shard_decoded[shard][1])
        _assert_roundtrip(shard_groups[shard], raw_shard_decoded[shard][1])
    per_shard_measurements = []
    for shard in range(8):
        z_encode, _ = _time_call(
            zlib_shard_encode_functions[shard], warmups=1, repeats=args.repeats
        )
        r_encode, _ = _time_call(
            raw_shard_encode_functions[shard], warmups=1, repeats=args.repeats
        )
        z_decode, _ = _time_call(
            zlib_shard_decode_functions[shard], warmups=1, repeats=args.repeats
        )
        r_decode, _ = _time_call(
            raw_shard_decode_functions[shard], warmups=1, repeats=args.repeats
        )
        per_shard_measurements.append(
            {
                "shard": shard,
                "lanes": shard_indices[shard],
                "rdb4_zlib1_encode": z_encode,
                "raw_encode": r_encode,
                "rdb4_zlib1_decode": z_decode,
                "raw_decode": r_decode,
            }
        )

    script_path = Path(__file__).resolve()
    source_root = script_path.parents[1]
    codec_path = source_root / "code/src/dagger_rtx/ndarray_batch_v4.py"
    v3_codec_path = source_root / "code/src/dagger_rtx/ndarray_batch_v3.py"
    contract_path = source_root / "code/src/dagger_rtx/contracts.py"
    result = {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": {
            "cpu_only": True,
            "production_default_changed": False,
            "lanes": 28,
            "source_mix": {"generated_controls": 14, "raytraced": 14},
            "production_codec": "exact pack_ndarray_batch_v4/unpack_ndarray_batch_v4, RDB4 zlib level 1",
            "raw_comparator": (
                "benchmark-only deterministic raw frame: RDB4 prefix, canonical header, "
                "identical lexical ndarray descriptors/hashes and contiguous C-order blob, no compression"
            ),
            "timing": (
                "perf_counter_ns; payload built before timing; warmups excluded; encode includes "
                "per-array canonicalization, hashes, raw join, header, and compression where applicable; "
                "decode includes blob/array hash verification and copied ndarray reconstruction"
            ),
            "full_capture_digest_note": (
                "hashes all 28 already-decoded full local captures sequentially; excludes file read, "
                "capture, and GPU time"
            ),
            "payload_origin": payload_origin,
            "fixture_decode_and_hash_setup_seconds_excluded": round(fixture_load_seconds, 6),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "digest_repeats": args.digest_repeats,
        },
        "payload": {
            "lean_raw_array_bytes": expected_raw_array_bytes,
            "array_count": sum(len(group) for group in groups),
            "generated_lane_array_names": sorted(groups[0]),
            "raytraced_lane_array_names": sorted(groups[14]),
            "profile": vars(profile),
            "full_capture_bytes_hashed_per_b28_pass": sum(
                array.nbytes for capture in full_captures for array in capture.values()
            ),
            "fixture_receipts": fixture_provenance,
        },
        "bodies": {
            "rdb4_zlib1": {
                "bytes": len(zlib_body),
                "header_bytes": zlib_header_len,
                "blob_bytes": compressed_blob_bytes,
                "sha256": sha256_bytes(zlib_body),
            },
            "raw_uncompressed": {
                "bytes": len(raw_body),
                "header_bytes": raw_header_len,
                "blob_bytes": raw_blob_bytes,
                "sha256": sha256_bytes(raw_body),
            },
            "saved_bytes": len(raw_body) - len(zlib_body),
            "reduction_percent_vs_raw": round(
                100.0 * (1.0 - len(zlib_body) / len(raw_body)), 6
            ),
            "compression_ratio_zlib_over_raw": round(len(zlib_body) / len(raw_body), 8),
            "eight_shards_lane_modulo_8": {
                "lane_counts": [len(indices) for indices in shard_indices],
                "rdb4_zlib1_aggregate_bytes": sum(map(len, zlib_shard_bodies)),
                "raw_uncompressed_aggregate_bytes": sum(map(len, raw_shard_bodies)),
                "rdb4_zlib1_per_shard_bytes": [len(body) for body in zlib_shard_bodies],
                "raw_uncompressed_per_shard_bytes": [len(body) for body in raw_shard_bodies],
                "rdb4_zlib1_per_shard_sha256": [sha256_bytes(body) for body in zlib_shard_bodies],
                "raw_uncompressed_per_shard_sha256": [sha256_bytes(body) for body in raw_shard_bodies],
            },
        },
        "measurements": {
            "rdb4_zlib1_encode": zlib_encode,
            "raw_uncompressed_encode": raw_encode,
            "rdb4_zlib1_local_decode": zlib_decode,
            "raw_uncompressed_local_decode": raw_decode,
            "lean_wire_digest_all_28": wire_digest_time,
            "full_capture_digest_all_28": full_capture_digest_time,
            "zlib1_compress_preassembled_raw_blob": zlib_only,
            "zlib1_decompress_to_raw_blob": zlib_decompress_only,
            "eight_shards": {
                "rdb4_zlib1_sequential_aggregate_encode": zlib_shard_sequential_encode,
                "raw_sequential_aggregate_encode": raw_shard_sequential_encode,
                "rdb4_zlib1_parallel_wall_encode": zlib_shard_parallel_encode,
                "raw_parallel_wall_encode": raw_shard_parallel_encode,
                "rdb4_zlib1_sequential_aggregate_decode": zlib_shard_sequential_decode,
                "raw_sequential_aggregate_decode": raw_shard_sequential_decode,
                "rdb4_zlib1_parallel_wall_decode": zlib_shard_parallel_decode,
                "raw_parallel_wall_decode": raw_shard_parallel_decode,
                "per_shard": per_shard_measurements,
            },
        },
        "verification": {
            "zlib_roundtrip_element_exact": True,
            "raw_roundtrip_element_exact": True,
            "wire_digests_deterministic": True,
            "full_capture_digests_unique": True,
            "no_gpu_api_called": True,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "logical_cpu_count": os.cpu_count(),
        },
        "source_sha256": {
            str(script_path): _file_sha256(script_path),
            str(codec_path): _file_sha256(codec_path),
            str(v3_codec_path): _file_sha256(v3_codec_path),
            str(contract_path): _file_sha256(contract_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded_result = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded_result, encoding="utf-8")
    print(encoded_result, end="")
    print(f"artifact={args.output.resolve()}")
    print(f"artifact_sha256={_file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
