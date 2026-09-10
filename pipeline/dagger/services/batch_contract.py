"""Lean binary transport for vectorized RTX-to-H200 DAgger batches.

The framing deliberately keeps identifiers and receipts in canonical JSON while
placing ndarray bytes directly after the header.  It avoids base64 expansion and
lets one HTTP request carry a complete vectorized simulator barrier.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CANONICAL_BETA, ContractError, VIEWS, exact_keys, identifier, sha256_digest
from .step_contract import (
    ALLOWED_DTYPES, StepKey, bytes_sha256, canonical_bytes, validate_metadata,
)


BATCH_CONTRACT = "roborender.dagger.rtx.v3"
BATCH_CONTENT_TYPE = "application/vnd.roborender.ndarray-batch.v3"
BATCH_MAGIC = b"RDB3"
BATCH_PREFIX = struct.Struct("<4sQ")
STREAM_CONTRACT = "roborender.dagger.rtx.v4"
STREAM_CONTENT_TYPE = "application/vnd.roborender.ndarray-batch.v4"
STREAM_MAGIC = b"RDB4"
MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_BATCH_SIZE = 256

_FIXED_ARRAYS = {
    "actor_obs": ((113,), "float32"),
    "teacher_action": ((26,), "float32"),
}


def _array(value: np.ndarray, name: str, shape: tuple[int, ...], dtype: str) -> None:
    if value.shape != shape or value.dtype.name != dtype:
        raise ContractError(
            f"{name} must be {dtype}{list(shape)}, got {value.dtype.name}{list(value.shape)}"
        )
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ContractError(f"{name} contains NaN or infinity")


def lean_arrays_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_bytes(metadata))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(canonical_bytes({
            "name": name, "dtype": value.dtype.name, "shape": list(value.shape),
        }))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_lean_arrays(arrays: Mapping[str, np.ndarray], source: str) -> None:
    required = set(_FIXED_ARRAYS)
    for view in VIEWS:
        if source == "raytraced":
            required.add(f"raytraced_rgb__{view}")
        else:
            required.update((f"depth_ctrl__{view}", f"rgb_robot__{view}"))
    if set(arrays) != required:
        raise ContractError(
            f"lean arrays differ; missing={sorted(required - set(arrays))}, "
            f"unknown={sorted(set(arrays) - required)}"
        )
    for name, (shape, dtype) in _FIXED_ARRAYS.items():
        _array(arrays[name], name, shape, dtype)
    for view in VIEWS:
        if source == "raytraced":
            name = f"raytraced_rgb__{view}"
            _array(arrays[name], name, (108, 192, 3), "float32")
            if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
                raise ContractError(f"{name} must be finite RGB in [0,1]")
        else:
            _array(arrays[f"depth_ctrl__{view}"], f"depth_ctrl__{view}", (240, 416), "uint8")
            _array(arrays[f"rgb_robot__{view}"], f"rgb_robot__{view}", (240, 416, 3), "uint8")


def anchor_arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    if set(arrays) != {f"anchor_rgb__{view}" for view in VIEWS}:
        raise ContractError("anchor registration must contain exactly three anchor RGB arrays")
    digest = hashlib.sha256()
    for view in VIEWS:
        name = f"anchor_rgb__{view}"
        value = arrays[name]
        _array(value, name, (240, 416, 3), "uint8")
        value = np.ascontiguousarray(value)
        digest.update(canonical_bytes({
            "view": view, "dtype": "uint8", "shape": [240, 416, 3],
        }))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _encode_binary_batch(
    header: Mapping[str, Any],
    lane_arrays: Sequence[Mapping[str, np.ndarray]],
    *,
    magic: bytes,
) -> bytes:
    """Encode a minimal batch with fast lossless compression.

    Compression is over only the concatenated raw ndarray blob.  Per-array
    descriptors and hashes continue to describe the original bytes exactly,
    so this changes transport cost without changing any model input value.
    """

    result = json.loads(json.dumps(header))
    lanes = result.get("lanes") or result.get("anchors")
    if not isinstance(lanes, list) or len(lanes) != len(lane_arrays):
        raise ContractError("header items and array batches differ")
    blob = bytearray()
    for item, arrays in zip(lanes, lane_arrays):
        descriptors = []
        for name in sorted(arrays):
            identifier(name, "array name")
            value = np.ascontiguousarray(arrays[name])
            raw = value.tobytes(order="C")
            descriptors.append({
                "name": name,
                "dtype": value.dtype.name,
                "shape": list(value.shape),
                "sha256": bytes_sha256(raw),
                "offset": len(blob),
                "nbytes": len(raw),
            })
            blob.extend(raw)
        item["arrays"] = descriptors
    raw_blob = bytes(blob)
    result.update({
        "blob_codec": "zlib-1",
        "raw_nbytes": len(raw_blob),
        "raw_sha256": bytes_sha256(raw_blob),
    })
    header_bytes = canonical_bytes(result)
    compressed = zlib.compress(raw_blob, level=1)
    body = BATCH_PREFIX.pack(magic, len(header_bytes)) + header_bytes + compressed
    if len(body) > MAX_BATCH_BYTES:
        raise ContractError("binary batch exceeds 64 MiB")
    return body


def encode_binary_batch(
    header: Mapping[str, Any], lane_arrays: Sequence[Mapping[str, np.ndarray]]
) -> bytes:
    return _encode_binary_batch(header, lane_arrays, magic=BATCH_MAGIC)


def encode_stream_batch(
    header: Mapping[str, Any], lane_arrays: Sequence[Mapping[str, np.ndarray]]
) -> bytes:
    return _encode_binary_batch(header, lane_arrays, magic=STREAM_MAGIC)


def _decode_binary_batch(
    raw: bytes, *, expected_magic: bytes
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    if len(raw) < BATCH_PREFIX.size:
        raise ContractError("binary batch is truncated")
    magic, header_size = BATCH_PREFIX.unpack_from(raw)
    if magic != expected_magic:
        raise ContractError("binary batch magic mismatch")
    header_end = BATCH_PREFIX.size + header_size
    if header_end > len(raw):
        raise ContractError("binary batch header is truncated")
    header_raw = raw[BATCH_PREFIX.size:header_end]
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid binary batch header: {exc}") from exc
    if not isinstance(header, dict) or canonical_bytes(header) != header_raw:
        raise ContractError("binary batch header is not canonical JSON")
    items = header.get("lanes") or header.get("anchors")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_SIZE:
        raise ContractError("binary batch must contain 1..256 items")
    if header.get("blob_codec") != "zlib-1":
        raise ContractError("binary batch blob codec must be zlib-1")
    raw_nbytes = header.get("raw_nbytes")
    if isinstance(raw_nbytes, bool) or not isinstance(raw_nbytes, int):
        raise ContractError("binary batch raw_nbytes must be an integer")
    if raw_nbytes < 0 or raw_nbytes > MAX_BATCH_BYTES:
        raise ContractError("binary batch uncompressed blob exceeds 64 MiB")
    raw_sha = sha256_digest(header.get("raw_sha256"), "raw_sha256")
    inflater = zlib.decompressobj()
    inflated = inflater.decompress(raw[header_end:], raw_nbytes + 1)
    if (
        len(inflated) != raw_nbytes or inflater.unconsumed_tail
        or inflater.unused_data or not inflater.eof
    ):
        raise ContractError("binary batch compressed blob length or framing is invalid")
    if bytes_sha256(inflated) != raw_sha:
        raise ContractError("binary batch uncompressed blob digest mismatch")
    blob = memoryview(inflated)
    expected_offset = 0
    decoded: list[dict[str, np.ndarray]] = []
    for item in items:
        descriptors = item.get("arrays") if isinstance(item, dict) else None
        if not isinstance(descriptors, list):
            raise ContractError("binary batch item has no array descriptors")
        arrays: dict[str, np.ndarray] = {}
        names = [descriptor.get("name") for descriptor in descriptors if isinstance(descriptor, dict)]
        if names != sorted(names) or len(set(names)) != len(names):
            raise ContractError("array descriptors must have unique lexical names")
        for descriptor in descriptors:
            exact_keys(descriptor, {"name", "dtype", "shape", "sha256", "offset", "nbytes"})
            name = identifier(descriptor["name"], "array name")
            if descriptor["dtype"] not in ALLOWED_DTYPES:
                raise ContractError(f"unsupported wire dtype {descriptor['dtype']!r}")
            dtype = np.dtype(descriptor["dtype"])
            shape = descriptor["shape"]
            offset = descriptor["offset"]
            nbytes = descriptor["nbytes"]
            if (
                not isinstance(shape, list) or len(shape) > 5
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
                or isinstance(offset, bool) or not isinstance(offset, int)
                or isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0
            ):
                raise ContractError("invalid binary ndarray descriptor")
            if offset != expected_offset or math.prod(shape) * dtype.itemsize != nbytes:
                raise ContractError("binary ndarray offsets or byte count are not contiguous")
            end = offset + nbytes
            if end > len(blob):
                raise ContractError("binary ndarray body is truncated")
            block = blob[offset:end]
            if bytes_sha256(block) != sha256_digest(descriptor["sha256"], "array sha256"):
                raise ContractError("binary ndarray SHA-256 mismatch")
            # Keep a read-only view over the immutable HTTP body. Inference and
            # persistence consume these arrays before the request returns, so a
            # second allocation/copy per tensor is unnecessary.
            arrays[name] = np.frombuffer(block, dtype=dtype).reshape(tuple(shape))
            expected_offset = end
        decoded.append(arrays)
    if expected_offset != len(blob):
        raise ContractError("binary batch contains unclaimed trailing bytes")
    return header, decoded


def decode_binary_batch(raw: bytes) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    return _decode_binary_batch(raw, expected_magic=BATCH_MAGIC)


def decode_stream_batch(raw: bytes) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    return _decode_binary_batch(raw, expected_magic=STREAM_MAGIC)


def parse_prepare_batch(
    header: Mapping[str, Any],
    arrays_by_lane: Sequence[Mapping[str, np.ndarray]],
    *,
    expected_contract: str = BATCH_CONTRACT,
    extra_header_keys: set[str] | None = None,
):
    exact_keys(header, {
        "contract_version", "kind", "batch_id", "lanes",
        "blob_codec", "raw_nbytes", "raw_sha256",
        *(extra_header_keys or set()),
    })
    if header["contract_version"] != expected_contract or header["kind"] not in {
        "prepare_batch", "exchange_batch",
    }:
        raise ContractError("prepare batch contract or kind mismatch")
    identifier(header["batch_id"], "batch_id")
    lanes = header["lanes"]
    if len(lanes) != len(arrays_by_lane):
        raise ContractError("prepare batch lane count mismatch")
    parsed = []
    seen = set()
    for lane, arrays in zip(lanes, arrays_by_lane):
        exact_keys(lane, {
            "contract_version", "idempotency_key", "step_key", "prepare_receipt",
            "capture_digest", "wire_digest", "beta", "metadata", "arrays",
        })
        if lane["contract_version"] != expected_contract:
            raise ContractError("lane contract_version mismatch")
        identity = identifier(lane["idempotency_key"], "idempotency_key")
        if identity in seen:
            raise ContractError("duplicate prepare idempotency key in batch")
        seen.add(identity)
        sha256_digest(lane["prepare_receipt"], "prepare_receipt")
        capture_digest = sha256_digest(lane["capture_digest"], "capture_digest")
        key = StepKey.parse(lane["step_key"])
        beta = lane["beta"]
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
            raise ContractError("beta must be finite")
        beta = float(beta)
        if beta != CANONICAL_BETA[key.round_id]:
            raise ContractError("beta differs from canonical round schedule")
        metadata = validate_metadata(
            lane["metadata"], key, beta, expected_contract=expected_contract
        )
        validate_lean_arrays(arrays, metadata["observation_source"])
        wire_digest = sha256_digest(lane["wire_digest"], "wire_digest")
        if lean_arrays_digest(metadata, arrays) != wire_digest:
            raise ContractError("wire_digest mismatch")
        if expected_contract == STREAM_CONTRACT:
            expected_capture = bytes_sha256(canonical_bytes({
                "contract_version": STREAM_CONTRACT,
                "step_key": key.as_dict(),
                "wire_digest": wire_digest,
            }))
            if capture_digest != expected_capture:
                raise ContractError("v4 capture_digest mismatch")
        parsed.append((key, beta, metadata, dict(arrays), dict(lane)))
    if len({(item[0].run_id, item[0].round_id, item[0].rollout_step_index) for item in parsed}) != 1:
        raise ContractError("all lanes in a prepare batch must share run, round, and vector step")
    return parsed
