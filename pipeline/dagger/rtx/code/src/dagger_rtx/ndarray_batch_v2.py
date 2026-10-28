"""Deterministic raw-ndarray container for the lean DAgger v2 boundary.

The wire representation is intentionally small and dependency-free::

    b"RDB2" || uint64_le(len(canonical_header)) || canonical_header || raw_blob

The header is canonical JSON, but ndarray payloads are contiguous raw bytes;
they are never base64 encoded. Descriptor offsets are relative to the first
byte of raw_blob and are contiguous across every record in vector order.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ALLOWED_WIRE_DTYPES,
    ContractError,
    canonical_json,
    canonical_wire_array,
    require_identifier,
    require_sha256,
    sha256_bytes,
)


NDARRAY_BATCH_MAGIC = b"RDB2"
NDARRAY_BATCH_CONTENT_TYPE = "application/vnd.roborender.ndarray-batch.v2"
NDARRAY_BATCH_PREFIX_BYTES = len(NDARRAY_BATCH_MAGIC) + 8
DESCRIPTOR_KEYS = frozenset({"name", "dtype", "shape", "sha256", "offset", "nbytes"})


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON header while rejecting non-JSON and non-finite values."""

    try:
        encoded = canonical_json(value)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("ndarray batch header must be canonical-JSON compatible") from exc
    if not isinstance(decoded, dict):
        raise ContractError("ndarray batch header must be an object")
    return decoded


def _record_key(header: Mapping[str, Any]) -> str:
    kind = header.get("kind")
    if kind == "prepare_batch":
        return "lanes"
    if kind == "anchor_batch":
        return "anchors"
    raise ContractError("binary ndarray batch kind must be prepare_batch or anchor_batch")


def pack_ndarray_batch(
    header: Mapping[str, Any],
    record_arrays: Sequence[Mapping[str, np.ndarray]],
    *,
    max_header_bytes: int = 4 << 20,
    max_blob_bytes: int = 64 << 20,
    max_body_bytes: int = 64 << 20,
) -> bytes:
    """Pack a deterministic v2 container and inject record descriptors."""

    value = _json_copy(header)
    key = _record_key(value)
    records = value.get(key)
    if not isinstance(records, list) or len(records) != len(record_arrays):
        raise ContractError(f"{key} and record_arrays must have equal lengths")
    if not 1 <= len(records) <= 256:
        raise ContractError("ndarray batch must contain between 1 and 256 records")

    offset = 0
    chunks: list[bytes] = []
    for record_index, (record, arrays) in enumerate(zip(records, record_arrays, strict=True)):
        if not isinstance(record, dict):
            raise ContractError(f"{key}[{record_index}] must be an object")
        if "arrays" in record:
            raise ContractError("caller must not supply arrays descriptors; the codec derives them")
        if not isinstance(arrays, Mapping) or not arrays:
            raise ContractError(f"{key}[{record_index}] arrays must be a non-empty mapping")
        descriptors: list[dict[str, Any]] = []
        for name in sorted(arrays):
            require_identifier(name, "array name", 96)
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
            if offset > max_blob_bytes:
                raise ContractError("ndarray batch raw blob exceeds configured limit")
        record["arrays"] = descriptors

    encoded_header = canonical_json(value)
    if len(encoded_header) > max_header_bytes:
        raise ContractError("ndarray batch canonical header exceeds configured limit")
    if NDARRAY_BATCH_PREFIX_BYTES + len(encoded_header) + offset > max_body_bytes:
        raise ContractError("encoded ndarray batch exceeds the 64 MiB service limit")
    return b"".join(
        (
            NDARRAY_BATCH_MAGIC,
            struct.pack("<Q", len(encoded_header)),
            encoded_header,
            *chunks,
        )
    )


def unpack_ndarray_batch(
    body: bytes,
    *,
    max_header_bytes: int = 4 << 20,
    max_blob_bytes: int = 64 << 20,
    max_arrays: int = 1024,
    max_rank: int = 5,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Strictly validate and decode a v2 container."""

    if not isinstance(body, bytes):
        raise ContractError("ndarray batch body must be bytes")
    if len(body) < NDARRAY_BATCH_PREFIX_BYTES:
        raise ContractError("truncated ndarray batch prefix")
    if body[:4] != NDARRAY_BATCH_MAGIC:
        raise ContractError("invalid ndarray batch magic")
    header_len = struct.unpack("<Q", body[4:12])[0]
    if header_len > max_header_bytes:
        raise ContractError("ndarray batch header exceeds configured limit")
    header_end = NDARRAY_BATCH_PREFIX_BYTES + header_len
    if header_end > len(body):
        raise ContractError("truncated ndarray batch header")
    raw_header = body[NDARRAY_BATCH_PREFIX_BYTES:header_end]
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid ndarray batch JSON header") from exc
    if not isinstance(header, dict) or canonical_json(header) != raw_header:
        raise ContractError("ndarray batch header is not canonical JSON")
    key = _record_key(header)
    records = header.get(key)
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ContractError(f"ndarray batch {key} must contain 1..256 records")
    blob = memoryview(body)[header_end:]
    if len(blob) > max_blob_bytes:
        raise ContractError("ndarray batch raw blob exceeds configured limit")

    decoded_records: list[dict[str, np.ndarray]] = []
    expected_offset = 0
    descriptor_count = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"{key}[{record_index}] must be an object")
        descriptors = record.get("arrays")
        if not isinstance(descriptors, list) or not descriptors:
            raise ContractError(f"{key}[{record_index}] descriptors must be non-empty")
        names = [item.get("name") if isinstance(item, Mapping) else None for item in descriptors]
        if names != sorted(names) or len(set(names)) != len(names):
            raise ContractError("array descriptors must be unique and lexically sorted per record")
        arrays: dict[str, np.ndarray] = {}
        for descriptor in descriptors:
            descriptor_count += 1
            if descriptor_count > max_arrays:
                raise ContractError("ndarray batch contains too many arrays")
            if not isinstance(descriptor, Mapping) or set(descriptor) != DESCRIPTOR_KEYS:
                raise ContractError("invalid ndarray descriptor keys")
            name = require_identifier(descriptor["name"], "array name", 96)
            dtype_name = descriptor["dtype"]
            if dtype_name not in ALLOWED_WIRE_DTYPES:
                raise ContractError(f"array {name} dtype is not allowed")
            shape = descriptor["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) > max_rank
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
            ):
                raise ContractError(f"array {name} shape is invalid")
            dtype = np.dtype(dtype_name)
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<")
            expected_nbytes = math.prod(shape) * dtype.itemsize
            offset = descriptor["offset"]
            nbytes = descriptor["nbytes"]
            if isinstance(offset, bool) or not isinstance(offset, int) or offset != expected_offset:
                raise ContractError("ndarray descriptor offsets must be globally contiguous")
            if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes != expected_nbytes:
                raise ContractError(f"array {name} nbytes does not match dtype and shape")
            end = offset + nbytes
            if end > len(blob):
                raise ContractError(f"array {name} extends beyond the raw blob")
            expected_sha = require_sha256(descriptor["sha256"], f"{name} sha256")
            raw = blob[offset:end]
            if sha256_bytes(raw.tobytes()) != expected_sha:
                raise ContractError(f"array {name} SHA-256 mismatch")
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
            expected_offset = end
        decoded_records.append(arrays)
    if expected_offset != len(blob):
        raise ContractError("ndarray batch has trailing or unreferenced raw bytes")
    return header, decoded_records
