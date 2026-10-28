"""Deterministic compressed-ndarray container for the lean DAgger v3 boundary.

Wire framing::

    b"RDB3" || uint64_le(len(canonical_header)) || canonical_header
             || zlib_level_1(raw_blob)

``raw_blob`` is the no-padding concatenation of canonical little-endian,
C-contiguous ndarray bytes. Descriptor offsets are relative to the
*uncompressed* blob and remain globally contiguous across lanes in vector
order. The canonical header binds the complete raw blob with ``raw_nbytes``
and ``raw_sha256`` in addition to retaining every per-array descriptor/hash.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any, Mapping, Sequence
import zlib

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


NDARRAY_BATCH_MAGIC_V3 = b"RDB3"
NDARRAY_BATCH_CONTENT_TYPE_V3 = "application/vnd.roborender.ndarray-batch.v3"
NDARRAY_BATCH_PREFIX_BYTES_V3 = len(NDARRAY_BATCH_MAGIC_V3) + 8
ZLIB_LEVEL_V3 = 1
ZLIB_CODEC_V3 = "zlib-1"
DESCRIPTOR_KEYS_V3 = frozenset(
    {"name", "dtype", "shape", "sha256", "offset", "nbytes"}
)
_RAW_HEADER_KEYS = frozenset({"blob_codec", "raw_nbytes", "raw_sha256"})


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = canonical_json(value)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("RDB3 header must be canonical-JSON compatible") from exc
    if not isinstance(decoded, dict):
        raise ContractError("RDB3 header must be an object")
    return decoded


def _record_key(header: Mapping[str, Any]) -> str:
    kind = header.get("kind")
    if kind == "prepare_batch":
        return "lanes"
    if kind == "anchor_batch":
        return "anchors"
    raise ContractError("RDB3 kind must be prepare_batch or anchor_batch")


def pack_ndarray_batch_v3(
    header: Mapping[str, Any],
    record_arrays: Sequence[Mapping[str, np.ndarray]],
    *,
    max_header_bytes: int = 4 << 20,
    max_raw_blob_bytes: int = 64 << 20,
    max_compressed_blob_bytes: int = 64 << 20,
    max_body_bytes: int = 64 << 20,
) -> bytes:
    """Pack one deterministic RFC1950-zlib RDB3 body at compression level 1."""

    value = _json_copy(header)
    if _RAW_HEADER_KEYS & set(value):
        raise ContractError("caller must not supply RDB3 raw blob attestations")
    key = _record_key(value)
    records = value.get(key)
    if not isinstance(records, list) or len(records) != len(record_arrays):
        raise ContractError(f"{key} and record_arrays must have equal lengths")
    if not 1 <= len(records) <= 256:
        raise ContractError("RDB3 must contain between 1 and 256 records")

    offset = 0
    chunks: list[bytes] = []
    for record_index, (record, arrays) in enumerate(
        zip(records, record_arrays, strict=True)
    ):
        if not isinstance(record, dict):
            raise ContractError(f"{key}[{record_index}] must be an object")
        if "arrays" in record:
            raise ContractError("caller must not supply RDB3 array descriptors")
        if not isinstance(arrays, Mapping) or not arrays:
            raise ContractError(f"{key}[{record_index}] arrays must be non-empty")
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
            if offset > max_raw_blob_bytes:
                raise ContractError("RDB3 raw blob exceeds configured limit")
        record["arrays"] = descriptors

    raw_blob = b"".join(chunks)
    if not raw_blob:
        raise ContractError("RDB3 raw blob must be non-empty")
    value["blob_codec"] = ZLIB_CODEC_V3
    value["raw_nbytes"] = len(raw_blob)
    value["raw_sha256"] = sha256_bytes(raw_blob)
    encoded_header = canonical_json(value)
    if len(encoded_header) > max_header_bytes:
        raise ContractError("RDB3 canonical header exceeds configured limit")
    compressed = zlib.compress(raw_blob, level=ZLIB_LEVEL_V3)
    if len(compressed) > max_compressed_blob_bytes:
        raise ContractError("RDB3 compressed blob exceeds configured limit")
    if NDARRAY_BATCH_PREFIX_BYTES_V3 + len(encoded_header) + len(compressed) > max_body_bytes:
        raise ContractError("encoded RDB3 body exceeds configured service limit")
    return b"".join(
        (
            NDARRAY_BATCH_MAGIC_V3,
            struct.pack("<Q", len(encoded_header)),
            encoded_header,
            compressed,
        )
    )


def _decompress_exact(
    compressed: bytes,
    *,
    declared_raw_nbytes: int,
    max_raw_blob_bytes: int,
) -> bytes:
    if declared_raw_nbytes > max_raw_blob_bytes:
        raise ContractError("RDB3 declared raw blob exceeds configured limit")
    decoder = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    try:
        raw = decoder.decompress(compressed, declared_raw_nbytes + 1)
    except zlib.error as exc:
        raise ContractError("invalid RDB3 zlib stream") from exc
    if len(raw) > declared_raw_nbytes:
        raise ContractError("RDB3 expands beyond declared raw_nbytes")
    # Never call decompressobj.flush() here: with a forged-small declaration,
    # flush would expand the remaining compressed input without a length cap.
    if decoder.unconsumed_tail:
        raise ContractError("RDB3 expands beyond declared raw_nbytes")
    if not decoder.eof:
        raise ContractError("truncated or over-expanding RDB3 zlib stream")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise ContractError("RDB3 zlib stream has trailing or unconsumed bytes")
    if len(raw) != declared_raw_nbytes:
        raise ContractError("RDB3 raw_nbytes does not match expanded blob")
    return raw


def unpack_ndarray_batch_v3(
    body: bytes,
    *,
    max_header_bytes: int = 4 << 20,
    max_raw_blob_bytes: int = 64 << 20,
    max_compressed_blob_bytes: int = 64 << 20,
    max_body_bytes: int = 64 << 20,
    max_arrays: int = 1024,
    max_rank: int = 5,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Strictly validate, decompress, attest, and decode one RDB3 body."""

    if not isinstance(body, bytes):
        raise ContractError("RDB3 body must be bytes")
    if len(body) > max_body_bytes:
        raise ContractError("RDB3 body exceeds configured service limit")
    if len(body) < NDARRAY_BATCH_PREFIX_BYTES_V3:
        raise ContractError("truncated RDB3 prefix")
    if body[:4] != NDARRAY_BATCH_MAGIC_V3:
        raise ContractError("invalid RDB3 magic")
    header_len = struct.unpack("<Q", body[4:12])[0]
    if header_len > max_header_bytes:
        raise ContractError("RDB3 header exceeds configured limit")
    header_end = NDARRAY_BATCH_PREFIX_BYTES_V3 + header_len
    if header_end > len(body):
        raise ContractError("truncated RDB3 header")
    raw_header = body[NDARRAY_BATCH_PREFIX_BYTES_V3:header_end]
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("invalid RDB3 JSON header") from exc
    try:
        is_canonical = isinstance(header, dict) and canonical_json(header) == raw_header
    except (TypeError, ValueError):
        is_canonical = False
    if not is_canonical:
        raise ContractError("RDB3 header is not canonical JSON")
    if not _RAW_HEADER_KEYS <= set(header):
        raise ContractError("RDB3 header is missing raw blob attestations")
    if header.get("blob_codec") != ZLIB_CODEC_V3:
        raise ContractError("RDB3 blob_codec must be zlib-1")
    raw_nbytes = header.get("raw_nbytes")
    if isinstance(raw_nbytes, bool) or not isinstance(raw_nbytes, int) or raw_nbytes <= 0:
        raise ContractError("RDB3 raw_nbytes must be a positive integer")
    raw_sha256 = require_sha256(header.get("raw_sha256"), "RDB3 raw_sha256")
    compressed = body[header_end:]
    if not compressed:
        raise ContractError("RDB3 compressed blob is missing")
    if len(compressed) > max_compressed_blob_bytes:
        raise ContractError("RDB3 compressed blob exceeds configured limit")
    raw_blob = _decompress_exact(
        compressed,
        declared_raw_nbytes=raw_nbytes,
        max_raw_blob_bytes=max_raw_blob_bytes,
    )
    if sha256_bytes(raw_blob) != raw_sha256:
        raise ContractError("RDB3 raw blob SHA-256 mismatch")

    key = _record_key(header)
    records = header.get(key)
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ContractError(f"RDB3 {key} must contain 1..256 records")
    blob = memoryview(raw_blob)
    decoded_records: list[dict[str, np.ndarray]] = []
    expected_offset = 0
    descriptor_count = 0
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"{key}[{record_index}] must be an object")
        descriptors = record.get("arrays")
        if not isinstance(descriptors, list) or not descriptors:
            raise ContractError(f"{key}[{record_index}] descriptors must be non-empty")
        if any(not isinstance(item, Mapping) for item in descriptors):
            raise ContractError("invalid RDB3 ndarray descriptor")
        names = [item.get("name") for item in descriptors]
        if any(not isinstance(name, str) for name in names):
            raise ContractError("RDB3 descriptor names must be strings")
        if names != sorted(names) or len(set(names)) != len(names):
            raise ContractError("RDB3 descriptors must be unique and lexically sorted per record")
        arrays: dict[str, np.ndarray] = {}
        for descriptor in descriptors:
            descriptor_count += 1
            if descriptor_count > max_arrays:
                raise ContractError("RDB3 contains too many arrays")
            if not isinstance(descriptor, Mapping) or set(descriptor) != DESCRIPTOR_KEYS_V3:
                raise ContractError("invalid RDB3 ndarray descriptor keys")
            name = require_identifier(descriptor["name"], "array name", 96)
            dtype_name = descriptor["dtype"]
            if not isinstance(dtype_name, str) or dtype_name not in ALLOWED_WIRE_DTYPES:
                raise ContractError(f"RDB3 array {name} dtype is not allowed")
            shape = descriptor["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) > max_rank
                or any(
                    isinstance(n, bool) or not isinstance(n, int) or n < 0
                    for n in shape
                )
            ):
                raise ContractError(f"RDB3 array {name} shape is invalid")
            dtype = np.dtype(dtype_name)
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<")
            expected_nbytes = math.prod(shape) * dtype.itemsize
            offset = descriptor["offset"]
            nbytes = descriptor["nbytes"]
            if isinstance(offset, bool) or not isinstance(offset, int) or offset != expected_offset:
                raise ContractError("RDB3 descriptor offsets must be globally contiguous")
            if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes != expected_nbytes:
                raise ContractError(f"RDB3 array {name} nbytes does not match dtype and shape")
            end = offset + nbytes
            if end > len(blob):
                raise ContractError(f"RDB3 array {name} extends beyond the raw blob")
            expected_sha = require_sha256(descriptor["sha256"], f"{name} sha256")
            raw = blob[offset:end]
            if sha256_bytes(raw.tobytes()) != expected_sha:
                raise ContractError(f"RDB3 array {name} SHA-256 mismatch")
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
            expected_offset = end
        decoded_records.append(arrays)
    if expected_offset != len(blob):
        raise ContractError("RDB3 raw blob has trailing or unreferenced bytes")
    return header, decoded_records
