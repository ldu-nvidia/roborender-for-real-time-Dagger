"""RDB4 framing for pipelined DAgger prepare/exchange batches.

RDB4 deliberately keeps the audited v3 ndarray representation: canonical JSON
header, globally contiguous little-endian C-order arrays, and a zlib level-one
blob.  Only the magic/content type and accepted semantic kinds change.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ContractError, canonical_json
from .ndarray_batch_v3 import pack_ndarray_batch_v3, unpack_ndarray_batch_v3


NDARRAY_BATCH_MAGIC_V4 = b"RDB4"
NDARRAY_BATCH_CONTENT_TYPE_V4 = "application/vnd.roborender.ndarray-batch.v4"
NDARRAY_BATCH_PREFIX_BYTES_V4 = 12
RDB4_BINARY_KINDS = frozenset({"prepare_batch", "exchange_batch"})


def _canonical_header_from_frame(body: bytes) -> tuple[dict[str, Any], bytes]:
    if not isinstance(body, bytes) or len(body) < NDARRAY_BATCH_PREFIX_BYTES_V4:
        raise ContractError("truncated RDB4 prefix")
    if body[:4] != NDARRAY_BATCH_MAGIC_V4:
        raise ContractError("invalid RDB4 magic")
    header_len = struct.unpack("<Q", body[4:12])[0]
    header_end = NDARRAY_BATCH_PREFIX_BYTES_V4 + header_len
    if header_end > len(body):
        raise ContractError("truncated RDB4 header")
    raw_header = body[12:header_end]
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("invalid RDB4 JSON header") from exc
    if not isinstance(header, dict) or canonical_json(header) != raw_header:
        raise ContractError("RDB4 header is not canonical JSON")
    return header, body[header_end:]


def pack_ndarray_batch_v4(
    header: Mapping[str, Any],
    record_arrays: Sequence[Mapping[str, np.ndarray]],
    *,
    max_header_bytes: int = 4 << 20,
    max_raw_blob_bytes: int = 64 << 20,
    max_compressed_blob_bytes: int = 64 << 20,
    max_body_bytes: int = 64 << 20,
    scratch: bytearray | None = None,
) -> bytes:
    """Pack a deterministic RDB4 body while reusing the audited RDB3 codec.

    ``scratch`` is optional caller-owned storage.  It avoids repeated bytearray
    allocation in a persistent per-gateway client, while the returned immutable
    bytes remain safe for idempotent retries.
    """

    kind = header.get("kind")
    if kind not in RDB4_BINARY_KINDS:
        raise ContractError("RDB4 kind must be prepare_batch or exchange_batch")
    if not isinstance(header.get("lanes"), list):
        raise ContractError("RDB4 binary header requires lanes")

    # The byte-level ndarray codec is intentionally identical to v3.  Present a
    # temporary v3 semantic kind, then restore the v4 kind and magic without
    # touching the compressed blob or its raw attestations.
    shim = dict(header)
    shim["kind"] = "prepare_batch"
    packed_v3 = pack_ndarray_batch_v3(
        shim,
        record_arrays,
        max_header_bytes=max_header_bytes,
        max_raw_blob_bytes=max_raw_blob_bytes,
        max_compressed_blob_bytes=max_compressed_blob_bytes,
        max_body_bytes=max_body_bytes,
    )
    header_len = struct.unpack("<Q", packed_v3[4:12])[0]
    header_end = 12 + header_len
    encoded = json.loads(packed_v3[12:header_end])
    encoded["kind"] = kind
    encoded_header = canonical_json(encoded)
    if len(encoded_header) > max_header_bytes:
        raise ContractError("RDB4 canonical header exceeds configured limit")
    total = 12 + len(encoded_header) + len(packed_v3[header_end:])
    if total > max_body_bytes:
        raise ContractError("encoded RDB4 body exceeds configured service limit")
    buffer = scratch if scratch is not None else bytearray()
    buffer.clear()
    buffer.extend(NDARRAY_BATCH_MAGIC_V4)
    buffer.extend(struct.pack("<Q", len(encoded_header)))
    buffer.extend(encoded_header)
    buffer.extend(packed_v3[header_end:])
    return bytes(buffer)


def unpack_ndarray_batch_v4(
    body: bytes,
    *,
    max_header_bytes: int = 4 << 20,
    max_raw_blob_bytes: int = 64 << 20,
    max_compressed_blob_bytes: int = 64 << 20,
    max_body_bytes: int = 64 << 20,
    max_arrays: int = 1024,
    max_rank: int = 5,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Strictly validate and decode one RDB4 body."""

    if len(body) > max_body_bytes:
        raise ContractError("RDB4 body exceeds configured service limit")
    header, compressed = _canonical_header_from_frame(body)
    kind = header.get("kind")
    if kind not in RDB4_BINARY_KINDS:
        raise ContractError("RDB4 kind must be prepare_batch or exchange_batch")
    if not isinstance(header.get("lanes"), list):
        raise ContractError("RDB4 binary header requires lanes")

    shim = dict(header)
    shim["kind"] = "prepare_batch"
    encoded_shim = canonical_json(shim)
    v3_body = b"".join(
        (
            b"RDB3",
            struct.pack("<Q", len(encoded_shim)),
            encoded_shim,
            compressed,
        )
    )
    _validated_shim, arrays = unpack_ndarray_batch_v3(
        v3_body,
        max_header_bytes=max_header_bytes,
        max_raw_blob_bytes=max_raw_blob_bytes,
        max_compressed_blob_bytes=max_compressed_blob_bytes,
        max_body_bytes=max_body_bytes,
        max_arrays=max_arrays,
        max_rank=max_rank,
    )
    return header, arrays
