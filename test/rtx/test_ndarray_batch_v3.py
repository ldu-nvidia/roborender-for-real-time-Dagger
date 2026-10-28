from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from dagger_rtx.contracts import ContractError, canonical_json, sha256_bytes
from dagger_rtx.ndarray_batch_v3 import (
    NDARRAY_BATCH_CONTENT_TYPE_V3,
    NDARRAY_BATCH_MAGIC_V3,
    pack_ndarray_batch_v3,
    unpack_ndarray_batch_v3,
)


def _header():
    return {
        "contract_version": "roborender.dagger.rtx.v3",
        "kind": "prepare_batch",
        "batch_id": "batch-a",
        "lanes": [{"lane": 0}, {"lane": 1}],
    }


def _parts(body: bytes):
    header_len = struct.unpack("<Q", body[4:12])[0]
    end = 12 + header_len
    return json.loads(body[12:end]), body[end:]


def _rewrite(header, compressed):
    encoded = canonical_json(header)
    return b"RDB3" + struct.pack("<Q", len(encoded)) + encoded + compressed


def test_rdb3_is_deterministic_level1_zlib_and_round_trips():
    big_endian = np.asarray([1.25, -2.5], dtype=">f4")
    noncontiguous = np.arange(12, dtype=np.uint8).reshape(3, 4)[:, ::2]
    groups_a = [
        {"z": noncontiguous, "a": big_endian},
        {"x": np.asarray([7, 8], dtype=np.uint16)},
    ]
    groups_b = [
        {"a": big_endian, "z": noncontiguous},
        {"x": np.asarray([7, 8], dtype=np.uint16)},
    ]
    first = pack_ndarray_batch_v3(_header(), groups_a)
    second = pack_ndarray_batch_v3(_header(), groups_b)

    assert first == second
    assert first[:4] == NDARRAY_BATCH_MAGIC_V3 == b"RDB3"
    assert (
        NDARRAY_BATCH_CONTENT_TYPE_V3
        == "application/vnd.roborender.ndarray-batch.v3"
    )
    header, compressed = _parts(first)
    raw = zlib.decompress(compressed)
    assert compressed == zlib.compress(raw, level=1)
    assert header["blob_codec"] == "zlib-1"
    assert header["raw_nbytes"] == len(raw)
    assert header["raw_sha256"] == sha256_bytes(raw)
    assert [item["name"] for item in header["lanes"][0]["arrays"]] == ["a", "z"]
    assert header["lanes"][0]["arrays"][0]["offset"] == 0
    assert header["lanes"][1]["arrays"][0]["offset"] == (
        groups_a[0]["a"].nbytes + groups_a[0]["z"].nbytes
    )

    decoded_header, decoded = unpack_ndarray_batch_v3(first)
    assert decoded_header == header
    assert decoded[0]["a"].dtype.str == "<f4"
    np.testing.assert_array_equal(decoded[0]["a"], big_endian.astype("<f4"))
    np.testing.assert_array_equal(decoded[0]["z"], noncontiguous)
    np.testing.assert_array_equal(decoded[1]["x"], groups_a[1]["x"])


def test_rdb3_rejects_noncanonical_header_bad_attestations_and_zlib_damage():
    body = pack_ndarray_batch_v3(
        {
            "contract_version": "roborender.dagger.rtx.v3",
            "kind": "anchor_batch",
            "batch_id": "anchors-a",
            "anchors": [{"anchor_sha256": "a" * 64}],
        },
        [{"anchor_rgb__head_color": np.zeros((2, 3, 3), np.uint8)}],
    )
    header, compressed = _parts(body)

    wrong_size = dict(header)
    wrong_size["raw_nbytes"] = header["raw_nbytes"] - 1
    with pytest.raises(ContractError, match="expands beyond|over-expanding"):
        unpack_ndarray_batch_v3(_rewrite(wrong_size, compressed))

    wrong_hash = dict(header)
    wrong_hash["raw_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="raw blob SHA-256"):
        unpack_ndarray_batch_v3(_rewrite(wrong_hash, compressed))

    wrong_codec = dict(header)
    wrong_codec["blob_codec"] = "gzip"
    with pytest.raises(ContractError, match="blob_codec"):
        unpack_ndarray_batch_v3(_rewrite(wrong_codec, compressed))

    with pytest.raises(ContractError, match="truncated|invalid"):
        unpack_ndarray_batch_v3(body[:-1])
    with pytest.raises(ContractError, match="trailing"):
        unpack_ndarray_batch_v3(body + b"x")
    with pytest.raises(ContractError, match="trailing"):
        unpack_ndarray_batch_v3(body + zlib.compress(b"another", level=1))

    pretty = json.dumps(header, indent=2).encode()
    noncanonical = b"RDB3" + struct.pack("<Q", len(pretty)) + pretty + compressed
    with pytest.raises(ContractError, match="canonical"):
        unpack_ndarray_batch_v3(noncanonical)
    with pytest.raises(ContractError, match="prefix"):
        unpack_ndarray_batch_v3(b"RDB3")
    with pytest.raises(ContractError, match="magic"):
        unpack_ndarray_batch_v3(b"BAD!" + body[4:])


def test_rdb3_rejects_descriptor_tamper_even_with_valid_whole_blob_attestation():
    body = pack_ndarray_batch_v3(
        _header(),
        [
            {"a": np.asarray([1, 2], np.uint8)},
            {"b": np.asarray([3, 4], np.uint8)},
        ],
    )
    header, compressed = _parts(body)

    stale_array_hash = json.loads(json.dumps(header))
    stale_array_hash["lanes"][0]["arrays"][0]["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="array a SHA-256"):
        unpack_ndarray_batch_v3(_rewrite(stale_array_hash, compressed))

    gap = json.loads(json.dumps(header))
    gap["lanes"][1]["arrays"][0]["offset"] += 1
    with pytest.raises(ContractError, match="globally contiguous"):
        unpack_ndarray_batch_v3(_rewrite(gap, compressed))

    duplicate = json.loads(json.dumps(header))
    duplicate["lanes"][0]["arrays"].append(
        dict(duplicate["lanes"][0]["arrays"][0])
    )
    with pytest.raises(ContractError, match="unique and lexically sorted"):
        unpack_ndarray_batch_v3(_rewrite(duplicate, compressed))


def test_rdb3_lane_order_is_preserved_and_bound():
    groups = [
        {"actor_obs": np.asarray([1], np.float32)},
        {"actor_obs": np.asarray([2], np.float32)},
    ]
    body = pack_ndarray_batch_v3(_header(), groups)
    _, decoded = unpack_ndarray_batch_v3(body)
    assert float(decoded[0]["actor_obs"][0]) == 1.0
    assert float(decoded[1]["actor_obs"][0]) == 2.0

    swapped = pack_ndarray_batch_v3(_header(), list(reversed(groups)))
    assert swapped != body


def test_rdb3_codec_enforces_expanded_and_total_body_limits():
    with pytest.raises(ContractError, match="raw blob exceeds"):
        pack_ndarray_batch_v3(
            {
                "contract_version": "roborender.dagger.rtx.v3",
                "kind": "anchor_batch",
                "batch_id": "raw-limit",
                "anchors": [{"anchor_sha256": "a" * 64}],
            },
            [{"anchor_rgb__head_color": np.zeros((128,), np.uint8)}],
            max_raw_blob_bytes=64,
        )
    with pytest.raises(ContractError, match="service limit"):
        pack_ndarray_batch_v3(
            {
                "contract_version": "roborender.dagger.rtx.v3",
                "kind": "anchor_batch",
                "batch_id": "body-limit",
                "anchors": [{"anchor_sha256": "a" * 64}],
            },
            [{"anchor_rgb__head_color": np.zeros((8,), np.uint8)}],
            max_raw_blob_bytes=1024,
            max_compressed_blob_bytes=1024,
            max_body_bytes=16,
        )


def test_rdb3_forged_small_raw_length_cannot_trigger_unbounded_inflate():
    body = pack_ndarray_batch_v3(
        {
            "contract_version": "roborender.dagger.rtx.v3",
            "kind": "anchor_batch",
            "batch_id": "inflate-guard",
            "anchors": [{"anchor_sha256": "a" * 64}],
        },
        [{"anchor_rgb__head_color": np.zeros((4 << 20,), np.uint8)}],
    )
    header, compressed = _parts(body)
    header["raw_nbytes"] = 1
    with pytest.raises(ContractError, match="expands beyond"):
        unpack_ndarray_batch_v3(_rewrite(header, compressed))


def test_rdb3_malformed_descriptor_types_and_empty_blob_fail_as_contract_errors():
    body = pack_ndarray_batch_v3(
        _header(),
        [
            {"a": np.asarray([1], np.uint8)},
            {"b": np.asarray([2], np.uint8)},
        ],
    )
    header, compressed = _parts(body)
    bad_name = json.loads(json.dumps(header))
    bad_name["lanes"][0]["arrays"][0]["name"] = ["not", "a", "name"]
    with pytest.raises(ContractError, match="names must be strings"):
        unpack_ndarray_batch_v3(_rewrite(bad_name, compressed))

    bad_dtype = json.loads(json.dumps(header))
    bad_dtype["lanes"][0]["arrays"][0]["dtype"] = ["uint8"]
    with pytest.raises(ContractError, match="dtype is not allowed"):
        unpack_ndarray_batch_v3(_rewrite(bad_dtype, compressed))

    with pytest.raises(ContractError, match="non-empty"):
        pack_ndarray_batch_v3(
            {
                "contract_version": "roborender.dagger.rtx.v3",
                "kind": "anchor_batch",
                "batch_id": "empty",
                "anchors": [{"anchor_sha256": "a" * 64}],
            },
            [{"empty": np.empty((0,), np.uint8)}],
        )
