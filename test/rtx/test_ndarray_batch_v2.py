from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from dagger_rtx.contracts import ContractError, canonical_json
from dagger_rtx.ndarray_batch_v2 import (
    NDARRAY_BATCH_CONTENT_TYPE,
    pack_ndarray_batch,
    unpack_ndarray_batch,
)


def _header():
    return {
        "contract_version": "roborender.dagger.rtx.v2",
        "kind": "prepare_batch",
        "batch_id": "batch-a",
        "lanes": [{"lane": 0}, {"lane": 1}],
    }


def test_rdb2_is_deterministic_raw_little_endian_and_round_trips():
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
    first = pack_ndarray_batch(_header(), groups_a)
    second = pack_ndarray_batch(_header(), groups_b)

    assert first == second
    assert first[:4] == b"RDB2"
    assert NDARRAY_BATCH_CONTENT_TYPE == "application/vnd.roborender.ndarray-batch.v2"
    assert b"data_b64" not in first
    header, decoded = unpack_ndarray_batch(first)
    assert [item["name"] for item in header["lanes"][0]["arrays"]] == ["a", "z"]
    assert header["lanes"][0]["arrays"][0]["offset"] == 0
    assert header["lanes"][1]["arrays"][0]["offset"] == (
        groups_a[0]["a"].nbytes + groups_a[0]["z"].nbytes
    )
    assert decoded[0]["a"].dtype.str == "<f4"
    np.testing.assert_array_equal(decoded[0]["a"], big_endian.astype("<f4"))
    np.testing.assert_array_equal(decoded[0]["z"], noncontiguous)
    np.testing.assert_array_equal(decoded[1]["x"], groups_a[1]["x"])


def test_rdb2_rejects_noncanonical_header_tamper_gap_truncation_and_trailer():
    body = pack_ndarray_batch(
        {
            "contract_version": "roborender.dagger.rtx.v2",
            "kind": "anchor_batch",
            "batch_id": "anchors-a",
            "anchors": [{"anchor_sha256": "a" * 64}],
        },
        [{"anchor_rgb__head_color": np.zeros((2, 3, 3), np.uint8)}],
    )
    header_len = struct.unpack("<Q", body[4:12])[0]
    header_end = 12 + header_len
    header = json.loads(body[12:header_end])

    tampered = bytearray(body)
    tampered[-1] ^= 1
    with pytest.raises(ContractError, match="SHA-256"):
        unpack_ndarray_batch(bytes(tampered))
    with pytest.raises(ContractError, match="trailing"):
        unpack_ndarray_batch(body + b"x")
    with pytest.raises(ContractError, match="beyond"):
        unpack_ndarray_batch(body[:-1])

    header["anchors"][0]["arrays"][0]["offset"] = 1
    rewritten = canonical_json(header)
    gap = b"RDB2" + struct.pack("<Q", len(rewritten)) + rewritten + body[header_end:]
    with pytest.raises(ContractError, match="contiguous"):
        unpack_ndarray_batch(gap)

    pretty = json.dumps(json.loads(body[12:header_end]), indent=2).encode()
    noncanonical = b"RDB2" + struct.pack("<Q", len(pretty)) + pretty + body[header_end:]
    with pytest.raises(ContractError, match="canonical"):
        unpack_ndarray_batch(noncanonical)
    with pytest.raises(ContractError, match="prefix"):
        unpack_ndarray_batch(b"RDB2")
    with pytest.raises(ContractError, match="magic"):
        unpack_ndarray_batch(b"BAD!" + body[4:])


def test_rdb2_production_b28_payload_budget_regression():
    state = (113 + 226 + 226 + 516 + 3 + 26 + 15) * 4
    controls = 3 * (240 * 416 + 240 * 416 * 3)
    raytraced = 3 * 108 * 192 * 3 * 4
    generated_lane = state + controls
    raytraced_lane = generated_lane + raytraced
    anchor_triplet = 3 * 240 * 416 * 3

    assert generated_lane == 1_202_580
    assert raytraced_lane == 1_949_076
    assert anchor_triplet == 898_560
    assert generated_lane * 28 == 33_672_240
    assert raytraced_lane * 28 == 54_574_128
    assert raytraced_lane * 28 < 64 << 20
    assert generated_lane * 56 > 64 << 20


def test_codec_enforces_total_body_limit_not_only_blob_limit():
    with pytest.raises(ContractError, match="service limit"):
        pack_ndarray_batch(
            {
                "contract_version": "roborender.dagger.rtx.v2",
                "kind": "anchor_batch",
                "batch_id": "small-limit",
                "anchors": [{"anchor_sha256": "a" * 64}],
            },
            [{"anchor_rgb__head_color": np.zeros((8,), np.uint8)}],
            max_blob_bytes=1024,
            max_body_bytes=16,
        )
