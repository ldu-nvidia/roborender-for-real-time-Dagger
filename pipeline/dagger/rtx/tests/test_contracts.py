from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from dagger_rtx.boundary import build_h200_prepare, selection_receipt_body, validate_h200_selection
from dagger_rtx.contracts import (
    MODEL_VIEW_MAP,
    RAVEN_VIEW_ORDER,
    ContractError,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    decode_array,
    encode_array,
    sha256_bytes,
    require_identifier,
)
from dagger_rtx.fake_h200 import FakeH200State

from conftest import PROJECT_ROOT, make_key, make_owner, prepare


def test_online_view_order_and_model_mapping_regression():
    assert RAVEN_VIEW_ORDER == ("head_color", "hand_left", "hand_right")
    assert MODEL_VIEW_MAP == {"ext1": "head_color", "ext2": "hand_left", "wrist": "hand_right"}
    assert MODEL_VIEW_MAP != {"ext1": "hand_left", "ext2": "hand_right", "wrist": "head_color"}


def test_step_key_separates_window_and_episode_indices():
    key = make_key(7, episode_id="episode-b", episode_step=2)
    assert StepKey.from_mapping(key.as_dict()) == key
    bad = key.as_dict()
    bad["round_id"] = "D1"
    with pytest.raises(ContractError):
        StepKey.from_mapping(bad)


def test_array_envelope_rejects_tamper_and_nan_action():
    envelope = encode_array(np.arange(26, dtype=np.float32))
    assert np.array_equal(decode_array(envelope), np.arange(26, dtype=np.float32))
    tampered = dict(envelope)
    tampered["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="sha256"):
        decode_array(tampered)


def test_wire_arrays_are_canonical_little_endian_and_dot_paths_are_rejected():
    big_endian = np.asarray([1.25, -2.5], dtype=">f4")
    decoded = decode_array(encode_array(big_endian))
    assert decoded.dtype.name == "float32"
    assert decoded.tobytes() == np.asarray([1.25, -2.5], dtype="<f4").tobytes()
    for value in (".", ".."):
        with pytest.raises(ContractError, match="dot path"):
            require_identifier(value, "unsafe")


def test_machine_schema_accepts_actual_prepare_and_response(tmp_path):
    owner, _ = make_owner(tmp_path, RavenShapeProfile())
    local = prepare(owner, make_key())
    remote = build_h200_prepare(local, "h200-prepare-key")
    fake = FakeH200State("token", selection_draw=0.99)
    response = fake.prepare(remote)
    schema_path = PROJECT_ROOT / "config/render_student_boundary.v1.schema.json"
    schema = json.loads(schema_path.read_text())
    # Resolve local refs by validating through a root wrapper.
    Draft202012Validator({"$ref": "#/$defs/prepareRequest", "$defs": schema["$defs"]}).validate(remote)
    Draft202012Validator({"$ref": "#/$defs/prepareResponse", "$defs": schema["$defs"]}).validate(response)
    validate_h200_selection(response, local, "h200-prepare-key")


def test_h200_must_concatenate_supplied_homie_exactly(tmp_path, tiny_profile):
    owner, _ = make_owner(tmp_path, tiny_profile)
    local = prepare(owner, make_key())
    remote = build_h200_prepare(local, "h200-prepare-key")
    response = FakeH200State("token", selection_draw=0.99).prepare(remote)
    selected = decode_array(response["selected_action"])
    selected[-1] += np.float32(1)
    response["selected_action"] = encode_array(selected)
    response["selection_receipt"] = sha256_bytes(canonical_json(selection_receipt_body(response)))
    with pytest.raises(ContractError, match="student11.*Homie"):
        validate_h200_selection(response, local, "h200-prepare-key")


def test_teacher_tail_must_equal_homie(tmp_path, tiny_profile):
    owner, _ = make_owner(tmp_path, tiny_profile)
    backend = owner.backend
    original = backend.capture_current

    def broken(*, key):
        capture = original(key=key)
        capture.arrays["teacher_action"][-1] += np.float32(1)
        return capture

    backend.capture_current = broken
    with pytest.raises(ContractError, match="Homie"):
        prepare(owner, make_key())
