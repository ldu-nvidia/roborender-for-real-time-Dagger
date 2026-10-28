from __future__ import annotations
from test.deployment.source_paths import RTX_ROOT

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    MODEL_VIEW_MAP,
    RAVEN_VIEW_ORDER,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
    payload_digest,
)
from dagger_rtx.lean_boundary_v3 import (
    CONTRACT_VERSION_V3,
    FIXED_PREPARE_ARRAYS_V3,
    anchor_records_from_prepares_v3,
    build_prepare_batch_v3,
    expected_prepare_array_names_v3,
    project_minimal_arrays_v3,
)
from dagger_rtx.ndarray_batch_v3 import unpack_ndarray_batch_v3


def _local(
    profile: RavenShapeProfile,
    source: str,
    *,
    lane: int = 0,
):
    key = StepKey(
        run_id="lean-v3-test",
        round_id=1,
        actor_id=f"lane-{lane:03d}",
        rollout_window_id="window-000000",
        rollout_step_index=0,
        episode_id=f"episode-{lane:03d}",
        episode_step_index=0,
    )
    homie = np.arange(15, dtype=np.float32) + lane
    arrays = {
        "actor_obs": np.full((profile.actor_obs,), lane, np.float32),
        "teacher_obs": np.full((profile.teacher_obs,), lane + 1, np.float32),
        "critic_obs": np.full((profile.critic_obs,), lane + 2, np.float32),
        "homie_obs": np.full((profile.homie_obs,), lane + 3, np.float32),
        "gt_obj_pos": np.asarray([lane, 1, 2], np.float32),
        "teacher_action": np.concatenate(
            (np.arange(11, dtype=np.float32), homie)
        ).astype(np.float32),
        "homie_action": homie,
    }
    anchors = {}
    for view_index, view in enumerate(RAVEN_VIEW_ORDER):
        anchors[f"anchor_rgb__{view}"] = np.full(
            (profile.control_h, profile.control_w, 3),
            lane + view_index + 10,
            np.uint8,
        )
        arrays[f"depth_metric__{view}"] = np.full(
            (profile.depth_h, profile.depth_w), lane + 0.5, np.float32
        )
        arrays[f"depth_rawq__{view}"] = np.full(
            (profile.depth_h, profile.depth_w), lane, np.uint16
        )
        arrays[f"depth_raw__{view}"] = np.full(
            (profile.control_h, profile.control_w), lane + 0.5, np.float16
        )
        arrays[f"depth_ctrl__{view}"] = np.full(
            (profile.control_h, profile.control_w), lane + view_index, np.uint8
        )
        arrays[f"rgb_robot__{view}"] = np.full(
            (profile.control_h, profile.control_w, 3),
            lane + view_index,
            np.uint8,
        )
        if source == "raytraced":
            arrays[f"raytraced_rgb__{view}"] = np.full(
                (profile.policy_h, profile.policy_w, 3),
                (lane + view_index + 1) / 10.0,
                np.float32,
            )
    arrays.update(anchors)
    assignment = {
        "use_raytraced": source == "raytraced",
        "anchor_sha256": anchor_rgb_digest(anchors),
        "raven_source": "/home/ldu/BC_render/rtx_BC_transfer",
        "raven_commit": "94d79851d9c471b1210be81c75116d9bea835a4d",
        "observation_source": source,
        "assignment_source": "pinned_raven_id_sample_anchor",
    }
    metadata = {
        "view_order": list(RAVEN_VIEW_ORDER),
        "model_view_map": dict(MODEL_VIEW_MAP),
        "observation_source": source,
        "prompt": f"lane {lane}",
        "env_id": lane,
        "seq": 0,
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
        "contract_version": CONTRACT_VERSION,
        "step_key": key.as_dict(),
        "beta": 0.75,
        "episode_assignment": assignment,
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "step_key": key.as_dict(),
        "request_id": f"prepare-{lane}",
        "prepare_receipt": f"{lane + 1:064x}",
        "capture_digest": payload_digest(metadata, arrays),
        "metadata": metadata,
        "arrays": arrays,
    }


def test_v3_generated_projection_is_exact_source_specific_set(tiny_profile):
    local = _local(tiny_profile, "generated_controls")
    projected = project_minimal_arrays_v3(local, shape_profile=tiny_profile)
    expected = set(FIXED_PREPARE_ARRAYS_V3)
    for view in RAVEN_VIEW_ORDER:
        expected.update({f"depth_ctrl__{view}", f"rgb_robot__{view}"})

    assert set(projected) == expected
    assert len(projected) == 8
    assert "homie_action" not in projected
    assert set(expected_prepare_array_names_v3("generated_controls")) == expected
    assert not any(
        name.startswith(
            (
                "depth_metric__",
                "depth_rawq__",
                "depth_raw__",
                "anchor_rgb__",
                "raytraced_rgb__",
            )
        )
        for name in projected
    )
    assert not {"teacher_obs", "critic_obs", "homie_obs", "gt_obj_pos"} & set(
        projected
    )


def test_v3_raytraced_projection_sends_only_rgb_beyond_three_fixed(tiny_profile):
    local = _local(tiny_profile, "raytraced")
    projected = project_minimal_arrays_v3(local, shape_profile=tiny_profile)
    expected = set(FIXED_PREPARE_ARRAYS_V3) | {
        f"raytraced_rgb__{view}" for view in RAVEN_VIEW_ORDER
    }

    assert set(projected) == expected
    assert len(projected) == 5
    assert "homie_action" not in projected
    assert not any(
        name.startswith(("depth_ctrl__", "rgb_robot__", "anchor_rgb__"))
        for name in projected
    )


def test_v3_projection_validates_shapes_ranges_and_homie_tail(tiny_profile):
    local = _local(tiny_profile, "raytraced")
    local["arrays"]["raytraced_rgb__head_color"][0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        project_minimal_arrays_v3(local, shape_profile=tiny_profile)

    local = _local(tiny_profile, "generated_controls")
    local["arrays"]["actor_obs"] = np.zeros((tiny_profile.actor_obs + 1,), np.float32)
    with pytest.raises(ValueError, match="actor_obs shape"):
        project_minimal_arrays_v3(local, shape_profile=tiny_profile)

    local = _local(tiny_profile, "generated_controls")
    local["arrays"]["teacher_action"][-1] += 1
    with pytest.raises(ValueError, match=r"teacher_action\[11:\]"):
        build_prepare_batch_v3(
            [local], ["h200-v3-prepare-lane-0"], shape_profile=tiny_profile
        )


def test_v3_prepare_carries_opaque_capture_digest_and_wire_attestations(
    tiny_profile,
):
    generated = _local(tiny_profile, "generated_controls", lane=0)
    raytraced = _local(tiny_profile, "raytraced", lane=1)
    batch_id, body, records, groups = build_prepare_batch_v3(
        [generated, raytraced],
        ["h200-v3-prepare-lane-0", "h200-v3-prepare-lane-1"],
        shape_profile=tiny_profile,
    )
    header, decoded = unpack_ndarray_batch_v3(body)

    assert batch_id == header["batch_id"]
    assert header["contract_version"] == CONTRACT_VERSION_V3
    assert header["blob_codec"] == "zlib-1"
    assert "raw_nbytes" in header and "raw_sha256" in header
    assert [row["step_key"] for row in records] == [
        generated["step_key"],
        raytraced["step_key"],
    ]
    assert [row["capture_digest"] for row in header["lanes"]] == [
        generated["capture_digest"],
        raytraced["capture_digest"],
    ]
    assert all("wire_digest" in row for row in header["lanes"])
    assert set(decoded[0]) == set(groups[0])
    assert set(decoded[1]) == set(groups[1])

    changed = copy.deepcopy(generated)
    changed["arrays"]["teacher_obs"][0] += 100
    changed["capture_digest"] = payload_digest(
        changed["metadata"], changed["arrays"]
    )
    assert changed["capture_digest"] != generated["capture_digest"]
    assert set(project_minimal_arrays_v3(changed, shape_profile=tiny_profile)) == set(
        project_minimal_arrays_v3(generated, shape_profile=tiny_profile)
    )
    _, _, changed_records, changed_groups = build_prepare_batch_v3(
        [changed], ["h200-v3-prepare-lane-0"], shape_profile=tiny_profile
    )
    assert changed_records[0]["wire_digest"] == records[0]["wire_digest"]
    for name in changed_groups[0]:
        np.testing.assert_array_equal(changed_groups[0][name], groups[0][name])


def test_v3_anchor_registration_filters_out_raytraced_lanes(tiny_profile):
    generated = _local(tiny_profile, "generated_controls", lane=0)
    raytraced = _local(tiny_profile, "raytraced", lane=1)

    records, groups = anchor_records_from_prepares_v3(
        [raytraced, generated], shape_profile=tiny_profile
    )
    assert len(records) == len(groups) == 1
    assert records[0]["anchor_sha256"] == generated["metadata"]["episode_assignment"][
        "anchor_sha256"
    ]


def test_v3_machine_readable_contract_matches_implementation():
    root = RTX_ROOT
    contract = json.loads(
        (root / "config" / "render_student_boundary.v3.json").read_text()
    )
    assert contract["contract_version"] == CONTRACT_VERSION_V3
    assert contract["framing"]["magic_ascii"] == "RDB3"
    assert contract["framing"]["blob_codec"] == "zlib-1"
    assert contract["prepare_arrays"]["source_exclusivity"][
        "generated_controls"
    ] == sorted(expected_prepare_array_names_v3("generated_controls"))
    assert contract["prepare_arrays"]["source_exclusivity"]["raytraced"] == sorted(
        expected_prepare_array_names_v3("raytraced")
    )
    assert contract["sharding"]["gateway_urls"] == [
        f"http://127.0.0.1:{port}" for port in range(18620, 18628)
    ]
