from test.deployment.source_paths import REPO_ROOT
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import materialize_isaac6 as port
import roborender_isaac6 as rr6


def test_quaternion_roundtrip_non_identity():
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.7, 0.5, 0.2, -0.1]])
    original = q.clone()
    torch.testing.assert_close(rr6.xyzw(q), q[:, [1, 2, 3, 0]])
    torch.testing.assert_close(rr6.wxyz(rr6.xyzw(q)), q)
    torch.testing.assert_close(q, original)
    assert rr6.xyzw((1, 0, 0, 0)) == (0, 0, 0, 1)
    assert rr6.wxyz([0, 0, 0, 1]) == [1, 0, 0, 0]
    assert rr6.xyzw(None) is None


def test_state_conversion_preserves_position_velocity_and_input():
    state = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    original = state.clone()
    converted = rr6.root_state(state, to_lab=True)
    torch.testing.assert_close(converted[:, :3], state[:, :3])
    torch.testing.assert_close(converted[:, 7:], state[:, 7:])
    torch.testing.assert_close(rr6.root_state(converted, to_lab=False), state)
    torch.testing.assert_close(state, original)
    with pytest.raises(ValueError):
        rr6.root_state(torch.zeros(12), to_lab=True)


def test_data_view_converts_only_quaternions():
    state = torch.arange(13, dtype=torch.float32).reshape(1, 13)
    joint = torch.tensor([[1.5, 0.4]])
    q = torch.tensor([[0.2, 0.3, 0.4, 0.1]])
    asset = SimpleNamespace(data=SimpleNamespace(root_state_w=state, joint_pos=joint, body_quat_w=q))
    legacy = rr6.data(asset)
    assert legacy.joint_pos is joint
    torch.testing.assert_close(legacy.body_quat_w, torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    torch.testing.assert_close(legacy.root_state_w[:, 3:7], state[:, [6, 3, 4, 5]])


def test_write_boundary_reorders_once():
    records = []
    asset = SimpleNamespace(write_root_state_to_sim=lambda state, ids: records.append((state, ids)))
    state = torch.arange(13, dtype=torch.float32).reshape(1, 13)
    rr6.write_root_state(asset, state, [0])
    torch.testing.assert_close(records[0][0], rr6.root_state(state, to_lab=True))
    assert records[0][1] == [0]


def test_ast_edits_leave_other_operations_unchanged():
    source = "a = self.robot.data.root_state_w\nb = a + 1\nself.robot.write_root_state_to_sim(a, ids)\n"
    result = port.boundary_edits(source, simulator=True)
    assert "rr6.data(self.robot).root_state_w" in result
    assert "rr6.write_root_state(self.robot, a, ids)" in result
    assert "b = a + 1\n" in result
    ast.parse(result)


def test_integer_semantics_sets_only_the_two_camera_nodes():
    calls = []
    camera = SimpleNamespace(
        _renderer=SimpleNamespace(cfg=SimpleNamespace(colorize_semantic_segmentation=False)),
        _render_data=SimpleNamespace(annotators={"semantic_segmentation": object()},
                                     render_product_paths=["/Render/Product_head"]),
    )
    native = SimpleNamespace(set_node_attributes=lambda *args: calls.append(args))
    rr6.require_integer_semantic_ids(camera, synthetic_data=native)
    assert calls == [
        ("SemanticSegmentationPostRender", {"inputs:colorize": False}, "/Render/Product_head"),
        ("semantic_segmentation", {"inputs:colorize": False}, "/Render/Product_head"),
    ]
    camera._renderer.cfg.colorize_semantic_segmentation = True
    with pytest.raises(RuntimeError, match="integer semantic IDs"):
        rr6.require_integer_semantic_ids(camera, synthetic_data=native)
    assert len(calls) == 2


def test_materialized_sources_are_explicit_derivatives(tmp_path):
    repo = REPO_ROOT
    output = tmp_path / "overlay"
    result = port.materialize(repo, output, repo / "pipeline/dagger/rtx/config/assets.manifest.json")
    assert result["production_qualified"] is False
    assert len(result["files"]) == 12
    assert any(row['path'].endswith('/legged_base_task/legged_robot_base.py')
               for row in result['files'])
    assert result["initial_reset_contract"] == "roborender.initial-reset-completion.v1"
    assert port.sha((output / "initial_reset_contract.py").read_bytes()) == result["initial_reset_helper_sha256"]
    manifest = json.loads((output / "assets.manifest.json").read_text())
    artifacts = {row["role"]: row for row in manifest["artifacts"]}
    assert artifacts["isaac_capture"]["source_sha256"] == port.PINS["gr00t/rl/simulator/isaacsim/isaacsim.py"]
    assert artifacts["isaac_capture"]["sha256"] != artifacts["isaac_capture"]["source_sha256"]
    assert "source_sha256" not in artifacts["raven_teacher_homie_trainer"]
    simulator = (output / port.SOURCE_PREFIX / "gr00t/rl/simulator/isaacsim/isaacsim.py").read_text()
    assert "obj_cfg.spawn.spawn_paths = [" in simulator
    assert "f'/World/envs/env_{i}/{name}'" in simulator
    assert "for i in range(self.scene.cfg.num_envs)" in simulator
    assert "translation=wpos, orientation=rr6.xyzw(wrot)," in simulator
    assert "isinstance(self._task[name], rr6.asset_base('RigidObject'))" in simulator
    assert "self._robot = Articulation(robot_articulation_config)" in simulator
    assert "renderer_cfg=IsaacRtxRendererCfg(colorize_semantic_segmentation=False)" in simulator
    assert "rr6.require_integer_semantic_ids(_camera)" in simulator
    grid = (output / port.SOURCE_PREFIX / "gr00t/rl/isaac_utils/playground/env_rand/table_grid.py").read_text()
    assert "orientation = prim_utils.wxyz(orientation)" in grid
    for row in result["files"]:
        assert port.sha((output / row["path"]).read_bytes()) == row["derived_sha256"]
        assert port.sha((repo / row["path"]).read_bytes()) == row["source_sha256"]
    with pytest.raises(FileExistsError):
        port.materialize(repo, output, repo / "pipeline/dagger/rtx/config/assets.manifest.json")


def test_unknown_source_fails_before_output_creation(tmp_path):
    source = tmp_path / "source"
    target = source / port.SOURCE_PREFIX / next(iter(port.PINS))
    target.parent.mkdir(parents=True)
    target.write_text("# unknown revision\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source hash mismatch"):
        port.materialize(source, output, tmp_path / "unused.json")
    assert not output.exists()
