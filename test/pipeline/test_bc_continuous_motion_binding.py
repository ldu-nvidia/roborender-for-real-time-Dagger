"""The reference reset loader retains unassigned motions and their RNG draws."""
import hashlib
import importlib.machinery
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pipeline.deployment.rtx_actor.native_artifact_io import canonical
from pipeline.deployment.rtx_actor.native_motion_assignment import explicit_motion_files_scene_binding


def fixture(tmp_path):
    inventory = []
    for i in range(4):
        path = tmp_path / f"motion-{i}.npz"
        path.write_bytes(bytes([i + 1]))
        inventory.append(dict(motion_id=f"m{i}", path=str(path),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    cfg = SimpleNamespace(num_envs=2,
        simulator=SimpleNamespace(config=SimpleNamespace(scene=SimpleNamespace(num_envs=2))),
        reset_from_dataset=SimpleNamespace(use_motion_file_dir=False,
            motion_files=[row["path"] for row in inventory], motion_assignment_offset=0))
    scenario = ModuleType("scenario")
    scenario.__file__ = str(tmp_path / "scenario.py")
    Path(scenario.__file__).write_text("# test scenario\n")
    scenario.get_table_grid_meta_data = lambda *args: "original"
    scenario.table_top_material_prim_paths = []
    scenario.leg_material_prim_paths = []
    motion_loader = SimpleNamespace(from_file=lambda p, device: SimpleNamespace(table_custom_data=p))
    return inventory, cfg, scenario, motion_loader


def test_complete_reset_inventory_preserved_but_geometry_uses_lane_prefix(tmp_path, monkeypatch):
    inventory, cfg, scenario, motion_loader = fixture(tmp_path)
    original = scenario.get_table_grid_meta_data
    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "exec_module", lambda *_: None)
    loader = importlib.machinery.SourceFileLoader("scenario", scenario.__file__)
    with explicit_motion_files_scene_binding(cfg, inventory[:2], scenario, motion_loader,
                                            motion_inventory=inventory) as evidence:
        loader.exec_module(scenario)
        assert cfg.reset_from_dataset.motion_files == [r["path"] for r in inventory]
        assert scenario.get_table_grid_meta_data(cfg, 1, "cpu") == inventory[1]["path"]
        assert evidence["motion_inventory_sha256"] == hashlib.sha256(canonical(inventory)).hexdigest()
        assert evidence["ordered_motion_sha256"] == hashlib.sha256(canonical(inventory[:2])).hexdigest()
        with pytest.raises(ValueError, match="undeclared"):
            scenario.get_table_grid_meta_data(cfg, 2, "cpu")
    assert scenario.get_table_grid_meta_data is original


@pytest.mark.parametrize("change", ["reordered_lanes", "truncated_loader", "unused_file_changed"])
def test_inventory_cannot_be_silently_reduced_or_changed(tmp_path, change):
    inventory, cfg, scenario, motion_loader = fixture(tmp_path)
    motions = inventory[:2]
    if change == "reordered_lanes":
        motions.reverse()
    elif change == "truncated_loader":
        cfg.reset_from_dataset.motion_files = cfg.reset_from_dataset.motion_files[:2]
    else:
        Path(inventory[-1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        with explicit_motion_files_scene_binding(cfg, motions, scenario, motion_loader,
                                                motion_inventory=inventory):
            pytest.fail("invalid inventory admitted")
