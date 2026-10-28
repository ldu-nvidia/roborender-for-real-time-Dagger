from copy import deepcopy
import hashlib
import importlib.machinery
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pipeline.deployment.rtx_actor.native_motion_assignment import (
    SCHEMA, NativeEpisodeAssignmentRecorder, assert_motion_environment,
    configure_motion_environment, explicit_scene_motion_binding,
    install_native_assignment_observer, load_motion_assignment, validate_motion_assignment,
)
from pipeline.deployment.rtx_actor.native_artifact_io import canonical


def fixture(tmp_path):
    lanes = []
    for lane in range(2):
        path = tmp_path / f"motion-{lane}.npz"
        path.write_bytes(bytes([lane + 1]) * 3)
        lanes.append({"lane": lane, "motion_id": f"motion-{lane}", "path": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assignment = {"schema": SCHEMA, "run_id": "run", "cohort_id": "cohort", "purpose": "validation",
        "simulator_seed": 42, "motion_inventory_sha256": "f" * 64, "lanes": lanes}
    config = SimpleNamespace(num_envs=99,
        simulator=SimpleNamespace(config=SimpleNamespace(scene=SimpleNamespace(num_envs=99))),
        reset_from_dataset=SimpleNamespace(use_motion_file_dir=True, motion_files=[], motion_assignment_offset=7))
    configure_motion_environment(config, assignment)
    return assignment, config


def key(episode="episode0", run="run", index=0):
    return SimpleNamespace(run_id=run, episode_id=episode, episode_step_index=index)


def test_explicit_configuration_and_immutable_load(tmp_path):
    assignment, config = fixture(tmp_path)
    assert assert_motion_environment(config, assignment) == assignment
    assert config.num_envs == config.simulator.config.scene.num_envs == 2
    assert config.reset_from_dataset.motion_assignment_offset == 0
    path = tmp_path / "assignment.json"
    raw = json.dumps(assignment).encode()
    path.write_bytes(raw)
    assert load_motion_assignment(path, hashlib.sha256(raw).hexdigest()) == assignment
    with pytest.raises(ValueError, match="changed"):
        load_motion_assignment(path, "0" * 64)
    config.reset_from_dataset.motion_files.reverse()
    with pytest.raises(ValueError, match="configuration"):
        assert_motion_environment(config, assignment)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(purpose="unknown"),
    lambda x: x.update(simulator_seed=True),
    lambda x: x["lanes"][0].update(lane=1),
    lambda x: x["lanes"][0].update(path="relative"),
    lambda x: x["lanes"][1].update(sha256=x["lanes"][0]["sha256"]),
    lambda x: x["lanes"][1].update(motion_id=x["lanes"][0]["motion_id"]),
    lambda x: x.update(unexpected=True),
])
def test_assignment_rejects_ambiguous_or_duplicate_lanes(tmp_path, mutation):
    assignment, _ = fixture(tmp_path)
    mutation(assignment)
    with pytest.raises(ValueError):
        validate_motion_assignment(assignment)


def test_scene_hook_binds_same_motion_geometry_and_restores_global_hooks(tmp_path, monkeypatch):
    assignment, config = fixture(tmp_path)
    module = ModuleType("test_scenario")
    module.__file__ = str(tmp_path / "scenario.py")
    Path(module.__file__).write_text("# fixture\n")
    original_metadata = lambda *a: "implicit-order"
    module.get_table_grid_meta_data = original_metadata
    module.table_top_material_prim_paths = [123]
    module.leg_material_prim_paths = [456]
    calls = []
    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "exec_module", lambda loader, mod: None)
    original_exec = importlib.machinery.SourceFileLoader.exec_module
    loader = importlib.machinery.SourceFileLoader("test_scenario", module.__file__)
    motion_loader = SimpleNamespace(from_file=lambda p, device: (
        calls.append((p, device)) or SimpleNamespace(table_custom_data={"path": p})))
    with explicit_scene_motion_binding(config, assignment, module, motion_loader) as evidence:
        loader.exec_module(module)
        for lane in range(2):
            assert module.get_table_grid_meta_data(config, lane, "cpu")["path"] == assignment["lanes"][lane]["path"]
        assert evidence["queried_lanes"] == [0, 1]
        assert module.table_top_material_prim_paths == ["123"]
    assert module.get_table_grid_meta_data is original_metadata
    assert importlib.machinery.SourceFileLoader.exec_module is original_exec
    assert len(calls) == 2


def test_episode_receipt_requires_real_reset_and_matches_export_contract(tmp_path):
    assignment, _ = fixture(tmp_path)
    recorder = NativeEpisodeAssignmentRecorder(assignment, tmp_path / "receipts")
    with pytest.raises(ValueError, match="newly observed"):
        recorder.bind(0, key(), {"use_raytraced": False})
    with pytest.raises(ValueError, match="declared lane"):
        recorder.observe_task_reset([0], [12], [0], [(0, 10), (10, 20)], "a" * 64)
    recorder.observe_task_reset([0, 1], [2, 12], [0, 1], [(0, 10), (10, 20)], "a" * 64)
    recorder.observe_episode_reset([0, 1])
    entry = recorder.bind(0, key(), {"use_raytraced": False})
    raw = Path(entry["native_assignment_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["native_assignment_sha256"]
    assert json.loads(raw) == {"schema": "roborender.native-episode-assignment.v1", "purpose": "validation",
        "run_id": "run", "episode_id": "episode0", "simulator_seed": 42, "motion_id": "motion-0",
        "motion_sha256": assignment["lanes"][0]["sha256"], "observation_source": "generated_controls"}
    assert recorder.bind(0, key(), {"use_raytraced": False}) == entry
    with pytest.raises(ValueError, match="changed observation"):
        recorder.bind(0, key(), {"use_raytraced": True})
    with pytest.raises(ValueError, match="newly observed"):
        recorder.bind(0, key("invented-next-episode"), {"use_raytraced": False})
    recorder.observe_task_reset([0], [3], [1], [(0, 10), (10, 20)], "b" * 64)
    recorder.observe_episode_reset([0])
    with pytest.raises(ValueError, match="reused identity"):
        recorder.bind(0, key(), {"use_raytraced": False})
    recorder.bind(0, key("real-next-episode"), {"use_raytraced": True})
    assert len(recorder.episode_assignments) == 2


def test_observer_only_observes_native_resets_and_restores_methods(tmp_path):
    assignment, config = fixture(tmp_path)
    calls = []
    env = SimpleNamespace(config=config,
        reset_dataset=SimpleNamespace(sample_idx_based_on_demo_clip_number=[(0, 10), (10, 20)],
            motion_files=[r["path"] for r in assignment["lanes"]]),
        simulator=SimpleNamespace(native_motion_assignment_sha256=hashlib.sha256(canonical(assignment)).hexdigest()))
    sentinel = object()

    def task(lanes):
        calls.append(("task", lanes))
        env.sample_idx = [2 + lane * 10 for lane in lanes]
        env.sample_idx_in_seq = [0] * len(lanes)
        return sentinel

    def reset(lanes):
        calls.append(("episode", lanes))
        env._reset_task_states(lanes)
        return sentinel

    env._reset_task_states, env.reset_envs_idx = task, reset
    trainer = SimpleNamespace(env=env)
    with install_native_assignment_observer(trainer, assignment, tmp_path / "observer",
            rng_fingerprint=lambda: "d" * 64) as recorder:
        assert calls == []
        assert env.reset_envs_idx([0, 1]) is sentinel
        assert calls == [("episode", [0, 1]), ("task", [0, 1])]
        trainer._native_episode_assignment_observer(0, key(), {"use_raytraced": True})
        assert len(recorder.episode_assignments) == 1
    assert env._reset_task_states is task and env.reset_envs_idx is reset
    assert not hasattr(trainer, "_native_episode_assignment_observer")
    assert json.loads((tmp_path / "observer/index.json").read_bytes())["actual_episode_resets_per_lane"] == [1, 1]


def test_observer_b2_native_subset_reset_lane_one_only(tmp_path):
    """Raven replaces sample arrays with subset-shaped results, not B-wide state."""
    assignment, config = fixture(tmp_path)
    calls = []
    env = SimpleNamespace(config=config,
        reset_dataset=SimpleNamespace(sample_idx_based_on_demo_clip_number=[(0, 10), (10, 20)],
            motion_files=[r["path"] for r in assignment["lanes"]]),
        simulator=SimpleNamespace(native_motion_assignment_sha256=hashlib.sha256(canonical(assignment)).hexdigest()))

    def task(lanes):
        calls.append(("task", list(lanes)))
        # Same storage semantics as Raven's reset_from_dataset_wsppt.py:
        # fresh len(env_ids) samples, ordered to correspond to env_ids.
        env.sample_idx = [3 + lane * 10 for lane in lanes]
        env.sample_idx_in_seq = [4] * len(lanes)

    def reset(lanes):
        calls.append(("episode", list(lanes)))
        env._reset_task_states(lanes)

    env._reset_task_states, env.reset_envs_idx = task, reset
    trainer = SimpleNamespace(env=env)
    with install_native_assignment_observer(trainer, assignment, tmp_path / "subset-observer",
            rng_fingerprint=lambda: "e" * 64) as recorder:
        env.reset_envs_idx([0, 1])
        lane0 = recorder.bind(0, key("lane0-first"), {"use_raytraced": True})
        recorder.bind(1, key("lane1-first"), {"use_raytraced": False})
        env.reset_envs_idx([1])
        assert env.sample_idx == [13] and env.sample_idx_in_seq == [4]
        assert recorder.episode_resets == [1, 2]
        assert recorder.bind(0, key("lane0-first"), {"use_raytraced": True}) == lane0
        lane1 = recorder.bind(1, key("lane1-next"), {"use_raytraced": True})
        assert lane1["motion_id"] == "motion-1"
        audit = json.loads(Path(lane1["native_assignment_path"]).with_suffix(".reset.json").read_bytes())
        assert audit["lane"] == 1 and audit["sample_idx"] == 13
        assert audit["sample_idx_in_seq"] == 4 and audit["native_episode_reset_index"] == 1
        assert calls == [("episode", [0, 1]), ("task", [0, 1]), ("episode", [1]), ("task", [1])]
    assert env._reset_task_states is task and env.reset_envs_idx is reset


def test_production_namespace_binds_actual_episode_and_local_audit(tmp_path):
    assignment, _ = fixture(tmp_path)
    recorder = NativeEpisodeAssignmentRecorder(assignment, tmp_path / 'namespaced',
        episode_namespace='prod-r1-e01-c000')
    recorder.observe_task_reset([0], [2], [4], [(0, 10), (10, 20)], 'a' * 64)
    recorder.observe_episode_reset([0])
    episode = 'prod-r1-e01-c000-episode-000-00000000'
    entry = recorder.bind(0, key(episode), {'use_raytraced': True})
    receipt = json.loads(Path(entry['native_assignment_path']).read_bytes())
    audit = json.loads(Path(entry['native_assignment_path']).with_suffix('.reset.json').read_bytes())
    assert receipt['episode_id'] == episode
    assert audit['service_episode_namespace'] == 'prod-r1-e01-c000'
    assert audit['local_service_episode_id'] == 'episode-000-00000000'
    assert audit['sample_idx'] == 2 and audit['sample_idx_in_seq'] == 4
    assert recorder.bind(0, key(episode), {'use_raytraced': True}) == entry


def test_invalid_production_episode_namespace_writes_no_receipt(tmp_path):
    assignment, _ = fixture(tmp_path)
    directory = tmp_path / 'rejected-namespace'
    recorder = NativeEpisodeAssignmentRecorder(assignment, directory, episode_namespace='prod-r1-e01-c000')
    recorder.observe_task_reset([0], [2], [4], [(0, 10), (10, 20)], 'a' * 64)
    recorder.observe_episode_reset([0])
    with pytest.raises(ValueError, match='cohort scope'):
        recorder.bind(0, key('episode-000-00000000'), {'use_raytraced': True})
    assert list(directory.glob('*.json')) == []
    assert not recorder.episode_assignments and recorder.bound_resets == [0, 0]
