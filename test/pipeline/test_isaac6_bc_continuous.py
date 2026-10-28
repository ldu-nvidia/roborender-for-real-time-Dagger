"""CPU-only continuous-collection contracts and original-writer differentials."""
from test.deployment.source_paths import REPO_ROOT
import ast
from copy import deepcopy
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from pipeline.deployment.rtx_actor import isaac6_bc_production as bc


def request(tmp_path, *, windows=4, lanes=2):
    inventory = [{"motion_id": f"motion-{i}", "path": str(tmp_path / f"motion-{i}.npz"),
                  "sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(lanes + 2)]
    return dict(schema=bc.SCHEMA_CONTINUOUS, run_id="continuous-test", cohort_id="train",
        split="train", process_seed=7341, rank=0, global_env_offset=0,
        motions=inventory[:lanes], motion_inventory=inventory, max_windows=windows,
        output_dir=str(tmp_path / "chunks"), sampling_plan_sha256="a" * 64)


def reset(ledger, lane):
    ledger.task_reset(lane, sample_idx=lane, sample_idx_in_seq=4, rng_sha256="f" * 64)
    ledger.episode_reset(lane)


def ledger_for(tmp_path, **kwargs):
    ledger = bc.ChunkProvenanceLedger(request(tmp_path, **kwargs))
    for lane in range(ledger.num_envs):
        reset(ledger, lane)
    return ledger


def step(ledger, dones=None):
    dones = dones or [False] * ledger.num_envs
    ledger.begin_transition([r["episode_step_index"] + 23 for r in ledger.current])
    for lane, done in enumerate(dones):
        if done:
            reset(ledger, lane)
    ledger.finish_transition(dones)


def publish_pending(ledger, journal):
    for emitted in ledger.pending():
        path = Path(ledger.request["output_dir"]) / f"chunk-{journal.count:07d}.npz"
        path.write_bytes(b"CPU fixture; not native payload evidence")
        record = ledger.describe(emitted, path)
        provenance = path.with_suffix(".provenance.json")
        bc.publish_json(provenance, record)
        journal.append(record, provenance)
        ledger.acknowledge(emitted)
    return journal.accept_window(journal.count)


def test_continuous_request_has_no_fake_quotas_and_keeps_full_inventory(tmp_path):
    raw = request(tmp_path)
    assert bc.validate_collection_request(raw) == raw
    assert "chunks_per_env" not in raw
    overrides = bc.native_overrides(raw)
    assert overrides["num_envs"] == "2"
    assert json.loads(overrides["env.config.reset_from_dataset.motion_files"]) == [
        row["path"] for row in raw["motion_inventory"]]
    assert overrides["algo.trl.num_total_batches"] == "4"
    assert overrides["algo.config.num_learning_epochs"] == "0"
    assert overrides["algo.config.init_at_random_ep_len"] == "true"
    with pytest.raises(ValueError, match="no ordinal"):
        bc.selected_chunk_ordinals(raw)


@pytest.mark.parametrize("field,value", [("max_windows", 0), ("max_windows", True),
    ("max_windows", 50001), ("rank", 1), ("global_env_offset", 2), ("split", "validation"),
    ("chunks_per_env", 10), ("selected_chunk_ordinals_per_env", [[0], [0]]),
    ("motion_inventory", [])])
def test_continuous_strict_single_scene_request(tmp_path, field, value):
    raw = request(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError):
        bc.validate_collection_request(raw)


def test_inventory_prefix_order_unique_and_required(tmp_path):
    raw = request(tmp_path)
    assert bc.validate_collection_request({**raw, "max_windows": 50000})["max_windows"] == 50000
    for broken in (
            {key: value for key, value in raw.items() if key != "motion_inventory"},
            {**raw, "motions": raw["motions"][::-1]},
            {**raw, "motion_inventory": raw["motion_inventory"] + [raw["motions"][0]]}):
        with pytest.raises(ValueError):
            bc.validate_collection_request(broken)
    duplicate_content = deepcopy(raw)
    duplicate_content["motion_inventory"][-1]["sha256"] = raw["motions"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        bc.validate_collection_request(duplicate_content)
    oversized = deepcopy(raw)
    oversized["motion_inventory"] += [raw["motion_inventory"][-1]] * 10000
    with pytest.raises(ValueError, match="inventory"):
        bc.validate_collection_request(oversized)


def test_all_emissions_across_many_windows_without_reset_or_quotas(tmp_path):
    ledger = ledger_for(tmp_path, windows=20)
    Path(ledger.request["output_dir"]).mkdir()
    journal = bc.ContinuousCollectionJournal(ledger)
    episodes = deepcopy(ledger.current)
    previous = None
    try:
        for window in range(1, 21):
            for _ in range(81):
                step(ledger)
            progress = publish_pending(ledger, journal)
            assert progress["accepted_windows"] == window
            assert progress["previous_receipt_sha256"] == previous
            previous = journal.last_receipt["sha256"]
            assert ledger.pending() == [] and ledger.buffers == [[], []]
            assert ledger.reset_counts == [1, 1]
            assert not hasattr(journal, "records")
            with pytest.raises(ValueError, match="quota"):
                ledger.quota_reached()
        journal.require_completed(native_saved_chunks=40, native_tail_rows=[0, 0], initial_reset_all_calls=1)
        assert [r["episode_id"] for r in ledger.current] == [r["episode_id"] for r in episodes]
        manifest = dict(schema=bc.CONTINUOUS_MANIFEST_SCHEMA, **journal.bindings())
        records = list(bc.iter_continuous_chunks(manifest))
        assert len(records) == 40
        assert all(r["selected_for_split"] and not r["overcollection"] and "rows" not in r for r in records)
        assert {r["emitted_at_window"] for r in records} == set(range(1, 21))
        with pytest.raises(ValueError, match="budget"):
            step(ledger)
        with pytest.raises(RuntimeError, match="exact order"):
            journal.accept_window(40)
    finally:
        journal.close()


def test_completion_preserves_natural_terminal_and_final_partial_tails(tmp_path):
    ledger = ledger_for(tmp_path, windows=2)
    Path(ledger.request["output_dir"]).mkdir()
    journal = bc.ContinuousCollectionJournal(ledger)
    try:
        for index in range(162):
            step(ledger, [index == 39, index == 80])
            if (index + 1) % 81 == 0:
                publish_pending(ledger, journal)
        assert ledger.discarded_tail_rows == [40, 0]
        assert ledger.emitted_counts == [1, 2]
        assert [len(rows) for rows in ledger.buffers] == [41, 0]
        assert ledger.sequences == [3, 4]  # row81 emits first, then advances2 on done.
        before = deepcopy(ledger.buffers)
        journal.require_completed(native_saved_chunks=3, native_tail_rows=[41, 0], initial_reset_all_calls=1)
        assert ledger.buffers == before  # No final partial-tail emission or flush.
    finally:
        journal.close()


@pytest.mark.parametrize("fault", ["short", "pending", "inflight", "native_count", "native_tail",
    "extra_reset", "missing_progress", "accounting"])
def test_normal_completion_rejects_incomplete_or_inconsistent_state(tmp_path, fault):
    ledger = ledger_for(tmp_path, windows=2, lanes=1)
    Path(ledger.request["output_dir"]).mkdir()
    journal = bc.ContinuousCollectionJournal(ledger)
    try:
        for window in range(1 if fault == "short" else 2):
            for _ in range(81):
                step(ledger)
            if fault != "pending":
                publish_pending(ledger, journal)
        kwargs = dict(native_saved_chunks=journal.count, native_tail_rows=[0], initial_reset_all_calls=1)
        if fault == "inflight": ledger.inflight = [{}]
        elif fault == "native_count": kwargs["native_saved_chunks"] += 1
        elif fault == "native_tail": kwargs["native_tail_rows"] = [1]
        elif fault == "extra_reset": kwargs["initial_reset_all_calls"] = 2
        elif fault == "missing_progress": journal.accepted_windows -= 1
        elif fault == "accounting": ledger.discarded_tail_rows[0] = 1
        with pytest.raises(RuntimeError):
            journal.require_completed(**kwargs)
    finally:
        journal.close()


def test_index_tampering_is_not_a_valid_complete_population(tmp_path):
    ledger = ledger_for(tmp_path, windows=1, lanes=1)
    Path(ledger.request["output_dir"]).mkdir()
    journal = bc.ContinuousCollectionJournal(ledger)
    for _ in range(81): step(ledger)
    publish_pending(ledger, journal)
    journal.close()
    manifest = dict(schema=bc.CONTINUOUS_MANIFEST_SCHEMA, **journal.bindings())
    broken = deepcopy(manifest)
    broken["chunk_index"]["count"] += 1
    with pytest.raises(ValueError, match="count"):
        list(bc.iter_continuous_chunks(broken))
    with journal.path.open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="bytes"):
        list(bc.iter_continuous_chunks(manifest))


def original_writer():
    source = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    if not source.is_file():
        pytest.skip("pinned original Raven source is not materialized")
    assert bc.file_sha256(source) == bc.NATIVE_COLLECTOR_SHA256
    method = next(node for node in ast.walk(ast.parse(source.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "_save_collected_chunks")
    namespace = dict(np=np, torch=torch, os=os, VIDEO_MODEL_CHUNK=81, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_save_collected_chunks"]


def writer_fixture(output, lanes):
    return SimpleNamespace(_ep_buf=None, _ep_seq_base=None, _vg_saved_chunks=0,
        _vg_run_tag="original-cpu", vg_collect_dir=str(output), vg_prompt="cpu-differential",
        vg_save_raw_depth_native=True, vg_save_raw_depth=True, vg_control_res=None,
        algo_obs_dim_dict={"actor_obs": 113, "vision_obs": 18},
        _depth_to_control_video=lambda value, view=None: value.squeeze(-1).numpy().astype(np.uint8))


def writer_window(writer, rows, dones):
    lanes = len(dones[0])
    writer._depth_chunk = [torch.full((lanes, 2, 3, 1), float(row % 10)) for row in rows]
    writer._robot_rgb_chunk = []
    writer._views_chunk = [{view: {"depth": depth, "robot_rgb": torch.full((lanes, 2, 3, 3), row % 255, dtype=torch.uint8)}
        for view in bc.VIEWS} for row, depth in zip(rows, writer._depth_chunk)]
    writer.storage = SimpleNamespace(
        actor_obs=torch.tensor(rows, dtype=torch.float32)[:, None, None].expand(-1, lanes, 113),
        gt_actions=torch.tensor(rows, dtype=torch.float32)[:, None, None].expand(-1, lanes, 26),
        dones=torch.tensor(dones, dtype=torch.bool)[..., None])


def test_original_writer_differential_multiwindow_natural_done_and_tail81(tmp_path):
    save = original_writer()
    ledger = ledger_for(tmp_path, windows=5, lanes=3)
    output = Path(ledger.request["output_dir"])
    output.mkdir()
    writer = writer_fixture(output, 3)
    for window in range(5):
        rows = list(range(window * 81, (window + 1) * 81))
        dones = [[index in (39, 179), index in (80, 242), index in (1, 100, 350)] for index in rows]
        for row_dones in dones: step(ledger, row_dones)
        writer_window(writer, rows, dones)
        first = writer._vg_saved_chunks
        save(writer)
        assert writer._vg_saved_chunks - first == len(ledger.pending())
        for offset, expected in enumerate(ledger.pending(), first):
            with np.load(output / f"chunk_original-cpu_{offset:07d}.npz") as actual:
                assert int(actual["env_id"]) == expected["local_env_id"]
                assert int(actual["seq"]) == expected["seq"]
                expected_rows = np.asarray([r["vector_step_index"] for r in expected["rows"]])
                np.testing.assert_array_equal(actual["gt_actions"][:, 0], expected_rows)
                np.testing.assert_array_equal(actual["obs_actor_obs"][:, 0], expected_rows)
                assert not actual["dones"].any()
                assert expected["selected_for_split"] is True
            ledger.acknowledge(expected)
        assert writer._ep_seq_base == ledger.sequences
        assert [len(buf.get("gt", [])) for buf in writer._ep_buf] == [len(buf) for buf in ledger.buffers]
    assert ledger.reset_counts == [3, 3, 4]  # Only the supplied native terminal events.
    assert ledger.vector_steps == 405


class AttrDict(dict):
    __getattr__ = dict.__getitem__


@pytest.mark.parametrize("failure", [None, "early_return", "quota_exception", "extra_reset", "step_fault"])
def test_native_adapter_normal_completion_and_faults_with_original_writer(tmp_path, monkeypatch, failure):
    """Execute the actual adapter class against CPU native-boundary doubles.

    The original writer is executed, not mocked; only physics/teacher inference
    are doubles. This exercises normal-return vs quota/error paths and wrapper
    restoration without claiming Isaac qualification.
    """
    from pipeline.deployment.rtx_actor import initial_reset_contract as reset_contract
    original_save = original_writer()
    raw = request(tmp_path, windows=3, lanes=2)
    request_file = tmp_path / "request.json"
    request_file.write_bytes(bc.canonical(raw) + b"\n")
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST", str(request_file))
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST_SHA256", bc.file_sha256(request_file))
    monkeypatch.setitem(sys.modules, "initial_reset_contract", reset_contract)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    torch.manual_seed(raw["process_seed"])

    class NativeEnv:
        def __init__(self):
            self.num_envs, self.device, self.steps = 2, "cpu", 0
            self.episode_length_buf = torch.zeros(2, dtype=torch.int64)
            self.stage_buf = torch.zeros(2, dtype=torch.int64)
            self.time_in_stage_buf = torch.zeros(2, dtype=torch.int64)
            self.actual_time_in_stage_buf = torch.zeros(2, dtype=torch.int64)
            self.need_to_refresh_envs = torch.zeros(2, dtype=torch.bool)
            self.obs_buf_dict = {key: torch.zeros(2, width) for key, width in
                (("actor_obs", 113), ("teacher_obs", 4), ("homie_obs", 3), ("gt_obj_pos", 3))}
            self.config = AttrDict(reset_from_dataset=AttrDict(), task=AttrDict(
                front_object="front", hold_object="hold", tray_object="tray"))
            self.reset_dataset = SimpleNamespace(
                motion_files=[row["path"] for row in raw["motion_inventory"]],
                sample_idx_based_on_demo_clip_number=[(i, i + 1) for i in range(len(raw["motion_inventory"]))])
            self.simulator = SimpleNamespace(
                bc_motion_binding_sha256=hashlib.sha256(bc.canonical(raw["motions"])).hexdigest(),
                robot_root_states=torch.zeros(2, 13), dof_pos=torch.zeros(2, 26), dof_vel=torch.zeros(2, 26),
                get_task_root_state=lambda name: torch.zeros(2, 13))

        def _reset_task_states(self, ids):
            self.sample_idx = ids.clone()
            self.sample_idx_in_seq = torch.zeros(len(ids), dtype=torch.int64)

        def reset_envs_idx(self, ids):
            self._reset_task_states(ids)
            self.episode_length_buf[ids] = 0

        def reset_all(self):
            reset_contract.begin_initial_reset(self)
            self.reset_envs_idx(torch.arange(2))
            self.need_to_refresh_envs.fill_(True)
            reset_contract.complete_initial_reset(self)
            return self.obs_buf_dict

        def step(self, actor_state):
            if failure == "step_fault" and self.steps == 97:
                raise RuntimeError("native fixture failure")
            dones = torch.tensor([self.steps == 39, self.steps == 80], dtype=torch.bool)
            self.episode_length_buf += 1
            self.steps += 1
            ids = torch.nonzero(dones).flatten()
            if len(ids):
                self.reset_envs_idx(ids)
            return self.obs_buf_dict, torch.zeros(2), dones, {}

    class NativeTrainer:
        _save_collected_chunks = original_save

        def train(self):
            self.env.reset_all()
            self.env.episode_length_buf.fill_(23)
            self.env.time_in_stage_buf.fill_(11)
            self.env.actual_time_in_stage_buf.copy_(self.env.time_in_stage_buf)
            for window in range(3):
                rows = list(range(window * 81, (window + 1) * 81))
                dones = []
                for _ in rows:
                    action = torch.zeros(2, 26)
                    result = self.env.step({"actions": action, "gt_actions": action.clone()})
                    dones.append(result[2].tolist())
                writer_window(self, rows, dones)
                self._save_collected_chunks()
                if window == 0:
                    if failure == "early_return": return
                    if failure == "quota_exception": raise bc._QuotaReached()
                    if failure == "extra_reset": self.env.reset_all()

    source = Path(bc.__file__)
    trainer_class = next(node for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ClassDef) and node.name == "Isaac6BCProductionTrainer")
    namespace = {**vars(bc), "inspect": inspect, "np": np, "torch": torch,
        "TRLDistillTrainerVideoGen": NativeTrainer,
        "NATIVE_COLLECTOR_SHA256": bc.file_sha256(inspect.getsourcefile(NativeTrainer))}
    exec(compile(ast.Module(body=[trainer_class], type_ignores=[]), str(source), "exec"), namespace)
    adapter = namespace["Isaac6BCProductionTrainer"]()
    adapter.__dict__.update(vars(writer_fixture(Path(raw["output_dir"]), 2)))
    teacher = tmp_path / "fixture-teacher.pt"
    teacher.write_bytes(b"CPU model identity fixture")
    adapter.config = AttrDict(init_at_random_ep_len=True,
        network_load_dict=SimpleNamespace(teacher_actor=SimpleNamespace(path=str(teacher))),
        homie_walk_model_path=str(teacher), homie_stand_model_path=str(teacher))
    adapter.env = NativeEnv()
    adapter.vg_mode, adapter.vg_rollout_action_source, adapter.num_steps_per_env = "collect", "teacher", 81
    adapter.args = SimpleNamespace(num_ppo_epochs=0, num_total_batches=3)
    adapter.accelerator = SimpleNamespace(num_processes=1)
    adapter.optimizer = SimpleNamespace(step=lambda: None)
    original_methods = (adapter.env.step, adapter.env.reset_envs_idx, adapter.env._reset_task_states,
                        adapter.env.reset_all, adapter.optimizer.step)
    if failure:
        with pytest.raises(RuntimeError): adapter.train()
    else:
        adapter.train()
    manifest = json.loads((Path(raw["output_dir"]) / "cohort-manifest.json").read_text())
    assert manifest["status"] == ("failed" if failure else "complete")
    assert manifest["schema"] == bc.CONTINUOUS_MANIFEST_SCHEMA
    assert manifest["completion_reason"] == (None if failure else "vector_window_budget")
    assert "chunks" not in manifest
    assert adapter._bc_records == []
    assert adapter._bc_journal.stream.closed
    assert (adapter.env.step, adapter.env.reset_envs_idx, adapter.env._reset_task_states,
            adapter.env.reset_all, adapter.optimizer.step) == original_methods
    if not failure:
        assert manifest["vector_steps"] == 243 and manifest["accepted_windows"] == 3
        assert manifest["emitted_chunks"] == manifest["selected_chunks"] == 5
        assert manifest["remaining_unsealed_tail_rows_per_lane"] == [41, 0]
        assert manifest["initial_reset_all_calls"] == 1
        assert manifest["overcollection_chunks"] == manifest["optimizer_steps"] == 0
        initial = manifest["initial_state_attestation"]["arrays"]
        assert initial["episode_length"] == bc.array_attestation(np.full(2, 23, dtype=np.int64))
        assert initial["time_in_stage"] == bc.array_attestation(np.full(2, 11, dtype=np.int64))
        assert initial["actual_time_in_stage"] == initial["time_in_stage"]
        assert len(list(bc.iter_continuous_chunks(manifest))) == 5
