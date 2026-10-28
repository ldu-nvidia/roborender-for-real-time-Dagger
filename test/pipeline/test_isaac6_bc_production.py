"""CPU contract and reset/chunk provenance tests; no Isaac/GPU imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    CHUNK_ROWS, SCHEMA, SCHEMA_V2, ChunkProvenanceLedger, canonical, load_request,
    publish_json, request_from_cohort, source_run_tag, validate_collection_request,
    array_attestation,
)


def request(tmp_path, *, quotas=1, count=2, max_windows=10):
    return {"schema": SCHEMA, "run_id": "bc-native-test", "cohort_id": "train-0000",
        "split": "train", "process_seed": 7341, "rank": 1, "global_env_offset": 40,
        "motions": [{"motion_id": f"motion-{i}", "path": str(tmp_path / f"motion-{i}.npz"),
                     "sha256": str(i) * 64} for i in range(count)],
        "chunks_per_env": quotas, "max_windows": max_windows,
        "output_dir": str(tmp_path / "chunks"), "sampling_plan_sha256": "a" * 64}


def reset(ledger, lane, sample=1):
    ledger.task_reset(lane, sample_idx=sample, sample_idx_in_seq=4, rng_sha256="f" * 64)
    ledger.episode_reset(lane)


def initialized(tmp_path, **kwargs):
    ledger = ChunkProvenanceLedger(request(tmp_path, **kwargs))
    for lane in range(ledger.num_envs):
        reset(ledger, lane)
    return ledger


def transition(ledger, dones=None):
    dones = dones or [False] * ledger.num_envs
    ledger.begin_transition([v["episode_step_index"] + 23 for v in ledger.current])
    for lane, done in enumerate(dones):
        if done:
            reset(ledger, lane, 2)
    ledger.finish_transition(dones)


def test_protocol_module_does_not_import_native_runtime():
    assert "isaaclab" not in sys.modules
    assert "gr00t.rl.trl.trainer.distill_trainer_videogen" not in sys.modules


def test_uniform_quota_normalized_without_mutating_request(tmp_path):
    raw = request(tmp_path, quotas=2)
    parsed = validate_collection_request(raw)
    assert raw["chunks_per_env"] == 2
    assert parsed["chunks_per_env"] == [2, 2]
    assert source_run_tag(parsed) == "bc-native-test-train-0000-r001"


def test_v2_nonprefix_ordinals_one_and_five_preserve_continuous_native_trajectory(tmp_path):
    raw = request(tmp_path, quotas=[2, 2], max_windows=10)
    raw.update(schema=SCHEMA_V2, selected_chunk_ordinals_per_env=[[1, 5], [0, 4]],
               exposure_span_chunks_per_env=[8, 8])
    ledger = ChunkProvenanceLedger(raw)
    for lane in range(2):
        reset(ledger, lane)
    episodes = [row["episode_id"] for row in ledger.current]
    selected = [[], []]
    emitted = [[], []]
    for ordinal in range(6):
        for _ in range(81):
            transition(ledger)
        for row in ledger.pending():
            lane = row["local_env_id"]
            assert row["emitted_chunk_ordinal"] == ordinal
            emitted[lane].append(ordinal)
            if row["selected_for_split"]:
                selected[lane].append(ordinal)
            ledger.acknowledge(row)
        assert ledger.quota_reached() is (ordinal == 5)
    assert selected == [[1, 5], [0, 4]] and emitted == [list(range(6)), list(range(6))]
    assert ledger.reset_counts == [1, 1] and ledger.vector_steps == 486
    assert [row["episode_id"] for row in ledger.current] == episodes


@pytest.mark.parametrize("selected,span,bound", [([1, 1], 8, 10), ([5, 1], 8, 10),
    ([1, 8], 8, 10), ([1, 5], 8, 5), ([1, True], 8, 10), ([1], 8, 10)])
def test_v2_rejects_ambiguous_or_out_of_bound_ordinals(tmp_path, selected, span, bound):
    raw = request(tmp_path, quotas=2, count=1, max_windows=bound)
    raw.update(schema=SCHEMA_V2, selected_chunk_ordinals_per_env=[selected], exposure_span_chunks_per_env=[span])
    with pytest.raises(ValueError, match="ordinals"):
        validate_collection_request(raw)


@pytest.mark.parametrize("field,value", [
    ("process_seed", True), ("process_seed", -1), ("rank", -1), ("max_windows", 0),
    ("split", "maybe"), ("chunks_per_env", [1]), ("chunks_per_env", [0, 1]),
    ("chunks_per_env", [True, 1]), ("chunks_per_env", 11), ("output_dir", "/"),
    ("output_dir", "relative"), ("sampling_plan_sha256", "missing"),
])
def test_request_rejects_ambiguous_or_unbounded_fields(tmp_path, field, value):
    raw = request(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError):
        validate_collection_request(raw)


def test_request_rejects_extra_keys_and_duplicate_motion(tmp_path):
    raw = request(tmp_path)
    raw["restart_from"] = "old-output"
    with pytest.raises(ValueError):
        validate_collection_request(raw)
    raw.pop("restart_from")
    raw["motions"][1] = deepcopy(raw["motions"][0])
    with pytest.raises(ValueError):
        validate_collection_request(raw)


def test_whole_chunks_and_different_quotas_track_overcollection(tmp_path):
    ledger = initialized(tmp_path, quotas=[1, 2])
    for _ in range(CHUNK_ROWS):
        transition(ledger)
    first = ledger.pending()
    assert [e["local_env_id"] for e in first] == [0, 1]
    assert ledger.emitted_counts == [1, 1]
    assert not ledger.quota_reached()
    episodes = [lane["episode_id"] for lane in ledger.current]
    for emitted in first:
        ledger.acknowledge(emitted)
    for _ in range(CHUNK_ROWS):
        transition(ledger)
    assert ledger.quota_reached()
    assert ledger.emitted_counts == [2, 2]
    assert [e["selected_for_split"] for e in ledger.pending()] == [False, True]
    assert [lane["episode_id"] for lane in ledger.current] == episodes
    assert ledger.reset_counts == [1, 1]  # No reset at the 81 boundary.
    assert ledger.current[0]["episode_step_index"] == 162


def test_done_discards_partial_tail_and_preserves_native_sequence_gap(tmp_path):
    ledger = initialized(tmp_path)
    for _ in range(39):
        transition(ledger)
    transition(ledger, [True, False])
    assert ledger.discarded_tail_rows == [40, 0]
    assert ledger.sequences == [2, 0]
    for _ in range(81):
        transition(ledger)
    emissions = ledger.pending()
    assert [(e["local_env_id"], e["seq"]) for e in emissions] == [(0, 2), (1, 0)]
    assert {r["episode_step_index"] for r in emissions[0]["rows"]} == set(range(81))
    assert len({r["episode_id"] for r in emissions[0]["rows"]}) == 1
    assert emissions[0]["rows"][0]["episode_id"].endswith("p00000001")


def test_done_at_exact_row_81_emits_then_bumps_sequence_without_tail(tmp_path):
    ledger = initialized(tmp_path, count=1)
    for _ in range(80):
        transition(ledger)
    transition(ledger, [True])
    emitted = ledger.pending()[0]
    assert emitted["seq"] == 0
    assert emitted["rows"][-1]["actual_done"] is True
    assert ledger.sequences == [3]
    assert ledger.discarded_tail_rows == [0]
    assert ledger.current[0]["episode_step_index"] == 0


def test_periodic_task_reset_is_not_an_episode_reset(tmp_path):
    ledger = initialized(tmp_path, count=1)
    episode = ledger.current[0]["episode_id"]
    transition(ledger)
    ledger.begin_transition([24])
    ledger.task_reset(0, sample_idx=7, sample_idx_in_seq=3, rng_sha256="b" * 64)
    ledger.finish_transition([False])
    transition(ledger)
    assert ledger.current[0]["episode_id"] == episode
    assert ledger.buffers[0][1]["sample_idx"] == 1
    assert ledger.buffers[0][2]["sample_idx"] == 7
    assert ledger.reset_counts == [1]
    assert ledger.task_counts == [2]


def test_false_done_or_unobserved_reset_fails_closed(tmp_path):
    ledger = initialized(tmp_path, count=1)
    ledger.begin_transition([20])
    with pytest.raises(ValueError, match="done"):
        ledger.finish_transition([True])
    ledger = initialized(tmp_path, count=1)
    ledger.begin_transition([20])
    reset(ledger, 0)
    with pytest.raises(ValueError, match="done"):
        ledger.finish_transition([False])


def test_extra_window_reset_missing_initial_reset_and_duplicate_step_rejected(tmp_path):
    ledger = ChunkProvenanceLedger(request(tmp_path, count=1))
    with pytest.raises(ValueError, match="initial"):
        ledger.begin_transition([0])
    with pytest.raises(ValueError, match="attestation"):
        ledger.episode_reset(0)
    reset(ledger, 0)
    with pytest.raises(ValueError, match="between windows"):
        ledger.episode_reset(0)
    ledger.begin_transition([0])
    with pytest.raises(ValueError, match="uncommitted"):
        ledger.begin_transition([0])
    reset(ledger, 0)
    with pytest.raises(ValueError, match="multiple"):
        ledger.episode_reset(0)


def test_provenance_is_bound_to_actual_bytes_rank_lane_seed_episode(tmp_path):
    ledger = initialized(tmp_path, count=1)
    for _ in range(81):
        transition(ledger)
    source = tmp_path / "native.npz"
    source.write_bytes(b"unchanged original native bytes")
    record = ledger.describe(ledger.pending()[0], source)
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (record["rank"], record["local_env_id"], record["global_env_id"]) == (1, 0, 40)
    assert record["simulator_seed"] == 7341
    assert record["motion_id"] == "motion-0"
    assert record["first_episode_step_index"] == 0
    assert record["last_episode_step_index"] == 80
    assert record["rows"][0]["native_episode_length"] == 23
    assert record["rows"][0]["sample_idx_in_seq"] == 4


def test_emission_order_guard_and_no_overwrite(tmp_path):
    ledger = initialized(tmp_path, count=1)
    for _ in range(162):
        transition(ledger)
    with pytest.raises(ValueError, match="ordering"):
        ledger.acknowledge(ledger.pending()[1])
    path = tmp_path / "manifest.json"
    publish_json(path, {"old": 1})
    with pytest.raises(FileExistsError):
        publish_json(path, {"new": 2})
    assert json.loads(path.read_bytes()) == {"old": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_request_checks_exact_frozen_bytes(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_bytes(canonical(request(tmp_path)))
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST", str(path))
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert load_request()["chunks_per_env"] == [1, 1]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="launch pin"):
        load_request()


def test_sampling_plan_cohort_resolves_exact_order_and_quotas(tmp_path):
    cohort = {"cohort_id": "train-001", "split": "train", "process_seed": 5566,
        "num_envs": 2, "slots": [
            {"motion_id": "b", "filename": "b.npz", "motion_sha256": "b" * 64, "target_chunks": 9},
            {"motion_id": "a", "filename": "a.npz", "motion_sha256": "a" * 64, "target_chunks": 8}]}
    plan = {"schema": "roborender.coverage-sampling-plan.v1", "partitions": {"train": {"cohorts": [cohort]}}}
    actual = request_from_cohort(plan, cohort, run_id="fresh", motion_root=tmp_path,
        output_dir=tmp_path / "out", rank=0, global_env_offset=0, max_windows=30)
    assert actual["chunks_per_env"] == [9, 8]
    assert [r["motion_id"] for r in actual["motions"]] == ["b", "a"]
    assert actual["sampling_plan_sha256"] == hashlib.sha256(canonical(plan)).hexdigest()
    changed = deepcopy(cohort)
    changed["process_seed"] += 1
    with pytest.raises(ValueError, match="exact member"):
        request_from_cohort(plan, changed, run_id="fresh", motion_root=tmp_path,
            output_dir=tmp_path / "out", rank=0, global_env_offset=0, max_windows=30)


def test_initial_state_attestation_is_c_order_finite_and_nonmutating():
    import numpy as np
    value = np.arange(12, dtype=np.float32).reshape(3, 4).T
    before = value.copy()
    result = array_attestation(value)
    assert result == {"dtype": "float32", "shape": [4, 3],
                      "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    np.testing.assert_array_equal(value, before)
    with pytest.raises(ValueError, match="finite"):
        array_attestation(np.asarray([float("nan")]))
