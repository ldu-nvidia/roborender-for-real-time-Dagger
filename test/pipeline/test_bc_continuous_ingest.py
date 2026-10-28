import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from pipeline.bc_collection import continuous_ingest as ingest
from pipeline.bc_collection.continuous_plan import build_plan, build_request, digest
from pipeline.deployment.rtx_actor import isaac6_bc_production as native
from pipeline.deployment.rtx_actor.bc_collection_rng import expected_receipt
from pipeline.bc_collection.collector import implementation_receipt

MODELS = {name: {"path": f"/models/{name}.pt", "sha256": "d" * 64}
          for name in ("teacher", "homie_walk", "homie_stand")}


def write(path, value):
    path.write_bytes(native.canonical(value) + b"\n")


def array(shape, dtype):
    return np.broadcast_to(np.asarray(0, dtype=dtype), shape)


def payload(path):
    values = {"gt_actions": array((81, 26), "float32"), "dones": array((81,), "bool"),
              "env_id": np.array(0), "seq": np.array(2), "prompt": np.array("Pinned teacher collection")}
    for key, width in (("actor_obs", 113), ("teacher_obs", 125), ("homie_obs", 10), ("gt_obj_pos", 3)):
        values["obs_" + key] = array((81, width), "float32")
    for view in ingest.VIEWS:
        values[f"depth_rawq__{view}"] = array((81, 480, 832), "uint16")
        values[f"depth_raw__{view}"] = array((81, 240, 416), "float16")
        values[f"depth_ctrl__{view}"] = array((81, 240, 416), "uint8")
        values[f"rgb_robot__{view}"] = array((81, 240, 416, 3), "uint8")
    np.savez_compressed(path, **values)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("continuous-native")
    plan = build_plan([{"motion_id": "m0", "path": "/motions/m0.npz", "sha256": "1" * 64},
                       {"motion_id": "m1", "path": "/motions/m1.npz", "sha256": "2" * 64}],
                      run_id="unit", max_windows=2, output_dir=str(root), num_envs=1)
    request = build_request(plan)
    # Deliberately use a noncanonical raw request hash, like a pretty launch file.
    raw_sha = hashlib.sha256(json.dumps(request, indent=2).encode()).hexdigest()
    ledger = native.ChunkProvenanceLedger(request, request_sha256=raw_sha)
    journal = native.ContinuousCollectionJournal(ledger)
    ledger.task_reset(0, sample_idx=0, sample_idx_in_seq=0, rng_sha256="b" * 64)
    ledger.episode_reset(0)
    initial_bindings = None
    saved = 0
    for window in range(2):
        for step in range(81):
            ledger.begin_transition([ledger.current[0]["episode_step_index"]])
            if initial_bindings is None:
                initial_bindings = copy.deepcopy(ledger.inflight)
            done = window == 0 and step == 39
            if done:
                ledger.task_reset(0, sample_idx=1, sample_idx_in_seq=2, rng_sha256="c" * 64)
                ledger.episode_reset(0)
            ledger.finish_transition([done])
        for expected in ledger.pending():
            path = root / f"chunk_{native.source_run_tag(request)}_{saved:07d}.npz"
            payload(path)
            record = ledger.describe(expected, path)
            sidecar = path.with_suffix(".provenance.json")
            native.publish_json(sidecar, record)
            journal.append(record, sidecar)
            ledger.acknowledge(expected)
            saved += 1
        journal.accept_window(saved)
    journal.require_completed(native_saved_chunks=saved, native_tail_rows=[41], initial_reset_all_calls=1)
    journal.close()
    reset = {"schema": "roborender.initial-reset-completion.v1", "generation": 1,
        "num_envs": 1, "boundary": "after_reset_all_writes_and_observations", "cleared_lanes": 1,
        "clear_calls": 1, "native_steps_added": 0, "ordinary_done_resets_modified": False}
    names = ("robot_root", "dof_pos", "dof_vel", "stage", "episode_length", "front_object", "hold_object",
             "tray_object", "actor_obs", "teacher_obs", "homie_obs", "gt_obj_pos")
    document = {"schema": ingest.NATIVE_SCHEMA, "status": "complete", "request": request,
        "request_sha256": raw_sha, "canonical_request_sha256": digest(request),
        "native_collector_sha256": native.NATIVE_COLLECTOR_SHA256, "adapter_sha256": "e" * 64,
        "vector_steps": 162, "max_windows": 2, "accepted_windows": 2,
        "completion_reason": "vector_window_budget", "selection_policy": "all_complete_emissions_v1",
        "init_at_random_ep_len": True, "initial_reset_all_calls": 1,
        "optimizer_steps": 0, "teacher_only": True, "native_payload_writer_unchanged": True,
        "collection_rng": expected_receipt(1, 2),
        "collection_implementation": implementation_receipt(),
        "production_admitted": False, "emitted_per_lane": [1], "emitted_chunks": 1,
        "selected_chunks": 1, "overcollection_chunks": 0,
        "discarded_reset_tail_rows_per_lane": [40], "remaining_unsealed_tail_rows_per_lane": [41],
        "model_assets": MODELS,
        "initial_state_attestation": {"initial_reset_completion": reset,
            "when": "first_pre_step_after_native_reset_and_random_episode_length_initialization",
            "motion_binding_sha256": digest(request["motions"]), "episode_bindings": initial_bindings,
            "arrays": {name: {"sha256": "a" * 64, "shape": [1, 3], "dtype": "float32"} for name in names}},
        **journal.bindings()}
    for name in ("time_in_stage", "actual_time_in_stage"):
        document["initial_state_attestation"]["arrays"][name] = {
            "sha256": "a" * 64, "shape": [1], "dtype": "int64"}
    write(root / "cohort-manifest.json", document)
    execution = {"schema": ingest.EXECUTION_SCHEMA, "verdict": "PASS", "native_runtime_tested": True,
        "request_sha256": raw_sha, "request_canonical_sha256": digest(request),
        "native_manifest_sha256": native.file_sha256(root / "cohort-manifest.json"),
        "host_result_sha256": "a" * 64, "launch_sha256": "b" * 64,
        "collector_execution": {"source_sha256": "f" * 64, "source_manifest_sha256": "a" * 64,
            "port_manifest_sha256": "b" * 64, "guard_sha256": "c" * 64, "image_id": "sha256:" + "d" * 64,
            "assets_manifest_sha256": "a" * 64, "model_sha256": {k: v["sha256"] for k, v in MODELS.items()}},
        "actual_num_envs": 1, "actual_vector_steps": 162, "initial_reset_all_calls": 1,
        "qualification_only": False, "production_admitted": False, "container_id": "7" * 64}
    write(root / "launch.json", {"request": request, "request_sha256": raw_sha,
        "collector_execution": execution["collector_execution"], "qualification_only": False})
    write(root / "host-result.json", {"passed": True, "failure": None, "exit_code": 0,
        "started_monotonic": 1.0, "finished_monotonic": 2.0, "replay_allowed": False,
        "production_admitted": False, "container_id": "7" * 64})
    execution["launch_sha256"] = native.file_sha256(root / "launch.json")
    execution["host_result_sha256"] = native.file_sha256(root / "host-result.json")
    write(root / "continuous-execution.json", execution)
    return root, plan


@pytest.fixture
def run(reference, tmp_path):
    source, plan = reference
    root = tmp_path / "mapped-native"
    shutil.copytree(source, root)
    return root, copy.deepcopy(plan)


def inspect(run, **kwargs):
    root, plan = run
    return ingest.inspect_run(plan, native_output_dir=root, execution_path=root / "continuous-execution.json",
        expected_source_sha256="f" * 64, expected_adapter_sha256="e" * 64,
        expected_model_assets=kwargs.pop("expected_model_assets", MODELS), **kwargs)


def revise_native(run, update):
    root, _ = run
    document = json.loads((root / "cohort-manifest.json").read_bytes())
    update(document)
    write(root / "cohort-manifest.json", document)
    execution = json.loads((root / "continuous-execution.json").read_bytes())
    execution["native_manifest_sha256"] = native.file_sha256(root / "cohort-manifest.json")
    write(root / "continuous-execution.json", execution)


def rewrite_bound_provenance(run, mutate):
    """Rebind every digest so a negative test exercises semantics, not stale SHA."""
    root, _ = run
    record = json.loads((root / "chunks.jsonl").read_bytes())
    sidecar = root / Path(record["provenance_path"]).name
    provenance = json.loads(sidecar.read_bytes())
    mutate(provenance)
    write(sidecar, provenance)
    record = {**{k: v for k, v in provenance.items() if k != "rows"},
              **{k: record[k] for k in ingest.WRAPPER_KEYS}}
    record["provenance_sha256"] = native.file_sha256(sidecar)
    record["provenance_bytes"] = sidecar.stat().st_size
    write(root / "chunks.jsonl", record)
    progress_path = root / "progress/window-000002.json"
    progress = json.loads(progress_path.read_bytes())
    progress["chunk_index"].update(sha256=native.file_sha256(root / "chunks.jsonl"),
                                  bytes=(root / "chunks.jsonl").stat().st_size)
    write(progress_path, progress)
    def update(doc):
        doc["chunk_index"] = progress["chunk_index"]
        doc["progress_receipts"]["final_receipt"].update(
            sha256=native.file_sha256(progress_path), bytes=progress_path.stat().st_size)
    revise_native(run, update)


def test_actual_native_journal_full_inventory_and_cross_window_clip(run):
    result = inspect(run)
    assert result["chunk_count"] == 1
    assert result["partition_counts"] == {"train": 1}
    assert len(result["motions"]) == 1 and len(result["motion_inventory"]) == 2
    assert result["discarded_reset_tail_rows_per_lane"] == [40]
    assert result["remaining_unsealed_tail_rows_per_lane"] == [41]
    assert result["training_admitted"] is False


@pytest.mark.parametrize("key,value", [
    ("status", "failed"), ("accepted_windows", 1), ("vector_steps", 81), ("max_windows", 1),
    ("completion_reason", "quota_reached"), ("optimizer_steps", 1), ("teacher_only", False),
    ("native_payload_writer_unchanged", False), ("init_at_random_ep_len", False),
    ("adapter_sha256", "a" * 64), ("native_collector_sha256", "a" * 64),
    ("canonical_request_sha256", "a" * 64), ("overcollection_chunks", 1), ("selected_chunks", 0),
    ("initial_reset_all_calls", 2), ("selection_policy", "prefix_quota_v1"),
    ("initial_reset_all_calls", True), ("optimizer_steps", False),
    ("discarded_reset_tail_rows_per_lane", [39]), ("remaining_unsealed_tail_rows_per_lane", [42]),
])
def test_native_fail_closed(run, key, value):
    revise_native(run, lambda doc: doc.update({key: value}))
    with pytest.raises(ValueError):
        inspect(run)


@pytest.mark.parametrize("key,value", [
    ("qualification_only", True), ("verdict", "FAIL"), ("actual_num_envs", 2),
    ("actual_vector_steps", 81), ("initial_reset_all_calls", 2), ("native_manifest_sha256", "0" * 64),
    ("request_sha256", "0" * 64), ("request_canonical_sha256", "0" * 64), ("native_runtime_tested", False),
    ("actual_num_envs", True),
])
def test_execution_must_attest_exact_completed_production_run(run, key, value):
    root, _ = run
    value_doc = json.loads((root / "continuous-execution.json").read_bytes())
    value_doc[key] = value
    write(root / "continuous-execution.json", value_doc)
    with pytest.raises(ValueError):
        inspect(run)


def test_reset_must_be_completed_before_first_action(run):
    revise_native(run, lambda doc: doc["initial_state_attestation"]["initial_reset_completion"].update(native_steps_added=1))
    with pytest.raises(ValueError, match="corrected initial reset"):
        inspect(run)


@pytest.mark.parametrize("clock", ["time_in_stage", "actual_time_in_stage"])
def test_initial_stage_clock_evidence_is_required(run, clock):
    revise_native(run, lambda doc: doc["initial_state_attestation"]["arrays"].pop(clock))
    with pytest.raises(ValueError, match="lacks required arrays"):
        inspect(run)


@pytest.mark.parametrize("field,value", [
    ("sha256", "b" * 64), ("shape", [1, 1]), ("dtype", "float32"),
])
def test_initial_actual_stage_clock_must_match_randomized_clock(run, field, value):
    revise_native(run, lambda doc: doc["initial_state_attestation"]["arrays"][
        "actual_time_in_stage"].update({field: value}))
    with pytest.raises(ValueError, match="actual stage clock"):
        inspect(run)


def test_wrong_expected_teacher_rejected(run):
    with pytest.raises(ValueError, match="teacher/Homie"):
        inspect(run, expected_model_assets={})


@pytest.mark.parametrize("filename,field,value", [
    ("launch.json", "qualification_only", True), ("launch.json", "request_sha256", "0" * 64),
    ("host-result.json", "passed", False), ("host-result.json", "exit_code", 1),
    ("host-result.json", "failure", "resource_guard"), ("host-result.json", "replay_allowed", True),
    ("host-result.json", "finished_monotonic", 0.5),
    ("host-result.json", "container_id", "8" * 64),
])
def test_hash_bound_launch_host_semantics(run, filename, field, value):
    root, _ = run
    doc = json.loads((root / filename).read_bytes())
    doc[field] = value
    write(root / filename, doc)
    execution = json.loads((root / "continuous-execution.json").read_bytes())
    key = "launch_sha256" if filename == "launch.json" else "host_result_sha256"
    execution[key] = native.file_sha256(root / filename)
    write(root / "continuous-execution.json", execution)
    with pytest.raises(ValueError, match="launch/host-result"):
        inspect(run)


def test_launch_raw_hash_chain_cannot_be_relabelled(run):
    root, _ = run
    with (root / "launch.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="launch/host-result"):
        inspect(run)


@pytest.mark.parametrize("kind", ["missing", "extra", "partial", "sidecar", "progress", "payload", "index"])
def test_missing_extra_corrupt_or_partial_artifacts_fail(run, kind):
    root, _ = run
    source = next(root.glob("*.npz"))
    if kind == "missing":
        source.unlink()
    elif kind == "extra":
        shutil.copyfile(source, root / "unindexed.npz")
    elif kind == "partial":
        (root / "unfinished.npz.part-123").write_bytes(b"partial")
    elif kind == "sidecar":
        source.with_suffix(".provenance.json").write_bytes(b"{}")
    elif kind == "progress":
        (root / "progress/window-000001.json").unlink()
    elif kind == "payload":
        with source.open("ab") as stream:
            stream.write(b"corrupt")
    else:
        with (root / "chunks.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect(run)


def test_no_quota_injected_into_continuous_request(run):
    revise_native(run, lambda doc: doc["request"].update(chunks_per_env=[1]))
    with pytest.raises(ValueError):
        inspect(run)


@pytest.mark.parametrize("mutation", [
    lambda p: p["rows"][40].update(actual_done=True),
    lambda p: p["rows"][40].update(episode_id="another-episode"),
    lambda p: p["rows"][40].update(vector_step_index=999),
    lambda p: p["rows"][40].update(episode_step_index=39),
    lambda p: p["rows"][40].update(reset_rng_sha256="unknown"),
    lambda p: p.update(selected_for_split=False),
    lambda p: p.update(overcollection=True),
    lambda p: p.update(selection_policy="prefix_quota_v1"),
    lambda p: p.update(emitted_chunk_ordinal=1),
    lambda p: p.update(frames=80),
    lambda p: p.update(teacher_only=1),
    lambda p: p.update(motion_sha256="9" * 64),
])
def test_rehashed_semantic_provenance_corruption_is_rejected(run, mutation):
    rewrite_bound_provenance(run, mutation)
    with pytest.raises(ValueError):
        inspect(run)


def test_rehashed_progress_prefix_cannot_cherry_pick(run):
    root, _ = run
    path = root / "progress/window-000002.json"
    progress = json.loads(path.read_bytes())
    progress["chunk_index"]["count"] = 0
    write(path, progress)
    revise_native(run, lambda doc: doc["progress_receipts"]["final_receipt"].update(
        sha256=native.file_sha256(path), bytes=path.stat().st_size))
    with pytest.raises(ValueError, match="prefix omits"):
        inspect(run)


def test_payload_schema_requires81_three_view_rows(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, gt_actions=np.zeros((80, 26), dtype=np.float32))
    with pytest.raises(ValueError, match="81-row"):
        ingest.validate_payload(path, {"local_env_id": 0, "seq": 0})


def test_publish_fresh_flat_raw_pool_only_and_refuse_overwrite(run, tmp_path):
    root, plan = run
    target = tmp_path / "fresh-pool"
    kwargs = dict(native_output_dir=root, execution_path=root / "continuous-execution.json",
                  expected_source_sha256="f" * 64, expected_adapter_sha256="e" * 64, expected_model_assets=MODELS)
    result = ingest.publish_pool(plan, dataset_root=target, **kwargs)
    rows = list(ingest.iter_pool_chunks(target))
    assert len(rows) == 1 and rows[0]["split"] == "train"
    assert (target / rows[0]["isaac_npz"]).is_file()
    assert result["rendered_chunks"] == 0 and result["training_admitted"] is False
    assert not (target / "rendered").exists()
    with pytest.raises(FileExistsError):
        ingest.publish_pool(plan, dataset_root=target, **kwargs)


def test_reader_rejects_historical_or_unsealed_pool(tmp_path):
    write(tmp_path / "manifest.json", {"schema": "roborender.bc_pool.native-cohorts.v1"})
    write(tmp_path / "source-seal.json", {})
    with pytest.raises(ValueError, match="unsupported or unsealed"):
        list(ingest.iter_pool_chunks(tmp_path))


def test_failed_publication_remains_unsealed_and_is_not_automatically_adopted(run, tmp_path):
    root, plan = run
    shutil.copyfile(next(root.glob("*.npz")), root / "unindexed.npz")
    target = tmp_path / "failed-pool"
    kwargs = dict(native_output_dir=root, execution_path=root / "continuous-execution.json",
                  expected_source_sha256="f" * 64, expected_adapter_sha256="e" * 64, expected_model_assets=MODELS)
    with pytest.raises(ValueError, match="unindexed"):
        ingest.publish_pool(plan, dataset_root=target, **kwargs)
    assert not (target / "manifest.json").exists()
    assert not (target / "source-seal.json").exists()
    assert len(list((target / "isaac").glob("*.npz"))) == 1
    with pytest.raises(FileExistsError):
        ingest.publish_pool(plan, dataset_root=target, **kwargs)
