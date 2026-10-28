from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.legacy.bc_collection.sampling_plan import build_plan
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    ChunkProvenanceLedger, NATIVE_COLLECTOR_SHA256, canonical, request_from_cohort, selected_chunk_ordinals,
)
from pipeline.deployment.rtx_actor.run_bc_cohort import build_command


def envelope(doc):
    return {"document": doc, "sha256": hashlib.sha256(canonical(doc)).hexdigest()}


def inputs(tmp_path, selection_policy="prefix_v1"):
    motions = [{"motion_id": f"motion-{i}", "filename": f"motion-{i}.npz", "bytes": 3,
                "sha256": str(i + 1) * 64} for i in range(2)]
    inventory = {"schema": "roborender.motion-inventory.v1", "motion_count": 2,
                 "motions": motions, "motion_records_sha256": hashlib.sha256(canonical(motions)).hexdigest()}
    plan = build_plan(inventory, seed=17, num_envs=2, chunks=20, success_cases=2, selection_policy=selection_policy)
    dataset = tmp_path / "dataset"
    (dataset / "isaac").mkdir(parents=True)
    cohorts, transfers = [], []
    for role, partition in plan["partitions"].items():
        for cohort in partition["cohorts"]:
            native_dir = tmp_path / cohort["cohort_id"]
            native_dir.mkdir()
            request = request_from_cohort(plan, cohort, run_id="native", motion_root=tmp_path,
                output_dir=native_dir, rank=0, global_env_offset=0, max_windows=12)
            ledger = ChunkProvenanceLedger(request)
            for lane in range(2):
                ledger.task_reset(lane, sample_idx=1, sample_idx_in_seq=0, rng_sha256="f" * 64)
                ledger.episode_reset(lane)
            rows, files = [], []
            # One extra whole window proves extras are retained only in audit.
            windows = max(max(values) + 1 for values in selected_chunk_ordinals(request)) + 1
            for _ in range(windows):
                for _ in range(81):
                    ledger.begin_transition([0, 0])
                    ledger.finish_transition([False, False])
                for emitted in ledger.pending():
                    path = native_dir / f"chunk_{cohort['cohort_id']}_{len(rows):07d}.npz"
                    raw = f"native unchanged {cohort['cohort_id']} {len(rows)}".encode()
                    path.write_bytes(raw)
                    row = ledger.describe(emitted, path)
                    rows.append({k: v for k, v in row.items() if k != "rows"})
                    ledger.acknowledge(emitted)
                    if row["selected_for_split"]:
                        destination = dataset / "isaac" / path.name
                        destination.write_bytes(raw)
                        files.append({"source_path": str(path), "source_sha256": row["source_sha256"],
                            "source_bytes": len(raw), "destination_path": str(destination),
                            "destination_sha256": row["source_sha256"], "destination_bytes": len(raw)})
            doc = {"schema": "roborender.bc-native-cohort.v1", "status": "complete", "request": request,
                "request_sha256": ledger.request_sha256, "vector_steps": ledger.vector_steps,
                "optimizer_steps": 0, "teacher_only": True, "native_payload_writer_unchanged": True,
                "selected_chunks": sum(request["chunks_per_env"]),
                "overcollection_chunks": len(rows) - sum(request["chunks_per_env"]),
                "emitted_per_lane": ledger.emitted_counts, "chunks": rows}
            doc["native_collector_sha256"] = NATIVE_COLLECTOR_SHA256
            doc["model_assets"] = {name: {"path": "/models/" + name + ".pt", "sha256": str(i) * 64}
                for i, name in enumerate(("teacher", "homie_walk", "homie_stand"), start=3)}
            cohorts.append(envelope(doc))
            transfers.append(envelope({"schema": "roborender.bc-cohort-transfer.v1", "verified": True,
                "cohort_id": cohort["cohort_id"], "cohort_manifest_sha256": cohorts[-1]["sha256"], "files": files}))
    return plan, cohorts, transfers, dataset


def test_metadata_only_ingest_preserves_preassigned_roles_and_excludes_extras(tmp_path, monkeypatch):
    plan, cohorts, transfers, dataset = inputs(tmp_path)
    # Ingestion must not deserialize/re-read any source or destination payload.
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("payload read")))
    manifest, seal = build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=20)
    assert manifest["chunk_count"] == 20
    assert manifest["partition_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert len(manifest["overcollection_excluded"]) == 6
    assert manifest["production_admitted"] is False
    assert seal["rendered_chunks"] == 0
    assert manifest["model_assets"] == cohorts[0]["document"]["model_assets"]
    assert manifest["native_collector_sha256"] == NATIVE_COLLECTOR_SHA256
    assert set(manifest["motion_bindings"]) == {"motion-0", "motion-1"}
    assert all(row["selected_for_split"] and not row["overcollection"] for row in manifest["chunks"])
    assert all(row["source_size_bytes"] == row["source_bytes"] and row["sequence_id"] == row["seq"]
               for row in manifest["chunks"])


@pytest.mark.parametrize("mutation", [
    lambda p, c, t: c.pop(),
    lambda p, c, t: t.pop(),
    lambda p, c, t: c[0]["document"].update(status="failed"),
    lambda p, c, t: c[0]["document"].update(optimizer_steps=1),
    lambda p, c, t: c[0]["document"].pop("model_assets"),
    lambda p, c, t: c[0]["document"].update(native_collector_sha256="0" * 64),
    lambda p, c, t: c[1]["document"]["model_assets"]["teacher"].update(sha256="a" * 64),
    lambda p, c, t: c[1]["document"]["model_assets"]["homie_walk"].update(sha256="a" * 64),
    lambda p, c, t: c[0]["document"]["chunks"][0].update(split="test"),
    lambda p, c, t: c[0]["document"]["chunks"][0].update(motion_id="wrong"),
    lambda p, c, t: c[0]["document"]["chunks"][0].update(overcollection=True),
    lambda p, c, t: c[0]["document"]["chunks"][0].update(episode_id=""),
    lambda p, c, t: c[0]["document"]["chunks"].pop(),
    lambda p, c, t: t[0]["document"].update(verified=False),
    lambda p, c, t: t[0]["document"].update(cohort_manifest_sha256="0" * 64),
    lambda p, c, t: t[0]["document"]["files"][0].update(destination_sha256="0" * 64),
])
def test_ingestion_rejects_incomplete_changed_or_unverified_cohorts(tmp_path, mutation):
    plan, cohorts, transfers, dataset = inputs(tmp_path)
    mutation(plan, cohorts, transfers)
    with pytest.raises(ValueError):
        build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=20)


def test_production_default_requires_the_complete_3200_plan(tmp_path):
    plan, cohorts, transfers, dataset = inputs(tmp_path)
    with pytest.raises(ValueError, match="sampling-plan size"):
        build_pool(plan, cohorts, transfers, dataset_root=dataset)


def test_cli_requires_explicit_quota_before_reading_any_inputs(monkeypatch, capsys):
    from pipeline.legacy.bc_collection.ingest_native_cohorts import main
    monkeypatch.setattr("sys.argv", ["ingest_native_cohorts", "--plan", "/absent/plan.json",
        "--cohort-manifests", "/absent/cohort.json", "--transfer-receipts", "/absent/transfer.json",
        "--dataset-root", "/absent/dataset", "--manifest", "/absent/manifest.json",
        "--seal", "/absent/seal.json"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--expected-total" in capsys.readouterr().err


def test_cli_explicit_twenty_chunk_diagnostic_ingestion(tmp_path, monkeypatch, capsys):
    from pipeline.legacy.bc_collection.ingest_native_cohorts import main
    plan, cohorts, transfers, dataset = inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan))
    cohort_paths, transfer_paths = [], []
    for index, (cohort, transfer) in enumerate(zip(cohorts, transfers)):
        cp, tp = tmp_path / f"cohort-{index}.json", tmp_path / f"transfer-{index}.json"
        cp.write_bytes(canonical(cohort["document"]))
        tp.write_bytes(canonical(transfer["document"]))
        cohort_paths.append(str(cp))
        transfer_paths.append(str(tp))
    manifest, seal = dataset / "manifest.json", dataset / "SEALED.json"
    monkeypatch.setattr("sys.argv", ["ingest_native_cohorts", "--plan", str(plan_path),
        "--cohort-manifests", *cohort_paths, "--transfer-receipts", *transfer_paths,
        "--dataset-root", str(dataset), "--expected-total", "20", "--manifest", str(manifest), "--seal", str(seal)])
    main()
    result = json.loads(manifest.read_bytes())
    assert result["partition_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert result["production_admitted"] is False
    assert json.loads(seal.read_bytes())["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert json.loads(capsys.readouterr().out)["chunks"] == 20


@pytest.mark.parametrize("expected_total", [0, -20, True, 21])
def test_ingestion_rejects_invalid_expected_total(tmp_path, expected_total):
    plan, cohorts, transfers, dataset = inputs(tmp_path)
    with pytest.raises(ValueError, match="sampling-plan size"):
        build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=expected_total)


def test_bounded_launcher_uses_native_teacher_adapter_with_pinned_motion_order(tmp_path):
    plan, cohorts, _, _ = inputs(tmp_path)
    request = deepcopy(cohorts[0]["document"]["request"])
    request.update(output_dir="/dagger-runtime/bc_isaac", chunks_per_env=[1, 1], max_windows=4)
    image = "sha256:" + "9" * 64
    calls = []

    def base(*args, **kwargs):
        calls.append((args, kwargs))
        return ["docker", "run", "--env", "PYTHONPATH=/isaac6-port", image,
                "python", "native.py", "+exp=old", "trainer._target_=old",
                "algo.trl.num_total_batches=3", "algo.config.videogen.collect_dir=old"]

    command = build_command(request=request, request_path=tmp_path / "request.json",
        request_sha256="f" * 64, name="bc-cohort-leo-proof", runtime=tmp_path / "runtime",
        overlay=tmp_path / "overlay", source=tmp_path / "source", physical_gpu=0,
        base_builder=base, image_id=image)
    assert calls[0][1]["mode"] == "bc-collect"
    assert calls[0][1]["seed"] == request["process_seed"]
    assert "trainer._target_=isaac6_bc_production.Isaac6BCProductionTrainer" in command
    assert "simulator._target_=isaac6_bc_production.IsaacSim" in command
    assert "algo.trl.num_total_batches=4" in command
    assert "algo.config.num_learning_epochs=0" in command
    assert "algo.config.videogen.rollout_action_source=teacher" in command
    assert "env.config.reset_from_dataset.use_motion_file_dir=false" in command
    assert "ROBORENDER_BC_COLLECTION_REQUEST_SHA256=" + "f" * 64 in command
    assert "PYTHONPATH=/bc-production-source:/isaac6-port" in command
    assert command.index("--mount") < command.index(image)
    assert sum(arg.startswith("algo.trl.num_total_batches=") for arg in command) == 1
    invalid = deepcopy(request)
    invalid["output_dir"] = "/other/namespace"
    with pytest.raises(ValueError, match="binding"):
        build_command(request=invalid, request_path=tmp_path / "request.json", request_sha256="f" * 64,
            name="bc-cohort-leo-proof", runtime=tmp_path, overlay=tmp_path, source=tmp_path,
            physical_gpu=0, base_builder=base, image_id=image)


def test_v2_ingest_admits_only_seed_preselected_nonprefix_ordinals(tmp_path):
    plan, cohorts, transfers, dataset = inputs(tmp_path, "matched_exposure_v2")
    manifest, _ = build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=20)
    assert manifest["partition_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert manifest["selection_policy"] == plan["selection_policy"]
    heldout = [r for r in manifest["chunks"] if r["split"] != "train"]
    assert any(r["emitted_chunk_ordinal"] > 0 for r in heldout)
    assert all(r["selection_policy"] == "explicit_preselected_ordinals_v2" for r in heldout)
    assert len(manifest["overcollection_excluded"]) > 6
    assert not any(r["selected_for_split"] for r in manifest["overcollection_excluded"])


@pytest.mark.parametrize("change", ["native_ordinal", "request_ordinals", "cross_split_seed", "selection_flag"])
def test_v2_ingestion_rejects_ordinal_tampering_and_split_seed_collision(tmp_path, change):
    plan, cohorts, transfers, dataset = inputs(tmp_path, "matched_exposure_v2")
    doc = cohorts[1]["document"]
    if change == "native_ordinal":
        doc["chunks"][0]["emitted_chunk_ordinal"] += 1
    elif change == "request_ordinals":
        doc["request"]["selected_chunk_ordinals_per_env"][0] = [7 if doc["request"]["selected_chunk_ordinals_per_env"][0] != [7] else 6]
    elif change == "cross_split_seed":
        doc["request"]["process_seed"] = cohorts[0]["document"]["request"]["process_seed"]
        plan["partitions"]["validation"]["cohorts"][0]["process_seed"] = doc["request"]["process_seed"]
    else:
        row = next(r for r in doc["chunks"] if not r["selected_for_split"])
        row.update(selected_for_split=True, overcollection=False)
    with pytest.raises(ValueError):
        build_pool(plan, cohorts, transfers, dataset_root=dataset, expected_total=20)
