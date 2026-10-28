"""Synthetic metadata fixtures only; never native qualification evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.legacy.bc_collection.sampling_plan import build_plan, canonical
from pipeline.deployment.rtx_actor.isaac6_bc_production import request_from_cohort, source_run_tag
from pipeline.bc_training import data_admission as admission
from pipeline.bc_training import production_protocol as protocol


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest()}


def make_complete(tmp_path):
    root = tmp_path / "pool"
    (root / "isaac").mkdir(parents=True)
    motions = [{"motion_id": f"mode-{i:03}", "filename": f"mode-{i:03}.npz",
                "bytes": 3, "sha256": h(str(i))} for i in range(295)]
    inventory = {"schema": "roborender.motion-inventory.v1", "motion_count": 295,
        "ordering": "explicit_filename_lexical_not_filesystem_order", "motions": motions,
        "motion_records_sha256": protocol.sha(canonical(motions))}
    inv_binding = put(tmp_path / "inventory.json", inventory)
    plan = build_plan(inventory, seed=41, num_envs=40, chunks=3200, success_cases=320,
                      selection_policy="matched_exposure_v2")
    plan_binding = put(tmp_path / "plan.json", plan)
    models = {key: h(key) for key in protocol.MODEL_NAMES}
    execution = dict(adapter_source_sha256=h("adapter-code"), source_release_sha256=h("source-release"),
        runtime_contract_sha256=h("runtime-contract"),
        container_image_id="sha256:" + h("container"), native_runtime_version="synthetic-no-native-execution",
        lifecycle_recipe="one_kit_fresh_seeded_scene_per_declared_cohort_v1",
        memory_buffer_recipe="original_raven_capture_lists_and_writer_v1")
    session_receipt = dict(schema="roborender.bc-session-execution.v1", verdict="PASS", session_id="synthetic-cpu-only",
        collector_execution=execution, native_initializations=1, scene_initializations=24,
        request_identities={protocol.sha(canonical(request_from_cohort(plan, c, run_id="synthetic-cpu-only",
            motion_root=tmp_path / "motions", output_dir=tmp_path / "native" / c["cohort_id"], rank=0,
            global_env_offset=0, max_windows=12))): protocol.sha(canonical(request_from_cohort(plan, c,
            run_id="synthetic-cpu-only", motion_root=tmp_path / "motions", output_dir=tmp_path / "native" / c["cohort_id"],
            rank=0, global_env_offset=0, max_windows=12))) for part in plan["partitions"].values() for c in part["cohorts"]},
        **{k: h(k) for k in ("source_manifest_raw_sha256", "port_manifest_raw_sha256", "launch_raw_sha256",
            "session_sha256", "session_canonical_sha256", "native_result_raw_sha256", "host_result_raw_sha256")})
    session_binding = put(tmp_path / "session-execution.json", session_receipt)
    assets = {key: {"path": f"/models/{key}.pt", "sha256": value} for key, value in models.items()}
    cohorts, transfers, cohort_bindings, transfer_bindings, execution_bindings = [], [], [], [], []
    for partition in plan["partitions"].values():
        for cohort in partition["cohorts"]:
            request = request_from_cohort(plan, cohort, run_id="synthetic-cpu-only",
                motion_root=tmp_path / "motions", output_dir=tmp_path / "native" / cohort["cohort_id"],
                rank=0, global_env_offset=0, max_windows=12)
            request_sha = protocol.sha(canonical(request))
            rows, files = [], []
            count = max(max(x) for x in request["selected_chunk_ordinals_per_env"]) + 1
            for lane, slot in enumerate(cohort["slots"]):
                for ordinal in range(count):
                    name = f"chunk_{cohort['cohort_id']}_{lane:02}_{ordinal:02}.npz"
                    payload = name.encode()
                    payload_sha = hashlib.sha256(payload).hexdigest()
                    source_path = str(Path(request["output_dir"]) / name)
                    selected = ordinal in slot["selected_chunk_ordinals"]
                    row = dict(source_run_tag=source_run_tag(request), rank=0, local_env_id=lane,
                        global_env_id=lane, seq=ordinal, source_bytes=len(payload), source_sha256=payload_sha,
                        source_path=source_path, first_episode_step_index=ordinal * 81,
                        last_episode_step_index=ordinal * 81 + 80, motion_id=slot["motion_id"],
                        motion_sha256=slot["motion_sha256"], simulator_seed=request["process_seed"],
                        split=cohort["split"], sampling_plan_sha256=request["sampling_plan_sha256"],
                        request_sha256=request_sha, teacher_only=True, frames=81,
                        episode_id=f"{cohort['cohort_id']}-lane-{lane}-episode-0", emitted_chunk_ordinal=ordinal,
                        selection_policy="explicit_preselected_ordinals_v2", selected_for_split=selected,
                        overcollection=not selected)
                    rows.append(row)
                    if selected:
                        dest = root / "isaac" / name
                        dest.write_bytes(payload)  # tiny non-NPZ fixture; admission never opens payloads
                        files.append(dict(source_path=source_path, source_sha256=payload_sha,
                            destination_sha256=payload_sha, source_bytes=len(payload), destination_bytes=len(payload),
                            destination_path=str(dest)))
            doc = dict(schema="roborender.bc-native-cohort.v1", status="complete", request=request,
                request_sha256=request_sha, vector_steps=count * 81, optimizer_steps=0,
                teacher_only=True, native_payload_writer_unchanged=True,
                native_collector_sha256=protocol.NATIVE_COLLECTOR_SHA256, model_assets=assets,
                selected_chunks=sum(request["chunks_per_env"]),
                overcollection_chunks=len(rows) - sum(request["chunks_per_env"]),
                emitted_per_lane=[count] * len(cohort["slots"]), chunks=rows)
            cb = put(tmp_path / "cohorts" / (cohort["cohort_id"] + ".json"), doc)
            td = dict(schema="roborender.bc-cohort-transfer.v1", verified=True, cohort_id=cohort["cohort_id"],
                      cohort_manifest_sha256=cb["raw_sha256"], files=files)
            tb = put(tmp_path / "transfers" / (cohort["cohort_id"] + ".json"), td)
            cohort_bindings.append(cb); transfer_bindings.append(tb)
            execution_bindings.append(put(tmp_path / "execution" / (cohort["cohort_id"] + ".json"), dict(
                schema="roborender.bc-cohort-execution.v1", verdict="PASS",
                cohort_manifest_raw_sha256=cb["raw_sha256"], request_sha256=request_sha,
                request_canonical_sha256=protocol.sha(canonical(request)),
                session_execution_canonical_sha256=protocol.sha(canonical(session_receipt)),
                collector_execution=execution, actual_num_envs=len(cohort["slots"]), actual_vector_steps=count * 81)))
            cohorts.append({"document": doc, "sha256": cb["raw_sha256"]})
            transfers.append({"document": td, "sha256": tb["raw_sha256"]})
    source, _ = build_pool(plan, cohorts, transfers, dataset_root=root)
    source_binding = put(root / "manifest.json", source)
    rendered = deepcopy(source)
    rendered.update(schema="roborender.bc_pool.rendered.v2", parent_manifest_sha256=source_binding["raw_sha256"],
        render_recipe="raven_3view_50step_independent_v2", render_contract=admission.INDEPENDENT_RECIPE,
        max_chain_chunks=1, validation={"verdict": "PASS"})
    for row in rendered["chunks"]:
        row.update(rendered_sha256=h(row["chunk_id"]), render_contract=admission.INDEPENDENT_RECIPE,
            render_chain_id=row["chunk_id"], render_chain_length=1, render_chain_index=0,
            render_inherits_previous_reference=False)
    rendered_binding = put(root / "manifests/independent_v2.json", rendered)
    splits = {}
    for role in admission.SPLITS:
        items = [{k: r[k] for k in ("chunk_id", "isaac_npz", "rendered_npz")}
                 for r in rendered["chunks"] if r["split"] == role]
        doc = dict(split=role, chunk_count=len(items), items=items,
            source_manifest="../manifests/independent_v2.json", source_manifest_sha256=rendered_binding["raw_sha256"])
        binding = put(root / "splits" / (role + ".json"), doc)
        splits[role] = Path(binding["path"])
    eligibility = dict(schema="roborender.native-motion-eligibility.cpu.v1", cpu_only=True,
        cuda_initialized=False, isaac_imported=False, motion_inventory_sha256=protocol.sha(canonical(inventory)),
        inventory_file_sha256=inv_binding["raw_sha256"], eligible_count=295,
        invalid_or_short=[], frame_count_mismatches=[], native_source_hashes=protocol.NATIVE_RESET_PINS,
        motions=[dict(motion_id=r["motion_id"], sha256=r["sha256"], eligible=True, frame_counts_match=True,
                      native_skips_short_motion=False, native_random_start_choices=6) for r in motions])
    eb = put(tmp_path / "eligibility.json", eligibility)
    capacities = []
    topology = dict(host_id="synthetic-host-not-native-proof", concurrent_collectors=1,
                    cgroup_limit_bytes_per_collector=1000, minimum_host_reserve_bytes=50)
    for role in admission.SPLITS:
      for count in (40, 15):
        capacities.append(put(tmp_path / f"capacity-{role}-{count}.json", dict(
            schema="roborender.native-bc-capacity.v1", purpose="full_bc_capture_and_save", verdict="PASS",
            native_runtime_tested=True, native_collector_sha256=protocol.NATIVE_COLLECTOR_SHA256,
            model_sha256=models, num_envs=count, vector_steps=810, full_chunks_saved=count * 10,
            emitted_chunks_per_lane=[10] * count,
            collector_execution=execution,
            collection_role=role, host_id=topology["host_id"], concurrent_collectors=1,
            simultaneous_collectors_observed=1, cgroup_limit_bytes=1000, maximum_completed_ordinal_tested=9,
            oom_events_delta=0, swap_bytes_peak=0, cgroup_memory_peak_bytes=100,
            runtime_min_mem_available_bytes=100)))
    foundation = dict(id="wan21_fun_control_1p3b_foundation", files=[
        dict(path="test-weights", bytes=3, sha256=h("raw"))])
    adapter = dict(id="roborender_offline_bc_video_teacher_50step_lora_step05294", bytes=3, sha256=h("raw"))
    registry = dict(schema="roborender.renderer_registry.v1", artifacts=[foundation, adapter])
    registry_binding = put(tmp_path / "registry.json", registry)
    source_identity = dict(schema="roborender.renderer-source-identity.v1",
        files=[dict(relative_path="worker.py", sha256=h("raw"), bytes=3)])
    source_identity_binding = put(tmp_path / "renderer-source.json", source_identity)
    render_models = dict(adapter=h("raw"), foundation_manifest=protocol.sha(canonical(foundation)),
                         renderer_source=protocol.sha(canonical(source_identity)), prompt=h("raw"))
    file_records = []
    for index, role in enumerate(("foundation/test-weights", "adapter", "source/worker.py", "prompt")):
        path = tmp_path / f"synthetic-asset-{index}"
        path.write_bytes(b"raw")
        stat = path.stat()
        file_records.append(dict(role=role, path=str(path), sha256=h("raw"), bytes=3,
            file_identity=dict(device=stat.st_dev, inode=stat.st_ino, bytes=3,
                               mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)))
    identity_binding = put(tmp_path / "asset-identity.json", dict(
        schema="roborender.bc-render-assets.v1", hashes_verified=True,
        registry_raw_sha256=registry_binding["raw_sha256"], model_sha256=render_models,
        source_manifest_raw_sha256=source_identity_binding["raw_sha256"], prompt_bytes=3, files=file_records))
    p = dict(schema=protocol.SCHEMA, purpose="production", expected_motion_count=295,
        training_chunk_batch_size=16, native_collector_sha256=protocol.NATIVE_COLLECTOR_SHA256,
        motion_inventory=inv_binding, sampling_plan=plan_binding, model_sha256=models,
        native_motion_eligibility=eb, capacity_receipts=capacities, source_pool=source_binding,
        cohort_manifests=cohort_bindings, transfer_receipts=transfer_bindings,
        render_model_sha256=render_models, execution_topology=topology, collector_execution=execution,
        cohort_execution_receipts=execution_bindings,
        session_execution_receipts=[session_binding],
        renderer_registry=registry_binding, renderer_source_manifest=source_identity_binding,
        render_asset_identity=identity_binding)
    pb = put(tmp_path / "protocol.json", p)
    rr = dict(schema="roborender.bc-independent-render-execution.v1", verdict="PASS",
        rendered_manifest_raw_sha256=rendered_binding["raw_sha256"],
        source_pool_raw_sha256=source_binding["raw_sha256"], recipe="raven_3view_50step_independent_v2",
        num_inference_steps=50, cfg_scale=5.0, seed=42, model_sha256=render_models,
        chunks={r["chunk_id"]: {k: r[k] for k in ("source_sha256", "rendered_sha256")} for r in rendered["chunks"]})
    rb = put(tmp_path / "render-execution.json", rr)
    return SimpleNamespace(root=tmp_path, protocol=p, pb=pb, plan=plan, rendered=rendered,
        rb=rb, manifest=Path(rendered_binding["path"]), splits=splits)


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    return make_complete(tmp_path_factory.mktemp("synthetic-metadata-only"))


def audited(f, **overrides):
    args = dict(require_independent_render=True, purpose="production", run_name="prod-fixture",
        protocol_path=f.pb["path"], protocol_sha256=f.pb["raw_sha256"],
        render_receipt_path=f.rb["path"], render_receipt_sha256=f.rb["raw_sha256"])
    args.update(overrides)
    return admission.audit_split_files(f.manifest, f.splits, **args)


def test_complete295_exact_v2_receipt_chain_and_no_payload_reads(complete, monkeypatch):
    original = Path.read_bytes
    def metadata_only(path):
        assert path.suffix != ".npz", "payload opened by metadata admission"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    report = audited(complete)
    assert report["production_admitted"] and report["protocol"]["population_motion_count"] == 295
    assert report["protocol"]["exact_role_quotas"] == {"train": 2560, "validation": 320, "test": 320}
    assert report["protocol"]["required_actual_num_envs"] == [15, 40]


def bounded_capacity_fixture(complete, size=2):
    p = deepcopy(complete.protocol)
    p["execution_topology"].update(cgroup_limit_bytes_per_collector=32 << 30, minimum_host_reserve_bytes=8 << 30)
    p["capacity_equivalence"] = dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=0.10,
                                    absolute_growth_slack_bytes=256 << 20)
    r = dict(schema="roborender.native-bc-capacity.v2", purpose="full_bc_capture_and_save", verdict="PASS",
        native_runtime_tested=True, collector_execution=deepcopy(p["collector_execution"]),
        native_collector_sha256=protocol.NATIVE_COLLECTOR_SHA256, model_sha256=p["model_sha256"],
        num_envs=size, host_id=p["execution_topology"]["host_id"], concurrent_collectors=1,
        simultaneous_collectors_observed=1, cgroup_limit_bytes=32 << 30,
        runtime_guard_enabled=True, active_guard_minimum_host_available_bytes=8 << 30,
        bounded_repeated_buffer_only=True, oom_events_delta=0, swap_bytes_peak=0,
        plateau_rule=p["capacity_equivalence"], per_flush_memory=[dict(vector_steps=81 * count,
            emitted_chunks_per_lane=[count] * size, memory_current_bytes=20 << 30,
            memory_peak_bytes=22 << 30, anon_bytes=14 << 30, shmem_bytes=0,
            kernel_bytes=1 << 30, file_bytes=5 << 30, host_available_bytes=20 << 30) for count in (1, 2)])
    return p, r


@pytest.mark.parametrize("size", [1, 2, 7, 8])
def test_capacity_v2_bounded_equivalence_preserves_actual_batch(complete, size):
    p, r = bounded_capacity_fixture(complete, size)
    r["collector_execution"]["source_release_sha256"] = h("only-readme-different")
    protocol.validate_bounded_capacity(r, p)
    assert r["num_envs"] == size


@pytest.mark.parametrize("mutation", ["native", "guard", "topology", "tail", "one_flush", "one_emission", "total_growth", "private_growth", "headroom", "oom", "swap", "rule"])
def test_capacity_v2_rejects_incompatible_or_unstable_runtime(complete, mutation):
    p, r = bounded_capacity_fixture(complete)
    if mutation == "native": r["collector_execution"]["adapter_source_sha256"] = h("changed")
    elif mutation == "guard": r["runtime_guard_enabled"] = False
    elif mutation == "topology": r["simultaneous_collectors_observed"] = 2
    elif mutation == "tail": r["num_envs"] = 1
    elif mutation == "one_flush": r["per_flush_memory"] = r["per_flush_memory"][-1:]
    elif mutation == "one_emission": r["per_flush_memory"][-1]["emitted_chunks_per_lane"] = [1, 2]
    elif mutation == "total_growth": r["per_flush_memory"][-1].update(memory_current_bytes=26 << 30, memory_peak_bytes=26 << 30)
    elif mutation == "private_growth": r["per_flush_memory"][-1]["anon_bytes"] = 18 << 30
    elif mutation == "headroom": r["per_flush_memory"][-1]["host_available_bytes"] = 7 << 30
    elif mutation == "oom": r["oom_events_delta"] = 1
    elif mutation == "swap": r["swap_bytes_peak"] = 1
    else: r["plateau_rule"] = {**r["plateau_rule"], "maximum_growth_fraction": 1.0}
    with pytest.raises(ValueError): protocol.validate_bounded_capacity(r, p)


def test_metadata_reader_rejects_oversize_before_read(tmp_path, monkeypatch):
    path = tmp_path / "huge.json"
    with path.open("wb") as stream:
        stream.truncate((32 << 20) + 1)
    with pytest.raises(ValueError, match="bounded"):
        protocol.read_metadata(path)


@pytest.mark.parametrize("purpose", [None, "", "metadata_only", "prod", "test"])
def test_missing_or_unknown_purpose_fails(complete, purpose):
    with pytest.raises(ValueError, match="purpose"):
        audited(complete, purpose=purpose)


def test_qualification_never_production_and_needs_explicit_training_flag(complete, tmp_path):
    report = audited(complete, purpose="qualification", run_name="smoke-proto",
        protocol_path=None, protocol_sha256=None, render_receipt_path=None, render_receipt_sha256=None)
    assert report["qualification_admitted"] and not report["production_admitted"]
    binding = put(tmp_path / "qualification.json", report)
    for kwargs in ({}, {"allow_qualification": True, "run_name": "BC"}):
        with pytest.raises(ValueError, match="explicit flag"):
            admission.validate_training_admission(Path(binding["path"]), complete.splits["train"],
                                                   complete.splits["validation"], **kwargs)
    assert admission.validate_training_admission(Path(binding["path"]), complete.splits["train"],
        complete.splits["validation"], allow_qualification=True, run_name="smoke-retest") == report


@pytest.mark.parametrize("mutation", ["wrong_raw_plan_hash", "raw_canonical_confusion", "capacity_wrong_B",
    "short_prefix", "missing_mode", "wrong_ordinal", "wrong_teacher", "eligibility_missing", "writer_changed"])
def test_frozen_protocol_inputs_fail_closed(complete, tmp_path, mutation):
    doc = deepcopy(complete.protocol)
    if mutation == "wrong_raw_plan_hash": doc["sampling_plan"]["raw_sha256"] = "0" * 64
    elif mutation == "raw_canonical_confusion": doc["sampling_plan"]["raw_sha256"] = protocol.sha(canonical(complete.plan))
    elif mutation == "capacity_wrong_B": doc["capacity_receipts"] = doc["capacity_receipts"][:1]
    elif mutation == "wrong_teacher": doc["model_sha256"]["teacher"] = "0" * 64
    elif mutation == "writer_changed": doc["native_collector_sha256"] = "0" * 64
    elif mutation == "eligibility_missing": doc["native_motion_eligibility"]["raw_sha256"] = "0" * 64
    else:
        changed = deepcopy(complete.plan)
        if mutation == "short_prefix": changed["schema"] = "roborender.coverage-sampling-plan.v1"
        elif mutation == "missing_mode": changed["partitions"]["test"]["cohorts"][0]["slots"].pop()
        else: changed["partitions"]["validation"]["cohorts"][0]["slots"][0]["selected_chunk_ordinals"] = [0]
        doc["sampling_plan"] = put(tmp_path / "changed-plan.json", changed)
    binding = put(tmp_path / "changed-protocol.json", doc)
    with pytest.raises(ValueError):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])


def test_small_observed_inventory_cannot_redefine_expected_population(complete, tmp_path):
    doc = deepcopy(complete.protocol)
    doc["expected_motion_count"] = 2
    binding = put(tmp_path / "tiny-protocol.json", doc)
    with pytest.raises(ValueError, match="full295"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])


def test_legacy_report_readable_but_never_new_production(complete, tmp_path):
    report = audited(complete)
    report["schema"] = admission.LEGACY_SCHEMA
    binding = put(tmp_path / "historical.json", report)
    assert admission.read_admission_report(Path(binding["path"])) == report
    with pytest.raises(ValueError, match="historical v1"):
        admission.validate_training_admission(Path(binding["path"]), complete.splits["train"], complete.splits["validation"])


def test_changed_renderer_receipt_rejected(complete, tmp_path):
    receipt = protocol.bound(complete.rb)
    receipt["num_inference_steps"] = 5
    binding = put(tmp_path / "wrong-renderer.json", receipt)
    with pytest.raises(ValueError, match="renderer execution lineage"):
        audited(complete, render_receipt_path=binding["path"], render_receipt_sha256=binding["raw_sha256"])


@pytest.mark.parametrize("batch,tail", [(8, 7), (2, 1)])
def test_actual_candidate_topologies_require_exact_tail_B_qualification(complete, tmp_path, batch, tail):
    doc = deepcopy(complete.protocol)
    inventory = protocol.bound(doc["motion_inventory"])
    plan = build_plan(inventory, seed=41, num_envs=batch, chunks=3200, success_cases=320,
                      selection_policy="matched_exposure_v2")
    doc["sampling_plan"] = put(tmp_path / "candidate-plan.json", plan)
    template = protocol.bound(doc["capacity_receipts"][0])
    doc["capacity_receipts"] = []
    for role in admission.SPLITS:
        record = {**template, "collection_role": role, "num_envs": batch, "full_chunks_saved": batch * 10,
                  "emitted_chunks_per_lane": [10] * batch}
        doc["capacity_receipts"].append(put(tmp_path / f"cap-{role}-{batch}.json", record))
    binding = put(tmp_path / "without-tail.json", doc)
    with pytest.raises(ValueError, match="actual role/B/ordinal"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])
    for role in admission.SPLITS:
        record = {**template, "collection_role": role, "num_envs": tail, "full_chunks_saved": tail * 10,
                  "emitted_chunks_per_lane": [10] * tail}
        doc["capacity_receipts"].append(put(tmp_path / f"cap-{role}-{tail}.json", record))
    binding = put(tmp_path / "with-tail.json", doc)
    *_, summary = protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])
    assert summary["required_actual_num_envs"] == [tail, batch]


@pytest.mark.parametrize("field,value", [("collection_role", "unknown"), ("concurrent_collectors", 2),
    ("simultaneous_collectors_observed", 0), ("host_id", "different-host"),
    ("runtime_min_mem_available_bytes", 1), ("maximum_completed_ordinal_tested", 0),
    ("vector_steps", 81), ("purpose", "teacher_capture_only"), ("oom_events_delta", 1)])
def test_capacity_is_role_duration_concurrency_and_host_specific(complete, tmp_path, field, value):
    doc = deepcopy(complete.protocol)
    record = protocol.bound(doc["capacity_receipts"][0])
    record[field] = value
    doc["capacity_receipts"][0] = put(tmp_path / "bad-capacity.json", record)
    binding = put(tmp_path / "bad-protocol.json", doc)
    with pytest.raises(ValueError, match="capacity"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])


def test_asset_registry_alone_is_not_current_bytes_proof(complete, tmp_path):
    doc = deepcopy(complete.protocol)
    receipt = protocol.bound(doc["render_asset_identity"])
    receipt["hashes_verified"] = False
    doc["render_asset_identity"] = put(tmp_path / "unverified-assets.json", receipt)
    binding = put(tmp_path / "unverified-protocol.json", doc)
    with pytest.raises(ValueError, match="verified renderer"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])


def test_prior_hash_receipt_rejected_after_asset_identity_changes(complete, tmp_path):
    doc = deepcopy(complete.protocol)
    receipt = protocol.bound(doc["render_asset_identity"])
    changed = tmp_path / "same-size-new-inode"
    changed.write_bytes(b"raw")
    receipt["files"][0]["path"] = str(changed)
    doc["render_asset_identity"] = put(tmp_path / "changed-assets.json", receipt)
    binding = put(tmp_path / "changed-protocol.json", doc)
    with pytest.raises(ValueError, match="metadata changed"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])


def test_resume_binds_qualification_report_hash_too():
    from pipeline.bc_training.train_bc_h200 import validate_resume_data_admission
    expected = dict(production_admitted=False, qualification_admitted=True, report_sha256="a" * 64)
    validate_resume_data_admission({"data_admission": {"report_sha256": "a" * 64}}, expected)
    for checkpoint in ({}, {"data_admission": {"report_sha256": "b" * 64}}):
        with pytest.raises(ValueError, match="including qualification"):
            validate_resume_data_admission(checkpoint, expected)
    with pytest.raises(ValueError, match="cannot drop"):
        validate_resume_data_admission({"data_admission": {"report_sha256": "a" * 64}}, None)


@pytest.mark.parametrize("mutation", ["adapter", "image", "missing_cohort", "missing_session", "changed_session"])
def test_execution_source_runtime_and_lifecycle_are_not_declared_writer_only(complete, tmp_path, mutation):
    doc = deepcopy(complete.protocol)
    if mutation == "adapter": doc["collector_execution"]["adapter_source_sha256"] = "0" * 64
    elif mutation == "image": doc["collector_execution"]["container_image_id"] = "sha256:" + "0" * 64
    elif mutation == "missing_cohort": doc["cohort_execution_receipts"].pop()
    elif mutation == "missing_session": doc["session_execution_receipts"] = []
    else:
        receipt = protocol.bound(doc["session_execution_receipts"][0])
        receipt["native_initializations"] = 2
        doc["session_execution_receipts"][0] = put(tmp_path / "changed-session.json", receipt)
    binding = put(tmp_path / "execution-protocol.json", doc)
    with pytest.raises(ValueError):
        audited(complete, protocol_path=binding["path"], protocol_sha256=binding["raw_sha256"])


def test_standalone_protocol_rejects_duplicate_rendered_chunk(complete, tmp_path):
    doc = deepcopy(complete.rendered)
    doc["chunks"][-1] = doc["chunks"][0]
    rendered = put(tmp_path / "duplicated.json", doc)
    with pytest.raises(ValueError, match="population size"):
        protocol.audit_production_protocol(complete.pb["path"], complete.pb["raw_sha256"], rendered["path"],
            complete.rb["path"], complete.rb["raw_sha256"])


def test_capacity_aggregate_count_cannot_hide_short_lane_exposure(complete, tmp_path):
    doc = deepcopy(complete.protocol)
    record = protocol.bound(doc["capacity_receipts"][0])
    record["emitted_chunks_per_lane"][0] = 9
    record["emitted_chunks_per_lane"][1] = 11
    doc["capacity_receipts"][0] = put(tmp_path / "skewed-capacity.json", record)
    binding = put(tmp_path / "skewed-protocol.json", doc)
    with pytest.raises(ValueError, match="capacity"):
        protocol.validate_protocol_inputs(binding["path"], binding["raw_sha256"])
