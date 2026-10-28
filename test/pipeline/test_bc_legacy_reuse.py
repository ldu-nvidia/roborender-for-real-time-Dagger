"""CPU-only legacy contract tests; synthetic fixtures are never production proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training import legacy_reuse as reuse
from pipeline.legacy.bc_collection import fresh_validation
from test.pipeline.test_bc_offline_roles_v3 import make_panel, put
from test.pipeline.test_bc_sampling_plan import inventory


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rows = []
    for index in range(2000):
        rows.append(dict(chunk_id=f"bc_chunk_{index:06d}", frames=81,
            source_sha256=reuse.sha(str(index).encode()), source_size_bytes=10,
            source_run_tag="historic-" + str(index // 250), rank=0, local_env_id=index % 28,
            global_env_id=index % 28, sequence_id=index // 28,
            collection_chunk_id=f"old-{index}", collection_filename=f"old-{index}.npz",
            simulator_seed=None if index < 490 else 2 + index // 250, source_batch=index // 500,
            isaac_npz=f"archive/isaac/bc_chunk_{index:06d}.npz", rendered_npz=f"old/{index}.npz"))
    science = dict(hashes=dict(teacher="1"*64, walk="2"*64, stand="3"*64,
                              trainer=reuse.NATIVE_COLLECTOR_SHA256))
    parent = dict(schema="roborender.bc_unified_pool.v2", chunk_count=2000,
                  scientific_identity=science, chunks=deepcopy(rows))
    parent_binding = put(tmp_path / "parent.json", parent)
    monkeypatch.setattr(reuse, "PARENT_SHA256", parent_binding["raw_sha256"])
    legacy = dict(schema="roborender.bc_pool.autoregressive.v1", status="sealed", frames_per_chunk=81,
        chunk_count=2000, parent_manifest_sha256=parent_binding["raw_sha256"],
        scientific_identity=science, chunks=rows)
    legacy_binding = put(tmp_path / "legacy.json", legacy)
    monkeypatch.setattr(reuse, "LEGACY_SHA256", legacy_binding["raw_sha256"])
    seal = put(tmp_path / "seal.json", dict(schema="roborender.bc_unified_pool_seal.v2", verdict="PASS",
        manifest_sha256=parent_binding["raw_sha256"], isaac_chunks=2000, frames_per_chunk=81))
    inv = inventory()
    inv_binding = put(tmp_path / "inventory.json", inv)
    _, panel_binding = make_panel(tmp_path, inv)
    fresh = fresh_validation.build_plan(motion_inventory=inv_binding, legacy_manifest=legacy_binding,
        closed_loop_test_panel=panel_binding, seed=761519)
    spec = put(tmp_path / "bootstrap.json", dict(checkpoint_sha256=reuse.BOOTSTRAP_SHA256,
        input_contract=dict(views=["head_color", "hand_left", "hand_right"]),
        output_contract=dict(train_action_slice=[0, 11])))
    return dict(run_name="bc-legacy-reuse-test", legacy_manifest=legacy_binding,
        original_manifest=parent_binding, original_seal=seal, legacy_dataset_root=str(tmp_path),
        fresh_validation_plan=put(tmp_path / "fresh.json", fresh), closed_loop_test_panel=panel_binding,
        bootstrap_spec=spec, bootstrap_path=str(tmp_path / "bootstrap.pt"), dataset_root=str(tmp_path))


def test_plan_exact_counts_unknowns_and_no_admission(inputs):
    plan = reuse.build_plan(**inputs)
    assert reuse.validate_plan(plan) == plan
    assert plan["counts"] == {"train": 2000, "validation": 320}
    assert plan["training_batches_per_epoch"] == 125
    assert plan["historical"]["unknown_process_seed_chunks"] == 490
    assert set(plan["historical"]["unknown_native_fields"].values()) == {2000}
    assert plan["closed_loop_test"]["episode_count"] == 295
    assert plan["production_admitted"] is False
    assert plan["full295_training_coverage_certified"] is False
    assert plan["historical"]["payloads_opened"] is False
    assert plan["all_old_split_assignments_retired"] is True


@pytest.mark.parametrize("key,value", [("production_admitted", True), ("planning_only", False),
    ("full295_training_coverage_certified", True), ("training_batches_per_epoch", 160),
    ("initialization", "legacy_checkpoint_resume"), ("all_old_split_assignments_retired", False)])
def test_plan_cannot_escalate_or_change_algorithm(inputs, key, value):
    plan = reuse.build_plan(**inputs)
    plan[key] = value
    with pytest.raises(ValueError, match="differs"):
        reuse.validate_plan(plan)


def test_historical_file_tamper_fails_before_payload_read(inputs):
    Path(inputs["original_seal"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        reuse.build_plan(**inputs)


def test_arbitrary_legacy_pool_is_not_admitted(inputs):
    inputs["legacy_manifest"]["raw_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="separately reviewed"):
        reuse.build_plan(**inputs)


def test_unknown_native_identity_cannot_be_fabricated(inputs, monkeypatch):
    path = Path(inputs["legacy_manifest"]["path"])
    doc = json.loads(path.read_bytes())
    doc["chunks"][0]["motion_id"] = "guessed-from-env-id"
    inputs["legacy_manifest"] = put(path, doc)
    monkeypatch.setattr(reuse, "LEGACY_SHA256", inputs["legacy_manifest"]["raw_sha256"])
    with pytest.raises(ValueError, match="fabricated"):
        reuse.build_plan(**inputs)


def test_trained_checkpoint_cannot_replace_clean_bootstrap(inputs):
    spec = reuse.bound(inputs["bootstrap_spec"])
    spec["checkpoint_sha256"] = "e" * 64
    inputs["bootstrap_spec"] = put(Path(inputs["bootstrap_spec"]["path"]), spec)
    with pytest.raises(ValueError, match="clean immutable"):
        reuse.build_plan(**inputs)


def test_fresh_plan_must_bind_same_old_population_and_panel(inputs, monkeypatch):
    fresh = reuse.bound(inputs["fresh_validation_plan"])
    fresh["legacy_manifest"] = dict(path="/elsewhere", raw_sha256="f"*64)
    inputs["fresh_validation_plan"] = put(Path(inputs["fresh_validation_plan"]["path"]), fresh)
    monkeypatch.setattr(fresh_validation, "validate_plan", lambda value: value)
    with pytest.raises(ValueError, match="different legacy"):
        reuse.build_plan(**inputs)


@pytest.mark.parametrize("flag,resume,state", [(False,None,None), (True,None,"/old-state.json")])
def test_training_requires_explicit_clean_bootstrap_before_any_receipt_read(flag, resume, state):
    with pytest.raises(ValueError, match="clean bootstrap"):
        reuse.validate_training_admission("/does-not-exist", "/train", "/val", allow_legacy_reuse=flag,
            run_name="bc-legacy-reuse-test", bootstrap_path="/bootstrap.pt", policy_spec_path="/spec.json",
            resume_checkpoint=resume, resume_validation_state=state)


def test_plan_cannot_be_used_as_admission(inputs, tmp_path):
    plan = reuse.build_plan(**inputs)
    path = tmp_path / "plan.json"
    put(path, plan)
    with pytest.raises(ValueError, match="invocation"):
        reuse.validate_training_admission(path, tmp_path/"train", tmp_path/"val", allow_legacy_reuse=True,
            run_name=plan["run_name"], bootstrap_path=inputs["bootstrap_path"],
            policy_spec_path=inputs["bootstrap_spec"]["path"], dataset_root=tmp_path)


def test_legacy_split_documents_share_one_manifest_and_keep_exact_roles(tmp_path):
    rows = [dict(chunk_id=role+str(i), split=role, stream_id=role+str(i), rendered_npz=f"{role}/{i}.npz")
            for role, n in reuse.COUNTS.items() for i in range(n)]
    pool = dict(schema=reuse.POOL_SCHEMA, counts=reuse.COUNTS, chunks=rows)
    b = put(tmp_path/"combined.json", pool)
    docs = reuse.split_documents(pool, b, split_dir=tmp_path/"splits")
    assert set(docs) == {"train", "validation"}
    assert [len(docs[r]["items"]) for r in docs] == [2000, 320]
    assert docs["train"]["source_manifest_sha256"] == docs["validation"]["source_manifest_sha256"]
    assert not ({r["stream_id"] for r in docs["train"]["items"]} & {r["stream_id"] for r in docs["validation"]["items"]})


def test_old_render_lineage_rejected_even_with_matching_receipt(monkeypatch, tmp_path):
    from pipeline.bc_training import render_execution
    source = dict(chunk_id="x", source_sha256="a"*64, frames=81)
    receipt = dict(source_pool_raw_sha256="d"*64)
    rendered = dict(schema="roborender.bc_pool.rendered.v2", render_contract="raven_three_view_autoregressive_v1",
        render_recipe="legacy", max_chain_chunks=6, frames_per_chunk=81,
        chunks=[dict(**source, rendered_sha256="c"*64)])
    evidence = {"protocol":put(tmp_path/"protocol.json",dict(source_pool=dict(raw_sha256="d"*64))),
        "render_plan":put(tmp_path/"plan.json",{}), "rendered_manifest":put(tmp_path/"rendered.json",rendered),
        "execution_receipt":put(tmp_path/"receipt.json",receipt)}
    monkeypatch.setattr(render_execution, "finish", lambda *args: receipt)
    with pytest.raises(ValueError, match="recipe"):
        reuse._rendered_population(evidence, {source["source_sha256"]:source})


def test_strict_fresh_data_gate_does_not_accept_legacy_schema(tmp_path):
    from pipeline.bc_training.data_admission import validate_training_admission
    p=tmp_path/"admission.json"
    put(p, dict(schema=reuse.ADMISSION_SCHEMA, legacy_reuse_admitted=True, production_admitted=False))
    with pytest.raises(ValueError, match="unknown admission"):
        validate_training_admission(p, tmp_path/"train", tmp_path/"val")


def test_exact_legacy_protocol_resume_preserves_persistence():
    admission=dict(schema=reuse.ADMISSION_SCHEMA, legacy_reuse_admitted=True, report_sha256="a"*64)
    reuse.validate_resume_identity(dict(data_admission=admission), admission)
    for old in ({}, dict(report_sha256="a"*64), {**admission,"report_sha256":"b"*64},
                {**admission,"legacy_reuse_admitted":False}):
        with pytest.raises(ValueError, match="exact same"):
            reuse.validate_resume_identity(dict(data_admission=old), admission)


@pytest.fixture
def native_complete(inputs, tmp_path):
    from test.pipeline.test_bc_fresh_validation import native_envelopes
    from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
    from test.pipeline.test_bc_capacity_reset_equivalence import fixture as capacity_fixture
    plan = reuse.build_plan(**inputs)
    fresh = reuse.bound(inputs["fresh_validation_plan"])
    root = tmp_path / "fresh-data"
    (root/"isaac").mkdir(parents=True)
    cohorts, transfers = native_envelopes(fresh, root)
    cb, tb, ab, sb = [], [], [], []
    execution = dict(adapter_source_sha256="a"*64, source_release_sha256="b"*64,
        container_image_id="sha256:"+"c"*64, native_runtime_version="synthetic-CPU-only",
        lifecycle_recipe="one_kit_fresh_seeded_scene_per_declared_cohort_v1",
        memory_buffer_recipe="original_raven_capture_lists_and_writer_v1", runtime_contract_sha256="d"*64)
    for index, (c, t) in enumerate(zip(cohorts, transfers)):
        for record in t["document"]["files"]:
            Path(record["destination_path"]).write_bytes(b"x"*123)
        # Preserve the raw digest used by native/transfer metadata fixtures.
        cp=root/f"cohort-{index}.json"; cp.write_bytes(reuse.canonical(c["document"]))
        tp=root/f"transfer-{index}.json"; tp.write_bytes(reuse.canonical(t["document"]))
        cb.append(reuse.binding(cp)); tb.append(reuse.binding(tp))
        doc=c["document"]
        session=dict(schema="roborender.bc-session-execution.v1", verdict="PASS", native_initializations=1,
            scene_initializations=1, collector_execution=execution,
            request_identities={doc["request_sha256"]:reuse.sha(reuse.canonical(doc["request"]))})
        for key in ("source_manifest_raw_sha256", "port_manifest_raw_sha256", "launch_raw_sha256",
                    "session_sha256", "session_canonical_sha256", "native_result_raw_sha256", "host_result_raw_sha256"):
            session[key] = reuse.sha(f"{key}-{index}".encode())
        sb.append(put(root/f"session-{index}.json",session))
        attestation=dict(schema="roborender.bc-cohort-execution.v1", verdict="PASS", collector_execution=execution,
            cohort_manifest_raw_sha256=c["sha256"], request_sha256=doc["request_sha256"],
            request_canonical_sha256=reuse.sha(reuse.canonical(doc["request"])),
            session_execution_canonical_sha256=reuse.sha(reuse.canonical(session)),
            actual_num_envs=len(doc["request"]["motions"]), actual_vector_steps=doc["vector_steps"])
        ab.append(put(root/f"attestation-{index}.json",attestation))
    pool,_=build_pool(fresh, cohorts, transfers, dataset_root=root, expected_total=320)
    capacities=[]
    for size in (1,6):
        protocol, receipt=capacity_fixture()
        receipt.update(num_envs=size, collector_execution=execution, model_sha256=plan["historical"]["model_sha256"])
        for idx,row in enumerate(receipt["per_flush_memory"]): row["emitted_chunks_per_lane"]=[idx]*size
        capacities.append(put(root/f"capacity-{size}.json",receipt))
    motions=[dict(motion_id=k,sha256=v,eligible=True,frame_counts_match=True,
                  native_skips_short_motion=False,native_random_start_choices=100)
             for k,v in plan["target_motion_bindings"].items()]
    eligibility=put(root/"eligibility.json",dict(schema="roborender.native-motion-eligibility.cpu.v1",
        motion_inventory_sha256=fresh["motion_inventory_sha256"], cpu_only=True, cuda_initialized=False,
        isaac_imported=False, eligible_count=295, motions=motions, native_source_hashes=reuse.NATIVE_RESET_PINS,
        invalid_or_short=[],frame_count_mismatches=[]))
    evidence=dict(source_pool=put(root/"manifest.json",pool),cohort_manifests=cb,transfer_receipts=tb,
        cohort_execution_receipts=ab,session_execution_receipts=sb,collector_execution=execution,
        execution_topology=protocol["execution_topology"],capacity_receipts=capacities,native_motion_eligibility=eligibility)
    return plan,fresh,pool,evidence


def test_genuine_native_ingestion_and_capacity_checks_are_reused(native_complete):
    plan,fresh,pool,evidence=native_complete
    assert reuse._native_execution(evidence,fresh,plan["historical"]["model_sha256"])==pool


@pytest.mark.parametrize("fault", ["missing_cohort", "missing_session", "wrong_model", "no_tail_capacity", "unsafe_reserve", "bad_eligibility"])
def test_missing_or_changed_fresh_runtime_proof_fails(native_complete, fault, tmp_path):
    plan,fresh,pool,evidence=native_complete
    models=plan["historical"]["model_sha256"]
    if fault=="missing_cohort": evidence["cohort_manifests"].pop()
    elif fault=="missing_session": evidence["session_execution_receipts"].pop()
    elif fault=="wrong_model": models={**models,"teacher":"f"*64}
    elif fault=="no_tail_capacity": evidence["capacity_receipts"]=evidence["capacity_receipts"][1:]
    elif fault=="unsafe_reserve": evidence["execution_topology"]["minimum_host_reserve_bytes"]=1
    else:
        doc=reuse.bound(evidence["native_motion_eligibility"]); doc["motions"][0]["eligible"]=False
        evidence["native_motion_eligibility"]=put(tmp_path/"bad-eligibility.json",doc)
    with pytest.raises(ValueError): reuse._native_execution(evidence,fresh,models)


def test_trainer_dispatch_is_explicit_and_preserves_strict_resume(monkeypatch, tmp_path):
    from argparse import Namespace
    from pipeline.bc_training import train_bc_h200 as trainer
    report=dict(schema=reuse.ADMISSION_SCHEMA, purpose="legacy_training_reuse", legacy_reuse_admitted=True,
                production_admitted=False)
    path=tmp_path/"admission.json"; put(path,report)
    args=Namespace(data_admission_json=path,require_production_data_admission=True,allow_qualification_data=False,
        allow_legacy_reuse=True,run_name="bc-legacy-reuse-unit",train_split=tmp_path/"train",validation_split=tmp_path/"val",
        bootstrap=tmp_path/"bootstrap",policy_spec=tmp_path/"spec",resume_checkpoint=None,
        resume_validation_state=None,dataset_root=tmp_path)
    captured={}
    def admit(*a,**kwargs): captured.update(kwargs); return report
    monkeypatch.setattr(reuse,"validate_training_admission",admit)
    result=trainer.load_data_admission(args)
    assert result["status"]=="legacy_reuse_admitted" and captured["allow_legacy_reuse"] is True
    trainer.validate_resume_data_admission(dict(data_admission=result),result)
    with pytest.raises(ValueError,match="exact same"):
        trainer.validate_resume_data_admission(dict(data_admission={}),result)
    args.allow_qualification_data=True
    with pytest.raises(ValueError,match="bypass"): trainer.load_data_admission(args)


@pytest.fixture
def admitted(native_complete, tmp_path, monkeypatch):
    """Complete native metadata path; renderer producer separately unit-tested."""
    from pipeline.bc_training import render_execution
    plan,fresh,pool,evidence=native_complete
    old=reuse.bound(plan["legacy_manifest"])
    rendering={}
    receipts={}
    for role,source in (("train",old),("validation",pool)):
        folder=tmp_path/("rendered-"+role); folder.mkdir()
        rendered=[]
        for row in source["chunks"]:
            name=row["chunk_id"]+".npz"
            (folder/name).write_bytes(b"r")
            rendered.append({**row,"rendered_npz":folder.name+"/"+name,
                "rendered_sha256":reuse.sha((role+name).encode()),"rendered_size_bytes":1,
                "render_contract":"raven_three_view_independent_v2","render_chain_length":1,
                "render_chain_index":0,"render_inherits_previous_reference":False})
        manifest=dict(schema="roborender.bc_pool.rendered.v2",frames_per_chunk=81,
            render_contract="raven_three_view_independent_v2",render_recipe="raven_3view_50step_independent_v2",
            max_chain_chunks=1,rendered_pool=str(folder),chunks=rendered)
        source_binding=plan["original_manifest"] if role=="train" else evidence["source_pool"]
        manifest_binding=put(tmp_path/f"rendered-{role}.json",manifest)
        receipt=dict(source_pool_raw_sha256=source_binding["raw_sha256"])
        receipts[manifest_binding["path"]]=receipt
        rendering[role]=dict(protocol=put(tmp_path/f"render-protocol-{role}.json",dict(source_pool=source_binding)),
            render_plan=put(tmp_path/f"render-plan-{role}.json",{}),rendered_manifest=manifest_binding,
            execution_receipt=put(tmp_path/f"render-receipt-{role}.json",receipt))
    monkeypatch.setattr(render_execution,"finish",lambda p,l,m:receipts[str(m)])
    population=reuse.assemble_population(plan,evidence,rendering)
    pbind=put(tmp_path/"combined.json",population)
    docs=reuse.split_documents(population,pbind,split_dir=tmp_path/"splits")
    splits={r:put(tmp_path/"splits"/(r+".json"),d) for r,d in docs.items()}
    plan_binding=put(tmp_path/"reuse-plan.json",plan)
    report=reuse.audit_admission(plan_binding=plan_binding,native_evidence=evidence,rendering=rendering,
        population_binding=pbind,split_files=splits)
    report_binding=put(tmp_path/"admission.json",report)
    return plan,report,report_binding


def test_full_legacy_admission_rederives_exact_native_render_and_split_bindings(admitted):
    plan,report,b=admitted
    assert report["legacy_reuse_admitted"] is True and report["production_admitted"] is False
    assert report["counts"]=={"train":2000,"validation":320}
    assert reuse.validate_training_admission(b["path"],report["split_files"]["train"]["path"],
        report["split_files"]["validation"]["path"],allow_legacy_reuse=True,run_name=plan["run_name"],
        bootstrap_path=plan["bootstrap_path"],policy_spec_path=plan["bootstrap_spec"]["path"],
        dataset_root=plan["dataset_root"])==report
    # A fresh immutable attempt name is allowed only for strict resume; model
    # checkpoint admission itself is checked by validate_resume_identity.
    assert reuse.validate_training_admission(b["path"],report["split_files"]["train"]["path"],
        report["split_files"]["validation"]["path"],allow_legacy_reuse=True,run_name="resume-attempt-2",
        bootstrap_path=plan["bootstrap_path"],policy_spec_path=plan["bootstrap_spec"]["path"],
        dataset_root=plan["dataset_root"],resume_checkpoint="/same-protocol/last.pt")==report
    with pytest.raises(ValueError,match="invocation"):
        reuse.validate_training_admission(b["path"],report["split_files"]["train"]["path"],
            report["split_files"]["validation"]["path"],allow_legacy_reuse=True,run_name="unrelated-fresh-run",
            bootstrap_path=plan["bootstrap_path"],policy_spec_path=plan["bootstrap_spec"]["path"],
            dataset_root=plan["dataset_root"])


def test_admitted_split_cannot_reintroduce_old_validation_or_test(admitted):
    plan,report,b=admitted
    path=Path(report["split_files"]["train"]["path"])
    doc=json.loads(path.read_bytes()); doc["items"][0]["split"]="test"
    path.write_bytes(reuse.canonical(doc))
    with pytest.raises(ValueError,match="changed"):
        reuse.validate_training_admission(b["path"],path,report["split_files"]["validation"]["path"],
            allow_legacy_reuse=True,run_name=plan["run_name"],bootstrap_path=plan["bootstrap_path"],
            policy_spec_path=plan["bootstrap_spec"]["path"],dataset_root=plan["dataset_root"])


def test_legacy_success_planner_uses_separate_test_after_mse_selection(admitted,tmp_path,monkeypatch):
    from pipeline.bc_training.plan_success_test import plan as success_plan
    from pipeline.deployment.rtx_actor import production_success_contract as contract
    plan,report,admission=admitted
    # This unit fixture is not the real panel. Bind only this test's synthetic
    # identity; production continues to require its accepted immutable panel.
    panel_ref=plan['closed_loop_test_panel']
    source_panel = reuse.bound(panel_ref)
    rendered = {**source_panel, 'panel_id': contract.PANEL_ID, 'observation_mode': 'image3f_3view'}
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    import production_success_contract as native_contract
    for selected in (contract, native_contract):
        monkeypatch.setattr(selected, 'LEGACY_PANEL_ID', source_panel['panel_id'])
        monkeypatch.setattr(selected, 'LEGACY_PANEL_RAW_SHA256', panel_ref['raw_sha256'])
        monkeypatch.setattr(selected, 'LEGACY_PANEL_SHA256', contract.panel_sha256(source_panel))
        monkeypatch.setattr(selected, 'PANEL_RAW_SHA256', hashlib.sha256(reuse.canonical(rendered) + b'\n').hexdigest())
        monkeypatch.setattr(selected, 'PANEL_SHA256', contract.panel_sha256(rendered))
    output=tmp_path/"trained"; (output/"validation_checkpoints").mkdir(parents=True)
    checkpoint=output/"validation_checkpoints"/"epoch_0001_step_000125.pt"
    checkpoint.write_bytes(b"synthetic test checkpoint, never torch loaded")
    (output/"best_validation.pt").hardlink_to(checkpoint)
    state=dict(status="complete",stop_reason="validation_saturation",selected_checkpoint=str(output/"best_validation.pt"),
        history=[dict(epoch=1,mse=.5,checkpoint=str(checkpoint))],best_epoch=1,best_mse=.5,
        train_split_sha256=report["split_files"]["train"]["raw_sha256"],
        validation_split_sha256=report["split_files"]["validation"]["raw_sha256"])
    config=dict(data_admission_json=admission["path"],allow_legacy_reuse=True,run_name=plan["run_name"],
        train_split=report["split_files"]["train"]["path"],validation_split=report["split_files"]["validation"]["path"],
        bootstrap=plan["bootstrap_path"],policy_spec=plan["bootstrap_spec"]["path"],dataset_root=plan["dataset_root"],
        output=str(output))
    args=dict(training_state=put(output/"validation_state.json",state),training_config=put(output/"run_config.json",config),
        data_admission=admission,output=tmp_path/"success-plan.json",command_id="bc-success-legacy-test",
        resident_source_sha256="a"*64,resident_gpu_uuid="GPU-example",resident_port=19100,
        resident_token_file="/owned/token",resident_source_root="/owned/source",resident_jobs_root="/owned/jobs",
        resident_assets_root="/owned/assets",panel_relative_path="panels/final.json",
        rendered_observation_config=put(tmp_path / 'renderer.json', rendering_config(rendered)))
    result=success_plan(**args)
    doc=reuse.bound(result["plan"])
    assert doc["legacy_reuse_training_completed"] is True and doc["production_training_completed"] is False
    assert doc["test_case_count"]==295 and doc["executed"] is False and doc["optimizer_steps"]==0
    assert doc['source_data_admission_test_panel'] == plan['closed_loop_test_panel']
    assert reuse.bound(doc['test_panel']) == rendered


def test_metadata_publisher_never_overwrites_existing_admission(admitted,tmp_path):
    _,report,_=admitted
    target=tmp_path/"new-admission"
    kwargs=dict(plan_binding=report["plan"],native_evidence=report["native_evidence"],
                rendering=report["rendering"],output_dir=target)
    result=reuse.publish_admission(**kwargs)
    assert result["legacy_reuse_admitted"] is True and result["production_admitted"] is False
    original=(target/"data-admission.json").read_bytes()
    with pytest.raises(FileExistsError,match="fresh"):
        reuse.publish_admission(**kwargs)
    assert (target/"data-admission.json").read_bytes()==original


def test_metadata_publisher_does_not_create_bundle_before_evidence_passes(admitted,tmp_path):
    _,report,_=admitted
    target=tmp_path/"not-admitted"
    report["native_evidence"]["capacity_receipts"]=[]
    with pytest.raises(ValueError,match="qualification"):
        reuse.publish_admission(plan_binding=report["plan"],native_evidence=report["native_evidence"],
                                rendering=report["rendering"],output_dir=target)
    assert not target.exists()
