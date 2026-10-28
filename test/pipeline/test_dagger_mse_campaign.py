"""CPU-only host/campaign/provenance gates; fixtures are not native evidence."""
from test.deployment.source_paths import REPO_ROOT
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pipeline.dagger.beta_campaign import immutable_json, make_first_round_config
from pipeline.dagger.mse_campaign import MSECampaign, validate_plan, PLAN_SCHEMA, import_final_test
from pipeline.dagger.export_heldout_manifest import export_manifest, validate_export_admission
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical
from pipeline.evaluation.beta_small_batch_panel import make_panel
from pipeline.deployment.h200.mse_worker_proxy import command
from pipeline.deployment.resident_campaign_site import ResidentSite
from test.pipeline.test_bc_launcher_data import published

ROOT = REPO_ROOT


@pytest.fixture
def setup(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"CPU fixture, not a model")
    panel = tmp_path / "panel.json"
    panel.write_bytes(canonical(make_panel("a" * 64, smoke=True)))
    config = make_first_round_config("dagger-leo-smoke-mse-fixture", seed=1234, checkpoint=parent,
        checkpoint_sha=file_sha256(parent), panel=panel, panel_sha=file_sha256(panel), source_sha="b" * 64,
        gateway_pids=[1, 2, 3, 4], smoke=True, deployment_host="leo-rtx-6k")
    plan = {"schema": PLAN_SCHEMA, "run_id": config["run_id"], "rounds": [
        {"round_id": rid, "validation_manifest_path": str(tmp_path / f"r{rid}-validation.json"),
         "validation_manifest_sha256": None, "validation_admission_path": str(tmp_path / f"r{rid}-admission.json"),
         "validation_admission_sha256": None, "interval_iterations": 1, "max_iterations": 3,
         "min_validation_epochs": 1, "patience": 1, "relative_improvement": .005, "absolute_improvement": 1e-6}
        for rid in range(1, 5)]}
    return config, plan


def collection(tmp_path, config, plan):
    store = StateStore(tmp_path / "ledger")
    payload = tmp_path / "real-shaped-but-not-loaded.npz"
    payload.write_bytes(b"metadata fixture deliberately not an ndarray archive")
    assignment = {"schema": "roborender.native-episode-assignment.v1", "purpose": "validation",
                  "run_id": "heldout-only", "episode_id": "actual-episode-1", "simulator_seed": 9876,
                  "motion_id": "motion-1", "motion_sha256": "f" * 64, "observation_source": "generated_controls"}
    evidence = tmp_path / "native-assignment.json"
    evidence.write_bytes(canonical(assignment))
    receipt = {"schema": "roborender.dagger-heldout-collection.v1", "purpose": "validation",
        "source_run_id": "heldout-only", "round_id": 1, "policy_spec_sha256": "a" * 64,
        "teacher_checkpoint_sha256": "c" * 64, "collection_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
        "teacher_beta": .75, "motion_inventory_sha256": "d" * 64, "motion_support": ["motion-1"],
        "training_simulator_seeds": [1234, 1235], "test_simulator_seeds": [1103, 2112], "chunk_ids": ["val-chunk"],
        "episode_assignments": [{**{key: assignment[key] for key in ("episode_id", "simulator_seed", "motion_id", "motion_sha256", "observation_source")},
            "native_assignment_path": str(evidence), "native_assignment_sha256": file_sha256(evidence)}]}
    path = tmp_path / "collection.json"
    path.write_bytes(canonical(receipt))
    meta = {"run_id": "heldout-only", "round_id": 1, "actor_id": "heldout-only-actor", "rollout_window_id": "heldout-unique-window",
            "checkpoint_sha256": config["student"]["initial_checkpoint_sha256"], "first_step_id": 0, "last_step_id": 80,
            "teacher_checkpoint_sha256": "c" * 64, "sealed_window_sha256": "e" * 64}
    with store.transaction() as db:
        owner = {"schema": "roborender.dagger-validation-collection.v1", "purpose": "validation",
            "run_id": "heldout-only", "round_id": 1, "actor_ids": ["heldout-only-actor"],
            "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
            "teacher_checkpoint_sha256": "c" * 64, "policy_spec_sha256": "a" * 64,
            "motion_inventory_sha256": "d" * 64, "teacher_beta": .75, "collection_plan_sha256": "b" * 64}
        db.execute("INSERT INTO validation_collections VALUES (?,?,?,?,?,?)", ("heldout-only",
            hashlib.sha256(canonical(owner)).hexdigest(), canonical(owner).decode(), "complete", 0, 0))
        db.execute("INSERT INTO validation_collection_actors VALUES (?,?,?,?,?)", ("heldout-only", "heldout-only-actor", "val-chunk", "{}", 0))
        db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)", ("val-chunk", file_sha256(payload), payload.stat().st_size,
                   canonical(meta).decode(), "committed", str(payload), 0, 0))
        db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)", ("val-chunk", file_sha256(payload), '["actual-episode-1"]',
                   "heldout-unique-window", "c" * 64, "e" * 64))
    return store, path, receipt


def admitted(tmp_path, config, plan):
    store, path, receipt = collection(tmp_path, config, plan)
    row = plan["rounds"][0]
    with store.connection() as db:
        manifest, report = export_manifest(db, path, file_sha256(path), Path(row["validation_manifest_path"]))
    immutable_json(Path(row["validation_admission_path"]), report)
    return store, path, receipt, manifest, report


def test_exporter_reads_only_actual_sealed_metadata_and_pins_assignments(setup, tmp_path):
    config, plan = setup
    with patch("numpy.load", side_effect=AssertionError("must not load arrays")):
        store, path, receipt, manifest, report = admitted(tmp_path, config, plan)
        actual = validate_export_admission(Path(plan["rounds"][0]["validation_admission_path"]),
            file_sha256(Path(plan["rounds"][0]["validation_admission_path"])),
            Path(plan["rounds"][0]["validation_manifest_path"]), report["manifest_sha256"])
    assert actual == receipt and report["array_payloads_read"] is False
    assert manifest["chunks"][0]["source_episode_ids"] == ["actual-episode-1"]


@pytest.mark.parametrize("error", ["missing_assignment", "overlap", "wrong_parent", "partial", "optimizer"])
def test_exporter_fails_closed_missing_or_wrong_real_provenance(setup, tmp_path, error):
    config, plan = setup
    store, path, receipt = collection(tmp_path, config, plan)
    if error == "missing_assignment":
        del receipt["episode_assignments"][0]["native_assignment_path"]
    if error == "overlap":
        receipt["training_simulator_seeds"] = [9876]
    if error == "wrong_parent":
        receipt["collection_checkpoint_sha256"] = "0" * 64
    if error == "partial":
        with store.transaction() as db:
            db.execute("UPDATE chunk_reservations SET status='reserved'")
    if error == "optimizer":
        with store.transaction() as db:
            db.execute("INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ("bad", "heldout-only", 1, "a" * 64,
                       '["val-chunk"]', "succeeded", "{}", None, "log", None, 0, 0))
    path.write_bytes(canonical(receipt))
    with store.connection() as db, pytest.raises(ValueError):
        export_manifest(db, path, file_sha256(path), Path(plan["rounds"][0]["validation_manifest_path"]))


def test_campaign_waits_for_real_parent_population_without_allocating(setup, tmp_path):
    config, plan = setup
    operations = Mock()
    campaign = MSECampaign(config, tmp_path / "campaign", operations, plan=plan)
    result = campaign.tick()
    assert result["phase"] == "PREPARE"
    assert result["progress"]["action"] == "awaiting_real_heldout_population"
    operations.start.assert_not_called()
    request = json.loads((campaign.root / "round-1/heldout-population-request.json").read_bytes())
    assert request["optimizer_authorized"] is False
    assert request["parent_checkpoint"]["initial_checkpoint_sha256"] == config["student"]["initial_checkpoint_sha256"]


def test_campaign_sidecar_admitted_once_and_threaded_to_controller(setup, tmp_path):
    config, plan = setup
    admitted(tmp_path, config, plan)
    controller = Mock()
    factory = Mock(return_value=controller)
    campaign = MSECampaign(config, tmp_path / "campaign", Mock(), plan=plan, controller_factory=factory)
    campaign.save(config_path=str(tmp_path / "config.json"), service_binding={"token_file": "/token",
                  "mse_worker_command": ["/python", "/owned-worker.py"]})
    campaign.get_controller()
    assert factory.call_args.kwargs["mse_worker_command"] == ["/python", "/owned-worker.py"]
    context = campaign.operation_context()
    assert context["mse_campaign"] is True
    assert file_sha256(Path(context["mse_protocol_path"])) == context["mse_protocol_file_sha256"]
    protocol = json.loads(Path(context["mse_protocol_path"]).read_bytes())
    assert protocol["validation_manifest_sha256"] != "0" * 64
    controller.ensure_registered.assert_called_once()
    # Altering a deferred artifact after its first admission is still rejected.
    binding = campaign.root / "round-1/heldout-admission-binding.json"
    changed = json.loads(binding.read_bytes())
    changed["validation_manifest_sha256"] = "0" * 64
    binding.chmod(0o644)
    binding.write_bytes(canonical(changed))
    with pytest.raises(RuntimeError, match="frozen"):
        campaign.protocol_path()


def test_resident_collect_explicitly_threads_mse_sidecar_but_legacy_stays_legacy(setup, tmp_path):
    config, plan = setup
    admitted(tmp_path, config, plan)
    campaign = MSECampaign(config, tmp_path / "campaign", Mock(), plan=plan)
    request = campaign.operation_context()
    cfg = tmp_path / "config.json"
    cfg.write_bytes(canonical(config))
    request["round_config_path"] = str(cfg)
    site = ResidentSite.__new__(ResidentSite)
    site.request, site.config = request, config
    site.jobs, site.remote = tmp_path, tmp_path / "remote"
    site.run, site.rid, site.gpus = config["run_id"], 1, [0, 1]
    site.barrier = Mock(return_value={"owned_actors_stopped": True})
    site.publish, site.read = Mock(), Mock(return_value={"fake": "fixture"})
    site.pool_action = Mock(side_effect=[{}, {"results": [{"receipt": {"actor_state": {"gpu": gpu}}} for gpu in (0, 1)]}])
    site.collect()
    commands = site.pool_action.call_args_list[0].args[1]
    assert all(row["operation"] == "collect_mse_round" for row in commands)
    assert all(row["payload"]["mse_protocol_sha256"] == request["mse_protocol_file_sha256"] for row in commands)
    site.request = {"round_config_path": str(cfg)}
    site.pool_action.reset_mock(side_effect=True)
    site.pool_action.side_effect = [{}, {"results": []}]
    site.collect()
    assert all(row["operation"] == "collect_round" for row in site.pool_action.call_args_list[0].args[1])
    assert all("mse_protocol_file" not in row["payload"] for row in site.pool_action.call_args_list[0].args[1])


def test_worker_proxy_uses_only_pinned_ssh_and_no_allocation(tmp_path):
    conf = tmp_path / "ssh.conf"
    conf.write_text("# fixture")
    args = SimpleNamespace(host="owned-pod", ssh_config=conf, run_id="run", source_root=tmp_path,
                           asset_root=tmp_path, netrc_file=tmp_path / "netrc", request=tmp_path / "request.json",
                           result=tmp_path / "result.json")
    argv = command(args)
    assert argv[0] == "ssh" and "ControlPath=none" in argv and "--run-id run" in argv[-1]
    assert "run_heldout_mse_worker_h200.sh" in argv[-1]
    args.host = "bad; command"
    with pytest.raises(ValueError):
        command(args)


def test_native_mse_schema_validate_and_state_filename_are_explicit():
    source = (ROOT / "pipeline/deployment/rtx_actor/resident_isaac_trainer.py").read_text()
    ast.parse(source)
    assert '"collect_mse_round"' in source
    assert 'read_mse_protocol(self.artifact(payload["mse_protocol_file"])' in source
    assert 'state_name = "mse-beta-round-state.json" if operation == "collect_mse_round"' in source
    assert 'values.update(DAGGER_MSE_PROTOCOL_FILE=' in source


def test_bc_launcher_defaults_are_fail_closed_and_assets_external(published):
    from pipeline.bc_training.launcher_data import resolve_selection
    source = (ROOT / "pipeline/bc_training/train_raven_bc_h200.sh").read_text()
    assert "REQUIRE_PRODUCTION_DATA_ADMISSION:-1" in source
    assert "pipeline/bc_training/launcher_data.py" in source
    assert "splits_independent_v2" not in source and "splits_autoregressive_v1" not in source
    assert "ROBORENDER_ASSET_ROOT/checkpoints/students/BC" in source
    selection = resolve_selection(project_root=ROOT, asset_root=Path('/historical-assets'),
                                  environ=published['environ'])
    assert selection == dict(TRAINING_PROTOCOL='fixed_final', DATASET_ROOT=str(published['root']),
        TRAIN_SPLIT=str(published['train']), VALIDATION_SPLIT=str(published['validation']),
        DATA_ADMISSION_JSON=str(published['report']))
    published['report'].unlink()
    with pytest.raises(ValueError, match='training-union admission is missing'):
        resolve_selection(project_root=ROOT, asset_root=Path('/historical-assets'),
                          environ=published['environ'])


def completed_campaign_for_display_test(setup, tmp_path, *, final=False):
    """CPU fixture exercises the real ADVANCE branch, not a native claim."""
    from pipeline.dagger.beta_campaign import next_round_config
    from pipeline.dagger.services.mse_status import next_round_display_fields
    config, plan = setup
    campaign = MSECampaign(config, tmp_path / "display-campaign", Mock(), plan=plan)
    selected = {"sha256": "e" * 64, "path": str(tmp_path / "selected.pt")}
    if final:
        for _ in range(3):
            config = next_round_config(config, "b" * 64, selected["sha256"], selected["path"], [1, 2, 3, 4])
    controller = Mock()
    controller.mse.context.return_value = {"status": "complete", "selection": {"best_checkpoint_sha256": selected["sha256"]}}
    campaign.get_controller = Mock(return_value=controller)
    stale = {key: {"stale_round": 1} for key in next_round_display_fields()}
    stale.update(final_checkpoint=selected, evaluation_report={"purpose": "test", "status": "complete",
        "completed": 2, "checkpoint_sha256": selected["sha256"]}, mse_completion={"completed_iterations": 1},
        logging_receipt={"online_metrics_verified": True}, heldout_population_reused=True)
    campaign.save("ADVANCE", config=config, round_id=config["round_id"], first_config_sha256="b" * 64,
        service_binding={"identity": "old-control-owner-must-be-retired"},
        completed_rounds=[{"round_id": i} for i in range(1, config["round_id"])], **stale)
    return campaign, selected


def test_advance_clears_stale_display_but_preserves_ownership_and_history(setup, tmp_path):
    from pipeline.dagger.services.mse_status import next_round_display_fields
    campaign, selected = completed_campaign_for_display_test(setup, tmp_path)
    state = campaign.tick()
    assert state["phase"] == "PREPARE" and state["round_id"] == 2
    assert state["config"]["student"]["initial_checkpoint_sha256"] == selected["sha256"]
    assert state["service_binding"] == {"identity": "old-control-owner-must-be-retired"}
    assert state["completed_rounds"][-1]["selected_checkpoint"] == selected
    for key, expected in next_round_display_fields().items():
        assert state[key] == expected
    campaign.ops.start.assert_not_called()


@pytest.mark.parametrize('round_id', [1, 2, 3])
def test_each_round_handoff_uses_selected_best_not_latest_candidate(setup, tmp_path, round_id):
    from pipeline.dagger.beta_campaign import next_round_config
    campaign, selected = completed_campaign_for_display_test(setup, tmp_path)
    config = campaign.state['config']
    for _ in range(round_id - 1):
        config = next_round_config(config, 'b' * 64, selected['sha256'], selected['path'], [1, 2, 3, 4])
    latest = 'f' * 64
    campaign.save('ADVANCE', config=config, round_id=round_id,
        final_learner={'output_checkpoint_sha256': latest},
        mse_completion={'completed_iterations': 3, 'selected_checkpoint_sha256': selected['sha256'],
                        'last_candidate_checkpoint_sha256': latest})
    state = campaign.tick()
    assert state['round_id'] == round_id + 1
    assert state['config']['student']['initial_checkpoint_sha256'] == selected['sha256']
    assert state['config']['student']['initial_checkpoint'] == selected['path']
    assert state['config']['student']['initial_checkpoint_sha256'] != latest
    assert state['completed_rounds'][-1]['selected_checkpoint'] == selected
    campaign.ops.start.assert_not_called()


def test_complete_replaces_stale_selected_pending_test_progress(setup, tmp_path):
    campaign, selected = completed_campaign_for_display_test(setup, tmp_path, final=True)
    state = campaign.tick()
    assert state["phase"] == "COMPLETE" and len(state["completed_rounds"]) == 4
    assert state["progress"] == {"action": "campaign_complete", "status": "complete", "completed_rounds": 4,
        "validation_status": "completed_heldout_action_mse", "test_status": "complete",
        "selected_checkpoint_sha256": selected["sha256"], "checkpoint_selection_metric": "heldout_action_mse"}
    campaign.ops.start.assert_not_called()
