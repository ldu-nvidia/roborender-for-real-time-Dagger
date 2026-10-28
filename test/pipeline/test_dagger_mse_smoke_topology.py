"""Explicit B4 smoke planning is metadata-only and cannot weaken legacy B12."""
import copy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pipeline.dagger.prepare_mse_smoke import prepare
from pipeline.dagger.mse_campaign import validate_plan
from pipeline.dagger.beta_campaign import next_round_config
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, validate_config, driver_config, canonical
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.beta_validation import load_panel
from pipeline.bc_training.handoff_common import binding
from pipeline.deployment.rtx_actor.smoke_success_contract import PREFIX
from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config


@pytest.fixture
def staged(tmp_path):
    lanes = [{"lane": i, "motion_id": f"motion-{i}", "path": f"/assets/motion-{i}.npz", "sha256": str(i + 1) * 64} for i in range(2)]
    panel_id = PREFIX + "cpu-topology"
    panel = {"schema": "roborender.explicit-motion-success-panel.v1", "panel_id": panel_id,
        "purpose": "test", "motion_inventory_sha256": "a" * 64, "observation_mode": "image3f_3view",
        "action_mode": "deterministic_action_mean_11_plus_homie_15", "metric": "native_goal_or_until_done",
        "max_vector_steps_per_cohort": 450, "episode_count": 2, "cohorts": [{"assignment": {
            "schema": "roborender.native-motion-cohort.v1", "run_id": panel_id,
            "cohort_id": "test-0", "purpose": "test", "simulator_seed": 9000,
            "motion_inventory_sha256": "a" * 64, "lanes": lanes}, "case_ids": ["test-0", "test-1"]}]}
    path = tmp_path / "panel.json"
    path.write_bytes(canonical(panel))
    dependencies = tmp_path / "rendered-dependencies.json"
    dependencies.write_bytes(canonical(rendering_config(panel)))
    args = dict(root=tmp_path / "smoke", run_id="dagger-leo-smoke-b4-fixture", seed=100,
        checkpoint=tmp_path / "pending-bc.pt", checkpoint_sha="b" * 64, panel_path=path,
        source_sha="c" * 64, teacher_sha="d" * 64, policy_spec_sha="e" * 64, max_iterations=2,
        corrected_success=True, rendered_observation_config=binding(dependencies))
    report = prepare(**args)
    return args, report, json.loads(Path(report["config_path"]).read_bytes()), json.loads(Path(report["plan_path"]).read_bytes())


def test_b4_rounds_same_modes_separate_seeds_no_allocation(staged):
    args, report, config, plan = staged
    assert report["allocated_gpus"] == 0 and report["native_runtime_validated"] is False
    assert report["rows_per_window"] == 324 and len(actor_ids(config)) == 4
    assert report["max_updates_per_round"] == 2
    assert load_panel(config)["episode_count"] == 2
    assert load_panel(config)["observation_mode"] == "image3f_3view"
    assert config["validation"]["observation_mode"] == "generated"
    assert config["validation"]["rendered_observation_config"] == args["rendered_observation_config"]
    assert validate_plan(plan, config) == plan
    assert prepare(**args) == report
    train_seeds, val_seeds = set(), set()
    for rid, row in enumerate(plan["rounds"], 1):
        for pin in row["training_assignments"]:
            item = json.loads(Path(pin["path"]).read_bytes())
            assert file_sha256(Path(pin["path"])) == pin["sha256"]
            assert item["simulator_seed"] == 100 + 100003 * (rid - 1) + pin["physical_gpu_index"]
            train_seeds.add(item["simulator_seed"])
        collection = json.loads(Path(row["collection_plan_path"]).read_bytes())
        val_seeds.update(c["simulator_seed"] for c in collection["cohorts"])
        assert row["validation_manifest_sha256"] is None
    assert len(train_seeds) == len(val_seeds) == 8 and not train_seeds & val_seeds
    assert 9000 not in train_seeds | val_seeds


@pytest.mark.parametrize("fault", ["missing_template", "changed_template_hash", "raytraced_panel"])
def test_b4_cannot_stage_without_exact_generated_dependencies(staged, tmp_path, fault):
    args, _, _, _ = staged
    args = {**args, "root": tmp_path / ("rejected-" + fault)}
    if fault == "missing_template":
        args.pop("rendered_observation_config")
    elif fault == "changed_template_hash":
        args["rendered_observation_config"] = {**args["rendered_observation_config"], "raw_sha256": "0" * 64}
    else:
        panel = json.loads(args["panel_path"].read_bytes())
        panel["observation_mode"] = "raytraced_3view"
        path = tmp_path / "raytraced-panel.json"
        path.write_bytes(canonical(panel))
        args["panel_path"] = path
    with pytest.raises(ValueError):
        prepare(**args)
    assert not (args["root"] / "first-round.json").exists()


def test_b4_runtime_normalization_and_next_round_pin_change(staged):
    _, report, config, plan = staged
    config["h200"]["gateway_pids"] = [1, 2, 3, 4]
    assert driver_config(config, "f" * 64)["num_envs_per_gpu"] == 2
    next_config = next_round_config(config, report["config_sha256"], "9" * 64, "/checkpoints/r1.pt", [1, 2, 3, 4])
    next_config["motion_assignments"] = plan["rounds"][1]["training_assignments"]
    assert validate_config(next_config, runtime=True)["seed"] == 100103
    assert next_config["teacher_beta"] == .5


def test_old_schema_cannot_accept_new_small_batch(staged):
    _, _, config, _ = staged
    config["schema"] = "roborender.cross_gpu_dagger_smoke.leo.v1"
    config.pop("motion_assignments")
    with pytest.raises(ValueError, match="environment batch"):
        validate_config(config)


def test_missing_motion_assignment_or_test_seed_overlap_rejected(staged):
    args, _, config, plan = staged
    broken = copy.deepcopy(plan)
    broken["rounds"][2].pop("training_assignments")
    with pytest.raises(ValueError, match="fields"):
        validate_plan(broken, config)
    with pytest.raises(ValueError, match="seeds overlap"):
        prepare(**{**args, "root": Path(args["root"]).parent / "overlap", "seed": 9000})
    with pytest.raises(ValueError):
        validate_config(None)


def test_b4_native_driver_and_full_window_completion_counts(staged, tmp_path):
    from test.pipeline.test_dagger_mse_actor import FakeMSECoordinator
    from pipeline.dagger.rtx.code.src.dagger_rtx.mse_round_driver import MseBetaRoundDriver
    from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import RolloutLifecycle
    from pipeline.dagger.mse_campaign import validate_mse_actor_completion
    _, _, config, _ = staged
    config["h200"]["gateway_pids"] = [1, 2, 3, 4]
    protocol = {"schema": "roborender.dagger-mse-round-protocol.v1", "run_id": config["run_id"], "round_id": 1,
        "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
        "validation_manifest_path": "/validation/real.json", "validation_manifest_sha256": "a" * 64,
        "interval_iterations": 1, "max_iterations": 3, "min_validation_epochs": 1, "patience": 1,
        "relative_improvement": .005, "absolute_improvement": 1e-6,
        "final_test": {"scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
                       "scenario_ids": ["test-0", "test-1"], "observation_mode": "generated"}}
    sha = hashlib.sha256(canonical(protocol)).hexdigest()
    control = FakeMSECoordinator(config, protocol)
    runner = SimpleNamespace(num_envs=2, actor_prefix=config["run_id"] + "-g0", phase_telemetry=None,
        lifecycle=RolloutLifecycle(Mock(), Mock()),
        run_window=Mock(return_value={"transition_batches": 81, "physical_transition_rows": 162}))
    driver = MseBetaRoundDriver(runner, control, tmp_path / "driver", config,
        hashlib.sha256(canonical(config)).hexdigest(), 0,
        mse_protocol=protocol, mse_protocol_file_sha256=sha, sleep=lambda _: None)
    first = driver.run()
    second = {**first, "physical_gpu": 1}
    receipt = {"actors": [first, second], "actor_ids": actor_ids(config), "owned_actors_stopped": True}
    result = validate_mse_actor_completion(config, receipt, control.rows, control.context()["mse_round"], sha)
    assert result["completed_iterations"] == 2
    assert first["physical_transition_rows"] == 324  # 2 windows ×81×2 per physical owner
    assert runner.run_window.call_count == 2
    assert control.posts == 4  # 2 per-owner lane barriers ×2 windows


def test_b4_campaign_does_not_import_legacy_five_motion_result(staged, tmp_path):
    from pipeline.dagger.mse_campaign import MSECampaign
    from pipeline.dagger.beta_campaign import Campaign
    _, _, config, plan = staged
    with pytest.raises(ValueError, match="requires MSECampaign"):
        Campaign(config, tmp_path / "wrong", Mock())
    campaign = MSECampaign(config, tmp_path / "campaign", Mock(), plan=plan)
    outcome = tmp_path / "explicit-result.json"
    outcome.write_bytes(canonical({"fixture": "not actual native evidence"}))
    campaign.save("EVALUATE", final_checkpoint={"sha256": "f" * 64, "path": "/selected.pt"})
    campaign.operation = Mock(return_value={"comparison_path": str(outcome), "comparison_sha256": file_sha256(outcome),
                                          "owned_actors_stopped": True})
    with pytest.raises(ValueError, match="corrected reset/panel identity"):
        campaign.tick()
    assert campaign.state["phase"] == "EVALUATE"
    with patch("pipeline.deployment.rtx_actor.smoke_success_contract.validate_smoke_result") as admit:
        assert campaign.tick()["phase"] == "IMPORT_VALIDATION"
    admit.assert_called_once()
    assert admit.call_args.args[2] == "f" * 64
    assert admit.call_args.kwargs["rendered_config"]["checkpoint_sha256"] == "f" * 64


def test_pinned_recovered_population_requires_admission_and_never_recollects(staged, tmp_path):
    from pipeline.dagger.mse_campaign import MSECampaign
    _, _, config, plan = staged
    plan["rounds"][0].update(validation_manifest_sha256="a"*64, validation_admission_sha256="b"*64)
    campaign = MSECampaign(config,tmp_path/"recovery",Mock(),plan=plan)
    campaign.admit_population=Mock(return_value={})
    with patch("pipeline.dagger.beta_campaign.Campaign.tick",lambda owner: owner.save("REGISTER",operation_id=None)):
        assert campaign.tick()["phase"]=="REGISTER"
    campaign.admit_population.assert_called_once()
    assert campaign.state["heldout_population_reused"] is True
    campaign.save("PREPARE")
    campaign.admit_population.side_effect=ValueError("parent or actual seed differs")
    with patch("pipeline.dagger.beta_campaign.Campaign.tick",lambda owner: owner.save("REGISTER",operation_id=None)):
        with pytest.raises(ValueError,match="parent or actual seed"):
            campaign.tick()
    plan["rounds"][0]["validation_admission_sha256"]=None
    with pytest.raises(ValueError,match="both hashes"):
        validate_plan(plan,config)
