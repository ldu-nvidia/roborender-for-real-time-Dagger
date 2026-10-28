import copy
import json

import pytest

from pipeline.deployment.plan import DEFAULT_PROFILE, load_profile, make_plan


def test_selected_profile_binds_real_bounded_evidence():
    profile, candidate = load_profile(DEFAULT_PROFILE)
    plan = make_plan(profile, candidate, "qualification")
    assert plan["rtx_gpu_indices"] == [1, 2]
    assert plan["environments_per_rtx"] == 6
    assert plan["total_environments"] == 12
    assert plan["rollout"]["total_transitions_per_window"] == 12 * 81
    assert plan["learner"]["microbatch_environments"] == 12
    assert plan["requested_qualification"]["windows"] == 3
    assert plan["max_qualified_windows"] == 3
    assert not plan.get("blocked", False)
    assert plan["host_memory_limit_gib_per_actor"] == 21
    assert plan["compact_host_heap"] is True
    assert plan["actor_environment"]["MALLOC_ARENA_MAX"] == "2"
    assert candidate["rtx"]["environments_per_actor"] == 6  # Historical evidence unchanged.
    assert plan["h200"]["render_student_workers"] == 4
    assert not plan["launch_performed"] and not plan["allocation_performed"]
    assert not plan["unattended_production_allowed"]


def test_bc_plan_does_not_allocate_rtx_or_four_h200s():
    profile, candidate = load_profile(DEFAULT_PROFILE)
    plan = make_plan(profile, candidate, "bc")
    assert plan["rtx_gpu_indices"] == []
    assert plan["h200_gpu_count"] == 1
    assert plan["batch_chunks"] == 16
    assert plan["requires_new_collection_or_render"] is False
    assert plan["checkpoint"]["epoch"] == 41


@pytest.mark.parametrize("stage", ["dagger", "evaluation"])
def test_no_unqualified_production_or_success_claim(stage):
    profile, candidate = load_profile(DEFAULT_PROFILE)
    profile = copy.deepcopy(profile)
    profile["qualification"]["beta_success_evaluation"] = False
    profile["success_evaluation"]["status"] = "running"
    plan = make_plan(profile, candidate, stage)
    assert plan["blocked"] is True
    assert "beta_success_evaluation" in plan["production_acceptance_pending"]
    assert "live_wandb" not in plan["production_acceptance_pending"]
    assert "four_worker_production_controller" in plan["production_acceptance_pending"]


def test_retired_success_evaluation_exposes_history_not_executable_topology():
    profile, candidate = load_profile(DEFAULT_PROFILE)
    plan = make_plan(profile, candidate, "evaluation")
    assert plan["blocked"] is True and plan["unattended_production_allowed"] is False
    assert plan["launch_performed"] is plan["allocation_performed"] is False
    assert "retired" in plan["reason"] and "rendered_observation_config" in plan["reason"]
    historical = plan["historical_evaluation"]
    assert historical == profile["success_evaluation"]
    assert historical["physical_gpu_index"] == 1 and historical["native_batch_size"] == 5
    assert historical["completed_episodes_per_checkpoint"] == 256
    assert historical["optimizer_steps"] == 0 and historical["parallel_arms"] is False
    assert historical["observation_mode"] == "raytraced_3view"
    assert not {'rtx_gpu_indices', 'rtx_gpu_uuids', 'h200_gpu_count', 'learner',
                'h200_pool_environment', 'actor_entrypoint', 'native_batch_size'} & plan.keys()


def test_completed_success_qualification_does_not_unblock_production_training():
    profile, candidate = load_profile(DEFAULT_PROFILE)
    profile = copy.deepcopy(profile)
    profile["success_evaluation"]["status"] = "passed"
    profile["qualification"]["beta_success_evaluation"] = True
    assert make_plan(profile, candidate, "evaluation")["blocked"]
    assert make_plan(profile, candidate, "dagger")["blocked"]


@pytest.mark.parametrize('status,qualified', [('passed', False), ('running', True)])
def test_inconsistent_historical_qualification_cannot_unretire_evaluation(status, qualified):
    profile, candidate = load_profile(DEFAULT_PROFILE)
    profile = copy.deepcopy(profile)
    profile["success_evaluation"]["status"] = status
    profile["qualification"]["beta_success_evaluation"] = qualified
    before = copy.deepcopy(profile)
    plan = make_plan(profile, candidate, "evaluation")
    assert plan['blocked'] is True and plan['unattended_production_allowed'] is False
    assert 'actor_entrypoint' not in plan
    assert profile == before and plan['historical_evaluation'] == before['success_evaluation']


def test_retired_evaluation_cli_exits_blocked(monkeypatch, capsys):
    from pipeline.deployment import plan
    monkeypatch.setattr('sys.argv', ['plan', '--stage', 'evaluation'])
    with pytest.raises(SystemExit) as exc:
        plan.main()
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result['blocked'] is True and result['launch_performed'] is False
    assert result['allocation_performed'] is False and 'actor_entrypoint' not in result


def test_old_failed_memory_profile_remains_blocked():
    profile, candidate = load_profile(DEFAULT_PROFILE)
    profile = copy.deepcopy(profile)
    profile["runtime_qualification"].update(status="failed_container_oom", environments_per_actor=4,
        host_memory_limit_gib_per_actor=23, compact_host_heap=False, reason="historical RAM failure")
    plan = make_plan(profile, candidate, "qualification")
    assert plan["blocked"] and plan["max_qualified_windows"] == 0
    assert plan["total_environments"] == 8
    assert plan["host_memory_limit_gib_per_actor"] == 23
    assert plan["actor_environment"] == {}


@pytest.mark.parametrize("field,value", [
    ("tested_candidate_sha256", "0" * 64),
    ("tested_candidate", "../outside.json"),
    ("rtx_gpu_uuids", {"0": "forbidden", "1": "x"}),
    ("qualification", {"fake_gate": "true"}),
])
def test_reject_profile_drift(tmp_path, field, value):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profile(path)
