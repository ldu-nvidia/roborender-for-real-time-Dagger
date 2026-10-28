"""CPU-only exact-panel admission and corrected student-result import tests."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(ACTOR_ROOT))
import production_success_contract as contract
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import canonical, panel_sha256, summarize
from test.deployment.rtx_actor.test_explicit_motion_panel import panel, result


@pytest.fixture
def accepted(monkeypatch):
    # Synthetic bytes exercise the gates without depending on runtime data.
    # Production constants are never changed outside this fixture.
    value = panel()
    template = deepcopy(value["cohorts"][0])
    value["cohorts"] = []
    value["episode_count"] = 295
    value['panel_id'] = contract.LEGACY_PANEL_ID
    for index in range(148):
        cohort = deepcopy(template)
        cohort["assignment"]["cohort_id"] = f"cohort-{index}"
        cohort['assignment']['run_id'] = contract.LEGACY_PANEL_ID
        cohort["assignment"]["simulator_seed"] = 1000 + index
        count = 1 if index == 147 else 2
        cohort["assignment"]["lanes"] = cohort["assignment"]["lanes"][:count]
        cohort["case_ids"] = [f"case-{index}-{lane}" for lane in range(count)]
        for lane in cohort["assignment"]["lanes"]:
            identifier = f"motion-{index}-{lane['lane']}"
            lane.update(motion_id=identifier, path=f"/motions/{identifier}.npz",
                        sha256=hashlib.sha256(identifier.encode()).hexdigest())
        value["cohorts"].append(cohort)
    monkeypatch.setattr(contract, 'LEGACY_PANEL_SHA256', panel_sha256(value))
    monkeypatch.setattr(contract, 'LEGACY_PANEL_RAW_SHA256', hashlib.sha256(canonical(value)).hexdigest())
    value.update(panel_id=contract.PANEL_ID, observation_mode='image3f_3view')
    monkeypatch.setattr(contract, "PANEL_SHA256", panel_sha256(value))
    monkeypatch.setattr(contract, "PANEL_RAW_SHA256", hashlib.sha256(canonical(value)).hexdigest())
    return value


def student_result(accepted, arm="bc"):
    from test.deployment.rtx_actor.test_rendered_student_success_contract import complete_result
    return complete_result(accepted, arm)


def test_bc_and_four_dagger_stages_share_one_protocol(accepted):
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    identity = contract.protocol_identity(accepted)
    for arm in ("bc", "dagger", "dagger", "dagger", "dagger"):
        value = student_result(accepted, arm)
        assert contract.validate_production_result(accepted, value, "b" * 64,
            rendered_config=rendering_config(accepted)) == value
        assert value["production_success_protocol"] == identity
    assert identity["teacher_action_probability"] == identity["optimizer_steps"] == 0


@pytest.mark.parametrize("mutation", [
    lambda p: p["cohorts"][0]["assignment"].update(simulator_seed=17),
    lambda p: p["cohorts"][0]["case_ids"].__setitem__(0, "replacement-case"),
    lambda p: p["cohorts"][0]["assignment"]["lanes"][0].update(sha256="e" * 64),
    lambda p: p.update(episode_count=294),
    lambda p: p["cohorts"].reverse(),
])
def test_changed_seed_case_motion_denominator_or_order_fails(accepted, mutation):
    altered = deepcopy(accepted)
    mutation(altered)
    with pytest.raises(ValueError):
        contract.evaluation_options(altered)


def test_wrong_raw_bytes_and_unpinned_small_panel_fail(accepted):
    with pytest.raises(ValueError):
        contract.evaluation_options(accepted, raw_sha256="a" * 64)
    with pytest.raises(ValueError):
        contract.evaluation_options(panel())


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("production_success_protocol"),
    lambda r: r.update(diagnostic_initial_robot_refresh="raven_original"),
    lambda r: r.update(arm="teacher"),
    lambda r: r.update(completed_episodes=294),
    lambda r: r.update(optimizer_steps=1),
    lambda r: r.update(policy_unchanged=False),
])
def test_original_teacher_incomplete_or_optimized_results_cannot_advance(accepted, mutation):
    value = student_result(accepted)
    mutation(value)
    with pytest.raises(ValueError):
        contract.validate_production_result(accepted, value, "b" * 64)


def test_actual_native_timing_is_checked_without_modification():
    env = SimpleNamespace(max_episode_length_s=8., dt=.02, sim_dt=.005, max_episode_length=400)
    before = vars(env).copy()
    assert contract.require_native_timing(env)["max_episode_seconds"] == 8
    assert vars(env) == before
    env.max_episode_length_s = 12
    with pytest.raises(ValueError, match="timing differs"):
        contract.require_native_timing(env)


def test_legacy_raytraced_receipts_are_read_only_not_relabelled_or_executable(accepted):
    from rendered_student_success_contract import raytraced_shape
    legacy = raytraced_shape(accepted)
    rows = []
    for spec in legacy['cohorts']:
        count = len(spec['case_ids'])
        row = result(spec, successes=[True] * count)
        row.update(completed=count, expected=count, original_raven_metric_completed=count)
        env = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count, dtype=torch.bool))
        row.update({refresh.FIELD: refresh.STUDENT_CLEAR_ONCE,
            'initial_robot_refresh_control': refresh.control_contract(refresh.STUDENT_CLEAR_ONCE),
            'initial_robot_refresh_attestation': refresh.clear_already_applied_initial_refresh(env,
                mode=refresh.STUDENT_CLEAR_ONCE, arm='bc', teacher_action_mode='distill')})
        rows.append(row)
    value = summarize(legacy, rows)
    identity = contract.protocol_identity(accepted)
    for key in ('sha256', 'raytraced_fallback_allowed', 'legacy_case_assignment_panel_sha256'):
        identity.pop(key)
    identity.update(schema='roborender.production-student-success-protocol.v1',
        panel_raw_sha256=contract.LEGACY_PANEL_RAW_SHA256,
        panel_sha256=contract.LEGACY_PANEL_SHA256, observation_mode='raytraced_3view')
    identity['sha256'] = hashlib.sha256(canonical(identity)).hexdigest()
    value['production_success_protocol'] = identity
    assert contract.validate_legacy_production_result(legacy, value, 'b' * 64) == value
    assert value['observation_mode'] == 'raytraced_3view'
    with pytest.raises(ValueError): contract.evaluation_options(legacy)
    with pytest.raises(ValueError): contract.validate_production_result(legacy, value, 'b' * 64)
