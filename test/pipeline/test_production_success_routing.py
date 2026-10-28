"""CPU routing/evidence tests; synthetic outcomes do not claim native success."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.bc_training.handoff_common import binding, bound, canonical, digest, publish
from pipeline.dagger import coverage_queue, mse_campaign
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_MSE_PRODUCTION_SCHEMA, LEO_MSE_SMOKE_SCHEMA
from pipeline.deployment import bc_success_dispatch as bc, resident_campaign_site as resident
from pipeline.deployment.rtx_actor.explicit_motion_panel import panel_sha256
from pipeline.deployment.rtx_actor.initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE
from pipeline.deployment.rtx_actor.production_success_contract import PANEL_RAW_SHA256, protocol_identity
from test.pipeline.test_production_resident_pool import selected, pool


def campaign_config(selection, *, round_id=1):
    panel = bound(selection['test_panel'])
    return dict(schema=LEO_MSE_PRODUCTION_SCHEMA, run_id='cpu-routing-fixture', round_id=round_id,
        validation=dict(scenario_manifest=selection['test_panel']['path'],
            scenario_manifest_sha256=selection['test_panel']['raw_sha256'],
            completed_episodes=295, panel_protocol=panel['panel_id'], observation_mode='generated',
            rendered_observation_config=selection['rendered_observation_config']),
        logging=dict(wandb_run_id='cpu-routing-fixture'))


def historical_result(result):
    result = deepcopy(result)
    for key in ('production_success_protocol', 'diagnostic_initial_robot_refresh',
                'initial_robot_refresh_control', 'initial_robot_refresh_attestations'):
        result.pop(key, None)
    return result


def report_for(selection, result):
    return dict(purpose='test', status='complete', completed=295, successes=result['successes'],
        goal_reached_rate=result['success_rate'], checkpoint_sha256=selection['checkpoint']['sha256'],
        scenario_manifest_sha256=selection['test_panel']['raw_sha256'],
        success_test_protocol=protocol_identity(bound(selection['test_panel'])),
        rendered_student_success_protocol=result['rendered_student_success_protocol'])


def test_bc_command_pins_panel_and_student_reset(selected, pool):
    selection, _ = selected
    plan = bc.command_plan(selection, pool)
    assert plan == bc.command_plan(selection, pool)
    payload = plan['command']['payload']
    assert payload['panel_sha256'] == PANEL_RAW_SHA256
    assert payload['teacher_action_mode'] == 'distill'
    assert payload['diagnostic_initial_robot_refresh'] == STUDENT_CLEAR_ONCE
    assert payload['arm'] == 'bc' and 'resume_file' not in payload
    suffix = plan['protocol_suffix']
    assert plan['command']['command_id'].endswith(suffix)
    assert payload['output_name'] == plan['command']['command_id']
    assert suffix in plan['command_destination'] and suffix in plan['panel_destination']
    assert plan['command']['command_id'] != 'bc-prod-1-full295-' + selection['checkpoint']['sha256'][:16]


def test_fresh_bc_campaign_does_not_reuse_historical_test_or_logging(selected, pool):
    selection, _ = selected
    original = bc.command_plan(selection, pool)
    original_identity = bc.campaign_identity(selection)
    selection['run_name'] = 'prod-10-21-clean-reset-20261021'
    fresh = bc.command_plan(selection, pool)
    assert fresh['command']['command_id'] != original['command']['command_id']
    assert fresh['panel_destination'] != original['panel_destination']
    assert bc.campaign_identity(selection)[1] != original_identity[1]
    assert fresh['command']['payload']['diagnostic_initial_robot_refresh'] == STUDENT_CLEAR_ONCE
    assert fresh['command']['payload']['panel_sha256'] == PANEL_RAW_SHA256


@pytest.mark.parametrize('name', ['', '../old', 'bad name', None, 'x'*161])
def test_bc_rejects_unsafe_campaign_name(selected, pool, name):
    selection, _ = selected
    selection['run_name'] = name
    with pytest.raises(ValueError):
        bc.command_plan(selection, pool)


def test_resident_public_entrypoint_routes_production_to_explicit_test(selected):
    selection, _ = selected
    site = object.__new__(resident.ResidentSite)
    site.config = campaign_config(selection)
    site.evaluate_explicit = Mock(return_value={'complete': 'cpu-routing-only'})
    assert site.evaluate() == {'complete': 'cpu-routing-only'}
    site.evaluate_explicit.assert_called_once_with()


@pytest.mark.parametrize('fault', ['bytes', 'seed', 'denominator'])
def test_bc_rejects_rebound_or_reduced_panel(selected, pool, tmp_path, fault):
    selection, _ = selected
    panel = bound(selection['test_panel'])
    if fault == 'seed':
        panel['cohorts'][0]['assignment']['simulator_seed'] += 1_000_000
    elif fault == 'denominator':
        panel['cohorts'] = panel['cohorts'][:1]
        panel['episode_count'] = 2
    # Even a semantically identical panel with different serialization is not
    # the accepted raw manifest to be submitted to the resident executor.
    path = tmp_path / 'substitute.json'
    path.write_text(json.dumps(panel, indent=2))
    selection['test_panel'] = dict(path=str(path), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        bc.command_plan(selection, pool)


@pytest.mark.parametrize('round_id', [1, 2, 3, 4])
@pytest.mark.parametrize('run_id', ['cpu-routing-fixture', 'production-' + 'long-campaign-' * 6])
def test_each_dagger_round_submits_same_corrected_student_protocol(selected, tmp_path, monkeypatch, round_id, run_id):
    selection, result = selected
    result['arm'] = 'dagger'
    site = object.__new__(resident.ResidentSite)
    site.config = campaign_config(selection, round_id=round_id)
    site.config['run_id'] = run_id
    site.request = dict(final_checkpoint=selection['checkpoint'])
    site.remote, site.jobs, site.round = tmp_path / 'jobs/round', tmp_path / 'jobs', tmp_path
    site.run, site.rid = site.config['run_id'], round_id
    site.barrier, site.copy_checkpoint, site.publish = Mock(return_value={}), Mock(), Mock()
    site.pool_action = Mock(side_effect=lambda action, commands=None: dict(results=[dict(receipt=dict(
        complete=True, purpose='test', optimizer_steps=0, result=result))]) if action == 'wait' else {})
    monkeypatch.setattr(resident, 'file_sha256', lambda path: 'a' * 64)
    outcome = site.evaluate_explicit()
    assert site.evaluate_explicit() == outcome
    submitted = next(call.args[1][0] for call in site.pool_action.call_args_list if call.args[0] == 'submit')
    assert [call.args[1][0] for call in site.pool_action.call_args_list if call.args[0] == 'submit'] == [submitted, submitted]
    assert submitted['payload']['panel_sha256'] == PANEL_RAW_SHA256
    assert submitted['payload']['arm'] == 'dagger'
    assert submitted['payload']['teacher_action_mode'] == 'distill'
    assert submitted['payload']['diagnostic_initial_robot_refresh'] == STUDENT_CLEAR_ONCE
    assert 'resume_file' not in submitted['payload']
    from pipeline.deployment.rendered_success_dispatch import artifact
    _,_,rendered_sha=artifact(selection['rendered_observation_config'],bound(selection['test_panel']),selection['checkpoint']['sha256'])
    suffix = '-image3f-' + rendered_sha[:12]
    assert submitted['command_id'] == f'mse-test-{hashlib.sha256(site.run.encode()).hexdigest()[:16]}-r{round_id}{suffix}'
    assert submitted['command_id'] == submitted['payload']['output_name']
    assert len(submitted['command_id']) <= 96
    from dagger_rtx.resident_service import SCHEMA as resident_schema, validate_command
    validate_command(dict(schema=resident_schema, worker_instance_id='a' * 32,
                          command_id=submitted['command_id'], operation=submitted['operation'], payload=submitted['payload']))
    assert submitted['payload']['output_name'].endswith(suffix)
    assert suffix in outcome['comparison_path']


def test_legacy_smaller_explicit_smoke_cannot_bypass_rendered_protocol(selected, tmp_path, monkeypatch):
    selection, result = selected
    panel = bound(selection['test_panel'])
    panel['cohorts'], panel['episode_count'] = panel['cohorts'][:1], 2
    selection['test_panel'] = publish(tmp_path / 'smoke.json', panel)
    config = campaign_config(selection)
    config['schema'], config['validation']['completed_episodes'] = LEO_MSE_SMOKE_SCHEMA, 2
    result = historical_result(result)
    result.update(arm='dagger', panel_sha256=panel_sha256(panel), completed_episodes=2, expected_episodes=2,
        runtime_physics_sha256_per_cohort=result['runtime_physics_sha256_per_cohort'][:1], outcomes=result['outcomes'][:2])
    with pytest.raises(ValueError):
        mse_campaign.validate_explicit_final_test(config, result, selection['checkpoint']['sha256'])
    site = object.__new__(resident.ResidentSite)
    site.config, site.request = config, dict(final_checkpoint=selection['checkpoint'])
    site.remote, site.jobs, site.round = tmp_path / 'jobs/round', tmp_path / 'jobs', tmp_path
    site.run, site.rid = config['run_id'], 1
    site.barrier, site.copy_checkpoint, site.publish = Mock(return_value={}), Mock(), Mock()
    site.pool_action = Mock(side_effect=lambda action, commands=None: dict(results=[dict(receipt=dict(
        complete=True, purpose='test', optimizer_steps=0, result=result))]) if action == 'wait' else {})
    monkeypatch.setattr(resident, 'file_sha256', lambda path: 'a' * 64)
    with pytest.raises(ValueError):site.evaluate_explicit()
    site.pool_action.assert_not_called()
    config['schema'] = LEO_MSE_PRODUCTION_SCHEMA
    with pytest.raises(ValueError):
        mse_campaign.validate_explicit_final_test(config, result, selection['checkpoint']['sha256'])


def test_old_result_cannot_register_or_import_production_outcomes(selected):
    selection, result = selected
    result = historical_result(result)
    result['arm'] = 'dagger'
    controller = SimpleNamespace(config=campaign_config(selection), run_id='cpu-routing-fixture',
        round_id=1, mse=Mock(), evaluation=Mock())
    with pytest.raises(ValueError, match='protocol identity'):
        mse_campaign.import_final_test(controller, result, 'b' * 64, selection['checkpoint']['sha256'])
    controller.mse.final_test.assert_not_called()
    controller.evaluation.post.assert_not_called()


def test_corrected_import_preserves_protocol_metadata(selected):
    selection, result = selected
    result['arm'] = 'dagger'
    for outcome in result['outcomes']:
        outcome['episode_reward'] = 0.  # Explicit synthetic native reward for ledger import.
    report = report_for(selection, result)
    report.pop('success_test_protocol')
    controller = SimpleNamespace(config=campaign_config(selection), run_id='cpu-routing-fixture',
        round_id=1, mse=Mock(), evaluation=Mock())
    controller.evaluation.get.return_value = {'evaluation': report}
    controller.evaluation.post.side_effect = lambda route, payload: dict(ok=True,
        evaluation_id='cpu-routing-fixture-r1-final-test', scenario_id=payload['scenario_id'])
    imported = mse_campaign.import_final_test(controller, result, 'b' * 64, selection['checkpoint']['sha256'])
    assert imported['success_test_protocol'] == result['production_success_protocol']
    assert controller.evaluation.post.call_count == 295


@pytest.mark.parametrize('phase', ['LOG_VALIDATION', 'ADVANCE'])
def test_resumed_production_cannot_log_or_advance_historical_result(selected, tmp_path, phase):
    selection, result = selected
    result['arm'] = 'dagger'
    path = tmp_path / 'old-test.json'
    path.write_bytes(canonical(historical_result(result)))
    campaign = object.__new__(mse_campaign.MSECampaign)
    campaign.state = dict(phase=phase, config=campaign_config(selection),
        final_checkpoint=selection['checkpoint'], evaluation_report=report_for(selection, result),
        evaluation_receipt=dict(comparison_path=str(path), comparison_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    campaign.log_validation, campaign.get_controller, campaign.save = Mock(), Mock(), Mock()
    with pytest.raises(ValueError, match='protocol identity'):
        campaign.tick()
    campaign.log_validation.assert_not_called()
    campaign.get_controller.assert_not_called()
    campaign.save.assert_not_called()


def test_bc_old_result_rejected_before_online_logging(selected, tmp_path, monkeypatch):
    selection, result = selected
    admission = Mock()
    monkeypatch.setattr(bc, 'admit_logging', admission)
    with pytest.raises(ValueError, match='protocol identity'):
        bc.log_success(historical_result(result), selection, tmp_path, wandb_module=Mock())
    admission.assert_not_called()


def test_bc_corrected_online_identity_is_stable_and_separate(selected, tmp_path, monkeypatch):
    selection, result = selected
    monkeypatch.setattr(bc, 'admit_logging', Mock(return_value=(object(), {'authenticated': True})))
    monkeypatch.setattr(bc, 'verify_remote_summary', Mock())
    wandb = SimpleNamespace(init=Mock(return_value=Mock(url='https://example.test/cpu-only')))
    first = bc.log_success(result, selection, tmp_path, wandb_module=wandb)
    assert first == bc.log_success(result, selection, tmp_path, wandb_module=wandb)
    from pipeline.deployment.rendered_success_dispatch import artifact
    _,_,rendered_sha=artifact(selection['rendered_observation_config'],bound(selection['test_panel']),selection['checkpoint']['sha256'])
    suffix = bc.protocol_suffix(bound(selection['test_panel']),rendered_sha)
    assert first['run_id'] == '20261016-prod-1-bc-success-' + suffix
    assert first['name'] == '2026-10-16-prod-1-BC-success-' + suffix
    assert all(call.kwargs['id'] == first['run_id'] for call in wandb.init.call_args_list)


def bc_handoff_fixture(selected, tmp_path, *, historical=False):
    selection, result = deepcopy(selected)
    selection.update(optimizer_state_preserved=True, rng_state_preserved=True,
        selected_validation_mse=.2, selected_global_step=160)
    for name in ('training_state', 'training_config', 'data_admission', 'policy_spec'):
        selection[name] = publish(tmp_path / (name + '.json'), {'cpu_fixture': name})
    selection_ref = publish(tmp_path / 'bc-selection.json', selection)
    result_ref = publish(tmp_path / 'bc-success.json', historical_result(result) if historical else result)
    queue = dict(root=str(tmp_path / 'queue'), bc_selection_path=selection_ref['path'],
        bc_success_path=result_ref['path'], panel=selection['test_panel'],
        policy_spec_sha256=digest(canonical(bound(selection['policy_spec']))))
    return queue, selection


def test_bc_to_dagger_handoff_requires_corrected_result(selected, tmp_path):
    queue, selection = bc_handoff_fixture(selected, tmp_path)
    assert coverage_queue.validate_bc_handoff(queue) == selection


@pytest.mark.parametrize('cached', [False, True])
def test_bc_to_dagger_rejects_historical_result_even_with_cached_admission(selected, tmp_path, monkeypatch, cached):
    queue, selection = bc_handoff_fixture(selected, tmp_path, historical=True)
    if cached:
        publish(Path(queue['root']) / 'bc-handoff-admission.json', dict(schema='roborender.full295-bc-handoff.v1',
            selection=binding(queue['bc_selection_path']), actual_success_test=binding(queue['bc_success_path']),
            checkpoint_sha256=selection['checkpoint']['sha256'], optimizer_state_preserved=True))
    staging = Mock()
    # This fixture isolates the success-protocol gate, not queue topology.
    # Full source/parent/capacity admission is covered by test_dagger_scaled_queue.
    monkeypatch.setattr(coverage_queue, 'validate_queue', lambda value: value)
    monkeypatch.setattr(coverage_queue, 'stage', staging)
    with pytest.raises(ValueError, match='protocol identity'):
        coverage_queue.prepare_campaign(queue)
    staging.assert_not_called()


def construct_history_campaign(selected, tmp_path, monkeypatch, *, phase, historical=False, fault=None):
    """Exercise the real persisted-state constructor without training setup."""
    from pipeline.dagger import beta_campaign
    selection, result = selected
    first = campaign_config(selection)
    current = deepcopy(first)
    current['round_id'] = 4 if phase == 'COMPLETE' else 2 if phase == 'PREPARE_NEXT' else 1
    actual_phase = 'PREPARE' if phase == 'PREPARE_NEXT' else phase
    count = current['round_id'] if actual_phase == 'COMPLETE' else current['round_id'] - 1
    rows = []
    for rid in range(1, count + 1):
        report = report_for(selection, result)
        if historical: report.pop('success_test_protocol')
        if fault == 'count': report['completed'] = 294
        if fault == 'panel': report['scenario_manifest_sha256'] = 'f' * 64
        if fault == 'checkpoint': report['checkpoint_sha256'] = 'f' * 64
        rows.append(dict(round_id=rid, test=report, selected_checkpoint=selection['checkpoint'],
            selected_checkpoint_sha256=selection['checkpoint']['sha256']))
    plan = {'cpu_test_plan': True}
    publish(tmp_path / 'mse-plan.json', plan)
    state = dict(schema='roborender.beta-campaign-state.v1', template_sha256=digest(canonical(first)),
        run_id=first['run_id'], config=current, phase=actual_phase, round_id=current['round_id'],
        completed_rounds=rows, mse_plan_sha256=digest(canonical(plan)))
    # Campaign immutable_json uses canonical bytes without the publication newline.
    (tmp_path / 'mse-plan.json').write_bytes(canonical(plan))
    publish(tmp_path / 'state.json', state)
    monkeypatch.setattr(mse_campaign, 'validate_plan', lambda plan, config: plan)
    monkeypatch.setattr(beta_campaign, 'validate_config', lambda config: config)
    class ProductionHistoryCampaign(mse_campaign.MSECampaign):
        supports_full295_production = True
    return ProductionHistoryCampaign(first, tmp_path, Mock(), plan=plan)


@pytest.mark.parametrize('phase', ['PREPARE_NEXT', 'COMPLETE'])
def test_historical_completed_rounds_rejected_on_constructor_resume(selected, tmp_path, monkeypatch, phase):
    with pytest.raises(ValueError, match='separately reviewed production campaign'):
        construct_history_campaign(selected, tmp_path, monkeypatch, phase=phase, historical=True)


@pytest.mark.parametrize('phase', ['PREPARE', 'PREPARE_NEXT', 'COMPLETE'])
def test_fresh_or_corrected_completed_history_can_resume(selected, tmp_path, monkeypatch, phase):
    campaign = construct_history_campaign(selected, tmp_path, monkeypatch, phase=phase)
    campaign.validate_production_history()
    if phase == 'COMPLETE':
        assert campaign.tick()['phase'] == 'COMPLETE'


@pytest.mark.parametrize('fault', ['count', 'panel', 'checkpoint'])
def test_completed_corrected_history_rechecks_count_panel_and_checkpoint(selected, tmp_path, monkeypatch, fault):
    with pytest.raises(ValueError, match='separately reviewed production campaign'):
        construct_history_campaign(selected, tmp_path, monkeypatch, phase='COMPLETE', fault=fault)


def test_online_success_metadata_is_separate_from_frozen_config(selected, monkeypatch):
    selection, result = selected
    config = campaign_config(selection)
    original = deepcopy(config)
    report = report_for(selection, result)
    from pipeline.deployment import wandb_preflight
    monkeypatch.setattr(wandb_preflight, 'admit_logging', Mock(return_value=(object(), {'entity': 'test'})))
    monkeypatch.setattr(wandb_preflight, 'verify_remote_summary', Mock())
    run = Mock(url='https://example.test/cpu-only')
    wandb = SimpleNamespace(init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    _, log = mse_campaign.online_mse_loggers('test')
    log(config, dict(iteration_id=148, train_mse=.2), report)
    protocol = result['production_success_protocol']
    assert wandb.init.call_args.kwargs['config'] == {'success_test_protocol': protocol}
    assert run.log.call_args.args[0]['test/success_test_protocol'] == protocol
    assert config == original
    report.pop('success_test_protocol')
    wandb.init.reset_mock()
    with pytest.raises(ValueError, match='corrected accepted295'):
        log(config, dict(iteration_id=148, train_mse=.2), report)
    wandb.init.assert_not_called()
