"""Mode wiring tests are CPU evidence, not native end-to-end certification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training.handoff_common import binding, canonical, publish
from pipeline.deployment.run_mode import recipe, stage, bc_job, stage_dagger, test_command as make_test_command
from pipeline.deployment.rtx_actor import production_success_contract as production
from pipeline.deployment.rtx_actor import smoke_success_contract as smoke
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_corrected_smoke, validate_config
from test.deployment.rtx_actor.test_production_success_contract import accepted


@pytest.fixture
def settings(accepted, tmp_path, monkeypatch):
    raw = canonical(accepted)
    monkeypatch.setattr(production, 'PANEL_RAW_SHA256', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(production, 'PANEL_SHA256', hashlib.sha256(raw).hexdigest())
    panel = tmp_path/'accepted.json'; panel.write_bytes(raw)
    rows = [dict(motion_id=m['motion_id'], filename=Path(m['path']).name,
                 sha256=m['sha256'], bytes=123) for c in accepted['cohorts'] for m in c['assignment']['lanes']]
    inventory = dict(schema='roborender.motion-inventory.v1', motion_count=295,
        ordering='explicit_filename_lexical_not_filesystem_order', motions=rows,
        motion_records_sha256=hashlib.sha256(canonical(rows)).hexdigest())
    inv = tmp_path/'inventory.json'; inv.write_bytes(canonical(inventory))
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    dependencies=publish(tmp_path/'rendered-dependencies.json',rendering_config(accepted))
    return dict(run_id='smoke-mode-test', seed=241018, inventory=binding(inv), test_panel=binding(panel),
        dataset_root=str(tmp_path/'dataset'), asset_root=str(tmp_path/'assets'), source_root=str(tmp_path/'source'),
        source_sha256='c'*64, teacher_sha256='d'*64, policy_spec_sha256='e'*64,
        rendered_observation_config=dependencies)


def test_recipe_minimal_real_batches_not_short_fake_windows():
    p = recipe('smoke')
    assert (p['bc']['train_chunks'], p['bc']['validation_chunks'], p['bc']['batch_chunks']) == (16,2,16)
    assert p['bc']['max_epochs'] == 2
    assert p['dagger']['rollout_steps'] == 81
    assert p['dagger']['teacher_probabilities'] == [.75,.5,.25,0]
    assert p['dagger']['max_updates_per_round'] == 2
    assert p['dagger']['validation_interval_updates'] == 1
    assert p['execution']['auto_release_compute'] is False
    assert p['execution']['fake_components_allowed'] is False
    assert recipe('production')['test']['cases'] == 295
    assert recipe('production')['dagger']['patience'] == 5
    with pytest.raises(ValueError): recipe('fast')


def test_smoke_stages_real_collection_bc_and_all_dagger_rounds(settings, tmp_path):
    p = stage('smoke', settings, tmp_path/'bundle')
    assert stage('smoke', settings, tmp_path/'bundle') == p
    sampling = json.loads(Path(p['bc_population']['path']).read_bytes())
    assert sampling['total_target_chunks'] == 18
    assert set(sampling['partitions']) == {'train','validation'}
    assert sampling['selection_uses_labels_rewards_or_loss'] is False
    job = bc_job(p)
    assert job['qualification'] is True and job['env']['WANDB_MODE'] == 'online'
    argv = job['stages'][-1]['argv']
    assert argv[argv.index('--max-epochs')+1] == '2'
    assert argv[argv.index('--seed')+1] == str(settings['seed'])
    assert argv[argv.index('--training-protocol')+1] == 'legacy_validation'
    cp = tmp_path/'selected.pt'; cp.write_bytes(b'CPU fixture, not a model')
    staged = stage_dagger(p, selected_checkpoint=cp, checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
    config = json.loads(Path(staged['config_path']).read_bytes())
    rounds = json.loads(Path(staged['plan_path']).read_bytes())['rounds']
    assert is_corrected_smoke(config)
    assert validate_config(config) == config
    assert len(rounds) == 4
    for row in rounds:
        assert row['max_iterations'] == row['min_validation_epochs'] == 2
        assert row['interval_iterations'] == 1
        heldout = json.loads(Path(row['collection_plan_path']).read_bytes())
        assert sum(len(c['lanes']) for c in heldout['cohorts']) == 4
    commands = [make_test_command(p, stage_name=name, checkpoint_sha256='f'*64,
                panel_file='tests/panel.json', output_name='smoke-'+name,
                rendered_config_file='tests/rendered.json') for name in p['test']['after_stages']]
    assert len({c['payload']['panel_sha256'] for c in commands}) == 1
    assert all(c['payload']['diagnostic_initial_robot_refresh'] == smoke.SMOKE_STUDENT_CLEAR_ONCE for c in commands)
    assert [c['payload']['arm'] for c in commands] == ['bc']+['dagger']*4
    assert p['runtime_certified'] is p['launched'] is False


def test_fail_closed_wrong_mode_or_mutated_settings(settings, tmp_path):
    with pytest.raises(ValueError): stage('production', settings, tmp_path/'prod')
    cfg = {**settings, 'run_id':'prod-new'}
    with pytest.raises(ValueError, match='bc_production_protocol'):
        stage('production', cfg, tmp_path/'prod')
    p = stage('smoke', settings, tmp_path/'s')
    with pytest.raises((ValueError, RuntimeError)):
        stage('smoke', {**settings,'seed':123}, tmp_path/'s')
    with pytest.raises(FileNotFoundError):
        stage_dagger(p, selected_checkpoint=tmp_path/'missing.pt', checkpoint_sha256='a'*64)
    with pytest.raises(ValueError):
        make_test_command(p, stage_name='bc', checkpoint_sha256='a'*64, panel_file='../bad', output_name='smoke-bc')


def test_smoke_panel_cannot_be_used_as_production(settings, tmp_path):
    p = stage('smoke', settings, tmp_path/'s')
    panel = json.loads(Path(p['test_panel']['path']).read_bytes())
    smoke.require_panel_smoke(panel)
    with pytest.raises(ValueError): production.require_panel(panel)
    with pytest.raises(ValueError): smoke.require_panel_smoke(json.loads(Path(settings['test_panel']['path']).read_bytes()))


def test_offline_direct_mode_flag_uses_same_recipe(tmp_path):
    from pipeline.bc_training.offline_pipeline import plan
    job = plan(dataset_root=tmp_path/'d', asset_root=tmp_path/'a', runtime_root=tmp_path/'r',
               run_name='smoke-direct', mode='smoke')
    assert job['qualification']
    argv=job['stages'][-1]['argv']
    assert argv[argv.index('--max-epochs')+1] == '2'


def test_native_schedule_transfers_exact_requests_without_cpu_panel_path(settings,tmp_path):
    from pipeline.deployment.mode_site import stage_native_schedule
    from pipeline.deployment.rtx_actor.isaac6_bc_session import validate_session
    p=stage('smoke',settings,tmp_path/'bundle')
    sent={};remote=Path('/home/ldu/roborender/runtime/mode-runs/smoke-test/schedule')
    ref=stage_native_schedule(p,tmp_path/'compiled',remote,lambda path,raw:sent.setdefault(str(path),raw))
    raw=sent[ref['path']]
    assert hashlib.sha256(raw).hexdigest()==ref['raw_sha256']
    schedule=json.loads(raw)
    assert schedule['target_chunks']==18 and len(schedule['sessions'])==1
    for row in schedule['sessions']:
        raw=sent[row['session_path']]
        assert hashlib.sha256(raw).hexdigest()==row['session_sha256']
        validate_session(json.loads(raw),read_bytes=lambda path:sent[str(Path(row['requests_dir'])/Path(path).name)])
        assert Path(row['session_path']).is_relative_to(remote)
        assert Path(row['requests_dir']).is_relative_to(remote)
