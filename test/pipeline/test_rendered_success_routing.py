"""CPU routing tests; synthetic receipts are not native qualification."""
from test.deployment.source_paths import REPO_ROOT
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.bc_training.handoff_common import binding, bound, canonical, publish
from pipeline.deployment import bc_success_dispatch as bc, resident_campaign_site as resident
from pipeline.deployment.rendered_success_dispatch import artifact, configuration
from pipeline.deployment.rtx_actor import production_success_contract as production
from pipeline.deployment.rtx_actor.rendered_training_support_contract import ADAPTER_SHA256
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_MSE_PRODUCTION_SCHEMA
from test.pipeline.test_production_resident_pool import pool

ROOT=REPO_ROOT


@pytest.fixture
def rendered_selection(tmp_path):
    legacy=json.loads((ROOT/'runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json').read_bytes())
    panel=production.build_rendered_panel(legacy)
    panel_ref=publish(tmp_path/'rendered-panel.json',panel)
    identity=dict(renderer='Image3F',adapter_sha256=ADAPTER_SHA256,engine_sha256='e'*64,
        steps=5,cfg=1.,teacache_threshold=.3,seed=42,server_rng_isolated=True)
    template=dict(renderer=dict(host='127.0.0.1',port=18550,identity=identity),
        anchors=dict(seed=20261024,files=[dict(path=f'/assets/anchors/{i:04d}.npz',
            sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(2048)]))
    ref=publish(tmp_path/'dependencies.json',template)
    checkpoint=tmp_path/'selected.pt';checkpoint.write_bytes(b'synthetic CPU routing fixture')
    return dict(schema='roborender.bc-production-selected.v1',run_name='new-rendered-BC',
        selection='minimum_complete_validation_mse',success_test_executed=False,expected_test_episodes=295,
        test_panel=panel_ref,rendered_observation_config=ref,
        checkpoint=dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),bytes=checkpoint.stat().st_size))


def test_new_bc_command_binds_renderer_panel_and_checkpoint(rendered_selection,pool):
    plan=bc.command_plan(rendered_selection,pool)
    command=plan['command'];payload=command['payload']
    assert len(command['command_id'])<=96
    assert 'image3f' in command['command_id']
    assert payload['rendered_observation_config_sha256']==plan['rendered_config_raw_sha256']
    assert plan['rendered_config']['checkpoint_sha256']==rendered_selection['checkpoint']['sha256']
    assert plan['rendered_config']['panel_sha256']==production.PANEL_SHA256
    assert payload['diagnostic_initial_robot_refresh']=='student_clear_already_applied_once_v1'
    assert len(plan['rendered_config']['anchors']['files'])==2048
    changed=deepcopy(rendered_selection)
    changed['checkpoint']['sha256']='f'*64
    assert bc.command_plan(changed,pool)['command']['command_id']!=command['command_id']


@pytest.mark.parametrize('fault',['missing','legacy','template-bytes','renderer','anchor'])
def test_new_bc_cannot_fall_back_or_use_stale_dependencies(rendered_selection,pool,tmp_path,fault):
    value=deepcopy(rendered_selection)
    if fault=='missing':value.pop('rendered_observation_config')
    elif fault=='legacy':
        value['test_panel']=binding(ROOT/'runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json')
    elif fault=='template-bytes':value['rendered_observation_config']['raw_sha256']='0'*64
    else:
        template=bound(value['rendered_observation_config'])
        if fault=='renderer':template['renderer']['identity']['renderer']='fake'
        else:template['anchors']['files'].pop()
        value['rendered_observation_config']=publish(tmp_path/'bad-template.json',template)
    with pytest.raises(ValueError):bc.command_plan(value,pool)


def test_bc_dispatch_stages_actual_configuration_before_submit(rendered_selection,pool,tmp_path,monkeypatch):
    calls=[]
    class Transport:
        def stage(self,source,destination,sha):
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==sha
            calls.append(('stage',str(destination)))
        def command(self,argv):
            action=argv[argv.index('--action')+1];calls.append((action,None))
            if action=='status':return canonical(dict(workers=[dict(worker_instance_id=w['worker_instance_id'],source_sha256=pool['source_sha256']) for w in pool['workers']]))
            if action=='submit':return canonical(dict(results=[dict(status='accepted')]))
            return canonical(dict(results=[dict(status='COMPLETE',receipt=dict(complete=True,optimizer_steps=0,native_initializations=1,
                resource_guard=dict(passed=True,samples=2,minimum_host_available_bytes=12<<30,swap_peak_bytes=0,oom_events=0,limit_pressure_events=0),
                result=dict(successes=0,success_rate=0)))]))
    verified=[]
    monkeypatch.setattr(bc,'validate_production_result',lambda panel,result,sha,**kw: verified.append(kw['rendered_config']) or result)
    monkeypatch.setattr(bc,'log_success',lambda *a,**k:dict(online_readback_verified=True))
    result=bc.execute(publish(tmp_path/'selection.json',rendered_selection),publish(tmp_path/'pool.json',pool),
        '/remote/pool.json','/remote/source',tmp_path/'dispatch',transport=Transport())
    assert result['status']=='BC295_COMPLETE'
    assert len([c for c in calls if c[0]=='stage'])==4
    submit=next(i for i,c in enumerate(calls) if c[0]=='submit')
    assert any('rendered-' in c[1] for c in calls[:submit] if c[0]=='stage')
    assert verified[0]['checkpoint_sha256']==rendered_selection['checkpoint']['sha256']


@pytest.mark.parametrize('round_id',[1,2,3,4])
def test_each_dagger_final_test_stages_config_and_rejects_old_cache(rendered_selection,tmp_path,monkeypatch,round_id):
    panel=bound(rendered_selection['test_panel'])
    site=object.__new__(resident.ResidentSite)
    site.config=dict(schema=LEO_MSE_PRODUCTION_SCHEMA,run_id='rendered-test',round_id=round_id,
        validation=dict(scenario_manifest=rendered_selection['test_panel']['path'],
            scenario_manifest_sha256=rendered_selection['test_panel']['raw_sha256'],
            completed_episodes=295,panel_protocol=panel['panel_id'],observation_mode='generated',
            rendered_observation_config=rendered_selection['rendered_observation_config']))
    site.request=dict(final_checkpoint=rendered_selection['checkpoint'])
    site.remote,site.jobs,site.round=tmp_path/'jobs/round',tmp_path/'jobs',tmp_path
    site.run,site.rid='rendered-test',round_id
    site.barrier,site.copy_checkpoint,site.publish=Mock(return_value={}),Mock(),Mock()
    site.pool_action=Mock(side_effect=lambda action,commands=None:dict(results=[dict(receipt=dict(
        complete=True,purpose='test',optimizer_steps=0,result={'synthetic':True}))]) if action=='wait' else {})
    seen=[]
    monkeypatch.setattr(production,'validate_production_result',lambda panel,result,sha,**kw: seen.append(kw['rendered_config']) or result)
    monkeypatch.setattr(resident,'file_sha256',lambda p:'a'*64)
    result=site.evaluate_explicit()
    submitted=next(c.args[1][0] for c in site.pool_action.call_args_list if c.args[0]=='submit')
    assert '-image3f-' in submitted['command_id']
    payload=submitted['payload']
    config_call=next(c for c in site.publish.call_args_list if 'rendered-success-' in str(c.args[0]))
    assert hashlib.sha256(config_call.args[1]).hexdigest()==payload['rendered_observation_config_sha256']
    assert json.loads(config_call.args[1])==seen[0]
    assert payload['arm']=='dagger' and payload['teacher_action_mode']=='distill'
    assert '-image3f-' in result['comparison_path']


def test_mode_recipe_and_command_require_actual_staged_renderer(rendered_selection):
    from pipeline.deployment.run_mode import recipe,test_command
    plan=recipe('production');plan['test_panel']=rendered_selection['test_panel']
    plan['settings']=dict(rendered_observation_config=rendered_selection['rendered_observation_config'])
    assert plan['test']['observation_mode']=='image3f_3view'
    args=dict(stage_name='bc',checkpoint_sha256=rendered_selection['checkpoint']['sha256'],
        panel_file='tests/panel.json',output_name='cpu-rendered-test')
    with pytest.raises(ValueError,match='staged'):test_command(plan,**args)
    command=test_command(plan,**args,rendered_config_file='tests/config.json')
    assert command['payload']['rendered_observation_config_file']=='tests/config.json'


def test_legacy_cpu_site_rejects_before_copy_publish_or_gpu():
    from pipeline.deployment.beta_site import Site
    site=object.__new__(Site)
    site.copy_checkpoint=Mock(side_effect=AssertionError('must not copy'))
    site.publish=Mock(side_effect=AssertionError('must not publish'))
    with pytest.raises(RuntimeError,match='Legacy ray-traced'):
        site.evaluate()
    site.copy_checkpoint.assert_not_called();site.publish.assert_not_called()


def test_legacy_evaluation_plan_is_always_nonexecuting():
    from pipeline.deployment.plan import load_profile,make_plan,DEFAULT_PROFILE
    profile,candidate=load_profile(DEFAULT_PROFILE)
    result=make_plan(profile,candidate,'evaluation')
    assert result['blocked'] is True and result['unattended_production_allowed'] is False
    assert 'rendered_observation_config' in result['reason']
    assert 'actor_entrypoint' not in result


@pytest.mark.parametrize('name',['run_small_batch_success','run_paired_student_success','run_success_diagnostic'])
def test_retired_student_cli_cannot_launch(name,monkeypatch):
    import importlib
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'pipeline/deployment/rtx_actor'))
    module=importlib.import_module(name)
    gate=Mock(side_effect=AssertionError('must not probe GPUs'))
    monkeypatch.setattr(module,'gpu_gate',gate)
    argv=[name,'--run-id','old-ray-smoke','--source-root','/unused',
        '--bc-checkpoint','/unused/bc.pt','--bc-sha256','b'*64,
        '--dagger-checkpoint','/unused/dagger.pt','--dagger-sha256','d'*64]
    if name=='run_small_batch_success':argv+=['--panel','/unused/panel.json','--panel-sha256','a'*64]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(RuntimeError,match='retired'):
        module.main()
    gate.assert_not_called()
