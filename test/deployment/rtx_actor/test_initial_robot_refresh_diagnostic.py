"""CPU-only reset-intervention tests; no native import, step or compute API."""
from test.deployment.source_paths import ACTOR_ROOT
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE=ACTOR_ROOT
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
import initial_robot_refresh_diagnostic as refresh
from explicit_motion_panel import (
    aggregate_partitions, completed_progress, make_cohort_receipt, make_partition,
    run_panel, run_partition, summarize, validate_cohort_receipt, validate_result,
)
from test.deployment.rtx_actor.test_explicit_motion_panel import panel
from test.deployment.rtx_actor.test_teacher_action_diagnostic import ppo_result


def evidence(count=2):
    env=SimpleNamespace(num_envs=count,need_to_refresh_envs=torch.ones(count,dtype=torch.bool))
    return refresh.clear_already_applied_initial_refresh(env)


def result(spec):
    value=ppo_result(spec)
    value.update({refresh.FIELD:refresh.CLEAR_ONCE,
        'initial_robot_refresh_control':refresh.control_contract(),
        'initial_robot_refresh_attestation':evidence(len(spec['case_ids']))})
    return value


def run_kwargs():
    return dict(expected_checkpoint_sha256='b'*64,expected_arm='teacher',teacher_action_mode='raven_ppo',
                diagnostic_initial_robot_refresh=refresh.CLEAR_ONCE)


def panel_environment(events, original_obs):
    class Base:
        def reset_all(self):
            from initial_reset_contract import begin_initial_reset, complete_initial_reset
            begin_initial_reset(self)
            events.extend(['native_robot_writes','forward_without_physics','observations_constructed'])
            self.need_to_refresh_envs.fill_(True)
            complete_initial_reset(self)
            return deepcopy(original_obs)
    source=HERE/'isaac6_panel_evaluator.py'
    cls=deepcopy(next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PanelResetEnvironment'))
    def cleared(env, **kwargs):
        events.append('clear_pending_once')
        return refresh.clear_already_applied_initial_refresh(env, **kwargs)
    scope={'SuccessAccountingEnvironment':Base,'torch':torch,'os':os,
        'ORIGINAL_INITIAL_REFRESH':refresh.ORIGINAL,'CLEAR_ONCE':refresh.CLEAR_ONCE,
        'require_corrected_execution':refresh.require_corrected_execution,'clear_already_applied_initial_refresh':cleared}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),scope)
    env=scope['PanelResetEnvironment']()
    env.num_envs=2;env.need_to_refresh_envs=torch.ones(2,dtype=torch.bool)
    env.history_handler=SimpleNamespace(history={'homie':torch.zeros(2,10)})
    env.simulator=SimpleNamespace(scene=SimpleNamespace(reset=lambda:events.append('scene_reset')),
        event_manager=SimpleNamespace(_reset_term_last_triggered_step_id=[torch.ones(2,dtype=torch.int64)],
                                      _reset_term_last_triggered_once=[torch.ones(2,dtype=torch.bool)]))
    for name in ('_homie_commands','_homie_commands_unclipped','_last_homie_commands'):
        setattr(env,name,torch.arange(14,dtype=torch.float32).reshape(2,7))
    env.target_robot_root_states=torch.arange(26,dtype=torch.float32).reshape(2,13)
    env.target_robot_dof_state=torch.arange(172,dtype=torch.float32).reshape(2,43,2)
    return env


def test_missing_or_original_reset_is_rejected_before_scene_mutation(monkeypatch):
    monkeypatch.delenv('DIAGNOSTIC_INITIAL_ROBOT_REFRESH',raising=False)
    events=[];env=panel_environment(events,{'obs':torch.ones(2,3)})
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='retired'):env.reset_panel()
    monkeypatch.setenv('DIAGNOSTIC_INITIAL_ROBOT_REFRESH',refresh.ORIGINAL)
    with pytest.raises(ValueError,match='retired'):env.reset_panel()
    assert env.need_to_refresh_envs.all()
    assert events==[]
    assert not hasattr(env,'_diagnostic_initial_refresh_attestation')
    assert torch.equal(torch.get_rng_state(),before)


def test_training_boundary_consumes_initial_pending_without_calling_success_reset(monkeypatch):
    monkeypatch.delenv('DIAGNOSTIC_INITIAL_ROBOT_REFRESH',raising=False)
    events=[];panel=panel_environment(events,{'obs':torch.ones(2,3)})
    source=HERE/'resident_isaac_trainer.py'
    cls=deepcopy(next(n for n in ast.parse(source.read_text()).body
        if isinstance(n,ast.ClassDef) and n.name=='ResidentTrainingEnvironment'))
    from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import FullResetInventoryMixin
    scope={'PanelResetEnvironment':type(panel),'FullResetInventoryMixin':FullResetInventoryMixin}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),scope)
    env=scope['ResidentTrainingEnvironment']();env.__dict__.update(panel.__dict__)
    env._resident_boundary_reset=True
    obs=env.reset_all()
    assert torch.equal(obs['obs'],torch.ones(2,3))
    assert events==['scene_reset','native_robot_writes','forward_without_physics','observations_constructed']
    assert not env.need_to_refresh_envs.any() and not env._resident_boundary_reset
    assert not hasattr(env,'_diagnostic_initial_refresh_attestation')
    events.clear();env.reset_all()
    assert events==['native_robot_writes','forward_without_physics','observations_constructed']
    # The same environment still cannot enter success evaluation without the
    # corrected explicit selector, even after a legitimate training reset.
    events.clear()
    with pytest.raises(ValueError,match='retired'):env.reset_panel()
    assert events==[]


def test_clear_once_occurs_after_native_writes_and_observations_only(monkeypatch):
    monkeypatch.setenv('DIAGNOSTIC_INITIAL_ROBOT_REFRESH',refresh.CLEAR_ONCE)
    monkeypatch.setenv('DIAGNOSTIC_ARM','teacher');monkeypatch.setenv('DIAGNOSTIC_TEACHER_ACTION_MODE','raven_ppo')
    events=[];obs={'teacher_obs':torch.ones(2,226),'homie_obs':torch.zeros(2,516)}
    env=panel_environment(events,obs)
    before={name:getattr(env,name).clone() for name in ('target_robot_root_states','target_robot_dof_state',
                                                      '_homie_commands','_last_homie_commands')}
    rng=torch.get_rng_state().clone();out=env.reset_panel()
    assert events[-2:]==['observations_constructed','clear_pending_once']
    assert all(torch.equal(out[key],value) for key,value in obs.items())
    assert all(torch.equal(getattr(env,key),value) for key,value in before.items())
    assert torch.equal(torch.get_rng_state(),rng)
    assert not env.need_to_refresh_envs.any()
    assert refresh.validate_attestation(env._diagnostic_initial_refresh_attestation)==evidence()
    with pytest.raises(RuntimeError,match='already applied'):
        refresh.clear_already_applied_initial_refresh(env)


@pytest.mark.parametrize('arm,mode,count,resume',[('bc','raven_ppo',2,False),('teacher','distill',2,False),
    ('teacher','raven_ppo',3,False),('teacher','raven_ppo',True,False),('teacher','raven_ppo',2,True)])
def test_wrong_arm_mode_batch_or_resume_rejected_before_mutation(arm,mode,count,resume):
    with pytest.raises(ValueError,match='fresh non-resume teacher PPO B1/B2'):
        refresh.validate_mode(refresh.CLEAR_ONCE,arm=arm,teacher_action_mode=mode,num_envs=count,has_resume=resume)


def test_inconsistent_flags_rejected_without_modifying_any_lane():
    env=SimpleNamespace(num_envs=2,need_to_refresh_envs=torch.tensor([True,False]))
    before=env.need_to_refresh_envs.clone()
    with pytest.raises(RuntimeError,match='not all already-applied'):
        refresh.clear_already_applied_initial_refresh(env)
    assert torch.equal(env.need_to_refresh_envs,before)


@pytest.mark.parametrize('mode', [refresh.CLEAR_ONCE, refresh.STUDENT_CLEAR_ONCE])
def test_original_post_step_ast_clear_only_avoids_duplicate_initial_write_and_preserves_done_resets(mode):
    source=REPO/'pipeline/GR00T-VisualSim2Real/gr00t/rl/envs/loco_manip/walk_stand_place_grasp_turn_homie.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='WalkStandPlaceGraspTurnHomie')
    method=deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_post_physics_step'))
    # Exact source prefix through the pending-refresh branch, before unrelated
    # camera/observation/history code. All statements in the reset path remain.
    end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.If) and 'len(refresh_env_ids)' in ast.unparse(n.test))
    method.body=method.body[:end+1]
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
    def execute(clear,done):
        writes=[];env=SimpleNamespace(num_envs=2,need_to_refresh_envs=torch.ones(2,dtype=torch.bool),
            episode_length_buf=torch.zeros(2,dtype=torch.int64),reset_buf=torch.tensor([done,done]),
            target_robot_root_states=torch.zeros(2,13),target_robot_dof_state=torch.zeros(2,43,2),
            config=SimpleNamespace(reset_from_dataset=SimpleNamespace(enable=True)))
        for name in ('_refresh_sim_tensors','_update_counters_each_step','_pre_compute_observations_callback',
                     '_update_tasks_callback','_check_termination','_compute_reward'):
            setattr(env,name,lambda:None)
        def reset(ids):
            if len(ids):writes.append('ordinary_task_reset');env.need_to_refresh_envs[ids]=True
        env.reset_envs_idx=reset
        env.simulator=SimpleNamespace(set_actor_root_state_tensor=lambda *_:writes.append('robot_root'),
                                     set_dof_state_tensor=lambda *_:writes.append('robot_dofs'))
        if clear:
            refresh.clear_already_applied_initial_refresh(env, mode=mode,
                arm='bc' if mode == refresh.STUDENT_CLEAR_ONCE else 'teacher',
                teacher_action_mode='distill' if mode == refresh.STUDENT_CLEAR_ONCE else 'raven_ppo')
        scope['_post_physics_step'](env)
        return writes,env.need_to_refresh_envs
    assert execute(False,False)[0]==['robot_root','robot_dofs']
    assert execute(True,False)[0]==[]
    assert execute(False,True)[0]==execute(True,True)[0]==['ordinary_task_reset','robot_root','robot_dofs']
    assert not execute(True,True)[1].any()


def test_full_panel_receipt_progress_and_summary_bind_refresh_mode_without_success_changes():
    p=panel();saved=[];events=[]
    actual=run_panel(p,execute_cohort=lambda spec,_:result(spec),save_receipt=saved.append,report=events.append,**run_kwargs())
    assert actual[refresh.FIELD]==refresh.CLEAR_ONCE
    assert actual['successes']==2 and actual['completed_episodes']==4
    assert len(actual['initial_robot_refresh_attestations'])==2
    kwargs=dict(checkpoint_sha256='b'*64,arm='teacher',teacher_action_mode='raven_ppo',
                diagnostic_initial_robot_refresh=refresh.CLEAR_ONCE)
    assert all(validate_cohort_receipt(p,row,**kwargs)==row for row in saved)
    assert completed_progress(p,saved,**kwargs)[refresh.FIELD]==refresh.CLEAR_ONCE
    assert any(row.get(refresh.FIELD)==refresh.CLEAR_ONCE for row in events)
    assert validate_result(p,actual,'b'*64,allowed_arms=('teacher',),teacher_action_mode='raven_ppo',
                           diagnostic_initial_robot_refresh=refresh.CLEAR_ONCE)==actual
    with pytest.raises(ValueError,match='differs from requested'):
        validate_result(p,actual,'b'*64,allowed_arms=('teacher',),teacher_action_mode='raven_ppo')
    with pytest.raises(ValueError,match='non-resume'):
        run_panel(p,execute_cohort=lambda *_:pytest.fail('must not replay'),completed_receipts=saved[:1],**run_kwargs())


def test_original_receipts_cannot_be_relabelled_and_intervention_evidence_cannot_be_forged():
    p=panel();spec=p['cohorts'][0]
    with pytest.raises(ValueError,match='differs from requested'):
        make_cohort_receipt(p,0,ppo_result(spec),checkpoint_sha256='b'*64,arm='teacher',teacher_action_mode='raven_ppo',
                            diagnostic_initial_robot_refresh=refresh.CLEAR_ONCE)
    forged=result(spec);forged['initial_robot_refresh_attestation']['pending_after'][0]=True
    with pytest.raises(ValueError,match='attestation differs'):
        summarize({**p,'cohorts':[spec],'episode_count':2},[forged])
    original=ppo_result(spec)
    assert refresh.FIELD not in make_cohort_receipt(p,0,original,checkpoint_sha256='b'*64,arm='teacher',teacher_action_mode='raven_ppo')


def test_rpc_rejects_clear_resume_and_student_before_loading_checkpoint(tmp_path):
    source=HERE/'resident_isaac_trainer.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ResidentRavenBackend')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    checked=[];scope={'os':os,'json':json,'Path':Path,'file_identity':lambda p,*_:checked.append(str(p))}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
    path=tmp_path/'panel.json';path.write_text(json.dumps(panel()))
    backend=SimpleNamespace(runtime_root=tmp_path,_consumed_outputs=set(),artifact=lambda _:path)
    base={'panel_file':'panel.json','panel_sha256':'a'*64,'checkpoint_sha256':'b'*64,'arm':'teacher',
          'output_name':'new-no-rewind','teacher_action_mode':'raven_ppo',refresh.FIELD:refresh.CLEAR_ONCE}
    scope['validate'](backend,'evaluate_explicit_panel',base)
    for extra in ({'arm':'bc'},{'teacher_action_mode':'distill'},
        {'completed_receipts_file':'old.json','completed_receipts_sha256':'c'*64}):
        checked.clear()
        with pytest.raises(ValueError,match='fresh non-resume|privileged teacher arm'):
            scope['validate'](backend,'evaluate_explicit_panel',{**base,**extra})
        assert not any('/diagnostic-assets/' in item for item in checked)


def test_initial_refresh_helper_import_is_stdlib_only():
    import subprocess
    subprocess.run([sys.executable,'-c',f"import sys;sys.path.insert(0,{str(HERE)!r});import initial_robot_refresh_diagnostic;assert 'torch' not in sys.modules"],check=True)


def test_same_initial_observation_yields_same_first_ppo_action_after_clear(monkeypatch):
    from test.deployment.rtx_actor.test_teacher_action_diagnostic import setup
    from teacher_action_diagnostic import diagnostic_step
    trainer,obs,_=setup();other=deepcopy(trainer)
    monkeypatch.setenv('DIAGNOSTIC_ARM','teacher');monkeypatch.setenv('DIAGNOSTIC_TEACHER_ACTION_MODE','raven_ppo')
    # Historical observation fixture only: the retired reset must never execute.
    initial_original=deepcopy(obs)
    clear=panel_environment([],obs)
    monkeypatch.setenv('DIAGNOSTIC_INITIAL_ROBOT_REFRESH',refresh.CLEAR_ONCE)
    initial_clear=clear.reset_panel()
    assert all(torch.equal(initial_original[key],initial_clear[key]) for key in obs)
    torch.manual_seed(451)
    left,_=diagnostic_step(trainer,initial_original,None,arm='teacher',mode='raven_ppo')
    after_original=torch.get_rng_state().clone()
    torch.manual_seed(451)
    right,_=diagnostic_step(other,initial_clear,None,arm='teacher',mode='raven_ppo')
    assert torch.equal(left['actions'],right['actions'])
    assert torch.equal(torch.get_rng_state(),after_original)


def test_resume_helper_itself_rejects_clear_even_before_reading_file():
    source=HERE/'resident_isaac_trainer.py'
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef)
                and n.name=='_explicit_completed_receipts')
    scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),scope)
    backend=SimpleNamespace(artifact=lambda *_:pytest.fail('cannot read a resume for clear-once mode'))
    with pytest.raises(ValueError,match='fresh non-resume'):
        scope['_explicit_completed_receipts'](backend,{'arm':'teacher','teacher_action_mode':'raven_ppo',
            refresh.FIELD:refresh.CLEAR_ONCE},{})
