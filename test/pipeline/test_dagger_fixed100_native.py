"""CPU-only native fixed100 admission and barrier regressions."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from pipeline.bc_training.handoff_common import publish, bound
from pipeline.dagger.coverage_production import make_plan, make_first_config, make_segment, digest
from test.pipeline.test_dagger_coverage_production import production, rendering_dependencies
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import validate_config, driver_config, canonical
from pipeline.dagger.rtx.code.src.dagger_rtx.mse_round_driver import MseBetaRoundDriver, read_mse_protocol
from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import ProductionSegmentDriver, validate_segment
from pipeline.dagger.rtx.code.src.dagger_rtx.qualification_driver import BetaRoundDriver, actor_ids


@pytest.fixture
def fixed_inputs(production,tmp_path):
    _,old=production
    plan=make_plan(**{key:old[key] for key in ('run_id','inventory','panel','excluded_process_seeds',
        'seed','motion_root','source_sha256','teacher_sha256','policy_spec_sha256')},
        environments_per_actor=16,fixed100=True)
    ref=publish(tmp_path/'plan.json',plan)
    cfg=make_first_config(plan,coverage_plan=ref,checkpoint=tmp_path/'parent.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan),gateway_pids=[101,102,103,104])
    panel=bound(plan['panel'])
    scenario_ids=[case for cohort in panel['cohorts'] for case in cohort['case_ids']]
    protocol=dict(schema='roborender.dagger-fixed100-round-protocol.v1',run_id=cfg['run_id'],round_id=1,
        initial_checkpoint_sha256='d'*64,interval_iterations=100,max_iterations=100,
        full_batch_environments=32,rollout_steps=81,checkpoint_selection='final_update_100',
        final_test=dict(scenario_manifest_sha256=cfg['validation']['scenario_manifest_sha256'],
            scenario_ids=scenario_ids,observation_mode='generated'))
    return plan,ref,cfg,protocol


def test_fixed100_config_protocol_and_final_segment_without_heldout(fixed_inputs,tmp_path):
    plan,ref,cfg,protocol=fixed_inputs
    assert validate_config(cfg)==cfg
    assert all('heldout' not in row for row in plan['rounds'])
    path=tmp_path/'protocol.json';path.write_bytes(canonical(protocol))
    assert read_mse_protocol(path,hashlib.sha256(path.read_bytes()).hexdigest(),cfg)==protocol
    cohort=plan['rounds'][0]['epochs'][4]['cohorts'][9]
    refs=[dict(physical_gpu_index=g,path=f'/native/g{g}.json',sha256=digest(a))
          for g,a in enumerate(cohort['assignments'])]
    segment=make_segment(plan,round_id=1,epoch=5,index=9,input_checkpoint_sha256='e'*64,
        mse_protocol_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),motion_assignments=refs,
        coverage_plan_sha256=ref['raw_sha256'])
    assert validate_segment(segment,cfg,plan)['start_iteration_index']==98


@pytest.mark.parametrize('fault', ['missing_observation','fallback','native_mode','small_batch','legacy_schema'])
def test_fixed100_config_cannot_downgrade_source_or_batch(fixed_inputs,fault):
    _,_,cfg,_=deepcopy(fixed_inputs)
    if fault=='missing_observation':cfg.pop('student_observation')
    elif fault=='fallback':cfg['student_observation']['raytraced_policy_fallback']=True
    elif fault=='native_mode':cfg['student_observation']['mode']='raytraced'
    elif fault=='small_batch':cfg['rtx_actors'][0]['concurrent_environments']=2
    else:cfg['schema']='roborender.cross_gpu_dagger_mse_full295.leo.v1'
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('field,value', [('validation_manifest_path','/heldout.json'),
    ('patience',5), ('max_iterations',99), ('interval_iterations',20),
    ('full_batch_environments',4), ('checkpoint_selection','best_validation')])
def test_fixed100_native_protocol_rejects_legacy_or_short_budget(fixed_inputs,tmp_path,field,value):
    _,_,cfg,protocol=deepcopy(fixed_inputs);protocol[field]=value
    path=tmp_path/'protocol.json';path.write_bytes(canonical(protocol))
    with pytest.raises(ValueError):read_mse_protocol(path,hashlib.sha256(path.read_bytes()).hexdigest(),cfg)


def actor_state(fixed_inputs, *, iteration=100, terminal=True):
    _,_,cfg,protocol=fixed_inputs
    obj=object.__new__(ProductionSegmentDriver)
    obj.config=driver_config(cfg,'c'*64);obj.round_id=1;obj.mse_protocol=protocol
    obj.mse_sha256=hashlib.sha256(canonical(protocol)).hexdigest()
    obj.timeout=1;obj.monotonic=lambda:0;obj.sleep=lambda _:None;obj.selected=None
    parent='e'*64;candidate='f'*64
    selection=dict(checkpoint_selection='final_update_100',best_checkpoint_sha256=candidate if terminal else 'd'*64,
        history=[dict(iteration=100,result=dict(checkpoint_sha256=candidate))] if terminal else [])
    context=dict(run_id=cfg['run_id'],round_id=1,protocol=protocol,protocol_sha256=obj.mse_sha256,
        status='selected_pending_test' if terminal else 'active',iteration_id=iteration,
        checkpoint_sha256=candidate,stop_reason='fixed_100_updates' if terminal else None,selection=selection)
    row=dict(round_id=1,iteration_id=iteration,input_checkpoint_sha256=parent,
        candidate_checkpoint_sha256=candidate,status='selected_pending_test' if terminal else 'advanced')
    rows=[row]
    if not terminal:rows.append(dict(round_id=1,iteration_id=iteration+1,status='collecting',input_checkpoint_sha256=candidate))
    status=dict(run=dict(status='selected_pending_test' if terminal else 'collecting',active_checkpoint_sha256=candidate),
        iterations=rows,barriers=[dict(round_id=1,iteration_id=iteration,actor_id=actor,
            checkpoint_sha256=parent,chunk_ids=[f'd1-{actor}-window-{iteration-1:06d}']) for actor in actor_ids(obj.config)])
    obj.request=lambda *args:dict(ok=True,mse_round=deepcopy(context))
    return obj,context,status,parent,candidate


def test_only_update100_final_checkpoint_can_finish(fixed_inputs,monkeypatch):
    obj,context,status,parent,candidate=actor_state(fixed_inputs)
    monkeypatch.setattr(BetaRoundDriver,'status',lambda self:status)
    assert obj._candidate(99,parent)==candidate
    assert obj.selected==dict(selected_checkpoint_sha256=candidate,stop_reason='fixed_100_updates')
    context['selection']['best_checkpoint_sha256']='d'*64
    with pytest.raises(RuntimeError):obj._candidate(99,parent)


@pytest.mark.parametrize('fault',['early','late','heldout','missing_actor'])
def test_fixed100_refuses_early_test_late_collection_or_heldout(fixed_inputs,monkeypatch,fault):
    obj,context,status,parent,_=actor_state(fixed_inputs,iteration=99 if fault=='early' else 100,
                                           terminal=fault not in {'late','heldout'})
    if fault=='heldout':status['iterations'][0]['status']='awaiting_mse_validation'
    if fault=='missing_actor':status['barriers'].pop()
    monkeypatch.setattr(BetaRoundDriver,'status',lambda self:status)
    with pytest.raises(RuntimeError):obj._candidate(context['iteration_id']-1,parent)
