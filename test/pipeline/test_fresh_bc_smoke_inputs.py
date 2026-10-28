"""Fresh parent admission tests. No renderer, learner, or simulator is run."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.bc_rendering import render_bc_teacher_8gpu as render
from pipeline.bc_rendering.render_recipes import CURRENT_RECIPE,recipe_config
from pipeline.bc_training import fresh_smoke_inputs as fresh
from pipeline.bc_training import qualify_raven_bc32_lifecycle as smoke


def staged(tmp_path):
    jobs=[dict(source_npz=f'/raw/chunk_{i:07d}.npz',output_npz=f'/rendered/chunk_{i:07d}.npz',
        source_payload_sha256=f'{i:064x}',sequence_id=i,stream_tag='continuous',
        env_id=i%2,global_env_id=i%2,episode_id=str(i),frame_count=81,has_done=False)
        for i in range(64)]
    chains=render.build_render_chains(jobs)
    lane=tmp_path/'lane.json'
    lane.write_text(json.dumps(dict(schema='roborender.bc_teacher_lane_plan.v3',config={},
        jobs=[j for c in chains for j in c['jobs']])))
    plan=dict(schema='roborender.bc_teacher_8gpu_plan.v3',status='complete',episode_count=64,
        frame_count=64*81,chain_count=64,config={},lane_plan_paths=[str(lane)],
        **recipe_config(CURRENT_RECIPE))
    return plan,jobs,lane


def test_complete64_native_jobs_only_and_exact_temporal_identity(tmp_path):
    plan,jobs,_=staged(tmp_path)
    actual=fresh.validate_render_plan(plan,jobs)
    assert len(actual)==64 and {j['sequence_id'] for j in actual.values()}==set(range(64))
    assert all(j['chain_length']==1 and j['frame_count']==81 for j in actual.values())


@pytest.mark.parametrize('mutation',['incomplete','partial','different_source','duplicate','changed_order','wrong_recipe'])
def test_incomplete_or_mismatched_render_evidence_rejected(tmp_path,mutation):
    plan,jobs,lane=staged(tmp_path);value=json.loads(lane.read_text())
    if mutation=='incomplete':plan['status']='running'
    elif mutation=='partial':value['jobs'].pop()
    elif mutation=='different_source':value['jobs'][0]['source_payload_sha256']='e'*64
    elif mutation=='duplicate':value['jobs'][0]=deepcopy(value['jobs'][1])
    elif mutation=='changed_order':value['jobs'][0]['sequence_id']+=3
    else:plan['max_chain_chunks']=6
    lane.write_text(json.dumps(value))
    with pytest.raises(ValueError):fresh.validate_render_plan(plan,jobs)


def test_fresh_mode_cannot_fall_back_to_retired_selector(monkeypatch):
    args=SimpleNamespace(fresh_continuous_pool=Path('/fresh'),fresh_render_plan=Path('/plan'),
        fresh_rendered_root=Path('/generated'))
    monkeypatch.setattr(smoke.base,'plan',lambda a:pytest.fail('retired selector used'))
    monkeypatch.setattr(fresh,'plan',lambda a,source_identity:{'fresh':True,'source_identity':source_identity})
    assert smoke.plan(args)['fresh'] is True


@pytest.mark.parametrize('field',fresh.FIELDS)
def test_partial_fresh_arguments_fail_closed(field):
    args=SimpleNamespace(**{k:None for k in fresh.FIELDS});setattr(args,field,Path('/fresh'))
    with pytest.raises(ValueError):fresh.requested(args)


def test_alias_path_rejected(tmp_path):
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real)
    with pytest.raises(ValueError):fresh.physical(alias,directory=True)


def test_unbound_render_files_are_not_execution_evidence():
    with pytest.raises(ValueError,match='identity-bound'):
        fresh.validate_render_execution({'config':{}},Path('/plan'),{})


@pytest.mark.parametrize('missing',[False,True])
def test_fresh_parent_requires_verified_exact_worker_coverage(monkeypatch,missing):
    from pipeline.bc_training import render_execution as execution
    binding={'path':'/identity','raw_sha256':'a'*64};protocol={'actual':'protocol'}
    expected={'/generated/one':{},'/generated/two':{}}
    verified=[]
    monkeypatch.setattr(execution,'bound',lambda actual:protocol if actual==binding else None)
    monkeypatch.setattr(execution,'verify_render_asset_identity',lambda value:verified.append(value))
    monkeypatch.setattr(execution,'_verified_outputs',lambda b,p,r:
        {'/generated/one':{}} if missing else expected)
    if missing:
        with pytest.raises(ValueError,match='exact64'):
            fresh.validate_render_execution({'config':{'execution_identity':binding}},Path('/plan'),expected)
    else:
        assert fresh.validate_render_execution({'config':{'execution_identity':binding}},Path('/plan'),expected)==binding
    assert verified==[protocol]
