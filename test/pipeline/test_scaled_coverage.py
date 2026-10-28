"""CPU contract tests only; these do not qualify any native GPU capacity."""
from collections import Counter
from copy import deepcopy
import math
import pytest

from test.pipeline.test_dagger_coverage_production import production,rendering_dependencies
from pipeline.dagger.coverage_production import make_plan,validate_plan,make_first_config,stage,SCALED_SCHEMA
from pipeline.bc_training.handoff_common import publish
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import validate_config,budgets,environments_per_actor
from pipeline.dagger.beta_campaign import next_round_config


def scaled(original,batch):
    values={k:original[k] for k in ('inventory','panel','excluded_process_seeds','seed','motion_root',
        'source_sha256','teacher_sha256','policy_spec_sha256')}
    return make_plan(run_id='dagger-leo-10-17-prod',environments_per_actor=batch,**values)


@pytest.mark.parametrize('batch',[2,4,8,16,32,64])
def test_scaled_schedule_covers_every_motion_without_hidden_padding(production,tmp_path,batch):
    _,original=production;plan=scaled(original,batch);validate_plan(plan)
    assert plan['schema']==SCALED_SCHEMA and original['schema']!=SCALED_SCHEMA
    total=batch*2;cohorts=math.ceil(295/total);interval=2*cohorts
    assert plan['updates_per_coverage_epoch']==interval
    assert plan['max_updates_per_round']==10*interval
    assert plan['train_frames_per_update']==81*total
    assert plan['stopping']['patience']==5
    for row in plan['rounds']:
        assert row['heldout']['chunks']==320
        assert sum(len(c['lanes']) for c in row['heldout']['cohorts'])==320
        assert all(len(c['lanes'])==min(batch,16) for c in row['heldout']['cohorts'])
        assert len({l['motion_id'] for c in row['heldout']['cohorts'] for l in c['lanes']})==295
        assert row['success_test']['panel']==original['panel']
        padding_counts=Counter()
        for epoch in row['epochs']:
            seen=Counter(l['motion_id'] for c in epoch['cohorts'] for a in c['assignments'] for l in a['lanes'])
            assert len(seen)==295 and sum(seen.values())==cohorts*total
            for motion in seen:padding_counts.setdefault(motion,0)
            padding_counts.update(p['motion_id'] for c in epoch['cohorts'] for p in c['padding_slots'])
            assert sum(len(c['padding_slots']) for c in epoch['cohorts'])==cohorts*total-295
            assert all(len(a['lanes'])==batch for c in epoch['cohorts'] for a in c['assignments'])
            assert [c['start_iteration'] for c in epoch['cohorts']]==list(range(
                (epoch['coverage_epoch']-1)*interval,epoch['coverage_epoch']*interval,2))
        assert max(padding_counts.values())-min(padding_counts.values())<=1
    ref=publish(tmp_path/'coverage.json',plan)
    cfg=make_first_config(plan,coverage_plan=ref,checkpoint=tmp_path/'parent.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    assert environments_per_actor(cfg)==batch and budgets(cfg)[0]==10*interval
    assert cfg['logging']['wandb_run_name']=='10-17-prod-dagger-teacher75'
    stage(plan,output_root=tmp_path/'staged',checkpoint=tmp_path/'parent.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    bad=deepcopy(cfg);bad['rtx_actors'][1]['concurrent_environments']=batch+1
    with pytest.raises(ValueError):validate_config(bad)
    bad=deepcopy(plan);bad['updates_per_coverage_epoch']+=1
    with pytest.raises(ValueError):validate_plan(bad)


def test_scaled_round_names_follow_the_new_run_identity(production,tmp_path):
    _,original=production;plan=scaled(original,16)
    cfg=make_first_config(plan,coverage_plan=publish(tmp_path/'coverage.json',plan),
        checkpoint=tmp_path/'parent.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    for percentage in (50,25,0):
        cfg=next_round_config(cfg,final_candidate='e'*64,checkpoint_path=tmp_path/'next.pt',
            first_config_sha='f'*64,gateway_pids=[])
        assert cfg['logging']['wandb_run_name']==f'10-17-prod-dagger-teacher{percentage}'
        assert environments_per_actor(cfg)==16
