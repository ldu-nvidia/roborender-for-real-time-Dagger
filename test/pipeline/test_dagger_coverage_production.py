"""Full295 coverage/controller CPU tests; generated files are not native data."""
from test.deployment.source_paths import REPO_ROOT
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest

from pipeline.dagger.coverage_production import (
    make_plan,validate_plan,digest,make_first_config,make_segment,stage,validate_qualification,
)
from pipeline.bc_training.handoff_common import binding,bound,publish
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    validate_config,is_mse_production,is_mse_smoke,budgets,environments_per_actor,
)

@pytest.fixture(scope='module')
def production(tmp_path_factory):
    root=tmp_path_factory.mktemp('full295')
    from pipeline.deployment.rtx_actor.production_success_contract import build_rendered_panel
    original=REPO_ROOT/'runtime/production-audits/20261015-raven-faithful/plan-only-native295-b2-20261015.success-panel.json'
    rendered=build_rendered_panel(json.loads(original.read_bytes()))
    rows=[dict(motion_id=lane['motion_id'],filename=Path(lane['path']).name,sha256=lane['sha256'])
          for cohort in rendered['cohorts'] for lane in cohort['assignment']['lanes']]
    inventory=dict(schema='roborender.motion-inventory.v1',motion_count=295,motions=rows,
                   motion_records_sha256=digest(rows))
    inv=publish(root/'inventory.json',inventory)
    panel=publish(root/'panel.json',rendered)
    rule=dict(maximum_coverage_epochs=2,minimum_coverage_epochs=2,patience=3,
              relative_improvement=.005,absolute_improvement=1e-6)
    plan=make_plan(run_id='dagger-leo-20261016-prod-1',inventory=inv,panel=panel,
        excluded_process_seeds=[0,1,2,3,4],seed=20261016,motion_root='/motions',
        source_sha256='a'*64,teacher_sha256='b'*64,policy_spec_sha256='c'*64,stopping=rule)
    return root,plan


def rendering_dependencies(root,plan):
    from test.deployment.rtx_actor.test_rendered_student_success_contract import rendering_config
    path=Path(root)/'rendering-dependencies.json'
    if not path.exists():publish(path,rendering_config(bound(plan['panel']),'d'*64))
    return binding(path)

def test_every_epoch_covers295_with_one_declared_padding_lane(production):
    _,plan=production;validate_plan(plan)
    for r in plan['rounds']:
        for e in r['epochs']:
            counts=Counter(l['motion_id'] for c in e['cohorts'] for a in c['assignments'] for l in a['lanes'])
            assert len(counts)==295 and sum(counts.values())==296
            assert sorted(counts.values())==[1]*294+[2]
            assert len(e['cohorts'])==74 and all(c['window_count']==2 for c in e['cohorts'])
            assert sum(bool(c['padding_slots']) for c in e['cohorts'])==1

def test_new_production_defaults_stage_five_checks_for_every_round(production, tmp_path):
    from pipeline.dagger.coverage_production import STOPPING
    from pipeline.bc_training.train_bc_h200 import EarlyStopping
    _, original = production
    plan = make_plan(**{key: original[key] for key in (
        'run_id', 'inventory', 'panel', 'excluded_process_seeds', 'seed', 'motion_root',
        'source_sha256', 'teacher_sha256', 'policy_spec_sha256')})
    assert plan['stopping'] == STOPPING and STOPPING['patience'] == 5
    assert original['stopping']['patience'] == 3  # Existing frozen histories are not rewritten.
    stage(plan, output_root=tmp_path / 'new-plan', checkpoint=tmp_path / 'parent.pt',
          checkpoint_sha256='d' * 64,rendered_observation_config=rendering_dependencies(tmp_path,plan))
    staged = json.loads((tmp_path / 'new-plan/mse-plan.json').read_bytes())
    assert [row['patience'] for row in staged['rounds']] == [5, 5, 5, 5]
    stop = EarlyStopping(min_epochs=STOPPING['minimum_coverage_epochs'],
        patience=STOPPING['patience'], relative_improvement=STOPPING['relative_improvement'],
        absolute_improvement=STOPPING['absolute_improvement'])
    stop.update(0, .2)
    for epoch in range(1, 5):
        assert stop.update(epoch, .21)[1] is False
    assert stop.update(5, .21)[1] is True

def test_all_train_heldout_test_seeds_are_disjoint(production):
    _,plan=production
    train=[a['simulator_seed'] for r in plan['rounds'] for e in r['epochs'] for c in e['cohorts'] for a in c['assignments']]
    val=[c['simulator_seed'] for r in plan['rounds'] for c in r['heldout']['cohorts']]
    assert len(set(train))==len(train) and len(set(val))==len(val)
    assert not set(train)&set(val) and not (set(train)|set(val))&set(plan['excluded_process_seeds'])
    for r in plan['rounds']:
        counts=Counter(l['motion_id'] for c in r['heldout']['cohorts'] for l in c['lanes'])
        assert len(counts)==295 and sum(counts.values())==320 and sorted(counts.values())==[1]*270+[2]*25

def test_no_mutation_of_legacy_smoke_contract(production):
    root,plan=production
    ref=publish(root/'coverage-test.json',plan)
    cfg=make_first_config(plan,coverage_plan=ref,checkpoint=root/'selected.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(root,plan))
    assert validate_config(cfg)==cfg and is_mse_production(cfg) and not is_mse_smoke(cfg)
    assert budgets(cfg)==(1480,295,'explicit-production-image3f-fixed295-v1') and environments_per_actor(cfg)==2
    bad=deepcopy(cfg);bad['rtx_actors'][0]['concurrent_environments']=6
    with pytest.raises(ValueError):validate_config(bad)

def test_source_seed_or_outcome_based_plan_change_rejected(production):
    _,plan=production;bad=deepcopy(plan)
    bad['rounds'][0]['heldout']['cohorts'][0]['simulator_seed']+=1
    with pytest.raises(ValueError,match='frozen'):validate_plan(bad)

def test_actual_raw_binding_enters_segment_not_semantic_hash(production):
    root,plan=production
    ref=publish(root/'coverage-newline.json',plan)
    assert ref['raw_sha256']!=digest(plan)
    cohort=plan['rounds'][0]['epochs'][0]['cohorts'][0]
    refs=[dict(physical_gpu_index=g,path=f'/jobs/g{g}.json',sha256=digest(a)) for g,a in enumerate(cohort['assignments'])]
    segment=make_segment(plan,round_id=1,epoch=1,index=0,input_checkpoint_sha256='d'*64,
        mse_protocol_file_sha256='e'*64,motion_assignments=refs,coverage_plan_sha256=ref['raw_sha256'])
    assert segment['coverage_plan_sha256']==ref['raw_sha256']
    assert segment['start_iteration_index']==0 and segment['episode_namespace']=='prod-r1-e01-c000'

def test_stage_requires_new_explicit_production_schema(production,tmp_path):
    _,plan=production
    result=stage(plan,output_root=tmp_path,checkpoint=tmp_path/'selected.pt',checkpoint_sha256='d'*64,
        rendered_observation_config=rendering_dependencies(tmp_path,plan))
    first=json.loads(Path(result['first_round']['path']).read_bytes())
    campaign=json.loads(Path(result['mse_plan']['path']).read_bytes())
    assert first['logging']['wandb_run_name']=='2026-10-16-prod-1-dagger-teacher75'
    assert all(r['interval_iterations']==148 and r['max_iterations']==296 for r in campaign['rounds'])
    assert result['native_qualification_completed'] is False and result['launched'] is False

def test_cpu_plan_cannot_substitute_for_native_qualification(production):
    _,plan=production
    with pytest.raises(ValueError,match='genuine'):validate_qualification(plan,dict(completed=True),'d'*64)

def test_native_assignment_plus_reset_bindings_are_checked_not_just_PASS(production,tmp_path):
    # Protocol fixtures only: no GPU run or qualification certificate is claimed.
    _,plan=production;references=[]
    for c,cohort in enumerate(plan['rounds'][0]['epochs'][0]['cohorts'][:2]):
        for gpu,assignment in enumerate(cohort['assignments']):
            for lane in assignment['lanes']:
                episode=f'fixture-{c}-{gpu}-{lane["lane"]}'
                native=publish(tmp_path/f'{episode}.json',dict(schema='roborender.native-episode-assignment.v1',
                    purpose='training',simulator_seed=assignment['simulator_seed'],motion_id=lane['motion_id'],
                    motion_sha256=lane['sha256'],episode_id=episode))
                reset=publish(tmp_path/f'{episode}.reset.json',dict(schema='roborender.native-episode-reset-evidence.v1',
                    episode_id=episode,assignment_sha256=digest(assignment)))
                references.append(dict(**native,reset_evidence=reset))
    jobs=[];parent='d'*64
    for i in range(4):
        candidate=hashlib.sha256(f'fixture-{i}'.encode()).hexdigest()
        jobs.append(publish(tmp_path/f'job-{i}.json',dict(schema='roborender.dagger_learner_candidate_receipt.v3',
            parent_checkpoint_sha256=parent,output_checkpoint_sha256=candidate,train_chunks=4,train_frames=324,
            replay_manifest_sha256=None,batching_semantics='raven_exact_flattened_bxt',
            optimizer_state_preserved=True,global_step=101+i,parent_global_step=100+i)))
        parent=candidate
    value=dict(schema='roborender.dagger-full295-two-cohort-qualification.v1',purpose='qualification',
        plan_sha256=digest(plan),parent_checkpoint_sha256='d'*64,source_sha256=plan['source_sha256'],
        native_cohorts=2,optimizer_updates=4,native_initializations_per_cohort=[1,1],vector_steps_per_cohort=[162,162],
        transitions=1296,full_batch_environments=4,no_replay=True,history_reset_at_window_boundary=False,
        minimum_host_available_bytes=9<<30,oom_events=0,swap_peak_bytes=0,limit_pressure_events=0,
        worker_uuid_pairs=[['fixture0','fixture1']]*2,h200_uuids=['fixtureH'+str(i) for i in range(4)],
        native_assignment_receipts=references,learner_receipts=jobs,completed=True,
        allocation_retained=True,checkpoint_optimizer_lineage_verified=True)
    assert validate_qualification(plan,value,'d'*64)==value
    bad=deepcopy(value);bad['native_assignment_receipts']=bad['native_assignment_receipts'][:-1]
    with pytest.raises(ValueError):validate_qualification(plan,bad,'d'*64)
    bad=deepcopy(value);bad['learner_receipts'][1]=bad['learner_receipts'][0]
    with pytest.raises(ValueError,match='optimizer'):validate_qualification(plan,bad,'d'*64)

def test_uncompleted_native_intent_is_never_replayed(tmp_path):
    from pipeline.deployment.resident_production_collection import completed_prefix
    directory=tmp_path/'cohort';directory.mkdir()
    assert completed_prefix(directory,{},1,1,0,'d'*64,[]) is None
    (directory/'intent.json').write_text('{}')
    with pytest.raises(RuntimeError,match='never replay'):completed_prefix(directory,{},1,1,0,'d'*64,[])
