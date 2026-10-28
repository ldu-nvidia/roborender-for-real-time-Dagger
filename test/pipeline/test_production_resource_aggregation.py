"""CPU-only guard metadata/aggregation fixtures, never native qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from pipeline.dagger.coverage_production import (aggregate_resource_guards,validate_resource_aggregate,
    validate_segment,validate_qualification,digest)
from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import validate_summary
from pipeline.bc_training.handoff_common import publish
from test.pipeline.test_dagger_coverage_production import production
from test.pipeline.test_production_resident_pool import selected,pool

GIB=1<<30


def guard(gpu=0,phase=0,initial=10,final=31):
    return dict(schema='roborender.native-phase-resource-guard.v2',policy_version='roborender.native-memory-policy.v2',
        path=f'/dagger-runtime/phases/cpu-only-g{gpu}-p{phase}/resource-guard.json',
        raw_sha256=hashlib.sha256(f'cpu-only-{gpu}-{phase}'.encode()).hexdigest(),samples=2,passed=True,
        minimum_host_available_bytes=9*GIB,swap_peak_bytes=0,oom_events=0,
        limit_pressure_events=final,cache_reclaim_events=final,unsafe_pressure_events=0,
        initial_limit_pressure_events=initial,phase_limit_pressure_events=final-initial,
        phase_cache_reclaim_events=final-initial,phase_unsafe_pressure_events=0,
        memory_limit_bytes=21*GIB,swap_limit_bytes=0,maximum_working_set_bytes=19*GIB,
        maximum_some_avg10=3.,maximum_full_avg10=1.)


def rows():
    return [dict(physical_gpu=0,summary=guard()),dict(physical_gpu=1,summary=guard(1,initial=3,final=7))]


def test_benign_actual_v2_reclaim_retains_nonzero_absolute_raw_max():
    value=aggregate_resource_guards(rows(),per_gpu=1)
    assert value['limit_pressure_events']==38 and value['cache_reclaim_events']==38
    assert value['unsafe_pressure_events']==0 and value['phase_limit_pressure_events']==25
    assert value['resource_guards']==rows()


def test_v3_aggregate_preserves_warning_history_and_rejects_mixed_versions():
    actual=rows()
    for row in actual:
        row['summary'].update(schema='roborender.native-phase-resource-guard.v3',
            policy_version='roborender.native-memory-policy.v3',maximum_full_avg10=5.07,
            pressure_window=dict(warning_samples=1,maximum_warning_seconds=1.,failed=False))
    value=aggregate_resource_guards(actual,per_gpu=1)
    assert value['resource_guards']==actual and value['limit_pressure_events']==38
    actual[0]=rows()[0]
    with pytest.raises(ValueError,match='mixed'):aggregate_resource_guards(actual,per_gpu=1)


def test_qualification_absolute_counts_do_not_double_count_prior_phases():
    actual=rows()+[dict(physical_gpu=0,summary=guard(0,1,31,34)),dict(physical_gpu=1,summary=guard(1,1,7,9))]
    value=aggregate_resource_guards(actual,per_gpu=2)
    assert value['limit_pressure_events']==43  # 34+9, not 31+7+34+9.
    assert value['cache_reclaim_events']==43 and value['phase_limit_pressure_events']==30


@pytest.mark.parametrize('field,value',[
    ('passed',False),('schema','roborender.native-phase-resource-guard.v1'),('policy_version','wrong'),
    ('unsafe_pressure_events',1),('cache_reclaim_events',0),('phase_cache_reclaim_events',0),
    ('phase_limit_pressure_events',0),('phase_unsafe_pressure_events',1),('oom_events',1),
    ('swap_peak_bytes',1),('swap_limit_bytes',1),('memory_limit_bytes',22*GIB),
    ('minimum_host_available_bytes',7*GIB),('maximum_working_set_bytes',20*GIB+1),
    ('maximum_some_avg10',11.),('maximum_full_avg10',6.),('samples',1),
])
def test_unsafe_unknown_policy_or_forged_classification_is_rejected(field,value):
    actual=rows();actual[0]['summary'][field]=value
    with pytest.raises((RuntimeError,ValueError)):aggregate_resource_guards(actual,per_gpu=1)


def test_v2_aggregate_cannot_claim_zero_or_invent_event_counts():
    value=aggregate_resource_guards(rows(),per_gpu=1)
    for field in ('limit_pressure_events','cache_reclaim_events','phase_limit_pressure_events'):
        bad=deepcopy(value);bad[field]=0
        with pytest.raises(ValueError,match='reproduce'):validate_resource_aggregate(bad,per_gpu=1,v2=True)


def test_duplicate_actual_path_or_digest_is_rejected():
    for field in ('path','raw_sha256'):
        actual=rows();actual[1]['summary'][field]=actual[0]['summary'][field]
        with pytest.raises(ValueError,match='duplicate'):aggregate_resource_guards(actual,per_gpu=1)


def test_counter_reset_or_wrong_gpu_cardinality_cannot_be_hidden():
    actual=rows()+[dict(physical_gpu=0,summary=guard(0,1,0,5)),dict(physical_gpu=1,summary=guard(1,1,7,9))]
    with pytest.raises(ValueError,match='backwards'):aggregate_resource_guards(actual,per_gpu=2)
    actual=rows();actual[1]['physical_gpu']=0
    with pytest.raises(ValueError,match='cardinality'):aggregate_resource_guards(actual,per_gpu=1)


def test_legacy_v1_requires_literal_zero_and_never_reclassifies_max():
    value=dict(passed=True,samples=2,minimum_host_available_bytes=9*GIB,swap_peak_bytes=0,oom_events=0,limit_pressure_events=0)
    assert validate_summary(value)==value
    assert validate_resource_aggregate(value,per_gpu=1,v2=False)==value
    value['limit_pressure_events']=31
    with pytest.raises(RuntimeError,match='legacy'):validate_summary(value)
    with pytest.raises(ValueError):validate_resource_aggregate(value,per_gpu=1,v2=False)


def test_bc_complete_dispatch_accepts_actual_v2_policy_without_erasing_max(selected,pool,tmp_path,monkeypatch):
    from pipeline.deployment import bc_success_dispatch as dispatcher
    selection,result=selected
    actual_guard=guard()
    receipt=dict(complete=True,optimizer_steps=0,native_initializations=1,result=result,resource_guard=actual_guard)
    class Transport:
        def stage(self,*args):pass
        def command(self,argv):
            action=argv[argv.index('--action')+1]
            if action=='status':
                return dispatcher.canonical(dict(workers=[dict(worker_instance_id=w['worker_instance_id'],
                    source_sha256=pool['source_sha256']) for w in pool['workers']]))
            if action=='submit':return b'{"results":[{"status":"accepted"}]}'
            return dispatcher.canonical(dict(results=[dict(status='COMPLETE',receipt=receipt)]))
    monkeypatch.setattr(dispatcher,'log_success',lambda *a,**k:dict(online_readback_verified=True))
    selection_ref=publish(tmp_path/'selection.json',selection);pool_ref=publish(tmp_path/'pool.json',pool)
    output=tmp_path/'bc/dispatch'
    done=dispatcher.execute(selection_ref,pool_ref,'/owned/pool.json','/owned/source',output,transport=Transport())
    assert done['status']=='BC295_COMPLETE'
    import json
    observed=json.loads((Path(done['dispatch_output'])/'native-command-result.json').read_bytes())
    assert observed['results'][0]['receipt']['resource_guard']['limit_pressure_events']==31


def test_segment_v2_uses_exact_actual_policy_proof_and_preserves_algorithm(production):
    _,plan=production;cohort=plan['rounds'][0]['epochs'][0]['cohorts'][0]
    windows=[];parent=dict(sha256='d'*64,optimizer_global_step=100);sha=parent['sha256']
    for index in range(2):
        out=hashlib.sha256(f'cpu-only-learner-{index}'.encode()).hexdigest()
        windows.append(dict(iteration_id=index+1,input_checkpoint_sha256=sha,
            output_checkpoint=dict(sha256=out,optimizer_global_step=101+index,optimizer_state_preserved=True),
            transition_batches=81,train_chunks=4,train_frames=324,optimizer_steps=1,full_batch_environments=4,
            replay_manifest_sha256=None,teacher_beta=.75,student_action_dimensions=11,train_mse=.2))
        sha=out
    record=dict(schema='roborender.dagger-production-segment-result.v2',run_id=plan['run_id'],round_id=1,
        coverage_epoch=1,cohort_index=0,plan_sha256=digest(plan),source_sha256=plan['source_sha256'],
        native_complete=True,native_quiescent=True,native_lifecycle_initializations_per_actor=[1,1],
        history_reset_at_window_boundary=False,assignments=[digest(a) for a in cohort['assignments']],
        windows=windows,allocation_retained=True,online_metrics_verified=False,**aggregate_resource_guards(rows(),per_gpu=1))
    assert validate_segment(plan,1,1,0,parent,record)['sha256']==sha
    bad=deepcopy(record);bad['cache_reclaim_events']=0
    with pytest.raises(ValueError):validate_segment(plan,1,1,0,parent,bad)
    bad=deepcopy(record);bad['schema']='roborender.dagger-production-segment-result.v1'
    with pytest.raises(ValueError):validate_segment(plan,1,1,0,parent,bad)


def test_qualification_v2_accepts_only_reproducible_actual_guards_and_unchanged_lineage(production,tmp_path):
    _,plan=production;references=[]
    for cohort_index,cohort in enumerate(plan['rounds'][0]['epochs'][0]['cohorts'][:2]):
        for gpu,assignment in enumerate(cohort['assignments']):
            for lane in assignment['lanes']:
                episode=f'cpu-only-{cohort_index}-{gpu}-{lane["lane"]}'
                native=publish(tmp_path/(episode+'.json'),dict(schema='roborender.native-episode-assignment.v1',
                    purpose='training',simulator_seed=assignment['simulator_seed'],motion_id=lane['motion_id'],
                    motion_sha256=lane['sha256'],episode_id=episode))
                reset=publish(tmp_path/(episode+'.reset.json'),dict(schema='roborender.native-episode-reset-evidence.v1',
                    episode_id=episode,assignment_sha256=digest(assignment)))
                references.append(dict(**native,reset_evidence=reset))
    jobs=[];parent='d'*64
    for index in range(4):
        candidate=hashlib.sha256(f'cpu-only-job-{index}'.encode()).hexdigest()
        jobs.append(publish(tmp_path/f'job-{index}.json',dict(schema='roborender.dagger_learner_candidate_receipt.v3',
            parent_checkpoint_sha256=parent,output_checkpoint_sha256=candidate,train_chunks=4,train_frames=324,
            replay_manifest_sha256=None,batching_semantics='raven_exact_flattened_bxt',optimizer_state_preserved=True,
            global_step=101+index,parent_global_step=100+index)))
        parent=candidate
    actual=rows()+[dict(physical_gpu=0,summary=guard(0,1,31,34)),dict(physical_gpu=1,summary=guard(1,1,7,9))]
    value=dict(schema='roborender.dagger-full295-two-cohort-qualification.v2',purpose='qualification',
        plan_sha256=digest(plan),parent_checkpoint_sha256='d'*64,source_sha256=plan['source_sha256'],
        native_cohorts=2,optimizer_updates=4,native_initializations_per_cohort=[1,1],vector_steps_per_cohort=[162,162],
        transitions=1296,full_batch_environments=4,no_replay=True,history_reset_at_window_boundary=False,
        worker_uuid_pairs=[['cpu-only-rtx0','cpu-only-rtx1']]*2,h200_uuids=['cpu-only-h200-'+str(i) for i in range(4)],
        native_assignment_receipts=references,learner_receipts=jobs,completed=True,allocation_retained=True,
        checkpoint_optimizer_lineage_verified=True,**aggregate_resource_guards(actual,per_gpu=2))
    assert validate_qualification(plan,value,'d'*64)==value
    bad=deepcopy(value);bad['limit_pressure_events']=0
    with pytest.raises(ValueError,match='reproduce'):validate_qualification(plan,bad,'d'*64)
