"""Versioned DAgger scheduling and receipt admission.

This is separate from the qualified two-motion smoke. Planning is CPU-only;
Legacy full-motion schedules require genuine two-cohort qualification. New
continuous rounds keep fixed lane motions and qualify their live four-update
prefix without replacing the scene. Both require bound host operations.
The module never allocates, releases, scales or terminates compute.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from pipeline.bc_training.handoff_common import binding,bound,canonical,publish
from pipeline.deployment.rtx_actor.native_motion_assignment import validate_motion_assignment
from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel,validate_result

SCHEMA='roborender.dagger-full295-production-plan.v1'
SCALED_SCHEMA='roborender.dagger-full295-production-plan.v2'
FIXED100_SCHEMA='roborender.dagger-full295-fixed100-plan.v1'
CONTINUOUS_FIXED100_SCHEMA='roborender.dagger-continuous-fixed100-plan.v1'
CONTINUOUS_FIXED100_8H200_SCHEMA='roborender.dagger-continuous-fixed100-plan.v2'
CONTINUOUS_SCHEMAS=(CONTINUOUS_FIXED100_SCHEMA,CONTINUOUS_FIXED100_8H200_SCHEMA)

def is_continuous_plan(plan):
    return plan.get('schema') in CONTINUOUS_SCHEMAS

def is_eight_h200_plan(plan):
    return plan.get('schema')==CONTINUOUS_FIXED100_8H200_SCHEMA
BETAS=(.75,.50,.25,0.)
TOPOLOGY=dict(host_id='leo-rtx-6k',physical_gpus=[0,1],environments_per_actor=2,
    total_environments=4,h200_workers=4,h200_shape='custom.gpu.4xh200',
    reservation='cosmos-solution-architect-xcwtffiq',allow_burst=False,
    minimum_host_available_bytes=8<<30,compute_release_permitted=False)
STOPPING=dict(maximum_coverage_epochs=10,minimum_coverage_epochs=2,patience=5,
              relative_improvement=.005,absolute_improvement=1e-6)
QUALIFICATION_FIELDS={'schema','purpose','plan_sha256','parent_checkpoint_sha256','source_sha256',
    'native_cohorts','optimizer_updates','native_initializations_per_cohort','vector_steps_per_cohort',
    'transitions','full_batch_environments','no_replay','history_reset_at_window_boundary',
    'minimum_host_available_bytes','oom_events','swap_peak_bytes','limit_pressure_events',
    'worker_uuid_pairs','h200_uuids','native_assignment_receipts','learner_receipts','completed',
    'allocation_retained','checkpoint_optimizer_lineage_verified'}
RESOURCE_V2_FIELDS={'cache_reclaim_events','unsafe_pressure_events','phase_limit_pressure_events',
    'phase_cache_reclaim_events','phase_unsafe_pressure_events','resource_guards'}

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def require(value,message):
    if not value:raise ValueError(message)


def aggregate_resource_guards(rows,*,per_gpu):
    """Preserve absolute raw counters without double-counting repeated phases.

    Absolute counters are each GPU's observed maximum, summed across GPUs.
    Phase deltas are summed across actual phases. Every number is reproduced
    from bound actual versioned guard summaries; classification never hides raw max.
    """
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import validate_summary
    require(type(per_gpu)is int and per_gpu>=1 and isinstance(rows,list) and len(rows)==2*per_gpu,
            'exact actual guard count per physical GPU required')
    grouped={0:[],1:[]};paths=set();hashes=set();versions=set()
    for row in rows:
        require(isinstance(row,dict) and set(row)=={'physical_gpu','summary'}
            and type(row['physical_gpu'])is int and row['physical_gpu'] in grouped,
            'guard summary must bind one exact physical GPU')
        guard=row['summary'];validate_summary(guard)
        require(guard.get('schema') in ('roborender.native-phase-resource-guard.v2','roborender.native-phase-resource-guard.v3'),
            'aggregate requires actual versioned memory-policy proof')
        versions.add(guard['schema'])
        path,sha=guard.get('path'),guard.get('raw_sha256')
        require(isinstance(path,str) and Path(path).is_absolute() and '..' not in Path(path).parts
            and isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),
            'actual guard path and raw file digest required')
        require(path not in paths and sha not in hashes,'duplicate actual resource guard evidence')
        paths.add(path);hashes.add(sha);grouped[row['physical_gpu']].append(guard)
    require(len(versions)==1,'mixed memory policies cannot form one qualification')
    require(all(len(group)==per_gpu for group in grouped.values()),'guard GPU cardinality differs')
    for group in grouped.values():
        for before,after in zip(group,group[1:]):
            require(after['initial_limit_pressure_events']>=before['limit_pressure_events'],
                    'owned raw resource counter moved backwards across phases')
    guards=[r['summary'] for r in rows]
    totals=dict(minimum_host_available_bytes=min(g['minimum_host_available_bytes'] for g in guards),
        swap_peak_bytes=max(g['swap_peak_bytes'] for g in guards),resource_guards=deepcopy(rows))
    for key in ('oom_events','limit_pressure_events','cache_reclaim_events','unsafe_pressure_events'):
        totals[key]=sum(max(g[key] for g in group) for group in grouped.values())
    for key in ('phase_limit_pressure_events','phase_cache_reclaim_events','phase_unsafe_pressure_events'):
        totals[key]=sum(g[key] for g in guards)
    return totals


def validate_resource_aggregate(receipt,*,per_gpu,v2):
    if not v2:
        require(receipt['minimum_host_available_bytes']>=8<<30 and receipt['oom_events']==0
            and receipt['swap_peak_bytes']==0 and receipt['limit_pressure_events']==0,
            'legacy raw-max-zero resource contract failed')
        return receipt
    expected=aggregate_resource_guards(receipt['resource_guards'],per_gpu=per_gpu)
    require(all(receipt.get(key)==value for key,value in expected.items()),
            'aggregate resource counters do not reproduce actual v2 guard summaries')
    return receipt

def make_plan(*,run_id,inventory,panel,excluded_process_seeds,seed,motion_root,
              source_sha256,teacher_sha256,policy_spec_sha256,stopping=None,
              environments_per_actor=None,fixed100=False,continuous_round=False,h200_workers=4):
    require(type(fixed100)is bool,'fixed100 opt-in must be explicit boolean')
    require(type(continuous_round)is bool and (not continuous_round or fixed100),
            'continuous native rounds require explicit fixed100 mode')
    require(type(h200_workers)is int and h200_workers in (4,8) and (h200_workers==4 or continuous_round),
            'eight H200 workers require the explicit continuous v2 plan')
    if fixed100:
        require(environments_per_actor==16 and stopping is None,
                'fixed100 requires B16 perRTX and forbids stopping/patience configuration')
    scaled=environments_per_actor is not None
    batch=2 if not scaled else environments_per_actor
    require(type(batch)is int and batch in (2,4,8,16,32,64),'explicit bounded per-RTX environment batch required')
    schema=((CONTINUOUS_FIXED100_8H200_SCHEMA if h200_workers==8 else CONTINUOUS_FIXED100_SCHEMA) if continuous_round else FIXED100_SCHEMA if fixed100
            else SCALED_SCHEMA if scaled else SCHEMA)
    total=2*batch;cohort_count=math.ceil(295/total);updates=2*cohort_count
    inv=bound(inventory);test_value=bound(panel)
    from pipeline.deployment.rtx_actor.scaled_success_contract import is_panel
    if is_panel(test_value):
        require(continuous_round and h200_workers==8,
                'paired scaled final-test panel requires continuous eight-H200 production')
        from pipeline.deployment.paired_scaled_evaluation import final_panel
        test=final_panel(test_value)
    else:
        test=validate_panel(test_value)
    rows=inv.get('motions',[])
    require(inv.get('schema')=='roborender.motion-inventory.v1' and len(rows)==295
        and inv.get('motion_count')==295 and inv.get('motion_records_sha256')==digest(rows),
        'exact295 motion inventory required')
    require(len({r['motion_id'] for r in rows})==295 and len({r['sha256'] for r in rows})==295,
            'unique295 motion identities required')
    support={r['motion_id']:r['sha256'] for r in rows}
    test_support={r['motion_id']:r['sha256'] for c in test['cohorts'] for r in c['assignment']['lanes']}
    test_counts=Counter(r['motion_id'] for c in test['cohorts'] for r in c['assignment']['lanes'])
    require(test.get('purpose')=='test' and test.get('episode_count')==295
        and test_support==support and set(test_counts.values())=={1},'fixed one-case-per-motion test required')
    require(isinstance(run_id,str) and run_id.startswith('dagger-leo-') and len(run_id)<=60,'bounded Leo run ID required')
    require(type(seed)is int and 0<=seed<2**31 and Path(motion_root).is_absolute(),'explicit seed/motion root required')
    for sha in (source_sha256,teacher_sha256,policy_spec_sha256):
        require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'asset SHA required')
    rule=deepcopy(STOPPING if stopping is None else stopping)
    if fixed100:
        rule['maximum_coverage_epochs']=5
    if scaled and not fixed100:
        require(rule.get('maximum_coverage_epochs')==10,'scaled production retains the ten-coverage-epoch ceiling')
    require(set(rule)==set(STOPPING) and 1<=rule['minimum_coverage_epochs']<=rule['maximum_coverage_epochs']<=40
        and type(rule['patience'])is int and rule['patience']>=1
        and all(type(rule[k])in(int,float) and math.isfinite(rule[k]) and rule[k]>=0
                for k in ('relative_improvement','absolute_improvement')),'explicit bounded stopping policy required')
    used=set(excluded_process_seeds)
    require(all(type(s)is int and 0<=s<2**31 for s in used),'invalid excluded seed')
    used.update(c['assignment']['simulator_seed'] for c in test['cohorts'])
    used.update(seed+100003*(rid-1)+gpu for rid in range(1,5) for gpu in (0,1))
    original_excluded=sorted(used)
    def fresh(role,rid,epoch,index,gpu):
        candidate=int.from_bytes(hashlib.sha256(canonical([schema,seed,role,rid,epoch,index,gpu])).digest()[:8],'little')%(2**31-1)
        require(candidate not in used,'seed collision: choose a new master seed before freezing')
        used.add(candidate);return candidate
    def assignment(chosen,*,purpose,rid,epoch,index,gpu):
        process_seed=fresh(purpose,rid,epoch,index,gpu)
        return validate_motion_assignment(dict(schema='roborender.native-motion-cohort.v1',
            run_id=run_id if purpose=='training' else f'heldout-{run_id}-r{rid}',
            cohort_id=f'{purpose}-r{rid}-e{epoch:02d}-c{index:03d}-g{gpu}',purpose=purpose,
            simulator_seed=process_seed,motion_inventory_sha256=digest(inv),
            lanes=[dict(lane=i,motion_id=r['motion_id'],sha256=r['sha256'],path=str(Path(motion_root)/r['filename']))
                   for i,r in enumerate(chosen)]))
    rounds=[]
    for rid,beta in enumerate(BETAS,1):
        ordered=sorted(rows,key=lambda r:digest([schema,seed,'training',rid,r['motion_id']]))
        padding=ordered[0]
        epochs=[]
        for epoch in range(1,1 if continuous_round else rule['maximum_coverage_epochs']+1):
            # Full-batch normalization requires complete batches. Declare every
            # padded slot; never silently drop motions or change test cases.
            extra=cohort_count*total-295
            # New scaled plans rotate the padded slice between epochs, so
            # full-batch normalization does not repeatedly oversample the same
            # motions. Preserve the historical v1 schedule byte-for-byte.
            offset=((epoch-1)*extra)%295 if scaled else 0
            epoch_order=ordered[offset:]+ordered[:offset]
            selected=epoch_order+epoch_order[:extra];cohorts=[]
            for i,start in enumerate(range(0,len(selected),total)):
                group=selected[start:start+total]
                pair=[assignment(group[gpu*batch:gpu*batch+batch],purpose='training',rid=rid,epoch=epoch,index=i,gpu=gpu)
                      for gpu in (0,1)]
                global_start=(epoch-1)*updates+i*2
                cohorts.append(dict(cohort_index=i,start_iteration=global_start,window_count=2,
                    assignments=pair,padding_slots=[dict(physical_gpu=j//batch,lane=j%batch,motion_id=r['motion_id'])
                        for j,r in enumerate(group) if start+j>=295],
                    reset_at_window_boundary=False,scene_change='explicit_new_native_lifecycle_at_cohort_boundary'))
            epochs.append(dict(coverage_epoch=epoch,cohorts=cohorts,optimizer_updates=updates,
                full_unique_motion_count=295,padding_motion_id=epoch_order[0]['motion_id'] if scaled else padding['motion_id']))
        val_order=sorted(rows,key=lambda r:digest([schema,seed,'validation',rid,r['motion_id']]))
        val_rows=val_order+val_order[:25];heldout=[]
        # Preserve all 320 held-out chunks. Bound collection batches to an
        # exact divisor of 320 and never exceed the qualified training batch.
        validation_batch=min(batch,16) if scaled else 2
        for i,start in enumerate(range(0,0 if fixed100 else 320,validation_batch)):
            gpu=i%2
            native=assignment(val_rows[start:start+validation_batch],purpose='validation',rid=rid,epoch=0,index=i,gpu=gpu)
            heldout.append(dict(cohort_id=native['cohort_id'],physical_gpu_index=gpu,
                simulator_seed=native['simulator_seed'],lanes=native['lanes']))
        rounds.append(dict(round_id=rid,teacher_beta=beta,student_probability=1-beta,
            wandb_project='dagger',wandb_run_name=f"{run_id.removeprefix('dagger-leo-') if scaled else '2026-10-16-prod-1'}-dagger-teacher{int(beta*100)}",
            wandb_run_id='prod1-'+digest([run_id,rid])[:24],epochs=epochs,
            parent_binding='BC_selected_policy_and_optimizer' if rid==1 else 'previous_round_absolute_minimum_MSE_policy_and_optimizer',
            heldout=dict(chunks=320,unique_motions=295,cohorts=heldout,
                checkpoint_binding='round_parent_frozen_before_training',parent_occupancy=True,
                policy_beta=beta,learner_submission_authorized=False),
            success_test=dict(panel=panel,completed_episodes=295,checkpoint_binding='absolute_minimum_heldout_MSE',
                test_outcomes_select_checkpoint=False,optimizer_steps=0)))
        if fixed100:
            rounds[-1].pop('heldout')
            rounds[-1]['parent_binding']='BC_selected_policy_and_optimizer' if rid==1 else 'previous_round_update100_policy_and_optimizer'
            rounds[-1]['success_test']['checkpoint_binding']='final_update_100'
        if continuous_round:
            # Match Raven's fixed scene/lane motion support for a whole round.
            # Natural episode resets may sample a new reference position within
            # that lane, but there is no scheduled scene or motion replacement.
            # This deliberately does NOT promise all295 training-motion coverage.
            pair=[assignment(rows[gpu*batch:(gpu+1)*batch],purpose='training',rid=rid,
                             epoch=0,index=0,gpu=gpu) for gpu in (0,1)]
            for gpu,native in enumerate(pair):
                native['cohort_id']=f'training-r{rid}-continuous-g{gpu}'
                validate_motion_assignment(native)
            rounds[-1].pop('epochs')
            rounds[-1]['native_round']=dict(cohort_id=f'training-r{rid}-continuous',
                start_iteration=0,window_count=100,qualification_after_updates=4,
                assignments=pair,reset_at_window_boundary=False,
                lifecycle_policy='native_episode_done_only',
                model_rollout_lifecycle='per_81_step_window')
    value=dict(schema=schema,run_id=run_id,inventory=inventory,panel=panel,seed=seed,motion_root=str(motion_root),
        source_sha256=source_sha256,teacher_sha256=teacher_sha256,policy_spec_sha256=policy_spec_sha256,
        topology={**TOPOLOGY,'environments_per_actor':batch,'total_environments':total},stopping=rule,excluded_process_seeds=original_excluded,
        frames_per_window=81,windows_per_native_cohort=2,cohorts_per_coverage_epoch=cohort_count,
        updates_per_coverage_epoch=updates,max_updates_per_round=updates*rule['maximum_coverage_epochs'],
        train_frames_per_update=total*81,train_action_dimensions=11,optimizer=f'Raven_AdamW_full{total}x81',
        learning_rate=.0002,max_gradient_norm=1.,previous_round_replay=False,
        observation_source='native_episode_50pct_raytraced_50pct_Image3F',
        online_renderer=dict(inference_steps=5,cfg_scale=1.,teacache_threshold=.3),
        augmentation='pinned_Raven_raytraced_native',random_initial_episode_lengths=True,
        stopping_only_after_complete_coverage_epoch=True,rounds=rounds,
        qualification_required='two_real_cohorts_two_continuous_windows_each_before_full_production',
        algorithm_extension='bounded_motion_cohort_lifecycles; ordinary81window_does_not_reset_native_history',
        production_admitted=False,planning_only=True)
    if fixed100:
        value.pop('stopping')
        value.pop('stopping_only_after_complete_coverage_epoch')
        value.update(maximum_coverage_epochs=5,optimizer_updates_per_round=100,
            checkpoint_selection='final_update_100',validation_stage=False,
            observation_source='online_Image3F_generated_only',augmentation='none_on_generated_policy_RGB',
            qualification_updates_count_toward_round_budget=True)
    if continuous_round:
        for key in ('maximum_coverage_epochs','windows_per_native_cohort',
                    'cohorts_per_coverage_epoch','updates_per_coverage_epoch'):
            value.pop(key)
        value.update(native_lifecycles_per_actor_per_round=1,
            initial_motion_support=dict(selection='first32_inventory_ordered',
                motion_ids=[r['motion_id'] for r in rows[:total]],
                motion_count=total,full_inventory_coverage_guaranteed=False),
            qualification_required='first_four_real_updates_in_same_native_round_before_update5',
            algorithm_extension='fixed32_motion_support_continuous_native_round_generated_only')
    if h200_workers==8:
        value['topology'].update(h200_workers=8,h200_shape='custom.gpu.8xh200',
            actor_gateway_slots={'0':[0,1,2,3],'1':[4,5,6,7]})
    return value

def validate_plan(plan):
    require(isinstance(plan,dict) and plan.get('schema') in (SCHEMA,SCALED_SCHEMA,FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS),'new production plan schema required')
    fields=('run_id','inventory','panel','excluded_process_seeds','seed','motion_root',
            'source_sha256','teacher_sha256','policy_spec_sha256','stopping')
    kwargs={'environments_per_actor':plan['topology']['environments_per_actor']} if plan['schema'] in (SCALED_SCHEMA,FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS) else {}
    if plan['schema'] in (FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS):
        fields=fields[:-1]
        kwargs['fixed100']=True
        kwargs['continuous_round']=is_continuous_plan(plan)
    if is_eight_h200_plan(plan):kwargs['h200_workers']=8
    expected=make_plan(**{k:plan[k] for k in fields},**kwargs)
    require(plan==expected,'production plan differs from frozen deterministic recipe')
    return plan

def load_plan(ref):
    return validate_plan(bound(ref))

def round_training_seeds(config):
    plan=load_plan(config['coverage_plan'])
    if is_continuous_plan(plan):
        return sorted(a['simulator_seed'] for a in plan['rounds'][config['round_id']-1]['native_round']['assignments'])
    return sorted(a['simulator_seed'] for e in plan['rounds'][config['round_id']-1]['epochs']
                  for c in e['cohorts'] for a in c['assignments'])

def make_continuous_segment(plan,*,round_id,input_checkpoint_sha256,
                            mse_protocol_file_sha256,motion_assignments,coverage_plan_sha256):
    """One source-bound native command, never a sequence of reset cohorts."""
    validate_plan(plan)
    require(is_continuous_plan(plan) and type(round_id)is int and 1<=round_id<=4,
            'continuous fixed100 plan and round required')
    for sha in (input_checkpoint_sha256,mse_protocol_file_sha256,coverage_plan_sha256):
        require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),
                'continuous native command requires exact artifact SHA bindings')
    native=plan['rounds'][round_id-1]['native_round']
    require([row['physical_gpu_index'] for row in motion_assignments]==[0,1],
            'both exact continuous motion bindings required')
    for row,expected in zip(motion_assignments,native['assignments']):
        require(set(row)=={'physical_gpu_index','path','sha256'} and Path(row['path']).is_absolute()
            and row['sha256']==digest(expected),'continuous assignment differs from frozen plan')
    return dict(schema='roborender.dagger-continuous-round.v2' if is_eight_h200_plan(plan) else 'roborender.dagger-continuous-round.v1',run_id=plan['run_id'],round_id=round_id,
        cohort_id=native['cohort_id'],episode_namespace=f'continuous-r{round_id}',
        start_iteration_index=0,window_count=100,input_checkpoint_sha256=input_checkpoint_sha256,
        mse_protocol_file_sha256=mse_protocol_file_sha256,motion_assignments=motion_assignments,
        coverage_plan_sha256=coverage_plan_sha256,lifecycle_policy='native_episode_done_only',
        qualification_after_updates=4)

def make_segment(plan,*,round_id,epoch,index,input_checkpoint_sha256,mse_protocol_file_sha256,
                 motion_assignments,coverage_plan_sha256):
    require(not is_continuous_plan(plan),
            'continuous rounds cannot use reset-cohort segment dispatch')
    cohort=plan['rounds'][round_id-1]['epochs'][epoch-1]['cohorts'][index]
    require([row['physical_gpu_index'] for row in motion_assignments]==[0,1],
            'both exact production motion bindings required')
    for row,expected in zip(motion_assignments,cohort['assignments']):
        require(set(row)=={'physical_gpu_index','path','sha256'} and Path(row['path']).is_absolute()
            and row['sha256']==digest(expected),'production assignment differs from coverage plan')
    return dict(schema='roborender.dagger-production-segment.v1',run_id=plan['run_id'],round_id=round_id,
        coverage_epoch=epoch,cohort_index=index,cohort_id=f'training-r{round_id}-e{epoch:02d}-c{index:03d}',
        episode_namespace=f'prod-r{round_id}-e{epoch:02d}-c{index:03d}',
        start_iteration_index=cohort['start_iteration'],window_count=2,
        input_checkpoint_sha256=input_checkpoint_sha256,mse_protocol_file_sha256=mse_protocol_file_sha256,
        motion_assignments=motion_assignments,coverage_plan_sha256=coverage_plan_sha256)

def validate_qualification(plan,receipt,parent_sha):
    if is_continuous_plan(plan):
        return validate_continuous_qualification(plan,receipt,parent_sha,
                                                round_id=receipt.get('round_id',1))
    total=plan['topology']['total_environments'];frames=plan['train_frames_per_update']
    v2=receipt.get('schema')=='roborender.dagger-full295-two-cohort-qualification.v2'
    require(set(receipt)==QUALIFICATION_FIELDS|(RESOURCE_V2_FIELDS if v2 else set()),'exact genuine qualification receipt required')
    require(receipt['schema'] in ('roborender.dagger-full295-two-cohort-qualification.v1','roborender.dagger-full295-two-cohort-qualification.v2')
        and receipt['purpose']=='qualification' and receipt['completed']is True
        and receipt['plan_sha256']==digest(plan) and receipt['source_sha256']==plan['source_sha256']
        and receipt['parent_checkpoint_sha256']==parent_sha,'qualification identity differs')
    require(receipt['native_cohorts']==2 and receipt['optimizer_updates']==4
        and receipt['native_initializations_per_cohort']==[1,1]
        and receipt['vector_steps_per_cohort']==[162,162] and receipt['transitions']==4*frames
        and receipt['full_batch_environments']==total and receipt['no_replay']is True
        and receipt['history_reset_at_window_boundary']is False
        and receipt['checkpoint_optimizer_lineage_verified']is True,'real fullbatch native continuity qualification required')
    validate_resource_aggregate(receipt,per_gpu=2,v2=v2)
    require(len(receipt['worker_uuid_pairs'])==2 and len(set(receipt['h200_uuids']))==4
        and len(receipt['native_assignment_receipts'])>=2*total and len(receipt['learner_receipts'])==4
        and receipt['allocation_retained']is True,'native/learner/allocation evidence incomplete')
    # Evidence is referenced by byte identity; a Boolean PASS cannot stand in
    # for the actual native owners, learner outputs, and allocation observations.
    expected={(a['simulator_seed'],lane['motion_id'],lane['sha256']):a
              for c in plan['rounds'][0]['epochs'][0]['cohorts'][:2]
              for a in c['assignments'] for lane in a['lanes']}
    observed=set()
    for ref in receipt['native_assignment_receipts']:
        require(set(ref)=={'path','raw_sha256','reset_evidence'},'exact native/reset evidence binding required')
        row=bound({k:ref[k] for k in ('path','raw_sha256')});reset=bound(ref['reset_evidence'])
        key=(row['simulator_seed'],row['motion_id'],row['motion_sha256'])
        require(row.get('schema')=='roborender.native-episode-assignment.v1'
            and row.get('purpose')=='training' and key in expected
            and reset.get('schema')=='roborender.native-episode-reset-evidence.v1'
            and reset.get('episode_id')==row['episode_id']
            and reset.get('assignment_sha256')==digest(expected[key]),
            'qualification is missing actual native motion/reset evidence')
        observed.add(key)
    require(observed==set(expected),'qualification did not cover all firsttwo nativecohort lanes')
    parent=parent_sha;global_step=None
    for ref in receipt['learner_receipts']:
        row=bound(ref)
        require(row.get('schema')=='roborender.dagger_learner_candidate_receipt.v3'
            and row.get('parent_checkpoint_sha256')==parent and row.get('train_chunks')==total
            and row.get('train_frames')==frames and row.get('replay_manifest_sha256')is None
            and row.get('batching_semantics')=='raven_exact_flattened_bxt'
            and row.get('optimizer_state_preserved')is True
            and row.get('global_step')==row.get('parent_global_step',-2)+1,
            'qualification optimizer/fullbatch evidence is incomplete')
        if global_step is not None:require(row['parent_global_step']==global_step,'qualification AdamW steps discontinuous')
        global_step=row['global_step'];parent=row['output_checkpoint_sha256']
    return receipt


def validate_continuous_native_receipt(plan,ref,*,round_id,count,phase):
    """Validate bound native evidence, not an observer's inferred liveness."""
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import validate_summary
    row=bound(ref)
    fields={'schema','physical_gpu','worker_instance_id','native_pid','command_id','source_sha256',
            'segment_sha256','motion_assignment_sha256','actor_state','resource_guard'}
    expected_schema=('roborender.dagger-continuous-native-prefix.v1' if phase=='WAITING_FOR_QUALIFICATION'
                     else 'roborender.dagger-continuous-native-terminal.v1')
    if phase == 'COMPLETE' and row.get('schema') == 'roborender.dagger-continuous-native-terminal.v2':
        expected_schema = row['schema']
        fields |= {'training_curriculum', 'training_curriculum_sha256', 'native_terminal_receipt_sha256',
                   'training_curriculum_initialization'}
        from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import build_restore_request
        build_restore_request(row)
    if 'population_curriculum' in row:
        from pipeline.dagger.rtx.code.src.dagger_rtx.population_curriculum import validate_progress
        validate_progress(row['population_curriculum'],count)
        fields.add('population_curriculum')
    if 'reset_inventory' in row:
        from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import descriptors, binding
        require(row['reset_inventory']==binding(descriptors(bound(plan['inventory']),plan['motion_root']),
                row['physical_gpu']*16) and row['actor_state'].get('reset_inventory')==row['reset_inventory'],
                'native full reset inventory/order/lane offset differs')
        fields.add('reset_inventory')
    require(phase in ('WAITING_FOR_QUALIFICATION','COMPLETE') and set(row)==fields and row['schema']==expected_schema,
        'exact bound continuous native receipt required')
    gpu=row['physical_gpu'];native=plan['rounds'][round_id-1]['native_round'];state=row['actor_state']
    require(type(gpu)is int and gpu in (0,1) and type(row['native_pid'])is int and row['native_pid']>0
        and isinstance(row['command_id'],str) and row['command_id']
        and isinstance(row['worker_instance_id'],str) and len(row['worker_instance_id'])==32
        and all(c in '0123456789abcdef' for c in row['worker_instance_id'])
        and row['source_sha256']==plan['source_sha256']
        and row['motion_assignment_sha256']==digest(native['assignments'][gpu]),
        'continuous native owner/source/assignment differs')
    require(state.get('phase')==phase and state.get('physical_gpu')==gpu
        and state.get('native_pid')==row['native_pid']
        and state.get('uncertain_reason')is None
        and state.get('completed_windows')==count and state.get('transition_batches')==81*count
        and state.get('physical_transition_rows')==81*count*16
        and state.get('native_environment_initializations')==1
        and state.get('model_rollout_initializations')==count
        and state.get('history_reset_at_window_boundary')is False,
        'continuous native progress/lifecycle proof differs')
    segment=state.get('segment',{})
    require(segment.get('schema')==('roborender.dagger-continuous-round.v2' if is_eight_h200_plan(plan) else 'roborender.dagger-continuous-round.v1')
        and segment.get('run_id')==plan['run_id'] and segment.get('round_id')==round_id
        and segment.get('window_count')==100 and segment.get('start_iteration_index')==0
        and segment.get('qualification_after_updates')==4
        and segment.get('lifecycle_policy')=='native_episode_done_only'
        and state.get('segment_file_sha256')==row['segment_sha256']
        and row['segment_sha256']==digest(segment),
        'continuous native command differs from bound segment')
    require(segment==make_continuous_segment(plan,round_id=round_id,
        input_checkpoint_sha256=segment['input_checkpoint_sha256'],
        mse_protocol_file_sha256=segment['mse_protocol_file_sha256'],
        motion_assignments=segment['motion_assignments'],coverage_plan_sha256=segment['coverage_plan_sha256']),
        'continuous native segment is not the exact frozen plan member')
    windows=state.get('windows',[])
    require(len(windows)==count and [w.get('index') for w in windows]==list(range(count)),
        'continuous native prefix is not contiguous')
    if is_eight_h200_plan(plan):
        from pipeline.dagger.rtx.code.src.dagger_rtx.episode_continuity import validate_window
        previous=None
        for index,window in enumerate(windows):
            evidence=validate_window(window.get('episode_continuity'),batch=16,index=index,previous_end=previous)
            previous=evidence['end']
    validate_summary(row['resource_guard'])
    return row


def validate_continuous_qualification(plan,receipt,parent_sha,*,round_id=1):
    """Actual first4 updates, paused in one still-active native command pair."""
    validate_plan(plan)
    fields={'schema','purpose','run_id','round_id','plan_sha256','parent_checkpoint_sha256',
            'source_sha256','optimizer_updates','full_batch_environments','native_prefix_receipts',
            'learner_receipts','allocation_retained','h200_uuids','native_assignment_receipts',
            'online_metrics_verified','online_metrics_receipt'}
    if is_eight_h200_plan(plan):fields |= {'runtime_config','h200_gateway_identities','renderer_attachment'}
    require(is_continuous_plan(plan) and set(receipt)==fields
        and receipt.get('schema')=='roborender.dagger-continuous-prefix-qualification.v1'
        and receipt.get('purpose')=='qualification' and receipt.get('run_id')==plan['run_id']
        and receipt.get('round_id')==round_id and type(round_id)is int and 1<=round_id<=4
        and receipt.get('plan_sha256')==digest(plan) and receipt.get('source_sha256')==plan['source_sha256']
        and receipt.get('parent_checkpoint_sha256')==parent_sha
        and receipt.get('optimizer_updates')==4 and receipt.get('full_batch_environments')==32
        and receipt.get('allocation_retained')is True,
        'exact current continuous-prefix qualification required; legacy cohorts cannot qualify it')
    require(len(receipt['native_prefix_receipts'])==2 and len(receipt['learner_receipts'])==4
        and len(receipt['h200_uuids'])==plan['topology']['h200_workers']
        and len(set(receipt['h200_uuids']))==len(receipt['h200_uuids']),
        'continuous native/learner/allocation evidence cardinality differs')
    native=[validate_continuous_native_receipt(plan,ref,round_id=round_id,count=4,
                phase='WAITING_FOR_QUALIFICATION') for ref in receipt['native_prefix_receipts']]
    require({r['physical_gpu'] for r in native}=={0,1}
        and len({r['worker_instance_id'] for r in native})==2
        and len({r['command_id'] for r in native})==2
        and len({r['segment_sha256'] for r in native})==1
        and all(r['actor_state']['segment']['input_checkpoint_sha256']==parent_sha for r in native),
        'continuous qualification did not use one exact paired native lifecycle')
    config={}
    if is_eight_h200_plan(plan):
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
            validate_config,is_eight_h200,gateway_identities_for_actor)
        config=validate_config(bound(receipt['runtime_config']),runtime=True)
        if config.get('reset_inventory_protocol') is not None:
            require(all('reset_inventory' in n for n in native),'full reset inventory proof is missing')
        if config.get('population_curriculum_protocol') is not None:
            from pipeline.dagger.rtx.code.src.dagger_rtx.population_curriculum import validate_progress
            populations=[validate_progress(n.get('population_curriculum'),4) for n in native]
            require(populations[0]==populations[1], 'paired curriculum proof differs between native actors')
        require(is_eight_h200(config) and config['source_sha256']==plan['source_sha256']
            and config['run_id']==plan['run_id'] and config['round_id']==round_id
            and load_plan(config['coverage_plan'])==plan
            and config['student']['initial_checkpoint_sha256']==parent_sha
            and receipt['h200_gateway_identities']==config['h200']['gateway_identities']
            and receipt['h200_uuids']==[r['gpu_uuid'] for r in receipt['h200_gateway_identities']]
            and all(n['actor_state'].get('h200_gateway_identities')==
                gateway_identities_for_actor(config,n['physical_gpu'])
                and n['actor_state'].get('config_sha256')==digest(config) for n in native),
            'actual eight-H200 fleet/exclusive native route/source binding differs')
        attachment=bound(receipt['renderer_attachment']);renderers=attachment.get('renderers',[])
        require(attachment.get('schema')=='roborender.retained-eight-renderer-attachment.v1'
            and attachment.get('allocation_retained')is True and attachment.get('compute_release_authorized')is False
            and attachment.get('render_requests')==0 and attachment.get('renderer_restarts')==0
            and all(isinstance(attachment.get(k),str) and attachment[k] for k in
                ('allocation_name','allocation_owner','hostname','node_id','boot_id'))
            and len(renderers)==8 and [r.get('slot') for r in renderers]==list(range(8))
            and [r.get('gpu_uuid') for r in renderers]==receipt['h200_uuids']
            and len({r.get('pid') for r in renderers})==8
            and all(type(r.get('pid'))is int and r['pid']>0
                and r.get('identity',{}).get('boot_id')==attachment['boot_id']
                and all(isinstance(r.get(k),str) and len(r[k])==64
                    and all(c in '0123456789abcdef' for c in r[k])
                    for k in ('source_sha256','generate_source_sha256')) for r in renderers),
            'fresh exact retained-eight renderer process/source/GPU attachment required')
    expected={(a['simulator_seed'],lane['motion_id'],lane['sha256']):a
        for a in plan['rounds'][round_id-1]['native_round']['assignments'] for lane in a['lanes']}
    observed=set()
    for ref in receipt['native_assignment_receipts']:
        require(set(ref)=={'path','raw_sha256','reset_evidence'},'exact native/reset evidence binding required')
        assignment=bound({k:ref[k] for k in ('path','raw_sha256')});reset=bound(ref['reset_evidence'])
        key=(assignment['simulator_seed'],assignment['motion_id'],assignment['motion_sha256'])
        require(assignment.get('schema')=='roborender.native-episode-assignment.v1'
            and assignment.get('purpose')=='training' and key in expected
            and reset.get('schema')=='roborender.native-episode-reset-evidence.v1'
            and reset.get('episode_id')==assignment['episode_id']
            and reset.get('assignment_sha256')==digest(expected[key]),
            'continuous qualification is missing actual initial native motion/reset evidence')
        observed.add(key)
    require(observed==set(expected),'continuous qualification did not cover all32 native lanes')
    parent=parent_sha;step=None
    for index,ref in enumerate(receipt['learner_receipts']):
        row=bound(ref)
        require(row.get('schema')=='roborender.dagger_learner_candidate_receipt.v3'
            and row.get('parent_checkpoint_sha256')==parent and row.get('train_chunks')==32
            and row.get('train_frames')==2592 and row.get('replay_manifest_sha256')is None
            and row.get('batching_semantics')=='raven_exact_flattened_bxt'
            and row.get('optimizer_state_preserved')is True
            and row.get('iteration_id')==index+1
            and type(row.get('train_mse')) in (int,float) and math.isfinite(row['train_mse'])
            and row['train_mse']>=0
            and row.get('global_step')==row.get('parent_global_step',-2)+1
            and (step is None or row['parent_global_step']==step),
            'continuous qualification optimizer/fullbatch evidence differs')
        candidate=row['output_checkpoint_sha256']
        if config.get('learner_protocol') is not None:
            from pipeline.dagger.services.learner_order import validate_receipt
            ids=row.get('train_chunk_ids')
            require(isinstance(ids,list) and len(ids)==32 and len(set(ids))==32,
                    'new learner qualification needs actual lane-ordered chunk identities')
            validate_receipt(row.get('learner_order_receipt'),dict(train_chunk_ids=ids))
            rng=row.get('learner_rng_receipt') or {}
            require(rng.get('schema')=='roborender.learner-rng-continuation.v1'
                    and rng.get('global_step')==row['global_step']
                    and rng.get('parent_checkpoint_sha256')==parent,
                    'new learner qualification lacks continued RNG evidence')
        require(candidate!=parent and all(
            n['actor_state']['windows'][index].get('input_checkpoint_sha256')==parent
            and n['actor_state']['windows'][index].get('output_checkpoint_sha256')==candidate
            for n in native),'continuous native and learner prefix lineages differ')
        step=row['global_step'];parent=candidate
    metrics=bound(receipt['online_metrics_receipt'])
    require(receipt['online_metrics_verified']is True
        and metrics.get('schema')=='roborender.dagger-continuous-online-prefix.v1'
        and metrics.get('run_id')==plan['run_id'] and metrics.get('round_id')==round_id
        and metrics.get('wandb_run_id')==plan['rounds'][round_id-1]['wandb_run_id']
        and metrics.get('source_sha256')==plan['source_sha256']
        and metrics.get('checkpoint_sha256')==parent and metrics.get('iteration_id')==4
        and metrics.get('train_mse')==row.get('train_mse')
        and metrics.get('online_metrics_verified')is True,
        'continuous qualification requires actual W&B prefix readback')
    return receipt

def validate_segment(plan,round_id,epoch,index,parent,receipt):
    total=plan['topology']['total_environments'];frames=plan['train_frames_per_update']
    cohort=plan['rounds'][round_id-1]['epochs'][epoch-1]['cohorts'][index]
    fields={'schema','run_id','round_id','coverage_epoch','cohort_index','plan_sha256','source_sha256',
        'native_complete','native_quiescent','native_lifecycle_initializations_per_actor','history_reset_at_window_boundary',
        'assignments','windows','minimum_host_available_bytes','oom_events','swap_peak_bytes','limit_pressure_events',
        'allocation_retained','online_metrics_verified'}
    v2=receipt.get('schema')=='roborender.dagger-production-segment-result.v2'
    require(set(receipt)==fields|(RESOURCE_V2_FIELDS if v2 else set())
        and receipt['schema'] in ('roborender.dagger-production-segment-result.v1','roborender.dagger-production-segment-result.v2'),
            'exact native segment result required')
    require((receipt['run_id'],receipt['round_id'],receipt['coverage_epoch'],receipt['cohort_index'])
        ==(plan['run_id'],round_id,epoch,index) and receipt['plan_sha256']==digest(plan)
        and receipt['source_sha256']==plan['source_sha256'],'segment identity differs')
    require(receipt['native_complete']is True and receipt['native_quiescent']is True
        and receipt['native_lifecycle_initializations_per_actor']==[1,1]
        and receipt['history_reset_at_window_boundary']is False
        and receipt['assignments']==[digest(a) for a in cohort['assignments']],
        'native assignment/lifecycle continuity differs')
    validate_resource_aggregate(receipt,per_gpu=1,v2=v2)
    require(receipt['allocation_retained']is True and type(receipt['online_metrics_verified'])is bool,
        'segment resource or logging status is invalid')
    rows=receipt['windows'];require(len(rows)==2,'exact two complete continuous windows required')
    current=deepcopy(parent)
    for offset,row in enumerate(rows):
        require(row['iteration_id']==cohort['start_iteration']+offset+1
            and row['input_checkpoint_sha256']==current['sha256'] and row['transition_batches']==81
            and row['train_chunks']==total and row['train_frames']==frames and row['optimizer_steps']==1
            and row['full_batch_environments']==total and row['replay_manifest_sha256']is None
            and row['teacher_beta']==BETAS[round_id-1] and row['student_action_dimensions']==11,
            'learner update/count/parent/beta differs')
        require(type(row['train_mse'])in(int,float) and math.isfinite(row['train_mse']) and row['train_mse']>=0,
            'actual finite train MSE required')
        require(row['output_checkpoint']['optimizer_global_step']==current['optimizer_global_step']+1
            and row['output_checkpoint']['optimizer_state_preserved']is True,'AdamW lineage differs')
        current=deepcopy(row['output_checkpoint'])
    return current

def empty_training_provenance(round_id):
    return dict(purpose='train',round_id=round_id,chunk_sha256s=[],source_episode_ids=[],source_window_ids=[])

def merge_provenance(old,new):
    require(set(old)==set(new) and old['round_id']==new['round_id'] and new['purpose']=='train','round provenance differs')
    return {**old,**{k:sorted(set(old[k])|set(new[k])) for k in ('chunk_sha256s','source_episode_ids','source_window_ids')}}

def make_first_config(plan,*,coverage_plan,checkpoint,checkpoint_sha256,gateway_pids=(),gateway_identities=(),rendered_observation_config=None):
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
        LEO_MSE_PRODUCTION_SCHEMA,LEO_MSE_SCALED_SCHEMA,LEO_FIXED100_SCHEMA,LEO_GPU_UUIDS,FIDELITY,validate_config,
    )
    test=bound(plan['panel'])
    from pipeline.deployment.rendered_success_dispatch import configuration
    configuration(rendered_observation_config, test, checkpoint_sha256)
    continuous=is_continuous_plan(plan)
    fixed=plan['schema'] in (FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS)
    fixed_schema=LEO_FIXED100_SCHEMA
    if continuous:
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_CONTINUOUS_FIXED100_SCHEMA
        fixed_schema=LEO_CONTINUOUS_FIXED100_SCHEMA
        if is_eight_h200_plan(plan):
            from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_CONTINUOUS_FIXED100_8H200_SCHEMA
            fixed_schema=LEO_CONTINUOUS_FIXED100_8H200_SCHEMA
    budget=plan['max_updates_per_round'] if plan['schema'] in (SCALED_SCHEMA,FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS) else 1480
    value=dict(schema=fixed_schema if fixed else LEO_MSE_SCALED_SCHEMA if plan['schema']==SCALED_SCHEMA else LEO_MSE_PRODUCTION_SCHEMA,run_id=plan['run_id'],round_id=1,
        teacher_beta=.75,student_action_probability=.25,seed=plan['seed'],source_sha256=plan['source_sha256'],
        rtx_actors=[dict(physical_gpu_index=gpu,gpu_uuid=uuid,concurrent_environments=plan['topology']['environments_per_actor'])
                    for gpu,uuid in LEO_GPU_UUIDS.items()],forbidden_gpu_indices=[2],
        raven_fidelity=dict(FIDELITY),replay=dict(enabled=False),coverage_plan=coverage_plan,
        execution=dict(window_count=budget,rollout_steps=81,wait_for_checkpoint_advance=True,
            learner_submission_authorized=True,validation_authorized=True),
        h200=dict(render_student_worker_count=plan['topology']['h200_workers'],boundary_contract='roborender.dagger.rtx.v3',
            resource_shape=plan['topology']['h200_shape'],reservation='cosmos-solution-architect-xcwtffiq',
            allow_burst=False,gateway_pids=list(gateway_pids)),
        student=dict(initial_checkpoint=str(Path(checkpoint).resolve()),initial_checkpoint_sha256=checkpoint_sha256,
            learning_rate=.0002,max_gradient_norm=1.,mini_batches_per_iteration=1,activation_checkpointing=False),
        validation=dict(interval_iterations=budget,completed_episodes=295,metric='raven_goal_reached_rate',
            observation_mode='generated',panel_protocol=test['panel_id'],
            rendered_observation_config=rendered_observation_config,
            scenario_manifest=plan['panel']['path'],scenario_manifest_sha256=plan['panel']['raw_sha256']),
        logging=dict(wandb_project='dagger',wandb_run_name=plan['rounds'][0]['wandb_run_name'],
            wandb_run_id=plan['rounds'][0]['wandb_run_id']))
    if is_eight_h200_plan(plan):
        value['h200'].update(actor_gateway_slots=deepcopy(plan['topology']['actor_gateway_slots']),
                             gateway_identities=list(gateway_identities))
    if fixed:
        from pipeline.dagger.rtx.code.src.dagger_rtx.observation_contract import GENERATED_ONLY_CONTRACT
        value['student_observation']=deepcopy(GENERATED_ONLY_CONTRACT)
    if continuous and is_eight_h200_plan(plan):
        # Bind changed learner semantics into the NEW config digest; never edit
        # an accepted job or silently upgrade an old campaign/checkpoint.
        from pipeline.dagger.services.learner_order import CONFIG_FIELD, PROTOCOL
        value[CONFIG_FIELD] = PROTOCOL
        from pipeline.dagger.rtx.code.src.dagger_rtx.behavior_selection import CONFIG_FIELD, PROTOCOL
        value[CONFIG_FIELD] = PROTOCOL
        from pipeline.dagger.rtx.code.src.dagger_rtx.population_curriculum import CONFIG_FIELD, PROTOCOL
        value[CONFIG_FIELD] = PROTOCOL
        from pipeline.dagger.rtx.code.src.dagger_rtx.reset_inventory import CONFIG_FIELD, PROTOCOL
        value[CONFIG_FIELD] = PROTOCOL
    return validate_config(value)

def stage(plan,*,output_root,checkpoint,checkpoint_sha256,gateway_pids=(),gateway_identities=(),rendered_observation_config=None):
    """Stage real-parent configs; no registration, allocation or GPU execution."""
    from pipeline.dagger.beta_campaign import immutable_json
    from pipeline.dagger.mse_campaign import NATIVE_PLAN_SCHEMA,validate_plan as validate_mse_plan
    validate_plan(plan);root=Path(output_root).resolve()
    coverage=root/'coverage-plan.json';immutable_json(coverage,plan)
    first=make_first_config(plan,coverage_plan=binding(coverage),checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,gateway_pids=gateway_pids,gateway_identities=gateway_identities,
        rendered_observation_config=rendered_observation_config)
    campaign=dict(schema=NATIVE_PLAN_SCHEMA,run_id=plan['run_id'],rounds=[])
    if plan['schema'] in (FIXED100_SCHEMA,*CONTINUOUS_SCHEMAS):
        from pipeline.dagger.services.fixed100_protocol import CAMPAIGN_SCHEMA
        campaign=dict(schema=CAMPAIGN_SCHEMA,run_id=plan['run_id'],
            rounds=[dict(round_id=rid,max_iterations=100,interval_iterations=100,
                         full_batch_environments=32,rollout_steps=81,checkpoint_selection='final_update_100')
                    for rid in range(1,5)])
        validate_mse_plan(campaign,first)
        immutable_json(root/'first-round.json',first);immutable_json(root/'fixed100-plan.json',campaign)
        return dict(coverage_plan=binding(coverage),first_round=binding(root/'first-round.json'),
            mse_plan=binding(root/'fixed100-plan.json'),fixed100_plan=binding(root/'fixed100-plan.json'),
            native_qualification_completed=False,launched=False)
    for row in plan['rounds']:
        rid=row['round_id'];directory=root/f'round-{rid}'
        heldout=dict(schema='roborender.dagger-heldout-plan.v1',parent_run_id=plan['run_id'],
            run_id=f"heldout-{plan['run_id']}-r{rid}",round_id=rid,
            teacher_checkpoint_sha256=plan['teacher_sha256'],policy_spec_sha256=plan['policy_spec_sha256'],
            motion_inventory_sha256=digest(bound(plan['inventory'])),
            motion_support=sorted(r['motion_id'] for r in bound(plan['inventory'])['motions']),
            cohorts=row['heldout']['cohorts'])
        collection=directory/'collection-plan.json';immutable_json(collection,heldout)
        campaign['rounds'].append(dict(round_id=rid,collection_plan_path=str(collection),
            collection_plan_sha256=binding(collection)['raw_sha256'],
            validation_manifest_path=str(directory/'heldout-manifest.json'),validation_manifest_sha256=None,
            validation_admission_path=str(directory/'heldout-admission.json'),validation_admission_sha256=None,
            interval_iterations=plan['updates_per_coverage_epoch'],max_iterations=plan['max_updates_per_round'],
            min_validation_epochs=plan['stopping']['minimum_coverage_epochs'],
            patience=plan['stopping']['patience'],relative_improvement=plan['stopping']['relative_improvement'],
            absolute_improvement=plan['stopping']['absolute_improvement']))
    validate_mse_plan(campaign,first)
    immutable_json(root/'first-round.json',first);immutable_json(root/'mse-plan.json',campaign)
    return dict(coverage_plan=binding(coverage),first_round=binding(root/'first-round.json'),
        mse_plan=binding(root/'mse-plan.json'),native_qualification_completed=False,launched=False)
