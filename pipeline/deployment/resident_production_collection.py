"""Versioned collection dispatcher; two native owners persist in Kit.

Legacy cohorts contain two81-step windows. The new continuous-round schema
delegates to continuous_production_collection for one uninterrupted100-window
native command per actor, never the legacy scene-cohort loop below.
H200 controller runs concurrently. This host operation never allocates/releases
compute and never repeats an uncertain resident command.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from pipeline.dagger.coverage_production import (load_plan,make_segment,digest,require,validate_segment,
    validate_qualification,aggregate_resource_guards)
from pipeline.bc_training.handoff_common import binding,bound
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.services.client import ServiceClient
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical,actor_ids,is_mse_production,is_fixed100
from pipeline.deployment.native_metadata_transfer import fetch_native_json

def completed_prefix(directory,plan,rid,epoch,index,expected_parent,coordinator_rows):
    """Only an immutable, fully reconciled cohort may be skipped on restart."""
    marker=directory/'completion.json'
    if not marker.is_file():
        if (directory/'intent.json').exists():
            raise RuntimeError('cohort intent has no completion; reconcile, never replay native work')
        return None
    saved=json.loads(marker.read_bytes());record=bound(saved['segment_result'])
    native=bound(saved['native_result'])
    require(saved['input_checkpoint_sha256']==expected_parent and saved['native_quiescent']is True,
        'completed cohort parent/quiescence changed')
    parent=dict(sha256=expected_parent,optimizer_global_step=saved['input_optimizer_global_step'])
    output=validate_segment(plan,rid,epoch,index,parent,record)
    require(output['sha256']==saved['output_checkpoint_sha256'] and len(native['results'])==2,
        'completed cohort receipt chain changed')
    jobs=[bound(ref) for ref in saved['learner_receipts']]
    require(len(jobs)==2,'completed cohort lost actual learner receipts')
    for w,job in zip(record['windows'],jobs):
        row=next((r for r in coordinator_rows if r['iteration_id']==w['iteration_id']),None)
        require(row is not None and row['input_checkpoint_sha256']==w['input_checkpoint_sha256']
            and row['candidate_checkpoint_sha256']==w['output_checkpoint']['sha256']
            and row['status'] in {'advanced','selected_pending_test'}
            and job['parent_checkpoint_sha256']==w['input_checkpoint_sha256']
            and job['output_checkpoint_sha256']==w['output_checkpoint']['sha256']
            and job['global_step']==w['output_checkpoint']['optimizer_global_step'],
            'completed cohort differs from authoritative coordinator/learner lineage')
    for ref in saved['native_assignment_receipts']:
        bound({k:ref[k] for k in ('path','raw_sha256')});bound(ref['reset_evidence'])
    return saved,record

def copy_native_assignments(site,response,commands,cohort,directory):
    references=[]
    for command,item in zip(commands,response['results']):
        gpu=command['gpu'];worker=next(w for w in site.pool['workers'] if w['physical_gpu']==gpu)
        require(item['receipt']['actor_state']['physical_gpu']==gpu,'native response lane order changed')
        expected=Path('/dagger-runtime/phases')/command['payload']['output_name']/'native-assignments'
        for evidence in item['receipt']['episode_assignments']:
            source=Path(evidence['native_assignment_path'])
            require(source.parent==expected and source.suffix=='.json','native assignment escaped exact phase')
        names=[Path(e['native_assignment_path']).name for e in item['receipt']['episode_assignments']]
        payloads=fetch_native_json(site,Path(worker['runtime_root'])/expected.relative_to('/dagger-runtime'),
            [n for name in names for n in (name,Path(name).with_suffix('.reset.json').name)])
        for evidence in item['receipt']['episode_assignments']:
            source=Path(evidence['native_assignment_path'])
            raw=payloads[source.name];require(hashlib.sha256(raw).hexdigest()==evidence['native_assignment_sha256'],
                'native assignment changed during evidence transfer')
            local=directory/f'g{gpu}'/source.name;site.publish(local,raw,mode=0o444)
            reset_raw=payloads[source.with_suffix('.reset.json').name];reset=json.loads(reset_raw)
            row=json.loads(raw);lane=reset.get('lane')
            require(type(lane)is int and 0<=lane<len(cohort['assignments'][gpu]['lanes']) and reset.get('episode_id')==row.get('episode_id')
                and reset.get('assignment_sha256')==digest(cohort['assignments'][gpu]),
                'native reset did not bind actual production cohort/lane')
            reset_path=local.with_suffix('.reset.json');site.publish(reset_path,reset_raw,mode=0o444)
            references.append(dict(**binding(local),reset_evidence=binding(reset_path)))
    return references

def qualification_receipt(site,plan,completed):
    require(site.rid==1 and len(completed)==2,'qualification is exact firsttwo production cohorts')
    records=[bound(s['segment_result']) for s in completed]
    retained=site.verify_retained()
    versions={r['schema'] for r in records}
    require(len(versions)==1,'qualification cannot mix incompatible resource receipt generations')
    v2=versions=={'roborender.dagger-production-segment-result.v2'}
    resources=(aggregate_resource_guards([g for r in records for g in r['resource_guards']],per_gpu=2) if v2 else
        dict(minimum_host_available_bytes=min(r['minimum_host_available_bytes'] for r in records),
             oom_events=sum(r['oom_events'] for r in records),swap_peak_bytes=max(r['swap_peak_bytes'] for r in records),
             limit_pressure_events=sum(r['limit_pressure_events'] for r in records)))
    value=dict(schema='roborender.dagger-full295-two-cohort-qualification.v2' if v2 else 'roborender.dagger-full295-two-cohort-qualification.v1',purpose='qualification',
        plan_sha256=digest(plan),parent_checkpoint_sha256=site.config['student']['initial_checkpoint_sha256'],
        source_sha256=site.config['source_sha256'],native_cohorts=2,optimizer_updates=4,
        native_initializations_per_cohort=[1,1],vector_steps_per_cohort=[162,162],transitions=4*plan['train_frames_per_update'],
        full_batch_environments=plan['topology']['total_environments'],no_replay=True,history_reset_at_window_boundary=False,
        worker_uuid_pairs=[s['worker_uuids'] for s in completed],h200_uuids=retained['gpu_uuids'],
        native_assignment_receipts=[ref for s in completed for ref in s['native_assignment_receipts']],
        learner_receipts=[ref for s in completed for ref in s['learner_receipts']],
        completed=True,allocation_retained=True,checkpoint_optimizer_lineage_verified=True,**resources)
    return validate_qualification(plan,value,site.config['student']['initial_checkpoint_sha256'])

def collection_result(site,plan,context,last,windows,results,epoch,barrier):
    """Rebuild only a fully reconciled terminal receipt, never a native window."""
    if is_fixed100(site.config):
        require(len(windows)==100 and len(results)==50 and epoch==5
            and context['status']=='selected_pending_test'
            and context['selection']['best_checkpoint_sha256']==last
            and context['stop_reason']=='fixed_100_updates',
            'fixed100 final receipt requires the exact accepted100-update prefix')
    output=dict(schema='roborender.dagger-production-collection-result.v1',run_id=site.run,round_id=site.rid,
        source_sha256=site.config['source_sha256'],coverage_plan_sha256=digest(plan),
        completed_updates=len(windows),completed_coverage_epochs=epoch,completed_cohorts=len(results),
        history_reset_at_window_boundary=False,actor_ids=actor_ids(site.config),
        last_candidate_checkpoint_sha256=last,selected_checkpoint_sha256=context['selection']['best_checkpoint_sha256'],
        allocation_retained=True,segments=[dict(coverage_epoch=r['coverage_epoch'],cohort_index=r['cohort_index']) for r in results],
        **barrier)
    immutable_json(site.round/'production-collection-result.json',output)
    return output


def collect(site, *, continuation=None):
    require(is_mse_production(site.config),'full295 production site contract required')
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_continuous_fixed100
    if is_continuous_fixed100(site.config):
        require(continuation is None, 'a historical cohort cannot restore continuous native state')
        from pipeline.deployment.continuous_production_collection import collect as continuous_collect
        return continuous_collect(site)
    site.barrier();plan=load_plan(site.config['coverage_plan'])
    historical_prefix=None
    if continuation is not None:
        from pipeline.deployment.cohort_continuation import HistoricalPrefix
        historical_prefix=HistoricalPrefix(continuation,site.config,site.root)
    batch=plan['topology']['environments_per_actor'];total=2*batch;frames=81*total
    round_plan=plan['rounds'][site.rid-1]
    require(round_plan['teacher_beta']==site.config['teacher_beta'],'production beta differs')
    token=site.token.read_text().strip()
    coordinator=ServiceClient('http://127.0.0.1:18610',token,timeout=120)
    learner=ServiceClient('http://127.0.0.1:18613',token,timeout=120)
    config_path=site.remote/'config.json'
    site.publish(config_path,Path(site.request['round_config_path']).read_bytes(),'beta',0o444)
    mse_path=Path(site.request['mse_protocol_path']);mse_sha=site.request['mse_protocol_file_sha256']
    require(hashlib.sha256(mse_path.read_bytes()).hexdigest()==mse_sha,'MSE protocol changed')
    remote_mse=site.remote/'mse-protocol.json';site.publish(remote_mse,mse_path.read_bytes(),'beta',0o444)
    coverage_path=Path(site.config['coverage_plan']['path'])
    coverage_sha=site.config['coverage_plan']['raw_sha256']
    require(hashlib.sha256(coverage_path.read_bytes()).hexdigest()==coverage_sha,'coverage plan raw binding changed')
    remote_coverage=site.remote/'coverage-plan.json'
    site.publish(remote_coverage,coverage_path.read_bytes(),'beta',0o444)
    connection_root=getattr(site,'connection_root',site.remote)
    connection=digest(site.read(connection_root/'supervisor.json','beta'))
    prefix=hashlib.sha256(site.run.encode()).hexdigest()[:12]
    limit=site.request.get('production_cohort_limit')
    require(limit is None or limit==2 and site.rid==1,'only exact firsttwo qualification cohorts may be bounded')
    results=[];all_windows=[];completed=[];initializations=0
    last=site.config['student']['initial_checkpoint_sha256']
    for epoch in round_plan['epochs']:
        for cohort in epoch['cohorts']:
            index=cohort['cohort_index'];number=epoch['coverage_epoch']
            global_start=cohort['start_iteration']
            status=coordinator.get(f'/v1/status/{site.run}')
            mse=coordinator.get(f'/v1/mse-rounds/{site.run}/{site.rid}')['mse_round']
            rows=sorted((r for r in status['iterations'] if r['round_id']==site.rid),key=lambda r:r['iteration_id'])
            local_directory=site.round/f'coverage/e{number:02d}-c{index:03d}'
            prefix_plan=(historical_prefix.plan_for(local_directory,number,index)
                         if historical_prefix is not None else plan)
            previous=completed_prefix(local_directory,prefix_plan,site.rid,number,index,last,rows)
            if previous is not None:
                saved,record=previous;results.append(record);completed.append(saved);all_windows+=record['windows']
                last=saved['output_checkpoint_sha256']
                if historical_prefix is not None and len(completed)>=20:
                    historical_prefix.record_requalification(completed,site.round,site.barrier())
                if limit==len(results):
                    proof=qualification_receipt(site,plan,completed)
                    path=site.round/'two-cohort-qualification.json';immutable_json(path,proof)
                    return dict(schema='roborender.dagger-production-qualification-result.v1',
                        qualification=binding(path),completed_cohorts=2,completed_updates=4,
                        allocation_retained=True,**site.barrier())
                if is_fixed100(site.config) and len(all_windows)==100:
                    # A crash between the last durable cohort marker and the
                    # aggregate receipt must not require a101st window/replay.
                    return collection_result(site,plan,mse,last,all_windows,results,number,site.barrier())
                continue
            require(status['run']['status']=='collecting' and mse['status']=='active'
                and rows[-1]['iteration_id']==global_start+1 and rows[-1]['status']=='collecting',
                'cohort does not begin at exact next global checkpoint barrier')
            parent=status['run']['active_checkpoint_sha256']
            require(parent==last,'next cohort parent differs from verified completed prefix')
            assignment_bindings=[];remote_assignments=[]
            directory=site.remote/f'coverage-e{number:02d}-c{index:03d}'
            for gpu,assignment in enumerate(cohort['assignments']):
                path=directory/f'training-assignment-g{gpu}.json';raw=canonical(assignment)
                site.publish(path,raw,'beta',0o444)
                assignment_bindings.append(dict(physical_gpu_index=gpu,path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
                remote_assignments.append(path)
            segment=make_segment(plan,round_id=site.rid,epoch=number,index=index,
                input_checkpoint_sha256=parent,mse_protocol_file_sha256=mse_sha,motion_assignments=assignment_bindings,
                coverage_plan_sha256=coverage_sha)
            segment_path=directory/'segment.json';segment_raw=canonical(segment)
            segment_sha=hashlib.sha256(segment_raw).hexdigest()
            site.publish(segment_path,segment_raw,'beta',0o444)
            commands=[]
            for gpu in (0,1):
                name=f'prod-{prefix}-r{site.rid}-e{number:02d}-c{index:03d}-g{gpu}'
                commands.append(dict(gpu=gpu,command_id=name,operation='collect_production_segment',payload=dict(
                    round_config=str(config_path.relative_to(site.jobs)),round_config_sha256=digest(site.config),
                    token_file=str((site.remote/'services.token').relative_to(site.jobs)),
                    connection_status_directory=str((connection_root/'supervisor').relative_to(site.jobs)),
                    connection_config_sha256=connection,output_name=name,
                    mse_protocol_file=str(remote_mse.relative_to(site.jobs)),mse_protocol_sha256=mse_sha,
                    segment_file=str(segment_path.relative_to(site.jobs)),segment_sha256=segment_sha,
                    coverage_plan_file=str(remote_coverage.relative_to(site.jobs)),coverage_plan_sha256=coverage_sha,
                    motion_assignment_file=str(remote_assignments[gpu].relative_to(site.jobs)),
                    motion_assignment_sha256=assignment_bindings[gpu]['sha256'])))
            # Exactly one command perGPU is submitted, then both must finish.
            immutable_json(local_directory/'intent.json',dict(segment_sha256=segment_sha,commands=commands,
                input_checkpoint_sha256=parent,source_sha256=site.config['source_sha256']))
            site.pool_action('submit',commands);response=site.pool_action('wait',commands)
            require(len(response['results'])==2,'both production native owners must complete')
            actor_states=[];guards=[]
            for item in response['results']:
                receipt=item['receipt'];state=receipt['actor_state']
                require(receipt.get('complete')is True and receipt.get('segment_sha256')==segment_sha
                    and state.get('phase')=='COMPLETE' and state.get('uncertain_reason')is None
                    and state.get('completed_windows')==2 and state.get('transition_batches')==162
                    and state.get('physical_transition_rows')==162*batch
                    and state.get('lifecycle_initializations')==1
                    and receipt.get('episode_assignments'),'native production segment is incomplete')
                actor_states.append(state);guards.append(receipt['resource_guard'])
            require({s['physical_gpu'] for s in actor_states}=={0,1},'wrong native GPU pair')
            from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import validate_summary
            for guard in guards:validate_summary(guard)
            barrier=site.barrier();initializations+=1
            immutable_json(local_directory/'native-result.json',response)
            assignment_refs=copy_native_assignments(site,response,commands,cohort,local_directory)
            windows=[];last=parent;input_global_step=None;learner_refs=[]
            for offset in (0,1):
                iid=global_start+offset+1
                job=learner.get(f'/v1/jobs/{site.run}-dagger-{site.rid}-iteration-{iid:04d}')['job']
                result=job.get('result')
                require(job['status']=='succeeded' and result and result['parent_checkpoint_sha256']==last
                    and result['train_chunks']==total and result['train_frames']==frames
                    and result['batching_semantics']=='raven_exact_flattened_bxt'
                    and result.get('replay_manifest_sha256')is None
                    and result['global_step']==result['parent_global_step']+1
                    and result['optimizer_state_preserved']is True,'actual learner/AdamW lineage differs')
                if input_global_step is None:input_global_step=result['parent_global_step']
                result_path=local_directory/f'learner-{iid:04d}.json'
                immutable_json(result_path,result);learner_refs.append(binding(result_path))
                windows.append(dict(iteration_id=iid,input_checkpoint_sha256=last,
                    output_checkpoint=dict(path=result['output_checkpoint_path'],sha256=result['output_checkpoint_sha256'],
                        optimizer_global_step=result['global_step'],optimizer_state_preserved=True),
                    transition_batches=81,train_chunks=total,train_frames=frames,optimizer_steps=1,
                    full_batch_environments=total,replay_manifest_sha256=None,teacher_beta=site.config['teacher_beta'],
                    student_action_dimensions=11,train_mse=result['train_mse']))
                last=result['output_checkpoint_sha256']
            expected_chain=[(w['iteration_id']-1,w['input_checkpoint_sha256'],w['output_checkpoint']['sha256']) for w in windows]
            require(all([(w['index'],w['input_checkpoint_sha256'],w['output_checkpoint_sha256'])
                         for w in state['windows']]==expected_chain for state in actor_states),
                    'native actor checkpoint chain differs from actual learner jobs')
            guard_versions={g.get('schema','roborender.native-phase-resource-guard.v1') for g in guards}
            require(len(guard_versions)==1,'one source-pinned resource policy per native segment required')
            v2=guard_versions in ({'roborender.native-phase-resource-guard.v2'},
                                 {'roborender.native-phase-resource-guard.v3'})
            resources=(aggregate_resource_guards([dict(physical_gpu=s['physical_gpu'],summary=g)
                for s,g in zip(actor_states,guards)],per_gpu=1) if v2 else
                dict(minimum_host_available_bytes=min(g['minimum_host_available_bytes'] for g in guards),
                    oom_events=sum(g['oom_events'] for g in guards),swap_peak_bytes=max(g['swap_peak_bytes'] for g in guards),
                    limit_pressure_events=sum(g['limit_pressure_events'] for g in guards)))
            record=dict(schema='roborender.dagger-production-segment-result.v2' if v2 else 'roborender.dagger-production-segment-result.v1',run_id=site.run,round_id=site.rid,
                coverage_epoch=number,cohort_index=index,plan_sha256=digest(plan),source_sha256=site.config['source_sha256'],
                native_complete=True,native_quiescent=True,native_lifecycle_initializations_per_actor=[1,1],
                history_reset_at_window_boundary=False,assignments=[digest(a) for a in cohort['assignments']],
                windows=windows,allocation_retained=True,online_metrics_verified=False,**resources)
            validate_segment(plan,site.rid,number,index,
                dict(sha256=parent,optimizer_global_step=input_global_step),record)
            # Real learner logging is checked atcoverage boundaries; no repeated
            # cloud history scans are added to the per-window critical path.
            immutable_json(local_directory/'segment-result.json',record)
            saved=dict(schema='roborender.dagger-production-completed-cohort.v1',
                input_checkpoint_sha256=parent,input_optimizer_global_step=input_global_step,
                output_checkpoint_sha256=last,native_quiescent=True,
                segment_result=binding(local_directory/'segment-result.json'),
                native_result=binding(local_directory/'native-result.json'),
                native_assignment_receipts=assignment_refs,learner_receipts=learner_refs,
                worker_uuids=[w['gpu_uuid'] for w in barrier['native_phase_barrier']['workers']])
            immutable_json(local_directory/'completion.json',saved);completed.append(saved)
            results.append(record);all_windows+=windows
            if historical_prefix is not None and len(completed)>=20:
                historical_prefix.record_requalification(completed,site.round,site.barrier())
            if limit==len(results):
                proof=qualification_receipt(site,plan,completed)
                path=site.round/'two-cohort-qualification.json';immutable_json(path,proof)
                return dict(schema='roborender.dagger-production-qualification-result.v1',
                    qualification=binding(path),completed_cohorts=2,completed_updates=4,
                    allocation_retained=True,**site.barrier())
            status=coordinator.get(f'/v1/status/{site.run}')
            context=coordinator.get(f'/v1/mse-rounds/{site.run}/{site.rid}')['mse_round']
            if context['status']=='selected_pending_test':
                require(index==plan['cohorts_per_coverage_epoch']-1 and len(all_windows)%plan['updates_per_coverage_epoch']==0,
                    'MSE selection occurred beforefull295coverage')
                return collection_result(site,plan,context,last,all_windows,results,number,barrier)
            require(context['status']=='active','unexpected MSE state after cohort')
    raise RuntimeError('fullproduction budget exhausted withoutMSE selection')
