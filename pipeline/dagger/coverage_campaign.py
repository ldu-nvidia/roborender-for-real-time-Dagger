"""Full295 MSE-selected campaign; retains the same four-H200 allocation.

The first two genuine cohorts are provisional qualification work. Only their
verified complete prefix can continue; no active simulator window is replayed.
Legacy smoke/50-window state machines keep their original contracts.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json

from pipeline.dagger.coverage_production import (load_plan,digest,require,validate_qualification,
    CONTINUOUS_FIXED100_SCHEMA,is_continuous_plan,is_eight_h200_plan,validate_continuous_qualification,validate_continuous_native_receipt)
from pipeline.dagger.mse_campaign import MSECampaign,validate_explicit_final_test
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_production,is_fixed100,is_continuous_fixed100,is_eight_h200,actor_ids
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.services.heldout_mse import RoundMSESelection
from pipeline.deployment.native_phase_barrier import native_quiescent
from pipeline.bc_training.handoff_common import bound

def validate_round_completion(config,receipt,iterations,context,coverage):
    if is_continuous_plan(coverage):
        return validate_continuous_round_completion(config,receipt,iterations,context,coverage)
    interval=coverage['updates_per_coverage_epoch']
    rows=sorted((r for r in iterations if r['round_id']==config['round_id']),key=lambda r:r['iteration_id'])
    count=len(rows)
    require(count>0 and count%interval==0 and count<=coverage['max_updates_per_round'],
            'production can select only after complete295-motion coverage epochs')
    require([r['iteration_id'] for r in rows]==list(range(1,count+1))
        and all(r['status']=='advanced' for r in rows[:-1]) and rows[-1]['status']=='selected_pending_test',
        'production learner history is incomplete')
    fixed=is_fixed100(config)
    if fixed:
        from types import SimpleNamespace
        selection=SimpleNamespace(best_checkpoint_sha256=context['selection']['best_checkpoint_sha256'],
                                  history=context['selection']['history'])
        require(count==100 and context['status']=='selected_pending_test' and context['iteration_id']==100
            and context['selection'].get('checkpoint_selection')=='final_update_100'
            and selection.history==[dict(iteration=100,result=dict(checkpoint_sha256=selection.best_checkpoint_sha256))],
            'fixed100 requires final selection after exactly100 accepted updates')
    else:
        selection=RoundMSESelection.restore(context['selection'])
        require(context['status']=='selected_pending_test' and context['iteration_id']==count
            and [r['iteration'] for r in selection.history]==list(range(0,count+1,interval)),
            'production fixed-validation history differs from coverage epochs')
    current=config['student']['initial_checkpoint_sha256'];known={current}
    for row in rows:
        require(row['input_checkpoint_sha256']==current and row['candidate_checkpoint_sha256'],
                'production checkpoint lineage is broken')
        current=row['candidate_checkpoint_sha256'];known.add(current)
    require(selection.best_checkpoint_sha256 in known and context['checkpoint_sha256']==current,
            'selected policy is outside the actual candidate lineage')
    if fixed:
        require(selection.best_checkpoint_sha256==current,'fixed100 must select the final candidate')
    require(receipt.get('schema')=='roborender.dagger-production-collection-result.v1'
        and receipt.get('run_id')==config['run_id'] and receipt.get('round_id')==config['round_id']
        and receipt.get('completed_updates')==count and receipt.get('completed_coverage_epochs')==count//interval
        and receipt.get('completed_cohorts')==count//2 and receipt.get('history_reset_at_window_boundary')is False
        and receipt.get('source_sha256')==config['source_sha256']
        and receipt.get('coverage_plan_sha256')==digest(coverage)
        and set(receipt.get('actor_ids',[]))==set(actor_ids(config))
        and receipt.get('last_candidate_checkpoint_sha256')==current
        and receipt.get('selected_checkpoint_sha256')==selection.best_checkpoint_sha256
        and receipt.get('allocation_retained')is True
        and native_quiescent(receipt,source_sha256=config['source_sha256']),
        'production native cohort completion/quiescence receipt differs')
    expected_reason=('fixed_100_updates' if fixed else
                     'validation_mse_plateau' if selection.history[-1]['saturated'] else 'maximum_iterations')
    require(context['stop_reason']==expected_reason
        and (expected_reason!='maximum_iterations' or count==coverage['max_updates_per_round']),
        'production stop is not justified by MSE patience or maximum')
    return dict(completed_iterations=count,completed_coverage_epochs=count//interval,
        selected_checkpoint_sha256=selection.best_checkpoint_sha256,last_candidate_checkpoint_sha256=current,
        stop_reason=expected_reason)


def validate_continuous_round_completion(config,receipt,iterations,context,plan):
    """Same accepted native commands from first4 qualification through update100."""
    require(is_continuous_fixed100(config) and is_continuous_plan(plan),
            'continuous completion requires the new source-bound configuration')
    rows=sorted((r for r in iterations if r['round_id']==config['round_id']),key=lambda r:r['iteration_id'])
    require([r['iteration_id'] for r in rows]==list(range(1,101))
        and all(r['status']=='advanced' for r in rows[:-1])
        and rows[-1]['status']=='selected_pending_test'
        and context.get('status')=='selected_pending_test' and context.get('iteration_id')==100
        and context.get('stop_reason')=='fixed_100_updates'
        and context.get('selection',{}).get('checkpoint_selection')=='final_update_100',
        'continuous round requires exactly100 accepted updates before testing')
    current=config['student']['initial_checkpoint_sha256']
    for row in rows:
        require(row['input_checkpoint_sha256']==current and row['candidate_checkpoint_sha256']
            and row['candidate_checkpoint_sha256']!=current,'continuous checkpoint lineage is broken')
        current=row['candidate_checkpoint_sha256']
    require(context['checkpoint_sha256']==current
        and context['selection']['best_checkpoint_sha256']==current
        and context['selection']['history']==[dict(iteration=100,result=dict(checkpoint_sha256=current))],
        'continuous round must select update100, never an earlier checkpoint')
    require(receipt.get('schema') in {'roborender.dagger-continuous-collection-result.v1',
                                    'roborender.dagger-continuous-collection-result.v2'}
        and receipt.get('run_id')==config['run_id'] and receipt.get('round_id')==config['round_id']
        and receipt.get('source_sha256')==config['source_sha256']
        and receipt.get('coverage_plan_sha256')==digest(plan)
        and receipt.get('completed_updates')==100
        and receipt.get('native_lifecycles_per_actor')==[1,1]
        and receipt.get('model_rollout_initializations_per_actor')==[100,100]
        and receipt.get('history_reset_at_window_boundary')is False
        and set(receipt.get('actor_ids',[]))==set(actor_ids(config))
        and receipt.get('last_candidate_checkpoint_sha256')==current
        and receipt.get('selected_checkpoint_sha256')==current
        and receipt.get('allocation_retained')is True
        and receipt.get('owned_actors_stopped')is not True
        and native_quiescent(receipt,source_sha256=config['source_sha256']),
        'continuous native completion/warm quiescence receipt differs')
    proof=bound(receipt['qualification'])
    validate_continuous_qualification(plan,proof,config['student']['initial_checkpoint_sha256'],
                                     round_id=config['round_id'])
    prefix={r['physical_gpu']:r for r in (bound(ref) for ref in proof['native_prefix_receipts'])}
    refs=receipt.get('native_terminal_receipts',[])
    require(len(refs)==2,'both exact terminal native receipts required')
    terminal=[validate_continuous_native_receipt(plan,ref,round_id=config['round_id'],count=100,
                                               phase='COMPLETE') for ref in refs]
    require({r['physical_gpu'] for r in terminal}=={0,1},'terminal native GPU pair differs')
    barrier={r['physical_gpu']:r for r in receipt['native_phase_barrier']['workers']}
    for native in terminal:
        if receipt['schema'] == 'roborender.dagger-continuous-collection-result.v2':
            from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import build_restore_request
            prior = build_restore_request(native)['predecessor']['binding']
            require(prior['boundary'] == 'production_update100' and prior['round_id'] == config['round_id']
                and prior['run_id'] == config['run_id'] and prior['physical_gpu'] == native['physical_gpu']
                and prior['final_checkpoint_sha256'] == current
                and all(prior[k] == native[k] for k in ('worker_instance_id','native_pid','command_id','source_sha256')),
                'production curriculum export is outside the exact final checkpoint/native lineage')
        initial=prefix[native['physical_gpu']]
        require(all(native[key]==initial[key] for key in ('worker_instance_id','native_pid','command_id',
                    'source_sha256','segment_sha256','motion_assignment_sha256')),
                'native command/lifecycle changed after continuous qualification')
        require(barrier[native['physical_gpu']]['worker_instance_id']==native['worker_instance_id'],
                'quiescence receipt belongs to a different continuous owner')
        if is_eight_h200_plan(plan):
            require(native['actor_state'].get('h200_gateway_identities')==initial['actor_state']['h200_gateway_identities']
                and native['actor_state'].get('config_sha256')==initial['actor_state']['config_sha256']
                and native['actor_state']['windows'][:4]==initial['actor_state']['windows'],
                'continuous v2 route or admitted cursor prefix changed after qualification')
        before,after=initial['resource_guard'],native['resource_guard']
        require(after.get('schema')==before.get('schema')
            and after['samples']>=before['samples']
            and after['limit_pressure_events']>=before['limit_pressure_events']
            and after['minimum_host_available_bytes']<=before['minimum_host_available_bytes'],
            'continuous native resource history regressed after qualification')
        for saved,row in zip(native['actor_state']['windows'],rows):
            require(saved['index']==row['iteration_id']-1
                and saved['input_checkpoint_sha256']==row['input_checkpoint_sha256']
                and saved['output_checkpoint_sha256']==row['candidate_checkpoint_sha256'],
                'terminal native and coordinator checkpoint histories differ')
    return dict(completed_iterations=100,native_lifecycles_per_actor=[1,1],
        selected_checkpoint_sha256=current,last_candidate_checkpoint_sha256=current,
        stop_reason='fixed_100_updates')

class CoverageMSECampaign(MSECampaign):
    supports_full295_production=True

    def __init__(self,*args,qualification=None,**kwargs):
        config=args[0] if args else kwargs['first_config']
        require(is_mse_production(config),'newproduction contract required')
        self.coverage=load_plan(config['coverage_plan'])
        require(is_continuous_fixed100(config)==(is_continuous_plan(self.coverage)),
            'continuous and legacy native lifecycle configurations cannot be interchanged')
        require(is_eight_h200(config)==is_eight_h200_plan(self.coverage),
            'four-H200 and eight-H200 continuous configurations cannot be interchanged')
        require(not is_continuous_plan(self.coverage) or qualification is None,
            'continuous rounds must earn qualification inside their actual native command')
        self.qualification_parent=config['student']['initial_checkpoint_sha256']
        if qualification is not None:
            validate_qualification(self.coverage,qualification,self.qualification_parent)
        super().__init__(*args,**kwargs)
        if qualification is not None and self.state.get('native_qualification') is None:
            self.save(native_qualification=qualification)
        if self.state.get('native_qualification') is not None:
            validate_qualification(self.coverage,self.state['native_qualification'],self.qualification_parent)

    def operation_context(self):
        value=super().operation_context()
        value.update(full295_production=True,coverage_plan=self.state['config']['coverage_plan'],
            retain_compute=True,compute_release_authorized=False)
        if is_continuous_plan(self.coverage):
            value.update(continuous_native_round=True,qualification_after_updates=4,
                         native_lifecycle_policy='native_episode_done_only')
            if self.state['phase'] == 'COLLECT_TRAIN':
                from pipeline.dagger.rtx.code.src.dagger_rtx.training_curriculum import bootstrap_request, build_restore_request
                if self.state['round_id'] == 1:
                    value['training_curriculum'] = [bootstrap_request(), bootstrap_request()]
                else:
                    previous = self.state.get('training_curriculum_predecessor') or {}
                    refs = previous.get('native_terminal_receipts', [])
                    require(previous.get('schema') == 'roborender.dagger-continuous-collection-result.v2'
                        and previous.get('round_id') == self.state['round_id'] - 1 and len(refs) == 2,
                        'later continuous training requires the prior curriculum-enabled round')
                    rows = sorted((bound(ref) for ref in refs), key=lambda r: r['physical_gpu'])
                    require([r['physical_gpu'] for r in rows] == [0,1], 'curriculum actor pair differs')
                    value['training_curriculum'] = [build_restore_request(row) for row in rows]
        elif self.state['phase']=='COLLECT_TRAIN' and self.state.get('native_qualification') is None:
            value.update(production_cohort_limit=2,qualification_stage=True)
        return value

    def tick(self):
        phase=self.state['phase'];config=self.state['config']
        if phase=='COLLECT_TRAIN':
            controller=self.get_controller();receipt=self.operation('collect');progress=controller.tick()
            if progress!=self.state.get('progress'):self.save(progress=progress)
            if receipt is None:return dict(self.state)
            if not is_continuous_plan(self.coverage) and self.state.get('native_qualification') is None:
                require(receipt.get('schema')=='roborender.dagger-production-qualification-result.v1'
                    and receipt.get('completed_cohorts')==2 and receipt.get('completed_updates')==4
                    and native_quiescent(receipt,source_sha256=config['source_sha256']),
                    'firsttwo genuine cohorts must finish before qualification admission')
                proof=bound(receipt['qualification'])
                validate_qualification(self.coverage,proof,self.qualification_parent)
                self.save(operation_id=None,native_qualification=proof,
                    native_qualification_binding=receipt['qualification'],qualification_completed=True)
                return dict(self.state)
            status=controller.coordinator.get(f"/v1/status/{config['run_id']}")
            context=controller.mse.context()
            selected=validate_round_completion(config,receipt,status['iterations'],context,self.coverage)
            count=selected['completed_iterations']
            job=controller.learner.get(f"/v1/jobs/{config['run_id']}-dagger-{config['round_id']}-iteration-{count:04d}")['job']
            result=job.get('result')
            require(job['status']=='succeeded' and result and result['train_chunks']==self.coverage['topology']['total_environments']
                and result['train_frames']==self.coverage['train_frames_per_update']
                and result['output_checkpoint_sha256']==selected['last_candidate_checkpoint_sha256']
                and result.get('replay_manifest_sha256')is None,'final full4x81 learner receipt differs')
            checkpoint=Path(context['selected_checkpoint_path'])
            require(file_sha256(checkpoint)==selected['selected_checkpoint_sha256'],'selected checkpoint unavailable')
            controller.mse.final_test()
            self.save('RELEASE_TRAINING',operation_id=None,
                final_checkpoint=dict(path=str(checkpoint),sha256=selected['selected_checkpoint_sha256']),
                final_learner=result,mse_completion=selected,coverage_completion=receipt)
            return dict(self.state)
        if phase=='RELEASE_TRAINING':
            # Compatibility phase name; the new bound operation retains all
            # compute and keeps the authoritative ledger on the same H200 host.
            receipt=self.operation('release')
            if receipt is None:return dict(self.state)
            require(receipt.get('h200_allocation_retained')is True
                and receipt.get('h200_allocation_released')is False
                and receipt.get('same_service_ledger_owner')is True
                and native_quiescent(receipt,source_sha256=config['source_sha256']),
                'production phase barrier must retain compute and ledger ownership')
            self.save('EVALUATE',operation_id=None,retention_receipt=receipt)
            return dict(self.state)
        if phase=='EVALUATE':
            receipt=self.operation('evaluate')
            if receipt is None:return dict(self.state)
            path=Path(receipt['comparison_path'])
            require(file_sha256(path)==receipt['comparison_sha256']
                and native_quiescent(receipt,source_sha256=config['source_sha256']),
                'production test evidence changed or native owners still active')
            validate_explicit_final_test(config,json.loads(path.read_bytes()),self.state['final_checkpoint']['sha256'])
            # Evaluation ran on RTX, while the existing H200 ledger stayed live.
            self.save('IMPORT_VALIDATION',operation_id=None,evaluation_receipt=receipt)
            return dict(self.state)
        previous=deepcopy(self.state)
        result=super().tick()
        if phase=='REGISTER' and self.state['phase']=='COLLECT_TRAIN':
            # A CPU lifetime budget is not an allocation or native expiry.
            self.save(collection_deadline=self.clock()+7*86400)
        if phase=='ADVANCE' and previous['round_id']<4 and self.state['phase']=='PREPARE':
            require(self.state['config']['coverage_plan']==previous['config']['coverage_plan'],
                'next round changed frozen coverage plan')
            if is_continuous_plan(self.coverage):
                completed = previous.get('coverage_completion') or {}
                require(completed.get('schema') == 'roborender.dagger-continuous-collection-result.v2',
                        'next round cannot silently discard missing native curriculum')
                self.save(training_curriculum_predecessor=completed)
        return result
