"""Queued BC-selected-policy -> four DAgger rounds with fixed295 success tests.

The default is a read-only readiness check. --watch --execute is explicit
authorization for the pinned retained-site operations, not compute allocation.
The observer deadline only detaches this CPU scheduler; it never releases
compute, kills native owners, or replays an interrupted operation.
Fresh v3 --qualification-only freezes a separate bounded scope and stops after
two genuine native cohorts/four optimizer updates; it never advances a round.
New continuous-round queues instead qualify their first four updates inside
the same native command, then continue that physical scene through update100.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from pipeline.bc_training.handoff_common import binding,bound,publish
from pipeline.dagger.coverage_production import make_plan,stage,require,digest,validate_qualification
from pipeline.dagger.coverage_campaign import CoverageMSECampaign
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.mse_campaign import online_mse_loggers
from pipeline.deployment.coverage_operations import CoverageCommandOperations
from pipeline.deployment.beta_operations import atomic,PHASES
from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options,validate_production_result
from pipeline.deployment.native_phase_barrier import native_quiescent

SCHEMA='roborender.bc-to-full295-dagger-queue.v1'
FRESH_SCHEMA='roborender.bc-to-full295-dagger-queue.v2'
SCALED_FRESH_SCHEMA='roborender.bc-to-full295-dagger-queue.v3'
FIXED100_QUEUE_SCHEMA='roborender.bc-to-fixed100-rendered-dagger-queue.v1'
CONTINUOUS_FIXED100_QUEUE_SCHEMA='roborender.bc-to-continuous-fixed100-rendered-dagger-queue.v1'
CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA='roborender.bc-to-continuous-fixed100-rendered-dagger-queue.v2'
RUNTIME_ROOT=Path('/workspace/project/roborender/runtime')
QUALIFICATION_SCOPE_SCHEMA='roborender.dagger-qualification-only-scope.v1'


def admit_execution_scope(queue,queue_sha256,*,qualification_only):
    """Pin a deliberately bounded execution; omission on resume never broadens it.

    Existing unscoped queues retain their behavior. A qualification-only scope
    can be created only before its first campaign state exists, not by relabeling
    an already-running production campaign. Call under the queue owner lock.
    """
    root=Path(queue['root']);path=root/'execution-scope.json'
    if not qualification_only:
        require(not path.exists(),'existing bounded queue requires --qualification-only on every execute/resume')
        return None
    require(queue['schema']==SCALED_FRESH_SCHEMA,
        '--qualification-only requires the fresh scaled queue v3')
    require(isinstance(queue_sha256,str) and len(queue_sha256)==64
        and all(c in '0123456789abcdef' for c in queue_sha256),'exact queue byte identity required')
    value=dict(schema=QUALIFICATION_SCOPE_SCHEMA,queue_raw_sha256=queue_sha256,
        run_id=queue['run_id'],source_sha256=queue['source_sha256'],round_id=1,
        teacher_beta=.75,native_training_cohorts=2,optimizer_updates=4,
        stop_after_qualification=True,automatic_continuation=False,
        keep_native_owners_warm=True,compute_release_permitted=False)
    if path.exists():
        require(bound(binding(path))==value,'immutable qualification execution scope changed')
    else:
        require(not (root/'campaign/state.json').exists(),
            'cannot retrofit a bounded scope onto an existing production campaign')
        publish(path,value)
    return binding(path)


class QualificationOnlyCampaign(CoverageMSECampaign):
    """Exactly the existing real qualification, with no following production tick.

    The genuine operation receipt is saved before the campaign records its
    qualification. Therefore a CPU crash at either boundary can be reattached
    without launching another cohort or inferring quiescence from a boolean.
    """
    def __init__(self,*args,execution_scope,**kwargs):
        self.execution_scope=execution_scope
        scope=bound(execution_scope)
        require(scope.get('schema')==QUALIFICATION_SCOPE_SCHEMA,
            'explicit qualification-only execution scope required')
        super().__init__(*args,**kwargs)
        self._check_scope()

    def _check_scope(self):
        scope=bound(self.execution_scope)
        config=self.state['config']
        require(scope.get('schema')==QUALIFICATION_SCOPE_SCHEMA
            and scope.get('run_id')==config['run_id']
            and scope.get('source_sha256')==config['source_sha256']
            and scope.get('round_id')==self.state['round_id']==config['round_id']==1
            and scope.get('teacher_beta')==config['teacher_beta']==.75
            and scope.get('native_training_cohorts')==2 and scope.get('optimizer_updates')==4
            and scope.get('stop_after_qualification')is True
            and scope.get('automatic_continuation')is False
            and scope.get('keep_native_owners_warm')is True
            and scope.get('compute_release_permitted')is False
            and self.state.get('completed_rounds')==[]
            and self.state['phase'] in {'PREPARE','COLLECT_HELDOUT','REGISTER','COLLECT_TRAIN'},
            'bounded reliability scope cannot enter later production or a subsequent round')
        return scope

    def _validate_native_receipt(self,receipt):
        config=self.state['config']
        require(receipt.get('schema')=='roborender.dagger-production-qualification-result.v1'
            and receipt.get('run_id')==config['run_id'] and receipt.get('round_id')==1
            and receipt.get('completed_cohorts')==2 and receipt.get('completed_updates')==4
            and receipt.get('native_phase_barrier') is not None
            and receipt.get('owned_actors_stopped') is not True
            and native_quiescent(receipt,source_sha256=config['source_sha256']),
            'bounded reliability requires the exact four-update warm native quiescence receipt')
        proof=bound(receipt['qualification'])
        validate_qualification(self.coverage,proof,self.qualification_parent)
        return proof

    def operation(self,phase):
        self._check_scope()
        expected={'PREPARE':'prepare','COLLECT_HELDOUT':'collect','COLLECT_TRAIN':'collect'}
        require(phase==expected.get(self.state['phase'])
            and self.state.get('qualification_completed') is not True
            and self.state.get('native_qualification') is None,
            'bounded reliability cannot dispatch a later or out-of-phase native operation')
        receipt=super().operation(phase)
        if (receipt is not None and phase=='collect' and self.state['phase']=='COLLECT_TRAIN'
                and self.state.get('native_qualification') is None):
            self._validate_native_receipt(receipt)
            immutable_json(self.root/'qualification-only-native-receipt.json',receipt)
        return receipt

    def qualification_completion(self):
        self._check_scope()
        require(self.state.get('qualification_completed')is True
            and self.state['phase']=='COLLECT_TRAIN' and self.state.get('operation_id') is None,
            'actual qualification has not reached its quiescent boundary')
        ref=binding(self.root/'qualification-only-native-receipt.json')
        receipt=bound(ref);proof=self._validate_native_receipt(receipt)
        require(self.state.get('native_qualification')==proof
            and self.state.get('native_qualification_binding')==receipt['qualification'],
            'saved qualification differs from the completed native operation')
        value=dict(schema='roborender.dagger-qualification-only-completion.v1',
            run_id=self.state['config']['run_id'],round_id=1,execution_scope=self.execution_scope,
            native_operation_receipt=ref,qualification=receipt['qualification'],
            completed_native_training_cohorts=2,completed_optimizer_updates=4,
            automatic_continuation=False,native_owners_kept_warm=True,
            compute_released=False,full_production_round_completed=False)
        immutable_json(self.root/'qualification-only-completion.json',value)
        return value

    def tick(self):
        self._check_scope()
        if self.state.get('qualification_completed')is True:
            self.qualification_completion()
            return dict(self.state)  # never dispatch the next collection operation
        require(self.state.get('native_qualification') is None,
            'qualification-only cannot adopt an externally supplied qualification')
        result=super().tick()
        if self.state.get('qualification_completed')is True:self.qualification_completion()
        return result


def _fresh_inputs(value):
    """Admit a new campaign from exact fresh BC provenance, never old data.

    The fresh queue deliberately derives its excluded simulator seeds from the
    bound BC sampling plan.  A caller cannot quietly retain an old queue's seed
    deny-list while changing only the visible run name or output directory.
    """
    source=bound(value['source_manifest'])
    require(source.get('schema')=='roborender.source-snapshot.v1'
        and source.get('source_sha256')==value['source_sha256']
        and source.get('file_count')==len(source.get('files',[]))
        and digest(source['files'])==source['source_sha256'],
        'fresh queue source manifest differs from the exact native source')
    sampling=bound(value['bc_sampling_plan'])
    require(sampling.get('schema')=='roborender.coverage-sampling-plan.v3'
        and set(sampling.get('partitions',{}))=={'train','validation'}
        and sampling.get('selection_uses_labels_rewards_or_loss')is False,
        'fresh BC train/validation sampling plan required')
    cohorts=[row for name in ('train','validation')
             for row in sampling['partitions'][name].get('cohorts',[])]
    seeds=[row.get('process_seed') for row in cohorts]
    require(cohorts and len(seeds)==len(set(seeds))
        and all(type(seed)is int and 0<=seed<2**31 for seed in seeds)
        and sorted(value['excluded_process_seeds'])==sorted(seeds),
        'DAgger excluded seeds must equal the fresh BC train/validation process seeds')
    admission=bound(value['bc_data_admission'])
    original_admission=admission
    from pipeline.bc_training import final_selection, train_union
    if admission.get('schema') == train_union.SCHEMA:
        require(value['schema']==CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA,
                'merged final-epoch BC requires the explicit continuous eight-H200 queue')
        selection=final_selection.validate_selection(bound(binding(value['bc_selection_path'])))
        require(selection['data_admission']==value['bc_data_admission']
            and selection['test_panel']==value['panel'],
            'merged queue differs from the sealed BC population or scaled test')
        base=admission['base_admission']
        admission=bound(dict(path=base['path'],raw_sha256=base['sha256']))
    inventory=bound(value['inventory']);panel=bound(value['panel'])
    require(admission.get('schema')=='roborender.bc-data-admission.v3'
        and admission.get('purpose')=='production'
        and admission.get('production_admitted')is True
        and admission.get('metadata_independence_passed')is True
        and admission.get('motion_support_matches')is True
        and admission.get('test_payloads_opened')is False
        and admission.get('closed_loop_test_executed')is False
        and admission.get('missing_provenance')=={}
        and admission.get('warnings')==[]
        and admission.get('protocol',{}).get('sampling_plan_canonical_sha256')==digest(sampling)
        and admission['protocol'].get('motion_inventory_canonical_sha256')==digest(inventory)
        and sampling.get('motion_inventory_sha256')==digest(inventory)
        and panel.get('purpose')=='test' and panel.get('episode_count')==295
        and not set(seeds)&{row['assignment']['simulator_seed'] for row in panel.get('cohorts',[])},
        'fresh BC isolation/admission does not bind the DAgger inventory and sealed test panel')
    return sampling,original_admission

def validate_queue(value):
    fields={'schema','run_id','root','bc_selection_path','bc_success_path','runtime_bindings_path',
        'inventory','panel','excluded_process_seeds','seed','motion_root','source_sha256',
        'teacher_sha256','policy_spec_sha256','wandb_entity','wandb_key_file'}
    scaled=value.get('schema') in (SCALED_FRESH_SCHEMA,FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA)
    fresh=value.get('schema') in (FRESH_SCHEMA,SCALED_FRESH_SCHEMA,FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA)
    fresh_fields={'campaign_root','source_manifest','bc_sampling_plan','bc_data_admission',
        'environments_per_actor'}
    scaled_fields={'capacity_admission','learner_capacity'}
    require(set(value)==fields|(fresh_fields if fresh else set())|(scaled_fields if scaled else set())
        and value.get('schema') in (SCHEMA,FRESH_SCHEMA,SCALED_FRESH_SCHEMA,FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA),
        'exact queued production contract required')
    for key in ('root','bc_selection_path','bc_success_path','runtime_bindings_path','motion_root'):
        require(Path(value[key]).is_absolute(),'absolute queued path required: '+key)
    root=Path(value['root']).resolve()
    if value['schema'] in (FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA):
        require(value.get('environments_per_actor')==16,'fixed100 requires exactlyB16 perRTX')
    if fresh:
        campaign=Path(value['campaign_root'])
        require(campaign.is_absolute() and '..' not in campaign.parts,
            'absolute fresh campaign root required')
        campaign=campaign.resolve()
        require(campaign.is_relative_to(RUNTIME_ROOT) and campaign!=RUNTIME_ROOT
            and root==campaign/'dagger/run'
            and Path(value['bc_selection_path']).resolve().is_relative_to(campaign)
            and Path(value['bc_success_path']).resolve().is_relative_to(campaign)
            and Path(value['runtime_bindings_path']).resolve().is_relative_to(campaign/'dagger')
            and type(value['environments_per_actor'])is int
            and value['environments_per_actor'] in ((4,8,16,32,64) if scaled else (2,)),
            'fresh queue paths and explicit topology must stay inside its bound campaign')
        _fresh_inputs(value)
        if scaled:
            from pipeline.deployment.capacity_admission import validate,validate_learner,validate_production_memory_policy
            native_capacity=validate(bound(value['capacity_admission']),source_sha256=value['source_sha256'],
                environments_per_actor=value['environments_per_actor'])
            validate_production_memory_policy(native_capacity)
            selection=bound(binding(value['bc_selection_path']))
            validate_learner(bound(value['learner_capacity']),
                environments_per_actor=value['environments_per_actor'],
                checkpoint_sha256=selection['checkpoint']['sha256'],
                source_manifest=bound(value['source_manifest']))
    else:
        require(root.is_relative_to('/workspace/project/roborender/runtime/experiments/20261016-prod-1/dagger'),
            'legacy production queue is scoped to its original immutable runtime')
    require(value['wandb_entity']=='nvidia','approved W&B entity required')
    if value['wandb_key_file'] is not None:
        require(Path(value['wandb_key_file']).is_absolute(),'credential must be a local file reference, never a token')
    return value

def validate_bc_handoff(queue):
    selection=bound(binding(queue['bc_selection_path']))
    from pipeline.bc_training import final_selection
    if selection.get('schema') == final_selection.SCHEMA:
        selection=final_selection.validate_selection(selection)
        require(queue['schema']==CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA,
                'final-epoch BC handoff requires the explicit continuous eight-H200 queue')
        require(selection['test_panel']==queue['panel']
            and selection['data_admission']==queue['bc_data_admission']
            and digest(bound(selection['policy_spec']))==queue['policy_spec_sha256'],
            'final-epoch BC/DAgger panel, admitted population or policy specification differs')
        from pipeline.deployment.paired_scaled_evaluation import validate_final_evidence
        from pipeline.deployment.rendered_success_dispatch import configuration
        panel=bound(queue['panel'])
        rendered_config=configuration(selection['rendered_observation_config'],panel,
                                      selection['checkpoint']['sha256'])
        evidence=bound(binding(queue['bc_success_path']))
        checked=validate_final_evidence(evidence,
            panel=panel, checkpoint_sha256=selection['checkpoint']['sha256'],
            source_sha256=queue['source_sha256'], arm='bc', expected_rendered_config=rendered_config)
        require(checked.get('completed_episodes')==295 and checked.get('arm')=='bc'
            and checked.get('qualification_only') is False,
            'actual complete scaled BC295 test required before DAgger')
        from pipeline.deployment.paired_bc_success import validate_online_handoff
        validate_online_handoff(evidence,selection_ref=binding(queue['bc_selection_path']),
            source_sha256=queue['source_sha256'],result=checked)
        return selection
    require(selection.get('schema')=='roborender.bc-production-selected.v1'
        and selection.get('selection')=='minimum_complete_validation_mse'
        and selection.get('optimizer_state_preserved')is True
        and selection.get('rng_state_preserved')is True
        and selection.get('expected_test_episodes')==295
        and type(selection.get('selected_validation_mse'))in(int,float)
        and math.isfinite(selection['selected_validation_mse']) and selection['selected_validation_mse']>=0
        and type(selection.get('selected_global_step'))is int and selection['selected_global_step']>0,
        'actual MSE-selected BC optimizer/policy handoff required')
    require(selection['test_panel']==queue['panel'],'BC and DAgger independent full295 test panel differs')
    if queue.get('schema') in (FRESH_SCHEMA,SCALED_FRESH_SCHEMA,FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA):
        require(selection.get('data_admission')==queue['bc_data_admission'],
            'fresh DAgger queue and selected BC data admission differ')
    for name in ('training_state','training_config','data_admission','policy_spec'):bound(selection[name])
    require(digest(bound(selection['policy_spec']))==queue['policy_spec_sha256'],
        'BC/DAgger policy specification differs')
    checkpoint=selection['checkpoint'];path=Path(checkpoint['path'])
    require(path.stat().st_size==checkpoint['bytes'],'selected BC checkpoint size changed')
    before=path.stat();sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):sha.update(block)
    after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        and sha.hexdigest()==checkpoint['sha256'],'selected BC checkpoint bytes changed')
    result=bound(binding(queue['bc_success_path']))
    panel=bound(queue['panel'])
    evaluation_options(panel,raw_sha256=queue['panel']['raw_sha256'])
    from pipeline.deployment.rendered_success_dispatch import configuration
    rendered_config=configuration(selection.get('rendered_observation_config'),panel,checkpoint['sha256'])
    checked=validate_production_result(panel,result,checkpoint['sha256'],allowed_arms=('bc',),rendered_config=rendered_config)
    require(checked['completed_episodes']==295,'actual complete BC295 test required before DAgger')
    return selection

def operations_for(runtime,root):
    fields={'schema','site','resident_pool','retained','source_sha256'}
    v2=runtime.get('schema')=='roborender.full295-runtime-bindings.v2'
    v3=runtime.get('schema')=='roborender.full295-runtime-bindings.v3'
    v4=runtime.get('schema')=='roborender.full295-runtime-bindings.v4'
    require(set(runtime)==fields|({'paired_evaluation'} if v4 else {'control_only'} if v2 or v3 else set())
        and runtime['schema'] in ('roborender.full295-runtime-bindings.v1','roborender.full295-runtime-bindings.v2','roborender.full295-runtime-bindings.v3','roborender.full295-runtime-bindings.v4'),
        'root-reviewed actual retained runtime bindings required')
    site=bound(runtime['site']);retained=bound(runtime['retained'])
    supported_retained=(('roborender.full295-retained-h200.v3',) if v4 else ('roborender.full295-retained-h200.v1','roborender.full295-retained-h200.v2') if not v2 and not v3
        else ('roborender.full295-retained-h200.v2',) if v3 else ('roborender.full295-retained-h200.v1',))
    require(site['schema']=='roborender.leo-site.v1' and retained['schema'] in supported_retained,
        'only exact Leo and retained H200 topology is supported')
    pool=runtime['resident_pool']
    require(set(pool)=={'path','sha256'} and Path(pool['path']).is_absolute()
        and len(pool['sha256'])==64,'actual remote canonical B2 pool binding required')
    # Control-only queue code may execute from a separately frozen management
    # release.  Bind the phase provider from that same verified code tree;
    # campaign data and checkpoints remain explicitly rooted in ``runtime``.
    code_root=Path(__file__).resolve().parents[2]
    entry=code_root/'pipeline/deployment/retained_campaign_site.py'
    argv=[site['cpu_python'],str(entry),'--site',runtime['site']['path'],'--site-sha256',runtime['site']['raw_sha256'],
        '--resident-pool',pool['path'],'--resident-pool-sha256',pool['sha256'],
        '--retained',runtime['retained']['path'],'--retained-sha256',runtime['retained']['raw_sha256']]
    if v4:
        from pipeline.deployment.paired_scaled_runtime import validate_runtime
        paired=runtime['paired_evaluation']
        validate_runtime(bound(paired),source_sha256=runtime['source_sha256'],retained=retained)
        require(bound(binding(site['source_manifest']))['source_sha256']==runtime['source_sha256'],
            'paired v4 requires the actual same-source native/provider release')
        argv+=['--paired-evaluation',paired['path'],'--paired-evaluation-sha256',paired['raw_sha256']]
    if v2 or v3:
        from pipeline.deployment.production_pool_control import validate_control
        control=runtime['control_only']
        require(set(control)=={'management_release','pool_control'} if v2 else
            {'management_release'}<=set(control)<={'management_release','pool_control'},
            'exact control-only recovery bindings required')
        for key in control:
            require(set(control[key])=={'path','raw_sha256'},'exact control binding reference required')
        management=bound(control['management_release'])
        require(management.get('native_source_sha256')==runtime['source_sha256'],
            'management admission changes native identity')
        if 'pool_control' in control:validate_control(bound(control['pool_control']),runtime['source_sha256'])
        for key,flag in (('management_release','management-release'),('pool_control','pool-control')):
            if key in control:argv+=['--'+flag,control[key]['path'],'--'+flag+'-sha256',control[key]['raw_sha256']]
    spec=dict(argv=argv,entrypoint=str(entry),entrypoint_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        timeout_seconds=7*86400)
    return dict(schema='roborender.full295-campaign-operations.v1',compute_release_permitted=False,
        commands={phase:dict(spec) for phase in PHASES})

def prepare_campaign(queue):
    # Recheck real source/parent/capacity bindings even for programmatic callers.
    # Never silently re-enter the historical B2 plan on a scaled handoff.
    validate_queue(queue)
    root=Path(queue['root']);handoff=root/'bc-handoff-admission.json'
    if handoff.is_file():
        admission=json.loads(handoff.read_bytes())
        require(admission['selection']==binding(queue['bc_selection_path'])
            and admission['actual_success_test']==binding(queue['bc_success_path']),
            'immutable BC handoff changed')
        # Cached admission binds immutable bytes but cannot certify a newer
        # evaluator protocol. Revalidate those exact bytes on every resume.
        selection=validate_bc_handoff(queue)
    else:
        selection=validate_bc_handoff(queue)
        publish(handoff,dict(schema='roborender.full295-bc-handoff.v1',
            selection=binding(queue['bc_selection_path']),actual_success_test=binding(queue['bc_success_path']),
            checkpoint_sha256=selection['checkpoint']['sha256'],optimizer_state_preserved=True))
    runtime=bound(binding(queue['runtime_bindings_path']))
    require(runtime['source_sha256']==queue['source_sha256'],'native/H200/queue source differs')
    from pipeline.deployment.rtx_actor.scaled_success_contract import is_panel
    require(is_panel(bound(queue['panel']))==(runtime.get('schema')=='roborender.full295-runtime-bindings.v4'),
        'paired scaled final tests require runtime v4; historical runtime/panel routes cannot be mixed')
    plan_args={k:queue[k] for k in ('run_id','inventory','panel','excluded_process_seeds','seed',
        'motion_root','source_sha256','teacher_sha256','policy_spec_sha256')}
    if queue.get('schema') in (FRESH_SCHEMA,SCALED_FRESH_SCHEMA,FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA):
        plan_args['environments_per_actor']=queue['environments_per_actor']
    if queue.get('schema') in (FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA):
        plan_args['fixed100']=True
        plan_args['continuous_round']=queue['schema'] in (CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA)
        if queue['schema']==CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA:plan_args['h200_workers']=8
    plan=make_plan(**plan_args)
    staged=stage(plan,output_root=root/'plans',checkpoint=selection['checkpoint']['path'],
        checkpoint_sha256=selection['checkpoint']['sha256'],
        rendered_observation_config=selection['rendered_observation_config'])
    ops=operations_for(runtime,root);immutable_json(root/'operations.json',ops)
    immutable_json(root/'stage.json',dict(**staged,bc_selection=binding(queue['bc_selection_path']),
        bc_success_test=binding(queue['bc_success_path']),runtime_binding=binding(queue['runtime_bindings_path'])))
    return staged,ops

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-config',type=Path,required=True);parser.add_argument('--queue-sha256',required=True)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-legacy-validation',action='store_true',
        help='Explicit historical MSE-validation execution; new runs use fixed100')
    parser.add_argument('--allow-legacy-cohort-lifecycles',action='store_true',
        help='Explicit historical two-window scene-reset execution; new fixed100 runs are continuous')
    parser.add_argument('--stage',action='store_true',
        help='validate actual BC rendered295/source/capacity receipts and stage immutable plans; no workloads')
    parser.add_argument('--qualification-only',action='store_true',
        help='fresh v3 only: stop after two genuine cohorts/four updates, with owners and compute retained')
    parser.add_argument('--poll-seconds',type=float,default=20);parser.add_argument('--observer-seconds',type=int,default=7*86400)
    args=parser.parse_args()
    require(1<=args.poll_seconds<=60 and 60<=args.observer_seconds<=7*86400,'bounded CPU observer parameters required')
    require(binding(args.queue_config)['raw_sha256']==args.queue_sha256,'queued config raw binding changed')
    queue=validate_queue(json.loads(args.queue_config.read_bytes()))
    if args.execute:
        from pipeline.deployment.training_launch_policy import require_training_launch
        require(queue['schema']!=FIXED100_QUEUE_SCHEMA or args.allow_legacy_cohort_lifecycles,
                'historical fixed100 cohort execution requires --allow-legacy-cohort-lifecycles')
        require_training_launch(validation_enabled=queue['schema'] not in (FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_QUEUE_SCHEMA,CONTINUOUS_FIXED100_8H200_QUEUE_SCHEMA),
            allow_legacy_validation=args.allow_legacy_validation)
    root=Path(queue['root']);root.mkdir(parents=True,exist_ok=True)
    if args.execute and not args.watch:parser.error('--execute requires --watch')
    if args.stage and (args.execute or args.watch):parser.error('--stage is a single CPU-only staging action')
    if args.qualification_only and not args.execute:
        parser.error('--qualification-only requires --watch --execute')
    stop=False
    def requested(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    started=time.monotonic();campaign=None;last=None
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope=(admit_execution_scope(queue,args.queue_sha256,qualification_only=args.qualification_only)
            if args.execute else None)
        while not stop and time.monotonic()-started<args.observer_seconds:
            missing=[k for k in ('bc_selection_path','bc_success_path','runtime_bindings_path') if not Path(queue[k]).is_file()]
            if missing:value=dict(phase='WAITING_FOR_ACTUAL_HANDOFF',missing=missing,production_started=False)
            elif not args.execute:
                value=dict(phase='READY_FOR_REVIEW',production_started=False,
                    native_qualification_completed=False,requires_explicit_execute=True)
                if args.stage:
                    staged,_=prepare_campaign(queue)
                    value.update(phase='STAGED',artifacts=staged)
            else:
                if campaign is None:
                    staged,ops=prepare_campaign(queue)
                    first=bound(staged['first_round']);mse=bound(staged['mse_plan'])
                    begin,log=online_mse_loggers(queue['wandb_entity'],queue['wandb_key_file'])
                    kind=QualificationOnlyCampaign if scope is not None else CoverageMSECampaign
                    extra={'execution_scope':scope} if scope is not None else {}
                    campaign=kind(first,root/'campaign',CoverageCommandOperations(ops,root/'operations'),
                        plan=mse,begin_logging=begin,log_validation=log,**extra)
                state=campaign.tick();value=dict(phase=state['phase'],round_id=state['round_id'],
                    progress=state.get('progress'),qualification_completed=state.get('qualification_completed',False),
                    completed_rounds=len(state['completed_rounds']),production_started=True)
                if scope is not None:
                    value.update(execution_scope=scope,qualification_only=True,
                        full_production_authorized=False,automatic_continuation=False)
                    if state.get('qualification_completed')is True:
                        completion=campaign.qualification_completion()
                        value.update(phase='QUALIFICATION_COMPLETE',
                            completed_optimizer_updates=completion['completed_optimizer_updates'],
                            completed_native_training_cohorts=completion['completed_native_training_cohorts'],
                            native_owners_kept_warm=True,
                            completion=binding(root/'campaign/qualification-only-completion.json'))
            value.update(schema='roborender.full295-production-queue-status.v1',checked_unix=time.time(),
                compute_release_requested=False,native_replay_allowed=False)
            atomic(root/'queue-status.json',value)
            stable={k:v for k,v in value.items() if k!='checked_unix'}
            if stable!=last:print(json.dumps(value,sort_keys=True),flush=True);last=stable
            if not args.watch or value['phase'] in {'COMPLETE','QUALIFICATION_COMPLETE'}:return
            time.sleep(args.poll_seconds)
        atomic(root/'queue-status.json',dict(schema='roborender.full295-production-queue-status.v1',
            phase='CPU_OBSERVER_DETACHED',checked_unix=time.time(),compute_release_requested=False,
            workload_stop_requested=False,native_replay_allowed=False))
if __name__=='__main__':main()
