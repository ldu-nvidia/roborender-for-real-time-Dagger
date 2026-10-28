"""Standalone final-epoch BC -> automatic paired qualification -> scaled295.

Explicit execution only, on already admitted native owners and retained eight
renderers. No allocation, native restart, sampling-process launch or DAgger
launch is implemented. The selected checkpoint is never tuned by probe scores.
"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from pipeline.bc_training import final_selection
from pipeline.deployment import paired_scaled_evaluation as paired
from pipeline.deployment import paired_scaled_runtime as runtime
from pipeline.deployment import scaled_success_dispatch as dispatch
from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
from pipeline.deployment.rendered_success_dispatch import configuration
from pipeline.deployment.wandb_preflight import admit_logging, verify_remote_summary

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='roborender.paired-bc-success-handoff.v1'
ONLINE_SCHEMA='roborender.paired-bc-success-online.v1'


def logging_identity(selection_ref,source_sha256):
    token=dispatch.digest(dict(selection=selection_ref,source_sha256=source_sha256))[:24]
    return 'bc-paired295-'+token


def expected_metrics(result,panel,checkpoint_sha256,source_sha256):
    return {'test/completed_episodes':295,'test/successes':result['successes'],
        'test/success_rate':result['success_rate'],'test/optimizer_steps':0,
        'test/checkpoint_sha256':checkpoint_sha256,'test/source_sha256':source_sha256,
        'test/success_test_protocol':scaled.protocol_identity(panel),
        'test/rendered_student_success_protocol':result['rendered_student_success_protocol']}


def validate_online_handoff(value, *, selection_ref, source_sha256, result):
    selection=final_selection.validate_selection(paired.bound(selection_ref))
    handoff=value.get('bc_handoff')
    if (not isinstance(handoff,dict) or set(handoff)!={'schema','selection','native_comparison','logging'}
            or handoff.get('schema')!=SCHEMA or handoff.get('selection')!=selection_ref
            or paired.bound(handoff['native_comparison'])!={k:v for k,v in value.items() if k!='bc_handoff'}):
        raise ValueError('actual paired BC result and its selected checkpoint need a bound online handoff')
    logging=paired.bound(handoff['logging']);panel=paired.bound(selection['test_panel'])
    expected=dict(schema=ONLINE_SCHEMA,entity='nvidia',project='dagger',
        run_id=logging_identity(selection_ref,source_sha256),
        checkpoint_sha256=selection['checkpoint']['sha256'],source_sha256=source_sha256,
        selection=selection_ref,native_comparison=handoff['native_comparison'],
        expected_summary=expected_metrics(result,panel,selection['checkpoint']['sha256'],source_sha256),
        authenticated=True,online_readback_verified=True)
    if logging!=expected:
        raise ValueError('BC logging receipt does not match the actual final295 metrics/checkpoint/source')
    return logging


class Context:
    """Read-only source/owner admission plus content-addressed checkpoint staging."""
    def __init__(self, bindings):
        from pipeline.deployment.beta_site import validate_site
        from pipeline.deployment.retained_campaign_site import load_retained,validate_eight_provider_root
        fields={'schema','site','resident_pool','retained','source_sha256','paired_evaluation'}
        if set(bindings)!=fields or bindings['schema']!='roborender.full295-runtime-bindings.v4':
            raise ValueError('standalone paired BC requires exact retained runtime v4')
        self.bindings=bindings;self.site=validate_site(paired.bound(bindings['site']))
        if self.site['schema']!='roborender.leo-site.v1':
            raise ValueError('paired BC supports only the explicitly admitted Leo RTX site')
        self.retained=load_retained(bindings['retained']['path'],bindings['retained']['raw_sha256'])
        self.pod=self.retained['allocation_name'];self.gpu_ssh=Path(self.retained['ssh_config'])
        self.source=Path(self.site['rtx_source_root']);self.pool_ref=bindings['resident_pool']
        if (set(self.pool_ref)!={'path','sha256'} or not Path(self.pool_ref['path']).is_absolute()
                or self.retained['schema']!='roborender.full295-retained-h200.v3'):
            raise ValueError('exact existing paired pool and eight-renderer retained allocation required')
        validate_eight_provider_root(ROOT,self.site['source_manifest'],bindings['source_sha256'])
        self.runtime=paired.bound(bindings['paired_evaluation'])
        self.native,self.fleet=runtime.validate_runtime(self.runtime,source_sha256=bindings['source_sha256'],retained=self.retained)
        if self.native['source_root']!=str(self.source):
            raise ValueError('native frozen source path differs from admitted BC evaluator')

    def ssh(self,host='beta'):
        from pipeline.deployment.beta_site import Site
        return Site.ssh(self,host)

    def command(self,argv,host='beta',*,timeout=180):
        try:
            value=subprocess.run(self.ssh(host)+[shlex.join(list(map(str,argv)))],
                capture_output=True,timeout=timeout,check=False)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('paired BC read-only admission response unavailable') from error
        if value.returncode:raise RuntimeError('paired BC read-only admission failed; no workload replay')
        if len(value.stdout)>8<<20:raise ValueError('bounded BC admission output required')
        return json.loads(value.stdout)

    def verify(self):
        from pipeline.deployment.rtx_actor.production_resident_pool import validate_pool
        code=(runtime.REJECT_OPTIMIZED_PYTHON+'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); '
            'assert p.is_absolute() and p.resolve()==p and p.stat().st_size<=4<<20; '
            'print(p.read_text())')
        pool=self.command(['python3','-B','-c',code,self.pool_ref['path']])
        validate_pool(pool)
        if dispatch.digest(pool)!=self.pool_ref['sha256']:
            raise ValueError('actual remote pool bytes/identity differ')
        runtime.validate_runtime(self.runtime,source_sha256=self.bindings['source_sha256'],pool=pool,retained=self.retained)
        code=(runtime.REJECT_OPTIMIZED_PYTHON+'import json,pathlib,sys;sys.path.insert(0,sys.argv[1]); '
            'from pipeline.deployment.h200.retained_renderer_attachment import verify_release; '
            'v=verify_release(pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]); '
            'print(json.dumps(dict(source_sha256=v["source_sha256"],file_count=v["file_count"])))')
        self.command(['python3','-B','-c',code,self.source,self.site['rtx_source_manifest'],self.bindings['source_sha256']])
        fleet=self.runtime['renderer_fleet']
        result=self.command([self.site['cpu_python'],'-B',ROOT/'pipeline/deployment/h200/retained_renderer_attachment.py',
            '--fleet',fleet['path'],'--fleet-sha256',fleet['raw_sha256']],host='gpu')
        if result.get('allocation_retained') is not True or result.get('compute_release_authorized') is not False:
            raise ValueError('actual retained renderer attachment required')
        return pool

    def stage_checkpoint(self,checkpoint):
        from pipeline.deployment.bc_success_dispatch import Leo
        class PinnedStager(Leo):
            def command(self,argv,**kwargs):
                # Legacy staging uses assert-based no-clobber/hash checks.
                # The new route must not inherit an optimizer-disabled guard.
                if argv[:2]!=['python3','-c'] or len(argv)<3:
                    raise ValueError('unexpected checkpoint staging operation')
                return super().command([*argv[:2],runtime.REJECT_OPTIMIZED_PYTHON+argv[2],*argv[3:]],**kwargs)
        destination=Path(self.native['assets_root'])/(checkpoint['sha256']+'.pt')
        PinnedStager().stage(checkpoint['path'],destination,checkpoint['sha256'])


def execute(selection_ref,bindings_ref,output,*,deadline=None,context_factory=Context,wandb_module=None):
    selection=final_selection.validate_selection(paired.bound(selection_ref))
    bindings=paired.bound(bindings_ref);context=context_factory(bindings)
    source=bindings['source_sha256'];panel=paired.final_panel(paired.bound(selection['test_panel']))
    config=configuration(selection['rendered_observation_config'],panel,selection['checkpoint']['sha256'])
    endpoints=config['renderer']['endpoints']
    if len(endpoints)!=8 or len(context.fleet['workers'])!=8 or any(any(worker.get(k)!=v for k,v in endpoint.items())
            for worker,endpoint in zip(context.fleet['workers'],endpoints)):
        raise ValueError('BC rendering dependencies differ from the actual retained eight-renderer fleet')
    root=Path(output)
    if not root.is_absolute() or root.resolve()!=root:raise ValueError('physical absolute BC output required')
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'bc-paired-controller.lock'
    if lock.is_symlink():raise ValueError('BC controller lock alias forbidden')
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=dict(schema=SCHEMA,selection=selection_ref,runtime=bindings_ref,source_sha256=source,
            checkpoint_sha256=selection['checkpoint']['sha256'],panel=selection['test_panel'],
            optimizer_steps=0,checkpoint_selection_feedback=False,compute_release_authorized=False)
        dispatch.immutable(root/'bc-intent.json',intent)
        context.verify()
        sampler=runtime.ResourceJournal(context.runtime['resource_sampler'],context.runtime)
        transport=runtime.Transport(context,context.native,root,sampler)
        os.environ['NETRC']=context.site['netrc_file']
        api,identity=admit_logging('online','nvidia')
        if identity.get('authenticated') is not True:raise RuntimeError('authenticated online BC logging required')
        if wandb_module is None:
            import wandb as wandb_module
        run_id=logging_identity(selection_ref,source)
        log=wandb_module.init(entity='nvidia',project='dagger',id=run_id,
            name=selection['run_name']+'-success-scaled-b16',group='BC',job_type='success-test',
            mode='online',resume='allow',dir=str(root),config=dict(checkpoint_sha256=selection['checkpoint']['sha256'],
                source_sha256=source,test_panel=selection['test_panel'],selection_feedback=False,optimizer_steps=0,
                success_test_protocol=scaled.protocol_identity(panel)))
        last={}
        def progress(label,value):
            key=(value['committed_cohorts'],value['complete'])
            if last.get(label)!=key:
                prefix='test' if label=='final' else 'qualification/'+label
                log.log({prefix+'/committed_cohorts':key[0],prefix+'/complete':key[1]});last[label]=key
        try:
            if deadline is not None and time.time()>=deadline:
                raise TimeoutError('BC observer deadline; native owners, sampler and compute retained')
            context.stage_checkpoint(selection['checkpoint'])
            plan=paired.make_run(panel,config,context.native['workers'],command_prefix=run_id,
                source_sha256=source,arm='bc',cpu_validation=context.runtime['cpu_validation'])
            value=paired.Run(plan,root,transport,sampler,deadline=deadline,progress=progress).execute()
            result=paired.validate_final_evidence(value,panel=panel,checkpoint_sha256=selection['checkpoint']['sha256'],
                source_sha256=source,arm='bc',expected_rendered_config=config)
            context.verify();paired.idle_barrier(transport,context.native['workers'])
            metrics=expected_metrics(result,panel,selection['checkpoint']['sha256'],source)
            log.log(metrics);log.finish()
            verify_remote_summary(api,'nvidia',run_id,metrics)
            native=paired.ref(root/'comparison.json')
            online=dict(schema=ONLINE_SCHEMA,entity='nvidia',project='dagger',run_id=run_id,
                checkpoint_sha256=selection['checkpoint']['sha256'],source_sha256=source,selection=selection_ref,
                native_comparison=native,expected_summary=metrics,authenticated=True,online_readback_verified=True)
            dispatch.immutable(root/'bc-wandb.json',online)
            final=dict(value,bc_handoff=dict(schema=SCHEMA,selection=selection_ref,native_comparison=native,
                logging=paired.ref(root/'bc-wandb.json')))
            validate_online_handoff(final,selection_ref=selection_ref,source_sha256=source,result=result)
            dispatch.immutable(root/'bc-complete.json',final)
            return dict(status='BC295_COMPLETE',result=paired.ref(root/'bc-complete.json'),
                completed_episodes=295,successes=result['successes'],success_rate=result['success_rate'],
                optimizer_steps=0,native_workers_retained=True,compute_release_authorized=False,
                dispatch_output=str(root),wandb=online)
        except BaseException:
            # Closing a logging session never stops native owners or sampler.
            # Same run ID and saved coordinator intents reconcile on re-entry.
            try:log.finish(exit_code=1)
            except Exception:pass
            raise
