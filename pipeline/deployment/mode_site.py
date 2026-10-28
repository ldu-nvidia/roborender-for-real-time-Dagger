"""Concrete Leo/retained-H200 adapter for the mode-selected workflow.

No fake backend, cloud allocation, release or automatic native restart exists
here. All learning and test phases delegate to the established implementations.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from pipeline.bc_training.handoff_common import binding,bound,canonical
from pipeline.bc_training.plan_success_test import checkpoint_identity
from pipeline.deployment.beta_operations import atomic
from pipeline.deployment.h200.bc_workload import BCWorkload
from pipeline.deployment.retained_campaign_site import load_retained
from pipeline.deployment.run_mode import bc_job,stage_dagger,test_command
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.deployment.rtx_actor.native_thp_policy import INHERITED, validate_policy


def native_policy_from_settings(settings):
    """Versioned execution opt-in; old production settings remain unchanged."""
    required={'schema','site','retained','bc_capacity','compute_release_authorized'}
    if settings.get('schema')=='roborender.mode-site.v1' and set(settings)==required:
        return INHERITED
    if settings.get('schema')=='roborender.mode-site.v2' and set(settings)==required|{'native_thp_policy'}:
        return validate_policy(settings['native_thp_policy'])
    raise ValueError('explicit mode-site bindings required')


def stage_native_schedule(plan,local_root,remote_root,publish):
    """Compile where the panel exists, transfer exact native requests only.

    CPU sampling/test metadata never becomes an assumed RTX filesystem path.
    Relocation changes schedule paths only, not request bytes, hashes or seeds.
    """
    from pipeline.legacy.bc_collection.plan_native_sessions import build_schedule,write_schedule
    sampling=bound(plan['bc_population'])
    sessions=build_schedule(sampling,run_id='bc-leo-'+plan['run_id'],rtx_gpus=(0,),cohorts_per_session=4,
        motion_root='/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775')
    local=Path(local_root);remote=Path(remote_root)
    schedule=write_schedule(local,sampling,sessions)
    for row in schedule['sessions']:
        for source in [Path(row['session_path']),*sorted(Path(row['requests_dir']).glob('*.json'))]:
            publish(remote/source.relative_to(local),source.read_bytes())
        row['session_path']=str(remote/Path(row['session_path']).relative_to(local))
        row['requests_dir']=str(remote/Path(row['requests_dir']).relative_to(local))
    raw=canonical(schedule);path=remote/'schedule.json';publish(path,raw)
    return dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest())


class ModeSite:
    def __init__(self,plan,settings):
        self.native_thp_policy=native_policy_from_settings(settings)
        self.plan,self.settings=plan,settings
        if settings['compute_release_authorized'] is not False:
            raise ValueError('mode execution cannot authorize compute release')
        self.site=bound(settings['site']);self.root=Path(plan['output_root'])
        self.retained=load_retained(settings['retained']['path'],settings['retained']['raw_sha256'])
        suffix=hashlib.sha256(canonical([plan['run_id'],plan['source_sha256']])).hexdigest()[:12]
        self.host=BCWorkload(self.site,'dagger-leo-bc-mode-'+suffix,gpus=4)
        self.host.pod=self.retained['allocation_name']
        self.host.gpu_ssh=Path(self.retained['ssh_config'])
        self.source=Path(self.site['rtx_source_root'])
        self.remote=Path('/home/ldu/roborender/runtime/mode-runs')/plan['run_id']/plan['source_sha256'][:12]
        self.poolrun='standing-isaac-leo-prod-'+('smoke-' if plan['mode']=='smoke' else '')+suffix
        self.poolroot=Path('/home/ldu/roborender/runtime/standing-isaac')/self.poolrun
        self.poolpath=self.poolroot/'pool.json'
        os.environ['NETRC']=self.site['netrc_file']
        os.environ['WANDB_ENTITY']='nvidia'

    def native(self,argv,**kwargs):return self.host.command(argv,'beta',**kwargs)
    def read(self,path):return self.host.read(path,'beta')

    def verify_h200(self):
        info=self.host.json(['python3','-c',
            "import os,json,subprocess;print(json.dumps({'owner':os.environ.get('ROBORENDER_BETA_RUN_ID'),'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()}))"],'gpu')
        if info!={'owner':self.retained['allocation_owner_run_id'],'gpus':self.retained['gpu_uuids']}:
            raise RuntimeError('retained H200 identity changed')
        return info

    def prepare_bc(self,completed):
        self.verify_h200()
        from pipeline.deployment.source_snapshot import verify
        verify(Path(self.plan['source_root']),json.loads(Path(self.site['source_manifest']).read_bytes()))
        from pipeline.deployment.wandb_preflight import admit_logging
        admit_logging('online','nvidia')
        if self.plan['mode']!='smoke':
            # Existing admitted production data is reused, never recollected
            # or silently downgraded to the smoke population.
            value=bound(self.plan['bc_population'])
            job=bc_job(self.plan)  # Revalidates reuse admission before any learner.
            test=value.get('closed_loop_test',{}).get('panel')
            if job.get('schema')=='roborender.bc-admitted-reuse-pipeline.v1' and test!=self.plan['test_panel']:
                # The immutable data admission predates the student observation
                # change. Reuse is allowed only for the exact same fixed cases;
                # no dataset admission or historical outcome is relabeled.
                from pipeline.deployment.rtx_actor.production_success_contract import build_rendered_panel
                if build_rendered_panel(bound(test)) != bound(self.plan['test_panel']):
                    raise ValueError('reused BC and production test panels differ')
            self._bootstrap_start()
            return dict(mode='production',protocol=self.plan['bc_population'],reused=True)
        from pipeline.legacy.bc_collection.transfer_native_cohort import LeoTransport,transfer_cohort,transfer_execution_evidence
        from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool,read_document
        sampling=bound(self.plan['bc_population'])
        schedule=stage_native_schedule(self.plan,self.root/'bc-native-schedule',self.remote/'bc-native-schedule',
            lambda path,raw:self.host.publish(path,raw,'beta',0o444))
        capacity=self.settings['bc_capacity']
        out=self.remote/'collection'
        argv=[self.site['rtx_python'],self.source/'pipeline/deployment/rtx_actor/collect_mode_bc.py',
            '--schedule',schedule['path'],'--schedule-sha256',schedule['raw_sha256'],
            '--capacity-receipt',capacity['path'],'--capacity-receipt-sha256',capacity['raw_sha256'],
            '--source-manifest',self.site['rtx_source_manifest'],'--run-id','bc-leo-'+self.plan['run_id'],
            '--output',out,'--execute']
        self.host.process('mode-bc-collection','start',host='beta',argv=argv,ttl=15000)
        self.host.await_process('mode-bc-collection',host='beta',timeout=15060)
        campaign=self.read(out/'campaign.json')
        dataset=Path(self.plan['settings']['dataset_root'])
        transport=LeoTransport(self.site['ssh_config'])
        cohorts=[];transfers=[]
        for session in campaign['sessions']:
            runtime=Path(session['runtime'])
            result=self.read(runtime/'bc-session-result.json')
            if result.get('status')!='complete':raise RuntimeError('BC native session incomplete')
            for row in result['cohorts']:
                directory=runtime/Path(row['output_dir']).relative_to('/dagger-runtime')
                raw=transport.read(str(directory/'cohort-manifest.json'))
                transfer=transfer_cohort(raw,remote_runtime=str(runtime),dataset_root=dataset,transport=transport)
                transfer_execution_evidence(raw,remote_runtime=str(runtime),dataset_root=dataset,transport=transport)
                # The transfer helper independently imports the original
                # native execution receipts; do not synthesize provenance.
                audit=dataset/'manifests/cohorts'/transfer['cohort_id']
                cohorts.append(read_document(audit/'native-cohort-manifest.json'))
                transfers.append(read_document(audit/'transfer-receipt.json'))
        manifest,seal=build_pool(sampling,cohorts,transfers,dataset_root=dataset,expected_total=18)
        atomic(dataset/'manifest.json',manifest);atomic(dataset/'SEALED.json',seal)
        self._bootstrap_start()
        return dict(mode='smoke',manifest=binding(dataset/'manifest.json'),seal=binding(dataset/'SEALED.json'),
            native_collection=True,train_chunks=16,validation_chunks=2)

    def _bootstrap_start(self):
        inventory=self.plan['settings']['inventory']
        dest=self.remote/'full-motion-inventory.json'
        self.host.publish(dest,Path(inventory['path']).read_bytes(),'beta',0o444)
        spec=dict(schema='roborender.production-b2-bootstrap.v1',mode=self.plan['mode'],
            source_root=str(self.source),source_manifest=self.site['rtx_source_manifest'],
            source_sha256=self.plan['source_sha256'],run_id=self.poolrun,root=str(self.poolroot),
            python=self.site['rtx_python'],motion_inventory=dict(path=str(dest),raw_sha256=inventory['raw_sha256']))
        if self.native_thp_policy!=INHERITED:
            spec['native_thp_policy']=self.native_thp_policy
        path=self.remote/'bootstrap.json';raw=canonical(spec)
        self.host.publish(path,raw,'beta',0o444)
        self.host.process('mode-native-bootstrap','start',host='beta',ttl=2400,argv=[self.site['rtx_python'],
            self.source/'pipeline/deployment/rtx_actor/bootstrap_production_resident_pool.py',
            '--spec',path,'--spec-sha256',hashlib.sha256(raw).hexdigest(),'--execute'])

    def train_bc(self,completed):
        self.verify_h200()
        job=bc_job(self.plan)
        path=self.root/'bc-job.json'
        if not path.exists():atomic(path,job)
        if json.loads(path.read_bytes())!=job:raise ValueError('BC job changed')
        # Invoke the same offline executor with the frozen generated argv.
        module=('reuse_pipeline' if job['schema']=='roborender.bc-admitted-reuse-pipeline.v1' else 'offline_pipeline')
        code=f"import json,sys;from pipeline.bc_training.{module} import execute;execute(json.load(open(sys.argv[1])))"
        training_budget=43200 if self.plan['mode']=='production' else 21600
        self.host.process('mode-bc-training','start',host='gpu',ttl=training_budget,
            argv=[self.site['cpu_python'],'-c',code,str(path)],
            env={'NETRC':self.site['netrc_file'],'WANDB_ENTITY':'nvidia','PYTHONPATH':self.plan['settings']['source_root']})
        self.host.await_process('mode-bc-training',host='gpu',timeout=training_budget+60)
        directory=Path(job['training_checkpoint_dir'])
        state=json.loads((directory/'validation_state.json').read_bytes())
        checkpoint=checkpoint_identity(state['selected_checkpoint'])
        return dict(training_state=binding(directory/'validation_state.json'),
            training_config=binding(directory/'run_config.json'),checkpoint=checkpoint)

    def _pool(self):
        if self.host.process('mode-native-bootstrap','status',host='beta')['state']=='ABSENT':
            self._bootstrap_start()
        self.host.await_process('mode-native-bootstrap',host='beta',timeout=2460)
        if self.native_thp_policy!=INHERITED:
            # The frozen startup receipt must attest the actual process setting;
            # a config field alone cannot qualify a different/old pool.
            self.native([self.site['rtx_python'],'-c',
                'import json,sys;sys.path.insert(0,sys.argv[1]);'
                'from production_resident_pool import read,validate_native_policy;'
                'validate_native_policy(json.loads(read(sys.argv[2])),sys.argv[3])',
                self.source/'pipeline/deployment/rtx_actor',self.poolpath,self.native_thp_policy])
        return self.read(self.poolpath)

    def _pool_action(self,action,commands=None):
        argv=[self.site['rtx_python'],self.source/'pipeline/deployment/rtx_actor/production_resident_pool.py',
              '--pool',self.poolpath,'--action',action]
        if commands:
            path=self.remote/(commands[0]['command_id']+'-'+action+'.json')
            self.host.publish(path,canonical(commands),'beta',0o444);argv+=['--commands',path]
        return self.host.json(argv,'beta',timeout=21900 if action=='wait' else 60)

    def test_bc(self,completed):
        train=bound(next(r['receipt'] for r in completed if r['phase']=='train_bc'))
        pool=self._pool();checkpoint=train['checkpoint']
        assets=Path(pool['assets_root']);jobs=Path(pool['jobs_root'])
        remote=assets/(checkpoint['sha256']+'.pt')
        subprocess.run(['scp','-q','-F',self.site['ssh_config'],checkpoint['path'],
            self.site['rtx_ssh_host']+':'+str(remote)],check=True,timeout=600)
        from pipeline.deployment.rendered_success_dispatch import artifact
        panel=bound(self.plan['test_panel'])
        rendered_config, rendered_raw, rendered_sha=artifact(
            self.plan['settings'].get('rendered_observation_config'),panel,checkpoint['sha256'])
        panel_file='image3f-test-panel-'+self.plan['test_panel']['raw_sha256'][:12]+'.json'
        config_file='rendered-success-'+rendered_sha+'.json'
        self.host.publish(jobs/panel_file,Path(self.plan['test_panel']['path']).read_bytes(),'beta',0o444)
        self.host.publish(jobs/config_file,rendered_raw,'beta',0o444)
        commands=[test_command(self.plan,stage_name='bc',checkpoint_sha256=checkpoint['sha256'],
            panel_file=panel_file,output_name=self.plan['run_id']+'-bc-image3f-'+rendered_sha[:12],
            rendered_config_file=config_file)]
        self._pool_action('submit',commands);result=self._pool_action('wait',commands)
        outcome=result['results'][0]['receipt']['result']
        if self.plan['mode']=='smoke':
            from pipeline.deployment.rtx_actor.smoke_success_contract import validate_smoke_result
            validate_smoke_result(panel,outcome,checkpoint['sha256'],allowed_arms=('bc',),rendered_config=rendered_config)
        else:
            from pipeline.deployment.rtx_actor.production_success_contract import validate_production_result
            validate_production_result(panel,outcome,checkpoint['sha256'],allowed_arms=('bc',),rendered_config=rendered_config)
        from pipeline.deployment.wandb_preflight import admit_logging,verify_remote_summary
        import wandb
        api,_=admit_logging('online','nvidia')
        cfg=bound(train['training_config'])
        logging=json.loads((Path(cfg['output'])/'wandb-run.json').read_bytes());rid=logging.get('id')
        if not rid:raise RuntimeError('BC run ID unavailable for online test logging')
        metrics={'test/successes':outcome['successes'],'test/completed':outcome['completed_episodes'],
            'test/goal_reached_rate':outcome['success_rate'],
            'test/observation_mode':panel['observation_mode'],
            'test/rendered_observation_config_sha256':rendered_sha}
        with wandb.init(entity='nvidia',project='dagger',id=rid,resume='must',mode='online') as run:
            run.log(metrics);run.summary.update(metrics)
        verify_remote_summary(api,'nvidia',rid,metrics)
        return dict(result=outcome,checkpoint_sha256=checkpoint['sha256'],online_metrics_verified=True,
            native_receipt=result)

    def dagger(self,completed):
        train=bound(next(r['receipt'] for r in completed if r['phase']=='train_bc'))
        checkpoint=train['checkpoint']
        staged=stage_dagger(self.plan,selected_checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'])
        from pipeline.dagger.coverage_queue import operations_for
        pool=self._pool()
        runtime=dict(schema='roborender.full295-runtime-bindings.v1',site=self.settings['site'],
            retained=self.settings['retained'],source_sha256=self.plan['source_sha256'],
            resident_pool=dict(path=str(self.poolpath),sha256=hashlib.sha256(canonical(pool)).hexdigest()))
        root=self.root/'dagger/campaign';root.mkdir(parents=True,exist_ok=True)
        token=root/'services.token'
        import secrets
        fd=os.open(token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:stream.write(secrets.token_hex(32)+'\n')
        ops=operations_for(runtime,root);immutable_json(root/'operations.json',ops)
        subprocess.run([self.site['cpu_python'],'-m','pipeline.dagger.mse_campaign',
            '--first-round-config',staged['config_path'],'--mse-plan',staged['plan_path'],
            '--root',str(root),'--operations',str(root/'operations.json'),'--execute'],check=True,
            cwd=self.plan['settings']['source_root'])
        return dict(campaign_state=binding(root/'state.json'))

    def validate(self,phase,receipt,completed):
        if phase=='prepare_bc':
            if self.plan['mode']=='smoke':
                manifest,seal=bound(receipt['manifest']),bound(receipt['seal'])
                if manifest.get('partition_counts')!={'train':16,'validation':2} or seal.get('verdict')!='PASS':
                    raise RuntimeError('smoke native data quotas/seal differ')
            else:bound(receipt['protocol'])
        elif phase=='train_bc':
            state=bound(receipt['training_state']);history=state.get('history',[])
            if state.get('status')!='complete' or state.get('global_step',0)<2 or len(history)<3:
                raise RuntimeError('BC optimizer and held-out checks incomplete')
            if any(not math.isfinite(r['mse']) for r in history):raise RuntimeError('nonfinite BC MSE')
            best=min(history,key=lambda r:(r['mse'],r['epoch']))
            if (state.get('best_mse'),state.get('best_epoch'))!=(best['mse'],best['epoch']):
                raise RuntimeError('BC did not select minimum validation MSE')
            if checkpoint_identity(receipt['checkpoint']['path'])['sha256']!=receipt['checkpoint']['sha256']:
                raise RuntimeError('BC selected checkpoint changed')
        elif phase=='test_bc':
            from pipeline.deployment.rtx_actor.smoke_success_contract import validate_smoke_result
            from pipeline.deployment.rtx_actor.production_success_contract import validate_production_result
            validator=validate_smoke_result if self.plan['mode']=='smoke' else validate_production_result
            from pipeline.deployment.rendered_success_dispatch import configuration
            panel=bound(self.plan['test_panel'])
            config=configuration(self.plan['settings'].get('rendered_observation_config'),panel,receipt['checkpoint_sha256'])
            validator(panel,receipt['result'],receipt['checkpoint_sha256'],allowed_arms=('bc',),rendered_config=config)
            if receipt.get('online_metrics_verified') is not True:raise RuntimeError('BC test not logged online')
        elif phase=='dagger':
            state=bound(receipt['campaign_state']);rounds=state.get('completed_rounds',[])
            if state.get('phase')!='COMPLETE' or [r['round_id'] for r in rounds]!=[1,2,3,4]:
                raise RuntimeError('all four real DAgger rounds are required')
            for r in rounds:
                if r['test'].get('completed')!=self.plan['test']['cases'] or not r['wandb'].get('online_metrics_verified'):
                    raise RuntimeError('complete independent test and online readback required for every round')
