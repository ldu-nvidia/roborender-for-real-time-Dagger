"""CPU orchestration checks only: no simulated receipts imply live readiness."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from pipeline.deployment import paired_bc_success as bc
from pipeline.deployment import paired_scaled_evaluation as paired
from pipeline.deployment import scaled_success_dispatch as dispatch
from pipeline.deployment import bc_success_dispatch as cli
from test.pipeline.test_scaled_success_qualification import evidence, accepted
from test.pipeline.test_paired_scaled_evaluation import owners, publish


@pytest.fixture
def setup(tmp_path, monkeypatch, evidence):
    checkpoint=tmp_path/'selected.pt';checkpoint.write_bytes(b'CPU fixture, never loaded')
    selection=dict(schema='roborender.bc-production-selected.v2',run_name='cpu-only-bc',
        checkpoint=dict(path=str(checkpoint),sha256=evidence['config']['checkpoint_sha256'],bytes=checkpoint.stat().st_size),
        test_panel=publish(tmp_path/'panel.json',evidence['panel']),
        rendered_observation_config=publish(tmp_path/'config.json',evidence['config']))
    selected=publish(tmp_path/'selected.json',selection)
    binding=publish(tmp_path/'runtime.json',dict(schema='roborender.full295-runtime-bindings.v4',source_sha256='a'*64))
    events=[];run_ids=[];logs=[]
    def validate(value):
        if value!=selection:raise ValueError('wrong immutable selected BC')
        return value
    monkeypatch.setattr(bc.final_selection,'validate_selection',validate)
    native=dict(workers=owners())
    context=SimpleNamespace(fleet=dict(workers=evidence['config']['renderer']['endpoints']),
        native=native,runtime=dict(cpu_validation={},resource_sampler={}),site=dict(netrc_file='/not-read.netrc'),
        verify=lambda:events.append('verify'),stage_checkpoint=lambda cp:events.append(('stage',cp)))
    sampler=object();transport=object()
    monkeypatch.setattr(bc.runtime,'ResourceJournal',lambda *args:sampler)
    monkeypatch.setattr(bc.runtime,'Transport',lambda *args:transport)
    monkeypatch.setattr(bc,'admit_logging',lambda *args:(events.append('authenticate') or object(),dict(authenticated=True)))
    monkeypatch.setattr(bc,'verify_remote_summary',lambda *args:events.append('readback'))
    monkeypatch.setattr(paired,'idle_barrier',lambda *args:events.append('idle'))
    result=dict(completed_episodes=295,successes=200,success_rate=200/295,
        arm='bc',qualification_only=False,rendered_student_success_protocol=dict(cpu_fixture=True))
    monkeypatch.setattr(paired,'validate_final_evidence',lambda *args,**kwargs:result)
    class Log:
        def log(self,value):logs.append(value)
        def finish(self,**kwargs):events.append(('finish',kwargs))
    def init(**kwargs):
        assert kwargs['mode']=='online' and kwargs['resume']=='allow'
        run_ids.append(kwargs['id']);events.append('wandb-init');return Log()
    wandb=SimpleNamespace(init=init)
    class Run:
        def __init__(self,plan,root,t,s,*,deadline,progress):
            assert t is transport and s is sampler
            assert plan['arm']=='bc' and plan['config']==evidence['config']
            self.root,self.plan,self.progress=root,plan,progress
        def execute(self):
            events.append('paired-run')
            self.progress('b16',dict(committed_cohorts=2,complete=True))
            self.progress('final',dict(committed_cohorts=19,complete=True))
            # Core paired tests exercise actual canonical native evidence.
            # Here only adapter publication/logging order is under test.
            value=dict(schema=paired.FINAL_SCHEMA,arm='bc',checkpoint_sha256=self.plan['config']['checkpoint_sha256'])
            dispatch.immutable(self.root/'comparison.json',value)
            return value
    monkeypatch.setattr(paired,'Run',Run)
    return dict(selected=selected,binding=binding,selection=selection,context=context,
        factory=lambda _:context,events=events,run_ids=run_ids,logs=logs,result=result,
        output=tmp_path/'output',wandb=wandb)


def execute(case, **kwargs):
    return bc.execute(case['selected'],case['binding'],case['output'],context_factory=case['factory'],
        wandb_module=case['wandb'],**kwargs)


def test_paired_bc_uses_same_coordinator_and_publishes_only_after_online_readback(setup):
    value=execute(setup)
    assert value['status']=='BC295_COMPLETE' and value['completed_episodes']==295
    assert value['optimizer_steps']==0 and value['compute_release_authorized'] is False
    final=paired.bound(value['result'])
    assert bc.validate_online_handoff(final,selection_ref=setup['selected'],source_sha256='a'*64,
        result=setup['result'])==value['wandb']
    events=setup['events']
    assert events.index('authenticate')<events.index('wandb-init')<events.index('paired-run')<events.index('readback')
    assert events.count('verify')==2 and events[-1]=='readback'
    assert any('qualification/b16/complete' in row for row in setup['logs'])
    assert any('test/completed_episodes' in row for row in setup['logs'])


def test_same_bc_intent_and_logging_identity_reconcile_without_new_namespace(setup):
    first=execute(setup);second=execute(setup)
    assert first==second and len(set(setup['run_ids']))==1
    intent=paired.bound(paired.ref(setup['output']/'bc-intent.json'))
    assert not intent['checkpoint_selection_feedback'] and not intent['compute_release_authorized']


@pytest.mark.parametrize('fault',['authentication','init','native','readback'])
def test_failure_never_publishes_completed_handoff_or_stops_native(setup,monkeypatch,fault):
    def fail(*a,**k):raise RuntimeError('CPU injected '+fault)
    if fault=='authentication':monkeypatch.setattr(bc,'admit_logging',fail)
    elif fault=='init':setup['wandb'].init=fail
    elif fault=='native':monkeypatch.setattr(paired.Run,'execute',fail)
    else:monkeypatch.setattr(bc,'verify_remote_summary',fail)
    with pytest.raises(RuntimeError,match='CPU injected'):execute(setup)
    assert not (setup['output']/'bc-complete.json').exists()
    assert not (setup['output']/'bc-wandb.json').exists()
    if fault in ('authentication','init'):
        assert not any(isinstance(event,tuple) and event[0]=='stage' for event in setup['events'])


def test_readback_recovery_reuses_native_comparison_and_same_online_run(setup,monkeypatch):
    def fail(*a):raise RuntimeError('readback pending')
    monkeypatch.setattr(bc,'verify_remote_summary',fail)
    with pytest.raises(RuntimeError,match='readback'):execute(setup)
    old=paired.ref(setup['output']/'comparison.json')
    monkeypatch.setattr(bc,'verify_remote_summary',lambda *args:None)
    value=execute(setup)
    assert paired.bound(value['result'])['bc_handoff']['native_comparison']==old
    assert len(set(setup['run_ids']))==1


def test_changed_renderer_and_changed_saved_intent_fail_before_staging(setup):
    old=deepcopy(setup['context'].fleet)
    setup['context'].fleet['workers']=old['workers'][:7]
    with pytest.raises(ValueError):execute(setup)
    assert not setup['events']
    setup['context'].fleet=old
    execute(setup);setup['events'].clear()
    other=publish(setup['output'].parent/'other-binding.json',dict(schema='different',source_sha256='a'*64))
    setup['binding']=other
    with pytest.raises(ValueError):execute(setup)
    assert not setup['events']


@pytest.mark.parametrize('fault',['selection','native','metrics','source','boolean'])
def test_online_receipt_does_not_admit_another_result_or_readiness_boolean(setup,fault):
    value=execute(setup);final=deepcopy(paired.bound(value['result']))
    online=deepcopy(value['wandb'])
    if fault=='selection':final['bc_handoff']['selection']['raw_sha256']='f'*64
    elif fault=='native':final['arm']='teacher'
    else:
        if fault=='metrics':online['expected_summary']['test/successes']+=1
        elif fault=='source':online['source_sha256']='b'*64
        else:online['online_readback_verified']=False
        final['bc_handoff']['logging']=publish(setup['output']/(fault+'.json'),online)
    with pytest.raises(ValueError):bc.validate_online_handoff(final,selection_ref=setup['selected'],
        source_sha256='a'*64,result=setup['result'])


def test_context_rejects_unversioned_or_mixed_runtime_before_remote_calls():
    with pytest.raises(ValueError,match='exact retained runtime v4'):bc.Context({})


def test_context_rejects_other_sites_before_loading_or_touching_native(tmp_path,monkeypatch):
    from pipeline.deployment import beta_site,retained_campaign_site
    monkeypatch.setattr(beta_site,'validate_site',lambda value:value)
    monkeypatch.setattr(retained_campaign_site,'load_retained',lambda *a:pytest.fail('non-Leo admission continued'))
    fields=dict(schema='roborender.full295-runtime-bindings.v4',
        site=publish(tmp_path/'site.json',dict(schema='roborender.beta-site.v2')),
        resident_pool={},retained={},source_sha256='a'*64,paired_evaluation={})
    with pytest.raises(ValueError,match='Leo RTX'):bc.Context(fields)


def test_bc_generated_admission_scripts_reject_optimized_python_before_reading(tmp_path,monkeypatch):
    from pipeline.deployment.rtx_actor import production_resident_pool
    monkeypatch.setattr(production_resident_pool,'validate_pool',lambda value:value)
    monkeypatch.setattr(bc.runtime,'validate_runtime',lambda *a,**k:None)
    context=object.__new__(bc.Context);context.pool_ref=dict(path='/not-read',sha256=dispatch.digest({}))
    context.runtime=dict(renderer_fleet=dict(path='/not-read',raw_sha256='a'*64))
    context.retained={};context.bindings=dict(source_sha256='a'*64);context.source=Path('/not-imported')
    context.site=dict(rtx_source_manifest='/not-read',cpu_python=sys.executable)
    scripts=[]
    def command(argv,**kwargs):
        if '-c' in argv:
            script=argv[argv.index('-c')+1];scripts.append(script)
            result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
            assert result.returncode!=0 and 'optimized Python is forbidden' in result.stderr
            assert 'IndexError' not in result.stderr and 'ModuleNotFoundError' not in result.stderr
            return {}
        return dict(allocation_retained=True,compute_release_authorized=False)
    context.command=command
    context.verify()
    assert len(scripts)==2


def test_new_checkpoint_staging_guards_all_legacy_remote_assert_scripts(tmp_path,monkeypatch):
    scripts=[];copied=[];invoke=subprocess.run
    def command(self,argv,**kwargs):
        scripts.append(argv[2])
        result=invoke([sys.executable,'-O','-B','-c',argv[2]],capture_output=True,text=True,timeout=10)
        assert result.returncode!=0 and 'optimized Python is forbidden' in result.stderr
        return b'missing' if len(scripts)==1 else b''
    monkeypatch.setattr(cli.Leo,'command',command)
    monkeypatch.setattr(cli.subprocess,'run',lambda *a,**k:copied.append(a))
    context=object.__new__(bc.Context);context.native=dict(assets_root=str(tmp_path/'assets'))
    context.stage_checkpoint(dict(path='/cpu-only-not-transferred',sha256='a'*64))
    assert len(scripts)==3 and len(copied)==1


def test_expired_observer_does_not_stage_or_submit_new_work(setup):
    with pytest.raises(TimeoutError,match='retained'):execute(setup,deadline=0)
    assert not any(isinstance(event,tuple) and event[0]=='stage' for event in setup['events'])
    assert 'paired-run' not in setup['events']
    assert not (setup['output']/'bc-complete.json').exists()


def test_cli_paired_route_does_not_call_legacy_executor(tmp_path,monkeypatch):
    selection=publish(tmp_path/'selection.json',{})
    runtime=publish(tmp_path/'runtime.json',{})
    output=tmp_path/'output';calls=[]
    monkeypatch.setattr(cli,'execute',lambda *a,**k:pytest.fail('legacy B2 path used'))
    def admitted(*args,**kwargs):
        calls.append((args,kwargs));return dict(dispatch_output=str(output),status='CPU-only')
    monkeypatch.setattr(bc,'execute',admitted)
    monkeypatch.setattr(sys,'argv',['bc-success','--selection',selection['path'],
        '--paired-runtime-bindings',runtime['path'],'--output',str(output),'--execute'])
    cli.main()
    assert len(calls)==1 and calls[0][0]==(selection,runtime,output)
    assert calls[0][1]['deadline']>0


@pytest.mark.parametrize('extra',[['--pool-local','/legacy'],['--remote-pool','/legacy'],['--remote-source-root','/legacy']])
def test_cli_rejects_mixed_legacy_overrides_before_execution(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,'argv',['bc-success','--selection','/none',
        '--paired-runtime-bindings','/none','--output',str(tmp_path/'out'),*extra])
    with pytest.raises(SystemExit):cli.main()
    assert not (tmp_path/'out').exists()


def test_cli_old_route_remains_explicit(tmp_path,monkeypatch):
    selection=publish(tmp_path/'selection.json',{})
    pool=publish(tmp_path/'pool.json',{})
    output=tmp_path/'out';calls=[]
    monkeypatch.setattr(cli,'execute',lambda *a,**k:calls.append((a,k)) or dict(dispatch_output=str(output)))
    monkeypatch.setattr(bc,'execute',lambda *a,**k:pytest.fail('paired route used for legacy'))
    monkeypatch.setattr(sys,'argv',['bc-success','--selection',selection['path'],'--pool-local',pool['path'],
        '--remote-pool','/remote-pool','--remote-source-root','/frozen','--output',str(output),'--execute'])
    cli.main()
    assert calls[0][0]==(selection,pool,Path('/remote-pool'),Path('/frozen'),output)
