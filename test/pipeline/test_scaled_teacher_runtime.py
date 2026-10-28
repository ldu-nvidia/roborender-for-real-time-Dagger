"""Synthetic CPU orchestration checks; no live/native qualification claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pipeline.deployment import scaled_teacher_runtime as runtime
from test.pipeline.test_scaled_teacher_dispatch import plan, accepted, Transport, guard, posts
from test.pipeline.test_scaled_success_resources import fixture as sample_fixture


def save(path, value):
    runtime.common.immutable(path, value)
    return runtime.paired.ref(path)


@pytest.fixture
def setup(plan, tmp_path, monkeypatch):
    # Source/scientific contracts have separate actual tests. These synthetic
    # fixtures exercise the real admission fields, CPU gate and coordinator.
    monkeypatch.setattr(runtime, 'verify_release', lambda root, manifest, sha: None)
    monkeypatch.setattr(runtime, 'source_jail', lambda root: None)
    monkeypatch.setattr(runtime, 'validate_fleet', lambda value: value)
    root = tmp_path/'source'; root.mkdir()
    owners = [dict(o, host_pid=100+i, container_id=str(i+3)*64, container_name=f'cpu-{i}',
        port=18640+i, token_file=f'/private/{i}.token', runtime_root=f'/runtime/{i}')
        for i,o in enumerate(plan['owners'])]
    plan = runtime.dispatch.build_plan(plan['panel'], owners, command_prefix=plan['command_prefix'],
        shadow_checkpoint_sha256=plan['shadow_not_behavior_checkpoint_sha256'])
    native = dict(schema='roborender.scaled-evaluation-native-bindings.v1', workers=owners,
        source_sha256='a'*64, local_source_root=str(root), source_root='/remote/frozen', jobs_root='/remote/jobs')
    fleet = dict(workers=[dict(server_instance=f'{i+10:032x}',gpu_uuid=f'GPU-cpu-renderer-{i}',
        source_sha256='c'*64) for i in range(8)])
    xml = tmp_path/'junit.xml'; xml.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    cpu = dict(schema='roborender.live-pipeline-smoke-cpu-validation.v1', source_sha256='a'*64,
        passed=True, tests=1, failures=0, errors=0, skipped=0, junit_xml=runtime.paired.ref(xml))
    auth = dict(schema=runtime.AUTH_SCHEMA, execute_scaled_teacher_full295=True,
        user_instruction_verbatim='Run the full teacher baseline.', **{k:False for k in runtime.FLAGS})
    refs = {key:save(tmp_path/(key+'.json'), value) for key,value in dict(dispatch_plan=plan,
        native=native, renderer_fleet=fleet, source_manifest={'source_sha256':'a'*64},
        cpu_validation=cpu, user_authorization=auth).items()}
    refs['sampler_binding']=save(tmp_path/'sampler_binding.json',dict(synthetic_cpu_only=True,
        source_sha256='a'*64,native=refs['native'],renderer_fleet=refs['renderer_fleet'],
        compute_release_authorized=False,artificial_gpu_activity=False))
    value = dict(schema=runtime.SCHEMA, source_sha256='a'*64, source_root=str(root),
        output=str(tmp_path/'output'), wandb=dict(entity='nvidia',project='dagger',id='cpu-teacher-runtime'),
        **refs, **{k:False for k in runtime.FLAGS})
    transport = Transport(plan, tmp_path/'output'/'dispatch')
    transport.resource_validator = guard
    monkeypatch.setattr(runtime.dispatch.teacher,'validate_result',lambda panel,result:deepcopy(result))
    monkeypatch.setattr(runtime.dispatch.teacher,'aggregate_results',lambda panel,values:
        dict(synthetic_cpu_only=True,completed=295,successes=0))
    ticks = iter(range(100,10000))
    def sleep(_):
        for entry in plan['commands']:
            if transport.rows.get(entry['body']['command_id'],{}).get('status')=='RUNNING':
                transport.finish(entry['cohort_index'])
    sampler = Sampler(value['sampler_binding'],fleet)
    reporter = Reporter()
    capability = lambda owner:dict(schema=runtime.CAPABILITY_SCHEMA, owner=owner,
        environment=runtime.required_environment(owner), container_id=owner['container_id'], host_pid=owner['host_pid'])
    return SimpleNamespace(value=value,plan=plan,native=native,fleet=fleet,transport=transport,
        sampler=sampler,reporter=reporter,capability=capability,clock=lambda:next(ticks),sleep=sleep)


class Sampler:
    def __init__(self,binding,fleet):
        self.binding,self.fleet=binding,fleet
        self.bad=False

    def healthy(self):
        if self.bad:raise RuntimeError('synthetic sampler stale')
        return {'state':'RUNNING'}

    def interval(self,started,ended,owners,config):
        sample=sample_fixture()[0][0]
        native=runtime.paired.descriptors(owners)
        renderers=runtime.paired.renderer_descriptors(config)
        rows=[]
        times=list(range(int(started)-1,int(ended)+2,5))
        if times[-1]<ended:times.append(ended+1)
        for at in times:
            row=deepcopy(sample);row['at_unix']=at
            row['native']=[dict(deepcopy(sample['native'][0]),**o) for o in native]
            row['renderers']=[dict(deepcopy(sample['renderers'][0]),**o) for o in renderers]
            rows.append(row)
        return dict(started_unix=started,ended_unix=ended,samples=rows,renderer_owners=renderers,
            summary=runtime.paired.summarize(rows,native_owners=native,renderer_owners=renderers,
                started_unix=started,ended_unix=ended))


class Reporter:
    def start(self,plan,root):
        self.plan=plan;self.rows=[]
        return dict(run_id=plan['wandb']['id'],url='https://wandb.ai/synthetic-cpu',authenticated=True)
    def progress(self,progress):self.rows.append(deepcopy(progress))
    def complete(self,values):
        return dict(online_readback_verified=True,run_id=self.plan['wandb']['id'],
                    url='https://wandb.ai/synthetic-cpu',metrics=values)


def execute(s):
    return runtime.execute(s.value,transport=s.transport,sampler=s.sampler,reporter=s.reporter,
        capability_reader=s.capability,clock=s.clock,sleep=s.sleep)


def test_actual_coordinator_all295_failures_retained_and_online_after_interval(setup):
    result=execute(setup)
    assert result['complete'] and result['completed']==295 and result['successes']==0
    assert result['success_rate']==0 and result['committed_cohorts']==19
    assert result['visual_student_certified'] is False
    assert len(posts(setup.transport))==19
    resources=runtime.paired.bound(result['whole_interval_resources'],aggregate=True)
    assert len(resources['summary']['native_owners'])==2
    assert len(resources['summary']['renderer_owners'])==8
    assert runtime.paired.bound(result['wandb_readback'])['metrics']['teacher_baseline/full295_complete']
    assert execute(setup)['status']=='UNKNOWN_REQUIRES_READ_ONLY_RECONCILIATION'
    assert len(posts(setup.transport))==19


@pytest.mark.parametrize('did_accept',[True,False])
def test_lost_reply_latches_outer_unknown_and_never_reposts(setup,did_accept):
    setup.transport.lost=True;setup.transport.accept=did_accept
    with pytest.raises(RuntimeError,match='reconciliation'):execute(setup)
    assert len(posts(setup.transport))==1
    setup.transport.lost=False
    value=execute(setup)
    assert value['status']=='UNKNOWN_REQUIRES_READ_ONLY_RECONCILIATION'
    assert len(posts(setup.transport))==1
    assert not (Path(setup.value['output'])/'result.json').exists()


@pytest.mark.parametrize('kind',['capability','busy','preexisting','sampler','logging'])
def test_admission_failures_never_submit(setup,kind):
    if kind=='capability':
        original=setup.capability
        setup.capability=lambda owner:dict(original(owner),environment={})
    elif kind=='busy':setup.transport.active[setup.plan['owners'][1]['worker_instance_id']]='unrelated'
    elif kind=='preexisting':setup.transport.rows[setup.plan['commands'][18]['body']['command_id']]={}
    elif kind=='sampler':setup.sampler.bad=True
    else:setup.reporter.start=lambda p,r:dict(run_id='wrong',authenticated=True)
    with pytest.raises((ValueError,RuntimeError)):execute(setup)
    assert not posts(setup.transport)


@pytest.mark.parametrize('kind',['resources','online','final_idle'])
def test_full_native_count_without_terminal_proofs_is_not_complete(setup,kind):
    if kind=='resources':
        original=setup.sampler.interval
        setup.sampler.interval=lambda *args:dict(original(*args),summary={})
    elif kind=='online':setup.reporter.complete=lambda m:dict(online_readback_verified=False)
    else:
        original=setup.reporter.progress
        def progress(p):
            original(p)
            if p['complete']:setup.transport.active[setup.plan['owners'][0]['worker_instance_id']]='unrelated'
        setup.reporter.progress=progress
    with pytest.raises(ValueError):execute(setup)
    assert len(posts(setup.transport))==19
    assert not (Path(setup.value['output'])/'result.json').exists()


@pytest.mark.parametrize('field,value',[('compute_release_authorized',True),('native_replay_allowed',True),
    ('automatic_restart',True),('source_sha256','d'*64)])
def test_runtime_source_and_no_replay_retention_admission(setup,field,value):
    setup.value[field]=value
    with pytest.raises(ValueError):execute(setup)
    assert not posts(setup.transport)


def test_changed_cpu_evidence_and_wrong_sampler_binding_fail_closed(setup):
    setup.sampler.binding={'path':'wrong','raw_sha256':'a'*64}
    with pytest.raises(ValueError,match='sampler'):execute(setup)
    setup.sampler.binding=setup.value['sampler_binding']
    Path(setup.value['cpu_validation']['path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):execute(setup)


def test_capability_read_is_bounded_metadata_only_and_checks_actual_optin(setup):
    owner=setup.plan['owners'][0];calls=[]
    site=SimpleNamespace(ssh=lambda host:['ssh','approved-leo'] if host=='beta' else None)
    def run(argv,**kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(returncode=0,stdout=json.dumps(setup.capability(owner)).encode())
    result=runtime.read_capability(site,setup.native,owner,run=run)
    assert result['environment']['RESIDENT_SCALED_TEACHER']==runtime.dispatch.teacher.SCHEMA
    assert calls[0][0][:2]==['ssh','approved-leo'] and calls[0][1]['timeout']==35
    assert 'docker exec' not in runtime.CAPABILITY_REMOTE
    assert '/v1/commands' not in runtime.CAPABILITY_REMOTE
    with pytest.raises(RuntimeError):
        runtime.read_capability(site,setup.native,owner,run=lambda *a,**k:SimpleNamespace(returncode=255,stdout=b''))


def test_teacher_metrics_never_invent_empty_rate_or_visual_score():
    values=runtime.metrics(dict(completed=0,successes=0,committed_cohorts=0,success_rate=None))
    assert 'teacher_baseline/success_rate' not in values
    assert all(k.startswith('teacher_baseline/') for k in values)
    assert not values['teacher_baseline/full295_complete']


def test_sampler_adapter_keeps_genuine_binding_and_brackets_before_read(tmp_path):
    binding=save(tmp_path/'sampler.json',{'synthetic_cpu_only':True});calls=[]
    sampler=SimpleNamespace(healthy=lambda:calls.append('healthy'),bracket=lambda t:calls.append(('bracket',t)))
    adapter=runtime.SamplerAdapter(sampler,binding,lambda *args:calls.append('read') or {'actual':'journal'})
    assert adapter.interval(1,2,[],{})=={'actual':'journal'}
    assert calls==['healthy',('bracket',2),'read']


@pytest.mark.parametrize('mismatch',[False,True])
def test_actual_online_reporter_orders_finish_before_exact_remote_readback(tmp_path,monkeypatch,mismatch):
    from pipeline.deployment import wandb_preflight
    events=[];summary={};run_id='cpu-online-teacher'
    run=SimpleNamespace(id=run_id,url='https://wandb.ai/synthetic-cpu',settings=SimpleNamespace(mode='online'))
    def log(values):events.append('log');summary.update(values)
    def finish():events.append('finish')
    run.log,run.finish=log,finish
    def remote(path):
        assert events[-1] in {'finish','readback'}
        assert path=='nvidia/dagger/'+run_id
        events.append('readback')
        return SimpleNamespace(summary=dict(summary,**{'teacher_baseline/successes':1}) if mismatch else summary)
    api=SimpleNamespace(flush=lambda:None,run=remote)
    def admit(mode,entity):
        assert (mode,entity)==('online','nvidia');events.append('auth')
        return api,{'authenticated':True}
    def init(**kwargs):
        assert events==['auth'] and kwargs['resume']=='never' and kwargs['mode']=='online'
        assert kwargs['job_type']=='scaled-privileged-teacher-baseline'
        assert kwargs['config']['visual_student_certified'] is False
        events.append('init');return run
    monkeypatch.setattr(wandb_preflight,'admit_logging',admit)
    verify=wandb_preflight.verify_remote_summary
    monkeypatch.setattr(wandb_preflight,'verify_remote_summary',lambda a,e,i,m:verify(a,e,i,m,sleep=lambda _:None))
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    reporter=runtime.OnlineReporter()
    reporter.start(dict(wandb=dict(entity='nvidia',project='dagger',id=run_id),
        source_sha256='a'*64,dispatch_plan={'synthetic_cpu_only':True}),tmp_path)
    values={'teacher_baseline/successes':0,'teacher_baseline/completed':295}
    if mismatch:
        with pytest.raises(RuntimeError,match='readback'):reporter.complete(values)
    else:
        result=reporter.complete(values)
        assert result['online_readback_verified'] and result['metrics']==values
    assert events[:4]==['auth','init','log','finish'] and events[4]=='readback'
