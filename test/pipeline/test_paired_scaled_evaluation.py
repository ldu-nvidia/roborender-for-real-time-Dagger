"""CPU-only synthetic orchestration/receipt tests; never live qualification."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from pipeline.deployment import paired_scaled_evaluation as paired
from pipeline.deployment import paired_scaled_runtime as runtime
from pipeline.deployment import scaled_success_dispatch as dispatch
from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
from test.pipeline.test_scaled_success_qualification import evidence, accepted
from test.pipeline.test_scaled_success_resources import fixture as resource_fixture


def publish(path, value):
    dispatch.immutable(path, value)
    return paired.ref(path)


def owners():
    return [dict(worker_instance_id=str(i+1)*32,source_sha256='a'*64,native_pid=1,
        physical_gpu=i,gpu_uuid='GPU-synthetic-'+str(i)) for i in range(2)]


def plan(evidence):
    return paired.make_run(evidence['panel'],evidence['config'],owners(),command_prefix='cpu-paired-test',
        source_sha256='a'*64,arm='bc',cpu_validation={'path':'/cpu-only.json','raw_sha256':'a'*64})


def ready(owner):
    return dict(schema='roborender.isaac-resident.v1',role='isaac_resident',ok=True,
        **{k:owner[k] for k in dispatch.OWNER_KEYS},native_initializations=1,
        phase='IDLE',active_command_id=None)


def test_real_plan_ladder_routes_both_actors_four_dedicated_renderers(evidence):
    value=plan(evidence)
    assert [t['label'] for t in value['trials']]==['b4','b8','b16','b16-repeat','tail7']
    assert [t['plan']['panel']['episode_count'] for t in value['trials']]==[8,16,32,32,7]
    for trial in value['trials']:
        row=trial['plan']
        assert row['qualification_evidence'] is None
        assert row['arm']=='bc' and row['checkpoint_sha256']==evidence['config']['checkpoint_sha256']
        assert [e['owner_index'] for e in row['commands']]==([0] if trial['label']=='tail7' else [0,1])
        assert row['config']['renderer']==evidence['config']['renderer']
        assert row['config']['anchors']==evidence['config']['anchors']
    assert value['trials'][2]['plan']['panel']==value['trials'][3]['plan']['panel']
    assert value['trials'][2]['plan']['command_prefix']!=value['trials'][3]['plan']['command_prefix']


@pytest.mark.parametrize('fault',['single','source','panel','batch','duplicate','arm'])
def test_no_partial_or_historical_fallback(evidence,fault):
    panel=deepcopy(evidence['panel']);config=deepcopy(evidence['config']);native=owners();arm='bc'
    if fault=='single':native=native[:1]
    elif fault=='source':native[1]['source_sha256']='b'*64
    elif fault=='panel':panel=panel['scaled_protocol']['source_panel']
    elif fault=='batch':panel=scaled.build_panel(panel['scaled_protocol']['source_panel'],batch_size=8)
    elif fault=='duplicate':native[1]['gpu_uuid']=native[0]['gpu_uuid']
    else:arm='teacher'
    with pytest.raises((ValueError,KeyError)):
        paired.make_run(panel,config,native,command_prefix='cpu',source_sha256='a'*64,arm=arm,cpu_validation={})


def test_both_idle_barriers_reject_changed_pid_running_or_active():
    native=owners();rows=[ready(o) for o in native]
    assert paired.validate_barrier(rows,native)==rows
    for key,value in [('native_pid',2),('phase','RUNNING'),('active_command_id','other')]:
        bad=deepcopy(rows);bad[1][key]=value
        with pytest.raises(ValueError):paired.validate_barrier(bad,native)


def actual_cpu_receipt(tmp_path,source='a'*64):
    xml=tmp_path/'junit.xml';xml.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    return publish(tmp_path/'cpu.json',dict(schema='roborender.live-pipeline-smoke-cpu-validation.v1',
        source_sha256=source,passed=True,tests=3,failures=0,errors=0,skipped=0,
        junit_xml=dict(path=str(xml),raw_sha256=hashlib.sha256(xml.read_bytes()).hexdigest())))


def test_cpu_gate_reads_actual_junit_counts_and_hash(tmp_path):
    reference=actual_cpu_receipt(tmp_path)
    assert runtime.cpu_gate(reference,'a'*64)['tests']==3
    with pytest.raises(ValueError):runtime.cpu_gate(reference,'b'*64)
    (tmp_path/'junit.xml').write_text('changed')
    with pytest.raises(ValueError):runtime.cpu_gate(reference,'a'*64)


def test_guard_copy_reproduces_actual_sustained_guard_and_rejects_relabel(tmp_path):
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import PhaseResourceGuard,SUSTAINED_VERSION
    samples,_=resource_fixture();sample=samples[0]['native'][0]['safety_sample']
    with PhaseResourceGuard(tmp_path/'guard.json',sample=lambda:deepcopy(sample),policy_version=SUSTAINED_VERSION) as guard:
        pass
    raw=(tmp_path/'guard.json').read_bytes();owner=owners()[0]
    record=dict(owner=owner,summary=guard.summary,verified=True,guard_raw_base64=base64.b64encode(raw).decode())
    receipt=dict(resource_guard=guard.summary)
    assert paired.validate_guard_copy(owner,receipt,record)['passed']
    bad=deepcopy(record);bad['owner']['worker_instance_id']='f'*32
    with pytest.raises(ValueError):paired.validate_guard_copy(owner,receipt,bad)
    bad=deepcopy(record);bad['guard_raw_base64']=base64.b64encode(b'{}').decode()
    with pytest.raises(ValueError):paired.validate_guard_copy(owner,receipt,bad)


@pytest.mark.parametrize('failure',['timeout','ssh','remote'])
def test_transport_single_attempt_and_unknown_latch_stops_peer(tmp_path,failure):
    calls=[]
    def invoke(argv,**kwargs):
        calls.append(argv)
        if failure=='timeout':raise subprocess.TimeoutExpired(argv,35)
        return SimpleNamespace(returncode=255 if failure=='ssh' else 1,stdout=b'',stderr=b'private')
    site=SimpleNamespace(ssh=lambda _:['ssh','exact-owner'])
    sampler=SimpleNamespace(healthy=lambda:None)
    native=dict(source_root='/frozen',source_sha256='a'*64,jobs_root='/jobs')
    transport=runtime.Transport(site,native,tmp_path,sampler,run=invoke)
    with pytest.raises((TimeoutError,ConnectionError,RuntimeError)):
        transport.submit(owners()[0],{'body':'test'})
    assert len(calls)==1
    if failure!='remote':
        assert transport.unknown
        with pytest.raises(ConnectionError):transport.submit(owners()[1],{'body':'never'})
        assert len(calls)==1


@pytest.mark.parametrize('mode',[['-O'],['-OO']])
def test_remote_rejects_optimized_python_before_parsing_importing_or_posting(mode):
    # No valid arguments, source path or network endpoint are provided: the
    # optimization rejection must run before any project import or RPC.
    result=subprocess.run([sys.executable,*mode,'-B','-c',runtime.REMOTE],
        capture_output=True,text=True,timeout=10)
    assert result.returncode!=0
    assert 'optimized Python is forbidden' in result.stderr
    assert 'IndexError' not in result.stderr and 'ModuleNotFoundError' not in result.stderr


def test_final_validator_never_imports_raw_aggregate_or_qualification(evidence):
    for value in ({'completed_episodes':295,'successes':295},evidence['proof']):
        with pytest.raises(ValueError):paired.validate_final_evidence(value,panel=evidence['panel'],
            checkpoint_sha256=evidence['config']['checkpoint_sha256'],source_sha256='a'*64,arm='bc')


def test_automatic_ladder_precedes_test_and_same_saved_plan_is_used(evidence,tmp_path,monkeypatch):
    value=plan(evidence);value['cpu_validation']=actual_cpu_receipt(tmp_path)
    run=paired.Run(value,tmp_path/'run',SimpleNamespace(),SimpleNamespace())
    labels=[];summaries=iter(evidence['proof']['trials']+[evidence['proof']['tail_trial']])
    def dispatch_one(label,actual):
        labels.append(label)
        if label=='final':
            assert labels[:-1]==[x[0] for x in paired.TRIALS]
            assert actual['qualification_evidence']['checkpoint_sha256']==value['config']['checkpoint_sha256']
            assert actual['panel']['episode_count']==295
            return {'actual-final':True}
        publish(run.root/label/'dispatch-evidence.json',{'label':label})
        return next(summaries)
    monkeypatch.setattr(run,'dispatch_one',dispatch_one)
    monkeypatch.setattr(paired,'trial_summary',lambda x:x)
    monkeypatch.setattr(paired,'validate_final_evidence',lambda *a,**kw:{})
    # Actual native qualification validator still checks the synthetic complete
    # ladder fixture; there is no boolean shortcut inside Run.execute.
    result=run.execute()
    assert result['qualification_only'] is False and labels[-1]=='final'
    assert paired.bound(result['qualification'])['trials']==evidence['proof']['trials']


def test_incomplete_ladder_cannot_dispatch_test(evidence,tmp_path,monkeypatch):
    value=plan(evidence);value['cpu_validation']=actual_cpu_receipt(tmp_path)
    run=paired.Run(value,tmp_path/'run',SimpleNamespace(),SimpleNamespace())
    labels=[]
    def dispatch_one(label,actual):
        labels.append(label)
        raise RuntimeError('actual fault')
    monkeypatch.setattr(run,'dispatch_one',dispatch_one)
    with pytest.raises(RuntimeError,match='actual fault'):run.execute()
    assert labels==['b4'] and not (run.root/'qualification.json').exists()


def test_unknown_reentry_preserves_interval_and_reconciles_same_plan(evidence,tmp_path,monkeypatch):
    value=plan(evidence);seen=[];ready_calls=[]
    transport=SimpleNamespace(unknown=False,resource_validator=lambda *_:None,
        get_ready=lambda o:ready_calls.append(o) or ready(o))
    run=paired.Run(value,tmp_path/'run',transport,SimpleNamespace(healthy=lambda:None),clock=lambda:101.)
    class Coordinator:
        def __init__(self,p,root,t,**kwargs):
            seen.append((deepcopy(p),root,kwargs))
            assert kwargs['max_submit_attempts']==1
            assert kwargs['stop_peer_on_uncertainty'] is True
        def tick(self):
            return dict(complete=False,workers=[{'status':'TRANSPORT_UNKNOWN_RECONCILE_SAME_OWNER'}])
    monkeypatch.setattr(dispatch,'Coordinator',Coordinator)
    trial=value['trials'][0]
    with pytest.raises(RuntimeError,match='UNKNOWN'):run.dispatch_one(trial['label'],trial['plan'])
    started=(run.root/'b4/interval-start.json').read_bytes()
    assert len(ready_calls)==2
    # An accepted native operation can still be running. Re-entry must ask the
    # same coordinator to reconcile; it must not impose a fresh-start barrier.
    transport.get_ready=lambda _:pytest.fail('must not reset/re-enter an in-flight trial')
    with pytest.raises(RuntimeError,match='UNKNOWN'):run.dispatch_one(trial['label'],trial['plan'])
    assert len(seen)==2 and seen[0]==seen[1]
    assert (run.root/'b4/interval-start.json').read_bytes()==started


def test_expired_observer_never_ticks_or_submits(evidence,tmp_path,monkeypatch):
    value=plan(evidence)
    transport=SimpleNamespace(resource_validator=lambda *a:None,unknown=False)
    run=paired.Run(value,tmp_path/'run',transport,SimpleNamespace(healthy=lambda:None),
        clock=lambda:100.,deadline=99.)
    monkeypatch.setattr(paired,'idle_barrier',lambda *a:[ready(o) for o in owners()])
    monkeypatch.setattr(dispatch.Coordinator,'tick',lambda *a:pytest.fail('expired observer submitted new work'))
    with pytest.raises(TimeoutError,match='retained'):
        run.dispatch_one('b4',value['trials'][0]['plan'])


def test_provider_never_uses_legacy_gpu0_route_for_scaled_panel(evidence,monkeypatch):
    from pipeline.deployment.retained_campaign_site import RetainedResidentSite
    from pipeline.dagger import beta_validation
    monkeypatch.setattr(beta_validation,'load_panel',lambda _:evidence['panel'])
    site=object.__new__(RetainedResidentSite);site.config={};site.paired_evaluation=None
    with pytest.raises(ValueError,match='no GPU0/B2 fallback'):site.evaluate_explicit()
    site.paired_evaluation={'explicit':'v4'}
    monkeypatch.setattr(runtime,'evaluate',lambda s,b:dict(paired=True,binding=b))
    assert site.evaluate_explicit()==dict(paired=True,binding=site.paired_evaluation)


def test_bc_v2_cannot_fall_back_to_legacy_b2():
    from pipeline.deployment.bc_success_dispatch import command_plan
    with pytest.raises(ValueError,match='legacy GPU0/B2 dispatch is forbidden'):
        command_plan({'schema':'roborender.bc-production-selected.v2'},None)


def test_scaled_mse_import_and_protocol_are_not_historical(evidence,monkeypatch):
    from pipeline.dagger import beta_validation,mse_campaign
    from pipeline.deployment import rendered_success_dispatch
    panel=evidence['panel'];config=evidence['config'];seen=[]
    monkeypatch.setattr(beta_validation,'load_panel',lambda _:panel)
    monkeypatch.setattr(rendered_success_dispatch,'configuration',lambda *a:config)
    monkeypatch.setattr(paired,'validate_final_evidence',lambda *a,**kw:seen.append(kw) or {'actual':True})
    cfg=dict(schema='roborender.cross_gpu_dagger_continuous_fixed100.leo.v2',source_sha256='a'*64,
             validation={'rendered_observation_config':{}})
    assert mse_campaign.validate_explicit_final_test(cfg,{},config['checkpoint_sha256'])=={'actual':True}
    assert seen[0]['expected_rendered_config']==config and seen[0]['arm']=='dagger'
    assert mse_campaign.production_test_protocol(cfg)==scaled.protocol_identity(panel)


def test_operations_v4_requires_eight_retained_and_explicit_paired_binding(tmp_path,monkeypatch):
    from pipeline.dagger import coverage_queue
    manifest=publish(tmp_path/'manifest.json',dict(source_sha256='a'*64))
    site=publish(tmp_path/'site.json',dict(schema='roborender.leo-site.v1',cpu_python='/usr/bin/python3',source_manifest=manifest['path']))
    retained=publish(tmp_path/'retained.json',dict(schema='roborender.full295-retained-h200.v3'))
    binding=publish(tmp_path/'paired.json',dict(schema=paired.RUNTIME_SCHEMA))
    monkeypatch.setattr(runtime,'validate_runtime',lambda *a,**kw:None)
    value=dict(schema='roborender.full295-runtime-bindings.v4',site=site,retained=retained,
        resident_pool=dict(path='/exact/pool.json',sha256='b'*64),source_sha256='a'*64,paired_evaluation=binding)
    result=coverage_queue.operations_for(value,tmp_path)
    for row in result['commands'].values():
        assert row['argv'][row['argv'].index('--paired-evaluation')+1]==binding['path']
    bad=deepcopy(value);bad.pop('paired_evaluation')
    with pytest.raises(ValueError):coverage_queue.operations_for(bad,tmp_path)
    bad=deepcopy(value);bad['schema']='roborender.full295-runtime-bindings.v3'
    with pytest.raises(ValueError):coverage_queue.operations_for(bad,tmp_path)


def committed_trial(evidence,tmp_path):
    """Real validators, synthetic isolated native outcomes and sampled counters."""
    from pipeline.dagger.rtx.code.src.dagger_rtx.phase_resource_guard import PhaseResourceGuard,SUSTAINED_VERSION
    trial=deepcopy(evidence['proof']['trials'][0]);native=owners()
    p=dispatch.build_plan(trial['panel'],trial['config'],native,command_prefix='cpu-evidence',arm='bc')
    plan_ref=publish(tmp_path/'plan.json',p);commits=[];guards=[];values=[]
    samples,_=resource_fixture();sample=samples[0]['native'][0]['safety_sample']
    for i,(entry,owner) in enumerate(zip(p['commands'],native)):
        with PhaseResourceGuard(tmp_path/f'actual-guard-{i}.json',sample=lambda:deepcopy(sample),policy_version=SUSTAINED_VERSION) as guard:
            pass
        raw=(tmp_path/f'actual-guard-{i}.json').read_bytes()
        record=dict(owner=owner,summary=guard.summary,verified=True,guard_raw_base64=base64.b64encode(raw).decode())
        guards.append(publish(tmp_path/f'guard-copy-{i}.json',record))
        receipt=deepcopy(trial['owners'][i]['native_receipt'])
        receipt.update(native_pid=1,purpose='test',profile='explicit_evaluation',resource_guard=guard.summary)
        command=dict(status='COMPLETE',payload_sha256=entry['payload_sha256'],receipt=receipt)
        checked=dispatch.validate_complete(p,entry,command,lambda o,r:paired.validate_guard_copy(o,r,record))
        commit=dict(schema=dispatch.COMMIT_SCHEMA,plan_sha256=dispatch.digest(p),cohort_index=i,owner=owner,
            command_id=entry['body']['command_id'],payload_sha256=entry['payload_sha256'],
            native_command_receipt=command,result=checked)
        commits.append(publish(tmp_path/f'commit-{i}.json',commit));values.append(checked)
    resources=trial['resources']
    for row in resources['samples']:
        for actual,owner in zip(row['native'],paired.descriptors(native)):
            actual.update(owner)
    resources.update(started_unix=trial['started_unix'],ended_unix=trial['ended_unix'])
    resources['summary']=paired.summarize(resources['samples'],native_owners=paired.descriptors(native),
        renderer_owners=resources['renderer_owners'],started_unix=resources['started_unix'],ended_unix=resources['ended_unix'])
    result=scaled.aggregate_results(p['panel'],values,p['checkpoint_sha256'],p['config'],arm='bc')
    return dict(dispatch_plan=plan_ref,cohort_commits=commits,resource_guards=guards,
        resources=publish(tmp_path/'resources.json',resources),result=publish(tmp_path/'result.json',result),
        barrier_owners=native,before_idle=[ready(o) for o in native],after_idle=[ready(o) for o in native])


def test_committed_native_outcomes_and_actual_guards_are_revalidated(evidence,tmp_path):
    value=committed_trial(evidence,tmp_path)
    p,result=paired.validate_dispatch_evidence(value)
    assert result['completed_episodes']==8 and result['qualification_only'] is True
    assert [x['physical_gpu'] for x in p['owners']]==[0,1]
    bad=deepcopy(value);bad['cohort_commits']=bad['cohort_commits'][:1]
    with pytest.raises(ValueError):paired.validate_dispatch_evidence(bad)
    bad=deepcopy(value);bad['resource_guards']=[]
    with pytest.raises(KeyError):paired.validate_dispatch_evidence(bad)
    bad=deepcopy(value);bad['after_idle'][1]['native_pid']=2
    with pytest.raises(ValueError):paired.validate_dispatch_evidence(bad)
    path=Path(value['cohort_commits'][0]['path']);path.chmod(0o644);path.write_text('{}')
    with pytest.raises(ValueError):paired.validate_dispatch_evidence(value)


def test_sampler_is_identity_bound_and_is_not_a_new_process(tmp_path,monkeypatch):
    import os
    source=tmp_path/'sampler.py';source.write_text('# synthetic CPU-only sampler source\n')
    source_ref=dict(path=str(source),raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    journal=tmp_path/'samples.jsonl';journal.write_text('{}\n')
    status=tmp_path/'status.json';status.write_text(json.dumps(dict(state='RUNNING',pid=os.getpid(),latest_at_unix=100.)))
    runtime_value=dict(source_sha256='a'*64,native={'path':'/native.json','raw_sha256':'b'*64},
        renderer_fleet={'path':'/fleet.json','raw_sha256':'c'*64})
    spec=dict(schema='roborender.paired-scaled-resource-journal.v1',**runtime_value,
        interval_seconds=5,compute_release_authorized=False,artificial_gpu_activity=False,
        journal=str(journal),status_file=str(status),sampler_source=source_ref)
    argv=['/synthetic/cpu/python',str(source),'--run']
    identity=dict(start_ticks='123',cmdline_sha256=hashlib.sha256(b''.join(os.fsencode(x)+b'\0' for x in argv)).hexdigest())
    monkeypatch.setattr(runtime,'process_identity',lambda pid:identity if pid==os.getpid() else None)
    process=publish(tmp_path/'process.json',dict(pid=os.getpid(),identity=identity,argv=argv,sampler_source=source_ref,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),sampler_spec_sha256=dispatch.digest(spec)))
    spec['process_receipt']=process;reference=publish(tmp_path/'sampler-spec.json',spec)
    observer=runtime.ResourceJournal(reference,runtime_value,clock=lambda:105.)
    assert observer.healthy()['pid']==os.getpid()
    status.write_text(json.dumps(dict(state='RUNNING',pid=os.getpid(),latest_at_unix=90.)))
    with pytest.raises(RuntimeError,match='stale'):observer.healthy()
    status.write_text(json.dumps(dict(state='RUNNING',pid=os.getpid()+1,latest_at_unix=100.)))
    with pytest.raises(RuntimeError):observer.healthy()


def test_large_anchor_artifact_uses_stdin_not_oversized_ssh_argument(tmp_path,evidence):
    calls=[]
    def invoke(argv,**kwargs):
        calls.append((argv,kwargs))
        if kwargs['input'] is None:
            renderer=evidence['config']['renderer']
            return SimpleNamespace(returncode=0,stdout=json.dumps(dict(
                schema='roborender.renderer-route-preflight.v1',renderer_sha256=hashlib.sha256(json.dumps(renderer,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                verified_slots=list(range(8)),metadata_only=True,native_steps=0,render_requests=0)).encode())
        raw=kwargs['input']
        return SimpleNamespace(returncode=0,stdout=json.dumps(dict(staged=True,raw_sha256=hashlib.sha256(raw).hexdigest())).encode())
    site=SimpleNamespace(ssh=lambda _:['ssh','exact-owner']);sampler=SimpleNamespace(healthy=lambda:None)
    transport=runtime.Transport(site,dict(source_root='/frozen',source_sha256='a'*64,jobs_root='/jobs'),tmp_path,sampler,run=invoke)
    value={**evidence['config'],'large-anchor-list':'x'*200000};raw=dispatch.canonical(value)
    artifact=dict(path='test/config.json',value=value,raw_sha256=hashlib.sha256(raw).hexdigest())
    transport.stage_artifacts(owners()[0],[artifact])
    assert len(calls)==2 and calls[1][1]['input']==raw
    assert len(calls[1][0][-1])<10000


def test_bounded_aggregate_limit_never_truncates_or_changes_legacy_policy(tmp_path,monkeypatch,evidence):
    data={'long':'x'*300}
    with pytest.raises(ValueError):dispatch.immutable(tmp_path/'small.json',data,limit=100)
    assert not (tmp_path/'small.json').exists()
    dispatch.immutable(tmp_path/'large.json',data,limit=400)
    assert dispatch.read(tmp_path/'large.json',limit=400)==data
    with pytest.raises(ValueError):dispatch.read(tmp_path/'large.json',limit=100)
    for limit in (0,True,dispatch.MAX_FINAL_RESULT_LIMIT+1):
        with pytest.raises(ValueError):dispatch.immutable(tmp_path/'bad.json',{},limit=limit)
    p=plan(evidence)['trials'][0]['plan']
    dispatch.Coordinator(p,tmp_path/'legacy',None,resource_validator=lambda *_:None)
    assert 'final_aggregate_limit_bytes' not in dispatch.read(tmp_path/'legacy/controller-policy.json')
    dispatch.Coordinator(p,tmp_path/'v4',None,resource_validator=lambda *_:None,result_limit=dispatch.MAX_FINAL_RESULT_LIMIT)
    assert dispatch.read(tmp_path/'v4/controller-policy.json')['final_aggregate_limit_bytes']==512<<20
    with pytest.raises(ValueError):
        dispatch.Coordinator(p,tmp_path/'v4',None,resource_validator=lambda *_:None)


def test_runtime_preflight_rejects_missing_sampler_before_training(tmp_path,monkeypatch):
    from pipeline.deployment.h200 import retained_renderer_attachment
    native=dict(schema='roborender.scaled-evaluation-native-bindings.v1',source_sha256='a'*64,
        workers=[dict(o,host_pid=100+i,container_id=str(i+1)*64) for i,o in enumerate(owners())])
    value=dict(schema=paired.RUNTIME_SCHEMA,source_sha256='a'*64,allocation_release_authorized=False,
        native=publish(tmp_path/'native.json',native),renderer_fleet=publish(tmp_path/'fleet.json',{}),
        cpu_validation={'path':'/cpu.json','raw_sha256':'c'*64},
        resource_sampler={'path':'/absent/sampler.json','raw_sha256':'d'*64})
    monkeypatch.setattr(retained_renderer_attachment,'validate_fleet',lambda x:x)
    monkeypatch.setattr(runtime,'cpu_gate',lambda *a:None)
    with pytest.raises(FileNotFoundError):runtime.validate_runtime(value,source_sha256='a'*64)
