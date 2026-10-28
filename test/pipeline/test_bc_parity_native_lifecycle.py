"""CPU-only lifecycle admission/one-shot tests; no SSH, GPUs or allocations."""
from copy import deepcopy
import base64
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATH = REPO/'runtime/raven-stage-certification-20261025-kBFQ3q/native_lifecycle.py'
# These are explicitly historical, run-local lifecycle transport tests. They
# are not the collection parity/algorithm suite and cannot run in a fresh
# checkout lacking its ignored evidence and pinned helper chain.
RUNTIME_DEPS = [PATH,
    PATH.with_name('user-authorization-bc-smoke.json'),
    REPO/'runtime/teacher-scaled295-20261025/native_upgrade.py',
    REPO/'runtime/teacher-scaled295-20261025/native-bindings.json',
    REPO/'runtime/teacher-scaled295-20261025/paired-native-deployment.json',
    REPO/'runtime/curriculum-certification-20261025/native_upgrade.py',
    REPO/'runtime/live-pipeline-smoke-20261025/paired_native_deploy.py',
    REPO/'runtime/live-pipeline-smoke-20261025/native_upgrade_bc_smoke.py']
if any(not path.is_file() for path in RUNTIME_DEPS):
    pytest.skip('optional run-local BC lifecycle evidence is absent; core BC parity tests are separate',
        allow_module_level=True)
spec = importlib.util.spec_from_file_location('bc_parity_lifecycle_under_test', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def repack(row):
    raw = m.canonical(row['state'])
    row.update(state_raw_base64=base64.b64encode(raw).decode(), state_raw_sha256=m.sha(raw))
    return row


def make_audit(native):
    rows = []
    for g, owner in enumerate(native['workers']):
        cid = 'fixture-completed-g'+str(g)
        command = dict(schema='roborender.isaac-resident.v1', command_id=cid,
            worker_instance_id=owner['worker_instance_id'], operation='evaluate_scaled_teacher_panel', payload={})
        identity = {k: owner[k] for k in ('worker_instance_id', 'source_sha256', 'gpu_uuid', 'native_pid')}
        state = dict(identity, phase='IDLE', active_command_id=None, commands={cid:dict(command=command,
            payload_sha256=m.sha(m.canonical(command)), status='COMPLETE',receipt=dict(complete=True))})
        ready = dict(identity,ok=True,phase='IDLE',active_command_id=None,native_initializations=1,
            current_event=dict(phase='phase_end',command_id=cid))
        rows.append(repack(dict(owner=owner,ready=ready,state=state,
            actual={k:owner[k] for k in ('host_pid','container_id')},image_id=m.n.IMAGE,
            memory_max=21*2**30,memory_swap_max=0,memory_swap_current=0,thp_enabled=0,
            memory_events=dict(max=38 if g==0 else 9167,oom=0,oom_kill=0,oom_group_kill=0))))
    return dict(schema=m.n.SCHEMA+'.old-boundary', owners=rows,previous_controller_recertified=False)


def exited(p):
    return dict(owner=p['owner'],phase='STOPPED',active_command_id=None,last=p['last'],
        shutdown_status='COMPLETE',shutdown_receipt=dict(shutdown_requested=True),
        shutdown_payload_sha256=m.sha(m.canonical(m.n.n.shutdown_body(p))),
        container_running=False,old_host_pid_present=False)


class FakeTransport:
    def __init__(self, audit):
        self.audit_value=audit; self.calls=[]; self.staged=False; self.lost=None; self.busy=None
        self.exit_alive=False

    def audit(self, native): return deepcopy(self.audit_value)

    def remote(self,p,action):
        self.calls.append(action)
        if action in ('verify','stage'):
            if action=='stage':
                self.staged=True
                if self.lost=='stage':raise TimeoutError('lost stage reply')
            return dict(verified=True,source_sha256=m.SOURCE,image_id=m.n.IMAGE,
                inventory_sha256='i'*64,checkpoint_loaded_at_startup=False,
                retired_checkpoint_required=False,staged=self.staged)
        if action=='observe':
            v=deepcopy(self.audit_value['owners'][p['owner']['physical_gpu']])
            if self.busy==p['owner']['physical_gpu']:v['ready']['phase']='RUNNING'
            return v
        if action=='exit':return dict(exited(p),old_host_pid_present=self.exit_alive)
        if action=='shutdown':
            if self.lost=='shutdown':raise TimeoutError('lost shutdown reply')
            b=p['body']
            return dict(schema='roborender.isaac-resident.v1',status='accepted',command_id=b['command_id'],
                worker_instance_id=b['worker_instance_id'],payload_sha256=m.sha(m.canonical(b)))
        if action=='launch':
            if self.lost=='launch':raise TimeoutError('lost launch reply')
            return dict(state='STARTING')
        if action=='status':return dict(state='OBSERVED_ONLY')
        if action=='bind':
            owner=deepcopy(p['old_owners'][p['gpu']]['owner'])
            owner['worker_instance_id']='new-lifetime-'+str(p['gpu'])
            return dict(owner=owner)
        raise AssertionError(action)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    native=m.read(m.PREVIOUS/'native-bindings.json')
    audit=make_audit(native)
    prior=tmp_path/'prior';prior.mkdir()
    m.immutable(prior/'native-bindings.json',native)
    auth=m.read(m.ROOT/'user-authorization-bc-smoke.json')
    auth['existing_bindings']=str(prior/'native-bindings.json')
    root=tmp_path/'evidence';root.mkdir()
    m.immutable(root/'user-authorization-bc-smoke.json',auth)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'PREVIOUS',prior)
    t=FakeTransport(audit)
    m.prepare(root/'user-authorization-bc-smoke.json','standing-isaac-leo-prod-bcparity-fixture',transport=t)
    t.calls.clear()
    return root,t,native,audit


def test_historical_max_events_retained_not_capacity_certification(prepared):
    root,t,native,audit=prepared
    assert m.validate_audit(audit,native)==audit
    assert [r['memory_events']['max'] for r in m.read(root/'lifecycle-old-boundary.json')['owners']]==[38,9167]
    p=m.read(root/'lifecycle-plan.json');c=p['restore_config']
    assert c['checkpoint_sha256'] is None and c['checkpoint_loaded_at_startup'] is False
    assert not c['production_ready'] and 'bc_diagnostic_parent' not in c
    assert c['assets_root'].endswith('/standing-isaac-leo-prod-bcparity-fixture/assets')
    assert c['source_sha256']==m.SOURCE


@pytest.mark.parametrize('key,value',[('phase','RUNNING'),('active_command_id','other'),('native_initializations',2),('worker_instance_id','wrong')])
def test_reject_changed_or_busy_ready(prepared,key,value):
    _,_,native,audit=prepared;audit['owners'][1]['ready'][key]=value
    with pytest.raises(ValueError):m.validate_audit(audit,native)


@pytest.mark.parametrize('fault',['queued','changed-payload','bad-journal','wrong-actual','oom','swap','wrong-image'])
def test_reject_unsafe_or_unfinished_audit(prepared,fault):
    _,_,native,audit=prepared;r=audit['owners'][0]
    if fault=='queued':
        next(iter(r['state']['commands'].values()))['status']='QUEUED';repack(r)
    elif fault=='changed-payload':
        next(iter(r['state']['commands'].values()))['payload_sha256']='bad';repack(r)
    elif fault=='bad-journal':r['state_raw_sha256']='bad'
    elif fault=='wrong-actual':r['actual']['host_pid']=1
    elif fault=='oom':r['memory_events']['oom']=1
    elif fault=='swap':r['memory_swap_current']=1
    else:r['image_id']='other'
    with pytest.raises(ValueError):m.validate_audit(audit,native)


@pytest.mark.parametrize('field',['compute_release_authorized','allocation_changes_authorized','native_replay_allowed','restore_retired_checkpoint_allowed'])
def test_no_broadened_authority(prepared,field):
    root,_,_,_=prepared;v=m.read(root/'user-authorization-bc-smoke.json');v[field]=True
    with pytest.raises(ValueError):m.authority(v,m.binding(m.PREVIOUS/'native-bindings.json'))


@pytest.mark.parametrize('action',['stage','shutdown','restore'])
def test_mutations_need_explicit_execute(prepared,action):
    _,t,_,_=prepared
    with pytest.raises(ValueError,match='execute'):m.operate(action,transport=t)
    assert t.calls==[]


def test_stage_lost_reply_is_verified_without_second_stage(prepared):
    root,t,_,_=prepared;t.lost='stage'
    with pytest.raises(TimeoutError):m.operate('stage',execute=True,transport=t)
    assert (root/'lifecycle-stage-intent.json').exists()
    t.lost=None
    assert m.operate('stage',execute=True,transport=t)['staged']
    assert t.calls.count('stage')==1


def test_busy_peer_blocks_both_shutdowns(prepared):
    root,t,_,_=prepared;m.operate('stage',execute=True,transport=t);t.busy=1
    with pytest.raises(ValueError):m.operate('shutdown',execute=True,transport=t)
    assert 'shutdown' not in t.calls and not (root/'lifecycle-shutdown-g0/intent.json').exists()


def test_lost_shutdown_never_posts_again(prepared):
    root,t,_,_=prepared;m.operate('stage',execute=True,transport=t);t.lost='shutdown'
    with pytest.raises(TimeoutError):m.operate('shutdown',execute=True,transport=t)
    assert t.calls.count('shutdown')==1
    t.lost=None;t.exit_alive=True
    with pytest.raises(ValueError):m.operate('shutdown',execute=True,transport=t)
    assert t.calls.count('shutdown')==1
    t.exit_alive=False
    result=m.operate('shutdown',execute=True,transport=t)
    assert result[0]['phase']=='STOPPED' and result[1]['status']=='accepted'
    assert t.calls.count('shutdown')==2


def test_restore_needs_sealed_exits(prepared):
    _,t,_,_=prepared;m.operate('stage',execute=True,transport=t)
    with pytest.raises(FileNotFoundError):m.operate('restore',execute=True,transport=t)
    assert 'launch' not in t.calls


def test_lost_restore_observes_only_never_relaunches(prepared):
    root,t,_,_=prepared;m.operate('stage',execute=True,transport=t)
    m.operate('reconcile',transport=t);t.lost='launch'
    with pytest.raises(TimeoutError):m.operate('restore',execute=True,transport=t)
    assert (root/'lifecycle-restore-intent-g0.unknown.json').exists()
    t.lost=None
    assert m.operate('restore',execute=True,transport=t)['state']=='OBSERVED_ONLY'
    assert t.calls.count('launch')==1


def test_complete_mock_lifecycle_keeps_compute_and_no_checkpoint(prepared):
    root,t,_,_=prepared
    m.operate('stage',execute=True,transport=t)
    m.operate('shutdown',execute=True,transport=t)
    m.operate('reconcile',transport=t)
    for g in (0,1):m.operate('restore',gpu=g,execute=True,transport=t)
    result=m.operate('bind',transport=t)
    assert not result['compute_release_authorized'] and not result['checkpoint_loaded_at_startup']
    assert not result['retired_checkpoint_restored'] and not result['production_certified']
    assert len(result['workers'])==2
    assert t.calls.count('shutdown')==2 and t.calls.count('launch')==2


def test_remote_programs_are_valid_and_transport_is_single_shot():
    compile(m.STAGE,'stage','exec');compile(m.n.REMOTE,'native','exec')
    assert callable(m.n.validate_boundary)
    import inspect
    body=inspect.getsource(m.n.Transport.invoke)
    assert body.count('subprocess.run(')==1 and 'while ' not in body
    assert 'ResidentClient' not in body
