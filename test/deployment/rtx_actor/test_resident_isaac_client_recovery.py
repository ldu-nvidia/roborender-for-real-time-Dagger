"""Read-only native-command polling survives transport loss, never owner loss."""
from test.deployment.source_paths import ACTOR_ROOT
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0,str(ACTOR_ROOT))
import resident_isaac_client as module


@pytest.fixture
def clock(monkeypatch):
    now=[0.0]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    def sleep(seconds):
        assert seconds>=0
        now[0]+=seconds
    monkeypatch.setattr(module.time,'sleep',sleep)
    return now


def ready(**changes):
    return dict(role='isaac_resident',schema='roborender.isaac-resident.v1',ok=True,
        source_sha256='a'*64,gpu_uuid='GPU-owned',native_initializations=1,
        worker_instance_id='owned-instance',native_pid=100,**changes)


def client_with(request):
    value=object.__new__(module.ResidentClient)
    value.source,value.gpu,value.instance='a'*64,'GPU-owned','owned-instance'
    value._request=request
    value.submit=lambda *_args,**_kwargs:pytest.fail('polling must never submit native work')
    return value


@pytest.mark.parametrize('failure',[URLError('transport down'),TimeoutError('transport down'),ConnectionError('transport down')])
@pytest.mark.parametrize('route',['/ready','/v1/commands/owned-command'])
def test_transport_outage_recovers_same_command_and_rechecks_exact_owner(clock,failure,route):
    calls=[]
    completed=dict(status='COMPLETE',receipt=dict(complete=True,unchanged_native_command=True))
    raised=False
    def request(path):
        nonlocal raised
        calls.append(path)
        if path==route and not raised:
            raised=True
            raise failure
        return ready() if path=='/ready' else completed
    result=client_with(request).wait('owned-command',wall_budget=10,interval=1)
    assert result is completed and raised
    assert set(calls)<= {'/ready','/v1/commands/owned-command'}
    assert calls.count('/ready')==2
    assert calls[-2:]==['/ready','/v1/commands/owned-command']


@pytest.mark.parametrize('status',[401,403,404,409,500])
@pytest.mark.parametrize('route',['/ready','/v1/commands/owned-command'])
def test_explicit_http_rejection_is_terminal_and_not_hidden_by_urlerror(clock,status,route):
    calls=[]
    error=HTTPError('http://127.0.0.1/owned',status,'private server reason',None,None)
    def request(path):
        calls.append(path)
        if path==route:raise error
        return ready()
    with pytest.raises(HTTPError) as exc:
        client_with(request).wait('owned-command',wall_budget=10,interval=1)
    assert exc.value is error and calls.count(route)==1 and clock[0]==0


@pytest.mark.parametrize('identity',[dict(worker_instance_id='replacement'),dict(source_sha256='b'*64),
    dict(gpu_uuid='GPU-other'),dict(native_initializations=2)])
def test_transport_recovery_never_treats_replacement_owner_as_resume(clock,identity):
    calls=[]
    def request(path):
        calls.append(path)
        if len(calls)==1:raise ConnectionError('temporary route loss')
        value=ready();value.update(identity)
        return value
    with pytest.raises(RuntimeError,match='identity mismatch|owner restarted'):
        client_with(request).wait('owned-command',wall_budget=10,interval=1)
    assert calls==['/ready','/ready']


@pytest.mark.parametrize('status',['FAILED','CRASH_UNCERTAIN','MISSING','UNKNOWN'])
def test_definitive_failed_or_uncertain_native_phase_is_terminal(clock,status):
    calls=[]
    def request(path):
        calls.append(path)
        return ready() if path=='/ready' else dict(status=status)
    with pytest.raises(RuntimeError,match='failed/uncertain'):
        client_with(request).wait('owned-command',wall_budget=10,interval=1)
    assert calls==['/ready','/v1/commands/owned-command'] and clock[0]==0


def test_zero_observer_budget_never_contacts_or_replays(clock):
    with pytest.raises(TimeoutError,match='still running'):
        client_with(lambda *_:pytest.fail('expired observer must detach')).wait('owned-command',wall_budget=0)


def test_repeated_transport_failure_expires_without_submit_stop_or_release(clock):
    calls=[]
    def request(path):
        calls.append(path)
        raise URLError('offline')
    with pytest.raises(TimeoutError,match='inspect its command receipt'):
        client_with(request).wait('owned-command',wall_budget=3,interval=1)
    assert calls==['/ready']*3 and clock[0]==3


def test_running_after_transport_recovery_then_complete_does_not_change_command(clock):
    calls=[]
    states=iter([dict(status='RUNNING'),ConnectionError('lost one read'),dict(status='COMPLETE',receipt={'ok':True})])
    def request(path):
        calls.append(path)
        if path=='/ready':return ready()
        value=next(states)
        if isinstance(value,Exception):raise value
        return value
    out=client_with(request).wait('owned-command',wall_budget=10,interval=1)
    assert out['receipt']=={'ok':True}
    assert calls==['/ready','/v1/commands/owned-command']*3
