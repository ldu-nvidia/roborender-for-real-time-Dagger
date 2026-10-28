"""Read-only pre-work recovery does not stitch samples or restart native work."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(name):
    spec=importlib.util.spec_from_file_location('test_'+name,ROOT/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

@pytest.mark.parametrize('state',['RUNNING','STOPPED_IDLE',None])
def test_recovery_requires_failed_exited_observer(state):
    m=module('telemetry_v5_recovery');p=dict(pid=1,identity={'a':1})
    with pytest.raises(ValueError,match='exited'):m.check_exited(dict(state=state,identity=p['identity']),p,lambda _:None)

def test_recovery_rejects_live_or_changed_observer():
    m=module('telemetry_v5_recovery');p=dict(pid=1,identity={'a':1});status=dict(state='FAILED',identity=p['identity'])
    with pytest.raises(ValueError,match='exited'):m.check_exited(status,p,lambda _:{'a':1})
    with pytest.raises(ValueError,match='exited'):m.check_exited(dict(status,identity={'b':1}),p,lambda _:None)
    m.check_exited(status,p,lambda _:None)

def test_fresh_scope_retains_source_owner_and_transport_is_metadata_only():
    m=module('live_v5r')
    assert m.v.v.ROOT==m.b.ROOT==m.ROOT==ROOT/'attempt-v5r'
    assert m.v.v.r is m.r and m.r.m.OUT==m.r.OUT
    assert m.r.m.SOCKET==ROOT.parent/'parity26-t6oMpN-v5r.sock'
    assert m.u.ROOT==ROOT/'native-v5'
    assert m.b.u is m.u
    assert m.r.v.ROOT==ROOT/'attempt-v5'
    assert m.r.m.load is m.r.load
    assert m.r.start_master is m.r.m.start_master
