import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('watchdog3_test',ROOT/'watchdog_v3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_same_watchdog_lock_and_new_actual_binding():
    assert m.m.base.ROOT==ROOT
    assert m.m.probe.BINDINGS==ROOT/'native-v5/warm-bindings.json'
    assert m.m.base.INTERVAL==600

def test_notice_uses_v3_for_check_and_ack(tmp_path):
    calls=[]
    def runner(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout='Queued message actual-transport-test')
    value=dict(phase='TEST_ONLY',health='HEALTHY',faults=[],complete=False,deadline_reached=False)
    result=m.deliver(value,root=tmp_path,runner=runner,now=100)
    assert result['delivery']=='QUEUED' and len(calls)==1
    assert calls[0][-1].count(str(tmp_path/'watchdog_v3.py'))==2
    assert 'watchdog_v2.py' not in calls[0][-1]
    m.deliver(value,root=tmp_path,runner=runner,now=200)
    assert len(calls)==1  # Healthy unchanged state never creates a second notice.
