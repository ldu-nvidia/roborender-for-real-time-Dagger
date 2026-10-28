import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('watchdog4_test',ROOT/'watchdog_v4.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_same_lock_interval_and_new_pair():
    assert m.m.base.ROOT==ROOT
    assert m.m.probe.BINDINGS==ROOT/'native-v5b/warm-bindings.json'
    assert m.m.base.INTERVAL==600

def test_current_notice_route_and_deduplication(tmp_path):
    calls=[]
    def runner(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout='Queued message test-double')
    value=dict(phase='TEST_ONLY',health='HEALTHY',faults=[],complete=False,deadline_reached=False)
    assert m.deliver(value,root=tmp_path,runner=runner,now=100)['delivery']=='QUEUED'
    assert calls[0][-1].count(str(tmp_path/'watchdog_v4.py'))==2
    m.deliver(value,root=tmp_path,runner=runner,now=200)
    assert len(calls)==1
