"""The replacement observer uses one existing queue/lock and exact V6 owners."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('watchdog5_test',ROOT/'watchdog_v5.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_same_lock_interval_new_pair():
    assert m.m.base.ROOT==ROOT
    assert m.m.probe.BINDINGS==ROOT/'native-v6/warm-bindings.json'
    assert m.m.base.INTERVAL==600
def test_route_and_no_duplicate(tmp_path):
    calls=[]
    def runner(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout='Queued message test-double')
    value=dict(phase='TEST_ONLY',health='HEALTHY',faults=[],complete=False,deadline_reached=False)
    assert m.deliver(value,root=tmp_path,runner=runner,now=100)['delivery']=='QUEUED'
    assert calls[0][-1].count(str(tmp_path/'watchdog_v5.py'))==2
    m.deliver(value,root=tmp_path,runner=runner,now=200)
    assert len(calls)==1
def test_horizon_retains_observation_only_for_existing_pending_work(tmp_path):
    value=dict(progress={'accepted_operations':[{'status_file':str(tmp_path/'status.json')}]})
    assert m.accepted_work_pending(value) is True  # Unknown requires observation.
    m.m.base.save(tmp_path/'status.json',{'phase':'RUNNING'})
    assert m.accepted_work_pending(value) is True
    m.m.base.save(tmp_path/'status.json',{'phase':'FAILED'})
    assert m.accepted_work_pending(value) is False
    m.m.base.save(tmp_path/'status.json',{'phase':'COMPLETE'})
    assert m.accepted_work_pending(value) is False
    assert m.accepted_work_pending({'progress':{'accepted_operations':[]}}) is False
