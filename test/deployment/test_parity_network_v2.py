"""No simulator/compute actions in a new route for the proven-ended SSH unit."""
import importlib.util
import json
from pathlib import Path
import time
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module():
    spec=importlib.util.spec_from_file_location('network_v2_tests',ROOT/'network_v2.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def test_route_only_uses_new_unit_and_same_restricted_ports():
    x=module();assert x.m.ROOT==ROOT/'network-v2'
    assert x.m.UNIT.endswith('-v2.service') and x.m.REMOTE_ROOT.endswith('-v2')
    assert x.OLD_REMOTE not in x.m.REMOTE
    assert x.m.PORTS==list(range(18560,18568))+[22036]
    assert "'--property=Restart=no'" in x.m.REMOTE
    assert "state['MainPID']=='0'" in x.CHECK and "state['ExecMainStatus']=='255'" in x.CHECK
def test_prepare_never_overwrites_accepted_route(monkeypatch,tmp_path):
    x=module();monkeypatch.setattr(x,'NEW',tmp_path)
    # Exercise duplicate-admission rejection inside the historical permission
    # window, independently of the wall clock when this regression is run.
    deadline=json.loads((ROOT/'user-authorization.json').read_bytes())['deadline_unix']
    monkeypatch.setattr(time,'time',lambda:deadline-1)
    monkeypatch.setattr(x.subprocess,'run',lambda *a,**k:pytest.fail('duplicate route must not invoke SSH'))
    with pytest.raises(ValueError,match='already prepared'):x.prepare()

def test_expired_historical_window_rejects_before_any_route_action(monkeypatch,tmp_path):
    x=module();monkeypatch.setattr(x,'NEW',tmp_path/'absent')
    deadline=json.loads((ROOT/'user-authorization.json').read_bytes())['deadline_unix']
    monkeypatch.setattr(time,'time',lambda:deadline)
    monkeypatch.setattr(x.subprocess,'run',lambda *a,**k:pytest.fail('expired authority must not invoke SSH'))
    with pytest.raises(ValueError,match='deadline reached'):x.prepare()
