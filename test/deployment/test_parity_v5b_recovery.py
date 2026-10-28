"""Fresh recovery retains algorithm/source and isolates only process placement."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.fixture
def m():return module(ROOT/'continuous_v5b.py','v5b_test_adapter')
def test_new_scopes_same_source_no_failed_scope_reuse(m):
    assert m.PLAN==ROOT/'continuous-v5b-plan.json'
    assert m.OUTPUT==ROOT/'continuous-v5b-output'
    assert m.u.ROOT==ROOT/'native-v5b'
    assert m.b.u is m.u and m.m.u is m.u
    assert m.u.SOURCE=='e6b88f56ea5545a00d5c509db6a88ef59777d0c6af465417cbff1d3257b07692'
    assert m.m.base.RUN==m.RUN and m.v.ROOT==ROOT/'attempt-v5b'
    assert m.host.SAMPLES==ROOT/'host-journal-v2/combined'
def test_only_learner_bundle_inherited_env_is_pinned(m,monkeypatch):
    calls=[]
    site=SimpleNamespace(process=lambda *a,**kw:calls.append((a,kw)),prepare=lambda:None)
    monkeypatch.setattr(m,'original_site',lambda *args:site)
    c=SimpleNamespace(fleet={'workers':[{'gpu_uuid':'GPU-first'}]})
    placed=m.make_site({},c,None,None,None,None,None)
    placed.process('gpu-services','start',host='gpu',env={'KEEP':'1'})
    assert calls[-1][1]['env']=={'KEEP':'1','CUDA_VISIBLE_DEVICES':'GPU-first'}
    placed.process('tunnels','start',host='beta')
    assert 'env' not in calls[-1][1]
    placed.process('gpu-services','status',host='gpu')
    assert 'env' not in calls[-1][1]
def test_accepted_controller_not_relaunched(m,monkeypatch,tmp_path):
    monkeypatch.setattr(m.m,'OUTPUT',tmp_path)
    m.s.immutable(tmp_path/'launch-intent.json',{'plan':{'same':True}},new=True)
    monkeypatch.setattr(m.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m.m,'load',lambda:({},None,None,None))
    monkeypatch.setattr(m.m,'binding',lambda p:{'same':True})
    monkeypatch.setattr(m.m,'observe',lambda:{'read_only':True})
    monkeypatch.setattr(m.m,'retained',lambda *a:pytest.fail('duplicate start'))
    assert m.execute()=={'read_only':True}
def test_resource_worker_unchanged_and_h200_producer_reused():
    host=module(ROOT/'host_resource_transport_v2.py','v5b_host_test')
    assert host.m.SAMPLES==host.SAMPLES
    assert host.m.OUT==host.OUT
    assert host.m.__file__==str(ROOT/'host_resource_transport_v2.py')
    assert host.m.c5.b.u.ROOT==ROOT/'native-v5b'
    assert "start-renderer" not in Path(host.__file__).read_text()
