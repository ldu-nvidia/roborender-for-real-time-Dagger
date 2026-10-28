"""Bounded fresh-source recovery, never replay or production qualification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.fixture
def m():return module(ROOT/'continuous_v6.py','v6_guard_test')
def test_fresh_scope_and_all_live_bindings(m):
    assert m.PLAN==ROOT/'continuous-v6-plan.json'
    assert m.u.ROOT==ROOT/'native-v6'
    assert m.b.u is m.u and m.m.u is m.u and m.old.u is m.u
    assert m.old.old.u is m.u
    assert m.m.base.RUN==m.RUN
    assert m.m.TEST.name=='test_parity_v6_recovery.py'
    assert m.v.Sampler is m.host.Sampler
    assert m.host.SAMPLES==ROOT/'host-journal-v3/combined'
def test_new_source_gate_and_exact_terminal_error(m):
    assert m.u.SOURCE=='6052af310ef5db8b7d9a5d1dee1522f3e7375b2fe4b0c0966668b97760e06c14'
    assert "command['error_type']=='AttributeError'" in m.u.REMOTE
    assert "'Actor' object has no attribute 'distribution'" in m.u.REMOTE
    assert 'check_exit(p)' in m.u.REMOTE
    assert m.u.cpu_gate()['path']==str(ROOT/'frozen-v6-ready.xml')
def test_reviewed_production_delta_only(m):
    previous=m.s.bound(m.s.read(ROOT/'v4-capacity-evidence.json')['manifest'])
    current=m.s.read(m.u.RELEASE/'source-manifest.json')
    changes=m.old.old.source_delta(previous,current)
    assert 'pipeline/dagger/rtx/code/src/dagger_rtx/native_service_trainer.py' in changes
    bad=dict(current,files=current['files']+[dict(path='pipeline/unreviewed.py',sha256='0'*64,bytes=1,executable=False)])
    with pytest.raises(ValueError,match='unreviewed'):m.old.old.source_delta(previous,bad)
def test_accepted_controller_is_only_observed(m,monkeypatch,tmp_path):
    monkeypatch.setattr(m.m,'OUTPUT',tmp_path)
    m.s.immutable(tmp_path/'launch-intent.json',{'plan':{'same':True}},new=True)
    monkeypatch.setattr(m.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m.m,'load',lambda:({},None,None,None))
    monkeypatch.setattr(m.m,'binding',lambda p:{'same':True})
    monkeypatch.setattr(m.m,'observe',lambda:{'read_only':True})
    monkeypatch.setattr(m.m,'retained',lambda *a:pytest.fail('duplicate start'))
    assert m.execute()=={'read_only':True}
def test_only_owned_learner_placement(m,monkeypatch):
    calls=[]
    site=SimpleNamespace(process=lambda *a,**kw:calls.append((a,kw)),prepare=lambda:None)
    monkeypatch.setattr(m.old,'original_site',lambda *a:site)
    placed=m.m.make_site({},SimpleNamespace(fleet={'workers':[{'gpu_uuid':'GPU-0'}]}),None,None,None,None,None)
    placed.process('gpu-services','start',host='gpu',env={'KEEP':'1'})
    assert calls[-1][1]['env']=={'KEEP':'1','CUDA_VISIBLE_DEVICES':'GPU-0'}
    placed.process('tunnels','start',host='beta')
    assert 'env' not in calls[-1][1]
def test_reuses_existing_h200_resource_producer(m):
    assert 'start-renderer' not in Path(m.host.__file__).read_text()
    assert m.host.m.c5.b.u.ROOT==ROOT/'native-v6'
    assert m.host.m.__file__==str(ROOT/'host_resource_transport_v3.py')
