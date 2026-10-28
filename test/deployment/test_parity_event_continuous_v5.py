"""Bounded requalification is not a relabelled capacity/production certificate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'continuous_v5.py'

@pytest.fixture
def m():
    spec=importlib.util.spec_from_file_location('event_continuous_tests',PATH)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_reviewed_delta_matches_exact_two_frozen_manifests(m):
    old=m.read('/workspace/project/roborender-artifacts/code_releases/raven-parity-route-20261026-v4/source-manifest.json')
    new=m.read(m.u.RELEASE/'source-manifest.json')
    changed=m.source_delta(old,new)
    assert 'pipeline/dagger/rtx/code/src/dagger_rtx/phase_reset_state.py' in changed
    assert len(m.ALLOWED)==7
    altered=dict(new,files=[*new['files'],dict(path='pipeline/unreviewed.py',sha256='a'*64)])
    with pytest.raises(ValueError,match='unreviewed'):m.source_delta(old,altered)
    with pytest.raises(ValueError,match='unreviewed'):m.source_delta(old,old)

def test_fresh_namespaces_unified_source_and_four_update_only(m):
    assert m.PLAN==ROOT/'continuous-v5-plan.json' and m.OUTPUT==ROOT/'continuous-v5-output'
    assert m.m.base.RUN==m.RUN and m.RUN.endswith('-v5')
    assert m.m.u is m.u and m.m.b.u is m.u
    assert m.m.qualify is m.qualify
    assert m.v.v.ROOT==ROOT/'attempt-v5r'
    assert m.v.v.__file__==str(ROOT/'live_v5r.py')
    assert m.v.b.__file__==str(ROOT/'live_v5r.py')
    assert m.m.__file__==str(PATH)

def test_bridge_never_promotes_historical_source(m,monkeypatch):
    old=m.read('/workspace/project/roborender-artifacts/code_releases/raven-parity-route-20261026-v4/source-manifest.json')
    proof={'real':'proof'};history=dict(schema='roborender.historical-capacity-evidence.v1',source_sha256=m.u.OLD_SOURCE,
        current_source_qualified=False,compute_release_authorized=False,live_plan={'id':'plan'},proof=proof,
        dispatches=[{'id':'dispatch'}],manifest={'id':'manifest'})
    calls=[]
    def bound(ref):
        if ref.get('id')=='plan':return {'run':{'panel':{},'config':{}}}
        if ref.get('id')=='manifest':return old
        if ref.get('id')=='dispatch':return {'actual':'dispatch'}
        return history
    monkeypatch.setattr(m.s,'binding',lambda p:{'path':str(p),'raw_sha256':'b'*64})
    monkeypatch.setattr(m.s,'bound',bound)
    monkeypatch.setattr(m.u,'cpu_gate',lambda:{'tested':True})
    c=SimpleNamespace(native={'source_sha256':m.u.SOURCE,'workers':[{'worker_instance_id':'new0'},{'worker_instance_id':'new1'}]},
        paired=SimpleNamespace(validate_qualification=lambda p,**kw:calls.append(kw['source_sha256']),
            validate_dispatch_evidence=lambda d:calls.append(d)))
    live={'source_sha256':m.u.SOURCE,'native':{'actual':'new'}}
    bridge,refs=m.qualify(c,live)
    assert calls[0]==m.u.OLD_SOURCE
    assert bridge['historical_source_sha256']!=bridge['current_source_sha256']
    assert bridge['current_source_capacity_certified'] is bridge['production_admission'] is False
    assert bridge['optimizer_updates_max']==4 and bridge['requires_actual_current_source_terminal_proof'] is True
    c.native['workers'][0]['worker_instance_id']=m.u.OLD_WORKERS[0]
    with pytest.raises(ValueError,match='fresh'):m.qualify(c,live)

def test_existing_accepted_training_can_only_be_observed(m,monkeypatch,tmp_path):
    monkeypatch.setattr(m.m,'OUTPUT',tmp_path)
    m.immutable(tmp_path/'launch-intent.json',{'plan':{'same':True}},new=True)
    monkeypatch.setattr(m.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m.m,'load',lambda:({},None,None,None))
    monkeypatch.setattr(m.m,'binding',lambda p:{'same':True})
    monkeypatch.setattr(m.m,'observe',lambda:{'read_only':True})
    monkeypatch.setattr(m.m,'retained',lambda *a:pytest.fail('duplicate service launch'))
    assert m.execute()=={'read_only':True}
