"""Only adapter wiring; the original semantic/receipt gates stay unchanged."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(name):
    path=ROOT/(name+'.py');spec=importlib.util.spec_from_file_location('test_'+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

@pytest.mark.parametrize('name',['candidate_v5','return_v5','final_test_v5'])
def test_fresh_scope_pinned_source_and_actual_gate(name):
    x=module(name)
    assert x.m.__file__==str(ROOT/(name+'.py'))
    assert x.RUN.endswith('-v5') and x.m.RUN==x.RUN
    assert x.PLAN==x.m.PLAN and x.OUT==x.m.OUT and 'v5' in x.PLAN.name
    assert x.u.SOURCE=='e6b88f56ea5545a00d5c509db6a88ef59777d0c6af465417cbff1d3257b07692'
    assert x.v.ROOT==ROOT/'attempt-v5r'
    assert x.m.execute is x.execute
    assert x.c4.OUTPUT==ROOT/'continuous-v5-output'
    if name=='candidate_v5':
        assert x.LABELS==('b4','b8','b16','b16-repeat','tail7')
        assert x.m.initial_candidate is x.initial_candidate
        assert x.v.Sampler is x.observer.Sampler
        assert x.v.r is x.observer.r
        assert x.c4.load is x.observed_continuous_load
    else:
        assert x.candidate.OUT==ROOT/'candidate-v5-output'
        assert 'compute_release_authorized' in x.FLAGS
    if name=='final_test_v5':
        assert x.LABELS==('b16','tail7') and x.ret.OUT==ROOT/'return-v5-output'

@pytest.mark.parametrize('name',['candidate_v5','return_v5','final_test_v5'])
def test_preparation_cannot_overwrite_an_existing_scope(name,tmp_path,monkeypatch):
    x=module(name);path=tmp_path/'plan.json';path.write_text('{}')
    monkeypatch.setattr(x.m,'PLAN',path);monkeypatch.setattr(x.u,'authority',lambda **kw:None)
    with pytest.raises(ValueError,match='fresh'):x.prepare()

@pytest.mark.parametrize('bad',[{},dict(status='COMPLETE'),dict(status='DIAGNOSTIC_COMPLETE',optimizer_updates=100)])
def test_candidate_requires_real_four_update_receipt(bad,monkeypatch):
    x=module('candidate_v5')
    monkeypatch.setattr(x.c4,'load',lambda:({},None,None,None))
    monkeypatch.setattr(x.m,'read',lambda _:bad)
    monkeypatch.setattr(x.m,'binding',lambda _: {})
    with pytest.raises(ValueError,match='completed current four-update'):x.initial_candidate()

def test_return_reuses_original_sampler_implementation():
    x=module('return_v5')
    assert x.Sampler is x.m.Sampler
    assert x.m.sample is x.sample and x.m.candidate is x.candidate
    assert x.m.c4 is x.c4

@pytest.mark.parametrize('name',['candidate_v5','return_v5','final_test_v5'])
def test_reviewed_base_cannot_drift(name,tmp_path):
    x=module(name);path=tmp_path/'changed.py';path.write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError,match='drift'):x.module(path,'never','0'*64)
