"""Renewed phase adapters retain genuine predecessor and state-isolation gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def module(name):
    spec=importlib.util.spec_from_file_location('test_warm_resumed_'+name,ROOT/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

@pytest.fixture(scope='module')
def ret():return module('return_resumed')
@pytest.fixture(scope='module')
def final():return module('final_test_resumed')

def test_actual_v6_predecessor_scopes_and_four_update_bound(ret,final):
    assert ret.c4.PLAN==ROOT/'continuous-v6-plan.json'
    assert ret.candidate.PLAN==ROOT/'candidate-resumed-plan.json'
    assert final.ret.PLAN==ROOT/'return-resumed-plan.json'
    assert ret.u.authority is ret.candidate.auth.authority
    assert final.u.SOURCE==ret.u.SOURCE=='6052af310ef5db8b7d9a5d1dee1522f3e7375b2fe4b0c0966668b97760e06c14'
    assert ret.m.predecessor.__code__.co_filename==str(ROOT/'return_v4.py')
    assert final.m.predecessor.__code__.co_filename==str(ROOT/'final_test_v4.py')
    assert 'update5_authorized' in ret.FLAGS and 'full295_claimed' in final.FLAGS
    assert final.LABELS==('b16','tail7')

@pytest.mark.parametrize('name',['return_resumed','final_test_resumed'])
def test_existing_preparation_is_not_replaced(name,tmp_path,monkeypatch):
    value=module(name);path=tmp_path/'existing.json';path.write_text('{}')
    monkeypatch.setattr(value.m,'PLAN',path)
    monkeypatch.setattr(value.m,'predecessor',lambda:pytest.fail('existing scope must reject first'))
    with pytest.raises(ValueError,match='fresh'):value.prepare()

def test_return_cannot_admit_missing_candidate_result(ret,monkeypatch):
    def fail():raise ValueError('actual candidate evidence missing')
    monkeypatch.setattr(ret.candidate,'initial_candidate',fail)
    with pytest.raises(ValueError,match='actual candidate'):ret.m.predecessor()

def test_final_cannot_admit_missing_return_result(final,monkeypatch):
    def fail():raise ValueError('actual return evidence missing')
    monkeypatch.setattr(final.ret,'load',fail)
    with pytest.raises(ValueError,match='actual return'):final.m.predecessor()

def test_metadata_reuses_exact_producers_only_until_boundary(ret):
    r=ret.resources
    assert r.old.OUT==ROOT/'host-journal-resumed-v1'
    assert r.r.OUT==r.OUT==ret.OUT/'metadata'
    assert r.SAMPLES==ret.OUT/'joint-sampler'
    assert r.r.m.SAMPLES==r.SAMPLES and r.r.m.__file__==str(ROOT/'return_resources.py')
    assert r.r.old is r.old and r.r.m.load is r.r.load
    assert ret.m.Sampler is r.Sampler

def test_return_sampler_cannot_duplicate_accepted_receiver(ret,tmp_path,monkeypatch):
    r=ret.resources;monkeypatch.setattr(r,'SAMPLES',tmp_path)
    (tmp_path/'launch-intent.json').write_text('{}')
    sampler=object.__new__(r.Sampler)
    monkeypatch.setattr(sampler,'healthy',lambda:dict(state='RUNNING'))
    monkeypatch.setattr(r,'start',lambda:pytest.fail('duplicate observer'))
    assert sampler.start()==dict(state='RUNNING')

def test_return_cannot_retire_a_live_candidate_receiver(ret,monkeypatch):
    r=ret.resources
    monkeypatch.setattr(r.old,'load',lambda: ({},None))
    monkeypatch.setattr(r.s,'read',lambda _:dict(state='RUNNING',identity={'pid':9}))
    with pytest.raises(ValueError,match='already be terminal'):r.retire()

def test_unchanged_return_core_preserves_learner_and_training_curriculum(ret):
    source=(ROOT/'return_v4.py').read_text()
    assert "training_curriculum=p['training_curriculum']" in source
    assert "diagnostic_learner_chain=p['chain']['path']" in source
    assert "row['optimizer_parameter_states']!=78" in source
    assert "r['input_rng_sha256']!=rng" in source
    assert "('gpu-services','status','gpu')" in source
    assert "if (OUT/'launch-intent.json').exists():raise" in source

def test_reviewed_semantic_adapter_cannot_drift(ret,tmp_path,monkeypatch):
    path=tmp_path/'changed.py';path.write_text('raise AssertionError("must not execute")')
    monkeypatch.setattr(ret,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='changed'):ret.module('changed','0'*64)
