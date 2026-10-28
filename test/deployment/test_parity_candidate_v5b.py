import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
@pytest.fixture
def m():
    spec=importlib.util.spec_from_file_location('fresh_candidate_test',ROOT/'candidate_v5b.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def test_same_actual_owners_and_full_ladder(m):
    assert m.LABELS==('b4','b8','b16','b16-repeat','tail7')
    assert m.u.ROOT==ROOT/'native-v5b'
    assert m.c4.OUTPUT==ROOT/'continuous-v5b-output'
    assert m.v.Sampler is m.c4.host.Sampler
    assert m.m.OUT==ROOT/'candidate-v5b-output'
def test_no_candidate_without_actual_completed_learning(m,monkeypatch,tmp_path):
    monkeypatch.setattr(m.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m.m,'PLAN',tmp_path/'plan.json')
    monkeypatch.setattr(m.m,'OUT',tmp_path/'output')
    def absent():raise ValueError('actual completed full32x81 native/learner scope required')
    monkeypatch.setattr(m.m,'initial_candidate',absent)
    with pytest.raises(ValueError,match='actual completed'):m.prepare()
    assert not (tmp_path/'plan.json').exists()
def test_prefix_never_replays_accepted_or_promotes_partial(m,tmp_path):
    assert m.next_prefix(tmp_path)=='b4'
    (tmp_path/'prefix').mkdir()
    (tmp_path/'prefix/b4-intent.json').write_text('{}')
    with pytest.raises(ValueError,match='reconciliation'):m.next_prefix(tmp_path)
    (tmp_path/'prefix/b4-result.json').write_text('{}')
    assert m.next_prefix(tmp_path)=='b8'
    for label in m.LABELS:(tmp_path/'prefix'/f'{label}-result.json').write_text('{}')
    with pytest.raises(ValueError,match='aggregation'):m.next_prefix(tmp_path)
def test_horizon_decline_occurs_before_native_or_launch_intent(m,monkeypatch,tmp_path):
    monkeypatch.setattr(m,'OUT',tmp_path)
    monkeypatch.setattr(m,'PLAN',tmp_path/'plan.json')
    m.s.immutable(m.PLAN,{'test':'only'},new=True)
    plan={'checkpoint':{'test':'only'},'deadline_unix':1100}
    m.s.immutable(tmp_path/'candidate-staged.json',{'plan':m.s.binding(m.PLAN),'checkpoint':plan['checkpoint']},new=True)
    monkeypatch.setattr(m.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m,'load',lambda:(plan,None,None))
    monkeypatch.setattr(m.time,'time',lambda:1000)
    monkeypatch.setattr(m.v,'Sampler',lambda *a:pytest.fail('no observer/native admission'))
    with pytest.raises(TimeoutError,match='insufficient horizon'):m.execute_next()
    assert not (tmp_path/'prefix').exists()
