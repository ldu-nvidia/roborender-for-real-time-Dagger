"""No synthetic capacity, same full ladder, no launch after horizon."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
def load():
    spec=importlib.util.spec_from_file_location('candidate_v6_test',ROOT/'candidate_v6.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_exact_current_continuous_scope():
    m=load()
    assert m.m.c4 is m.c4 and m.old.c4 is m.c4
    assert m.c4.OUTPUT==ROOT/'continuous-v6-output'
    assert m.u.ROOT==ROOT/'native-v6'
    assert m.m.LABELS==('b4','b8','b16','b16-repeat','tail7')
    assert m.m.TEST.name=='test_parity_candidate_v6.py'
def test_no_completion_receipt_no_candidate(monkeypatch,tmp_path):
    m=load()
    monkeypatch.setattr(m.c4,'load',lambda:(None,None,None,None))
    monkeypatch.setattr(m.c4,'OUTPUT',tmp_path)
    with pytest.raises(ValueError,match='regular input'):m.m.initial_candidate()
def test_incomplete_prefix_is_not_replayed(tmp_path):
    m=load();(tmp_path/'prefix').mkdir();(tmp_path/'prefix/b4-intent.json').touch()
    with pytest.raises(ValueError,match='reconciliation'):m.old.next_prefix(tmp_path)
