"""New authority must not bypass the actual four-update V6 candidate proof."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('renewed_candidate_test',ROOT/'candidate_resumed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_only_new_control_globals_change():
    assert m.u is not m.c4.u and m.v is not m.c4.v
    assert m.u.SOURCE==m.c4.u.SOURCE=='6052af310ef5db8b7d9a5d1dee1522f3e7375b2fe4b0c0966668b97760e06c14'
    assert m.u.authority is m.auth.authority
    assert m.c4.PLAN==ROOT/'continuous-v6-plan.json'
    assert m.m.LABELS==('b4','b8','b16','b16-repeat','tail7')
    assert m.v.Sampler is m.journal.Sampler
    with pytest.raises(ValueError,match='horizon'):m.c4.u.authority(new_work=True)

def test_read_only_historical_admission_is_not_modified(monkeypatch):
    live=dict(deadline_unix=1,sampler_output='old',native='unchanged',run='unchanged')
    rows=({},live,object(),{}, {}, {}, {'raw_sha256':'candidate'}, ['curriculum'])
    monkeypatch.setattr(m,'original_candidate',lambda: rows)
    result=m.initial_candidate()
    assert live['deadline_unix']==1 and live['sampler_output']=='old'
    assert result[1]==dict(live,deadline_unix=None,sampler_output=str(m.journal.SAMPLES))
    assert result[0] is rows[0] and result[6] is rows[6] and result[7] is rows[7]

def test_failed_candidate_evidence_cannot_be_renewed(monkeypatch):
    def fail():raise ValueError('actual four-update proof missing')
    monkeypatch.setattr(m,'original_candidate',fail)
    with pytest.raises(ValueError,match='four-update'):m.initial_candidate()

def test_synthetic_deadline_rejected(monkeypatch):
    monkeypatch.setattr(m,'original_load',lambda: ({'deadline_unix':10**12},{},object()))
    with pytest.raises(ValueError,match='synthetic horizon'):m.load()

def test_real_journal_admission_still_required(monkeypatch):
    monkeypatch.setattr(m,'original_load',lambda: ({'deadline_unix':None},{},object()))
    def fail():raise ValueError('journal identity changed')
    monkeypatch.setattr(m.journal,'load',fail)
    with pytest.raises(ValueError,match='journal identity'):m.load()

def test_helpers_bind_renewed_control_and_original_continuous_validator():
    refs=m.helpers()
    assert refs['continuous']['path']==str(ROOT/'continuous_v6.py')
    assert refs['adapter']['path']==str(ROOT/'candidate_resumed.py')
    assert refs['renewed_authority']['path']==str(ROOT/'resume_authority.py')
