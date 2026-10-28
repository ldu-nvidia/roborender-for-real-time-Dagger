"""Recovery changes CPU-controller attempt paths only, not native semantics."""
import importlib.util
import inspect
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
spec=importlib.util.spec_from_file_location('preflight_recovery_test',ROOT/'candidate_recover_preflight.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_reviewed_semantic_driver_diff_is_only_cpu_journals():
    function,source=m.recovered_function()
    original=inspect.getsource(m.c.m.execute)
    assert source.replace('recovery-1/launch-intent.json','launch-intent.json').replace(
        'recovery-1/failure.json','failure.json')==original
    assert function.__globals__['PLAN']==m.c.PLAN
    assert function.__globals__['OUT']==m.c.OUT
    assert function.__globals__['LABELS']==('b4','b8','b16','b16-repeat','tail7')

def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(m.c,'OUT',tmp_path)
    monkeypatch.setattr(m,'RECOVERY',tmp_path/'recovery-1')
    monkeypatch.setattr(m.s,'binding',lambda _: {'path':'old','raw_sha256':'same'})
    monkeypatch.setattr(m.s,'identity',lambda _: None)
    def read(path):
        if path.name=='launch-intent.json':return dict(plan={'path':'old','raw_sha256':'same'},identity={'pid':1})
        return dict(error_type='RuntimeError',message='exact-owner operation failed; no native replay')
    monkeypatch.setattr(m.s,'read',read)

@pytest.mark.parametrize('name',['submit-attempt-00.json','acceptance.json','commit.json','dispatch-evidence.json'])
def test_any_prior_native_attempt_blocks_recovery(monkeypatch,tmp_path,name):
    setup(monkeypatch,tmp_path);p=tmp_path/'b4/cohort-000';p.mkdir(parents=True);(p/name).write_text('{}')
    with pytest.raises(ValueError,match='accepted work exists'):m.check_local_boundary()

def test_live_old_controller_blocks_duplicate(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path);monkeypatch.setattr(m.s,'identity',lambda _: {'pid':1})
    with pytest.raises(ValueError,match='exited preflight'):m.check_local_boundary()

def test_second_recovery_rejected(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path);(tmp_path/'recovery-1').mkdir()
    with pytest.raises(ValueError,match='cannot be replayed'):m.check_local_boundary()
