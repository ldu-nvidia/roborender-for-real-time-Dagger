"""Synthetic no-GPU guard checks for the isolated live attempt."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'live_v4.py'
if not PATH.is_file():pytest.skip('runtime adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('parity_live_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_new_namespace_source_and_command_ids_do_not_reuse_failure():
    assert m.b.ROOT==ROOT/'attempt-v4' and m.b.PREFIX.endswith('-v4')
    assert m.b.u.ROOT==ROOT/'native-v4' and m.b.u.SOURCE!=m.b.u.u.SOURCE
    assert m.b.__file__==str(PATH)
    assert m.b.HELPERS['upgrade'][0]==Path(m.u.__file__)

@pytest.mark.parametrize('label',['full295','final','round1','b32',None])
def test_no_unbounded_evaluation_or_training(label,monkeypatch):
    monkeypatch.setattr(m.b,'load',lambda:pytest.fail('no I/O for invalid scope'))
    with pytest.raises(ValueError):m.b.evaluate(label)

def test_live_master_retained_existing_sampler_never_duplicated(tmp_path,monkeypatch):
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m.r,'verify',lambda _:None)
    m.immutable(tmp_path/'launch-intent.json',dict(existing=True),new=True)
    sampler=m.Sampler(dict(sampler_output=str(tmp_path)),SimpleNamespace(fleet={}))
    monkeypatch.setattr(sampler,'healthy',lambda:dict(healthy=True))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate sampler'))
    assert sampler.start()==dict(healthy=True)
