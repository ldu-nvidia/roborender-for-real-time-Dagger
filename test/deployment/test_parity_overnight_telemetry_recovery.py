"""Read-only telemetry replacement cannot replay GPU work."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'live_recovery.py'
if not PATH.is_file():pytest.skip('operational adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('parity_telemetry_tests',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('label',['full295','round1',None,'b32'])
def test_only_bounded_trials(label,monkeypatch):
    monkeypatch.setattr(m,'load',lambda:pytest.fail('must reject before I/O'))
    with pytest.raises(ValueError,match='bounded'):m.evaluate(label)

def test_sampler_existing_intent_is_not_restarted(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'OUT',tmp_path)
    m.immutable(tmp_path/'launch-intent.json',{'original':True},new=True)
    monkeypatch.setattr(m.b.u,'authority',lambda **kw:None)
    monkeypatch.setattr(m,'verify',lambda _:None)
    sampler=m.Sampler({'sampler_output':str(tmp_path)},SimpleNamespace(fleet={}))
    monkeypatch.setattr(sampler,'healthy',lambda:{'actual':'health'})
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('duplicate sampler'))
    assert sampler.start()=={'actual':'health'}

def test_no_gpu_or_allocation_mutation_in_telemetry_recovery():
    source=PATH.read_text()
    assert 'run.dispatch_one(label,trial)' in source
    for forbidden in ('run.execute()', 'release_allocation', 'kill(', 'docker stop', 'terminate()'):
        assert forbidden not in source
    assert "status['sample_count']!=0" in source
    assert "s.identity(old['pid']) is not None" in source
    assert "if SOCKET.exists() or SOCKET.is_symlink()" in source
    assert 'ControlPersist=no' in source and 'ClearAllForwardings=yes' in source
