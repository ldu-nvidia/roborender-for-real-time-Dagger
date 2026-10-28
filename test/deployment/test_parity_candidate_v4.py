"""Runtime candidate adapter guards; no GPU/SSH/simulator operations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'candidate_v4.py'
if not PATH.is_file():pytest.skip('runtime candidate adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('candidate_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_candidate_has_no_training_or_lifecycle_operations():
    source=PATH.read_text()
    assert m.LABELS==('b4','b8','b16','b16-repeat','tail7')
    assert 'driver.dispatch_one(label,trial)' in source
    assert 'sampler.stop_idle()' in source
    assert 'shutdown(' not in source and 'allocation(' not in source and 'optimizer.step(' not in source
    assert "arm='dagger'" in source

@pytest.mark.parametrize('bad',[
    {},dict(status='COMPLETE'),dict(schema=m.c4.SCHEMA+'.result',status='DIAGNOSTIC_COMPLETE',
                                  plan={},optimizer_updates=100),
])
def test_no_candidate_without_actual_completed_four_update_result(monkeypatch,bad):
    monkeypatch.setattr(m.c4,'load',lambda:({},None,None,None))
    monkeypatch.setattr(m,'read',lambda _:bad)
    monkeypatch.setattr(m,'binding',lambda _: {})
    with pytest.raises(ValueError,match='completed current four-update'):m.initial_candidate()

def test_prepare_cannot_replace_existing_plan_or_run(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'PLAN',tmp_path/'plan.json');m.PLAN.write_text('{}')
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m,'initial_candidate',lambda:pytest.fail('must reject existing namespace first'))
    with pytest.raises(ValueError,match='fresh candidate'):m.prepare()

def test_online_uncertain_write_only_reads_back(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'OUT',tmp_path)
    row=dict(successes=1,success_rate=.125,completed_episodes=8)
    expected={'qualification/successes':1,'qualification/success_rate':.125,'qualification/episodes':8,
              'qualification/optimizer_steps':0,'qualification/only':True,
              'qualification/source_sha256':m.u.SOURCE,'qualification/checkpoint_sha256':'a'*64}
    m.immutable(tmp_path/'b4/online-intent.json',dict(run_id=m.RUN+'-b4',expected=expected),new=True)
    seen=[]
    c=SimpleNamespace(paired=SimpleNamespace(bound=lambda *_a,**_k:row),deps=SimpleNamespace(q=SimpleNamespace(
        admit_logging=lambda *_:(None,None),verify_remote_summary=lambda *a:seen.append(a))))
    m.log_result('b4',dict(result={}),dict(checkpoint_sha256='a'*64),c)
    assert len(seen)==1 and m.read(tmp_path/'b4/online-readback.json')['online_readback_verified']
