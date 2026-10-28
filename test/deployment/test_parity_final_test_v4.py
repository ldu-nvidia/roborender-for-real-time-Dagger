"""Read-only/lineage guards for the final warm-cycle target test."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'final_test_v4.py'
if not PATH.is_file():pytest.skip('runtime final-test adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('final_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_only_actual_target_and_tail_not_new_full_ladder_or_test295():
    assert m.LABELS==('b16','tail7')
    source=PATH.read_text()
    assert 'driver.dispatch_one(label,trial[\'plan\'])' in source
    assert 'reference.validate_trial(p,context,trial,actual)' in source
    assert 'fresh_completed_episodes=39' in source
    assert 'sampler.stop_idle()' in source
    assert 'shutdown(' not in source and 'optimizer.step(' not in source and 'allocation(' not in source

@pytest.mark.parametrize('bad',[{},dict(status='COMPLETE'),dict(schema=m.ret.SCHEMA+'.result',status='DIAGNOSTIC_COMPLETE',plan={},optimizer_updates=100)])
def test_no_final_test_without_actual_completed_return(monkeypatch,bad):
    monkeypatch.setattr(m.ret,'load',lambda:({'execution':{}},None))
    monkeypatch.setattr(m,'read',lambda _:bad)
    monkeypatch.setattr(m,'binding',lambda _: {})
    with pytest.raises(ValueError,match='actual complete same-owner'):m.predecessor()

def test_existing_namespace_never_replaced(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'PLAN',tmp_path/'plan.json');m.PLAN.write_text('{}')
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m,'predecessor',lambda:pytest.fail('must reject existing namespace before remote actions'))
    with pytest.raises(ValueError,match='fresh final-test'):m.prepare()

def test_return_attestation_is_cpu_only_and_checks_all_adamw_rng_state():
    source=Path(m.reference.__file__).read_text()
    assert "map_location='cpu'" in source
    assert "set(states)!=set(range(78))" in source
    assert "actual_rng!=receipt['learner_rng_receipt']['output_rng_sha256']" in source
    assert "step!=10" in source

def test_mismatched_final_staging_rejects_before_gpu_dispatch(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'OUT',tmp_path)
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m,'load',lambda:({'checkpoint':{}},SimpleNamespace(c=None)))
    monkeypatch.setattr(m,'read',lambda _:dict(plan={'wrong':1},checkpoint={}))
    monkeypatch.setattr(m,'binding',lambda _: {})
    with pytest.raises(ValueError,match='actual final checkpoint staging'):m.execute()
