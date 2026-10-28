"""CPU-only scope, source admission and resume tests for overnight adapter."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'live_checks.py'
if not PATH.is_file():pytest.skip('operational adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('parity_live_tests',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('label',['final','full295','round1',None,'b32'])
def test_unknown_or_full_benchmark_never_enters_context(label,monkeypatch):
    monkeypatch.setattr(m,'load',lambda:pytest.fail('must reject before I/O'))
    with pytest.raises(ValueError,match='bounded'):m.evaluate(label)

@pytest.mark.parametrize('mutation',[
    lambda p:p.update(schema='retired'),lambda p:p.update(production_promoted=True),
    lambda p:p.update(compute_release_authorized=True),lambda p:p.update(online_api_readback=False),
    lambda p:p['checkpoint'].update(raw_sha256='a'*64),lambda p:p.update(source_sha256='b'*64),
])
def test_parent_rejects_wrong_or_retired_identity_before_torch(mutation,monkeypatch):
    ref=m.binding(m.FRESH/'fresh-parent-evidence-v2.json');proof=deepcopy(m.bound(ref));mutation(proof)
    monkeypatch.setattr(m,'bound',lambda _:proof)
    with pytest.raises(ValueError,match='fresh'):m.parent_admission(None)

def test_existing_sampler_intent_only_checks_health_without_respawn(tmp_path,monkeypatch):
    monkeypatch.setattr(m.u,'authority',lambda **_:None)
    m.immutable(tmp_path/'launch-intent.json',{'existing':True},new=True)
    sampler=m.Sampler({'sampler_output':str(tmp_path)},None)
    monkeypatch.setattr(sampler,'healthy',lambda:{'actual':'health'})
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate sampler'))
    assert sampler.start()=={'actual':'health'}

def test_uncertain_online_write_reconciles_without_logging_again(tmp_path,monkeypatch):
    label='b4';(tmp_path/label).mkdir()
    result={'successes':1,'success_rate':.125,'completed_episodes':8}
    plan={'parent':{'checkpoint':{'raw_sha256':'c'*64}}}
    expected={'qualification/successes':1,'qualification/success_rate':.125,'qualification/episodes':8,
        'qualification/optimizer_steps':0,'qualification/only':True,
        'qualification/source_sha256':m.u.SOURCE,'qualification/checkpoint_sha256':'c'*64}
    m.immutable(tmp_path/label/'online-intent.json',{'run_id':m.PREFIX+'-'+label,'expected':expected},new=True)
    calls=[]
    c=SimpleNamespace(paired=SimpleNamespace(bound=lambda *a,**kw:result),
        deps=SimpleNamespace(q=SimpleNamespace(admit_logging=lambda *a:('api',{}),
            verify_remote_summary=lambda *a:calls.append(a))))
    m.log_result(label,{'result':{}},plan,c,tmp_path)
    assert len(calls)==1
    assert m.read(tmp_path/label/'online-readback.json')['online_readback_verified'] is True

def test_production_run_uses_exact_coordinator_only():
    source=PATH.read_text()
    assert 'run.dispatch_one(label,trial)' in source and 'run.execute()' not in source
    assert 'release_allocation' not in source and 'kill(' not in source
