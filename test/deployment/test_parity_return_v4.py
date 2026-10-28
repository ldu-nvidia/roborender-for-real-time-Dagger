"""CPU guards for current-source warm return; no real service/native actions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]/'runtime/raven-parity-overnight-20261026-t6oMpN'
PATH=ROOT/'return_v4.py'
if not PATH.is_file():pytest.skip('runtime return adapter absent',allow_module_level=True)
spec=importlib.util.spec_from_file_location('return_v4_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_no_replacement_native_or_gpu_service_in_return():
    source=PATH.read_text()
    assert "('gpu-services','status','gpu')" in source
    assert "training_curriculum=p['training_curriculum']" in source
    assert "diagnostic_learner_chain=p['chain']['path']" in source
    assert "self.bundle=Path(e.initial_plan['campaign_root'])/'retained-services'" in source
    assert 'shutdown(' not in source and 'allocation(' not in source and "'gpu-services','start'" not in source
    assert "if (OUT/'launch-intent.json').exists():raise" in source

def test_existing_prepare_namespace_is_not_replaced(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'PLAN',tmp_path/'plan.json');m.PLAN.write_text('{}')
    monkeypatch.setattr(m,'helpers',lambda:None);monkeypatch.setattr(m.u,'authority',lambda **_:None)
    monkeypatch.setattr(m,'predecessor',lambda:pytest.fail('prior namespace must reject before validation/actions'))
    with pytest.raises(ValueError,match='fresh return'):m.prepare()

def test_route_lineage_rejects_other_run_or_ports(monkeypatch):
    e=SimpleNamespace(initial_plan=dict(campaign_root='/runtime/initial'),c=SimpleNamespace(native=dict(jobs_root='/runtime/native')),ready={})
    monkeypatch.setattr(m,'bound',lambda _:dict(services={'18610':dict(ok=True,role='coordinator',pid=23)}))
    config,status=m.route_config(e,'local')
    value=dict(upstream_config=dict(path=str(config)),upstream_status_path=str(status),
        expected_forwards=[dict(local_port=18610,remote_port=18610,expected_ready=dict(ok=True,role='coordinator',pid=23))])
    with pytest.raises(ValueError,match='route support'):m.route_lineage(value,e,'local')
    with pytest.raises(ValueError,match='only existing'):m.route_config(e,'other')

def fake_site(monkeypatch,tmp_path):
    calls=[]
    class Base:
        def __init__(self,site,request,*args,**kwargs):
            self.request=request;self.bundle=None;self.token=None;self.state_identity=None
        def process(self,*args,**kwargs):calls.append((args,kwargs));return dict(state='RUNNING')
    e=SimpleNamespace(api=SimpleNamespace(Site=Base),initial_plan=dict(campaign_root=str(tmp_path/'initial')),
        c=SimpleNamespace(native=dict(jobs_root='/runtime/native')))
    p=dict(original_token=dict(path='/runtime/initial/services.token'),initial_ready={'value':dict(state_identity='initial')},
        site={'value':{}},retained=dict(path='/runtime/retained.json',raw_sha256='a'*64),
        campaign_root=str(tmp_path/'new'),training_curriculum=[dict(mode='training0'),dict(mode='training1')])
    monkeypatch.setattr(m,'bound',lambda ref:ref['value'])
    monkeypatch.setattr(m,'binding',lambda _:dict(raw_sha256='b'*64))
    adapter=SimpleNamespace(action=lambda *a:dict(called=a))
    site=m.site_for_return(p,e,adapter,{},'/runtime/config.json','/runtime/protocol.json')
    return site,p,calls

@pytest.mark.parametrize('action',['start','stop','restart'])
def test_return_site_forbids_service_mutation(monkeypatch,tmp_path,action):
    site,p,calls=fake_site(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='only inspect'):site.process('gpu-services',action,host='gpu')
    assert not calls

def test_return_site_preserves_original_token_and_training_only_curriculum(monkeypatch,tmp_path):
    site,p,calls=fake_site(monkeypatch,tmp_path)
    assert str(site.token)==p['original_token']['path'] and site.state_identity=='initial'
    assert site.request['training_curriculum']==p['training_curriculum']
    assert site.request['run_id']==m.RUN
    assert site.process('gpu-services','status',host='gpu')==dict(state='RUNNING')
    assert len(calls)==1

def test_sampler_existing_intent_never_launches_a_duplicate(monkeypatch,tmp_path):
    sampler=object.__new__(m.Sampler);sampler.output=tmp_path;sampler.context=SimpleNamespace(fleet={})
    (tmp_path/'launch-intent.json').write_text('{}')
    monkeypatch.setattr(m.u,'authority',lambda **_:None);monkeypatch.setattr(m.v.r,'verify',lambda _:None)
    monkeypatch.setattr(sampler,'healthy',lambda:dict(state='RUNNING'))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *_a,**_k:pytest.fail('duplicate observer'))
    assert sampler.start()==dict(state='RUNNING')
