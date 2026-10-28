"""Only the selected live resident is a per-command polling dependency."""
import json
from pathlib import Path
import sys

import pytest

from test.pipeline.test_production_resident_pool import pool, pool_module
from pipeline.bc_training.handoff_common import publish
from resident_isaac_client import ResidentClient


def fake_clients(monkeypatch, pool, *, missing=1, mutation=None):
    contacted=[]
    monkeypatch.setattr(pool_module,'attest_live_owner',lambda *args,**kwargs:None)
    class Fake(ResidentClient):
        def __init__(self,port,token_file,*,source_sha256,gpu_uuid):
            self.port=port
            self.source,self.gpu=source_sha256,gpu_uuid
            self.instance=None
        def _request(self,route,body=None):
            gpu=self.port-18640
            contacted.append((gpu,route))
            if gpu==missing:
                raise ConnectionError('unrelated native owner unavailable')
            if route=='/ready':
                row=pool['workers'][gpu]
                result=dict(role='isaac_resident',schema='roborender.isaac-resident.v1',ok=True,
                    source_sha256=pool['source_sha256'],gpu_uuid=row['gpu_uuid'],native_initializations=1,
                    worker_instance_id=row['worker_instance_id'],native_pid=row['native_pid'],phase='IDLE')
                if mutation:result.update(mutation)
                return result
            if route=='/v1/commands':
                import hashlib
                return dict(status='accepted',command_id=body['command_id'],worker_instance_id=self.instance,
                    payload_sha256=hashlib.sha256(pool_module.json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
            q=json.loads(Path(pool['qualification']['path']).read_bytes())
            return dict(status='COMPLETE',receipt=q['results'][gpu]['receipt'])
    monkeypatch.setattr(pool_module,'ResidentClient',Fake)
    return contacted


@pytest.mark.parametrize('action',['submit','result','wait'])
@pytest.mark.parametrize('gpu',[0,1])
def test_per_command_never_contacts_unselected_missing_owner(pool,tmp_path,monkeypatch,capsys,action,gpu):
    contacted=fake_clients(monkeypatch,pool,missing=1-gpu)
    p=publish(tmp_path/'pool.json',pool)
    c=publish(tmp_path/'commands.json',[dict(gpu=gpu,command_id='owned',operation='evaluate_explicit_panel',payload={})])
    monkeypatch.setattr(sys,'argv',['pool','--pool',p['path'],'--action',action,'--commands',c['path']])
    pool_module.main()
    assert json.loads(capsys.readouterr().out)['results'][0]['status']==('accepted' if action=='submit' else 'COMPLETE')
    assert contacted and {row[0] for row in contacted}=={gpu}


@pytest.mark.parametrize('action',['status','evaluation-memory','training-memory'])
def test_pool_wide_status_and_memory_still_require_both_live_owners(pool,tmp_path,monkeypatch,action):
    contacted=fake_clients(monkeypatch,pool,missing=1)
    p=publish(tmp_path/'pool.json',pool)
    monkeypatch.setattr(sys,'argv',['pool','--pool',p['path'],'--action',action])
    with pytest.raises(ConnectionError,match='unrelated'):
        pool_module.main()
    assert {row[0] for row in contacted}=={0,1}


@pytest.mark.parametrize('mutation',[dict(native_pid=999),dict(worker_instance_id='changed'),
    dict(source_sha256='f'*64),dict(gpu_uuid='wrong'),dict(native_initializations=2)])
def test_selected_owner_native_source_instance_and_gpu_are_still_exact(pool,monkeypatch,mutation):
    contacted=fake_clients(monkeypatch,pool,missing=1,mutation=mutation)
    with pytest.raises(RuntimeError):pool_module.selected_clients(pool,[0])
    assert {row[0] for row in contacted}=={0}


def test_unselected_immutable_qualification_guard_is_not_skipped(pool,monkeypatch):
    contacted=fake_clients(monkeypatch,pool,missing=1)
    qualification=json.loads(Path(pool['qualification']['path']).read_bytes())
    guard_path=Path(qualification['results'][1]['receipt']['resource_guard']['path'])
    guard_path.write_bytes(b'changed guard evidence')
    with pytest.raises(RuntimeError,match='guard receipt changed'):
        pool_module.selected_clients(pool,[0])
    assert contacted==[]


@pytest.mark.parametrize('gpus',[[],[0,0],[2],[False],['0']])
def test_invalid_selection_fails_before_live_contact(pool,monkeypatch,gpus):
    contacted=fake_clients(monkeypatch,pool)
    with pytest.raises(ValueError):pool_module.selected_clients(pool,gpus)
    assert contacted==[]


def test_selected_native_pid_is_rechecked_on_every_ready_not_only_initial_poll(pool,monkeypatch):
    mutation={}
    fake_clients(monkeypatch,pool,missing=1,mutation=mutation)
    client=pool_module.selected_clients(pool,[0])[0]
    mutation['native_pid']=999
    with pytest.raises(RuntimeError,match='native owner changed'):
        client.wait('owned')
