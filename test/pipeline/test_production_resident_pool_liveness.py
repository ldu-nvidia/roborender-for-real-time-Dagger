"""Read-only exact native-container liveness before any retrying resident RPC."""
from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace

import pytest

from test.pipeline.test_production_resident_pool import pool,pool_module
from resident_isaac_client import ResidentClient


@pytest.fixture
def owner(pool):
    worker=pool['workers'][0]
    worker['container_name']='standing-isaac-leo-prod-cpu-liveness-g0'
    row={'Id':'a'*64,'Name':'/'+worker['container_name'],
        'State':{'Running':True,'Pid':90001},
        'Config':{'Labels':{'org.roborender.qualification':worker['container_name']},
            'Env':['RESIDENT_SOURCE_SHA256='+pool['source_sha256'],
                   'CUDA_VISIBLE_DEVICES='+worker['gpu_uuid'],'RESIDENT_PHYSICAL_GPU=0','RESIDENT_PORT=18640']},
        'HostConfig':{'DeviceRequests':[{'DeviceIDs':[worker['gpu_uuid']]}]},
        'Mounts':[{'Destination':'/dagger-runtime','Type':'bind','RW':True,'Source':worker['runtime_root']}]}
    return worker,row


def inspector(row,calls):
    def run(argv,**kwargs):
        calls.append(argv)
        assert argv[:2]==['docker','inspect']
        assert kwargs=={'check':True,'capture_output':True,'timeout':15}
        return SimpleNamespace(stdout=json.dumps([row]).encode())
    return run


def test_healthy_exact_container_is_read_only(owner,pool):
    worker,row=owner;calls=[]
    result=pool_module.attest_live_owner(worker,pool['source_sha256'],run=inspector(row,calls))
    assert result=={'container_id':'a'*64,'host_pid':90001}
    assert calls==[['docker','inspect',worker['container_name']]]


@pytest.mark.parametrize('fault',['missing','stopped','wrong_name','wrong_label','wrong_source','wrong_gpu',
    'wrong_env_gpu','wrong_physical_gpu','wrong_port','wrong_mount','mount_readonly','host_pid_zero','host_pid_string','duplicate_source'])
def test_missing_changed_or_unauthorized_owner_is_terminal(owner,pool,fault):
    worker,row=owner;row=deepcopy(row);calls=[]
    if fault=='missing':
        def run(*args,**kwargs):raise subprocess.CalledProcessError(1,['docker','inspect'],stderr=b'No such object')
    else:
        if fault=='stopped':row['State']['Running']=False
        elif fault=='wrong_name':row['Name']='/different'
        elif fault=='wrong_label':row['Config']['Labels']['org.roborender.qualification']='different'
        elif fault=='wrong_source':row['Config']['Env'][0]='RESIDENT_SOURCE_SHA256='+'f'*64
        elif fault=='wrong_gpu':row['HostConfig']['DeviceRequests'][0]['DeviceIDs']=['GPU-wrong']
        elif fault=='wrong_env_gpu':row['Config']['Env'][1]='CUDA_VISIBLE_DEVICES=GPU-wrong'
        elif fault=='wrong_physical_gpu':row['Config']['Env'][2]='RESIDENT_PHYSICAL_GPU=1'
        elif fault=='wrong_port':row['Config']['Env'][3]='RESIDENT_PORT=18641'
        elif fault=='wrong_mount':row['Mounts'][0]['Source']=worker['runtime_root']+'-different'
        elif fault=='mount_readonly':row['Mounts'][0]['RW']=False
        elif fault=='host_pid_zero':row['State']['Pid']=0
        elif fault=='host_pid_string':row['State']['Pid']='90001'
        else:row['Config']['Env'].append(row['Config']['Env'][0])
        run=inspector(row,calls)
    with pytest.raises(RuntimeError,match='native owner changed or stopped'):
        pool_module.attest_live_owner(worker,pool['source_sha256'],run=run)


def test_selected_missing_owner_fails_before_any_http_ready(owner,pool,monkeypatch):
    worker,row=owner
    monkeypatch.setattr(pool_module.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(
        subprocess.CalledProcessError(1,['docker','inspect'],stderr=b'No such object')))
    class Fake(ResidentClient):
        def __init__(self,*a,**k):pass
        def _request(self,*a,**k):pytest.fail('dead native owner must fail before retrying HTTP')
    monkeypatch.setattr(pool_module,'ResidentClient',Fake)
    with pytest.raises(RuntimeError,match='native owner changed or stopped'):
        pool_module.selected_clients(pool,[0])


def test_healthy_selected_owner_ignores_unrelated_absent_container(owner,pool,monkeypatch):
    worker,row=owner;calls=[]
    monkeypatch.setattr(pool_module.subprocess,'run',inspector(row,calls))
    class Fake(ResidentClient):
        def __init__(self,*a,**k):pass
        def ready(self):return {'native_pid':worker['native_pid']}
    monkeypatch.setattr(pool_module,'ResidentClient',Fake)
    selected=pool_module.selected_clients(pool,[0])
    assert set(selected)=={0}
    assert calls==[['docker','inspect',worker['container_name']]]


def test_native_pid_is_still_exact_after_container_precheck(owner,pool,monkeypatch):
    worker,row=owner;calls=[]
    monkeypatch.setattr(pool_module.subprocess,'run',inspector(row,calls))
    class Fake(ResidentClient):
        def __init__(self,*a,**k):pass
        def ready(self):return {'native_pid':worker['native_pid']+1}
    monkeypatch.setattr(pool_module,'ResidentClient',Fake)
    with pytest.raises(RuntimeError,match='native owner changed'):
        pool_module.selected_clients(pool,[0])
