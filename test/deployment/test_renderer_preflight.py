"""CPU-only route admission tests; no live qualification claim."""
from copy import deepcopy
import hashlib
import json
import pickle
import struct
from types import SimpleNamespace
import pytest
from pipeline.deployment import renderer_preflight as preflight
from pipeline.deployment import paired_scaled_runtime as runtime
from pipeline.deployment import scaled_success_dispatch as dispatch
from test.pipeline.test_scaled_success_qualification import evidence, accepted


def answer(renderer, endpoint):
    identity=renderer['identity']
    return dict(ok=True,attestation=dict(schema='roborender.image3f.observation-attestation.v1',
        contract=dict(schema='roborender.dagger.student-observation.v1',mode='generated_only',renderer='Image3F',
            renderer_steps=identity['steps'],renderer_cfg=identity['cfg'],teacache_threshold=identity['teacache_threshold'],raytraced_policy_fallback=False),
        backend_class='runtime.image3f_deploy.Image3FSession',generate_source_sha256=endpoint['generate_source_sha256']))


def test_all_eight_routes_checked_without_render_or_native_work(evidence):
    renderer=evidence['config']['renderer'];calls=[]
    def exchange(endpoint):calls.append(endpoint['slot']);return answer(renderer,endpoint)
    result=preflight.check_renderer_routes(renderer,exchange=exchange)
    assert sorted(calls)==list(range(8)) and result['verified_slots']==list(range(8))
    assert result['native_steps']==result['render_requests']==0


@pytest.mark.parametrize('fault',['absent','source','native_rgb','steps','backend','extra'])
def test_any_endpoint_failure_rejects_entire_admission(evidence,fault):
    renderer=evidence['config']['renderer']
    def exchange(endpoint):
        row=answer(renderer,endpoint)
        if endpoint['slot']==7:
            if fault=='absent':raise ConnectionRefusedError('route unavailable')
            if fault=='source':row['attestation']['generate_source_sha256']='e'*64
            if fault=='native_rgb':row['attestation']['contract']['raytraced_policy_fallback']=True
            if fault=='steps':row['attestation']['contract']['renderer_steps']=50
            if fault=='backend':row['attestation']['backend_class']='fake'
            if fault=='extra':row['unexpected']=True
        return row
    with pytest.raises((OSError,RuntimeError)):preflight.check_renderer_routes(renderer,exchange=exchange)


def test_actual_legacy_uint64_wire_and_contract_only_command():
    payload=pickle.dumps(dict(ok=True,attestation={}),protocol=pickle.HIGHEST_PROTOCOL)
    class Socket:
        def __init__(self):self.remaining=struct.pack('!Q',len(payload))+payload;self.sent=None
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def settimeout(self,n):assert n==5
        def sendall(self,raw):self.sent=raw
        def recv(self,n):out=self.remaining[:min(n,3)];self.remaining=self.remaining[len(out):];return out
    sock=Socket()
    assert preflight.observe(dict(host='127.0.0.1',port=18560),connect=lambda *a,**k:sock)==dict(ok=True,attestation={})
    assert struct.unpack('!Q',sock.sent[:8])[0]==len(sock.sent[8:])
    assert pickle.loads(sock.sent[8:])==dict(cmd='observation_contract')


@pytest.mark.parametrize('response',[b'',struct.pack('!Q',8193),struct.pack('!Q',0)])
def test_truncated_or_unbounded_metadata_fails(response):
    class Socket:
        def __init__(self):self.remaining=response
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def settimeout(self,n):pass
        def sendall(self,raw):pass
        def recv(self,n):out=self.remaining[:n];self.remaining=self.remaining[n:];return out
    with pytest.raises((ConnectionError,ValueError)):
        preflight.observe(dict(host='127.0.0.1',port=18560),connect=lambda *a,**k:Socket())


def test_missing_route_stops_before_artifact_stage_and_native_post(tmp_path,evidence):
    transport=runtime.Transport(None,{},tmp_path,None);calls=[]
    def invoke(owner,action,**kwargs):
        calls.append(action)
        raise RuntimeError('renderer route unavailable')
    transport.invoke=invoke
    value=evidence['config'];raw=dispatch.canonical(value)
    with pytest.raises(RuntimeError):transport.stage_artifacts({},[dict(value=value,path='config.json',raw_sha256=hashlib.sha256(raw).hexdigest())])
    assert calls==['renderer_preflight']


def test_remote_preflight_is_before_native_submit_and_requires_idle():
    assert "elif action=='renderer_preflight':\n assert ready['phase']=='IDLE'" in runtime.REMOTE
    assert 'result=check_renderer_routes' in runtime.REMOTE
