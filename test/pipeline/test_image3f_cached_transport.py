"""CPU-only fault injection: recover COMPLETE bytes, never recompute a render."""
from test.deployment.source_paths import PIPELINE_ROOT
import ast
import hashlib
import pickle
import socket
import struct
import threading
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.dagger.image3f_transport import CompleteReplyCache, ReplyRecoveryError, SCHEMA
from pipeline.dagger.services.render_student import _Image3FLoopbackClient, Image3FConnectionClosed


def setup_cache(**kwargs):
    cache=CompleteReplyCache(**kwargs)
    channel,connection=uuid.uuid4().hex,uuid.uuid4().hex
    cache.open_channel(channel,connection)
    payload=b'exact original controls, prompt, anchor and seed'
    binding=dict(schema=SCHEMA,server_instance=cache.instance,channel=channel,
                 sequence=0,payload_sha256=hashlib.sha256(payload).hexdigest())
    return cache,connection,payload,binding


def test_complete_reply_is_recovered_without_generation_and_rebinds_next_request():
    cache,connection,payload,binding=setup_cache()
    generate=Mock(return_value={'ok':True,'views':{'all':b'original RGB bytes'}})
    completed=cache.execute(binding,payload,connection,generate)
    new_connection=uuid.uuid4().hex
    recovered=cache.execute(binding,payload,new_connection,generate)
    assert recovered is completed
    assert recovered['views']['all']==b'original RGB bytes'
    assert generate.call_count==1
    assert recovered['transport_receipt']=={**binding,'state':'complete'}
    cache.execute({**binding,'sequence':1},payload,new_connection,generate)
    assert generate.call_count==2


@pytest.mark.parametrize('fault',['changed','restart','unseen','stale','evicted','failed'])
def test_ambiguous_or_unrecoverable_work_never_recomputes(fault):
    cache,connection,payload,binding=setup_cache(maximum_replies=1)
    generate=Mock(return_value={'ok':True,'views':{'all':b'RGB'}})
    if fault=='failed':
        generate.side_effect=RuntimeError('GPU failure')
        with pytest.raises(RuntimeError,match='GPU failure'):
            cache.execute(binding,payload,connection,generate)
    elif fault!='unseen':
        cache.execute(binding,payload,connection,generate)
    if fault=='changed':
        payload=b'changed controls'
        binding={**binding,'payload_sha256':hashlib.sha256(payload).hexdigest()}
    elif fault=='restart':
        cache=CompleteReplyCache()
    elif fault=='stale':
        cache.execute({**binding,'sequence':1},payload,connection,generate)
    elif fault=='evicted':
        channel=uuid.uuid4().hex
        cache.open_channel(channel,connection)
        cache.execute({**binding,'channel':channel},payload,connection,generate)
    before=generate.call_count
    with pytest.raises(ReplyRecoveryError):
        cache.execute(binding,payload,uuid.uuid4().hex,generate)
    assert generate.call_count==before


def test_inflight_duplicate_refused_without_waiting_or_second_render():
    cache,connection,payload,binding=setup_cache()
    def generate():
        with pytest.raises(ReplyRecoveryError,match='in-flight'):
            cache.execute(binding,payload,uuid.uuid4().hex,Mock())
        return {'ok':True,'views':{}}
    cache.execute(binding,payload,connection,generate)


def test_admission_history_is_not_evicted_and_used_channel_cannot_reopen():
    cache,connection,payload,binding=setup_cache(maximum_channels=1)
    cache.execute(binding,payload,connection,lambda:{'ok':True})
    with pytest.raises(ReplyRecoveryError,match='cannot be reopened'):
        cache.open_channel(binding['channel'],uuid.uuid4().hex)
    with pytest.raises(ReplyRecoveryError,match='capacity'):
        cache.open_channel(uuid.uuid4().hex,uuid.uuid4().hex)
    assert len(cache.channels)==1


class Clock:
    def __init__(self): self.now=0.;self.sleeps=[]
    def __call__(self): return self.now
    def sleep(self,seconds): self.sleeps.append(seconds);self.now+=seconds


def client_and_clock(timeout=75.):
    client=_Image3FLoopbackClient('127.0.0.1',1,transport_recovery=SCHEMA,recovery_timeout_s=timeout)
    clock=Clock()
    client._recovery_clock,client._recovery_sleep=clock,clock.sleep
    return client,clock


def test_reconnect_after_ten_second_forward_restart_returns_cached_identical_bytes():
    client,clock=client_and_clock()
    cache=CompleteReplyCache()
    connection=uuid.uuid4().hex
    wires=[]
    generated=Mock(return_value={'ok':True,'views':{'all':b'RGB'}})
    def exchange(wire,**kwargs):
        nonlocal connection
        request=pickle.loads(wire)
        if request['cmd']=='transport_open':
            return {'ok':True,'transport':cache.open_channel(request['channel'],connection)}
        wires.append(wire)
        if len(wires)==1:
            cache.execute(request['binding'],request['payload'],connection,generated)
            connection=uuid.uuid4().hex
            raise Image3FConnectionClosed('reply lost after COMPLETE')
        assert kwargs['deadline']==75.
        if clock.now<12:
            raise ConnectionRefusedError('forward is restarting')
        return cache.execute(request['binding'],request['payload'],connection,generated)
    client._exchange=exchange
    result=client._render_exchange(b'original serialized render')
    assert 12<=clock.now<75
    assert len(wires)>3 and len(set(wires))==1
    assert generated.call_count==1 and result['views']['all']==b'RGB'
    assert client.transport_receipt()['cached_replies_recovered']==1
    assert client.transport_receipt()['logical_requests_completed']==1
    assert not client.transport_receipt()['failed']


def test_recovery_deadline_poisoned_channel_forbids_later_requests():
    client,clock=client_and_clock(timeout=5)
    exchanges=[]
    def exchange(wire,**kwargs):
        request=pickle.loads(wire);exchanges.append(request)
        if request['cmd']=='transport_open':
            return {'ok':True,'transport':dict(schema=SCHEMA,server_instance='a'*32,channel=client._channel)}
        raise ConnectionRefusedError('offline')
    client._exchange=exchange
    with pytest.raises(TimeoutError,match='deadline'):
        client._render_exchange(b'original')
    count=len(exchanges)
    assert clock.now==5 and client.transport_receipt()['failed']
    with pytest.raises(RuntimeError,match='failed closed'):
        client._render_exchange(b'new work forbidden')
    assert len(exchanges)==count


@pytest.mark.parametrize('opened',[[],{'ok':True,'transport':[]},{'ok':True,'transport':None}])
def test_malformed_transport_admission_has_clear_error(opened):
    client,_=client_and_clock()
    client._exchange=Mock(return_value=opened)
    with pytest.raises(RuntimeError,match='was not admitted'):
        client._render_exchange(b'original')
    assert client._exchange.call_count==1 and client.transport_receipt()['failed']


@pytest.mark.parametrize('response',[{'ok':False,'error':'inflight'}, {'ok':True}, []])
def test_semantic_errors_never_retry(response):
    client,clock=client_and_clock()
    client._exchange=Mock(side_effect=[{'ok':True,'transport':dict(schema=SCHEMA,
        server_instance='a'*32,channel=client._channel)},response])
    with pytest.raises(RuntimeError,match='COMPLETE reply'):
        client._render_exchange(b'original')
    assert client._exchange.call_count==2 and clock.sleeps==[]


def test_legacy_default_uses_original_wire_without_admission_or_retry():
    client=_Image3FLoopbackClient('127.0.0.1',1)
    client._exchange=Mock(side_effect=Image3FConnectionClosed('legacy failure'))
    with pytest.raises(Image3FConnectionClosed): client._render_exchange(b'original')
    client._exchange.assert_called_once_with(b'original',max_response_bytes=16*1024*1024)


def test_absolute_deadline_applies_across_connect_send_and_each_recv(monkeypatch):
    client,clock=client_and_clock()
    timeouts=[]
    response=pickle.dumps({'ok':True})
    class Stream:
        def __init__(self): self.parts=[struct.pack('!Q',len(response)),response];self.closed=False
        def settimeout(self,value): self.timeout=value;timeouts.append(value)
        def setsockopt(self,*args): pass
        def close(self): self.closed=True
        def spend(self):
            elapsed=min(2.,self.timeout);clock.now+=elapsed
            if elapsed<2:raise TimeoutError('bounded blocking I/O')
        def sendall(self,value): self.spend()
        def recv(self,count): self.spend();return self.parts.pop(0)
    stream=Stream()
    def connect(address,timeout):
        assert timeout==5
        clock.now+=2
        return stream
    monkeypatch.setattr(socket,'create_connection',connect)
    with pytest.raises(TimeoutError):
        client._exchange(b'original',max_response_bytes=4096,deadline=5.)
    assert clock.now==5 and stream.closed and client._stream is None
    assert timeouts==[5.,3.,1.]


def test_fragmented_recv_recomputes_deadline_for_each_fragment():
    clock=Clock();timeouts=[]
    class Stream:
        def settimeout(self,value): timeouts.append(value)
        def recv(self,count): clock.now+=1.;return b'x'
    with pytest.raises(TimeoutError,match='deadline'):
        _Image3FLoopbackClient._recv_exact(Stream(),4,deadline=3.,clock=clock,timeout=120.)
    assert timeouts==[3.,2.,1.] and clock.now==3.


def test_exact_server_handler_publishes_reply_before_send_and_recovers_socket_loss():
    """Run real framed server handler/client over socketpair, without model imports."""
    source=PIPELINE_ROOT/'dagger/image3f_server.py'
    names={'_recv_exact','_recv_msg','_send_msg','handle_conn'}
    nodes=[node for node in ast.parse(source.read_text()).body
           if isinstance(node,ast.FunctionDef) and node.name in names]
    generated=Mock(return_value={'ok':True,'views':{'all':b'exact RGB'}})
    scope=dict(pickle=pickle,struct=struct,uuid=uuid,MAX_TCP_REQUEST_BYTES=64*1024*1024,
               render_frame_response=lambda *args:generated(),print=lambda *args,**kwargs:None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    original_send=scope['_send_msg'];dropped=[False];cache=CompleteReplyCache()
    def send(sock,value):
        if value.get('transport_receipt') and not dropped[0]:
            dropped[0]=True
            assert len(cache.replies)==1 and generated.call_count==1
            sock.close()
            raise ConnectionError('injected reply loss')
        if sock.fileno()<0: raise ConnectionError('already closed')
        original_send(sock,value)
    scope['_send_msg']=send
    client,clock=client_and_clock()
    threads=[]
    def connection(**kwargs):
        if client._stream is None:
            client._stream,server=socket.socketpair()
            client._stream.settimeout(2)
            thread=threading.Thread(target=scope['handle_conn'],args=(server,'cpu-test',None,
                threading.Lock(),cache),daemon=True)
            threads.append(thread);thread.start()
        return client._stream
    client._connection=connection
    payload=pickle.dumps({'cmd':'render_frame','seed':42})
    try:
        result=client._render_exchange(payload)
        assert result['views']['all']==b'exact RGB' and generated.call_count==1
        assert len(threads)==2 and client.transport_receipt()['cached_replies_recovered']==1
    finally:
        client._close()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
