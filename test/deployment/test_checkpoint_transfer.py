import hashlib
import json
import subprocess

import pytest

from pipeline.deployment import checkpoint_transfer as module


@pytest.fixture
def files(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'CHUNK_BYTES',65536)
    source=tmp_path/'source.pt';source.write_bytes(bytes(range(251))*1044)
    sha=hashlib.sha256(source.read_bytes()).hexdigest();target=tmp_path/'destination'/(sha+'.pt')
    def request(argv,data,timeout):
        result=subprocess.run(argv,input=data,stdout=subprocess.PIPE,check=True,timeout=timeout)
        return json.loads(result.stdout)
    return source,target,sha,request


def test_checkpoint_chunked_publish_and_existing_target_skips_payload(files):
    source,target,sha,request=files
    assert module.transfer(source,target,sha,request)['complete']
    assert target.read_bytes()==source.read_bytes() and not target.with_suffix('.part').exists()
    calls=[]
    def recording(argv,data,timeout):calls.append(argv[6]);return request(argv,data,timeout)
    module.transfer(source,target,sha,recording)
    assert calls==['status']


@pytest.mark.parametrize('lost_action',['chunk','finalize'])
def test_lost_ack_resumes_without_duplicate_bytes(files,lost_action):
    source,target,sha,request=files;lost=[False]
    def interrupted(argv,data,timeout):
        result=request(argv,data,timeout)
        if argv[6]==lost_action and not lost[0]:lost[0]=True;raise ConnectionError('test-only lost ack')
        return result
    result=module.transfer(source,target,sha,interrupted)
    assert result['transport_recoveries']==1 and target.read_bytes()==source.read_bytes()


def test_resume_existing_non_chunk_aligned_partial(files):
    source,target,sha,request=files;target.parent.mkdir()
    target.with_suffix('.part').write_bytes(source.read_bytes()[:12345])
    module.transfer(source,target,sha,request)
    assert target.read_bytes()==source.read_bytes()


def test_corrupt_partial_is_preserved_and_not_published(files):
    source,target,sha,request=files;target.parent.mkdir();partial=target.with_suffix('.part')
    partial.write_bytes(b'not source prefix')
    with pytest.raises(module.CheckpointTransferError,match='prefix'):module.transfer(source,target,sha,request)
    assert not target.exists() and partial.read_bytes()==b'not source prefix'


def test_truncated_chunk_never_appends(files):
    source,target,sha,request=files;data=source.read_bytes()[:65536]
    argv=['python3','-c',module.REMOTE,str(target),sha,str(source.stat().st_size),'chunk','0',str(len(data)),hashlib.sha256(data).hexdigest()]
    assert request(argv,data[:1024],30)['ok'] is False
    assert not target.exists() and not target.with_suffix('.part').exists()


def test_retry_and_wall_budgets_are_bounded(files):
    source,target,sha,request=files;calls=[]
    def offline(*args):calls.append(1);raise ConnectionError('test-only transport outage')
    with pytest.raises(module.CheckpointTransferError,match='retry budget'):
        module.transfer(source,target,sha,offline,retry_limit=6,sleep=lambda _:None)
    assert len(calls)==6
    times=iter([0,1000])
    with pytest.raises(module.CheckpointTransferError,match='deadline'):
        module.transfer(source,target,sha,request,budget_seconds=60,clock=lambda:next(times))


def test_recoverable_outage_longer_than_six_attempts_does_not_replay_bytes(files):
    source,target,sha,request=files;calls=[0];delays=[]
    def transient(argv,data,timeout):
        calls[0]+=1
        if calls[0]<=8:raise ConnectionError('test-only outage')
        return request(argv,data,timeout)
    result=module.transfer(source,target,sha,transient,sleep=delays.append)
    assert result['transport_recoveries']==8 and target.read_bytes()==source.read_bytes()
    assert len(delays)==8 and max(delays)<=5


def test_symlink_and_wrong_source_rejected(files):
    source,target,sha,request=files;target.parent.mkdir();target.symlink_to(source)
    with pytest.raises(module.CheckpointTransferError):module.transfer(source,target,sha,request)
    with pytest.raises(module.CheckpointTransferError,match='source digest'):module.transfer(source,target,'f'*64,request)
