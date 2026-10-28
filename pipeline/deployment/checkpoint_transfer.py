"""Bounded, content-addressed checkpoint transfer; never publishes partial bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

CHUNK_BYTES = 8 * 1024 * 1024
REMOTE = r'''
import fcntl,hashlib,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);expected=sys.argv[2];size=int(sys.argv[3]);action=sys.argv[4]
assert p.name==expected+'.pt' and len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
p.parent.mkdir(parents=True,exist_ok=True)
q=p.with_suffix('.part')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
try:
 assert not p.is_symlink() and not q.is_symlink() and not p.with_suffix('.transfer.lock').is_symlink()
 with p.with_suffix('.transfer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if p.exists():
   assert p.stat().st_size==size and sha(p)==expected
   result={'complete':True,'offset':size,'sha256':expected}
  elif action=='status':
   offset=q.stat().st_size if q.exists() else 0
   assert 0<=offset<=size
   result={'complete':False,'offset':offset,'sha256':sha(q) if q.exists() else hashlib.sha256(b'').hexdigest()}
  elif action=='chunk':
   offset=int(sys.argv[5]);length=int(sys.argv[6]);digest=sys.argv[7]
   assert 0<length<=8388608 and 0<=offset<offset+length<=size
   data=sys.stdin.buffer.read(length+1)
   assert len(data)==length and hashlib.sha256(data).hexdigest()==digest
   current=q.stat().st_size if q.exists() else 0
   if current==offset+length:
    with q.open('rb') as f:f.seek(offset);assert hashlib.sha256(f.read(length)).hexdigest()==digest
   else:
    assert current==offset
    with q.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
   result={'complete':False,'offset':offset+length,'sha256':digest}
  elif action=='finalize':
   assert q.stat().st_size==size and sha(q)==expected
   q.replace(p)
   fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
   result={'complete':True,'offset':size,'sha256':expected}
  else:raise AssertionError('unsupported action')
 print(json.dumps({'ok':True,**result}))
except (AssertionError,OSError,ValueError) as exc:
 print(json.dumps({'ok':False,'error_type':type(exc).__name__}))
'''


class CheckpointTransferError(RuntimeError):
    pass


def transfer(source, destination, digest, request, *, budget_seconds=360, clock=time.monotonic,
             retry_limit=48, sleep=time.sleep):
    """request(argv, data, timeout) returns parsed JSON; transport retries are safe.

    Each append is a verified, fsynced, idempotent chunk. An interrupted SSH
    stream never appends a partial chunk; a lost response resumes from durable
    size plus verified source-prefix hash. No policies/episodes are executed.
    """
    if type(retry_limit) is not int or not 1<=retry_limit<=128 or not 0<budget_seconds<=3600:
        raise ValueError('bounded checkpoint transport retry policy required')
    source=Path(source);size=source.stat().st_size
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
            raise CheckpointTransferError('checkpoint source digest differs')
    deadline=clock()+budget_seconds
    def call(action, data=None, *args):
        if clock()>=deadline:raise CheckpointTransferError('checkpoint transfer deadline exceeded')
        result=request(['python3','-c',REMOTE,str(destination),digest,str(size),action,*map(str,args)],
                       data,min(60,max(1,deadline-clock())))
        if not isinstance(result,dict) or result.get('ok') is not True:
            raise CheckpointTransferError('checkpoint receiver integrity/ownership check failed')
        return result
    failures=0
    while clock()<deadline:
        try:
            status=call('status')
            offset=status['offset']
            if type(offset) is not int or not 0<=offset<=size:
                raise CheckpointTransferError('checkpoint receiver offset differs')
            if status['complete']:
                if offset!=size or status['sha256']!=digest:raise CheckpointTransferError('published checkpoint differs')
                return {'complete':True,'bytes':size,'sha256':digest,'transport_recoveries':failures}
            prefix=hashlib.sha256()
            with source.open('rb') as stream:
                remaining=offset
                while remaining:
                    block=stream.read(min(1048576,remaining));prefix.update(block);remaining-=len(block)
                if prefix.hexdigest()!=status['sha256']:
                    raise CheckpointTransferError('retained partial checkpoint differs from source prefix')
                while offset<size:
                    block=stream.read(CHUNK_BYTES)
                    chunk_sha=hashlib.sha256(block).hexdigest()
                    result=call('chunk',block,offset,len(block),chunk_sha)
                    if result['offset']!=offset+len(block) or result['sha256']!=chunk_sha:
                        raise CheckpointTransferError('checkpoint chunk acknowledgement differs')
                    offset+=len(block)
            result=call('finalize')
            if result.get('complete') is not True or result['offset']!=size or result['sha256']!=digest:
                raise CheckpointTransferError('checkpoint publication acknowledgement differs')
            return {'complete':True,'bytes':size,'sha256':digest,'transport_recoveries':failures}
        except (ConnectionError,TimeoutError,subprocess.TimeoutExpired):
            failures+=1
            if failures>=retry_limit:raise CheckpointTransferError('checkpoint transport retry budget exhausted') from None
            # Retry only this idempotent byte transfer, never native work.
            # Six rapid failures previously abandoned a recoverable tunnel
            # before its supervisor's reconnect grace had elapsed.
            sleep(min(5., float(failures), max(0.,deadline-clock())))
    raise CheckpointTransferError('checkpoint transfer deadline exceeded')
