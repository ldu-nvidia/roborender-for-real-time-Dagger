"""Batch exact small native JSON receipts; never read image/trajectory payloads.

The caller still validates every assignment/reset binding. This only replaces
one SSH command per file with one bounded read per group of64 files.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path

BATCH_FILES = 64
MAX_FILE_BYTES = 65535
MAX_RESPONSE_BYTES = 6 << 20

# Standalone stdlib reader sent through the existing pinned SSH transport.
# Open relative to one directory FD, reject links/nonregular/changed files,
# and cap every read before constructing the response.
READER = r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
def require(value):
 if not value:raise ValueError('native metadata read rejected')
request=json.loads(sys.stdin.buffer.read(32769))
require(isinstance(request,dict) and set(request)=={'directory','names'})
directory=Path(request['directory']);names=request['names']
require(directory.is_absolute() and not directory.is_symlink())
require(isinstance(names,list) and 0<len(names)<=64)
require(all(isinstance(n,str) and Path(n).name==n and n.endswith('.json') and len(n)<=255 for n in names))
require(len(set(names))==len(names))
rows=[]
directory_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 for name in names:
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
  with os.fdopen(fd,'rb') as stream:
   before=os.fstat(stream.fileno())
   require(stat.S_ISREG(before.st_mode) and before.st_size<=65535)
   raw=stream.read(65536);after=os.fstat(stream.fileno())
   fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
   require(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k) for k in fields))
   require(isinstance(json.loads(raw),dict))
  rows.append(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),data_b64=base64.b64encode(raw).decode()))
finally:os.close(directory_fd)
sys.stdout.write(json.dumps(rows,separators=(',',':'),allow_nan=False))
'''


def fetch_native_json(site, directory, names):
    """Return byte-identical receipts, in bounded batches; no publication yet."""
    directory = Path(directory)
    if not directory.is_absolute() or '..' in directory.parts or len(str(directory)) > 4096:
        raise ValueError('exact absolute native metadata directory required')
    names = list(names)
    if any(not isinstance(n, str) or Path(n).name != n or not n.endswith('.json')
           or len(n) > 255 for n in names):
        raise ValueError('only exact native JSON basenames may be transferred')
    names = list(dict.fromkeys(names))
    result = {}
    for offset in range(0, len(names), BATCH_FILES):
        selected = names[offset:offset+BATCH_FILES]
        request = json.dumps(dict(directory=str(directory), names=selected), separators=(',', ':')).encode()
        if len(request) > 32768:
            raise ValueError('native metadata request exceeds bound')
        response = site.command(['python3', '-c', READER], 'beta', data=request)
        if not isinstance(response, bytes) or len(response) > MAX_RESPONSE_BYTES:
            raise ValueError('native metadata response exceeds bound')
        rows = json.loads(response)
        if not isinstance(rows, list) or len(rows) != len(selected):
            raise ValueError('native metadata response cardinality differs')
        for name, row in zip(selected, rows):
            if not isinstance(row, dict) or set(row) != {'name', 'sha256', 'data_b64'} or row['name'] != name:
                raise ValueError('native metadata response identity/order differs')
            raw = base64.b64decode(row['data_b64'], validate=True)
            if len(raw) > MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('native metadata response bytes changed')
            if not isinstance(json.loads(raw), dict):
                raise ValueError('native metadata must be a JSON object')
            result[name] = raw
    return result
