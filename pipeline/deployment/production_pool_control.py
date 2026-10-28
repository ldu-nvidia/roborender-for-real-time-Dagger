"""Explicit host-side reader overlay for an unchanged native B2 lifetime.

This admits only the three permission-reader/reconciliation source files. It
does not replace native code, restart a worker, or waive source verification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'roborender.production-pool-control.v1'
CONTROL_PATHS = frozenset((
    'pipeline/deployment/rtx_actor/production_resident_pool.py',
    'pipeline/deployment/rtx_actor/production_guard_access.py',
    'pipeline/deployment/rtx_actor/recover_production_resident_pool.py',
))


def validate_control(value, native_sha):
    if (set(value) != {'schema', 'native_source_sha256', 'source_root', 'source_manifest',
                       'native_manifest', 'changed_files'} or value['schema'] != SCHEMA
            or value['native_source_sha256'] != native_sha):
        raise ValueError('exact permission-reader control binding required')
    root = Path(value['source_root'])
    if (not root.is_absolute() or '..' in root.parts
            or not root.is_relative_to('/home/ldu/roborender/runtime/source-releases')
            or root.name != 'frozen'):
        raise ValueError('separate frozen control source required')
    for key in ('source_manifest', 'native_manifest'):
        ref = value[key]
        if (set(ref) != {'path', 'raw_sha256'} or not Path(ref['path']).is_absolute()
                or '..' in Path(ref['path']).parts or not re.fullmatch('[0-9a-f]{64}', ref['raw_sha256'])):
            raise ValueError('exact bounded manifest binding required')
    rows = value['changed_files']
    if not isinstance(rows, list) or [r.get('path') for r in rows] != sorted(CONTROL_PATHS):
        raise ValueError('only the exact three reader/reconciliation files may change')
    for row in rows:
        if (set(row) != {'path', 'old_sha256', 'new_sha256'}
                or not re.fullmatch('[0-9a-f]{64}', row['new_sha256'])
                or row['old_sha256'] is not None and not re.fullmatch('[0-9a-f]{64}', row['old_sha256'])):
            raise ValueError('exact reviewed old/new source hashes required')
    return value


# Deliberately stdlib-only: no unverified overlay module is imported before
# every declared source file has been compared with its pinned manifests.
REMOTE_VERIFY_AND_EXEC = r'''
import hashlib,json,pathlib,stat,subprocess,sys
v=json.loads(sys.argv[1]); root=pathlib.Path(v['source_root'])
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p,limit=8<<20):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>limit:raise RuntimeError('unsafe control source')
 return p.read_bytes()
def manifest(ref):
 raw=read(ref['path']); assert hashlib.sha256(raw).hexdigest()==ref['raw_sha256']
 m=json.loads(raw); rows=m['files']
 assert set(m)=={'schema','source_sha256','file_count','files'}
 assert m['schema']=='roborender.source-snapshot.v1' and m['file_count']==len(rows)
 assert hashlib.sha256(canonical(rows)).hexdigest()==m['source_sha256']
 assert [r['path'] for r in rows]==sorted({r['path'] for r in rows})
 return m,{r['path']:r for r in rows}
native,old=manifest(v['native_manifest']); control,new=manifest(v['source_manifest'])
assert native['source_sha256']==v['native_source_sha256']
assert not old.keys()-new.keys()
changed=sorted(p for p in old.keys()|new.keys() if old.get(p)!=new.get(p))
assert changed==[r['path'] for r in v['changed_files']]
for d in v['changed_files']:
 p=d['path']; assert old.get(p,{}).get('sha256')==d['old_sha256'] and new[p]['sha256']==d['new_sha256']
 assert new[p]['executable']==old.get(p,{}).get('executable',False)
for name,row in new.items():
 rel=pathlib.PurePosixPath(name); assert not rel.is_absolute() and '..' not in rel.parts
 p=root/name; assert not any((root/x).is_symlink() for x in (pathlib.Path(name),*pathlib.Path(name).parents))
 raw=read(p); assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 assert bool(p.stat().st_mode&0o111)==row['executable']
helper=root/'pipeline/deployment/rtx_actor/production_resident_pool.py'
raise SystemExit(subprocess.run([sys.executable,str(helper),*sys.argv[2:]]).returncode)
'''
