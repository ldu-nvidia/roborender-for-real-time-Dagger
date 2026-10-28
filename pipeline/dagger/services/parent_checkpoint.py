"""Content-addressed parent import for the registry's restricted source root.

The model is not rewritten or promoted. Shared run artifacts may live outside
the registry roots after storage reorganization; import their verified bytes
before registration instead of broadening the registry's path permissions.
"""
from pathlib import Path
import hashlib
import os
import re
import tempfile

def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def stage_parent(source, expected_sha256, destination_root):
    source,root=Path(source),Path(destination_root)
    if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
        raise ValueError('exact parent checkpoint digest required')
    if (not source.is_absolute() or not root.is_absolute() or '..' in root.parts
            or len(root.parts)<4 or root.is_symlink()):
        raise ValueError('explicit scoped parent import paths required')
    if not source.is_file() or source.is_symlink() or digest_file(source)!=expected_sha256:
        raise ValueError('parent checkpoint bytes differ')
    root.mkdir(parents=True,exist_ok=True)
    if root.resolve()!=root:raise ValueError('parent import root must not alias another directory')
    target=root/(expected_sha256+'.pt')
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or digest_file(target)!=expected_sha256:
            raise ValueError('existing parent import differs; do not overwrite')
        return target
    fd,name=tempfile.mkstemp(prefix='.parent-',suffix='.partial',dir=root)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as output,source.open('rb') as incoming:
            for block in iter(lambda:incoming.read(1<<20),b''):output.write(block)
            output.flush();os.fsync(output.fileno())
        if digest_file(temporary)!=expected_sha256:raise ValueError('parent changed during import')
        os.chmod(temporary,0o444)
        try:os.link(temporary,target)
        except FileExistsError:
            if target.is_symlink() or digest_file(target)!=expected_sha256:raise ValueError('concurrent parent import differs')
        directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        return target
    finally:
        temporary.unlink(missing_ok=True)
