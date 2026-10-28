"""Bounded management metadata helpers; no CUDA, simulator, or payload loading."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

MAX_METADATA = 32 << 20

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA:
        raise ValueError('bounded regular metadata required: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(MAX_METADATA + 1)
    if len(raw) > MAX_METADATA:
        raise ValueError('metadata grew beyond limit')
    return raw

def binding(path):
    path = Path(path).absolute()
    return dict(path=str(path), raw_sha256=digest(read(path)))

def bound(value):
    if (set(value) != {'path', 'raw_sha256'} or not Path(value['path']).is_absolute()
            or not re.fullmatch('[0-9a-f]{64}', value['raw_sha256'])):
        raise ValueError('absolute raw metadata binding required')
    raw = read(value['path'])
    if digest(raw) != value['raw_sha256']:
        raise ValueError('bound metadata changed: ' + value['path'])
    return json.loads(raw)

def publish(path, value):
    """Atomic no-clobber publication; never rewrites an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.handoff-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return binding(path)

def cli_binding(pair):
    return dict(path=str(Path(pair[0]).absolute()), raw_sha256=pair[1])
