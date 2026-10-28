"""One boot-time full-motion asset identity check for native production.

The JSON's raw identity and the inventory's canonical identity are distinct.
Assignments bind the latter; a native launch separately pins the raw file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import canonical_json, require_identifier, require_sha256


def read_inventory(path, raw_sha256, motion_root):
    path, root = Path(path), Path(motion_root)
    require_sha256(raw_sha256, 'native inventory raw file')
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 1 << 20 or not root.is_absolute()
            or '..' in root.parts or not root.is_dir()):
        raise ValueError('bounded native full-motion inventory and explicit asset root required')
    raw = path.read_bytes()
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError('native full-motion inventory raw bytes changed')
    value = json.loads(raw)
    rows = value.get('motions', [])
    if (value.get('schema') != 'roborender.motion-inventory.v1' or value.get('motion_count') != 295
            or len(rows) != 295
            or value.get('motion_records_sha256') != hashlib.sha256(canonical_json(rows)).hexdigest()):
        raise ValueError('exact full295 motion inventory required')
    for key in ('motion_id', 'filename', 'sha256'):
        if len({row.get(key) for row in rows}) != 295:
            raise ValueError('full-motion inventory contains duplicate identities')
    resolved_root = root.resolve(strict=True)
    for row in rows:
        if set(row) != {'motion_id', 'filename', 'sha256', 'bytes'}:
            raise ValueError('unknown native motion descriptor fields')
        require_identifier(row['motion_id'], 'motion id')
        require_sha256(row['sha256'], 'motion asset')
        name = row['filename']
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz')
                or type(row['bytes']) is not int or row['bytes'] <= 0):
            raise ValueError('invalid native motion asset descriptor')
        motion = root / name
        if (motion.is_symlink() or not motion.is_file() or motion.resolve().parent != resolved_root
                or motion.stat().st_size != row['bytes']):
            raise ValueError('native motion asset path/size changed')
        digest = hashlib.sha256()
        with motion.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            raise ValueError('native motion asset bytes changed')
    return value, hashlib.sha256(canonical_json(value)).hexdigest()
