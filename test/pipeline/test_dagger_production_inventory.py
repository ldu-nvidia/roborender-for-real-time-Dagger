"""Boot-time full295 inventory identity checks; tiny CPU-only test assets."""
import hashlib
import json
from pathlib import Path

import pytest

from dagger_rtx.contracts import canonical_json
from dagger_rtx.production_inventory import read_inventory


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inventory(tmp_path):
    root = tmp_path / 'motions'
    root.mkdir()
    rows = []
    for index in range(295):
        raw = f'cpu-only-motion-{index:03d}'.encode()
        name = f'motion-{index:03d}.npz'
        (root / name).write_bytes(raw)
        rows.append(dict(motion_id=f'motion-{index:03d}', filename=name,
                         sha256=sha(raw), bytes=len(raw)))
    value = dict(schema='roborender.motion-inventory.v1', motion_count=295,
                 motion_records_sha256=sha(canonical_json(rows)), motions=rows)
    path = tmp_path / 'inventory.json'
    return root, path, value


def publish_inventory(path, value, *, pretty=False):
    value['motion_records_sha256'] = sha(canonical_json(value['motions']))
    raw = (json.dumps(value, indent=2).encode() if pretty else canonical_json(value)) + b'\n'
    path.write_bytes(raw)
    return sha(raw)


def test_raw_publication_binding_is_distinct_from_canonical_inventory_identity(inventory):
    root, path, value = inventory
    first_raw = publish_inventory(path, value)
    actual, canonical = read_inventory(path, first_raw, root)
    assert actual == value and canonical == sha(canonical_json(value))
    assert canonical != first_raw
    second_raw = publish_inventory(path, value, pretty=True)
    assert second_raw != first_raw
    assert read_inventory(path, second_raw, root) == (value, canonical)
    with pytest.raises(ValueError, match='raw bytes changed'):
        read_inventory(path, first_raw, root)


@pytest.mark.parametrize('identity', ['motion_id', 'filename', 'sha256'])
def test_duplicate_identity_rejected_even_with_recomputed_records_digest(inventory, identity):
    root, path, value = inventory
    value['motions'][1][identity] = value['motions'][0][identity]
    raw = publish_inventory(path, value)
    with pytest.raises(ValueError, match='duplicate identities'):
        read_inventory(path, raw, root)


@pytest.mark.parametrize('fault', ['missing-record', 'declared-count', 'records-digest'])
def test_full295_cardinality_and_records_digest_are_required(inventory, fault):
    root, path, value = inventory
    if fault == 'missing-record':
        value['motions'].pop()
    elif fault == 'declared-count':
        value['motion_count'] = 294
    raw = publish_inventory(path, value)
    if fault == 'records-digest':
        value['motion_records_sha256'] = '0' * 64
        path.write_bytes(canonical_json(value))
        raw = sha(path.read_bytes())
    with pytest.raises(ValueError, match='exact full295'):
        read_inventory(path, raw, root)


@pytest.mark.parametrize('fault', ['missing', 'same-size-changed', 'size-changed', 'symlink'])
def test_every_tiny_asset_is_verified_not_only_inventory_metadata(inventory, fault, tmp_path):
    root, path, value = inventory
    raw = publish_inventory(path, value)
    asset = root / value['motions'][180]['filename']
    original = asset.read_bytes()
    if fault == 'missing':
        asset.unlink()
    elif fault == 'same-size-changed':
        asset.write_bytes(b'X' * len(original))
    elif fault == 'size-changed':
        asset.write_bytes(original + b'X')
    else:
        outside = tmp_path / 'outside.npz'
        outside.write_bytes(original)
        asset.unlink()
        asset.symlink_to(outside)
    with pytest.raises(ValueError, match='native motion asset'):
        read_inventory(path, raw, root)


@pytest.mark.parametrize('fault', ['relative-root', 'relative-file', 'inventory-symlink', 'unknown-field', 'filename-traversal'])
def test_paths_and_exact_descriptors_fail_closed(inventory, fault, tmp_path, monkeypatch):
    root, path, value = inventory
    if fault == 'unknown-field':
        value['motions'][0]['unused'] = True
    elif fault == 'filename-traversal':
        value['motions'][0]['filename'] = '../motion-000.npz'
    raw = publish_inventory(path, value)
    monkeypatch.chdir(tmp_path)
    if fault == 'relative-root':
        root = Path('motions')
    elif fault == 'relative-file':
        path = Path('inventory.json')
    elif fault == 'inventory-symlink':
        link = tmp_path / 'inventory-link.json'
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        read_inventory(path, raw, root)
