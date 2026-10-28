"""Immutable native receipts must not overwrite earlier evaluation evidence."""
from test.deployment.source_paths import ACTOR_ROOT
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(ACTOR_ROOT))
from native_artifact_io import canonical, file_sha256, publish_json


def test_canonical_json_is_stable_and_rejects_nan():
    assert canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical({"bad": float("nan")})


def test_publish_is_exclusive_and_hashes_exact_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    publish_json(path, {"ok": True})
    expected = b'{"ok":true}\n'
    assert path.read_bytes() == expected
    assert file_sha256(path) == hashlib.sha256(expected).hexdigest()
    with pytest.raises(FileExistsError):
        publish_json(path, {"ok": False})
    assert path.read_bytes() == expected
    assert list(tmp_path.iterdir()) == [path]


def test_failed_encoding_leaves_no_receipt_or_temp_file(tmp_path):
    with pytest.raises(ValueError):
        publish_json(tmp_path / "invalid.json", {"bad": float("nan")})
    assert list(tmp_path.iterdir()) == []
