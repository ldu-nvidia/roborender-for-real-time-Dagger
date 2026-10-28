import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from pipeline.deployment.source_snapshot import create, inventory, safe_relative, verify


def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("runtime/\n*.pt\nsecrets/\n")
    (root / "README.md").write_text("source only\n")
    (root / "runtime").mkdir()
    (root / "runtime/private.token").write_text("not shipped")
    (root / "model.pt").write_bytes(b"not shipped")
    return root


def test_snapshot_contains_only_code_and_verifies(tmp_path):
    root = repo(tmp_path)
    output = tmp_path / "release"
    result = create(root, output)
    manifest = json.loads((output / "source-manifest.json").read_text())
    assert verify(root, manifest)["ok"]
    assert result["files"] == 2
    with tarfile.open(output / "source.tar.gz") as archive:
        assert archive.getnames() == [".gitignore", "README.md"]
    with pytest.raises(FileExistsError):
        create(root, output)


@pytest.mark.parametrize("change", ["content", "mode", "extra", "missing"])
def test_drift_is_detected(tmp_path, change):
    root = repo(tmp_path)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    manifest = inventory(root)
    if change == "content":
        (root / "README.md").write_text("changed")
    elif change == "mode":
        (root / "README.md").chmod(0o755)
    elif change == "extra":
        (root / "extra.py").write_text("pass\n")
    else:
        (root / "README.md").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "runtime/code.py", "secret.key", "data/weights.py", "a\\b.py"])
def test_reject_unsafe_or_non_source_names(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_symlink_and_private_key_fail_closed(tmp_path):
    root = repo(tmp_path)
    (root / "outside.py").symlink_to(root / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        inventory(root)
    (root / "outside.py").unlink()
    (root / "bad.txt").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nnever copy\n")
    with pytest.raises(ValueError, match="private key"):
        inventory(root)


def test_manifest_tampering_is_rejected(tmp_path):
    root = repo(tmp_path)
    manifest = inventory(root)
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        verify(root, manifest)
