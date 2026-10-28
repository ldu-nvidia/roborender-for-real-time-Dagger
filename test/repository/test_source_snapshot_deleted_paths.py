"""A source-only snapshot follows deliberate working-tree test relocations."""
from pathlib import Path
import subprocess

import pytest

from pipeline.deployment.source_snapshot import inventory, verify


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_unstaged_test_move_is_inventoried_and_old_manifest_is_rejected(tmp_path):
    git(tmp_path, "init", "-q")
    old = tmp_path / "actor/test_actor.py"
    old.parent.mkdir()
    old.write_text("def test_actor():\n    assert True\n")
    git(tmp_path, "add", "actor/test_actor.py")
    before = inventory(tmp_path)
    moved = tmp_path / "tests/actor/test_actor.py"
    moved.parent.mkdir(parents=True)
    old.rename(moved)
    after = inventory(tmp_path)
    assert [row["path"] for row in after["files"]] == ["tests/actor/test_actor.py"]
    assert after["source_sha256"] != before["source_sha256"]
    assert verify(tmp_path, after)["ok"]
    with pytest.raises(ValueError, match="source drift"):
        verify(tmp_path, before)


def test_deleted_source_is_not_reintroduced_from_git_index(tmp_path):
    git(tmp_path, "init", "-q")
    for name in ("active.py", "obsolete.py"):
        (tmp_path / name).write_text("VALUE = 1\n")
    git(tmp_path, "add", "active.py", "obsolete.py")
    (tmp_path / "obsolete.py").unlink()
    assert [r["path"] for r in inventory(tmp_path)["files"]] == ["active.py"]
