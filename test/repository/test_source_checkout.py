"""CPU-only staged Git checkout cutovers in pytest temporary directories."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.deployment import source_snapshot
from roborender.infrastructure import source_checkout as checkout


@pytest.fixture
def staged(tmp_path):
    parent = tmp_path / "data1/workspaces/roborender"; parent.mkdir(parents=True)
    old = tmp_path / "data1/ldu/roborender"; old.mkdir(parents=True)
    (old / "modified.txt").write_text("preserve old modified checkout")
    stage, destination, saved = (parent / name for name in checkout.NAMES)
    stage.mkdir()
    subprocess.run(["git", "init", "-q", str(stage)], check=True)
    (stage / "README.md").write_text("new reviewed source\n")
    (stage / "entry.py").write_text("print('not executed')\n")
    destination.symlink_to(checkout.EXPECTED_LINK, target_is_directory=True)
    manifests = tmp_path / "data2/catalog"; manifests.mkdir(parents=True)
    manifest = manifests / "manifest.json"
    manifest.write_text(json.dumps(source_snapshot.inventory(stage)))
    arguments = dict(staging=str(stage), destination=str(destination), preserved_alias=str(saved),
        expected_link=checkout.EXPECTED_LINK, manifest=str(manifest), receipt_prefix=str(manifests / "cutover"))
    return arguments, stage, destination, saved, old


def test_default_preflight_preserves_all_paths_without_receipts(staged):
    args, stage, destination, saved, old = staged
    result = checkout.activate(**args)
    assert result["mode"] == "dry_run" and result["phase"] == "INITIAL"
    assert stage.is_dir() and destination.is_symlink() and not saved.exists()
    assert not Path(result["before_receipt"]).exists() and not Path(result["after_receipt"]).exists()
    assert (old / "modified.txt").read_text() == "preserve old modified checkout"


def test_two_no_replace_renames_preserve_old_clone_and_are_idempotent(staged):
    args, stage, destination, saved, old = staged
    stage_inode = stage.stat().st_ino; alias_inode = destination.lstat().st_ino
    result = checkout.activate(**args, execute=True)
    assert result["phase"] == "ACTIVE" and not stage.exists()
    assert destination.is_dir() and not destination.is_symlink()
    assert destination.stat().st_ino == stage_inode and saved.lstat().st_ino == alias_inode
    assert os.readlink(saved) == checkout.EXPECTED_LINK and saved.resolve() == old
    assert (old / "modified.txt").read_text() == "preserve old modified checkout"
    before = Path(result["before_receipt"]).read_bytes(); after = Path(result["after_receipt"]).read_bytes()
    assert checkout.activate(**args, execute=True)["phase"] == "ACTIVE"
    assert Path(result["before_receipt"]).read_bytes() == before
    assert Path(result["after_receipt"]).read_bytes() == after
    assert not result["old_clone_modified"] and not result["compute_mutated"]


@pytest.mark.parametrize("fault", ["wrong_alias", "existing_saved", "destination_dir", "staging_symlink", "source_drift", "manifest_drift", "wrong_names", "git_symlink", "missing_git"])
def test_preflight_errors_never_move_source_or_alias(staged, fault, tmp_path):
    args, stage, destination, saved, old = staged
    if fault == "wrong_alias": destination.unlink(); destination.symlink_to(old)
    elif fault == "existing_saved": saved.write_text("never overwrite")
    elif fault == "destination_dir": destination.unlink(); destination.mkdir()
    elif fault == "staging_symlink":
        other = stage.with_name("other"); stage.rename(other); stage.symlink_to(other)
    elif fault == "source_drift": (stage / "entry.py").write_text("changed")
    elif fault == "manifest_drift":
        path = Path(args["manifest"]); value = json.loads(path.read_text()); value["source_sha256"] = "f" * 64
        path.write_text(json.dumps(value))
    elif fault == "wrong_names": args["staging"] = str(stage.with_name("other"))
    elif fault == "git_symlink":
        git = stage / ".git"; moved = tmp_path / "git-data"; git.rename(moved); git.symlink_to(moved)
    elif fault == "missing_git": (stage / ".git").rename(stage.with_name("saved-git"))
    with pytest.raises((ValueError, FileNotFoundError)):
        checkout.activate(**args, execute=True)
    assert not Path(args["receipt_prefix"] + ".before.json").exists()
    assert (old / "modified.txt").exists()
    if fault == "existing_saved": assert saved.read_text() == "never overwrite"


@pytest.mark.parametrize("interrupt_after", [1, 2])
def test_lost_ack_after_either_rename_reconciles_without_replay(staged, monkeypatch, interrupt_after):
    args, stage, destination, saved, old = staged
    original = checkout._rename_noreplace
    calls = []
    def interrupted(source, target, identity):
        original(source, target, identity)
        calls.append((str(source), str(target)))
        if len(calls) == interrupt_after:
            raise ConnectionError("lost ACK")
    monkeypatch.setattr(checkout, "_rename_noreplace", interrupted)
    with pytest.raises(ConnectionError): checkout.activate(**args, execute=True)
    assert Path(args["receipt_prefix"] + ".before.json").is_file()
    assert not Path(args["receipt_prefix"] + ".after.json").exists()
    phase = checkout.activate(**args)["phase"]
    assert phase == ("ALIAS_PRESERVED" if interrupt_after == 1 else "ACTIVE")
    resumed = []
    def record(source, target, identity):
        resumed.append((str(source), str(target))); original(source, target, identity)
    monkeypatch.setattr(checkout, "_rename_noreplace", record)
    assert checkout.activate(**args, execute=True)["phase"] == "ACTIVE"
    assert len(resumed) == 2 - interrupt_after
    assert saved.resolve() == old


def test_unreceipted_partial_cutover_is_not_adopted(staged):
    args, stage, destination, saved, _ = staged
    destination.rename(saved)
    with pytest.raises(ValueError): checkout.activate(**args, execute=True)
    assert stage.is_dir() and not destination.exists() and saved.is_symlink()


def test_interruption_before_first_rename_is_recoverable(staged, monkeypatch):
    args, stage, destination, saved, _ = staged
    original = checkout._rename_noreplace
    def fail(*args): raise ConnectionError("before rename")
    monkeypatch.setattr(checkout, "_rename_noreplace", fail)
    with pytest.raises(ConnectionError): checkout.activate(**args, execute=True)
    assert stage.is_dir() and destination.is_symlink() and not saved.exists()
    monkeypatch.setattr(checkout, "_rename_noreplace", original)
    assert checkout.activate(**args, execute=True)["phase"] == "ACTIVE"


def test_unknown_destination_created_during_cutover_is_never_overwritten(staged, monkeypatch):
    args, stage, destination, saved, _ = staged
    original = checkout._rename_noreplace
    def race(source, target, identity):
        if source == stage:
            target.mkdir(); (target / "unrelated").write_text("preserve")
        original(source, target, identity)
    monkeypatch.setattr(checkout, "_rename_noreplace", race)
    with pytest.raises(FileExistsError): checkout.activate(**args, execute=True)
    assert (destination / "unrelated").read_text() == "preserve"
    assert stage.is_dir() and saved.is_symlink()
    with pytest.raises(ValueError, match="unrecognized"):
        checkout.activate(**args, execute=True)


def test_unknown_preserved_alias_is_never_overwritten_even_on_race(staged, monkeypatch):
    args, stage, destination, saved, _ = staged
    original = checkout._rename_noreplace
    def race(source, target, identity):
        saved.write_text("unrelated")
        original(source, target, identity)
    monkeypatch.setattr(checkout, "_rename_noreplace", race)
    with pytest.raises(FileExistsError): checkout.activate(**args, execute=True)
    assert saved.read_text() == "unrelated" and destination.is_symlink() and stage.is_dir()


def test_receipt_tampering_or_changed_manifest_blocks_reconciliation(staged, monkeypatch):
    args, stage, destination, saved, _ = staged
    def fail(*args): raise ConnectionError()
    monkeypatch.setattr(checkout, "_rename_noreplace", fail)
    with pytest.raises(ConnectionError): checkout.activate(**args, execute=True)
    before = Path(args["receipt_prefix"] + ".before.json")
    before.chmod(0o644); value = json.loads(before.read_text()); value["staged"]["inode"] += 1
    before.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"): checkout.activate(**args, execute=True)
    assert stage.exists() and destination.is_symlink() and not saved.exists()


@pytest.mark.parametrize("field", ["manifest", "receipt_prefix"])
def test_evidence_cannot_be_inside_a_moving_directory(staged, field):
    args, stage, destination, saved, _ = staged
    args[field] = str(stage / "evidence.json")
    with pytest.raises(ValueError, match="outside"): checkout.activate(**args, execute=True)
    assert stage.exists() and destination.is_symlink() and not saved.exists()


def test_symlink_parent_rejected(staged, tmp_path):
    args, stage, destination, saved, _ = staged
    alias = tmp_path / "parent-alias"; alias.symlink_to(stage.parent)
    args.update(staging=str(alias / stage.name), destination=str(alias / destination.name), preserved_alias=str(alias / saved.name))
    with pytest.raises(ValueError, match="parent"): checkout.activate(**args, execute=True)
    assert stage.exists() and destination.is_symlink() and not saved.exists()


def test_wrong_filesystem_rejected_before_alias_move(staged, monkeypatch):
    args, stage, destination, saved, _ = staged
    original = checkout._identity
    def different(path):
        result = original(path)
        if Path(path) == stage:
            result = deepcopy(result); result["device"] += 1
        return result
    monkeypatch.setattr(checkout, "_identity", different)
    with pytest.raises(ValueError, match="filesystem"): checkout.activate(**args, execute=True)
    assert destination.is_symlink() and not saved.exists()


@pytest.mark.parametrize("execute", ["false", 1, None])
def test_explicit_boolean_required(staged, execute):
    args, stage, destination, saved, _ = staged
    with pytest.raises(ValueError, match="boolean"): checkout.activate(**args, execute=execute)
    assert stage.exists() and destination.is_symlink() and not saved.exists()


def test_cli_default_is_read_only(staged):
    args, stage, destination, saved, _ = staged
    argv = [sys.executable, "-B", checkout.__file__]
    for name, value in args.items(): argv.extend(["--" + name.replace("_", "-"), value])
    result = subprocess.run(argv, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["mode"] == "dry_run"
    assert stage.exists() and destination.is_symlink() and not saved.exists()


def test_receipts_cannot_write_inside_old_clone(staged):
    args, stage, destination, saved, old = staged
    args["receipt_prefix"] = str(old / "cutover")
    with pytest.raises(ValueError, match="not modify the old clone"):
        checkout.activate(**args, execute=True)
    assert sorted(path.name for path in old.iterdir()) == ["modified.txt"]
    assert stage.exists() and destination.is_symlink() and not saved.exists()


def test_old_alias_cannot_indirectly_reference_staging(staged, tmp_path):
    args, stage, destination, saved, old = staged
    old.rename(old.with_name("preserved-test-old"))
    old.symlink_to(stage, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        checkout.activate(**args, execute=True)
    assert stage.exists() and destination.is_symlink() and not saved.exists()
