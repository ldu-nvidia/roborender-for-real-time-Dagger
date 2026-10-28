"""CPU-only no-clobber catalog tests; all mutations are inside pytest tmp_path."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from roborender.infrastructure import storage_layout as layout


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    model = source / "checkpoint.pt"
    model.write_bytes(b"test checkpoint bytes")
    root = tmp_path / "catalog"
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    plan = dict(schema=layout.SCHEMA, roots=[str(root)],
        directories=[str(root / "datasets"), str(root / "checkpoints")],
        links=[dict(path=str(root / "datasets" / "original"), target=str(source))],
        copy_files=[dict(path=str(root / "checkpoints" / "selected.pt"), target=str(model), sha256=checksum)])
    return plan, root, source


def test_default_dry_run_writes_nothing_and_verifies_sources(example):
    plan, root, source = example
    result = layout.apply_plan(plan, receipt=root / "receipt.json")
    assert result["mode"] == "dry_run" and not root.exists()
    assert not result["original_artifacts_modified"] and not result["compute_mutated"]
    assert (source / "checkpoint.pt").read_bytes() == b"test checkpoint bytes"


def test_create_links_verified_copy_receipt_and_exact_idempotence(example):
    plan, root, source = example
    receipt = root / "receipt.json"
    first = layout.apply_plan(plan, execute=True, receipt=receipt)
    assert first["mode"] == "applied"
    assert os.readlink(root / "datasets" / "original") == str(source)
    assert (root / "checkpoints" / "selected.pt").read_bytes() == (source / "checkpoint.pt").read_bytes()
    copied = (root / "checkpoints" / "selected.pt").stat()
    assert copied.st_ino != (source / "checkpoint.pt").stat().st_ino
    raw, inode = receipt.read_bytes(), receipt.stat().st_ino
    second = layout.apply_plan(plan, execute=True, receipt=receipt)
    assert all(row["status"] == "existing" for row in second["actions"])
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert receipt.read_bytes() == raw and receipt.stat().st_ino == inode
    value = json.loads(raw); claimed = value.pop("receipt_sha256")
    assert layout.digest(value) == claimed


@pytest.mark.parametrize("fault", ["schema", "extra", "op_extra", "wrong_hash", "missing_target", "duplicate", "out_of_root", "missing_parent", "root_is_file"])
def test_all_preflight_failures_precede_any_creation(example, fault, tmp_path):
    plan, root, source = example
    if fault == "schema": plan["schema"] = "unversioned"
    elif fault == "extra": plan["delete"] = True
    elif fault == "op_extra": plan["links"][0]["replace"] = True
    elif fault == "wrong_hash": plan["copy_files"][0]["sha256"] = "a" * 64
    elif fault == "missing_target": plan["links"][0]["target"] = str(source / "missing")
    elif fault == "duplicate": plan["links"].append(deepcopy(plan["links"][0]))
    elif fault == "out_of_root": plan["links"][0]["path"] = str(tmp_path / "outside")
    elif fault == "missing_parent": plan["links"][0]["path"] = str(root / "undeclared" / "link")
    elif fault == "root_is_file": root.write_bytes(b"never overwrite")
    with pytest.raises((ValueError, FileNotFoundError, NotADirectoryError)):
        layout.apply_plan(plan, execute=True)
    assert not root.is_dir()
    if root.is_file(): assert root.read_bytes() == b"never overwrite"


@pytest.mark.parametrize("path", ["relative", "/tmp/../elsewhere", "/tmp//alias", "/tmp/./alias", "//", "//tmp/catalog", "/tmp/$NAME", "/tmp/~name", "/tmp/new\nname", "/"])
def test_unsafe_or_broad_root_rejected(example, path):
    plan, root, _ = example
    plan["roots"] = [path]
    with pytest.raises(ValueError): layout.apply_plan(plan, execute=True)
    assert not root.exists()


@pytest.mark.parametrize("collision", ["directory_at_link", "wrong_link", "dangling_link", "wrong_file", "copy_is_link"])
def test_existing_collision_never_overwritten(example, collision, tmp_path):
    plan, root, source = example
    (root / "datasets").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    link = root / "datasets" / "original"
    copied = root / "checkpoints" / "selected.pt"
    if collision == "directory_at_link": link.mkdir()
    elif collision == "wrong_link": link.symlink_to(tmp_path)
    elif collision == "dangling_link": link.symlink_to(tmp_path / "missing")
    elif collision == "wrong_file": copied.write_bytes(b"preserved")
    elif collision == "copy_is_link": copied.symlink_to(source / "checkpoint.pt")
    before = (link.lstat() if link.is_symlink() or link.exists() else copied.lstat()).st_ino
    with pytest.raises(ValueError): layout.apply_plan(plan, execute=True)
    after = (link.lstat() if link.is_symlink() or link.exists() else copied.lstat()).st_ino
    assert before == after
    if collision == "wrong_file": assert copied.read_bytes() == b"preserved"


@pytest.mark.parametrize("where", ["root", "parent", "inside_root"])
def test_symlink_destination_ancestry_cannot_escape(example, where, tmp_path):
    plan, root, source = example
    outside = tmp_path / "external"; outside.mkdir()
    if where == "root": root.symlink_to(outside, target_is_directory=True)
    elif where == "inside_root":
        root.mkdir(); (root / "datasets").symlink_to(outside, target_is_directory=True)
    else:
        alias = tmp_path / "parent-alias"; alias.symlink_to(outside, target_is_directory=True)
        plan = dict(schema=layout.SCHEMA, roots=[str(alias / "catalog")], directories=[], links=[])
    with pytest.raises(ValueError): layout.apply_plan(plan, execute=True)
    assert list(outside.iterdir()) == []


def test_source_under_output_and_indirect_cycle_rejected(example, tmp_path):
    plan, root, _ = example
    root.mkdir()
    inner = root / "existing"; inner.mkdir()
    alias = tmp_path / "source-alias"; alias.symlink_to(inner, target_is_directory=True)
    for target in (inner, alias):
        plan["links"][0]["target"] = str(target)
        with pytest.raises(ValueError, match="under an output root"):
            layout.apply_plan(plan, execute=True)
    assert not (root / "datasets").exists()


def test_existing_readonly_source_alias_allowed_and_bound(example, tmp_path):
    plan, root, source = example
    alias = tmp_path / "asset-alias"; alias.symlink_to(source, target_is_directory=True)
    plan["links"][0]["target"] = str(alias)
    layout.apply_plan(plan, execute=True, receipt=root / "receipt.json")
    record = json.loads((root / "receipt.json").read_text())
    link_record = next(row for row in record["operations"] if row["kind"] == "link")
    assert link_record["target"] == str(alias) and link_record["resolved_target"] == str(source)


@pytest.mark.parametrize("fault", ["tampered", "missing_object", "different_plan", "receipt_symlink"])
def test_immutable_receipt_integrity_and_state_mismatch_fail_closed(example, fault, tmp_path):
    plan, root, _ = example
    receipt = root / "receipt.json"
    layout.apply_plan(plan, execute=True, receipt=receipt)
    if fault == "tampered":
        receipt.chmod(0o644)
        value = json.loads(receipt.read_text()); value["plan_sha256"] = "f" * 64
        receipt.write_text(json.dumps(value))
    elif fault == "missing_object": (root / "datasets" / "original").unlink()
    elif fault == "different_plan": plan["directories"].append(str(root / "new"))
    else:
        saved = tmp_path / "saved-receipt"; receipt.rename(saved); receipt.symlink_to(saved)
    with pytest.raises((ValueError, OSError)): layout.apply_plan(plan, execute=True, receipt=receipt)
    assert not (root / "new").exists()
    if fault == "missing_object": assert not (root / "datasets" / "original").exists()


def test_receipt_cannot_replace_operation_or_escape_root(example, tmp_path):
    plan, root, _ = example
    for receipt in (tmp_path / "outside.json", root / "datasets"):
        with pytest.raises(ValueError): layout.apply_plan(plan, execute=True, receipt=receipt)
    assert not root.exists()


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "plan.json"; path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate JSON"): layout.load_plan(path)


def test_symlink_parent_injected_after_preflight_still_cannot_write(example, monkeypatch, tmp_path):
    plan, root, _ = example
    outside = tmp_path / "external"; outside.mkdir()
    original = layout._create
    def injected(operation, binding):
        if operation["kind"] == "link":
            (root / "datasets").rmdir()
            (root / "datasets").symlink_to(outside, target_is_directory=True)
        return original(operation, binding)
    monkeypatch.setattr(layout, "_create", injected)
    with pytest.raises(OSError): layout.apply_plan(plan, execute=True, receipt=root / "receipt.json")
    assert list(outside.iterdir()) == [] and not (root / "receipt.json").exists()


def test_source_changed_after_preflight_is_not_copied(example, monkeypatch):
    plan, root, source = example
    original = layout._create
    def changed(operation, binding):
        if operation["kind"] == "directory" and operation["path"] == str(root):
            (source / "checkpoint.pt").write_bytes(b"changed by unrelated actor")
        return original(operation, binding)
    monkeypatch.setattr(layout, "_create", changed)
    with pytest.raises(ValueError, match="SHA-256"): layout.apply_plan(plan, execute=True, receipt=root / "receipt.json")
    assert not (root / "checkpoints" / "selected.pt").exists()
    assert not (root / "receipt.json").exists()


def test_cli_is_dry_run_without_execute(example, tmp_path):
    plan, root, _ = example
    plan_path = tmp_path / "plan.json"; plan_path.write_text(json.dumps(plan))
    result = subprocess.run([sys.executable, "-B", str(Path(layout.__file__)), "--plan", str(plan_path)],
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["mode"] == "dry_run" and not root.exists()


@pytest.mark.parametrize("execute", ["false", "true", 1, None])
def test_execution_requires_explicit_boolean(example, execute):
    plan, root, _ = example
    with pytest.raises(ValueError, match="explicit boolean"):
        layout.apply_plan(plan, execute=execute)
    assert not root.exists()


def test_concurrent_file_growth_cannot_exceed_hash_byte_limit(tmp_path, monkeypatch):
    path = tmp_path / "growing.pt"; path.write_bytes(b"abc")
    monkeypatch.setattr(layout, "MAX_COPY_BYTES", 4)
    original = layout.hashlib.sha256
    def grow():
        with path.open("ab") as stream: stream.write(b"defgh")
        return original()
    monkeypatch.setattr(layout.hashlib, "sha256", grow)
    with pytest.raises(ValueError, match="hash byte limit"):
        layout._file_hash(path)


def test_failed_copy_is_retained_never_receipted_or_overwritten(example, monkeypatch):
    plan, root, source = example
    original = layout._create
    def race(operation, binding):
        if operation["kind"] == "copy_file":
            (source / "checkpoint.pt").write_bytes(b"concurrently changed source")
        return original(operation, binding)
    monkeypatch.setattr(layout, "_create", race)
    with pytest.raises(ValueError, match="partial file retained"):
        layout.apply_plan(plan, execute=True, receipt=root / "receipt.json")
    partial = root / "checkpoints" / "selected.pt"
    assert partial.read_bytes() == b"concurrently changed source" and not (root / "receipt.json").exists()
    inode = partial.stat().st_ino
    with pytest.raises(ValueError, match="SHA-256"):
        layout.apply_plan(plan, execute=True, receipt=root / "receipt.json")
    assert partial.stat().st_ino == inode


def test_no_move_delete_overwrite_or_resource_commands_exposed():
    text = Path(layout.__file__).read_text()
    for unsafe in ("os.rename(", "os.replace(", "os.unlink(", "shutil.rmtree", "subprocess", "deployment.delete", "SIGTERM"):
        assert unsafe not in text
