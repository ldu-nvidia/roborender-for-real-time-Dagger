"""Narrow, no-overwrite activation of the staged October24 Leo Git checkout.

The old `source` symlink is renamed, never followed or changed. Its target clone
is untouched. A durable before receipt permits recovery after either rename;
unknown destinations/backups are never adopted. No automatic rollback: preserve
the activated checkout separately before explicitly restoring the recorded old
alias with a no-clobber rename. No files or compute resources are deleted.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import stat
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.deployment import source_snapshot
from roborender.infrastructure import storage_layout as storage

SCHEMA = "roborender.source-checkout-cutover.v1"
EXPECTED_LINK = "../../ldu/roborender"
NAMES = ("source.staging-20261024", "source", "source_legacy_20261024")


def _identity(path):
    info = storage._stat(path)
    if info is None:
        return None
    return dict(device=info.st_dev, inode=info.st_ino, kind=("directory" if stat.S_ISDIR(info.st_mode)
        else "symlink" if stat.S_ISLNK(info.st_mode) else "other"))


def _sealed(value):
    return dict(value, receipt_sha256=storage.digest(value))


def _read_receipt(path, phase):
    value = storage._read_json(path)
    if not isinstance(value, dict):
        raise ValueError("invalid cutover receipt")
    unsealed = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (value.get("schema") != SCHEMA or value.get("phase") != phase
            or value.get("receipt_sha256") != storage.digest(unsealed)):
        raise ValueError("cutover receipt integrity mismatch")
    return value


def _immutable(path, value):
    if storage._stat(path) is not None:
        if storage._read_json(path) != value:
            raise ValueError("immutable cutover receipt collision")
        return
    parent = storage._open_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=parent)
        with os.fdopen(fd, "wb") as output:
            output.write(storage.canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.fsync(parent)
    finally:
        os.close(parent)


def _verify_source(path, manifest):
    storage._check_parents(path)
    identity = _identity(path)
    if identity is None or identity["kind"] != "directory":
        raise ValueError("staged/activated checkout must be a real directory")
    git = _identity(path / ".git")
    if git is None or git["kind"] != "directory":
        raise ValueError("staged/activated checkout requires its own real .git directory")
    source_snapshot.verify(path, manifest)
    if _identity(path) != identity:
        raise ValueError("checkout identity changed during source verification")
    return identity


def _validate_inputs(staging, destination, preserved_alias, expected_link, manifest, receipt_prefix):
    paths = tuple(storage._path(str(value)) for value in (staging, destination, preserved_alias))
    if tuple(path.name for path in paths) != NAMES or len({path.parent for path in paths}) != 1:
        raise ValueError("exact October24 sibling staging/destination/preserved-alias paths required")
    if expected_link != EXPECTED_LINK:
        raise ValueError("exact expected old source symlink required")
    parent = paths[0].parent
    storage._check_parents(paths[0])
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("real existing common parent required")
    manifest_path = storage._path(str(manifest))
    prefix = storage._path(str(receipt_prefix))
    storage._check_parents(manifest_path)
    storage._check_parents(prefix)
    if not prefix.parent.is_dir() or prefix.parent.is_symlink():
        raise ValueError("receipt parent must be an existing real directory")
    for path in (manifest_path, prefix):
        if any(path == moving or path.is_relative_to(moving) for moving in paths):
            raise ValueError("manifest/receipts must be outside all moving checkout paths")
    before, after = Path(str(prefix) + ".before.json"), Path(str(prefix) + ".after.json")
    if manifest_path in (before, after):
        raise ValueError("manifest and receipt paths must differ")
    source_manifest = storage._read_json(manifest_path)
    request = dict(staging=str(paths[0]), destination=str(paths[1]), preserved_alias=str(paths[2]),
        expected_link=expected_link, manifest=str(manifest_path), manifest_sha256=storage.digest(source_manifest),
        receipt_prefix=str(prefix))
    return paths, source_manifest, request, before, after


def _initial_receipt(paths, manifest, request):
    staging, destination, preserved = paths
    staged = _verify_source(staging, manifest)
    parent_info = _identity(staging.parent)
    old = _identity(destination)
    if staged["device"] != parent_info["device"]:
        raise ValueError("staged source must be on the common parent's filesystem")
    if (old is None or old["kind"] != "symlink" or os.readlink(destination) != EXPECTED_LINK
            or old["device"] != parent_info["device"] or _identity(preserved) is not None):
        raise ValueError("exact old symlink and absent preserved-alias destination required")
    target = destination.resolve(strict=True)
    if not target.is_dir():
        raise ValueError("old alias target must be an existing directory")
    if (any(path == target or path.is_relative_to(target) or target.is_relative_to(path) for path in paths)
            or _identity(target) == staged):
        raise ValueError("old clone must be separate from every moving checkout path")
    receipt_prefix = Path(request["receipt_prefix"])
    if receipt_prefix == target or receipt_prefix.is_relative_to(target):
        raise ValueError("cutover receipts must not modify the old clone")
    return _sealed(dict(schema=SCHEMA, phase="BEFORE", request=request, parent=parent_info,
        staged=staged, old_alias=old, legacy_target=str(target), legacy_target_identity=_identity(target),
        source_sha256=manifest["source_sha256"], old_clone_modified=False, compute_mutated=False))


def _state(paths, manifest, request, before):
    staging, destination, preserved = paths
    if (set(before) != {"schema", "phase", "request", "parent", "staged", "old_alias", "legacy_target",
            "legacy_target_identity", "source_sha256", "old_clone_modified", "compute_mutated", "receipt_sha256"}
            or before["request"] != request or before["source_sha256"] != manifest["source_sha256"]
            or before["parent"] != _identity(staging.parent)
            or before["old_clone_modified"] is not False or before["compute_mutated"] is not False):
        raise ValueError("before receipt does not bind this exact cutover")
    stage_id, destination_id, preserved_id = (_identity(path) for path in paths)
    if stage_id == before["staged"] and destination_id == before["old_alias"] and preserved_id is None:
        phase, source, old = "INITIAL", staging, destination
    elif stage_id == before["staged"] and destination_id is None and preserved_id == before["old_alias"]:
        phase, source, old = "ALIAS_PRESERVED", staging, preserved
    elif stage_id is None and destination_id == before["staged"] and preserved_id == before["old_alias"]:
        phase, source, old = "ACTIVE", destination, preserved
    else:
        raise ValueError("unrecognized cutover filesystem state; no overwrite or automatic adoption")
    if (os.readlink(old) != EXPECTED_LINK or str(old.resolve(strict=True)) != before["legacy_target"]
            or _identity(Path(before["legacy_target"])) != before["legacy_target_identity"]):
        raise ValueError("preserved old alias/target identity changed")
    if _verify_source(source, manifest) != before["staged"]:
        raise ValueError("staged source identity differs")
    return phase


def _rename_noreplace(source, destination, expected):
    """Linux atomic no-clobber rename; never fall back to replacing rename()."""
    if source.parent != destination.parent:
        raise ValueError("rename must remain within the exact common parent")
    parent = storage._open_directory(source.parent)
    try:
        info = os.stat(source.name, dir_fd=parent, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != (expected["device"], expected["inode"]):
            raise ValueError("rename source identity changed")
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise RuntimeError("Linux renameat2 required; no unsafe fallback")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(parent, os.fsencode(source.name), parent, os.fsencode(destination.name), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        os.fsync(parent)
    finally:
        os.close(parent)


def activate(staging, destination, preserved_alias, *, expected_link, manifest, receipt_prefix, execute=False):
    if type(execute) is not bool:
        raise ValueError("execute must be an explicit boolean")
    paths, source_manifest, request, before_path, after_path = _validate_inputs(
        staging, destination, preserved_alias, expected_link, manifest, receipt_prefix)
    had_before = storage._stat(before_path) is not None
    if not had_before and storage._stat(after_path) is not None:
        raise ValueError("after receipt without before receipt is not an admitted cutover")
    before = _read_receipt(before_path, "BEFORE") if had_before else _initial_receipt(paths, source_manifest, request)
    phase = _state(paths, source_manifest, request, before)
    after = _sealed(dict(schema=SCHEMA, phase="AFTER", before_receipt_sha256=before["receipt_sha256"],
        request=request, source_sha256=source_manifest["source_sha256"], activated_identity=before["staged"],
        preserved_alias_identity=before["old_alias"], legacy_target=before["legacy_target"],
        old_clone_modified=False, compute_mutated=False))
    if storage._stat(after_path) is not None:
        if phase != "ACTIVE" or _read_receipt(after_path, "AFTER") != after:
            raise ValueError("after receipt does not match activated state")
    if execute:
        _immutable(before_path, before)
        if phase == "INITIAL":
            _rename_noreplace(paths[1], paths[2], before["old_alias"])
            phase = _state(paths, source_manifest, request, before)
        if phase == "ALIAS_PRESERVED":
            _rename_noreplace(paths[0], paths[1], before["staged"])
            phase = _state(paths, source_manifest, request, before)
        if phase != "ACTIVE":
            raise ValueError("checkout activation did not reach verified ACTIVE state")
        _immutable(after_path, after)
    return dict(schema=SCHEMA, mode="executed" if execute else "dry_run", phase=phase,
        source_sha256=source_manifest["source_sha256"], before_receipt=str(before_path), after_receipt=str(after_path),
        destination=str(paths[1]), preserved_alias=str(paths[2]), legacy_target=before["legacy_target"],
        old_clone_modified=False, compute_mutated=False,
        rollback="No automatic rollback. Preserve the new checkout elsewhere after checking writers, then explicitly restore the recorded old alias with a no-clobber rename; never overwrite either checkout.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--preserved-alias", required=True)
    parser.add_argument("--expected-link", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt-prefix", required=True)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    print(json.dumps(activate(**args), sort_keys=True))


if __name__ == "__main__":
    main()
