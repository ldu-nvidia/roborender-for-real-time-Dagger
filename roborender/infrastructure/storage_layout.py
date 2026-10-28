"""Create explicit storage catalogs without moving or overwriting artifacts.

Default CLI behavior is read-only preflight. --execute creates only declared
directories, links and SHA-verified file copies. Original assets, source releases,
ledgers, workloads and compute allocations are never modified. A failed copy is
retained for explicit reconciliation, never deleted or silently retried over.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

SCHEMA = "roborender.storage-catalog.v1"
RECEIPT_SCHEMA = "roborender.storage-catalog-receipt.v1"
MAX_METADATA_BYTES = 2 << 20
MAX_COPY_BYTES = 2 << 30
_BROAD_ROOTS = {"/", "/home", "/home/ldu", "/root", "/workspace", "/workspace/project", "/data1", "/data2", "/mnt", "/tmp"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _path(value):
    if (not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in ("$", "~", "\\"))):
        raise ValueError("literal canonical absolute path required")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or value.startswith("//") or parsed.as_posix() != value or ".." in parsed.parts:
        raise ValueError("literal canonical absolute path required")
    return Path(value)


def _below(path, roots):
    return any(path != root and path.is_relative_to(root) for root in roots)


def _stat(path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _check_parents(path):
    """Reject links/special files in every existing ancestor, even inside roots."""
    for parent in reversed(path.parents):
        info = _stat(parent)
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise ValueError("destination parent must be a real directory: " + str(parent))


def _open_directory(path):
    """Use no-follow dirfds also at mutation time, not only during preflight."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_json(path):
    _check_parents(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
            raise ValueError("bounded regular JSON file required")
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("JSON exceeded byte limit")
    return json.loads(raw, object_pairs_hook=_unique_object)


def load_plan(path):
    return _read_json(_path(str(path)))


def _file_hash(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_COPY_BYTES:
            raise ValueError("copy requires regular file at most 2 GiB")
        checksum, size = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(8 << 20), b""):
            size += len(block)
            if size > MAX_COPY_BYTES:
                raise ValueError("file exceeded hash byte limit while verifying")
            checksum.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("file changed while verifying")
    return checksum.hexdigest(), after.st_size


def _operations(plan):
    required = {"schema", "roots", "directories", "links"}
    if (not isinstance(plan, dict) or not required <= set(plan) <= required | {"copy_files"}
            or plan["schema"] != SCHEMA):
        raise ValueError("exact versioned catalog plan required; unknown keys forbidden")
    for name in ("roots", "directories", "links", "copy_files"):
        if not isinstance(plan.get(name, []), list) or len(plan.get(name, [])) > 1000:
            raise ValueError("bounded operation lists required")
    roots = [_path(value) for value in plan["roots"]]
    if not roots or len(set(roots)) != len(roots) or any(str(root) in _BROAD_ROOTS for root in roots):
        raise ValueError("distinct bounded owned roots required")
    if any(left != right and left.is_relative_to(right) for left in roots for right in roots):
        raise ValueError("owned roots must not overlap")
    directories = [_path(value) for value in plan["directories"]]
    if any(not _below(path, roots) for path in directories):
        raise ValueError("directory outside owned roots")
    operations = [dict(kind="directory", path=str(path)) for path in roots + directories]
    for key, kind, fields in (("links", "link", {"path", "target"}),
            ("copy_files", "copy_file", {"path", "target", "sha256"})):
        for value in plan.get(key, []):
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError("exact operation keys required")
            path, target = _path(value["path"]), _path(value["target"])
            if not _below(path, roots):
                raise ValueError("destination outside owned roots")
            if target in roots or _below(target, roots):
                raise ValueError("source cannot be under an output root")
            if kind == "copy_file" and (not isinstance(value["sha256"], str)
                    or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None):
                raise ValueError("copy requires expected SHA-256")
            operations.append(dict(kind=kind, **value))
    paths = [operation["path"] for operation in operations]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate/colliding destination")
    return roots, sorted(operations, key=lambda row: (row["kind"] != "directory", len(Path(row["path"]).parts), row["path"]))


def _verify_operation(operation, roots):
    path = Path(operation["path"])
    _check_parents(path)
    info = _stat(path)
    binding = dict(operation)
    if operation["kind"] == "directory":
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise ValueError("directory collision: " + str(path))
    else:
        target = Path(operation["target"])
        resolved = target.resolve(strict=True)
        if resolved in roots or _below(resolved, roots):
            raise ValueError("resolved source cannot be under an output root")
        target_info = resolved.stat()
        if not (stat.S_ISREG(target_info.st_mode) or stat.S_ISDIR(target_info.st_mode)):
            raise ValueError("source must be a regular file or directory")
        binding["resolved_target"] = str(resolved)
        if operation["kind"] == "link":
            binding["target_kind"] = "directory" if stat.S_ISDIR(target_info.st_mode) else "file"
            if info is not None and (not stat.S_ISLNK(info.st_mode) or os.readlink(path) != str(target)):
                raise ValueError("link destination collision: " + str(path))
        else:
            actual, size = _file_hash(resolved)
            if actual != operation["sha256"]:
                raise ValueError("copy source SHA-256 mismatch")
            binding["bytes"] = size
            if info is not None:
                if not stat.S_ISREG(info.st_mode) or _file_hash(path) != (actual, size):
                    raise ValueError("copy destination collision: " + str(path))
    return binding, info is not None


def _receipt(plan, bindings):
    value = dict(schema=RECEIPT_SCHEMA, plan_sha256=digest(plan), operations=bindings,
        original_artifacts_modified=False, compute_mutated=False)
    return dict(value, receipt_sha256=digest(value))


def preflight(plan, receipt=None):
    roots, operations = _operations(plan)
    planned_dirs = {Path(row["path"]) for row in operations if row["kind"] == "directory"}
    bindings, exists = [], []
    for operation in operations:
        path = Path(operation["path"])
        if not path.parent.is_dir() and path.parent not in planned_dirs:
            raise ValueError("missing parent must be an explicit planned directory: " + str(path.parent))
        binding, present = _verify_operation(operation, roots)
        bindings.append(binding)
        exists.append(present)
    receipt_path = _path(str(receipt)) if receipt is not None else None
    expected = _receipt(plan, bindings)
    if receipt_path is not None:
        if not _below(receipt_path, roots) or str(receipt_path) in {row["path"] for row in operations}:
            raise ValueError("receipt must be a distinct file within owned roots")
        _check_parents(receipt_path)
        if not receipt_path.parent.is_dir() and receipt_path.parent not in planned_dirs:
            raise ValueError("receipt parent must already exist or be explicitly planned")
        if _stat(receipt_path) is not None:
            actual = _read_json(receipt_path)
            if actual != expected or not all(exists):
                raise ValueError("existing immutable receipt or live catalog state differs")
    return roots, operations, bindings, exists, receipt_path, expected


def _create(operation, binding):
    path = Path(operation["path"])
    parent = _open_directory(path.parent)
    try:
        if operation["kind"] == "directory":
            os.mkdir(path.name, 0o755, dir_fd=parent)
        elif operation["kind"] == "link":
            os.symlink(operation["target"], path.name, dir_fd=parent)
        else:
            source_fd = os.open(binding["resolved_target"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(source_fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("copy source changed type")
                output_fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o444, dir_fd=parent)
                with os.fdopen(output_fd, "wb") as output:
                    checksum, size = hashlib.sha256(), 0
                    for block in iter(lambda: source.read(8 << 20), b""):
                        size += len(block)
                        if size > MAX_COPY_BYTES:
                            raise ValueError("copy grew beyond byte limit; partial file retained")
                        output.write(block)
                        checksum.update(block)
                    output.flush()
                    os.fsync(output.fileno())
                if checksum.hexdigest() != operation["sha256"] or size != binding["bytes"]:
                    raise ValueError("copy changed during transfer; partial file retained, no receipt")
        os.fsync(parent)
    finally:
        os.close(parent)


def apply_plan(plan, *, execute=False, receipt=None):
    if type(execute) is not bool:
        raise ValueError("execute must be explicit boolean")
    roots, operations, bindings, exists, receipt_path, expected = preflight(plan, receipt)
    actions = [dict(kind=row["kind"], path=row["path"], status="existing" if present else "create")
        for row, present in zip(operations, exists)]
    if execute:
        for operation, binding in zip(operations, bindings):
            current, present = _verify_operation(operation, roots)
            if current != binding:
                raise ValueError("source identity changed after preflight")
            if not present:
                _create(operation, binding)
            actual, present = _verify_operation(operation, roots)
            if not present or actual != binding:
                raise ValueError("post-create verification failed")
        if receipt_path is not None and _stat(receipt_path) is None:
            parent = _open_directory(receipt_path.parent)
            try:
                fd = os.open(receipt_path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o444, dir_fd=parent)
                with os.fdopen(fd, "wb") as output:
                    output.write(canonical(expected) + b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.fsync(parent)
            finally:
                os.close(parent)
        # Validate receipt bytes and all objects after apply, including an existing receipt.
        preflight(plan, receipt)
    return dict(schema=SCHEMA, mode="applied" if execute else "dry_run", plan_sha256=digest(plan),
        actions=actions, receipt=str(receipt_path) if receipt_path else None,
        receipt_sha256=expected["receipt_sha256"], original_artifacts_modified=False, compute_mutated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    print(json.dumps(apply_plan(load_plan(args.plan), execute=args.execute, receipt=args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
