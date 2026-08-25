#!/usr/bin/env python3
"""Fail-closed verification of the exact files consumed by paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROVENANCE_KEYS = {
    "source",
    "source_sha256",
    "output",
    "output_sha256",
    "removed_keys",
    "environment_state_restored",
}
REMOVABLE_CHECKPOINT_KEYS = {
    "env_state_dict",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
}


class EvaluationInputError(RuntimeError):
    """A staged checkpoint/config no longer has its contracted identity."""


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvaluationInputError(f"{label} must be a lowercase SHA-256")
    return value


def _path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise EvaluationInputError(f"{label} must be a normalized absolute path")
    return path


def _read_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationInputError(f"cannot safely open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationInputError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != before.st_size:
            raise EvaluationInputError(f"{label} changed while it was read")
        return payload, before
    finally:
        os.close(descriptor)


def _digest_regular(path: Path, label: str) -> tuple[str, os.stat_result]:
    """Digest a potentially large regular file without retaining it in memory."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationInputError(f"cannot safely open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationInputError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        observed_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                observed_bytes += len(chunk)
                digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or observed_bytes != before.st_size:
            raise EvaluationInputError(f"{label} changed while it was read")
        return digest.hexdigest(), before
    finally:
        os.close(descriptor)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInputError(f"duplicate provenance key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EvaluationInputError(f"non-finite provenance value {value!r}")


def _load_provenance(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except EvaluationInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError("staged checkpoint provenance is invalid JSON") from exc
    if not isinstance(document, Mapping) or set(document) != PROVENANCE_KEYS:
        raise EvaluationInputError("staged checkpoint provenance fields are not exact")
    return dict(document)


def _teacher_from_config(payload: bytes) -> str:
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvaluationInputError("staged evaluation config is invalid YAML") from exc
    if not isinstance(document, Mapping):
        raise EvaluationInputError("staged evaluation config must contain a mapping")
    teacher = document.get("teacher_actor_path")
    if not isinstance(teacher, str) or not Path(teacher).is_absolute():
        raise EvaluationInputError(
            "staged evaluation config teacher_actor_path must be absolute"
        )
    return teacher


def verify_staged_inputs(
    *,
    checkpoint_source: str | Path,
    expected_checkpoint_sha256: str,
    staged_checkpoint: str | Path,
    provenance: str | Path,
    evaluation_config: str | Path,
    staged_config: str | Path,
    expected_config_sha256: str,
    teacher_checkpoint: str | Path,
) -> dict[str, Any]:
    """Verify source and staged identities in one fail-closed pass."""

    source = _path(checkpoint_source, "checkpoint source")
    staged = _path(staged_checkpoint, "staged checkpoint")
    provenance_path = _path(provenance, "staged checkpoint provenance")
    config_source = _path(evaluation_config, "evaluation config")
    config_staged = _path(staged_config, "staged evaluation config")
    teacher = _path(teacher_checkpoint, "teacher checkpoint")
    checkpoint_sha = _sha(
        expected_checkpoint_sha256, "expected checkpoint SHA-256"
    )
    config_sha = _sha(expected_config_sha256, "expected config SHA-256")

    source_sha, _ = _digest_regular(source, "checkpoint source")
    staged_sha, _ = _digest_regular(staged, "staged checkpoint")
    provenance_bytes, _ = _read_regular(
        provenance_path, "staged checkpoint provenance"
    )
    config_source_bytes, _ = _read_regular(config_source, "evaluation config")
    config_staged_bytes, config_staged_stat = _read_regular(
        config_staged, "staged evaluation config"
    )
    if stat.S_IMODE(config_staged_stat.st_mode) != 0o444:
        raise EvaluationInputError(
            "staged evaluation config mode must be exactly 0444"
        )

    if source_sha != checkpoint_sha:
        raise EvaluationInputError("checkpoint source SHA-256 mismatch")
    if _digest(config_source_bytes) != config_sha:
        raise EvaluationInputError("evaluation config source SHA-256 mismatch")
    if _digest(config_staged_bytes) != config_sha:
        raise EvaluationInputError("staged evaluation config SHA-256 mismatch")
    if config_staged_bytes != config_source_bytes:
        raise EvaluationInputError("staged/source evaluation config bytes differ")
    if _teacher_from_config(config_staged_bytes) != str(teacher):
        raise EvaluationInputError(
            "staged evaluation config teacher path differs from the pinned mount target"
        )

    row = _load_provenance(provenance_bytes)
    try:
        resolved_source = str(source.resolve(strict=True))
    except OSError as exc:
        raise EvaluationInputError("cannot resolve checkpoint source") from exc
    expected = {
        "source": resolved_source,
        "source_sha256": checkpoint_sha,
        "output": str(staged),
        "output_sha256": staged_sha,
        "environment_state_restored": False,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise EvaluationInputError("staged checkpoint provenance mismatch")
    removed = row.get("removed_keys")
    if (
        not isinstance(removed, list)
        or any(not isinstance(key, str) for key in removed)
        or len(removed) != len(set(removed))
        or any(key not in REMOVABLE_CHECKPOINT_KEYS for key in removed)
    ):
        raise EvaluationInputError("staged checkpoint removed_keys are invalid")
    return {
        "status": "PASS",
        "checkpoint_source_sha256": checkpoint_sha,
        "staged_checkpoint_sha256": staged_sha,
        "evaluation_config_sha256": config_sha,
        "teacher_checkpoint": str(teacher),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--staged-checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--staged-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = verify_staged_inputs(
            checkpoint_source=arguments.checkpoint_source,
            expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
            staged_checkpoint=arguments.staged_checkpoint,
            provenance=arguments.provenance,
            evaluation_config=arguments.evaluation_config,
            staged_config=arguments.staged_config,
            expected_config_sha256=arguments.expected_config_sha256,
            teacher_checkpoint=arguments.teacher_checkpoint,
        )
    except EvaluationInputError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
