#!/usr/bin/env python3
"""Publish and verify the post-allocation OCI learner success marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "fast_roborender.round_dagger.remote_learner_success.v3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "macro_round",
    "learner_code_sha256",
    "learner_config_sha256",
    "learner_identity_sha256",
    "result_path",
    "result_sha256",
    "completion_attestation_path",
    "completion_attestation_sha256",
}


class RemoteLearnerSuccessError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _regular(path: str | os.PathLike[str], label: str) -> Path:
    result = Path(os.path.abspath(Path(path).expanduser()))
    if result.is_symlink() or not result.is_file():
        raise RemoteLearnerSuccessError(f"{label} is missing or unsafe")
    return result


def _strict_load(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RemoteLearnerSuccessError("success marker has duplicate keys")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteLearnerSuccessError(
            f"cannot read success marker ({type(exc).__name__})"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != _KEYS:
        raise RemoteLearnerSuccessError("success marker key-set mismatch")
    return dict(value)


def _expected(
    *,
    result: Path,
    completion_attestation: Path,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    learner_identity_sha256: str,
) -> dict[str, Any]:
    if _TOKEN.fullmatch(campaign_id) is None:
        raise RemoteLearnerSuccessError("campaign ID is malformed")
    if type(macro_round) is not int or macro_round < 0:
        raise RemoteLearnerSuccessError("macro round is invalid")
    for digest in (
        learner_code_sha256,
        learner_config_sha256,
        learner_identity_sha256,
    ):
        if _SHA256.fullmatch(digest) is None:
            raise RemoteLearnerSuccessError("learner identity digest is malformed")
    if result.name in {"", ".", ".."}:
        raise RemoteLearnerSuccessError("result basename is unsafe")
    if (
        completion_attestation.name in {"", ".", ".."}
        or completion_attestation.parent != result.parent
    ):
        raise RemoteLearnerSuccessError(
            "completion attestation must be co-located with the result"
        )
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_code_sha256": learner_code_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "result_path": result.name,
        "result_sha256": _sha256_file(result),
        "completion_attestation_path": completion_attestation.name,
        "completion_attestation_sha256": _sha256_file(completion_attestation),
    }


def publish(
    marker: str | os.PathLike[str],
    result: str | os.PathLike[str],
    completion_attestation: str | os.PathLike[str],
    *,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    learner_identity_sha256: str,
) -> dict[str, Any]:
    result_path = _regular(result, "learner result")
    attestation_path = _regular(completion_attestation, "learner completion attestation")
    document = _expected(
        result=result_path,
        completion_attestation=attestation_path,
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_code_sha256=learner_code_sha256,
        learner_config_sha256=learner_config_sha256,
        learner_identity_sha256=learner_identity_sha256,
    )
    marker_path = Path(os.path.abspath(Path(marker).expanduser()))
    if marker_path.is_symlink():
        raise RemoteLearnerSuccessError("success marker is a symlink")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    if marker_path.exists():
        if not marker_path.is_file() or marker_path.read_bytes() != payload:
            raise RemoteLearnerSuccessError("different success marker already exists")
        if marker_path.stat().st_mode & 0o222:
            raise RemoteLearnerSuccessError(
                "existing success marker must be read-only"
            )
        return document
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker_path.name}.", dir=marker_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, marker_path, follow_symlinks=False)
        except FileExistsError:
            if marker_path.is_symlink() or marker_path.read_bytes() != payload:
                raise RemoteLearnerSuccessError("competing success marker differs")
        os.chmod(marker_path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def verify(
    marker: str | os.PathLike[str],
    result: str | os.PathLike[str],
    completion_attestation: str | os.PathLike[str],
    *,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    learner_identity_sha256: str,
) -> dict[str, Any]:
    marker_path = _regular(marker, "success marker")
    result_path = _regular(result, "learner result")
    attestation_path = _regular(completion_attestation, "learner completion attestation")
    observed = _strict_load(marker_path)
    expected = _expected(
        result=result_path,
        completion_attestation=attestation_path,
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_code_sha256=learner_code_sha256,
        learner_config_sha256=learner_config_sha256,
        learner_identity_sha256=learner_identity_sha256,
    )
    if observed != expected:
        raise RemoteLearnerSuccessError("success marker/result lineage mismatch")
    if marker_path.stat().st_mode & 0o222:
        raise RemoteLearnerSuccessError("success marker must be read-only")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("publish", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--marker", required=True)
        command.add_argument("--result", required=True)
        command.add_argument("--completion-attestation", required=True)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--macro-round", required=True, type=int)
        command.add_argument("--learner-code-sha256", required=True)
        command.add_argument("--learner-config-sha256", required=True)
        command.add_argument("--learner-identity-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    function = publish if arguments.command == "publish" else verify
    document = function(
        arguments.marker,
        arguments.result,
        arguments.completion_attestation,
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
        learner_code_sha256=arguments.learner_code_sha256,
        learner_config_sha256=arguments.learner_config_sha256,
        learner_identity_sha256=arguments.learner_identity_sha256,
    )
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RemoteLearnerSuccessError as exc:
        print(f"remote learner success validation failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
