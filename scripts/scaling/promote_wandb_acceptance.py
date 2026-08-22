#!/usr/bin/env python3
"""Atomically promote a passing job-scoped W&B smoke record.

The destination is a campaign prerequisite.  Publication is create-only: an
existing byte-identical record is accepted for idempotency, while a different
record is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


class PromotionError(RuntimeError):
    pass


REPORT_SCHEMA = "fast_roborender.wandb_online_smoke.v2"
AUTHORIZED_EMAIL = "ldu@nvidia.com"
AUTHORIZED_ENTITY = "nvidia"
AUTHORIZED_SCOPE = "team"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise PromotionError(f"source contains duplicate JSON key {key!r}")
        document[key] = value
    return document


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_authorized_request(
    *, expected_email: str, expected_entity: str, authorization_scope: str
) -> None:
    if expected_email != AUTHORIZED_EMAIL:
        raise PromotionError("expected W&B account is outside the authorized scope")
    if expected_entity != AUTHORIZED_ENTITY:
        raise PromotionError("expected W&B entity is outside the authorized scope")
    if authorization_scope != AUTHORIZED_SCOPE:
        raise PromotionError("unsupported W&B authorization scope")


def load_and_validate(
    source: Path,
    *,
    expected_email: str,
    expected_entity: str,
    authorization_scope: str,
) -> bytes:
    validate_authorized_request(
        expected_email=expected_email,
        expected_entity=expected_entity,
        authorization_scope=authorization_scope,
    )
    if source.is_symlink():
        raise PromotionError(f"source must not be a symbolic link: {source}")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise PromotionError(f"source is not a non-empty regular file: {source}")
    payload = source.read_bytes()
    try:
        document: Any = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except PromotionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError("source is not valid JSON") from exc
    if not isinstance(document, dict):
        raise PromotionError("source must contain one JSON object")
    expected = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "account_email": expected_email,
        "entity": expected_entity,
        "authorization_scope": authorization_scope,
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
    }
    for key, value in expected.items():
        if document.get(key) != value or type(document.get(key)) is not type(value):
            raise PromotionError(f"W&B smoke record {key} mismatch")
    entity = document.get("entity")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entity) is None:
        raise PromotionError("W&B smoke record entity is malformed")
    observed = document.get("observed_keys")
    required = {"validation/scalar", "validation/image", "validation/video"}
    if (
        not isinstance(observed, list)
        or not all(isinstance(key, str) for key in observed)
        or len(observed) != len(set(observed))
        or not required.issubset(observed)
    ):
        raise PromotionError("W&B smoke record lacks scalar/image/video readback")
    return payload


def promote(
    source: Path,
    destination: Path,
    *,
    expected_email: str,
    expected_entity: str,
    authorization_scope: str,
) -> str:
    payload = load_and_validate(
        source,
        expected_email=expected_email,
        expected_entity=expected_entity,
        authorization_scope=authorization_scope,
    )
    if destination.is_symlink():
        raise PromotionError(f"destination must not be a symbolic link: {destination}")
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if destination.resolve(strict=False) == source.resolve(strict=True):
        raise PromotionError("job-scoped source and promoted destination must differ")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file():
                raise PromotionError(
                    f"existing destination is not a regular file: {destination}"
                ) from None
            if destination.read_bytes() != payload:
                raise PromotionError(
                    f"refusing to replace different W&B acceptance: {destination}"
                ) from None
        try:
            directory_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # The hard-link is the atomic no-clobber barrier. Some network
            # filesystems do not implement directory fsync.
            pass
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--expected-entity", required=True)
    parser.add_argument(
        "--authorization-scope", required=True, choices=(AUTHORIZED_SCOPE,)
    )
    args = parser.parse_args()
    try:
        digest = promote(
            args.source,
            args.destination,
            expected_email=args.expected_email,
            expected_entity=args.expected_entity,
            authorization_scope=args.authorization_scope,
        )
    except (OSError, PromotionError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(f"WANDB_ACCEPTANCE={args.destination}")
    print(f"WANDB_ACCEPTANCE_SHA256={digest}")


if __name__ == "__main__":
    main()
