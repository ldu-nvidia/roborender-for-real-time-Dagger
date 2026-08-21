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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_and_validate(source: Path, expected_email: str) -> bytes:
    if source.is_symlink():
        raise PromotionError(f"source must not be a symbolic link: {source}")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise PromotionError(f"source is not a non-empty regular file: {source}")
    payload = source.read_bytes()
    try:
        document: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError("source is not valid JSON") from exc
    if not isinstance(document, dict):
        raise PromotionError("source must contain one JSON object")
    expected = {
        "schema": "fast_roborender.wandb_online_smoke.v1",
        "status": "PASS",
        "account_email": expected_email,
        "personal": True,
    }
    for key, value in expected.items():
        if document.get(key) != value or type(document.get(key)) is not type(value):
            raise PromotionError(f"W&B smoke record {key} mismatch")
    entity = document.get("entity")
    if not isinstance(entity, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entity) is None:
        raise PromotionError("W&B smoke record entity is malformed")
    observed = document.get("observed_keys")
    required = {"validation/scalar", "validation/image", "validation/video"}
    if not isinstance(observed, list) or not required.issubset(observed):
        raise PromotionError("W&B smoke record lacks scalar/image/video readback")
    return payload


def promote(source: Path, destination: Path, expected_email: str) -> str:
    payload = load_and_validate(source, expected_email)
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
    args = parser.parse_args()
    try:
        digest = promote(args.source, args.destination, args.expected_email)
    except (OSError, PromotionError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(f"WANDB_ACCEPTANCE={args.destination}")
    print(f"WANDB_ACCEPTANCE_SHA256={digest}")


if __name__ == "__main__":
    main()
