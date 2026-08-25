#!/usr/bin/env python3
"""Content-bound identity for the Image3F serving semantics."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RENDERER_IDENTITY_SCHEMA = "fast_roborender.image3f_renderer_identity.v1"
REFERENCE_PROVENANCE_SCHEMA = "image3f.reference_provenance.v1"
PROTOCOL_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RENDERER_IDENTITY_KEYS = {
    "schema",
    "runtime_identity_sha256",
    "server_script_sha256",
    "protocol_version",
    "reference_provenance_schema",
}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def regular_file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    if not stat.S_ISREG(source.stat(follow_symlinks=False).st_mode):
        raise RuntimeError(f"renderer server script must be a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_renderer_identity(
    runtime_identity: Mapping[str, Any],
    *,
    server_script: str | Path,
    expected_server_script_sha256: str | None = None,
) -> dict[str, object]:
    """Bind model/runtime identity to the exact server implementation."""

    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise RuntimeError("renderer runtime identity is missing or malformed")
    script_sha256 = regular_file_sha256(server_script)
    if expected_server_script_sha256 is not None:
        if not _SHA256_RE.fullmatch(expected_server_script_sha256):
            raise RuntimeError("expected renderer server script SHA-256 is malformed")
        if script_sha256 != expected_server_script_sha256:
            raise RuntimeError(
                "renderer server script SHA-256 changed between planning and startup"
            )
    return {
        "schema": RENDERER_IDENTITY_SCHEMA,
        "runtime_identity_sha256": canonical_json_sha256(runtime_identity),
        "server_script_sha256": script_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "reference_provenance_schema": REFERENCE_PROVENANCE_SCHEMA,
    }


def validate_renderer_identity(
    identity: object,
    *,
    runtime_identity: Mapping[str, Any],
    expected_server_script_sha256: str,
) -> dict[str, object]:
    """Validate an endpoint identity against runtime and planned server bytes."""

    if not isinstance(identity, Mapping) or set(identity) != _RENDERER_IDENTITY_KEYS:
        raise RuntimeError("renderer identity is missing, malformed, or has unknown fields")
    expected = {
        "schema": RENDERER_IDENTITY_SCHEMA,
        "runtime_identity_sha256": canonical_json_sha256(runtime_identity),
        "server_script_sha256": expected_server_script_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "reference_provenance_schema": REFERENCE_PROVENANCE_SCHEMA,
    }
    normalized = dict(identity)
    if normalized != expected:
        raise RuntimeError(
            f"renderer identity mismatch: expected {expected!r}, got {normalized!r}"
        )
    return normalized
