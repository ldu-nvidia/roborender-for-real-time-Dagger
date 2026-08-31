"""Keep the portable release free of deployment state and credentials."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
MAX_VERSIONED_FILE_BYTES = 1_000_000
BANNED_SUBSTRINGS = (
    "/mnt/" + "data1",
    "/mnt/" + "data2",
    "/users/" + "ldu",
    "ldu" + "@nvidia.com",
    "zijiandu" + "@asu.edu",
    "wandb" + "_v1_",
    "cosmos" + "-solution",
)
TOKEN_PATTERNS = (
    re.compile(r"\bgh" + "p_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub" + "_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAK" + "IA[0-9A-Z]{16}\b"),
)


def _public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
    ]


def test_public_tree_has_no_private_mounts_or_known_credentials() -> None:
    offenders: list[str] = []
    for path in _public_files():
        relative = path.relative_to(ROOT)
        assert path.stat().st_size <= MAX_VERSIONED_FILE_BYTES, relative
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise AssertionError(f"binary file is not allowed in public tree: {relative}") from exc
        if any(marker.lower() in text.lower() for marker in BANNED_SUBSTRINGS):
            offenders.append(str(relative))
            continue
        if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
            offenders.append(str(relative))
    assert not offenders, f"private material detected in: {sorted(offenders)}"
