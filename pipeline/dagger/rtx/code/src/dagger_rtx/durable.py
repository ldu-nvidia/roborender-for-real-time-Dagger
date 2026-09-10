from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import canonical_json


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


class DurableJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def append(self, event: dict[str, Any]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self.path, flags, 0o600)
        try:
            os.write(fd, canonical_json(event) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(self.path.parent)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"corrupt journal line {line_no}: {self.path}") from exc
        return records
