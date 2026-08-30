#!/usr/bin/env python3
"""Fail-closed file transfer through a Lepton Kubernetes Web terminal.

This utility exists for the case where a Lepton Dev Pod is reachable through
the authenticated Web-terminal API but not through SSH.  It deliberately
transfers one regular file at a time.  Remote paths are confined to
``/workspace/projects/roborender`` and publication is an atomic hard-link from
a same-directory ``.incoming`` file.  Existing destinations are never
replaced.  ``--adopt-existing`` only accepts an already-published byte-for-byte
match.

Authentication is obtained from the local Lepton SDK profile.  There is no
command-line option for a bearer token or cookie, so credentials cannot be
placed in process arguments or ordinary logs.
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import hashlib
import json
import os
import posixpath
import re
import secrets
import selectors
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Protocol, Sequence
from urllib.parse import quote, urlsplit, urlunsplit


REMOTE_ROOT = "/workspace/projects/roborender"
DEFAULT_DEPLOYMENT = "leo-cosmos-h200-dev"
DEFAULT_CHUNK_BYTES = 48 * 1024
MAX_CHUNK_BYTES = 192 * 1024
FRAME_PREFIX = b"LXFT1"
STREAM_FRAME_PREFIX = b"LXFS1"
DOWNLOAD_FRAME_PREFIX = b"LXFD1"
FRAME_START = b"\x1e"
FRAME_END = b"\x1f"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class TransferError(RuntimeError):
    """A transfer was unsafe, incomplete, or could not be verified."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size < 0:
            raise TransferError("file size is invalid")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise TransferError("file SHA-256 is invalid")


@dataclasses.dataclass(frozen=True)
class ConnectionInfo:
    url: str
    authorization: str = dataclasses.field(repr=False)
    origin: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise TransferError("Lepton Web-terminal URL is malformed or unsafe")
        if (
            not self.authorization
            or "\r" in self.authorization
            or "\n" in self.authorization
        ):
            raise TransferError("Lepton SDK did not provide safe authentication")
        origin = urlsplit(self.origin)
        if (
            origin.scheme not in {"http", "https"}
            or not origin.netloc
            or origin.username is not None
            or origin.password is not None
            or origin.query
            or origin.fragment
        ):
            raise TransferError("Lepton SDK origin is malformed or unsafe")


@dataclasses.dataclass(frozen=True)
class CommandResult:
    pid: int
    worker_pid: int
    data: tuple[bytes, ...]


class TerminalTransport(Protocol):
    def send(self, payload: bytes) -> None: ...

    def receive(self, timeout_seconds: float) -> bytes: ...

    def close(self) -> None: ...


class RemoteFileStore(Protocol):
    def inspect(self, path: str) -> FileIdentity | None: ...

    def create_incoming(self, path: str, mode: int) -> None: ...

    def append(self, path: str, payload: bytes) -> None: ...

    def read_chunk(self, path: str, offset: int, length: int) -> bytes: ...

    def publish(
        self, incoming: str, destination: str, expected: FileIdentity, mode: int
    ) -> None: ...

    def cleanup_incoming(self, path: str) -> None: ...


def resumable_incoming_path(destination: str, expected: FileIdentity) -> str:
    """Return the deterministic, destination-bound streaming stage path."""

    destination = validate_remote_path(destination)
    destination_id = hashlib.sha256(destination.encode("utf-8")).hexdigest()[:16]
    name = (
        f".incoming.lxfs1.{destination_id}.{expected.sha256[:24]}.{expected.size}"
    )
    return validate_remote_path(posixpath.join(posixpath.dirname(destination), name))


def validate_remote_path(value: str) -> str:
    """Return a canonical absolute path strictly below ``REMOTE_ROOT``."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise TransferError("remote path is empty or malformed")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TransferError("remote path contains a control character")
    parsed = PurePosixPath(value)
    normalized = posixpath.normpath(value)
    if not parsed.is_absolute() or normalized != value or ".." in parsed.parts:
        raise TransferError("remote path must be a canonical absolute path")
    if value == REMOTE_ROOT or not value.startswith(REMOTE_ROOT + "/"):
        raise TransferError(f"remote path must be below {REMOTE_ROOT}")
    if parsed.name in {"", ".", ".."}:
        raise TransferError("remote path has an unsafe basename")
    return value


def incoming_path(destination: str, nonce: str | None = None) -> str:
    destination = validate_remote_path(destination)
    token = nonce or secrets.token_hex(16)
    if _TOKEN.fullmatch(token) is None:
        raise TransferError("incoming-file nonce is malformed")
    candidate = posixpath.join(posixpath.dirname(destination), f".incoming.{token}")
    return validate_remote_path(candidate)


def _quote(value: str) -> str:
    return shlex.quote(value)


def _identity_from_record(record: bytes) -> FileIdentity | None:
    try:
        text = record.decode("ascii")
    except UnicodeError as exc:
        raise TransferError("remote file metadata was not ASCII") from exc
    if text == "ABSENT":
        return None
    parts = text.split(":")
    if len(parts) != 3 or parts[0] != "REGULAR":
        raise TransferError("remote file metadata is malformed")
    try:
        size = int(parts[1])
    except ValueError as exc:
        raise TransferError("remote file size is malformed") from exc
    return FileIdentity(size=size, sha256=parts[2])


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransferError("local source is unavailable or unsafe") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise TransferError("local source must be a regular non-symlink file")
    return descriptor, info


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns)


def _hash_handle(handle: BinaryIO) -> FileIdentity:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return FileIdentity(size=size, sha256=digest.hexdigest())


def _hash_regular_path(path: Path) -> FileIdentity:
    descriptor, _ = _open_regular(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return _hash_handle(handle)
    finally:
        os.close(descriptor)


class SentinelParser:
    """Incrementally parse nonce-bound control frames from noisy PTY output."""

    def __init__(self, nonce: str, *, max_frame_bytes: int = 2 * 1024 * 1024):
        if _TOKEN.fullmatch(nonce) is None:
            raise TransferError("command nonce is malformed")
        self._marker = FRAME_START + FRAME_PREFIX + b":" + nonce.encode("ascii") + b":"
        self._max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()

    def feed(self, payload: bytes) -> list[tuple[str, bytes]]:
        self._buffer.extend(payload)
        records: list[tuple[str, bytes]] = []
        while True:
            start = self._buffer.find(self._marker)
            if start < 0:
                keep = max(len(self._marker) - 1, 0)
                if len(self._buffer) > keep:
                    del self._buffer[: len(self._buffer) - keep]
                return records
            if start:
                del self._buffer[:start]
            end = self._buffer.find(FRAME_END, len(self._marker))
            if end < 0:
                if len(self._buffer) > self._max_frame_bytes:
                    raise TransferError("remote control frame exceeded its size limit")
                return records
            body = bytes(self._buffer[len(self._marker) : end])
            del self._buffer[: end + 1]
            kind_bytes, separator, encoded = body.partition(b":")
            if not separator:
                raise TransferError("remote control frame is malformed")
            try:
                kind = kind_bytes.decode("ascii")
            except UnicodeError as exc:
                raise TransferError("remote control-frame kind is malformed") from exc
            if kind not in {"PID", "WORKER", "DATA", "DONE"}:
                raise TransferError("remote control-frame kind is unknown")
            records.append((kind, encoded))


class RemoteShell:
    """Execute quiet, nonce-framed commands over one terminal transport."""

    def __init__(
        self,
        transport: TerminalTransport,
        *,
        command_timeout_seconds: float = 300.0,
        max_result_bytes: int = 2 * 1024 * 1024,
    ):
        if command_timeout_seconds <= 0:
            raise TransferError("command timeout must be positive")
        self.transport = transport
        self.command_timeout_seconds = command_timeout_seconds
        self.max_result_bytes = max_result_bytes

    def run(self, script: str, *, timeout_seconds: float | None = None) -> CommandResult:
        if not script or "\x00" in script:
            raise TransferError("remote command is empty or malformed")
        nonce = secrets.token_hex(16)
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        prefix = f"\\036{FRAME_PREFIX.decode()}:{nonce}:"
        suffix = "\\037\\n"
        wrapper = (
            "set +e; "
            f"__lxft_n={_quote(nonce)}; "
            "__lxft_emit() { __lxft_p=$(printf '%s' \"$1\" | base64 | tr -d '\\n'); "
            f"printf '{prefix}DATA:%s{suffix}' \"$__lxft_p\"; }}; "
            f"printf '{prefix}PID:%s{suffix}' \"$$\"; "
            f"__lxft_s=$(printf '%s' {_quote(encoded)} | base64 -d); "
            "( set -eu; eval \"$__lxft_s\" ) & __lxft_w=$!; "
            f"printf '{prefix}WORKER:%s{suffix}' \"$__lxft_w\"; "
            "wait \"$__lxft_w\"; __lxft_rc=$?; "
            f"printf '{prefix}DONE:%s{suffix}' \"$__lxft_rc\"; "
            "unset -f __lxft_emit; unset __lxft_n __lxft_s __lxft_w __lxft_rc\n"
        )
        self.transport.send(b"\x00" + wrapper.encode("utf-8"))
        parser = SentinelParser(nonce, max_frame_bytes=self.max_result_bytes * 2)
        deadline = time.monotonic() + (
            timeout_seconds if timeout_seconds is not None else self.command_timeout_seconds
        )
        pid: int | None = None
        worker_pid: int | None = None
        data: list[bytes] = []
        data_size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                suffix_text = f" (remote PID {pid})" if pid is not None else ""
                raise TransferError(f"Lepton remote command timed out{suffix_text}")
            chunk = self.transport.receive(remaining)
            if not chunk:
                raise TransferError("Lepton terminal closed before command completion")
            for kind, encoded_payload in parser.feed(chunk):
                if kind in {"PID", "WORKER"}:
                    try:
                        value = int(encoded_payload.decode("ascii"))
                    except (UnicodeError, ValueError) as exc:
                        raise TransferError("remote PID frame is malformed") from exc
                    if value <= 0:
                        raise TransferError("remote PID frame is invalid")
                    if kind == "PID":
                        if pid is not None:
                            raise TransferError("remote PID frame was duplicated")
                        pid = value
                    else:
                        if worker_pid is not None:
                            raise TransferError("remote worker PID frame was duplicated")
                        worker_pid = value
                elif kind == "DATA":
                    try:
                        decoded = base64.b64decode(encoded_payload, validate=True)
                    except (ValueError, base64.binascii.Error) as exc:
                        raise TransferError("remote DATA frame is not valid base64") from exc
                    data_size += len(decoded)
                    if data_size > self.max_result_bytes:
                        raise TransferError("remote command result exceeded its size limit")
                    data.append(decoded)
                else:
                    if pid is None or worker_pid is None:
                        raise TransferError("remote command completed without PID framing")
                    try:
                        return_code = int(encoded_payload.decode("ascii"))
                    except (UnicodeError, ValueError) as exc:
                        raise TransferError("remote completion frame is malformed") from exc
                    if return_code != 0:
                        raise TransferError(
                            f"remote command failed with status {return_code} (PID {worker_pid})"
                        )
                    return CommandResult(pid=pid, worker_pid=worker_pid, data=tuple(data))

    def close(self) -> None:
        self.transport.close()


class StreamSentinelParser:
    """Parse PID, READY, and DONE records for one raw upload stream."""

    def __init__(self, nonce: str, *, max_frame_bytes: int = 4096):
        if _TOKEN.fullmatch(nonce) is None:
            raise TransferError("stream nonce is malformed")
        self._marker = (
            FRAME_START
            + STREAM_FRAME_PREFIX
            + b":"
            + nonce.encode("ascii")
            + b":"
        )
        self._max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()

    def feed(self, payload: bytes) -> list[tuple[str, bytes]]:
        self._buffer.extend(payload)
        records: list[tuple[str, bytes]] = []
        while True:
            start = self._buffer.find(self._marker)
            if start < 0:
                keep = max(len(self._marker) - 1, 0)
                if len(self._buffer) > keep:
                    del self._buffer[: len(self._buffer) - keep]
                return records
            if start:
                del self._buffer[:start]
            end = self._buffer.find(FRAME_END, len(self._marker))
            if end < 0:
                if len(self._buffer) > self._max_frame_bytes:
                    raise TransferError("stream control frame exceeded its size limit")
                return records
            body = bytes(self._buffer[len(self._marker) : end])
            del self._buffer[: end + 1]
            kind_bytes, separator, value = body.partition(b":")
            if not separator:
                raise TransferError("stream control frame is malformed")
            try:
                kind = kind_bytes.decode("ascii")
            except UnicodeError as exc:
                raise TransferError("stream control-frame kind is malformed") from exc
            if kind not in {"PID", "READY", "DONE"}:
                raise TransferError("stream control-frame kind is unknown")
            records.append((kind, value))


def _stream_remote_script(
    destination: str,
    incoming: str,
    expected: FileIdentity,
    mode: int,
    nonce: str,
) -> str:
    """Build the fixed-count, raw/no-echo receiver executed in the remote PTY."""

    destination = validate_remote_path(destination)
    incoming = validate_remote_path(incoming)
    if posixpath.dirname(destination) != posixpath.dirname(incoming):
        raise TransferError("stream incoming file must be beside its destination")
    if not posixpath.basename(incoming).startswith(".incoming.lxfs1."):
        raise TransferError("stream incoming-file name is unsafe")
    if type(mode) is not int or not 0 <= mode <= 0o777:
        raise TransferError("remote file mode is invalid")
    if _TOKEN.fullmatch(nonce) is None:
        raise TransferError("stream nonce is malformed")
    root = _quote(REMOTE_ROOT)
    parent = _quote(posixpath.dirname(destination))
    destination_q = _quote(destination)
    incoming_q = _quote(incoming)
    expected_size = str(expected.size)
    expected_sha = _quote(expected.sha256)
    prefix = f"\\036{STREAM_FRAME_PREFIX.decode()}:{nonce}:"
    suffix = "\\037\\n"
    # The script never reads until READY has been emitted after `stty raw -echo`.
    # A single G byte commits the sender to exactly `remaining` payload bytes; A
    # aborts and removes only this deterministic incoming file.
    return (
        "set +e\n"
        f"__root={root}\n"
        f"__parent={parent}\n"
        f"__dst={destination_q}\n"
        f"__incoming={incoming_q}\n"
        f"__expected_size={expected_size}\n"
        f"__expected_sha={expected_sha}\n"
        "__stty_state=\n"
        "__restore() { if test -n \"$__stty_state\"; then stty \"$__stty_state\" >/dev/null 2>&1 || true; __stty_state=; fi; }\n"
        f"__done() {{ __code=$1; __size=${{2:--}}; __sha=${{3:--}}; __restore; printf '{prefix}DONE:%s:%s:%s{suffix}' \"$__code\" \"$__size\" \"$__sha\"; exit 0; }}\n"
        "trap '__restore' EXIT\n"
        "trap '__restore; exit 129' HUP INT TERM\n"
        f"printf '{prefix}PID:%s{suffix}' \"$$\"\n"
        "test -d \"$__root\" && test ! -L \"$__root\" || __done 41\n"
        "__root_real=$(realpath -e -- \"$__root\") || __done 41\n"
        "test \"$__root_real\" = \"$__root\" || __done 41\n"
        "test -d \"$__parent\" || __done 41\n"
        "__parent_real=$(realpath -e -- \"$__parent\") || __done 41\n"
        "case \"$__parent_real\" in \"$__root\"|\"$__root\"/*) ;; *) __done 41 ;; esac\n"
        "test ! -L \"$__dst\" || __done 42\n"
        "test ! -e \"$__dst\" || __done 43\n"
        "test ! -L \"$__incoming\" || __done 44\n"
        "if test -e \"$__incoming\"; then\n"
        "  test -f \"$__incoming\" || __done 44\n"
        "else\n"
        "  umask 077\n"
        "  ( set -C; : > \"$__incoming\" ) || __done 44\n"
        "fi\n"
        "__offset=$(wc -c < \"$__incoming\" | tr -d '[:space:]') || __done 44\n"
        "case \"$__offset\" in ''|*[!0-9]*) __done 44 ;; esac\n"
        "if test \"$__offset\" -gt \"$__expected_size\"; then rm -f -- \"$__incoming\" || __done 44; __done 52; fi\n"
        "__prefix_sha=$(sha256sum -- \"$__incoming\" | awk '{print $1}') || __done 44\n"
        "__remaining=$((__expected_size - __offset))\n"
        "__stty_state=$(stty -g) || __done 45\n"
        "stty raw -echo || __done 45\n"
        f"printf '{prefix}READY:%s:%s:%s{suffix}' \"$__offset\" \"$__prefix_sha\" \"$__remaining\"\n"
        "__go=$(head -c 1)\n"
        "if test \"$__go\" != G; then rm -f -- \"$__incoming\"; __done 46; fi\n"
        "if test \"$__remaining\" -gt 0; then head -c \"$__remaining\" >> \"$__incoming\" || __done 47; fi\n"
        "__size=$(wc -c < \"$__incoming\" | tr -d '[:space:]') || __done 48\n"
        "__sha=$(sha256sum -- \"$__incoming\" | awk '{print $1}') || __done 48\n"
        "test \"$__size\" = \"$__expected_size\" || __done 48 \"$__size\" \"$__sha\"\n"
        "test \"$__sha\" = \"$__expected_sha\" || __done 48 \"$__size\" \"$__sha\"\n"
        f"chmod {mode:04o} \"$__incoming\" || __done 49 \"$__size\" \"$__sha\"\n"
        "ln -- \"$__incoming\" \"$__dst\" || __done 50 \"$__size\" \"$__sha\"\n"
        "rm -f -- \"$__incoming\" || __done 51 \"$__size\" \"$__sha\"\n"
        "__done 0 \"$__size\" \"$__sha\"\n"
    )


class StreamingUploader:
    """Send one exact-size file body through a single foreground PTY command."""

    def __init__(
        self,
        transport: TerminalTransport,
        *,
        chunk_bytes: int = 128 * 1024,
        setup_timeout_seconds: float = 60.0,
        completion_timeout_seconds: float = 300.0,
    ):
        if type(chunk_bytes) is not int or not 1024 <= chunk_bytes <= MAX_CHUNK_BYTES:
            raise TransferError("stream chunk size is invalid")
        if setup_timeout_seconds <= 0 or completion_timeout_seconds <= 0:
            raise TransferError("stream timeout must be positive")
        self.transport = transport
        self.chunk_bytes = chunk_bytes
        self.setup_timeout_seconds = setup_timeout_seconds
        self.completion_timeout_seconds = completion_timeout_seconds

    @staticmethod
    def _positive_pid(value: bytes) -> int:
        try:
            pid = int(value.decode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise TransferError("stream remote PID is malformed") from exc
        if pid <= 0:
            raise TransferError("stream remote PID is invalid")
        return pid

    @staticmethod
    def _ready(value: bytes, expected: FileIdentity) -> tuple[int, str, int]:
        try:
            offset_text, prefix_sha, remaining_text = value.decode("ascii").split(":")
            offset = int(offset_text)
            remaining = int(remaining_text)
        except (UnicodeError, ValueError) as exc:
            raise TransferError("stream READY frame is malformed") from exc
        if (
            offset < 0
            or offset > expected.size
            or remaining != expected.size - offset
            or _SHA256.fullmatch(prefix_sha) is None
        ):
            raise TransferError("stream READY frame is inconsistent")
        return offset, prefix_sha, remaining

    @staticmethod
    def _done(value: bytes, expected: FileIdentity, pid: int | None) -> None:
        try:
            code_text, size_text, sha = value.decode("ascii").split(":")
            code = int(code_text)
        except (UnicodeError, ValueError) as exc:
            raise TransferError("stream DONE frame is malformed") from exc
        suffix = f" (PID {pid})" if pid is not None else ""
        if code != 0:
            raise TransferError(f"stream receiver failed with status {code}{suffix}")
        try:
            observed = FileIdentity(int(size_text), sha)
        except (ValueError, TransferError) as exc:
            raise TransferError("stream DONE identity is malformed") from exc
        if observed != expected:
            raise TransferError("stream DONE identity does not match the source")

    def _records_until(
        self,
        parser: StreamSentinelParser,
        deadline: float,
        pending: collections.deque[tuple[str, bytes]],
    ) -> tuple[str, bytes]:
        while not pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransferError("stream receiver timed out")
            payload = self.transport.receive(remaining)
            if not payload:
                raise TransferError("Lepton terminal closed during stream upload")
            pending.extend(parser.feed(payload))
        return pending.popleft()

    def _abort_and_verify_cleanup(
        self,
        parser: StreamSentinelParser,
        pending: collections.deque[tuple[str, bytes]],
    ) -> None:
        self.transport.send(b"\x00A")
        deadline = time.monotonic() + self.completion_timeout_seconds
        while True:
            kind, value = self._records_until(parser, deadline, pending)
            if kind != "DONE":
                if kind == "PID":
                    raise TransferError("stream PID frame was duplicated during abort")
                raise TransferError("stream READY frame was duplicated during abort")
            if value != b"46:-:-":
                raise TransferError("stream receiver did not attest exact-stage cleanup")
            return

    def upload(
        self,
        handle: BinaryIO,
        destination: str,
        incoming: str,
        expected: FileIdentity,
        *,
        mode: int = 0o644,
    ) -> None:
        nonce = secrets.token_hex(16)
        script = _stream_remote_script(destination, incoming, expected, mode, nonce)
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        launcher = (
            f"__lxfs_s=$(printf '%s' {_quote(encoded)} | base64 -d); "
            "bash -c \"$__lxfs_s\"; __lxfs_rc=$?; "
            "unset __lxfs_s __lxfs_rc\n"
        )
        parser = StreamSentinelParser(nonce)
        pending: collections.deque[tuple[str, bytes]] = collections.deque()
        self.transport.send(b"\x00" + launcher.encode("utf-8"))
        deadline = time.monotonic() + self.setup_timeout_seconds
        pid: int | None = None
        ready: tuple[int, str, int] | None = None
        while ready is None:
            kind, value = self._records_until(parser, deadline, pending)
            if kind == "PID":
                if pid is not None:
                    raise TransferError("stream PID frame was duplicated")
                pid = self._positive_pid(value)
            elif kind == "READY":
                if pid is None:
                    raise TransferError("stream READY arrived before its PID")
                ready = self._ready(value, expected)
            else:
                self._done(value, expected, pid)
                raise TransferError("stream receiver completed before READY")

        offset, remote_prefix_sha, remaining = ready
        handle.seek(0)
        prefix_digest = hashlib.sha256()
        prefix_remaining = offset
        while prefix_remaining:
            chunk = handle.read(min(self.chunk_bytes, prefix_remaining))
            if not chunk:
                self._abort_and_verify_cleanup(parser, pending)
                raise TransferError("local source became shorter before stream resume")
            prefix_digest.update(chunk)
            prefix_remaining -= len(chunk)
        if prefix_digest.hexdigest() != remote_prefix_sha:
            # The receiver removes only the deterministic incoming file before DONE.
            self._abort_and_verify_cleanup(parser, pending)
            raise TransferError("resumable remote prefix does not match the local source")

        self.transport.send(b"\x00G")
        sent = 0
        streamed_digest = prefix_digest.copy()
        while sent < remaining:
            chunk = handle.read(min(self.chunk_bytes, remaining - sent))
            if not chunk:
                raise TransferError("local source became shorter during stream upload")
            self.transport.send(b"\x00" + chunk)
            streamed_digest.update(chunk)
            sent += len(chunk)
        if sent != remaining or handle.read(1):
            raise TransferError("local source size changed during stream upload")
        streamed = FileIdentity(offset + sent, streamed_digest.hexdigest())
        if streamed != expected:
            raise TransferError("local source changed during stream upload")

        deadline = time.monotonic() + self.completion_timeout_seconds
        while True:
            kind, value = self._records_until(parser, deadline, pending)
            if kind == "DONE":
                self._done(value, expected, pid)
                return
            if kind == "PID":
                raise TransferError("stream PID frame was duplicated")
            raise TransferError("stream READY frame was duplicated")


def _download_remote_script(source: str, nonce: str) -> str:
    """Build a fixed-count raw sender for one immutable remote file."""

    source = validate_remote_path(source)
    if _TOKEN.fullmatch(nonce) is None:
        raise TransferError("download stream nonce is malformed")
    root = _quote(REMOTE_ROOT)
    parent = _quote(posixpath.dirname(source))
    source_q = _quote(source)
    prefix = f"\\036{DOWNLOAD_FRAME_PREFIX.decode()}:{nonce}:"
    # No newline follows a control frame: after READY, the next remote byte is
    # payload byte zero, and after the fixed payload count the next byte starts
    # DONE.  This keeps PTY newline conversion out of the binary protocol.
    suffix = "\\037"
    return (
        "set +e\n"
        f"__root={root}\n"
        f"__parent={parent}\n"
        f"__src={source_q}\n"
        "__stty_state=\n"
        "__restore() { if test -n \"$__stty_state\"; then stty \"$__stty_state\" >/dev/null 2>&1 || true; __stty_state=; fi; }\n"
        f"__done() {{ __code=$1; __size=${{2:--}}; __sha=${{3:--}}; printf '{prefix}DONE:%s:%s:%s{suffix}' \"$__code\" \"$__size\" \"$__sha\"; __restore; exit 0; }}\n"
        "trap '__restore' EXIT\n"
        "trap '__restore; exit 129' HUP INT TERM\n"
        "__stty_state=$(stty -g) || exit 45\n"
        "stty raw -echo || exit 45\n"
        f"printf '{prefix}PID:%s{suffix}' \"$$\"\n"
        "test -d \"$__root\" && test ! -L \"$__root\" || __done 41\n"
        "__root_real=$(realpath -e -- \"$__root\") || __done 41\n"
        "test \"$__root_real\" = \"$__root\" || __done 41\n"
        "test -d \"$__parent\" || __done 41\n"
        "__parent_real=$(realpath -e -- \"$__parent\") || __done 41\n"
        "case \"$__parent_real\" in \"$__root\"|\"$__root\"/*) ;; *) __done 41 ;; esac\n"
        "test ! -L \"$__src\" && test -f \"$__src\" || __done 42\n"
        "__size=$(wc -c < \"$__src\" | tr -d '[:space:]') || __done 43\n"
        "case \"$__size\" in ''|*[!0-9]*) __done 43 ;; esac\n"
        "__sha=$(sha256sum -- \"$__src\" | awk '{print $1}') || __done 43\n"
        f"printf '{prefix}READY:%s:%s{suffix}' \"$__size\" \"$__sha\"\n"
        "__go=$(head -c 1)\n"
        "if test \"$__go\" != G; then __done 46; fi\n"
        "head -c \"$__size\" -- \"$__src\" 2>/dev/null || __done 47 \"$__size\" \"$__sha\"\n"
        "__after_size=$(wc -c < \"$__src\" | tr -d '[:space:]') || __done 48\n"
        "__after_sha=$(sha256sum -- \"$__src\" | awk '{print $1}') || __done 48\n"
        "test \"$__after_size\" = \"$__size\" || __done 48 \"$__after_size\" \"$__after_sha\"\n"
        "test \"$__after_sha\" = \"$__sha\" || __done 48 \"$__after_size\" \"$__after_sha\"\n"
        "__done 0 \"$__size\" \"$__sha\"\n"
    )


class DownloadControlParser:
    """Parse download controls while retaining adjacent raw payload bytes."""

    def __init__(self, nonce: str):
        if _TOKEN.fullmatch(nonce) is None:
            raise TransferError("download stream nonce is malformed")
        self.marker = (
            FRAME_START
            + DOWNLOAD_FRAME_PREFIX
            + b":"
            + nonce.encode("ascii")
            + b":"
        )
        self.buffer = bytearray()

    def feed(self, payload: bytes) -> None:
        self.buffer.extend(payload)

    def pop(self, *, allow_noise: bool) -> tuple[str, bytes] | None:
        start = self.buffer.find(self.marker)
        if start < 0:
            if allow_noise:
                keep = max(len(self.marker) - 1, 0)
                if len(self.buffer) > keep:
                    del self.buffer[: len(self.buffer) - keep]
                return None
            if len(self.buffer) >= len(self.marker):
                raise TransferError("download stream emitted bytes outside its protocol")
            return None
        if start:
            if not allow_noise:
                raise TransferError("download stream emitted bytes outside its protocol")
            del self.buffer[:start]
        end = self.buffer.find(FRAME_END, len(self.marker))
        if end < 0:
            if len(self.buffer) > 4096:
                raise TransferError("download control frame exceeded its size limit")
            return None
        body = bytes(self.buffer[len(self.marker) : end])
        del self.buffer[: end + 1]
        kind_bytes, separator, value = body.partition(b":")
        if not separator:
            raise TransferError("download control frame is malformed")
        try:
            kind = kind_bytes.decode("ascii")
        except UnicodeError as exc:
            raise TransferError("download control kind is malformed") from exc
        if kind not in {"PID", "READY", "DONE"}:
            raise TransferError("download control kind is unknown")
        return kind, value


class StreamingDownloader:
    """Receive one exact remote file body through a raw PTY stream."""

    def __init__(
        self,
        transport: TerminalTransport,
        *,
        setup_timeout_seconds: float = 60.0,
        completion_timeout_seconds: float = 300.0,
    ):
        if setup_timeout_seconds <= 0 or completion_timeout_seconds <= 0:
            raise TransferError("download stream timeout must be positive")
        self.transport = transport
        self.setup_timeout_seconds = setup_timeout_seconds
        self.completion_timeout_seconds = completion_timeout_seconds

    @staticmethod
    def _pid(value: bytes) -> int:
        try:
            result = int(value.decode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise TransferError("download stream PID is malformed") from exc
        if result <= 0:
            raise TransferError("download stream PID is invalid")
        return result

    @staticmethod
    def _ready(value: bytes) -> FileIdentity:
        try:
            size_text, sha256 = value.decode("ascii").split(":")
            return FileIdentity(int(size_text), sha256)
        except (UnicodeError, ValueError, TransferError) as exc:
            raise TransferError("download READY identity is malformed") from exc

    @staticmethod
    def _done(value: bytes, expected: FileIdentity, pid: int | None) -> None:
        try:
            code_text, size_text, sha256 = value.decode("ascii").split(":")
            code = int(code_text)
        except (UnicodeError, ValueError) as exc:
            raise TransferError("download DONE frame is malformed") from exc
        suffix = f" (PID {pid})" if pid is not None else ""
        if code != 0:
            raise TransferError(f"download receiver failed with status {code}{suffix}")
        try:
            observed = FileIdentity(int(size_text), sha256)
        except (ValueError, TransferError) as exc:
            raise TransferError("download DONE identity is malformed") from exc
        if observed != expected:
            raise TransferError("download DONE identity differs from READY")

    def _next_control(
        self,
        parser: DownloadControlParser,
        deadline: float,
        *,
        allow_noise: bool,
    ) -> tuple[str, bytes]:
        while True:
            record = parser.pop(allow_noise=allow_noise)
            if record is not None:
                return record
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransferError("download stream timed out")
            payload = self.transport.receive(remaining)
            if not payload:
                raise TransferError("Lepton terminal closed during stream download")
            parser.feed(payload)

    def _abort(
        self,
        parser: DownloadControlParser,
        expected: FileIdentity,
        pid: int | None,
    ) -> None:
        self.transport.send(b"\x00A")
        deadline = time.monotonic() + self.completion_timeout_seconds
        kind, value = self._next_control(parser, deadline, allow_noise=False)
        if kind != "DONE" or value != b"46:-:-":
            raise TransferError("download sender did not attest a clean abort")

    def download(
        self,
        source: str,
        handle: BinaryIO,
        expected: FileIdentity,
    ) -> None:
        source = validate_remote_path(source)
        nonce = secrets.token_hex(16)
        script = _download_remote_script(source, nonce)
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        launcher = (
            f"__lxfd_s=$(printf '%s' {_quote(encoded)} | base64 -d); "
            "bash -c \"$__lxfd_s\"; __lxfd_rc=$?; "
            "unset __lxfd_s __lxfd_rc\n"
        )
        parser = DownloadControlParser(nonce)
        self.transport.send(b"\x00" + launcher.encode("utf-8"))
        deadline = time.monotonic() + self.setup_timeout_seconds
        pid: int | None = None
        ready: FileIdentity | None = None
        while ready is None:
            kind, value = self._next_control(parser, deadline, allow_noise=True)
            if kind == "PID":
                if pid is not None:
                    raise TransferError("download PID frame was duplicated")
                pid = self._pid(value)
            elif kind == "READY":
                if pid is None:
                    raise TransferError("download READY arrived before PID")
                ready = self._ready(value)
            else:
                self._done(value, expected, pid)
                raise TransferError("download sender completed before READY")
        if ready != expected:
            self._abort(parser, expected, pid)
            raise TransferError("download READY identity changed after inspection")

        self.transport.send(b"\x00G")
        remaining = expected.size
        digest = hashlib.sha256()
        written = 0
        deadline = time.monotonic() + self.completion_timeout_seconds
        while remaining:
            if parser.buffer:
                count = min(remaining, len(parser.buffer))
                chunk = bytes(parser.buffer[:count])
                del parser.buffer[:count]
            else:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise TransferError("download payload timed out")
                chunk = self.transport.receive(timeout)
                if not chunk:
                    raise TransferError("Lepton terminal closed during download payload")
                if len(chunk) > remaining:
                    parser.buffer.extend(chunk[remaining:])
                    chunk = chunk[:remaining]
            if handle.write(chunk) != len(chunk):
                raise TransferError("local download destination accepted a short write")
            digest.update(chunk)
            written += len(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
        observed = FileIdentity(written, digest.hexdigest())
        kind, value = self._next_control(parser, deadline, allow_noise=False)
        if kind != "DONE":
            raise TransferError("download control was duplicated after payload")
        self._done(value, expected, pid)
        if observed != expected:
            raise TransferError("downloaded raw payload failed SHA-256 verification")


def _safe_remote_prelude(path: str) -> str:
    path = validate_remote_path(path)
    parent = posixpath.dirname(path)
    return (
        f"__root={_quote(REMOTE_ROOT)}; __path={_quote(path)}; __parent={_quote(parent)}; "
        "test -d \"$__root\"; test ! -L \"$__root\"; "
        "__root_real=$(realpath -e -- \"$__root\"); test \"$__root_real\" = \"$__root\"; "
        "test -d \"$__parent\"; "
        "__parent_real=$(realpath -e -- \"$__parent\"); "
        "case \"$__parent_real\" in \"$__root\"|\"$__root\"/*) ;; *) exit 41 ;; esac; "
    )


class ShellRemoteFileStore:
    def __init__(self, shell: RemoteShell):
        self.shell = shell

    @staticmethod
    def _one(result: CommandResult, label: str) -> bytes:
        if len(result.data) != 1:
            raise TransferError(f"remote {label} returned an unexpected record count")
        return result.data[0]

    def inspect(self, path: str) -> FileIdentity | None:
        script = _safe_remote_prelude(path) + (
            "if test -L \"$__path\"; then exit 42; fi; "
            "if test ! -e \"$__path\"; then __lxft_emit ABSENT; exit 0; fi; "
            "test -f \"$__path\"; "
            "__size=$(wc -c < \"$__path\" | tr -d '[:space:]'); "
            "__sha=$(sha256sum -- \"$__path\" | awk '{print $1}'); "
            "__lxft_emit \"REGULAR:$__size:$__sha\""
        )
        return _identity_from_record(self._one(self.shell.run(script), "inspection"))

    def create_incoming(self, path: str, mode: int) -> None:
        if mode < 0 or mode > 0o777:
            raise TransferError("remote file mode is invalid")
        script = _safe_remote_prelude(path) + (
            "test ! -e \"$__path\"; test ! -L \"$__path\"; "
            "umask 077; set -C; : > \"$__path\"; set +C; "
            f"chmod {mode:04o} \"$__path\""
        )
        self.shell.run(script)

    def append(self, path: str, payload: bytes) -> None:
        path = validate_remote_path(path)
        encoded = base64.b64encode(payload).decode("ascii")
        script = _safe_remote_prelude(path) + (
            "test ! -L \"$__path\"; test -f \"$__path\"; "
            f"printf '%s' {_quote(encoded)} | base64 -d >> \"$__path\""
        )
        self.shell.run(script)

    def read_chunk(self, path: str, offset: int, length: int) -> bytes:
        if type(offset) is not int or offset < 0 or type(length) is not int or length <= 0:
            raise TransferError("remote read range is invalid")
        script = _safe_remote_prelude(path) + (
            "test ! -L \"$__path\"; test -f \"$__path\"; "
            f"__chunk=$(dd if=\"$__path\" iflag=skip_bytes,count_bytes skip={offset} "
            f"count={length} status=none | base64 | tr -d '\\n'); "
            "__lxft_emit \"$__chunk\""
        )
        encoded = self._one(self.shell.run(script), "read")
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise TransferError("remote file chunk is not valid base64") from exc

    def publish(
        self, incoming: str, destination: str, expected: FileIdentity, mode: int
    ) -> None:
        incoming = validate_remote_path(incoming)
        destination = validate_remote_path(destination)
        if posixpath.dirname(incoming) != posixpath.dirname(destination):
            raise TransferError("incoming file must be beside its destination")
        if not posixpath.basename(incoming).startswith(".incoming."):
            raise TransferError("incoming-file name is unsafe")
        script = _safe_remote_prelude(destination) + (
            f"__incoming={_quote(incoming)}; "
            "test ! -L \"$__incoming\"; test -f \"$__incoming\"; "
            "test ! -e \"$__path\"; test ! -L \"$__path\"; "
            "__size=$(wc -c < \"$__incoming\" | tr -d '[:space:]'); "
            "__sha=$(sha256sum -- \"$__incoming\" | awk '{print $1}'); "
            f"test \"$__size\" = {_quote(str(expected.size))}; "
            f"test \"$__sha\" = {_quote(expected.sha256)}; "
            f"chmod {mode:04o} \"$__incoming\"; "
            "ln -- \"$__incoming\" \"$__path\"; rm -f -- \"$__incoming\""
        )
        self.shell.run(script)

    def cleanup_incoming(self, path: str) -> None:
        path = validate_remote_path(path)
        if not posixpath.basename(path).startswith(".incoming."):
            raise TransferError("refusing to clean a non-incoming remote path")
        # The exact quoted path is removed; no wildcard or directory cleanup is allowed.
        script = _safe_remote_prelude(path) + "rm -f -- \"$__path\""
        self.shell.run(script, timeout_seconds=30.0)


class TransferEngine:
    def __init__(self, remote: RemoteFileStore, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        if type(chunk_bytes) is not int or not 1024 <= chunk_bytes <= MAX_CHUNK_BYTES:
            raise TransferError(
                f"chunk size must be between 1024 and {MAX_CHUNK_BYTES} bytes"
            )
        self.remote = remote
        self.chunk_bytes = chunk_bytes

    def upload(
        self,
        local: str | os.PathLike[str],
        remote: str,
        *,
        adopt_existing: bool = False,
        mode: int = 0o644,
    ) -> dict[str, Any]:
        destination = validate_remote_path(remote)
        local_path = Path(os.path.abspath(Path(local).expanduser()))
        descriptor, initial_stat = _open_regular(local_path)
        incoming = incoming_path(destination)
        created = False
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                expected = _hash_handle(handle)
                if _fingerprint(os.fstat(descriptor)) != _fingerprint(initial_stat):
                    raise TransferError("local source changed while it was hashed")
                observed = self.remote.inspect(destination)
                if observed is not None:
                    if adopt_existing and observed == expected:
                        return self._report("upload", "adopted", expected, local_path, destination)
                    raise TransferError("remote destination already exists and was not adopted")
                self.remote.create_incoming(incoming, mode)
                created = True
                handle.seek(0)
                streamed = hashlib.sha256()
                streamed_size = 0
                for chunk in iter(lambda: handle.read(self.chunk_bytes), b""):
                    self.remote.append(incoming, chunk)
                    streamed.update(chunk)
                    streamed_size += len(chunk)
                streamed_identity = FileIdentity(streamed_size, streamed.hexdigest())
                if streamed_identity != expected:
                    raise TransferError("local source changed while it was uploaded")
                if _fingerprint(os.fstat(descriptor)) != _fingerprint(initial_stat):
                    raise TransferError("local source changed while it was uploaded")
                if self.remote.inspect(incoming) != expected:
                    raise TransferError("remote incoming file failed SHA-256/size verification")
                self.remote.publish(incoming, destination, expected, mode)
                created = False
                if self.remote.inspect(destination) != expected:
                    raise TransferError("published remote file failed SHA-256/size verification")
                return self._report("upload", "published", expected, local_path, destination)
        finally:
            os.close(descriptor)
            if created:
                try:
                    self.remote.cleanup_incoming(incoming)
                except Exception:
                    # Preserve the original exception.  The unique exact incoming path is
                    # never adopted automatically and is reported by the caller's failure.
                    pass

    def upload_streaming(
        self,
        local: str | os.PathLike[str],
        remote: str,
        streamer: StreamingUploader,
        *,
        adopt_existing: bool = False,
        mode: int = 0o644,
    ) -> dict[str, Any]:
        """Upload through one fixed-count PTY stream, retaining resumable state."""

        destination = validate_remote_path(remote)
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise TransferError("remote file mode is invalid")
        local_path = Path(os.path.abspath(Path(local).expanduser()))
        descriptor, initial_stat = _open_regular(local_path)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                expected = _hash_handle(handle)
                if _fingerprint(os.fstat(descriptor)) != _fingerprint(initial_stat):
                    raise TransferError("local source changed while it was hashed")
                observed = self.remote.inspect(destination)
                if observed is not None:
                    if adopt_existing and observed == expected:
                        return self._report(
                            "upload", "adopted", expected, local_path, destination
                        )
                    raise TransferError(
                        "remote destination already exists and was not adopted"
                    )
                incoming = resumable_incoming_path(destination, expected)
                streamer.upload(
                    handle,
                    destination,
                    incoming,
                    expected,
                    mode=mode,
                )
                if _fingerprint(os.fstat(descriptor)) != _fingerprint(initial_stat):
                    raise TransferError("local source changed while it was streamed")
                if self.remote.inspect(destination) != expected:
                    raise TransferError(
                        "published remote stream failed SHA-256/size verification"
                    )
                return self._report(
                    "upload", "published", expected, local_path, destination
                )
        finally:
            os.close(descriptor)

    def download_streaming(
        self,
        remote: str,
        local: str | os.PathLike[str],
        downloader: StreamingDownloader,
        *,
        adopt_existing: bool = False,
        mode: int = 0o644,
    ) -> dict[str, Any]:
        """Download through one fixed-count raw stream and publish atomically."""

        source = validate_remote_path(remote)
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise TransferError("local file mode is invalid")
        destination = Path(os.path.abspath(Path(local).expanduser()))
        if destination.is_symlink():
            raise TransferError("local destination must not be a symlink")
        expected = self.remote.inspect(source)
        if expected is None:
            raise TransferError("remote source does not exist")
        if os.path.lexists(destination):
            if adopt_existing and _hash_regular_path(destination) == expected:
                return self._report("download", "adopted", expected, destination, source)
            raise TransferError("local destination already exists and was not adopted")
        if not destination.parent.is_dir() or destination.parent.is_symlink():
            raise TransferError("local destination parent must be an existing real directory")
        temporary = destination.parent / f".incoming.{secrets.token_hex(16)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except OSError as exc:
            raise TransferError("cannot create local incoming file") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                downloader.download(source, handle, expected)
            if self.remote.inspect(source) != expected:
                raise TransferError("remote source changed while it was downloaded")
            os.chmod(temporary, mode)
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise TransferError("local destination appeared during publication") from exc
            finally:
                temporary.unlink(missing_ok=True)
            if _hash_regular_path(destination) != expected:
                raise TransferError("published local file failed SHA-256/size verification")
            return self._report("download", "published", expected, destination, source)
        finally:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def download(
        self,
        remote: str,
        local: str | os.PathLike[str],
        *,
        adopt_existing: bool = False,
        mode: int = 0o644,
    ) -> dict[str, Any]:
        source = validate_remote_path(remote)
        destination = Path(os.path.abspath(Path(local).expanduser()))
        if destination.is_symlink():
            raise TransferError("local destination must not be a symlink")
        expected = self.remote.inspect(source)
        if expected is None:
            raise TransferError("remote source does not exist")
        if os.path.lexists(destination):
            if adopt_existing and _hash_regular_path(destination) == expected:
                return self._report("download", "adopted", expected, destination, source)
            raise TransferError("local destination already exists and was not adopted")
        if not destination.parent.is_dir() or destination.parent.is_symlink():
            raise TransferError("local destination parent must be an existing real directory")
        temporary = destination.parent / f".incoming.{secrets.token_hex(16)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except OSError as exc:
            raise TransferError("cannot create local incoming file") from exc
        digest = hashlib.sha256()
        written = 0
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                while written < expected.size:
                    requested = min(self.chunk_bytes, expected.size - written)
                    chunk = self.remote.read_chunk(source, written, requested)
                    if len(chunk) != requested:
                        raise TransferError("remote source returned a short file chunk")
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            downloaded = FileIdentity(written, digest.hexdigest())
            if downloaded != expected:
                raise TransferError("downloaded file failed SHA-256/size verification")
            if self.remote.inspect(source) != expected:
                raise TransferError("remote source changed while it was downloaded")
            os.chmod(temporary, mode)
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise TransferError("local destination appeared during publication") from exc
            finally:
                temporary.unlink(missing_ok=True)
            if _hash_regular_path(destination) != expected:
                raise TransferError("published local file failed SHA-256/size verification")
            return self._report("download", "published", expected, destination, source)
        finally:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _report(
        operation: str,
        status: str,
        identity: FileIdentity,
        local: Path,
        remote: str,
    ) -> dict[str, Any]:
        return {
            "operation": operation,
            "status": status,
            "local": str(local),
            "remote": remote,
            "size": identity.size,
            "sha256": identity.sha256,
        }


_LEPTON_DISCOVERY = r"""
import json
import sys
from urllib.parse import quote, urlsplit, urlunsplit
from leptonai.api.v2.client import APIClient

deployment = sys.argv[1]
requested_replica = sys.argv[2] or None
client = APIClient()
response = client._get(f"/deployments/{quote(deployment, safe='')}/replicas")
response.raise_for_status()
replicas = response.json()
if requested_replica:
    replicas = [r for r in replicas if r.get("metadata", {}).get("id") == requested_replica]
if len(replicas) != 1:
    raise SystemExit(17)
replica = replicas[0]["metadata"]["id"]
parsed = urlsplit(client.url)
path = (
    f"{parsed.path}/deployments/{quote(deployment, safe='')}"
    f"/replicas/{quote(replica, safe='')}/shell"
)
url = urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, "", ""))
print(json.dumps({
    "url": url,
    "authorization": client._header["Authorization"],
    "origin": client._header["origin"],
}, separators=(",", ":")))
"""


def discover_connection(
    deployment: str,
    replica: str | None = None,
    *,
    sdk_python: str | os.PathLike[str] | None = None,
) -> ConnectionInfo:
    if _TOKEN.fullmatch(deployment) is None:
        raise TransferError("Lepton deployment name is malformed")
    if replica is not None and _TOKEN.fullmatch(replica) is None:
        raise TransferError("Lepton replica ID is malformed")
    environment_python = os.environ.get("LEPTON_SDK_PYTHON")
    python = str(
        sdk_python
        or environment_python
        or "/home/ldu/.local/share/pipx/venvs/leptonai/bin/python"
    )
    python_path = Path(python).expanduser()
    if not python_path.is_absolute():
        raise TransferError("Lepton SDK interpreter path must be absolute")
    if (sdk_python is not None or environment_python is not None) and (
        not python_path.is_file() or not os.access(python_path, os.X_OK)
    ):
        raise TransferError("Lepton SDK interpreter is missing or not executable")
    try:
        completed = subprocess.run(
            [python, "-c", _LEPTON_DISCOVERY, deployment, replica or ""],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TransferError("Lepton SDK connection discovery failed") from exc
    if completed.returncode != 0:
        raise TransferError(
            f"Lepton SDK connection discovery failed with status {completed.returncode}"
        )
    try:
        document = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TransferError("Lepton SDK connection discovery returned malformed data") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "url",
        "authorization",
        "origin",
    }:
        raise TransferError("Lepton SDK connection discovery returned an unsafe schema")
    return ConnectionInfo(
        url=document["url"],
        authorization=document["authorization"],
        origin=document["origin"],
    )


_NODE_BRIDGE = r"""
const readline = require("readline");
const WebSocket = require("ws");

const heartbeatMs = Number(process.env.LXFT_HEARTBEAT_MS || "15000");
const heartbeatTimeoutMs = Number(process.env.LXFT_HEARTBEAT_TIMEOUT_MS || "45000");
const maxBuffered = Number(process.env.LXFT_MAX_BUFFERED_BYTES || "1048576");
const maxMessage = Number(process.env.LXFT_MAX_MESSAGE_BYTES || "524288");
let lastActivityMs = Date.now();
let closed = false;

function emit(event) {
  const line = JSON.stringify(event) + "\n";
  if (!process.stdout.write(line) && ws._socket) {
    ws._socket.pause();
    process.stdout.once("drain", () => { if (ws._socket) ws._socket.resume(); });
  }
}

function fail(code) {
  if (closed) return;
  closed = true;
  clearInterval(heartbeat);
  emit({event: "error", code});
  try { ws.terminate(); } catch (_) {}
  process.stdin.pause();
  setTimeout(() => process.exit(1), 25);
}

const ws = new WebSocket(process.env.LEPTON_WSS_URL, ["v4.channel.k8s.io"], {
  headers: {
    Authorization: process.env.LEPTON_AUTHORIZATION,
    origin: process.env.LEPTON_ORIGIN,
  },
  handshakeTimeout: 15000,
  perMessageDeflate: false,
  maxPayload: maxMessage,
});

async function waitBackpressure() {
  const deadline = Date.now() + 30000;
  while (ws.bufferedAmount > maxBuffered) {
    if (Date.now() >= deadline) throw new Error("backpressure_timeout");
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

let queue = Promise.resolve();
function queueSend(payload, requestId = null) {
  queue = queue.then(async () => {
    if (ws.readyState !== WebSocket.OPEN) throw new Error("socket_not_open");
    if (payload.length > maxMessage) throw new Error("outgoing_message_too_large");
    await waitBackpressure();
    await new Promise((resolve, reject) => ws.send(payload, err => err ? reject(err) : resolve()));
    lastActivityMs = Date.now();
    if (requestId !== null) emit({event: "sent", id: requestId});
  }).catch(() => fail("send_failed"));
}

ws.on("open", () => {
  queueSend(Buffer.concat([Buffer.from([4]), Buffer.from(JSON.stringify({Width: 160, Height: 50}))]));
  emit({event: "open"});
});
ws.on("pong", () => { lastActivityMs = Date.now(); });
ws.on("message", data => {
  lastActivityMs = Date.now();
  const frame = Buffer.from(data);
  if (frame.length > maxMessage) return fail("incoming_message_too_large");
  emit({event: "data", data: frame.toString("base64")});
});
ws.on("unexpected-response", (_request, response) => {
  fail(`http_${Number(response.statusCode) || 0}`);
});
ws.on("error", () => fail("websocket_error"));
ws.on("close", code => {
  if (!closed) {
    closed = true;
    clearInterval(heartbeat);
    emit({event: "close", code: Number(code) || 0});
  }
  process.stdin.pause();
  setTimeout(() => process.exit(0), 25);
});

const heartbeat = setInterval(() => {
  if (ws.readyState !== WebSocket.OPEN) return;
  if (Date.now() - lastActivityMs > heartbeatTimeoutMs) {
    return fail("heartbeat_timeout");
  }
  try {
    ws.ping();
    // Some Kubernetes exec gateways do not forward WebSocket pong events.
    // A successful outbound ping is still transport activity; TCP/WebSocket
    // error and close handlers remain the authoritative dead-peer signals.
    lastActivityMs = Date.now();
  } catch (_) { fail("heartbeat_failed"); }
}, heartbeatMs);

readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on("line", line => {
  let request;
  try { request = JSON.parse(line); } catch (_) { return fail("bad_local_request"); }
  if (request.op === "send" && typeof request.data === "string") {
    if (!Number.isSafeInteger(request.id) || request.id < 1) return fail("bad_local_request");
    let payload;
    try { payload = Buffer.from(request.data, "base64"); } catch (_) { return fail("bad_local_base64"); }
    queueSend(payload, request.id);
  } else if (request.op === "close") {
    clearInterval(heartbeat);
    closed = true;
    process.stdin.pause();
    ws.close();
    setTimeout(() => process.exit(0), 250);
  } else {
    fail("bad_local_request");
  }
});
"""


class NodeWebTerminalTransport:
    """Authenticated Kubernetes-exec WebSocket transport via Node's ``ws``."""

    def __init__(
        self,
        connection: ConnectionInfo,
        *,
        open_timeout_seconds: float = 30.0,
        heartbeat_seconds: float = 15.0,
        heartbeat_timeout_seconds: float = 45.0,
        max_buffered_bytes: int = 1024 * 1024,
        max_message_bytes: int = 512 * 1024,
        popen: Any = subprocess.Popen,
    ):
        if heartbeat_seconds <= 0 or heartbeat_timeout_seconds <= heartbeat_seconds:
            raise TransferError("WebSocket heartbeat configuration is invalid")
        node = os.environ.get("LEPTON_NODE_BIN", "node")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "NODE_PATH", "LANG", "LC_ALL"}
        }
        environment.update(
            {
                "LEPTON_WSS_URL": connection.url,
                "LEPTON_AUTHORIZATION": connection.authorization,
                "LEPTON_ORIGIN": connection.origin,
                "LXFT_HEARTBEAT_MS": str(int(heartbeat_seconds * 1000)),
                "LXFT_HEARTBEAT_TIMEOUT_MS": str(int(heartbeat_timeout_seconds * 1000)),
                "LXFT_MAX_BUFFERED_BYTES": str(max_buffered_bytes),
                "LXFT_MAX_MESSAGE_BYTES": str(max_message_bytes),
            }
        )
        try:
            self._process = popen(
                [node, "-e", _NODE_BRIDGE],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                env=environment,
            )
        except OSError as exc:
            raise TransferError("cannot start the local WebSocket bridge") from exc
        if self._process.stdin is None or self._process.stdout is None:
            raise TransferError("local WebSocket bridge did not create pipes")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._stdout, selectors.EVENT_READ)
        self._buffer = bytearray()
        self._pending_events: collections.deque[dict[str, Any]] = collections.deque()
        self._next_send_id = 1
        self._closed = False
        event = self._event(open_timeout_seconds)
        if event.get("event") != "open":
            self.close()
            raise TransferError("Lepton Web-terminal did not open")

    def _line(self, timeout_seconds: float) -> bytes:
        deadline = time.monotonic() + timeout_seconds
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransferError("local WebSocket bridge timed out")
            events = self._selector.select(remaining)
            if not events:
                raise TransferError("local WebSocket bridge timed out")
            chunk = os.read(self._stdout.fileno(), 64 * 1024)
            if not chunk:
                raise TransferError("local WebSocket bridge closed unexpectedly")
            self._buffer.extend(chunk)
            if len(self._buffer) > 2 * 1024 * 1024:
                raise TransferError("local WebSocket bridge event exceeded its size limit")

    def _event(self, timeout_seconds: float) -> dict[str, Any]:
        try:
            event = json.loads(self._line(timeout_seconds))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise TransferError("local WebSocket bridge emitted malformed data") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise TransferError("local WebSocket bridge event schema is malformed")
        if event["event"] == "error":
            code = event.get("code")
            safe_code = code if isinstance(code, str) and _TOKEN.fullmatch(code) else "unknown"
            raise TransferError(f"Lepton Web-terminal bridge failed ({safe_code})")
        if event["event"] == "close":
            raise TransferError("Lepton Web-terminal closed unexpectedly")
        return event

    def send(self, payload: bytes) -> None:
        if self._closed:
            raise TransferError("Lepton Web-terminal transport is closed")
        request_id = self._next_send_id
        self._next_send_id += 1
        request = json.dumps(
            {
                "op": "send",
                "id": request_id,
                "data": base64.b64encode(payload).decode("ascii"),
            },
            separators=(",", ":"),
        ).encode("ascii") + b"\n"
        try:
            self._stdin.write(request)
            self._stdin.flush()
        except OSError as exc:
            raise TransferError("cannot write to local WebSocket bridge") from exc
        deadline = time.monotonic() + 30.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransferError("WebSocket send backpressure acknowledgement timed out")
            event = self._event(remaining)
            if event["event"] == "sent":
                if event.get("id") != request_id:
                    raise TransferError("WebSocket send acknowledgement is out of order")
                return
            self._pending_events.append(event)

    def receive(self, timeout_seconds: float) -> bytes:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._pending_events:
                event = self._pending_events.popleft()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransferError("Lepton Web-terminal receive timed out")
                event = self._event(remaining)
            if event["event"] != "data":
                continue
            encoded = event.get("data")
            if not isinstance(encoded, str):
                raise TransferError("WebSocket data event is malformed")
            try:
                frame = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise TransferError("WebSocket data event is not valid base64") from exc
            if not frame:
                continue
            channel = frame[0]
            if channel in {1, 2}:
                # Kubernetes terminal sessions may emit an empty stdout/stderr
                # frame while the PTY is being attached.  It is not EOF; only
                # the bridge's explicit close/error event is terminal.
                if len(frame) == 1:
                    continue
                return frame[1:]
            if channel == 3:
                raise TransferError("Lepton terminal status channel reported a failure")

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._stdin.write(b'{"op":"close"}\n')
            self._stdin.flush()
        except OSError:
            pass
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._selector.close()


def _mode(value: str) -> int:
    try:
        result = int(value, 8)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode must be an octal value such as 0644") from exc
    if result < 0 or result > 0o777:
        raise argparse.ArgumentTypeError("mode must be between 0000 and 0777")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  lepton_web_terminal_transfer.py upload --local bundle.tar.zst \\\n"
            "      --remote /workspace/projects/roborender/handoff/bundle.tar.zst\n"
            "  lepton_web_terminal_transfer.py download \\\n"
            "      --remote /workspace/projects/roborender/results/result.json \\\n"
            "      --local /mnt/data2/ldu/result.json\n\n"
            "Credentials come only from the active Lepton SDK profile."
        ),
    )
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--replica", help="exact replica ID; omitted requires exactly one replica")
    parser.add_argument(
        "--sdk-python",
        help=(
            "explicit Lepton-SDK interpreter; otherwise LEPTON_SDK_PYTHON or "
            "the pipx default is used"
        ),
    )
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--command-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=45.0)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("upload", "download"):
        command = subparsers.add_parser(name)
        command.add_argument("--local", required=True)
        command.add_argument("--remote", required=True)
        command.add_argument(
            "--adopt-existing",
            action="store_true",
            help="succeed only when an existing destination has identical size and SHA-256",
        )
        command.add_argument("--mode", type=_mode, default=0o644)
        if name == "upload":
            command.add_argument(
                "--legacy-chunked-upload",
                action="store_true",
                help="use the slower acknowledged-command path instead of raw streaming",
            )
        else:
            command.add_argument(
                "--legacy-chunked-download",
                action="store_true",
                help="use the slower acknowledged-command path instead of raw streaming",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    transport: NodeWebTerminalTransport | None = None
    try:
        connection = discover_connection(
            args.deployment, args.replica, sdk_python=args.sdk_python
        )
        transport = NodeWebTerminalTransport(
            connection,
            heartbeat_seconds=args.heartbeat_seconds,
            heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
        )
        shell = RemoteShell(
            transport, command_timeout_seconds=args.command_timeout_seconds
        )
        engine = TransferEngine(ShellRemoteFileStore(shell), chunk_bytes=args.chunk_bytes)
        if args.operation == "upload":
            if args.legacy_chunked_upload:
                result = engine.upload(
                    args.local,
                    args.remote,
                    adopt_existing=args.adopt_existing,
                    mode=args.mode,
                )
            else:
                streamer = StreamingUploader(
                    transport,
                    chunk_bytes=args.chunk_bytes,
                    completion_timeout_seconds=args.command_timeout_seconds,
                )
                result = engine.upload_streaming(
                    args.local,
                    args.remote,
                    streamer,
                    adopt_existing=args.adopt_existing,
                    mode=args.mode,
                )
        else:
            if args.legacy_chunked_download:
                result = engine.download(
                    args.remote,
                    args.local,
                    adopt_existing=args.adopt_existing,
                    mode=args.mode,
                )
            else:
                downloader = StreamingDownloader(
                    transport,
                    completion_timeout_seconds=args.command_timeout_seconds,
                )
                result = engine.download_streaming(
                    args.remote,
                    args.local,
                    downloader,
                    adopt_existing=args.adopt_existing,
                    mode=args.mode,
                )
        print(json.dumps(result, sort_keys=True))
        return 0
    except TransferError as exc:
        print(f"transfer failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
