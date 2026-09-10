"""Authenticated bounded HTTP transport for DAgger v2 batch messages."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from .capacity_telemetry import EventSink, record_if_present
from .contracts import ContractError, canonical_json
from .http_transport import (
    RemoteHttpError,
    TransientTransportError,
    TransportError,
    _TRANSIENT_HTTP_STATUSES,
)
from .ndarray_batch_v2 import NDARRAY_BATCH_CONTENT_TYPE


class BatchV2Transport(Protocol):
    def get_json(self, path: str, *, max_response_bytes: int) -> dict[str, Any]: ...

    def post_binary(
        self,
        path: str,
        body: bytes,
        *,
        batch_id: str,
        max_response_bytes: int,
    ) -> dict[str, Any]: ...

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        max_response_bytes: int,
    ) -> dict[str, Any]: ...


class BoundedBearerBatchV2Transport:
    """Send one bounded binary prepare/anchor batch and a small JSON commit.

    A connection is reused when the server permits HTTP keep-alive. Access is
    serialized because a single vector owner has exactly one in-flight batch
    phase at a time.
    """

    binary_content_type = NDARRAY_BATCH_CONTENT_TYPE
    protocol_label = "v2"
    telemetry_event = "http_batch_request_v2"

    def __init__(
        self,
        base_url: str,
        token_file: Path,
        *,
        timeout_s: float = 120.0,
        max_request_bytes: int = 64 << 20,
        telemetry: EventSink | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or parsed.path not in ("", "/"):
            raise ValueError("base_url must be an HTTP(S) origin")
        self.parsed = parsed
        self.token_file = Path(token_file)
        self.timeout_s = timeout_s
        self.max_request_bytes = max_request_bytes
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None
        self._attempts: dict[tuple[str, str], int] = {}

    def _token(self) -> str:
        stat = self.token_file.stat()
        if stat.st_mode & 0o077:
            raise PermissionError("bearer token file must not be group/world accessible")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise PermissionError("invalid bearer token file")
        return token

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = (
                http.client.HTTPSConnection
                if self.parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            self._conn = cls(
                self.parsed.hostname,
                self.parsed.port,
                timeout=self.timeout_s,
            )
        return self._conn

    def _discard_connection(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _validate_path(path: str) -> None:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("path must be an absolute, query-free service path")

    def _request(
        self,
        path: str,
        body: bytes,
        *,
        content_type: str,
        identity: str,
        max_response_bytes: int,
        method: str = "POST",
    ) -> dict[str, Any]:
        self._validate_path(path)
        if not isinstance(body, bytes):
            raise TypeError("HTTP request body must be bytes")
        if len(body) > self.max_request_bytes:
            raise ContractError(
                f"outbound {self.protocol_label} batch body exceeds configured limit"
            )
        token = self._token()
        key = (path, identity)
        with self._lock:
            attempt = self._attempts.get(key, 0) + 1
            self._attempts[key] = attempt
            started_ns = time.perf_counter_ns()
            response_status: int | None = None
            response_bytes = 0
            outcome = "error"
            retryable = False
            error_type: str | None = None
            response_headers: dict[str, str] = {}
            try:
                connection = self._connection()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Length": str(len(body)),
                    "Connection": "keep-alive",
                }
                if content_type:
                    headers["Content-Type"] = content_type
                connection.request(
                    method,
                    path,
                    body=body,
                    headers=headers,
                )
                response = connection.getresponse()
                response_status = int(response.status)
                for header in (
                    "Server-Timing",
                    "X-Roborender-Worker",
                    "X-Worker-Id",
                    "X-Queue-Depth",
                ):
                    value = response.getheader(header)
                    if value is not None:
                        response_headers[header.lower()] = value[:256]
                raw = response.read(max_response_bytes + 1)
                response_bytes = len(raw)
                if response.will_close:
                    self._discard_connection()
                if len(raw) > max_response_bytes:
                    # The response may still have unread bytes. Never reuse a
                    # connection whose HTTP message boundary was not drained.
                    self._discard_connection()
                    raise ContractError("remote response exceeds configured limit")
                if response.status < 200 or response.status >= 300:
                    message = f"remote HTTP {response.status}: {raw[:256]!r}"
                    if response.status in _TRANSIENT_HTTP_STATUSES:
                        retryable = True
                        raise TransientTransportError(message)
                    raise RemoteHttpError(response.status, raw)
                actual_type = response.getheader("Content-Type", "").split(";", 1)[0].strip()
                if actual_type != "application/json":
                    raise ContractError(
                        f"remote {self.protocol_label} response is not application/json"
                    )
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ContractError(
                        f"remote {self.protocol_label} response is not valid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise ContractError(
                        f"remote {self.protocol_label} JSON response must be an object"
                    )
                outcome = "success"
                return value
            except (ContractError, TransportError) as exc:
                error_type = type(exc).__name__
                if isinstance(exc, TransientTransportError):
                    self._discard_connection()
                raise
            except (TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
                error_type = type(exc).__name__
                retryable = True
                self._discard_connection()
                raise TransientTransportError(
                    f"transient HTTP transport failure: {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                duration_ns = time.perf_counter_ns() - started_ns
                record_if_present(
                    self.telemetry,
                    self.telemetry_event,
                    path=path,
                    batch_id=identity,
                    attempt=attempt,
                    content_type=content_type,
                    method=method,
                    request_body_bytes=len(body),
                    response_body_bytes=response_bytes,
                    total_body_bytes=len(body) + response_bytes,
                    duration_ns=duration_ns,
                    body_bytes_per_second=(
                        (len(body) + response_bytes) * 1_000_000_000 / duration_ns
                        if duration_ns > 0
                        else None
                    ),
                    http_status=response_status,
                    outcome=outcome,
                    retryable=retryable,
                    error_type=error_type,
                    response_headers=response_headers,
                )

    def post_binary(
        self,
        path: str,
        body: bytes,
        *,
        batch_id: str,
        max_response_bytes: int,
    ) -> dict[str, Any]:
        return self._request(
            path,
            body,
            content_type=self.binary_content_type,
            identity=batch_id,
            max_response_bytes=max_response_bytes,
        )

    def get_json(self, path: str, *, max_response_bytes: int) -> dict[str, Any]:
        return self._request(
            path,
            b"",
            content_type="",
            identity="ready",
            max_response_bytes=max_response_bytes,
            method="GET",
        )

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        max_response_bytes: int,
    ) -> dict[str, Any]:
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ContractError(f"{self.protocol_label} JSON batch requires batch_id")
        return self._request(
            path,
            canonical_json(payload),
            content_type="application/json",
            identity=batch_id,
            max_response_bytes=max_response_bytes,
        )

    def close(self) -> None:
        with self._lock:
            self._discard_connection()
