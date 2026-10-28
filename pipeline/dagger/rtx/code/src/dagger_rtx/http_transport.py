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


class TransportError(RuntimeError):
    """Base class for failures at the authenticated HTTP transport boundary."""


class TransientTransportError(TransportError):
    """A request may be replayed with its identical idempotency key."""


class RemoteHttpError(TransportError):
    """A non-transient HTTP response that must not be retried automatically."""

    def __init__(self, status: int, detail: bytes) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"remote HTTP {status}: {detail[:256]!r}")


_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class JsonTransport(Protocol):
    def post(self, path: str, payload: Mapping[str, Any], *, max_response_bytes: int) -> dict[str, Any]: ...


class BoundedBearerJsonTransport:
    def __init__(
        self,
        base_url: str,
        token_file: Path,
        *,
        timeout_s: float = 30.0,
        max_request_bytes: int = 64 << 20,
        telemetry: EventSink | None = None,
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or parsed.path not in ("", "/"):
            raise ValueError("base_url must be an HTTP(S) origin")
        self.parsed = parsed
        self.token_file = token_file
        self.timeout_s = timeout_s
        self.max_request_bytes = max_request_bytes
        self.telemetry = telemetry
        self._attempt_lock = threading.Lock()
        self._attempts: dict[tuple[str, str], int] = {}

    def _next_attempt(self, path: str, payload: Mapping[str, Any]) -> int:
        identity = payload.get("idempotency_key", "")
        if not isinstance(identity, str):
            identity = ""
        key = (path, identity)
        with self._attempt_lock:
            attempt = self._attempts.get(key, 0) + 1
            self._attempts[key] = attempt
        return attempt

    def _token(self) -> str:
        st = self.token_file.stat()
        if st.st_mode & 0o077:
            raise PermissionError("bearer token file must not be group/world accessible")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise PermissionError("invalid bearer token file")
        return token

    def post(self, path: str, payload: Mapping[str, Any], *, max_response_bytes: int) -> dict[str, Any]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("path must be an absolute, query-free service path")
        body = canonical_json(payload)
        if len(body) > self.max_request_bytes:
            raise ContractError("outbound JSON body exceeds configured limit")
        attempt = self._next_attempt(path, payload)
        started_ns = time.perf_counter_ns()
        response_status: int | None = None
        response_body_bytes = 0
        outcome = "error"
        error_type: str | None = None
        retryable = False
        response_headers: dict[str, str] = {}
        cls = http.client.HTTPSConnection if self.parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.parsed.hostname, self.parsed.port, timeout=self.timeout_s)
        # Resolve and validate the local credential before entering the network
        # error conversion below.  A missing/unsafe token is a configuration
        # failure, never a transient request failure.
        token = self._token()
        try:
            conn.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
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
            response_body_bytes = len(raw)
            if len(raw) > max_response_bytes:
                raise ContractError("remote response exceeds configured limit")
            if response.status < 200 or response.status >= 300:
                message = f"remote HTTP {response.status}: {raw[:256]!r}"
                if response.status in _TRANSIENT_HTTP_STATUSES:
                    retryable = True
                    raise TransientTransportError(message)
                raise RemoteHttpError(response.status, raw)
            if response.getheader("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise ContractError("remote response is not application/json")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractError("remote response is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ContractError("remote JSON response must be an object")
            outcome = "success"
            return value
        except (ContractError, TransportError) as exc:
            error_type = type(exc).__name__
            raise
        except (TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
            error_type = type(exc).__name__
            retryable = True
            raise TransientTransportError(
                f"transient HTTP transport failure: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            conn.close()
            duration_ns = time.perf_counter_ns() - started_ns
            key = payload.get("step_key")
            safe_key = dict(key) if isinstance(key, Mapping) else None
            record_if_present(
                self.telemetry,
                "http_request",
                path=path,
                idempotency_key=payload.get("idempotency_key"),
                step_key=safe_key,
                attempt=attempt,
                request_body_bytes=len(body),
                response_body_bytes=response_body_bytes,
                total_body_bytes=len(body) + response_body_bytes,
                duration_ns=duration_ns,
                body_bytes_per_second=(
                    (len(body) + response_body_bytes) * 1_000_000_000 / duration_ns
                    if duration_ns > 0
                    else None
                ),
                http_status=response_status,
                outcome=outcome,
                retryable=retryable,
                error_type=error_type,
                response_headers=response_headers,
            )
