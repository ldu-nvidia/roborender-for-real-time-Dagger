"""Small dependency-free client for the versioned H200 service boundaries."""

from __future__ import annotations

import json
import http.client
import hashlib
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
import time
from pathlib import Path
from typing import Any

from pipeline.dagger.services.recovery_policy import CONTROL_RECOVERY_SECONDS


class ServiceError(RuntimeError):
    pass


class RetryableTransportError(RuntimeError):
    pass


class ServiceClient:
    def __init__(self, base_url: str, token: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str = "application/json") -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path, data=body, method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"message": str(exc)}
            raise ServiceError(f"{exc.code} {detail.get('error')}: {detail.get('message')}") from exc

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", path,
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
        )

    def post_bytes(self, path: str, body: bytes, content_type: str) -> dict[str, Any]:
        """Post an already framed body, used by the lean ndarray batch protocol."""

        return self._request("POST", path, body, content_type)

    def put_file(self, path: str, source: Path) -> dict[str, Any]:
        """Stream a large chunk without holding a second copy in actor RAM."""
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ServiceError(f"unsupported service URL: {self.base_url}")
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=self.timeout)
        request_path = (parsed.path.rstrip("/") + "/" + path.lstrip("/")) or "/"
        size = source.stat().st_size
        try:
            connection.putrequest("PUT", request_path)
            connection.putheader("Authorization", f"Bearer {self.token}")
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    connection.send(block)
            response = connection.getresponse()
            raw = response.read()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ServiceError(f"invalid service response: HTTP {response.status}") from exc
            if response.status >= 400:
                raise ServiceError(
                    f"{response.status} {payload.get('error')}: {payload.get('message')}"
                )
            return payload
        finally:
            connection.close()

    def download_file(self, path: str, destination: Path, expected_sha256: str) -> Path:
        """Download atomically and reject partial or corrupted checkpoints."""
        request = urllib.request.Request(
            self.base_url + path, method="GET",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, partial.open("wb") as output:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                output.flush()
            actual = digest.hexdigest()
            if actual != expected_sha256:
                partial.unlink(missing_ok=True)
                raise ServiceError(f"download SHA-256 mismatch: {actual}")
            partial.replace(destination)
            return destination
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            try:
                detail = json.loads(exc.read())
            except Exception:
                detail = {"message": str(exc)}
            raise ServiceError(
                f"{exc.code} {detail.get('error')}: {detail.get('message')}"
            ) from exc
        except Exception:
            partial.unlink(missing_ok=True)
            raise


class IdempotentControlClient(ServiceClient):
    """Bounded retries of known idempotent metadata requests, never env.step.

    HTTP contract errors are intentionally not retried. POST retry sends the
    identical bytes and identity, so a lost learner acknowledgement cannot
    submit a second optimizer job.
    """
    def __init__(self, *args, attempts=5, sleep=time.sleep, monotonic=time.monotonic,
                 recovery_seconds=None, report=None, **kwargs):
        super().__init__(*args, **kwargs)
        if attempts is not None and (type(attempts) is not int or not 1 <= attempts <= 5):
            raise ValueError("control retry attempts must be 1..5")
        if recovery_seconds is not None and (isinstance(recovery_seconds, bool)
                or not isinstance(recovery_seconds, (int, float))
                or not math.isfinite(recovery_seconds)
                or not 0 < recovery_seconds <= CONTROL_RECOVERY_SECONDS):
            raise ValueError("invalid bounded control recovery window")
        if attempts is None and recovery_seconds is None:
            raise ValueError("control retries need an attempt limit or deadline")
        self.attempts, self.sleep, self.clock = attempts, sleep, monotonic
        self.recovery_seconds = recovery_seconds
        self.report = report if report is not None else self._report

    @staticmethod
    def _report(event):
        # Public route/type/timing only. Never print payloads, tokens, URLs,
        # exception messages or response bodies during recovery.
        print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)

    def _request(self, method, path, body=None, content_type="application/json"):
        allowed = (method == "GET" or (method == "POST" and (
            path in {"/v1/runs", "/v1/rounds/start", "/v1/jobs", "/v1/checkpoints/register-local",
                     "/v2/evaluations", "/v1/iterations/advance",
                     "/v1/mse-rounds/register", "/v1/mse-rounds/record", "/v1/mse-rounds/finish-test"}
            or (path.startswith("/v2/evaluations/") and path.endswith("/outcomes")))))
        if not allowed:
            raise ValueError("automatic retry is not permitted for this route")
        started = self.clock()
        deadline = None if self.recovery_seconds is None else started + self.recovery_seconds
        attempt = 0
        while True:
            attempt += 1
            try:
                if deadline is None:
                    result = super()._request(method, path, body, content_type)
                else:
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        self.report({"action": "control_transport_exhausted", "method": method,
                            "route": urllib.parse.urlsplit(path).path, "attempts": attempt - 1,
                            "elapsed_seconds": self.clock() - started,
                            "error_type": "RecoveryDeadline", "native_replay_allowed": False})
                        raise RetryableTransportError("control recovery deadline exhausted")
                    # Do not mutate self.timeout: concurrent users must retain
                    # their own deadlines. A plain one-shot client also avoids
                    # nesting another retry loop inside this recovery owner.
                    once = ServiceClient(self.base_url, self.token, timeout=min(self.timeout, remaining))
                    result = once._request(method, path, body, content_type)
                if deadline is not None and attempt > 1:
                    self.report({"action": "control_transport_recovered", "method": method,
                        "route": urllib.parse.urlsplit(path).path, "attempts": attempt,
                        "elapsed_seconds": self.clock() - started})
                return result
            except urllib.error.HTTPError:
                raise  # Never retry authorization or application HTTP errors.
            except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.HTTPException) as exc:
                remaining = None if deadline is None else deadline - self.clock()
                if ((self.attempts is not None and attempt >= self.attempts)
                        or (remaining is not None and remaining <= 0)):
                    if deadline is not None:
                        self.report({"action": "control_transport_exhausted", "method": method,
                            "route": urllib.parse.urlsplit(path).path, "attempts": attempt,
                            "elapsed_seconds": self.clock() - started,
                            "error_type": type(exc).__name__, "native_replay_allowed": False})
                    raise RetryableTransportError("control connection retry budget exhausted") from exc
                delay = min(2 ** min(attempt - 1, 3), 5)
                if remaining is not None:
                    delay = min(delay, remaining)
                    self.report({"action": "control_transport_recovering", "method": method,
                        "route": urllib.parse.urlsplit(path).path, "attempt": attempt,
                        "elapsed_seconds": self.clock() - started, "remaining_seconds": remaining,
                        "next_delay_seconds": delay, "error_type": type(exc).__name__,
                        "native_replay_allowed": False})
                self.sleep(delay)
