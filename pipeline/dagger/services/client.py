"""Small dependency-free client for the versioned H200 service boundaries."""

from __future__ import annotations

import json
import http.client
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ServiceError(RuntimeError):
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
