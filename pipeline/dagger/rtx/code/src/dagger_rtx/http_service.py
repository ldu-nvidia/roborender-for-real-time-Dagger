from __future__ import annotations

import hmac
import ipaddress
import json
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, canonical_json
from .owner import CrashUncertain, IsaacTransitionOwner, OwnerConflict

OWNER_BEGIN_PATH = "/v1/rollout-windows/begin"
OWNER_PREPARE_PATH = "/v1/steps/prepare"
OWNER_COMMIT_PATH = "/v1/steps/commit"


class _OwnerHandler(BaseHTTPRequestHandler):
    server_version = "RoboRenderRTXOwner/1"

    def log_message(self, format, *args):
        return

    def _authorized(self) -> bool:
        token_file: Path = self.server.token_file  # type: ignore[attr-defined]
        stat = token_file.stat()
        if stat.st_mode & 0o077:
            raise PermissionError("bearer token file must have mode 0600 or stricter")
        expected = token_file.read_text(encoding="utf-8").strip()
        supplied = self.headers.get("Authorization", "")
        return bool(expected) and hmac.compare_digest(supplied, f"Bearer {expected}")

    def do_GET(self):
        if self.path == "/live":
            return self._reply(200, {"ok": True, "service": "dagger-rtx-isaac-owner", "live": True})
        try:
            if not self._authorized():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            if self.path == "/v1/status":
                owner: IsaacTransitionOwner = self.server.owner  # type: ignore[attr-defined]
                return self._reply(200, owner.status())
            return self._reply(404, {"ok": False, "error": "not_found"})
        except PermissionError as exc:
            return self._reply(500, {"ok": False, "error": "unsafe_token_file", "message": str(exc)})

    def do_POST(self):
        try:
            if not self._authorized():
                return self._reply(401, {"ok": False, "error": "unauthorized"})
            if self.headers.get_content_type() != "application/json":
                return self._reply(415, {"ok": False, "error": "application_json_required"})
            length_text = self.headers.get("Content-Length")
            if length_text is None:
                return self._reply(411, {"ok": False, "error": "content_length_required"})
            try:
                length = int(length_text)
            except ValueError:
                return self._reply(400, {"ok": False, "error": "invalid_content_length"})
            max_body = self.server.max_body_bytes  # type: ignore[attr-defined]
            if length < 0 or length > max_body:
                return self._reply(413, {"ok": False, "error": "body_too_large"})
            raw = self.rfile.read(length)
            if len(raw) != length:
                return self._reply(400, {"ok": False, "error": "truncated_body"})
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ContractError("JSON body must be an object")
            owner: IsaacTransitionOwner = self.server.owner  # type: ignore[attr-defined]
            routes = {
                OWNER_BEGIN_PATH: owner.begin_window,
                OWNER_PREPARE_PATH: owner.prepare_step,
                OWNER_COMMIT_PATH: owner.commit_step,
            }
            handler = routes.get(self.path)
            if handler is None:
                return self._reply(404, {"ok": False, "error": "not_found"})
            return self._reply(200, handler(body))
        except json.JSONDecodeError as exc:
            return self._reply(400, {"ok": False, "error": "invalid_json", "message": str(exc)})
        except ContractError as exc:
            return self._reply(400, {"ok": False, "error": "invalid_contract", "message": str(exc)})
        except OwnerConflict as exc:
            return self._reply(409, {"ok": False, "error": "state_conflict", "message": str(exc)})
        except CrashUncertain as exc:
            return self._reply(503, {"ok": False, "error": "crash_uncertain", "message": str(exc)})
        except PermissionError as exc:
            return self._reply(500, {"ok": False, "error": "unsafe_token_file", "message": str(exc)})

    def _reply(self, status: int, value: Mapping[str, Any]):
        body = canonical_json(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


class OwnerHttpServer(AbstractContextManager):
    def __init__(
        self,
        owner: IsaacTransitionOwner,
        token_file: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_body_bytes: int = 64 << 20,
    ):
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Isaac owner service must bind a loopback address")
        if not token_file.is_file() or token_file.stat().st_mode & 0o077:
            raise PermissionError("bearer token file must exist with mode 0600 or stricter")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise PermissionError("bearer token file is empty or malformed")
        self.server = ThreadingHTTPServer((host, port), _OwnerHandler)
        self.server.owner = owner  # type: ignore[attr-defined]
        self.server.token_file = token_file  # type: ignore[attr-defined]
        self.server.max_body_bytes = max_body_bytes  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False
