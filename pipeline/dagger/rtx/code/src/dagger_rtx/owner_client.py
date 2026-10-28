from __future__ import annotations

from typing import Any, Mapping

from .http_service import OWNER_BEGIN_PATH, OWNER_COMMIT_PATH, OWNER_PREPARE_PATH
from .http_transport import JsonTransport


class HttpOwnerClient:
    def __init__(self, transport: JsonTransport):
        self.transport = transport

    def begin_window(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.transport.post(OWNER_BEGIN_PATH, request, max_response_bytes=1 << 20)

    def prepare_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.transport.post(OWNER_PREPARE_PATH, request, max_response_bytes=64 << 20)

    def commit_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.transport.post(OWNER_COMMIT_PATH, request, max_response_bytes=1 << 20)
