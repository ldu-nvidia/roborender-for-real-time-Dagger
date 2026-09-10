from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from dagger_rtx.binary_transport_v2 import BoundedBearerBatchV2Transport
from dagger_rtx.binary_transport_v3 import BoundedBearerBatchV3Transport
from dagger_rtx.contracts import canonical_json
from dagger_rtx.ndarray_batch_v2 import NDARRAY_BATCH_CONTENT_TYPE
from dagger_rtx.ndarray_batch_v3 import NDARRAY_BATCH_CONTENT_TYPE_V3


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "authorization": self.headers["Authorization"],
                "content_type": self.headers["Content-Type"],
                "body": body,
                "client_port": self.client_address[1],
            }
        )
        response = canonical_json({"ok": True}) + b"\n"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "authorization": self.headers["Authorization"],
                "content_type": self.headers.get("Content-Type"),
                "body": b"",
                "client_port": self.client_address[1],
            }
        )
        response = canonical_json({"ok": True}) + b"\n"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def _token(tmp_path: Path) -> Path:
    value = tmp_path / "token"
    value.write_text("secret-for-test\n")
    value.chmod(0o600)
    return value


def test_v2_transport_sends_binary_and_canonical_json_and_accepts_newline(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = BoundedBearerBatchV2Transport(
        f"http://127.0.0.1:{server.server_port}",
        _token(tmp_path),
    )
    try:
        assert transport.get_json("/ready", max_response_bytes=1024) == {"ok": True}
        assert transport.post_binary(
            "/v2/steps/prepare-batch",
            b"RDB2-test",
            batch_id="prepare-a",
            max_response_bytes=1024,
        ) == {"ok": True}
        payload = {
            "contract_version": "roborender.dagger.rtx.v2",
            "kind": "commit_batch",
            "batch_id": "commit-a",
            "lanes": [],
        }
        assert transport.post_json(
            "/v2/steps/commit-batch", payload, max_response_bytes=1024
        ) == {"ok": True}
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    ready, binary, commit = server.requests
    assert ready["path"] == "/ready"
    assert ready["content_type"] is None
    assert binary["content_type"] == NDARRAY_BATCH_CONTENT_TYPE
    assert binary["body"] == b"RDB2-test"
    assert commit["content_type"] == "application/json"
    assert commit["body"] == canonical_json(payload)
    assert (
        ready["authorization"]
        == binary["authorization"]
        == commit["authorization"]
        == "Bearer secret-for-test"
    )
    # HTTP/1.1 server kept one client connection for both batch phases.
    assert ready["client_port"] == binary["client_port"] == commit["client_port"]


def test_v3_transport_uses_exact_content_type_and_keeps_connection(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = BoundedBearerBatchV3Transport(
        f"http://127.0.0.1:{server.server_port}", _token(tmp_path)
    )
    try:
        assert transport.post_binary(
            "/v3/steps/prepare-batch",
            b"RDB3-test",
            batch_id="prepare-v3",
            max_response_bytes=1024,
        ) == {"ok": True}
        payload = {
            "contract_version": "roborender.dagger.rtx.v3",
            "kind": "commit_batch",
            "batch_id": "commit-v3",
            "lanes": [],
        }
        assert transport.post_json(
            "/v3/steps/commit-batch", payload, max_response_bytes=1024
        ) == {"ok": True}
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    binary, commit = server.requests
    assert binary["content_type"] == NDARRAY_BATCH_CONTENT_TYPE_V3
    assert binary["body"] == b"RDB3-test"
    assert commit["body"] == canonical_json(payload)
    assert binary["client_port"] == commit["client_port"]
