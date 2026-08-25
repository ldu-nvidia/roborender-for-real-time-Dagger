#!/usr/bin/env python3
"""Tiny protocol-v2 server used only by local fleet lifecycle tests."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import struct
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.server.renderer_identity import (
    canonical_json_sha256,
    describe_renderer_identity,
)


ADAPTER_SHA = "a" * 64
ENGINE_SHA = "b" * 64


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("client closed")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock: socket.socket) -> object:
    size = struct.unpack("!Q", recv_exact(sock, 8))[0]
    return pickle.loads(recv_exact(sock, size))  # noqa: S301 - local test process


def send_message(sock: socket.socket, value: object) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)


def ping_response() -> dict[str, object]:
    gpu_name = "Fake RTX"
    capability = "12.0"
    runtime_identity = {
        "schema_version": 1,
        "release_id": "fake-qualified-release",
        "adapter_sha256": ADAPTER_SHA,
        "generation": {
            "steps": 5,
            "cfg_scale": 1.0,
            "tea_cache_threshold": 0.3,
            "cuda_graph": False,
        },
        "runtime": {
            "backend": "tensorrt",
            "engine_loaded": True,
            "engine_verified_at_load": True,
            "gpu": gpu_name,
            "compute_capability": capability,
            "engine_path": "/test/fake.engine",
        },
        "engine": {
            "sha256": ENGINE_SHA,
            "engine_key": "fake-key",
            "export_key": "fake-export",
            "adapter_sha256": ADAPTER_SHA,
            "status": "built_microbenchmarked_and_correctness_checked",
            "precision": "bf16_fp16",
            "cuda_graph_enabled_for_runtime": False,
            "gpu": gpu_name,
            "compute_capability": capability,
            "tensorrt": "10.test",
            "cuda": "12.test",
            "torch": "2.test",
            "driver": "test",
        },
    }
    renderer_identity = describe_renderer_identity(
        runtime_identity,
        server_script=Path(__file__),
        expected_server_script_sha256=os.environ.get(
            "IMAGE3F_EXPECT_SERVER_SCRIPT_SHA256"
        ),
    )
    return {
        "ok": True,
        "protocol_version": 2,
        "server_id": os.environ["IMAGE3F_SERVER_ID"],
        "capabilities": {
            "lagged_reference_history": True,
            "reference_lag_max": int(os.environ["IMAGE3F_MAX_REFERENCE_LAG"]),
            "serialized_inference": True,
            "output_views": ["head_color", "hand_left", "hand_right"],
            "output_layout": "vertical",
            "output_tile_shape_hwc": [240, 416, 3],
            "output_stacked_shape_hwc": [720, 416, 3],
        },
        "runtime_identity": runtime_identity,
        "renderer_identity": renderer_identity,
        "renderer_identity_sha256": canonical_json_sha256(renderer_identity),
        "metrics": {"requests": 0, "errors": 0, "replays": 0},
    }


def main() -> int:
    endpoint = Path(os.environ["ENDPOINT_FILE"])
    queue = Path(os.environ["IMAGE3F_QUEUE"])
    stop = Path(os.environ["IMAGE3F_STOP_FILE"])
    ack = Path(os.environ["IMAGE3F_STOP_ACK_FILE"])
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", int(os.environ["IMAGE3F_PORT"])))
    server.listen(8)
    server.settimeout(0.05)
    port = server.getsockname()[1]
    atomic_text(endpoint, f"127.0.0.1 {port}\n")
    atomic_text(queue / "READY", f"{time.time()}\n")
    while not stop.exists():
        try:
            connection, _ = server.accept()
        except socket.timeout:
            continue
        with connection:
            request = recv_message(connection)
            if isinstance(request, dict) and request.get("cmd") == "ping":
                send_message(connection, ping_response())
            else:
                send_message(connection, {"ok": False, "error": "test ping only"})
    server.close()
    stop_bytes = stop.read_bytes()
    payload = {
        "schema": "fast_roborender.image3f_stop_ack.v1",
        "status": "STOP_observed",
        "server_id": os.environ["IMAGE3F_SERVER_ID"],
        "pid": os.getpid(),
        "stop_sentinel_path": str(stop.resolve(strict=True)),
        "stop_sentinel_sha256": hashlib.sha256(stop_bytes).hexdigest(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    ack.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(ack, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
