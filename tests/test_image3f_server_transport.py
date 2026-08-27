"""Socket-level protocol-v3 gates for the real Image3F server module."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from integrations.gr00t import image3f_frame_client as protocol


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "scripts/server/image3f_server.py"
SERVER_DIR = SERVER_PATH.parent
STANDALONE_SBATCH_PATH = ROOT / "scripts/server/image3f_server.sbatch"
ARRAY_SBATCH_PATH = ROOT / "scripts/server/image3f_server_array.sbatch"
LOCAL_FLEET_PATH = ROOT / "scripts/server/local_image3f_fleet.py"
TRANSFER_PATCH_PATH = (
    ROOT / "transfer/0004-Harden-production-DAgger-runtime-and-evaluation.patch"
)
TRANSFER_README_PATH = ROOT / "transfer/README.md"


def load_server_module(queue: Path):
    runtime = types.ModuleType("runtime")

    class Image3FRequest:
        def __init__(self, **values):
            self.__dict__.update(values)

    class Image3FSession:
        pass

    runtime.Image3FRequest = Image3FRequest
    runtime.Image3FSession = Image3FSession
    name = f"image3f_server_transport_test_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(
            os.environ,
            {
                "IMAGE3F_ROOT": str(ROOT),
                "IMAGE3F_QUEUE": str(queue),
                "IMAGE3F_SERVER_ID": "transport-test:000",
            },
            clear=False,
        ),
        mock.patch.dict(sys.modules, {"runtime": runtime}),
    ):
        sys.path.insert(0, str(SERVER_DIR))
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(SERVER_DIR))
    return module


class StubReferences:
    def stats(self):
        return {"streams": 0, "frames": 0, "bytes": 0, "in_flight": 0}

    def reset(self, _stream_id):
        return False


class CountingSession:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, *_args, **_kwargs):
        self.generate_calls += 1
        raise AssertionError("token mismatch reached inference")


class Image3FServerTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.module = load_server_module(Path(cls.temporary.name) / "queue")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _handler_pair(self, *, incarnation="c" * 32):
        module = self.module
        server_sock, client_sock = socket.socketpair()
        connections = module.ConnectionRegistry(
            max_connections=4, client_socket_timeout_s=2.0
        )
        session = CountingSession()
        thread = threading.Thread(
            target=module.handle_conn,
            args=(
                server_sock,
                ("local", 0),
                session,
                threading.Lock(),
                StubReferences(),
                module.ServerMetrics(),
                {"runtime": "test"},
                {"renderer": "test"},
                "d" * 64,
                incarnation,
                connections,
            ),
            daemon=True,
        )
        connections.start_handler(server_sock, thread)
        return client_sock, thread, session

    def test_v2_is_rejected_and_token_mismatch_never_reaches_inference(self):
        incarnation = "c" * 32
        client_sock, thread, session = self._handler_pair(incarnation=incarnation)
        try:
            protocol._send_msg(client_sock, {"cmd": "ping", "protocol_version": 2})
            rejected = protocol._recv_msg(client_sock)
            self.assertIs(rejected["ok"], False)
            self.assertIn("version mismatch", rejected["error"])

            protocol._send_msg(
                client_sock, {"cmd": "ping", "protocol_version": 3}
            )
            ping = protocol._recv_msg(client_sock)
            self.assertIs(ping["ok"], True)
            self.assertEqual(ping["protocol_version"], 3)
            self.assertEqual(ping["server_incarnation_id"], incarnation)
            self.assertIs(
                ping["capabilities"]["server_incarnation_binding"], True
            )

            protocol._send_msg(
                client_sock,
                {
                    "cmd": "render_frame",
                    "protocol_version": 3,
                    "expected_server_incarnation_id": "e" * 32,
                },
            )
            mismatch = protocol._recv_msg(client_sock)
            self.assertIs(mismatch["ok"], False)
            self.assertIn("incarnation changed", mismatch["error"])
            self.assertEqual(session.generate_calls, 0)
        finally:
            client_sock.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_connection_limit_rejects_excess_and_idle_socket_times_out(self):
        module = self.module
        with mock.patch.dict(
            os.environ, {"IMAGE3F_BIND_HOST": "127.0.0.1"}, clear=False
        ):
            listener_socket, port = module.create_tcp_server(0)
        stopped = threading.Event()
        connections = module.ConnectionRegistry(
            max_connections=1, client_socket_timeout_s=0.2
        )
        listener = threading.Thread(
            target=module.tcp_listener,
            args=(
                listener_socket,
                CountingSession(),
                threading.Lock(),
                StubReferences(),
                module.ServerMetrics(),
                {"runtime": "test"},
                {"renderer": "test"},
                "d" * 64,
                "c" * 32,
                stopped,
                connections,
            ),
            daemon=True,
        )
        listener.start()
        first = socket.create_connection(("127.0.0.1", port), timeout=1)
        deadline = time.monotonic() + 1
        while connections.count() != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(connections.count(), 1)
        second = socket.create_connection(("127.0.0.1", port), timeout=1)
        second.settimeout(1)
        try:
            self.assertEqual(second.recv(1), b"")
            self.assertEqual(connections.rejected_connections, 1)
            deadline = time.monotonic() + 1
            while connections.count() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(connections.count(), 0)
        finally:
            first.close()
            second.close()
            stopped.set()
            listener_socket.close()
            listener.join(timeout=2)
        self.assertFalse(listener.is_alive())

    def test_protocol_v3_exact_three_view_input_contract(self):
        depth = {
            view: np.zeros((240, 416), dtype=np.uint8)
            for view in ("head_color", "hand_left", "hand_right")
        }
        rgb = {
            view: np.zeros((240, 416, 3), dtype=np.uint8)
            for view in ("head_color", "hand_left", "hand_right")
        }
        self.module.validate_render_inputs(depth, rgb, rgb)
        malformed = dict(depth)
        malformed["hand_right"] = np.zeros((120, 208), dtype=np.uint8)
        with self.assertRaisesRegex(
            self.module.Image3FWireError, "must be uint8"
        ):
            self.module.validate_render_inputs(malformed, rgb, rgb)

    def test_bind_topologies_are_explicit_and_consistent(self):
        standalone = STANDALONE_SBATCH_PATH.read_text()
        array = ARRAY_SBATCH_PATH.read_text()
        local_fleet = LOCAL_FLEET_PATH.read_text()

        for remote_launcher in (standalone, array):
            self.assertIn(
                "IMAGE3F_BIND_HOST=${IMAGE3F_BIND_HOST:-0.0.0.0}",
                remote_launcher,
            )
        self.assertIn("hostname -f", standalone)
        self.assertIn('"IMAGE3F_ADVERTISE_HOST": args.advertise_host', local_fleet)
        self.assertIn('"IMAGE3F_BIND_HOST": args.advertise_host', local_fleet)
        self.assertIn('default="127.0.0.1"', local_fleet)

    def test_transfer_patch_cannot_reintroduce_legacy_client_protocol(self):
        transfer_patch = TRANSFER_PATCH_PATH.read_text()
        transfer_readme = TRANSFER_README_PATH.read_text()

        self.assertIn("+PROTOCOL_VERSION = 3", transfer_patch)
        self.assertNotIn("+PROTOCOL_VERSION = 2", transfer_patch)
        self.assertNotIn("trusted cluster protocol", transfer_patch.lower())
        self.assertIn("git apply ../transfer/0004-*.patch", transfer_readme)


if __name__ == "__main__":
    unittest.main()
