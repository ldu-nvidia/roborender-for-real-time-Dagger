import itertools
import os
import pickle
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from integrations.gr00t import image3f_frame_client as client_module


def stateful_capabilities(reference_lag_max=512):
    return {
        "lagged_reference_history": True,
        "server_incarnation_binding": True,
        "reference_lag_max": reference_lag_max,
        "output_views": ["head_color", "hand_left", "hand_right"],
        "output_layout": "vertical",
        "output_tile_shape_hwc": [240, 416, 3],
        "output_stacked_shape_hwc": [720, 416, 3],
    }


class FakeImage3FServer:
    def __init__(self, requests, *, reference_lag_max=512):
        self.requests = requests
        self.reference_lag_max = reference_lag_max
        self.received = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.incarnation = "1" * 32
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        conn, _ = self.server.accept()
        with conn:
            for _ in range(self.requests):
                request = client_module._recv_msg(conn)
                self.received.append(request)
                if request["cmd"] == "ping":
                    response = {
                        "ok": True,
                        "protocol_version": client_module.PROTOCOL_VERSION,
                        "server_incarnation_id": self.incarnation,
                        "capabilities": stateful_capabilities(
                            self.reference_lag_max
                        ),
                    }
                else:
                    step = int(request.get("step", 0))
                    lag = int(request.get("reference_lag", 0))
                    if step == 0:
                        source, kind, source_step = "bootstrap_anchor", "bootstrap_anchor", -1
                    elif step < lag:
                        source, kind, source_step = "generated_episode_frame0", "generated", 0
                    else:
                        source, kind, source_step = f"generated_t-{lag}", "generated", step - lag
                    decoded = {
                        view: client_module._decode_array(value)
                        for view, value in request["prev"].items()
                    }
                    digest = client_module._image_mapping_sha256(decoded)
                    response = {
                        "ok": True,
                        "views": request["prev"],
                        "reference_source": source,
                        "reference_provenance": {
                            "schema": client_module.REFERENCE_PROVENANCE_SCHEMA,
                            "request_step": step,
                            "configured_lag": lag,
                            "kind": kind,
                            "source_step": source_step,
                            "reference_sha256": digest,
                        },
                        "generated_sha256": digest,
                        "stream_step": step,
                        "server_incarnation_id": self.incarnation,
                    }
                client_module._send_msg(conn, response)
        self.server.close()

    def start(self):
        self.thread.start()


class RestartBetweenRetryServer:
    """Drop step zero, change incarnation, and reject the bound retry."""

    def __init__(self):
        self.first_incarnation = "a" * 32
        self.second_incarnation = "b" * 32
        self.inference_count = 0
        self.render_requests = []
        self.error = None
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(2)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _ping_response(incarnation):
        return {
            "ok": True,
            "protocol_version": client_module.PROTOCOL_VERSION,
            "server_incarnation_id": incarnation,
            "capabilities": stateful_capabilities(),
        }

    def _run(self):
        try:
            first, _ = self.server.accept()
            with first:
                handshake = client_module._recv_msg(first)
                if handshake.get("cmd") != "ping":
                    raise AssertionError("first stateful request was not a handshake")
                client_module._send_msg(
                    first, self._ping_response(self.first_incarnation)
                )
                render = client_module._recv_msg(first)
                self.render_requests.append(render)
                if render.get("expected_server_incarnation_id") != self.first_incarnation:
                    raise AssertionError("first render was not bound to its handshake")
                self.inference_count += 1
                # Model a committed step-zero result whose response is lost.
            second, _ = self.server.accept()
            with second:
                retry = client_module._recv_msg(second)
                self.render_requests.append(retry)
                if retry.get("expected_server_incarnation_id") != self.first_incarnation:
                    raise AssertionError("retry lost the original incarnation binding")
                client_module._send_msg(
                    second,
                    {
                        "ok": False,
                        "error": "server incarnation changed",
                        "retryable": False,
                        "server_incarnation_id": self.second_incarnation,
                    },
                )
        except BaseException as exc:  # surfaced by the test thread join
            self.error = exc
        finally:
            self.server.close()

    def start(self):
        self.thread.start()


class Image3FFrameClientTest(unittest.TestCase):
    def test_wire_decoder_rejects_globals_and_oversized_frames(self):
        payload = pickle.dumps(Path("/tmp/not-constructed"))
        with self.assertRaisesRegex(pickle.UnpicklingError, "forbidden global"):
            client_module._safe_pickle_loads(payload)

        receiver, sender = socket.socketpair()
        try:
            sender.sendall(struct.pack("!Q", 1025))
            with mock.patch.dict(
                os.environ, {"IMAGE3F_MAX_MESSAGE_BYTES": "1024"}, clear=True
            ):
                with self.assertRaisesRegex(
                    client_module.Image3FWireError, "exceeding the limit"
                ):
                    client_module._recv_msg(receiver)
        finally:
            receiver.close()
            sender.close()

    def test_stateful_client_rejects_renderer_process_restart(self):
        client = client_module.Image3FFrameClient("127.0.0.1", 1)
        client._bind_server_incarnation(
            {"server_incarnation_id": "a" * 32}, required=True
        )
        with self.assertRaisesRegex(
            client_module.Image3FWireError, "incarnation changed"
        ):
            client._bind_server_incarnation(
                {"server_incarnation_id": "b" * 32}, required=True
            )
        with self.assertRaisesRegex(
            client_module.Image3FWireError, "no valid server incarnation"
        ):
            client_module.Image3FFrameClient("127.0.0.1", 1)._bind_server_incarnation(
                {}, required=True
            )

    def test_ping_requires_exact_v3_stateful_capabilities(self):
        client = client_module.Image3FFrameClient("127.0.0.1", 1)
        base = {
            "ok": True,
            "protocol_version": client_module.PROTOCOL_VERSION,
            "server_incarnation_id": "a" * 32,
            "capabilities": stateful_capabilities(256),
        }
        client._validate_ping_response(base, required_reference_lag=256)
        old = dict(base, protocol_version=2)
        with self.assertRaisesRegex(
            client_module.Image3FWireError, "protocol version 3"
        ):
            client_module.Image3FFrameClient(
                "127.0.0.1", 1
            )._validate_ping_response(old)
        missing = dict(base, capabilities={"lagged_reference_history": True})
        with self.assertRaisesRegex(
            client_module.Image3FWireError, "incarnation binding"
        ):
            client_module.Image3FFrameClient(
                "127.0.0.1", 1
            )._validate_ping_response(missing)

        malformed_lag = dict(base, capabilities=stateful_capabilities(True))
        with self.assertRaisesRegex(
            client_module.Image3FWireError, "maximum reference lag is malformed"
        ):
            client_module.Image3FFrameClient(
                "127.0.0.1", 1
            )._validate_ping_response(malformed_lag)

        malformed_contracts = (
            ("output_views", ["hand_left", "head_color", "hand_right"]),
            ("output_layout", "horizontal"),
            ("output_tile_shape_hwc", [240, 415, 3]),
            ("output_stacked_shape_hwc", [719, 416, 3]),
        )
        for field, value in malformed_contracts:
            with self.subTest(field=field):
                capabilities = stateful_capabilities(256)
                capabilities[field] = value
                malformed = dict(base, capabilities=capabilities)
                with self.assertRaisesRegex(
                    client_module.Image3FWireError, "production contract"
                ):
                    client_module.Image3FFrameClient(
                        "127.0.0.1", 1
                    )._validate_ping_response(malformed)

    def test_cached_handshake_revalidates_lag_before_stateful_send(self):
        server = FakeImage3FServer(requests=1, reference_lag_max=128)
        server.start()
        client = client_module.Image3FFrameClient(
            "127.0.0.1", server.port, timeout_s=2
        )
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        condition = {"head_color": image}
        try:
            self.assertTrue(client.ping()["ok"])
            with self.assertRaisesRegex(
                client_module.Image3FWireError,
                "maximum reference lag 128 is below required 256",
            ):
                client.render_frame(
                    condition,
                    condition,
                    condition,
                    prompt="scene",
                    stream_id="run:rank0:env0",
                    episode_id=0,
                    step=0,
                    reference_lag=256,
                )
        finally:
            client.close()
            server.thread.join(timeout=2)
        self.assertFalse(server.thread.is_alive())
        self.assertEqual([request["cmd"] for request in server.received], ["ping"])

    def test_first_stateful_render_is_bound_before_lost_response_retry(self):
        server = RestartBetweenRetryServer()
        server.start()
        client = client_module.Image3FFrameClient(
            "127.0.0.1", server.port, timeout_s=2
        )
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        condition = {"head_color": image}
        try:
            with self.assertRaisesRegex(
                client_module.Image3FWireError, "incarnation changed"
            ):
                client.render_frame(
                    condition,
                    condition,
                    condition,
                    prompt="scene",
                    stream_id="run:rank0:env0",
                    episode_id=0,
                    step=0,
                    reference_lag=3,
                )
        finally:
            client.close()
            server.thread.join(timeout=3)
        self.assertFalse(server.thread.is_alive())
        if server.error is not None:
            raise server.error
        self.assertEqual(server.inference_count, 1)
        self.assertEqual(len(server.render_requests), 2)

    def test_production_output_contract_rejects_order_shape_and_dtype(self):
        good = {
            view: np.zeros((240, 416, 3), dtype=np.uint8)
            for view in ("head_color", "hand_left", "hand_right")
        }
        environment = {
            "IMAGE3F_EXPECT_OUTPUT_VIEWS": "head_color,hand_left,hand_right",
            "IMAGE3F_EXPECT_OUTPUT_SHAPE": "240,416,3",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            client_module._validate_output_views(good)
            reordered = {
                "hand_left": good["hand_left"],
                "head_color": good["head_color"],
                "hand_right": good["hand_right"],
            }
            with self.assertRaisesRegex(RuntimeError, "view order mismatch"):
                client_module._validate_output_views(reordered)
            malformed = dict(good)
            malformed["hand_right"] = np.zeros((240, 415, 3), dtype=np.uint8)
            with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
                client_module._validate_output_views(malformed)
            malformed = dict(good)
            malformed["hand_right"] = good["hand_right"].astype(np.float32)
            with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
                client_module._validate_output_views(malformed)

    def test_endpoint_manifest_shards_by_global_rank_and_local_environment(self):
        local_envs = 32
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            endpoints = []
            for index in range(52):
                endpoint = root / f"renderer_{index:03d}.txt"
                endpoint.write_text(
                    f"renderer-{index:03d}.example {20000 + index}\n",
                    encoding="utf-8",
                )
                endpoints.append(endpoint)
            manifest = root / "renderer_manifest.txt"
            manifest.write_text(
                "".join(f"{endpoint}\n" for endpoint in endpoints),
                encoding="utf-8",
            )

            # Each distributed rank is a separate process, so its rank-local
            # client counter starts at zero.  Verify both non-wrapped and
            # modulo-wrapped assignments over the complete 52-endpoint fleet.
            for rank in (0, 1, 7):
                with self.subTest(rank=rank):
                    environment = {
                        "IMAGE3F_ENDPOINT_MANIFEST": str(manifest),
                        "IMAGE3F_LOCAL_ENVS": str(local_envs),
                        "RANK": str(rank),
                    }
                    with (
                        mock.patch.dict(os.environ, environment, clear=True),
                        mock.patch.object(
                            client_module.Image3FFrameClient,
                            "_counter",
                            itertools.count(),
                        ),
                    ):
                        clients = [
                            client_module.Image3FFrameClient.from_endpoint_file(endpoints[0])
                            for _ in range(4)
                        ]

                    expected_global_slots = [rank * local_envs + slot for slot in range(4)]
                    expected_indices = [slot % len(endpoints) for slot in expected_global_slots]
                    self.assertEqual(
                        [Path(client.endpoint_file) for client in clients],
                        [endpoints[index] for index in expected_indices],
                    )
                    self.assertEqual(
                        [client.host for client in clients],
                        [f"renderer-{index:03d}.example" for index in expected_indices],
                    )
                    self.assertEqual(
                        [client.port for client in clients],
                        [20000 + index for index in expected_indices],
                    )
                    self.assertEqual(
                        [client._stream_slot for client in clients],
                        expected_global_slots,
                    )

    def test_endpoint_manifest_uses_independent_actor_rank_without_ddp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            endpoints = []
            for index in range(2):
                endpoint = root / f"renderer_{index:03d}.txt"
                endpoint.write_text(
                    f"renderer-{index:03d}.example {20000 + index}\n",
                    encoding="utf-8",
                )
                endpoints.append(endpoint)
            manifest = root / "renderer_manifest.txt"
            manifest.write_text(
                "".join(f"{endpoint}\n" for endpoint in endpoints),
                encoding="utf-8",
            )
            environment = {
                "IMAGE3F_ENDPOINT_MANIFEST": str(manifest),
                "IMAGE3F_LOCAL_ENVS": "1",
                "ROUND_DAGGER_ACTOR_RANK": "1",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                "IMAGE3F_RUN_ID": "independent-actor-test",
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(
                    client_module.Image3FFrameClient,
                    "_counter",
                    itertools.count(),
                ),
            ):
                # The trainer has already selected renderer 1.  The client's
                # legacy manifest remapper must not collapse isolated rank 1
                # back onto rank 0 merely because Torch RANK is absent.
                client = client_module.Image3FFrameClient.from_endpoint_file(endpoints[1])

            self.assertEqual(Path(client.endpoint_file), endpoints[1])
            self.assertEqual(client.host, "renderer-001.example")
            self.assertEqual(client.port, 20001)
            self.assertEqual(client._stream_slot, 1)
            self.assertEqual(client._stream_id, "independent-actor-test:rank1:env1")

    def test_protocol_and_automatic_episode_metadata(self):
        fake = FakeImage3FServer(requests=4)
        fake.start()
        client = client_module.Image3FFrameClient("127.0.0.1", fake.port, stream_slot=7)
        image = np.full((2, 3, 3), 11, dtype=np.uint8)
        conditions = {"head_color": image}

        self.assertTrue(client.ping()["ok"])
        client.render_frame(
            conditions,
            conditions,
            conditions,
            prompt="scene",
            reference_lag=3,
            trainer_step=17,
        )
        client.render_frame(
            conditions,
            conditions,
            conditions,
            prompt="scene",
            reference_lag=3,
        )
        # New arrays with identical bytes represent a newly sampled episode.
        new_conditions = {"head_color": image.copy()}
        client.render_frame(
            new_conditions,
            new_conditions,
            new_conditions,
            prompt="scene",
            reference_lag=3,
        )
        client.close()
        fake.thread.join(timeout=2)

        renders = fake.received[1:]
        self.assertTrue(
            all(
                request["protocol_version"] == client_module.PROTOCOL_VERSION
                for request in fake.received
            )
        )
        self.assertTrue(
            all(
                request["expected_server_incarnation_id"] == fake.incarnation
                for request in renders
            )
        )
        self.assertEqual([request["step"] for request in renders], [0, 1, 0])
        self.assertEqual([request["episode_id"] for request in renders], ["0", "0", "1"])
        self.assertTrue(all(request["reference_lag"] == 3 for request in renders))
        self.assertEqual(renders[0]["trainer_step"], 17)
        self.assertNotIn("trainer_step", renders[1])
        self.assertEqual(client.last_metadata["reference_provenance_valid"], True)

    def test_reference_provenance_gate_distinguishes_lag_boundary(self):
        frame_zero = {
            "schema": client_module.REFERENCE_PROVENANCE_SCHEMA,
            "request_step": 255,
            "configured_lag": 256,
            "kind": "generated",
            "source_step": 0,
            "reference_sha256": "a" * 64,
        }
        generated = {
            "schema": client_module.REFERENCE_PROVENANCE_SCHEMA,
            "request_step": 256,
            "configured_lag": 256,
            "kind": "generated",
            "source_step": 0,
            "reference_sha256": "b" * 64,
        }
        self.assertEqual(
            client_module._validate_reference_provenance(
                {
                    "reference_source": "generated_episode_frame0",
                    "reference_provenance": frame_zero,
                    "stream_step": 255,
                },
                step=255,
                lag=256,
            ),
            frame_zero,
        )
        self.assertEqual(
            client_module._validate_reference_provenance(
                {
                    "reference_source": "generated_t-256",
                    "reference_provenance": generated,
                    "stream_step": 256,
                },
                step=256,
                lag=256,
            ),
            generated,
        )
        with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
            client_module._validate_reference_provenance(
                {
                    "reference_source": "generated_episode_frame0",
                    "reference_provenance": frame_zero,
                    "stream_step": 256,
                },
                step=256,
                lag=256,
            )

    def test_renderer_identity_is_bound_when_cohort_supplies_digest(self):
        expected = "a" * 64
        with mock.patch.dict(
            os.environ, {"ROUND_DAGGER_RENDERER_SHA256": expected}, clear=True
        ):
            client_module._validate_renderer_identity(
                {"renderer_identity_sha256": expected}
            )
            with self.assertRaisesRegex(RuntimeError, "renderer identity mismatch"):
                client_module._validate_renderer_identity({})
            with self.assertRaisesRegex(RuntimeError, "renderer identity mismatch"):
                client_module._validate_renderer_identity(
                    {"renderer_identity_sha256": "b" * 64}
                )


if __name__ == "__main__":
    unittest.main()
