import itertools
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from integrations.gr00t import image3f_frame_client as client_module


class FakeImage3FServer:
    def __init__(self, requests):
        self.requests = requests
        self.received = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        conn, _ = self.server.accept()
        with conn:
            for _ in range(self.requests):
                request = client_module._recv_msg(conn)
                self.received.append(request)
                if request["cmd"] == "ping":
                    response = {"ok": True, "protocol_version": 2}
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
                    }
                client_module._send_msg(conn, response)
        self.server.close()

    def start(self):
        self.thread.start()


class Image3FFrameClientTest(unittest.TestCase):
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
