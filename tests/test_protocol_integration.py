import socket
import threading
import unittest

import numpy as np

from integrations.gr00t import image3f_frame_client as protocol
from scripts.server.reference_history import LaggedReferenceStore


class StoreBackedServer:
    """Tiny protocol-3 server that exercises the real history state machine."""

    def __init__(self):
        self.store = LaggedReferenceStore(max_streams=4)
        self.reference_values = []
        self.replays = 0
        self.drop_first_response = True
        self.stop_event = threading.Event()
        self.incarnation = "2" * 32
        self.server = socket.socket()
        self.server.settimeout(0.1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(8)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.server.close()
        self.thread.join(timeout=2)

    @staticmethod
    def _reference_metadata(*, step, lag, source, reference_sha256, generated):
        """Mirror the production server's exact lag-reference attestation."""

        if step == 0:
            expected_source, kind, source_step = "bootstrap_anchor", "bootstrap_anchor", -1
        elif step < lag:
            expected_source, kind, source_step = "generated_episode_frame0", "generated", 0
        else:
            expected_source, kind, source_step = f"generated_t-{lag}", "generated", step - lag
        if source != expected_source:
            raise RuntimeError(
                "reference history/source mismatch: "
                f"step={step} lag={lag} expected={expected_source!r} got={source!r}"
            )
        return {
            "reference_source": source,
            "reference_provenance": {
                "schema": protocol.REFERENCE_PROVENANCE_SCHEMA,
                "request_step": step,
                "configured_lag": lag,
                "kind": kind,
                "source_step": source_step,
                "reference_sha256": reference_sha256,
            },
            "generated_sha256": protocol._image_mapping_sha256(generated),
            "stream_step": step,
        }

    def _run(self):
        while not self.stop_event.is_set():
            try:
                conn, _ = self.server.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            while not self.stop_event.is_set():
                try:
                    request = protocol._recv_msg(conn)
                except (ConnectionError, OSError):
                    return
                if request["cmd"] == "ping":
                    protocol._send_msg(
                        conn,
                        {
                            "ok": True,
                            "protocol_version": protocol.PROTOCOL_VERSION,
                            "server_incarnation_id": self.incarnation,
                            "capabilities": {
                                "lagged_reference_history": True,
                                "server_incarnation_binding": True,
                                "reference_lag_max": 512,
                                "output_views": [
                                    "head_color",
                                    "hand_left",
                                    "hand_right",
                                ],
                                "output_layout": "vertical",
                                "output_tile_shape_hwc": [240, 416, 3],
                                "output_stacked_shape_hwc": [720, 416, 3],
                            },
                        },
                    )
                    continue
                stream = request["stream_id"]
                episode = request["episode_id"]
                step = request["step"]
                lag = request["reference_lag"]
                request_id = request["request_id"]
                anchor = {
                    view: protocol._decode_array(image)
                    for view, image in request["prev"].items()
                }
                replay = self.store.replay_with_provenance(
                    stream_id=stream,
                    episode_id=episode,
                    step=step,
                    request_id=request_id,
                )
                if replay is not None:
                    replay_views, reference_source, reference_sha256 = replay
                    self.replays += 1
                    protocol._send_msg(
                        conn,
                        {
                            "ok": True,
                            "views": {
                                view: protocol._encode_array(image)
                                for view, image in replay_views.items()
                            },
                            "replayed": True,
                            "server_incarnation_id": self.incarnation,
                            **self._reference_metadata(
                                step=step,
                                lag=lag,
                                source=reference_source,
                                reference_sha256=reference_sha256,
                                generated=replay_views,
                            ),
                        },
                    )
                    continue
                prepared = False
                try:
                    reference, reference_source, reference_sha256 = (
                        self.store.prepare_with_provenance(
                        stream_id=stream,
                        episode_id=episode,
                        step=step,
                        lag=lag,
                        anchor=anchor,
                        request_id=request_id,
                        )
                    )
                    prepared = True
                    self.reference_values.append(
                        int(next(iter(reference.values()))[0, 0, 0])
                    )
                    generated = {
                        view: np.full_like(image, int(step) + 10)
                        for view, image in anchor.items()
                    }
                    self.store.commit(
                        stream_id=stream,
                        episode_id=episode,
                        step=step,
                        generated=generated,
                        request_id=request_id,
                    )
                    prepared = False
                    if self.drop_first_response:
                        self.drop_first_response = False
                        return
                    protocol._send_msg(
                        conn,
                        {
                            "ok": True,
                            "views": {
                                view: protocol._encode_array(image)
                                for view, image in generated.items()
                            },
                            "server_incarnation_id": self.incarnation,
                            **self._reference_metadata(
                                step=step,
                                lag=lag,
                                source=reference_source,
                                reference_sha256=reference_sha256,
                                generated=generated,
                            ),
                        },
                    )
                except Exception as exc:
                    if prepared:
                        self.store.abort(stream_id=stream, episode_id=episode, step=step)
                    protocol._send_msg(
                        conn,
                        {
                            "ok": False,
                            "error": str(exc),
                            "server_incarnation_id": self.incarnation,
                        },
                    )


class ProtocolIntegrationTest(unittest.TestCase):
    def test_lost_response_replay_fixed_anchor_lag_and_episode_reset(self):
        server = StoreBackedServer()
        server.start()
        client = protocol.Image3FFrameClient("127.0.0.1", server.port, stream_slot=0)
        anchor = {"head_color": np.full((2, 3, 3), 99, dtype=np.uint8)}
        condition = {"head_color": np.zeros((2, 3), dtype=np.uint8)}
        mask = {"head_color": np.zeros((2, 3, 3), dtype=np.uint8)}
        try:
            metadata = []
            for step in range(4):
                output = client.render_frame(
                    condition,
                    mask,
                    anchor,
                    prompt="scene",
                    stream_id="run:rank0:env0",
                    episode_id=0,
                    step=step,
                    reference_lag=3,
                )
                self.assertEqual(int(output["head_color"][0, 0, 0]), step + 10)
                metadata.append(dict(client.last_metadata))
            self.assertGreaterEqual(server.replays, 1)
            self.assertEqual(server.reference_values[:4], [99, 10, 10, 10])
            self.assertTrue(metadata[0]["replayed"])
            self.assertEqual(metadata[0]["reference_source"], "bootstrap_anchor")
            self.assertEqual(metadata[0]["reference_provenance"]["source_step"], -1)
            self.assertEqual(metadata[2]["reference_source"], "generated_episode_frame0")
            self.assertEqual(metadata[3]["reference_source"], "generated_t-3")
            self.assertEqual(metadata[3]["reference_provenance"]["kind"], "generated")
            self.assertEqual(metadata[3]["reference_provenance"]["source_step"], 0)

            with self.assertRaisesRegex(RuntimeError, "anchor changed"):
                client.render_frame(
                    condition,
                    mask,
                    {"head_color": np.full((2, 3, 3), 7, dtype=np.uint8)},
                    prompt="scene",
                    stream_id="run:rank0:env0",
                    episode_id=0,
                    step=4,
                    reference_lag=3,
                )

            output = client.render_frame(
                condition,
                mask,
                {"head_color": np.full((2, 3, 3), 7, dtype=np.uint8)},
                prompt="scene",
                stream_id="run:rank0:env0",
                episode_id=1,
                step=0,
                reference_lag=3,
            )
            self.assertEqual(int(output["head_color"][0, 0, 0]), 10)
            self.assertEqual(client.last_metadata["reference_source"], "bootstrap_anchor")
            self.assertEqual(client.last_metadata["stream_step"], 0)
        finally:
            client.close()
            server.stop()


if __name__ == "__main__":
    unittest.main()
