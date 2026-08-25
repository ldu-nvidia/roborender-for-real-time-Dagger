import socket
import threading
import unittest

import numpy as np

from integrations.gr00t import image3f_frame_client as protocol
from scripts.server.reference_history import LaggedReferenceStore


class StoreBackedServer:
    """Tiny protocol-2 server that exercises the real history state machine."""

    def __init__(self):
        self.store = LaggedReferenceStore(max_streams=4)
        self.reference_values = []
        self.replays = 0
        self.drop_first_response = True
        self.stop_event = threading.Event()
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
    def _reference_metadata(*, step, lag, source):
        """Mirror the production server's exact lag-reference attestation."""

        expected_source = "anchor" if step < lag else f"generated_t-{lag}"
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
                "kind": "anchor" if step < lag else "generated",
                "source_step": 0 if step < lag else step - lag,
            },
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
                    protocol._send_msg(conn, {"ok": True, "protocol_version": 2})
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
                replay = self.store.replay_with_source(
                    stream_id=stream,
                    episode_id=episode,
                    step=step,
                    request_id=request_id,
                )
                if replay is not None:
                    replay_views, reference_source = replay
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
                            **self._reference_metadata(
                                step=step,
                                lag=lag,
                                source=reference_source,
                            ),
                        },
                    )
                    continue
                prepared = False
                try:
                    reference, reference_source = self.store.prepare(
                        stream_id=stream,
                        episode_id=episode,
                        step=step,
                        lag=lag,
                        anchor=anchor,
                        request_id=request_id,
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
                            **self._reference_metadata(
                                step=step,
                                lag=lag,
                                source=reference_source,
                            ),
                        },
                    )
                except Exception as exc:
                    if prepared:
                        self.store.abort(stream_id=stream, episode_id=episode, step=step)
                    protocol._send_msg(conn, {"ok": False, "error": str(exc)})


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
            self.assertEqual(server.reference_values[:4], [99, 99, 99, 10])
            self.assertTrue(metadata[0]["replayed"])
            self.assertEqual(metadata[0]["reference_source"], "anchor")
            self.assertEqual(metadata[0]["reference_provenance"]["source_step"], 0)
            self.assertEqual(metadata[2]["reference_source"], "anchor")
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
            self.assertEqual(client.last_metadata["reference_source"], "anchor")
            self.assertEqual(client.last_metadata["stream_step"], 0)
        finally:
            client.close()
            server.stop()


if __name__ == "__main__":
    unittest.main()
