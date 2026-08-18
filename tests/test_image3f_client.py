import socket
import threading
import unittest

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
                    response = {
                        "ok": True,
                        "views": request["prev"],
                        "reference_source": "anchor",
                    }
                client_module._send_msg(conn, response)
        self.server.close()

    def start(self):
        self.thread.start()


class Image3FFrameClientTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
