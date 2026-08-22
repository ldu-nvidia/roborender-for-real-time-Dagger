from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_shutdown_control():
    path = ROOT / "scripts/server/shutdown_control.py"
    spec = importlib.util.spec_from_file_location("shutdown_control_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shutdown_control = load_shutdown_control()


class StopControllerTests(unittest.TestCase):
    def test_stop_is_polled_and_acknowledged_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stop = base / "STOP"
            ack = base / "stop_acks/renderer_000.json"
            controller = shutdown_control.StopController(stop, ack, 0.05)
            stopped = threading.Event()
            watcher = threading.Thread(target=controller.watch, args=(stopped,))
            watcher.start()
            self.assertFalse(stopped.wait(0.1))
            stop.write_text("stop\n", encoding="utf-8")
            self.assertTrue(stopped.wait(1.0))
            watcher.join(timeout=1.0)
            self.assertIsNone(controller.error)
            payload = controller.acknowledge("fixture:000")
            self.assertEqual(payload["status"], "STOP_observed")
            self.assertEqual(payload["server_id"], "fixture:000")
            self.assertEqual(
                json.loads(ack.read_text(encoding="utf-8")), payload
            )
            self.assertFalse(any(ack.parent.glob(f".{ack.name}.*")))

    def test_poll_interval_and_ack_contract_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at most 2"):
            shutdown_control.StopController(Path("STOP"), Path("ACK"), 2.01)
        with self.assertRaisesRegex(ValueError, "ACK_FILE is required"):
            shutdown_control.StopController(Path("STOP"), None, 1.0)

    def test_symlink_stop_is_rejected_without_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.write_text("stop\n", encoding="utf-8")
            stop = base / "STOP"
            stop.symlink_to(target)
            controller = shutdown_control.StopController(
                stop, base / "renderer_000.json", 0.05
            )
            stopped = threading.Event()
            watcher = threading.Thread(target=controller.watch, args=(stopped,))
            watcher.start()
            self.assertTrue(stopped.wait(1.0))
            watcher.join(timeout=1.0)
            self.assertIsInstance(controller.error, RuntimeError)
            self.assertFalse((base / "renderer_000.json").exists())


if __name__ == "__main__":
    unittest.main()
