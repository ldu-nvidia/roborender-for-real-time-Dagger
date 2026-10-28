"""No Docker/GPU required: deployment isolation and cleanup regression tests."""
from test.deployment.source_paths import ACTOR_ROOT
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = (ACTOR_ROOT / "qualify_graphics.py")
spec = importlib.util.spec_from_file_location("graphics_plan", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GraphicsPlanTest(unittest.TestCase):
    def test_only_gpu1_no_host_services_or_assets(self):
        cmd = module.command("sha256:" + "a" * 64, "probe-test", "/tmp/probe-test")
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=" + module.GPU_UUID)
        self.assertEqual(cmd[cmd.index("--network") + 1], "none")
        self.assertEqual(cmd.count("--mount"), 1)
        self.assertEqual(cmd[cmd.index("--memory") + 1], "32g")
        self.assertEqual(cmd[cmd.index("--cpus") + 1], "8")
        self.assertNotIn("--privileged", cmd)
        self.assertNotIn("--pid=host", cmd)
        self.assertNotIn("--ipc=host", cmd)

    def test_reject_floating_or_injected_image(self):
        for value in ("latest", "nvcr.io/nvidia/isaac-sim:6.0.0", "--gpus=all"):
            with self.assertRaises(ValueError):
                module.command(value, "probe-test", "/tmp/probe-test")

    def test_cleanup_cannot_stop_another_container(self):
        other = [{"Config": {"Labels": {}}, "State": {"Running": True}}]
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(other))) as run:
            with self.assertRaises(RuntimeError):
                module.cleanup_owned("probe-test")
            self.assertEqual(run.call_count, 1)

    def test_missing_container_is_noop(self):
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="")) as run:
            module.cleanup_owned("probe-test")
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
