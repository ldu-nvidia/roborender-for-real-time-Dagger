"""Real-artifact miniature macro-round regression test."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "mini macro-round requires PyTorch")
class MiniMacroRoundTest(unittest.TestCase):
    def test_transfer_restart_evaluate_and_promote(self) -> None:
        source = ROOT / "scripts" / "round_dagger" / "smoke_mini_round.py"
        specification = importlib.util.spec_from_file_location(
            "round_dagger_mini_smoke", source
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        temp_parent = Path("/mnt/data1/ldu/tmp")
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
            report = module.run(Path(temporary) / "round")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["transitions"], 32)
        self.assertTrue(report["restart_exact"])
        self.assertEqual(report["telemetry_steps"], 40)
        self.assertLess(
            report["candidate_action_mse"], report["parent_action_mse"]
        )
        self.assertEqual(report["active_policy_version"], 1)


if __name__ == "__main__":
    unittest.main()
