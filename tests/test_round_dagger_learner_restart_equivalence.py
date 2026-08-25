"""Tests for the semantic learner restart-equivalence gate."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/round_dagger/verify_learner_restart_equivalence.py"


def _load_gate():
    specification = importlib.util.spec_from_file_location(
        "round_dagger_restart_equivalence_test_gate", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "restart-equivalence tests require PyTorch")
class RestartEquivalenceGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gate = _load_gate()

    def tearDown(self):
        self.temporary.cleanup()

    def _checkpoint(self, path: Path, *, weight: float = 1.0) -> None:
        state = {"linear.weight": torch.tensor([[weight]], dtype=torch.float32)}
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v1",
                "checkpoint_schema": "gr00t.trl.full_state.v2",
                "policy_state_dict": state,
                "policy_parameter_names": ["linear.weight"],
                "campaign_id": "test",
                "macro_round": 0,
                "model_lineage_sha256": "1" * 64,
                "code_sha256": "2" * 64,
                "learner_config_sha256": "3" * 64,
                "replay_snapshot_sha256": "4" * 64,
                "replay_snapshot_file_sha256": "5" * 64,
                "completed_steps": 2,
                "initial_loss": 1.25,
                "world_size": 1,
                "model_state_dict": state,
                "optimizer_state_dict": {
                    "state": {0: {"step": torch.tensor(2.0)}},
                    "param_groups": [{"params": [0], "lr": 1.0e-4}],
                },
                "lr_scheduler_state_dict": None,
                "value_state_dict": None,
                "rng_states": [
                    {
                        "python": (3, (1, 2, 3), None),
                        "numpy": ("MT19937", np.asarray([1, 2], dtype=np.uint32), 1, 0, 0.0),
                        "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
                        "torch_cuda": [],
                    }
                ],
            },
            path,
        )

    def test_identical_semantic_states_pass_even_when_files_have_distinct_archives(self):
        full = self.root / "full.pt"
        resumed = self.root / "resumed.pt"
        self._checkpoint(full)
        self._checkpoint(resumed)
        report = self.gate.verify(
            uninterrupted_checkpoint=full,
            resumed_checkpoint=resumed,
            expected_completed_steps=2,
        )
        self.assertTrue(report["semantic_equivalent"])
        self.assertIsNone(report["results_equivalent"])

    def test_one_tensor_difference_fails_closed(self):
        full = self.root / "full.pt"
        resumed = self.root / "resumed.pt"
        self._checkpoint(full)
        self._checkpoint(resumed, weight=1.0001)
        with self.assertRaisesRegex(
            self.gate.RestartEquivalenceError, "tensor bytes differ"
        ):
            self.gate.verify(
                uninterrupted_checkpoint=full,
                resumed_checkpoint=resumed,
                expected_completed_steps=2,
            )

    def test_completed_step_mismatch_fails_closed(self):
        full = self.root / "full.pt"
        resumed = self.root / "resumed.pt"
        self._checkpoint(full)
        self._checkpoint(resumed)
        with self.assertRaisesRegex(
            self.gate.RestartEquivalenceError, "completed_steps mismatch"
        ):
            self.gate.verify(
                uninterrupted_checkpoint=full,
                resumed_checkpoint=resumed,
                expected_completed_steps=3,
            )


if __name__ == "__main__":
    unittest.main()
