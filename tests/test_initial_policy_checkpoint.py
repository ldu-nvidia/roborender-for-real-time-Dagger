import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALLBACK = ROOT / (
    "GR00T-VisualSim2Real/gr00t/rl/trl/callbacks/model_save_callback.py"
)
CALLBACK_CONFIG = ROOT / (
    "GR00T-VisualSim2Real/gr00t/rl/config/callbacks/model_save.yaml"
)


class InitialPolicyCheckpointTest(unittest.TestCase):
    def test_callback_source_is_valid_and_saves_a_distinct_baseline(self):
        source = CALLBACK.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("def on_train_begin", source)
        self.assertIn('"initial_policy.pt"', source)
        self.assertIn("int(state.global_step) != 0", source)
        self.assertIn("refusing to overwrite iteration-zero baseline", source)

    def test_feature_is_off_by_default(self):
        config = CALLBACK_CONFIG.read_text(encoding="utf-8")
        self.assertIn("save_initial_checkpoint: false", config)

    def test_fixed_seed_checkpoint_discovery_cannot_confuse_baseline_for_update(self):
        gate = (ROOT / "scripts/eval/fixed_seed_gate.py").read_text(encoding="utf-8")
        self.assertIn('train_run_dir.glob("model_step_*.pt")', gate)
        self.assertNotIn("initial_policy.pt", gate)


if __name__ == "__main__":
    unittest.main()
