import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_ENV = (ROOT / "scripts/setup/check_env.sh").read_text(encoding="utf-8")


class CheckEnvProductionContractTest(unittest.TestCase):
    def test_trainer_manifest_must_match_the_staged_gr00t_revision(self):
        self.assertIn('MANIFEST_GR00T_REV=$(awk -F=', CHECK_ENV)
        self.assertIn('MANIFEST_GR00T_REV" = "$CURRENT_GR00T_REV', CHECK_ENV)

    def test_exact_three_view_integration_is_a_hard_preflight_gate(self):
        self.assertIn("scripts/scaling/install_gr00t_integration.py", CHECK_ENV)
        self.assertIn("--expected-revision", CHECK_ENV)
        self.assertIn('fail "three-view GR00T integration check failed"', CHECK_ENV)

    def test_target_local_engine_is_required_not_warned(self):
        self.assertIn("engine_build_manifest.json", CHECK_ENV)
        self.assertIn("I3F_CURRENT_ENGINES", CHECK_ENV)
        self.assertIn(
            'fail "no validated target-local Image3F TensorRT engine', CHECK_ENV
        )

    def test_exact_hydra_composition_report_is_a_hard_preflight_gate(self):
        self.assertIn("production_hydra_contract", CHECK_ENV)
        self.assertIn("production_hydra_contract_sha256", CHECK_ENV)
        self.assertIn("fast_roborender.production_hydra_contract.v1", CHECK_ENV)
        self.assertIn('payload["status"] == "validated"', CHECK_ENV)
        self.assertIn(
            'payload["policy_views"] == ["head_color", "hand_left", "hand_right"]',
            CHECK_ENV,
        )
        self.assertIn(
            'fail "production Hydra composition report is missing, stale, or invalid"',
            CHECK_ENV,
        )

    def test_slurm_account_query_uses_untruncated_pipe_output(self):
        self.assertIn("sacctmgr -nP show assoc", CHECK_ENV)
        self.assertIn("format=Account", CHECK_ENV)


if __name__ == "__main__":
    unittest.main()
