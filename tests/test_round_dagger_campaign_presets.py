from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path

from round_dagger import campaign_presets
from round_dagger.artifacts import load_strict_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONTRACTS = (
    ROOT / "configs/round_dagger/production_campaign_256x512.json",
    ROOT / "configs/round_dagger/production_learner_8rank_1200.json",
    ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json",
)


class CampaignPresetTests(unittest.TestCase):
    def _run_cli(self, *arguments: str) -> dict:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(campaign_presets.main(list(arguments)), 0)
        return json.loads(output.getvalue())

    def test_validation_profile_has_exact_bounded_shape(self) -> None:
        profile = campaign_presets.load_profile("validation_20m")
        self.assertEqual(
            profile["topology"],
            {
                "target_environments": 8,
                "active_environments": 8,
                "cohort_count": 1,
                "actor_ranks_per_cohort": 2,
                "environments_per_actor_rank": 4,
                "macro_round_count": 1,
                "steps_per_environment": 16,
                "transitions_per_cohort": 128,
                "transitions_per_macro_round": 128,
                "transitions_per_campaign": 128,
            },
        )
        self.assertEqual(profile["learner"]["world_size"], 8)
        self.assertEqual(profile["learner"]["optimizer_steps"], 20)
        self.assertEqual(profile["learner"]["save_every_steps"], 10)
        self.assertEqual(profile["evaluation"]["interval_steps"], 10)
        self.assertEqual(
            profile["evaluation"]["required_optimizer_steps"], [0, 10, 20]
        )
        self.assertEqual(profile["evaluation"]["heldout_cases"], 32)
        self.assertEqual(profile["time_budget"]["hard_wall_time_seconds"], 1200)

    def test_production_profile_is_an_exact_binding_not_a_redefinition(self) -> None:
        profile = campaign_presets.load_profile("production_256x512")
        campaign = load_strict_json(PRODUCTION_CONTRACTS[0])
        learner = load_strict_json(PRODUCTION_CONTRACTS[1])
        self.assertEqual(profile["topology"], campaign["topology"])
        self.assertEqual(
            profile["learner"]["world_size"], learner["allocation"]["world_size"]
        )
        self.assertEqual(
            profile["learner"]["optimizer_steps"],
            learner["optimization"]["optimizer_steps"],
        )
        bound = {row["path"]: row["sha256"] for row in profile["immutable_contracts"]}
        for path in PRODUCTION_CONTRACTS:
            relative = path.relative_to(ROOT).as_posix()
            self.assertEqual(bound[relative], sha256_file(path))

    def test_unknown_profile_and_unsafe_operator_name_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            campaign_presets.CampaignPresetError, "unknown campaign preset"
        ):
            campaign_presets.load_profile("ad_hoc_256")
        with self.assertRaisesRegex(
            campaign_presets.CampaignPresetError, "W&B run name"
        ):
            campaign_presets.select_profile(
                "validation_20m", wandb_run_name="bad name/with/slash"
            )

    def test_operator_binding_does_not_change_semantic_profile_identity(self) -> None:
        profile_path = campaign_presets.profile_path("validation_20m")
        raw_profile = profile_path.read_text(encoding="utf-8")
        self.assertNotIn("mini_prod_1", raw_profile)
        first = campaign_presets.select_profile(
            "validation_20m", wandb_run_name="mini_prod_1"
        )
        second = campaign_presets.select_profile(
            "validation_20m", wandb_run_name="mini_prod_2"
        )
        self.assertTrue(first["selection_only"])
        self.assertEqual(first["profile_sha256"], second["profile_sha256"])
        self.assertEqual(first["profile"], second["profile"])
        self.assertEqual(
            first["operator_bindings"], {"wandb_run_name": "mini_prod_1"}
        )
        self.assertNotEqual(first["operator_bindings"], second["operator_bindings"])

    def test_validation_and_production_namespaces_are_isolated(self) -> None:
        validation = campaign_presets.load_profile("validation_20m")
        production = campaign_presets.load_profile("production_256x512")
        self.assertNotEqual(
            validation["isolation"]["artifact_namespace"],
            production["isolation"]["artifact_namespace"],
        )
        self.assertEqual(validation["immutable_contracts"], [])
        self.assertFalse(
            validation["isolation"]["production_contract_mutation_allowed"]
        )
        self.assertFalse(
            production["isolation"]["production_contract_mutation_allowed"]
        )

    def test_every_cli_command_is_read_only_for_production_contracts(self) -> None:
        before = {
            path: (
                path.read_bytes(),
                os.lstat(path).st_mode,
                os.lstat(path).st_mtime_ns,
            )
            for path in PRODUCTION_CONTRACTS
        }
        listed = self._run_cli("list")
        shown = self._run_cli("show", "production_256x512")
        validated = self._run_cli("validate")
        selected = self._run_cli(
            "select", "validation_20m", "--wandb-run-name", "mini_prod_1"
        )
        self.assertEqual(len(listed["presets"]), 2)
        self.assertEqual(shown["preset_id"], "production_256x512")
        self.assertEqual(validated["status"], "valid")
        self.assertEqual(selected["preset_id"], "validation_20m")
        after = {
            path: (
                path.read_bytes(),
                os.lstat(path).st_mode,
                os.lstat(path).st_mtime_ns,
            )
            for path in PRODUCTION_CONTRACTS
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
