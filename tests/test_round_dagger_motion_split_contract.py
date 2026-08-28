"""Focused production gates for motion-split and seed-block identities."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import sha256_file
from round_dagger.motion_split_contract import (
    FIXED_SEEDS,
    HELDOUT_MOTIONS,
    MOTIONS_PER_SEED,
    MotionSplitContractError,
    load_verified_contract,
    seed_block,
    validate_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/"
    "33demos_675_775_heldout_256_v1.json"
)
PLAN = ROOT / "configs/round_dagger/production_heldout_motion_plan_8x32.json"
SPLIT_SHA = "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834"
PLAN_SHA = "394ba77b2c8ecbdb4904ef24894cda0b88bb5b606ab75e578ee0a91756630f19"


class MotionSplitProductionContractTest(unittest.TestCase):
    def _load(self):
        return load_verified_contract(
            manifest_path=SPLIT,
            manifest_sha256=SPLIT_SHA,
            plan_path=PLAN,
            plan_sha256=PLAN_SHA,
        )

    def test_checked_in_split_is_disjoint_and_maps_eight_exact_seed_blocks(self):
        plan = self._load()
        self.assertEqual(plan["partition"]["eval_count"], HELDOUT_MOTIONS)
        self.assertEqual(plan["partition"]["collection_training_count"], 39)
        self.assertEqual(
            [seed_block(plan, seed) for seed in FIXED_SEEDS],
            [(index * MOTIONS_PER_SEED, MOTIONS_PER_SEED) for index in range(8)],
        )
        self.assertEqual(
            len(
                {
                    case["motion_sha256"]
                    for case in plan["capture"]["cases"]
                }
            ),
            HELDOUT_MOTIONS,
        )

    def test_manifest_or_plan_byte_identity_mismatch_fails_closed(self):
        for keyword, value, message in (
            ("manifest_sha256", "0" * 64, "manifest SHA-256 mismatch"),
            ("plan_sha256", "0" * 64, "plan SHA-256 mismatch"),
        ):
            with self.subTest(keyword=keyword):
                arguments = {
                    "manifest_path": SPLIT,
                    "manifest_sha256": SPLIT_SHA,
                    "plan_path": PLAN,
                    "plan_sha256": PLAN_SHA,
                }
                arguments[keyword] = value
                with self.assertRaisesRegex(MotionSplitContractError, message):
                    load_verified_contract(**arguments)

    def test_duplicate_motion_or_changed_seed_mapping_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = json.loads(PLAN.read_text(encoding="utf-8"))
            changed = copy.deepcopy(document)
            changed["capture"]["seed_blocks"][1]["offset"] = 0
            path = root / "bad-plan.json"
            path.write_text(
                json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MotionSplitContractError, "mapping mismatch"):
                load_verified_contract(
                    manifest_path=SPLIT,
                    manifest_sha256=SPLIT_SHA,
                    plan_path=path,
                    plan_sha256=sha256_file(path),
                )

            changed = copy.deepcopy(document)
            duplicate = changed["capture"]["seed_blocks"][0]["cases"][0]
            changed["capture"]["seed_blocks"][0]["cases"][1][
                "motion_sha256"
            ] = duplicate["motion_sha256"]
            path.write_text(
                json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MotionSplitContractError, "mapping mismatch"):
                load_verified_contract(
                    manifest_path=SPLIT,
                    manifest_sha256=SPLIT_SHA,
                    plan_path=path,
                    plan_sha256=sha256_file(path),
                )

    def test_scenario_contract_must_cover_each_planned_seed_env_exactly_once(self):
        plan = self._load()
        scenarios = [
            {
                "seed": case["seed"],
                "env_index": case["env_index"],
                "scenario_id": case["planned_case_id"],
            }
            for case in plan["capture"]["cases"]
        ]
        validate_scenarios(plan, scenarios)
        scenarios[32]["seed"] = FIXED_SEEDS[0]
        with self.assertRaisesRegex(MotionSplitContractError, "mapping differs"):
            validate_scenarios(plan, scenarios)


if __name__ == "__main__":
    unittest.main()
