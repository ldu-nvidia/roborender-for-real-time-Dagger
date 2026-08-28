from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts
from round_dagger.heldout_admission import (
    CASES_PER_SEED,
    PLANNED_CASE_DOMAIN,
    SEEDS,
    HeldoutAdmissionError,
    _semantic_sha,
    build_fixed_heldout_bundle,
    verify_fixed_heldout_bundle,
)


SHA = "a" * 64


class FixedHeldoutAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        motions = []
        for index in range(295):
            motions.append(
                {
                    "path": f"motion-{index:03d}.npz",
                    "sha256": f"{index + 1:064x}",
                    "split": "eval" if index < 256 else "collection",
                }
            )
        self.split = self.root / "split.json"
        self.split.write_text(
            json.dumps(
                {
                    "schema": "roborender.motion_split.v1",
                    "dataset": "test",
                    "counts": {"total": 295, "collection": 39, "eval": 256},
                    "inventory_sha256": SHA,
                    "selection": {"method": "test"},
                    "motions": motions,
                },
                sort_keys=True,
            )
        )
        contract = {
            "schema": "fast_roborender.round_dagger.scenario_capture_contract.v1",
            "policy_views": ["head_color", "hand_left", "hand_right"],
        }
        contract_sha = _semantic_sha(contract)
        cases = []
        scenarios = []
        for index in range(256):
            seed = SEEDS[index // CASES_PER_SEED]
            env_index = index % CASES_PER_SEED
            motion = motions[index]
            case = {
                "seed": seed,
                "env_index": env_index,
                "global_case_index": index,
                "motion_relative_path": motion["path"],
                "motion_sha256": motion["sha256"],
            }
            case["planned_case_id"] = _semantic_sha(
                {
                    "domain": PLANNED_CASE_DOMAIN,
                    **case,
                    "capture_contract_sha256": contract_sha,
                }
            )
            cases.append(case)
            scenarios.append(
                {
                    "seed": seed,
                    "env_index": env_index,
                    "scenario_id": f"{index + 1000:064x}",
                }
            )
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "schema": "fast_roborender.round_dagger.heldout_motion_capture_plan.v2",
                    "capture": {
                        "case_count": 256,
                        "fixed_seeds": list(SEEDS),
                        "motions_per_seed": 32,
                        "seed_count": 8,
                        "unique_motion_count": 256,
                        "contract": contract,
                        "contract_sha256": contract_sha,
                        "cases": cases,
                    },
                    "source": {
                        "nested_split_manifest_sha256": artifacts.sha256_file(self.split)
                    },
                    "partition": {
                        "eval_count": 256,
                        "collection_training_count": 39,
                        "path_overlap_count": 0,
                        "content_overlap_count": 0,
                    },
                },
                sort_keys=True,
            )
        )
        self.scenarios = self.root / "scenarios.json"
        self.scenarios.write_text(json.dumps(scenarios, sort_keys=True))
        self.cohorts = []
        for cohort_index in range(8):
            cohort = self.root / f"cohort-{cohort_index}"
            envs = list(range(cohort_index * 32, (cohort_index + 1) * 32))
            artifacts.create_cohort(
                cohort,
                campaign_id="heldout-test",
                macro_round=0,
                cohort_index=cohort_index,
                purpose="eval",
                expected_ranks=[0],
                expected_global_env_ids=envs,
                expected_transitions=32,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [26]},
                policy_sha256=SHA,
                runtime_policy_sha256=SHA,
                teacher_sha256=SHA,
                renderer_sha256=SHA,
                beta_sha256=SHA,
                config_sha256=SHA,
            )
            expert = np.arange(32 * 26, dtype=np.float32).reshape(1, 32, 26)
            arrays = {
                "policy_rgb": np.stack(
                    [np.full((6, 4, 3), env % 251, np.uint8) for env in envs]
                )[None],
                "expert_actions": expert,
                "student_actions": np.zeros_like(expert),
                "behavior_actions": np.zeros_like(expert),
                "expert_mask": np.zeros((1, 32), np.bool_),
                "dones": np.zeros((1, 32), np.bool_),
                "observation_is_raytraced": np.zeros((1, 32), np.bool_),
                "global_env_ids": np.asarray(envs, np.int64)[None],
                "episode_ids": np.asarray(envs, np.int64)[None],
                "episode_steps": np.zeros((1, 32), np.int64),
                "obs__actor_obs": np.asarray(envs, np.float32).reshape(1, 32, 1),
            }
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=arrays,
                campaign_id="heldout-test",
                macro_round=0,
                cohort_index=cohort_index,
                purpose="eval",
                policy_sha256=SHA,
                runtime_policy_sha256=SHA,
                teacher_sha256=SHA,
                renderer_sha256=SHA,
                beta_sha256=SHA,
                config_sha256=SHA,
            )
            artifacts.seal_cohort(cohort)
            self.cohorts.append(cohort)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_and_recursive_verify_fixed_256_case_bundle(self) -> None:
        result = build_fixed_heldout_bundle(
            self.cohorts,
            plan=self.plan,
            motion_split=self.split,
            scenarios=self.scenarios,
            snapshot_output=self.root / "snapshot.json",
            manifest_output=self.root / "case-manifest.json",
        )
        self.assertEqual(result["snapshot"]["transition_count"], 256)
        self.assertEqual(result["case_manifest"]["shard_count"], 8)
        verified = verify_fixed_heldout_bundle(
            result["snapshot"]["path"],
            result["case_manifest"]["path"],
            plan=self.plan,
            motion_split=self.split,
            scenarios=self.scenarios,
        )
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["case_count"], 256)

    def test_rejects_semantically_relabeled_scenario_contract(self) -> None:
        rows = json.loads(self.scenarios.read_text())
        rows[0]["seed"] = SEEDS[1]
        bad = self.root / "bad-scenarios.json"
        bad.write_text(json.dumps(rows))
        with self.assertRaisesRegex(HeldoutAdmissionError, "duplicate|cover"):
            build_fixed_heldout_bundle(
                self.cohorts,
                plan=self.plan,
                motion_split=self.split,
                scenarios=bad,
                snapshot_output=self.root / "unused-snapshot.json",
                manifest_output=self.root / "unused-manifest.json",
            )


if __name__ == "__main__":
    unittest.main()
