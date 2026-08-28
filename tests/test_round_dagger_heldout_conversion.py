from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger.heldout_admission import PLANNED_CASE_DOMAIN, _semantic_sha
from round_dagger.heldout_conversion import (
    HeldoutConversionError,
    _array_sha,
    convert_capture_aggregate,
    verify_conversion_bundle,
)
from round_dagger.scenario_capture import EXPECTED_SEEDS


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


class HeldoutEvalConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.identity = {
            "policy_sha256": _sha(1),
            "runtime_policy_sha256": _sha(2),
            "teacher_sha256": _sha(3),
            "renderer_sha256": _sha(4),
            "beta_sha256": artifacts.canonical_json_sha256({"beta": 1.0}),
            "config_sha256": _sha(6),
        }
        self._make_fixed_sources()
        self._make_captures()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_fixed_sources(self) -> None:
        motions = [
            {
                "path": f"motion-{index:03d}.npz",
                "sha256": _sha(100 + index),
                "split": "eval" if index < 256 else "collection",
            }
            for index in range(295)
        ]
        self.split = self.root / "split.json"
        self.split.write_text(
            json.dumps(
                {
                    "schema": "roborender.motion_split.v1",
                    "dataset": "test",
                    "counts": {"total": 295, "collection": 39, "eval": 256},
                    "inventory_sha256": _sha(500),
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
            seed = EXPECTED_SEEDS[index // 32]
            env_index = index % 32
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
                    "scenario_id": _sha(1000 + index),
                }
            )
        self.cases = cases
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "schema": "fast_roborender.round_dagger.heldout_motion_capture_plan.v2",
                    "capture": {
                        "case_count": 256,
                        "fixed_seeds": list(EXPECTED_SEEDS),
                        "motions_per_seed": 32,
                        "seed_count": 8,
                        "unique_motion_count": 256,
                        "contract": contract,
                        "contract_sha256": contract_sha,
                        "cases": cases,
                    },
                    "source": {
                        "nested_split_manifest_sha256": artifacts.sha256_file(
                            self.split
                        )
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

    def _make_captures(self) -> None:
        self.lineages: list[dict[str, object]] = []
        aggregate_rows = []
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            ids = np.arange(seed_index * 32, (seed_index + 1) * 32, dtype=np.int64)
            expert = np.arange(32 * 26, dtype=np.float32).reshape(1, 32, 26)
            student = expert + np.float32(seed_index + 1)
            arrays = {
                "policy_rgb": np.stack(
                    [np.full((6, 4, 3), int(env) % 251, np.uint8) for env in ids]
                )[None],
                "expert_actions": expert,
                "student_actions": student,
                "behavior_actions": student.copy(),
                "expert_mask": np.zeros((1, 32), np.bool_),
                "dones": np.zeros((1, 32), np.bool_),
                "observation_is_raytraced": np.zeros((1, 32), np.bool_),
                "global_env_ids": ids[None],
                "episode_ids": ids[None].copy(),
                "episode_steps": np.zeros((1, 32), np.int64),
                "obs__actor_obs": ids.astype(np.float32).reshape(1, 32, 1),
                "teacher_obs": ids.astype(np.float32).reshape(1, 32, 1) + 0.5,
            }
            capture = self.root / f"capture-{seed}.npz"
            np.savez(capture, **arrays)
            lineage_path = self.root / f"lineage-{seed}.json"
            lineage_path.write_text("{}")
            lineage = {
                "seed": seed,
                "path": str(lineage_path),
                "file_sha256": artifacts.sha256_file(lineage_path),
                "lineage_sha256": _sha(2000 + seed_index),
                "capture": {
                    "path": str(capture),
                    "sha256": artifacts.sha256_file(capture),
                    "array_sha256": {
                        key: _array_sha(value) for key, value in sorted(arrays.items())
                    },
                },
                "provenance": {
                    "parent_checkpoint_sha256": self.identity["policy_sha256"],
                    "parent_runtime_policy_sha256": self.identity[
                        "runtime_policy_sha256"
                    ],
                    "renderer_cohort_sha256": self.identity["renderer_sha256"],
                    "evaluation_config_sha256": self.identity["config_sha256"],
                },
                "cases": self.cases[seed_index * 32 : (seed_index + 1) * 32],
            }
            self.lineages.append(lineage)
            aggregate_rows.append(
                {
                    "seed": seed,
                    "path": str(lineage_path),
                    "file_sha256": lineage["file_sha256"],
                    "lineage_sha256": lineage["lineage_sha256"],
                    "capture_sha256": lineage["capture"]["sha256"],
                }
            )
        self.aggregate_path = self.root / "aggregate.json"
        self.aggregate_path.write_text("{}")
        self.aggregate = {
            "path": str(self.aggregate_path),
            "file_sha256": artifacts.sha256_file(self.aggregate_path),
            "aggregate_sha256": _sha(3000),
            "plan_sha256": artifacts.sha256_file(self.plan),
            "lineages": aggregate_rows,
        }

    def _verification_patches(self):
        by_path = {item["path"]: item for item in self.lineages}
        return (
            mock.patch(
                "round_dagger.heldout_conversion.verify_capture_aggregate",
                return_value=self.aggregate,
            ),
            mock.patch(
                "round_dagger.heldout_conversion.verify_capture_lineage",
                side_effect=lambda path: by_path[str(path)],
            ),
        )

    def test_exact_8x32_conversion_strips_only_teacher_and_recursively_verifies(self) -> None:
        campaign_root = self.root / "converted"
        aggregate_patch, lineage_patch = self._verification_patches()
        with aggregate_patch, lineage_patch:
            report = convert_capture_aggregate(
                self.aggregate_path,
                campaign_root=campaign_root,
                campaign_id="heldout-eval-test",
                identity=self.identity,
                expected_beta=1.0,
                plan=self.plan,
                motion_split=self.split,
                scenarios=self.scenarios,
            )
            verified = verify_conversion_bundle(report["path"])
        self.assertEqual(report["case_count"], 256)
        self.assertEqual(verified["cohort_count"], 8)
        for seed in EXPECTED_SEEDS:
            cohort = campaign_root / f"eval_seed_{seed:03d}"
            seal = artifacts.verify_cohort(cohort)
            self.assertEqual(seal["purpose"], "eval")
            payload = cohort / seal["shards"][0]["payload_path"]
            with np.load(payload, allow_pickle=False) as archive:
                self.assertNotIn("teacher_obs", archive.files)
                self.assertIn("obs__actor_obs", archive.files)

    def test_fails_before_publication_on_provenance_mismatch(self) -> None:
        bad = dict(self.identity, renderer_sha256=_sha(9999))
        campaign_root = self.root / "must-not-exist"
        aggregate_patch, lineage_patch = self._verification_patches()
        with aggregate_patch, lineage_patch, self.assertRaisesRegex(
            HeldoutConversionError, "capture provenance"
        ):
            convert_capture_aggregate(
                self.aggregate_path,
                campaign_root=campaign_root,
                campaign_id="heldout-eval-test",
                identity=bad,
                expected_beta=1.0,
                plan=self.plan,
                motion_split=self.split,
                scenarios=self.scenarios,
            )
        self.assertFalse(campaign_root.exists())

    def test_recursive_verification_detects_changed_source_capture(self) -> None:
        campaign_root = self.root / "converted-tamper"
        aggregate_patch, lineage_patch = self._verification_patches()
        with aggregate_patch, lineage_patch:
            report = convert_capture_aggregate(
                self.aggregate_path,
                campaign_root=campaign_root,
                campaign_id="heldout-eval-tamper",
                identity=self.identity,
                expected_beta=1.0,
                plan=self.plan,
                motion_split=self.split,
                scenarios=self.scenarios,
            )
            capture = Path(self.lineages[0]["capture"]["path"])
            with capture.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(HeldoutConversionError, "source capture"):
                verify_conversion_bundle(report["path"])


if __name__ == "__main__":
    unittest.main()
