"""Tests for nested-manifest-aligned held-out motion capture planning."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import canonical_json_sha256
from round_dagger.evaluation import _validate_cases
from scripts.round_dagger.build_heldout_evaluation_plan import (
    CAPTURE_EVIDENCE_SCHEMA,
    COLLECTION_TRAINING_MOTION_COUNT,
    EVALUATION_MOTIONS_PER_SEED,
    EXPECTED_MOTION_COUNT,
    FIXED_SEEDS,
    HELDOUT_MOTION_COUNT,
    HeldoutPlanError,
    build_plan,
    finalize_capture,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/round_dagger/build_heldout_evaluation_plan.py"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class HeldoutEvaluationPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.motions = self.root / "motions"
        self.motions.mkdir()
        entries = []
        for index in range(EXPECTED_MOTION_COUNT):
            path = self.motions / f"motion_{index:03d}.npz"
            payload = b"fixture-motion-v2\0" + index.to_bytes(4, "big")
            path.write_bytes(payload)
            entries.append(
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "split": "collection" if index % 7 == 0 and index // 7 < 39 else "eval",
                }
            )
        # The expression above selects 39 collection cases (indices 0..266 by 7).
        self.assertEqual(sum(item["split"] == "collection" for item in entries), 39)
        inventory = hashlib.sha256(
            canonical_bytes(
                [
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in sorted(entries, key=lambda item: item["path"])
                ]
            )
        ).hexdigest()
        split = {
            "counts": {"collection": 39, "eval": 256, "total": 295},
            "dataset": "g1_wsdpt/33demos_675_775",
            "inventory_sha256": inventory,
            "motions": entries,
            "schema": "roborender.motion_split.v1",
            "selection": {
                "held_out_role": "eval",
                "method": "sha256_rank",
                "seed": "roborender-heldout-256-v1",
            },
        }
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        self.split_sha = hashlib.sha256(self.split.read_bytes()).hexdigest()
        self.plan_path = self.root / "outputs/capture_plan.json"
        self.evidence_path = self.root / "outputs/capture_evidence.json"
        self.finalization_path = self.root / "outputs/capture_finalization.json"
        self.scenario_path = self.root / "outputs/production_scenarios.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_kwargs(self):
        return {
            "motion_root": self.motions,
            "split_manifest": self.split,
            "expected_split_manifest_sha256": self.split_sha,
        }

    def build_cli(self):
        return [
            "python3",
            str(BUILDER),
            "build",
            "--motion-root",
            str(self.motions),
            "--split-manifest",
            str(self.split),
            "--expected-split-manifest-sha256",
            self.split_sha,
            "--manifest-output",
            str(self.plan_path),
        ]

    def test_plan_uses_nested_order_and_disjoint_seed_blocks(self) -> None:
        first = build_plan(**self.source_kwargs())
        second = build_plan(**self.source_kwargs())
        self.assertEqual(first, second)
        self.assertEqual(first["source"]["nested_split_manifest_sha256"], self.split_sha)
        self.assertEqual(first["partition"]["eval_count"], HELDOUT_MOTION_COUNT)
        self.assertEqual(
            first["partition"]["collection_training_count"],
            COLLECTION_TRAINING_MOTION_COUNT,
        )
        expected_eval = [
            f"motion_{index:03d}.npz"
            for index in range(EXPECTED_MOTION_COUNT)
            if not (index % 7 == 0 and index // 7 < 39)
        ]
        self.assertEqual(
            [item["relative_path"] for item in first["partition"]["eval_runtime_order"]],
            expected_eval,
        )
        self.assertEqual(
            first["capture"]["contract"]["policy_views"],
            ["head_color", "hand_left", "hand_right"],
        )
        self.assertEqual(first["capture"]["contract"]["view_layout"], "vertical")
        observed = set()
        for seed_index, seed in enumerate(FIXED_SEEDS):
            block = first["capture"]["seed_blocks"][seed_index]
            self.assertEqual(
                (block["seed"], block["offset"], block["limit"]),
                (seed, seed_index * 32, 32),
            )
            self.assertEqual([case["env_index"] for case in block["cases"]], list(range(32)))
            block_motions = {case["motion_sha256"] for case in block["cases"]}
            self.assertFalse(observed & block_motions)
            observed.update(block_motions)
            self.assertTrue(all("scenario_id" not in case for case in block["cases"]))
        self.assertEqual(len(observed), 256)

    def _build_plan_and_evidence(self):
        result = subprocess.run(self.build_cli(), text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(self.plan_path.read_text())
        plan_sha = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        cases = []
        for planned in plan["capture"]["cases"]:
            condition = hashlib.sha256(
                f"condition-{planned['global_case_index']}".encode()
            ).hexdigest()
            cases.append(
                {
                    **planned,
                    "first_condition_sha256": condition,
                    "live_condition_sha256": condition,
                    "live_reset_state_sha256": hashlib.sha256(
                        f"reset-{planned['global_case_index']}".encode()
                    ).hexdigest(),
                    "scenario_id": canonical_json_sha256(
                        {
                            "seed": planned["seed"],
                            "env_index": planned["env_index"],
                            "first_condition_sha256": condition,
                        }
                    ),
                }
            )
        bundles = [
            {
                "seed": seed,
                "path": str(self.root / f"first_condition_seed{seed}.npz"),
                "sha256": hashlib.sha256(f"bundle-{seed}".encode()).hexdigest(),
                "size": 1,
                "case_count": EVALUATION_MOTIONS_PER_SEED,
                "source_pass_index": 1,
            }
            for seed in FIXED_SEEDS
        ]
        evidence = {
            "schema": CAPTURE_EVIDENCE_SCHEMA,
            "plan_sha256": plan_sha,
            "cases": list(reversed(cases)),
            "first_condition_bundles": bundles,
            "first_condition_bundle_inventory_sha256": canonical_json_sha256(
                bundles
            ),
        }
        self.evidence_path.write_text(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return plan, evidence

    def test_finalization_requires_actual_conditions_and_matches_runtime_formula(self) -> None:
        plan, evidence = self._build_plan_and_evidence()
        finalization, scenarios = finalize_capture(
            **self.source_kwargs(),
            manifest_path=self.plan_path,
            evidence_path=self.evidence_path,
        )
        self.assertEqual(len(_validate_cases(scenarios)), 256)
        by_identity = {(case["seed"], case["env_index"]): case for case in evidence["cases"]}
        for scenario in scenarios:
            actual = by_identity[(scenario["seed"], scenario["env_index"])]
            self.assertEqual(
                scenario["scenario_id"],
                canonical_json_sha256(
                    {
                        "seed": scenario["seed"],
                        "env_index": scenario["env_index"],
                        "first_condition_sha256": actual["first_condition_sha256"],
                    }
                ),
            )
        self.assertEqual(
            finalization["capture_evidence_sha256"],
            hashlib.sha256(self.evidence_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(finalization["nested_split_manifest_sha256"], self.split_sha)
        self.assertEqual(finalization["status"], "capture_required")
        self.assertFalse(finalization["release_eligible"])
        self.assertNotIn("scenario_id", plan["capture"]["cases"][0])
        result = subprocess.run(
            [
                "python3",
                str(BUILDER),
                "finalize",
                "--motion-root",
                str(self.motions),
                "--split-manifest",
                str(self.split),
                "--expected-split-manifest-sha256",
                self.split_sha,
                "--manifest",
                str(self.plan_path),
                "--evidence",
                str(self.evidence_path),
                "--finalization-output",
                str(self.finalization_path),
                "--scenario-output",
                str(self.scenario_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.finalization_path.read_text()), finalization)
        self.assertEqual(json.loads(self.scenario_path.read_text()), scenarios)

    def test_finalizer_rejects_missing_duplicate_and_assignment_mismatch(self) -> None:
        _plan, evidence = self._build_plan_and_evidence()
        evidence["cases"].pop()
        self.evidence_path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(HeldoutPlanError, "exactly 256"):
            finalize_capture(
                **self.source_kwargs(),
                manifest_path=self.plan_path,
                evidence_path=self.evidence_path,
            )
        _plan, evidence = self._build_plan_and_evidence()
        evidence["cases"][-1]["seed"] = evidence["cases"][0]["seed"]
        evidence["cases"][-1]["env_index"] = evidence["cases"][0]["env_index"]
        self.evidence_path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(HeldoutPlanError, "duplicate seed/env identity"):
            finalize_capture(
                **self.source_kwargs(),
                manifest_path=self.plan_path,
                evidence_path=self.evidence_path,
            )
        _plan, evidence = self._build_plan_and_evidence()
        evidence["cases"][0]["motion_sha256"] = "0" * 64
        self.evidence_path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(HeldoutPlanError, "assignment mismatch"):
            finalize_capture(
                **self.source_kwargs(),
                manifest_path=self.plan_path,
                evidence_path=self.evidence_path,
            )

    def test_mutation_duplicate_and_output_drift_fail_closed(self) -> None:
        first = subprocess.run(self.build_cli(), text=True, capture_output=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        (self.motions / "motion_000.npz").write_bytes(b"mutated")
        second = subprocess.run(self.build_cli(), text=True, capture_output=True, check=False)
        self.assertEqual(second.returncode, 2)
        self.assertIn("content mismatch", second.stderr)
        # Restore, then prove a different existing output is never replaced.
        (self.motions / "motion_000.npz").write_bytes(
            b"fixture-motion-v2\0" + (0).to_bytes(4, "big")
        )
        self.plan_path.write_text("{}\n")
        third = subprocess.run(self.build_cli(), text=True, capture_output=True, check=False)
        self.assertEqual(third.returncode, 2)
        self.assertIn("refusing to replace different existing output", third.stderr)


if __name__ == "__main__":
    unittest.main()
