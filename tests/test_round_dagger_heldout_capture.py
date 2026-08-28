from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger.heldout_capture import (
    HeldoutCaptureError,
    bind_capture_lineage,
    build_capture_aggregate,
    capture_arrays,
    publish_raw_capture,
    verify_capture_aggregate,
    verify_capture_lineage,
)
from round_dagger.scenario_capture import EXPECTED_SEEDS


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


class HeldoutExpertCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.observations = {
            "vision_obs": np.full((32, 6, 4, 3), 0.5, dtype=np.float32),
            "actor_obs": np.arange(32 * 7, dtype=np.float32).reshape(32, 7),
        }
        self.actions = {
            "gt_actions": np.arange(32 * 26, dtype=np.float32).reshape(32, 26),
            "action_mean": np.zeros((32, 26), dtype=np.float32),
        }
        self.teacher_obs = np.arange(32 * 9, dtype=np.float32).reshape(32, 9)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_one_step_replay_arrays_and_create_only_publication(self) -> None:
        arrays = capture_arrays(
            self.observations,
            self.actions,
            global_env_ids=range(32),
            image_mean=None,
            image_std=None,
            teacher_observations=self.teacher_obs,
        )
        self.assertEqual(arrays["policy_rgb"].shape, (1, 32, 6, 4, 3))
        self.assertEqual(arrays["expert_actions"].shape, (1, 32, 26))
        self.assertEqual(arrays["obs__actor_obs"].shape, (1, 32, 7))
        self.assertEqual(arrays["teacher_obs"].shape, (1, 32, 9))
        self.assertTrue(np.array_equal(arrays["behavior_actions"], arrays["student_actions"]))
        output = self.root / "capture.npz"
        report = publish_raw_capture(
            output,
            self.observations,
            self.actions,
            global_env_ids=range(32),
            image_mean=None,
            image_std=None,
            teacher_observations=self.teacher_obs,
        )
        self.assertEqual(report["path"], str(output))
        with self.assertRaisesRegex(HeldoutCaptureError, "replace"):
            publish_raw_capture(
                output,
                self.observations,
                self.actions,
                global_env_ids=range(32),
                image_mean=None,
                image_std=None,
                teacher_observations=self.teacher_obs,
            )

    def test_rejects_incomplete_teacher_labels(self) -> None:
        bad = dict(self.actions)
        bad["gt_actions"] = np.zeros((32, 25), dtype=np.float32)
        with self.assertRaisesRegex(HeldoutCaptureError, "shape"):
            capture_arrays(
                self.observations,
                bad,
                global_env_ids=range(32),
                image_mean=None,
                image_std=None,
                teacher_observations=self.teacher_obs,
            )

    def test_pass2_binding_rejects_reordered_cases(self) -> None:
        capture = self.root / "capture.npz"
        publish_raw_capture(
            capture,
            self.observations,
            self.actions,
            global_env_ids=range(32),
            image_mean=None,
            image_std=None,
            teacher_observations=self.teacher_obs,
        )
        plan = self.root / "plan.json"
        passed = self.root / "pass.json"
        plan.write_text("{}")
        passed.write_text("{}")
        cases = [
            {
                "seed": 17,
                "env_index": index,
                "global_case_index": index,
                "planned_case_id": _sha(1000 + index),
                "motion_relative_path": f"motion-{index}.npz",
                "motion_sha256": _sha(index),
                "first_condition_sha256": _sha(2000 + index),
                "live_condition_sha256": _sha(3000 + index),
                "live_reset_state_sha256": _sha(4000 + index),
                "first_generated_sha256": _sha(5000 + index),
                "scenario_id": _sha(6000 + index),
            }
            for index in range(32)
        ]
        runtime = {
            "seed": 17,
            "pass_index": 2,
            "release_eligible": True,
            "cases": cases,
            "first_condition_bundle": {"sha256": _sha(7000)},
            "provenance": {"parent_checkpoint_sha256": _sha(8000)},
        }
        with mock.patch(
            "round_dagger.heldout_capture.load_capture_json",
            side_effect=[({}, b"{}"), ({}, b"{}")],
        ), mock.patch(
            "round_dagger.heldout_capture.validate_pass_document",
            return_value={"runtime": runtime},
        ):
            lineage = bind_capture_lineage(
                plan_path=plan,
                pass_artifact_path=passed,
                capture_path=capture,
                output_path=self.root / "lineage.json",
            )
        self.assertEqual(lineage["case_count"], 32)
        self.assertEqual(lineage["cases"][0]["planned_case_id"], cases[0]["planned_case_id"])

        # The same bytes cannot be semantically rebound to a different global
        # ordering, even if the rest of the pass document looks plausible.
        reordered = {**runtime, "cases": list(reversed(cases))}
        with mock.patch(
            "round_dagger.heldout_capture.load_capture_json",
            side_effect=[({}, b"{}"), ({}, b"{}")],
        ), mock.patch(
            "round_dagger.heldout_capture.validate_pass_document",
            return_value={"runtime": reordered},
        ), self.assertRaisesRegex(HeldoutCaptureError, "global IDs"):
            bind_capture_lineage(
                plan_path=plan,
                pass_artifact_path=passed,
                capture_path=capture,
                output_path=self.root / "reordered.json",
            )

    def test_aggregate_requires_exact_8x32_coverage(self) -> None:
        fake = []
        paths = []
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            path = self.root / f"lineage-{seed}.json"
            path.write_text("{}")
            paths.append(path)
            fake.append(
                {
                    "seed": seed,
                    "plan": {"sha256": _sha(9000)},
                    "provenance": {"identity": _sha(9001)},
                    "cases": [
                        {"global_case_index": seed_index * 32 + index}
                        for index in range(32)
                    ],
                    "path": str(path),
                    "file_sha256": _sha(9100 + seed_index),
                    "lineage_sha256": _sha(9200 + seed_index),
                    "capture": {"sha256": _sha(9300 + seed_index)},
                }
            )
        with mock.patch(
            "round_dagger.heldout_capture.verify_capture_lineage",
            side_effect=fake,
        ):
            aggregate = build_capture_aggregate(
                paths, output_path=self.root / "aggregate.json"
            )
        self.assertEqual(aggregate["case_count"], 256)
        self.assertEqual(len(aggregate["lineages"]), 8)
        with mock.patch(
            "round_dagger.heldout_capture.verify_capture_lineage",
            side_effect=fake,
        ):
            verified = verify_capture_aggregate(self.root / "aggregate.json")
        self.assertEqual(verified["aggregate_sha256"], aggregate["aggregate_sha256"])

        fake[-1] = {**fake[-1], "cases": fake[-1]["cases"][:-1]}
        with mock.patch(
            "round_dagger.heldout_capture.verify_capture_lineage",
            side_effect=fake,
        ), self.assertRaisesRegex(HeldoutCaptureError, "0..255"):
            build_capture_aggregate(paths, output_path=self.root / "bad.json")


if __name__ == "__main__":
    unittest.main()
