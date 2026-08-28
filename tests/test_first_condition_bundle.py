"""Focused safety and determinism tests for canonical Image3F conditions."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger.first_condition_bundle import (
    FirstConditionBundleError,
    VIEWS,
    image3f_payload_sha256,
    load_first_condition_bundle,
    publish_first_condition_bundle,
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class FirstConditionBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = 17
        self.plan_sha = sha("plan")
        self.prompt = "Pick up the object."
        self.requests = {}
        for env_index in range(2):
            depth = {
                view: np.full((240, 416), env_index + view_index, dtype=np.uint8)
                for view_index, view in enumerate(VIEWS)
            }
            mask = {
                view: np.full(
                    (240, 416, 3), env_index + view_index + 10, dtype=np.uint8
                )
                for view_index, view in enumerate(VIEWS)
            }
            previous = {
                view: np.full(
                    (240, 416, 3), env_index + view_index + 20, dtype=np.uint8
                )
                for view_index, view in enumerate(VIEWS)
            }
            self.requests[env_index] = (depth, mask, previous)
        self.cases = [
            {
                "seed": self.seed,
                "env_index": env_index,
                "global_case_index": env_index,
                "planned_case_id": sha(f"planned-{env_index}"),
                "motion_relative_path": f"motion_{env_index}.npz",
                "motion_sha256": sha(f"motion-{env_index}"),
            }
            for env_index in range(2)
        ]
        self.live = {
            str(env_index): image3f_payload_sha256(
                views=VIEWS,
                groups=tuple(
                    zip(("depth", "mask", "previous_rgb"), request, strict=True)
                ),
                prompt=self.prompt,
                seed=self.seed,
            )
            for env_index, request in self.requests.items()
        }
        self.reset = {str(index): sha(f"reset-{index}") for index in range(2)}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(self, name: str):
        return publish_first_condition_bundle(
            self.root / name,
            seed=self.seed,
            generation_seed=self.seed,
            prompt=self.prompt,
            views=VIEWS,
            requests_by_env=self.requests,
            cases=self.cases,
            plan_sha256=self.plan_sha,
            capture_run_id="capture-seed17-pass1",
            capture_nonce="1" * 32,
            provenance={"heldout_plan_sha256": self.plan_sha},
            live_condition_sha256_by_env=self.live,
            live_reset_state_sha256_by_env=self.reset,
        )

    def test_publish_load_is_deterministic_and_exact(self) -> None:
        first = self.publish("first.npz")
        second = self.publish("second.npz")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(
            (self.root / "first.npz").read_bytes(),
            (self.root / "second.npz").read_bytes(),
        )
        self.assertEqual(stat.S_IMODE((self.root / "first.npz").stat().st_mode), 0o444)
        self.assertEqual(first["first_condition_sha256_by_env"], self.live)
        self.assertEqual(first["live_reset_state_sha256_by_env"], self.reset)
        self.assertEqual(first["requests_by_env"][1][0][VIEWS[2]][0, 0], 3)
        loaded = load_first_condition_bundle(
            self.root / "first.npz",
            expected_sha256=first["sha256"],
            expected_seed=self.seed,
            expected_env_count=2,
            expected_views=VIEWS,
            expected_plan_sha256=self.plan_sha,
        )
        self.assertEqual(loaded["cases"], first["cases"])

    def test_rejects_hash_drift_and_symlink(self) -> None:
        bundle = self.publish("condition.npz")
        with self.assertRaisesRegex(FirstConditionBundleError, "byte identity"):
            load_first_condition_bundle(
                self.root / "condition.npz",
                expected_sha256=sha("wrong"),
                expected_seed=self.seed,
                expected_env_count=2,
                expected_views=VIEWS,
            )

    def test_rejects_writable_file_symlinked_ancestor_and_key_collision(self) -> None:
        bundle = self.publish("immutable.npz")
        os.chmod(self.root / "immutable.npz", 0o644)
        with self.assertRaisesRegex(FirstConditionBundleError, "immutable"):
            load_first_condition_bundle(
                self.root / "immutable.npz",
                expected_sha256=bundle["sha256"],
                expected_seed=self.seed,
                expected_env_count=2,
                expected_views=VIEWS,
            )

        real_parent = self.root / "real"
        real_parent.mkdir()
        linked_parent = self.root / "linked"
        os.symlink(real_parent, linked_parent)
        with self.assertRaisesRegex(FirstConditionBundleError, "symlink"):
            publish_first_condition_bundle(
                linked_parent / "condition.npz",
                seed=self.seed,
                generation_seed=self.seed,
                prompt=self.prompt,
                views=VIEWS,
                requests_by_env=self.requests,
                cases=self.cases,
                plan_sha256=self.plan_sha,
                capture_run_id="capture-seed17-pass1",
                capture_nonce="4" * 32,
                provenance={"heldout_plan_sha256": self.plan_sha},
                live_condition_sha256_by_env=self.live,
                live_reset_state_sha256_by_env=self.reset,
            )

        colliding = {0: self.live["0"], "0": self.live["0"], "1": self.live["1"]}
        with self.assertRaisesRegex(FirstConditionBundleError, "colliding"):
            publish_first_condition_bundle(
                self.root / "collision.npz",
                seed=self.seed,
                generation_seed=self.seed,
                prompt=self.prompt,
                views=VIEWS,
                requests_by_env=self.requests,
                cases=self.cases,
                plan_sha256=self.plan_sha,
                capture_run_id="capture-seed17-pass1",
                capture_nonce="5" * 32,
                provenance={"heldout_plan_sha256": self.plan_sha},
                live_condition_sha256_by_env=colliding,
                live_reset_state_sha256_by_env=self.reset,
            )
        link = self.root / "condition-link.npz"
        os.symlink(self.root / "condition.npz", link)
        with self.assertRaisesRegex(FirstConditionBundleError, "safely"):
            load_first_condition_bundle(
                link,
                expected_sha256=bundle["sha256"],
                expected_seed=self.seed,
                expected_env_count=2,
                expected_views=VIEWS,
            )

    def test_rejects_live_hash_or_shape_drift_before_publish(self) -> None:
        wrong_live = dict(self.live)
        wrong_live["0"] = sha("not-the-array")
        with self.assertRaisesRegex(FirstConditionBundleError, "live condition"):
            publish_first_condition_bundle(
                self.root / "bad-live.npz",
                seed=self.seed,
                generation_seed=self.seed,
                prompt=self.prompt,
                views=VIEWS,
                requests_by_env=self.requests,
                cases=self.cases,
                plan_sha256=self.plan_sha,
                capture_run_id="capture-seed17-pass1",
                capture_nonce="2" * 32,
                provenance={"heldout_plan_sha256": self.plan_sha},
                live_condition_sha256_by_env=wrong_live,
                live_reset_state_sha256_by_env=self.reset,
            )
        malformed = dict(self.requests)
        depth, mask, previous = malformed[0]
        malformed[0] = ({**depth, VIEWS[0]: depth[VIEWS[0]][:, :-1]}, mask, previous)
        with self.assertRaisesRegex(FirstConditionBundleError, "must be uint8"):
            publish_first_condition_bundle(
                self.root / "bad-shape.npz",
                seed=self.seed,
                generation_seed=self.seed,
                prompt=self.prompt,
                views=VIEWS,
                requests_by_env=malformed,
                cases=self.cases,
                plan_sha256=self.plan_sha,
                capture_run_id="capture-seed17-pass1",
                capture_nonce="3" * 32,
                provenance={"heldout_plan_sha256": self.plan_sha},
                live_condition_sha256_by_env=self.live,
                live_reset_state_sha256_by_env=self.reset,
            )


if __name__ == "__main__":
    unittest.main()
