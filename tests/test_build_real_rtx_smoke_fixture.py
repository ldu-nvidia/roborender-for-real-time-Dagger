import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts, replay
from scripts.round_dagger import build_real_rtx_smoke_fixture as fixture


SHA = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}

VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["ext1", "ext2", "wrist"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [2]}


def _arrays(env_ids, *, steps=6):
    env_ids = tuple(env_ids)
    batch = len(env_ids)
    time = np.arange(steps, dtype=np.float32)[:, None, None]
    env = np.asarray(env_ids, dtype=np.float32)[None, :, None]
    expert = np.concatenate((time + env, time - env), axis=2).astype(np.float32)
    student = expert + np.float32(0.5)
    expert_mask = np.broadcast_to(
        (np.arange(steps) % 2 == 0)[:, None], (steps, batch)
    ).copy()
    behavior = np.where(expert_mask[..., None], expert, student).astype(np.float32)
    rgb = np.empty((steps, batch, 6, 4, 3), dtype=np.uint8)
    for index, env_id in enumerate(env_ids):
        rgb[:, index] = np.uint8(env_id % 251)
    return {
        "policy_rgb": rgb,
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": behavior,
        "expert_mask": expert_mask.astype(np.bool_),
        "dones": np.zeros((steps, batch), dtype=np.bool_),
        "observation_is_raytraced": np.ones((steps, batch), dtype=np.bool_),
        "global_env_ids": np.broadcast_to(
            np.asarray(env_ids, dtype=np.int64)[None, :], (steps, batch)
        ).copy(),
        "episode_ids": np.broadcast_to(
            np.asarray(env_ids, dtype=np.int64)[None, :], (steps, batch)
        ).copy(),
        "episode_steps": np.broadcast_to(
            np.arange(steps, dtype=np.int64)[:, None], (steps, batch)
        ).copy(),
        "obs__actor_obs": np.broadcast_to(
            env + time, (steps, batch, 3)
        ).astype(np.float32).copy(),
    }


class RealRtxSmokeFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        artifacts.create_cohort(
            self.source,
            campaign_id="source-campaign",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            expected_ranks=[0, 1],
            expected_global_env_ids=[0, 1, 2, 3, 32, 33, 34, 35],
            expected_transitions=48,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **SHA,
        )
        for source_rank, env_ids in ((0, (0, 1, 2, 3)), (1, (32, 33, 34, 35))):
            artifacts.publish_rollout_shard(
                self.source,
                rank=source_rank,
                rollout_id=7,
                part_index=0,
                arrays=_arrays(env_ids),
                campaign_id="source-campaign",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                **SHA,
            )
        self.source_report = artifacts.seal_cohort(self.source)

    def build_source_heldout(
        self,
        *,
        collides_with_train=False,
        cases_per_cohort=1,
        reverse_array_order=False,
        one_dimensional_global_ids=False,
        campaign_id="source-campaign",
        digests=None,
    ):
        digests = SHA if digests is None else digests
        cohorts = []
        training_envs = (0, 1, 2, 3, 32, 33, 34, 35)
        for index in range(8):
            if collides_with_train:
                env_ids = (training_envs[index],)
            elif cases_per_cohort == 1:
                env_ids = (1_000 + index,)
            else:
                base = 1_000 + index * 100
                env_ids = tuple(range(base, base + cases_per_cohort))
            array_env_ids = (
                tuple(reversed(env_ids)) if reverse_array_order else env_ids
            )
            cohort = self.root / f"source-eval-{index}"
            artifacts.create_cohort(
                cohort,
                campaign_id=campaign_id,
                macro_round=0,
                cohort_index=index,
                purpose="eval",
                expected_ranks=[0],
                expected_global_env_ids=sorted(env_ids),
                expected_transitions=len(env_ids),
                view_contract=VIEW_CONTRACT,
                action_contract=ACTION_CONTRACT,
                **digests,
            )
            arrays = _arrays(array_env_ids, steps=1)
            if one_dimensional_global_ids:
                arrays["global_env_ids"] = np.asarray(
                    array_env_ids, dtype=np.int64
                )
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=arrays,
                campaign_id=campaign_id,
                macro_round=0,
                cohort_index=index,
                purpose="eval",
                **digests,
            )
            artifacts.seal_cohort(cohort)
            cohorts.append(cohort)
        snapshot_path = self.root / (
            "source-heldout-collision.json"
            if collides_with_train
            else "source-heldout.json"
        )
        snapshot = replay.build_heldout_snapshot(cohorts, output=snapshot_path)
        return snapshot_path, snapshot

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_eight_rank_real_slice_and_binds_origin(self):
        output = self.root / "fixture"
        result = fixture.build_fixture(
            source_cohort=self.source,
            output_root=output,
            campaign_id="real-smoke-test",
            source_ranks=(0, 1),
            environments_per_source_rank=4,
            steps=4,
        )

        self.assertEqual(result["schema"], fixture.FIXTURE_SCHEMA)
        self.assertTrue(result["source"]["full_recursive_verification"])
        self.assertEqual(
            result["source"]["cohort_sha256"],
            self.source_report["cohort_sha256"],
        )
        self.assertEqual(result["extraction"]["learner_world_size"], 8)
        self.assertEqual(
            result["extraction"]["global_env_ids"],
            [0, 1, 2, 3, 32, 33, 34, 35],
        )
        self.assertEqual(result["derived"]["transition_count"], 32)

        cohort = artifacts.verify_cohort(output / "cohort")
        self.assertEqual(cohort["ranks"], list(range(8)))
        self.assertEqual(cohort["transition_count"], 32)
        self.assertEqual(len(cohort["shards"]), 8)

        expected_envs = [0, 1, 2, 3, 32, 33, 34, 35]
        for learner_rank, env_id in enumerate(expected_envs):
            payload = (
                output
                / "cohort"
                / "shards"
                / f"rank-{learner_rank:04d}_rollout-00000000_part-0000.npz"
            )
            with np.load(payload, allow_pickle=False) as arrays:
                self.assertEqual(arrays["policy_rgb"].shape, (4, 1, 6, 4, 3))
                self.assertTrue(np.all(arrays["policy_rgb"] == env_id % 251))
                self.assertEqual(arrays["global_env_ids"].shape, (4, 1))
                self.assertTrue(np.all(arrays["global_env_ids"] == env_id))
                np.testing.assert_array_equal(
                    arrays["episode_steps"][:, 0], np.arange(4)
                )

        snapshot_record = result["derived"]["replay_snapshot"]
        snapshot = replay.load_snapshot(
            output / snapshot_record["path"],
            expected_file_sha256=snapshot_record["sha256"],
        )
        self.assertEqual(snapshot["transition_count"], 32)
        self.assertEqual(len(snapshot["resolved_shards"]), 8)

        manifest_path = output / "fixture_manifest.json"
        manifest = artifacts.load_strict_json(manifest_path)
        content_digest = manifest.pop("fixture_content_sha256")
        self.assertEqual(content_digest, artifacts.canonical_json_sha256(manifest))
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual((output / "replay.sqlite3").stat().st_mode & 0o777, 0o444)

    def test_tampered_source_fails_before_any_completion_marker(self):
        source_payload = next((self.source / "shards").glob("*.npz"))
        os.chmod(source_payload, 0o644)
        with source_payload.open("r+b") as handle:
            handle.seek(128)
            original = handle.read(1)
            handle.seek(128)
            handle.write(bytes([original[0] ^ 0xFF]))

        output = self.root / "tampered-output"
        with self.assertRaisesRegex(artifacts.ArtifactError, "SHA-256 mismatch"):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=output,
                campaign_id="real-smoke-tampered",
            )
        self.assertFalse((output / "fixture_manifest.json").exists())

    def test_refuses_to_reuse_an_existing_output_root(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(fixture.FixtureError, "must not already exist"):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=output,
                campaign_id="real-smoke-existing",
            )

    def test_derives_eight_rank_fixed_heldout_under_smoke_campaign(self):
        source_heldout, source_snapshot = self.build_source_heldout()
        output = self.root / "fixture-with-heldout"
        result = fixture.build_fixture(
            source_cohort=self.source,
            output_root=output,
            campaign_id="real-smoke-heldout",
            source_heldout_snapshot=source_heldout,
            expected_heldout_cases=8,
        )

        source_record = result["source"]["heldout"]
        self.assertEqual(
            source_record["snapshot_file_sha256"],
            source_snapshot["file_sha256"],
        )
        self.assertEqual(source_record["snapshot_sha256"], source_snapshot["snapshot_sha256"])
        self.assertTrue(source_record["full_recursive_verification"])
        self.assertEqual(source_record["source_shard_count"], 8)
        self.assertEqual(
            source_record["selection"],
            {
                "policy": "all_verified_cases",
                "source_transition_count": 8,
                "selected_transition_count": 8,
                "source_shard_count": 8,
                "selected_cases_per_source_shard": None,
            },
        )

        heldout = result["derived"]["heldout"]
        self.assertEqual(heldout["heldout_transition_count"], 8)
        disjoint = heldout["train_eval_payload_disjointness"]
        self.assertTrue(disjoint["enforced"])
        self.assertEqual(disjoint["intersection"], [])
        self.assertEqual(len(disjoint["training_payload_sha256s"]), 8)
        self.assertEqual(len(disjoint["heldout_payload_sha256s"]), 8)

        cohort = artifacts.verify_cohort(output / "heldout_cohort")
        self.assertEqual(cohort["campaign_id"], "real-smoke-heldout")
        self.assertEqual(cohort["purpose"], "eval")
        self.assertEqual(cohort["ranks"], list(range(8)))
        self.assertEqual(cohort["transition_count"], 8)
        snapshot_record = heldout["heldout_snapshot"]
        loaded = replay.load_heldout_snapshot(
            output / snapshot_record["path"],
            expected_file_sha256=snapshot_record["sha256"],
        )
        self.assertEqual(loaded["campaign_id"], "real-smoke-heldout")
        self.assertEqual(loaded["transition_count"], 8)
        self.assertEqual(len(loaded["resolved_shards"]), 8)

    def test_deterministically_downsamples_256_cases_to_balanced_32(self):
        source_heldout, source_snapshot = self.build_source_heldout(
            cases_per_cohort=32,
            reverse_array_order=True,
            one_dimensional_global_ids=True,
        )
        source_bytes = {
            path: path.read_bytes()
            for directory in self.root.glob("source-eval-*")
            for path in directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        output = self.root / "fixture-with-heldout-32"
        result = fixture.build_fixture(
            source_cohort=self.source,
            output_root=output,
            campaign_id="real-smoke-heldout-32",
            source_heldout_snapshot=source_heldout,
            expected_heldout_cases=32,
        )

        source_record = result["source"]["heldout"]
        self.assertEqual(source_snapshot["transition_count"], 256)
        self.assertEqual(source_record["transition_count"], 256)
        self.assertEqual(source_record["selected_transition_count"], 32)
        self.assertEqual(
            source_record["selection"],
            {
                "policy": "lowest_global_env_id_per_source_shard",
                "source_transition_count": 256,
                "selected_transition_count": 32,
                "source_shard_count": 8,
                "selected_cases_per_source_shard": 4,
            },
        )

        expected_selected = []
        for index, cohort_record in enumerate(source_record["cohorts"]):
            expected = list(range(1_000 + index * 100, 1_004 + index * 100))
            expected_selected.extend(expected)
            self.assertEqual(cohort_record["source_transition_count"], 32)
            self.assertEqual(cohort_record["selected_transition_count"], 4)
            self.assertEqual(cohort_record["selected_global_env_ids"], expected)
            self.assertEqual(
                cohort_record["selection_policy"],
                "lowest_global_env_id_per_source_shard",
            )

        heldout = result["derived"]["heldout"]
        self.assertEqual(heldout["heldout_transition_count"], 32)
        cohort = artifacts.verify_cohort(output / "heldout_cohort")
        self.assertEqual(cohort["campaign_id"], "real-smoke-heldout-32")
        self.assertEqual(cohort["ranks"], list(range(8)))
        self.assertEqual(cohort["transition_count"], 32)
        self.assertEqual(cohort["global_env_ids"], sorted(expected_selected))
        self.assertEqual(len(cohort["shards"]), 8)
        for rank, shard in enumerate(cohort["shards"]):
            expected = list(range(1_000 + rank * 100, 1_004 + rank * 100))
            self.assertEqual(shard["rank"], rank)
            self.assertEqual(shard["transition_count"], 4)
            self.assertEqual(shard["global_env_ids"], expected)
            with np.load(
                output / "heldout_cohort" / shard["payload_path"],
                allow_pickle=False,
            ) as arrays:
                for key, array in arrays.items():
                    if key == "global_env_ids" and array.ndim == 1:
                        self.assertEqual(array.shape, (4,))
                    else:
                        self.assertGreaterEqual(array.ndim, 2, key)
                        self.assertEqual(array.shape[1], 4, key)
                np.testing.assert_array_equal(
                    arrays["global_env_ids"], np.asarray(expected)
                )
                expected_pixels = (
                    np.asarray(expected)[None, :, None, None, None] % 251
                )
                self.assertTrue(
                    np.all(arrays["policy_rgb"] == expected_pixels)
                )

        loaded = replay.load_heldout_snapshot(
            output / heldout["heldout_snapshot"]["path"],
            expected_file_sha256=heldout["heldout_snapshot"]["sha256"],
        )
        self.assertEqual(loaded["transition_count"], 32)
        self.assertEqual(len(loaded["resolved_shards"]), 8)
        self.assertEqual(
            {
                path: path.read_bytes()
                for directory in self.root.glob("source-eval-*")
                for path in directory.rglob("*")
                if path.is_file() and not path.is_symlink()
            },
            source_bytes,
        )

    def test_explicitly_pins_reused_heldout_source_campaign(self):
        eval_digests = {
            **SHA,
            "beta_sha256": "7" * 64,
            "config_sha256": "8" * 64,
        }
        source_heldout, _ = self.build_source_heldout(
            cases_per_cohort=32,
            campaign_id="fixed-eval-source",
            digests=eval_digests,
        )

        unbound = self.root / "fixture-cross-campaign-unbound-schedule"
        with self.assertRaisesRegex(
            fixture.FixtureError,
            "explicit eval schedule rebinding is required",
        ):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=unbound,
                campaign_id="mini-campaign-unbound",
                source_heldout_snapshot=source_heldout,
                expected_source_heldout_campaign_id="fixed-eval-source",
                expected_heldout_cases=32,
            )
        self.assertFalse((unbound / "fixture_manifest.json").exists())

        output = self.root / "fixture-cross-campaign-heldout"
        result = fixture.build_fixture(
            source_cohort=self.source,
            output_root=output,
            campaign_id="mini-campaign",
            source_heldout_snapshot=source_heldout,
            expected_source_heldout_campaign_id="fixed-eval-source",
            allow_source_eval_schedule_rebinding=True,
            expected_heldout_cases=32,
        )
        self.assertEqual(
            result["source"]["heldout"]["campaign_id"],
            "fixed-eval-source",
        )
        self.assertEqual(
            result["derived"]["heldout"]["heldout_transition_count"], 32
        )
        rebinding = result["source"]["heldout"][
            "source_eval_schedule_rebinding"
        ]
        self.assertTrue(rebinding["authorized"])
        self.assertEqual(rebinding["source"], {
            "beta_sha256": "7" * 64,
            "config_sha256": "8" * 64,
        })
        self.assertEqual(rebinding["destination"], {
            "beta_sha256": SHA["beta_sha256"],
            "config_sha256": SHA["config_sha256"],
        })

        rejected = self.root / "fixture-cross-campaign-rejected"
        with self.assertRaisesRegex(
            fixture.FixtureError,
            "source held-out snapshot belongs to another campaign",
        ):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=rejected,
                campaign_id="mini-campaign-rejected",
                source_heldout_snapshot=source_heldout,
                expected_source_heldout_campaign_id="wrong-source",
                expected_heldout_cases=32,
            )
        self.assertFalse((rejected / "fixture_manifest.json").exists())

    def test_rejects_unsupported_heldout_downsample_shape(self):
        source_heldout, _ = self.build_source_heldout(cases_per_cohort=32)
        output = self.root / "fixture-with-heldout-64"
        with self.assertRaisesRegex(
            fixture.FixtureError, "only deterministic 32-from-256"
        ):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=output,
                campaign_id="real-smoke-heldout-64",
                source_heldout_snapshot=source_heldout,
                expected_heldout_cases=64,
            )
        self.assertFalse((output / "fixture_manifest.json").exists())

    def test_rejects_train_eval_payload_overlap(self):
        source_heldout, _ = self.build_source_heldout(collides_with_train=True)
        output = self.root / "fixture-collision"
        with self.assertRaisesRegex(fixture.FixtureError, "payload digests overlap"):
            fixture.build_fixture(
                source_cohort=self.source,
                output_root=output,
                campaign_id="real-smoke-collision",
                steps=1,
                source_heldout_snapshot=source_heldout,
                expected_heldout_cases=8,
            )
        self.assertFalse((output / "fixture_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
