import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts, replay
from scripts.round_dagger import rematerialize_portable_smoke_handoff as handoff


SHA = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}
VIEW = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["ext1", "ext2", "wrist"],
}
ACTION = {"dtype": "float32", "shape": [2]}
CAMPAIGN = "portable-handoff-test"


def _arrays(env_id, *, steps, offset=0):
    expert = np.empty((steps, 1, 2), dtype=np.float32)
    expert[:, 0, 0] = env_id + np.arange(steps)
    expert[:, 0, 1] = offset - np.arange(steps)
    student = expert + np.float32(0.25)
    expert_mask = (np.arange(steps) % 2 == 0)[:, None]
    return {
        "policy_rgb": np.full(
            (steps, 1, 6, 4, 3), (env_id + offset) % 251, dtype=np.uint8
        ),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": np.where(
            expert_mask[..., None], expert, student
        ).astype(np.float32),
        "expert_mask": expert_mask.astype(np.bool_),
        "dones": np.zeros((steps, 1), dtype=np.bool_),
        "observation_is_raytraced": np.ones((steps, 1), dtype=np.bool_),
        "global_env_ids": np.full((steps, 1), env_id, dtype=np.int64),
        "episode_ids": np.full((steps, 1), env_id, dtype=np.int64),
        "episode_steps": np.arange(steps, dtype=np.int64)[:, None],
        "obs__actor_obs": np.full((steps, 1, 3), env_id + offset, dtype=np.float32),
    }


def _publish_cohort(path, *, purpose, cohort_index, steps, collide=False):
    if purpose == "train" or collide:
        env_ids = list(range(8))
        offset = 0
    else:
        env_ids = list(range(100, 108))
        offset = 17
    artifacts.create_cohort(
        path,
        campaign_id=CAMPAIGN,
        macro_round=0,
        cohort_index=cohort_index,
        purpose=purpose,
        expected_ranks=list(range(8)),
        expected_global_env_ids=env_ids,
        expected_transitions=8 * steps,
        view_contract=VIEW,
        action_contract=ACTION,
        **SHA,
    )
    for rank, env_id in enumerate(env_ids):
        artifacts.publish_rollout_shard(
            path,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(env_id, steps=steps, offset=offset),
            campaign_id=CAMPAIGN,
            macro_round=0,
            cohort_index=cohort_index,
            purpose=purpose,
            **SHA,
        )
    return artifacts.seal_cohort(path)


class PortableHandoffTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, *, collide=False):
        source = self.root / ("source-collision" if collide else "source")
        source.mkdir()
        train_steps = 1 if collide else 2
        _publish_cohort(
            source / "cohort",
            purpose="train",
            cohort_index=0,
            steps=train_steps,
        )
        _publish_cohort(
            source / "heldout_cohort",
            purpose="eval",
            cohort_index=1,
            steps=1,
            collide=collide,
        )
        with replay.ReplayLedger(source / "replay.sqlite3") as ledger:
            ledger.ingest_cohort(source / "cohort")
            train_snapshot = ledger.build_snapshot(
                CAMPAIGN,
                through_macro_round=0,
                expected_train_cohorts=1,
                output=source / "training_snapshot.json",
            )
        heldout_snapshot = replay.build_heldout_snapshot(
            [source / "heldout_cohort"], output=source / "heldout_snapshot.json"
        )

        destination = self.root / ("destination-collision" if collide else "destination")
        destination.mkdir()
        shutil.copytree(source / "cohort", destination / "cohort")
        shutil.copytree(source / "heldout_cohort", destination / "heldout_cohort")
        shutil.copy2(
            source / "training_snapshot.json",
            destination / "source_training_snapshot.json",
        )
        shutil.copy2(
            source / "heldout_snapshot.json",
            destination / "source_heldout_snapshot.json",
        )
        return destination, train_snapshot, heldout_snapshot

    def invoke(self, destination, train_snapshot, heldout_snapshot):
        return handoff.rematerialize(
            destination_root=destination,
            train_cohort=destination / "cohort",
            heldout_cohort=destination / "heldout_cohort",
            source_training_snapshot=destination / "source_training_snapshot.json",
            source_heldout_snapshot=destination / "source_heldout_snapshot.json",
            expected_source_training_file_sha256=train_snapshot["file_sha256"],
            expected_source_training_snapshot_sha256=train_snapshot["snapshot_sha256"],
            expected_source_heldout_file_sha256=heldout_snapshot["file_sha256"],
            expected_source_heldout_snapshot_sha256=heldout_snapshot["snapshot_sha256"],
            expected_world_size=8,
            expected_heldout_cases=8,
        )

    def test_rebases_snapshots_without_touching_cohort_bytes(self):
        destination, source_train, source_heldout = self.prepare()
        before = {
            path.relative_to(destination).as_posix(): (
                os.stat(path).st_ino,
                os.stat(path).st_mtime_ns,
                artifacts.sha256_file(path),
            )
            for root in (destination / "cohort", destination / "heldout_cohort")
            for path in root.rglob("*")
            if path.is_file()
        }
        result = self.invoke(destination, source_train, source_heldout)
        after = {
            path.relative_to(destination).as_posix(): (
                os.stat(path).st_ino,
                os.stat(path).st_mtime_ns,
                artifacts.sha256_file(path),
            )
            for root in (destination / "cohort", destination / "heldout_cohort")
            for path in root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertTrue(result["cohort_immutability"]["verified"])
        self.assertEqual(
            result["equivalence"]["train_eval_payload_intersection"], []
        )

        train_identity = result["destination"]["training_snapshot"]
        train = replay.load_snapshot(
            destination / "training_snapshot.json",
            expected_file_sha256=train_identity["file_sha256"],
        )
        self.assertEqual(
            Path(train["cohorts"][0]["cohort_manifest_path"]),
            (destination / "cohort" / "seal.json").resolve(),
        )
        heldout_identity = result["destination"]["heldout_snapshot"]
        heldout = replay.load_heldout_snapshot(
            destination / "heldout_snapshot.json",
            expected_file_sha256=heldout_identity["file_sha256"],
        )
        self.assertEqual(
            Path(heldout["cohorts"][0]["cohort_manifest_path"]),
            (destination / "heldout_cohort" / "seal.json").resolve(),
        )
        completion = destination / "portable_handoff_manifest.json"
        self.assertEqual(completion.stat().st_mode & 0o777, 0o444)
        document = artifacts.load_strict_json(completion)
        digest = document.pop("handoff_content_sha256")
        self.assertEqual(digest, artifacts.canonical_json_sha256(document))

    def test_rejects_unpinned_source_snapshot_bytes(self):
        destination, source_train, source_heldout = self.prepare()
        source_copy = destination / "source_training_snapshot.json"
        os.chmod(source_copy, 0o644)
        with source_copy.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(handoff.HandoffError, "file SHA-256 differs"):
            self.invoke(destination, source_train, source_heldout)
        self.assertFalse((destination / "training_snapshot.json").exists())
        self.assertFalse((destination / "portable_handoff_manifest.json").exists())

    def test_rejects_train_eval_payload_overlap_before_publication(self):
        destination, source_train, source_heldout = self.prepare(collide=True)
        with self.assertRaisesRegex(handoff.HandoffError, "payload identities overlap"):
            self.invoke(destination, source_train, source_heldout)
        self.assertFalse((destination / "training_snapshot.json").exists())
        self.assertFalse((destination / "heldout_snapshot.json").exists())
        self.assertFalse((destination / "portable_handoff_manifest.json").exists())

    def test_refuses_to_overwrite_a_destination_snapshot(self):
        destination, source_train, source_heldout = self.prepare()
        (destination / "training_snapshot.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(handoff.HandoffError, "already exists"):
            self.invoke(destination, source_train, source_heldout)


if __name__ == "__main__":
    unittest.main()
