import fcntl
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts


SHA = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}

IDENTITY = {
    "campaign_id": "campaign-a",
    "macro_round": 2,
    "cohort_index": 1,
    "purpose": "train",
}

VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}

ACTION_CONTRACT = {"dtype": "float32", "shape": [2]}


def rollout_arrays(env_ids=(0,), *, steps=3, episode_offset=0):
    env_ids = tuple(env_ids)
    batch = len(env_ids)
    rgb = np.zeros((steps, batch, 6, 4, 3), dtype=np.uint8)
    for index, env_id in enumerate(env_ids):
        rgb[:, index] = env_id + 1
    expert = np.arange(steps * batch * 2, dtype=np.float32).reshape(steps, batch, 2)
    student = expert + np.float32(0.25)
    expert_mask = np.indices((steps, batch))[0] % 2 == 0
    behavior = np.where(expert_mask[..., None], expert, student).astype(np.float32)
    episodes = np.empty((steps, batch), dtype=np.int64)
    for index, env_id in enumerate(env_ids):
        episodes[:, index] = episode_offset + env_id
    return {
        "policy_rgb": rgb,
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": behavior,
        "expert_mask": expert_mask.astype(np.bool_),
        "dones": np.zeros((steps, batch), dtype=np.bool_),
        "observation_is_raytraced": np.zeros((steps, batch), dtype=np.bool_),
        "global_env_ids": np.asarray(env_ids, dtype=np.int64),
        "episode_ids": episodes,
        "episode_steps": np.broadcast_to(
            np.arange(steps, dtype=np.int64)[:, None], (steps, batch)
        ).copy(),
        "obs__actor_obs": np.ones((steps, batch, 5), dtype=np.float32),
        "obs__integer_feature": np.zeros((steps, batch, 1), dtype=np.int16),
    }


def add_reference_provenance(arrays, *, lag=2):
    steps, envs = arrays["episode_steps"].shape
    generated = np.empty((steps, envs, 32), dtype=np.uint8)
    reference = np.empty_like(generated)
    kinds = np.empty((steps, envs), dtype=np.uint8)
    source_steps = np.empty((steps, envs), dtype=np.int64)
    for time_index in range(steps):
        generated[time_index] = time_index + 1
        episode_step = int(arrays["episode_steps"][time_index, 0])
        if episode_step == 0:
            kinds[time_index] = 1
            source_steps[time_index] = -1
            reference[time_index] = 99
        elif episode_step < lag:
            kinds[time_index] = 2
            source_steps[time_index] = 0
            reference[time_index] = generated[0]
        else:
            kinds[time_index] = 3
            source_steps[time_index] = episode_step - lag
            reference[time_index] = generated[episode_step - lag]
    arrays.update(
        reference_kinds=kinds,
        reference_lags=np.full((steps, envs), lag, dtype=np.int64),
        reference_source_steps=source_steps,
        reference_sha256=reference,
        generated_sha256=generated,
        conditioning_depth_nonzero_fraction=np.full(
            (steps, envs, 3), 0.75, dtype=np.float32
        ),
        conditioning_mask_nonzero_fraction=np.full(
            (steps, envs, 3), 0.1, dtype=np.float32
        ),
    )
    return arrays


class ArtifactTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cohort = self.root / "cohort"

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, *, ranks=(0, 1), envs=(0, 1, 2, 3), transitions=12):
        return artifacts.create_cohort(
            self.cohort,
            expected_ranks=ranks,
            expected_global_env_ids=envs,
            expected_transitions=transitions,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **IDENTITY,
            **SHA,
        )

    def publish(self, rank, part, env_ids, *, arrays=None, rollout_id=7):
        return artifacts.publish_rollout_shard(
            self.cohort,
            rank=rank,
            rollout_id=rollout_id,
            part_index=part,
            arrays=arrays if arrays is not None else rollout_arrays(env_ids),
            **IDENTITY,
            **SHA,
        )


class RolloutShardPublicationTest(ArtifactTestCase):
    def test_published_cohort_is_readable_across_container_user_boundaries(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        report = self.publish(0, 0, (0,))
        artifacts.seal_cohort(self.cohort)

        published = (
            self.cohort / artifacts.CONTRACT_FILENAME,
            self.cohort / report["payload_path"],
            self.cohort / report["manifest_path"],
            self.cohort / artifacts.SEAL_FILENAME,
        )
        for path in published:
            self.assertEqual(path.stat().st_mode & 0o777, 0o444, path)

    def test_multiple_parts_per_rank_seal_with_exact_coverage(self):
        contract = self.create()
        reports = [
            self.publish(0, 0, (0,)),
            self.publish(0, 1, (1,)),
            self.publish(1, 0, (2,)),
            self.publish(1, 1, (3,)),
        ]
        self.assertEqual(contract["expected_steps_per_env"], 3)
        self.assertEqual(sum(item["transition_count"] for item in reports), 12)

        sealed = artifacts.seal_cohort(self.cohort)
        verified = artifacts.verify_cohort(self.cohort / artifacts.SEAL_FILENAME)
        self.assertEqual(sealed, verified)
        self.assertEqual(verified["schema"], artifacts.COHORT_SCHEMA)
        self.assertTrue(verified["sealed"])
        self.assertEqual(verified["ranks"], [0, 1])
        self.assertEqual(verified["global_env_ids"], [0, 1, 2, 3])
        self.assertEqual(verified["transition_count"], 12)
        self.assertEqual(verified["runtime_policy_sha256"], SHA["runtime_policy_sha256"])
        self.assertEqual(len(verified["shards"]), 4)
        self.assertEqual(
            verified["cohort_sha256"], artifacts.sha256_file(verified["manifest_path"])
        )
        self.assertEqual(
            verified["cohort_sha256"],
            artifacts.canonical_json_sha256(
                artifacts.load_strict_json(verified["manifest_path"])
            ),
        )
        for shard in verified["shards"]:
            self.assertTrue(shard["manifest_path"].startswith("shards/"))
            self.assertTrue(shard["payload_path"].startswith("shards/"))

    def test_identical_retry_is_idempotent_and_changed_retry_is_rejected(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        first = self.publish(0, 0, (0,))
        payload = self.cohort / first["payload_path"]
        sidecar = self.cohort / first["manifest_path"]
        first_payload_stat = payload.stat()
        first_sidecar_stat = sidecar.stat()

        second = self.publish(0, 0, (0,))
        self.assertEqual(first, second)
        self.assertEqual(payload.stat().st_ino, first_payload_stat.st_ino)
        self.assertEqual(sidecar.stat().st_ino, first_sidecar_stat.st_ino)

        changed = rollout_arrays((0,))
        changed["student_actions"][0, 0, 0] += 1.0
        if not changed["expert_mask"][0, 0]:
            changed["behavior_actions"][0, 0, 0] += 1.0
        with self.assertRaisesRegex(artifacts.ArtifactError, "immutable artifact conflict"):
            self.publish(0, 0, (0,), arrays=changed)

    def test_payload_first_failure_is_undiscoverable_and_retry_completes(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        original = artifacts.publish_immutable_json
        with mock.patch.object(
            artifacts,
            "publish_immutable_json",
            side_effect=artifacts.ArtifactError("injected sidecar failure"),
        ):
            with self.assertRaisesRegex(artifacts.ArtifactError, "injected"):
                self.publish(0, 0, (0,))
        self.assertEqual(artifacts.discover_rollout_shards(self.cohort), [])
        payloads = list((self.cohort / artifacts.SHARD_DIRECTORY).glob("*.npz"))
        self.assertEqual(len(payloads), 1)

        with mock.patch.object(artifacts, "publish_immutable_json", wraps=original):
            report = self.publish(0, 0, (0,))
        self.assertEqual(len(artifacts.discover_rollout_shards(self.cohort)), 1)
        self.assertEqual(report["payload_sha256"], artifacts.sha256_file(payloads[0]))

    def test_new_shard_after_seal_is_rejected_but_identical_retry_is_allowed(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        first = self.publish(0, 0, (0,))
        artifacts.seal_cohort(self.cohort)
        self.assertEqual(first, self.publish(0, 0, (0,)))
        with self.assertRaisesRegex(artifacts.ArtifactError, "after the cohort is sealed"):
            self.publish(0, 1, (0,), rollout_id=8)


class RolloutArrayValidationTest(ArtifactTestCase):
    def setUp(self):
        super().setUp()
        self.create(ranks=(0,), envs=(0,), transitions=3)

    def assert_bad(self, mutate, pattern):
        arrays = rollout_arrays((0,))
        mutate(arrays)
        with self.assertRaisesRegex(artifacts.ArtifactError, pattern):
            self.publish(0, 0, (0,), arrays=arrays)

    def test_rgb_must_be_uint8_five_dimensional_three_view_contract(self):
        self.assert_bad(
            lambda arrays: arrays.__setitem__(
                "policy_rgb", arrays["policy_rgb"].astype(np.float32)
            ),
            "policy_rgb",
        )
        self.assert_bad(
            lambda arrays: arrays.__setitem__("policy_rgb", arrays["policy_rgb"][:, :, :3]),
            "policy_rgb",
        )

    def test_actions_are_equal_contract_shape_float_and_finite(self):
        self.assert_bad(
            lambda arrays: arrays["expert_actions"].__setitem__((0, 0, 0), np.nan),
            "non-finite",
        )
        self.assert_bad(
            lambda arrays: arrays.__setitem__(
                "student_actions", arrays["student_actions"][..., :1]
            ),
            "student_actions",
        )
        self.assert_bad(
            lambda arrays: arrays.__setitem__(
                "behavior_actions", arrays["behavior_actions"].astype(np.float64)
            ),
            "behavior_actions",
        )

    def test_masks_ids_and_observations_are_strict(self):
        self.assert_bad(
            lambda arrays: arrays.__setitem__(
                "observation_is_raytraced",
                arrays["observation_is_raytraced"].astype(np.uint8),
            ),
            "observation_is_raytraced",
        )
        self.assert_bad(
            lambda arrays: arrays.__setitem__(
                "global_env_ids", arrays["global_env_ids"].astype(np.float32)
            ),
            "global_env_ids",
        )
        self.assert_bad(
            lambda arrays: arrays["obs__actor_obs"].__setitem__((0, 0, 0), np.inf),
            "obs__actor_obs contains non-finite",
        )
        self.assert_bad(
            lambda arrays: arrays.__setitem__("unexpected", np.zeros((3, 1))),
            "unknown=.*unexpected",
        )

    def test_behavior_source_and_episode_raytraced_mask_are_consistent(self):
        self.assert_bad(
            lambda arrays: arrays["behavior_actions"].__setitem__((0, 0, 0), 99.0),
            "according to expert_mask",
        )

        def change_raytraced_mid_episode(arrays):
            arrays["observation_is_raytraced"][1, 0] = True

        self.assert_bad(change_raytraced_mid_episode, "constant within each episode")

    def test_episode_steps_must_be_contiguous_and_reset_only_after_done(self):
        self.assert_bad(
            lambda arrays: arrays["episode_steps"].__setitem__((1, 0), 2),
            "episode steps must be contiguous",
        )

        def reset_without_done(arrays):
            arrays["episode_ids"][2, 0] += 1
            arrays["episode_steps"][2, 0] = 0

        self.assert_bad(reset_without_done, "episode steps must be contiguous")

    def test_reference_provenance_bundle_is_complete_and_digest_chained(self):
        valid = add_reference_provenance(rollout_arrays((0,)))
        self.publish(0, 0, (0,), arrays=valid)

        other = self.root / "other-reference"
        artifacts.create_cohort(
            other,
            expected_ranks=(0,),
            expected_global_env_ids=(0,),
            expected_transitions=3,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **IDENTITY,
            **SHA,
        )
        broken = add_reference_provenance(rollout_arrays((0,)))
        broken["reference_sha256"][2, 0] = 7
        with self.assertRaisesRegex(artifacts.ArtifactError, "digest does not match"):
            artifacts.publish_rollout_shard(
                other,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=broken,
                **IDENTITY,
                **SHA,
            )

        incomplete = rollout_arrays((0,))
        incomplete["reference_kinds"] = np.ones((3, 1), dtype=np.uint8)
        with self.assertRaisesRegex(artifacts.ArtifactError, "bundle is incomplete"):
            artifacts.publish_rollout_shard(
                other,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=incomplete,
                **IDENTITY,
                **SHA,
            )

        bad_coverage = add_reference_provenance(rollout_arrays((0,)))
        bad_coverage["conditioning_depth_nonzero_fraction"][1, 0, 0] = np.nan
        with self.assertRaisesRegex(artifacts.ArtifactError, "finite fractions"):
            artifacts.publish_rollout_shard(
                other,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=bad_coverage,
                **IDENTITY,
                **SHA,
            )

    def test_global_ids_may_be_constant_tb_but_not_change_with_time(self):
        arrays = rollout_arrays((0,))
        arrays["global_env_ids"] = np.zeros((3, 1), dtype=np.int32)
        self.publish(0, 0, (0,), arrays=arrays)

        other = self.root / "other"
        artifacts.create_cohort(
            other,
            expected_ranks=(0,),
            expected_global_env_ids=(0,),
            expected_transitions=3,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **IDENTITY,
            **SHA,
        )
        arrays = rollout_arrays((0,))
        arrays["global_env_ids"] = np.asarray([[0], [1], [0]], dtype=np.int32)
        with self.assertRaisesRegex(artifacts.ArtifactError, "remain constant"):
            artifacts.publish_rollout_shard(
                other,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=arrays,
                **IDENTITY,
                **SHA,
            )


class CohortIntegrityTest(ArtifactTestCase):
    def test_seal_waits_for_complete_payload_sidecar_publication(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        sidecar_entered = threading.Event()
        allow_sidecar = threading.Event()
        sealer_started = threading.Event()
        publisher_errors = []
        sealer_errors = []
        actual_publish = artifacts.publish_immutable_json
        target = "rank-0000_rollout-00000007_part-0000.json"

        def block_shard_sidecar(path, document):
            if Path(path).name == target:
                sidecar_entered.set()
                if not allow_sidecar.wait(timeout=5):
                    raise RuntimeError("test timed out waiting to publish sidecar")
            return actual_publish(path, document)

        def publish():
            try:
                self.publish(0, 0, (0,))
            except BaseException as exc:  # surfaced in the asserting thread
                publisher_errors.append(exc)

        sealed = []

        def seal():
            try:
                sealer_started.set()
                sealed.append(artifacts.seal_cohort(self.cohort))
            except BaseException as exc:  # surfaced in the asserting thread
                sealer_errors.append(exc)

        with mock.patch.object(
            artifacts, "publish_immutable_json", side_effect=block_shard_sidecar
        ):
            publisher = threading.Thread(target=publish)
            publisher.start()
            self.assertTrue(sidecar_entered.wait(timeout=5))

            sealer = threading.Thread(target=seal)
            sealer.start()
            self.assertTrue(sealer_started.wait(timeout=5))
            self.assertTrue(sealer.is_alive())
            self.assertFalse((self.cohort / artifacts.SEAL_FILENAME).exists())

            allow_sidecar.set()
            publisher.join(timeout=5)
            sealer.join(timeout=5)

        self.assertFalse(publisher.is_alive())
        self.assertFalse(sealer.is_alive())
        self.assertEqual(publisher_errors, [])
        self.assertEqual(sealer_errors, [])
        self.assertEqual(len(sealed), 1)
        self.assertEqual(sealed[0], artifacts.verify_cohort(self.cohort))

    def test_publisher_waiting_across_seal_rechecks_and_rejects_new_shard(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        self.publish(0, 0, (0,))
        seal_document = artifacts.seal_cohort(self.cohort)
        seal_document.pop("cohort_sha256")
        seal_document.pop("manifest_path")
        seal_path = self.cohort / artifacts.SEAL_FILENAME
        seal_path.unlink()

        publisher_started = threading.Event()
        publisher_errors = []

        def publish_new_shard():
            publisher_started.set()
            try:
                self.publish(0, 1, (0,), rollout_id=8)
            except BaseException as exc:  # surfaced in the asserting thread
                publisher_errors.append(exc)

        contract_handle = (self.cohort / artifacts.CONTRACT_FILENAME).open("rb")
        try:
            fcntl.flock(contract_handle.fileno(), fcntl.LOCK_EX)
            publisher = threading.Thread(target=publish_new_shard)
            publisher.start()
            self.assertTrue(publisher_started.wait(timeout=5))
            artifacts.publish_immutable_json(seal_path, seal_document)
        finally:
            fcntl.flock(contract_handle.fileno(), fcntl.LOCK_UN)
            contract_handle.close()

        publisher.join(timeout=5)
        self.assertFalse(publisher.is_alive())
        self.assertEqual(len(publisher_errors), 1)
        self.assertIsInstance(publisher_errors[0], artifacts.ArtifactError)
        self.assertIn("after the cohort is sealed", str(publisher_errors[0]))
        self.assertFalse(
            (
                self.cohort
                / artifacts.SHARD_DIRECTORY
                / "rank-0000_rollout-00000008_part-0001.npz"
            ).exists()
        )
        self.assertEqual(artifacts.verify_cohort(self.cohort)["sealed"], True)

    def test_seal_rejects_orphan_payload(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        self.publish(0, 0, (0,))
        (self.cohort / artifacts.SHARD_DIRECTORY / "orphan.npz").write_bytes(
            b"unreferenced"
        )
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "inventory is not exact.*orphan.npz"
        ):
            artifacts.seal_cohort(self.cohort)

    def test_seal_rejects_orphan_json_hidden_staging_and_special_entries(self):
        cases = (
            ("orphan.json", b"{}", "schema mismatch"),
            (".payload.staging", b"partial", "unrecognized entry"),
            ("unexpected.txt", b"operator data", "unrecognized entry"),
        )
        for index, (name, content, pattern) in enumerate(cases):
            with self.subTest(name=name):
                cohort = self.root / f"cohort-{index}"
                artifacts.create_cohort(
                    cohort,
                    expected_ranks=(0,),
                    expected_global_env_ids=(0,),
                    expected_transitions=3,
                    view_contract=VIEW_CONTRACT,
                    action_contract=ACTION_CONTRACT,
                    **IDENTITY,
                    **SHA,
                )
                artifacts.publish_rollout_shard(
                    cohort,
                    rank=0,
                    rollout_id=7,
                    part_index=0,
                    arrays=rollout_arrays((0,)),
                    **IDENTITY,
                    **SHA,
                )
                (cohort / artifacts.SHARD_DIRECTORY / name).write_bytes(content)
                with self.assertRaisesRegex(artifacts.ArtifactError, pattern):
                    artifacts.seal_cohort(cohort)

        special = self.root / "cohort-special"
        artifacts.create_cohort(
            special,
            expected_ranks=(0,),
            expected_global_env_ids=(0,),
            expected_transitions=3,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **IDENTITY,
            **SHA,
        )
        artifacts.publish_rollout_shard(
            special,
            rank=0,
            rollout_id=7,
            part_index=0,
            arrays=rollout_arrays((0,)),
            **IDENTITY,
            **SHA,
        )
        (special / artifacts.SHARD_DIRECTORY / "unexpected.json").mkdir()
        with self.assertRaisesRegex(artifacts.ArtifactError, "non-regular entry"):
            artifacts.seal_cohort(special)

    def test_verify_rejects_file_added_after_seal(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        self.publish(0, 0, (0,))
        artifacts.seal_cohort(self.cohort)
        extra = self.cohort / artifacts.SHARD_DIRECTORY / "orphan.npz"
        extra.write_bytes(b"post-seal contamination")
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "inventory is not exact.*orphan.npz"
        ):
            artifacts.verify_cohort(self.cohort)

    def test_seal_requires_exact_rank_environment_and_transition_coverage(self):
        self.create()
        self.publish(0, 0, (0, 1))
        self.publish(1, 0, (2,))
        with self.assertRaisesRegex(artifacts.ArtifactError, "environment coverage"):
            artifacts.seal_cohort(self.cohort)

    def test_environment_parts_must_be_disjoint(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        self.publish(0, 0, (0,))
        self.publish(0, 1, (0,), rollout_id=8)
        with self.assertRaisesRegex(artifacts.ArtifactError, "multiple shard parts"):
            artifacts.seal_cohort(self.cohort)

    def test_payload_corruption_is_detected_by_sha256(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        report = self.publish(0, 0, (0,))
        payload = self.cohort / report["payload_path"]
        payload.chmod(0o600)
        with payload.open("ab") as handle:
            handle.write(b"corruption")
            handle.flush()
            os.fsync(handle.fileno())
        with self.assertRaisesRegex(artifacts.ArtifactError, "size does not match"):
            artifacts.verify_rollout_shard(self.cohort / report["manifest_path"])

    def test_duplicate_json_key_and_unsafe_relative_path_are_rejected(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        report = self.publish(0, 0, (0,))
        sidecar = self.cohort / report["manifest_path"]
        document = artifacts.load_strict_json(sidecar)
        sidecar.chmod(0o600)
        duplicate = sidecar.read_text(encoding="utf-8")[:-1] + ',"rank":0}'
        sidecar.write_text(duplicate, encoding="utf-8")
        with self.assertRaisesRegex(artifacts.ArtifactError, "duplicate JSON key"):
            artifacts.verify_rollout_shard(sidecar)

        # Restore syntactically valid JSON, then prove traversal is rejected
        # before any referenced path is opened.
        document["payload"]["path"] = "../outside.npz"
        sidecar.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(artifacts.ArtifactError, "safe relative"):
            artifacts.verify_rollout_shard(sidecar)

    def test_symlinked_payload_is_never_followed(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        report = self.publish(0, 0, (0,))
        payload = self.cohort / report["payload_path"]
        real = self.root / "real-payload.npz"
        payload.rename(real)
        payload.symlink_to(real)
        with self.assertRaisesRegex(artifacts.ArtifactError, "symbolic links"):
            artifacts.verify_rollout_shard(self.cohort / report["manifest_path"])

    def test_changed_contract_retry_is_rejected(self):
        self.create(ranks=(0,), envs=(0,), transitions=3)
        with self.assertRaisesRegex(artifacts.ArtifactError, "immutable artifact conflict"):
            artifacts.create_cohort(
                self.cohort,
                expected_ranks=(0,),
                expected_global_env_ids=(0,),
                expected_transitions=3,
                view_contract=VIEW_CONTRACT,
                action_contract=ACTION_CONTRACT,
                **IDENTITY,
                **{**SHA, "runtime_policy_sha256": "9" * 64},
            )


if __name__ == "__main__":
    unittest.main()
