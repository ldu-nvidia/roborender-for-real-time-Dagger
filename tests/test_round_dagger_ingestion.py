import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger import orchestration
from round_dagger import replay
from round_dagger.orchestration import ProductionBackend
from round_dagger.production_data_lineage import (
    build_cohort_assignment,
    derive_data_lineage,
)
from round_dagger.seed_policy import derive_actor_seed_contract


def _view_contract():
    return {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": [324, 192, 3],
        "views": ["head_color", "hand_left", "hand_right"],
    }


def _action_contract():
    return {"dtype": "float32", "shape": [16]}


def _production_data_lineage():
    motion_split = {
        "manifest_sha256": (
            "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834"
        ),
        "heldout_plan_sha256": "7" * 64,
        "collection_role": "collection",
        "evaluation_role": "eval",
        "collection_motion_count": 39,
        "heldout_motion_count": 256,
    }
    return derive_data_lineage(
        {
            "contract_sha256": "8" * 64,
            "invariant_profile_sha256": "9" * 64,
            "live_capture": {
                "attestation_sha256": "a" * 64,
                "evidence_sha256": "b" * 64,
                "scenario_sha256": "c" * 64,
            },
            "capture_provenance": {
                "motion_split_manifest_sha256": motion_split["manifest_sha256"],
                "heldout_plan_sha256": motion_split["heldout_plan_sha256"],
            },
        },
        motion_split_contract=motion_split,
        topology={
            "cohort_count": 4,
            "actor_ranks_per_cohort": 2,
            "environments_per_actor_rank": 32,
            "active_environments": 64,
        },
    )


_PRODUCTION_PROVENANCE = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}


def _publish_minimal_production_shards(
    cohort, *, campaign_id, cohort_index, global_env_ids
):
    for rank, global_env_id in enumerate(global_env_ids):
        expert_actions = np.full(
            (1, 1, 2), float(global_env_id + 1), dtype=np.float32
        )
        arrays = {
            "policy_rgb": np.full(
                (1, 1, 6, 4, 3), global_env_id + 1, dtype=np.uint8
            ),
            "expert_actions": expert_actions,
            "student_actions": np.zeros((1, 1, 2), dtype=np.float32),
            "behavior_actions": expert_actions.copy(),
            "expert_mask": np.ones((1, 1), dtype=np.bool_),
            "dones": np.ones((1, 1), dtype=np.bool_),
            "observation_is_raytraced": np.ones((1, 1), dtype=np.bool_),
            "global_env_ids": np.array([global_env_id], dtype=np.int64),
            "episode_ids": np.zeros((1, 1), dtype=np.int64),
            "episode_steps": np.zeros((1, 1), dtype=np.int64),
        }
        artifacts.publish_rollout_shard(
            cohort,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=arrays,
            campaign_id=campaign_id,
            macro_round=0,
            cohort_index=cohort_index,
            purpose="train",
            **_PRODUCTION_PROVENANCE,
        )
    return artifacts.seal_cohort(cohort)


def _cohort(
    macro_round,
    cohort_index,
    *,
    purpose="train",
    campaign_id="campaign-a",
    transitions=(3, 2),
    digest_tag=None,
    view_contract=None,
    action_contract=None,
):
    tag = digest_tag or f"{macro_round}-{cohort_index}-{purpose}"

    def digest(label):
        return hashlib.sha256(f"{tag}:{label}".encode()).hexdigest()

    shards = [
        {
            "shard_id": f"r{macro_round:02d}-c{cohort_index:02d}-s{index:02d}",
            "manifest_sha256": digest(f"manifest-{index}"),
            "payload_sha256": digest(f"payload-{index}"),
            "transition_count": count,
        }
        for index, count in enumerate(transitions)
    ]
    return {
        "schema": replay.COHORT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "purpose": purpose,
        "sealed": True,
        "transition_count": sum(transitions),
        "view_contract": view_contract or _view_contract(),
        "action_contract": action_contract or _action_contract(),
        "shards": shards,
    }


class ArtifactHarness:
    def __init__(self, root):
        self.root = Path(root)
        self.documents = {}

    def add(self, name, document, *, bytes_tag=None):
        path = self.root / name
        path.write_text(bytes_tag or name, encoding="utf-8")
        document = json.loads(json.dumps(document))
        shard_root = self.root / "shards" / path.stem
        shard_root.mkdir(parents=True, exist_ok=True)
        for shard in document["shards"]:
            shard_id = shard["shard_id"]
            manifest = shard_root / f"{shard_id}.json"
            payload = shard_root / f"{shard_id}.npz"
            manifest.write_bytes(
                shard.pop("_manifest_content", f"{name}:{shard_id}:manifest").encode()
            )
            payload.write_bytes(
                shard.pop("_payload_content", f"{name}:{shard_id}:payload").encode()
            )
            shard["shard_manifest_path"] = manifest.relative_to(self.root).as_posix()
            shard["payload_path"] = payload.relative_to(self.root).as_posix()
            shard["manifest_sha256"] = self.sha256_file(manifest)
            shard["payload_sha256"] = self.sha256_file(payload)
        self.documents[str(path.resolve())] = document
        return path

    def verify_cohort(self, path):
        return self.documents[str(Path(path).resolve())]

    @staticmethod
    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def canonical_json_sha256(document):
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def publish_immutable_json(path, document):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError("refusing to replace immutable snapshot")
        else:
            path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def patches(self):
        return mock.patch.multiple(
            replay._artifacts,
            verify_cohort=self.verify_cohort,
            sha256_file=self.sha256_file,
            canonical_json_sha256=self.canonical_json_sha256,
            publish_immutable_json=self.publish_immutable_json,
            create=True,
        )


class ReplayLedgerIngestionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = ArtifactHarness(self.root)
        self.patch = self.artifacts.patches()
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def ledger(self):
        return replay.ReplayLedger(self.root / "replay.sqlite3")

    def test_transactional_ingest_and_identical_retry_are_idempotent(self):
        path = self.artifacts.add("cohort.json", _cohort(0, 0))
        with self.ledger() as ledger:
            accepted = ledger.ingest_cohort(path)
            repeated = ledger.ingest_cohort(path)
            counts = ledger.counts("campaign-a")
        self.assertEqual(accepted["status"], "ACCEPTED")
        self.assertEqual(repeated["status"], "ALREADY_ACCEPTED")
        self.assertEqual(accepted["manifest_sha256"], repeated["manifest_sha256"])
        self.assertEqual(counts["cohort_count"], 1)
        self.assertEqual(counts["transition_count"], 5)
        self.assertEqual(counts["by_round_cohort"], {"0:0:train": 5})

    def test_same_logical_id_with_different_manifest_digest_is_rejected(self):
        first = self.artifacts.add("first.json", _cohort(0, 0), bytes_tag="first")
        second = self.artifacts.add("second.json", _cohort(0, 0), bytes_tag="second")
        with self.ledger() as ledger:
            ledger.ingest_cohort(first)
            with self.assertRaisesRegex(replay.ReplayError, "logical ID.*different"):
                ledger.ingest_cohort(second)
            self.assertEqual(ledger.counts("campaign-a")["cohort_count"], 1)

    def test_contract_mismatch_rolls_back_complete_cohort(self):
        first = self.artifacts.add("first.json", _cohort(0, 0))
        wrong_view = _view_contract()
        wrong_view["views"] = ["head_color"]
        second = self.artifacts.add(
            "second.json", _cohort(0, 1, view_contract=wrong_view)
        )
        with self.ledger() as ledger:
            ledger.ingest_cohort(first)
            with self.assertRaisesRegex(replay.ReplayError, "schema/view/action"):
                ledger.ingest_cohort(second)
            counts = ledger.counts("campaign-a")
        self.assertEqual(counts["cohort_count"], 1)
        self.assertEqual(counts["transition_count"], 5)

    def test_campaign_rejects_mixed_v1_and_v2_cohort_schemas(self):
        for index, (first_schema, second_schema) in enumerate(
            (
                (replay.COHORT_SCHEMA, replay.PRODUCTION_COHORT_SCHEMA),
                (replay.PRODUCTION_COHORT_SCHEMA, replay.COHORT_SCHEMA),
            )
        ):
            first_document = _cohort(0, 0)
            first_document["schema"] = first_schema
            second_document = _cohort(0, 1)
            second_document["schema"] = second_schema
            first = self.artifacts.add(f"mixed-{index}-first.json", first_document)
            second = self.artifacts.add(
                f"mixed-{index}-second.json", second_document
            )
            with replay.ReplayLedger(self.root / f"mixed-{index}.sqlite3") as ledger:
                ledger.ingest_cohort(first)
                with self.assertRaisesRegex(
                    replay.ReplayError, "schema/view/action"
                ):
                    ledger.ingest_cohort(second)
                counts = ledger.counts("campaign-a")
            self.assertEqual(counts["cohort_count"], 1)
            self.assertEqual(counts["transition_count"], 5)

    def test_reused_shard_digest_rolls_back_without_partial_rows(self):
        first_document = _cohort(0, 0)
        second_document = _cohort(0, 1)
        first_document["shards"][0]["_manifest_content"] = "shared-manifest"
        second_document["shards"][0]["_manifest_content"] = "shared-manifest"
        first = self.artifacts.add("first.json", first_document)
        second = self.artifacts.add("second.json", second_document)
        with self.ledger() as ledger:
            ledger.ingest_cohort(first)
            with self.assertRaisesRegex(replay.ReplayError, "transactional"):
                ledger.ingest_cohort(second)
            counts = ledger.counts("campaign-a")
        self.assertEqual(counts["cohort_count"], 1)
        self.assertEqual(counts["by_round_cohort"], {"0:0:train": 5})

    def test_eval_cohorts_are_accepted_but_excluded_from_training_counts(self):
        train = self.artifacts.add("train.json", _cohort(0, 0))
        evaluate = self.artifacts.add("eval.json", _cohort(0, 0, purpose="eval"))
        with self.ledger() as ledger:
            ledger.ingest_cohort(train)
            ledger.ingest_cohort(evaluate)
            training = ledger.counts("campaign-a")
            all_counts = ledger.counts("campaign-a", include_eval=True)
        self.assertEqual(training["cohort_count"], 1)
        self.assertEqual(training["transition_count"], 5)
        self.assertEqual(all_counts["cohort_count"], 2)
        self.assertEqual(all_counts["transition_count"], 10)

    def test_ledger_restart_preserves_the_exact_ingestion_frontier(self):
        path = self.artifacts.add("cohort.json", _cohort(0, 0))
        with self.ledger() as ledger:
            first = ledger.ingest_cohort(path)
        with self.ledger() as ledger:
            second = ledger.ingest_cohort(path)
            counts = ledger.counts("campaign-a")
        self.assertEqual(first["status"], "ACCEPTED")
        self.assertEqual(second["status"], "ALREADY_ACCEPTED")
        self.assertEqual(counts["transition_count"], 5)

    def test_unsealed_wrong_schema_or_inconsistent_shard_counts_fail_before_insert(self):
        variants = []
        unsealed = _cohort(0, 0)
        unsealed["sealed"] = False
        variants.append(unsealed)
        wrong_schema = _cohort(0, 0)
        wrong_schema["schema"] = "unknown"
        variants.append(wrong_schema)
        bad_count = _cohort(0, 0)
        bad_count["transition_count"] += 1
        variants.append(bad_count)
        with self.ledger() as ledger:
            for index, document in enumerate(variants):
                path = self.artifacts.add(f"bad-{index}.json", document)
                with self.subTest(index=index), self.assertRaises(replay.ReplayError):
                    ledger.ingest_cohort(path)
            self.assertEqual(ledger.counts("campaign-a")["cohort_count"], 0)


class ReplaySnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = ArtifactHarness(self.root)
        self.patch = self.artifacts.patches()
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def _ingest_round(self, ledger, macro_round, cohort_order=(0, 1)):
        for cohort_index in cohort_order:
            path = self.artifacts.add(
                f"round-{macro_round}-cohort-{cohort_index}.json",
                _cohort(macro_round, cohort_index),
            )
            ledger.ingest_cohort(path)

    def test_snapshot_requires_every_cohort_in_every_round_from_zero(self):
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            path = self.artifacts.add("only.json", _cohort(0, 0))
            ledger.ingest_cohort(path)
            with self.assertRaisesRegex(replay.ReplayError, "exactly cohorts"):
                ledger.build_snapshot(
                    "campaign-a",
                    through_macro_round=0,
                    expected_train_cohorts=2,
                    output=self.root / "snapshot.json",
                )

    def test_snapshot_is_cumulative_sorted_and_lists_all_content_digests(self):
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            self._ingest_round(ledger, 0, cohort_order=(1, 0))
            self._ingest_round(ledger, 1, cohort_order=(1, 0))
            report = ledger.build_snapshot(
                "campaign-a",
                through_macro_round=1,
                expected_train_cohorts=2,
                output=self.root / "snapshot.json",
            )
        self.assertEqual(report["cohort_count"], 4)
        self.assertEqual(report["transition_count"], 20)
        self.assertEqual(report["counts_by_round"], {"0": 10, "1": 10})
        self.assertEqual(
            [(item["macro_round"], item["cohort_index"]) for item in report["cohorts"]],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        for cohort in report["cohorts"]:
            self.assertRegex(cohort["manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(cohort["shards"]), 2)
            for shard in cohort["shards"]:
                self.assertRegex(shard["manifest_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(shard["payload_sha256"], r"^[0-9a-f]{64}$")
        expected_semantic_digest = report.pop("snapshot_sha256")
        report.pop("path")
        report.pop("file_sha256")
        self.assertEqual(
            expected_semantic_digest,
            self.artifacts.canonical_json_sha256(report),
        )

    def test_existing_snapshot_does_not_change_when_later_round_arrives(self):
        output = self.root / "through-zero.json"
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            self._ingest_round(ledger, 0)
            before = ledger.build_snapshot(
                "campaign-a",
                through_macro_round=0,
                expected_train_cohorts=2,
                output=output,
            )
            before_bytes = output.read_bytes()
            self._ingest_round(ledger, 1)
            after = ledger.build_snapshot(
                "campaign-a",
                through_macro_round=0,
                expected_train_cohorts=2,
                output=output,
            )
        self.assertEqual(output.read_bytes(), before_bytes)
        self.assertEqual(after["snapshot_sha256"], before["snapshot_sha256"])
        self.assertEqual(after["transition_count"], 10)

    def test_eval_cohort_is_absent_from_snapshot(self):
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            self._ingest_round(ledger, 0)
            evaluate = self.artifacts.add("eval.json", _cohort(0, 0, purpose="eval"))
            ledger.ingest_cohort(evaluate)
            report = ledger.build_snapshot(
                "campaign-a",
                through_macro_round=0,
                expected_train_cohorts=2,
                output=self.root / "snapshot.json",
            )
        self.assertEqual(report["cohort_count"], 2)
        self.assertTrue(all(":train" in item["logical_id"] for item in report["cohorts"]))

    def test_loader_returns_safe_payload_paths_and_rechecks_every_digest(self):
        output = self.root / "snapshot.json"
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            self._ingest_round(ledger, 0)
            report = ledger.build_snapshot(
                "campaign-a",
                through_macro_round=0,
                expected_train_cohorts=2,
                output=output,
            )
        loaded = replay.load_snapshot(
            output, expected_file_sha256=report["file_sha256"]
        )
        self.assertEqual(len(loaded["resolved_shards"]), 4)
        for shard in loaded["resolved_shards"]:
            self.assertTrue(Path(shard["shard_manifest_path"]).is_absolute())
            self.assertTrue(Path(shard["payload_path"]).is_absolute())
            self.assertTrue(Path(shard["payload_path"]).is_file())

        Path(loaded["resolved_shards"][0]["payload_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(replay.ReplayError, "payload digest mismatch"):
            replay.load_snapshot(output)

    def test_ingest_rejects_unsafe_or_tampered_payload_paths(self):
        document = _cohort(0, 0)
        path = self.artifacts.add("cohort.json", document)
        verified = self.artifacts.documents[str(path.resolve())]
        verified["shards"][0]["payload_path"] = "../escape.npz"
        with replay.ReplayLedger(self.root / "ledger.sqlite3") as ledger:
            with self.assertRaisesRegex(replay.ReplayError, "unsafe"):
                ledger.ingest_cohort(path)

        path = self.artifacts.add("tamper.json", _cohort(0, 0))
        verified = self.artifacts.documents[str(path.resolve())]
        payload = self.root / verified["shards"][0]["payload_path"]
        payload.write_bytes(b"changed-after-verification")
        with replay.ReplayLedger(self.root / "other.sqlite3") as ledger:
            with self.assertRaisesRegex(replay.ReplayError, "payload digest changed"):
                ledger.ingest_cohort(path)


class DeterministicReplayHelpersTest(unittest.TestCase):
    def test_partition_is_stable_bounded_and_seeded(self):
        first = replay.deterministic_partition("env-17", 8, seed=42)
        self.assertEqual(first, replay.deterministic_partition("env-17", 8, seed=42))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 8)
        assignments = {
            replay.deterministic_partition(f"env-{index}", 8, seed=42)
            for index in range(64)
        }
        self.assertGreater(len(assignments), 1)

    def test_sample_is_order_independent_reproducible_and_without_replacement(self):
        population = [f"shard-{index}" for index in range(20)]
        first = replay.deterministic_sample(population, 7, seed=11)
        reverse = replay.deterministic_sample(reversed(population), 7, seed=11)
        self.assertEqual(first, reverse)
        self.assertEqual(len(first), 7)
        self.assertEqual(len(set(first)), 7)
        self.assertNotEqual(first, replay.deterministic_sample(population, 7, seed=12))

    def test_sample_rejects_duplicates_and_oversubscription(self):
        with self.assertRaisesRegex(replay.ReplayError, "unique"):
            replay.deterministic_sample(["a", "a"], 1, seed=0)
        with self.assertRaisesRegex(replay.ReplayError, "exceeds"):
            replay.deterministic_sample(["a"], 2, seed=0)


class RealArtifactReplayIntegrationTest(unittest.TestCase):
    def test_sealed_real_cohort_ingests_snapshots_and_resolves_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = root / "cohort"
            provenance = {
                "policy_sha256": "a" * 64,
                "runtime_policy_sha256": "b" * 64,
                "teacher_sha256": "c" * 64,
                "renderer_sha256": "d" * 64,
                "beta_sha256": "e" * 64,
                "config_sha256": "f" * 64,
            }
            view_contract = _view_contract()
            view_contract["shape"] = [6, 4, 3]
            action_contract = {"dtype": "float32", "shape": [2]}
            artifacts.create_cohort(
                cohort,
                campaign_id="campaign-real",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                expected_ranks=[0],
                expected_global_env_ids=[0, 1],
                expected_transitions=4,
                view_contract=view_contract,
                action_contract=action_contract,
                **provenance,
            )
            steps, envs = 2, 2
            episode_steps = np.repeat(
                np.arange(steps, dtype=np.int64)[:, None], envs, axis=1
            )
            arrays = {
                "policy_rgb": np.zeros((steps, envs, 6, 4, 3), np.uint8),
                "expert_actions": np.ones((steps, envs, 2), np.float32),
                "student_actions": np.zeros((steps, envs, 2), np.float32),
                "behavior_actions": np.ones((steps, envs, 2), np.float32),
                "expert_mask": np.ones((steps, envs), np.bool_),
                "dones": np.zeros((steps, envs), np.bool_),
                "observation_is_raytraced": np.zeros((steps, envs), np.bool_),
                "global_env_ids": np.array([0, 1], np.int64),
                "episode_ids": np.zeros((steps, envs), np.int64),
                "episode_steps": episode_steps,
            }
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=arrays,
                campaign_id="campaign-real",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                **provenance,
            )
            seal = artifacts.seal_cohort(cohort)
            self.assertEqual(seal["transition_count"], 4)

            output = root / "snapshot.json"
            with replay.ReplayLedger(root / "ledger.sqlite3") as ledger:
                accepted = ledger.ingest_cohort(cohort)
                snapshot = ledger.build_snapshot(
                    "campaign-real",
                    through_macro_round=0,
                    expected_train_cohorts=1,
                    output=output,
                )
            self.assertEqual(accepted["status"], "ACCEPTED")
            self.assertEqual(snapshot["transition_count"], 4)
            loaded = replay.load_snapshot(
                output, expected_file_sha256=snapshot["file_sha256"]
            )
            self.assertEqual(len(loaded["resolved_shards"]), 1)
            payload = Path(loaded["resolved_shards"][0]["payload_path"])
            self.assertTrue(payload.is_file())
            self.assertEqual(artifacts.sha256_file(payload), seal["shards"][0]["payload_sha256"])

    def test_production_backend_ingests_verified_v2_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_id = "production-v2-ingest"
            campaign = {"campaign_id": campaign_id}
            lineage = _production_data_lineage()
            round_contract = {
                "policy_checkpoint_sha256": _PRODUCTION_PROVENANCE[
                    "policy_sha256"
                ],
                "policy_runtime_sha256": _PRODUCTION_PROVENANCE[
                    "runtime_policy_sha256"
                ],
                "teacher_sha256": _PRODUCTION_PROVENANCE["teacher_sha256"],
                "renderer_sha256": _PRODUCTION_PROVENANCE["renderer_sha256"],
                "beta_sha256": _PRODUCTION_PROVENANCE["beta_sha256"],
                "training_config_sha256": _PRODUCTION_PROVENANCE[
                    "config_sha256"
                ],
                "policy_image_shape": [6, 4, 3],
                "view_names": ["head_color", "hand_left", "hand_right"],
                "action_dim": 2,
                "data_lineage": lineage,
                "cohorts": [
                    {
                        "motion_assignment": build_cohort_assignment(
                            lineage,
                            cohort_index=cohort_index,
                            expected_ranks=(0, 1),
                        )
                    }
                    for cohort_index in range(4)
                ],
            }
            backend = ProductionBackend(
                types.SimpleNamespace(campaign_root=root, macro_round=0)
            )
            collection = {"cohorts": []}
            with mock.patch.multiple(
                orchestration,
                ENVIRONMENTS_PER_COHORT=2,
                TRANSITIONS_PER_COHORT=2,
            ):
                for cohort_index in range(4):
                    cohort = (
                        root
                        / "rounds"
                        / "macro_round_000000"
                        / "cohorts"
                        / f"cohort_{cohort_index:02d}"
                    )
                    backend.ensure_cohort_contract(
                        cohort,
                        campaign=campaign,
                        contract=round_contract,
                        cohort_index=cohort_index,
                    )
                    global_env_ids = [
                        cohort_index * 2,
                        cohort_index * 2 + 1,
                    ]
                    seal = _publish_minimal_production_shards(
                        cohort,
                        campaign_id=campaign_id,
                        cohort_index=cohort_index,
                        global_env_ids=global_env_ids,
                    )
                    self.assertEqual(
                        seal["schema"], artifacts.PRODUCTION_COHORT_SCHEMA
                    )
                    collection["cohorts"].append(
                        {
                            "cohort_manifest_path": cohort.relative_to(root).as_posix()
                        }
                    )

            with mock.patch.object(
                backend, "load_campaign", return_value=campaign
            ):
                snapshot = backend.ingest_collection(collection)
            self.assertEqual(
                snapshot["cohort_manifest_schema"],
                artifacts.PRODUCTION_COHORT_SCHEMA,
            )
            self.assertEqual(snapshot["cohort_count"], 4)
            self.assertEqual(snapshot["transition_count"], 8)
            loaded = backend.load_snapshot(
                expected_file_sha256=snapshot["file_sha256"]
            )
            self.assertEqual(len(loaded["resolved_shards"]), 8)

    def test_tampered_v2_lineage_is_rejected_before_ledger_insert(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_id = "production-v2-invalid"
            lineage = _production_data_lineage()
            cohort = root / "cohort"
            artifacts.create_cohort(
                cohort,
                campaign_id=campaign_id,
                macro_round=0,
                cohort_index=0,
                purpose="train",
                expected_ranks=[0, 1],
                expected_global_env_ids=[0, 1],
                expected_transitions=2,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [2]},
                actor_seed_contract=derive_actor_seed_contract(
                    campaign_id=campaign_id,
                    macro_round=0,
                    cohort_index=0,
                    expected_ranks=[0, 1],
                    expected_global_env_ids=[0, 1],
                ),
                data_lineage=lineage,
                motion_assignment=build_cohort_assignment(
                    lineage, cohort_index=0, expected_ranks=(0, 1)
                ),
                **_PRODUCTION_PROVENANCE,
            )
            _publish_minimal_production_shards(
                cohort,
                campaign_id=campaign_id,
                cohort_index=0,
                global_env_ids=[0, 1],
            )
            seal_path = cohort / artifacts.SEAL_FILENAME
            seal = artifacts.load_strict_json(seal_path)
            seal["data_lineage_sha256"] = "0" * 64
            seal_path.chmod(0o600)
            seal_path.write_text(
                json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with replay.ReplayLedger(root / "ledger.sqlite3") as ledger:
                with self.assertRaisesRegex(
                    replay.ReplayError, "cohort verification failed"
                ):
                    ledger.ingest_cohort(cohort)
                self.assertEqual(
                    ledger.counts(campaign_id)["cohort_count"], 0
                )


if __name__ == "__main__":
    unittest.main()
