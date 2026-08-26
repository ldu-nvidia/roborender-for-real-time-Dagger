import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger.artifacts import (
    publish_immutable_json,
    publish_rollout_shard,
    seal_cohort,
    sha256_file,
)
from round_dagger.runtime import (
    RUNTIME_ADAPTER_EVIDENCE_ENV,
    RUNTIME_ADAPTER_MODE_ENV,
    RUNTIME_ADAPTER_SHADOW_ACK,
    RUNTIME_ADAPTER_SHADOW_ACK_ENV,
    RUNTIME_ADAPTER_VALIDATION_ACK,
    RUNTIME_ADAPTER_VALIDATION_ACK_ENV,
)
from round_dagger.runtime_ab import (
    ARMS,
    EVIDENCE_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    RUNTIME_EVIDENCE_SCHEMA,
    WARMUP_SCHEMA,
    RuntimeABError,
    anchor_inventory,
    authorize_runtime_ab,
    compare_runtime_ab,
    prepare_runtime_ab,
    runtime_ab_environment,
    runtime_ab_warmup,
    validate_runtime_ab,
)
from scripts.round_dagger import verify_sealed_source_snapshot as source_seal


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class RuntimeAdapterABTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.fast_revision = "1" * 40
        self.gr00t_revision = "2" * 40
        self.repository = self.base / (
            f"fast_roborender_{self.fast_revision[:7]}_"
            f"{self.gr00t_revision[:7]}_sealed"
        )
        self.gr00t = self.repository / "GR00T-VisualSim2Real"
        self.gr00t.mkdir(parents=True)
        manifest = {
            "schema": source_seal.MANIFEST_SCHEMA,
            "status": "sealed_tracked_hydrated_sources",
            "snapshot_name": self.repository.name,
            "fast_roborender_revision": self.fast_revision,
            "gr00t_revision": self.gr00t_revision,
            "fast_roborender_index_tree": "3" * 40,
            "gr00t_index_tree": "4" * 40,
            "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
            "lfs_file_count": 0,
            "lfs_total_bytes": 0,
            "tracked_only": True,
            "lfs_pointer_oids_verified": True,
            "symlinks_rejected": True,
            "bytecode_rejected": True,
        }
        (self.repository / ".source_snapshot_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        (self.repository / ".source_revision").write_text(
            self.fast_revision + "\n", encoding="ascii"
        )
        (self.gr00t / ".source_revision").write_text(
            self.gr00t_revision + "\n", encoding="ascii"
        )
        for path in self.repository.rglob("*"):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(self.repository, 0o555)
        records, fast_tree, gr00t_tree = source_seal._tree_records(self.repository)
        inventory = {
            "schema": source_seal.INVENTORY_SCHEMA,
            "file_count": len(records),
            "total_file_bytes": sum(record["bytes"] for record in records),
            "files": records,
        }
        self.source_inventory = self.repository.parent / (
            f"{self.repository.name}.inventory.json"
        )
        self.source_inventory.write_text(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(self.source_inventory, 0o444)
        self.source_inventory_sha = sha256_file(self.source_inventory)
        self.source_attestation = self.repository.parent / (
            f"{self.repository.name}.snapshot_attestation.json"
        )
        self.source_attestation.write_text(
            json.dumps(
                {
                    "schema": source_seal.ATTESTATION_SCHEMA,
                    "status": "PASS",
                    "snapshot_name": self.repository.name,
                    "fast_roborender_revision": self.fast_revision,
                    "gr00t_revision": self.gr00t_revision,
                    "fast_roborender_index_tree": "3" * 40,
                    "gr00t_index_tree": "4" * 40,
                    "fast_roborender_source_tree_sha256": fast_tree,
                    "gr00t_source_tree_sha256": gr00t_tree,
                    "inventory_sha256": self.source_inventory_sha,
                    "deterministic_archive_sha256": digest("archive"),
                    "file_count": len(records),
                    "total_file_bytes": sum(record["bytes"] for record in records),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.chmod(self.source_attestation, 0o444)
        self.policy = self.base / "base-policy.pt"
        self.policy.write_bytes(b"base policy checkpoint")
        self.policy_spec = Path(f"{self.policy}.policy_spec.json")
        self.policy_spec.write_text(
            json.dumps({"schema": "test.policy_spec.v1"}) + "\n",
            encoding="utf-8",
        )
        self.policy_manifest = Path(f"{self.policy}.manifest.json")
        self.policy_manifest.write_text(
            json.dumps(
                {
                    "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
                    "checkpoint_path": self.policy.name,
                    "checkpoint_sha256": sha256_file(self.policy),
                    "runtime_policy_sha256": digest("runtime-policy"),
                    "policy_spec_path": self.policy_spec.name,
                    "policy_spec_sha256": sha256_file(self.policy_spec),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.teacher = self.base / "teacher.pt"
        self.teacher.write_bytes(b"teacher checkpoint")

        self.anchors = self.base / "anchors"
        self.anchors.mkdir()
        for index in range(2):
            np.savez(
                self.anchors / f"anchor-{index:04d}.npz",
                prompt=np.array("lift the colored cube"),
                **{
                    f"gen__{view}": np.full(
                        (2, 240, 416, 3), 10 + index, dtype=np.uint8
                    )
                    for view in ("head_color", "hand_left", "hand_right")
                },
            )
        self.anchor_count, self.anchor_sha = anchor_inventory(self.anchors)

        self.descriptors = []
        for index, port in enumerate((43101, 43102)):
            descriptor = self.base / f"renderer-{index}.endpoint"
            descriptor.write_text(f"127.0.0.1 {port}\n", encoding="utf-8")
            self.descriptors.append(descriptor)
        self.source_manifest = self.base / "renderers.txt"
        self.source_manifest.write_text(
            "".join(f"{path}\n" for path in self.descriptors), encoding="utf-8"
        )
        self.root = self.base / "runtime-ab"
        self.report = self.prepare()
        self.plan_path = self.root / "plan.json"

    def tearDown(self):
        for directory, directories, files in os.walk(
            self.repository, topdown=False
        ):
            for name in files:
                os.chmod(Path(directory) / name, 0o644)
            for name in directories:
                os.chmod(Path(directory) / name, 0o755)
            os.chmod(directory, 0o755)
        self.temporary.cleanup()

    def prepare(self, root=None):
        return prepare_runtime_ab(
            root or self.root,
            run_id="runtime-ab-test",
            endpoint_manifest=self.source_manifest,
            renderer_slot=1,
            actor_gpu_id=7,
            policy_checkpoint=self.policy,
            teacher_checkpoint=self.teacher,
            renderer_sha256=digest("renderer"),
            anchors=self.anchors,
            anchor_count=self.anchor_count,
            anchor_inventory_sha256=self.anchor_sha,
            repository_root=self.repository,
            source_inventory_sha256=self.source_inventory_sha,
            gr00t_root=self.gr00t,
            actor_image="round-dagger:test",
            actor_image_id_sha256=digest("actor-image"),
        )

    def documents(self):
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        arms = {
            reference["arm"]: json.loads(
                Path(reference["contract_path"]).read_text(encoding="utf-8")
            )
            for reference in plan["arms"]
        }
        return plan, arms

    @staticmethod
    def rollout_arrays(*, rgb_delta=0):
        expert = np.zeros((2, 1, 26), dtype=np.float32)
        student = np.ones((2, 1, 26), dtype=np.float32)
        expert_mask = np.array([[True], [False]], dtype=np.bool_)
        rgb = np.full((2, 1, 324, 192, 3), 17, dtype=np.uint8)
        rgb[0, 0, 0, 0, 0] = np.uint8(17 + rgb_delta)
        return {
            "policy_rgb": rgb,
            "expert_actions": expert,
            "student_actions": student,
            "behavior_actions": np.where(expert_mask[..., None], expert, student),
            "expert_mask": expert_mask,
            "dones": np.zeros((2, 1), dtype=np.bool_),
            "observation_is_raytraced": np.ones((2, 1), dtype=np.bool_),
            "global_env_ids": np.zeros((2, 1), dtype=np.int64),
            "episode_ids": np.zeros((2, 1), dtype=np.int64),
            "episode_steps": np.array([[0], [1]], dtype=np.int64),
            "obs__actor_obs": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
        }

    def publish_arm(self, arm, *, rgb_delta=0):
        plan, arms = self.documents()
        document = arms[arm]
        publish_rollout_shard(
            document["cohort_dir"],
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=self.rollout_arrays(rgb_delta=rgb_delta),
            campaign_id=plan["campaign_id"],
            macro_round=plan["macro_round"],
            cohort_index=document["cohort_index"],
            purpose=plan["purpose"],
            policy_sha256=plan["policy_sha256"],
            runtime_policy_sha256=plan["runtime_policy_sha256"],
            teacher_sha256=plan["teacher_sha256"],
            renderer_sha256=plan["renderer_sha256"],
            beta_sha256=plan["beta_sha256"],
            config_sha256=plan["config_sha256"],
        )
        seal_cohort(document["cohort_dir"])

    @staticmethod
    def runtime_evidence_document(arm):
        cuda = {
            "enabled": True,
            "device": "cuda:0",
            "baseline_allocated_bytes": 100,
            "baseline_reserved_bytes": 120,
            "peak_allocated_bytes": 140,
            "peak_reserved_bytes": 180,
            "peak_delta_allocated_bytes": 40,
            "peak_delta_reserved_bytes": 60,
        }
        audit = {"enabled": False}
        if arm == "shadow":
            audit = {
                "enabled": True,
                "num_envs": 1,
                "expected_steps": 2,
                "decision_count": 2,
                "transition_count": 2,
                "callback_count": 6,
                "callback_order_valid": True,
                "sequential_steps_valid": True,
                "batch_dimensions_valid": True,
            }
        return {
            "schema": RUNTIME_EVIDENCE_SCHEMA,
            "status": "informational_only",
            "promotion_gate": False,
            "validation_ack": RUNTIME_ADAPTER_VALIDATION_ACK,
            "adapter_arm": arm,
            "rank": 0,
            "world_size": 1,
            "dimensions": {"local_num_envs": 1, "rollout_steps": 2},
            "rollout": {
                "wall_seconds": 1.25 if arm == "legacy" else 0.75,
                "synchronized": True,
                "cuda": cuda,
            },
            "audit": audit,
            "replay_export": {"successful": True, "published_shard_parts": 1},
        }

    def publish_observability(self, arm, *, replay=0, extra_step=False):
        _, arms = self.documents()
        document = arms[arm]
        run_dir = Path(document["run_dir"])
        scalar_dir = run_dir / "telemetry" / "scalars"
        scalar_dir.mkdir(parents=True)
        publish_immutable_json(
            document["runtime_evidence_path"], self.runtime_evidence_document(arm)
        )
        for step in range(2 + int(extra_step)):
            publish_immutable_json(
                scalar_dir / f"step_{step:09d}.json",
                {
                    "step": step,
                    "metrics": {
                        "renderer/inference_ms": 10.0 + step + (5 if arm == "legacy" else 0),
                        "renderer/reference_source_idempotent_replay": replay,
                        "renderer/reference_provenance_valid": 1,
                        "renderer/stream_step": step,
                        "renderer/parallel_batch_size": 1,
                    },
                },
            )

    def publish_warmup(self, **overrides):
        plan, _ = self.documents()
        document = {
            "schema": WARMUP_SCHEMA,
            "endpoint": plan["selected_endpoint_descriptor"],
            "endpoint_sha256": plan["selected_endpoint_descriptor_sha256"],
            "anchor": plan["warmup_anchor"],
            "anchor_sha256": plan["warmup_anchor_sha256"],
            "renderer_sha256": plan["renderer_sha256"],
            "stream_id": plan["warmup_stream_id"],
            "seed": 42,
            "reference_lag": 256,
            "replayed": False,
            "inference_ms": 12.5,
            "client_wall_ms": 14.25,
            "reference_source": "anchor",
            "generated_sha256": {
                view: digest(f"warmup-{view}")
                for view in ("head_color", "hand_left", "hand_right")
            },
        }
        document.update(overrides)
        publish_immutable_json(plan["warmup_evidence_path"], document)

    def complete(self, *, shadow_rgb_delta=0):
        self.publish_warmup()
        for arm in ARMS:
            self.publish_arm(
                arm, rgb_delta=shadow_rgb_delta if arm == "shadow" else 0
            )
            self.publish_observability(arm)

    def test_prepare_validate_and_exact_two_arm_identity(self):
        report = validate_runtime_ab(self.plan_path)
        plan, arms = self.documents()
        self.assertEqual(plan["schema"], PLAN_SCHEMA)
        self.assertEqual(report["states"], ["empty", "empty"])
        self.assertEqual([item["arm"] for item in plan["arms"]], list(ARMS))
        self.assertEqual(report["actor_gpu_id"], 7)
        self.assertEqual(report["renderer_slot"], 1)
        self.assertEqual(report["global_env_id"], 0)
        self.assertEqual(report["warmup"]["state"], "pending")
        self.assertEqual(report["warmup"]["stream_id"], "runtime-ab-test-warmup")
        self.assertEqual(report["source"]["repository_root"], str(self.repository))
        self.assertEqual(
            report["source"]["source_inventory_path"], str(self.source_inventory)
        )
        self.assertEqual(
            report["source"]["source_attestation_path"], str(self.source_attestation)
        )
        self.assertEqual(
            report["source"]["fast_roborender_revision"], self.fast_revision
        )
        self.assertEqual(report["source"]["gr00t_revision"], self.gr00t_revision)
        self.assertEqual(
            arms["legacy"]["config_sha256"], arms["shadow"]["config_sha256"]
        )
        self.assertEqual(
            arms["legacy"]["endpoint_manifest"], arms["shadow"]["endpoint_manifest"]
        )
        self.assertNotEqual(
            arms["legacy"]["image3f_run_id"], arms["shadow"]["image3f_run_id"]
        )
        self.assertEqual(
            Path(arms["legacy"]["runtime_evidence_path"]).parent,
            Path(arms["legacy"]["run_dir"]),
        )
        self.assertNotEqual(
            arms["legacy"]["runtime_evidence_path"],
            arms["legacy"]["comparison_evidence_path"],
        )
        selected = Path(plan["selected_endpoint_manifest"]).read_text(encoding="utf-8")
        self.assertEqual(selected, f"{self.descriptors[1]}\n")
        retry = self.prepare()
        self.assertEqual(retry["plan_sha256"], report["plan_sha256"])

    def test_environment_and_authorization_are_exact_and_arm_specific(self):
        pending = runtime_ab_warmup(self.plan_path)
        self.assertEqual(pending["state"], "pending")
        with self.assertRaisesRegex(RuntimeABError, "warm-up must be validated"):
            runtime_ab_environment(self.plan_path, "legacy")
        self.publish_warmup()
        validated = runtime_ab_warmup(self.plan_path)
        self.assertEqual(validated["state"], "validated")
        self.assertEqual(validated["evidence_sha256"], sha256_file(validated["output"]))
        legacy = runtime_ab_environment(self.plan_path, "legacy")
        shadow = runtime_ab_environment(self.plan_path, "shadow")
        self.assertEqual(legacy["ROUND_DAGGER_VALIDATION_RANKS"], "1")
        self.assertEqual(legacy["ROUND_DAGGER_VALIDATION_ENVS_PER_RANK"], "1")
        self.assertEqual(legacy["ROUND_DAGGER_VALIDATION_STEPS"], "2")
        self.assertEqual(legacy["ROUND_DAGGER_VALIDATION_PURPOSE"], "train")
        self.assertEqual(legacy["ROUND_DAGGER_ACTOR_DOCKER_GPU_ID"], "7")
        self.assertEqual(legacy[RUNTIME_ADAPTER_MODE_ENV], "legacy")
        self.assertEqual(shadow[RUNTIME_ADAPTER_MODE_ENV], "shadow")
        self.assertEqual(legacy[RUNTIME_ADAPTER_SHADOW_ACK_ENV], "")
        self.assertEqual(
            shadow[RUNTIME_ADAPTER_SHADOW_ACK_ENV], RUNTIME_ADAPTER_SHADOW_ACK
        )
        for environment in (legacy, shadow):
            self.assertEqual(
                environment[RUNTIME_ADAPTER_VALIDATION_ACK_ENV],
                RUNTIME_ADAPTER_VALIDATION_ACK,
            )
            runtime_path = Path(environment[RUNTIME_ADAPTER_EVIDENCE_ENV])
            self.assertEqual(runtime_path.parent, Path(environment["TRAIN_RUN_DIR"]))
            self.assertNotEqual(
                environment[RUNTIME_ADAPTER_EVIDENCE_ENV],
                environment["ROUND_DAGGER_RUNTIME_AB_EVIDENCE"],
            )
            self.assertEqual(
                environment["ROUND_DAGGER_RUNTIME_AB_WARMUP_EVIDENCE_SHA256"],
                validated["evidence_sha256"],
            )
        self.assertNotEqual(legacy["IMAGE3F_RUN_ID"], shadow["IMAGE3F_RUN_ID"])
        authorized = authorize_runtime_ab(
            self.plan_path,
            "shadow",
            self.report["plan_sha256"],
            environment=shadow,
        )
        self.assertEqual(authorized["status"], "authorized")
        tampered = dict(shadow)
        tampered[RUNTIME_ADAPTER_MODE_ENV] = "legacy"
        with self.assertRaisesRegex(RuntimeABError, "environment differs"):
            authorize_runtime_ab(
                self.plan_path,
                "shadow",
                self.report["plan_sha256"],
                environment=tampered,
            )
        with self.assertRaisesRegex(RuntimeABError, "differs from authorization"):
            authorize_runtime_ab(
                self.plan_path, "shadow", digest("wrong-plan"), environment=shadow
            )

    def test_noncanonical_root_and_mutated_input_fail_closed(self):
        with self.assertRaisesRegex(RuntimeABError, "root"):
            self.prepare(self.base / "unused" / ".." / "noncanonical")
        self.descriptors[1].write_text("127.0.0.1 44000\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeABError, "descriptor changed"):
            validate_runtime_ab(self.plan_path)

    def test_mutated_sealed_source_is_rejected_after_plan_publication(self):
        marker = self.repository / ".source_revision"
        os.chmod(marker, 0o644)
        marker.write_text("9" * 40 + "\n", encoding="ascii")
        os.chmod(marker, 0o444)
        with self.assertRaisesRegex(RuntimeABError, "sealed repository verification failed"):
            validate_runtime_ab(self.plan_path)

    def test_replayed_warmup_evidence_is_rejected(self):
        self.publish_warmup(replayed=True)
        with self.assertRaisesRegex(RuntimeABError, "warm-up evidence differs"):
            validate_runtime_ab(self.plan_path)

    def test_exact_comparison_publishes_separate_evidence_and_is_idempotent(self):
        self.complete()
        result = compare_runtime_ab(self.plan_path)
        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["exact_array_match"])
        self.assertTrue(result["exact_payload_match"])
        self.assertEqual(result["transition_count_per_arm"], 2)
        self.assertEqual(result["performance"]["legacy_rollout_wall_ms"], 1250.0)
        self.assertEqual(result["performance"]["shadow_rollout_wall_ms"], 750.0)
        self.assertAlmostEqual(
            result["performance"]["shadow_over_legacy_ratio"], 0.6
        )
        self.assertFalse(result["performance"]["promotion_gate"])
        for arm in ARMS:
            receipt = result["arms"][arm]
            comparison_path = Path(receipt["comparison_evidence_path"])
            runtime_path = Path(receipt["runtime_evidence_path"])
            self.assertNotEqual(comparison_path, runtime_path)
            evidence = json.loads(comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["schema"], EVIDENCE_SCHEMA)
            self.assertEqual(evidence["runtime_evidence"]["path"], str(runtime_path))
            self.assertEqual(evidence["telemetry"]["provenance_valid_count"], 2)
            self.assertEqual(evidence["telemetry"]["idempotent_replay_count"], 0)
        repeated = compare_runtime_ab(self.plan_path)
        self.assertEqual(repeated["result_sha256"], result["result_sha256"])

    def test_rollout_mismatch_publishes_no_comparison_artifacts(self):
        self.complete(shadow_rgb_delta=1)
        _, arms = self.documents()
        with self.assertRaisesRegex(RuntimeABError, "arrays differ exactly"):
            compare_runtime_ab(self.plan_path)
        self.assertFalse((self.root / "result.json").exists())
        for arm in ARMS:
            self.assertFalse(Path(arms[arm]["comparison_evidence_path"]).exists())

    def test_missing_runtime_evidence_publishes_nothing(self):
        self.publish_warmup()
        for arm in ARMS:
            self.publish_arm(arm)
        self.publish_observability("shadow")
        _, arms = self.documents()
        with self.assertRaisesRegex(RuntimeABError, "legacy runtime evidence"):
            compare_runtime_ab(self.plan_path)
        self.assertFalse((self.root / "result.json").exists())
        self.assertFalse(Path(arms["legacy"]["comparison_evidence_path"]).exists())
        self.assertFalse(Path(arms["shadow"]["comparison_evidence_path"]).exists())

    def test_replayed_or_extra_scalar_telemetry_fails_closed(self):
        self.publish_warmup()
        for arm in ARMS:
            self.publish_arm(arm)
        self.publish_observability("legacy", replay=1)
        self.publish_observability("shadow", extra_step=True)
        _, arms = self.documents()
        with self.assertRaisesRegex(RuntimeABError, "idempotent_replay"):
            compare_runtime_ab(self.plan_path)
        self.assertFalse((self.root / "result.json").exists())
        for arm in ARMS:
            self.assertFalse(Path(arms[arm]["comparison_evidence_path"]).exists())

    def test_mutable_runtime_evidence_is_rejected(self):
        self.complete()
        _, arms = self.documents()
        os.chmod(arms["legacy"]["runtime_evidence_path"], 0o644)
        with self.assertRaisesRegex(RuntimeABError, "mode must be exactly 0444"):
            compare_runtime_ab(self.plan_path)
        self.assertFalse((self.root / "result.json").exists())
        for arm in ARMS:
            self.assertFalse(Path(arms[arm]["comparison_evidence_path"]).exists())

    def test_hostile_shadow_target_cannot_leave_partial_legacy_evidence(self):
        self.complete()
        _, arms = self.documents()
        shadow_target = Path(arms["shadow"]["comparison_evidence_path"])
        shadow_target.write_bytes(b"hostile preexisting bytes")
        with self.assertRaisesRegex(RuntimeABError, "immutable comparison artifact conflict"):
            compare_runtime_ab(self.plan_path)
        self.assertFalse(Path(arms["legacy"]["comparison_evidence_path"]).exists())
        self.assertEqual(shadow_target.read_bytes(), b"hostile preexisting bytes")
        self.assertFalse((self.root / "result.json").exists())

    def test_cli_exposes_all_validation_operations(self):
        completed = subprocess.run(
            [sys.executable, "-m", "round_dagger.runtime_ab", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        for operation in (
            "prepare",
            "validate",
            "environment",
            "warmup",
            "compare",
            "authorize",
        ):
            self.assertIn(operation, completed.stdout)


if __name__ == "__main__":
    unittest.main()
