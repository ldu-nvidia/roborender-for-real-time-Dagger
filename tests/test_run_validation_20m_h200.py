from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts, campaign_presets
from scripts.round_dagger import run_validation_20m_h200 as runner


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Bundle:
    def __init__(self) -> None:
        self._mapping = {
            "schema": "test.validation.bundle.v1",
            "campaign_id": "mini-validation",
            "macro_round": 0,
            "batch_size": 1,
            "max_steps": 20,
            "save_every": 10,
            "heldout_eval_interval": 10,
        }

    @property
    def sha256(self) -> str:
        return artifacts.canonical_json_sha256(self._mapping)

    def to_mapping(self) -> dict[str, object]:
        return dict(self._mapping)


class Validation20mH200RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "workspace/projects/roborender"
        self.source = self.project / "src/fast_roborender"
        self.gr00t = self.source / "GR00T-VisualSim2Real"
        self.python = self.project / "envs/h200_learner/bin/python"
        self.parent = self.project / "checkpoints/bootstrap_v0/checkpoint.pt"
        self.spec = Path(f"{self.parent}.policy_spec.json")
        self.validation_root = self.project / "validation/validation_20m"
        for directory in (
            self.gr00t,
            self.python.parent,
            self.parent.parent,
            self.validation_root,
            self.source / "scripts/round_dagger",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.python.write_bytes(b"#!/bin/sh\n")
        self.python.chmod(0o755)
        self.parent.write_bytes(b"fixed parent checkpoint")
        self.spec.write_text("{}\n", encoding="utf-8")
        (self.source / "scripts/round_dagger/train_learner.py").write_text(
            "# test learner\n", encoding="utf-8"
        )
        self.parent_sha = artifacts.sha256_file(self.parent)
        patcher = mock.patch.multiple(
            runner,
            REMOTE_PROJECT_ROOT=self.project,
            VALIDATION_OUTPUT_ROOT=self.validation_root,
            SOURCE_ROOT=self.source,
            GR00T_ROOT=self.gr00t,
            H200_PYTHON=self.python,
            PARENT_CHECKPOINT=self.parent,
            PARENT_POLICY_SPEC=self.spec,
            PARENT_CHECKPOINT_SHA256=self.parent_sha,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.handoff = self.project / "handoff/mini-validation"
        self.handoff.mkdir(parents=True)
        self.training_path = self.handoff / "training_snapshot.json"
        self.heldout_path = self.handoff / "heldout_snapshot.json"
        self.training_path.write_text("{}\n", encoding="utf-8")
        self.heldout_path.write_text("{}\n", encoding="utf-8")
        self.training_path.chmod(0o444)
        self.heldout_path.chmod(0o444)
        self.training_file_sha = artifacts.sha256_file(self.training_path)
        self.heldout_file_sha = artifacts.sha256_file(self.heldout_path)
        self.training_semantic_sha = _digest("training semantic")
        self.heldout_semantic_sha = _digest("heldout semantic")
        self.train_seal_path = self.handoff / "cohort/seal.json"
        self.eval_seal_path = self.handoff / "heldout_cohort/seal.json"
        train_shards = [
            {
                "rank": rank,
                "global_env_ids": [rank * 4 + environment],
                "transition_count": 16,
            }
            for rank in range(2)
            for environment in range(4)
        ]
        self._write_readonly(
            self.train_seal_path,
            {
                "schema": "fast_roborender.round_dagger.cohort.v1",
                "sealed": True,
                "campaign_id": "mini-validation",
                "macro_round": 0,
                "purpose": "train",
                "policy_sha256": self.parent_sha,
                "runtime_policy_sha256": runner.PARENT_RUNTIME_POLICY_SHA256,
                "beta_sha256": artifacts.canonical_json_sha256(
                    {"beta": runner.DAGGER_BETA}
                ),
                "expected_transitions": 128,
                "transition_count": 128,
                "ranks": [0, 1],
                "expected_ranks": [0, 1],
                "global_env_ids": list(range(8)),
                "expected_global_env_ids": list(range(8)),
                "shards": train_shards,
            },
        )
        self._write_readonly(
            self.eval_seal_path,
            {
                "schema": "fast_roborender.round_dagger.cohort.v1",
                "sealed": True,
                "campaign_id": "mini-validation",
                "macro_round": 0,
                "purpose": "eval",
                "policy_sha256": self.parent_sha,
                "runtime_policy_sha256": runner.PARENT_RUNTIME_POLICY_SHA256,
                "beta_sha256": artifacts.canonical_json_sha256(
                    {"beta": runner.DAGGER_BETA}
                ),
                "expected_transitions": 32,
                "transition_count": 32,
                "shards": [
                    {"rank": rank, "transition_count": 4} for rank in range(8)
                ],
            },
        )
        self.train_seal_sha = artifacts.sha256_file(self.train_seal_path)
        self.eval_seal_sha = artifacts.sha256_file(self.eval_seal_path)
        self.manifest_path = self.handoff / "portable_handoff_manifest.json"
        manifest = {
            "schema": runner.PORTABLE_HANDOFF_SCHEMA,
            "complete": True,
            "destination_root": str(self.handoff),
            "campaign_id": "mini-validation",
            "source": {},
            "destination": {
                "training_snapshot": {
                    "path": str(self.training_path),
                    "file_sha256": self.training_file_sha,
                    "snapshot_sha256": self.training_semantic_sha,
                    "transition_count": 128,
                },
                "heldout_snapshot": {
                    "path": str(self.heldout_path),
                    "file_sha256": self.heldout_file_sha,
                    "snapshot_sha256": self.heldout_semantic_sha,
                    "transition_count": 32,
                },
                "training_cohort": {
                    "path": str(self.handoff / "cohort"),
                    "seal_sha256": self.train_seal_sha,
                },
                "heldout_cohort": {
                    "path": str(self.handoff / "heldout_cohort"),
                    "seal_sha256": self.eval_seal_sha,
                },
                "replay_admission_status": "ACCEPTED",
            },
            "equivalence": {
                "source_destination_cohort_seals_exact": True,
                "source_destination_training_payload_sets_exact": True,
                "source_destination_heldout_payload_sets_exact_via_seal": True,
                "training_payload_sha256s": [],
                "heldout_payload_sha256s": [],
                "train_eval_payload_intersection": [],
            },
            "cohort_immutability": {
                "verified": True,
                "content_and_inode_metadata_unchanged": True,
            },
            "builder": {"path": "/test/builder", "sha256": _digest("builder")},
        }
        manifest["handoff_content_sha256"] = artifacts.canonical_json_sha256(
            manifest
        )
        self._write_readonly(self.manifest_path, manifest)
        self.scene_dir = self.handoff / "scene_samples"
        self.scene_dir.mkdir()
        self.scene_image = (
            self.scene_dir / "step_000000000_conditions_generated.png"
        )
        self.scene_image.write_bytes(b"test-only-scene-sample")
        self.scene_image.chmod(0o444)
        scene_core = {
            "schema": "fast_roborender.round_dagger.scene_samples.v1",
            "campaign_id": "mini-validation",
            "actor_rank": 0,
            "rollout_steps": 16,
            "view_labels": ["head_color", "hand_left", "hand_right"],
            "row_labels": ["depth", "mask", "client_anchor", "generated"],
            "sampling_policy": {"name": "test"},
            "source": {"candidate_count": 1},
            "sampled_rollout_steps": [0],
            "samples": [
                {
                    "path": self.scene_image.name,
                    "rollout_step": 0,
                    "sha256": artifacts.sha256_file(self.scene_image),
                    "size_bytes": self.scene_image.stat().st_size,
                    "width": 1252,
                    "height": 1046,
                }
            ],
        }
        self.scene_manifest = {
            **scene_core,
            "sample_set_sha256": artifacts.canonical_json_sha256(scene_core),
        }
        self._write_readonly(
            self.scene_dir / "manifest.json", self.scene_manifest
        )
        self.contract_path = self.project / "contracts/mini_learner_contract.active.json"
        self.contract = {
            "schema": runner.CONTRACT_SCHEMA,
            "preset_id": "validation_20m",
            "profile_sha256": artifacts.sha256_file(
                campaign_presets.profile_path("validation_20m")
            ),
            "campaign_id": "mini-validation",
            "macro_round": 0,
            "learner_code_sha256": _digest("sealed learner source"),
            "runtime_container_sha256": runner.RUNTIME_CONTAINER_SHA256,
            "parent": {
                "checkpoint_path": str(self.parent),
                "checkpoint_sha256": self.parent_sha,
                "policy_spec_path": str(self.spec),
                "runtime_policy_sha256": runner.PARENT_RUNTIME_POLICY_SHA256,
            },
            "handoff": {
                "manifest_file_sha256": artifacts.sha256_file(self.manifest_path),
                "training_snapshot_file_sha256": self.training_file_sha,
                "training_snapshot_sha256": self.training_semantic_sha,
                "heldout_snapshot_file_sha256": self.heldout_file_sha,
                "heldout_snapshot_sha256": self.heldout_semantic_sha,
            },
            "allocation": dict(runner._ALLOCATION),
            "optimization": dict(runner._OPTIMIZATION),
            "wandb": dict(runner._WANDB),
        }
        self._write_readonly(self.contract_path, self.contract)
        self.output = self.validation_root / "mini-validation/learner"
        self.bundle = _Bundle()

    @staticmethod
    def _write_readonly(path: Path, document: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)

    @staticmethod
    def _shards(*, prefix: str, transitions: int) -> list[dict[str, object]]:
        return [
            {
                "logical_id": f"mini-validation:r000000000:c0000:{prefix}",
                "macro_round": 0,
                "cohort_index": 0 if prefix == "train" else 1,
                "shard_id": f"rank-{rank:04d}",
                "manifest_sha256": _digest(f"{prefix} manifest {rank}"),
                "payload_sha256": _digest(f"{prefix} payload {rank}"),
                "transition_count": transitions,
            }
            for rank in range(8)
        ]

    def _snapshots(self) -> tuple[dict[str, object], dict[str, object]]:
        training = {
            "snapshot_sha256": self.training_semantic_sha,
            "campaign_id": "mini-validation",
            "through_macro_round": 0,
            "expected_train_cohorts": 1,
            "cohort_count": 1,
            "counts_by_round": {"0": 128},
            "counts_by_round_cohort": {"0:0": 128},
            "cohorts": [
                {
                    "logical_id": "mini-validation:r000000000:c0000:train",
                    "macro_round": 0,
                    "cohort_index": 0,
                    "cohort_manifest_path": str(self.train_seal_path),
                    "manifest_sha256": self.train_seal_sha,
                }
            ],
            "transition_count": 128,
            "view_contract": dict(runner._VIEW_CONTRACT),
            "action_contract": dict(runner._ACTION_CONTRACT),
            "resolved_shards": self._shards(prefix="train", transitions=16),
        }
        heldout = {
            "snapshot_sha256": self.heldout_semantic_sha,
            "campaign_id": "mini-validation",
            "cohort_count": 1,
            "cohorts": [
                {
                    "logical_id": "mini-validation:r000000000:c0001:eval",
                    "macro_round": 0,
                    "cohort_index": 1,
                    "cohort_manifest_path": str(self.eval_seal_path),
                    "manifest_sha256": self.eval_seal_sha,
                }
            ],
            "transition_count": 32,
            "view_contract": dict(runner._VIEW_CONTRACT),
            "action_contract": dict(runner._ACTION_CONTRACT),
            "resolved_shards": self._shards(prefix="eval", transitions=4),
        }
        return training, heldout

    def _build(self, training=None, heldout=None) -> runner.RunnerPlan:
        default_training, default_heldout = self._snapshots()
        with (
            mock.patch.object(
                runner.replay,
                "load_snapshot",
                return_value=default_training if training is None else training,
            ),
            mock.patch.object(
                runner.replay,
                "load_heldout_snapshot",
                return_value=default_heldout if heldout is None else heldout,
            ),
            mock.patch.object(
                runner, "_prepare_gr00t_bundle", return_value=self.bundle
            ),
            mock.patch.object(runner, "_verify_sealed_source"),
        ):
            return runner.build_plan(
                handoff_root=self.handoff,
                output_dir=self.output,
                contract_path=self.contract_path,
                wandb_run_name="mini_prod_validation_1",
            )

    def test_dry_plan_is_exact_real_gr00t_world8_and_read_only(self) -> None:
        before = set(self.project.rglob("*"))
        plan = self._build()
        after = set(self.project.rglob("*"))
        self.assertEqual(before, after)
        command = list(plan.command)
        self.assertEqual(
            command[:4],
            [str(runner.TIMEOUT_BINARY), "--signal=TERM", "--kill-after=30s", "1170s"],
        )
        self.assertEqual(
            command[4:7], [str(self.python), "-m", "torch.distributed.run"]
        )
        self.assertEqual(command[command.index("--nproc-per-node") + 1], "8")
        self.assertEqual(command[command.index("--max-steps") + 1], "20")
        self.assertEqual(command[command.index("--save-every") + 1], "10")
        self.assertEqual(command[command.index("--heldout-eval-interval") + 1], "10")
        self.assertIn("round_dagger.gr00t_policy:create_vision_actor_obj_pred", command)
        self.assertIn("--wandb-resume-never", command)
        self.assertIn("--dagger-wandb-dashboard", command)
        self.assertEqual(
            command[command.index("--scene-sample-manifest") + 1],
            str(self.scene_dir / "manifest.json"),
        )
        self.assertNotIn("--resume-from", command)
        self.assertEqual(plan.environment["WANDB_NAME"], "mini_prod_validation_1")
        self.assertEqual(plan.environment["WANDB_PROJECT"], runner.WANDB_PROJECT)
        self.assertEqual(plan.environment["WANDB_ENTITY"], "nvidia")
        self.assertEqual(plan.environment["WANDB_RESUME"], "never")
        self.assertRegex(plan.learner_identity_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            plan.wandb_run_id,
            f"rrd-m000000-id{plan.learner_identity_sha256}",
        )
        report = plan.dry_run_report()
        self.assertEqual(report["required_heldout_frontiers"], [0, 10, 20])
        self.assertFalse(self.output.exists())

    def test_wrong_learner_shape_is_rejected_before_bundle_preparation(self) -> None:
        contract = json.loads(json.dumps(self.contract))
        contract["optimization"]["optimizer_steps"] = 19
        self.contract_path.chmod(0o644)
        self._write_readonly(self.contract_path, contract)
        with self.assertRaisesRegex(
            runner.Validation20mRunnerError, "optimization mismatch"
        ):
            self._build()

    def test_wrong_replay_count_or_rank_partition_is_rejected(self) -> None:
        training, heldout = self._snapshots()
        training["transition_count"] = 127
        with self.assertRaisesRegex(
            runner.Validation20mRunnerError, "128 transitions"
        ):
            self._build(training=training, heldout=heldout)
        training, heldout = self._snapshots()
        training["resolved_shards"] = training["resolved_shards"][:-1]
        with self.assertRaisesRegex(
            runner.Validation20mRunnerError, "exactly eight shards"
        ):
            self._build(training=training, heldout=heldout)

    def test_train_eval_payload_overlap_is_rejected(self) -> None:
        training, heldout = self._snapshots()
        heldout["resolved_shards"][0]["payload_sha256"] = training[
            "resolved_shards"
        ][0]["payload_sha256"]
        with self.assertRaisesRegex(runner.Validation20mRunnerError, "overlap"):
            self._build(training=training, heldout=heldout)

    def test_paths_and_operator_name_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            runner.Validation20mRunnerError, "operator run name"
        ):
            runner.build_plan(
                handoff_root=self.handoff,
                output_dir=self.output,
                contract_path=self.contract_path,
                wandb_run_name="unsafe name/with slash",
            )
        with self.assertRaisesRegex(runner.Validation20mRunnerError, "absolute"):
            runner.build_plan(
                handoff_root=Path("relative/handoff"),
                output_dir=self.output,
                contract_path=self.contract_path,
                wandb_run_name="safe_name",
            )

    def test_execute_requires_h200_and_propagates_exact_wandb_name(self) -> None:
        plan = self._build()

        class ExecCalled(RuntimeError):
            pass

        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda _index: "NVIDIA H200",
        )
        fake_torch = types.SimpleNamespace(cuda=cuda)
        with (
            mock.patch.object(runner, "H200_PYTHON", Path(sys.executable)),
            mock.patch.dict(
                os.environ,
                {
                    "WANDB_API_KEY": "test-only-not-a-real-key",
                    "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": (
                        runner.RUNTIME_CONTAINER_SHA256
                    ),
                },
                clear=False,
            ),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch.object(
                runner.os, "execvpe", side_effect=ExecCalled("exec boundary")
            ) as execute,
            mock.patch.object(runner.time, "monotonic", return_value=100.0),
            mock.patch.object(runner.signal, "alarm"),
        ):
            with self.assertRaises(ExecCalled):
                runner.execute(plan, deadline_monotonic=1300.0)
        command, environment = execute.call_args.args[1:]
        self.assertEqual(command[3], "1170s")
        self.assertEqual(environment["WANDB_NAME"], "mini_prod_validation_1")
        self.assertEqual(environment["WANDB_RESUME"], "never")
        self.assertTrue(plan.bundle_path.is_file())

    def test_execute_rejects_eight_non_h200_gpus_before_writing(self) -> None:
        plan = self._build()
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
            get_device_name=lambda _index: "NVIDIA A100-SXM4-80GB",
        )
        with (
            mock.patch.object(runner, "H200_PYTHON", Path(sys.executable)),
            mock.patch.dict(
                os.environ,
                {
                    "WANDB_API_KEY": "test-only-not-a-real-key",
                    "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": (
                        runner.RUNTIME_CONTAINER_SHA256
                    ),
                },
                clear=False,
            ),
            mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}),
        ):
            with self.assertRaisesRegex(runner.Validation20mRunnerError, "H200"):
                runner.execute(plan, deadline_monotonic=1300.0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
