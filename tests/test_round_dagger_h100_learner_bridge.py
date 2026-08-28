"""Contract and dry-run tests for the production H100 learner bridge."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts, replay
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from round_dagger.gr00t_policy import (
    BASE_MODULE_TARGET,
    POLICY_TARGET,
    build_portable_policy_spec,
)

try:
    import torch
except ImportError:  # The lightweight control-plane environment has no torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
PREPARER_PATH = ROOT / "scripts/round_dagger/prepare_gr00t_learner_bundle.py"
LAUNCHER_PATH = ROOT / "scripts/round_dagger/run_h100_learner.sh"
HOMIE_CONTROLLER_SHA256 = {"walk": "7" * 64, "stand": "8" * 64}


def _load_preparer():
    specification = importlib.util.spec_from_file_location(
        "round_dagger_test_bundle_preparer", PREPARER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_train_entrypoint():
    path = ROOT / "scripts/round_dagger/train_learner.py"
    specification = importlib.util.spec_from_file_location(
        "round_dagger_test_train_entrypoint", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _policy_spec(
    *, checkpoint_sha256: str = "1" * 64, policy_state_sha256: str = "2" * 64
) -> dict:
    policy_type = type("VisionActorObjPred", (), {})
    policy_type.__module__ = "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"
    policy = policy_type()
    policy.num_actions = 11
    policy.obj_pred_mlp = types.SimpleNamespace(output_dim=3)
    actor_config = {
        "_target_": POLICY_TARGET,
        "running_mean_std": False,
        "backbone": {
            "vision_module": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["vision_obs"],
                    "output_dim": ["vision_feature_dim"],
                    "layer_config": {
                        "type": "ResNet",
                        "resnet_type": "resnet18",
                        "pretrained": True,
                        "trainable": True,
                    },
                },
            },
            "mlp_module": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["actor_obs", "vision_feature_dim"],
                    "output_dim": [-1],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [512, 256, 128],
                        "activation": "SiLU",
                    },
                },
            },
            "obj_pred_mlp": {
                "_target_": BASE_MODULE_TARGET,
                "process_output_dim": True,
                "module_config_dict": {
                    "input_dim": ["vision_feature_dim"],
                    "output_dim": [3],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [256, 128, 64],
                        "activation": "SiLU",
                    },
                },
            },
        },
    }
    return build_portable_policy_spec(
        policy=policy,
        actor_config=actor_config,
        algo_config={
            "init_noise_std": 0.3,
            "freeze_noise_std": True,
            "clamp_noise_std": False,
        },
        module_dim_dict={"vision_feature_dim": 128},
        env_config=types.SimpleNamespace(
            robot=types.SimpleNamespace(
                algo_obs_dim_dict={
                    "actor_obs": 113,
                    "vision_obs": 324 * 192 * 3,
                },
                actions_dim=31,
            )
        ),
        policy_views=["head_color", "hand_left", "hand_right"],
        policy_view_layout="vertical",
        per_view_resolution=[108, 192],
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        homie_action_dim=15,
        replay_action_dim=26,
        checkpoint_sha256=checkpoint_sha256,
        policy_state_sha256=policy_state_sha256,
        homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
    )


def _bundle_kwargs(root: Path) -> dict:
    parent = root / "parent.pt"
    parent.write_bytes(b"strict-parent-checkpoint")
    parent_sha256 = artifacts.sha256_file(parent)
    spec = _policy_spec(checkpoint_sha256=parent_sha256)
    spec_path = root / "base.policy_spec.json"
    artifacts.publish_immutable_json(spec_path, spec)
    return {
        "base_policy_spec": spec_path,
        "parent_checkpoint": parent,
        "expected_parent_checkpoint_sha256": parent_sha256,
        "runtime_policy_sha256": spec["runtime_policy_sha256"],
        "campaign_id": "production-test",
        "macro_round": 0,
        "code_sha256": "4" * 64,
        "batch_size": 8,
        "max_steps": 11,
        "save_every": 3,
        "learning_rate": 2.5e-5,
        "weight_decay": 0.01,
        "seed": 17,
        "partition_seed": 23,
    }


def _rollout_arrays(global_env_id: int) -> dict[str, np.ndarray]:
    expert = np.full((1, 1, 26), global_env_id / 10.0, dtype=np.float32)
    student = np.zeros_like(expert)
    mask = np.ones((1, 1), dtype=np.bool_)
    return {
        "policy_rgb": np.full((1, 1, 324, 192, 3), global_env_id, dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": expert.copy(),
        "expert_mask": mask,
        "dones": np.zeros((1, 1), dtype=np.bool_),
        "observation_is_raytraced": np.ones((1, 1), dtype=np.bool_),
        "global_env_ids": np.asarray([global_env_id], dtype=np.int64),
        "episode_ids": np.asarray([[global_env_id]], dtype=np.int64),
        "episode_steps": np.zeros((1, 1), dtype=np.int64),
        "obs__actor_obs": np.full((1, 1, 113), global_env_id / 100.0, dtype=np.float32),
    }


def _production_snapshot(
    root: Path, *, parent_sha: str, runtime_sha: str
) -> tuple[Path, dict]:
    provenance = {
        "campaign_id": "production-test",
        "macro_round": 0,
        "purpose": "train",
        "policy_sha256": parent_sha,
        "runtime_policy_sha256": runtime_sha,
        "teacher_sha256": "5" * 64,
        "renderer_sha256": "6" * 64,
        "beta_sha256": "7" * 64,
        "config_sha256": "8" * 64,
    }
    view_contract = {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": [324, 192, 3],
        "views": ["head_color", "hand_left", "hand_right"],
    }
    action_contract = {"dtype": "float32", "shape": [26]}
    cohort_paths = []
    for cohort_index in range(4):
        cohort = root / f"cohort-{cohort_index}"
        cohort_provenance = dict(provenance, cohort_index=cohort_index)
        artifacts.create_cohort(
            cohort,
            expected_ranks=[0],
            expected_global_env_ids=[cohort_index],
            expected_transitions=1,
            view_contract=view_contract,
            action_contract=action_contract,
            **cohort_provenance,
        )
        artifacts.publish_rollout_shard(
            cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_rollout_arrays(cohort_index),
            **cohort_provenance,
        )
        artifacts.seal_cohort(cohort)
        cohort_paths.append(cohort)
    snapshot_path = root / "snapshot.json"
    with replay.ReplayLedger(root / "replay.sqlite3") as ledger:
        for cohort in cohort_paths:
            ledger.ingest_cohort(cohort)
        snapshot = ledger.build_snapshot(
            "production-test",
            through_macro_round=0,
            expected_train_cohorts=4,
            output=snapshot_path,
        )
    return snapshot_path, snapshot


def _heldout_snapshot(
    root: Path, *, parent_sha: str, runtime_sha: str
) -> tuple[Path, dict]:
    provenance = {
        "campaign_id": "production-test",
        "macro_round": 0,
        "cohort_index": 99,
        "purpose": "eval",
        "policy_sha256": parent_sha,
        "runtime_policy_sha256": runtime_sha,
        "teacher_sha256": "5" * 64,
        "renderer_sha256": "6" * 64,
        "beta_sha256": "7" * 64,
        "config_sha256": "8" * 64,
    }
    cohort = root / "heldout/cohort-99"
    artifacts.create_cohort(
        cohort,
        expected_ranks=list(range(8)),
        expected_global_env_ids=list(range(99, 107)),
        expected_transitions=8,
        view_contract={
            "dtype": "uint8",
            "layout": "vertical",
            "shape": [324, 192, 3],
            "views": ["head_color", "hand_left", "hand_right"],
        },
        action_contract={"dtype": "float32", "shape": [26]},
        **provenance,
    )
    for rank, global_env_id in enumerate(range(99, 107)):
        artifacts.publish_rollout_shard(
            cohort,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=_rollout_arrays(global_env_id),
            **provenance,
        )
    artifacts.seal_cohort(cohort)
    snapshot_path = root / "heldout/snapshot.json"
    snapshot = replay.build_heldout_snapshot([cohort], output=snapshot_path)
    return snapshot_path, snapshot


@unittest.skipIf(torch is None, "H100 learner bridge tests require PyTorch")
class ImmutableLearnerBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preparer = _load_preparer()
        self.kwargs = _bundle_kwargs(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_derivation_binds_current_parent_runtime_and_optimizer_contract(self):
        source = artifacts.load_strict_json(self.kwargs["base_policy_spec"])
        bundle = self.preparer.prepare_bundle(**self.kwargs)
        mapping = bundle.to_mapping()
        parent_sha = self.kwargs["expected_parent_checkpoint_sha256"]
        self.assertEqual(bundle.model_lineage_sha256, parent_sha)
        self.assertEqual(bundle.initial_checkpoint_sha256, parent_sha)
        self.assertEqual(mapping["model_config"]["checkpoint_sha256"], parent_sha)
        self.assertEqual(
            mapping["model_config"]["runtime_policy_sha256"],
            self.kwargs["runtime_policy_sha256"],
        )
        self.assertEqual(mapping["train_action_slice"], [0, 11])
        self.assertEqual((bundle.max_steps, bundle.save_every), (11, 3))
        self.assertEqual((bundle.learning_rate, bundle.weight_decay), (2.5e-5, 0.01))
        self.assertEqual(source["checkpoint_sha256"], parent_sha)
        self.assertEqual(
            artifacts.load_strict_json(self.kwargs["base_policy_spec"])[
                "checkpoint_sha256"
            ],
            parent_sha,
        )

    def test_parent_digest_and_symlink_are_rejected(self):
        bad_digest = dict(self.kwargs, expected_parent_checkpoint_sha256="f" * 64)
        with self.assertRaisesRegex(
            self.preparer.BundlePreparationError, "checkpoint SHA-256 mismatch"
        ):
            self.preparer.prepare_bundle(**bad_digest)
        link = self.root / "parent-link.pt"
        link.symlink_to(self.kwargs["parent_checkpoint"])
        bad_link = dict(self.kwargs, parent_checkpoint=link)
        with self.assertRaisesRegex(
            self.preparer.BundlePreparationError, "non-symlink"
        ):
            self.preparer.prepare_bundle(**bad_link)

    def test_active_spec_cannot_be_relabelled_with_another_runtime(self):
        with self.assertRaisesRegex(
            self.preparer.BundlePreparationError,
            "active policy spec runtime identity mismatch",
        ):
            self.preparer.prepare_bundle(
                **dict(self.kwargs, runtime_policy_sha256="f" * 64)
            )

    def test_initial_checkpoint_state_must_match_portable_spec(self):
        trainer = _load_train_entrypoint()
        model = torch.nn.Linear(2, 1)
        checkpoint = self.root / "initial.pt"
        torch.save({"policy_state_dict": model.state_dict()}, checkpoint)
        from round_dagger.actor import runtime_state_dict_sha256

        trainer._load_initial_checkpoint(
            model,
            checkpoint,
            expected_policy_state_sha256=runtime_state_dict_sha256(
                model.state_dict()
            ),
        )
        with self.assertRaisesRegex(
            trainer.LearnerError, "differs from portable policy spec"
        ):
            trainer._load_initial_checkpoint(
                model,
                checkpoint,
                expected_policy_state_sha256="f" * 64,
            )

    def test_publication_is_idempotent_but_configuration_drift_conflicts(self):
        bundle = self.preparer.prepare_bundle(**self.kwargs)
        output = self.root / "learner_bundle.json"
        first = artifacts.publish_immutable_json(output, bundle.to_mapping())
        second = artifacts.publish_immutable_json(output, bundle.to_mapping())
        self.assertEqual(first, second)
        self.assertEqual(first, bundle.sha256)
        changed = self.preparer.prepare_bundle(
            **dict(self.kwargs, learning_rate=5.0e-5)
        )
        self.assertNotEqual(changed.sha256, bundle.sha256)
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "immutable artifact conflict"
        ):
            artifacts.publish_immutable_json(output, changed.to_mapping())

    def test_train_entrypoint_cannot_override_immutable_step_settings(self):
        bundle = self.preparer.prepare_bundle(**self.kwargs)
        bundle_path = self.root / "learner_bundle.json"
        artifacts.publish_immutable_json(bundle_path, bundle.to_mapping())
        trainer = _load_train_entrypoint()
        common = [
            "--bundle",
            str(bundle_path),
            "--snapshot",
            str(self.root / "unused-snapshot.json"),
            "--output-dir",
            str(self.root / "unused-output"),
            "--model-factory",
            "unused:factory",
            "--initial-checkpoint",
            str(self.kwargs["parent_checkpoint"]),
            "--no-publish-result",
        ]
        publish_enabled = common[:-1]
        with self.assertRaisesRegex(
            trainer.LearnerError, "requires --no-publish-result"
        ):
            trainer.main(
                [*publish_enabled, "--validation-stop-at-step", "5"]
            )
        with self.assertRaisesRegex(
            trainer.LearnerError, "max-steps differs from the immutable"
        ):
            trainer.main([*common, "--max-steps", "12"])
        with self.assertRaisesRegex(
            trainer.LearnerError, "save-every differs from the immutable"
        ):
            trainer.main([*common, "--save-every", "4"])

        captured = {}

        def train_stub(_model, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                completed_steps=11,
                target_steps=11,
                validation_stopped=False,
                restart_checkpoint=str(self.root / "restart.pt"),
                replay_snapshot_sha256="a" * 64,
            )

        with (
            mock.patch.object(
                trainer,
                "_symbol",
                return_value=lambda _mapping: torch.nn.Linear(1, 1),
            ),
            mock.patch.object(trainer, "_load_initial_checkpoint"),
            mock.patch.object(trainer, "train_from_replay", side_effect=train_stub),
        ):
            self.assertEqual(trainer.main(common), 0)
        self.assertEqual(captured["max_steps"], 11)
        self.assertEqual(captured["save_every"], 3)

        captured.clear()
        with (
            mock.patch.object(
                trainer,
                "_symbol",
                return_value=lambda _mapping: torch.nn.Linear(1, 1),
            ),
            mock.patch.object(trainer, "_load_initial_checkpoint"),
            mock.patch.object(trainer, "train_from_replay", side_effect=train_stub),
        ):
            self.assertEqual(
                trainer.main([*common, "--validation-stop-at-step", "5"]), 0
            )
        self.assertEqual(captured["max_steps"], 11)
        self.assertEqual(captured["validation_stop_at_step"], 5)

    def test_train_entrypoint_requires_the_bundle_pinned_initial_checkpoint(self):
        bundle = self.preparer.prepare_bundle(**self.kwargs)
        bundle_path = self.root / "learner_bundle.json"
        artifacts.publish_immutable_json(bundle_path, bundle.to_mapping())
        trainer = _load_train_entrypoint()
        with self.assertRaisesRegex(
            trainer.LearnerError, "--initial-checkpoint is required"
        ):
            trainer.main(
                [
                    "--bundle",
                    str(bundle_path),
                    "--snapshot",
                    str(self.root / "unused-snapshot.json"),
                    "--output-dir",
                    str(self.root / "unused-output"),
                    "--model-factory",
                    "unused:factory",
                    "--no-publish-result",
                ]
            )

    def test_train_entrypoint_requires_strict_wandb_resume_for_pinned_restart(self):
        bundle = self.preparer.prepare_bundle(**self.kwargs)
        bundle_path = self.root / "learner_bundle.json"
        artifacts.publish_immutable_json(bundle_path, bundle.to_mapping())
        trainer = _load_train_entrypoint()
        telemetry = types.SimpleNamespace(close=lambda: None)
        trained = types.SimpleNamespace(
            completed_steps=11,
            target_steps=11,
            validation_stopped=False,
            restart_checkpoint=str(self.root / "restart.pt"),
            replay_snapshot_sha256="a" * 64,
        )
        arguments = [
            "--bundle",
            str(bundle_path),
            "--snapshot",
            str(self.root / "unused-snapshot.json"),
            "--output-dir",
            str(self.root / "unused-output"),
            "--model-factory",
            "unused:factory",
            "--initial-checkpoint",
            str(self.kwargs["parent_checkpoint"]),
            "--resume-from",
            str(self.root / "restart.pt"),
            "--wandb-project",
            "project",
            "--wandb-run-id",
            "durable-run-id",
            "--no-publish-result",
        ]
        with (
            mock.patch.object(
                trainer,
                "_symbol",
                return_value=lambda _mapping: torch.nn.Linear(1, 1),
            ),
            mock.patch.object(trainer, "_load_initial_checkpoint"),
            mock.patch.object(
                trainer,
                "load_snapshot",
                return_value={
                    "campaign_id": "production-test",
                    "through_macro_round": 0,
                    "file_sha256": "a" * 64,
                    "snapshot_sha256": "b" * 64,
                },
            ),
            mock.patch.object(trainer, "train_from_replay", return_value=trained),
            mock.patch.object(
                trainer, "JsonlTelemetry", return_value=telemetry
            ) as telemetry_factory,
        ):
            self.assertEqual(trainer.main(arguments), 0)
        self.assertTrue(telemetry_factory.call_args.kwargs["wandb_resume_must"])
        self.assertEqual(
            telemetry_factory.call_args.kwargs["wandb_config"][
                "replay_snapshot_file_sha256"
            ],
            "a" * 64,
        )
        self.assertEqual(
            telemetry_factory.call_args.kwargs["wandb_config"][
                "replay_snapshot_sha256"
            ],
            "b" * 64,
        )


@unittest.skipIf(torch is None, "H100 learner bridge tests require PyTorch")
class H100LauncherDryRunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preparer = _load_preparer()
        self.kwargs = _bundle_kwargs(self.root)
        self.bundle = self.preparer.prepare_bundle(**self.kwargs)
        self.snapshot_path, self.snapshot = _production_snapshot(
            self.root,
            parent_sha=self.kwargs["expected_parent_checkpoint_sha256"],
            runtime_sha=self.kwargs["runtime_policy_sha256"],
        )
        self.output = self.root / "learner"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _bind_identity(environment: dict[str, str]) -> dict[str, str]:
        identity = learner_identity_sha256(
            campaign_id=environment["ROUND_DAGGER_CAMPAIGN_ID"],
            macro_round=int(environment["ROUND_DAGGER_MACRO_ROUND"]),
            learner_code_sha256=environment["ROUND_DAGGER_LEARNER_CODE_SHA256"],
            learner_config_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
            ],
            replay_snapshot_file_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ],
            replay_snapshot_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"
            ],
            parent_checkpoint_sha256=environment["ROUND_DAGGER_POLICY_SHA256"],
            parent_runtime_policy_sha256=environment[
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256"
            ],
            runtime_container_sha256=environment[
                "ROUND_DAGGER_REMOTE_CONTAINER_SHA256"
            ],
            learner_contract_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
            ],
        )
        environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"] = identity
        environment["ROUND_DAGGER_WANDB_RUN_ID"] = learner_wandb_run_id(
            campaign_id=environment["ROUND_DAGGER_CAMPAIGN_ID"],
            macro_round=int(environment["ROUND_DAGGER_MACRO_ROUND"]),
            learner_identity_sha256=identity,
        )
        return environment

    def _environment(self) -> dict[str, str]:
        result_path = self.output / "learner_result_round_000000.json"
        environment = {
            **os.environ,
            "DRY_RUN": "1",
            "ROUND_DAGGER_H100_PYTHON": sys.executable,
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.root),
            "ROUND_DAGGER_CAMPAIGN_ID": "production-test",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(self.kwargs["parent_checkpoint"]),
            "ROUND_DAGGER_POLICY_SHA256": self.kwargs[
                "expected_parent_checkpoint_sha256"
            ],
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": self.kwargs["runtime_policy_sha256"],
            "ROUND_DAGGER_REPLAY_SNAPSHOT": str(self.snapshot_path),
            "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": self.snapshot["file_sha256"],
            "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": self.snapshot["snapshot_sha256"],
            "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(self.output),
            "ROUND_DAGGER_LEARNER_RESULT": str(result_path),
            "ROUND_DAGGER_LEARNER_CODE_SHA256": self.kwargs["code_sha256"],
            "ROUND_DAGGER_LEARNER_CONFIG_SHA256": self.bundle.sha256,
            "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": "5" * 64,
            "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": "6" * 64,
            "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": self.kwargs[
                "expected_parent_checkpoint_sha256"
            ],
            "ROUND_DAGGER_BASE_POLICY_SPEC": str(self.kwargs["base_policy_spec"]),
            "ROUND_DAGGER_LEARNER_BATCH_SIZE": str(self.kwargs["batch_size"]),
            "ROUND_DAGGER_LEARNER_MAX_STEPS": str(self.kwargs["max_steps"]),
            "ROUND_DAGGER_LEARNER_SAVE_EVERY": str(self.kwargs["save_every"]),
            "ROUND_DAGGER_LEARNER_LEARNING_RATE": str(self.kwargs["learning_rate"]),
            "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": str(self.kwargs["weight_decay"]),
            "ROUND_DAGGER_LEARNER_SEED": str(self.kwargs["seed"]),
            "ROUND_DAGGER_LEARNER_PARTITION_SEED": str(self.kwargs["partition_seed"]),
            "ROUND_DAGGER_LEARNER_NNODES": "1",
            "ROUND_DAGGER_LEARNER_NPROC_PER_NODE": "1",
            "ROUND_DAGGER_LEARNER_NODE_RANK": "0",
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
            "ROUND_DAGGER_COHORT_COUNT": "4",
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "131072",
            "ROUND_DAGGER_WANDB_PROJECT": "bridge-test",
            "ROUND_DAGGER_WANDB_ENTITY": "entity-test",
            "WANDB_API_KEY": "must-never-appear-in-launch-output",
        }
        return self._bind_identity(environment)

    def _launch(self, environment: dict[str, str] | None = None):
        return subprocess.run(
            ["bash", str(LAUNCHER_PATH)],
            cwd=ROOT,
            env=environment or self._environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )

    def test_dry_run_validates_replay_and_prints_exact_torchrun_without_secrets(self):
        completed = self._launch()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("H100_LEARNER_DRY_RUN_OK", completed.stdout)
        self.assertIn("torch.distributed.run", completed.stdout)
        self.assertIn(
            "round_dagger.gr00t_policy:create_vision_actor_obj_pred",
            completed.stdout,
        )
        self.assertIn(
            "round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory",
            completed.stdout,
        )
        self.assertIn("--device cuda", completed.stdout)
        self.assertIn("cuda_determinism=:4096:8", completed.stdout)
        self.assertNotIn("must-never-appear", completed.stdout + completed.stderr)
        self.assertFalse(self.output.exists())

    def test_conflicting_cublas_determinism_contract_is_rejected(self):
        environment = self._environment()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        completed = self._launch(environment)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "CUBLAS_WORKSPACE_CONFIG must be exactly :4096:8",
            completed.stderr,
        )

    def test_dry_run_plumbs_exact_heldout_snapshot_and_100_step_cadence(self):
        heldout_path, heldout = _heldout_snapshot(
            self.root,
            parent_sha=self.kwargs["expected_parent_checkpoint_sha256"],
            runtime_sha=self.kwargs["runtime_policy_sha256"],
        )
        heldout_bundle = self.preparer.prepare_bundle(
            **self.kwargs,
            heldout_snapshot=heldout_path,
            heldout_eval_interval=100,
        )
        environment = self._environment()
        environment.update(
            {
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256": heldout_bundle.sha256,
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": str(heldout_path),
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256": heldout[
                    "file_sha256"
                ],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256": heldout[
                    "snapshot_sha256"
                ],
                "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL": "100",
            }
        )
        self._bind_identity(environment)
        completed = self._launch(environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("H100_HELDOUT_PARTITION", completed.stderr)
        self.assertIn("--heldout-snapshot", completed.stdout)
        self.assertIn("--heldout-eval-interval 100", completed.stdout)
        self.assertIn("heldout_eval=1 cadence=100", completed.stdout)

        incomplete = self._environment()
        incomplete["ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT"] = str(heldout_path)
        rejected = self._launch(incomplete)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("supplied all-or-none", rejected.stderr)

    def test_dry_run_auto_resumes_only_a_matching_restart(self):
        self.output.mkdir()
        bundle_path = self.output / "learner_bundle_round_000000.json"
        artifacts.publish_immutable_json(bundle_path, self.bundle.to_mapping())
        restart = self.output / "restart.pt"
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
                "campaign_id": "production-test",
                "macro_round": 0,
                "model_lineage_sha256": self.bundle.model_lineage_sha256,
                "code_sha256": self.bundle.code_sha256,
                "learner_config_sha256": self.bundle.sha256,
                "learner_identity_sha256": self._environment()[
                    "ROUND_DAGGER_LEARNER_IDENTITY_SHA256"
                ],
                "runtime_container_sha256": "6" * 64,
                "learner_contract_sha256": "5" * 64,
                "replay_snapshot_sha256": self.snapshot["snapshot_sha256"],
                "replay_snapshot_file_sha256": self.snapshot["file_sha256"],
                "world_size": 1,
                "homie_controller_sha256": HOMIE_CONTROLLER_SHA256,
                "completed_steps": 3,
                "initial_loss": 1.25,
                "rng_states": [{}],
                "model_state_dict": {},
                "optimizer_state_dict": {},
            },
            restart,
        )
        completed = self._launch()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--resume-from", completed.stdout)
        self.assertIn(str(restart), completed.stdout)
        self.assertIn(f"resume={restart}", completed.stdout)

        document = torch.load(restart, map_location="cpu", weights_only=False)
        document.pop("initial_loss")
        torch.save(document, restart)
        rejected = self._launch()
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("restart initial_loss is invalid", rejected.stderr)

        document["initial_loss"] = 1.25
        document["world_size"] = 2
        torch.save(document, restart)
        rejected = self._launch()
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("restart checkpoint lineage mismatch", rejected.stderr)

    def test_current_round_runtime_provenance_drift_is_rejected(self):
        environment = self._environment()
        changed_spec = _policy_spec(
            checkpoint_sha256=self.kwargs["expected_parent_checkpoint_sha256"],
            policy_state_sha256="9" * 64,
        )
        changed_spec_path = self.root / "changed.policy_spec.json"
        artifacts.publish_immutable_json(changed_spec_path, changed_spec)
        changed_runtime = changed_spec["runtime_policy_sha256"]
        environment["ROUND_DAGGER_RUNTIME_POLICY_SHA256"] = changed_runtime
        environment["ROUND_DAGGER_BASE_POLICY_SPEC"] = str(changed_spec_path)
        # Recompute the immutable bundle so the failure is specifically at the
        # replay provenance gate rather than at bundle digest validation.
        changed = self.preparer.prepare_bundle(
            **dict(
                self.kwargs,
                base_policy_spec=changed_spec_path,
                runtime_policy_sha256=changed_runtime,
            )
        )
        environment["ROUND_DAGGER_LEARNER_CONFIG_SHA256"] = changed.sha256
        self._bind_identity(environment)
        completed = self._launch(environment)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "current-round replay provenance does not match active policy",
            completed.stderr,
        )

    def test_launcher_never_embeds_wandb_credentials(self):
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("WANDB_API_KEY", source)
        self.assertNotIn("set -x", source)


if __name__ == "__main__":
    unittest.main()
