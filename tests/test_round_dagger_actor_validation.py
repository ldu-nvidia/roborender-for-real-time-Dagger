"""Fail-closed dry-run tests for the bounded real actor validation rung."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from typing import Callable

from round_dagger.artifacts import create_cohort, publish_immutable_json, sha256_file
from round_dagger.gr00t_policy import (
    BASE_MODULE_TARGET,
    POLICY_TARGET,
    build_portable_policy_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/round_dagger/run_actor_validation.sh"
RUNTIME_SHA = "2" * 64
RENDERER_SHA = "3" * 64
BETA_SHA = "4" * 64
CONFIG_SHA = "5" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_FakePolicy = type("VisionActorObjPred", (), {})
_FakePolicy.__module__ = "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"


def _portable_spec(checkpoint_sha256: str) -> dict:
    policy = _FakePolicy()
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
        runtime_policy_sha256=RUNTIME_SHA,
    )


class ActorValidationDryRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.marker = self.root / "launched"
        self.fake_python = self._executable("gr00t-python")
        self.fake_accelerate = self._executable("accelerate")
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"independent-expert")
        self.anchors = self.root / "anchors"
        self.anchors.mkdir()
        (self.anchors / "anchor_00000.npz").write_bytes(b"unit-test-anchor")
        self.counter = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _executable(self, name: str) -> Path:
        path = self.root / name
        path.write_text(
            f"#!/usr/bin/env bash\nprintf invoked > {self.marker!s}\nexit 99\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _bundle(
        self,
        *,
        mutate_spec: Callable[[dict], None] | None = None,
        mutate_manifest: Callable[[dict], None] | None = None,
    ) -> tuple[Path, str]:
        self.counter += 1
        checkpoint = self.root / f"student-{self.counter}.pt"
        checkpoint.write_bytes(f"fresh-student-{self.counter}".encode("ascii"))
        checkpoint_sha = sha256_file(checkpoint)
        spec = _portable_spec(checkpoint_sha)
        if mutate_spec is not None:
            mutate_spec(spec)
        spec_path = Path(f"{checkpoint}.policy_spec.json")
        publish_immutable_json(spec_path, spec)
        manifest = {
            "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha,
            "runtime_policy_sha256": RUNTIME_SHA,
            "policy_spec_path": spec_path.name,
            "policy_spec_sha256": sha256_file(spec_path),
        }
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        publish_immutable_json(Path(f"{checkpoint}.manifest.json"), manifest)
        return checkpoint, checkpoint_sha

    def _environment(
        self,
        *,
        ranks: int = 1,
        envs_per_rank: int = 1,
        steps: int = 2,
        dry_run: bool = True,
        purpose: str = "eval",
        train_ack: bool = False,
        actor_rank: int | None = None,
        contract_overrides: dict | None = None,
        mutate_spec: Callable[[dict], None] | None = None,
        mutate_manifest: Callable[[dict], None] | None = None,
    ) -> dict[str, str]:
        checkpoint, policy_sha = self._bundle(
            mutate_spec=mutate_spec, mutate_manifest=mutate_manifest
        )
        expected_envs = ranks * envs_per_rank if ranks > 0 and envs_per_rank > 0 else 1
        expected_steps = steps if steps > 0 else 1
        values = {
            "campaign_id": "validation-campaign",
            "macro_round": 7,
            "cohort_index": 3,
            "purpose": purpose,
            "expected_ranks": list(range(ranks)) if ranks > 0 else [0],
            "expected_global_env_ids": list(range(expected_envs)),
            "expected_transitions": expected_envs * expected_steps,
            "view_contract": {
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [324, 192, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            "action_contract": {"dtype": "float32", "shape": [26]},
            "policy_sha256": policy_sha,
            "runtime_policy_sha256": RUNTIME_SHA,
            "teacher_sha256": _sha256(self.teacher),
            "renderer_sha256": RENDERER_SHA,
            "beta_sha256": BETA_SHA,
            "config_sha256": CONFIG_SHA,
        }
        values.update(contract_overrides or {})
        cohort = self.root / f"cohort-{self.counter}"
        create_cohort(cohort, **values)

        endpoint_paths = []
        for index in range(max(ranks, 1)):
            descriptor = self.root / f"endpoint-{self.counter}-{index}.json"
            descriptor.write_text("{}", encoding="utf-8")
            endpoint_paths.append(str(descriptor))
        endpoint_manifest = self.root / f"endpoints-{self.counter}.txt"
        endpoint_manifest.write_text("\n".join(endpoint_paths) + "\n", encoding="utf-8")

        environment = os.environ.copy()
        environment.update(
            {
                "GR00T_PYTHON": str(self.fake_python),
                "ACCELERATE": str(self.fake_accelerate),
                "ROUND_DAGGER_VALIDATION_RANKS": str(ranks),
                "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": str(envs_per_rank),
                "ROUND_DAGGER_VALIDATION_STEPS": str(steps),
                "ROUND_DAGGER_VALIDATION_PURPOSE": purpose,
                "DRY_RUN": "1" if dry_run else "0",
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_CAMPAIGN_ID": "validation-campaign",
                "ROUND_DAGGER_MACRO_ROUND": "7",
                "ROUND_DAGGER_COHORT_INDEX": "3",
                "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
                "ROUND_DAGGER_POLICY_SHA256": policy_sha,
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": RUNTIME_SHA,
                "ROUND_DAGGER_TEACHER_CHECKPOINT": str(self.teacher),
                "ROUND_DAGGER_TEACHER_SHA256": _sha256(self.teacher),
                "ROUND_DAGGER_RENDERER_SHA256": RENDERER_SHA,
                "ROUND_DAGGER_BETA": "0.5",
                "ROUND_DAGGER_BETA_SHA256": BETA_SHA,
                "ROUND_DAGGER_CONFIG_SHA256": CONFIG_SHA,
                "ROUND_DAGGER_GLOBAL_ENV_OFFSET": "0",
                "ROUND_DAGGER_EXPECTED_RENDERERS": str(max(ranks, 1)),
                "ROUND_DAGGER_MIN_ANCHORS": "1",
                "IMAGE3F_ENDPOINT_MANIFEST": str(endpoint_manifest),
                "ANCHORS": str(self.anchors),
                "TRAIN_RUN_DIR": str(self.root / f"run-{self.counter}"),
                # The launcher owns this value.  Supplying train must not make a
                # validation cohort collect train data.
                "ROUND_DAGGER_PURPOSE": "train",
            }
        )
        if train_ack:
            environment["ROUND_DAGGER_VALIDATION_TRAIN_ACK"] = "bounded-train-replay"
        if actor_rank is not None:
            environment["ROUND_DAGGER_VALIDATION_ACTOR_RANK"] = str(actor_rank)
        return environment

    def _run(self, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.marker.unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_one_rank_eval_dry_run_binds_exact_action_contract(self) -> None:
        result = self._run(self._environment(ranks=1, envs_per_rank=1, steps=2))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ROUND_ACTOR_VALIDATION_DRY_RUN ranks=1 actor_rank=none purpose=eval envs_per_rank=1 steps=2 transitions=2",
            result.stdout,
        )
        self.assertIn("num_envs=1", result.stdout)
        self.assertIn("algo.config.num_steps_per_env=2", result.stdout)
        self.assertIn("seed=0", result.stdout)
        self.assertIn("++algo.config.videogen.seed=42", result.stdout)
        self.assertIn(
            'export IMAGE3F_LOCAL_ENVS="$ENVS_PER_RANK"',
            SCRIPT.read_text(encoding="utf-8"),
        )
        self.assertFalse(self.marker.exists())

    def test_two_rank_eval_dry_run_is_one_isolated_rank_and_exact_coverage(self) -> None:
        result = self._run(
            self._environment(ranks=2, actor_rank=1, envs_per_rank=3, steps=4)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ROUND_ACTOR_VALIDATION_DRY_RUN ranks=2 actor_rank=1 purpose=eval envs_per_rank=3 steps=4 transitions=24",
            result.stdout,
        )
        self.assertNotIn("--multi_gpu", result.stdout)
        self.assertFalse(self.marker.exists())

    def test_lag_boundary_gate_disables_resets_only_for_long_eval(self) -> None:
        result = self._run(
            self._environment(ranks=1, envs_per_rank=1, steps=258)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for override in (
            "env.config.max_episode_length_s=8",
            "++env.config.reset_on_overtime=false",
            "env.config.reset_on_complete=false",
            "env.config.termination.terminate_object_by_low_height=false",
            "env.config.termination.terminate_by_face_mul=false",
            "env.config.termination.terminate_by_throw_object=false",
            "env.config.termination.terminate_by_fast_right_arm_qvel=false",
            "env.config.termination.terminate_by_hold_object_tray_distance=false",
            "env.config.termination.terminate_after_finish_turning_for_student=false",
        ):
            self.assertIn(override, result.stdout)

        short = self._run(
            self._environment(ranks=1, envs_per_rank=1, steps=256)
        )
        self.assertEqual(short.returncode, 0, short.stderr)
        self.assertNotIn("++env.config.reset_on_overtime=false", short.stdout)

    def test_train_validation_cannot_mutate_dynamics_for_lag_gate(self) -> None:
        result = self._run(
            self._environment(
                ranks=1,
                envs_per_rank=1,
                steps=258,
                purpose="train",
                train_ack=True,
            )
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("crossing lag 256 must use purpose=eval", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_multi_rank_without_isolated_rank_fails_before_launch(self) -> None:
        result = self._run(self._environment(ranks=2, envs_per_rank=1, steps=2))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must launch isolated actor ranks independently", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_bounded_train_requires_explicit_acknowledgement(self) -> None:
        rejected = self._run(self._environment(purpose="train"))
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("bounded train validation requires", rejected.stderr)
        self.assertFalse(self.marker.exists())

        accepted = self._run(
            self._environment(purpose="train", train_ack=True, dry_run=True)
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("purpose=train", accepted.stdout)
        self.assertFalse(self.marker.exists())

    def test_shadow_and_validation_evidence_require_immutable_ab_plan(self) -> None:
        shadow = self._environment()
        shadow.update(
            {
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
                "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
            }
        )
        rejected = self._run(shadow)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("shadow runtime requires an immutable runtime A/B plan", rejected.stderr)
        self.assertFalse(self.marker.exists())

        evidence = self._environment()
        evidence.update(
            {
                "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK": "paired-1x2-v1",
                "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE": str(
                    self.root / "unplanned-evidence.json"
                ),
            }
        )
        rejected = self._run(evidence)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("forbidden without an A/B plan", rejected.stderr)
        self.assertFalse(self.marker.exists())

    def test_partial_runtime_ab_contract_fails_before_launch(self) -> None:
        environment = self._environment()
        environment["ROUND_DAGGER_RUNTIME_AB_PLAN"] = str(self.root / "missing.json")
        rejected = self._run(environment)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("runtime adapter A/B launch is missing", rejected.stderr)
        self.assertFalse(self.marker.exists())

    def test_rank_environment_and_step_bounds_fail_before_launch(self) -> None:
        cases = (
            (0, 1, 2, "ranks"),
            (3, 1, 2, "ranks"),
            (1, 0, 2, "positive integers"),
            (1, 33, 2, "<= 32"),
            (1, 1, 0, "positive integers"),
            (1, 1, 301, "<= 300"),
        )
        for ranks, envs, steps, message in cases:
            with self.subTest(ranks=ranks, envs=envs, steps=steps):
                result = self._run(
                    self._environment(ranks=ranks, envs_per_rank=envs, steps=steps)
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.marker.exists())

    def test_policy_11_replay_26_and_environment_31_are_fail_closed(self) -> None:
        mutations = {
            "policy": lambda spec: spec["output_contract"].__setitem__(
                "policy_action_dim", 12
            ),
            "replay": lambda spec: spec["output_contract"].__setitem__(
                "replay_action_dim", 31
            ),
            "environment": lambda spec: spec["output_contract"].__setitem__(
                "env_command_dim", 30
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                result = self._run(
                    self._environment(dry_run=False, mutate_spec=mutation)
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.marker.exists())

        result = self._run(
            self._environment(
                dry_run=False,
                contract_overrides={
                    "action_contract": {"dtype": "float32", "shape": [31]}
                },
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("replay-action contract mismatch", result.stderr)
        self.assertFalse(self.marker.exists())

    def test_every_immutable_cohort_identity_field_is_checked_prelaunch(self) -> None:
        cases = {
            "campaign_id": {"campaign_id": "other-campaign"},
            "macro_round": {"macro_round": 8},
            "cohort_index": {"cohort_index": 4},
            "purpose": {"purpose": "train"},
            "expected_ranks": {"expected_ranks": [0, 1]},
            "expected_global_env_ids": {"expected_global_env_ids": [9]},
            "expected_transitions": {"expected_transitions": 3},
            "policy_sha256": {"policy_sha256": "a" * 64},
            "runtime_policy_sha256": {"runtime_policy_sha256": "b" * 64},
            "teacher_sha256": {"teacher_sha256": "c" * 64},
            "renderer_sha256": {"renderer_sha256": "d" * 64},
            "beta_sha256": {"beta_sha256": "e" * 64},
            "config_sha256": {"config_sha256": "f" * 64},
        }
        for field, override in cases.items():
            with self.subTest(field=field):
                result = self._run(
                    self._environment(dry_run=False, contract_overrides=override)
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"contract mismatch for {field}", result.stderr)
                self.assertFalse(self.marker.exists())

    def test_policy_bundle_manifest_is_bound_before_launch(self) -> None:
        mutations = {
            "runtime": lambda manifest: manifest.__setitem__(
                "runtime_policy_sha256", "f" * 64
            ),
            "spec digest": lambda manifest: manifest.__setitem__(
                "policy_spec_sha256", "e" * 64
            ),
            "unknown key": lambda manifest: manifest.__setitem__("extra", True),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                result = self._run(
                    self._environment(dry_run=False, mutate_manifest=mutation)
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.marker.exists())

    def test_zero_exit_trainer_without_a_shard_is_not_success(self) -> None:
        self.fake_python.write_text(
            f"#!/usr/bin/env bash\nprintf invoked > {self.marker!s}\nexit 0\n",
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)
        result = self._run(
            self._environment(
                ranks=2,
                actor_rank=0,
                envs_per_rank=1,
                steps=2,
                dry_run=False,
            )
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue(self.marker.exists(), "fake zero-exit trainer was not invoked")
        self.assertIn("returned without exact shard coverage", result.stderr)
        self.assertNotIn("ROUND_ACTOR_VALIDATION_RANK_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
