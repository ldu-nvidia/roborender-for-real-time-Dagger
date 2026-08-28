import tempfile
import subprocess
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger import handoff
from round_dagger.image3f_lifecycle import create_attempt_plan
from round_dagger.seed_policy import derive_actor_seed_contract
from round_dagger.gr00t_policy import (
    BASE_MODULE_TARGET,
    POLICY_TARGET,
    build_portable_policy_spec,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256
from round_dagger.production_data_lineage import (
    build_cohort_assignment,
    data_lineage_sha256,
    derive_data_lineage,
)


HOMIE_CONTROLLER_SHA256 = {"walk": "7" * 64, "stand": "8" * 64}
DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}


def _production_data_lineage() -> dict:
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
    bundles = [
        {
            "seed": seed,
            "path": f"/immutable/first-condition-seed-{seed}.npz",
            "sha256": f"{index + 1:x}" * 64,
            "size": 1,
            "case_count": 32,
            "source_pass_index": 1,
        }
        for index, seed in enumerate((17, 29, 43, 59, 71, 89, 101, 127))
    ]
    return derive_data_lineage(
        {
            "contract_sha256": "8" * 64,
            "invariant_profile_sha256": "9" * 64,
            "live_capture": {
                "attestation_sha256": "a" * 64,
                "evidence_sha256": "b" * 64,
                "scenario_sha256": "c" * 64,
                "first_condition_bundles": bundles,
                "first_condition_bundle_inventory_sha256": (
                    artifacts.canonical_json_sha256(bundles)
                ),
            },
            "capture_provenance": {
                "motion_split_manifest_sha256": motion_split[
                    "manifest_sha256"
                ],
                "heldout_plan_sha256": motion_split[
                    "heldout_plan_sha256"
                ],
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


_FakePolicy = type("VisionActorObjPred", (), {})
_FakePolicy.__module__ = "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"


def portable_policy_spec(checkpoint_sha256, policy_state_sha256):
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
        policy_state_sha256=policy_state_sha256,
        homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
    )


def publish_policy_bundle(checkpoint, policy_state_sha256, mutate_spec=None):
    checkpoint_sha256 = artifacts.sha256_file(checkpoint)
    spec = portable_policy_spec(checkpoint_sha256, policy_state_sha256)
    if mutate_spec is not None:
        mutate_spec(spec)
    spec_path = Path(f"{checkpoint}.policy_spec.json")
    manifest_path = Path(f"{checkpoint}.manifest.json")
    spec_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    artifacts.publish_immutable_json(spec_path, spec)
    artifacts.publish_immutable_json(
        manifest_path,
        {
            "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "runtime_policy_sha256": spec["runtime_policy_sha256"],
            "policy_spec_path": spec_path.name,
            "policy_spec_sha256": artifacts.sha256_file(spec_path),
        },
    )
    return spec_path, manifest_path


def arrays():
    expert = np.ones((2, 2, 2), dtype=np.float32)
    student = np.zeros_like(expert)
    mask = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    return {
        "policy_rgb": np.zeros((2, 2, 6, 4, 3), dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": np.where(mask[..., None], expert, student),
        "expert_mask": mask,
        "dones": np.zeros((2, 2), dtype=np.bool_),
        "observation_is_raytraced": np.zeros((2, 2), dtype=np.bool_),
        "global_env_ids": np.asarray([0, 1], dtype=np.int64),
        "episode_ids": np.zeros((2, 2), dtype=np.int64),
        "episode_steps": np.asarray([[0, 0], [1, 1]], dtype=np.int64),
        "obs__actor_obs": np.ones((2, 2, 3), dtype=np.float32),
    }


def create_cohort(path):
    identity = {
        "campaign_id": "handoff-test",
        "macro_round": 0,
        "cohort_index": 0,
        "purpose": "train",
    }
    artifacts.create_cohort(
        path,
        expected_ranks=[0],
        expected_global_env_ids=[0, 1],
        expected_transitions=4,
        view_contract={
            "dtype": "uint8",
            "layout": "vertical",
            "shape": [6, 4, 3],
            "views": ["head_color", "hand_left", "hand_right"],
        },
        action_contract={"dtype": "float32", "shape": [2]},
        **identity,
        **DIGESTS,
    )
    artifacts.publish_rollout_shard(
        path,
        rank=0,
        rollout_id=0,
        part_index=0,
        arrays=arrays(),
        **identity,
        **DIGESTS,
    )
    return artifacts.seal_cohort(path)


class CohortHandoffTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.report = create_cohort(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_local_transfer_is_atomic_exact_and_idempotent(self):
        destination = self.root / "remote" / "cohort_0"
        first = handoff.transfer_local(self.source, destination)
        self.assertEqual(first["cohort_sha256"], self.report["cohort_sha256"])
        self.assertEqual(
            handoff.verify_exact_stage(destination)["cohort_sha256"],
            self.report["cohort_sha256"],
        )
        second = handoff.transfer_local(self.source, destination)
        self.assertEqual(second["cohort_sha256"], first["cohort_sha256"])
        self.assertEqual(list(destination.parent.glob(".incoming-*")), [])

    def test_interrupted_stage_is_resumed_without_publishing_partial_data(self):
        destination = self.root / "remote" / "cohort_0"
        stage = destination.parent / ".incoming-cohort_0-test"
        original = handoff._copy_one
        calls = 0

        def interrupt(source, target, digest):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected transfer interruption")
            return original(source, target, digest)

        with mock.patch.object(handoff, "_copy_one", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "injected"):
                handoff.stage_cohort(self.source, stage)
        self.assertTrue(stage.is_dir())
        self.assertFalse(destination.exists())
        staged = handoff.stage_cohort(self.source, stage)
        self.assertEqual(staged["cohort_sha256"], self.report["cohort_sha256"])
        handoff.publish_stage(stage, destination)
        self.assertTrue(destination.is_dir())

    def test_corruption_extra_files_and_conflicting_destination_fail_closed(self):
        stage = self.root / "remote" / ".incoming-cohort"
        handoff.stage_cohort(self.source, stage)
        (stage / "unreferenced.bin").write_bytes(b"not part of the cohort")
        with self.assertRaisesRegex(handoff.HandoffError, "file set mismatch"):
            handoff.verify_exact_stage(stage)

        other_source = self.root / "other-source"
        other = create_cohort(other_source)
        # Change the immutable payload identity without weakening verification.
        self.assertEqual(other["cohort_sha256"], self.report["cohort_sha256"])
        payload = next((other_source / "shards").glob("*.npz"))
        payload.chmod(0o600)
        with payload.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaises(Exception):
            handoff.transfer_local(other_source, self.root / "unused")

    def test_symlinked_stage_and_relative_paths_are_rejected(self):
        with self.assertRaisesRegex(handoff.HandoffError, "absolute"):
            handoff.transfer_local(self.source, Path("relative"))
        parent = self.root / "remote"
        parent.mkdir()
        target = parent / "target"
        target.mkdir()
        link = parent / ".incoming-link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(handoff.HandoffError, "symlink"):
            handoff.stage_cohort(self.source, link)


class ActorLauncherContractTest(unittest.TestCase):
    def test_dry_run_binds_exact_two_rank_cohort_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "policy.pt"
            teacher = root / "teacher.pt"
            checkpoint.write_bytes(b"policy")
            teacher.write_bytes(b"teacher")
            policy_sha = artifacts.sha256_file(checkpoint)
            teacher_sha = artifacts.sha256_file(teacher)
            policy_state_sha = "2" * 64
            runtime_sha = runtime_policy_identity_sha256(
                policy_state_sha, HOMIE_CONTROLLER_SHA256
            )
            renderer_sha = "4" * 64
            beta_sha = "5" * 64
            config_sha = "6" * 64
            policy_spec, policy_manifest = publish_policy_bundle(
                checkpoint, policy_state_sha
            )
            cohort = root / "cohort"
            data_lineage = _production_data_lineage()
            artifacts.create_cohort(
                cohort,
                campaign_id="launcher-test",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                expected_ranks=[0, 1],
                expected_global_env_ids=list(range(64)),
                expected_transitions=64 * 512,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [324, 192, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [26]},
                policy_sha256=policy_sha,
                runtime_policy_sha256=runtime_sha,
                teacher_sha256=teacher_sha,
                renderer_sha256=renderer_sha,
                beta_sha256=beta_sha,
                config_sha256=config_sha,
                actor_seed_contract=derive_actor_seed_contract(
                    campaign_id="launcher-test",
                    macro_round=0,
                    cohort_index=0,
                    expected_ranks=[0, 1],
                    expected_global_env_ids=list(range(64)),
                ),
                data_lineage=data_lineage,
                motion_assignment=build_cohort_assignment(
                    data_lineage,
                    cohort_index=0,
                    expected_ranks=(0, 1),
                ),
            )
            endpoints = []
            for index in range(2):
                endpoint = root / f"endpoint_{index}.txt"
                endpoint.write_text(
                    f"127.0.0.1 {index + 1}\n", encoding="utf-8"
                )
                endpoints.append(endpoint)
            endpoint_manifest = root / "endpoints.txt"
            endpoint_manifest.write_text(
                "".join(f"{path}\n" for path in endpoints), encoding="utf-8"
            )
            anchors = root / "anchors"
            anchors.mkdir()
            (anchors / "anchor.npz").write_bytes(b"placeholder")
            accelerate = root / "accelerate"
            launch_marker = root / "accelerate-was-launched"
            accelerate.write_text(
                f"#!/bin/sh\ntouch {launch_marker!s}\nexit 99\n", encoding="utf-8"
            )
            accelerate.chmod(0o755)
            run_dir = root / "run"
            attempt_dir = run_dir / "actor_pair_attempts" / "attempt_000001"
            attempt_dir.mkdir(parents=True)
            stream_plan = attempt_dir / "image3f_stream_plan.json"
            plan = create_attempt_plan(
                stream_plan,
                campaign_id="launcher-test",
                macro_round=0,
                cohort_index=0,
                attempt_dir=attempt_dir,
                cohort_dir=cohort,
                endpoint_manifest=endpoint_manifest,
                expected_renderers=2,
                renderer_sha256=renderer_sha,
                local_environments=32,
                actor_world_size=2,
                launched_ranks=[0, 1],
            )
            environment = {
                "DRY_RUN": "1",
                "GR00T_ROOT": str(root),
                "GR00T_PYTHON": "/usr/bin/python3",
                "ACCELERATE": str(accelerate),
                "ROUND_DAGGER_ACTOR_RANK": "0",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_CAMPAIGN_ID": "launcher-test",
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_COHORT_INDEX": "0",
                "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
                "ROUND_DAGGER_POLICY_SHA256": policy_sha,
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
                "ROUND_DAGGER_TEACHER_CHECKPOINT": str(teacher),
                "ROUND_DAGGER_TEACHER_SHA256": teacher_sha,
                "ROUND_DAGGER_RENDERER_SHA256": renderer_sha,
                "ROUND_DAGGER_BETA": "0.5",
                "ROUND_DAGGER_BETA_SHA256": beta_sha,
                "ROUND_DAGGER_CONFIG_SHA256": config_sha,
                "ROUND_DAGGER_IMAGE3F_ATTEMPT_NAMESPACE": plan[
                    "attempt_namespace"
                ],
                "ROUND_DAGGER_IMAGE3F_STREAM_PLAN": str(stream_plan),
                "ROUND_DAGGER_POLICY_SPEC": str(policy_spec),
                "ROUND_DAGGER_POLICY_MANIFEST": str(policy_manifest),
                "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(
                    policy_spec
                ),
                "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(
                    policy_manifest
                ),
                "ROUND_DAGGER_MIN_ANCHORS": "1",
                "ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST": str(
                    Path(__file__).resolve().parents[1]
                    / "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/"
                    "33demos_675_775_heldout_256_v1.json"
                ),
                "ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256": (
                    "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834"
                ),
                "ROUND_DAGGER_DATA_LINEAGE_SHA256": data_lineage_sha256(
                    data_lineage
                ),
                "IMAGE3F_ENDPOINT_MANIFEST": str(endpoint_manifest),
                "IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE": "simulator_rgb",
                "ANCHORS": str(anchors),
                "TRAIN_RUN_DIR": str(run_dir),
            }
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts/round_dagger/run_actor_cohort.sh"
            )
            result = subprocess.run(
                ["bash", str(script)],
                env={**__import__("os").environ, **environment},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY_RUN", result.stdout)
            self.assertIn("wsdpt_round_actor_3view", result.stdout)
            self.assertIn("actor_rank=0 actor_world_size=2", result.stdout)
            self.assertNotIn("--multi_gpu", result.stdout)
            self.assertIn(
                "export IMAGE3F_LOCAL_ENVS=32",
                script.read_text(encoding="utf-8"),
            )
            self.assertIn("-m round_dagger.policy_bundle", script.read_text())
            self.assertFalse(launch_marker.exists())

            # Each production invocation is exactly one isolated actor.  The
            # logical rank is explicit, never inferred from Accelerate/NCCL,
            # and an ambient purpose cannot turn the train cohort into eval.
            rank_one = subprocess.run(
                ["bash", str(script)],
                env={
                    **__import__("os").environ,
                    **environment,
                    "ROUND_DAGGER_ACTOR_RANK": "1",
                    "ROUND_DAGGER_PURPOSE": "eval",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rank_one.returncode, 0, rank_one.stderr)
            self.assertIn("actor_rank=1 actor_world_size=2", rank_one.stdout)
            self.assertIn("rank_0001", rank_one.stdout)
            self.assertFalse(launch_marker.exists())

            # Hydra/Kit may print a fatal exception yet return process status
            # zero.  The production rank-local launcher must prove exact shard
            # publication rather than treating that status as success.
            zero_python = root / "zero-exit-python"
            zero_marker = root / "zero-exit-trainer-ran"
            zero_python.write_text(
                f"#!/bin/sh\ntouch {zero_marker!s}\nexit 0\n", encoding="utf-8"
            )
            zero_python.chmod(0o755)
            false_success = subprocess.run(
                ["bash", str(script)],
                env={
                    **__import__("os").environ,
                    **environment,
                    "DRY_RUN": "0",
                    "GR00T_PYTHON": str(zero_python),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(false_success.returncode, 2)
            self.assertTrue(zero_marker.exists())
            self.assertIn(
                "returned without exact shard coverage", false_success.stderr
            )
            self.assertNotIn("ROUND_ACTOR_RANK_OK", false_success.stdout)

            for overrides, expected_error in (
                ({"ROUND_DAGGER_ACTOR_RANK": "2"}, "rank must be exactly"),
                ({"ROUND_DAGGER_ACTOR_RANK": "01"}, "rank must be exactly"),
                ({"ROUND_DAGGER_ACTOR_WORLD_SIZE": "1"}, "world size must be exactly"),
                ({"ROUND_DAGGER_ACTOR_WORLD_SIZE": "02"}, "world size must be exactly"),
            ):
                invalid_identity = subprocess.run(
                    ["bash", str(script)],
                    env={
                        **__import__("os").environ,
                        **environment,
                        **overrides,
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(invalid_identity.returncode, 2)
                self.assertIn(expected_error, invalid_identity.stderr)
                self.assertFalse(launch_marker.exists())

            # W&B owns the common rank-zero stream only.  Rank one must reach
            # the actor without needing any W&B credentials or creating a
            # sidecar; malformed topology must fail before W&B validation.
            wandb_script = script.with_name("run_actor_cohort_with_wandb.sh")
            wrapper_environment = {
                **environment,
                "ROUND_DAGGER_ACTOR_RUN_DIR": str(run_dir),
                "ROUND_DAGGER_WANDB_ENABLED": "1",
            }
            rank_one_wrapper = subprocess.run(
                ["bash", str(wandb_script)],
                env={
                    **__import__("os").environ,
                    **wrapper_environment,
                    "ROUND_DAGGER_ACTOR_RANK": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                rank_one_wrapper.returncode, 0, rank_one_wrapper.stderr
            )
            self.assertIn("actor_rank=1 actor_world_size=2", rank_one_wrapper.stdout)

            invalid_wrapper = subprocess.run(
                ["bash", str(wandb_script)],
                env={
                    **__import__("os").environ,
                    **wrapper_environment,
                    "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(invalid_wrapper.returncode, 2)
            self.assertIn("world size must be exactly 2", invalid_wrapper.stderr)
            self.assertNotIn("WANDB_PYTHON", invalid_wrapper.stderr)

            split_telemetry = subprocess.run(
                ["bash", str(wandb_script)],
                env={
                    **__import__("os").environ,
                    **wrapper_environment,
                    "ROUND_DAGGER_ACTOR_RUN_DIR": str(root / "other-run"),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(split_telemetry.returncode, 2)
            self.assertIn("must equal TRAIN_RUN_DIR", split_telemetry.stderr)

            action_mutations = {
                "policy 11": (
                    lambda spec: spec["output_contract"].__setitem__(
                        "policy_action_dim", 12
                    ),
                    "policy action dimension must be 11",
                ),
                "Homie 15": (
                    lambda spec: spec["output_contract"].__setitem__(
                        "homie_action_dim", 14
                    ),
                    "Homie action dimension must be 15",
                ),
                "replay 26": (
                    lambda spec: spec["output_contract"].__setitem__(
                        "replay_action_dim", 27
                    ),
                    "replay action dimension",
                ),
                "environment 31": (
                    lambda spec: spec["output_contract"].__setitem__(
                        "env_command_dim", 30
                    ),
                    "simulator command dimension must be 31",
                ),
            }
            for label, (mutation, expected_error) in action_mutations.items():
                with self.subTest(label=label):
                    publish_policy_bundle(checkpoint, policy_state_sha, mutation)
                    invalid = subprocess.run(
                        ["bash", str(script)],
                        env={
                            **__import__("os").environ,
                            **environment,
                            "DRY_RUN": "0",
                            "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(
                                policy_spec
                            ),
                            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(
                                policy_manifest
                            ),
                        },
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertNotEqual(invalid.returncode, 0)
                    self.assertIn(expected_error, invalid.stderr)
                    self.assertFalse(launch_marker.exists())

            publish_policy_bundle(checkpoint, policy_state_sha)
            manifest_document = artifacts.load_strict_json(policy_manifest)
            policy_manifest.unlink()
            manifest_document["runtime_policy_sha256"] = "f" * 64
            artifacts.publish_immutable_json(policy_manifest, manifest_document)
            wrong_runtime = subprocess.run(
                ["bash", str(script)],
                env={
                    **__import__("os").environ,
                    **environment,
                    "DRY_RUN": "0",
                    "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(
                        policy_manifest
                    ),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(wrong_runtime.returncode, 0)
            self.assertIn(
                "bootstrap runtime policy identity differs from cohort",
                wrong_runtime.stderr,
            )
            self.assertFalse(launch_marker.exists())

            publish_policy_bundle(checkpoint, policy_state_sha)
            policy_manifest.unlink()
            missing = subprocess.run(
                ["bash", str(script)],
                env={
                    **__import__("os").environ,
                    **environment,
                    "DRY_RUN": "0",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("missing policy bundle file", missing.stderr)
            self.assertFalse(launch_marker.exists())


if __name__ == "__main__":
    unittest.main()
