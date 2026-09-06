import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger.conditioning_artifacts import (
    derive_dense_image_schedule,
    load_raw_conditioning_artifact,
    validate_conditioning_contract,
)
from round_dagger.actor import (
    ActorExportError,
    _native_generation_shard_ids_from_environment,
    _native_rank_motion_ids,
    _validate_native_export_role,
    denormalize_policy_rgb,
    export_trainer_rollout,
    runtime_policy_sha256,
    runtime_state_dict_sha256,
    trainer_rollout_arrays,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256


HOMIE_CONTROLLER_SHA256 = {"walk": "a" * 64, "stand": "b" * 64}


class FakeStorage:
    def __init__(self, values):
        self.values = values
        self.step = int(values["vision_obs"].shape[0])

    def query_key(self, key):
        return self.values[key]


class NumpyPolicy:
    def __init__(self):
        self.weight = np.arange(10, dtype=np.float32).reshape(2, 5)
        self.bias = np.zeros(2, dtype=np.float32)

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}


class TransferProbe:
    """Tensor-like spy proving that slicing precedes every host transfer."""

    def __init__(self, array, *, key, transfers):
        self.array = array
        self.key = key
        self.transfers = transfers

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, item):
        return type(self)(self.array[item], key=self.key, transfers=self.transfers)

    def detach(self):
        return self

    def cpu(self):
        self.transfers.append((self.key, tuple(self.array.shape)))
        return self

    def numpy(self):
        return self.array


def fake_trainer(*, steps=3, envs=4):
    mean = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    std = np.asarray([0.5, 0.4, 0.3], dtype=np.float32)
    rgb = np.arange(steps * envs * 6 * 4 * 3, dtype=np.uint8).reshape(
        steps, envs, 6, 4, 3
    )
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    expert = np.arange(steps * envs * 2, dtype=np.float32).reshape(steps, envs, 2)
    student = expert + np.float32(0.5)
    expert_mask = (np.indices((steps, envs))[0] % 2 == 0)[..., None]
    behavior = np.where(expert_mask, expert, student).astype(np.float32)
    ids = np.broadcast_to(
        np.arange(10, 10 + envs, dtype=np.int64)[None, :, None],
        (steps, envs, 1),
    ).copy()
    episode_steps = np.broadcast_to(
        np.arange(steps, dtype=np.int64)[:, None, None], (steps, envs, 1)
    ).copy()
    reference_kinds = np.full((steps, envs, 1), 2, dtype=np.uint8)
    reference_kinds[0] = 1
    reference_source_steps = np.zeros((steps, envs, 1), dtype=np.int64)
    reference_source_steps[0] = -1
    reference_digest = np.ones((steps, envs, 32), dtype=np.uint8)
    reference_digest[0] = 9
    generated_digest = np.empty((steps, envs, 32), dtype=np.uint8)
    for step in range(steps):
        generated_digest[step] = step + 1
    values = {
        "vision_obs": normalized,
        "gt_actions": expert,
        "student_actions": student,
        "actions": behavior,
        "expert_mask": expert_mask,
        "dones": np.zeros((steps, envs, 1), dtype=np.bool_),
        "round_observation_is_raytraced": np.zeros(
            (steps, envs, 1), dtype=np.bool_
        ),
        "round_global_env_ids": ids,
        "round_episode_ids": np.zeros((steps, envs, 1), dtype=np.int64),
        "round_episode_steps": episode_steps,
        "round_reference_kinds": reference_kinds,
        "round_reference_lags": np.full(
            (steps, envs, 1), 256, dtype=np.int64
        ),
        "round_reference_source_steps": reference_source_steps,
        "round_reference_sha256": reference_digest,
        "round_generated_sha256": generated_digest,
        "round_conditioning_depth_nonzero_fraction": np.full(
            (steps, envs, 3), 0.75, dtype=np.float32
        ),
        "round_conditioning_mask_nonzero_fraction": np.full(
            (steps, envs, 3), 0.1, dtype=np.float32
        ),
        "actor_obs": np.ones((steps, envs, 5), dtype=np.float32),
    }
    cameras = SimpleNamespace(image_mean=mean.tolist(), image_std=std.tolist())
    env = SimpleNamespace(
        config=SimpleNamespace(
            simulator=SimpleNamespace(config=SimpleNamespace(cameras=cameras))
        )
    )
    trainer = SimpleNamespace(
        storage=FakeStorage(values),
        num_steps_per_env=steps,
        env=env,
        algo_obs_dim_dict={"vision_obs": 72, "actor_obs": 5},
        accelerator=SimpleNamespace(process_index=0),
        homie_controller_sha256=dict(HOMIE_CONTROLLER_SHA256),
    )
    return trainer, rgb


def native_conditioning_slice(
    *, global_env_ids, episode_ids, episode_steps, **_kwargs
):
    schedule = derive_dense_image_schedule(
        global_env_ids, episode_ids, episode_steps
    )
    image_count = int(schedule["image_global_env_ids"].size)
    episodes = sorted(
        {
            (int(global_env_ids[t, b]), int(episode_ids[t, b]))
            for t in range(global_env_ids.shape[0])
            for b in range(global_env_ids.shape[1])
        }
    )
    result = {
        "anchor_valid": episode_steps == 0,
        "anchor_global_env_ids": np.asarray([row[0] for row in episodes], np.int64),
        "anchor_episode_ids": np.asarray([row[1] for row in episodes], np.int64),
        **schedule,
    }
    depth = np.broadcast_to(
        np.arange(416, dtype=np.uint8)[None, None, :],
        (image_count, 240, 416),
    ).copy()
    robot = np.zeros((image_count, 240, 416, 3), dtype=np.uint8)
    robot[:, 40:120, 80:200] = 100
    anchor = np.broadcast_to(
        np.arange(416, dtype=np.uint8)[None, None, :, None],
        (len(episodes), 240, 416, 3),
    ).copy()
    for view in ("head_color", "hand_left", "hand_right"):
        result[f"depth_ctrl__{view}"] = depth.copy()
        result[f"rgb_robot__{view}"] = robot.copy()
        result[f"rgb_anchor__{view}"] = anchor.copy()
    return result


class ActorConversionTest(unittest.TestCase):
    def test_denormalization_recovers_uint8_policy_observation(self):
        trainer, expected = fake_trainer()
        arrays = trainer_rollout_arrays(trainer)
        np.testing.assert_array_equal(arrays["policy_rgb"], expected)
        self.assertEqual(arrays["global_env_ids"].shape, (3, 4))
        self.assertEqual(arrays["expert_mask"].shape, (3, 4))
        self.assertEqual(arrays["obs__actor_obs"].shape, (3, 4, 5))
        self.assertEqual(arrays["reference_sha256"].shape, (3, 4, 32))
        self.assertEqual(
            arrays["reference_kinds"][:, 0].tolist(), [1, 2, 2]
        )

    def test_bad_normalization_and_incomplete_storage_fail_closed(self):
        with self.assertRaisesRegex(ActorExportError, "provided together"):
            denormalize_policy_rgb(
                np.zeros((1, 1, 2, 2, 3), dtype=np.float32),
                image_mean=[0, 0, 0],
                image_std=None,
            )
        trainer, _ = fake_trainer()
        trainer.storage.step -= 1
        with self.assertRaisesRegex(ActorExportError, "incomplete"):
            trainer_rollout_arrays(trainer)

    def test_native_replay_keeps_student_actor_obs_and_excludes_privileged_obs(self):
        trainer, _ = fake_trainer()
        trainer.observation_source = "native"
        trainer.algo_obs_dim_dict["teacher_obs"] = 7
        trainer.storage.values["teacher_obs"] = np.ones((3, 4, 7), dtype=np.float32)
        trainer.roborender_conditioning_slice = native_conditioning_slice
        arrays = trainer_rollout_arrays(trainer)
        self.assertIn("obs__actor_obs", arrays)
        self.assertNotIn("policy_rgb", arrays)
        self.assertIn("depth_ctrl__head_color", arrays)
        self.assertIn("rgb_robot__hand_left", arrays)
        self.assertIn("rgb_anchor__hand_right", arrays)
        self.assertNotIn("obs__teacher_obs", arrays)
        self.assertNotIn("observation_is_raytraced", arrays)
        self.assertIn("image_reference_indices", arrays)

    def test_validation_collection_is_confined_to_heldout_conditioning_root(self):
        trainer = SimpleNamespace(
            observation_source="native", native_role="validation_collect"
        )
        environment = {
            "ROUND_DAGGER_PURPOSE": "validation",
            "ROUND_DAGGER_VALIDATION_CONDITIONING_ROOT": "/heldout/validation",
        }
        _validate_native_export_role(
            trainer, environment, Path("/heldout/validation/panel-001")
        )
        with self.assertRaisesRegex(ActorExportError, "heldout validation root"):
            _validate_native_export_role(
                trainer, environment, Path("/campaign/train/cohort")
            )
        wrong = dict(environment, ROUND_DAGGER_PURPOSE="train")
        with self.assertRaisesRegex(ActorExportError, "purpose disagree"):
            _validate_native_export_role(
                trainer, wrong, Path("/heldout/validation/panel-001")
            )

    def test_runtime_digest_is_stable_and_parameter_sensitive(self):
        policy = NumpyPolicy()
        first = runtime_policy_sha256(
            policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
        )
        self.assertEqual(
            first,
            runtime_policy_sha256(
                policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
            ),
        )
        policy.weight[0, 0] += 1
        self.assertNotEqual(
            first,
            runtime_policy_sha256(
                policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
            ),
        )

    def test_runtime_digest_binds_state_and_both_homie_controllers(self):
        policy = NumpyPolicy()
        state_sha256 = runtime_state_dict_sha256(policy.state_dict())
        self.assertEqual(
            runtime_policy_sha256(
                policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
            ),
            runtime_policy_identity_sha256(
                state_sha256, HOMIE_CONTROLLER_SHA256
            ),
        )
        for role in HOMIE_CONTROLLER_SHA256:
            changed = dict(HOMIE_CONTROLLER_SHA256)
            changed[role] = "c" * 64
            with self.subTest(role=role):
                self.assertNotEqual(
                    runtime_policy_sha256(
                        policy, homie_controller_sha256=changed
                    ),
                    runtime_policy_identity_sha256(
                        state_sha256, HOMIE_CONTROLLER_SHA256
                    ),
                )
        with self.assertRaisesRegex(ActorExportError, "non-empty mapping"):
            runtime_state_dict_sha256({})

    def test_verified_checkpoint_runtime_hash_cli(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from round_dagger.artifacts import sha256_file

        script = Path(__file__).parents[1] / "scripts/round_dagger/checkpoint_runtime_sha256.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate.pt"
            state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            torch.save(
                {
                    "policy_state_dict": state,
                    "homie_controller_sha256": HOMIE_CONTROLLER_SHA256,
                },
                checkpoint,
            )
            expected = sha256_file(checkpoint)
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-sha256",
                    expected,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["checkpoint_sha256"], expected)
            self.assertEqual(
                report["policy_state_sha256"], runtime_state_dict_sha256(state)
            )
            self.assertEqual(
                report["runtime_policy_sha256"],
                runtime_policy_identity_sha256(
                    runtime_state_dict_sha256(state),
                    HOMIE_CONTROLLER_SHA256,
                ),
            )
            self.assertEqual(
                report["homie_controller_sha256"], HOMIE_CONTROLLER_SHA256
            )
            self.assertEqual(report["state_entries"], 1)

            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-sha256",
                    expected,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("mismatch", rejected.stderr)

            link = root / "candidate-link.pt"
            link.symlink_to(checkpoint)
            rejected_link = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(link),
                    "--expected-sha256",
                    sha256_file(checkpoint),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_link.returncode, 0)
            self.assertIn("symlink", rejected_link.stderr)


class NativeGenerationTopologyTest(unittest.TestCase):
    def test_generation_shards_use_eight_actors_in_one_wave(self):
        campaign = Path("/workspace/project/roborender/data")
        generation = campaign / "BC/isaac/collection"
        rank = 3
        part_count = 8
        expected_names = []
        for gpu in range(8):
            for part_index in range(part_count):
                shard_id = (
                    f"gen-000-bc__wave-000__gpu-{gpu:02d}"
                    f"__shard-{gpu * part_count + part_index:06d}"
                )
                expected_names.extend((f"{shard_id}.npz", f"{shard_id}.json"))
        environment = {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
            "ROUND_DAGGER_GENERATION_ROOT": str(generation),
            "ROUND_DAGGER_GENERATION_ID": "gen-000-bc",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_ACTOR_ROLE": "bc_collect",
            "ROUND_DAGGER_BETA": "1.0",
            "ROUND_DAGGER_SATURATION_PHASE": "collect",
            "ROUND_DAGGER_SATURATION_WAVE_INDEX": "0",
            "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON": json.dumps(expected_names),
            "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
            "ROUND_DAGGER_PURPOSE": "train",
            "ROUND_DAGGER_COHORT_DIR": str(
                generation / "conditioning/waves/wave-000"
            ),
            "ROUND_DAGGER_ACTOR_RANK": str(rank),
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "8",
        }
        observed = _native_generation_shard_ids_from_environment(
            environment,
            rank=rank,
            batch_size=32,
            environments_per_shard=4,
        )
        self.assertEqual(len(observed), 8)
        self.assertEqual(observed[0], "gen-000-bc__wave-000__gpu-03__shard-000024")
        self.assertEqual(observed[-1], "gen-000-bc__wave-000__gpu-03__shard-000031")

        wrong_world = {**environment, "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1"}
        wrong_world["ROUND_DAGGER_ACTOR_RANK"] = "0"
        with self.assertRaisesRegex(ActorExportError, "exact eight-rank"):
            _native_generation_shard_ids_from_environment(
                wrong_world,
                rank=0,
                batch_size=32,
                environments_per_shard=4,
            )

    def test_motion_offsets_follow_declared_world_size(self):
        trainer = SimpleNamespace(native_group_ids=("a", "b", "c"))
        production = {
            "ROUND_DAGGER_ACTOR_RANK": "0",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
            "ROUND_DAGGER_GROUP_ROTATION_JSON": json.dumps(
                {
                    "start_group_offset": 0,
                    "rank_group_offsets": [1],
                    "counts": {"a": 1, "b": 2, "c": 1},
                }
            ),
        }
        self.assertEqual(
            _native_rank_motion_ids(trainer, production, rank=0, batch_size=4),
            ("b", "c", "a", "b"),
        )

        heldout = {
            **production,
            "ROUND_DAGGER_ACTOR_RANK": "1",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            "ROUND_DAGGER_GROUP_ROTATION_JSON": json.dumps(
                {
                    "start_group_offset": 0,
                    "rank_group_offsets": [0, 2],
                    "counts": {"a": 1, "b": 1, "c": 1},
                }
            ),
        }
        self.assertEqual(
            _native_rank_motion_ids(trainer, heldout, rank=1, batch_size=3),
            ("c", "a", "b"),
        )


class ActorPublicationIntegrationTest(unittest.TestCase):
    def _cohort_environment(
        self,
        root,
        *,
        trainer,
        policy,
        environments_per_shard=2,
        campaign_id="actor-test",
    ):
        envs = int(trainer.storage.values["vision_obs"].shape[1])
        steps = int(trainer.num_steps_per_env)
        runtime_sha = runtime_policy_sha256(
            policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
        )
        digests = {
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": runtime_sha,
            "teacher_sha256": "3" * 64,
            "renderer_sha256": "4" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }
        identity = {
            "campaign_id": campaign_id,
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
        }
        cohort = Path(root) / "cohort"
        artifacts.create_cohort(
            cohort,
            expected_ranks=[0],
            expected_global_env_ids=list(range(10, 10 + envs)),
            expected_transitions=steps * envs,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [6, 4, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            action_contract={"dtype": "float32", "shape": [2]},
            **identity,
            **digests,
        )
        environment = {
            "ROUND_DAGGER_COHORT_DIR": str(cohort),
            "ROUND_DAGGER_CAMPAIGN_ID": identity["campaign_id"],
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_COHORT_INDEX": "0",
            "ROUND_DAGGER_POLICY_SHA256": digests["policy_sha256"],
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
            "ROUND_DAGGER_TEACHER_SHA256": digests["teacher_sha256"],
            "ROUND_DAGGER_RENDERER_SHA256": digests["renderer_sha256"],
            "ROUND_DAGGER_BETA_SHA256": digests["beta_sha256"],
            "ROUND_DAGGER_CONFIG_SHA256": digests["config_sha256"],
            "ROUND_DAGGER_ENVS_PER_SHARD": str(environments_per_shard),
            "ROUND_DAGGER_ROLLOUT_ID": "0",
        }
        return cohort, environment, runtime_sha, identity, digests

    def test_one_rank_exports_parts_and_seals_exactly(self):
        trainer, _ = fake_trainer()
        policy = NumpyPolicy()
        runtime_sha = runtime_policy_sha256(
            policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
        )
        digests = {
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": runtime_sha,
            "teacher_sha256": "3" * 64,
            "renderer_sha256": "4" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }
        identity = {
            "campaign_id": "actor-test",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
        }
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "cohort"
            artifacts.create_cohort(
                cohort,
                expected_ranks=[0],
                expected_global_env_ids=[10, 11, 12, 13],
                expected_transitions=12,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [2]},
                **identity,
                **digests,
            )
            environment = {
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_CAMPAIGN_ID": identity["campaign_id"],
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_COHORT_INDEX": "0",
                "ROUND_DAGGER_POLICY_SHA256": digests["policy_sha256"],
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
                "ROUND_DAGGER_TEACHER_SHA256": digests["teacher_sha256"],
                "ROUND_DAGGER_RENDERER_SHA256": digests["renderer_sha256"],
                "ROUND_DAGGER_BETA_SHA256": digests["beta_sha256"],
                "ROUND_DAGGER_CONFIG_SHA256": digests["config_sha256"],
                "ROUND_DAGGER_ENVS_PER_SHARD": "2",
                "ROUND_DAGGER_ROLLOUT_ID": "0",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )
            self.assertEqual(len(reports), 2)
            sealed = artifacts.seal_cohort(cohort)
            self.assertEqual(sealed["transition_count"], 12)
            self.assertEqual(sealed["global_env_ids"], [10, 11, 12, 13])

    def test_native_export_publishes_raw_conditioning_not_learner_replay(self):
        trainer, _ = fake_trainer(steps=3, envs=2)
        trainer.observation_source = "native"
        trainer.native_role = "bc_collect"
        trainer.native_split_manifest_sha256 = "7" * 64
        trainer.native_split_sha256 = "8" * 64
        trainer.native_group_ids = ("train-group",)
        trainer.native_panel_sha256 = None
        trainer.roborender_conditioning_contract = validate_conditioning_contract()
        trainer.roborender_prompt = "a robot moves a bottle into a tray"
        trainer.roborender_seed = 42
        trainer.roborender_reference_lag = 256
        trainer.env.config.simulator.config.cameras.conditioning_views = [
            {"name": name, "link": f"{name}_link", "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}
            for name in ("head_color", "hand_left", "hand_right")
        ]

        trainer.roborender_conditioning_slice = native_conditioning_slice
        policy = NumpyPolicy()
        runtime_sha = runtime_policy_sha256(
            policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
        )
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "conditioning/waves/wave-000"
            environment = {
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_CAMPAIGN_ID": "roborender-h200-v001",
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_COHORT_INDEX": "0",
                "ROUND_DAGGER_POLICY_SHA256": "1" * 64,
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
                "ROUND_DAGGER_TEACHER_SHA256": "3" * 64,
                "ROUND_DAGGER_BETA_SHA256": "5" * 64,
                "ROUND_DAGGER_CONFIG_SHA256": "6" * 64,
                "ROUND_DAGGER_ENVS_PER_SHARD": "1",
                "ROUND_DAGGER_ROLLOUT_ID": "0",
                "ROUND_DAGGER_PURPOSE": "train",
                "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
                "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256": "9" * 64,
                "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": "a" * 64,
                "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": "b" * 64,
                "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256": "c" * 64,
                "ROUND_DAGGER_GR00T_REVISION": "d" * 40,
                "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256": "e" * 64,
                "ROUND_DAGGER_GPU_NAME": "NVIDIA H200",
                "ROUND_DAGGER_COMPUTE_CAPABILITY": "9.0",
                "ROUND_DAGGER_GPU_COUNT": "8",
                "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": "[]",
                "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": "f" * 64,
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )
            self.assertEqual(len(reports), 2)
            self.assertTrue(all(report["learner_replay_ready"] is False for report in reports))
            sidecars = sorted((cohort / "shards").glob("*.json"))
            self.assertEqual(len(sidecars), 2)
            _, arrays = load_raw_conditioning_artifact(sidecars[0])
            self.assertNotIn("policy_rgb", arrays)
            self.assertIn("depth_ctrl__head_color", arrays)

    def test_export_slices_each_environment_part_before_host_transfer(self):
        trainer, _ = fake_trainer(steps=3, envs=5)
        transfers = []
        for key, value in trainer.storage.values.items():
            trainer.storage.values[key] = TransferProbe(
                value, key=key, transfers=transfers
            )
        policy = NumpyPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            cohort, environment, runtime_sha, _, _ = self._cohort_environment(
                temporary,
                trainer=trainer,
                policy=policy,
                environments_per_shard=2,
                campaign_id="bounded-export",
            )
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "round_dagger.actor.trainer_rollout_arrays",
                side_effect=AssertionError("full rollout conversion is forbidden"),
            ):
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )

            self.assertEqual(len(reports), 3)
            self.assertTrue(transfers)
            self.assertTrue(all(shape[1] <= 2 for _, shape in transfers))
            self.assertEqual(
                [shape[1] for key, shape in transfers if key == "vision_obs"],
                [2, 2, 1],
            )
            self.assertEqual(artifacts.seal_cohort(cohort)["transition_count"], 15)

    def test_streamed_export_is_byte_exact_with_full_rollout_split(self):
        trainer, _ = fake_trainer(steps=3, envs=5)
        policy = NumpyPolicy()
        complete = trainer_rollout_arrays(trainer)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_cohort, _, runtime_sha, identity, digests = (
                self._cohort_environment(
                    root / "reference",
                    trainer=trainer,
                    policy=policy,
                    environments_per_shard=2,
                    campaign_id="exact-export",
                )
            )
            reference_reports = []
            for part_index, start in enumerate(range(0, 5, 2)):
                stop = min(5, start + 2)
                reference_reports.append(
                    artifacts.publish_rollout_shard(
                        reference_cohort,
                        rank=0,
                        rollout_id=0,
                        part_index=part_index,
                        arrays={
                            key: np.ascontiguousarray(value[:, start:stop])
                            for key, value in complete.items()
                        },
                        **identity,
                        **digests,
                    )
                )

            _, environment, _, _, _ = self._cohort_environment(
                root / "streamed",
                trainer=trainer,
                policy=policy,
                environments_per_shard=2,
                campaign_id="exact-export",
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                streamed_reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )

            self.assertEqual(
                [report["payload_sha256"] for report in streamed_reports],
                [report["payload_sha256"] for report in reference_reports],
            )

    def test_storage_layout_failure_is_rejected_before_first_publish(self):
        trainer, _ = fake_trainer()
        trainer.storage.values["student_actions"] = trainer.storage.values[
            "student_actions"
        ][:, :-1]
        policy = NumpyPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            _, environment, runtime_sha, _, _ = self._cohort_environment(
                temporary,
                trainer=trainer,
                policy=policy,
                campaign_id="layout-failure",
            )
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "round_dagger.actor.publish_rollout_shard"
            ) as publish:
                with self.assertRaisesRegex(
                    ActorExportError, "disagree on environment dimension"
                ):
                    export_trainer_rollout(
                        trainer,
                        policy_module=policy,
                        initial_runtime_policy_sha256=runtime_sha,
                    )
            publish.assert_not_called()

    def test_eight_part_export_retry_skips_verified_prefix_and_completes(self):
        trainer, _ = fake_trainer(envs=32)
        policy = NumpyPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            cohort, environment, runtime_sha, _, _ = self._cohort_environment(
                temporary,
                trainer=trainer,
                policy=policy,
                environments_per_shard=4,
                campaign_id="retry-export",
            )
            actual_publish = artifacts.publish_rollout_shard
            calls = 0

            def fail_after_five_parts(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise OSError("injected export interruption")
                return actual_publish(*args, **kwargs)

            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "round_dagger.actor.publish_rollout_shard",
                side_effect=fail_after_five_parts,
            ):
                with self.assertRaisesRegex(OSError, "injected export interruption"):
                    export_trainer_rollout(
                        trainer,
                        policy_module=policy,
                        initial_runtime_policy_sha256=runtime_sha,
                    )

            sidecars = sorted((cohort / "shards").glob("*.json"))
            self.assertEqual(len(sidecars), 5)
            first_sidecar = cohort / "shards/rank-0000_rollout-00000000_part-0000.json"
            first_sha = artifacts.sha256_file(first_sidecar)
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "round_dagger.actor.publish_rollout_shard",
                wraps=actual_publish,
            ) as retry_publish:
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )
            self.assertEqual(len(reports), 8)
            self.assertEqual(retry_publish.call_count, 3)
            self.assertEqual(artifacts.sha256_file(first_sidecar), first_sha)
            self.assertEqual(artifacts.seal_cohort(cohort)["transition_count"], 96)

    def test_payload_only_crash_window_recovers_create_only_on_retry(self):
        trainer, _ = fake_trainer(envs=8)
        policy = NumpyPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            cohort, environment, runtime_sha, _, _ = self._cohort_environment(
                temporary,
                trainer=trainer,
                policy=policy,
                environments_per_shard=4,
                campaign_id="payload-only-retry",
            )
            actual_sidecar_publish = artifacts.publish_immutable_json
            target = "rank-0000_rollout-00000000_part-0001.json"

            def fail_before_second_sidecar(path, document):
                if Path(path).name == target:
                    raise OSError("injected sidecar crash")
                return actual_sidecar_publish(path, document)

            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "round_dagger.artifacts.publish_immutable_json",
                side_effect=fail_before_second_sidecar,
            ):
                with self.assertRaisesRegex(OSError, "injected sidecar crash"):
                    export_trainer_rollout(
                        trainer,
                        policy_module=policy,
                        initial_runtime_policy_sha256=runtime_sha,
                    )

            orphan = cohort / "shards/rank-0000_rollout-00000000_part-0001.npz"
            self.assertTrue(orphan.is_file())
            self.assertFalse((cohort / "shards" / target).exists())
            orphan_sha = artifacts.sha256_file(orphan)
            with mock.patch.dict(os.environ, environment, clear=False):
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )
            self.assertEqual(len(reports), 2)
            self.assertEqual(artifacts.sha256_file(orphan), orphan_sha)
            self.assertTrue((cohort / "shards" / target).is_file())
            self.assertEqual(artifacts.seal_cohort(cohort)["transition_count"], 24)

    def test_changed_policy_is_rejected_before_publish(self):
        trainer, _ = fake_trainer()
        policy = NumpyPolicy()
        before = runtime_policy_sha256(
            policy, homie_controller_sha256=HOMIE_CONTROLLER_SHA256
        )
        policy.bias += 1
        with self.assertRaisesRegex(ActorExportError, "changed during collection"):
            export_trainer_rollout(
                trainer,
                policy_module=policy,
                initial_runtime_policy_sha256=before,
            )


if __name__ == "__main__":
    unittest.main()
