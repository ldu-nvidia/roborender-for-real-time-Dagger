import ast
import sys
import tempfile
import types
import unittest
from collections import UserList
from pathlib import Path
from unittest import mock

from round_dagger.artifacts import sha256_file
from round_dagger.gr00t_policy import (
    ADAPTER_ID,
    BASE_MODULE_TARGET,
    POLICY_SPEC_SCHEMA,
    POLICY_TARGET,
    GR00TPolicyError,
    build_portable_policy_spec,
    create_vision_actor_obj_pred,
    load_checkpoint_strict,
    validate_policy_spec,
    vision_actor_obj_pred_adapter_factory,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256


ROOT = Path(__file__).resolve().parents[1]
GR00T_ROOT = ROOT / "GR00T-VisualSim2Real"
TRAINER = GR00T_ROOT / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
POLICY_STATE_SHA256 = "2" * 64
HOMIE_CONTROLLER_SHA256 = {"walk": "3" * 64, "stand": "4" * 64}


def _actor_config(pretrained=True):
    return {
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
                        "pretrained": pretrained,
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


class _FakeHead:
    output_dim = 3


VisionActorObjPred = type("VisionActorObjPred", (), {})
VisionActorObjPred.__module__ = (
    "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"
)


def _fake_policy():
    value = VisionActorObjPred()
    value.num_actions = 11
    value.obj_pred_mlp = _FakeHead()
    return value


def _spec():
    return build_portable_policy_spec(
        policy=_fake_policy(),
        actor_config=_actor_config(pretrained=True),
        algo_config={
            "init_noise_std": 0.3,
            "freeze_noise_std": True,
            "clamp_noise_std": False,
        },
        module_dim_dict={"vision_feature_dim": 128, "unused_teacher_feature_dim": 512},
        env_config=types.SimpleNamespace(
            robot=types.SimpleNamespace(
                algo_obs_dim_dict={
                    "actor_obs": 113,
                    "vision_obs": 324 * 192 * 3,
                    "teacher_obs": 901,
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
        checkpoint_sha256="1" * 64,
        policy_state_sha256=POLICY_STATE_SHA256,
        homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
    )


def _bundle(spec):
    return {
        "schema": "fast_roborender.round_dagger.learner_bundle.v1",
        "model_config": spec,
        "rgb_mean": spec["input_contract"]["rgb_mean"],
        "rgb_std": spec["input_contract"]["rgb_std"],
        "rgb_layout": "NHWC",
        "train_action_slice": [0, 11],
        "initial_checkpoint_sha256": spec["checkpoint_sha256"],
    }


class PortablePolicySpecTest(unittest.TestCase):
    def test_runtime_export_preserves_non_builtin_sequence_configs(self):
        actor = _actor_config(pretrained=True)
        for name in ("vision_module", "mlp_module", "obj_pred_mlp"):
            module = actor["backbone"][name]["module_config_dict"]
            module["input_dim"] = UserList(module["input_dim"])
            module["output_dim"] = UserList(module["output_dim"])
            hidden = module["layer_config"].get("hidden_dims")
            if hidden is not None:
                module["layer_config"]["hidden_dims"] = UserList(hidden)
        spec = build_portable_policy_spec(
            policy=_fake_policy(),
            actor_config=actor,
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
            checkpoint_sha256="1" * 64,
            policy_state_sha256=POLICY_STATE_SHA256,
            homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
        )
        backbone = spec["actor_config"]["backbone"]
        self.assertEqual(
            backbone["mlp_module"]["module_config_dict"]["layer_config"][
                "hidden_dims"
            ],
            [512, 256, 128],
        )
        self.assertEqual(
            backbone["obj_pred_mlp"]["module_config_dict"]["output_dim"], [3]
        )

    def test_runtime_export_is_resolved_offline_and_binds_all_dimensions(self):
        spec = _spec()
        self.assertEqual(spec["schema"], POLICY_SPEC_SCHEMA)
        self.assertEqual(spec["policy_state_sha256"], POLICY_STATE_SHA256)
        self.assertEqual(
            spec["homie_controller_sha256"], HOMIE_CONTROLLER_SHA256
        )
        self.assertEqual(
            spec["runtime_policy_sha256"],
            runtime_policy_identity_sha256(
                POLICY_STATE_SHA256, HOMIE_CONTROLLER_SHA256
            ),
        )
        self.assertEqual(spec["adapter_id"], ADAPTER_ID)
        layer = spec["actor_config"]["backbone"]["vision_module"]["module_config_dict"][
            "layer_config"
        ]
        self.assertFalse(layer["pretrained"])
        self.assertTrue(
            spec["initialization_contract"]["source_actor_requested_pretrained"]
        )
        self.assertEqual(
            spec["initialization_contract"]["frozen_parameter_groups"],
            ["obj_pred_mlp", "std"],
        )
        self.assertEqual(spec["input_contract"]["vision_shape"], [324, 192, 3])
        self.assertEqual(spec["input_contract"]["actor_obs_dim"], 113)
        self.assertEqual(
            spec["env_config"]["robot"]["algo_obs_dim_dict"],
            {"actor_obs": 113, "vision_obs": 324 * 192 * 3},
        )
        self.assertEqual(spec["module_dim_dict"], {"vision_feature_dim": 128})
        self.assertEqual(
            spec["actor_config"]["backbone"]["mlp_module"]["module_config_dict"][
                "output_dim"
            ],
            [11],
        )
        self.assertEqual(
            spec["output_contract"],
            {
                "policy_action_dim": 11,
                "homie_action_dim": 15,
                "replay_action_dim": 26,
                "env_command_dim": 31,
                "train_action_slice": [0, 11],
                "object_prediction_dim": 3,
            },
        )
        self.assertEqual(validate_policy_spec(spec), spec)

    def test_spec_rejects_downloads_camera_or_action_contract_drift(self):
        cases = []
        pretrained = _spec()
        pretrained["actor_config"]["backbone"]["vision_module"]["module_config_dict"][
            "layer_config"
        ]["pretrained"] = True
        cases.append(pretrained)
        wrong_image = _spec()
        wrong_image["input_contract"]["vision_shape"] = [108, 192, 3]
        cases.append(wrong_image)
        wrong_action = _spec()
        wrong_action["output_contract"]["replay_action_dim"] = 31
        cases.append(wrong_action)
        malformed_actor = _spec()
        del malformed_actor["actor_config"]["backbone"]["vision_module"][
            "module_config_dict"
        ]
        cases.append(malformed_actor)
        nonfinite_noise = _spec()
        nonfinite_noise["algo_config"]["init_noise_std"] = float("nan")
        cases.append(nonfinite_noise)
        wrong_flat_vision = _spec()
        wrong_flat_vision["env_config"]["robot"]["algo_obs_dim_dict"]["vision_obs"] -= 1
        cases.append(wrong_flat_vision)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(GR00TPolicyError):
                validate_policy_spec(candidate)

    def test_spec_rejects_unbound_or_incomplete_controller_lineage(self):
        mismatched_runtime = _spec()
        mismatched_runtime["homie_controller_sha256"]["walk"] = "9" * 64
        missing_controller = _spec()
        del missing_controller["homie_controller_sha256"]["stand"]
        missing_component = _spec()
        del missing_component["policy_state_sha256"]
        for candidate in (
            mismatched_runtime,
            missing_controller,
            missing_component,
        ):
            with self.subTest(candidate=candidate), self.assertRaises(
                GR00TPolicyError
            ):
                validate_policy_spec(candidate)

    def test_bundle_must_bind_normalization_slice_and_checkpoint(self):
        spec = _spec()
        for field, value in (
            ("rgb_mean", [0.0, 0.0, 0.0]),
            ("train_action_slice", [0, 10]),
            ("initial_checkpoint_sha256", "f" * 64),
        ):
            bundle = _bundle(spec)
            bundle[field] = value
            with self.subTest(field=field), self.assertRaises(GR00TPolicyError):
                create_vision_actor_obj_pred(bundle)

    def test_h100_module_has_no_simulator_import_and_initialize_skips_image3f(self):
        source = (ROOT / "round_dagger/gr00t_policy.py").read_text()
        tree = ast.parse(source)
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith(("omni", "isaaclab"))
                )
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                ("omni", "isaaclab")
            ):
                forbidden.append(node.module)
        self.assertEqual(forbidden, [])
        trainer = TRAINER.read_text()
        self.assertIn("and not self._round_initialize_only", trainer)
        self.assertIn("build_portable_policy_spec", trainer)
        self.assertIn('destination.suffix + ".policy_spec.json"', trainer)
        self.assertIn(
            "homie_action_dim=int(self.homie_walk_model.num_actions)", trainer
        )
        self.assertIn("replay_action_dim=int(self.num_act)", trainer)
        self.assertIn("str(staged_checkpoint)", trainer)
        self.assertIn("staged.chmod(0o444)", trainer)
        checkpoint_stage = trainer.index("(staged_checkpoint, destination)")
        spec_stage = trainer.index("(staged_spec, policy_spec_path)")
        manifest_commit = trainer.index("(staged_manifest, manifest_path)")
        self.assertLess(checkpoint_stage, spec_stage)
        self.assertLess(spec_stage, manifest_commit)
        image3f_block = trainer.index("from gr00t.rl.utils.image3f_frame_client")
        initialize_guard = trainer.rfind(
            "not self._round_initialize_only", 0, image3f_block
        )
        self.assertGreater(initialize_guard, 0)


try:
    import hydra  # noqa: F401
    import omegaconf  # noqa: F401
    import torch
    import torchvision

    ACTUAL_RUNTIME = True
except ImportError:
    ACTUAL_RUNTIME = False


@unittest.skipUnless(ACTUAL_RUNTIME, "requires torch, torchvision, Hydra and OmegaConf")
class ActualGR00TPolicyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(GR00T_ROOT))

    @classmethod
    def tearDownClass(cls):
        try:
            sys.path.remove(str(GR00T_ROOT))
        except ValueError:
            pass

    def test_actual_policy_constructs_offline_forwards_and_loads_strictly(self):
        spec = _spec()
        bundle = _bundle(spec)
        original = torchvision.models.resnet18
        constructor_flags = []

        def guarded_resnet18(*args, **kwargs):
            constructor_flags.append(kwargs.get("pretrained"))
            if kwargs.get("pretrained") is not False:
                raise AssertionError(
                    "ResNet factory attempted a pretrained/network path"
                )
            return original(*args, **kwargs)

        torch.manual_seed(123)
        with mock.patch.object(torchvision.models, "resnet18", guarded_resnet18):
            policy = create_vision_actor_obj_pred(bundle)
        self.assertEqual(constructor_flags, [False])
        simulator_modules = sorted(
            name
            for name in sys.modules
            if name == "omni"
            or name.startswith("omni.")
            or name == "isaaclab"
            or name.startswith("isaaclab.")
        )
        self.assertEqual(simulator_modules, [])
        self.assertEqual(policy.num_actions, 11)
        self.assertFalse(policy.std.requires_grad)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in policy.obj_pred_mlp.parameters()
            )
        )
        runtime_export = build_portable_policy_spec(
            policy=policy,
            actor_config=spec["actor_config"],
            algo_config=spec["algo_config"],
            module_dim_dict=spec["module_dim_dict"],
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
            checkpoint_sha256="1" * 64,
            policy_state_sha256=POLICY_STATE_SHA256,
            homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
        )
        self.assertEqual(runtime_export["actor_config"], spec["actor_config"])
        policy.eval()
        adapter = vision_actor_obj_pred_adapter_factory(bundle)
        with torch.no_grad():
            actions = adapter.actions(
                policy,
                {
                    "actor_obs": torch.zeros((1, 113), dtype=torch.float32),
                    "vision_obs": torch.zeros((1, 324, 192, 3), dtype=torch.float32),
                },
            )
        self.assertEqual(tuple(actions.shape), (1, 11))

        policy.train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1.0e-4)
        train_observations = {
            "actor_obs": torch.randn((2, 113), dtype=torch.float32),
            "vision_obs": torch.randn((2, 324, 192, 3), dtype=torch.float32),
        }
        target = torch.randn((2, 11), dtype=torch.float32)
        before = policy.mlp_module.module[-1].weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        prediction = adapter.actions(policy, train_observations)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for parameter in policy.obj_pred_mlp.parameters():
            self.assertIsNone(parameter.grad)
        optimizer.step()
        self.assertFalse(
            torch.equal(before, policy.mlp_module.module[-1].weight.detach())
        )
        policy.eval()

        temp_root = Path("/mnt/data1/ldu/tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            checkpoint = Path(temporary) / "policy.pt"
            torch.save({"policy_state_dict": policy.state_dict()}, checkpoint)
            symlink = Path(temporary) / "policy-link.pt"
            symlink.symlink_to(checkpoint)
            with self.assertRaisesRegex(GR00TPolicyError, "non-symlink"):
                load_checkpoint_strict(
                    policy,
                    symlink,
                    expected_sha256=sha256_file(checkpoint),
                )
            torch.manual_seed(999)
            with mock.patch.object(torchvision.models, "resnet18", guarded_resnet18):
                restored = create_vision_actor_obj_pred(bundle)
            load_checkpoint_strict(
                restored, checkpoint, expected_sha256=sha256_file(checkpoint)
            )
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(
                    value, restored.state_dict()[name], rtol=0, atol=0
                )

            damaged = dict(policy.state_dict())
            damaged.pop(next(iter(damaged)))
            damaged_path = Path(temporary) / "damaged.pt"
            torch.save({"policy_state_dict": damaged}, damaged_path)
            with self.assertRaisesRegex(GR00TPolicyError, "strict"):
                load_checkpoint_strict(
                    restored,
                    damaged_path,
                    expected_sha256=sha256_file(damaged_path),
                )


if __name__ == "__main__":
    unittest.main()
