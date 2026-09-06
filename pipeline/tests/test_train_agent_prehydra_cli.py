from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

from round_dagger.actor_identity import actor_process_identity


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py"


def _load_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "extract_pre_hydra_kit_args",
            "app_launcher_argv",
            "_publish_native_actor_trainer_success_marker",
        }
    ]
    namespace: dict[str, object] = {
        "os": os,
        "Path": Path,
        "actor_process_identity": actor_process_identity,
        "_NATIVE_ACTOR_TRAINER_SUCCESS_ENV": (
            "ROUND_DAGGER_TRAINER_SUCCESS_MARKER"
        ),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), namespace)
    return (
        namespace["extract_pre_hydra_kit_args"],
        namespace["app_launcher_argv"],
        namespace["_publish_native_actor_trainer_success_marker"],
    )


class TrainAgentPreHydraCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extract, launcher_argv, publish_success = _load_helpers()
        cls.extract = staticmethod(extract)
        cls.launcher_argv = staticmethod(launcher_argv)
        cls.publish_success = staticmethod(publish_success)

    def test_equals_form_removes_only_kit_payload(self):
        payload = "--portable --portable-root /sealed/cache --/log/file=/sealed/log"
        kit_args, hydra_argv = self.extract(
            ["train.py", f"--kit_args={payload}", "+exp=native", "num_envs=32"]
        )
        self.assertEqual(kit_args, payload)
        self.assertEqual(hydra_argv, ["train.py", "+exp=native", "num_envs=32"])

    def test_two_token_form_is_supported(self):
        kit_args, hydra_argv = self.extract(
            ["train.py", "--kit_args", "--portable --portable-root /sealed", "+exp=x"]
        )
        self.assertEqual(kit_args, "--portable --portable-root /sealed")
        self.assertEqual(hydra_argv, ["train.py", "+exp=x"])

    def test_native_actor_does_not_forward_hydra_overrides_to_kit(self):
        hydra_args = ["+exp=native", "num_envs=32", "checkpoint=/sealed/model.pt"]
        self.assertEqual(
            self.launcher_argv(
                "train.py", hydra_args, independent_actor=True
            ),
            ["train.py"],
        )

    def test_generic_launch_retains_unknown_arguments(self):
        hydra_args = ["+exp=generic", "num_envs=8"]
        self.assertEqual(
            self.launcher_argv(
                "train.py", hydra_args, independent_actor=False
            ),
            ["train.py", *hydra_args],
        )

    def test_absent_payload_leaves_hydra_argv_unchanged(self):
        argv = ["train.py", "+exp=x", "seed=42"]
        self.assertEqual(self.extract(argv), (None, argv))

    def test_duplicate_or_empty_payload_fails_closed(self):
        invalid = (
            ["train.py", "--kit_args="],
            ["train.py", "--kit_args"],
            ["train.py", "--kit_args=a", "--kit_args=b"],
            ["train.py", "--kit_args=a\n--unsafe"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.extract(argv)

    def test_native_trainer_success_proof_is_create_only_and_rank_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            process_state = Path(temporary).resolve() / "process-state"
            tmpdir = process_state / "tmp"
            tmpdir.mkdir(parents=True)
            physical_home = Path(temporary).resolve() / "physical-cache" / "home"
            physical_home.mkdir(parents=True)
            marker = process_state / "trainer-body.success"
            environment = {
                # HOME is deliberately hardware/runtime-scoped and reusable;
                # the create-only success proof must remain process-scoped.
                "HOME": str(physical_home),
                "TMPDIR": str(tmpdir),
                "ROUND_DAGGER_ACTOR_RANK": "0",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
                "ROUND_DAGGER_SATURATION_PHASE": "probe",
                "ROUND_DAGGER_TRAINER_SUCCESS_MARKER": str(marker),
            }
            expected = "NATIVE_ACTOR_TRAINER_BODY_OK rank=0 phase=probe"
            self.assertEqual(self.publish_success(environment), expected)
            self.assertEqual(marker.read_bytes(), (expected + "\n").encode("ascii"))
            self.assertEqual(marker.stat().st_mode & 0o777, 0o400)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.publish_success(environment)

    def test_native_trainer_success_proof_rejects_unbound_path_and_identity(self):
        self.assertIsNone(self.publish_success({}))
        with tempfile.TemporaryDirectory() as temporary:
            process_state = Path(temporary).resolve() / "process-state"
            tmpdir = process_state / "tmp"
            tmpdir.mkdir(parents=True)
            valid = {
                "TMPDIR": str(tmpdir),
                "ROUND_DAGGER_ACTOR_RANK": "0",
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "1",
                "ROUND_DAGGER_SATURATION_PHASE": "smoke",
                "ROUND_DAGGER_TRAINER_SUCCESS_MARKER": str(
                    process_state / "trainer-body.success"
                ),
            }
            corruptions = (
                {**valid, "ROUND_DAGGER_ACTOR_RANK": "1"},
                {**valid, "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2"},
                {
                    key: value
                    for key, value in valid.items()
                    if key != "ROUND_DAGGER_ACTOR_WORLD_SIZE"
                },
                {**valid, "ROUND_DAGGER_SATURATION_PHASE": "validation"},
                {
                    **valid,
                    "ROUND_DAGGER_TRAINER_SUCCESS_MARKER": str(
                        process_state / "wrong.success"
                    ),
                },
                {**valid, "TMPDIR": str(process_state)},
            )
            for environment in corruptions:
                with self.subTest(environment=environment), self.assertRaises(
                    RuntimeError
                ):
                    self.publish_success(environment)

    def test_native_rank_device_reaches_accelerator_kit_environment_and_policy(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        for marker in (
            "native_actor_cuda_device()",
            "torch.cuda.set_device(0)",
            'os.environ["ACCELERATE_TORCH_DEVICE"] = native_device',
            "device != native_device",
            "args_cli.device = device",
            'launcher_args["active_gpu"] = 0',
            'launcher_args["physics_gpu"] = 0',
            "CUDA_VISIBLE_DEVICES is deliberately absent",
            "custom_instantiate(config.env, device=device",
            ").to(device)",
        ):
            self.assertIn(marker, source)

    def test_native_actor_defers_and_explicitly_creates_stage_before_ready(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        markers = (
            'self._sim_app_config["create_new_stage"] = False',
            "simulation_lifecycle.attach(simulation_app)",
            "if usd_context.get_stage() is not None:",
            "usd_context.new_stage()",
            "if usd_context.get_stage() is None:",
            '"NATIVE_ACTOR_KIT_READY "',
        )
        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
