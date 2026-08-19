import ast
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import install_gr00t_integration as installer


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
)
GR00T = ROOT / "GR00T-VisualSim2Real"
EXPECTED_VIEWS = ("head_color", "hand_left", "hand_right")


def load_contract_helpers():
    """Load dependency-free gate helpers without importing Isaac/GR00T."""
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    names = {"_parse_contract_csv", "_enforce_policy_view_contract"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError("three-view production contract helpers are missing")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["_enforce_policy_view_contract"]


def load_functions(names, namespace):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"trainer helpers are missing: {names}")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace


def load_trainer_method(name, namespace):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRLDistillTrainerVideoGen"
    )
    method = next(
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace[name]


class FakeArray:
    def __init__(self, payload):
        self.payload = bytes(payload)

    def tobytes(self, order="C"):
        return self.payload


class FakeParameter:
    def __init__(self, payload, dtype="torch.float32"):
        self.payload = bytes(payload)
        self.dtype = dtype
        self.shape = (len(self.payload),)

    def detach(self):
        return self

    def contiguous(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def reshape(self, *args):
        return self

    def view(self, *args):
        return self

    def numpy(self):
        return FakeArray(self.payload)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def reshape(self, *args):
        return self

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def tolist(self):
        return list(self.values)


class FakeTorch:
    uint8 = "uint8"
    float64 = "float64"

    @staticmethod
    def tensor(values, **kwargs):
        return FakeTensor(values)

    @staticmethod
    def min(tensor):
        return SimpleNamespace(item=lambda: min(tensor.values))


class FakeAccelerator:
    def __init__(
        self,
        *,
        mismatch=False,
        world_size=2,
        process_index=0,
        gathered_losses=None,
    ):
        self.device = "fake-device"
        self.num_processes = world_size
        self.process_index = process_index
        self.mismatch = mismatch
        self.gathered_losses = gathered_losses
        self.gather_calls = 0

    def unwrap_model(self, model):
        return model

    def gather(self, tensor):
        self.gather_calls += 1
        if len(tensor.values) == hashlib.sha256().digest_size:
            rows = [list(tensor.values) for _ in range(self.num_processes)]
            if self.mismatch:
                rows[-1][0] ^= 1
            return FakeTensor([value for row in rows for value in row])
        if tensor.values == [1]:
            return FakeTensor([1] * self.num_processes)
        if self.gathered_losses is not None:
            return FakeTensor(self.gathered_losses)
        return FakeTensor(tensor.values * self.num_processes)


class ThreeViewPolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enforce = staticmethod(load_contract_helpers())

    def call(self, **overrides):
        values = {
            "actual_views": EXPECTED_VIEWS,
            "configured_count": 3,
            "actual_layout": "vertical",
            "actual_shape": (324, 192, 3),
            "expected_views": "head_color,hand_left,hand_right",
            "expected_count": "3",
            "expected_layout": "vertical",
            "expected_shape": "324,192,3",
        }
        values.update(overrides)
        return self.enforce(**values)

    def test_exact_three_view_contract_passes(self):
        self.call()

    def test_unset_environment_gates_preserve_nonproduction_behavior(self):
        self.call(
            actual_views=("head_color",),
            configured_count=1,
            actual_layout=None,
            actual_shape=(108, 192, 3),
            expected_views=None,
            expected_count=None,
            expected_layout=None,
            expected_shape=None,
        )

    def test_order_count_layout_and_shape_mismatches_fail_closed(self):
        cases = (
            ({"actual_views": tuple(reversed(EXPECTED_VIEWS))}, "policy views mismatch"),
            ({"configured_count": 2}, "policy view count mismatch"),
            ({"actual_layout": "horizontal"}, "policy view layout mismatch"),
            ({"actual_shape": (324, 191, 3)}, "policy image shape mismatch"),
            ({"expected_shape": "324x192x3"}, "invalid policy image shape"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, message):
                self.call(**changes)

    def test_installer_validates_exact_three_view_config_and_trainer_markers(self):
        report = installer.inspect(
            GR00T,
            ROOT / "integrations/gr00t/image3f_frame_client.py",
            ROOT / "integrations/gr00t/dagger_telemetry.py",
        )
        self.assertTrue(report["three_view_config"])
        self.assertEqual(report["policy_views"], EXPECTED_VIEWS)
        self.assertEqual(report["renderer_views"], EXPECTED_VIEWS)
        self.assertEqual(report["policy_view_count"], 3)
        self.assertEqual(report["policy_view_layout"], "vertical")
        self.assertEqual(report["policy_image_shape"], (324, 192, 3))
        self.assertTrue(report["trainer_runtime_gates_integrated"])
        self.assertTrue(report["trainer_ddp_sync_gate_integrated"])
        self.assertTrue(report["trainer_finite_bc_loss_gate_integrated"])
        self.assertEqual(installer.validate_base(report, allow_dirty=True), [])
        self.assertEqual(installer.validate_trainer_integration(report), [])

    def test_sim_rgb_camera_diagnostic_is_explicitly_opt_in(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"IMAGE_DAGGER_DUMP_SIM_RGB", "0"', source)
        self.assertIn('os.environ.get("IMAGE_DAGGER_DUMP_MAX_STEPS")', source)
        self.assertIn('os.environ.get("IMAGE_DAGGER_DUMP_MAX_ENVS")', source)
        self.assertIn("len(self._id_gen_ms) < self._id_dump_max_steps", source)
        self.assertIn("gen_envs[: self._id_dump_max_envs]", source)
        self.assertIn('sim_rgb = views[v].get("rgb")', source)
        self.assertIn("[depth_row, sim_row, generated_row]", source)

    def test_camera_diagnostic_bounds_default_and_validate(self):
        namespace = load_functions({"_parse_dump_bounds"}, {})
        parse = namespace["_parse_dump_bounds"]
        self.assertEqual(parse(), (15, None))
        self.assertEqual(parse("2", "1"), (2, 1))
        self.assertEqual(parse("0", "0"), (0, 0))
        for values in (("-1", None), ("1", "-1"), ("x", "1"), ("1", "x")):
            with self.subTest(values=values), self.assertRaisesRegex(
                RuntimeError, "must be non-negative integers"
            ):
                parse(*values)

    def test_installer_rejects_ambiguous_yaml_contract_values(self):
        self.assertIsNone(installer.yaml_inline_list("policy_views: ['head_color']\n", "policy_views"))
        self.assertIsNone(
            installer.yaml_inline_list(
                "policy_views: [head_color]\npolicy_views: [hand_left]\n",
                "policy_views",
            )
        )
        self.assertIsNone(installer.yaml_plain_scalar("policy_view_layout: 'vertical'\n", "policy_view_layout"))

    def test_installer_detects_missing_finite_bc_loss_gate_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            gr00t = Path(temporary)
            files = {
                "gr00t/rl/trl/trainer/distill_trainer_videogen.py": TRAINER,
                "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger_3view.yaml": (
                    GR00T
                    / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger_3view.yaml"
                ),
                "gr00t/rl/utils/image3f_frame_client.py": (
                    ROOT / "integrations/gr00t/image3f_frame_client.py"
                ),
                "gr00t/rl/utils/dagger_telemetry.py": (
                    ROOT / "integrations/gr00t/dagger_telemetry.py"
                ),
            }
            for relative, source in files.items():
                target = gr00t / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                document = source.read_text(encoding="utf-8")
                if source == TRAINER:
                    document = document.replace(
                        "FAST_ROBORENDER_FINITE_BC_LOSS_GATE_V1",
                        "REMOVED_FINITE_BC_LOSS_GATE",
                        1,
                    )
                target.write_text(document, encoding="utf-8")

            report = installer.inspect(
                gr00t,
                ROOT / "integrations/gr00t/image3f_frame_client.py",
                ROOT / "integrations/gr00t/dagger_telemetry.py",
            )
            self.assertFalse(report["trainer_finite_bc_loss_gate_integrated"])
            self.assertIn(
                installer.FINITE_BC_LOSS_MARKER,
                report["missing_trainer_finite_bc_loss_gate_markers"],
            )
            self.assertIn(
                "finite/nonnegative DAgger BC loss gate is not installed",
                installer.validate_trainer_integration(report),
            )


class DdpPolicySyncGateTest(unittest.TestCase):
    def helper_namespace(self):
        return load_functions(
            {
                "_sha256_named_parameters",
                "_decode_rank_digests",
                "_parse_dump_bounds",
                "_atomic_write_json",
            },
            {
                "hashlib": hashlib,
                "json": json,
                "os": os,
                "tempfile": tempfile,
                "Path": Path,
                "torch": FakeTorch,
            },
        )

    def test_parameter_digest_is_order_stable_and_bit_sensitive(self):
        namespace = self.helper_namespace()
        sha = namespace["_sha256_named_parameters"]
        a = FakeParameter(b"aaaa")
        b = FakeParameter(b"bbbb")
        digest_one, count = sha([("b", b), ("a", a)])
        digest_two, _ = sha([("a", a), ("b", b)])
        digest_changed, _ = sha([("a", FakeParameter(b"aaab")), ("b", b)])
        self.assertEqual(count, 2)
        self.assertEqual(digest_one, digest_two)
        self.assertNotEqual(digest_one, digest_changed)
        with self.assertRaisesRegex(RuntimeError, "no policy parameters"):
            sha([])

    def test_digest_decoder_fails_on_size_or_cross_rank_mismatch(self):
        decode = self.helper_namespace()["_decode_rank_digests"]
        digest = bytes(range(32))
        self.assertEqual(decode(digest + digest, 2), (digest.hex(), digest.hex()))
        with self.assertRaisesRegex(RuntimeError, "invalid digest payload"):
            decode(digest, 2)
        different = bytes([digest[0] ^ 1]) + digest[1:]
        with self.assertRaisesRegex(RuntimeError, "differ across ranks"):
            decode(digest + different, 2)

    def build_verifier(self, namespace):
        namespace.update(
            {
                "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
                "torch": FakeTorch,
            }
        )
        return load_trainer_method("_verify_ddp_policy_parameter_sync", namespace)

    def fake_trainer(self, accelerator):
        policy = SimpleNamespace(named_parameters=lambda: [])
        return SimpleNamespace(
            _ddp_sync_verified=False,
            _ddp_sync_digest=None,
            _ddp_sync_parameter_count=0,
            accelerator=accelerator,
            model=SimpleNamespace(policy=policy),
            state=SimpleNamespace(global_step=0),
        )

    def test_opt_in_verifier_gathers_writes_marker_and_runs_only_once(self):
        namespace = self.helper_namespace()
        namespace["_sha256_named_parameters"] = lambda parameters: ("00" * 32, 7)
        verify = self.build_verifier(namespace)
        accelerator = FakeAccelerator()
        trainer = self.fake_trainer(accelerator)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"TRAIN_RUN_DIR": temporary}, clear=False
        ):
            metrics = verify(trainer)
            marker_path = (
                Path(temporary) / "validation/ddp_policy_sync_step_000000001.json"
            )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertTrue(marker["verified"])
            self.assertEqual(marker["world_size"], 2)
            self.assertEqual(marker["policy_parameter_count"], 7)
            self.assertEqual(marker["policy_parameters_sha256"], "00" * 32)
            self.assertEqual(metrics["production/ddp_policy_sync_verified"], 1.0)
            self.assertEqual(accelerator.gather_calls, 2)
            verify(trainer)
            self.assertEqual(accelerator.gather_calls, 2)
            self.assertFalse(any(path.name.startswith(".") for path in marker_path.parent.iterdir()))

    def test_opt_in_verifier_fails_on_rank_divergence_without_marker(self):
        namespace = self.helper_namespace()
        namespace["_sha256_named_parameters"] = lambda parameters: ("00" * 32, 7)
        verify = self.build_verifier(namespace)
        trainer = self.fake_trainer(FakeAccelerator(mismatch=True))
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"TRAIN_RUN_DIR": temporary}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "differ across ranks"):
                verify(trainer)
            self.assertFalse((Path(temporary) / "validation").exists())


class FiniteBcLossGateTest(unittest.TestCase):
    def helper_namespace(self):
        return load_functions(
            {
                "_parse_boolean_gate",
                "_finite_nonnegative_loss_mean",
                "_atomic_write_json",
            },
            {
                "json": json,
                "math": math,
                "os": os,
                "tempfile": tempfile,
                "Path": Path,
            },
        )

    def build_verifier(self, namespace):
        namespace.update(
            {
                "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
                "torch": FakeTorch,
            }
        )
        return load_trainer_method("_verify_dagger_bc_loss", namespace)

    @staticmethod
    def fake_trainer(accelerator, losses=(0.25,), step=0):
        return SimpleNamespace(
            accelerator=accelerator,
            dagger_bc_loss_stats=FakeTensor(losses),
            state=SimpleNamespace(global_step=step),
        )

    def test_boolean_gate_is_explicit_and_rejects_ambiguous_values(self):
        parse = self.helper_namespace()["_parse_boolean_gate"]
        self.assertFalse(parse(None, "GATE"))
        for value in ("1", "true", "YES"):
            with self.subTest(value=value):
                self.assertTrue(parse(value, "GATE"))
        for value in ("0", "false", "NO"):
            with self.subTest(value=value):
                self.assertFalse(parse(value, "GATE"))
        with self.assertRaisesRegex(RuntimeError, "GATE must be a boolean value"):
            parse("enabled", "GATE")

    def test_loss_helper_checks_every_value_not_only_the_mean(self):
        validate = self.helper_namespace()["_finite_nonnegative_loss_mean"]
        self.assertEqual(validate([0.0, 0.25, 0.5]), 0.25)
        for values, message in (
            ([], "no loss values"),
            ([0.1, float("nan")], "non-finite"),
            ([0.1, float("inf")], "non-finite"),
            ([0.1, -0.01], "negative loss"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(
                RuntimeError, message
            ):
                validate(values)

    def test_rank_safe_gate_writes_atomic_marker_for_every_update(self):
        namespace = self.helper_namespace()
        verify = self.build_verifier(namespace)
        accelerator = FakeAccelerator(world_size=2)
        trainer = self.fake_trainer(accelerator)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"TRAIN_RUN_DIR": temporary}, clear=False
        ):
            first_metrics = verify(trainer)
            trainer.state.global_step = 1
            trainer.dagger_bc_loss_stats = FakeTensor([0.5])
            second_metrics = verify(trainer)

            validation = Path(temporary) / "validation"
            first_path = validation / "dagger_bc_loss_step_000000001.json"
            second_path = validation / "dagger_bc_loss_step_000000002.json"
            first = json.loads(first_path.read_text(encoding="utf-8"))
            second = json.loads(second_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(first),
                {
                    "schema",
                    "update",
                    "world_size",
                    "loss",
                    "finite_verified",
                    "nonnegative_verified",
                },
            )
            self.assertEqual(first["schema"], "fast_roborender.dagger_bc_loss.v1")
            self.assertEqual(first["update"], 1)
            self.assertEqual(first["world_size"], 2)
            self.assertEqual(first["loss"], 0.25)
            self.assertTrue(first["finite_verified"])
            self.assertTrue(first["nonnegative_verified"])
            self.assertEqual(second["update"], 2)
            self.assertEqual(second["loss"], 0.5)
            self.assertEqual(
                first_metrics["production/dagger_bc_loss_finite_verified"], 1.0
            )
            self.assertEqual(second_metrics["production/dagger_bc_loss"], 0.5)
            self.assertEqual(accelerator.gather_calls, 4)
            self.assertFalse(
                any(path.name.startswith(".") for path in validation.iterdir())
            )

    def test_cross_rank_nonfinite_or_negative_loss_fails_without_marker(self):
        namespace = self.helper_namespace()
        verify = self.build_verifier(namespace)
        for remote_loss, message in (
            (float("nan"), "non-finite"),
            (-0.01, "negative loss"),
        ):
            with self.subTest(remote_loss=remote_loss), tempfile.TemporaryDirectory() as temporary:
                accelerator = FakeAccelerator(
                    world_size=2, gathered_losses=[0.25, remote_loss]
                )
                trainer = self.fake_trainer(accelerator)
                with mock.patch.dict(
                    os.environ, {"TRAIN_RUN_DIR": temporary}, clear=False
                ), self.assertRaisesRegex(RuntimeError, message):
                    verify(trainer)
                self.assertFalse((Path(temporary) / "validation").exists())

    def test_gate_requires_absolute_run_directory(self):
        namespace = self.helper_namespace()
        verify = self.build_verifier(namespace)
        trainer = self.fake_trainer(FakeAccelerator())
        with mock.patch.dict(
            os.environ, {"TRAIN_RUN_DIR": "relative/run"}, clear=False
        ), self.assertRaisesRegex(RuntimeError, "absolute TRAIN_RUN_DIR"):
            verify(trainer)


if __name__ == "__main__":
    unittest.main()
