import ast
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_PATH = ROOT / (
    "GR00T-VisualSim2Real/gr00t/rl/utils/simulation_app_lifecycle.py"
)
TRAIN_PATH = ROOT / "GR00T-VisualSim2Real/gr00t/rl/train_agent_trl.py"


def _load_lifecycle_module():
    spec = importlib.util.spec_from_file_location(
        "simulation_app_lifecycle_under_test", LIFECYCLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeApp:
    def __init__(self, close_error=None):
        self.close_calls = 0
        self.close_kwargs = []
        self.close_error = close_error

    def close(self, **kwargs):
        self.close_calls += 1
        self.close_kwargs.append(kwargs)
        if self.close_error is not None:
            raise self.close_error


class SimulationAppLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_lifecycle_module()
        self.environment = mock.patch.dict(
            self.module.os.environ,
            {"IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP": "0"},
        )
        self.previous_handler = object()
        self.installed_handlers = []
        self.getsignal = mock.patch.object(
            self.module.signal,
            "getsignal",
            return_value=self.previous_handler,
        )
        self.signal = mock.patch.object(
            self.module.signal,
            "signal",
            side_effect=lambda _signum, handler: self.installed_handlers.append(handler),
        )
        self.environment.start()
        self.getsignal.start()
        self.signal.start()

    def tearDown(self):
        self.signal.stop()
        self.getsignal.stop()
        self.environment.stop()

    def test_success_closes_once_and_restores_sigterm_handler(self):
        app = _FakeApp()
        with self.module.SimulationAppLifecycle() as lifecycle:
            lifecycle.attach(app)

        self.assertEqual(app.close_calls, 1)
        self.assertEqual(app.close_kwargs, [{}])
        self.assertEqual(len(self.installed_handlers), 2)
        self.assertIs(self.installed_handlers[-1], self.previous_handler)

    def test_opt_in_uses_isaac_immediate_close(self):
        app = _FakeApp()
        with mock.patch.dict(
            self.module.os.environ,
            {"IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP": "1"},
        ):
            with self.module.SimulationAppLifecycle() as lifecycle:
                lifecycle.attach(app)

        self.assertEqual(app.close_calls, 1)
        self.assertEqual(app.close_kwargs, [{"skip_cleanup": True}])

    def test_invalid_opt_in_value_fails_closed_before_app_launch(self):
        with mock.patch.dict(
            self.module.os.environ,
            {"IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP": "true"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "IMAGE_DAGGER_SIMULATION_APP_SKIP_CLEANUP must be exactly 0 or 1",
            ):
                self.module.SimulationAppLifecycle()

    def test_body_error_wins_over_close_error(self):
        app = _FakeApp(close_error=OSError("close failed"))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with self.module.SimulationAppLifecycle() as lifecycle:
                    lifecycle.attach(app)
                    raise RuntimeError("training failed")

        self.assertEqual(app.close_calls, 1)
        self.assertIn("close failed", stderr.getvalue())
        self.assertIs(self.installed_handlers[-1], self.previous_handler)

    def test_close_error_is_propagated_after_success(self):
        app = _FakeApp(close_error=OSError("close failed"))
        with self.assertRaisesRegex(OSError, "close failed"):
            with self.module.SimulationAppLifecycle() as lifecycle:
                lifecycle.attach(app)

        self.assertEqual(app.close_calls, 1)
        self.assertIs(self.installed_handlers[-1], self.previous_handler)

    def test_sigterm_unwinds_through_close_and_preserves_exit_status(self):
        app = _FakeApp()
        with self.assertRaises(SystemExit) as raised:
            with self.module.SimulationAppLifecycle() as lifecycle:
                lifecycle.attach(app)
                installed_sigterm_handler = self.installed_handlers[-1]
                installed_sigterm_handler(self.module.signal.SIGTERM, None)

        self.assertEqual(raised.exception.code, 128 + self.module.signal.SIGTERM)
        self.assertEqual(app.close_calls, 1)
        self.assertIs(self.installed_handlers[-1], self.previous_handler)

    def test_lifecycle_rejects_duplicate_ownership_and_reuse(self):
        app = _FakeApp()
        lifecycle = self.module.SimulationAppLifecycle()
        with lifecycle:
            lifecycle.attach(app)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                lifecycle.attach(_FakeApp())
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            with lifecycle:
                pass
        self.assertEqual(app.close_calls, 1)


class TrainEntrypointLifecycleContractTest(unittest.TestCase):
    def test_all_training_work_is_owned_by_lifecycle_context(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_run_training", functions)
        self.assertIn("main", functions)
        main_source = ast.get_source_segment(source, functions["main"])
        training_source = ast.get_source_segment(source, functions["_run_training"])
        self.assertIn("with SimulationAppLifecycle() as simulation_lifecycle", main_source)
        self.assertIn("return _run_training(config, simulation_lifecycle)", main_source)
        self.assertIn("simulation_lifecycle.attach(simulation_app)", training_source)
        self.assertIn("trainer.train()", training_source)
        self.assertNotIn("simulation_app.close()", source)


if __name__ == "__main__":
    unittest.main()
