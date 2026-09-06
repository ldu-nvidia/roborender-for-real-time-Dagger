from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "GR00T-VisualSim2Real/gr00t/rl/apps"
CAMERA_KITS = (
    APP_DIR / "phc.isaaclab.python.headless.rendering.kit",
    APP_DIR / "isaaclab.python.headless.rendering.kit",
    APP_DIR / "phc.isaaclab.python.rendering.kit",
)
CONTRACT_PATH = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/utils/camera_rendering_contract.py"
)


def _load_contract_module():
    spec = importlib.util.spec_from_file_location("camera_rendering_contract", CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


class IsaacCameraRenderingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract_module()

    def test_all_camera_kits_disable_nrd_dependent_rendering_paths(self):
        required = {
            "rtx.reflections.enabled": "false",
            "rtx.reflections.denoiser.enabled": "false",
            "rtx.indirectDiffuse.enabled": "false",
            "rtx.indirectDiffuse.denoiser.enabled": "false",
            "rtx.directLighting.sampledLighting.enabled": "false",
            "rtx.directLighting.sampledLighting.autoEnable": "false",
            "rtx.directLighting.sampledLighting.enforceDenoiser": "false",
            "rtx.directLighting.sampledLighting.irradiance.denoiser.enabled": "false",
            "rtx-defaults.directLighting.sampledLighting.enabled": "false",
            "rtx-defaults.directLighting.sampledLighting.autoEnable": "false",
            "rtx-defaults.directLighting.sampledLighting.enforceDenoiser": "false",
            "rtx-defaults.directLighting.sampledLighting.irradiance.denoiser.enabled": "false",
            "rtx-defaults.reflections.enabled": "false",
            "rtx-defaults.reflections.denoiser.enabled": "false",
            "rtx-defaults.indirectDiffuse.enabled": "false",
            "rtx-defaults.indirectDiffuse.denoiser.enabled": "false",
            "rtx-defaults.indirectDiffuse.nrd.enabled": "false",
            "rtx-defaults.shadows.denoiser.enable": "false",
            "rtx-defaults.ambientOcclusion.enabled": "false",
            "rtx-defaults.translucency.enabled": "false",
            "rtx-defaults.raytracing.cached.enabled": "false",
            "rtx-defaults-transient.dlssg.enabled": "false",
            "rtx-defaults-transient.dldenoiser.enabled": "false",
            "rtx.indirectDiffuse.nrd.enabled": "false",
            "rtx.shadows.denoiser.enable": "false",
            "rtx.ambientOcclusion.enabled": "false",
            "rtx.ambientOcclusion.denoiserMode": "1",
            "rtx.raytracing.subpixel.mode": "0",
            "rtx-transient.dldenoiser.enabled": "false",
        }

        for path in CAMERA_KITS:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing camera kit: {path}")
                settings = _settings(path)
                for key, expected in required.items():
                    self.assertEqual(settings.get(key), expected, key)

    def test_camera_kits_keep_synchronous_tiled_rendering_contract(self):
        required = {
            "rtx.viewTile.limit": "1000000",
            "renderer.multiGpu.maxGpuCount": "1",
            "omni.replicator.asyncRendering": "false",
            "app.renderer.waitIdle": "true",
            "app.hydraEngine.waitIdle": "true",
        }

        for path in CAMERA_KITS:
            with self.subTest(path=path.name):
                settings = _settings(path)
                for key, expected in required.items():
                    self.assertEqual(settings.get(key), expected, key)

    def test_headless_camera_launch_forces_performance_mode(self):
        args = SimpleNamespace(
            enable_cameras=True,
            headless=True,
            rendering_mode="balanced",
            kit_args=(
                "--/unrelated/setting=kept "
                "--/rtx/directLighting/sampledLighting/enabled=true"
            ),
        )
        self.assertTrue(self.contract.configure_headless_camera_rendering(args))
        self.assertEqual(args.rendering_mode, "performance")
        tokens = args.kit_args.split()
        self.assertIn("--/unrelated/setting=kept", tokens)
        self.assertNotIn(
            "--/rtx/directLighting/sampledLighting/enabled=true", tokens
        )
        for path in self.contract.EARLY_DISABLED_CARB_SETTINGS:
            self.assertEqual(tokens.count(f"--{path}=false"), 1, path)

    def test_headless_camera_launch_preserves_portable_runtime_flags(self):
        preserved = (
            "--portable",
            "--portable-root",
            "/sealed/cache",
            "--/log/file=/sealed/log/kit.log",
            "--/crashreporter/enabled=false",
        )
        args = SimpleNamespace(
            enable_cameras=True,
            headless=True,
            rendering_mode="balanced",
            kit_args=" ".join(preserved),
        )
        self.assertTrue(self.contract.configure_headless_camera_rendering(args))
        tokens = args.kit_args.split()
        for token in preserved:
            self.assertEqual(tokens.count(token), 1)

    def test_non_headless_camera_launch_preserves_requested_mode(self):
        args = SimpleNamespace(
            enable_cameras=True,
            headless=False,
            rendering_mode="quality",
            kit_args="--/rtx/reflections/enabled=true",
        )
        self.assertFalse(self.contract.configure_headless_camera_rendering(args))
        self.assertEqual(args.rendering_mode, "quality")
        self.assertEqual(args.kit_args, "--/rtx/reflections/enabled=true")

    def test_resolved_settings_are_validated_fail_closed(self):
        values = {
            path: False for path in self.contract.REQUIRED_DISABLED_CARB_SETTINGS
        }

        class Settings:
            def get(self, path):
                return values.get(path)

        self.assertEqual(
            self.contract.validate_headless_camera_rendering(Settings()), values
        )

        values["/rtx/directLighting/sampledLighting/enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "NRD-dependent"):
            self.contract.validate_headless_camera_rendering(Settings())

        values.pop("/rtx/directLighting/sampledLighting/enabled")
        with self.assertRaisesRegex(RuntimeError, "NRD-dependent"):
            self.contract.validate_headless_camera_rendering(Settings())

    def test_simulation_render_cfg_preserves_direct_lighting(self):
        kwargs = self.contract.headless_camera_render_cfg_kwargs()
        self.assertEqual(kwargs["rendering_mode"], "performance")
        self.assertIs(kwargs["enable_direct_lighting"], True)
        carb_settings = kwargs["carb_settings"]
        self.assertEqual(
            set(carb_settings), set(self.contract.REQUIRED_DISABLED_CARB_SETTINGS)
        )
        self.assertTrue(all(value is False for value in carb_settings.values()))
        self.assertNotIn("/rtx/directLighting/enabled", carb_settings)
        self.assertIn(
            "/rtx-defaults/directLighting/sampledLighting/enabled", carb_settings
        )
        self.assertIn(
            "/rtx-defaults-transient/dldenoiser/enabled", carb_settings
        )

    def test_simulation_render_cfg_is_applied_before_context_and_cameras(self):
        path = (
            ROOT
            / "GR00T-VisualSim2Real/gr00t/rl/simulator/isaacsim/isaacsim.py"
        )
        source = path.read_text(encoding="utf-8")
        render_cfg_marker = 'simulation_render_kwargs["render"] = RenderCfg('
        context_marker = "SimulationContext(sim_config)"
        camera_marker = "TiledCamera(ego_camera_config)"
        self.assertIn(
            "from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg, SimulationContext",
            source,
        )
        self.assertIn(render_cfg_marker, source)
        self.assertLess(source.index(render_cfg_marker), source.index(context_marker))
        self.assertLess(source.index(context_marker), source.index(camera_marker))

    def test_train_and_eval_enforce_post_launch_contract(self):
        environment_creation = {
            "train_agent_trl.py": "env = custom_instantiate(config.env",
            "eval_agent_trl.py": "env = instantiate(config=config.env",
        }
        for name, creation_marker in environment_creation.items():
            path = ROOT / "GR00T-VisualSim2Real/gr00t/rl" / name
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=name):
                self.assertIn("configure_headless_camera_rendering(args_cli)", source)
                validation_marker = (
                    "resolved_rendering_settings = validate_headless_camera_rendering("
                )
                self.assertIn(validation_marker, source)
                self.assertLess(
                    source.index(creation_marker),
                    source.index(validation_marker),
                    "resolved rendering must be checked after SimulationContext creation",
                )


if __name__ == "__main__":
    unittest.main()
