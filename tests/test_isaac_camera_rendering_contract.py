from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "GR00T-VisualSim2Real/gr00t/rl/apps"
CAMERA_KITS = (
    APP_DIR / "phc.isaaclab.python.headless.rendering.kit",
    APP_DIR / "isaaclab.python.headless.rendering.kit",
    APP_DIR / "phc.isaaclab.python.rendering.kit",
)


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


if __name__ == "__main__":
    unittest.main()
