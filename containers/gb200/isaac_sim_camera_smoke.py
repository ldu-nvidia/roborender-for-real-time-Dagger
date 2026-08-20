#!/usr/bin/env python3
"""Render RGB/depth using only the Isaac Sim 5.1 base image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    app = SimulationApp(
        {
            "headless": True,
            "width": 128,
            "height": 96,
            "renderer": "RaytracedLighting",
        }
    )
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/World/Cube")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3d(8.0, 8.0, 0.1))
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(3000.0)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

        camera = rep.create.camera(position=(3.0, 3.0, 2.5), look_at=(0.0, 0.0, 0.4))
        product = rep.create.render_product(camera, (128, 96))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        rgb_annotator.attach(product)
        depth_annotator.attach(product)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=2)
        rgb = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data())
        if rgb.shape[:2] != (96, 128) or depth.shape[:2] != (96, 128):
            raise RuntimeError(f"unexpected camera tensors: rgb={rgb.shape}, depth={depth.shape}")
        finite_depth = np.isfinite(depth)
        if rgb.size == 0 or depth.size == 0 or not finite_depth.any():
            raise RuntimeError("camera annotators produced empty/non-finite output")
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args.output_dir / "base_rgb.png")
        np.save(args.output_dir / "base_depth.npy", depth)
        report = {
            "status": "PASS",
            "rgb_shape": list(rgb.shape),
            "rgb_min": int(rgb[..., :3].min()),
            "rgb_max": int(rgb[..., :3].max()),
            "depth_shape": list(depth.shape),
            "depth_finite_fraction": float(finite_depth.mean()),
            "depth_finite_min": float(depth[finite_depth].min()),
            "depth_finite_max": float(depth[finite_depth].max()),
        }
        (args.output_dir / "isaac_base_camera_smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print("HSG_GB200_ISAAC_BASE_CAMERA_PASS")
    finally:
        app.close()


if __name__ == "__main__":
    main()
