#!/usr/bin/env python3
"""Launch sealed Isaac Sim, step PhysX, and render RGB plus depth."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaacsim import SimulationApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases: list[str] = []

    def mark(phase: str) -> None:
        phases.append(phase)
        (args.output_dir / "isaac_camera_progress.json").write_text(
            json.dumps({"phases": phases}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"camera_smoke_phase={phase}", flush=True)

    # The full Isaac Sim application is intentional here.  Isaac Lab's 5.1
    # headless-camera experience enables NRD/DLSS paths that are not required
    # by RoboRender and do not compile on data-center GB200 GPUs.  The same
    # native RTX/Replicator camera path is exercised without that denoiser.
    app = SimulationApp(
        {
            "headless": True,
            "width": 128,
            "height": 96,
            "renderer": "RaytracedLighting",
        }
    )
    mark("simulation_app_started")
    try:
        import importlib
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

        kit_imports = {}
        for name in ("isaaclab_contrib", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"):
            mark(f"importing_{name}")
            module = importlib.import_module(name)
            kit_imports[name] = getattr(module, "__version__", "import-ok")
        mark("kit_ordered_imports_complete")

        stage = omni.usd.get_context().get_stage()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        cube = UsdGeom.Cube.Define(stage, "/World/Cube")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3d(8.0, 8.0, 0.1))
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(3000.0)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

        camera = rep.create.camera(position=(3.0, 3.0, 2.5), look_at=(0.0, 0.0, 0.4))
        product = rep.create.render_product(camera, (128, 96))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        rgb_annotator.attach(product)
        depth_annotator.attach(product)
        mark("scene_and_annotators_ready")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        initial_z = 2.0
        for _ in range(30):
            rep.orchestrator.step(rt_subframes=2)
        timeline.pause()
        mark("physics_and_render_steps_complete")
        final_transform = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        final_z = float(final_transform.ExtractTranslation()[2])
        rgb = np.asarray(rgb_annotator.get_data())
        depth = np.asarray(depth_annotator.get_data())
        if rgb.shape[:2] != (96, 128) or depth.shape[:2] != (96, 128):
            raise RuntimeError(f"unexpected camera tensors: rgb={rgb.shape}, depth={depth.shape}")
        finite_depth = np.isfinite(depth)
        if rgb.size == 0 or depth.size == 0 or not finite_depth.any():
            raise RuntimeError("camera annotators produced empty/non-finite output")
        if int(rgb[..., :3].max()) <= int(rgb[..., :3].min()):
            raise RuntimeError("camera produced a constant RGB image")
        if final_z >= initial_z - 0.02:
            raise RuntimeError(f"PhysX body did not fall: initial_z={initial_z}, final_z={final_z}")
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args.output_dir / "isaac_rgb.png")
        np.save(args.output_dir / "isaac_depth.npy", depth)
        report = {
            "status": "PASS",
            "rgb_shape": list(rgb.shape),
            "rgb_min": int(rgb[..., :3].min()),
            "rgb_max": int(rgb[..., :3].max()),
            "depth_shape": list(depth.shape),
            "depth_finite_fraction": float(finite_depth.mean()),
            "depth_finite_min": float(depth[finite_depth].min()),
            "depth_finite_max": float(depth[finite_depth].max()),
            "physics_initial_z": initial_z,
            "physics_final_z": final_z,
            "physics_fall_m": initial_z - final_z,
            "kit_ordered_imports": kit_imports,
        }
        (args.output_dir / "isaac_camera_smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print("HSG_GB200_ISAAC_CAMERA_PASS")
    except BaseException:
        (args.output_dir / "isaac_camera_failure.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
