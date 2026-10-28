"""Inside-container RGB/depth smoke test; no Raven environment or policy update."""
from __future__ import annotations

import hashlib
import faulthandler
import json
from pathlib import Path
import time


def main():
    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)

    def phase(name):
        print(f"QUALIFICATION_PHASE {time.monotonic() - started:.3f}s {name}", flush=True)

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": True, "width": 192, "height": 108,
        "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False,
        "limit_cpu_threads": 8,
    })
    try:
        # Import extensions only after SimulationApp has initialized them.
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        print("QUALIFICATION_APP_STARTED", flush=True)
        phase("configure_scene")
        stage = omni.usd.get_context().get_stage()
        cube = UsdGeom.Cube.Define(stage, "/World/ProbeCube")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.65, 0.3)])
        position = cube.AddTranslateOp()
        position.Set(Gf.Vec3d(0, 0, 0))
        light = UsdLux.DomeLight.Define(stage, "/World/ProbeLight")
        light.CreateIntensityAttr(1000.0)
        camera = UsdGeom.Camera.Define(stage, "/World/ProbeCamera")
        camera.AddTranslateOp().Set(Gf.Vec3d(0, 0, 4))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
        phase("create_render_product")
        product = rep.create.render_product(str(camera.GetPath()), (192, 108))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        phase("attach_annotators")
        rgb.attach([product])
        depth.attach([product])

        def capture(label):
            for index in range(4):
                phase(f"{label}:render_step_{index}:start")
                rep.orchestrator.step(rt_subframes=4)
                phase(f"{label}:render_step_{index}:end")
            color = np.asarray(rgb.get_data()).copy()
            distance = np.asarray(depth.get_data()).copy()
            if color.ndim != 3 or color.shape[:2] != (108, 192) or color.shape[2] not in (3, 4):
                raise RuntimeError(f"invalid RGB shape {color.shape}")
            if distance.shape not in ((108, 192), (108, 192, 1)):
                raise RuntimeError(f"invalid depth shape {distance.shape}")
            distance = distance.reshape(108, 192)
            valid = np.isfinite(distance) & (distance > 0) & (distance < 100)
            if not valid.any() or np.ptp(color[..., :3]) == 0:
                raise RuntimeError("empty/constant RGB or no valid metric depth")
            return color, distance

        rgb_before, depth_before = capture("before")
        position.Set(Gf.Vec3d(0.4, 0, 0))
        rgb_after, depth_after = capture("after")
        changed = int(np.count_nonzero(np.any(rgb_before[..., :3] != rgb_after[..., :3], axis=2)))
        depth_changed = int(np.count_nonzero(~np.isclose(depth_before, depth_after, equal_nan=True)))
        if changed == 0 or depth_changed == 0:
            raise RuntimeError("camera output did not respond to the cube transform")
        output = Path("/qualification")
        np.savez(output / "camera_probe.npz", rgb_before=rgb_before,
                 depth_before=depth_before, rgb_after=rgb_after, depth_after=depth_after)
        report = {
            "schema": "roborender.rtx.graphics-probe.v1", "result": "PASS",
            "scope": "synthetic_cube_rgb_depth_only", "raven_env_steps": 0,
            "teacher_inference_tested": False, "production_qualified": False,
            "elapsed_seconds": time.monotonic() - started,
            "rgb_changed_pixels": changed, "depth_changed_pixels": depth_changed,
            "rgb_shape": list(rgb_before.shape), "depth_shape": list(depth_before.shape),
            "rgb_sha256": hashlib.sha256(rgb_after.tobytes()).hexdigest(),
        }
        (output / "probe-result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("QUALIFICATION_CAMERA_PASS " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        app.close()


if __name__ == "__main__":
    main()
