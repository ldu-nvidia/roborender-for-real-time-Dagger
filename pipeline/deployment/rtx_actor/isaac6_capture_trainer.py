"""Qualification-only capture evidence, using the existing native callbacks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dagger_rtx.contracts import StepKey, payload_digest
from dagger_rtx.durable import atomic_write_json
from dagger_rtx.native_service_trainer import NativeDaggerRTXServiceTrainer


class Isaac6CaptureTrainer(NativeDaggerRTXServiceTrainer):
    def _capture_only(self, batch, runtime, output):
        lifecycle = self._rollout_lifecycle()
        lifecycle.begin()
        try:
            keys = [StepKey("isaac6-native-capture", 1, f"capture-lane-{lane:03d}",
                            "window-000000", 0, f"episode-{lane:03d}-00000000", 0)
                    for lane in range(batch.num_envs)]
            assignments = batch.bind_episodes(dict(enumerate(keys)))
            try:
                captures = list(batch.capture_batch(keys))
            except Exception:
                simulator = self.env.simulator
                cameras = {}
                for view in ("head_color", "hand_left", "hand_right"):
                    cam = simulator.scene.sensors[f"cond_cam_{view}"]
                    segment = cam.data.output["semantic_segmentation"]
                    if hasattr(segment, "torch"):
                        segment = segment.torch
                    cameras[view] = {
                        "renderer_cfg_colorize": cam._renderer.cfg.colorize_semantic_segmentation,
                        "camera_cfg_colorize": cam.cfg.renderer_cfg.colorize_semantic_segmentation,
                        "dtype": str(segment.dtype), "shape": list(segment.shape),
                        "unique_values": torch.unique(segment).cpu().tolist()[:100],
                        "info": cam.data.info,
                    }
                atomic_write_json(output.parent / "capture-error-cameras.json", cameras)
                raise
            rows = []
            for lane, capture in enumerate(captures):
                arrays = capture.arrays
                if not np.array_equal(arrays["teacher_action"][11:], arrays["homie_action"]):
                    raise RuntimeError("teacher/Homie action tail mismatch")
                for view in ("head_color", "hand_left", "hand_right"):
                    depth = arrays[f"depth_metric__{view}"]
                    if not np.any(depth > 0):
                        raise RuntimeError(f"empty metric depth: {view}")
                # A small qualification sample, not an extra production archive.
                artifact = output.parent / f"capture-lane-{lane:03d}.npz"
                np.savez(artifact, **arrays)
                rows.append({
                    "lane_index": lane, "step_key": capture.key.as_dict(),
                    "metadata": capture.metadata, "assignment": assignments[lane],
                    "capture_digest": payload_digest(capture.metadata, arrays),
                    "arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.name,
                                      "min": float(value.min()), "max": float(value.max()),
                                      "finite": bool(np.isfinite(value).all())}
                               for name, value in sorted(arrays.items())},
                    "artifact": artifact.name, "artifact_bytes": artifact.stat().st_size,
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                })
            derived = json.loads(Path("/isaac6-port/port-manifest.json").read_text())
            atomic_write_json(output, {
                "schema": "roborender.isaac6.native-capture.v1", "result": "PASS",
                "num_envs": batch.num_envs, "environment_step_calls": 0,
                "teacher_inference_tested": True, "student_inference_tested": False,
                "H200_renderer_tested": False, "optimizer_tested": False,
                "production_qualified": False, "derived_runtime": derived, "rows": rows,
            })
        finally:
            lifecycle.finish()
