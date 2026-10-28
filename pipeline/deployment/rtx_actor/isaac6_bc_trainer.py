"""Bounded BC qualification around Raven's unchanged collect-mode trainer.

Only instrumentation and stopping bounds live here. Teacher selection, camera
capture, per-episode 81-row buffering and reset-tail handling stay in Raven.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.rl.trl.trainer.distill_trainer_videogen import TRLDistillTrainerVideoGen


class Isaac6BCQualificationTrainer(TRLDistillTrainerVideoGen):
    def train(self):
        if self.vg_mode != "collect" or self.vg_rollout_action_source != "teacher":
            raise RuntimeError("BC qualification must use Raven teacher-only collect mode")
        if self.args.num_ppo_epochs != 0 or self.args.num_total_batches != 3 or self.num_steps_per_env != 81:
            raise RuntimeError("BC qualification is exactly three collection windows and zero updates")
        output = Path("/dagger-runtime")
        counts = {"vector_steps": 0, "optimizer_steps": 0, "teacher_action_equalities": 0}
        original_step = self.env.step
        original_optimizer_step = self.optimizer.step

        def checked_step(actor_state):
            if counts["vector_steps"] == 0:
                from initial_reset_contract import require_completed_initial_reset
                counts["initial_reset_completion"] = require_completed_initial_reset(self.env)
            actions, teacher = actor_state["actions"], actor_state["gt_actions"]
            if actions.shape != (self.env.num_envs, 26) or not torch.equal(actions, teacher):
                raise RuntimeError("BC behavior action differs from its privileged teacher label")
            if not torch.isfinite(actions).all():
                raise RuntimeError("non-finite BC teacher action")
            result = original_step(actor_state)
            counts["vector_steps"] += 1
            counts["teacher_action_equalities"] += self.env.num_envs
            if counts["vector_steps"] % 27 == 0:
                print(json.dumps({"phase": "bc_teacher_collection", **counts,
                                  "target_vector_steps": 243}), flush=True)
            return result

        def no_optimizer(*args, **kwargs):
            raise RuntimeError("optimizer execution is forbidden in BC collection qualification")

        self.env.step, self.optimizer.step = checked_step, no_optimizer
        try:
            super().train()
        finally:
            self.env.step, self.optimizer.step = original_step, original_optimizer_step
        if counts["vector_steps"] != 243:
            raise RuntimeError(f"incomplete native BC collection: {counts}")
        chunks = sorted(Path(self.vg_collect_dir).glob("chunk_*.npz"))
        if not chunks:
            raise RuntimeError("no complete Raven BC chunks; partial reset tails are not padded")
        # One deep sample is sufficient here; the native collector owns the
        # other chunks. Never rescan the full migrated dataset.
        sample = chunks[0]
        with np.load(sample, allow_pickle=False) as z:
            shapes = {key: {"shape": list(z[key].shape), "dtype": z[key].dtype.name}
                      for key in z.files}
            if z["gt_actions"].shape != (81, 26) or z["obs_actor_obs"].shape != (81, 113):
                raise RuntimeError("wrong BC teacher/state shape")
            for view in ("head_color", "hand_left", "hand_right"):
                for key, shape, dtype in (
                    (f"depth_ctrl__{view}", (81, 240, 416), np.uint8),
                    (f"rgb_robot__{view}", (81, 240, 416, 3), np.uint8),
                ):
                    if z[key].shape != shape or z[key].dtype != dtype or not np.any(z[key]):
                        raise RuntimeError(f"invalid BC conditioning: {key}")
            if not np.isfinite(z["gt_actions"]).all() or not np.isfinite(z["obs_actor_obs"]).all():
                raise RuntimeError("non-finite BC labels/state")
        report = {"schema": "roborender.isaac6.bc-collection.v1", "result": "PASS",
                  "num_envs": self.env.num_envs, **counts, "native_teacher_only": True,
                  "chunks": [{"path": str(p), "bytes": p.stat().st_size} for p in chunks],
                  "sample": {"path": str(sample), "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                             "arrays": shapes}, "production_qualified": False}
        (output / "bc-collection-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"phase": "bc_collection_complete", "chunks": len(chunks), **counts}), flush=True)
