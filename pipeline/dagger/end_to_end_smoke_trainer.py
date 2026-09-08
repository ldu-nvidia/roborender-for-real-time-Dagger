"""Diagnostic-only trainer for proving the live DAgger data path.

The production trainer remains Raven's ``TRLDistillTrainerVideoGen``.  This
subclass delegates every scientific operation to that trainer and only writes
compact JSONL receipts when ``DAGGER_SMOKE_TRACE`` is set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from gr00t.rl.trl.trainer.distill_trainer_videogen import (
    TRLDistillTrainerVideoGen,
)


class EndToEndSmokeTrainer(TRLDistillTrainerVideoGen):
    """Raven trainer plus non-invasive runtime receipts for a bounded smoke."""

    def _init_trl(self, *args, **kwargs):
        super()._init_trl(*args, **kwargs)
        trace = os.environ.get("DAGGER_SMOKE_TRACE")
        if not trace:
            raise RuntimeError("DAGGER_SMOKE_TRACE must name the smoke JSONL receipt")
        self._smoke_trace = Path(trace)
        self._smoke_trace.parent.mkdir(parents=True, exist_ok=True)
        self._smoke_trace.unlink(missing_ok=True)
        self._smoke_policy_event = None

        original_env_step = self.env.step

        def traced_env_step(actor_state, *step_args, **step_kwargs):
            behavior = actor_state["actions"].detach()
            teacher = actor_state.get("gt_actions")
            event = {
                "event": "env_step_input",
                "behavior_finite": bool(torch.isfinite(behavior).all()),
                "behavior_shape": list(behavior.shape),
            }
            if teacher is not None:
                teacher = teacher.detach()
                event.update(
                    teacher_finite=bool(torch.isfinite(teacher).all()),
                    teacher_shape=list(teacher.shape),
                    behavior_teacher_mse=float(torch.mean((behavior - teacher) ** 2).cpu()),
                )
            if self._smoke_policy_event is not None:
                prior = self._smoke_policy_event
                event["equals_policy_behavior"] = bool(
                    torch.equal(behavior.cpu(), prior["behavior"])
                )
            self._trace(event)
            result = original_env_step(actor_state, *step_args, **step_kwargs)
            obs, rewards, dones, _infos = result
            self._trace(
                {
                    "event": "env_step_output",
                    "reward_finite": bool(torch.isfinite(rewards).all()),
                    "reward_shape": list(rewards.shape),
                    "done_shape": list(dones.shape),
                    "observation_keys": sorted(obs),
                }
            )
            return result

        self.env.step = traced_env_step

    def _trace(self, event):
        event = {"utc_unix": time.time(), **event}
        with self._smoke_trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        value = value.detach().contiguous().cpu()
        return hashlib.sha256(value.numpy().tobytes()).hexdigest()

    def _policy_sha256(self) -> str:
        digest = hashlib.sha256()
        policy = self.accelerator.unwrap_model(self.model).policy
        for name, value in sorted(policy.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    def train(self):
        before = self._policy_sha256()
        self._trace({"event": "training_begin", "policy_sha256": before})
        result = super().train()
        after = self._policy_sha256()
        self._trace(
            {
                "event": "training_end",
                "policy_sha256": after,
                "parameters_changed": before != after,
                "global_step": int(self.state.global_step),
            }
        )
        return result

    def _image_dagger_relabel(self, obs_dict, cur_dones):
        source = obs_dict["vision_obs"].detach()
        result = super()._image_dagger_relabel(obs_dict, cur_dones)
        rendered = result["vision_obs"].detach()
        self._trace(
            {
                "event": "image3f_relabel",
                "input_shape": list(source.shape),
                "output_shape": list(rendered.shape),
                "input_finite": bool(torch.isfinite(source).all()),
                "output_finite": bool(torch.isfinite(rendered).all()),
                "input_sha256": self._tensor_sha256(source),
                "output_sha256": self._tensor_sha256(rendered),
                "pixels_changed": not torch.equal(source, rendered),
                "latest_render_ms": float(self._id_gen_ms[-1]),
            }
        )
        return result

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        result = super().policy_step(policy_model, obs_dict, cur_dones=cur_dones)
        if self._in_train_rollout and "gt_actions" in result:
            behavior = result["actions"].detach()
            student = result["action_mean"].detach()
            teacher = result["gt_actions"].detach()
            teacher_error = torch.mean((behavior - teacher) ** 2, dim=-1)
            student_error = torch.mean((behavior - student) ** 2, dim=-1)
            teacher_selected = teacher_error <= 1e-12
            student_selected = student_error <= 1e-12
            self._smoke_policy_event = {"behavior": behavior.cpu()}
            self._trace(
                {
                    "event": "policy_step",
                    "beta": float(self._beta_value),
                    "action_shape": list(behavior.shape),
                    "teacher_finite": bool(torch.isfinite(teacher).all()),
                    "student_finite": bool(torch.isfinite(student).all()),
                    "behavior_finite": bool(torch.isfinite(behavior).all()),
                    "teacher_selected": teacher_selected.cpu().tolist(),
                    "student_selected": student_selected.cpu().tolist(),
                    "student_teacher_mse": float(torch.mean((student - teacher) ** 2).cpu()),
                    "teacher_action_head": teacher[0, :6].cpu().tolist(),
                    "student_action_head": student[0, :6].cpu().tolist(),
                    "behavior_action_head": behavior[0, :6].cpu().tolist(),
                }
            )
        return result

    def _compute_loss(self, forward_results, mb_rollout_data):
        result = super()._compute_loss(forward_results, mb_rollout_data)
        self._trace(
            {
                "event": "loss_forward",
                "loss": float(result["loss"].detach().cpu()),
                "dagger_bc_loss": float(result["dagger_bc_loss"].detach().cpu()),
                "loss_finite": bool(torch.isfinite(result["loss"]).all()),
            }
        )
        return result

    def _gradient_clipping(self):
        norm = super()._gradient_clipping()
        value = float(norm.detach().cpu()) if isinstance(norm, torch.Tensor) else float(norm)
        self._trace(
            {
                "event": "backward_gradient",
                "gradient_norm": value,
                "gradient_finite": math.isfinite(value),
                "gradient_nonzero": value > 0.0,
            }
        )
        return norm
