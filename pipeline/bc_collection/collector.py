"""Standalone Raven-style BC collection, without the PPO training loop.

The Isaac bridge supplies the pinned policy/camera hooks, native environment,
storage, provenance and resource guards. This module owns collection ordering.
One scene lives for the entire explicit budget; 81 rows is NOT a reset budget.
Reference: Raven 94d7985 PPO/distill/videogen rollout methods. No learner,
validation, scheduler, checkpoint callback, data-loader or per-lane quota runs.
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

PROTOCOL = "roborender.bc-standalone-collector.v1"
FRAMES_PER_WINDOW = 81
DEFAULT_NUM_ENVS = 32


def implementation_receipt():
    return dict(protocol=PROTOCOL, **{
        name + "_sha256": hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
        for name in ("collector", "chunk_writer")})


def optional_teacher_hook(owner, name):
    # Match Raven's distillation wrapper. Do not swallow student/Isaac failures.
    teacher = getattr(owner, "ref_model", None)
    if teacher is not None and hasattr(teacher, name):
        try:
            getattr(teacher, name)()
        except Exception:
            pass


def collect_window(owner, model, observations):
    """Exactly81 ordered vector transitions; native done handles lane resets."""
    import torch

    owner._depth_chunk, owner._robot_rgb_chunk, owner._views_chunk = [], [], []
    owner._in_train_rollout = True
    try:
        optional_teacher_hook(owner, "init_rollout")
        owner._train_rollout_mode()
        model.policy.init_rollout()
        owner.storage.clear()
        device = owner.accelerator.device
        dones = torch.zeros(owner.env.num_envs, device=device)
        for _ in range(FRAMES_PER_WINDOW):
            # Existing hook captures pre-action depth/robot RGB and computes
            # teacher+Homie labels. The collection wrapper forces those actions.
            policy = owner.policy_step(model.policy, observations, cur_dones=dones)
            actions, labels = policy["actions"], policy["gt_actions"]
            if (actions.shape != (owner.env.num_envs, 26)
                    or not torch.equal(actions, labels) or not torch.isfinite(actions).all()):
                raise RuntimeError("collection requires finite teacher-only26-D behavior")
            for key, value in observations.items():
                owner.storage.update_key(key, value)
            for key, value in policy.items():
                if key != "hidden_states":
                    owner.storage.update_key(key, value)
            if "hidden_states" in policy:
                owner.storage._save_hidden_states(policy["hidden_states"])
            observations, rewards, dones, infos = owner.env.step(
                {"actions": actions, "gt_actions": labels})
            observations = {key: value.to(device) for key, value in observations.items()}
            rewards, dones = rewards.to(device), dones.to(device)
            owner.ep_infos.append(infos["episode"])
            owner.storage.update_key("rewards", rewards.clone().unsqueeze(1))
            owner.storage.update_key("dones", dones.unsqueeze(1))
            owner.storage.update_key("time_outs", infos["time_outs"].unsqueeze(1))
            owner.storage.increment_step()
            owner._process_env_step(rewards, dones, infos)
            owner.cur_reward_sum += rewards
            owner.cur_episode_length += 1
            ids = (dones > 0).nonzero(as_tuple=False)
            owner.state.rewbuffer.extend(owner.cur_reward_sum[ids][:, 0].cpu().numpy().tolist())
            owner.state.lenbuffer.extend(owner.cur_episode_length[ids][:, 0].cpu().numpy().tolist())
            owner.cur_reward_sum[ids] = 0
            owner.cur_episode_length[ids] = 0
        model.policy.clear_rollout()
        optional_teacher_hook(owner, "clear_rollout")
    finally:
        owner._in_train_rollout = False
    # Dynamic dispatch retains the native memory guards and provenance journal.
    owner._save_collected_chunks()
    return observations


def collect(owner, max_windows, *, target_chunks_per_lane=None):
    """Repeated episodes in one vector scene, not repeated short-lived cohorts.

Stop only at the declared overall vector-window budget (or fail closed).
Natural episode counts differ across lanes. Never force terminal completion or
flush/pad unfinished tails at shutdown. A lost Isaac lifetime cannot be resumed.
"""
    import torch
    from gr00t.rl.envs.base_task.staged_task_base import StagedTaskBase

    if target_chunks_per_lane is not None and (
            type(target_chunks_per_lane) is not int or target_chunks_per_lane != 20
            or owner.env.num_envs not in (40, 50) or max_windows != 40):
        raise ValueError("only the explicit B40/B50 20-chunk diagnostic stop is supported")

    if (type(max_windows) is not int or not 1 <= max_windows <= 50000
            or owner.num_steps_per_env != FRAMES_PER_WINDOW
            or owner.args.num_total_batches != max_windows
            or owner.args.num_ppo_epochs != 0 or owner.vg_mode != "collect"
            or owner.vg_rollout_action_source != "teacher" or not owner.vg_enabled
            or owner.value_model is not None or owner.policy_model.is_recurrent
            or owner.accelerator.num_processes != 1
            or owner.is_deepspeed_enabled
            or owner.dataloader is not None or owner.schedule_dict is not None
            or owner.train_with_evaluating_env
            or owner.env.config.get("reinit_sim_freq", 0)
            or not owner.config.get("init_at_random_ep_len", False)
            or owner.state.global_step != 0):
        raise ValueError("standalone collection needs fresh teacher-only nonrecurrent81-step scene")
    owner._register_stats_buffer()
    owner.model.train()
    owner.state.max_steps = max_windows
    device = owner.accelerator.device
    observations = owner.env.reset_all()  # The only whole-scene reset.
    owner.env.episode_length_buf = torch.randint_like(
        owner.env.episode_length_buf, high=int(owner.env.max_episode_length))
    if isinstance(owner.env, StagedTaskBase):
        owner.env.time_in_stage_buf[:] = torch.randint_like(
            owner.env.time_in_stage_buf, high=int(owner.env.max_stage_time[0]))
        owner.env.actual_time_in_stage_buf[:] = owner.env.time_in_stage_buf
    observations = {key: value.to(device) for key, value in observations.items()}
    for _ in range(max_windows):
        disk = getattr(owner, "_bc_disk_guard", None)
        if disk is not None:
            disk.admit_window()
        owner.state.episode += owner.args.batch_size
        with torch.no_grad():
            observations = collect_window(owner, owner.accelerator.unwrap_model(owner.model), observations)
        # Restore the original mode boundary and discarded RNG draws without
        # doing an optimizer step. The bridge validates the pinned architecture.
        owner._train_mode()
        owner.state.global_step += 1
        owner.state.tot_timesteps += FRAMES_PER_WINDOW * owner.env.num_envs
        owner.episode_env_tensors.mean_and_clear()
        owner.ep_infos.clear()
        torch.cuda.empty_cache()
        gc.collect()
        drain = getattr(owner._bc_ledger, "drain", None) if hasattr(owner, "_bc_ledger") else None
        if drain is not None and drain.complete:
            return
        if target_chunks_per_lane is not None and min(owner._bc_ledger.emitted_counts) >= target_chunks_per_lane:
            # Stop the whole scene at an accepted save barrier. Faster lanes
            # keep collecting; no per-lane pause/reset or output truncation.
            return
    if target_chunks_per_lane is not None:
        raise RuntimeError("smoke safety window ceiling reached before all lanes collected20 chunks")
    if getattr(getattr(owner, "_bc_ledger", None), "drain", None) is not None:
        raise RuntimeError("safety ceiling reached before total target and natural episode drain; incomplete")
