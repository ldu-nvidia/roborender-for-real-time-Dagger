"""Frozen teacher/student closed-loop evaluation on a vectorized Isaac runtime.

Original termination/reset functions remain unchanged. The shared production
accounting mixin preserves their pre-reset goal in the outgoing transition.
The admitted caller supplies population size and benchmark configuration.
The legacy five-case default is not a production capacity certification.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.success_rate.episode_metrics import (
    FirstEpisodeAccounting, require_paired_states, require_student_only_evaluation,
)
from pipeline.success_rate.policy_actions import (
    DISTILL, RAVEN_PPO, control_contract, select_policy_actions, numeric_state,
    prepare_ppo_teacher, validate_mode,
)
from initial_robot_refresh_diagnostic import (
    ORIGINAL as ORIGINAL_INITIAL_REFRESH, CLEAR_ONCE, FIELD as INITIAL_REFRESH_FIELD,
    control_contract as initial_refresh_contract, validate_mode as validate_initial_refresh,
    validate_attestation as validate_initial_refresh_attestation,
)
from dagger_rtx.terminal_goal_accounting import GOAL_ACCOUNTING_VERSION, TerminalGoalAccountingMixin
from gr00t.rl.envs.loco_manip.walk_stand_place_grasp_turn_homie import (
    WalkStandPlaceGraspTurnHomie as NativeEnvironment,
)
from gr00t.rl.simulator.isaacsim.isaacsim import IsaacSim as NativeIsaacSim
from gr00t.rl.trl.trainer.distill_trainer_obj_pred_homie_api import TRLDistillTrainerObjPredHomieAPI
from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
from gr00t.rl.utils.torch_utils import quat_rotate
from gr00t.rl.data.tasks.walk_stand_place_grasp_turn_homie.scenario_cfg import isaacsim as scenario

BC_SHA = "053b5386172373c8bbc1518bfc360b2371221979b54eb53e76ad15b55eba8718"
VIEWS = ("head_color", "hand_left", "hand_right")


class SuccessAccountingEnvironment(TerminalGoalAccountingMixin, NativeEnvironment):
    """Use the same corrected accounting as the production fixed-panel path."""


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def digest(array):
    value = np.ascontiguousarray(array)
    return {"dtype": value.dtype.name, "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def cpu(tensor):
    # Raven exposes ground_height as a Python float, other fields as tensors.
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().copy()
    return np.asarray(tensor).copy()


def table_metadata(config, env_idx, device):
    if config.reset_from_dataset.get("use_motion_file_dir", True):
        raise RuntimeError("diagnostic requires explicit held-out motions")
    paths = config.reset_from_dataset.motion_files
    return MotionLoaderWSPPT.from_file(str(paths[int(env_idx) % len(paths)]), device).table_custom_data


scenario.get_table_grid_meta_data = table_metadata


class IsaacSim(NativeIsaacSim):
    """Bind held-out table metadata and normalize native RGB exactly once."""
    def __init__(self, *args, **kwargs):
        scenario_path = Path(inspect.getsourcefile(scenario)).resolve()
        original = importlib.machinery.SourceFileLoader.exec_module
        bound = []

        def patch(loader, module):
            original(loader, module)
            if Path(str(getattr(loader, "path", ""))).resolve() == scenario_path:
                module.get_table_grid_meta_data = table_metadata
                for name in ("table_top_material_prim_paths", "leg_material_prim_paths"):
                    setattr(module, name, [str(item) for item in getattr(module, name)])
                bound.append(True)

        importlib.machinery.SourceFileLoader.exec_module = patch
        try:
            super().__init__(*args, **kwargs)
        finally:
            importlib.machinery.SourceFileLoader.exec_module = original
        if not bound:
            raise RuntimeError("held-out table binding did not execute")

    def get_policy_view_stack(self):
        raw = super().get_policy_view_stack()
        if raw is None:
            return None
        cameras = self.simulator_config.cameras
        if tuple(cameras.policy_views) != VIEWS or tuple(raw.shape[1:]) != (324, 192, 3):
            raise RuntimeError("three-view policy shape/order mismatch")
        if not getattr(self, "_diagnostic_rgb_checked", False):
            if not torch.isfinite(raw).all() or raw.min() < -1e-6 or raw.max() > 1.000001:
                raise RuntimeError("native RGB is not finite RGB01")
            self._diagnostic_rgb_checked = True
        mean = torch.as_tensor(cameras.image_mean, device=raw.device, dtype=raw.dtype)
        std = torch.as_tensor(cameras.image_std, device=raw.device, dtype=raw.dtype)
        return (raw - mean) / std


def initial_state(env, obs):
    state = {"robot_root": cpu(env.simulator.robot_root_states),
             "dof_pos": cpu(env.simulator.dof_pos), "dof_vel": cpu(env.simulator.dof_vel)}
    for name in ("stage_buf", "episode_length_buf", "time_in_stage_buf", "sample_idx", "sample_idx_in_seq"):
        state[name] = cpu(getattr(env, name))
    for name in ("front_object", "hold_object", "tray_object"):
        state[name] = cpu(env.simulator.get_task_root_state(getattr(env.config.task, name)))
    for name in ("actor_obs", "teacher_obs", "homie_obs", "gt_obj_pos"):
        state[name] = cpu(obs[name])
    return state


def terminal_snapshot(env):
    """Called after native _check_termination, before automatic reset."""
    values = {name: cpu(getattr(env, name)) for name in (
        "current_last_stage_completed_task_buf", "reset_buf", "time_out_buf",
        "episode_length_buf", "stage_buf", "time_in_stage_buf")}
    face = quat_rotate(env.base_quat, env.forward_vec) @ torch.tensor([-1., 0., 0.], device=env.device)
    values["face_dot"] = cpu(face)
    values["arm_qvel_max"] = cpu(env.simulator.dof_vel[:, env.right_arm_dof_indices].abs().amax(-1))
    values["arm_qvel_threshold"] = cpu(env.config.termination_scales.termination_fast_right_arm_qvel_threshold * env.termination_level)
    values["hold_tray_distance"] = cpu(env.get_hold_object_tray_distance())
    values["hold_hand_distance"] = cpu(env.get_hold_object_hand_distance())
    for name in ("front_object", "hold_object", "tray_object"):
        values[name] = cpu(env.simulator.get_task_root_state(getattr(env.config.task, name)))
    values["ground_height"] = cpu(env.ground_height)
    values["stage_time_limit"] = cpu(env.max_stage_time[env.stage_buf])
    return values


def runtime_physics_evidence(env):
    """Read loaded scene/collider/material properties; never edit physics.

    This records applied scene configuration, not a contact impulse/bounce
    experiment. Background walls must remain non-collidable per Raven.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    sim = env.simulator.sim
    stage = sim.stage
    physics = getattr(sim.cfg, "physics", getattr(sim.cfg, "physx", None))
    keys = ("solver_type", "bounce_threshold_velocity", "enable_ccd", "enable_stabilization",
            "min_position_iteration_count", "max_position_iteration_count",
            "min_velocity_iteration_count", "max_velocity_iteration_count")
    values = {key: getattr(physics, key, None) for key in keys}
    colliders = []
    for name in ("wall_back", "wall_left", "wall_right", "front_object", "hold_object", "tray_object"):
        path = "/World/envs/env_0/" + name
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            colliders.append({"object": name, "path": path, "present": False})
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            attributes = {}
            if material:
                mp = material.GetPrim()
                for key in ("physics:restitution", "physics:staticFriction", "physics:dynamicFriction",
                            "physxMaterial:restitutionCombineMode", "physxMaterial:frictionCombineMode"):
                    attr = mp.GetAttribute(key)
                    attributes[key] = attr.Get() if attr else None
            colliders.append({"object": name, "path": str(prim.GetPath()), "present": True,
                              "collision_enabled": enabled,
                              "physics_material_path": str(material.GetPath()) if material else None,
                              "material_attributes": attributes})
            if name.startswith("wall_") and enabled:
                raise RuntimeError("Raven camera-backdrop wall unexpectedly collidable")
    for name in ("front_object", "hold_object", "tray_object"):
        if not any(row["object"] == name and row.get("collision_enabled") for row in colliders):
            raise RuntimeError("missing enabled runtime task collider: " + name)
    return {"schema": "roborender.runtime-physics-observation.v1", "physics_dt": sim.cfg.dt,
            "applied_physics_config": values,
            "yaml_bounce_threshold_velocity": env.config.simulator.config.sim.physx.bounce_threshold_velocity,
            "colliders": colliders, "wall_bounce_experiment_performed": False,
            "physics_unchanged": True}


def lane_details(values, lane, config):
    row = {name: (value[lane].tolist() if value.ndim else value.item()) for name, value in values.items()}
    term, scales = config.termination, config.termination_scales
    reasons = []
    if row["time_out_buf"]:
        reasons.append("native_timeout")
    if config.get("reset_on_overtime", True) and row["time_in_stage_buf"] >= row["stage_time_limit"]:
        reasons.append("stage_overtime_predicate")
    if term.get("terminate_by_fast_right_arm_qvel", False) and row["arm_qvel_max"] > row["arm_qvel_threshold"]:
        reasons.append("fast_right_arm_qvel")
    ground = float(np.asarray(row["ground_height"]).reshape(-1)[0])
    if term.get("terminate_object_by_low_height", False):
        for name in ("front_object", "hold_object"):
            if row[name][2] - ground < scales.termination_object_by_low_height_threshold:
                reasons.append(name + "_low_height")
    if term.get("terminate_after_finish_turning_for_student", False) and row["stage_buf"] == 4 and row["face_dot"] > .9:
        reasons.append("finished_turning")
    if term.get("terminate_by_face_mul", False) and row["current_last_stage_completed_task_buf"] and row["face_dot"] < .7:
        reasons.append("overturn_after_goal")
    if term.get("terminate_by_throw_object", False) and row["hold_tray_distance"] > config.hold_object_tray_distance_threshold and row["hold_hand_distance"] > config.termin_hold_object_hand_distance_threshold:
        reasons.append("throw_object")
    if term.get("terminate_by_hold_object_tray_distance", False) and row["stage_buf"] >= 2 and max(abs(row["hold_object"][i] - row["tray_object"][i]) for i in (0, 1)) > scales.termination_hold_object_tray_distance_threshold:
        reasons.append("hold_object_outside_tray_xy")
    row["diagnostic_reason_predicates"] = reasons
    row["reason_list_exhaustive"] = False
    return row


class SuccessDiagnosticTrainer(TRLDistillTrainerObjPredHomieAPI):
    def _diagnostic_expected_envs(self):
        # The existing five-motion benchmark is unchanged. New, separately
        # versioned explicit-motion evaluators override this bounded count.
        return 5

    def _diagnostic_after_reset(self, obs, output):
        """Optional read-only native assignment attestation for a new panel."""

    def train(self):
        """Compatibility hook required by the existing trainer launcher."""
        return self.evaluate()

    def evaluate(self):
        """Run frozen-policy, vectorized closed-loop evaluation; never optimize."""
        output = Path(os.environ.get("DIAGNOSTIC_OUTPUT_DIR", "/dagger-runtime")).resolve()
        if not output.is_relative_to(Path("/dagger-runtime")):
            raise RuntimeError("evaluation output must remain in its owned runtime")
        output.mkdir(parents=True, exist_ok=True)
        arm = os.environ["DIAGNOSTIC_ARM"]
        require_student_only_evaluation(self.config, arm=arm)
        teacher_mode = validate_mode(os.environ.get("DIAGNOSTIC_TEACHER_ACTION_MODE", DISTILL), arm)
        numeric_steps = int(os.environ.get("DIAGNOSTIC_NUMERIC_TRACE_STEPS",
                                          "24" if teacher_mode == RAVEN_PPO else "0"))
        if not 0 <= numeric_steps <= 24:
            raise ValueError("numeric diagnostic trace must be bounded to 0..24 vector steps")
        count = self._diagnostic_expected_envs()
        scaled_teacher = getattr(self, '_scaled_teacher_benchmark', None)
        teacher_guard_checks = 0
        from initial_robot_refresh_diagnostic import require_corrected_execution
        initial_refresh = require_corrected_execution(
            os.environ.get("DIAGNOSTIC_INITIAL_ROBOT_REFRESH"),
            arm=arm, teacher_action_mode=teacher_mode, num_envs=count)
        if initial_refresh != ORIGINAL_INITIAL_REFRESH and os.environ.get("DIAGNOSTIC_RESEED_DATASET") != "1":
            raise ValueError("clear-once diagnostic requires an independent explicit panel reset")
        if (arm not in {"teacher", "bc", "dagger"} or type(count) is not int
                or not 1 <= count <= 40 or self.env.num_envs != count):
            raise RuntimeError("bounded diagnostic scope mismatch")
        if not isinstance(self.env, SuccessAccountingEnvironment):
            raise RuntimeError("corrected production goal accounting is required")
        cp = Path(os.environ.get("DIAGNOSTIC_CHECKPOINT_PATH",
                                 "/diagnostic-assets/best_validation.pt")).resolve(strict=True)
        if not cp.is_relative_to(Path("/diagnostic-assets")):
            raise RuntimeError("evaluation checkpoint must remain in its read-only asset mount")
        checkpoint_sha = os.environ.get("DIAGNOSTIC_CHECKPOINT_SHA256", BC_SHA)
        if len(checkpoint_sha) != 64 or hashlib.sha256(cp.read_bytes()).hexdigest() != checkpoint_sha:
            raise RuntimeError("evaluation checkpoint identity mismatch")
        checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
        if "env_state_dict" in checkpoint:
            raise RuntimeError("checkpoint environment state is forbidden")
        expected = checkpoint["policy_state_dict"]
        self.policy_model.load_state_dict(expected, strict=True)
        for key, value in self.policy_model.state_dict().items():
            if not torch.equal(value.cpu(), expected[key]):
                raise RuntimeError("student weight equality failed: " + key)
        self._eval_mode()
        self.policy_model.eval_mode()
        self.policy_model.init_rollout()
        self.ref_model.eval()
        if hasattr(self.ref_model, "init_rollout"):
            self.ref_model.init_rollout()
        teacher_attestation = (prepare_ppo_teacher(self, original_ppo=teacher_mode == RAVEN_PPO)
            if teacher_mode == RAVEN_PPO or (arm == "teacher" and numeric_steps) else None)
        mode_config = {"teacher_action_mode": teacher_mode,
                       "teacher_control": control_contract(teacher_mode),
                       "numeric_trace_steps": numeric_steps,
                       "teacher_parameter_attestation": teacher_attestation}
        if initial_refresh != ORIGINAL_INITIAL_REFRESH:
            mode_config.update({INITIAL_REFRESH_FIELD: initial_refresh,
                                "initial_robot_refresh_control": initial_refresh_contract(initial_refresh)})
        (output / "teacher-control-config.json").write_text(
            json.dumps(mode_config, indent=2, allow_nan=False) + "\n")
        # Both arms reset the RNG at precisely this native reset boundary.
        seed = int(os.environ["DIAGNOSTIC_SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if os.environ.get("DIAGNOSTIC_RESEED_DATASET") == "1":
            # Versioned small-batch panel only: each cohort owns its native
            # sampling pool, so a previous policy's episode lengths cannot
            # decide when a later cohort's dataset is resampled.
            self.env.reset_dataset.resample()
            self.env.reset_count = 0
        if os.environ.get("DIAGNOSTIC_RESEED_DATASET") == "1":
            obs = self.env.reset_panel()
        else:
            obs = self.env.reset_all()
        pooled = getattr(self, '_raven_pooled_benchmark', None)
        initial_clocks = None
        if pooled is not None:
            from pipeline.success_rate.episode_protocol import randomize_initial_clocks
            initial_clocks = randomize_initial_clocks(self.env)
        self._diagnostic_after_reset(obs, output)
        initial_refresh_attestation = None
        if initial_refresh != ORIGINAL_INITIAL_REFRESH:
            initial_refresh_attestation = validate_initial_refresh_attestation(
                getattr(self.env, "_diagnostic_initial_refresh_attestation", None), num_envs=count, mode=initial_refresh)
            emit({"phase": "diagnostic_initial_robot_refresh_cleared", **initial_refresh_attestation})
        physics_evidence = runtime_physics_evidence(self.env)
        physics_bytes = json.dumps(physics_evidence, sort_keys=True, allow_nan=False).encode()
        (output / "runtime-physics.json").write_bytes(physics_bytes + b"\n")
        self.env.init_eval_metrics_tracking(self.accelerator.device)
        episode_reward = torch.zeros(count, device=self.accelerator.device)
        episode_length = torch.zeros(count, dtype=torch.int64, device=self.accelerator.device)
        state = initial_state(self.env, obs)
        np.savez(output / "initial-state.npz", **state)
        identity = {name: digest(value) for name, value in sorted(state.items())}
        (output / "initial-state.json").write_text(json.dumps(identity, indent=2) + "\n")
        if numeric_steps:
            from native_physics_readback import physics_readback
            try:
                effective_physics = physics_readback(self.env)
            except Exception as exc:
                effective_physics = {"status": "unavailable",
                    "error_type": type(exc).__name__, "read_only": True}
            (output / "effective-physics-readback.json").write_text(
                json.dumps(effective_physics, indent=2, allow_nan=False) + "\n")
        reference = os.environ.get("DIAGNOSTIC_REFERENCE_DIR", "/paired-teacher" if arm == "bc" else "")
        if reference:
            require_paired_states(json.loads((Path(reference) / "initial-state.json").read_text()), identity)
        else:
            # First arm provides the matched-state reference; comparison is deferred.
            emit({"phase": "reference_reset_saved", "arm": arm})
        OmegaConf.save(self.env.config, output / "environment-config.yaml", resolve=True)
        OmegaConf.save(self.config, output / "policy-config.yaml", resolve=True)
        original_optimizer_step = self.optimizer.step
        self.optimizer.step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("optimizer forbidden"))
        original = self.env._check_termination
        captured = {}
        numeric_after = {}
        numeric_active = False

        def observed_termination():
            original()
            captured.clear()
            captured.update(terminal_snapshot(self.env))
            if numeric_active:
                numeric_after.clear()
                numeric_after.update(numeric_state(self.env))

        self.env._check_termination = observed_termination
        if pooled is None:
            account = FirstEpisodeAccounting(count)
        else:
            from pipeline.success_rate.episode_protocol import PooledEpisodeAccounting
            account = PooledEpisodeAccounting(count, pooled['minimum_completed_episodes'])
        dones = torch.zeros(count, device=self.accelerator.device)
        # The new scaled operation opts into host-wall telemetry only. Keep
        # historical diagnostic paths and every native/policy call unchanged.
        from pipeline.success_rate.episode_metrics import SuccessCycleTimings
        cycle_timings = (SuccessCycleTimings()
            if getattr(self, '_scaled_initialization_distribution', None) is not None else None)
        started = time.monotonic()
        try:
            numeric_context = ((output / "numeric-control-trace.jsonl").open("w")
                               if numeric_steps else nullcontext(None))
            timing_context = ((output / 'cycle-timings.jsonl').open('x')
                              if cycle_timings is not None else nullcontext(None))
            with torch.no_grad(), (output / "steps.jsonl").open("w") as trace, numeric_context as numeric_trace, timing_context as timing_trace:
                for step in range(450 if pooled is None else pooled['max_vector_steps']):
                    if scaled_teacher is not None:
                        resource_guard = getattr(self, '_native_resource_guard', None)
                        if not callable(resource_guard):
                            raise RuntimeError('scaled teacher requires the installed native resource guard')
                        resource_guard()
                        teacher_guard_checks += 1
                    if pooled is not None:
                        resource_guard = getattr(self, '_native_resource_guard', None)
                        if not callable(resource_guard):
                            raise RuntimeError('pooled benchmark requires the installed native resource guard')
                        resource_guard()
                    cycle = cycle_timings.start(step) if cycle_timings is not None else None
                    tick = time.monotonic()
                    obs = {name: value.to(self.accelerator.device) for name, value in obs.items()}
                    if cycle is not None:
                        cycle.mark('capture_control_seconds')
                    # Student evaluation requires the admitted generated-image
                    # transform. A render failure stops before policy inference
                    # or physics; teacher diagnostics keep their separate path.
                    from pipeline.success_rate.student_observations import apply_observation_transform
                    if arm == "teacher":
                        # Privileged state goes straight to the teacher; no renderer.
                        if getattr(self, "_success_observation_transform", None) is not None:
                            raise RuntimeError("teacher evaluation must not install a student renderer")
                    else:
                        # All lanes render their current state before student inference.
                        obs = apply_observation_transform(self, obs, dones, step)
                    if pooled is not None:
                        resource_guard()
                    if cycle is not None:
                        cycle.mark('rendered_observation_seconds')
                    numeric_active = step < numeric_steps
                    if numeric_active:
                        numeric_before = {"teacher_obs": cpu(obs["teacher_obs"]).tolist(),
                            "homie_obs": cpu(obs["homie_obs"]).tolist(),
                            "control_state": numeric_state(self.env)}
                    actions, raw_teacher = select_policy_actions(self, obs, dones, arm=arm,
                        mode=teacher_mode, capture_raw=numeric_active)
                    chosen_before_step = cpu(actions["actions"]).tolist() if numeric_active else None
                    native_step_keys = sorted(actions) if numeric_active else None
                    if cycle is not None:
                        cycle.mark('student_homie_seconds')
                    if pooled is not None:
                        resource_guard()
                    if scaled_teacher is not None:
                        resource_guard()
                        teacher_guard_checks += 1
                    # One batched call advances every lane, not a Python step per lane.
                    obs, reward, dones, infos = self.env.step(actions)
                    if cycle is not None:
                        cycle.mark('env_step_seconds')
                    episode_reward += reward
                    episode_length += 1
                    self.env.update_eval_metrics_per_step(infos)
                    if not captured:
                        raise RuntimeError("pre-reset observer did not execute")
                    exported = cpu(infos["grasp_goal_reached"]).reshape(-1).astype(bool)
                    if not np.array_equal(exported, cpu(self.env._transition_goal_before_reset)):
                        raise RuntimeError("outgoing goal differs from pre-reset snapshot")
                    done = cpu(dones).reshape(-1).astype(bool)
                    native_done = captured["reset_buf"].reshape(-1).astype(bool)
                    if not np.array_equal(done, native_done):
                        raise RuntimeError("pre-reset/post-step done flags differ")
                    details = [lane_details(captured, lane, self.env.config) for lane in range(count)]
                    # Small evaluation receipts, not image/state rescans. Keep
                    # the actual native reward so ledger import never invents it.
                    rewards = cpu(episode_reward).reshape(-1)
                    for lane, detail in enumerate(details):
                        detail["episode_reward"] = float(rewards[lane])
                    completed = account.update(step, captured["current_last_stage_completed_task_buf"],
                                               exported, done, captured["time_out_buf"], details)
                    if completed:
                        first_ids = torch.tensor([[row["lane"]] for row in completed], device=self.accelerator.device)
                        self.env.process_eval_episode_completions(first_ids, episode_reward, episode_length)
                    done_ids = (dones > 0).nonzero()
                    if len(done_ids):
                        episode_reward[done_ids] = 0
                        episode_length[done_ids] = 0
                        self.env.reset_eval_episode_tracking(done_ids)
                    event = {"step": step, "arm": arm, "seconds": time.monotonic() - tick,
                             "lanes": details, "exported_goals": exported.tolist(),
                             "completed_outcomes": completed}
                    trace.write(json.dumps(event, allow_nan=False) + "\n")
                    trace.flush()
                    if numeric_active:
                        numeric_trace.write(json.dumps({"schema": "roborender.teacher-control-step.v1",
                            "step": step, "arm": arm, "teacher_action_mode": teacher_mode,
                            "pre": numeric_before, "raw_teacher_action_11": cpu(raw_teacher).tolist(),
                            "chosen_action_26_before_native_delta": chosen_before_step,
                            "env_step_keys": native_step_keys,
                            "post_step_before_auto_reset": numeric_after,
                            "first_completed_outcomes": completed}, allow_nan=False) + "\n")
                        numeric_trace.flush()
                    if completed or step % 25 == 0:
                        emit({"phase": "rollout", "arm": arm, "vector_steps": step + 1,
                              **account.result(), "elapsed_seconds": time.monotonic() - started})
                    if cycle is not None:
                        timing_trace.write(json.dumps(cycle_timings.complete(cycle), allow_nan=False) + '\n')
                        timing_trace.flush()
                    if account.result()["complete"]:
                        break
        finally:
            self.env._check_termination = original
            self.optimizer.step = original_optimizer_step
        unchanged = all(torch.equal(value.cpu(), expected[key]) for key, value in self.policy_model.state_dict().items())
        if not unchanged:
            raise RuntimeError("student state changed during inference-only diagnostic")
        teacher_attestation_at_start = teacher_attestation
        if teacher_attestation is not None:
            # Check the final cohort too, not merely the next cohort's entry.
            # Original PPO may legally clamp only its discarded-sample std.
            teacher_attestation = prepare_ppo_teacher(
                self, original_ppo=teacher_mode == RAVEN_PPO)
        native_metric = self.env.eval_metrics["goal_reached_buffer"]
        if len(native_metric) != account.result()["completed"] or sum(native_metric) != account.result()["exported_successes"]:
            raise RuntimeError("native Raven metric and diagnostic aggregation disagree")
        result = {"schema": "roborender.teacher-bc-success-diagnostic.v1", "arm": arm,
                  "teacher_action_mode": teacher_mode, "teacher_control": control_contract(teacher_mode),
                  "teacher_parameter_attestation": teacher_attestation,
                  "teacher_parameter_attestation_at_start": teacher_attestation_at_start,
                  "numeric_trace_steps_requested": numeric_steps,
                  "numeric_trace_steps_recorded": min(step + 1, numeric_steps),
                  "numeric_trace_sha256": (hashlib.sha256((output / "numeric-control-trace.jsonl").read_bytes()).hexdigest()
                                             if numeric_steps else None),
                  "teacher_control_config_sha256": hashlib.sha256(
                      (output / "teacher-control-config.json").read_bytes()).hexdigest(),
                  "checkpoint_sha256": checkpoint_sha, "seed": seed, "observation_mode": "raytraced_3view",
                  "optimizer_steps": 0, "policy_unchanged": unchanged, "vector_steps": step + 1,
                  "goal_accounting_version": GOAL_ACCOUNTING_VERSION,
                  "runtime_physics_sha256": hashlib.sha256(physics_bytes).hexdigest(),
                  "original_raven_metric_successes": sum(native_metric),
                  "original_raven_metric_completed": len(native_metric),
                  "elapsed_seconds": time.monotonic() - started, "benchmark_certified": False,
                  "runtime_port": json.loads(Path("/isaac6-port/port-manifest.json").read_text()),
                  **account.result()}
        from pipeline.success_rate.student_observations import observation_receipt
        result.update(observation_receipt(self, step + 1))
        if cycle_timings is not None:
            result['scaled_cycle_timing'] = cycle_timings.receipt(expected_steps=step + 1)
        if initial_refresh != ORIGINAL_INITIAL_REFRESH:
            result.update({INITIAL_REFRESH_FIELD: initial_refresh,
                "initial_robot_refresh_control": initial_refresh_contract(initial_refresh),
                "initial_robot_refresh_attestation": initial_refresh_attestation})
        if pooled is not None:
            from pipeline.success_rate.episode_protocol import RESULT_SCHEMA, digest as benchmark_digest, validate_result
            result.update(schema=RESULT_SCHEMA, benchmark=pooled, benchmark_sha256=benchmark_digest(pooled),
                initial_clocks=initial_clocks, training_curriculum_restored=False,
                native_episode_resets=self._pooled_native_resets,
                native_reset_resample_every=int(self.env.config.reset_from_dataset.resample_every))
            if result['complete']:
                validate_result(result, pooled)
        if scaled_teacher is not None:
            result.update(teacher_resource_guard_checks=teacher_guard_checks,
                          training_curriculum_restored=False)
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        emit({key: value for key, value in result.items() if key != "runtime_port"})
        if not result["complete"]:
            raise RuntimeError("bounded diagnostic ended without its declared episode completion target")
        return result
