"""Persistent native B5 evaluation using the shared tested goal accounting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
from gr00t.rl.utils.torch_utils import quat_rotate

from isaac6_success_diagnostic import SuccessAccountingEnvironment, SuccessDiagnosticTrainer, emit
from beta_small_batch_panel import canonical, summarize, validate_panel
from warm_panel_session import read_request, run_pair
from dagger_rtx.process_memory import compact_host_heap
from initial_robot_refresh_diagnostic import (
    ORIGINAL as ORIGINAL_INITIAL_REFRESH, CLEAR_ONCE, clear_already_applied_initial_refresh,
    require_corrected_execution,
)


class PanelResetEnvironment(SuccessAccountingEnvironment):
    """Independent evaluation cohorts, without restarting the native simulator.

    Raven's regular episode reset intentionally retains some command state.
    A new benchmark cohort must instead restore the same cold command state
    and refresh reset kinematics before constructing its first observation.
    No physics integration or policy action occurs in this operation.
    """
    def reset_panel(self):
        initial_refresh = require_corrected_execution(
            os.environ.get("DIAGNOSTIC_INITIAL_ROBOT_REFRESH"),
            arm=os.environ.get("DIAGNOSTIC_ARM", "bc"),
            teacher_action_mode=os.environ.get("DIAGNOSTIC_TEACHER_ACTION_MODE", "distill"),
            num_envs=self.num_envs)
        obs = self._reset_cold_scene()
        self._diagnostic_initial_refresh_clear_calls = 0
        self._diagnostic_initial_refresh_attestation = clear_already_applied_initial_refresh(
            self, mode=initial_refresh, arm=os.environ.get("DIAGNOSTIC_ARM", "bc"),
            teacher_action_mode=os.environ.get("DIAGNOSTIC_TEACHER_ACTION_MODE", "distill"))
        return obs

    def _reset_cold_scene(self):
        """Shared cold-cohort mechanics, not a success-evaluation entrypoint.

        The versioned shared reset_all overlay consumes already-written flags
        for training and testing alike. reset_panel still requires its explicit
        success protocol and records the same once-only correction receipt.
        Ordinary done resets never enter this explicit phase boundary.
        """
        names = ("_homie_commands", "_homie_commands_unclipped", "_last_homie_commands")
        if not hasattr(self, "_panel_cold_commands"):
            self._panel_cold_commands = {n: getattr(self, n).detach().clone() for n in names}
        for name, value in self._panel_cold_commands.items():
            getattr(self, name).copy_(value)
        # Native history.reset multiplies by zero, retaining policy-dependent
        # signed zeros. Start a cohort with actual cold zero history instead.
        for history in self.history_handler.history.values():
            history.zero_()
        # Lab reset-event cooldowns otherwise depend on the preceding policy's
        # episode lengths. A skipped appearance event consumes fewer random
        # draws, changing even the subsequent native motion sample selection.
        # Reset only this explicit independent-cohort boundary to cold clocks;
        # ordinary episode resets retain Raven's 20-step event cooldown.
        manager = self.simulator.event_manager
        names = ("_reset_term_last_triggered_step_id", "_reset_term_last_triggered_once")
        if any(not isinstance(getattr(manager, n, None), list) for n in names):
            raise RuntimeError("unknown Isaac reset-event timer contract")
        clocks, flags = (getattr(manager, n) for n in names)
        if len(clocks) != len(flags):
            raise RuntimeError("inconsistent Isaac reset-event timer fields")
        for clock, flag in zip(clocks, flags):
            if clock.shape != (self.num_envs,) or flag.shape != clock.shape or flag.dtype != torch.bool:
                raise RuntimeError("unexpected Isaac reset-event timer shape/dtype")
            clock.zero_()
            flag.zero_()
        self.common_step_counter = 0
        self.simulator.scene.reset()
        self._panel_refresh_pending = True
        try:
            from initial_reset_contract import require_completed_initial_reset
            obs = super().reset_all()
            require_completed_initial_reset(self)
            return obs
        finally:
            self._panel_refresh_pending = False

    def _refresh_sim_tensors(self):
        panel_reset = getattr(self, "_panel_refresh_pending", False)
        if panel_reset:
            self.simulator.sim.forward()
            self.simulator.scene.update(self.simulator.sim.cfg.dt)
            self._panel_refresh_pending = False
        result = super()._refresh_sim_tensors()
        if panel_reset:
            # PhysX contact views report forces from the last integration;
            # forwarding freshly reset poses does not run that integration.
            # Discard those *previous episode* force observations at the cold
            # cohort boundary. Native step refresh replaces them normally.
            for name in ("object_to_hand_contact_forces", "hold_object_to_hand_contact_forces",
                         "front_object_to_hand_contact_forces", "front_object_table_contact_forces",
                         "table_to_hand_contact_forces", "table_to_body_contact_forces"):
                if hasattr(self.simulator, name):
                    getattr(self.simulator, name).zero_()
            # Raven derives this goal during reset before the new root/DOF
            # states have been written. Repeat its exact expression after the
            # kinematic refresh, not with the preceding policy's pelvis pose.
            offset = torch.tensor(self.config.lift_goal_offset_pelvis, device=self.device)
            offset = offset.unsqueeze(0).repeat(self.num_envs, 1)
            self.lift_goal_buf[:] = (self.simulator._rigid_body_pos[:, self.pelvis_idx, :]
                                    + quat_rotate(self.simulator.base_quat, offset))
            self.last_lift_distance[:] = (self.lift_goal_buf - self.simulator.get_task_root_state(
                self.config.task.front_object)[:, :3]).norm(dim=-1)
        return result


class BetaPanelEvaluator(SuccessDiagnosticTrainer):
    def train(self):
        return self.evaluate_panel(Path("/dagger-runtime"), os.environ.get("BETA_PANEL_REFERENCE", ""))

    def evaluate_panel(self, root, reference_root=""):
        root = Path(root).resolve()
        if not root.is_relative_to(Path("/dagger-runtime")):
            raise RuntimeError("panel output must remain in its owned runtime")
        root.mkdir(parents=True, exist_ok=True)
        panel_bytes = Path(os.environ.get("BETA_PANEL_PATH", "/panel.json")).read_bytes()
        panel = validate_panel(json.loads(panel_bytes))
        expected_sha = os.environ["BETA_PANEL_SHA256"]
        if hashlib.sha256(panel_bytes).hexdigest() != expected_sha or panel_bytes != canonical(panel):
            raise RuntimeError("immutable panel identity/encoding differs")
        cohort_limit = int(os.environ.get("BETA_PANEL_COHORT_LIMIT", "52"))
        if cohort_limit not in (2, 52):
            raise RuntimeError("only the two-cohort reset probe or complete panel is allowed")
        results = []
        compact_host_heap("evaluation_native_setup_complete")
        for cohort in panel["cohorts"][:cohort_limit]:
            name = f"cohort-{cohort['index']:03d}"
            output = root / name
            output.mkdir(exist_ok=False)
            if reference_root:
                # Optional concurrent BC/candidate workers: wait for the
                # reference cohort's completed evidence, not an open JSON file.
                reference_done = Path(reference_root) / name / "result.json"
                deadline = time.monotonic() + 600
                while not reference_done.is_file():
                    if time.monotonic() > deadline:
                        raise RuntimeError("paired reference cohort did not finish: " + name)
                    time.sleep(2)
            os.environ.update(DIAGNOSTIC_OUTPUT_DIR=str(output),
                              DIAGNOSTIC_SEED=str(cohort["seed"]), DIAGNOSTIC_RESEED_DATASET="1",
                              DIAGNOSTIC_REFERENCE_DIR=str(Path(reference_root) / name) if reference_root else "")
            emit({"phase": "cohort_start", "cohort_index": cohort["index"], "seed": cohort["seed"],
                  "case_ids": [c["case_id"] for c in cohort["cases"]]})
            result = super().train()
            results.append(result)
            compact_host_heap("evaluation_cohort_complete")
            scored = [r for spec, row in zip(panel["cohorts"], results)
                      for r in row["outcomes"] if r["lane"] in {c["lane"] for c in spec["cases"]}]
            progress = {"phase": "panel_progress", "arm": os.environ["DIAGNOSTIC_ARM"],
                        "cohorts_completed": len(results), "completed_episodes": len(scored),
                        "successes": sum(r["exported_success"] for r in scored), "expected_episodes": panel["episode_count"],
                        "panel_sha256": expected_sha, "optimizer_steps": 0}
            temp = root / "progress.tmp"
            temp.write_bytes(canonical(progress))
            temp.replace(root / "progress.json")
            emit(progress)
        if cohort_limit < len(panel["cohorts"]):
            result = {"schema": "roborender.beta-small-batch-reset-probe.v1", "complete": False,
                      "probe_complete": True, "completed_episodes": 10, "expected_episodes": 256,
                      "panel_sha256": expected_sha, "paired_initial_states": bool(reference_root),
                      "optimizer_steps": 0, "cohorts": results}
            (root / "result.json").write_bytes(canonical(result))
            emit({k: v for k, v in result.items() if k != "cohorts"})
            return
        result = summarize(panel, results)
        for key in ("runtime_physics_sha256", "checkpoint_sha256", "goal_accounting_version"):
            values = {r[key] for r in results}
            if len(values) != 1:
                raise RuntimeError("cohort invariant changed: " + key)
            result[key] = values.pop()
        result["paired_initial_states"] = bool(reference_root)
        result["policy_unchanged"] = True
        (root / "result.json").write_bytes(canonical(result))
        emit({k: v for k, v in result.items() if k != "outcomes"})
        return result


class WarmPanelEvaluator(BetaPanelEvaluator):
    """Reuse one B5 Isaac/camera/Homie lifecycle for a complete BC/candidate pair.

    Each cohort still performs the qualified cold-state reset. Only the
    simulator application is warm; no preceding policy state/goal is reused.
    """
    def train(self):
        root = Path("/dagger-runtime")
        panel = validate_panel(json.loads(Path("/panel.json").read_bytes()))
        request = read_request("/warm-request.json", os.environ["WARM_PANEL_REQUEST_SHA256"], panel)
        identity = (id(self.env), id(self.env.simulator), id(self.env.simulator.sim))
        started = time.monotonic()
        keys = ("DIAGNOSTIC_ARM", "DIAGNOSTIC_CHECKPOINT_PATH", "DIAGNOSTIC_CHECKPOINT_SHA256")
        saved = {key: os.environ.get(key) for key in keys}

        def report(row):
            value = {**row, "native_pid": os.getpid(), "native_initializations": 1,
                     "updated_at_ns": time.time_ns()}
            temporary = root / "warm-progress.part"
            temporary.write_bytes(canonical(value))
            temporary.replace(root / "warm-progress.json")
            emit(value)

        def evaluate(*, arm, checkpoint, checkpoint_sha256, output, reference):
            if identity != (id(self.env), id(self.env.simulator), id(self.env.simulator.sim)):
                raise RuntimeError("warm evaluator native owner changed")
            os.environ.update(DIAGNOSTIC_ARM=arm, DIAGNOSTIC_CHECKPOINT_PATH=checkpoint,
                              DIAGNOSTIC_CHECKPOINT_SHA256=checkpoint_sha256)
            result = self.evaluate_panel(output, str(reference) if reference else "")
            if identity != (id(self.env), id(self.env.simulator), id(self.env.simulator.sim)):
                raise RuntimeError("warm evaluator reinitialized native simulation")
            return result

        report({"phase": "warm_evaluator_ready"})
        try:
            _, receipt = run_pair(request, panel, root, evaluate, report=report)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        receipt.update(native_pid=os.getpid(), native_initializations=1,
                       checkpoint_switches=1, seconds=time.monotonic() - started,
                       residency="within_validation_phase")
        (root / "warm-session.json").write_bytes(canonical(receipt))
        report({"phase": "warm_evaluator_complete", "seconds": receipt["seconds"]})
