"""CPU-only, receipt-driven four-round automation for the beta deployment.

Host operations are explicitly bound and SHA-pinned, not inferred SSH aliases.
One scheduler owns the campaign. It may reattach to running host operations,
but never restarts a lost Isaac lifecycle or learns a partial window.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import time

from pipeline.dagger.production_controller import Controller
from pipeline.dagger.beta_validation import import_comparison, evaluation_id
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    validate_config, canonical, actor_ids, ROUND_BETAS, budgets, is_eight_h200,
)
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.beta_operations import CommandOperations, atomic
from pipeline.deployment.native_phase_barrier import native_quiescent


def immutable_json(path, value):
    raw = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("refusing to change a frozen round configuration")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


def next_round_config(previous, first_config_sha, final_candidate, checkpoint_path, gateway_pids):
    """Bind the caller's protocol-selected parent to the next teacher mixture.

    Legacy fixed-budget callers pass their final candidate; the separate MSE
    campaign passes its immutable minimum-validation checkpoint and AdamW.
    """
    validate_config(previous)
    if previous["round_id"] >= 4:
        raise ValueError("there is no fifth DAgger round")
    value = deepcopy(previous)
    rid = value["round_id"] + 1
    value.update(round_id=rid, teacher_beta=ROUND_BETAS[rid],
                 student_action_probability=1 - ROUND_BETAS[rid],
                 seed=value["seed"] + 100003, existing_run_config_sha256=first_config_sha)
    value["student"].update(initial_checkpoint=str(checkpoint_path), initial_checkpoint_sha256=final_candidate)
    value["h200"]["gateway_pids"] = list(gateway_pids)
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_production,LEO_MSE_SCALED_SCHEMA,is_fixed100
    value["logging"]["wandb_run_name"] = (f"2026-10-16-prod-1-dagger-teacher{int(100 * ROUND_BETAS[rid])}"
        if is_mse_production(value) else value["run_id"] + f"-r{rid}-teacher{int(100 * ROUND_BETAS[rid])}")
    if value['schema']==LEO_MSE_SCALED_SCHEMA or is_fixed100(value):
        value['logging']['wandb_run_name']=f"{value['run_id'].removeprefix('dagger-leo-')}-dagger-teacher{int(100*ROUND_BETAS[rid])}"
    value["logging"]["wandb_run_id"] = ("prod1-" if is_mse_production(value) else "beta-") + hashlib.sha256(
        canonical([value["run_id"], rid])).hexdigest()[:24]
    return validate_config(value)


def bind_preparation(config, receipt):
    """New runtime identities come only from an actual prepared service receipt."""
    if (receipt.get('source_sha256') != config['source_sha256']
            or receipt.get('resource_shape') != config['h200']['resource_shape']
            or receipt.get('reservation') != 'cosmos-solution-architect-xcwtffiq'
            or receipt.get('allow_burst') is not False or receipt.get('genuine_backends') is not True
            or receipt.get('owned_control_plane') is not True
            or not isinstance(receipt.get('service_state_identity'), str)
            or len(receipt['service_state_identity']) != 64):
        raise RuntimeError('prepared service/resource identities differ')
    value = deepcopy(config)
    value['h200']['gateway_pids'] = receipt['gateway_pids']
    if is_eight_h200(config):
        attachment = receipt.get('retained_renderer_attachment', {})
        if (attachment.get('schema') != 'roborender.retained-eight-renderer-attachment.v1'
                or len(attachment.get('renderers', [])) != 8
                or attachment.get('renderer_restarts') != 0 or attachment.get('allocation_retained') is not True
                or attachment.get('compute_release_authorized') is not False):
            raise RuntimeError('actual retained8 renderer attachment is required')
        identities = receipt.get('gateway_identities', [])
        if ([r.get('gpu_uuid') for r in identities] != [r.get('gpu_uuid') for r in attachment['renderers']]
                or receipt.get('gpu_uuids') != [r.get('gpu_uuid') for r in identities]):
            raise RuntimeError('actual renderer/gateway/allocation GPU mappings differ')
        value['h200']['gateway_identities'] = identities
    return validate_config(value, runtime=True)


def validate_actor_completion(config, receipt, iterations):
    expected = actor_ids(config)
    windows = budgets(config)[0]
    rows = sorted((r for r in iterations if r["round_id"] == config["round_id"]),
                  key=lambda r: r["iteration_id"])
    if ([r["iteration_id"] for r in rows] != list(range(1, windows + 1))
            or any(r["status"] != "advanced" for r in rows[:-1])
            or rows[-1]["status"] not in {"awaiting_validation", "validating"}):
        raise RuntimeError("all declared iterations must finish before evaluation")
    parent = config["student"]["initial_checkpoint_sha256"]
    for row in rows:
        if row["input_checkpoint_sha256"] != parent or not row["candidate_checkpoint_sha256"]:
            raise RuntimeError("checkpoint chain broken")
        parent = row["candidate_checkpoint_sha256"]
    actors = receipt.get("actors", [])
    if len(actors) != 2 or {a.get("physical_gpu") for a in actors} != {a["physical_gpu_index"] for a in config["rtx_actors"]}:
        raise RuntimeError("both exact beta actors must finish")
    for actor in actors:
        if (actor.get("run_id") != config["run_id"] or actor.get("round_id") != config["round_id"]
                or actor.get("phase") != "COMPLETE" or actor.get("uncertain_reason") is not None
                or actor.get("lifecycle_initializations") != 1
                or actor.get("transition_batches") != 81 * windows
                or actor.get("physical_transition_rows") != 81 * windows * 6
                or actor.get("final_candidate_checkpoint_sha256") != parent
                or actor.get("config_sha256") != hashlib.sha256(canonical(config)).hexdigest()
                or [(r.get("index"), r.get("input_checkpoint_sha256"), r.get("output_checkpoint_sha256"))
                    for r in actor.get("windows", [])] != [
                        (r["iteration_id"] - 1, r["input_checkpoint_sha256"], r["candidate_checkpoint_sha256"])
                        for r in rows]):
            raise RuntimeError("actor lifecycle/count/checkpoint receipt differs")
    if set(receipt.get("actor_ids", [])) != set(expected) or not native_quiescent(receipt, source_sha256=config["source_sha256"]):
        raise RuntimeError("actor shutdown/identity receipt missing")
    return parent


class Campaign:
    supports_explicit_mse_smoke = False

    def __init__(self, first_config, root, operations, *, controller_factory=Controller,
                 log_validation=None, begin_logging=None, clock=time.time):
        validate_config(first_config)
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_smoke
        if is_mse_smoke(first_config) and not self.supports_explicit_mse_smoke:
            raise ValueError("explicit B4 MSE smoke requires MSECampaign, never the legacy success-selection controller")
        if first_config["round_id"] != 1:
            raise ValueError("campaign must be bound to its original BC-starting round1")
        self.root, self.ops, self.factory, self.log_validation = Path(root).resolve(), operations, controller_factory, log_validation
        self.clock, self.first = clock, first_config
        self.begin_logging = begin_logging
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        original_sha = hashlib.sha256(canonical(first_config)).hexdigest()
        if self.path.exists():
            self.state = json.loads(self.path.read_bytes())
            if self.state.get("template_sha256") != original_sha:
                raise RuntimeError("campaign configuration changed on resume")
        else:
            self.state = {"schema": "roborender.beta-campaign-state.v1", "run_id": first_config["run_id"],
                          "template_sha256": original_sha, "phase": "PREPARE", "round_id": 1,
                          "config": first_config, "completed_rounds": [], "operation_id": None}
            atomic(self.path, self.state)
        if "continuation_source_sha256" in self.state:
            admission_path = self.root / "recovery-admission.json"
            admission = json.loads(admission_path.read_bytes())
            if (file_sha256(admission_path) != self.state.get("recovery_admission_sha256")
                    or admission.get("schema") != "roborender.completed-round-recovery.v1"
                    or admission.get("run_id") != self.state["run_id"]
                    or admission.get("continuation_source_sha256") != self.state["continuation_source_sha256"]
                    or admission.get("native_restart_allowed") is not False
                    or admission.get("completed_training_repeated") is not False):
                raise RuntimeError("completed-round recovery admission changed or missing")
        if "boundary_recovery" in self.state:
            from pipeline.deployment.round_boundary_recovery import validate_binding
            validate_binding(self.root, self.state)
        if "pending_evaluation_recovery" in self.state:
            from pipeline.deployment.pending_evaluation_recovery import validate_binding
            validate_binding(self.root, self.state)
        self.controller = None

    def save(self, phase=None, **updates):
        old_phase = self.state["phase"]
        if phase:
            self.state["phase"] = phase
        self.state.update(updates, updated_at=self.clock())
        atomic(self.path, self.state)
        if phase and phase != old_phase:
            event = {"run_id": self.state["run_id"], "round_id": self.state["round_id"],
                     "phase": phase, "previous_phase": old_phase, "time": self.clock()}
            with (self.root / "phases.jsonl").open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)

    def operation_context(self):
        context = {"run_id": self.state["run_id"], "round_id": self.state["round_id"],
                       "config": self.state["config"], "campaign_root": str(self.root.resolve()),
                       "round_config_path": self.state.get("config_path"),
                       "final_checkpoint": self.state.get("final_checkpoint"),
                       "service_binding": self.state.get("service_binding"),
                       "baseline_checkpoint_sha256": self.first["student"]["initial_checkpoint_sha256"]}
        context["baseline_checkpoint_path"] = self.first["student"]["initial_checkpoint"]
        if "infrastructure_attempts" in self.state:
            context["infrastructure_attempts"] = self.state["infrastructure_attempts"]
        if "pending_evaluation_recovery" in self.state:
            context["evaluation_attempt"] = self.state["pending_evaluation_recovery"]["recovery_id"]
        if "control_only_release" in self.state:
            # Explicit reconciled pre-dispatch recovery gets a fresh operation
            # identity. This never relaunches an accepted native command.
            release = self.state["control_only_release"]
            if (set(release) != {"path", "sha256"}
                    or file_sha256(Path(release["path"])) != release["sha256"]):
                raise RuntimeError("control-only release binding changed")
            context["control_only_release"] = release
        return context

    def operation(self, phase):
        if self.state["operation_id"] is None:
            self.save(operation_id=self.ops.start(phase, self.operation_context()))
        return self.ops.poll(self.state["operation_id"])

    def get_controller(self):
        if self.controller is None:
            binding = self.state["service_binding"]
            self.controller = self.factory(Path(self.state["config_path"]), Path(binding["token_file"]), wandb_mode="online")
            self.controller.ensure_registered()
        return self.controller

    def tick(self):
        phase = self.state["phase"]
        config = self.state["config"]
        windows, episodes, _ = budgets(config)
        if phase == "COMPLETE":
            return dict(self.state)
        if phase == "PREPARE":
            receipt = self.operation("prepare")
            if receipt is None:
                return dict(self.state)
            value = bind_preparation(config, receipt)
            path = self.root / f"round-{value['round_id']}" / "config.json"
            digest = immutable_json(path, value)
            self.save("REGISTER", operation_id=None, config=value, config_path=str(path.resolve()),
                      first_config_sha256=self.state.get("first_config_sha256", digest), service_binding=receipt)
        elif phase == "REGISTER":
            self.get_controller()
            if self.begin_logging is not None:
                logging = self.begin_logging(config)
                if logging.get("online_metrics_verified") is not True:
                    raise RuntimeError("W&B round registration was not verified")
                self.save(round_logging=logging)
            self.save("COLLECT_TRAIN", operation_id=None, collection_deadline=self.clock() + 21600)
        elif phase == "COLLECT_TRAIN":
            controller = self.get_controller()
            receipt = self.operation("collect")
            progress = controller.tick()
            if progress["action"] == "collection_budget_complete":
                raise RuntimeError("beta coordinator advanced past the mandatory final evaluation")
            if progress != self.state.get("progress"):
                self.save(progress=progress)
                print(json.dumps({"phase": phase, **progress}), flush=True)
            if self.clock() > self.state["collection_deadline"]:
                raise TimeoutError("round collection/learner deadline; do not restart Isaac")
            if receipt is not None:
                status = controller.coordinator.get(f"/v1/status/{config['run_id']}")
                candidate = validate_actor_completion(config, receipt, status["iterations"])
                job_id = f"{config['run_id']}-dagger-{config['round_id']}-iteration-{windows:04d}"
                job = controller.learner.get(f"/v1/jobs/{job_id}")["job"]
                result = job.get("result")
                if (job.get("status") != "succeeded" or not result
                        or result["output_checkpoint_sha256"] != candidate
                        or result["iteration_id"] != windows or result["train_chunks"] != 12 or result["train_frames"] != 972
                        or result.get("replay_manifest_sha256") is not None):
                    raise RuntimeError("final learner receipt differs from the completed actor chain")
                checkpoint = Path(result["output_checkpoint_path"])
                if file_sha256(checkpoint) != candidate:
                    raise RuntimeError("final checkpoint bytes are missing or corrupt")
                self.save("RELEASE_TRAINING", operation_id=None,
                          final_checkpoint={"path": str(checkpoint), "sha256": candidate}, final_learner=result)
        elif phase == "RELEASE_TRAINING":
            receipt = self.operation("release")
            if receipt is not None:
                if not native_quiescent(receipt, source_sha256=config["source_sha256"]) or receipt.get("h200_allocation_released") is not True:
                    raise RuntimeError("GPU allocation must be released before the RTX-only evaluation")
                self.controller = None
                self.save("EVALUATE", operation_id=None, release_receipt=receipt)
        elif phase == "EVALUATE":
            receipt = self.operation("evaluate")
            if receipt is not None:
                path = Path(receipt["comparison_path"])
                if file_sha256(path) != receipt["comparison_sha256"] or not native_quiescent(receipt, source_sha256=config["source_sha256"]):
                    raise RuntimeError("evaluation evidence or actor exit receipt differs")
                comparison = json.loads(path.read_bytes())
                from pipeline.dagger.beta_validation import outcome_payloads
                outcome_payloads(config, comparison, self.first["student"]["initial_checkpoint_sha256"],
                                 self.state["final_checkpoint"]["sha256"])
                self.save("CONTROL_ONLY", operation_id=None, evaluation_receipt=receipt)
        elif phase == "CONTROL_ONLY":
            # The same persistent ledger is opened by CPU-only control roles,
            # after its GPU-host owners stopped. No WAL is read across hosts.
            receipt = self.operation("control_only")
            if receipt is not None:
                if (receipt.get("h200_gpu_count") != 0 or receipt.get("exclusive_ledger_owner") is not True
                        or receipt.get("service_state_identity") != self.state["service_binding"].get("service_state_identity")):
                    raise RuntimeError("CPU-only control restoration must preserve exclusive ledger ownership")
                self.save("IMPORT_VALIDATION", operation_id=None, service_binding=receipt)
        elif phase == "IMPORT_VALIDATION":
            controller = self.get_controller()
            evidence = self.state["evaluation_receipt"]
            if file_sha256(Path(evidence["comparison_path"])) != evidence["comparison_sha256"]:
                raise RuntimeError("evaluation comparison changed after acceptance")
            comparison = json.loads(Path(evidence["comparison_path"]).read_bytes())
            report = import_comparison(controller, comparison, self.first["student"]["initial_checkpoint_sha256"],
                                       self.state["final_checkpoint"]["sha256"])
            self.save("LOG_VALIDATION", evaluation_report=report)
        elif phase == "LOG_VALIDATION":
            if self.log_validation is None:
                raise RuntimeError("online W&B validation logging must be bound before advancing")
            logging = self.log_validation(config, self.state["final_learner"], self.state["evaluation_report"])
            if logging.get("online_metrics_verified") is not True:
                raise RuntimeError("W&B metric readback failed; no next round")
            self.save("ADVANCE", logging_receipt=logging)
        elif phase == "ADVANCE":
            controller = self.get_controller()
            ack = controller.coordinator.post("/v1/iterations/advance", {
                "run_id": config["run_id"], "round_id": config["round_id"], "evaluation_id": evaluation_id(config)})
            if (not ack.get("ok") or ack.get("completed_iteration_id") != windows
                    or ack.get("next_iteration_id") is not None or ack.get("stop_reason") != "round_iteration_budget"):
                raise RuntimeError("final success validation did not close exactly this round")
            completed = self.state["completed_rounds"] + [{
                "round_id": config["round_id"], "teacher_beta": config["teacher_beta"],
                "final_checkpoint": self.state["final_checkpoint"],
                "successes": self.state["evaluation_report"]["successes"], "completed_episodes": episodes,
                "wandb": self.state["logging_receipt"], "evaluation": self.state["evaluation_receipt"]}]
            if config["round_id"] == 4:
                self.save("COMPLETE", completed_rounds=completed)
            else:
                next_config = next_round_config(config, self.state["first_config_sha256"],
                    self.state["final_checkpoint"]["sha256"], self.state["final_checkpoint"]["path"], config["h200"]["gateway_pids"])
                if "continuation_source_sha256" in self.state:
                    # Adopt the attested infrastructure release only at a real
                    # round lifecycle boundary, never inside a native window.
                    next_config["source_sha256"] = self.state["continuation_source_sha256"]
                    validate_config(next_config, runtime=True)
                self.controller = None
                self.save("PREPARE", config=next_config, round_id=next_config["round_id"], operation_id=None,
                          completed_rounds=completed, progress=None)
        else:
            raise RuntimeError("unknown campaign phase: " + phase)
        return dict(self.state)


def online_loggers(entity, key_file=None):
    from pipeline.deployment.wandb_preflight import admit_logging, verify_remote_summary
    api, identity = admit_logging("online", entity, key_file)

    def begin(config):
        import wandb
        run_id = config["logging"]["wandb_run_id"]
        run = wandb.init(project="dagger", entity=entity, id=run_id, resume="allow", mode="online",
                         name=config["logging"]["wandb_run_name"], config={"campaign": config},
                         settings=wandb.Settings(init_timeout=45))
        try:
            metrics = {"orchestration/round_id": config["round_id"], "orchestration/collection_started": False}
            run.log(metrics)
            run.summary.update(metrics)
            url = run.url
        finally:
            run.finish()
        verify_remote_summary(api, entity, run_id, metrics)
        return {"online_metrics_verified": True, "run_id": run_id, "url": url}

    def log(config, learner, report):
        import wandb
        run_id = config["logging"]["wandb_run_id"]
        # Prove real final training telemetry arrived before adding evaluation.
        windows, episodes, _ = budgets(config)
        verify_remote_summary(api, entity, run_id, {"last_iteration_id": windows, "last_train_mse": learner["train_mse"]})
        run = wandb.init(project="dagger", entity=entity, id=run_id, resume="must", mode="online")
        metrics = {"iteration_id": windows, "validation/goal_reached_rate": report["goal_reached_rate"],
                   "validation/successes": report["successes"], "validation/completed_episodes": episodes,
                   "orchestration/collection_started": True, "orchestration/round_complete": True}
        try:
            run.log(metrics)
            run.summary.update({"validation_status": "complete", "final_checkpoint_sha256": report["checkpoint_sha256"]})
            url = run.url
        finally:
            run.finish()
        verify_remote_summary(api, entity, run_id, metrics)
        return {"online_metrics_verified": True, "run_id": run_id, "url": url, "entity": identity["entity"]}
    return begin, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-round-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--operations", type=Path)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only print a non-mutating plan")
    parser.add_argument("--wandb-entity", default="nvidia")
    parser.add_argument("--wandb-key-file", type=Path)
    args = parser.parse_args()
    config = validate_config(json.loads(args.first_round_config.read_bytes()))
    if not args.execute:
        print(json.dumps({"run_id": config["run_id"], "allocation_performed": False, "launch_performed": False,
                          "rounds": [{"round_id": rid, "teacher_beta": beta, "iterations": budgets(config)[0],
                                      "final_success_episodes": budgets(config)[1]} for rid, beta in ROUND_BETAS.items()],
                          "replay": False, "phases": ["prepare", "collect/train", "release H200", "RTX evaluate",
                                                      "CPU ledger/logging", "next round"]}, indent=2))
        return
    if args.operations is None:
        parser.error("--execute requires SHA-pinned host operations; no implicit GPU allocation")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = json.loads(args.operations.read_bytes())
        state_path = args.root / "state.json"
        saved = json.loads(state_path.read_bytes()) if state_path.exists() else {}
        if "pending_evaluation_recovery" in saved:
            from pipeline.deployment.pending_evaluation_recovery import validate_binding
            validate_binding(args.root, saved)
            recovery = saved["pending_evaluation_recovery"]["recovery_id"]
            binding_path = args.root / "recoveries" / recovery / "operations.json"
        elif "boundary_recovery" in saved:
            from pipeline.deployment.round_boundary_recovery import validate_binding
            validate_binding(args.root, saved)
            recovery = saved["boundary_recovery"]["recovery_id"]
            binding_path = args.root / "recoveries" / recovery / "operations.json"
        else:
            binding_path = args.root / "operations.json"
        binding_sha = immutable_json(binding_path, binding)
        operations = CommandOperations(binding, args.root / "operations")
        begin, logger = online_loggers(args.wandb_entity, args.wandb_key_file)
        campaign = Campaign(config, args.root, operations, log_validation=logger, begin_logging=begin)
        campaign.save(operations_sha256=binding_sha)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("campaign stop requested")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        try:
            heartbeat_at = 0.
            while campaign.state["phase"] != "COMPLETE":
                if time.monotonic() - heartbeat_at >= 15:
                    atomic(args.root / "heartbeat.json", {"run_id": config["run_id"], "updated_at_ns": time.time_ns()})
                    heartbeat_at = time.monotonic()
                campaign.tick()
                time.sleep(1)
        except BaseException as exc:
            # A host driver performs scoped cleanup, never an arbitrary kill.
            # Native uncertainty is preserved; a new scheduler must reconcile.
            atomic(args.root / "failure.json", {"phase": campaign.state["phase"], "error_type": type(exc).__name__,
                                               "time": time.time(), "native_restart_allowed": False})
            abort_id = operations.start("abort", {**campaign.operation_context(), "native_restart_allowed": False})
            deadline = time.monotonic() + binding["commands"]["abort"]["timeout_seconds"] + 5
            while True:
                if time.monotonic() - heartbeat_at >= 15:
                    atomic(args.root / "heartbeat.json", {"run_id": config["run_id"], "updated_at_ns": time.time_ns()})
                    heartbeat_at = time.monotonic()
                receipt = operations.poll(abort_id)
                if receipt is not None:
                    if receipt.get("owned_actors_stopped") is not True or receipt.get("h200_allocation_released") is not True:
                        raise RuntimeError("scoped abort did not confirm resource release") from exc
                    campaign.save("ABANDONED", abort_receipt=receipt)
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("scoped cleanup requires reconciliation; no automatic restart") from exc
                time.sleep(1)
            raise


def make_first_round_config(run_id, *, seed, checkpoint, checkpoint_sha, panel, panel_sha, source_sha, gateway_pids, smoke=False, deployment_host="465d6b0-lcedt"):
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import FIDELITY, SCHEMA, SMOKE_SCHEMA, LEO_SCHEMA, LEO_SMOKE_SCHEMA, gpu_uuids
    if deployment_host not in {"465d6b0-lcedt", "leo-rtx-6k"}:
        raise ValueError("unknown deployment host")
    leo = deployment_host == "leo-rtx-6k"
    schema = (LEO_SMOKE_SCHEMA if smoke else LEO_SCHEMA) if leo else (SMOKE_SCHEMA if smoke else SCHEMA)
    windows, episodes, panel_protocol = budgets({"schema": schema})
    value = {
        "schema": schema, "run_id": run_id, "round_id": 1, "teacher_beta": .75,
        "student_action_probability": .25, "seed": seed, "source_sha256": source_sha,
        "rtx_actors": [{"physical_gpu_index": gpu, "gpu_uuid": uuid, "concurrent_environments": 6}
                       for gpu, uuid in gpu_uuids({"schema": schema}).items()], "forbidden_gpu_indices": [2] if leo else [0],
        "raven_fidelity": dict(FIDELITY), "replay": {"enabled": False},
        "execution": {"window_count": windows, "rollout_steps": 81, "wait_for_checkpoint_advance": True,
                      "learner_submission_authorized": True, "validation_authorized": True},
        "h200": {"render_student_worker_count": 4, "boundary_contract": "roborender.dagger.rtx.v3",
                 "resource_shape": "custom.gpu.4xh200", "reservation": "cosmos-solution-architect-xcwtffiq",
                 "allow_burst": False, "gateway_pids": list(gateway_pids)},
        "student": {"initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": checkpoint_sha,
                    "learning_rate": .0002, "max_gradient_norm": 1., "mini_batches_per_iteration": 1,
                    "activation_checkpointing": False},
        "validation": {"interval_iterations": windows, "completed_episodes": episodes,
                       "metric": "raven_goal_reached_rate", "observation_mode": "raytraced",
                       "panel_protocol": panel_protocol,
                       "scenario_manifest": str(panel), "scenario_manifest_sha256": panel_sha},
        "logging": {"wandb_project": "dagger", "wandb_run_name": run_id + "-r1-teacher75",
                    "wandb_run_id": "beta-" + hashlib.sha256(canonical([run_id, 1])).hexdigest()[:24]},
    }
    return validate_config(value)


if __name__ == "__main__":
    main()
