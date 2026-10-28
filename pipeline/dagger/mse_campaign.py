"""Separate MSE-selected campaign admission; legacy fifty-window checks stay strict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import validate_config, canonical, actor_ids
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_mse_smoke, is_mse_production, uses_explicit_motion, environments_per_actor
from pipeline.dagger.beta_validation import panel_seeds, panel_scenario_ids
from pipeline.dagger.services.mse_protocol import validate_protocol
from pipeline.dagger.services.heldout_mse import RoundMSESelection
from pipeline.deployment.native_phase_barrier import native_quiescent
from pipeline.dagger.beta_campaign import Campaign, immutable_json, next_round_config
from pipeline.dagger.services.contracts import file_sha256
from pipeline.dagger.services.mse_status import completed_mse_summary, next_round_display_fields
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_fixed100
from pipeline.dagger.services.fixed100_protocol import CAMPAIGN_SCHEMA as FIXED100_PLAN_SCHEMA, SCHEMA as FIXED100_PROTOCOL_SCHEMA

PLAN_SCHEMA = "roborender.dagger-mse-campaign-plan.v1"
NATIVE_PLAN_SCHEMA = "roborender.dagger-mse-campaign-plan.v2"
ROUND_FIELDS = {"round_id", "validation_manifest_path", "validation_manifest_sha256",
                "validation_admission_path", "validation_admission_sha256", "interval_iterations",
                "max_iterations", "min_validation_epochs", "patience", "relative_improvement", "absolute_improvement"}


def _validate_training_assignments(rows, config, panel):
    from pipeline.deployment.rtx_actor.native_motion_assignment import load_motion_assignment
    if (not isinstance(rows, list) or len(rows) != 2
            or [row.get("physical_gpu_index") for row in rows] != [0, 1]):
        raise ValueError("two explicit native training assignments are required")
    test_modes = {(lane["motion_id"], lane["sha256"]) for cohort in panel["cohorts"] for lane in cohort["assignment"]["lanes"]}
    result = []
    for row in rows:
        if set(row) != {"physical_gpu_index", "path", "sha256"}:
            raise ValueError("invalid native training assignment pin")
        assignment = load_motion_assignment(row["path"], row["sha256"])
        if (assignment["purpose"] != "training" or assignment["run_id"] != config["run_id"]
                or assignment["simulator_seed"] != config["seed"] + row["physical_gpu_index"]
                or assignment["simulator_seed"] in panel_seeds(panel)
                or len(assignment["lanes"]) != 2
                or assignment["motion_inventory_sha256"] != panel["motion_inventory_sha256"]
                or {(lane["motion_id"], lane["sha256"]) for lane in assignment["lanes"]} != test_modes):
            raise ValueError("training/source/test motion support or independent seed differs")
        result.append(assignment)
    return result


def validate_plan(plan, config):
    """Freeze scheduling paths; later round parent/data are not known yet."""
    from pipeline.dagger.beta_validation import load_panel
    if set(plan) != {"schema", "run_id", "rounds"} or plan["schema"] not in {PLAN_SCHEMA, NATIVE_PLAN_SCHEMA, FIXED100_PLAN_SCHEMA} or plan["run_id"] != config["run_id"]:
        raise ValueError("explicit matching MSE campaign plan required")
    if is_fixed100(config) != (plan['schema']==FIXED100_PLAN_SCHEMA):
        raise ValueError('fixed100 and legacy selection plans cannot be interchanged')
    if is_mse_smoke(config) and plan["schema"] != NATIVE_PLAN_SCHEMA:
        raise ValueError("matched B2 MSE smoke requires explicit native collection scheduling")
    if not isinstance(plan["rounds"], list) or [row.get("round_id") for row in plan["rounds"]] != [1, 2, 3, 4]:
        raise ValueError("all four fixed validation populations must be admitted before training")
    panel = load_panel(config)
    test = {"scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
            "scenario_ids": panel_scenario_ids(panel),
            "observation_mode": config["validation"]["observation_mode"]}
    if uses_explicit_motion(config):
        if test['observation_mode'] != 'generated' or panel['observation_mode'] != 'image3f_3view':
            raise ValueError('new student final tests require genuine Image3F three-view observations')
        from pipeline.deployment.rendered_success_dispatch import configuration
        configuration(config['validation'].get('rendered_observation_config'), panel,
                      config['student']['initial_checkpoint_sha256'])
    for row in plan["rounds"]:
        if is_fixed100(config):
            validate_protocol(dict(schema=FIXED100_PROTOCOL_SCHEMA,run_id=config['run_id'],
                initial_checkpoint_sha256=config['student']['initial_checkpoint_sha256'],**row,final_test=test))
            continue
        native = plan["schema"] == NATIVE_PLAN_SCHEMA
        extras = {"collection_plan_path", "collection_plan_sha256"} if native else set()
        if is_mse_smoke(config):
            extras.add("training_assignments")
        if set(row) != ROUND_FIELDS | extras:
            raise ValueError("invalid MSE round plan fields")
        protocol = {"schema": "roborender.dagger-mse-round-protocol.v1", "run_id": config["run_id"],
                    "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
                    **{key: value for key, value in row.items() if not key.startswith(("validation_admission", "collection_plan", "training_assignments"))},
                    "final_test": test}
        # A later population is generated only after its actual parent is
        # selected. Null pins mean defer admission, never accept unknown data.
        if (row["validation_manifest_sha256"] is None) != (row["validation_admission_sha256"] is None):
            raise ValueError("deferred population requires both hashes to be unset")
        if protocol["validation_manifest_sha256"] is None:
            protocol["validation_manifest_sha256"] = "0" * 64
        validate_protocol(protocol)
        if row["max_iterations"] > config["execution"]["window_count"]:
            raise ValueError("MSE budget exceeds the explicit native maximum")
        if not Path(row["validation_admission_path"]).is_absolute():
            raise ValueError("held-out admission path must be absolute")
        if native:
            from pipeline.dagger.heldout_collection_plan import load_plan
            upcoming = {**config, "round_id": row["round_id"], "seed": config["seed"] + 100003 * (row["round_id"] - 1)}
            load_plan(row["collection_plan_path"], row["collection_plan_sha256"], upcoming,
                      panel_seeds(panel))
        if is_mse_smoke(config):
            _validate_training_assignments(row["training_assignments"], upcoming, panel)
            if row["round_id"] == 1 and row["training_assignments"] != config["motion_assignments"]:
                raise ValueError("first native training assignments differ from frozen run config")
    return plan


def validate_explicit_final_test(config, comparison, selected_sha):
    """Route by the declared campaign schema, never by a receipt denominator."""
    from pipeline.dagger.beta_validation import load_panel
    from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_result
    panel = load_panel(config)
    from pipeline.deployment.rendered_success_dispatch import configuration
    rendered_config = configuration(config['validation'].get('rendered_observation_config'), panel, selected_sha)
    from pipeline.deployment.rtx_actor.scaled_success_contract import is_panel
    if is_panel(panel):
        from pipeline.deployment.paired_scaled_evaluation import validate_final_evidence
        if config['schema']!='roborender.cross_gpu_dagger_continuous_fixed100.leo.v2':
            raise ValueError('scaled final-test import requires continuous eight-H200 production')
        return validate_final_evidence(comparison,panel=panel,checkpoint_sha256=selected_sha,
            source_sha256=config['source_sha256'],arm='dagger',expected_rendered_config=rendered_config)
    if is_mse_production(config):
        from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options, validate_production_result
        evaluation_options(panel, raw_sha256=config['validation']['scenario_manifest_sha256'])
        return validate_production_result(panel, comparison, selected_sha, allowed_arms=('dagger',), rendered_config=rendered_config)
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_corrected_smoke
    if is_corrected_smoke(config):
        from pipeline.deployment.rtx_actor.smoke_success_contract import validate_smoke_result
        return validate_smoke_result(panel, comparison, selected_sha, allowed_arms=('dagger',), rendered_config=rendered_config)
    return validate_result(panel, comparison, selected_sha)


def production_test_protocol(config):
    from pipeline.dagger.beta_validation import load_panel
    from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options, protocol_identity
    panel = load_panel(config)
    from pipeline.deployment.rtx_actor import scaled_success_contract as scaled
    if scaled.is_panel(panel):
        from pipeline.deployment.paired_scaled_evaluation import final_panel
        final_panel(panel)
        return scaled.protocol_identity(panel)
    evaluation_options(panel, raw_sha256=config['validation']['scenario_manifest_sha256'])
    return protocol_identity(panel)


def validate_production_test_report(config, report):
    protocol = production_test_protocol(config)
    from pipeline.dagger.beta_validation import load_panel
    from pipeline.deployment.rendered_success_dispatch import configuration
    from pipeline.deployment.rtx_actor.rendered_student_success_contract import protocol_identity as rendered_protocol
    panel=load_panel(config)
    rendering=configuration(config['validation'].get('rendered_observation_config'),panel,report.get('checkpoint_sha256'))
    if (report.get('success_test_protocol') != protocol or report.get('purpose') != 'test'
            or report.get('status') != 'complete' or report.get('completed') != 295
            or report.get('rendered_student_success_protocol') != rendered_protocol(panel,rendering)
            or report.get('scenario_manifest_sha256') != config['validation']['scenario_manifest_sha256']):
        raise ValueError('production success report lacks corrected accepted295 student protocol')
    return protocol


def import_final_test(controller, comparison, baseline_sha, selected_sha):
    """Small outcome receipts only; test results never reach the MSE stopper."""
    from pipeline.dagger.beta_validation import outcome_payloads
    eid = f"{controller.run_id}-r{controller.round_id}-final-test"
    # Reject stale original-reset results before registering or importing any
    # ledger outcomes. Production routing comes from the frozen config schema.
    checked = (validate_explicit_final_test(controller.config, comparison, selected_sha)
               if uses_explicit_motion(controller.config) else None)
    controller.mse.final_test()  # Registers the exact immutable test, not a native replay.
    report = controller.evaluation.get(f"/v2/evaluations/{eid}")["evaluation"]
    if (report["purpose"] != "test" or report["checkpoint_sha256"] != selected_sha
            or report["scenario_manifest_sha256"] != controller.config["validation"]["scenario_manifest_sha256"]):
        raise ValueError("final test ledger does not bind the selected checkpoint/panel")
    if uses_explicit_motion(controller.config):
        import math
        payloads = []
        for row in checked["outcomes"]:
            reward = row.get("episode_reward")
            if type(reward) not in (int, float) or not math.isfinite(reward) or type(row.get("timeout")) is not bool:
                raise ValueError("explicit native success outcome lacks finite actual reward/timeout")
            success = row["exported_success"]
            payloads.append({"idempotency_key": "replaced-below", "episode_id": "replaced-below",
                "scenario_id": row["case_id"], "goal_reached": success, "reward": float(reward),
                "steps": row["steps"], "termination": "timeout" if row["timeout"] else "success" if success else "failure",
                "safety_violation": False, "checkpoint_sha256": selected_sha,
                "scenario_manifest_sha256": controller.config["validation"]["scenario_manifest_sha256"]})
        expected_successes, expected_rate = checked["successes"], checked["success_rate"]
    else:
        payloads = outcome_payloads(controller.config, comparison, baseline_sha, selected_sha)
        expected_successes = comparison["arms"]["dagger"]["successes"]
        expected_rate = comparison["arms"]["dagger"]["goal_reached_rate"]
    for payload in payloads:
        payload["idempotency_key"] = hashlib.sha256(canonical([eid, payload["scenario_id"]])).hexdigest()
        payload["episode_id"] = eid + "-" + payload["scenario_id"]
        ack = controller.evaluation.post(f"/v2/evaluations/{eid}/outcomes", payload)
        if not ack.get("ok") or ack.get("evaluation_id") != eid or ack.get("scenario_id") != payload["scenario_id"]:
            raise RuntimeError("final test acknowledgement changed")
    report = controller.evaluation.get(f"/v2/evaluations/{eid}")["evaluation"]
    if (report["status"] != "complete" or report["completed"] != controller.config["validation"]["completed_episodes"]
            or report["successes"] != expected_successes or report["goal_reached_rate"] != expected_rate):
        raise RuntimeError("final native test and ledger disagree")
    if is_mse_production(controller.config):
        report = {**report, 'success_test_protocol': production_test_protocol(controller.config),
                  'rendered_student_success_protocol': checked['rendered_student_success_protocol']}
    return report


class MSECampaign(Campaign):
    """Explicit separate state machine; inherited legacy phases are unchanged."""
    supports_explicit_mse_smoke = True
    supports_full295_production = False

    def validate_production_history(self):
        """Historical completed rounds cannot acquire a corrected label on resume."""
        config = self.state['config']
        if not is_mse_production(config):
            return
        failure = ('production completed-round history lacks the corrected accepted295 protocol; '
                   'start a separately reviewed production campaign without relabeling historical rounds')
        completed = self.state.get('completed_rounds')
        expected = config['round_id'] if self.state['phase'] == 'COMPLETE' else config['round_id'] - 1
        if (not isinstance(completed, list) or len(completed) != expected
                or self.state.get('round_id') != config['round_id']):
            raise ValueError(failure)
        for index, row in enumerate(completed, 1):
            if not isinstance(row, dict) or row.get('round_id') != index:
                raise ValueError(failure)
            report, checkpoint = row.get('test'), row.get('selected_checkpoint')
            if not isinstance(report, dict) or not isinstance(checkpoint, dict):
                raise ValueError(failure)
            try:
                validate_production_test_report(config, report)
            except ValueError as error:
                raise ValueError(failure) from error
            sha = checkpoint.get('sha256')
            if (not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                    or report.get('checkpoint_sha256') != sha or row.get('selected_checkpoint_sha256') != sha):
                raise ValueError(failure)

    def validate_production_test_evidence(self):
        """Recheck persisted evidence when resuming logging or round advance."""
        config = self.state['config']
        if not is_mse_production(config):
            return
        evidence = self.state['evaluation_receipt']
        path = Path(evidence['comparison_path'])
        if file_sha256(path) != evidence['comparison_sha256']:
            raise ValueError('production final test evidence changed')
        checked = validate_explicit_final_test(config, json.loads(path.read_bytes()),
                                               self.state['final_checkpoint']['sha256'])
        report = self.state['evaluation_report']
        validate_production_test_report(config, report)
        if (report.get('checkpoint_sha256') != checked['checkpoint_sha256']
                or report.get('successes') != checked['successes']
                or report.get('goal_reached_rate') != checked['success_rate']):
            raise ValueError('production final test report differs from native evidence')

    def __init__(self, first_config, root, operations, *, plan, **kwargs):
        if is_mse_production(first_config) and not self.supports_full295_production:
            raise ValueError("full295 production requires its explicit coverage lifecycle controller")
        validate_plan(plan, first_config)
        marker = Path(root).resolve() / "mse-plan.json"
        if (Path(root) / "state.json").exists() and not marker.exists():
            raise ValueError("cannot silently adopt a legacy campaign into MSE selection")
        self.plan = plan
        self.plan_sha = immutable_json(marker, plan)
        super().__init__(first_config, root, operations, **kwargs)
        if self.state.get("mse_plan_sha256", self.plan_sha) != self.plan_sha:
            raise ValueError("immutable MSE campaign plan changed")
        self.validate_production_history()
        self.save(mse_plan_sha256=self.plan_sha)

    def protocol_path(self):
        from pipeline.dagger.beta_validation import load_panel
        config = self.state["config"]
        row = {**self.plan["rounds"][config["round_id"] - 1],
               **({} if is_fixed100(config) else self.admit_population())}
        panel = load_panel(config)
        protocol = {"schema": FIXED100_PROTOCOL_SCHEMA if is_fixed100(config) else "roborender.dagger-mse-round-protocol.v1", "run_id": config["run_id"],
                    "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
                    **{key: value for key, value in row.items() if not key.startswith(("validation_admission", "collection_plan", "training_assignments"))},
                    "final_test": {"scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
                        "scenario_ids": panel_scenario_ids(panel),
                        "observation_mode": config["validation"]["observation_mode"]}}
        validate_protocol(protocol)
        path = self.root / f"round-{config['round_id']}" / "mse-protocol.json"
        immutable_json(path, protocol)
        return path

    def admit_population(self):
        from pipeline.dagger.beta_validation import load_panel
        from pipeline.dagger.services.mse_protocol import empty_provenance
        from pipeline.dagger.services.heldout_mse import load_validation_manifest
        from pipeline.dagger.export_heldout_manifest import validate_export_admission
        config = self.state["config"]
        row = self.plan["rounds"][config["round_id"] - 1]
        manifest, admission = Path(row["validation_manifest_path"]), Path(row["validation_admission_path"])
        if not manifest.is_file() or not admission.is_file():
            raise FileNotFoundError("actual parent-policy held-out population is not yet sealed/admitted")
        manifest_sha, admission_sha = file_sha256(manifest), file_sha256(admission)
        if row["validation_manifest_sha256"] not in (None, manifest_sha) or row["validation_admission_sha256"] not in (None, admission_sha):
            raise ValueError("pre-pinned held-out population changed")
        collection = validate_export_admission(admission, admission_sha, manifest, manifest_sha)
        if is_mse_production(config):
            from pipeline.dagger.coverage_production import round_training_seeds
            expected_seeds = set(round_training_seeds(config))
        else:
            expected_seeds = {config["seed"] + index for index, _ in enumerate(config["rtx_actors"])}
        test_seeds = set(panel_seeds(load_panel(config)))
        if (collection["collection_checkpoint_sha256"] != config["student"]["initial_checkpoint_sha256"]
                or collection["round_id"] != config["round_id"] or collection["source_run_id"] == config["run_id"]
                or set(collection["training_simulator_seeds"]) != expected_seeds
                or set(collection["test_simulator_seeds"]) != test_seeds):
            raise ValueError("held-out parent/run/seed population differs from upcoming round")
        if is_mse_smoke(config):
            training = _validate_training_assignments(config["motion_assignments"], config, load_panel(config))
            if set(collection["motion_support"]) != {lane["motion_id"] for row in training for lane in row["lanes"]}:
                raise ValueError("held-out motion support differs from explicit training population")
        load_validation_manifest(manifest, manifest_sha, empty_provenance(config["round_id"]))
        binding = {"validation_manifest_sha256": manifest_sha, "validation_admission_sha256": admission_sha}
        immutable_json(self.root / f"round-{config['round_id']}" / "heldout-admission-binding.json", binding)
        return binding

    def operation_context(self):
        value = super().operation_context()
        if self.plan["schema"] == NATIVE_PLAN_SCHEMA and self.state["phase"] in {"PREPARE", "COLLECT_HELDOUT"}:
            value["mse_campaign"] = True
            if self.state["phase"] == "COLLECT_HELDOUT":
                row = self.plan["rounds"][self.state["round_id"] - 1]
                value["heldout_collection"] = {"plan_path": row["collection_plan_path"],
                    "plan_sha256": row["collection_plan_sha256"], "manifest_path": row["validation_manifest_path"],
                    "admission_path": row["validation_admission_path"]}
            return value
        path = self.protocol_path()
        return {**value, "mse_campaign": True, "mse_protocol_path": str(path),
                "mse_protocol_file_sha256": file_sha256(path)}

    def get_controller(self):
        if self.controller is None:
            binding = self.state["service_binding"]
            fixed=is_fixed100(self.state['config'])
            command = [] if fixed else binding.get("mse_worker_command") or self.state.get("mse_worker_command")
            if not command and not fixed:
                raise RuntimeError("MSE campaign requires an explicitly owned H200 validation worker")
            self.save(mse_worker_command=command)
            self.controller = self.factory(Path(self.state["config_path"]), Path(binding["token_file"]),
                wandb_mode="online", mse_protocol_path=self.protocol_path(), mse_worker_command=command)
            self.controller.ensure_registered()
        return self.controller

    def tick(self):
        phase, config = self.state["phase"], self.state["config"]
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_corrected_smoke
        if is_corrected_smoke(config) and phase == 'RELEASE_TRAINING':
            receipt=self.operation('release')
            if receipt is None:return dict(self.state)
            if (receipt.get('h200_allocation_retained') is not True
                    or receipt.get('h200_allocation_released') is not False
                    or receipt.get('same_service_ledger_owner') is not True
                    or not native_quiescent(receipt,source_sha256=config['source_sha256'])):
                raise RuntimeError('corrected smoke must retain compute and authoritative ledger')
            self.save('EVALUATE',operation_id=None,retention_receipt=receipt)
            return dict(self.state)
        if phase == 'COMPLETE':
            self.validate_production_history()
        if phase == "PREPARE":
            if is_fixed100(config):
                return Campaign.tick(self)
            if self.plan["schema"] == NATIVE_PLAN_SCHEMA:
                # Allocation precedes the genuine online held-out collection;
                # main training is not registered until data are sealed/admitted.
                super().tick()
                if self.state["phase"] == "REGISTER":
                    row = self.plan["rounds"][config["round_id"] - 1]
                    if row["validation_manifest_sha256"] is not None:
                        # Explicit pre-pinned recovery may reuse a completed,
                        # separately owned population. Preserve its source IDs;
                        # admit_population verifies real parent/seed/motion and
                        # immutable ledger evidence, without replaying Isaac.
                        self.admit_population()
                        self.save("REGISTER", operation_id=None, heldout_population_reused=True)
                    else:
                        self.save("COLLECT_HELDOUT", operation_id=None)
                return dict(self.state)
            try:
                self.admit_population()
            except FileNotFoundError:
                # This explicit CPU-only handoff is not a fake population or
                # permission to replay a training window. A collect-only native
                # executor must supply actual admitted artifacts before advance.
                request = {"schema": "roborender.heldout-population-request.v1", "purpose": "validation",
                    "run_id": config["run_id"], "round_id": config["round_id"],
                    "parent_checkpoint": config["student"], "teacher_beta": config["teacher_beta"],
                    "population": self.plan["rounds"][config["round_id"] - 1], "optimizer_authorized": False}
                immutable_json(self.root / f"round-{config['round_id']}" / "heldout-population-request.json", request)
                self.save(progress={"action": "awaiting_real_heldout_population", "allocation_performed": False})
                return dict(self.state)
            return super().tick()
        if phase == "COLLECT_HELDOUT":
            receipt = self.operation("collect")
            if receipt is None:
                return dict(self.state)
            if (receipt.get("collection_complete") is not True or receipt.get("purpose") != "validation"
                    or receipt.get("optimizer_steps") != 0 or receipt.get("array_payloads_read") is not False
                    or not native_quiescent(receipt, source_sha256=config["source_sha256"])):
                raise RuntimeError("held-out host operation lacks complete native-only collection evidence")
            self.admit_population()
            self.save("REGISTER", operation_id=None, heldout_collection_receipt=receipt)
            return dict(self.state)
        if phase == "COLLECT_TRAIN":
            controller = self.get_controller()
            receipt = self.operation("collect")
            progress = controller.tick()
            if progress["action"] == "collection_budget_complete":
                raise RuntimeError("MSE campaign crossed its explicit maximum without selecting")
            if progress != self.state.get("progress"):
                self.save(progress=progress)
            if self.clock() > self.state["collection_deadline"]:
                raise TimeoutError("MSE round deadline; no native lifecycle restart")
            if receipt is None:
                return dict(self.state)
            status = controller.coordinator.get(f"/v1/status/{config['run_id']}")
            context = controller.mse.context()
            selection = validate_mse_actor_completion(config, receipt, status["iterations"], context,
                                                       file_sha256(self.protocol_path()))
            count = selection["completed_iterations"]
            job = controller.learner.get(f"/v1/jobs/{config['run_id']}-dagger-{config['round_id']}-iteration-{count:04d}")["job"]
            result = job.get("result")
            if (job["status"] != "succeeded" or not result or result["iteration_id"] != count
                    or result["output_checkpoint_sha256"] != selection["last_candidate_checkpoint_sha256"]
                    or result["train_chunks"] != len(actor_ids(config)) or result["train_frames"] != len(actor_ids(config)) * 81
                    or result.get("replay_manifest_sha256") is not None):
                raise RuntimeError("MSE final learner receipt differs from complete actor chain")
            path = Path(context["selected_checkpoint_path"])
            if file_sha256(path) != selection["selected_checkpoint_sha256"]:
                raise RuntimeError("MSE-selected checkpoint bytes are unavailable")
            controller.mse.final_test()
            self.save("RELEASE_TRAINING", operation_id=None, final_checkpoint={"path": str(path), "sha256": selection["selected_checkpoint_sha256"]},
                      final_learner=result, mse_completion=selection)
        elif phase == "EVALUATE" and is_mse_smoke(config):
            receipt = self.operation("evaluate")
            if receipt is None:
                return dict(self.state)
            path = Path(receipt["comparison_path"])
            if file_sha256(path) != receipt["comparison_sha256"] or not native_quiescent(receipt, source_sha256=config["source_sha256"]):
                raise RuntimeError("explicit final-test receipt or native phase barrier differs")
            validate_explicit_final_test(config, json.loads(path.read_bytes()), self.state["final_checkpoint"]["sha256"])
            self.save("IMPORT_VALIDATION" if is_corrected_smoke(config) else "CONTROL_ONLY",
                      operation_id=None, evaluation_receipt=receipt)
        elif phase == "IMPORT_VALIDATION":
            controller = self.get_controller()
            evidence = self.state["evaluation_receipt"]
            if file_sha256(Path(evidence["comparison_path"])) != evidence["comparison_sha256"]:
                raise RuntimeError("final test evidence changed")
            report = import_final_test(controller, json.loads(Path(evidence["comparison_path"]).read_bytes()),
                                      self.first["student"]["initial_checkpoint_sha256"], self.state["final_checkpoint"]["sha256"])
            self.save("LOG_VALIDATION", evaluation_report=report)
        elif phase == "LOG_VALIDATION":
            self.validate_production_test_evidence()
            if self.log_validation is None:
                raise RuntimeError("online final-test metric logger is required")
            logging = self.log_validation(config, self.state["final_learner"], self.state["evaluation_report"])
            if logging.get("online_metrics_verified") is not True:
                raise RuntimeError("final test logging readback failed")
            self.get_controller().mse.final_test()
            self.save("ADVANCE", logging_receipt=logging)
        elif phase == "ADVANCE":
            self.validate_production_test_evidence()
            context = self.get_controller().mse.context()
            if (context["status"] != "complete" or context["selection"]["best_checkpoint_sha256"] != self.state["final_checkpoint"]["sha256"]):
                raise RuntimeError("independent final test has not completed the MSE-selected round")
            completed = self.state["completed_rounds"] + [{"round_id": config["round_id"], "teacher_beta": config["teacher_beta"],
                "selected_checkpoint": self.state["final_checkpoint"], **self.state["mse_completion"],
                "test": self.state["evaluation_report"], "wandb": self.state["logging_receipt"]}]
            if config["round_id"] == 4:
                self.save("COMPLETE", completed_rounds=completed,
                    progress={"action": "campaign_complete", "status": "complete",
                        "completed_rounds": len(completed),
                        **completed_mse_summary(self.state["evaluation_report"], fixed100=is_fixed100(config))})
            else:
                # Reuse only the config layout/beta schedule helper: this
                # explicit protocol continues the MSE-selected optimizer state.
                next_config = next_round_config(config, self.state["first_config_sha256"],
                    self.state["final_checkpoint"]["sha256"], self.state["final_checkpoint"]["path"], config["h200"]["gateway_pids"])
                if is_mse_smoke(next_config):
                    next_config["motion_assignments"] = self.plan["rounds"][next_config["round_id"] - 1]["training_assignments"]
                    validate_config(next_config)
                self.controller = None
                self.save("PREPARE", config=next_config, round_id=next_config["round_id"], operation_id=None,
                          completed_rounds=completed, **next_round_display_fields())
        else:
            return super().tick()
        return dict(self.state)


def validate_mse_actor_completion(config, receipt, iterations, mse_context, protocol_file_sha256):
    """Return selected and newest checkpoint separately after all actors quiesce."""
    validate_config(config, runtime=True)
    protocol = validate_protocol(mse_context["protocol"])
    protocol_sha = hashlib.sha256(canonical(protocol)).hexdigest()
    selected = RoundMSESelection.restore(mse_context["selection"])
    rows = sorted((row for row in iterations if row["round_id"] == config["round_id"]),
                  key=lambda row: row["iteration_id"])
    count = len(rows)
    if (mse_context["protocol_sha256"] != protocol_sha
            or protocol["run_id"] != config["run_id"] or protocol["round_id"] != config["round_id"]
            or protocol["initial_checkpoint_sha256"] != config["student"]["initial_checkpoint_sha256"]
            or protocol["max_iterations"] > config["execution"]["window_count"]
            or not 1 <= count <= protocol["max_iterations"]
            or mse_context["status"] != "selected_pending_test"
            or mse_context["iteration_id"] != count
            or [row["iteration_id"] for row in rows] != list(range(1, count + 1))
            or any(row["status"] != "advanced" for row in rows[:-1])
            or rows[-1]["status"] != "selected_pending_test"
            or selected.history[-1]["iteration"] != count):
        raise RuntimeError("MSE campaign completion does not match the selected full-window barrier")
    expected_stop = "validation_mse_plateau" if selected.history[-1]["saturated"] else "maximum_iterations"
    if (mse_context["stop_reason"] != expected_stop
            or (expected_stop == "maximum_iterations" and count != protocol["max_iterations"])):
        raise RuntimeError("MSE round stopping reason is not justified by selection history/budget")
    parent = config["student"]["initial_checkpoint_sha256"]
    known = {parent}
    for row in rows:
        if row["input_checkpoint_sha256"] != parent or not row["candidate_checkpoint_sha256"]:
            raise RuntimeError("MSE campaign checkpoint/optimizer lineage is broken")
        parent = row["candidate_checkpoint_sha256"]
        known.add(parent)
    if selected.best_checkpoint_sha256 not in known or mse_context["checkpoint_sha256"] != parent:
        raise RuntimeError("MSE selection checkpoint is outside the observed learner chain")
    actors = receipt.get("actors", [])
    expected_gpus = {actor["physical_gpu_index"] for actor in config["rtx_actors"]}
    if len(actors) != len(expected_gpus) or {actor.get("physical_gpu") for actor in actors} != expected_gpus:
        raise RuntimeError("all exact native actors must finish before final test")
    config_sha = hashlib.sha256(canonical(config)).hexdigest()
    for actor in actors:
        if (actor.get("completion_schema") != "roborender.mse-actor-completion.v1"
                or actor.get("run_id") != config["run_id"] or actor.get("round_id") != config["round_id"]
                or actor.get("config_sha256") != config_sha or actor.get("mse_protocol_sha256") != protocol_sha
                or actor.get("mse_protocol_file_sha256") != protocol_file_sha256
                or actor.get("completed_windows") != count or actor.get("declared_max_iterations") != protocol["max_iterations"]
                or actor.get("phase") != "COMPLETE" or actor.get("uncertain_reason") is not None
                or actor.get("lifecycle_initializations") != 1
                or actor.get("transition_batches") != count * 81
                or actor.get("physical_transition_rows") != count * 81 * environments_per_actor(config)
                or actor.get("final_candidate_checkpoint_sha256") != parent
                or actor.get("selected_checkpoint_sha256") != selected.best_checkpoint_sha256
                or actor.get("stop_reason") != expected_stop
                or [(item.get("index"), item.get("input_checkpoint_sha256"), item.get("output_checkpoint_sha256"))
                    for item in actor.get("windows", [])] != [(row["iteration_id"] - 1, row["input_checkpoint_sha256"],
                                                             row["candidate_checkpoint_sha256"]) for row in rows]):
            raise RuntimeError("MSE native completion identity/count/checkpoint receipt differs")
    if set(receipt.get("actor_ids", [])) != set(actor_ids(config)) or not native_quiescent(receipt, source_sha256=config["source_sha256"]):
        raise RuntimeError("native actor shutdown/identity receipt is missing")
    return {"completed_iterations": count, "selected_checkpoint_sha256": selected.best_checkpoint_sha256,
            "last_candidate_checkpoint_sha256": parent, "stop_reason": expected_stop}


def online_mse_loggers(entity, key_file=None):
    from pipeline.dagger.beta_campaign import online_loggers
    from pipeline.deployment.wandb_preflight import admit_logging, verify_remote_summary
    begin, _ = online_loggers(entity, key_file)
    api, identity = admit_logging("online", entity, key_file)
    def log(config, learner, report):
        import wandb
        if report.get("purpose") != "test":
            raise ValueError("success is a final test, not the selection metric")
        protocol = validate_production_test_report(config, report) if is_mse_production(config) else None
        run_id = config["logging"]["wandb_run_id"]
        iteration = learner["iteration_id"]
        verify_remote_summary(api, entity, run_id, {"last_iteration_id": iteration, "last_train_mse": learner["train_mse"]})
        logging_config = {'config': {'success_test_protocol': protocol}} if protocol is not None else {}
        run = wandb.init(project="dagger", entity=entity, id=run_id, resume="must", mode="online", **logging_config)
        metrics = {"iteration_id": iteration, "test/goal_reached_rate": report["goal_reached_rate"],
                   "test/successes": report["successes"], "test/completed_episodes": report["completed"],
                   "test/observation_mode": config['validation']['observation_mode'],
                   "orchestration/round_complete": True}
        if protocol is not None:
            metrics['test/success_test_protocol'] = protocol
            metrics['test/rendered_student_success_protocol'] = report['rendered_student_success_protocol']
        summary = {**metrics, **completed_mse_summary(report, fixed100=is_fixed100(config))}
        try:
            run.log(metrics)
            run.summary.update(summary)
            url = run.url
        finally:
            run.finish()
        verify_remote_summary(api, entity, run_id, summary)
        return {"online_metrics_verified": True, "run_id": run_id, "url": url, "entity": identity["entity"]}
    return begin, log


def main():
    import argparse
    import fcntl
    import os
    import signal
    import threading
    from pipeline.deployment.beta_operations import CommandOperations, atomic
    parser = argparse.ArgumentParser(description="Opt-in resident MSE-selected DAgger campaign; no implicit allocation")
    for name in ("first-round-config", "mse-plan", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--operations", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument('--allow-legacy-validation', action='store_true',
        help='Explicit historical MSE-validation execution; new runs use fixed100')
    parser.add_argument("--wandb-entity", default="nvidia")
    parser.add_argument("--wandb-key-file", type=Path)
    args = parser.parse_args()
    config = validate_config(json.loads(args.first_round_config.read_bytes()))
    plan = validate_plan(json.loads(args.mse_plan.read_bytes()), config)
    if args.execute:
        from pipeline.deployment.training_launch_policy import require_training_launch
        require_training_launch(validation_enabled=not is_fixed100(config),
            allow_legacy_validation=args.allow_legacy_validation)
    if not args.execute:
        print(json.dumps({"run_id": config["run_id"], "protocol": plan["schema"], "rounds": plan["rounds"],
            "allocation_performed": False, "launch_performed": False,
            "requires_real_parent_policy_heldout_population_before_each_round": True,
            "explicit_matched_motion_smoke": is_mse_smoke(config),
            "full_295_motion_population_certified": False}, indent=2))
        return
    if args.operations is None:
        parser.error("--execute requires SHA-pinned resident host operations")
    binding = json.loads(args.operations.read_bytes())
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import is_corrected_smoke
    retained=is_corrected_smoke(config) or is_mse_production(config)
    entry='retained_campaign_site.py' if retained else 'resident_campaign_site.py'
    if any(Path(row["entrypoint"]).name != entry for row in binding["commands"].values()):
        raise ValueError("MSE campaign requires explicit resident operations supporting its sidecar")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable_json(args.root / "operations.json", binding)
        from pipeline.deployment.coverage_operations import CoverageCommandOperations
        ops = (CoverageCommandOperations if retained else CommandOperations)(binding, args.root / "operations")
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        begin, log = online_mse_loggers(args.wandb_entity, args.wandb_key_file)
        campaign_type=MSECampaign
        if is_mse_production(config):
            from pipeline.dagger.coverage_campaign import CoverageMSECampaign
            campaign_type=CoverageMSECampaign
        campaign = campaign_type(config, args.root, ops, plan=plan, begin_logging=begin, log_validation=log)
        stop = threading.Event()
        def heartbeat():
            while not stop.is_set():
                atomic(args.root / "heartbeat.json", {"run_id": config["run_id"], "updated_at_ns": time.time_ns()})
                stop.wait(15)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        def interrupted(signum, frame):
            raise KeyboardInterrupt("MSE campaign stop requested")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        try:
            while campaign.state["phase"] != "COMPLETE":
                campaign.tick()
                time.sleep(1)
        except BaseException as exc:
            atomic(args.root / "failure.json", {"phase": campaign.state["phase"], "error_type": type(exc).__name__,
                                               "native_restart_allowed": False})
            if retained:
                # Preserve authoritative state for reconciliation. In-flight
                # physics is never replayed and a failure never releases GPUs.
                raise
            # No resources exist while waiting for population admission. If a
            # deployment began, reuse only its scoped ownership cleanup.
            if campaign.state.get("operation_id") or campaign.state["phase"] not in {"PREPARE", "COMPLETE"}:
                abort_id = ops.start("abort", {**Campaign.operation_context(campaign), "mse_campaign": True,
                                               "native_restart_allowed": False})
                deadline = time.monotonic() + binding["commands"]["abort"]["timeout_seconds"] + 5
                while time.monotonic() < deadline:
                    receipt = ops.poll(abort_id)
                    if receipt is not None:
                        if not receipt.get("owned_actors_stopped") or not receipt.get("h200_allocation_released"):
                            raise RuntimeError("MSE scoped abort requires resource reconciliation") from exc
                        campaign.save("ABANDONED", abort_receipt=receipt)
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("MSE scoped abort timed out; no automatic native replay") from exc
            raise
        finally:
            stop.set()
            thread.join(timeout=20)


if __name__ == "__main__":
    main()
