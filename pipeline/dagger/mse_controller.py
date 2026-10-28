"""Explicit controller adapter for held-out MSE selection and final-only test.

No GPU job is launched by importing or constructing this adapter. Callers must
provide a concrete validation worker command on allocated inference compute.
The native actor must understand selected_pending_test/round_complete barriers;
legacy fixed-window actors are not automatically upgraded by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from pipeline.dagger.services.mse_protocol import validate_protocol


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class MSEControllerAdapter:
    def __init__(self, controller, protocol_path: Path, worker_command: list[str], *, output_root: Path):
        self.controller, self.worker_command = controller, list(worker_command)
        self.protocol = validate_protocol(json.loads(protocol_path.read_bytes()))
        from pipeline.dagger.services.fixed100_protocol import is_fixed100
        self.fixed100 = is_fixed100(self.protocol)
        if self.fixed100:
            if worker_command:
                raise ValueError('fixed100 forbids a held-out validation worker')
        elif not worker_command or not all(isinstance(item, str) and item for item in worker_command):
            raise ValueError("MSE validation requires an explicit allocated-compute worker command")
        config = controller.config
        if (self.protocol["run_id"] != controller.run_id
                or self.protocol["round_id"] != controller.round_id
                or self.protocol["initial_checkpoint_sha256"] != config["student"]["initial_checkpoint_sha256"]
                or self.protocol["max_iterations"] > config["execution"]["window_count"]
                or self.protocol["final_test"]["scenario_manifest_sha256"] != config["validation"]["scenario_manifest_sha256"]
                or self.protocol["final_test"]["observation_mode"] != config["validation"]["observation_mode"]):
            raise ValueError("MSE sidecar differs from run/parent/budget/final-test contract")
        self.output = output_root / controller.run_id / f"round-{controller.round_id}"
        self.protocol_sha256 = digest(self.protocol)

    def context(self):
        return self.controller.coordinator.get(
            f"/v1/mse-rounds/{self.controller.run_id}/{self.controller.round_id}"
        )["mse_round"]

    def ensure_registered(self):
        self.controller.coordinator.post("/v1/mse-rounds/register", {
            "protocol": self.protocol, "protocol_sha256": self.protocol_sha256,
        })
        if self.context()["status"] == "awaiting_baseline":
            self.evaluate_current()

    def evaluate_current(self):
        if self.fixed100:
            raise ValueError('fixed100 has no validation stage')
        context = self.context()
        if context["protocol_sha256"] != self.protocol_sha256:
            raise ValueError("registered MSE protocol changed")
        iteration = context["iteration_id"]
        request = {
            "schema": "roborender.dagger-heldout-worker.v1",
            "checkpoint_path": context["checkpoint_path"],
            "checkpoint_sha256": context["checkpoint_sha256"],
            "manifest_path": self.protocol["validation_manifest_path"],
            "manifest_sha256": self.protocol["validation_manifest_sha256"],
            "training_provenance": context["training_provenance"],
            "iteration_id": iteration,
            "logging": {**self.controller.config["logging"], "wandb_mode": self.controller.wandb_mode,
                        "wandb_entity": os.environ.get("WANDB_ENTITY"),
                        "wandb_run_id": self.controller.config["logging"].get("wandb_run_id", self.controller.run_id)},
        }
        output = self.output / f"iteration-{iteration:04d}"
        request_path, result_path = output / "request.json", output / "result.json"
        if request_path.exists() and json.loads(request_path.read_bytes()) != request:
            raise ValueError("immutable MSE worker request changed during recovery")
        if not request_path.exists():
            atomic_json(request_path, request)
        if not result_path.exists():
            subprocess.run([*self.worker_command, "--request", str(request_path),
                            "--result", str(result_path)], check=True)
        receipt = json.loads(result_path.read_bytes())
        if set(receipt) != {"request_sha256", "result"} or receipt["request_sha256"] != digest(request):
            raise ValueError("MSE worker receipt does not bind exact request/provenance")
        return self.controller.coordinator.post("/v1/mse-rounds/record", {
            "run_id": self.controller.run_id, "round_id": self.controller.round_id,
            "iteration_id": iteration, "result": receipt["result"],
        })

    def final_test(self):
        context = self.context()
        if context["status"] == "complete":
            return "mse_round_complete"
        if context["status"] != "selected_pending_test":
            raise ValueError("final success test requested before MSE selection")
        evaluation_id = f"{self.controller.run_id}-r{self.controller.round_id}-final-test"
        request = {"evaluation_id": evaluation_id, "run_id": self.controller.run_id,
                   "round_id": self.controller.round_id, "purpose": "test",
                   "checkpoint_sha256": context["selection"]["best_checkpoint_sha256"],
                   "metric": "raven_goal_reached_rate", **self.protocol["final_test"]}
        self.controller.evaluation.post("/v2/evaluations", request)
        # This file is the explicit handoff to the native success-test executor;
        # registration is not execution or completion of that physical test.
        atomic_json(self.output / "final-test-request.json", request)
        report = self.controller.evaluation.get(f"/v2/evaluations/{evaluation_id}")["evaluation"]
        if report["status"] != "complete":
            return "mse_selected_waiting_final_test"
        logging_path = self.output / "final-test-logging.json"
        if logging_path.exists():
            if json.loads(logging_path.read_bytes()).get("report_sha256") != digest(report):
                raise ValueError("immutable final success-test report changed")
        else:
            logging = self.controller.config["logging"]
            if self.controller.wandb_mode != "disabled":
                import wandb
                run = wandb.init(project=logging["wandb_project"], name=logging["wandb_run_name"], entity=os.environ.get("WANDB_ENTITY"),
                                 id=logging.get("wandb_run_id", self.controller.run_id),
                                 resume="allow", mode=self.controller.wandb_mode)
                run.log({"test/round_id": self.controller.round_id,
                         "test/goal_reached_rate": report["goal_reached_rate"],
                         "test/successes": report["successes"],
                         "test/completed_episodes": report["completed"]})
                run.summary.update({"test_checkpoint_sha256": report["checkpoint_sha256"],
                                    "checkpoint_selection_metric": 'final_update_100' if self.fixed100 else "heldout_action_mse"})
                run.finish()
            atomic_json(logging_path, {"report_sha256": digest(report), "mode": self.controller.wandb_mode})
        self.controller.coordinator.post("/v1/mse-rounds/finish-test", {
            "run_id": self.controller.run_id, "round_id": self.controller.round_id,
            "evaluation_id": evaluation_id,
        })
        atomic_json(self.output / "final-test-result.json", report)
        return "mse_final_test_complete"
