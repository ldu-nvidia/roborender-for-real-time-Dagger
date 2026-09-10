#!/usr/bin/env python3
"""Restart-safe H200 controller for one cross-GPU DAgger macro-round.

The RTX process owns Isaac and posts actor barriers.  This controller owns only
the H200 transitions between durable states: one Raven learner update,
intermediate checkpoint activation, and fixed-panel success evaluation at the
configured cadence.  It never starts Isaac or executes an environment step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.raven_success import expand_validation_panel
from pipeline.dagger.services.client import ServiceClient
from pipeline.dagger.services.contracts import CANONICAL_BETA, file_sha256


CONTROL_GET_ATTEMPTS = 5
CONTROL_GET_INITIAL_BACKOFF_SECONDS = 1.0


def get_with_transport_retry(
    client: ServiceClient,
    path: str,
    *,
    attempts: int = CONTROL_GET_ATTEMPTS,
    initial_backoff_seconds: float = CONTROL_GET_INITIAL_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Retry a read-only control-plane request after transient transport faults.

    GET is side-effect free, so an ambiguous timeout can be retried without
    changing coordinator, learner, evaluator, or simulator state.  Protocol and
    HTTP errors remain fail-closed and are deliberately not caught here.
    """

    if attempts <= 0 or initial_backoff_seconds < 0:
        raise ValueError("invalid control GET retry policy")
    for attempt in range(1, attempts + 1):
        try:
            return client.get(path)
        except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            if attempt == attempts:
                raise
            delay = min(initial_backoff_seconds * (2 ** (attempt - 1)), 5.0)
            print(
                json.dumps(
                    {
                        "action": "retry_control_get",
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "next_delay_seconds": delay,
                        "path": path,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "roborender.cross_gpu_dagger_round.v1":
        raise ValueError("unsupported production config schema")
    round_id = config.get("round_id")
    if round_id not in CANONICAL_BETA:
        raise ValueError("round_id must be 1..4")
    if config.get("teacher_beta") != CANONICAL_BETA[round_id]:
        raise ValueError("teacher beta differs from the canonical round schedule")
    actor = config.get("rtx_actor", {})
    if actor.get("physical_gpu_index") != 1:
        raise ValueError("production RTX actor must use physical GPU1")
    if actor.get("gpu_uuid") != "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9":
        raise ValueError("production RTX GPU1 UUID differs from the certified device")
    if 0 not in actor.get("forbidden_gpu_indices", []):
        raise ValueError("RTX GPU0 must be explicitly forbidden")
    if actor.get("concurrent_environments") != 28:
        raise ValueError("production actor batch must contain 28 environments")
    if actor.get("rollout_steps_per_iteration") != 81:
        raise ValueError("Raven rollout windows must contain 81 steps")
    if actor.get("transitions_per_iteration") != 28 * 81:
        raise ValueError("transitions_per_iteration must equal 28*81")
    validation = config.get("validation", {})
    if validation.get("interval_iterations") != 40:
        raise ValueError("production success validation cadence must be 40 iterations")
    if validation.get("minimum_validation_epochs") != 10:
        raise ValueError("minimum_validation_epochs must be 10")
    if validation.get("maximum_validation_epochs") != 40:
        raise ValueError("maximum_validation_epochs must be 40")
    if validation.get("patience") != 4:
        raise ValueError("success validation patience must be 4")
    if validation.get("minimum_absolute_improvement") != 0.01:
        raise ValueError("minimum success-rate improvement must be 0.01")
    if validation.get("metric") != "raven_goal_reached_rate":
        raise ValueError("DAgger validation metric must be Raven goal-reached rate")
    checkpoint = Path(config["student"]["initial_checkpoint"])
    if file_sha256(checkpoint) != config["student"]["initial_checkpoint_sha256"]:
        raise ValueError("initial BC checkpoint hash mismatch")
    panel = Path(validation["scenario_manifest"])
    if file_sha256(panel) != validation["scenario_manifest_sha256"]:
        raise ValueError("fixed validation panel hash mismatch")
    episodes = expand_validation_panel(json.loads(panel.read_text(encoding="utf-8")))
    if len(episodes) != validation["completed_episodes"]:
        raise ValueError("fixed validation panel episode count mismatch")
    return config


class Controller:
    def __init__(self, config_path: Path, token_file: Path, *, wandb_mode: str):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        token = token_file.read_text(encoding="utf-8").strip()
        self.coordinator = ServiceClient("http://127.0.0.1:18610", token, timeout=30)
        self.learner = ServiceClient("http://127.0.0.1:18613", token, timeout=30)
        self.evaluation = ServiceClient("http://127.0.0.1:18614", token, timeout=30)
        self.token_file = token_file.resolve()
        self.wandb_mode = wandb_mode
        self.run_id = self.config["run_id"]
        self.round_id = self.config["round_id"]
        self.actor_ids = [
            f"{self.run_id}-g1-lane-{index:03d}"
            for index in range(self.config["rtx_actor"]["concurrent_environments"])
        ]

    def ensure_registered(self) -> None:
        checkpoint = self.config["student"]["initial_checkpoint_sha256"]
        self.coordinator.post("/v1/runs", {
            "run_id": self.run_id,
            "actor_ids": self.actor_ids,
            "rollout_steps": 81,
            "config_sha256": file_sha256(self.config_path),
            "initial_checkpoint_sha256": checkpoint,
            "validation_interval_iterations": self.config["validation"]["interval_iterations"],
        })
        status = get_with_transport_retry(
            self.coordinator, f"/v1/status/{self.run_id}"
        )["run"]
        if status["current_round"] == 0:
            self.coordinator.post("/v1/rounds/start", {
                "run_id": self.run_id,
                "round_id": self.round_id,
                "beta": self.config["teacher_beta"],
                "checkpoint_sha256": checkpoint,
            })
        elif status["current_round"] != self.round_id:
            raise RuntimeError("configured round does not match the durable run")

    def _run(self, arguments: list[str]) -> None:
        subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)

    def _submit_learning(
        self, iteration: dict[str, Any], status: dict[str, Any]
    ) -> None:
        """Submit the already barrier-admitted current window directly.

        The coordinator and trajectory store already enforce that every one of
        the 28 actors contributed exactly one immutable current-iteration
        chunk.  Re-running a split script, scanning the same rows, writing two
        manifest files, then starting a second submission script added a
        control-plane delay without changing the learner input.  The learner
        endpoint independently rechecks this exact admitted chunk set.
        """

        iteration_id = iteration["iteration_id"]
        barriers = [
            item for item in status["barriers"]
            if item.get("round_id") == self.round_id
            and item.get("iteration_id") == iteration_id
        ]
        by_actor = {item["actor_id"]: item for item in barriers}
        if set(by_actor) != set(self.actor_ids) or len(barriers) != len(self.actor_ids):
            raise RuntimeError("learner submission requires exactly one barrier per actor")
        chunk_ids: list[str] = []
        for actor_id in self.actor_ids:
            values = by_actor[actor_id].get("chunk_ids")
            if not isinstance(values, list) or len(values) != 1:
                raise RuntimeError("each actor barrier must admit exactly one learner chunk")
            chunk_ids.extend(values)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError("a learner chunk was admitted by multiple actor barriers")

        self.learner.post("/v1/jobs", {
            "job_id": (
                f"{self.run_id}-dagger-{self.round_id}-iteration-{iteration_id:04d}"
            ),
            "run_id": self.run_id,
            "round_id": self.round_id,
            "input_checkpoint_sha256": iteration["input_checkpoint_sha256"],
            "train_chunk_ids": chunk_ids,
            "parameters": {
                "learning_rate": self.config["student"]["learning_rate"],
                "max_grad_norm": self.config["student"]["max_gradient_norm"],
                "num_mini_batches": self.config["student"]["mini_batches_per_iteration"],
                "microbatch_environments": 7,
                "seed": self.config["seed"],
                "iteration_id": iteration_id,
                "deep_validation_samples": 0,
                "wandb_mode": self.wandb_mode,
                "wandb_project": self.config["logging"]["wandb_project"],
                "wandb_group": f"dagger_{self.round_id}",
                "wandb_run_name": self.config["logging"]["wandb_run_name"],
                "wandb_run_id": self.run_id,
                "wandb_resume": "allow",
                "wandb_extra_config": {
                    "rtx_gpu_index": 1,
                    "rtx_gpu_uuid": self.config["rtx_actor"]["gpu_uuid"],
                    "concurrent_environments": len(self.actor_ids),
                    "rollout_steps_per_environment": 81,
                    "total_transitions": len(self.actor_ids) * 81,
                    "student_action_probability": 1.0 - self.config["teacher_beta"],
                    "teacher_action_probability": self.config["teacher_beta"],
                    "renderer_worker_count": self.config["h200"][
                        "render_student_worker_count"
                    ],
                },
            },
        })

    def _register_validation(self, iteration: dict[str, Any]) -> None:
        iteration_id = iteration["iteration_id"]
        interval = self.config["validation"]["interval_iterations"]
        if iteration_id % interval:
            raise RuntimeError("coordinator requested success validation off cadence")
        evaluation_id = f"{self.run_id}-r{self.round_id}-i{iteration_id:04d}"
        output = (
            PROJECT_ROOT / f"data/dagger_{self.round_id}/runs/{self.run_id}/evaluations/"
            f"iteration_{iteration_id:04d}/assignment.json"
        )
        self._run([
            sys.executable, str(PROJECT_ROOT / "pipeline/dagger/success_validation.py"),
            "register",
            "--evaluation-id", evaluation_id,
            "--run-id", self.run_id,
            "--round-id", str(self.round_id),
            "--checkpoint-sha256", iteration["candidate_checkpoint_sha256"],
            "--panel", self.config["validation"]["scenario_manifest"],
            "--observation-mode", self.config["validation"]["observation_mode"],
            "--assignment-output", str(output),
            "--token-file", str(self.token_file),
        ])

    def _finalize_validation(self, iteration: dict[str, Any]) -> bool:
        evaluation_id = iteration.get("evaluation_id")
        if not evaluation_id:
            return False
        report = get_with_transport_retry(
            self.evaluation, f"/v2/evaluations/{evaluation_id}"
        )["evaluation"]
        if report["status"] != "complete":
            return False
        output = (
            PROJECT_ROOT / f"data/dagger_{self.round_id}/runs/{self.run_id}/evaluations/"
            f"iteration_{iteration['iteration_id']:04d}/result.json"
        )
        self._run([
            sys.executable, str(PROJECT_ROOT / "pipeline/dagger/success_validation.py"),
            "finalize",
            "--evaluation-id", evaluation_id,
            "--iteration-id", str(iteration["iteration_id"]),
            "--result-output", str(output),
            "--token-file", str(self.token_file),
            "--wandb-mode", self.wandb_mode,
            "--wandb-run-id", self.run_id,
        ])
        return True

    def tick(self) -> dict[str, Any]:
        status = get_with_transport_retry(
            self.coordinator, f"/v1/status/{self.run_id}"
        )
        run = status["run"]
        iteration = status["iterations"][-1]
        state = run["status"]
        action = "wait"
        if state == "ready_for_learning":
            self._submit_learning(iteration, status)
            action = "learner_submitted"
        elif state == "awaiting_validation":
            self._register_validation(iteration)
            action = "validation_registered"
        elif state == "validating" and self._finalize_validation(iteration):
            action = "validation_finalized"
        elif state in {"learner_failed"}:
            raise RuntimeError(f"production run failed closed: {state}")
        current_barriers = [
            barrier
            for barrier in status["barriers"]
            if barrier.get("round_id") == self.round_id
            and barrier.get("iteration_id") == iteration["iteration_id"]
        ]
        return {
            "run_id": self.run_id,
            "round_id": self.round_id,
            "iteration_id": iteration["iteration_id"],
            "status": state,
            "action": action,
            "actor_barriers": len(current_barriers),
            "expected_actor_barriers": len(self.actor_ids),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
    )
    parser.add_argument(
        "--token-file", type=Path,
        default=PROJECT_ROOT / "runtime/secrets/dagger_services.token",
    )
    # This local control-plane GET is cheap and read-only.  Five-second polling
    # added several seconds of pure scheduling latency at every learner barrier.
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    controller = Controller(args.config, args.token_file, wandb_mode=args.wandb_mode)
    controller.ensure_registered()
    while True:
        report = controller.tick()
        print(json.dumps(report, sort_keys=True), flush=True)
        if args.once or report["status"] == "round_complete":
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
