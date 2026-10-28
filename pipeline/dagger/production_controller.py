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
import shlex
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
from pipeline.dagger.replay import build_replay_manifest, load_replay_config
from pipeline.dagger.services.client import ServiceClient, IdempotentControlClient, ServiceError, RetryableTransportError
from pipeline.dagger.services.recovery_policy import CONTROL_RECOVERY_SECONDS, CONTROL_REQUEST_TIMEOUT_SECONDS
from pipeline.dagger.services.contracts import CANONICAL_BETA, file_sha256
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    SCHEMAS as BETA_SCHEMAS, actor_ids as beta_actor_ids, validate_config as validate_beta_config,
)


CONTROL_GET_ATTEMPTS = 5
CONTROL_GET_INITIAL_BACKOFF_SECONDS = 1.0

RAVEN_FIDELITY_CONTRACT = {
    "raytraced_native_augmentation": True,
    "full_batch_sync_batchnorm": True,
    "random_initial_episode_lengths": True,
}


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
    if isinstance(client, IdempotentControlClient) and client.recovery_seconds is not None:
        # One monotonic deadline owns recovery. Repeating an already exhausted
        # 120-second loop here would hide a permanent outage and starve leases.
        return client.get(path)
    for attempt in range(1, attempts + 1):
        try:
            return client.get(path)
        except urllib.error.HTTPError:
            raise
        except (TimeoutError, ConnectionError, urllib.error.URLError, RetryableTransportError) as exc:
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
    if config.get("schema") in BETA_SCHEMAS:
        validate_beta_config(config, runtime=True)
        student, validation = config["student"], config["validation"]
        if file_sha256(Path(student["initial_checkpoint"])) != student["initial_checkpoint_sha256"]:
            raise ValueError("beta initial checkpoint hash mismatch")
        panel = Path(validation["scenario_manifest"])
        if file_sha256(panel) != validation["scenario_manifest_sha256"]:
            raise ValueError("beta fixed panel hash mismatch")
        from pipeline.dagger.beta_validation import load_panel
        load_panel(config)
        return config
    if config.get("schema") != "roborender.cross_gpu_dagger_round.v1":
        raise ValueError("unsupported production config schema")
    round_id = config.get("round_id")
    if round_id not in CANONICAL_BETA:
        raise ValueError("round_id must be 1..4")
    if config.get("teacher_beta") != CANONICAL_BETA[round_id]:
        raise ValueError("teacher beta differs from the canonical round schedule")
    existing_run_config = config.get("existing_run_config_sha256")
    if existing_run_config is not None and (
        not isinstance(existing_run_config, str)
        or len(existing_run_config) != 64
        or any(character not in "0123456789abcdef" for character in existing_run_config)
        or round_id == 1
    ):
        raise ValueError(
            "rounds 2-4 may attach only to a lowercase SHA256-pinned existing run"
        )
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
    raven_fidelity = config.get("raven_fidelity")
    if raven_fidelity is not None and raven_fidelity != RAVEN_FIDELITY_CONTRACT:
        raise ValueError(
            "raven_fidelity must either be absent for a legacy continuation or "
            f"equal {RAVEN_FIDELITY_CONTRACT!r}"
        )
    replay = load_replay_config(config.get("replay"))
    if replay.get("enabled") and replay["total_chunks_per_update"] != 28:
        raise ValueError(
            "three-view production replay must preserve the B=28 update size"
        )
    validation = config.get("validation", {})
    expected_validation_interval = 40 if round_id == 1 else 50
    if validation.get("interval_iterations") != expected_validation_interval:
        raise ValueError(
            "production success validation cadence must be 40 iterations for "
            "round 1 and 50 iterations for rounds 2-4"
        )
    execution = config.get("execution")
    expected_execution_keys = {
        "window_count",
        "stop_after_actor_barriers",
        "wait_for_checkpoint_advance",
        "learner_submission_authorized",
        "validation_authorized",
    }
    total_windows = execution.get("window_count") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or set(execution) != expected_execution_keys
        or isinstance(total_windows, bool)
        or not isinstance(total_windows, int)
        or total_windows < validation["interval_iterations"]
        or execution.get("stop_after_actor_barriers") is not False
        or execution.get("wait_for_checkpoint_advance") is not True
        or execution.get("learner_submission_authorized") is not True
        or execution.get("validation_authorized") is not True
    ):
        raise ValueError(
            "production execution must authorize learning/validation and request "
            "an integer window_count at least as large as the validation interval"
        )
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
    # Also preserves legacy lightweight Controller.__new__ test/adapter usage.
    mse = None

    def __init__(self, config_path: Path, token_file: Path, *, wandb_mode: str,
                 mse_protocol_path: Path | None = None, mse_worker_command: list[str] | None = None,
                 continuous_diagnostic_scope: Path | None = None,
                 diagnostic_learner_chain: Path | None = None):
        if diagnostic_learner_chain is not None and continuous_diagnostic_scope is None:
            raise ValueError('learner-chain overlay requires its separate native diagnostic scope')
        self.continuous_diagnostic_scope = continuous_diagnostic_scope
        self.diagnostic_learner_chain = diagnostic_learner_chain
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        token = token_file.read_text(encoding="utf-8").strip()
        self.coordinator = ServiceClient("http://127.0.0.1:18610", token, timeout=30)
        self.learner = ServiceClient("http://127.0.0.1:18613", token, timeout=30)
        self.evaluation = ServiceClient("http://127.0.0.1:18614", token, timeout=30)
        self.registry = ServiceClient("http://127.0.0.1:18612", token, timeout=120)
        if self.config.get("schema") in BETA_SCHEMAS:
            info = token_file.stat()
            if info.st_mode & 0o077 or len(token) < 32:
                raise ValueError("beta control token must be owner-only and nonempty")
            for name, port in (("coordinator", 18610), ("registry", 18612), ("learner", 18613), ("evaluation", 18614)):
                setattr(self, name, IdempotentControlClient(f"http://127.0.0.1:{port}", token,
                    timeout=CONTROL_REQUEST_TIMEOUT_SECONDS, attempts=None,
                    recovery_seconds=CONTROL_RECOVERY_SECONDS))
        self.token_file = token_file.resolve()
        self.wandb_mode = wandb_mode
        self.run_id = self.config["run_id"]
        self.round_id = self.config["round_id"]
        self.actor_ids = beta_actor_ids(self.config) if self.config.get("schema") in BETA_SCHEMAS else [
            f"{self.run_id}-g1-lane-{index:03d}"
            for index in range(self.config["rtx_actor"]["concurrent_environments"])
        ]
        self.mse = None
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import uses_explicit_motion
        if uses_explicit_motion(self.config) and mse_protocol_path is None:
            raise ValueError("explicit-motion MSE smoke cannot use legacy success-based stopping")
        if mse_protocol_path is not None:
            from pipeline.dagger.mse_controller import MSEControllerAdapter
            self.mse = MSEControllerAdapter(self, mse_protocol_path, mse_worker_command or [],
                                           output_root=PROJECT_ROOT / "runtime/dagger_mse_selection")
        elif mse_worker_command:
            raise ValueError("MSE worker requires an explicit --mse-protocol")

    def ensure_registered(self) -> None:
        checkpoint = self.config["student"]["initial_checkpoint_sha256"]
        if self.config.get("schema") in BETA_SCHEMAS:
            try:
                record = self.registry.get("/v1/checkpoint-records/" + checkpoint)["checkpoint"]
                if record["sha256"] != checkpoint:
                    raise RuntimeError("registered round parent digest differs")
            except ServiceError as exc:
                if not str(exc).startswith("404 "):
                    raise
                if self.round_id != 1:
                    raise RuntimeError("previous final learner checkpoint is missing from its registry") from exc
                from pipeline.dagger.services.parent_checkpoint import stage_parent
                parent = stage_parent(self.config["student"]["initial_checkpoint"], checkpoint,
                                      PROJECT_ROOT / "checkpoints" / "admitted-round-parents")
                self.registry.post("/v1/checkpoints/register-local", {
                    "path": str(parent), "sha256": checkpoint,
                    "metadata": {"role": "beta_round_parent", "run_id": self.run_id, "round_id": self.round_id},
                })
        existing_run_config = self.config.get("existing_run_config_sha256")
        if existing_run_config is None:
            self.coordinator.post("/v1/runs", {
                "run_id": self.run_id,
                "actor_ids": self.actor_ids,
                "rollout_steps": 81,
                "config_sha256": file_sha256(self.config_path),
                "initial_checkpoint_sha256": checkpoint,
                "validation_interval_iterations": self.config["validation"]["interval_iterations"],
                **({'student_observation': self.config['student_observation']}
                   if 'student_observation' in self.config else {}),
            })
        if self.mse is not None:
            self.mse.ensure_registered()
        status_payload = get_with_transport_retry(
            self.coordinator, f"/v1/status/{self.run_id}"
        )
        status = status_payload["run"]
        if self.config.get("schema") in BETA_SCHEMAS:
            expected_hash = existing_run_config or file_sha256(self.config_path)
            if (status["config_sha256"] != expected_hash or status["actor_ids"] != self.actor_ids
                    or status["rollout_steps"] != 81
                    or status["validation_interval_iterations"] != self.config["validation"]["interval_iterations"]):
                raise RuntimeError("beta coordinator configuration differs")
        if existing_run_config is not None:
            if status["config_sha256"] != existing_run_config:
                raise RuntimeError("existing coordinator run config identity differs")
            if status["actor_ids"] != self.actor_ids or status["rollout_steps"] != 81:
                raise RuntimeError("existing coordinator actor contract differs")
        if status["current_round"] == self.round_id - 1:
            expected_status = "created" if self.round_id == 1 else "round_complete"
            if status["status"] != expected_status:
                raise RuntimeError("previous DAgger round is not complete")
            if self.round_id == 1:
                if status["active_checkpoint_sha256"] != checkpoint:
                    raise RuntimeError("round1 does not start from its initial checkpoint")
            else:
                if self.mse is not None:
                    previous = self.coordinator.get(f"/v1/mse-rounds/{self.run_id}/{self.round_id - 1}")["mse_round"]
                    if (previous["status"] != "complete"
                            or previous["selection"]["best_checkpoint_sha256"] != checkpoint):
                        raise RuntimeError("next round must use previous MSE-selected checkpoint including optimizer")
                else:
                    previous_final_iteration = self.config["execution"]["window_count"]
                    final_rows = [
                        row for row in status_payload.get("iterations", [])
                        if row.get("round_id") == self.round_id - 1
                        and row.get("iteration_id") == previous_final_iteration
                    ]
                    if (len(final_rows) != 1 or final_rows[0].get("status") != "complete"
                            or final_rows[0].get("candidate_checkpoint_sha256") != checkpoint):
                        raise RuntimeError("next round does not start from the previous final learner candidate")
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
        the registered actors contributed exactly one immutable current-iteration
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

        replay_config = load_replay_config(self.config.get("replay"))
        replay_manifest = None
        if replay_config.get("enabled", False):
            replay_manifest = build_replay_manifest(
                config=replay_config, run_id=self.run_id,
                round_id=self.round_id, iteration_id=iteration_id,
                barriers=status["barriers"], iterations=status["iterations"],
            )
        if replay_manifest is not None:
            chunk_ids = [
                sample["chunk_id"] for sample in replay_manifest["samples"]
                if sample["source"] != "bc"
            ]

        raven_exact = self.config.get("raven_fidelity") == RAVEN_FIDELITY_CONTRACT
        learner_batch = len(self.actor_ids) if raven_exact else 7
        request = {
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
                # Preserve Raven's complete local environment batch in every
                # temporal policy forward. This matters because the ResNet
                # contains SyncBatchNorm: four B=7 forwards are not equivalent
                # to B=28 even when their losses form the same global mean.
                "microbatch_environments": learner_batch,
                "raven_exact_full_batch": raven_exact,
                **({"activation_checkpointing": False} if self.config.get("schema") in BETA_SCHEMAS else {}),
                **({"replay_manifest": replay_manifest} if replay_manifest else {}),
                "seed": self.config["seed"],
                "iteration_id": iteration_id,
                "final_validation_iteration": (self.mse.protocol["max_iterations"] if self.mse is not None
                                               else self.config["execution"]["window_count"]),
                "validation_interval_iterations": (self.mse.protocol["interval_iterations"] if self.mse is not None
                                                   else self.config["validation"]["interval_iterations"]),
                **({"metric_protocol": ('fixed100_final_success' if getattr(self.mse, 'fixed100', False)
                                        else "heldout_mse_final_success")} if self.mse is not None else {}),
                "deep_validation_samples": 0,
                "wandb_mode": self.wandb_mode,
                "wandb_project": self.config["logging"]["wandb_project"],
                "wandb_group": f"dagger_{self.round_id}",
                "wandb_run_name": self.config["logging"]["wandb_run_name"],
                "wandb_run_id": self.config["logging"].get("wandb_run_id", self.run_id),
                "wandb_resume": "allow",
                "wandb_extra_config": {
                    **({"rtx_actors": self.config["rtx_actors"]}
                       if self.config.get("schema") in BETA_SCHEMAS else {
                           "rtx_gpu_index": 1, "rtx_gpu_uuid": self.config["rtx_actor"]["gpu_uuid"]}),
                    "concurrent_environments": len(self.actor_ids),
                    "rollout_steps_per_environment": 81,
                    "total_transitions": len(self.actor_ids) * 81,
                    "student_action_probability": 1.0 - self.config["teacher_beta"],
                    "teacher_action_probability": self.config["teacher_beta"],
                    "renderer_worker_count": self.config["h200"][
                        "render_student_worker_count"
                    ],
                    "raven_exact_raytraced_augmentation": raven_exact and not getattr(self.mse, 'fixed100', False),
                    "raven_random_initial_episode_lengths": raven_exact,
                },
            },
        }
        scope_path = getattr(self, 'continuous_diagnostic_scope', None)
        from pipeline.dagger.services import learner_order
        if self.config.get(learner_order.CONFIG_FIELD) == learner_order.PROTOCOL:
            request['parameters'].update({
                learner_order.PARAMETER: learner_order.PROTOCOL,
                'learner_rng_protocol': 'roborender.learner-rng-continuation.v1',
            })
            learner_order.enabled(request)
        if scope_path is not None:
            from pipeline.dagger.services.bc32_diagnostic_parent import PARAMETER, controller_admission
            request['parameters'][PARAMETER] = controller_admission(Path(scope_path), self.config, request)
        chain_path = getattr(self, 'diagnostic_learner_chain', None)
        if chain_path is not None:
            from pipeline.dagger.services import diagnostic_reentry
            admission = diagnostic_reentry.controller_admission(
                chain_path, scope_path, self.config, request)
            diagnostic_reentry.validate_runtime_files(admission, controller_file=Path(__file__).resolve())
            request['parameters'][diagnostic_reentry.PARAMETER] = admission
        self.learner.post("/v1/jobs", request)

    def _register_validation(self, iteration: dict[str, Any]) -> None:
        if self.config.get("schema") in BETA_SCHEMAS:
            from pipeline.dagger.beta_validation import register_validation
            register_validation(self, iteration)
            return
        iteration_id = iteration["iteration_id"]
        interval = self.config["validation"]["interval_iterations"]
        final_iteration = self.config["execution"]["window_count"]
        if iteration_id % interval and iteration_id != final_iteration:
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
        if self.config.get("schema") in BETA_SCHEMAS:
            # The campaign imports the qualified RTX panel and records cloud
            # logging before advancing. The controller must not race it.
            return False
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
            "--wandb-run-id", self.config["logging"].get("wandb_run_id", self.run_id),
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
        current_barriers = [
            barrier
            for barrier in status["barriers"]
            if barrier.get("round_id") == self.round_id
            and barrier.get("iteration_id") == iteration["iteration_id"]
        ]
        total_windows = self.mse.protocol["max_iterations"] if self.mse else self.config["execution"]["window_count"]
        if iteration["iteration_id"] > total_windows:
            if (
                iteration["iteration_id"] != total_windows + 1
                or state != "collecting"
                or current_barriers
            ):
                raise RuntimeError(
                    "coordinator crossed the immutable production window budget "
                    "in a nonempty or unexpected state"
                )
            action = "collection_budget_complete"
        elif state == "ready_for_learning":
            self._submit_learning(iteration, status)
            action = "learner_submitted"
        elif state == "awaiting_mse_validation":
            if self.mse is None:
                raise RuntimeError("MSE round requires its pinned --mse-protocol controller")
            self.mse.evaluate_current()
            action = "mse_validation_recorded"
        elif state == "selected_pending_test":
            if self.mse is None:
                raise RuntimeError("MSE-selected round requires its explicit final-test controller")
            action = self.mse.final_test()
        elif state == "awaiting_validation":
            if self.mse is not None:
                raise RuntimeError("MSE protocol reached a legacy success-selection state")
            self._register_validation(iteration)
            action = "validation_registered"
        elif state == "validating" and self._finalize_validation(iteration):
            action = "validation_finalized"
        elif state in {"learner_failed"}:
            raise RuntimeError(f"production run failed closed: {state}")
        return {
            "run_id": self.run_id,
            "round_id": self.round_id,
            "iteration_id": iteration["iteration_id"],
            "status": state,
            "action": action,
            "actor_barriers": len(current_barriers),
            "expected_actor_barriers": len(self.actor_ids),
            "total_windows": total_windows,
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
    parser.add_argument("--mse-protocol", type=Path, default=None,
                        help="Explicit opt-in held-out MSE selection sidecar; legacy behavior is unchanged without it.")
    parser.add_argument("--mse-worker-command", default=None,
                        help="Argv-quoted inference worker command on allocated compute, never an allocation request.")
    parser.add_argument('--continuous-diagnostic-scope', type=Path, default=None,
                        help='Explicit immutable four-update scope for a BC32 diagnostic parent; never production.')
    parser.add_argument('--diagnostic-learner-chain', type=Path, default=None,
                        help='Separately admitted learner-only RNG/re-entry overlay; native source stays unchanged.')
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    controller = Controller(args.config, args.token_file, wandb_mode=args.wandb_mode,
                            mse_protocol_path=args.mse_protocol,
                            mse_worker_command=shlex.split(args.mse_worker_command) if args.mse_worker_command else None,
                            continuous_diagnostic_scope=args.continuous_diagnostic_scope,
                            diagnostic_learner_chain=args.diagnostic_learner_chain)
    controller.ensure_registered()
    while True:
        report = controller.tick()
        print(json.dumps(report, sort_keys=True), flush=True)
        if (
            args.once
            or report["status"] == "round_complete"
            or report["action"] == "collection_budget_complete"
        ):
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
