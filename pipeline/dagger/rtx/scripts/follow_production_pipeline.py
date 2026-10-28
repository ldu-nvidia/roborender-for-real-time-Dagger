#!/usr/bin/env python3
"""Read-only, high-level terminal progress for a production DAgger run.

The actor intentionally keeps phase telemetry out of stdout.  This follower
reopens the atomically replaced JSON state on every poll and prints only when a
meaningful pipeline boundary changes (plus a low-rate heartbeat).  It never
writes into the run directory and, when coordinator access is enabled, issues
GET requests only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


WINDOW_STEPS = 81
TOTAL_WINDOWS = 40
BATCH_SIZE = 28
TOTAL_VECTOR_STEPS = WINDOW_STEPS * TOTAL_WINDOWS
TOTAL_TRANSITIONS = TOTAL_VECTOR_STEPS * BATCH_SIZE
EXPECTED_PROGRESS_SCHEMA = "roborender.dagger.rtx.phase-boundary.v1"
EXPECTED_DRIVER_SCHEMA = "roborender.dagger.rtx.production-driver.v1"
COMPONENT_ROOT = Path(__file__).resolve().parents[1]


class ProgressError(RuntimeError):
    """The dashboard cannot safely interpret the observed state."""


class CoordinatorUnavailable(RuntimeError):
    """The optional read-only H200 status request is temporarily unavailable."""


def configure_cardinality(config: Mapping[str, Any]) -> int:
    """Bind display totals to the immutable training budget, not validation cadence."""

    execution = config.get("execution")
    validation = config.get("validation")
    total = execution.get("window_count") if isinstance(execution, Mapping) else None
    interval = (
        validation.get("interval_iterations")
        if isinstance(validation, Mapping)
        else None
    )
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval not in (40, 50)
        or total < interval
    ):
        raise ProgressError(
            "immutable config must cover its 40- or 50-window validation cadence"
        )
    global TOTAL_WINDOWS, TOTAL_VECTOR_STEPS, TOTAL_TRANSITIONS
    TOTAL_WINDOWS = total
    TOTAL_VECTOR_STEPS = WINDOW_STEPS * total
    TOTAL_TRANSITIONS = TOTAL_VECTOR_STEPS * BATCH_SIZE
    return total


PHASE_LABELS = {
    "capture_teacher": "ISAAC  | collect 3-view conditions + teacher/Homie actions",
    "prepare_encode_and_persist": "UPLOAD | package 28 synchronized condition records",
    "prepare_local_total": "UPLOAD | package 28 synchronized condition records",
    "prepare_encode_send_wait": "H200   | Image3F render -> student action -> beta selection",
    "commit_intent_journal": "ISAAC  | make selected-action batch durable",
    "native_env_step": "ISAAC  | apply one selected-action batch, env.step([28,26])",
    "commit_outcomes_and_persist": "COMMIT | save rewards/dones/resets for 28 lanes",
    "commit_local_total": "COMMIT | save rewards/dones/resets for 28 lanes",
    "commit_encode_send_wait": "COMMIT | send 28 outcomes to H200 learner ledger",
    "seal_local_audits": "SEAL   | build 28 x 81-transition window manifests",
    "seal_ack_barrier": "SEAL   | verify all 28 lane chunks and barriers",
    "recovery": "RETRY  | idempotent transport recovery; no second env.step",
    "vector_step_barrier": "DONE   | make vector step durable on RTX + H200",
}


@dataclass(frozen=True)
class View:
    run_id: str
    iteration: int
    window: int
    local_step: int
    durable_steps: int
    stage: str
    health: str
    detail: str = ""
    terminal: bool = False
    failed: bool = False

    @property
    def key(self) -> tuple[Any, ...]:
        # Deliberately omit detail (start/end and timing) so the terminal emits
        # one concise line per high-level stage, not a low-level event stream.
        return (
            self.iteration,
            self.window,
            self.local_step,
            self.durable_steps,
            self.stage,
            self.health,
            self.terminal,
            self.failed,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    """Reopen by pathname so atomically renamed updates are always observed."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProgressError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProgressError(f"expected JSON object: {path}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProgressError(f"{name} is not a valid integer")
    return value


def validate_local_identity(
    progress: Mapping[str, Any], driver: Mapping[str, Any], run_id: str
) -> None:
    if progress:
        if progress.get("schema") != EXPECTED_PROGRESS_SCHEMA:
            raise ProgressError("progress schema mismatch")
        if progress.get("run_id") != run_id:
            raise ProgressError("progress belongs to another run")
    if driver:
        if driver.get("schema") != EXPECTED_DRIVER_SCHEMA:
            raise ProgressError("production driver schema mismatch")
        spec = driver.get("spec")
        if not isinstance(spec, Mapping) or spec.get("run_id") != run_id:
            raise ProgressError("production driver belongs to another run")
        if (
            spec.get("num_envs") != BATCH_SIZE
            or spec.get("rollout_steps") != WINDOW_STEPS
            or spec.get("total_windows") != TOTAL_WINDOWS
        ):
            raise ProgressError("production cardinality differs from configured B28 x 81 x window budget")


def load_config(path: Path, run_id: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProgressError(f"invalid production config: {path}") from exc
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProgressError("production config run identity mismatch")
    return value, hashlib.sha256(raw).hexdigest()


class CoordinatorReader:
    """Authenticated GET-only view of the loopback H200 coordinator."""

    def __init__(self, url: str, token_file: Path, run_id: str, config: Mapping[str, Any], config_sha: str):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
            raise ProgressError("coordinator must be loopback HTTP")
        if parsed.port != 18610:
            raise ProgressError("coordinator must use port 18610")
        mode = token_file.stat().st_mode
        if mode & 0o077:
            raise ProgressError("coordinator token must be owner-only")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise ProgressError("invalid coordinator token")
        self.host = parsed.hostname
        self.port = parsed.port
        self.token = token
        self.run_id = run_id
        self.config = dict(config)
        self.config_sha = config_sha

    def get(self) -> dict[str, Any]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
        try:
            connection.request(
                "GET",
                f"/v1/status/{self.run_id}",
                headers={"Authorization": f"Bearer {self.token}"},
            )
            response = connection.getresponse()
            raw = response.read((8 << 20) + 1)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise CoordinatorUnavailable(str(exc)) from exc
        finally:
            connection.close()
        if len(raw) > 8 << 20:
            raise ProgressError("coordinator response exceeds 8 MiB")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProgressError("coordinator returned malformed JSON") from exc
        if not 200 <= response.status < 300:
            raise CoordinatorUnavailable(f"HTTP {response.status}")
        if not isinstance(value, dict) or value.get("ok") is not True:
            raise ProgressError("coordinator status shape mismatch")
        self._validate_identity(value)
        return value

    def _validate_identity(self, value: Mapping[str, Any]) -> None:
        run = value.get("run")
        iterations = value.get("iterations")
        barriers = value.get("barriers")
        if not isinstance(run, Mapping) or not isinstance(iterations, list) or not isinstance(barriers, list):
            raise ProgressError("coordinator status shape mismatch")
        actor_ids = [f"{self.run_id}-g1-lane-{i:03d}" for i in range(BATCH_SIZE)]
        rtx = self.config.get("rtx_actor", {})
        student = self.config.get("student", {})
        validation = self.config.get("validation", {})
        expected = {
            "run_id": self.run_id,
            "config_sha256": self.config_sha,
            "current_round": self.config.get("round_id"),
            "beta": self.config.get("teacher_beta"),
            "rollout_steps": rtx.get("rollout_steps_per_iteration"),
            "actor_ids": actor_ids,
            "initial_checkpoint_sha256": student.get("initial_checkpoint_sha256"),
            "validation_interval_iterations": validation.get("interval_iterations"),
        }
        for name, wanted in expected.items():
            actual = run.get(name)
            if name == "beta":
                try:
                    matches = float(actual) == float(wanted)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == wanted
            if not matches:
                raise ProgressError(f"coordinator run identity mismatch: {name}")


def _coordinator_stage(status: Mapping[str, Any], iteration: int) -> tuple[str, str]:
    run = status.get("run") if isinstance(status, Mapping) else None
    rows = status.get("iterations") if isinstance(status, Mapping) else None
    if not isinstance(run, Mapping):
        return "", ""
    run_status = str(run.get("status", "unknown"))
    current: Mapping[str, Any] = {}
    if isinstance(rows, list):
        current = next(
            (x for x in rows if isinstance(x, Mapping) and x.get("iteration_id") == iteration),
            {},
        )
    iteration_status = str(current.get("status", "unknown"))
    state = run_status if run_status != "collecting" else iteration_status
    if state == "ready_for_learning":
        return "LEARN  | optimizer update queued on H200", state
    if state == "learning":
        return "LEARN  | optimizing model on H200 (one global-mean MSE update)", state
    if state in {"advanced", "collecting"}:
        return "LEARN  | register and activate next student checkpoint", state
    if state == "awaiting_validation":
        return "VALIDATE | validation: fixed-panel success evaluation queued", state
    if state == "validating":
        return "VALIDATE | validation: fixed-panel success evaluation running", state
    if state == "learner_failed" or "fail" in state or "error" in state:
        return "H200: fail-closed", state
    return f"H200 coordinator: {state}", state


def _validation_metric(status: Mapping[str, Any]) -> Any:
    run = status.get("run") if isinstance(status, Mapping) else None
    if isinstance(run, Mapping) and run.get("validation_goal_reached_rate") is not None:
        return run.get("validation_goal_reached_rate")
    rows = status.get("iterations") if isinstance(status, Mapping) else None
    if isinstance(rows, list):
        candidates = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("validation_goal_reached_rate") is not None
        ]
        if candidates:
            candidates.sort(key=lambda row: int(row.get("iteration_id", 0)))
            return candidates[-1].get("validation_goal_reached_rate")
    return None


def build_view(
    run_id: str,
    progress: Mapping[str, Any],
    driver: Mapping[str, Any],
    coordinator: Mapping[str, Any] | None = None,
) -> View:
    validate_local_identity(progress, driver, run_id)
    driver_phase = str(driver.get("phase", "CREATED")) if driver else "CREATED"
    completed_windows = driver.get("completed_windows", []) if driver else []
    if not isinstance(completed_windows, list):
        raise ProgressError("completed_windows must be a list")
    durable = 0
    global_index = 0
    if progress:
        global_index = _integer(progress.get("vector_step", 0), "vector_step")
        counts = progress.get("phase_completion_counts", {})
        if not isinstance(counts, Mapping):
            raise ProgressError("phase_completion_counts must be an object")
        durable = _integer(counts.get("vector_step_barrier", 0), "durable vector steps")
    if global_index >= TOTAL_VECTOR_STEPS or durable > TOTAL_VECTOR_STEPS:
        raise ProgressError("progress exceeds the frozen production cardinality")
    window = min(global_index // WINDOW_STEPS + 1, TOTAL_WINDOWS)
    local_step = global_index % WINDOW_STEPS + 1
    iteration = driver.get("current_iteration_id", window) if driver else window
    iteration = _integer(iteration, "current_iteration_id", minimum=1)

    if driver_phase == "CRASH_UNCERTAIN":
        detail = str(driver.get("uncertain_reason") or driver.get("error_message") or "operator audit required")
        counts = progress.get("counts", {}) if isinstance(progress, Mapping) else {}
        upstream = counts.get("error_message") if isinstance(counts, Mapping) else None
        if upstream:
            detail += f" | cause: {str(upstream)[:600]}"
        return View(run_id, iteration, window, local_step, durable, "FAIL-CLOSED: simulator state is uncertain", "FAILED", detail, True, True)
    if driver_phase == "INITIAL_ATTESTING":
        return View(run_id, iteration, window, local_step, durable, "RTX/H200: attest run and checkpoint identity", "healthy")
    if driver_phase == "READY":
        return View(run_id, iteration, max(1, len(completed_windows) + 1), 1, durable, "RTX: open next rollout window", "healthy")
    if driver_phase == "POSTING_ACTOR_BARRIERS":
        return View(run_id, iteration, max(1, len(completed_windows) + 1), WINDOW_STEPS, durable, "RTX/H200: post and verify 28 actor barriers", "healthy")
    if driver_phase == "WAITING_FOR_CHECKPOINT_ADVANCE":
        stage, raw = _coordinator_stage(coordinator or {}, iteration)
        if not stage:
            unavailable = (coordinator or {}).get("_unavailable")
            stage = "LEARN  | optimize model and activate next checkpoint"
            raw = f"coordinator unavailable: {unavailable}" if unavailable else "coordinator status not enabled"
            return View(run_id, iteration, max(1, len(completed_windows) + 1), WINDOW_STEPS, durable, stage, "attention", raw)
        failed = stage == "H200: fail-closed"
        return View(run_id, iteration, max(1, len(completed_windows) + 1), WINDOW_STEPS, durable, stage, "FAILED" if failed else "healthy", raw, failed, failed)
    if driver_phase == "COMPLETE":
        stage, raw = _coordinator_stage(coordinator or {}, iteration)
        metric = _validation_metric(coordinator or {})
        if metric is not None:
            return View(run_id, iteration, min(iteration, TOTAL_WINDOWS), WINDOW_STEPS, durable, "H200 validation: complete", "healthy", f"goal_reached_rate={metric}", True)
        if not stage:
            stage, raw = (
                f"RTX segment through iteration {iteration} complete; awaiting H200 validation status",
                "status unavailable",
            )
        failed = stage == "H200: fail-closed"
        return View(run_id, iteration, min(iteration, TOTAL_WINDOWS), WINDOW_STEPS, durable, stage, "FAILED" if failed else "healthy", raw, failed, failed)
    if driver_phase not in {"WINDOW_RUNNING", "CREATED"}:
        return View(run_id, iteration, window, local_step, durable, f"RTX driver: {driver_phase}", "healthy")
    if not progress:
        return View(run_id, iteration, window, local_step, durable, "RTX: waiting for native initialization", "healthy")

    phase = str(progress.get("phase", "unknown"))
    event = str(progress.get("event", "unknown"))
    status = str(progress.get("status", "unknown"))
    stage = PHASE_LABELS.get(phase, f"Pipeline: {phase}")
    detail_parts = [event]
    if progress.get("elapsed_ms") is not None:
        try:
            detail_parts.append(f"{float(progress['elapsed_ms']) / 1000:.2f}s")
        except (TypeError, ValueError):
            pass
    # A phase-level error can be followed by the owner's idempotent recovery.
    # Only the durable driver/coordinator may declare the run fail-closed.
    attention = event == "error" or status in {"error", "failed"}
    health = "attention" if attention else "healthy"
    if event == "end" and phase == "vector_step_barrier" and status == "complete":
        stage = "DONE   | vector step durable on RTX + H200"
    return View(run_id, iteration, window, local_step, durable, stage, health, " · ".join(detail_parts))


def format_view(view: View) -> str:
    transitions = min(view.durable_steps * BATCH_SIZE, TOTAL_TRANSITIONS)
    short = view.run_id
    line = (
        f"{utc_now()} | {short} | iter {view.iteration:02d}/{TOTAL_WINDOWS} | "
        f"window {view.window:02d}/{TOTAL_WINDOWS} | step {view.local_step:02d}/{WINDOW_STEPS} | "
        f"durable {view.durable_steps}/{TOTAL_VECTOR_STEPS} "
        f"({transitions}/{TOTAL_TRANSITIONS} transitions) | {view.stage} | {view.health}"
    )
    if view.detail:
        line += f" | {view.detail}"
    return line


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--token-file", type=Path, help="optional owner-only token for GET-only H200 status")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:18610")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--heartbeat", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.heartbeat <= 0:
        parser.error("interval and heartbeat must be positive")
    if (args.config is None) != (args.token_file is None):
        parser.error("--config and --token-file must be supplied together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = args.runtime_root or (
        Path(
            os.environ.get(
                "DAGGER_RTX_RUNTIME_ROOT",
                str(COMPONENT_ROOT / "runtime"),
            )
        )
        / "production-runs"
        / args.run_id
        / "D1"
        / f"{args.run_id}-g1"
    )
    if not runtime.is_dir():
        print(
            f"{utc_now()} | {args.run_id} | dashboard error | "
            f"runtime directory does not exist: {runtime}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    coordinator = None
    if args.config is not None:
        config, config_sha = load_config(args.config, args.run_id)
        configure_cardinality(config)
        coordinator = CoordinatorReader(
            args.coordinator_url, args.token_file, args.run_id, config, config_sha
        )

    last_key: tuple[Any, ...] | None = None
    last_print = 0.0
    coordinator_status: dict[str, Any] = {}
    next_coordinator_poll = 0.0
    while True:
        try:
            progress = load_json(runtime / "progress.json")
            driver = load_json(runtime / "production-driver-state.json")
            now = time.monotonic()
            if coordinator is not None and now >= next_coordinator_poll:
                try:
                    coordinator_status = coordinator.get()
                except CoordinatorUnavailable as exc:
                    coordinator_status = {"_unavailable": str(exc)}
                next_coordinator_poll = now + max(1.0, args.interval)
            view = build_view(args.run_id, progress, driver, coordinator_status)
            if args.once or view.key != last_key or now - last_print >= args.heartbeat:
                print(format_view(view), flush=True)
                last_key = view.key
                last_print = now
            if args.once:
                return 2 if view.failed else 0
            if view.failed:
                return 2
            if view.terminal:
                return 0
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 130
        except ProgressError as exc:
            print(f"{utc_now()} | {args.run_id} | dashboard error | {exc}", file=sys.stderr, flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
