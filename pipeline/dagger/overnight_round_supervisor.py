#!/usr/bin/env python3
"""Eight-hour H200 supervisor for the frozen four-round DAgger schedule.

This process owns only control-plane recovery and sequential round activation.
It never touches RTX/Isaac, renderer workers, checkpoints, or learner data.  A
new round is materialized only from the previous round's durable promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.materialize_next_round import materialize
from pipeline.dagger.services.contracts import file_sha256


def atomic_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def snapshot(database_path: Path, run_id: str) -> dict:
    from pipeline.dagger.services.ledger_owner import open_readonly
    database = open_readonly(database_path)
    database.row_factory = sqlite3.Row
    try:
        run = database.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise RuntimeError("scheduled coordinator run is missing")
        iteration = database.execute(
            "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
            "ORDER BY iteration_id DESC LIMIT 1",
            (run_id, run["current_round"]),
        ).fetchone()
        learner = database.execute(
            "SELECT j.status,j.result_json,i.iteration_id FROM learner_jobs j "
            "JOIN learner_job_iterations i ON i.job_id=j.job_id "
            "WHERE i.run_id=? AND i.round_id=? ORDER BY i.iteration_id DESC LIMIT 1",
            (run_id, run["current_round"]),
        ).fetchone()
        return {
            "run": dict(run),
            "iteration": dict(iteration) if iteration is not None else None,
            "learner": dict(learner) if learner is not None else None,
        }
    finally:
        database.close()


def controller_is_running(config_path: Path) -> bool:
    needle = str(config_path.resolve()).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"pipeline.dagger.production_controller" in command and needle in command:
            return True
    return False


def launch_controller(
    config_path: Path, token_file: Path, log_path: Path, wandb_mode: str
) -> int:
    if controller_is_running(config_path):
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pipeline.dagger.production_controller",
            "--config",
            str(config_path.resolve()),
            "--token-file",
            str(token_file.resolve()),
            "--poll-seconds",
            "0.25",
            "--wandb-mode",
            wandb_mode,
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return process.pid


def reload_control_services(token_file: Path) -> dict:
    """Load the latest coordinator/learner code at a quiescent round boundary."""
    targets = {
        "coordinator": ("roborender_h200_services:0.0", 18610),
        "learner": ("roborender_h200_services:3.0", 18613),
    }
    old_pids = {}
    for role, (pane, _port) in targets.items():
        old_pids[role] = int(subprocess.run(
            ["tmux", "display-message", "-p", "-t", pane, "#{pane_pid}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    for pane, _port in targets.values():
        subprocess.run(["tmux", "respawn-pane", "-k", "-t", pane], check=True)

    token = token_file.read_text(encoding="utf-8").strip()
    deadline = time.monotonic() + 60.0
    ready = {}
    while time.monotonic() < deadline:
        ready.clear()
        for role, (_pane, port) in targets.items():
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/ready",
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    value = json.loads(response.read())
                if value.get("ok") is True and value.get("role") == role:
                    ready[role] = value
            except Exception:
                pass
        if len(ready) == len(targets):
            break
        time.sleep(0.5)
    if len(ready) != len(targets):
        raise RuntimeError("reloaded H200 control services failed readiness")
    new_pids = {
        role: int(subprocess.run(
            ["tmux", "display-message", "-p", "-t", pane, "#{pane_pid}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        for role, (pane, _port) in targets.items()
    }
    if any(new_pids[role] == old_pids[role] for role in targets):
        raise RuntimeError("H200 control service PID did not change on reload")
    return {"old_pids": old_pids, "new_pids": new_pids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=PROJECT_ROOT / "pipeline/dagger/config/future_round_schedule.v1.json",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "pipeline/dagger/config/runs/"
        "dagger-r1-raven-fidelity-seed911223830-b28-50iter-20260911T223830Z/"
        "production-config.json",
    )
    parser.add_argument(
        "--state-root", type=Path,
        default=PROJECT_ROOT / "runtime/dagger_services",
    )
    parser.add_argument(
        "--token-file", type=Path,
        default=PROJECT_ROOT / "runtime/secrets/dagger_services.token",
    )
    parser.add_argument(
        "--heartbeat-log", type=Path,
        default=PROJECT_ROOT / "runtime/dagger_services/diagnostics/"
        "overnight-round-supervisor.jsonl",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--duration-seconds", type=float, default=0.0,
        help="0 keeps supervising until all four rounds complete",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    schedule_path = args.schedule.resolve()
    base_config_path = args.base_config.resolve()
    state_root = args.state_root.resolve()
    token_file = args.token_file.resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    run_id = schedule["coordinator_run_id"]
    entries = {item["round_id"]: item for item in schedule["rounds"]}
    configs = {1: base_config_path}
    for configured_round, entry in entries.items():
        candidate = (
            PROJECT_ROOT / "pipeline/dagger/config/runs"
            / entry["wandb_run_id"] / "production-config.json"
        )
        if candidate.exists():
            configs[configured_round] = candidate
    deadline = (
        None if args.duration_seconds <= 0
        else time.monotonic() + args.duration_seconds
    )
    next_heartbeat = 0.0
    reloaded_after_rounds: set[int] = set()

    while deadline is None or time.monotonic() < deadline:
        state = snapshot(state_root / "state.sqlite3", run_id)
        run = state["run"]
        round_id = int(run["current_round"])
        now = time.monotonic()
        if now >= next_heartbeat:
            iteration = state["iteration"] or {}
            learner = state["learner"] or {}
            iteration_id = iteration.get("iteration_id")
            validation_boundaries = (40, 50) if round_id == 1 else (50,)
            next_validation = next(
                (
                    boundary
                    for boundary in validation_boundaries
                    if iteration_id is None or boundary >= iteration_id
                ),
                None,
            )
            atomic_jsonl(args.heartbeat_log.resolve(), {
                "schema": "roborender.dagger.overnight-heartbeat.v1",
                "observed_at_unix": time.time(),
                "run_id": run_id,
                "round_id": round_id,
                "teacher_beta": run["beta"],
                "run_status": run["status"],
                "iteration_id": iteration_id,
                "iteration_status": iteration.get("status"),
                "next_validation_iteration": next_validation,
                "active_checkpoint_sha256": run["active_checkpoint_sha256"],
                "learner_iteration_id": learner.get("iteration_id"),
                "learner_status": learner.get("status"),
            })
            next_heartbeat = now + args.heartbeat_seconds

        # Recover a missing H200 controller without changing healthy workers.
        current_config = configs.get(round_id)
        if current_config is not None and run["status"] in {
            "ready_for_learning", "awaiting_validation", "validating"
        }:
            launch_controller(
                current_config,
                token_file,
                state_root / "logs" / f"overnight-controller-round{round_id}.log",
                args.wandb_mode,
            )

        if run["status"] == "round_complete":
            if round_id == 4:
                atomic_jsonl(args.heartbeat_log.resolve(), {
                    "schema": "roborender.dagger.overnight-heartbeat.v1",
                    "observed_at_unix": time.time(),
                    "run_id": run_id,
                    "status": "all_rounds_complete",
                })
                return 0
            next_round = round_id + 1
            if round_id not in reloaded_after_rounds:
                reload_receipt = reload_control_services(token_file)
                atomic_jsonl(args.heartbeat_log.resolve(), {
                    "schema": "roborender.dagger.overnight-control-reload.v1",
                    "observed_at_unix": time.time(),
                    "run_id": run_id,
                    "completed_round_id": round_id,
                    **reload_receipt,
                })
                reloaded_after_rounds.add(round_id)
            entry = entries[next_round]
            output = (
                PROJECT_ROOT / "pipeline/dagger/config/runs"
                / entry["wandb_run_id"] / "production-config.json"
            )
            if not output.exists():
                materialize(
                    schedule_path=schedule_path,
                    base_config_path=base_config_path,
                    state_root=state_root,
                    round_id=next_round,
                    output=output,
                )
            configs[next_round] = output
            pid = launch_controller(
                output,
                token_file,
                state_root / "logs" / f"overnight-controller-round{next_round}.log",
                args.wandb_mode,
            )
            if pid:
                atomic_jsonl(args.heartbeat_log.resolve(), {
                    "schema": "roborender.dagger.overnight-round-activation.v1",
                    "observed_at_unix": time.time(),
                    "run_id": run_id,
                    "round_id": next_round,
                    "teacher_beta": entry["teacher_beta"],
                    "config_path": str(output),
                    "config_sha256": file_sha256(output),
                    "controller_pid": pid,
                })

        time.sleep(args.poll_seconds)
    atomic_jsonl(args.heartbeat_log.resolve(), {
        "schema": "roborender.dagger.overnight-heartbeat.v1",
        "observed_at_unix": time.time(),
        "run_id": run_id,
        "status": "eight_hour_supervision_window_complete",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
