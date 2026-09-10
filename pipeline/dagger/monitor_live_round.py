#!/usr/bin/env python3
"""Read-only live gauge for the cross-host DAgger vector-step barrier."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


H200_PHASES = (
    "h200_image3f_render_and_student_inference",
    "h200_beta_action_selection",
    "h200_learner_buffer_append",
    "h200_window_seal_and_publish",
)


def phase_snapshot(path: Path, run_id: str, step_index: int, lanes: int,
                   checkpoint_sha256: str | None = None) -> dict:
    paths = sorted(path.glob("*-phases.jsonl")) if path.is_dir() else [path]
    paths = [candidate for candidate in paths if candidate.is_file()]
    if not paths:
        return {"available": False, "path": str(path)}
    counts = {
        phase: {"start": {}, "end": {}, "error": {}} for phase in H200_PHASES
    }
    # Tail only compact records from every gateway. Batch IDs make retries
    # idempotent in this read-only display and lane_count restores B=28 totals.
    for source in paths:
        with source.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 512 * 1024)
            stream.seek(start)
            if start:
                stream.readline()
            for raw_line in stream:
                try:
                    item = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                phase = item.get("phase")
                details = item.get("details") or {}
                if (
                    item.get("run_id") != run_id
                    or item.get("vector_step_index") != step_index
                    or phase not in counts
                    or (
                        checkpoint_sha256 is not None
                        and details.get("checkpoint_sha256") not in {None, checkpoint_sha256}
                    )
                ):
                    continue
                event = item.get("event")
                if event not in counts[phase]:
                    continue
                identity = (
                    str(source), details.get("batch_id"), item.get("actor_id"), phase
                )
                counts[phase][event][identity] = int(details.get("lane_count", 1))
    phases = {}
    for phase in H200_PHASES:
        if phase == "h200_window_seal_and_publish":
            expected = lanes if step_index == 80 else 0
        else:
            expected = lanes
        started = min(expected, sum(counts[phase]["start"].values()))
        completed = min(expected, sum(counts[phase]["end"].values()))
        failed = min(expected, sum(counts[phase]["error"].values()))
        phases[phase] = {
            "started": started,
            "completed": completed,
            "failed": failed,
            "total": expected,
            "in_progress": max(0, started - completed - failed),
        }
    current = next(
        (phase for phase in H200_PHASES if phases[phase]["completed"] < phases[phase]["total"]),
        "h200_step_complete",
    )
    return {
        "available": True, "path": str(path), "files": len(paths),
        "current_phase": current, "phases": phases,
    }


def snapshot(database_path: Path, run_id: str, *, phase_path: Path | None = None,
             learner_progress: Path | None = None) -> dict:
    database = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro", uri=True, timeout=2
    )
    database.row_factory = sqlite3.Row
    try:
        run = database.execute(
            "SELECT actor_ids_json,rollout_steps,status,validation_interval_iterations "
            "FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise RuntimeError(f"unknown run_id: {run_id}")
        lanes = len(json.loads(run["actor_ids_json"]))
        current_iteration = database.execute(
            "SELECT round_id,iteration_id,status,input_checkpoint_sha256 FROM "
            "round_iterations WHERE run_id=? ORDER BY round_id DESC,iteration_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if current_iteration is None:
            raise RuntimeError(f"run has no active iteration: {run_id}")
        rows = list(database.execute(
            """
            SELECT p.rollout_step_index AS step_index,
                   COUNT(*) AS render_student_responses,
                   SUM(CASE WHEN c.idempotency_key IS NOT NULL THEN 1 ELSE 0 END) AS commits
            FROM step_prepares AS p
            LEFT JOIN step_commits AS c
              ON c.prepare_idempotency_key=p.idempotency_key
            WHERE p.run_id=? AND p.round_id=? AND p.checkpoint_sha256=?
            GROUP BY p.rollout_step_index
            ORDER BY p.rollout_step_index
            """,
            (
                run_id, current_iteration["round_id"],
                current_iteration["input_checkpoint_sha256"],
            ),
        ))
        durable_steps = sum(int(row["commits"]) == lanes for row in rows)
        current = rows[-1] if rows else None
        current_step = int(current["step_index"]) if current else 0
        prepared = int(current["render_student_responses"]) if current else 0
        committed = int(current["commits"]) if current else 0
        if current is None or prepared < lanes:
            phase = "rtx_capture_send__h200_render_student_selection"
        elif committed == 0:
            phase = "rtx_vector_env_step__commit_start"
        elif committed < lanes:
            phase = "h200_commit_acknowledgements"
        elif durable_steps < int(run["rollout_steps"]):
            phase = "next_vector_step"
        else:
            phase = "seal_and_barrier"
        barriers = 0
        if current_iteration is not None:
            barriers = database.execute(
                "SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? AND round_id=? "
                "AND iteration_id=?",
                (run_id, current_iteration["round_id"], current_iteration["iteration_id"]),
            ).fetchone()[0]
        result = {
            "run_id": run_id,
            "status": run["status"],
            "phase": phase,
            "current_step_index_zero_based": current_step,
            "durable_steps": durable_steps,
            "rollout_steps": int(run["rollout_steps"]),
            "completion_percent": round(100.0 * durable_steps / int(run["rollout_steps"]), 2),
            "lanes": lanes,
            "h200_render_student_selection_responses": prepared,
            "h200_commit_acknowledgements": committed,
            "sealed_actor_barriers": int(barriers),
            "iteration": dict(current_iteration) if current_iteration is not None else None,
            "validation_interval_iterations": run["validation_interval_iterations"],
            "next_validation_iteration": (
                (
                    (current_iteration["iteration_id"] - 1)
                    // run["validation_interval_iterations"]
                    + 1
                )
                * run["validation_interval_iterations"]
            ),
        }
        if phase_path is not None:
            result["h200_substeps"] = phase_snapshot(
                phase_path, run_id, current_step, lanes,
                current_iteration["input_checkpoint_sha256"],
            )
        if learner_progress is not None and learner_progress.is_file():
            try:
                result["learner"] = json.loads(learner_progress.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result["learner"] = {"available": False, "path": str(learner_progress)}
        return result
    finally:
        database.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--state-root", type=Path,
        default=Path("/workspace/project/roborender/runtime/dagger_services"),
    )
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--phase-telemetry-jsonl", type=Path,
        default=Path("/workspace/project/roborender/runtime/image3f_pool/telemetry"),
    )
    parser.add_argument("--learner-progress", type=Path)
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()
    previous = None
    last_print = 0.0
    while True:
        report = snapshot(
            args.state_root / "state.sqlite3", args.run_id,
            phase_path=args.phase_telemetry_jsonl,
            learner_progress=args.learner_progress,
        )
        now = time.monotonic()
        signature = json.dumps(report, sort_keys=True)
        should_print = signature != previous or now - last_print >= args.heartbeat_seconds
        if not should_print and args.follow:
            time.sleep(args.interval)
            continue
        previous, last_print = signature, now
        if args.human:
            print(
                f"step {report['durable_steps']}/{report['rollout_steps']} "
                f"({report['completion_percent']:.2f}%) | {report['phase']}",
                flush=True,
            )
            substeps = report.get("h200_substeps", {})
            if substeps.get("available"):
                name = substeps["current_phase"]
                if name == "h200_step_complete":
                    print("  H200: render -> student -> select -> persist complete", flush=True)
                else:
                    values = substeps["phases"][name]
                    print(
                        f"  H200: {name} {values['completed']}/{values['total']} "
                        f"({values['in_progress']} running, {values['failed']} failed)",
                        flush=True,
                    )
            if "learner" in report:
                learner = report["learner"]
                print(
                    f"  learner: {learner.get('phase', 'unknown')} "
                    f"{learner.get('status', 'unknown')}", flush=True,
                )
        else:
            print(json.dumps(report, sort_keys=True), flush=True)
        if not args.follow:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
