#!/usr/bin/env python3
"""Publish an atomic, read-only-derived progress snapshot for a native batch run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rpc_counts(telemetry_path: Path, rollout_index: int) -> tuple[int, int, int]:
    prepares: set[str] = set()
    commits: set[str] = set()
    errors = 0
    try:
        lines = telemetry_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return prepares, commits, errors
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step_key = row.get("step_key") or {}
        if step_key.get("rollout_step_index") != rollout_index or not row.get("path"):
            continue
        success = row.get("http_status") == 200 and row.get("outcome") == "success"
        identity = json.dumps(step_key, sort_keys=True, separators=(",", ":"))
        if row["path"] == "/v1/steps/prepare" and success:
            prepares.add(identity)
        elif row["path"] == "/v1/steps/commit" and success:
            commits.add(identity)
        elif not success:
            errors += 1
    return len(prepares), len(commits), errors


def batch_states(orchestrator_root: Path) -> tuple[int, dict, int]:
    candidates = sorted(orchestrator_root.glob("**/batch-step-*.json"))
    if not candidates:
        return 0, {}, 0
    path = candidates[-1]
    try:
        index = int(path.name.split("-")[2])
    except (IndexError, ValueError):
        index = 0
    completed = sum(load_json(candidate).get("phase") == "COMPLETE" for candidate in candidates)
    return index, load_json(path), completed


def snapshot(runtime_root: Path, run_id: str, batch_size: int) -> dict:
    owner = load_json(runtime_root / "owner" / "batch_owner_state.json")
    index, batch, completed = batch_states(runtime_root / "orchestrator")
    phase = batch.get("phase", "NOT_STARTED")
    prepare_acks, commit_acks, errors = rpc_counts(
        runtime_root / "run-telemetry.jsonl", index
    )
    capture_ready = batch_size if phase in {
        "LOCAL_PREPARED", "H200_SELECTED", "LOCAL_COMMITTED", "COMPLETE"
    } else 0
    selected = batch_size if phase in {"H200_SELECTED", "LOCAL_COMMITTED", "COMPLETE"} else 0
    env_step = phase in {"LOCAL_COMMITTED", "COMPLETE"}
    physical_steps = int(owner.get("next_rollout_step") or 0)
    if phase == "COMPLETE":
        # COMPLETE is written only after B validated remote acknowledgements;
        # a dedicated recovery process may supply them without sharing the
        # original transport telemetry sink.
        commit_acks = batch_size
    return {
        "schema": "roborender.dagger.live-progress.v1",
        "run_id": run_id,
        "batch_size": batch_size,
        "rollout_step_index": index,
        "phase": phase,
        "capture_teacher_ready_lanes": capture_ready,
        "prepare_requests_sent": (
            batch_size
            if phase in {"LOCAL_PREPARED", "H200_SELECTED", "LOCAL_COMMITTED", "COMPLETE"}
            else 0
        ),
        "render_student_selection_responses": min(prepare_acks, batch_size),
        "single_env_step_executed": env_step,
        "commit_acknowledgements": min(commit_acks, batch_size),
        "physical_vector_steps": physical_steps,
        "durable_vector_steps": completed,
        "target_vector_steps": 81,
        "transport_errors_current_step": errors,
        "owner_phase": owner.get("phase"),
        "updated_time_ns": time.time_ns(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    output = args.runtime_root / "progress.json"
    while True:
        value = snapshot(args.runtime_root, args.run_id, args.batch_size)
        atomic_json(output, value)
        if value["durable_vector_steps"] >= value["target_vector_steps"]:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
