#!/usr/bin/env python3
"""Materialize the next immutable DAgger round from its final candidate.

Success validation still records the best checkpoint as the round promotion,
but scientific training continuity uses the exact candidate produced by the
previous round's declared final learner iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.production_controller import load_config
from pipeline.dagger.services.contracts import file_sha256


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def materialize(
    *, schedule_path: Path, base_config_path: Path, state_root: Path,
    round_id: int, output: Path, replay_schedule_path: Path | None = None,
) -> dict:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("schema") != "roborender.dagger.future_round_schedule.v1":
        raise ValueError("unsupported future-round schedule")
    entries = [item for item in schedule["rounds"] if item["round_id"] == round_id]
    if len(entries) != 1 or round_id not in (2, 3, 4):
        raise ValueError("round_id must identify exactly one future round")
    entry = entries[0]
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    if base["run_id"] != schedule["coordinator_run_id"]:
        raise ValueError("base config and schedule coordinator differ")
    if file_sha256(base_config_path) != schedule["coordinator_config_sha256"]:
        raise ValueError("base coordinator config hash differs from schedule")

    previous_round_id = round_id - 1
    if previous_round_id == 1:
        final_iteration_id = base["execution"]["window_count"]
    else:
        previous_entries = [
            item for item in schedule["rounds"]
            if item["round_id"] == previous_round_id
        ]
        if len(previous_entries) != 1:
            raise ValueError("previous round is missing from the frozen schedule")
        final_iteration_id = previous_entries[0]["iterations"]
    if type(final_iteration_id) is not int or final_iteration_id <= 0:
        raise ValueError("previous round final iteration must be a positive integer")

    from pipeline.dagger.services.ledger_owner import open_readonly
    database = open_readonly(state_root / "state.sqlite3")
    database.row_factory = sqlite3.Row
    try:
        run = database.execute(
            "SELECT * FROM runs WHERE run_id=?", (base["run_id"],)
        ).fetchone()
        promotion = database.execute(
            "SELECT * FROM promotions WHERE run_id=?", (base["run_id"],)
        ).fetchone()
        final_iteration = database.execute(
            "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
            "AND iteration_id=?",
            (base["run_id"], previous_round_id, final_iteration_id),
        ).fetchone()
    finally:
        database.close()
    if run is None or run["status"] != "round_complete":
        raise RuntimeError("previous round has not completed validation and promotion")
    if run["current_round"] != round_id - 1:
        raise RuntimeError("requested round is not the next sequential round")
    if promotion is None or promotion["round_id"] != previous_round_id:
        raise RuntimeError("previous-round promotion receipt is missing")
    if (
        final_iteration is None
        or final_iteration["status"] != "complete"
        or not final_iteration["candidate_checkpoint_sha256"]
    ):
        raise RuntimeError("previous round final learner candidate is not complete")
    checkpoint_sha = final_iteration["candidate_checkpoint_sha256"]
    checkpoint_path = state_root / "checkpoints" / f"{checkpoint_sha}.pt"
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("final iteration checkpoint bytes are missing or corrupt")

    config = deepcopy(base)
    config["existing_run_config_sha256"] = schedule["coordinator_config_sha256"]
    config["round_id"] = round_id
    config["teacher_beta"] = entry["teacher_beta"]
    config["student_action_probability"] = entry["student_action_probability"]
    config["seed"] = entry["seed"]
    config["execution"]["window_count"] = entry["iterations"]
    config["rtx_actor"]["simulator_seed"] = entry["seed"]
    config["rtx_actor"]["runtime_root"] = (
        f"/mnt/data2/ldu/roborender-runtime/{entry['wandb_run_id']}"
    )
    config["student"]["initial_checkpoint"] = str(checkpoint_path.resolve())
    config["student"]["initial_checkpoint_sha256"] = checkpoint_sha
    config["student"]["initial_checkpoint_bytes"] = checkpoint_path.stat().st_size
    config["validation"]["interval_iterations"] = entry["validation_iterations"][0]
    config["logging"]["wandb_run_id"] = entry["wandb_run_id"]
    config["logging"]["wandb_run_name"] = entry["wandb_run_name"]
    if config.get("replay", {}).get("enabled", False):
        if replay_schedule_path is None:
            replay_schedule_path = (
                PROJECT_ROOT / "pipeline/dagger/config/replay_schedule.v1.json"
            )
        replay_schedule = json.loads(replay_schedule_path.read_text(encoding="utf-8"))
        if replay_schedule.get("schema") != "roborender.dagger.replay_schedule.v1":
            raise ValueError("unsupported replay schedule")
        replay_entries = [
            row for row in replay_schedule["rounds"] if row["round_id"] == round_id
        ]
        if len(replay_entries) != 1:
            raise ValueError("replay schedule lacks exactly one requested round")
        config["replay"]["fractions"] = replay_entries[0]["fractions"]
        config["replay"]["total_chunks_per_update"] = replay_schedule[
            "total_chunks_per_update"
        ]
        config["replay"]["bc_train_manifest"] = replay_schedule["bc_train_manifest"]
        replay_bc_manifest = Path(replay_schedule["bc_train_manifest"]).resolve()
        replay_bc_sha256 = replay_schedule["bc_train_manifest_sha256"]
        if replay_bc_sha256 == "AUTO":
            if not replay_bc_manifest.is_file():
                raise FileNotFoundError(
                    "corrected autoregressive BC split must be activated before "
                    f"materializing replay: {replay_bc_manifest}"
                )
            replay_bc_sha256 = file_sha256(replay_bc_manifest)
        config["replay"]["bc_train_manifest_sha256"] = replay_bc_sha256
        config["replay"]["empty_pool_fallback"] = replay_schedule[
            "empty_pool_fallback"
        ]
    atomic_json(output, config)
    load_config(output)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument(
        "--schedule", type=Path,
        default=PROJECT_ROOT / "pipeline/dagger/config/future_round_schedule.v1.json",
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--replay-schedule", type=Path,
        default=PROJECT_ROOT / "pipeline/dagger/config/replay_schedule.v1.json",
    )
    parser.add_argument(
        "--state-root", type=Path,
        default=PROJECT_ROOT / "runtime/dagger_services",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = materialize(
        schedule_path=args.schedule.resolve(),
        base_config_path=args.base_config.resolve(),
        state_root=args.state_root.resolve(),
        round_id=args.round_id,
        output=args.output.resolve(),
        replay_schedule_path=args.replay_schedule.resolve(),
    )
    print(json.dumps({
        "ok": True,
        "round_id": config["round_id"],
        "teacher_beta": config["teacher_beta"],
        "wandb_run_id": config["logging"]["wandb_run_id"],
        "config_path": str(args.output.resolve()),
        "config_sha256": file_sha256(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
