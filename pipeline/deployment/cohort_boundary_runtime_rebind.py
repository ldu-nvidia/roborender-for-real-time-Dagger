"""One-shot owner-host rebind of inert imported state to fresh services.

The prefix importer intentionally preserves the original round configuration.
After four fresh services are running, but before any tunnel/controller/native
request is submitted, the default path admits only gateway PID changes. An
explicit, revalidated transport-source admission additionally permits its exact
execution source and coverage-plan binding. The transaction changes only the
copied run's config hash. Historical config/source/plan bindings remain in the
durable intent and result. This never touches the quarantined source database,
starts work, stops services, or releases compute.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from pipeline.bc_training.handoff_common import binding, bound, canonical, publish
from pipeline.dagger.coverage_production import digest
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, validate_config
from pipeline.dagger.services.store import StateStore
from pipeline.deployment.cohort_boundary_recovery import (
    HEX64, RECEIPT_SCHEMA, _validate_training_checkpoint, require, stable_file_ref,
)


SCHEMA = "roborender.dagger-cohort-runtime-rebind.v2"
SOURCE_TRANSITION_SCHEMA = "roborender.dagger-cohort-runtime-rebind.v3"
SERVICE_SCHEMA = "roborender.dagger-cohort-runtime-services.v1"


def _proc_identity(pid):
    require(type(pid) is int and pid > 1, "gateway PID must be a positive non-init process")
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    after = stat[stat.rfind(")") + 2:].split()
    require(after[0] not in {"Z", "X", "x"}, "gateway process is terminal")
    return {
        "pid": pid,
        "start_ticks": after[19],
        "cmdline_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest(),
    }


def _exact_configs(old_ref, new_ref, source_transition=None):
    old = validate_config(bound(old_ref), runtime=True)
    new = validate_config(bound(new_ref), runtime=True)
    expected = json.loads(json.dumps(old))
    old_pids = expected["h200"]["gateway_pids"]
    new_pids = new["h200"]["gateway_pids"]
    expected["h200"]["gateway_pids"] = new_pids
    if source_transition is not None:
        expected["source_sha256"] = source_transition["execution_source_sha256"]
        expected["coverage_plan"] = source_transition["execution_coverage_plan"]
    require(new == expected,
            "runtime rebind may change only h200.gateway_pids and an explicitly admitted source/plan transition")
    require(isinstance(new_pids, list) and len(new_pids) == len(set(new_pids)) == 4
            and all(type(pid) is int and pid > 1 for pid in new_pids)
            and new_pids != old_pids,
            "exactly four fresh positive gateway PIDs are required")
    return old, new


def _validated_source_transition(ref, import_ref, old_config_ref, new_config_ref):
    """Rebuild the bound certificate; a claimed approval is insufficient."""
    from pipeline.deployment.cohort_source_transition import SCHEMA as TRANSITION_SCHEMA, build_admission

    value = bound(ref)
    require(isinstance(value, dict) and value.get("schema") == TRANSITION_SCHEMA
            and value.get("import_receipt") == import_ref
            and value.get("historical_config") == old_config_ref
            and value.get("execution_config") == new_config_ref,
            "source transition does not bind the exact import and runtime configs")
    expected = build_admission(
        historical_root=value["historical_source_root"], execution_root=value["execution_source_root"],
        historical_manifest=value["historical_manifest"], execution_manifest=value["execution_manifest"],
        historical_config=old_config_ref, execution_config=new_config_ref, import_receipt=import_ref,
    )
    require(value == expected, "source-transition receipt differs from actual validated artifacts")
    return value


def _validate_services(ref, *, run_id, source_sha256, state_root, gateway_pids):
    """Admit readiness only; ``requests_submitted`` counts mutating work.

    Process readiness probes and identity-pinned SSH forwarding are allowed so
    the four real PIDs can be observed.  Coordinator writes, learner submits,
    native collection commands, and prepare/commit requests must still be zero.
    """
    value = bound(ref)
    fields = {"schema", "run_id", "source_sha256", "ledger_root", "services",
              "fresh_services", "requests_submitted", "allocation_retained", "compute_release_requested"}
    require(isinstance(value, dict) and set(value) == fields and value["schema"] == SERVICE_SCHEMA,
            "invalid fresh-service admission")
    require(value["run_id"] == run_id and value["source_sha256"] == source_sha256
            and value["ledger_root"] == str(state_root)
            and value["fresh_services"] is True and value["requests_submitted"] == 0
            and value["allocation_retained"] is True and value["compute_release_requested"] is False,
            "fresh-service admission differs from inert imported runtime")
    rows = value["services"]
    require(isinstance(rows, list) and len(rows) == 4
            and {row.get("pid") for row in rows} == set(gateway_pids),
            "service admission does not bind the four config gateways")
    for row in rows:
        require(set(row) == {"pid", "start_ticks", "cmdline_sha256"}
                and isinstance(row["start_ticks"], str)
                and isinstance(row["cmdline_sha256"], str) and HEX64.fullmatch(row["cmdline_sha256"]),
                "invalid gateway process identity")
        require(_proc_identity(row["pid"]) == row, "gateway process identity changed")
    return value


def _clean_boundary(db, config, receipt):
    run_id = config["run_id"]
    updates = receipt["completed_updates"]
    next_iteration = updates + 1
    run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    require(run is not None and run["current_round"] == 1 and run["status"] == "collecting"
            and run["beta"] == .75 and run["rollout_steps"] == 81
            and json.loads(run["actor_ids_json"]) == actor_ids(config)
            and run["active_checkpoint_sha256"] == receipt["parent_checkpoint_sha256"],
            "imported round1 run differs from the pinned population/parent boundary")
    rows = db.execute("SELECT * FROM round_iterations WHERE run_id=? ORDER BY iteration_id", (run_id,)).fetchall()
    require(len(rows) == next_iteration
            and [row["iteration_id"] for row in rows] == list(range(1, next_iteration + 1))
            and all(row["round_id"] == 1 for row in rows)
            and all(row["status"] == "advanced" and row["candidate_checkpoint_sha256"] for row in rows[:updates])
            and rows[-1]["status"] == "collecting" and rows[-1]["candidate_checkpoint_sha256"] is None
            and rows[-1]["evaluation_id"] is None and rows[-1]["validation_goal_reached_rate"] is None
            and rows[0]["input_checkpoint_sha256"] == run["initial_checkpoint_sha256"]
            and all(current["input_checkpoint_sha256"] == previous["candidate_checkpoint_sha256"]
                    for previous, current in zip(rows, rows[1:]))
            and rows[-1]["input_checkpoint_sha256"] == receipt["parent_checkpoint_sha256"],
            f"ledger is not the exact clean post-update{updates} boundary")
    jobs = db.execute("SELECT * FROM learner_jobs WHERE run_id=?", (run_id,)).fetchall()
    links = db.execute("SELECT * FROM learner_job_iterations WHERE run_id=? ORDER BY iteration_id", (run_id,)).fetchall()
    require(len(jobs) == len(links) == updates
            and all(row["round_id"] == 1 and row["status"] == "succeeded" for row in jobs)
            and [row["iteration_id"] for row in links] == list(range(1, next_iteration))
            and all(row["round_id"] == 1 for row in links)
            and {row["job_id"] for row in jobs} == {row["job_id"] for row in links}
            and db.execute("SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? AND iteration_id>=?",
                           (run_id, next_iteration)).fetchone()[0] == 0
            and db.execute("SELECT COUNT(*) FROM step_prepares WHERE run_id=? AND rollout_window_id>=?",
                           (run_id, f"window-{updates:06d}")).fetchone()[0] == 0
            and db.execute("SELECT COUNT(*) FROM success_evaluations WHERE run_id=?",
                           (run_id,)).fetchone()[0] == 0
            and db.execute("SELECT COUNT(*) FROM promotions WHERE run_id=?",
                           (run_id,)).fetchone()[0] == 0,
            f"iteration{next_iteration} already has learner/native effects or prefix lineage changed")
    return dict(run)


def _validate_checkpoint_tree(state_root, receipt, db):
    checkpoint_count = receipt["completed_updates"] + 1
    materialization = receipt.get("checkpoint_materialization")
    fields = {
        "schema", "checkpoint_count", "total_size_bytes", "hardlinked_count",
        "reflinked_count", "copied_count", "content_addressed_directory",
        "all_hashes_verified", "all_paths_rewritten", "source_paths_preserved",
        "active_checkpoint",
    }
    require(isinstance(materialization, dict) and set(materialization) == fields
            and materialization["schema"] == "roborender.dagger-cohort-checkpoint-materialization.v1"
            and materialization["checkpoint_count"] == checkpoint_count
            and sum(materialization[key] for key in
                    ("hardlinked_count", "reflinked_count", "copied_count")) == checkpoint_count
            and materialization["content_addressed_directory"] == str(state_root / "checkpoints")
            and materialization["all_hashes_verified"] is True
            and materialization["all_paths_rewritten"] is True
            and materialization["source_paths_preserved"] is True,
            "prefix receipt lacks exact content-addressed checkpoint materialization")
    rows = [dict(row) for row in db.execute("SELECT * FROM checkpoints ORDER BY sha256")]
    require(len(rows) == checkpoint_count and len({row["sha256"] for row in rows}) == checkpoint_count,
            "imported checkpoint registry is not the exact completed-prefix lineage")
    total = 0
    for row in rows:
        expected = state_root / "checkpoints" / f"{row['sha256']}.pt"
        require(row["path"] == str(expected) and expected.is_file() and not expected.is_symlink()
                and expected.stat().st_mode & 0o222 == 0,
                "checkpoint registry path is not a read-only local content address")
        ref = stable_file_ref(expected)
        require(ref["sha256"] == row["sha256"] and ref["size_bytes"] == row["size_bytes"],
                "local content-addressed checkpoint bytes changed")
        total += row["size_bytes"]
    require(total == materialization["total_size_bytes"],
            "materialized checkpoint byte total differs from registry")
    active = receipt.get("parent_checkpoint_sha256")
    require(any(row["sha256"] == active for row in rows),
            "active checkpoint is outside the materialized registry")
    loaded = _validate_training_checkpoint(
        state_root / "checkpoints" / f"{active}.pt",
        active,
        receipt.get("parent_optimizer_global_step"),
        receipt.get("parent_learning_rate"),
    )
    require(materialization["active_checkpoint"] == loaded,
            "active checkpoint load proof differs from actual policy/AdamW payload")
    return rows


def inspect_rebind(state_root, import_ref, old_config_ref, new_config_ref, service_ref,
                   *, source_transition_ref=None):
    state_root = Path(state_root).resolve()
    receipt = bound(import_ref)
    cohorts = receipt.get("completed_cohorts")
    updates = receipt.get("completed_updates")
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("import_complete") is True
            and receipt.get("destination_root") == str(state_root)
            and type(cohorts) is int and cohorts >= 1
            and type(updates) is int and updates == 2 * cohorts
            and receipt.get("next_iteration") == updates + 1
            and receipt.get("native_work_started") is False
            and receipt.get("compute_release_requested") is False,
            "prefix-import receipt does not admit runtime rebind")
    transition = (_validated_source_transition(source_transition_ref, import_ref, old_config_ref, new_config_ref)
                  if source_transition_ref is not None else None)
    old, new = _exact_configs(old_config_ref, new_config_ref, transition)
    require(old["run_id"] == new["run_id"] == receipt["run_id"]
            and old["source_sha256"] == receipt["source_sha256"],
            "runtime configs differ from imported scientific identity")
    _validate_services(service_ref, run_id=old["run_id"], source_sha256=new["source_sha256"],
                       state_root=state_root, gateway_pids=new["h200"]["gateway_pids"])
    store = StateStore(state_root)
    with store.connection() as db:
        row = _clean_boundary(db, old, receipt)
        require(row["config_sha256"] in {digest(old), digest(new)}, "ledger config is neither pre- nor post-rebind")
        _validate_checkpoint_tree(state_root, receipt, db)
    result = {
        "schema": SOURCE_TRANSITION_SCHEMA if transition is not None else SCHEMA,
        "run_id": old["run_id"], "source_sha256": new["source_sha256"],
        "import_receipt": import_ref, "old_config": old_config_ref, "new_config": new_config_ref,
        "service_admission": service_ref, "old_config_sha256": digest(old), "new_config_sha256": digest(new),
        "runtime_only_delta": {"h200.gateway_pids": new["h200"]["gateway_pids"]},
        "completed_updates": updates, "next_iteration": updates + 1, "requests_submitted": 0,
        "allocation_retained": True, "compute_release_requested": False,
    }
    if transition is not None:
        result.update(
            source_transition=source_transition_ref,
            historical_source_sha256=old["source_sha256"],
            execution_source_sha256=new["source_sha256"],
            historical_coverage_plan=old["coverage_plan"],
            execution_coverage_plan=new["coverage_plan"],
            source_transition_delta={"source_sha256": new["source_sha256"], "coverage_plan": new["coverage_plan"]},
            historical_artifacts_preserved=True,
            fresh_native_runtime_required=True,
            fresh_native_qualification_required=True,
            native_replay_allowed=False,
            execution_authorized=False,
        )
    return result


def execute_rebind(state_root, import_ref, old_config_ref, new_config_ref, service_ref,
                   *, source_transition_ref=None):
    state_root = Path(state_root).resolve()
    output = state_root / "runtime-rebind.json"
    intent_path = state_root / "runtime-rebind-intent.json"
    lock_path = state_root / "runtime-rebind.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = inspect_rebind(state_root, import_ref, old_config_ref, new_config_ref, service_ref,
                               source_transition_ref=source_transition_ref)
        intent = {**value, "database_update_authorized": True}
        if intent_path.exists():
            require(bound(binding(intent_path)) == intent, "runtime-rebind intent changed")
        else:
            publish(intent_path, intent)
        store = StateStore(state_root)
        with store.transaction() as db:
            row = _clean_boundary(db, bound(old_config_ref), bound(import_ref))
            if row["config_sha256"] == value["old_config_sha256"]:
                db.execute("UPDATE runs SET config_sha256=? WHERE run_id=?",
                           (value["new_config_sha256"], value["run_id"]))
            else:
                require(row["config_sha256"] == value["new_config_sha256"],
                        "runtime config changed outside admitted transaction")
        result = {**value, "database_update_complete": True, "native_work_started": False}
        if output.exists():
            require(bound(binding(output)) == result, "runtime-rebind result changed")
        else:
            publish(output, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    for name in ("import-receipt", "old-config", "new-config", "service-admission"):
        parser.add_argument("--" + name, nargs=2, metavar=("PATH", "RAW_SHA256"), required=True)
    parser.add_argument("--source-transition", nargs=2, metavar=("PATH", "RAW_SHA256"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ref = lambda pair: {"path": str(Path(pair[0]).resolve()), "raw_sha256": pair[1]}
    inputs = (args.state_root, ref(args.import_receipt), ref(args.old_config),
              ref(args.new_config), ref(args.service_admission))
    kwargs = {"source_transition_ref": ref(args.source_transition) if args.source_transition else None}
    result = execute_rebind(*inputs, **kwargs) if args.execute else inspect_rebind(*inputs, **kwargs)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
