"""Recover a failed next-round PREPARE, never a lost simulation/window.

Default: live read-only admission checks. --admit publishes a new infrastructure
attempt in the existing campaign; it does not allocate, launch, train or eval.
The scheduler, service ledger, and predecessor evidence retain one identity.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from pipeline.dagger.beta_campaign import immutable_json, next_round_config
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import budgets, canonical, validate_config
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.beta_operations import CommandOperations, atomic, process_identity, round_runtime_path
from pipeline.deployment.native_phase_barrier import native_quiescent

SCHEMA = "roborender.round-boundary-recovery.v1"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_boundary(root, first, state, failure):
    """Validate small immutable results; never read a rollout/image archive."""
    root = Path(root).resolve()
    validate_config(first)
    config = validate_config(state["config"])
    rid, completed = state["round_id"], state["completed_rounds"]
    if (state.get("phase") != "ABANDONED" or failure.get("phase") != "PREPARE"
            or failure.get("native_restart_allowed") is not False
            or type(rid) is not int or rid not in (2, 3, 4)
            or first["round_id"] != 1 or state["run_id"] != first["run_id"]
            or state.get("template_sha256") != digest(first)
            or config["run_id"] != first["run_id"] or config["round_id"] != rid
            or [r["round_id"] for r in completed] != list(range(1, rid))):
        raise RuntimeError("recovery requires a failed PREPARE after fully completed rounds")
    abort = state.get("abort_receipt", {})
    if (abort.get("ok") is not True or abort.get("phase") != "abort"
            or abort.get("run_id") != first["run_id"] or abort.get("round_id") != rid
            or abort.get("owned_actors_stopped") is not True
            or abort.get("h200_allocation_released") is not True
            or abort.get("actor_process_reconciliation_warnings") != []):
        raise RuntimeError("scoped cleanup is incomplete or uncertain")
    if (root / f"round-{rid}" / "config.json").exists():
        raise RuntimeError("next round was already prepared/registered; recovery is not admitted")
    previous = None
    for record in completed:
        old = validate_config(json.loads((root / f"round-{record['round_id']}" / "config.json").read_bytes()))
        windows, episodes, _ = budgets(old)
        evaluation, logging = record["evaluation"], record["wandb"]
        if (old["run_id"] != first["run_id"] or old["round_id"] != record["round_id"]
                or record["teacher_beta"] != old["teacher_beta"]
                or record["completed_episodes"] != episodes
                or type(record["successes"]) is not int or not 0 <= record["successes"] <= episodes
                or logging.get("online_metrics_verified") is not True
                or logging.get("run_id") != old["logging"]["wandb_run_id"]
                or evaluation.get("ok") is not True or evaluation.get("phase") != "evaluate"
                or evaluation.get("run_id") != first["run_id"] or evaluation.get("round_id") != old["round_id"]
                or not native_quiescent(evaluation, source_sha256=old["source_sha256"])):
            raise RuntimeError("completed round/evaluation/W&B identity differs")
        if file_sha256(Path(evaluation["comparison_path"])) != evaluation["comparison_sha256"]:
            raise RuntimeError("completed evaluation receipt changed")
        if previous is not None:
            expected = next_round_config(previous["config"], state["first_config_sha256"],
                previous["final_checkpoint"]["sha256"], previous["final_checkpoint"]["path"], old["h200"]["gateway_pids"])
            if old != expected:
                raise RuntimeError("completed round checkpoint/config chain differs")
        else:
            expected = deepcopy(first)
            expected["h200"]["gateway_pids"] = old["h200"]["gateway_pids"]
            if old != expected or state.get("first_config_sha256") != digest(old):
                raise RuntimeError("original round configuration identity differs")
        previous = {**record, "config": old}
    expected = next_round_config(previous["config"], state["first_config_sha256"],
        previous["final_checkpoint"]["sha256"], previous["final_checkpoint"]["path"], config["h200"]["gateway_pids"])
    if config != expected:
        raise RuntimeError("next round must preserve seed, beta, checkpoint and scientific configuration")
    parent = previous["final_checkpoint"]
    if file_sha256(Path(parent["path"])) != parent["sha256"]:
        raise RuntimeError("last completed learner checkpoint is missing/corrupt")
    return parent


def ledger_snapshot(root, state):
    """Read only the closed SQLite metadata, on its owning CPU/shared mount."""
    path = Path(root) / "services/state.sqlite3"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("existing shared ledger is required; never create an empty replacement")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        run, rid = state["run_id"], state["round_id"]
        current = db.execute("SELECT current_round,status FROM runs WHERE run_id=?", (run,)).fetchone()
        if current is None or tuple(current) != (rid - 1, "round_complete"):
            raise RuntimeError("ledger is not at the completed previous-round barrier")
        tables = ("round_iterations", "iteration_barriers", "step_prepares", "learner_jobs", "success_evaluations")
        counts = {table: db.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=? AND round_id>=?",
                                   (run, rid)).fetchone()[0] for table in tables}
        if any(counts.values()):
            raise RuntimeError("next-round activity exists; no boundary recovery or native replay")
        rounds = []
        for record in state["completed_rounds"]:
            old = json.loads((Path(root) / f"round-{record['round_id']}/config.json").read_bytes())
            windows, episodes, _ = budgets(old)
            rows = list(db.execute("SELECT iteration_id,input_checkpoint_sha256,candidate_checkpoint_sha256,status "
                "FROM round_iterations WHERE run_id=? AND round_id=? ORDER BY iteration_id", (run, record["round_id"])))
            parent = old["student"]["initial_checkpoint_sha256"]
            if [r["iteration_id"] for r in rows] != list(range(1, windows+1)):
                raise RuntimeError("completed learner iteration budget differs")
            for row in rows:
                if (row["input_checkpoint_sha256"] != parent or not row["candidate_checkpoint_sha256"]
                        or row["status"] != ("complete" if row["iteration_id"] == windows else "advanced")):
                    raise RuntimeError("completed learner checkpoint chain differs")
                parent = row["candidate_checkpoint_sha256"]
            jobs = list(db.execute("SELECT status,result_json FROM learner_jobs WHERE run_id=? AND round_id=?",
                                   (run, record["round_id"])))
            results = [json.loads(j["result_json"]) for j in jobs if j["status"] == "succeeded" and j["result_json"]]
            results.sort(key=lambda r: r["iteration_id"])
            if (len(jobs) != windows or len(results) != windows
                    or any(r.get("iteration_id") != row["iteration_id"]
                        or r.get("parent_checkpoint_sha256") != row["input_checkpoint_sha256"]
                        or r.get("output_checkpoint_sha256") != row["candidate_checkpoint_sha256"]
                        or r.get("train_chunks") != 12 or r.get("train_frames") != 972
                        or r.get("replay_manifest_sha256") is not None for r, row in zip(results, rows))
                    or parent != record["final_checkpoint"]["sha256"]):
                raise RuntimeError("completed learner receipts differ")
            evaluation = db.execute("SELECT evaluation_id,status,checkpoint_sha256,scenario_manifest_sha256 "
                "FROM success_evaluations WHERE run_id=? AND round_id=? AND purpose='validation'",
                (run, record["round_id"])).fetchall()
            if (len(evaluation) != 1 or evaluation[0]["status"] != "complete"
                    or evaluation[0]["checkpoint_sha256"] != parent
                    or evaluation[0]["scenario_manifest_sha256"] != old["validation"]["scenario_manifest_sha256"]):
                raise RuntimeError("completed success evaluation identity differs")
            total, successes = db.execute("SELECT COUNT(*),COALESCE(SUM(goal_reached),0) "
                "FROM success_evaluation_outcomes WHERE evaluation_id=?", (evaluation[0]["evaluation_id"],)).fetchone()
            if (total, successes) != (episodes, record["successes"]):
                raise RuntimeError("completed success accounting differs")
            rounds.append({"round_id": record["round_id"], "updates": windows, "successes": successes, "episodes": total})
    return {"next_round_counts": counts, "completed_rounds": rounds}


def validate_evidence(root, first, state, failure, evidence, *, now=None):
    parent = validate_boundary(root, first, state, failure)
    now = time.time() if now is None else now
    if (evidence.get("run_id") != state["run_id"] or evidence.get("round_id") != state["round_id"]
            or not 0 <= now - evidence.get("checked_at", 0) <= 300
            or evidence.get("allocation_absent") is not True
            or evidence.get("management_owners_stopped") is not True
            or evidence.get("next_round_native_commands") != 0):
        raise RuntimeError("fresh allocation/owner/no-native-work attestation is required")
    barrier = evidence.get("native_phase_barrier")
    if not native_quiescent({"native_phase_barrier": barrier}, source_sha256=state["config"]["source_sha256"]):
        raise RuntimeError("both attested resident owners must remain IDLE")
    prior = state["completed_rounds"][-1]["evaluation"].get("native_phase_barrier")
    if prior is None or sorted(prior["workers"], key=lambda r: r["physical_gpu"]) != sorted(barrier["workers"], key=lambda r: r["physical_gpu"]):
        raise RuntimeError("resident owner generation changed; simulator restore is not supported")
    if evidence.get("ledger") != ledger_snapshot(root, state):
        raise RuntimeError("closed ledger changed during admission")
    return parent


def validate_binding(root, state):
    binding = state.get("boundary_recovery")
    if binding is None:
        return
    if not isinstance(binding.get("recovery_id"), str) or not re.fullmatch(r"[a-f0-9]{32}", binding["recovery_id"]):
        raise RuntimeError("invalid round-boundary recovery identity")
    directory = Path(root) / "recoveries" / binding["recovery_id"]
    admission = json.loads((directory / "admission.json").read_bytes())
    prior = json.loads((directory / "prior-state.json").read_bytes())
    if (file_sha256(directory / "admission.json") != binding["admission_sha256"]
            or admission.get("schema") != SCHEMA or admission.get("run_id") != state["run_id"]
            or admission.get("template_sha256") != state["template_sha256"]
            or admission.get("prior_state_sha256") != digest(prior)
            or admission.get("native_restart_allowed") is not False
            or admission.get("completed_training_repeated") is not False
            or admission.get("completed_evaluation_repeated") is not False
            or state.get("infrastructure_attempts", {}).get(str(admission["round_id"])) != binding["recovery_id"]
            or state["round_id"] < admission["round_id"]):
        raise RuntimeError("round-boundary recovery admission changed or missing")
    if state["completed_rounds"][:admission["round_id"]-1] != prior["completed_rounds"]:
        raise RuntimeError("completed history changed after boundary recovery")
    if state["round_id"] == admission["round_id"]:
        current = deepcopy(state["config"])
        current["h200"]["gateway_pids"] = prior["config"]["h200"]["gateway_pids"]
        if digest(current) != admission["config_sha256"]:
            raise RuntimeError("scientific configuration changed after boundary recovery")


def publish_recovery(root, first, state, failure, evidence, operations, recovery_id):
    """Caller holds both scheduler.lock and services/bundle-owner.lock."""
    root = Path(root).resolve()
    round_runtime_path(root, state["round_id"], {str(state["round_id"]): recovery_id})
    validate_binding(root, state)  # Retain the chain for a second boundary recovery.
    if state.get("boundary_recovery", {}).get("recovery_id") == recovery_id:
        if state["phase"] == "ABANDONED":
            raise RuntimeError("a failed recovery attempt requires a fresh recovery ID")
        admission = json.loads((root / "recoveries" / recovery_id / "admission.json").read_bytes())
        if admission["operations_sha256"] != digest(operations):
            raise RuntimeError("recovery operations changed on retry")
        return admission
    parent = validate_evidence(root, first, state, failure, evidence)
    if json.loads((root / "state.json").read_bytes()) != state:
        raise RuntimeError("campaign changed during recovery admission")
    directory = root / "recoveries" / recovery_id
    if directory.is_symlink() or directory.parent.is_symlink():
        raise RuntimeError("recovery archive must not be a symlink")
    admission = {"schema": SCHEMA, "recovery_id": recovery_id, "run_id": state["run_id"],
        "round_id": state["round_id"], "template_sha256": state["template_sha256"],
        "prior_state_sha256": digest(state), "config_sha256": digest(state["config"]),
        "parent_checkpoint": parent, "evidence_sha256": digest(evidence),
        "operations_sha256": digest(operations), "native_restart_allowed": False,
        "completed_training_repeated": False, "completed_evaluation_repeated": False}
    if directory.exists():
        saved = json.loads((directory / "admission.json").read_bytes())
        if {k: v for k, v in saved.items() if k != "evidence_sha256"} != {
                k: v for k, v in admission.items() if k != "evidence_sha256"}:
            raise RuntimeError("recovery ID was already used for another boundary")
        retained = json.loads((directory / "evidence.json").read_bytes())
        if digest(retained) != saved["evidence_sha256"]:
            raise RuntimeError("retained recovery evidence changed")
        immutable_json(directory / ("recheck-" + digest(evidence) + ".json"), evidence)
        admission, evidence = saved, retained
    # Append-only evidence first, one atomic control-state switch last. Reusing
    # the same explicit ID after a pre-switch crash is safe; old files differ -> reject.
    for name, value in (("prior-state.json", state), ("prior-failure.json", failure),
                        ("evidence.json", evidence), ("operations.json", operations), ("admission.json", admission)):
        immutable_json(directory / name, value)
    for path in (directory, directory.parent, root):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    updated = deepcopy(state)
    updated.update(phase="PREPARE", operation_id=None, progress=None, updated_at=time.time(),
        infrastructure_attempts={**state.get("infrastructure_attempts", {}), str(state["round_id"]): recovery_id},
        boundary_recovery={"recovery_id": recovery_id, "admission_sha256": file_sha256(directory / "admission.json")},
        operations_sha256=digest(operations))
    for stale in ("abort_receipt", "service_binding", "release_receipt", "evaluation_receipt", "evaluation_report",
                  "final_checkpoint", "final_learner", "logging_receipt", "round_logging", "collection_deadline", "config_path"):
        updated.pop(stale, None)
    atomic(root / "state.json", updated)
    return admission


def inspect_live(site, root, state):
    """Only GETs, owned-process status, and bounded resident journal metadata."""
    allocation = site.allocation("inspect")
    if allocation.get("present") is not False:
        raise RuntimeError("old H200 request still exists; reconcile/release it first")
    for directory in (site.round, round_runtime_path(root, state["round_id"]-1, site.attempts)):
        from pipeline.deployment.owned_process import status
        for label in ("cpu-services", "tunnels", "lease-watchdog"):
            if status(directory / "processes" / label)["state"] not in {"ABSENT", "COMPLETE", "STOPPED", "FAILED"}:
                raise RuntimeError("old management owner is still active/uncertain")
    for directory in (Path(root) / "operations").glob(f"r{state['round_id']}-*"):
        request = json.loads((directory / "request.json").read_bytes())
        if request["phase"] not in {"prepare", "abort"}:
            raise RuntimeError("next-round operation already started")
        saved = json.loads((directory / "process.json").read_bytes())
        actual = process_identity(saved["pid"])
        if ((actual is not None and actual == saved.get("identity"))
                or (actual is None and (Path("/proc") / str(saved["pid"])).exists())):
            raise RuntimeError("old boundary operation is still alive")
    barrier = site.barrier()["native_phase_barrier"]
    # /ready authenticates the live owners; this public journal check additionally
    # rejects even an accepted-but-not-executed command for the unstarted round.
    code = """import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_bytes());run=sys.argv[2];rid=int(sys.argv[3]);count=0
for w in p['workers']:
 s=json.loads((pathlib.Path(w['runtime_root'])/'resident-state.json').read_bytes())
 assert s['worker_instance_id']==w['worker_instance_id'] and s['phase']=='IDLE'
 count+=sum(any(cid.startswith(f'{run}-r{r}-') for r in range(rid,5)) for cid in s['commands'])
print(json.dumps(count))
"""
    commands = site.json([site.site["rtx_python"], "-c", code, site.pool_file, state["run_id"], str(state["round_id"])], "beta")
    return {"run_id": state["run_id"], "round_id": state["round_id"], "checked_at": time.time(),
        "allocation_absent": True, "management_owners_stopped": True,
        "native_phase_barrier": barrier, "next_round_native_commands": commands,
        "ledger": ledger_snapshot(root, state)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-round-config", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--recovery-id", default=None)
    parser.add_argument("--admit", action="store_true", help="Publish continuation only; does not start GPU work")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if "runtime" not in root.parts or args.root.is_symlink():
        parser.error("an existing scoped runtime campaign is required")
    first = json.loads(args.first_round_config.read_bytes())
    operations = json.loads(args.operations.read_bytes())
    CommandOperations(operations, root / "operations")  # Validate executable/hash pins, launch nothing.
    prepare = operations["commands"]["prepare"]
    argv = prepare["argv"]
    if any(any(spec[k] != prepare[k] for k in ("argv", "entrypoint", "entrypoint_sha256"))
           for spec in operations["commands"].values()):
        parser.error("all recovery phases must use the same pinned resident implementation")
    if Path(prepare["entrypoint"]).name != "resident_campaign_site.py":
        parser.error("this recovery version requires the qualified resident site")
    def argument(flag):
        return argv[argv.index(flag)+1]
    site_path = Path(argument("--site"))
    if file_sha256(site_path) != argument("--site-sha256"):
        raise RuntimeError("site identity changed")
    # Never hot-patch a pinned old entrypoint and accidentally run it without
    # attempt isolation. A newly frozen, reviewed binding must contain this version.
    if args.admit and file_sha256(Path(prepare["entrypoint"])) != file_sha256(Path(__file__).with_name("resident_campaign_site.py")):
        raise RuntimeError("admission requires a frozen operations binding with round-boundary recovery support")
    with ExitStack() as stack:
        for path in (root / "scheduler.lock", root / "services/bundle-owner.lock"):
            lock = stack.enter_context(path.open("r+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root / "state.json").read_bytes())
        failure = json.loads((root / "failure.json").read_bytes())
        if args.admit and args.recovery_id and state.get("boundary_recovery", {}).get("recovery_id") == args.recovery_id and state["phase"] != "ABANDONED":
            validate_binding(root, state)
            admission = json.loads((root / "recoveries" / args.recovery_id / "admission.json").read_bytes())
            if admission["operations_sha256"] != digest(operations):
                raise RuntimeError("recovery operations changed on retry")
            print(json.dumps({"published": True, "already_admitted": True, "recovery_id": args.recovery_id,
                              "allocation_performed": False, "workloads_started": False}))
            return
        validate_boundary(root, first, state, failure)
        from pipeline.deployment.resident_campaign_site import ResidentSite
        request = {"phase": "prepare", "run_id": state["run_id"], "round_id": state["round_id"],
            "campaign_root": str(root), "config": state["config"],
            "infrastructure_attempts": state.get("infrastructure_attempts", {})}
        site = ResidentSite(json.loads(site_path.read_bytes()), request,
                            argument("--resident-pool"), argument("--resident-pool-sha256"),
                            management_release=argument("--management-release") if "--management-release" in argv else None,
                            management_release_sha256=argument("--management-release-sha256") if "--management-release-sha256" in argv else None)
        evidence = inspect_live(site, root, state)
        validate_evidence(root, first, state, failure, evidence)
        result = {"admissible": True, "round_id": state["round_id"], "completed_rounds_preserved": len(state["completed_rounds"]),
                  "parent_checkpoint": state["config"]["student"]["initial_checkpoint_sha256"],
                  "allocation_performed": False, "workloads_started": False, "published": False}
        if args.admit:
            # Admission must not bless a mutable/mismatched source tree merely
            # because the single entrypoint hash was refreshed for a dry-run.
            site.verify_sources()
            admission = publish_recovery(root, first, state, failure, evidence, operations, args.recovery_id or uuid.uuid4().hex)
            result.update(published=True, recovery_id=admission["recovery_id"])
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
