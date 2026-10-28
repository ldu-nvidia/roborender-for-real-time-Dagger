"""Admit a finished round after the attested Docker/cgroup teardown race.

This never starts or restores a simulator. The caller must reconcile the
stopped native owners and closed ledger over the authenticated host boundary.
Old failure evidence stays unchanged; continuation starts at evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from pipeline.dagger.beta_campaign import immutable_json, validate_actor_completion
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, budgets, canonical, validate_config
from pipeline.dagger.services.contracts import file_sha256
from pipeline.deployment.beta_operations import atomic

SCHEMA = "roborender.completed-round-recovery.v1"
TEARDOWN_FAILURE = "RuntimeError('host swap changed and owned memory attribution is unavailable')"


def validate_completed_round(config, evidence):
    validate_config(config, runtime=True)
    windows, _, _ = budgets(config)
    pair = evidence["pair"]
    config_sha = hashlib.sha256(canonical(config)).hexdigest()
    if (pair.get("passed") is not False or pair.get("failure") != TEARDOWN_FAILURE
            or pair.get("exit_codes") != [0, 0]
            or pair.get("run_id") != config["run_id"] or pair.get("round_id") != config["round_id"]
            or pair.get("config_sha256") != config_sha):
        raise RuntimeError("not the attested completed-owner teardown race")
    if (len(evidence.get("native_results", [])) != 2
            or any(r.get("result") != "PASS" for r in evidence["native_results"])
            or evidence.get("evaluation_owner_state") != "ABSENT"
            or evidence.get("evaluation_artifacts_absent") is not True
            or evidence.get("allocation_absent") is not True
            or evidence.get("ledger_owners_stopped") is not True):
        raise RuntimeError("native completion, shutdown or absent-evaluation gate differs")
    receipt = {"actors": evidence["actors"], "actor_ids": actor_ids(config), "owned_actors_stopped": True}
    candidate = validate_actor_completion(config, receipt, evidence["iterations"])
    expected_counts = {"prepares": windows * 81 * 12, "commits": windows * 81 * 12,
        "uncommitted_prepares": 0, "chunks": windows * 12, "barriers": windows * 12,
        "learner_jobs": windows, "evaluation_outcomes": 0}
    if evidence.get("counts") != expected_counts:
        raise RuntimeError("closed-ledger round cardinalities differ")
    jobs = sorted(evidence["learner_jobs"], key=lambda j: j["result"]["iteration_id"])
    rows = sorted(evidence["iterations"], key=lambda r: r["iteration_id"])
    if len(jobs) != windows or len(rows) != windows:
        raise RuntimeError("completed learner budget differs")
    for index, (job, row) in enumerate(zip(jobs, rows), 1):
        r = job["result"]
        if (job.get("status") != "succeeded" or r.get("iteration_id") != index
                or r.get("parent_checkpoint_sha256") != row["input_checkpoint_sha256"]
                or r.get("output_checkpoint_sha256") != row["candidate_checkpoint_sha256"]
                or r.get("train_chunks") != 12 or r.get("train_frames") != 972
                or r.get("replay_manifest_sha256") is not None):
            raise RuntimeError("learner checkpoint chain or full update population differs")
    final = jobs[-1]["result"]
    if file_sha256(Path(final["output_checkpoint_path"])) != candidate:
        raise RuntimeError("completed candidate bytes differ")
    return {"schema": SCHEMA, "run_id": config["run_id"], "round_id": config["round_id"],
        "collection_config_sha256": config_sha,
        "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest(),
        "final_checkpoint": {"path": final["output_checkpoint_path"], "sha256": candidate},
        "final_learner": final, "native_restart_allowed": False,
        "completed_training_repeated": False, "resume_phase": "EVALUATE"}


def continuation_state(previous, admission, *, config_path, source_sha256):
    """Build new durable control state; never edit the abandoned predecessor."""
    abort = previous.get("abort_receipt", {})
    if (previous.get("phase") != "ABANDONED" or previous.get("run_id") != admission["run_id"]
            or previous.get("round_id") != admission["round_id"]
            or abort.get("owned_actors_stopped") is not True
            or abort.get("h200_allocation_released") is not True
            or previous["config"] is None
            or hashlib.sha256(canonical(previous["config"])).hexdigest() != admission["collection_config_sha256"]
            or [r["round_id"] for r in previous["completed_rounds"]] != list(range(1, admission["round_id"]))
            or not isinstance(source_sha256, str) or not re.fullmatch("[0-9a-f]{64}", source_sha256)):
        raise RuntimeError("completed-round continuation state is not admissible")
    value = deepcopy(previous)
    value.update(phase="EVALUATE", operation_id=None, config_path=str(config_path),
        final_checkpoint=admission["final_checkpoint"], final_learner=admission["final_learner"],
        continuation_source_sha256=source_sha256,
        recovery_admission_sha256=hashlib.sha256(canonical(admission)).hexdigest())
    return value


def publish_admission(root, previous, evidence, source_sha256):
    """Publish only small control receipts. Ledger attachment is separate."""
    root = Path(root)
    if root.exists():
        raise RuntimeError("continuation output exists; refusing replay/overwrite")
    admission = validate_completed_round(previous["config"], evidence)
    admission["continuation_source_sha256"] = source_sha256
    path = root / f"round-{previous['round_id']}" / "config.json"
    state = continuation_state(previous, admission, config_path=path, source_sha256=source_sha256)
    root.mkdir(parents=True)
    immutable_json(path, previous["config"])
    immutable_json(root / "recovery-admission.json", admission)
    immutable_json(root / "recovery-evidence.json", evidence)
    atomic(root / "state.json", state)
    return admission
