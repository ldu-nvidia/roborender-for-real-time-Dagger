"""No GPUs: reject unsafe continuations; never fabricate a live PASS."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from pipeline.dagger.beta_campaign import Campaign, make_first_round_config
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import actor_ids, canonical
from pipeline.deployment.completed_round_recovery import (
    TEARDOWN_FAILURE, publish_admission, validate_completed_round,
)
from pipeline.evaluation.beta_small_batch_panel import make_panel


@pytest.fixture
def completed(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_bytes(canonical(make_panel("d" * 64, smoke=True)))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"CPU test fixture, not a policy")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = make_first_round_config("dagger-leo-smoke-unit-recovery", seed=610191,
        checkpoint=checkpoint, checkpoint_sha=digest(checkpoint), panel=panel,
        panel_sha=digest(panel), source_sha="b" * 64, gateway_pids=[101, 102, 103, 104],
        smoke=True, deployment_host="leo-rtx-6k")
    parent, iterations, jobs = cfg["student"]["initial_checkpoint_sha256"], [], []
    for i in range(1, 4):
        p = tmp_path / f"candidate-{i}.pt"
        p.write_bytes(f"CPU fixture {i}, not a policy".encode())
        candidate = digest(p)
        iterations.append({"round_id": 1, "iteration_id": i,
            "status": "advanced" if i < 3 else "validating",
            "input_checkpoint_sha256": parent, "candidate_checkpoint_sha256": candidate})
        jobs.append({"status": "succeeded", "result": {"iteration_id": i,
            "parent_checkpoint_sha256": parent, "output_checkpoint_sha256": candidate,
            "output_checkpoint_path": str(p), "train_chunks": 12, "train_frames": 972,
            "replay_manifest_sha256": None}})
        parent = candidate
    sha = hashlib.sha256(canonical(cfg)).hexdigest()
    actors = [{"physical_gpu": g, "run_id": cfg["run_id"], "round_id": 1,
        "phase": "COMPLETE", "uncertain_reason": None, "lifecycle_initializations": 1,
        "transition_batches": 243, "physical_transition_rows": 1458,
        "final_candidate_checkpoint_sha256": parent, "config_sha256": sha,
        "windows": [{"index": r["iteration_id"] - 1,
            "input_checkpoint_sha256": r["input_checkpoint_sha256"],
            "output_checkpoint_sha256": r["candidate_checkpoint_sha256"]} for r in iterations]}
        for g in (0, 1)]
    evidence = {"pair": {"passed": False, "failure": TEARDOWN_FAILURE, "exit_codes": [0, 0],
        "run_id": cfg["run_id"], "round_id": 1, "config_sha256": sha}, "actors": actors,
        "native_results": [{"result": "PASS"}, {"result": "PASS"}],
        "iterations": iterations, "learner_jobs": jobs,
        "evaluation_owner_state": "ABSENT", "evaluation_artifacts_absent": True,
        "allocation_absent": True, "ledger_owners_stopped": True,
        "counts": {"prepares": 2916, "commits": 2916, "uncommitted_prepares": 0,
            "chunks": 36, "barriers": 36, "learner_jobs": 3, "evaluation_outcomes": 0}}
    previous = {"schema": "roborender.beta-campaign-state.v1", "phase": "ABANDONED",
        "template_sha256": sha, "first_config_sha256": sha, "run_id": cfg["run_id"],
        "round_id": 1, "completed_rounds": [], "config": cfg,
        "abort_receipt": {"owned_actors_stopped": True, "h200_allocation_released": True}}
    return cfg, evidence, previous


def test_completed_teardown_continuation_starts_at_evaluation_not_collection(completed, tmp_path):
    cfg, evidence, previous = completed
    before = deepcopy(previous)
    root = tmp_path / "continuation"
    admission = publish_admission(root, previous, evidence, "e" * 64)
    assert previous == before
    assert admission["native_restart_allowed"] is False
    assert admission["completed_training_repeated"] is False
    campaign = Campaign(cfg, root, SimpleNamespace())
    assert campaign.state["phase"] == "EVALUATE"
    assert campaign.state["operation_id"] is None
    assert campaign.state["continuation_source_sha256"] == "e" * 64
    # Collection source identity remains attached to the old completed work.
    assert campaign.state["config"]["source_sha256"] == "b" * 64
    with pytest.raises(RuntimeError, match="overwrite"):
        publish_admission(root, previous, evidence, "e" * 64)


@pytest.mark.parametrize("bad", ["error", "exit", "pass", "partial", "uncertain", "chain",
    "count", "evaluation", "evaluation_owner", "owners", "allocation", "learner", "microbatch"])
def test_unsafe_training_or_started_evaluation_never_recovers(completed, bad):
    cfg, evidence, _ = completed
    e = deepcopy(evidence)
    if bad == "error": e["pair"]["failure"] = "OOM"
    if bad == "exit": e["pair"]["exit_codes"][1] = 137
    if bad == "pass": e["native_results"][1]["result"] = "FAIL"
    if bad == "partial": e["actors"][1]["transition_batches"] = 242
    if bad == "uncertain": e["actors"][1]["uncertain_reason"] = "unknown-step"
    if bad == "chain": e["iterations"][1]["input_checkpoint_sha256"] = "a" * 64
    if bad == "count": e["counts"]["uncommitted_prepares"] = 1
    if bad == "evaluation": e["evaluation_artifacts_absent"] = False
    if bad == "evaluation_owner": e["evaluation_owner_state"] = "RUNNING"
    if bad == "owners": e["ledger_owners_stopped"] = False
    if bad == "allocation": e["allocation_absent"] = False
    if bad == "learner": e["learner_jobs"][2]["status"] = "queued"
    if bad == "microbatch": e["learner_jobs"][1]["result"]["train_frames"] = 243
    with pytest.raises(RuntimeError):
        validate_completed_round(cfg, e)


def test_recovery_admission_change_is_rejected_before_any_operation(completed, tmp_path):
    cfg, evidence, previous = completed
    root = tmp_path / "continuation"
    publish_admission(root, previous, evidence, "e" * 64)
    p = root / "recovery-admission.json"
    admission = json.loads(p.read_bytes())
    admission["continuation_source_sha256"] = "f" * 64
    p.write_bytes(canonical(admission))
    with pytest.raises(RuntimeError, match="admission"):
        Campaign(cfg, root, SimpleNamespace())
