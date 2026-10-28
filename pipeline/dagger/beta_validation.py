"""Import the actual beta B5 evaluator into the existing success ledger.

No metric is inferred from MSE. Only a complete, identity-matched paired
evaluation may advance the coordinator; retries keep identical outcome keys.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pipeline.evaluation.beta_small_batch_panel import validate_comparison, validate_panel, canonical


def evaluation_id(config):
    return f"{config['run_id']}-r{config['round_id']}-i{config['execution']['window_count']:04d}"


def load_panel(config):
    validation = config["validation"]
    raw = Path(validation["scenario_manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != validation["scenario_manifest_sha256"]:
        raise ValueError("beta evaluation panel hash differs")
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import uses_explicit_motion
    if uses_explicit_motion(config):
        from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel as validate_explicit
        from pipeline.deployment.rtx_actor.scaled_success_contract import is_panel, require_panel
        value=json.loads(raw)
        panel = require_panel(value) if is_panel(value) else validate_explicit(value)
        if panel["purpose"] != "test":
            raise ValueError("MSE-selected smoke requires an independent final test panel")
    else:
        panel = validate_panel(json.loads(raw))
    if (panel["episode_count"] != validation["completed_episodes"]
            or panel["panel_id"] != validation["panel_protocol"]):
        raise ValueError("evaluation panel does not match the declared smoke/production budget")
    return panel


def panel_seeds(panel):
    return [row["assignment"]["simulator_seed"] if "assignment" in row else row["seed"] for row in panel["cohorts"]]


def panel_scenario_ids(panel):
    return [case for row in panel["cohorts"] for case in
            (row["case_ids"] if "assignment" in row else [item["case_id"] for item in row["cases"]])]


def register_validation(controller, iteration):
    config = controller.config
    if iteration["iteration_id"] != config["execution"]["window_count"]:
        raise ValueError("beta success evaluation is only at the declared final iteration")
    panel = load_panel(config)
    return controller.evaluation.post("/v2/evaluations", {
        "evaluation_id": evaluation_id(config), "run_id": config["run_id"],
        "round_id": config["round_id"], "purpose": "validation",
        "checkpoint_sha256": iteration["candidate_checkpoint_sha256"],
        "scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
        "scenario_ids": [case["case_id"] for cohort in panel["cohorts"] for case in cohort["cases"]],
        "observation_mode": "raytraced", "metric": "raven_goal_reached_rate",
    })


def outcome_payloads(config, comparison, baseline_sha, candidate_sha):
    panel = load_panel(config)
    if comparison.get("complete") is not True or comparison.get("probe_complete") is True:
        raise ValueError("a probe is not a completed success evaluation")
    validate_comparison(panel, comparison["arms"], {"bc": baseline_sha, "dagger": candidate_sha})
    eid = evaluation_id(config)
    payloads = []
    for row in comparison["arms"]["dagger"]["outcomes"]:
        reward = row.get("reward")
        if type(reward) not in (float, int) or not math.isfinite(reward):
            raise ValueError("native reward is absent; historical summaries cannot be invented/imported")
        if type(row["steps"]) is not int or row["steps"] < 1 or type(row["timeout"]) is not bool:
            raise ValueError("invalid native episode outcome")
        # Native timeout wins over a previously latched goal. The independent
        # goal_reached field preserves Raven's OR-over-episode success metric.
        termination = "timeout" if row["timeout"] else "success" if row["success"] else "failure"
        payloads.append({
            "idempotency_key": hashlib.sha256(canonical([eid, row["case_id"]])).hexdigest(),
            "scenario_id": row["case_id"], "episode_id": f"{eid}-{row['case_id']}",
            "goal_reached": row["success"], "reward": float(reward), "steps": row["steps"],
            "termination": termination,
            # This native protocol has no separate safety-violation category;
            # this field is not a claim that a safety monitor was evaluated.
            "safety_violation": False,
            "checkpoint_sha256": candidate_sha,
            "scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
        })
    return payloads


def import_comparison(controller, comparison, baseline_sha, candidate_sha):
    """All 256 small receipts; never load image arrays or rescan trajectories."""
    eid = evaluation_id(controller.config)
    payloads = outcome_payloads(controller.config, comparison, baseline_sha, candidate_sha)
    registered = controller.evaluation.get(f"/v2/evaluations/{eid}")["evaluation"]
    if (registered["checkpoint_sha256"] != candidate_sha
            or registered["scenario_manifest_sha256"] != controller.config["validation"]["scenario_manifest_sha256"]
            or registered["run_id"] != controller.run_id or registered["round_id"] != controller.round_id):
        raise ValueError("evaluation ledger identity mismatch")
    for payload in payloads:
        ack = controller.evaluation.post(f"/v2/evaluations/{eid}/outcomes", payload)
        if not ack.get("ok") or ack.get("evaluation_id") != eid or ack.get("scenario_id") != payload["scenario_id"]:
            raise RuntimeError("evaluation outcome acknowledgement differs")
    report = controller.evaluation.get(f"/v2/evaluations/{eid}")["evaluation"]
    candidate = comparison["arms"]["dagger"]
    count = controller.config["validation"]["completed_episodes"]
    if (report["status"] != "complete" or report["completed"] != count or report["total"] != count
            or report["successes"] != candidate["successes"]
            or report["goal_reached_rate"] != candidate["goal_reached_rate"]):
        raise RuntimeError("native result and ledger disagree")
    return report
