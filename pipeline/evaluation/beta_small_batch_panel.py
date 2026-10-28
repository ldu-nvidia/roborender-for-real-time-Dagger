"""Explicit beta/Isaac6 B5 panel; never relabel the old B64 panel.

All 52 native cohorts finish five first episodes. Only lane zero of the last
cohort is scored, yielding exactly 256 predetermined cases (not early stopping
on a target number of successes). The other four outcomes remain in evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "roborender.beta-small-batch-panel.v1"
SEED = 1103


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def make_panel(motion_inventory_sha256, *, smoke=False):
    cohorts = []
    for index in range(2 if smoke else 52):
        lanes = list(range(5 if index < 51 else 1))
        cohorts.append({"index": index, "seed": SEED + index * 1009,
                        "cases": [{"case_id": f"beta-b5-{index:03d}-{lane}", "lane": lane,
                                   "motion_index": lane} for lane in lanes]})
    return {"schema": SCHEMA, "panel_id": "beta-isaac6-b5-heldout-10-smoke-v1" if smoke else "beta-isaac6-b5-heldout-256-v4",
            "episode_count": 10 if smoke else 256, "native_batch_size": 5, "cohorts": cohorts,
            "motion_inventory_sha256": motion_inventory_sha256,
            "observation_mode": "raytraced_3view", "optimizer_steps": 0,
            "reset_protocol": "seed_resample_cold_commands_history_contacts_goal_event_clocks_v4",
            "initial_state_match": "byte_exact_physical_and_state_observations",
            "goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
            "legacy_b64_panel_equivalent": False}


def validate_panel(value):
    sha = value.get("motion_inventory_sha256", "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("motion inventory SHA256 required")
    smoke = value.get("panel_id") == "beta-isaac6-b5-heldout-10-smoke-v1"
    if value != make_panel(sha, smoke=smoke):
        raise ValueError("small-batch panel differs from its versioned definition")
    return value


def summarize(panel, cohort_results):
    validate_panel(panel)
    if len(cohort_results) != len(panel["cohorts"]):
        raise ValueError("every cohort in the declared panel is required")
    outcomes = []
    for spec, row in zip(panel["cohorts"], cohort_results):
        if (row.get("seed") != spec["seed"] or row.get("completed") != 5
                or not row.get("complete") or row.get("optimizer_steps") != 0
                or row.get("policy_unchanged") is not True or row.get("accounting_disagreements") != 0):
            raise ValueError("cohort identity/completion/inference-only check failed")
        by_lane = {item["lane"]: item for item in row["outcomes"]}
        if len(row["outcomes"]) != 5 or set(by_lane) != set(range(5)):
            raise ValueError("cohort must contain five unique first-episode outcomes")
        for case in spec["cases"]:
            actual = by_lane[case["lane"]]
            if type(actual["exported_success"]) is not bool:
                raise ValueError("success must be Boolean")
            outcomes.append({**case, "cohort_index": spec["index"], "seed": spec["seed"],
                             "success": actual["exported_success"], "steps": actual["steps"],
                             "timeout": actual["timeout"],
                             **({"reward": actual["episode_reward"]}
                                if "episode_reward" in actual else {})})
    count = panel["episode_count"]
    if len({row["case_id"] for row in outcomes}) != count:
        raise ValueError("panel case identity mismatch")
    successes = sum(row["success"] for row in outcomes)
    return {"schema": "roborender.beta-small-batch-result.v1", "complete": True,
            "panel_sha256": hashlib.sha256(canonical(panel)).hexdigest(),
            "panel_id": panel["panel_id"], "observation_mode": panel["observation_mode"],
            "completed_episodes": count, "successes": successes, "goal_reached_rate": successes / count,
            "native_completed_episodes": 5 * len(panel["cohorts"]), "unscored_episodes": 5 * len(panel["cohorts"]) - count,
            "optimizer_steps": 0, "outcomes": outcomes,
            "legacy_b64_panel_equivalent": False}


def validate_comparison(panel, results, checkpoint_hashes):
    """Small final receipt check; no trajectory/image payloads are rescanned."""
    validate_panel(panel)
    if set(results) != {"bc", "dagger"} or set(checkpoint_hashes) != set(results):
        raise ValueError("a complete BC/candidate pair is required")
    panel_sha = hashlib.sha256(canonical(panel)).hexdigest()
    count, native_count = panel["episode_count"], 5 * len(panel["cohorts"])
    expected_ids = [(case["case_id"], cohort["index"], cohort["seed"], case["lane"], case["motion_index"])
                    for cohort in panel["cohorts"] for case in cohort["cases"]]
    for arm, row in results.items():
        if (row.get("complete") is not True or row.get("completed_episodes") != count
                or row.get("native_completed_episodes") != native_count or row.get("unscored_episodes") != native_count - count
                or row.get("optimizer_steps") != 0 or row.get("policy_unchanged") is not True
                or row.get("checkpoint_sha256") != checkpoint_hashes[arm]
                or row.get("panel_sha256") != panel_sha
                or row.get("goal_accounting_version") != panel["goal_accounting_version"]
                or row.get("observation_mode") != panel["observation_mode"]):
            raise ValueError("paired result identity/completion/inference-only mismatch: " + arm)
        outcomes = row["outcomes"]
        if ([(x["case_id"], x["cohort_index"], x["seed"], x["lane"], x["motion_index"])
             for x in outcomes] != expected_ids or any(type(x["success"]) is not bool for x in outcomes)):
            raise ValueError("paired case identities differ: " + arm)
        successes = sum(x["success"] for x in outcomes)
        if row.get("successes") != successes or row.get("goal_reached_rate") != successes / count:
            raise ValueError("success numerator/denominator mismatch: " + arm)
    if (results["dagger"].get("paired_initial_states") is not True
            or results["bc"].get("runtime_physics_sha256") != results["dagger"].get("runtime_physics_sha256")):
        raise ValueError("paired reset or runtime physics differs")
    physics = results["bc"].get("runtime_physics_sha256", "")
    if len(physics) != 64 or any(c not in "0123456789abcdef" for c in physics):
        raise ValueError("runtime physics attestation missing")
    cells = dict(both_success=0, bc_only=0, candidate_only=0, neither=0)
    for baseline, candidate in zip(results["bc"]["outcomes"], results["dagger"]["outcomes"]):
        key = ("both_success" if candidate["success"] else "bc_only") if baseline["success"] else (
            "candidate_only" if candidate["success"] else "neither")
        cells[key] += 1
    return {"paired_complete": True, "case_count": count, "paired_outcomes": cells,
            "panel_sha256": panel_sha, "runtime_physics_sha256": physics,
            "quality_improvement_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Distinct ten-case smoke panel; not a 256-case estimate")
    args = parser.parse_args()
    raw = canonical(make_panel(hashlib.sha256(args.inventory.read_bytes()).hexdigest(), smoke=args.smoke))
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"panel": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(), "cases": 10 if args.smoke else 256}))


if __name__ == "__main__":
    main()
