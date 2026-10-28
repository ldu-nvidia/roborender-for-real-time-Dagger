"""Replay live diagnostic info through Raven's actual CPU metric callbacks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def summarize_events(trace):
    first_goal = [None] * 5
    max_stage = [0] * 5
    finished = [False] * 5
    outcomes = []
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        for lane, state in enumerate(event["lanes"]):
            if finished[lane]:
                continue
            max_stage[lane] = max(max_stage[lane], state["stage_buf"])
            if state["current_last_stage_completed_task_buf"] and first_goal[lane] is None:
                first_goal[lane] = {"step": event["step"], "terminal": bool(state["reset_buf"]),
                                    "exported_same_step": event["exported_goals"][lane]}
        for row in event["completed_outcomes"]:
            lane = row["lane"]
            outcomes.append({"lane": lane, "steps": row["steps"], "max_stage": max_stage[lane],
                             "first_native_goal": first_goal[lane], "native_success": row["native_success"],
                             "reported_success": row["exported_success"], "timeout": row["timeout"],
                             "reason_predicates": row["diagnostic_reason_predicates"],
                             "arm_qvel_max": row["arm_qvel_max"],
                             "arm_qvel_threshold": row["arm_qvel_threshold"],
                             "front_object_height": row["front_object"][2],
                             "hold_object_height": row["hold_object"][2]})
            finished[lane] = True
    return sorted(outcomes, key=lambda row: row["lane"])


def replay(trace, source):
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WalkStandGrasp")
    names = {"init_eval_metrics_tracking", "update_eval_metrics_per_step",
             "process_eval_episode_completions", "reset_eval_episode_tracking"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(methods) != 4:
        raise RuntimeError("native metric callback structure changed")
    for node in methods:
        node.decorator_list = []
    namespace = {"torch": torch}
    module = ast.Module(body=[ast.ClassDef(name="NativeMetricCallbacks", bases=[], keywords=[],
                                          body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    metric = namespace["NativeMetricCallbacks"]()
    metric.num_envs = 5
    metric.get_object_class_names = lambda: None
    metric.init_eval_metrics_tracking("cpu")
    finished = torch.zeros(5, dtype=torch.bool)
    length = torch.zeros(5, dtype=torch.int64)
    reward = torch.zeros(5)
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        metric.update_eval_metrics_per_step({"grasp_goal_reached": torch.tensor(row["exported_goals"])})
        length += 1
        done = torch.tensor([lane["reset_buf"] for lane in row["lanes"]], dtype=torch.bool).reshape(5)
        first = (done & ~finished).nonzero()
        metric.process_eval_episode_completions(first, reward, length)
        finished |= done
        length[done] = 0
        metric.reset_eval_episode_tracking(done.nonzero())
    values = metric.eval_metrics["goal_reached_buffer"]
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "completed": len(values), "successes": sum(values),
            "success_rate": float(np.mean(values)) if values else None,
            "episode_lengths": metric.eval_metrics["length_buffer"]}


def audit_pair(run, source, arms=("teacher", "bc")):
    if len(arms) != 2 or len(set(arms)) != 2 or any(arm not in {"teacher", "bc", "dagger"} for arm in arms):
        raise ValueError("two distinct diagnostic arms are required")
    initial = [json.loads((run / arm / "initial-state.json").read_text()) for arm in arms]
    if initial[0] != initial[1]:
        raise RuntimeError("initial physical/state observations are not byte-identical")
    result = {}
    identities = []
    for arm in arms:
        original = json.loads((run / arm / "result.json").read_text())
        if (not original["complete"] or original["completed"] != 5
                or original["optimizer_steps"] != 0 or not original["policy_unchanged"]
                or original["accounting_disagreements"] != 0):
            raise RuntimeError("incomplete or mutated inference-only diagnostic")
        identities.append({key: original.get(key) for key in
                           ("goal_accounting_version", "runtime_physics_sha256")})
        actual = replay(run / arm / "steps.jsonl", source)
        if actual["successes"] != original["exported_successes"] or actual["completed"] != original["completed"]:
            raise RuntimeError("diagnostic and native aggregation differ")
        actual["checkpoint_sha256"] = original["checkpoint_sha256"]
        actual["cases"] = summarize_events(run / arm / "steps.jsonl")
        result[arm] = actual
    if identities[0] != identities[1]:
        raise RuntimeError("paired accounting or runtime physics identity differs")
    report = {"schema": "roborender.teacher-bc-diagnostic-audit.v1",
              "matched_initial_array_count": len(initial[0]),
              "paired_physical_and_state_observations_byte_identical": True,
              "sample_size_per_arm": 5, "observation_mode": "native_raytraced_three_view",
              "benchmark_certified": False, "wall_collision_certified": False, "arms": result}
    report.update(identities[0])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--arms", nargs=2, choices=["teacher", "bc", "dagger"], default=["teacher", "bc"])
    args = parser.parse_args()
    report = audit_pair(args.run, args.native_source, tuple(args.arms))
    (args.run / "native-metric-replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
