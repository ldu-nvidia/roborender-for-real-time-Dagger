"""Schedule immutable motion/role cohorts without changing native trajectories.

This emits plans only, not GPU admission or execution. Sessions keep a fixed B
because a different B requires separate native lifecycle qualification. The
whole-cohort lane quotas, independent seeds and sampling roles are preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from pipeline.deployment.rtx_actor.isaac6_bc_production import canonical, request_from_cohort, publish_json
from pipeline.deployment.rtx_actor.isaac6_bc_session import validate_session
from pipeline.legacy.bc_collection.sampling_plan import SCHEMA, SCHEMA_V2, SCHEMA_V3, validate_selection_plan
from pipeline.legacy.bc_collection.artifact_roles import offline_roles
from pipeline.legacy.bc_collection.fresh_validation import (
    SCHEMA as FRESH_VALIDATION_SCHEMA, validate_plan as validate_fresh_validation,
    request_from_cohort as fresh_validation_request,
)


def build_schedule(plan, *, run_id, motion_root, rtx_gpus=(0, 1), window_multiplier=3, extra_windows=4,
                   cohorts_per_session=300):
    if (not re.fullmatch(r"bc-leo-[a-z0-9-]{1,36}", run_id)
            or not rtx_gpus or len(set(rtx_gpus)) != len(rtx_gpus)
            or any(type(gpu) is not int or gpu not in (0, 1) for gpu in rtx_gpus)
            or type(window_multiplier) is not int or not 1 <= window_multiplier <= 10
            or type(extra_windows) is not int or not 0 <= extra_windows <= 40
            or type(cohorts_per_session) is not int or not 1 <= cohorts_per_session <= 300
            or not Path(motion_root).is_absolute() or ".." in Path(motion_root).parts):
        raise ValueError("invalid bounded Leo BC schedule")
    if plan.get("schema") not in {SCHEMA, SCHEMA_V2, SCHEMA_V3, FRESH_VALIDATION_SCHEMA}:
        raise ValueError("versioned sampling plan required")
    fresh_validation = plan["schema"] == FRESH_VALIDATION_SCHEMA
    if fresh_validation:
        validate_fresh_validation(plan)
        roles = ("validation",)
    else:
        validate_selection_plan(plan)
        roles = offline_roles(plan)
    explicit_ordinals = plan["schema"] != SCHEMA
    project_request = fresh_validation_request if fresh_validation else request_from_cohort
    cohorts = [cohort for role in roles
               for cohort in plan["partitions"][role]["cohorts"]]
    if (not cohorts or len({c["cohort_id"] for c in cohorts}) != len(cohorts)
            or len({c["process_seed"] for c in cohorts}) != len(cohorts)
            or sum(c["target_chunks"] for c in cohorts) != plan["total_target_chunks"]):
        raise ValueError("sampling plan has duplicate identities/seeds or incorrect quota")
    assignments, totals = defaultdict(list), {gpu: 0 for gpu in rtx_gpus}
    offsets, offset = {}, 0
    for cohort in cohorts:
        offsets[cohort["cohort_id"]] = offset
        offset += cohort["num_envs"]
        gpu = min(rtx_gpus, key=lambda value: (totals[value], value))
        assignments[(gpu, cohort["num_envs"])].append(cohort)
        totals[gpu] += (cohort["minimum_complete_chunk_windows"] * cohort["num_envs"]
                        if explicit_ordinals else cohort["target_chunks"])
    sessions = []
    for (gpu, batch_size), assigned in sorted(assignments.items()):
        # Legacy limit remains 300. The guarded production runner explicitly
        # requests at most four; only lifecycle grouping changes, never seeds,
        # lane/motion assignment, selected ordinals or native chunk semantics.
        for segment, start in enumerate(range(0, len(assigned), cohorts_per_session)):
            group = assigned[start:start + cohorts_per_session]
            session_id = f"bc-session-leo-{run_id.removeprefix('bc-leo-')}-g{gpu}-b{batch_size}-s{segment}"
            requests, descriptors = {}, []
            for index, cohort in enumerate(group):
                out = ("/dagger-runtime/bc_isaac" if index == 0
                       else f"/dagger-runtime/cohorts/{cohort['cohort_id']}/bc_isaac")
                required_windows = (max(slot["selected_chunk_ordinals"][-1] + 1 for slot in cohort["slots"])
                                    if explicit_ordinals else max(slot["target_chunks"] for slot in cohort["slots"]))
                max_windows = required_windows * window_multiplier + extra_windows
                request = project_request(plan, cohort, run_id=run_id, motion_root=motion_root,
                    output_dir=out, rank=gpu, global_env_offset=offsets[cohort["cohort_id"]], max_windows=max_windows)
                name = cohort["cohort_id"] + ".json"
                raw = canonical(request) + b"\n"
                requests[name] = raw
                descriptors.append({"path": "/bc-session-requests/" + name, "sha256": hashlib.sha256(raw).hexdigest()})
            session = {"schema": "roborender.bc-native-session.v1", "session_id": session_id, "requests": descriptors}
            validate_session(session, read_bytes=lambda path: requests[Path(path).name])
            sessions.append({"session": session, "requests": requests, "physical_gpu": gpu,
                "num_envs": batch_size, "target_chunks": sum(c["target_chunks"] for c in group),
                "maximum_vector_steps": sum(json.loads(raw)["max_windows"] * 81 for raw in requests.values())})
    return sessions


def write_schedule(output, plan, sessions):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    publish_json(root / "sampling-plan.json", plan)
    rows = []
    for value in sessions:
        directory = root / value["session"]["session_id"]
        requests_dir = directory / "requests"
        requests_dir.mkdir(parents=True)
        for name, raw in value["requests"].items():
            path = requests_dir / name
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(0o444)
        path = directory / "session.json"
        publish_json(path, value["session"])
        path.chmod(0o444)
        rows.append({k: v for k, v in value.items() if k not in {"requests", "session"}} | {
            "session_id": value["session"]["session_id"], "session_path": str(path.resolve()),
            "session_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "requests_dir": str(requests_dir.resolve())})
    manifest = {"schema": "roborender.bc-native-schedule.v1", "sessions": rows,
        "target_chunks": sum(r["target_chunks"] for r in rows),
        "production_admitted": False, "gpu_capacity_qualified": False,
        "sampling_plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "collection_cost": plan.get("collection_cost"),
        "execution_note": "Fixed-B sessions; native memory/lifecycle admission required before launch; no episode reset at81."}
    publish_json(root / "schedule.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--rtx-gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--window-multiplier", type=int, default=3)
    parser.add_argument("--extra-windows", type=int, default=4)
    parser.add_argument("--cohorts-per-session", type=int, default=300,
                        help="Legacy default 300; use 4 or fewer for guarded production sessions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    sessions = build_schedule(plan, run_id=args.run_id, motion_root=args.motion_root,
        rtx_gpus=args.rtx_gpus, window_multiplier=args.window_multiplier, extra_windows=args.extra_windows,
        cohorts_per_session=args.cohorts_per_session)
    result = write_schedule(args.output, plan, sessions)
    print(json.dumps({"sessions": len(result["sessions"]), "target_chunks": result["target_chunks"],
                      "production_admitted": False}))


if __name__ == "__main__":
    main()
