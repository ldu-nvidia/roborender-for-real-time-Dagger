"""Validate the selected deployment and print a CPU-only, non-launching plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "pipeline/deployment/profiles/beta_2rtx_4h200.v1.json"


def load_profile(path: Path, root: Path = ROOT) -> tuple[dict, dict]:
    profile = json.loads(path.read_text())
    if profile.get("schema") != "roborender.deployment-profile.v1":
        raise ValueError("unknown deployment profile")
    relative = Path(profile["tested_candidate"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("candidate must be a repository-relative path")
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != profile["tested_candidate_sha256"]:
        raise ValueError("tested candidate changed; requalification required")
    candidate = json.loads(raw)
    if candidate["benchmark_only"] is not True or candidate["production_promoted"] is not False:
        raise ValueError("bounded evidence must not be relabeled as production")
    if candidate["rtx"]["physical_gpu_indices"] != [1, 2] or "0" in profile["rtx_gpu_uuids"]:
        raise ValueError("profile may use only beta GPUs 1 and 2")
    if set(profile["rtx_gpu_uuids"]) != {"1", "2"} or len(set(profile["rtx_gpu_uuids"].values())) != 2:
        raise ValueError("distinct identities for both RTX GPUs are required")
    h200 = candidate["h200"]
    if (h200["resource_shape"] != "custom.gpu.4xh200" or h200["allow_burst"] is not False
            or h200["reservation"] != "cosmos-solution-architect-xcwtffiq"
            or h200["render_student_workers"] != 4):
        raise ValueError("profile violates the approved resource contract")
    if any(type(value) is not bool for value in profile["qualification"].values()):
        raise ValueError("qualification values must be explicit booleans")
    return profile, candidate


def make_plan(profile: dict, candidate: dict, stage: str) -> dict:
    rtx, rollout, learner = candidate["rtx"], candidate["rollout"], candidate["learner"]
    qualification = profile.get("runtime_qualification")
    qualified_windows = 1
    compact_heap = False
    if qualification:
        qualified_windows = qualification["windows"] if qualification["status"] == "passed" else 0
        compact_heap = qualification.get("compact_host_heap", False)
        if type(compact_heap) is not bool:
            raise ValueError("compact_host_heap must be an explicit boolean")
        memory_limit = qualification.get("host_memory_limit_gib_per_actor", rtx["host_memory_limit_gib_per_actor"])
        if type(memory_limit) is not int or not 1 <= memory_limit <= 40:
            raise ValueError("invalid host memory cap")
        rtx = {**rtx, "host_memory_limit_gib_per_actor": memory_limit}
    if stage == "qualification" and qualification:
        count = qualification["environments_per_actor"]
        if type(count) is not int or count not in (4, 6) or qualification["windows"] != 3:
            raise ValueError("unsupported bounded runtime qualification")
        rtx = {**rtx, "environments_per_actor": count}
        rollout = {**rollout, "total_chunks_per_window": 2 * count,
                   "total_transitions_per_window": 2 * count * 81}
        learner = {**learner, "microbatch_environments": 2 * count}
    missing = [name for name, passed in profile["qualification"].items() if not passed]
    plan = {
        "profile_id": profile["profile_id"], "stage": stage,
        "allocation_performed": False, "launch_performed": False,
        "checkpoint": profile["checkpoint"], "wandb": profile["logging"],
        "production_acceptance_pending": missing,
    }
    if stage == "bc":
        plan.update({"rtx_gpu_indices": [], "h200_gpu_count": 1,
                     "resource_shape": "custom.gpu.1xh200",
                     "reservation": candidate["h200"]["reservation"], "allow_burst": False,
                     "requires_new_collection_or_render": False,
                     "trainer": "pipeline/bc_training/train_raven_bc_h200.sh",
                     "batch_chunks": 16, "frames_per_chunk": 81,
                     "metrics": ["train/mse_step", "validation/mse"],
                     "success_evaluation_is_separate": True})
    elif stage == "evaluation":
        # Preserve historical profile parsing, but never advertise the retired
        # ray-traced B5 route as an executable future student evaluation.
        plan.update({"blocked": True, "unattended_production_allowed": False,
                     "reason": "Legacy ray-traced student evaluation is retired. Use run_mode "
                               "with the rendered fixed295/smoke panel and hash-pinned "
                               "rendered_observation_config.",
                     "historical_evaluation": profile.get("success_evaluation", {})})
        return plan
    else:
        plan.update({"rtx_gpu_indices": rtx["physical_gpu_indices"],
                     "rtx_gpu_uuids": profile["rtx_gpu_uuids"],
                     "environments_per_rtx": rtx["environments_per_actor"],
                     "total_environments": rollout["total_chunks_per_window"],
                     "h200": candidate["h200"], "rollout": rollout, "learner": learner,
                     "h200_pool_environment": {
                         "DAGGER_RENDER_GPUS": "0,1,2,3", "IMAGE3F_PORTS": "18500,18501,18502,18503",
                         "DAGGER_GATEWAY_PORTS": "18620,18621,18622,18623",
                         "DAGGER_EPHEMERAL_WINDOW_CACHE": "false",
                     },
                     "qualified_wire_protocol": "v3",
                     "actor_entrypoint": "pipeline/deployment/rtx_actor/run_throughput_pair.py",
                     "host_memory_limit_gib_per_actor": rtx["host_memory_limit_gib_per_actor"],
                     "max_qualified_windows": qualified_windows,
                     "compact_host_heap": compact_heap,
                     "actor_environment": ({"ROBORENDER_RTX_COMPACT_HEAP": "1", "MALLOC_ARENA_MAX": "2",
                                             "MALLOC_TRIM_THRESHOLD_": "131072"} if compact_heap else {}),
                     "unattended_production_allowed": False})
        if stage == "qualification" and qualification:
            plan["requested_qualification"] = qualification
            if qualification["status"] == "failed_container_oom":
                plan["blocked"] = True
                plan["reason"] = qualification["reason"]
        if stage == "dagger":
            plan["blocked"] = True
            plan["reason"] = "Do not use the legacy alpha eight-gateway launcher on beta; complete the listed acceptance gates."
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--stage", choices=["bc", "qualification", "dagger", "evaluation"], default="qualification")
    args = parser.parse_args()
    profile, candidate = load_profile(args.profile)
    plan = make_plan(profile, candidate, args.stage)
    print(json.dumps(plan, indent=2))
    if plan.get("blocked"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
