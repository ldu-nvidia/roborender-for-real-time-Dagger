"""Plan-only shared-population BC/DAgger/held-out/test motion scheduling.

This is a declared scheduling extension, not an original Raven feature. Raven
resets before its training loop and carries state between rollout windows. A
bounded scene cannot silently rotate a 295-motion inventory at each 81-step
window. Here each training cohort is an independent native trainer lifecycle;
its complete windows are continuous. Cohort switching needs explicit runtime
qualification and is not authorized merely by generating this plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from pipeline.legacy.bc_collection.sampling_plan import SCHEMA as SAMPLING_SCHEMA, SCHEMA_V2 as SAMPLING_SCHEMA_V2, SCHEMA_V3 as SAMPLING_SCHEMA_V3, canonical, validate_selection_plan
from pipeline.legacy.bc_collection.artifact_roles import OFFLINE_ROLES, validate_artifact_contract
from pipeline.deployment.rtx_actor.native_motion_assignment import SCHEMA as ASSIGNMENT_SCHEMA, validate_motion_assignment

SCHEMA = "roborender.matched-motion-cohort-schedule.v1"
BETAS = (0.75, 0.5, 0.25, 0.0)
LIFECYCLE_RULE = "native_train_entry_to_declared_cohort_completion; never_reset_at_81_boundary"


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)


def _validate_inputs(inventory, sampling_plan):
    rows = inventory.get("motions", [])
    if (inventory.get("schema") != "roborender.motion-inventory.v1"
            or not rows or inventory.get("motion_count") != len(rows)
            or inventory.get("motion_records_sha256") != hashlib.sha256(canonical(rows)).hexdigest()
            or len({r.get("motion_id") for r in rows}) != len(rows)
            or len({r.get("sha256") for r in rows}) != len(rows)):
        raise ValueError("invalid unique pinned motion inventory")
    modes = {}
    for row in rows:
        if (not _identifier(row.get("motion_id")) or not _sha(row.get("sha256"))
                or Path(row.get("filename", "")).name != row.get("filename")
                or row.get("filename") != row["motion_id"] + ".npz"):
            raise ValueError("invalid motion filename/hash")
        modes[row["motion_id"]] = row["sha256"]
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    explicit_native_test = sampling_plan.get("schema") == SAMPLING_SCHEMA_V3
    expected_roles = set(OFFLINE_ROLES) if explicit_native_test else {"train", "validation", "test"}
    if (sampling_plan.get("schema") not in {SAMPLING_SCHEMA, SAMPLING_SCHEMA_V2, SAMPLING_SCHEMA_V3}
            or sampling_plan.get("motion_inventory_sha256") != inventory_sha
            or sampling_plan.get("frames_per_chunk") != 81
            or set(sampling_plan.get("partitions", {})) != expected_roles):
        raise ValueError("BC sampling plan differs from common motion population")
    validate_selection_plan(sampling_plan)
    total = sampling_plan.get("total_target_chunks")
    if explicit_native_test:
        # The v3 validator binds positive 8:1 offline quotas/full B16 plus a
        # separately frozen native panel. Never invent an offline test split.
        if type(total) is not int or total <= 0:
            raise ValueError("invalid v3 offline chunk total")
    elif (type(total) is not int or total <= 0 or total % 10
          or {role: p["target_chunks"] for role, p in sampling_plan["partitions"].items()}
          != {"train": total * 8 // 10, "validation": total // 10, "test": total // 10}):
        raise ValueError("BC quotas must preserve explicit 80/10/10 counts")
    seeds = set()
    for role, partition in sampling_plan["partitions"].items():
        seen = {}
        count = 0
        for cohort in partition["cohorts"]:
            seed = cohort["process_seed"]
            if type(seed) is not int or not 0 <= seed < 2**31 or seed in seeds:
                raise ValueError("BC cohorts require disjoint simulator seeds")
            seeds.add(seed)
            if cohort["split"] != role or cohort["num_envs"] != len(cohort["slots"]):
                raise ValueError("BC cohort role/lane count differs")
            for slot in cohort["slots"]:
                if slot["motion_id"] in seen or type(slot["target_chunks"]) is not int or slot["target_chunks"] <= 0:
                    raise ValueError("BC motion duplicated or missing quota")
                seen[slot["motion_id"]] = slot["motion_sha256"]
                count += slot["target_chunks"]
        if seen != modes or count != partition["target_chunks"]:
            raise ValueError("all BC roles must cover the identical pinned motion population")
    if explicit_native_test:
        contract = validate_artifact_contract(sampling_plan, motion_bindings=modes, forbidden_seeds=seeds)
        seeds.update(contract["closed_loop_test"]["simulator_seeds"])
        return rows, modes, seeds, inventory_sha
    tests = sampling_plan["success_test"]["cases"]
    if sampling_plan["success_test"]["episode_count"] != len(tests):
        raise ValueError("success case count differs")
    test_support, case_ids = {}, set()
    for case in tests:
        if (case["motion_id"] not in modes or case["motion_sha256"] != modes[case["motion_id"]]
                or case["case_id"] in case_ids or type(case["process_seed"]) is not int
                or not 0 <= case["process_seed"] < 2**31 or case["process_seed"] in seeds):
            raise ValueError("success cases require pinned motions and disjoint case seeds")
        test_support[case["motion_id"]] = case["motion_sha256"]
        case_ids.add(case["case_id"])
        seeds.add(case["process_seed"])
    if test_support != modes:
        raise ValueError("success panel must cover the same motion population")
    return rows, modes, seeds, inventory_sha


def build_schedule(inventory, sampling_plan, *, run_prefix, seed, num_envs,
                   windows_per_training_cohort, motion_root, physical_gpus=(0, 1)):
    rows, modes, used_seeds, inventory_sha = _validate_inputs(inventory, sampling_plan)
    motion_root = Path(motion_root)
    if (not _identifier(run_prefix) or type(seed) is not int or not 0 <= seed < 2**31
            or type(num_envs) is not int or not 1 <= num_envs <= 40
            or type(windows_per_training_cohort) is not int or not 1 <= windows_per_training_cohort <= 10000
            or not motion_root.is_absolute() or ".." in motion_root.parts
            or not isinstance(physical_gpus, (tuple, list)) or not physical_gpus
            or any(type(g) is not int or g not in (0, 1) for g in physical_gpus)
            or len(set(physical_gpus)) != len(physical_gpus)):
        raise ValueError("invalid explicit bounded cohort scheduling parameters")

    def independent_seed(role, round_id, cohort_index):
        candidate = int.from_bytes(hashlib.sha256(canonical([SCHEMA, seed, role, round_id, cohort_index])).digest()[:8], "little") % (2**31 - 1)
        if candidate in used_seeds:
            raise ValueError("cohort seed collision; choose a different schedule seed")
        used_seeds.add(candidate)
        return candidate

    def assignment(run_id, cohort_id, purpose, process_seed, selected):
        return validate_motion_assignment({"schema": ASSIGNMENT_SCHEMA, "run_id": run_id,
            "cohort_id": cohort_id, "purpose": purpose, "simulator_seed": process_seed,
            "motion_inventory_sha256": inventory_sha,
            "lanes": [{"lane": i, "motion_id": row["motion_id"],
                       "sha256": row["sha256"], "path": str(motion_root / row["filename"])}
                      for i, row in enumerate(selected)]})

    rounds = []
    for round_id, beta in enumerate(BETAS, 1):
        groups = {}
        for purpose in ("training", "validation"):
            ordered = sorted(rows, key=lambda r: hashlib.sha256(canonical([seed, purpose, round_id, r["motion_id"]])).digest())
            cohorts = []
            for i, offset in enumerate(range(0, len(ordered), num_envs)):
                selected = ordered[offset:offset + num_envs]
                cid = f"{purpose}-r{round_id}-{i:04d}"
                run_id = f"{run_prefix}-r{round_id}" + ("-heldout" if purpose == "validation" else "")
                if purpose == "training" and len(ordered) > num_envs:
                    # Different B/actor sets cannot reuse one immutable native
                    # coordinator registration. They are explicit child runs,
                    # linked by student+optimizer state at lifecycle boundaries.
                    run_id += f"-cohort-{i:04d}"
                native = assignment(run_id, cid, purpose, independent_seed(purpose, round_id, i), selected)
                windows = windows_per_training_cohort if purpose == "training" else 1
                cohorts.append({"cohort_id": cid, "physical_gpu_index": physical_gpus[i % len(physical_gpus)],
                    "num_envs": len(selected), "assignment": native,
                    "assignment_semantic_sha256": hashlib.sha256(canonical(native)).hexdigest(),
                    "continuous_windows": windows, "vector_steps": windows * 81,
                    "expected_transitions": len(selected) * windows * 81,
                    "optimizer_steps": windows if purpose == "training" else 0,
                    "reset_at_window_boundary": False, "lifecycle_rule": LIFECYCLE_RULE})
                if purpose == "training" and len(ordered) > num_envs:
                    cohorts[-1]["initial_checkpoint_binding"] = (
                        "previous_completed_cohort_student_and_optimizer" if i else
                        "BC_selected" if round_id == 1 else f"round_{round_id-1}_selected")
            groups[purpose] = cohorts
        rounds.append({"round_id": round_id, "teacher_beta": beta,
            "student_probability": 1 - beta, "training": groups["training"],
            "heldout": groups["validation"],
            "planned_optimizer_steps": sum(c["optimizer_steps"] for c in groups["training"]),
            "planned_training_transitions": sum(c["expected_transitions"] for c in groups["training"]),
            "heldout_checkpoint_binding": "prior_round_selected_checkpoint; immutable_before_training",
            "initial_checkpoint_binding": "BC_selected" if round_id == 1 else f"round_{round_id-1}_selected",
            "cohort_checkpoint_transfer": "carry_current_student_and_optimizer; never_restart_from_BC",
            "optimizer_execution": "sequential_cohorts_with_checkpoint_barrier; no_implicit_concurrent_learners"})

    if sampling_plan["schema"] == SAMPLING_SCHEMA_V3:
        closed = sampling_plan["artifact_roles"]["closed_loop_test"]
        with Path(closed["panel"]["path"]).open("rb") as stream:
            raw = stream.read((32 << 20) + 1)
        if len(raw) > 32 << 20 or hashlib.sha256(raw).hexdigest() != closed["panel"]["raw_sha256"]:
            raise ValueError("native success panel changed while planning")
        panel = json.loads(raw)
        if hashlib.sha256(canonical(panel)).hexdigest() != closed["panel_canonical_sha256"]:
            raise ValueError("native success panel semantic identity changed")
        success_test = {
            "schema": "roborender.matched-motion-success-plan.v2",
            "reuse_exact_cases_after_BC_and_each_selected_round": True,
            "completed_episodes": panel["episode_count"], "optimizer_steps": 0,
            "panel_reference": dict(closed["panel"]),
            "panel_canonical_sha256": closed["panel_canonical_sha256"],
            "seed_semantics": "one_native_process_seed_per_declared_cohort",
            "cohorts": [{**cohort, "physical_gpu_index": physical_gpus[i % len(physical_gpus)]}
                        for i, cohort in enumerate(panel["cohorts"])],
            "native_executor_qualified": False,
        }
    else:
        by_id = {row["motion_id"]: row for row in rows}
        success = []
        for i, case in enumerate(sampling_plan["success_test"]["cases"]):
            native = assignment(f"{run_prefix}-success", case["case_id"], "test", case["process_seed"], [by_id[case["motion_id"]]])
            success.append({**case, "physical_gpu_index": physical_gpus[i % len(physical_gpus)],
                "num_envs": 1, "completed_episodes": 1, "assignment": native,
                "assignment_semantic_sha256": hashlib.sha256(canonical(native)).hexdigest(),
                "reason_for_single_lane": "existing fixed panel has an independent process seed per case"})
        success_test = {"reuse_exact_cases_after_BC_and_each_selected_round": True,
                       "completed_episodes": len(success), "optimizer_steps": 0, "cases": success}
    return {"schema": SCHEMA, "plan_only": True, "production_admitted": False,
        "sampling_plan_sha256": hashlib.sha256(canonical(sampling_plan)).hexdigest(),
        "motion_inventory_sha256": inventory_sha, "motion_support": modes,
        "frames_per_window": 81, "num_envs_ceiling": num_envs,
        "windows_per_training_cohort": windows_per_training_cohort,
        "bc_partition_chunks": {role: p["target_chunks"] for role, p in sampling_plan["partitions"].items()},
        "rounds": rounds, "success_test": success_test,
        "scheduling_extension": "finite independent native trainer lifecycles over bounded motion cohorts; not Raven's single fixed scene",
        "native_boundary_evidence": "pinned Raven ppo_trainer.py:1361 reset_all precedes batch loop at1376",
        "activation_requirements": ["native explicit scene/reset/seed attestation per cohort",
            "qualified B and host/GPU memory", "checkpoint plus optimizer continuity across cohorts",
            "no in-flight prepare/step/commit at lifecycle retirement",
            "fixed independent held-out collection sealed before optimization",
            "explicit success evaluator cohort support; no legacy B5 relabeling"],
        "coverage_claim": "matched motion population with independent episode/seed cohorts; not unseen-motion generalization"}


def build_grouped_success_panel(inventory, sampling_plan, *, panel_id, seed, num_envs,
                                motion_root, excluded_seeds=()):
    """Create a NEW test panel, never reinterpret independent-seed old cases.

    One declared process seed drives each ordered B-lane cohort. Every motion
    gets one completed episode; this is explicitly not the older 320-case
    independently seeded panel. Runtime must use the explicit-panel evaluator.
    """
    if sampling_plan.get("schema") == SAMPLING_SCHEMA_V3:
        raise ValueError("v3 already pins its native success panel; do not replace it during DAgger planning")
    rows, _, used_seeds, inventory_sha = _validate_inputs(inventory, sampling_plan)
    root = Path(motion_root)
    if (not _identifier(panel_id) or type(seed) is not int or not 0 <= seed < 2**31
            or type(num_envs) is not int or not 1 <= num_envs <= 40
            or not root.is_absolute() or ".." in root.parts
            or any(type(s) is not int or not 0 <= s < 2**31 for s in excluded_seeds)):
        raise ValueError("invalid grouped-success panel parameters")
    used_seeds.update(excluded_seeds)
    cohorts = []
    ordered = sorted(rows, key=lambda r: hashlib.sha256(canonical([seed, "grouped-success", r["motion_id"]])).digest())
    for i, offset in enumerate(range(0, len(ordered), num_envs)):
        selected = ordered[offset:offset + num_envs]
        process_seed = int.from_bytes(hashlib.sha256(canonical([
            "roborender.explicit-motion-success-panel.v1", seed, panel_id, i])).digest()[:8], "little") % (2**31 - 1)
        if process_seed in used_seeds:
            raise ValueError("grouped-success seed collides with another role; choose a different panel seed")
        used_seeds.add(process_seed)
        native = validate_motion_assignment({"schema": ASSIGNMENT_SCHEMA, "run_id": panel_id,
            "cohort_id": f"success-{i:04d}", "purpose": "test", "simulator_seed": process_seed,
            "motion_inventory_sha256": inventory_sha,
            "lanes": [{"lane": lane, "motion_id": row["motion_id"], "sha256": row["sha256"],
                       "path": str(root / row["filename"])} for lane, row in enumerate(selected)]})
        cohorts.append({"assignment": native,
            "case_ids": [f"{panel_id}-case-{offset + lane:04d}" for lane in range(len(selected))]})
    return {"schema": "roborender.explicit-motion-success-panel.v1", "panel_id": panel_id,
        "purpose": "test", "motion_inventory_sha256": inventory_sha,
        "observation_mode": "raytraced_3view", "action_mode": "deterministic_action_mean_11_plus_homie_15",
        "metric": "native_goal_or_until_done", "max_vector_steps_per_cohort": 450,
        "episode_count": len(rows), "cohorts": cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sampling-plan", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--windows-per-training-cohort", type=int, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-grouped-success-panel", type=Path,
                        help="optional NEW grouped-seed panel; does not replace the sampling plan's independent-seed cases")
    args = parser.parse_args()
    inventory, plan = json.loads(args.inventory.read_bytes()), json.loads(args.sampling_plan.read_bytes())
    result = build_schedule(inventory, plan,
        run_prefix=args.run_prefix, seed=args.seed, num_envs=args.num_envs,
        windows_per_training_cohort=args.windows_per_training_cohort, motion_root=args.motion_root,
        physical_gpus=args.physical_gpus)
    panel = None
    if args.new_grouped_success_panel is not None:
        if args.new_grouped_success_panel.exists() or args.new_grouped_success_panel == args.output:
            raise FileExistsError("new grouped panel requires its own unused output")
        excluded = [c["assignment"]["simulator_seed"] for r in result["rounds"]
                    for role in ("training", "heldout") for c in r[role]]
        panel = build_grouped_success_panel(inventory, plan, panel_id=args.run_prefix + "-grouped-test",
            seed=args.seed, num_envs=args.num_envs, motion_root=args.motion_root, excluded_seeds=excluded)
    raw = canonical(result) + b"\n"
    with args.output.open("xb") as stream:
        stream.write(raw)
    if panel is not None:
        with args.new_grouped_success_panel.open("xb") as stream:
            stream.write(canonical(panel) + b"\n")
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "plan_only": True, "production_admitted": False}))


if __name__ == "__main__":
    main()
