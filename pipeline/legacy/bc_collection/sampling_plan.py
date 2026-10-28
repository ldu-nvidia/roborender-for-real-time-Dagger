"""Freeze in-distribution motion coverage and disjoint role/seed cohorts.

This is the explicitly requested sampling extension, not a claim Raven used
these splits. It changes only dataset scheduling, never native task/reset
physics. A native collector must attest the same ordered motion files for
both reset sampling and scene geometry before it may execute a cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from copy import deepcopy

from pipeline.legacy.bc_collection.artifact_roles import (
    PLAN_SCHEMA as SCHEMA_V3, CONTRACT_SCHEMA, OFFLINE_ROLES, offline_roles,
    panel_reference, validate_artifact_contract,
)

SCHEMA = "roborender.coverage-sampling-plan.v1"
SCHEMA_V2 = "roborender.coverage-sampling-plan.v2"
ROLES = ("train", "validation", "test")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def motion_inventory(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    paths = sorted(directory.glob("*.npz"), key=lambda p: p.name)
    if not paths or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("motion inventory requires regular NPZ files")
    rows = [{"motion_id": p.stem, "filename": p.name, "bytes": p.stat().st_size,
             "sha256": hash_file(p)} for p in paths]
    return {"schema": "roborender.motion-inventory.v1", "motion_count": len(rows),
            "ordering": "explicit_filename_lexical_not_filesystem_order",
            "motions": rows, "motion_records_sha256": hashlib.sha256(canonical(rows)).hexdigest()}


def preselected_ordinals(seed, role, motion_id, quota, span):
    if (type(quota) is not int or type(span) is not int or not 1 <= quota <= span
            or role not in ROLES):
        raise ValueError("invalid selected quota/exposure span")
    if role == "train":
        if quota != span:
            raise ValueError("training exposure span must equal its complete-chunk quota")
        return list(range(span))
    ranked = sorted(range(span), key=lambda ordinal: hashlib.sha256(canonical(
        [SCHEMA_V2, seed, role, motion_id, "complete_chunk_ordinal", ordinal])).digest())
    return sorted(ranked[:quota])


def validate_selection_plan(plan):
    """Reject cherry-picked ordinals or role exposure drift in the v2 recipe."""
    if plan.get("schema") == SCHEMA:
        return
    if (plan.get("schema") not in {SCHEMA_V2, SCHEMA_V3}
            or plan.get("selection_policy") != "preselected_complete_chunk_ordinals_over_training_exposure_v2"
            or plan.get("selection_uses_labels_rewards_or_loss") is not False):
        raise ValueError("explicit v2 predeclared ordinal selection policy required")
    if plan["schema"] == SCHEMA_V3:
        if set(plan["partitions"]) != set(OFFLINE_ROLES):
            raise ValueError("v3 has only train/validation offline artifacts")
        motions = {s["motion_id"]: s["motion_sha256"]
                   for c in plan["partitions"]["train"]["cohorts"] for s in c["slots"]}
        seeds = [c["process_seed"] for p in plan["partitions"].values() for c in p["cohorts"]]
        contract = validate_artifact_contract(plan, motion_bindings=motions, forbidden_seeds=seeds)
        if (contract["closed_loop_test"]["motion_inventory_sha256"] != plan["motion_inventory_sha256"]
                or contract["offline_chunk_counts"] != {r: p["target_chunks"] for r, p in plan["partitions"].items()}
                or sum(contract["offline_chunk_counts"].values()) != plan["total_target_chunks"]):
            raise ValueError("v3 artifact population/quotas differ")
    spans = {s["motion_id"]: s["target_chunks"]
             for c in plan["partitions"]["train"]["cohorts"] for s in c["slots"]}
    for role, partition in plan["partitions"].items():
        if role not in ROLES:
            raise ValueError("unknown sampling partition")
        for cohort in partition["cohorts"]:
            stops = []
            for slot in cohort["slots"]:
                span = spans.get(slot["motion_id"])
                if span is None or slot.get("exposure_span_chunks") != span:
                    raise ValueError("all roles must share the same per-motion training exposure span")
                expected = preselected_ordinals(plan["seed"], role, slot["motion_id"], slot["target_chunks"], span)
                if slot.get("selected_chunk_ordinals") != expected:
                    raise ValueError("selected chunk ordinals differ from the predeclared seed-only recipe")
                stops.append(expected[-1] + 1)
            if cohort.get("minimum_complete_chunk_windows") != max(stops):
                raise ValueError("cohort complete-window cost differs from its selected ordinals")


def build_plan(inventory: dict, *, seed: int, num_envs: int,
               chunks: int = 3200, success_cases: int = 320,
               selection_policy: str = "prefix_v1") -> dict:
    if selection_policy not in {"prefix_v1", "matched_exposure_v2"}:
        raise ValueError("unknown native complete-chunk selection policy")
    schema = SCHEMA_V2 if selection_policy == "matched_exposure_v2" else SCHEMA
    rows = inventory["motions"]
    if (inventory.get("schema") != "roborender.motion-inventory.v1"
            or inventory.get("motion_count") != len(rows)
            or hashlib.sha256(canonical(rows)).hexdigest() != inventory.get("motion_records_sha256")
            or not rows or len({r["motion_id"] for r in rows}) != len(rows)
            or len({r["sha256"] for r in rows}) != len(rows)):
        raise ValueError("motion inventory identity, count, or uniqueness differs")
    if (type(seed) is not int or not 0 <= seed < 2**31 or type(num_envs) is not int
            or not 1 <= num_envs <= 40 or type(chunks) is not int or chunks % 10
            or type(success_cases) is not int or success_cases < len(rows)):
        raise ValueError("invalid bounded sampling parameters")
    counts = {"train": chunks * 8 // 10, "validation": chunks // 10, "test": chunks // 10}
    if counts["train"] % 16 or min(counts.values()) < len(rows):
        raise ValueError("require full 16-chunk BC batches and every motion in every partition")
    used_seeds = set()

    def independent_seed(role, index):
        value = int.from_bytes(hashlib.sha256(canonical([schema, seed, role, index])).digest()[:8], "little") % (2**31 - 1)
        if value in used_seeds:
            raise ValueError("role/cohort seed collision; choose a different plan seed")
        used_seeds.add(value)
        return value

    partitions = {}
    for role in ROLES:
        ordered = sorted(rows, key=lambda r: hashlib.sha256(canonical([seed, role, r["motion_id"]])).digest())
        base, extra = divmod(counts[role], len(rows))
        slots = [{"motion_id": r["motion_id"], "motion_sha256": r["sha256"],
                  "filename": r["filename"], "target_chunks": base + int(i < extra)}
                 for i, r in enumerate(ordered)]
        cohorts = []
        for index, offset in enumerate(range(0, len(slots), num_envs)):
            selected = slots[offset:offset + num_envs]
            cohorts.append({"cohort_id": f"{role}-{index:04d}", "split": role,
                "process_seed": independent_seed(role, index), "num_envs": len(selected),
                "slots": selected, "target_chunks": sum(x["target_chunks"] for x in selected),
                "physics_gpu_lane_assignment": "scheduler_bound_before_launch",
                "episode_reset_rule": "native_only_no_reset_at_81_boundary"})
        partitions[role] = {"target_chunks": counts[role], "motion_count": len(rows), "cohorts": cohorts}

    ordered = sorted(rows, key=lambda r: hashlib.sha256(canonical([seed, "success_test", r["motion_id"]])).digest())
    tests = [{"case_id": f"success-test-{i:04d}",
              "motion_id": ordered[i % len(rows)]["motion_id"],
              "motion_sha256": ordered[i % len(rows)]["sha256"],
              "process_seed": independent_seed("success_test", i)} for i in range(success_cases)]
    result = {"schema": schema, "seed": seed, "frames_per_chunk": 81,
            "motion_inventory_sha256": hashlib.sha256(canonical(inventory)).hexdigest(),
            "num_envs_ceiling": num_envs, "gpu_capacity_qualified": False,
            "total_target_chunks": chunks, "partitions": partitions,
            "success_test": {"purpose": "test_only_not_hyperparameter_selection",
                             "episode_count": success_cases, "cases": tests,
                             "observation_mode": "raytraced_3view"},
            "sampling_interpretation": "same motion population, disjoint seed/episode cohorts; not unseen-motion OOD",
            "native_assignment_attestation_required": True,
            "production_admitted": False}
    if selection_policy == "matched_exposure_v2":
        spans = {slot["motion_id"]: slot["target_chunks"]
                 for cohort in partitions["train"]["cohorts"] for slot in cohort["slots"]}
        emitted_minimum = windows_minimum = 0
        for role, partition in partitions.items():
            for cohort in partition["cohorts"]:
                stops = []
                for slot in cohort["slots"]:
                    span = spans[slot["motion_id"]]
                    if slot["target_chunks"] > span:
                        raise ValueError("held-out selected quota exceeds the training exposure span")
                    selected = preselected_ordinals(seed, role, slot["motion_id"], slot["target_chunks"], span)
                    slot.update(selected_chunk_ordinals=selected, exposure_span_chunks=span)
                    stops.append(selected[-1] + 1)
                cohort["minimum_complete_chunk_windows"] = max(stops)
                emitted_minimum += sum(stops)
                windows_minimum += max(stops)
        result.update(selection_policy="preselected_complete_chunk_ordinals_over_training_exposure_v2",
            selection_uses_labels_rewards_or_loss=False,
            ordinal_semantics="zero-based completed native chunk count per lane, across actual episodes; not seq or task stage",
            collection_cost={"selected_chunks": chunks,
                "minimum_native_complete_chunks_emitted": emitted_minimum,
                "minimum_unselected_complete_chunks_retained": emitted_minimum - chunks,
                "minimum_vector_windows_across_cohorts": windows_minimum,
                "minimum_vector_steps_across_cohorts": windows_minimum * 81,
                "caveat": "lower bounds; native done-tail discard and synchronization can require additional complete chunks/steps"},
            coverage_limit="matched ordinal exposure support, not guaranteed equal task-stage/episode-phase frequency")
        validate_selection_plan(result)
    return result


def build_offline_plan(inventory, *, seed, num_envs, success_panel,
                       train_chunks=2560, validation_chunks=320):
    """Opt-in successor: reuse exact matched-exposure recipe, never raw-test RGB."""
    if (type(train_chunks) is not int or type(validation_chunks) is not int
            or train_chunks != validation_chunks * 8 or train_chunks % 16):
        raise ValueError("explicit offline quotas require full B16 and an 8:1 ratio")
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    support = {r["motion_id"]: r["sha256"] for r in inventory["motions"]}
    external = panel_reference(success_panel, inventory_sha256=inventory_sha, motion_bindings=support)
    # Retain v2's documented seed/order/ordinal recipe, not any v2 dataset bytes.
    parent = build_plan(inventory, seed=seed, num_envs=num_envs, chunks=validation_chunks * 10,
                        success_cases=external["episode_count"], selection_policy="matched_exposure_v2")
    result = deepcopy(parent)
    result["schema"] = SCHEMA_V3
    result["partitions"] = {role: result["partitions"][role] for role in OFFLINE_ROLES}
    result.pop("success_test")
    result["total_target_chunks"] = train_chunks + validation_chunks
    result["artifact_roles"] = dict(schema=CONTRACT_SCHEMA, offline_roles=list(OFFLINE_ROLES),
        offline_chunk_counts=dict(train=train_chunks, validation=validation_chunks), offline_test_chunks=0,
        closed_loop_test=external)
    result["seed_order_ordinal_recipe"] = "coverage-sampling-plan.v2; retain train/validation recipes only"
    cohorts = [c for p in result["partitions"].values() for c in p["cohorts"]]
    emitted = sum(s["selected_chunk_ordinals"][-1] + 1 for c in cohorts for s in c["slots"])
    windows = sum(c["minimum_complete_chunk_windows"] for c in cohorts)
    result["collection_cost"].update(selected_chunks=result["total_target_chunks"],
        minimum_native_complete_chunks_emitted=emitted,
        minimum_unselected_complete_chunks_retained=emitted-result["total_target_chunks"],
        minimum_vector_windows_across_cohorts=windows, minimum_vector_steps_across_cohorts=81*windows)
    validate_selection_plan(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    scan = sub.add_parser("inventory")
    scan.add_argument("--directory", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--inventory", type=Path, required=True)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--num-envs", type=int, required=True)
    plan.add_argument("--chunks", type=int, default=3200)
    plan.add_argument("--success-cases", type=int, default=320)
    plan.add_argument("--selection-policy", choices=("prefix_v1", "matched_exposure_v2"), default="prefix_v1")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--closed-loop-test-panel", type=Path)
    plan.add_argument("--closed-loop-test-panel-sha256")
    plan.add_argument("--offline-train-chunks", type=int, default=2560)
    plan.add_argument("--offline-validation-chunks", type=int, default=320)
    args = parser.parse_args()
    if args.action == "inventory":
        result = motion_inventory(args.directory)
    elif args.closed_loop_test_panel is not None or args.closed_loop_test_panel_sha256 is not None:
        if args.closed_loop_test_panel is None or args.closed_loop_test_panel_sha256 is None:
            raise ValueError("v3 requires both external success-panel file and raw SHA")
        result = build_offline_plan(json.loads(args.inventory.read_bytes()), seed=args.seed, num_envs=args.num_envs,
            success_panel={"path": str(args.closed_loop_test_panel.resolve()), "raw_sha256": args.closed_loop_test_panel_sha256},
            train_chunks=args.offline_train_chunks, validation_chunks=args.offline_validation_chunks)
    else:
        result = build_plan(json.loads(args.inventory.read_bytes()), seed=args.seed, num_envs=args.num_envs,
            chunks=args.chunks, success_cases=args.success_cases, selection_policy=args.selection_policy)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"path": str(args.output), "sha256": hash_file(args.output),
                      "schema": result["schema"], "production_admitted": False}))


if __name__ == "__main__":
    main()
