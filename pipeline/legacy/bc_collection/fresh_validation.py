"""Plan fresh held-out BC collection while reusing a legacy training population.

This explicit successor does not pretend the old 2,000 demonstrations have
recoverable motion/reset provenance. Only 320 new validation chunks are
scheduled. The original native request.v2, writer and guarded launcher remain
unchanged; final native success testing is a separate frozen artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

from pipeline.legacy.bc_collection.artifact_roles import panel_reference
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    SCHEMA_V2 as REQUEST_SCHEMA, canonical, publish_json, validate_collection_request,
)

SCHEMA = "roborender.bc-fresh-validation-plan.v1"
SELECTION_POLICY = "predeclared_validation_complete_chunk_ordinals_v1"
METADATA_LIMIT = 16 << 20


def read_binding(reference):
    """Read one bounded, raw-hash-pinned metadata object; never an NPZ."""
    if (not isinstance(reference, dict) or set(reference) != {"path", "raw_sha256"}
            or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()
            or not isinstance(reference["raw_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["raw_sha256"])):
        raise ValueError("absolute metadata path and raw SHA binding required")
    path = Path(reference["path"])
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > METADATA_LIMIT:
        raise ValueError("metadata input exceeds bounded file limit")
    with path.open("rb") as stream:
        raw = stream.read(METADATA_LIMIT + 1)
    after = path.stat()
    if (len(raw) > METADATA_LIMIT or hashlib.sha256(raw).hexdigest() != reference["raw_sha256"]
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("pinned metadata changed")
    return json.loads(raw)


def binding(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > METADATA_LIMIT:
        raise ValueError("metadata input exceeds bounded file limit")
    return {"path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _inventory(reference):
    value = read_binding(reference)
    rows = value.get("motions", [])
    if (value.get("schema") != "roborender.motion-inventory.v1"
            or value.get("motion_count") != 295 or len(rows) != 295
            or hashlib.sha256(canonical(rows)).hexdigest() != value.get("motion_records_sha256")
            or len({r.get("motion_id") for r in rows}) != 295
            or len({r.get("sha256") for r in rows}) != 295):
        raise ValueError("complete 295-motion inventory and exact records hash required")
    for row in rows:
        if (not isinstance(row.get("motion_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", row["motion_id"])
                or row.get("filename") != row["motion_id"] + ".npz"
                or not isinstance(row.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or type(row.get("bytes")) is not int or row["bytes"] <= 0):
            raise ValueError("invalid motion metadata descriptor")
    return value


def _legacy(reference):
    value = read_binding(reference)
    rows = value.get("chunks", [])
    if (value.get("chunk_count") != 2000 or len(rows) != 2000
            or len({r.get("chunk_id") for r in rows}) != 2000
            or len({r.get("source_sha256") for r in rows}) != 2000
            or any(r.get("frames") != 81 for r in rows)):
        raise ValueError("exact existing 2000 independent source identities required")
    known, unknown = set(), 0
    for row in rows:
        value = row.get("simulator_seed")
        if value is None:
            unknown += 1
        elif type(value) is not int or not 0 <= value < 2**31:
            raise ValueError("invalid recorded legacy simulator seed")
        else:
            known.add(value)
        if (not isinstance(row.get("source_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])):
            raise ValueError("invalid legacy source hash")
    return dict(chunk_count=2000, known_process_seeds=sorted(known), unknown_seed_chunks=unknown,
                motion_id_present=sum(bool(r.get("motion_id")) for r in rows),
                episode_id_present=sum(bool(r.get("episode_id")) for r in rows),
                new_training_collection=False, historical_training_coverage_certified=False)


def selected_ordinals(seed, motion_id, quota, span):
    ranked = sorted(range(span), key=lambda ordinal: hashlib.sha256(canonical(
        [SCHEMA, seed, "validation", motion_id, "complete_chunk_ordinal", ordinal])).digest())
    return sorted(ranked[:quota])


def build_plan(*, motion_inventory, legacy_manifest, closed_loop_test_panel,
               seed, num_envs=6, chunks=320, exposure_span_chunks=9,
               excluded_process_seeds=()):
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(num_envs) is not int or num_envs not in (1, 6)
            or chunks != 320 or type(chunks) is not int
            or type(exposure_span_chunks) is not int or exposure_span_chunks != 9
            or any(type(s) is not int or not 0 <= s < 2**31 for s in excluded_process_seeds)):
        raise ValueError("fresh validation requires 320 chunks, nine-complete-chunk exposure and explicit B1/B6")
    inventory = _inventory(motion_inventory)
    legacy = _legacy(legacy_manifest)
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    support = {r["motion_id"]: r["sha256"] for r in inventory["motions"]}
    test = panel_reference(closed_loop_test_panel, inventory_sha256=inventory_sha,
                           motion_bindings=support, forbidden_seeds=legacy["known_process_seeds"])
    if test["episode_count"] != 295 or any(n != 1 for n in test["motion_case_counts"].values()):
        raise ValueError("separate final test must have exactly one native case for each of 295 motions")
    forbidden = set(legacy["known_process_seeds"]) | set(test["simulator_seeds"]) | set(excluded_process_seeds)
    ordered = sorted(inventory["motions"], key=lambda row: hashlib.sha256(canonical(
        [SCHEMA, seed, "validation", row["motion_id"]])).digest())
    slots = []
    for index, row in enumerate(ordered):
        quota = 1 + int(index < 25)
        slots.append(dict(motion_id=row["motion_id"], motion_sha256=row["sha256"],
            filename=row["filename"], target_chunks=quota, exposure_span_chunks=exposure_span_chunks,
            selected_chunk_ordinals=selected_ordinals(seed, row["motion_id"], quota, exposure_span_chunks)))
    cohorts, seeds = [], set()
    for index, start in enumerate(range(0, len(slots), num_envs)):
        value = int.from_bytes(hashlib.sha256(canonical([SCHEMA, seed, "process", index])).digest()[:8],
                              "little") % (2**31 - 1)
        if value in forbidden or value in seeds:
            raise ValueError("declared process seed collision; choose a new master seed")
        seeds.add(value)
        selected = slots[start:start + num_envs]
        cohorts.append(dict(cohort_id=f"validation-{index:04d}", split="validation", process_seed=value,
            num_envs=len(selected), slots=selected, target_chunks=sum(s["target_chunks"] for s in selected),
            minimum_complete_chunk_windows=max(s["selected_chunk_ordinals"][-1] + 1 for s in selected),
            physics_gpu_lane_assignment="scheduler_bound_before_launch",
            episode_reset_rule="native_only_no_reset_at_81_boundary"))
    emitted = sum(s["selected_chunk_ordinals"][-1] + 1 for s in slots)
    windows = sum(c["minimum_complete_chunk_windows"] for c in cohorts)
    return dict(schema=SCHEMA, collection_purpose="fresh_validation_only", seed=seed,
        motion_inventory=motion_inventory, motion_inventory_sha256=inventory_sha,
        legacy_manifest=legacy_manifest, legacy_training=legacy,
        closed_loop_test_panel=closed_loop_test_panel, closed_loop_test=test,
        explicitly_excluded_process_seeds=sorted(set(excluded_process_seeds)),
        num_envs_ceiling=num_envs, frames_per_chunk=81, total_target_chunks=320,
        new_training_chunks=0, offline_test_chunks=0,
        exposure_span_chunks=exposure_span_chunks, selection_policy=SELECTION_POLICY,
        selection_uses_labels_rewards_or_loss=False,
        ordinal_semantics="zero-based complete native emissions per lane across actual episodes, not task phase",
        partitions={"validation": dict(target_chunks=320, motion_count=295, cohorts=cohorts)},
        collection_cost=dict(selected_chunks=320, minimum_native_complete_chunks_emitted=emitted,
            minimum_unselected_complete_chunks_retained=emitted-320,
            minimum_vector_windows_across_cohorts=windows, minimum_vector_steps_across_cohorts=81*windows,
            caveat="lower bounds; native done-tail discard and synchronization can require more steps/emissions"),
        sampling_interpretation="broad declared validation support, not proven matched historical training coverage or phase frequency",
        independence_limit="known process seeds are disjoint; missing legacy seeds/reset states prevent a proof of physical trajectory independence",
        native_assignment_attestation_required=True, production_admitted=False, gpu_capacity_qualified=False)


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("explicit fresh-validation-only plan required")
    try:
        rebuilt = build_plan(motion_inventory=plan["motion_inventory"], legacy_manifest=plan["legacy_manifest"],
            closed_loop_test_panel=plan["closed_loop_test_panel"], seed=plan["seed"],
            num_envs=plan["num_envs_ceiling"], chunks=plan["total_target_chunks"],
            exposure_span_chunks=plan["exposure_span_chunks"],
            excluded_process_seeds=plan["explicitly_excluded_process_seeds"])
    except KeyError as error:
        raise ValueError("fresh validation plan is missing its frozen inputs") from error
    if plan != rebuilt:
        raise ValueError("fresh validation plan differs from its predeclared deterministic recipe")
    return plan


def request_from_cohort(plan, cohort, *, run_id, motion_root, output_dir, rank,
                        global_env_offset, max_windows):
    """Management-only projection to the already qualified native request.v2."""
    if (plan.get("schema") != SCHEMA
            or cohort not in plan["partitions"]["validation"]["cohorts"]):
        raise ValueError("cohort is not an exact fresh validation plan member")
    return validate_collection_request(dict(schema=REQUEST_SCHEMA, run_id=run_id,
        cohort_id=cohort["cohort_id"], split="validation", process_seed=cohort["process_seed"],
        rank=rank, global_env_offset=global_env_offset,
        motions=[dict(motion_id=s["motion_id"], path=str(Path(motion_root)/s["filename"]),
                      sha256=s["motion_sha256"]) for s in cohort["slots"]],
        chunks_per_env=[s["target_chunks"] for s in cohort["slots"]],
        selected_chunk_ordinals_per_env=[s["selected_chunk_ordinals"] for s in cohort["slots"]],
        exposure_span_chunks_per_env=[s["exposure_span_chunks"] for s in cohort["slots"]],
        max_windows=max_windows, output_dir=str(output_dir),
        sampling_plan_sha256=hashlib.sha256(canonical(plan)).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-inventory", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--closed-loop-test-panel", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, choices=(1, 6), default=6)
    parser.add_argument("--exclude-process-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(motion_inventory=binding(args.motion_inventory), legacy_manifest=binding(args.legacy_manifest),
        closed_loop_test_panel=binding(args.closed_loop_test_panel), seed=args.seed, num_envs=args.num_envs,
        excluded_process_seeds=args.exclude_process_seeds)
    validate_plan(plan)
    publish_json(args.output, plan)
    print(json.dumps(dict(plan=binding(args.output), selected_validation_chunks=320,
                          new_training_chunks=0, execute=False, production_admitted=False), sort_keys=True))


if __name__ == "__main__":
    main()
