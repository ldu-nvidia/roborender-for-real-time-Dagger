"""Immutable explicit-motion schedules for validation-only native collection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical
from pipeline.dagger.services.contracts import identifier, sha256_digest, file_sha256
from pipeline.deployment.rtx_actor.native_motion_assignment import validate_motion_assignment

SCHEMA = "roborender.dagger-heldout-plan.v1"


def load_plan(path, raw_sha, config, test_seeds):
    path = Path(path)
    if not path.is_absolute() or file_sha256(path) != raw_sha or path.stat().st_size > 4 << 20:
        raise ValueError("held-out collection plan must be bounded, absolute and immutable")
    value = json.loads(path.read_bytes())
    keys = {"schema", "parent_run_id", "run_id", "round_id", "teacher_checkpoint_sha256",
            "policy_spec_sha256", "motion_inventory_sha256", "motion_support", "cohorts"}
    if (set(value) != keys or value["schema"] != SCHEMA or value["parent_run_id"] != config["run_id"]
            or value["round_id"] != config["round_id"] or not value["run_id"].startswith("heldout-")):
        raise ValueError("held-out collection schedule differs from target run/round")
    identifier(value["run_id"], "held-out run")
    for key in ("teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256"):
        sha256_digest(value[key], key)
    support = value["motion_support"]
    if not support or support != sorted(set(support)):
        raise ValueError("explicit sorted motion support required")
    seen, seeds, motions, actors = set(), set(), set(), []
    from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_MSE_SCALED_SCHEMA,environments_per_actor
    limit=min(16,environments_per_actor(config)) if config['schema']==LEO_MSE_SCALED_SCHEMA else 6
    reserved_seeds = {config["seed"] + i for i, _ in enumerate(config["rtx_actors"])} | set(test_seeds)
    for cohort in value["cohorts"]:
        if set(cohort) != {"cohort_id", "physical_gpu_index", "simulator_seed", "lanes"}:
            raise ValueError("invalid explicit native cohort schedule")
        if cohort["physical_gpu_index"] not in {0, 1} or type(cohort["physical_gpu_index"]) is not int:
            raise ValueError("held-out cohorts require an explicitly owned Leo GPU")
        assignment = assignment_for(value, cohort)
        if len(cohort["cohort_id"]) > 64 or not cohort["cohort_id"].replace("-", "").isalnum() or cohort["cohort_id"].lower() != cohort["cohort_id"]:
            raise ValueError("native phase cohort identifier must be short lowercase alphanumeric/hyphen")
        if (cohort["cohort_id"] in seen or cohort["simulator_seed"] in seeds | reserved_seeds
                or len(cohort["lanes"]) > limit):
            raise ValueError("cohort identity/seed is reused or exceeds the declared collection batch")
        seen.add(cohort["cohort_id"])
        seeds.add(cohort["simulator_seed"])
        motions.update(row["motion_id"] for row in assignment["lanes"])
        prefix = value["run_id"] + "-" + cohort["cohort_id"]
        for lane in range(len(cohort["lanes"])):
            actor = f"{prefix}-lane-{lane:03d}"
            identifier(actor, "held-out actor")
            actors.append(actor)
    if not seen or motions != set(support):
        raise ValueError("actual native cohort schedule must cover declared motion support")
    return value


def assignment_for(plan, cohort):
    return validate_motion_assignment({"schema": "roborender.native-motion-cohort.v1",
        "run_id": plan["run_id"], "cohort_id": cohort["cohort_id"], "purpose": "validation",
        "simulator_seed": cohort["simulator_seed"], "motion_inventory_sha256": plan["motion_inventory_sha256"],
        "lanes": cohort["lanes"]})


def registration_for(plan, config, plan_sha):
    from pipeline.dagger.services.validation_collection import validate
    actors = sorted(f"{plan['run_id']}-{cohort['cohort_id']}-lane-{lane:03d}"
        for cohort in plan["cohorts"] for lane in range(len(cohort["lanes"])))
    value = validate({"schema": "roborender.dagger-validation-collection.v1", "purpose": "validation",
        "run_id": plan["run_id"], "round_id": config["round_id"], "actor_ids": actors,
        "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
        "teacher_beta": config["teacher_beta"], "collection_plan_sha256": plan_sha,
        **{key: plan[key] for key in ("teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256")}})
    return {"registration": value, "registration_sha256": hashlib.sha256(canonical(value)).hexdigest()}
