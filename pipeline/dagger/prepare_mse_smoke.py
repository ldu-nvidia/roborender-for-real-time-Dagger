"""Stage an explicit two-motion, B2-per-RTX MSE smoke; never allocate GPUs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from pipeline.dagger.beta_campaign import make_first_round_config, immutable_json
from pipeline.dagger.beta_validation import panel_seeds
from pipeline.dagger.mse_campaign import NATIVE_PLAN_SCHEMA, validate_plan
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_MSE_SMOKE_SCHEMA, validate_config
from pipeline.dagger.services.contracts import file_sha256, sha256_digest
from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel
from pipeline.deployment.rtx_actor.native_motion_assignment import validate_motion_assignment


def prepare(root, *, run_id, seed, checkpoint, checkpoint_sha, panel_path,
            source_sha, teacher_sha, policy_spec_sha, max_iterations=3, corrected_success=False,
            rendered_observation_config=None):
    """Freeze topology/purpose/seed pins, not uncollected data or future parents."""
    root, panel_path = Path(root).resolve(), Path(panel_path).resolve()
    panel = validate_panel(json.loads(panel_path.read_bytes()))
    if corrected_success:
        from pipeline.deployment.rtx_actor.smoke_success_contract import require_panel_smoke
        require_panel_smoke(panel)
    if panel["purpose"] != "test" or not panel["panel_id"].startswith("explicit-"):
        raise ValueError("the smoke requires a separately named independent final test")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 3:
        raise ValueError("bounded diagnostic smoke accepts at most three updates per round")
    lanes = panel["cohorts"][0]["assignment"]["lanes"]
    support = {(r["motion_id"], r["sha256"], r["path"]) for r in lanes}
    if len(lanes) != 2 or any({(r["motion_id"], r["sha256"], r["path"])
            for r in c["assignment"]["lanes"]} != support for c in panel["cohorts"]):
        raise ValueError("the B4 smoke requires the same exact two-motion support in every test cohort")
    for value in (source_sha, teacher_sha, policy_spec_sha, checkpoint_sha):
        sha256_digest(value, "immutable asset")
    training_seeds = {seed + 100003 * (rid - 1) + gpu for rid in range(1, 5) for gpu in (0, 1)}
    reserved = training_seeds | set(panel_seeds(panel))
    if len(training_seeds) != 8 or training_seeds & set(panel_seeds(panel)):
        raise ValueError("training and independent test seeds overlap")
    plan = {"schema": NATIVE_PLAN_SCHEMA, "run_id": run_id, "rounds": []}
    next_seed = seed + 500000
    for rid in range(1, 5):
        directory = root / f"round-{rid}"
        pins = []
        for gpu in (0, 1):
            assignment = validate_motion_assignment({"schema": "roborender.native-motion-cohort.v1",
                "run_id": run_id, "cohort_id": f"training-r{rid}-g{gpu}", "purpose": "training",
                "simulator_seed": seed + 100003 * (rid - 1) + gpu,
                "motion_inventory_sha256": panel["motion_inventory_sha256"], "lanes": copy.deepcopy(lanes)})
            path = directory / f"training-g{gpu}.json"
            pins.append({"physical_gpu_index": gpu, "path": str(path), "sha256": immutable_json(path, assignment)})
        cohorts = []
        for gpu in (0, 1):
            while next_seed in reserved:
                next_seed += 1
            reserved.add(next_seed)
            cohorts.append({"cohort_id": f"validation-r{rid}-g{gpu}", "physical_gpu_index": gpu,
                            "simulator_seed": next_seed, "lanes": copy.deepcopy(lanes)})
            next_seed += 1
        collection = {"schema": "roborender.dagger-heldout-plan.v1", "parent_run_id": run_id,
            "run_id": f"heldout-{run_id}-r{rid}", "round_id": rid,
            "teacher_checkpoint_sha256": teacher_sha, "policy_spec_sha256": policy_spec_sha,
            "motion_inventory_sha256": panel["motion_inventory_sha256"],
            "motion_support": sorted(r["motion_id"] for r in lanes), "cohorts": cohorts}
        collection_path = directory / "collection-plan.json"
        collection_sha = immutable_json(collection_path, collection)
        plan["rounds"].append({"round_id": rid, "training_assignments": pins,
            "collection_plan_path": str(collection_path), "collection_plan_sha256": collection_sha,
            "validation_manifest_path": str(directory / "heldout-manifest.json"), "validation_manifest_sha256": None,
            "validation_admission_path": str(directory / "heldout-admission.json"), "validation_admission_sha256": None,
            "interval_iterations": 1, "max_iterations": max_iterations,
            "min_validation_epochs": min(2, max_iterations) if corrected_success else 1,
            "patience": 2, "relative_improvement": .005, "absolute_improvement": 1e-6})
    config = make_first_round_config(run_id, seed=seed, checkpoint=Path(checkpoint).resolve(),
        checkpoint_sha=checkpoint_sha, panel=panel_path, panel_sha=file_sha256(panel_path), source_sha=source_sha,
        gateway_pids=[], smoke=True, deployment_host="leo-rtx-6k")
    config["schema"] = LEO_MSE_SMOKE_SCHEMA
    if corrected_success:
        from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import LEO_MSE_SMOKE_CORRECTED_SCHEMA
        config["schema"] = LEO_MSE_SMOKE_CORRECTED_SCHEMA
    config["motion_assignments"] = plan["rounds"][0]["training_assignments"]
    for actor in config["rtx_actors"]:
        actor["concurrent_environments"] = 2
    from pipeline.deployment.rendered_success_dispatch import configuration
    configuration(rendered_observation_config, panel, checkpoint_sha)
    config["validation"].update(completed_episodes=panel["episode_count"], panel_protocol=panel["panel_id"],
        observation_mode='generated', rendered_observation_config=rendered_observation_config)
    validate_config(config)
    validate_plan(plan, config)
    return {"config_path": str(root / "first-round.json"), "config_sha256": immutable_json(root / "first-round.json", config),
            "plan_path": str(root / "mse-plan.json"), "plan_sha256": immutable_json(root / "mse-plan.json", plan),
            "training_environments": 4, "heldout_environments": 4, "rows_per_window": 324,
            "max_updates_per_round": max_iterations, "final_test_episodes": panel["episode_count"],
            "allocated_gpus": 0, "native_runtime_validated": False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "panel-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "checkpoint-sha", "source-sha", "teacher-sha", "policy-spec-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--corrected-success", action="store_true")
    parser.add_argument('--rendered-observation-config', type=Path, required=True)
    args=vars(parser.parse_args())
    from pipeline.bc_training.handoff_common import binding
    args['rendered_observation_config']=binding(args['rendered_observation_config'])
    print(json.dumps(prepare(**args), sort_keys=True))


if __name__ == "__main__":
    main()
