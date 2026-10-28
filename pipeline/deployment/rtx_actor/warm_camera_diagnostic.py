"""Bounded fresh-versus-warm camera experiment; never a production admission.

The two arms share the final motion assignment and seed. The warm arm first
uses eleven declared scene lifecycles, with three teacher steps in each. This
tests scene reuse, not replay of the failed mixed-policy production trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from native_motion_assignment import validate_motion_assignment

SCHEMA = "roborender.warm-camera-diagnostic.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_plan(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "inventory", "arms"}
            or value["schema"] != SCHEMA or set(value["arms"]) != {"0", "1"}):
        raise ValueError("exact fresh/warm camera diagnostic plan required")
    inventory = value["inventory"]
    rows = inventory.get("motions", [])
    if (inventory.get("schema") != "roborender.motion-inventory.v1"
            or inventory.get("motion_count") != 295 or len(rows) != 295
            or inventory.get("motion_records_sha256") != hashlib.sha256(canonical(rows)).hexdigest()):
        raise ValueError("diagnostic needs the pinned full motion inventory")
    identities = {row["motion_id"]: row for row in rows}
    if len(identities) != 295:
        raise ValueError("duplicate inventory identities")
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    for gpu, expected_count in (("0", 1), ("1", 12)):
        stages = value["arms"][gpu]
        if not isinstance(stages, list) or len(stages) != expected_count:
            raise ValueError("fresh arm needs one scene; warm arm needs twelve")
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) != {"assignment", "steps"}:
                raise ValueError("unknown diagnostic stage fields")
            assignment = validate_motion_assignment(stage["assignment"])
            expected_steps = 324 if index == len(stages)-1 else 3
            if (type(stage["steps"]) is not int or stage["steps"] != expected_steps
                    or len(assignment["lanes"]) != 16
                    or assignment["purpose"] not in {"training", "validation"}
                    or assignment["motion_inventory_sha256"] != inventory_sha):
                raise ValueError("diagnostic scene budget/purpose/inventory mismatch")
            for lane in assignment["lanes"]:
                reference = identities.get(lane["motion_id"], {})
                if (lane["sha256"] != reference.get("sha256")
                        or Path(lane["path"]).name != reference.get("filename")):
                    raise ValueError("diagnostic motion not in pinned inventory")
    if value["arms"]["0"][-1] != value["arms"]["1"][-1]:
        raise ValueError("fresh and warm final scenes must have identical assignment and seed")
    if value["arms"]["0"][-1]["assignment"]["purpose"] != "training":
        raise ValueError("final diagnostic scene must be a training cohort")
    return value


def read_plan(path, expected_sha256):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 1 << 20):
        raise ValueError("bounded regular diagnostic plan required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("diagnostic plan digest mismatch")
    return validate_plan(json.loads(raw))


def run_stages(stages, *, switch, execute, report):
    """One call per irreversible budget, no retries or exception swallowing."""
    results = []
    completed = 0
    for index, stage in enumerate(stages):
        report({"phase": "diagnostic_scene_start", "scene_index": index,
                "scene_count": len(stages), "completed_vector_steps": completed})
        switch(stage["assignment"])
        result = execute(stage["steps"], lambda row: report({**row,
            "scene_index": index, "scene_count": len(stages),
            "total_completed_vector_steps": completed + row.get("completed_vector_steps", 0)}))
        if (result.get("result") != "PASS"
                or result.get("environment_step_calls") != stage["steps"]
                or result.get("physical_transition_rows") != 16 * stage["steps"]
                or result.get("optimizer_steps") != 0 or result.get("chunks_published") != 0):
            raise RuntimeError("diagnostic scene did not finish its exact teacher-only budget")
        results.append(result)
        completed += stage["steps"]
        report({"phase": "diagnostic_scene_complete", "scene_index": index,
                "scene_count": len(stages), "completed_vector_steps": completed})
    return results


def run_native(trainer, output, report):
    # These imports are deliberately deferred until native Kit is initialized.
    import os
    from dagger_rtx.durable import atomic_write_json
    from dagger_rtx.native_entrypoints import NativeRavenRuntime, build_origin_attested_entrypoints
    from dagger_rtx.native_raven import build_native_raven_batch
    from leo_capacity_probe import run_teacher_capacity, run_teacher_camera_soak, step_acknowledgement
    from resident_isaac_trainer import ResidentRavenBackend, phase_environment, reset_rollout_for_scene

    digest = os.environ["ROBORENDER_WARM_CAMERA_PLAN_SHA256"]
    plan = read_plan(os.environ["ROBORENDER_WARM_CAMERA_PLAN"], digest)
    gpu = os.environ["ROBORENDER_WARM_CAMERA_GPU"]
    stages = plan["arms"][gpu]
    recorded_reference = None
    if os.environ.get("ROBORENDER_RECORDED_ACTION_REFERENCE"):
        from recorded_action_camera import read_reference
        recorded_reference = read_reference(os.environ["ROBORENDER_RECORDED_ACTION_REFERENCE"],
            os.environ["ROBORENDER_RECORDED_ACTION_SHA256"],
            hashlib.sha256(canonical(stages[-1]["assignment"])).hexdigest())
    inventory_path = output.parent / "diagnostic-inventory.json"
    atomic_write_json(inventory_path, plan["inventory"], mode=0o644)
    motion_root = str(Path(stages[-1]["assignment"]["lanes"][0]["path"]).parent)
    env = {"RESIDENT_MOTION_INVENTORY": str(inventory_path), "RESIDENT_PRODUCTION_FULL295": "1",
           "RESIDENT_MOTION_INVENTORY_SHA256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
           "RESIDENT_PRODUCTION_MOTION_ROOT": motion_root,
           "RESIDENT_INITIAL_SCENE_SEED": os.environ["ROBORENDER_WARM_CAMERA_BOOTSTRAP_SEED"]}
    with phase_environment(env):
        backend = ResidentRavenBackend(trainer, jobs_root=output.parent, runtime_root=output.parent,
                                       source_sha256=os.environ["ROBORENDER_WARM_CAMERA_SOURCE_SHA256"])
    trainer.env.simulator.sim.pause()

    def switch(assignment):
        backend._switch_scene("training", seed=assignment["simulator_seed"], panel=None,
            report=lambda event, **fields: report({"phase": event, **fields}), assignment=assignment)
        trainer.env.simulator.sim.play()
        for model in (trainer.policy_model, trainer.ref_model):
            reset_rollout_for_scene(model)
        backend._prepare_training(assignment["simulator_seed"])

    def execute(steps, stage_report):
        runtime = NativeRavenRuntime.initialize(trainer)
        batch = build_native_raven_batch(trainer, build_origin_attested_entrypoints(trainer, runtime))
        probe = run_teacher_camera_soak if steps == 324 else run_teacher_capacity
        ack = ("I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS" if steps == 324
               else step_acknowledgement(steps))
        if steps == 324 and recorded_reference is not None:
            from recorded_action_camera import ACK
            from leo_capacity_probe import run_recorded_action_camera_soak
            result = run_recorded_action_camera_soak(batch,trainer._rollout_lifecycle(),steps=steps,
                acknowledgement=ACK,report=stage_report,reference=recorded_reference)
        else:
            result = probe(batch, trainer._rollout_lifecycle(), steps=steps, acknowledgement=ack, report=stage_report)
        trainer.env.simulator.sim.pause()
        return result

    results = run_stages(stages, switch=switch, execute=execute, report=report)
    final = dict(results[-1])
    final["warm_scene_diagnostic"] = {
        "schema": SCHEMA, "plan_sha256": digest, "arm": "fresh" if gpu == "0" else "warm",
        "scene_count": len(stages), "scene_initializations": backend.scene_initializations,
        "native_initializations": 1, "total_environment_step_calls": sum(s["steps"] for s in stages),
        "total_physical_transition_rows": 16 * sum(s["steps"] for s in stages),
        "stage_results": results, "same_failed_mixed_policy_trajectory": False,
    }
    atomic_write_json(output, final, mode=0o644)
    return final
