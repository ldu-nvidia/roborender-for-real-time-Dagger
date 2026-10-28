"""Bounded warm training-phase diagnostic, never a learner/data producer.

The privileged teacher/Homie and physical step callbacks are the same retained
native entrypoints used by DAgger. This does not exercise student rendering,
optimization, BC chunk collection, or a production checkpoint handoff.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

try:
    from .native_motion_assignment import validate_motion_assignment
except ImportError:
    from native_motion_assignment import validate_motion_assignment

SCHEMA = "roborender.native-training-phase-probe.v1"
RESULT_SCHEMA = "roborender.native-training-phase-probe-result.v1"
MODE = "training-phase-probe"
RUN_ID = "leo-" + MODE
ACK = "I_ACKNOWLEDGE_81_B16_NATIVE_TEACHER_ONLY_PHASE_PROBE_STEPS"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_plan(value):
    fields = {"schema", "purpose", "command_id", "source_sha256", "physical_gpu",
              "native_pid", "worker_instance_id", "checkpoint_sha256", "assignment",
              "num_envs", "steps", "acknowledgement"}
    if (not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA
            or value["purpose"] != "warm_phase_smoke"
            or not isinstance(value["command_id"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value["command_id"])
            or type(value["physical_gpu"]) is not int or value["physical_gpu"] not in (0, 1)
            or type(value["native_pid"]) is not int or value["native_pid"] <= 0
            or not isinstance(value["worker_instance_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", value["worker_instance_id"])
            or any(not isinstance(value[k], str) or not re.fullmatch(r"[0-9a-f]{64}", value[k])
                   for k in ("source_sha256", "checkpoint_sha256"))
            or type(value["num_envs"]) is not int or value["num_envs"] != 16
            or type(value["steps"]) is not int or value["steps"] != 81
            or value["acknowledgement"] != ACK):
        raise ValueError("exact owner-bound B16/81-step warm-phase smoke plan required")
    assignment = validate_motion_assignment(value["assignment"])
    if (assignment["purpose"] != "training" or assignment["run_id"] != RUN_ID
            or assignment["cohort_id"] != value["command_id"] or len(assignment["lanes"]) != 16):
        raise ValueError("phase probe needs its own exact training-only B16 assignment")
    canonical(value)
    return deepcopy(value)


def read_plan(path, expected_sha256):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 1 << 20):
        raise ValueError("bounded regular phase-probe plan required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("phase-probe plan digest differs")
    return validate_plan(json.loads(raw))


def validate_owner(plan, state, *, source_sha256, physical_gpu, native_pid, inventory):
    """Read-only admission before scene mutation; no existing command adoption."""
    validate_plan(plan)
    if (state.get("schema") != "roborender.isaac-resident.v1"
            or plan["source_sha256"] != source_sha256 or plan["physical_gpu"] != physical_gpu
            or plan["native_pid"] != native_pid or state.get("native_pid") != native_pid
            or state.get("worker_instance_id") != plan["worker_instance_id"]
            or state.get("source_sha256") != source_sha256
            or state.get("native_initializations") != 1
            or state.get("phase") not in {"IDLE", "RUNNING"}
            or (state.get("phase") == "IDLE" and state.get("active_command_id") is not None)
            or (state.get("phase") == "RUNNING"
                and state.get("active_command_id") != plan["command_id"])):
        raise ValueError("phase-probe owner/source/lifecycle identity differs")
    rows = inventory.get("motions", [])
    if (inventory.get("schema") != "roborender.motion-inventory.v1"
            or inventory.get("motion_count") != 295 or len(rows) != 295
            or inventory.get("motion_records_sha256") != hashlib.sha256(canonical(rows)).hexdigest()
            or plan["assignment"]["motion_inventory_sha256"] != hashlib.sha256(canonical(inventory)).hexdigest()):
        raise ValueError("phase-probe inventory identity differs")
    indexed = {row["motion_id"]: row for row in rows}
    if len(indexed) != 295:
        raise ValueError("phase-probe inventory contains duplicate motions")
    for lane in plan["assignment"]["lanes"]:
        original = indexed.get(lane["motion_id"], {})
        if (lane["sha256"] != original.get("sha256")
                or Path(lane["path"]).name != original.get("filename")):
            raise ValueError("phase-probe motion differs from the owner's inventory")
    return plan


def parameter_fingerprints(trainer):
    """Phase-boundary tensor readback, not per-step copying or weight mutation."""
    import torch
    models = {"student": trainer.policy_model, "teacher": trainer.ref_model,
              "homie_walk": trainer.unwrapped_model.homie_walk_model,
              "homie_stand": trainer.unwrapped_model.homie_stand_model}
    result = {}
    for role, model in models.items():
        digest = hashlib.sha256()
        state = model.state_dict()
        if not state:
            raise RuntimeError("phase probe cannot attest an empty model")
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError("non-finite model state before/after phase probe")
            digest.update(canonical([name, str(value.dtype), list(value.shape)]))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[role] = {"sha256": digest.hexdigest(), "state_tensors": len(state)}
    return result


def require_rollout_inference_modes(trainer):
    # Raven's training-rollout path is inference mode, not learner train mode.
    # In particular, no BatchNorm statistics may change in this diagnostic.
    models = (trainer.policy_model, trainer.ref_model,
              trainer.unwrapped_model.homie_walk_model, trainer.unwrapped_model.homie_stand_model)
    if any(model.training is not False for model in models):
        raise RuntimeError("phase probe requires Raven's frozen teacher/student/Homie rollout modes")


def run_probe_steps(batch, lifecycle, plan, *, report):
    """Reuse the existing exactly-once teacher probe, never its production path."""
    from leo_capacity_probe import _run_teacher_steps
    validate_plan(plan)
    if batch.num_envs != 16:
        raise ValueError("actual native phase-probe batch is not B16")
    result = _run_teacher_steps(batch, lifecycle, steps=81, report=report, mode=MODE)
    if (result.get("result") != "PASS" or result.get("environment_step_calls") != 81
            or result.get("physical_transition_rows") != 1296
            or result.get("optimizer_steps") != 0 or result.get("chunks_published") != 0
            or result.get("observation_sources") != {"raytraced": 0, "generated_controls": 1296}):
        raise RuntimeError("phase probe did not finish its exact generated-control teacher-step budget")
    return result


def run_native(backend, plan, output, report):
    # All native imports are deferred until the already-owning Kit executes.
    import os
    from dagger_rtx.durable import atomic_write_json
    from dagger_rtx.native_entrypoints import NativeRavenRuntime, build_origin_attested_entrypoints
    from dagger_rtx.native_raven import build_native_raven_batch
    from dagger_rtx.observation_contract import configure_native_observations, GENERATED_ONLY_CONTRACT
    from native_motion_assignment import install_native_assignment_observer
    from resident_isaac_trainer import reset_rollout_for_scene, seeded, initial_state, digest

    validate_plan(plan)
    trainer = backend.trainer
    assignment = plan["assignment"]
    backend._kit_unchanged()
    guard = getattr(trainer, "_native_resource_guard", None)
    if not callable(guard):
        raise RuntimeError("phase probe requires the actual native resource guard")
    guard()
    start_kit = backend.kit_identity
    start_scene = backend.scene_initializations
    backend._switch_scene("training", seed=assignment["simulator_seed"], panel=None,
                          report=report, assignment=assignment)
    trainer.env.simulator.sim.play()
    for model in (trainer.policy_model, trainer.ref_model):
        reset_rollout_for_scene(model)
    seeded(assignment["simulator_seed"])
    before = parameter_fingerprints(trainer)
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None or not callable(getattr(optimizer, "step", None)):
        raise RuntimeError("phase probe cannot install its optimizer prohibition")
    original_step = optimizer.step

    def forbid_optimizer(*_args, **_kwargs):
        raise RuntimeError("warm phase probe forbids all optimizer updates")

    def progress(row):
        guard()
        backend._kit_unchanged()
        atomic_write_json(output / "progress.json", {
            "phase": row["phase"], "completed": row.get("completed_vector_steps", 0),
            "total": 81, "batch_size": 16, "native_pid": os.getpid(),
            "irreversible_step_intents": row.get("irreversible_step_intents"),
        }, mode=0o644)
        report(row["phase"], **{k: v for k, v in row.items() if k != "phase"})

    optimizer.step = forbid_optimizer
    try:
        with install_native_assignment_observer(trainer, assignment, output / "native-assignments") as observer:
            backend._prepare_training(assignment["simulator_seed"])
            # These remain controls only: no RGB, native or generated, is sent
            # to a student by this teacher-only diagnostic.
            configure_native_observations(trainer, GENERATED_ONLY_CONTRACT)
            runtime = NativeRavenRuntime.initialize(trainer)
            require_rollout_inference_modes(trainer)
            initial = {key: digest(value) for key, value in sorted(initial_state(trainer.env, runtime.obs_dict).items())}
            reset_receipt = deepcopy(trainer.env._rr_collection_initial_reset)
            batch = build_native_raven_batch(trainer, build_origin_attested_entrypoints(trainer, runtime))
            guard()
            result = run_probe_steps(batch, trainer._rollout_lifecycle(), plan, report=progress)
            assignments = deepcopy(observer.episode_assignments)
            resets = list(observer.episode_resets)
        after = parameter_fingerprints(trainer)
        if before != after:
            raise RuntimeError("model parameters or normalization buffers changed during teacher-only phase probe")
        if len(assignments) < 16 or len(resets) != 16 or any(n < 1 for n in resets):
            raise RuntimeError("phase probe lacks actual native reset evidence for every lane")
        backend._kit_unchanged()
        if backend.kit_identity != start_kit:
            raise RuntimeError("phase probe changed its Kit instance")
        guard()
        trainer.env.simulator.sim.pause()
    finally:
        optimizer.step = original_step
    return dict(schema=RESULT_SCHEMA, complete=True, diagnostic_only=True,
        production_qualified=False, learner_integration_tested=False,
        mixed_policy_tested=False, student_inference_tested=False, H200_renderer_tested=False,
        optimizer_steps=0, chunks_published=0, checkpoint_promotions=0,
        environment_step_calls=81, physical_transition_rows=1296, num_envs=16,
        teacher_action_equalities=1296, checkpoint_sha256=plan["checkpoint_sha256"],
        checkpoint_binding_scope="context_only_not_loaded_or_inferred_by_this_teacher_probe",
        native_initializations=1, native_pid=os.getpid(), kit_identity=start_kit,
        worker_instance_id=plan["worker_instance_id"], command_id=plan["command_id"],
        source_sha256=backend.source_sha256, profile=backend.profile,
        scene_initializations=backend.scene_initializations, scene_rebuilds=backend.scene_initializations-start_scene,
        phase_resets=backend.phase_resets, assignment=assignment,
        initial_state=initial, initial_reset_completion=reset_receipt,
        model_state_before=before, model_state_after=after, model_parameters_unchanged=True,
        episode_assignments=assignments, actual_episode_resets_per_lane=resets, step_probe=result)
