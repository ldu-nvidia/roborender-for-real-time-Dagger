"""Native resident adapter; imported only after the sole AppLauncher boot.

Legacy training keeps B6 and evaluation the qualified B5 panel. The explicit
MSE smoke uses separately pinned B2 motion scenes; no legacy scene is relabeled.
A batch/profile change replaces the scene/context, NOT Kit.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import random
import time

# Kit already exists, but the resident scene and its capture buffers do not.
# Only an explicitly pinned opt-in affects this process; default is unchanged.
from native_thp_policy import configure_from_environment, attest as attest_native_thp
NATIVE_THP_CONFIGURATION = configure_from_environment()

import numpy as np
import torch
from omegaconf import OmegaConf
import omni.kit.app
from isaaclab.sim import SimulationContext

from dagger_rtx.beta_round import read_config, acknowledgement, gpu_uuids
from dagger_rtx.contracts import canonical_json
from dagger_rtx.durable import atomic_write_json
from dagger_rtx.native_service_trainer import NativeDaggerRTXServiceTrainer
from dagger_rtx.native_entrypoints import NativeRavenRuntime, build_origin_attested_entrypoints
from dagger_rtx.native_raven import build_native_raven_batch
from dagger_rtx.process_memory import compact_host_heap
from dagger_rtx.phase_reset_state import ColdPhaseState
from dagger_rtx.reset_inventory import FullResetInventoryMixin
from dagger_rtx.provenance import verify_assets
from dagger_rtx.resident_service import ResidentOwner, serve, bounded_native_progress

# Importing the diagnostic module applies a legacy global scenario hook. Keep
# it scoped to its evaluation profile; it must not infect training construction.
from gr00t.rl.data.tasks.walk_stand_place_grasp_turn_homie.scenario_cfg import isaacsim as scenario
_training_table_metadata = scenario.get_table_grid_meta_data
from isaac6_panel_evaluator import PanelResetEnvironment, BetaPanelEvaluator
from isaac6_success_diagnostic import IsaacSim as EvaluationIsaacSim, initial_state, digest
from gr00t.rl.simulator.isaacsim.isaacsim import IsaacSim as NativeIsaacSim
scenario.get_table_grid_meta_data = _training_table_metadata
from beta_small_batch_panel import validate_panel
from warm_panel_session import read_request, run_pair


def normalize_material_paths(module):
    """Lab3 config copies need ordinary strings, not non-pickleable Vt arrays.

    Preserve every selected path and its order. Do not replace the native
    training motion metadata with the held-out evaluator's metadata hook.
    """
    for name in ("table_top_material_prim_paths", "leg_material_prim_paths"):
        setattr(module, name, [str(item) for item in getattr(module, name)])


def reset_rollout_for_scene(model):
    """Drop old observation references without double-calling native cleanup.

    Raven's Actor.clear_rollout deletes distribution unconditionally, so it
    is not idempotent after the owning window has already finished. Its
    init_rollout performs the exact buffer reset and is safe at a new phase.
    """
    if hasattr(model, "init_rollout"):
        model.init_rollout()
    if hasattr(model, "clear_rollout"):
        # Native Actor starts with distribution=None and deletes it at
        # finish, but init_rollout does not recreate it. Restore that pristine
        # phase state so the next real window can finish exactly once too.
        model.distribution = None


def release_native_scene(env, *, flush_app, acquire_render_product=None, inspect_only=False):
    """Dispose only this scene's RTX products; keep AppLauncher/Kit alive.

    IsaacLab3's camera cleanup detaches annotators but its RTX renderer does
    not destroy Replicator's globally retained HydraTexture objects. Closing
    the USD stage and collecting Python objects alone therefore leaves those
    render resources alive across motion-cohort switches. Capture the owned
    paths before STOP invalidates camera render data, detach first, and use
    the pinned Replicator handle's destroy operation before closing the stage.

    No task reset, action step, seed, camera setting or physics parameter is
    changed here. inspect_only attests actual initialized-camera handles at
    bootstrap without stopping physics, destroying resources or updating Kit.
    Exceptions are terminal: constructing a new scene over a
    partially released old one is not a safe recovery path.
    """
    simulator = env.simulator
    sensors = list(simulator.scene.sensors.values())
    paths = set()
    cameras = []
    for sensor in sensors:
        data = getattr(sensor, "_render_data", None)
        if data is None:
            continue
        cameras.append(sensor)
        for path in data.render_product_paths:
            if not isinstance(path, str) or not path.startswith("/Render/"):
                raise RuntimeError("old scene reported an invalid owned render-product path")
            paths.add(path)
    if paths and acquire_render_product is None:
        # Resolve ONLY existing handles. Replicator1.13.25's attach helper
        # searches context "", whereas ordinary camera products are registered
        # under "default"; a miss CREATES a second texture. That helper is not a
        # lookup API. Enumerate the pinned registry instead, across its existing
        # contexts, and match the texture's authoritative native product path.
        from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager
        registry = ViewportManager()._hydra_textures

        def acquire_render_product(path):
            matches = []
            for textures in registry._hydra_textures.values():
                for texture in textures:
                    native = getattr(texture, "hydra_texture", None)
                    if native is not None and str(native.get_render_product_path()) == path:
                        matches.append(texture)
            if len(matches) != 1:
                raise RuntimeError("owned render product has no unique existing registry handle")
            return matches[0]
    products = []
    for path in sorted(paths):
        product = acquire_render_product(path)
        if (getattr(product, "path", None) != path
                or not callable(getattr(product, "destroy", None))):
            raise RuntimeError("owned render product lacks its pinned destruction handle")
        products.append(product)

    if inspect_only:
        return {"existing_render_products": len(products), "render_product_paths": sorted(paths),
                "camera_sensors": len(cameras), "kit_updates": 0, "environment_step_calls": 0,
                "native_mutations": 0, "inspect_only": True}

    simulator.sim.stop()
    # STOP normally invalidates cameras. Complete that same native cleanup if
    # a callback left render data attached, before destroying its product.
    for camera in cameras:
        if getattr(camera, "_render_data", None) is not None:
            camera._invalidate_initialize_callback(None)
    for product in products:
        product.destroy()
    # Drain deferred renderer work while physics is stopped; this is a Kit
    # update, not env.step or a simulation reset.
    flush_app()
    simulator.scene_creation_callback = None
    for name in ("event_manager", "viewport_camera_controller"):
        if hasattr(simulator, name):
            delattr(simulator, name)
    SimulationContext.clear_instance()
    flush_app()
    env.simulator = None
    # The detached environment can contain bound-method cycles. Break the
    # native owner/scene edge explicitly so those cycles cannot pin sensors.
    simulator.scene = None
    simulator.conditioning_cameras = {}
    simulator.ego_camera = None
    simulator.depth_hi_camera = None
    return {"destroyed_render_products": len(products), "camera_sensors": len(cameras),
            "kit_updates": 2, "environment_step_calls": 0}


def native_device_string(device):
    """Keep Isaac's required indexed device inside the one-UUID container."""
    value = torch.device(device)
    if value.type != "cuda":
        raise ValueError("the resident native simulator requires CUDA")
    index = torch.cuda.current_device() if value.index is None else value.index
    if index != 0 or torch.cuda.device_count() != 1:
        raise RuntimeError("resident CUDA must expose exactly its pinned GPU")
    return f"cuda:{index}"


class ResidentTrainingIsaacSim(NativeIsaacSim):
    """Only normalize USD material path containers; native training RGB stays RGB01."""
    def __init__(self, *args, **kwargs):
        scenario_path = Path(inspect.getsourcefile(scenario)).resolve()
        original = importlib.machinery.SourceFileLoader.exec_module
        bound = []

        def patch(loader, module):
            original(loader, module)
            if Path(str(getattr(loader, "path", ""))).resolve() == scenario_path:
                normalize_material_paths(module)
                bound.append(True)

        importlib.machinery.SourceFileLoader.exec_module = patch
        try:
            super().__init__(*args, **kwargs)
        finally:
            importlib.machinery.SourceFileLoader.exec_module = original
        if not bound:
            raise RuntimeError("native training material path normalization did not execute")


# Raven's entrypoint boots Kit only when the configured target's final name
# is exactly IsaacSim. Keep that launcher contract while using our subclass.
IsaacSim = ResidentTrainingIsaacSim


class ResidentTrainingEnvironment(FullResetInventoryMixin, PanelResetEnvironment):
    """Cold phase reset, ordinary native episode resets within a rollout."""
    def reset_all(self):
        if getattr(self, "_resident_boundary_reset", False):
            self._resident_boundary_reset = False
            return self._reset_cold_scene()
        return super().reset_all()


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def phase_environment(values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def file_identity(path, expected=None, maximum=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or maximum is not None and path.stat().st_size > maximum:
        raise ValueError("expected a bounded regular phase artifact")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    actual = value.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("phase artifact content identity changed")
    return actual


class ResidentRavenBackend:
    def __init__(self, trainer, *, jobs_root, runtime_root, source_sha256):
        self.trainer = trainer
        self.jobs_root = Path(jobs_root).resolve(strict=True)
        self.runtime_root = Path(runtime_root).resolve(strict=True)
        self.source_sha256 = source_sha256
        verify_assets(Path(os.environ["DAGGER_ASSET_MANIFEST"]),
                      runtime_raven_root=Path(os.environ["DAGGER_RAVEN_RUNTIME_ROOT"]))
        inventory_path = Path(os.environ["RESIDENT_MOTION_INVENTORY"])
        if os.environ.get("RESIDENT_PRODUCTION_FULL295") == "1":
            from dagger_rtx.production_inventory import read_inventory
            self.motion_inventory, self.motion_inventory_sha256 = read_inventory(inventory_path,
                os.environ["RESIDENT_MOTION_INVENTORY_SHA256"], os.environ["RESIDENT_PRODUCTION_MOTION_ROOT"])
        else:
            self.motion_inventory_sha256 = file_identity(inventory_path, maximum=1 << 20)
            self.motion_inventory = json.loads(inventory_path.read_bytes())
            for row in self.motion_inventory["motions"]:
                motion = Path(row["path"])
                if motion.stat().st_size != row["size_bytes"]:
                    raise ValueError("held-out motion size differs")
                file_identity(motion, row["sha256"])
        self.kit = omni.kit.app.get_app()
        self.kit_identity = id(self.kit)
        self.native_device = native_device_string(trainer.env.device)
        self.base_env_config = OmegaConf.create(OmegaConf.to_container(trainer.env.config, resolve=True))
        self.base_algo_config = OmegaConf.create(OmegaConf.to_container(trainer.config, resolve=True))
        self.profile = "training"
        self.scene_seed = int(os.environ["RESIDENT_INITIAL_SCENE_SEED"])
        self.scene_initializations = 1
        self.phase_resets = 0
        self._consumed_outputs = set()
        self.active_output = None
        self.active_arm = None
        self.cold_phase_state = ColdPhaseState(trainer.env)
        # Build before any command/rollout. PPO diagnostics must never mutate
        # the shared distillation teacher or poison a later training phase.
        if os.environ.get('RESIDENT_SCALED_TEACHER') == 'roborender.scaled-teacher-benchmark.v1':
            from isolated_ppo_teacher import build_isolated_teacher
            trainer._scaled_teacher_private = build_isolated_teacher(trainer)
        if "startup" in trainer.env.simulator.event_manager.available_modes:
            raise RuntimeError("warm reuse has no certified replay for startup physics randomization")

    def progress(self):
        root, arm = self.active_output, self.active_arm
        return bounded_native_progress(root / arm if root is not None and arm else root)

    def artifact(self, name):
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("phase artifacts must be relative to the jobs mount")
        value = (self.jobs_root / name).resolve(strict=True)
        if not value.is_relative_to(self.jobs_root):
            raise ValueError("phase artifact escapes the jobs mount")
        return value

    def validate(self, operation, payload):
        if operation == 'run_phase':
            from dagger_rtx.phase_modes import resolve
            _, native_operation, parameters = resolve(payload)
            self.validate(native_operation, parameters)
            if native_operation == 'collect_production_segment':
                from dagger_rtx.production_segment import is_continuous_segment
                _, segment, _ = self._production_segment_inputs(parameters)
                if not is_continuous_segment(segment) or 'training_curriculum' not in parameters:
                    raise ValueError('DAGGER_TRAIN requires continuous episodes and an explicit curriculum handoff')
            return
        if operation == 'idle':
            if payload:
                raise ValueError('idle takes no parameters')
            return
        if operation == 'collect_bc_continuous':
            adapter = getattr(self, 'bc_collection_adapter', None)
            if adapter is None:
                raise ValueError('BC_COLLECT requires a bound resident BC adapter; standalone guards cannot be bypassed')
            adapter.validate(payload)
            return
        if operation == 'evaluate_pair':
            raise ValueError('legacy success evaluator is retired; use corrected evaluate_explicit_panel')
        if operation == "shutdown":
            return
        if operation == 'evaluate_scaled_rendered_panel':
            self._scaled_success_inputs(payload)
            return
        if operation == 'evaluate_raven_pooled_rendered':
            self._scaled_success_inputs(payload, raven_pooled=True)
            return
        if operation == 'evaluate_scaled_teacher_panel':
            self._scaled_teacher_inputs(payload)
            return
        if operation == 'probe_training_phase':
            self._training_phase_probe_inputs(payload)
            return
        if operation == 'probe_continuous_training':
            self._continuous_diagnostic_inputs(payload)
            return
        if operation == "evaluate_rendered_continuation":
            continuation_fields = {"continuation_admission_file", "continuation_admission_sha256",
                                   "continuation_evidence_file", "continuation_evidence_sha256"}
            scientific_fields = {"panel_file", "panel_sha256", "checkpoint_sha256", "arm", "output_name",
                "teacher_action_mode", "diagnostic_initial_robot_refresh", "rendered_observation_config_file",
                "rendered_observation_config_sha256"}
            if set(payload) != scientific_fields | continuation_fields or payload["arm"] != "bc":
                raise ValueError("rendered continuation requires its separate exact admission payload")
            # All old corrected-reset/fresh-rendered guards remain in force.
            self.validate("evaluate_explicit_panel", {key: payload[key] for key in scientific_fields})
            self._rendered_continuation_inputs(payload)
            return
        schemas = {
            "diagnose_scene_pressure": {"output_name", "plan_file", "plan_sha256"},
            "capture_initial_b2": {"output_name"},
            "capture_reset": {"profile", "seed", "output_name", "panel_file", "panel_sha256"},
            "prepare_scene": {"profile", "seed", "output_name", "panel_file", "panel_sha256"},
            "collect_round": {"round_config", "round_config_sha256", "token_file", "connection_status_directory",
                              "connection_config_sha256", "output_name"},
            "collect_mse_round": {"round_config", "round_config_sha256", "token_file", "connection_status_directory",
                                  "connection_config_sha256", "output_name", "mse_protocol_file", "mse_protocol_sha256"},
            "collect_production_segment": {"round_config", "round_config_sha256", "token_file", "connection_status_directory",
                                  "connection_config_sha256", "output_name", "mse_protocol_file", "mse_protocol_sha256",
                                  "segment_file", "segment_sha256", "coverage_plan_file", "coverage_plan_sha256",
                                  "motion_assignment_file", "motion_assignment_sha256"},
            "collect_mse_validation": {"collection_actor_file", "collection_actor_sha256", "motion_assignment_file",
                                       "motion_assignment_sha256", "token_file", "output_name"},
            "evaluate_pair": {"panel_file", "panel_sha256", "request_file", "request_sha256", "output_name"},
            "evaluate_explicit_panel": {"panel_file", "panel_sha256", "checkpoint_sha256", "arm", "output_name"},
        }
        extra_training = {"motion_assignment_file", "motion_assignment_sha256"}
        curriculum_options_valid = (operation == 'collect_production_segment'
            and schemas[operation] <= set(payload) <= schemas[operation] | {
                'training_curriculum','population_scope_file','population_scope_sha256'})
        evaluation_options = {"reference_output_name", "completed_receipts_file", "completed_receipts_sha256",
                              "teacher_action_mode", "numeric_trace_steps", "diagnostic_initial_robot_refresh",
                              "diagnostic_protocol", "training_support_selection_file", "training_support_selection_sha256",
                              "rendered_observation_config_file", "rendered_observation_config_sha256"}
        explicit_options_valid = (operation == "evaluate_explicit_panel"
            and schemas[operation] <= set(payload) <= schemas[operation] | evaluation_options
            and (("completed_receipts_file" in payload) == ("completed_receipts_sha256" in payload)))
        if set(payload) != schemas[operation] and not (
                operation == "collect_mse_round" and set(payload) == schemas[operation] | extra_training) and not (
                explicit_options_valid) and not curriculum_options_valid:
            raise ValueError("unknown or missing native phase fields")
        name = payload["output_name"]
        import re
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", name):
            raise ValueError("invalid fresh phase output name")
        if name in self._consumed_outputs or (self.runtime_root / "phases" / name).exists():
            raise ValueError("existing native phase cannot be replayed")
        if operation == "diagnose_scene_pressure":
            from scene_pressure_diagnostic import read_plan
            if (os.environ.get("RESIDENT_PRODUCTION_FULL295") != "1"
                    or not os.environ.get("RESIDENT_SCENE_LOCK_FILE")
                    or os.environ.get("RESIDENT_RESOURCE_POLICY") != "roborender.native-memory-policy.v3"):
                raise ValueError("scene diagnostic requires the unchanged production guard and shared lock")
            plan = read_plan(self.artifact(payload["plan_file"]), payload["plan_sha256"])
            if hashlib.sha256(canonical_json(plan['inventory'])).hexdigest() != self.motion_inventory_sha256:
                raise ValueError("scene diagnostic inventory differs from native owner")
        if operation == "capture_initial_b2" and (os.environ.get("RESIDENT_PRODUCTION_FULL295") != "1"
                or int(self.trainer.env.num_envs) != 2 or self.profile != "training"):
            raise ValueError("B2 bootstrap capture is an explicit production-only initial-scene operation")
        if operation in {"capture_reset", "prepare_scene"}:
            if payload["profile"] not in {"training", "evaluation"} or type(payload["seed"]) is not int or not 0 <= payload["seed"] < 2**31:
                raise ValueError("invalid native profile/seed")
        if operation in {"capture_reset", "prepare_scene", "evaluate_pair"}:
            p = self.artifact(payload["panel_file"])
            file_identity(p, payload["panel_sha256"], 1 << 20)
            panel = validate_panel(json.loads(p.read_bytes()))
            if panel["motion_inventory_sha256"] != self.motion_inventory_sha256:
                raise ValueError("panel motion identity differs from native held-out assets")
        if operation == "evaluate_pair":
            request = self.artifact(payload["request_file"])
            panel = validate_panel(json.loads(self.artifact(payload["panel_file"]).read_bytes()))
            value = read_request(request, payload["request_sha256"], panel)
            for arm, expected in value["checkpoint_sha256"].items():
                file_identity(Path("/diagnostic-assets") / (expected+".pt"), expected)
        if operation in {"collect_round", "collect_mse_round", "collect_production_segment"}:
            config = read_config(self.artifact(payload["round_config"]), payload["round_config_sha256"])
            if operation == 'collect_production_segment' and 'training_curriculum' in payload:
                self._curriculum_predecessor(payload['training_curriculum'], config)
            if operation == 'collect_production_segment':
                self._population_scope(payload, config)
            if os.environ.get('RESIDENT_PIPELINE_MODE')=='smoke':
                from dagger_rtx.beta_round import is_corrected_smoke
                if operation!='collect_mse_round' or not is_corrected_smoke(config):
                    raise ValueError('smoke native owner requires corrected smoke collection')
            gpu = int(os.environ["RESIDENT_PHYSICAL_GPU"])
            if (config["source_sha256"] != self.source_sha256 or gpu_uuids(config).get(gpu) != os.environ["CUDA_VISIBLE_DEVICES"]):
                raise ValueError("resident source/GPU differs from the round")
            token = self.artifact(payload["token_file"])
            if token.stat().st_mode & 0o077:
                raise ValueError("round bearer token must be owner-only")
            status = self.artifact(payload["connection_status_directory"])
            if not status.is_dir():
                raise ValueError("connection status directory missing")
            from dagger_rtx.beta_round import is_mse_production
            if is_mse_production(config) != (operation == "collect_production_segment"):
                raise ValueError("full-motion production requires a bounded explicit cohort operation")
            if operation == "collect_production_segment":
                self._production_segment_inputs(payload, config=config)
            if operation == "collect_mse_round":
                from dagger_rtx.mse_round_driver import read_mse_protocol
                read_mse_protocol(self.artifact(payload["mse_protocol_file"]), payload["mse_protocol_sha256"], config)
                from dagger_rtx.beta_round import is_mse_smoke
                if is_mse_smoke(config):
                    from native_motion_assignment import load_motion_assignment
                    binding = next(row for row in config["motion_assignments"] if row["physical_gpu_index"] == gpu)
                    if not extra_training <= set(payload) or payload["motion_assignment_sha256"] != binding["sha256"]:
                        raise ValueError("explicit MSE training assignment is missing or changed")
                    assignment = load_motion_assignment(self.artifact(payload["motion_assignment_file"]), binding["sha256"])
                    if (assignment["purpose"] != "training" or assignment["run_id"] != config["run_id"]
                            or assignment["simulator_seed"] != config["seed"] + gpu or len(assignment["lanes"]) != 2):
                        raise ValueError("native training assignment run/seed/B differs")
                elif extra_training & set(payload):
                    raise ValueError("legacy MSE training cannot silently adopt an explicit-motion topology")
        if operation == "collect_mse_validation":
            from dagger_rtx.heldout_collection_driver import read_actor
            from native_motion_assignment import load_motion_assignment
            spec = read_actor(self.artifact(payload["collection_actor_file"]), payload["collection_actor_sha256"])
            assignment = load_motion_assignment(self.artifact(payload["motion_assignment_file"]), payload["motion_assignment_sha256"])
            if (spec["source_sha256"] != self.source_sha256 or spec["gpu_uuid"] != os.environ["CUDA_VISIBLE_DEVICES"]
                    or spec["physical_gpu_index"] != int(os.environ["RESIDENT_PHYSICAL_GPU"])
                    or spec["motion_assignment_sha256"] != payload["motion_assignment_sha256"]
                    or Path(spec["motion_assignment_path"]).resolve() != self.artifact(payload["motion_assignment_file"])
                    or assignment["purpose"] != "validation" or assignment["run_id"] != spec["registration"]["run_id"]
                    or assignment["simulator_seed"] != spec["simulator_seed"] or len(assignment["lanes"]) != spec["num_envs"]
                    or assignment["motion_inventory_sha256"] != spec["registration"]["motion_inventory_sha256"]):
                raise ValueError("held-out resident source/GPU/native assignment differs")
            if self.artifact(payload["token_file"]).stat().st_mode & 0o077:
                raise ValueError("held-out bearer token must be owner-only")
        if operation == "evaluate_explicit_panel":
            from explicit_motion_panel import validate_panel as validate_explicit
            from teacher_action_diagnostic import DISTILL, validate_mode, validate_trace_steps
            from initial_robot_refresh_diagnostic import SCALED_TEACHER_CLEAR_ONCE
            if payload.get('diagnostic_initial_robot_refresh') == SCALED_TEACHER_CLEAR_ONCE:
                raise ValueError('scaled teacher reset requires the separate evaluate_scaled_teacher_panel operation')
            validate_mode(payload.get("teacher_action_mode", DISTILL), payload["arm"])
            if "numeric_trace_steps" in payload:
                validate_trace_steps(payload["numeric_trace_steps"])
            path = self.artifact(payload["panel_file"])
            file_identity(path, payload["panel_sha256"], 4 << 20)
            panel = validate_explicit(json.loads(path.read_bytes()))
            from scaled_success_contract import is_panel as is_scaled_panel
            if is_scaled_panel(panel):
                raise ValueError('scaled panel requires the separate evaluate_scaled_rendered_panel operation')
            from initial_robot_refresh_diagnostic import ORIGINAL, STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE, validate_panel_mode, require_corrected_execution
            require_corrected_execution(payload.get('diagnostic_initial_robot_refresh'),
                arm=payload['arm'], teacher_action_mode=payload.get('teacher_action_mode', DISTILL),
                has_resume='completed_receipts_file' in payload)
            validate_panel_mode(panel, payload.get("diagnostic_initial_robot_refresh", ORIGINAL),
                arm=payload["arm"], teacher_action_mode=payload.get("teacher_action_mode", DISTILL),
                has_resume="completed_receipts_file" in payload)
            if panel["purpose"] != "test" or payload["arm"] not in {"bc", "dagger", "teacher"}:
                raise ValueError("explicit success phase is independent final testing only")
            training_support_fields = {"diagnostic_protocol", "training_support_selection_file", "training_support_selection_sha256"}
            rendered_fields = {"rendered_observation_config_file", "rendered_observation_config_sha256"}
            if rendered_fields & set(payload) and training_support_fields & set(payload):
                from training_support_success_contract import read_source_selection
                from rendered_training_support_contract import (SCHEMA as rendered_schema, read_inputs,
                    validate_execution, evaluation_options as rendered_options)
                if (not (training_support_fields | rendered_fields) <= set(payload)
                        or payload["diagnostic_protocol"] != rendered_schema
                        or os.environ.get("RESIDENT_PRODUCTION_FULL295") != "1"
                        or os.environ.get("RESIDENT_PIPELINE_MODE") != "production"
                        or {"reference_output_name", "completed_receipts_file", "completed_receipts_sha256", "numeric_trace_steps"} & set(payload)):
                    raise ValueError("rendered training-support diagnostic requires complete fresh paired admission")
                selection = read_source_selection(self.artifact(payload["training_support_selection_file"]),
                                                  payload["training_support_selection_sha256"])
                rendered_config, paired_baseline = read_inputs(self.artifact(payload["rendered_observation_config_file"]),
                    payload["rendered_observation_config_sha256"], panel, artifact=self.artifact)
                validate_execution(panel, selection, payload["checkpoint_sha256"], rendered_config,
                                   paired_baseline, arm=payload["arm"])
                required = rendered_options(panel, raw_sha256=payload["panel_sha256"])
                if (panel["motion_inventory_sha256"] != self.motion_inventory_sha256
                        or any(payload.get(key) != value for key, value in required.items())):
                    raise ValueError("rendered diagnostic inventory/corrected student options differ")
            elif rendered_fields & set(payload):
                from rendered_student_success_contract import read_inputs, validate_execution
                if (not rendered_fields <= set(payload) or payload["arm"] not in {"bc", "dagger"}
                        or {"reference_output_name", "completed_receipts_file", "completed_receipts_sha256",
                            "numeric_trace_steps"} & set(payload)):
                    raise ValueError("rendered student success requires complete fresh configuration")
                config = read_inputs(self.artifact(payload["rendered_observation_config_file"]),
                    payload["rendered_observation_config_sha256"], panel,
                    checkpoint_sha256=payload["checkpoint_sha256"])
                validate_execution(panel, payload["checkpoint_sha256"], config, arm=payload["arm"])
                if payload.get("diagnostic_initial_robot_refresh") == SMOKE_STUDENT_CLEAR_ONCE:
                    if (os.environ.get("RESIDENT_PRODUCTION_FULL295") == "1"
                            and os.environ.get("RESIDENT_PIPELINE_MODE") != "smoke"):
                        raise ValueError("production owner requires the full rendered295 panel, not a smoke subset")
                    from smoke_success_contract import evaluation_options as student_options
                    required = student_options(panel)
                else:
                    from production_success_contract import evaluation_options as student_options
                    required = student_options(panel, raw_sha256=payload["panel_sha256"])
                    if (os.environ.get("RESIDENT_PRODUCTION_FULL295") != "1"
                            or os.environ.get("RESIDENT_PIPELINE_MODE") != "production"):
                        raise ValueError("rendered production success requires its admitted production owner")
                if (panel["motion_inventory_sha256"] != self.motion_inventory_sha256
                        or any(payload.get(key) != value for key, value in required.items())):
                    raise ValueError("rendered student success inventory/corrected options differ")
            elif training_support_fields & set(payload):
                if payload["arm"] in {"bc", "dagger"}:
                    raise ValueError("future student success requires rendered observations; raytraced diagnostic is historical only")
                from training_support_success_contract import (SCHEMA as diagnostic_schema,
                    read_source_selection, validate_execution, evaluation_options as diagnostic_options)
                if (not training_support_fields <= set(payload)
                        or payload["diagnostic_protocol"] != diagnostic_schema
                        or os.environ.get("RESIDENT_PRODUCTION_FULL295") != "1"
                        or os.environ.get("RESIDENT_PIPELINE_MODE") != "production"
                        or {"reference_output_name", "completed_receipts_file", "completed_receipts_sha256", "numeric_trace_steps"} & set(payload)):
                    raise ValueError("training-support diagnostic requires its exact fresh, unpaired production-owner request")
                selection = read_source_selection(self.artifact(payload["training_support_selection_file"]),
                                                  payload["training_support_selection_sha256"])
                validate_execution(panel, selection, payload["checkpoint_sha256"], arm=payload["arm"])
                required = diagnostic_options(panel, raw_sha256=payload["panel_sha256"])
                if (panel["motion_inventory_sha256"] != self.motion_inventory_sha256
                        or any(payload.get(key) != value for key, value in required.items())):
                    raise ValueError("training-support diagnostic inventory/corrected student options differ")
            elif payload["arm"] in {"bc", "dagger"} and (
                    (os.environ.get("RESIDENT_PRODUCTION_FULL295") == "1"
                     and os.environ.get('RESIDENT_PIPELINE_MODE')!='smoke')
                    or payload.get("diagnostic_initial_robot_refresh") == STUDENT_CLEAR_ONCE):
                raise ValueError("production student success requires rendered observation configuration")
            elif payload.get("diagnostic_initial_robot_refresh") == SMOKE_STUDENT_CLEAR_ONCE:
                raise ValueError("smoke student success requires rendered observation configuration")
            elif payload["arm"] in {"bc", "dagger"}:
                raise ValueError("student success requires rendered observation configuration")
            if (os.environ.get('RESIDENT_PIPELINE_MODE')=='smoke'
                    and payload.get('diagnostic_initial_robot_refresh')!=SMOKE_STUDENT_CLEAR_ONCE):
                raise ValueError('smoke owner cannot execute a production or diagnostic success test')
            file_identity(Path("/diagnostic-assets") / (payload["checkpoint_sha256"] + ".pt"), payload["checkpoint_sha256"])
            if "reference_output_name" in payload:
                self._explicit_test_reference(payload, panel)
            if "completed_receipts_file" in payload:
                self._explicit_completed_receipts(payload, panel)

    def validate_command_identity(self, command):
        if command['operation'] == 'run_phase':
            from dagger_rtx.phase_modes import normalize_command
            command = normalize_command(command)
        if command['operation'] == 'collect_bc_continuous':
            self.bc_collection_adapter.validate_command_identity(command)
        if command['operation'] == 'probe_continuous_training':
            scope = self._continuous_diagnostic_inputs(command['payload'])
            owner = scope['owners'][int(os.environ['RESIDENT_PHYSICAL_GPU'])]
            if (command['command_id'] != owner['command_id']
                    or command['worker_instance_id'] != owner['worker_instance_id']):
                raise ValueError('continuous diagnostic envelope differs from exact scope owner')
        if command['operation'] == 'probe_training_phase':
            plan = self._training_phase_probe_inputs(command['payload'])
            if (command['command_id'] != plan['command_id']
                    or command['worker_instance_id'] != plan['worker_instance_id']):
                raise ValueError('phase-probe command envelope differs from admitted identity')
        if command['operation'] == 'evaluate_scaled_rendered_panel':
            _, _, _, admission = self._scaled_success_inputs(command['payload'])
            if (command['command_id'] != admission['command_id']
                    or command['worker_instance_id'] != admission['worker_instance_id']):
                raise ValueError('scaled command envelope differs from admitted identity')
        if command['operation'] == 'evaluate_raven_pooled_rendered':
            _, _, _, benchmark = self._scaled_success_inputs(command['payload'], raven_pooled=True)
            if (command['command_id'] != benchmark['command_id']
                    or command['worker_instance_id'] != benchmark['owner']['worker_instance_id']):
                raise ValueError('pooled benchmark envelope differs from admitted identity')
        if command['operation'] == 'evaluate_scaled_teacher_panel':
            _, benchmark = self._scaled_teacher_inputs(command['payload'])
            if (command['command_id'] != benchmark['command_id']
                    or command['worker_instance_id'] != benchmark['owner']['worker_instance_id']):
                raise ValueError('scaled teacher envelope differs from admitted identity')

    def _training_phase_probe_inputs(self, payload):
        from phase_probe import read_plan, validate_owner
        if set(payload) != {'output_name', 'plan_file', 'plan_sha256'}:
            raise ValueError('phase probe requires its exact bounded diagnostic payload')
        if (os.environ.get('RESIDENT_PRODUCTION_FULL295') != '1'
                or os.environ.get('RESIDENT_PIPELINE_MODE') != 'production'
                or not os.environ.get('RESIDENT_SCENE_LOCK_FILE')
                or os.environ.get('RESIDENT_RESOURCE_POLICY') != 'roborender.native-memory-policy.v3'):
            raise ValueError('phase probe requires the current production guard and shared scene lock')
        plan = read_plan(self.artifact(payload['plan_file']), payload['plan_sha256'])
        name = plan['command_id']
        if (payload['output_name'] != name or name in self._consumed_outputs
                or (self.runtime_root / 'phases' / name).exists()):
            raise ValueError('phase probe requires its fresh command-bound output; no replay')
        state = json.loads((self.runtime_root / 'resident-state.json').read_bytes())
        validate_owner(plan, state, source_sha256=self.source_sha256,
            physical_gpu=int(os.environ['RESIDENT_PHYSICAL_GPU']), native_pid=os.getpid(),
            inventory=self.motion_inventory)
        # This is only an immutable diagnostic context, not a student load or
        # inference claim. Testing has its own separately admitted checkpoint.
        file_identity(Path('/diagnostic-assets') / (plan['checkpoint_sha256'] + '.pt'),
                      plan['checkpoint_sha256'])
        return plan

    def _continuous_diagnostic_inputs(self, payload):
        from dagger_rtx.continuous_diagnostic import read_scope, require_owner
        extra = {'diagnostic_scope_file', 'diagnostic_scope_sha256'}
        if not extra <= set(payload):
            raise ValueError('continuous diagnostic requires its separate immutable scope')
        base = {k: v for k, v in payload.items() if k not in extra}
        # Every production binding, resource policy, assignment and source
        # check still applies. Only the separately authorized terminal differs.
        self.validate('collect_production_segment', base)
        config, segment, _ = self._production_segment_inputs(base)
        scope = read_scope(self.artifact(payload['diagnostic_scope_file']), payload['diagnostic_scope_sha256'],
            config_sha256=payload['round_config_sha256'], segment_sha256=payload['segment_sha256'],
            segment=segment, source_sha256=self.source_sha256)
        owner = require_owner(scope, json.loads((self.runtime_root / 'resident-state.json').read_bytes()),
            gpu=int(os.environ['RESIDENT_PHYSICAL_GPU']), native_pid=os.getpid(), command_id=payload['output_name'])
        if config['source_sha256'] != scope['source_sha256'] or owner['gpu_uuid'] != os.environ['CUDA_VISIBLE_DEVICES']:
            raise ValueError('continuous diagnostic physical source/GPU differs')
        if 'training_curriculum' in scope:
            request = scope['training_curriculum'][int(os.environ['RESIDENT_PHYSICAL_GPU'])]
            if payload.get('training_curriculum') != request:
                raise ValueError('diagnostic curriculum request differs from exact scope')
        elif 'training_curriculum' in payload:
            raise ValueError('curriculum diagnostic requires its versioned scope')
        return scope

    def _population_scope(self, payload, config):
        from dagger_rtx.population_curriculum import CONFIG_FIELD, PROTOCOL, digest
        fields={'population_scope_file','population_scope_sha256'}
        if config.get(CONFIG_FIELD) is None:
            if fields & set(payload):raise ValueError('population scope requires its explicit protocol')
            return None
        if not fields <= set(payload):raise ValueError('shared curriculum requires both admitted native identities')
        path=self.artifact(payload['population_scope_file'])
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=payload['population_scope_sha256']:
            raise ValueError('native population scope file changed')
        scope=json.loads(raw)
        state=json.loads((self.runtime_root/'resident-state.json').read_bytes())
        gpu=int(os.environ['RESIDENT_PHYSICAL_GPU'])
        if (scope.get('schema')!=PROTOCOL or scope.get('config_sha256')!=payload['round_config_sha256']
                or scope.get('segment_sha256')!=payload['segment_sha256']
                or scope.get('source_sha256')!=self.source_sha256 or len(scope.get('owners',[]))!=2):
            raise ValueError('native population scope/config/source mismatch')
        own=scope['owners'][gpu]
        if (own.get('physical_gpu')!=gpu or own.get('native_pid')!=os.getpid()
                or own.get('command_id')!=payload['output_name']
                or any(own.get(k)!=state[k] for k in ('worker_instance_id','gpu_uuid','source_sha256'))):
            raise ValueError('native population lifecycle identity mismatch')
        return scope

    def _curriculum_predecessor(self, request, config):
        from dagger_rtx.training_curriculum import resolve_predecessor
        state = json.loads((self.runtime_root / 'resident-state.json').read_bytes())
        prior = resolve_predecessor(request, state,
            parent_checkpoint_sha256=config['student']['initial_checkpoint_sha256'],
            physical_gpu=int(os.environ['RESIDENT_PHYSICAL_GPU']), source_sha256=self.source_sha256)
        if prior is None:
            if config['round_id'] != 1:
                raise ValueError('later training rounds require their actual predecessor curriculum')
            parent = config['student']['initial_checkpoint_sha256']
            if any((row.get('receipt') or {}).get('training_curriculum', {}).get('binding', {}).get(
                    'final_checkpoint_sha256') == parent for row in state.get('commands', {}).values()):
                raise ValueError('a learned native parent requires curriculum restore, not configuration bootstrap')
        elif config.get('population_curriculum_protocol') is not None:
            from dagger_rtx.training_curriculum import SHARED_POPULATION
            if prior['population']!=SHARED_POPULATION:
                raise ValueError('shared population cannot resume independent actor curriculum')
        return prior

    def _scaled_teacher_inputs(self, payload):
        from scaled_teacher_benchmark import SCHEMA as TEACHER_SCHEMA, validate_benchmark, validate_owner, evaluation_options
        from scaled_success_contract import SCHEMA
        fields = {'panel_file','panel_sha256','benchmark_file','benchmark_sha256','checkpoint_sha256',
                  'arm','teacher_action_mode','diagnostic_initial_robot_refresh','output_name'}
        if (not isinstance(payload, dict) or set(payload) != fields
                or any(payload.get(k) != v for k,v in evaluation_options().items())
                or os.environ.get('RESIDENT_SCALED_TEACHER') != TEACHER_SCHEMA
                or os.environ.get('RESIDENT_SCALED_SUCCESS') != SCHEMA
                or os.environ.get('RESIDENT_PRODUCTION_FULL295') != '1'
                or os.environ.get('RESIDENT_PIPELINE_MODE') != 'production'
                or not os.environ.get('RESIDENT_SCENE_LOCK_FILE')
                or os.environ.get('RESIDENT_RESOURCE_POLICY') != 'roborender.native-memory-policy.v3'):
            raise ValueError('scaled teacher requires its separate exact native admission')
        path = self.artifact(payload['panel_file'])
        file_identity(path, payload['panel_sha256'], 4 << 20)
        panel = json.loads(path.read_bytes())
        path = self.artifact(payload['benchmark_file'])
        file_identity(path, payload['benchmark_sha256'], 1 << 20)
        benchmark = validate_benchmark(json.loads(path.read_bytes()), panel)
        name = payload['output_name']
        if (benchmark['output_name'] != name or benchmark['shadow_not_behavior_checkpoint_sha256'] != payload['checkpoint_sha256']
                or panel['motion_inventory_sha256'] != self.motion_inventory_sha256
                or name in self._consumed_outputs or (self.runtime_root/'phases'/name).exists()):
            raise ValueError('scaled teacher output/checkpoint/inventory changed or was consumed')
        state = json.loads((self.runtime_root/'resident-state.json').read_bytes())
        validate_owner(benchmark, state, source_sha256=self.source_sha256,
            physical_gpu=int(os.environ['RESIDENT_PHYSICAL_GPU']), native_pid=os.getpid())
        file_identity(Path('/diagnostic-assets')/(payload['checkpoint_sha256']+'.pt'), payload['checkpoint_sha256'])
        if getattr(self.trainer, '_scaled_teacher_private', None) is None:
            raise ValueError('scaled teacher requires a fresh native lifecycle with isolated teacher')
        return panel, benchmark

    def _scaled_success_inputs(self, payload, *, raven_pooled=False):
        from scaled_success_contract import SCHEMA, require_panel, evaluation_options, single_partition, validate_admission
        from rendered_student_success_contract import read_inputs
        fields = {'panel_file', 'panel_sha256', 'checkpoint_sha256', 'arm', 'output_name',
            'teacher_action_mode', 'diagnostic_initial_robot_refresh',
            'rendered_observation_config_file', 'rendered_observation_config_sha256',
            'partition_file', 'partition_sha256', 'scaled_admission_file', 'scaled_admission_sha256'}
        qualification_fields = {'qualification_file', 'qualification_sha256'}
        if raven_pooled:
            fields = (fields - {'scaled_admission_file', 'scaled_admission_sha256'}) | {'benchmark_file', 'benchmark_sha256'}
            if qualification_fields & set(payload):
                raise ValueError('pooled benchmark cannot inherit fixed295 qualification outcomes')
        if set(payload) not in (fields, fields | qualification_fields):
            raise ValueError('scaled success requires its exact separate admission payload')
        import re
        name = payload['output_name']
        if (not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', name)
                or name in self._consumed_outputs or (self.runtime_root / 'phases' / name).exists()):
            raise ValueError('scaled phase output must be fresh and cannot replay')
        if (os.environ.get('RESIDENT_SCALED_SUCCESS') != SCHEMA
                or os.environ.get('RESIDENT_PRODUCTION_FULL295') != '1'
                or os.environ.get('RESIDENT_PIPELINE_MODE') != 'production'):
            raise ValueError('scaled success requires its explicit current-source native owner')
        path = self.artifact(payload['panel_file'])
        file_identity(path, payload['panel_sha256'], 4 << 20)
        panel = require_panel(json.loads(path.read_bytes()))
        if (panel['motion_inventory_sha256'] != self.motion_inventory_sha256
                or payload['arm'] not in {'bc', 'dagger'}
                or any(payload.get(k) != v for k, v in evaluation_options(panel).items())):
            raise ValueError('scaled student motion inventory or corrected execution differs')
        config = read_inputs(self.artifact(payload['rendered_observation_config_file']),
            payload['rendered_observation_config_sha256'], panel, checkpoint_sha256=payload['checkpoint_sha256'])
        path = self.artifact(payload['partition_file'])
        file_identity(path, payload['partition_sha256'], 1 << 20)
        partition = json.loads(path.read_bytes())
        if not isinstance(partition, dict) or partition != single_partition(panel, partition.get('cohort_indices')):
            raise ValueError('scaled operation must name one exact original-panel cohort')
        from rendered_endpoint_pool import expected_physical_gpu
        expected_gpu = expected_physical_gpu(config['renderer'], partition['cohort_indices'][0])
        if expected_gpu is not None and os.environ.get('RESIDENT_PHYSICAL_GPU') != str(expected_gpu):
            raise ValueError('scaled cohort renderer shard differs from actual physical RTX owner')
        admission_key = 'benchmark' if raven_pooled else 'scaled_admission'
        path = self.artifact(payload[admission_key + '_file'])
        file_identity(path, payload[admission_key + '_sha256'], 1 << 20)
        admission = json.loads(path.read_bytes())
        state = json.loads((self.runtime_root / 'resident-state.json').read_bytes())
        if raven_pooled:
            from raven_pooled_benchmark import validate_benchmark, validate_owner
            validate_benchmark(admission, panel, config)
            validate_owner(admission, state, source_sha256=self.source_sha256,
                physical_gpu=int(os.environ['RESIDENT_PHYSICAL_GPU']), native_pid=os.getpid())
            if (admission['output_name'] != name or admission['cohort_index'] != partition['cohort_indices'][0]
                    or admission['arm'] != payload['arm']):
                raise ValueError('pooled benchmark output/partition differs')
            file_identity(Path('/diagnostic-assets') / (payload['checkpoint_sha256'] + '.pt'), payload['checkpoint_sha256'])
            return panel, config, partition, admission
        validate_admission(admission, panel, config, source_sha256=self.source_sha256,
            native_state=state, output_name=name, partition=partition, arm=payload['arm'])
        if panel['scaled_protocol']['purpose'] != 'qualification':
            if not qualification_fields <= set(payload):
                raise ValueError('scaled full test requires a genuinely validated current-source qualification proof')
            from pipeline.evaluation.scaled_success_qualification import validate_qualification
            path = self.artifact(payload['qualification_file'])
            if payload['qualification_sha256'] != admission['qualification_evidence_sha256']:
                raise ValueError('scaled qualification proof differs from admitted exact bytes')
            file_identity(path, payload['qualification_sha256'], 64 << 20)
            validate_qualification(json.loads(path.read_bytes()), source_sha256=self.source_sha256,
                                   panel=panel, config=config)
        elif qualification_fields & set(payload):
            raise ValueError('qualification prefix cannot inherit claimed full readiness')
        file_identity(Path('/diagnostic-assets') / (payload['checkpoint_sha256'] + '.pt'), payload['checkpoint_sha256'])
        return panel, config, partition, admission

    def _rendered_continuation_inputs(self, payload):
        from explicit_motion_panel import validate_panel
        from rendered_student_success_contract import read_inputs
        from rendered_success_continuation import read_evidence, read_pinned, validate_admission
        panel = validate_panel(json.loads(self.artifact(payload["panel_file"]).read_bytes()))
        config = read_inputs(self.artifact(payload["rendered_observation_config_file"]),
            payload["rendered_observation_config_sha256"], panel, checkpoint_sha256=payload["checkpoint_sha256"])
        evidence = read_evidence(self.artifact(payload["continuation_evidence_file"]),
            payload["continuation_evidence_sha256"], panel, config)
        admission = read_pinned(self.artifact(payload["continuation_admission_file"]),
            payload["continuation_admission_sha256"])
        state = json.loads((self.runtime_root / "resident-state.json").read_bytes())
        validate_admission(admission, evidence, source_sha256=self.source_sha256,
            native_state=state, output_name=payload["output_name"])
        return evidence, admission

    def _explicit_completed_receipts(self, payload, panel):
        from explicit_motion_panel import completed_progress, panel_sha256
        from teacher_action_diagnostic import DISTILL, validate_mode
        from initial_robot_refresh_diagnostic import ORIGINAL, validate_mode as validate_initial_refresh
        validate_initial_refresh(payload.get("diagnostic_initial_robot_refresh", ORIGINAL),
            arm=payload["arm"], teacher_action_mode=payload.get("teacher_action_mode", DISTILL), has_resume=True)
        path = self.artifact(payload["completed_receipts_file"])
        file_identity(path, payload["completed_receipts_sha256"], 16 << 20)
        document = json.loads(path.read_bytes())
        mode = validate_mode(payload.get("teacher_action_mode", DISTILL), payload["arm"])
        legacy = {"schema", "panel_sha256", "checkpoint_sha256", "arm", "cohort_receipts"}
        version_valid = ((set(document) == legacy
                and document.get("schema") == "roborender.explicit-motion-success-resume.v1"
                and mode == DISTILL)
            or (set(document) == legacy | {"teacher_action_mode"}
                and document.get("schema") == "roborender.explicit-motion-success-resume.v2"
                and document.get("teacher_action_mode") == mode))
        if (not version_valid
                or document["panel_sha256"] != panel_sha256(panel)
                or document["checkpoint_sha256"] != payload["checkpoint_sha256"]
                or document["arm"] != payload["arm"]):
            raise ValueError("success resume document panel/checkpoint/arm differs")
        completed_progress(panel, document["cohort_receipts"],
            checkpoint_sha256=payload["checkpoint_sha256"], arm=payload["arm"], teacher_action_mode=mode)
        return document["cohort_receipts"]

    def _production_segment_inputs(self, payload, *, config=None):
        from dagger_rtx.beta_round import is_mse_production
        from dagger_rtx.mse_round_driver import read_mse_protocol
        from dagger_rtx.production_segment import read_segment, validate_assignment_document, is_continuous_segment
        from native_motion_assignment import load_motion_assignment
        config = config or read_config(self.artifact(payload["round_config"]), payload["round_config_sha256"])
        if not is_mse_production(config) or payload["coverage_plan_sha256"] != config["coverage_plan"]["raw_sha256"]:
            raise ValueError("production coverage identity differs from full-motion contract")
        mse=read_mse_protocol(self.artifact(payload["mse_protocol_file"]), payload["mse_protocol_sha256"], config)
        plan_path = self.artifact(payload["coverage_plan_file"])
        segment = read_segment(self.artifact(payload["segment_file"]), payload["segment_sha256"], config, plan_path)
        coverage=json.loads(plan_path.read_bytes())
        continuous = is_continuous_segment(segment)
        expected_interval = 100 if continuous else coverage['updates_per_coverage_epoch']
        expected_updates = coverage['optimizer_updates_per_round'] if continuous else coverage['max_updates_per_round']
        if (mse['interval_iterations'] != expected_interval
                or mse['max_iterations'] != expected_updates):
            raise ValueError('MSE cadence/budget differs from complete coverage epochs')
        if segment["mse_protocol_file_sha256"] != payload["mse_protocol_sha256"]:
            raise ValueError("production segment is bound to a different MSE protocol")
        gpu = int(os.environ["RESIDENT_PHYSICAL_GPU"])
        binding = segment["motion_assignments"][gpu]
        if binding["sha256"] != payload["motion_assignment_sha256"]:
            raise ValueError("native production assignment changed")
        assignment = load_motion_assignment(self.artifact(payload["motion_assignment_file"]), binding["sha256"])
        validate_assignment_document(segment, json.loads(plan_path.read_bytes()), gpu, assignment)
        if assignment["motion_inventory_sha256"] != self.motion_inventory_sha256:
            raise ValueError("production assignment uses different native motion assets")
        return config, segment, assignment

    def _explicit_test_reference(self, payload, panel):
        """Pair only against a completed owned teacher phase, never arbitrary paths."""
        import re
        from explicit_motion_panel import validate_result
        name = payload["reference_output_name"]
        if (payload["arm"] != "bc" or not isinstance(name, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", name)
                or name == payload["output_name"]):
            raise ValueError("paired reference must name a previous teacher phase for a BC diagnostic")
        phases = (self.runtime_root / "phases").resolve()
        root = (phases / name).resolve()
        if root.parent != phases:
            raise ValueError("paired reference escaped the owned native phase root")
        receipt = json.loads((root / "native-phase-receipt.json").read_bytes())
        if receipt.get("complete") is not True or receipt.get("optimizer_steps") != 0:
            raise ValueError("paired teacher reference phase is incomplete")
        reference_mode = receipt["result"].get("teacher_action_mode", "distill")
        if reference_mode == "distill":
            validate_result(panel, receipt["result"], payload["checkpoint_sha256"], allowed_arms=("teacher",))
        else:
            validate_result(panel, receipt["result"], payload["checkpoint_sha256"], allowed_arms=("teacher",),
                            teacher_action_mode=reference_mode,
                            diagnostic_initial_robot_refresh=receipt["result"].get("diagnostic_initial_robot_refresh", "raven_original"))
        return root

    def _kit_unchanged(self):
        if id(omni.kit.app.get_app()) != self.kit_identity:
            raise RuntimeError("resident Kit identity changed")

    def _switch_scene(self, profile, *, seed, panel, report, assignment=None, preserve_shared_teacher=False,
                      reset_inventory=None, reset_lane_offset=0):
        self._kit_unchanged()
        # Reject mixed-purpose requests before releasing the current scene,
        # including when its profile/seed would otherwise permit reuse.
        if profile not in {"training", "evaluation", "explicit_evaluation"}:
            raise ValueError("unknown resident scene profile")
        if profile == "explicit_evaluation" and (
                assignment is None or assignment.get("purpose") != "test"):
            raise ValueError("explicit test scene needs an admitted motion assignment")
        if assignment is not None and profile == "training" and assignment.get("purpose") not in {"training", "validation"}:
            raise ValueError("test assignment cannot enter a training scene")
        if reset_inventory is not None:
            if profile != 'training' or assignment is None or assignment['purpose'] != 'training':
                raise ValueError('full reset inventory is specific to admitted native training')
            from native_motion_assignment import _reset_inventory
            _reset_inventory([{k:r[k] for k in ('motion_id','path','sha256')} for r in assignment['lanes']],
                             reset_inventory, reset_lane_offset)
        elif reset_lane_offset:
            raise ValueError('reset offset requires a full reset inventory')
        if profile == "training":
            self._require_collection_teacher()
        from scene_reconfiguration_lock import scene_reconfiguration_lock
        with scene_reconfiguration_lock(report):
            return self._switch_scene_admitted(profile,seed=seed,panel=panel,report=report,assignment=assignment,
                                               preserve_shared_teacher=preserve_shared_teacher,
                                               reset_inventory=reset_inventory,reset_lane_offset=reset_lane_offset)

    def _switch_scene_admitted(self, profile, *, seed, panel, report, assignment=None, preserve_shared_teacher=False,
                               reset_inventory=None, reset_lane_offset=0):
        assignment_sha = hashlib.sha256(canonical_json(assignment)).hexdigest() if assignment is not None else None
        reset_sha = (hashlib.sha256(canonical_json(dict(motions=reset_inventory, offset=reset_lane_offset))).hexdigest()
                     if reset_inventory is not None else None)
        if (self.profile == profile and self.scene_seed == seed and getattr(self, "scene_assignment_sha", None) == assignment_sha
                and getattr(self, 'scene_reset_inventory_sha', None) == reset_sha):
            report("scene_reused", profile=profile, num_envs=self.trainer.env.num_envs)
            return
        report("scene_reconfigure_start", old_profile=self.profile, profile=profile)
        started = time.monotonic()
        models = (self.trainer.policy_model,) if preserve_shared_teacher else (self.trainer.policy_model, self.trainer.ref_model)
        for model in models:
            reset_rollout_for_scene(model)
        # Release globally retained camera products before Lab3 closes the old
        # USD stage. AppLauncher/Kit and the policy models remain resident.
        old_env = self.trainer.env
        self.trainer.env = None
        self.cold_phase_state = None
        cleanup = release_native_scene(old_env, flush_app=self.kit.update)
        del old_env
        gc.collect()
        compact_host_heap("resident_scene_released")
        report("scene_resources_released", **cleanup)
        seeded(seed)
        cfg = copy.deepcopy(self.base_env_config)
        OmegaConf.set_struct(cfg, False)
        cfg.num_envs = 5 if profile == "evaluation" else 6
        cfg.simulator.config.scene.num_envs = cfg.num_envs
        if profile == "explicit_evaluation":
            if assignment is None or assignment["purpose"] != "test":
                raise ValueError("explicit test scene needs an admitted motion assignment")
            from native_motion_assignment import configure_motion_environment, explicit_scene_motion_binding
            from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
            cfg.simulator._target_ = "isaac6_motion_evaluator.IsaacSim"
            cfg.domain_rand.image_augmentation.enabled = False
            configure_motion_environment(cfg, assignment)
            with explicit_scene_motion_binding(cfg, assignment, scenario, MotionLoaderWSPPT) as evidence:
                self.trainer.env = PanelResetEnvironment(cfg, self.native_device)
                self.trainer.env.simulator.native_motion_assignment_sha256 = evidence["assignment_sha256"]
        elif profile == "evaluation":
            cfg.simulator._target_ = "isaac6_success_diagnostic.IsaacSim"
            cfg.reset_from_dataset.use_motion_file_dir = False
            cfg.reset_from_dataset.motion_files = [row["path"] for row in self.motion_inventory["motions"]]
            cfg.reset_from_dataset.motion_assignment_offset = 0
            cfg.domain_rand.image_augmentation.enabled = False
            self.trainer.env = PanelResetEnvironment(cfg, self.native_device)
        else:
            scenario.get_table_grid_meta_data = _training_table_metadata
            cfg.simulator._target_ = "resident_isaac_trainer.IsaacSim"
            if assignment is not None:
                from native_motion_assignment import configure_motion_environment, explicit_scene_motion_binding
                from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
                configure_motion_environment(cfg, assignment, motion_inventory=reset_inventory, lane_offset=reset_lane_offset)
                with explicit_scene_motion_binding(cfg, assignment, scenario, MotionLoaderWSPPT,
                        motion_inventory=reset_inventory, lane_offset=reset_lane_offset) as evidence:
                    self.trainer.env = ResidentTrainingEnvironment(cfg, self.native_device)
                    self.trainer.env.simulator.native_motion_assignment_sha256 = evidence["assignment_sha256"]
                    if reset_inventory is not None:
                        self.trainer.env.simulator.native_reset_inventory_binding = {
                            k:evidence[k] for k in ('motion_inventory_sha256','reset_lane_offset','reset_motion_count')}
            else:
                self.trainer.env = ResidentTrainingEnvironment(cfg, self.native_device)
        self.profile = profile
        self.scene_seed = seed
        self.scene_assignment_sha = assignment_sha
        self.scene_reset_inventory_sha = reset_sha
        self.cold_phase_state = ColdPhaseState(self.trainer.env)
        self.scene_initializations += 1
        self._kit_unchanged()
        report("scene_reconfigure_end", profile=profile, seconds=time.monotonic()-started,
               scene_initializations=self.scene_initializations, native_initializations=1)

    def _require_collection_teacher(self):
        # Historical original-PPO diagnostics alter max_noise_std and may
        # clamp the shared teacher std in-place. Never silently reuse such a
        # model for Raven distillation collection; use a fresh pinned owner.
        if hasattr(self.trainer.ref_model, "_original_ppo_std_policy"):
            raise RuntimeError("PPO diagnostic teacher cannot enter collection; a fresh pinned owner is required")

    def _prepare_training(self, seed):
        self._require_collection_teacher()
        # Clear only the previous phase's explicit observation override. A new
        # fixed100 native train call rebinds generated_only before initialization;
        # legacy capture/heldout operations retain their original source mixture.
        if hasattr(self.trainer, '_roborender_student_observation_contract'):
            from dagger_rtx.observation_contract import configure_native_observations
            configure_native_observations(self.trainer, None)
        self.cold_phase_state.restore(self.trainer.env)
        for name in ("cur_reward_sum", "cur_episode_length"):
            value = getattr(self.trainer, name, None)
            if isinstance(value, torch.Tensor):
                if value.shape != (self.trainer.env.num_envs,):
                    setattr(self.trainer, name, torch.zeros(self.trainer.env.num_envs, dtype=value.dtype, device=value.device))
                else:
                    value.zero_()
                if hasattr(self.trainer, "state"):
                    setattr(self.trainer.state, name, getattr(self.trainer, name))
        self.trainer.config = copy.deepcopy(self.base_algo_config)
        self.trainer._id_anchors = None
        self.trainer._id_prompts = None
        self.trainer._id_use_raytraced = None
        self.trainer._id_rng = np.random.RandomState(int(self.trainer.vg_seed))
        seeded(seed)
        self.trainer.env.reset_dataset.resample()
        self.trainer.env.reset_count = 0
        self.trainer.env._resident_boundary_reset = True
        self.phase_resets += 1

    def phase_capabilities(self):
        from dagger_rtx.phase_modes import capabilities
        return capabilities(bc_collection=getattr(self, 'bc_collection_adapter', None) is not None)

    def _run_phase(self, payload, report):
        from dagger_rtx.phase_modes import resolve, validate_completion
        mode, operation, parameters = resolve(payload)
        self.validate('run_phase', payload)  # Re-read stage inputs on the native thread.
        report('mode_preparing', mode=mode, operation=operation)
        try:
            # The stage owns its actual prepare/ready/run/drain boundaries.
            # Do not manufacture READY merely because source validation passed.
            receipt = self.run(operation, parameters, report)
            report('mode_finalizing', mode=mode, operation=operation)
            validate_completion(mode, receipt)
            torch.cuda.synchronize(self.native_device)
            self.trainer.env.simulator.sim.pause()
            if self.trainer.env.simulator.sim.is_playing():
                raise RuntimeError('phase did not leave physics idle')
            self._kit_unchanged()
            result = dict(receipt, mode=mode, physics_idle=True, kit_identity=self.kit_identity,
                          native_initializations=1, compute_released=False)
            report('mode_idle', mode=mode, operation=operation)
            return result
        except BaseException:
            # No recovery/reset/replay here; the command owner records uncertainty.
            try:
                self.trainer.env.simulator.sim.pause()
            except Exception as exc:
                report('phase_failure_pause_failed', error_type=type(exc).__name__)
            raise

    def run(self, operation, payload, report):
        if operation == 'run_phase':
            return self._run_phase(payload, report)
        report("phase_transition_start", operation=operation, profile=self.profile,
               scene_initializations=self.scene_initializations)
        try:
            receipt = self._run_guarded(operation, payload, report)
            if not isinstance(receipt, dict) or receipt.get("complete") is not True:
                raise RuntimeError("native phase has no complete receipt")
            self._kit_unchanged()
        except BaseException:
            # ResidentOwner will make this terminal (CRASH_UNCERTAIN). Stop
            # physics even if a reset/evaluation fails before its normal tail.
            # Never reset, step, retry, or mark the owner reusable here.
            try:
                env = self.trainer.env
                simulator = getattr(env, "simulator", None)
                if simulator is not None:
                    simulator.sim.pause()
            except Exception as pause_error:
                report("phase_failure_pause_failed", error_type=type(pause_error).__name__)
            raise
        self.active_output = None
        self.active_arm = None
        report("phase_transition_complete", operation=operation, profile=self.profile,
               scene_initializations=self.scene_initializations, native_initializations=1)
        return receipt

    def _run_guarded(self, operation, payload, report):
        if operation == 'idle':
            self.trainer.env.simulator.sim.pause()
            return dict(complete=True, optimizer_steps=0, native_initializations=1)
        if operation == 'collect_bc_continuous':
            self.validate(operation, payload)
            return self.bc_collection_adapter.run(payload, report)
        if (os.environ.get("RESIDENT_PRODUCTION_FULL295") == "1"
                and operation not in {"shutdown", "collect_production_segment", "probe_continuous_training"}):
            # Explicit production-only monitoring. The external owned launcher
            # also enforces the cap/floor; this receipt binds actual phase
            # samples without scanning data or changing simulator callbacks.
            from dagger_rtx.phase_resource_guard import PhaseResourceGuard,POLICY_VERSION
            self.validate(operation, payload)
            output = self.runtime_root / "phases" / payload["output_name"]
            previous_guard = getattr(self.trainer, "_native_resource_guard", None)
            with PhaseResourceGuard(output / "resource-guard.json",
                    policy_version=os.environ.get("RESIDENT_RESOURCE_POLICY",POLICY_VERSION)) as guard:
                self.trainer._native_resource_guard = guard.check
                try:
                    receipt = self._run_operation(operation, payload, report)
                finally:
                    self.trainer._native_resource_guard = previous_guard
            receipt["resource_guard"] = guard.summary
            atomic_write_json(output / "production-phase-receipt.json", receipt, mode=0o644)
            return receipt
        return self._run_operation(operation, payload, report)

    def _run_operation(self, operation, payload, report):
        if operation == "shutdown":
            self._kit_unchanged()
            return {"complete": True, "native_initializations": 1, "shutdown_requested": True}
        if operation == "collect_mse_validation":
            return self._run_heldout(payload, report)
        if operation == "capture_initial_b2":
            return self._run_initial_b2_capture(payload, report)
        if operation == "diagnose_scene_pressure":
            return self._run_scene_pressure_diagnostic(payload, report)
        if operation == "probe_training_phase":
            return self._run_training_phase_probe(payload, report)
        if operation == "collect_production_segment":
            return self._run_production_segment(payload, report)
        if operation == "probe_continuous_training":
            return self._run_production_segment(payload, report, diagnostic=True)
        if operation == "evaluate_explicit_panel":
            return self._run_explicit_test(payload, report)
        if operation == "evaluate_rendered_continuation":
            return self._run_explicit_test(payload, report, continuation=True)
        if operation == 'evaluate_scaled_teacher_panel':
            return self._run_explicit_test(payload, report, scaled_teacher=True)
        if operation == 'evaluate_scaled_rendered_panel':
            return self._run_explicit_test(payload, report, scaled=True)
        if operation == 'evaluate_raven_pooled_rendered':
            return self._run_explicit_test(payload, report, scaled=True, raven_pooled=True)
        # A command accepted earlier is revalidated against any changed files.
        self.validate(operation, payload)
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        panel = None
        if operation in {"capture_reset", "prepare_scene", "evaluate_pair"}:
            panel = validate_panel(json.loads(self.artifact(payload["panel_file"]).read_bytes()))
        collecting = operation in {"collect_round", "collect_mse_round"}
        config = read_config(self.artifact(payload["round_config"]), payload["round_config_sha256"]) if collecting else None
        profile = "training" if collecting else "evaluation" if operation == "evaluate_pair" else payload["profile"]
        seed = config["seed"] + list(gpu_uuids(config)).index(int(os.environ["RESIDENT_PHYSICAL_GPU"])) if config else payload.get("seed", 1103)
        assignment = None
        if collecting and "motion_assignment_file" in payload:
            from native_motion_assignment import load_motion_assignment
            assignment = load_motion_assignment(self.artifact(payload["motion_assignment_file"]), payload["motion_assignment_sha256"])
        self._switch_scene(profile, seed=seed, panel=panel, report=report, assignment=assignment)
        self.trainer.env.simulator.sim.play()
        self._kit_unchanged()
        for model in (self.trainer.policy_model, self.trainer.ref_model):
            reset_rollout_for_scene(model)
        if profile == "training":
            self._prepare_training(seed)
        else:
            self.cold_phase_state.restore(self.trainer.env)
        if operation == "prepare_scene":
            receipt = {"complete": True, "environment_step_calls": 0,
                       "num_envs": self.trainer.env.num_envs, "scene_only": True}
        elif operation == "capture_reset":
            if profile == "training":
                runtime = NativeRavenRuntime.initialize(self.trainer)
                batch = build_native_raven_batch(self.trainer, build_origin_attested_entrypoints(self.trainer, runtime))
                self.trainer._capture_only(batch, runtime, output / "capture-smoke.json")
                receipt = {"capture": str(output / "capture-smoke.json"),
                           "capture_receipt": json.loads((output / "capture-smoke.json").read_bytes())}
            else:
                seeded(seed)
                self.trainer.env.reset_dataset.resample()
                self.trainer.env.reset_count = 0
                obs = self.trainer.env.reset_panel()
                state = initial_state(self.trainer.env, obs)
                identity = {key: digest(value) for key, value in sorted(state.items())}
                atomic_write_json(output / "initial-state.json", identity, mode=0o644)
                receipt = {"initial_state": identity}
            receipt.update(complete=True, environment_step_calls=0, num_envs=self.trainer.env.num_envs)
        elif collecting:
            gpu = int(os.environ["RESIDENT_PHYSICAL_GPU"])
            values = {"DAGGER_RTX_MODE": "beta-production", "DAGGER_RUNTIME_ROOT": str(output),
                "DAGGER_BETA_CONFIG": str(self.artifact(payload["round_config"])),
                "DAGGER_BETA_CONFIG_SHA256": payload["round_config_sha256"],
                "DAGGER_RUN_ID": config["run_id"], "DAGGER_ROUND_ID": config["round_id"],
                "DAGGER_ACTOR_PREFIX": config["run_id"]+f"-g{gpu}", "DAGGER_WINDOW_INDEX": 0,
                "DAGGER_WIRE_PROTOCOL": "v3", "DAGGER_H200_RPC_WORKERS": 4,
                "DAGGER_H200_TOKEN_FILE": str(self.artifact(payload["token_file"])),
                "DAGGER_EXPECTED_H200_PIDS": ",".join(map(str, config["h200"]["gateway_pids"])),
                "DAGGER_H200_BASE_URL": "http://127.0.0.1:18620",
                "DAGGER_H200_GATEWAY_URLS": ",".join(f"http://127.0.0.1:{18620+i}" for i in range(4)),
                "DAGGER_CONNECTION_STATUS": str(self.artifact(payload["connection_status_directory"])/"status.json"),
                "DAGGER_CONNECTION_CONFIG_SHA256": payload["connection_config_sha256"],
                "DAGGER_NATIVE_STEP_ACK": acknowledgement(config), "DAGGER_PHYSICAL_GPU": gpu,
                "DAGGER_PHYSICAL_GPU_UUID": os.environ["CUDA_VISIBLE_DEVICES"],
                "DAGGER_LAUNCH_STARTED_NS": time.time_ns(), "DAGGER_COLLECTION_WANDB_ENABLED": "0"}
            if operation == "collect_mse_round":
                values.update(DAGGER_MSE_PROTOCOL_FILE=str(self.artifact(payload["mse_protocol_file"])),
                              DAGGER_MSE_PROTOCOL_SHA256=payload["mse_protocol_sha256"])
            observer = nullcontext(None)
            if assignment is not None:
                from native_motion_assignment import install_native_assignment_observer
                observer = install_native_assignment_observer(self.trainer, assignment, output / "native-assignments")
            with observer as recorder, phase_environment(values):
                result = NativeDaggerRTXServiceTrainer.train(self.trainer)
            state_name = "mse-beta-round-state.json" if operation == "collect_mse_round" else "beta-round-state.json"
            receipt = {"complete": result["result"] == "PASS", "native_receipt": str(output / "native-service-result.json"),
                       "round_id": config["round_id"], "teacher_beta": config["teacher_beta"],
                       "actor_state": json.loads((output / state_name).read_bytes())}
            if assignment is not None:
                receipt.update(episode_assignments=list(recorder.episode_assignments),
                               motion_assignment_sha256=payload["motion_assignment_sha256"])
        else:
            request = read_request(self.artifact(payload["request_file"]), payload["request_sha256"], panel)
            evaluator = BetaPanelEvaluator.__new__(BetaPanelEvaluator)
            evaluator.__dict__ = self.trainer.__dict__.copy()
            evaluator.config = copy.deepcopy(self.base_algo_config)
            evaluator.config.init_at_random_ep_len = False
            evaluator.config.raven_exact_raytraced_augmentation = False
            evaluator.vg_raytraced_mix_prob = 1.0
            evaluator._id_use_raytraced = torch.ones(5, dtype=torch.bool, device=self.trainer.accelerator.device)
            def evaluate(*, arm, checkpoint, checkpoint_sha256, output, reference):
                self.active_arm = arm
                # Content-addressed files can be staged between phases without
                # replacing a bind-mounted checkpoint inode or reloading Kit.
                with phase_environment({"DIAGNOSTIC_ARM": arm,
                        "DIAGNOSTIC_CHECKPOINT_PATH": "/diagnostic-assets/"+checkpoint_sha256+".pt",
                        "DIAGNOSTIC_CHECKPOINT_SHA256": checkpoint_sha256,
                        "BETA_PANEL_PATH": str(self.artifact(payload["panel_file"])),
                        "BETA_PANEL_SHA256": payload["panel_sha256"],
                        "BETA_PANEL_COHORT_LIMIT": request["cohort_limit"]}):
                    return evaluator.evaluate_panel(output, str(reference) if reference else "")
            _, receipt = run_pair(request, panel, output, evaluate, report=lambda row: report(**row))
            receipt.update(native_initializations=1, residency="across_rounds_and_phases")
            atomic_write_json(output / "warm-session.json", receipt, mode=0o644)
        self._kit_unchanged()
        self.trainer.env.simulator.sim.pause()
        receipt.update(native_initializations=1, scene_initializations=self.scene_initializations,
                       phase_resets=self.phase_resets, native_pid=os.getpid(), profile=self.profile)
        self.active_output = None
        self.active_arm = None
        return receipt

    def _run_scene_pressure_diagnostic(self, payload, report):
        from scene_pressure_diagnostic import read_plan, run_native
        self.validate("diagnose_scene_pressure", payload)
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        plan = read_plan(self.artifact(payload["plan_file"]), payload["plan_sha256"])
        receipt = run_native(self, plan, int(os.environ["RESIDENT_PHYSICAL_GPU"]), report)
        receipt.update(native_initializations=1, native_pid=os.getpid(),
                       scene_initializations=self.scene_initializations, profile=self.profile,
                       source_sha256=self.source_sha256, plan_sha256=payload["plan_sha256"],
                       native_thp_policy=attest_native_thp(NATIVE_THP_CONFIGURATION))
        atomic_write_json(output / "scene-pressure-result.json", receipt, mode=0o644)
        return receipt

    def _run_training_phase_probe(self, payload, report):
        from phase_probe import run_native
        plan = self._training_phase_probe_inputs(payload)
        name = payload['output_name']
        self._consumed_outputs.add(name)
        output = self.runtime_root / 'phases' / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        receipt = run_native(self, plan, output, report)
        receipt.update(plan_sha256=payload['plan_sha256'],
                       native_thp_policy=attest_native_thp(NATIVE_THP_CONFIGURATION))
        atomic_write_json(output / 'training-phase-probe-result.json', receipt, mode=0o644)
        return receipt

    def _run_initial_b2_capture(self, payload, report):
        self.validate("capture_initial_b2", payload)
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        self.trainer.env.simulator.sim.play()
        for model in (self.trainer.policy_model, self.trainer.ref_model):
            reset_rollout_for_scene(model)
        self._prepare_training(self.scene_seed)
        runtime = NativeRavenRuntime.initialize(self.trainer)
        batch = build_native_raven_batch(self.trainer, build_origin_attested_entrypoints(self.trainer, runtime))
        capture_path = output / "capture-smoke.json"
        self.trainer._capture_only(batch, runtime, capture_path)
        scene_inventory = release_native_scene(self.trainer.env, flush_app=self.kit.update, inspect_only=True)
        if not scene_inventory["existing_render_products"] or not scene_inventory["camera_sensors"]:
            raise RuntimeError("initial B2 capture lacks actual owned renderer lookup attestation")
        self._kit_unchanged()
        self.trainer.env.simulator.sim.pause()
        receipt = dict(complete=True, num_envs=int(self.trainer.env.num_envs), environment_step_calls=0,
            optimizer_steps=0, native_initializations=1, native_pid=os.getpid(), profile=self.profile,
            source_sha256=self.source_sha256, scene_initializations=self.scene_initializations,
            phase_resets=self.phase_resets, capture_receipt=json.loads(capture_path.read_bytes()),
            scene_resource_inventory=scene_inventory,
            native_thp_policy=attest_native_thp(NATIVE_THP_CONFIGURATION))
        self.active_output = None
        return receipt

    def _run_production_segment(self, payload, report, *, diagnostic=False):
        """One physical scene lifetime; versioned legacy or continuous windows."""
        from dagger_rtx.phase_resource_guard import PhaseResourceGuard,POLICY_VERSION
        from dagger_rtx.production_segment import is_continuous_segment
        from dagger_rtx.beta_round import gateway_slots, gateway_pids_for_actor
        from native_motion_assignment import install_native_assignment_observer
        scope = self._continuous_diagnostic_inputs(payload) if diagnostic else None
        base = {k: v for k, v in payload.items() if k not in {'diagnostic_scope_file', 'diagnostic_scope_sha256'}}
        if not diagnostic:
            self.validate("collect_production_segment", payload)
        config, segment, assignment = self._production_segment_inputs(base)
        from dagger_rtx.reset_inventory import CONFIG_FIELD as RESET_FIELD, PROTOCOL as RESET_PROTOCOL, descriptors
        reset_inventory, reset_lane_offset = None, 0
        if config.get(RESET_FIELD) == RESET_PROTOCOL:
            reset_inventory = descriptors(self.motion_inventory, os.environ['RESIDENT_PRODUCTION_MOTION_ROOT'])
            reset_lane_offset = int(os.environ['RESIDENT_PHYSICAL_GPU']) * len(assignment['lanes'])
        population_scope=self._population_scope(payload,config)
        from dagger_rtx.training_curriculum import POPULATION, SHARED_POPULATION
        population=SHARED_POPULATION if population_scope is not None else POPULATION
        population_receipt=None
        continuous = is_continuous_segment(segment)
        curriculum_request = payload.get('training_curriculum')
        if continuous and not diagnostic and curriculum_request is None:
            raise ValueError('new continuous production requires explicit training curriculum handoff')
        curriculum_prior = (self._curriculum_predecessor(curriculum_request, config)
                            if curriculum_request is not None else None)
        if curriculum_request is not None and not continuous:
            raise ValueError('curriculum handoff requires a continuous training command')
        if curriculum_prior is not None and not diagnostic and (
                curriculum_prior['binding']['boundary'] != 'production_update100'
                or curriculum_prior['binding']['round_id'] != config['round_id'] - 1):
            raise ValueError('production curriculum must come from exact previous round update100')
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        gpu = int(os.environ["RESIDENT_PHYSICAL_GPU"])
        values = {"DAGGER_RTX_MODE": "beta-production", "DAGGER_RUNTIME_ROOT": str(output),
            "DAGGER_BETA_CONFIG": str(self.artifact(payload["round_config"])),
            "DAGGER_BETA_CONFIG_SHA256": payload["round_config_sha256"],
            "DAGGER_RUN_ID": config["run_id"], "DAGGER_ROUND_ID": config["round_id"],
            "DAGGER_ACTOR_PREFIX": config["run_id"] + f"-g{gpu}",
            "DAGGER_WINDOW_INDEX": segment["start_iteration_index"],
            "DAGGER_WIRE_PROTOCOL": "v3", "DAGGER_H200_RPC_WORKERS": 4,
            "DAGGER_H200_TOKEN_FILE": str(self.artifact(payload["token_file"])),
            "DAGGER_EXPECTED_H200_PIDS": ",".join(map(str, gateway_pids_for_actor(config, gpu))),
            "DAGGER_H200_BASE_URL": f"http://127.0.0.1:{18620+gateway_slots(config, gpu)[0]}",
            "DAGGER_H200_GATEWAY_URLS": ",".join(f"http://127.0.0.1:{18620+i}" for i in gateway_slots(config, gpu)),
            "DAGGER_CONNECTION_STATUS": str(self.artifact(payload["connection_status_directory"]) / "status.json"),
            "DAGGER_CONNECTION_CONFIG_SHA256": payload["connection_config_sha256"],
            "DAGGER_NATIVE_STEP_ACK": acknowledgement(config), "DAGGER_PHYSICAL_GPU": gpu,
            "DAGGER_PHYSICAL_GPU_UUID": os.environ["CUDA_VISIBLE_DEVICES"],
            "DAGGER_LAUNCH_STARTED_NS": time.time_ns(), "DAGGER_COLLECTION_WANDB_ENABLED": "0",
            "DAGGER_MSE_PROTOCOL_FILE": str(self.artifact(payload["mse_protocol_file"])),
            "DAGGER_MSE_PROTOCOL_SHA256": payload["mse_protocol_sha256"],
            "DAGGER_PRODUCTION_SEGMENT_FILE": str(self.artifact(payload["segment_file"])),
            "DAGGER_PRODUCTION_SEGMENT_SHA256": payload["segment_sha256"],
            "DAGGER_PRODUCTION_COVERAGE_PLAN_FILE": str(self.artifact(payload["coverage_plan_file"])),
            "DAGGER_CONTINUOUS_DIAGNOSTIC_FILE": str(self.artifact(payload['diagnostic_scope_file'])) if diagnostic else "",
            "DAGGER_CONTINUOUS_DIAGNOSTIC_SHA256": payload['diagnostic_scope_sha256'] if diagnostic else "",
            "DAGGER_HELDOUT_COLLECTION_FILE": "", "DAGGER_HELDOUT_COLLECTION_SHA256": ""}
        old_guard = getattr(self.trainer, "_native_resource_guard", None)
        old_summary = getattr(self.trainer, "_native_resource_summary", None)
        old_assignments = getattr(self.trainer, "_native_assignment_snapshot", None)
        old_population_snapshot = getattr(self.trainer, '_native_population_snapshot', None)
        with PhaseResourceGuard(output / "resource-guard.json",
                policy_version=os.environ.get("RESIDENT_RESOURCE_POLICY",POLICY_VERSION)) as guard:
            self.trainer._native_resource_guard = guard.check
            if continuous:
                self.trainer._native_resource_summary = lambda: guard.snapshot(output / 'qualification-prefix-resources.json')
            try:
                seed = assignment["simulator_seed"]
                self._switch_scene("training", seed=seed, panel=None, report=report, assignment=assignment,
                                   reset_inventory=reset_inventory, reset_lane_offset=reset_lane_offset)
                guard.check()
                self.trainer.env.simulator.sim.play()
                for model in (self.trainer.policy_model, self.trainer.ref_model):
                    reset_rollout_for_scene(model)
                seeded(seed)
                with install_native_assignment_observer(self.trainer, assignment, output / "native-assignments",
                        episode_namespace=segment["episode_namespace"], motion_inventory=reset_inventory,
                        lane_offset=reset_lane_offset) as recorder:
                    if continuous:
                        self.trainer._native_assignment_snapshot = lambda: list(recorder.episode_assignments)
                    self._prepare_training(seed)
                    curriculum_initialization = None
                    if curriculum_request is not None:
                        from dagger_rtx.training_curriculum import restore_training_curriculum, bootstrap_training_curriculum
                        if curriculum_prior is None:
                            curriculum_initialization = bootstrap_training_curriculum(self.trainer.env, phase='training')
                        else:
                            curriculum_initialization = restore_training_curriculum(self.trainer.env,
                                curriculum_prior, expected_binding=curriculum_request['predecessor']['binding'],
                                phase='training',population=population)
                        curriculum_initialization['population']=population
                    from dagger_rtx.training_curriculum import observe_initial_reset
                    population_context=nullcontext(None)
                    if population_scope is not None:
                        from dagger_rtx.population_curriculum import PairedExchange,shared_population
                        # Jobs/configs stay read-only. Both exact actors share
                        # only the separate pool-owned barrier journal mount.
                        shared_root=Path(os.environ.get('RESIDENT_POPULATION_BARRIERS',''))
                        if (str(shared_root)!='/resident/population-barriers'
                                or not shared_root.is_dir() or shared_root.is_symlink()):
                            raise ValueError('shared native population journal mount is unavailable')
                        exchange=PairedExchange(shared_root,population_scope,gpu,guard=guard.check)
                        population_context=shared_population(self.trainer.env,exchange)
                    with phase_environment(values), observe_initial_reset(self.trainer.env, curriculum_initialization), population_context as population_receipt:
                        self.trainer._native_population_snapshot=(lambda:dict(population_receipt)) if population_receipt is not None else None
                        result = NativeDaggerRTXServiceTrainer.train(self.trainer)
                    assignments = list(recorder.episode_assignments)
                guard.check()
            finally:
                self.trainer._native_resource_guard = old_guard
                self.trainer._native_resource_summary = old_summary
                self.trainer._native_assignment_snapshot = old_assignments
                self.trainer._native_population_snapshot = old_population_snapshot
        state = json.loads((output / "production-segment-state.json").read_bytes())
        expected_windows = 4 if diagnostic else segment['window_count'] if continuous else 2
        expected_steps = 81 * expected_windows
        if (state["phase"] != ('DIAGNOSTIC_COMPLETE' if diagnostic else 'COMPLETE')
                or state.get("completed_windows") != expected_windows
                or state.get("transition_batches") != expected_steps
                or state.get("physical_transition_rows") != expected_steps * len(assignment['lanes'])
                or state.get("history_reset_at_window_boundary") is not False
                or (continuous and (state.get('environment_initializations') != 1
                    or state.get('model_rollout_lifecycles') != expected_windows
                    or state.get('environment_reset_at_window_boundary') is not False))
                or result["result"] != "PASS" or not assignments):
            raise RuntimeError("native production segment did not complete its exact physical/model lifecycle budget")
        if diagnostic and (state.get('diagnostic_scope_sha256') != payload['diagnostic_scope_sha256']
                or state.get('selection') is not None or state.get('fifth_window_started') is not False
                or state.get('production_round_complete') is not False
                or result.get('diagnostic_only') is not True or not state.get('qualification_proof')):
            raise RuntimeError('continuous diagnostic lacks its proof-bound clean four-update terminal')
        self._kit_unchanged()
        self.trainer.env.simulator.sim.pause()
        receipt = {"complete": True, "purpose": "training", "actor_state": state,
            "segment_sha256": payload["segment_sha256"], "episode_namespace": segment["episode_namespace"],
            "episode_assignments": assignments, "motion_assignment_sha256": payload["motion_assignment_sha256"],
            "resource_guard": guard.summary, "native_initializations": 1,
            "scene_initializations": self.scene_initializations, "phase_resets": self.phase_resets,
            "native_pid": os.getpid(), "profile": self.profile}
        if reset_inventory is not None:
            receipt['reset_inventory'] = dict(protocol=RESET_PROTOCOL,
                **self.trainer.env.simulator.native_reset_inventory_binding)
        if population_receipt is not None:
            from dagger_rtx.population_curriculum import validate_progress
            validate_progress(population_receipt,expected_windows)
            receipt['population_curriculum']=population_receipt
        if diagnostic:
            receipt.update(purpose='diagnostic_only', diagnostic_only=True,
                diagnostic_scope_sha256=payload['diagnostic_scope_sha256'], production_round_complete=False,
                selected_checkpoint_sha256=None, optimizer_updates=4)
        if curriculum_request is not None:
            from dagger_rtx.training_curriculum import export_training_curriculum, digest as curriculum_digest
            owner_state = json.loads((self.runtime_root / 'resident-state.json').read_bytes())
            export_binding = dict(run_id=config['run_id'], round_id=config['round_id'], actor_id=f'rtx-g{gpu}',
                physical_gpu=gpu, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], native_pid=os.getpid(),
                worker_instance_id=owner_state['worker_instance_id'], source_sha256=self.source_sha256,
                command_id=name, num_envs=len(assignment['lanes']), completed_updates=expected_windows,
                boundary='diagnostic_update4' if diagnostic else 'production_update100',
                final_checkpoint_sha256=state['final_candidate_checkpoint_sha256'],
                motion_assignment_sha256=payload['motion_assignment_sha256'], config_sha256=payload['round_config_sha256'],
                diagnostic_scope_sha256=payload['diagnostic_scope_sha256'] if diagnostic else None)
            curriculum_export = export_training_curriculum(self.trainer.env, export_binding, phase='training',population=population)
            atomic_write_json(output / 'training-curriculum-export.json', curriculum_export, mode=0o444)
            receipt.update(training_curriculum=curriculum_export,
                training_curriculum_sha256=curriculum_digest(curriculum_export),
                training_curriculum_initialization=curriculum_initialization)
        atomic_write_json(output / ('continuous-diagnostic-native-receipt.json' if diagnostic
                                   else 'production-segment-native-receipt.json'), receipt, mode=0o644)
        self.active_output = None
        return receipt

    def _run_heldout(self, payload, report):
        """Dedicated native collection phase; never posts a learner barrier."""
        from dagger_rtx.heldout_collection_driver import read_actor
        from native_motion_assignment import load_motion_assignment, install_native_assignment_observer
        self.validate("collect_mse_validation", payload)
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        output.mkdir(parents=True, exist_ok=False)
        self.active_output = output
        spec = read_actor(self.artifact(payload["collection_actor_file"]), payload["collection_actor_sha256"])
        assignment = load_motion_assignment(self.artifact(payload["motion_assignment_file"]), payload["motion_assignment_sha256"])
        self._switch_scene("training", seed=spec["simulator_seed"], panel=None, report=report, assignment=assignment)
        self.trainer.env.simulator.sim.play()
        for model in (self.trainer.policy_model, self.trainer.ref_model):
            reset_rollout_for_scene(model)
        registration = spec["registration"]
        values = {"DAGGER_RTX_MODE": "single-window", "DAGGER_RUNTIME_ROOT": str(output),
            "DAGGER_RUN_ID": registration["run_id"], "DAGGER_ROUND_ID": registration["round_id"],
            "DAGGER_ACTOR_PREFIX": spec["actor_prefix"], "DAGGER_WINDOW_INDEX": 0,
            "DAGGER_WIRE_PROTOCOL": "v3", "DAGGER_H200_RPC_WORKERS": len(spec["gateway_pids"]),
            "DAGGER_H200_TOKEN_FILE": str(self.artifact(payload["token_file"])),
            "DAGGER_EXPECTED_H200_PIDS": ",".join(map(str, spec["gateway_pids"])),
            "DAGGER_H200_BASE_URL": "http://127.0.0.1:18620",
            "DAGGER_H200_GATEWAY_URLS": ",".join(f"http://127.0.0.1:{18620+i}" for i in range(len(spec["gateway_pids"]))),
            "DAGGER_NATIVE_STEP_ACK": "I_ACKNOWLEDGE_ONE_NATIVE_81_STEP_WINDOW",
            "DAGGER_PHYSICAL_GPU": spec["physical_gpu_index"], "DAGGER_PHYSICAL_GPU_UUID": spec["gpu_uuid"],
            "DAGGER_LAUNCH_STARTED_NS": time.time_ns(), "DAGGER_COLLECTION_WANDB_ENABLED": "0",
            "DAGGER_HELDOUT_COLLECTION_FILE": str(self.artifact(payload["collection_actor_file"])),
            "DAGGER_HELDOUT_COLLECTION_SHA256": payload["collection_actor_sha256"],
            "DAGGER_MSE_PROTOCOL_FILE": "", "DAGGER_MSE_PROTOCOL_SHA256": ""}
        # The observer attests the actual process seed, not the previous phase's
        # seed. Its hooks must be installed before the first phase reset.
        seeded(spec["simulator_seed"])
        with install_native_assignment_observer(self.trainer, assignment, output / "native-assignments") as recorder:
            self._prepare_training(spec["simulator_seed"])
            with phase_environment(values):
                result = NativeDaggerRTXServiceTrainer.train(self.trainer)
            assignments = list(recorder.episode_assignments)
        state = json.loads((output / "heldout-collection-state.json").read_bytes())
        if state["phase"] != "COMPLETE" or result["result"] != "PASS":
            raise RuntimeError("held-out native window is incomplete")
        receipt = {"complete": True, "purpose": "validation", "optimizer_steps": 0,
            "actor_state": state, "episode_assignments": assignments, "native_initializations": 1,
            "scene_initializations": self.scene_initializations, "phase_resets": self.phase_resets,
            "native_pid": os.getpid(), "profile": self.profile}
        atomic_write_json(output / "heldout-native-receipt.json", receipt, mode=0o644)
        self._kit_unchanged()
        self.trainer.env.simulator.sim.pause()
        self.active_output = None
        return receipt

    def _run_explicit_test(self, payload, report, *, continuation=False, scaled=False, raven_pooled=False, scaled_teacher=False):
        from explicit_motion_panel import validate_panel as validate_explicit
        from isaac6_motion_evaluator import evaluate_explicit_panel
        self.validate('evaluate_scaled_teacher_panel' if scaled_teacher else 'evaluate_raven_pooled_rendered' if raven_pooled else 'evaluate_scaled_rendered_panel' if scaled else
            "evaluate_rendered_continuation" if continuation else "evaluate_explicit_panel", payload)
        scaled_inputs = self._scaled_success_inputs(payload, raven_pooled=True) if raven_pooled else self._scaled_success_inputs(payload) if scaled else None
        scaled_partition = scaled_inputs[2] if scaled_inputs else None
        teacher_benchmark = self._scaled_teacher_inputs(payload)[1] if scaled_teacher else None
        name = payload["output_name"]
        self._consumed_outputs.add(name)
        output = self.runtime_root / "phases" / name
        self.active_output = output
        panel = validate_explicit(json.loads(self.artifact(payload["panel_file"]).read_bytes()))
        training_support_selection = None
        rendered_config = paired_baseline = None
        if "training_support_selection_file" in payload:
            from training_support_success_contract import read_source_selection
            training_support_selection = read_source_selection(
                self.artifact(payload["training_support_selection_file"]), payload["training_support_selection_sha256"])
        if "rendered_observation_config_file" in payload:
            if training_support_selection is not None:
                from rendered_training_support_contract import read_inputs
                rendered_config, paired_baseline = read_inputs(self.artifact(payload["rendered_observation_config_file"]),
                    payload["rendered_observation_config_sha256"], panel, artifact=self.artifact)
            else:
                from rendered_student_success_contract import read_inputs
                rendered_config = read_inputs(self.artifact(payload["rendered_observation_config_file"]),
                    payload["rendered_observation_config_sha256"], panel,
                    checkpoint_sha256=payload["checkpoint_sha256"])
        def prepare(assignment):
            self._switch_scene("explicit_evaluation", seed=assignment["simulator_seed"], panel=None,
                               report=report, assignment=assignment, preserve_shared_teacher=scaled_teacher)
            self.cold_phase_state.restore(self.trainer.env)
            self.trainer.env.simulator.sim.play()
            self.trainer.config = copy.deepcopy(self.base_algo_config)
            models = (self.trainer.policy_model,) if scaled_teacher else (self.trainer.policy_model, self.trainer.ref_model)
            for model in models:
                reset_rollout_for_scene(model)
            self.phase_resets += 1
        continuation_options = ({"_continuation": self._rendered_continuation_inputs(payload)} if continuation else {})
        if scaled:
            continuation_options['_scaled_partition'] = scaled_partition
        if raven_pooled:
            continuation_options['_raven_benchmark'] = scaled_inputs[3]
        if scaled_teacher:
            continuation_options['_teacher_benchmark'] = teacher_benchmark
        result = evaluate_explicit_panel(self.trainer, panel, root=output,
            checkpoint_path=str(Path("/diagnostic-assets") / (payload["checkpoint_sha256"] + ".pt")),
            checkpoint_sha256=payload["checkpoint_sha256"], prepare_cohort=prepare,
            report=lambda row: report(row.pop("phase"), **row), arm=payload["arm"],
            teacher_action_mode=payload.get("teacher_action_mode", "distill"),
            numeric_trace_steps=payload.get("numeric_trace_steps"),
            diagnostic_initial_robot_refresh=payload["diagnostic_initial_robot_refresh"],
            training_support_selection=training_support_selection,
            rendered_observation_config=rendered_config, rendered_paired_baseline=paired_baseline,
            reference_root=str(self._explicit_test_reference(payload, panel)) if "reference_output_name" in payload else "",
            completed_receipts=self._explicit_completed_receipts(payload, panel)
                if "completed_receipts_file" in payload else (), **continuation_options)
        receipt = {"complete": True, "purpose": "test", "optimizer_steps": 0, "result": result,
            "native_initializations": 1, "scene_initializations": self.scene_initializations,
            "phase_resets": self.phase_resets, "native_pid": os.getpid(), "profile": self.profile,
            "source_sha256": self.source_sha256}
        if "training_support_selection_file" in payload:
            receipt.update(diagnostic_only=True, exact_historical_replay=False,
                           source_sha256=self.source_sha256)
        if raven_pooled:
            self.trainer.env.simulator.sim.pause()
            self._kit_unchanged()
            benchmark = scaled_inputs[3]
            receipt.update(command_id=benchmark['command_id'],
                worker_instance_id=benchmark['owner']['worker_instance_id'],
                kit_identity=self.kit_identity, same_kit=True, physics_pause_requested=True)
        if scaled_teacher:
            self.trainer.env.simulator.sim.pause()
            self._kit_unchanged()
            benchmark = teacher_benchmark
            receipt.update(command_id=benchmark['command_id'],
                worker_instance_id=benchmark['owner']['worker_instance_id'],
                kit_identity=self.kit_identity, same_kit=True, physics_pause_requested=True)
        atomic_write_json(output / "native-phase-receipt.json", receipt, mode=0o644)
        if not (raven_pooled or scaled_teacher):
            self._kit_unchanged()
            self.trainer.env.simulator.sim.pause()
        self.active_output = None
        return receipt


class ResidentIsaacTrainer(NativeDaggerRTXServiceTrainer):
    """The inherited Hydra entrypoint closes AppLauncher only after shutdown."""
    def train(self):
        root = Path(os.environ["RESIDENT_RUNTIME_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        backend = ResidentRavenBackend(self, jobs_root=os.environ["RESIDENT_JOBS_ROOT"],
                                      runtime_root=root, source_sha256=os.environ["RESIDENT_SOURCE_SHA256"])
        self.env.simulator.sim.pause()
        owner = ResidentOwner(root, backend, gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
                              source_sha256=os.environ["RESIDENT_SOURCE_SHA256"])
        serve(owner, port=int(os.environ["RESIDENT_PORT"]), token_file=os.environ["RESIDENT_TOKEN_FILE"])
