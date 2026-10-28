"""One Kit boot, independent fresh native BC scenes and trainers per cohort.

This module is CPU-safe until Hydra requests its trainer class. It does not
resume simulator state: a failed cohort ends the entire session. The native
collector, camera conversions, actions and episode/chunk rules are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

try:
    from .isaac6_bc_production import canonical, publish_json, validate_collection_request
except ImportError:
    from isaac6_bc_production import canonical, publish_json, validate_collection_request

SCHEMA = "roborender.bc-native-session.v1"


def memory_snapshot():
    """Read small Linux counters; missing counters are explicit, not zero."""
    import resource
    result = {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        entry = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / entry.lstrip("/")
        if not (root / "memory.current").is_file():
            root = Path("/sys/fs/cgroup")
        for key in ("current", "peak", "max", "swap.current"):
            raw = (root / ("memory." + key)).read_text().strip()
            result["memory_" + key] = int(raw) if raw != "max" else "max"
        for key in ("stat", "events"):
            values = dict(line.split() for line in (root / ("memory." + key)).read_text().splitlines())
            result[key] = {name: int(value) for name, value in values.items()
                if key == "events" or name in {"anon", "file", "kernel", "shmem"}}
    except (OSError, ValueError, StopIteration) as exc:
        result["sampling_error"] = str(exc)
    return result


def validate_session(value, *, read_bytes=None):
    if (not isinstance(value, dict) or set(value) != {"schema", "session_id", "requests"}
            or value["schema"] != SCHEMA
            or not isinstance(value["session_id"], str)
            or not re.fullmatch(r"bc-session-leo-[a-z0-9-]{1,80}", value["session_id"])
            or not isinstance(value["requests"], list) or not 1 <= len(value["requests"]) <= 300):
        raise ValueError("invalid bounded native BC session")
    reader = read_bytes or (lambda p: Path(p).read_bytes())
    loaded = []
    for row in value["requests"]:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts
                or not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError("invalid request identity")
        raw = reader(row["path"])
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("session request bytes changed")
        loaded.append(validate_collection_request(json.loads(raw)))
        if loaded[-1]['schema'] == 'roborender.bc-native-continuous-request.v1':
            raise ValueError('continuous collection requires its single-scene guarded launcher, not a cohort session')
    first = loaded[0]
    if first["output_dir"] != "/dagger-runtime/bc_isaac":
        raise ValueError("first cohort must use the native launch output")
    for request in loaded[1:]:
        out = Path(request["output_dir"])
        if not out.is_relative_to("/dagger-runtime/cohorts") or len(out.parts) < 5:
            raise ValueError("later cohorts require distinct scoped runtime outputs")
    for key in ("cohort_id", "output_dir"):
        if len({r[key] for r in loaded}) != len(loaded):
            raise ValueError("cohort identity/output may not be replayed")
    for key in ("run_id", "rank", "sampling_plan_sha256"):
        if any(r[key] != first[key] for r in loaded):
            raise ValueError("session cohorts must belong to one pinned collection plan/actor")
    if any(len(r["motions"]) != len(first["motions"]) for r in loaded):
        raise ValueError("native warm sessions require a fixed environment count; short tails need a separate session")
    return deepcopy(value), loaded


@contextmanager
def request_environment(row):
    values = {"ROBORENDER_BC_COLLECTION_REQUEST": row["path"],
              "ROBORENDER_BC_COLLECTION_REQUEST_SHA256": row["sha256"]}
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_session(session, requests, *, run_cohort, rebuild_scene, release_scene,
                    kit_identity, report, output, sample_memory=lambda: {}):
    """Testable lifecycle coordinator. No retry or partial-cohort promotion."""
    initial_kit = kit_identity()
    if len(session["requests"]) != len(requests) or not requests:
        raise ValueError("session/request cardinality differs")
    completed = []
    scene_initializations = 1
    started = time.monotonic()
    failure = None
    try:
        for index, (row, request) in enumerate(zip(session["requests"], requests)):
            if kit_identity() != initial_kit:
                raise RuntimeError("native Kit identity changed")
            with request_environment(row):
                if index:
                    report("cohort_scene_rebuild_start", cohort_index=index, cohort_id=request["cohort_id"])
                    release_scene()
                    rebuild_scene(request)
                    scene_initializations += 1
                    report("cohort_scene_rebuild_complete", cohort_index=index, native_initializations=1)
                report("cohort_collection_start", cohort_index=index, cohort_id=request["cohort_id"])
                result = run_cohort(request)
                if (result.get("status") != "complete" or result.get("request_sha256") != row["sha256"]
                        or result.get("optimizer_steps") != 0
                        or result.get("selected_chunks") != sum(request["chunks_per_env"])):
                    raise RuntimeError("native BC cohort did not complete its exact quota")
                completed.append({"cohort_id": request["cohort_id"], "request_sha256": row["sha256"],
                    "output_dir": request["output_dir"], "selected_chunks": result["selected_chunks"],
                    "vector_steps": result["vector_steps"], "memory": sample_memory()})
                report("cohort_collection_complete", cohort_index=index, **completed[-1])
                if kit_identity() != initial_kit:
                    raise RuntimeError("native Kit identity changed during collection")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        publish_json(output, {"schema": "roborender.bc-native-session-result.v1",
            "session_id": session["session_id"], "status": "complete" if failure is None else "failed",
            "native_initializations": 1, "scene_initializations": scene_initializations,
            "cohorts": completed, "failure": failure, "elapsed_seconds": time.monotonic() - started,
            "production_admitted": False})


def _native_trainer():
    import gc
    import random
    import numpy as np
    import torch
    import omni.kit.app
    from isaaclab.sim import SimulationContext
    from omegaconf import OmegaConf
    from dagger_rtx.process_memory import compact_host_heap
    from isaac6_bc_production import Isaac6BCProductionTrainer

    class Isaac6BCSessionTrainer:
        def __init__(self, **kwargs):
            raw = Path(os.environ["ROBORENDER_BC_SESSION"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != os.environ["ROBORENDER_BC_SESSION_SHA256"]:
                raise ValueError("session bytes differ from immutable launch")
            self.session, self.requests = validate_session(json.loads(raw))
            first = self.session["requests"][0]
            if (os.environ.get("ROBORENDER_BC_COLLECTION_REQUEST") != first["path"]
                    or os.environ.get("ROBORENDER_BC_COLLECTION_REQUEST_SHA256") != first["sha256"]):
                raise ValueError("first native scene request differs from session")
            if (kwargs.get("checkpoint") is not None or kwargs.get("optimizers", (None, None)) != (None, None)
                    or kwargs.get("train_dataset") is not None or kwargs.get("eval_dataset") is not None
                    or kwargs["accelerator"].num_processes != 1):
                raise ValueError("session requires fresh, single-process, collection-only trainers")
            self.accelerator = kwargs["accelerator"]
            self.env = kwargs["env"]
            self.env_type = type(self.env)
            self.device = str(self.env.device)
            if self.device == "cuda":
                self.device = "cuda:0"
            if self.device != "cuda:0" or torch.cuda.device_count() != 1:
                raise RuntimeError("session must expose exactly one pinned native GPU")
            self.env_template = OmegaConf.create(OmegaConf.to_container(self.env.config, resolve=True))
            self.template = {k: deepcopy(v) for k, v in kwargs.items()
                if k not in {"env", "accelerator", "model", "ref_model", "value_model", "reward_model"}}
            self.models = {k: deepcopy(kwargs.get(k)).cpu() if kwargs.get(k) is not None else None
                for k in ("model", "ref_model", "value_model", "reward_model")}
            # Initial trainer is constructed exactly through Raven's constructor.
            self.trainer = Isaac6BCProductionTrainer(**kwargs)

        def _release(self):
            old_env = self.env
            self.env = None
            self.trainer = None
            self.accelerator.free_memory()
            old_env.simulator.sim.stop()
            SimulationContext.clear_instance()
            # Raven's entrypoint retains its original `env` local. It is never
            # used after trainer.train(). Retire only after public context
            # cleanup; leave scalar identity/config for postmortem inspection.
            # Native Raven env classes have no destructor or subscriptions;
            # camera/PhysX subscription cleanup belongs to the context above.
            for key in list(vars(old_env)):
                if key not in {"config", "device", "num_envs"}:
                    delattr(old_env, key)
            del old_env
            gc.collect()
            torch.cuda.empty_cache()
            compact_host_heap("bc_cohort_scene_released")

        def _rebuild(self, request):
            seed = request["process_seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cfg = deepcopy(self.env_template)
            OmegaConf.set_struct(cfg, False)
            cfg.num_envs = len(request["motions"])
            cfg.simulator.config.scene.num_envs = cfg.num_envs
            cfg.simulator._target_ = "isaac6_bc_production.IsaacSim"
            cfg.reset_from_dataset.use_motion_file_dir = False
            cfg.reset_from_dataset.motion_files = [r["path"] for r in request["motions"]]
            self.env = self.env_type(cfg, self.device)
            kwargs = deepcopy(self.template)
            kwargs.update(env=self.env, accelerator=self.accelerator,
                **{k: deepcopy(v).to(self.device) if v is not None else None for k, v in self.models.items()})
            kwargs["local_seed"] = seed
            kwargs["args"].seed = seed
            kwargs["args"].num_total_batches = request["max_windows"]
            kwargs["args"].output_dir = str(Path(request["output_dir"]).parent / "trainer")
            kwargs["log_dir"] = Path(request["output_dir"]).parent / "trainer"
            kwargs["config"].videogen.collect_dir = request["output_dir"]
            self.trainer = Isaac6BCProductionTrainer(**kwargs)

        def train(self):
            def report(phase, **values):
                print(json.dumps({"phase": phase, "monotonic_seconds": time.monotonic(), **values}), flush=True)

            def run(request):
                # Match the qualified native service's one-time reclamation
                # of unused startup heap; no tensors or simulator state change.
                compact_host_heap("bc_cohort_setup_complete")
                # Read actual USD geometry, not only the requested motion path.
                # Do not import the evaluator: its global scenario hook must
                # never alter this native collection scene.
                import omni.usd
                from pxr import UsdGeom
                stage = omni.usd.get_context().get_stage()
                lanes = []
                for lane in range(len(request["motions"])):
                    prim = stage.GetPrimAtPath(f"/World/envs/env_{lane}/table_grid")
                    if not prim.IsValid():
                        raise RuntimeError("native cohort table geometry is missing")
                    metadata = dict(prim.GetCustomData())
                    tables = []
                    for table in prim.GetChildren():
                        top = table.GetChild("table_top")
                        if not top.IsValid():
                            continue
                        matrix = UsdGeom.Xformable(table).GetLocalTransformation()
                        top_matrix = UsdGeom.Xformable(top).GetLocalTransformation()
                        tables.append({"name": table.GetName(),
                            "local_transform": [[float(x) for x in row] for row in matrix],
                            "top_local_transform": [[float(x) for x in row] for row in top_matrix],
                            "top_size": float(UsdGeom.Cube(top).GetSizeAttr().Get())})
                    if not tables:
                        raise RuntimeError("native cohort has no table-top geometry")
                    lanes.append({"lane": lane, "motion_id": request["motions"][lane]["motion_id"],
                        "custom_data": metadata, "tables": tables})
                geometry = {"schema": "roborender.bc-native-scene-geometry.v1", "lanes": lanes,
                    "sha256": hashlib.sha256(canonical(lanes)).hexdigest()}
                publish_json(Path(request["output_dir"]).parent / "scene-geometry.json", geometry)
                return self.trainer.train()

            return execute_session(self.session, self.requests,
                run_cohort=run, rebuild_scene=self._rebuild,
                release_scene=self._release, kit_identity=lambda: id(omni.kit.app.get_app()),
                report=report, sample_memory=memory_snapshot,
                output=Path("/dagger-runtime/bc-session-result.json"))

    return Isaac6BCSessionTrainer


def __getattr__(name):
    if name != "Isaac6BCSessionTrainer":
        raise AttributeError(name)
    trainer = _native_trainer()
    globals()[name] = trainer
    return trainer
