"""Explicit scene/reset motion binding and observed episode-purpose receipts.

No task reset, rollout action, camera transform, or physics equation is replaced.
Scene binding is installed before construction. Episode observations wrap native
reset calls and the service's existing episode-ID binding, not the step wire.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.machinery
import inspect
import json
from pathlib import Path
import re

try:
    from .native_artifact_io import canonical, file_sha256, publish_json
except ImportError:
    from native_artifact_io import canonical, file_sha256, publish_json

SCHEMA = "roborender.native-motion-cohort.v1"
EPISODE_SCHEMA = "roborender.native-episode-assignment.v1"


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", value)


def validate_motion_assignment(value):
    keys = {"schema", "run_id", "cohort_id", "purpose", "simulator_seed",
            "motion_inventory_sha256", "lanes"}
    if (not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA
            or not all(_identifier(value[k]) for k in ("run_id", "cohort_id"))
            or value["purpose"] not in {"training", "validation", "test"}
            or type(value["simulator_seed"]) is not int or not 0 <= value["simulator_seed"] < 2**31
            or not _sha(value["motion_inventory_sha256"])):
        raise ValueError("invalid explicit native motion/purpose assignment")
    lanes = value["lanes"]
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 64:
        raise ValueError("explicit B=1..64 native lanes required")
    for index, row in enumerate(lanes):
        if (not isinstance(row, dict) or set(row) != {"lane", "motion_id", "path", "sha256"}
                or type(row["lane"]) is not int or row["lane"] != index
                or not _identifier(row["motion_id"]) or not _sha(row["sha256"])
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts):
            raise ValueError("native motion lanes must be explicit, unique and ordered")
    for key in ("motion_id", "path", "sha256"):
        if len({row[key] for row in lanes}) != len(lanes):
            raise ValueError("a native cohort must not duplicate a motion")
    return deepcopy(value)


def load_motion_assignment(path, raw_sha256):
    path = Path(path)
    if not _sha(raw_sha256) or path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError("a bounded regular assignment file and raw SHA256 are required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError("native motion assignment changed")
    return validate_motion_assignment(json.loads(raw))


def assignment_from_bc_request(request, motion_inventory_sha256):
    return validate_motion_assignment({"schema": SCHEMA, "run_id": request["run_id"],
        "cohort_id": request["cohort_id"], "purpose": {"train": "training"}.get(request["split"], request["split"]),
        "simulator_seed": request["process_seed"], "motion_inventory_sha256": motion_inventory_sha256,
        "lanes": [{"lane": i, **row} for i, row in enumerate(request["motions"])]})


def _reset_inventory(motions, motion_inventory, lane_offset):
    if type(lane_offset) is not int or lane_offset < 0:
        raise ValueError('invalid reset inventory lane offset')
    if motion_inventory is None:
        if lane_offset:
            raise ValueError('offset requires the full reset inventory')
        return motions
    if (not isinstance(motion_inventory, list) or not len(motions) <= len(motion_inventory) <= 10000
            or motion_inventory[lane_offset:lane_offset+len(motions)] != motions
            or any(not isinstance(row, dict) or set(row) != {'motion_id', 'path', 'sha256'}
                   or not _identifier(row['motion_id']) or not _sha(row['sha256'])
                   or not isinstance(row['path'], str) or not Path(row['path']).is_absolute()
                   or '..' in Path(row['path']).parts for row in motion_inventory)
            or any(len({row[key] for row in motion_inventory}) != len(motion_inventory)
                   for key in ('motion_id', 'path', 'sha256'))):
        raise ValueError('full ordered reset inventory must contain the exact lane slice')
    return motion_inventory


def configure_motion_environment(cfg, assignment, *, motion_inventory=None, lane_offset=0):
    value = validate_motion_assignment(assignment)
    motions = [{k: r[k] for k in ('motion_id', 'path', 'sha256')} for r in value['lanes']]
    reset_motions = _reset_inventory(motions, motion_inventory, lane_offset)
    cfg.num_envs = len(value["lanes"])
    cfg.simulator.config.scene.num_envs = cfg.num_envs
    cfg.reset_from_dataset.use_motion_file_dir = False
    cfg.reset_from_dataset.motion_files = [row["path"] for row in reset_motions]
    if motion_inventory is not None:
        cfg.reset_from_dataset.roborender_reset_lane_offset = lane_offset
    # Native task sampling is intentionally not offset or randomized here.
    # Each declared lane already owns its corresponding motion/scene geometry.
    if hasattr(cfg.reset_from_dataset, "motion_assignment_offset"):
        cfg.reset_from_dataset.motion_assignment_offset = 0


def assert_motion_environment(cfg, assignment, *, motion_inventory=None, lane_offset=0):
    value = validate_motion_assignment(assignment)
    motions = [{k: r[k] for k in ('motion_id', 'path', 'sha256')} for r in value['lanes']]
    reset_motions = _reset_inventory(motions, motion_inventory, lane_offset)
    if (int(cfg.num_envs) != len(value["lanes"])
            or int(cfg.simulator.config.scene.num_envs) != len(value["lanes"])
            or cfg.reset_from_dataset.use_motion_file_dir
            or list(cfg.reset_from_dataset.motion_files) != [r["path"] for r in reset_motions]
            or getattr(cfg.reset_from_dataset, 'roborender_reset_lane_offset', None)
               != (lane_offset if motion_inventory is not None else None)
            or int(getattr(cfg.reset_from_dataset, "motion_assignment_offset", 0)) != 0):
        raise ValueError("native reset/scene configuration differs from explicit motion assignment")
    return value


@contextmanager
def explicit_scene_motion_binding(cfg, assignment, scenario_module, motion_loader, *, motion_inventory=None, lane_offset=0):
    """Purpose-aware scene binding; the lower-level hook is shared with BC."""
    value = assert_motion_environment(cfg, assignment, motion_inventory=motion_inventory, lane_offset=lane_offset)
    motions = [{key: row[key] for key in ("motion_id", "path", "sha256")} for row in value["lanes"]]
    with explicit_motion_files_scene_binding(cfg, motions, scenario_module, motion_loader,
            motion_inventory=motion_inventory, lane_offset=lane_offset) as evidence:
        evidence["assignment_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
        yield evidence


@contextmanager
def explicit_motion_files_scene_binding(cfg, motions, scenario_module, motion_loader, *, motion_inventory=None, lane_offset=0):
    """Scope Raven's scenario lookup without inventing an inventory identity.

    BC binds its inventory through the immutable sampling plan; online phase
    requests additionally bind purpose/seed directly. Both use this exact scene
    implementation, leaving reset sampling and physics native. Continuous BC
    additionally supplies the complete ordered reset inventory. Even motions
    without an assigned lane must stay in that loader: its periodic resampling
    consumes random numbers for the entire inventory, as in Raven.
    """
    if (not isinstance(motions, list) or not 1 <= len(motions) <= 64
            or any(not isinstance(row, dict) or set(row) != {"motion_id", "path", "sha256"}
                   or not _identifier(row["motion_id"]) or not _sha(row["sha256"])
                   or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                   or ".." in Path(row["path"]).parts for row in motions)
            or any(len({row[key] for row in motions}) != len(motions) for key in ("motion_id", "path", "sha256"))):
        raise ValueError("invalid ordered native scene motion descriptors")
    paths = [row["path"] for row in motions]
    reset_motions = _reset_inventory(motions, motion_inventory, lane_offset)
    reset_paths = [row["path"] for row in reset_motions]

    def verify_config(config):
        if (int(config.num_envs) != len(motions) or int(config.simulator.config.scene.num_envs) != len(motions)
                or config.reset_from_dataset.use_motion_file_dir
                or list(config.reset_from_dataset.motion_files) != reset_paths
                or int(getattr(config.reset_from_dataset, "motion_assignment_offset", 0)) != 0):
            raise ValueError("native scene and reset motion order differ")

    verify_config(cfg)
    for row in reset_motions:
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("native motion file bytes differ from assignment")
    source = Path(inspect.getsourcefile(scenario_module)).resolve()
    original_exec = importlib.machinery.SourceFileLoader.exec_module
    original_metadata = scenario_module.get_table_grid_meta_data
    evidence = {"schema": "roborender.native-scene-motion-binding.v1",
        "ordered_motion_sha256": hashlib.sha256(canonical(motions)).hexdigest(),
        "queried_lanes": [], "module_bindings": 0}
    if motion_inventory is not None:
        evidence["motion_inventory_sha256"] = hashlib.sha256(canonical(motion_inventory)).hexdigest()
        evidence['reset_lane_offset'] = lane_offset
        evidence['reset_motion_count'] = len(motion_inventory)

    def metadata(config, env_idx, device):
        verify_config(config)
        index = int(env_idx)
        if not 0 <= index < len(motions):
            raise ValueError("scenario requested an undeclared motion lane")
        if index not in evidence["queried_lanes"]:
            evidence["queried_lanes"].append(index)
        return motion_loader.from_file(paths[index], device).table_custom_data

    def bind(loader, module):
        original_exec(loader, module)
        if Path(str(getattr(loader, "path", ""))).resolve() == source:
            module.get_table_grid_meta_data = metadata
            for name in ("table_top_material_prim_paths", "leg_material_prim_paths"):
                setattr(module, name, [str(item) for item in getattr(module, name)])
            evidence["module_bindings"] += 1

    scenario_module.get_table_grid_meta_data = metadata
    importlib.machinery.SourceFileLoader.exec_module = bind
    try:
        yield evidence
        if evidence["module_bindings"] < 1:
            raise RuntimeError("explicit scenario motion hook did not run during native construction")
    finally:
        importlib.machinery.SourceFileLoader.exec_module = original_exec
        scenario_module.get_table_grid_meta_data = original_metadata


def _list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class NativeEpisodeAssignmentRecorder:
    def __init__(self, assignment, output_dir, *, episode_namespace=None):
        self.assignment = validate_motion_assignment(assignment)
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute() or self.output_dir.exists():
            raise ValueError("episode assignment output must be fresh and absolute")
        self.output_dir.mkdir(parents=True)
        count = len(self.assignment["lanes"])
        self.latest_task_reset = [None] * count
        self.episode_resets = [0] * count
        self.bound_resets = [0] * count
        self.episode_assignments = []
        self.by_episode = {}
        if episode_namespace is not None and not _identifier(episode_namespace):
            raise ValueError("invalid production episode namespace")
        self.episode_namespace = episode_namespace

    def observe_task_reset(self, lanes, samples, indices, ranges, rng_sha256):
        if not _sha(rng_sha256) or not len(lanes) == len(samples) == len(indices):
            raise ValueError("native reset sample cardinality/RNG attestation differs")
        if len(ranges) != len(self.assignment["lanes"]):
            raise ValueError("native reset dataset dropped a declared motion")
        for lane, sample, index in zip(lanes, samples, indices):
            # Raven stores inclusive final sample indices. randint excludes that
            # endpoint, but processed_ratio==1 legitimately selects it.
            if (type(lane) is not int or not 0 <= lane < len(ranges)
                    or type(sample) is not int or type(index) is not int or index < 0
                    or not int(ranges[lane][0]) <= sample <= int(ranges[lane][1])):
                raise ValueError("native reset sample does not belong to its declared lane motion")
            self.latest_task_reset[lane] = {"sample_idx": sample, "sample_idx_in_seq": index,
                                           "reset_rng_sha256": rng_sha256}

    def observe_episode_reset(self, lanes):
        for lane in lanes:
            if not 0 <= lane < len(self.episode_resets) or self.latest_task_reset[lane] is None:
                raise ValueError("episode reset lacks a real native task reset sample")
            self.episode_resets[lane] += 1

    def bind(self, lane, step_key, source_assignment):
        fields = step_key.as_dict() if hasattr(step_key, "as_dict") else vars(step_key)
        if (fields["run_id"] != self.assignment["run_id"] or fields["episode_step_index"] != 0
                or type(lane) is not int or not 0 <= lane < len(self.episode_resets)
                or type(source_assignment["use_raytraced"]) is not bool):
            raise ValueError("service episode binding differs from explicit native cohort")
        episode_id = fields["episode_id"]
        if not _identifier(episode_id):
            raise ValueError("invalid actual native service episode identity")
        motion = self.assignment["lanes"][lane]
        receipt = {"schema": EPISODE_SCHEMA, "purpose": self.assignment["purpose"],
            "run_id": fields["run_id"], "episode_id": episode_id,
            "simulator_seed": self.assignment["simulator_seed"], "motion_id": motion["motion_id"],
            "motion_sha256": motion["sha256"],
            "observation_source": "raytraced" if source_assignment["use_raytraced"] else "generated_controls"}
        if episode_id in self.by_episode:
            prior = self.by_episode[episode_id]
            if (prior["receipt"] != receipt or prior["lane"] != lane
                    or prior["native_reset_count"] != self.episode_resets[lane]):
                raise ValueError("changed observation assignment or reused identity for an existing episode")
            return self.by_episode[episode_id]["entry"]
        if self.episode_resets[lane] <= self.bound_resets[lane]:
            raise ValueError("service episode has no newly observed native episode reset")
        path = self.output_dir / (hashlib.sha256(episode_id.encode()).hexdigest() + ".json")
        audit = {"schema": "roborender.native-episode-reset-evidence.v1", "episode_id": episode_id,
            "lane": lane, "native_episode_reset_index": self.episode_resets[lane] - 1,
            "assignment_sha256": hashlib.sha256(canonical(self.assignment)).hexdigest(),
            **self.latest_task_reset[lane]}
        if self.episode_namespace is not None:
            prefix = self.episode_namespace + "-"
            if not episode_id.startswith(prefix + f"episode-{lane:03d}-"):
                raise ValueError("service episode lacks the declared production cohort scope")
            audit.update(service_episode_namespace=self.episode_namespace,
                         local_service_episode_id=episode_id.removeprefix(prefix))
        # Reject a mismatched service scope before publishing either receipt.
        publish_json(path, receipt)
        publish_json(path.with_suffix(".reset.json"), audit)
        entry = {key: receipt[key] for key in ("episode_id", "simulator_seed", "motion_id", "motion_sha256", "observation_source")}
        entry.update(native_assignment_path=str(path), native_assignment_sha256=file_sha256(path))
        self.bound_resets[lane] = self.episode_resets[lane]
        self.episode_assignments.append(entry)
        self.by_episode[episode_id] = {"receipt": receipt, "entry": entry,
            "lane": lane, "native_reset_count": self.episode_resets[lane]}
        return entry


@contextmanager
def install_native_assignment_observer(trainer, assignment, output_dir, rng_fingerprint=None, *, episode_namespace=None,
                                       motion_inventory=None, lane_offset=0):
    """Install before the first native reset, for one complete cohort lifecycle."""
    value = assert_motion_environment(trainer.env.config, assignment, motion_inventory=motion_inventory, lane_offset=lane_offset)
    env = trainer.env
    reset_motions = motion_inventory if motion_inventory is not None else value['lanes']
    if (list(env.reset_dataset.motion_files) != [r["path"] for r in reset_motions]
            or getattr(env.simulator, "native_motion_assignment_sha256", None)
            != hashlib.sha256(canonical(value)).hexdigest()):
        raise ValueError("native scene/reset dataset did not attest the explicit motion cohort")
    if motion_inventory is not None:
        expected = dict(motion_inventory_sha256=hashlib.sha256(canonical(motion_inventory)).hexdigest(),
                        reset_lane_offset=lane_offset, reset_motion_count=len(motion_inventory))
        if getattr(env.simulator, 'native_reset_inventory_binding', None) != expected:
            raise ValueError('native scene did not attest the full reset inventory')
    recorder = NativeEpisodeAssignmentRecorder(value, output_dir, episode_namespace=episode_namespace)
    original_task, original_reset = env._reset_task_states, env.reset_envs_idx
    original_observer = getattr(trainer, "_native_episode_assignment_observer", None)
    if original_observer is not None:
        raise ValueError("native assignment observer already installed")
    if rng_fingerprint is None:
        import torch
        if torch.initial_seed() != value["simulator_seed"]:
            raise ValueError("native RNG seed differs from declared cohort")

        def rng_fingerprint():
            result = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
            result.update(torch.cuda.get_rng_state(env.device).cpu().numpy().tobytes())
            return result.hexdigest()

    def task_reset(lanes, *args, **kwargs):
        digest = rng_fingerprint() if len(lanes) else None
        result = original_task(lanes, *args, **kwargs)
        if len(lanes):
            # Pinned Raven assigns both arrays afresh with len(lanes) elements
            # in reset_from_dataset_wsppt._reset_task_states. They are already
            # subset-ordered; indexing by global lane IDs again is incorrect.
            ranges = env.reset_dataset.sample_idx_based_on_demo_clip_number
            if len(ranges) != len(reset_motions):
                raise ValueError('native reset dataset dropped an inventory motion')
            recorder.observe_task_reset(_list(lanes), _list(env.sample_idx), _list(env.sample_idx_in_seq),
                ranges[lane_offset:lane_offset+len(value['lanes'])], digest)
        return result

    def episode_reset(lanes, *args, **kwargs):
        result = original_reset(lanes, *args, **kwargs)
        recorder.observe_episode_reset(_list(lanes))
        return result

    env._reset_task_states, env.reset_envs_idx = task_reset, episode_reset
    trainer._native_episode_assignment_observer = recorder.bind
    try:
        yield recorder
    finally:
        env._reset_task_states, env.reset_envs_idx = original_task, original_reset
        del trainer._native_episode_assignment_observer
        publish_json(recorder.output_dir / "index.json", {
            "schema": "roborender.native-assignment-index.v1", "assignment": value,
            "episode_assignments": recorder.episode_assignments,
            "actual_episode_resets_per_lane": recorder.episode_resets,
            "receipt_scope": "native observed episode assignments; not a complete dataset seal"})
