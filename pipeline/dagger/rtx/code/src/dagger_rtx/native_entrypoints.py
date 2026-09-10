"""Concrete immutable-source Raven callbacks.

The pinned Raven snapshot exposes several complete methods, but its
teacher/Homie prefix and raw-depth archival statements are embedded in larger
methods.  This module executes those exact source nodes behind full-file and
AST hashes.  It never copies their numerical expressions into this project.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import BackendInvariantError, TransitionResult
from .contracts import PINNED_RAVEN_COMMIT, PINNED_RAVEN_ROOT, RAVEN_VIEW_ORDER, StepKey
from .native_raven import (
    NATIVE_CALLBACK_ORIGINS,
    NativeActionLabels,
    NativeCurrentBatch,
    NativeDepthArchive,
    NativeRavenEntrypoints,
)


RAVEN_ROOT = Path(os.environ.get("DAGGER_RAVEN_RUNTIME_ROOT", PINNED_RAVEN_ROOT))
VIDEOGEN_SOURCE = RAVEN_ROOT / (
    "pipeline/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/"
    "distill_trainer_videogen.py"
)
TEACHER_HOMIE_SOURCE = RAVEN_ROOT / (
    "pipeline/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/"
    "distill_trainer_obj_pred_homie_api.py"
)

EXPECTED_SOURCE_SHA256 = {
    VIDEOGEN_SOURCE: "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
    TEACHER_HOMIE_SOURCE: "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584",
}
EXPECTED_RAW_ARCHIVE_AST_SHA256 = {
    "vg_save_raw_depth_native": "795fb4ad19b66b98a04b2f1b5ed3c99d27d1087923a6297fd8ccae940d15a4af",
    "vg_save_raw_depth": "6965f1da58f623611f1f199de9392a6f0ba7b93ddb3fe8505315f52f9f7509e7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function(tree: ast.AST, class_name: str, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise BackendInvariantError(f"pinned AST lacks {class_name}.{function_name}")


def _ast_digest(node: ast.AST) -> str:
    # Python 3.13 changed ``ast.dump`` to omit empty fields by default.  Ask
    # for them explicitly when that option exists so the attestation preimage
    # remains identical to Python 3.10--3.12 and does not depend on which
    # supported interpreter performs the audit.
    dump_kwargs: dict[str, Any] = {"include_attributes": False}
    if "show_empty" in inspect.signature(ast.dump).parameters:
        dump_kwargs["show_empty"] = True
    return hashlib.sha256(ast.dump(node, **dump_kwargs).encode("utf-8")).hexdigest()


def verify_origin_sources() -> dict[str, Any]:
    """Fail if the immutable Raven source or required AST nodes changed."""

    hashes = {}
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise BackendInvariantError(f"pinned Raven source hash mismatch: {path}")
        hashes[str(path)] = actual

    video_tree = ast.parse(VIDEOGEN_SOURCE.read_text(encoding="utf-8"))
    save = _function(video_tree, "TRLDistillTrainerVideoGen", "_save_collected_chunks")
    raw_nodes: dict[str, ast.If] = {}
    for node in ast.walk(save):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Attribute):
            continue
        if node.test.attr in EXPECTED_RAW_ARCHIVE_AST_SHA256:
            raw_nodes[node.test.attr] = node
    if set(raw_nodes) != set(EXPECTED_RAW_ARCHIVE_AST_SHA256):
        raise BackendInvariantError("pinned raw-depth archival AST nodes are missing")
    for name, expected in EXPECTED_RAW_ARCHIVE_AST_SHA256.items():
        if _ast_digest(raw_nodes[name]) != expected:
            raise BackendInvariantError(f"pinned raw-depth AST mismatch: {name}")

    teacher_tree = ast.parse(TEACHER_HOMIE_SOURCE.read_text(encoding="utf-8"))
    policy = _function(
        teacher_tree, "TRLDistillTrainerObjPredHomieAPI", "policy_step"
    )
    calls = {
        node.func.attr
        for node in ast.walk(policy)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if not {"homie_walk_model", "homie_stand_model"}.issubset(
        {
            node.attr
            for node in ast.walk(policy)
            if isinstance(node, ast.Attribute)
        }
    ) or not ({"act_inference", "act"} & calls):
        raise BackendInvariantError("teacher/Homie native prefix lost required calls")
    return {"source_sha256": hashes, "raw_archive_ast_sha256": {
        name: _ast_digest(node) for name, node in raw_nodes.items()
    }}


def _compile_raw_archive_extractor():
    verify_origin_sources()
    tree = ast.parse(VIDEOGEN_SOURCE.read_text(encoding="utf-8"))
    save = _function(tree, "TRLDistillTrainerVideoGen", "_save_collected_chunks")
    nodes: dict[str, ast.If] = {}
    for node in ast.walk(save):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute):
            if node.test.attr in EXPECTED_RAW_ARCHIVE_AST_SHA256:
                nodes[node.test.attr] = copy.deepcopy(node)
    # Execute only the two attested computational nodes.  No I/O node from the
    # surrounding save method is present in the generated function.
    function = ast.FunctionDef(
        name="_origin_raw_archive",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="vdepth"), ast.arg(arg="vname"), ast.arg(arg="view_arrays")],
            kwonlyargs=[], kw_defaults=[], defaults=[], vararg=None, kwarg=None,
        ),
        body=[
            nodes["vg_save_raw_depth_native"],
            nodes["vg_save_raw_depth"],
            ast.Return(value=ast.Name(id="view_arrays", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, Any] = {"np": np}
    exec(compile(module, str(VIDEOGEN_SOURCE), "exec"), namespace, namespace)
    return namespace["_origin_raw_archive"]


class _LabelsReady(BaseException):
    def __init__(self, policy_state: Mapping[str, Any], homie: Any):
        super().__init__("native label prefix complete")
        self.policy_state = policy_state
        self.homie = homie


class _StopBeforeStudent:
    """Sentinel policy that stops the pinned method at its student boundary."""

    def __init__(self, origin_code: Any):
        self.origin_code = origin_code

    def rollout_with_obj_pred(self, **_kwargs):
        caller = inspect.currentframe().f_back
        if caller is None or caller.f_code is not self.origin_code:
            raise BackendInvariantError("label sentinel called outside pinned policy_step")
        state = caller.f_locals.get("policy_state_dict")
        homie = caller.f_locals.get("homie_actions")
        if not isinstance(state, Mapping) or "gt_actions" not in state or homie is None:
            raise BackendInvariantError("pinned teacher/Homie prefix produced no labels")
        raise _LabelsReady(state, homie)


@dataclass
class NativeRavenRuntime:
    """Mutable cursor held only inside the initialized Raven process."""

    trainer: Any
    obs_dict: Mapping[str, Any]
    cur_dones: Any
    device: Any
    seqs: np.ndarray

    @classmethod
    def initialize(cls, trainer: Any) -> "NativeRavenRuntime":
        """Run Raven's native reset/init calls without an environment step."""

        verify_origin_sources()
        config = getattr(trainer, "config", {})
        if bool(config.get("init_at_random_ep_len", False)):
            raise BackendInvariantError(
                "native DAgger service requires init_at_random_ep_len=false"
            )
        import torch

        trainer._train_rollout_mode()
        # Policy/ref-model recurrent lifecycle belongs to the 81-step rollout
        # owner.  Initializing either here would create an unmatched second
        # initialization outside that begin/clear boundary.
        obs_dict = trainer.env.reset_all()
        device = trainer.accelerator.device
        obs_dict = {name: value.to(device) for name, value in obs_dict.items()}
        cur_dones = torch.zeros(trainer.env.num_envs, device=device)
        return cls(
            trainer=trainer,
            obs_dict=obs_dict,
            cur_dones=cur_dones,
            device=device,
            seqs=np.zeros(trainer.env.num_envs, dtype=np.int64),
        )


def _to_numpy(value: Any, name: str) -> np.ndarray:
    try:
        result = value.detach().to("cpu").numpy()
    except Exception as exc:
        raise BackendInvariantError(f"native {name} is not a tensor") from exc
    return np.ascontiguousarray(result)


def _find_teacher_policy_method(trainer: Any):
    for cls in type(trainer).__mro__:
        if cls.__name__ == "TRLDistillTrainerObjPredHomieAPI" and "policy_step" in cls.__dict__:
            method = cls.__dict__["policy_step"]
            source = Path(inspect.getsourcefile(method) or "").resolve()
            if source != TEACHER_HOMIE_SOURCE.resolve():
                raise BackendInvariantError("teacher/Homie method came from an unpinned path")
            return method
    raise BackendInvariantError("initialized trainer lacks pinned teacher/Homie policy_step")


def _find_video_save_method(trainer: Any):
    for cls in type(trainer).__mro__:
        if cls.__name__ == "TRLDistillTrainerVideoGen" and "_save_collected_chunks" in cls.__dict__:
            method = cls.__dict__["_save_collected_chunks"]
            source = Path(inspect.getsourcefile(method) or "").resolve()
            if source != VIDEOGEN_SOURCE.resolve():
                raise BackendInvariantError("raw-depth method came from an unpinned path")
            return method
    raise BackendInvariantError("initialized trainer lacks pinned video trainer")


def build_origin_attested_entrypoints(
    trainer: Any, runtime: NativeRavenRuntime
) -> NativeRavenEntrypoints:
    """Bind all concrete callbacks to one initialized certified trainer."""

    verify_origin_sources()
    policy_method = _find_teacher_policy_method(trainer)
    _find_video_save_method(trainer)  # attest the source owning the extracted nodes
    raw_extractor = _compile_raw_archive_extractor()

    def prepare_episode_slots(native_trainer: Any, batch: int) -> None:
        if native_trainer is not trainer or batch != trainer.env.num_envs:
            raise BackendInvariantError("episode slot callback trainer/batch mismatch")
        if native_trainer._id_anchors is None:
            native_trainer._id_anchors = [None] * batch
            native_trainer._id_prompts = [None] * batch
        if len(native_trainer._id_anchors) != batch or len(native_trainer._id_prompts) != batch:
            raise BackendInvariantError("existing native episode slot count mismatch")
        # Do not sample here.  NativeRavenBatchComposition calls the retained
        # sampler exactly once for each newly bound lane.

    def read_current_batch(keys: Sequence[StepKey]) -> NativeCurrentBatch:
        if len(keys) != trainer.env.num_envs:
            raise BackendInvariantError("current key batch does not match env.num_envs")
        expected = ("actor_obs", "teacher_obs", "critic_obs", "homie_obs", "gt_obj_pos")
        arrays: dict[str, np.ndarray] = {}
        for name in expected:
            if name not in runtime.obs_dict:
                raise BackendInvariantError(f"native obs_dict lacks {name}")
            value = _to_numpy(runtime.obs_dict[name], name).astype(np.float32, copy=False)
            if name == "gt_obj_pos" and value.ndim == 2 and value.shape[1] >= 3:
                value = value[:, :3]
            arrays[name] = np.ascontiguousarray(value)
        return NativeCurrentBatch(
            keys=tuple(keys),
            obs_dict=runtime.obs_dict,
            cur_dones=runtime.cur_dones,
            arrays=arrays,
            env_ids=np.arange(trainer.env.num_envs, dtype=np.int64),
            seqs=runtime.seqs.copy(),
        )

    def teacher_homie_labels(
        native_trainer: Any, current: NativeCurrentBatch
    ) -> NativeActionLabels:
        if native_trainer is not trainer or current.obs_dict is not runtime.obs_dict:
            raise BackendInvariantError("label callback is not using current native cursor")
        if getattr(trainer, "export_homie_model", False):
            raise BackendInvariantError("export_homie_model must be disabled in DAgger service")
        import torch

        sentinel = _StopBeforeStudent(policy_method.__code__)
        try:
            with torch.no_grad():
                policy_method(trainer, sentinel, runtime.obs_dict, cur_dones=runtime.cur_dones)
        except _LabelsReady as stop:
            teacher = _to_numpy(stop.policy_state["gt_actions"], "gt_actions")
            homie = _to_numpy(stop.homie, "homie_actions")
            return NativeActionLabels(
                teacher_action=np.ascontiguousarray(teacher, dtype=np.float32),
                homie_action=np.ascontiguousarray(homie, dtype=np.float32),
            )
        raise BackendInvariantError("pinned teacher/Homie method crossed into student execution")

    def native_depth_archive(
        native_trainer: Any, views: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, NativeDepthArchive]:
        if native_trainer is not trainer:
            raise BackendInvariantError("raw archive callback trainer mismatch")
        if set(RAVEN_VIEW_ORDER) - set(views):
            raise BackendInvariantError("raw archive views incomplete")
        facade = SimpleNamespace(
            vg_save_raw_depth_native=True,
            vg_save_raw_depth=True,
            vg_control_res=list(trainer.vg_control_res),
        )
        result: dict[str, NativeDepthArchive] = {}
        for view in RAVEN_VIEW_ORDER:
            import torch

            depth = views[view]["depth"].detach().to(device="cpu", dtype=torch.float32)
            arrays = raw_extractor(facade, depth[None], view, {})
            result[view] = NativeDepthArchive(
                depth_rawq=np.ascontiguousarray(arrays[f"depth_rawq__{view}"][0]),
                depth_raw=np.ascontiguousarray(arrays[f"depth_raw__{view}"][0]),
            )
        return result

    def build_action_state(
        native_trainer: Any, selected: np.ndarray, teachers: np.ndarray
    ) -> Mapping[str, Any]:
        if native_trainer is not trainer:
            raise BackendInvariantError("action callback trainer mismatch")
        import torch

        actions_t = torch.as_tensor(selected.copy(), device=runtime.device, dtype=torch.float32)
        teachers_t = torch.as_tensor(teachers.copy(), device=runtime.device, dtype=torch.float32)
        return {"actions": actions_t, "gt_actions": teachers_t}

    def decode_env_step(native_trainer: Any, raw: Any) -> Sequence[TransitionResult]:
        if native_trainer is not trainer or not isinstance(raw, tuple) or len(raw) != 4:
            raise BackendInvariantError("native env.step result is malformed")
        obs_dict, rewards, dones, infos = raw
        obs_dict = {name: value.to(runtime.device) for name, value in obs_dict.items()}
        rewards = rewards.to(runtime.device)
        dones = dones.to(runtime.device)
        if not isinstance(infos, Mapping) or "time_outs" not in infos:
            raise BackendInvariantError("native env.step infos lacks time_outs")
        timeouts = infos["time_outs"].to(runtime.device)
        # Exact Raven recurrent-policy/teacher reset hook; env.step already did
        # the physical episode reset and must never be called/reset again here.
        trainer._process_env_step(rewards, dones, infos)
        runtime.obs_dict = obs_dict
        runtime.cur_dones = dones
        rewards_np = _to_numpy(rewards, "rewards").reshape(-1)
        dones_np = _to_numpy(dones, "dones").reshape(-1)
        timeouts_np = _to_numpy(timeouts, "time_outs").reshape(-1)
        ended = np.logical_or(dones_np.astype(bool), timeouts_np.astype(bool))
        runtime.seqs[ended] += 1
        return tuple(
            TransitionResult(
                reward=float(rewards_np[index]),
                done=bool(dones_np[index]),
                timeout=bool(timeouts_np[index]),
                metadata={"native_env_index": index},
            )
            for index in range(trainer.env.num_envs)
        )

    return NativeRavenEntrypoints(
        raven_root=PINNED_RAVEN_ROOT,
        raven_commit=PINNED_RAVEN_COMMIT,
        origins=dict(NATIVE_CALLBACK_ORIGINS),
        prepare_episode_slots=prepare_episode_slots,
        read_current_batch=read_current_batch,
        teacher_homie_labels=teacher_homie_labels,
        native_depth_archive=native_depth_archive,
        build_action_state=build_action_state,
        decode_env_step=decode_env_step,
    )
