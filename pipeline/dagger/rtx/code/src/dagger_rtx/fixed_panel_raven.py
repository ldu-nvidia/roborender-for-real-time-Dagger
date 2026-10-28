"""Native Raven adapter for sealed fixed-panel success validation.

This module is imported only inside the certified Isaac/Raven container.  It
keeps the pinned checkout read-only, applies the proven exactly-once three-view
normalization, executes every paired seed as one native B=64 cohort, and wraps
Raven's stock deterministic ``action_mean`` evaluation. The versioned accounting
mixin fixes terminal-frame success export without changing native goal/reset
predicates or physics.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import inspect
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dagger_rtx.fixed_panel_validation import (
    ACTION_POLICY,
    GENERATED_OBSERVATION_MODE,
    EXPECTED_VIEWS,
    OBSERVATION_MODE,
    TEACHER_ACTION_POLICY,
    AuthenticatedEvaluationClient,
    COHORT_ENVIRONMENTS,
    DurableOutcomeJournal,
    EpisodeLatch,
    FixedPanelCohort,
    FixedPanelValidationError,
    attest_runtime_reset,
    assignment_from_environment,
    calibration_from_environment,
    file_sha256,
)
from dagger_rtx.durable import atomic_write_json
from dagger_rtx.terminal_goal_accounting import (
    GOAL_ACCOUNTING_VERSION,
    TerminalGoalAccountingMixin,
)

from gr00t.rl.envs.loco_manip.walk_stand_place_grasp_turn_homie import (
    WalkStandPlaceGraspTurnHomie as _PinnedWalkStandPlaceGraspTurnHomie,
)
from gr00t.rl.simulator.isaacsim.isaacsim import IsaacSim as _PinnedIsaacSim
from gr00t.rl.trl.trainer.distill_trainer_videogen import TRLDistillTrainerVideoGen
from gr00t.rl.trl.trainer.distill_trainer import TRLDistillTrainer
from gr00t.rl.utils.teleop_reset_lib.motion_loader_wsppt import MotionLoaderWSPPT
from gr00t.rl.data.tasks.walk_stand_place_grasp_turn_homie.scenario_cfg import (
    isaacsim as _scenario_isaacsim,
)
from gr00t.rl.isaac_utils.playground.utils import usd_utils as _usd_utils

# Preserve the beta/alpha runtime fix: pxr.Vt.StringArray is not deepcopyable
# by IsaacLab configclass. The native config field is an ordered list of paths.
_scenario_isaacsim.table_top_material_prim_paths = [
    str(value) for value in _scenario_isaacsim.table_top_material_prim_paths
]
_scenario_isaacsim.leg_material_prim_paths = [
    str(value) for value in _scenario_isaacsim.leg_material_prim_paths
]
_ORIGINAL_PRELOAD_MATERIALS = _usd_utils.preload_materials


def _preload_material_paths(*args: Any, **kwargs: Any) -> list[str]:
    return [str(value) for value in _ORIGINAL_PRELOAD_MATERIALS(*args, **kwargs)]


_usd_utils.preload_materials = _preload_material_paths


PINNED_ISAACSIM_SHA256 = "1d03ebca26f8c5f64d65d2146e79bbf8ceab6af2b83dfb65afa690b784b1073f"
PINNED_RESET_SHA256 = "d42185ef7535e193bb5affcda6e7ad2fbac92e1c478feb9a846ed455f476ac94"
PINNED_SCENARIO_SHA256 = "1a1531fcedd29c4ef6f89b514d59794084afb8fabc0a02589adb61d0ef78c473"
PINNED_STOCK_EVAL_SHA256 = "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584"
PINNED_TEACHER_LOADER_SHA256 = "dc386dcf8b2be0c151403fd0e8841129c939d99dc5e9e7850e92829b3ad50ca2"
PINNED_HOMIE_WALK_PATH = Path(
    "/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/models/model_walk.pt"
)
PINNED_HOMIE_WALK_SIZE = 9_785_586
PINNED_HOMIE_WALK_SHA256 = "2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e"
PINNED_HOMIE_STAND_PATH = Path(
    "/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/models/model_stand.pt"
)
PINNED_HOMIE_STAND_SIZE = 9_785_586
PINNED_HOMIE_STAND_SHA256 = "ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054"
EXPECTED_POLICY_TENSOR_COUNT = 139
EXPECTED_POLICY_NUMEL = 11_615_725
EXPECTED_STUDENT_ACTIONS = 11
EXPECTED_HOMIE_ACTIONS = 15
EXPECTED_EXECUTED_ACTIONS = 26
PINNED_PRIVILEGED_TEACHER_SIZE = 6_805_725
PINNED_PRIVILEGED_TEACHER_SHA256 = (
    "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"
)
MOTION_INVENTORY_SCHEMA = "roborender.raven_success_validation_motion_inventory.v1"


def _source_sha(value: Any) -> str:
    path = Path(inspect.getsourcefile(value) or "").resolve(strict=True)
    return file_sha256(path)


def _attest_pinned_sources() -> dict[str, str]:
    from gr00t.rl.envs.loco_manip import reset_from_dataset_wsppt
    from gr00t.rl.trl.trainer import distill_trainer_obj_pred_homie_api
    from gr00t.rl.trl.trainer import distill_trainer

    sources = {
        "isaacsim": (_PinnedIsaacSim, PINNED_ISAACSIM_SHA256),
        "reset": (reset_from_dataset_wsppt, PINNED_RESET_SHA256),
        "scenario": (_scenario_isaacsim, PINNED_SCENARIO_SHA256),
        "stock_eval": (distill_trainer_obj_pred_homie_api, PINNED_STOCK_EVAL_SHA256),
        "teacher_loader": (distill_trainer, PINNED_TEACHER_LOADER_SHA256),
    }
    result = {}
    for name, (value, expected) in sources.items():
        path = Path(inspect.getsourcefile(value) or value.__file__).resolve(strict=True)
        actual = file_sha256(path)
        if actual != expected:
            raise FixedPanelValidationError(
                f"pinned Raven {name} source hash mismatch: {actual} != {expected}"
            )
        result[str(path)] = actual
    return result


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise FixedPanelValidationError(f"required environment variable {name} is unset")
    return value


@lru_cache(maxsize=1)
def _load_motion_inventory() -> tuple[Path, ...]:
    assignment = assignment_from_environment()
    path = Path(_required_environment("DAGGER_FIXED_PANEL_MOTION_INVENTORY")).resolve(
        strict=True
    )
    if path.is_symlink() or not path.is_file():
        raise FixedPanelValidationError("motion inventory must be a regular non-symlink file")
    expected = _required_environment("DAGGER_FIXED_PANEL_MOTION_INVENTORY_SHA256")
    if file_sha256(path) != expected:
        raise FixedPanelValidationError("motion inventory SHA-256 mismatch")
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping) or set(value) != {"schema", "motions"}:
        raise FixedPanelValidationError("motion inventory keys differ")
    if value["schema"] != MOTION_INVENTORY_SCHEMA or not isinstance(value["motions"], list):
        raise FixedPanelValidationError("motion inventory schema differs")
    paths = []
    names = []
    for row in value["motions"]:
        if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
            raise FixedPanelValidationError("motion inventory row keys differ")
        motion = Path(row["path"]).resolve(strict=True)
        if motion.is_symlink() or not motion.is_file() or motion.name != Path(row["path"]).name:
            raise FixedPanelValidationError("motion input is not a regular absolute file")
        if motion.stat().st_size != row["size_bytes"] or file_sha256(motion) != row["sha256"]:
            raise FixedPanelValidationError(f"motion identity mismatch: {motion}")
        paths.append(motion)
        names.append(motion.name)
    if tuple(names) != assignment.motion_groups:
        raise FixedPanelValidationError(
            "motion inventory order differs from the sealed panel group order"
        )
    return tuple(paths)


def _require_zero_motion_offset(config: Any) -> None:
    reset = config.reset_from_dataset
    value = reset.get("motion_assignment_offset", None)
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise FixedPanelValidationError(
            "one-process B=64 validation requires motion_assignment_offset=0"
        )


_ORIGINAL_TABLE_METADATA = _scenario_isaacsim.get_table_grid_meta_data


def _fixed_panel_table_grid_meta_data(env_config: Any, env_idx: int, sim_device: Any):
    """Bind table geometry to the native cyclic held-out motion for this lane."""

    paths = _load_motion_inventory()
    _require_zero_motion_offset(env_config)
    motion = paths[int(env_idx) % len(paths)]
    return MotionLoaderWSPPT.from_file(str(motion), sim_device).table_custom_data


# ``get_table_grid_spawner_cfg`` resolves this module global at call time.  The
# patch is process-local and the source checkout remains byte-identical.
_scenario_isaacsim.get_table_grid_meta_data = _fixed_panel_table_grid_meta_data


class FixedPanelWalkStandPlaceGraspTurnHomie(
    TerminalGoalAccountingMixin, _PinnedWalkStandPlaceGraspTurnHomie
):
    """Pinned environment preserving Raven's native B=64 lane/RNG semantics."""

    def __init__(self, config: Any, device: Any):
        _attest_pinned_sources()
        expected = tuple(str(path) for path in _load_motion_inventory())
        reset = config.reset_from_dataset
        configured = tuple(str(Path(path).resolve()) for path in reset.motion_files)
        if reset.get("use_motion_file_dir", True) or configured != expected:
            raise FixedPanelValidationError(
                "fixed-panel reset must use the exact ordered motion_files inventory"
            )
        _require_zero_motion_offset(config)
        super().__init__(config, device)


class NormalizedIsaacSim(_PinnedIsaacSim):
    """Proven exact-once ImageNet normalization for the three-view RGB stack."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Raven reloads the scenario from its file under a temporary module
        # name. Bind that hash-attested copy, not only the imported module.
        scenario_path = Path(inspect.getsourcefile(_scenario_isaacsim) or "").resolve(strict=True)
        original_exec_module = importlib.machinery.SourceFileLoader.exec_module
        bound = False

        def exec_module(loader: Any, module: Any) -> None:
            nonlocal bound
            original_exec_module(loader, module)
            if Path(str(getattr(loader, "path", ""))).resolve() != scenario_path:
                return
            if file_sha256(scenario_path) != PINNED_SCENARIO_SHA256:
                raise FixedPanelValidationError("dynamically loaded Raven scenario source hash mismatch")
            module.get_table_grid_meta_data = _fixed_panel_table_grid_meta_data
            module.table_top_material_prim_paths = [str(value) for value in module.table_top_material_prim_paths]
            module.leg_material_prim_paths = [str(value) for value in module.leg_material_prim_paths]
            bound = True

        importlib.machinery.SourceFileLoader.exec_module = exec_module
        try:
            super().__init__(*args, **kwargs)
        finally:
            importlib.machinery.SourceFileLoader.exec_module = original_exec_module
        if not bound:
            raise FixedPanelValidationError("Raven did not load the hash-attested fixed-panel scenario")

    def get_policy_view_stack(self):
        raw = super().get_policy_view_stack()
        if raw is None:
            return None
        calls = int(getattr(self, "_three_view_normalization_calls", 0)) + 1
        self._three_view_normalization_calls = calls
        cameras = self.simulator_config.cameras
        views = tuple(str(value) for value in cameras.policy_views)
        if views != EXPECTED_VIEWS:
            raise FixedPanelValidationError(f"unexpected policy view order: {views!r}")
        if raw.ndim != 4 or tuple(raw.shape[1:]) != (324, 192, 3):
            raise FixedPanelValidationError(
                f"raytraced policy RGB must be [B,324,192,3], got {tuple(raw.shape)}"
            )
        mean = torch.as_tensor(cameras.image_mean, device=raw.device, dtype=raw.dtype)
        std = torch.as_tensor(cameras.image_std, device=raw.device, dtype=raw.dtype)
        if tuple(mean.shape) != (3,) or tuple(std.shape) != (3,) or not bool(torch.all(std > 0)):
            raise FixedPanelValidationError("invalid camera normalization vectors")
        if calls == 1:
            finite = bool(torch.isfinite(raw).all().item())
            raw_min = float(raw.min().item())
            raw_max = float(raw.max().item())
            if not finite or raw_min < -1e-6 or raw_max > 1.0 + 1e-6:
                raise FixedPanelValidationError(
                    "policy view stack is not raw RGB01; refusing double normalization"
                )
            self._three_view_normalization_first_raw = {
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "min": raw_min,
                "max": raw_max,
                "mean": float(raw.float().mean().item()),
            }
        normalized = (raw - mean) / std
        if calls == 1:
            self._three_view_normalization_first_output = {
                "shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "min": float(normalized.min().item()),
                "max": float(normalized.max().item()),
                "mean": float(normalized.float().mean().item()),
            }
        return normalized


# Raven initializes SimulationApp only when the configured target ends in
# IsaacSim. Retain the normalized subclass under that required public name.
IsaacSim = NormalizedIsaacSim


def _tensor_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        parts = (
            str(name).encode("utf-8"),
            str(value.dtype).encode("ascii"),
            ",".join(str(int(item)) for item in value.shape).encode("ascii"),
            value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"),
        )
        for part in parts:
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _bool_vector(value: Any, batch: int, label: str) -> list[bool]:
    if value is None:
        return [False] * batch
    try:
        array = value.detach().to("cpu").numpy()
    except AttributeError:
        array = np.asarray(value)
    array = np.asarray(array).reshape(-1)
    if array.size != batch:
        raise FixedPanelValidationError(f"{label} width differs from env.num_envs")
    return [bool(item) for item in array]


def _float_vector(value: Any, batch: int, label: str) -> list[float]:
    try:
        array = value.detach().to("cpu").numpy()
    except AttributeError:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size != batch or not bool(np.isfinite(array).all()):
        raise FixedPanelValidationError(f"{label} must be a finite B-vector")
    return [float(item) for item in array]


def _int_vector(value: Any, batch: int, label: str) -> list[int]:
    try:
        array = value.detach().to("cpu").numpy()
    except AttributeError:
        array = np.asarray(value)
    array = np.asarray(array).reshape(-1)
    if array.size != batch or array.dtype.kind not in {"i", "u"}:
        raise FixedPanelValidationError(f"{label} must be an integer B-vector")
    return [int(item) for item in array]


def _attest_homie_assets(config: Any) -> dict[str, dict[str, Any]]:
    expected = {
        "walk": (
            "homie_walk_model_path",
            PINNED_HOMIE_WALK_PATH,
            PINNED_HOMIE_WALK_SIZE,
            PINNED_HOMIE_WALK_SHA256,
        ),
        "stand": (
            "homie_stand_model_path",
            PINNED_HOMIE_STAND_PATH,
            PINNED_HOMIE_STAND_SIZE,
            PINNED_HOMIE_STAND_SHA256,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (config_key, pinned_path, pinned_size, pinned_sha) in expected.items():
        configured = Path(str(config.get(config_key, ""))).resolve(strict=True)
        if (
            configured != pinned_path
            or configured.is_symlink()
            or not configured.is_file()
            or configured.stat().st_size != pinned_size
            or file_sha256(configured) != pinned_sha
        ):
            raise FixedPanelValidationError(f"pinned Homie {name} asset identity mismatch")
        result[name] = {
            "path": str(configured),
            "size_bytes": pinned_size,
            "sha256": pinned_sha,
            "action_dimension": EXPECTED_HOMIE_ACTIONS,
        }
    return result


def _attest_native_reset(
    env: Any,
    cohort: FixedPanelCohort,
    motion_groups: Sequence[str],
) -> dict[str, Any]:
    dataset = getattr(env, "reset_dataset", None)
    bounds_value = getattr(dataset, "sample_idx_based_on_demo_clip_number", None)
    samples_per_motion = getattr(dataset, "num_per_sample", None)
    if bounds_value is None or samples_per_motion is None:
        raise FixedPanelValidationError(
            "Raven reset dataset lacks sample interval attestation fields"
        )
    try:
        bounds = [[int(value[0]), int(value[1])] for value in bounds_value]
    except (TypeError, ValueError, IndexError) as exc:
        raise FixedPanelValidationError("Raven reset sample intervals are malformed") from exc
    if not hasattr(env, "sample_idx") or not hasattr(env, "sample_idx_in_seq"):
        raise FixedPanelValidationError(
            "Raven reset did not expose sample_idx/sample_idx_in_seq"
        )
    return attest_runtime_reset(
        cohort,
        motion_groups=motion_groups,
        motion_sample_bounds=bounds,
        sample_indices=_int_vector(env.sample_idx, COHORT_ENVIRONMENTS, "sample_idx"),
        sample_indices_in_sequence=_int_vector(
            env.sample_idx_in_seq, COHORT_ENVIRONMENTS, "sample_idx_in_seq"
        ),
        samples_per_motion=int(samples_per_motion),
    )


def _require_action_contract(
    trainer: Any,
    action_state: Any,
    *,
    action_policy: str = ACTION_POLICY,
) -> None:
    if not isinstance(action_state, Mapping):
        raise FixedPanelValidationError("Raven action state must be a mapping")
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    policy_dim = int(trainer.policy_model.num_actions)
    walk_dim = int(unwrapped.homie_walk_model.num_actions)
    stand_dim = int(unwrapped.homie_stand_model.num_actions)
    if (
        policy_dim != EXPECTED_STUDENT_ACTIONS
        or walk_dim != EXPECTED_HOMIE_ACTIONS
        or stand_dim != EXPECTED_HOMIE_ACTIONS
        or int(trainer.num_act) != EXPECTED_EXECUTED_ACTIONS
        or policy_dim + walk_dim != EXPECTED_EXECUTED_ACTIONS
    ):
        raise FixedPanelValidationError("Raven 11D student + 15D Homie action contract differs")
    action_mean = action_state.get("action_mean")
    actions = action_state.get("actions")
    if (
        not torch.is_tensor(action_mean)
        or not torch.is_tensor(actions)
        or tuple(action_mean.shape) != (COHORT_ENVIRONMENTS, EXPECTED_EXECUTED_ACTIONS)
        or tuple(actions.shape) != tuple(action_mean.shape)
        or not bool(torch.isfinite(action_mean).all().item())
        or not bool(torch.isfinite(actions).all().item())
        or not torch.equal(actions, action_mean)
    ):
        raise FixedPanelValidationError(
            "executed action must equal deterministic finite action_mean [64,26]"
        )
    if action_policy == TEACHER_ACTION_POLICY:
        teacher = action_state.get("gt_actions")
        if (
            not torch.is_tensor(teacher)
            or tuple(teacher.shape) != tuple(actions.shape)
            or not bool(torch.isfinite(teacher).all().item())
            or not torch.equal(actions, teacher)
            or not torch.equal(action_mean, teacher)
        ):
            raise FixedPanelValidationError(
                "teacher oracle must execute exact deterministic gt_actions [64,26]"
            )
    elif action_policy != ACTION_POLICY:
        raise FixedPanelValidationError("unknown fixed-panel action policy")


class FixedPanelEvalTrainer(TRLDistillTrainerVideoGen):
    """Stock deterministic Raven evaluation plus sealed outcome accounting."""

    fixed_panel_action_policy = ACTION_POLICY

    def load_teacher_actor(self):
        if getattr(self, "ref_model", None) is not None:
            raise FixedPanelValidationError(
                "fixed-panel validation forbids a privileged teacher model"
            )
        logging.getLogger(__name__).info(
            "fixed-panel inference: privileged teacher intentionally disabled"
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        if file_sha256(checkpoint_path) != _required_environment(
            "DAGGER_FIXED_PANEL_CHECKPOINT_SHA256"
        ):
            raise FixedPanelValidationError("candidate checkpoint SHA-256 mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise FixedPanelValidationError("candidate checkpoint must be a mapping")
        if "env_state_dict" in checkpoint:
            raise FixedPanelValidationError(
                "fixed-panel validation rejects checkpoint env_state_dict replay"
            )
        self._fixed_panel_checkpoint_keys = sorted(str(key) for key in checkpoint)
        del checkpoint
        optimizer_loader = self.optimizer.load_state_dict
        scheduler_loader = self.lr_scheduler.load_state_dict
        self.optimizer.load_state_dict = lambda _state: logging.getLogger(__name__).info(
            "fixed-panel inference: optimizer state intentionally not restored"
        )
        self.lr_scheduler.load_state_dict = lambda _state: logging.getLogger(__name__).info(
            "fixed-panel inference: scheduler state intentionally not restored"
        )
        try:
            return super().load_checkpoint(str(checkpoint_path))
        finally:
            self.optimizer.load_state_dict = optimizer_loader
            self.lr_scheduler.load_state_dict = scheduler_loader

    def _attest_policy(self, checkpoint_path: Path) -> dict[str, Any]:
        if file_sha256(checkpoint_path) != _required_environment(
            "DAGGER_FIXED_PANEL_CHECKPOINT_SHA256"
        ):
            raise FixedPanelValidationError("candidate checkpoint SHA-256 mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("policy_state_dict"), Mapping
        ):
            raise FixedPanelValidationError("checkpoint lacks policy_state_dict")
        expected = checkpoint["policy_state_dict"]
        runtime = self.accelerator.unwrap_model(self.model).policy.state_dict()
        if set(expected) != set(runtime):
            raise FixedPanelValidationError("runtime/checkpoint policy key sets differ")
        unequal = []
        max_abs = 0.0
        for name in sorted(expected):
            left = expected[name].detach().cpu()
            right = runtime[name].detach().cpu()
            if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                unequal.append(name)
                if left.shape == right.shape and left.is_floating_point() and right.is_floating_point():
                    max_abs = max(max_abs, float((left.float() - right.float()).abs().max()))
        count = len(expected)
        numel = sum(int(value.numel()) for value in expected.values())
        if unequal or count != EXPECTED_POLICY_TENSOR_COUNT or numel != EXPECTED_POLICY_NUMEL:
            raise FixedPanelValidationError(
                f"policy tensor equality gate failed: count={count} numel={numel} unequal={unequal[:8]}"
            )
        expected_digest = _tensor_digest(expected)
        runtime_digest = _tensor_digest(runtime)
        if expected_digest != runtime_digest or max_abs != 0.0:
            raise FixedPanelValidationError("runtime policy aggregate digest differs")
        return {
            "tensor_count": count,
            "numel": numel,
            "max_abs_difference": max_abs,
            "checkpoint_tensor_digest": expected_digest,
            "runtime_tensor_digest": runtime_digest,
        }

    def _attest_action_owner(self, checkpoint_path: Path) -> dict[str, Any]:
        return {"student": self._attest_policy(checkpoint_path)}

    def _runtime_camera_rows(self) -> list[dict[str, Any]]:
        cameras = self.env.config.simulator.config.cameras
        return [
            {
                "name": str(camera["name"]),
                "link": str(camera["link"]),
                "position": [float(item) for item in camera["pos"]],
                "rotation_wxyz": [float(item) for item in camera["rot"]],
            }
            for camera in cameras.conditioning_views
        ]

    def eval(self, save_dirpath=None):
        _attest_pinned_sources()
        assignment = assignment_from_environment()
        if assignment.action_policy != self.fixed_panel_action_policy:
            raise FixedPanelValidationError(
                "assignment action policy does not match evaluator class"
            )
        cohort_index = int(_required_environment("DAGGER_FIXED_PANEL_COHORT_INDEX"))
        cohorts = assignment.cohorts()
        if cohort_index not in range(len(cohorts)):
            raise FixedPanelValidationError("fixed-panel cohort index must be in [0,3]")
        cohort = cohorts[cohort_index]
        image_augmentation = self.env.config.domain_rand.image_augmentation
        expect_generated = assignment.observation_mode == GENERATED_OBSERVATION_MODE
        expect_teacher = assignment.action_policy == TEACHER_ACTION_POLICY
        if (
            int(self.env.num_envs) != COHORT_ENVIRONMENTS
            or int(_required_environment("DAGGER_FIXED_PANEL_EFFECTIVE_SEED"))
            != cohort.paired_seed
            or not bool(self.config.eval.get("eval_num_envs_episodes", False))
            or int(self.config.eval.get("num_eval_episodes", -1)) != COHORT_ENVIRONMENTS
            or not isinstance(self.env, FixedPanelWalkStandPlaceGraspTurnHomie)
            or self.env.goal_accounting_version != GOAL_ACCOUNTING_VERSION
            or not isinstance(self.env.simulator, NormalizedIsaacSim)
            or bool(image_augmentation.get("enabled", True))
            or os.environ.get("IMAGE_DAGGER_EVAL_RELABEL") != (
                "1" if expect_generated else "0"
            )
            or bool(self.config.get("use_dagger", True))
            or (getattr(self, "ref_model", None) is not None) != expect_teacher
            or (
                getattr(self.accelerator.unwrap_model(self.model), "ref_model", None)
                is not None
            ) != expect_teacher
        ):
            raise FixedPanelValidationError("native fixed-panel cohort/config gate failed")
        _require_zero_motion_offset(self.env.config)
        views = tuple(str(value) for value in self.env.config.simulator.config.cameras.policy_views)
        if views != EXPECTED_VIEWS:
            raise FixedPanelValidationError("three-view policy order mismatch")
        if expect_generated and (
            not bool(getattr(self, "vg_enabled", False))
            or getattr(self, "vg_mode", None) != "image_dagger"
            or float(getattr(self, "vg_raytraced_mix_prob", -1.0)) != 0.0
        ):
            raise FixedPanelValidationError(
                "generated ablation requires Image3F and raytraced_mix_prob=0"
            )
        if assignment.wrist_calibration_sha256 is None:
            camera_attestation = {
                "legacy_unsealed_calibration": True,
                "runtime_cameras": self._runtime_camera_rows(),
            }
        else:
            calibration = calibration_from_environment(require_certified=True)
            camera_attestation = calibration.attest_runtime(self._runtime_camera_rows())
        checkpoint_path = Path(_required_environment("DAGGER_FIXED_PANEL_CHECKPOINT")).resolve(
            strict=True
        )
        policy_attestation = self._attest_action_owner(checkpoint_path)
        homie_attestation = _attest_homie_assets(self.config)
        # This dimension check is repeated at every executed action boundary.
        synthetic = {
            "action_mean": torch.zeros(
                (COHORT_ENVIRONMENTS, EXPECTED_EXECUTED_ACTIONS),
                device=self.accelerator.device,
            ),
            "actions": torch.zeros(
                (COHORT_ENVIRONMENTS, EXPECTED_EXECUTED_ACTIONS),
                device=self.accelerator.device,
            ),
        }
        if expect_teacher:
            synthetic["gt_actions"] = synthetic["actions"].clone()
        _require_action_contract(
            self, synthetic, action_policy=assignment.action_policy
        )
        client = AuthenticatedEvaluationClient(
            "http://127.0.0.1:18614",
            Path(_required_environment("DAGGER_FIXED_PANEL_TOKEN_FILE")),
        )
        journal = DurableOutcomeJournal(
            Path(_required_environment("DAGGER_FIXED_PANEL_LEDGER_ROOT")), assignment
        )
        journal.verify_remote(client, require_complete=False)
        latch = EpisodeLatch(cohort)
        staged_by_lane: dict[int, Any] = {}
        reset_attestation: dict[str, Any] | None = None
        action_contract_calls = 0
        original_reset_all = self.env.reset_all
        original_step = self.env.step
        original_process = self.env.process_eval_episode_completions

        def tracked_reset_all():
            nonlocal reset_attestation
            if reset_attestation is not None:
                raise FixedPanelValidationError(
                    "fixed-panel evaluator invoked reset_all more than once"
                )
            observations = original_reset_all()
            reset_attestation = _attest_native_reset(
                self.env, cohort, assignment.motion_groups
            )
            return observations

        def tracked_step(action_state):
            nonlocal action_contract_calls
            if reset_attestation is None:
                raise FixedPanelValidationError("policy step occurred before reset attestation")
            _require_action_contract(
                self, action_state, action_policy=assignment.action_policy
            )
            action_contract_calls += 1
            obs, rewards, dones, infos = original_step(action_state)
            if not isinstance(infos, Mapping) or "grasp_goal_reached" not in infos:
                raise FixedPanelValidationError(
                    "Raven infos lacks grasp_goal_reached; metric cannot be evaluated"
                )
            timeouts = infos.get("time_outs", getattr(self.env, "time_out_buf", None))
            safety = infos.get("safety_violation")
            outcomes = latch.observe(
                rewards=_float_vector(rewards, COHORT_ENVIRONMENTS, "rewards"),
                dones=_bool_vector(dones, COHORT_ENVIRONMENTS, "dones"),
                grasp_goal_reached=_bool_vector(
                    infos["grasp_goal_reached"], COHORT_ENVIRONMENTS, "grasp_goal_reached"
                ),
                timeouts=_bool_vector(timeouts, COHORT_ENVIRONMENTS, "timeouts"),
                safety_violations=_bool_vector(safety, COHORT_ENVIRONMENTS, "safety_violation"),
            )
            for outcome in outcomes:
                lane = cohort.cases.index(outcome.case)
                if lane in staged_by_lane:
                    raise FixedPanelValidationError("lane produced two first-episode outcomes")
                journal.stage(outcome)
                staged_by_lane[lane] = outcome
            return obs, rewards, dones, infos

        def tracked_process(completed_env_ids, cur_reward_sum, cur_episode_length):
            lanes = [int(value) for value in completed_env_ids.reshape(-1).detach().cpu()]
            for lane in lanes:
                if lane not in staged_by_lane:
                    raise FixedPanelValidationError("Raven completion lacks staged latch outcome")
                raven_goal = bool(
                    self.env.eval_metrics["episode_goal_reached"][lane]
                    .reshape(-1)[0]
                    .item()
                )
                if raven_goal != staged_by_lane[lane].goal_reached:
                    raise FixedPanelValidationError(
                        "independent OR latch differs from Raven episode_goal_reached"
                    )
            result = original_process(completed_env_ids, cur_reward_sum, cur_episode_length)
            for lane in lanes:
                journal.post_staged(cohort.cases[lane], client)
            return result

        self.env.reset_all = tracked_reset_all
        self.env.step = tracked_step
        self.env.process_eval_episode_completions = tracked_process
        try:
            result = super().eval(save_dirpath=save_dirpath)
        finally:
            self.env.reset_all = original_reset_all
            self.env.step = original_step
            self.env.process_eval_episode_completions = original_process
        expected_scenarios = {case.scenario_id for case in cohort.cases}
        if latch.completed != COHORT_ENVIRONMENTS or not expected_scenarios.issubset(
            journal.acked_scenarios()
        ):
            raise FixedPanelValidationError(
                "cohort did not durably acknowledge exactly 64 cases"
            )
        if reset_attestation is None or action_contract_calls <= 0:
            raise FixedPanelValidationError("reset/action runtime attestation was not exercised")
        normalization_calls = int(
            getattr(self.env.simulator, "_three_view_normalization_calls", 0)
        )
        if normalization_calls <= 0:
            raise FixedPanelValidationError("three-view normalization shim was never invoked")
        generated_calls = len(getattr(self, "_id_gen_ms", []))
        if expect_generated and generated_calls <= 0:
            raise FixedPanelValidationError("generated ablation never called Image3F")
        if not expect_generated and generated_calls != 0:
            raise FixedPanelValidationError("raytraced ablation unexpectedly called Image3F")
        output = Path(_required_environment("DAGGER_FIXED_PANEL_COHORT_OUTPUT"))
        cohort_result = {
                "schema": "roborender.raven_success_validation_cohort_result.v2",
                "evaluation_id": assignment.evaluation_id,
                "run_id": assignment.run_id,
                "round_id": assignment.round_id,
                "checkpoint_sha256": assignment.checkpoint_sha256,
                "scenario_manifest_sha256": assignment.scenario_manifest_sha256,
                "assignment_sha256": assignment.assignment_sha256,
                "cohort_index": cohort.cohort_index,
                "cohort_id": cohort.cohort_id,
                "paired_seed": cohort.paired_seed,
                "environment_count": COHORT_ENVIRONMENTS,
                "completed": latch.completed,
                "scenario_ids": [case.scenario_id for case in cohort.cases],
                "policy_role": assignment.policy_role,
                "policy": assignment.action_policy,
                "observation_mode": assignment.observation_mode,
                "teacher_checkpoint_sha256": assignment.teacher_checkpoint_sha256,
                "views": list(EXPECTED_VIEWS),
                "image_augmentation_enabled": False,
                "goal_accounting_version": GOAL_ACCOUNTING_VERSION,
                "action_contract": {
                    "student_dimension": EXPECTED_STUDENT_ACTIONS,
                    "homie_dimension": EXPECTED_HOMIE_ACTIONS,
                    "executed_dimension": EXPECTED_EXECUTED_ACTIONS,
                    "executed_policy": assignment.action_policy,
                    "validated_step_calls": action_contract_calls,
                },
                "reset_attestation": reset_attestation,
                "normalization_calls": normalization_calls,
                "normalization_first_raw": self.env.simulator._three_view_normalization_first_raw,
                "normalization_first_output": self.env.simulator._three_view_normalization_first_output,
                "image3f_relabel_calls": generated_calls,
                "camera_calibration_attestation": camera_attestation,
                "policy_attestation": policy_attestation,
                "checkpoint_keys": self._fixed_panel_checkpoint_keys,
                "homie_attestation": homie_attestation,
                "pinned_source_sha256": _attest_pinned_sources(),
            }
        atomic_write_json(output, cohort_result)
        # The generic metrics adapter requires a mapping.  The stock evaluator
        # returns None, so expose only this compact contract result.
        return cohort_result


class FixedPanelTeacherOracleEvalTrainer(FixedPanelEvalTrainer):
    """Execute Raven's exact privileged 11D teacher plus exact 15D Homie tail.

    A normal three-view student checkpoint is still loaded as an inert model
    scaffold because Raven's trainer constructs that model unconditionally.
    The assignment and result identify the separately attested teacher bytes,
    and every environment step is guarded so only ``gt_actions`` can execute.
    """

    fixed_panel_action_policy = TEACHER_ACTION_POLICY

    def load_teacher_actor(self):
        _attest_pinned_sources()
        assignment = assignment_from_environment()
        expected = assignment.teacher_checkpoint_sha256
        if expected is None or expected != PINNED_PRIVILEGED_TEACHER_SHA256:
            raise FixedPanelValidationError("teacher assignment identity is not pinned")
        configured = Path(
            str(self.config.network_load_dict.teacher_actor.path)
        ).resolve(strict=True)
        supplied = Path(
            _required_environment("DAGGER_FIXED_PANEL_TEACHER_CHECKPOINT")
        ).resolve(strict=True)
        if (
            configured != supplied
            or supplied.is_symlink()
            or not supplied.is_file()
            or supplied.stat().st_size != PINNED_PRIVILEGED_TEACHER_SIZE
            or file_sha256(supplied) != expected
        ):
            raise FixedPanelValidationError("privileged teacher file identity mismatch")
        # Call the pinned Raven loader, not a locally reconstructed loader.
        TRLDistillTrainer.load_teacher_actor(self)
        checkpoint = torch.load(supplied, map_location="cpu", weights_only=False)
        state = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
        if not isinstance(state, Mapping):
            raise FixedPanelValidationError("teacher checkpoint lacks actor state")
        runtime = self.ref_model.state_dict()
        if set(state) != set(runtime):
            raise FixedPanelValidationError("teacher runtime/checkpoint key sets differ")
        unequal = [
            name for name in sorted(state)
            if state[name].shape != runtime[name].shape
            or state[name].dtype != runtime[name].dtype
            or not torch.equal(state[name].detach().cpu(), runtime[name].detach().cpu())
        ]
        if unequal:
            raise FixedPanelValidationError(
                f"teacher runtime differs from checkpoint: {unequal[:8]}"
            )
        checkpoint_digest = _tensor_digest(state)
        runtime_digest = _tensor_digest(runtime)
        if checkpoint_digest != runtime_digest:
            raise FixedPanelValidationError("teacher runtime aggregate digest differs")
        self._teacher_attestation = {
            "path": str(supplied),
            "size_bytes": supplied.stat().st_size,
            "sha256": expected,
            "tensor_count": len(state),
            "numel": sum(int(value.numel()) for value in state.values()),
            "checkpoint_tensor_digest": checkpoint_digest,
            "runtime_tensor_digest": runtime_digest,
            "loader_source_sha256": PINNED_TEACHER_LOADER_SHA256,
            "action_path": "ref_model.act_inference(teacher_obs)+homie_action",
        }

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        # Reuse the pinned Raven policy_step prefix and deliberately stop at
        # its student boundary. This avoids performing or depending on an
        # inert student prediction during the privileged-oracle baseline.
        from dagger_rtx.native_entrypoints import (
            _find_teacher_policy_method,
            _LabelsReady,
            _StopBeforeStudent,
        )

        policy_method = _find_teacher_policy_method(self)
        sentinel = _StopBeforeStudent(policy_method.__code__)
        try:
            with torch.no_grad():
                policy_method(self, sentinel, obs_dict, cur_dones=cur_dones)
        except _LabelsReady as ready:
            teacher = ready.policy_state.get("gt_actions")
        else:
            raise FixedPanelValidationError(
                "pinned teacher path crossed into student inference"
            )
        if (
            not torch.is_tensor(teacher)
            or tuple(teacher.shape) != (COHORT_ENVIRONMENTS, EXPECTED_EXECUTED_ACTIONS)
            or not bool(torch.isfinite(teacher).all().item())
        ):
            raise FixedPanelValidationError(
                "pinned teacher/Homie policy path produced invalid gt_actions"
            )
        return {
            "gt_actions": teacher.clone(),
            "actions": teacher.clone(),
            "action_mean": teacher.clone(),
        }

    def _attest_action_owner(self, checkpoint_path: Path) -> dict[str, Any]:
        if not hasattr(self, "_teacher_attestation"):
            raise FixedPanelValidationError("privileged teacher was not loaded/attested")
        return {
            "teacher": dict(self._teacher_attestation),
            "inert_student_scaffold": self._attest_policy(checkpoint_path),
        }


__all__ = [
    "FixedPanelEvalTrainer",
    "FixedPanelTeacherOracleEvalTrainer",
    "FixedPanelWalkStandPlaceGraspTurnHomie",
    "IsaacSim",
    "NormalizedIsaacSim",
]
