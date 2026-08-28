#!/usr/bin/env python3
"""Validate and materialize the environment for one production macro-round.

This program is the non-mutating preflight used by
``launch_production_macro_round.sh``.  It deliberately does not initialize a
campaign, change coordinator state, or submit a job.  A successful result
means that the two checked-in production contracts, the durable campaign, the
strict per-round orchestration config, every local static input, and the
sealed OCI-NRT source snapshot all agree byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
import json
import netrc
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger import artifacts, policy_bundle, replay, state  # noqa: E402
from round_dagger import motion_split_contract  # noqa: E402
from round_dagger import production_data_lineage  # noqa: E402
from round_dagger.gr00t_policy import (  # noqa: E402
    ENV_COMMAND_DIM,
    HOMIE_ACTION_DIM,
    OBJECT_PREDICTION_DIM,
    POLICY_ACTION_DIM,
    REPLAY_ACTION_DIM,
    TRAIN_ACTION_SLICE,
    VISION_SHAPE,
    load_policy_spec,
    validate_policy_spec,
)
from round_dagger.orchestration import (  # noqa: E402
    COHORT_COUNT,
    ENVIRONMENTS_PER_COHORT,
    MacroRoundConfig,
    PRODUCTION_MACRO_ROUND_COUNT,
    STEPS_PER_ENVIRONMENT,
    TARGET_ENVIRONMENTS,
    TRANSITIONS_PER_COHORT,
    TRANSITIONS_PER_MACRO_ROUND,
    TRANSITIONS_PER_CAMPAIGN,
)
from scripts.round_dagger.verify_anchor_pool import verify_anchor_pool  # noqa: E402
from scripts.round_dagger.verify_sealed_source_snapshot import (  # noqa: E402
    VerificationError as SealedSourceVerificationError,
    verify as verify_sealed_source_snapshot,
)


CAMPAIGN_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.production_campaign_contract.v2"
)
LEARNER_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.production_learner_contract.v1"
)
LEARNER_BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
SEALED_REPORT_SCHEMA = "fast_roborender.sealed_source_verification.v1"
PLAN_SCHEMA = "fast_roborender.round_dagger.production_launch_plan.v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
SAFE_REMOTE_PATH_RE = re.compile(r"/[A-Za-z0-9_./-]+")
VALIDATION_ROOT = Path("/mnt/data2/ldu/roborender_round_dagger_validation")
BOOTSTRAP_ROOT = (
    VALIDATION_ROOT / "fresh_bootstrap_v6_complete_identity_20260827T061620Z"
)
FLEET_ROOT = VALIDATION_ROOT / "fleet_two_gpu_provenance_v2"
PRODUCTION_ASSET_ROOT = Path("/mnt/data2/ldu/roborender_round_dagger_assets")
WANDB_NETRC = Path("/mnt/data2/ldu/roborender_round_dagger_credentials/.netrc")
CAMERA_ACCEPTANCE = (
    VALIDATION_ROOT
    / "production_inputs_v1/camera/camera_calibration_acceptance.json"
)
TEACHER_CHECKPOINT_RELATIVE = Path(
    "GR00T-VisualSim2Real/logs_rl/"
    "walk_stand_place_grasp_turn_homie_predrop_project_name-wsdpt_teacher_"
    "BTFv8q9_qben_exppredrop_leanz_hand_predrop_pos_distance-10.0_"
    "penalty_bottle_lean_during_pick--3500.0_penalty_bottle_non_z_up_velocity_"
    "during_pick--1000.0-20251105_004910/model_step_044500.pt"
)
ACTOR_GR00T_ROOT_RELATIVE = Path("GR00T-VisualSim2Real")
ACTOR_PAIR_EXECUTOR_RELATIVE = Path(
    "scripts/round_dagger/run_actor_rank_docker.sh"
)
ACTOR_PAIR_RANK_SCRIPT_RELATIVE = Path(
    "scripts/round_dagger/run_actor_cohort_with_wandb.sh"
)
PRODUCTION_CAMPAIGN_CONTRACT_RELATIVE = Path(
    "configs/round_dagger/production_campaign_256x512.json"
)
MOTION_SPLIT_MANIFEST_RELATIVE = Path(
    "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/"
    "33demos_675_775_heldout_256_v1.json"
)
HELDOUT_EVALUATION_PLAN_RELATIVE = Path(
    "configs/round_dagger/production_heldout_motion_plan_8x32.json"
)
ANCHOR_POOL = PRODUCTION_ASSET_ROOT / "anchor_pool_2048"
EVALUATION_CONFIG = (
    VALIDATION_ROOT
    / "production_inputs_v2/evaluation/config.yaml"
)
OCI_STAGE_ROOT = Path(
    "/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/"
    "users/ldu/roborender_h100_build"
)

CAMPAIGN_CONTRACT_KEYS = {
    "schema",
    "campaign_id",
    "topology",
    "observation_contract",
    "action_contract",
    "beta_schedule",
    "motion_split",
    "artifacts",
    "actor_runtime",
    "telemetry",
    "wandb",
    "evaluation",
}
LEARNER_CONTRACT_KEYS = {
    "schema",
    "allocation",
    "optimization",
    "cluster",
    "container",
    "handoff",
    "wandb",
    "source",
}


class ProductionLaunchError(RuntimeError):
    """A production launch precondition is missing or ambiguous."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ProductionLaunchError(
            f"{label} fields differ: missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProductionLaunchError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProductionLaunchError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionLaunchError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ProductionLaunchError(f"{label} must be a normalized absolute path")
    return path


def _source_binding_root(value: Any, relative: Path, label: str) -> Path:
    """Return the legacy source root for one source-owned contract path.

    The checked-in production profile predates relocatable source snapshots
    and therefore records four repository-owned paths as absolute paths.  The
    absolute prefix is not an artifact identity: the fixed relative suffix is.
    Validate that encoding without consulting the legacy checkout (which may
    no longer exist), then let the loader bind the suffix to the repository
    that is actually executing this code.
    """

    path = _absolute(value, label)
    if str(path) != value:
        raise ProductionLaunchError(
            f"{label} must be a lexically normalized absolute path"
        )
    relative_parts = relative.parts
    if (
        relative.is_absolute()
        or not relative_parts
        or len(path.parts) <= len(relative_parts)
        or path.parts[-len(relative_parts) :] != relative_parts
    ):
        raise ProductionLaunchError(
            f"{label} repository path binding mismatch: expected suffix "
            f"{relative.as_posix()!r}"
        )
    root = Path(*path.parts[: -len(relative_parts)])
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or str(root) != root.as_posix()
    ):
        raise ProductionLaunchError(
            f"{label} repository path binding has an unsafe source root"
        )
    return root


def _materialize_source_owned_paths(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind exactly four validated source-owned paths to this checkout."""

    bindings = (
        (
            ("artifacts", "teacher", "checkpoint_path"),
            TEACHER_CHECKPOINT_RELATIVE,
            "teacher checkpoint path",
        ),
        (
            ("actor_runtime", "gr00t_root"),
            ACTOR_GR00T_ROOT_RELATIVE,
            "actor GR00T root",
        ),
        (
            ("actor_runtime", "pair_executor"),
            ACTOR_PAIR_EXECUTOR_RELATIVE,
            "actor pair executor",
        ),
        (
            ("actor_runtime", "pair_rank_script"),
            ACTOR_PAIR_RANK_SCRIPT_RELATIVE,
            "actor pair rank script",
        ),
    )
    roots: list[Path] = []
    for keys, relative, label in bindings:
        value: Any = contract
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                raise ProductionLaunchError(
                    f"{label} repository path binding is missing"
                )
            value = value[key]
        roots.append(_source_binding_root(value, relative, label))
    if len(set(roots)) != 1:
        raise ProductionLaunchError(
            "source-owned repository paths do not share one canonical source root"
        )

    materialized = copy.deepcopy(contract)
    for keys, relative, _label in bindings:
        target: dict[str, Any] = materialized
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = str(REPOSITORY_ROOT / relative)
    return materialized


def _regular(path: Path, label: str) -> Path:
    try:
        path.lstat()
    except OSError as exc:
        raise ProductionLaunchError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ProductionLaunchError(f"{label} must be a non-symlink regular file")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        path.lstat()
    except OSError as exc:
        raise ProductionLaunchError(f"{label} is unavailable: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise ProductionLaunchError(f"{label} must be a non-symlink directory")
    return path


def _require_evaluation_case_capacity(
    scenario_path: Path, threshold_path: Path
) -> None:
    """Reject an evaluation set that cannot satisfy its promotion minimum."""

    try:
        scenarios = artifacts.load_strict_json(scenario_path)
        thresholds = artifacts.load_strict_json(threshold_path)
    except Exception as exc:
        raise ProductionLaunchError(
            f"cannot read paired-evaluation capacity contracts: {exc}"
        ) from exc
    # The immutable scenario artifact consumed by round_dagger.evaluation is a
    # canonical top-level list.  Do not confuse it with the later published
    # evaluation contract, whose ``scenario_cases`` field wraps that list.
    cases = scenarios if isinstance(scenarios, list) else None
    minimum_cases = (
        thresholds.get("minimum_paired_cases")
        if isinstance(thresholds, Mapping)
        else None
    )
    if (
        not isinstance(cases, list)
        or isinstance(minimum_cases, bool)
        or not isinstance(minimum_cases, int)
        or minimum_cases < 1
        or len(cases) < minimum_cases
    ):
        observed = len(cases) if isinstance(cases, list) else 0
        raise ProductionLaunchError(
            "evaluation scenario contract contains "
            f"{observed} cases but production requires {minimum_cases!r}"
        )


def _file_identity(path: Path, expected_sha: str, label: str) -> None:
    _regular(path, label)
    if artifacts.sha256_file(path) != _require_sha(expected_sha, f"{label} SHA-256"):
        raise ProductionLaunchError(f"{label} bytes differ from the production contract")


def _verify_first_condition_bundles(
    campaign_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runtime = campaign_contract.get("production_runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ProductionLaunchError(
            "materialized campaign has no production runtime contract"
        )
    capture = runtime.get("live_capture")
    if not isinstance(capture, Mapping):
        raise ProductionLaunchError("production runtime contract has no live capture")
    expected_keys = {
        "seed",
        "path",
        "sha256",
        "size",
        "case_count",
        "source_pass_index",
    }
    expected_seeds = [17, 29, 43, 59, 71, 89, 101, 127]
    rows = capture.get("first_condition_bundles")
    if not isinstance(rows, list) or len(rows) != len(expected_seeds):
        raise ProductionLaunchError(
            "first-condition bundle inventory is not exact eight-seed coverage"
        )
    inventory_sha = _require_sha(
        capture.get("first_condition_bundle_inventory_sha256"),
        "first-condition bundle inventory SHA-256",
    )
    if artifacts.canonical_json_sha256(rows) != inventory_sha:
        raise ProductionLaunchError(
            "first-condition bundle inventory digest is non-canonical"
        )
    lineage = campaign_contract.get("data_lineage")
    heldout = lineage.get("heldout_evaluation") if isinstance(lineage, Mapping) else None
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("first_condition_bundle_inventory_sha256") != inventory_sha
        or heldout.get("first_condition_bundles") != rows
    ):
        raise ProductionLaunchError(
            "first-condition bundle inventory differs from durable data lineage"
        )
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, (raw, seed) in enumerate(zip(rows, expected_seeds, strict=True)):
        row = _exact(raw, expected_keys, f"first-condition bundle {index}")
        if (
            row["seed"] != seed
            or type(row["size"]) is not int
            or row["size"] < 1
            or row["case_count"] != 32
            or row["source_pass_index"] != 1
        ):
            raise ProductionLaunchError(
                f"first-condition bundle for seed {seed} is malformed"
            )
        path = _regular(
            _absolute(row["path"], f"first-condition bundle {seed}"),
            f"first-condition bundle {seed}",
        )
        try:
            if path.resolve(strict=True) != path:
                raise ProductionLaunchError(
                    f"first-condition bundle {seed} path is not canonical"
                )
        except OSError as exc:
            raise ProductionLaunchError(
                f"cannot resolve first-condition bundle {seed}"
            ) from exc
        if str(path) in seen_paths:
            raise ProductionLaunchError("first-condition bundle paths are not unique")
        seen_paths.add(str(path))
        metadata = path.stat()
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ProductionLaunchError(
                f"first-condition bundle {seed} mode must be exactly 0444"
            )
        if metadata.st_size != row["size"]:
            raise ProductionLaunchError(
                f"first-condition bundle {seed} size differs from inventory"
            )
        _file_identity(path, row["sha256"], f"first-condition bundle {seed}")
        normalized.append(dict(row))
    return normalized


def _load_contract(
    path: Path,
    schema: str,
    keys: set[str],
    label: str,
    *,
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    _regular(path, label)
    try:
        document = artifacts.load_strict_json(path)
    except Exception as exc:
        raise ProductionLaunchError(f"cannot load strict {label}") from exc
    optional = optional_keys or set()
    if not isinstance(document, Mapping):
        raise ProductionLaunchError(f"{label} must be a mapping")
    actual = set(document)
    if not keys <= actual or actual - keys - optional:
        raise ProductionLaunchError(
            f"{label} fields differ: missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys - optional)}"
        )
    result = dict(document)
    _require_equal(result["schema"], schema, f"{label} schema")
    return result


def load_campaign_contract(path: Path) -> dict[str, Any]:
    raw_contract = _load_contract(
        path, CAMPAIGN_CONTRACT_SCHEMA, CAMPAIGN_CONTRACT_KEYS, "campaign contract"
    )
    contract = _materialize_source_owned_paths(raw_contract)
    _require_equal(contract["campaign_id"], "dagger_prod_run1", "campaign ID")
    _require_equal(
        _exact(
            contract["topology"],
            {
                "target_environments",
                "active_environments",
                "cohort_count",
                "environments_per_actor_rank",
                "actor_ranks_per_cohort",
                "macro_round_count",
                "steps_per_environment",
                "transitions_per_cohort",
                "transitions_per_macro_round",
                "transitions_per_campaign",
            },
            "campaign topology",
        ),
        {
            "target_environments": TARGET_ENVIRONMENTS,
            "active_environments": ENVIRONMENTS_PER_COHORT,
            "cohort_count": COHORT_COUNT,
            "environments_per_actor_rank": 32,
            "actor_ranks_per_cohort": 2,
            "macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
            "steps_per_environment": STEPS_PER_ENVIRONMENT,
            "transitions_per_cohort": TRANSITIONS_PER_COHORT,
            "transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
            "transitions_per_campaign": TRANSITIONS_PER_CAMPAIGN,
        },
        "campaign topology",
    )
    _require_equal(
        _exact(
            contract["observation_contract"],
            {
                "primary_view",
                "policy_views",
                "policy_image_shape",
                "policy_view_layout",
                "bootstrap_anchor_source",
                "reference_lag_frames",
            },
            "observation contract",
        ),
        {
            "primary_view": "head_color",
            "policy_views": ["head_color", "hand_left", "hand_right"],
            "policy_image_shape": [324, 192, 3],
            "policy_view_layout": "vertical",
            "bootstrap_anchor_source": "simulator_rgb",
            "reference_lag_frames": 256,
        },
        "observation contract",
    )
    _require_equal(
        _exact(
            contract["action_contract"],
            {
                "policy_action_dim",
                "homie_action_dim",
                "replay_action_dim",
                "simulator_action_dim",
                "train_action_slice",
            },
            "action contract",
        ),
        {
            "policy_action_dim": 11,
            "homie_action_dim": 15,
            "replay_action_dim": 26,
            "simulator_action_dim": 31,
            "train_action_slice": [0, 11],
        },
        "action contract",
    )
    _require_equal(
        _exact(
            contract["beta_schedule"],
            {"kind", "start", "end", "decay_rounds", "rate"},
            "beta schedule",
        ),
        {"kind": "linear", "start": 1.0, "end": 0.0, "decay_rounds": 4, "rate": 1.0},
        "beta schedule",
    )

    split = _exact(
        contract["motion_split"],
        {
            "manifest_relative_path",
            "manifest_sha256",
            "heldout_plan_relative_path",
            "heldout_plan_sha256",
            "collection_role",
            "evaluation_role",
            "verify_content",
            "collection_motion_count",
            "heldout_motion_count",
            "evaluation_seed_count",
            "motions_per_evaluation_seed",
        },
        "motion split contract",
    )
    _require_equal(
        split,
        {
            "manifest_relative_path": MOTION_SPLIT_MANIFEST_RELATIVE.as_posix(),
            "manifest_sha256": (
                "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834"
            ),
            "heldout_plan_relative_path": (
                HELDOUT_EVALUATION_PLAN_RELATIVE.as_posix()
            ),
            "heldout_plan_sha256": (
                "394ba77b2c8ecbdb4904ef24894cda0b88bb5b606ab75e578ee0a91756630f19"
            ),
            "collection_role": "collection",
            "evaluation_role": "eval",
            "verify_content": True,
            "collection_motion_count": 39,
            "heldout_motion_count": 256,
            "evaluation_seed_count": 8,
            "motions_per_evaluation_seed": 32,
        },
        "motion split contract",
    )
    _require_sha(split["manifest_sha256"], "motion split manifest SHA-256")
    _require_sha(split["heldout_plan_sha256"], "held-out plan SHA-256")

    artifacts_doc = _exact(
        contract["artifacts"],
        {
            "bootstrap_bundle",
            "teacher",
            "renderer",
            "endpoint_manifest",
            "anchor_pool",
            "camera_acceptance",
            "actor_image",
        },
        "campaign artifacts",
    )
    bootstrap = _exact(
        artifacts_doc["bootstrap_bundle"],
        {
            "checkpoint_path",
            "checkpoint_sha256",
            "runtime_policy_sha256",
            "policy_spec_path",
            "policy_spec_sha256",
            "manifest_path",
            "manifest_sha256",
        },
        "bootstrap bundle",
    )
    _require_equal(
        bootstrap["checkpoint_sha256"],
        "720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a",
        "bootstrap checkpoint identity",
    )
    _require_equal(
        bootstrap["runtime_policy_sha256"],
        "cd6f70aa9d589e810b7ab22999941930742a94621870457aadd068d2394deb9a",
        "bootstrap runtime identity",
    )
    _require_equal(
        bootstrap["policy_spec_sha256"],
        "3e903c55218ca2aae4c970fab6aeb820de0a35d8965a34262b8206fd62f74dee",
        "bootstrap policy spec identity",
    )
    _require_equal(
        bootstrap["manifest_sha256"],
        "33a1a07a4ffa09e02805bbb5ab70ba32ecf3d1d2d195d9f1bcb61021c20ad6ae",
        "bootstrap manifest identity",
    )
    for key in ("checkpoint_path", "policy_spec_path", "manifest_path"):
        _absolute(bootstrap[key], f"bootstrap {key}")
    _require_equal(
        bootstrap["checkpoint_path"], str(BOOTSTRAP_ROOT / "student-v0.pt"),
        "bootstrap checkpoint path",
    )
    _require_equal(
        bootstrap["policy_spec_path"], str(BOOTSTRAP_ROOT / "student-v0.pt.policy_spec.json"),
        "bootstrap policy spec path",
    )
    _require_equal(
        bootstrap["manifest_path"], str(BOOTSTRAP_ROOT / "student-v0.pt.manifest.json"),
        "bootstrap policy manifest path",
    )

    teacher = _exact(
        artifacts_doc["teacher"], {"checkpoint_path", "checkpoint_sha256"}, "teacher"
    )
    _require_equal(
        teacher["checkpoint_sha256"],
        "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38",
        "teacher identity",
    )
    _absolute(teacher["checkpoint_path"], "teacher checkpoint")
    _require_equal(
        teacher["checkpoint_path"],
        str(REPOSITORY_ROOT / TEACHER_CHECKPOINT_RELATIVE),
        "teacher checkpoint path",
    )
    renderer = _exact(
        artifacts_doc["renderer"],
        {"cohort_identity_sha256", "runtime_identity_sha256"},
        "renderer",
    )
    _require_equal(
        renderer,
        {
            "cohort_identity_sha256": "a82f639a9038789db220793475dfce3eb5651ff3145287be748f387a20caeb9f",
            "runtime_identity_sha256": "8a172fc0819cdeb6f047ca60605c7f2a00920c88ce8ef202b40a4ab43e656809",
        },
        "renderer identities",
    )
    endpoint = _exact(
        artifacts_doc["endpoint_manifest"],
        {"path", "sha256", "endpoint_directory", "renderer_count"},
        "renderer endpoint manifest",
    )
    _absolute(endpoint["path"], "renderer endpoint manifest")
    _absolute(endpoint["endpoint_directory"], "renderer endpoint directory")
    _require_equal(endpoint["renderer_count"], 2, "renderer endpoint count")
    _require_equal(endpoint["path"], str(FLEET_ROOT / "endpoint_manifest.txt"), "endpoint manifest path")
    _require_equal(endpoint["endpoint_directory"], str(FLEET_ROOT / "endpoints"), "endpoint directory path")
    _require_equal(
        endpoint["sha256"],
        "d075c02b3e8aace7c1a4b2ef1e0b694738052da66bea6300299a7c4fd6956be6",
        "renderer endpoint manifest identity",
    )
    anchors = _exact(
        artifacts_doc["anchor_pool"],
        {
            "path",
            "anchor_count",
            "inventory_sha256",
            "inventory_algorithm",
            "usage",
        },
        "anchor pool",
    )
    _absolute(anchors["path"], "anchor pool")
    _require_equal(anchors["path"], str(ANCHOR_POOL), "anchor pool path")
    _require_equal(anchors["anchor_count"], 2048, "anchor count")
    _require_equal(
        anchors["inventory_sha256"],
        "ee358a97581512a5d03730d06efd5804e53844f517112dc0a845ed86b6dae46f",
        "anchor inventory identity",
    )
    _require_equal(
        anchors["inventory_algorithm"],
        "len64be(path_utf8)+path_utf8+size64be+sha256_binary",
        "anchor inventory algorithm",
    )
    _require_equal(
        anchors["usage"],
        "compatibility_fallback_only",
        "anchor pool usage",
    )
    camera = _exact(
        artifacts_doc["camera_acceptance"], {"path", "sha256", "reviewer"}, "camera acceptance"
    )
    _absolute(camera["path"], "camera acceptance")
    _require_equal(camera["path"], str(CAMERA_ACCEPTANCE), "camera acceptance path")
    _require_equal(
        camera,
        {
            "path": camera["path"],
            "sha256": "53234dfe2eab9b1ae02df21e143f996ecde831aadb8aacb2e5f2df632d3d8d27",
            "reviewer": "ldu@nvidia.com",
        },
        "camera acceptance identity",
    )
    actor_image = _exact(
        artifacts_doc["actor_image"], {"name", "image_id"}, "actor image"
    )
    _require_equal(
        actor_image,
        {
            "name": "roborender-round-actor:isaaclab2.3.0-local-v2",
            "image_id": "sha256:65a0cbe64eb71ee2edb672bed3ec7bdaa8adfeb22298c0e99af8b91d74dc78f8",
        },
        "actor image identity",
    )

    actor_runtime = _exact(
        contract["actor_runtime"],
        {
            "docker_bin",
            "gr00t_root",
            "gr00t_python",
            "pair_executor",
            "pair_rank_script",
            "gpu_ids",
            "ipc",
            "shm_size",
            "mount_roots",
            "rank_stagger_seconds",
            "environments_per_shard",
            "rollout_id",
        },
        "actor runtime",
    )
    _require_equal(
        actor_runtime,
        {
            "docker_bin": "/usr/bin/docker",
            "gr00t_root": str(REPOSITORY_ROOT / ACTOR_GR00T_ROOT_RELATIVE),
            "gr00t_python": "/opt/round-dagger-rtx/bin/python",
            "pair_executor": str(REPOSITORY_ROOT / ACTOR_PAIR_EXECUTOR_RELATIVE),
            "pair_rank_script": str(
                REPOSITORY_ROOT / ACTOR_PAIR_RANK_SCRIPT_RELATIVE
            ),
            "gpu_ids": "0,1",
            "ipc": "private",
            "shm_size": "32g",
            "mount_roots": "/mnt/data1,/mnt/data2",
            "rank_stagger_seconds": 10,
            "environments_per_shard": 4,
            "rollout_id": 0,
        },
        "actor runtime",
    )
    telemetry = _exact(
        contract["telemetry"],
        {
            "required",
            "require_video",
            "scalar_every_steps",
            "image_every_steps",
            "clip_every_steps",
            "clip_frames",
            "clip_fps",
        },
        "actor telemetry",
    )
    _require_equal(
        telemetry,
        {
            "required": True,
            "require_video": True,
            "scalar_every_steps": 10,
            "image_every_steps": 300,
            "clip_every_steps": 300,
            "clip_frames": 32,
            "clip_fps": 10,
        },
        "actor telemetry",
    )
    wandb = _exact(
        contract["wandb"],
        {"enabled", "expected_email", "entity", "authorization_scope", "project", "python", "netrc"},
        "actor W&B",
    )
    _require_equal(
        wandb,
        {
            "enabled": True,
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "project": "roborender-round-dagger",
            "python": "/opt/round-dagger-rtx/bin/python",
            "netrc": str(WANDB_NETRC),
        },
        "actor W&B identity",
    )
    evaluation = _exact(
        contract["evaluation"],
        {
            "hydra_config_path",
            "hydra_config_sha256",
            "scenario_contract",
            "scenario_contract_sha256",
            "threshold_contract",
            "threshold_contract_sha256",
        },
        "evaluation contract",
    )
    _absolute(evaluation["hydra_config_path"], "evaluation Hydra config")
    _require_equal(evaluation["hydra_config_path"], str(EVALUATION_CONFIG), "evaluation Hydra config path")
    _require_equal(
        evaluation["hydra_config_sha256"],
        "bd3d61082149e3b87855ea3ab396e7cae7479d9bfe9d9d3aab99c839353ed7a2",
        "evaluation Hydra config identity",
    )
    _require_equal(
        evaluation["scenario_contract"],
        "configs/round_dagger/production_paired_scenarios_2x8.json",
        "evaluation scenario contract path",
    )
    _require_equal(
        evaluation["scenario_contract_sha256"],
        "99081ebee907ef36c388cf62a1d134c015d289776526cacc1db114a87074e724",
        "evaluation scenario contract identity",
    )
    _require_equal(
        evaluation["threshold_contract"],
        "configs/round_dagger/production_paired_thresholds.json",
        "evaluation threshold contract path",
    )
    _require_equal(
        evaluation["threshold_contract_sha256"],
        "05d9fbb64c41311e586f2d14f889902a3ea7fe56e31bd8b16cd539a65c035563",
        "evaluation threshold contract identity",
    )
    for key in (
        "hydra_config_sha256",
        "scenario_contract_sha256",
        "threshold_contract_sha256",
    ):
        _require_sha(evaluation[key], f"evaluation {key}")
    return contract


def materialize_campaign_contract(
    invariant_profile_path: Path,
    *,
    runtime_contract_path: Path,
    runtime_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Merge a checked-in invariant profile with post-seal runtime evidence.

    ``runtime_contract_sha256`` is the SHA-256 of the complete canonical JSON
    file; the file intentionally contains no circular self-hash field.
    """

    invariant_sha = artifacts.sha256_file(invariant_profile_path)
    profile = load_campaign_contract(invariant_profile_path)
    try:
        runtime_contract = production_data_lineage.load_verified_runtime_contract(
            runtime_contract_path,
            expected_sha256=runtime_contract_sha256,
            expected_invariant_profile_sha256=invariant_sha,
            motion_split_contract=profile["motion_split"],
        )
        lineage = production_data_lineage.derive_data_lineage(
            runtime_contract,
            motion_split_contract=profile["motion_split"],
            topology=profile["topology"],
        )
    except production_data_lineage.ProductionDataLineageError as exc:
        raise ProductionLaunchError(
            f"production runtime contract is invalid: {exc}"
        ) from exc

    materialized = copy.deepcopy(profile)
    camera_document = artifacts.load_strict_json(
        runtime_contract["camera_acceptance"]["path"]
    )
    reviewer = camera_document.get("reviewer") if isinstance(camera_document, Mapping) else None
    if not isinstance(reviewer, str) or not reviewer:
        raise ProductionLaunchError("runtime camera acceptance has no reviewer")
    materialized["artifacts"]["camera_acceptance"] = {
        "path": runtime_contract["camera_acceptance"]["path"],
        "sha256": runtime_contract["camera_acceptance"]["sha256"],
        "reviewer": reviewer,
    }
    fleet = runtime_contract["renderer_fleet"]
    materialized["artifacts"]["renderer"] = {
        "cohort_identity_sha256": fleet["cohort_identity_sha256"],
        "runtime_identity_sha256": fleet["runtime_identity_sha256"],
    }
    materialized["artifacts"]["endpoint_manifest"] = {
        "path": fleet["endpoint_manifest_path"],
        "sha256": fleet["endpoint_manifest_sha256"],
        "endpoint_directory": fleet["endpoint_directory"],
        "renderer_count": fleet["renderer_count"],
    }
    # Consume the field exactly where it lives in the external contract.  The
    # verifier also returns a normalized convenience alias, but launch must
    # not depend on a shape that is absent from the immutable JSON itself.
    materialized["evaluation"]["scenario_contract"] = runtime_contract[
        "live_capture"
    ]["scenario_path"]
    materialized["evaluation"]["scenario_contract_sha256"] = runtime_contract[
        "live_capture"
    ]["scenario_sha256"]
    materialized["production_runtime_contract"] = runtime_contract
    materialized["data_lineage"] = lineage
    materialized["data_lineage_sha256"] = (
        production_data_lineage.data_lineage_sha256(lineage)
    )
    materialized["invariant_profile_sha256"] = invariant_sha
    return materialized, runtime_contract, lineage


def load_learner_contract(path: Path) -> dict[str, Any]:
    contract = _load_contract(
        path,
        LEARNER_CONTRACT_SCHEMA,
        LEARNER_CONTRACT_KEYS,
        "learner contract",
        optional_keys={"heldout_evaluation"},
    )
    _require_equal(
        _exact(
            contract["allocation"],
            {"nodes", "gpus_per_node", "gpu_architecture", "batch_size_per_rank", "global_batch_size"},
            "learner allocation",
        ),
        {
            "nodes": 1,
            "gpus_per_node": 8,
            "gpu_architecture": "H100-SM90",
            "batch_size_per_rank": 8,
            "global_batch_size": 64,
        },
        "learner allocation",
    )
    optimization = _exact(
        contract["optimization"],
        {"optimizer_steps", "save_every_steps", "learning_rate", "weight_decay", "seed", "partition_seed"},
        "learner optimization",
    )
    _require_equal(
        optimization,
        {
            "optimizer_steps": 1200,
            "save_every_steps": 5,
            "learning_rate": 2.5e-5,
            "weight_decay": 0.01,
            "seed": 17,
            "partition_seed": 23,
        },
        "learner optimization",
    )
    cluster = _exact(
        contract["cluster"],
        {"host", "account", "partition", "qos", "slurm_bin", "stage_root", "wandb_netrc"},
        "learner cluster",
    )
    _require_equal(cluster["host"], "oci-nrt", "learner host")
    _require_equal(cluster["account"], "general_cs_infra__ngcsre", "learner account")
    _require_equal(cluster["partition"], "batch_singlenode", "learner partition")
    _require_equal(cluster["qos"], "normal", "learner QOS")
    for key in ("slurm_bin", "stage_root", "wandb_netrc"):
        _absolute(cluster[key], f"cluster {key}")
    _require_equal(cluster["slurm_bin"], "/cm/shared/apps/slurm/current/bin", "Slurm binary root")
    _require_equal(cluster["stage_root"], str(OCI_STAGE_ROOT), "OCI stage root")
    _require_equal(cluster["wandb_netrc"], "/home/ldu/.netrc", "remote W&B netrc path")
    container = _exact(contract["container"], {"path", "sha256", "python"}, "learner container")
    _absolute(container["path"], "learner container")
    _absolute(container["python"], "learner container Python")
    _require_equal(
        container["path"],
        str(OCI_STAGE_ROOT / "containers/roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh"),
        "learner container path",
    )
    _require_equal(container["python"], "/opt/viral-h100/bin/python", "learner container Python")
    _require_equal(
        container["sha256"],
        "f3391b76a66182618f9e71b67e89b0af2d758ffe16afdb164760fcbb4860ebe6",
        "learner container identity",
    )
    handoff = _exact(
        contract["handoff"], {"local_root", "poll_seconds", "maximum_attempts"}, "learner handoff"
    )
    _absolute(handoff["local_root"], "learner handoff root")
    _require_equal(
        handoff["local_root"],
        "/mnt/data2/ldu/roborender_round_dagger_handoff",
        "learner handoff root",
    )
    _require_equal(handoff["poll_seconds"], 30, "learner poll interval")
    _require_equal(handoff["maximum_attempts"], 3, "learner maximum attempts")
    _require_equal(
        _exact(
            contract["wandb"],
            {"mode", "expected_email", "entity", "authorization_scope", "project"},
            "learner W&B",
        ),
        {
            "mode": "online",
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "project": "roborender-round-dagger",
        },
        "learner W&B identity",
    )
    _require_equal(
        _exact(
            contract["source"],
            {"must_be_sealed", "inventory_schema", "attestation_schema"},
            "learner source",
        ),
        {
            "must_be_sealed": True,
            "inventory_schema": "fast_roborender.hsg_source_inventory.v1",
            "attestation_schema": "fast_roborender.hsg_source_snapshot_attestation.v2",
        },
        "learner source",
    )
    if "heldout_evaluation" in contract:
        heldout = _exact(
            contract["heldout_evaluation"],
            {
                "snapshot_path",
                "snapshot_file_sha256",
                "snapshot_sha256",
                "interval_steps",
            },
            "learner held-out evaluation",
        )
        snapshot_path = _absolute(
            heldout["snapshot_path"], "held-out replay snapshot"
        )
        _file_identity(
            snapshot_path,
            heldout["snapshot_file_sha256"],
            "held-out replay snapshot",
        )
        _require_sha(
            heldout["snapshot_sha256"],
            "held-out replay snapshot semantic SHA-256",
        )
        _require_equal(
            heldout["interval_steps"], 100, "held-out evaluation interval"
        )
        try:
            heldout_snapshot = replay.load_heldout_snapshot(
                snapshot_path,
                expected_file_sha256=heldout["snapshot_file_sha256"],
            )
        except Exception as exc:
            raise ProductionLaunchError(
                f"held-out replay snapshot validation failed: {exc}"
            ) from exc
        _require_equal(
            heldout_snapshot["snapshot_sha256"],
            heldout["snapshot_sha256"],
            "held-out replay snapshot semantic identity",
        )
    return contract


def _verify_static_inputs(campaign_contract: Mapping[str, Any]) -> None:
    campaign_artifacts = campaign_contract["artifacts"]
    bootstrap = campaign_artifacts["bootstrap_bundle"]
    _file_identity(Path(bootstrap["checkpoint_path"]), bootstrap["checkpoint_sha256"], "bootstrap checkpoint")
    _file_identity(Path(bootstrap["policy_spec_path"]), bootstrap["policy_spec_sha256"], "bootstrap policy spec")
    _file_identity(Path(bootstrap["manifest_path"]), bootstrap["manifest_sha256"], "bootstrap policy manifest")
    try:
        policy_bundle.validate_policy_bundle_identity(
            checkpoint=bootstrap["checkpoint_path"],
            policy_spec=bootstrap["policy_spec_path"],
            manifest=bootstrap["manifest_path"],
            expected_checkpoint_sha256=bootstrap["checkpoint_sha256"],
            expected_runtime_policy_sha256=bootstrap["runtime_policy_sha256"],
        )
    except policy_bundle.PolicyBundleError as exc:
        raise ProductionLaunchError(f"bootstrap policy bundle is invalid: {exc}") from exc

    teacher = campaign_artifacts["teacher"]
    _file_identity(Path(teacher["checkpoint_path"]), teacher["checkpoint_sha256"], "teacher checkpoint")
    endpoint = campaign_artifacts["endpoint_manifest"]
    _file_identity(Path(endpoint["path"]), endpoint["sha256"], "renderer endpoint manifest")
    endpoint_dir = _directory(Path(endpoint["endpoint_directory"]), "renderer endpoint directory")
    try:
        lines = [line.strip() for line in Path(endpoint["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise ProductionLaunchError("cannot read renderer endpoint manifest") from exc
    _require_equal(len(lines), endpoint["renderer_count"], "renderer endpoint count")
    for index, raw in enumerate(lines):
        descriptor = _regular(_absolute(raw, f"renderer endpoint {index}"), f"renderer endpoint {index}")
        try:
            descriptor.relative_to(endpoint_dir)
        except ValueError as exc:
            raise ProductionLaunchError("renderer endpoint descriptor escaped its accepted directory") from exc

    anchors = campaign_artifacts["anchor_pool"]
    try:
        verify_anchor_pool(
            Path(anchors["path"]),
            expected_count=anchors["anchor_count"],
            expected_sha256=anchors["inventory_sha256"],
        )
    except Exception as exc:
        raise ProductionLaunchError(f"anchor pool verification failed: {exc}") from exc
    camera = campaign_artifacts["camera_acceptance"]
    _file_identity(Path(camera["path"]), camera["sha256"], "camera acceptance")
    split = campaign_contract["motion_split"]
    runtime_contract = campaign_contract.get("production_runtime_contract")
    if isinstance(runtime_contract, Mapping):
        _verify_first_condition_bundles(campaign_contract)
        split_manifest = Path(runtime_contract["live_capture"]["split_manifest_path"])
        heldout_plan = Path(runtime_contract["live_capture"]["heldout_plan_path"])
    else:
        split_manifest = REPOSITORY_ROOT / split["manifest_relative_path"]
        heldout_plan = REPOSITORY_ROOT / split["heldout_plan_relative_path"]
    try:
        verified_plan = motion_split_contract.load_verified_contract(
            manifest_path=split_manifest,
            manifest_sha256=split["manifest_sha256"],
            plan_path=heldout_plan,
            plan_sha256=split["heldout_plan_sha256"],
        )
    except motion_split_contract.MotionSplitContractError as exc:
        raise ProductionLaunchError(f"motion split verification failed: {exc}") from exc
    evaluation = campaign_contract["evaluation"]
    _file_identity(Path(evaluation["hydra_config_path"]), evaluation["hydra_config_sha256"], "evaluation Hydra config")
    scenario_path = Path(evaluation["scenario_contract"])
    if isinstance(runtime_contract, Mapping):
        if (
            not scenario_path.is_absolute()
            or str(scenario_path) != runtime_contract["scenario_path"]
        ):
            raise ProductionLaunchError(
                "materialized evaluation scenario path differs from live capture"
            )
    elif scenario_path.is_absolute() or ".." in scenario_path.parts:
        raise ProductionLaunchError(
            "legacy evaluation scenario path must be repository-relative"
        )
    if not scenario_path.is_absolute():
        scenario_path = REPOSITORY_ROOT / scenario_path
    _file_identity(
        scenario_path,
        evaluation["scenario_contract_sha256"],
        "evaluation scenarios",
    )
    threshold_relative = evaluation["threshold_contract"]
    if (
        not isinstance(threshold_relative, str)
        or Path(threshold_relative).is_absolute()
        or ".." in Path(threshold_relative).parts
    ):
        raise ProductionLaunchError(
            "evaluation threshold path must be repository-relative"
        )
    threshold_path = REPOSITORY_ROOT / threshold_relative
    _file_identity(
        threshold_path,
        evaluation["threshold_contract_sha256"],
        "evaluation thresholds",
    )
    _require_evaluation_case_capacity(
        scenario_path,
        threshold_path,
    )
    try:
        motion_split_contract.validate_scenarios(
            verified_plan,
            artifacts.load_strict_json(
                scenario_path
            ),
        )
    except Exception as exc:
        raise ProductionLaunchError(
            f"evaluation scenarios differ from held-out motion plan: {exc}"
        ) from exc

    runtime = campaign_contract["actor_runtime"]
    docker = _regular(Path(runtime["docker_bin"]), "Docker executable")
    if not os.access(docker, os.X_OK):
        raise ProductionLaunchError("Docker executable is not executable")
    _directory(Path(runtime["gr00t_root"]), "GR00T root")
    for key in ("pair_executor", "pair_rank_script"):
        executable = _regular(Path(runtime[key]), f"actor {key}")
        if not os.access(executable, os.X_OK):
            raise ProductionLaunchError(f"actor {key} is not executable")
    image = campaign_artifacts["actor_image"]
    try:
        completed = subprocess.run(
            [str(docker), "image", "inspect", image["name"], "--format", "{{.Id}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionLaunchError("cannot inspect the accepted actor image") from exc
    if completed.returncode != 0 or completed.stdout.strip() != image["image_id"]:
        raise ProductionLaunchError("actor Docker image ID differs from the accepted image")

    wandb = campaign_contract["wandb"]
    netrc_path = _regular(Path(wandb["netrc"]), "local W&B netrc")
    if netrc_path.stat().st_mode & 0o777 != 0o600:
        raise ProductionLaunchError("local W&B netrc mode must be exactly 0600")
    try:
        credential = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError) as exc:
        raise ProductionLaunchError("local W&B netrc is invalid") from exc
    if (
        credential is None
        or credential[0] != "user"
        or not isinstance(credential[2], str)
        or not credential[2].startswith("wandb_v1_")
    ):
        raise ProductionLaunchError("local W&B netrc has no supported API credential")


def _verify_camera_acceptance_source_bindings(
    campaign_contract: Mapping[str, Any],
    *,
    learner_source_report: Mapping[str, Any],
    evaluation_source_report: Mapping[str, Any],
) -> None:
    """Bind accepted camera evidence to the nested renderer implementation.

    The outer repository may receive control-plane hardening after an operator
    accepts a calibration panel.  The nested GR00T source, however, owns the
    Isaac camera construction and render cache.  Any change to that exact tree
    invalidates the visual acceptance and must force a new calibration review.
    """

    camera = campaign_contract["artifacts"]["camera_acceptance"]
    path = Path(camera["path"])
    _file_identity(path, camera["sha256"], "camera acceptance")
    try:
        acceptance = artifacts.load_strict_json(path)
    except Exception as exc:
        raise ProductionLaunchError("cannot load strict camera acceptance") from exc
    if not isinstance(acceptance, Mapping):
        raise ProductionLaunchError("camera acceptance must be a JSON object")

    accepted_revision = acceptance.get("gr00t_revision")
    if not isinstance(accepted_revision, str) or REVISION_RE.fullmatch(
        accepted_revision
    ) is None:
        raise ProductionLaunchError("camera acceptance GR00T revision is malformed")
    accepted_tree = _require_sha(
        acceptance.get("gr00t_source_tree_sha256"),
        "camera acceptance GR00T source tree",
    )
    for label, report in (
        ("learner", learner_source_report),
        ("evaluation", evaluation_source_report),
    ):
        _require_equal(
            report.get("gr00t_revision"),
            accepted_revision,
            f"{label} source/camera acceptance GR00T revision",
        )
        _require_equal(
            _require_sha(
                report.get("gr00t_source_tree_sha256"),
                f"{label} GR00T source tree",
            ),
            accepted_tree,
            f"{label} source/camera acceptance GR00T tree",
        )


def _verify_remote_source(
    root: str,
    inventory_sha256: str,
    *,
    learner_contract: Mapping[str, Any],
    learner_contract_path: Path,
) -> dict[str, Any]:
    if (
        SAFE_REMOTE_PATH_RE.fullmatch(root) is None
        or "//" in root
        or "/../" in root
        or "/./" in root
        or root.endswith(("/..", "/."))
    ):
        raise ProductionLaunchError("sealed remote source root is unsafe")
    accepted_prefix = learner_contract["cluster"]["stage_root"] + "/src_snapshots/"
    if not root.startswith(accepted_prefix) or root == accepted_prefix:
        raise ProductionLaunchError(
            "sealed remote source root is outside the accepted OCI source area"
        )
    _require_sha(inventory_sha256, "sealed source inventory SHA-256")
    ssh = shutil.which("ssh")
    if ssh is None:
        raise ProductionLaunchError("ssh is unavailable for sealed remote source verification")
    host = learner_contract["cluster"]["host"]
    verifier = f"{root}/scripts/round_dagger/verify_sealed_source_snapshot.py"
    command = [
        ssh,
        "-o",
        "BatchMode=yes",
        host,
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "python3",
        verifier,
        "--root",
        root,
        "--expected-inventory-sha256",
        inventory_sha256,
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionLaunchError("sealed remote source verification could not run") from exc
    if completed.returncode != 0:
        raise ProductionLaunchError("sealed remote source verification failed")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProductionLaunchError("sealed source verifier returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise ProductionLaunchError("sealed source verifier returned a non-object")
    _require_equal(report.get("schema"), SEALED_REPORT_SCHEMA, "sealed source report schema")
    _require_equal(report.get("status"), "PASS", "sealed source status")
    _require_equal(report.get("root"), root, "sealed source report root")
    _require_equal(report.get("inventory_sha256"), inventory_sha256, "sealed source inventory")
    for key in ("fast_roborender_revision", "gr00t_revision"):
        if not isinstance(report.get(key), str) or REVISION_RE.fullmatch(report[key]) is None:
            raise ProductionLaunchError(f"sealed source report {key} is malformed")

    # The remote learner must consume the exact checked-in learner contract,
    # not another file that happens to be covered by the same operator-provided
    # inventory string.
    remote_contract = f"{root}/configs/round_dagger/{learner_contract_path.name}"
    try:
        remote_hash = subprocess.run(
            [ssh, "-o", "BatchMode=yes", host, "sha256sum", "--", remote_contract],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionLaunchError("sealed learner contract verification could not run") from exc
    expected_contract_sha = artifacts.sha256_file(learner_contract_path)
    if (
        remote_hash.returncode != 0
        or not remote_hash.stdout.split()
        or remote_hash.stdout.split()[0] != expected_contract_sha
    ):
        raise ProductionLaunchError("sealed source contains a different learner contract")
    return report


def _verify_local_evaluation_seal(
    root: str | Path,
    inventory_sha256: str,
    *,
    campaign_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the RTX evaluator to an independently versioned local source seal.

    The learner may have completed from an older OCI snapshot.  Evaluation has
    its own explicit inventory identity so a later hardening commit can execute
    without rewriting learner lineage.  No caller may supply the teacher or
    evaluator path independently from this verified root.
    """

    local_root = _absolute(str(root), "local sealed source root")
    try:
        resolved_root = local_root.resolve(strict=True)
    except OSError as exc:
        raise ProductionLaunchError(
            "local sealed source root is unavailable"
        ) from exc
    if resolved_root != local_root:
        raise ProductionLaunchError(
            "local sealed source root must be an exact real path"
        )
    try:
        publication_info = local_root.parent.lstat()
    except OSError as exc:
        raise ProductionLaunchError(
            "local sealed source publication parent is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(publication_info.st_mode)
        or not stat.S_ISDIR(publication_info.st_mode)
        or stat.S_IMODE(publication_info.st_mode) != 0o555
    ):
        raise ProductionLaunchError(
            "local sealed source publication parent must be exactly 0555"
        )
    if not any(
        local_root == mount or mount in local_root.parents
        for mount in (Path("/mnt/data1"), Path("/mnt/data2"))
    ):
        raise ProductionLaunchError(
            "local sealed source root is outside the RTX Docker mount roots"
        )

    expected_inventory = _require_sha(
        inventory_sha256, "evaluation source inventory SHA-256"
    )
    try:
        local_report = verify_sealed_source_snapshot(
            local_root, expected_inventory
        )
    except (OSError, SealedSourceVerificationError) as exc:
        raise ProductionLaunchError(
            "local sealed source verification failed"
        ) from exc
    _require_equal(
        local_report.get("schema"),
        SEALED_REPORT_SCHEMA,
        "local sealed source report schema",
    )
    _require_equal(local_report.get("status"), "PASS", "local sealed source status")
    _require_equal(
        local_report.get("root"),
        str(local_root),
        "local sealed source report root",
    )
    _require_equal(
        local_report.get("inventory_sha256"),
        expected_inventory,
        "evaluation sealed source inventory",
    )
    _require_equal(
        local_report.get("snapshot_name"),
        local_root.name,
        "evaluation sealed source snapshot name",
    )

    sealed_teacher = local_root / TEACHER_CHECKPOINT_RELATIVE
    teacher = campaign_contract["artifacts"]["teacher"]
    _file_identity(
        sealed_teacher,
        teacher["checkpoint_sha256"],
        "local sealed evaluation teacher",
    )
    teacher_stat = sealed_teacher.stat()
    if teacher_stat.st_mode & 0o777 != 0o444:
        raise ProductionLaunchError(
            "local sealed evaluation teacher mode must be exactly 0444"
        )
    if teacher_stat.st_nlink != 1:
        raise ProductionLaunchError(
            "local sealed evaluation teacher must have exactly one hard link"
        )
    configured_teacher = _regular(
        Path(teacher["checkpoint_path"]), "configured teacher checkpoint"
    )
    configured_stat = configured_teacher.stat()
    if (configured_stat.st_dev, configured_stat.st_ino) == (
        teacher_stat.st_dev,
        teacher_stat.st_ino,
    ):
        raise ProductionLaunchError(
            "local sealed teacher and configured target must be distinct files"
        )
    evaluation = campaign_contract["evaluation"]
    split = campaign_contract["motion_split"]
    for relative_key, sha_key, label in (
        (
            "manifest_relative_path",
            "manifest_sha256",
            "motion split manifest",
        ),
        (
            "heldout_plan_relative_path",
            "heldout_plan_sha256",
            "held-out evaluation plan",
        ),
    ):
        _file_identity(
            local_root / split[relative_key],
            split[sha_key],
            f"sealed {label}",
        )
    # The finalized scenario contract is post-seal live-capture evidence and
    # therefore must not require a Git update.  Thresholds remain a checked-in
    # invariant; the scenario bytes were already verified from the external
    # runtime contract and its two-pass attestation.
    _file_identity(
        local_root / evaluation["threshold_contract"],
        evaluation["threshold_contract_sha256"],
        "sealed evaluation thresholds",
    )
    sealed_campaign_contract = artifacts.load_strict_json(
        _regular(
            local_root / PRODUCTION_CAMPAIGN_CONTRACT_RELATIVE,
            "sealed production campaign contract",
        )
    )
    if (
        sealed_campaign_contract.get("artifacts", {}).get("anchor_pool")
        != campaign_contract["artifacts"]["anchor_pool"]
        or sealed_campaign_contract.get("motion_split")
        != campaign_contract["motion_split"]
    ):
        raise ProductionLaunchError(
            "sealed evaluation anchor/split contract differs from launch contract"
        )
    return {
        **local_report,
        "sealed_teacher_checkpoint": str(sealed_teacher),
        "sealed_teacher_sha256": teacher["checkpoint_sha256"],
    }


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=not binary,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionLaunchError(
            f"cannot inspect local source repository: {repository}"
        ) from exc


def _git_text(repository: Path, arguments: Sequence[str], label: str) -> str:
    completed = _run_git(repository, arguments)
    if completed.returncode != 0:
        raise ProductionLaunchError(f"cannot read {label}: {repository}")
    return completed.stdout.strip()


def _validate_allowed_untracked(
    repository: Path,
    *,
    label: str,
    allowed: set[str],
) -> list[str]:
    completed = _run_git(
        repository,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        binary=True,
    )
    if completed.returncode != 0:
        raise ProductionLaunchError(f"cannot enumerate untracked {label} source")
    raw_output = completed.stdout
    if not isinstance(raw_output, bytes):
        raise ProductionLaunchError(f"invalid untracked {label} source response")
    observed = [
        item.decode("utf-8", "surrogateescape")
        for item in raw_output.split(b"\0")
        if item
    ]
    unexpected = sorted(set(observed) - allowed)
    if unexpected:
        raise ProductionLaunchError(
            f"unexpected untracked {label} source: {unexpected[0]}"
        )
    if len(observed) != len(set(observed)):
        raise ProductionLaunchError(f"duplicate untracked {label} path")
    for relative in observed:
        path = repository / relative
        try:
            info = path.lstat()
        except OSError as exc:
            raise ProductionLaunchError(
                f"allowed untracked {label} sentinel vanished: {relative}"
            ) from exc
        if path.is_symlink() or not path.is_file():
            raise ProductionLaunchError(
                f"allowed untracked {label} sentinel is unsafe: {relative}"
            )
        if relative == "core":
            if label != "GR00T" or info.st_uid != 0:
                raise ProductionLaunchError(
                    "untracked GR00T core is allowed only when it is root-owned"
                )
        elif relative == ".source_revision":
            # This marker is generated evidence from an earlier sealed export,
            # not live source.  Match the staging script's policy by allowing
            # the exact regular-file pathname without trusting its contents.
            pass
        elif relative != "-" or label != "fast_roborender":
            raise ProductionLaunchError(
                f"unrecognized allowed untracked {label} sentinel: {relative}"
            )
    return observed


def _verify_one_local_repository(
    repository: Path,
    *,
    label: str,
    expected_revision: str,
    allowed_untracked: set[str],
) -> dict[str, Any]:
    _directory(repository, f"local {label} repository")
    toplevel = Path(
        _git_text(repository, ["rev-parse", "--show-toplevel"], f"{label} toplevel")
    )
    try:
        resolved_repository = repository.resolve(strict=True)
        resolved_toplevel = toplevel.resolve(strict=True)
    except OSError as exc:
        raise ProductionLaunchError(f"cannot resolve local {label} repository") from exc
    if resolved_toplevel != resolved_repository:
        raise ProductionLaunchError(f"local {label} repository toplevel mismatch")
    revision = _git_text(
        repository, ["rev-parse", "--verify", "HEAD^{commit}"], f"{label} HEAD"
    )
    if REVISION_RE.fullmatch(revision) is None:
        raise ProductionLaunchError(f"local {label} HEAD is malformed")
    _require_equal(revision, expected_revision, f"local/remote {label} revision")
    for arguments, dirty_label in (
        (["diff", "--quiet", "--no-ext-diff", "--ignore-submodules", "--"], "worktree"),
        (["diff", "--cached", "--quiet", "--no-ext-diff", "--ignore-submodules", "--"], "index"),
    ):
        completed = _run_git(repository, arguments)
        if completed.returncode == 1:
            raise ProductionLaunchError(f"local {label} tracked {dirty_label} is dirty")
        if completed.returncode != 0:
            raise ProductionLaunchError(f"cannot verify local {label} tracked {dirty_label}")
    # With both diffs clean, HEAD's tree is also the exact index tree.  Read it
    # via rev-parse instead of `git write-tree`, which may create Git objects
    # and would make a validation-only dry run unnecessarily stateful.
    index_tree = _git_text(
        repository, ["rev-parse", "--verify", "HEAD^{tree}"], f"{label} index tree"
    )
    if REVISION_RE.fullmatch(index_tree) is None:
        raise ProductionLaunchError(f"local {label} index tree is malformed")
    observed_untracked = _validate_allowed_untracked(
        repository,
        label=label,
        allowed=allowed_untracked,
    )
    return {
        "revision": revision,
        "index_tree": index_tree,
        "allowed_untracked": sorted(observed_untracked),
    }


def _verify_local_source_checkout(
    source_report: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    fast_revision = source_report.get("fast_roborender_revision")
    gr00t_revision = source_report.get("gr00t_revision")
    if not isinstance(fast_revision, str) or REVISION_RE.fullmatch(fast_revision) is None:
        raise ProductionLaunchError("sealed source fast_roborender revision is malformed")
    if not isinstance(gr00t_revision, str) or REVISION_RE.fullmatch(gr00t_revision) is None:
        raise ProductionLaunchError("sealed source GR00T revision is malformed")
    root = repository_root.resolve(strict=True)
    gr00t = root / "GR00T-VisualSim2Real"
    fast = _verify_one_local_repository(
        root,
        label="fast_roborender",
        expected_revision=fast_revision,
        allowed_untracked={"-", ".source_revision"},
    )
    nested = _verify_one_local_repository(
        gr00t,
        label="GR00T",
        expected_revision=gr00t_revision,
        allowed_untracked={".source_revision", "core"},
    )
    return {
        "schema": "fast_roborender.round_dagger.local_source_checkout.v1",
        "status": "clean_and_revision_matched",
        "fast_roborender": fast,
        "gr00t": nested,
    }


def derive_learner_bundle_sha256(
    *,
    base_policy_spec: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_sha256: str,
    runtime_policy_sha256: str,
    campaign_id: str,
    macro_round: int,
    code_sha256: str,
    learner_contract: Mapping[str, Any],
) -> str:
    """Derive the exact H100 ``LearnerBundleConfig`` digest without Torch.

    ``prepare_gr00t_learner_bundle.prepare_bundle`` imports the GPU learner
    module and therefore Torch.  Production preflight intentionally runs in a
    lightweight control-plane Python, so reproduce its canonical mapping here
    using only the portable policy validator.  The mapping below is the exact
    normalized output of ``LearnerBundleConfig.to_mapping``.
    """

    expected_parent = _require_sha(
        expected_parent_checkpoint_sha256, "active parent checkpoint SHA-256"
    )
    actual_parent = artifacts.sha256_file(
        _regular(parent_checkpoint, "active parent checkpoint")
    )
    _require_equal(actual_parent, expected_parent, "active parent checkpoint identity")
    runtime_sha = _require_sha(runtime_policy_sha256, "active runtime SHA-256")
    code_sha = _require_sha(code_sha256, "learner code SHA-256")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ProductionLaunchError("campaign ID must be a non-empty string")
    if type(macro_round) is not int or macro_round < 0:
        raise ProductionLaunchError("macro-round must be a non-negative integer")

    try:
        policy_spec = load_policy_spec(
            _regular(base_policy_spec, "active base policy spec")
        )
        policy_spec = validate_policy_spec(policy_spec)
    except Exception as exc:
        raise ProductionLaunchError(
            f"cannot derive the active per-round learner bundle: {exc}"
        ) from exc
    _require_equal(
        policy_spec["checkpoint_sha256"],
        actual_parent,
        "active base policy spec/checkpoint identity",
    )
    _require_equal(
        policy_spec["runtime_policy_sha256"],
        runtime_sha,
        "active base policy spec/runtime identity",
    )
    if policy_spec["input_contract"]["vision_shape"] != list(VISION_SHAPE):
        raise ProductionLaunchError("portable policy does not use the production view")
    expected_output = {
        "policy_action_dim": POLICY_ACTION_DIM,
        "homie_action_dim": HOMIE_ACTION_DIM,
        "replay_action_dim": REPLAY_ACTION_DIM,
        "env_command_dim": ENV_COMMAND_DIM,
        "train_action_slice": list(TRAIN_ACTION_SLICE),
        "object_prediction_dim": OBJECT_PREDICTION_DIM,
    }
    _require_equal(
        policy_spec["output_contract"], expected_output, "portable policy action contract"
    )

    allocation = learner_contract["allocation"]
    optimization = learner_contract["optimization"]
    inputs = policy_spec["input_contract"]
    heldout_fields: dict[str, Any] = {}
    heldout_contract = learner_contract.get("heldout_evaluation")
    if heldout_contract is not None:
        heldout_path = _regular(
            _absolute(
                heldout_contract["snapshot_path"], "held-out replay snapshot"
            ),
            "held-out replay snapshot",
        )
        try:
            heldout_snapshot = replay.load_heldout_snapshot(
                heldout_path,
                expected_file_sha256=heldout_contract[
                    "snapshot_file_sha256"
                ],
            )
        except Exception as exc:
            raise ProductionLaunchError(
                f"cannot derive held-out learner identity: {exc}"
            ) from exc
        _require_equal(
            heldout_snapshot["campaign_id"], campaign_id, "held-out campaign"
        )
        _require_equal(
            heldout_snapshot["snapshot_sha256"],
            heldout_contract["snapshot_sha256"],
            "held-out semantic identity",
        )
        _require_equal(
            heldout_snapshot["view_contract"],
            {
                "dtype": "uint8",
                "layout": "vertical",
                "shape": list(VISION_SHAPE),
                "views": ["head_color", "hand_left", "hand_right"],
            },
            "held-out view contract",
        )
        _require_equal(
            heldout_snapshot["action_contract"],
            {"dtype": "float32", "shape": [REPLAY_ACTION_DIM]},
            "held-out action contract",
        )
        _require_equal(
            heldout_contract["interval_steps"],
            100,
            "held-out evaluation interval",
        )
        heldout_fields = {
            "heldout_snapshot_sha256": heldout_snapshot["snapshot_sha256"],
            "heldout_snapshot_file_sha256": heldout_snapshot["file_sha256"],
            "heldout_eval_interval": 100,
        }
    bundle = {
        "schema": LEARNER_BUNDLE_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "model_lineage_sha256": actual_parent,
        "code_sha256": code_sha,
        "rgb_mean": [float(item) for item in inputs["rgb_mean"]],
        "rgb_std": [float(item) for item in inputs["rgb_std"]],
        "rgb_input_key": "vision_obs",
        "rgb_layout": "NHWC",
        "output_key": "actions",
        "observation_key_map": {},
        "train_action_slice": list(TRAIN_ACTION_SLICE),
        "batch_size": allocation["batch_size_per_rank"],
        "max_steps": optimization["optimizer_steps"],
        "save_every": optimization["save_every_steps"],
        "learning_rate": float(optimization["learning_rate"]),
        "weight_decay": float(optimization["weight_decay"]),
        "seed": optimization["seed"],
        "partition_seed": optimization["partition_seed"],
        "initial_checkpoint_sha256": actual_parent,
        "model_config": policy_spec,
        **heldout_fields,
    }
    try:
        return artifacts.canonical_json_sha256(bundle)
    except Exception as exc:
        raise ProductionLaunchError(
            f"cannot canonicalize the active per-round learner bundle: {exc}"
        ) from exc


def _validate_orchestration_and_campaign(
    orchestration_path: Path,
    campaign_contract: Mapping[str, Any],
    learner_contract: Mapping[str, Any],
    learner_deployment_contract_sha256: str,
    source_inventory_sha256: str,
    evaluation_source_root: Path,
) -> tuple[MacroRoundConfig, dict[str, Any], dict[str, Any], str]:
    _regular(orchestration_path, "orchestration config")
    if orchestration_path.stat().st_mode & 0o222:
        raise ProductionLaunchError("orchestration config must be immutable (non-writable)")
    try:
        configuration = MacroRoundConfig.load(orchestration_path)
    except Exception as exc:
        raise ProductionLaunchError(f"strict orchestration config is invalid: {exc}") from exc
    if configuration.macro_round_count != PRODUCTION_MACRO_ROUND_COUNT:
        raise ProductionLaunchError(
            "production orchestration config must bind exactly five macro-rounds"
        )
    if not 0 <= configuration.macro_round < PRODUCTION_MACRO_ROUND_COUNT:
        raise ProductionLaunchError(
            "production campaign contains exactly five macro-rounds (0 through 4)"
        )
    root = _directory(configuration.campaign_root, "campaign root")
    heldout_contract = learner_contract.get("heldout_evaluation")
    if heldout_contract is not None:
        heldout_path = _regular(
            _absolute(
                heldout_contract["snapshot_path"], "held-out replay snapshot"
            ),
            "held-out replay snapshot",
        )
        try:
            heldout_relative = heldout_path.relative_to(root)
        except ValueError as exc:
            raise ProductionLaunchError(
                "held-out replay snapshot must be inside the campaign"
            ) from exc
        if heldout_relative.parts and heldout_relative.parts[0] in {
            "actors",
            "operator",
        }:
            raise ProductionLaunchError(
                "held-out replay snapshot is excluded from learner projection"
            )
        try:
            heldout_snapshot = replay.load_heldout_snapshot(
                heldout_path,
                expected_file_sha256=heldout_contract[
                    "snapshot_file_sha256"
                ],
            )
        except Exception as exc:
            raise ProductionLaunchError(
                f"held-out replay snapshot validation failed: {exc}"
            ) from exc
        referenced_paths = [
            *(
                Path(item["cohort_manifest_path"])
                for item in heldout_snapshot["cohorts"]
            ),
            *(
                Path(item["shard_manifest_path"])
                for item in heldout_snapshot["resolved_shards"]
            ),
            *(
                Path(item["payload_path"])
                for item in heldout_snapshot["resolved_shards"]
            ),
        ]
        for referenced_path in referenced_paths:
            try:
                relative = referenced_path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise ProductionLaunchError(
                    "held-out replay artifact must be inside the campaign"
                ) from exc
            if relative.parts and relative.parts[0] in {"actors", "operator"}:
                raise ProductionLaunchError(
                    "held-out replay artifact is excluded from learner projection"
                )
    _require_equal(configuration.actor_root, root / "actors", "orchestration actor root")
    _require_equal(configuration.learner_code_sha256, source_inventory_sha256, "learner/source identity")
    _require_equal(
        configuration.evaluator_config_sha256,
        campaign_contract["evaluation"]["hydra_config_sha256"],
        "evaluator config identity",
    )
    expected_hooks = {
        "collect": (str(REPOSITORY_ROOT / "scripts/round_dagger/run_production_actor_cohort_hook.sh"), 14400),
        "transfer": None,
        "ingest": None,
        "train": (str(REPOSITORY_ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"), 172800),
        "evaluate": (
            str(
                evaluation_source_root
                / "scripts/round_dagger/run_production_paired_evaluation_hook.sh"
            ),
            28800,
        ),
        "promote": None,
    }
    for name, expected in expected_hooks.items():
        observed = configuration.hooks[name]
        if expected is None:
            if observed is not None:
                raise ProductionLaunchError(f"{name} must use the coordinator built-in")
            continue
        if observed is None or observed.argv != (expected[0],) or observed.timeout_seconds != expected[1]:
            raise ProductionLaunchError(f"{name} hook differs from the production primitive")

    try:
        campaign = state.load_campaign(root)
        coordinator = state.load_state(root)
    except Exception as exc:
        raise ProductionLaunchError(f"durable campaign artifacts are invalid: {exc}") from exc
    expected_campaign = {
        "campaign_id": campaign_contract["campaign_id"],
        "target_environments": 256,
        "active_environments": 64,
        "cohort_count": 4,
        "steps_per_environment": 512,
        "macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
        "view_names": ["head_color", "hand_left", "hand_right"],
        "policy_image_shape": [324, 192, 3],
        "action_dim": 26,
        "teacher_sha256": campaign_contract["artifacts"]["teacher"]["checkpoint_sha256"],
        "renderer_sha256": campaign_contract["artifacts"]["renderer"]["cohort_identity_sha256"],
        "camera_mapping_sha256": campaign_contract["artifacts"]["camera_acceptance"]["sha256"],
        "training_config_sha256": campaign_contract["evaluation"]["hydra_config_sha256"],
        "code_sha256": source_inventory_sha256,
        # This is the immutable bootstrap/deployment contract identity.  It is
        # deliberately distinct from the per-round LearnerBundleConfig digest
        # consumed by the H100 learner.
        "policy_config_sha256": learner_deployment_contract_sha256,
        "beta_schedule": campaign_contract["beta_schedule"],
        "data_lineage": campaign_contract["data_lineage"],
        "data_lineage_sha256": campaign_contract["data_lineage_sha256"],
    }
    mismatches = [key for key, value in expected_campaign.items() if campaign.get(key) != value]
    if mismatches:
        raise ProductionLaunchError(f"campaign manifest differs from production contract: {mismatches}")
    _require_equal(
        configuration.campaign_contract_sha256,
        campaign_contract["production_runtime_contract"]["contract_sha256"],
        "orchestration/runtime campaign contract identity",
    )
    active = coordinator["active_policy"]
    _require_equal(configuration.runtime_policy_sha256, active["runtime_policy_sha256"], "active runtime identity")
    _require_equal(configuration.learner_model_lineage_sha256, active["checkpoint_sha256"], "active model lineage")
    if coordinator["phase"] == "READY":
        if coordinator["next_macro_round"] not in {
            configuration.macro_round,
            configuration.macro_round + 1,
        }:
            raise ProductionLaunchError("orchestration macro-round is stale or skips READY state")
    elif coordinator.get("open_macro_round") != configuration.macro_round:
        raise ProductionLaunchError("orchestration macro-round differs from the open durable round")

    checkpoint = root / active["checkpoint_path"]
    try:
        policy_bundle.validate_policy_bundle_identity(
            checkpoint=checkpoint,
            policy_spec=f"{checkpoint}.policy_spec.json",
            manifest=f"{checkpoint}.manifest.json",
            expected_checkpoint_sha256=active["checkpoint_sha256"],
            expected_runtime_policy_sha256=active["runtime_policy_sha256"],
        )
    except policy_bundle.PolicyBundleError as exc:
        raise ProductionLaunchError(f"active campaign policy bundle is invalid: {exc}") from exc

    derived_learner_bundle_sha256 = derive_learner_bundle_sha256(
        base_policy_spec=Path(f"{checkpoint}.policy_spec.json"),
        parent_checkpoint=checkpoint,
        expected_parent_checkpoint_sha256=active["checkpoint_sha256"],
        runtime_policy_sha256=active["runtime_policy_sha256"],
        campaign_id=campaign["campaign_id"],
        macro_round=configuration.macro_round,
        code_sha256=source_inventory_sha256,
        learner_contract=learner_contract,
    )
    _require_equal(
        configuration.learner_config_sha256,
        derived_learner_bundle_sha256,
        "derived learner bundle identity",
    )
    return configuration, campaign, coordinator, derived_learner_bundle_sha256


def build_environment(
    campaign_contract: Mapping[str, Any],
    learner_contract: Mapping[str, Any],
    *,
    sealed_source_root: str,
    sealed_source_inventory_sha256: str,
    evaluation_source_root: str,
    evaluation_source_inventory_sha256: str,
    sealed_teacher_checkpoint: str,
    learner_contract_sha256: str,
) -> dict[str, str]:
    artifact = campaign_contract["artifacts"]
    observation = campaign_contract["observation_contract"]
    runtime = campaign_contract["actor_runtime"]
    telemetry = campaign_contract["telemetry"]
    wandb = campaign_contract["wandb"]
    evaluation = campaign_contract["evaluation"]
    split = campaign_contract["motion_split"]
    production_runtime = campaign_contract["production_runtime_contract"]
    cluster = learner_contract["cluster"]
    container = learner_contract["container"]
    handoff = learner_contract["handoff"]
    optimization = learner_contract["optimization"]
    environment = {
        "ROUND_DAGGER_TEACHER_CHECKPOINT": artifact["teacher"]["checkpoint_path"],
        "ROUND_DAGGER_TEACHER_SHA256": artifact["teacher"]["checkpoint_sha256"],
        "ROUND_DAGGER_TEACHER_CHECKPOINT_RELATIVE": TEACHER_CHECKPOINT_RELATIVE.as_posix(),
        "ROUND_DAGGER_EVALUATION_SEALED_TEACHER": sealed_teacher_checkpoint,
        "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": evaluation_source_root,
        "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": evaluation_source_inventory_sha256,
        "ROUND_DAGGER_COLLECTION_MOTION_SPLIT_MANIFEST": str(
            Path(evaluation_source_root) / split["manifest_relative_path"]
        ),
        "ROUND_DAGGER_EVALUATION_MOTION_SPLIT_MANIFEST": str(
            Path(evaluation_source_root) / split["manifest_relative_path"]
        ),
        "ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256": split["manifest_sha256"],
        "ROUND_DAGGER_CAMPAIGN_CONTRACT": production_runtime["contract_path"],
        "ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256": production_runtime[
            "contract_sha256"
        ],
        "ROUND_DAGGER_DATA_LINEAGE_SHA256": campaign_contract[
            "data_lineage_sha256"
        ],
        "ROUND_DAGGER_INVARIANT_PROFILE_SHA256": campaign_contract[
            "invariant_profile_sha256"
        ],
        "ROUND_DAGGER_CAPTURE_ATTESTATION": production_runtime["live_capture"][
            "attestation_path"
        ],
        "ROUND_DAGGER_CAPTURE_ATTESTATION_SHA256": production_runtime[
            "live_capture"
        ]["attestation_sha256"],
        "ROUND_DAGGER_CAPTURE_EVIDENCE": production_runtime["live_capture"][
            "evidence_path"
        ],
        "ROUND_DAGGER_CAPTURE_EVIDENCE_SHA256": production_runtime[
            "live_capture"
        ]["evidence_sha256"],
        "ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256": (
            production_runtime["live_capture"][
                "first_condition_bundle_inventory_sha256"
            ]
        ),
        "ROUND_DAGGER_HELDOUT_EVALUATION_PLAN": str(
            Path(evaluation_source_root) / split["heldout_plan_relative_path"]
        ),
        "ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256": split[
            "heldout_plan_sha256"
        ],
        "ROUND_DAGGER_LEARNER_SOURCE_INVENTORY_SHA256": sealed_source_inventory_sha256,
        "ROUND_DAGGER_RENDERER_SHA256": artifact["renderer"]["cohort_identity_sha256"],
        "ROUND_DAGGER_RENDERER_RUNTIME_SHA256": artifact["renderer"]["runtime_identity_sha256"],
        "IMAGE3F_ENDPOINT_MANIFEST": artifact["endpoint_manifest"]["path"],
        "ROUND_DAGGER_ENDPOINT_MANIFEST_SHA256": artifact[
            "endpoint_manifest"
        ]["sha256"],
        "ANCHORS": artifact["anchor_pool"]["path"],
        "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": artifact["anchor_pool"]["inventory_sha256"],
        "ROUND_DAGGER_ANCHOR_INDEX": "0",
        "IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE": observation[
            "bootstrap_anchor_source"
        ],
        "ROUND_DAGGER_CAMERA_ACCEPTANCE": artifact["camera_acceptance"]["path"],
        "ROUND_DAGGER_CAMERA_ACCEPTANCE_SHA256": artifact["camera_acceptance"]["sha256"],
        "ROUND_DAGGER_RTX_IMAGE": artifact["actor_image"]["name"],
        "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": artifact["actor_image"]["image_id"].removeprefix("sha256:"),
        "ROUND_DAGGER_DOCKER_BIN": runtime["docker_bin"],
        "GR00T_ROOT": runtime["gr00t_root"],
        "GR00T_PYTHON": runtime["gr00t_python"],
        "ROUND_DAGGER_ACTOR_PAIR_EXECUTOR": runtime["pair_executor"],
        "ROUND_DAGGER_ACTOR_PAIR_RANK_SCRIPT": runtime["pair_rank_script"],
        "ROUND_DAGGER_ACTOR_PAIR_STAGGER_SECONDS": str(runtime["rank_stagger_seconds"]),
        "ROUND_DAGGER_ACTOR_DOCKER_IPC": runtime["ipc"],
        "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": runtime["shm_size"],
        "ROUND_DAGGER_ACTOR_DOCKER_GPU_IDS": runtime["gpu_ids"],
        "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": runtime["mount_roots"],
        "ROUND_DAGGER_EXPECTED_RENDERERS": str(artifact["endpoint_manifest"]["renderer_count"]),
        "ROUND_DAGGER_MIN_ANCHORS": str(artifact["anchor_pool"]["anchor_count"]),
        "ROUND_DAGGER_ENVS_PER_SHARD": str(runtime["environments_per_shard"]),
        "ROUND_DAGGER_ROLLOUT_ID": str(runtime["rollout_id"]),
        "ROUND_DAGGER_MACRO_ROUND_COUNT": str(PRODUCTION_MACRO_ROUND_COUNT),
        "ROUND_DAGGER_TRANSITIONS_PER_CAMPAIGN": str(TRANSITIONS_PER_CAMPAIGN),
        # Production remains pinned to the previously validated rollout.  The
        # runtime adapter shadow is a separate bounded-validation surface and
        # must never be selected through an ambient operator environment.
        "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "legacy",
        "IMAGE_DAGGER_TELEMETRY_REQUIRED": "1" if telemetry["required"] else "0",
        "IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO": "1" if telemetry["require_video"] else "0",
        "IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY": str(telemetry["scalar_every_steps"]),
        "IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY": str(telemetry["image_every_steps"]),
        "IMAGE_DAGGER_TELEMETRY_CLIP_EVERY": str(telemetry["clip_every_steps"]),
        "IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES": str(telemetry["clip_frames"]),
        "IMAGE_DAGGER_TELEMETRY_CLIP_FPS": str(telemetry["clip_fps"]),
        "ROUND_DAGGER_WANDB_ENABLED": "1" if wandb["enabled"] else "0",
        "ROUND_DAGGER_WANDB_PYTHON": wandb["python"],
        "ROUND_DAGGER_WANDB_ENTITY": wandb["entity"],
        "ROUND_DAGGER_WANDB_PROJECT": wandb["project"],
        "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": wandb["expected_email"],
        "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE": wandb["authorization_scope"],
        "ROUND_DAGGER_WANDB_NETRC": wandb["netrc"],
        "ROUND_DAGGER_EVALUATION_CONFIG": evaluation["hydra_config_path"],
        "ROUND_DAGGER_EVALUATION_SCENARIOS": str(
            Path(evaluation["scenario_contract"])
            if Path(evaluation["scenario_contract"]).is_absolute()
            else Path(evaluation_source_root) / evaluation["scenario_contract"]
        ),
        "ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256": evaluation["scenario_contract_sha256"],
        "ROUND_DAGGER_EVALUATION_THRESHOLDS": str(Path(evaluation_source_root) / evaluation["threshold_contract"]),
        "ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256": evaluation["threshold_contract_sha256"],
        "ROUND_DAGGER_EVALUATION_ENDPOINT_DIR": artifact["endpoint_manifest"]["endpoint_directory"],
        "ROUND_DAGGER_CONTROL_PYTHON": "/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python",
        "ROUND_DAGGER_PYTHON": "/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python",
        "ROUND_DAGGER_REMOTE_HOST": cluster["host"],
        "ROUND_DAGGER_REMOTE_ACCOUNT": cluster["account"],
        "ROUND_DAGGER_REMOTE_PARTITION": cluster["partition"],
        "ROUND_DAGGER_REMOTE_QOS": cluster["qos"],
        "ROUND_DAGGER_REMOTE_STAGE_ROOT": cluster["stage_root"],
        "ROUND_DAGGER_REMOTE_SLURM_BIN": cluster["slurm_bin"],
        "ROUND_DAGGER_REMOTE_WANDB_NETRC": cluster["wandb_netrc"],
        "ROUND_DAGGER_REMOTE_SOURCE_ROOT": sealed_source_root,
        "ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256": sealed_source_inventory_sha256,
        "ROUND_DAGGER_REMOTE_CONTAINER": container["path"],
        "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": container["sha256"],
        "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": _require_sha(
            learner_contract_sha256, "learner deployment contract SHA-256"
        ),
        "ROUND_DAGGER_REMOTE_HANDOFF_ROOT": handoff["local_root"],
        "ROUND_DAGGER_REMOTE_POLL_SECONDS": str(handoff["poll_seconds"]),
        "ROUND_DAGGER_REMOTE_MAX_ATTEMPTS": str(handoff["maximum_attempts"]),
        "ROUND_DAGGER_LEARNER_BATCH_SIZE": str(learner_contract["allocation"]["batch_size_per_rank"]),
        "ROUND_DAGGER_LEARNER_MAX_STEPS": str(optimization["optimizer_steps"]),
        "ROUND_DAGGER_LEARNER_SAVE_EVERY": str(optimization["save_every_steps"]),
        "ROUND_DAGGER_LEARNER_LEARNING_RATE": "2.5e-5",
        "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": str(optimization["weight_decay"]),
        "ROUND_DAGGER_LEARNER_SEED": str(optimization["seed"]),
        "ROUND_DAGGER_LEARNER_PARTITION_SEED": str(optimization["partition_seed"]),
    }
    heldout = learner_contract.get("heldout_evaluation")
    if heldout is not None:
        environment.update(
            {
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": heldout[
                    "snapshot_path"
                ],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256": heldout[
                    "snapshot_file_sha256"
                ],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256": heldout[
                    "snapshot_sha256"
                ],
                "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL": str(
                    heldout["interval_steps"]
                ),
            }
        )
    return environment


def validate_launch(
    *,
    orchestration_config: Path,
    campaign_contract_path: Path,
    campaign_contract_sha256: str,
    learner_contract_path: Path,
    sealed_source_root: str,
    sealed_source_inventory_sha256: str,
    evaluation_sealed_source_root: str | Path,
    evaluation_sealed_source_inventory_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    campaign_contract_sha = _require_sha(
        campaign_contract_sha256, "caller campaign contract SHA-256"
    )
    invariant_profile_path = REPOSITORY_ROOT / PRODUCTION_CAMPAIGN_CONTRACT_RELATIVE
    campaign_contract, runtime_contract, _lineage = materialize_campaign_contract(
        invariant_profile_path,
        runtime_contract_path=campaign_contract_path,
        runtime_contract_sha256=campaign_contract_sha,
    )
    learner_deployment_contract_sha = artifacts.sha256_file(learner_contract_path)
    orchestration_sha = artifacts.sha256_file(orchestration_config)
    learner_contract = load_learner_contract(learner_contract_path)
    _verify_static_inputs(campaign_contract)
    source_report = _verify_remote_source(
        sealed_source_root,
        sealed_source_inventory_sha256,
        learner_contract=learner_contract,
        learner_contract_path=learner_contract_path,
    )
    local_seal_report = _verify_local_evaluation_seal(
        evaluation_sealed_source_root,
        evaluation_sealed_source_inventory_sha256,
        campaign_contract=campaign_contract,
    )
    capture = runtime_contract["live_capture"]
    expected_capture_sources = {
        "split_manifest_path": str(
            Path(local_seal_report["root"])
            / campaign_contract["motion_split"]["manifest_relative_path"]
        ),
        "heldout_plan_path": str(
            Path(local_seal_report["root"])
            / campaign_contract["motion_split"]["heldout_plan_relative_path"]
        ),
    }
    if any(capture[key] != value for key, value in expected_capture_sources.items()):
        raise ProductionLaunchError(
            "live capture split/plan are not derived from the sealed evaluation source"
        )
    _verify_camera_acceptance_source_bindings(
        campaign_contract,
        learner_source_report=source_report,
        evaluation_source_report=local_seal_report,
    )
    capture_provenance = runtime_contract["capture_provenance"]
    provenance_expected = {
        "fast_roborender_revision": source_report.get(
            "fast_roborender_revision"
        ),
        "gr00t_revision": source_report.get("gr00t_revision"),
        "actor_source_inventory_sha256": (
            evaluation_sealed_source_inventory_sha256
        ),
        "evaluation_source_inventory_sha256": (
            evaluation_sealed_source_inventory_sha256
        ),
        "gr00t_source_tree_sha256": source_report.get(
            "gr00t_source_tree_sha256"
        ),
        "camera_acceptance_sha256": campaign_contract["artifacts"][
            "camera_acceptance"
        ]["sha256"],
        "renderer_runtime_sha256": campaign_contract["artifacts"]["renderer"][
            "runtime_identity_sha256"
        ],
        "renderer_cohort_sha256": campaign_contract["artifacts"]["renderer"][
            "cohort_identity_sha256"
        ],
        "anchor_inventory_sha256": campaign_contract["artifacts"]["anchor_pool"][
            "inventory_sha256"
        ],
        "evaluation_config_sha256": campaign_contract["evaluation"][
            "hydra_config_sha256"
        ],
    }
    provenance_mismatches = [
        key
        for key, value in provenance_expected.items()
        if capture_provenance.get(key) != value
    ]
    if provenance_mismatches:
        raise ProductionLaunchError(
            "live capture provenance differs from deployment: "
            f"{provenance_mismatches}"
        )
    (
        configuration,
        campaign,
        coordinator,
        derived_learner_bundle_sha,
    ) = _validate_orchestration_and_campaign(
        orchestration_config,
        campaign_contract,
        learner_contract,
        learner_deployment_contract_sha,
        sealed_source_inventory_sha256,
        Path(local_seal_report["root"]),
    )
    environment = build_environment(
        campaign_contract,
        learner_contract,
        sealed_source_root=sealed_source_root,
        sealed_source_inventory_sha256=sealed_source_inventory_sha256,
        evaluation_source_root=local_seal_report["root"],
        evaluation_source_inventory_sha256=evaluation_sealed_source_inventory_sha256,
        sealed_teacher_checkpoint=local_seal_report[
            "sealed_teacher_checkpoint"
        ],
        learner_contract_sha256=learner_deployment_contract_sha,
    )
    environment["ROUND_DAGGER_LAUNCH_ORCHESTRATION_SHA256"] = orchestration_sha
    environment["ROUND_DAGGER_CAMPAIGN_CONTRACT"] = str(campaign_contract_path)
    environment["ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256"] = campaign_contract_sha
    environment["ROUND_DAGGER_DATA_LINEAGE_SHA256"] = campaign_contract[
        "data_lineage_sha256"
    ]
    if artifacts.sha256_file(campaign_contract_path) != campaign_contract_sha:
        raise ProductionLaunchError("campaign contract changed during preflight")
    if artifacts.sha256_file(learner_contract_path) != learner_deployment_contract_sha:
        raise ProductionLaunchError("learner contract changed during preflight")
    if artifacts.sha256_file(orchestration_config) != orchestration_sha:
        raise ProductionLaunchError("orchestration config changed during preflight")
    # Actor collection still uses the live RTX checkout, whereas learner code
    # uses the sealed OCI snapshot and evaluation uses its independent local
    # seal.  Preserve the actor/learner compatibility check immediately before
    # handing control to the orchestrator.
    local_source = _verify_local_source_checkout(source_report)
    environment["ROUND_DAGGER_FAST_ROBORENDER_REVISION"] = local_source[
        "fast_roborender"
    ]["revision"]
    environment["ROUND_DAGGER_GR00T_REVISION"] = local_source["gr00t"]["revision"]
    environment["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"] = _require_sha(
        source_report.get("gr00t_source_tree_sha256"),
        "sealed source GR00T tree",
    )
    for name, value in environment.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or "\x00" in value or "\n" in value:
            raise ProductionLaunchError("generated environment is unsafe")
        if re.search(r"(?i)(api[_-]?key|password|secret|token)", name):
            raise ProductionLaunchError("generated environment unexpectedly contains a secret field")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "validated_no_submission",
        "campaign_id": campaign["campaign_id"],
        "campaign_root": str(configuration.campaign_root),
        "macro_round": configuration.macro_round,
        "coordinator_phase": coordinator["phase"],
        "topology": "4 cohorts x 2 actors x 32 environments x 512 steps",
        "transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
        "macro_round_count": PRODUCTION_MACRO_ROUND_COUNT,
        "transitions_per_campaign": TRANSITIONS_PER_CAMPAIGN,
        "completed_macro_rounds": coordinator["next_macro_round"],
        "remaining_macro_rounds": (
            PRODUCTION_MACRO_ROUND_COUNT - coordinator["next_macro_round"]
        ),
        "reference_lag_frames": 256,
        "bootstrap_anchor_source": campaign_contract["observation_contract"][
            "bootstrap_anchor_source"
        ],
        "active_policy_sha256": coordinator["active_policy"]["checkpoint_sha256"],
        "learner_sealed_source_root": sealed_source_root,
        "learner_source_inventory_sha256": sealed_source_inventory_sha256,
        "evaluation_sealed_source_root": local_seal_report["root"],
        "evaluation_source_inventory_sha256": evaluation_sealed_source_inventory_sha256,
        "evaluation_sealed_teacher_sha256": local_seal_report[
            "sealed_teacher_sha256"
        ],
        "learner_source_revision": source_report.get("fast_roborender_revision"),
        "evaluation_source_revision": local_seal_report.get(
            "fast_roborender_revision"
        ),
        "local_source": local_source,
        "learner_deployment_contract_sha256": learner_deployment_contract_sha,
        "derived_learner_bundle_sha256": derived_learner_bundle_sha,
        "campaign_contract_sha256": campaign_contract_sha,
        "invariant_profile_sha256": campaign_contract[
            "invariant_profile_sha256"
        ],
        "data_lineage_sha256": campaign_contract["data_lineage_sha256"],
        "orchestration_config_sha256": orchestration_sha,
        "learner": "1 OCI-NRT node / 8 H100 / batch 8 per rank / 1200 optimizer steps",
        "wandb": "ldu@nvidia.com -> nvidia/roborender-round-dagger",
        "hooks": {
            name: None if hook is None else {"argv": list(hook.argv), "timeout_seconds": hook.timeout_seconds}
            for name, hook in configuration.hooks.items()
        },
        "exported_environment_names": sorted(environment),
        "credentials_in_plan": False,
    }
    # Re-hash every exact condition bundle as the final preflight operation.
    # The evaluation hook mounts these same files read-only and repeats the
    # check before contract publication, container launch, and after execution.
    _verify_first_condition_bundles(campaign_contract)
    return environment, plan


def _write_nul_environment(path: Path, environment: Mapping[str, str]) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ProductionLaunchError("environment output must be an absolute non-symlink path")
    payload = b"".join(
        name.encode("ascii") + b"\0" + value.encode("utf-8") + b"\0"
        for name, value in sorted(environment.items())
    )
    try:
        path.write_bytes(payload)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise ProductionLaunchError("cannot write validated environment handoff") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-config", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--campaign-contract-sha256", required=True)
    parser.add_argument("--learner-contract", type=Path, required=True)
    parser.add_argument("--sealed-source-root", required=True)
    parser.add_argument("--sealed-source-inventory-sha256", required=True)
    parser.add_argument("--evaluation-sealed-source-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation-sealed-source-inventory-sha256", required=True
    )
    parser.add_argument("--env-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        environment, plan = validate_launch(
            orchestration_config=arguments.orchestration_config,
            campaign_contract_path=arguments.campaign_contract,
            campaign_contract_sha256=arguments.campaign_contract_sha256,
            learner_contract_path=arguments.learner_contract,
            sealed_source_root=arguments.sealed_source_root,
            sealed_source_inventory_sha256=arguments.sealed_source_inventory_sha256,
            evaluation_sealed_source_root=arguments.evaluation_sealed_source_root,
            evaluation_sealed_source_inventory_sha256=(
                arguments.evaluation_sealed_source_inventory_sha256
            ),
        )
        _write_nul_environment(arguments.env_output, environment)
    except ProductionLaunchError as exc:
        print(f"production macro-round preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
