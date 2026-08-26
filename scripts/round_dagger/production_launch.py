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

from round_dagger import artifacts, policy_bundle, state  # noqa: E402
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
    STEPS_PER_ENVIRONMENT,
    TARGET_ENVIRONMENTS,
    TRANSITIONS_PER_COHORT,
    TRANSITIONS_PER_MACRO_ROUND,
)
from scripts.round_dagger.verify_anchor_pool import verify_anchor_pool  # noqa: E402
from scripts.round_dagger.verify_sealed_source_snapshot import (  # noqa: E402
    VerificationError as SealedSourceVerificationError,
    verify as verify_sealed_source_snapshot,
)


CAMPAIGN_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.production_campaign_contract.v1"
)
LEARNER_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.production_learner_contract.v1"
)
LEARNER_BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
SEALED_REPORT_SCHEMA = "fast_roborender.sealed_source_verification.v1"
PLAN_SCHEMA = "fast_roborender.round_dagger.production_launch_plan.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
SAFE_REMOTE_PATH_RE = re.compile(r"/[A-Za-z0-9_./-]+")
VALIDATION_ROOT = Path("/mnt/data2/ldu/roborender_round_dagger_validation")
BOOTSTRAP_ROOT = VALIDATION_ROOT / "fresh_bootstrap_v5_20260825T0456Z"
FLEET_ROOT = VALIDATION_ROOT / "fleet_two_gpu_provenance_v2"
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
TEACHER_CHECKPOINT = REPOSITORY_ROOT / TEACHER_CHECKPOINT_RELATIVE
PRODUCTION_CAMPAIGN_CONTRACT_RELATIVE = Path(
    "configs/round_dagger/production_campaign_256x300.json"
)
ANCHOR_POOL = REPOSITORY_ROOT / "runtime/hsg_stage_transfer/anchor_pool_2048"
EVALUATION_CONFIG = (
    VALIDATION_ROOT
    / "actor_train_smoke_4cohort_1env_2step_v1/cohort_00/run/config.yaml"
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


def _file_identity(path: Path, expected_sha: str, label: str) -> None:
    _regular(path, label)
    if artifacts.sha256_file(path) != _require_sha(expected_sha, f"{label} SHA-256"):
        raise ProductionLaunchError(f"{label} bytes differ from the production contract")


def _load_contract(path: Path, schema: str, keys: set[str], label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        document = artifacts.load_strict_json(path)
    except Exception as exc:
        raise ProductionLaunchError(f"cannot load strict {label}") from exc
    result = _exact(document, keys, label)
    _require_equal(result["schema"], schema, f"{label} schema")
    return result


def load_campaign_contract(path: Path) -> dict[str, Any]:
    contract = _load_contract(
        path, CAMPAIGN_CONTRACT_SCHEMA, CAMPAIGN_CONTRACT_KEYS, "campaign contract"
    )
    _require_equal(contract["campaign_id"], "roborender-dagger-256x300-v1", "campaign ID")
    _require_equal(
        _exact(
            contract["topology"],
            {
                "target_environments",
                "active_environments",
                "cohort_count",
                "environments_per_actor_rank",
                "actor_ranks_per_cohort",
                "steps_per_environment",
                "transitions_per_cohort",
                "transitions_per_macro_round",
            },
            "campaign topology",
        ),
        {
            "target_environments": TARGET_ENVIRONMENTS,
            "active_environments": ENVIRONMENTS_PER_COHORT,
            "cohort_count": COHORT_COUNT,
            "environments_per_actor_rank": 32,
            "actor_ranks_per_cohort": 2,
            "steps_per_environment": STEPS_PER_ENVIRONMENT,
            "transitions_per_cohort": TRANSITIONS_PER_COHORT,
            "transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
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
                "reference_lag_frames",
            },
            "observation contract",
        ),
        {
            "primary_view": "head_color",
            "policy_views": ["head_color", "hand_left", "hand_right"],
            "policy_image_shape": [324, 192, 3],
            "policy_view_layout": "vertical",
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
        {"kind": "linear", "start": 1.0, "end": 0.0, "decay_rounds": 10, "rate": 1.0},
        "beta schedule",
    )

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
        "3b3a9ac168eaf163a566772679a5bb4b8ec73c9947c9b51668ac08f9a014abae",
        "bootstrap checkpoint identity",
    )
    _require_equal(
        bootstrap["runtime_policy_sha256"],
        "fd1fd8f290fd603e05251f3798f40fd29b8028041e701e6240ac1ae4541f67b6",
        "bootstrap runtime identity",
    )
    _require_equal(
        bootstrap["policy_spec_sha256"],
        "216cf3a2802ede98bf208ff7fb3925163973e2e32d823c1d6b1efa4dfeec24cb",
        "bootstrap policy spec identity",
    )
    _require_equal(
        bootstrap["manifest_sha256"],
        "ca103a67843c004dd2293a3b96927807e951ac477847023d646ffb091ff20379",
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
    _require_equal(teacher["checkpoint_path"], str(TEACHER_CHECKPOINT), "teacher checkpoint path")
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
        {"path", "anchor_count", "inventory_sha256", "inventory_algorithm"},
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
            "gr00t_root": str(REPOSITORY_ROOT / "GR00T-VisualSim2Real"),
            "gr00t_python": "/opt/round-dagger-rtx/bin/python",
            "pair_executor": str(REPOSITORY_ROOT / "scripts/round_dagger/run_actor_rank_docker.sh"),
            "pair_rank_script": str(REPOSITORY_ROOT / "scripts/round_dagger/run_actor_cohort_with_wandb.sh"),
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
            "netrc": str(REPOSITORY_ROOT / "runtime/round_dagger_credentials/.netrc"),
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
        "e29857c6e4b51bc2241c6e1101c0bff12204ca6dc8d81070169d6b8927fe7933",
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
        "1721049b8fcd8c4e0eefdbd581fcca6d77bc6cf8d5f9ffc789ae4a817d7a36ee",
        "evaluation threshold contract identity",
    )
    for key in (
        "hydra_config_sha256",
        "scenario_contract_sha256",
        "threshold_contract_sha256",
    ):
        _require_sha(evaluation[key], f"evaluation {key}")
    return contract


def load_learner_contract(path: Path) -> dict[str, Any]:
    contract = _load_contract(
        path, LEARNER_CONTRACT_SCHEMA, LEARNER_CONTRACT_KEYS, "learner contract"
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
    _require_equal(
        _exact(
            contract["optimization"],
            {"optimizer_steps", "save_every_steps", "learning_rate", "weight_decay", "seed", "partition_seed"},
            "learner optimization",
        ),
        {
            "optimizer_steps": 500,
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
    evaluation = campaign_contract["evaluation"]
    _file_identity(Path(evaluation["hydra_config_path"]), evaluation["hydra_config_sha256"], "evaluation Hydra config")
    for relative_key, sha_key, label in (
        ("scenario_contract", "scenario_contract_sha256", "evaluation scenarios"),
        ("threshold_contract", "threshold_contract_sha256", "evaluation thresholds"),
    ):
        relative = evaluation[relative_key]
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProductionLaunchError(f"{label} path must be repository-relative")
        _file_identity(REPOSITORY_ROOT / relative, evaluation[sha_key], label)

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
    for relative_key, sha_key, label in (
        ("scenario_contract", "scenario_contract_sha256", "evaluation scenarios"),
        ("threshold_contract", "threshold_contract_sha256", "evaluation thresholds"),
    ):
        _file_identity(
            local_root / evaluation[relative_key],
            evaluation[sha_key],
            f"sealed {label}",
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
    ):
        raise ProductionLaunchError(
            "sealed evaluation anchor contract differs from launch contract"
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
        policy_spec["checkpoint_sha256"] = actual_parent
        policy_spec["runtime_policy_sha256"] = runtime_sha
        policy_spec = validate_policy_spec(policy_spec)
    except Exception as exc:
        raise ProductionLaunchError(
            f"cannot derive the active per-round learner bundle: {exc}"
        ) from exc
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
    root = _directory(configuration.campaign_root, "campaign root")
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
        "steps_per_environment": 300,
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
    }
    mismatches = [key for key, value in expected_campaign.items() if campaign.get(key) != value]
    if mismatches:
        raise ProductionLaunchError(f"campaign manifest differs from production contract: {mismatches}")
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
    runtime = campaign_contract["actor_runtime"]
    telemetry = campaign_contract["telemetry"]
    wandb = campaign_contract["wandb"]
    evaluation = campaign_contract["evaluation"]
    cluster = learner_contract["cluster"]
    container = learner_contract["container"]
    handoff = learner_contract["handoff"]
    optimization = learner_contract["optimization"]
    return {
        "ROUND_DAGGER_TEACHER_CHECKPOINT": artifact["teacher"]["checkpoint_path"],
        "ROUND_DAGGER_TEACHER_SHA256": artifact["teacher"]["checkpoint_sha256"],
        "ROUND_DAGGER_TEACHER_CHECKPOINT_RELATIVE": TEACHER_CHECKPOINT_RELATIVE.as_posix(),
        "ROUND_DAGGER_EVALUATION_SEALED_TEACHER": sealed_teacher_checkpoint,
        "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": evaluation_source_root,
        "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": evaluation_source_inventory_sha256,
        "ROUND_DAGGER_LEARNER_SOURCE_INVENTORY_SHA256": sealed_source_inventory_sha256,
        "ROUND_DAGGER_RENDERER_SHA256": artifact["renderer"]["cohort_identity_sha256"],
        "ROUND_DAGGER_RENDERER_RUNTIME_SHA256": artifact["renderer"]["runtime_identity_sha256"],
        "IMAGE3F_ENDPOINT_MANIFEST": artifact["endpoint_manifest"]["path"],
        "ANCHORS": artifact["anchor_pool"]["path"],
        "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": artifact["anchor_pool"]["inventory_sha256"],
        "ROUND_DAGGER_ANCHOR_INDEX": "0",
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
        "ROUND_DAGGER_EVALUATION_SCENARIOS": str(Path(evaluation_source_root) / evaluation["scenario_contract"]),
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


def validate_launch(
    *,
    orchestration_config: Path,
    campaign_contract_path: Path,
    learner_contract_path: Path,
    sealed_source_root: str,
    sealed_source_inventory_sha256: str,
    evaluation_sealed_source_root: str | Path,
    evaluation_sealed_source_inventory_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    campaign_contract_sha = artifacts.sha256_file(campaign_contract_path)
    learner_deployment_contract_sha = artifacts.sha256_file(learner_contract_path)
    orchestration_sha = artifacts.sha256_file(orchestration_config)
    campaign_contract = load_campaign_contract(campaign_contract_path)
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
        "topology": "4 cohorts x 2 actors x 32 environments x 300 steps",
        "transitions_per_macro_round": TRANSITIONS_PER_MACRO_ROUND,
        "reference_lag_frames": 256,
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
        "orchestration_config_sha256": orchestration_sha,
        "learner": "1 OCI-NRT node / 8 H100 / batch 8 per rank / 500 optimizer steps",
        "wandb": "ldu@nvidia.com -> nvidia/roborender-round-dagger",
        "hooks": {
            name: None if hook is None else {"argv": list(hook.argv), "timeout_seconds": hook.timeout_seconds}
            for name, hook in configuration.hooks.items()
        },
        "exported_environment_names": sorted(environment),
        "credentials_in_plan": False,
    }
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
