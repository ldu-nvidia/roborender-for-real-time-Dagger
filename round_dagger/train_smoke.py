"""Prepare and verify the four-cohort real train-replay smoke gate.

This gate is deliberately smaller than a production collection: four
independent one-rank cohorts, one logical environment and two transitions per
cohort.  Together they produce eight immutable train transitions, exactly one
batch for the bounded learner validation.  The plan binds every cohort to one
of the two local renderer descriptors and to the matching physical GPU slot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

from .actor_pair import ActorPairError, inspect_actor_rank
from .artifacts import (
    ArtifactError,
    create_cohort,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
    verify_cohort,
)


PLAN_SCHEMA = "fast_roborender.round_dagger.actor_train_smoke_plan.v1"
CONFIG_SCHEMA = "fast_roborender.round_dagger.actor_train_smoke_config.v1"
BETA_SCHEMA = "fast_roborender.round_dagger.beta.v1"
CAMPAIGN_ID = "rtx-train-smoke-v1"
MACRO_ROUND = 0
COHORT_COUNT = 4
RANKS_PER_COHORT = 1
ENVIRONMENTS_PER_RANK = 1
STEPS_PER_ENVIRONMENT = 2
REFERENCE_LAG = 256
TRAIN_ACKNOWLEDGEMENT = "bounded-train-replay"
VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [26]}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_KEYS = {
    "accelerate",
    "acknowledgement",
    "actor_image",
    "anchors",
    "anchor_count",
    "beta_path",
    "beta_sha256",
    "campaign_id",
    "cohorts",
    "config_path",
    "config_sha256",
    "fleet_plan",
    "fleet_plan_sha256",
    "fleet_readiness",
    "fleet_readiness_sha256",
    "gr00t_python",
    "gr00t_root",
    "macro_round",
    "policy_checkpoint",
    "policy_manifest",
    "policy_sha256",
    "policy_spec",
    "purpose",
    "renderer_sha256",
    "repository_root",
    "runtime_policy_sha256",
    "schema",
    "source_endpoint_manifest",
    "source_endpoint_manifest_sha256",
    "teacher_checkpoint",
    "teacher_sha256",
}
_COHORT_PLAN_KEYS = {
    "cohort_dir",
    "cohort_index",
    "contract_sha256",
    "endpoint_descriptor",
    "endpoint_descriptor_sha256",
    "endpoint_manifest",
    "endpoint_manifest_sha256",
    "global_env_id",
    "gpu_id",
    "renderer_slot",
    "run_dir",
}


class TrainSmokeError(RuntimeError):
    """The bounded train-smoke plan is unsafe, inconsistent, or incomplete."""


def _absolute_regular(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or value.is_symlink() or not value.is_file():
        raise TrainSmokeError(f"{label} must be an absolute regular non-symlink file")
    # sha256_file also rejects a symbolic-link ancestor and a non-regular inode.
    try:
        sha256_file(value)
    except ArtifactError as exc:
        raise TrainSmokeError(f"unsafe {label}: {exc}") from exc
    return value


def _absolute_directory(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or value.is_symlink() or not value.is_dir():
        raise TrainSmokeError(
            f"{label} must be an absolute existing non-symlink directory"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TrainSmokeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_endpoint_descriptors(manifest_path: Path) -> list[Path]:
    raw = manifest_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if len(lines) != 2 or any(not line or line.strip() != line for line in lines):
        raise TrainSmokeError(
            "source endpoint manifest must contain exactly two canonical lines"
        )
    if raw != "\n".join(lines) + "\n" or len(set(lines)) != 2:
        raise TrainSmokeError("source endpoint manifest is non-canonical or repeats an endpoint")
    descriptors: list[Path] = []
    for slot, line in enumerate(lines):
        descriptor = _absolute_regular(line, f"renderer {slot} endpoint descriptor")
        if descriptor.name != f"renderer_{slot:03d}.txt":
            raise TrainSmokeError("renderer descriptor order is not slot-addressable")
        fields = descriptor.read_text(encoding="utf-8").split()
        if len(fields) != 2 or fields[0] != "127.0.0.1":
            raise TrainSmokeError("renderer descriptor must use a local 127.0.0.1 endpoint")
        try:
            port = int(fields[1])
        except ValueError as exc:
            raise TrainSmokeError("renderer endpoint port is not an integer") from exc
        if not 1 <= port <= 65535:
            raise TrainSmokeError("renderer endpoint port is outside 1..65535")
        descriptors.append(descriptor)
    return descriptors


def _publish_immutable_text(path: Path, text: str) -> str:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TrainSmokeError(f"symbolic links are forbidden: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise TrainSmokeError(f"immutable artifact conflict at {path}")
    else:
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return sha256_file(path)


def _load_bootstrap_identity(checkpoint: Path) -> tuple[Path, Path, str]:
    spec = _absolute_regular(
        Path(f"{checkpoint}.policy_spec.json"), "bootstrap policy spec"
    )
    manifest_path = _absolute_regular(
        Path(f"{checkpoint}.manifest.json"), "bootstrap policy manifest"
    )
    manifest = load_strict_json(manifest_path)
    expected_keys = {
        "schema",
        "checkpoint_path",
        "checkpoint_sha256",
        "runtime_policy_sha256",
        "policy_spec_path",
        "policy_spec_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_keys:
        raise TrainSmokeError("bootstrap policy manifest key-set mismatch")
    if manifest.get("schema") != "fast_roborender.round_dagger.bootstrap_policy.v1":
        raise TrainSmokeError("bootstrap policy manifest schema mismatch")
    if manifest.get("checkpoint_path") != checkpoint.name:
        raise TrainSmokeError("bootstrap manifest/checkpoint path mismatch")
    if manifest.get("policy_spec_path") != spec.name:
        raise TrainSmokeError("bootstrap manifest/spec path mismatch")
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise TrainSmokeError("bootstrap manifest/checkpoint digest mismatch")
    if manifest.get("policy_spec_sha256") != sha256_file(spec):
        raise TrainSmokeError("bootstrap manifest/spec digest mismatch")
    return spec, manifest_path, _sha(
        manifest.get("runtime_policy_sha256"), "runtime policy identity"
    )


def _validate_fleet_evidence(
    *,
    endpoint_manifest: Path,
    descriptors: list[Path],
    fleet_plan: Path,
    fleet_readiness: Path,
    renderer_sha256: str,
) -> None:
    fleet = load_strict_json(fleet_plan)
    if fleet.get("schema") != "fast_roborender.local_image3f_fleet.v1.plan":
        raise TrainSmokeError("renderer fleet plan schema mismatch")
    if fleet.get("gpus") != ["0", "1"]:
        raise TrainSmokeError("renderer fleet is not pinned to physical GPUs 0 and 1")
    if Path(fleet.get("fleet_dir", "")) != endpoint_manifest.parent:
        raise TrainSmokeError("endpoint manifest does not belong to the fleet plan")
    readiness = load_strict_json(fleet_readiness)
    if readiness.get("schema") != "fast_roborender.local_image3f_fleet.v1.readiness":
        raise TrainSmokeError("renderer fleet readiness schema mismatch")
    if readiness.get("capable") != 2 or readiness.get("discovered") != 2:
        raise TrainSmokeError("renderer fleet readiness does not prove two capable endpoints")
    endpoints = readiness.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise TrainSmokeError("renderer readiness endpoint cardinality mismatch")
    for slot, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, Mapping):
            raise TrainSmokeError("renderer readiness endpoint is not a mapping")
        if endpoint.get("file") != str(descriptors[slot]):
            raise TrainSmokeError("renderer readiness endpoint order differs from fleet order")
        response = endpoint.get("response")
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise TrainSmokeError("renderer readiness does not contain a successful response")
        if response.get("renderer_identity_sha256") != renderer_sha256:
            raise TrainSmokeError("renderer readiness identity differs from requested identity")
        if response.get("protocol_version") != 2:
            raise TrainSmokeError("renderer readiness does not prove protocol version 2")


def prepare_train_smoke(
    root: str | Path,
    *,
    endpoint_manifest: str | Path,
    policy_checkpoint: str | Path,
    teacher_checkpoint: str | Path,
    renderer_sha256: str,
    anchors: str | Path,
    repository_root: str | Path,
    gr00t_root: str | Path,
    actor_image: str,
    gr00t_python: str = "/opt/round-dagger-rtx/bin/python",
    accelerate: str = "/opt/round-dagger-rtx/bin/accelerate",
) -> dict[str, Any]:
    """Publish an idempotent, content-bound four-cohort train-smoke plan."""

    target = Path(root).expanduser()
    if not target.is_absolute() or target.is_symlink():
        raise TrainSmokeError("train-smoke root must be an absolute non-symlink path")
    target.mkdir(parents=True, exist_ok=True)
    endpoint_manifest_path = _absolute_regular(
        endpoint_manifest, "source endpoint manifest"
    )
    descriptors = _canonical_endpoint_descriptors(endpoint_manifest_path)
    checkpoint = _absolute_regular(policy_checkpoint, "policy checkpoint")
    teacher = _absolute_regular(teacher_checkpoint, "teacher checkpoint")
    anchors_path = _absolute_directory(anchors, "anchor pool")
    repository = _absolute_directory(repository_root, "repository root")
    gr00t = _absolute_directory(gr00t_root, "GR00T root")
    renderer_sha = _sha(renderer_sha256, "renderer identity")
    if not actor_image or any(character.isspace() for character in actor_image):
        raise TrainSmokeError("actor image must be a non-empty token")
    for value, label in ((gr00t_python, "GR00T Python"), (accelerate, "accelerate")):
        if not Path(value).is_absolute() or any(character in value for character in "\r\n\0"):
            raise TrainSmokeError(f"{label} must be an absolute container path")

    fleet_plan = _absolute_regular(endpoint_manifest_path.parent / "plan.json", "fleet plan")
    fleet_readiness = _absolute_regular(
        endpoint_manifest_path.parent / "readiness.json", "fleet readiness"
    )
    _validate_fleet_evidence(
        endpoint_manifest=endpoint_manifest_path,
        descriptors=descriptors,
        fleet_plan=fleet_plan,
        fleet_readiness=fleet_readiness,
        renderer_sha256=renderer_sha,
    )
    spec, policy_manifest, runtime_policy_sha = _load_bootstrap_identity(checkpoint)
    anchor_count = sum(1 for path in anchors_path.glob("*.npz") if path.is_file())
    if anchor_count < 2048:
        raise TrainSmokeError(f"anchor pool has {anchor_count} NPZ files; at least 2048 required")

    beta_document = {
        "schedule": "constant",
        "schema": BETA_SCHEMA,
        "value": 0.5,
    }
    beta_path = target / "beta.json"
    beta_sha = publish_immutable_json(beta_path, beta_document)
    config_document = {
        "action_contract": {
            "environment": 31,
            "homie": 15,
            "policy": 11,
            "replay": 26,
        },
        "beta_sha256": beta_sha,
        "cohort_count": COHORT_COUNT,
        "environments_per_rank": ENVIRONMENTS_PER_RANK,
        "purpose": "train",
        "ranks_per_cohort": RANKS_PER_COHORT,
        "reference_lag": REFERENCE_LAG,
        "rollout_action_source": "beta",
        "schema": CONFIG_SCHEMA,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "views": VIEW_CONTRACT["views"],
    }
    config_path = target / "config.json"
    config_sha = publish_immutable_json(config_path, config_document)

    cohorts: list[dict[str, Any]] = []
    for cohort_index in range(COHORT_COUNT):
        slot = cohort_index % 2
        cohort_root = target / f"cohort_{cohort_index:02d}"
        cohort_dir = cohort_root / "cohort"
        selected_manifest = cohort_root / "renderer_endpoint_manifest.txt"
        endpoint_manifest_sha = _publish_immutable_text(
            selected_manifest, f"{descriptors[slot]}\n"
        )
        create_cohort(
            cohort_dir,
            campaign_id=CAMPAIGN_ID,
            macro_round=MACRO_ROUND,
            cohort_index=cohort_index,
            purpose="train",
            expected_ranks=[0],
            expected_global_env_ids=[cohort_index],
            expected_transitions=STEPS_PER_ENVIRONMENT,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            policy_sha256=sha256_file(checkpoint),
            runtime_policy_sha256=runtime_policy_sha,
            teacher_sha256=sha256_file(teacher),
            renderer_sha256=renderer_sha,
            beta_sha256=beta_sha,
            config_sha256=config_sha,
        )
        cohorts.append(
            {
                "cohort_dir": str(cohort_dir),
                "cohort_index": cohort_index,
                "contract_sha256": sha256_file(cohort_dir / "contract.json"),
                "endpoint_descriptor": str(descriptors[slot]),
                "endpoint_descriptor_sha256": sha256_file(descriptors[slot]),
                "endpoint_manifest": str(selected_manifest),
                "endpoint_manifest_sha256": endpoint_manifest_sha,
                "global_env_id": cohort_index,
                "gpu_id": slot,
                "renderer_slot": slot,
                "run_dir": str(cohort_root / "run"),
            }
        )

    plan = {
        "accelerate": accelerate,
        "acknowledgement": TRAIN_ACKNOWLEDGEMENT,
        "actor_image": actor_image,
        "anchors": str(anchors_path),
        "anchor_count": anchor_count,
        "beta_path": str(beta_path),
        "beta_sha256": beta_sha,
        "campaign_id": CAMPAIGN_ID,
        "cohorts": cohorts,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "fleet_plan": str(fleet_plan),
        "fleet_plan_sha256": sha256_file(fleet_plan),
        "fleet_readiness": str(fleet_readiness),
        "fleet_readiness_sha256": sha256_file(fleet_readiness),
        "gr00t_python": gr00t_python,
        "gr00t_root": str(gr00t),
        "macro_round": MACRO_ROUND,
        "policy_checkpoint": str(checkpoint),
        "policy_manifest": str(policy_manifest),
        "policy_sha256": sha256_file(checkpoint),
        "policy_spec": str(spec),
        "purpose": "train",
        "renderer_sha256": renderer_sha,
        "repository_root": str(repository),
        "runtime_policy_sha256": runtime_policy_sha,
        "schema": PLAN_SCHEMA,
        "source_endpoint_manifest": str(endpoint_manifest_path),
        "source_endpoint_manifest_sha256": sha256_file(endpoint_manifest_path),
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256_file(teacher),
    }
    plan_path = target / "plan.json"
    publish_immutable_json(plan_path, plan)
    return validate_train_smoke(plan_path)


def _plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = _absolute_regular(path, "train-smoke plan")
    document = load_strict_json(plan_path)
    if not isinstance(document, Mapping) or set(document) != _PLAN_KEYS:
        raise TrainSmokeError("train-smoke plan key-set mismatch")
    if document.get("schema") != PLAN_SCHEMA:
        raise TrainSmokeError("train-smoke plan schema mismatch")
    if document.get("campaign_id") != CAMPAIGN_ID or document.get("macro_round") != 0:
        raise TrainSmokeError("train-smoke campaign identity mismatch")
    if document.get("purpose") != "train":
        raise TrainSmokeError("train-smoke purpose must be train")
    if document.get("acknowledgement") != TRAIN_ACKNOWLEDGEMENT:
        raise TrainSmokeError("train-smoke acknowledgement mismatch")
    return plan_path, dict(document)


def validate_train_smoke(path: str | Path) -> dict[str, Any]:
    """Verify every immutable input, mapping, and cohort contract in a plan."""

    plan_path, plan = _plan(path)
    checkpoint = _absolute_regular(plan["policy_checkpoint"], "policy checkpoint")
    teacher = _absolute_regular(plan["teacher_checkpoint"], "teacher checkpoint")
    if sha256_file(checkpoint) != plan.get("policy_sha256"):
        raise TrainSmokeError("policy checkpoint changed after plan publication")
    if sha256_file(teacher) != plan.get("teacher_sha256"):
        raise TrainSmokeError("teacher checkpoint changed after plan publication")
    spec, policy_manifest, runtime_sha = _load_bootstrap_identity(checkpoint)
    if str(spec) != plan.get("policy_spec") or str(policy_manifest) != plan.get("policy_manifest"):
        raise TrainSmokeError("bootstrap policy bundle paths changed")
    if runtime_sha != plan.get("runtime_policy_sha256"):
        raise TrainSmokeError("runtime policy identity changed")
    renderer_sha = _sha(plan.get("renderer_sha256"), "renderer identity")
    source_manifest = _absolute_regular(
        plan["source_endpoint_manifest"], "source endpoint manifest"
    )
    if sha256_file(source_manifest) != plan.get("source_endpoint_manifest_sha256"):
        raise TrainSmokeError("source endpoint manifest changed after plan publication")
    descriptors = _canonical_endpoint_descriptors(source_manifest)
    fleet_plan = _absolute_regular(plan["fleet_plan"], "fleet plan")
    fleet_readiness = _absolute_regular(plan["fleet_readiness"], "fleet readiness")
    if sha256_file(fleet_plan) != plan.get("fleet_plan_sha256"):
        raise TrainSmokeError("fleet plan changed after train-smoke publication")
    if sha256_file(fleet_readiness) != plan.get("fleet_readiness_sha256"):
        raise TrainSmokeError("fleet readiness changed after train-smoke publication")
    _validate_fleet_evidence(
        endpoint_manifest=source_manifest,
        descriptors=descriptors,
        fleet_plan=fleet_plan,
        fleet_readiness=fleet_readiness,
        renderer_sha256=renderer_sha,
    )

    beta_path = _absolute_regular(plan["beta_path"], "beta document")
    config_path = _absolute_regular(plan["config_path"], "train-smoke config")
    if sha256_file(beta_path) != plan.get("beta_sha256"):
        raise TrainSmokeError("beta document digest mismatch")
    if load_strict_json(beta_path) != {
        "schedule": "constant",
        "schema": BETA_SCHEMA,
        "value": 0.5,
    }:
        raise TrainSmokeError("beta document content mismatch")
    if sha256_file(config_path) != plan.get("config_sha256"):
        raise TrainSmokeError("train-smoke config digest mismatch")
    expected_config = {
        "action_contract": {
            "environment": 31,
            "homie": 15,
            "policy": 11,
            "replay": 26,
        },
        "beta_sha256": plan["beta_sha256"],
        "cohort_count": COHORT_COUNT,
        "environments_per_rank": ENVIRONMENTS_PER_RANK,
        "purpose": "train",
        "ranks_per_cohort": RANKS_PER_COHORT,
        "reference_lag": REFERENCE_LAG,
        "rollout_action_source": "beta",
        "schema": CONFIG_SCHEMA,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "views": VIEW_CONTRACT["views"],
    }
    config = load_strict_json(config_path)
    if config != expected_config:
        raise TrainSmokeError("train-smoke config content mismatch")
    anchors = _absolute_directory(plan["anchors"], "anchor pool")
    anchor_count = sum(1 for item in anchors.glob("*.npz") if item.is_file())
    if anchor_count != plan.get("anchor_count") or anchor_count < 2048:
        raise TrainSmokeError("anchor pool count changed or is incomplete")

    cohorts = plan.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != COHORT_COUNT:
        raise TrainSmokeError("train-smoke plan must contain exactly four cohorts")
    states: list[str] = []
    for cohort_index, item in enumerate(cohorts):
        if not isinstance(item, Mapping):
            raise TrainSmokeError("train-smoke cohort entry is not a mapping")
        if set(item) != _COHORT_PLAN_KEYS:
            raise TrainSmokeError(f"cohort {cohort_index} plan key-set mismatch")
        slot = cohort_index % 2
        expected_item = {
            "cohort_index": cohort_index,
            "global_env_id": cohort_index,
            "gpu_id": slot,
            "renderer_slot": slot,
        }
        for key, value in expected_item.items():
            if item.get(key) != value:
                raise TrainSmokeError(f"cohort {cohort_index} mapping mismatch for {key}")
        descriptor = _absolute_regular(item["endpoint_descriptor"], "endpoint descriptor")
        if descriptor != descriptors[slot] or sha256_file(descriptor) != item.get(
            "endpoint_descriptor_sha256"
        ):
            raise TrainSmokeError(f"cohort {cohort_index} renderer descriptor changed")
        selected_manifest = _absolute_regular(
            item["endpoint_manifest"], "selected endpoint manifest"
        )
        if selected_manifest.read_text(encoding="utf-8") != f"{descriptor}\n":
            raise TrainSmokeError(f"cohort {cohort_index} endpoint mapping is not exact")
        if sha256_file(selected_manifest) != item.get("endpoint_manifest_sha256"):
            raise TrainSmokeError(f"cohort {cohort_index} endpoint manifest changed")
        cohort_dir = _absolute_directory(item["cohort_dir"], "cohort directory")
        contract_path = _absolute_regular(cohort_dir / "contract.json", "cohort contract")
        if sha256_file(contract_path) != item.get("contract_sha256"):
            raise TrainSmokeError(f"cohort {cohort_index} contract changed")
        contract = load_strict_json(contract_path)
        expected_contract = {
            "campaign_id": CAMPAIGN_ID,
            "macro_round": 0,
            "cohort_index": cohort_index,
            "purpose": "train",
            "expected_ranks": [0],
            "expected_global_env_ids": [cohort_index],
            "expected_transitions": 2,
            "expected_steps_per_env": 2,
            "view_contract": VIEW_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "policy_sha256": plan["policy_sha256"],
            "runtime_policy_sha256": plan["runtime_policy_sha256"],
            "teacher_sha256": plan["teacher_sha256"],
            "renderer_sha256": renderer_sha,
            "beta_sha256": plan["beta_sha256"],
            "config_sha256": plan["config_sha256"],
        }
        mismatches = [
            key for key, value in expected_contract.items() if contract.get(key) != value
        ]
        if mismatches:
            raise TrainSmokeError(
                f"cohort {cohort_index} immutable contract mismatch: {mismatches}"
            )
        seal = cohort_dir / "seal.json"
        if seal.exists() or seal.is_symlink():
            report = verify_cohort(cohort_dir)
            if (
                report.get("purpose") != "train"
                or report.get("ranks") != [0]
                or report.get("global_env_ids") != [cohort_index]
                or report.get("transition_count") != 2
            ):
                raise TrainSmokeError(f"cohort {cohort_index} seal coverage mismatch")
            states.append("sealed")
        else:
            states.append(inspect_actor_rank(cohort_dir, 0)["state"])

    return {
        "schema": "fast_roborender.round_dagger.actor_train_smoke_validation.v1",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "campaign_id": CAMPAIGN_ID,
        "cohort_count": COHORT_COUNT,
        "expected_transitions": COHORT_COUNT * STEPS_PER_ENVIRONMENT,
        "global_env_ids": list(range(COHORT_COUNT)),
        "gpu_schedule": [0, 1, 0, 1],
        "renderer_schedule": [0, 1, 0, 1],
        "states": states,
    }


def train_smoke_environment(path: str | Path, cohort_index: int) -> dict[str, str]:
    """Return the exact environment for one fresh cohort invocation."""

    report = validate_train_smoke(path)
    if type(cohort_index) is not int or not 0 <= cohort_index < COHORT_COUNT:
        raise TrainSmokeError("cohort index must be one of 0, 1, 2, or 3")
    state = report["states"][cohort_index]
    if state != "empty":
        raise TrainSmokeError(
            f"cohort {cohort_index} is {state}; refusing to overwrite immutable evidence"
        )
    _, plan = _plan(path)
    cohort = plan["cohorts"][cohort_index]
    return {
        "ACCELERATE": plan["accelerate"],
        "ANCHORS": plan["anchors"],
        "DRY_RUN": "0",
        "GR00T_PYTHON": plan["gr00t_python"],
        "GR00T_ROOT": plan["gr00t_root"],
        "IMAGE3F_ENDPOINT_MANIFEST": cohort["endpoint_manifest"],
        "ROUND_DAGGER_ACTOR_DOCKER_GPU_ID": str(cohort["gpu_id"]),
        "ROUND_DAGGER_BETA": "0.5",
        "ROUND_DAGGER_BETA_SHA256": plan["beta_sha256"],
        "ROUND_DAGGER_CAMPAIGN_ID": CAMPAIGN_ID,
        "ROUND_DAGGER_COHORT_DIR": cohort["cohort_dir"],
        "ROUND_DAGGER_COHORT_INDEX": str(cohort_index),
        "ROUND_DAGGER_CONFIG_SHA256": plan["config_sha256"],
        "ROUND_DAGGER_ENVS_PER_SHARD": "1",
        "ROUND_DAGGER_EXPECTED_RENDERERS": "1",
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(cohort_index),
        "ROUND_DAGGER_MACRO_ROUND": "0",
        "ROUND_DAGGER_MIN_ANCHORS": "2048",
        "ROUND_DAGGER_POLICY_CHECKPOINT": plan["policy_checkpoint"],
        "ROUND_DAGGER_POLICY_MANIFEST": plan["policy_manifest"],
        "ROUND_DAGGER_POLICY_SHA256": plan["policy_sha256"],
        "ROUND_DAGGER_POLICY_SPEC": plan["policy_spec"],
        "ROUND_DAGGER_RENDERER_SHA256": plan["renderer_sha256"],
        "ROUND_DAGGER_REPOSITORY_ROOT": plan["repository_root"],
        "ROUND_DAGGER_ROLLOUT_ID": "0",
        "ROUND_DAGGER_RTX_IMAGE": plan["actor_image"],
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": plan["runtime_policy_sha256"],
        "ROUND_DAGGER_TEACHER_CHECKPOINT": plan["teacher_checkpoint"],
        "ROUND_DAGGER_TEACHER_SHA256": plan["teacher_sha256"],
        "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": "1",
        "ROUND_DAGGER_VALIDATION_PURPOSE": "train",
        "ROUND_DAGGER_VALIDATION_RANKS": "1",
        "ROUND_DAGGER_VALIDATION_STEPS": "2",
        "ROUND_DAGGER_VALIDATION_TRAIN_ACK": TRAIN_ACKNOWLEDGEMENT,
        "ROBORENDER_TELEMETRY_DIR": f"{cohort['run_dir']}/telemetry",
        "ROLLOUT_STEPS": "2",
        "TRAIN_RUN_DIR": cohort["run_dir"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--endpoint-manifest", required=True, type=Path)
    prepare.add_argument("--policy-checkpoint", required=True, type=Path)
    prepare.add_argument("--teacher-checkpoint", required=True, type=Path)
    prepare.add_argument("--renderer-sha256", required=True)
    prepare.add_argument("--anchors", required=True, type=Path)
    prepare.add_argument("--repository-root", required=True, type=Path)
    prepare.add_argument("--gr00t-root", required=True, type=Path)
    prepare.add_argument("--actor-image", required=True)
    prepare.add_argument("--gr00t-python", default="/opt/round-dagger-rtx/bin/python")
    prepare.add_argument("--accelerate", default="/opt/round-dagger-rtx/bin/accelerate")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", required=True, type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("--plan", required=True, type=Path)
    status.add_argument("--cohort-index", required=True, type=int)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--plan", required=True, type=Path)
    environment.add_argument("--cohort-index", required=True, type=int)
    environment.add_argument("--null", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "prepare":
        result = prepare_train_smoke(
            arguments.root,
            endpoint_manifest=arguments.endpoint_manifest,
            policy_checkpoint=arguments.policy_checkpoint,
            teacher_checkpoint=arguments.teacher_checkpoint,
            renderer_sha256=arguments.renderer_sha256,
            anchors=arguments.anchors,
            repository_root=arguments.repository_root,
            gr00t_root=arguments.gr00t_root,
            actor_image=arguments.actor_image,
            gr00t_python=arguments.gr00t_python,
            accelerate=arguments.accelerate,
        )
    elif arguments.operation == "validate":
        result = validate_train_smoke(arguments.plan)
    elif arguments.operation == "status":
        report = validate_train_smoke(arguments.plan)
        if not 0 <= arguments.cohort_index < COHORT_COUNT:
            raise TrainSmokeError("cohort index must be one of 0, 1, 2, or 3")
        result = {
            "cohort_index": arguments.cohort_index,
            "state": report["states"][arguments.cohort_index],
        }
    else:
        result = train_smoke_environment(arguments.plan, arguments.cohort_index)
        if arguments.null:
            for key in sorted(result):
                sys.stdout.buffer.write(key.encode("utf-8") + b"\0")
                sys.stdout.buffer.write(result[key].encode("utf-8") + b"\0")
            return 0
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TrainSmokeError, ActorPairError, ArtifactError, KeyError) as error:
        print(f"train-smoke gate failed: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "TrainSmokeError",
    "prepare_train_smoke",
    "train_smoke_environment",
    "validate_train_smoke",
]
