#!/usr/bin/env python3
"""Fail-closed eight-H200 learner runner for the ``validation_20m`` preset.

This entry point is intentionally independent of the production learner
launcher.  It accepts one portable RTX handoff and one immutable runtime
contract, validates the complete bounded shape, derives the canonical learner
and W&B identities, and finally replaces itself with the real GR00T DDP
learner.  It has no resume path: a validation run is a fresh, bounded proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger import artifacts, campaign_presets, replay  # noqa: E402
from round_dagger.learner_identity import (  # noqa: E402
    learner_identity_sha256,
    learner_wandb_run_id,
)


CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.validation_20m_learner_contract.v1"
)
PORTABLE_HANDOFF_SCHEMA = "fast_roborender.round_dagger.portable_handoff.v1"
PRESET_ID = "validation_20m"
WORLD_SIZE = 8
TRAIN_TRANSITIONS = 128
HELDOUT_TRANSITIONS = 32
TRAIN_TRANSITIONS_PER_RANK = 16
HELDOUT_TRANSITIONS_PER_RANK = 4
MAX_STEPS = 20
SAVE_EVERY = 10
HELDOUT_EVAL_INTERVAL = 10
BATCH_SIZE_PER_RANK = 1
GLOBAL_BATCH_SIZE = 8
LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 0.01
DAGGER_BETA = 0.5
SEED = 17
PARTITION_SEED = 23
MASTER_PORT = 29641
HARD_WALL_TIME_SECONDS = 1200
TIMEOUT_BINARY = Path("/usr/bin/timeout")

REMOTE_PROJECT_ROOT = Path("/workspace/projects/roborender")
VALIDATION_OUTPUT_ROOT = REMOTE_PROJECT_ROOT / "validation" / PRESET_ID
SOURCE_ROOT = REMOTE_PROJECT_ROOT / "src" / "fast_roborender"
SOURCE_INVENTORY = SOURCE_ROOT.parent / f"{SOURCE_ROOT.name}.inventory.json"
GR00T_ROOT = SOURCE_ROOT / "GR00T-VisualSim2Real"
H200_PYTHON = REMOTE_PROJECT_ROOT / "envs" / "h200_learner" / "bin" / "python"
PARENT_CHECKPOINT = (
    REMOTE_PROJECT_ROOT / "checkpoints" / "bootstrap_v0" / "checkpoint.pt"
)
PARENT_POLICY_SPEC = Path(f"{PARENT_CHECKPOINT}.policy_spec.json")
PARENT_CHECKPOINT_SHA256 = (
    "720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a"
)
PARENT_RUNTIME_POLICY_SHA256 = (
    "cd6f70aa9d589e810b7ab22999941930742a94621870457aadd068d2394deb9a"
)
RUNTIME_CONTAINER_SHA256 = (
    "2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
)
WANDB_PROJECT = "lepton-dagger-production-1"
WANDB_ENTITY = "nvidia"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CONTRACT_KEYS = {
    "schema",
    "preset_id",
    "profile_sha256",
    "campaign_id",
    "macro_round",
    "learner_code_sha256",
    "runtime_container_sha256",
    "parent",
    "handoff",
    "allocation",
    "optimization",
    "wandb",
}
_PARENT_KEYS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "policy_spec_path",
    "runtime_policy_sha256",
}
_HANDOFF_KEYS = {
    "manifest_file_sha256",
    "training_snapshot_file_sha256",
    "training_snapshot_sha256",
    "heldout_snapshot_file_sha256",
    "heldout_snapshot_sha256",
}
_ALLOCATION = {
    "nodes": 1,
    "world_size": WORLD_SIZE,
    "batch_size_per_rank": BATCH_SIZE_PER_RANK,
    "global_batch_size": GLOBAL_BATCH_SIZE,
}
_OPTIMIZATION = {
    "optimizer_steps": MAX_STEPS,
    "save_every_steps": SAVE_EVERY,
    "heldout_eval_interval_steps": HELDOUT_EVAL_INTERVAL,
    "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "partition_seed": PARTITION_SEED,
}
_WANDB = {
    "mode": "online",
    "entity": WANDB_ENTITY,
    "project": WANDB_PROJECT,
    "resume": "never",
}
_VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
_ACTION_CONTRACT = {"dtype": "float32", "shape": [26]}


class Validation20mRunnerError(RuntimeError):
    """The validation learner request is unsafe or semantically inconsistent."""


@dataclass(frozen=True)
class RunnerPlan:
    contract: Mapping[str, Any]
    contract_sha256: str
    handoff_root: Path
    output_dir: Path
    training_snapshot: Path
    heldout_snapshot: Path
    scene_sample_manifest: Path
    bundle_path: Path
    bundle_mapping: Mapping[str, Any]
    learner_config_sha256: str
    learner_identity_sha256: str
    wandb_run_id: str
    wandb_run_name: str
    command: tuple[str, ...]
    environment: Mapping[str, str]

    def dry_run_report(self) -> dict[str, Any]:
        return {
            "schema": "fast_roborender.round_dagger.validation_20m_h200_plan.v1",
            "status": "DRY_RUN_PASS",
            "preset_id": PRESET_ID,
            "campaign_id": self.contract["campaign_id"],
            "macro_round": self.contract["macro_round"],
            "world_size": WORLD_SIZE,
            "train_transitions": TRAIN_TRANSITIONS,
            "heldout_transitions": HELDOUT_TRANSITIONS,
            "scene_sample_manifest": str(self.scene_sample_manifest),
            "optimizer_steps": MAX_STEPS,
            "save_every_steps": SAVE_EVERY,
            "heldout_eval_interval_steps": HELDOUT_EVAL_INTERVAL,
            "required_heldout_frontiers": [0, 10, 20],
            "hard_wall_time_seconds": HARD_WALL_TIME_SECONDS,
            "contract_sha256": self.contract_sha256,
            "learner_config_sha256": self.learner_config_sha256,
            "learner_identity_sha256": self.learner_identity_sha256,
            "wandb_project": WANDB_PROJECT,
            "wandb_entity": WANDB_ENTITY,
            "wandb_run_id": self.wandb_run_id,
            "wandb_run_name": self.wandb_run_name,
            "wandb_resume": "never",
            "bundle_path": str(self.bundle_path),
            "command": list(self.command),
        }


def _mapping(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Validation20mRunnerError(f"{label} has an unexpected key-set")
    return dict(value)


def _prepare_gr00t_bundle(**arguments: Any) -> Any:
    """Import the Torch-backed bundle builder only in the H200 runtime."""

    from scripts.round_dagger.prepare_gr00t_learner_bundle import prepare_bundle

    return prepare_bundle(**arguments)


def _partition_counts(
    shards: Sequence[Mapping[str, Any]], *, world_size: int, seed: int
) -> list[int]:
    """Mirror the learner's deterministic LPT assignment without importing Torch."""

    normalized: list[tuple[int, str, str]] = []
    identities: set[str] = set()
    for raw in shards:
        logical_id = raw.get("logical_id")
        shard_id = raw.get("shard_id")
        transitions = raw.get("transition_count")
        if not isinstance(logical_id, str) or not isinstance(shard_id, str):
            raise Validation20mRunnerError("replay shard identity is malformed")
        if type(transitions) is not int or transitions < 1:
            raise Validation20mRunnerError("replay shard transition count is malformed")
        identity = f"{logical_id}/{shard_id}"
        if identity in identities:
            raise Validation20mRunnerError("replay contains duplicate shard identity")
        identities.add(identity)
        digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
        normalized.append((transitions, digest, identity))
    normalized.sort(key=lambda row: (-row[0], row[1], row[2]))
    counts = [0] * world_size
    shard_counts = [0] * world_size
    for transitions, _digest, _identity in normalized:
        rank = min(
            range(world_size),
            key=lambda candidate: (
                counts[candidate],
                shard_counts[candidate],
                candidate,
            ),
        )
        counts[rank] += transitions
        shard_counts[rank] += 1
    return counts


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Validation20mRunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise Validation20mRunnerError(f"{label} must be a safe token")
    return value


def _absolute(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if not path.is_absolute() or os.path.abspath(raw) != raw:
        raise Validation20mRunnerError(f"{label} must be canonical and absolute")
    return path


def _regular(path: Path, label: str, *, immutable: bool = False) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise Validation20mRunnerError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise Validation20mRunnerError(f"{label} must be a regular non-symlink file")
    if immutable and info.st_mode & 0o222:
        raise Validation20mRunnerError(f"{label} must be read-only")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise Validation20mRunnerError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise Validation20mRunnerError(f"{label} must be a real directory")
    return path


def _executable(path: Path, label: str) -> Path:
    """Accept a venv symlink while validating its final executable target."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Validation20mRunnerError(f"{label} is unavailable") from exc
    _regular(resolved, f"{label} target")
    if not os.access(path, os.X_OK):
        raise Validation20mRunnerError(f"{label} is not executable")
    return path


def _beneath(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Validation20mRunnerError(f"{label} is outside {root}") from exc
    if not relative.parts:
        raise Validation20mRunnerError(f"{label} must be below {root}")


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = _regular(_absolute(path, "learner contract"), "learner contract", immutable=True)
    _beneath(path, REMOTE_PROJECT_ROOT, "learner contract")
    try:
        document = _mapping(
            artifacts.load_strict_json(path), _CONTRACT_KEYS, "learner contract"
        )
    except Validation20mRunnerError:
        raise
    except Exception as exc:
        raise Validation20mRunnerError("cannot load learner contract") from exc
    if document["schema"] != CONTRACT_SCHEMA:
        raise Validation20mRunnerError("learner contract schema mismatch")
    if document["preset_id"] != PRESET_ID:
        raise Validation20mRunnerError("learner contract is not validation_20m")
    profile_path = campaign_presets.profile_path(PRESET_ID)
    campaign_presets.load_profile(PRESET_ID)
    if document["profile_sha256"] != artifacts.sha256_file(profile_path):
        raise Validation20mRunnerError("learner contract profile digest mismatch")
    _token(document["campaign_id"], "campaign ID")
    if document["macro_round"] != 0:
        raise Validation20mRunnerError("validation_20m macro round must be zero")
    _sha(document["learner_code_sha256"], "learner code SHA-256")
    if _sha(
        document["runtime_container_sha256"], "runtime container SHA-256"
    ) != RUNTIME_CONTAINER_SHA256:
        raise Validation20mRunnerError("runtime container digest mismatch")

    parent = _mapping(document["parent"], _PARENT_KEYS, "parent binding")
    expected_parent = {
        "checkpoint_path": str(PARENT_CHECKPOINT),
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "policy_spec_path": str(PARENT_POLICY_SPEC),
        "runtime_policy_sha256": PARENT_RUNTIME_POLICY_SHA256,
    }
    if parent != expected_parent:
        raise Validation20mRunnerError("parent policy binding mismatch")
    handoff = _mapping(document["handoff"], _HANDOFF_KEYS, "handoff binding")
    for key, value in handoff.items():
        _sha(value, f"handoff {key}")
    if document["allocation"] != _ALLOCATION:
        raise Validation20mRunnerError("validation learner allocation mismatch")
    if document["optimization"] != _OPTIMIZATION:
        raise Validation20mRunnerError("validation learner optimization mismatch")
    if document["wandb"] != _WANDB:
        raise Validation20mRunnerError("validation learner W&B contract mismatch")
    return document, artifacts.sha256_file(path)


def _validate_handoff_manifest(
    root: Path, contract: Mapping[str, Any]
) -> tuple[Path, Path]:
    root = _directory(_absolute(root, "handoff root"), "handoff root")
    _beneath(root, REMOTE_PROJECT_ROOT, "handoff root")
    manifest_path = _regular(
        root / "portable_handoff_manifest.json",
        "portable handoff manifest",
        immutable=True,
    )
    binding = contract["handoff"]
    if artifacts.sha256_file(manifest_path) != binding["manifest_file_sha256"]:
        raise Validation20mRunnerError("portable handoff manifest digest mismatch")
    try:
        document = artifacts.load_strict_json(manifest_path)
    except Exception as exc:
        raise Validation20mRunnerError("cannot load portable handoff manifest") from exc
    expected_keys = {
        "schema",
        "complete",
        "destination_root",
        "campaign_id",
        "source",
        "destination",
        "equivalence",
        "cohort_immutability",
        "builder",
        "handoff_content_sha256",
    }
    document = _mapping(document, expected_keys, "portable handoff manifest")
    if document["schema"] != PORTABLE_HANDOFF_SCHEMA or document["complete"] is not True:
        raise Validation20mRunnerError("portable handoff is not complete")
    if document["destination_root"] != str(root):
        raise Validation20mRunnerError("portable handoff root changed after publication")
    if document["campaign_id"] != contract["campaign_id"]:
        raise Validation20mRunnerError("portable handoff campaign mismatch")
    semantic = dict(document)
    declared = _sha(semantic.pop("handoff_content_sha256"), "handoff content SHA-256")
    if artifacts.canonical_json_sha256(semantic) != declared:
        raise Validation20mRunnerError("portable handoff semantic digest mismatch")

    destination = document["destination"]
    if not isinstance(destination, Mapping):
        raise Validation20mRunnerError("portable handoff destination is malformed")
    training = destination.get("training_snapshot")
    heldout = destination.get("heldout_snapshot")
    if not isinstance(training, Mapping) or not isinstance(heldout, Mapping):
        raise Validation20mRunnerError("portable handoff snapshots are missing")
    training_path = root / "training_snapshot.json"
    heldout_path = root / "heldout_snapshot.json"
    expected_training = {
        "path": str(training_path),
        "file_sha256": binding["training_snapshot_file_sha256"],
        "snapshot_sha256": binding["training_snapshot_sha256"],
        "transition_count": TRAIN_TRANSITIONS,
    }
    expected_heldout = {
        "path": str(heldout_path),
        "file_sha256": binding["heldout_snapshot_file_sha256"],
        "snapshot_sha256": binding["heldout_snapshot_sha256"],
        "transition_count": HELDOUT_TRANSITIONS,
    }
    if dict(training) != expected_training or dict(heldout) != expected_heldout:
        raise Validation20mRunnerError("portable handoff snapshot binding mismatch")
    equivalence = document["equivalence"]
    if not isinstance(equivalence, Mapping) or equivalence.get(
        "train_eval_payload_intersection"
    ) != []:
        raise Validation20mRunnerError("portable handoff train/eval data overlap")
    required_true = (
        "source_destination_cohort_seals_exact",
        "source_destination_training_payload_sets_exact",
        "source_destination_heldout_payload_sets_exact_via_seal",
    )
    if any(equivalence.get(key) is not True for key in required_true):
        raise Validation20mRunnerError("portable handoff equivalence is incomplete")
    immutability = document["cohort_immutability"]
    if not isinstance(immutability, Mapping) or immutability.get("verified") is not True:
        raise Validation20mRunnerError("portable handoff cohort immutability is unverified")
    if immutability.get("content_and_inode_metadata_unchanged") is not True:
        raise Validation20mRunnerError("portable handoff cohort bytes changed")
    return training_path, heldout_path


def _validate_scene_samples(root: Path, *, campaign_id: str) -> Path:
    """Validate the bounded qualitative evidence shipped with the handoff."""

    directory = _directory(root / "scene_samples", "scene sample directory")
    manifest_path = _regular(
        directory / "manifest.json", "scene sample manifest", immutable=True
    )
    try:
        document = artifacts.load_strict_json(manifest_path)
    except Exception as exc:
        raise Validation20mRunnerError("cannot load scene sample manifest") from exc
    if (
        document.get("schema")
        != "fast_roborender.round_dagger.scene_samples.v1"
        or document.get("campaign_id") != campaign_id
    ):
        raise Validation20mRunnerError("scene sample manifest lineage mismatch")
    semantic = dict(document)
    declared = semantic.pop("sample_set_sha256", None)
    if not isinstance(declared, str) or artifacts.canonical_json_sha256(semantic) != declared:
        raise Validation20mRunnerError("scene sample semantic digest mismatch")
    if document.get("view_labels") != ["head_color", "hand_left", "hand_right"]:
        raise Validation20mRunnerError("scene sample view labels mismatch")
    if document.get("row_labels") != ["depth", "mask", "client_anchor", "generated"]:
        raise Validation20mRunnerError("scene sample row labels mismatch")
    samples = document.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 4:
        raise Validation20mRunnerError("scene sample count must be in [1,4]")
    expected_entries = {"manifest.json"}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise Validation20mRunnerError("scene sample entry is malformed")
        name = sample.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise Validation20mRunnerError("scene sample path is unsafe")
        image = _regular(directory / name, "scene sample image", immutable=True)
        expected_entries.add(name)
        if (
            sample.get("width") != 1252
            or sample.get("height") != 1046
            or artifacts.sha256_file(image) != sample.get("sha256")
            or image.stat().st_size != sample.get("size_bytes")
        ):
            raise Validation20mRunnerError("scene sample image identity mismatch")
    if {path.name for path in directory.iterdir()} != expected_entries:
        raise Validation20mRunnerError("scene sample directory contains extra entries")
    return manifest_path


def _validate_snapshots(
    training_path: Path,
    heldout_path: Path,
    contract: Mapping[str, Any],
    handoff_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = contract["handoff"]
    _regular(training_path, "training replay snapshot", immutable=True)
    _regular(heldout_path, "held-out replay snapshot", immutable=True)
    try:
        training = replay.load_snapshot(
            training_path,
            expected_file_sha256=binding["training_snapshot_file_sha256"],
        )
        heldout = replay.load_heldout_snapshot(
            heldout_path,
            expected_file_sha256=binding["heldout_snapshot_file_sha256"],
        )
    except Exception as exc:
        raise Validation20mRunnerError(f"replay snapshot verification failed: {exc}") from exc
    if training["snapshot_sha256"] != binding["training_snapshot_sha256"]:
        raise Validation20mRunnerError("training snapshot semantic digest mismatch")
    if heldout["snapshot_sha256"] != binding["heldout_snapshot_sha256"]:
        raise Validation20mRunnerError("held-out snapshot semantic digest mismatch")
    campaign_id = contract["campaign_id"]
    if training.get("campaign_id") != campaign_id or heldout.get("campaign_id") != campaign_id:
        raise Validation20mRunnerError("replay snapshot campaign mismatch")
    if training.get("through_macro_round") != 0 or training.get(
        "expected_train_cohorts"
    ) != 1:
        raise Validation20mRunnerError("training replay is not one round/one cohort")
    if training.get("counts_by_round") != {"0": TRAIN_TRANSITIONS} or training.get(
        "counts_by_round_cohort"
    ) != {"0:0": TRAIN_TRANSITIONS}:
        raise Validation20mRunnerError("training replay round counts are inconsistent")
    if training.get("cohort_count") != 1 or len(training.get("cohorts", [])) != 1:
        raise Validation20mRunnerError("training replay must contain one cohort")
    if heldout.get("cohort_count") != 1 or len(heldout.get("cohorts", [])) != 1:
        raise Validation20mRunnerError("held-out replay must contain one cohort")
    if training.get("transition_count") != TRAIN_TRANSITIONS:
        raise Validation20mRunnerError("training replay must contain 128 transitions")
    if heldout.get("transition_count") != HELDOUT_TRANSITIONS:
        raise Validation20mRunnerError("held-out replay must contain 32 cases")
    if training.get("view_contract") != _VIEW_CONTRACT or heldout.get(
        "view_contract"
    ) != _VIEW_CONTRACT:
        raise Validation20mRunnerError("replay three-view contract mismatch")
    if training.get("action_contract") != _ACTION_CONTRACT or heldout.get(
        "action_contract"
    ) != _ACTION_CONTRACT:
        raise Validation20mRunnerError("replay action contract mismatch")

    train_shards = training.get("resolved_shards")
    eval_shards = heldout.get("resolved_shards")
    if not isinstance(train_shards, list) or len(train_shards) != WORLD_SIZE:
        raise Validation20mRunnerError("training replay must have exactly eight shards")
    if not isinstance(eval_shards, list) or len(eval_shards) != WORLD_SIZE:
        raise Validation20mRunnerError("held-out replay must have exactly eight shards")
    if any(row.get("transition_count") != TRAIN_TRANSITIONS_PER_RANK for row in train_shards):
        raise Validation20mRunnerError("each training shard must contain 16 transitions")
    if any(row.get("transition_count") != HELDOUT_TRANSITIONS_PER_RANK for row in eval_shards):
        raise Validation20mRunnerError("each held-out shard must contain four cases")
    for label, shards, expected in (
        ("training", train_shards, TRAIN_TRANSITIONS_PER_RANK),
        ("held-out", eval_shards, HELDOUT_TRANSITIONS_PER_RANK),
    ):
        counts = _partition_counts(
            shards, world_size=WORLD_SIZE, seed=PARTITION_SEED
        )
        if counts != [expected] * WORLD_SIZE:
            raise Validation20mRunnerError(
                f"{label} replay does not partition evenly across eight ranks"
            )
    train_payloads = {row["payload_sha256"] for row in train_shards}
    eval_payloads = {row["payload_sha256"] for row in eval_shards}
    train_manifests = {row["manifest_sha256"] for row in train_shards}
    eval_manifests = {row["manifest_sha256"] for row in eval_shards}
    if train_payloads & eval_payloads or train_manifests & eval_manifests:
        raise Validation20mRunnerError("training and held-out shards overlap")
    _validate_cohort_seal(
        training["cohorts"][0],
        campaign_id=campaign_id,
        purpose="train",
        expected_transitions=TRAIN_TRANSITIONS,
        expected_shard_transitions=TRAIN_TRANSITIONS_PER_RANK,
        handoff_root=handoff_root,
        require_actor_topology=True,
    )
    _validate_cohort_seal(
        heldout["cohorts"][0],
        campaign_id=campaign_id,
        purpose="eval",
        expected_transitions=HELDOUT_TRANSITIONS,
        expected_shard_transitions=HELDOUT_TRANSITIONS_PER_RANK,
        handoff_root=handoff_root,
        require_actor_topology=False,
    )
    return training, heldout


def _validate_cohort_seal(
    record: Mapping[str, Any],
    *,
    campaign_id: str,
    purpose: str,
    expected_transitions: int,
    expected_shard_transitions: int,
    handoff_root: Path,
    require_actor_topology: bool,
) -> None:
    """Recheck the recursively bound cohort's topology and parent lineage."""

    raw_path = record.get("cohort_manifest_path")
    if not isinstance(raw_path, str):
        raise Validation20mRunnerError(f"{purpose} cohort manifest path is missing")
    path = _regular(
        _absolute(Path(raw_path), f"{purpose} cohort manifest"),
        f"{purpose} cohort manifest",
        immutable=True,
    )
    _beneath(path, handoff_root, f"{purpose} cohort manifest")
    expected_sha = _sha(record.get("manifest_sha256"), f"{purpose} cohort SHA-256")
    if artifacts.sha256_file(path) != expected_sha:
        raise Validation20mRunnerError(f"{purpose} cohort digest mismatch")
    try:
        seal = artifacts.load_strict_json(path)
    except Exception as exc:
        raise Validation20mRunnerError(f"cannot load {purpose} cohort seal") from exc
    expected = {
        "schema": "fast_roborender.round_dagger.cohort.v1",
        "sealed": True,
        "campaign_id": campaign_id,
        "macro_round": 0,
        "purpose": purpose,
        "policy_sha256": PARENT_CHECKPOINT_SHA256,
        "runtime_policy_sha256": PARENT_RUNTIME_POLICY_SHA256,
        "beta_sha256": artifacts.canonical_json_sha256({"beta": DAGGER_BETA}),
        "expected_transitions": expected_transitions,
        "transition_count": expected_transitions,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise Validation20mRunnerError(f"{purpose} cohort lineage/topology mismatch")
    shards = seal.get("shards")
    if not isinstance(shards, list) or len(shards) != WORLD_SIZE:
        raise Validation20mRunnerError(f"{purpose} cohort must seal eight shards")
    if any(
        not isinstance(shard, Mapping)
        or shard.get("transition_count") != expected_shard_transitions
        for shard in shards
    ):
        raise Validation20mRunnerError(
            f"{purpose} cohort shard transition shape mismatch"
        )
    if require_actor_topology:
        ranks = seal.get("ranks")
        expected_ranks = seal.get("expected_ranks")
        if ranks != [0, 1] or expected_ranks != [0, 1]:
            raise Validation20mRunnerError(
                "training cohort must come from exactly two actor ranks"
            )
        shard_ranks = [shard.get("rank") for shard in shards]
        if shard_ranks.count(0) != 4 or shard_ranks.count(1) != 4:
            raise Validation20mRunnerError(
                "training cohort must contain four environment shards per actor rank"
            )
        global_env_ids = seal.get("global_env_ids")
        expected_global_env_ids = seal.get("expected_global_env_ids")
        shard_global_env_ids = [
            value
            for shard in shards
            for value in shard.get("global_env_ids", [])
        ]
        if (
            not isinstance(global_env_ids, list)
            or len(global_env_ids) != 8
            or len(set(global_env_ids)) != 8
            or expected_global_env_ids != global_env_ids
            or sorted(shard_global_env_ids) != sorted(global_env_ids)
            or any(
                not isinstance(shard.get("global_env_ids"), list)
                or len(shard["global_env_ids"]) != 1
                for shard in shards
            )
        ):
            raise Validation20mRunnerError(
                "training cohort does not represent 2x4 distinct environments"
            )


def _validate_runtime_files() -> None:
    _directory(SOURCE_ROOT, "sealed source root")
    _directory(GR00T_ROOT, "GR00T root")
    _regular(SOURCE_ROOT / "scripts/round_dagger/train_learner.py", "learner program")
    _regular(PARENT_CHECKPOINT, "parent checkpoint")
    _regular(PARENT_POLICY_SPEC, "parent policy spec")
    _executable(H200_PYTHON, "H200 Python")
    _executable(TIMEOUT_BINARY, "timeout binary")
    if artifacts.sha256_file(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise Validation20mRunnerError("fixed parent checkpoint digest mismatch")


def _verify_sealed_source(expected_inventory_sha256: str) -> None:
    """Bind the executable source tree to the externally pinned inventory."""

    expected = _sha(expected_inventory_sha256, "source inventory SHA-256")
    inventory_path = _regular(
        SOURCE_INVENTORY, "sealed source inventory", immutable=True
    )
    if artifacts.sha256_file(inventory_path) != expected:
        raise Validation20mRunnerError("sealed source inventory digest mismatch")
    try:
        inventory = artifacts.load_strict_json(inventory_path)
        records = inventory["files"]
    except Exception as exc:
        raise Validation20mRunnerError("sealed source inventory is malformed") from exc
    verifier = SOURCE_ROOT / "scripts/round_dagger/verify_sealed_source_snapshot.py"
    relative = verifier.relative_to(SOURCE_ROOT).as_posix()
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("path") == relative
    ]
    if len(matches) != 1:
        raise Validation20mRunnerError("source inventory does not bind one verifier")
    verifier_sha256 = _sha(matches[0].get("sha256"), "source verifier SHA-256")
    _regular(verifier, "sealed-source verifier", immutable=True)
    if artifacts.sha256_file(verifier) != verifier_sha256:
        raise Validation20mRunnerError("sealed-source verifier digest mismatch")
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    try:
        completed = subprocess.run(
            [
                str(H200_PYTHON),
                str(verifier),
                "--root",
                str(SOURCE_ROOT),
                "--expected-inventory-sha256",
                expected,
            ],
            cwd=SOURCE_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Validation20mRunnerError("sealed-source verifier could not run") from exc
    if completed.returncode != 0:
        raise Validation20mRunnerError("sealed source verification failed")


def _validate_output_path(
    path: Path, handoff_root: Path, *, campaign_id: str
) -> Path:
    path = _absolute(path, "learner output directory")
    _beneath(path, VALIDATION_OUTPUT_ROOT, "learner output directory")
    expected = VALIDATION_OUTPUT_ROOT / campaign_id / "learner"
    if path != expected:
        raise Validation20mRunnerError(
            f"learner output must be the campaign-derived path {expected}"
        )
    if path == handoff_root or path in handoff_root.parents or handoff_root in path.parents:
        raise Validation20mRunnerError("learner output and handoff trees must be disjoint")
    current = path
    while not current.exists():
        current = current.parent
    while True:
        if current.is_symlink():
            raise Validation20mRunnerError("learner output path contains a symlink")
        if current == Path(current.anchor):
            break
        current = current.parent
    if path.exists():
        _directory(path, "learner output directory")
        if any(path.iterdir()):
            raise Validation20mRunnerError(
                "fresh validation learner output directory must be empty"
            )
    return path


def build_plan(
    *,
    handoff_root: Path,
    output_dir: Path,
    contract_path: Path,
    wandb_run_name: str,
) -> RunnerPlan:
    """Validate all inputs and return the exact non-executed learner plan."""

    run_name = _token(wandb_run_name, "W&B operator run name")
    contract, contract_sha256 = _load_contract(contract_path)
    handoff_root = _directory(_absolute(handoff_root, "handoff root"), "handoff root")
    training_path, heldout_path = _validate_handoff_manifest(handoff_root, contract)
    _validate_snapshots(training_path, heldout_path, contract, handoff_root)
    scene_sample_manifest = _validate_scene_samples(
        handoff_root, campaign_id=contract["campaign_id"]
    )
    _validate_runtime_files()
    _verify_sealed_source(contract["learner_code_sha256"])
    output_dir = _validate_output_path(
        output_dir, handoff_root, campaign_id=contract["campaign_id"]
    )

    bundle_path = output_dir / "learner_bundle_round_000000.json"
    try:
        bundle = _prepare_gr00t_bundle(
            base_policy_spec=PARENT_POLICY_SPEC,
            parent_checkpoint=PARENT_CHECKPOINT,
            expected_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            runtime_policy_sha256=PARENT_RUNTIME_POLICY_SHA256,
            campaign_id=contract["campaign_id"],
            macro_round=0,
            code_sha256=contract["learner_code_sha256"],
            batch_size=BATCH_SIZE_PER_RANK,
            max_steps=MAX_STEPS,
            save_every=SAVE_EVERY,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
            partition_seed=PARTITION_SEED,
            heldout_snapshot=heldout_path,
            heldout_eval_interval=HELDOUT_EVAL_INTERVAL,
        )
    except Exception as exc:
        raise Validation20mRunnerError(f"cannot prepare GR00T learner bundle: {exc}") from exc
    bundle_mapping = bundle.to_mapping()
    config_sha256 = bundle.sha256
    identity = learner_identity_sha256(
        campaign_id=contract["campaign_id"],
        macro_round=0,
        learner_code_sha256=contract["learner_code_sha256"],
        learner_config_sha256=config_sha256,
        replay_snapshot_file_sha256=contract["handoff"][
            "training_snapshot_file_sha256"
        ],
        replay_snapshot_sha256=contract["handoff"]["training_snapshot_sha256"],
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        parent_runtime_policy_sha256=PARENT_RUNTIME_POLICY_SHA256,
        runtime_container_sha256=RUNTIME_CONTAINER_SHA256,
        learner_contract_sha256=contract_sha256,
    )
    run_id = learner_wandb_run_id(
        campaign_id=contract["campaign_id"],
        macro_round=0,
        learner_identity_sha256=identity,
    )
    command = (
        str(TIMEOUT_BINARY),
        "--signal=TERM",
        "--kill-after=30s",
        f"{HARD_WALL_TIME_SECONDS - 30}s",
        str(H200_PYTHON),
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(WORLD_SIZE),
        "--node-rank",
        "0",
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(MASTER_PORT),
        str(SOURCE_ROOT / "scripts/round_dagger/train_learner.py"),
        "--bundle",
        str(bundle_path),
        "--snapshot",
        str(training_path),
        "--heldout-snapshot",
        str(heldout_path),
        "--output-dir",
        str(output_dir),
        "--model-factory",
        "round_dagger.gr00t_policy:create_vision_actor_obj_pred",
        "--adapter-factory",
        "round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory",
        "--initial-checkpoint",
        str(PARENT_CHECKPOINT),
        "--expected-snapshot-file-sha256",
        contract["handoff"]["training_snapshot_file_sha256"],
        "--learner-identity-sha256",
        identity,
        "--runtime-container-sha256",
        RUNTIME_CONTAINER_SHA256,
        "--learner-contract-sha256",
        contract_sha256,
        "--max-steps",
        str(MAX_STEPS),
        "--save-every",
        str(SAVE_EVERY),
        "--heldout-eval-interval",
        str(HELDOUT_EVAL_INTERVAL),
        "--device",
        "cuda",
        "--telemetry-jsonl",
        str(output_dir / "telemetry.jsonl"),
        "--heldout-telemetry-jsonl",
        str(output_dir / "heldout_telemetry.jsonl"),
        "--training-quality-telemetry-jsonl",
        str(output_dir / "training_quality_telemetry.jsonl"),
        "--dagger-wandb-dashboard",
        "--scene-sample-manifest",
        str(scene_sample_manifest),
        "--dagger-beta",
        str(DAGGER_BETA),
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-run-id",
        run_id,
        "--wandb-resume-never",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{SOURCE_ROOT}:{GR00T_ROOT}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "WANDB_MODE": "online",
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_RUN_ID": run_id,
            "WANDB_NAME": run_name,
            "WANDB_RESUME": "never",
        }
    )
    return RunnerPlan(
        contract=contract,
        contract_sha256=contract_sha256,
        handoff_root=handoff_root,
        output_dir=output_dir,
        training_snapshot=training_path,
        heldout_snapshot=heldout_path,
        scene_sample_manifest=scene_sample_manifest,
        bundle_path=bundle_path,
        bundle_mapping=bundle_mapping,
        learner_config_sha256=config_sha256,
        learner_identity_sha256=identity,
        wandb_run_id=run_id,
        wandb_run_name=run_name,
        command=command,
        environment=environment,
    )


def execute(plan: RunnerPlan, *, deadline_monotonic: float) -> None:
    """Publish the bundle and exec the real eight-rank learner exactly once."""

    if not os.environ.get("WANDB_API_KEY"):
        raise Validation20mRunnerError("WANDB_API_KEY is required for the online run")
    if (
        os.environ.get("ROUND_DAGGER_REMOTE_CONTAINER_SHA256")
        != RUNTIME_CONTAINER_SHA256
    ):
        raise Validation20mRunnerError(
            "runtime image attestation is missing or differs from the pinned H200 image"
        )
    resolved_python = H200_PYTHON.resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != resolved_python:
        raise Validation20mRunnerError("runner is not executing under pinned H200 Python")
    try:
        import torch
    except Exception as exc:
        raise Validation20mRunnerError("Torch is unavailable in the H200 runtime") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise Validation20mRunnerError("validation_20m requires exactly eight visible GPUs")
    device_names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if any("H200" not in name.upper() for name in device_names):
        raise Validation20mRunnerError(
            f"validation_20m requires eight H200 GPUs, observed={device_names!r}"
        )
    if plan.output_dir.exists():
        _directory(plan.output_dir, "learner output directory")
        if any(plan.output_dir.iterdir()):
            raise Validation20mRunnerError("learner output changed after validation")
    else:
        plan.output_dir.mkdir(parents=True, exist_ok=False)
        _directory(plan.output_dir, "learner output directory")
    artifacts.publish_immutable_json(plan.bundle_path, plan.bundle_mapping)
    if artifacts.canonical_json_sha256(plan.bundle_mapping) != plan.learner_config_sha256:
        raise Validation20mRunnerError("published learner bundle identity changed")
    remaining = int(deadline_monotonic - time.monotonic())
    if remaining <= 31:
        raise Validation20mRunnerError(
            "insufficient global validation budget remains to start the learner"
        )
    command = list(plan.command)
    command[3] = f"{remaining - 30}s"
    signal.alarm(0)
    os.execvpe(command[0], command, dict(plan.environment))
    raise Validation20mRunnerError("torch distributed exec unexpectedly returned")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="recursively validate and print the exact plan without writing or launching",
    )
    return parser


def _deadline_expired(_signum: int, _frame: Any) -> None:
    raise Validation20mRunnerError("validation_20m exceeded its global 1200-second limit")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    deadline = time.monotonic() + HARD_WALL_TIME_SECONDS
    previous_handler = signal.signal(signal.SIGALRM, _deadline_expired)
    signal.alarm(HARD_WALL_TIME_SECONDS)
    try:
        plan = build_plan(
            handoff_root=arguments.handoff_root,
            output_dir=arguments.output_dir,
            contract_path=arguments.contract,
            wandb_run_name=arguments.wandb_run_name,
        )
        if arguments.dry_run:
            print(json.dumps(plan.dry_run_report(), sort_keys=True, indent=2))
            return 0
        execute(plan, deadline_monotonic=deadline)
        return 0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Validation20mRunnerError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(2) from error
