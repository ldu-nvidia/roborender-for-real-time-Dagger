#!/usr/bin/env python3
"""Fail-closed gates for the two-node HSG Image3F fixed-seed evaluation.

This is deliberately separate from the simulator-RGB policy-mechanics gate.
It proves that the policy consumed the certified three-view Image3F stream,
that all seven warm renderers served successful requests, and that reset
conditions were paired across baseline/early/mid/final checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scaling"))

import fixed_seed_gate  # noqa: E402
import camera_calibration_gate as camera_gate  # noqa: E402
import hsg_gb200_smoke_gate as smoke_gate  # noqa: E402
import wandb_team_contract as team_contract  # noqa: E402


MANIFEST_SCHEMA = "fast_roborender.hsg_image3f_fixed_seed_manifest.v1"
PREFLIGHT_SCHEMA = "fast_roborender.hsg_image3f_fixed_seed_preflight.v2"
ATTESTATION_SCHEMA = "fast_roborender.hsg_image3f_fixed_seed_runtime.v1"
ACCEPTANCE_SCHEMA = "fast_roborender.hsg_image3f_fixed_seed_acceptance.v2"
WANDB_SCHEMA = "fast_roborender.hsg_image3f_fixed_seed_wandb_upload.v2"
OBSERVATION_MODE = "image3f_generated_rgb_3view"
ROLE_ORDER = ("baseline", "early", "mid", "final")
VIEWS = ("head_color", "hand_left", "hand_right")
SEQUENTIAL_TOPOLOGY = {
    "nodes": 2,
    "gpus_per_node": 4,
    "renderer_services": 7,
    "evaluator_gpus": 1,
    "concurrent_checkpoint_evaluators": 1,
    "envs_per_renderer_capacity": 5,
    "checkpoint_execution": "baseline_early_mid_final_sequential_same_warm_fleet",
}
RUNTIME_CONTRACT = {
    "inference_steps": 5,
    "tea_cache_threshold": 0.3,
    "cfg_scale": 1.0,
    "cuda_graph": False,
    "tensorrt_engine_required": True,
    "reference_lag": 0,
    "raytraced_mix_prob": 0.0,
    "policy_views": list(VIEWS),
    "policy_view_count": 3,
    "policy_view_layout": "vertical",
    "policy_image_shape_hwc": [324, 192, 3],
    "output_tile_shape_hwc": [240, 416, 3],
    "output_stacked_shape_hwc": [720, 416, 3],
    "eval_relabel": True,
    "init_at_random_ep_len": False,
    "one_episode_per_environment": True,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,180}\Z")


class GateError(RuntimeError):
    """A production-evaluation artifact violated its immutable contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {type(exc).__name__}") from None
    if not isinstance(payload, dict):
        raise GateError(f"{path} must contain one JSON object")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {type(exc).__name__}") from None
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_file(path: Path, label: str) -> Path:
    try:
        result = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} {path}: {type(exc).__name__}") from None
    if not result.is_file() or result.stat().st_size <= 0:
        raise GateError(f"{label} is not a non-empty regular file: {result}")
    return result


def require_dir(path: Path, label: str) -> Path:
    try:
        result = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {label} {path}: {type(exc).__name__}") from None
    if not result.is_dir():
        raise GateError(f"{label} is not a directory: {result}")
    return result


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase SHA-256")
    return value


def require_hash(path: Path, expected: str, label: str) -> str:
    expected = require_sha256(expected, f"expected {label}")
    actual = sha256_file(require_file(path, label))
    if actual != expected:
        raise GateError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def inventory_sha256(root: Path, pattern: str = "*.npz") -> tuple[str, int]:
    """Content-address every anchor byte, name, and size in deterministic order."""
    root = require_dir(root, "anchor pool")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise GateError(f"anchor pool has no files matching {pattern}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_hash = bytes.fromhex(sha256_file(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(file_hash)
    return digest.hexdigest(), len(files)


def parse_seeds(raw: str) -> list[int]:
    try:
        result = [int(piece) for piece in raw.split(",") if piece]
    except ValueError:
        raise GateError("seeds must be comma-separated integers") from None
    if len(result) < 2 or len(set(result)) != len(result) or any(v < 0 for v in result):
        raise GateError("seeds must contain at least two unique nonnegative integers")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest = fixed_seed_gate.create_manifest(
        train_run_dir=args.train_run_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        reference_config=args.reference_config,
        seeds=parse_seeds(args.seeds),
        num_envs=args.num_envs,
        output_path=args.output,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_reference_config_sha256=args.expected_reference_config_sha256,
    )
    if args.num_envs != 32:
        raise GateError("production Image3F evaluation is pinned to 32 environments")
    effective_overrides = {
        "+checkpoint": "<content-addressed-sanitized-role-checkpoint>",
        "+headless": True,
        "+num_envs": 32,
        "+seed": "<fixed-seed-block>",
        "++algo.config.videogen.primary_view": "head_color",
        "++algo.config.videogen.reference_lag": 0,
        "++algo.config.videogen.seed": "<fixed-seed-block>",
        "algo.config.eval.num_eval_episodes": 32,
        "algo.config.eval.eval_num_envs_episodes": True,
        "algo.config.eval.save_trajectories": False,
        "algo.config.eval.save_videos": False,
        "algo.config.init_at_random_ep_len": False,
        "algo.config.videogen.anchor_index": 0,
        "algo.config.videogen.anchor_pool_dir": "<content-addressed-anchor-pool>",
        "algo.config.videogen.image3f_endpoint_files": "<probed-selected-endpoint-manifest>",
        "algo.config.videogen.raytraced_mix_prob": 0.0,
        "algo.config.videogen.view_names": list(VIEWS),
        "++env.config.image_dagger_policy_view_count": 3,
        "simulator.config.cameras.policy_views": list(VIEWS),
    }
    manifest.update(
        {
            "manifest_schema": MANIFEST_SCHEMA,
            "observation_mode": OBSERVATION_MODE,
            "image3f_in_the_loop": True,
            "production_observation_distribution": True,
            "runtime_contract": RUNTIME_CONTRACT,
            "topology": SEQUENTIAL_TOPOLOGY,
            "effective_overrides": effective_overrides,
            "effective_contract_sha256": canonical_sha256(
                {
                    "runtime_contract": RUNTIME_CONTRACT,
                    "topology": SEQUENTIAL_TOPOLOGY,
                    "effective_overrides": effective_overrides,
                }
            ),
        }
    )
    atomic_json(args.output, manifest)
    return manifest


def validate_manifest(
    path: Path,
    *,
    train_run_dir: Path,
    expected_seeds: Sequence[int],
    expected_num_envs: int,
) -> dict[str, Any]:
    path = require_file(path, "Image3F fixed-seed manifest")
    manifest = load_json(path)
    fixed_seed_gate._require_manifest(manifest)
    expected = {
        "manifest_schema": MANIFEST_SCHEMA,
        "protocol": "one_episode_per_environment_fixed_seed_blocks",
        "observation_mode": OBSERVATION_MODE,
        "image3f_in_the_loop": True,
        "production_observation_distribution": True,
        "num_envs_per_seed": expected_num_envs,
        "expected_episodes_per_seed": expected_num_envs,
        "seeds": list(expected_seeds),
        "runtime_contract": RUNTIME_CONTRACT,
        "topology": SEQUENTIAL_TOPOLOGY,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise GateError(f"manifest {key} contract changed")
    if expected_num_envs != 32:
        raise GateError("Image3F production gate requires 32 environments")
    effective = {
        "runtime_contract": manifest.get("runtime_contract"),
        "topology": manifest.get("topology"),
        "effective_overrides": manifest.get("effective_overrides"),
    }
    if manifest.get("effective_contract_sha256") != canonical_sha256(effective):
        raise GateError("effective evaluation contract hash mismatch")

    run = require_dir(train_run_dir, "training run")
    declared_run = require_dir(Path(str(manifest.get("train_run_dir", ""))), "declared run")
    if run != declared_run:
        raise GateError("manifest points at a different training run")
    config = require_file(Path(str(manifest.get("reference_config", ""))), "reference config")
    if config != (run / "config.yaml").resolve():
        raise GateError("reference config is not the run's exact config.yaml")
    config_hash = require_sha256(
        manifest.get("reference_config_sha256"), "reference config SHA-256"
    )
    if sha256_file(config) != config_hash:
        raise GateError("reference config bytes changed")
    if manifest.get("launch_time_reference_config_sha256") != config_hash:
        raise GateError("reference config lacks its launch-time binding")

    roles = manifest.get("checkpoint_roles")
    if not isinstance(roles, list) or [r.get("label") for r in roles] != list(ROLE_ORDER):
        raise GateError("checkpoint roles/order changed")
    normalized = []
    steps = []
    seen: set[Path] = set()
    for index, role in enumerate(roles):
        if not isinstance(role, dict):
            raise GateError(f"checkpoint role {index} is malformed")
        label = ROLE_ORDER[index]
        checkpoint = require_file(Path(str(role.get("checkpoint", ""))), f"{label} checkpoint")
        if checkpoint in seen:
            raise GateError("checkpoint roles must use distinct files")
        seen.add(checkpoint)
        digest = require_sha256(role.get("checkpoint_sha256"), f"{label} checkpoint hash")
        if sha256_file(checkpoint) != digest:
            raise GateError(f"{label} checkpoint bytes changed")
        step = role.get("checkpoint_step")
        if index == 0:
            if checkpoint != (run / "initial_policy.pt").resolve() or step is not None:
                raise GateError("baseline must be the iteration-zero initial_policy.pt")
            if manifest.get("launch_time_baseline_sha256") != digest:
                raise GateError("baseline lacks its launch-time hash binding")
        else:
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise GateError(f"{label} checkpoint step is invalid")
            if checkpoint.parent != run or checkpoint.name != f"model_step_{step:06d}.pt":
                raise GateError(f"{label} checkpoint path/step mismatch")
            steps.append(step)
        normalized.append(
            {"label": label, "checkpoint": str(checkpoint), "sha256": digest, "step": step}
        )
    if steps != sorted(set(steps)) or len(steps) != 3:
        raise GateError("early/mid/final steps are not strictly increasing")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(roles):
        raise GateError("manifest must contain exactly four checkpoint tasks")
    for index, (task, role) in enumerate(zip(tasks, roles, strict=True)):
        if not isinstance(task, dict) or task.get("task_index") != index:
            raise GateError(f"checkpoint task {index} identity mismatch")
        for key in ("label", "checkpoint", "checkpoint_step", "checkpoint_sha256"):
            if task.get(key) != role.get(key):
                raise GateError(f"checkpoint task {index} {key} differs from its role")
        if task.get("seeds") != list(expected_seeds):
            raise GateError(f"checkpoint task {index} seed list changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "reference_config": str(config),
        "reference_config_sha256": config_hash,
        "effective_contract_sha256": manifest["effective_contract_sha256"],
        "roles": normalized,
        "seeds": list(expected_seeds),
        "num_envs": expected_num_envs,
    }


def verify_camera_acceptance_binding(
    args: argparse.Namespace, base: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind visual camera acceptance to this exact source/asset preflight."""

    acceptance_hash = require_hash(
        args.camera_acceptance,
        args.camera_acceptance_sha256,
        "camera calibration acceptance",
    )
    verification = camera_gate.verify_acceptance(
        args.camera_acceptance,
        args.fast_revision,
        args.gr00t_revision,
        expected_acceptance_sha256=acceptance_hash,
        expected_fast_source_tree_sha256=base[
            "fast_roborender_source_tree_sha256"
        ],
        expected_gr00t_source_tree_sha256=base["gr00t_source_tree_sha256"],
        expected_source_snapshot_attestation_sha256=base[
            "source_snapshot_attestation_sha256"
        ],
        expected_source_snapshot_inventory_sha256=base[
            "source_snapshot_inventory_sha256"
        ],
        expected_asset_contract_sha256=base["asset_contract_sha256"],
    )
    expected = {
        "schema": camera_gate.VERIFICATION_SCHEMA,
        "status": "verified",
        "acceptance_sha256": acceptance_hash,
        "fast_roborender_revision": args.fast_revision,
        "gr00t_revision": args.gr00t_revision,
        "fast_roborender_source_tree_sha256": base[
            "fast_roborender_source_tree_sha256"
        ],
        "gr00t_source_tree_sha256": base["gr00t_source_tree_sha256"],
        "source_snapshot_attestation_sha256": base[
            "source_snapshot_attestation_sha256"
        ],
        "source_snapshot_inventory_sha256": base[
            "source_snapshot_inventory_sha256"
        ],
        "immutable_asset_contract_sha256": base["asset_contract_sha256"],
        "policy_views": list(VIEWS),
        "panels_verified": 2,
    }
    for key, value in expected.items():
        if verification.get(key) != value:
            raise GateError(f"camera calibration verification {key} mismatch")
    return verification


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    expected_seeds = parse_seeds(args.expected_seeds)
    fixed = validate_manifest(
        args.fixed_seed_manifest,
        train_run_dir=args.train_run_dir,
        expected_seeds=expected_seeds,
        expected_num_envs=args.expected_num_envs,
    )
    if fixed["sha256"] != require_sha256(
        args.fixed_seed_manifest_sha256, "expected manifest SHA-256"
    ):
        raise GateError("fixed-seed manifest changed before preflight")

    smoke_args = argparse.Namespace(
        profile="image3f_fixed_seed_production_eval",
        root=args.root,
        gr00t_root=args.gr00t_root,
        fast_revision=args.fast_revision,
        gr00t_revision=args.gr00t_revision,
        renderer_container=args.renderer_container,
        renderer_sha256=args.renderer_container_sha256,
        trainer_container=args.trainer_container,
        trainer_sha256=args.trainer_container_sha256,
        image_root=args.image_root,
        model_root=args.model_root,
        generated_root=args.generated_root,
        anchors=args.anchors,
        anchor_min_files=args.anchor_min_files,
        teacher=args.teacher,
        teacher_sha256=args.teacher_sha256,
        release_id=args.release_id,
        adapter_sha256=args.adapter_sha256,
    )
    base = smoke_gate.preflight(smoke_args)
    anchor_inventory, anchor_count = inventory_sha256(args.anchors)
    if anchor_count != base["anchor_count"]:
        raise GateError("anchor inventory count changed during preflight")
    camera_verification = verify_camera_acceptance_binding(args, base)
    camera_hash = camera_verification["acceptance_sha256"]
    resnet_hash = require_hash(args.resnet18, args.resnet18_sha256, "ResNet18 weights")
    wandb_path = require_file(args.wandb_acceptance, "authorized-team W&B acceptance")
    wandb_hash = require_hash(
        wandb_path,
        args.wandb_acceptance_sha256,
        "authorized-team W&B acceptance",
    )
    try:
        wandb_acceptance = team_contract.validate_online_acceptance(
            team_contract.load_json(wandb_path)
        )
    except team_contract.TeamContractError as exc:
        raise GateError(str(exc)) from None
    explicit_sources = {}
    for relative in (
        "scripts/eval/eval_agent_metrics_adapter.py",
        "scripts/eval/fixed_seed_gate.py",
        "scripts/eval/hsg_gb200_image3f_fixed_seed_gate.py",
        "scripts/eval/hsg_gb200_image3f_fixed_seed_policy_eval.sbatch",
        "scripts/eval/hsg_gb200_image3f_fixed_seed_task.sh",
        "scripts/eval/hsg_gb200_image3f_fixed_seed_wandb.py",
        "scripts/eval/wandb_team_contract.py",
        "scripts/scaling/camera_calibration_gate.py",
        "scripts/scaling/image3f_probe.py",
        "scripts/server/image3f_server.py",
        "scripts/server/image3f_server_array.sbatch",
    ):
        source = require_file(args.root / relative, f"source {relative}")
        explicit_sources[relative] = sha256_file(source)
    for relative in (
        "gr00t/rl/eval_agent_trl.py",
        "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
        "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
        "gr00t/rl/utils/image3f_frame_client.py",
    ):
        source = require_file(args.gr00t_root / relative, f"GR00T source {relative}")
        explicit_sources[f"GR00T-VisualSim2Real/{relative}"] = sha256_file(source)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "validated",
        "observation_mode": OBSERVATION_MODE,
        "image3f_in_the_loop": True,
        "production_observation_distribution": True,
        "runtime_contract": RUNTIME_CONTRACT,
        "topology": SEQUENTIAL_TOPOLOGY,
        "fixed_seed": fixed,
        "fast_roborender_revision": base["fast_roborender_revision"],
        "fast_roborender_source_tree_sha256": base[
            "fast_roborender_source_tree_sha256"
        ],
        "gr00t_revision": base["gr00t_revision"],
        "gr00t_source_tree_sha256": base["gr00t_source_tree_sha256"],
        "renderer_container_sha256": base["renderer_container_sha256"],
        "trainer_container_sha256": base["trainer_container_sha256"],
        "release_id": base["release_id"],
        "adapter_sha256": base["adapter_sha256"],
        "deployment_manifest_sha256": base["deployment_manifest_sha256"],
        "foundation_contract_sha256": base["foundation_contract_sha256"],
        "engine_manifest_sha256": base["engine_manifest_sha256"],
        "engine_sha256": base["engine_sha256"],
        "teacher_sha256": base["teacher_sha256"],
        "immutable_asset_contract_sha256": base["asset_contract_sha256"],
        "foundation": base["foundation"],
        "anchor_count": anchor_count,
        "anchor_inventory_sha256": anchor_inventory,
        "validated_anchor_inventory_sha256": base["anchor_inventory_sha256"],
        "camera_calibration_acceptance_sha256": camera_hash,
        "camera_calibration_verification": camera_verification,
        "camera_calibration_verification_sha256": canonical_sha256(
            camera_verification
        ),
        "resnet18_sha256": resnet_hash,
        "wandb_online_acceptance": {
            "path": str(wandb_path),
            "sha256": wandb_hash,
            **wandb_acceptance,
        },
        "explicit_source_sha256": explicit_sources,
    }


def _selected_endpoints(report: Mapping[str, Any], expected: int) -> list[dict[str, Any]]:
    selected = report.get("selected")
    if not isinstance(selected, list) or len(selected) != expected:
        raise GateError(f"renderer report does not contain exactly {expected} selected endpoints")
    by_file = {}
    for row in selected:
        if not isinstance(row, dict):
            raise GateError("renderer report selected row is malformed")
        path = str(row.get("file", ""))
        if not path or path in by_file:
            raise GateError("renderer report has a missing/duplicate endpoint path")
        by_file[path] = row
    return [by_file[path] for path in sorted(by_file)]


def _finite_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GateError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise GateError(f"{label} must be an integer") from None
    if result < 0 or float(value) != result:
        raise GateError(f"{label} must be a nonnegative integer")
    return result


def attest(args: argparse.Namespace) -> dict[str, Any]:
    if SAFE_ID_RE.fullmatch(args.run_id) is None:
        raise GateError("Image3F run/stream ID is malformed")
    before_path = require_file(args.before, "renderer before-snapshot")
    after_path = require_file(args.after, "renderer after-snapshot")
    metrics_path = require_file(args.metrics, "evaluation metrics")
    before = load_json(before_path)
    after = load_json(after_path)
    before_rows = _selected_endpoints(before, args.expected_renderers)
    after_rows = _selected_endpoints(after, args.expected_renderers)
    if [row["file"] for row in before_rows] != [row["file"] for row in after_rows]:
        raise GateError("selected endpoint fleet changed during evaluation")
    endpoint_manifest = require_file(args.endpoint_manifest, "selected endpoint manifest")
    selected_paths = [
        line.strip()
        for line in endpoint_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if selected_paths != [row["file"] for row in before_rows]:
        raise GateError("renderer snapshots differ from the selected endpoint manifest")

    request_deltas = []
    server_ids = []
    identity_hashes = []
    runtime_identity: dict[str, Any] | None = None
    for old, new in zip(before_rows, after_rows, strict=True):
        old_response = old.get("response")
        new_response = new.get("response")
        if not isinstance(old_response, dict) or not isinstance(new_response, dict):
            raise GateError("renderer snapshot response is malformed")
        old_identity = old_response.get("runtime_identity")
        new_identity = new_response.get("runtime_identity")
        if not isinstance(old_identity, dict) or old_identity != new_identity:
            raise GateError("renderer runtime identity changed during evaluation")
        if runtime_identity is None:
            runtime_identity = old_identity
        identity_hashes.append(canonical_sha256(old_identity))
        old_id, new_id = old_response.get("server_id"), new_response.get("server_id")
        if not isinstance(old_id, str) or old_id != new_id:
            raise GateError("renderer server identity changed during evaluation")
        server_ids.append(old_id)
        old_metrics, new_metrics = old_response.get("metrics"), new_response.get("metrics")
        if not isinstance(old_metrics, dict) or not isinstance(new_metrics, dict):
            raise GateError("renderer metrics are missing")
        old_requests = _finite_int(old_metrics.get("requests"), "before requests")
        new_requests = _finite_int(new_metrics.get("requests"), "after requests")
        old_errors = _finite_int(old_metrics.get("errors"), "before errors")
        new_errors = _finite_int(new_metrics.get("errors"), "after errors")
        old_replays = _finite_int(old_metrics.get("replays"), "before replays")
        new_replays = _finite_int(new_metrics.get("replays"), "after replays")
        if old_errors != 0 or new_errors != 0 or old_replays != 0 or new_replays != 0:
            raise GateError("renderer fleet recorded errors or lag-0 replays")
        delta = new_requests - old_requests
        if delta <= 0:
            raise GateError("every selected renderer must serve this seed block")
        request_deltas.append(delta)
    if len(set(server_ids)) != args.expected_renderers:
        raise GateError("renderer server IDs are not unique")
    if len(set(identity_hashes)) != 1:
        raise GateError("renderer fleet does not share one exact runtime identity")
    if runtime_identity is None:
        raise GateError("renderer runtime identity is missing")
    engine = runtime_identity.get("engine")
    generation_identity = runtime_identity.get("generation")
    if not isinstance(engine, dict) or not isinstance(generation_identity, dict):
        raise GateError("renderer runtime engine/generation identity is missing")
    runtime_engine_sha256 = require_sha256(
        engine.get("sha256"), "runtime TensorRT engine SHA-256"
    )
    runtime_adapter_sha256 = require_sha256(
        runtime_identity.get("adapter_sha256"), "runtime adapter SHA-256"
    )
    total_requests = sum(request_deltas)
    if total_requests < args.num_envs or total_requests % args.num_envs != 0:
        raise GateError("served request count is incompatible with all-environment relabeling")

    metrics = load_json(metrics_path)
    if _finite_int(metrics.get("collected_episodes"), "collected episodes") != args.num_envs:
        raise GateError("evaluation did not collect exactly one episode per environment")
    adapter = metrics.get("_roborender_eval_adapter")
    if not isinstance(adapter, dict) or adapter.get("schema_version") != 1:
        raise GateError("evaluation metrics lack the fail-closed evaluator adapter marker")
    runtime = metrics.get("_roborender_image3f_eval")
    if not isinstance(runtime, dict):
        raise GateError("evaluation metrics lack Image3F runtime evidence")
    expected_runtime = {
        "schema": "fast_roborender.image3f_fixed_eval_runtime.v1",
        "eval_relabel_enabled": True,
        "raytraced_mix_prob": 0.0,
        "reference_lag": 0,
        "generation_seed": args.seed,
        "policy_views": list(VIEWS),
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            raise GateError(f"Image3F runtime evidence {key} mismatch")
    generated_steps = _finite_int(runtime.get("generated_policy_steps"), "generated steps")
    generated_frames = _finite_int(runtime.get("generated_frames"), "generated frames")
    if generated_steps <= 0 or generated_frames != generated_steps * args.num_envs:
        raise GateError("not every policy observation was Image3F generated")
    if generated_frames != total_requests:
        raise GateError("client generated-frame count does not match renderer served requests")

    expected_env_keys = {str(i) for i in range(args.num_envs)}
    conditions = runtime.get("first_condition_sha256_by_env")
    generated = runtime.get("first_generated_sha256_by_env")
    for label, mapping in (("condition", conditions), ("generated", generated)):
        if not isinstance(mapping, dict) or set(mapping) != expected_env_keys:
            raise GateError(f"first {label} evidence does not cover every environment")
        for digest in mapping.values():
            require_sha256(digest, f"first {label} hash")

    episodes = metrics.get("_roborender_fixed_seed_episodes")
    if not isinstance(episodes, list) or len(episodes) != args.num_envs:
        raise GateError("per-environment episode evidence is incomplete")
    normalized_episodes = []
    seen_envs = set()
    for record in episodes:
        if not isinstance(record, dict):
            raise GateError("per-environment episode record is malformed")
        env_index = _finite_int(record.get("env_index"), "episode env_index")
        if env_index >= args.num_envs or env_index in seen_envs:
            raise GateError("episode environment identities are duplicate/out of range")
        seen_envs.add(env_index)
        key = str(env_index)
        if record.get("seed") != args.seed:
            raise GateError("episode seed does not match the fixed block")
        if record.get("first_condition_sha256") != conditions[key]:
            raise GateError("episode scenario condition hash mismatch")
        if record.get("first_generated_sha256") != generated[key]:
            raise GateError("episode generated observation hash mismatch")
        require_sha256(record.get("scenario_id"), "paired scenario ID")
        if not isinstance(record.get("success"), bool):
            raise GateError("per-environment success must be boolean")
        if _finite_int(record.get("episode_length"), "episode length") <= 0:
            raise GateError("episode length must be positive")
        reward = record.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise GateError("episode reward must be finite")
        normalized_episodes.append(record)
    normalized_episodes.sort(key=lambda row: row["env_index"])
    if seen_envs != set(range(args.num_envs)):
        raise GateError("episode evidence does not cover env indices 0..N-1")
    per_env_successes = sum(1 for row in normalized_episodes if row["success"])
    success_rate = metrics.get("goal_reached_rate")
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not math.isfinite(float(success_rate))
        or not math.isclose(
            float(success_rate), per_env_successes / args.num_envs, abs_tol=1e-9
        )
    ):
        raise GateError("aggregate success rate differs from per-environment outcomes")

    report = {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS",
        "checkpoint_label": args.label,
        "seed": args.seed,
        "run_id": args.run_id,
        "observation_mode": OBSERVATION_MODE,
        "image3f_in_the_loop": True,
        "production_observation_distribution": True,
        "renderer_count": args.expected_renderers,
        "server_ids": server_ids,
        "renderer_runtime_identity_sha256": identity_hashes[0],
        "endpoint_manifest_sha256": sha256_file(endpoint_manifest),
        "release_id": runtime_identity.get("release_id"),
        "adapter_sha256": runtime_adapter_sha256,
        "engine_sha256": runtime_engine_sha256,
        "generation_identity": generation_identity,
        "requests_per_renderer": request_deltas,
        "served_requests": total_requests,
        "renderer_errors": 0,
        "renderer_replays": 0,
        "generated_policy_steps": generated_steps,
        "generated_frames": generated_frames,
        "scenario_ids": [row["scenario_id"] for row in normalized_episodes],
        "first_condition_sha256_by_env": conditions,
        "first_generated_sha256_by_env": generated,
        "episode_outcomes": normalized_episodes,
        "per_env_successes": per_env_successes,
        "before_snapshot_sha256": sha256_file(before_path),
        "after_snapshot_sha256": sha256_file(after_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    atomic_json(args.output, report)
    return report


def _without_created_at(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def postrun(args: argparse.Namespace) -> dict[str, Any]:
    preflight_path = require_file(args.preflight, "Image3F evaluation preflight")
    require_hash(preflight_path, args.preflight_sha256, "Image3F evaluation preflight")
    preflight_payload = load_json(preflight_path)
    expected_preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "validated",
        "observation_mode": OBSERVATION_MODE,
        "image3f_in_the_loop": True,
        "production_observation_distribution": True,
        "runtime_contract": RUNTIME_CONTRACT,
        "topology": SEQUENTIAL_TOPOLOGY,
    }
    for key, value in expected_preflight.items():
        if preflight_payload.get(key) != value:
            raise GateError(f"preflight {key} contract changed")
    preflight_wandb = preflight_payload.get("wandb_online_acceptance")
    if not isinstance(preflight_wandb, dict):
        raise GateError("Image3F preflight lacks its authorized-team W&B binding")
    try:
        team_contract.validate_online_acceptance(preflight_wandb)
        team_contract.validate_authorized_request(
            expected_email=args.expected_wandb_email,
            entity=args.expected_wandb_entity,
            authorization_scope=team_contract.AUTHORIZED_SCOPE,
        )
    except team_contract.TeamContractError as exc:
        raise GateError(str(exc)) from None
    preflight_wandb_hash = require_sha256(
        preflight_wandb.get("sha256"),
        "preflight W&B online acceptance SHA-256",
    )
    manifest_path = require_file(args.fixed_seed_manifest, "fixed-seed manifest")
    manifest_hash = require_hash(
        manifest_path, args.fixed_seed_manifest_sha256, "fixed-seed manifest"
    )
    manifest = load_json(manifest_path)
    fixed_binding = preflight_payload.get("fixed_seed")
    if not isinstance(fixed_binding, dict) or fixed_binding.get("sha256") != manifest_hash:
        raise GateError("postrun manifest differs from the preflight binding")

    summary_path = require_file(args.summary, "fixed-seed summary")
    summary = load_json(summary_path)
    if summary.get("gate", {}).get("status") != "PASS":
        raise GateError("paired policy-improvement gate did not pass")
    recomputed = fixed_seed_gate.summarize_results(
        manifest=manifest,
        results_dir=args.results_dir,
        min_success_delta=args.min_success_delta,
    )
    if _without_created_at(summary) != _without_created_at(recomputed):
        raise GateError("fixed-seed summary differs from recomputed results")

    expected_count = len(ROLE_ORDER) * len(manifest["seeds"])
    attestation_paths = sorted(args.results_dir.glob("*/seed_*/image3f_attestation.json"))
    result_paths = sorted(args.results_dir.glob("*/seed_*/result.json"))
    if len(attestation_paths) != expected_count or len(result_paths) != expected_count:
        raise GateError("fixed-seed result/attestation inventory is incomplete")
    attestations: dict[tuple[str, int], dict[str, Any]] = {}
    identity_hashes = set()
    endpoint_manifest_hashes = set()
    stream_ids = set()
    warm_fleet_server_ids = None
    total_requests = 0
    for path in attestation_paths:
        row = load_json(path)
        if row.get("schema") != ATTESTATION_SCHEMA or row.get("status") != "PASS":
            raise GateError(f"runtime attestation is not PASS: {path}")
        label, seed = row.get("checkpoint_label"), row.get("seed")
        if label not in ROLE_ORDER or seed not in manifest["seeds"]:
            raise GateError(f"runtime attestation identity mismatch: {path}")
        key = (label, int(seed))
        if key in attestations:
            raise GateError("duplicate runtime attestation identity")
        if row.get("image3f_in_the_loop") is not True or row.get("renderer_errors") != 0:
            raise GateError("runtime attestation does not prove a clean Image3F loop")
        if row.get("generated_frames") != row.get("served_requests"):
            raise GateError("runtime client/server frame count mismatch")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or SAFE_ID_RE.fullmatch(run_id) is None:
            raise GateError("runtime stream ID is missing or malformed")
        if run_id in stream_ids:
            raise GateError("runtime stream IDs must be unique per role/seed")
        stream_ids.add(run_id)
        server_ids = row.get("server_ids")
        if warm_fleet_server_ids is None:
            warm_fleet_server_ids = server_ids
        elif server_ids != warm_fleet_server_ids:
            raise GateError("checkpoint/seed blocks did not share the same warm fleet")
        if row.get("release_id") != preflight_payload.get("release_id"):
            raise GateError("runtime Image3F release differs from preflight")
        if row.get("adapter_sha256") != preflight_payload.get("adapter_sha256"):
            raise GateError("runtime adapter differs from preflight")
        if row.get("engine_sha256") != preflight_payload.get("engine_sha256"):
            raise GateError("runtime TensorRT engine differs from preflight")
        generation_identity = row.get("generation_identity")
        if not isinstance(generation_identity, dict):
            raise GateError("runtime generation identity is missing")
        generation_expected = {
            "steps": RUNTIME_CONTRACT["inference_steps"],
            "tea_cache_threshold": RUNTIME_CONTRACT["tea_cache_threshold"],
            "cfg_scale": RUNTIME_CONTRACT["cfg_scale"],
            "cuda_graph": RUNTIME_CONTRACT["cuda_graph"],
        }
        for generation_key, generation_value in generation_expected.items():
            if generation_identity.get(generation_key) != generation_value:
                raise GateError(
                    f"runtime generation identity {generation_key} differs from preflight"
                )
        identity_hashes.add(row.get("renderer_runtime_identity_sha256"))
        endpoint_manifest_hashes.add(row.get("endpoint_manifest_sha256"))
        total_requests += _finite_int(row.get("served_requests"), "served requests")
        metrics_path = require_file(path.parent / "metrics_eval.json", "seed metrics")
        if sha256_file(metrics_path) != row.get("metrics_sha256"):
            raise GateError("runtime attestation is not bound to its seed metrics")
        result_path = require_file(path.parent / "result.json", "seed result")
        result = load_json(result_path)
        if result.get("checkpoint_label") != label or result.get("seed") != seed:
            raise GateError("seed result identity differs from its runtime attestation")
        raw_metrics = result.get("raw_metrics")
        if not isinstance(raw_metrics, dict):
            raise GateError("seed result lacks raw evaluator metrics")
        raw_runtime = raw_metrics.get("_roborender_image3f_eval")
        raw_episodes = raw_metrics.get("_roborender_fixed_seed_episodes")
        if not isinstance(raw_runtime, dict) or not isinstance(raw_episodes, list):
            raise GateError("seed result lacks Image3F/per-env runtime evidence")
        if raw_runtime.get("generated_frames") != row.get("generated_frames"):
            raise GateError("seed result Image3F frame count differs from attestation")
        if [record.get("scenario_id") for record in raw_episodes] != row.get("scenario_ids"):
            raise GateError("seed result scenario identities differ from attestation")
        attestations[key] = row
    if len(identity_hashes) != 1 or None in identity_hashes:
        raise GateError("seed blocks did not use one exact renderer runtime identity")
    if len(endpoint_manifest_hashes) != 1 or None in endpoint_manifest_hashes:
        raise GateError("seed blocks did not use one exact selected endpoint manifest")

    # Every role must begin from the exact same condition and deterministic
    # generated first observation for each seed/environment pair.
    paired_outcome_transitions = {
        "baseline_fail_final_fail": 0,
        "baseline_fail_final_success": 0,
        "baseline_success_final_fail": 0,
        "baseline_success_final_success": 0,
    }
    for seed in manifest["seeds"]:
        baseline = attestations[("baseline", int(seed))]
        for label in ROLE_ORDER[1:]:
            candidate = attestations[(label, int(seed))]
            for field in (
                "scenario_ids",
                "first_condition_sha256_by_env",
                "first_generated_sha256_by_env",
            ):
                if candidate.get(field) != baseline.get(field):
                    raise GateError(
                        f"paired reset/Image3F evidence differs for seed {seed}, role {label}: {field}"
                    )
        final = attestations[("final", int(seed))]
        baseline_outcomes = baseline.get("episode_outcomes")
        final_outcomes = final.get("episode_outcomes")
        if not isinstance(baseline_outcomes, list) or not isinstance(final_outcomes, list):
            raise GateError("paired per-environment outcomes are missing")
        if len(baseline_outcomes) != len(final_outcomes):
            raise GateError("paired per-environment outcome counts differ")
        for before, after in zip(baseline_outcomes, final_outcomes, strict=True):
            before_success = bool(before.get("success"))
            after_success = bool(after.get("success"))
            transition = (
                f"baseline_{'success' if before_success else 'fail'}_"
                f"final_{'success' if after_success else 'fail'}"
            )
            paired_outcome_transitions[transition] += 1

    wandb_path = require_file(args.wandb_upload, "authorized-team W&B upload marker")
    wandb = load_json(wandb_path)
    expected_wandb = {
        "schema": WANDB_SCHEMA,
        "status": "PASS",
        **team_contract.team_binding(),
        "manifest_sha256": manifest_hash,
        "summary_sha256": sha256_file(summary_path),
        "wandb_online_acceptance_sha256": preflight_wandb_hash,
        "wandb_online_acceptance_observed_keys": sorted(
            team_contract.REQUIRED_OBSERVED_KEYS
        ),
    }
    for key, value in expected_wandb.items():
        if wandb.get(key) != value:
            raise GateError(f"W&B readback marker {key} mismatch")
    if wandb.get("observed_checkpoint_ordinals") != [0, 1, 2, 3]:
        raise GateError("W&B readback lacks all four checkpoints")
    required_history = {
        "eval/checkpoint_ordinal",
        "eval/checkpoint_label",
        "eval/goal_reached_rate",
        "eval/reward",
        "eval/episodes",
        "image3f/served_requests",
        "image3f/generated_frames",
    }
    if set(wandb.get("observed_history_keys", [])) != required_history:
        raise GateError("W&B readback lacks production Image3F scalar keys")
    required_summary = {
        "policy_improvement_gate/status",
        "policy_improvement_gate/paired_success_delta",
        "policy_improvement_gate/paired_success_delta_ci95_low",
        "policy_improvement_gate/paired_success_delta_ci95_high",
        "image3f_gate/status",
        "image3f_gate/total_served_requests",
    }
    if set(wandb.get("observed_summary_keys", [])) != required_summary:
        raise GateError("W&B readback lacks production Image3F summary keys")

    report = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "PASS",
        "scientific_gate": "paired_final_minus_baseline_goal_reached_rate",
        "observation_mode": OBSERVATION_MODE,
        "image3f_in_the_loop": True,
        "production_observation_distribution_validated": True,
        "readiness_scope": "production_image3f_three_view_policy_loop",
        "preflight": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "fixed_seed_manifest": str(manifest_path),
        "fixed_seed_manifest_sha256": manifest_hash,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "wandb_upload": str(wandb_path),
        "wandb_upload_sha256": sha256_file(wandb_path),
        "wandb_account_email": team_contract.AUTHORIZED_EMAIL,
        "wandb_entity": team_contract.AUTHORIZED_ENTITY,
        "wandb_authorization_scope": team_contract.AUTHORIZED_SCOPE,
        "wandb_authorization": team_contract.team_binding(),
        "wandb_online_acceptance_sha256": preflight_wandb_hash,
        "renderer_runtime_identity_sha256": next(iter(identity_hashes)),
        "endpoint_manifest_sha256": next(iter(endpoint_manifest_hashes)),
        "total_served_requests": total_requests,
        "renderer_errors": 0,
        "paired_environment_outcome_transitions": paired_outcome_transitions,
        "paired_success_delta": summary["gate"]["paired_success_delta"],
        "paired_success_delta_ci95_low": summary["gate"][
            "paired_success_delta_seed_block_bootstrap95_low"
        ],
        "results": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in result_paths
        ],
        "runtime_attestations": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in attestation_paths
        ],
    }
    atomic_json(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    create = sub.add_parser("prepare")
    create.add_argument("--train-run-dir", type=Path, required=True)
    create.add_argument("--baseline-checkpoint", type=Path, required=True)
    create.add_argument("--reference-config", type=Path, required=True)
    create.add_argument("--seeds", required=True)
    create.add_argument("--num-envs", type=int, required=True)
    create.add_argument("--expected-baseline-sha256", required=True)
    create.add_argument("--expected-reference-config-sha256", required=True)
    create.add_argument("--output", type=Path, required=True)

    pre = sub.add_parser("preflight")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--gr00t-root", type=Path, required=True)
    pre.add_argument("--fast-revision", required=True)
    pre.add_argument("--gr00t-revision", required=True)
    pre.add_argument("--renderer-container", type=Path, required=True)
    pre.add_argument("--renderer-container-sha256", required=True)
    pre.add_argument("--trainer-container", type=Path, required=True)
    pre.add_argument("--trainer-container-sha256", required=True)
    pre.add_argument("--image-root", type=Path, required=True)
    pre.add_argument("--model-root", type=Path, required=True)
    pre.add_argument("--generated-root", type=Path, required=True)
    pre.add_argument("--anchors", type=Path, required=True)
    pre.add_argument("--anchor-min-files", type=int, default=2048)
    pre.add_argument("--teacher", type=Path, required=True)
    pre.add_argument("--teacher-sha256", required=True)
    pre.add_argument("--release-id", required=True)
    pre.add_argument("--adapter-sha256", required=True)
    pre.add_argument("--camera-acceptance", type=Path, required=True)
    pre.add_argument("--camera-acceptance-sha256", required=True)
    pre.add_argument("--resnet18", type=Path, required=True)
    pre.add_argument("--resnet18-sha256", required=True)
    pre.add_argument("--wandb-acceptance", type=Path, required=True)
    pre.add_argument("--wandb-acceptance-sha256", required=True)
    pre.add_argument("--fixed-seed-manifest", type=Path, required=True)
    pre.add_argument("--fixed-seed-manifest-sha256", required=True)
    pre.add_argument("--train-run-dir", type=Path, required=True)
    pre.add_argument("--expected-seeds", required=True)
    pre.add_argument("--expected-num-envs", type=int, required=True)

    evidence = sub.add_parser("attest")
    evidence.add_argument("--before", type=Path, required=True)
    evidence.add_argument("--after", type=Path, required=True)
    evidence.add_argument("--metrics", type=Path, required=True)
    evidence.add_argument("--endpoint-manifest", type=Path, required=True)
    evidence.add_argument("--label", choices=ROLE_ORDER, required=True)
    evidence.add_argument("--seed", type=int, required=True)
    evidence.add_argument("--run-id", required=True)
    evidence.add_argument("--num-envs", type=int, required=True)
    evidence.add_argument("--expected-renderers", type=int, default=7)
    evidence.add_argument("--output", type=Path, required=True)

    post = sub.add_parser("postrun")
    post.add_argument("--preflight", type=Path, required=True)
    post.add_argument("--preflight-sha256", required=True)
    post.add_argument("--fixed-seed-manifest", type=Path, required=True)
    post.add_argument("--fixed-seed-manifest-sha256", required=True)
    post.add_argument("--results-dir", type=Path, required=True)
    post.add_argument("--summary", type=Path, required=True)
    post.add_argument("--min-success-delta", type=float, required=True)
    post.add_argument("--wandb-upload", type=Path, required=True)
    post.add_argument("--expected-wandb-email", required=True)
    post.add_argument("--expected-wandb-entity", required=True)
    post.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            payload = prepare(args)
        elif args.command == "preflight":
            payload = preflight(args)
        elif args.command == "attest":
            payload = attest(args)
        else:
            payload = postrun(args)
    except (
        GateError,
        fixed_seed_gate.GateError,
        smoke_gate.GateError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
