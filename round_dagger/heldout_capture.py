"""Immutable expert-labelled capture for the fixed 256-case held-out set.

The live Isaac/Image3F evaluator owns the tensors used by the policy.  This
module gives that evaluator a small, dependency-light publication surface and
then binds the resulting NPZ to a validated pass-2 scenario-capture artifact.
Neither an NPZ without semantic lineage nor a lineage without the exact NPZ is
admissible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from round_dagger.actor import denormalize_policy_rgb
from round_dagger.artifacts import load_strict_json, publish_immutable_json, sha256_file
from round_dagger.scenario_capture import (
    EXPECTED_ENVS_PER_SEED,
    EXPECTED_SEEDS,
    canonical_sha256,
    load_strict_json as load_capture_json,
    validate_pass_document,
)


RAW_SCHEMA = "fast_roborender.round_dagger.heldout_expert_capture.v1"
LINEAGE_SCHEMA = "fast_roborender.round_dagger.heldout_expert_lineage.v1"
AGGREGATE_SCHEMA = "fast_roborender.round_dagger.heldout_expert_aggregate.v1"
ACTION_DIM = 26
POLICY_VIEWS = ("head_color", "hand_left", "hand_right")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_ARRAYS = {
    "policy_rgb",
    "expert_actions",
    "student_actions",
    "behavior_actions",
    "expert_mask",
    "dones",
    "observation_is_raytraced",
    "global_env_ids",
    "episode_ids",
    "episode_steps",
    "obs__actor_obs",
    "teacher_obs",
}


class HeldoutCaptureError(RuntimeError):
    """A raw held-out capture or its semantic binding is invalid."""


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(np.asarray(value))


def _absolute_regular(path: str | os.PathLike[str], label: str) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        info = os.lstat(target)
    except OSError as exc:
        raise HeldoutCaptureError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise HeldoutCaptureError(f"{label} must be a regular non-symlink file")
    return target


def _digest_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _publish_npz_create_only(path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]) -> str:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not target.is_absolute():  # pragma: no cover - abspath is defensive
        raise HeldoutCaptureError("capture output must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise HeldoutCaptureError(f"refusing to replace capture output: {target}")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fchmod(handle.fileno(), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise HeldoutCaptureError(f"capture output appeared concurrently: {target}") from exc
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return sha256_file(target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture_arrays(
    observations: Mapping[str, Any],
    action_state: Mapping[str, Any],
    *,
    global_env_ids: Sequence[int],
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
    teacher_observations: Any | None = None,
) -> dict[str, np.ndarray]:
    """Convert the exact policy call into the canonical one-step replay schema."""

    if not isinstance(observations, Mapping) or "vision_obs" not in observations:
        raise HeldoutCaptureError("policy observations do not contain vision_obs")
    if not isinstance(action_state, Mapping):
        raise HeldoutCaptureError("policy action state is not a mapping")
    for key in ("gt_actions", "action_mean"):
        if key not in action_state:
            raise HeldoutCaptureError(f"policy action state lacks {key}")
    env_ids = np.asarray(tuple(global_env_ids), dtype=np.int64)
    if env_ids.shape != (EXPECTED_ENVS_PER_SEED,) or len(set(env_ids.tolist())) != len(env_ids):
        raise HeldoutCaptureError("held-out capture requires exactly 32 unique global case IDs")
    vision = _as_numpy(observations["vision_obs"])
    if vision.ndim != 4 or vision.shape[0] != len(env_ids) or vision.shape[-1] != 3:
        raise HeldoutCaptureError("vision_obs must have shape [32,H,W,3]")
    expert = _as_numpy(action_state["gt_actions"]).astype(np.float32, copy=False)
    student = _as_numpy(action_state["action_mean"]).astype(np.float32, copy=False)
    expected_action_shape = (len(env_ids), ACTION_DIM)
    if expert.shape != expected_action_shape or student.shape != expected_action_shape:
        raise HeldoutCaptureError(
            f"expert/student actions must have shape {expected_action_shape}, got "
            f"{expert.shape}/{student.shape}"
        )
    if not np.isfinite(expert).all() or not np.isfinite(student).all():
        raise HeldoutCaptureError("expert/student actions contain non-finite values")
    rgb = denormalize_policy_rgb(
        vision[None], image_mean=image_mean, image_std=image_std
    )
    arrays: dict[str, np.ndarray] = {
        "policy_rgb": rgb,
        "expert_actions": expert[None],
        "student_actions": student[None],
        "behavior_actions": student[None].copy(),
        "expert_mask": np.zeros((1, len(env_ids)), dtype=np.bool_),
        "dones": np.zeros((1, len(env_ids)), dtype=np.bool_),
        "observation_is_raytraced": np.zeros((1, len(env_ids)), dtype=np.bool_),
        "global_env_ids": env_ids[None],
        "episode_ids": env_ids[None].copy(),
        "episode_steps": np.zeros((1, len(env_ids)), dtype=np.int64),
    }
    for key in sorted(observations):
        if key == "vision_obs":
            continue
        value = _as_numpy(observations[key])
        if value.shape[0] != len(env_ids):
            raise HeldoutCaptureError(f"policy observation {key!r} has wrong batch dimension")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise HeldoutCaptureError(f"policy observation {key!r} is not finite numeric data")
        arrays[f"obs__{key}"] = value.astype(np.float32, copy=False)[None]
    if "obs__actor_obs" not in arrays:
        raise HeldoutCaptureError("held-out capture requires actor_obs")
    if teacher_observations is not None:
        teacher = _as_numpy(teacher_observations)
        if (
            teacher.shape[0] != len(env_ids)
            or not np.issubdtype(teacher.dtype, np.number)
            or not np.isfinite(teacher).all()
        ):
            raise HeldoutCaptureError(
                "teacher_obs must be finite numeric data with the 32-case batch"
            )
        # Teacher state is evidence for auditing expert labels, never a student
        # input; deliberately do not use the obs__ prefix consumed by learners.
        arrays["teacher_obs"] = teacher.astype(np.float32, copy=False)[None]
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def publish_raw_capture(
    output: str | os.PathLike[str],
    observations: Mapping[str, Any],
    action_state: Mapping[str, Any],
    *,
    global_env_ids: Sequence[int],
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
    teacher_observations: Any | None = None,
) -> dict[str, Any]:
    arrays = capture_arrays(
        observations,
        action_state,
        global_env_ids=global_env_ids,
        image_mean=image_mean,
        image_std=image_std,
        teacher_observations=teacher_observations,
    )
    file_sha = _publish_npz_create_only(output, arrays)
    return {
        "schema": RAW_SCHEMA,
        "path": str(Path(os.path.abspath(os.path.expanduser(os.fspath(output))))),
        "file_sha256": file_sha,
        "array_sha256": {key: _digest_array(value) for key, value in sorted(arrays.items())},
    }


def _load_capture(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) < _REQUIRED_ARRAYS:
                raise HeldoutCaptureError("capture NPZ lacks required replay arrays")
            if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in archive.files):
                raise HeldoutCaptureError("capture NPZ contains an unsafe array name")
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    except HeldoutCaptureError:
        raise
    except Exception as exc:
        raise HeldoutCaptureError(f"cannot load capture NPZ: {exc}") from exc
    batch = EXPECTED_ENVS_PER_SEED
    if arrays["policy_rgb"].ndim != 5 or arrays["policy_rgb"].shape[0] != 1 or arrays["policy_rgb"].shape[1] != batch or arrays["policy_rgb"].shape[-1] != 3 or arrays["policy_rgb"].dtype != np.uint8:
        raise HeldoutCaptureError("capture policy RGB does not have one 32-case uint8 step")
    for key in ("expert_actions", "student_actions", "behavior_actions"):
        if arrays[key].shape != (1, batch, ACTION_DIM) or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise HeldoutCaptureError(f"capture {key} contract mismatch")
    if not np.array_equal(arrays["behavior_actions"], arrays["student_actions"]):
        raise HeldoutCaptureError("held-out behavior actions must equal frozen student actions")
    for key in ("expert_mask", "dones", "observation_is_raytraced"):
        if arrays[key].shape != (1, batch) or arrays[key].dtype != np.bool_ or arrays[key].any():
            raise HeldoutCaptureError(f"capture {key} must be all-false [1,32]")
    for key in ("global_env_ids", "episode_ids", "episode_steps"):
        if arrays[key].shape != (1, batch) or arrays[key].dtype != np.int64:
            raise HeldoutCaptureError(f"capture {key} contract mismatch")
    if not np.array_equal(arrays["episode_ids"], arrays["global_env_ids"]) or arrays["episode_steps"].any():
        raise HeldoutCaptureError("capture episode identity is not one-step deterministic")
    obs_keys = sorted(key for key in arrays if key.startswith("obs__"))
    if "obs__actor_obs" not in obs_keys:
        raise HeldoutCaptureError("capture lacks actor_obs")
    for key in obs_keys:
        if arrays[key].shape[:2] != (1, batch) or not np.issubdtype(arrays[key].dtype, np.floating) or not np.isfinite(arrays[key]).all():
            raise HeldoutCaptureError(f"capture observation {key} contract mismatch")
    teacher = arrays["teacher_obs"]
    if (
        teacher.shape[:2] != (1, batch)
        or teacher.dtype != np.float32
        or not np.isfinite(teacher).all()
    ):
        raise HeldoutCaptureError("capture teacher_obs contract mismatch")
    return arrays


def bind_capture_lineage(
    *,
    plan_path: str | os.PathLike[str],
    pass_artifact_path: str | os.PathLike[str],
    capture_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind one pass-2 expert capture to its exact 32 planned cases."""

    plan_file = _absolute_regular(plan_path, "held-out plan")
    pass_file = _absolute_regular(pass_artifact_path, "scenario pass artifact")
    capture_file = _absolute_regular(capture_path, "expert capture")
    plan, plan_payload = load_capture_json(plan_file, "held-out plan")
    passed, _ = load_capture_json(pass_file, "scenario pass artifact")
    plan_sha = hashlib.sha256(plan_payload).hexdigest()
    passed = validate_pass_document(passed, plan=plan, plan_sha256=plan_sha, production=True)
    runtime = passed["runtime"]
    if runtime["pass_index"] != 2 or not runtime["release_eligible"]:
        raise HeldoutCaptureError("expert capture must come from release-eligible replay pass 2")
    arrays = _load_capture(capture_file)
    cases = runtime["cases"]
    expected_ids = [case["global_case_index"] for case in cases]
    if arrays["global_env_ids"][0].tolist() != expected_ids:
        raise HeldoutCaptureError("capture global IDs differ from pass-2 planned ordering")
    rows = []
    obs_keys = sorted(key for key in arrays if key.startswith("obs__"))
    for offset, case in enumerate(cases):
        rows.append(
            {
                **case,
                "policy_rgb_sha256": _digest_array(arrays["policy_rgb"][0, offset]),
                "expert_actions_sha256": _digest_array(arrays["expert_actions"][0, offset]),
                "student_actions_sha256": _digest_array(arrays["student_actions"][0, offset]),
                "teacher_observations_sha256": _digest_array(
                    arrays["teacher_obs"][0, offset]
                ),
                "policy_observations_sha256": canonical_sha256(
                    {key: _digest_array(arrays[key][0, offset]) for key in obs_keys}
                ),
            }
        )
    document = {
        "schema": LINEAGE_SCHEMA,
        "seed": runtime["seed"],
        "case_count": len(rows),
        "policy_views": list(POLICY_VIEWS),
        "action_dim": ACTION_DIM,
        "observation_keys": obs_keys,
        "plan": {"path": str(plan_file), "sha256": plan_sha},
        "scenario_pass": {"path": str(pass_file), "sha256": sha256_file(pass_file)},
        "first_condition_bundle": dict(runtime["first_condition_bundle"]),
        "capture": {
            "path": str(capture_file),
            "sha256": sha256_file(capture_file),
            "array_sha256": {key: _digest_array(value) for key, value in sorted(arrays.items())},
        },
        "provenance": dict(runtime["provenance"]),
        "cases": rows,
    }
    document["lineage_sha256"] = canonical_sha256(document)
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
    file_sha = publish_immutable_json(output, document)
    return {**document, "path": str(output), "file_sha256": file_sha}


def verify_capture_lineage(path: str | os.PathLike[str]) -> dict[str, Any]:
    lineage_file = _absolute_regular(path, "expert capture lineage")
    document = load_strict_json(lineage_file)
    if not isinstance(document, Mapping) or document.get("schema") != LINEAGE_SCHEMA:
        raise HeldoutCaptureError("expert capture lineage schema mismatch")
    semantic = dict(document)
    declared = semantic.pop("lineage_sha256", None)
    if not isinstance(declared, str) or _SHA.fullmatch(declared) is None or canonical_sha256(semantic) != declared:
        raise HeldoutCaptureError("expert capture lineage semantic digest mismatch")
    rebuilt = bind_capture_lineage(
        plan_path=document["plan"]["path"],
        pass_artifact_path=document["scenario_pass"]["path"],
        capture_path=document["capture"]["path"],
        output_path=lineage_file,
    )
    comparable = dict(rebuilt)
    comparable.pop("path", None)
    comparable.pop("file_sha256", None)
    if comparable != dict(document):
        raise HeldoutCaptureError("expert capture lineage differs from source bytes")
    return {**dict(document), "path": str(lineage_file), "file_sha256": sha256_file(lineage_file)}


def build_capture_aggregate(
    lineage_paths: Sequence[str | os.PathLike[str]],
    *,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify and index exactly eight immutable 32-case expert captures."""

    if len(lineage_paths) != len(EXPECTED_SEEDS):
        raise HeldoutCaptureError("held-out aggregate requires exactly eight lineage files")
    lineages = [verify_capture_lineage(path) for path in lineage_paths]
    lineages.sort(key=lambda item: item["seed"])
    if [item["seed"] for item in lineages] != list(EXPECTED_SEEDS):
        raise HeldoutCaptureError("held-out aggregate seed coverage is not exact")
    plan_shas = {item["plan"]["sha256"] for item in lineages}
    provenance = {canonical_sha256(item["provenance"]) for item in lineages}
    if len(plan_shas) != 1 or len(provenance) != 1:
        raise HeldoutCaptureError("held-out aggregate source/provenance is inconsistent")
    cases = [case for item in lineages for case in item["cases"]]
    if [case["global_case_index"] for case in cases] != list(range(256)):
        raise HeldoutCaptureError("held-out aggregate does not cover global cases 0..255 exactly")
    document = {
        "schema": AGGREGATE_SCHEMA,
        "case_count": 256,
        "seed_count": 8,
        "seeds": list(EXPECTED_SEEDS),
        "plan_sha256": next(iter(plan_shas)),
        "lineages": [
            {
                "seed": item["seed"],
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "lineage_sha256": item["lineage_sha256"],
                "capture_sha256": item["capture"]["sha256"],
            }
            for item in lineages
        ],
        "case_inventory_sha256": canonical_sha256(cases),
    }
    document["aggregate_sha256"] = canonical_sha256(document)
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
    file_sha = publish_immutable_json(output, document)
    return {**document, "path": str(output), "file_sha256": file_sha}


def verify_capture_aggregate(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Recursively verify an aggregate and all eight source captures."""

    aggregate_file = _absolute_regular(path, "held-out expert aggregate")
    document = load_strict_json(aggregate_file)
    if not isinstance(document, Mapping) or document.get("schema") != AGGREGATE_SCHEMA:
        raise HeldoutCaptureError("held-out expert aggregate schema mismatch")
    semantic = dict(document)
    declared = semantic.pop("aggregate_sha256", None)
    if (
        not isinstance(declared, str)
        or _SHA.fullmatch(declared) is None
        or canonical_sha256(semantic) != declared
    ):
        raise HeldoutCaptureError("held-out expert aggregate semantic digest mismatch")
    rows = document.get("lineages")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_SEEDS):
        raise HeldoutCaptureError("held-out expert aggregate lineage inventory mismatch")
    rebuilt = build_capture_aggregate(
        [row["path"] for row in rows], output_path=aggregate_file
    )
    comparable = dict(rebuilt)
    comparable.pop("path", None)
    comparable.pop("file_sha256", None)
    if comparable != dict(document):
        raise HeldoutCaptureError("held-out expert aggregate differs from source bytes")
    return {
        **dict(document),
        "path": str(aggregate_file),
        "file_sha256": sha256_file(aggregate_file),
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--plan", required=True)
    bind.add_argument("--pass-artifact", required=True)
    bind.add_argument("--capture", required=True)
    bind.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--lineage", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--lineage", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    verify_aggregate = sub.add_parser("verify-aggregate")
    verify_aggregate.add_argument("--aggregate", required=True)
    args = parser.parse_args(argv)
    if args.command == "bind":
        result = bind_capture_lineage(
            plan_path=args.plan,
            pass_artifact_path=args.pass_artifact,
            capture_path=args.capture,
            output_path=args.output,
        )
    elif args.command == "verify":
        result = verify_capture_lineage(args.lineage)
    elif args.command == "aggregate":
        result = build_capture_aggregate(args.lineage, output_path=args.output)
    else:
        result = verify_capture_aggregate(args.aggregate)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "HeldoutCaptureError",
    "bind_capture_lineage",
    "build_capture_aggregate",
    "capture_arrays",
    "publish_raw_capture",
    "verify_capture_aggregate",
    "verify_capture_lineage",
]
