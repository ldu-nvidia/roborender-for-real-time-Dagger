#!/usr/bin/env python3
"""Prepare and score a fixed-seed, checkpoint-matched policy evaluation.

The production DAgger loss is an optimization diagnostic, not evidence that the
closed-loop student got better.  This utility builds a four-checkpoint
evaluation (baseline, early, mid, final), validates one result for every fixed
seed block, and computes a paired bootstrap interval for final-minus-baseline
success and reward.

Each seed block is intended to run with ``eval_num_envs_episodes=true``: every
Isaac environment contributes exactly one episode.  That keeps reset scenarios
matched across checkpoints and prevents a policy that terminates early from
advancing its reset RNG farther than another policy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
CHECKPOINT_RE = re.compile(r"^model_step_(\d+)\.pt$")
DEFAULT_SEEDS = (1729, 3253, 4519, 5987, 7411, 8863, 10193, 12203)
ROLE_ORDER = ("baseline", "early", "mid", "final")
EVALUATION_CHECKPOINT_SCHEMA = "fast_roborender.eval_checkpoint.v1"
EVALUATION_PROVENANCE_SCHEMA = "fast_roborender.eval_checkpoint_provenance.v1"
POLICY_STATE_KEYS = ("actor_model_state_dict", "policy_state_dict")
AUXILIARY_STATE_KEYS = ("value_state_dict", "homie_state_dict")
EVALUATION_STATE_KEYS = frozenset((*POLICY_STATE_KEYS, *AUXILIARY_STATE_KEYS))


class GateError(RuntimeError):
    """The evaluation contract is incomplete or malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from None
    if not isinstance(value, dict):
        raise GateError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_file(
    path: Path,
    *,
    expected_sha256: str | None,
    weights_only: bool,
    label: str,
) -> tuple[Mapping[str, Any], str, os.stat_result]:
    """Hash and load one checkpoint from the same race-checked descriptor.

    Untrusted staged checkpoints must use ``weights_only=True``.  A contracted
    source is hashed before its legacy pickle is allowed to execute.
    """

    try:
        import torch
    except ImportError:
        raise GateError("torch is required to inspect an evaluation checkpoint") from None
    candidate = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise GateError(f"cannot safely open {label}: {candidate}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GateError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                observed_bytes += len(chunk)
                digest.update(chunk)
                chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or observed_bytes != before.st_size:
            raise GateError(f"{label} changed while it was inspected")
        payload = b"".join(chunks)
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise GateError(f"{label} SHA-256 differs from the contract")
    finally:
        os.close(descriptor)
    # Deserialize only the immutable byte snapshot whose complete digest and
    # descriptor identity were accepted above.  Loading from the still-live
    # file descriptor would permit an in-place write between hash and pickle
    # execution, even if a later fstat detected the change.
    try:
        checkpoint = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=weights_only,
        )
    except TypeError as exc:
        if weights_only:
            raise GateError(
                "safe staged checkpoint loading requires PyTorch weights_only"
            ) from exc
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as exc:
        mode = "safe" if weights_only else "contracted"
        raise GateError(
            f"cannot {mode}-load {label}: {type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(checkpoint, Mapping):
        raise GateError(f"{label} must contain a checkpoint mapping")
    return checkpoint, actual_sha256, before


def minimal_evaluation_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Extract only model state consumed by the evaluation loader."""

    if any(not isinstance(key, str) for key in checkpoint):
        raise GateError("checkpoint top-level keys must be strings")
    policy_keys = [key for key in POLICY_STATE_KEYS if key in checkpoint]
    if len(policy_keys) != 1:
        raise GateError(
            "checkpoint must contain exactly one supported policy state dict"
        )
    retained = [policy_keys[0]]
    for key in AUXILIARY_STATE_KEYS:
        if checkpoint.get(key) is not None:
            retained.append(key)
    minimal = {key: checkpoint[key] for key in retained}
    validate_evaluation_checkpoint(minimal)
    retained.sort()
    removed = sorted(set(checkpoint) - set(retained))
    return minimal, retained, removed


def validate_evaluation_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Require the exact tensor-only structure accepted for evaluation."""

    try:
        import torch
    except ImportError:
        raise GateError("torch is required to inspect an evaluation checkpoint") from None
    keys = set(checkpoint)
    if any(not isinstance(key, str) for key in checkpoint) or not keys <= EVALUATION_STATE_KEYS:
        raise GateError("evaluation checkpoint contains an unsupported top-level key")
    if len(keys.intersection(POLICY_STATE_KEYS)) != 1:
        raise GateError("evaluation checkpoint policy state is missing or ambiguous")
    for section, state_dict in checkpoint.items():
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise GateError(f"evaluation checkpoint {section} must be a non-empty mapping")
        if any(not isinstance(name, str) or not name for name in state_dict):
            raise GateError(f"evaluation checkpoint {section} has an unsafe tensor name")
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise GateError(
                    f"evaluation checkpoint {section}.{name} is not a tensor"
                )
            if (
                tensor.layout != torch.strided
                or tensor.is_sparse
                or tensor.is_quantized
                or tensor.device.type == "meta"
            ):
                raise GateError(
                    f"evaluation checkpoint {section}.{name} has unsupported tensor storage"
                )


def _semantic_field(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def checkpoint_semantic_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Hash named tensor values independent of torch.save container bytes."""

    try:
        import torch
    except ImportError:
        raise GateError("torch is required to inspect an evaluation checkpoint") from None
    validate_evaluation_checkpoint(checkpoint)
    digest = hashlib.sha256()
    digest.update(EVALUATION_CHECKPOINT_SCHEMA.encode("ascii") + b"\0")
    for section in sorted(checkpoint):
        _semantic_field(digest, section.encode("utf-8"))
        state_dict = checkpoint[section]
        for name in sorted(state_dict):
            tensor = state_dict[name].detach().to(device="cpu").contiguous()
            _semantic_field(digest, name.encode("utf-8"))
            _semantic_field(digest, str(tensor.dtype).encode("ascii"))
            digest.update(struct.pack(">Q", tensor.ndim))
            for dimension in tensor.shape:
                digest.update(struct.pack(">q", int(dimension)))
            byte_view = tensor.reshape(-1).view(torch.uint8)
            digest.update(struct.pack(">Q", byte_view.numel()))
            digest.update(memoryview(byte_view.numpy()))
    return digest.hexdigest()


def sanitize_checkpoint(source: Path, output: Path, expected_sha256: str) -> dict[str, Any]:
    """Create a tensor-only evaluation checkpoint bound to the source semantics.

    GR00T checkpoints contain ``env_state_dict`` and its loader restores that state after
    Isaac has been seeded.  Leaving it in would make nominally fixed-seed comparisons use
    checkpoint-dependent simulator state.  All non-model state is omitted, making the
    result safe to independently load with ``weights_only=True``.
    """

    source = Path(os.path.abspath(source))
    checkpoint, actual_sha256, _ = load_checkpoint_file(
        source,
        expected_sha256=expected_sha256,
        weights_only=False,
        label="checkpoint source",
    )
    minimal, retained_keys, removed_keys = minimal_evaluation_checkpoint(checkpoint)
    semantic_sha256 = checkpoint_semantic_sha256(minimal)
    try:
        import torch
    except ImportError:
        raise GateError("torch is required to sanitize an evaluation checkpoint") from None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        torch.save(minimal, temporary)
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise GateError(f"cannot write sanitized checkpoint {output}: {exc}") from None
    return {
        "schema": EVALUATION_PROVENANCE_SCHEMA,
        "source": str(source),
        "source_sha256": actual_sha256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_semantic_sha256": semantic_sha256,
        "retained_keys": retained_keys,
        "removed_keys": removed_keys,
        "environment_state_restored": False,
    }


def parse_seed_list(raw: str) -> list[int]:
    try:
        seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    except ValueError:
        raise GateError("seeds must be a comma-separated list of integers") from None
    if len(seeds) < 2:
        raise GateError("at least two fixed seed blocks are required")
    if len(set(seeds)) != len(seeds):
        raise GateError("fixed seed blocks must be unique")
    if any(seed < 0 for seed in seeds):
        raise GateError("fixed seeds must be non-negative")
    return seeds


def discover_training_checkpoints(train_run_dir: Path) -> list[tuple[int, Path]]:
    snapshots: list[tuple[int, Path]] = []
    for candidate in train_run_dir.glob("model_step_*.pt"):
        match = CHECKPOINT_RE.fullmatch(candidate.name)
        if match and candidate.is_file():
            snapshots.append((int(match.group(1)), candidate.resolve()))
    snapshots.sort(key=lambda item: item[0])
    return snapshots


def select_checkpoint_roles(
    baseline_checkpoint: Path,
    train_run_dir: Path,
) -> list[dict[str, Any]]:
    baseline_checkpoint = baseline_checkpoint.resolve()
    if not baseline_checkpoint.is_file():
        raise GateError(f"baseline checkpoint does not exist: {baseline_checkpoint}")
    snapshots = discover_training_checkpoints(train_run_dir)
    if len(snapshots) < 3:
        raise GateError(
            "need at least three saved training snapshots for distinct early/mid/final "
            f"evaluation; found {len(snapshots)} in {train_run_dir}"
        )

    chosen = {
        "early": snapshots[0],
        "mid": snapshots[(len(snapshots) - 1) // 2],
        "final": snapshots[-1],
    }
    chosen_paths = [item[1] for item in chosen.values()]
    if len(set(chosen_paths)) != 3:
        raise GateError("early, mid, and final checkpoint selection was not distinct")

    roles: list[dict[str, Any]] = [
        {
            "label": "baseline",
            "checkpoint": str(baseline_checkpoint),
            "checkpoint_step": None,
            "checkpoint_sha256": sha256_file(baseline_checkpoint),
        }
    ]
    for label in ROLE_ORDER[1:]:
        step, checkpoint = chosen[label]
        roles.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "checkpoint_step": step,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return roles


def create_manifest(
    *,
    train_run_dir: Path,
    baseline_checkpoint: Path,
    reference_config: Path,
    seeds: Sequence[int],
    num_envs: int,
    output_path: Path,
    expected_baseline_sha256: str | None = None,
    expected_reference_config_sha256: str | None = None,
) -> dict[str, Any]:
    train_run_dir = train_run_dir.resolve()
    reference_config = reference_config.resolve()
    if not train_run_dir.is_dir():
        raise GateError(f"training run directory does not exist: {train_run_dir}")
    if not reference_config.is_file():
        raise GateError(f"reference config does not exist: {reference_config}")
    if num_envs < 1:
        raise GateError("num_envs must be positive")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise GateError("at least two unique fixed seeds are required")

    roles = select_checkpoint_roles(baseline_checkpoint, train_run_dir)
    reference_config_sha256 = sha256_file(reference_config)
    if (
        expected_baseline_sha256
        and roles[0]["checkpoint_sha256"] != expected_baseline_sha256
    ):
        raise GateError("baseline checkpoint changed after production submission")
    if (
        expected_reference_config_sha256
        and reference_config_sha256 != expected_reference_config_sha256
    ):
        raise GateError("reference config changed after production submission")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "protocol": "one_episode_per_environment_fixed_seed_blocks",
        "observation_mode": "raytraced",
        "train_run_dir": str(train_run_dir),
        "reference_config": str(reference_config),
        "reference_config_sha256": reference_config_sha256,
        "launch_time_baseline_sha256": expected_baseline_sha256,
        "launch_time_reference_config_sha256": expected_reference_config_sha256,
        "num_envs_per_seed": int(num_envs),
        "expected_episodes_per_seed": int(num_envs),
        "seeds": [int(seed) for seed in seeds],
        "checkpoint_roles": roles,
        "tasks": [
            {"task_index": index, **role, "seeds": [int(seed) for seed in seeds]}
            for index, role in enumerate(roles)
        ],
    }
    _atomic_json(output_path, payload)
    return payload


def _require_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"unsupported manifest schema: {payload.get('schema_version')!r}")
    labels = [role.get("label") for role in payload.get("checkpoint_roles", [])]
    if labels != list(ROLE_ORDER):
        raise GateError(f"checkpoint roles must be {ROLE_ORDER}, got {labels}")


def task_from_manifest(manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    _require_manifest(manifest)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not 0 <= index < len(tasks):
        raise GateError(f"task index {index} is outside the manifest")
    task = tasks[index]
    if not isinstance(task, dict):
        raise GateError(f"task {index} is malformed")
    return task


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GateError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise GateError(f"{name} must be numeric") from None
    if not math.isfinite(number):
        raise GateError(f"{name} must be finite")
    return number


def build_result(
    *,
    manifest: Mapping[str, Any],
    label: str,
    seed: int,
    metrics: Mapping[str, Any],
    evaluated_checkpoint_sha256: str | None = None,
    evaluated_config_sha256: str | None = None,
) -> dict[str, Any]:
    _require_manifest(manifest)
    role_by_label = {role["label"]: role for role in manifest["checkpoint_roles"]}
    if label not in role_by_label:
        raise GateError(f"unknown checkpoint label: {label}")
    if seed not in manifest["seeds"]:
        raise GateError(f"seed {seed} is not part of the fixed evaluation set")

    expected_episodes = int(manifest["expected_episodes_per_seed"])
    episodes_value = metrics.get("collected_episodes", metrics.get("completed_episodes"))
    episodes = int(_finite_number(episodes_value, "collected_episodes"))
    if episodes != expected_episodes:
        raise GateError(
            f"seed {seed} label {label} collected {episodes} episodes; "
            f"expected exactly {expected_episodes}"
        )

    success_rate = _finite_number(metrics.get("goal_reached_rate"), "goal_reached_rate")
    if not 0.0 <= success_rate <= 1.0:
        raise GateError("goal_reached_rate must be in [0, 1]")
    successes_float = success_rate * episodes
    successes = int(round(successes_float))
    if abs(successes_float - successes) > 1e-4:
        raise GateError(
            "goal_reached_rate is inconsistent with an integer number of successes "
            f"({success_rate} * {episodes})"
        )
    reward = _finite_number(metrics.get("reward"), "reward")
    episode_length = _finite_number(metrics.get("episode_length"), "episode_length")

    role = role_by_label[label]
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _utc_now(),
        "checkpoint_label": label,
        "checkpoint": role["checkpoint"],
        "checkpoint_step": role["checkpoint_step"],
        "checkpoint_sha256": role["checkpoint_sha256"],
        "reference_config_sha256": manifest["reference_config_sha256"],
        "evaluated_checkpoint_sha256": (
            evaluated_checkpoint_sha256 or role["checkpoint_sha256"]
        ),
        "evaluated_config_sha256": (
            evaluated_config_sha256 or manifest["reference_config_sha256"]
        ),
        "seed": int(seed),
        "num_envs": int(manifest["num_envs_per_seed"]),
        "episodes": episodes,
        "successes": successes,
        "goal_reached_rate": success_rate,
        "reward": reward,
        "episode_length": episode_length,
        "raw_metrics": dict(metrics),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise GateError("cannot compute a quantile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 910_241,
) -> tuple[float, float]:
    if len(values) < 2:
        raise GateError("at least two seed blocks are needed for a confidence interval")
    rng = random.Random(seed)
    count = len(values)
    sampled_means = []
    for _ in range(resamples):
        sampled_means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    sampled_means.sort()
    alpha = (1.0 - confidence) / 2.0
    return _quantile(sampled_means, alpha), _quantile(sampled_means, 1.0 - alpha)


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise GateError("invalid binomial counts for Wilson interval")
    rate = successes / episodes
    z2 = z * z
    denominator = 1.0 + z2 / episodes
    center = (rate + z2 / (2.0 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / episodes + z2 / (4.0 * episodes * episodes))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    total_episodes = sum(int(row["episodes"]) for row in rows)
    return sum(float(row[key]) * int(row["episodes"]) for row in rows) / total_episodes


def _checkpoint_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    rate = successes / episodes
    success_ci = wilson_interval(successes, episodes)
    reward_blocks = [float(row["reward"]) for row in rows]
    reward_ci = bootstrap_mean_ci(reward_blocks)
    length_blocks = [float(row["episode_length"]) for row in rows]
    length_ci = bootstrap_mean_ci(length_blocks)
    return {
        "checkpoint": rows[0]["checkpoint"],
        "checkpoint_step": rows[0]["checkpoint_step"],
        "seed_blocks": len(rows),
        "episodes": episodes,
        "successes": successes,
        "goal_reached_rate": rate,
        "goal_reached_rate_wilson95_low": success_ci[0],
        "goal_reached_rate_wilson95_high": success_ci[1],
        "reward": _weighted_mean(rows, "reward"),
        "reward_seed_block_bootstrap95_low": reward_ci[0],
        "reward_seed_block_bootstrap95_high": reward_ci[1],
        "episode_length": _weighted_mean(rows, "episode_length"),
        "episode_length_seed_block_bootstrap95_low": length_ci[0],
        "episode_length_seed_block_bootstrap95_high": length_ci[1],
    }


def load_complete_results(
    *, manifest: Mapping[str, Any], results_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    _require_manifest(manifest)
    expected_seeds = [int(seed) for seed in manifest["seeds"]]
    by_label: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for label in ROLE_ORDER:
        rows = []
        for seed in expected_seeds:
            path = results_dir / label / f"seed_{seed}" / "result.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            row = _load_json(path)
            if row.get("checkpoint_label") != label or row.get("seed") != seed:
                raise GateError(f"result identity mismatch in {path}")
            if row.get("reference_config_sha256") != manifest["reference_config_sha256"]:
                raise GateError(f"reference config hash mismatch in {path}")
            role = next(role for role in manifest["checkpoint_roles"] if role["label"] == label)
            if row.get("checkpoint_sha256") != role["checkpoint_sha256"]:
                raise GateError(f"checkpoint hash mismatch in {path}")
            if not row.get("evaluated_checkpoint_sha256"):
                raise GateError(f"missing evaluated checkpoint hash in {path}")
            if row.get("evaluated_config_sha256") != manifest["reference_config_sha256"]:
                raise GateError(f"evaluated config hash mismatch in {path}")
            if int(row.get("episodes", -1)) != int(manifest["expected_episodes_per_seed"]):
                raise GateError(f"episode count mismatch in {path}")
            rows.append(row)
        evaluated_hashes = {row["evaluated_checkpoint_sha256"] for row in rows}
        if len(evaluated_hashes) > 1:
            raise GateError(f"seed blocks for {label} used different staged checkpoints")
        by_label[label] = rows
    if missing:
        preview = ", ".join(missing[:4])
        suffix = " ..." if len(missing) > 4 else ""
        raise GateError(f"missing {len(missing)} fixed-seed result(s): {preview}{suffix}")
    return by_label


def summarize_results(
    *,
    manifest: Mapping[str, Any],
    results_dir: Path,
    min_success_delta: float,
) -> dict[str, Any]:
    if not 0.0 <= min_success_delta <= 1.0:
        raise GateError("min_success_delta must be in [0, 1]")
    rows = load_complete_results(manifest=manifest, results_dir=results_dir)
    summaries = {label: _checkpoint_summary(rows[label]) for label in ROLE_ORDER}

    baseline_by_seed = {int(row["seed"]): row for row in rows["baseline"]}
    final_by_seed = {int(row["seed"]): row for row in rows["final"]}
    seeds = [int(seed) for seed in manifest["seeds"]]
    paired_success_delta = [
        float(final_by_seed[seed]["goal_reached_rate"])
        - float(baseline_by_seed[seed]["goal_reached_rate"])
        for seed in seeds
    ]
    paired_reward_delta = [
        float(final_by_seed[seed]["reward"]) - float(baseline_by_seed[seed]["reward"])
        for seed in seeds
    ]
    success_delta = sum(paired_success_delta) / len(paired_success_delta)
    reward_delta = sum(paired_reward_delta) / len(paired_reward_delta)
    success_delta_ci = bootstrap_mean_ci(paired_success_delta, seed=710_039)
    reward_delta_ci = bootstrap_mean_ci(paired_reward_delta, seed=710_041)

    delta_large_enough = success_delta >= min_success_delta
    statistically_positive = success_delta_ci[0] > 0.0
    passed = delta_large_enough and statistically_positive
    reason_codes = []
    if not delta_large_enough:
        reason_codes.append("SUCCESS_DELTA_BELOW_MINIMUM")
    if not statistically_positive:
        reason_codes.append("SUCCESS_DELTA_CI_INCLUDES_ZERO")
    if passed:
        reason_codes.append("PAIRED_SUCCESS_IMPROVEMENT_CONFIRMED")

    best_label = max(ROLE_ORDER, key=lambda label: summaries[label]["goal_reached_rate"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "gate": {
            "status": "PASS" if passed else "FAIL",
            "primary_metric": "paired_final_minus_baseline_goal_reached_rate",
            "min_required_absolute_success_delta": min_success_delta,
            "paired_success_delta": success_delta,
            "paired_success_delta_seed_block_bootstrap95_low": success_delta_ci[0],
            "paired_success_delta_seed_block_bootstrap95_high": success_delta_ci[1],
            "paired_reward_delta": reward_delta,
            "paired_reward_delta_seed_block_bootstrap95_low": reward_delta_ci[0],
            "paired_reward_delta_seed_block_bootstrap95_high": reward_delta_ci[1],
            "reward_is_diagnostic_not_a_gate": True,
            "reason_codes": reason_codes,
        },
        "best_checkpoint_by_success": best_label,
        "checkpoint_summaries": summaries,
        "protocol": {
            "seeds": seeds,
            "seed_blocks": len(seeds),
            "num_envs_per_seed": manifest["num_envs_per_seed"],
            "episodes_per_checkpoint": len(seeds) * int(manifest["num_envs_per_seed"]),
            "one_episode_per_environment": True,
            "same_reference_config_sha256": manifest["reference_config_sha256"],
            "training_loss_used_as_evidence": False,
        },
    }


def upload_summary_to_wandb(
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    project: str,
    entity: str,
    run_name: str,
    group: str | None,
) -> None:
    try:
        import wandb
    except ImportError:
        raise GateError("wandb is not installed in the report environment") from None

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        job_type="fixed-seed-policy-evaluation",
        config={
            "protocol": manifest["protocol"],
            "seeds": manifest["seeds"],
            "num_envs_per_seed": manifest["num_envs_per_seed"],
            "reference_config": manifest["reference_config"],
            "reference_config_sha256": manifest["reference_config_sha256"],
            "training_loss_used_as_evidence": False,
        },
    )
    try:
        for ordinal, label in enumerate(ROLE_ORDER):
            metrics = summary["checkpoint_summaries"][label]
            run.log(
                {
                    "eval/checkpoint_ordinal": ordinal,
                    "eval/checkpoint_label": label,
                    "eval/checkpoint_step": (
                        metrics["checkpoint_step"]
                        if metrics["checkpoint_step"] is not None
                        else -1
                    ),
                    "eval/goal_reached_rate": metrics["goal_reached_rate"],
                    "eval/goal_reached_rate_wilson95_low": metrics[
                        "goal_reached_rate_wilson95_low"
                    ],
                    "eval/goal_reached_rate_wilson95_high": metrics[
                        "goal_reached_rate_wilson95_high"
                    ],
                    "eval/reward": metrics["reward"],
                    "eval/reward_bootstrap95_low": metrics[
                        "reward_seed_block_bootstrap95_low"
                    ],
                    "eval/reward_bootstrap95_high": metrics[
                        "reward_seed_block_bootstrap95_high"
                    ],
                    "eval/episodes": metrics["episodes"],
                },
                step=ordinal,
            )
        gate = summary["gate"]
        run.summary["policy_improvement_gate/status"] = gate["status"]
        run.summary["policy_improvement_gate/paired_success_delta"] = gate[
            "paired_success_delta"
        ]
        run.summary["policy_improvement_gate/paired_success_delta_ci95_low"] = gate[
            "paired_success_delta_seed_block_bootstrap95_low"
        ]
        run.summary["policy_improvement_gate/paired_success_delta_ci95_high"] = gate[
            "paired_success_delta_seed_block_bootstrap95_high"
        ]
        run.summary["policy_improvement_gate/paired_reward_delta"] = gate[
            "paired_reward_delta"
        ]
        run.summary["policy_improvement_gate/reason_codes"] = ",".join(gate["reason_codes"])
        run.summary["policy_improvement_gate/best_checkpoint"] = summary[
            "best_checkpoint_by_success"
        ]
    finally:
        run.finish(exit_code=0 if summary["gate"]["status"] == "PASS" else 1)


def _prepare_command(args: argparse.Namespace) -> int:
    manifest = create_manifest(
        train_run_dir=Path(args.train_run_dir),
        baseline_checkpoint=Path(args.baseline_checkpoint),
        reference_config=Path(args.reference_config),
        seeds=parse_seed_list(args.seeds),
        num_envs=args.num_envs,
        output_path=Path(args.output),
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_reference_config_sha256=args.expected_reference_config_sha256,
    )
    print(len(manifest["tasks"]))
    return 0


def _task_field_command(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest))
    task = task_from_manifest(manifest, args.index)
    allowed = {
        "label",
        "checkpoint",
        "checkpoint_step",
        "checkpoint_sha256",
        "seeds",
    }
    if args.field not in allowed:
        raise GateError(f"unsupported task field: {args.field}")
    value = task[args.field]
    if isinstance(value, list):
        print(",".join(str(item) for item in value))
    elif value is None:
        print("")
    else:
        print(value)
    return 0


def _record_command(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest))
    metrics = _load_json(Path(args.metrics))
    result = build_result(
        manifest=manifest,
        label=args.label,
        seed=args.seed,
        metrics=metrics,
        evaluated_checkpoint_sha256=args.evaluated_checkpoint_sha256,
        evaluated_config_sha256=args.evaluated_config_sha256,
    )
    _atomic_json(Path(args.output), result)
    return 0


def _validate_result_command(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest))
    row = _load_json(Path(args.result))
    role = next(
        (role for role in manifest["checkpoint_roles"] if role["label"] == args.label), None
    )
    valid = bool(
        role
        and row.get("checkpoint_label") == args.label
        and row.get("seed") == args.seed
        and row.get("checkpoint_sha256") == role["checkpoint_sha256"]
        and row.get("reference_config_sha256") == manifest["reference_config_sha256"]
        and row.get("evaluated_checkpoint_sha256") == args.evaluated_checkpoint_sha256
        and row.get("evaluated_config_sha256") == args.evaluated_config_sha256
        and row.get("episodes") == manifest["expected_episodes_per_seed"]
    )
    return 0 if valid else 1


def _sanitize_command(args: argparse.Namespace) -> int:
    provenance = sanitize_checkpoint(
        Path(args.checkpoint), Path(args.output), args.expected_sha256
    )
    if args.provenance:
        provenance_path = Path(args.provenance)
        _atomic_json(provenance_path, provenance)
        os.chmod(provenance_path, 0o444)
    return 0


def _summarize_command(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest))
    try:
        summary = summarize_results(
            manifest=manifest,
            results_dir=Path(args.results_dir),
            min_success_delta=args.min_success_delta,
        )
    except GateError as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "gate": {
                "status": "INCOMPLETE",
                "primary_metric": "paired_final_minus_baseline_goal_reached_rate",
                "reason_codes": ["INCOMPLETE_RESULTS"],
                "detail": str(exc),
            },
        }
        _atomic_json(Path(args.output), summary)
        print(json.dumps(summary["gate"], sort_keys=True))
        return 2
    _atomic_json(Path(args.output), summary)
    print(json.dumps(summary["gate"], sort_keys=True))
    return 0 if summary["gate"]["status"] == "PASS" else 3


def _upload_command(args: argparse.Namespace) -> int:
    manifest = _load_json(Path(args.manifest))
    summary = _load_json(Path(args.summary))
    if summary.get("gate", {}).get("status") not in {"PASS", "FAIL"}:
        raise GateError("refusing to upload an incomplete fixed-seed evaluation")
    upload_summary_to_wandb(
        summary=summary,
        manifest=manifest,
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
        group=args.group,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="select checkpoints and write a manifest")
    prepare.add_argument("--train-run-dir", required=True)
    prepare.add_argument("--baseline-checkpoint", required=True)
    prepare.add_argument("--reference-config", required=True)
    prepare.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    prepare.add_argument("--num-envs", type=int, default=32)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--expected-baseline-sha256")
    prepare.add_argument("--expected-reference-config-sha256")
    prepare.set_defaults(func=_prepare_command)

    task_field = subparsers.add_parser("task-field", help="read one task field safely")
    task_field.add_argument("--manifest", required=True)
    task_field.add_argument("--index", type=int, required=True)
    task_field.add_argument("--field", required=True)
    task_field.set_defaults(func=_task_field_command)

    sanitize = subparsers.add_parser(
        "sanitize-checkpoint", help="remove environment and optimizer state for fixed-seed eval"
    )
    sanitize.add_argument("--checkpoint", required=True)
    sanitize.add_argument("--expected-sha256", required=True)
    sanitize.add_argument("--output", required=True)
    sanitize.add_argument("--provenance")
    sanitize.set_defaults(func=_sanitize_command)

    record = subparsers.add_parser("record", help="validate and enrich one metrics file")
    record.add_argument("--manifest", required=True)
    record.add_argument("--label", required=True, choices=ROLE_ORDER)
    record.add_argument("--seed", required=True, type=int)
    record.add_argument("--metrics", required=True)
    record.add_argument("--output", required=True)
    record.add_argument("--evaluated-checkpoint-sha256", required=True)
    record.add_argument("--evaluated-config-sha256", required=True)
    record.set_defaults(func=_record_command)

    validate_result = subparsers.add_parser(
        "validate-result", help="check that a restart-safe result used the current staged bytes"
    )
    validate_result.add_argument("--manifest", required=True)
    validate_result.add_argument("--label", required=True, choices=ROLE_ORDER)
    validate_result.add_argument("--seed", required=True, type=int)
    validate_result.add_argument("--result", required=True)
    validate_result.add_argument("--evaluated-checkpoint-sha256", required=True)
    validate_result.add_argument("--evaluated-config-sha256", required=True)
    validate_result.set_defaults(func=_validate_result_command)

    summarize = subparsers.add_parser("summarize", help="compute the paired improvement gate")
    summarize.add_argument("--manifest", required=True)
    summarize.add_argument("--results-dir", required=True)
    summarize.add_argument("--min-success-delta", type=float, default=0.02)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=_summarize_command)

    upload = subparsers.add_parser("upload", help="log a complete report to W&B")
    upload.add_argument("--manifest", required=True)
    upload.add_argument("--summary", required=True)
    upload.add_argument("--project", required=True)
    upload.add_argument("--entity", required=True)
    upload.add_argument("--run-name", required=True)
    upload.add_argument("--group")
    upload.set_defaults(func=_upload_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
