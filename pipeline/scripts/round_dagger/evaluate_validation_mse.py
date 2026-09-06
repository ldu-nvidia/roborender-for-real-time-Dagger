#!/usr/bin/env python3
"""Evaluate one checkpoint on a sealed validation-only replay artifact.

This executable is intentionally separate from ``train_learner.py``.  Its
parser has no train snapshot argument, and the learner parser has no validation
argument, so validation bytes cannot enter the optimizer process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import load_strict_json, publish_immutable_json  # noqa: E402
from round_dagger.checkpoint_compat import (  # noqa: E402
    CheckpointCompatibilityError,
    load_restricted_restart_document,
)
from round_dagger.learner import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    CHECKPOINT_SCHEMA,
    LearnerBundleConfig,
    LearnerError,
    MappingPolicyAdapter,
    _aligned_learned_actions,
    _partition_loaded_snapshot,
    configure_deterministic_runtime,
    optimizer_exposure_summary,
)
from round_dagger.replay import ReplayError, load_validation_snapshot  # noqa: E402
from round_dagger.native_actor_protocol import train_round_relative_root  # noqa: E402


RESULT_SCHEMA = "fast_roborender.round_dagger.validation_mse_result.v1"
EVALUATOR_WORLD_SIZE = 8
EXPECTED_VALIDATION_EXAMPLES = 256 * 512
CANONICAL_VALIDATION_ROOT = Path(
    "/workspace/project/roborender/data/validation/heldout"
)


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LearnerError(f"{label} must be a lowercase SHA-256")
    return value


def _rooted_path(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=path.exists())
    root = root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LearnerError(f"{label} lies outside {root}") from exc
    return resolved


def _symbol(specification: str) -> Any:
    if ":" not in specification:
        raise LearnerError("factory must use module.path:symbol syntax")
    module_name, symbol_name = specification.rsplit(":", 1)
    try:
        return getattr(importlib.import_module(module_name), symbol_name)
    except (ImportError, AttributeError) as exc:
        raise LearnerError(f"cannot import {specification}: {exc}") from exc


def _pinned_checkpoint(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Read the external inode once, then decode that exact private snapshot."""

    expected_sha256 = _sha256(expected_sha256, "expected checkpoint SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LearnerError(f"cannot safely open validation checkpoint: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LearnerError("validation checkpoint is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read()
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LearnerError("validation checkpoint SHA-256 mismatch")
    try:
        with tempfile.TemporaryDirectory(prefix="round-dagger-validation-") as name:
            private = Path(name) / "checkpoint.pt"
            private.write_bytes(payload)
            os.chmod(private, stat.S_IRUSR)
            return load_restricted_restart_document(private, map_location="cpu")
    except CheckpointCompatibilityError as exc:
        raise LearnerError(f"validation checkpoint is invalid: {exc}") from exc


def _pinned_bundle(path: Path, expected_sha256: str) -> LearnerBundleConfig:
    """Decode the exact hash-pinned learner bundle inode once."""

    expected_sha256 = _sha256(
        expected_sha256, "expected learner bundle SHA-256"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LearnerError(f"cannot safely open validation learner bundle: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LearnerError("validation learner bundle is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read()
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise LearnerError("validation learner bundle SHA-256 mismatch")
    try:
        with tempfile.TemporaryDirectory(prefix="round-dagger-validation-bundle-") as name:
            private = Path(name) / "learner-bundle.json"
            private.write_bytes(payload)
            os.chmod(private, stat.S_IRUSR)
            return LearnerBundleConfig.from_mapping(load_strict_json(private))
    except Exception as exc:
        if isinstance(exc, LearnerError):
            raise
        raise LearnerError(f"validation learner bundle is invalid: {exc}") from exc


def _device(requested: str) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and requested != "cpu" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    if requested == "auto":
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        return torch.device(f"cuda:{local_rank}")
    result = torch.device(requested)
    return torch.device(f"cuda:{local_rank}") if result.type == "cuda" and result.index is None else result


def _verify_evaluator_hardware(device: torch.device) -> tuple[int, int]:
    if not torch.distributed.is_initialized():
        raise LearnerError("validation MSE requires one eight-rank process group")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        world_size != EVALUATOR_WORLD_SIZE
        or not 0 <= rank < EVALUATOR_WORLD_SIZE
        or local_rank != rank
        or device != torch.device(f"cuda:{local_rank}")
        or torch.cuda.device_count() != EVALUATOR_WORLD_SIZE
    ):
        raise LearnerError("validation MSE requires exactly all eight local H200 ranks")
    properties = torch.cuda.get_device_properties(device)
    if (
        properties.name != "NVIDIA H200"
        or (properties.major, properties.minor) != (9, 0)
        or int(properties.total_memory) < 140_000 * 1024 * 1024
    ):
        raise LearnerError("validation MSE rank is not assigned an H200")
    return rank, world_size


def _validation_partition(arguments, config, *, rank: int, world_size: int):
    try:
        snapshot = load_validation_snapshot(
            arguments.validation_artifact,
            expected_file_sha256=arguments.validation_artifact_sha256,
            artifact_root=arguments.artifact_root,
        )
    except ReplayError as exc:
        raise LearnerError(f"validation replay snapshot is invalid: {exc}") from exc
    expected = {
        "purpose": "validation",
        "campaign_id": config.campaign_id,
        "snapshot_sha256": arguments.validation_snapshot_sha256,
        "split_sha256": arguments.split_sha256,
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise LearnerError("validation replay identity mismatch")
    return _partition_loaded_snapshot(
        snapshot,
        config,
        rank=rank,
        world_size=world_size,
        training_frontier=None,
    )


def _validation_mse(model, partition, adapter, device) -> tuple[float, int]:
    """Stream only squared-error sum/count; do not gather raw validation rows."""

    model.eval()
    statistics = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, partition.transition_count, partition.config.batch_size):
            stop = min(start + partition.config.batch_size, partition.transition_count)
            observations, targets = partition.batch(range(start, stop), device=device)
            predictions = adapter.actions(model, observations)
            predictions, targets, _indices = _aligned_learned_actions(
                predictions, targets, partition.config.train_action_slice
            )
            difference = predictions.to(torch.float64) - targets.to(torch.float64)
            statistics[0] += torch.square(difference).sum()
            statistics[1] += difference.numel()
    if torch.distributed.is_initialized():
        gathered = [torch.empty_like(statistics) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, statistics)
        statistics = gathered[0].clone()
        for item in gathered[1:]:
            statistics.add_(item)
    count = int(statistics[1].item())
    if count <= 0:
        raise LearnerError("validation MSE population is empty")
    mse = float((statistics[0] / statistics[1]).item())
    if not torch.isfinite(torch.tensor(mse)):
        raise LearnerError("validation MSE is non-finite")
    return mse, sum(partition.transition_counts_by_rank)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute MSE on purpose=validation replay")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-artifact", required=True, type=Path)
    parser.add_argument("--validation-artifact-sha256", required=True)
    parser.add_argument("--validation-snapshot-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--learner-identity-sha256", required=True)
    parser.add_argument("--runtime-environment-sha256", required=True)
    parser.add_argument("--learner-contract-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--model-factory", required=True)
    parser.add_argument("--adapter-factory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    for value, label in (
        (arguments.checkpoint_sha256, "checkpoint SHA-256"),
        (arguments.bundle_sha256, "learner bundle SHA-256"),
        (arguments.validation_artifact_sha256, "validation artifact SHA-256"),
        (arguments.validation_snapshot_sha256, "validation snapshot SHA-256"),
        (arguments.split_sha256, "validation split SHA-256"),
        (arguments.learner_identity_sha256, "learner identity SHA-256"),
        (arguments.runtime_environment_sha256, "runtime environment SHA-256"),
        (arguments.learner_contract_sha256, "learner contract SHA-256"),
    ):
        _sha256(value, label)
    if (
        arguments.artifact_root is None
        or arguments.artifact_root.resolve(strict=True)
        != CANONICAL_VALIDATION_ROOT.resolve(strict=True)
    ):
        raise LearnerError(
            "--artifact-root must be the exact sealed validation root"
        )
    for path, label in (
        (arguments.bundle, "learner bundle"),
        (arguments.checkpoint, "checkpoint"),
    ):
        _rooted_path(path, CANONICAL_CAMPAIGN_ROOT, label)
    _rooted_path(
        arguments.validation_artifact,
        CANONICAL_VALIDATION_ROOT,
        "validation artifact",
    )
    output = _rooted_path(
        arguments.output, CANONICAL_CAMPAIGN_ROOT, "validation result"
    )
    # Admit every pathname before reading even the learner bundle.  In
    # particular, a caller cannot point this validation-only process at the
    # private test tree and have it decoded before the boundary check.
    config = _pinned_bundle(arguments.bundle, arguments.bundle_sha256)
    expected_output_root = (
        CANONICAL_CAMPAIGN_ROOT
        / "validation/results"
        / train_round_relative_root(config.generation_id)
        / "mse"
    ).resolve(strict=True)
    if output.parent != expected_output_root:
        raise LearnerError(
            "validation result must be in the round-bound validation MSE directory"
        )
    device = _device(arguments.device)
    rank, world_size = _verify_evaluator_hardware(device)
    configure_deterministic_runtime(device)
    torch.manual_seed(config.seed)
    checkpoint = _pinned_checkpoint(arguments.checkpoint, arguments.checkpoint_sha256)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise LearnerError("validation checkpoint schema mismatch")
    expected_checkpoint = {
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "model_lineage_sha256": config.model_lineage_sha256,
        "learner_identity_sha256": arguments.learner_identity_sha256,
        "runtime_container_sha256": arguments.runtime_environment_sha256,
        "learner_contract_sha256": arguments.learner_contract_sha256,
        "world_size": EVALUATOR_WORLD_SIZE,
    }
    if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
        raise LearnerError("validation checkpoint lineage mismatch")
    completed_steps = checkpoint.get("completed_steps")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps < 0:
        raise LearnerError("validation checkpoint optimizer frontier is invalid")
    factory = _symbol(arguments.model_factory)
    model = factory(config.to_mapping())
    if not isinstance(model, nn.Module):
        raise LearnerError("model factory did not return torch.nn.Module")
    try:
        model.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    except Exception as exc:
        raise LearnerError(f"validation checkpoint policy state is incompatible: {exc}") from exc
    adapter = (
        _symbol(arguments.adapter_factory)(config.to_mapping())
        if arguments.adapter_factory
        else MappingPolicyAdapter(config.output_key)
    )
    partition = _validation_partition(
        arguments, config, rank=rank, world_size=world_size
    )
    model = model.to(device)
    validation_mse, validation_examples = _validation_mse(
        model, partition, adapter, device
    )
    if validation_examples != EXPECTED_VALIDATION_EXAMPLES:
        raise LearnerError(
            "validation snapshot must contain exactly 256x512 labeled transitions"
        )
    training_world_size = checkpoint.get("world_size")
    if (
        isinstance(training_world_size, bool)
        or not isinstance(training_world_size, int)
        or training_world_size != EVALUATOR_WORLD_SIZE
    ):
        raise LearnerError("checkpoint training world size is invalid")
    exposure = optimizer_exposure_summary(
        {"counts_by_round": checkpoint.get("training_counts_by_round")},
        config,
        completed_steps=completed_steps,
        world_size=training_world_size,
    )
    if any(checkpoint.get(key) != value for key, value in exposure.items()):
        raise LearnerError("checkpoint optimizer-example exposure mismatch")
    document = {
        "schema": RESULT_SCHEMA,
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "checkpoint_sha256": arguments.checkpoint_sha256,
        "validation_artifact_sha256": arguments.validation_artifact_sha256,
        "validation_snapshot_sha256": arguments.validation_snapshot_sha256,
        "split_sha256": arguments.split_sha256,
        **exposure,
        "optimizer_examples_per_step": config.batch_size * training_world_size,
        "validation_examples": validation_examples,
        "validation_mse": validation_mse,
    }
    if rank == 0:
        digest = publish_immutable_json(arguments.output, document)
        print(f"VALIDATION_MSE_RESULT path={arguments.output} sha256={digest}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LearnerError as error:
        print(f"validation MSE evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
