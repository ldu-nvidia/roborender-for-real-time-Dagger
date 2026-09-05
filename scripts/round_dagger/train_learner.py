#!/usr/bin/env python3
"""Launch the camera-free round-based DAgger learner under Python or torchrun."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import sha256_file  # noqa: E402
from round_dagger.actor import ActorExportError, runtime_state_dict_sha256  # noqa: E402
from round_dagger.checkpoint_compat import (  # noqa: E402
    CheckpointCompatibilityError,
    load_hash_pinned_policy_state_dict,
)
from round_dagger.learner_identity import learner_identity_sha256  # noqa: E402
from round_dagger.learner import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    JsonlTelemetry,
    LearnerBundleConfig,
    LearnerError,
    MappingPolicyAdapter,
    PolicyAdapter,
    configure_deterministic_runtime,
    publish_learner_result,
    train_from_replay,
)
from round_dagger.native_actor_protocol import train_round_relative_root  # noqa: E402
from round_dagger.replay import load_snapshot  # noqa: E402


def _require_campaign_path(path: Path, label: str, *, current_generation: Path | None = None) -> Path:
    resolved = path.expanduser().resolve(strict=path.exists())
    root = CANONICAL_CAMPAIGN_ROOT.resolve(strict=True)
    required_root = root if current_generation is None else current_generation
    try:
        resolved.relative_to(required_root)
    except ValueError as exc:
        raise LearnerError(f"{label} lies outside canonical campaign storage") from exc
    return resolved


def _symbol(specification: str) -> Any:
    if ":" not in specification:
        raise LearnerError("factory must use module.path:symbol syntax")
    module_name, symbol_name = specification.rsplit(":", 1)
    try:
        return getattr(importlib.import_module(module_name), symbol_name)
    except (ImportError, AttributeError) as exc:
        raise LearnerError(f"cannot import {specification}: {exc}") from exc


def _load_initial_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_policy_state_sha256: str | None = None,
) -> None:
    try:
        state = load_hash_pinned_policy_state_dict(
            path,
            expected_sha256=expected_checkpoint_sha256,
            map_location="cpu",
        )
    except CheckpointCompatibilityError as exc:
        raise LearnerError(f"cannot load initial policy checkpoint: {exc}") from exc
    try:
        policy_state_sha256 = runtime_state_dict_sha256(state)
    except ActorExportError as exc:
        raise LearnerError(f"cannot identify initial policy state: {exc}") from exc
    if (
        expected_policy_state_sha256 is not None
        and policy_state_sha256 != expected_policy_state_sha256
    ):
        raise LearnerError(
            "initial policy state identity differs from portable policy spec"
        )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise LearnerError(f"initial policy state is incompatible: {exc}") from exc


def _distributed_device(requested: str) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and requested != "cpu" else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "auto":
        return torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    if requested == "cuda":
        return torch.device(f"cuda:{local_rank}")
    device = torch.device(requested)
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{local_rank}")
    return device


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _publish_frontier(source: Path, destination: Path) -> str:
    """Create one read-only checkpoint for a validation frontier."""

    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source.is_symlink() or not source.is_file():
        raise LearnerError("learner restart checkpoint is missing or unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source_digest = sha256_file(source)
    try:
        descriptor = os.open(destination, flags, 0o444)
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != source_digest
        ):
            raise LearnerError("validation frontier path already contains different bytes")
        return source_digest
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
            writer.flush()
            os.fchmod(writer.fileno(), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.fsync(writer.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return source_digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a policy from a sealed cumulative DAgger replay snapshot"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "campaign/projection root used to resolve producer-absolute replay "
            "paths after replay materialization"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-factory", required=True)
    parser.add_argument(
        "--adapter-factory",
        help="optional module:function returning a PolicyAdapter from the bundle mapping",
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--expected-snapshot-file-sha256")
    parser.add_argument("--learner-identity-sha256")
    parser.add_argument("--runtime-container-sha256")
    parser.add_argument("--learner-contract-sha256")
    parser.add_argument(
        "--target-optimizer-examples",
        required=True,
        type=int,
        help="global optimizer-example frontier pinned by the learner bundle",
    )
    parser.add_argument(
        "--checkpoint-every-optimizer-examples",
        required=True,
        type=int,
        help="global optimizer-example interval pinned by the learner bundle",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--telemetry-jsonl", type=Path)
    parser.add_argument(
        "--frontier-output",
        required=True,
        type=Path,
        help="generation-local immutable checkpoint evaluated at this frontier",
    )
    parser.add_argument(
        "--no-publish-result",
        action="store_true",
        help="write only restart.pt (useful for an intentional restart smoke)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Production passes ``--device cuda``.  Establish and validate its strict
    # numerical contract before importing/calling a model factory, so a future
    # factory cannot accidentally initialize CUDA under nondeterministic
    # defaults.  ``train_from_replay`` repeats this gate after rank resolution.
    if arguments.device != "auto":
        try:
            requested_device = torch.device(arguments.device)
        except Exception as exc:
            raise LearnerError(f"invalid learner device: {arguments.device!r}") from exc
        if requested_device.type == "cuda":
            configure_deterministic_runtime(requested_device)
    config = LearnerBundleConfig.load(arguments.bundle)
    generation_root = (
        CANONICAL_CAMPAIGN_ROOT / train_round_relative_root(config.generation_id)
    ).resolve(strict=True)
    if arguments.artifact_root is None or arguments.artifact_root.resolve(strict=True) != CANONICAL_CAMPAIGN_ROOT.resolve(strict=True):
        raise LearnerError("--artifact-root must be the canonical H200 data root")
    for path, label in (
        (arguments.bundle, "learner bundle"),
        (arguments.snapshot, "train replay snapshot"),
        (arguments.output_dir, "learner output"),
        (arguments.frontier_output, "validation frontier checkpoint"),
    ):
        _require_campaign_path(path, label, current_generation=generation_root)
    for path, label in (
        (arguments.initial_checkpoint, "initial checkpoint"),
        (arguments.resume_from, "restart checkpoint"),
    ):
        if path is not None:
            _require_campaign_path(path, label)
    if arguments.target_optimizer_examples > config.target_optimizer_examples:
        raise LearnerError(
            "--target-optimizer-examples exceeds the immutable stage maximum"
        )
    if (
        arguments.checkpoint_every_optimizer_examples
        != config.checkpoint_every_optimizer_examples
    ):
        raise LearnerError(
            "--checkpoint-every-optimizer-examples differs from the immutable bundle"
        )
    identity_arguments = (
        arguments.learner_identity_sha256,
        arguments.runtime_container_sha256,
        arguments.learner_contract_sha256,
    )
    if any(value is not None for value in identity_arguments) and any(
        value is None for value in identity_arguments
    ):
        raise LearnerError("learner semantic identity arguments are incomplete")
    if not arguments.no_publish_result and any(
        value is None for value in identity_arguments
    ):
        raise LearnerError(
            "published learner results require --learner-identity-sha256, "
            "--runtime-container-sha256, and --learner-contract-sha256"
        )
    bundle_mapping = config.to_mapping()
    if (
        config.initial_checkpoint_sha256 is not None
        and arguments.initial_checkpoint is None
    ):
        raise LearnerError(
            "immutable learner bundle pins an initial checkpoint; "
            "--initial-checkpoint is required"
        )
    if arguments.initial_checkpoint is not None:
        actual = sha256_file(arguments.initial_checkpoint)
        if config.initial_checkpoint_sha256 is None:
            raise LearnerError(
                "bundle must pin initial_checkpoint_sha256 when --initial-checkpoint is used"
            )
        if actual != config.initial_checkpoint_sha256:
            raise LearnerError("initial policy checkpoint SHA-256 mismatch")
    factory = _symbol(arguments.model_factory)
    model = factory(bundle_mapping)
    if not isinstance(model, nn.Module):
        raise LearnerError("model factory did not return torch.nn.Module")
    if arguments.initial_checkpoint is not None:
        _load_initial_checkpoint(
            model,
            arguments.initial_checkpoint,
            expected_checkpoint_sha256=config.initial_checkpoint_sha256,
            expected_policy_state_sha256=bundle_mapping["model_config"].get(
                "policy_state_sha256"
            ),
        )
    adapter: PolicyAdapter
    if arguments.adapter_factory:
        adapter_factory = _symbol(arguments.adapter_factory)
        adapter = adapter_factory(bundle_mapping)
        if not hasattr(adapter, "actions"):
            raise LearnerError(
                "adapter factory result lacks actions(model, observations)"
            )
    else:
        adapter = MappingPolicyAdapter(config.output_key)

    device = _distributed_device(arguments.device)
    rank = _rank()
    snapshot_identity = load_snapshot(
        arguments.snapshot,
        expected_file_sha256=arguments.expected_snapshot_file_sha256,
        artifact_root=arguments.artifact_root,
    )
    if (
        snapshot_identity["campaign_id"] != config.campaign_id
        or snapshot_identity["through_macro_round"] != config.macro_round
    ):
        raise LearnerError("replay snapshot lineage differs from learner bundle")
    parent_runtime_sha256 = config.to_mapping()["model_config"][
        "runtime_policy_sha256"
    ]
    if all(value is not None for value in identity_arguments):
        identity_sha256 = learner_identity_sha256(
            campaign_id=config.campaign_id,
            macro_round=config.macro_round,
            learner_code_sha256=config.code_sha256,
            learner_config_sha256=config.sha256,
            replay_snapshot_file_sha256=snapshot_identity["file_sha256"],
            replay_snapshot_sha256=snapshot_identity["snapshot_sha256"],
            parent_checkpoint_sha256=config.model_lineage_sha256,
            parent_runtime_policy_sha256=parent_runtime_sha256,
            runtime_container_sha256=arguments.runtime_container_sha256,
            learner_contract_sha256=arguments.learner_contract_sha256,
        )
        if identity_sha256 != arguments.learner_identity_sha256:
            raise LearnerError(
                "learner identity SHA-256 does not bind the exact semantic run"
            )
    telemetry = JsonlTelemetry(
        arguments.telemetry_jsonl,
        enabled=rank == 0,
    )
    result = None
    telemetry_exit_code = 2
    try:
        result = train_from_replay(
            model,
            snapshot_path=arguments.snapshot,
            config=config,
            output_dir=arguments.output_dir,
            target_optimizer_examples=arguments.target_optimizer_examples,
            adapter=adapter,
            device=device,
            resume_from=arguments.resume_from,
            expected_snapshot_file_sha256=arguments.expected_snapshot_file_sha256,
            artifact_root=arguments.artifact_root,
            checkpoint_every_optimizer_examples=(
                arguments.checkpoint_every_optimizer_examples
            ),
            telemetry=telemetry,
            learner_identity_sha256=arguments.learner_identity_sha256,
            runtime_container_sha256=arguments.runtime_container_sha256,
            learner_contract_sha256=arguments.learner_contract_sha256,
        )
        telemetry_exit_code = 0
    finally:
        # Close local telemetry before the distributed ranks leave together.
        try:
            telemetry.close(exit_code=telemetry_exit_code)
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
    if result is None:
        raise LearnerError("learner returned without a training result")
    frontier_sha256 = None
    if rank == 0:
        frontier_sha256 = _publish_frontier(
            Path(result.restart_checkpoint), arguments.frontier_output
        )
    if rank == 0 and not arguments.no_publish_result:
        report = publish_learner_result(
            output_dir=arguments.output_dir,
            result=result,
            config=config,
            learner_identity_sha256=arguments.learner_identity_sha256,
            runtime_container_sha256=arguments.runtime_container_sha256,
            learner_contract_sha256=arguments.learner_contract_sha256,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    elif rank == 0:
        print(
            json.dumps(
                {
                    "completed_steps": result.completed_steps,
                    "target_steps": result.target_steps,
                    **dict(result.exposure),
                    "restart_checkpoint": result.restart_checkpoint,
                    "frontier_checkpoint": str(arguments.frontier_output),
                    "frontier_checkpoint_sha256": frontier_sha256,
                    "replay_snapshot_sha256": result.replay_snapshot_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LearnerError as error:
        print(f"round DAgger learner failed: {error}", file=sys.stderr)
        raise SystemExit(2)
