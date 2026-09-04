#!/usr/bin/env python3
"""Launch the camera-free round-based DAgger learner under Python or torchrun."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
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
from round_dagger.learner_identity import (  # noqa: E402
    learner_identity_sha256,
    learner_wandb_run_id,
)
from round_dagger.learner import (  # noqa: E402
    JsonlTelemetry,
    LearnerBundleConfig,
    LearnerError,
    MappingPolicyAdapter,
    PolicyAdapter,
    configure_deterministic_runtime,
    publish_learner_result,
    train_from_replay,
)
from round_dagger.replay import load_heldout_snapshot, load_snapshot  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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


def _wandb_resume_modes(arguments: argparse.Namespace) -> tuple[bool, bool]:
    """Return strict W&B modes without changing ordinary production behavior."""

    resume_must = (
        arguments.resume_from is not None and arguments.wandb_run_id is not None
    )
    resume_never = bool(arguments.wandb_resume_never)
    if resume_never and resume_must:
        raise LearnerError(
            "--wandb-resume-never cannot be combined with --resume-from"
        )
    if resume_never and (
        arguments.wandb_project is None or arguments.wandb_run_id is None
    ):
        raise LearnerError(
            "--wandb-resume-never requires --wandb-project and --wandb-run-id"
        )
    return resume_must, resume_never


def _expected_wandb_run_id(
    canonical_run_id: str, binding_sha256: str | None
) -> str:
    """Derive the exact external run ID, optionally bound to a live gate.

    Ordinary production learners retain the canonical semantic learner ID.
    The heldout-100 validation gate additionally binds its fresh campaign,
    held-out snapshot, and production contract into ``binding_sha256``.  Do
    not accept an arbitrary caller suffix: it must be the deterministic first
    32 hex characters of a full SHA-256 supplied on its dedicated argument.
    """

    if binding_sha256 is None:
        return canonical_run_id
    if _SHA256_RE.fullmatch(binding_sha256) is None:
        raise LearnerError("W&B run binding must be a lowercase SHA-256")
    run_id = f"{canonical_run_id}-b{binding_sha256[:32]}"
    if len(run_id) > 128:
        raise LearnerError("binding-qualified W&B run ID exceeds 128 characters")
    return run_id


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
            "paths after an RTX-to-H200 handoff"
        ),
    )
    parser.add_argument(
        "--heldout-snapshot",
        type=Path,
        help="immutable purpose=eval replay snapshot pinned by the learner bundle",
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
        "--max-steps",
        type=int,
        help="must match the immutable learner bundle (defaults to its value)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="must match the immutable learner bundle (defaults to its value)",
    )
    parser.add_argument(
        "--heldout-eval-interval",
        type=int,
        help=(
            "must match heldout_eval_interval in the immutable learner bundle "
            "(production target: 100)"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--telemetry-jsonl", type=Path)
    parser.add_argument("--heldout-telemetry-jsonl", type=Path)
    parser.add_argument("--training-quality-telemetry-jsonl", type=Path)
    parser.add_argument(
        "--dagger-wandb-dashboard",
        action="store_true",
        help=(
            "publish the sampled-scene, training-quality, and OOD-validation "
            "W&B namespaces"
        ),
    )
    parser.add_argument(
        "--scene-sample-manifest",
        type=Path,
        help="immutable sampled condition/render contact-sheet manifest",
    )
    parser.add_argument(
        "--dagger-beta",
        type=float,
        help="scheduled expert-behavior probability bound by the replay contract",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--wandb-run-binding-sha256",
        help=(
            "validation-only full SHA-256 used to derive the exact -b<32hex> "
            "suffix on the canonical learner W&B run ID"
        ),
    )
    parser.add_argument(
        "--wandb-resume-never",
        action="store_true",
        help=(
            "fail if the canonical W&B run ID already exists; validation-only "
            "first-invocation protection against stale deterministic history"
        ),
    )
    parser.add_argument(
        "--no-publish-result",
        action="store_true",
        help="write only restart.pt (useful for an intentional restart smoke)",
    )
    parser.add_argument(
        "--validation-stop-at-step",
        type=int,
        help=(
            "validation-only absolute optimizer frontier; must be below max_steps "
            "and requires --no-publish-result"
        ),
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
    max_steps = config.max_steps if arguments.max_steps is None else arguments.max_steps
    save_every = (
        config.save_every if arguments.save_every is None else arguments.save_every
    )
    if max_steps != config.max_steps:
        raise LearnerError("--max-steps differs from the immutable learner bundle")
    if save_every != config.save_every:
        raise LearnerError("--save-every differs from the immutable learner bundle")
    if config.heldout_evaluation_enabled != (
        arguments.heldout_snapshot is not None
    ):
        raise LearnerError(
            "--heldout-snapshot and immutable held-out bundle fields must be "
            "supplied together"
        )
    if arguments.heldout_eval_interval is not None and (
        arguments.heldout_eval_interval != config.heldout_eval_interval
    ):
        raise LearnerError(
            "--heldout-eval-interval differs from the immutable learner bundle"
        )
    if (
        arguments.heldout_telemetry_jsonl is not None
        and not config.heldout_evaluation_enabled
    ):
        raise LearnerError(
            "--heldout-telemetry-jsonl requires held-out evaluation"
        )
    if arguments.dagger_wandb_dashboard and (
        not config.heldout_evaluation_enabled
        or arguments.training_quality_telemetry_jsonl is None
        or arguments.scene_sample_manifest is None
        or arguments.dagger_beta is None
        or arguments.wandb_project is None
    ):
        raise LearnerError(
            "--dagger-wandb-dashboard requires held-out evaluation, W&B, "
            "--training-quality-telemetry-jsonl, --scene-sample-manifest, "
            "and --dagger-beta"
        )
    if (
        not arguments.dagger_wandb_dashboard
        and (
            arguments.training_quality_telemetry_jsonl is not None
            or arguments.scene_sample_manifest is not None
            or arguments.dagger_beta is not None
        )
    ):
        raise LearnerError(
            "training-quality telemetry and scene samples require "
            "--dagger-wandb-dashboard"
        )
    if (
        arguments.validation_stop_at_step is not None
        and not arguments.no_publish_result
    ):
        raise LearnerError(
            "--validation-stop-at-step requires --no-publish-result"
        )
    if arguments.validation_stop_at_step is not None and not (
        1 <= arguments.validation_stop_at_step < max_steps
    ):
        raise LearnerError(
            "--validation-stop-at-step must be in [1, max_steps)"
        )
    wandb_resume_must, wandb_resume_never = _wandb_resume_modes(arguments)
    if arguments.wandb_run_binding_sha256 is not None and (
        arguments.wandb_project is None or arguments.wandb_run_id is None
    ):
        raise LearnerError(
            "--wandb-run-binding-sha256 requires --wandb-project and "
            "--wandb-run-id"
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
    wandb_config = {
        "campaign_id": config.campaign_id,
        "round_dagger/campaign_id": config.campaign_id,
        "macro_round": config.macro_round,
        "learner_code_sha256": config.code_sha256,
        "learner_config_sha256": config.sha256,
        "model_lineage_sha256": config.model_lineage_sha256,
    }
    if arguments.dagger_wandb_dashboard:
        wandb_config.update(
            {
                "dashboard_schema": "fast_roborender.dagger_wandb.v2",
                "validation_distribution": (
                    "motion_trajectory_ood_same_task_scene_renderer"
                ),
                "validation_is_closed_loop": False,
                "validation_is_unseen_task_or_scene": False,
                "scheduled_dagger_beta": arguments.dagger_beta,
            }
        )
    if arguments.heldout_snapshot is not None:
        heldout_identity = load_heldout_snapshot(
            arguments.heldout_snapshot,
            expected_file_sha256=config.heldout_snapshot_file_sha256,
            artifact_root=arguments.artifact_root,
        )
        if heldout_identity["snapshot_sha256"] != config.heldout_snapshot_sha256:
            raise LearnerError("held-out replay snapshot semantic digest mismatch")
        if heldout_identity["campaign_id"] != config.campaign_id:
            raise LearnerError("held-out replay campaign differs from learner bundle")
        wandb_config.update(
            heldout_snapshot_file_sha256=heldout_identity["file_sha256"],
            heldout_snapshot_sha256=heldout_identity["snapshot_sha256"],
            heldout_eval_interval=config.heldout_eval_interval,
            heldout_transitions=heldout_identity["transition_count"],
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
    snapshot_identity = None
    parent_runtime_sha256 = None
    if arguments.wandb_project or all(
        value is not None for value in identity_arguments
    ):
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
        wandb_config.update(
            parent_runtime_policy_sha256=parent_runtime_sha256,
            replay_snapshot_file_sha256=snapshot_identity["file_sha256"],
            replay_snapshot_sha256=snapshot_identity["snapshot_sha256"],
        )
    if all(value is not None for value in identity_arguments):
        assert snapshot_identity is not None and parent_runtime_sha256 is not None
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
        if arguments.wandb_project:
            expected_run_id = _expected_wandb_run_id(
                learner_wandb_run_id(
                    campaign_id=config.campaign_id,
                    macro_round=config.macro_round,
                    learner_identity_sha256=identity_sha256,
                ),
                arguments.wandb_run_binding_sha256,
            )
            if arguments.wandb_run_id != expected_run_id:
                raise LearnerError("W&B run ID is not bound to learner identity")
        wandb_config.update(
            learner_identity_sha256=identity_sha256,
            runtime_container_sha256=arguments.runtime_container_sha256,
            learner_contract_sha256=arguments.learner_contract_sha256,
        )
        if arguments.wandb_run_binding_sha256 is not None:
            wandb_config["wandb_run_binding_sha256"] = (
                arguments.wandb_run_binding_sha256
            )
    telemetry = JsonlTelemetry(
        arguments.telemetry_jsonl,
        enabled=rank == 0,
        heldout_path=arguments.heldout_telemetry_jsonl,
        training_quality_path=arguments.training_quality_telemetry_jsonl,
        dagger_dashboard=arguments.dagger_wandb_dashboard,
        scene_sample_manifest=arguments.scene_sample_manifest,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_run_id=arguments.wandb_run_id,
        wandb_resume_must=wandb_resume_must,
        wandb_resume_never=wandb_resume_never,
        wandb_config=wandb_config,
    )
    result = None
    telemetry_exit_code = 2
    try:
        result = train_from_replay(
            model,
            snapshot_path=arguments.snapshot,
            config=config,
            output_dir=arguments.output_dir,
            max_steps=max_steps,
            adapter=adapter,
            device=device,
            resume_from=arguments.resume_from,
            expected_snapshot_file_sha256=arguments.expected_snapshot_file_sha256,
            heldout_snapshot_path=arguments.heldout_snapshot,
            artifact_root=arguments.artifact_root,
            save_every=save_every,
            telemetry=telemetry,
            validation_stop_at_step=arguments.validation_stop_at_step,
            learner_identity_sha256=arguments.learner_identity_sha256,
            runtime_container_sha256=arguments.runtime_container_sha256,
            learner_contract_sha256=arguments.learner_contract_sha256,
            dagger_beta=arguments.dagger_beta,
        )
        telemetry_exit_code = 0
    finally:
        # A canonical learner result is the recovery commit point.  Close and
        # flush the authenticated W&B run before publishing it so a process
        # killed during telemetry finalization can leave only restart state,
        # never a result that falsely implies telemetry completion.  Always
        # enter the distributed barrier even if rank-zero W&B finalization
        # raises, otherwise the other ranks can hang indefinitely.
        try:
            telemetry.close(exit_code=telemetry_exit_code)
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
    if result is None:
        raise LearnerError("learner returned without a training result")
    if rank == 0 and not arguments.no_publish_result:
        report = publish_learner_result(
            output_dir=arguments.output_dir, result=result, config=config
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    elif rank == 0:
        print(
            json.dumps(
                {
                    "completed_steps": result.completed_steps,
                    "target_steps": result.target_steps,
                    "validation_stopped": result.validation_stopped,
                    "restart_checkpoint": result.restart_checkpoint,
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
