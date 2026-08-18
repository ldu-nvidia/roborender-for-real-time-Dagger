#!/usr/bin/env python3
"""Validate and summarize a distributed RoboRender DAgger production plan."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScalePlan:
    global_envs: int
    rollout_steps: int
    num_rollout_iterations: int
    trainer_nodes: int
    local_envs: int
    raytraced_mix: float
    generated_envs_capacity: int
    envs_per_renderer: int
    required_renderers: int
    transitions_per_rollout_iteration: int
    generated_requests_per_rollout_iteration_capacity: int
    reference_lag: int
    num_mini_batches: int
    local_mini_batch: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_learning_epochs: int
    optimizer_steps_per_rollout_iteration: int
    transitions_per_optimizer_step: int
    total_collected_transitions: int
    total_optimizer_steps: int
    behavior_source: str
    rollout_action_source: str
    beta_schedule: str
    beta_start: float
    beta_end: float
    rollout_with_teacher_num_steps: int
    enforce_teacher_rollout: bool
    teacher_rollout_ratio: float
    primary_view: str


def build_plan(
    *,
    global_envs: int = 256,
    rollout_steps: int = 300,
    num_rollout_iterations: int = 1,
    trainer_nodes: int = 8,
    raytraced_mix: float = 0.0,
    envs_per_renderer: int = 5,
    reference_lag: int = 256,
    num_mini_batches: int = 4,
    per_device_train_batch_size: int = 1,
    num_learning_epochs: int = 1,
    behavior_source: str = "student",
    teacher_rollout_ratio: float = 0.0,
    primary_view: str = "head_color",
) -> ScalePlan:
    values = {
        "global_envs": global_envs,
        "rollout_steps": rollout_steps,
        "num_rollout_iterations": num_rollout_iterations,
        "trainer_nodes": trainer_nodes,
        "envs_per_renderer": envs_per_renderer,
        "reference_lag": reference_lag,
        "num_mini_batches": num_mini_batches,
        "per_device_train_batch_size": per_device_train_batch_size,
        "num_learning_epochs": num_learning_epochs,
    }
    for name, value in values.items():
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not 0.0 <= raytraced_mix <= 1.0:
        raise ValueError(f"raytraced_mix must be in [0, 1], got {raytraced_mix}")
    if not 0.0 <= teacher_rollout_ratio <= 1.0:
        raise ValueError(
            "teacher_rollout_ratio must be in [0, 1], got "
            f"{teacher_rollout_ratio}"
        )
    if behavior_source not in {"student", "mixed", "teacher"}:
        raise ValueError(
            "behavior_source must be one of student, mixed, teacher, got "
            f"{behavior_source!r}"
        )
    if behavior_source == "student" and teacher_rollout_ratio != 0.0:
        raise ValueError("student behavior requires teacher_rollout_ratio=0")
    if behavior_source == "mixed" and not 0.0 < teacher_rollout_ratio < 1.0:
        raise ValueError("mixed behavior requires 0 < teacher_rollout_ratio < 1")
    if behavior_source == "teacher" and teacher_rollout_ratio != 1.0:
        raise ValueError("teacher behavior requires teacher_rollout_ratio=1")
    supported_primary_views = {"hand_left", "hand_right", "head_color"}
    if primary_view not in supported_primary_views:
        raise ValueError(
            f"primary_view must be one of {sorted(supported_primary_views)}, got "
            f"{primary_view!r}"
        )
    if global_envs % trainer_nodes:
        raise ValueError(
            f"global_envs ({global_envs}) must divide evenly across trainer_nodes "
            f"({trainer_nodes})"
        )
    local_envs = global_envs // trainer_nodes
    if local_envs % num_mini_batches:
        raise ValueError(
            f"local_envs ({local_envs}) must divide num_mini_batches ({num_mini_batches})"
        )
    local_mini_batch = local_envs // num_mini_batches
    if local_mini_batch % per_device_train_batch_size:
        raise ValueError(
            f"local mini-batch ({local_mini_batch}) must divide per-device batch "
            f"({per_device_train_batch_size})"
        )

    generated_envs = math.ceil(global_envs * (1.0 - raytraced_mix) - 1e-12)
    required_renderers = (
        math.ceil(generated_envs / envs_per_renderer) if generated_envs else 0
    )
    transitions = global_envs * rollout_steps
    return ScalePlan(
        global_envs=global_envs,
        rollout_steps=rollout_steps,
        num_rollout_iterations=num_rollout_iterations,
        trainer_nodes=trainer_nodes,
        local_envs=local_envs,
        raytraced_mix=raytraced_mix,
        generated_envs_capacity=generated_envs,
        envs_per_renderer=envs_per_renderer,
        required_renderers=required_renderers,
        transitions_per_rollout_iteration=transitions,
        generated_requests_per_rollout_iteration_capacity=generated_envs * rollout_steps,
        reference_lag=reference_lag,
        num_mini_batches=num_mini_batches,
        local_mini_batch=local_mini_batch,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=local_mini_batch // per_device_train_batch_size,
        num_learning_epochs=num_learning_epochs,
        optimizer_steps_per_rollout_iteration=num_mini_batches * num_learning_epochs,
        transitions_per_optimizer_step=(
            local_mini_batch * trainer_nodes * rollout_steps
        ),
        total_collected_transitions=transitions * num_rollout_iterations,
        total_optimizer_steps=(
            num_mini_batches * num_learning_epochs * num_rollout_iterations
        ),
        behavior_source=behavior_source,
        rollout_action_source={
            "student": "student",
            "mixed": "beta",
            "teacher": "teacher",
        }[behavior_source],
        beta_schedule="constant",
        beta_start=teacher_rollout_ratio,
        beta_end=teacher_rollout_ratio,
        rollout_with_teacher_num_steps=0,
        # Raven's videogen rollout_action_source is authoritative. Keep the
        # legacy parent-trainer mixer disabled to avoid applying two mixtures.
        enforce_teacher_rollout=False,
        teacher_rollout_ratio=teacher_rollout_ratio,
        primary_view=primary_view,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--num-rollout-iterations", type=int, default=1)
    parser.add_argument("--trainer-nodes", type=int, default=8)
    parser.add_argument("--raytraced-mix", type=float, default=0.0)
    parser.add_argument("--envs-per-renderer", type=int, default=5)
    parser.add_argument("--reference-lag", type=int, default=256)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--num-learning-epochs", type=int, default=1)
    parser.add_argument(
        "--behavior-source", choices=("student", "mixed", "teacher"), default="student"
    )
    parser.add_argument("--teacher-rollout-ratio", type=float, default=0.0)
    parser.add_argument(
        "--primary-view",
        choices=("hand_left", "hand_right", "head_color"),
        default="head_color",
    )
    parser.add_argument("--format", choices=("json", "shell", "summary"), default="summary")
    parser.add_argument("--field", choices=tuple(ScalePlan.__annotations__), default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = build_plan(
            global_envs=args.global_envs,
            rollout_steps=args.rollout_steps,
            num_rollout_iterations=args.num_rollout_iterations,
            trainer_nodes=args.trainer_nodes,
            raytraced_mix=args.raytraced_mix,
            envs_per_renderer=args.envs_per_renderer,
            reference_lag=args.reference_lag,
            num_mini_batches=args.num_mini_batches,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_learning_epochs=args.num_learning_epochs,
            behavior_source=args.behavior_source,
            teacher_rollout_ratio=args.teacher_rollout_ratio,
            primary_view=args.primary_view,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid scale plan: {exc}") from exc

    data = asdict(plan)
    if args.field:
        value = data[args.field]
        print(str(value).lower() if isinstance(value, bool) else value)
    elif args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.format == "shell":
        for key, value in data.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            print(f"{key.upper()}={rendered}")
    else:
        print(
            f"DAgger scale plan: {plan.global_envs} global envs = "
            f"{plan.trainer_nodes} ranks x {plan.local_envs}; "
            f"{plan.rollout_steps} steps = "
            f"{plan.transitions_per_rollout_iteration:,} transitions/rollout iteration"
        )
        print(
            f"Observation mix: {plan.raytraced_mix:.3f} ray-traced; capacity for "
            f"{plan.generated_envs_capacity} generated envs; "
            f"{plan.required_renderers} renderers at "
            f"{plan.envs_per_renderer} envs/renderer"
        )
        print(
            f"Rank-local optimizer: {plan.num_mini_batches} mini-batches, "
            f"local mini-batch {plan.local_mini_batch}, microbatch "
            f"{plan.per_device_train_batch_size}, accumulation "
            f"{plan.gradient_accumulation_steps}; RGB reference lag "
            f"{plan.reference_lag}"
        )
        print(
            f"Learning: {plan.num_learning_epochs} epoch(s), "
            f"{plan.optimizer_steps_per_rollout_iteration} optimizer steps/rollout "
            f"iteration, {plan.transitions_per_optimizer_step:,} transitions/optimizer "
            f"step; behavior={plan.behavior_source} "
            f"(videogen rollout source {plan.rollout_action_source}, "
            f"teacher ratio {plan.teacher_rollout_ratio:.3f}), "
            f"primary view={plan.primary_view}"
        )
        print(
            f"Run length: {plan.num_rollout_iterations} rollout iteration(s), "
            f"{plan.total_collected_transitions:,} collected transitions, "
            f"{plan.total_optimizer_steps} optimizer steps"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
