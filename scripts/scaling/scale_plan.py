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
    trainer_nodes: int
    local_envs: int
    raytraced_mix: float
    generated_envs_capacity: int
    envs_per_renderer: int
    required_renderers: int
    transitions_per_update: int
    generated_requests_per_update_capacity: int
    reference_lag: int
    num_mini_batches: int
    local_mini_batch: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int


def build_plan(
    *,
    global_envs: int = 256,
    rollout_steps: int = 300,
    trainer_nodes: int = 8,
    raytraced_mix: float = 0.0,
    envs_per_renderer: int = 5,
    reference_lag: int = 256,
    num_mini_batches: int = 4,
    per_device_train_batch_size: int = 1,
) -> ScalePlan:
    values = {
        "global_envs": global_envs,
        "rollout_steps": rollout_steps,
        "trainer_nodes": trainer_nodes,
        "envs_per_renderer": envs_per_renderer,
        "reference_lag": reference_lag,
        "num_mini_batches": num_mini_batches,
        "per_device_train_batch_size": per_device_train_batch_size,
    }
    for name, value in values.items():
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not 0.0 <= raytraced_mix <= 1.0:
        raise ValueError(f"raytraced_mix must be in [0, 1], got {raytraced_mix}")
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
        trainer_nodes=trainer_nodes,
        local_envs=local_envs,
        raytraced_mix=raytraced_mix,
        generated_envs_capacity=generated_envs,
        envs_per_renderer=envs_per_renderer,
        required_renderers=required_renderers,
        transitions_per_update=transitions,
        generated_requests_per_update_capacity=generated_envs * rollout_steps,
        reference_lag=reference_lag,
        num_mini_batches=num_mini_batches,
        local_mini_batch=local_mini_batch,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=local_mini_batch // per_device_train_batch_size,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--trainer-nodes", type=int, default=8)
    parser.add_argument("--raytraced-mix", type=float, default=0.0)
    parser.add_argument("--envs-per-renderer", type=int, default=5)
    parser.add_argument("--reference-lag", type=int, default=256)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--format", choices=("json", "shell", "summary"), default="summary")
    parser.add_argument("--field", choices=tuple(ScalePlan.__annotations__), default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = build_plan(
            global_envs=args.global_envs,
            rollout_steps=args.rollout_steps,
            trainer_nodes=args.trainer_nodes,
            raytraced_mix=args.raytraced_mix,
            envs_per_renderer=args.envs_per_renderer,
            reference_lag=args.reference_lag,
            num_mini_batches=args.num_mini_batches,
            per_device_train_batch_size=args.per_device_train_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid scale plan: {exc}") from exc

    data = asdict(plan)
    if args.field:
        print(data[args.field])
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
            f"{plan.rollout_steps} steps = {plan.transitions_per_update:,} "
            f"transitions/update"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
