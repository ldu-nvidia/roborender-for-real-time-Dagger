#!/usr/bin/env python3
"""Compose and validate the exact three-view production DAgger configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


EXPECTED_VIEWS = ("head_color", "hand_left", "hand_right")
EXPECTED_POLICY_SHAPE = (324, 192, 3)


def resolve_git_head(repo: Path) -> str:
    """Resolve HEAD without requiring a git executable in the Isaac image."""

    git_dir = repo / ".git"
    if not git_dir.exists():
        marker = repo / ".source_revision"
        if marker.is_file():
            revision = marker.read_text(encoding="utf-8").strip()
            if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
                return revision
            raise RuntimeError(f"malformed GR00T source revision marker: {marker}")
        raise RuntimeError(f"GR00T checkout has neither .git nor .source_revision: {repo}")
    if git_dir.is_file():
        value = git_dir.read_text(encoding="utf-8").strip()
        if not value.startswith("gitdir: "):
            raise RuntimeError("malformed GR00T .git file")
        git_dir = (repo / value.removeprefix("gitdir: ")).resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == ref:
                    return revision
    raise RuntimeError(f"cannot resolve GR00T HEAD ref {ref}")


def production_overrides() -> list[str]:
    """Mirror the exact launcher values while using inert placeholder paths."""

    return [
        "+exp=loco_manip/wsdpt_image_dagger_3view",
        "num_envs=32",
        "algo.config.num_steps_per_env=300",
        "algo.config.num_mini_batches=1",
        "algo.config.num_learning_epochs=1",
        "algo.config.rollout_with_teacher_num_steps=0",
        "algo.config.enforce_teacher_rollout=true",
        "algo.config.ratio_teacher_rollout=0.0",
        "algo.config.value_loss_coef=0.0",
        "algo.config.entropy_coef=0.0",
        "algo.config.videogen.rollout_action_source=beta",
        "++algo.config.videogen.beta_schedule=linear",
        "++algo.config.videogen.beta_start=1.0",
        "++algo.config.videogen.beta_end=0.0",
        "++algo.config.videogen.beta_decay_iters=500",
        "algo.trl.per_device_train_batch_size=1",
        "algo.trl.gradient_accumulation_steps=32",
        "algo.trl.num_total_batches=1",
        "algo.trl.total_episodes=null",
        "callbacks.model_save.save_frequency=1",
        "callbacks.model_save.save_initial_checkpoint=true",
        "headless=true",
        "checkpoint=null",
        "teacher_actor_path=/validation/expert.pt",
        "algo.config.videogen.raytraced_mix_prob=0.0",
        "++algo.config.videogen.seed=42",
        "++algo.config.videogen.reference_lag=0",
        "++algo.config.videogen.primary_view=head_color",
        "simulator.config.cameras.policy_views=[head_color,hand_left,hand_right]",
        "policy_view_count=3",
        "++env.config.image_dagger_policy_view_count=3",
        "algo.config.videogen.image3f_endpoint_files=[/validation/endpoint.txt]",
        "algo.config.videogen.anchor_pool_dir=/validation/anchors",
        "algo.config.videogen.anchor_index=0",
        "simulator.config.render_results=false",
        "use_wandb=true",
        "wandb.wandb_entity=ldu",
        "wandb.wandb_group=null",
        "project_name=roborender_dagger_config_validation",
        "experiment_name=production_contract",
        "experiment_dir=/tmp/roborender-production-contract",
        "hydra.run.dir=/tmp/roborender-production-contract/.hydra/rank_0",
        "output_dir=/tmp/roborender-production-contract/output/rank_0",
    ]


def compose_and_validate(gr00t_root: Path, expected_revision: str | None) -> dict[str, object]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    revision = resolve_git_head(gr00t_root)
    if expected_revision and revision != expected_revision:
        raise RuntimeError(
            f"GR00T revision {revision} does not match expected {expected_revision}"
        )

    resolvers = {
        "eval": eval,
        "if": lambda pred, a, b: a if pred else b,
        "eq": lambda x, y: x.lower() == y.lower(),
        "sqrt": lambda x: math.sqrt(float(x)),
        "sum": lambda x: sum(x),
        "ceil": lambda x: math.ceil(x),
        "int": lambda x: int(x),
        "len": lambda x: len(x),
        "sum_list": lambda values: sum(values),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)

    config_dir = gr00t_root / "gr00t/rl/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="base", overrides=production_overrides())

    views = tuple(str(value) for value in cfg.simulator.config.cameras.policy_views)
    renderer_views = tuple(str(value) for value in cfg.algo.config.videogen.view_names)
    count = int(cfg.policy_view_count)
    camera_hw = tuple(int(value) for value in cfg.simulator.config.cameras.camera_resolutions)
    shape = (camera_hw[0] * count, camera_hw[1], 3)
    checks = {
        "policy_views": views == EXPECTED_VIEWS,
        "renderer_views": renderer_views == EXPECTED_VIEWS,
        "policy_view_count": count == 3,
        "trainer_policy_view_count": int(cfg.env.config.image_dagger_policy_view_count)
        == count,
        "policy_view_layout": cfg.simulator.config.cameras.policy_view_layout == "vertical",
        "policy_image_shape": shape == EXPECTED_POLICY_SHAPE,
        "rollout_horizon": int(cfg.algo.config.num_steps_per_env) == 300,
        "pure_bc": float(cfg.algo.config.value_loss_coef) == 0.0
        and float(cfg.algo.config.entropy_coef) == 0.0,
        "beta_behavior": cfg.algo.config.videogen.rollout_action_source == "beta"
        and cfg.algo.config.videogen.beta_schedule == "linear"
        and float(cfg.algo.config.videogen.beta_start) == 1.0
        and float(cfg.algo.config.videogen.beta_end) == 0.0
        and int(cfg.algo.config.videogen.beta_decay_iters) == 500,
        "ft_fixed": int(cfg.algo.config.videogen.reference_lag) == 0,
        "generated_only": float(cfg.algo.config.videogen.raytraced_mix_prob) == 0.0,
        "batch_contract": int(cfg.algo.config.num_mini_batches) == 1
        and int(cfg.algo.config.num_learning_epochs) == 1
        and int(cfg.algo.trl.per_device_train_batch_size) == 1
        and int(cfg.algo.trl.gradient_accumulation_steps) == 32,
        "fresh_student": cfg.checkpoint is None,
        "initial_checkpoint": bool(cfg.callbacks.model_save.save_initial_checkpoint),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("production Hydra contract failed: " + ", ".join(failed))
    return {
        "schema": "fast_roborender.production_hydra_contract.v1",
        "status": "validated",
        "gr00t_revision": revision,
        "config": "+exp=loco_manip/wsdpt_image_dagger_3view",
        "policy_views": list(views),
        "renderer_views": list(renderer_views),
        "policy_image_shape": list(shape),
        "checks": checks,
    }


def atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-root", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = compose_and_validate(args.gr00t_root.resolve(), args.expected_revision)
    if args.output:
        atomic_json(args.output, report)
    if args.json or not args.output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
