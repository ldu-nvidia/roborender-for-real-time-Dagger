#!/usr/bin/env python3
"""Run a tiny, production-shaped 8-rank DAgger learner smoke.

The smoke deliberately exercises the real immutable cohort, cumulative replay,
held-out replay, DDP learner, durable telemetry, restart, and candidate
publication paths.  It does not import Isaac, PhysX, or RoboRender, so it can be
run inside the learner-only Lepton H200 image.

One invocation runs an uninterrupted control and an interrupted/resumed learner
from the same sealed inputs.  The invocation passes only after the two final
states are bit-exact, the fixed 256-case held-out metrics improve, and telemetry
contains one unambiguous row at every expected optimizer frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger import artifacts, replay  # noqa: E402
from round_dagger.actor import runtime_state_dict_sha256  # noqa: E402
from round_dagger.learner import (  # noqa: E402
    BUNDLE_SCHEMA,
    CUDA_CUBLAS_WORKSPACE_CONFIG,
    HELDOUT_TELEMETRY_SCHEMA,
    TELEMETRY_SCHEMA,
    LearnerBundleConfig,
)
from scripts.round_dagger.smoke_mini_round import _portable_spec  # noqa: E402
from scripts.round_dagger.verify_learner_restart_equivalence import (  # noqa: E402
    verify as verify_restart_equivalence,
)


REPORT_SCHEMA = "fast_roborender.round_dagger.lepton_h200_learner_smoke.v1"
FIXTURE_SCHEMA = "fast_roborender.round_dagger.lepton_h200_smoke_fixture.v1"
CAMPAIGN_ID = "lepton-h200-learner-smoke"
VIEW_SHAPE = (6, 2, 3)
ACTION_DIM = 2
DEFAULT_WORLD_SIZE = 8
DEFAULT_HELDOUT_CASES = 256
_SAFE_WANDB_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


class SmokeError(RuntimeError):
    """A smoke input, subprocess, or semantic validation failed closed."""


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class TinyPolicy(nn.Module):
    """Small linear policy with the production observation mapping boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def forward(self, observations: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb = observations["vision_obs"].mean(dim=(1, 2))
        actor = observations["actor_obs"].float()
        return {"actions": self.linear(torch.cat((rgb, actor), dim=-1))}


def create_tiny_policy(_bundle: Mapping[str, object]) -> nn.Module:
    """Importable model factory consumed by the production learner entrypoint."""

    return TinyPolicy()


def _atomic_torch_save(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(document), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _rollout_arrays(
    global_env_ids: Sequence[int], *, steps: int
) -> dict[str, np.ndarray]:
    """Create a deterministic learnable fixture with bounded train/eval features."""

    env_ids = np.asarray(global_env_ids, dtype=np.int64)
    if env_ids.ndim != 1 or env_ids.size == 0 or steps < 1:
        raise SmokeError("rollout fixture needs non-empty environments and steps")
    environments = int(env_ids.size)
    rgb = np.empty((steps, environments, *VIEW_SHAPE), dtype=np.uint8)
    actor = np.empty((steps, environments, 1), dtype=np.float32)
    expert = np.empty((steps, environments, ACTION_DIM), dtype=np.float32)
    for step in range(steps):
        for local_index, global_env_id in enumerate(env_ids.tolist()):
            channels = np.asarray(
                [
                    (17 * step + 31 * global_env_id + 3) % 256,
                    (43 * step + 11 * global_env_id + 7) % 256,
                    (71 * step + 19 * global_env_id + 13) % 256,
                ],
                dtype=np.uint8,
            )
            rgb[step, local_index] = channels
            actor_value = ((global_env_id % 17) - 8) / 8.0 + step / max(steps, 2)
            actor[step, local_index, 0] = np.float32(actor_value)
            normalized = (channels.astype(np.float32) / 255.0 - 0.5) / 0.25
            expert[step, local_index, 0] = np.float32(
                normalized @ np.asarray([0.35, -0.2, 0.5], dtype=np.float32)
                + 0.75 * actor_value
            )
            # This fixed suffix proves that train_action_slice is honored.
            expert[step, local_index, 1] = np.float32(1000.0 + step)
    zeros = np.zeros((steps, environments), dtype=np.bool_)
    episode_ids = np.broadcast_to(env_ids[None, :], (steps, environments)).copy()
    episode_steps = np.broadcast_to(
        np.arange(steps, dtype=np.int64)[:, None], (steps, environments)
    ).copy()
    return {
        "policy_rgb": rgb,
        "expert_actions": expert,
        "student_actions": np.zeros_like(expert),
        "behavior_actions": expert.copy(),
        "expert_mask": np.ones((steps, environments), dtype=np.bool_),
        "dones": zeros.copy(),
        "observation_is_raytraced": zeros.copy(),
        "global_env_ids": env_ids,
        "episode_ids": episode_ids,
        "episode_steps": episode_steps,
        "obs__actor_obs": actor,
    }


def _cohort_common(*, purpose: str, cohort_index: int) -> dict[str, object]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "macro_round": 0,
        "cohort_index": cohort_index,
        "purpose": purpose,
        "policy_sha256": _sha("smoke-parent-policy"),
        "runtime_policy_sha256": _sha("smoke-parent-runtime"),
        "teacher_sha256": _sha("smoke-teacher"),
        "renderer_sha256": _sha("smoke-renderer"),
        "beta_sha256": _sha("smoke-beta"),
        "config_sha256": _sha("smoke-actor-config"),
    }


def _publish_cohort(
    path: Path,
    *,
    purpose: str,
    cohort_index: int,
    environment_groups: Sequence[Sequence[int]],
    steps: int,
) -> dict[str, object]:
    world_size = len(environment_groups)
    all_environment_ids = [item for group in environment_groups for item in group]
    common = _cohort_common(purpose=purpose, cohort_index=cohort_index)
    artifacts.create_cohort(
        path,
        expected_ranks=list(range(world_size)),
        expected_global_env_ids=all_environment_ids,
        expected_transitions=len(all_environment_ids) * steps,
        view_contract={
            "dtype": "uint8",
            "layout": "vertical",
            "shape": list(VIEW_SHAPE),
            "views": ["ext1", "ext2", "wrist"],
        },
        action_contract={"dtype": "float32", "shape": [ACTION_DIM]},
        **common,
    )
    for rank, environment_ids in enumerate(environment_groups):
        artifacts.publish_rollout_shard(
            path,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=_rollout_arrays(environment_ids, steps=steps),
            **common,
        )
    return artifacts.seal_cohort(path)


def prepare_fixture(
    output_root: Path,
    *,
    world_size: int,
    heldout_cases: int,
    max_steps: int,
    save_every: int,
    heldout_eval_interval: int,
) -> dict[str, object]:
    """Publish all immutable inputs and return their absolute identities."""

    if world_size < 1:
        raise SmokeError("world_size must be positive")
    if heldout_cases < world_size or heldout_cases % world_size:
        raise SmokeError("heldout_cases must be a positive multiple of world_size")
    if max_steps < 2 or not 1 <= save_every <= max_steps:
        raise SmokeError("max_steps/save_every are invalid")
    if not 1 <= heldout_eval_interval <= max_steps:
        raise SmokeError("heldout_eval_interval is invalid")

    fixture_root = output_root / "fixture"
    fixture_root.mkdir(parents=True, exist_ok=False)
    train_groups = [
        list(range(rank * 4, rank * 4 + 4)) for rank in range(world_size)
    ]
    train_cohort = fixture_root / "train_cohort"
    _publish_cohort(
        train_cohort,
        purpose="train",
        cohort_index=0,
        environment_groups=train_groups,
        steps=4,
    )
    snapshot_path = fixture_root / "training_snapshot.json"
    with replay.ReplayLedger(fixture_root / "replay.sqlite3") as ledger:
        ledger.ingest_cohort(train_cohort)
        snapshot = ledger.build_snapshot(
            CAMPAIGN_ID,
            through_macro_round=0,
            expected_train_cohorts=1,
            output=snapshot_path,
        )

    cases_per_rank = heldout_cases // world_size
    heldout_ids = list(range(10_000, 10_000 + heldout_cases))
    heldout_groups = [
        heldout_ids[index * cases_per_rank : (index + 1) * cases_per_rank]
        for index in range(world_size)
    ]
    heldout_cohort = fixture_root / "heldout_cohort"
    _publish_cohort(
        heldout_cohort,
        purpose="eval",
        cohort_index=1,
        environment_groups=heldout_groups,
        steps=1,
    )
    heldout_snapshot_path = fixture_root / "heldout_snapshot.json"
    heldout = replay.build_heldout_snapshot(
        [heldout_cohort], output=heldout_snapshot_path
    )
    if heldout["transition_count"] != heldout_cases:
        raise SmokeError("held-out snapshot transition count changed")

    bootstrap = TinyPolicy()
    bootstrap_path = fixture_root / "bootstrap.pt"
    _atomic_torch_save(
        bootstrap_path, {"policy_state_dict": bootstrap.state_dict()}
    )
    bootstrap_sha256 = artifacts.sha256_file(bootstrap_path)
    policy_state_sha256 = runtime_state_dict_sha256(bootstrap.state_dict())
    portable_spec = _portable_spec(bootstrap_sha256, policy_state_sha256)

    config = LearnerBundleConfig.from_mapping(
        {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "macro_round": 0,
            "model_lineage_sha256": bootstrap_sha256,
            "code_sha256": _sha("lepton-h200-smoke-learner-code-v1"),
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.25, 0.25, 0.25],
            "rgb_input_key": "vision_obs",
            "rgb_layout": "NHWC",
            "output_key": "actions",
            "train_action_slice": [0, 1],
            "batch_size": 8,
            "max_steps": max_steps,
            "save_every": save_every,
            "learning_rate": 0.03,
            "weight_decay": 0.0,
            "seed": 1701,
            "partition_seed": 2301,
            "initial_checkpoint_sha256": bootstrap_sha256,
            "heldout_snapshot_sha256": heldout["snapshot_sha256"],
            "heldout_snapshot_file_sha256": heldout["file_sha256"],
            "heldout_eval_interval": heldout_eval_interval,
            "model_config": portable_spec,
        }
    )
    bundle_path = fixture_root / "learner_bundle.json"
    artifacts.publish_immutable_json(bundle_path, config.to_mapping())

    document: dict[str, object] = {
        "schema": FIXTURE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "world_size": world_size,
        "training_shards": world_size,
        "training_transitions": snapshot["transition_count"],
        "heldout_shards": world_size,
        "heldout_cases": heldout_cases,
        "training_snapshot": str(snapshot_path.resolve()),
        "training_snapshot_file_sha256": snapshot["file_sha256"],
        "training_snapshot_sha256": snapshot["snapshot_sha256"],
        "heldout_snapshot": str(heldout_snapshot_path.resolve()),
        "heldout_snapshot_file_sha256": heldout["file_sha256"],
        "heldout_snapshot_sha256": heldout["snapshot_sha256"],
        "bootstrap_checkpoint": str(bootstrap_path.resolve()),
        "bootstrap_checkpoint_sha256": bootstrap_sha256,
        "bundle": str(bundle_path.resolve()),
        "bundle_sha256": config.sha256,
        "max_steps": max_steps,
        "save_every": save_every,
        "heldout_eval_interval": heldout_eval_interval,
    }
    manifest_path = fixture_root / "fixture_manifest.json"
    artifacts.publish_immutable_json(manifest_path, document)
    document["manifest_path"] = str(manifest_path.resolve())
    document["manifest_file_sha256"] = artifacts.sha256_file(manifest_path)
    return document


def _run_phase(
    *,
    name: str,
    fixture: Mapping[str, object],
    output_root: Path,
    world_size: int,
    device: str,
    resume_from: Path | None = None,
    stop_at_step: int | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    wandb_resume_never: bool = False,
) -> None:
    phase_dir = output_root / name
    phase_dir.mkdir(parents=True, exist_ok=resume_from is not None)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(REPOSITORY_ROOT / "scripts/round_dagger/train_learner.py"),
        "--bundle",
        str(fixture["bundle"]),
        "--snapshot",
        str(fixture["training_snapshot"]),
        "--heldout-snapshot",
        str(fixture["heldout_snapshot"]),
        "--output-dir",
        str(phase_dir),
        "--model-factory",
        "scripts.round_dagger.lepton_h200_learner_smoke:create_tiny_policy",
        "--initial-checkpoint",
        str(fixture["bootstrap_checkpoint"]),
        "--expected-snapshot-file-sha256",
        str(fixture["training_snapshot_file_sha256"]),
        "--heldout-eval-interval",
        str(fixture["heldout_eval_interval"]),
        "--device",
        device,
        "--telemetry-jsonl",
        str(phase_dir / "telemetry.jsonl"),
        "--heldout-telemetry-jsonl",
        str(phase_dir / "heldout_telemetry.jsonl"),
    ]
    if resume_from is not None:
        command.extend(("--resume-from", str(resume_from)))
    if stop_at_step is not None:
        command.extend(
            (
                "--validation-stop-at-step",
                str(stop_at_step),
                "--no-publish-result",
            )
        )
    if wandb_project is not None:
        command.extend(("--wandb-project", wandb_project))
        if wandb_entity is not None:
            command.extend(("--wandb-entity", wandb_entity))
        assert wandb_run_id is not None
        command.extend(("--wandb-run-id", wandb_run_id))
        if wandb_resume_never:
            command.append("--wandb-resume-never")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH", ""))
        if item
    )
    if device.startswith("cuda"):
        environment["CUBLAS_WORKSPACE_CONFIG"] = CUDA_CUBLAS_WORKSPACE_CONFIG
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _atomic_bytes(phase_dir / f"{name}.stdout.log", completed.stdout)
    _atomic_bytes(phase_dir / f"{name}.stderr.log", completed.stderr)
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise SmokeError(
            f"{name} learner exited {completed.returncode}; stderr tail:\n{tail}"
        )


def _jsonl(path: Path, *, schema: str) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"missing or unsafe telemetry: {path}")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise SmokeError(f"empty or torn telemetry: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except Exception as exc:
            raise SmokeError(f"invalid telemetry line {line_number}: {exc}") from exc
        if not isinstance(row, dict) or row.get("schema") != schema:
            raise SmokeError(f"telemetry schema mismatch at line {line_number}")
        rows.append(row)
    return rows


def _validate_telemetry(
    path: Path,
    heldout_path: Path,
    *,
    max_steps: int,
    eval_interval: int,
    world_size: int,
    heldout_cases: int,
) -> dict[str, object]:
    train_rows = _jsonl(path, schema=TELEMETRY_SCHEMA)
    train_steps = [int(row["optimizer_step"]) for row in train_rows]
    if train_steps != list(range(1, max_steps + 1)):
        raise SmokeError("training telemetry is not exactly optimizer steps 1..N")
    if any(int(row["world_size"]) != world_size for row in train_rows):
        raise SmokeError("training telemetry world size changed")
    losses = [float(row["loss"]) for row in train_rows]
    if any(not math.isfinite(value) for value in losses) or len(set(losses)) < 2:
        raise SmokeError("training loss is non-finite or did not change")

    heldout_rows = _jsonl(heldout_path, schema=HELDOUT_TELEMETRY_SCHEMA)
    expected_steps = [0]
    expected_steps.extend(range(eval_interval, max_steps + 1, eval_interval))
    if expected_steps[-1] != max_steps:
        expected_steps.append(max_steps)
    heldout_steps = [int(row["optimizer_step"]) for row in heldout_rows]
    if heldout_steps != expected_steps:
        raise SmokeError(
            f"held-out telemetry frontiers differ: {heldout_steps} != {expected_steps}"
        )
    if any(
        int(row["heldout_transitions_global"]) != heldout_cases
        for row in heldout_rows
    ):
        raise SmokeError("held-out telemetry does not cover the fixed cohort")
    heldout_mse = [float(row["action_mse"]) for row in heldout_rows]
    if (
        any(not math.isfinite(value) for value in heldout_mse)
        or len(set(heldout_mse)) < 2
        or heldout_mse[-1] >= heldout_mse[0]
    ):
        raise SmokeError("held-out student quality did not measurably improve")
    return {
        "train_rows": len(train_rows),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "heldout_rows": len(heldout_rows),
        "heldout_action_mse_first": heldout_mse[0],
        "heldout_action_mse_last": heldout_mse[-1],
        "heldout_relative_mse_reduction": (
            heldout_mse[0] - heldout_mse[-1]
        )
        / heldout_mse[0],
    }


def run(
    output_root: Path,
    *,
    world_size: int = DEFAULT_WORLD_SIZE,
    device: str = "cuda",
    heldout_cases: int = DEFAULT_HELDOUT_CASES,
    max_steps: int = 6,
    split_step: int = 3,
    save_every: int = 1,
    heldout_eval_interval: int = 1,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
) -> dict[str, object]:
    output_root = Path(os.path.abspath(output_root.expanduser()))
    if output_root.exists() or output_root.is_symlink():
        raise SmokeError(f"refusing to reuse smoke output root: {output_root}")
    if not 1 <= split_step < max_steps:
        raise SmokeError("split_step must be in [1, max_steps)")
    if wandb_project is None and (wandb_entity is not None or wandb_run_id is not None):
        raise SmokeError("W&B entity/run ID requires a W&B project")
    if wandb_project is not None and not wandb_project.strip():
        raise SmokeError("W&B project must be non-empty")
    if wandb_run_id is not None and _SAFE_WANDB_ID.fullmatch(wandb_run_id) is None:
        raise SmokeError("W&B run ID is not a safe canonical identifier")
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise SmokeError("smoke output parent must not be a symlink")
    os.mkdir(output_root, 0o750)

    fixture = prepare_fixture(
        output_root,
        world_size=world_size,
        heldout_cases=heldout_cases,
        max_steps=max_steps,
        save_every=save_every,
        heldout_eval_interval=heldout_eval_interval,
    )
    canonical_run_id = wandb_run_id
    if wandb_project is not None and canonical_run_id is None:
        canonical_run_id = f"h200-smoke-{str(fixture['bundle_sha256'])[:24]}"

    _run_phase(
        name="uninterrupted",
        fixture=fixture,
        output_root=output_root,
        world_size=world_size,
        device=device,
    )
    _run_phase(
        name="resumed",
        fixture=fixture,
        output_root=output_root,
        world_size=world_size,
        device=device,
        stop_at_step=split_step,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_id=canonical_run_id,
        wandb_resume_never=wandb_project is not None,
    )
    resumed_restart = output_root / "resumed/restart.pt"
    _run_phase(
        name="resumed",
        fixture=fixture,
        output_root=output_root,
        world_size=world_size,
        device=device,
        resume_from=resumed_restart,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_id=canonical_run_id,
    )

    result_name = "learner_result_round_000000.json"
    equivalence = verify_restart_equivalence(
        uninterrupted_checkpoint=output_root / "uninterrupted/restart.pt",
        resumed_checkpoint=resumed_restart,
        expected_completed_steps=max_steps,
        uninterrupted_result=output_root / "uninterrupted" / result_name,
        resumed_result=output_root / "resumed" / result_name,
    )
    equivalence_path = output_root / "restart_equivalence.json"
    equivalence_sha256 = artifacts.publish_immutable_json(
        equivalence_path, equivalence
    )
    uninterrupted_quality = _validate_telemetry(
        output_root / "uninterrupted/telemetry.jsonl",
        output_root / "uninterrupted/heldout_telemetry.jsonl",
        max_steps=max_steps,
        eval_interval=heldout_eval_interval,
        world_size=world_size,
        heldout_cases=heldout_cases,
    )
    resumed_quality = _validate_telemetry(
        output_root / "resumed/telemetry.jsonl",
        output_root / "resumed/heldout_telemetry.jsonl",
        max_steps=max_steps,
        eval_interval=heldout_eval_interval,
        world_size=world_size,
        heldout_cases=heldout_cases,
    )
    if uninterrupted_quality != resumed_quality:
        raise SmokeError("resumed quality telemetry differs from uninterrupted control")

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "world_size": world_size,
        "device": device,
        "max_steps": max_steps,
        "split_step": split_step,
        "training_transitions": fixture["training_transitions"],
        "heldout_cases": heldout_cases,
        "heldout_snapshot_sha256": fixture["heldout_snapshot_sha256"],
        "restart_semantic_equivalent": equivalence["semantic_equivalent"],
        "restart_results_equivalent": equivalence["results_equivalent"],
        "restart_equivalence_sha256": equivalence_sha256,
        "student_quality": resumed_quality,
        "wandb_enabled": wandb_project is not None,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_run_id": canonical_run_id,
        "completed_at_unix": int(time.time()),
    }
    report["report_sha256"] = artifacts.canonical_json_sha256(report)
    artifacts.publish_immutable_json(output_root / "smoke_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--heldout-cases", type=int, default=DEFAULT_HELDOUT_CASES)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--split-step", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--heldout-eval-interval", type=int, default=1)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run(
        arguments.output_root,
        world_size=arguments.world_size,
        device=arguments.device,
        heldout_cases=arguments.heldout_cases,
        max_steps=arguments.max_steps,
        split_step=arguments.split_step,
        save_every=arguments.save_every,
        heldout_eval_interval=arguments.heldout_eval_interval,
        wandb_project=arguments.wandb_project,
        wandb_entity=arguments.wandb_entity,
        wandb_run_id=arguments.wandb_run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as exc:
        print(f"Lepton H200 learner smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
