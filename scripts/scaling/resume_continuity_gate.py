#!/usr/bin/env python3
"""Snapshot and verify a true new-process DAgger checkpoint resume.

The gate is deliberately artifact-based: take one snapshot after the first
process has drained W&B, restart from that checkpoint in the *same* run
directory, take a second snapshot, then verify both.  Snapshotting requires
the sealed trainer environment (PyTorch and PyYAML); verification uses only
the Python standard library and is therefore suitable for CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA = "fast_roborender.resume_continuity.snapshot.v1"
REPORT_SCHEMA = "fast_roborender.resume_continuity.report.v1"
RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class GateError(RuntimeError):
    """A resume artifact violated the production contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_checkpoint(
    torch: Any,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Hash and deserialize a checkpoint through the same safe descriptor."""

    expected = None if expected_sha256 in (None, "", "NONE") else expected_sha256
    if expected is not None and (
        not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None
    ):
        raise GateError("expected checkpoint SHA-256 is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GateError(f"cannot safely open checkpoint {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GateError(f"checkpoint is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if expected is not None and actual_sha256 != expected:
                raise GateError(
                    "checkpoint SHA-256 mismatch at actual load "
                    f"({actual_sha256} != {expected})"
                )
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise GateError("checkpoint payload is not a dictionary")
    return payload, actual_sha256


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state_step(state: object) -> int:
    value = state.get("global_step") if isinstance(state, dict) else getattr(
        state, "global_step", None
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError("checkpoint state has no numeric global_step")
    step = int(value)
    if step < 0 or float(value) != step:
        raise GateError(f"checkpoint global_step is invalid: {value!r}")
    return step


def _policy_digest(torch: Any, state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise GateError("checkpoint policy state is empty or malformed")
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(state.items(), key=lambda item: str(item[0])):
        if not torch.is_tensor(tensor):
            raise GateError(f"policy state {name!r} is not a tensor")
        cpu = tensor.detach().contiguous().to(device="cpu")
        if (cpu.is_floating_point() or cpu.is_complex()) and not bool(
            torch.isfinite(cpu).all().item()
        ):
            raise GateError(f"policy state {name!r} contains NaN or infinity")
        metadata = json.dumps(
            {"name": str(name), "dtype": str(cpu.dtype), "shape": list(cpu.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise GateError("checkpoint policy has no parameters")
    return digest.hexdigest()


def _policy_schema_digest(
    torch: Any, state: Mapping[str, Any], names: list[str]
) -> str:
    """Digest exact ordered parameter names, dtypes, and shapes."""

    digest = hashlib.sha256()
    for name in names:
        tensor = state.get(name)
        if not torch.is_tensor(tensor):
            raise GateError(f"policy parameter {name!r} is not a tensor")
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
    return digest.hexdigest()


def _validate_resumable_state(payload: Mapping[str, Any], checkpoint_step: int = 0) -> None:
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping):
        raise GateError("checkpoint optimizer state is malformed")
    if not isinstance(optimizer.get("state"), Mapping):
        raise GateError("checkpoint optimizer state mapping is malformed")
    if checkpoint_step > 0 and not optimizer["state"]:
        raise GateError("checkpoint optimizer has no state entries after step zero")
    param_groups = optimizer.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise GateError("checkpoint optimizer has no parameter groups")
    if any(not isinstance(group, Mapping) for group in param_groups):
        raise GateError("checkpoint optimizer parameter groups are malformed")
    scheduler = payload.get("lr_scheduler_state_dict")
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise GateError("checkpoint scheduler state is malformed")
    if not isinstance(payload.get("env_state_dict"), Mapping):
        raise GateError("checkpoint environment state is malformed")


def _read_run_id(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{label} is missing or unsafe: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if RUN_ID_RE.fullmatch(value) is None:
        raise GateError(f"{label} is malformed")
    return value


def _telemetry_files(run_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted((run_dir / "telemetry" / "scalars").glob("step_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError(f"invalid telemetry scalar {path.name}: {type(exc).__name__}")
        step = payload.get("step") if isinstance(payload, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise GateError(f"telemetry scalar has invalid step: {path.name}")
        key = str(step)
        if key in output:
            raise GateError(f"duplicate telemetry scalar step {step}")
        output[key] = _sha256_file(path)
    if not output:
        raise GateError("run has no scalar telemetry artifacts")
    return output


def snapshot(
    checkpoint: Path,
    run_dir: Path,
    expected_world_size: int,
    *,
    require_wandb: bool = True,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        import torch
        import yaml
    except ImportError as exc:
        raise GateError(
            "snapshot requires the sealed trainer environment with torch and PyYAML"
        ) from exc

    checkpoint = Path(os.path.abspath(checkpoint.expanduser()))
    run_dir = run_dir.expanduser().resolve()
    if checkpoint.parent.resolve() != run_dir:
        raise GateError("strict resume checkpoint must be inside the reused run directory")
    payload, checkpoint_sha256 = _load_torch_checkpoint(
        torch, checkpoint, expected_sha256=expected_checkpoint_sha256
    )
    if payload.get("checkpoint_schema") != "gr00t.trl.full_state.v2":
        raise GateError("checkpoint is not an atomic full-state v2 checkpoint")
    required = {
        "policy_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "state",
        "args",
        "env_state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise GateError(f"checkpoint is missing strict-resume fields: {missing}")
    step = _state_step(payload["state"])
    _validate_resumable_state(payload, step)
    saved_args = payload["args"]
    absolute_target = (
        saved_args.get("num_total_batches")
        if isinstance(saved_args, dict)
        else getattr(saved_args, "num_total_batches", None)
    )
    if (
        isinstance(absolute_target, bool)
        or not isinstance(absolute_target, int)
        or absolute_target < step
    ):
        raise GateError("checkpoint has no valid absolute training target")

    trainer_run_id = None
    sidecar_run_id = None
    upload_state: dict[str, Any] = {}
    if require_wandb:
        meta_path = run_dir / "meta.yaml"
        if not meta_path.is_file() or meta_path.is_symlink():
            raise GateError("trainer W&B metadata is missing or unsafe")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)
        trainer_run_id = meta.get("wandb_run") if isinstance(meta, dict) else None
        if (
            not isinstance(trainer_run_id, str)
            or RUN_ID_RE.fullmatch(trainer_run_id) is None
        ):
            raise GateError("meta.yaml has no valid trainer W&B run ID")
        sidecar_run_id = _read_run_id(
            run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID", "media W&B run ID"
        )
        state_path = run_dir / "telemetry" / ".wandb_upload_state.json"
        try:
            decoded_upload_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError(f"invalid W&B upload state: {type(exc).__name__}")
        if not isinstance(decoded_upload_state, dict):
            raise GateError("W&B upload state is not a JSON object")
        upload_state = decoded_upload_state
        if upload_state.get("wandb_run_id") != sidecar_run_id:
            raise GateError("W&B upload state is not bound to the media run ID")

    runtime_summaries = []
    runtime_rollout_steps: set[int] = set()
    runtime_local_envs: set[int] = set()
    for rank in range(expected_world_size):
        path = (
            run_dir
            / "resume_state"
            / f"rank_{rank:04d}_step_{step:09d}.pt"
        )
        if not path.is_file() or path.is_symlink():
            raise GateError(f"rank {rank} runtime resume state is missing")
        runtime, runtime_sha256 = _load_torch_checkpoint(torch, path)
        expected = {
            "schema_version": 1,
            "checkpoint_step": step,
            "rank": rank,
            "world_size": expected_world_size,
            "beta_iteration": step,
        }
        typed_contract = {
            "schema_version": 0,
            "checkpoint_step": 0,
            "rank": 0,
            "world_size": 1,
            "beta_iteration": 0,
            "telemetry_step": 0,
            "rollout_steps": 1,
            "local_num_envs": 1,
        }
        for key, minimum in typed_contract.items():
            value = runtime.get(key)
            if type(value) is not int or value < minimum:
                raise GateError(
                    f"rank {rank} runtime {key} must be an integer >= {minimum}"
                )
        mismatch = {
            key: (runtime.get(key), value)
            for key, value in expected.items()
            if runtime.get(key) != value
        }
        if mismatch:
            raise GateError(f"rank {rank} runtime resume state mismatch: {mismatch}")
        if runtime["telemetry_step"] != step * runtime["rollout_steps"]:
            raise GateError(
                f"rank {rank} runtime telemetry is not checkpoint-aligned"
            )
        runtime_rollout_steps.add(runtime["rollout_steps"])
        runtime_local_envs.add(runtime["local_num_envs"])
        runtime_summaries.append(
            {
                "rank": rank,
                "sha256": runtime_sha256,
                "telemetry_step": int(runtime["telemetry_step"]),
                "rollout_steps": int(runtime["rollout_steps"]),
                "local_num_envs": int(runtime["local_num_envs"]),
            }
        )
    if len(runtime_rollout_steps) != 1:
        raise GateError("per-rank runtime rollout lengths are inconsistent")
    if len(runtime_local_envs) != 1:
        raise GateError("per-rank local environment counts are inconsistent")

    return {
        "schema": SNAPSHOT_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": step,
        "absolute_target": absolute_target,
        "policy_sha256": _policy_digest(torch, payload["policy_state_dict"]),
        "optimizer_state_present": bool(payload["optimizer_state_dict"]),
        "optimizer_state_entry_count": len(payload["optimizer_state_dict"]["state"]),
        "optimizer_param_group_count": len(
            payload["optimizer_state_dict"]["param_groups"]
        ),
        "scheduler_state_present": bool(payload["lr_scheduler_state_dict"]),
        "run_dir": str(run_dir),
        "trainer_wandb_run_id": trainer_run_id,
        "sidecar_wandb_run_id": sidecar_run_id,
        "wandb_required": require_wandb,
        "sidecar_upload_artifacts": dict(upload_state.get("artifacts", {})),
        "telemetry_scalars": _telemetry_files(run_dir),
        "runtime_ranks": runtime_summaries,
    }


def verify(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expected_before_step: int,
    expected_after_step: int,
    rollout_steps: int,
    expected_world_size: int,
    require_wandb: bool = True,
) -> dict[str, Any]:
    for label, value in (("before", before), ("after", after)):
        if value.get("schema") != SNAPSHOT_SCHEMA:
            raise GateError(f"{label} snapshot schema is invalid")
    if expected_after_step <= expected_before_step:
        raise GateError("expected after step must be greater than before step")
    if before.get("checkpoint_step") != expected_before_step:
        raise GateError("before checkpoint step does not match the contract")
    if after.get("checkpoint_step") != expected_after_step:
        raise GateError("after checkpoint step does not match the contract")
    if before.get("absolute_target") != after.get("absolute_target"):
        raise GateError("absolute training target changed across restart")
    if after.get("absolute_target", -1) < expected_after_step:
        raise GateError("after checkpoint exceeded its absolute training target")
    if before.get("run_dir") != after.get("run_dir"):
        raise GateError("resume used a different experiment directory")
    if before.get("checkpoint_sha256") == after.get("checkpoint_sha256"):
        raise GateError("after checkpoint is byte-identical to the intermediate checkpoint")
    if before.get("policy_sha256") == after.get("policy_sha256"):
        raise GateError("policy parameters did not advance after resume")
    for key in ("optimizer_state_present", "scheduler_state_present"):
        if before.get(key) is not True or after.get(key) is not True:
            raise GateError(f"{key} is not present in both checkpoints")
    for label, value, step in (
        ("before", before, expected_before_step),
        ("after", after, expected_after_step),
    ):
        optimizer_entries = value.get("optimizer_state_entry_count")
        parameter_groups = value.get("optimizer_param_group_count")
        if type(optimizer_entries) is not int or optimizer_entries < 0:
            raise GateError(f"{label} optimizer state entry count is invalid")
        if step > 0 and optimizer_entries == 0:
            raise GateError(f"{label} optimizer state is empty after step zero")
        if type(parameter_groups) is not int or parameter_groups < 1:
            raise GateError(f"{label} optimizer parameter group count is invalid")
    if before.get("wandb_required") != require_wandb or after.get(
        "wandb_required"
    ) != require_wandb:
        raise GateError("snapshot W&B mode does not match the verification contract")
    if require_wandb:
        for key in ("trainer_wandb_run_id", "sidecar_wandb_run_id"):
            if before.get(key) != after.get(key):
                raise GateError(f"{key} changed across the new process")

    before_scalars = dict(before.get("telemetry_scalars", {}))
    after_scalars = dict(after.get("telemetry_scalars", {}))
    if not before_scalars or not after_scalars:
        raise GateError("telemetry scalar inventory is empty")
    for step, digest in before_scalars.items():
        if after_scalars.get(step) != digest:
            raise GateError(f"pre-restart telemetry step {step} was overwritten")
    new_steps = sorted(set(map(int, after_scalars)).difference(map(int, before_scalars)))
    if not new_steps or max(new_steps) <= max(map(int, before_scalars)):
        raise GateError("telemetry did not advance monotonically after restart")

    if require_wandb:
        before_uploads = dict(before.get("sidecar_upload_artifacts", {}))
        after_uploads = dict(after.get("sidecar_upload_artifacts", {}))
        for path, fingerprint in before_uploads.items():
            if after_uploads.get(path) != fingerprint:
                raise GateError(f"W&B upload continuity lost for {path}")

    for label, value, step in (
        ("before", before, expected_before_step),
        ("after", after, expected_after_step),
    ):
        ranks = value.get("runtime_ranks")
        if not isinstance(ranks, list) or len(ranks) != expected_world_size:
            raise GateError(f"{label} snapshot has incomplete per-rank resume state")
        local_env_counts: set[int] = set()
        for expected_rank, rank in enumerate(ranks):
            if not isinstance(rank, Mapping):
                raise GateError(f"{label} rank {expected_rank} summary is malformed")
            for key, minimum in (
                ("rank", 0),
                ("telemetry_step", 0),
                ("rollout_steps", 1),
                ("local_num_envs", 1),
            ):
                field = rank.get(key)
                if type(field) is not int or field < minimum:
                    raise GateError(
                        f"{label} rank {expected_rank} {key} must be an integer >= {minimum}"
                    )
            if rank["rank"] != expected_rank:
                raise GateError(f"{label} rank ordering is invalid")
            if rank["rollout_steps"] != rollout_steps:
                raise GateError(f"{label} rank rollout length changed")
            if rank["telemetry_step"] != step * rollout_steps:
                raise GateError(f"{label} rank telemetry offset is not checkpoint-aligned")
            local_env_counts.add(rank["local_num_envs"])
        if len(local_env_counts) != 1:
            raise GateError(f"{label} per-rank local environment counts differ")

    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "run_dir": after["run_dir"],
        "before_step": expected_before_step,
        "after_step": expected_after_step,
        "steps_advanced": expected_after_step - expected_before_step,
        "world_size": expected_world_size,
        "rollout_steps": rollout_steps,
        "trainer_wandb_run_id": after["trainer_wandb_run_id"],
        "sidecar_wandb_run_id": after["sidecar_wandb_run_id"],
        "wandb_required": require_wandb,
        "new_telemetry_steps": new_steps,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON payload is not an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snap = subparsers.add_parser("snapshot")
    snap.add_argument("--checkpoint", type=Path, required=True)
    snap.add_argument("--run-dir", type=Path, required=True)
    snap.add_argument("--expected-world-size", type=int, required=True)
    snap.add_argument(
        "--wandb-mode", choices=("required", "disabled"), default="required"
    )
    snap.add_argument(
        "--expected-checkpoint-sha256",
        default="NONE",
        help="optional lowercase SHA-256 pinned by the launcher",
    )
    snap.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--before", type=Path, required=True)
    check.add_argument("--after", type=Path, required=True)
    check.add_argument("--expected-before-step", type=int, required=True)
    check.add_argument("--expected-after-step", type=int, required=True)
    check.add_argument("--rollout-steps", type=int, required=True)
    check.add_argument("--expected-world-size", type=int, required=True)
    check.add_argument(
        "--wandb-mode", choices=("required", "disabled"), default="required"
    )
    check.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.expected_world_size < 1:
            raise GateError("expected world size must be positive")
        if args.command == "snapshot":
            result = snapshot(
                args.checkpoint,
                args.run_dir,
                args.expected_world_size,
                require_wandb=args.wandb_mode == "required",
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
        else:
            if args.rollout_steps < 1:
                raise GateError("rollout steps must be positive")
            result = verify(
                _load_json(args.before),
                _load_json(args.after),
                expected_before_step=args.expected_before_step,
                expected_after_step=args.expected_after_step,
                rollout_steps=args.rollout_steps,
                expected_world_size=args.expected_world_size,
                require_wandb=args.wandb_mode == "required",
            )
        _atomic_json(args.output, result)
    except GateError as exc:
        print(f"resume continuity gate: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
