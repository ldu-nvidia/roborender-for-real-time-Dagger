#!/usr/bin/env python3
"""Fail closed until trainer scalars and sidecar scalar/image/video exist remotely."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import yaml

from wandb_preflight import _load_wandb, query_identity
from wandb_run_watcher import discover_artifacts, discover_scalars, load_scalar_file


RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ReadbackError(RuntimeError):
    pass


def _run_id(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ReadbackError(f"{label} is missing or unsafe")
    value = path.read_text(encoding="utf-8").strip()
    if RUN_ID_RE.fullmatch(value) is None:
        raise ReadbackError(f"{label} is malformed")
    return value


def _trainer_run_id(run_dir: Path) -> str:
    path = run_dir / "meta.yaml"
    if not path.is_file() or path.is_symlink():
        raise ReadbackError("trainer W&B metadata is missing or unsafe")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = payload.get("wandb_run") if isinstance(payload, dict) else None
    if not isinstance(value, str) or RUN_ID_RE.fullmatch(value) is None:
        raise ReadbackError("trainer W&B run ID is missing or malformed")
    return value


def _expected_sidecar_keys(run_dir: Path) -> tuple[set[str], dict[str, int]]:
    scalar_artifacts = discover_scalars(run_dir, min_age_s=0.0)
    media_artifacts = discover_artifacts(run_dir, min_age_s=0.0)
    by_kind = {
        "scalar": len(scalar_artifacts),
        "image": sum(item.kind == "image" for item in media_artifacts),
        "video": sum(item.kind == "video" for item in media_artifacts),
    }
    if any(count < 1 for count in by_kind.values()):
        raise ReadbackError(f"local scalar/image/video telemetry is incomplete: {by_kind}")
    keys = {"trainer_step"}
    for artifact in scalar_artifacts:
        _, metrics = load_scalar_file(artifact.path)
        keys.update(metrics)
    keys.update(item.key for item in media_artifacts)
    return keys, by_kind


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")

    run_dir = args.run_dir.resolve(strict=True)
    trainer_run_id = _trainer_run_id(run_dir)
    sidecar_run_id = _run_id(
        run_dir / "telemetry" / "WANDB_SIDECAR_RUN_ID", "sidecar W&B run ID"
    )
    sidecar_keys, local_counts = _expected_sidecar_keys(run_dir)
    trainer_keys = {"production/dagger_bc_loss"}

    wandb = _load_wandb()
    identity = query_identity(
        wandb,
        expected_email=args.expected_email,
        requested_entity=args.entity,
        require_personal=True,
    )
    api = wandb.Api(timeout=30)
    observed_trainer: set[str] = set()
    observed_sidecar: set[str] = set()
    last_error = ""
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        try:
            trainer = api.run(f"{identity.entity}/{args.project}/{trainer_run_id}")
            for row in trainer.scan_history(keys=sorted(trainer_keys), page_size=100):
                observed_trainer.update(key for key in trainer_keys if key in row)
            sidecar = api.run(f"{identity.entity}/{args.project}/{sidecar_run_id}")
            for row in sidecar.scan_history(keys=sorted(sidecar_keys), page_size=500):
                observed_sidecar.update(key for key in sidecar_keys if key in row)
            if trainer_keys <= observed_trainer and sidecar_keys <= observed_sidecar:
                break
        except Exception as exc:  # Never persist third-party error text.
            last_error = type(exc).__name__
        time.sleep(5)

    status = (
        "PASS"
        if trainer_keys <= observed_trainer and sidecar_keys <= observed_sidecar
        else "FAIL"
    )
    report: dict[str, object] = {
        "schema": "fast_roborender.wandb_training_readback.v1",
        "status": status,
        "account_email": identity.email,
        "entity": identity.entity,
        "personal": identity.personal,
        "project": args.project,
        "trainer_run_id": trainer_run_id,
        "sidecar_run_id": sidecar_run_id,
        "required_trainer_keys": sorted(trainer_keys),
        "observed_trainer_keys": sorted(observed_trainer),
        "required_sidecar_keys": sorted(sidecar_keys),
        "observed_sidecar_keys": sorted(observed_sidecar),
        "local_artifact_counts": local_counts,
        "last_api_error_type": last_error,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
