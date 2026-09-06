#!/usr/bin/env python3
"""Fail if BC collection drifts from Raven's executable GR00T contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


REFERENCE_COMMIT = "94d7985"
REFERENCE_CONFIG = "gr00t/rl/config/exp/loco_manip/wsdpt_videogen_collect.yaml"
REFERENCE_TRAINER = "gr00t/rl/trl/trainer/distill_trainer_videogen.py"


class ContractMismatch(RuntimeError):
    pass


def _load_yaml(text: str) -> dict:
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ContractMismatch("configuration is not a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate(repo_root: Path) -> dict:
    gr00t = repo_root / "pipeline" / "GR00T-VisualSim2Real"
    active_path = gr00t / REFERENCE_CONFIG
    reference_text = subprocess.run(
        ["git", "-C", str(gr00t), "show", f"{REFERENCE_COMMIT}:{REFERENCE_CONFIG}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_text = active_path.read_text(encoding="utf-8")
    if active_text != reference_text:
        raise ContractMismatch("active BC collection YAML is not byte-identical to Raven")
    active = _load_yaml(active_text)
    if active.get("trainer", {}).get("_target_") != (
        "gr00t.rl.trl.trainer.distill_trainer_videogen.TRLDistillTrainerVideoGen"
    ):
        raise ContractMismatch("active config does not select Raven's exact trainer")

    head = subprocess.run(
        ["git", "-C", str(gr00t), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head.startswith(REFERENCE_COMMIT):
        raise ContractMismatch(f"active GR00T checkout is {head}, not {REFERENCE_COMMIT}")
    drift = subprocess.run(
        ["git", "-C", str(gr00t), "status", "--porcelain", "--", "gr00t"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if drift:
        raise ContractMismatch("active Raven GR00T source tree has local drift")
    trainer_path = gr00t / REFERENCE_TRAINER
    expected_trainer_sha = json.loads(
        (repo_root / "pipeline/bc_collection/raven_bc_collection_contract.v1.json")
        .read_text(encoding="utf-8")
    )["source"]["trainer_sha256"]
    if _sha256(trainer_path) != expected_trainer_sha:
        raise ContractMismatch("active Raven collection trainer hash drift")

    return {
        "status": "pass",
        "reference_commit": REFERENCE_COMMIT,
        "reference_config": REFERENCE_CONFIG,
        "active_config": str(active_path),
        "active_gr00t_head": head,
        "active_yaml_byte_identical": True,
        "active_trainer_sha256": expected_trainer_sha,
        "allowed_yaml_differences": [],
        "only_launch_time_scientific_override": "num_envs",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.repo_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
