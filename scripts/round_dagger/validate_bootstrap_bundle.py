#!/usr/bin/env python3
"""Fail-closed validation of a portable fresh-student bootstrap bundle."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GR00T_ROOT = ROOT / "GR00T-VisualSim2Real"
for source_root in (GR00T_ROOT, ROOT):
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from round_dagger.actor import runtime_policy_sha256  # noqa: E402
from round_dagger.artifacts import load_strict_json, sha256_file  # noqa: E402
from round_dagger.gr00t_policy import (  # noqa: E402
    create_vision_actor_obj_pred,
    load_checkpoint_strict,
    load_policy_spec,
)


MANIFEST_SCHEMA = "fast_roborender.round_dagger.bootstrap_policy.v1"
MANIFEST_KEYS = {
    "schema",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
}
HOMIE_CONTROLLER_PATHS = {
    "walk": GR00T_ROOT / "models/model_walk.pt",
    "stand": GR00T_ROOT / "models/model_stand.pt",
}


def _regular_read_only(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise RuntimeError(f"{label} must have immutable mode 0444")
    return resolved


def validate(checkpoint: Path) -> dict[str, object]:
    checkpoint = _regular_read_only(checkpoint, "checkpoint")
    spec_path = _regular_read_only(
        Path(str(checkpoint) + ".policy_spec.json"), "policy spec"
    )
    manifest_path = _regular_read_only(
        Path(str(checkpoint) + ".manifest.json"), "bootstrap manifest"
    )
    manifest = load_strict_json(manifest_path)
    if set(manifest) != MANIFEST_KEYS or manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("bootstrap manifest schema or key set mismatch")
    if manifest["checkpoint_path"] != checkpoint.name:
        raise RuntimeError("bootstrap manifest checkpoint path mismatch")
    if manifest["policy_spec_path"] != spec_path.name:
        raise RuntimeError("bootstrap manifest policy-spec path mismatch")
    checkpoint_sha = sha256_file(checkpoint)
    spec_sha = sha256_file(spec_path)
    if manifest["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("bootstrap checkpoint digest mismatch")
    if manifest["policy_spec_sha256"] != spec_sha:
        raise RuntimeError("bootstrap policy-spec digest mismatch")
    spec = load_policy_spec(spec_path)
    if spec["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("portable policy spec checkpoint identity mismatch")
    if spec["runtime_policy_sha256"] != manifest["runtime_policy_sha256"]:
        raise RuntimeError("portable policy spec runtime identity mismatch")
    actual_controller_sha256 = {}
    for role, path in HOMIE_CONTROLLER_PATHS.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Homie {role} controller must be a regular non-symlink file"
            )
        actual_controller_sha256[role] = sha256_file(path)
    if actual_controller_sha256 != spec["homie_controller_sha256"]:
        raise RuntimeError("portable policy Homie controller identity mismatch")

    model = create_vision_actor_obj_pred(spec)
    load_checkpoint_strict(model, checkpoint, expected_sha256=checkpoint_sha)
    runtime_sha = runtime_policy_sha256(
        model, homie_controller_sha256=actual_controller_sha256
    )
    if runtime_sha != manifest["runtime_policy_sha256"]:
        raise RuntimeError("reconstructed policy runtime identity mismatch")
    hidden_staging = sorted(
        item.name
        for item in checkpoint.parent.iterdir()
        if item.name.startswith(f".{checkpoint.name}.bootstrap.")
    )
    if hidden_staging:
        raise RuntimeError(f"bootstrap staging debris remains: {hidden_staging}")
    return {
        "schema": "fast_roborender.round_dagger.bootstrap_validation.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "policy_spec_sha256": spec_sha,
        "runtime_policy_sha256": runtime_sha,
        "homie_controller_sha256": actual_controller_sha256,
        "policy_action_dim": int(model.num_actions),
        "output_contract": spec["output_contract"],
        "immutable_mode": "0444",
        "hidden_staging_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
