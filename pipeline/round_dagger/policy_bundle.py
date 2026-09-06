"""Lightweight identity gate for portable round-DAgger actor bundles.

This module intentionally performs no policy construction and imports no
simulator or PyTorch runtime.  It is safe to use in launcher control planes
before allocating GPUs or starting Isaac Lab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from round_dagger.artifacts import load_strict_json, sha256_file
from round_dagger.gr00t_policy import (
    ENV_COMMAND_DIM,
    HOMIE_ACTION_DIM,
    OBJECT_PREDICTION_DIM,
    POLICY_ACTION_DIM,
    REPLAY_ACTION_DIM,
    TRAIN_ACTION_SLICE,
    GR00TPolicyError,
    load_policy_spec,
)


BOOTSTRAP_MANIFEST_SCHEMA = "fast_roborender.round_dagger.bootstrap_policy.v1"
BOOTSTRAP_MANIFEST_KEYS = {
    "schema",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
}
PRODUCTION_OUTPUT_CONTRACT = {
    "policy_action_dim": POLICY_ACTION_DIM,
    "homie_action_dim": HOMIE_ACTION_DIM,
    "replay_action_dim": REPLAY_ACTION_DIM,
    "env_command_dim": ENV_COMMAND_DIM,
    "train_action_slice": list(TRAIN_ACTION_SLICE),
    "object_prediction_dim": OBJECT_PREDICTION_DIM,
}


class PolicyBundleError(RuntimeError):
    """A portable actor checkpoint bundle is incomplete or inconsistent."""


def _absolute_regular(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PolicyBundleError(f"{label} path must be absolute")
    if candidate.is_symlink() or not candidate.is_file():
        raise PolicyBundleError(f"{label} must be a regular non-symlink file")
    return candidate


def validate_policy_bundle_identity(
    *,
    checkpoint: str | Path,
    policy_spec: str | Path,
    manifest: str | Path,
    expected_checkpoint_sha256: str,
    expected_runtime_policy_sha256: str,
) -> dict[str, Any]:
    """Bind checkpoint bytes, portable architecture, and runtime identity.

    The returned document is suitable for logging as preflight evidence.  The
    policy itself is not instantiated here; strict reconstruction remains a
    runtime gate in the actor and learner environments.
    """

    checkpoint_path = _absolute_regular(checkpoint, "policy checkpoint")
    spec_path = _absolute_regular(policy_spec, "policy spec")
    manifest_path = _absolute_regular(manifest, "bootstrap manifest")

    document = load_strict_json(manifest_path)
    if set(document) != BOOTSTRAP_MANIFEST_KEYS:
        raise PolicyBundleError("bootstrap policy manifest key-set mismatch")
    if document.get("schema") != BOOTSTRAP_MANIFEST_SCHEMA:
        raise PolicyBundleError("bootstrap policy manifest schema mismatch")
    if document.get("checkpoint_path") != checkpoint_path.name:
        raise PolicyBundleError("bootstrap policy manifest checkpoint path mismatch")
    if document.get("policy_spec_path") != spec_path.name:
        raise PolicyBundleError("bootstrap policy manifest spec path mismatch")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    policy_spec_sha256 = sha256_file(spec_path)
    if document.get("checkpoint_sha256") != checkpoint_sha256:
        raise PolicyBundleError("bootstrap policy manifest/checkpoint mismatch")
    if document.get("policy_spec_sha256") != policy_spec_sha256:
        raise PolicyBundleError("bootstrap policy manifest/spec mismatch")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise PolicyBundleError("bootstrap policy identity differs from cohort")
    if document.get("runtime_policy_sha256") != expected_runtime_policy_sha256:
        raise PolicyBundleError("bootstrap runtime policy identity differs from cohort")

    try:
        spec = load_policy_spec(spec_path)
    except GR00TPolicyError as exc:
        raise PolicyBundleError(f"portable policy spec is invalid: {exc}") from exc
    if spec.get("checkpoint_sha256") != checkpoint_sha256:
        raise PolicyBundleError("portable policy spec/checkpoint mismatch")
    if spec.get("runtime_policy_sha256") != document["runtime_policy_sha256"]:
        raise PolicyBundleError("portable policy spec/runtime identity mismatch")
    if spec.get("output_contract") != PRODUCTION_OUTPUT_CONTRACT:
        raise PolicyBundleError("portable policy 11/15/26/31 action contract mismatch")

    return {
        "schema": "fast_roborender.round_dagger.policy_bundle_preflight.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": spec["policy_state_sha256"],
        "homie_controller_sha256": dict(spec["homie_controller_sha256"]),
        "runtime_policy_sha256": document["runtime_policy_sha256"],
        "policy_spec_sha256": policy_spec_sha256,
        "output_contract": dict(PRODUCTION_OUTPUT_CONTRACT),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-runtime-policy-sha256", required=True)
    arguments = parser.parse_args()
    report = validate_policy_bundle_identity(
        checkpoint=arguments.checkpoint,
        policy_spec=arguments.policy_spec,
        manifest=arguments.manifest,
        expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
        expected_runtime_policy_sha256=arguments.expected_runtime_policy_sha256,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
