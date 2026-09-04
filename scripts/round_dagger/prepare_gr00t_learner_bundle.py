#!/usr/bin/env python3
"""Prepare one immutable, parent-bound GR00T learner bundle.

The portable policy specification is the exact active-parent identity.  This
command refuses to relabel it, adds optimizer/restart settings, and publishes
the canonical ``LearnerBundleConfig`` consumed by every H100 rank.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.gr00t_policy import (  # noqa: E402
    ENV_COMMAND_DIM,
    HOMIE_ACTION_DIM,
    POLICY_ACTION_DIM,
    REPLAY_ACTION_DIM,
    TRAIN_ACTION_SLICE,
    VISION_SHAPE,
    GR00TPolicyError,
    load_policy_spec,
    validate_policy_spec,
)
from round_dagger.learner import (  # noqa: E402
    BUNDLE_SCHEMA,
    LearnerBundleConfig,
    LearnerError,
)
from round_dagger.replay import load_heldout_snapshot  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CAMPAIGN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class BundlePreparationError(RuntimeError):
    """The parent, architecture, or learner contract is inconsistent."""


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BundlePreparationError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_file(value: Path, label: str) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        raise BundlePreparationError(f"{label} must be an absolute path")
    if path.is_symlink() or not path.is_file():
        raise BundlePreparationError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _absolute_output(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        raise BundlePreparationError("output must be an absolute path")
    if path.is_symlink():
        raise BundlePreparationError("output must not be a symbolic link")
    return Path(os.path.abspath(path))


def prepare_bundle(
    *,
    base_policy_spec: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_sha256: str,
    runtime_policy_sha256: str,
    campaign_id: str,
    macro_round: int,
    code_sha256: str,
    batch_size: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    partition_seed: int,
    heldout_snapshot: Path | None = None,
    heldout_eval_interval: int | None = None,
    artifact_root: Path | None = None,
) -> LearnerBundleConfig:
    """Derive a canonical bundle without writing it."""

    if not isinstance(campaign_id, str) or _CAMPAIGN_RE.fullmatch(campaign_id) is None:
        raise BundlePreparationError("campaign_id is not a safe token")
    spec_path = _absolute_file(base_policy_spec, "base policy spec")
    checkpoint_path = _absolute_file(parent_checkpoint, "parent checkpoint")
    expected_parent = _sha256(
        expected_parent_checkpoint_sha256, "expected parent checkpoint SHA-256"
    )
    actual_parent = sha256_file(checkpoint_path)
    if actual_parent != expected_parent:
        raise BundlePreparationError("parent checkpoint SHA-256 mismatch")

    derived_spec = load_policy_spec(spec_path)
    expected_runtime = _sha256(
        runtime_policy_sha256, "runtime policy SHA-256"
    )
    if derived_spec["checkpoint_sha256"] != actual_parent:
        raise BundlePreparationError(
            "active policy spec checkpoint identity mismatch"
        )
    if derived_spec["runtime_policy_sha256"] != expected_runtime:
        raise BundlePreparationError(
            "active policy spec runtime identity mismatch"
        )
    derived_spec = validate_policy_spec(derived_spec)
    if derived_spec["input_contract"]["vision_shape"] != list(VISION_SHAPE):
        raise BundlePreparationError("portable policy does not use the production view")
    output_contract = derived_spec["output_contract"]
    if output_contract != {
        "policy_action_dim": POLICY_ACTION_DIM,
        "homie_action_dim": HOMIE_ACTION_DIM,
        "replay_action_dim": REPLAY_ACTION_DIM,
        "env_command_dim": ENV_COMMAND_DIM,
        "train_action_slice": list(TRAIN_ACTION_SLICE),
        "object_prediction_dim": 3,
    }:
        raise BundlePreparationError(
            "portable policy action contract is not production"
        )

    heldout_fields: dict[str, object] = {}
    if (heldout_snapshot is None) != (heldout_eval_interval is None):
        raise BundlePreparationError(
            "held-out snapshot and evaluation interval must be supplied together"
        )
    if heldout_snapshot is not None:
        heldout_path = _absolute_file(heldout_snapshot, "held-out replay snapshot")
        try:
            heldout_identity = load_heldout_snapshot(
                heldout_path, artifact_root=artifact_root
            )
        except Exception as exc:
            raise BundlePreparationError(
                f"held-out replay snapshot validation failed: {exc}"
            ) from exc
        if heldout_identity["campaign_id"] != campaign_id:
            raise BundlePreparationError(
                "held-out replay snapshot campaign mismatch"
            )
        heldout_fields = {
            "heldout_snapshot_sha256": heldout_identity["snapshot_sha256"],
            "heldout_snapshot_file_sha256": heldout_identity["file_sha256"],
            "heldout_eval_interval": heldout_eval_interval,
        }

    try:
        bundle = LearnerBundleConfig.from_mapping(
            {
                "schema": BUNDLE_SCHEMA,
                "campaign_id": campaign_id,
                "macro_round": macro_round,
                # The parent checkpoint is the sole model-lineage frontier for
                # this update.  Runtime identity remains independently bound in
                # the embedded policy specification.
                "model_lineage_sha256": actual_parent,
                "code_sha256": _sha256(code_sha256, "learner code SHA-256"),
                "rgb_mean": derived_spec["input_contract"]["rgb_mean"],
                "rgb_std": derived_spec["input_contract"]["rgb_std"],
                "rgb_input_key": "vision_obs",
                "rgb_layout": "NHWC",
                "output_key": "actions",
                "observation_key_map": {},
                "train_action_slice": output_contract["train_action_slice"],
                "batch_size": batch_size,
                "max_steps": max_steps,
                "save_every": save_every,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
                "partition_seed": partition_seed,
                "initial_checkpoint_sha256": actual_parent,
                "model_config": derived_spec,
                **heldout_fields,
            }
        )
    except LearnerError as exc:
        raise BundlePreparationError(f"invalid learner settings: {exc}") from exc
    # Exercise the concrete bundle/spec cross-check before publication.
    validate_policy_spec(bundle.to_mapping()["model_config"])
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy-spec", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--runtime-policy-sha256", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", required=True, type=int)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--save-every", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--partition-seed", required=True, type=int)
    parser.add_argument("--heldout-snapshot", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="campaign/projection root for relocated held-out replay artifacts",
    )
    parser.add_argument(
        "--heldout-eval-interval",
        type=int,
        help="evaluation cadence in optimizer steps (production target: 100)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report the canonical bundle without publishing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = prepare_bundle(
        base_policy_spec=arguments.base_policy_spec,
        parent_checkpoint=arguments.parent_checkpoint,
        expected_parent_checkpoint_sha256=arguments.expected_parent_checkpoint_sha256,
        runtime_policy_sha256=arguments.runtime_policy_sha256,
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
        code_sha256=arguments.code_sha256,
        batch_size=arguments.batch_size,
        max_steps=arguments.max_steps,
        save_every=arguments.save_every,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
        partition_seed=arguments.partition_seed,
        heldout_snapshot=arguments.heldout_snapshot,
        heldout_eval_interval=arguments.heldout_eval_interval,
        artifact_root=arguments.artifact_root,
    )
    expected = arguments.expected_bundle_sha256
    if expected is not None and bundle.sha256 != _sha256(
        expected, "expected learner bundle SHA-256"
    ):
        raise BundlePreparationError("learner bundle semantic digest mismatch")
    output = _absolute_output(arguments.output)
    published_sha = None
    if not arguments.dry_run:
        published_sha = publish_immutable_json(output, bundle.to_mapping())
        loaded = LearnerBundleConfig.load(output)
        if loaded.sha256 != bundle.sha256 or loaded.to_mapping() != bundle.to_mapping():
            raise BundlePreparationError("published learner bundle did not round-trip")
        if published_sha != bundle.sha256:
            raise BundlePreparationError("bundle file and semantic digests differ")
    print(
        json.dumps(
            {
                "bundle_path": str(output),
                "bundle_sha256": bundle.sha256,
                "published": not arguments.dry_run,
                "published_sha256": published_sha,
                "parent_checkpoint_sha256": bundle.initial_checkpoint_sha256,
                "runtime_policy_sha256": bundle.to_mapping()["model_config"][
                    "runtime_policy_sha256"
                ],
                "macro_round": bundle.macro_round,
                "max_steps": bundle.max_steps,
                "heldout_eval_interval": bundle.heldout_eval_interval,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ArtifactError,
        BundlePreparationError,
        GR00TPolicyError,
        LearnerError,
    ) as error:
        print(f"GR00T learner bundle preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
