#!/usr/bin/env python3
"""Publish the validation-MSE-selected frontier as a runnable policy bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import sha256_file  # noqa: E402
from round_dagger.checkpoint_compat import load_restricted_restart_document  # noqa: E402
from round_dagger.learner import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    LearnerBundleConfig,
    LearnerError,
    TrainResult,
    publish_learner_result,
)


EXPOSURE_FIELDS = (
    "sampling_policy",
    "optimizer_examples",
    "newest_source_examples",
    "prior_source_examples",
    "newest_source_population",
    "prior_source_population",
    "balanced_exposure_unit_examples",
    "balanced_exposure_units",
    "newest_expected_passes",
    "prior_expected_passes",
)


def _last_loss(path: Path, completed_steps: int) -> float:
    if path.is_symlink() or not path.is_file():
        raise LearnerError("local learner telemetry is missing")
    matched = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("optimizer_step") == completed_steps:
            matched = row.get("loss")
    if (
        isinstance(matched, bool)
        or not isinstance(matched, (int, float))
        or not math.isfinite(float(matched))
    ):
        raise LearnerError("selected frontier has no matching finite training loss")
    return float(matched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--frontier-sha256", required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learner-identity-sha256", required=True)
    parser.add_argument("--runtime-environment-sha256", required=True)
    parser.add_argument("--learner-contract-sha256", required=True)
    arguments = parser.parse_args()

    if sha256_file(arguments.frontier) != arguments.frontier_sha256:
        raise LearnerError("selected frontier checkpoint SHA-256 differs")
    config = LearnerBundleConfig.load(arguments.bundle)
    checkpoint = load_restricted_restart_document(arguments.frontier, map_location="cpu")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise LearnerError("selected frontier checkpoint schema differs")
    expected = {
        "campaign_id": config.campaign_id,
        "generation_id": config.generation_id,
        "macro_round": config.macro_round,
        "model_lineage_sha256": config.model_lineage_sha256,
        "learner_config_sha256": config.sha256,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise LearnerError("selected frontier lineage differs from learner bundle")
    completed_steps = checkpoint.get("completed_steps")
    if type(completed_steps) is not int or completed_steps < 1:
        raise LearnerError("selected frontier has no optimizer progress")
    exposure = {field: checkpoint.get(field) for field in EXPOSURE_FIELDS}
    if any(value is None for value in exposure.values()):
        raise LearnerError("selected frontier exposure evidence is incomplete")
    result = TrainResult(
        completed_steps=completed_steps,
        target_steps=completed_steps,
        exposure=exposure,
        initial_loss=float(checkpoint["initial_loss"]),
        final_loss=_last_loss(arguments.telemetry, completed_steps),
        restart_checkpoint=str(arguments.frontier.resolve(strict=True)),
        replay_snapshot_sha256=checkpoint["replay_snapshot_sha256"],
        learner_config_sha256=config.sha256,
    )
    report = publish_learner_result(
        output_dir=arguments.output_dir,
        result=result,
        config=config,
        learner_identity_sha256=arguments.learner_identity_sha256,
        runtime_container_sha256=arguments.runtime_environment_sha256,
        learner_contract_sha256=arguments.learner_contract_sha256,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, LearnerError) as error:
        print(f"native H200 frontier selection failed: {error}", file=sys.stderr)
        raise SystemExit(2)
