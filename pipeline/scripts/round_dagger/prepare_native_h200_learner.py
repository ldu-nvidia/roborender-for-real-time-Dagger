#!/usr/bin/env python3
"""Create one train-only, exposure-normalized eight-H200 learner stage."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import (  # noqa: E402
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.gr00t_policy import load_policy_spec  # noqa: E402
from round_dagger.learner import (  # noqa: E402
    BUNDLE_SCHEMA,
    CANONICAL_CAMPAIGN_ROOT,
    LearnerBundleConfig,
    LearnerError,
)
from round_dagger.learner_identity import learner_identity_sha256  # noqa: E402
from round_dagger.replay import load_snapshot  # noqa: E402


PLAN_SCHEMA = "fast_roborender.round_dagger.native_h200_learner_stage.v1"


def _align(value: Fraction, quantum: int) -> int:
    numerator = value.numerator
    denominator = value.denominator * quantum
    return ((numerator + denominator - 1) // denominator) * quantum


def _sha(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LearnerError(f"{label} must be a lowercase SHA-256")
    return value


def prepare(arguments: argparse.Namespace) -> tuple[dict, LearnerBundleConfig]:
    campaign = arguments.campaign_root.resolve(strict=True)
    if campaign != CANONICAL_CAMPAIGN_ROOT.resolve(strict=True):
        raise LearnerError("campaign root is not the canonical H200 data root")
    if arguments.world_size != 8:
        raise LearnerError("native learner world size must be exactly eight")
    checkpoint = arguments.parent_checkpoint.resolve(strict=True)
    spec_path = arguments.parent_policy_spec.resolve(strict=True)
    if checkpoint.is_symlink() or spec_path.is_symlink():
        raise LearnerError("parent policy inputs must not be symbolic links")
    parent_sha = _sha(arguments.parent_checkpoint_sha256, "parent checkpoint")
    if sha256_file(checkpoint) != parent_sha:
        raise LearnerError("parent checkpoint SHA-256 differs")
    spec = load_policy_spec(spec_path)
    if spec["checkpoint_sha256"] != parent_sha:
        raise LearnerError("parent policy specification names another checkpoint")

    snapshot = load_snapshot(
        arguments.train_snapshot,
        expected_file_sha256=arguments.train_snapshot_file_sha256,
        artifact_root=campaign,
    )
    round_number = arguments.macro_round
    if snapshot["campaign_id"] != "roborender-h200-v001" or snapshot["through_macro_round"] != round_number:
        raise LearnerError("training snapshot lineage differs from requested stage")
    counts = snapshot["counts_by_round"]
    expected_rounds = {str(index) for index in range(round_number + 1)}
    if set(counts) != expected_rounds:
        raise LearnerError("training snapshot does not contain a contiguous round history")
    newest_population = int(counts[str(round_number)])
    prior_population = sum(int(counts[str(index)]) for index in range(round_number))
    if newest_population <= 0:
        raise LearnerError("training source population is empty")
    unit_examples = newest_population

    protocol = load_strict_json(arguments.protocol)
    rule = protocol["optimization"]["stopping_rule"]
    interval_units = Fraction(str(rule["validation_interval_balanced_exposure_units"]))
    minimum_units = Fraction(str(rule["minimum_balanced_exposure_units"]))
    maximum_units = Fraction(str(rule["maximum_balanced_exposure_units"]))
    if not (0 < interval_units <= minimum_units <= maximum_units):
        raise LearnerError("validation exposure rule is invalid")
    global_batch = arguments.batch_size_per_rank * arguments.world_size
    interval_examples = _align(interval_units * unit_examples, global_batch)
    minimum_examples = _align(minimum_units * unit_examples, global_batch)
    maximum_examples = _align(maximum_units * unit_examples, global_batch)
    frontiers: list[int] = []
    index = 1
    while True:
        frontier = _align(index * interval_units * unit_examples, global_batch)
        if frontier > maximum_examples:
            break
        if not frontiers or frontier != frontiers[-1]:
            frontiers.append(frontier)
        index += 1
    if not frontiers or frontiers[-1] != maximum_examples:
        frontiers.append(maximum_examples)

    generation = protocol["generations"][round_number]["generation_id"]
    bundle = LearnerBundleConfig.from_mapping(
        {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": "roborender-h200-v001",
            "generation_id": generation,
            "macro_round": round_number,
            "model_lineage_sha256": parent_sha,
            "code_sha256": _sha(arguments.code_sha256, "learner code"),
            "rgb_mean": spec["input_contract"]["rgb_mean"],
            "rgb_std": spec["input_contract"]["rgb_std"],
            "rgb_input_key": "vision_obs",
            "rgb_layout": "NHWC",
            "output_key": "actions",
            "observation_key_map": {},
            "train_action_slice": spec["output_contract"]["train_action_slice"],
            "batch_size": arguments.batch_size_per_rank,
            "target_optimizer_examples": maximum_examples,
            "checkpoint_every_optimizer_examples": interval_examples,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "seed": arguments.seed,
            "partition_seed": arguments.partition_seed,
            "initial_checkpoint_sha256": parent_sha,
            "model_config": spec,
        }
    )
    identity = learner_identity_sha256(
        campaign_id=bundle.campaign_id,
        macro_round=round_number,
        learner_code_sha256=bundle.code_sha256,
        learner_config_sha256=bundle.sha256,
        replay_snapshot_file_sha256=snapshot["file_sha256"],
        replay_snapshot_sha256=snapshot["snapshot_sha256"],
        parent_checkpoint_sha256=parent_sha,
        parent_runtime_policy_sha256=spec["runtime_policy_sha256"],
        runtime_container_sha256=_sha(
            arguments.runtime_environment_sha256, "runtime environment"
        ),
        learner_contract_sha256=_sha(
            arguments.learner_contract_sha256, "learner contract"
        ),
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": bundle.campaign_id,
        "generation_id": generation,
        "macro_round": round_number,
        "sampling_policy": "current_round_only_uniform_v1",
        "newest_source_population": newest_population,
        "prior_source_population": prior_population,
        "balanced_exposure_unit_examples": unit_examples,
        "global_batch_size": global_batch,
        "validation_interval_optimizer_examples": interval_examples,
        "minimum_optimizer_examples": minimum_examples,
        "maximum_optimizer_examples": maximum_examples,
        "frontier_optimizer_examples": frontiers,
        "train_snapshot_path": str(arguments.train_snapshot.resolve(strict=True)),
        "train_snapshot_file_sha256": snapshot["file_sha256"],
        "train_snapshot_sha256": snapshot["snapshot_sha256"],
        "parent_checkpoint_path": str(checkpoint),
        "parent_checkpoint_sha256": parent_sha,
        "parent_policy_spec_path": str(spec_path),
        "learner_bundle_sha256": bundle.sha256,
        "learner_identity_sha256": identity,
        "runtime_environment_sha256": arguments.runtime_environment_sha256,
        "learner_contract_sha256": arguments.learner_contract_sha256,
    }
    return plan, bundle


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-root", type=Path, default=CANONICAL_CAMPAIGN_ROOT)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--macro-round", type=int, choices=range(5), required=True)
    result.add_argument("--train-snapshot", type=Path, required=True)
    result.add_argument("--train-snapshot-file-sha256", required=True)
    result.add_argument("--parent-checkpoint", type=Path, required=True)
    result.add_argument("--parent-checkpoint-sha256", required=True)
    result.add_argument("--parent-policy-spec", type=Path, required=True)
    result.add_argument("--code-sha256", required=True)
    result.add_argument("--runtime-environment-sha256", required=True)
    result.add_argument("--learner-contract-sha256", required=True)
    result.add_argument("--batch-size-per-rank", type=int, default=8)
    result.add_argument("--world-size", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=2.5e-5)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--partition-seed", type=int, default=23)
    result.add_argument("--bundle-output", type=Path, required=True)
    result.add_argument("--plan-output", type=Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    plan, bundle = prepare(arguments)
    if arguments.bundle_output.parent != arguments.plan_output.parent:
        raise LearnerError("bundle and stage plan must share one generation-local directory")
    bundle_sha = publish_immutable_json(arguments.bundle_output, bundle.to_mapping())
    if bundle_sha != bundle.sha256:
        raise LearnerError("learner bundle publication digest differs")
    plan = {**plan, "learner_bundle_path": str(arguments.bundle_output.resolve())}
    plan_sha = publish_immutable_json(arguments.plan_output, plan)
    print(json.dumps({"bundle_sha256": bundle_sha, "plan_sha256": plan_sha, **plan}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LearnerError as error:
        print(f"native H200 learner preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
