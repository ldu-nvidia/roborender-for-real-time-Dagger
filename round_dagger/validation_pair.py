"""Exact completion gate for bounded two-rank actor validation cohorts.

Unlike :mod:`round_dagger.actor_pair`, whose contract is deliberately fixed to
the production ``2 x 32 x 512`` train shape, this module derives a bounded
validation shape from the immutable cohort contract.  It accepts exactly two
logical ranks, 1--32 environments per rank, and 1--300 steps per environment.
Both eval cohorts and explicitly acknowledged train-validation cohorts use the
same byte-level rank and seal verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .actor_pair import ActorPairError, inspect_actor_rank, require_actor_rank_completion
from .artifacts import ArtifactError, load_strict_json, seal_cohort, verify_cohort


VALIDATION_RANKS = [0, 1]
MAX_ENVIRONMENTS_PER_RANK = 32
MAX_STEPS_PER_ENVIRONMENT = 300


class ValidationPairError(RuntimeError):
    """A bounded validation cohort is unsafe, inconsistent, or incomplete."""


def _root(cohort_dir: str | Path) -> Path:
    root = Path(cohort_dir).expanduser()
    if not root.is_absolute():
        raise ValidationPairError("validation cohort path must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValidationPairError("validation cohort path must be a safe directory")
    return root


def validation_contract(cohort_dir: str | Path) -> dict[str, Any]:
    """Load and reduce the immutable contract to its bounded launch shape."""

    root = _root(cohort_dir)
    try:
        contract = load_strict_json(root / "contract.json")
    except Exception as exc:
        raise ValidationPairError(
            f"cannot load validation cohort contract: {exc}"
        ) from exc

    ranks = contract.get("expected_ranks")
    if ranks != VALIDATION_RANKS:
        raise ValidationPairError(
            "validation pair contract must contain exactly logical ranks 0 and 1"
        )
    purpose = contract.get("purpose")
    if purpose not in {"eval", "train"}:
        raise ValidationPairError("validation pair purpose must be eval or train")
    environments = contract.get("expected_global_env_ids")
    if (
        not isinstance(environments, list)
        or any(type(item) is not int or item < 0 for item in environments)
        or environments != sorted(set(environments))
        or len(environments) < 2
        or len(environments) % 2
    ):
        raise ValidationPairError(
            "validation pair environments must be sorted, unique, non-negative, "
            "and evenly divisible across two ranks"
        )
    if environments != list(range(environments[0], environments[0] + len(environments))):
        raise ValidationPairError("validation pair environments must be consecutive")
    environments_per_rank = len(environments) // 2
    if environments_per_rank > MAX_ENVIRONMENTS_PER_RANK:
        raise ValidationPairError("validation pair envs/rank must be <= 32")
    steps = contract.get("expected_steps_per_env")
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS_PER_ENVIRONMENT:
        raise ValidationPairError("validation pair steps must be from 1 through 300")
    transitions = len(environments) * steps
    if contract.get("expected_transitions") != transitions:
        raise ValidationPairError(
            "validation pair expected_transitions is inconsistent with its shape"
        )
    return {
        "schema": "fast_roborender.round_dagger.validation_pair_contract.v1",
        "cohort_dir": str(root),
        "purpose": purpose,
        "ranks": VALIDATION_RANKS,
        "rank_count": len(VALIDATION_RANKS),
        "global_env_offset": environments[0],
        "expected_global_env_ids": environments,
        "environments_per_rank": environments_per_rank,
        "steps_per_environment": steps,
        "expected_transitions": transitions,
    }


def _assert_only_expected_rank_entries(root: Path) -> None:
    shard_root = root / "shards"
    if not shard_root.is_dir() or shard_root.is_symlink():
        raise ValidationPairError("validation cohort has no safe shards directory")
    prefixes = tuple(f"rank-{rank:04d}_" for rank in VALIDATION_RANKS)
    for path in shard_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValidationPairError(f"unsafe validation shard entry: {path.name}")
        if not path.name.startswith(prefixes):
            raise ValidationPairError(f"unexpected validation shard rank: {path.name}")


def _verify_sealed_report(
    report: Mapping[str, Any], shape: Mapping[str, Any]
) -> None:
    expected = {
        "purpose": shape["purpose"],
        "sealed": True,
        "expected_ranks": VALIDATION_RANKS,
        "ranks": VALIDATION_RANKS,
        "expected_global_env_ids": shape["expected_global_env_ids"],
        "global_env_ids": shape["expected_global_env_ids"],
        "expected_transitions": shape["expected_transitions"],
        "transition_count": shape["expected_transitions"],
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise ValidationPairError(
            "sealed validation cohort failed exact verification: "
            + ", ".join(mismatches)
        )


def inspect_validation_pair(cohort_dir: str | Path) -> dict[str, Any]:
    """Inspect both exact rank partitions without publishing a seal."""

    root = _root(cohort_dir)
    shape = validation_contract(root)
    _assert_only_expected_rank_entries(root)
    seal_path = root / "seal.json"
    if seal_path.exists() or seal_path.is_symlink():
        try:
            report = verify_cohort(root)
        except ArtifactError as exc:
            raise ValidationPairError(
                f"existing validation cohort seal is invalid: {exc}"
            ) from exc
        _verify_sealed_report(report, shape)
        states = {str(rank): "complete" for rank in VALIDATION_RANKS}
        sealed = True
    else:
        states = {}
        try:
            for rank in VALIDATION_RANKS:
                states[str(rank)] = inspect_actor_rank(root, rank)["state"]
        except ActorPairError as exc:
            raise ValidationPairError(
                f"validation actor rank inspection failed: {exc}"
            ) from exc
        sealed = False
    return {
        "schema": "fast_roborender.round_dagger.validation_pair_status.v1",
        "cohort_dir": str(root),
        "sealed": sealed,
        "purpose": shape["purpose"],
        "environments_per_rank": shape["environments_per_rank"],
        "steps_per_environment": shape["steps_per_environment"],
        "expected_transitions": shape["expected_transitions"],
        "rank_states": states,
    }


def require_validation_rank_completion(
    cohort_dir: str | Path, rank: int
) -> dict[str, Any]:
    """Require exact contract-derived shard coverage for one logical rank."""

    validation_contract(cohort_dir)
    if rank not in VALIDATION_RANKS:
        raise ValidationPairError("validation actor rank must be exactly 0 or 1")
    try:
        return require_actor_rank_completion(cohort_dir, rank)
    except ActorPairError as exc:
        raise ValidationPairError(str(exc)) from exc


def seal_validation_pair(cohort_dir: str | Path) -> dict[str, Any]:
    """Seal only after both bounded validation ranks are exactly complete."""

    status = inspect_validation_pair(cohort_dir)
    if status["sealed"]:
        report = verify_cohort(cohort_dir)
        _verify_sealed_report(report, validation_contract(cohort_dir))
        return report
    if status["rank_states"] != {"0": "complete", "1": "complete"}:
        raise ValidationPairError(
            "cannot seal before both bounded validation ranks are exactly complete"
        )
    try:
        report = seal_cohort(cohort_dir)
        report = verify_cohort(cohort_dir)
    except ArtifactError as exc:
        raise ValidationPairError(f"validation cohort sealing failed: {exc}") from exc
    _verify_sealed_report(report, validation_contract(cohort_dir))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("contract", "status", "rank", "seal"))
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "contract":
        if arguments.rank is not None:
            raise ValidationPairError("--rank is valid only for the rank operation")
        result = validation_contract(arguments.cohort_dir)
    elif arguments.operation == "status":
        if arguments.rank is not None:
            raise ValidationPairError("--rank is valid only for the rank operation")
        result = inspect_validation_pair(arguments.cohort_dir)
    elif arguments.operation == "rank":
        if arguments.rank is None:
            raise ValidationPairError("rank operation requires --rank")
        result = require_validation_rank_completion(
            arguments.cohort_dir, arguments.rank
        )
    else:
        if arguments.rank is not None:
            raise ValidationPairError("--rank is valid only for the rank operation")
        result = seal_validation_pair(arguments.cohort_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationPairError, ActorPairError, ArtifactError) as error:
        print(f"validation pair gate failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)


__all__ = [
    "ValidationPairError",
    "inspect_validation_pair",
    "require_validation_rank_completion",
    "seal_validation_pair",
    "validation_contract",
]
