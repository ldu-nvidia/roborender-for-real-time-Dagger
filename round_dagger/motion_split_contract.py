"""Strict outer-runtime contract for the production motion split.

The nested GR00T loader owns motion loading and content verification.  This
module owns the production control-plane identity: it verifies the exact split
manifest and held-out plan bytes, proves that collection and evaluation are
disjoint, and resolves the one immutable 32-motion window assigned to an
evaluation seed.  It deliberately cannot manufacture simulator scenario IDs.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from round_dagger.artifacts import load_strict_json, sha256_file


SPLIT_SCHEMA = "roborender.motion_split.v1"
PLAN_SCHEMA = "fast_roborender.round_dagger.heldout_motion_capture_plan.v2"
FIXED_SEEDS = (17, 29, 43, 59, 71, 89, 101, 127)
TOTAL_MOTIONS = 295
COLLECTION_MOTIONS = 39
HELDOUT_MOTIONS = 256
MOTIONS_PER_SEED = 32
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MotionSplitContractError(RuntimeError):
    """A production split or held-out plan is unsafe or inconsistent."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise MotionSplitContractError(
            f"{label} fields differ: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MotionSplitContractError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MotionSplitContractError(f"{label} must be a lowercase SHA-256")
    return value


def _regular(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise MotionSplitContractError(f"{label} is unavailable: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MotionSplitContractError(
            f"{label} must be a non-symlink regular file"
        )
    return candidate


def _identity(path: Path, expected_sha256: str, label: str) -> None:
    expected = _sha(expected_sha256, f"{label} SHA-256")
    if sha256_file(path) != expected:
        raise MotionSplitContractError(f"{label} SHA-256 mismatch")


def _motion_rows(
    manifest: Mapping[str, Any], *, role: str
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    names: set[str] = set()
    motions = manifest.get("motions")
    if not isinstance(motions, list) or len(motions) != TOTAL_MOTIONS:
        raise MotionSplitContractError(
            f"motion split must enumerate exactly {TOTAL_MOTIONS} motions"
        )
    for index, raw in enumerate(motions):
        row = _exact(raw, {"path", "sha256", "split"}, f"split motion {index}")
        name = row["path"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).suffix != ".npz"
            or name in names
        ):
            raise MotionSplitContractError(
                f"split motion {index} has an unsafe or duplicate path"
            )
        names.add(name)
        split = row["split"]
        if split not in {"collection", "eval"}:
            raise MotionSplitContractError(f"split motion {name} has invalid role")
        normalized = {"path": name, "sha256": _sha(row["sha256"], name), "split": split}
        if split == role:
            result.append(normalized)
    return sorted(result, key=lambda item: item["path"])


def _validate_manifest(document: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifest = _exact(
        document,
        {"schema", "dataset", "counts", "inventory_sha256", "selection", "motions"},
        "motion split manifest",
    )
    if manifest["schema"] != SPLIT_SCHEMA:
        raise MotionSplitContractError("motion split manifest schema mismatch")
    if manifest["dataset"] != "g1_wsdpt/33demos_675_775":
        raise MotionSplitContractError("motion split dataset identity mismatch")
    if manifest["counts"] != {
        "total": TOTAL_MOTIONS,
        "collection": COLLECTION_MOTIONS,
        "eval": HELDOUT_MOTIONS,
    }:
        raise MotionSplitContractError("motion split counts are not production counts")
    if manifest["selection"] != {
        "method": "sha256_rank",
        "seed": "roborender-heldout-256-v1",
        "held_out_role": "eval",
    }:
        raise MotionSplitContractError("motion split selection policy mismatch")
    _sha(manifest["inventory_sha256"], "motion split inventory")
    collection = _motion_rows(manifest, role="collection")
    evaluation = _motion_rows(manifest, role="eval")
    if len(collection) != COLLECTION_MOTIONS or len(evaluation) != HELDOUT_MOTIONS:
        raise MotionSplitContractError("motion split role counts differ from its manifest")
    if {row["path"] for row in collection} & {row["path"] for row in evaluation}:
        raise MotionSplitContractError("collection and evaluation paths overlap")
    if {row["sha256"] for row in collection} & {row["sha256"] for row in evaluation}:
        raise MotionSplitContractError("collection and evaluation content overlaps")
    return collection, evaluation


def _runtime_rows(
    raw: Any, *, role: str, expected: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise MotionSplitContractError(f"plan {role} runtime order is incomplete")
    index_name = f"runtime_{role}_index"
    result: list[dict[str, Any]] = []
    for index, (item, expected_motion) in enumerate(zip(raw, expected, strict=True)):
        row = _exact(
            item,
            {index_name, "relative_path", "size", "sha256"},
            f"plan {role} motion {index}",
        )
        if row[index_name] != index:
            raise MotionSplitContractError(f"plan {role} motion index is non-canonical")
        _integer(row["size"], f"plan {role} motion size", minimum=1)
        if (
            row["relative_path"] != expected_motion["path"]
            or row["sha256"] != expected_motion["sha256"]
        ):
            raise MotionSplitContractError(
                f"plan {role} runtime order differs from the split manifest"
            )
        result.append(row)
    return result


def load_verified_contract(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    plan_path: str | Path,
    plan_sha256: str,
) -> dict[str, Any]:
    """Verify both files and return the normalized held-out plan."""

    manifest_file = _regular(manifest_path, "motion split manifest")
    plan_file = _regular(plan_path, "held-out evaluation plan")
    _identity(manifest_file, manifest_sha256, "motion split manifest")
    _identity(plan_file, plan_sha256, "held-out evaluation plan")
    try:
        manifest = load_strict_json(manifest_file)
        plan = load_strict_json(plan_file)
    except Exception as exc:
        raise MotionSplitContractError("cannot load strict motion split inputs") from exc
    collection, evaluation = _validate_manifest(manifest)
    document = _exact(
        plan,
        {
            "schema",
            "status",
            "release_eligible",
            "remaining_gate",
            "motion_path_base",
            "source",
            "partition",
            "capture",
        },
        "held-out evaluation plan",
    )
    if document["schema"] != PLAN_SCHEMA:
        raise MotionSplitContractError("held-out evaluation plan schema mismatch")
    if (
        document["status"] != "capture_required"
        or document["release_eligible"] is not False
        or document["remaining_gate"] != "live_capture_attestation_and_repeatability"
        or document["motion_path_base"] != "relative_to_verified_motion_root"
    ):
        raise MotionSplitContractError("held-out plan release-state contract mismatch")
    source = _exact(
        document["source"],
        {
            "motion_count",
            "total_file_bytes",
            "inventory_sha256",
            "inventory_algorithm",
            "nested_split_manifest_name",
            "nested_split_manifest_sha256",
            "nested_split_inventory_sha256",
            "nested_split_inventory_algorithm",
        },
        "held-out plan source",
    )
    if source["motion_count"] != TOTAL_MOTIONS:
        raise MotionSplitContractError("held-out plan source count mismatch")
    _integer(source["total_file_bytes"], "held-out plan source bytes", minimum=1)
    _sha(source["inventory_sha256"], "held-out source inventory")
    if source["nested_split_manifest_sha256"] != manifest_sha256:
        raise MotionSplitContractError("held-out plan binds a different split manifest")
    partition = _exact(
        document["partition"],
        {
            "eval_count",
            "collection_training_count",
            "path_overlap_count",
            "content_overlap_count",
            "eval_runtime_order",
            "collection_runtime_order",
        },
        "held-out plan partition",
    )
    if {
        "eval_count": partition["eval_count"],
        "collection_training_count": partition["collection_training_count"],
        "path_overlap_count": partition["path_overlap_count"],
        "content_overlap_count": partition["content_overlap_count"],
    } != {
        "eval_count": HELDOUT_MOTIONS,
        "collection_training_count": COLLECTION_MOTIONS,
        "path_overlap_count": 0,
        "content_overlap_count": 0,
    }:
        raise MotionSplitContractError("held-out plan partition counts/leakage mismatch")
    eval_rows = _runtime_rows(
        partition["eval_runtime_order"], role="eval", expected=evaluation
    )
    _runtime_rows(
        partition["collection_runtime_order"], role="collection", expected=collection
    )
    capture = _exact(
        document["capture"],
        {
            "contract",
            "contract_sha256",
            "fixed_seeds",
            "seed_count",
            "motions_per_seed",
            "unique_motion_count",
            "case_count",
            "seed_blocks",
            "cases",
        },
        "held-out plan capture",
    )
    _sha(capture["contract_sha256"], "capture contract identity")
    if (
        capture["fixed_seeds"] != list(FIXED_SEEDS)
        or capture["seed_count"] != len(FIXED_SEEDS)
        or capture["motions_per_seed"] != MOTIONS_PER_SEED
        or capture["unique_motion_count"] != HELDOUT_MOTIONS
        or capture["case_count"] != HELDOUT_MOTIONS
    ):
        raise MotionSplitContractError("held-out plan seed topology mismatch")
    blocks = capture["seed_blocks"]
    cases = capture["cases"]
    if not isinstance(blocks, list) or len(blocks) != len(FIXED_SEEDS):
        raise MotionSplitContractError("held-out plan seed blocks are incomplete")
    flattened: list[dict[str, Any]] = []
    for seed_index, (seed, raw_block) in enumerate(zip(FIXED_SEEDS, blocks, strict=True)):
        block = _exact(raw_block, {"seed", "offset", "limit", "cases"}, f"seed block {seed}")
        offset = seed_index * MOTIONS_PER_SEED
        if block["seed"] != seed or block["offset"] != offset or block["limit"] != MOTIONS_PER_SEED:
            raise MotionSplitContractError(f"seed block {seed} mapping mismatch")
        block_cases = block["cases"]
        if not isinstance(block_cases, list) or len(block_cases) != MOTIONS_PER_SEED:
            raise MotionSplitContractError(f"seed block {seed} is incomplete")
        for env_index, (case, motion) in enumerate(
            zip(block_cases, eval_rows[offset : offset + MOTIONS_PER_SEED], strict=True)
        ):
            row = _exact(
                case,
                {
                    "seed",
                    "env_index",
                    "global_case_index",
                    "planned_case_id",
                    "motion_relative_path",
                    "motion_sha256",
                },
                f"seed {seed} case {env_index}",
            )
            if (
                row["seed"] != seed
                or row["env_index"] != env_index
                or row["global_case_index"] != offset + env_index
                or row["motion_relative_path"] != motion["relative_path"]
                or row["motion_sha256"] != motion["sha256"]
            ):
                raise MotionSplitContractError(f"seed {seed} case mapping mismatch")
            _sha(row["planned_case_id"], f"seed {seed} planned case ID")
            flattened.append(row)
    if cases != flattened:
        raise MotionSplitContractError("held-out plan flat cases differ from seed blocks")
    if len({row["motion_sha256"] for row in flattened}) != HELDOUT_MOTIONS:
        raise MotionSplitContractError("held-out plan reuses motion content")
    return document


def seed_block(plan: Mapping[str, Any], seed: int) -> tuple[int, int]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MotionSplitContractError("evaluation seed must be an integer")
    blocks = plan.get("capture", {}).get("seed_blocks", [])
    matches = [block for block in blocks if block.get("seed") == seed]
    if len(matches) != 1:
        raise MotionSplitContractError(f"evaluation seed {seed} has no unique seed block")
    return int(matches[0]["offset"]), int(matches[0]["limit"])


def validate_scenarios(plan: Mapping[str, Any], scenarios: Any) -> None:
    """Require a finalized scenario list to use every planned seed/env once."""

    if not isinstance(scenarios, list):
        raise MotionSplitContractError("scenario contract must be a list")
    expected = [
        (case["seed"], case["env_index"])
        for case in plan["capture"]["cases"]
    ]
    observed: list[tuple[int, int]] = []
    for index, raw in enumerate(scenarios):
        row = _exact(raw, {"seed", "env_index", "scenario_id"}, f"scenario {index}")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise MotionSplitContractError(f"scenario {index} seed is invalid")
        if isinstance(row["env_index"], bool) or not isinstance(row["env_index"], int):
            raise MotionSplitContractError(f"scenario {index} env index is invalid")
        _sha(row["scenario_id"], f"scenario {index} ID")
        observed.append((row["seed"], row["env_index"]))
    if observed != expected:
        raise MotionSplitContractError(
            "scenario seed/env mapping differs from the immutable held-out plan"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "block"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = load_verified_contract(
            manifest_path=arguments.manifest,
            manifest_sha256=arguments.manifest_sha256,
            plan_path=arguments.plan,
            plan_sha256=arguments.plan_sha256,
        )
        if arguments.scenarios is not None:
            scenarios_path = _regular(arguments.scenarios, "scenario contract")
            validate_scenarios(plan, load_strict_json(scenarios_path))
        if arguments.command == "block":
            if arguments.seed is None:
                raise MotionSplitContractError("block resolution requires --seed")
            offset, limit = seed_block(plan, arguments.seed)
            print(offset)
            print(limit)
        else:
            if arguments.seed is not None:
                raise MotionSplitContractError("verify does not accept --seed")
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "collection_motion_count": COLLECTION_MOTIONS,
                        "heldout_motion_count": HELDOUT_MOTIONS,
                        "seed_count": len(FIXED_SEEDS),
                        "motions_per_seed": MOTIONS_PER_SEED,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    except MotionSplitContractError as exc:
        print(f"motion split contract failed: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
