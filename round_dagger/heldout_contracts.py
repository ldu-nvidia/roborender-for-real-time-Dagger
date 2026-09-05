"""Create-only train/validation/test control data for the native-H200 protocol.

The public campaign commitment transitively binds the private test cases:
``commitment -> sealed test registry -> detailed cases``.  Commitment checks
therefore never need to open the detailed case file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import secrets
import stat
from typing import Any

from round_dagger import native_h200_experiment as experiment


MOTION_PARTITION_SCHEMA = "fast_roborender.round_dagger.motion_partition.v1"
PRIVATE_TEST_CASES_SCHEMA = "fast_roborender.round_dagger.sealed_test_cases.v1"
TEST_COMMITMENT_SCHEMA = (
    "fast_roborender.round_dagger.sealed_test_registry_commitment.v1"
)
TEST_CASE_COUNT = 512
TEST_SEED_COUNT = 8
CASES_PER_TEST_SEED = TEST_CASE_COUNT // TEST_SEED_COUNT
CASES_PER_GROUP_PER_SEED = CASES_PER_TEST_SEED // 5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json"
)
GROUP_REGISTRY_PATH = REPOSITORY_ROOT / "configs/round_dagger/native_h200_group_registry.json"
VALIDATION_PANEL_PATH = REPOSITORY_ROOT / "configs/round_dagger/native_h200_validation_panel.json"
CAMPAIGN_CONTRACTS = experiment.CANONICAL_CAMPAIGN_ROOT / "BC/contracts"
HELDOUT_ROOT = experiment.CANONICAL_ARTIFACT_ROOT
VALIDATION_CONTRACTS = HELDOUT_ROOT / "validation/heldout/contracts"
TEST_CONTRACTS = HELDOUT_ROOT / "test/heldout/contracts"

OUTPUT_PATHS = {
    "split": CAMPAIGN_CONTRACTS / "group_split.v1.json",
    "train_partition": CAMPAIGN_CONTRACTS / "train_motion_partition.v1.json",
    "test_commitment": CAMPAIGN_CONTRACTS / "test_registry_commitment.v1.json",
    "validation_partition": VALIDATION_CONTRACTS / "motion_partition.v1.json",
    "validation_registry": VALIDATION_CONTRACTS / "validation_case_registry.v1.json",
    "test_partition": TEST_CONTRACTS / "motion_partition.v1.json",
    "test_registry": TEST_CONTRACTS / "test_registry.v1.json",
    "test_cases": TEST_CONTRACTS / "test_cases.private.v1.json",
}


class HeldoutContractError(RuntimeError):
    """A held-out artifact is malformed, exposed, or would be overwritten."""


def _reject_constant(value: str) -> None:
    raise HeldoutContractError(f"JSON contains forbidden constant {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HeldoutContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise HeldoutContractError(f"not a regular non-symlink file: {path}")
        raw = path.read_bytes()
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except HeldoutContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(f"cannot load JSON {path}: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HeldoutContractError(f"value is not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise HeldoutContractError(f"{label} key-set differs")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HeldoutContractError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_directory(path: Path) -> None:
    if not path.is_absolute():
        raise HeldoutContractError(f"output directory must be absolute: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise HeldoutContractError(f"output directory is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HeldoutContractError(f"unsafe output directory component: {current}")


def publish_create_only(path: Path, document: Mapping[str, Any], *, mode: int) -> str:
    """Publish canonical JSON without ever accepting or replacing an old path."""

    if mode not in {0o400, 0o444}:
        raise HeldoutContractError("published contract mode must be 0400 or 0444")
    _safe_directory(path.parent)
    payload = canonical_bytes(document)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    created: tuple[int, int] | None = None
    try:
        descriptor = os.open(path, flags, mode)
        info = os.fstat(descriptor)
        created = (info.st_dev, info.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HeldoutContractError(f"short write while publishing {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise HeldoutContractError(f"refusing to overwrite existing artifact: {path}") from exc
    except Exception:
        if created is not None:
            try:
                current = os.lstat(path)
                if (current.st_dev, current.st_ino) == created:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(payload).hexdigest()


def motion_partition(split: Mapping[str, Any], purpose: str) -> dict[str, Any]:
    if purpose not in {"train", "validation", "test"}:
        raise HeldoutContractError("motion partition purpose is invalid")
    groups_by_id = {row["group_id"]: row for row in split["groups"]}
    assigned = split["assignments"][purpose]
    rows = []
    for group_id in assigned:
        source = groups_by_id[group_id]
        relative_path = Path(source["relative_path"]).name
        if relative_path != source["group_id"]:
            raise HeldoutContractError(
                "motion partition path must be the group basename relative to motion_file_dir"
            )
        rows.append(
            {
                "group_id": source["group_id"],
                "relative_path": relative_path,
                "size_bytes": source["size_bytes"],
                "motion_sha256": source["motion_sha256"],
            }
        )
    return {
        "schema": MOTION_PARTITION_SCHEMA,
        "split_sha256": split["split_sha256"],
        "purpose": purpose,
        "groups": rows,
    }


def _validation_registry(
    contract: Mapping[str, Any], split: Mapping[str, Any], panel: Any
) -> dict[str, Any]:
    document = _exact(
        panel,
        {"schema", "panel_id", "paired_seeds", "case_ids", "case_count"},
        "validation registry",
    )
    configured = contract["validation_panel"]
    if (
        document["schema"] != experiment.VALIDATION_REGISTRY_SCHEMA
        or document["paired_seeds"] != configured["paired_seeds"]
        or len(document["case_ids"]) != configured["cases_per_seed"]
        or document["case_count"]
        != len(document["paired_seeds"]) * len(document["case_ids"])
        or digest(document) != configured["registry_sha256"]
    ):
        raise HeldoutContractError("validation registry differs from protocol commitment")
    case_groups = []
    for case_id in document["case_ids"]:
        if not isinstance(case_id, str) or "__reset_" not in case_id:
            raise HeldoutContractError("validation case ID does not bind a motion group")
        group_id, reset = case_id.rsplit("__reset_", 1)
        if not reset.isdigit():
            raise HeldoutContractError("validation reset index is malformed")
        case_groups.append(group_id)
    counts = Counter(case_groups)
    if (
        set(counts) != set(split["assignments"]["validation"])
        or max(counts.values()) - min(counts.values()) > 1
        or set(counts) & set(split["assignments"]["test"])
    ):
        raise HeldoutContractError("validation cases do not match the validation split")
    return document


def fresh_test_seeds(forbidden: Sequence[int]) -> list[int]:
    """Generate eight unique nonnegative 63-bit evaluation seeds."""

    blocked = set(forbidden)
    result: list[int] = []
    while len(result) != TEST_SEED_COUNT:
        candidate = secrets.randbits(63)
        if candidate not in blocked and candidate not in result:
            result.append(candidate)
    return result


def _joint_stratum(strata: Mapping[str, str]) -> str:
    aliases = {
        "table_height": "h",
        "initial_planar_distance_bin": "d",
        "initial_world_bearing_bin": "b",
        "trajectory_length_bin": "l",
    }
    return ".".join(f"{aliases[field]}_{strata[field]}" for field in aliases)


def _test_case_documents(
    contract: Mapping[str, Any],
    split: Mapping[str, Any],
    evaluation_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seeds = list(evaluation_seeds)
    validation_seeds = set(contract["validation_panel"]["paired_seeds"])
    if (
        len(seeds) != TEST_SEED_COUNT
        or len(set(seeds)) != TEST_SEED_COUNT
        or any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds)
        or set(seeds) & validation_seeds
    ):
        raise HeldoutContractError("test seeds must be eight fresh unique 63-bit integers")
    test_groups = list(split["assignments"]["test"])
    if len(test_groups) != 5 or test_groups != sorted(set(test_groups)):
        raise HeldoutContractError("test split must contain five sorted unique groups")
    by_group = {row["group_id"]: row for row in split["groups"]}

    # Every seed has 64 cases: four groups receive 13 and one receives 12.
    # Across eight seeds, two groups are omitted once and three twice, yielding
    # exact overall totals 103/103/102/102/102 in a seed-derived permutation.
    seed_commitment = digest({"split_sha256": split["split_sha256"], "seeds": seeds})
    ranked_groups = sorted(
        test_groups,
        key=lambda group_id: hashlib.sha256(
            f"{seed_commitment}:group:{group_id}".encode("utf-8")
        ).hexdigest(),
    )
    omission_multiset = [ranked_groups[0], ranked_groups[1]] + [
        group_id for group_id in ranked_groups[2:] for _ in range(2)
    ]
    omissions = [
        group_id
        for occurrence, group_id in sorted(
            enumerate(omission_multiset),
            key=lambda item: hashlib.sha256(
                f"{seed_commitment}:omit:{item[1]}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
    ]

    cases: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        omitted = omissions[seed_index]
        for group_id in test_groups:
            count = CASES_PER_GROUP_PER_SEED + (group_id != omitted)
            for reset_index in range(count):
                identity = hashlib.sha256(
                    (
                        f"{split['split_sha256']}:{seed_index}:{seed}:"
                        f"{group_id}:{reset_index}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
                cases.append(
                    {
                        "case_id": f"test_case_{len(cases):03d}_{identity}",
                        "evaluation_seed": seed,
                        "group_id": group_id,
                        "reset_index": reset_index,
                    }
                )
    if len(cases) != TEST_CASE_COUNT or len({row["case_id"] for row in cases}) != len(cases):
        raise HeldoutContractError("test case construction is incomplete or duplicated")

    group_counts = Counter(row["group_id"] for row in cases)
    if max(group_counts.values()) - min(group_counts.values()) > 1:
        raise HeldoutContractError("test groups differ by more than one case")
    strata_counts: dict[str, dict[str, int]] = {
        field: {} for field in contract["group_split"]["stratify_by"]
    }
    joint_counts: Counter[str] = Counter()
    for group_id, count in group_counts.items():
        strata = by_group[group_id]["strata"]
        joint_counts[_joint_stratum(strata)] += count
        for field, value in strata.items():
            strata_counts[field][value] = strata_counts[field].get(value, 0) + count
    private_cases = {
        "schema": PRIVATE_TEST_CASES_SCHEMA,
        "protocol_id": contract["protocol_id"],
        "split_sha256": split["split_sha256"],
        "case_count": TEST_CASE_COUNT,
        "evaluation_seeds": seeds,
        "cases": cases,
        "expected_counts": {
            "overall": TEST_CASE_COUNT,
            "by_group": dict(sorted(group_counts.items())),
            "by_stratum": {
                field: dict(sorted(counts.items()))
                for field, counts in strata_counts.items()
            },
        },
    }
    cases_sha256 = digest(private_cases)
    test_registry = {
        "schema": experiment.TEST_REGISTRY_SCHEMA,
        "groups": test_groups,
        "case_count": TEST_CASE_COUNT,
        "cases_sha256": cases_sha256,
        "stratum_case_counts": dict(sorted(joint_counts.items())),
    }
    test_registry_sha256 = digest(test_registry)
    commitment = {
        "schema": TEST_COMMITMENT_SCHEMA,
        "protocol_id": contract["protocol_id"],
        "split_sha256": split["split_sha256"],
        "test_registry_sha256": test_registry_sha256,
        "test_cases_sha256": cases_sha256,
    }
    return private_cases, test_registry, commitment


def build_documents(
    contract: Mapping[str, Any],
    registry: Any,
    validation_panel: Any,
    *,
    evaluation_seeds: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    checked_contract = experiment.validate_contract(contract)
    split = experiment.build_group_split(checked_contract, registry)
    panel = _validation_registry(checked_contract, split, validation_panel)
    seeds = (
        fresh_test_seeds(checked_contract["validation_panel"]["paired_seeds"])
        if evaluation_seeds is None
        else list(evaluation_seeds)
    )
    private_cases, test_registry, commitment = _test_case_documents(
        checked_contract, split, seeds
    )
    return {
        "split": split,
        "train_partition": motion_partition(split, "train"),
        "validation_partition": motion_partition(split, "validation"),
        "validation_registry": panel,
        "test_partition": motion_partition(split, "test"),
        "test_cases": private_cases,
        "test_registry": test_registry,
        "test_commitment": commitment,
    }


def verify_commitment(commitment_path: Path, test_registry_path: Path) -> dict[str, Any]:
    """Verify the public-to-private commitment without reading detailed cases."""

    commitment = _exact(
        load_json(commitment_path),
        {
            "schema",
            "protocol_id",
            "split_sha256",
            "test_registry_sha256",
            "test_cases_sha256",
        },
        "test commitment",
    )
    registry = _exact(
        load_json(test_registry_path),
        {"schema", "groups", "case_count", "cases_sha256", "stratum_case_counts"},
        "sealed test registry",
    )
    if commitment["schema"] != TEST_COMMITMENT_SCHEMA:
        raise HeldoutContractError("test commitment schema differs")
    if registry["schema"] != experiment.TEST_REGISTRY_SCHEMA:
        raise HeldoutContractError("sealed test registry schema differs")
    if digest(registry) != _sha(
        commitment["test_registry_sha256"], "test registry commitment"
    ):
        raise HeldoutContractError("sealed test registry digest differs")
    if registry["cases_sha256"] != _sha(
        commitment["test_cases_sha256"], "test cases commitment"
    ):
        raise HeldoutContractError("transitive test case commitment differs")
    mode = stat.S_IMODE(os.lstat(test_registry_path).st_mode)
    if mode != 0o400:
        raise HeldoutContractError("private test registry must have mode 0400")
    return {
        "status": "verified",
        "test_registry_sha256": commitment["test_registry_sha256"],
        "test_cases_sha256": commitment["test_cases_sha256"],
        "case_count": registry["case_count"],
    }


def materialize() -> dict[str, Any]:
    contract = load_json(PROTOCOL_PATH)
    registry = load_json(GROUP_REGISTRY_PATH)
    panel = load_json(VALIDATION_PANEL_PATH)
    documents = build_documents(contract, registry, panel)
    modes = {name: (0o400 if name.startswith("test_") and name != "test_commitment" else 0o444)
             for name in OUTPUT_PATHS}
    # The public commitment is deliberately last: its existence means all
    # transitive private and role-only inputs were published successfully.
    order = [name for name in OUTPUT_PATHS if name != "test_commitment"] + [
        "test_commitment"
    ]
    files = {}
    for name in order:
        path = OUTPUT_PATHS[name]
        files[name] = {
            "path": str(path),
            "mode": format(modes[name], "04o"),
            "file_sha256": publish_create_only(
                path, documents[name], mode=modes[name]
            ),
            "document_sha256": digest(documents[name]),
        }
    verification = verify_commitment(
        OUTPUT_PATHS["test_commitment"], OUTPUT_PATHS["test_registry"]
    )
    return {
        "status": "materialized",
        "split_sha256": documents["split"]["split_sha256"],
        "validation_registry_sha256": digest(documents["validation_registry"]),
        "commitment_verification": verification["status"],
        "test_registry_sha256": verification["test_registry_sha256"],
        "test_cases_sha256": verification["test_cases_sha256"],
        "case_count": verification["case_count"],
        "files": files,
    }


__all__ = [
    "HeldoutContractError",
    "OUTPUT_PATHS",
    "build_documents",
    "digest",
    "materialize",
    "motion_partition",
    "publish_create_only",
    "verify_commitment",
]
