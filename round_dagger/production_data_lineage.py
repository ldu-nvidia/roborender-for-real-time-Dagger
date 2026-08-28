"""Immutable production data/runtime lineage for round-based DAgger.

The checked-in production JSON is an invariant profile, not a place to write
hashes that only exist after camera acceptance, renderer-fleet validation, and
live held-out capture.  Those post-seal identities live in one external,
create-once runtime contract.  This module verifies that contract byte for
byte and derives the compact lineage copied into every durable campaign,
round, cohort, shard, seal, and paired-evaluation identity.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from round_dagger.first_condition_bundle import (
    FirstConditionBundleError,
    VIEWS as FIRST_CONDITION_VIEWS,
    load_first_condition_bundle,
)

from round_dagger.scenario_capture import (
    ATTESTATION_SCHEMA,
    EXPECTED_ENVS_PER_SEED,
    EXPECTED_SEEDS,
    canonical_sha256 as canonical_json_sha256,
    load_strict_json as _scenario_load_strict_json,
    verify_campaign_outputs,
)


RUNTIME_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.production_runtime_contract.v2"
)
DATA_LINEAGE_SCHEMA = "fast_roborender.round_dagger.production_data_lineage.v2"
COLLECTION_POLICY_SCHEMA = (
    "fast_roborender.round_dagger.collection_assignment_policy.v1"
)
COHORT_ASSIGNMENT_SCHEMA = (
    "fast_roborender.round_dagger.cohort_motion_assignment.v1"
)
COLLECTION_ASSIGNMENT_POLICY = (
    "cohort_rank_contiguous_offset_modulo_collection_motion_count.v1"
)

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_RUNTIME_KEYS = {
    "schema",
    "status",
    "invariant_profile_sha256",
    "camera_acceptance",
    "renderer_fleet",
    "live_capture",
}
_CAMERA_KEYS = {"path", "sha256"}
_FLEET_KEYS = {
    "endpoint_manifest_path",
    "endpoint_manifest_sha256",
    "endpoint_directory",
    "renderer_count",
    "cohort_identity_sha256",
    "runtime_identity_sha256",
}
_CAPTURE_KEYS = {
    "split_manifest_path",
    "split_manifest_sha256",
    "heldout_plan_path",
    "heldout_plan_sha256",
    "pass_artifacts",
    "attestation_path",
    "attestation_sha256",
    "evidence_path",
    "evidence_sha256",
    "scenario_path",
    "scenario_sha256",
    "first_condition_bundles",
    "first_condition_bundle_inventory_sha256",
}
_PASS_KEYS = {"seed", "pass_index", "path", "sha256"}
_BUNDLE_KEYS = {
    "seed",
    "path",
    "sha256",
    "size",
    "case_count",
    "source_pass_index",
}
_LINEAGE_KEYS = {
    "schema",
    "runtime_contract_sha256",
    "invariant_profile_sha256",
    "motion_split",
    "collection_assignment_policy",
    "heldout_evaluation",
}
_LINEAGE_SPLIT_KEYS = {
    "manifest_sha256",
    "heldout_plan_sha256",
    "split_contract_sha256",
    "collection_role",
    "evaluation_role",
    "collection_motion_count",
    "heldout_motion_count",
}
_COLLECTION_POLICY_KEYS = {
    "schema",
    "policy",
    "cohort_count",
    "actor_ranks_per_cohort",
    "environments_per_actor_rank",
    "cohort_stride",
    "rank_stride",
    "motion_assignment_base_offset",
}
_HELDOUT_KEYS = {
    "capture_attestation_sha256",
    "capture_evidence_sha256",
    "scenario_contract_sha256",
    "capture_provenance_sha256",
    "case_count",
    "fixed_seeds",
    "environments_per_seed",
    "capture_attestation_schema",
    "first_condition_bundles",
    "first_condition_bundle_inventory_sha256",
}
_COHORT_ASSIGNMENT_KEYS = {
    "schema",
    "policy",
    "data_lineage_sha256",
    "split_manifest_sha256",
    "collection_role",
    "collection_motion_count",
    "cohort_index",
    "cohort_assignment_offset",
    "rank_assignments",
}
_RANK_ASSIGNMENT_KEYS = {
    "rank",
    "environment_count",
    "motion_assignment_offset",
}


class ProductionDataLineageError(RuntimeError):
    """A materialized runtime or durable data-lineage identity is invalid."""


def load_strict_json(path: str | os.PathLike[str]) -> Any:
    try:
        document, _ = _scenario_load_strict_json(path, "production lineage JSON")
    except Exception as exc:
        raise ProductionDataLineageError(f"cannot load strict JSON: {exc}") from exc
    return document


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionDataLineageError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ProductionDataLineageError(
            f"{label} fields differ: missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ProductionDataLineageError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProductionDataLineageError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _absolute(value: Any, label: str, *, directory: bool = False) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionDataLineageError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ProductionDataLineageError(
            f"{label} must be a normalized absolute path"
        )
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProductionDataLineageError(f"{label} is unavailable: {path}") from exc
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if stat.S_ISLNK(info.st_mode) or not expected:
        kind = "directory" if directory else "regular file"
        raise ProductionDataLineageError(
            f"{label} must be a non-symlink {kind}: {path}"
        )
    return path


def _file(value: Any, expected_sha: Any, label: str) -> Path:
    path = _absolute(value, label)
    expected = _sha(expected_sha, f"{label} SHA-256")
    if sha256_file(path) != expected:
        raise ProductionDataLineageError(f"{label} byte identity mismatch")
    return path


def _load_json_bytes(path: Path, label: str) -> tuple[Any, bytes]:
    # scenario_capture/load_strict_json already rejects duplicate keys and
    # unsafe path types.  Re-read only after the lstat/hash gate above.
    document = load_strict_json(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:  # pragma: no cover - prior open normally catches it
        raise ProductionDataLineageError(f"cannot read {label}") from exc
    return document, payload


def _validate_capture_provenance(
    provenance: Any,
    *,
    split_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ProductionDataLineageError("capture provenance must be an object")
    normalized = dict(provenance)
    for key, value in normalized.items():
        if key in {"fast_roborender_revision", "gr00t_revision"}:
            if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
                raise ProductionDataLineageError(
                    f"capture provenance {key} is malformed"
                )
        else:
            _sha(value, f"capture provenance {key}")
    if normalized.get("motion_split_manifest_sha256") != split_sha256:
        raise ProductionDataLineageError(
            "capture provenance uses another motion split"
        )
    if normalized.get("heldout_plan_sha256") != plan_sha256:
        raise ProductionDataLineageError("capture provenance uses another plan")
    return normalized


def _validate_first_condition_bundles(
    value: Any,
    *,
    inventory_sha256: Any,
    plan_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_SEEDS):
        raise ProductionDataLineageError(
            "live capture must bind exactly eight first-condition bundles"
        )
    rows: list[dict[str, Any]] = []
    bundle_provenance: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _BUNDLE_KEYS, f"first_condition_bundles[{index}]")
        seed = _integer(row["seed"], "first-condition bundle seed")
        if seed != EXPECTED_SEEDS[index]:
            raise ProductionDataLineageError(
                "first-condition bundles are not in canonical fixed-seed order"
            )
        path = _absolute(row["path"], "first-condition bundle")
        digest = _sha(row["sha256"], "first-condition bundle SHA-256")
        size = _integer(row["size"], "first-condition bundle size", minimum=1)
        if (
            row["case_count"] != EXPECTED_ENVS_PER_SEED
            or row["source_pass_index"] != 1
        ):
            raise ProductionDataLineageError(
                "first-condition bundle row is not source-pass-1 32-case data"
            )
        try:
            bundle = load_first_condition_bundle(
                path,
                expected_sha256=digest,
                expected_seed=seed,
                expected_env_count=EXPECTED_ENVS_PER_SEED,
                expected_views=FIRST_CONDITION_VIEWS,
                expected_plan_sha256=plan_sha256,
            )
        except FirstConditionBundleError as exc:
            raise ProductionDataLineageError(
                f"first-condition bundle deep verification failed: {exc}"
            ) from exc
        if bundle["size"] != size:
            raise ProductionDataLineageError("first-condition bundle size mismatch")
        rows.append(row)
        # Do not retain the ~64 MiB request arrays for all eight seeds at once.
        bundle_provenance.append(dict(bundle["provenance"]))
    expected_inventory = _sha(
        inventory_sha256, "first-condition bundle inventory SHA-256"
    )
    if canonical_json_sha256(rows) != expected_inventory:
        raise ProductionDataLineageError(
            "first-condition bundle inventory SHA-256 mismatch"
        )
    return rows, bundle_provenance


def load_verified_runtime_contract(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_invariant_profile_sha256: str,
    motion_split_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a post-seal runtime contract and all live-capture evidence.

    The caller must supply the runtime-contract digest out of band.  Merely
    swapping the path therefore cannot replace camera/fleet/capture evidence.
    """

    contract_path = _file(path, expected_sha256, "production runtime contract")
    raw_document, raw_payload = _load_json_bytes(
        contract_path, "production runtime contract"
    )
    canonical_payload = (
        json.dumps(
            raw_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw_payload != canonical_payload:
        raise ProductionDataLineageError(
            "production runtime contract is not canonical JSON"
        )
    document = _exact(
        raw_document, _RUNTIME_KEYS, "production runtime contract"
    )
    if document["schema"] != RUNTIME_CONTRACT_SCHEMA:
        raise ProductionDataLineageError("production runtime contract schema mismatch")
    if document["status"] != "release_eligible":
        raise ProductionDataLineageError(
            "production runtime contract is not release eligible"
        )
    invariant_sha = _sha(
        document["invariant_profile_sha256"], "invariant profile SHA-256"
    )
    if invariant_sha != _sha(
        expected_invariant_profile_sha256, "expected invariant profile SHA-256"
    ):
        raise ProductionDataLineageError(
            "runtime contract belongs to another invariant profile"
        )

    camera = _exact(document["camera_acceptance"], _CAMERA_KEYS, "camera acceptance")
    _file(camera["path"], camera["sha256"], "camera acceptance")

    fleet = _exact(document["renderer_fleet"], _FLEET_KEYS, "renderer fleet")
    _file(
        fleet["endpoint_manifest_path"],
        fleet["endpoint_manifest_sha256"],
        "renderer endpoint manifest",
    )
    _absolute(fleet["endpoint_directory"], "renderer endpoint directory", directory=True)
    _integer(fleet["renderer_count"], "renderer_count", minimum=1)
    _sha(fleet["cohort_identity_sha256"], "renderer cohort identity")
    _sha(fleet["runtime_identity_sha256"], "renderer runtime identity")

    capture = _exact(document["live_capture"], _CAPTURE_KEYS, "live capture")
    split_path = _file(
        capture["split_manifest_path"],
        capture["split_manifest_sha256"],
        "motion split manifest",
    )
    plan_path = _file(
        capture["heldout_plan_path"],
        capture["heldout_plan_sha256"],
        "held-out capture plan",
    )
    expected_split = _sha(
        motion_split_contract.get("manifest_sha256"),
        "invariant motion split SHA-256",
    )
    expected_plan = _sha(
        motion_split_contract.get("heldout_plan_sha256"),
        "invariant held-out plan SHA-256",
    )
    if capture["split_manifest_sha256"] != expected_split:
        raise ProductionDataLineageError("runtime contract uses another motion split")
    if capture["heldout_plan_sha256"] != expected_plan:
        raise ProductionDataLineageError("runtime contract uses another held-out plan")

    bundle_rows, bundle_provenance = _validate_first_condition_bundles(
        capture["first_condition_bundles"],
        inventory_sha256=capture["first_condition_bundle_inventory_sha256"],
        plan_sha256=expected_plan,
    )

    pass_rows = capture["pass_artifacts"]
    if not isinstance(pass_rows, list) or len(pass_rows) != 16:
        raise ProductionDataLineageError(
            "live capture must bind exactly 16 seed/pass artifacts"
        )
    expected_pairs = [
        (seed, pass_index)
        for seed in EXPECTED_SEEDS
        for pass_index in (1, 2)
    ]
    observed_pairs: list[tuple[int, int]] = []
    pass_paths: list[Path] = []
    for index, raw in enumerate(pass_rows):
        row = _exact(raw, _PASS_KEYS, f"live capture pass_artifacts[{index}]")
        seed = _integer(row["seed"], "capture pass seed")
        pass_index = _integer(row["pass_index"], "capture pass index", minimum=1)
        observed_pairs.append((seed, pass_index))
        pass_paths.append(
            _file(row["path"], row["sha256"], "scenario capture pass artifact")
        )
    if observed_pairs != expected_pairs:
        raise ProductionDataLineageError(
            "live capture pass artifacts are not canonical fixed-seed two-pass coverage"
        )

    attestation_path = _file(
        capture["attestation_path"],
        capture["attestation_sha256"],
        "scenario capture attestation",
    )
    evidence_path = _file(
        capture["evidence_path"],
        capture["evidence_sha256"],
        "scenario capture evidence",
    )
    scenario_path = _file(
        capture["scenario_path"],
        capture["scenario_sha256"],
        "held-out scenario contract",
    )
    try:
        verification = verify_campaign_outputs(
            plan_path=plan_path,
            pass_paths=pass_paths,
            attestation_path=attestation_path,
            evidence_path=evidence_path,
            scenario_path=scenario_path,
            production=True,
        )
    except Exception as exc:
        raise ProductionDataLineageError(
            f"live capture bundle verification failed: {exc}"
        ) from exc
    expected_output = {
        "attestation_sha256": capture["attestation_sha256"],
        "capture_evidence_sha256": capture["evidence_sha256"],
        "scenario_contract_sha256": capture["scenario_sha256"],
        "first_condition_bundle_inventory_sha256": capture[
            "first_condition_bundle_inventory_sha256"
        ],
        "case_count": 256,
        "status": "PASS",
        "release_eligible": True,
    }
    if verification != expected_output:
        raise ProductionDataLineageError(
            "live capture verification identities differ from runtime contract"
        )
    attestation, _ = _load_json_bytes(attestation_path, "capture attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("release_eligible") is not True
        or attestation.get("case_count") != 256
        or attestation.get("first_condition_bundles") != bundle_rows
        or attestation.get("first_condition_bundle_inventory_sha256")
        != capture["first_condition_bundle_inventory_sha256"]
    ):
        raise ProductionDataLineageError(
            "live capture attestation is not release-eligible 8x32 evidence"
        )
    provenance = _validate_capture_provenance(
        attestation.get("provenance"),
        split_sha256=expected_split,
        plan_sha256=expected_plan,
    )
    if any(item != provenance for item in bundle_provenance):
        raise ProductionDataLineageError(
            "first-condition bundle provenance differs from capture attestation"
        )

    normalized = {
        **document,
        "contract_path": str(contract_path),
        "contract_sha256": expected_sha256,
        "camera_acceptance": camera,
        "renderer_fleet": fleet,
        "live_capture": capture,
        "capture_verification": verification,
        "capture_provenance": provenance,
        "split_manifest_path": str(split_path),
        "heldout_plan_path": str(plan_path),
        "scenario_path": str(scenario_path),
    }
    return normalized


def derive_data_lineage(
    runtime_contract: Mapping[str, Any],
    *,
    motion_split_contract: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact compact identity persisted by the coordinator."""

    capture = runtime_contract["live_capture"]
    provenance = runtime_contract["capture_provenance"]
    policy = {
        "schema": COLLECTION_POLICY_SCHEMA,
        "policy": COLLECTION_ASSIGNMENT_POLICY,
        "cohort_count": _integer(topology.get("cohort_count"), "cohort_count", minimum=1),
        "actor_ranks_per_cohort": _integer(
            topology.get("actor_ranks_per_cohort"),
            "actor_ranks_per_cohort",
            minimum=1,
        ),
        "environments_per_actor_rank": _integer(
            topology.get("environments_per_actor_rank"),
            "environments_per_actor_rank",
            minimum=1,
        ),
        "cohort_stride": _integer(
            topology.get("active_environments"), "active_environments", minimum=1
        ),
        "rank_stride": _integer(
            topology.get("environments_per_actor_rank"),
            "rank_stride",
            minimum=1,
        ),
        "motion_assignment_base_offset": 0,
    }
    if policy["cohort_stride"] != (
        policy["actor_ranks_per_cohort"] * policy["environments_per_actor_rank"]
    ):
        raise ProductionDataLineageError(
            "collection assignment topology is not rank-complete"
        )
    split = {
        "manifest_sha256": _sha(
            motion_split_contract.get("manifest_sha256"), "split manifest SHA-256"
        ),
        "heldout_plan_sha256": _sha(
            motion_split_contract.get("heldout_plan_sha256"), "held-out plan SHA-256"
        ),
        "split_contract_sha256": canonical_json_sha256(
            dict(motion_split_contract)
        ),
        "collection_role": motion_split_contract.get("collection_role"),
        "evaluation_role": motion_split_contract.get("evaluation_role"),
        "collection_motion_count": motion_split_contract.get(
            "collection_motion_count"
        ),
        "heldout_motion_count": motion_split_contract.get("heldout_motion_count"),
    }
    lineage = {
        "schema": DATA_LINEAGE_SCHEMA,
        "runtime_contract_sha256": _sha(
            runtime_contract.get("contract_sha256"), "runtime contract SHA-256"
        ),
        "invariant_profile_sha256": _sha(
            runtime_contract.get("invariant_profile_sha256"),
            "invariant profile SHA-256",
        ),
        "motion_split": split,
        "collection_assignment_policy": policy,
        "heldout_evaluation": {
            "capture_attestation_sha256": capture["attestation_sha256"],
            "capture_evidence_sha256": capture["evidence_sha256"],
            "scenario_contract_sha256": capture["scenario_sha256"],
            "capture_provenance_sha256": canonical_json_sha256(provenance),
            "case_count": 256,
            "fixed_seeds": list(EXPECTED_SEEDS),
            "environments_per_seed": EXPECTED_ENVS_PER_SEED,
            "capture_attestation_schema": ATTESTATION_SCHEMA,
            "first_condition_bundles": capture["first_condition_bundles"],
            "first_condition_bundle_inventory_sha256": capture[
                "first_condition_bundle_inventory_sha256"
            ],
        },
    }
    return validate_data_lineage(lineage)


def validate_data_lineage(value: Any) -> dict[str, Any]:
    lineage = _exact(value, _LINEAGE_KEYS, "production data lineage")
    if lineage["schema"] != DATA_LINEAGE_SCHEMA:
        raise ProductionDataLineageError("production data lineage schema mismatch")
    _sha(lineage["runtime_contract_sha256"], "runtime contract SHA-256")
    _sha(lineage["invariant_profile_sha256"], "invariant profile SHA-256")
    split = _exact(lineage["motion_split"], _LINEAGE_SPLIT_KEYS, "motion split lineage")
    for key in ("manifest_sha256", "heldout_plan_sha256", "split_contract_sha256"):
        _sha(split[key], f"motion split {key}")
    if split["collection_role"] != "collection" or split["evaluation_role"] != "eval":
        raise ProductionDataLineageError("motion split roles are not collection/eval")
    if (
        _integer(split["collection_motion_count"], "collection_motion_count", minimum=1)
        != 39
        or _integer(split["heldout_motion_count"], "heldout_motion_count", minimum=1)
        != 256
    ):
        raise ProductionDataLineageError("production motion split is not exact 39/256")
    policy = _exact(
        lineage["collection_assignment_policy"],
        _COLLECTION_POLICY_KEYS,
        "collection assignment policy",
    )
    if (
        policy["schema"] != COLLECTION_POLICY_SCHEMA
        or policy["policy"] != COLLECTION_ASSIGNMENT_POLICY
        or policy["cohort_count"] != 4
        or policy["actor_ranks_per_cohort"] != 2
        or policy["environments_per_actor_rank"] != 32
        or policy["cohort_stride"] != 64
        or policy["rank_stride"] != 32
        or policy["motion_assignment_base_offset"] != 0
    ):
        raise ProductionDataLineageError(
            "production collection assignment policy is not exact 4x2x32"
        )
    heldout = _exact(
        lineage["heldout_evaluation"], _HELDOUT_KEYS, "held-out evaluation lineage"
    )
    for key in (
        "capture_attestation_sha256",
        "capture_evidence_sha256",
        "scenario_contract_sha256",
        "capture_provenance_sha256",
        "first_condition_bundle_inventory_sha256",
    ):
        _sha(heldout[key], f"held-out {key}")
    if (
        heldout["case_count"] != 256
        or heldout["fixed_seeds"] != list(EXPECTED_SEEDS)
        or heldout["environments_per_seed"] != EXPECTED_ENVS_PER_SEED
        or heldout["capture_attestation_schema"] != ATTESTATION_SCHEMA
    ):
        raise ProductionDataLineageError(
            "held-out evaluation lineage is not exact release-eligible 8x32"
        )
    rows = heldout["first_condition_bundles"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(EXPECTED_SEEDS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != _BUNDLE_KEYS
            or row.get("seed") != EXPECTED_SEEDS[index]
            or row.get("case_count") != EXPECTED_ENVS_PER_SEED
            or row.get("source_pass_index") != 1
            for index, row in enumerate(rows)
        )
        or canonical_json_sha256(rows)
        != heldout["first_condition_bundle_inventory_sha256"]
    ):
        raise ProductionDataLineageError(
            "held-out first-condition bundle inventory is not exact"
        )
    for row in rows:
        _sha(row["sha256"], "held-out first-condition bundle SHA-256")
        if (
            not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or type(row["size"]) is not int
            or row["size"] <= 0
        ):
            raise ProductionDataLineageError(
                "held-out first-condition bundle row is malformed"
            )
    return lineage


def data_lineage_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(validate_data_lineage(value))


def build_cohort_assignment(
    data_lineage: Mapping[str, Any],
    *,
    cohort_index: int,
    expected_ranks: Sequence[int],
) -> dict[str, Any]:
    lineage = validate_data_lineage(data_lineage)
    policy = lineage["collection_assignment_policy"]
    index = _integer(cohort_index, "cohort_index")
    if index >= policy["cohort_count"]:
        raise ProductionDataLineageError("cohort_index is outside assignment policy")
    ranks = list(expected_ranks)
    if ranks != list(range(policy["actor_ranks_per_cohort"])):
        raise ProductionDataLineageError(
            "expected ranks differ from production assignment policy"
        )
    cohort_offset = (
        policy["motion_assignment_base_offset"] + index * policy["cohort_stride"]
    )
    assignment = {
        "schema": COHORT_ASSIGNMENT_SCHEMA,
        "policy": COLLECTION_ASSIGNMENT_POLICY,
        "data_lineage_sha256": data_lineage_sha256(lineage),
        "split_manifest_sha256": lineage["motion_split"]["manifest_sha256"],
        "collection_role": lineage["motion_split"]["collection_role"],
        "collection_motion_count": lineage["motion_split"][
            "collection_motion_count"
        ],
        "cohort_index": index,
        "cohort_assignment_offset": cohort_offset,
        "rank_assignments": [
            {
                "rank": rank,
                "environment_count": policy["environments_per_actor_rank"],
                "motion_assignment_offset": cohort_offset
                + rank * policy["rank_stride"],
            }
            for rank in ranks
        ],
    }
    return validate_cohort_assignment(
        assignment,
        data_lineage=lineage,
        cohort_index=index,
        expected_ranks=ranks,
    )


def validate_cohort_assignment(
    value: Any,
    *,
    data_lineage: Mapping[str, Any],
    cohort_index: int,
    expected_ranks: Sequence[int],
) -> dict[str, Any]:
    assignment = _exact(value, _COHORT_ASSIGNMENT_KEYS, "cohort motion assignment")
    lineage = validate_data_lineage(data_lineage)
    policy = lineage["collection_assignment_policy"]
    index = _integer(cohort_index, "cohort_index")
    if (
        assignment["schema"] != COHORT_ASSIGNMENT_SCHEMA
        or assignment["policy"] != COLLECTION_ASSIGNMENT_POLICY
        or assignment["data_lineage_sha256"] != data_lineage_sha256(lineage)
        or assignment["split_manifest_sha256"]
        != lineage["motion_split"]["manifest_sha256"]
        or assignment["collection_role"] != "collection"
        or assignment["collection_motion_count"] != 39
        or assignment["cohort_index"] != index
    ):
        raise ProductionDataLineageError("cohort assignment lineage mismatch")
    expected_offset = index * policy["cohort_stride"]
    if assignment["cohort_assignment_offset"] != expected_offset:
        raise ProductionDataLineageError("cohort motion assignment offset mismatch")
    rows = assignment["rank_assignments"]
    ranks = list(expected_ranks)
    if not isinstance(rows, list) or len(rows) != len(ranks):
        raise ProductionDataLineageError("rank assignment coverage is incomplete")
    for position, (raw, rank) in enumerate(zip(rows, ranks, strict=True)):
        row = _exact(raw, _RANK_ASSIGNMENT_KEYS, f"rank_assignments[{position}]")
        if row != {
            "rank": rank,
            "environment_count": policy["environments_per_actor_rank"],
            "motion_assignment_offset": expected_offset
            + rank * policy["rank_stride"],
        }:
            raise ProductionDataLineageError("rank motion assignment offset mismatch")
    return assignment


def rank_assignment(
    assignment: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    for row in assignment.get("rank_assignments", []):
        if isinstance(row, Mapping) and row.get("rank") == rank:
            return dict(row)
    raise ProductionDataLineageError("rank is absent from cohort motion assignment")


def _profile(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    target = _absolute(str(path), "invariant profile")
    document = load_strict_json(target)
    if not isinstance(document, Mapping) or not isinstance(
        document.get("motion_split"), Mapping
    ):
        raise ProductionDataLineageError(
            "invariant profile has no motion_split contract"
        )
    return dict(document), sha256_file(target)


def _parse_pass_spec(value: str) -> dict[str, Any]:
    try:
        seed_text, pass_text, path_text = value.split(":", 2)
        seed = int(seed_text)
        pass_index = int(pass_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "capture pass must be SEED:PASS_INDEX:/absolute/path.json"
        ) from exc
    path = Path(path_text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("capture pass path must be absolute")
    return {
        "seed": seed,
        "pass_index": pass_index,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _parse_bundle_spec(value: str) -> dict[str, Any]:
    try:
        seed_text, path_text = value.split(":", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "first-condition bundle must be SEED:/absolute/path.npz"
        ) from exc
    path = Path(path_text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            "first-condition bundle path must be absolute"
        )
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except (OSError, ProductionDataLineageError) as exc:
        raise argparse.ArgumentTypeError(
            "first-condition bundle cannot be inspected"
        ) from exc
    return {
        "seed": seed,
        "path": str(path),
        "sha256": digest,
        "size": size,
        "case_count": EXPECTED_ENVS_PER_SEED,
        "source_pass_index": 1,
    }


def _publish_runtime_contract(
    output: Path,
    document: Mapping[str, Any],
    *,
    invariant_profile: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ProductionDataLineageError(
            "runtime contract output must be a new absolute path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        verified = load_verified_runtime_contract(
            temporary,
            expected_sha256=digest,
            expected_invariant_profile_sha256=document[
                "invariant_profile_sha256"
            ],
            motion_split_contract=invariant_profile["motion_split"],
        )
        os.link(temporary, output, follow_symlinks=False)
        directory = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest, verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--invariant-profile", required=True)
    materialize.add_argument("--camera-acceptance", required=True)
    materialize.add_argument("--endpoint-manifest", required=True)
    materialize.add_argument("--endpoint-directory", required=True)
    materialize.add_argument("--renderer-count", type=int, required=True)
    materialize.add_argument("--renderer-cohort-sha256", required=True)
    materialize.add_argument("--renderer-runtime-sha256", required=True)
    materialize.add_argument("--split-manifest", required=True)
    materialize.add_argument("--heldout-plan", required=True)
    materialize.add_argument(
        "--capture-pass",
        action="append",
        type=_parse_pass_spec,
        required=True,
        help="repeat 16 times as SEED:PASS_INDEX:/absolute/pass.json",
    )
    materialize.add_argument("--capture-attestation", required=True)
    materialize.add_argument("--capture-evidence", required=True)
    materialize.add_argument("--scenario-contract", required=True)
    materialize.add_argument(
        "--first-condition-bundle",
        action="append",
        type=_parse_bundle_spec,
        required=True,
        help="repeat eight times as SEED:/absolute/first_condition_bundle.npz",
    )
    materialize.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--invariant-profile", required=True)
    verify.add_argument("--campaign-contract", required=True)
    verify.add_argument("--campaign-contract-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile, profile_sha = _profile(args.invariant_profile)
        if args.command == "materialize":
            first_condition_bundles = sorted(
                args.first_condition_bundle, key=lambda row: row["seed"]
            )
            document = {
                "schema": RUNTIME_CONTRACT_SCHEMA,
                "status": "release_eligible",
                "invariant_profile_sha256": profile_sha,
                "camera_acceptance": {
                    "path": str(_absolute(args.camera_acceptance, "camera acceptance")),
                    "sha256": sha256_file(args.camera_acceptance),
                },
                "renderer_fleet": {
                    "endpoint_manifest_path": str(
                        _absolute(args.endpoint_manifest, "endpoint manifest")
                    ),
                    "endpoint_manifest_sha256": sha256_file(
                        args.endpoint_manifest
                    ),
                    "endpoint_directory": str(
                        _absolute(
                            args.endpoint_directory,
                            "endpoint directory",
                            directory=True,
                        )
                    ),
                    "renderer_count": args.renderer_count,
                    "cohort_identity_sha256": args.renderer_cohort_sha256,
                    "runtime_identity_sha256": args.renderer_runtime_sha256,
                },
                "live_capture": {
                    "split_manifest_path": str(
                        _absolute(args.split_manifest, "split manifest")
                    ),
                    "split_manifest_sha256": sha256_file(args.split_manifest),
                    "heldout_plan_path": str(
                        _absolute(args.heldout_plan, "held-out plan")
                    ),
                    "heldout_plan_sha256": sha256_file(args.heldout_plan),
                    "pass_artifacts": args.capture_pass,
                    "attestation_path": str(
                        _absolute(args.capture_attestation, "capture attestation")
                    ),
                    "attestation_sha256": sha256_file(args.capture_attestation),
                    "evidence_path": str(
                        _absolute(args.capture_evidence, "capture evidence")
                    ),
                    "evidence_sha256": sha256_file(args.capture_evidence),
                    "scenario_path": str(
                        _absolute(args.scenario_contract, "scenario contract")
                    ),
                    "scenario_sha256": sha256_file(args.scenario_contract),
                    "first_condition_bundles": first_condition_bundles,
                    "first_condition_bundle_inventory_sha256": canonical_json_sha256(
                        first_condition_bundles
                    ),
                },
            }
            digest, runtime = _publish_runtime_contract(
                args.output,
                document,
                invariant_profile=profile,
            )
        else:
            digest = _sha(
                args.campaign_contract_sha256, "campaign contract SHA-256"
            )
            runtime = load_verified_runtime_contract(
                args.campaign_contract,
                expected_sha256=digest,
                expected_invariant_profile_sha256=profile_sha,
                motion_split_contract=profile["motion_split"],
            )
        lineage = derive_data_lineage(
            runtime,
            motion_split_contract=profile["motion_split"],
            topology=profile["topology"],
        )
        result = {
            "status": "PASS",
            "campaign_contract_sha256": digest,
            "invariant_profile_sha256": profile_sha,
            "data_lineage": lineage,
            "data_lineage_sha256": data_lineage_sha256(lineage),
        }
    except (OSError, ProductionDataLineageError) as exc:
        print(f"production runtime contract failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "COHORT_ASSIGNMENT_SCHEMA",
    "COLLECTION_ASSIGNMENT_POLICY",
    "DATA_LINEAGE_SCHEMA",
    "ProductionDataLineageError",
    "RUNTIME_CONTRACT_SCHEMA",
    "build_cohort_assignment",
    "data_lineage_sha256",
    "derive_data_lineage",
    "load_verified_runtime_contract",
    "rank_assignment",
    "validate_cohort_assignment",
    "validate_data_lineage",
]


if __name__ == "__main__":
    raise SystemExit(main())
