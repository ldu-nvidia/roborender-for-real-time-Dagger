#!/usr/bin/env python3
"""Build, capture, finalize, and verify the 256-case held-out evaluation plan.

The nested GR00T runtime owns the motion split.  This tool therefore never
repartitions the 295 hydrated NPZ files.  It verifies the checked-in nested
split manifest and exact file bytes, preserves the runtime's lexicographic
evaluation order, and maps its 256 evaluation motions to eight disjoint
contiguous 32-environment blocks.  The remaining 39 motions are collection
only.

Planning and runtime scenario identity are deliberately separate.  A planned
case ID binds seed, block offset, motion path/hash, and capture semantics.  A
production ``scenario_id`` cannot exist until Isaac has captured the actual
first condition.  The ``finalize`` command consumes complete per-case capture
evidence and derives scenario IDs with the evaluator's exact formula::

    sha256(canonical_json({seed, env_index, first_condition_sha256}))

No synthetic condition or scenario hash is ever emitted as a production
scenario contract.

This tool proves assignment and byte-level consistency only.  Raw capture
evidence is not, by itself, an attestation that Isaac/Image3F produced it.
Consequently finalized output remains ``status=capture_required`` and is not
release eligible until the separate live capture-attestation/repeatability
gate binds the runtime identities and per-seed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

PLAN_SCHEMA = "fast_roborender.round_dagger.heldout_motion_capture_plan.v2"
CAPTURE_EVIDENCE_SCHEMA = "fast_roborender.round_dagger.motion_capture_evidence.v1"
FINALIZATION_SCHEMA = "fast_roborender.round_dagger.motion_capture_finalization.v1"
CAPTURE_CONTRACT_SCHEMA = "fast_roborender.round_dagger.scenario_capture_contract.v1"
NESTED_SPLIT_SCHEMA = "roborender.motion_split.v1"
EXPECTED_MOTION_COUNT = 295
HELDOUT_MOTION_COUNT = 256
COLLECTION_TRAINING_MOTION_COUNT = 39
EVALUATION_MOTIONS_PER_SEED = 32
FIXED_SEEDS = (17, 29, 43, 59, 71, 89, 101, 127)
POLICY_VIEWS = ("head_color", "hand_left", "hand_right")
EXPECTED_SPLIT_MANIFEST_NAME = "33demos_675_775_heldout_256_v1.json"
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834"
)

INVENTORY_ALGORITHM = "len64be(path_utf8)+path_utf8+size64be+sha256_binary"
NESTED_INVENTORY_ALGORITHM = "sha256(canonical_json([{path,sha256},...]))"
PLANNED_CASE_DOMAIN = "fast_roborender.round_dagger.planned_motion_case.v1"
SCENARIO_ID_ALGORITHM = (
    "sha256(canonical_json({seed,env_index,first_condition_sha256}))"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class HeldoutPlanError(RuntimeError):
    """The source corpus, split, capture, or finalized plan is invalid."""


@dataclass(frozen=True)
class MotionRecord:
    relative_path: str
    size: int
    sha256: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    """Match round_dagger.artifacts.canonical_json_sha256 without import-path coupling."""
    return _sha256_bytes(_canonical_json(value))


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise HeldoutPlanError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HeldoutPlanError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise HeldoutPlanError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except HeldoutPlanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutPlanError(f"cannot decode {label}: {exc}") from None


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HeldoutPlanError(f"cannot safely open {label} {path}: {exc}") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HeldoutPlanError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or len(payload) != before.st_size:
        raise HeldoutPlanError(f"{label} changed while it was read")
    return payload


def _safe_motion_record(root: Path, path: Path) -> MotionRecord:
    relative = path.relative_to(root).as_posix()
    if not relative or any(character in relative for character in "\x00\t\n\r"):
        raise HeldoutPlanError(f"unsafe motion relative path: {relative!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HeldoutPlanError(f"cannot safely open motion {relative}: {exc}") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HeldoutPlanError(f"motion is not a regular file: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _stream_sha256(stream)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after):
        raise HeldoutPlanError(f"motion mutated while hashing: {relative}")
    return MotionRecord(relative, before.st_size, digest)


def _walk(root: Path) -> tuple[list[Path], tuple[tuple[str, int, int, int, int], ...]]:
    motions: list[Path] = []
    tree: list[tuple[str, int, int, int, int]] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    except OSError as exc:
        raise HeldoutPlanError(f"cannot enumerate motion corpus: {exc}") from None
    for path in entries:
        relative = path.relative_to(root).as_posix()
        try:
            info = path.lstat()
        except OSError as exc:
            raise HeldoutPlanError(f"cannot inspect motion entry {relative}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise HeldoutPlanError(f"symbolic links are forbidden in motion corpus: {relative}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise HeldoutPlanError(f"special files are forbidden in motion corpus: {relative}")
        tree.append((relative, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if stat.S_ISREG(info.st_mode):
            if path.suffix != ".npz":
                raise HeldoutPlanError(f"unexpected non-NPZ file in motion corpus: {relative}")
            motions.append(path)
    return motions, tuple(tree)


def scan_motion_corpus(root: Path) -> list[MotionRecord]:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise HeldoutPlanError(f"cannot inspect motion-corpus root {root}: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise HeldoutPlanError("motion-corpus root must be a real directory, not a symlink")
    root = root.resolve(strict=True)
    motions, tree_before = _walk(root)
    if len(motions) != EXPECTED_MOTION_COUNT:
        raise HeldoutPlanError(
            f"motion corpus contains {len(motions)} NPZ files; expected exactly "
            f"{EXPECTED_MOTION_COUNT}"
        )
    records = [_safe_motion_record(root, path) for path in motions]
    motions_after, tree_after = _walk(root)
    if tree_after != tree_before or [path.relative_to(root) for path in motions_after] != [
        path.relative_to(root) for path in motions
    ]:
        raise HeldoutPlanError("motion corpus mutated while inventory was calculated")
    if len({record.sha256 for record in records}) != len(records):
        raise HeldoutPlanError(
            "motion corpus contains duplicate NPZ content; split leakage cannot be excluded"
        )
    return records


def inventory_sha256(records: Sequence[MotionRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        relative = record.relative_path.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(record.size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(record.sha256))
    return digest.hexdigest()


def _nested_inventory_sha256(entries: Sequence[Mapping[str, str]]) -> str:
    canonical = [
        {"path": entry["path"], "sha256": entry["sha256"]}
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return _sha256_bytes(_canonical_json(canonical))


def _split_manifest_path(motion_root: Path, supplied: Path | None) -> Path:
    return supplied if supplied is not None else motion_root.parent / EXPECTED_SPLIT_MANIFEST_NAME


def _load_split(
    records: Sequence[MotionRecord],
    *,
    split_manifest: Path,
    expected_split_manifest_sha256: str,
) -> tuple[dict[str, Any], list[MotionRecord], list[MotionRecord], str]:
    expected_sha = _sha(expected_split_manifest_sha256, "expected split-manifest SHA-256")
    payload = _read_regular_bytes(split_manifest, "nested motion-split manifest")
    observed_sha = _sha256_bytes(payload)
    if observed_sha != expected_sha:
        raise HeldoutPlanError(
            f"nested motion-split manifest SHA-256 mismatch: {observed_sha} != {expected_sha}"
        )
    document = _decode_json(payload, "nested motion-split manifest")
    if not isinstance(document, Mapping) or set(document) != {
        "schema",
        "dataset",
        "counts",
        "inventory_sha256",
        "selection",
        "motions",
    }:
        raise HeldoutPlanError("nested motion-split manifest fields are not exact")
    if document["schema"] != NESTED_SPLIT_SCHEMA:
        raise HeldoutPlanError("nested motion-split manifest schema mismatch")
    if document["dataset"] != "g1_wsdpt/33demos_675_775":
        raise HeldoutPlanError("nested motion-split dataset identity mismatch")
    if document["counts"] != {
        "total": EXPECTED_MOTION_COUNT,
        "collection": COLLECTION_TRAINING_MOTION_COUNT,
        "eval": HELDOUT_MOTION_COUNT,
    }:
        raise HeldoutPlanError("nested motion-split counts are not production counts")
    if document["selection"] != {
        "method": "sha256_rank",
        "seed": "roborender-heldout-256-v1",
        "held_out_role": "eval",
    }:
        raise HeldoutPlanError("nested motion-split selection contract mismatch")
    raw_motions = document["motions"]
    if not isinstance(raw_motions, list) or len(raw_motions) != EXPECTED_MOTION_COUNT:
        raise HeldoutPlanError("nested motion-split motions are incomplete")
    entries: list[dict[str, str]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_motions):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "split"}:
            raise HeldoutPlanError(f"nested split motion {index} fields are not exact")
        name = raw["path"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).suffix != ".npz"
            or name in names
        ):
            raise HeldoutPlanError(f"nested split motion {index} has unsafe/duplicate path")
        names.add(name)
        role = raw["split"]
        if role not in {"eval", "collection"}:
            raise HeldoutPlanError(f"nested split motion {name} has invalid role")
        entries.append({"path": name, "sha256": _sha(raw["sha256"], f"motion {name}"), "split": role})
    entries.sort(key=lambda item: item["path"])
    record_by_path = {record.relative_path: record for record in records}
    if list(record_by_path) != sorted(record_by_path):
        raise HeldoutPlanError("internal motion scan order is not canonical")
    if [entry["path"] for entry in entries] != sorted(record_by_path):
        raise HeldoutPlanError("nested split does not exactly enumerate hydrated motions")
    for entry in entries:
        if record_by_path[entry["path"]].sha256 != entry["sha256"]:
            raise HeldoutPlanError(f"nested split content mismatch for {entry['path']}")
    nested_inventory = _nested_inventory_sha256(entries)
    if nested_inventory != _sha(document["inventory_sha256"], "nested inventory SHA-256"):
        raise HeldoutPlanError("nested split inventory SHA-256 mismatch")
    evaluation = [record_by_path[item["path"]] for item in entries if item["split"] == "eval"]
    collection = [
        record_by_path[item["path"]] for item in entries if item["split"] == "collection"
    ]
    if len(evaluation) != HELDOUT_MOTION_COUNT or len(collection) != COLLECTION_TRAINING_MOTION_COUNT:
        raise HeldoutPlanError("nested split role counts changed after resolution")
    if {item.sha256 for item in evaluation} & {item.sha256 for item in collection}:
        raise HeldoutPlanError("nested eval and collection splits overlap by content")
    return dict(document), evaluation, collection, observed_sha


def _motion_document(record: MotionRecord, *, index_name: str, index: int) -> dict[str, Any]:
    return {index_name: index, **asdict(record)}


def _planned_case_id(
    *, seed: int, env_index: int, global_case_index: int, motion: MotionRecord, contract_sha: str
) -> str:
    return _canonical_json_sha256(
        {
            "domain": PLANNED_CASE_DOMAIN,
            "seed": seed,
            "env_index": env_index,
            "global_case_index": global_case_index,
            "motion_relative_path": motion.relative_path,
            "motion_sha256": motion.sha256,
            "capture_contract_sha256": contract_sha,
        }
    )


def build_plan(
    motion_root: Path,
    *,
    split_manifest: Path | None = None,
    expected_split_manifest_sha256: str = EXPECTED_SPLIT_MANIFEST_SHA256,
) -> dict[str, Any]:
    records = scan_motion_corpus(motion_root)
    split_path = _split_manifest_path(motion_root, split_manifest)
    split_document, evaluation, collection, split_sha = _load_split(
        records,
        split_manifest=split_path,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    capture_contract = {
        "schema": CAPTURE_CONTRACT_SCHEMA,
        "motion_source": "checked_in_nested_motion_split_eval_role",
        "motion_split_manifest_sha256": split_sha,
        "motion_split_inventory_sha256": split_document["inventory_sha256"],
        "motion_order": "lexicographic_path_then_eval_filter",
        "initialization": "deterministic_motion_replay",
        "episodes_per_case": 1,
        "policy_views": list(POLICY_VIEWS),
        "view_layout": "vertical",
        "seed_roles": ["simulator", "generation"],
        "assignment": "one_disjoint_contiguous_32_motion_block_per_fixed_seed",
        "runtime_scenario_id_deferred_until_actual_first_condition_capture": True,
    }
    contract_sha = _canonical_json_sha256(capture_contract)
    cases: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(FIXED_SEEDS):
        offset = seed_index * EVALUATION_MOTIONS_PER_SEED
        block_motions = evaluation[offset : offset + EVALUATION_MOTIONS_PER_SEED]
        if len(block_motions) != EVALUATION_MOTIONS_PER_SEED:
            raise HeldoutPlanError("nested eval block is incomplete")
        block_cases: list[dict[str, Any]] = []
        for env_index, motion in enumerate(block_motions):
            global_index = offset + env_index
            case = {
                "seed": seed,
                "env_index": env_index,
                "global_case_index": global_index,
                "planned_case_id": _planned_case_id(
                    seed=seed,
                    env_index=env_index,
                    global_case_index=global_index,
                    motion=motion,
                    contract_sha=contract_sha,
                ),
                "motion_relative_path": motion.relative_path,
                "motion_sha256": motion.sha256,
            }
            cases.append(case)
            block_cases.append(case)
        blocks.append(
            {
                "seed": seed,
                "offset": offset,
                "limit": EVALUATION_MOTIONS_PER_SEED,
                "cases": block_cases,
            }
        )
    if len(cases) != HELDOUT_MOTION_COUNT:
        raise HeldoutPlanError("capture plan does not cover all 256 nested eval motions")
    if len({case["motion_sha256"] for case in cases}) != HELDOUT_MOTION_COUNT:
        raise HeldoutPlanError("capture plan repeats an evaluation motion")
    return {
        "schema": PLAN_SCHEMA,
        "status": "capture_required",
        "release_eligible": False,
        "remaining_gate": "live_capture_attestation_and_repeatability",
        "motion_path_base": "relative_to_verified_motion_root",
        "source": {
            "motion_count": len(records),
            "total_file_bytes": sum(record.size for record in records),
            "inventory_sha256": inventory_sha256(records),
            "inventory_algorithm": INVENTORY_ALGORITHM,
            "nested_split_manifest_name": split_path.name,
            "nested_split_manifest_sha256": split_sha,
            "nested_split_inventory_sha256": split_document["inventory_sha256"],
            "nested_split_inventory_algorithm": NESTED_INVENTORY_ALGORITHM,
        },
        "partition": {
            "eval_count": len(evaluation),
            "collection_training_count": len(collection),
            "path_overlap_count": 0,
            "content_overlap_count": 0,
            "eval_runtime_order": [
                _motion_document(record, index_name="runtime_eval_index", index=index)
                for index, record in enumerate(evaluation)
            ],
            "collection_runtime_order": [
                _motion_document(record, index_name="runtime_collection_index", index=index)
                for index, record in enumerate(collection)
            ],
        },
        "capture": {
            "contract": capture_contract,
            "contract_sha256": contract_sha,
            "fixed_seeds": list(FIXED_SEEDS),
            "seed_count": len(FIXED_SEEDS),
            "motions_per_seed": EVALUATION_MOTIONS_PER_SEED,
            "unique_motion_count": HELDOUT_MOTION_COUNT,
            "case_count": len(cases),
            "seed_blocks": blocks,
            "cases": cases,
        },
    }


def _load_json_file(path: Path, label: str) -> tuple[Any, bytes]:
    payload = _read_regular_bytes(path, label)
    return _decode_json(payload, label), payload


def verify_plan(
    motion_root: Path,
    *,
    manifest_path: Path,
    split_manifest: Path | None = None,
    expected_split_manifest_sha256: str = EXPECTED_SPLIT_MANIFEST_SHA256,
) -> dict[str, Any]:
    expected = build_plan(
        motion_root,
        split_manifest=split_manifest,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    observed, payload = _load_json_file(manifest_path, "capture plan")
    if observed != expected:
        raise HeldoutPlanError("capture plan does not match the nested split and motion corpus")
    return {
        "status": "verified_capture_required",
        "manifest": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha256_bytes(payload),
        "source_inventory_sha256": expected["source"]["inventory_sha256"],
        "nested_split_manifest_sha256": expected["source"]["nested_split_manifest_sha256"],
        "heldout_motion_count": expected["partition"]["eval_count"],
        "collection_training_motion_count": expected["partition"][
            "collection_training_count"
        ],
        "planned_capture_count": expected["capture"]["case_count"],
    }


_PLANNED_CASE_KEYS = {
    "seed",
    "env_index",
    "global_case_index",
    "planned_case_id",
    "motion_relative_path",
    "motion_sha256",
}
_EVIDENCE_CASE_KEYS = _PLANNED_CASE_KEYS | {"first_condition_sha256"}


def finalize_capture(
    motion_root: Path,
    *,
    manifest_path: Path,
    evidence_path: Path,
    split_manifest: Path | None = None,
    expected_split_manifest_sha256: str = EXPECTED_SPLIT_MANIFEST_SHA256,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verify_plan(
        motion_root,
        manifest_path=manifest_path,
        split_manifest=split_manifest,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    plan, plan_payload = _load_json_file(manifest_path, "capture plan")
    evidence, evidence_payload = _load_json_file(evidence_path, "capture evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {"schema", "plan_sha256", "cases"}:
        raise HeldoutPlanError("capture evidence fields are not exact")
    if evidence["schema"] != CAPTURE_EVIDENCE_SCHEMA:
        raise HeldoutPlanError("capture evidence schema mismatch")
    plan_sha = _sha256_bytes(plan_payload)
    if _sha(evidence["plan_sha256"], "evidence plan SHA-256") != plan_sha:
        raise HeldoutPlanError("capture evidence is bound to a different plan")
    raw_cases = evidence["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != HELDOUT_MOTION_COUNT:
        raise HeldoutPlanError("capture evidence must contain exactly 256 cases")
    evidence_by_identity: dict[tuple[int, int], dict[str, Any]] = {}
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping) or set(raw) != _EVIDENCE_CASE_KEYS:
            raise HeldoutPlanError(f"capture evidence case {index} fields are not exact")
        identity = (raw["seed"], raw["env_index"])
        if (
            type(identity[0]) is not int
            or type(identity[1]) is not int
            or identity in evidence_by_identity
        ):
            raise HeldoutPlanError("capture evidence has malformed or duplicate seed/env identity")
        normalized = dict(raw)
        normalized["first_condition_sha256"] = _sha(
            raw["first_condition_sha256"], "first-condition SHA-256"
        )
        evidence_by_identity[identity] = normalized
    bindings: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for planned in plan["capture"]["cases"]:
        identity = (planned["seed"], planned["env_index"])
        actual = evidence_by_identity.pop(identity, None)
        if actual is None:
            raise HeldoutPlanError(f"capture evidence is missing planned case {identity}")
        if {key: actual[key] for key in _PLANNED_CASE_KEYS} != planned:
            raise HeldoutPlanError(f"capture evidence assignment mismatch for case {identity}")
        scenario_payload = {
            "seed": planned["seed"],
            "env_index": planned["env_index"],
            "first_condition_sha256": actual["first_condition_sha256"],
        }
        scenario_id = _canonical_json_sha256(scenario_payload)
        scenarios.append(
            {
                "seed": planned["seed"],
                "env_index": planned["env_index"],
                "scenario_id": scenario_id,
            }
        )
        bindings.append(
            {
                **planned,
                "first_condition_sha256": actual["first_condition_sha256"],
                "scenario_id": scenario_id,
            }
        )
    if evidence_by_identity:
        raise HeldoutPlanError("capture evidence contains unplanned seed/env identities")
    scenario_payload = _canonical_json(scenarios) + b"\n"
    finalization = {
        "schema": FINALIZATION_SCHEMA,
        "status": "capture_required",
        "release_eligible": False,
        "evidence_trust": "raw_capture_evidence_not_runtime_attestation",
        "remaining_gate": "live_capture_attestation_and_repeatability",
        "plan_sha256": plan_sha,
        "nested_split_manifest_sha256": plan["source"]["nested_split_manifest_sha256"],
        "source_inventory_sha256": plan["source"]["inventory_sha256"],
        "capture_evidence_sha256": _sha256_bytes(evidence_payload),
        "capture_evidence_case_inventory_sha256": _canonical_json_sha256(bindings),
        "scenario_contract_sha256": _sha256_bytes(scenario_payload),
        "scenario_id_algorithm": SCENARIO_ID_ALGORITHM,
        "fixed_seeds": list(FIXED_SEEDS),
        "envs_per_seed": EVALUATION_MOTIONS_PER_SEED,
        "case_count": len(bindings),
        "bindings": bindings,
    }
    return finalization, scenarios


def verify_finalized(
    motion_root: Path,
    *,
    manifest_path: Path,
    evidence_path: Path,
    finalization_path: Path,
    scenario_path: Path,
    split_manifest: Path | None = None,
    expected_split_manifest_sha256: str = EXPECTED_SPLIT_MANIFEST_SHA256,
) -> dict[str, Any]:
    expected_finalization, expected_scenarios = finalize_capture(
        motion_root,
        manifest_path=manifest_path,
        evidence_path=evidence_path,
        split_manifest=split_manifest,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    observed_finalization, finalization_payload = _load_json_file(
        finalization_path, "capture finalization"
    )
    observed_scenarios, scenario_payload = _load_json_file(scenario_path, "scenario contract")
    if observed_finalization != expected_finalization:
        raise HeldoutPlanError("capture finalization does not match its plan/evidence")
    if observed_scenarios != expected_scenarios:
        raise HeldoutPlanError("scenario contract does not match actual capture evidence")
    if _sha256_bytes(scenario_payload) != expected_finalization["scenario_contract_sha256"]:
        raise HeldoutPlanError("scenario contract byte hash differs from finalization")
    return {
        "status": "verified_raw_capture_evidence_not_release_attested",
        "finalization": str(finalization_path.resolve(strict=True)),
        "finalization_sha256": _sha256_bytes(finalization_payload),
        "scenario_contract": str(scenario_path.resolve(strict=True)),
        "scenario_contract_sha256": _sha256_bytes(scenario_payload),
        "capture_evidence_sha256": expected_finalization["capture_evidence_sha256"],
        "case_count": expected_finalization["case_count"],
    }


def _publish_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise HeldoutPlanError(f"output path is not a regular file: {path}")
        if _read_regular_bytes(path, "existing output") == payload:
            return
        raise HeldoutPlanError(f"refusing to replace different existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError as exc:
        raise HeldoutPlanError(f"cannot publish output {path}: {exc}") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--expected-split-manifest-sha256", default=EXPECTED_SPLIT_MANIFEST_SHA256
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="publish the immutable assignment/capture plan")
    _source_arguments(build)
    build.add_argument("--manifest-output", type=Path, required=True)
    verify = commands.add_parser("verify-plan", help="rehash and verify a capture plan")
    _source_arguments(verify)
    verify.add_argument("--manifest", type=Path, required=True)
    finalize = commands.add_parser(
        "finalize", help="derive runtime scenario IDs from actual first-condition captures"
    )
    _source_arguments(finalize)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--evidence", type=Path, required=True)
    finalize.add_argument("--finalization-output", type=Path, required=True)
    finalize.add_argument("--scenario-output", type=Path, required=True)
    verify_final = commands.add_parser(
        "verify-finalized", help="verify plan, evidence, finalization, and scenarios"
    )
    _source_arguments(verify_final)
    verify_final.add_argument("--manifest", type=Path, required=True)
    verify_final.add_argument("--evidence", type=Path, required=True)
    verify_final.add_argument("--finalization", type=Path, required=True)
    verify_final.add_argument("--scenarios", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = {
        "motion_root": args.motion_root,
        "split_manifest": args.split_manifest,
        "expected_split_manifest_sha256": args.expected_split_manifest_sha256,
    }
    try:
        if args.command == "build":
            plan = build_plan(**source)
            _publish_idempotent(args.manifest_output, _canonical_json(plan) + b"\n")
            result = verify_plan(**source, manifest_path=args.manifest_output)
        elif args.command == "verify-plan":
            result = verify_plan(**source, manifest_path=args.manifest)
        elif args.command == "finalize":
            if args.finalization_output.absolute() == args.scenario_output.absolute():
                raise HeldoutPlanError("finalization and scenario outputs must differ")
            finalization, scenarios = finalize_capture(
                **source, manifest_path=args.manifest, evidence_path=args.evidence
            )
            _publish_idempotent(
                args.finalization_output, _canonical_json(finalization) + b"\n"
            )
            _publish_idempotent(args.scenario_output, _canonical_json(scenarios) + b"\n")
            result = verify_finalized(
                **source,
                manifest_path=args.manifest,
                evidence_path=args.evidence,
                finalization_path=args.finalization_output,
                scenario_path=args.scenario_output,
            )
        else:
            result = verify_finalized(
                **source,
                manifest_path=args.manifest,
                evidence_path=args.evidence,
                finalization_path=args.finalization,
                scenario_path=args.scenarios,
            )
    except HeldoutPlanError as exc:
        print(f"held-out evaluation plan failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
