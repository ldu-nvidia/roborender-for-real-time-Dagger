#!/usr/bin/env python3
"""Transactional cumulative replay indexing for round-based DAgger.

The actor-side artifact module owns byte-level validation and immutable cohort
publication.  This module consumes only cohorts that pass ``verify_cohort`` and
records them in a small SQLite ledger.  A replay snapshot is an immutable,
deterministically ordered description of every accepted *training* shard from
round zero through a completely sealed macro-round.

No sample bytes are copied into SQLite.  The ledger stores content identities
and counts; the learner opens the manifest/payload paths after independently
verifying the snapshot and cohort artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from round_dagger import artifacts as _artifacts
from round_dagger.seed_policy import ActorSeedError, validate_actor_seed_contract


LEDGER_SCHEMA = "fast_roborender.round_dagger.replay_ledger.v1"
SNAPSHOT_SCHEMA = "fast_roborender.round_dagger.replay_snapshot.v1"
COHORT_SCHEMA = "fast_roborender.round_dagger.cohort.v1"

_ACTOR_SEED_FIELD = "actor_seed_contract"
_TRAJECTORY_FINGERPRINT_FIELD = (
    "physical_teacher_trajectory_sha256_by_global_env"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class ReplayError(RuntimeError):
    """A cohort or replay operation violated the cumulative replay contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    helper = getattr(_artifacts, "canonical_json_sha256", None)
    if helper is not None:
        return str(helper(value))
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    helper = getattr(_artifacts, "sha256_file", None)
    if helper is not None:
        return str(helper(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReplayError(f"{label} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReplayError(f"{label} is unsafe")
    return relative


def _safe_artifact_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    """Resolve a regular artifact while rejecting symlinks in every component."""

    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ReplayError(f"{label} root is unsafe")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ReplayError(f"{label} must not contain symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ReplayError(f"{label} escapes or is missing from its cohort") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ReplayError(f"{label} is not a regular file")
    return resolved


def _publish_immutable_json(path: Path, document: Mapping[str, Any]) -> str:
    """Call the artifact layer's create-only JSON publisher.

    Keeping this adapter in one place makes the replay module insensitive to
    whether the artifact helper calls the operation ``publish`` or ``write``.
    Both accepted spellings retain the same required create-only semantics.
    """

    helper = getattr(_artifacts, "publish_immutable_json", None)
    if helper is None:
        helper = getattr(_artifacts, "atomic_publish_json", None)
    if helper is None:
        raise ReplayError("artifact API lacks an immutable JSON publisher")
    result = helper(path, document)
    if isinstance(result, str) and _SHA256_RE.fullmatch(result):
        return result
    return _sha256_file(path)


def _true_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayError(f"{label} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ReplayError(f"{label} is malformed")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReplayError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ReplayError(f"{label} must be a non-empty mapping")
    # Round-trip through canonical JSON to reject values SQLite could not bind
    # deterministically (NaN is rejected by allow_nan=False).
    try:
        return json.loads(_canonical_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{label} is not canonical JSON") from exc


def _verified_document(path: Path) -> dict[str, Any]:
    try:
        verified = _artifacts.verify_cohort(path)
    except Exception as exc:
        if isinstance(exc, ReplayError):
            raise
        raise ReplayError(f"cohort verification failed: {exc}") from exc
    if not isinstance(verified, Mapping):
        raise ReplayError("verify_cohort must return the verified manifest mapping")
    return dict(verified)


def _cohort_record(path: Path) -> dict[str, Any]:
    document = _verified_document(path)
    schema = document.get("schema")
    if schema != COHORT_SCHEMA:
        raise ReplayError(f"unsupported cohort schema: {schema!r}")
    campaign_id = _identifier(document.get("campaign_id"), "campaign_id")
    macro_round = _true_int(document.get("macro_round"), "macro_round")
    cohort_index = _true_int(document.get("cohort_index"), "cohort_index")
    purpose = document.get("purpose")
    if purpose not in {"train", "eval"}:
        raise ReplayError("cohort purpose must be exactly 'train' or 'eval'")
    if document.get("sealed") is not True:
        raise ReplayError("replay accepts only sealed cohort manifests")
    transition_count = _true_int(
        document.get("transition_count"), "transition_count", minimum=1
    )
    view_contract = _exact_mapping(document.get("view_contract"), "view_contract")
    action_contract = _exact_mapping(
        document.get("action_contract"), "action_contract"
    )
    has_seed = _ACTOR_SEED_FIELD in document
    has_fingerprints = _TRAJECTORY_FINGERPRINT_FIELD in document
    if has_seed != has_fingerprints:
        raise ReplayError(
            "cohort seed and physical/teacher trajectory provenance is incomplete"
        )
    actor_seed_contract: dict[str, Any] | None = None
    trajectory_fingerprints: dict[str, str] | None = None
    if has_seed:
        actor_seed_contract = _exact_mapping(
            document[_ACTOR_SEED_FIELD], _ACTOR_SEED_FIELD
        )
        raw_fingerprints = document[_TRAJECTORY_FINGERPRINT_FIELD]
        if not isinstance(raw_fingerprints, Mapping) or not raw_fingerprints:
            raise ReplayError(
                "physical/teacher trajectory fingerprints must be a non-empty mapping"
            )
        trajectory_fingerprints = {}
        for global_env_id, digest in raw_fingerprints.items():
            if not isinstance(global_env_id, str) or not global_env_id.isdigit():
                raise ReplayError("trajectory fingerprint environment ID is malformed")
            trajectory_fingerprints[global_env_id] = _sha256(
                digest, "physical/teacher trajectory SHA-256"
            )
        assignments = actor_seed_contract.get("assignments")
        if not isinstance(assignments, list):
            raise ReplayError("actor seed contract assignments are malformed")
        try:
            ranks = sorted(int(item["rank"]) for item in assignments)
            environments = sorted(
                int(global_env_id)
                for item in assignments
                for global_env_id in item["global_env_ids"]
            )
            validate_actor_seed_contract(
                actor_seed_contract,
                campaign_id=campaign_id,
                macro_round=macro_round,
                cohort_index=cohort_index,
                expected_ranks=ranks,
                expected_global_env_ids=environments,
            )
        except (ActorSeedError, KeyError, TypeError, ValueError) as exc:
            raise ReplayError(f"invalid actor seed provenance: {exc}") from exc
        if set(trajectory_fingerprints) != {str(item) for item in environments}:
            raise ReplayError(
                "physical/teacher trajectory fingerprints do not cover seeded environments"
            )
    raw_shards = document.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ReplayError("sealed cohort must contain at least one shard")

    supplied_path = Path(os.path.abspath(path.expanduser()))
    if supplied_path.is_symlink():
        raise ReplayError("cohort input must not be a symbolic link")
    if supplied_path.is_dir():
        cohort_root = supplied_path.resolve(strict=True)
        declared_cohort_manifest = document.get(
            "cohort_manifest_path",
            document.get("manifest_path", document.get("seal_path", "seal.json")),
        )
        if isinstance(declared_cohort_manifest, str) and Path(
            declared_cohort_manifest
        ).is_absolute():
            try:
                declared_resolved = Path(declared_cohort_manifest).resolve(strict=True)
                cohort_manifest_relative = PurePosixPath(
                    declared_resolved.relative_to(cohort_root).as_posix()
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise ReplayError("cohort_manifest_path escapes its cohort") from exc
        else:
            cohort_manifest_relative = _safe_relative_path(
                declared_cohort_manifest, "cohort_manifest_path"
            )
        cohort_manifest_path = _safe_artifact_file(
            cohort_root, cohort_manifest_relative, "cohort manifest"
        )
    elif supplied_path.is_file():
        cohort_manifest_path = supplied_path.resolve(strict=True)
        cohort_root = cohort_manifest_path.parent
    else:
        raise ReplayError("cohort input must be a manifest file or cohort directory")
    cohort_manifest_sha256 = _sha256_file(cohort_manifest_path)
    declared_cohort_sha = document.get(
        "cohort_manifest_sha256",
        document.get("cohort_sha256", document.get("seal_sha256")),
    )
    if declared_cohort_sha is not None and _sha256(
        declared_cohort_sha, "cohort_manifest_sha256"
    ) != cohort_manifest_sha256:
        raise ReplayError("cohort manifest digest changed after verification")

    shards: list[dict[str, Any]] = []
    shard_ids: set[str] = set()
    manifest_digests: set[str] = set()
    shard_transition_total = 0
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise ReplayError(f"shards[{index}] must be a mapping")
        shard_id = _identifier(raw.get("shard_id"), f"shards[{index}].shard_id")
        shard_manifest_relative = _safe_relative_path(
            raw.get(
                "shard_manifest_path",
                raw.get("sidecar_path", raw.get("manifest_path")),
            ),
            f"shards[{index}].shard_manifest_path",
        )
        payload_relative = _safe_relative_path(
            raw.get("payload_path"), f"shards[{index}].payload_path"
        )
        shard_manifest_path = _safe_artifact_file(
            cohort_root,
            shard_manifest_relative,
            f"shards[{index}] manifest",
        )
        payload_path = _safe_artifact_file(
            cohort_root, payload_relative, f"shards[{index}] payload"
        )
        manifest_sha256 = _sha256(
            raw.get(
                "manifest_sha256",
                raw.get("shard_manifest_sha256", raw.get("sidecar_sha256")),
            ),
            f"shards[{index}].manifest_sha256",
        )
        payload_sha256 = _sha256(
            raw.get("payload_sha256"), f"shards[{index}].payload_sha256"
        )
        count = _true_int(
            raw.get("transition_count"),
            f"shards[{index}].transition_count",
            minimum=1,
        )
        if _sha256_file(shard_manifest_path) != manifest_sha256:
            raise ReplayError(f"shards[{index}] manifest digest changed after verification")
        if _sha256_file(payload_path) != payload_sha256:
            raise ReplayError(f"shards[{index}] payload digest changed after verification")
        if shard_id in shard_ids:
            raise ReplayError(f"duplicate shard ID in cohort: {shard_id}")
        if manifest_sha256 in manifest_digests:
            raise ReplayError("duplicate shard manifest digest in cohort")
        shard_ids.add(shard_id)
        manifest_digests.add(manifest_sha256)
        shard_transition_total += count
        shards.append(
            {
                "shard_id": shard_id,
                "shard_manifest_path": shard_manifest_relative.as_posix(),
                "manifest_sha256": manifest_sha256,
                "payload_path": payload_relative.as_posix(),
                "payload_sha256": payload_sha256,
                "transition_count": count,
            }
        )
    if shard_transition_total != transition_count:
        raise ReplayError(
            "cohort transition_count does not equal the sum of shard counts"
        )
    shards.sort(key=lambda item: item["shard_id"])

    logical_id = f"{campaign_id}:r{macro_round:09d}:c{cohort_index:04d}:{purpose}"
    declared_logical_id = document.get("logical_id")
    if declared_logical_id is not None and declared_logical_id != logical_id:
        raise ReplayError("cohort logical_id does not match its identity fields")
    record = {
        "logical_id": logical_id,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "purpose": purpose,
        "cohort_manifest_path": str(cohort_manifest_path),
        "manifest_sha256": cohort_manifest_sha256,
        "manifest_schema": schema,
        "transition_count": transition_count,
        "view_contract": view_contract,
        "action_contract": action_contract,
        "shards": shards,
    }
    if actor_seed_contract is not None and trajectory_fingerprints is not None:
        record[_ACTOR_SEED_FIELD] = actor_seed_contract
        record[_TRAJECTORY_FINGERPRINT_FIELD] = trajectory_fingerprints
    return record


def _stored_seed_provenance(path: str, expected_sha256: str) -> tuple[Any, Any]:
    manifest = Path(path)
    if manifest.is_symlink() or not manifest.is_file():
        raise ReplayError("accepted cohort manifest is missing or unsafe")
    if _sha256_file(manifest) != expected_sha256:
        raise ReplayError("accepted cohort manifest changed after ingestion")
    document = _verified_document(manifest)
    return document.get(_ACTOR_SEED_FIELD), document.get(
        _TRAJECTORY_FINGERPRINT_FIELD
    )


def _reject_repeated_round_provenance(
    connection: sqlite3.Connection, record: Mapping[str, Any]
) -> None:
    """Reject repeated RNG streams or physical trajectories in one macro-round."""

    current_seed = record.get(_ACTOR_SEED_FIELD)
    current_fingerprints = record.get(_TRAJECTORY_FINGERPRINT_FIELD)
    rows = connection.execute(
        "SELECT cohort_index, cohort_manifest_path, manifest_sha256 FROM cohorts "
        "WHERE campaign_id = ? AND macro_round = ? AND purpose = ?",
        (record["campaign_id"], record["macro_round"], record["purpose"]),
    ).fetchall()
    if not rows:
        return
    for row in rows:
        stored_seed, stored_fingerprints = _stored_seed_provenance(
            row["cohort_manifest_path"], row["manifest_sha256"]
        )
        if (current_seed is None) != (stored_seed is None):
            raise ReplayError(
                "macro-round mixes cohorts with and without actor seed provenance"
            )
        if current_seed is None:
            continue
        current_simulator = {
            int(item["simulator_seed"]) for item in current_seed["assignments"]
        }
        current_generation = {
            int(item["generation_seed"]) for item in current_seed["assignments"]
        }
        stored_simulator = {
            int(item["simulator_seed"]) for item in stored_seed["assignments"]
        }
        stored_generation = {
            int(item["generation_seed"]) for item in stored_seed["assignments"]
        }
        if current_simulator & stored_simulator:
            raise ReplayError("macro-round reuses an actor simulator seed")
        if current_generation & stored_generation:
            raise ReplayError("macro-round reuses an Image3F generation seed")
        duplicate_trajectories = set(current_fingerprints.values()) & set(
            stored_fingerprints.values()
        )
        if duplicate_trajectories:
            raise ReplayError(
                "macro-round contains repeated physical/teacher trajectories"
            )


class ReplayLedger:
    """SQLite-backed accepted-cohort index with transactional ingestion."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        if self.path.is_symlink():
            raise ReplayError("replay ledger must not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def __enter__(self) -> "ReplayLedger":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None  # type: ignore[assignment]

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS replay_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_contracts (
                    campaign_id TEXT PRIMARY KEY,
                    manifest_schema TEXT NOT NULL,
                    view_contract_json TEXT NOT NULL,
                    action_contract_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cohorts (
                    logical_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    macro_round INTEGER NOT NULL CHECK (macro_round >= 0),
                    cohort_index INTEGER NOT NULL CHECK (cohort_index >= 0),
                    purpose TEXT NOT NULL CHECK (purpose IN ('train', 'eval')),
                    cohort_manifest_path TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL UNIQUE,
                    manifest_schema TEXT NOT NULL,
                    transition_count INTEGER NOT NULL CHECK (transition_count > 0),
                    view_contract_json TEXT NOT NULL,
                    action_contract_json TEXT NOT NULL,
                    UNIQUE(campaign_id, macro_round, cohort_index, purpose),
                    FOREIGN KEY(campaign_id) REFERENCES campaign_contracts(campaign_id)
                );
                CREATE TABLE IF NOT EXISTS shards (
                    logical_id TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    shard_manifest_path TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL UNIQUE,
                    payload_path TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    transition_count INTEGER NOT NULL CHECK (transition_count > 0),
                    PRIMARY KEY(logical_id, shard_id),
                    FOREIGN KEY(logical_id) REFERENCES cohorts(logical_id)
                        ON DELETE RESTRICT
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM replay_metadata WHERE key = 'schema'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO replay_metadata(key, value) VALUES('schema', ?)",
                    (LEDGER_SCHEMA,),
                )
            elif row["value"] != LEDGER_SCHEMA:
                raise ReplayError("replay ledger schema is unsupported")

    def ingest_cohort(self, manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
        """Verify and transactionally accept one immutable cohort manifest."""

        record = _cohort_record(Path(manifest_path))
        view_json = _canonical_bytes(record["view_contract"]).decode("ascii")
        action_json = _canonical_bytes(record["action_contract"]).decode("ascii")
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT manifest_sha256, transition_count FROM cohorts "
                "WHERE logical_id = ?",
                (record["logical_id"],),
            ).fetchone()
            if existing is not None:
                if existing["manifest_sha256"] != record["manifest_sha256"]:
                    raise ReplayError(
                        "same cohort logical ID was presented with a different digest"
                    )
                stored_shards = connection.execute(
                    "SELECT shard_id, shard_manifest_path, manifest_sha256, "
                    "payload_path, payload_sha256, transition_count "
                    "FROM shards WHERE logical_id = ? ORDER BY shard_id",
                    (record["logical_id"],),
                ).fetchall()
                expected_shards = [
                    (
                        item["shard_id"],
                        item["shard_manifest_path"],
                        item["manifest_sha256"],
                        item["payload_path"],
                        item["payload_sha256"],
                        item["transition_count"],
                    )
                    for item in record["shards"]
                ]
                actual_shards = [tuple(row) for row in stored_shards]
                if actual_shards != expected_shards:
                    raise ReplayError("idempotent cohort retry disagrees with ledger shards")
                connection.commit()
                return {
                    "status": "ALREADY_ACCEPTED",
                    "logical_id": record["logical_id"],
                    "manifest_sha256": record["manifest_sha256"],
                    "transition_count": existing["transition_count"],
                }

            _reject_repeated_round_provenance(connection, record)

            contract = connection.execute(
                "SELECT manifest_schema, view_contract_json, action_contract_json "
                "FROM campaign_contracts WHERE campaign_id = ?",
                (record["campaign_id"],),
            ).fetchone()
            if contract is None:
                connection.execute(
                    "INSERT INTO campaign_contracts VALUES (?, ?, ?, ?)",
                    (
                        record["campaign_id"],
                        record["manifest_schema"],
                        view_json,
                        action_json,
                    ),
                )
            elif (
                contract["manifest_schema"] != record["manifest_schema"]
                or contract["view_contract_json"] != view_json
                or contract["action_contract_json"] != action_json
            ):
                raise ReplayError(
                    "cohort schema/view/action contract differs from its campaign"
                )

            connection.execute(
                "INSERT INTO cohorts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["logical_id"],
                    record["campaign_id"],
                    record["macro_round"],
                    record["cohort_index"],
                    record["purpose"],
                    record["cohort_manifest_path"],
                    record["manifest_sha256"],
                    record["manifest_schema"],
                    record["transition_count"],
                    view_json,
                    action_json,
                ),
            )
            connection.executemany(
                "INSERT INTO shards VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record["logical_id"],
                        shard["shard_id"],
                        shard["shard_manifest_path"],
                        shard["manifest_sha256"],
                        shard["payload_path"],
                        shard["payload_sha256"],
                        shard["transition_count"],
                    )
                    for shard in record["shards"]
                ],
            )
            connection.commit()
        except ReplayError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise ReplayError(f"transactional cohort ingestion failed: {exc}") from exc
        return {
            "status": "ACCEPTED",
            "logical_id": record["logical_id"],
            "manifest_sha256": record["manifest_sha256"],
            "transition_count": record["transition_count"],
        }

    def counts(self, campaign_id: str, *, include_eval: bool = False) -> dict[str, Any]:
        """Return deterministic accepted transition counts grouped by round/cohort."""

        campaign_id = _identifier(campaign_id, "campaign_id")
        purpose_clause = "" if include_eval else " AND purpose = 'train'"
        rows = self._connection.execute(
            "SELECT macro_round, cohort_index, purpose, transition_count "
            "FROM cohorts WHERE campaign_id = ?"
            + purpose_clause
            + " ORDER BY macro_round, cohort_index, purpose",
            (campaign_id,),
        ).fetchall()
        by_round: dict[str, int] = {}
        by_round_cohort: dict[str, int] = {}
        for row in rows:
            round_key = str(row["macro_round"])
            cohort_key = f"{row['macro_round']}:{row['cohort_index']}:{row['purpose']}"
            by_round[round_key] = by_round.get(round_key, 0) + row["transition_count"]
            by_round_cohort[cohort_key] = row["transition_count"]
        return {
            "cohort_count": len(rows),
            "transition_count": sum(row["transition_count"] for row in rows),
            "by_round": by_round,
            "by_round_cohort": by_round_cohort,
        }

    def build_snapshot(
        self,
        campaign_id: str,
        *,
        through_macro_round: int,
        output: str | os.PathLike[str],
        expected_train_cohorts: int = 4,
    ) -> dict[str, Any]:
        """Publish an immutable cumulative training snapshot through round ``k``."""

        campaign_id = _identifier(campaign_id, "campaign_id")
        through_macro_round = _true_int(
            through_macro_round, "through_macro_round"
        )
        expected_train_cohorts = _true_int(
            expected_train_cohorts, "expected_train_cohorts", minimum=1
        )
        contract = self._connection.execute(
            "SELECT manifest_schema, view_contract_json, action_contract_json "
            "FROM campaign_contracts WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        if contract is None:
            raise ReplayError("campaign has no accepted cohorts")
        rows = self._connection.execute(
            "SELECT * FROM cohorts WHERE campaign_id = ? AND purpose = 'train' "
            "AND macro_round <= ? ORDER BY macro_round, cohort_index",
            (campaign_id, through_macro_round),
        ).fetchall()
        expected_indices = set(range(expected_train_cohorts))
        present_rounds: dict[int, set[int]] = {}
        for row in rows:
            present_rounds.setdefault(row["macro_round"], set()).add(row["cohort_index"])
        expected_rounds = set(range(through_macro_round + 1))
        if set(present_rounds) != expected_rounds:
            raise ReplayError("replay snapshot range contains an unsealed macro-round")
        for macro_round in range(through_macro_round + 1):
            if present_rounds[macro_round] != expected_indices:
                raise ReplayError(
                    f"macro-round {macro_round} does not contain exactly "
                    f"cohorts 0..{expected_train_cohorts - 1}"
                )

        cohorts: list[dict[str, Any]] = []
        by_round: dict[str, int] = {}
        by_round_cohort: dict[str, int] = {}
        for row in rows:
            shard_rows = self._connection.execute(
                "SELECT shard_id, shard_manifest_path, manifest_sha256, "
                "payload_path, payload_sha256, transition_count "
                "FROM shards WHERE logical_id = ? ORDER BY shard_id",
                (row["logical_id"],),
            ).fetchall()
            shards = [dict(shard) for shard in shard_rows]
            cohort = {
                "logical_id": row["logical_id"],
                "macro_round": row["macro_round"],
                "cohort_index": row["cohort_index"],
                "cohort_manifest_path": row["cohort_manifest_path"],
                "manifest_sha256": row["manifest_sha256"],
                "transition_count": row["transition_count"],
                "shards": shards,
            }
            seed_contract, trajectory_fingerprints = _stored_seed_provenance(
                row["cohort_manifest_path"], row["manifest_sha256"]
            )
            if (seed_contract is None) != (trajectory_fingerprints is None):
                raise ReplayError(
                    "accepted cohort seed/trajectory provenance is incomplete"
                )
            if seed_contract is not None:
                cohort[_ACTOR_SEED_FIELD] = seed_contract
                cohort[_TRAJECTORY_FINGERPRINT_FIELD] = trajectory_fingerprints
            cohorts.append(cohort)
            round_key = str(row["macro_round"])
            cohort_key = f"{row['macro_round']}:{row['cohort_index']}"
            by_round[round_key] = by_round.get(round_key, 0) + row["transition_count"]
            by_round_cohort[cohort_key] = row["transition_count"]

        content: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "campaign_id": campaign_id,
            "through_macro_round": through_macro_round,
            "expected_train_cohorts": expected_train_cohorts,
            "cohort_manifest_schema": contract["manifest_schema"],
            "view_contract": json.loads(contract["view_contract_json"]),
            "action_contract": json.loads(contract["action_contract_json"]),
            "cohort_count": len(cohorts),
            "transition_count": sum(item["transition_count"] for item in cohorts),
            "counts_by_round": by_round,
            "counts_by_round_cohort": by_round_cohort,
            "cohorts": cohorts,
        }
        content["snapshot_sha256"] = _canonical_sha256(content)
        output_path = Path(os.path.abspath(Path(output).expanduser()))
        try:
            file_sha256 = _publish_immutable_json(output_path, content)
        except Exception as exc:
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError(f"replay snapshot publication failed: {exc}") from exc
        result = dict(content)
        result["path"] = str(output_path)
        result["file_sha256"] = file_sha256
        return result


def load_snapshot(
    path: str | os.PathLike[str], *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    """Load a frozen replay snapshot and revalidate every referenced artifact.

    The returned ``resolved_shards`` list is the learner-facing payload
    discovery surface.  It is deliberately derived at load time and is not
    part of the immutable snapshot's semantic digest.
    """

    snapshot_path = Path(os.path.abspath(Path(path).expanduser()))
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ReplayError("replay snapshot must be a regular non-symlink file")
    actual_file_sha256 = _sha256_file(snapshot_path)
    if expected_file_sha256 is not None and _sha256(
        expected_file_sha256, "expected_file_sha256"
    ) != actual_file_sha256:
        raise ReplayError("replay snapshot file digest mismatch")
    try:
        loader = getattr(_artifacts, "load_strict_json")
        document = loader(snapshot_path)
    except Exception as exc:
        raise ReplayError(f"cannot load strict replay snapshot: {exc}") from exc
    expected_fields = {
        "schema",
        "campaign_id",
        "through_macro_round",
        "expected_train_cohorts",
        "cohort_manifest_schema",
        "view_contract",
        "action_contract",
        "cohort_count",
        "transition_count",
        "counts_by_round",
        "counts_by_round_cohort",
        "cohorts",
        "snapshot_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_fields:
        raise ReplayError("replay snapshot has unexpected or missing fields")
    document = dict(document)
    if document["schema"] != SNAPSHOT_SCHEMA:
        raise ReplayError("replay snapshot schema mismatch")
    _identifier(document["campaign_id"], "snapshot campaign_id")
    through_round = _true_int(
        document["through_macro_round"], "snapshot through_macro_round"
    )
    expected_cohorts = _true_int(
        document["expected_train_cohorts"],
        "snapshot expected_train_cohorts",
        minimum=1,
    )
    _exact_mapping(document["view_contract"], "snapshot view_contract")
    _exact_mapping(document["action_contract"], "snapshot action_contract")
    declared_semantic_sha = _sha256(
        document["snapshot_sha256"], "snapshot_sha256"
    )
    semantic_body = dict(document)
    semantic_body.pop("snapshot_sha256")
    if _canonical_sha256(semantic_body) != declared_semantic_sha:
        raise ReplayError("replay snapshot semantic digest mismatch")

    raw_cohorts = document["cohorts"]
    if not isinstance(raw_cohorts, list):
        raise ReplayError("replay snapshot cohorts must be a list")
    if _true_int(document["cohort_count"], "snapshot cohort_count") != len(
        raw_cohorts
    ):
        raise ReplayError("replay snapshot cohort count mismatch")
    expected_pairs = [
        (macro_round, cohort_index)
        for macro_round in range(through_round + 1)
        for cohort_index in range(expected_cohorts)
    ]
    actual_pairs: list[tuple[int, int]] = []
    logical_ids: set[str] = set()
    cohort_digests: set[str] = set()
    shard_digests: set[str] = set()
    resolved_shards: list[dict[str, Any]] = []
    by_round: dict[str, int] = {}
    by_round_cohort: dict[str, int] = {}
    seed_presence_by_round: dict[int, set[bool]] = {}
    simulator_seeds_by_round: dict[int, set[int]] = {}
    generation_seeds_by_round: dict[int, set[int]] = {}
    trajectory_digests_by_round: dict[int, set[str]] = {}
    total = 0
    for cohort_number, raw_cohort in enumerate(raw_cohorts):
        if not isinstance(raw_cohort, Mapping):
            raise ReplayError(f"snapshot cohorts[{cohort_number}] is malformed")
        required_cohort = {
            "logical_id",
            "macro_round",
            "cohort_index",
            "cohort_manifest_path",
            "manifest_sha256",
            "transition_count",
            "shards",
        }
        extended_cohort = required_cohort | {
            _ACTOR_SEED_FIELD,
            _TRAJECTORY_FINGERPRINT_FIELD,
        }
        cohort_keys = set(raw_cohort)
        if cohort_keys != required_cohort and cohort_keys != extended_cohort:
            raise ReplayError("snapshot cohort has unexpected or missing fields")
        has_seed = _ACTOR_SEED_FIELD in raw_cohort
        logical_id = str(raw_cohort["logical_id"])
        if logical_id in logical_ids:
            raise ReplayError("snapshot contains a duplicate cohort logical ID")
        logical_ids.add(logical_id)
        macro_round = _true_int(raw_cohort["macro_round"], "snapshot macro_round")
        cohort_index = _true_int(
            raw_cohort["cohort_index"], "snapshot cohort_index"
        )
        actual_pairs.append((macro_round, cohort_index))
        cohort_count = _true_int(
            raw_cohort["transition_count"],
            "snapshot cohort transition_count",
            minimum=1,
        )
        manifest_sha = _sha256(
            raw_cohort["manifest_sha256"], "snapshot cohort manifest digest"
        )
        if manifest_sha in cohort_digests:
            raise ReplayError("snapshot contains a duplicate cohort manifest digest")
        cohort_digests.add(manifest_sha)
        cohort_manifest = Path(str(raw_cohort["cohort_manifest_path"]))
        if not cohort_manifest.is_absolute():
            raise ReplayError("snapshot cohort manifest path must be absolute")
        if cohort_manifest.is_symlink() or not cohort_manifest.is_file():
            raise ReplayError("snapshot cohort manifest path is unsafe")
        cohort_manifest = cohort_manifest.resolve(strict=True)
        if _sha256_file(cohort_manifest) != manifest_sha:
            raise ReplayError("snapshot cohort manifest digest mismatch at load")
        cohort_root = cohort_manifest.parent
        stored_seed, stored_fingerprints = _stored_seed_provenance(
            str(cohort_manifest), manifest_sha
        )
        if has_seed:
            seed_contract = _exact_mapping(
                raw_cohort[_ACTOR_SEED_FIELD], "snapshot actor seed contract"
            )
            fingerprints = _exact_mapping(
                raw_cohort[_TRAJECTORY_FINGERPRINT_FIELD],
                "snapshot physical/teacher trajectory fingerprints",
            )
            if seed_contract != stored_seed or fingerprints != stored_fingerprints:
                raise ReplayError(
                    "snapshot seed/trajectory provenance differs from cohort manifest"
                )
            assignments = seed_contract.get("assignments")
            if not isinstance(assignments, list):
                raise ReplayError("snapshot actor seed assignments are malformed")
            try:
                ranks = sorted(int(item["rank"]) for item in assignments)
                environments = sorted(
                    int(global_env_id)
                    for item in assignments
                    for global_env_id in item["global_env_ids"]
                )
                validate_actor_seed_contract(
                    seed_contract,
                    campaign_id=document["campaign_id"],
                    macro_round=macro_round,
                    cohort_index=cohort_index,
                    expected_ranks=ranks,
                    expected_global_env_ids=environments,
                )
            except (ActorSeedError, KeyError, TypeError, ValueError) as exc:
                raise ReplayError(f"snapshot actor seed provenance is invalid: {exc}") from exc
            if set(fingerprints) != {str(item) for item in environments}:
                raise ReplayError(
                    "snapshot trajectory fingerprints do not cover seeded environments"
                )
            for digest in fingerprints.values():
                _sha256(digest, "snapshot physical/teacher trajectory SHA-256")
            simulator = {int(item["simulator_seed"]) for item in assignments}
            generation = {int(item["generation_seed"]) for item in assignments}
            trajectory = set(fingerprints.values())
            observed_simulator = simulator_seeds_by_round.setdefault(macro_round, set())
            observed_generation = generation_seeds_by_round.setdefault(macro_round, set())
            observed_trajectories = trajectory_digests_by_round.setdefault(
                macro_round, set()
            )
            if observed_simulator & simulator:
                raise ReplayError("snapshot macro-round reuses an actor simulator seed")
            if observed_generation & generation:
                raise ReplayError(
                    "snapshot macro-round reuses an Image3F generation seed"
                )
            if observed_trajectories & trajectory:
                raise ReplayError(
                    "snapshot macro-round repeats a physical/teacher trajectory"
                )
            observed_simulator.update(simulator)
            observed_generation.update(generation)
            observed_trajectories.update(trajectory)
        elif stored_seed is not None or stored_fingerprints is not None:
            raise ReplayError(
                "snapshot omitted seed/trajectory provenance from its cohort manifest"
            )
        seed_presence_by_round.setdefault(macro_round, set()).add(has_seed)

        raw_shards = raw_cohort["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ReplayError("snapshot cohort has no shards")
        shard_total = 0
        for raw_shard in raw_shards:
            required_shard = {
                "shard_id",
                "shard_manifest_path",
                "manifest_sha256",
                "payload_path",
                "payload_sha256",
                "transition_count",
            }
            if not isinstance(raw_shard, Mapping) or set(raw_shard) != required_shard:
                raise ReplayError("snapshot shard has unexpected or missing fields")
            shard_id = _identifier(raw_shard["shard_id"], "snapshot shard_id")
            manifest_relative = _safe_relative_path(
                raw_shard["shard_manifest_path"], "snapshot shard manifest path"
            )
            payload_relative = _safe_relative_path(
                raw_shard["payload_path"], "snapshot payload path"
            )
            shard_manifest = _safe_artifact_file(
                cohort_root, manifest_relative, "snapshot shard manifest"
            )
            payload = _safe_artifact_file(
                cohort_root, payload_relative, "snapshot shard payload"
            )
            shard_manifest_sha = _sha256(
                raw_shard["manifest_sha256"], "snapshot shard manifest digest"
            )
            payload_sha = _sha256(
                raw_shard["payload_sha256"], "snapshot payload digest"
            )
            if shard_manifest_sha in shard_digests:
                raise ReplayError("snapshot contains a duplicate shard manifest digest")
            shard_digests.add(shard_manifest_sha)
            if _sha256_file(shard_manifest) != shard_manifest_sha:
                raise ReplayError("snapshot shard manifest digest mismatch at load")
            if _sha256_file(payload) != payload_sha:
                raise ReplayError("snapshot shard payload digest mismatch at load")
            count = _true_int(
                raw_shard["transition_count"],
                "snapshot shard transition_count",
                minimum=1,
            )
            shard_total += count
            resolved_shards.append(
                {
                    "logical_id": logical_id,
                    "macro_round": macro_round,
                    "cohort_index": cohort_index,
                    "shard_id": shard_id,
                    "shard_manifest_path": str(shard_manifest),
                    "manifest_sha256": shard_manifest_sha,
                    "payload_path": str(payload),
                    "payload_sha256": payload_sha,
                    "transition_count": count,
                }
            )
        if shard_total != cohort_count:
            raise ReplayError("snapshot cohort/shard transition counts disagree")
        total += cohort_count
        round_key = str(macro_round)
        cohort_key = f"{macro_round}:{cohort_index}"
        by_round[round_key] = by_round.get(round_key, 0) + cohort_count
        by_round_cohort[cohort_key] = cohort_count

    if actual_pairs != expected_pairs:
        raise ReplayError("snapshot does not contain a complete ordered round range")
    if any(len(values) != 1 for values in seed_presence_by_round.values()):
        raise ReplayError(
            "snapshot macro-round mixes seeded and legacy cohort provenance"
        )
    if _true_int(document["transition_count"], "snapshot transition_count") != total:
        raise ReplayError("snapshot total transition count mismatch")
    if document["counts_by_round"] != by_round:
        raise ReplayError("snapshot counts_by_round mismatch")
    if document["counts_by_round_cohort"] != by_round_cohort:
        raise ReplayError("snapshot counts_by_round_cohort mismatch")
    result = dict(document)
    result["path"] = str(snapshot_path)
    result["file_sha256"] = actual_file_sha256
    result["resolved_shards"] = resolved_shards
    return result


def deterministic_partition(key: str, partitions: int, *, seed: int = 0) -> int:
    """Map a stable logical key to one partition without Python hash randomization."""

    if not isinstance(key, str) or not key:
        raise ReplayError("partition key must be a non-empty string")
    partitions = _true_int(partitions, "partitions", minimum=1)
    seed = _true_int(seed, "seed")
    digest = hashlib.blake2b(
        f"{seed}\0{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % partitions


def deterministic_sample(
    items: Iterable[str], count: int, *, seed: int, namespace: str = "replay"
) -> tuple[str, ...]:
    """Select a deterministic, order-independent sample without replacement."""

    count = _true_int(count, "count")
    seed = _true_int(seed, "seed")
    if not isinstance(namespace, str) or not namespace:
        raise ReplayError("sample namespace must be a non-empty string")
    normalized = list(items)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ReplayError("sample items must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ReplayError("sample items must be unique")
    if count > len(normalized):
        raise ReplayError("sample count exceeds the population")

    def score(item: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            f"{seed}\0{namespace}\0{item}".encode("utf-8")
        ).digest()
        return digest, item

    return tuple(item for _digest, item in sorted(map(score, normalized))[:count])
