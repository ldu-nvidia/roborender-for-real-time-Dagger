"""Deterministic, provenance-preserving replay selection for three-view DAgger.

Replay is deliberately a learner-side concern.  The RTX owner still publishes
the complete current window and never receives historical images.  A disabled
or absent replay config preserves the historical current-iteration-only path.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.dagger.services.contracts import canonical_json, file_sha256


REPLAY_SCHEMA = "roborender.dagger.replay.v1"
REPLAY_MANIFEST_SCHEMA = "roborender.dagger.replay_manifest.v1"
SOURCES = ("current_dagger", "bc", "prior_dagger")
CORRECTED_BC_RENDER_CONTRACT = "raven_three_view_autoregressive_v1"


def validate_corrected_bc_chunk(path: Path) -> None:
    """Reject the old independently rendered BC pool at the replay boundary."""

    required = {
        "render_contract", "render_chain_id", "render_chain_index",
        "render_chain_length", "render_inherits_previous_reference",
    }
    try:
        with np.load(path, allow_pickle=False) as chunk:
            if not required.issubset(chunk.files):
                raise ValueError("BC replay chunk lacks autoregressive chain provenance")
            if str(chunk["render_contract"].item()) != CORRECTED_BC_RENDER_CONTRACT:
                raise ValueError("BC replay chunk uses an unsupported render contract")
            chain_id = str(chunk["render_chain_id"].item())
            index = int(chunk["render_chain_index"].item())
            length = int(chunk["render_chain_length"].item())
            inherits = bool(chunk["render_inherits_previous_reference"].item())
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BC replay chunk"):
            raise
        raise ValueError(f"BC replay chunk provenance is unreadable: {path}") from exc
    if not chain_id or index < 0 or length < 1 or index >= length:
        raise ValueError("BC replay chunk has invalid chain identity or bounds")
    if inherits != (index > 0):
        raise ValueError("BC replay previous-reference flag differs from chain index")


def load_replay_config(config: Any) -> dict[str, Any]:
    """Validate replay configuration without changing the legacy default."""

    if config is None:
        return {"enabled": False}
    if not isinstance(config, dict):
        raise ValueError("replay must be an object")
    if config == {"enabled": False}:
        return config
    required = {
        "schema", "enabled", "total_chunks_per_update", "fractions", "seed",
        "bc_train_manifest", "bc_train_manifest_sha256", "empty_pool_fallback",
    }
    if set(config) != required:
        raise ValueError(
            f"replay keys differ: missing={sorted(required - set(config))}, "
            f"unknown={sorted(set(config) - required)}"
        )
    if config["schema"] != REPLAY_SCHEMA or config["enabled"] is not True:
        raise ValueError("enabled replay must use roborender.dagger.replay.v1")
    total = config["total_chunks_per_update"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("replay total_chunks_per_update must be positive")
    fractions = config["fractions"]
    if not isinstance(fractions, dict) or set(fractions) != set(SOURCES):
        raise ValueError(f"replay fractions must contain exactly {list(SOURCES)}")
    values = []
    for source in SOURCES:
        value = fractions[source]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"replay fraction {source} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"replay fraction {source} must be in [0,1]")
        values.append(value)
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("replay fractions must sum to one")
    if fractions["current_dagger"] <= 0:
        raise ValueError("replay must retain current-iteration data")
    if not isinstance(config["seed"], str) or not config["seed"]:
        raise ValueError("replay seed must be a non-empty string")
    if config["empty_pool_fallback"] != "bc_then_current":
        raise ValueError("only deterministic bc_then_current fallback is supported")
    digest = config["bc_train_manifest_sha256"]
    if (
        not isinstance(digest, str) or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("bc_train_manifest_sha256 must be a SHA-256 digest")
    manifest = Path(config["bc_train_manifest"]).resolve()
    if not manifest.is_file() or file_sha256(manifest) != digest:
        raise ValueError("BC train manifest is missing or its hash differs")
    return config


def _rank(seed: str, source: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{source}\0{identity}".encode()).hexdigest()


def _target_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {source: total * float(fractions[source]) for source in SOURCES}
    result = {source: int(math.floor(raw[source])) for source in SOURCES}
    remaining = total - sum(result.values())
    order = sorted(SOURCES, key=lambda source: (-(raw[source] - result[source]), source))
    for source in order[:remaining]:
        result[source] += 1
    return result


def _bc_pool(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    split = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split.get("schema") != "roborender.bc_split.v1" or split.get("split") != "train":
        raise ValueError("BC replay catalog must be the sealed train split")
    if split.get("chunk_count") != len(split.get("items", [])):
        raise ValueError("BC replay train split count differs from its items")
    source_path = (manifest_path.parent / split["source_manifest"]).resolve()
    expected_source_sha = split["source_manifest_sha256"]
    if file_sha256(source_path) != expected_source_sha:
        raise ValueError("BC replay source manifest hash differs from train split")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    indexed = {item["chunk_id"]: item for item in source["chunks"]}
    pool = []
    seen = set()
    for item in split["items"]:
        chunk_id = item["chunk_id"]
        if chunk_id in seen or chunk_id not in indexed:
            raise ValueError("BC replay split contains a duplicate or unknown chunk")
        seen.add(chunk_id)
        authoritative = indexed[chunk_id]
        if item["rendered_npz"] != authoritative["rendered_npz"]:
            raise ValueError("BC replay split path differs from source manifest")
        path = (source_path.parent / authoritative["rendered_npz"]).resolve()
        pool.append({
            "source": "bc", "chunk_id": chunk_id, "path": str(path),
            "sha256": authoritative["rendered_sha256"], "split": "train",
        })
    return pool, {
        "bc_train_manifest": str(manifest_path.resolve()),
        "bc_train_manifest_sha256": file_sha256(manifest_path),
        "bc_source_manifest_sha256": expected_source_sha,
    }


def build_replay_manifest(
    *, config: dict[str, Any], run_id: str, round_id: int, iteration_id: int,
    barriers: Iterable[dict[str, Any]], iterations: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select a deterministic update from current, BC, and eligible history."""

    config = load_replay_config(config)
    if not config.get("enabled", False):
        return None
    current, prior = [], []
    iteration_states = {
        (row["round_id"], row["iteration_id"]): row["status"] for row in iterations
    }
    for barrier in barriers:
        key = (barrier["round_id"], barrier["iteration_id"])
        for chunk_id in barrier["chunk_ids"]:
            sample = {
                "source": "current_dagger" if key == (round_id, iteration_id)
                else "prior_dagger",
                "chunk_id": chunk_id, "run_id": run_id,
                "round_id": key[0], "iteration_id": key[1],
                "actor_id": barrier["actor_id"],
            }
            if key == (round_id, iteration_id):
                current.append(sample)
            elif key < (round_id, iteration_id) and iteration_states.get(key) in {
                "advanced", "complete"
            }:
                prior.append(sample)
    if not current:
        raise ValueError("replay selection requires current-iteration chunks")
    bc, catalog = _bc_pool(Path(config["bc_train_manifest"]).resolve())
    pools = {"current_dagger": current, "bc": bc, "prior_dagger": prior}
    total = config["total_chunks_per_update"]
    targets = _target_counts(total, config["fractions"])
    # Iteration one has no history.  Move any unavailable historical quota to
    # BC first, then current, without changing the total update size.
    actual = dict(targets)
    deficit = 0
    for source in SOURCES:
        if actual[source] > len(pools[source]):
            deficit += actual[source] - len(pools[source])
            actual[source] = len(pools[source])
    for source in ("bc", "current_dagger", "prior_dagger"):
        room = len(pools[source]) - actual[source]
        moved = min(room, deficit)
        actual[source] += moved
        deficit -= moved
    if deficit:
        raise ValueError("replay pools cannot satisfy total_chunks_per_update")
    seed = f"{config['seed']}:{run_id}:r{round_id}:i{iteration_id}"
    selected = []
    for source in SOURCES:
        ranked = sorted(
            pools[source], key=lambda row: (_rank(seed, source, row["chunk_id"]), row["chunk_id"])
        )
        selected.extend(ranked[:actual[source]])
    manifest = {
        "schema": REPLAY_MANIFEST_SCHEMA,
        "run_id": run_id, "round_id": round_id, "iteration_id": iteration_id,
        "seed": seed, "total_chunks": total,
        "requested_fractions": {key: float(config["fractions"][key]) for key in SOURCES},
        "requested_counts": targets, "selected_counts": actual,
        "fallback": config["empty_pool_fallback"], "catalog": catalog,
        "samples": selected,
    }
    manifest["sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return manifest


def validate_replay_manifest(
    manifest: Any, *, db: sqlite3.Connection, run_id: str, round_id: int,
    iteration_id: int, admitted_current: set[str], bc_root: Path,
) -> tuple[list[str], list[Path]]:
    """Verify a submitted manifest against durable coordinator provenance."""

    if not isinstance(manifest, dict) or manifest.get("schema") != REPLAY_MANIFEST_SCHEMA:
        raise ValueError("invalid replay manifest schema")
    required_manifest_keys = {
        "schema", "run_id", "round_id", "iteration_id", "seed", "total_chunks",
        "requested_fractions", "requested_counts", "selected_counts", "fallback",
        "catalog", "samples", "sha256",
    }
    if set(manifest) != required_manifest_keys:
        raise ValueError("replay manifest keys differ from the frozen contract")
    supplied_sha = manifest.get("sha256")
    body = dict(manifest)
    body.pop("sha256", None)
    expected_sha = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    if supplied_sha != expected_sha:
        raise ValueError("replay manifest digest differs")
    if (manifest.get("run_id"), manifest.get("round_id"), manifest.get("iteration_id")) != (
        run_id, round_id, iteration_id
    ):
        raise ValueError("replay manifest targets a different learner iteration")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != manifest.get("total_chunks"):
        raise ValueError("replay sample count differs")
    fractions = manifest.get("requested_fractions")
    total_chunks = manifest.get("total_chunks")
    if (
        isinstance(total_chunks, bool) or not isinstance(total_chunks, int)
        or total_chunks < 1
        or not isinstance(fractions, dict) or set(fractions) != set(SOURCES)
        or any(
            isinstance(fractions[source], bool)
            or not isinstance(fractions[source], (int, float))
            or not math.isfinite(float(fractions[source]))
            or not 0 <= float(fractions[source]) <= 1
            for source in SOURCES
        )
        or not math.isclose(sum(float(fractions[source]) for source in SOURCES), 1.0,
                            rel_tol=0, abs_tol=1e-9)
        or _target_counts(total_chunks, fractions)
        != manifest.get("requested_counts")
        or manifest.get("fallback") != "bc_then_current"
    ):
        raise ValueError("replay requested mixture is internally inconsistent")
    dagger_ids, bc_paths, identities = [], [], set()
    selected_counts = {source: 0 for source in SOURCES}
    bc_root = bc_root.resolve()
    catalog = manifest.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("replay manifest lacks its BC catalog identity")
    bc_manifest_path = Path(catalog.get("bc_train_manifest", "")).resolve()
    if (
        not bc_manifest_path.is_file()
        or file_sha256(bc_manifest_path) != catalog.get("bc_train_manifest_sha256")
    ):
        raise ValueError("replay BC train catalog is missing or changed")
    bc_pool, authoritative_catalog = _bc_pool(bc_manifest_path)
    if authoritative_catalog != catalog:
        raise ValueError("replay BC catalog provenance differs")
    bc_by_id = {sample["chunk_id"]: sample for sample in bc_pool}
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("source") not in SOURCES:
            raise ValueError("invalid replay sample")
        source = sample["source"]
        expected_sample_keys = (
            {"source", "chunk_id", "path", "sha256", "split"}
            if source == "bc"
            else {"source", "chunk_id", "run_id", "round_id", "iteration_id", "actor_id"}
        )
        if set(sample) != expected_sample_keys:
            raise ValueError("replay sample keys differ from the frozen contract")
        identity = (source, sample.get("chunk_id"))
        if identity in identities:
            raise ValueError("duplicate replay sample")
        identities.add(identity)
        selected_counts[source] += 1
        if source == "bc":
            if sample.get("split") != "train":
                raise ValueError("validation/test BC data cannot enter replay")
            path = Path(sample.get("path", "")).resolve()
            if path.parent != bc_root or not path.is_file():
                raise ValueError("BC replay path is outside the pinned rendered train root")
            expected = bc_by_id.get(sample.get("chunk_id"))
            if expected != sample:
                raise ValueError("BC replay sample is not in the pinned train split")
            validate_corrected_bc_chunk(path)
            stat = path.stat()
            attested = db.execute(
                "SELECT size_bytes,mtime_ns,sha256 FROM replay_bc_attestations "
                "WHERE path=?", (str(path),),
            ).fetchone()
            if not (
                attested is not None
                and attested["size_bytes"] == stat.st_size
                and attested["mtime_ns"] == stat.st_mtime_ns
                and attested["sha256"] == sample["sha256"]
            ):
                if file_sha256(path) != sample["sha256"]:
                    raise ValueError("BC replay chunk bytes differ from source manifest")
                db.execute(
                    "INSERT OR REPLACE INTO replay_bc_attestations VALUES (?,?,?,?,?)",
                    (str(path), stat.st_size, stat.st_mtime_ns, sample["sha256"], time.time()),
                )
            bc_paths.append(path)
            continue
        chunk_id = sample.get("chunk_id")
        if sample.get("run_id") != run_id:
            raise ValueError("DAgger replay sample belongs to a different run")
        dagger_ids.append(chunk_id)
        if source == "current_dagger":
            if chunk_id not in admitted_current:
                raise ValueError("current replay sample was not admitted by this barrier")
            if (sample.get("round_id"), sample.get("iteration_id")) != (round_id, iteration_id):
                raise ValueError("current replay provenance differs")
        else:
            key = (sample.get("round_id"), sample.get("iteration_id"))
            if key >= (round_id, iteration_id):
                raise ValueError("prior replay sample is not from the past")
            rows = db.execute(
                "SELECT i.status,b.chunk_ids_json FROM iteration_barriers b "
                "JOIN round_iterations i "
                "ON i.run_id=b.run_id AND i.round_id=b.round_id AND "
                "i.iteration_id=b.iteration_id WHERE b.run_id=? AND b.round_id=? "
                "AND b.iteration_id=? AND b.actor_id=?",
                (run_id, key[0], key[1], sample.get("actor_id")),
            ).fetchall()
            valid = any(
                row["status"] in {"advanced", "complete"}
                and chunk_id in json.loads(row["chunk_ids_json"])
                for row in rows
            )
            if not valid:
                raise ValueError("prior replay sample lacks a completed durable barrier")
    if selected_counts != manifest.get("selected_counts"):
        raise ValueError("replay selected_counts differs from samples")
    return dagger_ids, bc_paths
