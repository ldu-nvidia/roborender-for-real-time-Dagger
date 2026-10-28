"""Export genuine held-out chunks from a sealed ledger without reading arrays.

The native collector must emit purpose/seed/motion assignment receipts. Missing
historical receipts cannot be reconstructed from an env index or window name.
This exporter does not collect data, create mock populations or validate pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from pipeline.dagger.beta_campaign import immutable_json
from pipeline.dagger.services.contracts import file_sha256, sha256_digest
from pipeline.dagger.services.heldout_mse import MANIFEST_SCHEMA, SCOPED_MANIFEST_SCHEMA, VIEWS, canonical_digest, source_window_identity, source_episode_identity

COLLECTION_SCHEMA = "roborender.dagger-heldout-collection.v1"
SCOPED_COLLECTION_SCHEMA = "roborender.dagger-heldout-collection.v2"
ADMISSION_SCHEMA = "roborender.dagger-heldout-admission.v1"
ASSIGNMENT_SCHEMA = "roborender.native-episode-assignment.v1"
ASSIGNMENT_KEYS = {"episode_id", "simulator_seed", "motion_id", "motion_sha256", "observation_source"}


def scoped_assignment_key(collection, row):
    if collection["schema"] == SCOPED_COLLECTION_SCHEMA:
        return source_episode_identity({"run_id": collection["source_run_id"], "round_id": collection["round_id"],
                                       "actor_id": row["actor_id"]}, row["episode_id"])
    return row["episode_id"]


def chunk_episode_keys(collection, metadata, raw_ids):
    if collection["schema"] == SCOPED_COLLECTION_SCHEMA:
        return sorted(source_episode_identity(metadata, value) for value in set(raw_ids))
    return sorted(set(raw_ids))


def load_collection(path, digest):
    path = Path(path)
    if not path.is_absolute() or file_sha256(path) != digest:
        raise ValueError("immutable native held-out collection receipt changed")
    value = json.loads(path.read_bytes())
    fields = {"schema", "purpose", "source_run_id", "round_id", "policy_spec_sha256", "teacher_checkpoint_sha256",
              "collection_checkpoint_sha256", "teacher_beta", "motion_inventory_sha256", "motion_support",
              "training_simulator_seeds", "test_simulator_seeds", "chunk_ids", "episode_assignments"}
    scoped = value.get("schema") == SCOPED_COLLECTION_SCHEMA
    if (set(value) != fields | ({"collection_plan_path", "collection_plan_sha256"} if scoped else set())
            or value["schema"] not in {COLLECTION_SCHEMA, SCOPED_COLLECTION_SCHEMA} or value["purpose"] != "validation"):
        raise ValueError("only a real validation-purpose collection receipt is accepted")
    actors = {}
    if scoped:
        plan_path = Path(value["collection_plan_path"])
        if not plan_path.is_absolute() or file_sha256(plan_path) != value["collection_plan_sha256"]:
            raise ValueError("scoped collection plan changed")
        plan = json.loads(plan_path.read_bytes())
        if (plan.get("schema") != "roborender.dagger-heldout-plan.v1"
                or plan["run_id"] != value["source_run_id"] or plan["round_id"] != value["round_id"]
                or any(plan[key] != value[key] for key in ("policy_spec_sha256", "teacher_checkpoint_sha256", "motion_inventory_sha256", "motion_support"))):
            raise ValueError("scoped collection and native cohort plan differ")
        from pipeline.dagger.heldout_collection_plan import assignment_for
        for cohort in plan["cohorts"]:
            assignment = assignment_for(plan, cohort)
            for lane in assignment["lanes"]:
                actor = f"{plan['run_id']}-{cohort['cohort_id']}-lane-{lane['lane']:03d}"
                if actor in actors:
                    raise ValueError("duplicate native cohort actor")
                actors[actor] = (assignment, lane)
    if type(value["round_id"]) is not int or value["round_id"] not in (1, 2, 3, 4):
        raise ValueError("invalid held-out round")
    if value["teacher_beta"] != {1: .75, 2: .5, 3: .25, 4: 0.}[value["round_id"]]:
        raise ValueError("held-out behavior mixture must match its declared training round")
    for key in ("policy_spec_sha256", "teacher_checkpoint_sha256", "collection_checkpoint_sha256", "motion_inventory_sha256"):
        sha256_digest(value[key], key)
    for key in ("motion_support", "chunk_ids", "training_simulator_seeds", "test_simulator_seeds"):
        rows = value[key]
        if not isinstance(rows, list) or not rows or len(rows) != len(set(rows)):
            raise ValueError(f"unique nonempty {key} pins are required")
    for key in ("training_simulator_seeds", "test_simulator_seeds"):
        if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in value[key]):
            raise ValueError("simulator seeds must be actual nonnegative integer seeds")
    episodes, seeds, motions = {}, set(), set()
    for row in value["episode_assignments"]:
        extras = {"actor_id", "native_reset_path", "native_reset_sha256"} if scoped else set()
        if set(row) != ASSIGNMENT_KEYS | {"native_assignment_path", "native_assignment_sha256"} | extras:
            raise ValueError("each episode requires explicit native assignment evidence")
        key = scoped_assignment_key(value, row)
        if not row["episode_id"] or key in episodes:
            raise ValueError("duplicate or missing actual episode identity")
        if type(row["simulator_seed"]) is not int or not 0 <= row["simulator_seed"] < 2**31:
            raise ValueError("actual native simulator seed is missing")
        if not isinstance(row["motion_id"], str) or not row["motion_id"] or row["observation_source"] not in {"raytraced", "generated_controls"}:
            raise ValueError("actual motion/observation identity is missing")
        sha256_digest(row["motion_sha256"], "motion_sha256")
        evidence = Path(row["native_assignment_path"])
        if not evidence.is_absolute() or file_sha256(evidence) != row["native_assignment_sha256"]:
            raise ValueError("native episode assignment evidence changed")
        recorded = json.loads(evidence.read_bytes())
        expected = {"schema": ASSIGNMENT_SCHEMA, "purpose": "validation", "run_id": value["source_run_id"],
                    **{key: row[key] for key in ASSIGNMENT_KEYS}}
        if recorded != expected:
            raise ValueError("declared assignment does not match actual native receipt")
        if scoped:
            if row["actor_id"] not in actors:
                raise ValueError("native episode actor is outside the frozen collection plan")
            assignment, lane = actors[row["actor_id"]]
            reset_path = Path(row["native_reset_path"])
            if not reset_path.is_absolute() or file_sha256(reset_path) != row["native_reset_sha256"]:
                raise ValueError("native reset evidence changed")
            reset = json.loads(reset_path.read_bytes())
            if (reset.get("schema") != "roborender.native-episode-reset-evidence.v1"
                    or reset.get("episode_id") != row["episode_id"] or reset.get("lane") != lane["lane"]
                    or reset.get("assignment_sha256") != canonical_digest(assignment)
                    or row["simulator_seed"] != assignment["simulator_seed"]
                    or row["motion_id"] != lane["motion_id"] or row["motion_sha256"] != lane["sha256"]):
                raise ValueError("native reset/lane/seed/motion does not bind the declared actor")
        episodes[key] = row
        seeds.add(row["simulator_seed"])
        motions.add(row["motion_id"])
    if not episodes or motions != set(value["motion_support"]):
        raise ValueError("held-out collection lacks full declared motion support")
    if seeds & (set(value["training_simulator_seeds"]) | set(value["test_simulator_seeds"])):
        raise ValueError("validation simulator seeds overlap training or final test")
    if set(value["training_simulator_seeds"]) & set(value["test_simulator_seeds"]):
        raise ValueError("training and final-test simulator seeds overlap")
    return value, episodes


def export_manifest(db, collection_path, collection_sha256, output):
    collection, episodes = load_collection(collection_path, collection_sha256)
    run_id = collection["source_run_id"]
    if db.execute("SELECT 1 FROM learner_jobs WHERE run_id=? LIMIT 1", (run_id,)).fetchone():
        raise ValueError("validation collection was submitted to an optimizer")
    owner = db.execute("SELECT * FROM validation_collections WHERE run_id=?", (run_id,)).fetchone()
    completed = list(db.execute("SELECT actor_id,chunk_id FROM validation_collection_actors WHERE run_id=? ORDER BY actor_id", (run_id,)))
    if owner is None or owner["status"] != "complete":
        raise ValueError("dedicated validation-only collection is not fully complete")
    registration = json.loads(owner["registration_json"])
    from pipeline.dagger.services.validation_collection import validate
    validate(registration)
    if (canonical_digest(registration) != owner["registration_sha256"]
            or registration["round_id"] != collection["round_id"]
            or registration["initial_checkpoint_sha256"] != collection["collection_checkpoint_sha256"]
            or any(registration[key] != collection[key] for key in
                   ("policy_spec_sha256", "teacher_checkpoint_sha256", "motion_inventory_sha256", "teacher_beta"))
            or {row["actor_id"] for row in completed} != set(registration["actor_ids"])
            or {row["chunk_id"] for row in completed} != set(collection["chunk_ids"])):
        raise ValueError("validation-only owner/complete actors differ from native collection")
    scoped = collection["schema"] == SCOPED_COLLECTION_SCHEMA
    if scoped and registration["collection_plan_sha256"] != collection["collection_plan_sha256"]:
        raise ValueError("registered native collection plan differs")
    actor_chunks = {row["chunk_id"]: row["actor_id"] for row in completed}
    manifest = {"schema": SCOPED_MANIFEST_SCHEMA if scoped else MANIFEST_SCHEMA, "purpose": "validation", "round_id": collection["round_id"],
                "policy_spec_sha256": collection["policy_spec_sha256"],
                "teacher_checkpoint_sha256": collection["teacher_checkpoint_sha256"], "view_order": list(VIEWS), "chunks": []}
    evidence, used = [], set()
    for chunk_id in sorted(collection["chunk_ids"]):
        row = db.execute("SELECT * FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)).fetchone()
        provenance = db.execute("SELECT * FROM chunk_provenance WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None or row["status"] != "committed" or provenance is None:
            raise ValueError("held-out chunk lacks a complete committed ledger/provenance")
        meta = json.loads(row["metadata_json"])
        ids = json.loads(provenance["episode_ids_json"])
        keys = chunk_episode_keys(collection, meta, ids)
        if (meta["run_id"] != run_id or meta["round_id"] != collection["round_id"]
                or actor_chunks.get(chunk_id) != meta["actor_id"]
                or meta.get("checkpoint_sha256") != collection["collection_checkpoint_sha256"]
                or meta.get("last_step_id", -1) - meta.get("first_step_id", 0) + 1 != 81
                or meta["teacher_checkpoint_sha256"] != collection["teacher_checkpoint_sha256"]
                or meta["teacher_checkpoint_sha256"] != provenance["teacher_checkpoint_sha256"]
                or meta["rollout_window_id"] != provenance["rollout_window_id"]
                or meta["sealed_window_sha256"] != provenance["sealed_window_sha256"]
                or row["sha256"] != provenance["chunk_sha256"] or not ids or not set(keys) <= set(episodes)):
            raise ValueError("held-out chunk metadata differs from verified native seal/assignments")
        path = Path(row["path"])
        if not path.is_absolute() or not path.is_file() or path.stat().st_size != row["size_bytes"]:
            raise ValueError("held-out payload is missing; exporter never fabricates data")
        manifest["chunks"].append({"chunk_id": chunk_id, "path": str(path), "sha256": row["sha256"],
            "size_bytes": row["size_bytes"], "frames": 81, "source_episode_ids": keys,
            "source_window_ids": [source_window_identity(meta)],
            "observation_sources": sorted({episodes[identity]["observation_source"] for identity in keys}),
            **({"source_scope": {key: meta[key] for key in ("run_id", "round_id", "actor_id")},
                "raw_source_episode_ids": sorted(set(ids))} if scoped else {})})
        evidence.append({"chunk_id": chunk_id, "reservation": dict(row), "provenance": dict(provenance)})
        used.update(keys)
    if used != set(episodes):
        raise ValueError("declared motion/seed coverage includes episodes outside selected chunks")
    output = Path(output).resolve()
    digest = immutable_json(output, manifest)
    report = {"schema": ADMISSION_SCHEMA, "purpose": "validation", "manifest_path": str(output),
              "manifest_sha256": digest, "collection_receipt_path": str(Path(collection_path).resolve()),
              "collection_receipt_sha256": collection_sha256, "ledger_evidence": evidence,
              "collection_owner": dict(owner), "completed_actors": [dict(row) for row in completed],
              "ledger_evidence_sha256": canonical_digest({"rows": evidence}),
              "actual_episode_count": len(used), "actual_motion_support": sorted(collection["motion_support"]),
              "actual_simulator_seeds": sorted({row["simulator_seed"] for row in episodes.values()}),
              "array_payloads_read": False, "qualification": "metadata_only_actual_array_hash_verified_at_evaluation"}
    return manifest, report


def validate_export_admission(path, expected_sha, manifest_path, manifest_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("held-out admission receipt changed")
    report = json.loads(Path(path).read_bytes())
    if (report.get("schema") != ADMISSION_SCHEMA or report.get("purpose") != "validation"
            or report.get("manifest_path") != str(Path(manifest_path).resolve())
            or report.get("manifest_sha256") != manifest_sha or file_sha256(Path(manifest_path)) != manifest_sha
            or report.get("array_payloads_read") is not False):
        raise ValueError("held-out manifest/admission binding differs")
    collection, episodes = load_collection(report["collection_receipt_path"], report["collection_receipt_sha256"])
    owner = report["collection_owner"]
    registration = json.loads(owner["registration_json"])
    from pipeline.dagger.services.validation_collection import validate
    validate(registration)
    if (owner["status"] != "complete" or canonical_digest(registration) != owner["registration_sha256"]
            or registration["run_id"] != collection["source_run_id"]
            or registration["round_id"] != collection["round_id"]
            or registration["initial_checkpoint_sha256"] != collection["collection_checkpoint_sha256"]
            or any(registration[key] != collection[key] for key in
                   ("policy_spec_sha256", "teacher_checkpoint_sha256", "motion_inventory_sha256", "teacher_beta"))
            or {row["actor_id"] for row in report["completed_actors"]} != set(registration["actor_ids"])
            or {row["chunk_id"] for row in report["completed_actors"]} != set(collection["chunk_ids"])):
        raise ValueError("admitted validation owner/completion identity differs")
    if canonical_digest({"rows": report["ledger_evidence"]}) != report["ledger_evidence_sha256"]:
        raise ValueError("held-out sealed ledger evidence changed")
    manifest = json.loads(Path(manifest_path).read_bytes())
    scoped = collection["schema"] == SCOPED_COLLECTION_SCHEMA
    if scoped and (manifest["schema"] != SCOPED_MANIFEST_SCHEMA
            or registration["collection_plan_sha256"] != collection["collection_plan_sha256"]):
        raise ValueError("scoped manifest or native collection plan differs")
    actor_chunks = {row["chunk_id"]: row["actor_id"] for row in report["completed_actors"]}
    by_id = {row["chunk_id"]: row for row in report["ledger_evidence"]}
    if (set(by_id) != set(collection["chunk_ids"]) or {row["chunk_id"] for row in manifest["chunks"]} != set(by_id)
            or manifest["round_id"] != collection["round_id"]
            or manifest["teacher_checkpoint_sha256"] != collection["teacher_checkpoint_sha256"]
            or manifest["policy_spec_sha256"] != collection["policy_spec_sha256"]):
        raise ValueError("manifest population differs from admitted ledger")
    for row in manifest["chunks"]:
        record = by_id[row["chunk_id"]]
        meta = json.loads(record["reservation"]["metadata_json"])
        raw_ids = json.loads(record["provenance"]["episode_ids_json"])
        keys = chunk_episode_keys(collection, meta, raw_ids)
        if (record["reservation"]["status"] != "committed" or meta["run_id"] != collection["source_run_id"]
                or actor_chunks.get(row["chunk_id"]) != meta["actor_id"]
                or meta["round_id"] != collection["round_id"]
                or meta["checkpoint_sha256"] != collection["collection_checkpoint_sha256"]
                or meta["teacher_checkpoint_sha256"] != collection["teacher_checkpoint_sha256"]
                or meta["sealed_window_sha256"] != record["provenance"]["sealed_window_sha256"]
                or meta["rollout_window_id"] != record["provenance"]["rollout_window_id"]
                or row["sha256"] != record["provenance"]["chunk_sha256"]
                or row["sha256"] != record["reservation"]["sha256"]
                or row["path"] != record["reservation"]["path"]
                or row["size_bytes"] != record["reservation"]["size_bytes"] or row["frames"] != 81
                or row["source_episode_ids"] != keys
                or row["source_window_ids"] != [source_window_identity(meta)]
                or row["observation_sources"] != sorted({episodes[identity]["observation_source"] for identity in row["source_episode_ids"]})):
            raise ValueError("validation manifest identity differs from admitted real chunk")
        if scoped and (row.get("raw_source_episode_ids") != sorted(set(raw_ids))
                or row.get("source_scope") != {key: meta[key] for key in ("run_id", "round_id", "actor_id")}):
            raise ValueError("raw episode IDs or actor scope differ from sealed metadata")
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "collection-receipt", "output", "admission-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--collection-receipt-sha256", required=True)
    args = parser.parse_args()
    from pipeline.dagger.services.ledger_owner import open_readonly
    with open_readonly(args.ledger) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")  # One metadata snapshot; never inspect live GPU arrays.
        _, admission = export_manifest(db, args.collection_receipt, args.collection_receipt_sha256, args.output)
    immutable_json(args.admission_output, admission)
    print(json.dumps({"manifest_sha256": admission["manifest_sha256"], "admission_sha256": file_sha256(args.admission_output),
                      "array_payloads_read": False}))


if __name__ == "__main__":
    main()
