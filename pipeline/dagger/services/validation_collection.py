"""Independent, collect-only held-out owners; never valid learner inputs."""
from __future__ import annotations

import hashlib
import json
import time

from .contracts import CANONICAL_BETA, canonical_json, identifier, sha256_digest

SCHEMA = "roborender.dagger-validation-collection.v1"


def validate(value):
    fields = {"schema", "purpose", "run_id", "round_id", "actor_ids", "initial_checkpoint_sha256",
              "teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256",
              "teacher_beta", "collection_plan_sha256"}
    if set(value) != fields or value["schema"] != SCHEMA or value["purpose"] != "validation":
        raise ValueError("explicit validation-only collection identity is required")
    identifier(value["run_id"], "collection run")
    if not value["run_id"].startswith("heldout-"):
        raise ValueError("validation collections need a visibly separate heldout- namespace")
    if type(value["round_id"]) is not int or value["round_id"] not in CANONICAL_BETA or value["teacher_beta"] != CANONICAL_BETA[value["round_id"]]:
        raise ValueError("held-out behavior beta differs from target round")
    actors = value["actor_ids"]
    if not isinstance(actors, list) or not actors or actors != sorted(set(actors)):
        raise ValueError("collection actors must be a unique sorted list")
    for actor in actors:
        identifier(actor, "actor")
        if not actor.startswith(value["run_id"] + "-"):
            raise ValueError("held-out actor is outside its isolated namespace")
    for name in ("initial_checkpoint_sha256", "teacher_checkpoint_sha256", "policy_spec_sha256", "motion_inventory_sha256", "collection_plan_sha256"):
        sha256_digest(value[name], name)
    return value


def register(store, payload):
    if set(payload) != {"registration", "registration_sha256"}:
        raise ValueError("exact validation collection registration and digest required")
    value = validate(payload["registration"])
    raw = canonical_json(value)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    if payload["registration_sha256"] != digest:
        raise ValueError("collection registration digest changed")
    now = time.time()
    with store.transaction() as db:
        old = db.execute("SELECT * FROM validation_collections WHERE run_id=?", (value["run_id"],)).fetchone()
        if old:
            if old["registration_sha256"] != digest:
                raise ValueError("validation collection identity changed on retry")
            return {"ok": True, "idempotent": True, "status": old["status"]}
        if db.execute("SELECT 1 FROM runs WHERE run_id=?", (value["run_id"],)).fetchone():
            raise ValueError("existing training run cannot become held-out validation")
        if db.execute("SELECT 1 FROM checkpoints WHERE sha256=?", (value["initial_checkpoint_sha256"],)).fetchone() is None:
            raise ValueError("held-out parent checkpoint is not registered")
        db.execute("INSERT INTO validation_collections VALUES (?,?,?,?,?,?)", (value["run_id"], digest, raw, "collecting", now, now))
        # This is explicitly a collection owner at a target round, not fake
        # prior training/optimizer history. The learner rejects this namespace.
        db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (value["run_id"], canonical_json(value["actor_ids"]),
            81, digest, value["initial_checkpoint_sha256"], value["initial_checkpoint_sha256"], value["round_id"],
            value["teacher_beta"], "collecting", 1, now, now))
        db.execute("INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)", (value["run_id"], value["round_id"], 1,
            value["initial_checkpoint_sha256"], None, None, None, "collecting", now, now))
    return {"ok": True, "idempotent": False, "status": "collecting"}


def context(store, run_id):
    with store.connection() as db:
        row = db.execute("SELECT * FROM validation_collections WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("validation collection is not registered")
        actors = [dict(item) for item in db.execute("SELECT actor_id,chunk_id FROM validation_collection_actors WHERE run_id=? ORDER BY actor_id", (run_id,))]
        return {"ok": True, "collection": {"registration": json.loads(row["registration_json"]),
            "registration_sha256": row["registration_sha256"], "status": row["status"], "actors": actors,
            "optimizer_authorized": False}}


def finish_actor(store, payload):
    if set(payload) != {"run_id", "actor_id", "chunk_id", "registration_sha256"}:
        raise ValueError("exact validation actor completion fields required")
    raw = canonical_json(payload)
    with store.transaction() as db:
        collection = db.execute("SELECT * FROM validation_collections WHERE run_id=?", (payload["run_id"],)).fetchone()
        if collection is None or collection["registration_sha256"] != payload["registration_sha256"]:
            raise ValueError("validation actor completion has no matching owner")
        value = json.loads(collection["registration_json"])
        if payload["actor_id"] not in value["actor_ids"]:
            raise ValueError("actor is outside validation collection")
        old = db.execute("SELECT receipt_json FROM validation_collection_actors WHERE run_id=? AND actor_id=?",
                         (payload["run_id"], payload["actor_id"])).fetchone()
        if old and old[0] != raw:
            raise ValueError("validation actor completion changed on retry")
        row = db.execute("SELECT * FROM chunk_reservations WHERE chunk_id=?", (payload["chunk_id"],)).fetchone()
        if row is None or row["status"] != "committed":
            raise ValueError("complete sealed held-out chunk is required")
        meta = json.loads(row["metadata_json"])
        if (meta["run_id"] != value["run_id"] or meta["round_id"] != value["round_id"]
                or meta["actor_id"] != payload["actor_id"] or meta["rollout_window_id"] != "window-000000"
                or meta["checkpoint_sha256"] != value["initial_checkpoint_sha256"]
                or meta["teacher_checkpoint_sha256"] != value["teacher_checkpoint_sha256"]
                or meta["last_step_id"] - meta["first_step_id"] + 1 != 81
                or db.execute("SELECT 1 FROM chunk_provenance WHERE chunk_id=?", (payload["chunk_id"],)).fetchone() is None):
            raise ValueError("validation chunk parent/teacher/identity/81-step provenance differs")
        now = time.time()
        db.execute("INSERT OR IGNORE INTO validation_collection_actors VALUES (?,?,?,?,?)",
                   (value["run_id"], payload["actor_id"], payload["chunk_id"], raw, now))
        count = db.execute("SELECT COUNT(*) FROM validation_collection_actors WHERE run_id=?", (value["run_id"],)).fetchone()[0]
        if count == len(value["actor_ids"]):
            db.execute("UPDATE validation_collections SET status='complete',updated_at=? WHERE run_id=?", (now, value["run_id"]))
            db.execute("UPDATE runs SET status='validation_collected',updated_at=? WHERE run_id=?", (now, value["run_id"]))
            db.execute("UPDATE round_iterations SET status='validation_collected',updated_at=? WHERE run_id=?", (now, value["run_id"]))
        return {"ok": True, "actor_id": payload["actor_id"], "chunk_id": payload["chunk_id"],
                "arrived": count, "expected": len(value["actor_ids"]), "optimizer_authorized": False}


def forbid_learning(db, run_id, chunk_ids=()):
    if db.execute("SELECT 1 FROM validation_collections WHERE run_id=?", (run_id,)).fetchone():
        raise ValueError("validation-only collection cannot train, advance or become a training run")
    for chunk_id in chunk_ids:
        row = db.execute("SELECT metadata_json FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row and db.execute("SELECT 1 FROM validation_collections WHERE run_id=?", (json.loads(row[0])["run_id"],)).fetchone():
            raise ValueError("held-out validation chunks are forbidden in optimization/replay")
