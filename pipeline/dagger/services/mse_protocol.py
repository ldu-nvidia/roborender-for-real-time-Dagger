"""Opt-in round MSE selection; legacy success-selection runs are untouched.

All controller transitions are durable and occur at checkpoint barriers. The
held-out population must be collected separately and frozen before registration.
This module never computes success or uses it to choose a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .contracts import canonical_json, file_sha256, identifier, sha256_digest


SCHEMA = "roborender.dagger-mse-round-protocol.v1"


def persist_chunk_provenance(store, reservation, sealed):
    """Preserve verified seal identities, not image/state payloads, once/window."""
    episodes = sorted(set(sealed["episode_ids"]))
    if not episodes or len(sealed["episode_ids"]) != 81:
        raise ValueError("compact provenance requires all 81 actual episode identities")
    if (hashlib.sha256(canonical_json(sealed).encode()).hexdigest() != reservation.sealed_window_sha256
            or sealed["rollout_window_id"] != reservation.rollout_window_id
            or sealed["provenance"]["teacher_checkpoint_sha256"] != reservation.teacher_checkpoint_sha256):
        raise ValueError("compact provenance does not bind the admitted seal")
    values = (reservation.chunk_id, reservation.sha256, canonical_json(episodes),
              reservation.rollout_window_id, reservation.teacher_checkpoint_sha256,
              reservation.sealed_window_sha256)
    with store.transaction() as db:
        old = db.execute("SELECT * FROM chunk_provenance WHERE chunk_id=?", (reservation.chunk_id,)).fetchone()
        if old is not None and tuple(old) != values:
            raise ValueError("compact chunk provenance changed on retry")
        db.execute("INSERT OR IGNORE INTO chunk_provenance VALUES (?,?,?,?,?,?)", values)


def get_protocol(db, run_id, round_id):
    return db.execute("SELECT * FROM mse_round_protocols WHERE run_id=? AND round_id=?",
                      (run_id, round_id)).fetchone()


def validate_protocol(value):
    from .fixed100_protocol import is_fixed100, validate_protocol as validate_fixed
    if is_fixed100(value):
        return validate_fixed(value)
    fields = {"schema", "run_id", "round_id", "initial_checkpoint_sha256",
              "validation_manifest_path", "validation_manifest_sha256",
              "interval_iterations", "max_iterations", "min_validation_epochs",
              "patience", "relative_improvement", "absolute_improvement", "final_test"}
    if set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("invalid opt-in MSE protocol fields/schema")
    identifier(value["run_id"], "run_id")
    if type(value["round_id"]) is not int or value["round_id"] not in (1, 2, 3, 4):
        raise ValueError("MSE protocol round must be 1..4")
    for key in ("initial_checkpoint_sha256", "validation_manifest_sha256"):
        sha256_digest(value[key], key)
    for key in ("interval_iterations", "max_iterations", "min_validation_epochs", "patience"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["interval_iterations"] > value["max_iterations"]:
        raise ValueError("MSE cadence exceeds explicit maximum iterations")
    for key in ("relative_improvement", "absolute_improvement"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not Path(value["validation_manifest_path"]).is_absolute():
        raise ValueError("held-out manifest path must be absolute")
    test = value["final_test"]
    if set(test) != {"scenario_manifest_sha256", "scenario_ids", "observation_mode"}:
        raise ValueError("final independent test must be pinned before training")
    sha256_digest(test["scenario_manifest_sha256"], "test manifest")
    ids = test["scenario_ids"]
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("final test requires unique scenario IDs")
    for item in ids:
        identifier(item, "test scenario")
    if test["observation_mode"] not in {"raytraced", "generated"}:
        raise ValueError("invalid final test observation mode")
    return value


def empty_provenance(round_id):
    return {"purpose": "train", "round_id": round_id, "chunk_sha256s": [],
            "source_episode_ids": [], "source_window_ids": []}


def training_provenance(db, run_id, round_id, extra_chunks=(), *, scoped_episodes=False):
    """Metadata only; complete union of every submitted round training chunk."""
    chunk_ids = set(extra_chunks)
    for row in db.execute("SELECT chunk_ids_json FROM learner_jobs WHERE run_id=? AND round_id=?",
                          (run_id, round_id)):
        chunk_ids.update(json.loads(row[0]))
    result = empty_provenance(round_id)
    hashes, episodes, windows = set(), set(), set()
    for chunk_id in sorted(chunk_ids):
        row = db.execute("SELECT * FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None or row["status"] != "committed":
            raise ValueError("training chunk is not durably committed")
        meta = json.loads(row["metadata_json"])
        if (meta["run_id"], meta["round_id"]) != (run_id, round_id):
            raise ValueError("MSE protocol forbids untracked previous-round/replay chunks")
        audit = db.execute("SELECT * FROM chunk_provenance WHERE chunk_id=?", (chunk_id,)).fetchone()
        if (audit is None or audit["chunk_sha256"] != row["sha256"]
                or audit["rollout_window_id"] != meta["rollout_window_id"]
                or audit["teacher_checkpoint_sha256"] != meta["teacher_checkpoint_sha256"]
                or audit["sealed_window_sha256"] != meta["sealed_window_sha256"]):
            raise ValueError("training chunk lacks durable compact episode provenance")
        identities = json.loads(audit["episode_ids_json"])
        if not identities:
            raise ValueError("training chunk episode provenance is empty")
        hashes.add(row["sha256"])
        from .heldout_mse import source_window_identity, source_episode_identity
        if scoped_episodes:
            episodes.update(source_episode_identity(meta, identity) for identity in identities)
        else:
            episodes.update(identities)
        windows.add(source_window_identity(meta))
    result.update(chunk_sha256s=sorted(hashes), source_episode_ids=sorted(episodes),
                  source_window_ids=sorted(windows))
    return result


def check_training_admission(db, run_id, round_id, extra_chunks=()):
    """Reject leakage before optimizer submission; never scan training arrays."""
    row = get_protocol(db, run_id, round_id)
    if row is None:
        return None
    from .heldout_mse import load_validation_manifest, SCOPED_MANIFEST_SCHEMA
    protocol = json.loads(row["protocol_json"])
    from .fixed100_protocol import is_fixed100
    if is_fixed100(protocol):
        return training_provenance(db, run_id, round_id, extra_chunks, scoped_episodes=True)
    path = Path(protocol["validation_manifest_path"])
    if file_sha256(path) != protocol["validation_manifest_sha256"]:
        raise ValueError("validation manifest digest changed")
    scoped = json.loads(path.read_bytes())["schema"] == SCOPED_MANIFEST_SCHEMA
    provenance = training_provenance(db, run_id, round_id, extra_chunks, scoped_episodes=scoped)
    manifest = load_validation_manifest(Path(protocol["validation_manifest_path"]),
                                        protocol["validation_manifest_sha256"], provenance)
    for chunk_id in extra_chunks:
        meta = json.loads(db.execute("SELECT metadata_json FROM chunk_reservations WHERE chunk_id=?",
                                     (chunk_id,)).fetchone()[0])
        if meta["teacher_checkpoint_sha256"] != manifest["teacher_checkpoint_sha256"]:
            raise ValueError("training and held-out teacher checkpoints differ")
    return provenance


def register(store, payload):
    from .heldout_mse import canonical_digest, load_validation_manifest
    if set(payload) != {"protocol", "protocol_sha256"}:
        raise ValueError("MSE registration requires exact protocol and digest")
    protocol = validate_protocol(payload["protocol"])
    digest = canonical_digest(protocol)
    if payload["protocol_sha256"] != digest:
        raise ValueError("MSE protocol digest mismatch")
    from .fixed100_protocol import is_fixed100, initial_selection
    fixed = is_fixed100(protocol)
    if not fixed:
        load_validation_manifest(Path(protocol["validation_manifest_path"]),
                                 protocol["validation_manifest_sha256"], empty_provenance(protocol["round_id"]))
    now = time.time()
    with store.transaction() as db:
        old = get_protocol(db, protocol["run_id"], protocol["round_id"])
        if old:
            if old["protocol_sha256"] != digest:
                raise ValueError("MSE round already has a different immutable protocol")
            return {"ok": True, "idempotent": True, "status": old["status"]}
        run = db.execute("SELECT * FROM runs WHERE run_id=?", (protocol["run_id"],)).fetchone()
        if fixed:
            from pipeline.dagger.rtx.code.src.dagger_rtx.observation_contract import GENERATED_ONLY_CONTRACT
            observation = db.execute('SELECT contract_json FROM run_observation_contracts WHERE run_id=?',
                                     (protocol['run_id'],)).fetchone()
            if observation is None or json.loads(observation['contract_json']) != GENERATED_ONLY_CONTRACT:
                raise ValueError('fixed100 requires the registered generated-only observation contract')
            if (run is None or len(json.loads(run['actor_ids_json'])) != 32 or run['rollout_steps'] != 81
                    or run['validation_interval_iterations'] != 100):
                raise ValueError('fixed100 requires exactly32 admitted lanes,81steps and100-update budget')
        if (run is None or run["current_round"] != protocol["round_id"] - 1
                or run["status"] not in {"created", "round_complete"}
                or run["active_checkpoint_sha256"] != protocol["initial_checkpoint_sha256"]):
            raise ValueError("MSE protocol must be registered before round start on its exact parent")
        if run["current_round"] and get_protocol(db, protocol["run_id"], run["current_round"]) is None:
            raise ValueError("cannot silently switch legacy round selection protocol")
        if run['current_round']:
            previous = get_protocol(db, protocol['run_id'], run['current_round'])
            if is_fixed100(json.loads(previous['protocol_json'])) != fixed:
                raise ValueError('cannot change the selection protocol between rounds')
        db.execute("INSERT INTO mse_round_protocols (run_id,round_id,protocol_sha256,protocol_json,selection_json,status,stop_reason,final_test_id,training_provenance_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (protocol["run_id"], protocol["round_id"], digest, canonical_json(protocol),
                    canonical_json(initial_selection(protocol)) if fixed else None,
                    'active' if fixed else "awaiting_baseline", None, None,
                    canonical_json(empty_provenance(protocol["round_id"])), now, now))
    return {"ok": True, "status": 'active' if fixed else "awaiting_baseline", "idempotent": False}


def context(store, run_id, round_id):
    with store.connection() as db:
        row = get_protocol(db, run_id, round_id)
        if row is None:
            raise ValueError("MSE protocol is not registered")
        protocol = json.loads(row["protocol_json"])
        # Status polling is cheap: the complete union is admitted atomically
        # with each learner job and cached here, not re-scanned every GET.
        provenance = json.loads(row["training_provenance_json"])
        if not provenance:
            raise ValueError("MSE protocol predates complete cached training provenance")
        result = dict(row)
        result["protocol"] = result.pop("protocol_json")
        result["protocol"] = json.loads(result["protocol"])
        result["selection"] = json.loads(result.pop("selection_json") or "null")
        result.pop("training_provenance_json")
        if row["status"] == "awaiting_baseline":
            iteration_id, digest = 0, protocol["initial_checkpoint_sha256"]
        else:
            iteration = db.execute("SELECT * FROM round_iterations WHERE run_id=? AND round_id=? ORDER BY iteration_id DESC LIMIT 1",
                                   (run_id, round_id)).fetchone()
            iteration_id = iteration["iteration_id"] if iteration else 0
            digest = (iteration["candidate_checkpoint_sha256"] or iteration["input_checkpoint_sha256"]) if iteration else protocol["initial_checkpoint_sha256"]
        checkpoint = db.execute("SELECT path FROM checkpoints WHERE sha256=?", (digest,)).fetchone()
        if checkpoint is None:
            raise ValueError("MSE target checkpoint is not registered")
        result.update(iteration_id=iteration_id, checkpoint_sha256=digest,
                      checkpoint_path=checkpoint["path"], training_provenance=provenance)
        if result["selection"] is not None:
            selected = db.execute("SELECT path FROM checkpoints WHERE sha256=?",
                                  (result["selection"]["best_checkpoint_sha256"],)).fetchone()
            if selected is None:
                raise ValueError("MSE-selected checkpoint is not registered")
            result["selected_checkpoint_path"] = selected["path"]
        return {"ok": True, "mse_round": result}


def record(store, payload):
    from pipeline.bc_training.train_bc_h200 import EarlyStopping
    from .heldout_mse import RoundMSESelection
    if set(payload) != {"run_id", "round_id", "iteration_id", "result"}:
        raise ValueError("invalid MSE receipt fields")
    if not isinstance(payload["result"], dict) or "checkpoint_sha256" not in payload["result"]:
        raise ValueError("MSE receipt result is incomplete")
    run_id, round_id, iteration_id = payload["run_id"], payload["round_id"], payload["iteration_id"]
    now = time.time()
    with store.transaction() as db:
        row = get_protocol(db, run_id, round_id)
        if row is None:
            raise ValueError("MSE protocol is not registered")
        protocol = json.loads(row["protocol_json"])
        from .fixed100_protocol import is_fixed100
        if is_fixed100(protocol):
            raise ValueError('fixed100 forbids validation receipts; selection is atomic with update100')
        selection = RoundMSESelection.restore(json.loads(row["selection_json"])) if row["selection_json"] else RoundMSESelection(
            round_id, protocol["validation_manifest_sha256"], EarlyStopping(
                min_epochs=protocol["min_validation_epochs"], patience=protocol["patience"],
                relative_improvement=protocol["relative_improvement"], absolute_improvement=protocol["absolute_improvement"]))
        prior = next((item for item in selection.history if item["iteration"] == iteration_id), None)
        if prior is not None:
            selection.record(iteration_id, payload["result"])
            return {"ok": True, "idempotent": True, "status": row["status"]}
        check_training_admission(db, run_id, round_id)
        manifest = json.loads(Path(protocol["validation_manifest_path"]).read_bytes())
        if (payload["result"].get("frames") != sum(item["frames"] for item in manifest["chunks"])
                or payload["result"].get("chunk_count") != len(manifest["chunks"])):
            raise ValueError("MSE receipt population differs from the fixed manifest")
        if iteration_id == 0:
            if row["status"] != "awaiting_baseline":
                raise ValueError("unexpected MSE baseline")
            expected = protocol["initial_checkpoint_sha256"]
        else:
            iteration = db.execute("SELECT * FROM round_iterations WHERE run_id=? AND round_id=? ORDER BY iteration_id DESC LIMIT 1", (run_id, round_id)).fetchone()
            if (iteration is None or iteration["iteration_id"] != iteration_id
                    or iteration["status"] != "awaiting_mse_validation"
                    or iteration_id > protocol["max_iterations"]
                    or (iteration_id % protocol["interval_iterations"] and iteration_id != protocol["max_iterations"])):
                raise ValueError("MSE receipt is not for the current configured validation barrier")
            expected = iteration["candidate_checkpoint_sha256"]
        if payload["result"]["checkpoint_sha256"] != expected:
            raise ValueError("MSE receipt checkpoint differs from the pinned candidate")
        decision = selection.record(iteration_id, payload["result"])
        final = iteration_id > 0 and (decision["saturated"] or iteration_id == protocol["max_iterations"])
        status = "selected_pending_test" if final else "active"
        reason = ("validation_mse_plateau" if decision["saturated"] else "maximum_iterations") if final else None
        db.execute("UPDATE mse_round_protocols SET selection_json=?,status=?,stop_reason=?,updated_at=? WHERE run_id=? AND round_id=?",
                   (canonical_json(selection.state_dict()), status, reason, now, run_id, round_id))
        if iteration_id > 0:
            if final:
                db.execute("UPDATE round_iterations SET status='selected_pending_test',updated_at=? WHERE run_id=? AND round_id=? AND iteration_id=?", (now, run_id, round_id, iteration_id))
                db.execute("UPDATE runs SET status='selected_pending_test',updated_at=? WHERE run_id=?", (now, run_id))
            else:
                db.execute("UPDATE round_iterations SET status='advanced',updated_at=? WHERE run_id=? AND round_id=? AND iteration_id=?", (now, run_id, round_id, iteration_id))
                db.execute("INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)", (run_id, round_id, iteration_id + 1, expected, None, None, None, "collecting", now, now))
                db.execute("UPDATE runs SET active_checkpoint_sha256=?,status='collecting',updated_at=? WHERE run_id=?", (expected, now, run_id))
    return {"ok": True, "idempotent": False, "status": status,
            "best_checkpoint_sha256": selection.best_checkpoint_sha256, "stop_reason": reason}


def admit_final_test(db, payload):
    """Reject any pre-selection success query for an MSE-selected round."""
    row = get_protocol(db, payload["run_id"], payload["round_id"])
    if row is None:
        return
    protocol = json.loads(row["protocol_json"])
    selected = json.loads(row["selection_json"] or "null")
    if (payload["purpose"] != "test" or row["status"] not in {"selected_pending_test", "complete"}
            or selected is None or payload["checkpoint_sha256"] != selected["best_checkpoint_sha256"]
            or any(payload[key] != protocol["final_test"][key] for key in ("scenario_manifest_sha256", "scenario_ids", "observation_mode"))):
        raise ValueError("success test is allowed only after immutable MSE selection on the pinned test panel")
    if row["final_test_id"] and row["final_test_id"] != payload["evaluation_id"]:
        raise ValueError("MSE round already has its one final test")
    db.execute("UPDATE mse_round_protocols SET final_test_id=?,updated_at=? WHERE run_id=? AND round_id=?",
               (payload["evaluation_id"], time.time(), payload["run_id"], payload["round_id"]))


def finish_test(store, payload):
    if set(payload) != {"run_id", "round_id", "evaluation_id"}:
        raise ValueError("invalid final-test receipt fields")
    now = time.time()
    with store.transaction() as db:
        row = get_protocol(db, payload["run_id"], payload["round_id"])
        if row is None or row["final_test_id"] != payload["evaluation_id"]:
            raise ValueError("final test is not bound to the selected round")
        if row["status"] == "complete":
            return {"ok": True, "idempotent": True, "status": "round_complete"}
        evaluation = db.execute("SELECT * FROM success_evaluations WHERE evaluation_id=?", (payload["evaluation_id"],)).fetchone()
        if evaluation is None or evaluation["status"] != "complete":
            raise ValueError("independent final success test is not complete")
        count = db.execute("SELECT COUNT(*) FROM success_evaluation_outcomes WHERE evaluation_id=?", (payload["evaluation_id"],)).fetchone()[0]
        if count != len(json.loads(evaluation["scenario_ids_json"])):
            raise ValueError("independent final success outcomes are incomplete")
        best = json.loads(row["selection_json"])["best_checkpoint_sha256"]
        if evaluation["purpose"] != "test" or evaluation["checkpoint_sha256"] != best:
            raise ValueError("final success test did not use the MSE-selected checkpoint")
        # No success value is read here: even 0% cannot alter the selection or
        # cause another tuning/training attempt on the same test population.
        db.execute("UPDATE mse_round_protocols SET status='complete',updated_at=? WHERE run_id=? AND round_id=?", (now, payload["run_id"], payload["round_id"]))
        db.execute("UPDATE round_iterations SET status='complete',updated_at=? WHERE run_id=? AND round_id=? AND status='selected_pending_test'", (now, payload["run_id"], payload["round_id"]))
        db.execute("UPDATE runs SET active_checkpoint_sha256=?,status='round_complete',updated_at=? WHERE run_id=?", (best, now, payload["run_id"]))
        db.execute("INSERT INTO promotions VALUES (?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET round_id=excluded.round_id,checkpoint_sha256=excluded.checkpoint_sha256,updated_at=excluded.updated_at", (payload["run_id"], payload["round_id"], best, now))
    return {"ok": True, "idempotent": False, "status": "round_complete", "best_checkpoint_sha256": best}
