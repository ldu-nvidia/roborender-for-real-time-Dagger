"""CPU/loopback contracts for genuine, optimizer-forbidden held-out owners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from pipeline.dagger.services import validation_collection as vc
from pipeline.dagger.services.contracts import canonical_json, file_sha256
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.services.client import ServiceError
from test.pipeline.test_dagger_h200_services import running_service
from pipeline.dagger.rtx.code.src.dagger_rtx.heldout_collection_driver import HeldoutCollectionDriver


def fixture(tmp_path, rid=1):
    store = StateStore(tmp_path)
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"CPU fixture only")
    sha = file_sha256(checkpoint)
    with store.transaction() as db:
        db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (sha, checkpoint.stat().st_size, "{}", str(checkpoint), 0))
    run = f"heldout-round-{rid}"
    value = {"schema": vc.SCHEMA, "purpose": "validation", "run_id": run, "round_id": rid,
        "actor_ids": [f"{run}-g0-lane-{i:03d}" for i in range(2)], "initial_checkpoint_sha256": sha,
        "teacher_checkpoint_sha256": "b" * 64, "policy_spec_sha256": "c" * 64,
        "motion_inventory_sha256": "d" * 64, "teacher_beta": {1: .75, 2: .5, 3: .25, 4: 0.}[rid],
        "collection_plan_sha256": "e" * 64}
    payload = {"registration": value, "registration_sha256": hashlib.sha256(canonical_json(value).encode()).hexdigest()}
    return store, value, payload


def seal(store, value, actor):
    chunk = f"d{value['round_id']}-{actor}-window-000000"
    metadata = {"run_id": value["run_id"], "round_id": value["round_id"], "actor_id": actor,
        "rollout_window_id": "window-000000", "checkpoint_sha256": value["initial_checkpoint_sha256"],
        "teacher_checkpoint_sha256": value["teacher_checkpoint_sha256"], "first_step_id": 0, "last_step_id": 80}
    with store.transaction() as db:
        db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)", (chunk, "f" * 64, 1, canonical_json(metadata), "committed", "unused", 0, 0))
        db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)", (chunk, "f" * 64, '["actual-episode"]', "window-000000", "b" * 64, "a" * 64))
    return chunk


@pytest.mark.parametrize("rid", [1, 2, 3, 4])
def test_direct_target_round_is_collection_only_and_exact_retry(tmp_path, rid):
    store, value, payload = fixture(tmp_path, rid)
    assert not vc.register(store, payload)["idempotent"]
    assert vc.register(store, payload)["idempotent"]
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM round_iterations").fetchone()[0] == 1
        assert db.execute("SELECT current_round FROM runs").fetchone()[0] == rid
        with pytest.raises(ValueError, match="cannot train"):
            vc.forbid_learning(db, value["run_id"])
    for i, actor in enumerate(value["actor_ids"]):
        chunk = seal(store, value, actor)
        body = {"run_id": value["run_id"], "actor_id": actor, "chunk_id": chunk,
                "registration_sha256": payload["registration_sha256"]}
        first = vc.finish_actor(store, body)
        assert vc.finish_actor(store, body) == first
        assert first["arrived"] == i + 1 and not first["optimizer_authorized"]
    assert vc.context(store, value["run_id"])["collection"]["status"] == "complete"
    with store.connection() as db:
        with pytest.raises(ValueError, match="forbidden"):
            vc.forbid_learning(db, "other-train-run", [chunk])
        assert db.execute("SELECT COUNT(*) FROM learner_jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM iteration_barriers").fetchone()[0] == 0


def test_real_http_cannot_start_training_round_and_rejects_changed_identity(tmp_path):
    store, value, payload = fixture(tmp_path)
    with running_service("coordinator", tmp_path) as client:
        assert client.post("/v1/validation-collections/register", payload)["ok"]
        assert client.get("/v1/validation-collections/" + value["run_id"])["collection"]["optimizer_authorized"] is False
        with pytest.raises(ServiceError, match="cannot train"):
            client.post("/v1/rounds/start", {"run_id": value["run_id"], "round_id": 1,
                "beta": .75, "checkpoint_sha256": value["initial_checkpoint_sha256"]})
        changed = {**value, "collection_plan_sha256": "a" * 64}
        with pytest.raises(ServiceError, match="changed"):
            client.post("/v1/validation-collections/register", {"registration": changed,
                "registration_sha256": hashlib.sha256(canonical_json(changed).encode()).hexdigest()})


def test_shared_window_checkpoint_cache_never_admits_unlisted_heldout_lane(tmp_path):
    from pipeline.dagger.services.render_student_server import RenderStudentHTTPServer, NotReadyError
    store, value, payload = fixture(tmp_path)
    vc.register(store, payload)
    server = RenderStudentHTTPServer.__new__(RenderStudentHTTPServer)
    server.store, server.window_cache_lock, server.window_checkpoints = store, threading.Lock(), {}
    keys = [SimpleNamespace(run_id=value["run_id"], round_id=1, actor_id=actor,
                            rollout_window_id="window-000000") for actor in value["actor_ids"]]
    assert server.batch_window_checkpoint(keys) == value["initial_checkpoint_sha256"]
    bad = SimpleNamespace(**{**vars(keys[0]), "actor_id": value["run_id"] + "-not-registered"})
    with pytest.raises(NotReadyError, match="undeclared"):
        server.batch_window_checkpoint([keys[0], bad])
    with pytest.raises(NotReadyError, match="undeclared"):
        server.window_checkpoint(bad)


def test_native_driver_returns_after_one_window_without_learner_barrier(tmp_path):
    store, value, payload = fixture(tmp_path)
    vc.register(store, payload)
    calls = []
    def request(method, path, body=None):
        calls.append((method, path))
        return vc.context(store, value["run_id"]) if method == "GET" else vc.finish_actor(store, body)
    def run_window(index):
        assert index == 0
        for actor in value["actor_ids"]:
            seal(store, value, actor)
        return {"transition_batches": 81, "physical_transition_rows": 162}
    runner = SimpleNamespace(num_envs=2, actor_prefix=value["run_id"] + "-g0", run_window=run_window)
    spec = {**payload, "num_envs": 2, "actor_prefix": runner.actor_prefix}
    state = HeldoutCollectionDriver(runner, request, tmp_path / "actor", spec, "c" * 64).run()
    assert state["phase"] == "COMPLETE" and state["optimizer_steps"] == 0
    assert all("barriers" not in path and "jobs" not in path for _, path in calls)
    with pytest.raises(RuntimeError, match="restart forbidden"):
        HeldoutCollectionDriver(runner, request, tmp_path / "actor", spec, "c" * 64)


def test_partial_or_wrong_parent_seal_rejected(tmp_path):
    store, value, payload = fixture(tmp_path)
    vc.register(store, payload)
    actor = value["actor_ids"][0]
    chunk = seal(store, value, actor)
    with store.transaction() as db:
        row = json.loads(db.execute("SELECT metadata_json FROM chunk_reservations").fetchone()[0])
        row["checkpoint_sha256"] = "a" * 64
        db.execute("UPDATE chunk_reservations SET metadata_json=?", (canonical_json(row),))
    with pytest.raises(ValueError, match="parent"):
        vc.finish_actor(store, {"run_id": value["run_id"], "actor_id": actor, "chunk_id": chunk,
                               "registration_sha256": payload["registration_sha256"]})
