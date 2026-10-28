"""CPU-only contracts for pinned-population cohort-boundary imports."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest
import torch

from pipeline.bc_training.handoff_common import binding, publish
from pipeline.dagger.coverage_production import digest
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    LEO_MSE_PRODUCTION_SCHEMA, LEO_MSE_SCALED_SCHEMA,
)
from pipeline.dagger.services import mse_protocol
from pipeline.dagger.services.render_student import content_addressed_checkpoint
from pipeline.dagger.services.store import StateStore
from pipeline.deployment import cohort_boundary_recovery as recovery


RUN = "dagger-leo-prefix-recovery"
SOURCE = "a" * 64


def checkpoint(root: Path, index: int):
    path = root / f"checkpoint-{index:02d}.pt"
    step = 2560 + index
    parameter = torch.tensor([float(index + 1)], dtype=torch.float32)
    torch.save({
        "schema": "roborender.bc_student_checkpoint.v3",
        "global_step": step,
        "policy_spec": {"fixture": True},
        "policy_state_dict": {"vision_module.fixture.weight": parameter},
        "optimizer_state_dict": {
            "param_groups": [{
                "params": [0], "weight_decay": 0.0, "amsgrad": False,
                "betas": (0.9, 0.999), "eps": 1e-8, "maximize": False,
                "lr": 0.0002,
            }],
            "state": {0: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }},
        },
        "lr_scheduler_state_dict": None,
        "fixture_index": index,
    }, path)
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest(), len(raw)


def metadata(root: Path, name: str, value=None):
    return publish(root / name, {} if value is None else value)


def make_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    completed_updates: int = 26,
    incomplete_owner_steps: int = 75,
    first_owner_steps: int = 81,
    environments_per_actor: int = 2,
):
    assert completed_updates > 0 and completed_updates % 2 == 0
    assert 0 <= incomplete_owner_steps < 81
    assert 0 <= first_owner_steps <= 81
    completed_cohorts = completed_updates // 2
    failed_iteration = completed_updates + 1
    root = tmp_path / "source"
    store = StateStore(root)
    actors = [f"{RUN}-g{gpu}-lane-{lane:03d}" for gpu in (0, 1)
              for lane in range(environments_per_actor)]
    checkpoints = [checkpoint(tmp_path, index) for index in range(failed_iteration)]
    old_config = {
        "schema": LEO_MSE_PRODUCTION_SCHEMA if environments_per_actor == 2 else LEO_MSE_SCALED_SCHEMA,
        "run_id": RUN,
        "rtx_actors": [{"physical_gpu_index": gpu, "concurrent_environments": environments_per_actor}
                       for gpu in (0, 1)],
        "student": {"initial_checkpoint_sha256": checkpoints[0][1], "learning_rate": 0.0002},
    }
    plan = {"source_sha256": SOURCE}
    validation_interval = 2 * ((295 + len(actors) - 1) // len(actors))
    manifest = metadata(tmp_path, "heldout-manifest.json", {"chunks": []})

    with store.transaction() as db:
        for path, sha, size in checkpoints:
            db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (sha, size, "{}", str(path), 1.0))
        db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (RUN, json.dumps(actors), 81, digest(old_config), checkpoints[0][1], checkpoints[-1][1],
             1, .75, "collecting", validation_interval, 1.0, 2.0),
        )
        for iteration in range(1, failed_iteration + 1):
            db.execute(
                "INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (RUN, 1, iteration, checkpoints[iteration - 1][1],
                 checkpoints[iteration][1] if iteration <= completed_updates else None,
                 None, None, "advanced" if iteration <= completed_updates else "collecting", 1.0, 2.0),
            )

        for iteration in range(1, completed_updates + 1):
            chunk_ids = []
            window = f"window-{iteration - 1:06d}"
            for actor in actors:
                chunk_id = f"d1-{actor}-{window}"
                chunk_path = tmp_path / f"{chunk_id}.npz"
                chunk_path.write_bytes(chunk_id.encode())
                chunk_sha = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
                meta = {"run_id": RUN, "round_id": 1, "actor_id": actor,
                        "rollout_window_id": window, "teacher_checkpoint_sha256": "b" * 64,
                        "sealed_window_sha256": hashlib.sha256((chunk_id + "-seal").encode()).hexdigest()}
                db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",
                           (chunk_id, chunk_sha, chunk_path.stat().st_size, json.dumps(meta, sort_keys=True),
                            "committed", str(chunk_path), 1.0, 2.0))
                db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)",
                           (chunk_id, chunk_sha, json.dumps([f"episode-{iteration}-{actor}"]), window,
                            "b" * 64, meta["sealed_window_sha256"]))
                db.execute("INSERT INTO iteration_barriers VALUES (?,?,?,?,?,?,?)",
                           (RUN, 1, iteration, actor, checkpoints[iteration - 1][1],
                            json.dumps([chunk_id]), 1.0))
                chunk_ids.append(chunk_id)
                for step in range(81):
                    key = f"prepare-{iteration}-{actor}-{step}"
                    db.execute("INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (key, "c" * 64, RUN, 1, actor, window, step, f"episode-{iteration}-{actor}",
                                step, "d" * 64, "receipt", "{}", "{}", "/artifact", checkpoints[iteration - 1][1],
                                "committed", 1.0, 2.0))
                    db.execute("INSERT INTO step_commits VALUES (?,?,?,?,?,?,?,?)",
                               (f"commit-{iteration}-{actor}-{step}", "e" * 64, key, "{}", None, "{}", 1.0, 2.0))
            result = {"iteration_id": iteration, "parent_checkpoint_sha256": checkpoints[iteration - 1][1],
                      "output_checkpoint_sha256": checkpoints[iteration][1], "optimizer_state_preserved": True,
                      "replay_manifest_sha256": None, "parent_global_step": 2559 + iteration,
                      "global_step": 2560 + iteration}
            job = f"{RUN}-dagger-1-iteration-{iteration:04d}"
            db.execute("INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (job, RUN, 1, checkpoints[iteration - 1][1], json.dumps(chunk_ids), "succeeded", "{}",
                        json.dumps(result), "/log", None, 1.0, 2.0))
            db.execute("INSERT INTO learner_job_iterations VALUES (?,?,?,?)", (job, RUN, 1, iteration))

        tail_window = f"window-{completed_updates:06d}"
        for actor in actors:
            count = first_owner_steps if "-g0-" in actor else incomplete_owner_steps
            for step in range(count):
                key = f"tail-prepare-{actor}-{step}"
                db.execute("INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (key, "f" * 64, RUN, 1, actor, tail_window, step, f"tail-{actor}", step,
                            "1" * 64, "receipt", "{}", "{}", "/artifact", checkpoints[-1][1],
                            "committed", 1.0, 2.0))
                db.execute("INSERT INTO step_commits VALUES (?,?,?,?,?,?,?,?)",
                           (f"tail-commit-{actor}-{step}", "2" * 64, key, "{}", None, "{}", 1.0, 2.0))
            if count == 81:
                chunk_id = f"d1-{actor}-{tail_window}"
                chunk_path = tmp_path / f"{chunk_id}.npz"
                chunk_path.write_bytes(chunk_id.encode())
                chunk_sha = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
                meta = {"run_id": RUN, "round_id": 1, "actor_id": actor,
                        "rollout_window_id": tail_window, "teacher_checkpoint_sha256": "b" * 64,
                        "sealed_window_sha256": hashlib.sha256((chunk_id + "-seal").encode()).hexdigest()}
                db.execute("INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",
                           (chunk_id, chunk_sha, chunk_path.stat().st_size, json.dumps(meta, sort_keys=True),
                            "committed", str(chunk_path), 1.0, 2.0))
                db.execute("INSERT INTO chunk_provenance VALUES (?,?,?,?,?,?)",
                           (chunk_id, chunk_sha, json.dumps([f"tail-{actor}"]), tail_window,
                            "b" * 64, meta["sealed_window_sha256"]))
                db.execute("INSERT INTO iteration_barriers VALUES (?,?,?,?,?,?,?)",
                           (RUN, 1, failed_iteration, actor, checkpoints[-1][1], json.dumps([chunk_id]), 1.0))

        protocol = {"schema": mse_protocol.SCHEMA, "run_id": RUN, "round_id": 1,
                    "initial_checkpoint_sha256": checkpoints[0][1], "validation_manifest_path": manifest["path"],
                    "validation_manifest_sha256": manifest["raw_sha256"], "interval_iterations": validation_interval,
                    "max_iterations": 10 * validation_interval, "min_validation_epochs": 2, "patience": 5,
                    "relative_improvement": .005, "absolute_improvement": 1e-6,
                    "final_test": {"scenario_manifest_sha256": "3" * 64,
                                   "scenario_ids": ["case-0"], "observation_mode": "raytraced"}}
        provenance = mse_protocol.training_provenance(db, RUN, 1)
        selection = {"history": [{"iteration": index}
                                 for index in range(0, completed_updates + 1, validation_interval)],
                     "best_checkpoint_sha256": checkpoints[0][1]}
        db.execute("INSERT INTO mse_round_protocols VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (RUN, 1, "4" * 64, json.dumps(protocol), json.dumps(selection), "active", None, None,
                    json.dumps(provenance, sort_keys=True), 1.0, 2.0))

    references = [
        metadata(tmp_path, f"completion-{index:02d}.json")
        for index in range(completed_cohorts)
    ]
    closed_database = recovery.stable_file_ref(store.path)
    workers = [{"physical_gpu": gpu, "identity_checked": True, "unit_state": "inactive",
                "container_absent": True} for gpu in (0, 1)]
    h200_attestation = metadata(tmp_path, "h200-closed.json", {
        "label": "h200-services", "result": {"state": "STOPPED"}, "allocation_retained": True,
        "compute_release_requested": False, "closed_database": closed_database,
    })
    native_attestation = metadata(tmp_path, "native-terminal.json", {
        "schema": "roborender.native-recovery-terminal-attestation.v1", "workers": workers,
        "native_replay_allowed": False, "compute_release_requested": False,
    })
    reconciliation = {
        "schema": recovery.SCHEMA, "run_id": RUN, "round_id": 1, "source_sha256": SOURCE,
        "completed_cohorts": completed_cohorts, "completed_updates": completed_updates,
        "failed_iteration": failed_iteration, "failed_cohort_index": completed_cohorts,
        "tail_disposition": recovery.TAIL_DISPOSITION,
        "native_replay_allowed": False, "fresh_native_runtime_required": True,
        "old_namespace_preserved": True, "allocation_retained": True, "compute_release_requested": False,
        "closed_database": closed_database,
        "round_config": metadata(tmp_path, "old-config.json", old_config),
        "coverage_plan": metadata(tmp_path, "coverage-plan.json"), "heldout_manifest": manifest,
        "heldout_admission": metadata(tmp_path, "heldout-admission.json"),
        "failed_cohort_intent": metadata(tmp_path, "failed-intent.json"),
        "completed_cohort_receipts": references,
        "controller": {"state": "FAILED", "identity_checked": True},
        "h200_services": {"state": "STOPPED", "identity_checked": True,
                          "allocation_retained": True, "compute_released": False},
        "h200_services_attestation": h200_attestation,
        "native_workers": workers, "native_terminal_attestation": native_attestation,
    }
    monkeypatch.setattr(recovery, "_validate_artifacts",
                        lambda value, rows: (
                            old_config,
                            plan,
                            checkpoints[-1][1],
                            2560 + completed_updates,
                        ))
    return reconciliation, store.path


def test_exact_prefix_import_preserves_source_and_excludes_partial_tail(tmp_path, monkeypatch):
    reconciliation, source = make_fixture(tmp_path, monkeypatch)
    before = source.read_bytes()
    preview = recovery.inspect_prefix(reconciliation)["report"]
    assert preview["completed_updates"] == 26 and preview["completed_cohorts"] == 13
    assert preview["excluded_tail"]["actor_committed_steps"] == {
        f"{RUN}-g0-lane-000": 81, f"{RUN}-g0-lane-001": 81,
        f"{RUN}-g1-lane-000": 75, f"{RUN}-g1-lane-001": 75,
    }
    destination = tmp_path / "imported"
    result = recovery.import_prefix(reconciliation, destination)
    assert source.read_bytes() == before and result["source_database_unchanged"] is True
    assert result["compute_release_requested"] is False and result["native_work_started"] is False
    materialized = result["checkpoint_materialization"]
    assert materialized["checkpoint_count"] == 27
    assert (materialized["hardlinked_count"] + materialized["reflinked_count"]
            + materialized["copied_count"]) == 27
    assert materialized["all_hashes_verified"] is True
    assert materialized["all_paths_rewritten"] is True
    assert materialized["active_checkpoint"]["sha256"] == result["parent_checkpoint_sha256"]
    assert materialized["active_checkpoint"]["global_step"] == 2586
    assert materialized["active_checkpoint"]["optimizer_state_load_verified"] is True
    with sqlite3.connect(destination / "state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM learner_jobs").fetchone()[0] == 26
        assert db.execute("SELECT COUNT(*) FROM iteration_barriers").fetchone()[0] == 104
        assert db.execute("SELECT COUNT(*) FROM step_prepares").fetchone()[0] == 26 * 4 * 81
        assert db.execute("SELECT status,candidate_checkpoint_sha256 FROM round_iterations WHERE iteration_id=27").fetchone() == ("collecting", None)
        checkpoints = db.execute("SELECT sha256,size_bytes,path FROM checkpoints ORDER BY sha256").fetchall()
    assert len(checkpoints) == 27
    for sha, size, path in checkpoints:
        resolved = content_addressed_checkpoint(destination, sha)
        assert resolved == destination / "checkpoints" / f"{sha}.pt"
        assert path == str(resolved) and resolved.stat().st_size == size
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == sha
        assert resolved.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="fresh destination"):
        recovery.import_prefix(reconciliation, destination)


def test_iteration31_prefix_import_preserves_30_updates_and_excludes_zero_step_peer(
    tmp_path, monkeypatch
):
    reconciliation, source = make_fixture(
        tmp_path,
        monkeypatch,
        completed_updates=30,
        incomplete_owner_steps=0,
    )
    before = source.read_bytes()
    preview = recovery.inspect_prefix(reconciliation)["report"]
    assert preview["completed_updates"] == 30
    assert preview["completed_cohorts"] == 15
    assert preview["next_iteration"] == 31
    assert preview["next_cohort_index"] == 15
    assert preview["parent_optimizer_global_step"] == 2590
    assert preview["excluded_tail"]["actor_committed_steps"] == {
        f"{RUN}-g0-lane-000": 81,
        f"{RUN}-g0-lane-001": 81,
        f"{RUN}-g1-lane-000": 0,
        f"{RUN}-g1-lane-001": 0,
    }

    destination = tmp_path / "iteration31-imported"
    result = recovery.import_prefix(reconciliation, destination)
    assert source.read_bytes() == before
    assert result["source_database_unchanged"] is True
    assert result["checkpoint_materialization"]["checkpoint_count"] == 31
    assert result["checkpoint_materialization"]["active_checkpoint"]["global_step"] == 2590
    assert result["excluded_tail"]["replay_allowed"] is False
    assert result["native_work_started"] is False
    assert result["compute_release_requested"] is False
    with sqlite3.connect(destination / "state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM learner_jobs").fetchone()[0] == 30
        assert db.execute("SELECT COUNT(*) FROM iteration_barriers").fetchone()[0] == 120
        assert db.execute("SELECT COUNT(*) FROM step_prepares").fetchone()[0] == 30 * 4 * 81
        assert db.execute(
            "SELECT status,candidate_checkpoint_sha256 FROM round_iterations "
            "WHERE iteration_id=31"
        ).fetchone() == ("collecting", None)
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 31


@pytest.mark.parametrize("first_owner_steps,peer_steps", [(81, 0), (28, 29)])
def test_b16_iteration37_import_preserves_36_updates_and_excludes_tail(
    tmp_path, monkeypatch, first_owner_steps, peer_steps,
):
    reconciliation, source = make_fixture(
        tmp_path, monkeypatch, completed_updates=36, environments_per_actor=16,
        first_owner_steps=first_owner_steps, incomplete_owner_steps=peer_steps,
    )
    before = source.read_bytes()
    destination = tmp_path / "b16-imported"
    result = recovery.import_prefix(reconciliation, destination)
    assert source.read_bytes() == before
    assert result["completed_cohorts"] == 18
    assert result["completed_updates"] == 36
    assert result["next_iteration"] == 37
    assert result["next_cohort_index"] == 18
    assert result["mse_history_iterations"] == [0, 20]
    assert result["checkpoint_materialization"]["checkpoint_count"] == 37
    assert result["checkpoint_materialization"]["active_checkpoint"]["global_step"] == 2596
    tail = result["excluded_tail"]
    assert tail["actor_committed_steps"] == {
        f"{RUN}-g{gpu}-lane-{lane:03d}": first_owner_steps if gpu == 0 else peer_steps
        for gpu in (0, 1) for lane in range(16)
    }
    assert tail["prepare_count"] == tail["commit_count"] == 16 * (first_owner_steps + peer_steps)
    assert tail["barrier_count"] == tail["chunk_count"] == (16 if first_owner_steps == 81 else 0)
    assert tail["replay_allowed"] is False
    assert result["allocation_retained"] is True
    assert result["compute_release_requested"] is False
    with sqlite3.connect(destination / "state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM learner_jobs").fetchone()[0] == 36
        assert db.execute("SELECT COUNT(*) FROM iteration_barriers").fetchone()[0] == 36 * 32
        assert db.execute("SELECT COUNT(*) FROM chunk_reservations").fetchone()[0] == 36 * 32
        assert db.execute("SELECT COUNT(*) FROM step_prepares").fetchone()[0] == 36 * 32 * 81
        assert db.execute("SELECT COUNT(*) FROM step_commits").fetchone()[0] == 36 * 32 * 81
        assert db.execute(
            "SELECT status,candidate_checkpoint_sha256 FROM round_iterations WHERE iteration_id=37"
        ).fetchone() == ("collecting", None)


@pytest.mark.parametrize("fault,message", [
    ("actor_population", "pinned two-GPU configuration"),
    ("learner_population", "full population"),
    ("prefix_barrier", "actor barriers are incomplete"),
    ("tail_barrier", "complete configured native owner"),
    ("pending_prepare", "only durably committed"),
    ("missing_commit", "unmatched prepare/commit"),
    ("unequal_lanes", "unequal durable lengths"),
    ("unknown_actor", "outside the pinned population"),
])
def test_b16_import_rejects_unsafe_population_or_tail(tmp_path, monkeypatch, fault, message):
    reconciliation, source = make_fixture(
        tmp_path, monkeypatch, completed_updates=2, environments_per_actor=16,
        incomplete_owner_steps=1,
    )
    with sqlite3.connect(source) as db:
        if fault == "actor_population":
            actors = json.loads(db.execute("SELECT actor_ids_json FROM runs").fetchone()[0])
            actors[-1] = f"{RUN}-g1-lane-099"
            db.execute("UPDATE runs SET actor_ids_json=?", (json.dumps(actors),))
        elif fault == "learner_population":
            job, chunks = db.execute("SELECT job_id,chunk_ids_json FROM learner_jobs LIMIT 1").fetchone()
            db.execute("UPDATE learner_jobs SET chunk_ids_json=? WHERE job_id=?",
                       (json.dumps(json.loads(chunks)[:-1]), job))
        elif fault in {"prefix_barrier", "tail_barrier"}:
            db.execute("DELETE FROM iteration_barriers WHERE iteration_id=? LIMIT 1",
                       (1 if fault == "prefix_barrier" else 3,))
        elif fault == "pending_prepare":
            db.execute("UPDATE step_prepares SET status='prepared' WHERE idempotency_key=?",
                       (f"tail-prepare-{RUN}-g1-lane-000-0",))
        elif fault in {"missing_commit", "unequal_lanes"}:
            db.execute("DELETE FROM step_commits WHERE idempotency_key=?",
                       (f"tail-commit-{RUN}-g1-lane-000-0",))
            if fault == "unequal_lanes":
                db.execute("DELETE FROM step_prepares WHERE idempotency_key=?",
                           (f"tail-prepare-{RUN}-g1-lane-000-0",))
        else:
            db.execute("UPDATE step_prepares SET actor_id=? WHERE idempotency_key=?",
                       (f"{RUN}-g1-lane-099", f"tail-prepare-{RUN}-g1-lane-000-0"))
    reconciliation["closed_database"] = recovery.stable_file_ref(source)
    attestation_path = Path(reconciliation["h200_services_attestation"]["path"])
    attestation = json.loads(attestation_path.read_bytes())
    attestation["closed_database"] = reconciliation["closed_database"]
    attestation_path.write_text(json.dumps(attestation))
    reconciliation["h200_services_attestation"] = binding(attestation_path)
    with pytest.raises(ValueError, match=message):
        recovery.inspect_prefix(reconciliation)


def test_readonly_checkpoint_sources_are_hardlinked_without_source_mutation(tmp_path, monkeypatch):
    reconciliation, source = make_fixture(tmp_path, monkeypatch)
    with sqlite3.connect(source) as db:
        rows = db.execute("SELECT sha256,path FROM checkpoints ORDER BY sha256").fetchall()
    for _, path in rows:
        Path(path).chmod(0o444)
    before = {sha: (os.stat(path).st_ino, Path(path).read_bytes()) for sha, path in rows}
    destination = tmp_path / "hardlinked"
    result = recovery.import_prefix(reconciliation, destination)
    assert result["checkpoint_materialization"]["hardlinked_count"] == 27
    assert result["checkpoint_materialization"]["copied_count"] == 0
    assert result["checkpoint_materialization"]["reflinked_count"] == 0
    for sha, path in rows:
        resolved = content_addressed_checkpoint(destination, sha)
        assert resolved.stat().st_ino == before[sha][0]
        assert Path(path).read_bytes() == before[sha][1]
        assert Path(path).stat().st_mode & 0o222 == 0


def test_writable_checkpoint_sources_use_independent_fsynced_copy(tmp_path, monkeypatch):
    reconciliation, source = make_fixture(tmp_path, monkeypatch)
    with sqlite3.connect(source) as db:
        rows = db.execute("SELECT sha256,path FROM checkpoints ORDER BY sha256").fetchall()

    def unsupported_reflink(*_args, **_kwargs):
        raise OSError(errno.EOPNOTSUPP, "reflink unavailable")

    monkeypatch.setattr(recovery.fcntl, "ioctl", unsupported_reflink)
    destination = tmp_path / "copied"
    result = recovery.import_prefix(reconciliation, destination)
    assert result["checkpoint_materialization"]["hardlinked_count"] == 0
    assert result["checkpoint_materialization"]["reflinked_count"] == 0
    assert result["checkpoint_materialization"]["copied_count"] == 27
    for sha, path in rows:
        resolved = content_addressed_checkpoint(destination, sha)
        assert resolved.stat().st_ino != Path(path).stat().st_ino
        assert resolved.read_bytes() == Path(path).read_bytes()
        assert resolved.stat().st_mode & 0o222 == 0


def test_any_historical_checkpoint_tamper_is_rejected(tmp_path, monkeypatch):
    reconciliation, source = make_fixture(tmp_path, monkeypatch)
    with sqlite3.connect(source) as db:
        path = Path(db.execute("SELECT path FROM checkpoints ORDER BY created_at,sha256 LIMIT 1").fetchone()[0])
    path.write_bytes(b"tampered historical checkpoint")
    with pytest.raises(ValueError, match="prefix checkpoint bytes changed"):
        recovery.inspect_prefix(reconciliation)


@pytest.mark.parametrize("fault", ["tail_candidate", "missing_commit", "one_barrier", "wrong_boundary"])
def test_unsafe_or_noncohort_boundary_import_is_rejected(tmp_path, monkeypatch, fault):
    reconciliation, source = make_fixture(tmp_path, monkeypatch)
    if fault == "wrong_boundary":
        reconciliation["completed_cohorts"] = 12
    else:
        with sqlite3.connect(source) as db:
            if fault == "tail_candidate":
                db.execute("UPDATE round_iterations SET candidate_checkpoint_sha256=? WHERE iteration_id=27", ("9" * 64,))
            elif fault == "missing_commit":
                db.execute("DELETE FROM step_commits WHERE idempotency_key LIKE 'tail-commit-%' LIMIT 1")
            else:
                db.execute("DELETE FROM iteration_barriers WHERE iteration_id=27 LIMIT 1")
        reconciliation["closed_database"] = recovery.stable_file_ref(source)
        h200 = json.loads(Path(reconciliation["h200_services_attestation"]["path"]).read_bytes())
        h200["closed_database"] = reconciliation["closed_database"]
        Path(reconciliation["h200_services_attestation"]["path"]).write_text(json.dumps(h200))
        reconciliation["h200_services_attestation"] = binding(reconciliation["h200_services_attestation"]["path"])
    with pytest.raises((ValueError, RuntimeError)):
        recovery.inspect_prefix(reconciliation)
