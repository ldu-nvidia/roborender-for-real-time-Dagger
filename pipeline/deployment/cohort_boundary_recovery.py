"""Offline, fail-closed import of a completed DAgger cohort prefix.

This is deliberately not a live-ledger rollback and not an Isaac snapshot
resume.  A stopped owner first binds a closed copy of the failed ledger.  The
importer validates the complete two-window cohort prefix, accounts for every
row in the failed first window, and creates a *new* owner-host ledger containing
only the completed prefix plus a clean collecting iteration.  The source
database and failed native namespace are never changed or reused.

The destination is inert: this module does not launch services, simulators,
learners, controllers, or allocations.  Starting a continuation additionally
requires a fresh native runtime generation and independently admitted service
and campaign bindings.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from pipeline.bc_training.handoff_common import binding, bound, canonical
from pipeline.dagger.coverage_production import digest, load_plan
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    actor_ids,
    environments_per_actor,
    is_mse_production,
    validate_config,
)
from pipeline.dagger.services import mse_protocol
from pipeline.dagger.services.render_student import content_addressed_checkpoint
from pipeline.dagger.services.store import StateStore
from pipeline.dagger.export_heldout_manifest import validate_export_admission
from pipeline.deployment.resident_production_collection import completed_prefix


SCHEMA = "roborender.dagger-cohort-boundary-reconciliation.v2"
RECEIPT_SCHEMA = "roborender.dagger-cohort-prefix-import.v3"
TAIL_DISPOSITION = "abandoned_never_replayed_or_learned"
HEX64 = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError(message)


def stable_file_ref(path: Path, *, recorded_path: Path | None = None):
    """Bind one regular file while rejecting path substitution and mutation."""
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "regular non-symlink file required")
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file changed while binding",
    )
    return {
        "path": str(Path(recorded_path).absolute() if recorded_path is not None else path),
        "size_bytes": after.st_size,
        "sha256": sha.hexdigest(),
    }


def _validate_file_ref(ref):
    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "size_bytes", "sha256"}
        and Path(ref["path"]).is_absolute()
        and type(ref["size_bytes"]) is int
        and ref["size_bytes"] >= 0
        and isinstance(ref["sha256"], str)
        and HEX64.fullmatch(ref["sha256"]),
        "invalid closed-file binding",
    )
    require(stable_file_ref(Path(ref["path"])) == ref, "closed-file binding changed")
    return ref


def _closed_database(ref):
    """Open only a bound, sidecar-free rollback-journal snapshot."""
    _validate_file_ref(ref)
    path = Path(ref["path"])
    require(path.stat().st_size <= 2 << 30, "closed database exceeds bounded metadata size")
    require(
        not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
        "closed database has a live or uncertain sidecar",
    )
    with path.open("rb") as stream:
        header = stream.read(20)
    require(
        header[:16] == b"SQLite format 3\0" and header[18:20] == b"\x01\x01",
        "closed database is not rollback-journal SQLite",
    )
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    require(db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete", "closed database is not DELETE journal")
    require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "closed database failed quick_check")
    require(not db.execute("PRAGMA foreign_key_check").fetchall(), "closed database failed foreign-key check")
    return db


def _rows(db, query, parameters=()):
    return [dict(row) for row in db.execute(query, parameters)]


def _single(rows, message):
    require(len(rows) == 1, message)
    return rows[0]


def _load_ref(ref, label):
    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "raw_sha256"}
        and Path(ref["path"]).is_absolute()
        and isinstance(ref["raw_sha256"], str)
        and HEX64.fullmatch(ref["raw_sha256"]),
        f"invalid {label} binding",
    )
    return bound(ref)


def _validate_reconciliation(value):
    fields = {
        "schema", "run_id", "round_id", "source_sha256", "completed_cohorts",
        "completed_updates", "failed_iteration", "failed_cohort_index",
        "tail_disposition", "native_replay_allowed", "fresh_native_runtime_required",
        "old_namespace_preserved", "allocation_retained", "compute_release_requested",
        "closed_database", "round_config", "coverage_plan", "heldout_manifest",
        "heldout_admission", "failed_cohort_intent", "completed_cohort_receipts",
        "controller", "h200_services", "h200_services_attestation",
        "native_workers", "native_terminal_attestation",
    }
    require(isinstance(value, dict) and set(value) == fields and value.get("schema") == SCHEMA,
            "invalid cohort-boundary reconciliation schema/fields")
    require(isinstance(value["run_id"], str) and value["run_id"], "run_id required")
    require(value["round_id"] == 1, "only active round1 prefix import is admitted")
    require(isinstance(value["source_sha256"], str) and HEX64.fullmatch(value["source_sha256"]),
            "source SHA required")
    cohorts, updates = value["completed_cohorts"], value["completed_updates"]
    require(type(cohorts) is int and cohorts >= 1
            and type(updates) is int and updates == 2 * cohorts,
            "recovery requires an exact positive two-update-per-cohort prefix")
    require(value["failed_iteration"] == updates + 1 and value["failed_cohort_index"] == cohorts,
            "failed tail is not the first window after the completed cohort prefix")
    require(value["tail_disposition"] == TAIL_DISPOSITION
            and value["native_replay_allowed"] is False
            and value["fresh_native_runtime_required"] is True
            and value["old_namespace_preserved"] is True,
            "failed native tail must be preserved, excluded, and never replayed")
    require(value["allocation_retained"] is True and value["compute_release_requested"] is False,
            "recovery must retain the allocation without requesting release")
    _validate_file_ref(value["closed_database"])
    for key in ("round_config", "coverage_plan", "heldout_manifest", "heldout_admission", "failed_cohort_intent",
                "h200_services_attestation", "native_terminal_attestation"):
        _load_ref(value[key], key)
    refs = value["completed_cohort_receipts"]
    require(isinstance(refs, list) and len(refs) == cohorts, "exact completed cohort receipt prefix required")
    for ref in refs:
        _load_ref(ref, "completed cohort")
    controller = value["controller"]
    require(isinstance(controller, dict) and controller.get("state") in {"FAILED", "STOPPED"}
            and controller.get("identity_checked") is True,
            "old controller must be identity-checked and terminal")
    services = value["h200_services"]
    require(isinstance(services, dict) and services.get("state") == "STOPPED"
            and services.get("identity_checked") is True
            and services.get("allocation_retained") is True
            and services.get("compute_released") is False,
            "old H200 service owner must be stopped while compute remains retained")
    service_attestation = _load_ref(value["h200_services_attestation"], "H200 service attestation")
    require(service_attestation.get("label") == "h200-services"
            and service_attestation.get("result", {}).get("state") == "STOPPED"
            and service_attestation.get("allocation_retained") is True
            and service_attestation.get("compute_release_requested") is False
            and service_attestation.get("closed_database") == value["closed_database"],
            "H200 service closure does not bind the closed source ledger")
    workers = value["native_workers"]
    require(isinstance(workers, list) and len(workers) == 2
            and {row.get("physical_gpu") for row in workers} == {0, 1},
            "both old native workers must be reconciled")
    for worker in workers:
        require(worker.get("identity_checked") is True
                and worker.get("unit_state") in {"inactive", "failed"}
                and worker.get("container_absent") is True,
                "old native worker is not terminal")
    native_attestation = _load_ref(value["native_terminal_attestation"], "native terminal attestation")
    require(native_attestation.get("schema") == "roborender.native-recovery-terminal-attestation.v1"
            and native_attestation.get("workers") == workers
            and native_attestation.get("native_replay_allowed") is False
            and native_attestation.get("compute_release_requested") is False,
            "native terminal attestation differs from reconciled workers")
    return value


def _actor_gpu(actor):
    match = re.search(r"-g([01])-lane-[0-9]{3}$", actor)
    require(match is not None, "production actor ID does not encode an exact physical GPU")
    return int(match.group(1))


def _validate_artifacts(reconciliation, coordinator_rows):
    config = validate_config(_load_ref(reconciliation["round_config"], "round config"), runtime=True)
    require(is_mse_production(config) and config["run_id"] == reconciliation["run_id"]
            and config["round_id"] == reconciliation["round_id"]
            and config["source_sha256"] == reconciliation["source_sha256"],
            "round configuration differs from failed production run")
    require(config["coverage_plan"] == reconciliation["coverage_plan"],
            "round config does not bind the recovery coverage plan")
    plan = load_plan(reconciliation["coverage_plan"])
    require(plan["run_id"] == config["run_id"] and plan["source_sha256"] == config["source_sha256"],
            "coverage plan differs from round/source")
    require(plan["topology"]["environments_per_actor"] == environments_per_actor(config)
            and plan["topology"]["total_environments"] == len(actor_ids(config)),
            "coverage plan population differs from the pinned round configuration")
    protocol_manifest = reconciliation["heldout_manifest"]
    admission_ref = reconciliation["heldout_admission"]
    admission = _load_ref(admission_ref, "heldout admission")
    collection = validate_export_admission(
        Path(admission_ref["path"]), admission_ref["raw_sha256"],
        Path(protocol_manifest["path"]), protocol_manifest["raw_sha256"],
    )
    manifest = _load_ref(protocol_manifest, "heldout manifest")
    require(
        admission.get("schema") == "roborender.dagger-heldout-admission.v1"
        and admission.get("purpose") == "validation"
        and admission.get("qualification") == "metadata_only_actual_array_hash_verified_at_evaluation"
        and admission.get("array_payloads_read") is False
        and len(admission.get("completed_actors", [])) == 320
        and len(manifest.get("chunks", [])) == 320
        and collection.get("round_id") == reconciliation["round_id"],
        "fixed heldout admission is not the exact complete 320-chunk population",
    )

    expected = []
    for epoch in plan["rounds"][0]["epochs"]:
        for cohort in epoch["cohorts"]:
            expected.append((epoch["coverage_epoch"], cohort["cohort_index"]))
    refs = reconciliation["completed_cohort_receipts"]
    require(len(refs) < len(expected), "failed prefix must leave a next cohort")
    parent = config["student"]["initial_checkpoint_sha256"]
    global_step = None
    observed = []
    for ordinal, ref in enumerate(refs):
        path = Path(ref["path"])
        require(path.name == "completion.json" and binding(path) == ref,
                "completed cohort receipt binding changed")
        saved = _load_ref(ref, "completed cohort")
        if global_step is None:
            global_step = saved.get("input_optimizer_global_step")
            require(type(global_step) is int and global_step >= 0, "initial optimizer step missing")
        epoch, index = expected[ordinal]
        checked = completed_prefix(path.parent, plan, 1, epoch, index, parent, coordinator_rows)
        require(checked is not None, "completed cohort prefix receipt missing")
        checked_saved, record = checked
        require(checked_saved == saved, "completed cohort changed during validation")
        observed.append((record["coverage_epoch"], record["cohort_index"]))
        parent = saved["output_checkpoint_sha256"]
        global_step += 2
    require(observed == expected[:len(refs)], "completed receipts are not the deterministic plan prefix")

    intent = _load_ref(reconciliation["failed_cohort_intent"], "failed cohort intent")
    next_epoch, next_index = expected[len(refs)]
    commands = intent.get("commands")
    require(intent.get("input_checkpoint_sha256") == parent
            and intent.get("source_sha256") == config["source_sha256"]
            and isinstance(intent.get("segment_sha256"), str)
            and HEX64.fullmatch(intent["segment_sha256"])
            and isinstance(commands, list) and len(commands) == 2
            and {row.get("gpu") for row in commands} == {0, 1}
            and all(row.get("operation") == "collect_production_segment" for row in commands)
            and all(row.get("payload", {}).get("segment_sha256") == intent["segment_sha256"] for row in commands)
            and all(f"-e{next_epoch:02d}-c{next_index:03d}-g{row['gpu']}" in row.get("command_id", "") for row in commands),
            "failed intent is not the exact next planned cohort")
    return config, plan, parent, global_step


def _validate_checkpoint_files(rows):
    references = []
    for row in rows:
        path = Path(row["path"])
        ref = stable_file_ref(path)
        require(ref["sha256"] == row["sha256"] and ref["size_bytes"] == row["size_bytes"],
                "prefix checkpoint bytes changed")
        references.append(ref)
    return references


def _clone_or_copy_checkpoint_exclusive(source: Path, target: Path):
    """Install one independent, read-only checkpoint via reflink or copy."""
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.part"
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    target_fd = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400
    )
    method = "reflink"
    try:
        try:
            # Linux FICLONE: an independent copy-on-write inode, never a
            # writable alias back into the preserved failed namespace.
            fcntl.ioctl(target_fd, 0x40049409, source_fd)
        except OSError as error:
            clone_fallback = {
                errno.EXDEV, errno.EPERM, errno.EACCES, errno.EINVAL,
                errno.ENOSYS, errno.ENOTTY, errno.EOPNOTSUPP,
                getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            }
            if error.errno not in clone_fallback:
                raise
            method = "copy"
            os.ftruncate(target_fd, 0)
            os.lseek(source_fd, 0, os.SEEK_SET)
            os.lseek(target_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(source_fd), "rb") as reader, os.fdopen(
                os.dup(target_fd), "wb"
            ) as writer:
                shutil.copyfileobj(reader, writer, length=1 << 20)
                writer.flush()
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o444)
    except BaseException:
        os.close(target_fd)
        os.close(source_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    os.close(target_fd)
    os.close(source_fd)
    os.rename(temporary, target)
    return method


def _materialize_checkpoints(stage: Path, destination: Path, rows):
    """Create the exact content-addressed checkpoint tree for a fresh ledger.

    Registry paths are rewritten to their post-publication destination.  Bytes
    are first installed below the hidden staging directory, so neither a
    partial checkpoint nor a ledger that refers to it is externally visible.
    Hard links preserve the admitted payload without duplicating it; a bounded
    set of filesystem limitations falls back to an exclusive, fsynced copy.
    """
    stage = Path(stage).absolute()
    destination = Path(destination).absolute()
    checkpoint_stage = stage / "checkpoints"
    checkpoint_final = destination / "checkpoints"
    checkpoint_stage.mkdir(mode=0o700)
    require(rows and len({row["sha256"] for row in rows}) == len(rows),
            "checkpoint registry must contain unique content addresses")
    source_refs = _validate_checkpoint_files(rows)
    copied = 0
    reflinked = 0
    hardlinked = 0
    rewritten = []
    for row, source_ref in zip(rows, source_refs):
        digest_value = row["sha256"]
        source = Path(row["path"]).absolute()
        staged = checkpoint_stage / f"{digest_value}.pt"
        require(row["size_bytes"] > 0, "checkpoint payload must be non-empty")
        if (source.stat().st_mode & 0o222) == 0:
            try:
                os.link(source, staged, follow_symlinks=False)
                hardlinked += 1
                method = None
            except OSError as error:
                link_fallback = {
                    errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK,
                    errno.EROFS, errno.EOPNOTSUPP,
                    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                }
                if error.errno not in link_fallback:
                    raise
                method = _clone_or_copy_checkpoint_exclusive(source, staged)
            if method == "reflink":
                reflinked += 1
            elif method == "copy":
                copied += 1
        else:
            method = _clone_or_copy_checkpoint_exclusive(source, staged)
            reflinked += method == "reflink"
            copied += method == "copy"
        installed = stable_file_ref(staged)
        require(installed["sha256"] == digest_value
                and installed["size_bytes"] == row["size_bytes"],
                "materialized checkpoint bytes differ from registry")
        require(stable_file_ref(source) == source_ref,
                "source checkpoint changed during materialization")
        resolved = content_addressed_checkpoint(stage, digest_value)
        with resolved.open("rb") as installed_stream, source.open("rb") as source_stream:
            require(resolved == staged and installed_stream.read(1) == source_stream.read(1),
                    "materialized checkpoint is not resolver-loadable")
        value = dict(row)
        value["path"] = str(checkpoint_final / staged.name)
        rewritten.append(value)
    directory_fd = os.open(checkpoint_stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return rewritten, {
        "schema": "roborender.dagger-cohort-checkpoint-materialization.v1",
        "checkpoint_count": len(rewritten),
        "total_size_bytes": sum(row["size_bytes"] for row in rewritten),
        "hardlinked_count": hardlinked,
        "reflinked_count": reflinked,
        "copied_count": copied,
        "content_addressed_directory": str(checkpoint_final),
        "all_hashes_verified": True,
        "all_paths_rewritten": True,
        "source_paths_preserved": True,
    }


def _validate_training_checkpoint(path: Path, digest_value: str, global_step: int, learning_rate: float):
    """Fully deserialize the active policy+AdamW payload using learner rules."""
    import torch
    from pipeline.dagger.services.raven_learner_worker import validate_mse_parent_checkpoint

    path = Path(path)
    ref = stable_file_ref(path)
    require(ref["sha256"] == digest_value, "active checkpoint digest changed before load")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict) and isinstance(checkpoint.get("policy_spec"), dict),
            "active checkpoint lacks its embedded policy specification")
    validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=learning_rate)
    require(checkpoint.get("global_step") == global_step,
            "active checkpoint optimizer step differs from admitted learner lineage")
    return {
        "sha256": digest_value,
        "path": str(path),
        "global_step": global_step,
        "schema": checkpoint["schema"],
        "policy_spec_embedded": True,
        "optimizer_state_load_verified": True,
    }


def _validate_payload_metadata(rows):
    """Check array references without re-reading historical tensor bytes."""
    for row in rows:
        path = Path(row["path"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["size_bytes"],
                "referenced chunk is missing, substituted, or has a different size")


def _training_provenance(db, run_id, round_id, protocol):
    from pipeline.dagger.services.heldout_mse import SCOPED_MANIFEST_SCHEMA
    manifest = json.loads(Path(protocol["validation_manifest_path"]).read_bytes())
    return mse_protocol.training_provenance(
        db, run_id, round_id, scoped_episodes=manifest.get("schema") == SCOPED_MANIFEST_SCHEMA,
    )


def inspect_prefix(reconciliation):
    """Validate a closed failed ledger and return an import snapshot.

    The returned private ``tables`` member is consumed only by ``import_prefix``;
    ``report`` is the serializable audit preview.
    """
    reconciliation = _validate_reconciliation(reconciliation)
    source_ref = reconciliation["closed_database"]
    run_id = reconciliation["run_id"]
    round_id = reconciliation["round_id"]
    updates = reconciliation["completed_updates"]
    failed_iteration = reconciliation["failed_iteration"]
    with closing(_closed_database(source_ref)) as db:
        run = _single(_rows(db, "SELECT * FROM runs WHERE run_id=?", (run_id,)), "production run missing or duplicated")
        actors = json.loads(run["actor_ids_json"])
        require(run["current_round"] == round_id and run["beta"] == .75 and run["status"] == "collecting",
                "run is not active round1 collection")

        iterations = _rows(db, "SELECT * FROM round_iterations WHERE run_id=? ORDER BY iteration_id", (run_id,))
        config, plan, artifact_parent, optimizer_step = _validate_artifacts(reconciliation, iterations)
        batch = environments_per_actor(config)
        population = 2 * batch
        require(actors == actor_ids(config) and len(set(actors)) == population
                and {_actor_gpu(actor) for actor in actors} == {0, 1}
                and run["rollout_steps"] == 81,
                "ledger actors/rollout length differ from the pinned two-GPU configuration")
        require(len(iterations) == failed_iteration
                and [row["iteration_id"] for row in iterations] == list(range(1, failed_iteration + 1)),
                "iteration prefix is not contiguous through the failed tail")
        checkpoints = [run["initial_checkpoint_sha256"]]
        for index, row in enumerate(iterations, 1):
            require(row["round_id"] == round_id and row["input_checkpoint_sha256"] == checkpoints[-1],
                    "iteration checkpoint chain changed")
            if index <= updates:
                candidate = row["candidate_checkpoint_sha256"]
                require(row["status"] == "advanced" and isinstance(candidate, str)
                        and HEX64.fullmatch(candidate) and candidate not in checkpoints,
                        "completed iteration is not one unique advanced candidate")
                checkpoints.append(candidate)
            else:
                require(row["status"] == "collecting" and row["candidate_checkpoint_sha256"] is None
                        and row["evaluation_id"] is None and row["validation_goal_reached_rate"] is None,
                        "failed tail already learned, validated, or selected")
        require(run["active_checkpoint_sha256"] == checkpoints[-1], "active checkpoint differs from completed prefix")

        jobs = _rows(db, "SELECT * FROM learner_jobs WHERE run_id=? ORDER BY created_at,job_id", (run_id,))
        links = _rows(db, "SELECT * FROM learner_job_iterations WHERE run_id=? ORDER BY iteration_id", (run_id,))
        require(len(jobs) == len(links) == updates
                and [row["iteration_id"] for row in links] == list(range(1, updates + 1))
                and len({row["job_id"] for row in links}) == updates,
                "learner jobs are not the exact completed update prefix")
        jobs_by_id = {row["job_id"]: row for row in jobs}
        prefix_chunk_ids = []
        for link in links:
            job = jobs_by_id.get(link["job_id"])
            require(job is not None and job["round_id"] == round_id and job["status"] == "succeeded"
                    and job["input_checkpoint_sha256"] == checkpoints[link["iteration_id"] - 1],
                    "completed learner job lineage differs")
            result = json.loads(job["result_json"] or "null")
            require(isinstance(result, dict)
                    and result.get("iteration_id") == link["iteration_id"]
                    and result.get("parent_checkpoint_sha256") == checkpoints[link["iteration_id"] - 1]
                    and result.get("output_checkpoint_sha256") == checkpoints[link["iteration_id"]]
                    and result.get("optimizer_state_preserved") is True
                    and result.get("replay_manifest_sha256") is None,
                    "learner receipt does not preserve the admitted optimizer/checkpoint chain")
            chunk_ids = json.loads(job["chunk_ids_json"])
            require(len(chunk_ids) == population and len(set(chunk_ids)) == population,
                    "each completed update requires the configured full population of unique chunks")
            prefix_chunk_ids.extend(chunk_ids)
        require(len(prefix_chunk_ids) == population * updates and len(set(prefix_chunk_ids)) == len(prefix_chunk_ids),
                "completed updates reused a training chunk")

        checkpoint_rows = _rows(
            db,
            "SELECT * FROM checkpoints WHERE sha256 IN (%s) ORDER BY created_at,sha256"
            % ",".join("?" for _ in checkpoints),
            checkpoints,
        )
        require({row["sha256"] for row in checkpoint_rows} == set(checkpoints)
                and db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == len(checkpoints),
                "checkpoint registry contains missing or post-prefix candidates")

        chunk_rows = _rows(
            db,
            "SELECT * FROM chunk_reservations WHERE chunk_id IN (%s) ORDER BY chunk_id"
            % ",".join("?" for _ in prefix_chunk_ids),
            prefix_chunk_ids,
        )
        provenance_rows = _rows(
            db,
            "SELECT * FROM chunk_provenance WHERE chunk_id IN (%s) ORDER BY chunk_id"
            % ",".join("?" for _ in prefix_chunk_ids),
            prefix_chunk_ids,
        )
        require(len(chunk_rows) == len(provenance_rows) == len(prefix_chunk_ids)
                and all(row["status"] == "committed" for row in chunk_rows),
                "completed learner population lacks committed chunk provenance")

        barriers = _rows(db, "SELECT * FROM iteration_barriers WHERE run_id=? ORDER BY iteration_id,actor_id", (run_id,))
        prefix_barriers = [row for row in barriers if row["iteration_id"] <= updates]
        tail_barriers = [row for row in barriers if row["iteration_id"] == failed_iteration]
        require(len(prefix_barriers) == population * updates
                and all(sum(row["iteration_id"] == iid for row in prefix_barriers) == population
                        for iid in range(1, updates + 1)),
                "completed iteration actor barriers are incomplete")
        chunks_by_iteration = {
            link["iteration_id"]: set(json.loads(jobs_by_id[link["job_id"]]["chunk_ids_json"]))
            for link in links
        }
        for iid in range(1, updates + 1):
            rows = [row for row in prefix_barriers if row["iteration_id"] == iid]
            require(
                {row["actor_id"] for row in rows} == set(actors)
                and all(len(json.loads(row["chunk_ids_json"])) == 1 for row in rows)
                and {item for row in rows for item in json.loads(row["chunk_ids_json"])} == chunks_by_iteration[iid]
                and all(row["checkpoint_sha256"] == checkpoints[iid - 1] for row in rows),
                "completed actor barriers differ from learner population/parent",
            )
        barrier_actors = {row["actor_id"] for row in tail_barriers}
        require(len(tail_barriers) in {0, batch}
                and len(barrier_actors) == len(tail_barriers)
                and (not tail_barriers
                     or len({_actor_gpu(row["actor_id"]) for row in tail_barriers}) == 1),
                "failed tail may seal at most one complete configured native owner")
        require(not [row for row in barriers if row["iteration_id"] > failed_iteration], "post-failure barriers exist")

        window = f"window-{failed_iteration - 1:06d}"
        expected_prefix_windows = {f"window-{index:06d}" for index in range(updates)}
        all_prepares = _rows(
            db,
            "SELECT * FROM step_prepares WHERE run_id=? AND round_id=? "
            "ORDER BY rollout_window_id,actor_id,rollout_step_index",
            (run_id, round_id),
        )
        require({row["rollout_window_id"] for row in all_prepares}
                <= expected_prefix_windows | {window},
                "step ledger contains a future or unrecognized rollout window")
        require(all(row["actor_id"] in actors for row in all_prepares),
                "step ledger contains an actor outside the pinned population")
        prefix_prepares = [
            row for row in all_prepares if row["rollout_window_id"] in expected_prefix_windows
        ]
        tail_prepares = [row for row in all_prepares if row["rollout_window_id"] == window]
        require(all(row["status"] == "committed" for row in tail_prepares),
                "failed tail must contain only durably committed selections")
        require(all(row["checkpoint_sha256"] == checkpoints[-1] for row in tail_prepares)
                and all(row["checkpoint_sha256"] == checkpoints[-1] for row in tail_barriers),
                "failed tail differs from the verified completed-prefix parent")
        tail_keys = [row["idempotency_key"] for row in tail_prepares]
        all_commits = _rows(
            db,
            "SELECT c.* FROM step_commits c JOIN step_prepares p ON p.idempotency_key=c.prepare_idempotency_key "
            "WHERE p.run_id=? AND p.round_id=? "
            "ORDER BY p.rollout_window_id,p.actor_id,p.rollout_step_index",
            (run_id, round_id),
        )
        tail_key_set = set(tail_keys)
        tail_commits = [row for row in all_commits if row["prepare_idempotency_key"] in tail_key_set]
        require(len(tail_commits) == len(tail_prepares)
                and {row["prepare_idempotency_key"] for row in tail_commits} == set(tail_keys),
                "failed tail has unmatched prepare/commit effects")
        lengths = {}
        for actor in actors:
            indices = [row["rollout_step_index"] for row in tail_prepares if row["actor_id"] == actor]
            require(indices == list(range(len(indices))) and len(indices) <= 81,
                    "failed actor tail is not one contiguous window prefix")
            lengths[actor] = len(indices)
        actors_by_gpu = {
            gpu: sorted(actor for actor in actors if _actor_gpu(actor) == gpu)
            for gpu in (0, 1)
        }
        require(all(len(gpu_actors) == batch for gpu_actors in actors_by_gpu.values())
                and all(len({lengths[actor] for actor in gpu_actors}) == 1
                        for gpu_actors in actors_by_gpu.values()),
                "failed tail lanes on a physical GPU have unequal durable lengths")
        require(all((lengths[actor] == 81) == (actor in barrier_actors) for actor in actors),
                "a failed-tail lane is sealed without 81 steps or has 81 steps without a barrier")
        if barrier_actors:
            sealed_gpu = _actor_gpu(next(iter(barrier_actors)))
            require(barrier_actors == set(actors_by_gpu[sealed_gpu]),
                    "failed tail barriers are not the exact configured lanes of one physical GPU")

        tail_chunk_lists = [json.loads(row["chunk_ids_json"]) for row in tail_barriers]
        require(all(isinstance(items, list) and len(items) == 1 for items in tail_chunk_lists),
                "each failed-tail barrier must bind exactly one excluded chunk")
        tail_chunk_ids = [item for items in tail_chunk_lists for item in items]
        require(len(tail_chunk_ids) == len(tail_barriers)
                and len(set(tail_chunk_ids)) == len(tail_chunk_ids),
                "failed tail chunk cardinality differs from sealed lanes")
        if tail_chunk_ids:
            placeholders = ",".join("?" for _ in tail_chunk_ids)
            tail_chunks = _rows(
                db,
                f"SELECT * FROM chunk_reservations WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id",
                tail_chunk_ids,
            )
            tail_provenance = _rows(
                db,
                f"SELECT * FROM chunk_provenance WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id",
                tail_chunk_ids,
            )
        else:
            tail_chunks, tail_provenance = [], []
        require(len(tail_chunks) == len(tail_provenance) == len(tail_chunk_ids)
                and all(row["status"] == "committed" for row in tail_chunks),
                "failed sealed lanes lack exact excluded chunk provenance")
        all_training_chunks = set(prefix_chunk_ids) | set(tail_chunk_ids)
        unexpected_chunks = []
        for row in _rows(db, "SELECT * FROM chunk_reservations"):
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("chunk registry contains invalid metadata") from error
            if metadata.get("run_id") == run_id and row["chunk_id"] not in all_training_chunks:
                unexpected_chunks.append(row["chunk_id"])
        require(not unexpected_chunks, "production run contains chunks outside prefix and failed tail")

        require(len(prefix_prepares) == updates * population * 81
                and all(row["status"] == "committed" for row in prefix_prepares)
                and {row["rollout_window_id"] for row in prefix_prepares}
                    == expected_prefix_windows,
                "completed prefix step ledger cardinality differs")
        for index in range(updates):
            prefix_window = f"window-{index:06d}"
            for actor in actors:
                actor_rows = [
                    row for row in prefix_prepares
                    if row["rollout_window_id"] == prefix_window and row["actor_id"] == actor
                ]
                require([row["rollout_step_index"] for row in actor_rows] == list(range(81))
                        and all(row["checkpoint_sha256"] == checkpoints[index] for row in actor_rows),
                        "completed prefix actor window is incomplete or has the wrong parent")
        prefix_keys = [row["idempotency_key"] for row in prefix_prepares]
        prefix_key_set = set(prefix_keys)
        prefix_commits = [row for row in all_commits if row["prepare_idempotency_key"] in prefix_key_set]
        require(len(prefix_commits) == len(prefix_prepares)
                and {row["prepare_idempotency_key"] for row in prefix_commits} == set(prefix_keys),
                "completed prefix has unmatched step effects")
        require(len(all_commits) == len(prefix_commits) + len(tail_commits),
                "step commit ledger contains effects outside prefix and failed tail")

        protocols = _rows(db, "SELECT * FROM mse_round_protocols WHERE run_id=?", (run_id,))
        protocol = _single(protocols, "exact active MSE protocol required")
        protocol_value = mse_protocol.validate_protocol(json.loads(protocol["protocol_json"]))
        selection = json.loads(protocol["selection_json"] or "null")
        require(protocol["round_id"] == round_id and protocol["status"] == "active"
                and protocol["stop_reason"] is None and protocol["final_test_id"] is None
                and protocol_value["initial_checkpoint_sha256"] == run["initial_checkpoint_sha256"]
                and protocol_value["validation_manifest_path"] == reconciliation["heldout_manifest"]["path"]
                and protocol_value["validation_manifest_sha256"] == reconciliation["heldout_manifest"]["raw_sha256"]
                and isinstance(selection, dict)
                and [row.get("iteration") for row in selection.get("history", [])]
                    == list(range(0, updates + 1, protocol_value["interval_iterations"])),
                "MSE history is not the exact completed-prefix selection state")
        computed_provenance = _training_provenance(db, run_id, round_id, protocol_value)
        require(computed_provenance == json.loads(protocol["training_provenance_json"]),
                "cached training provenance differs from completed learner prefix")
        require(not _rows(db, "SELECT * FROM promotions WHERE run_id=?", (run_id,))
                and not _rows(db, "SELECT * FROM success_evaluations WHERE run_id=?", (run_id,)),
                "round selection/test already started")

        final_job = jobs_by_id[links[-1]["job_id"]]
        require(run["config_sha256"] == digest(config)
                and run["initial_checkpoint_sha256"] == config["student"]["initial_checkpoint_sha256"]
                and artifact_parent == checkpoints[-1]
                and optimizer_step == json.loads(final_job["result_json"])["global_step"],
                "artifact prefix differs from closed ledger lineage")
        require(reconciliation["source_sha256"] == plan["source_sha256"], "reconciled source changed")

        # Every registered policy+AdamW payload remains selectable by the
        # fixed-validation protocol.  Re-hash the complete admitted checkpoint
        # lineage; historical rollout arrays remain metadata-bound and are
        # only stat-checked here.
        _validate_checkpoint_files(checkpoint_rows)
        _validate_payload_metadata(chunk_rows + tail_chunks)

        tables = {
            "checkpoints": checkpoint_rows,
            "runs": [run],
            "round_iterations": iterations,
            "chunk_reservations": chunk_rows,
            "chunk_provenance": provenance_rows,
            "iteration_barriers": prefix_barriers,
            "learner_jobs": jobs,
            "learner_job_iterations": links,
            "mse_round_protocols": [protocol],
            "step_prepares": prefix_prepares,
            "step_commits": prefix_commits,
        }
        tail = {
            "iteration_id": failed_iteration,
            "checkpoint_sha256": checkpoints[-1],
            "prepare_count": len(tail_prepares),
            "commit_count": len(tail_commits),
            "actor_committed_steps": {actor: lengths[actor] for actor in sorted(lengths)},
            "barrier_count": len(tail_barriers),
            "barrier_actors": sorted(barrier_actors),
            "chunk_count": len(tail_chunks),
            "chunk_sha256s": sorted(row["sha256"] for row in tail_chunks),
            "chunk_refs": [
                {key: row[key] for key in ("chunk_id", "path", "size_bytes", "sha256")}
                for row in tail_chunks
            ],
            "learner_jobs": 0,
            "candidate_checkpoint": None,
            "disposition": TAIL_DISPOSITION,
            "physical_outcome": (
                "no_durable_effects" if not tail_prepares
                else "one_native_owner_sealed_peer_partial" if tail_barriers
                else "partial_unsealed"
            ),
            "replay_allowed": False,
        }
        report = {
            "schema": RECEIPT_SCHEMA,
            "run_id": run_id,
            "round_id": round_id,
            "source_sha256": reconciliation["source_sha256"],
            "completed_cohorts": reconciliation["completed_cohorts"],
            "completed_updates": updates,
            "next_iteration": failed_iteration,
            "next_cohort_index": reconciliation["failed_cohort_index"],
            "parent_checkpoint_sha256": checkpoints[-1],
            "parent_optimizer_global_step": optimizer_step,
            "parent_learning_rate": config["student"]["learning_rate"],
            "fixed_heldout_manifest": reconciliation["heldout_manifest"],
            "fixed_heldout_admission": reconciliation["heldout_admission"],
            "mse_history_iterations": [row["iteration"] for row in selection["history"]],
            "training_provenance_counts": {
                key: len(computed_provenance[key])
                for key in ("chunk_sha256s", "source_episode_ids", "source_window_ids")
            },
            "imported_table_counts": {key: len(value) for key, value in tables.items()},
            "excluded_tail": tail,
            "source_database": source_ref,
            "source_database_unchanged": True,
            "old_namespace_preserved": True,
            "fresh_native_runtime_required": True,
            "native_replay_allowed": False,
            "allocation_retained": True,
            "compute_release_requested": False,
            "services_started": False,
            "native_work_started": False,
        }
    require(stable_file_ref(Path(source_ref["path"])) == source_ref, "closed source changed after inspection")
    return {"report": report, "tables": tables}


def _insert_rows(db, table, rows):
    if not rows:
        return
    columns = list(rows[0])
    require(all(list(row) == columns for row in rows), f"{table} row columns changed")
    sql = "INSERT INTO " + table + " (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")"
    db.executemany(sql, [tuple(row[column] for column in columns) for row in rows])


def _publish_exclusive(path, value):
    raw = canonical(value) + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return hashlib.sha256(raw).hexdigest()


def import_prefix(reconciliation, destination: Path):
    """Create a fresh owner-host ledger; never alter or attach the source."""
    destination = Path(destination).absolute()
    require(destination.parent.is_dir() and not destination.exists(), "fresh destination ledger namespace required")
    snapshot = inspect_prefix(reconciliation)
    source_before = stable_file_ref(Path(reconciliation["closed_database"]["path"]))
    stage = destination.parent / ("." + destination.name + ".import-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    state = StateStore(stage)
    tables = {
        name: [dict(row) for row in rows]
        for name, rows in snapshot["tables"].items()
    }
    tables["checkpoints"], checkpoint_materialization = _materialize_checkpoints(
        stage, destination, tables["checkpoints"]
    )
    active_sha = snapshot["report"]["parent_checkpoint_sha256"]
    checkpoint_materialization["active_checkpoint"] = _validate_training_checkpoint(
        content_addressed_checkpoint(stage, active_sha),
        active_sha,
        snapshot["report"]["parent_optimizer_global_step"],
        snapshot["report"]["parent_learning_rate"],
    )
    checkpoint_materialization["active_checkpoint"]["path"] = str(
        destination / "checkpoints" / f"{active_sha}.pt"
    )
    order = (
        "checkpoints", "runs", "round_iterations", "chunk_reservations", "chunk_provenance",
        "iteration_barriers", "learner_jobs", "learner_job_iterations", "mse_round_protocols",
        "step_prepares", "step_commits",
    )
    with state.transaction() as db:
        for table in order:
            _insert_rows(db, table, tables[table])
        require(not db.execute("PRAGMA foreign_key_check").fetchall(), "imported ledger foreign-key check failed")
        protocol = json.loads(snapshot["tables"]["mse_round_protocols"][0]["protocol_json"])
        provenance = _training_provenance(db, reconciliation["run_id"], reconciliation["round_id"], protocol)
        expected = json.loads(snapshot["tables"]["mse_round_protocols"][0]["training_provenance_json"])
        require(provenance == expected, "imported training provenance changed")
    database = stage / "state.sqlite3"
    require(sqlite3.connect(database).execute("PRAGMA quick_check").fetchone()[0] == "ok",
            "imported ledger failed quick_check")
    final_database = destination / "state.sqlite3"
    report = dict(snapshot["report"])
    report.update(
        destination_root=str(destination),
        destination_database=stable_file_ref(database, recorded_path=final_database),
        destination_owner=json.loads((stage / "ledger-owner.json").read_bytes()),
        checkpoint_materialization=checkpoint_materialization,
        import_complete=True,
    )
    _publish_exclusive(stage / "cohort-prefix-import.json", report)
    require(stable_file_ref(Path(reconciliation["closed_database"]["path"])) == source_before,
            "source database changed during import")
    os.rename(stage, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    require(stable_file_ref(final_database) == report["destination_database"],
            "destination ledger changed during publication")
    for row in tables["checkpoints"]:
        resolved = content_addressed_checkpoint(destination, row["sha256"])
        require(str(resolved) == row["path"]
                and stable_file_ref(resolved)["sha256"] == row["sha256"],
                "published checkpoint registry is not content-addressed/loadable")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--reconciliation-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(HEX64.fullmatch(args.reconciliation_sha256) is not None, "reconciliation SHA must be lowercase hex")
    require(binding(args.reconciliation)["raw_sha256"] == args.reconciliation_sha256,
            "reconciliation bytes changed")
    reconciliation = json.loads(args.reconciliation.read_bytes())
    if args.execute:
        result = import_prefix(reconciliation, args.destination)
    else:
        result = inspect_prefix(reconciliation)["report"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
