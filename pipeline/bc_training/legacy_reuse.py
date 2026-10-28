"""Explicit legacy-training reuse; never synthesize missing native provenance.

Planning is metadata-only and cannot admit training. Final admission rederives
the 2000-training/320-fresh-validation population, native execution, rendering
and separate 295-case test bindings. The strict fresh-population gate is not
modified. A legacy admission is deliberately not a full295 training certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from pipeline.bc_training.production_protocol import (
    bound, canonical, read_metadata, sha, valid_sha, validate_bounded_capacity,
    NATIVE_RESET_PINS, NATIVE_COLLECTOR_SHA256,
)
from pipeline.legacy.bc_collection.artifact_roles import panel_reference

PLAN_SCHEMA = "roborender.bc-legacy-reuse-plan.v1"
ADMISSION_SCHEMA = "roborender.bc-legacy-reuse-admission.v1"
POOL_SCHEMA = "roborender.bc-legacy-reuse-population.v1"
LEGACY_SHA256 = "67f2652b1d1a8ade14dd2e2a33d98032f3e6bf63ab8f6e2c62a63e19f8fa68ef"
PARENT_SHA256 = "4045cf241f0f5de13b5eec7f8f4c9d1e27187525fb6aae7b9840aa9d8e223b54"
BOOTSTRAP_SHA256 = "720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a"
COUNTS = {"train": 2000, "validation": 320}
UNKNOWN_SEED_COUNT = 490
UNKNOWN_FIELDS = ("motion_id", "motion_sha256", "episode_id", "request_sha256", "sampling_plan_sha256")
SOURCE_FIELDS = ("chunk_id", "frames", "source_sha256", "source_size_bytes",
                 "source_run_tag", "rank", "local_env_id", "global_env_id", "sequence_id",
                 "collection_chunk_id", "collection_filename", "simulator_seed", "source_batch")
MODEL_FIELDS = {"teacher": "teacher", "homie_walk": "walk", "homie_stand": "stand"}
LIMITATIONS = [
    "Legacy training motion/episode/reset coverage is unknown; full295 training coverage is not certified.",
    "490 historical process seeds are unknown; exact reset nonduplication cannot be proved retrospectively.",
    "Fresh all295 validation support is a target benchmark, not proof of matching legacy state/phase frequency.",
    "Historical/current simulator physics equivalence is not established.",
    "Only independent three-view renders are admitted; historical six-clip RGB is not relabeled.",
]


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "raw_sha256": sha(read_metadata(path))}


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("canonical absolute path required")
    return path.resolve()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def historical_population(legacy_manifest, original_manifest, original_seal, *, dataset_root):
    """Preserve actual old identities, never infer a motion/episode from env ID."""
    if legacy_manifest.get("raw_sha256") != LEGACY_SHA256 or original_manifest.get("raw_sha256") != PARENT_SHA256:
        raise ValueError("only the separately reviewed immutable historical 2000-chunk pool is eligible")
    legacy, parent, seal = map(bound, (legacy_manifest, original_manifest, original_seal))
    if (legacy.get("schema") != "roborender.bc_pool.autoregressive.v1"
            or legacy.get("status") != "sealed" or legacy.get("frames_per_chunk") != 81
            or legacy.get("chunk_count") != COUNTS["train"]
            or legacy.get("parent_manifest_sha256") != PARENT_SHA256
            or parent.get("schema") != "roborender.bc_unified_pool.v2"
            or parent.get("chunk_count") != COUNTS["train"]
            or seal.get("schema") != "roborender.bc_unified_pool_seal.v2"
            or seal.get("verdict") != "PASS" or seal.get("manifest_sha256") != PARENT_SHA256
            or seal.get("isaac_chunks") != COUNTS["train"] or seal.get("frames_per_chunk") != 81
            or legacy.get("scientific_identity") != parent.get("scientific_identity")):
        raise ValueError("historical source/seal/identity chain differs")
    rows, originals = legacy.get("chunks", []), parent.get("chunks", [])
    if (len(rows) != COUNTS["train"] or len(originals) != len(rows)
            or len({r["chunk_id"] for r in rows}) != len(rows)
            or len({r["source_sha256"] for r in rows}) != len(rows)):
        raise ValueError("complete unique historical population required")
    before = {r["chunk_id"]: r for r in originals}
    if set(before) != {r["chunk_id"] for r in rows}:
        raise ValueError("historical source population differs")
    root = _absolute(dataset_root)
    sources, seeds, process_ids = [], set(), set()
    unknown_seeds = 0
    for row in rows:
        if (any(row.get(k) != before[row["chunk_id"]].get(k) for k in SOURCE_FIELDS)
                or row.get("frames") != 81 or not valid_sha(row.get("source_sha256"))
                or not _integer(row.get("source_size_bytes"), 1)
                or any(row.get(k) is not None for k in UNKNOWN_FIELDS)):
            raise ValueError("legacy provenance changed or missing native history was fabricated")
        seed = row.get("simulator_seed")
        if seed is None:
            unknown_seeds += 1
        elif not _integer(seed) or seed >= 2**31:
            raise ValueError("historical simulator seed is malformed")
        else:
            seeds.add(seed)
        relative = Path(row["isaac_npz"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("historical source path must remain inside its dataset")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("historical source path escapes dataset")
        sources.append({"chunk_id": row["chunk_id"], "path": str(path),
                        "sha256": row["source_sha256"], "bytes": row["source_size_bytes"]})
        process_ids.add((row["source_run_tag"], row["rank"]))
    if unknown_seeds != UNKNOWN_SEED_COUNT:
        raise ValueError("historical unknown-seed count changed")
    hashes = legacy["scientific_identity"]["hashes"]
    models = {key: hashes[value] for key, value in MODEL_FIELDS.items()}
    if any(not valid_sha(v) for v in models.values()) or hashes.get("trainer") != NATIVE_COLLECTOR_SHA256:
        raise ValueError("historical teacher/Homie/native writer identities are missing")
    return dict(chunks=rows, source_records=sources,
        source_records_sha256=sha(canonical(sources)), known_process_seeds=sorted(seeds),
        unknown_process_seed_chunks=unknown_seeds, native_process_count=len(process_ids),
        unknown_native_fields={k: len(rows) for k in UNKNOWN_FIELDS}, model_sha256=models,
        historical_scientific_identity=legacy["scientific_identity"],
        payload_hashes_freshly_verified=False, payloads_opened=False)


def build_plan(*, run_name, legacy_manifest, original_manifest, original_seal,
               legacy_dataset_root, fresh_validation_plan, closed_loop_test_panel,
               bootstrap_spec, bootstrap_path, dataset_root):
    """Freeze intent; the returned plan cannot authorize training."""
    if (not isinstance(run_name, str) or not run_name or "/" in run_name
            or run_name.startswith("smoke-") or "legacy" not in run_name):
        raise ValueError("explicit non-smoke legacy-reuse run name required")
    old = historical_population(legacy_manifest, original_manifest, original_seal,
                               dataset_root=legacy_dataset_root)
    from pipeline.legacy.bc_collection.fresh_validation import validate_plan as validate_fresh_plan
    fresh = bound(fresh_validation_plan)
    validate_fresh_plan(fresh)
    if (fresh.get("legacy_manifest") != legacy_manifest
            or fresh.get("closed_loop_test_panel") != closed_loop_test_panel):
        raise ValueError("fresh validation references a different legacy population or final panel")
    partitions = fresh.get("partitions", {})
    cohorts = partitions.get("validation", {}).get("cohorts", [])
    if (set(partitions) != {"validation"} or fresh.get("total_target_chunks") != COUNTS["validation"]
            or not cohorts):
        raise ValueError("fresh plan must contain exactly 320 validation-only chunks")
    seeds = [c["process_seed"] for c in cohorts]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(old["known_process_seeds"]):
        raise ValueError("fresh validation process seeds overlap known training or one another")
    motions = {s["motion_id"]: s["motion_sha256"] for c in cohorts for s in c["slots"]}
    if len(motions) != 295:
        raise ValueError("fresh validation must cover the frozen 295-motion target")
    panel = panel_reference(closed_loop_test_panel, inventory_sha256=fresh["motion_inventory_sha256"],
        motion_bindings=motions, forbidden_seeds=seeds + old["known_process_seeds"])
    if panel["episode_count"] != 295 or set(panel["motion_case_counts"].values()) != {1}:
        raise ValueError("independent final test requires exactly one case for each of295 motions")
    spec = bound(bootstrap_spec)
    if (spec.get("checkpoint_sha256") != BOOTSTRAP_SHA256
            or spec.get("input_contract", {}).get("views") != ["head_color", "hand_left", "hand_right"]
            or spec.get("output_contract", {}).get("train_action_slice") != [0, 11]):
        raise ValueError("clean immutable three-view bootstrap required, not a trained BC checkpoint")
    return dict(schema=PLAN_SCHEMA, purpose="legacy_training_reuse", run_name=run_name,
        legacy_manifest=legacy_manifest, original_manifest=original_manifest, original_seal=original_seal,
        legacy_dataset_root=str(_absolute(legacy_dataset_root)), dataset_root=str(_absolute(dataset_root)),
        fresh_validation_plan=fresh_validation_plan, closed_loop_test_panel=closed_loop_test_panel,
        bootstrap_spec=bootstrap_spec, bootstrap_path=str(_absolute(bootstrap_path)),
        bootstrap_sha256=BOOTSTRAP_SHA256, initialization="clean_bootstrap_only",
        resume_policy="same_admission_and_sampling_identity_only",
        counts=COUNTS, training_chunk_batch_size=16, training_batches_per_epoch=125,
        all_old_split_assignments_retired=True, legacy_assignment="retrospective_train_only",
        historical={k: v for k, v in old.items() if k != "chunks"},
        fresh_validation_seeds=sorted(seeds), target_motion_bindings=motions,
        closed_loop_test=panel, renderer_recipe="raven_3view_50step_independent_v2",
        limitations=LIMITATIONS, planning_only=True, production_admitted=False,
        full295_training_coverage_certified=False)


def validate_plan(plan):
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("explicit legacy-reuse plan schema required")
    keys = ("run_name", "legacy_manifest", "original_manifest", "original_seal", "legacy_dataset_root",
            "fresh_validation_plan", "closed_loop_test_panel", "bootstrap_spec", "bootstrap_path", "dataset_root")
    if plan != build_plan(**{k: plan[k] for k in keys}):
        raise ValueError("legacy-reuse plan differs from rederived pinned inputs")
    return plan


def _native_execution(evidence, fresh_plan, models):
    """Reapply native ingestion/capacity/lifecycle gates, not a report PASS flag."""
    from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
    cohorts = [{"document": bound(b), "sha256": b["raw_sha256"]} for b in evidence["cohort_manifests"]]
    transfers = [{"document": bound(b), "sha256": b["raw_sha256"]} for b in evidence["transfer_receipts"]]
    source = bound(evidence["source_pool"])
    rebuilt, _ = build_pool(fresh_plan, cohorts, transfers,
        dataset_root=Path(evidence["source_pool"]["path"]).parent, expected_total=320,
        verify_destination_stats=True)
    if source != rebuilt or {k: a["sha256"] for k, a in source["model_assets"].items()} != models:
        raise ValueError("fresh validation does not reproduce exact native evidence/model identities")
    execution = evidence["collector_execution"]
    if (set(execution) != {"adapter_source_sha256", "source_release_sha256", "container_image_id",
                         "native_runtime_version", "lifecycle_recipe", "memory_buffer_recipe", "runtime_contract_sha256"}
            or any(not valid_sha(execution.get(k)) for k in ("adapter_source_sha256", "source_release_sha256", "runtime_contract_sha256"))
            or not isinstance(execution.get("container_image_id"), str)
            or not execution["container_image_id"].startswith("sha256:")
            or not valid_sha(execution["container_image_id"][7:])
            or not isinstance(execution.get("native_runtime_version"), str) or not execution["native_runtime_version"]
            or execution.get("lifecycle_recipe") != "one_kit_fresh_seeded_scene_per_declared_cohort_v1"
            or execution.get("memory_buffer_recipe") != "original_raven_capture_lists_and_writer_v1"):
        raise ValueError("complete native execution/source/runtime contract required")
    sessions = [bound(b) for b in evidence["session_execution_receipts"]]
    by_session = {sha(canonical(s)): s for s in sessions}
    attestations = [bound(b) for b in evidence["cohort_execution_receipts"]]
    by_cohort = {a.get("cohort_manifest_raw_sha256"): a for a in attestations}
    if (not sessions or len(by_session) != len(sessions) or len(by_cohort) != len(cohorts)
            or len(attestations) != len(cohorts)):
        raise ValueError("unique completed native session/cohort execution evidence required")
    for s in sessions:
        identities = s.get("request_identities", {})
        if (s.get("schema") != "roborender.bc-session-execution.v1" or s.get("verdict") != "PASS"
                or s.get("native_initializations") != 1 or s.get("collector_execution") != execution
                or not _integer(s.get("scene_initializations"), 1)
                or len(identities) != s["scene_initializations"]
                or any(not valid_sha(k) or not valid_sha(v) for k, v in identities.items())
                or any(not valid_sha(s.get(k)) for k in ("source_manifest_raw_sha256", "port_manifest_raw_sha256",
                    "launch_raw_sha256", "session_sha256", "session_canonical_sha256",
                    "native_result_raw_sha256", "host_result_raw_sha256"))):
            raise ValueError("fresh native session execution/source identity differs")
    for envelope in cohorts:
        doc = envelope["document"]
        a = by_cohort.get(envelope["sha256"], {})
        s = by_session.get(a.get("session_execution_canonical_sha256"), {})
        if (a.get("schema") != "roborender.bc-cohort-execution.v1" or a.get("verdict") != "PASS"
                or a.get("collector_execution") != execution or a.get("request_sha256") != doc["request_sha256"]
                or a.get("request_canonical_sha256") != sha(canonical(doc["request"]))
                or s.get("request_identities", {}).get(doc["request_sha256"]) != sha(canonical(doc["request"]))
                or a.get("actual_num_envs") != len(doc["request"]["motions"])
                or a.get("actual_vector_steps") != doc["vector_steps"]):
            raise ValueError("cohort native execution identity differs")
    counts = Counter(a["session_execution_canonical_sha256"] for a in attestations)
    if any(counts.get(k) != s["scene_initializations"] for k, s in by_session.items()):
        raise ValueError("session execution is incomplete")
    topology = evidence["execution_topology"]
    if (topology.get("host_id") != "leo-rtx-6k" or topology.get("concurrent_collectors") != 1
            or not _integer(topology.get("minimum_host_reserve_bytes"), 8 << 30)
            or not _integer(topology.get("cgroup_limit_bytes_per_collector"), 1)
            or topology["cgroup_limit_bytes_per_collector"] > 43 << 30):
        raise ValueError("fresh validation requires the explicitly qualified serial Leo memory topology")
    protocol = dict(model_sha256=models, collector_execution=execution, execution_topology=topology,
        capacity_equivalence=dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=0.10,
                                  absolute_growth_slack_bytes=256 << 20))
    receipts = [bound(b) for b in evidence["capacity_receipts"]]
    for r in receipts:
        if r.get("schema") != "roborender.native-bc-capacity.v2":
            raise ValueError("bounded full-buffer capacity v2 receipt required")
        validate_bounded_capacity(r, protocol)
    if {r["num_envs"] for r in receipts} != {len(e["document"]["request"]["motions"]) for e in cohorts}:
        raise ValueError("actual validation cohort sizes/tails lack exact bounded full-buffer qualification")
    eligibility = bound(evidence["native_motion_eligibility"])
    actual_motions = {s["motion_id"]: s["motion_sha256"]
                     for c in fresh_plan["partitions"]["validation"]["cohorts"] for s in c["slots"]}
    eligible = eligibility.get("motions", [])
    if (eligibility.get("schema") != "roborender.native-motion-eligibility.cpu.v1"
            or eligibility.get("motion_inventory_sha256") != fresh_plan["motion_inventory_sha256"]
            or eligibility.get("cpu_only") is not True or eligibility.get("cuda_initialized") is not False
            or eligibility.get("isaac_imported") is not False or eligibility.get("eligible_count") != 295
            or len(eligible) != 295 or {r["motion_id"]: r["sha256"] for r in eligible} != actual_motions
            or eligibility.get("native_source_hashes") != NATIVE_RESET_PINS
            or eligibility.get("invalid_or_short") != [] or eligibility.get("frame_count_mismatches") != []
            or any(r.get("eligible") is not True or r.get("frame_counts_match") is not True
                   or r.get("native_skips_short_motion") is not False
                   or not _integer(r.get("native_random_start_choices"), 1) for r in eligible)):
        raise ValueError("full target native motion eligibility evidence required")
    return source


def _rendered_population(evidence, expected_sources):
    from pipeline.bc_training.render_execution import finish
    # Calling the genuine producer again checks actual lane/source/output/file
    # identities. This reads small JSON and stat() only, never rendered arrays.
    protocol = bound(evidence["protocol"])
    bound(evidence["render_plan"])
    rendered = bound(evidence["rendered_manifest"])
    receipt = bound(evidence["execution_receipt"])
    actual = finish(evidence["protocol"], evidence["render_plan"]["path"], evidence["rendered_manifest"]["path"])
    if receipt != actual or actual["source_pool_raw_sha256"] != protocol["source_pool"]["raw_sha256"]:
        raise ValueError("independent render receipt differs from genuine execution")
    rows = rendered.get("chunks", [])
    if (rendered.get("schema") != "roborender.bc_pool.rendered.v2"
            or rendered.get("render_contract") != "raven_three_view_independent_v2"
            or rendered.get("render_recipe") != "raven_3view_50step_independent_v2"
            or rendered.get("max_chain_chunks") != 1 or rendered.get("frames_per_chunk") != 81
            or len(rows) != len(expected_sources)
            or len({r["chunk_id"] for r in rows}) != len(rows)
            or len({r["source_sha256"] for r in rows}) != len(rows)
            or len({r["rendered_sha256"] for r in rows}) != len(rows)
            or {r["source_sha256"] for r in rows} != set(expected_sources)):
        raise ValueError("rendered population/recipe differs from exact selected sources")
    for row in rows:
        if (row.get("frames") != 81 or row.get("render_chain_length") != 1
                or row.get("render_chain_index") != 0 or row.get("render_inherits_previous_reference") is not False
                or not valid_sha(row.get("rendered_sha256"))):
            raise ValueError("independent per-chunk render lineage missing")
        original = expected_sources[row["source_sha256"]]
        if "sampling_plan_sha256" not in original and any(row.get(k) is not None for k in UNKNOWN_FIELDS):
            raise ValueError("renderer fabricated missing historical native provenance")
        fields = SOURCE_FIELDS if "sampling_plan_sha256" not in original else tuple(
            k for k in original if k not in {"isaac_npz", "rendered_npz"})
        if any(row.get(k) != original.get(k) for k in fields):
            raise ValueError("rendering changed recorded source/native provenance")
    return rendered


def assemble_population(plan, native_evidence, rendering):
    """Return a combined manifest; does not write/admit any data by itself."""
    validate_plan(plan)
    old = historical_population(plan["legacy_manifest"], plan["original_manifest"], plan["original_seal"],
                               dataset_root=plan["legacy_dataset_root"])
    fresh = _native_execution(native_evidence, bound(plan["fresh_validation_plan"]), old["model_sha256"])
    if set(rendering) != {"train", "validation"}:
        raise ValueError("exact train and validation render evidence required")
    by_role = {"train": old["chunks"], "validation": fresh["chunks"]}
    overlap = set(r["source_sha256"] for r in by_role["train"]) & set(r["source_sha256"] for r in by_role["validation"])
    if overlap:
        raise ValueError("fresh validation reuses a historical source payload")
    old_processes = {(r["source_run_tag"], r["rank"]) for r in old["chunks"]}
    if any((r["source_run_tag"], r["rank"]) in old_processes for r in fresh["chunks"]):
        raise ValueError("fresh validation reuses historical native process identity")
    output, rendered_hashes = [], set()
    root = Path(plan["dataset_root"])
    for role in ("train", "validation"):
        expected_source = plan["original_manifest"] if role == "train" else native_evidence["source_pool"]
        if bound(rendering[role]["protocol"]).get("source_pool") != expected_source:
            raise ValueError("renderer protocol source pool is not the exact admitted role source")
        rows = _rendered_population(rendering[role], {r["source_sha256"]: r for r in by_role[role]})["chunks"]
        manifest_path = Path(rendering[role]["rendered_manifest"]["path"])
        rendered_doc = bound(rendering[role]["rendered_manifest"])
        for row in rows:
            path = (Path(rendered_doc["rendered_pool"]) / Path(row["rendered_npz"]).name).resolve(strict=True)
            if (not path.is_relative_to(root) or not path.is_file()
                    or path.stat().st_size != row["rendered_size_bytes"] or row["rendered_sha256"] in rendered_hashes):
                raise ValueError("rendered path/size/duplicate identity differs")
            rendered_hashes.add(row["rendered_sha256"])
            output.append(dict(chunk_id=role + "-" + row["chunk_id"], original_chunk_id=row["chunk_id"],
                split=role, rendered_npz=str(path.relative_to(root)), source_sha256=row["source_sha256"],
                rendered_sha256=row["rendered_sha256"], frames=81,
                stream_id=sha(canonical([row["source_run_tag"], row["rank"], row["local_env_id"]])),
                provenance_origin="historical_training_unknown_native_fields" if role == "train" else "fresh_native_validation",
                source_rendered_manifest=str(manifest_path),
                source_rendered_manifest_sha256=rendering[role]["rendered_manifest"]["raw_sha256"]))
    if Counter(r["split"] for r in output) != COUNTS:
        raise ValueError("exact legacy training/fresh validation counts required")
    return dict(schema=POOL_SCHEMA, plan_canonical_sha256=sha(canonical(plan)),
        chunk_count=sum(COUNTS.values()), frames_per_chunk=81, counts=COUNTS, chunks=output,
        closed_loop_test=plan["closed_loop_test"], limitations=LIMITATIONS,
        full295_training_coverage_certified=False, payloads_opened=False)


def split_documents(population, manifest_binding, *, split_dir):
    """Learner-compatible exact indexes over a shared combined metadata manifest."""
    if population.get("schema") != POOL_SCHEMA or population.get("counts") != COUNTS:
        raise ValueError("legacy combined population schema/counts required")
    return {role: dict(schema="roborender.bc_split.v2", split=role, chunk_count=COUNTS[role],
        source_manifest=os.path.relpath(manifest_binding["path"], _absolute(split_dir)),
        source_manifest_sha256=manifest_binding["raw_sha256"],
        items=[r for r in population["chunks"] if r["split"] == role]) for role in COUNTS}


def audit_admission(*, plan_binding, native_evidence, rendering, population_binding, split_files):
    plan = validate_plan(bound(plan_binding))
    actual = assemble_population(plan, native_evidence, rendering)
    if bound(population_binding) != actual or set(split_files) != set(COUNTS):
        raise ValueError("combined legacy population differs from actual evidence")
    for role, b in split_files.items():
        expected = split_documents(actual, population_binding, split_dir=Path(b["path"]).parent)[role]
        if bound(b) != expected:
            raise ValueError("legacy reuse split differs from exact admitted population")
    return dict(schema=ADMISSION_SCHEMA, purpose="legacy_training_reuse", run_name=plan["run_name"],
        legacy_reuse_admitted=True, production_admitted=False, full295_training_coverage_certified=False,
        plan=plan_binding, native_evidence=native_evidence, rendering=rendering,
        source_manifest=population_binding["path"], source_manifest_sha256=population_binding["raw_sha256"],
        split_files=split_files, counts=COUNTS, limitations=LIMITATIONS,
        initialization="clean_bootstrap_only", resume_policy="same_admission_and_sampling_identity_only",
        bootstrap_path=plan["bootstrap_path"],
        bootstrap_sha256=plan["bootstrap_sha256"], bootstrap_spec=plan["bootstrap_spec"],
        closed_loop_test=plan["closed_loop_test"], payloads_opened=False, closed_loop_test_executed=False)


def validate_training_admission(report_path, train_path, validation_path, *, allow_legacy_reuse,
                                run_name, bootstrap_path, policy_spec_path, resume_checkpoint=None,
                                resume_validation_state=None, dataset_root=None):
    if not allow_legacy_reuse or (resume_validation_state is not None and resume_checkpoint is None):
        raise ValueError("legacy reuse requires explicit opt-in and clean bootstrap or exact-protocol resume")
    report = json.loads(read_metadata(report_path))
    if (report.get("schema") != ADMISSION_SCHEMA or not isinstance(run_name, str) or not run_name
            or (resume_checkpoint is None and report.get("run_name") != run_name)
            or set(report.get("split_files", {})) != set(COUNTS)
            or Path(report["split_files"]["train"]["path"]).resolve() != Path(train_path).resolve()
            or Path(report["split_files"]["validation"]["path"]).resolve() != Path(validation_path).resolve()
            or Path(report["bootstrap_path"]).resolve() != Path(bootstrap_path).resolve()
            or binding(policy_spec_path) != report.get("bootstrap_spec")):
        raise ValueError("legacy training invocation differs from its immutable admission")
    plan = bound(report["plan"])
    if dataset_root is None or Path(dataset_root).resolve() != Path(plan["dataset_root"]):
        raise ValueError("legacy training dataset root differs")
    actual = audit_admission(plan_binding=report["plan"], native_evidence=report["native_evidence"],
        rendering=report["rendering"], population_binding={"path": report["source_manifest"],
        "raw_sha256": report["source_manifest_sha256"]}, split_files=report["split_files"])
    if report != actual:
        raise ValueError("legacy admission differs from rederived execution evidence")
    # A resumed attempt may have a fresh output/run name. The trainer's later
    # validate_resume_identity plus existing sampling/RNG/optimizer checks bind
    # it to this exact admission; the display/output name is not data identity.
    # The trainer's existing bootstrap SHA check verifies the actual model bytes
    # once before loading. This metadata gate never rereads a 138MB checkpoint.
    return report


def validate_resume_identity(checkpoint, expected_admission):
    """Same-dataset continuation is allowed; old or other-data BC is not."""
    previous = checkpoint.get("data_admission", {})
    if (previous.get("schema") != ADMISSION_SCHEMA
            or previous.get("legacy_reuse_admitted") is not True
            or previous.get("report_sha256") != expected_admission.get("report_sha256")
            or not valid_sha(previous.get("report_sha256"))):
        raise ValueError("legacy reuse resume must originate from the exact same admitted dataset/protocol")


def publish_admission(*, plan_binding, native_evidence, rendering, output_dir):
    """Publish only a fresh metadata bundle after all actual evidence passes.

    The original data/manifests remain untouched. An interrupted partial output
    directory is preserved and never implicitly adopted or overwritten.
    """
    from pipeline.deployment.rtx_actor.isaac6_bc_production import publish_json
    root = _absolute(output_dir)
    if root.exists():
        raise FileExistsError("fresh legacy admission output directory required")
    plan = validate_plan(bound(plan_binding))
    population = assemble_population(plan, native_evidence, rendering)
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "population.json"
    publish_json(manifest_path, population)
    manifest_binding = binding(manifest_path)
    documents = split_documents(population, manifest_binding, split_dir=root)
    splits = {}
    for role, document in documents.items():
        path = root / (role + ".json")
        publish_json(path, document)
        splits[role] = binding(path)
    report = audit_admission(plan_binding=plan_binding, native_evidence=native_evidence,
        rendering=rendering, population_binding=manifest_binding, split_files=splits)
    path = root / "data-admission.json"
    publish_json(path, report)
    return dict(admission=binding(path), split_files=splits, counts=COUNTS,
                legacy_reuse_admitted=True, production_admitted=False,
                full295_training_coverage_certified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-inputs", type=Path,
                        help="Small JSON containing build_plan keyword arguments; no workload is launched")
    group.add_argument("--admission-inputs", type=Path,
        help="Small JSON containing plan_binding/native_evidence/rendering; requires completed actual evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.admission_inputs is not None:
        result = publish_admission(**json.loads(read_metadata(args.admission_inputs)), output_dir=args.output)
        print(json.dumps(result, sort_keys=True))
        return
    plan = build_plan(**json.loads(read_metadata(args.plan_inputs)))
    with args.output.open("xb") as stream:
        stream.write(canonical(plan))
    print(json.dumps({"path": str(args.output.resolve()), "raw_sha256": sha(canonical(plan)),
                      "planning_only": True, "production_admitted": False}))


if __name__ == "__main__":
    main()
