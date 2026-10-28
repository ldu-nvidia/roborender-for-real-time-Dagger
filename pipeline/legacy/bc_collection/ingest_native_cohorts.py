"""Metadata-only publication of fresh native BC cohorts after verified transfer.

No source NPZ is opened, copied, renamed, or rehashed here. Immutable cohort
manifests describe native outputs; separate transfer receipts attest identical
bytes on H200. This verifies exact sampling-plan membership and per-lane quotas,
then selects only scheduled chunks. Extra native chunks stay in the audit and
are not silently admitted to any train/validation/test partition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    SCHEMA as REQUEST_SCHEMA, SCHEMA_V2 as REQUEST_SCHEMA_V2, NATIVE_COLLECTOR_SHA256, canonical, publish_json,
    source_run_tag, validate_collection_request, selected_chunk_ordinals,
)
from pipeline.legacy.bc_collection.sampling_plan import SCHEMA as PLAN_SCHEMA, SCHEMA_V2 as PLAN_SCHEMA_V2, SCHEMA_V3 as PLAN_SCHEMA_V3, validate_selection_plan
from pipeline.legacy.bc_collection.artifact_roles import offline_roles
from pipeline.legacy.bc_collection.fresh_validation import (
    SCHEMA as FRESH_VALIDATION_SCHEMA, validate_plan as validate_fresh_validation,
)


def read_document(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "document": json.loads(raw)}


def build_pool(plan, cohorts, transfers, *, dataset_root, expected_total=3200,
               verify_destination_stats=True):
    """Return (source manifest, source seal), touching only destination stat()."""
    if (type(expected_total) is not int or expected_total <= 0
            or (plan.get("schema") != PLAN_SCHEMA_V3 and expected_total % 10)
            or plan.get("schema") not in {PLAN_SCHEMA, PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, FRESH_VALIDATION_SCHEMA}
            or plan.get("total_target_chunks") != expected_total):
        raise ValueError("unexpected full sampling-plan size/schema")
    fresh_validation = plan["schema"] == FRESH_VALIDATION_SCHEMA
    if fresh_validation:
        validate_fresh_validation(plan)
    else:
        validate_selection_plan(plan)
    explicit_ordinals = plan["schema"] != PLAN_SCHEMA
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    roles = ("validation",) if fresh_validation else offline_roles(plan)
    expected_roles = ({"validation": 320} if fresh_validation else
                      plan["artifact_roles"]["offline_chunk_counts"] if plan["schema"] == PLAN_SCHEMA_V3 else
                      dict(zip(roles, (expected_total * 8 // 10, expected_total // 10, expected_total // 10))))
    planned, motion_bindings = {}, {}
    for role in roles:
        partition = plan["partitions"][role]
        if partition["target_chunks"] != expected_roles[role]:
            raise ValueError("sampling-plan role quota is not 80/10/10")
        for cohort in partition["cohorts"]:
            if cohort["cohort_id"] in planned or cohort["split"] != role:
                raise ValueError("duplicate/misassigned sampling-plan cohort")
            planned[cohort["cohort_id"]] = cohort
            for slot in cohort["slots"]:
                prior = motion_bindings.setdefault(slot["motion_id"], slot["motion_sha256"])
                if prior != slot["motion_sha256"]:
                    raise ValueError("one motion ID has different bytes across role cohorts")
    by_cohort, by_transfer = {}, {}
    for envelope in cohorts:
        doc = envelope["document"]
        key = doc["request"]["cohort_id"]
        if key in by_cohort:
            raise ValueError("duplicate cohort manifest")
        by_cohort[key] = envelope
    for envelope in transfers:
        doc = envelope["document"]
        key = doc["cohort_id"]
        if key in by_transfer:
            raise ValueError("duplicate cohort transfer receipt")
        by_transfer[key] = envelope
    if set(by_cohort) != set(planned) or set(by_transfer) != set(planned):
        raise ValueError("all and only planned cohorts/verified transfers are required")

    root = Path(dataset_root).resolve()
    source_root = root / "isaac"
    selected, overcollection, cohort_audits = [], [], []
    streams, all_source_hashes, destination_names, seeds = set(), set(), set(), set()
    chunk_keys, episode_ends = set(), {}
    common_model_assets = None
    for batch, (cohort_id, expected) in enumerate(planned.items()):
        envelope, transfer_envelope = by_cohort[cohort_id], by_transfer[cohort_id]
        doc, transfer = envelope["document"], transfer_envelope["document"]
        request = validate_collection_request(doc["request"])
        requested_ordinals = selected_chunk_ordinals(request)
        slots = expected["slots"]
        model_assets = doc.get("model_assets")
        if (not isinstance(model_assets, dict)
                or set(model_assets) != {"teacher", "homie_walk", "homie_stand"}
                or any(not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
                       or not isinstance(asset["path"], str) or not Path(asset["path"]).is_absolute()
                       or not isinstance(asset["sha256"], str)
                       or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
                       for asset in model_assets.values())
                or doc.get("native_collector_sha256") != NATIVE_COLLECTOR_SHA256):
            raise ValueError("missing model pins or incorrect original native collector pin")
        if common_model_assets is None:
            common_model_assets = model_assets
        elif model_assets != common_model_assets:
            raise ValueError("teacher or Homie assets differ across collection cohorts")
        if (doc.get("schema") != "roborender.bc-native-cohort.v1" or doc.get("status") != "complete"
                or doc.get("optimizer_steps") != 0 or doc.get("teacher_only") is not True
                or doc.get("native_payload_writer_unchanged") is not True
                or doc["vector_steps"] <= 0 or doc["vector_steps"] % 81
                or doc["vector_steps"] > request["max_windows"] * 81
                or request["schema"] != (REQUEST_SCHEMA_V2 if explicit_ordinals else REQUEST_SCHEMA)
                or request["sampling_plan_sha256"] != plan_sha
                or request["split"] != expected["split"] or request["process_seed"] != expected["process_seed"]
                or len(request["motions"]) != expected["num_envs"]
                or request["chunks_per_env"] != [slot["target_chunks"] for slot in slots]
                or [(r["motion_id"], r["sha256"]) for r in request["motions"]]
                != [(s["motion_id"], s["motion_sha256"]) for s in slots]):
            raise ValueError("cohort identity, native completion, or sampling assignment differs")
        if explicit_ordinals and (
                requested_ordinals != [slot["selected_chunk_ordinals"] for slot in slots]
                or request["exposure_span_chunks_per_env"] != [slot["exposure_span_chunks"] for slot in slots]):
            raise ValueError("native request changed the frozen complete-chunk selection/exposure plan")
        if request["process_seed"] in seeds:
            raise ValueError("collection cohorts must use independent process seeds")
        seeds.add(request["process_seed"])
        if (transfer.get("schema") != "roborender.bc-cohort-transfer.v1"
                or transfer.get("verified") is not True
                or transfer.get("cohort_manifest_sha256") != envelope["sha256"]):
            raise ValueError("unverified transfer or cohort manifest hash mismatch")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (doc["request_sha256"], envelope["sha256"], transfer_envelope["sha256"])):
            raise ValueError("invalid cohort/request/transfer digest")
        moved = {}
        for row in transfer["files"]:
            if row["source_path"] in moved:
                raise ValueError("duplicate transferred source")
            moved[row["source_path"]] = row
        emitted, chosen = Counter(), Counter()
        for row in doc["chunks"]:
            lane = row["local_env_id"]
            if type(lane) is not int or not 0 <= lane < len(slots):
                raise ValueError("invalid native lane")
            slot = slots[lane]
            key = (row["source_run_tag"], row["rank"], lane, row["seq"])
            if (type(row["seq"]) is not int or row["seq"] < 0 or key in chunk_keys
                    or type(row["source_bytes"]) is not int or row["source_bytes"] <= 0
                    or not isinstance(row["source_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
                    or not Path(row["source_path"]).is_absolute()
                    or type(row["first_episode_step_index"]) is not int
                    or row["first_episode_step_index"] < 0):
                raise ValueError("invalid/duplicate native sequence, source or episode interval")
            chunk_keys.add(key)
            if (row["source_run_tag"] != source_run_tag(request) or row["rank"] != request["rank"]
                    or row["global_env_id"] != request["global_env_offset"] + lane
                    or row["motion_id"] != slot["motion_id"] or row["motion_sha256"] != slot["motion_sha256"]
                    or row["simulator_seed"] != request["process_seed"] or row["split"] != request["split"]
                    or row["sampling_plan_sha256"] != plan_sha or row["request_sha256"] != doc["request_sha256"]
                    or row["teacher_only"] is not True or row["frames"] != 81
                    or not isinstance(row["episode_id"], str) or not row["episode_id"]
                    or row["last_episode_step_index"] - row["first_episode_step_index"] != 80):
                raise ValueError("chunk native motion/episode/seed provenance differs")
            episode_key = (row["source_run_tag"], row["rank"], lane, row["episode_id"])
            if row["first_episode_step_index"] <= episode_ends.get(episode_key, -1):
                raise ValueError("native episode chunk intervals overlap or reverse")
            episode_ends[episode_key] = row["last_episode_step_index"]
            if ((request["schema"] == REQUEST_SCHEMA_V2 or "emitted_chunk_ordinal" in row)
                    and (type(row.get("emitted_chunk_ordinal")) is not int or row["emitted_chunk_ordinal"] != emitted[lane])):
                raise ValueError("native emitted chunk ordinal is missing or out of order")
            if request["schema"] == REQUEST_SCHEMA_V2 and row.get("selection_policy") != "explicit_preselected_ordinals_v2":
                raise ValueError("native chunk did not attest explicit ordinal selection")
            should_select = emitted[lane] in requested_ordinals[lane]
            if row["selected_for_split"] is not should_select or row["overcollection"] is not (not should_select):
                raise ValueError("only predeclared complete-chunk ordinals per lane may enter the pool")
            emitted[lane] += 1
            if not should_select:
                overcollection.append(row)
                continue
            chosen[lane] += 1
            try:
                receipt = moved[row["source_path"]]
            except KeyError as exc:
                raise ValueError("selected native chunk has not been transferred") from exc
            if (receipt["source_sha256"] != row["source_sha256"]
                    or receipt["destination_sha256"] != row["source_sha256"]
                    or receipt["source_bytes"] != row["source_bytes"]
                    or receipt["destination_bytes"] != row["source_bytes"]):
                raise ValueError("transfer did not attest byte-identical source/destination")
            destination = Path(receipt["destination_path"]).resolve()
            if destination.parent != source_root or destination.name != Path(row["source_path"]).name:
                raise ValueError("verified chunks must use the flat immutable Isaac pool with original filenames")
            if verify_destination_stats and (not destination.is_file() or destination.stat().st_size != row["source_bytes"]):
                raise ValueError("verified destination is missing or changed in size")
            if row["source_sha256"] in all_source_hashes or destination.name in destination_names:
                raise ValueError("duplicate source payload or destination name")
            all_source_hashes.add(row["source_sha256"])
            destination_names.add(destination.name)
            selected.append({**row, "chunk_id": f"bc_chunk_{len(selected):06d}", "source_batch": batch,
                "sequence_id": row["seq"], "source_size_bytes": row["source_bytes"],
                "collection_filename": destination.name, "isaac_npz": "isaac/" + destination.name,
                "rendered_npz": "rendered_independent_v2/" + destination.name,
                "cohort_manifest_sha256": envelope["sha256"], "transfer_receipt_sha256": transfer_envelope["sha256"]})
        for lane, quota in enumerate(request["chunks_per_env"]):
            key = (source_run_tag(request), request["rank"], lane)
            if chosen[lane] != quota or key in streams:
                raise ValueError("incomplete lane quota or reused collection stream")
            streams.add(key)
        if (doc["selected_chunks"] != sum(chosen.values())
                or doc["overcollection_chunks"] != sum(emitted.values()) - sum(chosen.values())
                or doc["emitted_per_lane"] != [emitted[i] for i in range(len(slots))]):
            raise ValueError("cohort summary does not match native output records")
        cohort_audits.append({"cohort_id": cohort_id, "manifest_sha256": envelope["sha256"],
                             "transfer_receipt_sha256": transfer_envelope["sha256"]})
    if len(selected) != expected_total or Counter(row["split"] for row in selected) != expected_roles:
        raise ValueError("complete source pool does not match exact 80/10/10 quotas")
    support = [{row["motion_id"] for row in selected if row["split"] == role} for role in roles]
    if not support[0] or any(modes != support[0] for modes in support[1:]):
        raise ValueError("all roles must cover the same declared motion population")
    manifest = {"schema": "roborender.bc_pool.native-cohorts.v1", "status": "sealed",
        "frames_per_chunk": 81, "chunk_count": len(selected), "sampling_plan_sha256": plan_sha,
        "partition_counts": expected_roles, "motion_count": len(support[0]), "chunks": selected,
        "model_assets": common_model_assets, "native_collector_sha256": NATIVE_COLLECTOR_SHA256,
        "motion_bindings": motion_bindings,
        "cohort_audit": cohort_audits, "overcollection_excluded": overcollection,
        "selection_policy": plan.get("selection_policy", "prefix_quota_v1"),
        "collection_cost": plan.get("collection_cost"),
        "payloads_read": False, "production_admitted": False}
    if plan["schema"] == PLAN_SCHEMA_V3:
        manifest["artifact_roles"] = plan["artifact_roles"]
        manifest["sampling_plan_schema"] = PLAN_SCHEMA_V3
    elif fresh_validation:
        manifest["sampling_plan_schema"] = FRESH_VALIDATION_SCHEMA
        manifest["collection_purpose"] = "fresh_validation_only"
        manifest["legacy_training_manifest"] = plan["legacy_manifest"]
        manifest["closed_loop_test_panel"] = plan["closed_loop_test_panel"]
    seal = {"schema": "roborender.bc-source-seal.v2", "verdict": "PASS",
        "isaac_chunks": len(selected), "rendered_chunks": 0,
        "sampling_plan_sha256": plan_sha, "payloads_read": False,
        "qualification_scope": "native provenance, exact role quotas, verified byte transfer; rendering/training not yet qualified"}
    return manifest, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cohort-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--transfer-receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, required=True,
                        help="explicit exact selected quota: 2880 for the full v3 production plan; smaller counts require a separate diagnostic plan")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.seal.exists():
        raise FileExistsError("immutable pool manifest/seal already exists")
    manifest, seal = build_pool(json.loads(args.plan.read_bytes()),
        [read_document(p) for p in args.cohort_manifests], [read_document(p) for p in args.transfer_receipts],
        dataset_root=args.dataset_root, expected_total=args.expected_total)
    publish_json(args.manifest, manifest)
    seal["manifest_sha256"] = read_document(args.manifest)["sha256"]
    publish_json(args.seal, seal)
    print(json.dumps({"manifest": str(args.manifest), "seal": str(args.seal),
                      "chunks": manifest["chunk_count"], "payloads_read": False}))


if __name__ == "__main__":
    main()
