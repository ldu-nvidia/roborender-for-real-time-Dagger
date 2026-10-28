"""Metadata-only split/lineage admission; never loads images or teacher arrays.

Raven did not supply a held-out MSE protocol. These checks implement the
requested statistical extension, independently of the native loss/optimizer.
Historical datasets remain readable but are not silently called production
qualified. Motion coverage requires recorded identities, not guessed env IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from pipeline.legacy.bc_collection.artifact_roles import offline_roles, validate_artifact_contract

LEGACY_SCHEMA = "roborender.bc-data-admission.v1"
SCHEMA = "roborender.bc-data-admission.v2"
SCHEMA_V3 = "roborender.bc-data-admission.v3"
INDEPENDENT_RECIPE = "raven_three_view_independent_v2"
SPLITS = ("train", "validation", "test")


def digest(path: Path) -> str:
    return hashlib.sha256(metadata_bytes(path)).hexdigest()


def metadata_bytes(path):
    path = Path(path)
    if path.stat().st_size > 32 << 20:
        raise ValueError("metadata exceeds bounded 32 MiB limit")
    with path.open("rb") as stream:
        raw = stream.read((32 << 20) + 1)
    if len(raw) > 32 << 20:
        raise ValueError("metadata grew beyond bounded 32 MiB limit")
    return raw


def stream(row: dict) -> tuple:
    return row["source_run_tag"], row["rank"], row["local_env_id"]


def audit_partitions(source_manifest: dict, documents: dict, *,
                     require_independent_render: bool = False) -> dict:
    roles = offline_roles(source_manifest)
    if set(documents) != set(roles):
        raise ValueError("all three split indexes are required; no test payload is opened")
    rows = source_manifest["chunks"]
    by_id = {r["chunk_id"]: r for r in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("empty source pool or duplicate chunk identities")
    if source_manifest.get("frames_per_chunk") != 81:
        raise ValueError("Raven chunks must contain 81 time rows")
    owners, groups, stats = {}, defaultdict(set), {}
    motion_hashes = defaultdict(set)
    missing = Counter()
    for split in roles:
        doc = documents[split]
        items = doc.get("items", [])
        if doc.get("split") != split or doc.get("chunk_count") != len(items) or not items:
            raise ValueError("invalid/empty split index: " + split)
        modes, seeds, strata = Counter(), Counter(), Counter()
        for item in items:
            cid = item["chunk_id"]
            if cid not in by_id or cid in owners:
                raise ValueError("unknown or repeated chunk across split indexes: " + cid)
            owners[cid] = split
            row = by_id[cid]
            for field in ("isaac_npz", "rendered_npz"):
                if item[field] != row[field]:
                    raise ValueError("split/source path mismatch: " + cid)
                groups[(field, row[field])].add(split)
            groups[("collection_stream", stream(row))].add(split)
            if row.get("episode_id"):
                groups[("episode_id", row["episode_id"])].add(split)
            else:
                missing["episode_id"] += 1
            for field in ("motion_sha256", "request_sha256", "sampling_plan_sha256"):
                value = row.get(field)
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    missing[field] += 1
            if row.get("teacher_only") is not True:
                missing["teacher_only_attestation"] += 1
            if (row.get("split") != split or row.get("selected_for_split") is not True
                    or row.get("overcollection") is not False):
                missing["collection_time_split_assignment"] += 1
            if row.get("source_sha256"):
                groups[("source_sha256", row["source_sha256"])].add(split)
            else:
                missing["source_sha256"] += 1
            if row.get("render_chain_id"):
                groups[("render_chain", row["render_chain_id"])].add(split)
            recipe = row.get("render_contract", source_manifest.get("render_contract"))
            if require_independent_render and (
                recipe != INDEPENDENT_RECIPE or row.get("render_chain_length") != 1
                or row.get("render_chain_index") != 0
                or row.get("render_inherits_previous_reference") is not False
            ):
                raise ValueError("chunk does not use independent Raven three-view rendering: " + cid)
            motion = row.get("motion_id")
            if motion is None:
                missing["motion_id"] += 1
            else:
                modes[str(motion)] += 1
                if row.get("motion_sha256"):
                    motion_hashes[str(motion)].add(row["motion_sha256"])
            seed = row.get("simulator_seed")
            if seed is None:
                missing["simulator_seed"] += 1
            else:
                seeds[str(seed)] += 1
                if source_manifest.get("artifact_roles"):
                    groups[("simulator_process_seed", seed)].add(split)
            if motion is not None and seed is not None:
                groups[("motion_process_seed", (motion, seed))].add(split)
            strata[f"batch={row.get('source_batch')},rank={row['rank']}"] += 1
        stats[split] = {"chunks": len(items), "frames": 81 * len(items),
                        "motion_counts": dict(sorted(modes.items())),
                        "seed_counts": dict(sorted(seeds.items())),
                        "batch_rank_counts": dict(sorted(strata.items()))}
    if set(owners) != set(by_id):
        raise ValueError("split indexes do not cover the complete source pool")
    if any(len(values) != 1 for values in motion_hashes.values()):
        raise ValueError("one motion identity refers to different asset bytes across the pool")
    overlaps = Counter(kind for (kind, _), value in groups.items() if len(value) > 1)
    if overlaps:
        raise ValueError("cross-split dependency leakage: " + json.dumps(dict(overlaps), sort_keys=True))
    populations = [set(stats[s]["motion_counts"]) for s in roles]
    common_modes = set.intersection(*populations)
    all_modes = set.union(*populations)
    coverage_recorded = not missing["motion_id"] and bool(all_modes)
    matching_support = coverage_recorded and common_modes == all_modes
    warnings = []
    if missing:
        warnings.append("Missing provenance prevents certifying distribution coverage.")
    if coverage_recorded and not matching_support:
        warnings.append("Some motion modes are absent from a split; review population coverage.")
    extra = {}
    if source_manifest.get("artifact_roles"):
        contract = validate_artifact_contract(source_manifest,
            motion_bindings={key: next(iter(values)) for key, values in motion_hashes.items()},
            forbidden_seeds={row["simulator_seed"] for row in rows})
        if {s: stats[s]["chunks"] for s in roles} != contract["offline_chunk_counts"]:
            raise ValueError("offline role counts differ from the frozen artifact contract")
        extra = dict(artifact_roles=contract, closed_loop_test_executed=False)
    return {"schema": SCHEMA_V3 if extra else SCHEMA, "metadata_independence_passed": True,
            "independent_render_required": require_independent_render,
            "splits": stats, "missing_provenance": dict(missing),
            "motion_support_matches": matching_support, "common_motion_count": len(common_modes),
            "observed_motion_count": len(all_modes), "warnings": warnings,
            "test_payloads_opened": False,
            "purpose": "metadata_only", "production_admitted": False,
            "coverage_note": "Recorded motion support only; not proof of matched reset/physics/appearance distribution.", **extra}


def audit_split_files(source_manifest_path: Path, split_paths: dict, *,
                      require_independent_render: bool = False, purpose=None, run_name=None,
                      protocol_path=None, protocol_sha256=None,
                      render_receipt_path=None, render_receipt_sha256=None) -> dict:
    if purpose not in {"qualification", "production"}:
        raise ValueError("explicit qualification or production purpose is required")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("admission run name is required")
    if purpose == "qualification" and not run_name.startswith("smoke-"):
        raise ValueError("qualification must use a smoke- run name")
    if not require_independent_render:
        raise ValueError("purpose-aware admission requires independent render lineage")
    source_manifest_path = source_manifest_path.resolve(strict=True)
    source_sha = digest(source_manifest_path)
    documents = {}
    for split, path in split_paths.items():
        path = Path(path).resolve(strict=True)
        doc = json.loads(metadata_bytes(path))
        if ((path.parent / doc["source_manifest"]).resolve() != source_manifest_path
                or doc["source_manifest_sha256"] != source_sha):
            raise ValueError("split references a different source manifest")
        documents[split] = doc
    report = audit_partitions(json.loads(metadata_bytes(source_manifest_path)), documents,
                              require_independent_render=require_independent_render)
    report.update(source_manifest=str(source_manifest_path), source_manifest_sha256=source_sha,
                  split_files={name: {"path": str(Path(path).resolve()), "sha256": digest(Path(path))}
                               for name, path in split_paths.items()})
    if report["missing_provenance"] or not report["motion_support_matches"]:
        raise ValueError("complete native provenance and common support are required even for qualification")
    report.update(purpose=purpose, run_name=run_name, qualification_admitted=purpose == "qualification",
                  production_admitted=False, protocol=None)
    if purpose == "production":
        from pipeline.bc_training.production_protocol import audit_production_protocol
        report["protocol"] = audit_production_protocol(
            protocol_path, protocol_sha256, source_manifest_path,
            render_receipt_path, render_receipt_sha256)
        report["production_admitted"] = True
    elif any(x is not None for x in (protocol_path, protocol_sha256, render_receipt_path, render_receipt_sha256)):
        raise ValueError("qualification cannot borrow a production protocol")
    return report


def read_admission_report(report_path: Path) -> dict:
    """Read historical bytes, not training authorization; use validate_training_admission."""
    report = json.loads(metadata_bytes(report_path))
    if report.get("schema") not in {SCHEMA, SCHEMA_V3, LEGACY_SCHEMA}:
        raise ValueError("unknown admission report schema")
    return report


def validate_training_admission(report_path: Path, train_path: Path, validation_path: Path,
                                *, allow_qualification=False, run_name=None) -> dict:
    """Bind a production gate to exact current files, without reading test arrays."""
    report = read_admission_report(report_path)
    if report.get("schema") not in {SCHEMA, SCHEMA_V3}:
        raise ValueError("historical v1 is readable but cannot authorize a new training run")
    purpose = report.get("purpose")
    if purpose == "qualification":
        if (not allow_qualification or not isinstance(run_name, str) or not run_name.startswith("smoke-")
                or report.get("production_admitted") is not False
                or report.get("qualification_admitted") is not True):
            raise ValueError("qualification training requires explicit flag and smoke- run name")
    elif purpose != "production" or report.get("production_admitted") is not True:
        raise ValueError("production data admission is absent or failed")
    roles = offline_roles(report)
    if (report["schema"] == SCHEMA_V3) != (roles != SPLITS):
        raise ValueError("admission schema and artifact roles differ")
    paths = {s: Path(report["split_files"][s]["path"]) for s in roles}
    if paths["train"].resolve() != train_path.resolve() or paths["validation"].resolve() != validation_path.resolve():
        raise ValueError("production admission binds different train/validation indexes")
    if any(digest(path) != report["split_files"][s]["sha256"] for s, path in paths.items()):
        raise ValueError("split metadata changed after admission")
    protocol = report.get("protocol") or {}
    render = protocol.get("render_execution_receipt") or {}
    actual = audit_split_files(Path(report["source_manifest"]), paths, require_independent_render=True,
        purpose=purpose, run_name=report["run_name"], protocol_path=protocol.get("path"),
        protocol_sha256=protocol.get("raw_sha256"), render_receipt_path=render.get("path"),
        render_receipt_sha256=render.get("raw_sha256"))
    if actual != report:
        raise ValueError("data admission report differs from recomputed metadata audit")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-independent-render", action="store_true")
    parser.add_argument("--purpose", choices=("qualification", "production"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--render-execution-receipt", type=Path)
    parser.add_argument("--render-execution-receipt-sha256")
    args = parser.parse_args()
    roles = offline_roles(json.loads(metadata_bytes(args.source_manifest)))
    report = audit_split_files(args.source_manifest,
        {s: args.split_dir / (s + ".json") for s in roles},
        require_independent_render=args.require_independent_render, purpose=args.purpose,
        run_name=args.run_name, protocol_path=args.protocol, protocol_sha256=args.protocol_sha256,
        render_receipt_path=args.render_execution_receipt,
        render_receipt_sha256=args.render_execution_receipt_sha256)
    with args.output.open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
