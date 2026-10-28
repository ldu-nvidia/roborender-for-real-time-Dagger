"""Stage one continuous Raven-style teacher scene; never launch or select data.

The caller supplies the explicitly resolved eligible motion ordering. Its first
B entries remain assigned to the same lanes for the whole declared window
budget. Native admission must independently check their bytes and eligibility.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

from pipeline.deployment.rtx_actor.isaac6_bc_production import canonical, file_sha256, publish_json
from pipeline.deployment.rtx_actor.isaac6_bc_production import CHUNK_TARGET_SMOKE
from pipeline.deployment.rtx_actor.bc_collection_rng import PROTOCOL as RNG_PROTOCOL
from pipeline.deployment.rtx_actor.bounded_bc_storage import PROTOCOL as BOUNDED_STORAGE_PROTOCOL
from pipeline.bc_collection.collector import PROTOCOL as COLLECTOR_PROTOCOL
from pipeline.bc_collection.episode_drain import PROTOCOL as DRAIN_PROTOCOL, is_bounded_qualification

SCHEMA = "roborender.bc-continuous-plan.v1"
SCHEMA_CONTINUOUS = "roborender.bc-native-continuous-request.v1"
VIEWS = ["head_color", "hand_left", "hand_right"]
SELECTION = "all_complete_native_emissions"
REFERENCE_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _identity(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", value)


def _path(value):
    return (isinstance(value, str) and PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value)


def resolve_inventory(directory, native_motion_root):
    """Hash regular NPZ files in Raven's os.listdir order, without unpickling.

    This does not certify duration/sample eligibility. Native startup rechecks
    every entry and fails if the supplied full inventory drops a motion. The
    returned explicit order is reproducible evidence of this filesystem now,
    not a reconstruction of an unavailable historical directory enumeration.
    """
    root = Path(directory).absolute()
    if root.is_symlink() or root.resolve() != root or not root.is_dir() or not _path(str(native_motion_root)):
        raise ValueError("exact local directory and absolute native motion root are required")
    names = [name for name in os.listdir(root) if name.endswith(".npz")]
    if not 1 <= len(names) <= 10000:
        raise ValueError("motion directory requires1..10000 NPZ entries")
    rows = []
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file() or not _identity(path.stem):
            raise ValueError("motion entries must be regular, safely named NPZ files without aliases")
        rows.append({"motion_id": path.stem, "path": str(PurePosixPath(native_motion_root) / name),
                     "sha256": file_sha256(path)})
    return rows


def build_plan(eligible_motions, *, run_id, max_windows, output_dir,
               num_envs=32, process_seed=0, collection_rng_protocol=RNG_PROTOCOL,
               collection_storage_protocol=None, collection_loop_protocol=COLLECTOR_PROTOCOL,
               target_chunks_per_lane=None, target_total_chunks=None, production_collection=False):
    """Build a CPU-only, train-only plan with an explicitly required budget."""
    smoke = target_chunks_per_lane is not None
    from pipeline.bc_collection.production_gate import marker, is_production
    production = is_production(dict(production_collection=marker(), motions=[None] * num_envs,
        episode_drain=dict(target_chunks=target_total_chunks), max_windows=max_windows))
    if type(production_collection) is not bool or (production_collection and (not production or smoke)):
        raise ValueError('production requires explicit one-RTX B32,total10000 and a fail-closed safety ceiling')
    production = production_collection and production
    if target_total_chunks is not None and (
            type(target_total_chunks) is not int or target_total_chunks < 1 or smoke
            or collection_loop_protocol != COLLECTOR_PROTOCOL
            or collection_storage_protocol != BOUNDED_STORAGE_PROTOCOL):
        raise ValueError("total chunk drain requires standalone bounded writer and no per-lane target")
    if smoke and (type(target_chunks_per_lane) is not int or target_chunks_per_lane != 20
            or num_envs not in (40, 50) or max_windows != 40 or collection_storage_protocol != BOUNDED_STORAGE_PROTOCOL
            or collection_loop_protocol != COLLECTOR_PROTOCOL):
        raise ValueError("chunk-target smoke requires B40/B50,20 chunks,40-window safety ceiling and bounded writer")
    if (not _identity(run_id) or len(run_id) > 170
            or type(num_envs) is not int or not 1 <= num_envs <= (50 if smoke else 40)
            or type(max_windows) is not int or not 1 <= max_windows <= 50000
            or type(process_seed) is not int or not 0 <= process_seed < 2**31
            or not _path(str(output_dir)) or len(PurePosixPath(str(output_dir)).parts) < 3):
        raise ValueError("invalid identity, B=1..40, explicit window budget1..50000, seed or output path")
    if not isinstance(eligible_motions, list) or not 1 <= len(eligible_motions) <= 10000:
        raise ValueError("an explicitly ordered eligible-motion list of1..10000 descriptors is required")
    for row in eligible_motions:
        if (not isinstance(row, dict) or set(row) != {"motion_id", "path", "sha256"}
                or not _identity(row["motion_id"]) or not _path(row["path"])
                or not row["path"].endswith(".npz")
                or not isinstance(row["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("invalid resolved eligible-motion descriptor")
    if any(len({row[key] for row in eligible_motions}) != len(eligible_motions)
           for key in ("motion_id", "path", "sha256")):
        raise ValueError("eligible motions must have unique identities, paths and content hashes")
    if len(eligible_motions) < num_envs:
        raise ValueError("eligible motions < B: repeated modulo lane mapping is not supported by this version")
    if collection_rng_protocol not in (None, RNG_PROTOCOL):
        raise ValueError("unknown collection RNG protocol")
    if collection_loop_protocol not in (None, COLLECTOR_PROTOCOL):
        raise ValueError("unknown standalone collection protocol")
    if collection_loop_protocol is not None and collection_rng_protocol != RNG_PROTOCOL:
        raise ValueError("standalone collection requires Raven incidental RNG continuation")
    drain_probe = is_bounded_qualification(dict(motions=[None] * num_envs, max_windows=max_windows,
        episode_drain=dict(protocol=DRAIN_PROTOCOL, target_chunks=target_total_chunks)))
    if (collection_storage_protocol not in (None, BOUNDED_STORAGE_PROTOCOL)
            or (collection_storage_protocol is not None and
                (collection_rng_protocol != RNG_PROTOCOL or
                 (max_windows not in (4, 8) and not smoke and not drain_probe and not production)))):
        raise ValueError("bounded storage requires corrected-RNG capacity or bounded drain qualification")
    if target_total_chunks is not None and not drain_probe and not production:
        raise ValueError("production total-target admission is not yet implemented")
    result = {
        "schema": SCHEMA, "run_id": run_id, "cohort_id": "continuous-0000",
        "split": "train", "process_seed": process_seed, "num_envs": num_envs,
        "rank": 0, "global_env_offset": 0, "max_windows": max_windows,
        "frames_per_window": 81, "vector_steps": 81 * max_windows,
        "output_dir": str(output_dir), "reference_commit": REFERENCE_COMMIT,
        "ordered_eligible_motions": deepcopy(eligible_motions),
        "eligible_motion_order_sha256": digest(eligible_motions),
        "motions": deepcopy(eligible_motions[:num_envs]),
        "assignment": "fixed_prefix_B_of_explicit_eligible_order_no_rotation",
        "eligible_order_origin": "explicit_resolved_input_not_claimed_historical_filesystem_order",
        "eligibility_validation": "native_rechecks_every_declared_lane_before_collection",
        "scene_count": 1, "conditioning_views": list(VIEWS),
        "selection_policy": SELECTION, "selection_uses_outcomes": False,
        "episode_reset_rule": "native_done_timeout_only_no_window_reset",
        "initial_clock_rule": "original_random_episode_and_stage_clocks_once_after_reset",
        "tail_rule": "emit_81_before_done_flush_discard_partial_terminal_and_final_tails",
        "optimizer_steps": 0,
        "explicit_exceptions": ["corrected_initial_robot_object_refresh",
                                "zero_optimizer_generated_only_student_training"],
        "production_admitted": False, "gpu_capacity_qualified": False,
        "active_dataset_selector_written": False,
    }
    if collection_rng_protocol is not None:
        result["collection_rng_protocol"] = collection_rng_protocol
    if production:
        result['production_collection'] = marker()
    if collection_storage_protocol is not None:
        result["collection_storage_protocol"] = collection_storage_protocol
    if collection_loop_protocol is not None:
        result["collection_loop_protocol"] = collection_loop_protocol
    if smoke:
        result["diagnostic_stop"] = dict(protocol=CHUNK_TARGET_SMOKE, chunks_per_lane=20)
    if target_total_chunks is not None:
        # Bounded functional qualification is distinct from full-buffer capacity.
        # The window count is a FAIL-CLOSED safety ceiling, not normal completion.
        result["episode_drain"] = dict(protocol=DRAIN_PROTOCOL, target_chunks=target_total_chunks)
        from pipeline.bc_collection.streaming_storage import policy as streaming_policy
        result["streaming_storage"] = streaming_policy(num_envs)
        result["tail_rule"] = "emit_81_before_done_discard_terminal_tail_drain_recorded_episodes"
        result["explicit_exceptions"].append("total_chunk_target_with_natural_episode_drain")
    return result


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("continuous plan must be an object")
    try:
        expected = build_plan(plan["ordered_eligible_motions"], run_id=plan["run_id"],
            max_windows=plan["max_windows"], output_dir=plan["output_dir"],
            num_envs=plan["num_envs"], process_seed=plan["process_seed"],
            collection_rng_protocol=plan.get("collection_rng_protocol"),
            collection_storage_protocol=plan.get("collection_storage_protocol"),
            collection_loop_protocol=plan.get("collection_loop_protocol"),
            target_chunks_per_lane=plan.get("diagnostic_stop", {}).get("chunks_per_lane"),
            target_total_chunks=plan.get("episode_drain", {}).get("target_chunks"),
            production_collection='production_collection' in plan)
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete continuous plan") from exc
    if plan != expected:
        raise ValueError("continuous plan differs from its exact single-scene protocol")
    return deepcopy(plan)


def build_request(plan):
    plan = validate_plan(plan)
    return {"schema": SCHEMA_CONTINUOUS,
            **({'production_collection': deepcopy(plan['production_collection'])} if 'production_collection' in plan else {}),
            **({"episode_drain": deepcopy(plan["episode_drain"])} if "episode_drain" in plan else {}),
            **({"streaming_storage": deepcopy(plan["streaming_storage"])} if "streaming_storage" in plan else {}),
            **({"diagnostic_stop": deepcopy(plan["diagnostic_stop"])} if "diagnostic_stop" in plan else {}),
            **{key: deepcopy(plan[key]) for key in (
                "run_id", "cohort_id", "split", "process_seed", "rank",
                "global_env_offset", "motions", "max_windows", "output_dir")},
            "motion_inventory": deepcopy(plan["ordered_eligible_motions"]),
            **({"collection_rng_protocol": plan["collection_rng_protocol"]} if "collection_rng_protocol" in plan else {}),
            **({"collection_storage_protocol": plan["collection_storage_protocol"]}
               if "collection_storage_protocol" in plan else {}),
            **({"collection_loop_protocol": plan["collection_loop_protocol"]}
               if "collection_loop_protocol" in plan else {}),
            "sampling_plan_sha256": digest(plan)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eligible-motions", type=Path,
                        help="JSON array of explicit, resolved eligible path/motion_id/sha256 descriptors")
    source.add_argument("--motion-dir", type=Path,
                        help="resolve local NPZ files in actual os.listdir order; native eligibility remains mandatory")
    parser.add_argument("--native-motion-root", help="required with --motion-dir; container-visible motion directory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--output-dir", required=True, help="native/container-visible fresh output path")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--target-chunks-per-lane", type=int, choices=[20],
                        help="explicit diagnostic B40/B50,40-window ceiling; stop after every lane saves20 chunks")
    parser.add_argument("--target-total-chunks", type=int,
                        help="minimum total complete chunks; natural episode drain, not a per-lane cutoff")
    parser.add_argument('--production-collection', action='store_true',
                        help='stage explicit B32/10000 request; live capacity and drain proofs still required')
    parser.add_argument("--process-seed", type=int, default=0)
    parser.add_argument("--collection-storage-protocol", choices=[BOUNDED_STORAGE_PROTOCOL],
                        help="opt-in qualification-only GPU-window/lane-writer; default stays unchanged")
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--request-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.motion_dir) != bool(args.native_motion_root):
        parser.error("--native-motion-root is required exactly when --motion-dir is used")
    if args.plan_out.resolve() == args.request_out.resolve():
        parser.error("plan and request output paths must differ")
    if any(path.exists() or path.is_symlink() for path in (args.plan_out, args.request_out)):
        raise FileExistsError("immutable staging outputs already exist")
    motions = (resolve_inventory(args.motion_dir, args.native_motion_root) if args.motion_dir
               else json.loads(args.eligible_motions.read_bytes()))
    plan = build_plan(motions, run_id=args.run_id,
        max_windows=args.max_windows, output_dir=args.output_dir,
        num_envs=args.num_envs, process_seed=args.process_seed,
        collection_storage_protocol=args.collection_storage_protocol,
        target_chunks_per_lane=args.target_chunks_per_lane,
        target_total_chunks=args.target_total_chunks, production_collection=args.production_collection)
    for path in (args.plan_out, args.request_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    publish_json(args.plan_out, plan)
    publish_json(args.request_out, build_request(plan))
    print(json.dumps({"staged_only": True, "plan_sha256": digest(plan),
                      "request_sha256": hashlib.sha256(args.request_out.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
