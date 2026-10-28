"""Verify and publish ALL emissions of a completed continuous teacher run.

This CPU-only path never resumes Isaac, changes dataset selectors, admits
qualification output, or claims that raw conditions are rendered training data.
The native/container output directory may be mapped to an exact local directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

from pipeline.bc_collection.continuous_plan import build_request, digest, validate_plan
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    NATIVE_COLLECTOR_SHA256, canonical, file_sha256, publish_json, source_run_tag,
)

NATIVE_SCHEMA = "roborender.bc-native-continuous-manifest.v1"
WINDOW_SCHEMA = "roborender.bc-native-continuous-window.v1"
EXECUTION_SCHEMA = "roborender.bc-continuous-execution.v1"
POOL_SCHEMA = "roborender.bc-pool.continuous.v1"
WRAPPER_KEYS = {"emitted_at_window", "provenance_path", "provenance_sha256", "provenance_bytes"}
VIEWS = ("head_color", "hand_left", "hand_right")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapped(path, request, native_root):
    original, base = PurePosixPath(path), PurePosixPath(request["output_dir"])
    if not original.is_absolute() or ".." in original.parts or not original.is_relative_to(base):
        raise ValueError("native evidence path escaped the exact output root")
    target = native_root / original.relative_to(base)
    if target.is_symlink() or target.resolve() != target or not target.is_file():
        raise ValueError("native evidence must be a regular file without symlink aliases")
    return target


def _bound_file(reference, request, native_root):
    if (not isinstance(reference, dict) or not _sha(reference.get("sha256"))
            or not _integer(reference.get("bytes"))):
        raise ValueError("invalid immutable evidence binding")
    path = _mapped(reference["path"], request, native_root)
    if path.stat().st_size != reference["bytes"] or file_sha256(path) != reference["sha256"]:
        raise ValueError("immutable evidence size/hash mismatch")
    return path


def _accounting(value, *, width, window, drained=False):
    names = ("emitted_per_lane", "discarded_reset_tail_rows_per_lane", "remaining_unsealed_tail_rows_per_lane")
    for name in names:
        rows = value.get(name)
        if not isinstance(rows, list) or len(rows) != width or any(not _integer(n) for n in rows):
            raise ValueError("missing full-lane emission/tail accounting")
    excluded = value.get('unrecorded_rows_per_lane') if drained else [0] * width
    if not isinstance(excluded, list) or len(excluded) != width or any(not _integer(n) for n in excluded):
        raise ValueError('missing post-drain unrecorded row accounting')
    for emitted, discarded, tail, ignored in zip(*(value[name] for name in names), excluded):
        if tail >= 81 or 81 * emitted + discarded + tail + ignored != 81 * window:
            raise ValueError("complete clips + discarded tails + final tail do not cover every native row")


def validate_payload(path, record):
    """Check all image headers and load only small numeric state/action arrays."""
    import numpy as np
    headers = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or any(not n.endswith(".npy") or "/" in n for n in members):
            raise ValueError("invalid/duplicate NPZ members")
        for name in members:
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported native NPY header")
                if dtype.hasobject:
                    raise ValueError("pickled native payload is forbidden")
                expected_bytes = stream.tell() + int(np.prod(shape, dtype=object)) * dtype.itemsize
                if archive.getinfo(name).file_size != expected_bytes:
                    raise ValueError("truncated native array payload")
                headers[name[:-4]] = (shape, dtype)
    expected = {"gt_actions": ((81, 26), np.dtype("float32")),
                "dones": ((81,), np.dtype("bool"))}
    for view in VIEWS:
        expected.update({f"depth_rawq__{view}": ((81, 480, 832), np.dtype("uint16")),
                         f"depth_raw__{view}": ((81, 240, 416), np.dtype("float16")),
                         f"depth_ctrl__{view}": ((81, 240, 416), np.dtype("uint8")),
                         f"rgb_robot__{view}": ((81, 240, 416, 3), np.dtype("uint8"))})
    if any(headers.get(key) != value for key, value in expected.items()):
        raise ValueError("native81-row teacher/three-view payload shape or dtype differs")
    observations = [key for key in headers if key.startswith("obs_")]
    if (not {"obs_actor_obs", "obs_teacher_obs", "obs_homie_obs", "obs_gt_obj_pos"}.issubset(observations)
            or "obs_vision_obs" in headers or "rgb_gen" in headers):
        raise ValueError("raw teacher conditions require nonvision observations, not rendered/student input")
    with np.load(path, allow_pickle=False) as archive:
        for key in ["gt_actions", *observations]:
            value = archive[key]
            if value.ndim < 2 or value.shape[0] != 81 or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("native action/observation rows are invalid")
        if archive["dones"].any():
            raise ValueError("original Raven payload dones must remain all false")
        for key, expected_value in (("env_id", record["local_env_id"]), ("seq", record["seq"])):
            if archive[key].shape != () or int(archive[key]) != expected_value:
                raise ValueError("native payload lane/sequence differs from provenance")
        if archive["prompt"].shape != () or not str(archive["prompt"]):
            raise ValueError("native prompt missing")


def _validate_initial(document, request):
    initial = document.get("initial_state_attestation")
    width = len(request["motions"])
    if (not isinstance(initial, dict)
            or initial.get("when") != "first_pre_step_after_native_reset_and_random_episode_length_initialization"
            or initial.get("motion_binding_sha256") != digest(request["motions"])):
        raise ValueError("missing exact first-pre-action native initialization evidence")
    reset = initial.get("initial_reset_completion")
    expected = {"schema": "roborender.initial-reset-completion.v1", "generation": 1,
        "num_envs": width, "boundary": "after_reset_all_writes_and_observations",
        "cleared_lanes": width, "clear_calls": 1, "native_steps_added": 0,
        "ordinary_done_resets_modified": False}
    if (reset != expected or any(type(reset.get(k)) is not int for k in (
            "generation", "num_envs", "cleared_lanes", "clear_calls", "native_steps_added"))
            or reset.get("ordinary_done_resets_modified") is not False):
        raise ValueError("corrected initial reset is absent, repeated or modified")
    arrays = initial.get("arrays", {})
    required = {"robot_root", "dof_pos", "dof_vel", "stage", "episode_length",
                "time_in_stage", "actual_time_in_stage",
                "front_object", "hold_object", "tray_object", "actor_obs", "teacher_obs", "homie_obs", "gt_obj_pos"}
    if not required.issubset(arrays):
        raise ValueError("initial native state evidence lacks required arrays")
    for name in required:
        value = arrays[name]
        if (not isinstance(value, dict) or not _sha(value.get("sha256"))
                or not isinstance(value.get("shape"), list) or not value["shape"]
                or value["shape"][0] != width or not isinstance(value.get("dtype"), str)):
            raise ValueError("invalid initial state array attestation")
    if (arrays["time_in_stage"]["shape"] != [width]
            or arrays["time_in_stage"]["dtype"] != "int64"
            or arrays["actual_time_in_stage"] != arrays["time_in_stage"]):
        raise ValueError("initial actual stage clock must copy Raven's randomized stage clock")
    bindings = initial.get("episode_bindings")
    if (not isinstance(bindings, list) or len(bindings) != width
            or any(row.get("vector_step_index") != 0 or row.get("episode_step_index") != 0
                   or not _sha(row.get("reset_rng_sha256")) for row in bindings)):
        raise ValueError("initial episode/RNG binding does not precede the first action")


def _validate_row(record, provenance, request, *, ordinal, emitted, previous, window):
    lane = record.get("local_env_id")
    if not _integer(lane) or lane >= len(request["motions"]):
        raise ValueError("invalid emitted lane")
    motion = request["motions"][lane]
    required = {"schema": "roborender.bc-chunk-provenance.v1", "source_run_tag": source_run_tag(request),
        "run_id": request["run_id"], "cohort_id": request["cohort_id"], "split": "train", "rank": 0,
        "global_env_id": lane, "motion_id": motion["motion_id"], "motion_sha256": motion["sha256"],
        "simulator_seed": request["process_seed"], "frames": 81, "teacher_only": True,
        "sampling_plan_sha256": request["sampling_plan_sha256"], "selected_for_split": True,
        "overcollection": False, "emitted_chunk_ordinal": emitted[lane], "emitted_at_window": window,
        "selection_policy": "all_complete_emissions_v1"}
    if any(record.get(key) != value or type(record.get(key)) is not type(value)
           for key, value in required.items()):
        raise ValueError("native clip selection/identity/order differs")
    if ({k: v for k, v in record.items() if k not in WRAPPER_KEYS}
            != {k: v for k, v in provenance.items() if k != "rows"}):
        raise ValueError("per-step sidecar differs from the immutable chunk index")
    rows = provenance.get("rows")
    if not isinstance(rows, list) or len(rows) != 81:
        raise ValueError("every native clip needs exactly81 provenance rows")
    first = record.get("first_episode_step_index")
    seq = record.get("seq")
    if (not _integer(first) or first % 81 or not _integer(seq)
            or record.get("last_episode_step_index") != first + 80
            or not isinstance(record.get("episode_id"), str) or not record["episode_id"]):
        raise ValueError("invalid complete within-episode interval")
    start = rows[0].get("vector_step_index")
    if (not _integer(start) or (start + 80) // 81 + 1 != window
            or start + 80 >= request["max_windows"] * 81):
        raise ValueError("clip emission window differs from actual transition indices")
    for index, row in enumerate(rows):
        if (row.get("episode_id") != record["episode_id"] or row.get("episode_step_index") != first + index
                or row.get("vector_step_index") != start + index or type(row.get("actual_done")) is not bool
                or (index < 80 and row["actual_done"]) or not _sha(row.get("reset_rng_sha256"))
                or any(not _integer(row.get(k)) for k in ("task_reset_index", "sample_idx", "sample_idx_in_seq", "native_episode_length"))):
            raise ValueError("clip crosses a reset, skips/reorders rows or lacks actual reset evidence")
    prior = previous.get(lane)
    if prior:
        same = prior["episode_id"] == record["episode_id"]
        if (start <= prior["last_vector"] or seq <= prior["seq"]
                or (same and (first != prior["last_episode"] + 1 or seq != prior["seq"] + 1))
                or (not same and (first != 0 or seq < prior["seq"] + 3))):
            raise ValueError("native lane chronology or sequence-gap convention differs")
    elif first != 0:
        raise ValueError("first retained clip is not the first complete clip of its episode")
    previous[lane] = {"episode_id": record["episode_id"], "last_episode": first + 80,
                      "last_vector": start + 80, "seq": seq}
    expected_name = f"chunk_{source_run_tag(request)}_{ordinal:07d}.npz"
    if PurePosixPath(record["source_path"]).name != expected_name:
        raise ValueError("native full-emission filename sequence has a gap")
    emitted[lane] += 1


def inspect_run(plan, *, native_output_dir, execution_path, expected_source_sha256,
                expected_adapter_sha256, expected_model_assets, on_chunk=None):
    """Verify a complete run; optionally stream verified chunks to a publisher.

    The callback cannot constitute admission: only the terminal return/seal does.
    An exception leaves any copied staging evidence unsealed, never reusable as a
    completed pool. There is deliberately no partial-run or qualification opt-in.
    """
    plan = validate_plan(plan)
    request = build_request(plan)
    root = Path(native_output_dir).absolute()
    if root.resolve() != root or root.is_symlink() or not root.is_dir():
        raise ValueError("exact nonsymlink native output directory required")
    doc, manifest_sha = _read(root / "cohort-manifest.json")
    execution, execution_sha = _read(execution_path)
    from pipeline.bc_collection.production_gate import is_production
    drained = is_production(request)
    windows = doc.get('accepted_windows') if drained else plan['max_windows']
    if drained:
        from pipeline.bc_collection.run_continuous import validate_completion
        validate_completion(doc, request, doc.get('request_sha256'))
    if (doc.get("schema") != NATIVE_SCHEMA or doc.get("request") != request
            or doc.get("status") != "complete" or doc.get("completion_reason") !=
                ('total_chunks_and_episodes_drained' if drained else 'vector_window_budget')
            or doc.get("canonical_request_sha256") != digest(request)
            or not _sha(doc.get("request_sha256")) or doc.get("native_collector_sha256") != NATIVE_COLLECTOR_SHA256
            or not _sha(expected_adapter_sha256) or doc.get("adapter_sha256") != expected_adapter_sha256
            or doc.get("accepted_windows") != windows or doc.get("max_windows") != plan["max_windows"]
            or doc.get("vector_steps") != windows * 81 or doc.get("optimizer_steps") != 0
            or doc.get("teacher_only") is not True or doc.get("native_payload_writer_unchanged") is not (not drained)
            or doc.get("init_at_random_ep_len") is not True or doc.get("production_admitted") is not False
            or doc.get("overcollection_chunks") != 0 or doc.get("initial_reset_all_calls") != 1
            or doc.get("selection_policy") != "all_complete_emissions_v1"):
        raise ValueError("native complete-budget/teacher/source/request admission failed")
    if any(type(doc.get(k)) is not int for k in ("accepted_windows", "max_windows", "vector_steps",
            "optimizer_steps", "overcollection_chunks", "initial_reset_all_calls", "emitted_chunks", "selected_chunks")):
        raise ValueError("native counters must be exact integers")
    collector = execution.get("collector_execution", {})
    if (execution.get("schema") != EXECUTION_SCHEMA or execution.get("verdict") != "PASS"
            or execution.get("native_runtime_tested") is not True or execution.get("qualification_only") is not False
            or execution.get("production_admitted") is not False
            or execution.get("request_sha256") != doc["request_sha256"]
            or execution.get("request_canonical_sha256") != digest(request)
            or execution.get("native_manifest_sha256") != manifest_sha
            or execution.get("actual_num_envs") != plan["num_envs"]
            or execution.get("actual_vector_steps") != windows * 81
            or execution.get("initial_reset_all_calls") != 1
            or not _sha(expected_source_sha256) or collector.get("source_sha256") != expected_source_sha256
            or any(not _sha(execution.get(k)) for k in ("host_result_sha256", "launch_sha256"))
            or any(not _sha(collector.get(k)) for k in ("source_manifest_sha256", "port_manifest_sha256", "guard_sha256"))
            or not isinstance(collector.get("image_id"), str) or not collector["image_id"]):
        raise ValueError("missing exact production execution/source/resource receipt; qualification output is excluded")
    if any(type(execution.get(k)) is not int for k in ("actual_num_envs", "actual_vector_steps", "initial_reset_all_calls")):
        raise ValueError("execution counters must be exact integers")
    evidence_root = Path(execution_path).absolute().parent
    for path in (Path(execution_path).absolute(), evidence_root / "launch.json", evidence_root / "host-result.json"):
        if path.resolve() != path or path.is_symlink():
            raise ValueError("execution evidence must not use symlink aliases")
    launch, launch_sha = _read(evidence_root / "launch.json")
    host, host_sha = _read(evidence_root / "host-result.json")
    if (launch_sha != execution["launch_sha256"] or host_sha != execution["host_result_sha256"]
            or launch.get("collector_execution") != collector or launch.get("request") != request
            or launch.get("request_sha256") != doc["request_sha256"] or launch.get("qualification_only") is not False
            or not _sha(execution.get("container_id")) or host.get("container_id") != execution["container_id"]
            or host.get("passed") is not True or host.get("failure") is not None
            or type(host.get("exit_code")) is not int or host["exit_code"] != 0
            or host.get("replay_allowed") is not False or host.get("production_admitted") is not False
            or type(host.get("started_monotonic")) not in (int, float)
            or type(host.get("finished_monotonic")) not in (int, float)
            or not 0 <= host["started_monotonic"] < host["finished_monotonic"]):
        raise ValueError("launch/host-result byte chain does not attest this completed nonreplayable run")
    if drained:
        from pipeline.bc_collection.run_continuous import read_capacity, validate_capacity, read_drain_proof, storage_budget
        admission = launch.get('production_admission', {})
        binding = launch.get('capacity_binding', {})
        if (admission.get('schema') != 'roborender.bc-production-launch-admission.v1'
                or admission.get('ready') is not True or admission.get('request_sha256') != doc['request_sha256']
                or admission.get('source_sha256') != expected_source_sha256
                or admission.get('diagnostic_data_included') is not False
                or admission.get('capacity') != binding):
            raise ValueError('completed production lacks its exact prelaunch admission')
        capacity = read_capacity(binding['path'], binding['raw_sha256'])
        validate_capacity(capacity, launch['capacity_contract'])
        proof = admission.get('natural_drain', {})
        if (read_drain_proof(proof['path'], proof['raw_sha256'], launch['capacity_contract']) != proof
                or admission.get('rolling_storage') != storage_budget(launch['storage_policy'], request)):
            raise ValueError('production drain/storage admission differs from actual proof')
    models = doc.get("model_assets", {})
    if (set(models) != {"teacher", "homie_walk", "homie_stand"}
            or any(not isinstance(v, dict) or not _sha(v.get("sha256"))
                   or not isinstance(v.get("path"), str) or not Path(v["path"]).is_absolute() for v in models.values())
            or models != expected_model_assets or not _sha(collector.get("assets_manifest_sha256"))
            or collector.get("model_sha256") != {key: value["sha256"] for key, value in models.items()}):
        raise ValueError("privileged teacher/Homie identity differs")
    _validate_initial(doc, request)
    if request.get('collection_loop_protocol') is not None:
        from pipeline.bc_collection.collector import implementation_receipt
        if doc.get('collection_implementation') != implementation_receipt():
            raise ValueError('standalone collector/writer source receipt differs')
    if request.get('collection_rng_protocol') is not None:
        from pipeline.deployment.rtx_actor.bc_collection_rng import expected_receipt
        expected_rng = expected_receipt(plan['num_envs'], windows)
        if doc.get('collection_rng') != expected_rng:
            raise ValueError('missing exact completed collection RNG continuation receipt')
    _accounting(doc, width=plan["num_envs"], window=windows, drained=drained)
    index_ref = doc.get("chunk_index", {})
    index = _bound_file(index_ref, request, root)
    count = index_ref.get("count")
    if (index.name != "chunks.jsonl" or index.parent != root or index_ref.get("format") != "jsonl"
            or not _integer(count, 1) or count != sum(doc["emitted_per_lane"])
            or count != doc.get("emitted_chunks") or count != doc.get("selected_chunks")):
        raise ValueError("complete nonempty emission index required")
    progress = doc.get("progress_receipts", {})
    if progress.get("count") != windows or progress.get("directory") != request["output_dir"] + "/progress":
        raise ValueError("progress receipt count/path differs")
    final_progress = _bound_file(progress.get("final_receipt"), request, root)
    expected_names = {f"window-{i:06d}.json" for i in range(1, windows + 1)}
    if {p.name for p in (root / "progress").iterdir()} != expected_names:
        raise ValueError("missing or extra accepted-window progress receipts")
    emitted, previous, names = Counter(), {}, set()
    prefix, prefix_bytes, ordinal, previous_sha = hashlib.sha256(), 0, 0, None
    with index.open("rb") as stream:
        pending = stream.readline()
        for window in range(1, windows + 1):
            progress_path = root / "progress" / f"window-{window:06d}.json"
            if progress_path.is_symlink():
                raise ValueError("aliased progress receipt")
            receipt, receipt_sha = _read(progress_path)
            if (receipt.get("schema") != WINDOW_SCHEMA or receipt.get("status") != "accepted"
                    or receipt.get("request_sha256") != doc["request_sha256"]
                    or receipt.get("accepted_windows") != window or receipt.get("max_windows") != plan["max_windows"]
                    or receipt.get("vector_steps") != 81 * window
                    or receipt.get("previous_receipt_sha256") != previous_sha):
                raise ValueError("accepted-window chain is incomplete or inconsistent")
            _accounting(receipt, width=plan["num_envs"], window=window, drained=drained)
            while pending:
                if not pending.endswith(b"\n"):
                    raise ValueError("partial native JSONL record")
                record = json.loads(pending)
                if record.get("emitted_at_window") != window:
                    break
                prefix.update(pending)
                prefix_bytes += len(pending)
                if record.get("request_sha256") != doc["request_sha256"]:
                    raise ValueError("chunk raw request identity differs")
                provenance_path = _bound_file({"path": record["provenance_path"],
                    "sha256": record["provenance_sha256"], "bytes": record["provenance_bytes"]}, request, root)
                provenance, _ = _read(provenance_path)
                _validate_row(record, provenance, request, ordinal=ordinal,
                              emitted=emitted, previous=previous, window=window)
                payload = _bound_file({"path": record["source_path"], "sha256": record["source_sha256"],
                                       "bytes": record["source_bytes"]}, request, root)
                if payload.parent != root or payload.name in names or provenance_path != payload.with_suffix(".provenance.json"):
                    raise ValueError("duplicate or misplaced native payload/provenance")
                names.add(payload.name)
                validate_payload(payload, record)
                if on_chunk is not None:
                    on_chunk(record, payload, provenance_path)
                ordinal += 1
                pending = stream.readline()
            expected_index = {**index_ref, "sha256": prefix.hexdigest(), "bytes": prefix_bytes, "count": ordinal}
            if (receipt.get("chunk_index") != expected_index
                    or receipt.get("emitted_per_lane") != [emitted[i] for i in range(plan["num_envs"])]
                    or receipt.get("cumulative_emitted_chunks") != ordinal):
                raise ValueError("window prefix omits/adds/reorders complete native clips")
            previous_sha = receipt_sha
        if pending or stream.read(1) or ordinal != count:
            raise ValueError("extra, missing or out-of-order complete emissions")
    if (final_progress != progress_path or progress["final_receipt"]["sha256"] != previous_sha
            or any(receipt.get(k) != doc.get(k) for k in ("emitted_per_lane", "discarded_reset_tail_rows_per_lane", "remaining_unsealed_tail_rows_per_lane"))
            or {p.name for p in root.glob("*.npz")} != names
            or {p.name for p in root.glob("*.provenance.json")} != {Path(n).with_suffix(".provenance.json").name for n in names}
            or any(".part" in p.name or p.suffix == ".tmp" for p in root.iterdir())):
        raise ValueError("unindexed/partial outputs or terminal accounting mismatch")
    return {"schema": POOL_SCHEMA, "status": "sealed", "run_id": request["run_id"],
        "frames_per_chunk": 81, "chunk_count": count, "partition_counts": {"train": count},
        "selection_policy": "all_complete_native_emissions", "sampling_plan_sha256": digest(plan),
        "native_manifest_sha256": manifest_sha, "execution_receipt_sha256": execution_sha,
        "source_sha256": expected_source_sha256, "adapter_sha256": expected_adapter_sha256,
        "container_id": execution["container_id"],
        "model_assets": models, "motions": request["motions"], "motion_inventory": request["motion_inventory"],
        "vector_steps": windows * 81, "accepted_windows": windows,
        "emitted_per_lane": doc["emitted_per_lane"],
        "discarded_reset_tail_rows_per_lane": doc["discarded_reset_tail_rows_per_lane"],
        "remaining_unsealed_tail_rows_per_lane": doc["remaining_unsealed_tail_rows_per_lane"],
        "rendered_chunks": 0, "training_admitted": False, "production_admitted": False,
        "qualification_scope": "complete raw teacher conditions and provenance; rendering/learning not admitted",
        "active_dataset_selector_written": False}


def publish_pool(plan, *, dataset_root, **kwargs):
    """Copy to a fresh flat pool; publish a terminal seal only after full checks."""
    root = Path(dataset_root).absolute()
    if root.resolve() != root or root.exists() or root.is_symlink():
        raise FileExistsError("fresh nonsymlink dataset root required; partial publication is not automatically adopted")
    root.mkdir(parents=True)
    (root / "isaac").mkdir()
    (root / "provenance").mkdir()
    (root / "evidence").mkdir()
    with (root / "chunks.jsonl").open("xb") as output:
        def copy_chunk(record, payload, sidecar):
            destination = root / "isaac" / payload.name
            with payload.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, 8 << 20)
                target.flush()
                os.fsync(target.fileno())
            if file_sha256(destination) != record["source_sha256"]:
                raise ValueError("published payload differs; unsealed evidence retained")
            provenance = root / "provenance" / sidecar.name
            with provenance.open("xb") as target:
                target.write(sidecar.read_bytes())
                target.flush()
                os.fsync(target.fileno())
            if file_sha256(provenance) != record["provenance_sha256"]:
                raise ValueError("published provenance changed during copying; unsealed evidence retained")
            output.write(canonical({**record, "isaac_npz": "isaac/" + payload.name,
                "collection_filename": payload.name, "provenance": "provenance/" + sidecar.name}) + b"\n")
        manifest = inspect_run(plan, on_chunk=copy_chunk, **kwargs)
        output.flush()
        os.fsync(output.fileno())
    native_root = Path(kwargs["native_output_dir"])
    execution_root = Path(kwargs["execution_path"]).parent
    for name, source in (("native-manifest.json", native_root / "cohort-manifest.json"),
                         ("native-chunks.jsonl", native_root / "chunks.jsonl"),
                         ("execution.json", Path(kwargs["execution_path"])),
                         ("launch.json", execution_root / "launch.json"),
                         ("host-result.json", execution_root / "host-result.json")):
        with source.open("rb") as original, (root / "evidence" / name).open("xb") as target:
            shutil.copyfileobj(original, target, 8 << 20)
            target.flush()
            os.fsync(target.fileno())
    (root / "evidence" / "progress").mkdir()
    for source in sorted((native_root / "progress").iterdir()):
        with (root / "evidence" / "progress" / source.name).open("xb") as target:
            target.write(source.read_bytes())
            target.flush()
            os.fsync(target.fileno())
    # Recheck copied evidence against the identities admitted above, not mutable
    # source paths: publication must not seal a concurrent source replacement.
    copied_native, native_sha = _read(root / "evidence/native-manifest.json")
    copied_execution, execution_sha = _read(root / "evidence/execution.json")
    if (native_sha != manifest["native_manifest_sha256"] or execution_sha != manifest["execution_receipt_sha256"]
            or file_sha256(root / "evidence/native-chunks.jsonl") != copied_native["chunk_index"]["sha256"]
            or file_sha256(root / "evidence/launch.json") != copied_execution["launch_sha256"]
            or file_sha256(root / "evidence/host-result.json") != copied_execution["host_result_sha256"]):
        raise ValueError("execution evidence changed during publication; unsealed evidence retained")
    previous_sha = None
    for window in range(1, manifest["accepted_windows"] + 1):
        receipt, receipt_sha = _read(root / "evidence/progress" / f"window-{window:06d}.json")
        if receipt.get("previous_receipt_sha256") != previous_sha:
            raise ValueError("copied progress chain changed during publication")
        previous_sha = receipt_sha
    if previous_sha != copied_native["progress_receipts"]["final_receipt"]["sha256"]:
        raise ValueError("copied terminal progress receipt changed")
    publish_json(root / "evidence" / "plan.json", validate_plan(plan))
    manifest["chunks_index"] = {"path": "chunks.jsonl", "sha256": file_sha256(root / "chunks.jsonl"),
        "bytes": (root / "chunks.jsonl").stat().st_size, "count": manifest["chunk_count"], "format": "jsonl"}
    for directory in (root / "isaac", root / "provenance", root / "evidence/progress", root / "evidence", root):
        _fsync_directory(directory)
    publish_json(root / "manifest.json", manifest)
    _fsync_directory(root)
    publish_json(root / "source-seal.json", {"schema": "roborender.bc-continuous-source-seal.v1", "verdict": "PASS",
        "manifest_sha256": file_sha256(root / "manifest.json"), "raw_chunks": manifest["chunk_count"],
        "rendered_chunks": 0, "training_admitted": False, "qualification_scope": manifest["qualification_scope"]})
    _fsync_directory(root)
    return manifest


def iter_pool_chunks(dataset_root):
    """Explicit new-schema reader; historical flat-list manifests are rejected."""
    root = Path(dataset_root).resolve(strict=True)
    manifest, manifest_sha = _read(root / "manifest.json")
    seal, _ = _read(root / "source-seal.json")
    if (manifest.get("schema") != POOL_SCHEMA or manifest.get("status") != "sealed"
            or seal.get("schema") != "roborender.bc-continuous-source-seal.v1"
            or seal.get("verdict") != "PASS" or seal.get("manifest_sha256") != manifest_sha):
        raise ValueError("unsupported or unsealed continuous raw pool")
    ref = manifest["chunks_index"]
    if ref.get("path") != "chunks.jsonl" or ref.get("format") != "jsonl":
        raise ValueError("unsupported pool index")
    index = root / "chunks.jsonl"
    if index.is_symlink() or index.stat().st_size != ref["bytes"] or file_sha256(index) != ref["sha256"]:
        raise ValueError("pool index changed")
    count = 0
    with index.open("rb") as stream:
        for line in stream:
            count += 1
            yield json.loads(line)
    if count != ref["count"] or count != manifest["chunk_count"]:
        raise ValueError("pool index count differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--native-output-dir", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--model-assets", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = publish_pool(json.loads(args.plan.read_bytes()), dataset_root=args.dataset_root,
        native_output_dir=args.native_output_dir, execution_path=args.execution,
        expected_source_sha256=args.expected_source_sha256, expected_adapter_sha256=args.expected_adapter_sha256,
        expected_model_assets=json.loads(args.model_assets.read_bytes()))
    print(json.dumps({"raw_chunks": result["chunk_count"], "rendered_chunks": 0, "training_admitted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
