"""Small receipts from the existing source-pinned native BC session lifecycle.

Runs on the host launcher, never in the simulator's capture/writer path. It
hashes source/JSON metadata only: no image, NPZ payload or model rescans.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

RUNTIME_PATHS = tuple(sorted("pipeline/deployment/rtx_actor/" + name for name in (
    "isaac6_bc_production.py", "isaac6_bc_session.py", "run_bc_session.py", "run_bc_cohort.py",
    "native_motion_assignment.py", "materialize_isaac6.py", "roborender_isaac6.py", "initial_reset_contract.py",
    "isaac6_bc_trainer.py", "isaac6_capture_trainer.py", "qualify_native_isaac6.py"))) + (
    "pipeline/dagger/rtx/code/src/dagger_rtx/process_memory.py",)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def collector_identity(source_manifest, port_manifest, image_id, *, runtime_overlays=None):
    """Inputs were already file-verified by run_bc_session before Docker starts."""
    rows = source_manifest["files"]
    if (digest(canonical(rows)) != source_manifest["source_sha256"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or port_manifest.get("schema") != "roborender.isaac6.migration.v1"):
        raise ValueError("unverified source manifest, native port or immutable image")
    adapter = "pipeline/deployment/rtx_actor/isaac6_bc_production.py"
    matches = [row for row in rows if row["path"] == adapter]
    if len(matches) != 1:
        raise ValueError("source release lacks exactly one native collector adapter")
    relevant = {row["path"]: row["sha256"] for row in rows if row["path"] in RUNTIME_PATHS}
    overlays = runtime_overlays or {}
    if set(relevant) != set(RUNTIME_PATHS) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in overlays.values()):
        raise ValueError("runtime-relevant source or actual guard overlay hashes are missing")
    result = dict(adapter_source_sha256=matches[0]["sha256"],
        source_release_sha256=source_manifest["source_sha256"], container_image_id=image_id,
        native_runtime_version=f"isaac_sim={port_manifest['isaac_sim']};isaaclab={port_manifest['isaaclab_commit']}",
        lifecycle_recipe="one_kit_fresh_seeded_scene_per_declared_cohort_v1",
        memory_buffer_recipe="original_raven_capture_lists_and_writer_v1")
    contract = dict(schema="roborender.bc-runtime-contract.v1", source_files=relevant,
        derived_port_canonical_sha256=digest(canonical(port_manifest)), runtime_overlays=overlays,
        **{key: result[key] for key in ("container_image_id", "native_runtime_version", "lifecycle_recipe", "memory_buffer_recipe")})
    result["runtime_contract_sha256"] = digest(canonical(contract))
    return result


def build_session_receipts(*, launch, launch_raw_sha256, session, result,
                           result_raw_sha256, host_result, host_result_raw_sha256,
                           cohort_envelopes):
    """Only a completed, exact lifecycle can emit PASS; historical logs cannot."""
    if (host_result.get("passed") is not True or host_result.get("exit_code") != 0
            or host_result.get("failure") is not None or result.get("status") != "complete"
            or result.get("native_initializations") != 1
            or result.get("scene_initializations") != len(session["requests"])
            or result.get("failure") is not None
            or result.get("session_id") != session["session_id"]
            or launch.get("session_canonical_sha256") != digest(canonical(session))
            or len(result.get("cohorts", [])) != len(session["requests"])
            or len(cohort_envelopes) != len(session["requests"])):
        raise ValueError("incomplete/unbound native session cannot emit execution evidence")
    execution = launch["collector_execution"]
    requests = launch.get("request_identities", {})
    if set(requests) != {row["sha256"] for row in session["requests"]}:
        raise ValueError("launch lacks verified raw/canonical request identities")
    session_receipt = dict(schema="roborender.bc-session-execution.v1", verdict="PASS",
        session_id=session["session_id"], collector_execution=execution,
        source_manifest_raw_sha256=launch["source_manifest_raw_sha256"],
        port_manifest_raw_sha256=launch["port_manifest_raw_sha256"],
        launch_raw_sha256=launch_raw_sha256, session_sha256=launch["session_sha256"],
        session_canonical_sha256=launch["session_canonical_sha256"],
        native_result_raw_sha256=result_raw_sha256, host_result_raw_sha256=host_result_raw_sha256,
        native_initializations=1, scene_initializations=len(session["requests"]), request_identities=requests)
    session_sha = digest(canonical(session_receipt))
    outputs = []
    for expected, actual, envelope in zip(session["requests"], result["cohorts"], cohort_envelopes):
        doc = envelope["document"]
        if (actual.get("request_sha256") != expected["sha256"]
                or doc.get("request_sha256") != expected["sha256"]
                or digest(canonical(doc["request"])) != requests[expected["sha256"]]
                or doc.get("status") != "complete" or doc.get("optimizer_steps") != 0
                or doc.get("teacher_only") is not True or doc.get("native_payload_writer_unchanged") is not True
                or actual.get("cohort_id") != doc["request"]["cohort_id"]
                or actual.get("vector_steps") != doc["vector_steps"]):
            raise ValueError("native cohort/result/request identity differs")
        outputs.append(dict(schema="roborender.bc-cohort-execution.v1", verdict="PASS",
            cohort_manifest_raw_sha256=envelope["raw_sha256"], request_sha256=expected["sha256"],
            request_canonical_sha256=requests[expected["sha256"]],
            collector_execution=execution, session_execution_canonical_sha256=session_sha,
            actual_num_envs=len(doc["request"]["motions"]), actual_vector_steps=doc["vector_steps"]))
    return session_receipt, outputs


def publish_completed_session(runtime, session, host_result, publish):
    runtime = Path(runtime).resolve(strict=True)
    launch_raw = (runtime / "launch.json").read_bytes()
    result_raw = (runtime / "bc-session-result.json").read_bytes()
    host_raw = (runtime / "host-result.json").read_bytes()
    if json.loads(host_raw) != host_result:
        raise ValueError("host completion changed before evidence publication")
    result = json.loads(result_raw)
    envelopes, paths = [], []
    for row in result["cohorts"]:
        relative = Path(row["output_dir"]).relative_to("/dagger-runtime")
        path = (runtime / relative / "cohort-manifest.json").resolve(strict=True)
        if not path.is_relative_to(runtime):
            raise ValueError("cohort output escaped private runtime")
        raw = path.read_bytes()
        envelopes.append(dict(document=json.loads(raw), raw_sha256=digest(raw)))
        paths.append(path.with_name("execution-receipt.json"))
    receipt, outputs = build_session_receipts(launch=json.loads(launch_raw), launch_raw_sha256=digest(launch_raw),
        session=session, result=result, result_raw_sha256=digest(result_raw), host_result=host_result,
        host_result_raw_sha256=digest(host_raw), cohort_envelopes=envelopes)
    publish(runtime / "session-execution.json", receipt)
    for path, doc in zip(paths, outputs):
        publish(path, doc)
    return {"session_execution": str(runtime / "session-execution.json"),
            "cohort_execution_receipts": [str(path) for path in paths]}
