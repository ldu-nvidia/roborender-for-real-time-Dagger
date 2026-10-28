"""Fail-closed, metadata-only successor production admission.

All bindings explicitly distinguish raw file hashes from canonical JSON hashes.
The normalized capacity/render/eligibility receipts below are required evidence,
not generated here. Missing real qualification cannot be replaced by plan flags.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.legacy.bc_collection.sampling_plan import build_plan, build_offline_plan, canonical, SCHEMA_V3 as PLAN_SCHEMA_V3
from pipeline.legacy.bc_collection.artifact_roles import validate_artifact_contract
from pipeline.deployment.rtx_actor.isaac6_bc_production import NATIVE_COLLECTOR_SHA256

SCHEMA = "roborender.bc-production-protocol.v2"
SCHEMA_V3 = "roborender.bc-production-protocol.v3"
SCHEMA_V4 = "roborender.bc-production-protocol.v4"
MOTION_COUNT = 295
MODEL_NAMES = {"teacher", "homie_walk", "homie_stand"}
MAX_METADATA_BYTES = 32 << 20
NATIVE_RESET_PINS = {
    "gr00t/rl/config/env/walk_stand_place_grasp_turn_homie.yaml": "bd948b21d4f0ca96c084649488336071a3f4f426d322e9d78df791be48db90e6",
    "gr00t/rl/envs/loco_manip/reset_from_dataset_wsppt.py": "d42185ef7535e193bb5affcda6e7ad2fbac92e1c478feb9a846ed455f476ac94",
    "gr00t/rl/utils/teleop_reset_lib/dataset_wsppt.py": "949cc147ad9abc32600e277fb69a0579ad5fd87e7f2086249785b966d88de1a3",
    "gr00t/rl/utils/teleop_reset_lib/motion_loader_wsppt.py": "b67bab4f063723830f506762ed75b9c72bbe384ed114328a7e116f9e20aecfc7",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bound(binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "raw_sha256"}
            or not valid_sha(binding["raw_sha256"]) or not Path(binding["path"]).is_absolute()):
        raise ValueError("binding requires absolute path and explicit raw_sha256")
    path = Path(binding["path"]).resolve(strict=True)
    raw = read_metadata(path)
    if sha(raw) != binding["raw_sha256"]:
        raise ValueError("immutable bound metadata changed: " + str(path))
    return json.loads(raw)


def read_metadata(path):
    path = Path(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds bounded 32 MiB limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata grew beyond bounded 32 MiB limit")
    return raw


def validate_topology(topology, *, common=False):
    """A mixed campaign shares host/concurrency, never an invented memory limit."""
    keys = {"host_id", "concurrent_collectors", "minimum_host_reserve_bytes"}
    if not common:
        keys.add("cgroup_limit_bytes_per_collector")
    if (not isinstance(topology, dict) or set(topology) != keys
            or not isinstance(topology.get("host_id"), str) or not topology["host_id"]
            or any(type(topology.get(k)) is not int or topology[k] <= 0 for k in keys - {"host_id"})):
        raise ValueError("explicit host/concurrency/cgroup/reserve topology required")
    return topology


def capacity_profiles(protocol):
    """Return exact, raw-receipt-keyed profiles; v2/v3 retain their scalar gate."""
    if protocol["schema"] != SCHEMA_V4:
        return None
    common = validate_topology(protocol.get("execution_topology"), common=True)
    bindings = protocol.get("capacity_receipts", [])
    hashes = [b["raw_sha256"] for b in bindings]
    profiles = protocol.get("capacity_profiles")
    if (not hashes or len(set(hashes)) != len(hashes) or not isinstance(profiles, dict)
            or set(profiles) != set(hashes)):
        raise ValueError("mixed capacity profiles must bind each unique capacity receipt exactly")
    for topology in profiles.values():
        validate_topology(topology)
        if {k: topology[k] for k in common} != common:
            raise ValueError("mixed capacity may not change host/concurrency/reserve")
    return profiles


def validate_session_resources(protocol):
    """Independently bind every completed session to its actual qualified limit.

    Only small launch/host/session receipts are read. No payloads, qualification
    substitutions, or retrospective relabeling of an earlier cgroup are allowed.
    """
    profiles = capacity_profiles(protocol)
    if profiles is None:
        return {}
    expected = {b["raw_sha256"]: b for b in protocol.get("session_execution_receipts", [])}
    rows = protocol.get("session_resource_bindings", [])
    if not expected or len(rows) != len(expected) or len(expected) != len(protocol["session_execution_receipts"]):
        raise ValueError("every session requires one immutable resource binding")
    capacities = {b["raw_sha256"]: bound(b) for b in protocol["capacity_receipts"]}
    seen, widths = set(), {}
    for row in rows:
        if set(row) != {"session_execution", "launch", "host_result", "capacity_receipt_raw_sha256"}:
            raise ValueError("session resource binding has unexpected fields")
        session_binding = row["session_execution"]
        identity = session_binding["raw_sha256"]
        if identity in seen or expected.get(identity) != session_binding:
            raise ValueError("session resource binding is missing, duplicated or substituted")
        seen.add(identity)
        session, launch, host = (bound(row[k]) for k in ("session_execution", "launch", "host_result"))
        capacity_sha = row["capacity_receipt_raw_sha256"]
        if capacity_sha not in capacities:
            raise ValueError("session uses an unbound capacity profile")
        topology = profiles[capacity_sha]
        if (session.get("launch_raw_sha256") != row["launch"]["raw_sha256"]
                or session.get("host_result_raw_sha256") != row["host_result"]["raw_sha256"]
                or session.get("collector_execution") != protocol["collector_execution"]
                or launch.get("collector_execution") != protocol["collector_execution"]
                or launch.get("execution_topology") != topology
                or (launch.get("capacity_binding") or {}).get("raw_sha256") != capacity_sha
                or launch.get("qualification_only") is not False
                or launch.get("session_sha256") != session.get("session_sha256")
                or launch.get("session_canonical_sha256") != session.get("session_canonical_sha256")
                or launch.get("request_identities") != session.get("request_identities")
                or host.get("passed") is not True or host.get("exit_code") != 0 or host.get("failure") is not None):
            raise ValueError("session launch/host identity differs from its actual qualified profile")
        peak = host.get("sampled_cgroup_peak_bytes")
        available = host.get("minimum_host_available_kib")
        sample = host.get("last_live_sample", {})
        events = sample.get("events", {})
        if (type(peak) is not int or not 0 < peak <= topology["cgroup_limit_bytes_per_collector"]
                or type(available) is not int or available * 1024 < topology["minimum_host_reserve_bytes"]
                or sample.get("oom_killed") is not False or sample.get("memory_swap.current") != 0
                or any(type(events.get(k)) is not int or events[k] != 0
                       for k in ("max", "oom", "oom_kill", "oom_group_kill"))):
            raise ValueError("session resource evidence lacks safe memory/headroom/zero-OOM bounds")
        key = sha(canonical(session))
        if key in widths:
            raise ValueError("duplicate canonical session resource identity")
        widths[key] = capacities[capacity_sha]["num_envs"]
    return widths


def verify_render_asset_identity(protocol):
    """Reuse a prior per-workload full hash only while exact file identities hold.

    This reader NEVER treats the old registry alone as proof of current bytes.
    It does not hash multi-GB assets per chunk. A trusted workload bootstrap must
    create the hash-verified receipt once; absent or changed evidence fails shut.
    """
    registry = bound(protocol["renderer_registry"])
    receipt = bound(protocol["render_asset_identity"])
    if registry.get("schema") != "roborender.renderer_registry.v1":
        raise ValueError("pinned renderer registry required")
    artifacts = {x["id"]: x for x in registry["artifacts"]}
    foundation = artifacts["wan21_fun_control_1p3b_foundation"]
    adapter = artifacts["roborender_offline_bc_video_teacher_50step_lora_step05294"]
    expected = {"foundation/" + x["path"]: (x["sha256"], x["bytes"]) for x in foundation["files"]}
    expected["adapter"] = (adapter["sha256"], adapter["bytes"])
    models = protocol.get("render_model_sha256", {})
    if (set(models) != {"adapter", "foundation_manifest", "renderer_source", "prompt"}
            or any(not valid_sha(value) for value in models.values())
            or models["adapter"] != adapter["sha256"]
            or models["foundation_manifest"] != sha(canonical(foundation))
            or receipt.get("schema") != "roborender.bc-render-assets.v1"
            or receipt.get("hashes_verified") is not True
            or receipt.get("registry_raw_sha256") != protocol["renderer_registry"]["raw_sha256"]
            or receipt.get("model_sha256") != models
            or receipt.get("source_manifest_raw_sha256") != protocol["renderer_source_manifest"]["raw_sha256"]):
        raise ValueError("once-per-workload verified renderer/model identity is missing")
    source_manifest = bound(protocol["renderer_source_manifest"])
    if (source_manifest.get("schema") != "roborender.renderer-source-identity.v1"
            or sha(canonical(source_manifest)) != models["renderer_source"]
            or not source_manifest.get("files")):
        raise ValueError("renderer source manifest identity differs")
    for row in source_manifest["files"]:
        if row["relative_path"].startswith("/") or ".." in Path(row["relative_path"]).parts:
            raise ValueError("source manifest path is not relative")
        expected["source/" + row["relative_path"]] = (row["sha256"], row["bytes"])
    expected["prompt"] = (models["prompt"], receipt.get("prompt_bytes"))
    records = receipt.get("files", [])
    if len(records) != len(expected) or {r["role"] for r in records} != set(expected):
        raise ValueError("renderer asset receipt does not cover exactly the expected files")
    for row in records:
        if row.get("sha256") != expected[row["role"]][0] or row.get("bytes") != expected[row["role"]][1]:
            raise ValueError("renderer asset hash/size differs from expected registry/source")
        path = Path(row["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("hash-reuse requires the exact regular asset path")
        stat = path.stat()
        actual = {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        if row.get("file_identity") != actual or stat.st_size != row["bytes"]:
            raise ValueError("asset metadata changed; rehash once for this workload before reuse")
    return protocol["render_asset_identity"]


def validate_protocol_inputs(protocol_path, protocol_sha256):
    """Pre-GPU gate for immutable expected population, assets and runtime capacity."""
    if protocol_path is None or not valid_sha(protocol_sha256):
        raise ValueError("production requires an independently pinned protocol file/hash")
    protocol = bound({"path": str(protocol_path), "raw_sha256": protocol_sha256})
    if (protocol.get("schema") not in {SCHEMA, SCHEMA_V3, SCHEMA_V4} or protocol.get("purpose") != "production"
            or protocol.get("expected_motion_count") != MOTION_COUNT
            or protocol.get("training_chunk_batch_size") != 16
            or protocol.get("native_collector_sha256") != NATIVE_COLLECTOR_SHA256):
        raise ValueError("full295 production protocol, original writer and full B16 update required")
    inventory = bound(protocol["motion_inventory"])
    records = inventory.get("motions", [])
    if (inventory.get("schema") != "roborender.motion-inventory.v1"
            or inventory.get("motion_count") != MOTION_COUNT or len(records) != MOTION_COUNT
            or len({r["motion_id"] for r in records}) != MOTION_COUNT
            or len({r["sha256"] for r in records}) != MOTION_COUNT
            or any(not valid_sha(r["sha256"]) for r in records)
            or inventory.get("motion_records_sha256") != sha(canonical(records))):
        raise ValueError("complete unique frozen motion inventory required")
    inventory_sha = sha(canonical(inventory))
    plan = bound(protocol["sampling_plan"])
    expected_plan_schema = PLAN_SCHEMA_V3 if protocol["schema"] in {SCHEMA_V3, SCHEMA_V4} else "roborender.coverage-sampling-plan.v2"
    if (plan.get("schema") != expected_plan_schema
            or plan.get("motion_inventory_sha256") != inventory_sha):
        raise ValueError("full-inventory matched_exposure_v2 plan required")
    # Rebuild the complete seed-only recipe: this checks role/mode quotas, order,
    # independent seeds, selected ordinals, exposure spans and success-test cases.
    if protocol["schema"] in {SCHEMA_V3, SCHEMA_V4}:
        expected = build_offline_plan(inventory, seed=plan["seed"], num_envs=plan["num_envs_ceiling"],
            success_panel=protocol["closed_loop_test_panel"], train_chunks=2560, validation_chunks=320)
    else:
        expected = build_plan(inventory, seed=plan["seed"], num_envs=plan["num_envs_ceiling"],
            chunks=plan["total_target_chunks"], success_cases=plan["success_test"]["episode_count"],
            selection_policy="matched_exposure_v2")
    if plan != expected:
        raise ValueError("sampling plan differs from the immutable seed-only recipe")
    model_pins = protocol.get("model_sha256")
    if (not isinstance(model_pins, dict) or set(model_pins) != MODEL_NAMES
            or any(not valid_sha(value) for value in model_pins.values())):
        raise ValueError("teacher and both Homie pins required")
    execution = protocol.get("collector_execution", {})
    if (set(execution) != {"adapter_source_sha256", "source_release_sha256", "container_image_id",
                           "native_runtime_version", "lifecycle_recipe", "memory_buffer_recipe", "runtime_contract_sha256"}
            or not valid_sha(execution.get("adapter_source_sha256"))
            or not valid_sha(execution.get("source_release_sha256"))
            or not valid_sha(execution.get("runtime_contract_sha256"))
            or not isinstance(execution.get("container_image_id"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", execution["container_image_id"])
            or not isinstance(execution.get("native_runtime_version"), str) or not execution["native_runtime_version"]
            or execution.get("lifecycle_recipe") != "one_kit_fresh_seeded_scene_per_declared_cohort_v1"
            or execution.get("memory_buffer_recipe") != "original_raven_capture_lists_and_writer_v1"):
        raise ValueError("executed adapter/source/runtime/lifecycle pins required, not only Raven writer SHA")
    eligibility = bound(protocol["native_motion_eligibility"])
    expected_modes = {r["motion_id"]: r["sha256"] for r in records}
    eligible_rows = eligibility.get("motions", [])
    if (eligibility.get("schema") != "roborender.native-motion-eligibility.cpu.v1"
            or eligibility.get("cpu_only") is not True or eligibility.get("cuda_initialized") is not False
            or eligibility.get("isaac_imported") is not False
            or eligibility.get("motion_inventory_sha256") != inventory_sha
            or eligibility.get("inventory_file_sha256") != protocol["motion_inventory"]["raw_sha256"]
            or eligibility.get("eligible_count") != MOTION_COUNT
            or len(eligible_rows) != MOTION_COUNT
            or {r["motion_id"]: r["sha256"] for r in eligible_rows} != expected_modes
            or any(r.get("eligible") is not True or r.get("frame_counts_match") is not True
                   or r.get("native_skips_short_motion") is not False
                   or r.get("native_random_start_choices", 0) < 1 for r in eligible_rows)
            or eligibility.get("invalid_or_short") != [] or eligibility.get("frame_count_mismatches") != []
            or eligibility.get("native_source_hashes") != NATIVE_RESET_PINS):
        raise ValueError("all295 motions need pinned CPU native reset/range eligibility evidence")
    topology = protocol.get("execution_topology", {})
    profiles = capacity_profiles(protocol)
    if profiles is None:
        validate_topology(topology)
    requirements = {}
    for role, partition in plan["partitions"].items():
        for cohort in partition["cohorts"]:
            key = (role, cohort["num_envs"])
            ordinal = max(max(s["selected_chunk_ordinals"]) for s in cohort["slots"])
            requirements[key] = max(requirements.get(key, -1), ordinal)
    actual_sizes = {size for _, size in requirements}
    qualified_sizes, covered, bounded_sizes = set(), {}, set()
    for binding in protocol["capacity_receipts"]:
        receipt = bound(binding)
        if receipt.get("schema") == "roborender.native-bc-capacity.v2":
            validate_bounded_capacity(receipt, protocol, topology=(profiles[binding["raw_sha256"]] if profiles is not None else None))
            bounded_sizes.add(receipt["num_envs"])
            qualified_sizes.add(receipt["num_envs"])
            continue
        if profiles is not None:
            raise ValueError("per-session resource profiles require bounded v2 capacity receipts")
        if (receipt.get("schema") != "roborender.native-bc-capacity.v1"
                or receipt.get("purpose") != "full_bc_capture_and_save"
                or receipt.get("verdict") != "PASS" or receipt.get("native_runtime_tested") is not True
                or receipt.get("native_collector_sha256") != NATIVE_COLLECTOR_SHA256
                or receipt.get("model_sha256") != model_pins
                or receipt.get("collector_execution") != execution
                or receipt.get("collection_role") not in plan["partitions"]
                or receipt.get("host_id") != topology["host_id"]
                or receipt.get("concurrent_collectors") != topology["concurrent_collectors"]
                or receipt.get("simultaneous_collectors_observed") != topology["concurrent_collectors"]
                or receipt.get("cgroup_limit_bytes") != topology["cgroup_limit_bytes_per_collector"]
                or type(receipt.get("maximum_completed_ordinal_tested")) is not int
                or receipt["maximum_completed_ordinal_tested"] < 0
                or type(receipt.get("num_envs")) is not int or receipt["num_envs"] < 1
                or type(receipt.get("vector_steps")) is not int or receipt["vector_steps"] < 81
                or receipt["vector_steps"] % 81 or receipt.get("full_chunks_saved", 0) < receipt["num_envs"]
                or receipt.get("oom_events_delta") != 0 or receipt.get("swap_bytes_peak") != 0
                or type(receipt.get("cgroup_memory_peak_bytes")) is not int
                or receipt["cgroup_memory_peak_bytes"] <= 0
                or type(receipt.get("runtime_min_mem_available_bytes")) is not int
                or receipt["runtime_min_mem_available_bytes"] < topology["minimum_host_reserve_bytes"]
                or receipt["cgroup_memory_peak_bytes"] > receipt["cgroup_limit_bytes"]
                or not isinstance(receipt.get("emitted_chunks_per_lane"), list)
                or len(receipt["emitted_chunks_per_lane"]) != receipt["num_envs"]
                or any(type(n) is not int or n < receipt["maximum_completed_ordinal_tested"] + 1
                       for n in receipt["emitted_chunks_per_lane"])
                or sum(receipt["emitted_chunks_per_lane"]) != receipt["full_chunks_saved"]
                or receipt["vector_steps"] < 81 * (receipt["maximum_completed_ordinal_tested"] + 1)
                or receipt["full_chunks_saved"] < receipt["num_envs"] * (receipt["maximum_completed_ordinal_tested"] + 1)):
            raise ValueError("actual full-buffer BC capacity receipt required; teacher-step-only is insufficient")
        qualified_sizes.add(receipt["num_envs"])
        key = (receipt["collection_role"], receipt["num_envs"])
        covered[key] = max(covered.get(key, -1), receipt["maximum_completed_ordinal_tested"])
    if any(key[1] not in bounded_sizes and covered.get(key, -1) < ordinal for key, ordinal in requirements.items()):
        raise ValueError("capacity is unqualified for actual role/B/ordinal duration, including short final cohorts")
    validate_session_resources(protocol)
    verify_render_asset_identity(protocol)
    return protocol, inventory, plan, {
        "motion_inventory_canonical_sha256": inventory_sha,
        "sampling_plan_canonical_sha256": sha(canonical(plan)),
        "required_actual_num_envs": sorted(actual_sizes), "qualified_num_envs": sorted(qualified_sizes),
        "bounded_equivalence_num_envs": sorted(bounded_sizes), "long_campaign_execution_proven": False}


def validate_bounded_capacity(receipt, protocol, *, topology=None):
    """Resource qualification for an unchanged bounded-buffer runtime, not duration proof."""
    topology = protocol["execution_topology"] if topology is None else topology
    rule = dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=0.10,
                absolute_growth_slack_bytes=256 << 20)
    expected = {k: v for k, v in protocol["collector_execution"].items() if k != "source_release_sha256"}
    actual = {k: v for k, v in receipt.get("collector_execution", {}).items() if k != "source_release_sha256"}
    size = receipt.get("num_envs")
    if (protocol.get("capacity_equivalence") != rule or receipt.get("plateau_rule") != rule
            or receipt.get("purpose") != "full_bc_capture_and_save" or receipt.get("verdict") != "PASS"
            or receipt.get("native_runtime_tested") is not True or actual != expected
            or not valid_sha(receipt.get("collector_execution", {}).get("source_release_sha256"))
            or receipt.get("native_collector_sha256") != NATIVE_COLLECTOR_SHA256
            or receipt.get("model_sha256") != protocol["model_sha256"]
            or type(size) is not int or size < 1
            or receipt.get("host_id") != topology["host_id"]
            or receipt.get("concurrent_collectors") != topology["concurrent_collectors"]
            or receipt.get("simultaneous_collectors_observed") != topology["concurrent_collectors"]
            or receipt.get("cgroup_limit_bytes") != topology["cgroup_limit_bytes_per_collector"]
            or receipt.get("runtime_guard_enabled") is not True
            or receipt.get("active_guard_minimum_host_available_bytes", 0) < topology["minimum_host_reserve_bytes"]
            or receipt.get("bounded_repeated_buffer_only") is not True
            or receipt.get("oom_events_delta") != 0 or receipt.get("swap_bytes_peak") != 0):
        raise ValueError("bounded capacity runtime/topology/guard contract differs")
    samples = receipt.get("per_flush_memory", [])
    if len(samples) < 2:
        raise ValueError("bounded capacity requires repeated complete-buffer observations")
    for row in samples:
        keys = ("vector_steps", "memory_current_bytes", "memory_peak_bytes", "anon_bytes", "shmem_bytes",
                "kernel_bytes", "file_bytes", "host_available_bytes")
        emitted = row.get("emitted_chunks_per_lane", [])
        if (any(type(row.get(k)) is not int or row[k] < 0 for k in keys)
                or row["vector_steps"] < 81 or row["vector_steps"] % 81
                or len(emitted) != size or any(type(n) is not int or n < 0 for n in emitted)
                or row["host_available_bytes"] < topology["minimum_host_reserve_bytes"]
                or not 0 < row["memory_current_bytes"] <= row["memory_peak_bytes"] <= topology["cgroup_limit_bytes_per_collector"]):
            raise ValueError("bounded capacity memory/headroom evidence is invalid")
    # Every original writer call stacks full B x 81 capture buffers, including
    # lanes whose reset tail emits no complete chunk. Require cumulative native
    # coverage, not a simultaneous per-lane emission on the final two flushes.
    if min(samples[-1]["emitted_chunks_per_lane"]) < 2:
        raise ValueError("bounded capacity did not observe two native emissions per lane")
    for previous, current in zip(samples, samples[1:]):
        if (current["vector_steps"] - previous["vector_steps"] < 81
                or any(b < a for a, b in zip(previous["emitted_chunks_per_lane"], current["emitted_chunks_per_lane"]))):
            raise ValueError("bounded capacity flush/monotonic emission sequence is invalid")
    before, after = samples[-2:]
    private = lambda row: sum(row[key] for key in ("anon_bytes", "shmem_bytes", "kernel_bytes"))
    for first, last in ((before["memory_current_bytes"], after["memory_current_bytes"]), (private(before), private(after))):
        if last - first > rule["absolute_growth_slack_bytes"] + rule["maximum_growth_fraction"] * first:
            raise ValueError("bounded capacity does not satisfy the predeclared memory plateau")


def audit_production_protocol(protocol_path, protocol_sha256, rendered_manifest_path,
                              render_receipt_path=None, render_receipt_sha256=None):
    protocol, inventory, plan, summary = validate_protocol_inputs(protocol_path, protocol_sha256)
    rendered_path = Path(rendered_manifest_path).resolve(strict=True)
    rendered_raw = read_metadata(rendered_path)
    rendered = json.loads(rendered_raw)
    source = bound(protocol["source_pool"])
    cohorts = [{"document": bound(b), "sha256": b["raw_sha256"]} for b in protocol["cohort_manifests"]]
    transfers = [{"document": bound(b), "sha256": b["raw_sha256"]} for b in protocol["transfer_receipts"]]
    attestations = [bound(b) for b in protocol.get("cohort_execution_receipts", [])]
    sessions = [bound(b) for b in protocol.get("session_execution_receipts", [])]
    by_session = {sha(canonical(s)): s for s in sessions}
    qualified_session_widths = validate_session_resources(protocol)
    if not sessions or len(by_session) != len(sessions):
        raise ValueError("completed collector session evidence is missing or duplicated")
    for session in sessions:
        if (session.get("schema") != "roborender.bc-session-execution.v1"
                or session.get("verdict") != "PASS" or session.get("native_initializations") != 1
                or type(session.get("scene_initializations")) is not int or session["scene_initializations"] < 1
                or session.get("collector_execution") != protocol["collector_execution"]
                or not isinstance(session.get("request_identities"), dict)
                or len(session["request_identities"]) != session["scene_initializations"]
                or any(not valid_sha(k) or not valid_sha(v) for k, v in session["request_identities"].items())
                or any(not valid_sha(session.get(k)) for k in ("source_manifest_raw_sha256",
                    "port_manifest_raw_sha256", "launch_raw_sha256", "session_sha256",
                    "session_canonical_sha256", "native_result_raw_sha256", "host_result_raw_sha256"))):
            raise ValueError("native session lifecycle/source evidence differs")
    by_cohort = {a.get("cohort_manifest_raw_sha256"): a for a in attestations}
    if len(by_cohort) != len(cohorts) or len(attestations) != len(cohorts):
        raise ValueError("every cohort requires its own executed source/runtime attestation")
    for envelope in cohorts:
        doc = envelope["document"]
        attestation = by_cohort.get(envelope["sha256"], {})
        session = by_session.get(attestation.get("session_execution_canonical_sha256"), {})
        if (attestation.get("schema") != "roborender.bc-cohort-execution.v1"
                or attestation.get("verdict") != "PASS"
                or attestation.get("request_sha256") != doc["request_sha256"]
                or attestation.get("collector_execution") != protocol["collector_execution"]
                or attestation.get("session_execution_canonical_sha256") not in by_session
                or attestation.get("request_canonical_sha256") != sha(canonical(doc["request"]))
                or session.get("request_identities", {}).get(doc["request_sha256"]) != sha(canonical(doc["request"]))
                or attestation.get("actual_num_envs") != len(doc["request"]["motions"])
                or (protocol["schema"] == SCHEMA_V4 and qualified_session_widths.get(
                    attestation.get("session_execution_canonical_sha256")) != len(doc["request"]["motions"]))
                or attestation.get("actual_vector_steps") != doc["vector_steps"]):
            raise ValueError("cohort execution does not bind the declared source/runtime/lifecycle")
    counts = Counter(a["session_execution_canonical_sha256"] for a in attestations)
    if any(counts.get(key) != value["scene_initializations"] for key, value in by_session.items()):
        raise ValueError("cohort execution coverage differs from completed session lifecycle")
    # The original ingestion code verifies every cohort/lane, request, per-step
    # episode interval, ordinal, quota and transfer hash. No payload deserialize.
    rebuilt, _ = build_pool(plan, cohorts, transfers,
        dataset_root=Path(protocol["source_pool"]["path"]).parent,
        expected_total=plan["total_target_chunks"], verify_destination_stats=True)
    if rebuilt != source:
        raise ValueError("source pool does not reproduce exact native cohort/transfer evidence")
    if protocol["schema"] in {SCHEMA_V3, SCHEMA_V4}:
        if (source.get("artifact_roles") != plan["artifact_roles"]
                or rendered.get("artifact_roles") != plan["artifact_roles"]):
            raise ValueError("rendered source lost the frozen external success-test contract")
    actual_models = {name: asset["sha256"] for name, asset in source["model_assets"].items()}
    if actual_models != protocol["model_sha256"]:
        raise ValueError("collected teacher/Homie pins differ from declared protocol")
    if (rendered.get("schema") != "roborender.bc_pool.rendered.v2"
            or rendered.get("parent_manifest_sha256") != protocol["source_pool"]["raw_sha256"]
            or rendered.get("sampling_plan_sha256") != summary["sampling_plan_canonical_sha256"]
            or rendered.get("render_recipe") != "raven_3view_50step_independent_v2"
            or rendered.get("render_contract") != "raven_three_view_independent_v2"
            or rendered.get("max_chain_chunks") != 1
            or rendered.get("validation", {}).get("verdict") != "PASS"):
        raise ValueError("independent rendered pool does not bind the exact native source pool")
    original = {r["chunk_id"]: r for r in source["chunks"]}
    if (len(rendered.get("chunks", [])) != len(original)
            or {r["chunk_id"] for r in rendered["chunks"]} != set(original)):
        raise ValueError("rendered population size differs")
    for row in rendered["chunks"]:
        before = original.get(row["chunk_id"])
        if before is None or any(row.get(k) != value for k, value in before.items()):
            raise ValueError("rendered native provenance changed or selected ordinal differs")
        if (not valid_sha(row.get("rendered_sha256")) or row.get("render_chain_length") != 1
                or row.get("render_chain_index") != 0 or row.get("render_inherits_previous_reference") is not False):
            raise ValueError("render identity/independent ancestry missing")
    if render_receipt_path is None or not valid_sha(render_receipt_sha256):
        raise ValueError("post-render execution receipt requires a separate immutable binding")
    render_receipt = bound({"path": str(render_receipt_path), "raw_sha256": render_receipt_sha256})
    expected_rows = {r["chunk_id"]: {k: r[k] for k in ("source_sha256", "rendered_sha256")} for r in rendered["chunks"]}
    if (render_receipt.get("schema") != "roborender.bc-independent-render-execution.v1"
            or render_receipt.get("verdict") != "PASS"
            or render_receipt.get("rendered_manifest_raw_sha256") != sha(rendered_raw)
            or render_receipt.get("source_pool_raw_sha256") != protocol["source_pool"]["raw_sha256"]
            or render_receipt.get("recipe") != "raven_3view_50step_independent_v2"
            or render_receipt.get("num_inference_steps") != 50 or render_receipt.get("cfg_scale") != 5.0
            or render_receipt.get("seed") != 42
            or render_receipt.get("model_sha256") != protocol.get("render_model_sha256")
            or not isinstance(protocol.get("render_model_sha256"), dict)
            or set(protocol["render_model_sha256"]) != {"adapter", "foundation_manifest", "renderer_source", "prompt"}
            or any(not valid_sha(v) for v in protocol["render_model_sha256"].values())
            or render_receipt.get("chunks") != expected_rows):
        raise ValueError("complete independently pinned renderer execution lineage required")
    return {"path": str(Path(protocol_path).resolve()), "raw_sha256": protocol_sha256,
        "render_execution_receipt": {"path": str(Path(render_receipt_path).resolve()),
                                     "raw_sha256": render_receipt_sha256},
        **summary, "population_motion_count": len(inventory["motions"]),
        "exact_role_quotas": source["partition_counts"],
        "cohort_count": len(cohorts), "payloads_opened": False,
        "scope": "data-purpose admission; not proof of end-to-end convergence or paired concurrent capacity"}
