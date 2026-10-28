"""Image3F-only paired observation diagnostic; never a production benchmark.

Only the panel label and observation mode differ from the admitted ray-traced
200-case diagnostic. Its assignments, seeds, case order, policy and physics stay
fixed. Every cohort must match the actual baseline's physical/state digests
before its first policy action; outcomes are always newly executed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

try:
    from . import training_support_success_contract as baseline
    from .explicit_motion_panel import canonical, validate_panel, validate_result, summarize
    from .initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE
except ImportError:
    import training_support_success_contract as baseline
    from explicit_motion_panel import canonical, validate_panel, validate_result, summarize
    from initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE

SCHEMA = "roborender.bc-training-support-rendered-success-protocol.v1"
CONFIG_SCHEMA = "roborender.rendered-training-support-config.v1"
BASELINE_SCHEMA = "roborender.rendered-training-support-paired-baseline.v1"
MODE = "image3f_3view"
PANEL_ID = "bc116-training-support-image3f-200-20261023"
PANEL_SHA256 = "cc052a9717ac7579565f117342f05b1721171cae23394f67fec848249e3707b4"
PANEL_RAW_SHA256 = "9fb272f27b8a2c09088f3ddb5a9a749609319fc1f1748a28b221c4afe610b351"
BASELINE_RESULT_RAW_SHA256 = "dcb6a6c6ca1e7f1f06a3ef80d072fc096130f1ac9325861969db5c422ce3ecbd"
BASELINE_MANIFEST_RAW_SHA256 = "731d619ec5a15cfee20cabb58854960925f124ed3893ec471e377eb848eced9c"
PAIRED_BASELINE_RAW_SHA256 = "9c6428d66e68e35003d7bca87fd628c956182423351e1d16bc48b7e86341e617"
ADAPTER_SHA256 = "c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588"
STATE_FIELDS = {"actor_obs", "dof_pos", "dof_vel", "episode_length_buf", "front_object",
    "gt_obj_pos", "hold_object", "homie_obs", "robot_root", "sample_idx", "sample_idx_in_seq",
    "stage_buf", "teacher_obs", "time_in_stage_buf", "tray_object"}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def raytraced_shape(panel):
    """Generic receipt checks normalize only vision/label, not assignments.

    Singleton panels used internally to validate a cohort preserve the original
    assignment run_id; the full execution gate below still pins all200 cases.
    """
    if not isinstance(panel, dict) or panel.get("observation_mode") != MODE or panel.get("panel_id") != PANEL_ID:
        raise ValueError("unknown rendered success panel")
    value = deepcopy(panel)
    value["observation_mode"] = "raytraced_3view"
    value["panel_id"] = baseline.PANEL_ID
    return value


def build_panel(raytraced_panel):
    value = baseline.require_panel(raytraced_panel)
    value["observation_mode"] = MODE
    value["panel_id"] = PANEL_ID
    return require_panel(value)


def require_panel(panel, *, raw_sha256=None):
    value = validate_panel(panel)
    if (_digest(value) != PANEL_SHA256 or value["episode_count"] != 200
            or (raw_sha256 is not None and raw_sha256 != PANEL_RAW_SHA256)):
        raise ValueError("exact paired Image3F200 panel required")
    baseline.require_panel(raytraced_shape(value))
    return value


def evaluation_options(panel, *, raw_sha256=None):
    require_panel(panel, raw_sha256=raw_sha256)
    return {"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE}


def physics_sha256(value):
    # Match the already executed baseline formatter, not a new canonical form.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_baseline(bundle, panel):
    require_panel(panel)
    ray = raytraced_shape(panel)
    if (not isinstance(bundle, dict) or set(bundle) != {"schema", "baseline_panel_sha256",
            "baseline_result_raw_sha256", "baseline_reference_manifest_raw_sha256", "baseline_result", "cohorts"}
            or bundle["schema"] != BASELINE_SCHEMA
            or bundle["baseline_panel_sha256"] != baseline.PANEL_SHA256
            or bundle["baseline_result_raw_sha256"] != BASELINE_RESULT_RAW_SHA256
            or bundle["baseline_reference_manifest_raw_sha256"] != BASELINE_MANIFEST_RAW_SHA256
            or not isinstance(bundle["cohorts"], list) or len(bundle["cohorts"]) != 100):
        raise ValueError("actual complete paired ray-traced baseline required")
    baseline.validate_training_support_result(ray, bundle["baseline_result"], baseline.CHECKPOINT_SHA256)
    if hashlib.sha256(canonical(bundle) + b"\n").hexdigest() != PAIRED_BASELINE_RAW_SHA256:
        raise ValueError("exact independently preserved baseline state/physics bundle required")
    if hashlib.sha256(canonical(bundle["baseline_result"]) + b"\n").hexdigest() != BASELINE_RESULT_RAW_SHA256:
        raise ValueError("actual completed ray-traced baseline result bytes differ")
    native = []
    for index, row in enumerate(bundle["cohorts"]):
        if (not isinstance(row, dict) or set(row) != {"cohort_index", "initial_state", "runtime_physics", "result", "source_files"}
                or row["cohort_index"] != index or set(row["initial_state"]) != STATE_FIELDS
                or set(row["source_files"]) != {"initial_state", "runtime_physics", "result"}):
            raise ValueError("paired baseline cohort coverage/state identity differs")
        for descriptor in row["initial_state"].values():
            if (not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "sha256"}
                    or not _sha(descriptor["sha256"]) or not isinstance(descriptor["shape"], list)
                    or not descriptor["shape"] or descriptor["shape"][0] != 2
                    or not isinstance(descriptor["dtype"], str)):
                raise ValueError("paired baseline state tensor descriptor differs")
        for descriptor in row["source_files"].values():
            if (not isinstance(descriptor, dict) or set(descriptor) != {"raw_sha256", "bytes"}
                    or not _sha(descriptor["raw_sha256"]) or type(descriptor["bytes"]) is not int
                    or not 0 < descriptor["bytes"] <= 1 << 20):
                raise ValueError("paired baseline original-file provenance missing")
        initial = {name: {"dtype": value["dtype"], "shape": value["shape"], "sha256": value["sha256"]}
                   for name, value in sorted(row["initial_state"].items())}
        original_bytes = {"initial_state": json.dumps(initial, indent=2).encode() + b"\n",
            "runtime_physics": json.dumps(row["runtime_physics"], sort_keys=True, allow_nan=False).encode() + b"\n",
            "result": canonical(row["result"]) + b"\n"}
        for name, raw in original_bytes.items():
            if row["source_files"][name] != {"raw_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}:
                raise ValueError("baseline bundle differs from original native file bytes")
        if (row["result"].get("runtime_physics_sha256") != physics_sha256(row["runtime_physics"])
                or row["runtime_physics"].get("physics_unchanged") is not True):
            raise ValueError("paired baseline actual physics evidence differs")
        native.append(row["result"])
    aggregate = summarize(ray, native)
    for key, value in aggregate.items():
        if bundle["baseline_result"].get(key) != value:
            raise ValueError("baseline cohort receipts disagree with actual completed ray-traced result")
    return deepcopy(bundle)


def validate_renderer_identity(identity):
    if (not isinstance(identity, dict) or identity.get("renderer") != "Image3F"
            or identity.get("adapter_sha256") != ADAPTER_SHA256 or not _sha(identity.get("engine_sha256"))
            or identity.get("steps") != 5 or identity.get("cfg") != 1.0
            or identity.get("teacache_threshold") != .3 or identity.get("seed") != 42
            or identity.get("server_rng_isolated") is not True):
        raise ValueError("exact genuine Image3F5 renderer identity required")
    canonical(identity)


def validate_config(config, panel):
    require_panel(panel)
    if (not isinstance(config, dict) or set(config) != {"schema", "checkpoint_sha256", "source_selection_sha256",
            "panel_sha256", "renderer", "anchors", "paired_baseline_file", "paired_baseline_sha256"}
            or config["schema"] != CONFIG_SCHEMA or config["checkpoint_sha256"] != baseline.CHECKPOINT_SHA256
            or config["source_selection_sha256"] != baseline.SOURCE_SELECTION_RAW_SHA256
            or config["panel_sha256"] != PANEL_SHA256
            or config["paired_baseline_sha256"] != PAIRED_BASELINE_RAW_SHA256):
        raise ValueError("bound rendered diagnostic configuration differs")
    renderer, anchors = config["renderer"], config["anchors"]
    if (not isinstance(renderer, dict) or not isinstance(anchors, dict)
            or set(renderer) != {"host", "port", "identity"} or renderer["host"] != "127.0.0.1"
            or type(renderer["port"]) is not int or not 1024 <= renderer["port"] <= 65535
            or set(anchors) != {"files", "seed"} or type(anchors["seed"]) is not int
            or not 0 <= anchors["seed"] < 2 ** 32 or not isinstance(anchors["files"], list)
            or len(anchors["files"]) != 2048):
        raise ValueError("loopback renderer and full2048 frozen anchor pool required")
    validate_renderer_identity(renderer["identity"])
    for row in anchors["files"]:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts or not _sha(row["sha256"])):
            raise ValueError("hash-bound absolute anchor file required")
    if len({r["path"] for r in anchors["files"]}) != 2048:
        raise ValueError("duplicate anchor path")
    ref = Path(config["paired_baseline_file"])
    if ref.is_absolute() or ".." in ref.parts or ref.name in {"", "."}:
        raise ValueError("paired baseline must be an owned jobs-relative artifact")
    return deepcopy(config)


def read_json(path, raw_sha256, *, limit=16 << 20):
    path = Path(path)
    if not _sha(raw_sha256) or path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("bounded hash-bound regular rendered diagnostic artifact required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError("rendered diagnostic artifact bytes changed")
    return json.loads(raw)


def read_inputs(config_path, config_sha256, panel, *, artifact):
    config = validate_config(read_json(config_path, config_sha256), panel)
    reference = validate_baseline(read_json(artifact(config["paired_baseline_file"]),
                                           config["paired_baseline_sha256"]), panel)
    return config, reference


def validate_execution(panel, selection, checkpoint_sha256, config, reference, *, arm="bc", has_resume=False):
    require_panel(panel)
    if (arm != "bc" or checkpoint_sha256 != baseline.CHECKPOINT_SHA256 or has_resume
            or build_panel(baseline.build_panel(selection)) != panel):
        raise ValueError("rendered diagnostic requires exact fresh BC116 paired200, without result reuse")
    validate_config(config, panel)
    validate_baseline(reference, panel)


def compare_initial_state(reference, *, cohort_index, initial_state, runtime_physics):
    """Called synchronously after reset and BEFORE rendering/policy/physics."""
    if type(cohort_index) is not int or not 0 <= cohort_index < 100:
        raise ValueError("invalid paired cohort index")
    expected = reference["cohorts"][cohort_index]
    if (expected["cohort_index"] != cohort_index or expected["initial_state"] != initial_state
            or expected["runtime_physics"] != runtime_physics):
        raise RuntimeError("paired initial physical/state evidence differs; no first action permitted")
    return {"schema": "roborender.rendered-paired-initial-state.v1", "cohort_index": cohort_index,
        "baseline_result_raw_sha256": reference["baseline_result_raw_sha256"],
        "initial_state_sha256": _digest(initial_state), "runtime_physics_sha256": physics_sha256(runtime_physics),
        "exact_initial_state_match": True, "exact_physics_match": True,
        "checked_before_first_action": True, "physical_steps_before_check": 0}


def validate_cohort_observations(result, *, count):
    receipt, paired = result.get("rendered_observation_receipt"), result.get("paired_initial_state_attestation")
    steps = result.get("vector_steps")
    if (not isinstance(receipt, dict) or receipt.get("schema") != "roborender.rendered-success-observation-receipt.v1"
            or receipt.get("observation_mode") != MODE or type(steps) is not int or not 1 <= steps <= 450
            or receipt.get("vector_steps_rendered") != steps or receipt.get("render_requests_completed") != count * steps
            or receipt.get("raytraced_fallbacks") != 0 or receipt.get("failed") is not False
            or receipt.get("view_order") != ["head_color", "hand_left", "hand_right"]
            or receipt.get("normalizations_per_vector_step") != 1
            or receipt.get("anchor_policy") != "fixed_frame0_per_episode"
            or receipt.get("offline_BC_teacher50_equivalent") is not False
            or receipt.get("controls_from_current_simulator_state") is not True
            or receipt.get("simulator_steps_added") != 0 or receipt.get("optimizer_steps") != 0
            or not _sha(receipt.get("trace_sha256"))
            or not isinstance(paired, dict) or paired.get("schema") != "roborender.rendered-paired-initial-state.v1"
            or paired.get("baseline_result_raw_sha256") != BASELINE_RESULT_RAW_SHA256
            or type(paired.get("cohort_index")) is not int or not 0 <= paired["cohort_index"] < 100
            or paired.get("exact_initial_state_match") is not True or paired.get("exact_physics_match") is not True
            or paired.get("checked_before_first_action") is not True or paired.get("physical_steps_before_check") != 0
            or not _sha(paired.get("initial_state_sha256"))
            or paired.get("runtime_physics_sha256") != result.get("runtime_physics_sha256")):
        raise ValueError("rendered cohort lacks exact all-lane generation and pre-action paired-state evidence")
    validate_renderer_identity(receipt.get("renderer_identity"))


def protocol_identity(panel, config):
    validate_config(config, panel)
    value = {"schema": SCHEMA, "panel_sha256": PANEL_SHA256, "panel_raw_sha256": PANEL_RAW_SHA256,
        "baseline_panel_sha256": baseline.PANEL_SHA256, "baseline_result_raw_sha256": BASELINE_RESULT_RAW_SHA256,
        "source_selection_raw_sha256": baseline.SOURCE_SELECTION_RAW_SHA256,
        "checkpoint_sha256": baseline.CHECKPOINT_SHA256, "observation_mode": MODE,
        "configuration_sha256": _digest(config), "renderer_identity": deepcopy(config["renderer"]["identity"]),
        "anchor_inventory_sha256": _digest(config["anchors"]["files"]), "anchor_seed": config["anchors"]["seed"],
        "cohort_anchor_seed_rule": "(anchor_seed + original_cohort_index) mod 2**32; independent RandomState",
        "paired_baseline_sha256": config["paired_baseline_sha256"], "cases": 200, "cohorts": 100,
        "assignments_seeds_case_ids_unchanged": True, "exact_initial_state_and_physics_required": True,
        "teacher_action_probability": 0, "optimizer_steps": 0, "max_episode_seconds": 8,
        "control_hz": 50, "physics_hz": 200, "goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
        "diagnostic_only": True, "exact_historical_replay": False, "fixed295_benchmark": False,
        "production_promotion_allowed": False, "offline_BC_teacher50_equivalent": False}
    return {**value, "sha256": _digest(value)}


def validate_rendered_result(panel, result, checkpoint_sha256, config, reference):
    validate_config(config, panel)
    validate_baseline(reference, panel)
    if (checkpoint_sha256 != baseline.CHECKPOINT_SHA256
            or result.get("rendered_training_support_protocol") != protocol_identity(panel, config)
            or result.get("diagnostic_only") is not True or result.get("exact_historical_replay") is not False
            or {"production_success_protocol", "smoke_success_protocol", "training_support_success_protocol"} & set(result)):
        raise ValueError("rendered result protocol/scope differs")
    validate_result(panel, result, checkpoint_sha256, allowed_arms=("bc",), **evaluation_options(panel))
    receipts, pairs = result.get("rendered_observation_receipts"), result.get("paired_initial_state_attestations")
    if not isinstance(receipts, list) or not isinstance(pairs, list) or len(receipts) != 100 or len(pairs) != 100:
        raise ValueError("complete rendered and paired cohort evidence required")
    for index, (receipt, paired) in enumerate(zip(receipts, pairs)):
        expected = compare_initial_state(reference, cohort_index=index,
            initial_state=reference["cohorts"][index]["initial_state"],
            runtime_physics=reference["cohorts"][index]["runtime_physics"])
        if (paired != expected or receipt["renderer_identity"] != config["renderer"]["identity"]
                or receipt.get("anchor_seed") != (config["anchors"]["seed"] + index) % (2 ** 32)):
            raise ValueError("per-cohort rendered/paired evidence differs from admitted configuration")
        anchor_hashes = {row["sha256"] for row in config["anchors"]["files"]}
        anchors = receipt.get("anchor_selections")
        if (not isinstance(anchors, list) or len(anchors) < 2
                or any(row.get("file_sha256") not in anchor_hashes or row.get("frame_index") != 0
                       or row.get("lane") not in (0, 1) or not _sha(row.get("prompt_sha256"))
                       for row in anchors)
                or {row["lane"] for row in anchors if row.get("selected_at_vector_step") == 0} != {0, 1}):
            raise ValueError("rendered cohort anchor evidence differs from admitted frame-zero pool")
    return deepcopy(result)
