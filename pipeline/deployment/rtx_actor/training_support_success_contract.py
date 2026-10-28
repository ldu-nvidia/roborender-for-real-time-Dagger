"""One separately admitted BC116 training-support fresh-reset diagnostic.

The immutable draw selects training CHUNKS, not restorable simulator states.
Repeated motions/episodes retain their original sampling weight. This contract
cannot promote a checkpoint or replace the fixed295 stage-end benchmark.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

try:
    from .explicit_motion_panel import canonical, panel_sha256, validate_panel, validate_result
    from .initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE, control_contract
except ImportError:
    from explicit_motion_panel import canonical, panel_sha256, validate_panel, validate_result
    from initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE, control_contract

SCHEMA = "roborender.bc-training-support-success-protocol.v1"
PANEL_ID = "bc116-training-support-fresh-reset-200-20261023"
SOURCE_SELECTION_RAW_SHA256 = "c3751f38cb2d523428504298ea4ee0491112db5af31caa2defbbcc3c67d220ab"
CHECKPOINT_SHA256 = "2ec87a3c7b76aaa8fbb57d4fd50d02335dbfed75c164f5d741bea0067a33d7a9"
TRAIN_INDEX_RAW_SHA256 = "9e379a173b0fcceca5770d87c36b3cc868445d27ebb062f7edd049583a9d9a76"
MOTION_INVENTORY_SHA256 = "b35f5b7459204c24aea0cb2201b21faf9f75d44124fefa1ac8e1d3f2855e4d74"
MOTION_ROOT = "/workspace/rtx_BC_transfer/pipeline/GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775"
CASE_COUNT = 200
# Canonical panel and canonical-plus-newline file identities are frozen below.
PANEL_SHA256 = "051c31fd60daa738df632f4f8cb7a15db3ea8455a8fc235e1e018ea8f81a057c"
PANEL_RAW_SHA256 = "1c7ac3b68a16e231988640da118767b74d270838f57dbafedd8feb0af1d3d9e4"


def validate_source_selection(value):
    """Require the original frozen selection; do not relabel its old approval flags."""
    if (not isinstance(value, dict)
            or hashlib.sha256(canonical(value) + b"\n").hexdigest() != SOURCE_SELECTION_RAW_SHA256
            or value.get("schema") != "roborender.bc-train-chunk-source-selection.v1"
            or value.get("selected_epoch") != 116
            or value.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
            or value.get("bindings", {}).get("train_index", {}).get("raw_sha256") != TRAIN_INDEX_RAW_SHA256):
        raise ValueError("exact frozen BC116 training-source selection required")
    sampling, samples = value["sampling"], value["samples"]
    if (sampling["unit"] != "admitted_training_chunk" or sampling["sample_size"] != CASE_COUNT
            or sampling["population_size"] != 2560 or sampling["seed"] != 20261023
            or sampling["without_replacement"] is not True or sampling["deduplication_applied"] is not False
            or sampling["sample_order"] != "random_draw_order" or len(samples) != CASE_COUNT
            or len({row["train_index_item"]["chunk_id"] for row in samples}) != CASE_COUNT):
        raise ValueError("training-source draw unit/order/count differs")
    for index, row in enumerate(samples):
        record = row["source_record"]
        if (row["draw_index"] != index
                or row["admitted_train_index"] != sampling["selected_index_ordinals"][index]
                or record["chunk_id"] != row["train_index_item"]["chunk_id"]
                or record["split"] != "train" or record["selected_for_split"] is not True
                or record["overcollection"] is not False or record["teacher_only"] is not True
                or record["frames"] != 81):
            raise ValueError("sample is not the exact admitted training chunk")
    return deepcopy(value)


def read_source_selection(path, raw_sha256):
    path = Path(path)
    if (raw_sha256 != SOURCE_SELECTION_RAW_SHA256 or path.is_symlink()
            or not path.is_file() or path.stat().st_size > 1 << 20):
        raise ValueError("bounded frozen training-source selection file required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SELECTION_RAW_SHA256:
        raise ValueError("training-source selection bytes changed")
    return validate_source_selection(json.loads(raw))


def cohort_seed(index):
    """New deterministic reset seeds, not historical BC process/RNG replay."""
    if type(index) is not int or not 0 <= index < 100:
        raise ValueError("training-support cohort index must be 0..99")
    digest = hashlib.sha256(canonical([SCHEMA, SOURCE_SELECTION_RAW_SHA256,
                                      "fresh_cohort_seed", index])).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def build_panel(selection):
    selection = validate_source_selection(selection)
    panel = {"schema": "roborender.explicit-motion-success-panel.v1",
        "panel_id": PANEL_ID, "purpose": "test", "motion_inventory_sha256": MOTION_INVENTORY_SHA256,
        "observation_mode": "raytraced_3view", "action_mode": "deterministic_action_mean_11_plus_homie_15",
        "metric": "native_goal_or_until_done", "max_vector_steps_per_cohort": 450,
        "episode_count": CASE_COUNT, "cohorts": []}
    for index in range(100):
        records = [row["source_record"] for row in selection["samples"][2 * index:2 * index + 2]]
        assignment = {"schema": "roborender.native-motion-cohort.v1", "run_id": PANEL_ID,
            "cohort_id": f"training-support-{index:03d}", "purpose": "test", "simulator_seed": cohort_seed(index),
            "motion_inventory_sha256": MOTION_INVENTORY_SHA256,
            "lanes": [{"lane": lane, "motion_id": row["motion_id"], "sha256": row["motion_sha256"],
                       "path": str(Path(MOTION_ROOT) / (row["motion_id"] + ".npz"))}
                      for lane, row in enumerate(records)]}
        # The accepted draw has no identical motions within an adjacent pair.
        # Never deduplicate/reorder it to evade the shared native lane gate.
        panel["cohorts"].append({"assignment": assignment,
            "case_ids": [f"train-draw-{2 * index + lane:03d}-{row['chunk_id'].replace('_', '-')}"
                         for lane, row in enumerate(records)]})
    return validate_panel(panel)


def require_panel(panel, *, raw_sha256=None):
    value = validate_panel(panel)
    if (value["panel_id"] != PANEL_ID or value["purpose"] != "test"
            or value["episode_count"] != CASE_COUNT or len(value["cohorts"]) != 100
            or any(len(row["case_ids"]) != 2 for row in value["cohorts"])
            or panel_sha256(value) != PANEL_SHA256
            or (raw_sha256 is not None and raw_sha256 != PANEL_RAW_SHA256)):
        raise ValueError("exact frozen 200-case training-support diagnostic panel required")
    return value


def evaluation_options(panel, *, raw_sha256=None):
    require_panel(panel, raw_sha256=raw_sha256)
    return {"teacher_action_mode": "distill", "diagnostic_initial_robot_refresh": STUDENT_CLEAR_ONCE}


def validate_execution(panel, selection, checkpoint_sha256, *, arm="bc", has_resume=False):
    """Pre-native gate, also called by the evaluator independently of resident admission."""
    require_panel(panel)
    if (arm != "bc" or checkpoint_sha256 != CHECKPOINT_SHA256 or has_resume
            or build_panel(selection) != panel):
        raise ValueError("training-support diagnostic requires fresh exact BC116/source-panel execution")


def protocol_identity(panel):
    require_panel(panel)
    value = {"schema": SCHEMA, "panel_raw_sha256": PANEL_RAW_SHA256, "panel_sha256": PANEL_SHA256,
        "source_selection_raw_sha256": SOURCE_SELECTION_RAW_SHA256,
        "training_index_raw_sha256": TRAIN_INDEX_RAW_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_sampling_unit": "admitted_training_chunk", "source_selection_seed": 20261023,
        "source_draw_order_preserved": True, "deduplication_applied": False,
        "completed_episodes": CASE_COUNT, "cohort_sizes": [2] * 100,
        "case_semantics": "fresh_native_reset_on_sampled_training_motion_not_historical_chunk_state",
        "exact_historical_replay": False, "diagnostic_only": True,
        "fixed295_benchmark": False, "production_promotion_allowed": False,
        "observation_mode": "raytraced_3view", "max_episode_seconds": 8,
        "control_hz": 50, "physics_hz": 200, "metric": "native_goal_or_until_done",
        "goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
        "behavior": "deterministic_student_action_mean_11_plus_homie_15",
        "teacher_action_probability": 0, "optimizer_steps": 0,
        "reset_control": control_contract(STUDENT_CLEAR_ONCE)}
    return {**value, "sha256": hashlib.sha256(canonical(value)).hexdigest()}


def validate_training_support_result(panel, result, checkpoint_sha256, *, allowed_arms=("bc",)):
    options = evaluation_options(panel)
    if (tuple(allowed_arms) != ("bc",) or checkpoint_sha256 != CHECKPOINT_SHA256
            or not isinstance(result, dict)
            or result.get("training_support_success_protocol") != protocol_identity(panel)
            or result.get("diagnostic_only") is not True
            or result.get("exact_historical_replay") is not False
            or "production_success_protocol" in result or "smoke_success_protocol" in result):
        raise ValueError("training-support result identity/scope differs; no production promotion")
    return validate_result(panel, result, checkpoint_sha256, allowed_arms=("bc",), **options)
