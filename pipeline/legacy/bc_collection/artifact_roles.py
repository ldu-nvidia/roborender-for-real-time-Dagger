"""Explicit offline MSE roles and a separate frozen native success-test panel."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pipeline.deployment.rtx_actor.explicit_motion_panel import validate_panel

PLAN_SCHEMA = "roborender.coverage-sampling-plan.v3"
CONTRACT_SCHEMA = "roborender.bc-artifact-roles.v1"
OFFLINE_ROLES = ("train", "validation")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def panel_reference(binding, *, inventory_sha256=None, motion_bindings=None, forbidden_seeds=()):
    if (not isinstance(binding, dict) or set(binding) != {"path", "raw_sha256"}
            or not isinstance(binding["raw_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", binding["raw_sha256"])
            or not Path(binding["path"]).is_absolute()):
        raise ValueError("success test requires an absolute immutable panel/raw SHA binding")
    path = Path(binding["path"])
    if path.stat().st_size > 32 << 20:
        raise ValueError("success panel metadata exceeds bounded limit")
    with path.open("rb") as stream:
        raw = stream.read((32 << 20) + 1)
    if len(raw) > 32 << 20 or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]:
        raise ValueError("success panel bytes changed")
    panel = validate_panel(json.loads(raw))
    if panel["purpose"] != "test" or (inventory_sha256 is not None and panel["motion_inventory_sha256"] != inventory_sha256):
        raise ValueError("final success panel must be test-only and bind the same inventory")
    seeds = {c["assignment"]["simulator_seed"] for c in panel["cohorts"]}
    if seeds & set(forbidden_seeds):
        raise ValueError("success-test simulator seeds overlap offline training/validation")
    actual, counts = {}, Counter()
    for cohort in panel["cohorts"]:
        for lane in cohort["assignment"]["lanes"]:
            if actual.setdefault(lane["motion_id"], lane["sha256"]) != lane["sha256"]:
                raise ValueError("one success-test motion has different bytes")
            counts[lane["motion_id"]] += 1
    if motion_bindings is not None and actual != motion_bindings:
        raise ValueError("success-test motion support/bytes differs from the offline population")
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("success-test motion weighting must be predeclared balanced case counts")
    return dict(panel=binding, panel_canonical_sha256=hashlib.sha256(canonical(panel)).hexdigest(),
        purpose="test_only_not_checkpoint_selection", metric=panel["metric"],
        episode_count=panel["episode_count"], motion_inventory_sha256=panel["motion_inventory_sha256"],
        motion_bindings=actual, motion_case_counts=dict(sorted(counts.items())),
        simulator_seeds=sorted(seeds), observation_mode=panel["observation_mode"],
        executed=False, result_required_after_checkpoint_selection=True)


def offline_roles(document):
    contract = document.get("artifact_roles")
    if contract is None:
        return ("train", "validation", "test")
    if (contract.get("schema") != CONTRACT_SCHEMA or contract.get("offline_roles") != list(OFFLINE_ROLES)
            or contract.get("offline_test_chunks") != 0
            or set(contract.get("offline_chunk_counts", {})) != set(OFFLINE_ROLES)):
        raise ValueError("unknown/malformed explicit offline artifact roles")
    counts = contract["offline_chunk_counts"]
    if (any(type(v) is not int or v <= 0 for v in counts.values())
            or counts["train"] % 16 or counts["train"] != counts["validation"] * 8):
        raise ValueError("offline quotas require full B16 and declared 8:1 train/validation ratio")
    return OFFLINE_ROLES


def validate_artifact_contract(document, *, motion_bindings=None, forbidden_seeds=()):
    if offline_roles(document) != OFFLINE_ROLES:
        raise ValueError("explicit two-offline-role contract required")
    contract = document["artifact_roles"]
    expected = panel_reference(contract["closed_loop_test"]["panel"],
        inventory_sha256=contract["closed_loop_test"]["motion_inventory_sha256"],
        motion_bindings=motion_bindings, forbidden_seeds=forbidden_seeds)
    if contract["closed_loop_test"] != expected:
        raise ValueError("closed-loop test metadata differs from its frozen native panel")
    return contract
