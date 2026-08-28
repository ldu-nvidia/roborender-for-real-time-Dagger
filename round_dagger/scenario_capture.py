"""Fail-closed attestation for live held-out Image3F scenario capture.

The scenario contract used by paired evaluation cannot be planned from motion
filenames alone: its identity includes the first condition bytes produced by
the exact Isaac/Image3F stack.  This module validates two independent capture
passes for each planned seed block, verifies renderer counter deltas, requires
repeatable condition/scenario identities, and publishes the raw evidence format
consumed by ``build_heldout_evaluation_plan.py finalize``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "fast_roborender.round_dagger.heldout_motion_capture_plan.v2"
RUNTIME_SCHEMA = "fast_roborender.round_dagger.scenario_capture_runtime.v1"
PASS_SCHEMA = "fast_roborender.round_dagger.scenario_capture_pass.v1"
ATTESTATION_SCHEMA = "fast_roborender.round_dagger.scenario_capture_attestation.v1"
EVIDENCE_SCHEMA = "fast_roborender.round_dagger.motion_capture_evidence.v1"
EXPECTED_SEEDS = (17, 29, 43, 59, 71, 89, 101, 127)
EXPECTED_ENVS_PER_SEED = 32
EXPECTED_CASES = 256
EXPECTED_PASSES = (1, 2)
POLICY_VIEWS = ("head_color", "hand_left", "hand_right")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
NONCE_RE = re.compile(r"[0-9a-f]{32}")

PLANNED_CASE_KEYS = {
    "seed",
    "env_index",
    "global_case_index",
    "planned_case_id",
    "motion_relative_path",
    "motion_sha256",
}
RUNTIME_CASE_KEYS = PLANNED_CASE_KEYS | {
    "first_condition_sha256",
    "first_generated_sha256",
    "scenario_id",
}
PROVENANCE_KEYS = {
    "fast_roborender_revision",
    "gr00t_revision",
    "actor_source_inventory_sha256",
    "evaluation_source_inventory_sha256",
    "gr00t_source_tree_sha256",
    "heldout_plan_sha256",
    "motion_split_manifest_sha256",
    "evaluation_config_sha256",
    "renderer_runtime_sha256",
    "camera_acceptance_sha256",
    "anchor_inventory_sha256",
    "trainer_container_sha256",
    "renderer_cohort_sha256",
    "parent_checkpoint_sha256",
    "parent_runtime_policy_sha256",
}
RUNTIME_KEYS = {
    "schema",
    "plan_sha256",
    "seed",
    "pass_index",
    "capture_run_id",
    "capture_nonce",
    "capture_profile",
    "release_eligible",
    "expected_env_count",
    "cases",
    "execution",
    "runtime",
    "renderer_streams",
    "provenance",
}
EXECUTION_DOCUMENT = {
    "capture_mode": True,
    "paired_evaluation": False,
    "eval_relabel_enabled": True,
    "training_enabled": False,
    "policy_step_calls": 1,
    "env_step_calls": 0,
    "generated_policy_steps": 1,
    "renderer_errors_delta": 0,
    "renderer_replays_delta": 0,
}
RUNTIME_CONFIG_KEYS = {
    "num_envs",
    "generation_seed",
    "motion_split_role",
    "motion_split_offset",
    "motion_split_limit",
    "policy_views",
    "view_layout",
    "reference_lag",
    "raytraced_mix_prob",
    "anchor_source",
}
STREAM_KEYS = {
    "env_index",
    "endpoint_file",
    "server_id",
    "server_incarnation_id",
}
PASS_KEYS = {
    "schema",
    "runtime_artifact_sha256",
    "renderer_before_sha256",
    "renderer_after_sha256",
    "served_requests",
    "renderer_runtime_sha256",
    "expected_renderer_count",
    "runtime",
    "renderer_before",
    "renderer_after",
}


class ScenarioCaptureError(RuntimeError):
    """Live capture evidence or its immutable assignment is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ScenarioCaptureError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ScenarioCaptureError(f"{label} must be an integer")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ScenarioCaptureError(f"{label} fields are not exact")
    return dict(value)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScenarioCaptureError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ScenarioCaptureError(f"non-finite JSON constant is forbidden: {value}")


def load_strict_json(path: str | os.PathLike[str], label: str) -> tuple[Any, bytes]:
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ScenarioCaptureError(f"cannot safely open {label}: {exc}") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ScenarioCaptureError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise ScenarioCaptureError(f"{label} changed while it was read")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ScenarioCaptureError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioCaptureError(f"cannot decode {label}: {exc}") from None
    return document, payload


def _publish_immutable(path: str | os.PathLike[str], document: Any) -> None:
    target = Path(path)
    payload = canonical_bytes(document) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        observed, observed_payload = load_strict_json(target, "existing output")
        if observed == document and observed_payload == payload:
            return
        raise ScenarioCaptureError(f"refusing to replace different output: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(temporary, target, follow_symlinks=False)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_planned_case(raw: Any, *, seed: int, env_index: int) -> dict[str, Any]:
    case = _exact(raw, PLANNED_CASE_KEYS, "planned capture case")
    if case["seed"] != seed or case["env_index"] != env_index:
        raise ScenarioCaptureError("planned capture seed/env assignment mismatch")
    _integer(case["global_case_index"], "global_case_index", minimum=0)
    _sha(case["planned_case_id"], "planned_case_id")
    _sha(case["motion_sha256"], "motion_sha256")
    relative = case["motion_relative_path"]
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ScenarioCaptureError("motion relative path is unsafe")
    return case


def validate_plan_document(
    document: Any,
    *,
    plan_sha256: str,
    production: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(document, Mapping) or document.get("schema") != PLAN_SCHEMA:
        raise ScenarioCaptureError("held-out capture plan schema mismatch")
    capture = document.get("capture")
    if not isinstance(capture, Mapping) or not isinstance(capture.get("cases"), list):
        raise ScenarioCaptureError("held-out capture plan has no case assignments")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_global: set[int] = set()
    seen_motion: set[str] = set()
    for raw in capture["cases"]:
        if not isinstance(raw, Mapping):
            raise ScenarioCaptureError("planned capture case is not an object")
        seed = _integer(raw.get("seed"), "planned seed")
        env_index = _integer(raw.get("env_index"), "planned env index", minimum=0)
        case = _validate_planned_case(raw, seed=seed, env_index=env_index)
        identity = case["global_case_index"]
        motion = case["motion_sha256"]
        if identity in seen_global or motion in seen_motion:
            raise ScenarioCaptureError("held-out plan repeats a case or motion")
        seen_global.add(identity)
        seen_motion.add(motion)
        groups[seed].append(case)
    for seed, cases in groups.items():
        cases.sort(key=lambda case: case["env_index"])
        if [case["env_index"] for case in cases] != list(range(len(cases))):
            raise ScenarioCaptureError(f"seed {seed} env indices are not contiguous")
        indices = [case["global_case_index"] for case in cases]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ScenarioCaptureError(f"seed {seed} global case block is not contiguous")
    if production:
        if tuple(sorted(groups)) != EXPECTED_SEEDS:
            raise ScenarioCaptureError("production capture plan seeds are not exact")
        if any(len(groups[seed]) != EXPECTED_ENVS_PER_SEED for seed in EXPECTED_SEEDS):
            raise ScenarioCaptureError("production capture plan is not exact 8x32")
        if len(seen_global) != EXPECTED_CASES:
            raise ScenarioCaptureError("production capture plan does not cover 256 cases")
    _sha(plan_sha256, "plan SHA-256")
    return dict(groups)


def _scenario_id(seed: int, env_index: int, condition_sha: str) -> str:
    return canonical_sha256(
        {
            "seed": seed,
            "env_index": env_index,
            "first_condition_sha256": condition_sha,
        }
    )


def validate_runtime_document(
    document: Any,
    *,
    plan: Any,
    plan_sha256: str,
    production: bool = True,
) -> dict[str, Any]:
    runtime = _exact(document, RUNTIME_KEYS, "scenario capture runtime")
    if runtime["schema"] != RUNTIME_SCHEMA:
        raise ScenarioCaptureError("scenario capture runtime schema mismatch")
    if runtime["plan_sha256"] != plan_sha256:
        raise ScenarioCaptureError("runtime capture is bound to another plan")
    groups = validate_plan_document(plan, plan_sha256=plan_sha256, production=production)
    seed = _integer(runtime["seed"], "capture seed")
    pass_index = _integer(runtime["pass_index"], "capture pass index", minimum=1)
    if seed not in groups or (production and pass_index not in EXPECTED_PASSES):
        raise ScenarioCaptureError("capture seed/pass is outside the plan")
    run_id = runtime["capture_run_id"]
    nonce = runtime["capture_nonce"]
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise ScenarioCaptureError("capture run ID is unsafe")
    if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        raise ScenarioCaptureError("capture nonce is malformed")

    expected_cases = groups[seed] if production else groups[seed][:2]
    expected_count = len(expected_cases)
    expected_profile = "production32" if production else "smoke2"
    if (
        runtime["capture_profile"] != expected_profile
        or type(runtime["release_eligible"]) is not bool
        or runtime["release_eligible"] is not production
        or runtime["expected_env_count"] != expected_count
    ):
        raise ScenarioCaptureError("capture profile/release eligibility mismatch")
    if production and expected_count != EXPECTED_ENVS_PER_SEED:
        raise ScenarioCaptureError("release-eligible capture must contain exactly 32 envs")
    if not production and expected_count != 2:
        raise ScenarioCaptureError("smoke capture must contain exactly two envs")

    raw_cases = runtime["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(expected_cases):
        raise ScenarioCaptureError("runtime capture case coverage is incomplete")
    normalized_cases: list[dict[str, Any]] = []
    for env_index, (raw, planned) in enumerate(zip(raw_cases, expected_cases, strict=True)):
        case = _exact(raw, RUNTIME_CASE_KEYS, "runtime capture case")
        if {key: case[key] for key in PLANNED_CASE_KEYS} != planned:
            raise ScenarioCaptureError("runtime capture assignment differs from plan")
        condition = _sha(case["first_condition_sha256"], "first condition SHA-256")
        _sha(case["first_generated_sha256"], "first generated SHA-256")
        if case["scenario_id"] != _scenario_id(seed, env_index, condition):
            raise ScenarioCaptureError("runtime scenario ID is not condition-derived")
        normalized_cases.append(case)

    execution = runtime["execution"]
    expected_execution_keys = set(EXECUTION_DOCUMENT) | {
        "generated_frames",
        "renderer_requests_delta",
    }
    execution = _exact(execution, expected_execution_keys, "capture execution")
    if any(execution[key] != value for key, value in EXECUTION_DOCUMENT.items()):
        raise ScenarioCaptureError("capture execution is not non-stepping generated eval")
    expected_count = len(expected_cases)
    if (
        execution["generated_frames"] != expected_count
        or execution["renderer_requests_delta"] != expected_count
    ):
        raise ScenarioCaptureError("capture did not render exactly one frame per case")

    config = _exact(runtime["runtime"], RUNTIME_CONFIG_KEYS, "capture runtime config")
    expected_offset = expected_cases[0]["global_case_index"]
    expected_config = {
        "num_envs": expected_count,
        "generation_seed": seed,
        "motion_split_role": "eval",
        "motion_split_offset": expected_offset,
        "motion_split_limit": expected_count,
        "policy_views": list(POLICY_VIEWS),
        "view_layout": "vertical",
        "reference_lag": 256,
        "raytraced_mix_prob": 0.0,
        "anchor_source": "simulator_rgb",
    }
    if config != expected_config:
        raise ScenarioCaptureError("capture runtime configuration differs from contract")

    streams = runtime["renderer_streams"]
    if not isinstance(streams, list) or len(streams) != expected_count:
        raise ScenarioCaptureError("renderer stream coverage is incomplete")
    for env_index, raw in enumerate(streams):
        stream = _exact(raw, STREAM_KEYS, "renderer stream")
        if stream["env_index"] != env_index:
            raise ScenarioCaptureError("renderer stream env ordering is ambiguous")
        for key in ("endpoint_file", "server_id", "server_incarnation_id"):
            if not isinstance(stream[key], str) or not stream[key]:
                raise ScenarioCaptureError(f"renderer stream {key} is missing")

    provenance = _exact(runtime["provenance"], PROVENANCE_KEYS, "capture provenance")
    for key, value in provenance.items():
        if key in {"fast_roborender_revision", "gr00t_revision"}:
            if not isinstance(value, str) or REVISION_RE.fullmatch(value) is None:
                raise ScenarioCaptureError(f"capture provenance {key} is malformed")
        else:
            _sha(value, f"capture provenance {key}")
    if provenance["heldout_plan_sha256"] != plan_sha256:
        raise ScenarioCaptureError("capture provenance plan SHA mismatch")
    source = plan.get("source", {})
    if provenance["motion_split_manifest_sha256"] != source.get(
        "nested_split_manifest_sha256"
    ):
        raise ScenarioCaptureError("capture provenance split SHA mismatch")
    return runtime


def _selected_renderer_rows(snapshot: Any) -> list[dict[str, Any]]:
    if not isinstance(snapshot, Mapping):
        raise ScenarioCaptureError("renderer snapshot must be an object")
    rows = snapshot.get("selected")
    if not isinstance(rows, list) or not rows:
        raise ScenarioCaptureError("renderer snapshot has no selected fleet")
    normalized: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("file"), str):
            raise ScenarioCaptureError("renderer snapshot row is malformed")
        if raw["file"] in normalized:
            raise ScenarioCaptureError("renderer endpoint is duplicated")
        normalized[raw["file"]] = dict(raw)
    return [normalized[key] for key in sorted(normalized)]


def renderer_delta(
    before: Any,
    after: Any,
    *,
    expected_runtime_sha256: str,
) -> tuple[int, int]:
    old_rows = _selected_renderer_rows(before)
    new_rows = _selected_renderer_rows(after)
    if [row["file"] for row in old_rows] != [row["file"] for row in new_rows]:
        raise ScenarioCaptureError("renderer fleet changed during capture")
    served = 0
    server_ids: set[str] = set()
    for old, new in zip(old_rows, new_rows, strict=True):
        old_response = old.get("response")
        new_response = new.get("response")
        if not isinstance(old_response, Mapping) or not isinstance(new_response, Mapping):
            raise ScenarioCaptureError("renderer response is malformed")
        identity = old_response.get("runtime_identity")
        if not isinstance(identity, Mapping) or identity != new_response.get("runtime_identity"):
            raise ScenarioCaptureError("renderer runtime identity changed during capture")
        if canonical_sha256(identity) != expected_runtime_sha256:
            raise ScenarioCaptureError("renderer runtime identity differs from contract")
        server_id = old_response.get("server_id")
        if not isinstance(server_id, str) or server_id != new_response.get("server_id"):
            raise ScenarioCaptureError("renderer server identity changed during capture")
        server_ids.add(server_id)
        old_metrics = old_response.get("metrics")
        new_metrics = new_response.get("metrics")
        if not isinstance(old_metrics, Mapping) or not isinstance(new_metrics, Mapping):
            raise ScenarioCaptureError("renderer counters are missing")
        old_requests = _integer(old_metrics.get("requests"), "before renderer requests")
        new_requests = _integer(new_metrics.get("requests"), "after renderer requests")
        for name in ("errors", "replays"):
            if _integer(old_metrics.get(name), f"before renderer {name}") != 0:
                raise ScenarioCaptureError(f"renderer has pre-existing {name}")
            if _integer(new_metrics.get(name), f"after renderer {name}") != 0:
                raise ScenarioCaptureError(f"renderer has capture {name}")
        if new_requests < old_requests:
            raise ScenarioCaptureError("renderer request counter moved backwards")
        served += new_requests - old_requests
    if len(server_ids) != len(old_rows):
        raise ScenarioCaptureError("renderer fleet server identities are not unique")
    return served, len(old_rows)


def build_pass_attestation(
    *,
    plan_path: str | os.PathLike[str],
    runtime_path: str | os.PathLike[str],
    renderer_before_path: str | os.PathLike[str],
    renderer_after_path: str | os.PathLike[str],
    production: bool = True,
) -> dict[str, Any]:
    plan, plan_payload = load_strict_json(plan_path, "held-out capture plan")
    runtime, _runtime_payload = load_strict_json(runtime_path, "capture runtime artifact")
    before, _before_payload = load_strict_json(renderer_before_path, "renderer before snapshot")
    after, _after_payload = load_strict_json(renderer_after_path, "renderer after snapshot")
    plan_sha = _sha_bytes(plan_payload)
    runtime = validate_runtime_document(
        runtime, plan=plan, plan_sha256=plan_sha, production=production
    )
    served, renderer_count = renderer_delta(
        before,
        after,
        expected_runtime_sha256=runtime["provenance"]["renderer_runtime_sha256"],
    )
    if served != len(runtime["cases"]):
        raise ScenarioCaptureError("server request delta differs from captured case count")
    return {
        "schema": PASS_SCHEMA,
        # Hash the normalized embedded documents.  This makes a pass artifact
        # self-verifying even when the producer used pretty-printed JSON.
        "runtime_artifact_sha256": _sha_bytes(canonical_bytes(runtime) + b"\n"),
        "renderer_before_sha256": _sha_bytes(canonical_bytes(before) + b"\n"),
        "renderer_after_sha256": _sha_bytes(canonical_bytes(after) + b"\n"),
        "served_requests": served,
        "renderer_runtime_sha256": runtime["provenance"]["renderer_runtime_sha256"],
        "expected_renderer_count": renderer_count,
        "runtime": runtime,
        "renderer_before": before,
        "renderer_after": after,
    }


def validate_pass_document(
    document: Any,
    *,
    plan: Any,
    plan_sha256: str,
    production: bool = True,
) -> dict[str, Any]:
    capture = _exact(document, PASS_KEYS, "scenario capture pass")
    if capture["schema"] != PASS_SCHEMA:
        raise ScenarioCaptureError("scenario capture pass schema mismatch")
    runtime = validate_runtime_document(
        capture["runtime"], plan=plan, plan_sha256=plan_sha256, production=production
    )
    if capture["runtime_artifact_sha256"] != _sha_bytes(
        canonical_bytes(runtime) + b"\n"
    ):
        raise ScenarioCaptureError("embedded runtime artifact byte hash mismatch")
    for key, document_key in (
        ("renderer_before_sha256", "renderer_before"),
        ("renderer_after_sha256", "renderer_after"),
    ):
        if capture[key] != _sha_bytes(canonical_bytes(capture[document_key]) + b"\n"):
            raise ScenarioCaptureError(f"embedded {document_key} byte hash mismatch")
    served, count = renderer_delta(
        capture["renderer_before"],
        capture["renderer_after"],
        expected_runtime_sha256=runtime["provenance"]["renderer_runtime_sha256"],
    )
    if (
        served != len(runtime["cases"])
        or capture["served_requests"] != served
        or capture["expected_renderer_count"] != count
        or capture["renderer_runtime_sha256"]
        != runtime["provenance"]["renderer_runtime_sha256"]
    ):
        raise ScenarioCaptureError("capture pass renderer accounting mismatch")
    return capture


def build_campaign_attestation(
    *,
    plan_path: str | os.PathLike[str],
    pass_paths: Sequence[str | os.PathLike[str]],
    require_generated_repeat: bool = True,
    production: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not production:
        raise ScenarioCaptureError(
            "two-environment smoke captures can never produce a release attestation"
        )
    plan, plan_payload = load_strict_json(plan_path, "held-out capture plan")
    plan_sha = _sha_bytes(plan_payload)
    groups = validate_plan_document(plan, plan_sha256=plan_sha, production=production)
    expected_pass_count = len(groups) * 2
    if len(pass_paths) != expected_pass_count:
        raise ScenarioCaptureError(
            f"capture attestation requires exactly {expected_pass_count} pass artifacts"
        )
    captures: dict[tuple[int, int], tuple[dict[str, Any], str]] = {}
    for raw_path in pass_paths:
        document, payload = load_strict_json(raw_path, "scenario capture pass artifact")
        capture = validate_pass_document(
            document, plan=plan, plan_sha256=plan_sha, production=production
        )
        runtime = capture["runtime"]
        identity = (runtime["seed"], runtime["pass_index"])
        if identity in captures:
            raise ScenarioCaptureError("duplicate capture seed/pass artifact")
        captures[identity] = (capture, _sha_bytes(payload))
    expected_identities = {
        (seed, pass_index) for seed in groups for pass_index in EXPECTED_PASSES
    }
    if set(captures) != expected_identities:
        raise ScenarioCaptureError("capture seed/pass coverage is incomplete")

    run_ids: set[str] = set()
    nonces: set[str] = set()
    canonical_provenance: dict[str, Any] | None = None
    evidence_cases: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    pass_inventory: list[dict[str, Any]] = []
    for seed in sorted(groups):
        first, first_hash = captures[(seed, 1)]
        second, second_hash = captures[(seed, 2)]
        first_runtime = first["runtime"]
        second_runtime = second["runtime"]
        if first_runtime["capture_run_id"] == second_runtime["capture_run_id"]:
            raise ScenarioCaptureError("repeat capture reused its Image3F run namespace")
        if first_runtime["capture_nonce"] == second_runtime["capture_nonce"]:
            raise ScenarioCaptureError("repeat capture reused its freshness nonce")
        for runtime in (first_runtime, second_runtime):
            if runtime["capture_run_id"] in run_ids or runtime["capture_nonce"] in nonces:
                raise ScenarioCaptureError("capture run identity is reused across seed blocks")
            run_ids.add(runtime["capture_run_id"])
            nonces.add(runtime["capture_nonce"])
            if canonical_provenance is None:
                canonical_provenance = dict(runtime["provenance"])
            elif runtime["provenance"] != canonical_provenance:
                raise ScenarioCaptureError("capture pass provenance is not campaign-identical")
        for case_a, case_b in zip(
            first_runtime["cases"], second_runtime["cases"], strict=True
        ):
            repeat_keys = {
                "first_condition_sha256",
                "scenario_id",
            }
            if require_generated_repeat:
                repeat_keys.add("first_generated_sha256")
            if any(case_a[key] != case_b[key] for key in repeat_keys):
                raise ScenarioCaptureError(
                    "fresh capture passes are not identity-repeatable"
                )
            evidence_cases.append(
                {
                    **{key: case_a[key] for key in PLANNED_CASE_KEYS},
                    "first_condition_sha256": case_a["first_condition_sha256"],
                }
            )
            scenarios.append(
                {
                    "seed": case_a["seed"],
                    "env_index": case_a["env_index"],
                    "scenario_id": case_a["scenario_id"],
                }
            )
        for capture, payload_hash in ((first, first_hash), (second, second_hash)):
            runtime = capture["runtime"]
            pass_inventory.append(
                {
                    "seed": runtime["seed"],
                    "pass_index": runtime["pass_index"],
                    "capture_run_id": runtime["capture_run_id"],
                    "capture_nonce": runtime["capture_nonce"],
                    "pass_artifact_sha256": payload_hash,
                    "runtime_artifact_sha256": capture["runtime_artifact_sha256"],
                    "renderer_before_sha256": capture["renderer_before_sha256"],
                    "renderer_after_sha256": capture["renderer_after_sha256"],
                    "served_requests": capture["served_requests"],
                }
            )
    if canonical_provenance is None:
        raise ScenarioCaptureError("capture campaign has no provenance")
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "plan_sha256": plan_sha,
        "cases": evidence_cases,
    }
    evidence_payload = canonical_bytes(evidence) + b"\n"
    scenario_payload = canonical_bytes(scenarios) + b"\n"
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS",
        "release_eligible": True,
        "plan_sha256": plan_sha,
        "motion_split_manifest_sha256": canonical_provenance[
            "motion_split_manifest_sha256"
        ],
        "seed_count": len(groups),
        "envs_per_seed": len(next(iter(groups.values()))),
        "case_count": len(evidence_cases),
        "passes_per_seed": 2,
        "pass_artifacts": pass_inventory,
        "repeatability": {
            "first_condition_sha256": True,
            "scenario_id": True,
            "first_generated_sha256": bool(require_generated_repeat),
        },
        "renderer_accounting": {
            "served_requests_per_pass": len(next(iter(groups.values()))),
            "errors": 0,
            "replays": 0,
        },
        "provenance": canonical_provenance,
        "capture_evidence_sha256": _sha_bytes(evidence_payload),
        "scenario_contract_sha256": _sha_bytes(scenario_payload),
    }
    return attestation, evidence, scenarios


def verify_campaign_outputs(
    *,
    plan_path: str | os.PathLike[str],
    pass_paths: Sequence[str | os.PathLike[str]],
    attestation_path: str | os.PathLike[str],
    evidence_path: str | os.PathLike[str],
    scenario_path: str | os.PathLike[str],
    require_generated_repeat: bool = True,
    production: bool = True,
) -> dict[str, Any]:
    expected_attestation, expected_evidence, expected_scenarios = (
        build_campaign_attestation(
            plan_path=plan_path,
            pass_paths=pass_paths,
            require_generated_repeat=require_generated_repeat,
            production=production,
        )
    )
    attestation, attestation_payload = load_strict_json(
        attestation_path, "scenario capture campaign attestation"
    )
    evidence, evidence_payload = load_strict_json(
        evidence_path, "scenario capture evidence"
    )
    scenarios, scenario_payload = load_strict_json(
        scenario_path, "scenario contract"
    )
    if attestation != expected_attestation:
        raise ScenarioCaptureError("campaign attestation differs from live pass evidence")
    if evidence != expected_evidence:
        raise ScenarioCaptureError("capture evidence differs from live pass evidence")
    if scenarios != expected_scenarios:
        raise ScenarioCaptureError("scenario contract differs from live pass evidence")
    if _sha_bytes(evidence_payload) != attestation["capture_evidence_sha256"]:
        raise ScenarioCaptureError("capture evidence byte hash mismatch")
    if _sha_bytes(scenario_payload) != attestation["scenario_contract_sha256"]:
        raise ScenarioCaptureError("scenario contract byte hash mismatch")
    return {
        "status": "PASS",
        "release_eligible": True,
        "attestation_sha256": _sha_bytes(attestation_payload),
        "capture_evidence_sha256": _sha_bytes(evidence_payload),
        "scenario_contract_sha256": _sha_bytes(scenario_payload),
        "case_count": attestation["case_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pass_command = commands.add_parser("attest-pass")
    pass_command.add_argument("--plan", required=True)
    pass_command.add_argument("--runtime", required=True)
    pass_command.add_argument("--renderer-before", required=True)
    pass_command.add_argument("--renderer-after", required=True)
    pass_command.add_argument("--output", required=True)
    pass_command.add_argument("--smoke2", action="store_true")
    verify_pass = commands.add_parser("verify-pass")
    verify_pass.add_argument("--plan", required=True)
    verify_pass.add_argument("--pass-artifact", required=True)
    verify_pass.add_argument("--smoke2", action="store_true")
    campaign = commands.add_parser("attest-campaign")
    campaign.add_argument("--plan", required=True)
    campaign.add_argument("--pass-artifact", action="append", required=True)
    campaign.add_argument("--attestation-output", required=True)
    campaign.add_argument("--evidence-output", required=True)
    campaign.add_argument("--scenario-output", required=True)
    campaign.add_argument("--allow-nondeterministic-generated", action="store_true")
    verify = commands.add_parser("verify-campaign")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--pass-artifact", action="append", required=True)
    verify.add_argument("--attestation", required=True)
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--scenarios", required=True)
    verify.add_argument("--allow-nondeterministic-generated", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "attest-pass":
            document = build_pass_attestation(
                plan_path=args.plan,
                runtime_path=args.runtime,
                renderer_before_path=args.renderer_before,
                renderer_after_path=args.renderer_after,
                production=not args.smoke2,
            )
            _publish_immutable(args.output, document)
            result = {
                "status": "PASS",
                "pass_artifact_sha256": _sha_bytes(
                    canonical_bytes(document) + b"\n"
                ),
                "served_requests": document["served_requests"],
            }
        elif args.command == "verify-pass":
            plan, plan_payload = load_strict_json(args.plan, "held-out capture plan")
            document, payload = load_strict_json(
                args.pass_artifact, "scenario capture pass artifact"
            )
            normalized = validate_pass_document(
                document,
                plan=plan,
                plan_sha256=_sha_bytes(plan_payload),
                production=not args.smoke2,
            )
            result = {
                "status": "PASS",
                "pass_artifact_sha256": _sha_bytes(payload),
                "served_requests": normalized["served_requests"],
            }
        elif args.command == "attest-campaign":
            attestation, evidence, scenarios = build_campaign_attestation(
                plan_path=args.plan,
                pass_paths=args.pass_artifact,
                require_generated_repeat=not args.allow_nondeterministic_generated,
            )
            outputs = {
                Path(args.attestation_output).absolute(),
                Path(args.evidence_output).absolute(),
                Path(args.scenario_output).absolute(),
            }
            if len(outputs) != 3:
                raise ScenarioCaptureError("campaign output paths must be distinct")
            _publish_immutable(args.attestation_output, attestation)
            _publish_immutable(args.evidence_output, evidence)
            _publish_immutable(args.scenario_output, scenarios)
            result = verify_campaign_outputs(
                plan_path=args.plan,
                pass_paths=args.pass_artifact,
                attestation_path=args.attestation_output,
                evidence_path=args.evidence_output,
                scenario_path=args.scenario_output,
                require_generated_repeat=not args.allow_nondeterministic_generated,
            )
        else:
            result = verify_campaign_outputs(
                plan_path=args.plan,
                pass_paths=args.pass_artifact,
                attestation_path=args.attestation,
                evidence_path=args.evidence,
                scenario_path=args.scenarios,
                require_generated_repeat=not args.allow_nondeterministic_generated,
            )
    except ScenarioCaptureError as exc:
        print(f"scenario capture attestation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
