"""Read-only health projection for the reserved Lepton H200 learner node.

This module deliberately exposes no mutation methods.  It validates the same
sealed backend contract used by the learner bridge, reads the canonical
control deployment, and lists only active Jobs that claim the exact reserved
allocation.  Its JSON projection is safe for the workstation-side production
monitor: no environment variables, secret values, container environment, or
raw provider objects are serialized.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from round_dagger.lepton_job import (
    BACKEND,
    CONTROL_DEPLOYMENT,
    EXCLUSIVE_LIFECYCLE_SCHEMA,
    NODE_GROUP,
    RESERVATION_ID,
    RESOURCE_SHAPE,
    LeptonBackendContract,
    LeptonJobError,
    LeptonRunRequest,
    load_contract,
    load_request,
    validate_control_deployment,
    validate_job_identity,
)


HEALTH_SCHEMA = "fast_roborender.round_dagger.lepton_live_health.v1"
ACTIVE_JOB_STATES = frozenset(
    {"Starting", "Running", "Queueing", "Awaiting", "PendingRetry", "Restarting"}
)
_JOB_NAME_RE = re.compile(r"rrd-h200-([0-9a-f]{24})-a([1-3])")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class LeptonMonitorError(RuntimeError):
    """The provider response could not prove the reserved allocation healthy."""


def _member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _state(value: Any) -> str | None:
    raw = _member(_member(value, "status"), "state")
    observed = getattr(raw, "value", raw)
    return observed if isinstance(observed, str) else None


def _labels(value: Any) -> dict[str, str] | None:
    raw = _member(_member(value, "metadata"), "labels")
    if not isinstance(raw, Mapping):
        return None
    return {str(key): str(item) for key, item in raw.items()}


def _job_projection(
    job: Any, *, contract: LeptonBackendContract,
    request: LeptonRunRequest | None = None,
) -> dict[str, Any]:
    """Return a sanitized exact-identity projection for one active Job."""

    metadata = _member(job, "metadata")
    spec = _member(job, "spec")
    name = _member(metadata, "name")
    state = _state(job)
    labels = _labels(job)
    if not isinstance(name, str) or state not in ACTIVE_JOB_STATES or labels is None:
        raise LeptonMonitorError("active Job metadata or state is malformed")
    match = _JOB_NAME_RE.fullmatch(name)
    if match is None:
        raise LeptonMonitorError("active Job name is not a RoboRender learner name")

    reservation = _member(spec, "reservation_config")
    affinity = _member(spec, "affinity")
    if (
        _member(reservation, "reservation_id") != RESERVATION_ID
        or _member(reservation, "allow_burst_to_other_reservations")
        not in {None, False}
        or list(_member(affinity, "allowed_dedicated_node_groups") or ())
        != [NODE_GROUP]
        or _member(spec, "resource_shape") != RESOURCE_SHAPE
    ):
        raise LeptonMonitorError("active Job allocation differs from the reserved node")

    required = {
        "roborender-dagger-backend",
        "roborender-dagger-identity-hi",
        "roborender-dagger-identity-lo",
        "roborender-dagger-contract-hi",
        "roborender-dagger-contract-lo",
        "roborender-dagger-request-hi",
        "roborender-dagger-request-lo",
        "roborender-dagger-attempt",
    }
    if set(labels) != required or labels["roborender-dagger-backend"] != BACKEND:
        raise LeptonMonitorError("active Job labels differ from the learner contract")
    identity = (
        labels["roborender-dagger-identity-hi"]
        + labels["roborender-dagger-identity-lo"]
    )
    request_digest = (
        labels["roborender-dagger-request-hi"]
        + labels["roborender-dagger-request-lo"]
    )
    contract_sha = (
        labels["roborender-dagger-contract-hi"]
        + labels["roborender-dagger-contract-lo"]
    )
    attempt = labels["roborender-dagger-attempt"]
    if (
        _SHA_RE.fullmatch(identity) is None
        or _SHA_RE.fullmatch(request_digest) is None
        or contract_sha != contract.sha256
        or match.group(1) != identity[:24]
        or attempt != match.group(2)
    ):
        raise LeptonMonitorError("active Job digest binding is malformed")
    if request is not None:
        if identity != request.identity_sha256 or request_digest != request.sha256:
            raise LeptonMonitorError("active Job differs from the current request")
        try:
            validate_job_identity(
                job,
                contract=contract,
                request=request,
                attempt=int(attempt),
            )
        except LeptonJobError as exc:
            raise LeptonMonitorError(
                "active Job differs from the current request"
            ) from exc
    return {
        "name": name,
        "state": state,
        "semantic_attempt": int(attempt),
        "identity_sha256": identity,
        "request_sha256": request_digest,
        "valid": True,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise LeptonMonitorError("lifecycle state is unsafe")
        pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=list)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeptonMonitorError("lifecycle state is unreadable") from exc
    if not isinstance(pairs, list):
        raise LeptonMonitorError("lifecycle state is malformed")
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise LeptonMonitorError("lifecycle state is malformed")
        result[key] = value
    return result


def _current_request(
    *,
    handoff_root: Path,
    campaign_id: str,
    macro_round: int,
    contract: LeptonBackendContract,
) -> tuple[LeptonRunRequest | None, dict[str, Any] | None]:
    round_root = (
        handoff_root
        / BACKEND
        / campaign_id
        / f"macro_round_{macro_round:06d}"
    )
    if not round_root.exists():
        return None, None
    if round_root.is_symlink() or not round_root.is_dir():
        raise LeptonMonitorError("current handoff round is unsafe")
    requests = sorted(round_root.glob("identity_*/lepton_learner_request.json"))
    if len(requests) > 1:
        raise LeptonMonitorError("multiple current learner requests exist")
    if not requests:
        return None, None
    request_path = requests[0]
    try:
        request = load_request(request_path, contract=contract, immutable=True)
    except LeptonJobError as exc:
        raise LeptonMonitorError("current learner request is invalid") from exc
    if request.campaign_id != campaign_id or request.macro_round != macro_round:
        raise LeptonMonitorError("current learner request scope is invalid")
    expected_parent = round_root / f"identity_{request.identity_sha256}"
    if request_path.parent != expected_parent:
        raise LeptonMonitorError("current learner request path is noncanonical")
    lifecycle_path = request_path.parent / "exclusive_lifecycle.json"
    if not lifecycle_path.exists():
        return request, None
    lifecycle = _strict_json(lifecycle_path)
    expected_keys = {
        "schema",
        "backend",
        "control_deployment",
        "contract_sha256",
        "request_sha256",
        "learner_identity_sha256",
        "phase",
        "semantic_attempt",
    }
    if set(lifecycle) != expected_keys:
        raise LeptonMonitorError("current lifecycle key set is invalid")
    expected_values = {
        "schema": EXCLUSIVE_LIFECYCLE_SCHEMA,
        "backend": BACKEND,
        "control_deployment": CONTROL_DEPLOYMENT,
        "contract_sha256": contract.sha256,
        "request_sha256": request.sha256,
        "learner_identity_sha256": request.identity_sha256,
    }
    if any(lifecycle.get(key) != value for key, value in expected_values.items()):
        raise LeptonMonitorError("current lifecycle identity is invalid")
    if lifecycle.get("phase") not in {
        "control_ready",
        "control_stop_requested",
        "control_stopped",
        "control_restore_requested",
    } or not isinstance(lifecycle.get("semantic_attempt"), int):
        raise LeptonMonitorError("current lifecycle phase is invalid")
    return request, lifecycle


def collect_health(
    *,
    backend_contract: Path,
    backend_contract_sha256: str,
    handoff_root: Path,
    campaign_id: str,
    macro_round: int,
    api_client: Any | None = None,
) -> dict[str, Any]:
    """Read and validate the exact reserved Lepton allocation."""

    contract = load_contract(backend_contract, immutable=True)
    if contract.sha256 != backend_contract_sha256:
        raise LeptonMonitorError("Lepton backend contract SHA-256 mismatch")
    if api_client is None:  # pragma: no cover - exercised by live integration
        try:
            from leptonai.api.v2.client import APIClient
        except ImportError as exc:
            raise LeptonMonitorError("Lepton SDK is unavailable") from exc
        api_client = APIClient()

    handoff_root = Path(os.path.abspath(handoff_root))
    if _TOKEN_RE.fullmatch(campaign_id) is None or macro_round < 0:
        raise LeptonMonitorError("campaign monitor scope is malformed")
    if handoff_root.is_symlink() or not handoff_root.is_dir():
        raise LeptonMonitorError("Lepton handoff root is unsafe")
    request, lifecycle = _current_request(
        handoff_root=handoff_root,
        campaign_id=campaign_id,
        macro_round=macro_round,
        contract=contract,
    )

    try:
        workspace = api_client.get_workspace_id()
        expected_workspace = contract.document["cluster"]["workspace"]
        if workspace != expected_workspace:
            raise LeptonMonitorError("Lepton workspace differs from the contract")
        deployment = api_client.deployment.get(CONTROL_DEPLOYMENT)
        control_state = validate_control_deployment(deployment, contract=contract)
        jobs: Sequence[Any] = api_client.job.list_all(
            job_query_mode="alive_only",
            status=sorted(ACTIVE_JOB_STATES),
        )
    except (LeptonJobError, OSError) as exc:
        raise LeptonMonitorError("Lepton control plane could not be validated") from exc
    except Exception as exc:  # provider exception types vary between SDK releases
        raise LeptonMonitorError("Lepton control plane query failed") from exc

    active: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for job in jobs:
        spec = _member(job, "spec")
        reservation = _member(spec, "reservation_config")
        if _member(reservation, "reservation_id") != RESERVATION_ID:
            continue
        name = _member(_member(job, "metadata"), "name")
        state = _state(job)
        try:
            active.append(
                _job_projection(job, contract=contract, request=request)
            )
        except LeptonMonitorError:
            malformed.append(
                {
                    "name": (
                        name
                        if isinstance(name, str)
                        and _JOB_NAME_RE.fullmatch(name) is not None
                        else "<malformed>"
                    ),
                    "state": (
                        state if state in ACTIVE_JOB_STATES else "<malformed>"
                    ),
                    "valid": False,
                    "error": "identity_or_allocation_mismatch",
                }
            )

    reason_codes: list[str] = []
    if malformed:
        reason_codes.append("reserved_job_identity_invalid")
    if len(active) > 1:
        reason_codes.append("multiple_active_reserved_jobs")
    if active and request is None:
        reason_codes.append("orphan_active_reserved_job")

    phase = lifecycle.get("phase") if lifecycle is not None else None
    if phase in {None, "control_ready"}:
        if control_state != "Ready":
            reason_codes.append("control_not_ready_without_owned_transition")
        if active:
            reason_codes.append("active_job_while_control_ready")
    elif phase == "control_stop_requested":
        if control_state not in {"Ready", "Stopping", "Stopped"} or active:
            reason_codes.append("stop_transition_inconsistent")
    elif phase == "control_stopped":
        if control_state != "Stopped" or len(active) > 1:
            reason_codes.append("stopped_transition_inconsistent")
    elif phase == "control_restore_requested":
        if control_state not in {
            "Stopped", "Starting", "Updating", "Scaling", "Not Ready", "Ready"
        } or active:
            reason_codes.append("restore_transition_inconsistent")

    allocation_consistent = not reason_codes
    request_projection = None
    if request is not None:
        request_projection = {
            "campaign_id": request.campaign_id,
            "macro_round": request.macro_round,
            "request_sha256": request.sha256,
            "learner_identity_sha256": request.identity_sha256,
            "lifecycle_phase": phase,
            "semantic_attempt": (
                lifecycle.get("semantic_attempt") if lifecycle is not None else None
            ),
        }
    return {
        "schema": HEALTH_SCHEMA,
        "backend": BACKEND,
        "reachable": True,
        "healthy": allocation_consistent,
        "backend_contract_sha256": contract.sha256,
        "workspace": expected_workspace,
        "reservation": RESERVATION_ID,
        "node_group": NODE_GROUP,
        "resource_shape": RESOURCE_SHAPE,
        "control_deployment": CONTROL_DEPLOYMENT,
        "control_state": control_state,
        "active_job_count": len(active),
        "active_jobs": sorted(active, key=lambda item: item["name"]),
        "malformed_active_jobs": sorted(
            malformed, key=lambda item: (item["name"], item["state"])
        ),
        "current_request": request_projection,
        "reason_codes": sorted(set(reason_codes)),
        "read_only": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-contract", type=Path, required=True)
    parser.add_argument("--backend-contract-sha256", required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--macro-round", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = collect_health(
            backend_contract=arguments.backend_contract,
            backend_contract_sha256=arguments.backend_contract_sha256,
            handoff_root=arguments.handoff_root,
            campaign_id=arguments.campaign_id,
            macro_round=arguments.macro_round,
        )
    except (LeptonMonitorError, LeptonJobError, OSError, ValueError) as exc:
        print(f"Lepton health probe failed: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ACTIVE_JOB_STATES",
    "HEALTH_SCHEMA",
    "LeptonMonitorError",
    "collect_health",
    "main",
]
