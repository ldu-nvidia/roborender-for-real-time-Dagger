#!/usr/bin/env python3
"""Plan or run one fixed four-step Lepton H200 exclusive-lifecycle smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.lepton_job import (  # noqa: E402
    ExclusiveNodeLifecycle,
    LeptonJobError,
    build_job_document,
    fetch_completion_pair,
    find_attempt_jobs,
    load_contract,
    require_finalized_supply_chain,
    submit_or_adopt_attempt,
    validate_job_identity,
    verify_control_preflight,
    verify_reserved_node_job_exclusivity,
    wait_for_terminal,
)
from round_dagger.lepton_smoke_job import (  # noqa: E402
    load_smoke_request,
    smoke_contract_view,
    validate_smoke_completion_marker,
)


DEFAULT_TRANSFER = ROOT / "scripts/round_dagger/lepton_web_terminal_transfer.py"


def _clients() -> tuple[Any, Any]:
    try:
        from leptonai.api.v2.client import APIClient
    except ImportError as exc:
        raise LeptonJobError("Lepton SDK is not installed") from exc
    client = APIClient()
    return client.job, client.deployment


def _load(arguments: argparse.Namespace):
    if arguments.command == "submit" and (
        not arguments.contract_sha256 or not arguments.request_sha256
    ):
        raise LeptonJobError(
            "live bounded smoke requires explicit contract and request SHA-256 values"
        )
    contract = load_contract(arguments.contract, immutable=False)
    if arguments.contract_sha256 and contract.sha256 != arguments.contract_sha256:
        raise LeptonJobError("backend contract SHA-256 mismatch")
    request = load_smoke_request(
        arguments.request,
        contract=contract,
        immutable=arguments.command == "submit",
    )
    if arguments.request_sha256 and request.sha256 != arguments.request_sha256:
        raise LeptonJobError("bounded-smoke request SHA-256 mismatch")
    return contract, request


def _plan(arguments: argparse.Namespace) -> int:
    contract, request = _load(arguments)
    document = build_job_document(
        smoke_contract_view(contract),
        request,
        request_path=arguments.remote_request,
        contract_path=arguments.remote_contract,
        attempt=1,
    )
    print(json.dumps(document, sort_keys=True, indent=2))
    return 0


def _submit(arguments: argparse.Namespace) -> int:
    if not arguments.apply or not arguments.bounded_smoke:
        raise LeptonJobError("live validation requires --apply --bounded-smoke")
    if not arguments.exclusive_node_lifecycle:
        raise LeptonJobError(
            "live validation on the sole H200 node requires "
            "--exclusive-node-lifecycle"
        )
    if not arguments.lifecycle_state or not arguments.control_exec_program:
        raise LeptonJobError(
            "exclusive-node lifecycle requires --lifecycle-state and "
            "--control-exec-program"
        )
    contract, request = _load(arguments)
    # Fail locally before constructing a provider client or reading cluster state.
    require_finalized_supply_chain(contract)
    job_contract = smoke_contract_view(contract)
    transfer = Path(arguments.transfer_program).expanduser().resolve()
    if transfer.is_symlink() or not transfer.is_file():
        raise LeptonJobError("completion transfer program is missing or unsafe")
    control_exec = Path(arguments.control_exec_program).expanduser()
    job_client, deployment_client = _clients()

    with ExclusiveNodeLifecycle(
        arguments.lifecycle_state,
        contract=job_contract,
        request=request,
        client=deployment_client,
        timeout_seconds=arguments.control_timeout_seconds,
    ) as lifecycle:

        def preflight() -> None:
            verify_reserved_node_job_exclusivity(
                job_client, contract=job_contract, request=request
            )
            verify_control_preflight(
                control_exec_program=control_exec,
                contract=job_contract,
                request=request,
                remote_contract=arguments.remote_contract,
                remote_request=arguments.remote_request,
            )

        try:
            observed = find_attempt_jobs(
                job_client, contract=job_contract, request=request
            )
            if set(observed) - {1}:
                raise LeptonJobError(
                    "bounded smoke permits no semantic attempt after attempt 1"
                )
            job = observed.get(1)
            adopted = job is not None
            state = (
                validate_job_identity(
                    job, contract=job_contract, request=request, attempt=1
                )
                if job is not None
                else ""
            )
            terminal = {
                "Completed",
                "Archived",
                "Failed",
                "Stopped",
                "Stopping",
                "Deleting",
                "Deleted",
            }
            if state not in terminal:
                lifecycle.ensure_stopped(1, preflight=preflight)
                if job is None:
                    observed = find_attempt_jobs(
                        job_client, contract=job_contract, request=request
                    )
                    if set(observed) - {1}:
                        raise LeptonJobError(
                            "bounded smoke acquired an unexpected retry attempt"
                        )
                    document = build_job_document(
                        job_contract,
                        request,
                        request_path=arguments.remote_request,
                        contract_path=arguments.remote_contract,
                        attempt=1,
                    )
                    job, adopted = submit_or_adopt_attempt(
                        job_client,
                        contract=job_contract,
                        request=request,
                        job_document=document,
                        attempt=1,
                        existing=observed,
                    )
                    state = validate_job_identity(
                        job, contract=job_contract, request=request, attempt=1
                    )
                if state not in terminal:
                    job, state = wait_for_terminal(
                        job_client,
                        job,
                        contract=job_contract,
                        request=request,
                        attempt=1,
                        timeout_seconds=arguments.wait_timeout_seconds,
                        heartbeat=lambda current: print(
                            json.dumps(
                                {
                                    "event": "lepton_bounded_smoke_heartbeat",
                                    "identity": request.identity_sha256,
                                    "state": current,
                                    "adopted": adopted,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        ),
                    )

            lifecycle.ensure_ready()
            if state not in {"Completed", "Archived"}:
                raise LeptonJobError(
                    f"bounded smoke reached terminal state {state}"
                )
            marker, checksum, owner = fetch_completion_pair(
                transfer_program=transfer,
                deployment=contract.document["orchestration"][
                    "control_deployment"
                ],
                remote_marker=request.paths["success_marker"],
            )
            try:
                evidence = validate_smoke_completion_marker(
                    marker, request=request
                )
                marker_sha256 = checksum.read_text(encoding="ascii").split()[0]
            finally:
                owner.cleanup()
            print(
                json.dumps(
                    {
                        "schema": (
                            "fast_roborender.round_dagger."
                            "lepton_bounded_smoke_bridge_result.v1"
                        ),
                        "status": "PASS",
                        "smoke_identity_sha256": request.identity_sha256,
                        "adopted": adopted,
                        "success_marker_sha256": marker_sha256,
                        "report_file_sha256": evidence["report_file_sha256"],
                        "wandb_readback_file_sha256": evidence[
                            "wandb_readback_file_sha256"
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        finally:
            lifecycle.ensure_ready()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "submit"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", required=True)
        command.add_argument("--contract-sha256")
        command.add_argument("--request", required=True)
        command.add_argument("--request-sha256")
        command.add_argument("--remote-contract", required=True)
        command.add_argument("--remote-request", required=True)
        if name == "submit":
            command.add_argument("--apply", action="store_true")
            command.add_argument("--bounded-smoke", action="store_true")
            command.add_argument("--exclusive-node-lifecycle", action="store_true")
            command.add_argument("--lifecycle-state")
            command.add_argument("--control-exec-program")
            command.add_argument("--control-timeout-seconds", type=int, default=3600)
            command.add_argument("--wait-timeout-seconds", type=int, default=3600)
            command.add_argument("--transfer-program", default=str(DEFAULT_TRANSFER))
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    return _plan(arguments) if arguments.command == "plan" else _submit(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeptonJobError as exc:
        print(f"Lepton bounded-smoke bridge failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
