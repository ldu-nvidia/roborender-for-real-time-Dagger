#!/usr/bin/env python3
"""Plan, submit, adopt, and synchronously verify one Lepton H200 learner Job.

Planning is the default-safe operation.  Live mutation requires the explicit
``submit --apply`` form.  The bridge never deletes jobs and never asks Lepton
or Kubernetes to retry them.  It owns at most three semantic attempts and
adopts an exact existing attempt after coordinator restart.
"""

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
    LeptonControlDeploymentAdapter,
    LeptonJobError,
    build_job_document,
    fetch_completion_pair,
    find_attempt_jobs,
    load_contract,
    load_request,
    require_finalized_supply_chain,
    submit_or_adopt_attempt,
    validate_completion_marker,
    validate_job_identity,
    verify_control_preflight,
    verify_reserved_node_job_exclusivity,
    wait_for_terminal,
)


DEFAULT_TRANSFER = ROOT / "scripts/round_dagger/lepton_web_terminal_transfer.py"


def _clients() -> tuple[Any, Any]:
    try:
        from leptonai.api.v2.client import APIClient
    except ImportError as exc:
        raise LeptonJobError("Lepton SDK is not installed") from exc
    client = APIClient()
    return client.job, LeptonControlDeploymentAdapter(client.deployment)


def _load(args: argparse.Namespace):
    if args.command == "submit" and (
        not args.contract_sha256 or not args.request_sha256
    ):
        raise LeptonJobError(
            "live submission requires explicit contract and request SHA-256 values"
        )
    # Local checked-in config may remain writable for development.  Its exact
    # digest is mandatory for live submission; the remote copies are separately
    # required to be immutable by the in-job executor.
    contract = load_contract(args.contract, immutable=False)
    if args.contract_sha256 and contract.sha256 != args.contract_sha256:
        raise LeptonJobError("backend contract SHA-256 mismatch")
    request = load_request(
        args.request, contract=contract, immutable=args.command == "submit"
    )
    if args.request_sha256 and request.sha256 != args.request_sha256:
        raise LeptonJobError("run request SHA-256 mismatch")
    return contract, request


def _plan(args: argparse.Namespace) -> int:
    contract, request = _load(args)
    document = build_job_document(
        contract,
        request,
        request_path=args.remote_request,
        contract_path=args.remote_contract,
        attempt=args.attempt,
    )
    print(json.dumps(document, sort_keys=True, indent=2))
    return 0


def _submit(args: argparse.Namespace) -> int:
    if not args.apply:
        raise LeptonJobError("live submission requires --apply")
    if not args.exclusive_node_lifecycle:
        raise LeptonJobError(
            "live submission on the sole H200 node requires "
            "--exclusive-node-lifecycle"
        )
    if not args.lifecycle_state or not args.control_exec_program:
        raise LeptonJobError(
            "exclusive-node lifecycle requires --lifecycle-state and "
            "--control-exec-program"
        )
    contract, request = _load(args)
    require_finalized_supply_chain(contract)
    transfer = Path(args.transfer_program).expanduser().resolve()
    if transfer.is_symlink() or not transfer.is_file():
        raise LeptonJobError("completion transfer program is missing or unsafe")
    control_exec = Path(args.control_exec_program).expanduser()
    job_client, deployment_client = _clients()

    with ExclusiveNodeLifecycle(
        args.lifecycle_state,
        contract=contract,
        request=request,
        client=deployment_client,
        timeout_seconds=args.control_timeout_seconds,
    ) as lifecycle:
        def exclusive_preflight() -> None:
            verify_reserved_node_job_exclusivity(
                job_client, contract=contract, request=request
            )
            verify_control_preflight(
                control_exec_program=control_exec,
                contract=contract,
                request=request,
                remote_contract=args.remote_contract,
                remote_request=args.remote_request,
            )

        try:
            for attempt in range(1, contract.semantic_max_attempts + 1):
                observed = find_attempt_jobs(
                    job_client, contract=contract, request=request
                )
                existing = observed.get(attempt)
                adopted = existing is not None
                if existing is not None:
                    state = validate_job_identity(
                        existing,
                        contract=contract,
                        request=request,
                        attempt=attempt,
                    )
                    job = existing
                else:
                    state = ""
                    job = None

                if state not in {
                    "Completed",
                    "Archived",
                    "Failed",
                    "Stopped",
                    "Stopping",
                    "Deleting",
                    "Deleted",
                }:
                    lifecycle.ensure_stopped(
                        attempt,
                        preflight=exclusive_preflight,
                    )
                    if job is None:
                        # Never skip an unknown gap: every earlier attempt must
                        # be an exact terminal Job.  Re-read after stopping so
                        # create-by-name races are adopted rather than doubled.
                        observed = find_attempt_jobs(
                            job_client, contract=contract, request=request
                        )
                        for earlier in range(1, attempt):
                            prior = observed.get(earlier)
                            if prior is None:
                                raise LeptonJobError(
                                    "semantic attempt history contains a gap"
                                )
                            prior_state = validate_job_identity(
                                prior,
                                contract=contract,
                                request=request,
                                attempt=earlier,
                            )
                            if prior_state not in {
                                "Failed",
                                "Stopped",
                                "Stopping",
                                "Deleting",
                                "Deleted",
                                "Completed",
                                "Archived",
                            }:
                                raise LeptonJobError(
                                    "an earlier semantic attempt is not terminal"
                                )
                        document = build_job_document(
                            contract,
                            request,
                            request_path=args.remote_request,
                            contract_path=args.remote_contract,
                            attempt=attempt,
                        )
                        job, adopted = submit_or_adopt_attempt(
                            job_client,
                            contract=contract,
                            request=request,
                            job_document=document,
                            attempt=attempt,
                            existing=observed,
                        )
                        state = validate_job_identity(
                            job,
                            contract=contract,
                            request=request,
                            attempt=attempt,
                        )
                    if state not in {
                        "Completed",
                        "Archived",
                        "Failed",
                        "Stopped",
                        "Stopping",
                        "Deleting",
                        "Deleted",
                    }:
                        job, state = wait_for_terminal(
                            job_client,
                            job,
                            contract=contract,
                            request=request,
                            attempt=attempt,
                            timeout_seconds=args.wait_timeout_seconds,
                            heartbeat=lambda current: print(
                                json.dumps(
                                    {
                                        "event": "lepton_job_heartbeat",
                                        "identity": request.identity_sha256,
                                        "attempt": attempt,
                                        "state": current,
                                        "adopted": adopted,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            ),
                        )

                # The sole node is returned to its control deployment after
                # *every* terminal attempt.  Completion evidence is fetched
                # only after the control Pod is observably Ready again.
                lifecycle.ensure_ready()
                if state in {"Completed", "Archived"}:
                    try:
                        marker, checksum, owner = fetch_completion_pair(
                            transfer_program=transfer,
                            deployment=contract.document["orchestration"][
                                "control_deployment"
                            ],
                            remote_marker=request.paths["success_marker"],
                        )
                        try:
                            evidence = validate_completion_marker(
                                marker, request=request
                            )
                            marker_sha256 = checksum.read_text(
                                encoding="ascii"
                            ).split()[0]
                        finally:
                            owner.cleanup()
                    except LeptonJobError:
                        # A transport failure or completed Job without exact
                        # evidence consumes one semantic attempt.  A retry uses
                        # the same output/restart/W&B identity.
                        continue
                    print(
                        json.dumps(
                            {
                                "schema": "fast_roborender.round_dagger.lepton_bridge_result.v1",
                                "status": "PASS",
                                "learner_identity_sha256": request.identity_sha256,
                                "semantic_attempt": attempt,
                                "adopted": adopted,
                                "success_marker_sha256": marker_sha256,
                                "result_sha256": evidence["result_sha256"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    return 0
                # Terminal platform failure: the next bridge-owned semantic
                # attempt starts another stop/job/restore cycle.  Platform
                # retries remain zero and no Job is deleted or cancelled.
        finally:
            # This also repairs a crash after stop intent or restart request.
            # If a Job is still active, Lepton queues the restored Pod on the
            # reservation until that exact Job relinquishes the sole node.
            lifecycle.ensure_ready()

    raise LeptonJobError("all bridge-owned semantic attempts failed")


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
        if name == "plan":
            command.add_argument("--attempt", type=int, default=1)
        else:
            command.add_argument("--apply", action="store_true")
            command.add_argument(
                "--exclusive-node-lifecycle",
                action="store_true",
                help=(
                    "explicitly authorize the journaled stop/job/restore cycle "
                    "for the sole reserved H200 node"
                ),
            )
            command.add_argument(
                "--lifecycle-state",
                help="absolute local crash-recovery journal path",
            )
            command.add_argument(
                "--control-exec-program",
                help="authenticated fixed-command helper for control-Pod preflight",
            )
            command.add_argument(
                "--control-timeout-seconds", type=int, default=3600
            )
            command.add_argument(
                "--wait-timeout-seconds", type=int, default=172800
            )
            command.add_argument("--transfer-program", default=str(DEFAULT_TRANSFER))
    return parser


def main() -> int:
    args = _parser().parse_args()
    return _plan(args) if args.command == "plan" else _submit(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeptonJobError as exc:
        print(f"Lepton learner bridge failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
