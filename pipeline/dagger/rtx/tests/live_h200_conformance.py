#!/usr/bin/env python3
"""One-step fake-backed conformance probe for a real H200 port-18620 service.

This never imports or initializes Isaac. The only transition is the in-memory
`DeterministicFakeRaven.step_once` callback.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.boundary import (
    H200_COMMIT_PATH,
    H200_PREPARE_PATH,
    build_h200_commit,
    build_h200_prepare,
    build_owner_commit,
    validate_h200_commit_ack,
    validate_h200_selection,
)
from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    decode_arrays,
    payload_digest,
    sha256_bytes,
)
from dagger_rtx.fake import DeterministicFakeRaven, fake_static_provenance
from dagger_rtx.http_transport import BoundedBearerJsonTransport
from dagger_rtx.owner import IsaacTransitionOwner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18620")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    args.scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    profile = RavenShapeProfile()
    key = StepKey("conformance", 1, "rtx-fake", "window-000", 0, "episode-000", 0)
    transport = BoundedBearerJsonTransport(args.base_url, args.token_file)
    with tempfile.TemporaryDirectory(prefix="h200-conformance-", dir=args.scratch_root) as temp:
        work = Path(temp)
        backend = DeterministicFakeRaven(profile)
        owner = IsaacTransitionOwner(
            backend,
            work / "owner",
            work / "staging",
            shape_profile=profile,
            static_provenance=fake_static_provenance(),
        )
        owner.begin_window(
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "begin-conformance-window",
                "step_key": key.as_dict(),
            }
        )
        local_prepare = owner.prepare_step(
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "prepare-conformance-step",
                "step_key": key.as_dict(),
                "beta": ROUND_BETAS[1],
            }
        )
        prepare_request = build_h200_prepare(local_prepare, "h200-prepare-conformance")
        first_selection_raw = transport.post(
            H200_PREPARE_PATH, prepare_request, max_response_bytes=1 << 20
        )
        first_selection = validate_h200_selection(
            first_selection_raw,
            local_prepare,
            expected_idempotency_key=prepare_request["idempotency_key"],
        )
        first_selection.pop("_selected_action_array", None)
        retry_selection = transport.post(
            H200_PREPARE_PATH, prepare_request, max_response_bytes=1 << 20
        )
        if retry_selection != first_selection:
            raise RuntimeError("identical prepare retry did not return the exact persisted response")

        changed = copy.deepcopy(prepare_request)
        changed["metadata"]["prompt"] += " changed"
        changed_arrays = decode_arrays(changed["arrays"])
        changed["capture_digest"] = payload_digest(changed["metadata"], changed_arrays)
        request_digest = sha256_bytes(
            canonical_json(
                {
                    "step_key": key.as_dict(),
                    "request_id": local_prepare["request_id"],
                    "beta": ROUND_BETAS[1],
                }
            )
        )
        changed["prepare_receipt"] = sha256_bytes(
            canonical_json(
                {"request_digest": request_digest, "capture_digest": changed["capture_digest"]}
            )
        )
        conflict_status = None
        try:
            transport.post(H200_PREPARE_PATH, changed, max_response_bytes=1 << 20)
        except RuntimeError as exc:
            if "HTTP 409" not in str(exc):
                raise
            conflict_status = 409
        if conflict_status != 409:
            raise RuntimeError("changed prepare body reused an idempotency key without HTTP 409")

        local_commit = owner.commit_step(build_owner_commit(first_selection))
        commit_request = build_h200_commit(
            first_selection, local_commit, "h200-commit-conformance"
        )
        first_ack = transport.post(H200_COMMIT_PATH, commit_request, max_response_bytes=1 << 20)
        validate_h200_commit_ack(first_ack, commit_request)
        retry_ack = transport.post(H200_COMMIT_PATH, commit_request, max_response_bytes=1 << 20)
        validate_h200_commit_ack(retry_ack, commit_request)
        if retry_ack != first_ack:
            raise RuntimeError("identical commit retry did not return the exact persisted response")
        if backend.step_calls != 1:
            raise RuntimeError(f"fake backend step count {backend.step_calls} != 1")

        print(
            json.dumps(
                {
                    "schema": CONTRACT_VERSION,
                    "result": "PASS",
                    "no_isaac_import_or_initialization": True,
                    "fake_backend_step_calls": backend.step_calls,
                    "prepare_retry_exact": True,
                    "changed_prepare_same_key_status": conflict_status,
                    "commit_retry_exact": True,
                    "selection_receipt": first_selection["selection_receipt"],
                    "remote_commit_receipt": first_ack["remote_commit_receipt"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
