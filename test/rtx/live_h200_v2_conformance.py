#!/usr/bin/env python3
"""Bounded fake-backed RTX/H200 v2 wire conformance; never initializes Isaac."""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.binary_transport_v2 import BoundedBearerBatchV2Transport
from dagger_rtx.boundary import build_owner_commit
from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    sha256_bytes,
)
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.http_transport import RemoteHttpError
from dagger_rtx.lean_boundary_v2 import (
    CONTRACT_VERSION_V2,
    H200_ANCHOR_BATCH_PATH_V2,
    H200_COMMIT_BATCH_PATH_V2,
    H200_PREPARE_BATCH_PATH_V2,
    anchor_records_from_prepares,
    build_anchor_batch,
    build_commit_batch,
    build_prepare_batch,
    validate_anchor_batch_response,
    validate_commit_batch_response,
    validate_prepare_batch_response,
)


def _token(path: Path) -> str:
    if path.stat().st_mode & 0o077:
        raise PermissionError("token file must be mode 0600 or stricter")
    value = path.read_text().strip()
    if not value or "\n" in value or "\r" in value:
        raise PermissionError("invalid token file")
    return value


def _ready(base_url: str, token_file: Path) -> dict:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    try:
        connection.request(
            "GET",
            "/ready",
            headers={"Authorization": f"Bearer {_token(token_file)}"},
        )
        response = connection.getresponse()
        raw = response.read(1 << 20)
        if response.status != 200:
            raise RuntimeError(f"/ready returned {response.status}: {raw[:256]!r}")
        value = json.loads(raw)
    finally:
        connection.close()
    if not value.get("ok"):
        raise RuntimeError("H200 v2 boundary is not ready")
    versions = value.get("supported_contract_versions", [])
    if CONTRACT_VERSION_V2 not in versions:
        raise RuntimeError("H200 /ready does not advertise the v2 contract")
    return value


def _keys() -> list[StepKey]:
    return [
        StepKey(
            run_id="v2-cross-host-conformance",
            round_id=1,
            actor_id=f"fake-lane-{lane:03d}",
            rollout_window_id=f"window-{lane:03d}",
            rollout_step_index=0,
            episode_id=f"episode-{lane:03d}",
            episode_step_index=0,
        )
        for lane in range(2)
    ]


def run(base_url: str, token_file: Path, output: Path) -> dict:
    ready = _ready(base_url, token_file)
    profile = RavenShapeProfile()
    backend = DeterministicFakeBatchRaven(profile, num_envs=2)
    with tempfile.TemporaryDirectory(
        prefix="dagger-v2-conformance-", dir="/mnt/data1/ldu/research"
    ) as temp_name:
        temp = Path(temp_name)
        owner = IsaacBatchTransitionOwner(
            backend,
            temp / "owner",
            temp / "staging",
            shape_profile=profile,
            static_provenance=fake_static_provenance(),
            persist_chunks=False,
        )
        keys = _keys()
        owner.begin_windows(
            [
                {
                    "contract_version": CONTRACT_VERSION,
                    "request_id": f"begin-{lane}",
                    "step_key": key.as_dict(),
                }
                for lane, key in enumerate(keys)
            ]
        )
        local = owner.prepare_batch_lean(
            [
                {
                    "contract_version": CONTRACT_VERSION,
                    "request_id": f"prepare-{lane}",
                    "step_key": key.as_dict(),
                    "beta": ROUND_BETAS[key.round_id],
                }
                for lane, key in enumerate(keys)
            ]
        )
        transport = BoundedBearerBatchV2Transport(base_url, token_file)
        try:
            anchor_records, anchor_groups = anchor_records_from_prepares(local)
            anchor_id, anchor_body = build_anchor_batch(anchor_records, anchor_groups)
            anchor_response = transport.post_binary(
                H200_ANCHOR_BATCH_PATH_V2,
                anchor_body,
                batch_id=anchor_id,
                max_response_bytes=1 << 20,
            )
            anchor_acks = validate_anchor_batch_response(
                anchor_response, batch_id=anchor_id, expected=anchor_records
            )

            prepare_idems = [
                f"h200-prepare-{sha256_bytes(canonical_json(key.as_dict()))[:48]}"
                for key in keys
            ]
            prepare_id, prepare_body, prepare_records, _ = build_prepare_batch(
                local, prepare_idems, shape_profile=profile
            )
            first_prepare = transport.post_binary(
                H200_PREPARE_BATCH_PATH_V2,
                prepare_body,
                batch_id=prepare_id,
                max_response_bytes=4 << 20,
            )
            first_selections = validate_prepare_batch_response(
                first_prepare,
                batch_id=prepare_id,
                request_records=prepare_records,
                local_prepares=local,
            )
            retry_prepare = transport.post_binary(
                H200_PREPARE_BATCH_PATH_V2,
                prepare_body,
                batch_id=prepare_id,
                max_response_bytes=4 << 20,
            )
            retry_selections = validate_prepare_batch_response(
                retry_prepare,
                batch_id=prepare_id,
                request_records=prepare_records,
                local_prepares=local,
            )
            if canonical_json(first_prepare) != canonical_json(retry_prepare):
                raise RuntimeError("identical prepare retry changed the response")

            changed = []
            for item in local:
                changed.append(
                    {
                        **item,
                        "metadata": json.loads(canonical_json(item["metadata"])),
                        "arrays": dict(item["arrays"]),
                    }
                )
            changed[0]["arrays"]["actor_obs"] = changed[0]["arrays"]["actor_obs"].copy()
            changed[0]["arrays"]["actor_obs"][0] += 1.0
            _same_id, changed_body, _changed_records, _ = build_prepare_batch(
                changed,
                prepare_idems,
                shape_profile=profile,
                batch_id=prepare_id,
            )
            changed_status = None
            try:
                transport.post_binary(
                    H200_PREPARE_BATCH_PATH_V2,
                    changed_body,
                    batch_id=prepare_id,
                    max_response_bytes=4 << 20,
                )
            except RemoteHttpError as exc:
                changed_status = exc.status
            if changed_status != 409:
                raise RuntimeError(
                    f"changed prepare under stable idempotency keys returned {changed_status}, expected 409"
                )

            local_commits = owner.commit_batch(
                [
                    build_owner_commit(
                        {
                            name: value
                            for name, value in selection.items()
                            if name != "_selected_action_array"
                        }
                    )
                    for selection in first_selections
                ]
            )
            commit_idems = [
                f"h200-commit-{sha256_bytes(canonical_json(key.as_dict()))[:48]}"
                for key in keys
            ]
            clean_selections = [
                {
                    name: value
                    for name, value in selection.items()
                    if name != "_selected_action_array"
                }
                for selection in first_selections
            ]
            commit_id, commit_payload, commit_lanes = build_commit_batch(
                clean_selections, local_commits, commit_idems
            )
            first_commit = transport.post_json(
                H200_COMMIT_BATCH_PATH_V2,
                commit_payload,
                max_response_bytes=4 << 20,
            )
            first_acks = validate_commit_batch_response(
                first_commit, batch_id=commit_id, request_lanes=commit_lanes
            )
            retry_commit = transport.post_json(
                H200_COMMIT_BATCH_PATH_V2,
                commit_payload,
                max_response_bytes=4 << 20,
            )
            retry_acks = validate_commit_batch_response(
                retry_commit, batch_id=commit_id, request_lanes=commit_lanes
            )
            if canonical_json(first_commit) != canonical_json(retry_commit):
                raise RuntimeError("identical commit retry changed the response")
        finally:
            transport.close()

    report = {
        "schema": "roborender.dagger.rtx.v2-cross-host-conformance.v1",
        "result": "PASS",
        "contract_version": CONTRACT_VERSION_V2,
        "ready": {
            "role": ready.get("role"),
            "schema": ready.get("schema"),
            "supported_contract_versions": ready.get("supported_contract_versions"),
            "features": ready.get("features"),
            "pid": ready.get("pid"),
        },
        "batch_size": 2,
        "anchor_count": len(anchor_acks),
        "anchor_statuses": [item["status"] for item in anchor_acks],
        "prepare_batch_id": prepare_id,
        "prepare_body_bytes": len(prepare_body),
        "prepare_body_sha256": sha256_bytes(prepare_body),
        "prepare_lane_count": len(first_selections),
        "prepare_identical_retry": True,
        "changed_body_conflict_status": changed_status,
        "commit_batch_id": commit_id,
        "commit_body_bytes": len(canonical_json(commit_payload)),
        "commit_body_sha256": sha256_bytes(canonical_json(commit_payload)),
        "commit_lane_count": len(first_acks),
        "commit_identical_retry": first_acks == retry_acks,
        "fake_vector_env_step_calls": backend.batch_step_calls,
        "isaac_initialized": False,
        "live_renderer_used": False,
        "learner_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.write_bytes(canonical_json(report) + b"\n")
    output.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18620")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.token_file, args.output), indent=2))


if __name__ == "__main__":
    main()
