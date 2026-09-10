#!/usr/bin/env python3
"""Resume only the H200 acknowledgement phase of a sealed vector batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.contracts import StepKey
from dagger_rtx.durable import atomic_write_json
from dagger_rtx.http_transport import BoundedBearerJsonTransport


class NoNativeOwner:
    """Tripwire: recovery must never cross an RTX-native boundary."""

    def prepare_batch(self, requests):
        raise AssertionError("recovery attempted native prepare")

    def commit_batch(self, requests):
        raise AssertionError("recovery attempted env.step")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18620")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--rpc-workers", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted((args.runtime_root / "orchestrator").glob("**/batch-step-080-*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one final batch receipt, found {len(candidates)}")
    state_path = candidates[0]
    before = json.loads(state_path.read_text(encoding="utf-8"))
    if before.get("phase") not in {"LOCAL_COMMITTED", "COMPLETE"}:
        raise RuntimeError(f"unsafe recovery phase {before.get('phase')!r}")
    keys = [StepKey.from_mapping(value) for value in before.get("step_keys", [])]
    if len(keys) != args.batch_size or any(key.rollout_step_index != 80 for key in keys):
        raise RuntimeError("final batch identity is not exactly B lanes at rollout index 80")
    commits = before.get("local_commits")
    if not isinstance(commits, list) or len(commits) != args.batch_size:
        raise RuntimeError("final batch lacks B durable local commits")
    if not all(item.get("sealed") is True for item in commits):
        raise RuntimeError("not every local lane is sealed")
    owner_state_path = args.runtime_root / "owner" / "batch_owner_state.json"
    owner_state = json.loads(owner_state_path.read_text(encoding="utf-8"))
    if owner_state.get("phase") != "SEALED" or owner_state.get("next_rollout_step") != 81:
        raise RuntimeError("owner is not durably SEALED at 81 physical steps")

    started = time.time_ns()
    transport = BoundedBearerJsonTransport(
        args.base_url,
        args.token_file,
        timeout_s=args.timeout_s,
    )
    orchestrator = BatchStepOrchestrator(
        NoNativeOwner(),
        transport,
        args.runtime_root / "orchestrator",
        num_envs=args.batch_size,
        max_parallel_requests=args.rpc_workers,
    )
    result = orchestrator.run_batch(keys)
    if result.get("phase") != "COMPLETE":
        raise RuntimeError("acknowledgement recovery did not reach COMPLETE")
    acknowledgements = result.get("remote_acknowledgements")
    if not isinstance(acknowledgements, list) or len(acknowledgements) != args.batch_size:
        raise RuntimeError("recovery did not return B acknowledgements")
    receipts = [item.get("remote_commit_receipt") for item in acknowledgements]
    if any(not isinstance(item, str) or len(item) != 64 for item in receipts):
        raise RuntimeError("recovery returned an invalid remote receipt")
    if len(set(receipts)) != args.batch_size:
        raise RuntimeError("recovery remote receipts are not unique")

    report = {
        "schema": "roborender.dagger.ack-recovery.v1",
        "result": "PASS",
        "recovery_only": True,
        "native_owner_tripwire": "not_called",
        "physical_env_step_calls_added": 0,
        "batch_state_path": str(state_path),
        "batch_state_sha256": sha256_file(state_path),
        "owner_state_path": str(owner_state_path),
        "owner_phase": owner_state["phase"],
        "physical_vector_steps": owner_state["next_rollout_step"],
        "batch_size": args.batch_size,
        "rollout_step_index": 80,
        "remote_acknowledgements": args.batch_size,
        "unique_remote_commit_receipts": len(set(receipts)),
        "remote_commit_receipts": receipts,
        "elapsed_ns": time.time_ns() - started,
        "completed_time_ns": time.time_ns(),
    }
    atomic_write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "remote_commit_receipts"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
