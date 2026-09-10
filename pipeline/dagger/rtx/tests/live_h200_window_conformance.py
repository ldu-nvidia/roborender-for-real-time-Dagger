#!/usr/bin/env python3
"""One complete fake-RTX 81-step window against the real H200 test boundary.

This script never imports or initializes Isaac.  It deliberately uses the full
production tensor profile so H200 exercises the frozen wire contract, ledger,
seal cross-check, chunk assembly, and isolated test-sink publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.contracts import CONTRACT_VERSION, RavenShapeProfile
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.http_transport import BoundedBearerJsonTransport
from dagger_rtx.window_runner import BatchWindowRunner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18620")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.base_url != "http://127.0.0.1:18620":
        raise SystemExit("live conformance is restricted to loopback port 18620")

    work = args.scratch_root / args.run_id
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    profile = RavenShapeProfile()
    backend = DeterministicFakeBatchRaven(
        profile,
        num_envs=1,
        # The transition at rollout index 39 terminates episode serial zero;
        # index 40 must bind serial one at episode step zero.
        done_at_calls={0: {39}},
    )
    owner = IsaacBatchTransitionOwner(
        backend,
        work / "owner",
        work / "staging",
        shape_profile=profile,
        static_provenance=fake_static_provenance(),
    )
    transport = BoundedBearerJsonTransport(
        args.base_url, args.token_file, timeout_s=180.0
    )
    orchestrator = BatchStepOrchestrator(
        owner, transport, work / "orchestrator", num_envs=1
    )
    runner = BatchWindowRunner(
        owner,
        orchestrator,
        run_id=args.run_id,
        round_id=1,
        actor_prefix="rtx-fake-window",
        initial_episode_serials=[0],
    )

    started_ns = time.time_ns()
    result = runner.run_window(0)
    elapsed_seconds = (time.time_ns() - started_ns) / 1e9
    if backend.batch_step_calls != 81:
        raise RuntimeError(f"fake vector step count {backend.batch_step_calls} != 81")
    if result["transition_batches"] != 81 or result["physical_transition_rows"] != 81:
        raise RuntimeError("runner returned the wrong 81-step cardinality")

    lane = owner.status()["lanes"][0]
    seal = lane.get("sealed")
    if not isinstance(seal, dict):
        raise RuntimeError("RTX lane did not produce its index-80 audit seal")
    chunk_root = (
        work
        / "staging"
        / args.run_id
        / "D1"
        / "rtx-fake-window-lane-000"
        / "window-000000"
    )
    artifact_path = chunk_root / seal["path"]
    manifest_path = chunk_root / "manifest.json"
    if _sha256(artifact_path) != seal["sha256"]:
        raise RuntimeError("sealed artifact bytes differ from the durable seal")
    with np.load(artifact_path, allow_pickle=False) as data:
        if data["rollout_step_indices"].tolist() != list(range(81)):
            raise RuntimeError("sealed rollout indices are not exactly 0..80")
        if not bool(data["dones"][39]) or bool(data["dones"][:39].any()) or bool(
            data["dones"][40:].any()
        ):
            raise RuntimeError("sealed done boundary is not exactly rollout index 39")
        if int(data["episode_step_indices"][39]) != 39:
            raise RuntimeError("terminal episode step should be 39")
        if int(data["episode_step_indices"][40]) != 0:
            raise RuntimeError("post-done episode did not reset to step zero")
        if str(data["episode_ids"][39]) == str(data["episode_ids"][40]):
            raise RuntimeError("post-done row did not bind a new episode identity")

    final_ack = result["final"]["remote_acknowledgements"][0]
    report = {
        "schema": "roborender.dagger.rtx.live-window-conformance.v1",
        "contract_version": CONTRACT_VERSION,
        "result": "PASS",
        "no_isaac_import_or_initialization": True,
        "fake_rtx_backend": True,
        "batch_size": 1,
        "transition_batches": 81,
        "fake_vector_env_step_calls": backend.batch_step_calls,
        "done_rollout_step_index": 39,
        "post_done_episode_reset_index": 40,
        "index_80_sealed": True,
        "remote_index_80_acknowledged": final_ack.get("status") == "committed",
        "elapsed_seconds": elapsed_seconds,
        "rtx_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "rtx_sealed_npz": {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
        },
        "final_remote_ack": final_ack,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.chmod(0o600)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
