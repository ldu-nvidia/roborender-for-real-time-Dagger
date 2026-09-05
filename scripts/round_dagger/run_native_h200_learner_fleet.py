#!/usr/bin/env python3
"""Advance one train-only BC/DAgger frontier on all eight local H200 GPUs."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import load_strict_json  # noqa: E402
from round_dagger.native_h200_experiment import Controller  # noqa: E402
from round_dagger.native_h200_fleet import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    NativeH200FleetError,
    canonical_sha256,
    query_h200_inventory,
    require_environment,
    require_sha256,
    run_checked,
    safe_directory,
    safe_python_launcher,
    safe_regular,
    sanitized_environment,
    sha256_file,
    strict_document,
)


MANIFEST_SCHEMA = "fast_roborender.round_dagger.train_replay_manifest.v3"
PLAN_SCHEMA = "fast_roborender.round_dagger.native_h200_learner_stage.v1"
LEARNER_CONTRACT = ROOT / "configs/round_dagger/production_learner_local_h200_8rank_1200.json"
PROTOCOL = ROOT / "configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json"
PREPARE = ROOT / "scripts/round_dagger/prepare_native_h200_learner.py"
LAUNCHER = ROOT / "scripts/round_dagger/run_8rank_learner.sh"


def _source_sha256() -> str:
    """Hash the local learner/GR00T source bytes, independent of checkout path."""

    roots = (ROOT / "round_dagger", ROOT / "scripts/round_dagger", ROOT / "GR00T-VisualSim2Real/gr00t/rl")
    rows: list[tuple[str, int, str]] = []
    for source_root in roots:
        safe_directory(source_root, "learner source root")
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(ROOT).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise NativeH200FleetError(f"learner source contains a symlink: {relative}")
            if not stat.S_ISREG(info.st_mode):
                continue
            if path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
                continue
            rows.append((relative, info.st_size, sha256_file(path)))
    if not rows:
        raise NativeH200FleetError("learner source inventory is empty")
    return canonical_sha256(rows)


def _load_local_contract() -> tuple[dict, str, Path]:
    contract = load_strict_json(LEARNER_CONTRACT)
    expected = {
        "backend": "local_h200",
        "mode": "direct",
        "nodes": 1,
        "python": str(ROOT / "envs/h200_learner/bin/python"),
        "entrypoint": "scripts/round_dagger/run_8rank_learner.sh",
    }
    execution = contract.get("execution", {})
    if any(execution.get(key) != value for key, value in expected.items()):
        raise NativeH200FleetError("local learner contract execution differs")
    allocation = contract.get("allocation", {})
    if allocation != {
        "world_size": 8,
        "batch_size_per_rank": 8,
        "global_batch_size": 64,
    }:
        raise NativeH200FleetError("local learner allocation is not exact 8xH200")
    python = safe_python_launcher(
        Path(execution["python"]), "learner Python", target_root=Path("/usr/bin")
    )
    return contract, sha256_file(LEARNER_CONTRACT), python


def _validate_binding(values: dict[str, str]) -> tuple[dict, Path, Path, Path]:
    required = (
        "NATIVE_H200_TRAIN_REPLAY_MANIFEST",
        "NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256",
        "ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT",
        "ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256",
        "ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_SHA256",
        "ROUND_DAGGER_PARENT_CHECKPOINT",
        "ROUND_DAGGER_PARENT_CHECKPOINT_SHA256",
        "ROUND_DAGGER_PARENT_POLICY_SPEC",
        "ROUND_DAGGER_PARENT_POLICY_SPEC_SHA256",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise NativeH200FleetError("learner binding is incomplete: " + ", ".join(missing))
    manifest = strict_document(
        Path(values["NATIVE_H200_TRAIN_REPLAY_MANIFEST"]),
        "train replay manifest",
        root=CANONICAL_CAMPAIGN_ROOT,
        expected_sha256=values["NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256"],
    )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("generation_id") != values["ROUND_DAGGER_GENERATION_ID"]
        or manifest.get("split_sha256") != values["ROUND_DAGGER_SPLIT_SHA256"]
    ):
        raise NativeH200FleetError("train replay manifest lineage differs")
    snapshot_binding = manifest.get("train_replay_snapshot")
    expected_snapshot = {
        "path": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT"],
        "file_sha256": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256"],
        "snapshot_sha256": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_SHA256"],
        "through_macro_round": 0 if manifest.get("round") is None else manifest.get("round"),
    }
    if snapshot_binding != expected_snapshot:
        raise NativeH200FleetError("train replay snapshot binding differs")
    snapshot = safe_regular(
        Path(expected_snapshot["path"]),
        "train replay snapshot",
        root=CANONICAL_CAMPAIGN_ROOT,
        expected_sha256=expected_snapshot["file_sha256"],
    )
    parent = safe_regular(
        Path(values["ROUND_DAGGER_PARENT_CHECKPOINT"]),
        "parent checkpoint",
        root=CANONICAL_CAMPAIGN_ROOT,
        expected_sha256=values["ROUND_DAGGER_PARENT_CHECKPOINT_SHA256"],
    )
    policy_spec = safe_regular(
        Path(values["ROUND_DAGGER_PARENT_POLICY_SPEC"]),
        "parent policy specification",
        root=CANONICAL_CAMPAIGN_ROOT,
        expected_sha256=values["ROUND_DAGGER_PARENT_POLICY_SPEC_SHA256"],
    )
    return manifest, snapshot, parent, policy_spec


def _stage_paths(generation_root: Path) -> tuple[Path, Path, Path]:
    candidate = safe_directory(
        generation_root / "checkpoints/candidates",
        "candidate checkpoint directory",
        root=CANONICAL_CAMPAIGN_ROOT,
    )
    return (
        candidate / "learner-bundle.v1.json",
        candidate / "learner-stage.v1.json",
        candidate,
    )


def main() -> int:
    values = require_environment("learner")
    if values.get("ROUND_DAGGER_LEARNER_NNODES") != "1" or values.get(
        "ROUND_DAGGER_LEARNER_NPROC_PER_NODE"
    ) != "8":
        raise NativeH200FleetError("learner launch topology must be 1x8")
    generation_root = Path(values["ROUND_DAGGER_GENERATION_ROOT"])
    manifest, snapshot, parent, policy_spec = _validate_binding(values)
    round_number = 0 if manifest["round"] is None else int(manifest["round"])
    state = Controller(CANONICAL_CAMPAIGN_ROOT).status()
    if state["phase"] not in {"bc_running", "dagger_running"}:
        raise NativeH200FleetError("controller does not admit learner execution")
    if state["train_replay_manifest_sha256"] != values[
        "NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256"
    ]:
        raise NativeH200FleetError("controller train replay binding changed")

    inventory = query_h200_inventory()
    contract, learner_contract_sha, learner_python = _load_local_contract()
    source_sha = _source_sha256()
    runtime_sha = canonical_sha256(
        {
            "schema": "fast_roborender.round_dagger.native_h200_learner_runtime.v1",
            "gpu_inventory": inventory,
            "learner_python": str(learner_python),
            "learner_contract_sha256": learner_contract_sha,
        }
    )
    bundle_path, plan_path, candidate_dir = _stage_paths(generation_root)
    if bundle_path.exists() != plan_path.exists():
        raise NativeH200FleetError("learner stage publication is partial")
    if not bundle_path.exists():
        run_checked(
            [
                learner_python,
                PREPARE,
                "--campaign-root", CANONICAL_CAMPAIGN_ROOT,
                "--protocol", PROTOCOL,
                "--macro-round", str(round_number),
                "--train-snapshot", snapshot,
                "--train-snapshot-file-sha256", values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256"],
                "--parent-checkpoint", parent,
                "--parent-checkpoint-sha256", values["ROUND_DAGGER_PARENT_CHECKPOINT_SHA256"],
                "--parent-policy-spec", policy_spec,
                "--code-sha256", source_sha,
                "--runtime-environment-sha256", runtime_sha,
                "--learner-contract-sha256", learner_contract_sha,
                "--batch-size-per-rank", "8",
                "--world-size", "8",
                "--bundle-output", bundle_path,
                "--plan-output", plan_path,
            ],
            environment=sanitized_environment({"PYTHONPATH": str(ROOT)}),
            cwd=ROOT,
        )
    plan = strict_document(plan_path, "learner stage plan", root=CANONICAL_CAMPAIGN_ROOT)
    bundle_path = safe_regular(bundle_path, "learner bundle", root=CANONICAL_CAMPAIGN_ROOT)
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": "dagger-redo-v001",
        "generation_id": values["ROUND_DAGGER_GENERATION_ID"],
        "macro_round": round_number,
        "train_snapshot_file_sha256": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256"],
        "train_snapshot_sha256": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_SHA256"],
        "parent_checkpoint_sha256": values["ROUND_DAGGER_PARENT_CHECKPOINT_SHA256"],
        "runtime_environment_sha256": runtime_sha,
        "learner_contract_sha256": learner_contract_sha,
        "learner_bundle_path": str(bundle_path),
        "learner_bundle_sha256": sha256_file(bundle_path),
    }
    mismatches = [key for key, expected in expected_plan.items() if plan.get(key) != expected]
    if mismatches:
        raise NativeH200FleetError("learner stage plan differs: " + ", ".join(mismatches))
    frontiers = plan.get("frontier_optimizer_examples")
    if (
        not isinstance(frontiers, list)
        or not frontiers
        or any(type(item) is not int or item <= 0 for item in frontiers)
        or frontiers != sorted(set(frontiers))
    ):
        raise NativeH200FleetError("learner frontier schedule is malformed")
    evaluated = [int(row["optimizer_examples"]) for row in state["history"]]
    next_index = len(evaluated)
    if next_index >= len(frontiers):
        raise NativeH200FleetError("learner exhausted its maximum exposure without a stop")
    target = frontiers[next_index]
    if evaluated and evaluated != frontiers[:next_index]:
        raise NativeH200FleetError("controller validation history skipped a learner frontier")
    frontier = candidate_dir / f"frontier-examples-{target:012d}.pt"
    if frontier.exists():
        safe_regular(frontier, "learner frontier", root=CANONICAL_CAMPAIGN_ROOT)
        print(f"NATIVE_H200_LEARNER_FRONTIER_READY path={frontier} sha256={sha256_file(frontier)}")
        return 0

    launch_environment = sanitized_environment(
        {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(CANONICAL_CAMPAIGN_ROOT),
            "ROUND_DAGGER_CAMPAIGN_ID": "dagger-redo-v001",
            "ROUND_DAGGER_GENERATION_ID": values["ROUND_DAGGER_GENERATION_ID"],
            "ROUND_DAGGER_MACRO_ROUND": str(round_number),
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(parent),
            "ROUND_DAGGER_POLICY_SHA256": values["ROUND_DAGGER_PARENT_CHECKPOINT_SHA256"],
            "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot),
            "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": values["ROUND_DAGGER_TRAIN_REPLAY_SNAPSHOT_FILE_SHA256"],
            "ROUND_DAGGER_LEARNER_BUNDLE": str(bundle_path),
            "ROUND_DAGGER_LEARNER_BUNDLE_SHA256": sha256_file(bundle_path),
            "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(candidate_dir),
            "ROUND_DAGGER_TARGET_OPTIMIZER_EXAMPLES": str(target),
            "ROUND_DAGGER_VALIDATION_INTERVAL_OPTIMIZER_EXAMPLES": str(plan["validation_interval_optimizer_examples"]),
            "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": require_sha256(plan["learner_identity_sha256"], "learner identity"),
            "ROUND_DAGGER_RUNTIME_ENVIRONMENT_SHA256": runtime_sha,
            "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": learner_contract_sha,
            "ROUND_DAGGER_LEARNER_PYTHON": str(learner_python),
            "NATIVE_H200_TRAIN_REPLAY_MANIFEST": values["NATIVE_H200_TRAIN_REPLAY_MANIFEST"],
            "NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256": values["NATIVE_H200_TRAIN_REPLAY_MANIFEST_SHA256"],
        }
    )
    run_checked([LAUNCHER], environment=launch_environment, cwd=ROOT)
    safe_regular(frontier, "published learner frontier", root=CANONICAL_CAMPAIGN_ROOT)
    print(f"NATIVE_H200_LEARNER_FLEET_OK path={frontier} sha256={sha256_file(frontier)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, NativeH200FleetError) as error:
        print(f"native H200 learner fleet failed: {error}", file=sys.stderr)
        raise SystemExit(2)
