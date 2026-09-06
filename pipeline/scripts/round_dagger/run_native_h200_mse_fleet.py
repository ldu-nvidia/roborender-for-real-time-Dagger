#!/usr/bin/env python3
"""Compute one candidate's sealed validation MSE on all eight local H200s."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round_dagger.artifacts import load_strict_json  # noqa: E402
from round_dagger.native_h200_experiment import (  # noqa: E402
    Controller,
    VALIDATION_MSE_RESULT_SCHEMA,
)
from round_dagger.native_h200_fleet import (  # noqa: E402
    CANONICAL_CAMPAIGN_ROOT,
    NativeH200FleetError,
    query_h200_inventory,
    require_environment,
    run_checked,
    safe_directory,
    safe_python_launcher,
    safe_regular,
    sanitized_environment,
    sha256_file,
    strict_document,
)
from round_dagger.native_h200_validation import verify_collection  # noqa: E402
from round_dagger.native_actor_protocol import train_round_relative_root  # noqa: E402


EVALUATOR = ROOT / "scripts/round_dagger/evaluate_validation_mse.py"
LEARNER_CONTRACT = ROOT / "configs/round_dagger/production_learner_local_h200_8rank_1200.json"
VALIDATION_ROOT = CANONICAL_CAMPAIGN_ROOT / "validation/heldout"
STUDENT_CHECKPOINT_ROOT = Path("/workspace/project/roborender/checkpoints/students")


def _validation_snapshot() -> dict:
    seals = VALIDATION_ROOT / "seals"
    safe_directory(seals, "validation seal directory", root=VALIDATION_ROOT)
    candidates = sorted(seals.glob("*.seal.json"))
    if len(candidates) != 1:
        raise NativeH200FleetError(
            "validation must have exactly one immutable dataset seal"
        )
    try:
        return verify_collection(candidates[0])
    except Exception as exc:
        raise NativeH200FleetError(f"sealed validation dataset is invalid: {exc}") from exc


def _learner_python() -> Path:
    contract = load_strict_json(LEARNER_CONTRACT)
    execution = contract.get("execution", {})
    expected = ROOT.parent / "runtime/envs/h200_learner/bin/python"
    if (
        execution.get("backend") != "local_h200"
        or execution.get("mode") != "direct"
        or execution.get("nodes") != 1
        or execution.get("python") != str(expected)
    ):
        raise NativeH200FleetError("validation learner runtime is not local 8xH200")
    return safe_python_launcher(
        expected, "learner Python", target_root=Path("/usr/bin")
    )


def main() -> int:
    values = require_environment("validation")
    if values.get("ROUND_DAGGER_EVALUATION_PURPOSE") != "validation_mse" or values.get(
        "ROUND_DAGGER_EVALUATOR_WORLD_SIZE"
    ) != "8":
        raise NativeH200FleetError("validation fleet must be offline MSE on eight ranks")
    required = (
        "ROUND_DAGGER_POLICY_CHECKPOINT",
        "ROUND_DAGGER_POLICY_SHA256",
        "ROUND_DAGGER_VALIDATION_REGISTRY",
        "ROUND_DAGGER_VALIDATION_REGISTRY_SHA256",
        "ROUND_DAGGER_VALIDATION_RESULT_DIR",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise NativeH200FleetError("validation binding is incomplete: " + ", ".join(missing))
    generation_root = Path(values["ROUND_DAGGER_GENERATION_ROOT"])
    checkpoint = safe_regular(
        Path(values["ROUND_DAGGER_POLICY_CHECKPOINT"]),
        "validation checkpoint",
        root=generation_root,
        expected_sha256=values["ROUND_DAGGER_POLICY_SHA256"],
    )
    strict_document(
        Path(values["ROUND_DAGGER_VALIDATION_REGISTRY"]),
        "committed validation registry",
        root=ROOT,
        expected_sha256=values["ROUND_DAGGER_VALIDATION_REGISTRY_SHA256"],
        read_only=False,
    )
    result_dir = safe_directory(
        Path(values["ROUND_DAGGER_VALIDATION_RESULT_DIR"]),
        "validation result directory",
        root=CANONICAL_CAMPAIGN_ROOT,
    )
    expected_result_dir = (
        CANONICAL_CAMPAIGN_ROOT
        / "validation/results"
        / train_round_relative_root(values["ROUND_DAGGER_GENERATION_ID"])
        / "mse"
    )
    if result_dir != expected_result_dir:
        raise NativeH200FleetError("validation result directory is not round-bound")

    state = Controller(CANONICAL_CAMPAIGN_ROOT).status()
    if state["phase"] not in {"bc_running", "dagger_running"}:
        raise NativeH200FleetError("controller does not admit validation MSE")
    stage = "BC" if generation_root == CANONICAL_CAMPAIGN_ROOT / "BC/isaac" else generation_root.name
    candidate_root = STUDENT_CHECKPOINT_ROOT / stage / "training_steps"
    plan_path = candidate_root / "learner-stage.v1.json"
    bundle_path = candidate_root / "learner-bundle.v1.json"
    plan = strict_document(plan_path, "learner stage plan", root=STUDENT_CHECKPOINT_ROOT)
    bundle = safe_regular(bundle_path, "learner bundle", root=STUDENT_CHECKPOINT_ROOT)
    if (
        plan.get("generation_id") != values["ROUND_DAGGER_GENERATION_ID"]
        or plan.get("learner_bundle_path") != str(bundle)
        or plan.get("learner_bundle_sha256") != sha256_file(bundle)
    ):
        raise NativeH200FleetError("validation learner-stage binding differs")
    validation = _validation_snapshot()
    if validation.get("case_count") != 256 or validation.get("transition_count") != 256 * 512:
        raise NativeH200FleetError("validation population is not exact 256x512")
    query_h200_inventory()
    learner_python = _learner_python()
    result_path = result_dir / f"validation-{values['ROUND_DAGGER_POLICY_SHA256']}.json"
    if result_path.exists():
        result = strict_document(result_path, "validation MSE result", root=generation_root)
        if (
            result.get("schema") != VALIDATION_MSE_RESULT_SCHEMA
            or result.get("checkpoint_sha256") != values["ROUND_DAGGER_POLICY_SHA256"]
        ):
            raise NativeH200FleetError("existing validation result differs")
        print(f"NATIVE_H200_MSE_RESULT_READY path={result_path} sha256={sha256_file(result_path)}")
        return 0

    environment = sanitized_environment(
        {
            "PYTHONPATH": f"{ROOT}:{ROOT / 'GR00T-VisualSim2Real'}",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    run_checked(
        [
            learner_python,
            "-m", "torch.distributed.run",
            "--nnodes", "1",
            "--nproc-per-node", "8",
            "--master-addr", "127.0.0.1",
            "--master-port", "29641",
            EVALUATOR,
            "--bundle", bundle,
            "--bundle-sha256", sha256_file(bundle),
            "--checkpoint", checkpoint,
            "--checkpoint-sha256", values["ROUND_DAGGER_POLICY_SHA256"],
            "--validation-artifact", validation["snapshot_path"],
            "--validation-artifact-sha256", validation["snapshot_file_sha256"],
            "--validation-snapshot-sha256", validation["snapshot_sha256"],
            "--split-sha256", values["ROUND_DAGGER_SPLIT_SHA256"],
            "--learner-identity-sha256", plan["learner_identity_sha256"],
            "--runtime-environment-sha256", plan["runtime_environment_sha256"],
            "--learner-contract-sha256", plan["learner_contract_sha256"],
            "--artifact-root", VALIDATION_ROOT,
            "--model-factory", "round_dagger.gr00t_policy:create_vision_actor_obj_pred",
            "--adapter-factory", "round_dagger.gr00t_policy:vision_actor_obj_pred_adapter_factory",
            "--device", "cuda",
            "--output", result_path,
        ],
        environment=environment,
        cwd=ROOT,
    )
    result = strict_document(result_path, "published validation MSE", root=generation_root)
    if (
        result.get("schema") != VALIDATION_MSE_RESULT_SCHEMA
        or result.get("checkpoint_sha256") != values["ROUND_DAGGER_POLICY_SHA256"]
        or result.get("validation_artifact_sha256") != validation["snapshot_file_sha256"]
        or result.get("validation_snapshot_sha256") != validation["snapshot_sha256"]
    ):
        raise NativeH200FleetError("published validation result lineage differs")
    print(f"NATIVE_H200_MSE_FLEET_OK path={result_path} sha256={sha256_file(result_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, NativeH200FleetError) as error:
        print(f"native H200 MSE fleet failed: {error}", file=sys.stderr)
        raise SystemExit(2)
