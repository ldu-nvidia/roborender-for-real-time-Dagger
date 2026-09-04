"""Bounded recovery of one production learner from durable INGESTED state.

This path deliberately does not instantiate :class:`MacroRoundOrchestrator`:
collection, replay ingestion, and orchestration telemetry are immutable inputs
that must not be replayed merely because the external learner allocation
failed.  The helper validates one exact INGESTED revision, invokes only the
explicitly selected backend through the inventory-bound coordinator source
seal, stages its candidate, and commits INGESTED->TRAINED with a revision
compare-and-swap.  Reserved Lepton H200 is the production default; OCI/Slurm
is retained only as an explicitly selected compatibility backend.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from round_dagger import (
    artifacts,
    lepton_job,
    learner_acceptance,
    learner_identity,
    orchestration,
    policy_bundle,
    promotion,
    replay,
    state,
)
from round_dagger.orchestration import (
    BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
    COHORT_COUNT,
    ENVIRONMENTS_PER_COHORT,
    HookSpec,
    LEARNER_BACKEND_LEPTON_JOB_V1,
    LEARNER_BACKEND_OCI_NRT,
    MacroRoundConfig,
    ProductionBackend,
    STEPS_PER_ENVIRONMENT,
    SubprocessHookRunner,
    TARGET_ENVIRONMENTS,
    TRANSITIONS_PER_MACRO_ROUND,
    trusted_train_hook_relative,
    validate_learner_backend,
    validate_production_topology,
)
from scripts.round_dagger import production_launch


PLAN_SCHEMA = "fast_roborender.round_dagger.ingested_learner_recovery_plan.v1"
RECEIPT_SCHEMA = "fast_roborender.round_dagger.ingested_learner_recovery_receipt.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OCI_BRIDGE = REPOSITORY_ROOT / trusted_train_hook_relative(
    LEARNER_BACKEND_OCI_NRT
)
LEPTON_BRIDGE = REPOSITORY_ROOT / trusted_train_hook_relative(
    LEARNER_BACKEND_LEPTON_JOB_V1
)
# Compatibility for callers that imported the old constant.  It now names the
# production default rather than silently selecting the legacy OCI backend.
BRIDGE = LEPTON_BRIDGE
COLLECT_HOOK = REPOSITORY_ROOT / "scripts/round_dagger/run_production_actor_cohort_hook.sh"
EVALUATE_HOOK = REPOSITORY_ROOT / "scripts/round_dagger/run_production_paired_evaluation_hook.sh"


class LearnerRecoveryError(RuntimeError):
    """The bounded learner recovery contract was not satisfied."""


BridgeRunner = Callable[[Path, Mapping[str, str], Path, int], None]


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LearnerRecoveryError(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not path.is_absolute() or path.is_symlink():
        raise LearnerRecoveryError(f"{label} must be an absolute non-symlink path")
    return path


def _empty_or_absent_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise LearnerRecoveryError(f"{label} is a symlink")
    if not path.exists():
        return
    if not path.is_dir() or any(path.iterdir()):
        raise LearnerRecoveryError(f"{label} must be absent or empty")


def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": candidate["policy_version"],
        "macro_round": candidate["macro_round"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_path": candidate["checkpoint_path"],
        "runtime_policy_sha256": candidate["runtime_policy_sha256"],
        "policy_spec_sha256": candidate["policy_spec_sha256"],
        "policy_bundle_manifest_sha256": candidate[
            "policy_bundle_manifest_sha256"
        ],
        "manifest_sha256": candidate["manifest_sha256"],
        "manifest_path": candidate["manifest_path"],
        "parent_policy_version": candidate["parent_policy_version"],
        "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
        "replay_snapshot_sha256": candidate["replay_snapshot_sha256"],
        "code_sha256": candidate["code_sha256"],
        "policy_config_sha256": candidate["learner_config_sha256"],
    }


def _default_bridge_runner(
    bridge: Path, environment: Mapping[str, str], workdir: Path, timeout: int
) -> None:
    try:
        SubprocessHookRunner().run(
            "train",
            HookSpec((str(bridge),), timeout),
            environment=environment,
            workdir=workdir,
        )
    except orchestration.OrchestrationError as exc:
        raise LearnerRecoveryError(
            f"sealed learner bridge could not complete: {exc}"
        ) from exc


def _train_bridge(learner_backend: str) -> Path:
    return REPOSITORY_ROOT / trusted_train_hook_relative(learner_backend)


def _validate_remote_source_root(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(
        token in value for token in ("\x00", "\n", "\r", ",", ":", "/../")
    ):
        raise LearnerRecoveryError(f"{label} is unsafe")
    path = PurePosixPath(value)
    if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise LearnerRecoveryError(f"{label} must be canonical and absolute")
    return value


def _validate_executable(path: str | os.PathLike[str], label: str) -> Path:
    executable = _absolute(path, label)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LearnerRecoveryError(f"{label} must be an executable regular file")
    return executable


def _verify_sealed_train_hook(
    *,
    bridge: Path,
    environment: Mapping[str, str],
    expected_bridge_sha256: str,
) -> None:
    """Verify the exact train primitive through the normal sealed dispatcher."""

    descriptor: int | None = None
    try:
        descriptor, _argv, _root, _inventory = orchestration._verified_source_hook(
            "train",
            HookSpec((str(bridge),), 172800),
            environment,
        )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    except (OSError, orchestration.OrchestrationError) as exc:
        raise LearnerRecoveryError(
            f"sealed learner bridge verification failed: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if digest.hexdigest() != expected_bridge_sha256:
        raise LearnerRecoveryError("sealed learner bridge identity mismatch")


def _recovery_config(
    *,
    campaign_root: Path,
    macro_round: int,
    active: Mapping[str, Any],
    learner_code_sha256: str,
    learner_config_sha256: str,
    evaluator_config_sha256: str,
    macro_round_count: int,
    campaign_contract_sha256: str,
    learner_backend: str,
    learner_backend_contract_sha256: str,
    bridge: Path,
) -> dict[str, Any]:
    return {
        # This record is published immutably before the bounded recovery
        # bridge runs.  Keep its established V4 schema so a retry of an
        # already-ingested campaign reopens the same bytes rather than
        # attempting an incompatible metadata migration.
        "schema": BACKEND_BOUND_PRODUCTION_ORCHESTRATION_CONFIG_SCHEMA,
        "campaign_root": str(campaign_root),
        "actor_root": str(campaign_root / "actors"),
        "macro_round": macro_round,
        "macro_round_count": macro_round_count,
        "campaign_contract_sha256": campaign_contract_sha256,
        "learner_backend": learner_backend,
        "learner_backend_contract_sha256": learner_backend_contract_sha256,
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "learner_code_sha256": learner_code_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_model_lineage_sha256": active["checkpoint_sha256"],
        "evaluator_config_sha256": evaluator_config_sha256,
        "hooks": {
            "collect": {"argv": [str(COLLECT_HOOK)], "timeout_seconds": 14400},
            "transfer": None,
            "ingest": None,
            "train": {"argv": [str(bridge)], "timeout_seconds": 172800},
            "evaluate": {"argv": [str(EVALUATE_HOOK)], "timeout_seconds": 28800},
            "promote": None,
        },
    }


def _bridge_environment(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    round_contract: Mapping[str, Any],
    macro_round: int,
    active: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    learner_output: Path,
    learner_result: Path,
    scene_sample_manifest: Path,
    source_root: str,
    source_sha256: str,
    learner_config_sha256: str,
    learner_contract_sha256: str,
    learner_contract_path: Path,
    learner_contract: Mapping[str, Any],
    learner_backend: str,
    runtime_container_sha256: str,
    evaluation_source_root: Path,
    evaluation_source_inventory_sha256: str,
    learner_backend_contract_path: Path,
    learner_backend_contract_sha256: str,
    lepton_remote_backend_contract: str | None,
    lepton_control_exec_program: Path | None,
    lepton_sdk_python: Path | None,
) -> dict[str, str]:
    handoff = learner_contract["handoff"]
    optimization = learner_contract["optimization"]
    allocation = learner_contract["allocation"]
    wandb = learner_contract["wandb"]
    checkpoint = campaign_root / active["checkpoint_path"]
    identity_sha256 = learner_identity.learner_identity_sha256(
        campaign_id=str(campaign["campaign_id"]),
        macro_round=macro_round,
        learner_code_sha256=source_sha256,
        learner_config_sha256=learner_config_sha256,
        replay_snapshot_file_sha256=str(snapshot["file_sha256"]),
        replay_snapshot_sha256=str(snapshot["snapshot_sha256"]),
        parent_checkpoint_sha256=str(active["checkpoint_sha256"]),
        parent_runtime_policy_sha256=str(active["runtime_policy_sha256"]),
        runtime_container_sha256=runtime_container_sha256,
        learner_contract_sha256=learner_contract_sha256,
    )
    run_id = learner_identity.learner_wandb_run_id(
        campaign_id=str(campaign["campaign_id"]),
        macro_round=macro_round,
        learner_identity_sha256=identity_sha256,
    )
    environment = {
        "DRY_RUN": "0",
        # Preserve a virtual-environment entry point.  Resolving its symlink
        # selects the base interpreter and silently drops the validated venv's
        # NumPy/PyTorch site-packages.
        "ROUND_DAGGER_PYTHON": os.path.abspath(sys.executable),
        "ROUND_DAGGER_CONTROL_PYTHON": os.path.abspath(sys.executable),
        "ROUND_DAGGER_LEARNER_BACKEND": learner_backend,
        "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": str(evaluation_source_root),
        "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": (
            evaluation_source_inventory_sha256
        ),
        "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign_root),
        "ROUND_DAGGER_CAMPAIGN_ID": str(campaign["campaign_id"]),
        "ROUND_DAGGER_MACRO_ROUND": str(macro_round),
        "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
        "ROUND_DAGGER_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
        "ROUND_DAGGER_POLICY_MANIFEST": f"{checkpoint}.manifest.json",
        "ROUND_DAGGER_BASE_POLICY_SPEC": f"{checkpoint}.policy_spec.json",
        "ROUND_DAGGER_POLICY_SPEC_SHA256": str(active["policy_spec_sha256"]),
        "ROUND_DAGGER_POLICY_MANIFEST_SHA256": str(
            active["policy_bundle_manifest_sha256"]
        ),
        "ROUND_DAGGER_POLICY_SHA256": str(active["checkpoint_sha256"]),
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": str(
            active["runtime_policy_sha256"]
        ),
        "ROUND_DAGGER_BETA": repr(float(round_contract["beta"])),
        "ROUND_DAGGER_BETA_SHA256": str(round_contract["beta_sha256"]),
        "ROUND_DAGGER_SCENE_SAMPLE_MANIFEST": str(scene_sample_manifest),
        "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot["path"]),
        "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": str(snapshot["file_sha256"]),
        "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": str(snapshot["snapshot_sha256"]),
        "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(learner_output),
        "ROUND_DAGGER_LEARNER_RESULT": str(learner_result),
        "ROUND_DAGGER_LEARNER_CODE_SHA256": source_sha256,
        "ROUND_DAGGER_LEARNER_CONFIG_SHA256": learner_config_sha256,
        "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": learner_contract_sha256,
        "ROUND_DAGGER_LEARNER_CONTRACT": str(learner_contract_path),
        "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": identity_sha256,
        "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": str(
            active["checkpoint_sha256"]
        ),
        "ROUND_DAGGER_TARGET_ENVIRONMENTS": str(TARGET_ENVIRONMENTS),
        "ROUND_DAGGER_COHORT_COUNT": str(COHORT_COUNT),
        "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": str(ENVIRONMENTS_PER_COHORT),
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": str(
            TRANSITIONS_PER_MACRO_ROUND
        ),
        "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": runtime_container_sha256,
        "ROUND_DAGGER_REMOTE_HANDOFF_ROOT": str(handoff["local_root"]),
        "ROUND_DAGGER_LEARNER_BATCH_SIZE": str(allocation["batch_size_per_rank"]),
        "ROUND_DAGGER_LEARNER_MAX_STEPS": str(optimization["optimizer_steps"]),
        "ROUND_DAGGER_LEARNER_SAVE_EVERY": str(optimization["save_every_steps"]),
        # The sealed bridge treats the production hyperparameter tuple as a
        # byte-exact contract.  ``str(2.5e-5)`` emits ``2.5e-05`` on Python,
        # which is numerically equivalent but intentionally rejected there.
        "ROUND_DAGGER_LEARNER_LEARNING_RATE": "2.5e-5",
        "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": str(optimization["weight_decay"]),
        "ROUND_DAGGER_LEARNER_SEED": str(optimization["seed"]),
        "ROUND_DAGGER_LEARNER_PARTITION_SEED": str(
            optimization["partition_seed"]
        ),
        "ROUND_DAGGER_WANDB_PROJECT": str(wandb["project"]),
        "ROUND_DAGGER_WANDB_ENTITY": str(wandb["entity"]),
        "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": str(wandb["expected_email"]),
        "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE": str(
            wandb["authorization_scope"]
        ),
        "ROUND_DAGGER_WANDB_RUN_ID": run_id,
    }
    if learner_backend == LEARNER_BACKEND_OCI_NRT:
        cluster = learner_contract["cluster"]
        container = learner_contract["container"]
        environment.update(
            {
                "ROUND_DAGGER_REMOTE_HOST": str(cluster["host"]),
                "ROUND_DAGGER_REMOTE_ACCOUNT": str(cluster["account"]),
                "ROUND_DAGGER_REMOTE_PARTITION": str(cluster["partition"]),
                "ROUND_DAGGER_REMOTE_QOS": str(cluster["qos"]),
                "ROUND_DAGGER_REMOTE_STAGE_ROOT": str(cluster["stage_root"]),
                "ROUND_DAGGER_REMOTE_SLURM_BIN": str(cluster["slurm_bin"]),
                "ROUND_DAGGER_REMOTE_WANDB_NETRC": str(cluster["wandb_netrc"]),
                "ROUND_DAGGER_REMOTE_SOURCE_ROOT": source_root,
                "ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256": source_sha256,
                "ROUND_DAGGER_REMOTE_CONTAINER": str(container["path"]),
                "ROUND_DAGGER_REMOTE_POLL_SECONDS": str(handoff["poll_seconds"]),
                "ROUND_DAGGER_REMOTE_MAX_ATTEMPTS": str(
                    handoff["maximum_attempts"]
                ),
            }
        )
    elif learner_backend == LEARNER_BACKEND_LEPTON_JOB_V1:
        assert lepton_remote_backend_contract is not None
        assert lepton_control_exec_program is not None
        assert lepton_sdk_python is not None
        environment.update(
            {
                "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT": str(
                    learner_backend_contract_path
                ),
                "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256": (
                    learner_backend_contract_sha256
                ),
                "ROUND_DAGGER_LEPTON_REMOTE_BACKEND_CONTRACT": (
                    lepton_remote_backend_contract
                ),
                "ROUND_DAGGER_LEPTON_CONTROL_EXEC_PROGRAM": str(
                    lepton_control_exec_program
                ),
                "ROUND_DAGGER_LEPTON_SDK_PYTHON": str(lepton_sdk_python),
            }
        )
    else:  # validate_learner_backend should make this unreachable.
        raise LearnerRecoveryError("unknown learner backend")
    heldout = learner_contract.get("heldout_evaluation")
    if heldout is not None:
        environment.update(
            {
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": str(
                    heldout["snapshot_path"]
                ),
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256": str(
                    heldout["snapshot_file_sha256"]
                ),
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256": str(
                    heldout["snapshot_sha256"]
                ),
                "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL": str(
                    heldout["interval_steps"]
                ),
            }
        )
    return environment


def _validate_backend_binding(
    *,
    learner_backend: str,
    learner_contract: Mapping[str, Any],
    learner_contract_path: Path,
    learner_contract_sha256: str,
    learner_backend_contract_path: str | os.PathLike[str] | None,
    expected_learner_backend_contract_sha256: str | None,
    sealed_source_root: str,
    sealed_source_inventory_sha256: str,
    lepton_remote_backend_contract: str | None,
    lepton_control_exec_program: str | os.PathLike[str] | None,
    lepton_sdk_python: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    try:
        selected = validate_learner_backend(learner_backend)
        contract_backend = production_launch.learner_contract_backend(
            learner_contract
        )
    except (orchestration.OrchestrationError, production_launch.ProductionLaunchError) as exc:
        raise LearnerRecoveryError("learner backend binding is invalid") from exc

    if selected == LEARNER_BACKEND_OCI_NRT:
        if contract_backend != LEARNER_BACKEND_OCI_NRT:
            raise LearnerRecoveryError(
                "OCI recovery requires an explicitly OCI-bound learner contract"
            )
        if any(
            value is not None
            for value in (
                lepton_remote_backend_contract,
                lepton_control_exec_program,
                lepton_sdk_python,
            )
        ):
            raise LearnerRecoveryError("OCI recovery forbids Lepton-specific inputs")
        backend_path = (
            learner_contract_path
            if learner_backend_contract_path is None
            else _absolute(learner_backend_contract_path, "OCI backend contract")
        )
        backend_sha = (
            learner_contract_sha256
            if expected_learner_backend_contract_sha256 is None
            else _sha(
                expected_learner_backend_contract_sha256,
                "OCI backend contract SHA-256",
            )
        )
        if backend_path != learner_contract_path or backend_sha != learner_contract_sha256:
            raise LearnerRecoveryError(
                "OCI backend contract must be the campaign learner contract"
            )
        return {
            "backend": selected,
            "backend_contract_path": backend_path,
            "backend_contract_sha256": backend_sha,
            "runtime_container_sha256": _sha(
                str(learner_contract["container"]["sha256"]),
                "OCI runtime container SHA-256",
            ),
            "lepton_remote_backend_contract": None,
            "lepton_control_exec_program": None,
            "lepton_sdk_python": None,
        }

    if contract_backend not in {None, LEARNER_BACKEND_LEPTON_JOB_V1}:
        raise LearnerRecoveryError(
            "portable/Lepton learner contract is required for Lepton recovery"
        )
    if learner_contract.get("heldout_evaluation") is None:
        raise LearnerRecoveryError(
            "Lepton production recovery requires the held-out-bound learner contract"
        )
    if any(
        value is None
        for value in (
            learner_backend_contract_path,
            expected_learner_backend_contract_sha256,
            lepton_remote_backend_contract,
            lepton_control_exec_program,
            lepton_sdk_python,
        )
    ):
        raise LearnerRecoveryError("Lepton backend inputs are incomplete")
    assert learner_backend_contract_path is not None
    assert expected_learner_backend_contract_sha256 is not None
    assert lepton_remote_backend_contract is not None
    assert lepton_control_exec_program is not None
    assert lepton_sdk_python is not None
    backend_path = _absolute(
        learner_backend_contract_path, "Lepton backend contract"
    )
    backend_sha = _sha(
        expected_learner_backend_contract_sha256,
        "Lepton backend contract SHA-256",
    )
    try:
        backend_contract = lepton_job.load_contract(backend_path, immutable=True)
        lepton_job.require_finalized_supply_chain(backend_contract)
    except lepton_job.LeptonJobError as exc:
        raise LearnerRecoveryError(
            f"finalized Lepton backend contract is invalid: {exc}"
        ) from exc
    if backend_contract.sha256 != backend_sha:
        raise LearnerRecoveryError("Lepton backend contract identity mismatch")
    if (
        str(backend_contract.source_root) != sealed_source_root
        or backend_contract.source_inventory_sha256
        != sealed_source_inventory_sha256
    ):
        raise LearnerRecoveryError(
            "Lepton backend contract differs from the sealed learner source"
        )
    remote_contract = PurePosixPath(lepton_remote_backend_contract)
    try:
        remote_contract.relative_to(backend_contract.project_root)
    except ValueError as exc:
        raise LearnerRecoveryError(
            "remote Lepton backend contract is outside the project root"
        ) from exc
    if (
        not remote_contract.is_absolute()
        or remote_contract.as_posix() != lepton_remote_backend_contract
        or any(part in {"", ".", ".."} for part in remote_contract.parts)
    ):
        raise LearnerRecoveryError(
            "remote Lepton backend contract must be canonical and absolute"
        )
    control_exec = _validate_executable(
        lepton_control_exec_program, "Lepton control helper"
    )
    sdk_python = Path(os.path.abspath(Path(lepton_sdk_python).expanduser()))
    if sdk_python != production_launch.LEPTON_SDK_PYTHON:
        raise LearnerRecoveryError("Lepton SDK Python path differs from its pin")
    try:
        resolved_sdk = sdk_python.resolve(strict=True)
    except OSError as exc:
        raise LearnerRecoveryError("Lepton SDK Python is unavailable") from exc
    if not resolved_sdk.is_file() or not os.access(sdk_python, os.X_OK):
        raise LearnerRecoveryError("Lepton SDK Python is not executable")
    freeze = production_launch.LEPTON_SDK_FREEZE
    if (
        freeze.is_symlink()
        or not freeze.is_file()
        or artifacts.sha256_file(freeze)
        != production_launch.LEPTON_SDK_FREEZE_SHA256
    ):
        raise LearnerRecoveryError(
            "Lepton control environment freeze identity mismatch"
        )
    return {
        "backend": selected,
        "backend_contract_path": backend_path,
        "backend_contract_sha256": backend_sha,
        "runtime_container_sha256": backend_contract.image_digest,
        "lepton_remote_backend_contract": lepton_remote_backend_contract,
        "lepton_control_exec_program": control_exec,
        "lepton_sdk_python": sdk_python,
    }


def _validate_lepton_request_environment(
    *,
    environment: Mapping[str, str],
    backend_contract_path: Path,
    campaign_root: Path,
) -> None:
    """Exercise the public Lepton request contract without remote side effects.

    Merely verifying the sealed shell bytes cannot detect interface drift
    between recovery and the Lepton request materializer.  A throwaway request
    runs the same allowlist, path rebasing, beta lineage, dashboard, identity,
    and backend-contract checks used immediately before a real submission.
    """

    try:
        backend_contract = lepton_job.load_contract(
            backend_contract_path, immutable=True
        )
        identity = environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
        remote_campaign = (
            backend_contract.project_root
            / ".recovery-preflight"
            / identity
            / "campaign_input"
        )
        macro_round = int(environment["ROUND_DAGGER_MACRO_ROUND"])
        remote_output = (
            remote_campaign
            / "rounds"
            / f"macro_round_{macro_round:06d}"
            / "learner"
        )
        with tempfile.TemporaryDirectory(
            prefix="round_dagger_lepton_request_preflight_"
        ) as temporary:
            request = lepton_job.materialize_request(
                Path(temporary) / "request.json",
                contract=backend_contract,
                local_campaign_root=campaign_root,
                remote_campaign_root=str(remote_campaign),
                remote_learner_output=str(remote_output),
                source_environment=environment,
            )
    except (KeyError, TypeError, ValueError, lepton_job.LeptonJobError) as exc:
        raise LearnerRecoveryError(
            f"Lepton request environment preflight failed: {exc}"
        ) from exc
    if request.identity_sha256 != identity:
        raise LearnerRecoveryError("Lepton request preflight identity mismatch")


@orchestration.campaign_mutation_guard(
    operation="recover_ingested_learner",
    root_parameter="campaign_root",
    error_type=LearnerRecoveryError,
)
def recover_ingested_learner(
    *,
    campaign_root: str | os.PathLike[str],
    macro_round: int,
    expected_state_revision: int,
    expected_campaign_sha256: str,
    expected_replay_snapshot_file_sha256: str,
    expected_replay_snapshot_sha256: str,
    expected_parent_checkpoint_sha256: str,
    expected_parent_runtime_policy_sha256: str,
    campaign_contract_path: str | os.PathLike[str],
    expected_campaign_contract_sha256: str,
    learner_contract_path: str | os.PathLike[str],
    expected_learner_contract_sha256: str,
    sealed_source_root: str,
    sealed_source_inventory_sha256: str,
    evaluation_sealed_source_root: str | os.PathLike[str],
    evaluation_sealed_source_inventory_sha256: str,
    expected_bridge_sha256: str,
    learner_backend: str = LEARNER_BACKEND_LEPTON_JOB_V1,
    learner_backend_contract_path: str | os.PathLike[str] | None = None,
    expected_learner_backend_contract_sha256: str | None = None,
    lepton_remote_backend_contract: str | None = None,
    lepton_control_exec_program: str | os.PathLike[str] | None = None,
    lepton_sdk_python: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    bridge_runner: BridgeRunner | None = None,
) -> dict[str, Any]:
    """Run only the learner handoff and CAS an exact INGESTED round to TRAINED."""

    root = _absolute(campaign_root, "campaign root")
    if not root.is_dir():
        raise LearnerRecoveryError("campaign root is missing")
    if type(macro_round) is not int or macro_round < 0:
        raise LearnerRecoveryError("macro_round must be a non-negative integer")
    if type(expected_state_revision) is not int or expected_state_revision < 0:
        raise LearnerRecoveryError("expected_state_revision must be non-negative")
    campaign_sha = _sha(expected_campaign_sha256, "campaign SHA-256")
    snapshot_file_sha = _sha(
        expected_replay_snapshot_file_sha256, "snapshot file SHA-256"
    )
    snapshot_semantic_sha = _sha(
        expected_replay_snapshot_sha256, "snapshot semantic SHA-256"
    )
    parent_sha = _sha(expected_parent_checkpoint_sha256, "parent checkpoint SHA-256")
    parent_runtime_sha = _sha(
        expected_parent_runtime_policy_sha256, "parent runtime SHA-256"
    )
    source_sha = _sha(sealed_source_inventory_sha256, "sealed source inventory SHA-256")
    sealed_source_root = _validate_remote_source_root(
        sealed_source_root, "sealed learner source root"
    )
    evaluation_source_sha = _sha(
        evaluation_sealed_source_inventory_sha256,
        "sealed coordinator source inventory SHA-256",
    )
    evaluation_source_root = _absolute(
        evaluation_sealed_source_root, "sealed coordinator source root"
    )
    bridge_sha = _sha(expected_bridge_sha256, "bridge SHA-256")
    try:
        selected_backend = validate_learner_backend(learner_backend)
    except orchestration.OrchestrationError as exc:
        raise LearnerRecoveryError("learner backend is invalid") from exc
    bridge = _train_bridge(selected_backend)

    campaign_contract_path = _absolute(campaign_contract_path, "campaign contract")
    learner_contract_path = _absolute(learner_contract_path, "learner contract")
    campaign_contract_sha = _sha(
        expected_campaign_contract_sha256, "campaign contract SHA-256"
    )
    learner_contract_sha = _sha(
        expected_learner_contract_sha256, "learner contract SHA-256"
    )
    for path, expected, label in (
        (
            campaign_contract_path,
            campaign_contract_sha,
            "campaign contract",
        ),
        (
            learner_contract_path,
            learner_contract_sha,
            "learner contract",
        ),
        (bridge, bridge_sha, "learner bridge"),
    ):
        if not path.is_file() or artifacts.sha256_file(path) != expected:
            raise LearnerRecoveryError(f"{label} identity mismatch")

    try:
        # Recovery consumes the same exact production-v2 deployment contract
        # as initial launch.  Reusing the strict loader prevents a v1 relabel
        # from omitting the bounded horizon or held-out split identities.
        campaign_contract = production_launch.load_campaign_contract(
            campaign_contract_path
        )
    except production_launch.ProductionLaunchError as exc:
        raise LearnerRecoveryError(
            "campaign contract schema/identity is invalid"
        ) from exc
    try:
        learner_contract, learner_contract_sha = (
            production_launch.load_campaign_owned_learner_contract(
                learner_contract_path,
                campaign_root=root,
                expected_sha256=learner_contract_sha,
                expected_backend=selected_backend,
            )
        )
    except production_launch.ProductionLaunchError as exc:
        raise LearnerRecoveryError(
            "campaign-owned learner contract is invalid"
        ) from exc
    backend_binding = _validate_backend_binding(
        learner_backend=selected_backend,
        learner_contract=learner_contract,
        learner_contract_path=learner_contract_path,
        learner_contract_sha256=learner_contract_sha,
        learner_backend_contract_path=learner_backend_contract_path,
        expected_learner_backend_contract_sha256=(
            expected_learner_backend_contract_sha256
        ),
        sealed_source_root=sealed_source_root,
        sealed_source_inventory_sha256=source_sha,
        lepton_remote_backend_contract=lepton_remote_backend_contract,
        lepton_control_exec_program=lepton_control_exec_program,
        lepton_sdk_python=lepton_sdk_python,
    )
    campaign = state.load_campaign(root)
    current = state.load_state(root)
    contract = state.load_round_contract(root, macro_round)
    validate_production_topology(campaign, contract)
    if (
        artifacts.canonical_json_sha256(campaign) != campaign_sha
        or current["campaign_sha256"] != campaign_sha
        or campaign_contract["campaign_id"] != campaign["campaign_id"]
        or learner_contract_sha != campaign["policy_config_sha256"]
    ):
        raise LearnerRecoveryError("campaign identity mismatch")
    if (
        current["phase"] != "INGESTED"
        or current["open_macro_round"] != macro_round
        or current["revision"] != expected_state_revision
        or current["last_ingested_macro_round"] != macro_round
        or current["candidate_policy"] is not None
    ):
        raise LearnerRecoveryError("coordinator is not the exact requested INGESTED state")
    active = current["active_policy"]
    if (
        active["checkpoint_sha256"] != parent_sha
        or active["runtime_policy_sha256"] != parent_runtime_sha
        or contract["policy_checkpoint_sha256"] != parent_sha
        or contract["policy_runtime_sha256"] != parent_runtime_sha
        or current["replay_snapshot_sha256"] != snapshot_file_sha
    ):
        raise LearnerRecoveryError("parent policy/replay coordinator lineage mismatch")
    checkpoint = root / active["checkpoint_path"]
    try:
        policy_bundle.validate_policy_bundle_identity(
            checkpoint=checkpoint,
            policy_spec=f"{checkpoint}.policy_spec.json",
            manifest=f"{checkpoint}.manifest.json",
            expected_checkpoint_sha256=parent_sha,
            expected_runtime_policy_sha256=parent_runtime_sha,
        )
    except policy_bundle.PolicyBundleError as exc:
        raise LearnerRecoveryError(f"active policy bundle is invalid: {exc}") from exc

    snapshot_path = (
        root
        / "replay/snapshots"
        / f"through_macro_round_{macro_round:06d}.json"
    )
    snapshot = replay.load_snapshot(
        snapshot_path, expected_file_sha256=snapshot_file_sha
    )
    if (
        snapshot["campaign_id"] != campaign["campaign_id"]
        or snapshot["through_macro_round"] != macro_round
        or snapshot["snapshot_sha256"] != snapshot_semantic_sha
        or snapshot["expected_train_cohorts"] != COHORT_COUNT
        or snapshot["cohort_count"] != COHORT_COUNT * (macro_round + 1)
        or snapshot["transition_count"]
        != TRANSITIONS_PER_MACRO_ROUND * (macro_round + 1)
    ):
        raise LearnerRecoveryError("immutable replay snapshot coverage/identity mismatch")

    learner_config_sha = production_launch.derive_learner_bundle_sha256(
        base_policy_spec=Path(f"{checkpoint}.policy_spec.json"),
        parent_checkpoint=checkpoint,
        expected_parent_checkpoint_sha256=parent_sha,
        runtime_policy_sha256=parent_runtime_sha,
        campaign_id=campaign["campaign_id"],
        macro_round=macro_round,
        code_sha256=source_sha,
        learner_contract=learner_contract,
    )
    round_dir = state.round_directory(root, macro_round)
    scene_sample_manifest = round_dir / "scene_samples" / "manifest.json"
    if scene_sample_manifest.is_symlink() or not scene_sample_manifest.is_file():
        raise LearnerRecoveryError(
            "immutable learner scene sample manifest is missing or unsafe"
        )
    try:
        scene_sample_document = artifacts.load_strict_json(scene_sample_manifest)
    except artifacts.ArtifactError as exc:
        raise LearnerRecoveryError(
            "immutable learner scene sample manifest is malformed"
        ) from exc
    if (
        not isinstance(scene_sample_document, Mapping)
        or scene_sample_document.get("schema")
        != "fast_roborender.round_dagger.scene_samples.v1"
        or scene_sample_document.get("campaign_id") != campaign["campaign_id"]
        or scene_sample_document.get("rollout_steps") != STEPS_PER_ENVIRONMENT
    ):
        raise LearnerRecoveryError(
            "immutable learner scene sample manifest identity mismatch"
        )
    learner_output = round_dir / "learner"
    learner_result_path = learner_output / f"learner_result_round_{macro_round:06d}.json"
    candidate_dir = root / "policies" / (
        f"policy_{state._next_candidate_policy_version(current):06d}"
    )
    recovery_dir = round_dir / "learner_recovery"
    config_path = recovery_dir / "orchestration_config.json"
    environment_path = recovery_dir / "bridge_environment.json"
    plan_path = recovery_dir / "plan.json"
    receipt_path = recovery_dir / "trained_receipt.json"

    # A first recovery must begin artifact-free. Identical immutable recovery
    # metadata permits a retry after the bridge returned but before the CAS.
    if plan_path.is_symlink() or (plan_path.exists() and not plan_path.is_file()):
        raise LearnerRecoveryError("recovery plan path is unsafe")
    if not plan_path.exists():
        _empty_or_absent_directory(learner_output, "learner output")
        _empty_or_absent_directory(candidate_dir, "candidate policy directory")
        _empty_or_absent_directory(recovery_dir, "learner recovery directory")

    config_document = _recovery_config(
        campaign_root=root,
        macro_round=macro_round,
        active=active,
        learner_code_sha256=source_sha,
        learner_config_sha256=learner_config_sha,
        evaluator_config_sha256=campaign["training_config_sha256"],
        macro_round_count=campaign["macro_round_count"],
        campaign_contract_sha256=campaign_contract_sha,
        learner_backend=selected_backend,
        learner_backend_contract_sha256=backend_binding[
            "backend_contract_sha256"
        ],
        bridge=bridge,
    )
    config_sha = artifacts.canonical_json_sha256(config_document)
    environment = _bridge_environment(
        campaign_root=root,
        campaign=campaign,
        round_contract=contract,
        macro_round=macro_round,
        active=active,
        snapshot=snapshot,
        learner_output=learner_output,
        learner_result=learner_result_path,
        scene_sample_manifest=scene_sample_manifest,
        source_root=sealed_source_root,
        source_sha256=source_sha,
        learner_config_sha256=learner_config_sha,
        learner_contract_sha256=learner_contract_sha,
        learner_contract_path=learner_contract_path,
        learner_contract=learner_contract,
        learner_backend=selected_backend,
        runtime_container_sha256=backend_binding[
            "runtime_container_sha256"
        ],
        evaluation_source_root=evaluation_source_root,
        evaluation_source_inventory_sha256=evaluation_source_sha,
        learner_backend_contract_path=backend_binding[
            "backend_contract_path"
        ],
        learner_backend_contract_sha256=backend_binding[
            "backend_contract_sha256"
        ],
        lepton_remote_backend_contract=backend_binding[
            "lepton_remote_backend_contract"
        ],
        lepton_control_exec_program=backend_binding[
            "lepton_control_exec_program"
        ],
        lepton_sdk_python=backend_binding["lepton_sdk_python"],
    )
    if selected_backend == LEARNER_BACKEND_LEPTON_JOB_V1:
        _validate_lepton_request_environment(
            environment=environment,
            backend_contract_path=backend_binding["backend_contract_path"],
            campaign_root=root,
        )
    environment_sha = artifacts.canonical_json_sha256(environment)
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign_sha,
        "macro_round": macro_round,
        "expected_ingested_revision": expected_state_revision,
        "replay_snapshot_file_sha256": snapshot_file_sha,
        "replay_snapshot_sha256": snapshot_semantic_sha,
        "parent_policy_version": active["version"],
        "parent_checkpoint_sha256": parent_sha,
        "parent_runtime_policy_sha256": parent_runtime_sha,
        "sealed_source_root": sealed_source_root,
        "sealed_source_inventory_sha256": source_sha,
        "evaluation_sealed_source_root": str(evaluation_source_root),
        "evaluation_sealed_source_inventory_sha256": evaluation_source_sha,
        "learner_config_sha256": learner_config_sha,
        "campaign_contract_sha256": campaign_contract_sha,
        "learner_contract_sha256": learner_contract_sha,
        "learner_backend": selected_backend,
        "learner_backend_contract_path": str(
            backend_binding["backend_contract_path"]
        ),
        "learner_backend_contract_sha256": backend_binding[
            "backend_contract_sha256"
        ],
        "bridge_path": str(bridge),
        "bridge_sha256": bridge_sha,
        "orchestration_config_path": config_path.relative_to(root).as_posix(),
        "orchestration_config_sha256": config_sha,
        "bridge_environment_path": environment_path.relative_to(root).as_posix(),
        "bridge_environment_sha256": environment_sha,
        "learner_result_path": learner_result_path.relative_to(root).as_posix(),
        "candidate_manifest_path": (
            candidate_dir / "candidate.json"
        ).relative_to(root).as_posix(),
    }
    _verify_sealed_train_hook(
        bridge=bridge,
        environment=environment,
        expected_bridge_sha256=bridge_sha,
    )
    if dry_run:
        return {**plan, "status": "validated_no_submission", "bridge_environment": environment}

    # Publish the plan first. A crash after this point is an explicitly bound
    # recovery retry; publishing config/environment first would make their own
    # existence indistinguishable from unowned stale artifacts.
    plan_sha = artifacts.publish_immutable_json(plan_path, plan)
    artifacts.publish_immutable_json(config_path, config_document)
    artifacts.publish_immutable_json(environment_path, environment)
    configuration = MacroRoundConfig.load(config_path)
    if configuration.sha256 != config_sha:
        raise LearnerRecoveryError("published recovery configuration identity mismatch")
    if (
        configuration.learner_backend != selected_backend
        or configuration.learner_backend_contract_sha256
        != backend_binding["backend_contract_sha256"]
    ):
        raise LearnerRecoveryError(
            "published recovery backend binding identity mismatch"
        )
    backend = ProductionBackend(configuration)

    acceptance_sha256: str | None = None
    if selected_backend == LEARNER_BACKEND_LEPTON_JOB_V1:
        acceptance_path = learner_acceptance.acceptance_path(root, macro_round)
        # A returned result without the bridge-owned acceptance receipt is an
        # uncommitted handoff. Re-enter the idempotent bridge even when result
        # bytes already exist, then require a full local revalidation.
        if not os.path.lexists(acceptance_path):
            (bridge_runner or _default_bridge_runner)(
                bridge, environment, root, 172800
            )
        try:
            learner_acceptance.validate_acceptance(
                acceptance_path,
                campaign_root=root,
                macro_round=macro_round,
                backend_contract_path=backend_binding[
                    "backend_contract_path"
                ],
                expected_environment=environment,
            )
        except learner_acceptance.LearnerAcceptanceError as exc:
            raise LearnerRecoveryError(
                f"Lepton learner acceptance is invalid: {exc}"
            ) from exc
        acceptance_sha256 = artifacts.sha256_file(acceptance_path)
    elif not learner_result_path.exists():
        (bridge_runner or _default_bridge_runner)(
            bridge, environment, root, 172800
        )
    after_bridge = state.load_state(root)
    if after_bridge != current:
        raise LearnerRecoveryError("coordinator changed while learner bridge was running")
    learner_result = backend.load_learner_result(
        campaign=campaign, snapshot=snapshot
    )
    candidate = backend.stage_candidate(
        learner_result,
        snapshot,
        learner_acceptance_sha256=acceptance_sha256,
    )
    before_cas = state.load_state(root)
    if before_cas != current:
        raise LearnerRecoveryError("coordinator changed before candidate CAS")
    summary = _candidate_summary(candidate)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "TRAINED",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": campaign_sha,
        "macro_round": macro_round,
        "ingested_revision": expected_state_revision,
        "trained_revision": expected_state_revision + 1,
        "replay_snapshot_file_sha256": snapshot_file_sha,
        "replay_snapshot_sha256": snapshot_semantic_sha,
        "sealed_source_inventory_sha256": source_sha,
        "learner_config_sha256": learner_config_sha,
        "learner_result_path": learner_result_path.relative_to(root).as_posix(),
        "learner_result_sha256": artifacts.sha256_file(learner_result_path),
        "candidate_manifest_path": candidate["manifest_path"],
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "plan_sha256": plan_sha,
    }
    # Publish the exact intended post-CAS receipt before the CAS. Evaluation
    # requires both this receipt and matching TRAINED state, so a failed CAS
    # cannot be mistaken for completion; a crash immediately after the CAS
    # cannot leave TRAINED without its mandatory recovery provenance.
    receipt_sha = artifacts.publish_immutable_json(receipt_path, receipt)
    trained = state.transition(
        root,
        expected_phase="INGESTED",
        next_phase="TRAINED",
        macro_round=macro_round,
        evidence_path=candidate["manifest_path"],
        evidence_sha256=candidate["manifest_sha256"],
        candidate_policy=summary,
        expected_revision=expected_state_revision,
    )
    if trained["revision"] != receipt["trained_revision"]:
        raise LearnerRecoveryError("TRAINED revision differs from recovery receipt")
    return {
        **receipt,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": receipt_sha,
    }


__all__ = [
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "LearnerRecoveryError",
    "recover_ingested_learner",
]
