"""Bounded recovery of one production learner from durable INGESTED state.

This path deliberately does not instantiate :class:`MacroRoundOrchestrator`:
collection, replay ingestion, and orchestration telemetry are immutable inputs
that must not be replayed merely because the external learner allocation
failed.  The helper validates one exact INGESTED revision, invokes only the
sealed OCI learner bridge, stages its candidate, and commits INGESTED->TRAINED
with a revision compare-and-swap.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from round_dagger import (
    artifacts,
    learner_identity,
    policy_bundle,
    promotion,
    replay,
    state,
)
from round_dagger.orchestration import (
    COHORT_COUNT,
    ENVIRONMENTS_PER_COHORT,
    MacroRoundConfig,
    ProductionBackend,
    STEPS_PER_ENVIRONMENT,
    TARGET_ENVIRONMENTS,
    TRANSITIONS_PER_MACRO_ROUND,
    validate_production_topology,
)
from scripts.round_dagger import production_launch


PLAN_SCHEMA = "fast_roborender.round_dagger.ingested_learner_recovery_plan.v1"
RECEIPT_SCHEMA = "fast_roborender.round_dagger.ingested_learner_recovery_receipt.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPOSITORY_ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
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
    child = dict(os.environ)
    child.update(environment)
    try:
        completed = subprocess.run(
            [str(bridge)],
            cwd=workdir,
            env=child,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LearnerRecoveryError(
            f"learner bridge could not complete ({type(exc).__name__})"
        ) from exc
    if completed.returncode != 0:
        raise LearnerRecoveryError(
            f"learner bridge failed with exit status {completed.returncode}"
        )


def _recovery_config(
    *,
    campaign_root: Path,
    macro_round: int,
    active: Mapping[str, Any],
    learner_code_sha256: str,
    learner_config_sha256: str,
    evaluator_config_sha256: str,
    macro_round_count: int,
) -> dict[str, Any]:
    return {
        "schema": "fast_roborender.round_dagger.orchestration_config.v2",
        "campaign_root": str(campaign_root),
        "actor_root": str(campaign_root / "actors"),
        "macro_round": macro_round,
        "macro_round_count": macro_round_count,
        "runtime_policy_sha256": active["runtime_policy_sha256"],
        "learner_code_sha256": learner_code_sha256,
        "learner_config_sha256": learner_config_sha256,
        "learner_model_lineage_sha256": active["checkpoint_sha256"],
        "evaluator_config_sha256": evaluator_config_sha256,
        "hooks": {
            "collect": {"argv": [str(COLLECT_HOOK)], "timeout_seconds": 14400},
            "transfer": None,
            "ingest": None,
            "train": {"argv": [str(BRIDGE)], "timeout_seconds": 172800},
            "evaluate": {"argv": [str(EVALUATE_HOOK)], "timeout_seconds": 28800},
            "promote": None,
        },
    }


def _bridge_environment(
    *,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    macro_round: int,
    active: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    learner_output: Path,
    learner_result: Path,
    source_root: str,
    source_sha256: str,
    learner_config_sha256: str,
    learner_contract_sha256: str,
    learner_contract: Mapping[str, Any],
) -> dict[str, str]:
    cluster = learner_contract["cluster"]
    container = learner_contract["container"]
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
        runtime_container_sha256=str(container["sha256"]),
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
        "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot["path"]),
        "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": str(snapshot["file_sha256"]),
        "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": str(snapshot["snapshot_sha256"]),
        "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(learner_output),
        "ROUND_DAGGER_LEARNER_RESULT": str(learner_result),
        "ROUND_DAGGER_LEARNER_CODE_SHA256": source_sha256,
        "ROUND_DAGGER_LEARNER_CONFIG_SHA256": learner_config_sha256,
        "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": learner_contract_sha256,
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
        "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": str(container["sha256"]),
        "ROUND_DAGGER_REMOTE_HANDOFF_ROOT": str(handoff["local_root"]),
        "ROUND_DAGGER_REMOTE_POLL_SECONDS": str(handoff["poll_seconds"]),
        "ROUND_DAGGER_REMOTE_MAX_ATTEMPTS": str(handoff["maximum_attempts"]),
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
    expected_bridge_sha256: str,
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
    _sha(expected_bridge_sha256, "bridge SHA-256")
    if not sealed_source_root.startswith("/") or any(
        token in sealed_source_root for token in ("\n", "\r", ",", ":", "/../")
    ):
        raise LearnerRecoveryError("sealed source root is unsafe")

    campaign_contract_path = _absolute(campaign_contract_path, "campaign contract")
    learner_contract_path = _absolute(learner_contract_path, "learner contract")
    for path, expected, label in (
        (
            campaign_contract_path,
            _sha(expected_campaign_contract_sha256, "campaign contract SHA-256"),
            "campaign contract",
        ),
        (
            learner_contract_path,
            _sha(expected_learner_contract_sha256, "learner contract SHA-256"),
            "learner contract",
        ),
        (BRIDGE, expected_bridge_sha256, "learner bridge"),
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
    learner_contract = production_launch.load_learner_contract(learner_contract_path)
    campaign = state.load_campaign(root)
    current = state.load_state(root)
    contract = state.load_round_contract(root, macro_round)
    validate_production_topology(campaign, contract)
    if (
        artifacts.canonical_json_sha256(campaign) != campaign_sha
        or current["campaign_sha256"] != campaign_sha
        or campaign_contract["campaign_id"] != campaign["campaign_id"]
        or expected_learner_contract_sha256 != campaign["policy_config_sha256"]
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
    learner_output = round_dir / "learner"
    learner_result_path = learner_output / f"learner_result_round_{macro_round:06d}.json"
    candidate_dir = root / "policies" / f"policy_{active['version'] + 1:06d}"
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
    )
    config_sha = artifacts.canonical_json_sha256(config_document)
    environment = _bridge_environment(
        campaign_root=root,
        campaign=campaign,
        macro_round=macro_round,
        active=active,
        snapshot=snapshot,
        learner_output=learner_output,
        learner_result=learner_result_path,
        source_root=sealed_source_root,
        source_sha256=source_sha,
        learner_config_sha256=learner_config_sha,
        learner_contract_sha256=expected_learner_contract_sha256,
        learner_contract=learner_contract,
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
        "learner_config_sha256": learner_config_sha,
        "campaign_contract_sha256": expected_campaign_contract_sha256,
        "learner_contract_sha256": expected_learner_contract_sha256,
        "bridge_sha256": expected_bridge_sha256,
        "orchestration_config_path": config_path.relative_to(root).as_posix(),
        "orchestration_config_sha256": config_sha,
        "bridge_environment_path": environment_path.relative_to(root).as_posix(),
        "bridge_environment_sha256": environment_sha,
        "learner_result_path": learner_result_path.relative_to(root).as_posix(),
        "candidate_manifest_path": (
            candidate_dir / "candidate.json"
        ).relative_to(root).as_posix(),
    }
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
    backend = ProductionBackend(configuration)

    if not learner_result_path.exists():
        (bridge_runner or _default_bridge_runner)(
            BRIDGE, environment, root, 172800
        )
    after_bridge = state.load_state(root)
    if after_bridge != current:
        raise LearnerRecoveryError("coordinator changed while learner bridge was running")
    learner_result = backend.load_learner_result(
        campaign=campaign, snapshot=snapshot
    )
    candidate = backend.stage_candidate(learner_result, snapshot)
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
