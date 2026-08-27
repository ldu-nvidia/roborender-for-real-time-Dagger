#!/usr/bin/env python3
"""Run a small real-artifact DAgger macro-round on CPU.

This is the bridge between unit tests and the hardware rollout ladder.  It uses
the same immutable cohort, transfer, cumulative replay, restartable learner,
candidate, evaluation, and promotion code as production, but only four logical
environments with eight transitions each and a tiny policy.  It intentionally
does not import Isaac, cameras, or RoboRender.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    canonical_json_sha256,
    create_cohort,
    publish_immutable_json,
    publish_rollout_shard,
    seal_cohort,
    sha256_file,
    verify_cohort,
)
from round_dagger.handoff import transfer_local  # noqa: E402
from round_dagger.actor import runtime_state_dict_sha256  # noqa: E402
from round_dagger.gr00t_policy import (  # noqa: E402
    BASE_MODULE_TARGET,
    POLICY_TARGET,
    build_portable_policy_spec,
)
from round_dagger.learner import (  # noqa: E402
    BUNDLE_SCHEMA,
    JsonlTelemetry,
    LearnerBundleConfig,
    load_replay_partition,
    publish_learner_result,
    supervised_action_loss,
    train_from_replay,
)
from round_dagger.promotion import (  # noqa: E402
    load_active_policy,
    promote_candidate,
    stage_bootstrap_policy,
    stage_candidate,
    write_evaluation,
)
from round_dagger.replay import ReplayLedger  # noqa: E402
from round_dagger.state import (  # noqa: E402
    initialize_campaign,
    load_state,
    open_macro_round,
    seal_collection,
    transition,
)


CAMPAIGN_ID = "mini-round-smoke"
COHORTS = 4
STEPS = 8
VIEW_SHAPE = (6, 2, 3)
FULL_ACTION_DIM = 2
HOMIE_CONTROLLER_SHA256 = {"walk": "4" * 64, "stand": "5" * 64}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _portable_spec(checkpoint_sha: str, policy_state_sha: str) -> dict:
    """Return a production-shaped portable spec for control-plane smoke gates."""

    policy_type = type("VisionActorObjPred", (), {})
    policy_type.__module__ = (
        "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred"
    )
    policy = policy_type()
    policy.num_actions = 11
    policy.obj_pred_mlp = types.SimpleNamespace(output_dim=3)
    module = {
        "_target_": BASE_MODULE_TARGET,
        "process_output_dim": True,
    }
    actor_config = {
        "_target_": POLICY_TARGET,
        "running_mean_std": False,
        "backbone": {
            "vision_module": {
                **module,
                "module_config_dict": {
                    "input_dim": ["vision_obs"],
                    "output_dim": ["vision_feature_dim"],
                    "layer_config": {
                        "type": "ResNet",
                        "resnet_type": "resnet18",
                        "pretrained": False,
                        "trainable": True,
                    },
                },
            },
            "mlp_module": {
                **module,
                "module_config_dict": {
                    "input_dim": ["actor_obs", "vision_feature_dim"],
                    "output_dim": [-1],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [512, 256, 128],
                        "activation": "SiLU",
                    },
                },
            },
            "obj_pred_mlp": {
                **module,
                "module_config_dict": {
                    "input_dim": ["vision_feature_dim"],
                    "output_dim": [3],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [256, 128, 64],
                        "activation": "SiLU",
                    },
                },
            },
        },
    }
    return build_portable_policy_spec(
        policy=policy,
        actor_config=actor_config,
        algo_config={
            "init_noise_std": 0.3,
            "freeze_noise_std": True,
            "clamp_noise_std": False,
        },
        module_dim_dict={"vision_feature_dim": 128},
        env_config=types.SimpleNamespace(
            robot=types.SimpleNamespace(
                algo_obs_dim_dict={
                    "actor_obs": 113,
                    "vision_obs": 324 * 192 * 3,
                },
                actions_dim=31,
            )
        ),
        policy_views=["head_color", "hand_left", "hand_right"],
        policy_view_layout="vertical",
        per_view_resolution=[108, 192],
        rgb_mean=[0.5, 0.5, 0.5],
        rgb_std=[0.25, 0.25, 0.25],
        homie_action_dim=15,
        replay_action_dim=26,
        checkpoint_sha256=checkpoint_sha,
        policy_state_sha256=policy_state_sha,
        homie_controller_sha256=HOMIE_CONTROLLER_SHA256,
    )


class TinyPolicy(nn.Module):
    """Small observation-to-action model used only by this systems smoke."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, observations):
        rgb = observations["vision_obs"].mean(dim=(1, 2))
        actor = observations["actor_obs"].float()
        return {"actions": self.linear(torch.cat((rgb, actor), dim=-1))}


def _load_bootstrap(model: nn.Module, path: Path) -> None:
    document = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(document["policy_state_dict"], strict=True)


def _rollout_arrays(global_env_id: int) -> dict[str, np.ndarray]:
    rgb = np.empty((STEPS, 1, *VIEW_SHAPE), dtype=np.uint8)
    actor = np.empty((STEPS, 1, 1), dtype=np.float32)
    expert = np.empty((STEPS, 1, FULL_ACTION_DIM), dtype=np.float32)
    student = np.zeros_like(expert)
    for step in range(STEPS):
        channels = np.asarray(
            [
                (17 * step + 31 * global_env_id) % 256,
                (43 * step + 11 * global_env_id + 7) % 256,
                (71 * step + 19 * global_env_id + 3) % 256,
            ],
            dtype=np.uint8,
        )
        rgb[step, 0] = channels
        actor[step, 0, 0] = np.float32((step - global_env_id) / 7.0)
        normalized = (channels.astype(np.float32) / 255.0 - 0.5) / 0.25
        expert[step, 0, 0] = (
            normalized @ np.asarray([0.35, -0.2, 0.5], np.float32)
            + 0.75 * actor[step, 0, 0]
        )
        # Represents a fixed, non-trainable action suffix.
        expert[step, 0, 1] = np.float32(1000.0 + step)
    expert_mask = ((np.arange(STEPS) + global_env_id) % 2 == 0)[:, None]
    behavior = np.where(expert_mask[..., None], expert, student)
    dones = np.zeros((STEPS, 1), dtype=np.bool_)
    dones[-1, 0] = True
    return {
        "policy_rgb": rgb,
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": behavior,
        "expert_mask": expert_mask.astype(np.bool_),
        "dones": dones,
        "observation_is_raytraced": np.zeros((STEPS, 1), dtype=np.bool_),
        "global_env_ids": np.asarray([global_env_id], dtype=np.int64),
        "episode_ids": np.full((STEPS, 1), global_env_id, dtype=np.int64),
        "episode_steps": np.arange(STEPS, dtype=np.int64)[:, None],
        "obs__actor_obs": actor,
    }


def _full_loss(
    model: nn.Module,
    *,
    snapshot: Path,
    snapshot_file_sha256: str,
    config: LearnerBundleConfig,
) -> float:
    partition = load_replay_partition(
        snapshot,
        config,
        expected_snapshot_file_sha256=snapshot_file_sha256,
    )
    observations, targets = partition.batch(
        list(range(partition.transition_count)), device="cpu"
    )
    model.eval()
    with torch.no_grad():
        prediction = model(observations)["actions"]
        return float(
            supervised_action_loss(
                prediction, targets, config.train_action_slice
            ).item()
        )


def _candidate_summary(candidate: dict) -> dict:
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


def run(output_root: Path) -> dict:
    output_root = Path(os.path.abspath(output_root.expanduser()))
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"refusing to reuse smoke output: {output_root}")
    output_root.mkdir(parents=True)
    campaign_root = output_root / "campaign"
    actor_root = output_root / "actor"

    torch.manual_seed(1234)
    bootstrap_model = TinyPolicy()
    bootstrap_source = output_root / "bootstrap_source.pt"
    torch.save({"policy_state_dict": bootstrap_model.state_dict()}, bootstrap_source)
    bootstrap_policy_state_sha = runtime_state_dict_sha256(
        bootstrap_model.state_dict()
    )
    bootstrap_source_sha = sha256_file(bootstrap_source)
    bootstrap_spec = _portable_spec(
        bootstrap_source_sha, bootstrap_policy_state_sha
    )
    bootstrap_runtime_sha = bootstrap_spec["runtime_policy_sha256"]
    bootstrap_spec_path = Path(f"{bootstrap_source}.policy_spec.json")
    bootstrap_spec_sha = publish_immutable_json(
        bootstrap_spec_path, bootstrap_spec
    )
    publish_immutable_json(
        Path(f"{bootstrap_source}.manifest.json"),
        {
            "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
            "checkpoint_path": bootstrap_source.name,
            "checkpoint_sha256": bootstrap_source_sha,
            "runtime_policy_sha256": bootstrap_runtime_sha,
            "policy_spec_path": bootstrap_spec_path.name,
            "policy_spec_sha256": bootstrap_spec_sha,
        },
    )
    bootstrap = stage_bootstrap_policy(
        campaign_root,
        checkpoint_source=bootstrap_source,
        code_sha256=_sha("mini-code-v1"),
        policy_config_sha256=_sha("mini-policy-config-v1"),
    )
    initialize_campaign(
        campaign_root,
        campaign_id=CAMPAIGN_ID,
        bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
        bootstrap_policy_path=bootstrap["checkpoint_path"],
        target_environments=COHORTS,
        active_environments=1,
        steps_per_environment=STEPS,
        view_names=("ext1", "ext2", "wrist"),
        policy_image_shape=VIEW_SHAPE,
        action_dim=FULL_ACTION_DIM,
        teacher_sha256=_sha("mini-teacher-v1"),
        renderer_sha256=_sha("mini-renderer-v1"),
        camera_mapping_sha256=_sha("mini-camera-v1"),
        training_config_sha256=_sha("mini-training-v1"),
        beta_schedule={
            "kind": "constant",
            "start": 0.5,
            "end": 0.5,
            "decay_rounds": 1,
            "rate": 0.99,
        },
    )
    contract = open_macro_round(
        campaign_root, expected_revision=load_state(campaign_root)["revision"]
    )
    runtime_sha = contract["policy_runtime_sha256"]
    accepted_manifests: list[Path] = []
    for cohort_index in range(COHORTS):
        source = actor_root / f"cohort_{cohort_index:02d}"
        common = {
            "campaign_id": CAMPAIGN_ID,
            "macro_round": 0,
            "cohort_index": cohort_index,
            "purpose": "train",
            "policy_sha256": contract["policy_checkpoint_sha256"],
            "runtime_policy_sha256": runtime_sha,
            "teacher_sha256": contract["teacher_sha256"],
            "renderer_sha256": contract["renderer_sha256"],
            "beta_sha256": contract["beta_sha256"],
            "config_sha256": contract["training_config_sha256"],
        }
        create_cohort(
            source,
            expected_ranks=[0],
            expected_global_env_ids=[cohort_index],
            expected_transitions=STEPS,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": list(VIEW_SHAPE),
                "views": ["ext1", "ext2", "wrist"],
            },
            action_contract={"dtype": "float32", "shape": [FULL_ACTION_DIM]},
            **common,
        )
        publish_rollout_shard(
            source,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_rollout_arrays(cohort_index),
            **common,
        )
        seal_cohort(source)
        accepted = (
            campaign_root
            / "rounds"
            / "macro_round_000000"
            / "cohorts"
            / f"cohort_{cohort_index:02d}"
        )
        transfer_local(source, accepted)
        report = verify_cohort(accepted)
        if report["transition_count"] != STEPS:
            raise RuntimeError("transferred cohort transition count changed")
        accepted_manifests.append(accepted / "seal.json")

    collection = seal_collection(
        campaign_root, macro_round=0, cohort_manifests=accepted_manifests
    )
    transition(
        campaign_root,
        expected_phase="COLLECTING",
        next_phase="COLLECTED",
        macro_round=0,
        evidence_path=collection["manifest_path"],
        evidence_sha256=collection["manifest_sha256"],
    )
    snapshot_path = campaign_root / "replay" / "snapshot_000000.json"
    with ReplayLedger(campaign_root / "replay" / "replay.sqlite3") as ledger:
        for manifest in accepted_manifests:
            ledger.ingest_cohort(manifest)
        snapshot = ledger.build_snapshot(
            CAMPAIGN_ID,
            through_macro_round=0,
            output=snapshot_path,
            expected_train_cohorts=COHORTS,
        )
    transition(
        campaign_root,
        expected_phase="COLLECTED",
        next_phase="INGESTED",
        macro_round=0,
        evidence_path=snapshot_path,
        evidence_sha256=snapshot["file_sha256"],
    )

    config = LearnerBundleConfig.from_mapping(
        {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "macro_round": 0,
            "model_lineage_sha256": bootstrap["checkpoint_sha256"],
            "code_sha256": _sha("mini-learner-code-v1"),
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.25, 0.25, 0.25],
            "rgb_input_key": "vision_obs",
            "rgb_layout": "NHWC",
            "output_key": "actions",
            "train_action_slice": [0, 1],
            "batch_size": 8,
            "learning_rate": 0.025,
            "weight_decay": 0.0,
            "seed": 17,
            "partition_seed": 23,
            "initial_checkpoint_sha256": bootstrap["checkpoint_sha256"],
            "model_config": bootstrap_spec,
        }
    )

    parent = TinyPolicy()
    _load_bootstrap(parent, campaign_root / bootstrap["checkpoint_path"])
    parent_loss = _full_loss(
        parent,
        snapshot=snapshot_path,
        snapshot_file_sha256=snapshot["file_sha256"],
        config=config,
    )

    control = TinyPolicy()
    _load_bootstrap(control, campaign_root / bootstrap["checkpoint_path"])
    control_result = train_from_replay(
        control,
        snapshot_path=snapshot_path,
        config=config,
        output_dir=output_root / "learner_control",
        max_steps=40,
        save_every=7,
        expected_snapshot_file_sha256=snapshot["file_sha256"],
    )

    interrupted = TinyPolicy()
    _load_bootstrap(interrupted, campaign_root / bootstrap["checkpoint_path"])
    telemetry_path = output_root / "learner_resumed" / "telemetry.jsonl"
    telemetry = JsonlTelemetry(telemetry_path, enabled=True)
    first_stage = train_from_replay(
        interrupted,
        snapshot_path=snapshot_path,
        config=config,
        output_dir=output_root / "learner_resumed",
        max_steps=7,
        save_every=7,
        expected_snapshot_file_sha256=snapshot["file_sha256"],
        telemetry=telemetry,
    )
    telemetry.close()
    resumed = TinyPolicy()
    _load_bootstrap(resumed, campaign_root / bootstrap["checkpoint_path"])
    telemetry = JsonlTelemetry(telemetry_path, enabled=True)
    resumed_result = train_from_replay(
        resumed,
        snapshot_path=snapshot_path,
        config=config,
        output_dir=output_root / "learner_resumed",
        max_steps=40,
        save_every=7,
        resume_from=first_stage.restart_checkpoint,
        expected_snapshot_file_sha256=snapshot["file_sha256"],
        telemetry=telemetry,
    )
    telemetry.close()
    restart_exact = all(
        torch.equal(value, resumed.state_dict()[name])
        for name, value in control.state_dict().items()
    )
    if not restart_exact:
        raise RuntimeError("interrupted learner did not exactly match uninterrupted run")
    telemetry_steps = [
        json.loads(line)["optimizer_step"]
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    if telemetry_steps != list(range(1, 41)):
        raise RuntimeError("restart telemetry frontier is not exactly 1..40")
    candidate_loss = _full_loss(
        resumed,
        snapshot=snapshot_path,
        snapshot_file_sha256=snapshot["file_sha256"],
        config=config,
    )
    if not candidate_loss < parent_loss:
        raise RuntimeError("synthetic candidate did not improve supervised action loss")

    learner_report = publish_learner_result(
        output_dir=output_root / "learner_resumed",
        result=resumed_result,
        config=config,
    )
    candidate = stage_candidate(
        campaign_root,
        macro_round=0,
        checkpoint_source=Path(learner_report["manifest_path"]).parent
        / learner_report["checkpoint_path"],
        replay_snapshot_sha256=snapshot["file_sha256"],
        code_sha256=config.code_sha256,
        learner_config_sha256=config.sha256,
        optimizer_steps=resumed_result.completed_steps,
    )
    transition(
        campaign_root,
        expected_phase="INGESTED",
        next_phase="TRAINED",
        macro_round=0,
        evidence_path=candidate["manifest_path"],
        evidence_sha256=candidate["manifest_sha256"],
        candidate_policy=_candidate_summary(candidate),
    )
    evidence_bundle = (
        campaign_root
        / "rounds"
        / "macro_round_000000"
        / "evaluation"
        / "results"
        / "evaluation_bundle.json"
    )
    evidence_bundle_sha = publish_immutable_json(
        evidence_bundle,
        {
            "schema": "fast_roborender.round_dagger.mini_evaluation_bundle.v1",
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "parent_action_mse": parent_loss,
            "candidate_action_mse": candidate_loss,
            "restart_exact": restart_exact,
        },
    )
    evaluation = write_evaluation(
        campaign_root,
        candidate_manifest=candidate["manifest_path"],
        passed=restart_exact and candidate_loss < parent_loss,
        metrics={
            "parent_action_mse": parent_loss,
            "candidate_action_mse": candidate_loss,
            "relative_mse_reduction": (parent_loss - candidate_loss) / parent_loss,
            "restart_exact": float(restart_exact),
        },
        evaluator_config_sha256=_sha("mini-evaluator-v1"),
        evidence_bundle_path=evidence_bundle.relative_to(campaign_root).as_posix(),
        evidence_bundle_sha256=evidence_bundle_sha,
    )
    transition(
        campaign_root,
        expected_phase="TRAINED",
        next_phase="EVALUATED",
        macro_round=0,
        evidence_path=evaluation["manifest_path"],
        evidence_sha256=evaluation["evidence_sha256"],
    )
    promotion = promote_candidate(
        campaign_root,
        macro_round=0,
        candidate_manifest=candidate["manifest_path"],
        evaluation_manifest=evaluation["manifest_path"],
    )
    active = load_active_policy(campaign_root)
    final_state = load_state(campaign_root)
    if (
        final_state["phase"] != "READY"
        or final_state["next_macro_round"] != 1
        or active["version"] != 1
        or active["checkpoint_sha256"] != candidate["checkpoint_sha256"]
    ):
        raise RuntimeError("promotion did not atomically advance the active policy")

    report = {
        "schema": "fast_roborender.round_dagger.mini_smoke_report.v1",
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "macro_round": 0,
        "cohort_count": COHORTS,
        "logical_environments": COHORTS,
        "steps_per_environment": STEPS,
        "transitions": COHORTS * STEPS,
        "replay_snapshot_file_sha256": snapshot["file_sha256"],
        "replay_snapshot_sha256": snapshot["snapshot_sha256"],
        "parent_action_mse": parent_loss,
        "candidate_action_mse": candidate_loss,
        "restart_exact": restart_exact,
        "restart_step": first_stage.completed_steps,
        "optimizer_steps": resumed_result.completed_steps,
        "telemetry_steps": len(telemetry_steps),
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "promotion_sha256": promotion["promotion_sha256"],
        "active_policy_version": active["version"],
        "final_state_revision": final_state["revision"],
        "control_final_loss": control_result.final_loss,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    publish_immutable_json(output_root / "validation_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/data2/ldu/roborender_round_dagger_validation"
        )
        / f"mini_round_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run(arguments.output_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
