from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from round_dagger.artifacts import publish_immutable_json, sha256_file
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from scripts.round_dagger import learner_completion_attestation as completion
from scripts.round_dagger import remote_learner_success
from tests.policy_bundle_helpers import (
    HOMIE_CONTROLLER_SHA256,
    complete_runtime_policy_sha256,
    publish_policy_bundle,
)


class LearnerCompletionAttestationTest(unittest.TestCase):
    @staticmethod
    def _write_telemetry(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, object]:
        output = root / "learner"
        output.mkdir()
        campaign = "completion-crash-window"
        macro_round = 0
        code_sha = "1" * 64
        parent_sha = "2" * 64
        snapshot_file_sha = "3" * 64
        snapshot_sha = "4" * 64
        candidate_policy_state = "5" * 64
        candidate_runtime = complete_runtime_policy_sha256(
            candidate_policy_state
        )
        parent_runtime = "6" * 64
        container_sha = "7" * 64
        contract_sha = "8" * 64

        bundle = {
            "schema": completion.BUNDLE_SCHEMA,
            "campaign_id": campaign,
            "macro_round": macro_round,
            "model_lineage_sha256": parent_sha,
            "code_sha256": code_sha,
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_std": [0.229, 0.224, 0.225],
            "rgb_input_key": "vision_obs",
            "rgb_layout": "NHWC",
            "output_key": "actions",
            "observation_key_map": {},
            "train_action_slice": [0, 11],
            "batch_size": 8,
            "max_steps": 1200,
            "save_every": 5,
            "learning_rate": 2.5e-5,
            "weight_decay": 0.01,
            "seed": 17,
            "partition_seed": 23,
            "initial_checkpoint_sha256": parent_sha,
            "model_config": {"runtime_policy_sha256": parent_runtime},
        }
        bundle_path = output / "learner_bundle_round_000000.json"
        config_sha = publish_immutable_json(bundle_path, bundle)
        identity_sha = learner_identity_sha256(
            campaign_id=campaign,
            macro_round=macro_round,
            learner_code_sha256=code_sha,
            learner_config_sha256=config_sha,
            replay_snapshot_file_sha256=snapshot_file_sha,
            replay_snapshot_sha256=snapshot_sha,
            parent_checkpoint_sha256=parent_sha,
            parent_runtime_policy_sha256=parent_runtime,
            runtime_container_sha256=container_sha,
            learner_contract_sha256=contract_sha,
        )
        run_id = learner_wandb_run_id(
            campaign_id=campaign,
            macro_round=macro_round,
            learner_identity_sha256=identity_sha,
        )
        restart = output / "restart.pt"
        torch.save(
            {
                "schema": completion.CHECKPOINT_SCHEMA,
                "campaign_id": campaign,
                "macro_round": macro_round,
                "model_lineage_sha256": parent_sha,
                "code_sha256": code_sha,
                "learner_config_sha256": config_sha,
                "learner_identity_sha256": identity_sha,
                "runtime_container_sha256": container_sha,
                "learner_contract_sha256": contract_sha,
                "replay_snapshot_sha256": snapshot_sha,
                "replay_snapshot_file_sha256": snapshot_file_sha,
                "completed_steps": 1200,
                "world_size": 8,
                "homie_controller_sha256": HOMIE_CONTROLLER_SHA256,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "rng_states": [None] * 8,
            },
            restart,
        )
        candidate = output / "candidate_round_000000.pt"
        candidate.write_bytes(restart.read_bytes())
        spec, manifest = publish_policy_bundle(
            candidate, candidate_policy_state
        )
        telemetry = output / "telemetry.jsonl"
        telemetry.write_text("sealed telemetry\n", encoding="utf-8")
        readback = output / "wandb_learner_readback_round_000000.json"
        publish_immutable_json(
            readback,
            {
                "schema": completion.READBACK_SCHEMA,
                "status": "PASS",
                "account_email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "project": "roborender-round-dagger",
                "run_id": run_id,
                "run_state": "finished",
                "campaign_id": campaign,
                "macro_round": macro_round,
                "learner_code_sha256": code_sha,
                "learner_config_sha256": config_sha,
                "learner_identity_sha256": identity_sha,
                "model_lineage_sha256": parent_sha,
                "parent_runtime_policy_sha256": parent_runtime,
                "runtime_container_sha256": container_sha,
                "learner_contract_sha256": contract_sha,
                "replay_snapshot_file_sha256": snapshot_file_sha,
                "replay_snapshot_sha256": snapshot_sha,
                "verified_optimizer_steps": 1200,
                "first_optimizer_step": 1,
                "last_optimizer_step": 1200,
            },
        )
        result = output / "learner_result_round_000000.json"
        result_doc = {
            "schema": completion.RESULT_SCHEMA,
            "campaign_id": campaign,
            "macro_round": macro_round,
            "checkpoint_path": candidate.name,
            "checkpoint_sha256": sha256_file(candidate),
            "runtime_policy_sha256": candidate_runtime,
            "policy_spec_path": spec.name,
            "policy_spec_sha256": sha256_file(spec),
            "policy_bundle_manifest_path": manifest.name,
            "policy_bundle_manifest_sha256": sha256_file(manifest),
            "model_lineage_sha256": parent_sha,
            "replay_snapshot_sha256": snapshot_sha,
            "code_sha256": code_sha,
            "learner_config_sha256": config_sha,
            "optimizer_steps": 1200,
            "initial_loss": 1.0,
            "final_loss": 0.5,
        }
        publish_immutable_json(result, result_doc)
        artifacts = {
            "bundle": {"path": bundle_path.name, "sha256": sha256_file(bundle_path)},
            "result": {"path": result.name, "sha256": sha256_file(result)},
            "restart": {"path": restart.name, "sha256": sha256_file(restart)},
            "candidate": {"path": candidate.name, "sha256": sha256_file(candidate)},
            "policy_spec": {"path": spec.name, "sha256": sha256_file(spec)},
            "policy_manifest": {"path": manifest.name, "sha256": sha256_file(manifest)},
            "telemetry": {"path": telemetry.name, "sha256": sha256_file(telemetry)},
            "wandb_readback": {"path": readback.name, "sha256": sha256_file(readback)},
        }
        attestation = output / "learner_completion_round_000000.json"
        publish_immutable_json(
            attestation,
            {
                "schema": completion.SCHEMA,
                "status": "PASS",
                "campaign_id": campaign,
                "macro_round": macro_round,
                "code_sha256": code_sha,
                "learner_config_sha256": config_sha,
                "learner_contract_sha256": contract_sha,
                "learner_identity_sha256": identity_sha,
                "model_lineage_sha256": parent_sha,
                "parent_runtime_policy_sha256": parent_runtime,
                "runtime_container_sha256": container_sha,
                "runtime_policy_sha256": candidate_runtime,
                "replay_snapshot_file_sha256": snapshot_file_sha,
                "replay_snapshot_sha256": snapshot_sha,
                "world_size": 8,
                "batch_size_per_rank": 8,
                "optimizer_steps": 1200,
                "save_every": 5,
                "artifacts": artifacts,
                "wandb": {
                    "entity": "nvidia",
                    "project": "roborender-round-dagger",
                    "run_id": run_id,
                    "run_state": "finished",
                    "verified_optimizer_steps": 1200,
                },
            },
        )
        return {
            "output": output,
            "attestation": attestation,
            "result": result,
            "restart": restart,
            "campaign_id": campaign,
            "macro_round": macro_round,
            "code_sha256": code_sha,
            "config_sha256": config_sha,
            "parent_sha256": parent_sha,
            "snapshot_file_sha256": snapshot_file_sha,
            "snapshot_sha256": snapshot_sha,
            "runtime_container_sha256": container_sha,
            "learner_contract_sha256": contract_sha,
            "learner_identity_sha256": identity_sha,
        }

    def test_heldout_schedule_covers_zero_each_100_steps_and_final(self) -> None:
        self.assertEqual(
            completion._evaluation_steps(1200, 100),
            list(range(0, 1201, 100)),
        )
        self.assertEqual(
            completion._evaluation_steps(250, 100),
            [0, 100, 200, 250],
        )

    def test_heldout_telemetry_requires_every_1200_step_frontier(self) -> None:
        learned_actions = [0, 1]
        rows = []
        for step in completion._evaluation_steps(1200, 100):
            rows.append(
                {
                    "schema": completion.HELDOUT_TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": step,
                    "heldout_snapshot_file_sha256": "a" * 64,
                    "heldout_snapshot_sha256": "b" * 64,
                    "world_size": 8,
                    "heldout_transitions_rank": 2,
                    "heldout_transitions_per_rank": [2] * 8,
                    "heldout_transitions_global": 16,
                    "learned_action_indices": learned_actions,
                    "action_mse": 0.25,
                    "action_mae": 0.4,
                    "action_rmse": 0.5,
                    "action_cosine": 0.75,
                    "per_action": {
                        str(index): {
                            "mse": 0.25,
                            "mae": 0.4,
                            "rmse": 0.5,
                            "cosine": 0.75,
                        }
                        for index in learned_actions
                    },
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heldout.jsonl"
            self._write_telemetry(path, rows)
            observed = completion._heldout_telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                snapshot_file_sha256="a" * 64,
                snapshot_sha256="b" * 64,
                optimizer_steps=1200,
                eval_interval=100,
                learned_action_indices=learned_actions,
            )
            self.assertEqual(sorted(observed), list(range(0, 1201, 100)))

            self._write_telemetry(path, rows[:-1])
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "frontier coverage is incomplete",
            ):
                completion._heldout_telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    optimizer_steps=1200,
                    eval_interval=100,
                    learned_action_indices=learned_actions,
                )

    def test_wandb_readback_accepts_step_zero_heldout_quality_row(self) -> None:
        training = {
            1: {
                "loss": 0.5,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
            }
        }
        heldout_record = {
            "action_mse": 0.25,
            "action_mae": 0.4,
            "action_rmse": 0.5,
            "action_cosine": 0.75,
            "heldout_transitions_global": 16,
            "per_action": {
                "0": {"mse": 0.25, "mae": 0.4, "rmse": 0.5, "cosine": 0.75}
            },
        }
        config = {
            "campaign_id": "campaign",
            "macro_round": 0,
            "learner_code_sha256": "1" * 64,
            "learner_config_sha256": "2" * 64,
            "learner_identity_sha256": "3" * 64,
            "model_lineage_sha256": "4" * 64,
            "parent_runtime_policy_sha256": "5" * 64,
            "runtime_container_sha256": "6" * 64,
            "learner_contract_sha256": "7" * 64,
            "replay_snapshot_file_sha256": "8" * 64,
            "replay_snapshot_sha256": "9" * 64,
            "heldout_snapshot_file_sha256": "a" * 64,
            "heldout_snapshot_sha256": "b" * 64,
            "heldout_eval_interval": 100,
        }
        quality_row = {
            "_step": 0,
            "quality/heldout_action_mse": 0.25,
            "quality/heldout_action_mae": 0.4,
            "quality/heldout_action_rmse": 0.5,
            "quality/heldout_action_cosine": 0.75,
            "quality/heldout_transitions": 16,
            "quality/action_0/mse": 0.25,
            "quality/action_0/mae": 0.4,
            "quality/action_0/rmse": 0.5,
            "quality/action_0/cosine": 0.75,
        }
        training_row = {
            "schema": completion.TELEMETRY_SCHEMA,
            "campaign_id": "campaign",
            "macro_round": 0,
            "optimizer_step": 1,
            "loss": 0.5,
            "world_size": 8,
            "replay_transitions_rank": 1,
            "replay_transitions_per_rank": [1] * 8,
            "replay_transitions_global": 8,
        }

        class FakeRun:
            state = "finished"

            def __init__(self) -> None:
                self.config = config

            @staticmethod
            def scan_history(*, page_size: int):
                self.assertEqual(page_size, 500)
                return [quality_row, training_row]

        fake_wandb = SimpleNamespace(
            Api=lambda timeout: SimpleNamespace(run=lambda path: FakeRun())
        )
        identity = SimpleNamespace(
            email="ldu@nvidia.com",
            entity="nvidia",
            authorization_scope="team",
            team_member=True,
            entity_is_team=True,
            personal=False,
        )
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ):
            readback = completion._wandb_readback(
                telemetry=training,
                campaign_id="campaign",
                macro_round=0,
                code_sha256="1" * 64,
                config_sha256="2" * 64,
                learner_identity_sha256="3" * 64,
                parent_sha256="4" * 64,
                parent_runtime_policy_sha256="5" * 64,
                runtime_container_sha256="6" * 64,
                learner_contract_sha256="7" * 64,
                snapshot_file_sha256="8" * 64,
                snapshot_sha256="9" * 64,
                entity="nvidia",
                project="project",
                run_id="run",
                expected_email="ldu@nvidia.com",
                timeout_seconds=1,
                heldout_telemetry={0: heldout_record},
                heldout_snapshot_file_sha256="a" * 64,
                heldout_snapshot_sha256="b" * 64,
                heldout_eval_interval=100,
            )
        self.assertEqual(readback["heldout_evaluation"]["verified_steps"], [0])

    @staticmethod
    def _verify_args(fixture: dict[str, object]) -> dict[str, object]:
        return {
            key: fixture[key]
            for key in (
                "campaign_id",
                "macro_round",
                "code_sha256",
                "config_sha256",
                "parent_sha256",
                "snapshot_file_sha256",
                "snapshot_sha256",
                "runtime_container_sha256",
                "learner_contract_sha256",
                "learner_identity_sha256",
            )
        }

    def test_exact_crash_window_is_adopted_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            verified = completion.verify_attestation(
                fixture["attestation"],
                result=fixture["result"],
                **self._verify_args(fixture),
            )
            self.assertEqual(verified["status"], "PASS")
            marker = Path(temporary) / "job/learner_success.json"
            first = remote_learner_success.publish(
                marker,
                fixture["result"],
                fixture["attestation"],
                campaign_id=fixture["campaign_id"],
                macro_round=fixture["macro_round"],
                learner_code_sha256=fixture["code_sha256"],
                learner_config_sha256=fixture["config_sha256"],
                learner_identity_sha256=fixture["learner_identity_sha256"],
            )
            self.assertEqual(
                remote_learner_success.verify(
                    marker,
                    fixture["result"],
                    fixture["attestation"],
                    campaign_id=fixture["campaign_id"],
                    macro_round=fixture["macro_round"],
                    learner_code_sha256=fixture["code_sha256"],
                    learner_config_sha256=fixture["config_sha256"],
                    learner_identity_sha256=fixture[
                        "learner_identity_sha256"
                    ],
                ),
                first,
            )

    def test_restart_or_readback_tamper_is_rejected(self) -> None:
        for artifact in ("restart", "wandb_learner_readback_round_000000.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary))
                path = (
                    fixture["restart"]
                    if artifact == "restart"
                    else fixture["output"] / artifact
                )
                path.chmod(0o644)
                path.write_bytes(path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(
                    completion.CompletionAttestationError, "changed after completion"
                ):
                    completion.verify_attestation(
                        fixture["attestation"],
                        result=fixture["result"],
                        **self._verify_args(fixture),
                    )

    def test_wrong_snapshot_or_missing_attestation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            arguments = self._verify_args(fixture)
            arguments["snapshot_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "lineage mismatch"
            ):
                completion.verify_attestation(
                    fixture["attestation"], result=fixture["result"], **arguments
                )
            fixture["attestation"].unlink()
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "missing or unsafe"
            ):
                completion.verify_attestation(
                    fixture["attestation"],
                    result=fixture["result"],
                    **self._verify_args(fixture),
                )

    def test_container_contract_or_identity_argument_tamper_is_rejected(self) -> None:
        for field in (
            "runtime_container_sha256",
            "learner_contract_sha256",
            "learner_identity_sha256",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary))
                arguments = self._verify_args(fixture)
                arguments[field] = "f" * 64
                with self.assertRaisesRegex(
                    completion.CompletionAttestationError, "lineage mismatch"
                ):
                    completion.verify_attestation(
                        fixture["attestation"],
                        result=fixture["result"],
                        **arguments,
                    )

    def test_telemetry_binds_per_rank_counts_to_snapshot_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            rows = [
                {
                    "schema": completion.TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": step,
                    "loss": 1.0 / step,
                    "replay_transitions_rank": 1,
                    "replay_transitions_per_rank": [1] * 8,
                    "replay_transitions_global": 8,
                    "world_size": 8,
                }
                for step in (1, 2)
            ]
            self._write_telemetry(path, rows)
            observed = completion._telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                optimizer_steps=2,
                replay_transitions_global=8,
            )
            self.assertEqual(set(observed), {1, 2})

            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "contract mismatch"
            ):
                completion._telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    optimizer_steps=2,
                    replay_transitions_global=9,
                )

    def test_telemetry_rejects_partition_change_between_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            rows = [
                {
                    "schema": completion.TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": 1,
                    "loss": 1.0,
                    "replay_transitions_rank": 9,
                    "replay_transitions_per_rank": [9, 1, 1, 1, 1, 1, 1, 1],
                    "replay_transitions_global": 16,
                    "world_size": 8,
                },
                {
                    "schema": completion.TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": 2,
                    "loss": 0.5,
                    "replay_transitions_rank": 8,
                    "replay_transitions_per_rank": [8, 2, 1, 1, 1, 1, 1, 1],
                    "replay_transitions_global": 16,
                    "world_size": 8,
                },
            ]
            self._write_telemetry(path, rows)
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "partition changed between steps",
            ):
                completion._telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    optimizer_steps=2,
                    replay_transitions_global=16,
                )


if __name__ == "__main__":
    unittest.main()
