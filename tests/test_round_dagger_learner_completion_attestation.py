from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from round_dagger.artifacts import (
    canonical_json_sha256,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from round_dagger.learner import JsonlTelemetry, TELEMETRY_SCHEMA
from scripts.round_dagger import learner_completion_attestation as completion
from scripts.round_dagger import remote_learner_success
from tests.policy_bundle_helpers import (
    HOMIE_CONTROLLER_SHA256,
    complete_runtime_policy_sha256,
    publish_policy_bundle,
)


class LearnerCompletionAttestationTest(unittest.TestCase):
    def test_output_file_accepts_argparse_path_basename(self) -> None:
        """Publish parses --result as Path, while retaining basename confinement."""

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            result = output / "learner_result_round_000000.json"
            result.write_text("{}", encoding="utf-8")
            self.assertEqual(
                completion._output_file(output, Path(result.name), "learner result"),
                result,
            )
            for unsafe in (
                result,
                Path("nested") / result.name,
                Path("."),
                Path(".."),
            ):
                with self.assertRaises(completion.CompletionAttestationError):
                    completion._output_file(output, unsafe, "learner result")

    def test_publish_wandb_readback_call_matches_runtime_signature(self) -> None:
        """Prevent an untested publish/readback keyword interface regression."""

        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(completion.publish_attestation)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_wandb_readback"
        ]
        self.assertEqual(len(calls), 1)
        call_keywords = {
            keyword.arg for keyword in calls[0].keywords if keyword.arg is not None
        }
        signature = inspect.signature(completion._wandb_readback)
        accepted_keywords = set(signature.parameters)
        self.assertEqual(call_keywords - accepted_keywords, set())
        required_keywords = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        self.assertEqual(required_keywords - call_keywords, set())

        validation_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_readback"
        ]
        self.assertEqual(len(validation_calls), 1)
        validation_keywords = {
            keyword.arg: keyword.value
            for keyword in validation_calls[0].keywords
            if keyword.arg is not None
        }
        optimizer_steps = validation_keywords.get("optimizer_steps")
        self.assertIsInstance(optimizer_steps, ast.Name)
        self.assertEqual(optimizer_steps.id, "optimizer_steps")

    @staticmethod
    def _write_telemetry(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _probe_summary(
        *,
        campaign_id: str,
        macro_round: int,
        snapshot_file_sha256: str,
        snapshot_sha256: str,
        replay_counts: list[int],
        selected_counts: list[int],
        current_counts: list[int],
        historical_counts: list[int],
        partition_seed: int = 23,
    ) -> dict[str, object]:
        rank_digests = [f"{rank + 1:064x}" for rank in range(len(replay_counts))]
        base = {
            "policy": completion.TRAINING_QUALITY_PROBE_POLICY,
            "budget": completion.TRAINING_QUALITY_PROBE_BUDGET,
            "campaign_id": campaign_id,
            "macro_round": macro_round,
            "snapshot_file_sha256": snapshot_file_sha256,
            "snapshot_sha256": snapshot_sha256,
            "partition_seed": partition_seed,
            "world_size": len(replay_counts),
            "replay_transitions_per_rank": replay_counts,
            "replay_transitions_global": sum(replay_counts),
            "transitions_per_rank": selected_counts,
            "transitions_global": sum(selected_counts),
            "current_round_transitions_per_rank": current_counts,
            "current_round_transitions_global": sum(current_counts),
            "historical_transitions_per_rank": historical_counts,
            "historical_transitions_global": sum(historical_counts),
            "rank_selection_sha256s": rank_digests,
        }
        digest_payload = {
            "domain": completion.TRAINING_QUALITY_PROBE_DOMAIN,
            **{
                key: base[key]
                for key in sorted(
                    completion._TRAINING_QUALITY_PROBE_SUMMARY_KEYS
                    - {"selection_sha256"}
                )
            },
        }
        return {
            **base,
            "selection_sha256": canonical_json_sha256(digest_payload),
        }

    @staticmethod
    def _training_quality_row(
        *,
        probe: dict[str, object],
        optimizer_step: int,
        learned_action_indices: list[int],
        action_mse: float,
    ) -> dict[str, object]:
        return {
            "schema": completion.TRAINING_QUALITY_TELEMETRY_SCHEMA,
            "campaign_id": probe["campaign_id"],
            "macro_round": probe["macro_round"],
            "optimizer_step": optimizer_step,
            "training_snapshot_file_sha256": probe["snapshot_file_sha256"],
            "training_snapshot_sha256": probe["snapshot_sha256"],
            "training_probe_policy": probe["policy"],
            "training_probe_budget": probe["budget"],
            "training_probe_partition_seed": probe["partition_seed"],
            "training_probe_selection_sha256": probe["selection_sha256"],
            "training_probe_rank_selection_sha256s": probe[
                "rank_selection_sha256s"
            ],
            "training_probe_record_rank": 0,
            "training_probe_rank_selection_sha256": probe[
                "rank_selection_sha256s"
            ][0],
            "training_probe_transitions_rank": probe["transitions_per_rank"][0],
            "training_probe_transitions_per_rank": probe["transitions_per_rank"],
            "training_probe_transitions_global": probe["transitions_global"],
            "training_probe_current_round_transitions_rank": probe[
                "current_round_transitions_per_rank"
            ][0],
            "training_probe_current_round_transitions_per_rank": probe[
                "current_round_transitions_per_rank"
            ],
            "training_probe_current_round_transitions_global": probe[
                "current_round_transitions_global"
            ],
            "training_probe_historical_transitions_rank": probe[
                "historical_transitions_per_rank"
            ][0],
            "training_probe_historical_transitions_per_rank": probe[
                "historical_transitions_per_rank"
            ],
            "training_probe_historical_transitions_global": probe[
                "historical_transitions_global"
            ],
            "training_replay_transitions_rank": probe[
                "replay_transitions_per_rank"
            ][0],
            "training_replay_transitions_per_rank": probe[
                "replay_transitions_per_rank"
            ],
            "training_replay_transitions_global": probe[
                "replay_transitions_global"
            ],
            "world_size": probe["world_size"],
            "learned_action_indices": learned_action_indices,
            "action_mse": action_mse,
            "action_mae": 0.4,
            "action_rmse": math.sqrt(action_mse),
            "action_cosine": 0.75,
            "action_absolute_error_p90": 0.8,
            "action_l2_p90": 1.2,
            "per_action": {
                str(index): {
                    "mse": action_mse,
                    "mae": 0.4,
                    "rmse": math.sqrt(action_mse),
                    "cosine": 0.75,
                }
                for index in learned_action_indices
            },
            "per_action_absolute_error_p90": {
                str(index): 0.8 for index in learned_action_indices
            },
        }

    @staticmethod
    def _training_optimizer_aliases(
        record: dict[str, object],
    ) -> dict[str, object]:
        return completion._training_optimizer_wandb_fields(record)

    @staticmethod
    def _validation_ood_aliases(
        record: dict[str, object],
    ) -> dict[str, object]:
        return completion._validation_ood_wandb_fields(record)

    @staticmethod
    def _training_context(
        *,
        campaign_id: str,
        macro_round: int,
        config_sha256: str,
        parent_sha256: str,
        snapshot_file_sha256: str,
        snapshot_sha256: str,
        probe_selection_sha256: str,
        replay_transitions: int,
    ) -> dict[str, object]:
        return {
            "schema": completion.TRAINING_CONTEXT_TELEMETRY_SCHEMA,
            "campaign_id": campaign_id,
            "macro_round": macro_round,
            "learner_config_sha256": config_sha256,
            "model_lineage_sha256": parent_sha256,
            "replay_snapshot_file_sha256": snapshot_file_sha256,
            "replay_snapshot_sha256": snapshot_sha256,
            "training_probe_selection_sha256": probe_selection_sha256,
            "metrics": {
                "parent_rollout_action_mse": 0.25,
                "parent_rollout_action_mae": 0.4,
                "parent_rollout_action_rmse": 0.5,
                "parent_rollout_action_cosine": 0.75,
                "parent_rollout_action_l2_p90": 1.2,
                "expert_intervention_rate": 0.5,
                "behavior_expert_match_rate": 0.5,
                "expert_label_coverage": 1.0,
                "replay_transitions": replay_transitions,
                "replay_environments": min(8, replay_transitions),
                "done_rate": 0.125,
                "scheduled_beta": 0.5,
            },
        }

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
                "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
        training_quality = output / "training_quality_telemetry.jsonl"
        probe = self._probe_summary(
            campaign_id=campaign,
            macro_round=macro_round,
            snapshot_file_sha256=snapshot_file_sha,
            snapshot_sha256=snapshot_sha,
            replay_counts=[1] * 8,
            selected_counts=[1] * 8,
            current_counts=[1] * 8,
            historical_counts=[0] * 8,
        )
        training_quality_rows = []
        for step in completion._evaluation_steps(1200, 100):
            action_mse = 0.25 + step / 100000.0
            training_quality_rows.append(
                self._training_quality_row(
                    probe=probe,
                    optimizer_step=step,
                    learned_action_indices=list(range(11)),
                    action_mse=action_mse,
                )
            )
        self._write_telemetry(training_quality, training_quality_rows)
        training_context = output / "training_context.json"
        context_record = self._training_context(
            campaign_id=campaign,
            macro_round=macro_round,
            config_sha256=config_sha,
            parent_sha256=parent_sha,
            snapshot_file_sha256=snapshot_file_sha,
            snapshot_sha256=snapshot_sha,
            probe_selection_sha256=str(probe["selection_sha256"]),
            replay_transitions=8,
        )
        training_context.write_text(
            json.dumps(context_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
                "training_quality_evaluation": {
                    "snapshot_file_sha256": snapshot_file_sha,
                    "snapshot_sha256": snapshot_sha,
                    "probe_selection_sha256": probe["selection_sha256"],
                    "probe_transitions": probe["transitions_global"],
                    "verified_steps": completion._evaluation_steps(1200, 100),
                },
                "training_context": {
                    "file_sha256": sha256_file(training_context),
                    "metric_keys": sorted(completion.TRAINING_CONTEXT_METRIC_KEYS),
                    "verified_optimizer_step": 0,
                },
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
            "training_quality_telemetry": {
                "path": training_quality.name,
                "sha256": sha256_file(training_quality),
            },
            "training_context": {
                "path": training_context.name,
                "sha256": sha256_file(training_context),
            },
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
                "training_quality_evaluation": {
                    "snapshot_file_sha256": snapshot_file_sha,
                    "snapshot_sha256": snapshot_sha,
                    "eval_interval": 100,
                    "verified_steps": completion._evaluation_steps(1200, 100),
                    "probe": probe,
                },
                "training_context": {
                    "schema": completion.TRAINING_CONTEXT_TELEMETRY_SCHEMA,
                    "training_probe_selection_sha256": probe[
                        "selection_sha256"
                    ],
                    "metrics": context_record["metrics"],
                    "verified_wandb_step": 0,
                },
                "wandb": {
                    "entity": "nvidia",
                    "project": "roborender-round-dagger",
                    "run_id": run_id,
                    "run_state": "finished",
                    "verified_optimizer_steps": 1200,
                    "verified_training_quality_steps": completion._evaluation_steps(
                        1200, 100
                    ),
                    "training_quality_probe_selection_sha256": probe[
                        "selection_sha256"
                    ],
                    "training_context_verified_step": 0,
                    "training_context_metric_keys": sorted(
                        completion.TRAINING_CONTEXT_METRIC_KEYS
                    ),
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

    def test_completion_requires_candidate_runtime_to_differ_from_parent(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            completion.CompletionAttestationError,
            "bit-identical to its parent",
        ):
            completion._require_updated_runtime_policy("a" * 64, "a" * 64)
        self.assertEqual(
            completion._require_updated_runtime_policy("a" * 64, "b" * 64),
            "a" * 64,
        )

    def test_heldout_telemetry_requires_every_1200_step_frontier(self) -> None:
        learned_actions = [0, 1]
        rows = []
        for step in completion._evaluation_steps(1200, 100):
            action_mse = 0.25 + step / 100000.0
            rows.append(
                {
                    "schema": completion.HELDOUT_TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": step,
                    "eval_frontier": step,
                    "heldout_snapshot_file_sha256": "a" * 64,
                    "heldout_snapshot_sha256": "b" * 64,
                    "world_size": 8,
                    "heldout_transitions_rank": 2,
                    "heldout_transitions_per_rank": [2] * 8,
                    "heldout_transitions_global": 16,
                    "learned_action_indices": learned_actions,
                    "action_mse": action_mse,
                    "action_mae": 0.4,
                    "action_rmse": action_mse**0.5,
                    "action_cosine": 0.75,
                    "action_absolute_error_p90": 0.8,
                    "action_l2_p90": 1.2,
                    "action_mse_delta_from_step0": action_mse - 0.25,
                    "action_mse_relative_improvement_from_step0": (
                        0.25 - action_mse
                    )
                    / 0.25,
                    "per_action": {
                        str(index): {
                            "mse": action_mse,
                            "mae": 0.4,
                            "rmse": action_mse**0.5,
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

    def test_heldout_telemetry_rejects_bit_identical_quality_frontiers(
        self,
    ) -> None:
        learned_actions = [0]
        rows = []
        for step in completion._evaluation_steps(200, 100):
            rows.append(
                {
                    "schema": completion.HELDOUT_TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": step,
                    "eval_frontier": step,
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
                    "action_absolute_error_p90": 0.8,
                    "action_l2_p90": 1.2,
                    "action_mse_delta_from_step0": 0.0,
                    "action_mse_relative_improvement_from_step0": 0.0,
                    "per_action": {
                        "0": {
                            "mse": 0.25,
                            "mae": 0.4,
                            "rmse": 0.5,
                            "cosine": 0.75,
                        }
                    },
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "heldout.jsonl"
            self._write_telemetry(path, rows)
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "bit-identical",
            ):
                completion._heldout_telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    optimizer_steps=200,
                    eval_interval=100,
                    learned_action_indices=learned_actions,
                )

    def test_training_quality_requires_changing_complete_frontiers(self) -> None:
        probe = self._probe_summary(
            campaign_id="campaign",
            macro_round=0,
            snapshot_file_sha256="a" * 64,
            snapshot_sha256="b" * 64,
            replay_counts=[2] * 8,
            selected_counts=[2] * 8,
            current_counts=[2] * 8,
            historical_counts=[0] * 8,
        )

        def rows(*, changing: bool) -> list[dict[str, object]]:
            result = []
            for step in (0, 100, 200):
                action_mse = 0.25 + (step / 100000.0 if changing else 0.0)
                result.append(
                    self._training_quality_row(
                        probe=probe,
                        optimizer_step=step,
                        learned_action_indices=[0],
                        action_mse=action_mse,
                    )
                )
            return result

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_quality.jsonl"
            changing = rows(changing=True)
            self._write_telemetry(path, changing)
            observed = completion._training_quality_telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                snapshot_file_sha256="a" * 64,
                snapshot_sha256="b" * 64,
                optimizer_steps=200,
                eval_interval=100,
                learned_action_indices=[0],
                world_size=8,
                partition_seed=23,
                probe_summary=probe,
            )
            self.assertEqual(sorted(observed), [0, 100, 200])

            self._write_telemetry(path, changing[:-1])
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "coverage is incomplete"
            ):
                completion._training_quality_telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    optimizer_steps=200,
                    eval_interval=100,
                    learned_action_indices=[0],
                    world_size=8,
                    partition_seed=23,
                    probe_summary=probe,
                )

            self._write_telemetry(path, rows(changing=False))
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "bit-identical"
            ):
                completion._training_quality_telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    optimizer_steps=200,
                    eval_interval=100,
                    learned_action_indices=[0],
                    world_size=8,
                    partition_seed=23,
                    probe_summary=probe,
                )

    def test_training_quality_attests_bounded_probe_not_full_replay(self) -> None:
        probe = self._probe_summary(
            campaign_id="campaign",
            macro_round=0,
            snapshot_file_sha256="a" * 64,
            snapshot_sha256="b" * 64,
            replay_counts=[125_000] * 8,
            selected_counts=[32] * 8,
            current_counts=[32] * 8,
            historical_counts=[0] * 8,
        )
        rows = [
            self._training_quality_row(
                probe=probe,
                optimizer_step=step,
                learned_action_indices=[0],
                action_mse=0.25 + step / 100000.0,
            )
            for step in (0, 100, 200)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_quality.jsonl"
            self._write_telemetry(path, rows)
            observed = completion._training_quality_telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                snapshot_file_sha256="a" * 64,
                snapshot_sha256="b" * 64,
                optimizer_steps=200,
                eval_interval=100,
                learned_action_indices=[0],
                world_size=8,
                partition_seed=23,
                probe_summary=probe,
            )
            self.assertEqual(observed[0]["training_probe_transitions_global"], 256)
            self.assertEqual(
                observed[0]["training_replay_transitions_global"], 1_000_000
            )

            tampered_rows = [dict(row) for row in rows]
            tampered_rows[1]["training_probe_selection_sha256"] = "f" * 64
            self._write_telemetry(path, tampered_rows)
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "contract mismatch"
            ):
                completion._training_quality_telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    optimizer_steps=200,
                    eval_interval=100,
                    learned_action_indices=[0],
                    world_size=8,
                    partition_seed=23,
                    probe_summary=probe,
                )

            tampered_probe = dict(probe)
            tampered_probe["snapshot_sha256"] = "c" * 64
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "lineage|digest",
            ):
                completion._training_quality_probe_summary(
                    tampered_probe,
                    campaign_id="campaign",
                    macro_round=0,
                    snapshot_file_sha256="a" * 64,
                    snapshot_sha256="b" * 64,
                    partition_seed=23,
                    world_size=8,
                )

    def test_wandb_readback_accepts_step_zero_heldout_quality_row(self) -> None:
        training = {
            1: {
                "loss": 0.5,
                "gradient_l2_norm": 0.75,
                "learning_rate": 2.5e-5,
                "global_samples_seen": 64,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
            }
        }
        heldout_record = {
            "eval_frontier": 0,
            "action_mse": 0.25,
            "action_mae": 0.4,
            "action_rmse": 0.5,
            "action_cosine": 0.75,
            "action_absolute_error_p90": 0.8,
            "action_l2_p90": 1.2,
            "action_mse_delta_from_step0": 0.0,
            "action_mse_relative_improvement_from_step0": 0.0,
            "heldout_transitions_global": 16,
            "per_action": {
                "0": {"mse": 0.25, "mae": 0.4, "rmse": 0.5, "cosine": 0.75}
            },
        }
        probe = self._probe_summary(
            campaign_id="campaign",
            macro_round=0,
            snapshot_file_sha256="8" * 64,
            snapshot_sha256="9" * 64,
            replay_counts=[2] * 8,
            selected_counts=[2] * 8,
            current_counts=[2] * 8,
            historical_counts=[0] * 8,
        )
        training_quality_record = self._training_quality_row(
            probe=probe,
            optimizer_step=0,
            learned_action_indices=[0],
            action_mse=0.25,
        )
        training_context_record = self._training_context(
            campaign_id="campaign",
            macro_round=0,
            config_sha256="2" * 64,
            parent_sha256="4" * 64,
            snapshot_file_sha256="8" * 64,
            snapshot_sha256="9" * 64,
            probe_selection_sha256=str(probe["selection_sha256"]),
            replay_transitions=16,
        )
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
            "scheduled_dagger_beta": 0.5,
        }
        quality_row = {
            "_step": 0,
            "quality/eval_frontier": 0,
            "quality/heldout_action_mse": 0.25,
            "quality/heldout_action_mae": 0.4,
            "quality/heldout_action_rmse": 0.5,
            "quality/heldout_action_cosine": 0.75,
            "quality/heldout_action_absolute_error_p90": 0.8,
            "quality/heldout_action_l2_p90": 1.2,
            "quality/heldout_action_mse_delta_from_step0": 0.0,
            "quality/heldout_action_mse_relative_improvement_from_step0": 0.0,
            "quality/heldout_transitions": 16,
            "quality/action_0/mse": 0.25,
            "quality/action_0/mae": 0.4,
            "quality/action_0/rmse": 0.5,
            "quality/action_0/cosine": 0.75,
        }
        quality_row.update(
            completion._training_quality_wandb_fields(training_quality_record)
        )
        quality_row.update(
            completion._training_context_wandb_fields(
                training_context_record
            )
        )
        quality_row.update(self._validation_ood_aliases(heldout_record))
        training_row = {
            "schema": completion.TELEMETRY_SCHEMA,
            "campaign_id": "campaign",
            "macro_round": 0,
            "optimizer_step": 1,
            "loss": 0.5,
            "gradient_l2_norm": 0.75,
            "learning_rate": 2.5e-5,
            "global_samples_seen": 64,
            "world_size": 8,
            "gradient_reduction": "ordered_allgather_rank_sum_v1",
            "replay_transitions_rank": 1,
            "replay_transitions_per_rank": [1] * 8,
            "replay_transitions_global": 8,
        }
        training_row.update(self._training_optimizer_aliases(training[1]))

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
        readback_arguments = {
            "telemetry": training,
            "campaign_id": "campaign",
            "macro_round": 0,
            "code_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "learner_identity_sha256": "3" * 64,
            "parent_sha256": "4" * 64,
            "parent_runtime_policy_sha256": "5" * 64,
            "runtime_container_sha256": "6" * 64,
            "learner_contract_sha256": "7" * 64,
            "snapshot_file_sha256": "8" * 64,
            "snapshot_sha256": "9" * 64,
            "entity": "nvidia",
            "project": "project",
            "run_id": "run",
            "expected_email": "ldu@nvidia.com",
            "timeout_seconds": 1,
            "training_quality_telemetry": {0: training_quality_record},
            "training_context": training_context_record,
            "training_context_file_sha256": "c" * 64,
            "heldout_telemetry": {0: heldout_record},
            "heldout_snapshot_file_sha256": "a" * 64,
            "heldout_snapshot_sha256": "b" * 64,
            "heldout_eval_interval": 100,
        }
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ):
            readback = completion._wandb_readback(**readback_arguments)
        self.assertEqual(readback["heldout_evaluation"]["verified_steps"], [0])
        self.assertEqual(
            readback["training_quality_evaluation"]["verified_steps"], [0]
        )
        self.assertEqual(
            readback["training_context"],
            {
                "file_sha256": "c" * 64,
                "metric_keys": sorted(completion.TRAINING_CONTEXT_METRIC_KEYS),
                "verified_optimizer_step": 0,
            },
        )

        missing_ood = quality_row.pop("3_validation_ood/action_mse")
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ):
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "held-out history differs",
            ):
                completion._wandb_readback(**readback_arguments)
        quality_row["3_validation_ood/action_mse"] = missing_ood

        missing_optimizer = training_row.pop(
            "2_training_quality/optimization_loss"
        )
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ):
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "visible training-quality optimizer aliases differ",
            ):
                completion._wandb_readback(**readback_arguments)
        training_row["2_training_quality/optimization_loss"] = missing_optimizer

        missing_context = quality_row.pop(
            "2_training_quality/expert_intervention_rate"
        )
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ):
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "training-context history differs",
            ):
                completion._wandb_readback(**readback_arguments)
        quality_row[
            "2_training_quality/expert_intervention_rate"
        ] = missing_context

    def test_wandb_readback_rejects_partial_training_quality_frontier(self) -> None:
        probe = self._probe_summary(
            campaign_id="campaign",
            macro_round=0,
            snapshot_file_sha256="8" * 64,
            snapshot_sha256="9" * 64,
            replay_counts=[2] * 8,
            selected_counts=[2] * 8,
            current_counts=[2] * 8,
            historical_counts=[0] * 8,
        )
        quality = self._training_quality_row(
            probe=probe,
            optimizer_step=0,
            learned_action_indices=[0],
            action_mse=0.25,
        )
        training = {
            1: {
                "loss": 0.5,
                "gradient_l2_norm": 0.75,
                "learning_rate": 2.5e-5,
                "global_samples_seen": 64,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
            }
        }
        training_row = {
            "schema": completion.TELEMETRY_SCHEMA,
            "campaign_id": "campaign",
            "macro_round": 0,
            "optimizer_step": 1,
            "loss": 0.5,
            "gradient_l2_norm": 0.75,
            "learning_rate": 2.5e-5,
            "global_samples_seen": 64,
            "world_size": 8,
            "gradient_reduction": "ordered_allgather_rank_sum_v1",
            "replay_transitions_rank": 1,
            "replay_transitions_per_rank": [1] * 8,
            "replay_transitions_global": 8,
        }
        training_row.update(self._training_optimizer_aliases(training[1]))
        partial_quality_row = {
            "_step": 0,
            "2_training_quality/action_mse": 0.25,
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
        }

        class FakeRun:
            state = "finished"

            def __init__(self) -> None:
                self.config = config

            @staticmethod
            def scan_history(*, page_size: int):
                self.assertEqual(page_size, 500)
                return [partial_quality_row, training_row]

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
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "training-quality history is partial",
            ):
                completion._wandb_readback(
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
                    training_quality_telemetry={0: quality},
                )

        # A completely absent remote quality row must time out rather than
        # allowing the valid local JSONL to satisfy completion by itself.
        partial_quality_row.clear()
        monotonic = iter((0.0, 0.0, 2.0))
        with mock.patch(
            "scripts.scaling.wandb_preflight._load_wandb",
            return_value=fake_wandb,
        ), mock.patch(
            "scripts.scaling.wandb_preflight.query_identity",
            return_value=identity,
        ), mock.patch.object(
            completion.time, "monotonic", side_effect=lambda: next(monotonic)
        ), mock.patch.object(completion.time, "sleep"):
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "completion readback failed",
            ):
                completion._wandb_readback(
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
                    training_quality_telemetry={0: quality},
                )

    def test_wandb_readback_rejects_conflicting_duplicate_quality_frontier(
        self,
    ) -> None:
        training = {
            1: {
                "loss": 0.5,
                "gradient_l2_norm": 0.75,
                "learning_rate": 2.5e-5,
                "global_samples_seen": 64,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
            }
        }
        heldout_record = {
            "eval_frontier": 0,
            "action_mse": 0.25,
            "action_mae": 0.4,
            "action_rmse": 0.5,
            "action_cosine": 0.75,
            "action_absolute_error_p90": 0.8,
            "action_l2_p90": 1.2,
            "action_mse_delta_from_step0": 0.0,
            "action_mse_relative_improvement_from_step0": 0.0,
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
            "quality/eval_frontier": 0,
            "quality/heldout_action_mse": 0.25,
            "quality/heldout_action_mae": 0.4,
            "quality/heldout_action_rmse": 0.5,
            "quality/heldout_action_cosine": 0.75,
            "quality/heldout_action_absolute_error_p90": 0.8,
            "quality/heldout_action_l2_p90": 1.2,
            "quality/heldout_action_mse_delta_from_step0": 0.0,
            "quality/heldout_action_mse_relative_improvement_from_step0": 0.0,
            "quality/heldout_transitions": 16,
            "quality/action_0/mse": 0.25,
            "quality/action_0/mae": 0.4,
            "quality/action_0/rmse": 0.5,
            "quality/action_0/cosine": 0.75,
        }
        quality_row.update(self._validation_ood_aliases(heldout_record))
        conflicting = dict(quality_row)
        conflicting["quality/action_0/mse"] = 0.3
        training_row = {
            "schema": completion.TELEMETRY_SCHEMA,
            "campaign_id": "campaign",
            "macro_round": 0,
            "optimizer_step": 1,
            "loss": 0.5,
            "gradient_l2_norm": 0.75,
            "learning_rate": 2.5e-5,
            "global_samples_seen": 64,
            "world_size": 8,
            "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
                return [quality_row, conflicting, training_row]

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
            with self.assertRaisesRegex(
                completion.CompletionAttestationError,
                "held-out history differs",
            ):
                completion._wandb_readback(
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
                    "gradient_l2_norm": 0.5 / step,
                    "learning_rate": 2.5e-5,
                    "global_samples_seen": step * 64,
                    "replay_transitions_rank": 1,
                    "replay_transitions_per_rank": [1] * 8,
                    "replay_transitions_global": 8,
                    "world_size": 8,
                    "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
                world_size=8,
                batch_size_per_rank=8,
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
                    world_size=8,
                    batch_size_per_rank=8,
                )

    def test_telemetry_sample_frontier_uses_declared_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            row = {
                "schema": completion.TELEMETRY_SCHEMA,
                "campaign_id": "campaign",
                "macro_round": 0,
                "optimizer_step": 3,
                "loss": 0.25,
                "gradient_l2_norm": 0.5,
                "learning_rate": 2.5e-5,
                "global_samples_seen": 3 * 4 * 2,
                "replay_transitions_rank": 5,
                "replay_transitions_per_rank": [5, 7],
                "replay_transitions_global": 12,
                "world_size": 2,
                "gradient_reduction": "ordered_allgather_rank_sum_v1",
            }
            rows = [
                dict(row, optimizer_step=step, global_samples_seen=step * 8)
                for step in (1, 2, 3)
            ]
            self._write_telemetry(path, rows)
            observed = completion._telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                optimizer_steps=3,
                replay_transitions_global=12,
                world_size=2,
                batch_size_per_rank=4,
            )
            self.assertEqual(set(observed), {1, 2, 3})
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "contract mismatch"
            ):
                completion._telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    optimizer_steps=3,
                    replay_transitions_global=12,
                    world_size=2,
                    batch_size_per_rank=8,
                )

    def test_publish_forwards_campaign_root_to_replay_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            parent = root / "parent.pt"
            snapshot = root / "replay.json"
            parent.write_bytes(b"parent")
            snapshot.write_bytes(b"snapshot")
            arguments = SimpleNamespace(
                output_dir=output,
                campaign_id="campaign",
                code_sha256="1" * 64,
                config_sha256="2" * 64,
                learner_contract_sha256="3" * 64,
                runtime_container_sha256="4" * 64,
                learner_identity_sha256="5" * 64,
                parent_sha256=sha256_file(parent),
                snapshot_file_sha256=sha256_file(snapshot),
                snapshot_sha256="6" * 64,
                world_size=8,
                artifact_root=root,
                parent_checkpoint=parent,
                snapshot=snapshot,
                macro_round=0,
            )
            sentinel = RuntimeError("loader-called")

            def load(*args, **kwargs):
                self.assertEqual(args, (snapshot,))
                self.assertEqual(
                    kwargs,
                    {
                        "expected_file_sha256": sha256_file(snapshot),
                        "artifact_root": root,
                    },
                )
                raise sentinel

            with mock.patch.object(completion, "load_snapshot", side_effect=load):
                with self.assertRaisesRegex(RuntimeError, "loader-called"):
                    completion.publish_attestation(arguments)

    def test_learner_writer_output_is_accepted_by_completion_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = JsonlTelemetry(path, enabled=True)
            emitted = {
                "schema": TELEMETRY_SCHEMA,
                "campaign_id": "campaign",
                "macro_round": 0,
                "optimizer_step": 1,
                "loss": 0.5,
                "gradient_l2_norm": 0.75,
                "learning_rate": 2.5e-5,
                "global_samples_seen": 64,
                "replay_transitions_rank": 4,
                "replay_transitions_per_rank": [4] * 8,
                "replay_transitions_global": 32,
                "world_size": 8,
                "gradient_reduction": "ordered_allgather_rank_sum_v1",
            }
            writer.log(emitted)
            writer.close(exit_code=0)

            records = completion._telemetry_records(
                path,
                campaign_id="campaign",
                macro_round=0,
                optimizer_steps=1,
                replay_transitions_global=32,
                world_size=8,
                batch_size_per_rank=8,
            )
            self.assertEqual(records, {1: emitted})

            corrupted = dict(emitted, global_samples_seen=63)
            self._write_telemetry(path, [corrupted])
            with self.assertRaisesRegex(
                completion.CompletionAttestationError, "contract mismatch"
            ):
                completion._telemetry_records(
                    path,
                    campaign_id="campaign",
                    macro_round=0,
                    optimizer_steps=1,
                    replay_transitions_global=32,
                    world_size=8,
                    batch_size_per_rank=8,
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
                    "gradient_l2_norm": 0.5,
                    "learning_rate": 2.5e-5,
                    "global_samples_seen": 64,
                    "replay_transitions_rank": 9,
                    "replay_transitions_per_rank": [9, 1, 1, 1, 1, 1, 1, 1],
                    "replay_transitions_global": 16,
                    "world_size": 8,
                    "gradient_reduction": "ordered_allgather_rank_sum_v1",
                },
                {
                    "schema": completion.TELEMETRY_SCHEMA,
                    "campaign_id": "campaign",
                    "macro_round": 0,
                    "optimizer_step": 2,
                    "loss": 0.5,
                    "gradient_l2_norm": 0.25,
                    "learning_rate": 2.5e-5,
                    "global_samples_seen": 128,
                    "replay_transitions_rank": 8,
                    "replay_transitions_per_rank": [8, 2, 1, 1, 1, 1, 1, 1],
                    "replay_transitions_global": 16,
                    "world_size": 8,
                    "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
                    world_size=8,
                    batch_size_per_rank=8,
                )


if __name__ == "__main__":
    unittest.main()
