import copy
import json
import os
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # The artifact/control-plane test environment is CPU-only.
    torch = None
    nn = types.SimpleNamespace(Module=object)

from round_dagger import artifacts, replay
from round_dagger.learner_identity import learner_identity_sha256
from tests.policy_bundle_helpers import portable_policy_spec

if torch is not None:
    from round_dagger.learner import (
        BUNDLE_SCHEMA,
        CUDA_CUBLAS_WORKSPACE_CONFIG,
        JsonlTelemetry,
        LearnerBundleConfig,
        LearnerError,
        MappingPolicyAdapter,
        ORDERED_ALLGATHER_GRADIENT_REDUCTION,
        RESULT_SCHEMA,
        SINGLE_PROCESS_GRADIENT_REDUCTION,
        TRAINING_QUALITY_PROBE_BUDGET,
        TrainingQualityProbe,
        _ordered_allgather_gradient_hook,
        build_training_quality_probe,
        configure_deterministic_runtime,
        evaluate_heldout_actions,
        load_replay_partition,
        partition_shards,
        publish_learner_result,
        supervised_action_loss,
        train_from_replay,
    )


SHA = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, observations):
        rgb = observations["vision_obs"].mean(dim=(1, 2))
        actor = observations["actor_obs"].float()
        features = torch.cat((rgb, actor), dim=-1)
        return {"actions": self.linear(features)}


class PrefixOnlyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, observations):
        rgb = observations["vision_obs"].mean(dim=(1, 2))
        features = torch.cat((rgb, observations["actor_obs"].float()), dim=-1)
        return {"actions": self.linear(features)}


class ProbeRecordingPolicy(TinyPolicy):
    """Tiny policy that records only the learner's eval-mode probe inputs."""

    def __init__(self):
        super().__init__()
        self.probe_batches = []

    def forward(self, observations):
        if not self.training:
            self.probe_batches.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in observations.items()
                }
            )
        return super().forward(observations)


def tiny_policy_factory(_bundle):
    """Importable factory used by the torchrun/CLI smoke path."""

    return TinyPolicy()


def _config(
    *,
    model_lineage="a" * 64,
    mean=(0.5, 0.5, 0.5),
):
    mapping = {
        "schema": BUNDLE_SCHEMA,
        "campaign_id": "learner-test",
        "generation_id": "gen-000-bc",
        "macro_round": 0,
        "model_lineage_sha256": model_lineage,
        "code_sha256": "b" * 64,
        "rgb_mean": list(mean),
        "rgb_std": [0.25, 0.25, 0.25],
        "rgb_input_key": "vision_obs",
        "rgb_layout": "NHWC",
        "output_key": "actions",
        "train_action_slice": [0, 1],
        "batch_size": 8,
        "target_optimizer_examples": 960,
        "checkpoint_every_optimizer_examples": 8,
        "learning_rate": 0.025,
        "weight_decay": 0.0,
        "seed": 17,
        "partition_seed": 23,
        "model_config": portable_policy_spec(model_lineage, "2" * 64),
    }
    return LearnerBundleConfig.from_mapping(mapping)


def _train_for_steps(model, *, optimizer_steps, config, **kwargs):
    """Express compact CPU fixtures through the optimizer-example API."""

    return train_from_replay(
        model,
        config=config,
        target_optimizer_examples=optimizer_steps * config.batch_size,
        checkpoint_every_optimizer_examples=(
            config.checkpoint_every_optimizer_examples
        ),
        **kwargs,
    )


def _rollout_arrays(env_id, *, steps=8):
    rgb = np.empty((steps, 1, 3, 2, 3), dtype=np.uint8)
    actor = np.empty((steps, 1, 1), dtype=np.float32)
    expert = np.empty((steps, 1, 2), dtype=np.float32)
    for step in range(steps):
        values = np.asarray(
            [
                (17 * step + 31 * env_id) % 256,
                (43 * step + 11 * env_id + 7) % 256,
                (71 * step + 19 * env_id + 3) % 256,
            ],
            dtype=np.uint8,
        )
        rgb[step, 0] = values
        actor[step, 0, 0] = np.float32((step - env_id) / 7.0)
        normalized = (values.astype(np.float32) / 255.0 - 0.5) / 0.25
        expert[step, 0, 0] = (
            normalized @ np.asarray([0.35, -0.2, 0.5], np.float32)
            + 0.75 * actor[step, 0, 0]
        )
        # This dimension is intentionally excluded by train_action_slice.  It
        # represents fixed Homie dimensions that must not contribute gradients.
        expert[step, 0, 1] = np.float32(1000.0 + 100.0 * step)
    zeros = np.zeros((steps, 1), dtype=np.bool_)
    return {
        "policy_rgb": rgb,
        "expert_actions": expert,
        "student_actions": np.zeros_like(expert),
        "behavior_actions": expert.copy(),
        "expert_mask": np.ones((steps, 1), dtype=np.bool_),
        "dones": zeros.copy(),
        "observation_is_raytraced": zeros.copy(),
        "global_env_ids": np.asarray([env_id], dtype=np.int64),
        "episode_ids": np.full((steps, 1), env_id, dtype=np.int64),
        "episode_steps": np.arange(steps, dtype=np.int64)[:, None],
        "obs__actor_obs": actor,
        "obs__integer_feature": np.full((steps, 1, 1), env_id, dtype=np.int16),
    }


@unittest.skipIf(torch is None, "ordered gradient reduction tests require PyTorch")
class OrderedGradientReductionTest(unittest.TestCase):
    def test_hook_averages_gathered_gradients_in_explicit_rank_order(self):
        source = torch.tensor([17.0, -5.0], dtype=torch.float32)
        source_before = source.clone()
        rank_gradients = [
            torch.tensor([1.0e20, 1.0], dtype=torch.float32),
            torch.tensor([-1.0e20, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 3.0], dtype=torch.float32),
            torch.tensor([4.0, 4.0], dtype=torch.float32),
        ]
        bucket = mock.Mock()
        bucket.buffer.return_value = source

        def gather(outputs, input_tensor):
            self.assertIs(input_tensor, source)
            self.assertEqual(len(outputs), len(rank_gradients))
            for output, gradient in zip(outputs, rank_gradients):
                output.copy_(gradient)

        with mock.patch(
            "round_dagger.learner.torch.distributed.all_gather",
            side_effect=gather,
        ) as all_gather:
            reduced = _ordered_allgather_gradient_hook(
                len(rank_gradients), bucket
            ).wait()

        # The first component is deliberately non-associative: this value is
        # produced only by (((rank0 + rank1) + rank2) + rank3) / world_size.
        torch.testing.assert_close(
            reduced,
            torch.tensor([1.75, 2.5], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(source, source_before, rtol=0, atol=0)
        all_gather.assert_called_once()

    def test_hook_rejects_a_single_rank_contract(self):
        bucket = mock.Mock()
        with self.assertRaisesRegex(LearnerError, "requires multiple ranks"):
            _ordered_allgather_gradient_hook(1, bucket)
        bucket.buffer.assert_not_called()


@unittest.skipIf(torch is None, "round DAgger learner tests require PyTorch")
class LearnerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.campaign_root = Path(self.temporary.name) / "data"
        self.root = self.campaign_root / "BC"
        self.root.mkdir(parents=True)
        self.campaign_root_patch = mock.patch(
            "round_dagger.learner.CANONICAL_CAMPAIGN_ROOT", self.campaign_root
        )
        self.campaign_root_patch.start()
        self.addCleanup(self.campaign_root_patch.stop)
        self.campaign_id_patch = mock.patch(
            "round_dagger.learner.CANONICAL_CAMPAIGN_ID", "learner-test"
        )
        self.campaign_id_patch.start()
        self.addCleanup(self.campaign_id_patch.stop)
        self.cohort = self.root / "cohort"
        artifacts.create_cohort(
            self.cohort,
            campaign_id="learner-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            expected_ranks=[0],
            expected_global_env_ids=[0, 1],
            expected_transitions=16,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [3, 2, 3],
                "views": ["ext1", "ext2", "wrist"],
            },
            action_contract={"dtype": "float32", "shape": [2]},
            **SHA,
        )
        for part, env_id in enumerate((0, 1)):
            artifacts.publish_rollout_shard(
                self.cohort,
                rank=0,
                rollout_id=0,
                part_index=part,
                arrays=_rollout_arrays(env_id),
                campaign_id="learner-test",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                **SHA,
            )
        artifacts.seal_cohort(self.cohort)
        self.snapshot_path = self.root / "snapshot.json"
        with replay.ReplayLedger(self.root / "replay.sqlite3") as ledger:
            ledger.ingest_cohort(self.cohort)
            self.snapshot = ledger.build_snapshot(
                "learner-test",
                through_macro_round=0,
                output=self.snapshot_path,
            )

    def tearDown(self):
        self.temporary.cleanup()

    def learner_attestation(self, config):
        container_sha = "c" * 64
        contract_sha = "d" * 64
        identity_sha = learner_identity_sha256(
            campaign_id=config.campaign_id,
            macro_round=config.macro_round,
            learner_code_sha256=config.code_sha256,
            learner_config_sha256=config.sha256,
            replay_snapshot_file_sha256=self.snapshot["file_sha256"],
            replay_snapshot_sha256=self.snapshot["snapshot_sha256"],
            parent_checkpoint_sha256=config.model_lineage_sha256,
            parent_runtime_policy_sha256=config.to_mapping()["model_config"][
                "runtime_policy_sha256"
            ],
            runtime_container_sha256=container_sha,
            learner_contract_sha256=contract_sha,
        )
        return {
            "learner_identity_sha256": identity_sha,
            "runtime_container_sha256": container_sha,
            "learner_contract_sha256": contract_sha,
        }


class ReplayInputContractTest(LearnerFixture):
    def test_train_only_bundle_rejects_validation_fields(self):
        raw = _config().to_mapping()
        raw["validation_snapshot_sha256"] = "9" * 64
        with self.assertRaisesRegex(LearnerError, "forbids validation/test fields"):
            LearnerBundleConfig.from_mapping(raw)

    def test_rgb_is_normalized_and_all_policy_observations_are_reconstructed(self):
        config = _config()
        partition = load_replay_partition(
            self.snapshot_path,
            config,
            expected_snapshot_file_sha256=self.snapshot["file_sha256"],
        )
        observations, targets = partition.batch([0], device="cpu")
        source = _rollout_arrays(0)["policy_rgb"][0, 0].astype(np.float32)
        expected = (source / 255.0 - 0.5) / 0.25
        torch.testing.assert_close(
            observations["vision_obs"], torch.from_numpy(expected)[None]
        )
        self.assertIn("actor_obs", observations)
        self.assertIn("integer_feature", observations)
        self.assertEqual(observations["integer_feature"].dtype, torch.int16)
        self.assertEqual(tuple(targets.shape), (1, 2))

    def test_rank_partition_is_stable_disjoint_and_complete(self):
        shards = [
            {
                "logical_id": f"campaign:r{index // 5}",
                "shard_id": f"shard-{index:02d}",
                "macro_round": index // 5,
                "cohort_index": index % 5,
                "transition_count": index + 1,
            }
            for index in range(20)
        ]
        assignments = [
            partition_shards(shards, rank=rank, world_size=4, seed=91)
            for rank in range(4)
        ]
        identities = [
            {f"{item['logical_id']}/{item['shard_id']}" for item in assigned}
            for assigned in assignments
        ]
        self.assertEqual(
            set.union(*identities),
            {f"{x['logical_id']}/{x['shard_id']}" for x in shards},
        )
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(identities[left] & identities[right])
        self.assertEqual(
            assignments[2], partition_shards(shards, rank=2, world_size=4, seed=91)
        )
        reversed_assignments = [
            partition_shards(list(reversed(shards)), rank=rank, world_size=4, seed=91)
            for rank in range(4)
        ]
        self.assertEqual(assignments, reversed_assignments)
        transition_counts = [
            sum(item["transition_count"] for item in assigned)
            for assigned in assignments
        ]
        self.assertLessEqual(max(transition_counts) - min(transition_counts), 20)

    def test_attempt03_equal_shards_are_exactly_balanced(self):
        shards = [
            {
                "logical_id": (
                    "roborender-dagger-256x512-v1:"
                    f"r000000000:c{cohort:04d}:train"
                ),
                "shard_id": (
                    f"rank-{collector_rank:04d}_rollout-00000000_part-{part:04d}"
                ),
                "macro_round": 0,
                "cohort_index": cohort,
                "transition_count": 2048,
            }
            for cohort in range(4)
            for collector_rank in range(2)
            for part in range(8)
        ]
        assignments = [
            partition_shards(shards, rank=rank, world_size=8, seed=23)
            for rank in range(8)
        ]
        self.assertEqual([len(assigned) for assigned in assignments], [8] * 8)
        self.assertEqual(
            [
                sum(item["transition_count"] for item in assigned)
                for assigned in assignments
            ],
            [16384] * 8,
        )

    def test_unequal_shards_are_transition_balanced(self):
        shards = [
            {
                "logical_id": "campaign:r000000000:c0000:train",
                "shard_id": f"shard-{index:02d}",
                "macro_round": 0,
                "cohort_index": 0,
                "transition_count": transitions,
            }
            for index, transitions in enumerate([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        ]
        assignments = [
            partition_shards(shards, rank=rank, world_size=3, seed=7)
            for rank in range(3)
        ]
        self.assertEqual(
            [
                sum(item["transition_count"] for item in assigned)
                for assigned in assignments
            ],
            [19, 18, 18],
        )

    @staticmethod
    def _probe_snapshot(*, through_round: int, rows_per_shard: int = 1024):
        shards = [
            {
                "logical_id": f"campaign:r{macro_round:09d}:c{rank:04d}:train",
                "shard_id": f"rank-{rank:04d}_part-0000",
                "macro_round": macro_round,
                "cohort_index": rank,
                "transition_count": rows_per_shard,
            }
            for macro_round in range(through_round + 1)
            for rank in range(8)
        ]
        return {
            "campaign_id": "probe-campaign",
            "through_macro_round": through_round,
            "file_sha256": "a" * 64,
            "snapshot_sha256": "b" * 64,
            "transition_count": sum(
                shard["transition_count"] for shard in shards
            ),
            "resolved_shards": shards,
        }

    def test_training_quality_probe_is_fixed_bounded_and_lineage_bound(self):
        snapshot = self._probe_snapshot(through_round=4)
        first = build_training_quality_probe(
            snapshot, world_size=8, partition_seed=23
        )
        repeated = build_training_quality_probe(
            {**snapshot, "resolved_shards": list(reversed(snapshot["resolved_shards"]))},
            world_size=8,
            partition_seed=23,
        )
        summary = first.summary()
        self.assertEqual(summary, repeated.summary())
        self.assertEqual(first.indices_by_rank, repeated.indices_by_rank)
        self.assertEqual(
            summary["transitions_global"], TRAINING_QUALITY_PROBE_BUDGET
        )
        self.assertEqual(summary["current_round_transitions_global"], 128)
        self.assertEqual(summary["historical_transitions_global"], 128)
        self.assertTrue(all(count > 0 for count in summary["transitions_per_rank"]))
        self.assertEqual(
            sum(summary["transitions_per_rank"]),
            sum(len(indices) for indices in first.indices_by_rank),
        )

        changed_lineage = build_training_quality_probe(
            {**snapshot, "file_sha256": "c" * 64},
            world_size=8,
            partition_seed=23,
        )
        self.assertNotEqual(
            summary["selection_sha256"],
            changed_lineage.summary()["selection_sha256"],
        )
        tampered = first.summary()
        tampered["transitions_per_rank"][0] += 1
        with self.assertRaisesRegex(LearnerError, "coverage|counts"):
            TrainingQualityProbe(
                summary_document=tampered,
                indices_by_rank=first.indices_by_rank,
            )
        overlapping = [list(indices) for indices in first.indices_by_rank]
        overlapping[0][1] = overlapping[0][0]
        with self.assertRaisesRegex(LearnerError, "duplicate"):
            TrainingQualityProbe(
                summary_document=summary,
                indices_by_rank=tuple(tuple(indices) for indices in overlapping),
            )

    def test_round_zero_training_quality_probe_is_all_current(self):
        probe = build_training_quality_probe(
            self._probe_snapshot(through_round=0),
            world_size=8,
            partition_seed=23,
        )
        summary = probe.summary()
        self.assertEqual(summary["transitions_global"], 256)
        self.assertEqual(summary["current_round_transitions_global"], 256)
        self.assertEqual(summary["historical_transitions_global"], 0)

    def test_training_quality_forward_count_is_independent_of_replay_growth(self):
        class CountingPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.examples = 0

            def forward(self, observations):
                self.calls += 1
                self.examples += int(observations["feature"].shape[0])
                return {"actions": observations["feature"]}

        class SyntheticPartition:
            def __init__(self, population):
                self.rank = 0
                self.world_size = 1
                self.transition_count = population
                self.transition_counts_by_rank = (population,)
                self.global_transition_count = population
                self.config = types.SimpleNamespace(
                    batch_size=8,
                    train_action_slice=None,
                    campaign_id="probe-campaign",
                    macro_round=0,
                )
                self.snapshot = {
                    "snapshot_sha256": "b" * 64,
                    "file_sha256": "a" * 64,
                }

            @staticmethod
            def batch(indices, *, device):
                values = torch.as_tensor(indices, dtype=torch.float32, device=device)
                values = values.unsqueeze(-1).remainder(17).div(17)
                return {"feature": values}, torch.zeros_like(values)

        observed = []
        for population in (1_024, 1_000_000):
            model = CountingPolicy()
            record = evaluate_heldout_actions(
                model,
                partition=SyntheticPartition(population),
                adapter=MappingPolicyAdapter(),
                device="cpu",
                optimizer_step=0,
                indices=tuple(range(TRAINING_QUALITY_PROBE_BUDGET)),
                transition_counts_by_rank=[TRAINING_QUALITY_PROBE_BUDGET],
                include_per_action_p90=True,
            )
            observed.append((model.calls, model.examples))
            self.assertEqual(record["heldout_transitions_global"], 256)
            self.assertIn("0", record["per_action_absolute_error_p90"])
        self.assertEqual(observed, [(32, 256), (32, 256)])

    def test_partition_rejects_missing_or_invalid_transition_count(self):
        base = {
            "logical_id": "campaign:r000000000:c0000:train",
            "shard_id": "shard-00",
            "macro_round": 0,
            "cohort_index": 0,
        }
        for value in (None, True, 0, -1, 1.5):
            shard = dict(base)
            if value is not None:
                shard["transition_count"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                LearnerError, "transition_count"
            ):
                partition_shards([shard], rank=0, world_size=1, seed=0)

    def test_partition_rejects_duplicate_identity(self):
        shard = {
            "logical_id": "campaign:r000000000:c0000:train",
            "shard_id": "shard-00",
            "macro_round": 0,
            "cohort_index": 0,
            "transition_count": 1,
        }
        with self.assertRaisesRegex(LearnerError, "duplicate shard identity"):
            partition_shards([shard, dict(shard)], rank=0, world_size=1, seed=0)

    def test_payload_change_after_snapshot_is_rejected(self):
        payload = Path(self.snapshot["cohorts"][0]["shards"][0]["payload_path"])
        # Paths embedded in a replay snapshot are cohort-relative.
        payload = self.cohort / payload
        payload.chmod(0o600)
        payload.write_bytes(b"tampered")
        with self.assertRaisesRegex(LearnerError, "snapshot validation failed"):
            load_replay_partition(self.snapshot_path, _config())


class SupervisedTrainingAndResumeTest(LearnerFixture):
    def _initial_state(self):
        torch.manual_seed(1234)
        return copy.deepcopy(TinyPolicy().state_dict())

    def test_cuda_runtime_determinism_contract_is_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaisesRegex(
                LearnerError, "requires CUBLAS_WORKSPACE_CONFIG"
            ):
                configure_deterministic_runtime("cuda:0")
            with self.assertRaisesRegex(
                LearnerError, "requires CUBLAS_WORKSPACE_CONFIG"
            ):
                _train_for_steps(
                    TinyPolicy(),
                    snapshot_path=self.root / "must-not-be-read.json",
                    config=_config(),
                    output_dir=self.root / "must-not-be-created",
                    optimizer_steps=1,
                    device="cuda:0",
                )
            self.assertFalse((self.root / "must-not-be-created").exists())

        original = {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "precision": torch.get_float32_matmul_precision(),
        }
        try:
            with mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": CUDA_CUBLAS_WORKSPACE_CONFIG},
            ):
                configure_deterministic_runtime("cuda:0")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        finally:
            torch.use_deterministic_algorithms(
                original["enabled"], warn_only=original["warn_only"]
            )
            torch.backends.cudnn.benchmark = original["benchmark"]
            torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
            torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]
            torch.backends.cuda.matmul.allow_tf32 = original["matmul_tf32"]
            torch.set_float32_matmul_precision(original["precision"])

    def test_supervised_loss_decreases_and_fixed_action_dimensions_are_masked(self):
        model = TinyPolicy()
        initial_second = model.linear.weight.detach()[1].clone()
        result = _train_for_steps(
            model,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "single",
            optimizer_steps=120,
        )
        self.assertLess(result.final_loss, result.initial_loss * 0.08)
        torch.testing.assert_close(
            model.linear.weight.detach()[1], initial_second, rtol=0, atol=0
        )
        report = publish_learner_result(
            output_dir=self.root / "single", result=result, config=_config()
        )
        candidate_path = self.root / "single" / report["checkpoint_path"]
        self.assertTrue(candidate_path.is_file())
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        self.assertEqual(candidate["checkpoint_schema"], "gr00t.trl.full_state.v2")
        self.assertIn("policy_state_dict", candidate)
        self.assertEqual(
            candidate["homie_controller_sha256"],
            _config().to_mapping()["model_config"][
                "homie_controller_sha256"
            ],
        )
        self.assertEqual(report["optimizer_steps"], 120)
        self.assertEqual(
            Path(report["manifest_path"]).parent
            / report["policy_spec_path"],
            Path(f"{candidate_path}.policy_spec.json"),
        )
        self.assertTrue(
            (Path(report["manifest_path"]).parent / report["policy_spec_path"]).is_file()
        )
        self.assertTrue(
            (
                Path(report["manifest_path"]).parent
                / report["policy_bundle_manifest_path"]
            ).is_file()
        )

    def test_candidate_publication_rejects_controller_lineage_drift(self):
        result = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "controller-drift",
            optimizer_steps=1,
        )
        checkpoint = torch.load(
            result.restart_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint["homie_controller_sha256"]["walk"] = "f" * 64
        torch.save(checkpoint, result.restart_checkpoint)
        with self.assertRaisesRegex(
            LearnerError, "controllers differ from learner policy spec"
        ):
            publish_learner_result(
                output_dir=self.root / "controller-drift-publish",
                result=result,
                config=_config(),
            )

    def test_attested_candidate_publication_rechecks_saved_execution_identity(self):
        config = _config()
        attestation = self.learner_attestation(config)
        result = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=config,
            output_dir=self.root / "attested",
            optimizer_steps=1,
            **attestation,
        )
        report = publish_learner_result(
            output_dir=self.root / "attested",
            result=result,
            config=config,
            **attestation,
        )
        self.assertEqual(report["schema"], RESULT_SCHEMA)
        for field, expected in attestation.items():
            self.assertEqual(report[field], expected)
        persisted_result = artifacts.load_strict_json(
            self.root / "attested" / "learner_result_round_000000.json"
        )
        self.assertEqual(persisted_result, {
            key: value
            for key, value in report.items()
            if key not in {"manifest_path", "manifest_sha256"}
        })
        checkpoint = torch.load(
            self.root / "attested" / report["checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        for field, expected in attestation.items():
            self.assertEqual(checkpoint[field], expected)

        with self.assertRaisesRegex(
            LearnerError, "runtime_container_sha256"
        ):
            publish_learner_result(
                output_dir=self.root / "attested-mismatch",
                result=result,
                config=config,
                **dict(attestation, runtime_container_sha256="e" * 64),
            )
        self.assertFalse((self.root / "attested-mismatch").exists())

    def test_attested_publication_requires_the_complete_identity_triplet(self):
        config = _config()
        result = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=config,
            output_dir=self.root / "partial-attestation",
            optimizer_steps=1,
        )
        with self.assertRaisesRegex(
            LearnerError, "candidate attestation fields are incomplete"
        ):
            publish_learner_result(
                output_dir=self.root / "partial-attestation-publish",
                result=result,
                config=config,
                learner_identity_sha256="a" * 64,
            )
        self.assertFalse((self.root / "partial-attestation-publish").exists())

    def test_source_mutation_after_private_load_cannot_poison_candidate(self):
        config = _config()
        attestation = self.learner_attestation(config)
        result = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=config,
            output_dir=self.root / "race-source",
            optimizer_steps=1,
            **attestation,
        )
        restart_path = Path(result.restart_checkpoint).resolve()
        restart_sha256 = artifacts.sha256_file(restart_path)
        source_mutated = False
        from round_dagger import learner as learner_module

        original_loader = learner_module.load_restricted_restart_document

        def mutate_source_after_load(path, **kwargs):
            nonlocal source_mutated
            document = original_loader(path, **kwargs)
            if Path(path).resolve() != restart_path and not source_mutated:
                source_mutated = True
                restart_path.chmod(0o644)
                with restart_path.open("ab") as stream:
                    stream.write(b"concurrent mutation")
            return document

        publish_output = self.root / "race-publish"
        with mock.patch(
            "round_dagger.learner.load_restricted_restart_document",
            side_effect=mutate_source_after_load,
        ):
            report = publish_learner_result(
                output_dir=publish_output,
                result=result,
                config=config,
                **attestation,
            )
        self.assertTrue(source_mutated)
        self.assertNotEqual(artifacts.sha256_file(restart_path), restart_sha256)
        self.assertEqual(report["checkpoint_sha256"], restart_sha256)
        self.assertEqual(
            artifacts.sha256_file(publish_output / report["checkpoint_path"]),
            restart_sha256,
        )

    def test_private_snapshot_race_leaves_no_published_artifact(self):
        config = _config()
        attestation = self.learner_attestation(config)
        result = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=config,
            output_dir=self.root / "private-race-source",
            optimizer_steps=1,
            **attestation,
        )
        from round_dagger import learner as learner_module

        original_state_digest = learner_module.runtime_state_dict_sha256
        publish_output = self.root / "private-race-publish"
        private_mutated = False

        def mutate_private_snapshot(policy_state):
            nonlocal private_mutated
            digest = original_state_digest(policy_state)
            snapshots = list(
                publish_output.glob(".candidate_round_000000.pt.snapshot.*")
            )
            self.assertEqual(len(snapshots), 1)
            snapshots[0].chmod(0o600)
            with snapshots[0].open("ab") as stream:
                stream.write(b"private snapshot mutation")
            private_mutated = True
            return digest

        with mock.patch(
            "round_dagger.learner.runtime_state_dict_sha256",
            side_effect=mutate_private_snapshot,
        ):
            with self.assertRaisesRegex(
                LearnerError, "private candidate checkpoint changed"
            ):
                publish_learner_result(
                    output_dir=publish_output,
                    result=result,
                    config=config,
                    **attestation,
                )
        self.assertTrue(private_mutated)
        self.assertFalse(publish_output.exists())

    def test_initial_and_final_losses_use_the_identical_deterministic_probe(self):
        model = ProbeRecordingPolicy()
        result = _train_for_steps(
            model,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "fixed-probe",
            optimizer_steps=7,
        )
        self.assertEqual(len(model.probe_batches), 2)
        self.assertEqual(
            set(model.probe_batches[0]), set(model.probe_batches[1])
        )
        for key in model.probe_batches[0]:
            torch.testing.assert_close(
                model.probe_batches[0][key],
                model.probe_batches[1][key],
                rtol=0,
                atol=0,
            )
        self.assertTrue(np.isfinite(result.initial_loss))
        self.assertTrue(np.isfinite(result.final_loss))
        self.assertNotEqual(result.initial_loss, result.final_loss)

    def test_reduced_width_actor_trains_only_the_expert_action_prefix(self):
        # Mirror the production 11-D VisionActor / 26-D replay contract.
        predictions = torch.zeros((3, 11), requires_grad=True)
        targets = torch.cat((torch.ones((3, 11)), torch.full((3, 15), 10000.0)), dim=-1)
        loss = supervised_action_loss(predictions, targets, (0, 11))
        self.assertEqual(float(loss.detach()), 1.0)
        loss.backward()
        torch.testing.assert_close(
            predictions.grad, torch.full_like(predictions, -2.0 / predictions.numel())
        )
        with self.assertRaisesRegex(LearnerError, "prefix slice"):
            supervised_action_loss(predictions.detach(), targets, (1, 12))

        # The same reduced-output contract works through the full replay loop.
        model = PrefixOnlyPolicy()
        result = _train_for_steps(
            model,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "prefix-only",
            optimizer_steps=100,
        )
        self.assertLess(result.final_loss, result.initial_loss * 0.1)

    def test_restart_matches_uninterrupted_training_exactly(self):
        initial = self._initial_state()
        full = TinyPolicy()
        full.load_state_dict(initial)
        full_result = _train_for_steps(
            full,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "full",
            optimizer_steps=40,
        )

        partial = TinyPolicy()
        partial.load_state_dict(initial)
        partial_result = _train_for_steps(
            partial,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "resumed",
            optimizer_steps=13,
        )
        self.assertEqual(partial_result.completed_steps, 13)
        self.assertEqual(partial_result.target_steps, 13)
        resumed = TinyPolicy()
        resumed_result = _train_for_steps(
            resumed,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "resumed",
            optimizer_steps=40,
            resume_from=partial_result.restart_checkpoint,
        )
        self.assertEqual(full_result.completed_steps, resumed_result.completed_steps)
        self.assertEqual(full_result.initial_loss, partial_result.initial_loss)
        self.assertEqual(full_result.initial_loss, resumed_result.initial_loss)
        self.assertEqual(full_result.final_loss, resumed_result.final_loss)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)

        restart = torch.load(
            resumed_result.restart_checkpoint, map_location="cpu", weights_only=False
        )
        self.assertEqual(restart["completed_steps"], 40)
        self.assertEqual(restart["initial_loss"], full_result.initial_loss)
        self.assertEqual(
            restart["gradient_reduction"], SINGLE_PROCESS_GRADIENT_REDUCTION
        )
        optimizer_group = restart["optimizer_state_dict"]["param_groups"][0]
        self.assertIs(optimizer_group["foreach"], False)
        self.assertIs(optimizer_group["fused"], False)
        self.assertEqual(
            Path(resumed_result.restart_checkpoint).stat().st_mode & 0o777,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        )

    def test_jsonl_telemetry_is_durable_and_step_monotonic(self):
        path = self.root / "telemetry" / "learner.jsonl"
        telemetry = JsonlTelemetry(path, enabled=True)
        try:
            _train_for_steps(
                TinyPolicy(),
                snapshot_path=self.snapshot_path,
                config=_config(),
                output_dir=self.root / "telemetry-run",
                optimizer_steps=3,
                telemetry=telemetry,
            )
        finally:
            telemetry.close()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([item["optimizer_step"] for item in records], [1, 2, 3])
        self.assertTrue(all(np.isfinite(item["loss"]) for item in records))
        self.assertTrue(all(item["macro_round"] == 0 for item in records))
        self.assertTrue(
            all(
                item["schema"]
                == "fast_roborender.round_dagger.learner_telemetry.v3"
                for item in records
            )
        )
        self.assertTrue(
            all(item["generation_id"] == "gen-000-bc" for item in records)
        )
        self.assertTrue(
            all(item["replay_transitions_per_rank"] == [16] for item in records)
        )
        self.assertTrue(
            all(item["replay_transitions_global"] == 16 for item in records)
        )

    def test_jsonl_telemetry_write_failure_preserves_durable_prefix(self):
        path = self.root / "atomic-telemetry" / "learner.jsonl"
        telemetry = JsonlTelemetry(path, enabled=True)
        telemetry.log({"optimizer_step": 1, "loss": 1.0})
        durable = path.read_bytes()
        with (
            mock.patch(
                "round_dagger.learner.os.write",
                side_effect=OSError("simulated scheduler interruption"),
            ),
            self.assertRaisesRegex(OSError, "scheduler interruption"),
        ):
            telemetry.log({"optimizer_step": 2, "loss": 0.5})
        self.assertEqual(path.read_bytes(), durable)

    def test_jsonl_telemetry_rejects_symlink_and_appends_canonical_record(self):
        root = self.root / "unsafe-telemetry"
        root.mkdir()
        target = root / "target.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        link = root / "learner.jsonl"
        link.symlink_to(target)
        with self.assertRaisesRegex(LearnerError, "must not be a symbolic link"):
            JsonlTelemetry(link, enabled=True).log({"optimizer_step": 1})

        link.unlink()
        link.write_text('{"optimizer_step":0}\n', encoding="utf-8")
        JsonlTelemetry(link, enabled=True).log({"optimizer_step": 1})
        self.assertEqual(
            [json.loads(line)["optimizer_step"] for line in link.read_text().splitlines()],
            [0, 1],
        )

    def test_resume_rejects_model_config_and_snapshot_lineage_mismatch(self):
        model = TinyPolicy()
        partial = _train_for_steps(
            model,
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "mismatch",
            optimizer_steps=2,
        )
        with self.assertRaisesRegex(LearnerError, "restart lineage mismatch"):
            _train_for_steps(
                TinyPolicy(),
                snapshot_path=self.snapshot_path,
                config=_config(model_lineage="c" * 64),
                output_dir=self.root / "mismatch-2",
                optimizer_steps=3,
                resume_from=partial.restart_checkpoint,
            )
        tampered = torch.load(partial.restart_checkpoint, weights_only=False)
        tampered["gradient_reduction"] = ORDERED_ALLGATHER_GRADIENT_REDUCTION
        tampered_path = self.root / "wrong-gradient-reduction.pt"
        torch.save(tampered, tampered_path)
        with self.assertRaisesRegex(LearnerError, "gradient_reduction"):
            _train_for_steps(
                TinyPolicy(),
                snapshot_path=self.snapshot_path,
                config=_config(),
                output_dir=self.root / "mismatch-gradient-reduction",
                optimizer_steps=3,
                resume_from=tampered_path,
            )
        tampered = torch.load(partial.restart_checkpoint, weights_only=False)
        tampered["replay_snapshot_sha256"] = "d" * 64
        tampered_path = self.root / "wrong-snapshot.pt"
        torch.save(tampered, tampered_path)
        with self.assertRaisesRegex(LearnerError, "replay_snapshot_sha256"):
            _train_for_steps(
                TinyPolicy(),
                snapshot_path=self.snapshot_path,
                config=_config(),
                output_dir=self.root / "mismatch-4",
                optimizer_steps=3,
                resume_from=tampered_path,
            )
        with self.assertRaisesRegex(LearnerError, "restart lineage mismatch"):
            _train_for_steps(
                TinyPolicy(),
                snapshot_path=self.snapshot_path,
                config=_config(mean=(0.4, 0.5, 0.5)),
                output_dir=self.root / "mismatch-3",
                optimizer_steps=3,
                resume_from=partial.restart_checkpoint,
            )

    def test_resume_rejects_identity_container_or_contract_drift(self):
        identity = "a" * 64
        container = "b" * 64
        contract = "c" * 64
        partial = _train_for_steps(
            TinyPolicy(),
            snapshot_path=self.snapshot_path,
            config=_config(),
            output_dir=self.root / "semantic-restart",
            optimizer_steps=2,
            learner_identity_sha256=identity,
            runtime_container_sha256=container,
            learner_contract_sha256=contract,
        )
        for field, values in {
            "learner_identity_sha256": ("d" * 64, container, contract),
            "runtime_container_sha256": (identity, "e" * 64, contract),
            "learner_contract_sha256": (identity, container, "f" * 64),
        }.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                LearnerError, field
            ):
                _train_for_steps(
                    TinyPolicy(),
                    snapshot_path=self.snapshot_path,
                    config=_config(),
                    output_dir=self.root / f"semantic-restart-{field}",
                    optimizer_steps=3,
                    resume_from=partial.restart_checkpoint,
                    learner_identity_sha256=values[0],
                    runtime_container_sha256=values[1],
                    learner_contract_sha256=values[2],
                )


if __name__ == "__main__":
    unittest.main()
