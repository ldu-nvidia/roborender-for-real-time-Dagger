import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from round_dagger.artifacts import canonical_json_sha256, publish_immutable_json
from round_dagger.learner import (
    BUNDLE_SCHEMA,
    CumulativeDaggerBatchSchedule,
    LearnerBundleConfig,
    LearnerError,
    optimizer_exposure_summary,
    train_from_replay,
)
from round_dagger.replay import (
    SNAPSHOT_SCHEMA,
    VALIDATION_SNAPSHOT_SCHEMA,
    ReplayError,
    load_snapshot,
    load_validation_snapshot,
)
from scripts.round_dagger import train_learner


def _bundle(*, macro_round: int = 0, batch_size: int = 8) -> LearnerBundleConfig:
    return LearnerBundleConfig.from_mapping(
        {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": "dagger-redo-v001",
            "generation_id": (
                "gen-000-bc"
                if macro_round == 0
                else {
                    1: "gen-001-dagger-beta-0p80",
                    2: "gen-002-dagger-beta-0p60",
                }[macro_round]
            ),
            "macro_round": macro_round,
            "model_lineage_sha256": "a" * 64,
            "code_sha256": "b" * 64,
            "rgb_mean": [0.5, 0.5, 0.5],
            "rgb_std": [0.25, 0.25, 0.25],
            "batch_size": batch_size,
            "model_config": {},
        }
    )


class TrainBoundaryTest(unittest.TestCase):
    def test_bundle_rejects_every_validation_or_test_key(self):
        for key in (
            "validation_manifest",
            "validation_snapshot_sha256",
            "test_manifest",
            "heldout_snapshot_sha256",
            "eval_interval",
        ):
            raw = _bundle().to_mapping()
            raw[key] = "x"
            with self.subTest(key=key), self.assertRaisesRegex(
                LearnerError, "train-only learner bundle forbids"
            ):
                LearnerBundleConfig.from_mapping(raw)

    def test_training_api_has_no_validation_or_test_surface(self):
        names = set(inspect.signature(train_from_replay).parameters)
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("validation", "heldout", "test", "eval")
            )
        )

    def test_training_cli_rejects_validation_arguments(self):
        argv = [
            "--bundle",
            "bundle.json",
            "--snapshot",
            "train.json",
            "--output-dir",
            "out",
            "--model-factory",
            "module:factory",
            "--target-optimizer-examples",
            "64",
            "--checkpoint-every-optimizer-examples",
            "64",
            "--validation-snapshot",
            "validation.json",
        ]
        with self.assertRaises(SystemExit):
            train_learner._parser().parse_args(argv)

    def test_learner_launcher_masks_heldout_in_private_mount_namespace(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_8rank_learner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CAMPAIGN=/workspace/project/roborender/data", launcher)
        self.assertIn("unshare --mount --propagation private", launcher)
        self.assertIn('"$CAMPAIGN/validation" "$CAMPAIGN/test"', launcher)
        self.assertIn('mount --bind "$mask_root" "$target"', launcher)
        self.assertIn(
            "--bounding-set=-sys_admin,-dac_override,-dac_read_search", launcher
        )
        self.assertIn("[[ ! -r $target && ! -x $target ]]", launcher)
        self.assertIn('"${masked_targets[@]}"', launcher)

    def test_bc_schedule_is_all_bc_and_uniform_per_epoch(self):
        partition = SimpleNamespace(
            config=SimpleNamespace(macro_round=0),
            newest_round_indices=tuple(range(8)),
            cumulative_prior_indices=(),
        )
        schedule = CumulativeDaggerBatchSchedule(partition, 4, seed=11)
        selected = schedule.indices(0) + schedule.indices(1)
        self.assertEqual(sorted(selected), list(range(8)))
        self.assertEqual(
            schedule.source_counts(2),
            {"newest_source_examples": 8, "prior_source_examples": 0},
        )

    def test_dagger_batch_is_exact_half_and_each_source_is_uniform(self):
        newest = tuple(range(8))
        prior = tuple(range(100, 112))
        partition = SimpleNamespace(
            config=SimpleNamespace(macro_round=2),
            newest_round_indices=newest,
            cumulative_prior_indices=prior,
        )
        schedule = CumulativeDaggerBatchSchedule(partition, 4, seed=19)
        for step in range(12):
            batch = schedule.indices(step)
            self.assertEqual(sum(index in newest for index in batch), 2)
            self.assertEqual(sum(index in prior for index in batch), 2)
        newest_epoch = [
            index
            for step in range(4)
            for index in schedule.indices(step)
            if index in newest
        ]
        prior_epoch = [
            index
            for step in range(6)
            for index in schedule.indices(step)
            if index in prior
        ]
        self.assertEqual(sorted(newest_epoch), list(newest))
        self.assertEqual(sorted(prior_epoch), list(prior))

    def test_dagger_odd_batch_is_rejected(self):
        partition = SimpleNamespace(
            config=SimpleNamespace(macro_round=1),
            newest_round_indices=(0,),
            cumulative_prior_indices=(1,),
        )
        with self.assertRaisesRegex(LearnerError, "must be even"):
            CumulativeDaggerBatchSchedule(partition, 3, seed=0)

    def test_optimizer_exposure_uses_population_scaled_balanced_units(self):
        config = _bundle(macro_round=2, batch_size=8)
        summary = optimizer_exposure_summary(
            {"counts_by_round": {"0": 100, "1": 300, "2": 250}},
            config,
            completed_steps=100,
            world_size=2,
        )
        self.assertEqual(summary["optimizer_examples"], 1600)
        self.assertEqual(summary["newest_source_examples"], 800)
        self.assertEqual(summary["prior_source_examples"], 800)
        self.assertEqual(summary["balanced_exposure_unit_examples"], 800)
        self.assertEqual(summary["balanced_exposure_units"], 2.0)

    def test_train_loader_rejects_non_train_snapshot_before_payload_load(self):
        document = {
            "schema": SNAPSHOT_SCHEMA,
            "purpose": "test",
            "campaign_id": "boundary-test",
            "through_macro_round": 0,
            "cohort_counts_by_round": {"0": 1},
            "cohort_manifest_schema": "fast_roborender.round_dagger.cohort.v1",
            "view_contract": {},
            "action_contract": {},
            "cohort_count": 0,
            "transition_count": 0,
            "counts_by_round": {},
            "counts_by_round_cohort": {},
            "cohorts": [],
        }
        document["snapshot_sha256"] = canonical_json_sha256(document)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "not-train.json"
            publish_immutable_json(path, document)
            with self.assertRaisesRegex(ReplayError, "purpose.*train"):
                load_snapshot(path)

    def test_validation_loader_rejects_test_purpose(self):
        document = {
            "schema": VALIDATION_SNAPSHOT_SCHEMA,
            "purpose": "test",
            "split_sha256": "c" * 64,
            "campaign_id": "boundary-test",
            "cohort_manifest_schema": "fast_roborender.round_dagger.cohort.v1",
            "view_contract": {},
            "action_contract": {},
            "cohort_count": 0,
            "transition_count": 0,
            "cohorts": [],
        }
        document["snapshot_sha256"] = canonical_json_sha256(document)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "test.json"
            publish_immutable_json(path, document)
            with self.assertRaisesRegex(ReplayError, "purpose.*validation"):
                load_validation_snapshot(path)


if __name__ == "__main__":
    unittest.main()
