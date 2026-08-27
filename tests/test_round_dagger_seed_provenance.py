"""Regression tests for disjoint actor RNG and trajectory provenance."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from round_dagger import artifacts, orchestration, replay, state
from round_dagger.seed_policy import (
    ActorSeedError,
    derive_actor_seed_contract,
    validate_actor_seed_contract,
)
from tests.test_round_dagger_state import _arrays as state_arrays
from tests.test_round_dagger_state import _campaign


ROOT = Path(__file__).resolve().parents[1]
ACTOR_SCRIPT = ROOT / "scripts/round_dagger/run_actor_cohort.sh"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [2]}
PROVENANCE = {
    "policy_sha256": _sha("policy"),
    "runtime_policy_sha256": _sha("runtime-policy"),
    "teacher_sha256": _sha("teacher"),
    "renderer_sha256": _sha("renderer"),
    "beta_sha256": _sha("beta"),
    "config_sha256": _sha("config"),
}


def _trajectory(
    global_env_id: int, *, rgb_value: int, episode_id: int, actor_offset: float = 0.0
) -> dict[str, np.ndarray]:
    expert = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]], dtype=np.float32)
    student = expert + np.float32(0.25)
    expert_mask = np.ones((3, 1), dtype=np.bool_)
    return {
        "policy_rgb": np.full((3, 1, 6, 4, 3), rgb_value, dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": expert.copy(),
        "expert_mask": expert_mask,
        "dones": np.array([[False], [False], [True]], dtype=np.bool_),
        "observation_is_raytraced": np.ones((3, 1), dtype=np.bool_),
        "global_env_ids": np.array([global_env_id], dtype=np.int64),
        "episode_ids": np.full((3, 1), episode_id, dtype=np.int64),
        "episode_steps": np.arange(3, dtype=np.int64)[:, None],
        "obs__actor_obs": np.full(
            (3, 1, 4), 1.0 + actor_offset, dtype=np.float32
        ),
    }


def _publish_cohort(
    root: Path,
    *,
    campaign_id: str,
    macro_round: int,
    cohort_index: int,
    global_env_id: int,
    arrays: dict[str, np.ndarray],
    seeded: bool = True,
) -> tuple[Path, dict]:
    cohort = root / f"cohort_{cohort_index:02d}"
    identity = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "cohort_index": cohort_index,
        "purpose": "train",
    }
    seed_contract = None
    if seeded:
        seed_contract = derive_actor_seed_contract(
            campaign_id=campaign_id,
            macro_round=macro_round,
            cohort_index=cohort_index,
            expected_ranks=[0],
            expected_global_env_ids=[global_env_id],
        )
    artifacts.create_cohort(
        cohort,
        expected_ranks=[0],
        expected_global_env_ids=[global_env_id],
        expected_transitions=3,
        view_contract=VIEW_CONTRACT,
        action_contract=ACTION_CONTRACT,
        actor_seed_contract=seed_contract,
        **identity,
        **PROVENANCE,
    )
    artifacts.publish_rollout_shard(
        cohort,
        rank=0,
        rollout_id=0,
        part_index=0,
        arrays=arrays,
        **identity,
        **PROVENANCE,
    )
    return cohort, artifacts.seal_cohort(cohort)


class ActorSeedPolicyTest(unittest.TestCase):
    def test_four_cohorts_are_deterministic_and_disjoint(self) -> None:
        contracts = [
            derive_actor_seed_contract(
                campaign_id="production-campaign",
                macro_round=3,
                cohort_index=cohort,
                expected_ranks=[0, 1],
                expected_global_env_ids=list(range(cohort * 64, (cohort + 1) * 64)),
            )
            for cohort in range(4)
        ]
        self.assertEqual(
            contracts[2],
            derive_actor_seed_contract(
                campaign_id="production-campaign",
                macro_round=3,
                cohort_index=2,
                expected_ranks=[0, 1],
                expected_global_env_ids=list(range(128, 192)),
            ),
        )
        simulator = [
            assignment["simulator_seed"]
            for contract in contracts
            for assignment in contract["assignments"]
        ]
        generation = [
            assignment["generation_seed"]
            for contract in contracts
            for assignment in contract["assignments"]
        ]
        self.assertEqual(len(simulator), len(set(simulator)))
        self.assertEqual(len(generation), len(set(generation)))
        changed = dict(contracts[0])
        changed["simulator_base_seed"] += 1
        with self.assertRaisesRegex(ActorSeedError, "deterministic derivation"):
            validate_actor_seed_contract(
                changed,
                campaign_id="production-campaign",
                macro_round=3,
                cohort_index=0,
                expected_ranks=[0, 1],
                expected_global_env_ids=list(range(64)),
            )

    def test_actor_command_binds_both_runtime_seed_paths(self) -> None:
        source = ACTOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('seed="$ROUND_DAGGER_SIMULATOR_BASE_SEED"', source)
        self.assertIn(
            'algo.config.videogen.seed="$ROUND_DAGGER_GENERATION_SEED"', source
        )
        self.assertIn("validate_actor_seed_contract", source)
        self.assertIn("simulator seed does not match trainer rank-offset", source)


class SeededArtifactReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_seed_and_trajectory_provenance_survives_snapshot(self) -> None:
        cohort, seal = _publish_cohort(
            self.root,
            campaign_id="seeded-replay",
            macro_round=0,
            cohort_index=0,
            global_env_id=0,
            arrays=_trajectory(0, rgb_value=1, episode_id=7),
        )
        self.assertIn("actor_seed_contract", seal)
        self.assertEqual(
            set(seal["physical_teacher_trajectory_sha256_by_global_env"]), {"0"}
        )
        snapshot_path = self.root / "snapshot.json"
        with replay.ReplayLedger(self.root / "replay.sqlite3") as ledger:
            ledger.ingest_cohort(cohort)
            snapshot = ledger.build_snapshot(
                "seeded-replay",
                through_macro_round=0,
                expected_train_cohorts=1,
                output=snapshot_path,
            )
        self.assertEqual(snapshot["cohorts"][0]["actor_seed_contract"], seal["actor_seed_contract"])
        loaded = replay.load_snapshot(
            snapshot_path, expected_file_sha256=snapshot["file_sha256"]
        )
        self.assertEqual(
            loaded["cohorts"][0][
                "physical_teacher_trajectory_sha256_by_global_env"
            ],
            seal["physical_teacher_trajectory_sha256_by_global_env"],
        )

    def test_replay_rejects_same_physics_with_new_rgb_id_and_episode_id(self) -> None:
        first, first_seal = _publish_cohort(
            self.root,
            campaign_id="duplicate-gate",
            macro_round=0,
            cohort_index=0,
            global_env_id=0,
            arrays=_trajectory(0, rgb_value=1, episode_id=7),
        )
        second, second_seal = _publish_cohort(
            self.root,
            campaign_id="duplicate-gate",
            macro_round=0,
            cohort_index=1,
            global_env_id=1,
            arrays=_trajectory(1, rgb_value=223, episode_id=999),
        )
        first_digest = next(
            iter(first_seal["physical_teacher_trajectory_sha256_by_global_env"].values())
        )
        second_digest = next(
            iter(second_seal["physical_teacher_trajectory_sha256_by_global_env"].values())
        )
        self.assertEqual(first_digest, second_digest)
        with replay.ReplayLedger(self.root / "duplicate.sqlite3") as ledger:
            ledger.ingest_cohort(first)
            with self.assertRaisesRegex(
                replay.ReplayError, "repeated physical/teacher trajectories"
            ):
                ledger.ingest_cohort(second)

    def test_legacy_unseeded_artifacts_remain_readable(self) -> None:
        cohort, seal = _publish_cohort(
            self.root,
            campaign_id="legacy-replay",
            macro_round=0,
            cohort_index=0,
            global_env_id=0,
            arrays=_trajectory(0, rgb_value=1, episode_id=7),
            seeded=False,
        )
        self.assertNotIn("actor_seed_contract", seal)
        snapshot_path = self.root / "legacy.json"
        with replay.ReplayLedger(self.root / "legacy.sqlite3") as ledger:
            ledger.ingest_cohort(cohort)
            snapshot = ledger.build_snapshot(
                "legacy-replay",
                through_macro_round=0,
                expected_train_cohorts=1,
                output=snapshot_path,
            )
        self.assertNotIn("actor_seed_contract", snapshot["cohorts"][0])
        replay.load_snapshot(
            snapshot_path, expected_file_sha256=snapshot["file_sha256"]
        )


class CollectionDuplicateGateTest(unittest.TestCase):
    def test_collection_seal_rejects_repeated_physical_teacher_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            campaign_root = _campaign(parent)
            contract = state.open_macro_round(
                campaign_root,
                expected_revision=state.load_state(campaign_root)["revision"],
            )
            paths = []
            for cohort_index in range(4):
                cohort = campaign_root / f"seeded_state_cohort_{cohort_index}"
                identity = {
                    "campaign_id": "state-test",
                    "macro_round": 0,
                    "cohort_index": cohort_index,
                    "purpose": "train",
                    "policy_sha256": contract["policy_checkpoint_sha256"],
                    "runtime_policy_sha256": contract["policy_runtime_sha256"],
                    "teacher_sha256": contract["teacher_sha256"],
                    "renderer_sha256": contract["renderer_sha256"],
                    "beta_sha256": contract["beta_sha256"],
                    "config_sha256": contract["training_config_sha256"],
                }
                seed_contract = derive_actor_seed_contract(
                    campaign_id="state-test",
                    macro_round=0,
                    cohort_index=cohort_index,
                    expected_ranks=[0],
                    expected_global_env_ids=[cohort_index],
                )
                artifacts.create_cohort(
                    cohort,
                    expected_ranks=[0],
                    expected_global_env_ids=[cohort_index],
                    expected_transitions=2,
                    view_contract={
                        "dtype": "uint8",
                        "layout": "vertical",
                        "shape": [6, 2, 3],
                        "views": ["ext1", "ext2", "wrist"],
                    },
                    action_contract={"dtype": "float32", "shape": [2]},
                    actor_seed_contract=seed_contract,
                    **identity,
                )
                artifacts.publish_rollout_shard(
                    cohort,
                    rank=0,
                    rollout_id=0,
                    part_index=0,
                    arrays=state_arrays(cohort_index),
                    **identity,
                )
                artifacts.seal_cohort(cohort)
                paths.append(cohort / "seal.json")
            with self.assertRaisesRegex(
                state.StateError, "repeated physical/teacher trajectory"
            ):
                state.seal_collection(
                    campaign_root, macro_round=0, cohort_manifests=paths
                )

    def test_production_backend_always_creates_seeded_cohort_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            backend = orchestration.ProductionBackend(
                SimpleNamespace(campaign_root=root, macro_round=0)
            )
            cohort = root / "cohort_00"
            backend.ensure_cohort_contract(
                cohort,
                campaign={"campaign_id": "production-seed-gate"},
                contract={
                    "policy_image_shape": [324, 192, 3],
                    "view_names": ["head_color", "hand_left", "hand_right"],
                    "action_dim": 26,
                    "policy_checkpoint_sha256": _sha("policy"),
                    "policy_runtime_sha256": _sha("runtime"),
                    "teacher_sha256": _sha("teacher"),
                    "renderer_sha256": _sha("renderer"),
                    "beta_sha256": _sha("beta"),
                    "training_config_sha256": _sha("training"),
                },
                cohort_index=0,
            )
            contract = artifacts.load_strict_json(cohort / "contract.json")
            self.assertIn("actor_seed_contract", contract)
            self.assertEqual(
                len(contract["actor_seed_contract"]["assignments"]), 2
            )


if __name__ == "__main__":
    unittest.main()
