"""Integration coverage for RTX-to-H200 replay path relocation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts, replay
from scripts.round_dagger import production_lepton_learner_bridge as bridge


CAMPAIGN_ID = "portable-production-replay"
MODEL_DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}
VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [26]}


def _arrays(global_env_ids: list[int]) -> dict[str, np.ndarray]:
    environments = len(global_env_ids)
    ids = np.asarray(global_env_ids, dtype=np.int64)[None, :]
    expert = np.repeat(ids[..., None], 26, axis=-1).astype(np.float32) / 1000.0
    student = expert + np.float32(0.125)
    expert_mask = np.ones((1, environments), dtype=np.bool_)
    return {
        "policy_rgb": np.broadcast_to(
            (ids % 251)[..., None, None, None].astype(np.uint8),
            (1, environments, 6, 4, 3),
        ).copy(),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": expert.copy(),
        "expert_mask": expert_mask,
        "dones": np.zeros((1, environments), dtype=np.bool_),
        "observation_is_raytraced": np.ones((1, environments), dtype=np.bool_),
        "global_env_ids": ids,
        "episode_ids": ids.copy(),
        "episode_steps": np.zeros((1, environments), dtype=np.int64),
        "obs__actor_obs": np.repeat(ids[..., None], 3, axis=-1).astype(np.float32),
    }


def _publish_cohort(
    path: Path, *, cohort_index: int, purpose: str, global_env_ids: list[int]
) -> None:
    provenance = {
        "campaign_id": CAMPAIGN_ID,
        "macro_round": 0,
        "cohort_index": cohort_index,
        "purpose": purpose,
        **MODEL_DIGESTS,
    }
    artifacts.create_cohort(
        path,
        expected_ranks=[0],
        expected_global_env_ids=global_env_ids,
        expected_transitions=len(global_env_ids),
        view_contract=VIEW_CONTRACT,
        action_contract=ACTION_CONTRACT,
        **provenance,
    )
    artifacts.publish_rollout_shard(
        path,
        rank=0,
        rollout_id=0,
        part_index=0,
        arrays=_arrays(global_env_ids),
        **provenance,
    )
    artifacts.seal_cohort(path)


def _production_campaign(root: Path) -> tuple[dict, dict, Path]:
    (root / "policies").mkdir(parents=True)
    (root / "replay/snapshots").mkdir(parents=True)
    accepted = root / "rounds/macro_round_000000/cohorts"
    accepted.mkdir(parents=True)
    (root / "campaign.json").write_text(
        json.dumps({"campaign_id": CAMPAIGN_ID}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "coordinator_state.json").write_text(
        json.dumps({"phase": "INGESTED"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "policies/policy_v0.pt").write_bytes(b"parent-policy")

    training_paths: list[Path] = []
    for cohort_index in range(4):
        cohort = accepted / f"cohort_{cohort_index:02d}"
        _publish_cohort(
            cohort,
            cohort_index=cohort_index,
            purpose="train",
            global_env_ids=[cohort_index],
        )
        training_paths.append(cohort)
    training_path = root / "replay/snapshots/through_macro_round_000000.json"
    with replay.ReplayLedger(root / "replay/replay.sqlite3") as ledger:
        for cohort in training_paths:
            ledger.ingest_cohort(cohort)
        training = ledger.build_snapshot(
            CAMPAIGN_ID,
            through_macro_round=0,
            expected_train_cohorts=4,
            output=training_path,
        )

    heldout_paths: list[Path] = []
    for cohort_index in range(8):
        cohort = root / f"eval_seed_{cohort_index:03d}"
        first = 1000 + cohort_index * 32
        _publish_cohort(
            cohort,
            cohort_index=cohort_index,
            purpose="eval",
            global_env_ids=list(range(first, first + 32)),
        )
        heldout_paths.append(cohort)
    heldout_path = root / "heldout_snapshot.json"
    heldout = replay.build_heldout_snapshot(heldout_paths, output=heldout_path)
    scene_samples = root / "scene_samples/round_000000.json"
    scene_samples.parent.mkdir()
    scene_samples.write_text(
        json.dumps({"campaign_id": CAMPAIGN_ID, "samples": []}) + "\n",
        encoding="utf-8",
    )
    return training, heldout, scene_samples


class PortableProjectionIntegrationTest(unittest.TestCase):
    def test_projection_relocates_four_train_and_eight_heldout_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            rtx_campaign = base / "rtx/campaign"
            rtx_campaign.mkdir(parents=True)
            training, heldout, scene_samples = _production_campaign(rtx_campaign)
            projection = base / "handoff/campaign_input"
            projection.parent.mkdir(parents=True)

            environment = {
                "ROUND_DAGGER_CAMPAIGN_ROOT": str(rtx_campaign),
                "ROUND_DAGGER_CAMPAIGN_ID": CAMPAIGN_ID,
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_COHORT_COUNT": "4",
                "ROUND_DAGGER_REPLAY_SNAPSHOT": training["path"],
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": training["file_sha256"],
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": training["snapshot_sha256"],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": heldout["path"],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256": heldout[
                    "file_sha256"
                ],
                "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256": heldout[
                    "snapshot_sha256"
                ],
                "ROUND_DAGGER_SCENE_SAMPLE_MANIFEST": str(scene_samples),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(
                    bridge.build_campaign_projection(rtx_campaign, projection),
                    "published",
                )
                bridge.validate_projected_inputs(rtx_campaign, projection)
                validated = bridge.validate_projected_replay_snapshots(
                    rtx_campaign, projection
                )

            self.assertEqual(validated["training"]["cohort_count"], 4)
            self.assertEqual(validated["heldout"]["cohort_count"], 8)
            self.assertEqual(validated["heldout"]["transition_count"], 256)
            for snapshot in validated.values():
                for cohort in snapshot["cohorts"]:
                    Path(cohort["cohort_manifest_path"]).relative_to(projection)
                for shard in snapshot["resolved_shards"]:
                    Path(shard["payload_path"]).relative_to(projection)

            # Simulate extraction at a second H200 root, then make the original
            # producer paths unavailable.  Snapshot bytes and semantic identity
            # must stay unchanged while every payload resolves below H200.
            h200_campaign = base / "h200/campaign_input"
            shutil.copytree(projection, h200_campaign)
            rtx_campaign.rename(base / "rtx/campaign-offline")
            h200_training_path = (
                h200_campaign
                / "replay/snapshots/through_macro_round_000000.json"
            )
            loaded_training = replay.load_snapshot(
                h200_training_path,
                expected_file_sha256=training["file_sha256"],
                artifact_root=h200_campaign,
            )
            loaded_heldout = replay.load_heldout_snapshot(
                h200_campaign / "heldout_snapshot.json",
                expected_file_sha256=heldout["file_sha256"],
                artifact_root=h200_campaign,
            )
            self.assertEqual(
                loaded_training["snapshot_sha256"], training["snapshot_sha256"]
            )
            self.assertEqual(
                loaded_heldout["snapshot_sha256"], heldout["snapshot_sha256"]
            )
            for snapshot in (loaded_training, loaded_heldout):
                for cohort in snapshot["cohorts"]:
                    Path(cohort["cohort_manifest_path"]).relative_to(h200_campaign)
                for shard in snapshot["resolved_shards"]:
                    Path(shard["payload_path"]).relative_to(h200_campaign)

    def test_relocation_fails_closed_on_ambiguous_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            training, _heldout, _scene = _production_campaign(source)
            destination = base / "destination"
            shutil.copytree(source, destination)
            snapshot = destination / "replay/snapshots/through_macro_round_000000.json"
            duplicate = destination / "duplicate/cohort_00"
            duplicate.parent.mkdir()
            shutil.copytree(
                destination / "rounds/macro_round_000000/cohorts/cohort_00",
                duplicate,
            )
            source.rename(base / "source-offline")
            with self.assertRaisesRegex(replay.ReplayError, "ambiguous"):
                replay.load_snapshot(
                    snapshot,
                    expected_file_sha256=training["file_sha256"],
                    artifact_root=destination,
                )


if __name__ == "__main__":
    unittest.main()
