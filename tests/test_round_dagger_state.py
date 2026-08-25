"""Lifecycle and fail-closed tests for the round-based DAgger coordinator."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger.artifacts import (
    ArtifactError,
    create_cohort,
    publish_immutable_json,
    publish_rollout_shard,
    seal_cohort,
)
from round_dagger.promotion import (
    PromotionError,
    load_active_policy,
    load_candidate_manifest,
    promote_candidate,
    stage_bootstrap_policy,
    stage_candidate,
    write_evaluation,
)
from round_dagger.replay import ReplayLedger
from round_dagger.state import (
    StateError,
    _replace_state,
    beta_for_macro_round,
    initialize_campaign,
    load_collection_manifest,
    load_state,
    open_macro_round,
    seal_collection,
    transition,
)
from tests.policy_bundle_helpers import publish_policy_bundle


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _campaign(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "campaign"
    source = tmp_path / "bootstrap-source.pt"
    source.write_bytes(b"fresh-policy-v0")
    publish_policy_bundle(source, _sha("runtime-policy-0"))
    bootstrap = stage_bootstrap_policy(
        root,
        checkpoint_source=source,
        code_sha256=_sha("code-v0"),
        policy_config_sha256=_sha("policy-config-v0"),
    )
    initialize_campaign(
        root,
        campaign_id="state-test",
        bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
        bootstrap_policy_path=bootstrap["checkpoint_path"],
        target_environments=4,
        active_environments=1,
        steps_per_environment=2,
        view_names=("ext1", "ext2", "wrist"),
        policy_image_shape=(6, 2, 3),
        action_dim=2,
        teacher_sha256=_sha("teacher"),
        renderer_sha256=_sha("renderer"),
        camera_mapping_sha256=_sha("camera"),
        training_config_sha256=_sha("training"),
        beta_schedule={
            "kind": "linear",
            "start": 0.8,
            "end": 0.2,
            "decay_rounds": 3,
            "rate": 0.99,
        },
    )
    return root


def _arrays(global_env_id: int) -> dict[str, np.ndarray]:
    expert = np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32)
    student = np.array([[[0.5, 0.6]], [[0.7, 0.8]]], dtype=np.float32)
    mask = np.array([[True], [False]], dtype=np.bool_)
    return {
        "policy_rgb": np.full((2, 1, 6, 2, 3), global_env_id, dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": np.where(mask[..., None], expert, student),
        "expert_mask": mask,
        "dones": np.array([[False], [True]], dtype=np.bool_),
        "observation_is_raytraced": np.ones((2, 1), dtype=np.bool_),
        "global_env_ids": np.array([global_env_id], dtype=np.int64),
        "episode_ids": np.zeros((2, 1), dtype=np.int64),
        "episode_steps": np.array([[0], [1]], dtype=np.int64),
        "obs__actor_obs": np.ones((2, 1, 4), dtype=np.float32),
    }


def _collect(root: Path) -> tuple[dict, list[Path]]:
    contract = open_macro_round(root, expected_revision=load_state(root)["revision"])
    cohort_manifests: list[Path] = []
    runtime_sha = contract["policy_runtime_sha256"]
    view_contract = {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": [6, 2, 3],
        "views": ["ext1", "ext2", "wrist"],
    }
    action_contract = {"dtype": "float32", "shape": [2]}
    for cohort_index in range(4):
        cohort_dir = (
            root
            / "rounds"
            / f"macro_round_{contract['macro_round']:06d}"
            / "cohorts"
            / f"cohort_{cohort_index:04d}"
        )
        kwargs = {
            "campaign_id": "state-test",
            "macro_round": contract["macro_round"],
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
            cohort_dir,
            expected_ranks=[0],
            expected_global_env_ids=[cohort_index],
            expected_transitions=2,
            view_contract=view_contract,
            action_contract=action_contract,
            **kwargs,
        )
        publish_rollout_shard(
            cohort_dir,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(cohort_index),
            **kwargs,
        )
        seal_cohort(cohort_dir)
        cohort_manifests.append(cohort_dir / "seal.json")
    collection = seal_collection(
        root,
        macro_round=contract["macro_round"],
        cohort_manifests=cohort_manifests,
    )
    return collection, cohort_manifests


def _to_ingested(root: Path) -> tuple[dict, Path]:
    collection, cohort_manifests = _collect(root)
    macro_round = collection["macro_round"]
    transition(
        root,
        expected_phase="COLLECTING",
        next_phase="COLLECTED",
        macro_round=macro_round,
        evidence_path=collection["manifest_path"],
        evidence_sha256=collection["manifest_sha256"],
    )
    ledger_path = root / "replay" / "ledger.sqlite3"
    snapshot_path = root / "replay" / f"snapshot_{macro_round:06d}.json"
    with ReplayLedger(ledger_path) as ledger:
        for path in cohort_manifests:
            ledger.ingest_cohort(path)
        snapshot = ledger.build_snapshot(
            "state-test",
            through_macro_round=macro_round,
            output=snapshot_path,
            expected_train_cohorts=4,
        )
    transition(
        root,
        expected_phase="COLLECTED",
        next_phase="INGESTED",
        macro_round=macro_round,
        evidence_path=snapshot_path,
        evidence_sha256=snapshot["file_sha256"],
    )
    return snapshot, snapshot_path


def _stage_and_bind(root: Path, tmp_path: Path) -> dict:
    state = load_state(root)
    candidate_source = tmp_path / f"candidate-r{state['open_macro_round']}.pt"
    candidate_source.write_bytes(f"candidate-{state['open_macro_round']}".encode())
    publish_policy_bundle(
        candidate_source, _sha(f"runtime-policy-{state['open_macro_round'] + 1}")
    )
    candidate = stage_candidate(
        root,
        macro_round=state["open_macro_round"],
        checkpoint_source=candidate_source,
        replay_snapshot_sha256=state["replay_snapshot_sha256"],
        code_sha256=_sha("learner-code"),
        learner_config_sha256=_sha("learner-config"),
        optimizer_steps=7,
    )
    transition(
        root,
        expected_phase="INGESTED",
        next_phase="TRAINED",
        macro_round=candidate["macro_round"],
        evidence_path=candidate["manifest_path"],
        evidence_sha256=candidate["manifest_sha256"],
        candidate_policy={
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
        },
    )
    return candidate


def _evaluate_and_bind(root: Path, candidate: dict, *, passed: bool) -> dict:
    bundle_path = (
        root
        / "rounds"
        / f"macro_round_{candidate['macro_round']:06d}"
        / "evaluation"
        / "results"
        / "evaluation_bundle.json"
    )
    bundle_sha = publish_immutable_json(
        bundle_path,
        {
            "schema": "test.evaluation_bundle.v1",
            "candidate_manifest_sha256": candidate["manifest_sha256"],
        },
    )
    evidence = write_evaluation(
        root,
        candidate_manifest=candidate["manifest_path"],
        passed=passed,
        metrics={"success_rate": 0.75, "action_mse": 0.01},
        evaluator_config_sha256=_sha("evaluator"),
        evidence_bundle_path=bundle_path.relative_to(root).as_posix(),
        evidence_bundle_sha256=bundle_sha,
    )
    transition(
        root,
        expected_phase="TRAINED",
        next_phase="EVALUATED",
        macro_round=candidate["macro_round"],
        evidence_path=evidence["manifest_path"],
        evidence_sha256=evidence["evidence_sha256"],
    )
    return evidence


class StatePromotionLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_campaign_initialization_and_beta_freeze(self) -> None:
        root = _campaign(self.tmp_path)
        initial = load_state(root)
        self.assertEqual(initial["phase"], "READY")
        self.assertEqual(initial["next_macro_round"], 0)
        contract = open_macro_round(root, expected_revision=0)
        self.assertEqual(contract["campaign_id"], "state-test")
        self.assertEqual(contract["macro_round"], 0)
        self.assertEqual(contract["policy_runtime_sha256"], _sha("runtime-policy-0"))
        self.assertAlmostEqual(contract["beta"], 0.8)
        self.assertEqual(
            [item["cohort_index"] for item in contract["cohorts"]], [0, 1, 2, 3]
        )
        reloaded = load_state(root)
        self.assertEqual(reloaded["open_macro_round"], 0)
        self.assertEqual(reloaded["phase"], "COLLECTING")
        self.assertAlmostEqual(
            beta_for_macro_round(
                {"kind": "linear", "start": 0.8, "end": 0.2, "decay_rounds": 3},
                1,
            ),
            0.6,
        )

    def test_transition_ordering_and_revision_cas_fail_closed(self) -> None:
        root = _campaign(self.tmp_path)
        stale = load_state(root)
        collection, _ = _collect(root)
        with self.assertRaisesRegex(StateError, "unsupported"):
            transition(
                root,
                expected_phase="COLLECTING",
                next_phase="INGESTED",
                macro_round=0,
                evidence_path=collection["manifest_path"],
            )
        with self.assertRaisesRegex(StateError, "compare-and-swap"):
            _replace_state(root, stale, expected_revision=stale["revision"])
        with self.assertRaisesRegex(StateError, "revision changed"):
            open_macro_round(root, expected_revision=0)

    def test_collection_manifest_revalidates_all_cohorts(self) -> None:
        root = _campaign(self.tmp_path)
        collection, _ = _collect(root)
        verified = load_collection_manifest(
            root, collection["manifest_path"], expected_macro_round=0
        )
        self.assertEqual(verified["cohort_count"], 4)
        self.assertEqual(verified["transition_count"], 8)
        self.assertEqual(
            [item["cohort_index"] for item in verified["cohorts"]], [0, 1, 2, 3]
        )
        altered = dict(json.loads((root / collection["manifest_path"]).read_text()))
        altered["cohorts"] = altered["cohorts"][:-1]
        forged = root / "rounds" / "macro_round_000000" / "forged_collection.json"
        forged.write_text(json.dumps(altered))
        with self.assertRaises(StateError):
            load_collection_manifest(root, forged, expected_macro_round=0)

    def test_candidate_publish_crash_window_is_idempotent(self) -> None:
        root = _campaign(self.tmp_path)
        _to_ingested(root)
        state = load_state(root)
        source = self.tmp_path / "learner.pt"
        source.write_bytes(b"learner-result")
        publish_policy_bundle(source, _sha("runtime-policy-1"))
        kwargs = {
            "macro_round": 0,
            "checkpoint_source": source,
            "replay_snapshot_sha256": state["replay_snapshot_sha256"],
            "code_sha256": _sha("learner-code"),
            "learner_config_sha256": _sha("learner-config"),
            "optimizer_steps": 9,
        }
        first = stage_candidate(root, **kwargs)
        self.assertEqual(load_state(root)["phase"], "INGESTED")
        second = stage_candidate(root, **kwargs)
        self.assertEqual(second, first)
        self.assertEqual(
            load_candidate_manifest(root, first["manifest_path"])["manifest_sha256"],
            first["manifest_sha256"],
        )

    def test_policy_sidecars_are_required_and_staged_at_canonical_names(self) -> None:
        missing_root = self.tmp_path / "missing-sidecar-campaign"
        missing_source = self.tmp_path / "missing-sidecar.pt"
        missing_source.write_bytes(b"no-sidecars")
        with self.assertRaises((PromotionError, ArtifactError, StateError, RuntimeError)):
            stage_bootstrap_policy(
                missing_root,
                checkpoint_source=missing_source,
                code_sha256=_sha("code"),
                policy_config_sha256=_sha("config"),
            )
        self.assertFalse(
            (missing_root / "policies" / "policy_000000" / "bootstrap.json").exists()
        )

        source = self.tmp_path / "renamed-source.pt"
        source.write_bytes(b"complete-bundle")
        runtime_sha = _sha("canonical-runtime")
        publish_policy_bundle(source, runtime_sha)
        root = self.tmp_path / "canonical-sidecar-campaign"
        bootstrap = stage_bootstrap_policy(
            root,
            checkpoint_source=source,
            code_sha256=_sha("code"),
            policy_config_sha256=_sha("config"),
        )
        checkpoint = root / bootstrap["checkpoint_path"]
        self.assertTrue(Path(f"{checkpoint}.policy_spec.json").is_file())
        self.assertTrue(Path(f"{checkpoint}.manifest.json").is_file())
        self.assertEqual(bootstrap["runtime_policy_sha256"], runtime_sha)
        self.assertEqual(
            bootstrap["policy_spec_path"],
            "policies/policy_000000/checkpoint.pt.policy_spec.json",
        )
        self.assertEqual(
            bootstrap["policy_bundle_manifest_path"],
            "policies/policy_000000/checkpoint.pt.manifest.json",
        )

    def test_corrupt_and_symlinked_candidates_are_rejected(self) -> None:
        root = _campaign(self.tmp_path)
        _to_ingested(root)
        candidate = _stage_and_bind(root, self.tmp_path)
        checkpoint = root / candidate["checkpoint_path"]
        checkpoint.chmod(0o600)
        checkpoint.write_bytes(b"corrupt")
        with self.assertRaisesRegex((PromotionError, ArtifactError), "corrupt|SHA"):
            load_candidate_manifest(root, candidate["manifest_path"])

        other_root = _campaign(self.tmp_path / "other")
        _to_ingested(other_root)
        outside = self.tmp_path / "outside-candidate.json"
        outside.write_text("{}")
        symlink = other_root / "policies" / "policy_000001" / "candidate.json"
        symlink.parent.mkdir(parents=True)
        symlink.symlink_to(outside)
        with self.assertRaises((PromotionError, ArtifactError, StateError)):
            load_candidate_manifest(other_root, symlink)

    def test_failed_evaluation_is_never_promoted(self) -> None:
        root = _campaign(self.tmp_path)
        _to_ingested(root)
        candidate = _stage_and_bind(root, self.tmp_path)
        evaluation = _evaluate_and_bind(root, candidate, passed=False)
        self.assertEqual(load_state(root)["phase"], "EVALUATED")
        with self.assertRaisesRegex(PromotionError, "failed"):
            promote_candidate(
                root,
                macro_round=0,
                candidate_manifest=candidate["manifest_path"],
                evaluation_manifest=evaluation["manifest_path"],
            )
        self.assertEqual(load_active_policy(root)["version"], 0)

    def test_atomic_promotion_and_exact_retry_survive_reload(self) -> None:
        root = _campaign(self.tmp_path)
        _to_ingested(root)
        candidate = _stage_and_bind(root, self.tmp_path)
        evaluation = _evaluate_and_bind(root, candidate, passed=True)
        promotion = promote_candidate(
            root,
            macro_round=0,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )
        state = load_state(root)
        self.assertEqual(state["phase"], "READY")
        self.assertEqual(state["next_macro_round"], 1)
        self.assertEqual(state["active_policy"]["version"], 1)
        self.assertEqual(
            state["active_policy"]["checkpoint_sha256"], candidate["checkpoint_sha256"]
        )
        self.assertEqual(state["last_promotion_sha256"], promotion["promotion_sha256"])
        self.assertEqual(
            load_active_policy(root)["checkpoint_sha256"],
            candidate["checkpoint_sha256"],
        )
        retry = promote_candidate(
            root,
            macro_round=0,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )
        self.assertEqual(retry, promotion)
        self.assertEqual(load_state(root)["revision"], state["revision"])
        next_contract = open_macro_round(root, expected_revision=state["revision"])
        self.assertEqual(next_contract["macro_round"], 1)
        self.assertEqual(next_contract["policy_version"], 1)
        self.assertEqual(
            next_contract["policy_runtime_sha256"],
            candidate["runtime_policy_sha256"],
        )
        self.assertAlmostEqual(next_contract["beta"], 0.6)

    def test_stale_candidate_cannot_promote_after_next_round_opens(self) -> None:
        root = _campaign(self.tmp_path)
        _to_ingested(root)
        candidate = _stage_and_bind(root, self.tmp_path)
        evaluation = _evaluate_and_bind(root, candidate, passed=True)
        promote_candidate(
            root,
            macro_round=0,
            candidate_manifest=candidate["manifest_path"],
            evaluation_manifest=evaluation["manifest_path"],
        )
        open_macro_round(root)
        with self.assertRaises(PromotionError):
            promote_candidate(
                root,
                macro_round=0,
                candidate_manifest=candidate["manifest_path"],
                evaluation_manifest=evaluation["manifest_path"],
            )

    def test_bootstrap_source_symlink_is_rejected(self) -> None:
        source = self.tmp_path / "source.pt"
        source.write_bytes(b"base")
        link = self.tmp_path / "source-link.pt"
        link.symlink_to(source)
        with self.assertRaises((PromotionError, ArtifactError, StateError)):
            stage_bootstrap_policy(
                self.tmp_path / "campaign",
                checkpoint_source=link,
                code_sha256=_sha("code"),
                policy_config_sha256=_sha("config"),
            )


if __name__ == "__main__":
    unittest.main()
