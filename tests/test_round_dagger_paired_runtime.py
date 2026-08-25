"""Unit tests for the trainer-side paired evaluation evidence accumulator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "GR00T-VisualSim2Real"
    / "gr00t"
    / "rl"
    / "utils"
    / "round_dagger_paired_eval.py"
)
SPEC = importlib.util.spec_from_file_location("round_dagger_paired_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
paired_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired_runtime)

PairedEvaluationEvidence = paired_runtime.PairedEvaluationEvidence
PairedEvaluationEvidenceError = paired_runtime.PairedEvaluationEvidenceError
POLICY_VIEWS = paired_runtime.POLICY_VIEWS
RUNTIME_SCHEMA = paired_runtime.RUNTIME_SCHEMA
generated_view_sha256 = paired_runtime.generated_view_sha256


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract(num_envs: int = 2, seed: int = 17) -> dict:
    return {
        "schema": paired_runtime.CONTRACT_SCHEMA,
        "runtime": {
            "policy_views": list(POLICY_VIEWS),
            "view_layout": "vertical",
            "raytraced_mix_prob": 0.0,
            "reference_lag": 256,
            "training_enabled": False,
            "one_episode_per_environment": True,
        },
        "num_envs_per_seed": num_envs,
        "policies": [
            {"role": "parent", "runtime_policy_sha256": _sha("parent")},
            {"role": "candidate", "runtime_policy_sha256": _sha("candidate")},
        ],
        "scenario_cases": [
            {"seed": seed, "env_index": env, "scenario_id": _sha(f"{seed}:{env}")}
            for env in range(num_envs)
        ],
        "evaluation_config_sha256": _sha("config"),
        "evaluator_code_sha256": _sha("evaluator"),
    }


def _accumulator(num_envs: int = 2, seed: int = 17) -> PairedEvaluationEvidence:
    return PairedEvaluationEvidence.from_contract(
        _contract(num_envs, seed),
        num_envs=num_envs,
        generation_seed=seed,
        runtime_policy_sha256=_sha("candidate"),
        policy_views=POLICY_VIEWS,
        view_layout="vertical",
        raytraced_mix_prob=0.0,
        reference_lag=256,
        training_enabled=False,
        eval_relabel_enabled=True,
        one_episode_per_environment=True,
    )


def _views(step: int, num_envs: int = 2) -> dict[int, dict[str, np.ndarray]]:
    return {
        env: {
            view: np.full((2, 3, 3), step * 20 + env * 3 + index, dtype=np.uint8)
            for index, view in enumerate(POLICY_VIEWS)
        }
        for env in range(num_envs)
    }


class PairedEvaluationRuntimeTest(unittest.TestCase):
    def test_complete_evidence_emits_exact_gate_structure(self) -> None:
        evidence = _accumulator()
        first_views = _views(0)
        evidence.record_generation_step(first_views)
        evidence.record_generation_step(_views(1))
        evidence.record_action_batch(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            np.array([[1.0, 1.0], [0.0, 3.0]], dtype=np.float32),
        )
        evidence.record_action_batch(
            np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
            np.array([[0.0, 2.0], [4.0, 2.0]], dtype=np.float32),
        )
        evidence.record_episode_outcome(
            1,
            scenario_id=_sha("17:1"),
            success=False,
            safety_violation=True,
            episode_length=2,
        )
        evidence.record_episode_outcome(
            0,
            scenario_id=_sha("17:0"),
            success=True,
            safety_violation=False,
            episode_length=2,
        )

        result = evidence.finalize()
        self.assertEqual(
            set(result),
            {
                "_roborender_paired_evaluation",
                "generated_policy_steps",
                "generated_frames",
                "_roborender_paired_episodes",
            },
        )
        self.assertEqual(result["generated_policy_steps"], 2)
        self.assertEqual(result["generated_frames"], 4)
        runtime = result["_roborender_paired_evaluation"]
        self.assertEqual(runtime["schema"], RUNTIME_SCHEMA)
        self.assertEqual(runtime["policy_views"], list(POLICY_VIEWS))
        self.assertEqual(runtime["reference_lag"], 256)
        self.assertFalse(runtime["training_enabled"])
        episodes = result["_roborender_paired_episodes"]
        self.assertEqual([item["env_index"] for item in episodes], [0, 1])
        self.assertAlmostEqual(episodes[0]["action_imitation_mse"], 1.25)
        self.assertAlmostEqual(episodes[1]["action_imitation_mse"], 2.0)
        self.assertEqual(
            episodes[0]["generated_view_sha256"],
            {
                view: generated_view_sha256(first_views[0][view])
                for view in POLICY_VIEWS
            },
        )
        # Finalization is idempotent and callers cannot mutate retained evidence.
        result["_roborender_paired_episodes"][0]["success"] = False
        self.assertTrue(evidence.finalize()["_roborender_paired_episodes"][0]["success"])
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "finalized"):
            evidence.record_renderer_status()

    def test_from_environment_binds_contract_and_actual_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text(json.dumps(_contract()), encoding="utf-8")
            evidence = PairedEvaluationEvidence.from_environment(
                num_envs=2,
                generation_seed=17,
                actual_runtime_policy_sha256=_sha("parent"),
                policy_views=POLICY_VIEWS,
                view_layout="vertical",
                raytraced_mix_prob=0.0,
                reference_lag=256,
                training_enabled=False,
                eval_relabel_enabled=True,
                one_episode_per_environment=True,
                environ={
                    "IMAGE_DAGGER_PAIRED_EVAL": "1",
                    "IMAGE_DAGGER_PAIRED_EVAL_CONTRACT": str(path),
                    "IMAGE_DAGGER_EXPECT_RUNTIME_POLICY_SHA256": _sha("parent"),
                },
            )
            self.assertEqual(evidence.runtime_policy_sha256, _sha("parent"))

            with self.assertRaisesRegex(
                PairedEvaluationEvidenceError, "loaded runtime policy identity"
            ):
                PairedEvaluationEvidence.from_environment(
                    num_envs=2,
                    generation_seed=17,
                    actual_runtime_policy_sha256=_sha("candidate"),
                    policy_views=POLICY_VIEWS,
                    view_layout="vertical",
                    raytraced_mix_prob=0.0,
                    reference_lag=256,
                    training_enabled=False,
                    eval_relabel_enabled=True,
                    one_episode_per_environment=True,
                    environ={
                        "IMAGE_DAGGER_PAIRED_EVAL": "1",
                        "IMAGE_DAGGER_PAIRED_EVAL_CONTRACT": str(path),
                        "IMAGE_DAGGER_EXPECT_RUNTIME_POLICY_SHA256": _sha("parent"),
                    },
                )

            with self.assertRaisesRegex(
                PairedEvaluationEvidenceError, "actual evaluator runtime"
            ):
                PairedEvaluationEvidence.from_environment(
                    num_envs=2,
                    generation_seed=17,
                    actual_runtime_policy_sha256=_sha("parent"),
                    policy_views=("head_color",),
                    view_layout="vertical",
                    raytraced_mix_prob=0.0,
                    reference_lag=256,
                    training_enabled=False,
                    eval_relabel_enabled=True,
                    one_episode_per_environment=True,
                    environ={
                        "IMAGE_DAGGER_PAIRED_EVAL": "1",
                        "IMAGE_DAGGER_PAIRED_EVAL_CONTRACT": str(path),
                        "IMAGE_DAGGER_EXPECT_RUNTIME_POLICY_SHA256": _sha("parent"),
                    },
                )

    def test_generation_is_transactional_and_requires_exact_coverage(self) -> None:
        evidence = _accumulator()
        incomplete = _views(0)
        incomplete.pop(1)
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "every environment"):
            evidence.record_generation_step(incomplete)
        malformed = _views(0)
        malformed[1].pop("hand_right")
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "exactly"):
            evidence.record_generation_step(malformed)
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "no generated"):
            evidence.finalize()

    def test_episode_rejects_missing_actions_wrong_length_and_scenario(self) -> None:
        evidence = _accumulator()
        evidence.record_generation_step(_views(0))
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "scenario identity"):
            evidence.record_episode_outcome(
                0,
                scenario_id=_sha("wrong"),
                success=True,
                safety_violation=False,
                episode_length=1,
            )
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "no expert/student"):
            evidence.record_episode_outcome(
                0,
                scenario_id=_sha("17:0"),
                success=True,
                safety_violation=False,
                episode_length=1,
            )
        evidence.record_action_pair(0, [0.0], [0.0])
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "episode length"):
            evidence.record_episode_outcome(
                0,
                scenario_id=_sha("17:0"),
                success=True,
                safety_violation=False,
                episode_length=2,
            )
        evidence.record_episode_outcome(
            0,
            scenario_id=_sha("17:0"),
            success=True,
            safety_violation=False,
            episode_length=1,
        )
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "duplicate"):
            evidence.record_episode_outcome(
                0,
                scenario_id=_sha("17:0"),
                success=True,
                safety_violation=False,
                episode_length=1,
            )

    def test_actions_reject_shape_nonfinite_and_post_episode_updates(self) -> None:
        evidence = _accumulator()
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "identical shapes"):
            evidence.record_action_pair(0, [1.0, 2.0], [1.0])
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "finite"):
            evidence.record_action_pair(0, [np.nan], [0.0])
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "leading dimension"):
            evidence.record_action_batch(
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
            )

        evidence.record_generation_step(_views(0))
        evidence.record_action_pair(0, [0.0], [0.0])
        evidence.record_episode_outcome(
            0,
            scenario_id=_sha("17:0"),
            success=True,
            safety_violation=False,
            episode_length=1,
        )
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "already completed"):
            evidence.record_action_pair(0, [0.0], [0.0])

    def test_renderer_errors_and_incomplete_episode_coverage_fail_closed(self) -> None:
        evidence = _accumulator()
        evidence.record_generation_step(_views(0))
        evidence.record_action_batch(np.zeros((2, 1)), np.zeros((2, 1)))
        evidence.record_episode_outcome(
            0,
            scenario_id=_sha("17:0"),
            success=True,
            safety_violation=False,
            episode_length=1,
        )
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "coverage"):
            evidence.finalize()
        evidence.record_episode_outcome(
            1,
            scenario_id=_sha("17:1"),
            success=False,
            safety_violation=False,
            episode_length=1,
        )
        evidence.record_renderer_status(replays_delta=1)
        with self.assertRaisesRegex(PairedEvaluationEvidenceError, "errors or replays"):
            evidence.finalize()

    def test_generated_hash_binds_type_shape_dtype_and_content(self) -> None:
        array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        digest = generated_view_sha256(array)
        self.assertEqual(digest, generated_view_sha256(array.copy()))
        self.assertNotEqual(digest, generated_view_sha256(array.reshape(1, 4, 3)))
        self.assertNotEqual(digest, generated_view_sha256(array.astype(np.int16)))
        changed = array.copy()
        changed[0, 0, 0] += 1
        self.assertNotEqual(digest, generated_view_sha256(changed))
        self.assertEqual(_sha("precomputed"), generated_view_sha256(_sha("precomputed")))


if __name__ == "__main__":
    unittest.main()
