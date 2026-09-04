"""Tests for the pre-collection fixed256 active-policy baseline."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from round_dagger import artifacts
from round_dagger.heldout_baseline import (
    BASELINE_RESULT_SCHEMA,
    BASELINE_WANDB_RECEIPT_SCHEMA,
    CANONICAL_WANDB_RECEIPT_RELATIVE,
    HeldoutBaselineError,
    _active_policy_v0,
    baseline_wandb_run_id,
    evaluate_active_policy_v0,
    verify_active_policy_v0_result,
    wandb_metrics,
)
from round_dagger.state import open_macro_round
from tests.test_round_dagger_state import _campaign


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _record() -> dict:
    return {
        "schema": "fast_roborender.round_dagger.learner_heldout_telemetry.v1",
        "campaign_id": "baseline-test",
        "macro_round": 0,
        "optimizer_step": 0,
        "heldout_snapshot_sha256": _sha("heldout-semantic"),
        "heldout_snapshot_file_sha256": _sha("heldout-file"),
        "heldout_transitions_rank": 256,
        "heldout_transitions_per_rank": [256],
        "heldout_transitions_global": 256,
        "world_size": 1,
        "learned_action_indices": [0],
        "action_mse": 0.25,
        "action_mae": 0.4,
        "action_rmse": 0.5,
        "action_cosine": 0.75,
        "per_action": {
            "0": {"mse": 0.25, "mae": 0.4, "rmse": 0.5, "cosine": 0.75}
        },
    }


@unittest.skipUnless(TORCH_AVAILABLE, "Torch/GR00T runtime is not installed")
class HeldoutBaselineTest(unittest.TestCase):
    def test_active_policy_binding_allows_ready_and_collecting_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary))
            _, ready = _active_policy_v0(root)
            self.assertEqual(ready["phase"], "READY")
            open_macro_round(root, expected_revision=ready["revision"])
            _, collecting = _active_policy_v0(root)
            self.assertEqual(collecting["phase"], "COLLECTING")

    def test_evaluation_publishes_exact_result_without_state_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "campaign"
            checkpoint = root / "policies/policy_000000/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            spec = Path(f"{checkpoint}.policy_spec.json")
            manifest = Path(f"{checkpoint}.manifest.json")
            spec.write_bytes(b"spec")
            manifest.write_bytes(b"manifest")
            heldout = root / "heldout_snapshot.json"
            heldout.write_bytes(b"heldout")
            active = {
                "version": 0,
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": artifacts.sha256_file(checkpoint),
                "runtime_policy_sha256": _sha("runtime"),
                "policy_spec_sha256": artifacts.sha256_file(spec),
                "policy_bundle_manifest_sha256": artifacts.sha256_file(manifest),
            }
            campaign = {"campaign_id": "baseline-test"}
            coordinator = {
                "phase": "COLLECTING",
                "next_macro_round": 0,
                "open_macro_round": 0,
                "active_policy": active,
                "candidate_policy": None,
            }
            heldout_identity = {
                "campaign_id": "baseline-test",
                "transition_count": 256,
                "snapshot_sha256": _sha("heldout-semantic"),
                "file_sha256": artifacts.sha256_file(heldout),
            }
            config = SimpleNamespace(
                sha256=_sha("evaluation-bundle"),
                to_mapping=lambda: {"model_config": {}},
            )
            adapter = SimpleNamespace(actions=lambda *_: None)
            record = _record()
            with (
                mock.patch(
                    "round_dagger.heldout_baseline._active_policy_v0",
                    return_value=(campaign, coordinator),
                ),
                mock.patch(
                    "round_dagger.heldout_baseline.validate_policy_bundle_identity"
                ),
                mock.patch(
                    "round_dagger.heldout_baseline.load_heldout_snapshot",
                    return_value=heldout_identity,
                ),
                mock.patch(
                    "round_dagger.heldout_baseline._evaluation_bundle",
                    return_value=config,
                ),
                mock.patch("round_dagger.heldout_baseline.load_checkpoint_strict"),
                mock.patch(
                    "round_dagger.heldout_baseline.load_heldout_partition",
                    return_value=object(),
                ),
                mock.patch(
                    "round_dagger.heldout_baseline.evaluate_heldout_actions",
                    return_value=record,
                ) as evaluate,
                mock.patch(
                    "round_dagger.heldout_baseline.state.load_state",
                    return_value=coordinator,
                ),
            ):
                result = evaluate_active_policy_v0(
                    campaign_root=root,
                    heldout_snapshot_path=heldout,
                    heldout_snapshot_file_sha256=heldout_identity["file_sha256"],
                    code_sha256=_sha("code"),
                    model_factory=lambda _: torch.nn.Linear(1, 1),
                    adapter_factory=lambda _: adapter,
                )
                repeated = evaluate_active_policy_v0(
                    campaign_root=root,
                    heldout_snapshot_path=heldout,
                    heldout_snapshot_file_sha256=heldout_identity["file_sha256"],
                    code_sha256=_sha("code"),
                    model_factory=lambda _: torch.nn.Linear(1, 1),
                    adapter_factory=lambda _: adapter,
                )
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(result["schema"], BASELINE_RESULT_SCHEMA)
            self.assertEqual(result["case_count"], 256)
            self.assertEqual(result["optimizer_step"], 0)
            self.assertEqual(result["result_sha256"], repeated["result_sha256"])
            self.assertEqual(
                Path(result["result_path"]), root / "quality/policy_v0_fixed256.json"
            )

    def test_wandb_mapping_and_run_identity_are_stable(self) -> None:
        result = {
            "schema": BASELINE_RESULT_SCHEMA,
            "case_count": 256,
            "metrics": _record(),
        }
        payload = wandb_metrics(result)
        self.assertEqual(payload["baseline/evaluation_complete"], 1)
        self.assertEqual(payload["quality/heldout_action_mse"], 0.25)
        self.assertEqual(payload["quality/action_0/cosine"], 0.75)
        first = baseline_wandb_run_id(
            campaign_id="baseline-test",
            campaign_sha256=_sha("campaign"),
            checkpoint_sha256=_sha("checkpoint"),
            heldout_snapshot_sha256=_sha("heldout"),
        )
        second = baseline_wandb_run_id(
            campaign_id="baseline-test",
            campaign_sha256=_sha("campaign"),
            checkpoint_sha256=_sha("checkpoint"),
            heldout_snapshot_sha256=_sha("heldout"),
        )
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 128)

    def test_result_conflict_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            artifacts.publish_immutable_json(path, {"different": True})
            from round_dagger.heldout_baseline import _publish_idempotent

            with self.assertRaisesRegex(HeldoutBaselineError, "conflicts"):
                _publish_idempotent(path, {"different": False})

    def test_verifier_binds_existing_result_to_current_policy_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            active = {
                "version": 0,
                "checkpoint_path": "policies/policy_000000/checkpoint.pt",
                "checkpoint_sha256": _sha("checkpoint"),
                "runtime_policy_sha256": _sha("runtime"),
                "policy_spec_sha256": _sha("spec"),
                "policy_bundle_manifest_sha256": _sha("manifest"),
            }
            campaign = {"campaign_id": "baseline-test"}
            coordinator = {
                "phase": "READY",
                "next_macro_round": 0,
                "open_macro_round": None,
                "active_policy": active,
                "candidate_policy": None,
            }
            result = {
                "schema": BASELINE_RESULT_SCHEMA,
                "campaign_id": "baseline-test",
                "campaign_sha256": artifacts.canonical_json_sha256(campaign),
                "policy_version": 0,
                **{key: value for key, value in active.items() if key != "version"},
                "evaluation_bundle_sha256": _sha("bundle"),
                "heldout_snapshot_path": "/heldout.json",
                "heldout_snapshot_file_sha256": _sha("heldout-file"),
                "heldout_snapshot_sha256": _sha("heldout"),
                "case_count": 256,
                "optimizer_step": 0,
                "metrics": _record(),
            }
            path = root / "quality/policy_v0_fixed256.json"
            artifacts.publish_immutable_json(path, result)
            with mock.patch(
                "round_dagger.heldout_baseline._policy_v0_for_verification",
                return_value=(campaign, coordinator),
            ):
                verified = verify_active_policy_v0_result(root)
            self.assertEqual(verified["checkpoint_sha256"], active["checkpoint_sha256"])
            receipt = {
                "schema": BASELINE_WANDB_RECEIPT_SCHEMA,
                "campaign_id": "baseline-test",
                "checkpoint_sha256": active["checkpoint_sha256"],
                "heldout_snapshot_sha256": result["heldout_snapshot_sha256"],
                "result_sha256": artifacts.sha256_file(path),
                "run_id": baseline_wandb_run_id(
                    campaign_id="baseline-test",
                    campaign_sha256=result["campaign_sha256"],
                    checkpoint_sha256=active["checkpoint_sha256"],
                    heldout_snapshot_sha256=result["heldout_snapshot_sha256"],
                ),
                "entity": "nvidia",
                "project": "lepton-dagger-production-1",
                "optimizer_step": 0,
            }
            artifacts.publish_immutable_json(
                root / CANONICAL_WANDB_RECEIPT_RELATIVE, receipt
            )
            with mock.patch(
                "round_dagger.heldout_baseline._policy_v0_for_verification",
                return_value=(campaign, coordinator),
            ):
                verified = verify_active_policy_v0_result(
                    root, require_wandb_receipt=True
                )
            self.assertIn("wandb_receipt_sha256", verified)
            tampered = dict(coordinator)
            tampered["active_policy"] = {**active, "checkpoint_sha256": _sha("other")}
            with (
                mock.patch(
                    "round_dagger.heldout_baseline._policy_v0_for_verification",
                    return_value=(campaign, tampered),
                ),
                self.assertRaisesRegex(HeldoutBaselineError, "binding differs"),
            ):
                verify_active_policy_v0_result(root)

    def test_launcher_wires_baseline_before_orchestrator_and_learner_keeps_100_step_eval(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        launcher = (
            repository / "scripts/round_dagger/launch_production_macro_round.sh"
        ).read_text(encoding="utf-8")
        baseline_call = launcher.index("run_or_verify_policy_v0_baseline")
        orchestrator_exec = launcher.rindex('exec "$PYTHON" "$ORCHESTRATOR"')
        self.assertLess(baseline_call, orchestrator_exec)
        self.assertIn('[[ $ROUND_DAGGER_MACRO_ROUND == 0 ]]', launcher)
        contract = artifacts.load_strict_json(
            repository
            / "configs/round_dagger/production_learner_8xh100_1200_dagger_prod_run1.json"
        )
        self.assertEqual(contract["optimization"]["optimizer_steps"], 1200)
        self.assertEqual(contract["heldout_evaluation"]["interval_steps"], 100)


if __name__ == "__main__":
    unittest.main()
