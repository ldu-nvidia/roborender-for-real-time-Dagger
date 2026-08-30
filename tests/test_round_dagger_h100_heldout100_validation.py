from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "round_dagger"
    / "validation_learner_1x8h100_heldout100.json"
)
OLD_CONTRACT = (
    ROOT / "configs" / "round_dagger" / "validation_learner_1x8h100_2step.json"
)
HELDOUT = Path(
    "/mnt/data2/ldu/roborender_round_dagger_production/"
    "dagger_prod_run1_resilient_20260829/heldout_snapshot.json"
)
HELDOUT_FILE_SHA = "a" * 64
HELDOUT_SHA = "b" * 64

from scripts.round_dagger import h100_heldout100_validation as validation  # noqa: E402
from scripts.round_dagger import train_learner  # noqa: E402
from round_dagger.learner import JsonlTelemetry, LearnerError  # noqa: E402
from round_dagger.artifacts import sha256_file  # noqa: E402
from round_dagger.learner_identity import (  # noqa: E402
    learner_identity_sha256,
    learner_wandb_run_id,
)

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    torch = None


def quality_row(step: int) -> dict[str, object]:
    improvement_factor = 1.0 if step == 0 else 0.995
    per_action = {}
    for index in range(11):
        mse = 0.01 * (index + 1) * improvement_factor
        per_action[str(index)] = {
            "mse": mse,
            "mae": mse / 2,
            "rmse": math.sqrt(mse),
            "cosine": 0.5,
        }
    mse = 0.25 * improvement_factor
    return {
        "schema": "fast_roborender.round_dagger.learner_heldout_telemetry.v1",
        "campaign_id": "dagger_prod_run1",
        "macro_round": 0,
        "optimizer_step": step,
        "heldout_snapshot_sha256": HELDOUT_SHA,
        "heldout_snapshot_file_sha256": HELDOUT_FILE_SHA,
        "heldout_transitions_rank": 32,
        "heldout_transitions_per_rank": [32] * 8,
        "heldout_transitions_global": 256,
        "world_size": 8,
        "learned_action_indices": list(range(11)),
        "action_mse": mse,
        "action_mae": mse / 2,
        "action_rmse": math.sqrt(mse),
        "action_cosine": 0.25,
        "per_action": per_action,
    }


def training_row(step: int) -> dict[str, object]:
    return {
        "schema": "fast_roborender.round_dagger.learner_telemetry.v2",
        "campaign_id": "dagger_prod_run1",
        "macro_round": 0,
        "optimizer_step": step,
        "loss": 1.0 / step,
        "replay_transitions_rank": 16,
        "replay_transitions_per_rank": [16] * 8,
        "replay_transitions_global": 128,
        "world_size": 8,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FakeRemote:
    def __init__(self, quality: list[dict[str, object]]):
        self.quality = quality

    def scan_history(self, *, keys, page_size):
        del page_size
        if "optimizer_step" not in keys:
            return []
        return self.quality


class FakeWandbRun:
    def define_metric(self, *args, **kwargs):
        del args, kwargs

    def log(self, *args, **kwargs):
        del args, kwargs

    def finish(self, *args, **kwargs):
        del args, kwargs
        return None


class FakeWandbInit:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def init(self, **kwargs):
        self.calls.append(dict(kwargs))
        return FakeWandbRun()


class FullFakeRun:
    def __init__(
        self,
        run_id: str,
        training: list[dict[str, object]],
        quality: list[dict[str, object]],
        config: dict[str, object],
    ):
        self.id = run_id
        self.state = "finished"
        self.training = training
        self.quality = quality
        self.config = config

    def scan_history(self, *, keys, page_size):
        del page_size
        return list(
            self.quality
            if "quality/heldout_action_mse" in keys
            else self.training
        )


class FullFakeApi:
    def __init__(self, run: FullFakeRun):
        self.viewer = {
            "email": "ldu@nvidia.com",
            "username": "leo",
            "teams": ["nvidia"],
        }
        self.expected_run = run

    def run(self, path: str):
        if path != (
            "nvidia/roborender-round-dagger-validation/"
            f"{self.expected_run.id}"
        ):
            raise KeyError(path)
        return self.expected_run


class FullFakeWandb:
    def __init__(self, run: FullFakeRun):
        self.api = FullFakeApi(run)

    def Api(self, timeout):
        del timeout
        return self.api


class Heldout100ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_checked_in_contract_is_reusable_and_pins_gate(self) -> None:
        document = validation.load_contract(CONTRACT, source_root=ROOT)
        self.assertEqual(document["optimization"]["optimizer_steps"], 100)
        self.assertEqual(document["optimization"]["intentional_stop_step"], 50)
        self.assertEqual(document["validation_target"]["macro_round"], 0)
        self.assertEqual(
            document["heldout_evaluation"]["required_optimizer_steps"], [0, 100]
        )
        self.assertEqual(document["heldout_evaluation"]["case_count"], 256)
        self.assertEqual(
            document["quality_gate"]["minimum_relative_action_mse_reduction"],
            0.001,
        )
        self.assertEqual(
            document["restart_validation"]["first_invocation_resume"], "never"
        )
        self.assertEqual(document["cluster"]["time_limit_minutes"], 225)
        serialized = CONTRACT.read_text(encoding="utf-8")
        self.assertNotIn("dagger_prod_run1_resilient_20260829", serialized)
        self.assertNotIn("snapshot_file_sha256", document["heldout_evaluation"])
        self.assertNotIn("learner_contract_sha256", document["production_target"])

    def test_historical_two_step_contract_is_untouched(self) -> None:
        document = json.loads(OLD_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(document["optimization"]["optimizer_steps"], 2)
        self.assertEqual(document["optimization"]["intentional_stop_step"], 1)
        self.assertNotIn("heldout_evaluation", document)

    def test_runtime_binding_is_fresh_campaign_specific(self) -> None:
        contract_sha = sha256_file(CONTRACT)
        production_sha = "c" * 64
        reports = []
        for name in ("campaign-a", "campaign-b"):
            campaign = self.root / name
            campaign.mkdir()
            (campaign / "campaign.json").write_text(
                json.dumps(
                    {
                        "campaign_id": "fresh-campaign",
                        "policy_config_sha256": production_sha,
                    }
                ),
                encoding="utf-8",
            )
            heldout = campaign / "heldout_snapshot.json"
            heldout.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                validation,
                "verify_heldout_snapshot",
                return_value={
                    "file_sha256": HELDOUT_FILE_SHA,
                    "snapshot_sha256": HELDOUT_SHA,
                },
            ):
                reports.append(
                    validation.runtime_binding(
                        contract=CONTRACT,
                        contract_sha256=contract_sha,
                        source_root=ROOT,
                        campaign_root=campaign,
                        campaign_id="fresh-campaign",
                        heldout_snapshot=heldout,
                        heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                        heldout_snapshot_sha256=HELDOUT_SHA,
                        production_contract_sha256=production_sha,
                    )
                )
        self.assertNotEqual(
            reports[0]["binding_sha256"], reports[1]["binding_sha256"]
        )
        self.assertEqual(
            reports[0]["production_contract_sha256"], production_sha
        )

    def test_runtime_binding_rejects_non_campaign_production_contract(self) -> None:
        campaign = self.root / "campaign"
        campaign.mkdir()
        (campaign / "campaign.json").write_text(
            json.dumps(
                {
                    "campaign_id": "fresh-campaign",
                    "policy_config_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )
        heldout = campaign / "heldout_snapshot.json"
        heldout.write_text("{}\n", encoding="utf-8")
        with (
            mock.patch.object(
                validation,
                "verify_heldout_snapshot",
                return_value={
                    "file_sha256": HELDOUT_FILE_SHA,
                    "snapshot_sha256": HELDOUT_SHA,
                },
            ),
            self.assertRaisesRegex(
                validation.Heldout100ValidationError,
                "does not own",
            ),
        ):
            validation.runtime_binding(
                contract=CONTRACT,
                contract_sha256=sha256_file(CONTRACT),
                source_root=ROOT,
                campaign_root=campaign,
                campaign_id="fresh-campaign",
                heldout_snapshot=heldout,
                heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                heldout_snapshot_sha256=HELDOUT_SHA,
                production_contract_sha256="d" * 64,
            )

    def test_contract_rejects_weakened_optimizer_horizon(self) -> None:
        document = json.loads(CONTRACT.read_text(encoding="utf-8"))
        document["optimization"]["optimizer_steps"] = 99
        path = self.root / "contract.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "semantics mismatch"
        ):
            validation.load_contract(path)

    def test_contract_rejects_static_snapshot_identity(self) -> None:
        document = json.loads(CONTRACT.read_text(encoding="utf-8"))
        document["heldout_evaluation"]["snapshot_sha256"] = "0" * 64
        path = self.root / "contract.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(validation.Heldout100ValidationError):
            validation.load_contract(path)

    def test_exact_live_fixed256_snapshot_recursively_verifies(self) -> None:
        if not HELDOUT.exists():
            self.skipTest("live fixed-256 snapshot is not materialized")
        snapshot = validation.verify_heldout_snapshot(
            HELDOUT,
            campaign_root=HELDOUT.parent,
            campaign_id="dagger_prod_run1",
            expected_file_sha256=sha256_file(HELDOUT),
            expected_snapshot_sha256=json.loads(
                HELDOUT.read_text(encoding="utf-8")
            )["snapshot_sha256"],
        )
        self.assertEqual(snapshot["transition_count"], 256)
        self.assertEqual(len(snapshot["resolved_shards"]), 8)

    def test_training_history_requires_contiguous_1_through_100(self) -> None:
        path = self.root / "training.jsonl"
        write_jsonl(path, [training_row(step) for step in range(1, 101)])
        _, rows = validation.training_rows(
            path, campaign_id="dagger_prod_run1", macro_round=0
        )
        self.assertEqual([rows[0]["optimizer_step"], rows[-1]["optimizer_step"]], [1, 100])

    def test_training_history_rejects_99_steps(self) -> None:
        path = self.root / "training.jsonl"
        write_jsonl(path, [training_row(step) for step in range(1, 100)])
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "exactly optimizer steps 1..100"
        ):
            validation.training_rows(
                path, campaign_id="dagger_prod_run1", macro_round=0
            )

    def test_quality_history_requires_exact_steps_zero_and_100(self) -> None:
        path = self.root / "quality.jsonl"
        write_jsonl(path, [quality_row(0), quality_row(100)])
        _, rows = validation.heldout_rows(
            path,
            campaign_id="dagger_prod_run1",
            macro_round=0,
            heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
            heldout_snapshot_sha256=HELDOUT_SHA,
        )
        self.assertEqual([row["optimizer_step"] for row in rows], [0, 100])
        self.assertEqual(len(validation.quality_payload(rows[0])), 49)
        improvement = validation.quality_improvement(rows)
        self.assertGreaterEqual(
            improvement["relative_action_mse_reduction"], 0.001
        )

    def test_quality_gate_rejects_no_mse_improvement(self) -> None:
        rows = [quality_row(0), quality_row(100)]
        rows[1]["action_mse"] = rows[0]["action_mse"]
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "did not improve"
        ):
            validation.quality_improvement(rows)

    def test_telemetry_rejects_nonzero_macro_round(self) -> None:
        path = self.root / "training.jsonl"
        write_jsonl(path, [training_row(step) for step in range(1, 101)])
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "round 0"
        ):
            validation.training_rows(
                path, campaign_id="dagger_prod_run1", macro_round=1
            )

    def test_quality_history_rejects_missing_step_100(self) -> None:
        path = self.root / "quality.jsonl"
        write_jsonl(path, [quality_row(0)])
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "only steps 0 and 100"
        ):
            validation.heldout_rows(
                path,
                campaign_id="dagger_prod_run1",
                macro_round=0,
                heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                heldout_snapshot_sha256=HELDOUT_SHA,
            )

    def test_quality_history_rejects_wrong_case_count(self) -> None:
        rows = [quality_row(0), quality_row(100)]
        rows[1]["heldout_transitions_global"] = 255
        path = self.root / "quality.jsonl"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "identity mismatch"
        ):
            validation.heldout_rows(
                path,
                campaign_id="dagger_prod_run1",
                macro_round=0,
                heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                heldout_snapshot_sha256=HELDOUT_SHA,
            )

    def test_quality_history_rejects_torn_record(self) -> None:
        path = self.root / "quality.jsonl"
        path.write_text(json.dumps(quality_row(0)), encoding="utf-8")
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "torn final record"
        ):
            validation.heldout_rows(
                path,
                campaign_id="dagger_prod_run1",
                macro_round=0,
                heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                heldout_snapshot_sha256=HELDOUT_SHA,
            )

    def test_remote_quality_projection_has_all_49_keys_at_both_steps(self) -> None:
        rows = []
        for step in (0, 100):
            local = quality_row(step)
            rows.append(
                {"optimizer_step": step, **validation.quality_payload(local)}
            )
        observed = validation._remote_quality(FakeRemote(rows))
        self.assertEqual([row["optimizer_step"] for row in observed], [0, 100])
        self.assertEqual(len(observed[0]), 50)

    def test_wandb_first_invocation_uses_resume_never(self) -> None:
        fake = FakeWandbInit()
        with mock.patch.dict("sys.modules", {"wandb": fake}):
            telemetry = JsonlTelemetry(
                self.root / "never.jsonl",
                enabled=True,
                wandb_project="validation",
                wandb_entity="nvidia",
                wandb_run_id="canonical-run",
                wandb_resume_never=True,
            )
            telemetry.close()
        self.assertEqual(fake.calls[0]["resume"], "never")

    def test_wandb_resumed_invocation_uses_resume_must(self) -> None:
        fake = FakeWandbInit()
        with mock.patch.dict("sys.modules", {"wandb": fake}):
            telemetry = JsonlTelemetry(
                self.root / "must.jsonl",
                enabled=True,
                wandb_project="validation",
                wandb_entity="nvidia",
                wandb_run_id="canonical-run",
                wandb_resume_must=True,
            )
            telemetry.close()
        self.assertEqual(fake.calls[0]["resume"], "must")

    def test_wandb_resume_never_and_must_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(LearnerError, "mutually exclusive"):
            JsonlTelemetry(
                self.root / "invalid.jsonl",
                enabled=True,
                wandb_project="validation",
                wandb_run_id="canonical-run",
                wandb_resume_must=True,
                wandb_resume_never=True,
            )

    def test_wandb_resume_never_requires_canonical_run_id(self) -> None:
        with self.assertRaisesRegex(LearnerError, "canonical run ID"):
            JsonlTelemetry(
                self.root / "invalid-no-id.jsonl",
                enabled=True,
                wandb_project="validation",
                wandb_resume_never=True,
            )

    def test_train_cli_resume_modes_preserve_allow_never_and_must(self) -> None:
        ordinary = SimpleNamespace(
            resume_from=None,
            wandb_run_id="canonical-run",
            wandb_resume_never=False,
            wandb_project="production",
        )
        self.assertEqual(train_learner._wandb_resume_modes(ordinary), (False, False))
        first = SimpleNamespace(
            resume_from=None,
            wandb_run_id="canonical-run",
            wandb_resume_never=True,
            wandb_project="validation",
        )
        self.assertEqual(train_learner._wandb_resume_modes(first), (False, True))
        resumed = SimpleNamespace(
            resume_from=Path("restart.pt"),
            wandb_run_id="canonical-run",
            wandb_resume_never=False,
            wandb_project="validation",
        )
        self.assertEqual(train_learner._wandb_resume_modes(resumed), (True, False))
        first.resume_from = Path("restart.pt")
        with self.assertRaisesRegex(LearnerError, "cannot be combined"):
            train_learner._wandb_resume_modes(first)

    def test_train_cli_accepts_only_exact_binding_qualified_run_id(self) -> None:
        canonical = "rrd-m000000-id" + "a" * 64
        binding = "b" * 64
        self.assertEqual(
            train_learner._expected_wandb_run_id(canonical, None), canonical
        )
        self.assertEqual(
            train_learner._expected_wandb_run_id(canonical, binding),
            canonical + "-b" + binding[:32],
        )
        for malformed in ("b" * 63, "B" * 64, "b" * 65, "arbitrary"):
            with self.assertRaisesRegex(LearnerError, "lowercase SHA-256"):
                train_learner._expected_wandb_run_id(canonical, malformed)

    def test_shell_profiles_are_syntax_valid_and_pin_topology(self) -> None:
        names = (
            "submit_oci_nrt_h100_heldout100.sh",
            "oci_nrt_h100_heldout100.sbatch",
            "run_oci_nrt_h100_heldout100_container.sh",
        )
        combined = ""
        for name in names:
            path = ROOT / "scripts" / "round_dagger" / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            self.assertTrue(os.access(path, os.X_OK))
            combined += path.read_text(encoding="utf-8")
        for token in (
            "--gpus-per-node=8",
            "expected exactly 31 heldout100 arguments",
            "ROUND_DAGGER_LEARNER_MAX_STEPS=100",
            "--validation-stop-at-step 50",
            "--heldout-eval-interval 100",
            "--expected-completed-steps 100",
            "--wandb-resume-never",
            "--wandb-run-binding-sha256",
            "heldout_snapshot.json",
            "--time=03:45:00",
        ):
            self.assertIn(token, combined)
        self.assertNotIn("REMOTE_HELDOUT", combined)
        self.assertNotIn("LOCAL_HELDOUT", combined)
        self.assertIn("#SBATCH --account=general_cs_infra__ngcsre", combined)
        self.assertIn("#SBATCH --partition=batch_singlenode", combined)
        self.assertIn("tree_disjoint", combined)
        self.assertIn("realpath -e", combined)
        self.assertNotIn("--time=04:00:00", combined)

    def test_remote_layout_accepts_only_canonical_disjoint_roots(self) -> None:
        campaign = self.root / "campaign"
        output = campaign / "validation" / "heldout100"
        scratch = self.root / "scratch"
        job = scratch / "job"
        log = scratch / "log"
        for path in (output, job, log):
            path.mkdir(parents=True)
        report = validation.validate_remote_layout(
            campaign=campaign,
            output=output,
            job=job,
            log=log,
            output_relative_path="validation/heldout100",
        )
        self.assertEqual(report["output"], str(output))
        nested_log = job / "nested"
        nested_log.mkdir()
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "pairwise tree-disjoint"
        ):
            validation.validate_remote_layout(
                campaign=campaign,
                output=output,
                job=job,
                log=nested_log,
                output_relative_path="validation/heldout100",
            )

    def test_remote_layout_rejects_symlink_alias_and_campaign_alias(self) -> None:
        campaign = self.root / "campaign"
        output = campaign / "output"
        scratch = self.root / "scratch"
        job = scratch / "job"
        log = scratch / "log"
        for path in (output, job, log):
            path.mkdir(parents=True)
        alias = self.root / "output-alias"
        alias.symlink_to(output, target_is_directory=True)
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "non-symlink"
        ):
            validation.validate_remote_layout(
                campaign=campaign,
                output=alias,
                job=job,
                log=log,
                output_relative_path="output",
            )
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "nested campaign target"
        ):
            validation.validate_remote_layout(
                campaign=campaign,
                output=campaign,
                job=job,
                log=log,
                output_relative_path="output",
            )

    def test_no_secret_is_embedded_in_new_profile(self) -> None:
        paths = [
            CONTRACT,
            ROOT / "scripts/round_dagger/h100_heldout100_validation.py",
            ROOT / "scripts/round_dagger/submit_oci_nrt_h100_heldout100.sh",
            ROOT / "scripts/round_dagger/oci_nrt_h100_heldout100.sbatch",
            ROOT / "scripts/round_dagger/run_oci_nrt_h100_heldout100_container.sh",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("wandb_v1_", combined)


@unittest.skipIf(torch is None, "full heldout100 readback test requires PyTorch")
class Heldout100FullReadbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="heldout100_readback_")
        self.root = Path(self.temporary.name)
        self.campaign_id = "dagger_prod_run1"
        self.macro_round = 0
        self.code_sha = "1" * 64
        self.config_sha = "2" * 64
        self.lineage_sha = "3" * 64
        self.replay_file_sha = "4" * 64
        self.replay_sha = "5" * 64
        self.runtime_sha = "6" * 64
        self.container_sha = "7" * 64
        self.contract_sha = "8" * 64
        self.binding_sha = "9" * 64
        self.production_contract_sha = "c" * 64
        self.identity_sha = learner_identity_sha256(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_code_sha256=self.code_sha,
            learner_config_sha256=self.config_sha,
            replay_snapshot_file_sha256=self.replay_file_sha,
            replay_snapshot_sha256=self.replay_sha,
            parent_checkpoint_sha256=self.lineage_sha,
            parent_runtime_policy_sha256=self.runtime_sha,
            runtime_container_sha256=self.container_sha,
            learner_contract_sha256=self.contract_sha,
        )
        self.run_id = learner_wandb_run_id(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_identity_sha256=self.identity_sha,
        ) + f"-b{self.binding_sha[:32]}"
        self.training = [training_row(step) for step in range(1, 101)]
        self.quality = [quality_row(0), quality_row(100)]
        self.full_training = self.root / "full_training.jsonl"
        self.resumed_training = self.root / "resumed_training.jsonl"
        self.full_quality = self.root / "full_quality.jsonl"
        self.resumed_quality = self.root / "resumed_quality.jsonl"
        write_jsonl(self.full_training, self.training)
        write_jsonl(self.resumed_training, self.training)
        write_jsonl(self.full_quality, self.quality)
        write_jsonl(self.resumed_quality, self.quality)

        self.full_checkpoint = self.root / "full_restart.pt"
        self.stopped_checkpoint = self.root / "stopped_restart.pt"
        self.resumed_checkpoint = self.root / "resumed_restart.pt"
        self._checkpoint(self.full_checkpoint, 100)
        self._checkpoint(self.stopped_checkpoint, 50)
        self._checkpoint(self.resumed_checkpoint, 100)
        self.stopped_checkpoint.chmod(0o444)
        self.full_result = self.root / "full_result.json"
        self.resumed_result = self.root / "resumed_result.json"
        self._result(self.full_result, self.full_checkpoint, final_loss=0.75)
        self._result(self.resumed_result, self.resumed_checkpoint, final_loss=0.75)
        self.resume_evidence = self.root / "resume_evidence.json"
        self._resume_evidence()

        quality_history = [
            {
                "optimizer_step": row["optimizer_step"],
                **validation.quality_payload(row),
            }
            for row in self.quality
        ]
        self.remote_config = {
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
            "learner_code_sha256": self.code_sha,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
            "replay_snapshot_file_sha256": self.replay_file_sha,
            "replay_snapshot_sha256": self.replay_sha,
            "parent_runtime_policy_sha256": self.runtime_sha,
            "learner_identity_sha256": self.identity_sha,
            "runtime_container_sha256": self.container_sha,
            "learner_contract_sha256": self.contract_sha,
            "heldout_snapshot_file_sha256": HELDOUT_FILE_SHA,
            "heldout_snapshot_sha256": HELDOUT_SHA,
            "heldout_eval_interval": 100,
            "heldout_transitions": 256,
        }
        self.fake_wandb = FullFakeWandb(
            FullFakeRun(
                self.run_id,
                self.training,
                quality_history,
                self.remote_config,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _checkpoint(self, path: Path, completed_steps: int) -> None:
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
                "checkpoint_schema": "gr00t.trl.full_state.v2",
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "model_lineage_sha256": self.lineage_sha,
                "code_sha256": self.code_sha,
                "learner_config_sha256": self.config_sha,
                "learner_identity_sha256": self.identity_sha,
                "runtime_container_sha256": self.container_sha,
                "learner_contract_sha256": self.contract_sha,
                "replay_snapshot_sha256": self.replay_sha,
                "replay_snapshot_file_sha256": self.replay_file_sha,
                "completed_steps": completed_steps,
                "initial_loss": 1.0,
                "world_size": 8,
                "rng_states": [{} for _ in range(8)],
            },
            path,
        )

    def _result(self, path: Path, checkpoint: Path, *, final_loss: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "fast_roborender.round_dagger.learner_result.v2",
                    "campaign_id": self.campaign_id,
                    "macro_round": self.macro_round,
                    "code_sha256": self.code_sha,
                    "learner_config_sha256": self.config_sha,
                    "model_lineage_sha256": self.lineage_sha,
                    "replay_snapshot_sha256": self.replay_sha,
                    "optimizer_steps": 100,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "initial_loss": 1.0,
                    "final_loss": final_loss,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _resume_evidence(self) -> None:
        self.resume_evidence.write_text(
            json.dumps(
                {
                    "schema": validation.RESUME_SCHEMA,
                    "status": "PASS",
                    "run_id": self.run_id,
                    "stop_step": 50,
                    "target_step": 100,
                    "first_invocation_resume_mode": "never",
                    "second_invocation_resume_mode": "must",
                    "learner_identity_sha256": self.identity_sha,
                    "learner_config_sha256": self.config_sha,
                    "learner_code_sha256": self.code_sha,
                    "replay_snapshot_file_sha256": self.replay_file_sha,
                    "replay_snapshot_sha256": self.replay_sha,
                    "heldout_snapshot_file_sha256": HELDOUT_FILE_SHA,
                    "heldout_snapshot_sha256": HELDOUT_SHA,
                    "validation_binding_sha256": self.binding_sha,
                    "production_contract_sha256": self.production_contract_sha,
                    "runtime_container_sha256": self.container_sha,
                    "learner_contract_sha256": self.contract_sha,
                    "stopped_checkpoint_sha256": sha256_file(
                        self.stopped_checkpoint
                    ),
                    "resumed_checkpoint_sha256": sha256_file(
                        self.resumed_checkpoint
                    ),
                    "canonical_wandb_lineage": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.resume_evidence.chmod(0o444)

    def _readback(self):
        return validation.readback(
            run_id=self.run_id,
            uninterrupted_telemetry=self.full_training,
            resumed_telemetry=self.resumed_training,
            uninterrupted_heldout=self.full_quality,
            resumed_heldout=self.resumed_quality,
            uninterrupted_result=self.full_result,
            resumed_result=self.resumed_result,
            uninterrupted_checkpoint=self.full_checkpoint,
            stopped_checkpoint=self.stopped_checkpoint,
            resumed_checkpoint=self.resumed_checkpoint,
            resume_evidence=self.resume_evidence,
            campaign_id=self.campaign_id,
            macro_round=0,
            learner_code_sha256=self.code_sha,
            learner_config_sha256=self.config_sha,
            model_lineage_sha256=self.lineage_sha,
            replay_snapshot_file_sha256=self.replay_file_sha,
            replay_snapshot_sha256=self.replay_sha,
            parent_runtime_policy_sha256=self.runtime_sha,
            learner_identity_sha256=self.identity_sha,
            runtime_container_sha256=self.container_sha,
            validation_contract_sha256=self.contract_sha,
            validation_binding_sha256=self.binding_sha,
            production_contract_sha256=self.production_contract_sha,
            heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
            heldout_snapshot_sha256=HELDOUT_SHA,
            project="roborender-round-dagger-validation",
            expected_email="ldu@nvidia.com",
            entity="nvidia",
            authorization_scope="team",
            timeout_s=0.1,
            poll_interval_s=0.001,
            wandb_module=self.fake_wandb,
        )

    def test_full_readback_proves_authenticated_quality_and_improvement(self) -> None:
        report = self._readback()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["quality_improvement_verified"])
        self.assertTrue(report["loss_improvement_verified"])
        self.assertGreaterEqual(
            report["quality_improvement"]["relative_action_mse_reduction"],
            0.001,
        )
        self.assertEqual(report["remote"]["quality_history_steps"], [0, 100])
        self.assertEqual(report["remote"]["quality_key_count"], 49)
        self.assertEqual(report["validation_contract_sha256"], self.contract_sha)
        self.assertEqual(report["learner_contract_sha256"], self.contract_sha)
        self.assertEqual(
            report["stale_wandb_history_policy"], "fail_closed_no_adoption"
        )

    def test_full_readback_rejects_no_quality_improvement(self) -> None:
        no_improvement = [quality_row(0), quality_row(100)]
        no_improvement[1]["action_mse"] = no_improvement[0]["action_mse"]
        no_improvement[1]["action_rmse"] = no_improvement[0]["action_rmse"]
        write_jsonl(self.full_quality, no_improvement)
        write_jsonl(self.resumed_quality, no_improvement)
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "did not improve"
        ):
            self._readback()

    def test_full_readback_rejects_non_decreasing_final_loss(self) -> None:
        self._result(self.full_result, self.full_checkpoint, final_loss=1.0)
        self._result(self.resumed_result, self.resumed_checkpoint, final_loss=1.0)
        with self.assertRaisesRegex(
            validation.Heldout100ValidationError, "strictly below"
        ):
            self._readback()

    def test_readback_contract_aliases_feed_attestation_without_translation(self) -> None:
        readback_path = self.root / "readback.json"
        readback_path.write_text(
            json.dumps(self._readback(), sort_keys=True), encoding="utf-8"
        )
        restart_path = self.root / "restart_equivalence.json"
        restart_path.write_text(
            json.dumps(
                {
                    "semantic_equivalent": True,
                    "results_equivalent": True,
                    "expected_completed_steps": 100,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        contract_document = {
            "purpose": "live_h100_fixed256_step100_wandb_restart_gate",
            "allocation": {"nodes": 1, "gpus_per_node": 8},
        }
        with (
            mock.patch.object(
                validation, "load_contract", return_value=contract_document
            ),
            mock.patch.object(
                validation,
                "runtime_binding",
                return_value={
                    "binding_sha256": self.binding_sha,
                    "production_contract_sha256": self.production_contract_sha,
                    "heldout_snapshot_file_sha256": HELDOUT_FILE_SHA,
                    "heldout_snapshot_sha256": HELDOUT_SHA,
                },
            ),
        ):
            report = validation.build_attestation(
                contract=self.root / "contract.json",
                contract_sha256=self.contract_sha,
                source_root=self.root,
                campaign_root=self.root,
                heldout_snapshot=self.root / "heldout.json",
                heldout_snapshot_file_sha256=HELDOUT_FILE_SHA,
                heldout_snapshot_sha256=HELDOUT_SHA,
                production_contract_sha256=self.production_contract_sha,
                validation_binding_sha256=self.binding_sha,
                campaign_id=self.campaign_id,
                readback_path=readback_path,
                restart_report=restart_path,
                output=self.root / "attestation.json",
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["validation_contract_sha256"], self.contract_sha
        )
        self.assertTrue(report["quality_improvement_verified"])


if __name__ == "__main__":
    unittest.main()
