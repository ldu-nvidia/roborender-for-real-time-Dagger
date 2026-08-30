"""Fail-closed tests for the bounded Lepton exclusive-lifecycle smoke."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest import mock

from round_dagger import artifacts, lepton_job, lepton_smoke_job
from scripts.round_dagger import submit_lepton_h200_smoke_and_wait as submitter


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json"


class LeptonBoundedSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract = lepton_job.load_contract(CONTRACT)
        self.request_path = self.root / "smoke-request.json"
        self.request = lepton_smoke_job.materialize_smoke_request(
            self.request_path, contract=self.contract
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_request_is_immutable_exact_and_cannot_become_production(self) -> None:
        self.assertEqual(self.request_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(self.request.execution["world_size"], 8)
        self.assertEqual(self.request.execution["max_steps"], 4)
        self.assertEqual(self.request.execution["split_step"], 2)
        self.assertEqual(self.request.execution["heldout_cases"], 256)
        self.assertNotIn("environment", self.request.document)
        self.assertNotIn("ROUND_DAGGER_LEARNER_MAX_STEPS", json.dumps(self.request.document))

        document = json.loads(self.request_path.read_text())
        document["execution"]["max_steps"] = 1200
        self.request_path.chmod(0o644)
        self.request_path.write_text(json.dumps(document) + "\n")
        self.request_path.chmod(0o444)
        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "fixed four-step validation contract"
        ):
            lepton_smoke_job.load_smoke_request(
                self.request_path, contract=self.contract, immutable=True
            )

    def test_plan_uses_smoke_entrypoint_and_zero_platform_retry(self) -> None:
        document = lepton_job.build_job_document(
            lepton_smoke_job.smoke_contract_view(self.contract),
            self.request,
            request_path=(
                "/workspace/projects/roborender/handoff/incoming/smoke.json"
            ),
            contract_path=(
                "/workspace/projects/roborender/manifests/lepton_backend.json"
            ),
            attempt=1,
        )
        spec = document["spec"]
        command = spec["container"]["command"]
        self.assertEqual(
            command[0],
            "/workspace/projects/roborender/src/fast_roborender/"
            "scripts/round_dagger/run_lepton_h200_smoke_job.sh",
        )
        self.assertNotIn("run_lepton_h200_learner_job.sh", json.dumps(document))
        self.assertEqual(spec["resource_shape"], "my.gpu.8xh200")
        self.assertEqual(spec["max_failure_retry"], 0)
        self.assertEqual(spec["max_job_failure_retry"], 0)
        self.assertEqual(spec["envs"][-1]["value_from"]["secret_name_ref"],
                         "LEO_ROBORENDER_WANDB_API_KEY.ldu")

    def test_execute_orders_supply_chain_smoke_readback_then_marker(self) -> None:
        source = self.root / "sealed-source"
        source.mkdir()
        output = self.root / "remote-output"
        contract = replace(
            self.contract,
            source_root=source,
            project_root=self.root,
            python=Path(os.sys.executable),
            source_inventory_sha256="a" * 64,
            python_freeze_sha256="b" * 64,
        )
        paths = dict(self.request.paths)
        paths.update(
            learner_output=output,
            success_marker=output / "lepton_learner_success.json",
            success_checksum=output / "lepton_learner_success.json.sha256",
        )
        request = replace(self.request, paths=paths)
        events: list[str] = []

        def smoke_runner(*, output: Path, **_kwargs) -> None:
            events.append("smoke")
            output.mkdir(parents=True)
            report = {
                "schema": "fast_roborender.round_dagger.lepton_h200_learner_smoke.v1",
                "status": "PASS",
                "campaign_id": lepton_smoke_job.SMOKE_CAMPAIGN_ID,
                "world_size": 8,
                "device": "cuda",
                "max_steps": 4,
                "split_step": 2,
                "heldout_cases": 256,
                "restart_semantic_equivalent": True,
                "restart_results_equivalent": True,
                "wandb_enabled": True,
                "wandb_project": request.wandb["project"],
                "wandb_entity": request.wandb["entity"],
                "wandb_run_id": request.wandb_run_id,
                "student_quality": {
                    "heldout_action_mse_first": 2.0,
                    "heldout_action_mse_last": 1.0,
                },
            }
            report["report_sha256"] = artifacts.canonical_json_sha256(report)
            artifacts.publish_immutable_json(output / "smoke_report.json", report)

        def readback_runner(*, output: Path, **_kwargs):
            events.append("readback")
            run_path = (
                f"{request.wandb['entity']}/{request.wandb['project']}/"
                f"{request.wandb_run_id}"
            )
            document = {
                "schema": "fast_roborender.round_dagger.learner_wandb_readback.v1",
                "status": "PASS",
                "run_path": run_path,
                "run_state": "finished",
                "exact_training_match": True,
                "exact_heldout_match": True,
                "identity_config_match": True,
            }
            artifacts.publish_immutable_json(output / "wandb_readback.json", document)
            return document

        with (
            mock.patch.object(
                lepton_job, "load_contract", return_value=contract
            ),
            mock.patch.object(
                lepton_smoke_job, "load_smoke_request", return_value=request
            ),
            mock.patch.object(
                lepton_smoke_job,
                "verify_sealed_contract_source",
                side_effect=lambda *_args, **_kwargs: events.append("source"),
            ),
            mock.patch.object(
                lepton_smoke_job,
                "verify_python_environment_freeze",
                side_effect=lambda *_args, **_kwargs: events.append("freeze"),
            ),
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "redacted"}, clear=False),
        ):
            result = lepton_smoke_job.execute_smoke_request(
                contract_path=self.root / "backend.json",
                contract_sha256=contract.sha256,
                request_path=self.root / "request.json",
                request_sha256=request.sha256,
                semantic_attempt=1,
                smoke_runner=smoke_runner,
                readback_runner=readback_runner,
            )
        self.assertEqual(events, ["source", "freeze", "smoke", "readback"])
        self.assertEqual(result["status"], "PASS")
        marker = Path(paths["success_marker"])
        validated = lepton_smoke_job.validate_smoke_completion_marker(
            marker, request=request
        )
        self.assertEqual(validated["world_size"], 8)
        self.assertTrue(Path(paths["success_checksum"]).is_file())

    def test_attempt_two_is_rejected_before_any_gate(self) -> None:
        with (
            mock.patch.object(lepton_job, "load_contract", return_value=self.contract),
            mock.patch.object(
                lepton_smoke_job, "load_smoke_request", return_value=self.request
            ),
            mock.patch.object(
                lepton_smoke_job, "require_finalized_supply_chain"
            ),
            mock.patch.dict(os.environ, {"WANDB_API_KEY": "redacted"}, clear=False),
        ):
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "exactly semantic attempt 1"
            ):
                lepton_smoke_job.execute_smoke_request(
                    contract_path=self.root / "backend.json",
                    contract_sha256=self.contract.sha256,
                    request_path=self.request_path,
                    request_sha256=self.request.sha256,
                    semantic_attempt=2,
                )

    def test_submit_restores_control_and_accepts_only_exact_smoke_marker(self) -> None:
        transfer = self.root / "transfer"
        transfer.write_text("#!/bin/sh\nexit 0\n")
        transfer.chmod(0o700)
        control = self.root / "control"
        control.write_text("#!/bin/sh\nexit 0\n")
        control.chmod(0o700)
        finalized = replace(
            self.contract,
            source_inventory_sha256="a" * 64,
            python_freeze_sha256="b" * 64,
        )
        events: list[str] = []

        class Lifecycle:
            def __init__(inner, *_args, **_kwargs):
                pass

            def __enter__(inner):
                events.append("lifecycle_open")
                return inner

            def __exit__(inner, *_args):
                events.append("lifecycle_close")

            def ensure_stopped(inner, _attempt, *, preflight):
                preflight()
                events.append("stop")

            def ensure_ready(inner):
                events.append("ready")

        job = types.SimpleNamespace()

        def fetch(**_kwargs):
            events.append("fetch")
            owner = tempfile.TemporaryDirectory(dir=self.root)
            marker = Path(owner.name) / "lepton_learner_success.json"
            run_path = (
                f"{self.request.wandb['entity']}/{self.request.wandb['project']}/"
                f"{self.request.wandb_run_id}"
            )
            marker.write_text(
                json.dumps(
                    {
                        "schema": lepton_smoke_job.SMOKE_SUCCESS_SCHEMA,
                        "status": "PASS",
                        "smoke_identity_sha256": self.request.identity_sha256,
                        "backend_contract_sha256": self.request.document[
                            "backend_contract_sha256"
                        ],
                        "request_sha256": self.request.sha256,
                        "semantic_attempt": 1,
                        "world_size": 8,
                        "max_steps": 4,
                        "heldout_cases": 256,
                        "report_path": "attempt_1/smoke_report.json",
                        "report_file_sha256": "c" * 64,
                        "wandb_readback_path": "attempt_1/wandb_readback.json",
                        "wandb_readback_file_sha256": "d" * 64,
                        "wandb_run_path": run_path,
                    }
                )
                + "\n"
            )
            checksum = Path(f"{marker}.sha256")
            checksum.write_text(f"{'e' * 64}  {marker.name}\n")
            return marker, checksum, owner

        arguments = Namespace(
            command="submit",
            apply=True,
            bounded_smoke=True,
            exclusive_node_lifecycle=True,
            lifecycle_state=str(self.root / "lifecycle.json"),
            control_exec_program=str(control),
            control_timeout_seconds=30,
            wait_timeout_seconds=30,
            transfer_program=str(transfer),
            contract=str(CONTRACT),
            contract_sha256=finalized.sha256,
            request=str(self.request_path),
            request_sha256=self.request.sha256,
            remote_contract=(
                "/workspace/projects/roborender/manifests/backend.json"
            ),
            remote_request=(
                "/workspace/projects/roborender/handoff/incoming/smoke.json"
            ),
        )
        with (
            mock.patch.object(
                submitter, "_load", return_value=(finalized, self.request)
            ),
            mock.patch.object(submitter, "_clients", return_value=(object(), object())),
            mock.patch.object(submitter, "ExclusiveNodeLifecycle", Lifecycle),
            mock.patch.object(
                submitter, "find_attempt_jobs", side_effect=[{}, {}]
            ),
            mock.patch.object(
                submitter,
                "verify_reserved_node_job_exclusivity",
                side_effect=lambda *_args, **_kwargs: events.append("inventory"),
            ),
            mock.patch.object(
                submitter,
                "verify_control_preflight",
                side_effect=lambda **_kwargs: events.append("preflight"),
            ),
            mock.patch.object(
                submitter,
                "submit_or_adopt_attempt",
                side_effect=lambda *_args, **_kwargs: (job, False),
            ),
            mock.patch.object(
                submitter, "validate_job_identity", return_value="Completed"
            ),
            mock.patch.object(
                submitter, "fetch_completion_pair", side_effect=fetch
            ),
        ):
            self.assertEqual(submitter._submit(arguments), 0)
        self.assertEqual(
            events,
            [
                "lifecycle_open",
                "inventory",
                "preflight",
                "stop",
                "ready",
                "fetch",
                "ready",
                "lifecycle_close",
            ],
        )

    def test_live_submit_rejects_placeholder_digests_before_provider_client(self) -> None:
        arguments = Namespace(
            command="submit",
            apply=True,
            bounded_smoke=True,
            exclusive_node_lifecycle=True,
            lifecycle_state=str(self.root / "lifecycle.json"),
            control_exec_program=str(self.root / "unused-control"),
            control_timeout_seconds=30,
            wait_timeout_seconds=30,
            transfer_program=str(self.root / "unused-transfer"),
            contract=str(CONTRACT),
            contract_sha256=self.contract.sha256,
            request=str(self.request_path),
            request_sha256=self.request.sha256,
            remote_contract=(
                "/workspace/projects/roborender/manifests/backend.json"
            ),
            remote_request=(
                "/workspace/projects/roborender/handoff/incoming/smoke.json"
            ),
        )
        with mock.patch.object(submitter, "_clients") as clients:
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "unfinalized placeholders"
            ):
                submitter._submit(arguments)
        clients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
