"""Focused control-plane tests for the Lepton eight-H200 learner backend."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest import mock

from round_dagger import lepton_job
from round_dagger.learner_identity import learner_identity_sha256
from scripts.round_dagger import submit_lepton_round_learner_and_wait as submitter


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json"


def _environment(identity: str, campaign: str, macro_round: int) -> dict[str, str]:
    campaign_root = f"/workspace/projects/roborender/campaigns/{campaign}"
    output = f"{campaign_root}/rounds/macro_round_{macro_round:06d}/learner_output"
    sha = lambda character: character * 64
    return {
        "ROUND_DAGGER_CAMPAIGN_ROOT": campaign_root,
        "ROUND_DAGGER_CAMPAIGN_ID": campaign,
        "ROUND_DAGGER_MACRO_ROUND": str(macro_round),
        "ROUND_DAGGER_POLICY_CHECKPOINT": f"{campaign_root}/policies/policy_v0.pt",
        "ROUND_DAGGER_POLICY_SHA256": sha("1"),
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": sha("2"),
        "ROUND_DAGGER_REPLAY_SNAPSHOT": f"{campaign_root}/replay/snapshot.json",
        "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": sha("3"),
        "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": sha("4"),
        "ROUND_DAGGER_LEARNER_OUTPUT_DIR": output,
        "ROUND_DAGGER_LEARNER_RESULT": f"{output}/learner_result_round_{macro_round:06d}.json",
        "ROUND_DAGGER_LEARNER_CODE_SHA256": sha("5"),
        "ROUND_DAGGER_LEARNER_CONFIG_SHA256": sha("6"),
        "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": sha("7"),
        "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": identity,
        "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": sha("1"),
        "ROUND_DAGGER_BASE_POLICY_SPEC": f"{campaign_root}/policies/policy_v0.pt.policy_spec.json",
        "ROUND_DAGGER_LEARNER_BATCH_SIZE": "8",
        "ROUND_DAGGER_LEARNER_MAX_STEPS": "1200",
        "ROUND_DAGGER_LEARNER_SAVE_EVERY": "5",
        "ROUND_DAGGER_LEARNER_LEARNING_RATE": "2.5e-5",
        "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": "0.01",
        "ROUND_DAGGER_LEARNER_SEED": "17",
        "ROUND_DAGGER_LEARNER_PARTITION_SEED": "23",
        "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
        "ROUND_DAGGER_COHORT_COUNT": "4",
        "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
        "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
        "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "131072",
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT": f"{campaign_root}/replay/heldout_snapshot.json",
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256": sha("8"),
        "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_SHA256": sha("9"),
        "ROUND_DAGGER_HELDOUT_EVAL_INTERVAL": "100",
        "ROUND_DAGGER_WANDB_PROJECT": "roborender-round-dagger",
        "ROUND_DAGGER_WANDB_ENTITY": "nvidia",
        "ROUND_DAGGER_WANDB_RUN_ID": f"rrd-m{macro_round:06d}-id{identity}",
        "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": "ldu@nvidia.com",
        "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE": "team",
    }


class FakeClient:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.created = []

    def list_all(self, **_kwargs):
        return list(self.jobs)

    def create(self, spec):
        self.created.append(spec)
        return self.jobs[-1]

    def get(self, job, **_kwargs):
        return job


def _control_deployment(state: str, *, minimum: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(name=lepton_job.CONTROL_DEPLOYMENT),
        spec=types.SimpleNamespace(
            is_pod=True,
            resource_requirement=types.SimpleNamespace(
                resource_shape=lepton_job.RESOURCE_SHAPE,
                shared_memory_size=524288,
                min_replicas=minimum,
                max_replicas=minimum,
                affinity=types.SimpleNamespace(
                    allowed_dedicated_node_groups=[lepton_job.NODE_GROUP]
                ),
            ),
            reservation_config=types.SimpleNamespace(
                reservation_id=lepton_job.RESERVATION_ID,
                allow_burst_to_other_reservations=None,
            ),
            mounts=[
                types.SimpleNamespace(
                    path=str(lepton_job.WORKSPACE_SOURCE),
                    from_="node-nfs:nfs",
                    mount_path=str(lepton_job.WORKSPACE_MOUNT),
                )
            ],
        ),
        status=types.SimpleNamespace(state=state),
    )


class FakeDeploymentClient:
    def __init__(self, state: str = "Ready"):
        self.state = state
        self.events: list[str] = []
        self.raise_before_scale_down = False
        self.raise_after_scale_down = False

    def get(self, name):
        self.events.append(f"get:{name}:{self.state}")
        return _control_deployment(
            self.state, minimum=1 if self.state == "Ready" else 0
        )

    def scale(self, name, replicas):
        self.events.append(f"scale:{name}:{replicas}")
        if replicas == 0 and self.raise_before_scale_down:
            raise RuntimeError("provider detail must not escape")
        self.state = "Stopped" if replicas == 0 else "Ready"
        if replicas == 0 and self.raise_after_scale_down:
            raise RuntimeError("provider detail must not escape")
        return self.get(name)


def _job(request, contract, attempt, state):
    metadata = types.SimpleNamespace(
        name=lepton_job.job_name(request.identity_sha256, attempt),
        labels=lepton_job._job_labels(
            contract=contract, request=request, attempt=attempt
        ),
    )
    return types.SimpleNamespace(
        metadata=metadata,
        spec=types.SimpleNamespace(
            affinity=types.SimpleNamespace(
                allowed_dedicated_node_groups=[lepton_job.NODE_GROUP]
            ),
            reservation_config=types.SimpleNamespace(
                reservation_id=lepton_job.RESERVATION_ID,
                allow_burst_to_other_reservations=False,
            ),
        ),
        status=types.SimpleNamespace(state=types.SimpleNamespace(value=state)),
    )


def _unknown_job(request, contract, *, failed=0):
    job = _job(request, contract, 1, "UNK")
    job.status.ready = 0
    job.status.active = 0
    job.status.failed = failed
    job.status.succeeded = 0
    job.status.completion_time = None
    return job


class LeptonJobBackendTest(unittest.TestCase):
    def setUp(self):
        self.contract = lepton_job.load_contract(CONTRACT)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        campaign = "dagger-prod-test"
        macro_round = 0
        identity = learner_identity_sha256(
            campaign_id=campaign,
            macro_round=macro_round,
            learner_code_sha256="5" * 64,
            learner_config_sha256="6" * 64,
            replay_snapshot_file_sha256="3" * 64,
            replay_snapshot_sha256="4" * 64,
            parent_checkpoint_sha256="1" * 64,
            parent_runtime_policy_sha256="2" * 64,
            runtime_container_sha256=self.contract.image_digest,
            learner_contract_sha256="7" * 64,
        )
        environment = _environment(identity, campaign, macro_round)
        output = Path(environment["ROUND_DAGGER_LEARNER_OUTPUT_DIR"])
        document = {
            "schema": lepton_job.REQUEST_SCHEMA,
            "campaign_id": campaign,
            "macro_round": macro_round,
            "learner_identity_sha256": identity,
            "backend_contract_sha256": self.contract.sha256,
            "environment": environment,
            "paths": {
                "source_root": "/workspace/projects/roborender/src/fast_roborender",
                "campaign_root": environment["ROUND_DAGGER_CAMPAIGN_ROOT"],
                "learner_output": str(output),
                "learner_result": environment["ROUND_DAGGER_LEARNER_RESULT"],
                "completion_attestation": str(output / "learner_completion_round_000000.json"),
                "success_marker": str(output / "lepton_learner_success.json"),
                "success_checksum": str(output / "lepton_learner_success.json.sha256"),
            },
        }
        self.request_path = self.root / "request.json"
        self.request_path.write_text(json.dumps(document, sort_keys=True) + "\n")
        self.request = lepton_job.load_request(
            self.request_path, contract=self.contract
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_checked_in_contract_pins_reserved_private_node(self):
        contract = self.contract
        self.assertEqual(contract.document["schema"], lepton_job.CONTRACT_SCHEMA)
        self.assertEqual(contract.semantic_max_attempts, 3)
        self.assertEqual(contract.image_digest, "2140e699" + "b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c")
        self.assertEqual(
            contract.document["cluster"]["reservation"],
            "cosmos-solution-architect-xcwtffiq",
        )
        self.assertEqual(
            contract.document["cluster"]["node_group"],
            "neb-cdg-lepton-001-trrjqlvo",
        )
        self.assertEqual(contract.document["orchestration"]["visibility"], "private")
        self.assertEqual(
            contract.source_inventory,
            Path("/workspace/projects/roborender/src/fast_roborender.inventory.json"),
        )
        self.assertEqual(
            contract.python_freeze,
            Path("/workspace/projects/roborender/envs/h200_learner.freeze.txt"),
        )
        self.assertEqual(contract.source_inventory_sha256, "0" * 64)
        self.assertEqual(contract.python_freeze_sha256, "0" * 64)

    def test_job_document_has_eight_h200_zero_platform_retry_and_secret_reference(self):
        document = lepton_job.build_job_document(
            self.contract,
            self.request,
            request_path="/workspace/projects/roborender/handoff/incoming/request.json",
            contract_path="/workspace/projects/roborender/manifests/backend.json",
            attempt=1,
        )
        spec = document["spec"]
        self.assertEqual(spec["resource_shape"], "my.gpu.8xh200")
        self.assertEqual(spec["parallelism"], 1)
        self.assertEqual(spec["max_failure_retry"], 0)
        self.assertEqual(spec["max_job_failure_retry"], 0)
        self.assertEqual(
            spec["reservation_config"],
            {
                "reservation_id": "cosmos-solution-architect-xcwtffiq",
                "allow_burst_to_other_reservations": False,
            },
        )
        secret = spec["envs"][-1]
        self.assertEqual(secret["name"], "WANDB_API_KEY")
        self.assertEqual(
            secret["value_from"]["secret_name_ref"],
            "LEO_ROBORENDER_WANDB_API_KEY.ldu",
        )
        self.assertNotIn("value", secret)
        serialized = json.dumps(document)
        self.assertNotIn("wandb_v1_", serialized)
        self.assertTrue(
            all(len(value) <= 63 for value in document["metadata"]["labels"].values())
        )
        # The production Lepton SDK venv exercises conversion in the external
        # validation command.  Lightweight test environments may omit it.
        try:
            sdk_job = lepton_job.to_sdk_job(document)
        except lepton_job.LeptonJobError as exc:
            self.assertIn("SDK is not installed", str(exc))
        else:
            self.assertEqual(sdk_job.spec.resource_shape, "my.gpu.8xh200")
            self.assertEqual(sdk_job.metadata.visibility.value, "private")
            self.assertEqual(sdk_job.spec.max_failure_retry, 0)
            self.assertEqual(
                sdk_job.spec.envs[-1].value_from.secret_name_ref,
                "LEO_ROBORENDER_WANDB_API_KEY.ldu",
            )

    def test_request_rejects_path_escape_and_credential_environment(self):
        document = json.loads(self.request_path.read_text())
        document["paths"]["success_marker"] = (
            "/workspace/projects/roborender/campaigns/x/../escape.json"
        )
        escaped = self.root / "escaped.json"
        escaped.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "normalized"):
            lepton_job.load_request(escaped, contract=self.contract)

        document = json.loads(self.request_path.read_text())
        document["environment"]["WANDB_API_KEY"] = "wandb_v1_never-store"
        credential = self.root / "credential.json"
        credential.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "credentials"):
            lepton_job.load_request(credential, contract=self.contract)

        document = json.loads(self.request_path.read_text())
        document["environment"].pop("ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT")
        no_heldout = self.root / "no-heldout.json"
        no_heldout.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "missing=.*HELDOUT"):
            lepton_job.load_request(no_heldout, contract=self.contract)

        document = json.loads(self.request_path.read_text())
        document["environment"][
            "ROUND_DAGGER_HELDOUT_REPLAY_SNAPSHOT_FILE_SHA256"
        ] = "not-a-digest"
        malformed_heldout = self.root / "malformed-heldout.json"
        malformed_heldout.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "lowercase SHA-256"):
            lepton_job.load_request(malformed_heldout, contract=self.contract)

    def test_materializer_rebases_only_reviewed_campaign_paths_and_seals_request(self):
        local = self.root / "campaign"
        local.mkdir()
        environment = dict(self.request.environment)
        old_root = Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"])
        environment["ROUND_DAGGER_CAMPAIGN_ROOT"] = str(local)
        for key in lepton_job._CAMPAIGN_PATH_ENV & set(environment):
            relative = Path(environment[key]).relative_to(old_root)
            environment[key] = str(local / relative)
        environment["ROUND_DAGGER_LEARNER_OUTPUT_DIR"] = str(local / "old-output")
        environment["ROUND_DAGGER_LEARNER_RESULT"] = str(local / "old-result.json")
        # Ambient credentials and unrelated variables are deliberately ignored.
        environment["WANDB_API_KEY"] = "wandb_v1_must_not_be_projected"
        environment["UNRELATED"] = "not-projected"
        destination = self.root / "materialized.json"
        request = lepton_job.materialize_request(
            destination,
            contract=self.contract,
            local_campaign_root=local,
            remote_campaign_root=(
                "/workspace/projects/roborender/campaigns/materialized"
            ),
            remote_learner_output=(
                "/workspace/projects/roborender/campaigns/materialized/"
                "rounds/macro_round_000000/learner_output"
            ),
            source_environment=environment,
        )
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o444)
        self.assertNotIn("WANDB_API_KEY", request.environment)
        self.assertNotIn("UNRELATED", request.environment)
        self.assertTrue(
            request.environment["ROUND_DAGGER_POLICY_CHECKPOINT"].startswith(
                "/workspace/projects/roborender/campaigns/materialized/"
            )
        )

    def test_exact_job_is_adopted_and_multiple_active_attempts_fail_closed(self):
        first = _job(self.request, self.contract, 1, "Running")
        observed = lepton_job.find_attempt_jobs(
            FakeClient([first]), contract=self.contract, request=self.request
        )
        self.assertEqual(observed, {1: first})

        second = _job(self.request, self.contract, 2, "Starting")
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "multiple semantic"):
            lepton_job.find_attempt_jobs(
                FakeClient([first, second]),
                contract=self.contract,
                request=self.request,
            )

        missing_first = _job(self.request, self.contract, 2, "Failed")
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "history contains a gap"):
            lepton_job.find_attempt_jobs(
                FakeClient([missing_first]),
                contract=self.contract,
                request=self.request,
            )

    def test_pristine_unknown_admission_is_bounded_as_starting(self):
        pristine = _unknown_job(self.request, self.contract)
        self.assertEqual(
            lepton_job.validate_job_identity(
                pristine, contract=self.contract, request=self.request, attempt=1
            ),
            "Starting",
        )
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "unsupported state"):
            lepton_job.validate_job_identity(
                _unknown_job(self.request, self.contract, failed=1),
                contract=self.contract,
                request=self.request,
                attempt=1,
            )

    def test_reserved_node_preflight_is_scoped_to_exact_reservation(self):
        exact = _job(self.request, self.contract, 1, "Running")
        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "active Job already claims"
        ):
            lepton_job.verify_reserved_node_job_exclusivity(
                FakeClient([exact]), contract=self.contract, request=self.request
            )
        # The dedicated node group is shared by unrelated reservations.  Such
        # Jobs cannot consume our pinned reservation and must not deadlock its
        # exclusive lifecycle preflight.
        foreign = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="unrelated-job", labels={}),
            spec=types.SimpleNamespace(
                affinity=types.SimpleNamespace(
                    allowed_dedicated_node_groups=[lepton_job.NODE_GROUP]
                ),
                reservation_config=types.SimpleNamespace(
                    reservation_id=None,
                    allow_burst_to_other_reservations=True,
                ),
            ),
            status=types.SimpleNamespace(state="Running"),
        )
        lepton_job.verify_reserved_node_job_exclusivity(
            FakeClient([foreign]), contract=self.contract, request=self.request
        )

        malformed = _job(self.request, self.contract, 1, "Running")
        malformed.spec.reservation_config.allow_burst_to_other_reservations = True
        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "malformed active Job claims"
        ):
            lepton_job.verify_reserved_node_job_exclusivity(
                FakeClient([malformed]), contract=self.contract, request=self.request
            )

    def test_completion_checksum_and_identity_detect_tampering(self):
        marker = self.root / "lepton_learner_success.json"
        document = {
            "schema": "fast_roborender.round_dagger.remote_learner_success.v4",
            "status": "PASS",
            "campaign_id": self.request.campaign_id,
            "macro_round": self.request.macro_round,
            "learner_code_sha256": self.request.environment["ROUND_DAGGER_LEARNER_CODE_SHA256"],
            "learner_config_sha256": self.request.environment["ROUND_DAGGER_LEARNER_CONFIG_SHA256"],
            "learner_identity_sha256": self.request.identity_sha256,
            "result_path": self.request.paths["learner_result"].name,
            "result_sha256": "a" * 64,
            "completion_attestation_path": self.request.paths["completion_attestation"].name,
            "completion_attestation_sha256": "b" * 64,
            "learner_output_manifest_path": str(
                lepton_job.PROJECT_ROOT
                / "results"
                / self.request.identity_sha256
                / "learner_output.manifest.json"
            ),
            "learner_output_manifest_sha256": "c" * 64,
            "learner_output_archive_sha256": "d" * 64,
            "learner_output_inventory_sha256": "e" * 64,
            "learner_output_tree_sha256": "f" * 64,
        }
        marker.write_text(json.dumps(document, sort_keys=True) + "\n")
        checksum = Path(f"{marker}.sha256")
        lepton_job.publish_checksum(marker, checksum)
        self.assertEqual(lepton_job.verify_checksum(marker, checksum), lepton_job.sha256_file(marker))
        self.assertEqual(
            lepton_job.validate_completion_marker(marker, request=self.request)["status"],
            "PASS",
        )
        marker.write_text(marker.read_text().replace('"PASS"', '"FAIL"'))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "checksum"):
            lepton_job.verify_checksum(marker, checksum)

    def test_exclusive_lifecycle_stops_only_after_preflight_and_restores_ready(self):
        client = FakeDeploymentClient()
        lifecycle_path = self.root / "exclusive-node.json"
        calls: list[str] = []
        with lepton_job.ExclusiveNodeLifecycle(
            lifecycle_path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as lifecycle:
            lifecycle.ensure_stopped(1, preflight=lambda: calls.append("preflight"))
            self.assertEqual(calls, ["preflight"])
            self.assertEqual(client.state, "Stopped")
            self.assertEqual(lifecycle.phase, "control_stopped")
            lifecycle.ensure_ready()
            self.assertEqual(client.state, "Ready")
            self.assertEqual(lifecycle.phase, "control_ready")
        self.assertLess(
            client.events.index("scale:leo-cosmos-h200-dev:0"),
            client.events.index("scale:leo-cosmos-h200-dev:1"),
        )
        document = json.loads(lifecycle_path.read_text())
        self.assertEqual(document["learner_identity_sha256"], self.request.identity_sha256)
        self.assertEqual(document["phase"], "control_ready")
        self.assertEqual(stat.S_IMODE(lifecycle_path.stat().st_mode), 0o600)

    def test_control_pod_zero_replica_not_ready_is_normalized_as_stopped(self):
        stopped = _control_deployment("Not Ready", minimum=0)
        self.assertEqual(
            lepton_job.validate_control_deployment(stopped, contract=self.contract),
            "Stopped",
        )
        transient = _control_deployment("Ready", minimum=0)
        self.assertEqual(
            lepton_job.validate_control_deployment(transient, contract=self.contract),
            "Stopping",
        )

    def test_control_adapter_uses_symmetric_partial_replica_updates(self):
        class RawClient:
            def __init__(self):
                self.calls = []

            def get(self, name):
                return name

            def update(self, name, partial):
                requirement = partial.spec.resource_requirement
                self.calls.append(
                    (name, requirement.min_replicas, requirement.max_replicas)
                )
                return partial

        raw = RawClient()
        def partial(count):
            return types.SimpleNamespace(
                spec=types.SimpleNamespace(
                    resource_requirement=types.SimpleNamespace(
                        min_replicas=count,
                        max_replicas=count,
                    )
                )
            )

        adapter = lepton_job.LeptonControlDeploymentAdapter(
            raw, partial_builder=partial
        )
        adapter.scale(lepton_job.CONTROL_DEPLOYMENT, 0)
        adapter.scale(lepton_job.CONTROL_DEPLOYMENT, 1)
        self.assertEqual(
            raw.calls,
            [
                (lepton_job.CONTROL_DEPLOYMENT, 0, 0),
                (lepton_job.CONTROL_DEPLOYMENT, 1, 1),
            ],
        )
        for invalid in (False, -1, 2):
            with self.assertRaises(lepton_job.LeptonJobError):
                adapter.scale(lepton_job.CONTROL_DEPLOYMENT, invalid)
        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "control deployment name mismatch"
        ):
            adapter.scale("another-deployment", 0)

    def test_exclusive_lifecycle_recovers_crash_after_effective_stop(self):
        client = FakeDeploymentClient()
        client.raise_after_scale_down = True
        lifecycle_path = self.root / "exclusive-node-crash.json"
        with lepton_job.ExclusiveNodeLifecycle(
            lifecycle_path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as lifecycle:
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "could not scale down"
            ):
                lifecycle.ensure_stopped(1, preflight=lambda: None)
            self.assertEqual(client.state, "Stopped")
            self.assertEqual(lifecycle.phase, "control_stop_requested")

        client.raise_after_scale_down = False
        calls: list[str] = []
        with lepton_job.ExclusiveNodeLifecycle(
            lifecycle_path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as recovered:
            recovered.ensure_stopped(1, preflight=lambda: calls.append("must-not-run"))
            self.assertEqual(calls, [])
            self.assertEqual(recovered.phase, "control_stopped")
            recovered.ensure_ready()
        self.assertEqual(client.state, "Ready")

    def test_exclusive_lifecycle_rechecks_preflight_after_crash_before_scale(self):
        client = FakeDeploymentClient()
        client.raise_before_scale_down = True
        lifecycle_path = self.root / "exclusive-node-before-scale.json"
        with lepton_job.ExclusiveNodeLifecycle(
            lifecycle_path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as lifecycle:
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "could not scale down"
            ):
                lifecycle.ensure_stopped(1, preflight=lambda: None)
            self.assertEqual(client.state, "Ready")
            self.assertEqual(lifecycle.phase, "control_stop_requested")

        client.raise_before_scale_down = False
        calls = []
        with lepton_job.ExclusiveNodeLifecycle(
            lifecycle_path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as recovered:
            recovered.ensure_stopped(
                1, preflight=lambda: calls.append("fresh-preflight")
            )
            self.assertEqual(calls, ["fresh-preflight"])
            recovered.ensure_ready()

    def test_exclusive_lifecycle_rejects_unowned_stopped_control_pod(self):
        client = FakeDeploymentClient("Stopped")
        with lepton_job.ExclusiveNodeLifecycle(
            self.root / "unowned-stopped.json",
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
            sleep=lambda _seconds: None,
        ) as lifecycle:
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "not Ready.*not owned"
            ):
                lifecycle.ensure_stopped(1, preflight=lambda: None)
        self.assertNotIn("scale:leo-cosmos-h200-dev:1", client.events)

    def test_exclusive_lifecycle_identity_and_lock_are_fail_closed(self):
        path = self.root / "exclusive-identity.json"
        client = FakeDeploymentClient()
        first = lepton_job.ExclusiveNodeLifecycle(
            path,
            contract=self.contract,
            request=self.request,
            client=client,
            timeout_seconds=30,
        )
        with first:
            second = lepton_job.ExclusiveNodeLifecycle(
                path,
                contract=self.contract,
                request=self.request,
                client=client,
                timeout_seconds=30,
            )
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "another exclusive-node coordinator"
            ):
                second.__enter__()
        document = json.loads(path.read_text())
        document["learner_identity_sha256"] = "f" * 64
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "identity mismatch"):
            with lepton_job.ExclusiveNodeLifecycle(
                path,
                contract=self.contract,
                request=self.request,
                client=client,
                timeout_seconds=30,
            ):
                pass

    def test_control_preflight_uses_only_public_exact_paths_and_suppresses_output(self):
        helper = self.root / "control-exec"
        helper.write_text("#!/bin/sh\nexit 0\n")
        helper.chmod(0o700)
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return types.SimpleNamespace(returncode=0)

        lepton_job.verify_control_preflight(
            control_exec_program=helper,
            contract=self.contract,
            request=self.request,
            remote_contract=(
                "/workspace/projects/roborender/manifests/backend.json"
            ),
            remote_request=(
                "/workspace/projects/roborender/handoff/incoming/request.json"
            ),
            command_runner=runner,
        )
        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(argv[:2], [str(helper), lepton_job.CONTROL_DEPLOYMENT])
        self.assertIn(self.contract.sha256, argv[2])
        self.assertIn(self.request.sha256, argv[2])
        self.assertLessEqual(len(argv[2].encode("utf-8")), 2800)
        self.assertIn("base64.b85decode", argv[2])
        self.assertNotIn("wandb_v1_", " ".join(argv))
        self.assertIs(kwargs["stdout"], lepton_job.subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], lepton_job.subprocess.DEVNULL)

        def rejected(_argv, **_kwargs):
            return types.SimpleNamespace(returncode=38)

        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "rejected staged bytes or node idleness"
        ):
            lepton_job.verify_control_preflight(
                control_exec_program=helper,
                contract=self.contract,
                request=self.request,
                remote_contract=(
                    "/workspace/projects/roborender/manifests/backend.json"
                ),
                remote_request=(
                    "/workspace/projects/roborender/handoff/incoming/request.json"
                ),
                command_runner=rejected,
            )

    def test_supply_chain_paths_are_canonical_and_hashes_are_mandatory(self):
        document = json.loads(CONTRACT.read_text())
        document["schema"] = "fast_roborender.round_dagger.lepton_job_contract.v1"
        malformed = self.root / "bad-supply-chain.json"
        malformed.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "schema/name mismatch"):
            lepton_job.load_contract(malformed)

        document = json.loads(CONTRACT.read_text())
        document["supply_chain"]["source_inventory"] = (
            "/workspace/projects/roborender/manifests/unrelated.json"
        )
        malformed.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "non-canonical"):
            lepton_job.load_contract(malformed)

        document = json.loads(CONTRACT.read_text())
        document["supply_chain"]["python_freeze_sha256"] = "pending"
        malformed.write_text(json.dumps(document))
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "lowercase SHA-256"):
            lepton_job.load_contract(malformed)

        with self.assertRaisesRegex(
            lepton_job.LeptonJobError, "unfinalized placeholders"
        ):
            lepton_job.require_finalized_supply_chain(self.contract)

    def test_source_inventory_and_python_freeze_are_verified_before_use(self):
        source = self.root / "sealed-source"
        verifier = source / "scripts/round_dagger/verify_sealed_source_snapshot.py"
        verifier.parent.mkdir(parents=True)
        verifier.write_text("raise SystemExit(0)\n")
        verifier.chmod(0o444)
        verifier_digest = sha256(verifier.read_bytes()).hexdigest()
        inventory = self.root / "sealed-source.inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema": "fast_roborender.hsg_source_inventory.v1",
                    "file_count": 1,
                    "total_file_bytes": verifier.stat().st_size,
                    "files": [
                        {
                            "path": "scripts/round_dagger/verify_sealed_source_snapshot.py",
                            "mode": "0o444",
                            "bytes": verifier.stat().st_size,
                            "sha256": verifier_digest,
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n"
        )
        inventory.chmod(0o444)
        freeze = self.root / "h200_learner.freeze.txt"
        freeze.write_bytes(b"example==1.0\npip==26.0\n")
        freeze.chmod(0o444)
        contract = replace(
            self.contract,
            source_root=source,
            source_inventory=inventory,
            source_inventory_sha256=sha256(inventory.read_bytes()).hexdigest(),
            source_verifier=verifier,
            python_freeze=freeze,
            python_freeze_sha256=sha256(freeze.read_bytes()).hexdigest(),
        )
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            stdout = freeze.read_bytes() if argv[-3:] == ["pip", "freeze", "--all"] else b""
            return types.SimpleNamespace(returncode=0, stdout=stdout)

        lepton_job.verify_sealed_contract_source(contract, runner=runner)
        lepton_job.verify_python_environment_freeze(contract, runner=runner)
        self.assertEqual(len(calls), 2)
        self.assertIn("--expected-inventory-sha256", calls[0][0])
        self.assertEqual(calls[1][0][-4:], ["-m", "pip", "freeze", "--all"])

        def drifted(_argv, **_kwargs):
            return types.SimpleNamespace(returncode=0, stdout=b"different==1\n")

        with self.assertRaisesRegex(lepton_job.LeptonJobError, "differs"):
            lepton_job.verify_python_environment_freeze(contract, runner=drifted)

        verifier.chmod(0o644)
        verifier.write_text("raise SystemExit(1)\n")
        verifier.chmod(0o444)
        with self.assertRaisesRegex(lepton_job.LeptonJobError, "inventory-bound"):
            lepton_job.verify_sealed_contract_source(contract, runner=runner)

    def test_in_job_executor_runs_supply_chain_gates_before_training(self):
        project = self.root / "remote-project"
        source = project / "src/sealed"
        entrypoint = source / "scripts/round_dagger/run_lepton_h200_learner_job.sh"
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("#!/bin/sh\nexit 0\n")
        entrypoint.chmod(0o555)
        contract = replace(
            self.contract,
            project_root=project,
            source_root=source,
            entrypoint=entrypoint,
            python=Path(sys.executable),
        )
        paths = dict(self.request.paths)
        paths["source_root"] = source
        request = replace(self.request, paths=paths)
        events: list[str] = []

        def learner_runner(_argv, **_kwargs):
            events.append("train")
            return types.SimpleNamespace(returncode=19)

        with (
            mock.patch.object(lepton_job, "load_contract", return_value=contract),
            mock.patch.object(lepton_job, "load_request", return_value=request),
            mock.patch.object(
                lepton_job,
                "verify_sealed_contract_source",
                side_effect=lambda *_args, **_kwargs: events.append("source"),
            ),
            mock.patch.object(
                lepton_job,
                "verify_python_environment_freeze",
                side_effect=lambda *_args, **_kwargs: events.append("freeze"),
            ),
            mock.patch.dict(
                os.environ, {lepton_job.WANDB_SECRET_ENV: "redacted"}, clear=False
            ),
        ):
            with self.assertRaisesRegex(lepton_job.LeptonJobError, "status 19"):
                lepton_job.execute_request(
                    contract_path=self.root / "contract.json",
                    contract_sha256=contract.sha256,
                    request_path=self.root / "request.json",
                    request_sha256=request.sha256,
                    semantic_attempt=1,
                    runner=learner_runner,
                )
        self.assertEqual(events, ["source", "freeze", "train"])

    def test_in_job_executor_packs_exact_output_then_publishes_bound_marker(self):
        project = self.root / "remote-project-pack"
        output = project / "campaign/rounds/macro_round_000000/learner_output"
        output.mkdir(parents=True)
        result = output / "learner_result_round_000000.json"
        attestation = output / "learner_completion_round_000000.json"
        marker = output / "lepton_learner_success.json"
        checksum = Path(f"{marker}.sha256")
        contract = replace(
            self.contract,
            project_root=project,
            source_root=ROOT,
            entrypoint=ROOT / "scripts/round_dagger/run_lepton_h200_learner_job.sh",
            python=Path(sys.executable),
        )
        paths = {
            **self.request.paths,
            "source_root": ROOT,
            "learner_output": output,
            "learner_result": result,
            "completion_attestation": attestation,
            "success_marker": marker,
            "success_checksum": checksum,
        }
        environment = {
            **self.request.environment,
            "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(output),
            "ROUND_DAGGER_LEARNER_RESULT": str(result),
        }
        request = replace(self.request, paths=paths, environment=environment)
        events: list[str] = []

        def runner(argv, **kwargs):
            if Path(argv[0]).name == "run_h100_learner.sh":
                events.append("train")
                self.assertFalse(marker.exists())
                result.write_text('{"status":"PASS"}\n')
                attestation.write_text('{"status":"PASS"}\n')
                return types.SimpleNamespace(returncode=0)
            self.assertEqual(
                Path(argv[1]).name, "lepton_campaign_archive_transfer.py"
            )
            events.append("pack")
            self.assertFalse(marker.exists())
            return __import__("subprocess").run(argv, **kwargs)

        patches = (
            mock.patch.object(lepton_job, "load_contract", return_value=contract),
            mock.patch.object(lepton_job, "load_request", return_value=request),
            mock.patch.object(lepton_job, "verify_sealed_contract_source"),
            mock.patch.object(lepton_job, "verify_python_environment_freeze"),
            mock.patch.dict(
                os.environ, {lepton_job.WANDB_SECRET_ENV: "redacted"}, clear=False
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            execution = lepton_job.execute_request(
                contract_path=self.root / "contract.json",
                contract_sha256=contract.sha256,
                request_path=self.root / "request.json",
                request_sha256=request.sha256,
                semantic_attempt=1,
                runner=runner,
            )
        self.assertEqual(events, ["train", "pack"])
        self.assertEqual(execution["publication"], "published")
        document = json.loads(marker.read_text())
        self.assertEqual(
            document["schema"],
            "fast_roborender.round_dagger.remote_learner_success.v4",
        )
        bundle_paths = lepton_job.learner_output_bundle_paths(contract, request)
        self.assertEqual(
            document["learner_output_manifest_path"],
            str(bundle_paths["manifest"]),
        )
        observed = lepton_job._validate_learner_output_bundle(
            contract=contract,
            request=request,
            manifest_sha256=document["learner_output_manifest_sha256"],
            archive_sha256=document["learner_output_archive_sha256"],
            inventory_sha256=document["learner_output_inventory_sha256"],
            tree_sha256=document["learner_output_tree_sha256"],
        )
        self.assertEqual(
            execution["learner_output_manifest_sha256"],
            observed["manifest_sha256"],
        )
        self.assertEqual(checksum.read_bytes(), lepton_job.checksum_payload(marker))
        self.assertFalse(Path(bundle_paths["archive"]).stat().st_mode & 0o222)
        self.assertFalse(Path(bundle_paths["inventory"]).stat().st_mode & 0o222)
        self.assertFalse(Path(bundle_paths["manifest"]).stat().st_mode & 0o222)

        def must_not_run(*_args, **_kwargs):
            raise AssertionError("an exact completed identity must be adopted")

        with (
            mock.patch.object(lepton_job, "load_contract", return_value=contract),
            mock.patch.object(lepton_job, "load_request", return_value=request),
            mock.patch.object(lepton_job, "verify_sealed_contract_source"),
            mock.patch.object(lepton_job, "verify_python_environment_freeze"),
            mock.patch.dict(
                os.environ, {lepton_job.WANDB_SECRET_ENV: "redacted"}, clear=False
            ),
        ):
            adopted = lepton_job.execute_request(
                contract_path=self.root / "contract.json",
                contract_sha256=contract.sha256,
                request_path=self.root / "request.json",
                request_sha256=request.sha256,
                semantic_attempt=2,
                runner=must_not_run,
            )
        self.assertEqual(adopted["publication"], "adopted")
        self.assertEqual(
            adopted["success_marker_sha256"],
            execution["success_marker_sha256"],
        )

    def test_in_job_executor_never_publishes_success_when_bundle_pack_fails(self):
        project = self.root / "remote-project-pack-failure"
        output = project / "campaign/learner_output"
        output.mkdir(parents=True)
        result = output / "learner_result_round_000000.json"
        attestation = output / "learner_completion_round_000000.json"
        marker = output / "lepton_learner_success.json"
        contract = replace(
            self.contract,
            project_root=project,
            source_root=ROOT,
            entrypoint=ROOT / "scripts/round_dagger/run_lepton_h200_learner_job.sh",
            python=Path(sys.executable),
        )
        request = replace(
            self.request,
            paths={
                **self.request.paths,
                "source_root": ROOT,
                "learner_output": output,
                "learner_result": result,
                "completion_attestation": attestation,
                "success_marker": marker,
                "success_checksum": Path(f"{marker}.sha256"),
            },
        )

        def runner(argv, **_kwargs):
            if Path(argv[0]).name == "run_h100_learner.sh":
                result.write_text("result\n")
                attestation.write_text("attestation\n")
                return types.SimpleNamespace(returncode=0)
            return types.SimpleNamespace(returncode=29, stdout=b"")

        with (
            mock.patch.object(lepton_job, "load_contract", return_value=contract),
            mock.patch.object(lepton_job, "load_request", return_value=request),
            mock.patch.object(lepton_job, "verify_sealed_contract_source"),
            mock.patch.object(lepton_job, "verify_python_environment_freeze"),
            mock.patch.dict(
                os.environ, {lepton_job.WANDB_SECRET_ENV: "redacted"}, clear=False
            ),
        ):
            with self.assertRaisesRegex(lepton_job.LeptonJobError, "status 29"):
                lepton_job.execute_request(
                    contract_path=self.root / "contract.json",
                    contract_sha256=contract.sha256,
                    request_path=self.root / "request.json",
                    request_sha256=request.sha256,
                    semantic_attempt=1,
                    runner=runner,
                )
        self.assertFalse(marker.exists())
        self.assertFalse(Path(f"{marker}.sha256").exists())

    def test_live_bridge_fake_orders_stop_job_restore_then_fetch(self):
        self.request_path.chmod(0o444)
        transfer = self.root / "transfer"
        transfer.write_text("#!/bin/sh\nexit 0\n")
        transfer.chmod(0o700)
        control_exec = self.root / "control-exec"
        control_exec.write_text("#!/bin/sh\nexit 0\n")
        control_exec.chmod(0o700)
        client = FakeClient()
        deployment = FakeDeploymentClient()
        events: list[str] = []
        completed_job = _job(self.request, self.contract, 1, "Completed")
        original_scale = deployment.scale

        def scale(name, replicas):
            events.append("stop" if replicas == 0 else "restart")
            return original_scale(name, replicas)

        deployment.scale = scale

        def submit_attempt(*_args, **_kwargs):
            events.append("job_submit")
            return completed_job, False

        def fetch(**_kwargs):
            self.assertEqual(deployment.state, "Ready")
            events.append("fetch_after_ready")
            owner = tempfile.TemporaryDirectory(dir=self.root)
            marker = Path(owner.name) / "lepton_learner_success.json"
            document = {
                "schema": "fast_roborender.round_dagger.remote_learner_success.v4",
                "status": "PASS",
                "campaign_id": self.request.campaign_id,
                "macro_round": self.request.macro_round,
                "learner_code_sha256": self.request.environment[
                    "ROUND_DAGGER_LEARNER_CODE_SHA256"
                ],
                "learner_config_sha256": self.request.environment[
                    "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
                ],
                "learner_identity_sha256": self.request.identity_sha256,
                "result_path": self.request.paths["learner_result"].name,
                "result_sha256": "a" * 64,
                "completion_attestation_path": self.request.paths[
                    "completion_attestation"
                ].name,
                "completion_attestation_sha256": "b" * 64,
                "learner_output_manifest_path": str(
                    lepton_job.PROJECT_ROOT
                    / "results"
                    / self.request.identity_sha256
                    / "learner_output.manifest.json"
                ),
                "learner_output_manifest_sha256": "c" * 64,
                "learner_output_archive_sha256": "d" * 64,
                "learner_output_inventory_sha256": "e" * 64,
                "learner_output_tree_sha256": "f" * 64,
            }
            marker.write_text(json.dumps(document))
            checksum = Path(f"{marker}.sha256")
            checksum.write_text(f"{'c' * 64}  {marker.name}\n")
            return marker, checksum, owner

        arguments = Namespace(
            command="submit",
            apply=True,
            exclusive_node_lifecycle=True,
            lifecycle_state=str(self.root / "bridge-lifecycle.json"),
            control_exec_program=str(control_exec),
            control_timeout_seconds=30,
            wait_timeout_seconds=30,
            transfer_program=str(transfer),
            contract=str(CONTRACT),
            contract_sha256=self.contract.sha256,
            request=str(self.request_path),
            request_sha256=self.request.sha256,
            remote_contract=(
                "/workspace/projects/roborender/manifests/backend.json"
            ),
            remote_request=(
                "/workspace/projects/roborender/handoff/incoming/request.json"
            ),
        )
        with (
            mock.patch.object(submitter, "_clients", return_value=(client, deployment)),
            mock.patch.object(submitter, "require_finalized_supply_chain"),
            mock.patch.object(
                submitter,
                "verify_reserved_node_job_exclusivity",
                side_effect=lambda *_args, **_kwargs: events.append("job_inventory"),
            ),
            mock.patch.object(
                submitter,
                "verify_control_preflight",
                side_effect=lambda **_kwargs: events.append("remote_preflight"),
            ),
            mock.patch.object(
                submitter, "submit_or_adopt_attempt", side_effect=submit_attempt
            ),
            mock.patch.object(submitter, "fetch_completion_pair", side_effect=fetch),
        ):
            self.assertEqual(submitter._submit(arguments), 0)
        self.assertEqual(
            events,
            [
                "job_inventory",
                "remote_preflight",
                "stop",
                "job_submit",
                "restart",
                "fetch_after_ready",
            ],
        )
        self.assertLess(
            deployment.events.index("scale:leo-cosmos-h200-dev:0"),
            deployment.events.index("scale:leo-cosmos-h200-dev:1"),
        )


if __name__ == "__main__":
    unittest.main()
