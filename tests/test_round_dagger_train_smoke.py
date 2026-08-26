"""Content, mapping, and Docker-isolation gates for real train smoke cohorts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import publish_immutable_json, sha256_file
from round_dagger.train_smoke import (
    TrainSmokeError,
    prepare_train_smoke,
    train_smoke_environment,
    validate_train_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/round_dagger/run_actor_train_smoke_docker.sh"
RENDERER_SHA = "a" * 64


class TrainSmokePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "train-smoke"
        self.gr00t = self.root / "gr00t"
        self.gr00t.mkdir()
        self.anchors = self.root / "anchors"
        self.anchors.mkdir()
        for index in range(2048):
            (self.anchors / f"anchor_{index:04d}.npz").touch()

        self.checkpoint = self.root / "student-v0.pt"
        self.checkpoint.write_bytes(b"fresh policy")
        self.spec = Path(f"{self.checkpoint}.policy_spec.json")
        publish_immutable_json(self.spec, {"schema": "test-policy-spec"})
        self.policy_manifest = Path(f"{self.checkpoint}.manifest.json")
        publish_immutable_json(
            self.policy_manifest,
            {
                "schema": "fast_roborender.round_dagger.bootstrap_policy.v1",
                "checkpoint_path": self.checkpoint.name,
                "checkpoint_sha256": sha256_file(self.checkpoint),
                "runtime_policy_sha256": "b" * 64,
                "policy_spec_path": self.spec.name,
                "policy_spec_sha256": sha256_file(self.spec),
            },
        )
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"teacher")

        self.fleet = self.root / "fleet"
        endpoints = self.fleet / "endpoints"
        endpoints.mkdir(parents=True)
        self.descriptors = []
        for slot, port in enumerate((41001, 41002)):
            descriptor = endpoints / f"renderer_{slot:03d}.txt"
            descriptor.write_text(f"127.0.0.1 {port}\n", encoding="utf-8")
            self.descriptors.append(descriptor)
        self.source_manifest = self.fleet / "endpoint_manifest.txt"
        self.source_manifest.write_text(
            "".join(f"{item}\n" for item in self.descriptors), encoding="utf-8"
        )
        publish_immutable_json(
            self.fleet / "plan.json",
            {
                "schema": "fast_roborender.local_image3f_fleet.v1.plan",
                "fleet_dir": str(self.fleet),
                "gpus": ["0", "1"],
            },
        )
        publish_immutable_json(
            self.fleet / "readiness.json",
            {
                "schema": "fast_roborender.local_image3f_fleet.v1.readiness",
                "capable": 2,
                "discovered": 2,
                "endpoints": [
                    {
                        "file": str(descriptor),
                        "response": {
                            "ok": True,
                            "protocol_version": 2,
                            "renderer_identity_sha256": RENDERER_SHA,
                        },
                    }
                    for descriptor in self.descriptors
                ],
            },
        )
        self.plan = self.output / "plan.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self) -> dict:
        return prepare_train_smoke(
            self.output,
            endpoint_manifest=self.source_manifest,
            policy_checkpoint=self.checkpoint,
            teacher_checkpoint=self.teacher,
            renderer_sha256=RENDERER_SHA,
            anchors=self.anchors,
            repository_root=ROOT,
            gr00t_root=self.gr00t,
            actor_image="actor:test",
        )

    def test_exact_four_cohort_contract_and_renderer_gpu_schedule(self) -> None:
        report = self.prepare()
        self.assertEqual(report["cohort_count"], 4)
        self.assertEqual(report["expected_transitions"], 8)
        self.assertEqual(report["global_env_ids"], [0, 1, 2, 3])
        self.assertEqual(report["gpu_schedule"], [0, 1, 0, 1])
        self.assertEqual(report["renderer_schedule"], [0, 1, 0, 1])
        self.assertEqual(report["states"], ["empty"] * 4)

        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        self.assertEqual(plan["acknowledgement"], "bounded-train-replay")
        for index, item in enumerate(plan["cohorts"]):
            contract = json.loads(
                (Path(item["cohort_dir"]) / "contract.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(contract["campaign_id"], "rtx-train-smoke-v1")
            self.assertEqual(contract["cohort_index"], index)
            self.assertEqual(contract["purpose"], "train")
            self.assertEqual(contract["expected_ranks"], [0])
            self.assertEqual(contract["expected_global_env_ids"], [index])
            self.assertEqual(contract["expected_steps_per_env"], 2)
            self.assertEqual(contract["expected_transitions"], 2)
            self.assertEqual(
                Path(item["endpoint_manifest"]).read_text(encoding="utf-8"),
                f"{self.descriptors[index % 2]}\n",
            )

        # Identical preparation is create-only and idempotent.
        self.assertEqual(self.prepare()["plan_sha256"], report["plan_sha256"])

    def test_launch_environment_is_exact_and_has_no_distributed_identity(self) -> None:
        self.prepare()
        environment = train_smoke_environment(self.plan, 3)
        self.assertEqual(environment["ROUND_DAGGER_VALIDATION_RANKS"], "1")
        self.assertEqual(environment["ROUND_DAGGER_VALIDATION_ENVS_PER_RANK"], "1")
        self.assertEqual(environment["ROUND_DAGGER_VALIDATION_STEPS"], "2")
        self.assertEqual(environment["ROUND_DAGGER_VALIDATION_PURPOSE"], "train")
        self.assertEqual(
            environment["ROUND_DAGGER_VALIDATION_TRAIN_ACK"],
            "bounded-train-replay",
        )
        self.assertEqual(environment["ROUND_DAGGER_GLOBAL_ENV_OFFSET"], "3")
        self.assertEqual(environment["ROUND_DAGGER_ACTOR_DOCKER_GPU_ID"], "1")
        self.assertEqual(
            environment["IMAGE3F_ENDPOINT_MANIFEST"],
            str(self.output / "cohort_03/renderer_endpoint_manifest.txt"),
        )
        for forbidden in ("RANK", "WORLD_SIZE", "ROUND_DAGGER_ACTOR_RANK"):
            self.assertNotIn(forbidden, environment)

    def test_changed_endpoint_or_fleet_evidence_fails_closed(self) -> None:
        self.prepare()
        self.descriptors[0].write_text("127.0.0.1 49999\n", encoding="utf-8")
        with self.assertRaisesRegex(TrainSmokeError, "descriptor changed"):
            validate_train_smoke(self.plan)

    def test_nonmatching_renderer_identity_fails_before_publication(self) -> None:
        plan_sha = sha256_file(self.plan) if self.plan.exists() else None
        readiness_path = self.fleet / "readiness.json"
        readiness_path.chmod(0o644)
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness["endpoints"][1]["response"]["renderer_identity_sha256"] = "c" * 64
        readiness_path.write_text(
            json.dumps(readiness, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(TrainSmokeError, "identity differs"):
            self.prepare()
        if plan_sha is None:
            self.assertFalse(self.plan.exists())
        else:
            self.assertEqual(sha256_file(self.plan), plan_sha)


class TrainSmokeDockerLauncherTest(TrainSmokePlanTest):
    def setUp(self) -> None:
        super().setUp()
        self.prepare()
        self.argv_file = self.root / "docker.argv"
        self.fake_docker = self.root / "docker"
        self.fake_docker.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$@\" > {self.argv_file!s}\n",
            encoding="utf-8",
        )
        self.fake_docker.chmod(0o755)

    def run_launcher(
        self, cohort_index: int, *, acknowledgement: bool = True, ipc: str = "private"
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "ROUND_DAGGER_DOCKER_BIN": str(self.fake_docker),
            "ROUND_DAGGER_ACTOR_DOCKER_IPC": ipc,
            "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": "24g",
            "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": str(self.root),
            "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow",
            "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
            "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_EVIDENCE": "/host/adapter-evidence.json",
            "ROUND_DAGGER_RUNTIME_ADAPTER_PLAN": "/host/adapter-plan.json",
            "ROUND_DAGGER_RUNTIME_AB_EVIDENCE": "/host/ab-evidence.json",
            "ROUND_DAGGER_RUNTIME_AB_PLAN": "/host/ab-plan.json",
            "ROUND_DAGGER_RUNTIME_AB_MODE": "shadow",
        }
        environment.pop("ROUND_DAGGER_VALIDATION_TRAIN_ACK", None)
        if acknowledgement:
            environment["ROUND_DAGGER_VALIDATION_TRAIN_ACK"] = (
                "bounded-train-replay"
            )
        return subprocess.run(
            ["bash", str(LAUNCHER), str(self.plan), str(cohort_index)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_launcher_uses_planned_gpu_private_ipc_and_no_wandb_secret(self) -> None:
        result = self.run_launcher(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv_file.read_text(encoding="utf-8").splitlines()
        self.assertIn("--name", argv)
        self.assertIn("rdt-rtx-train-smoke-v1-m0-c1-r0", argv)
        self.assertIn('"device=1"', argv)
        self.assertIn("--ipc=private", argv)
        self.assertIn("--shm-size=24g", argv)
        self.assertIn("actor:test", argv)
        self.assertIn(str(ROOT / "scripts/round_dagger/run_actor_validation.sh"), argv)
        self.assertNotIn("WANDB_API_KEY", argv)
        self.assertNotIn("ROUND_DAGGER_ACTOR_RANK", argv)
        self.assertIn("ROUND_DAGGER_RUNTIME_ADAPTER_MODE=legacy", argv)
        self.assertIn("ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK=", argv)
        self.assertNotIn("metadata-only-v1", argv)
        runtime_args = [
            value
            for value in argv
            if value.startswith("ROUND_DAGGER_RUNTIME_ADAPTER_")
            or value.startswith("ROUND_DAGGER_RUNTIME_AB_")
        ]
        self.assertEqual(
            runtime_args,
            [
                "ROUND_DAGGER_RUNTIME_ADAPTER_MODE=legacy",
                "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK=",
                "ROUND_DAGGER_RUNTIME_ADAPTER_VALIDATION_ACK=",
                "ROUND_DAGGER_RUNTIME_ADAPTER_INPUT_PROFILE=",
                "ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE=",
                "ROUND_DAGGER_RUNTIME_AB_PLAN=",
                "ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256=",
                "ROUND_DAGGER_RUNTIME_AB_ARM=",
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT=",
                "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256=",
                "ROUND_DAGGER_RUNTIME_AB_EVIDENCE=",
            ],
        )
        for host_value in (
            "/host/adapter-evidence.json",
            "/host/adapter-plan.json",
            "/host/ab-evidence.json",
            "/host/ab-plan.json",
            "shadow",
        ):
            self.assertNotIn(host_value, argv)

    def test_explicit_acknowledgement_is_required_before_docker(self) -> None:
        result = self.run_launcher(0, acknowledgement=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires explicit", result.stderr)
        self.assertFalse(self.argv_file.exists())

    def test_host_ipc_is_rejected_before_docker(self) -> None:
        result = self.run_launcher(2, ipc="host")
        self.assertEqual(result.returncode, 2)
        self.assertIn("private IPC", result.stderr)
        self.assertFalse(self.argv_file.exists())


if __name__ == "__main__":
    unittest.main()
