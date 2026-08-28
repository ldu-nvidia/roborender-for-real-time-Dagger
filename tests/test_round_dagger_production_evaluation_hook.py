"""Fail-closed tests for the candidate-aware production evaluation hook."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import canonical_json_sha256, sha256_file
from round_dagger.evaluation import DEFAULT_THRESHOLDS
from round_dagger.production_data_lineage import (
    RUNTIME_CONTRACT_SCHEMA,
    data_lineage_sha256,
    derive_data_lineage,
)
from round_dagger.state import load_campaign, load_state
from scripts.round_dagger.production_launch import TEACHER_CHECKPOINT_RELATIVE

try:
    from test_round_dagger_state import _campaign, _sha, _stage_and_bind, _to_ingested
except ImportError:  # pragma: no cover - package-style unittest invocation
    from tests.test_round_dagger_state import (  # type: ignore[no-redef]
        _campaign,
        _sha,
        _stage_and_bind,
        _to_ingested,
    )


class ProductionPairedEvaluationHookTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(
            "/mnt/data2/ldu/roborender_round_dagger_validation/unit_tests"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.tmp = Path(self.temporary.name)
        self.live_repository_root = Path(__file__).parents[1]
        self.repository_root = self.tmp / "evaluation-seal"
        for directory in ("round_dagger", "scripts", "configs"):
            shutil.copytree(
                self.live_repository_root / directory,
                self.repository_root / directory,
            )
        (self.tmp / "evaluation-seal.inventory.json").write_text("{}\n")
        (self.tmp / "evaluation-seal.snapshot_attestation.json").write_text("{}\n")
        self.teacher_checkpoint = self.tmp / "configured-teacher.pt"
        self.sealed_teacher = self.repository_root / TEACHER_CHECKPOINT_RELATIVE
        # _campaign pins sha256(b"teacher") as the expert identity.
        teacher_bytes = b"teacher"
        self.teacher_checkpoint.write_bytes(teacher_bytes)
        self.sealed_teacher.parent.mkdir(parents=True, exist_ok=True)
        self.sealed_teacher.write_bytes(teacher_bytes)
        self.teacher_checkpoint.chmod(0o600)
        self.sealed_teacher.chmod(0o444)
        self.teacher_sha = sha256_file(self.teacher_checkpoint)
        self.anchor_pool = self.tmp / "anchors"
        self.anchor_pool.mkdir()
        (self.anchor_pool / "anchor_000000.npz").write_bytes(b"anchor fixture")
        self.anchor_inventory_sha = _sha("anchor-inventory")
        self.config = self.tmp / "evaluation_config.yaml"
        self.config.write_text(
            f"teacher_actor_path: {self.teacher_checkpoint}\n"
            "algo:\n"
            "  config:\n"
            "    videogen:\n"
            f"      anchor_pool_dir: {self.anchor_pool}\n"
            "      anchor_index: 0\n"
            "fixed_seed: true\n",
            encoding="utf-8",
        )
        self.gr00t = self.repository_root / "GR00T-VisualSim2Real"
        split_relative = Path(
            "gr00t/rl/data/motions/g1_wsdpt/"
            "33demos_675_775_heldout_256_v1.json"
        )
        split_source = (
            self.live_repository_root / "GR00T-VisualSim2Real" / split_relative
        )
        split_target = self.gr00t / split_relative
        split_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_source, split_target)
        self.motion_split = split_target
        self.motion_split_sha = sha256_file(split_target)
        self.heldout_plan = (
            self.repository_root
            / "configs/round_dagger/production_heldout_motion_plan_8x32.json"
        )
        self.heldout_plan_sha = sha256_file(self.heldout_plan)
        plan = json.loads(self.heldout_plan.read_text(encoding="utf-8"))
        scenarios = [
            {
                "seed": case["seed"],
                "env_index": case["env_index"],
                "scenario_id": case["planned_case_id"],
            }
            for case in plan["capture"]["cases"]
        ]
        scenario_path = (
            self.repository_root
            / "configs/round_dagger/production_paired_scenarios_2x8.json"
        )
        scenario_path.write_text(
            json.dumps(scenarios, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.root = _campaign(self.tmp)
        _to_ingested(self.root)
        self.candidate = _stage_and_bind(self.root, self.tmp)
        self.state = load_state(self.root)
        self.campaign = load_campaign(self.root)
        for relative in (
            "gr00t/rl/eval_agent_trl.py",
            "gr00t/rl/trl/trainer/anchor_io.py",
            "gr00t/rl/trl/trainer/checkpoint_io.py",
            "gr00t/rl/trl/trainer/distill_trainer.py",
            "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
            "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
            "gr00t/rl/trl/trainer/ppo_trainer.py",
            "gr00t/rl/utils/image3f_frame_client.py",
            "gr00t/rl/utils/round_dagger_paired_eval.py",
            "gr00t/rl/utils/simulator_rgb_anchor.py",
        ):
            source = self.gr00t / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# fixture: {relative}\n", encoding="utf-8")
        sealed_campaign_path = (
            self.repository_root
            / "configs/round_dagger/production_campaign_256x512.json"
        )
        sealed_campaign = json.loads(sealed_campaign_path.read_text(encoding="utf-8"))
        sealed_campaign["artifacts"]["anchor_pool"] = {
            "path": str(self.anchor_pool),
            "anchor_count": 1,
            "inventory_sha256": self.anchor_inventory_sha,
            "inventory_algorithm": "len64be(path_utf8)+path_utf8+size64be+sha256_binary",
            "usage": "compatibility_fallback_only",
        }
        sealed_campaign_path.write_text(json.dumps(sealed_campaign), encoding="utf-8")
        self.endpoints = self.tmp / "endpoints"
        self.endpoints.mkdir()
        for rank in range(2):
            (self.endpoints / f"renderer_{rank}.txt").write_text(
                f"127.0.0.1 {19000 + rank}\n", encoding="utf-8"
            )
        self.endpoint_manifest = self.tmp / "endpoint_manifest.txt"
        self.endpoint_manifest.write_text(
            "".join(
                f"{self.endpoints / f'renderer_{rank}.txt'}\n"
                for rank in range(2)
            ),
            encoding="utf-8",
        )
        self.capture_attestation = self.tmp / "capture_attestation.json"
        self.capture_evidence = self.tmp / "capture_evidence.json"
        camera_acceptance = self.tmp / "camera_acceptance.json"
        camera_acceptance.write_text(
            json.dumps({"reviewer": "fixture-reviewer"}) + "\n",
            encoding="utf-8",
        )
        capture_passes = []
        for seed in (17, 29, 43, 59, 71, 89, 101, 127):
            for pass_index in (1, 2):
                path = self.tmp / f"capture_pass_{seed}_{pass_index}.json"
                path.write_text("{}\n", encoding="utf-8")
                capture_passes.append(
                    {
                        "seed": seed,
                        "pass_index": pass_index,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )
        bundle_root = self.tmp / "first-condition-bundles"
        bundle_root.mkdir()
        first_condition_bundles = []
        for seed in (17, 29, 43, 59, 71, 89, 101, 127):
            path = bundle_root / f"seed_{seed}.npz"
            path.write_bytes(f"first-condition:{seed}\n".encode())
            path.chmod(0o444)
            first_condition_bundles.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                    "case_count": 32,
                    "source_pass_index": 1,
                }
            )
        self.first_condition_bundles = first_condition_bundles
        self.first_condition_bundle_inventory_sha = canonical_json_sha256(
            first_condition_bundles
        )
        capture_inventory = {
            "first_condition_bundles": first_condition_bundles,
            "first_condition_bundle_inventory_sha256": (
                self.first_condition_bundle_inventory_sha
            ),
        }
        self.capture_attestation.write_text(
            json.dumps(capture_inventory, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        self.capture_evidence.write_text(
            json.dumps(capture_inventory, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        runtime_contract_document = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
            "status": "release_eligible",
            "invariant_profile_sha256": sha256_file(sealed_campaign_path),
            "camera_acceptance": {
                "path": str(camera_acceptance),
                "sha256": sha256_file(camera_acceptance),
            },
            "renderer_fleet": {
                "endpoint_manifest_path": str(self.endpoint_manifest),
                "endpoint_manifest_sha256": sha256_file(self.endpoint_manifest),
                "endpoint_directory": str(self.endpoints),
                "renderer_count": 2,
                "cohort_identity_sha256": _sha("renderer-cohort"),
                "runtime_identity_sha256": _sha("renderer-runtime"),
            },
            "live_capture": {
                "split_manifest_path": str(self.motion_split),
                "split_manifest_sha256": self.motion_split_sha,
                "heldout_plan_path": str(self.heldout_plan),
                "heldout_plan_sha256": self.heldout_plan_sha,
                "pass_artifacts": capture_passes,
                "attestation_path": str(self.capture_attestation),
                "attestation_sha256": sha256_file(self.capture_attestation),
                "evidence_path": str(self.capture_evidence),
                "evidence_sha256": sha256_file(self.capture_evidence),
                "scenario_path": str(scenario_path),
                "scenario_sha256": sha256_file(scenario_path),
                "first_condition_bundles": first_condition_bundles,
                "first_condition_bundle_inventory_sha256": (
                    self.first_condition_bundle_inventory_sha
                ),
            },
        }
        self.runtime_contract = self.tmp / "production_runtime_contract.json"
        self.runtime_contract.write_text(
            json.dumps(
                runtime_contract_document,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self.runtime_contract_sha = sha256_file(self.runtime_contract)
        normalized_runtime = {
            **runtime_contract_document,
            "contract_sha256": self.runtime_contract_sha,
            "capture_provenance": {
                "motion_split_manifest_sha256": self.motion_split_sha,
                "heldout_plan_sha256": self.heldout_plan_sha,
            },
        }
        self.data_lineage = derive_data_lineage(
            normalized_runtime,
            motion_split_contract=sealed_campaign["motion_split"],
            topology=sealed_campaign["topology"],
        )
        self.data_lineage_sha = data_lineage_sha256(self.data_lineage)

        self.image_id_sha = _sha("actor-image-id")
        self.docker_log = self.tmp / "docker.log"
        self.docker = self.tmp / "fake-docker"
        self.docker.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "printf '%s\\n' \"$*\" >>\"$DOCKER_LOG\"\n"
            "if [[ ${1:-} == image && ${2:-} == inspect ]]; then\n"
            f"  printf '%s\\n' 'sha256:{self.image_id_sha}'\n"
            "fi\n",
            encoding="utf-8",
        )
        self.docker.chmod(0o755)
        round_dir = self.root / "rounds/macro_round_000000/evaluation"
        active = self.state["active_policy"]
        self.contract = round_dir / "contract.json"
        self.result = round_dir / "result.json"
        self.script = (
            self.repository_root
            / "scripts/round_dagger/run_production_paired_evaluation_hook.sh"
        )
        self.control_python = self.tmp / "control-python"
        self.control_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "if [[ ${1:-} == -m && ${2:-} == "
            "round_dagger.production_data_lineage && ${3:-} == verify ]]; then\n"
            "  printf '{\"status\":\"PASS\",\"data_lineage_sha256\":\"%s\"}\\n' "
            "\"$FIXTURE_DATA_LINEAGE_SHA\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ ${1:-} == -m && ${2:-} == round_dagger.evaluation "
            "&& ${3:-} == prepare ]]; then\n"
            "  output=\n"
            "  while (($#)); do\n"
            "    if [[ $1 == --output ]]; then output=$2; break; fi\n"
            "    shift\n"
            "  done\n"
            "  [[ -n $output ]]\n"
            "  mkdir -p \"$(dirname \"$output\")\"\n"
            "  printf '{\"schema\":\"fixture-contract\"}\\n' >\"$output\"\n"
            "  exit 0\n"
            "fi\n"
            "for value in \"$@\"; do\n"
            "  case \"$value\" in\n"
            "    *verify_sealed_source_snapshot.py|*verify_anchor_pool.py) exit 0 ;;\n"
            "  esac\n"
            "done\n"
            f"exec {sys.executable} \"$@\"\n",
            encoding="utf-8",
        )
        self.control_python.chmod(0o755)
        self.environment = {
            **os.environ,
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(self.root),
            "ROUND_DAGGER_CAMPAIGN_ID": self.campaign["campaign_id"],
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(
                self.root / active["checkpoint_path"]
            ),
            "ROUND_DAGGER_POLICY_SHA256": active["checkpoint_sha256"],
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": active[
                "runtime_policy_sha256"
            ],
            "ROUND_DAGGER_CANDIDATE_MANIFEST": str(
                self.root / self.candidate["manifest_path"]
            ),
            "ROUND_DAGGER_CANDIDATE_SHA256": self.candidate["manifest_sha256"],
            "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": sha256_file(self.config),
            "ROUND_DAGGER_EVALUATOR_RESULT": str(self.result),
            "ROUND_DAGGER_EVALUATION_CONFIG": str(self.config),
            "ROUND_DAGGER_EVALUATION_SCENARIOS": str(
                self.repository_root
                / "configs/round_dagger/production_paired_scenarios_2x8.json"
            ),
            "ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256": sha256_file(
                self.repository_root
                / "configs/round_dagger/production_paired_scenarios_2x8.json"
            ),
            "ROUND_DAGGER_EVALUATION_MOTION_SPLIT_MANIFEST": str(
                self.motion_split
            ),
            "ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256": self.motion_split_sha,
            "ROUND_DAGGER_HELDOUT_EVALUATION_PLAN": str(self.heldout_plan),
            "ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256": self.heldout_plan_sha,
            "ROUND_DAGGER_CAMPAIGN_CONTRACT": str(self.runtime_contract),
            "ROUND_DAGGER_CAMPAIGN_CONTRACT_SHA256": self.runtime_contract_sha,
            "ROUND_DAGGER_DATA_LINEAGE_SHA256": self.data_lineage_sha,
            "ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256": (
                self.first_condition_bundle_inventory_sha
            ),
            "ROUND_DAGGER_CAPTURE_ATTESTATION": str(
                self.capture_attestation
            ),
            "ROUND_DAGGER_CAPTURE_ATTESTATION_SHA256": sha256_file(
                self.capture_attestation
            ),
            "ROUND_DAGGER_CAPTURE_EVIDENCE": str(self.capture_evidence),
            "ROUND_DAGGER_CAPTURE_EVIDENCE_SHA256": sha256_file(
                self.capture_evidence
            ),
            "ROUND_DAGGER_EVALUATION_THRESHOLDS": str(
                self.repository_root
                / "configs/round_dagger/production_paired_thresholds.json"
            ),
            "ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256": sha256_file(
                self.repository_root
                / "configs/round_dagger/production_paired_thresholds.json"
            ),
            "ROUND_DAGGER_TEACHER_CHECKPOINT": str(self.teacher_checkpoint),
            "ROUND_DAGGER_TEACHER_SHA256": self.teacher_sha,
            "ROUND_DAGGER_EVALUATION_SEALED_TEACHER": str(self.sealed_teacher),
            "ROUND_DAGGER_TEACHER_CHECKPOINT_RELATIVE": TEACHER_CHECKPOINT_RELATIVE.as_posix(),
            "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": str(self.repository_root),
            "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH": "1",
            "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": _sha("evaluation-inventory"),
            "ROUND_DAGGER_EVALUATION_ENDPOINT_DIR": str(self.endpoints),
            "ROUND_DAGGER_RENDERER_RUNTIME_SHA256": _sha("renderer-runtime"),
            "ROUND_DAGGER_EXPECTED_RENDERERS": "2",
            "GR00T_PYTHON": "/opt/round-dagger-rtx/bin/python",
            "ROUND_DAGGER_DOCKER_BIN": str(self.docker),
            "ROUND_DAGGER_RTX_IMAGE": "round-dagger:test",
            "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": self.image_id_sha,
            # Preserve the virtual-environment entry point.  Resolving this
            # symlink would execute the base interpreter without its packages.
            "ROUND_DAGGER_CONTROL_PYTHON": str(self.control_python),
            "ANCHORS": str(self.anchor_pool),
            "ROUND_DAGGER_MIN_ANCHORS": "1",
            "ROUND_DAGGER_ANCHOR_INDEX": "0",
            "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": self.anchor_inventory_sha,
            "IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE": "simulator_rgb",
            "DOCKER_LOG": str(self.docker_log),
            "FIXTURE_DATA_LINEAGE_SHA": self.data_lineage_sha,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *, dry_run: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.repository_root,
            env={**self.environment, "DRY_RUN": "1" if dry_run else "0"},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_production_case_and_threshold_files_are_exact(self) -> None:
        scenarios = json.loads(
            (
                self.repository_root
                / "configs/round_dagger/production_paired_scenarios_2x8.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(scenarios), 256)
        self.assertEqual(
            [(row["seed"], row["env_index"]) for row in scenarios],
            [
                (seed, env)
                for seed in (17, 29, 43, 59, 71, 89, 101, 127)
                for env in range(32)
            ],
        )
        self.assertEqual(
            json.loads(
                (
                    self.repository_root
                    / "configs/round_dagger/production_paired_thresholds.json"
                ).read_text(encoding="utf-8")
            ),
            {
                **DEFAULT_THRESHOLDS,
                "allow_bounded_nonregression": False,
                "minimum_paired_cases": 256,
            },
        )

    def test_direct_execution_without_trusted_dispatch_fails_closed(self) -> None:
        trusted = self.environment.pop(
            "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH"
        )
        try:
            completed = self._run(dry_run=True)
        finally:
            self.environment["ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH"] = trusted
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "missing ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH",
            completed.stderr,
        )
        self.assertFalse(self.contract.exists())

    def test_dry_run_validates_lineage_without_publishing_contract(self) -> None:
        completed = self._run(dry_run=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PRODUCTION_PAIRED_EVALUATION_DRY_RUN_OK", completed.stdout)
        self.assertIn("round_dagger.evaluation prepare", completed.stdout)
        self.assertIn("--ipc=private", completed.stdout)
        self.assertIn("--network=host", completed.stdout)
        self.assertIn(f"--workdir {self.gr00t}", completed.stdout)
        self.assertIn("device=0", completed.stdout)
        self.assertIn(str(self.result), completed.stdout)
        self.assertIn("--teacher-checkpoint", completed.stdout)
        self.assertIn("--teacher-sha256", completed.stdout)
        self.assertIn("--sealed-teacher-checkpoint", completed.stdout)
        self.assertIn("--require-teacher-readonly-mount", completed.stdout)
        self.assertIn("--motion-split-manifest", completed.stdout)
        self.assertIn("--heldout-plan", completed.stdout)
        runner_source = (
            self.repository_root
            / "scripts/round_dagger/run_paired_evaluation.sh"
        ).read_text(encoding="utf-8")
        for override in (
            "motion_split_role=eval",
            "motion_split_verify_content=true",
            'motion_split_offset="$MOTION_OFFSET"',
            'motion_split_limit="$MOTION_LIMIT"',
            "motion_assignment_offset=0",
        ):
            self.assertIn(override, runner_source)
        self.assertIn("IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE", completed.stdout)
        teacher_mount = (
            f"type=bind,src={self.sealed_teacher},"
            f"dst={self.teacher_checkpoint},readonly"
        )
        seal_mount = (
            f"type=bind,src={self.sealed_teacher},"
            f"dst={self.sealed_teacher},readonly"
        )
        unescaped_stdout = completed.stdout.replace("\\,", ",")
        self.assertIn(seal_mount, unescaped_stdout)
        self.assertIn(teacher_mount, unescaped_stdout)
        campaign_mount = (
            f"type=bind,src={self.root},dst={self.root},readonly"
        )
        evaluation_mount = (
            f"type=bind,src={self.contract.parent},dst={self.contract.parent}"
        )
        contract_mount = (
            f"type=bind,src={self.contract},dst={self.contract},readonly"
        )
        self.assertIn(campaign_mount, unescaped_stdout)
        self.assertIn(evaluation_mount, unescaped_stdout)
        self.assertIn(contract_mount, unescaped_stdout)
        for row in self.first_condition_bundles:
            self.assertIn(
                f"type=bind,src={row['path']},dst={row['path']},readonly",
                unescaped_stdout,
            )
        self.assertIn("--tmpfs", unescaped_stdout)
        self.assertIn("/run/round-dagger-eval:rw,nosuid,nodev,noexec,size=1g", unescaped_stdout)
        self.assertNotIn("type=bind,src=/mnt/data2,dst=/mnt/data2", unescaped_stdout)
        self.assertLess(
            unescaped_stdout.index(seal_mount),
            unescaped_stdout.index(teacher_mount),
        )
        self.assertFalse(self.contract.exists())
        self.assertFalse(self.result.exists())
        docker_calls = self.docker_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(docker_calls), 1)
        self.assertTrue(docker_calls[0].startswith("image inspect"))

    def test_normal_hook_prepares_contract_and_requires_real_result(self) -> None:
        completed = self._run(dry_run=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("paired evaluation evidence validation failed", completed.stderr)
        self.assertTrue(self.contract.is_file())
        self.assertFalse(self.result.exists())
        docker_calls = self.docker_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(docker_calls), 2)
        invocation = docker_calls[1]
        self.assertTrue(invocation.startswith("run --rm"))
        self.assertIn("--ipc=private", invocation)
        self.assertIn("--shm-size=32g", invocation)
        self.assertIn("--network=host", invocation)
        self.assertIn(f"--workdir {self.gr00t}", invocation)
        self.assertIn("run_paired_evaluation.sh", invocation)
        self.assertIn(str(self.result), invocation)
        teacher_mount = (
            f"type=bind,src={self.sealed_teacher},"
            f"dst={self.teacher_checkpoint},readonly"
        )
        seal_mount = (
            f"type=bind,src={self.sealed_teacher},"
            f"dst={self.sealed_teacher},readonly"
        )
        self.assertIn(seal_mount, invocation)
        self.assertIn(teacher_mount, invocation)
        self.assertIn(
            f"type=bind,src={self.root},dst={self.root},readonly",
            invocation,
        )
        self.assertIn(
            f"type=bind,src={self.contract.parent},dst={self.contract.parent}",
            invocation,
        )
        self.assertIn(
            f"type=bind,src={self.contract},dst={self.contract},readonly",
            invocation,
        )
        self.assertNotIn("type=bind,src=/mnt/data2,dst=/mnt/data2", invocation)
        self.assertLess(
            invocation.index(seal_mount),
            invocation.index(teacher_mount),
        )
        self.assertIn(f"--teacher-checkpoint {self.teacher_checkpoint}", invocation)
        self.assertIn(f"--teacher-sha256 {self.teacher_sha}", invocation)
        self.assertIn(
            f"--sealed-teacher-checkpoint {self.sealed_teacher}", invocation
        )
        self.assertIn("--require-teacher-readonly-mount", invocation)

    def test_mutated_first_condition_bundle_fails_before_contract_publication(self) -> None:
        bundle = Path(self.first_condition_bundles[0]["path"])
        bundle.chmod(0o644)
        bundle.write_bytes(b"mutated first-condition fixture\n")
        bundle.chmod(0o444)
        completed = self._run(dry_run=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "first-condition bundle inventory verification failed",
            completed.stderr,
        )
        self.assertFalse(self.contract.exists())

    def test_candidate_manifest_and_image_id_mismatches_fail_closed(self) -> None:
        for key, value, message in (
            (
                "ROUND_DAGGER_CANDIDATE_SHA256",
                _sha("wrong-candidate"),
                "candidate manifest SHA mismatch",
            ),
            (
                "ROUND_DAGGER_RTX_IMAGE_ID_SHA256",
                _sha("wrong-image"),
                "actor Docker image ID mismatch",
            ),
        ):
            with self.subTest(key=key):
                original = self.environment[key]
                self.environment[key] = value
                try:
                    completed = self._run(dry_run=True)
                finally:
                    self.environment[key] = original
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(self.contract.exists())

    def test_mutated_scenario_and_threshold_inputs_fail_closed(self) -> None:
        for variable, path, message in (
            (
                "ROUND_DAGGER_EVALUATION_SCENARIOS",
                self.tmp / "mutated-scenarios.json",
                "evaluation scenario cases SHA mismatch",
            ),
            (
                "ROUND_DAGGER_EVALUATION_THRESHOLDS",
                self.tmp / "mutated-thresholds.json",
                "evaluation thresholds SHA mismatch",
            ),
        ):
            with self.subTest(variable=variable):
                path.write_text("{}\n", encoding="utf-8")
                original = self.environment[variable]
                self.environment[variable] = str(path)
                try:
                    completed = self._run(dry_run=True)
                finally:
                    self.environment[variable] = original
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(self.contract.exists())

    def test_mutated_teacher_inputs_fail_closed(self) -> None:
        for target, message in (
            (self.teacher_checkpoint, "configured teacher checkpoint SHA mismatch"),
            (self.sealed_teacher, "sealed teacher checkpoint SHA mismatch"),
        ):
            with self.subTest(target=target.name):
                target.chmod(0o644)
                original = target.read_bytes()
                try:
                    target.write_bytes(b"mutated teacher fixture\n")
                    if target == self.sealed_teacher:
                        target.chmod(0o444)
                    completed = self._run(dry_run=True)
                finally:
                    target.chmod(0o644)
                    target.write_bytes(original)
                    target.chmod(
                        0o444 if target == self.sealed_teacher else 0o600
                    )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(self.contract.exists())

    def test_sealed_teacher_requires_exact_read_only_mode(self) -> None:
        self.sealed_teacher.chmod(0o644)
        completed = self._run(dry_run=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "sealed teacher checkpoint mode must be exactly 0444",
            completed.stderr,
        )
        self.assertFalse(self.contract.exists())

    def test_sealed_teacher_rejects_hard_link_aliases(self) -> None:
        os.link(self.sealed_teacher, self.tmp / "sealed-teacher-alias.pt")
        completed = self._run(dry_run=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "sealed teacher checkpoint must have exactly one hard link",
            completed.stderr,
        )
        self.assertFalse(self.contract.exists())

    def test_teacher_identity_must_match_campaign(self) -> None:
        for path in (self.teacher_checkpoint, self.sealed_teacher):
            path.chmod(0o644)
            path.write_bytes(b"different teacher")
        self.teacher_checkpoint.chmod(0o600)
        self.sealed_teacher.chmod(0o444)
        self.environment["ROUND_DAGGER_TEACHER_SHA256"] = sha256_file(
            self.teacher_checkpoint
        )
        completed = self._run(dry_run=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "teacher checkpoint SHA differs from campaign", completed.stderr
        )
        self.assertFalse(self.contract.exists())

    def test_pinned_config_must_name_mounted_teacher_target(self) -> None:
        self.config.write_text(
            f"teacher_actor_path: {self.tmp / 'other-teacher.pt'}\n"
            "algo:\n"
            "  config:\n"
            "    videogen:\n"
            f"      anchor_pool_dir: {self.anchor_pool}\n"
            "      anchor_index: 0\n"
            "fixed_seed: true\n",
            encoding="utf-8",
        )
        self.environment["ROUND_DAGGER_EVALUATOR_CONFIG_SHA256"] = sha256_file(
            self.config
        )
        completed = self._run(dry_run=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "pinned evaluation config teacher path mismatch", completed.stderr
        )
        self.assertFalse(self.contract.exists())

    def test_pinned_config_must_name_mounted_anchor_pool_and_index(self) -> None:
        for anchor, index, expected in (
            (self.tmp / "other-anchors", 0, "anchor pool path mismatch"),
            (self.anchor_pool, 1, "anchor index mismatch"),
        ):
            with self.subTest(expected=expected):
                self.config.write_text(
                    f"teacher_actor_path: {self.teacher_checkpoint}\n"
                    "algo:\n"
                    "  config:\n"
                    "    videogen:\n"
                    f"      anchor_pool_dir: {anchor}\n"
                    f"      anchor_index: {index}\n",
                    encoding="utf-8",
                )
                self.environment["ROUND_DAGGER_EVALUATOR_CONFIG_SHA256"] = (
                    sha256_file(self.config)
                )
                completed = self._run(dry_run=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected, completed.stderr)
                self.assertFalse(self.contract.exists())


if __name__ == "__main__":
    unittest.main()
