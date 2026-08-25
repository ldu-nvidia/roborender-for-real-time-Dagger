"""Fail-closed tests for the candidate-aware production evaluation hook."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import sha256_file
from round_dagger.evaluation import DEFAULT_THRESHOLDS
from round_dagger.state import load_campaign, load_state

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
        self.repository_root = Path(__file__).parents[1]
        self.root = _campaign(self.tmp)
        _to_ingested(self.root)
        self.candidate = _stage_and_bind(self.root, self.tmp)
        self.state = load_state(self.root)
        self.campaign = load_campaign(self.root)

        self.teacher_checkpoint = self.tmp / "configured-teacher.pt"
        self.sealed_teacher = self.tmp / "sealed-teacher.pt"
        # _campaign pins sha256(b"teacher") as the expert identity.
        teacher_bytes = b"teacher"
        self.teacher_checkpoint.write_bytes(teacher_bytes)
        self.sealed_teacher.write_bytes(teacher_bytes)
        self.teacher_checkpoint.chmod(0o600)
        self.sealed_teacher.chmod(0o444)
        self.teacher_sha = sha256_file(self.teacher_checkpoint)
        self.config = self.tmp / "evaluation_config.yaml"
        self.config.write_text(
            f"teacher_actor_path: {self.teacher_checkpoint}\nfixed_seed: true\n",
            encoding="utf-8",
        )
        self.gr00t = self.tmp / "GR00T-VisualSim2Real"
        for relative in (
            "gr00t/rl/eval_agent_trl.py",
            "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
            "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
            "gr00t/rl/utils/image3f_frame_client.py",
            "gr00t/rl/utils/round_dagger_paired_eval.py",
        ):
            source = self.gr00t / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# fixture: {relative}\n", encoding="utf-8")
        self.endpoints = self.tmp / "endpoints"
        self.endpoints.mkdir()
        for rank in range(2):
            (self.endpoints / f"renderer_{rank}.txt").write_text(
                f"127.0.0.1 {19000 + rank}\n", encoding="utf-8"
            )

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
            "ROUND_DAGGER_EVALUATION_ENDPOINT_DIR": str(self.endpoints),
            "ROUND_DAGGER_RENDERER_RUNTIME_SHA256": _sha("renderer-runtime"),
            "ROUND_DAGGER_EXPECTED_RENDERERS": "2",
            "GR00T_ROOT": str(self.gr00t),
            "GR00T_PYTHON": "/opt/round-dagger-rtx/bin/python",
            "ROUND_DAGGER_DOCKER_BIN": str(self.docker),
            "ROUND_DAGGER_RTX_IMAGE": "round-dagger:test",
            "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": self.image_id_sha,
            # Preserve the virtual-environment entry point.  Resolving this
            # symlink would execute the base interpreter without its packages.
            "ROUND_DAGGER_CONTROL_PYTHON": sys.executable,
            "DOCKER_LOG": str(self.docker_log),
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
        self.assertEqual(len(scenarios), 16)
        self.assertEqual(
            [(row["seed"], row["env_index"]) for row in scenarios],
            [(seed, env) for seed in (17, 29) for env in range(8)],
        )
        self.assertEqual(
            json.loads(
                (
                    self.repository_root
                    / "configs/round_dagger/production_paired_thresholds.json"
                ).read_text(encoding="utf-8")
            ),
            DEFAULT_THRESHOLDS,
        )

    def test_dry_run_validates_lineage_without_publishing_contract(self) -> None:
        completed = self._run(dry_run=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PRODUCTION_PAIRED_EVALUATION_DRY_RUN_OK", completed.stdout)
        self.assertIn("round_dagger.evaluation prepare", completed.stdout)
        self.assertIn("--ipc=private", completed.stdout)
        self.assertIn("--network=host", completed.stdout)
        self.assertIn("device=0", completed.stdout)
        self.assertIn(str(self.result), completed.stdout)
        self.assertIn("--teacher-checkpoint", completed.stdout)
        self.assertIn("--teacher-sha256", completed.stdout)
        self.assertIn("--sealed-teacher-checkpoint", completed.stdout)
        self.assertIn("--require-teacher-readonly-mount", completed.stdout)
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
        self.assertLess(
            unescaped_stdout.index("type=bind,src=/mnt/data2,dst=/mnt/data2"),
            unescaped_stdout.index(seal_mount),
        )
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
        self.assertLess(
            invocation.index("type=bind,src=/mnt/data2,dst=/mnt/data2"),
            invocation.index(seal_mount),
        )
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


if __name__ == "__main__":
    unittest.main()
