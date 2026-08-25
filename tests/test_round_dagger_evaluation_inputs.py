"""Regression tests for paired-evaluation staged-input TOCTOU gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from round_dagger.evaluation_inputs import (
    EvaluationInputError,
    verify_staged_inputs,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class EvaluationInputVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"teacher")
        self.source = self.root / "source.pt"
        self.source.write_bytes(b"source checkpoint")
        self.staged = self.root / "stage/checkpoint.pt"
        self.staged.parent.mkdir()
        self.staged.write_bytes(b"sanitized checkpoint")
        self.config = self.root / "evaluation.yaml"
        self.staged_config = self.staged.parent / "config.yaml"
        self.config_bytes = (
            f"teacher_actor_path: {self.teacher}\nfixed_seed: true\n".encode()
        )
        self.config.write_bytes(self.config_bytes)
        self.staged_config.write_bytes(self.config_bytes)
        self.staged_config.chmod(0o444)
        self.provenance = self.staged.parent / "provenance.json"
        self.provenance.write_text(
            json.dumps(
                {
                    "source": str(self.source.resolve()),
                    "source_sha256": _sha(self.source.read_bytes()),
                    "output": str(self.staged),
                    "output_sha256": _sha(self.staged.read_bytes()),
                    "removed_keys": ["env_state_dict"],
                    "environment_state_restored": False,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.staged_config.chmod(0o644)
        self.temporary.cleanup()

    def _verify(self) -> dict:
        return verify_staged_inputs(
            checkpoint_source=self.source,
            expected_checkpoint_sha256=_sha(b"source checkpoint"),
            staged_checkpoint=self.staged,
            provenance=self.provenance,
            evaluation_config=self.config,
            staged_config=self.staged_config,
            expected_config_sha256=_sha(self.config_bytes),
            teacher_checkpoint=self.teacher,
        )

    def test_exact_first_stage_passes(self) -> None:
        self.assertEqual(self._verify()["status"], "PASS")

    def test_copy_time_config_substitution_fails_closed(self) -> None:
        self.staged_config.chmod(0o644)
        self.staged_config.write_text(
            f"teacher_actor_path: {self.root / 'other.pt'}\nfixed_seed: true\n",
            encoding="utf-8",
        )
        self.staged_config.chmod(0o444)
        with self.assertRaisesRegex(
            EvaluationInputError, "staged evaluation config SHA-256 mismatch"
        ):
            self._verify()

    def test_digest_valid_config_cannot_redirect_teacher_mount(self) -> None:
        wrong_teacher = self.root / "other.pt"
        wrong_config = (
            f"teacher_actor_path: {wrong_teacher}\nfixed_seed: true\n".encode()
        )
        self.config.write_bytes(wrong_config)
        self.staged_config.chmod(0o644)
        self.staged_config.write_bytes(wrong_config)
        self.staged_config.chmod(0o444)
        with self.assertRaisesRegex(
            EvaluationInputError,
            "teacher path differs from the pinned mount target",
        ):
            verify_staged_inputs(
                checkpoint_source=self.source,
                expected_checkpoint_sha256=_sha(b"source checkpoint"),
                staged_checkpoint=self.staged,
                provenance=self.provenance,
                evaluation_config=self.config,
                staged_config=self.staged_config,
                expected_config_sha256=_sha(wrong_config),
                teacher_checkpoint=self.teacher,
            )

    def test_source_config_change_before_or_during_evaluator_fails_closed(self) -> None:
        self.assertEqual(self._verify()["status"], "PASS")
        self.config.write_text("teacher_actor_path: /tmp/other.pt\n", encoding="utf-8")
        with self.assertRaisesRegex(
            EvaluationInputError, "evaluation config source SHA-256 mismatch"
        ):
            self._verify()

    def test_stage_change_during_evaluator_fails_post_run_gate(self) -> None:
        self.assertEqual(self._verify()["status"], "PASS")
        self.staged_config.chmod(0o644)
        self.staged_config.write_bytes(self.config_bytes + b"changed: true\n")
        self.staged_config.chmod(0o444)
        with self.assertRaises(EvaluationInputError):
            self._verify()

    def test_staged_config_requires_exact_read_only_mode(self) -> None:
        self.staged_config.chmod(0o644)
        with self.assertRaisesRegex(EvaluationInputError, "mode must be exactly 0444"):
            self._verify()

    def test_launcher_orders_first_pre_and_post_verification(self) -> None:
        script = (
            Path(__file__).parents[1]
            / "scripts/round_dagger/run_paired_evaluation.sh"
        ).read_text(encoding="utf-8")
        first = script.index("new staged evaluation inputs failed verification")
        pre = script.index("staged evaluation inputs changed before evaluator launch")
        launch = script.index('        "${EVAL[@]}"', pre)
        post = script.index("staged evaluation inputs changed during evaluator run")
        attest = script.index('        "${ATTEST[@]}"', post)
        self.assertLess(first, pre)
        self.assertLess(pre, launch)
        self.assertLess(launch, post)
        self.assertLess(post, attest)


if __name__ == "__main__":
    unittest.main()
