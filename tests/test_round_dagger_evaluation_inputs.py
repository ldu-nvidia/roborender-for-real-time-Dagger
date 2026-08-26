"""Regression tests for paired-evaluation staged-input TOCTOU gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised in the production venv
    torch = None

from round_dagger.evaluation_inputs import (
    EvaluationInputError,
    verify_staged_inputs,
)
from scripts.eval.fixed_seed_gate import (
    checkpoint_semantic_sha256,
    sanitize_checkpoint,
    sha256_file,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class EvaluationInputVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch is required for checkpoint semantic tests")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"teacher")
        self.source = self.root / "source.pt"
        torch.save(
            {
                "checkpoint_schema": "fixture",
                "policy_state_dict": {
                    "linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                    "linear.bias": torch.tensor([0.25, -0.5, 1.0]),
                },
                "value_state_dict": {"value.weight": torch.ones(1, 3)},
                "env_state_dict": {"forbidden": "simulator state"},
                "optimizer_state_dict": {"step": 9},
            },
            self.source,
        )
        self.source_sha = sha256_file(self.source)
        self.staged = self.root / "stage/checkpoint.pt"
        self.staged.parent.mkdir()
        provenance = sanitize_checkpoint(
            self.source,
            self.staged,
            self.source_sha,
        )
        self.config = self.root / "evaluation.yaml"
        self.staged_config = self.staged.parent / "config.yaml"
        self.config_bytes = (
            f"teacher_actor_path: {self.teacher}\nfixed_seed: true\n".encode()
        )
        self.config.write_bytes(self.config_bytes)
        self.staged_config.write_bytes(self.config_bytes)
        self.staged_config.chmod(0o444)
        self.provenance = self.staged.parent / "provenance.json"
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")
        self.provenance.chmod(0o444)

    def tearDown(self) -> None:
        for path in (self.staged, self.staged_config, self.provenance):
            path.chmod(0o644)
        self.temporary.cleanup()

    def _verify(self) -> dict:
        return verify_staged_inputs(
            checkpoint_source=self.source,
            expected_checkpoint_sha256=self.source_sha,
            staged_checkpoint=self.staged,
            provenance=self.provenance,
            evaluation_config=self.config,
            staged_config=self.staged_config,
            expected_config_sha256=_sha(self.config_bytes),
            teacher_checkpoint=self.teacher,
        )

    def test_exact_first_stage_passes(self) -> None:
        self.assertEqual(self._verify()["status"], "PASS")

    def test_exact_resume_reuse_passes(self) -> None:
        first = self._verify()
        second = self._verify()
        self.assertEqual(
            first["staged_checkpoint_semantic_sha256"],
            second["staged_checkpoint_semantic_sha256"],
        )

    def test_coherent_stage_and_provenance_substitution_fails_closed(self) -> None:
        malicious = {
            "policy_state_dict": {
                "linear.weight": torch.full((3, 4), 99.0),
                "linear.bias": torch.full((3,), -99.0),
            },
            "value_state_dict": {"value.weight": torch.zeros(1, 3)},
        }
        self.staged.chmod(0o644)
        torch.save(malicious, self.staged)
        self.staged.chmod(0o444)
        row = json.loads(self.provenance.read_text(encoding="utf-8"))
        row["output_sha256"] = sha256_file(self.staged)
        row["output_semantic_sha256"] = checkpoint_semantic_sha256(malicious)
        self.provenance.chmod(0o644)
        self.provenance.write_text(json.dumps(row), encoding="utf-8")
        self.provenance.chmod(0o444)
        with self.assertRaisesRegex(
            EvaluationInputError,
            "semantics differ from the contracted source",
        ):
            self._verify()

    def test_staged_checkpoint_requires_exact_read_only_mode(self) -> None:
        self.staged.chmod(0o644)
        with self.assertRaisesRegex(EvaluationInputError, "mode must be exactly 0444"):
            self._verify()

    def test_provenance_requires_exact_read_only_mode(self) -> None:
        self.provenance.chmod(0o644)
        with self.assertRaisesRegex(EvaluationInputError, "mode must be exactly 0444"):
            self._verify()

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
                expected_checkpoint_sha256=self.source_sha,
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
        latch = script.index(
            'STAGED_SHA=$(printf \'%s\' "$VERIFIED_STAGE"', pre
        )
        launch = script.index(
            'EXPECTED_TRAIN_CHECKPOINT_SHA256="$STAGED_SHA" "${EVAL[@]}"',
            latch,
        )
        post = script.index("staged evaluation inputs changed during evaluator run")
        attest = script.index('        "${ATTEST[@]}"', post)
        self.assertLess(first, pre)
        self.assertLess(pre, latch)
        self.assertLess(latch, launch)
        self.assertLess(pre, launch)
        self.assertLess(launch, post)
        self.assertLess(post, attest)


if __name__ == "__main__":
    unittest.main()
