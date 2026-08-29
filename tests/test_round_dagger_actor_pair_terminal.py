"""Immutable actor-pair terminal telemetry tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from round_dagger import artifacts
from round_dagger.actor_pair_terminal import (
    ActorPairTerminalError,
    _atomic_finish_marker,
    _immutable_bytes,
    publish_actor_pair_terminal,
)
from tests.test_round_dagger_actor_pair import DIGESTS, _arrays, _cohort


class ActorPairTerminalTest(unittest.TestCase):
    def test_pending_delivery_is_not_reported_as_false_success_and_precedes_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / "run"
            attempt = run / "actor_pair_attempts" / "attempt_pending"
            attempt.mkdir(parents=True)
            cohort = root / "cohort"
            _cohort(cohort)
            for rank in (0, 1):
                artifacts.publish_rollout_shard(
                    cohort,
                    rank=rank,
                    rollout_id=0,
                    part_index=0,
                    arrays=_arrays(rank),
                    campaign_id="actor-pair-test",
                    macro_round=0,
                    cohort_index=0,
                    purpose="train",
                    **DIGESTS,
                )
            artifacts.seal_cohort(cohort)
            finish = attempt / "finish"
            result = publish_actor_pair_terminal(
                attempt_dir=attempt,
                run_dir=run,
                cohort_dir=cohort,
                campaign_id="actor-pair-test",
                macro_round=0,
                cohort_index=0,
                rank0_exit_code=0,
                rank1_exit_code=0,
                lifecycle_exit_code=0,
                quality_exit_code=0,
                delivery_pending=True,
                finish_marker=finish,
            )
            self.assertEqual(result["outcome"], "PENDING_WANDB_DELIVERY")
            self.assertTrue(result["delivery_pending"])
            metrics = json.loads(
                Path(result["scalar_path"]).read_text(encoding="utf-8")
            )["metrics"]
            self.assertEqual(metrics["collection/succeeded"], 0)
            self.assertEqual(metrics["collection/terminal_pending"], 1)
            self.assertEqual(finish.read_text(encoding="ascii"), "0\n")

    def test_sigkill_truncated_staging_is_cleaned_without_exposing_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "terminal.json"
            staging = root / ".terminal.json.staging"
            killed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,signal,sys; "
                        "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); "
                        "os.write(fd,b'truncated'); os.fsync(fd); "
                        "os.kill(os.getpid(),signal.SIGKILL)"
                    ),
                    str(staging),
                ],
                check=False,
            )
            self.assertEqual(killed.returncode, -9)
            self.assertFalse(target.exists())
            _immutable_bytes(target, b"complete\n")
            self.assertEqual(target.read_bytes(), b"complete\n")
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            self.assertFalse(staging.exists())

    def test_truncated_final_is_conflict_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "terminal.json"
            target.write_bytes(b'{"partial":')
            target.chmod(0o444)
            before = target.read_bytes()
            with self.assertRaisesRegex(ActorPairTerminalError, "conflicts"):
                _immutable_bytes(target, b'{"complete":true}\n')
            self.assertEqual(target.read_bytes(), before)

    def test_finish_marker_publication_is_create_only_under_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary).resolve() / "finish"
            barrier = threading.Barrier(2)
            outcomes: list[str] = []

            def publish(exit_code: int) -> None:
                barrier.wait()
                try:
                    _atomic_finish_marker(marker, exit_code)
                    outcomes.append("published")
                except ActorPairTerminalError:
                    outcomes.append("conflict")

            threads = [
                threading.Thread(target=publish, args=(exit_code,))
                for exit_code in (0, 2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertEqual(sorted(outcomes), ["conflict", "published"])
            self.assertIn(marker.read_text(encoding="ascii"), {"0\n", "2\n"})
            self.assertEqual(marker.stat().st_mode & 0o222, 0)

    def test_unsafe_staging_symlink_is_not_cleaned_or_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "terminal.json"
            victim = root / "victim"
            victim.write_bytes(b"preserve")
            staging = root / ".terminal.json.staging"
            staging.symlink_to(victim)
            with self.assertRaisesRegex(ActorPairTerminalError, "staging artifact"):
                _immutable_bytes(target, b"complete\n")
            self.assertTrue(staging.is_symlink())
            self.assertEqual(victim.read_bytes(), b"preserve")
            self.assertFalse(target.exists())

    def test_failed_pair_is_immutable_visible_and_finishes_wandb_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / "run"
            attempt = run / "actor_pair_attempts" / "attempt_failure"
            attempt.mkdir(parents=True)
            cohort = root / "cohort"
            _cohort(cohort)
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=_arrays(0),
                campaign_id="actor-pair-test",
                macro_round=0,
                cohort_index=0,
                purpose="train",
                **DIGESTS,
            )
            finish = attempt / "wandb_pair_finish"

            result = publish_actor_pair_terminal(
                attempt_dir=attempt,
                run_dir=run,
                cohort_dir=cohort,
                campaign_id="actor-pair-test",
                macro_round=0,
                cohort_index=0,
                rank0_exit_code=0,
                rank1_exit_code=137,
                lifecycle_exit_code=0,
                quality_exit_code=0,
                finish_marker=finish,
            )
            terminal = attempt / "actor_pair_terminal.json"
            payload = json.loads(terminal.read_text(encoding="utf-8"))
            self.assertEqual(result["outcome"], "FAILED")
            self.assertEqual(payload["failure_code"], "actor_rank_nonzero")
            self.assertEqual(
                payload["cohort_contract_sha256"],
                artifacts.sha256_file(cohort / "contract.json"),
            )
            self.assertEqual(payload["rank_exit_codes"], {"0": 0, "1": 137})
            self.assertEqual(payload["accepted_transitions"], 0)
            self.assertEqual(payload["committed_rank_transitions"], 32 * 512)
            self.assertEqual(terminal.stat().st_mode & 0o222, 0)
            self.assertEqual(finish.read_text(encoding="ascii"), "2\n")

            scalar = Path(result["scalar_path"])
            metrics = json.loads(scalar.read_text(encoding="utf-8"))["metrics"]
            self.assertEqual(metrics["collection/failed"], 1)
            self.assertEqual(metrics["collection/data_ready"], 0)
            self.assertEqual(metrics["collection/rank1_exit_code"], 137)
            self.assertEqual(metrics["collection/accepted_transitions"], 0)
            self.assertEqual(
                metrics["collection/committed_rank_transitions"], 32 * 512
            )
            self.assertEqual(metrics["collection/committed_rank_fraction"], 0.5)
            self.assertEqual(scalar.stat().st_mode & 0o222, 0)

            # Exact replay is byte-idempotent; contradictory terminal status is rejected.
            repeated = publish_actor_pair_terminal(
                attempt_dir=attempt,
                run_dir=run,
                cohort_dir=cohort,
                campaign_id="actor-pair-test",
                macro_round=0,
                cohort_index=0,
                rank0_exit_code=0,
                rank1_exit_code=137,
                lifecycle_exit_code=0,
                quality_exit_code=0,
                finish_marker=finish,
            )
            self.assertEqual(repeated["outcome"], "FAILED")
            with self.assertRaisesRegex(ActorPairTerminalError, "conflicts"):
                publish_actor_pair_terminal(
                    attempt_dir=attempt,
                    run_dir=run,
                    cohort_dir=cohort,
                    campaign_id="actor-pair-test",
                    macro_round=0,
                    cohort_index=0,
                    rank0_exit_code=0,
                    rank1_exit_code=0,
                    lifecycle_exit_code=0,
                    quality_exit_code=0,
                    finish_marker=finish,
                )

    def test_terminal_identity_is_bound_to_immutable_cohort_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / "run"
            attempt = run / "actor_pair_attempts" / "attempt_identity"
            attempt.mkdir(parents=True)
            cohort = root / "cohort"
            _cohort(cohort)
            with self.assertRaisesRegex(
                ActorPairTerminalError, "does not match actor cohort contract"
            ):
                publish_actor_pair_terminal(
                    attempt_dir=attempt,
                    run_dir=run,
                    cohort_dir=cohort,
                    campaign_id="another-campaign",
                    macro_round=0,
                    cohort_index=0,
                    rank0_exit_code=2,
                    rank1_exit_code=2,
                    lifecycle_exit_code=0,
                    quality_exit_code=0,
                )
            self.assertFalse((attempt / "actor_pair_terminal.json").exists())

    def test_symlink_in_run_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            run = real / "run"
            attempt = run / "actor_pair_attempts" / "attempt_symlink"
            attempt.mkdir(parents=True)
            cohort = root / "cohort"
            _cohort(cohort)
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ActorPairTerminalError, "symbolic link"):
                publish_actor_pair_terminal(
                    attempt_dir=linked / "run/actor_pair_attempts/attempt_symlink",
                    run_dir=linked / "run",
                    cohort_dir=cohort,
                    campaign_id="actor-pair-test",
                    macro_round=0,
                    cohort_index=0,
                    rank0_exit_code=2,
                    rank1_exit_code=2,
                    lifecycle_exit_code=0,
                    quality_exit_code=0,
                )


if __name__ == "__main__":
    unittest.main()
