from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import wandb_delivery
from scripts.scaling import wandb_run_watcher as watcher


class WandbDeliveryTest(unittest.TestCase):
    @staticmethod
    def _receipt(run: Path, run_id: str) -> None:
        state = run / "telemetry" / ".wandb_upload_state.json"
        (run / "telemetry" / "WANDB_DELIVERY_RECEIPT.json").write_text(
            json.dumps(
                {
                    "schema": watcher.DELIVERY_RECEIPT_SCHEMA,
                    "wandb_run_id": run_id,
                    "upload_state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                    "outcome": "SUCCEEDED",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_requires_durable_coverage_and_replays_late_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary).resolve() / "run"
            attempt = run / "actor_pair_attempts" / "attempt_1"
            scalars = run / "telemetry" / "scalars"
            attempt.mkdir(parents=True)
            scalars.mkdir(parents=True)
            run_id = "stable_run"
            (run / "telemetry" / "WANDB_SIDECAR_RUN_ID").write_text(
                run_id + "\n", encoding="utf-8"
            )
            terminal = scalars / "step_000000_actor_pair_terminal_attempt_1.json"
            terminal.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.actor_pair_terminal_scalar.v1",
                        "step": 0,
                        "metrics": {"collection/terminal_pending": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            state_path = run / "telemetry" / ".wandb_upload_state.json"
            watcher.save_state(state_path, {}, run_id=run_id)
            self.assertIsNone(wandb_delivery.verify_delivery(run))

            relative = str(terminal.relative_to(run))
            watcher.save_state(
                state_path, {relative: watcher._fingerprint(terminal)}, run_id=run_id
            )
            self._receipt(run, run_id)
            self.assertEqual(
                wandb_delivery.verify_delivery(run)["terminal_scalar_count"], 1
            )

            failure = wandb_delivery.publish_delivery_failure(
                attempt_dir=attempt,
                run_dir=run,
                macro_round=0,
                cohort_index=0,
                wrapper_exit_code=7,
            )
            self.assertIsNone(wandb_delivery.verify_delivery(run))
            current = watcher.load_state(state_path, expected_run_id=run_id)
            current[str(failure.relative_to(run))] = watcher._fingerprint(failure)
            watcher.save_state(state_path, current, run_id=run_id)
            self._receipt(run, run_id)
            report = wandb_delivery.verify_delivery(run)
            self.assertEqual(report["recovered_delivery_failure_count"], 1)

    def test_cross_identity_permission_error_is_reported_precisely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_id = Path(temporary) / "WANDB_SIDECAR_RUN_ID"
            run_id.write_text("stable_run\n", encoding="utf-8")
            with mock.patch.object(
                Path, "read_text", side_effect=PermissionError("denied")
            ):
                with self.assertRaisesRegex(
                    wandb_delivery.WandbDeliveryError,
                    "not readable across runtime identities",
                ):
                    wandb_delivery._safe_run_id(run_id)


if __name__ == "__main__":
    unittest.main()
