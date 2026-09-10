from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.dagger.monitor_live_round import phase_snapshot
from pipeline.dagger.services.phase_telemetry import PhaseTelemetry


class PhaseTelemetryTests(unittest.TestCase):
    def test_success_and_error_spans_are_ordered_and_latest_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reporter = PhaseTelemetry(root / "timing.jsonl", latest_path=root / "progress.json")
            with reporter.span(
                run_id="run", round_id=1, vector_step_index=3,
                phase="capture", total=28,
            ) as details:
                details["environments"] = 28
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with reporter.span(run_id="run", round_id=1, phase="render"):
                    raise RuntimeError("boom")
            reporter.close()

            records = [json.loads(line) for line in (root / "timing.jsonl").read_text().splitlines()]
            self.assertEqual(
                [(item["phase"], item["event"]) for item in records],
                [("capture", "start"), ("capture", "end"),
                 ("render", "start"), ("render", "error")],
            )
            self.assertEqual(records[1]["completed"], 28)
            self.assertEqual(records[1]["details"]["environments"], 28)
            self.assertGreaterEqual(records[1]["elapsed_ms"], 0.0)
            latest = json.loads((root / "progress.json").read_text())
            self.assertEqual(latest["phase"], "render")
            self.assertEqual(latest["event"], "error")

    def test_v3_monitor_aggregates_batched_gateway_lane_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gpu0-phases.jsonl"
            reporter = PhaseTelemetry(path)
            details = {"batch_id": "batch-0", "lane_count": 2, "checkpoint_sha256": "a" * 64}
            for phase in (
                "h200_image3f_render_and_student_inference",
                "h200_beta_action_selection",
                "h200_learner_buffer_append",
                "h200_window_seal_and_publish",
            ):
                with reporter.span(
                    run_id="run", round_id=1, vector_step_index=80,
                    actor_id="lane-0", phase=phase, details=details,
                ):
                    pass
            reporter.close()

            report = phase_snapshot(root, "run", 80, lanes=2, checkpoint_sha256="a" * 64)
            self.assertEqual(report["current_phase"], "h200_step_complete")
            self.assertEqual(
                report["phases"]["h200_learner_buffer_append"]["completed"], 2
            )
            self.assertEqual(
                report["phases"]["h200_window_seal_and_publish"]["completed"], 2
            )


if __name__ == "__main__":
    unittest.main()
