from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.round_dagger import validate_learner_wandb_readback as readback


class FakeRun:
    id = "run-1"
    state = "finished"
    config = {"learner_identity_sha256": "a" * 64}

    def __init__(self, *, conflicting: bool = False) -> None:
        self.conflicting = conflicting

    def scan_history(self, *, keys, page_size):
        self.assert_page_size(page_size)
        if "loss" in keys:
            return [
                {"optimizer_step": 1, "loss": 3.0},
                {"optimizer_step": 2, "loss": 2.0},
            ]
        rows = [
            self.quality(0, 5.0),
            self.quality(1, 4.0),
            self.quality(1, 4.5 if self.conflicting else 4.0),
            self.quality(2, 3.0),
        ]
        return rows

    @staticmethod
    def assert_page_size(value):
        if value != 1000:
            raise AssertionError("validator changed page size")

    @staticmethod
    def quality(step: int, mse: float):
        return {
            "optimizer_step": step,
            "quality/heldout_action_mse": mse,
            "quality/heldout_action_mae": mse + 0.1,
            "quality/heldout_action_rmse": mse + 0.2,
            "quality/heldout_action_cosine": 0.5 - step / 10,
            "quality/heldout_transitions": 256,
        }


class FakeApi:
    def __init__(self, run: FakeRun) -> None:
        self.remote = run

    def run(self, path: str):
        if path != "entity/project/run-1":
            raise AssertionError(path)
        return self.remote


class LearnerWandbReadbackTest(unittest.TestCase):
    def fixtures(self, root: Path) -> tuple[Path, Path]:
        train = root / "train.jsonl"
        heldout = root / "heldout.jsonl"
        train.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": readback.TELEMETRY_SCHEMA,
                        "optimizer_step": step,
                        "loss": loss,
                    },
                    sort_keys=True,
                )
                + "\n"
                for step, loss in ((1, 3.0), (2, 2.0))
            )
        )
        heldout.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": readback.HELDOUT_TELEMETRY_SCHEMA,
                        "optimizer_step": step,
                        "action_mse": mse,
                        "action_mae": mse + 0.1,
                        "action_rmse": mse + 0.2,
                        "action_cosine": 0.5 - step / 10,
                        "heldout_transitions_global": 256,
                    },
                    sort_keys=True,
                )
                + "\n"
                for step, mse in ((0, 5.0), (1, 4.0), (2, 3.0))
            )
        )
        return train, heldout

    def test_sparse_quality_and_loss_are_read_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, heldout = self.fixtures(root)
            output = root / "report.json"
            report = readback.validate(
                run_path="entity/project/run-1",
                train_telemetry=train,
                heldout_telemetry=heldout,
                expected_config={"learner_identity_sha256": "a" * 64},
                output=output,
                api=FakeApi(FakeRun()),
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["training_optimizer_steps"], [1, 2])
            self.assertEqual(report["quality_optimizer_steps"], [0, 1, 2])
            self.assertTrue(report["duplicate_quality_rows_are_identical"])
            self.assertTrue(output.is_file())

    def test_conflicting_duplicate_quality_frontier_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, heldout = self.fixtures(root)
            with self.assertRaisesRegex(readback.ReadbackError, "conflicting"):
                readback.validate(
                    run_path="entity/project/run-1",
                    train_telemetry=train,
                    heldout_telemetry=heldout,
                    expected_config={"learner_identity_sha256": "a" * 64},
                    output=root / "report.json",
                    api=FakeApi(FakeRun(conflicting=True)),
                )

    def test_local_duplicate_frontier_fails_before_remote_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, heldout = self.fixtures(root)
            train.write_text(train.read_text() + train.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(readback.ReadbackError, "duplicated or unordered"):
                readback.validate(
                    run_path="entity/project/run-1",
                    train_telemetry=train,
                    heldout_telemetry=heldout,
                    expected_config={"learner_identity_sha256": "a" * 64},
                    output=root / "report.json",
                    api=FakeApi(FakeRun()),
                )


if __name__ == "__main__":
    unittest.main()
