import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger.reference_audit import (
    ReferenceAuditError,
    audit_reference_telemetry,
    audit_uninterrupted_rollout,
)


def write_scalar(root: Path, *, step: int, lag: int) -> None:
    generated = step >= lag
    bootstrap = step == 0
    frame_zero = 0 < step < lag
    metrics = {
        "renderer/reference_provenance_valid": 1,
        "renderer/reference_provenance/request_step": step,
        "renderer/reference_provenance/configured_lag": lag,
        "renderer/reference_provenance/source_step": step - lag if generated else (-1 if bootstrap else 0),
        "renderer/reference_source_anchor": int(bootstrap),
        "renderer/reference_source_bootstrap_anchor": int(bootstrap),
        "renderer/reference_source_generated_frame0": int(frame_zero),
        "renderer/reference_source_generated_lag": int(generated),
        "renderer/reference_source_client": 0,
        "renderer/reference_source_idempotent_replay": 0,
        "renderer/reference_source_unknown": 0,
    }
    if generated:
        metrics["renderer/reference_lag_frames"] = lag
    (root / f"step_{step:09d}.json").write_text(
        json.dumps({"step": step, "metrics": metrics}), encoding="utf-8"
    )


class ReferenceAuditTest(unittest.TestCase):
    def test_proves_anchor_to_generated_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in range(5):
                write_scalar(root, step=step, lag=3)
            report = audit_reference_telemetry(root, lag=3, expected_steps=5)
            self.assertEqual(report["bootstrap_anchor_step_count"], 1)
            self.assertEqual(report["generated_frame_zero_step_count"], 2)
            self.assertEqual(report["generated_step_count"], 2)
            self.assertEqual(report["first_generated_request_step"], 3)
            self.assertEqual(report["last_generated_source_step"], 1)
            self.assertRegex(report["scalar_chain_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_missing_or_false_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in range(5):
                write_scalar(root, step=step, lag=3)
            (root / "step_000000003.json").unlink()
            with self.assertRaisesRegex(ReferenceAuditError, "missing scalar"):
                audit_reference_telemetry(root, lag=3, expected_steps=5)

            write_scalar(root, step=3, lag=3)
            path = root / "step_000000003.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["metrics"]["renderer/reference_source_anchor"] = 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ReferenceAuditError, "reference_source_anchor"):
                audit_reference_telemetry(root, lag=3, expected_steps=5)

    def test_binds_reference_proof_to_uninterrupted_rollout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in range(5):
                write_scalar(root, step=step, lag=3)
            shard = root / "rank-0000_rollout-00000000_part-0000.npz"
            np.savez(
                shard,
                dones=np.zeros((5, 1), dtype=np.bool_),
                episode_ids=np.full((5, 1), 7, dtype=np.int64),
                episode_steps=np.arange(5, dtype=np.int64)[:, None],
            )
            report = audit_reference_telemetry(
                root, lag=3, expected_steps=5, rollout_shard=shard
            )
            self.assertEqual(report["uninterrupted_rollout"]["episode_id"], 7)
            self.assertEqual(
                report["uninterrupted_rollout"]["last_episode_step"], 4
            )

    def test_rejects_rollout_episode_reset_even_if_scalar_counter_is_contiguous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "rank-0000_rollout-00000000_part-0000.npz"
            np.savez(
                shard,
                dones=np.array([[False], [True], [False], [False]], dtype=np.bool_),
                episode_ids=np.array([[0], [0], [1], [1]], dtype=np.int64),
                episode_steps=np.array([[0], [1], [0], [1]], dtype=np.int64),
            )
            with self.assertRaisesRegex(ReferenceAuditError, r"terminated.*\[1\]"):
                audit_uninterrupted_rollout(shard, expected_steps=4)


if __name__ == "__main__":
    unittest.main()
