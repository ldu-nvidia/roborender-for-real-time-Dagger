from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.round_dagger import remote_learner_success as success


class RemoteLearnerSuccessTest(unittest.TestCase):
    def test_publish_verify_and_exact_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "learner_result.json"
            result.write_bytes(b"result")
            attestation = root / "learner_completion.json"
            attestation.write_bytes(b"completion")
            marker = root / "job/success.json"
            arguments = {
                "campaign_id": "campaign-1",
                "macro_round": 0,
                "learner_code_sha256": "1" * 64,
                "learner_config_sha256": "2" * 64,
            }
            first = success.publish(marker, result, attestation, **arguments)
            self.assertEqual(success.publish(marker, result, attestation, **arguments), first)
            self.assertEqual(success.verify(marker, result, attestation, **arguments), first)
            self.assertEqual(marker.stat().st_mode & 0o777, 0o444)

    def test_result_or_lineage_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "learner_result.json"
            result.write_bytes(b"result")
            attestation = root / "learner_completion.json"
            attestation.write_bytes(b"completion")
            marker = root / "success.json"
            arguments = {
                "campaign_id": "campaign-1",
                "macro_round": 0,
                "learner_code_sha256": "1" * 64,
                "learner_config_sha256": "2" * 64,
            }
            success.publish(marker, result, attestation, **arguments)
            result.write_bytes(b"changed")
            with self.assertRaisesRegex(
                success.RemoteLearnerSuccessError, "lineage mismatch"
            ):
                success.verify(marker, result, attestation, **arguments)
            with self.assertRaisesRegex(
                success.RemoteLearnerSuccessError, "different success marker"
            ):
                success.publish(marker, result, attestation, **arguments)

    def test_writable_or_symlink_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_bytes(b"result")
            attestation = root / "learner_completion.json"
            attestation.write_bytes(b"completion")
            marker = root / "success.json"
            arguments = {
                "campaign_id": "campaign-1",
                "macro_round": 0,
                "learner_code_sha256": "1" * 64,
                "learner_config_sha256": "2" * 64,
            }
            success.publish(marker, result, attestation, **arguments)
            marker.chmod(0o644)
            with self.assertRaisesRegex(
                success.RemoteLearnerSuccessError, "read-only"
            ):
                success.verify(marker, result, attestation, **arguments)
            with self.assertRaisesRegex(
                success.RemoteLearnerSuccessError, "read-only"
            ):
                success.publish(marker, result, attestation, **arguments)
            marker.unlink()
            marker.symlink_to(result)
            with self.assertRaisesRegex(
                success.RemoteLearnerSuccessError, "unsafe"
            ):
                success.verify(marker, result, attestation, **arguments)


if __name__ == "__main__":
    unittest.main()
