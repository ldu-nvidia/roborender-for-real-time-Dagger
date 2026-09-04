from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts, learner_acceptance


class LearnerAcceptancePublicationTest(unittest.TestCase):
    @staticmethod
    def _document(value: str = "a") -> dict[str, object]:
        document: dict[str, object] = {
            key: value * 64 for key in learner_acceptance._KEYS
        }
        document.update(
            schema=learner_acceptance.SCHEMA,
            status="PASS",
            backend=learner_acceptance.BACKEND,
            campaign_id="campaign",
            macro_round=0,
            completion_checksum_source_basename=(
                learner_acceptance.REMOTE_MARKER_BASENAME
            ),
        )
        document.pop("content_sha256")
        document["content_sha256"] = artifacts.canonical_json_sha256(document)
        return document

    def _publish(self, root: Path, document: dict[str, object]):
        receipt = learner_acceptance.acceptance_path(root, 0)
        root.mkdir(exist_ok=True)
        document = dict(document)
        document.update(
            request_path=str(root / "request.json"),
            completion_marker_path=str(root / "marker.json"),
            completion_checksum_path=str(root / "marker.json.sha256"),
            returned_manifest_path=str(root / "bundle/manifest.json"),
            returned_output_path="rounds/macro_round_000000/learner",
        )
        document.pop("content_sha256")
        document["content_sha256"] = artifacts.canonical_json_sha256(document)
        with mock.patch.object(
            learner_acceptance, "_collect", return_value=document
        ):
            return learner_acceptance.publish_acceptance(
                receipt_path=receipt,
                campaign_root=root,
                macro_round=0,
                backend_contract_path=root / "contract.json",
                request_path=root / "request.json",
                marker_path=root / "marker.json",
                checksum_path=root / "marker.json.sha256",
                return_bundle_root=root / "bundle",
                learner_output=root / "rounds/macro_round_000000/learner",
                expected_environment={},
            )

    def test_identical_receipt_is_adopted_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            document = self._document()
            first = self._publish(root, document)
            path = learner_acceptance.acceptance_path(root, 0)
            inode = path.stat().st_ino
            second = self._publish(root, document)
            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_ino, inode)

    def test_conflicting_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            self._publish(root, self._document("a"))
            with self.assertRaisesRegex(
                learner_acceptance.LearnerAcceptanceError, "conflicts"
            ):
                self._publish(root, self._document("b"))


if __name__ == "__main__":
    unittest.main()
