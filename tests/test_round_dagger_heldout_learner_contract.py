from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.gr00t_policy import REPLAY_ACTION_DIM, VISION_SHAPE
from round_dagger.heldout_learner_contract import (
    EVALUATION_INTERVAL_STEPS,
    HeldoutLearnerContractError,
    materialize_heldout_learner_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = (
    ROOT / "configs/round_dagger/production_learner_8xh100_1200.json"
)
CAMPAIGN_ID = "dagger_prod_run1"


class HeldoutLearnerContractMaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr-heldout-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.snapshot = self.root / "heldout_snapshot.json"
        self.snapshot.write_bytes(b"fixture-heldout-snapshot\n")
        self.snapshot_file_sha = artifacts.sha256_file(self.snapshot)
        self.snapshot_semantic_sha = "8" * 64
        self.attestation = self.root / "heldout_conversion_attestation.json"
        self.attestation.write_text(
            json.dumps(
                {
                    "fixed_bundle": {
                        "snapshot_path": str(self.snapshot),
                        "snapshot_file_sha256": self.snapshot_file_sha,
                        "snapshot_sha256": self.snapshot_semantic_sha,
                        "case_manifest_path": str(
                            self.root / "heldout_case_manifest.json"
                        ),
                        "case_manifest_file_sha256": "1" * 64,
                        "case_manifest_sha256": "2" * 64,
                        "verified_snapshot_sha256": self.snapshot_semantic_sha,
                    }
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.output = self.root / "production-learner-heldout.json"

    def _verified_conversion(self, *, campaign_id: str = CAMPAIGN_ID) -> dict:
        return {
            "status": "verified",
            "path": str(self.attestation),
            "file_sha256": artifacts.sha256_file(self.attestation),
            "attestation_sha256": "3" * 64,
            "campaign_id": campaign_id,
            "case_count": 256,
            "cohort_count": 8,
            "snapshot_path": str(self.snapshot),
            "snapshot_sha256": self.snapshot_semantic_sha,
            "case_manifest_path": str(self.root / "heldout_case_manifest.json"),
        }

    def _snapshot_document(
        self,
        *,
        campaign_id: str = CAMPAIGN_ID,
        view_shape: tuple[int, int, int] = VISION_SHAPE,
    ) -> dict:
        return {
            "campaign_id": campaign_id,
            "cohort_count": 8,
            "transition_count": 256,
            "snapshot_sha256": self.snapshot_semantic_sha,
            "file_sha256": self.snapshot_file_sha,
            "view_contract": {
                "dtype": "uint8",
                "layout": "vertical",
                "shape": list(view_shape),
                "views": ["head_color", "hand_left", "hand_right"],
            },
            "action_contract": {
                "dtype": "float32",
                "shape": [REPLAY_ACTION_DIM],
            },
        }

    def _load_snapshot(self, _path: Path, *, expected_file_sha256: str) -> dict:
        if expected_file_sha256 != self.snapshot_file_sha:
            raise RuntimeError("held-out replay snapshot file digest mismatch")
        return self._snapshot_document()

    def test_binds_exact_snapshot_and_interval_for_dagger_prod_run1(self) -> None:
        verified = self._verified_conversion()
        with (
            mock.patch(
                "round_dagger.heldout_learner_contract.verify_conversion_bundle",
                return_value=verified,
            ) as verify,
            mock.patch(
                "round_dagger.heldout_learner_contract.replay.load_heldout_snapshot",
                side_effect=self._load_snapshot,
            ),
        ):
            report = materialize_heldout_learner_contract(
                base_contract_path=BASE_CONTRACT,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(report["campaign_id"], CAMPAIGN_ID)
        self.assertEqual(report["case_count"], 256)
        self.assertEqual(report["cohort_count"], 8)
        self.assertEqual(report["output_file_sha256"], artifacts.sha256_file(self.output))
        document = artifacts.load_strict_json(self.output)
        self.assertEqual(
            document["heldout_evaluation"],
            {
                "snapshot_path": str(self.snapshot),
                "snapshot_file_sha256": self.snapshot_file_sha,
                "snapshot_sha256": self.snapshot_semantic_sha,
                "interval_steps": EVALUATION_INTERVAL_STEPS,
            },
        )
        self.assertEqual(os.stat(self.output).st_mode & 0o777, 0o444)

    def test_wrong_campaign_fails_before_publication(self) -> None:
        with (
            mock.patch(
                "round_dagger.heldout_learner_contract.verify_conversion_bundle",
                return_value=self._verified_conversion(campaign_id="other-campaign"),
            ),
            self.assertRaisesRegex(
                HeldoutLearnerContractError, "fixed-256 campaign bundle"
            ),
        ):
            materialize_heldout_learner_contract(
                base_contract_path=BASE_CONTRACT,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_unverified_snapshot_digest_fails_before_publication(self) -> None:
        document = json.loads(self.attestation.read_text(encoding="utf-8"))
        document["fixed_bundle"]["snapshot_file_sha256"] = "f" * 64
        self.attestation.write_text(json.dumps(document) + "\n", encoding="utf-8")
        verified = self._verified_conversion()
        with (
            mock.patch(
                "round_dagger.heldout_learner_contract.verify_conversion_bundle",
                return_value=verified,
            ),
            mock.patch(
                "round_dagger.heldout_learner_contract.replay.load_heldout_snapshot",
                side_effect=self._load_snapshot,
            ),
            self.assertRaisesRegex(
                HeldoutLearnerContractError, "snapshot is invalid"
            ),
        ):
            materialize_heldout_learner_contract(
                base_contract_path=BASE_CONTRACT,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_non_production_view_shape_fails_before_publication(self) -> None:
        def wrong_shape(_path: Path, *, expected_file_sha256: str) -> dict:
            self.assertEqual(expected_file_sha256, self.snapshot_file_sha)
            return self._snapshot_document(view_shape=(6, 4, 3))

        with (
            mock.patch(
                "round_dagger.heldout_learner_contract.verify_conversion_bundle",
                return_value=self._verified_conversion(),
            ),
            mock.patch(
                "round_dagger.heldout_learner_contract.replay.load_heldout_snapshot",
                side_effect=wrong_shape,
            ),
            self.assertRaisesRegex(
                HeldoutLearnerContractError, "three-view 26D evaluation data"
            ),
        ):
            materialize_heldout_learner_contract(
                base_contract_path=BASE_CONTRACT,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_output_is_create_only(self) -> None:
        self.output.write_text("do not replace\n", encoding="utf-8")
        with self.assertRaisesRegex(
            HeldoutLearnerContractError, "already exists"
        ):
            materialize_heldout_learner_contract(
                base_contract_path=BASE_CONTRACT,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertEqual(self.output.read_text(encoding="utf-8"), "do not replace\n")

    def test_already_bound_base_cannot_be_rebound(self) -> None:
        base = artifacts.load_strict_json(BASE_CONTRACT)
        base["heldout_evaluation"] = {
            "snapshot_path": str(self.snapshot),
            "snapshot_file_sha256": self.snapshot_file_sha,
            "snapshot_sha256": self.snapshot_semantic_sha,
            "interval_steps": 100,
        }
        bound = self.root / "already-bound.json"
        bound.write_text(json.dumps(base) + "\n", encoding="utf-8")
        with (
            mock.patch(
                "round_dagger.heldout_learner_contract.replay.load_heldout_snapshot",
                side_effect=self._load_snapshot,
            ),
            self.assertRaisesRegex(
                HeldoutLearnerContractError, "rebinding is forbidden"
            ),
        ):
            materialize_heldout_learner_contract(
                base_contract_path=bound,
                conversion_attestation_path=self.attestation,
                campaign_root=self.root,
                campaign_id=CAMPAIGN_ID,
                output_path=self.output,
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
