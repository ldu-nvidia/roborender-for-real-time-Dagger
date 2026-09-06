from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger import heldout_contracts as heldout


ROOT = Path(__file__).resolve().parents[1]
MOTION_ROOT = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/data/motions/g1_wsdpt/33demos_675_775"
)
MOTION_SPLIT_MODULE = (
    ROOT
    / "GR00T-VisualSim2Real/gr00t/rl/utils/teleop_reset_lib/motion_split.py"
)
SPEC = importlib.util.spec_from_file_location("native_motion_split_resolver", MOTION_SPLIT_MODULE)
assert SPEC is not None and SPEC.loader is not None
motion_split = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_split
SPEC.loader.exec_module(motion_split)


class NativeH200HeldoutContractsTest(unittest.TestCase):
    def _publish_pair(self, root: Path) -> tuple[Path, Path]:
        cases_sha = "a" * 64
        registry = {
            "schema": "fast_roborender.round_dagger.sealed_test_registry.v1",
            "groups": [f"group_{index}" for index in range(5)],
            "case_count": 512,
            "cases_sha256": cases_sha,
            "stratum_case_counts": {
                "s0": 103,
                "s1": 103,
                "s2": 102,
                "s3": 102,
                "s4": 102,
            },
        }
        commitment = {
            "schema": heldout.TEST_COMMITMENT_SCHEMA,
            "protocol_id": "roborender_h200_v001",
            "split_sha256": "b" * 64,
            "test_registry_sha256": heldout.digest(registry),
            "test_cases_sha256": cases_sha,
        }
        registry_path = root / "registry.json"
        commitment_path = root / "commitment.json"
        heldout.publish_create_only(registry_path, registry, mode=0o400)
        heldout.publish_create_only(commitment_path, commitment, mode=0o444)
        return commitment_path, registry_path

    def test_create_only_publication_refuses_even_identical_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="heldout-create-only-") as temporary:
            path = Path(temporary) / "artifact.json"
            heldout.publish_create_only(path, {"value": 1}, mode=0o444)
            with self.assertRaisesRegex(heldout.HeldoutContractError, "overwrite"):
                heldout.publish_create_only(path, {"value": 1}, mode=0o444)

    def test_commitment_verification_does_not_open_detailed_cases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="heldout-commitment-") as temporary:
            root = Path(temporary)
            commitment, registry = self._publish_pair(root)
            # No detailed cases file exists. Verification follows only the
            # transitive digest recorded in the controller-compatible registry.
            report = heldout.verify_commitment(commitment, registry)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["case_count"], 512)

            payload = json.loads(commitment.read_text(encoding="utf-8"))
            serialized = json.dumps(payload, sort_keys=True)
            self.assertNotIn("group_", serialized)
            self.assertNotIn("evaluation_seed", serialized)
            self.assertNotIn("case_id", serialized)

    def test_private_case_builder_is_fixed_balanced_and_transitively_bound(self) -> None:
        groups = []
        test_ids = []
        fields = [
            "table_height",
            "initial_planar_distance_bin",
            "initial_world_bearing_bin",
            "trajectory_length_bin",
        ]
        for index in range(5):
            group_id = f"test_group_{index}"
            test_ids.append(group_id)
            groups.append(
                {
                    "group_id": group_id,
                    "relative_path": f"motions/{group_id}.npz",
                    "size_bytes": 1,
                    "motion_sha256": str(index + 1) * 64,
                    "strata": {
                        "table_height": f"h{index}",
                        "initial_planar_distance_bin": f"d{index}",
                        "initial_world_bearing_bin": f"b{index}",
                        "trajectory_length_bin": f"l{index}",
                    },
                }
            )
        contract = {
            "protocol_id": "roborender_h200_v001",
            "validation_panel": {"paired_seeds": [11, 22, 33, 44]},
            "group_split": {"stratify_by": fields},
        }
        split = {
            "split_sha256": "c" * 64,
            "groups": groups,
            "assignments": {"test": test_ids},
        }
        seeds = [101, 202, 303, 404, 505, 606, 707, 808]
        cases, registry, commitment = heldout._test_case_documents(  # noqa: SLF001
            contract, split, seeds
        )
        self.assertEqual(cases["evaluation_seeds"], seeds)
        self.assertEqual(len(cases["cases"]), 512)
        counts = sorted(cases["expected_counts"]["by_group"].values())
        self.assertEqual(counts, [102, 102, 102, 103, 103])
        self.assertEqual(sum(registry["stratum_case_counts"].values()), 512)
        self.assertEqual(registry["cases_sha256"], heldout.digest(cases))
        self.assertEqual(commitment["test_registry_sha256"], heldout.digest(registry))

    def test_real_role_partition_resolves_from_configured_motion_directory(self) -> None:
        contract = heldout.load_json(heldout.PROTOCOL_PATH)
        registry = heldout.load_json(heldout.GROUP_REGISTRY_PATH)
        panel = heldout.load_json(heldout.VALIDATION_PANEL_PATH)
        documents = heldout.build_documents(
            contract,
            registry,
            panel,
            evaluation_seeds=[101, 202, 303, 404, 505, 606, 707, 808],
        )
        partition = documents["validation_partition"]
        group_ids = [row["group_id"] for row in partition["groups"]]
        self.assertEqual(
            [row["relative_path"] for row in partition["groups"]], group_ids
        )
        with tempfile.TemporaryDirectory(prefix="heldout-resolver-") as temporary:
            manifest = Path(temporary) / "motion_partition.v1.json"
            manifest_sha256 = heldout.publish_create_only(
                manifest, partition, mode=0o444
            )
            resolved = motion_split.resolve_motion_files(
                MOTION_ROOT,
                manifest_path=manifest,
                expected_manifest_sha256=manifest_sha256,
                role="validation",
                group_ids=group_ids,
            )
        self.assertEqual([Path(path).name for path in resolved], group_ids)


if __name__ == "__main__":
    unittest.main()
