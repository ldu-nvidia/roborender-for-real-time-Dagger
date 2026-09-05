from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

from round_dagger import native_h200_experiment as protocol


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/round_dagger/build_native_h200_group_registry.py"
REGISTRY_PATH = ROOT / "configs/round_dagger/native_h200_group_registry.json"
AUDIT_PATH = ROOT / "configs/round_dagger/native_h200_group_registry_audit.json"
CONTRACT_PATH = (
    ROOT / "configs/round_dagger/native_h200_bc_dagger5_protocol.v1.json"
)
PANEL_PATH = ROOT / "configs/round_dagger/native_h200_validation_panel.json"
PANEL_AUDIT_PATH = ROOT / "configs/round_dagger/native_h200_validation_panel_audit.json"

SPEC = importlib.util.spec_from_file_location("native_h200_group_registry_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NativeH200GroupRegistryTest(unittest.TestCase):
    def test_checked_in_registry_and_audit_rebuild_from_exact_original_motions(self) -> None:
        registry, audit = builder.build(ROOT)

        self.assertEqual(registry, json.loads(REGISTRY_PATH.read_text(encoding="utf-8")))
        self.assertEqual(audit, json.loads(AUDIT_PATH.read_text(encoding="utf-8")))
        self.assertEqual(registry["schema"], builder.REGISTRY_SCHEMA)
        self.assertEqual(len(registry["groups"]), 33)

        for row in registry["groups"]:
            self.assertEqual(row["group_id"], Path(row["relative_path"]).name)
            self.assertTrue(row["group_id"].startswith("record_2025-10-14_"))
            self.assertEqual(
                set(row),
                {"group_id", "relative_path", "size_bytes", "motion_sha256", "strata"},
            )
            self.assertEqual(
                set(row["strata"]),
                {
                    "table_height",
                    "initial_planar_distance_bin",
                    "initial_world_bearing_bin",
                    "trajectory_length_bin",
                },
            )

        summary = audit["summary"]
        self.assertEqual(summary["motion_count"], 33)
        self.assertEqual(
            summary["stratum_counts"]["table_height"],
            {"675mm": 6, "700mm": 6, "725mm": 6, "750mm": 7, "775mm": 8},
        )
        for field in (
            "initial_planar_distance_bin",
            "initial_world_bearing_bin",
            "trajectory_length_bin",
        ):
            self.assertEqual(sorted(summary["stratum_counts"][field].values()), [11, 11, 11])
        self.assertNotIn("assignments", json.dumps(audit))

    def test_real_split_and_validation_panel_are_representative_and_test_sealed(self) -> None:
        contract = protocol.load_contract(CONTRACT_PATH)
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        split = protocol.build_group_split(contract, registry)
        panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
        panel_audit = json.loads(PANEL_AUDIT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(split["counts"], {"train": 23, "validation": 5, "test": 5})
        self.assertLessEqual(
            split["balance_report"]["max_absolute_residual"], 2.0 / 3.0 + 1e-12
        )
        for row in split["balance_report"]["strata"]:
            self.assertGreater(row["by_split"]["train"], 0)
            self.assertGreater(row["by_split"]["validation"], 0)
            self.assertGreater(row["by_split"]["test"], 0)

        case_groups = [case_id.rsplit("__reset_", 1)[0] for case_id in panel["case_ids"]]
        group_counts = Counter(case_groups)
        self.assertEqual(set(group_counts), set(split["assignments"]["validation"]))
        self.assertTrue(set(group_counts).isdisjoint(split["assignments"]["test"]))
        self.assertEqual(sorted(group_counts.values()), [12, 13, 13, 13, 13])
        self.assertEqual(len(panel["case_ids"]), 64)
        self.assertEqual(panel["case_count"], 256)
        self.assertEqual(panel["paired_seeds"], contract["validation_panel"]["paired_seeds"])
        for group_id, count in group_counts.items():
            indices = sorted(
                int(case_id.rsplit("__reset_", 1)[1])
                for case_id in panel["case_ids"]
                if case_id.startswith(f"{group_id}__reset_")
            )
            self.assertEqual(indices, list(range(count)))

        self.assertEqual(
            panel_audit["validation_registry_sha256"],
            contract["validation_panel"]["registry_sha256"],
        )
        self.assertEqual(panel_audit["split_sha256"], split["split_sha256"])
        self.assertEqual(panel_audit["split_balance_report"], split["balance_report"])
        by_id = {row["group_id"]: row for row in registry["groups"]}
        expected_group_rows = [
            {
                "group_id": group_id,
                "cases_per_seed": group_counts[group_id],
                "episode_count": group_counts[group_id] * len(panel["paired_seeds"]),
                "strata": by_id[group_id]["strata"],
            }
            for group_id in split["assignments"]["validation"]
        ]
        self.assertEqual(panel_audit["group_episode_counts"], expected_group_rows)
        for field in contract["group_split"]["stratify_by"]:
            expected = Counter()
            for row in expected_group_rows:
                expected[row["strata"][field]] += row["episode_count"]
            self.assertEqual(
                panel_audit["per_stratum_episode_counts"][field],
                dict(sorted(expected.items())),
            )
        self.assertFalse(panel_audit["test_boundary"]["checked_in_test_registry"])
        self.assertNotIn("assignments", json.dumps(panel_audit))


if __name__ == "__main__":
    unittest.main()
