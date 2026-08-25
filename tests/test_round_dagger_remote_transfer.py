from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.round_dagger import remote_campaign_transfer as transfer


class RemoteCampaignTransferTest(unittest.TestCase):
    def test_inventory_is_sorted_content_bound_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            (root / "rounds/r0").mkdir(parents=True)
            (root / "rounds/r0/z.bin").write_bytes(b"payload")
            (root / "campaign.json").write_text("{}\n", encoding="utf-8")
            inventory = transfer.build_inventory(root)
            output = Path(temporary) / "inventory.json"
            transfer.publish_inventory(output, inventory)

            self.assertEqual(
                [item["path"] for item in inventory["files"]],
                ["campaign.json", "rounds/r0/z.bin"],
            )
            self.assertEqual(transfer.verify_inventory(root, output), inventory)

            (root / "rounds/r0/z.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(
                transfer.TransferInventoryError, "differs"
            ):
                transfer.verify_inventory(root, output)

    def test_symlink_and_special_entries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            root.mkdir()
            (root / "real").write_text("x", encoding="utf-8")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(
                transfer.TransferInventoryError, "non-regular"
            ):
                transfer.build_inventory(root)

    def test_publish_is_atomic_and_retry_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage = base / ".incoming-campaign"
            stage.mkdir()
            (stage / "value").write_bytes(b"one")
            inventory = base / "inventory.json"
            transfer.publish_inventory(
                inventory, transfer.build_inventory(stage)
            )
            destination = base / "campaign"

            first = transfer.publish_tree(stage, destination, inventory)
            self.assertFalse(stage.exists())
            self.assertEqual(first["tree_sha256"], transfer.build_inventory(destination)["tree_sha256"])

            retry = base / ".incoming-retry"
            retry.mkdir()
            (retry / "value").write_bytes(b"one")
            second = transfer.publish_tree(retry, destination, inventory)
            self.assertEqual(second, first)
            self.assertFalse(retry.exists())

            (destination / "value").write_bytes(b"two")
            retry.mkdir()
            (retry / "value").write_bytes(b"one")
            with self.assertRaisesRegex(
                transfer.TransferInventoryError, "differs"
            ):
                transfer.publish_tree(retry, destination, inventory)

    def test_inventory_is_permission_independent_but_structure_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            root.mkdir()
            value = root / "value"
            value.write_bytes(b"one")
            before = transfer.build_inventory(root)
            value.chmod(0o444)
            root.chmod(0o555)
            self.assertEqual(transfer.build_inventory(root), before)
            self.assertEqual(
                transfer.validate_tree_structure(root),
                {"file_count": 1, "total_bytes": 3},
            )

    def test_publish_rejects_same_or_overlapping_stage_and_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage"
            root.mkdir()
            (root / "value").write_bytes(b"one")
            inventory = Path(temporary) / "inventory.json"
            transfer.publish_inventory(inventory, transfer.build_inventory(root))

            for destination in (root, root / "nested", root.parent):
                with self.subTest(destination=destination), self.assertRaisesRegex(
                    transfer.TransferInventoryError, "disjoint"
                ):
                    transfer.publish_tree(root, destination, inventory)
                self.assertTrue(root.is_dir())
                self.assertEqual((root / "value").read_bytes(), b"one")

    def test_duplicate_or_unsafe_inventory_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            body = {
                "schema": transfer.SCHEMA,
                "file_count": 1,
                "total_bytes": 0,
                "files": [
                    {"path": "../escape", "size": 0, "sha256": "0" * 64}
                ],
                "tree_sha256": "0" * 64,
            }
            path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(
                transfer.TransferInventoryError, "unsafe path"
            ):
                transfer.load_inventory(path)


if __name__ == "__main__":
    unittest.main()
