import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "containers/h100/patch_image3f_exporter.py"
SPEC = importlib.util.spec_from_file_location("patch_image3f_exporter", PATCHER_PATH)
PATCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PATCHER)


class H100ExporterPatchTest(unittest.TestCase):
    def test_patch_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            exporter = package_root / PATCHER.RELATIVE_EXPORTER
            exporter.parent.mkdir(parents=True)
            exporter.write_text(
                "before\n" + PATCHER.ORIGINAL + "after\n", encoding="utf-8"
            )

            self.assertEqual(PATCHER.patch(package_root), exporter)
            first = exporter.read_text(encoding="utf-8")
            self.assertIn(PATCHER.PATCHED, first)
            self.assertNotIn(PATCHER.ORIGINAL, first)
            self.assertIn("with torch.inference_mode(False):", first)

            self.assertEqual(PATCHER.patch(package_root), exporter)
            self.assertEqual(exporter.read_text(encoding="utf-8"), first)

    def test_patch_rejects_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            exporter = package_root / PATCHER.RELATIVE_EXPORTER
            exporter.parent.mkdir(parents=True)
            exporter.write_text("unknown exporter\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "insertion point"):
                PATCHER.patch(package_root)


if __name__ == "__main__":
    unittest.main()
