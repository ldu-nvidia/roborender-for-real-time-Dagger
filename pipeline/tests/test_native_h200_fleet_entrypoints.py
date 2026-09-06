from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from round_dagger.native_h200_experiment import (
    NativeH200ExperimentError,
    _safe_launch_executable,
)
from round_dagger.native_h200_fleet import (
    NativeH200FleetError,
    safe_python_launcher,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "run_native_h200_learner_fleet.py",
    "run_native_h200_mse_fleet.py",
    "run_native_h200_test_fleet.py",
)


class NativeH200FleetEntrypointTest(unittest.TestCase):
    def test_protocol_entrypoints_are_executable_regular_files(self) -> None:
        for name in ENTRYPOINTS:
            with self.subTest(name=name):
                relative = f"scripts/round_dagger/{name}"
                self.assertEqual(
                    _safe_launch_executable(ROOT, relative), ROOT / relative
                )

    def test_launch_entrypoint_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fleet.py"
            executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            executable.chmod(0o755)
            alias = root / "alias.py"
            alias.symlink_to(executable)
            with self.assertRaises(NativeH200ExperimentError):
                _safe_launch_executable(root, alias.name)

    def test_python_launcher_target_cannot_escape_its_pinned_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admitted = root / "admitted"
            outside = root / "outside"
            admitted.mkdir()
            outside.mkdir()
            target = outside / "python"
            target.write_bytes(b"python")
            target.chmod(0o755)
            launcher = admitted / "python"
            launcher.symlink_to(target)
            with self.assertRaises(NativeH200FleetError):
                safe_python_launcher(
                    launcher, "test Python", target_root=admitted
                )


if __name__ == "__main__":
    unittest.main()
