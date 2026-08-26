"""Trusted coordinator-side dispatch gates for a later evaluation seal."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger.orchestration import (
    HookSpec,
    OrchestrationError,
    SubprocessHookRunner,
    _create_sealable_memfd,
)


HOOK_RELATIVE = "scripts/round_dagger/run_production_paired_evaluation_hook.sh"


class EvaluationDispatchTest(unittest.TestCase):
    def test_libc_memfd_fallback_is_sealable(self) -> None:
        with mock.patch.object(os, "memfd_create", None, create=True):
            descriptor = _create_sealable_memfd("round-dagger-fallback-test")
        try:
            os.write(descriptor, b"trusted")
            from round_dagger import orchestration

            seals = (
                orchestration._F_SEAL_WRITE
                | orchestration._F_SEAL_GROW
                | orchestration._F_SEAL_SHRINK
                | orchestration._F_SEAL_SEAL
            )
            import fcntl

            fcntl.fcntl(descriptor, orchestration._F_ADD_SEALS, seals)
            self.assertEqual(
                fcntl.fcntl(descriptor, orchestration._F_GET_SEALS) & seals,
                seals,
            )
            with self.assertRaises(OSError):
                os.write(descriptor, b"compromised")
        finally:
            os.close(descriptor)

    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        base = Path(temporary.name)
        parent = base / "sealed-evaluation"
        root = parent / "fast_roborender_deadbee_cafef00_sealed"
        hook = root / HOOK_RELATIVE
        hook.parent.mkdir(parents=True)
        marker = base / "hook-ran.txt"
        hook.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "test \"${ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH:-}\" = 1\n"
            "printf trusted > \"$DISPATCH_TEST_MARKER\"\n",
            encoding="utf-8",
        )
        os.chmod(hook, 0o555)
        hook_bytes = hook.read_bytes()
        inventory = {
            "schema": "fast_roborender.hsg_source_inventory.v1",
            "file_count": 1,
            "total_file_bytes": len(hook_bytes),
            "files": [
                {
                    "path": HOOK_RELATIVE,
                    "mode": "0o555",
                    "bytes": len(hook_bytes),
                    "sha256": hashlib.sha256(hook_bytes).hexdigest(),
                }
            ],
        }
        inventory_path = parent / f"{root.name}.inventory.json"
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(inventory_path, 0o444)
        for directory in (hook.parent, hook.parent.parent, root, parent):
            os.chmod(directory, 0o555)
        inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        report = {"inventory_path": str(inventory_path)}
        environment = {
            "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": str(root),
            "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": inventory_sha,
            "DISPATCH_TEST_MARKER": str(marker),
        }
        return temporary, root, hook, marker, report, environment

    def _cleanup(self, temporary: tempfile.TemporaryDirectory) -> None:
        base = Path(temporary.name)
        for directory in sorted(
            (path for path in base.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o755)
        temporary.cleanup()

    def test_verified_descriptor_is_the_only_executed_hook(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        try:
            with mock.patch(
                "round_dagger.orchestration.source_seal.verify",
                return_value=report,
            ):
                SubprocessHookRunner().run(
                    "evaluate",
                    HookSpec((str(hook),), 10),
                    environment=environment,
                    workdir=root,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "trusted")
        finally:
            self._cleanup(temporary)

    def test_hook_mutation_after_full_tree_check_cannot_execute(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        try:
            def mutate_after_verification(*_args, **_kwargs):
                os.chmod(hook, 0o755)
                hook.write_text(
                    "#!/usr/bin/env bash\nprintf compromised > \"$DISPATCH_TEST_MARKER\"\n",
                    encoding="utf-8",
                )
                os.chmod(hook, 0o555)
                return report

            with mock.patch(
                "round_dagger.orchestration.source_seal.verify",
                side_effect=mutate_after_verification,
            ):
                with self.assertRaisesRegex(
                    OrchestrationError, "bytes differ from verified inventory"
                ):
                    SubprocessHookRunner().run(
                        "evaluate",
                        HookSpec((str(hook),), 10),
                        environment=environment,
                        workdir=root,
                    )
            self.assertFalse(marker.exists())
        finally:
            self._cleanup(temporary)

    def test_hook_mutation_after_snapshot_cannot_change_executed_bytes(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        real_run = subprocess.run
        try:
            def mutate_immediately_before_spawn(*args, **kwargs):
                os.chmod(hook, 0o755)
                hook.write_text(
                    "#!/usr/bin/env bash\nprintf compromised > \"$DISPATCH_TEST_MARKER\"\n",
                    encoding="utf-8",
                )
                os.chmod(hook, 0o555)
                return real_run(*args, **kwargs)

            with (
                mock.patch(
                    "round_dagger.orchestration.source_seal.verify",
                    return_value=report,
                ),
                mock.patch(
                    "round_dagger.orchestration.subprocess.run",
                    side_effect=mutate_immediately_before_spawn,
                ),
            ):
                SubprocessHookRunner().run(
                    "evaluate",
                    HookSpec((str(hook),), 10),
                    environment=environment,
                    workdir=root,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "trusted")
        finally:
            self._cleanup(temporary)

    def test_writable_seal_parent_is_rejected_before_hook(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        try:
            os.chmod(root.parent, 0o755)
            with mock.patch(
                "round_dagger.orchestration.source_seal.verify",
                return_value=report,
            ) as verifier:
                with self.assertRaisesRegex(
                    OrchestrationError, "immutable 0555 directory"
                ):
                    SubprocessHookRunner().run(
                        "evaluate",
                        HookSpec((str(hook),), 10),
                        environment=environment,
                        workdir=root,
                    )
            verifier.assert_not_called()
            self.assertFalse(marker.exists())
        finally:
            self._cleanup(temporary)


if __name__ == "__main__":
    unittest.main()
