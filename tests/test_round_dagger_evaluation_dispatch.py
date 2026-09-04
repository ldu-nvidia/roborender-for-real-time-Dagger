"""Trusted coordinator-side dispatch gates for a later evaluation seal."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
COLLECT_HOOK_RELATIVE = "scripts/round_dagger/run_production_actor_cohort_hook.sh"
TRAIN_HOOK_RELATIVE = (
    "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
            "test \"$(command -v python3)\" = \"$EXPECTED_CONTROL_PYTHON3\"\n"
            "test \"${PYTHONNOUSERSITE:-}\" = 1\n"
            "test \"${PYTHONDONTWRITEBYTECODE:-}\" = 1\n"
            "test \"${VIRTUAL_ENV:-}\" = \"$EXPECTED_VIRTUAL_ENV\"\n"
            "test -z \"${BASH_ENV+x}\"\n"
            "test -z \"${ENV+x}\"\n"
            "test -z \"${PYTHONBREAKPOINT+x}\"\n"
            "test -z \"${PYTHONHOME+x}\"\n"
            "test -z \"${PYTHONINSPECT+x}\"\n"
            "test -z \"${PYTHONPATH+x}\"\n"
            "test -z \"${PYTHONSTARTUP+x}\"\n"
            "python3 -c 'import numpy'\n"
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
            "ROUND_DAGGER_CONTROL_PYTHON": sys.executable,
            "EXPECTED_CONTROL_PYTHON3": str(Path(sys.executable).parent / "python3"),
            "EXPECTED_VIRTUAL_ENV": str(Path(sys.executable).parent.parent),
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

    def test_trusted_hook_scrubs_poisoned_ambient_python_environment(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        try:
            shell_poison = Path(temporary.name) / "ambient-shell-poison.sh"
            poison_marker = Path(temporary.name) / "ambient-shell-poisoned"
            shell_poison.write_text(
                f'printf poisoned > "{poison_marker}"\n', encoding="utf-8"
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "BASH_ENV": str(shell_poison),
                        "ENV": str(shell_poison),
                        "PATH": "/poison/bin",
                        "PYTHONBREAKPOINT": "poison.breakpoint",
                        "PYTHONHOME": "/poison/home",
                        "PYTHONINSPECT": "1",
                        "PYTHONPATH": "/poison/modules",
                        "PYTHONNOUSERSITE": "0",
                        "PYTHONSTARTUP": str(shell_poison),
                        "VIRTUAL_ENV": "/poison/venv",
                    },
                    clear=False,
                ),
                mock.patch(
                    "round_dagger.orchestration.source_seal.verify",
                    return_value=report,
                ),
            ):
                SubprocessHookRunner().run(
                    "evaluate",
                    HookSpec((str(hook),), 10),
                    environment=environment,
                    workdir=root,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "trusted")
            self.assertFalse(poison_marker.exists())
        finally:
            self._cleanup(temporary)

    def test_trusted_hook_rejects_missing_or_relative_control_python(self) -> None:
        for value in (None, "python3"):
            with self.subTest(value=value):
                temporary, root, hook, marker, report, environment = self._fixture()
                try:
                    if value is None:
                        environment.pop("ROUND_DAGGER_CONTROL_PYTHON")
                    else:
                        environment["ROUND_DAGGER_CONTROL_PYTHON"] = value
                    with mock.patch(
                        "round_dagger.orchestration.source_seal.verify",
                        return_value=report,
                    ) as verifier:
                        with self.assertRaisesRegex(
                            OrchestrationError, "absolute control Python"
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

    def test_trusted_hook_rejects_mismatched_python3_sibling(self) -> None:
        temporary, root, hook, marker, report, environment = self._fixture()
        try:
            fake_venv = Path(temporary.name) / "fake-venv"
            fake_bin = fake_venv / "bin"
            fake_bin.mkdir(parents=True)
            (fake_venv / "pyvenv.cfg").write_text("home = /invalid\n", encoding="utf-8")
            control = fake_bin / "python"
            sibling = fake_bin / "python3"
            control.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            sibling.write_text("#!/bin/sh\nexit 0\n# different inode\n", encoding="utf-8")
            control.chmod(0o555)
            sibling.chmod(0o555)
            environment["ROUND_DAGGER_CONTROL_PYTHON"] = str(control)
            with mock.patch(
                "round_dagger.orchestration.source_seal.verify",
                return_value=report,
            ) as verifier:
                with self.assertRaisesRegex(
                    OrchestrationError, "python3 does not match"
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

    def _non_evaluation_fixture(self, name: str, relative: str):
        temporary = tempfile.TemporaryDirectory()
        base = Path(temporary.name)
        parent = base / "sealed-coordinator"
        root = parent / "fast_roborender_deadbee_cafef00_sealed"
        hook = root / relative
        hook.parent.mkdir(parents=True)
        marker = base / f"{name}-ran.txt"
        required_marker = {
            "collect": "ROUND_DAGGER_TRUSTED_COLLECT_DISPATCH",
            "train": "ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH",
        }[name]
        hook.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            f'test "${{{required_marker}:-}}" = 1\n'
            'test "${ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT:-}" = '
            '"$EXPECTED_SOURCE_ROOT"\n'
            'test "$(command -v python3)" = "$EXPECTED_CONTROL_PYTHON3"\n'
            'test "${PYTHONNOUSERSITE:-}" = 1\n'
            'test "${PYTHONDONTWRITEBYTECODE:-}" = 1\n'
            'test "${VIRTUAL_ENV:-}" = "$EXPECTED_VIRTUAL_ENV"\n'
            'test -z "${BASH_ENV+x}"\n'
            'test -z "${ENV+x}"\n'
            'test -z "${PYTHONBREAKPOINT+x}"\n'
            'test -z "${PYTHONHOME+x}"\n'
            'test -z "${PYTHONINSPECT+x}"\n'
            'test -z "${PYTHONPATH+x}"\n'
            'test -z "${PYTHONSTARTUP+x}"\n'
            "python3 -c 'import numpy'\n"
            f'printf trusted > "{marker}"\n',
            encoding="utf-8",
        )
        os.chmod(hook, 0o555)
        payload = hook.read_bytes()
        inventory = {
            "schema": "fast_roborender.hsg_source_inventory.v1",
            "file_count": 1,
            "total_file_bytes": len(payload),
            "files": [
                {
                    "path": relative,
                    "mode": "0o555",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
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
        environment = {
            "ROUND_DAGGER_EVALUATION_SOURCE_ROOT": str(root),
            "ROUND_DAGGER_EVALUATION_SOURCE_INVENTORY_SHA256": inventory_sha,
            "ROUND_DAGGER_CONTROL_PYTHON": sys.executable,
            "EXPECTED_CONTROL_PYTHON3": str(Path(sys.executable).parent / "python3"),
            "EXPECTED_VIRTUAL_ENV": str(Path(sys.executable).parent.parent),
            "EXPECTED_SOURCE_ROOT": str(root),
        }
        if name == "collect":
            # The real seal provides an inventory-bound teacher beneath the
            # same root.  The synthetic hook is sufficient as an in-root path
            # because this test exercises dispatch rather than model loading.
            environment["ROUND_DAGGER_EVALUATION_SEALED_TEACHER"] = str(hook)
        elif name == "train":
            # This synthetic fixture uses the explicitly named legacy OCI
            # bridge above, so bind that backend rather than relying on an
            # ambient/default route.
            environment["ROUND_DAGGER_LEARNER_BACKEND"] = "oci_nrt"
        configured = REPOSITORY_ROOT / relative
        return (
            temporary,
            root,
            configured,
            marker,
            {"inventory_path": str(inventory_path)},
            environment,
        )

    def test_collect_and_train_execute_only_from_common_verified_seal(self) -> None:
        for name, relative in (
            ("collect", COLLECT_HOOK_RELATIVE),
            ("train", TRAIN_HOOK_RELATIVE),
        ):
            with self.subTest(name=name):
                temporary, root, configured, marker, report, environment = (
                    self._non_evaluation_fixture(name, relative)
                )
                try:
                    with mock.patch(
                        "round_dagger.orchestration.source_seal.verify",
                        return_value=report,
                    ):
                        SubprocessHookRunner().run(
                            name,
                            HookSpec((str(configured),), 10),
                            environment=environment,
                            workdir=root,
                        )
                    self.assertEqual(marker.read_text(encoding="utf-8"), "trusted")
                finally:
                    self._cleanup(temporary)


if __name__ == "__main__":
    unittest.main()
