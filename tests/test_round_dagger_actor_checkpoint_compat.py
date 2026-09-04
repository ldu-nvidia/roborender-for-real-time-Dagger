"""Unit tests for the narrow RTX NumPy-2 actor warm-start bridge."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from round_dagger.actor_checkpoint_compat import (
    ActorCheckpointCompatibilityError,
    install_actor_warmstart_checkpoint_compat,
)


CHECKPOINT_IO_MODULE = "gr00t.rl.trl.trainer.checkpoint_io"


def _module_tree(checkpoint_io: types.ModuleType) -> dict[str, types.ModuleType]:
    names = CHECKPOINT_IO_MODULE.split(".")
    modules: dict[str, types.ModuleType] = {}
    parent: types.ModuleType | None = None
    for index in range(1, len(names)):
        name = ".".join(names[:index])
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        if parent is not None:
            setattr(parent, names[index - 1], module)
        parent = module
    assert parent is not None
    setattr(parent, names[-1], checkpoint_io)
    modules[CHECKPOINT_IO_MODULE] = checkpoint_io
    return modules


class ActorWarmstartCheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.policy = self.root / "policy.pt"
        self.policy.write_bytes(b"hash-pinned-policy")
        self.sha = hashlib.sha256(self.policy.read_bytes()).hexdigest()
        self.teacher = self.root / "teacher.pt"
        self.teacher.write_bytes(b"teacher")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_numpy2_policy_falls_back_to_restricted_hash_pinned_loader(self):
        checkpoint_io = types.ModuleType(CHECKPOINT_IO_MODULE)
        calls: list[tuple[str, str | None]] = []

        def native(_torch, path, *, expected_sha256=None, **_kwargs):
            calls.append((str(path), expected_sha256))
            if str(path) == str(self.policy):
                raise ModuleNotFoundError("missing NumPy-2 alias", name="numpy._core")
            return {"native": True}, "f" * 64

        checkpoint_io.load_verified_torch_checkpoint = native
        modules = _module_tree(checkpoint_io)
        expected_document = {
            "policy_state_dict": {"weight": object()},
            "homie_controller_sha256": {"walk": "a" * 64, "stand": "b" * 64},
        }
        with mock.patch.dict(sys.modules, modules, clear=False), mock.patch(
            "round_dagger.actor_checkpoint_compat.load_hash_pinned_restart_document",
            return_value=expected_document,
        ) as fallback:
            install_actor_warmstart_checkpoint_compat(
                policy_checkpoint=self.policy,
                policy_sha256=self.sha,
            )
            result, actual = checkpoint_io.load_verified_torch_checkpoint(
                object(),
                self.policy,
                map_location="cpu",
                expected_sha256=self.sha,
            )
            self.assertIs(result, expected_document)
            self.assertEqual(actual, self.sha)
            fallback.assert_called_once_with(
                str(self.policy), expected_sha256=self.sha, map_location="cpu"
            )
            # A non-policy checkpoint remains on GR00T's native code path.
            teacher, teacher_sha = checkpoint_io.load_verified_torch_checkpoint(
                object(), self.teacher, map_location="cpu", expected_sha256=None
            )
            self.assertEqual(teacher, {"native": True})
            self.assertEqual(teacher_sha, "f" * 64)
        self.assertEqual(calls, [(str(self.policy), self.sha), (str(self.teacher), None)])

    def test_policy_fallback_requires_the_contract_bound_digest(self):
        checkpoint_io = types.ModuleType(CHECKPOINT_IO_MODULE)
        checkpoint_io.load_verified_torch_checkpoint = lambda *_args, **_kwargs: None
        with mock.patch.dict(sys.modules, _module_tree(checkpoint_io), clear=False):
            install_actor_warmstart_checkpoint_compat(
                policy_checkpoint=self.policy,
                policy_sha256=self.sha,
            )
            with self.assertRaisesRegex(
                ActorCheckpointCompatibilityError, "contract-bound policy SHA-256"
            ):
                checkpoint_io.load_verified_torch_checkpoint(
                    object(),
                    self.policy,
                    map_location="cpu",
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
