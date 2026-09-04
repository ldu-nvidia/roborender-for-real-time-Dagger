import builtins
import contextlib
import hashlib
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from round_dagger.checkpoint_compat import (
    CheckpointCompatibilityError,
    load_hash_pinned_policy_state_dict,
    load_hash_pinned_restart_document,
    load_restricted_restart_document,
)


try:
    import numpy as np
    import torch

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


_MISSING = object()


@contextlib.contextmanager
def _temporary_pickle_class(module_name, class_name):
    """Publish one importable class under an external package name for save."""

    names = [
        ".".join(module_name.split(".")[:index])
        for index in range(1, len(module_name.split(".")) + 1)
    ]
    previous = {name: sys.modules.get(name, _MISSING) for name in names}
    try:
        parent = None
        for index, name in enumerate(names):
            module = types.ModuleType(name)
            if index < len(names) - 1:
                module.__path__ = []
            sys.modules[name] = module
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[1], module)
            parent = module
        auxiliary = type(class_name, (), {})
        auxiliary.__module__ = module_name
        setattr(sys.modules[module_name], class_name, auxiliary)
        yield auxiliary
    finally:
        for name in reversed(names):
            old = previous[name]
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


@unittest.skipUnless(HAS_RUNTIME, "requires torch and numpy")
class HashPinnedCheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_known_missing_trainer_global_is_discarded_and_bytes_unchanged(self):
        checkpoint = self.root / "bootstrap.pt"
        state = {"weight": torch.arange(4, dtype=torch.float32)}
        with _temporary_pickle_class(
            "trl.trainer.utils", "OnlineTrainerState"
        ) as trainer_state_class:
            trainer_state = trainer_state_class()
            trainer_state.secret_bookkeeping = {"step": 228270}
            torch.save(
                {
                    "policy_state_dict": state,
                    "trainer_state": trainer_state,
                },
                checkpoint,
            )
        expected = self._sha256(checkpoint)
        original = checkpoint.read_bytes()
        real_import = builtins.__import__

        def reject_trainer_import(name, *args, **kwargs):
            if name == "trl" or name.startswith("trl."):
                raise AssertionError("compatibility loader imported TRL")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_trainer_import):
            loaded = load_hash_pinned_policy_state_dict(
                checkpoint, expected_sha256=expected
            )
        torch.testing.assert_close(loaded["weight"], state["weight"], rtol=0, atol=0)
        self.assertEqual(checkpoint.read_bytes(), original)
        self.assertEqual(self._sha256(checkpoint), expected)

    def test_unknown_pickle_global_fails_without_executing_it(self):
        marker = self.root / "must-not-exist"

        class Exploit:
            def __reduce__(self):
                expression = (
                    "__import__('pathlib').Path(" + repr(str(marker)) + ")"
                    ".write_text('executed')"
                )
                return eval, (expression,)

        checkpoint = self.root / "unknown-global.pt"
        torch.save(
            {
                "policy_state_dict": {
                    "weight": torch.ones(1, dtype=torch.float32)
                },
                "unreviewed_auxiliary": Exploit(),
            },
            checkpoint,
        )
        with self.assertRaisesRegex(
            CheckpointCompatibilityError,
            r"(?:builtins|__builtin__)\.eval is not allowlisted",
        ):
            load_hash_pinned_policy_state_dict(
                checkpoint, expected_sha256=self._sha256(checkpoint)
            )
        self.assertFalse(marker.exists())

    def test_learner_candidate_restart_auxiliary_state_is_discarded(self):
        """A promoted candidate stays loadable as the next round's parent."""

        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        loss = model(torch.ones((2, 3))).square().mean()
        loss.backward()
        optimizer.step()
        state = model.state_dict()
        checkpoint = self.root / "candidate_round_000001.pt"
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
                "policy_state_dict": state,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_states": [
                    {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": [],
                    }
                ],
            },
            checkpoint,
        )
        expected = self._sha256(checkpoint)
        original = checkpoint.read_bytes()
        loaded = load_hash_pinned_policy_state_dict(
            checkpoint, expected_sha256=expected
        )
        self.assertEqual(tuple(loaded), tuple(state))
        for name, value in state.items():
            torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_hash_pinned_restart_document_retains_actor_provenance(self):
        """RTX warm-start needs controller provenance, not just model weights."""

        checkpoint = self.root / "candidate_with_controller.pt"
        state = {"weight": torch.arange(4, dtype=torch.float32)}
        controller_sha = {"walk": "a" * 64, "stand": "b" * 64}
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
                "policy_state_dict": state,
                "homie_controller_sha256": controller_sha,
                "rng_states": [{"numpy": np.random.get_state()}],
            },
            checkpoint,
        )
        expected = self._sha256(checkpoint)
        loaded = load_hash_pinned_restart_document(
            checkpoint,
            expected_sha256=expected,
        )
        self.assertEqual(loaded["homie_controller_sha256"], controller_sha)
        torch.testing.assert_close(
            loaded["policy_state_dict"]["weight"], state["weight"]
        )
        with self.assertRaisesRegex(CheckpointCompatibilityError, "SHA-256 mismatch"):
            load_hash_pinned_restart_document(
                checkpoint,
                expected_sha256="0" * 64,
            )

    def test_hash_mismatch_and_symlink_fail_before_deserialization(self):
        checkpoint = self.root / "policy.pt"
        torch.save(
            {"policy_state_dict": {"weight": torch.ones(1)}}, checkpoint
        )
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "SHA-256 mismatch"
        ):
            load_hash_pinned_policy_state_dict(
                checkpoint, expected_sha256="f" * 64
            )
        link = self.root / "policy-link.pt"
        link.symlink_to(checkpoint)
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "non-symlink"
        ):
            load_hash_pinned_policy_state_dict(
                link, expected_sha256=self._sha256(checkpoint)
            )

    def test_unreviewed_torch_storage_type_fails_pickle_preflight(self):
        checkpoint = self.root / "double-storage.pt"
        torch.save(
            {
                "policy_state_dict": {
                    "weight": torch.ones(1, dtype=torch.float64)
                }
            },
            checkpoint,
        )
        with self.assertRaisesRegex(
            CheckpointCompatibilityError,
            r"torch\.DoubleStorage is not allowlisted",
        ):
            load_hash_pinned_policy_state_dict(
                checkpoint, expected_sha256=self._sha256(checkpoint)
            )

    def test_only_policy_state_dict_is_accepted(self):
        checkpoint = self.root / "wrong-key.pt"
        torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "lacks.*policy_state_dict"
        ):
            load_hash_pinned_policy_state_dict(
                checkpoint, expected_sha256=self._sha256(checkpoint)
            )

    def test_restricted_restart_retains_optimizer_and_rng_state(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        loss = model(torch.ones((2, 3))).square().mean()
        loss.backward()
        optimizer.step()
        numpy_state = np.random.get_state()
        checkpoint = self.root / "restart.pt"
        document = {
            "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_states": [
                {
                    "python": random.getstate(),
                    "numpy": numpy_state,
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": [],
                }
            ],
        }
        torch.save(document, checkpoint)

        loaded = load_restricted_restart_document(checkpoint)

        self.assertEqual(loaded["schema"], document["schema"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                loaded["model_state_dict"][name], value, rtol=0, atol=0
            )
        self.assertEqual(
            set(loaded["optimizer_state_dict"]),
            set(document["optimizer_state_dict"]),
        )
        restored_numpy = loaded["rng_states"][0]["numpy"]
        self.assertEqual(restored_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
        self.assertEqual(restored_numpy[2:], numpy_state[2:])

    def test_restricted_restart_rejects_unknown_global_without_execution(self):
        marker = self.root / "restart-must-not-exist"

        class Exploit:
            def __reduce__(self):
                expression = (
                    "__import__('pathlib').Path(" + repr(str(marker)) + ")"
                    ".write_text('executed')"
                )
                return eval, (expression,)

        checkpoint = self.root / "malicious-restart.pt"
        torch.save({"rng_states": [], "payload": Exploit()}, checkpoint)
        with self.assertRaisesRegex(
            CheckpointCompatibilityError,
            r"(?:builtins|__builtin__)\.eval is not allowlisted",
        ):
            load_restricted_restart_document(checkpoint)
        self.assertFalse(marker.exists())

    def test_restricted_restart_rejects_symlink(self):
        checkpoint = self.root / "restart.pt"
        torch.save({"rng_states": []}, checkpoint)
        link = self.root / "restart-link.pt"
        link.symlink_to(checkpoint)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "non-symlink"):
            load_restricted_restart_document(link)


if __name__ == "__main__":
    unittest.main()
