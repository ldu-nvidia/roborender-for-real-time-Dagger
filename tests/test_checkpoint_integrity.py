import hashlib
import importlib.util
import math
import os
import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.scaling.checkpoint_policy_gate import inspect, validate_change
from scripts.scaling.resume_continuity_gate import (
    GateError,
    _validate_resumable_state,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_IO = ROOT / (
    "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/checkpoint_io.py"
)
PPO_TRAINER = ROOT / (
    "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/ppo_trainer.py"
)


def _load_checkpoint_io():
    spec = importlib.util.spec_from_file_location("checkpoint_io_under_test", CHECKPOINT_IO)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeReduction:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, values, *, dtype="float32", shape=None):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = tuple(shape if shape is not None else (len(self.values),))

    def detach(self):
        return self

    def contiguous(self):
        return self

    def to(self, **_kwargs):
        return self

    def is_floating_point(self):
        return self.dtype.startswith("float")

    def is_complex(self):
        return self.dtype.startswith("complex")

    def reshape(self, *_shape):
        return self

    def view(self, _dtype):
        return self

    def numpy(self):
        return self

    def tobytes(self, order="C"):
        del order
        return pickle.dumps((self.dtype, self.shape, self.values), protocol=4)


class FakeTorch:
    uint8 = "uint8"

    @staticmethod
    def load(handle, *, map_location, weights_only):
        del map_location, weights_only
        return pickle.load(handle)

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(tensor):
        return FakeReduction(all(math.isfinite(value) for value in tensor.values))


def _write_checkpoint(
    path: Path,
    *,
    step: int,
    values=(1.0, 2.0),
    name="weight",
    dtype="float32",
    shape=None,
) -> str:
    payload = {
        "checkpoint_schema": "gr00t.trl.full_state.v2",
        "state": {"global_step": step},
        "policy_parameter_names": [name],
        "policy_state_dict": {
            name: FakeTensor(values, dtype=dtype, shape=shape),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return _sha256(path)


class CheckpointFileDescriptorTest(unittest.TestCase):
    def test_hash_and_deserialization_use_the_same_descriptor(self):
        checkpoint_io = _load_checkpoint_io()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            replacement = root / "replacement.pt"
            checkpoint.write_bytes(pickle.dumps({"version": "original"}))
            replacement.write_bytes(pickle.dumps({"version": "replacement"}))
            expected = _sha256(checkpoint)

            class ReplacingTorch:
                @staticmethod
                def load(handle, *, map_location, weights_only):
                    del map_location, weights_only
                    os.replace(replacement, checkpoint)
                    return pickle.load(handle)

            payload, actual = checkpoint_io.load_verified_torch_checkpoint(
                ReplacingTorch,
                checkpoint,
                map_location="cpu",
                expected_sha256=expected,
                require_expected_sha256=True,
            )
            self.assertEqual(payload, {"version": "original"})
            self.assertEqual(actual, expected)
            self.assertEqual(pickle.loads(checkpoint.read_bytes()), {"version": "replacement"})

    def test_production_digest_is_required_and_checked_before_load(self):
        checkpoint_io = _load_checkpoint_io()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.write_bytes(pickle.dumps({"ok": True}))
            with self.assertRaisesRegex(RuntimeError, "requires a checkpoint SHA-256"):
                checkpoint_io.load_verified_torch_checkpoint(
                    FakeTorch,
                    checkpoint,
                    map_location="cpu",
                    expected_sha256="NONE",
                    require_expected_sha256=True,
                )
            with self.assertRaisesRegex(RuntimeError, "mismatch at actual load"):
                checkpoint_io.load_verified_torch_checkpoint(
                    FakeTorch,
                    checkpoint,
                    map_location="cpu",
                    expected_sha256="0" * 64,
                    require_expected_sha256=True,
                )

    def test_rank_runtime_digest_contract_is_ordered_and_fail_closed(self):
        checkpoint_io = _load_checkpoint_io()
        hashes = ("1" * 64, "2" * 64)
        self.assertEqual(
            checkpoint_io.select_rank_checkpoint_sha256(
                ",".join(hashes), rank=1, world_size=2, required=True
            ),
            hashes[1],
        )
        with self.assertRaisesRegex(RuntimeError, "requires per-rank"):
            checkpoint_io.select_rank_checkpoint_sha256(
                "NONE", rank=0, world_size=2, required=True
            )
        with self.assertRaisesRegex(RuntimeError, "count does not match"):
            checkpoint_io.select_rank_checkpoint_sha256(
                hashes[0], rank=0, world_size=2, required=True
            )
        with self.assertRaisesRegex(RuntimeError, "list is malformed"):
            checkpoint_io.select_rank_checkpoint_sha256(
                f"{hashes[0]},BAD", rank=0, world_size=2, required=True
            )

    def test_trainer_requires_digest_for_strict_production_resume(self):
        source = PPO_TRAINER.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("RUN_PROFILE") == "production_256x300"', source)
        self.assertIn("require_expected_sha256=production_resume", source)
        self.assertIn("load_verified_torch_checkpoint", source)


class PolicyIntegrityGateTest(unittest.TestCase):
    def test_finite_changed_policy_with_identical_schema_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial_path = root / "initial.pt"
            final_path = root / "final.pt"
            initial_hash = _write_checkpoint(initial_path, step=0, values=(1.0, 2.0))
            final_hash = _write_checkpoint(final_path, step=500, values=(1.0, 3.0))
            initial = inspect(FakeTorch, initial_path, 0, initial_hash)
            final = inspect(FakeTorch, final_path, 500, final_hash)
            validate_change(initial, final)

    def test_nan_and_infinity_are_rejected(self):
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as temporary:
                checkpoint = Path(temporary) / "checkpoint.pt"
                digest = _write_checkpoint(
                    checkpoint, step=500, values=(1.0, nonfinite)
                )
                with self.assertRaisesRegex(GateError, "contains NaN or infinity"):
                    inspect(FakeTorch, checkpoint, 500, digest)

    def test_parameter_name_dtype_and_shape_schema_mismatches_are_rejected(self):
        variants = (
            {"name": "different"},
            {"dtype": "float64"},
            {"shape": (1, 2)},
        )
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                initial_path = root / "initial.pt"
                final_path = root / "final.pt"
                initial_hash = _write_checkpoint(initial_path, step=0)
                final_hash = _write_checkpoint(final_path, step=500, **variant)
                initial = inspect(FakeTorch, initial_path, 0, initial_hash)
                final = inspect(FakeTorch, final_path, 500, final_hash)
                with self.assertRaisesRegex(GateError, "schemas do not match exactly"):
                    validate_change(initial, final)

    def test_expected_checkpoint_digest_is_enforced_by_policy_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            _write_checkpoint(checkpoint, step=0)
            with self.assertRaisesRegex(GateError, "mismatch at actual load"):
                inspect(FakeTorch, checkpoint, 0, "0" * 64)


class ResumableStateValidationTest(unittest.TestCase):
    def test_step_zero_optimizer_with_empty_state_is_resumable(self):
        _validate_resumable_state(
            {
                "optimizer_state_dict": {
                    "state": {},
                    "param_groups": [{"params": []}],
                },
                "lr_scheduler_state_dict": {"last_epoch": 0},
                "env_state_dict": {},
            }
        )

    def test_post_update_optimizer_with_empty_state_is_not_resumable(self):
        with self.assertRaisesRegex(GateError, "no state entries after step zero"):
            _validate_resumable_state(
                {
                    "optimizer_state_dict": {
                        "state": {},
                        "param_groups": [{"params": []}],
                    },
                    "lr_scheduler_state_dict": {"last_epoch": 1},
                    "env_state_dict": {},
                },
                checkpoint_step=1,
            )

    def test_malformed_optimizer_scheduler_and_environment_are_rejected(self):
        valid = {
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [{"params": []}],
            },
            "lr_scheduler_state_dict": {"last_epoch": 0},
            "env_state_dict": {},
        }
        variants = (
            {"optimizer_state_dict": {}},
            {"optimizer_state_dict": {"state": {}, "param_groups": []}},
            {"lr_scheduler_state_dict": {}},
            {"env_state_dict": None},
        )
        for replacement in variants:
            with self.subTest(replacement=replacement):
                payload = dict(valid)
                payload.update(replacement)
                with self.assertRaises(GateError):
                    _validate_resumable_state(payload)


if __name__ == "__main__":
    unittest.main()
