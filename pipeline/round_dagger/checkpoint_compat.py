"""Fail-closed loading for the hash-pinned RoboRender bootstrap policy.

The pinned bootstrap checkpoint contains a valid ``policy_state_dict`` plus
trainer bookkeeping objects from TRL, Transformers, and Accelerate.  The
camera-free learner intentionally does not install those training stacks.  A
normal ``torch.load(..., weights_only=False)`` therefore either fails on the
missing imports or, if used on an untrusted file, permits arbitrary pickle
globals.

This module reads one regular, non-symlink file into memory, verifies the exact
bytes against a caller-provided SHA-256, then uses a restricted unpickler.  It
allows only the small set of Torch tensor reconstruction primitives present in
the pinned artifact and maps the exact known auxiliary classes to an inert
object whose state is discarded.  Unknown globals fail closed.  Only the
``policy_state_dict`` escapes this module; discarded trainer state is never
returned to the learner.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import pickletools
import re
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# These globals are the complete non-primitive tensor reconstruction surface
# observed across the immutable bootstrap and learner-published candidate
# artifacts.  ``torch.serialization`` wraps our Unpickler and handles Storage
# classes before delegating, so a separate pickle-opcode preflight below is what
# constrains its wrapper to these exact storage types.
_TENSOR_GLOBALS = frozenset(
    {
        ("collections", "OrderedDict"),
        ("collections", "deque"),
        ("torch", "FloatStorage"),
        ("torch", "ByteStorage"),
        ("torch", "LongStorage"),
        ("torch", "device"),
        ("torch._utils", "_rebuild_tensor_v2"),
    }
)

# These values are auxiliary trainer/configuration provenance.  They are not
# consulted when reconstructing, identifying, or training the student policy.
# Do not broaden this list to make an arbitrary checkpoint load: a new global
# requires an explicit code review and a new test fixture.
_DISCARDED_AUXILIARY_GLOBALS = frozenset(
    {
        ("accelerate.state", "PartialState"),
        ("accelerate.utils.dataclasses", "DistributedType"),
        # A learner-published candidate intentionally retains optimizer and
        # per-rank RNG state so it is also a restart artifact.  Initial-policy
        # loading needs none of the NumPy RNG value, so reconstruct its exact
        # reviewed pickle surface as discarded auxiliary state.  Torch tensor
        # values in the optimizer/RNG payload remain constrained by the tensor
        # allowlist above.
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        # NumPy 1.x and 2.x serialize the same ndarray reconstruction helper
        # under these two module aliases respectively.  Both are emitted by
        # supported producer/consumer NumPy runtimes and are discarded
        # identically.
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("transformers.trainer_pt_utils", "AcceleratorConfig"),
        ("transformers.trainer_utils", "HubStrategy"),
        ("transformers.trainer_utils", "IntervalStrategy"),
        ("transformers.trainer_utils", "SaveStrategy"),
        ("transformers.trainer_utils", "SchedulerType"),
        ("transformers.training_args", "OptimizerNames"),
        ("trl.trainer.ppo_config", "PPOConfig"),
        ("trl.trainer.utils", "OnlineTrainerState"),
    }
)

_ALLOWED_PICKLE_GLOBALS = _TENSOR_GLOBALS | _DISCARDED_AUXILIARY_GLOBALS
# A learner restart must retain optimizer and RNG state rather than discarding
# it.  These are the complete additional globals observed in learner-published
# restart/candidate artifacts.  Unlike the bootstrap loader above, this loader
# resolves the exact NumPy reconstruction primitives after the pickle opcode
# surface has been preflighted.  Arbitrary application globals remain denied.
_RESTART_STATE_GLOBALS = frozenset(
    {
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)
_ALLOWED_RESTART_GLOBALS = _TENSOR_GLOBALS | _RESTART_STATE_GLOBALS
_FORBIDDEN_DYNAMIC_GLOBAL_OPCODES = frozenset(
    {"EXT1", "EXT2", "EXT4", "INST", "STACK_GLOBAL"}
)
_MAX_PICKLE_RECORD_BYTES = 16 * 1024 * 1024


class CheckpointCompatibilityError(RuntimeError):
    """The pinned checkpoint cannot be reduced to a trusted policy state."""


class _DiscardedAuxiliary:
    """Inert target for reviewed trainer-only pickle globals.

    Both ``REDUCE`` (used by enum values) and ``NEWOBJ``/``BUILD`` (used by
    dataclass-like objects) can construct this type.  Constructor arguments and
    serialized state are deliberately ignored.
    """

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> "_DiscardedAuxiliary":
        del args, kwargs
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        del state


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        global_name = (module, name)
        if global_name in _DISCARDED_AUXILIARY_GLOBALS:
            return _DiscardedAuxiliary
        if global_name in _TENSOR_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"checkpoint global {module}.{name} is not allowlisted"
        )


class _RestrictedCheckpointPickleModule:
    """Minimal pickle-module interface consumed by ``torch.load``."""

    __name__ = "round_dagger_restricted_checkpoint_pickle"
    Unpickler = _RestrictedCheckpointUnpickler

    @staticmethod
    def load(file: Any, **kwargs: Any) -> Any:
        return _RestrictedCheckpointUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(payload: bytes, **kwargs: Any) -> Any:
        return _RestrictedCheckpointUnpickler(io.BytesIO(payload), **kwargs).load()


class _RestrictedRestartUnpickler(pickle.Unpickler):
    """Resolve only tensor and NumPy RNG primitives used by learner restarts."""

    def find_class(self, module: str, name: str) -> Any:
        # NumPy 2 writes this helper under ``numpy._core`` while the validated
        # The native H200 Isaac runtime intentionally carries NumPy 1.26, where
        # the exact same ABI helper lives under ``numpy.core``.  Resolve only
        # this one reviewed reconstruction primitive; do not install a
        # process-global module alias or broaden the checkpoint pickle surface.
        if (module, name) == ("numpy._core.multiarray", "_reconstruct"):
            from numpy.core.multiarray import _reconstruct

            return _reconstruct
        if (module, name) in _ALLOWED_RESTART_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"restart checkpoint global {module}.{name} is not allowlisted"
        )


class _RestrictedRestartPickleModule:
    """Restricted pickle-module interface for full learner restart state."""

    __name__ = "round_dagger_restricted_restart_pickle"
    Unpickler = _RestrictedRestartUnpickler

    @staticmethod
    def load(file: Any, **kwargs: Any) -> Any:
        return _RestrictedRestartUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(payload: bytes, **kwargs: Any) -> Any:
        return _RestrictedRestartUnpickler(io.BytesIO(payload), **kwargs).load()


def _validated_sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointCompatibilityError(
            "expected checkpoint SHA-256 must be 64 lowercase hexadecimal characters"
        )
    return value


def _read_regular_file(checkpoint_path: str | os.PathLike[str]) -> tuple[bytes, str]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointCompatibilityError(
            "policy checkpoint must be a readable regular non-symlink file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckpointCompatibilityError(
                "policy checkpoint must be a regular non-symlink file"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest()


def _read_pinned_regular_file(
    checkpoint_path: str | os.PathLike[str], expected_sha256: str
) -> bytes:
    expected = _validated_sha256(expected_sha256)
    payload, observed = _read_regular_file(checkpoint_path)
    if observed != expected:
        raise CheckpointCompatibilityError("policy checkpoint SHA-256 mismatch")
    return payload


def _preflight_checkpoint_pickle(
    payload: bytes, *, allowed_globals: frozenset[tuple[str, str]]
) -> None:
    """Verify every callable/global reference before invoking Torch.

    PyTorch inserts its own Unpickler subclass and recognizes Storage globals
    before it calls our ``find_class``.  Inspecting the embedded pickle first
    ensures even that wrapper cannot silently broaden this artifact contract.
    Protocol mechanisms that resolve globals without a literal ``GLOBAL``
    opcode are rejected outright.
    """

    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            records = [
                item
                for item in archive.infolist()
                if item.filename == "data.pkl" or item.filename.endswith("/data.pkl")
            ]
            if len(records) != 1:
                raise CheckpointCompatibilityError(
                    "policy checkpoint must contain exactly one data.pkl record"
                )
            record = records[0]
            if record.file_size > _MAX_PICKLE_RECORD_BYTES:
                raise CheckpointCompatibilityError(
                    "policy checkpoint data.pkl exceeds the reviewed size limit"
                )
            pickle_payload = archive.read(record)
        for opcode, argument, _ in pickletools.genops(pickle_payload):
            if opcode.name in _FORBIDDEN_DYNAMIC_GLOBAL_OPCODES:
                raise CheckpointCompatibilityError(
                    f"policy checkpoint uses forbidden {opcode.name} opcode"
                )
            if opcode.name != "GLOBAL":
                continue
            if not isinstance(argument, str) or " " not in argument:
                raise CheckpointCompatibilityError(
                    "policy checkpoint contains a malformed GLOBAL opcode"
                )
            module, name = argument.split(" ", 1)
            if (module, name) not in allowed_globals:
                raise CheckpointCompatibilityError(
                    f"checkpoint global {module}.{name} is not allowlisted"
                )
    except CheckpointCompatibilityError:
        raise
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"cannot preflight policy checkpoint pickle: {type(exc).__name__}: {exc}"
        ) from exc


def load_hash_pinned_policy_state_dict(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    map_location: str = "cpu",
) -> Mapping[str, Any]:
    """Return only ``policy_state_dict`` from exact, reviewed checkpoint bytes.

    The original file is opened read-only and is never rewritten or migrated.
    Hash verification covers the exact in-memory bytes subsequently passed to
    Torch, avoiding a path-based verify/load race.
    """

    payload = _read_pinned_regular_file(checkpoint_path, expected_sha256)
    # The restricted pickle module's ``load`` method is also safe, but the
    # production artifact contract is Torch's modern ZIP serialization.  This
    # rejects legacy formats and prevents an accidental expansion of the
    # compatibility surface.
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise CheckpointCompatibilityError(
            "policy checkpoint must use Torch ZIP serialization"
        )
    _preflight_checkpoint_pickle(payload, allowed_globals=_ALLOWED_PICKLE_GLOBALS)
    try:
        import torch

        document = torch.load(
            io.BytesIO(payload),
            map_location=map_location,
            pickle_module=_RestrictedCheckpointPickleModule,
            weights_only=False,
        )
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"restricted policy checkpoint load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise CheckpointCompatibilityError(
            "policy checkpoint must contain a top-level mapping"
        )
    state = document.get("policy_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise CheckpointCompatibilityError(
            "policy checkpoint lacks a string-keyed policy_state_dict"
        )
    return state


def load_restricted_restart_document(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: str = "cpu",
) -> Mapping[str, Any]:
    """Load full learner restart state without permitting arbitrary globals.

    Restart files are mutable recovery frontiers and therefore cannot be
    caller-hash-pinned before every update.  The file is nevertheless opened
    once with ``O_NOFOLLOW``, read into memory, and pickle-preflighted before
    Torch sees it.  Only the exact tensor and NumPy RNG reconstruction globals
    emitted by the learner are resolved; a foreign application class fails
    before deserialization.
    """

    payload, _observed_sha256 = _read_regular_file(checkpoint_path)
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise CheckpointCompatibilityError(
            "restart checkpoint must use Torch ZIP serialization"
        )
    _preflight_checkpoint_pickle(
        payload, allowed_globals=_ALLOWED_RESTART_GLOBALS
    )
    try:
        import torch

        document = torch.load(
            io.BytesIO(payload),
            map_location=map_location,
            pickle_module=_RestrictedRestartPickleModule,
            weights_only=False,
        )
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"restricted restart checkpoint load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise CheckpointCompatibilityError(
            "restart checkpoint must contain a top-level mapping"
        )
    return document


def load_hash_pinned_restart_document(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    map_location: Any = "cpu",
) -> Mapping[str, Any]:
    """Load one exact full learner checkpoint through the restart allowlist.

    This is the actor warm-start counterpart to
    :func:`load_hash_pinned_policy_state_dict`.  It retains the small amount
    of reviewed provenance (notably ``homie_controller_sha256``) consumed by
    GR00T's post-load identity gate, while safely discarding neither model
    weights nor required controller provenance.  The exact bytes are pinned
    before the restricted NumPy-1/NumPy-2 compatible unpickler sees them.
    """

    payload = _read_pinned_regular_file(checkpoint_path, expected_sha256)
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise CheckpointCompatibilityError(
            "restart checkpoint must use Torch ZIP serialization"
        )
    _preflight_checkpoint_pickle(payload, allowed_globals=_ALLOWED_RESTART_GLOBALS)
    try:
        import torch

        document = torch.load(
            io.BytesIO(payload),
            map_location=map_location,
            pickle_module=_RestrictedRestartPickleModule,
            weights_only=False,
        )
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"hash-pinned restart checkpoint load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise CheckpointCompatibilityError(
            "restart checkpoint must contain a top-level mapping"
        )
    state = document.get("policy_state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise CheckpointCompatibilityError(
            "restart checkpoint lacks a string-keyed policy_state_dict"
        )
    return document


__all__ = [
    "CheckpointCompatibilityError",
    "load_hash_pinned_policy_state_dict",
    "load_hash_pinned_restart_document",
    "load_restricted_restart_document",
]
