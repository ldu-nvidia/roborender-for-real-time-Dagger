"""Narrow NumPy-2 learner-checkpoint bridge for the native H200 GR00T actor.

The accepted nested GR00T checkout is intentionally immutable.  Its generic
checkpoint reader uses the Isaac runtime's normal ``torch.load`` path, which
cannot deserialize NumPy-2 RNG bookkeeping on the NumPy-1.26 Isaac image.  An
actor warm-start needs the whole document because GR00T verifies the Homie
controller provenance after loading, so extracting only model weights is not
correct.

This module patches that reader in-process *before* GR00T imports the PPO
trainer.  It leaves every ordinary checkpoint on the native reader and
activates only after that reader raises the precise NumPy module-rename error
for the exact hash-pinned policy selected by the round contract.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from round_dagger.checkpoint_compat import (
    CheckpointCompatibilityError,
    load_hash_pinned_restart_document,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NUMPY2_MODULE_ERRORS = frozenset(("numpy._core", "numpy._core.multiarray"))
_PATCH_MARKER = "_roborender_actor_numpy2_warmstart_patch"


class ActorCheckpointCompatibilityError(RuntimeError):
    """The actor warm-start checkpoint cannot use the restricted bridge."""


def _absolute_regular_path(value: str | os.PathLike[str], label: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ActorCheckpointCompatibilityError(
            f"{label} must be an absolute regular non-symlink file"
        )
    return os.fspath(path)


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ActorCheckpointCompatibilityError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def install_actor_warmstart_checkpoint_compat(
    *, policy_checkpoint: str | os.PathLike[str], policy_sha256: str
) -> None:
    """Install the exact-policy NumPy bridge into nested GR00T checkpoint I/O.

    The patch never changes the accepted nested source tree.  A candidate is
    still loaded through GR00T's native reader whenever possible; the
    restricted fallback is limited to the exact policy path and SHA selected
    by the immutable actor contract.
    """

    checkpoint = _absolute_regular_path(policy_checkpoint, "actor policy checkpoint")
    expected_sha256 = _sha256(policy_sha256, "actor policy checkpoint SHA-256")

    try:
        from gr00t.rl.trl.trainer import checkpoint_io
    except Exception as exc:  # pragma: no cover - exercised in the Isaac image
        raise ActorCheckpointCompatibilityError(
            "cannot import GR00T checkpoint I/O before actor startup"
        ) from exc

    existing = getattr(checkpoint_io, _PATCH_MARKER, None)
    if existing is not None:
        if existing != (checkpoint, expected_sha256):
            raise ActorCheckpointCompatibilityError(
                "actor checkpoint compatibility bridge was already bound to a different policy"
            )
        return

    native_loader: Callable[..., tuple[Any, str]] = checkpoint_io.load_verified_torch_checkpoint

    def _load_with_numpy2_fallback(
        torch_module: Any,
        checkpoint_path: str | os.PathLike[str],
        *,
        map_location: Any,
        expected_sha256: str | None = None,
        require_expected_sha256: bool = False,
        weights_only: bool = False,
    ) -> tuple[Any, str]:
        requested = os.fspath(Path(checkpoint_path).expanduser())
        selected_policy = requested == checkpoint
        supplied_sha = None if expected_sha256 in (None, "", "NONE") else expected_sha256
        if selected_policy:
            if supplied_sha != expected_sha256_bound:
                raise ActorCheckpointCompatibilityError(
                    "actor warm-start must provide the contract-bound policy SHA-256"
                )
            if weights_only:
                raise ActorCheckpointCompatibilityError(
                    "actor NumPy compatibility bridge refuses weights-only policy loading"
                )
        try:
            return native_loader(
                torch_module,
                checkpoint_path,
                map_location=map_location,
                expected_sha256=expected_sha256,
                require_expected_sha256=require_expected_sha256,
                weights_only=weights_only,
            )
        except ModuleNotFoundError as exc:
            if not selected_policy or exc.name not in _NUMPY2_MODULE_ERRORS:
                raise
        try:
            document = load_hash_pinned_restart_document(
                checkpoint,
                expected_sha256=expected_sha256_bound,
                map_location=map_location,
            )
        except CheckpointCompatibilityError as exc:
            raise ActorCheckpointCompatibilityError(
                "restricted actor warm-start fallback failed"
            ) from exc
        return document, expected_sha256_bound

    # Bind immutable values into the closure, then expose a private marker for
    # idempotence checks in a single interpreter.
    expected_sha256_bound = expected_sha256
    checkpoint_io.load_verified_torch_checkpoint = _load_with_numpy2_fallback
    setattr(checkpoint_io, _PATCH_MARKER, (checkpoint, expected_sha256))


__all__ = [
    "ActorCheckpointCompatibilityError",
    "install_actor_warmstart_checkpoint_compat",
]
