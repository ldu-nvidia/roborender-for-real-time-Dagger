from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from .contracts import StepKey


@dataclass(frozen=True)
class Capture:
    key: StepKey
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class TransitionResult:
    reward: float
    done: bool
    timeout: bool
    metadata: dict[str, Any]


class RavenBackend(Protocol):
    """Narrow adapter implemented around the already initialized Raven trainer.

    The concrete backend MUST call the pinned BC_render methods for synchronized
    camera capture, h5_v4, teacher+Homie construction, and exactly one env.step.
    This package deliberately does not reimplement those scientific semantics.
    """

    def bind_episode(self, *, key: StepKey) -> Mapping[str, Any]: ...

    def capture_current(self, *, key: StepKey) -> Capture: ...

    def step_once(self, action: np.ndarray) -> TransitionResult: ...


class BackendInvariantError(RuntimeError):
    pass
