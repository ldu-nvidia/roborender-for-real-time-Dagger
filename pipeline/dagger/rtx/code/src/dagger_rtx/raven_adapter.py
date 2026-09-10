from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .backend import Capture, TransitionResult
from .contracts import StepKey


@dataclass
class RavenHooks:
    """Callbacks bound to one live, already initialized native Raven trainer.

    Integration code inside the proven RTX Isaac container supplies these
    callbacks. `capture` must call the retained trainer/camera/teacher methods;
    `step_once` must be the sole callsite that reaches `env.step`.
    """

    bind_episode: Callable[[StepKey], Mapping[str, Any]]
    capture: Callable[[StepKey], Capture]
    step_once: Callable[[np.ndarray], TransitionResult]
