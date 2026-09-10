from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .backend import BackendInvariantError, Capture, TransitionResult
from .contracts import PINNED_RAVEN_COMMIT, PINNED_RAVEN_ROOT, RAVEN_VIEW_ORDER, StepKey

PINNED_GR00T_COMMIT = PINNED_RAVEN_COMMIT
PINNED_RAVEN_PATH = Path(PINNED_RAVEN_ROOT)
PINNED_TRAINER = PINNED_RAVEN_PATH / "pipeline/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
PINNED_IMAGE_DAGGER_CONFIG = (
    PINNED_RAVEN_PATH
    / "pipeline/GR00T-VisualSim2Real/gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger_3view.yaml"
)


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


class PinnedRavenAdapter:
    def __init__(self, hooks: RavenHooks):
        self.verify_hook_signatures(hooks)
        self._hooks = hooks
        self._episode: str | None = None
        self._next_step = 0
        self._step_calls = 0

    @staticmethod
    def verify_hook_signatures(hooks: RavenHooks) -> None:
        for name in ("bind_episode", "capture", "step_once"):
            fn = getattr(hooks, name)
            if not callable(fn):
                raise BackendInvariantError(f"Raven hook {name} is not callable")
            if inspect.iscoroutinefunction(fn):
                raise BackendInvariantError(f"Raven hook {name} must be synchronous in the Isaac owner")

    def bind_episode(self, *, key: StepKey) -> Mapping[str, Any]:
        if key.episode_step_index != 0:
            raise BackendInvariantError("an episode binding must begin at episode step zero")
        # The native Raven hook invokes its pinned `_id_sample_anchor`; the
        # wrapper neither resamples nor emulates the 50% assignment.
        result = dict(self._hooks.bind_episode(key))
        self._episode = key.episode_id
        self._next_step = 0
        return result

    def capture_current(self, *, key: StepKey) -> Capture:
        if key.episode_id != self._episode or key.episode_step_index != self._next_step:
            raise BackendInvariantError("Raven capture is out of episode/step order")
        capture = self._hooks.capture(key)
        if capture.key != key:
            raise BackendInvariantError("Raven capture returned the wrong step key")
        if tuple(capture.metadata.get("view_order", ())) != RAVEN_VIEW_ORDER:
            raise BackendInvariantError("Raven capture must declare canonical conditioning view order")
        return capture

    def step_once(self, action: np.ndarray) -> TransitionResult:
        self._step_calls += 1
        result = self._hooks.step_once(action)
        self._next_step += 1
        return result

    @property
    def observed_step_calls(self) -> int:
        return self._step_calls
