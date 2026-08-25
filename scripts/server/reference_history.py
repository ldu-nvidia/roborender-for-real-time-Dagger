#!/usr/bin/env python3
"""Bounded, thread-safe RGB reference history for stateful Image3F streams.

The fine-tuned Image3F model was trained with a lag-256 reference policy:

* steps ``0..255`` use the episode's frame-0 RGB anchor;
* step ``t >= 256`` uses the generated RGB from ``t - 256``.

Keeping this history in the renderer avoids retaining roughly 55 GiB of raw
three-view RGB in a 256-environment trainer process.  This module deliberately
has no dependency on the Image3F runtime so it can be unit tested on a login
node.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


class ReferenceHistoryError(RuntimeError):
    """Raised when a stream violates the ordered lag-reference contract."""


@dataclass
class _StreamState:
    episode_id: str
    lag: int
    anchor: dict[str, np.ndarray] = field(default_factory=dict)
    next_step: int = 0
    pending_step: int | None = None
    pending_request_id: str | None = None
    pending_reference_source: str | None = None
    frames: deque[dict[str, np.ndarray]] = field(default_factory=deque)
    last_request_id: str | None = None
    last_generated: dict[str, np.ndarray] | None = None
    last_reference_source: str | None = None
    last_access: float = field(default_factory=time.monotonic)


class LaggedReferenceStore:
    """Maintain fixed-lag generated RGB histories for independent env streams.

    Calls for one stream must be ordered.  Different streams may prepare and
    commit concurrently.  Arrays are copied on commit because request buffers
    may be reused by the networking layer.
    """

    def __init__(self, max_streams: int = 1024):
        if max_streams < 1:
            raise ValueError("max_streams must be positive")
        self.max_streams = int(max_streams)
        self._streams: dict[str, _StreamState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _validate_images(images: Mapping[str, np.ndarray], label: str) -> None:
        if not images:
            raise ReferenceHistoryError(f"{label} must contain at least one view")
        for view, image in images.items():
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
                raise ReferenceHistoryError(
                    f"{label}[{view!r}] must be an HxWx3 numpy array; got "
                    f"{type(image).__name__} {getattr(image, 'shape', None)}"
                )

    def _evict_if_needed(self) -> None:
        while len(self._streams) >= self.max_streams:
            candidates = [
                (state.last_access, key)
                for key, state in self._streams.items()
                if state.pending_step is None
            ]
            if not candidates:
                raise ReferenceHistoryError(
                    f"all {self.max_streams} reference streams have in-flight requests"
                )
            _, key = min(candidates)
            del self._streams[key]

    def prepare(
        self,
        *,
        stream_id: str,
        episode_id: str | int,
        step: int,
        lag: int,
        anchor: Mapping[str, np.ndarray],
        request_id: str,
    ) -> tuple[dict[str, np.ndarray], str]:
        """Return the correct reference and reserve ``step`` for generation."""
        if not stream_id:
            raise ReferenceHistoryError("stream_id is required")
        step = int(step)
        lag = int(lag)
        episode_id = str(episode_id)
        if step < 0:
            raise ReferenceHistoryError(f"step must be non-negative, got {step}")
        if lag < 1:
            raise ReferenceHistoryError(f"lag must be positive, got {lag}")
        self._validate_images(anchor, "anchor")

        with self._lock:
            state = self._streams.get(stream_id)
            if state is not None and state.episode_id != episode_id and state.pending_step is not None:
                raise ReferenceHistoryError(
                    f"cannot reset stream {stream_id!r} while step {state.pending_step} is in flight"
                )
            if state is None or state.episode_id != episode_id:
                if step != 0:
                    raise ReferenceHistoryError(
                        f"new episode for {stream_id!r} must begin at step 0, got {step}"
                    )
                if state is None:
                    self._evict_if_needed()
                state = _StreamState(
                    episode_id=episode_id,
                    lag=lag,
                    anchor={
                        view: np.ascontiguousarray(image).copy()
                        for view, image in anchor.items()
                    },
                )
                state.frames = deque(maxlen=lag)
                self._streams[stream_id] = state
            if state.lag != lag:
                raise ReferenceHistoryError(
                    f"lag changed within stream {stream_id!r}: {state.lag} -> {lag}"
                )
            if state.pending_step is not None:
                raise ReferenceHistoryError(
                    f"stream {stream_id!r} already has step {state.pending_step} in flight"
                )
            if step != state.next_step:
                raise ReferenceHistoryError(
                    f"out-of-order stream {stream_id!r}: expected step "
                    f"{state.next_step}, got {step}"
                )
            if set(anchor) != set(state.anchor) or any(
                anchor[view].shape != state.anchor[view].shape
                or anchor[view].dtype != state.anchor[view].dtype
                or not np.array_equal(anchor[view], state.anchor[view])
                for view in state.anchor
            ):
                raise ReferenceHistoryError(
                    f"episode anchor changed within stream {stream_id!r} episode {episode_id!r}"
                )
            if step < lag:
                reference = state.anchor
                source = "anchor"
            else:
                if len(state.frames) != lag:
                    raise ReferenceHistoryError(
                        f"stream {stream_id!r} has {len(state.frames)} frames, needs {lag}"
                    )
                reference = state.frames[0]
                source = f"generated_t-{lag}"
            state.pending_step = step
            state.pending_request_id = str(request_id)
            state.pending_reference_source = source
            state.last_access = time.monotonic()
            return reference, source

    def replay_with_source(
        self,
        *,
        stream_id: str,
        episode_id: str | int,
        step: int,
        request_id: str,
    ) -> tuple[dict[str, np.ndarray], str] | None:
        """Return a committed retry together with its effective reference source."""
        with self._lock:
            state = self._streams.get(stream_id)
            if (
                state is not None
                and state.episode_id == str(episode_id)
                and state.next_step == int(step) + 1
                and state.last_request_id == str(request_id)
                and state.last_generated is not None
                and state.last_reference_source is not None
            ):
                state.last_access = time.monotonic()
                return state.last_generated, state.last_reference_source
            return None

    def replay(
        self,
        *,
        stream_id: str,
        episode_id: str | int,
        step: int,
        request_id: str,
    ) -> dict[str, np.ndarray] | None:
        """Return the last committed output for an idempotent network retry."""
        replay = self.replay_with_source(
            stream_id=stream_id,
            episode_id=episode_id,
            step=step,
            request_id=request_id,
        )
        return replay[0] if replay is not None else None

    def commit(
        self,
        *,
        stream_id: str,
        episode_id: str | int,
        step: int,
        generated: Mapping[str, np.ndarray],
        request_id: str,
    ) -> None:
        """Commit a successful generation and advance the stream by one step."""
        self._validate_images(generated, "generated")
        episode_id = str(episode_id)
        step = int(step)
        with self._lock:
            state = self._streams.get(stream_id)
            if state is None or state.episode_id != episode_id:
                raise ReferenceHistoryError(f"unknown stream/episode {stream_id!r}/{episode_id!r}")
            if state.pending_step != step or state.pending_request_id != str(request_id):
                raise ReferenceHistoryError(
                    f"commit mismatch for {stream_id!r}: pending "
                    f"{state.pending_step}/{state.pending_request_id}, got {step}/{request_id}"
                )
            committed = {
                view: np.ascontiguousarray(image).copy() for view, image in generated.items()
            }
            state.frames.append(committed)
            state.last_generated = committed
            state.last_request_id = str(request_id)
            state.last_reference_source = state.pending_reference_source
            state.pending_step = None
            state.pending_request_id = None
            state.pending_reference_source = None
            state.next_step = step + 1
            state.last_access = time.monotonic()

    def abort(self, *, stream_id: str, episode_id: str | int, step: int) -> None:
        """Release a failed request so the same step can be retried."""
        with self._lock:
            state = self._streams.get(stream_id)
            if (
                state is not None
                and state.episode_id == str(episode_id)
                and state.pending_step == int(step)
            ):
                state.pending_step = None
                state.pending_request_id = None
                state.pending_reference_source = None
                state.last_access = time.monotonic()

    def reset(self, stream_id: str) -> bool:
        """Drop one stream.  Returns whether it existed."""
        with self._lock:
            state = self._streams.get(stream_id)
            if state is not None and state.pending_step is not None:
                raise ReferenceHistoryError(
                    f"cannot reset stream {stream_id!r} while step {state.pending_step} is in flight"
                )
            return self._streams.pop(stream_id, None) is not None

    def stats(self) -> dict[str, int]:
        """Return lightweight health telemetry without exposing frame contents."""
        with self._lock:
            frames = sum(len(state.frames) for state in self._streams.values())
            frame_bytes = sum(
                image.nbytes
                for state in self._streams.values()
                for frame in state.frames
                for image in frame.values()
            )
            anchor_bytes = sum(
                image.nbytes
                for state in self._streams.values()
                for image in state.anchor.values()
            )
            in_flight = sum(state.pending_step is not None for state in self._streams.values())
            return {
                "streams": len(self._streams),
                "frames": frames,
                "bytes": frame_bytes + anchor_bytes,
                "in_flight": in_flight,
            }
