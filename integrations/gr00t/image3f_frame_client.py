#!/usr/bin/env python3
"""Production Image3F TCP client for the external GR00T DAgger trainer.

This file is installed into ``GR00T-VisualSim2Real/gr00t/rl/utils`` by
``scripts/scaling/install_gr00t_integration.py``.  Its legacy ``render_frame``
call remains compatible with Raven's trainer while optionally adding:

* rank-aware endpoint sharding across a Slurm renderer fleet;
* stable stream/episode/step metadata for server-side lag-256 references;
* socket timeouts and per-request telemetry.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import pickle
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np


PROTOCOL_VERSION = 2
REFERENCE_PROVENANCE_SCHEMA = "image3f.reference_provenance.v1"


def _logical_actor_rank() -> int:
    """Return the actor rank, including independent non-DDP collectors."""

    raw = os.environ.get(
        "ROUND_DAGGER_ACTOR_RANK",
        os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")),
    )
    try:
        rank = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid actor rank {raw!r}") from exc
    if rank < 0:
        raise ValueError(f"invalid actor rank {rank}")
    return rank


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Image3F server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_msg(sock: socket.socket, obj: object) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_msg(sock: socket.socket) -> object:
    size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    return pickle.loads(_recv_exact(sock, size))  # noqa: S301 - trusted cluster protocol


def _encode_array(array: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(array)
    return {
        "__nd__": True,
        "dtype": str(array.dtype),
        "shape": tuple(array.shape),
        "data": array.tobytes(),
    }


def _decode_array(obj: object) -> np.ndarray:
    if not isinstance(obj, dict) or not obj.get("__nd__"):
        raise TypeError("malformed Image3F ndarray response")
    return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()


def _validate_output_views(views: dict[str, np.ndarray]) -> None:
    """Enforce the opt-in production output order and per-view tensor contract."""
    expected_raw = os.environ.get("IMAGE3F_EXPECT_OUTPUT_VIEWS")
    shape_raw = os.environ.get("IMAGE3F_EXPECT_OUTPUT_SHAPE")
    if expected_raw is None and shape_raw is None:
        return
    if not expected_raw or not shape_raw:
        raise RuntimeError(
            "IMAGE3F_EXPECT_OUTPUT_VIEWS and IMAGE3F_EXPECT_OUTPUT_SHAPE must be set together"
        )
    expected_views = tuple(value.strip() for value in expected_raw.split(","))
    try:
        expected_shape = tuple(int(value.strip()) for value in shape_raw.split(","))
    except ValueError as exc:
        raise RuntimeError("invalid IMAGE3F_EXPECT_OUTPUT_SHAPE") from exc
    if any(not value for value in expected_views) or len(expected_shape) != 3 or any(
        value <= 0 for value in expected_shape
    ):
        raise RuntimeError("invalid Image3F output contract")
    if tuple(views) != expected_views:
        raise RuntimeError(
            "Image3F response view order mismatch: "
            f"expected {expected_views}, got {tuple(views)}"
        )
    for view, image in views.items():
        if image.dtype != np.uint8 or tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"Image3F response shape/dtype mismatch for {view}: expected "
                f"uint8 {expected_shape}, got {image.dtype} {image.shape}"
            )


def _validate_reference_provenance(
    response: Mapping[str, object], *, step: int, lag: int
) -> dict[str, object]:
    """Fail closed unless the server attests the exact effective RGB reference."""

    step = int(step)
    lag = int(lag)
    expected_source = "anchor" if step < lag else f"generated_t-{lag}"
    expected = {
        "schema": REFERENCE_PROVENANCE_SCHEMA,
        "request_step": step,
        "configured_lag": lag,
        "kind": "anchor" if step < lag else "generated",
        "source_step": 0 if step < lag else step - lag,
    }
    observed = response.get("reference_provenance")
    if observed != expected:
        raise RuntimeError(
            "Image3F reference provenance mismatch: "
            f"expected {expected!r}, got {observed!r}"
        )
    if response.get("reference_source") != expected_source:
        raise RuntimeError(
            "Image3F reference source mismatch: "
            f"expected {expected_source!r}, got {response.get('reference_source')!r}"
        )
    if response.get("stream_step") != step:
        raise RuntimeError(
            "Image3F stream step mismatch: "
            f"expected {step}, got {response.get('stream_step')!r}"
        )
    return expected


def _validate_renderer_identity(response: Mapping[str, object]) -> None:
    """Bind every DAgger render response to the cohort's renderer identity."""

    expected = os.environ.get("ROUND_DAGGER_RENDERER_SHA256")
    if expected is None:
        return
    observed = response.get("renderer_identity_sha256")
    if observed != expected:
        raise RuntimeError(
            "Image3F renderer identity mismatch: "
            f"expected {expected!r}, got {observed!r}"
        )


def _read_endpoint(path: str | os.PathLike[str]) -> tuple[str, int]:
    fields = Path(path).read_text(encoding="utf-8").split()
    if len(fields) != 2:
        raise ValueError(f"malformed endpoint file {path}: expected 'host port'")
    return fields[0], int(fields[1])


class Image3FFrameClient:
    """One persistent socket and one ordered DAgger environment stream."""

    _counter = itertools.count()
    _counter_lock = threading.Lock()

    def __init__(
        self,
        host: str,
        port: int,
        *,
        stream_slot: int | None = None,
        timeout_s: float | None = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s or os.environ.get("IMAGE3F_REQUEST_TIMEOUT_S", 120))
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._stream_slot = int(stream_slot or 0)
        rank = _logical_actor_rank()
        run_id = os.environ.get("IMAGE3F_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
        self._stream_id = f"{run_id}:rank{rank}:env{self._stream_slot}"
        self._auto_episode = -1
        self._auto_step = 0
        self._anchor_object_ids: tuple[tuple[str, int], ...] | None = None
        self._anchor_signature: bytes | None = None
        self.last_metadata: dict[str, object] = {}

    @classmethod
    def from_endpoint_file(cls, endpoint_file: str | os.PathLike[str]):
        """Create a client, optionally remapping it across the production fleet.

        Raven's trainer creates clients rank-locally starting at environment 0.
        When ``IMAGE3F_ENDPOINT_DIR`` is set, this method offsets allocation by
        global rank so all ranks use the complete fleet instead of repeatedly
        concentrating traffic on its first endpoints.
        """
        with cls._counter_lock:
            local_slot = next(cls._counter)
        selected = Path(endpoint_file)
        endpoint_manifest = os.environ.get("IMAGE3F_ENDPOINT_MANIFEST")
        endpoint_dir = os.environ.get("IMAGE3F_ENDPOINT_DIR")
        local_envs = int(os.environ.get("IMAGE3F_LOCAL_ENVS", "0"))
        if endpoint_manifest or endpoint_dir:
            if endpoint_manifest:
                files = [
                    Path(line.strip())
                    for line in Path(endpoint_manifest).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                files = sorted(Path(endpoint_dir).glob("renderer_*.txt"))
            if not files:
                raise FileNotFoundError(
                    f"no selected Image3F endpoints in {endpoint_manifest or endpoint_dir}"
                )
            rank = _logical_actor_rank()
            if local_envs < 1:
                raise ValueError("IMAGE3F_LOCAL_ENVS must be set with IMAGE3F_ENDPOINT_DIR")
            global_slot = rank * local_envs + local_slot
            selected = files[global_slot % len(files)]
        else:
            global_slot = local_slot
        host, port = _read_endpoint(selected)
        client = cls(host, port, stream_slot=global_slot)
        client.endpoint_file = str(selected)
        return client

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._socket.settimeout(self.timeout_s)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._socket

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None

    @staticmethod
    def _signature(previous_rgb: Mapping[str, np.ndarray], prompt: str) -> bytes:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(prompt.encode("utf-8"))
        for view in sorted(previous_rgb):
            array = np.ascontiguousarray(previous_rgb[view])
            digest.update(view.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(memoryview(array))
        return digest.digest()

    def _auto_metadata(
        self, previous_rgb: Mapping[str, np.ndarray], prompt: str
    ) -> tuple[str, int, int]:
        # Raven's trainer retains each sampled anchor array for the episode.
        # Object identity lets us detect an episode reset even in the rare case
        # that the same anchor file is sampled twice consecutively.
        object_ids = tuple((view, id(previous_rgb[view])) for view in sorted(previous_rgb))
        signature = self._signature(previous_rgb, prompt)
        if object_ids != self._anchor_object_ids or signature != self._anchor_signature:
            self._anchor_object_ids = object_ids
            self._anchor_signature = signature
            self._auto_episode += 1
            self._auto_step = 0
        return self._stream_id, self._auto_episode, self._auto_step

    def _request(
        self, payload: dict[str, object], *, retries: int | None = None
    ) -> dict[str, object]:
        retries = int(
            retries if retries is not None else os.environ.get("IMAGE3F_REQUEST_RETRIES", "10")
        )
        response = None
        last_error = None
        for attempt in range(retries + 1):
            try:
                with self._lock:
                    sock = self._connect()
                    _send_msg(sock, payload)
                    response = _recv_msg(sock)
                if not isinstance(response, dict):
                    raise RuntimeError(
                        f"malformed Image3F response: {type(response).__name__}"
                    )
                if response.get("ok"):
                    return response
                if response.get("retryable") and attempt < retries:
                    time.sleep(min(2.0, 0.1 * (2**attempt)))
                    continue
                break
            except (ConnectionError, OSError, EOFError, pickle.UnpicklingError) as exc:
                last_error = exc
                with self._lock:
                    if self._socket is not None:
                        try:
                            self._socket.close()
                        finally:
                            self._socket = None
                if attempt >= retries:
                    raise
                time.sleep(min(2.0, 0.1 * (2**attempt)))
        if response is None:
            raise RuntimeError(f"Image3F request produced no response: {last_error}")
        if not response.get("ok"):
            raise RuntimeError(f"Image3F render failed: {response.get('error', 'unknown error')}")
        return response

    def ping(self) -> dict[str, object]:
        started = time.perf_counter()
        response = self._request({"cmd": "ping", "protocol_version": PROTOCOL_VERSION})
        response["client_rtt_ms"] = (time.perf_counter() - started) * 1000.0
        return response

    def render_frame(
        self,
        depth: Mapping[str, np.ndarray],
        mask: Mapping[str, np.ndarray],
        previous_rgb: Mapping[str, np.ndarray],
        *,
        prompt: str,
        seed: int = 42,
        stream_id: str | None = None,
        episode_id: str | int | None = None,
        step: int | None = None,
        reference_lag: int | None = None,
        trainer_step: int | None = None,
    ) -> dict[str, np.ndarray]:
        reference_lag = int(
            reference_lag
            if reference_lag is not None
            else os.environ.get("IMAGE3F_REFERENCE_LAG", "0")
        )
        using_auto_metadata = reference_lag > 0 and (
            stream_id is None or episode_id is None or step is None
        )
        if using_auto_metadata:
            stream_id, episode_id, step = self._auto_metadata(previous_rgb, prompt)
        payload: dict[str, object] = {
            "cmd": "render_frame",
            "protocol_version": PROTOCOL_VERSION,
            "depth": {view: _encode_array(image) for view, image in depth.items()},
            "mask": {view: _encode_array(image) for view, image in mask.items()},
            "prev": {view: _encode_array(image) for view, image in previous_rgb.items()},
            "prompt": str(prompt),
            "seed": int(seed),
        }
        if reference_lag > 0:
            payload.update(
                stream_id=str(stream_id),
                episode_id=str(episode_id),
                step=int(step),
                reference_lag=reference_lag,
                request_id=f"{stream_id}|{episode_id}|{step}",
            )
        if trainer_step is not None:
            if isinstance(trainer_step, bool) or int(trainer_step) < 0:
                raise ValueError("trainer_step must be a non-negative integer")
            payload["trainer_step"] = int(trainer_step)
        started = time.perf_counter()
        response = self._request(payload)
        _validate_renderer_identity(response)
        if reference_lag > 0:
            _validate_reference_provenance(
                response, step=int(step), lag=reference_lag
            )
        self.last_metadata = {
            key: value for key, value in response.items() if key not in ("ok", "views")
        }
        if reference_lag > 0:
            self.last_metadata["reference_provenance_valid"] = True
        self.last_metadata["client_wall_ms"] = (time.perf_counter() - started) * 1000.0
        if using_auto_metadata:
            self._auto_step += 1
        views = response.get("views")
        if not isinstance(views, dict):
            raise RuntimeError("Image3F response has no views")
        decoded = {view: _decode_array(image) for view, image in views.items()}
        _validate_output_views(decoded)
        return decoded

    def reset_stream(self) -> bool:
        response = self._request({"cmd": "reset_stream", "stream_id": self._stream_id})
        self._anchor_object_ids = None
        self._anchor_signature = None
        self._auto_step = 0
        return bool(response.get("removed"))

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()
