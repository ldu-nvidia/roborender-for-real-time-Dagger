"""H200 Image3F + three-view student inference for the live DAgger boundary."""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from PIL import Image

from .contracts import VIEWS, file_sha256
from .step_contract import observation_input_digest


@dataclass(frozen=True)
class InferenceResult:
    policy_rgb: dict[str, np.ndarray]
    student_action_11: np.ndarray
    observation_digest: str
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Image3FRenderResult:
    views: dict[str, np.ndarray]
    runtime_ms: float
    worker_total_ms: float
    loopback_total_ms: float


class RenderStudentBackend(Protocol):
    def infer(
        self,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        checkpoint_sha256: str,
    ) -> InferenceResult: ...


class RenderStudentBackendPool:
    """Bounded dispatcher with one exclusive inference slot per GPU backend.

    CUDA models and Image3F sessions remain owned by their assigned backend.
    The public HTTP service may process unrelated actors concurrently, while
    this pool guarantees that a single GPU worker never receives overlapping
    inference calls.
    """

    def __init__(self, backends: list[RenderStudentBackend]):
        if not backends:
            raise ValueError("render/student backend pool cannot be empty")
        self.backends = tuple(backends)
        self._available = deque(range(len(self.backends)))
        self._condition = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.backends)

    def ready(self) -> tuple[bool, str]:
        failures = []
        for index, backend in enumerate(self.backends):
            check = getattr(backend, "ready", lambda: (True, "ready"))()
            if not check[0]:
                failures.append(f"worker {index}: {check[1]}")
        if failures:
            return False, "; ".join(failures)
        return True, f"{self.size} render/student workers ready"

    def infer(self, metadata, arrays, checkpoint_sha256) -> InferenceResult:
        wait_started = time.perf_counter()
        with self._condition:
            while not self._available:
                self._condition.wait()
            index = self._available.popleft()
        queue_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        try:
            result = self.backends[index].infer(metadata, arrays, checkpoint_sha256)
            if isinstance(result, InferenceResult):
                worker = self.backends[index]
                return replace(result, telemetry={
                    **result.telemetry,
                    "worker_slot": index,
                    "gpu_device": getattr(worker, "device_name", None),
                    "queue_wait_ms": queue_wait_ms,
                })
            return result
        finally:
            with self._condition:
                self._available.append(index)
                self._condition.notify()


class FakeRenderStudentBackend:
    """Deterministic test double which still exercises all tensor boundaries."""

    def __init__(self) -> None:
        self.calls = 0

    def ready(self) -> tuple[bool, str]:
        return True, "fake backend ready"

    def infer(self, metadata, arrays, checkpoint_sha256) -> InferenceResult:
        self.calls += 1
        if metadata["observation_source"] == "raytraced":
            views = {view: arrays[f"raytraced_rgb__{view}"].copy() for view in VIEWS}
        else:
            views = {}
            for index, view in enumerate(VIEWS):
                depth = arrays[f"depth_ctrl__{view}"]
                image = np.stack((depth, np.roll(depth, index + 1, axis=0), depth), axis=-1)
                views[view] = np.asarray(
                    Image.fromarray(image).resize((192, 108), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                ).astype(np.float32) / np.float32(255.0)
        actor = arrays["actor_obs"]
        action = np.tanh(actor[:11] + np.float32(int(checkpoint_sha256[:2], 16) / 255.0))
        action = np.ascontiguousarray(action, dtype=np.float32)
        return InferenceResult(
            policy_rgb=views,
            student_action_11=action,
            observation_digest=observation_input_digest(metadata, arrays),
            telemetry={
                "worker_slot": 0,
                "gpu_device": "fake",
                "queue_wait_ms": 0.0,
                "image3f_runtime_ms": 0.0,
                "image3f_worker_total_ms": 0.0,
                "image3f_total_ms": 0.0,
                "student_ms": 0.0,
                "backend_total_ms": 0.0,
            },
        )


class _Image3FLoopbackClient:
    """Private adapter to Raven's warm worker; never exposed outside loopback."""

    def __init__(self, host: str, port: int, timeout: float = 120.0):
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("the legacy Image3F protocol is restricted to loopback")
        self.host = host
        self.port = port
        self.timeout = timeout
        self._stream: socket.socket | None = None
        # One persistent framed stream cannot be multiplexed by unrelated
        # threads: a readiness ping and a render must never interleave their
        # send/receive frames or close the socket underneath one another.
        self._io_lock = threading.Lock()

    def _close_unlocked(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def _close(self) -> None:
        with self._io_lock:
            self._close_unlocked()

    def _connection(self) -> socket.socket:
        if self._stream is None:
            self._stream = socket.create_connection((self.host, self.port), self.timeout)
            self._stream.settimeout(self.timeout)
            self._stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._stream

    def _exchange(self, payload: bytes, *, max_response_bytes: int) -> Any:
        """Use the worker's persistent protocol; discard a broken stream safely."""

        with self._io_lock:
            stream = self._connection()
            try:
                stream.sendall(struct.pack("!Q", len(payload)) + payload)
                size = struct.unpack("!Q", self._recv_exact(stream, 8))[0]
                if size > max_response_bytes:
                    raise RuntimeError("Image3F worker response exceeds its size limit")
                return pickle.loads(self._recv_exact(stream, size))
            except BaseException:
                self._close_unlocked()
                raise

    @staticmethod
    def _encode(array: np.ndarray) -> dict[str, Any]:
        value = np.ascontiguousarray(array)
        return {
            "__nd__": True,
            "dtype": value.dtype.name,
            "shape": tuple(value.shape),
            "data": value.tobytes(order="C"),
        }

    @staticmethod
    def _recv_exact(stream: socket.socket, count: int) -> bytes:
        blocks = bytearray()
        while len(blocks) < count:
            block = stream.recv(count - len(blocks))
            if not block:
                raise RuntimeError("Image3F worker closed the connection")
            blocks.extend(block)
        return bytes(blocks)

    def render(self, *, prompt: str, depth: Mapping[str, np.ndarray],
               mask: Mapping[str, np.ndarray], previous: Mapping[str, np.ndarray],
               seed: int = 42) -> Image3FRenderResult:
        started = time.perf_counter()
        request = {
            "cmd": "render_frame", "prompt": prompt, "seed": seed,
            "depth": {view: self._encode(depth[view]) for view in VIEWS},
            "mask": {view: self._encode(mask[view]) for view in VIEWS},
            "prev": {view: self._encode(previous[view]) for view in VIEWS},
        }
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        response = self._exchange(payload, max_response_bytes=16 * 1024 * 1024)
        if not isinstance(response, dict) or not response.get("ok"):
            raise RuntimeError(f"Image3F worker failed: {response!r}")
        if set(response.get("views", {})) != set(VIEWS):
            raise RuntimeError("Image3F worker returned the wrong view set")
        result: dict[str, np.ndarray] = {}
        for view in VIEWS:
            envelope = response["views"][view]
            if not isinstance(envelope, dict) or not envelope.get("__nd__"):
                raise RuntimeError("invalid internal Image3F ndarray envelope")
            array = np.frombuffer(envelope["data"], dtype=np.dtype(envelope["dtype"])).reshape(
                envelope["shape"]
            ).copy()
            if array.shape != (240, 416, 3) or array.dtype != np.uint8:
                raise RuntimeError(f"Image3F returned invalid {view} tensor")
            result[view] = array
        runtime_ms = float(response.get("runtime_ms", response.get("ms", 0.0)))
        worker_total_ms = float(response.get("total_ms", response.get("ms", runtime_ms)))
        loopback_total_ms = (time.perf_counter() - started) * 1000.0
        for label, value in (
            ("runtime_ms", runtime_ms),
            ("worker_total_ms", worker_total_ms),
            ("loopback_total_ms", loopback_total_ms),
        ):
            if not np.isfinite(value) or value < 0:
                raise RuntimeError(f"Image3F returned invalid {label}")
        return Image3FRenderResult(result, runtime_ms, worker_total_ms, loopback_total_ms)

    def ping(self) -> bool:
        payload = pickle.dumps({"cmd": "ping"}, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            return self._exchange(payload, max_response_bytes=1024) == {"ok": True}
        except (OSError, RuntimeError, pickle.PickleError, struct.error):
            return False


class LiveRenderStudentBackend:
    """Exact Raven Image3F observation construction plus the selected BC student."""

    def __init__(
        self,
        *,
        checkpoint_resolver: Callable[[str], Path],
        policy_spec: Path,
        image3f_host: str = "127.0.0.1",
        image3f_port: int = 18500,
        device: str = "cuda:0",
        verify_checkpoint_hash: bool = True,
    ):
        self.resolve_checkpoint = checkpoint_resolver
        self.policy_spec_path = policy_spec.resolve()
        self.policy_spec: dict[str, Any] | None = None
        self.image3f = _Image3FLoopbackClient(image3f_host, image3f_port)
        self.device_name = device
        self.verify_checkpoint_hash = bool(verify_checkpoint_hash)
        self._model = None
        self._model_sha256: str | None = None
        self._mean = None
        self._std = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """A live backend owns exactly one exclusive GPU/session slot."""

        return 1

    def ready(self) -> tuple[bool, str]:
        if not self.policy_spec_path.is_file():
            return False, "policy specification is missing"
        try:
            self.policy_spec = json.loads(self.policy_spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"policy specification is invalid: {exc}"
        if not self.image3f.ping():
            return False, "loopback Image3F worker is unavailable"
        return True, "Image3F and student specification ready"

    def _load_policy(self, checkpoint_sha256: str):
        if self._model_sha256 == checkpoint_sha256 and self._model is not None:
            return self._model
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        checkpoint_path = self.resolve_checkpoint(checkpoint_sha256).resolve()
        # Registry/learner publication already verifies content once before a
        # digest becomes active. Re-hashing the same ~139 MB file in every one
        # of eight gateways at every iteration was redundant hot-path I/O.
        if self.verify_checkpoint_hash and file_sha256(checkpoint_path) != checkpoint_sha256:
            raise RuntimeError("student checkpoint resolver returned the wrong bytes")
        sidecar_path = (
            checkpoint_path.parent.parent / "inference_checkpoints"
            / f"{checkpoint_sha256}.pt"
        )
        if not self.verify_checkpoint_hash and sidecar_path.is_file():
            checkpoint = torch.load(
                sidecar_path, map_location="cpu", weights_only=False
            )
            if (
                checkpoint.get("schema")
                != "roborender.student_inference_sidecar.v1"
                or checkpoint.get("full_checkpoint_sha256") != checkpoint_sha256
            ):
                raise RuntimeError("student inference sidecar identity mismatch")
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        state = checkpoint["policy_state_dict"]
        if self._model is not None:
            # Architecture and policy_spec are immutable for a run. Retain the
            # resident module/CUDA allocations and replace only learned
            # weights when the checkpoint barrier advances.
            self._model.load_state_dict(state, strict=True)
            self._model.eval()
            self._model_sha256 = checkpoint_sha256
            return self._model
        if self.policy_spec is None:
            self.policy_spec = json.loads(self.policy_spec_path.read_text(encoding="utf-8"))
        spec = self.policy_spec
        env = OmegaConf.create({
            "robot": spec["env_config"]["robot"],
            "simulator": {"config": {"cameras": {
                "camera_resolutions": [108, 192], "camera_types": [{"rgb": True}]
            }}},
        })
        model = instantiate(
            OmegaConf.create(spec["actor_config"]),
            env_config=env,
            algo_config=OmegaConf.create(spec["algo_config"]),
            module_dim_dict=spec["module_dim_dict"],
            _recursive_=False,
        )
        model.load_state_dict(state, strict=True)
        model.eval().to(torch.device(self.device_name))
        self._mean = torch.tensor(
            spec["input_contract"]["rgb_mean"], dtype=torch.float32, device=self.device_name
        )
        self._std = torch.tensor(
            spec["input_contract"]["rgb_std"], dtype=torch.float32, device=self.device_name
        )
        self._model = model
        self._model_sha256 = checkpoint_sha256
        return model

    @staticmethod
    def _policy_size(views: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        # Preserve Raven's exact online path: concatenate the three 240x416
        # images first, convert to float [0,1], then resize the complete tiled
        # image to 324x192 with torch bilinear/align_corners=False.
        tiled = np.concatenate([views[view] for view in VIEWS], axis=0)
        tensor = torch.from_numpy(tiled.copy()).to(torch.float32).div_(255.0)
        tensor = functional.interpolate(
            tensor.permute(2, 0, 1).unsqueeze(0),
            size=(108 * len(VIEWS), 192),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).contiguous()
        resized = tensor.numpy()
        return {
            view: np.ascontiguousarray(resized[index * 108:(index + 1) * 108])
            for index, view in enumerate(VIEWS)
        }

    def infer(self, metadata, arrays, checkpoint_sha256) -> InferenceResult:
        import torch

        backend_started = time.perf_counter()
        with self._lock, torch.inference_mode():
            source = metadata["observation_source"]
            image3f_runtime_ms = None
            image3f_worker_total_ms = None
            image3f_total_ms = None
            if source == "raytraced":
                policy_rgb = {view: arrays[f"raytraced_rgb__{view}"].copy() for view in VIEWS}
            else:
                generated_result = self.image3f.render(
                    prompt=metadata["prompt"],
                    depth={view: arrays[f"depth_ctrl__{view}"] for view in VIEWS},
                    mask={view: arrays[f"rgb_robot__{view}"] for view in VIEWS},
                    previous={view: arrays[f"anchor_rgb__{view}"] for view in VIEWS},
                    seed=42,
                )
                policy_rgb = self._policy_size(generated_result.views)
                image3f_runtime_ms = generated_result.runtime_ms
                image3f_worker_total_ms = generated_result.worker_total_ms
                image3f_total_ms = generated_result.loopback_total_ms

            student_started = time.perf_counter()
            stacked = np.concatenate([policy_rgb[view] for view in VIEWS], axis=0)
            if stacked.shape != (324, 192, 3):
                raise RuntimeError("student policy image has the wrong three-view layout")
            # Raven's get_policy_view_stack contract is float32 RGB in [0,1].
            # Image3F uint8 output is converted to that same representation in
            # _policy_size, so both observation branches enter identical
            # normalization and can coexist in one learner window losslessly.
            vision = torch.from_numpy(stacked.copy()).to(torch.float32)
            # VisionActorObjPred's live path retains an explicit sequence axis:
            # [B,1,H,W,C] and [B,1,D]. This is the same bridge used by the
            # validated portable adapter; omitting it makes actor and vision
            # features differ in rank inside the GR00T model.
            model = self._load_policy(checkpoint_sha256)
            vision = vision.to(self.device_name)
            vision = ((vision - self._mean) / self._std).unsqueeze(0).unsqueeze(1)
            actor = torch.from_numpy(arrays["actor_obs"].copy()).to(
                self.device_name, torch.float32
            ).unsqueeze(0).unsqueeze(1)
            output = model({
                "vision_obs": vision, "actor_obs": actor
            })
            action = output["actions"].reshape(-1).detach().cpu().numpy().astype(np.float32)
            if action.shape != (11,) or not np.isfinite(action).all():
                raise RuntimeError("student returned an invalid action")
            digest = observation_input_digest(metadata, arrays)
            return InferenceResult(
                policy_rgb,
                np.ascontiguousarray(action),
                digest,
                telemetry={
                    "worker_slot": 0,
                    "gpu_device": self.device_name,
                    "queue_wait_ms": 0.0,
                    "image3f_runtime_ms": image3f_runtime_ms,
                    "image3f_worker_total_ms": image3f_worker_total_ms,
                    "image3f_total_ms": image3f_total_ms,
                    "student_ms": (time.perf_counter() - student_started) * 1000.0,
                    "backend_total_ms": (time.perf_counter() - backend_started) * 1000.0,
                },
            )


def content_addressed_checkpoint(root: Path, digest: str) -> Path:
    path = root.resolve() / "checkpoints" / f"{digest}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"registered checkpoint is missing: {digest}")
    return path
