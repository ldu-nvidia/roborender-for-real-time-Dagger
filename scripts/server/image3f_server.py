#!/usr/bin/env python3
"""Persistent warm Image3F server with a file-based request queue.

Watches QUEUE/inbox/*.json; each request:
  {"chunks": [npz paths], "prev_mode": "fixed|chain|black|zero",
   "tag_prefix": "x_", "anchor_index": 0, "depth_variant": "droid",
   "out": "/path", "seed": 42, "max_frames": 81,
   "init_prev_dir": "/path/to/teacher_fullres"}

Writes generations to <out>/<chunk>_gen_<tag>.npz, then moves the request to
QUEUE/done/<name>.json with a .result.json beside it (timing + status).
Session loads ONCE; each request costs only generation time (~175 ms/frame).
NOTE: "zero" mode patches the pipeline globally and cannot be undone in-process;
zero requests are rejected here — use a batch job for those.
"""
import io, json, os, pickle, secrets, shutil, socket, struct, sys, threading, time, traceback
from pathlib import Path

import numpy as np
from PIL import Image
from reference_history import LaggedReferenceStore
from renderer_identity import (
    REFERENCE_PROVENANCE_SCHEMA,
    canonical_json_sha256,
    describe_renderer_identity,
)
from runtime_identity import describe_runtime_identity
from shutdown_control import StopController

BUNDLE = Path(os.environ["IMAGE3F_ROOT"])
sys.path.insert(0, str(BUNDLE))
from runtime import Image3FRequest, Image3FSession  # noqa: E402

QUEUE = Path(os.environ["IMAGE3F_QUEUE"])
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
W, H = 416, 240
PROTOCOL_VERSION = 3
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DEFAULT_CLIENT_SOCKET_TIMEOUT_S = 180.0
DEFAULT_MAX_CONNECTIONS = 256
MAX_WIRE_NESTING = 32
MAX_WIRE_ITEMS = 1_000_000

DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0)),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
}


class Image3FWireError(RuntimeError):
    """Raised before unsafe or malformed TCP bytes reach the renderer."""


class _PrimitiveUnpickler(pickle.Unpickler):
    """Accept protocol primitives but never import a pickled global."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Image3F wire payload requested forbidden global {module}.{name}"
        )


def _max_message_bytes():
    raw = os.environ.get("IMAGE3F_MAX_MESSAGE_BYTES", str(DEFAULT_MAX_MESSAGE_BYTES))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise Image3FWireError("IMAGE3F_MAX_MESSAGE_BYTES must be an integer") from exc
    if value < 1024 or value > 1024 * 1024 * 1024:
        raise Image3FWireError(
            "IMAGE3F_MAX_MESSAGE_BYTES must be between 1 KiB and 1 GiB"
        )
    return value


def _client_socket_timeout_s():
    raw = os.environ.get(
        "IMAGE3F_CLIENT_SOCKET_TIMEOUT_S", str(DEFAULT_CLIENT_SOCKET_TIMEOUT_S)
    )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise Image3FWireError(
            "IMAGE3F_CLIENT_SOCKET_TIMEOUT_S must be numeric"
        ) from exc
    if not np.isfinite(value) or value < 1.0 or value > 3600.0:
        raise Image3FWireError(
            "IMAGE3F_CLIENT_SOCKET_TIMEOUT_S must be between 1 and 3600 seconds"
        )
    return value


def _max_connections():
    raw = os.environ.get("IMAGE3F_MAX_CONNECTIONS", str(DEFAULT_MAX_CONNECTIONS))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise Image3FWireError("IMAGE3F_MAX_CONNECTIONS must be an integer") from exc
    if value < 1 or value > 4096:
        raise Image3FWireError("IMAGE3F_MAX_CONNECTIONS must be between 1 and 4096")
    return value


def _bind_host():
    value = os.environ.get("IMAGE3F_BIND_HOST", "127.0.0.1").strip()
    if not value or any(character.isspace() for character in value):
        raise Image3FWireError(
            "IMAGE3F_BIND_HOST must be one non-empty host token"
        )
    return value


def _validate_wire_value(value):
    remaining = MAX_WIRE_ITEMS
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        remaining -= 1
        if remaining < 0:
            raise pickle.UnpicklingError("Image3F wire payload contains too many values")
        if depth > MAX_WIRE_NESTING:
            raise pickle.UnpicklingError("Image3F wire payload nesting is too deep")
        if current is None or isinstance(current, (bool, int, float, str, bytes)):
            continue
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise pickle.UnpicklingError("Image3F wire mapping keys must be strings")
            stack.extend((item, depth + 1) for item in current.values())
            continue
        if isinstance(current, (list, tuple)):
            stack.extend((item, depth + 1) for item in current)
            continue
        raise pickle.UnpicklingError(
            f"Image3F wire payload contains forbidden {type(current).__name__}"
        )


def _safe_pickle_loads(payload):
    value = _PrimitiveUnpickler(io.BytesIO(payload)).load()
    _validate_wire_value(value)
    return value


def reference_provenance(
    *, step: int, lag: int, source: str, reference_sha256: str
) -> dict[str, object]:
    """Build a strict proof of the RGB reference selected by the history store."""

    step = int(step)
    lag = int(lag)
    if step == 0:
        expected_source = "bootstrap_anchor"
        kind = "bootstrap_anchor"
        source_step = -1
    elif step < lag:
        expected_source = "generated_episode_frame0"
        kind = "generated"
        source_step = 0
    else:
        expected_source = f"generated_t-{lag}"
        kind = "generated"
        source_step = step - lag
    if source != expected_source:
        raise RuntimeError(
            "reference history/source mismatch: "
            f"step={step} lag={lag} expected={expected_source!r} got={source!r}"
        )
    if (
        not isinstance(reference_sha256, str)
        or len(reference_sha256) != 64
        or any(value not in "0123456789abcdef" for value in reference_sha256)
    ):
        raise RuntimeError("reference SHA-256 is malformed")
    return {
        "schema": REFERENCE_PROVENANCE_SCHEMA,
        "request_step": step,
        "configured_lag": lag,
        "kind": kind,
        "source_step": source_step,
        "reference_sha256": reference_sha256,
    }


def h5v4(d, raw_clip, inv_clip, raw_pct=None, invalid_fill="far"):
    d = d.astype(np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    d[bad] = raw_clip[1] if invalid_fill == "far" else raw_clip[0]
    if raw_pct is not None:
        lo, hi = np.percentile(d, raw_pct)
        d = np.clip(d, lo, hi)
    d = np.clip(d, raw_clip[0], raw_clip[1])
    dn = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    inv = np.clip(1.0 / (dn + 1e-3), inv_clip[0], inv_clip[1])
    inv = np.log(inv)
    inv = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
    return (inv * 255).astype(np.uint8)


def to_pil(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a).convert("RGB").resize((W, H), Image.BILINEAR)


def run_request(session, session_lock, req):
    mode = req.get("prev_mode", "fixed")
    if mode == "zero":
        raise ValueError("zero mode not supported in persistent server (global patch)")
    tag = f'{req.get("tag_prefix", "q_")}{mode}'
    out = Path(req["out"]); out.mkdir(parents=True, exist_ok=True)
    anchor_idx = int(req.get("anchor_index", 0))
    init_prev_dir = req.get("init_prev_dir") or os.environ.get("IMAGE3F_INIT_PREV_DIR")
    if mode in ("fixed", "chain") and not init_prev_dir:
        raise ValueError("fixed/chain mode requires init_prev_dir")
    init_dir = Path(init_prev_dir) if init_prev_dir else QUEUE
    seed = int(req.get("seed", 42))
    stats = {}
    for cpath in req["chunks"]:
        d = np.load(cpath, allow_pickle=False)
        name = Path(cpath).stem
        prompt = str(d["prompt"])
        T = min(int(d["depth_ctrl__head_color"].shape[0]), int(req.get("max_frames", 81)))
        depth_seq = {}
        for _, view in VIEW_MAP:
            if req.get("depth_variant", "droid") == "droid":
                depth_seq[view] = h5v4(d[f"depth_raw__{view}"].astype(np.float32)[:T], **DROID_PARAMS[view])
            else:
                depth_seq[view] = d[f"depth_ctrl__{view}"][:T]
        prev = {v: Image.new("RGB", (W, H)) for _, v in VIEW_MAP}
        gen_frames = {v: [] for _, v in VIEW_MAP}
        t_start = 0
        if mode in ("fixed", "chain"):
            anchor = np.load(init_dir / f"{name}_teacher3v.npz", allow_pickle=False)
            for _, v in VIEW_MAP:
                prev[v] = Image.fromarray(anchor[f"gen__{v}"][anchor_idx])
            if mode == "chain":
                for _, v in VIEW_MAP:
                    gen_frames[v].append(anchor[f"gen__{v}"][anchor_idx])
                t_start = 1
        times = []
        for t in range(t_start, T):
            r = Image3FRequest(
                prompt=prompt,
                depth=tuple(to_pil(depth_seq[v][t]) for _, v in VIEW_MAP),
                mask=tuple(to_pil(d[f"rgb_robot__{v}"][t]) for _, v in VIEW_MAP),
                previous_rgb=tuple(prev[v] for _, v in VIEW_MAP),
                seed=seed,
            )
            t0 = time.perf_counter()
            with session_lock:
                imgs, timing = session.generate(r, output_type="rgb")
            wall = (time.perf_counter() - t0) * 1000
            stacked = np.asarray(imgs[0])
            for vi, (_, view) in enumerate(VIEW_MAP):
                fr = stacked[vi * H:(vi + 1) * H]
                gen_frames[view].append(fr)
                if mode == "chain":
                    prev[view] = Image.fromarray(fr)
            ms = timing.get("pipeline_ms", wall) if isinstance(timing, dict) else wall
            times.append(float(ms))
        np.savez_compressed(out / f"{name}_gen_{tag}.npz",
                            **{f"gen__{v}": np.stack(gen_frames[v]) for _, v in VIEW_MAP},
                            **{f"depth__{v}": depth_seq[v] for _, v in VIEW_MAP},
                            prompt=prompt)
        warm = times[1:] if len(times) > 1 else times
        stats[name] = {"mean_ms": float(np.mean(warm)), "p90_ms": float(np.percentile(warm, 90)),
                       "frames": len(times)}
        print(f"[{tag}/{name}] {len(times)} frames, warm mean {np.mean(warm):.1f}ms", flush=True)
    return {"tag": tag, "stats": stats}


def encode_array(arr):
    arr = np.ascontiguousarray(arr)
    return {"__nd__": True, "dtype": str(arr.dtype), "shape": tuple(arr.shape),
            "data": arr.tobytes()}


def decode_array(obj):
    if not isinstance(obj, dict) or set(obj) != {"__nd__", "dtype", "shape", "data"}:
        raise Image3FWireError("malformed Image3F ndarray envelope")
    if obj.get("__nd__") is not True or not isinstance(obj.get("dtype"), str):
        raise Image3FWireError("malformed Image3F ndarray metadata")
    try:
        dtype = np.dtype(obj["dtype"])
    except (TypeError, ValueError) as exc:
        raise Image3FWireError("invalid Image3F ndarray dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise Image3FWireError("object and structured Image3F dtypes are forbidden")
    shape = obj.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) not in (2, 3)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape)
    ):
        raise Image3FWireError("invalid Image3F ndarray shape")
    data = obj.get("data")
    if not isinstance(data, bytes):
        raise Image3FWireError("Image3F ndarray data must be raw bytes")
    expected_bytes = dtype.itemsize
    for dimension in shape:
        if expected_bytes > _max_message_bytes() // dimension:
            raise Image3FWireError("Image3F ndarray shape exceeds the message limit")
        expected_bytes *= dimension
    if expected_bytes != len(data) or expected_bytes > _max_message_bytes():
        raise Image3FWireError("Image3F ndarray byte length does not match its shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def validate_render_inputs(depth, mask, previous_rgb):
    """Require the exact production three-view conditioning tensor contract."""

    expected_views = tuple(view for _, view in VIEW_MAP)
    for label, mapping in (
        ("depth", depth),
        ("mask", mask),
        ("previous_rgb", previous_rgb),
    ):
        if not isinstance(mapping, dict) or set(mapping) != set(expected_views):
            raise Image3FWireError(
                f"Image3F {label} views must be exactly {expected_views!r}"
            )
    for view in expected_views:
        if depth[view].dtype != np.uint8 or depth[view].shape != (H, W):
            raise Image3FWireError(
                f"Image3F depth[{view!r}] must be uint8 {(H, W)}"
            )
        for label, mapping in (("mask", mask), ("previous_rgb", previous_rgb)):
            if mapping[view].dtype != np.uint8 or mapping[view].shape != (H, W, 3):
                raise Image3FWireError(
                    f"Image3F {label}[{view!r}] must be uint8 {(H, W, 3)}"
                )


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("client closed")
        buf += c
    return buf


def _send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _max_message_bytes():
        raise Image3FWireError(
            f"Image3F outbound message is {len(payload)} bytes, exceeding the limit"
        )
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_msg(sock):
    n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if n > _max_message_bytes():
        raise Image3FWireError(
            f"Image3F inbound message declares {n} bytes, exceeding the limit"
        )
    return _safe_pickle_loads(_recv_exact(sock, n))


def handle_conn(
    sock,
    addr,
    session,
    session_lock,
    references,
    metrics,
    runtime_identity,
    renderer_identity,
    renderer_identity_sha256,
    server_incarnation_id,
    connections,
):
    print(f"tcp client {addr}", flush=True)
    try:
        while True:
            req = _recv_msg(sock)
            if not isinstance(req, dict):
                raise Image3FWireError("Image3F request must be a mapping")
            cmd = req.get("cmd", "render_frame")
            if cmd not in {"ping", "reset_stream", "render_frame"}:
                _send_msg(sock, {
                    "ok": False,
                    "error": f"unsupported Image3F command {cmd!r}",
                    "retryable": False,
                    "server_incarnation_id": server_incarnation_id,
                })
                continue
            if req.get("protocol_version") != PROTOCOL_VERSION:
                _send_msg(sock, {
                    "ok": False,
                    "error": "Image3F protocol version mismatch",
                    "retryable": False,
                    "server_incarnation_id": server_incarnation_id,
                })
                continue
            expected_incarnation = req.get("expected_server_incarnation_id")
            if (
                expected_incarnation is not None
                and expected_incarnation != server_incarnation_id
            ):
                _send_msg(sock, {
                    "ok": False,
                    "error": (
                        "Image3F server incarnation changed; lagged reference "
                        "history is unavailable"
                    ),
                    "retryable": False,
                    "server_incarnation_id": server_incarnation_id,
                })
                continue
            reference_lag = 0
            if cmd == "render_frame":
                raw_reference_lag = req.get("reference_lag", 0)
                if (
                    isinstance(raw_reference_lag, bool)
                    or not isinstance(raw_reference_lag, int)
                    or raw_reference_lag < 0
                ):
                    _send_msg(sock, {
                        "ok": False,
                        "error": "Image3F reference_lag must be a non-negative integer",
                        "retryable": False,
                        "server_incarnation_id": server_incarnation_id,
                    })
                    continue
                reference_lag = raw_reference_lag
                if (
                    reference_lag > 0
                    and expected_incarnation != server_incarnation_id
                ):
                    _send_msg(sock, {
                        "ok": False,
                        "error": (
                            "stateful Image3F render requires the exact current "
                            "server incarnation"
                        ),
                        "retryable": False,
                        "server_incarnation_id": server_incarnation_id,
                    })
                    continue
            if cmd == "ping":
                _send_msg(sock, {
                    "ok": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "server_id": os.environ.get("IMAGE3F_SERVER_ID", "legacy"),
                    "server_incarnation_id": server_incarnation_id,
                    "capabilities": {
                        "lagged_reference_history": True,
                        "server_incarnation_binding": True,
                        "reference_lag_max": int(os.environ.get("IMAGE3F_MAX_REFERENCE_LAG", "512")),
                        "serialized_inference": True,
                        "output_views": [view for _, view in VIEW_MAP],
                        "output_layout": "vertical",
                        "output_tile_shape_hwc": [H, W, 3],
                        "output_stacked_shape_hwc": [len(VIEW_MAP) * H, W, 3],
                        "client_socket_timeout_s": connections.client_socket_timeout_s,
                        "max_connections": connections.max_connections,
                    },
                    "runtime_identity": runtime_identity,
                    "renderer_identity": renderer_identity,
                    "renderer_identity_sha256": renderer_identity_sha256,
                    "history": references.stats(),
                    "metrics": metrics.snapshot(),
                })
                continue
            if cmd == "reset_stream":
                try:
                    reset_stream_id = str(req.get("stream_id", ""))
                    if not reset_stream_id:
                        raise ValueError("stream_id is required for reset_stream")
                    removed = references.reset(reset_stream_id)
                    _send_msg(sock, {
                        "ok": True,
                        "removed": removed,
                        "stream_id": reset_stream_id,
                        "server_id": os.environ.get("IMAGE3F_SERVER_ID", "legacy"),
                        "renderer_identity_sha256": renderer_identity_sha256,
                        "server_incarnation_id": server_incarnation_id,
                    })
                except Exception as exc:
                    _send_msg(sock, {
                        "ok": False,
                        "error": f"{exc}",
                        "retryable": False,
                        "server_incarnation_id": server_incarnation_id,
                    })
                continue
            prepared = False
            stream_id = str(req.get("stream_id", ""))
            episode_id = str(req.get("episode_id", ""))
            step = int(req.get("step", -1))
            try:
                t0 = time.perf_counter()
                depth = {v: decode_array(req["depth"][v]) for v in req["depth"]}
                mask = {v: decode_array(req["mask"][v]) for v in req["mask"]}
                prev = {v: decode_array(req["prev"][v]) for v in req["prev"]}
                validate_render_inputs(depth, mask, prev)
                prompt = req.get("prompt")
                if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 16384:
                    raise Image3FWireError(
                        "Image3F prompt must be a non-empty UTF-8 string of at most 16 KiB"
                    )
                seed = req.get("seed", 42)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise Image3FWireError("Image3F seed must be an integer")
                reference_source = "client"
                if reference_lag:
                    request_id = str(req.get("request_id", ""))
                    if (
                        not stream_id
                        or not episode_id
                        or len(stream_id) > 512
                        or len(episode_id) > 256
                        or len(request_id) > 1024
                        or request_id != f"{stream_id}|{episode_id}|{step}"
                    ):
                        raise ValueError(
                            "stateful Image3F stream/episode/request identity is malformed"
                        )
                    max_lag = int(os.environ.get("IMAGE3F_MAX_REFERENCE_LAG", "512"))
                    if reference_lag > max_lag:
                        raise ValueError(
                            f"reference_lag {reference_lag} exceeds server maximum {max_lag}"
                        )
                    replay = references.replay_with_provenance(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        step=step,
                        request_id=request_id,
                    )
                    if replay is not None:
                        replay_views, replay_source, replay_reference_sha256 = replay
                        provenance = reference_provenance(
                            step=step,
                            lag=reference_lag,
                            source=replay_source,
                            reference_sha256=replay_reference_sha256,
                        )
                        metrics.record_replay()
                        _send_msg(sock, {
                            "ok": True,
                            "views": {
                                v: encode_array(image) for v, image in replay_views.items()
                            },
                            "ms": 0.0,
                            "queue_ms": 0.0,
                            "inference_ms": 0.0,
                            "reference_source": replay_source,
                            "reference_provenance": provenance,
                            "generated_sha256": references.image_mapping_sha256(
                                replay_views
                            ),
                            "renderer_identity_sha256": renderer_identity_sha256,
                            "server_incarnation_id": server_incarnation_id,
                            "stream_step": step,
                            "replayed": True,
                        })
                        continue
                    prev, reference_source, reference_sha256 = (
                        references.prepare_with_provenance(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        step=step,
                        lag=reference_lag,
                        anchor=prev,
                        request_id=request_id,
                        )
                    )
                    prepared = True
                order = [v for _, v in VIEW_MAP]
                r = Image3FRequest(
                    prompt=prompt,
                    depth=tuple(to_pil(depth[v]) for v in order),
                    mask=tuple(to_pil(mask[v]) for v in order),
                    previous_rgb=tuple(to_pil(prev[v]) for v in order),
                    seed=seed,
                )
                queue_start = time.perf_counter()
                with session_lock:
                    queue_ms = (time.perf_counter() - queue_start) * 1000
                    infer_start = time.perf_counter()
                    imgs, timing = session.generate(r, output_type="rgb")
                    inference_ms = (time.perf_counter() - infer_start) * 1000
                stacked = np.asarray(imgs[0])
                expected_shape = (len(order) * H, W, 3)
                if stacked.shape != expected_shape or stacked.dtype != np.uint8:
                    raise RuntimeError(
                        "Image3F output contract mismatch: expected uint8 "
                        f"{expected_shape}, got {stacked.dtype} {stacked.shape}"
                    )
                out_arrays = {
                    v: np.ascontiguousarray(stacked[i * H:(i + 1) * H])
                    for i, v in enumerate(order)
                }
                if prepared:
                    references.commit(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        step=step,
                        generated=out_arrays,
                        request_id=request_id,
                    )
                    prepared = False
                out = {v: encode_array(image) for v, image in out_arrays.items()}
                generated_sha256 = references.image_mapping_sha256(out_arrays)
                ms = (time.perf_counter() - t0) * 1000
                metrics.record(ms=ms, queue_ms=queue_ms, inference_ms=inference_ms)
                provenance = (
                    reference_provenance(
                        step=step,
                        lag=reference_lag,
                        source=reference_source,
                        reference_sha256=reference_sha256,
                    )
                    if reference_lag
                    else None
                )
                _send_msg(sock, {
                    "ok": True,
                    "views": out,
                    "ms": ms,
                    "queue_ms": queue_ms,
                    "inference_ms": inference_ms,
                    "reference_source": reference_source,
                    "reference_provenance": provenance,
                    "generated_sha256": generated_sha256,
                    "renderer_identity_sha256": renderer_identity_sha256,
                    "server_incarnation_id": server_incarnation_id,
                    "stream_step": step if reference_lag else None,
                })
            except Exception as exc:
                if prepared:
                    references.abort(stream_id=stream_id, episode_id=episode_id, step=step)
                metrics.record_error()
                error = f"{exc}"
                _send_msg(sock, {
                    "ok": False,
                    "error": error,
                    "retryable": "in flight" in error,
                    "server_incarnation_id": server_incarnation_id,
                })
    except (
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
        OSError,
        struct.error,
        pickle.UnpicklingError,
        Image3FWireError,
    ):
        pass
    finally:
        sock.close()
        connections.remove(sock, threading.current_thread())
        print(f"tcp client {addr} gone", flush=True)


class ServerMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.replays = 0
        self.total_ms = 0.0
        self.total_queue_ms = 0.0
        self.total_inference_ms = 0.0

    def record(self, *, ms, queue_ms, inference_ms):
        with self._lock:
            self.requests += 1
            self.total_ms += float(ms)
            self.total_queue_ms += float(queue_ms)
            self.total_inference_ms += float(inference_ms)

    def record_error(self):
        with self._lock:
            self.errors += 1

    def record_replay(self):
        with self._lock:
            self.replays += 1

    def snapshot(self):
        with self._lock:
            denom = max(1, self.requests)
            return {
                "requests": self.requests,
                "errors": self.errors,
                "replays": self.replays,
                "mean_ms": self.total_ms / denom,
                "mean_queue_ms": self.total_queue_ms / denom,
                "mean_inference_ms": self.total_inference_ms / denom,
            }


class ConnectionRegistry:
    """Track live TCP handlers so STOP cannot race an in-flight request."""

    def __init__(self, *, max_connections, client_socket_timeout_s):
        self._lock = threading.Lock()
        self._sockets = set()
        self._threads = set()
        self.max_connections = int(max_connections)
        self.client_socket_timeout_s = float(client_socket_timeout_s)
        self.rejected_connections = 0

    def start_handler(self, sock, thread):
        with self._lock:
            if len(self._threads) >= self.max_connections:
                self.rejected_connections += 1
                return False
            self._sockets.add(sock)
            self._threads.add(thread)
            try:
                thread.start()
            except BaseException:
                self._sockets.discard(sock)
                self._threads.discard(thread)
                raise
            return True

    def remove(self, sock, thread):
        with self._lock:
            self._sockets.discard(sock)
            self._threads.discard(thread)

    def close_clients(self):
        with self._lock:
            sockets = list(self._sockets)
        for client in sockets:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass

    def drain(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._lock:
                threads = list(self._threads)
            if not threads:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            for thread in threads:
                thread.join(timeout=min(0.25, remaining))

    def count(self):
        with self._lock:
            return len(self._threads)


def create_tcp_server(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((_bind_host(), port))
    srv.listen(int(os.environ.get("IMAGE3F_LISTEN_BACKLOG", "256")))
    return srv, int(srv.getsockname()[1])


def tcp_listener(
    srv,
    session,
    session_lock,
    references,
    metrics,
    runtime_identity,
    renderer_identity,
    renderer_identity_sha256,
    server_incarnation_id,
    stopped,
    connections,
):
    print(f"tcp listening on {srv.getsockname()[1]}", flush=True)
    srv.settimeout(1.0)
    while not stopped.is_set():
        try:
            sock, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            if stopped.is_set():
                return
            raise
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(connections.client_socket_timeout_s)
        handler = threading.Thread(
            target=handle_conn,
            args=(
                sock,
                addr,
                session,
                session_lock,
                references,
                metrics,
                runtime_identity,
                renderer_identity,
                renderer_identity_sha256,
                server_incarnation_id,
                connections,
            ),
            daemon=True,
        )
        if not connections.start_handler(sock, handler):
            sock.close()


def acknowledge_stop(stop_controller):
    if stop_controller.error is not None:
        raise RuntimeError("Image3F STOP watcher failed") from stop_controller.error
    acknowledgement = stop_controller.acknowledge(
        os.environ.get("IMAGE3F_SERVER_ID", "legacy")
    )
    print(
        f"STOP_REQUESTED ack={stop_controller.ack_path} "
        f"server_id={acknowledgement['server_id']}",
        flush=True,
    )
    print("SERVER_STOPPED_CLEANLY", flush=True)


def main():
    stop_controller = StopController.from_environment()
    if stop_controller.requested():
        acknowledge_stop(stop_controller)
        return
    stopped = threading.Event()
    stop_watcher = threading.Thread(
        target=stop_controller.watch, args=(stopped,), daemon=True
    )
    stop_watcher.start()
    inbox = QUEUE / "inbox"; done = QUEUE / "done"
    inbox.mkdir(parents=True, exist_ok=True); done.mkdir(parents=True, exist_ok=True)
    print("loading Image3FSession ...", flush=True)
    t0 = time.time()
    session = Image3FSession(memory_mode="auto", vae_mode="untiled",
                             model_root=os.environ["ROBORENDER_MODEL_BASE"])
    runtime_identity = describe_runtime_identity(session)
    renderer_identity = describe_renderer_identity(
        runtime_identity,
        server_script=Path(__file__),
        expected_server_script_sha256=os.environ.get(
            "IMAGE3F_EXPECT_SERVER_SCRIPT_SHA256"
        ),
    )
    renderer_identity_sha256 = canonical_json_sha256(renderer_identity)
    server_incarnation_id = secrets.token_hex(16)
    session_lock = threading.Lock()
    references = LaggedReferenceStore(
        max_streams=int(os.environ.get("IMAGE3F_MAX_STREAMS", "64"))
    )
    metrics = ServerMetrics()
    print(f"SESSION_READY in {time.time()-t0:.0f}s", flush=True)
    if stopped.is_set():
        acknowledge_stop(stop_controller)
        return
    job_digits = "".join(c for c in os.environ.get("SLURM_JOB_ID", "0") if c.isdigit())
    default_port = 18500 + int(job_digits or "0") % 1000
    requested_port = int(os.environ.get("IMAGE3F_PORT", str(default_port)))
    srv, port = create_tcp_server(requested_port)
    connections = ConnectionRegistry(
        max_connections=_max_connections(),
        client_socket_timeout_s=_client_socket_timeout_s(),
    )
    listener = threading.Thread(
        target=tcp_listener,
        args=(
            srv,
            session,
            session_lock,
            references,
            metrics,
            runtime_identity,
            renderer_identity,
            renderer_identity_sha256,
            server_incarnation_id,
            stopped,
            connections,
        ),
        daemon=True,
    )
    listener.start()
    import subprocess
    host = os.environ.get("IMAGE3F_ADVERTISE_HOST", "").strip()
    bind_host = _bind_host()
    if not host and bind_host != "0.0.0.0":
        host = bind_host
    if not host:
        host = subprocess.run(
            ["hostname", "-f"], capture_output=True, text=True
        ).stdout.strip()
    if not host:
        host = subprocess.run(
            ["hostname"], capture_output=True, text=True
        ).stdout.strip()
    endpoint = Path(os.environ["ENDPOINT_FILE"])
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint_tmp = endpoint.with_name(f".{endpoint.name}.{os.getpid()}.tmp")
    endpoint_tmp.write_text(f"{host} {port}\n")
    os.replace(endpoint_tmp, endpoint)
    (QUEUE / "READY").write_text(str(time.time()))
    while not stopped.is_set():
        reqs = list(inbox.glob("*.json"))
        if os.environ.get("IMAGE3F_ALLOW_PY_QUEUE", "0") == "1":
            reqs += list(inbox.glob("*.py"))
        reqs = sorted(reqs)
        if not reqs:
            stopped.wait(stop_controller.poll_seconds)
            continue
        rp = reqs[0]
        print(f"processing {rp.name}", flush=True)
        result = {"request": rp.name, "status": "ok"}
        try:
            t0 = time.time()
            if rp.suffix == ".py":
                # warm-GPU script: runs with `session` in scope; set RESULT dict
                import io, contextlib
                g = {"session": session, "np": np, "Image": Image,
                     "Image3FRequest": Image3FRequest, "RESULT": {}}
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(rp.read_text(), rp.name, "exec"), g)
                result["stdout"] = buf.getvalue()[-20000:]
                result["result"] = g.get("RESULT", {})
            else:
                req = json.loads(rp.read_text())
                result.update(run_request(session, session_lock, req))
            result["seconds"] = time.time() - t0
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{exc}\n{traceback.format_exc()}"
        (done / (rp.stem + ".result.json")).write_text(json.dumps(result, indent=1))
        shutil.move(str(rp), done / rp.name)
        print(f"done {rp.name}: {result['status']}", flush=True)
    srv.close()
    listener.join(timeout=2.0)
    if listener.is_alive():
        raise RuntimeError("Image3F STOP timed out closing the TCP listener")
    connections.close_clients()
    # A handler may have passed socket receive and entered GPU inference just
    # before STOP.  Closing clients prevents new requests; taking the session
    # lock proves any in-flight generation has left the critical section.
    with session_lock:
        pass
    connections.close_clients()
    if not connections.drain(timeout_seconds=10.0):
        raise RuntimeError(
            f"Image3F STOP timed out draining {connections.count()} TCP handlers"
        )
    acknowledge_stop(stop_controller)


if __name__ == "__main__":
    main()
