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
import json, os, pickle, shutil, socket, struct, sys, threading, time, traceback
from pathlib import Path

import numpy as np
from PIL import Image
from reference_history import LaggedReferenceStore
from runtime_identity import describe_runtime_identity

BUNDLE = Path(os.environ["IMAGE3F_ROOT"])
sys.path.insert(0, str(BUNDLE))
from runtime import Image3FRequest, Image3FSession  # noqa: E402

QUEUE = Path(os.environ["IMAGE3F_QUEUE"])
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
W, H = 416, 240
PROTOCOL_VERSION = 2

DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0)),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
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
        d = np.load(cpath, allow_pickle=True)
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
            anchor = np.load(init_dir / f"{name}_teacher3v.npz", allow_pickle=True)
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
    if isinstance(obj, dict) and obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


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
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_msg(sock):
    n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    return pickle.loads(_recv_exact(sock, n))


def handle_conn(
    sock, addr, session, session_lock, references, metrics, runtime_identity
):
    print(f"tcp client {addr}", flush=True)
    try:
        while True:
            req = _recv_msg(sock)
            cmd = req.get("cmd", "render_frame")
            if cmd == "ping":
                _send_msg(sock, {
                    "ok": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "server_id": os.environ.get("IMAGE3F_SERVER_ID", "legacy"),
                    "capabilities": {
                        "lagged_reference_history": True,
                        "reference_lag_max": int(os.environ.get("IMAGE3F_MAX_REFERENCE_LAG", "512")),
                        "serialized_inference": True,
                        "output_views": [view for _, view in VIEW_MAP],
                        "output_layout": "vertical",
                        "output_tile_shape_hwc": [H, W, 3],
                        "output_stacked_shape_hwc": [len(VIEW_MAP) * H, W, 3],
                    },
                    "runtime_identity": runtime_identity,
                    "history": references.stats(),
                    "metrics": metrics.snapshot(),
                })
                continue
            if cmd == "reset_stream":
                try:
                    removed = references.reset(str(req.get("stream_id", "")))
                    _send_msg(sock, {"ok": True, "removed": removed})
                except Exception as exc:
                    _send_msg(sock, {"ok": False, "error": f"{exc}"})
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
                reference_lag = int(req.get("reference_lag", 0))
                reference_source = "client"
                if reference_lag:
                    request_id = str(req.get("request_id", ""))
                    if not request_id:
                        raise ValueError("request_id is required with reference_lag")
                    max_lag = int(os.environ.get("IMAGE3F_MAX_REFERENCE_LAG", "512"))
                    if reference_lag > max_lag:
                        raise ValueError(
                            f"reference_lag {reference_lag} exceeds server maximum {max_lag}"
                        )
                    replay = references.replay(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        step=step,
                        request_id=request_id,
                    )
                    if replay is not None:
                        metrics.record_replay()
                        _send_msg(sock, {
                            "ok": True,
                            "views": {v: encode_array(image) for v, image in replay.items()},
                            "ms": 0.0,
                            "queue_ms": 0.0,
                            "inference_ms": 0.0,
                            "reference_source": "idempotent_replay",
                            "stream_step": step,
                            "replayed": True,
                        })
                        continue
                    prev, reference_source = references.prepare(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        step=step,
                        lag=reference_lag,
                        anchor=prev,
                        request_id=request_id,
                    )
                    prepared = True
                order = [v for _, v in VIEW_MAP]
                r = Image3FRequest(
                    prompt=str(req["prompt"]),
                    depth=tuple(to_pil(depth[v]) for v in order),
                    mask=tuple(to_pil(mask[v]) for v in order),
                    previous_rgb=tuple(to_pil(prev[v]) for v in order),
                    seed=int(req.get("seed", 42)),
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
                ms = (time.perf_counter() - t0) * 1000
                metrics.record(ms=ms, queue_ms=queue_ms, inference_ms=inference_ms)
                _send_msg(sock, {
                    "ok": True,
                    "views": out,
                    "ms": ms,
                    "queue_ms": queue_ms,
                    "inference_ms": inference_ms,
                    "reference_source": reference_source,
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
                })
    except (ConnectionError, ConnectionResetError, BrokenPipeError, OSError, struct.error):
        pass
    finally:
        sock.close()
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


def create_tcp_server(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(int(os.environ.get("IMAGE3F_LISTEN_BACKLOG", "256")))
    return srv, int(srv.getsockname()[1])


def tcp_listener(srv, session, session_lock, references, metrics, runtime_identity):
    print(f"tcp listening on {srv.getsockname()[1]}", flush=True)
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(
            target=handle_conn,
            args=(
                sock,
                addr,
                session,
                session_lock,
                references,
                metrics,
                runtime_identity,
            ),
            daemon=True,
        ).start()


def main():
    inbox = QUEUE / "inbox"; done = QUEUE / "done"
    inbox.mkdir(parents=True, exist_ok=True); done.mkdir(parents=True, exist_ok=True)
    print("loading Image3FSession ...", flush=True)
    t0 = time.time()
    session = Image3FSession(memory_mode="auto", vae_mode="untiled",
                             model_root=os.environ["ROBORENDER_MODEL_BASE"])
    runtime_identity = describe_runtime_identity(session)
    session_lock = threading.Lock()
    references = LaggedReferenceStore(
        max_streams=int(os.environ.get("IMAGE3F_MAX_STREAMS", "64"))
    )
    metrics = ServerMetrics()
    print(f"SESSION_READY in {time.time()-t0:.0f}s", flush=True)
    job_digits = "".join(c for c in os.environ.get("SLURM_JOB_ID", "0") if c.isdigit())
    default_port = 18500 + int(job_digits or "0") % 1000
    requested_port = int(os.environ.get("IMAGE3F_PORT", str(default_port)))
    srv, port = create_tcp_server(requested_port)
    threading.Thread(
        target=tcp_listener,
        args=(srv, session, session_lock, references, metrics, runtime_identity),
        daemon=True,
    ).start()
    import subprocess
    host = os.environ.get("IMAGE3F_ADVERTISE_HOST", "").strip()
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
    while True:
        reqs = list(inbox.glob("*.json"))
        if os.environ.get("IMAGE3F_ALLOW_PY_QUEUE", "1") == "1":
            reqs += list(inbox.glob("*.py"))
        reqs = sorted(reqs)
        if not reqs:
            time.sleep(2)
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


if __name__ == "__main__":
    main()
