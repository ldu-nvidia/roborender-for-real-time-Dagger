#!/usr/bin/env python
"""Render full-res 3-view teacher (50-step) generations for given chunks."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("ROBORENDER_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "GR00T-VisualSim2Real"))
from gr00t.rl.utils.videogen_client import VideoGenRenderClient

VIEWS = ["hand_left", "hand_right", "head_color"]

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

client = VideoGenRenderClient(host=args.host, port=args.port,
                              connect_retries=1, connect_retry_interval_s=5)

def render_retry(depths, masks, prompt, seed, tries=90, wait=30):
    global client
    for i in range(tries):
        try:
            return client.render_chunk(depths, mask_per_view=masks, prompt=prompt, seed=seed)
        except (ConnectionError, OSError, RuntimeError) as exc:
            print(f"render attempt {i} failed ({exc}); retry in {wait}s", flush=True)
            try:
                client.close()
            except Exception:
                pass
            time.sleep(wait)
            client = VideoGenRenderClient(host=args.host, port=args.port,
                                          connect_retries=1, connect_retry_interval_s=5)
    raise RuntimeError("render server never became ready")

for path in args.chunks:
    name = os.path.basename(path).replace(".npz", "")
    with np.load(path, allow_pickle=True) as z:
        depths = {v: z[f"depth_ctrl__{v}"].copy() for v in VIEWS}
        masks = {v: z[f"rgb_robot__{v}"].copy() for v in VIEWS}
        prompt = str(z["prompt"])
    t0 = time.time()
    resp = render_retry(depths, masks, prompt, 42)
    dt = time.time() - t0
    np.savez_compressed(os.path.join(args.out, f"{name}_teacher3v.npz"),
                        **{f"gen__{v}": np.asarray(resp[v], dtype=np.uint8) for v in VIEWS},
                        prompt=prompt, render_seconds=dt)
    print(f"{name}: rendered 3-view in {dt:.1f}s", flush=True)
print("TEACHER_RENDER_DONE", flush=True)
