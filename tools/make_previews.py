#!/usr/bin/env python
"""Directly render N chunks (single-clip requests) and build 3-view grid
previews. Bypasses the follow-mode renderer. Rows = each view (depth|gen)."""
import sys, os, glob, numpy as np, cv2
sys.path.insert(0, "/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real")
from gr00t.rl.utils.videogen_client import VideoGenRenderClient

VIEWS = ["hand_left", "hand_right", "head_color"]
DS = "/svl/u/ravenh/fast_roborender/videogen_dataset"
with open(os.environ.get("ENDPOINT_FILE","/svl/u/ravenh/fast_roborender/raven_server_endpoint.txt")) as f:
    host, port = f.read().split()
client = VideoGenRenderClient(host=host, port=int(port), timeout_s=1800)
PROMPT_FILE = os.environ.get("PREVIEW_PROMPT_FILE")
SUFFIX = os.environ.get("PREVIEW_SUFFIX", "_preview")
PROMPT_OVERRIDE = open(PROMPT_FILE).read().strip() if PROMPT_FILE else None

# pick 4 chunks from distinct envs for variety
files = sorted(glob.glob(f"{DS}/chunk_*.npz"))
chosen, seen = [], set()
for f in files:
    e = int(np.load(f)["env_id"])
    if e not in seen:
        seen.add(e); chosen.append(f)
    if len(chosen) == 4: break

for f in chosen:
    d = {k: np.load(f)[k] for k in np.load(f).files}
    depths = {v: d[f"depth_ctrl__{v}"] for v in VIEWS}
    masks = {v: d[f"rgb_robot__{v}"] for v in VIEWS} if all(f"rgb_robot__{v}" in d for v in VIEWS) else None
    prompt = PROMPT_OVERRIDE or str(d.get("prompt", ""))
    resp = client.render_chunk(depths, mask_per_view=masks, prompt=prompt, seed=42)
    T = depths[VIEWS[0]].shape[0]
    out = f.replace(".npz", SUFFIX + ".mp4")
    gh, gw = resp[VIEWS[0]].shape[1], resp[VIEWS[0]].shape[2]
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (gw*2, gh*len(VIEWS)))
    for t in range(T):
        rows = []
        for v in VIEWS:
            c = cv2.resize(depths[v][t], (gw, gh), interpolation=cv2.INTER_NEAREST)
            rows.append(np.concatenate([np.repeat(c[..., None], 3, -1), resp[v][t]], axis=1))
        wr.write(cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"PREVIEW {out}", flush=True)
print("ALL_PREVIEWS_DONE", flush=True)
