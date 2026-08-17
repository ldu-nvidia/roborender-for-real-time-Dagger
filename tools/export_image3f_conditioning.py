import json, os, sys
from pathlib import Path
import numpy as np
import cv2

def h5v4(d, raw_clip, inv_clip, raw_pct=None, invalid_fill="far"):
    d = d.astype(np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    d[bad] = raw_clip[1] if invalid_fill == "far" else raw_clip[0]
    if raw_pct is not None:
        lo, hi = np.percentile(d, raw_pct)
        d = np.clip(d, lo, hi)
    d = np.clip(d, raw_clip[0], raw_clip[1])
    dn = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    inv = 1.0 / (dn + 1e-3)
    inv = np.clip(inv, inv_clip[0], inv_clip[1])
    inv = np.log(inv)
    inv = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
    return (inv * 255).astype(np.uint8)

DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0)),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
}
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]

POOL = Path("/svl/u/ravenh/fast_roborender/videogen_dataset")
OUT = Path("/svl/u/ravenh/fast_roborender/image3f_reproduction")
CHUNKS = ["chunk_16222830_0001309", "chunk_16222830_0000051", "chunk_16222830_0000501"]
W, H = 416, 240

def save_png(path, a):
    if a.ndim == 2:
        a = np.stack([a]*3, -1)
    if a.shape[:2] != (H, W):
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), a[:, :, ::-1])

OUT.mkdir(exist_ok=True)
black = np.zeros((H, W, 3), np.uint8)
save_png(OUT / "black_prev.png", black)

for cn in CHUNKS:
    d = np.load(POOL / f"{cn}.npz", allow_pickle=True)
    cdir = OUT / cn
    prompt = str(d["prompt"])
    (cdir).mkdir(exist_ok=True)
    (cdir / "prompt.txt").write_text(prompt + "\n")
    T = d["depth_ctrl__head_color"].shape[0]
    depth = {}
    for _, view in VIEW_MAP:
        raw = d[f"depth_raw__{view}"].astype(np.float32)
        depth[view] = h5v4(raw, **DROID_PARAMS[view])
    for sub in ["depth_droid", "depth_brecipe", "mask", "requests"]:
        (cdir / sub).mkdir(exist_ok=True)
    for t in range(T):
        names = {}
        for slot, view in VIEW_MAP:
            save_png(cdir / "depth_droid" / f"f{t:03d}_{slot}.png", depth[view][t])
            save_png(cdir / "depth_brecipe" / f"f{t:03d}_{slot}.png", d[f"depth_ctrl__{view}"][t])
            save_png(cdir / "mask" / f"f{t:03d}_{slot}.png", d[f"rgb_robot__{view}"][t])
        req = {
            "prompt": prompt,
            "seed": 42,
            "depth": [f"../depth_droid/f{t:03d}_{s}.png" for s, _ in VIEW_MAP],
            "mask": [f"../mask/f{t:03d}_{s}.png" for s, _ in VIEW_MAP],
            "previous_rgb": ["../../black_prev.png"] * 3,
        }
        (cdir / "requests" / f"f{t:03d}.json").write_text(json.dumps(req, indent=1))
    print(cn, "frames:", T, flush=True)
print("EXPORT_DONE", flush=True)
