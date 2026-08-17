#!/usr/bin/env python
"""A/B prep: copy N chunks, apply sharper head-depth mapping from stored raw,
assign a varied per-chunk prompt, strip rgb_gen for re-render."""
import glob, os, shutil, sys
import numpy as np
sys.path.insert(0, "/svl/u/ravenh/fast_roborender")
from renormalize_depth import h5v4_frame_seq
from videogen_prompts import build_prompt

SRC = "/svl/u/ravenh/fast_roborender/videogen_dataset"
DST = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
N = 200
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(f"{SRC}/chunk_*.npz"))[:N]
for i, f in enumerate(files):
    with np.load(f) as z:
        data = {k: z[k].copy() for k in z.files}
    raw = data.get("depth_raw__head_color")
    if raw is None:
        print("no raw, skip", f); continue
    # sharper head mapping: norm_range [0.05,2.6], inv_clip [1,40]
    data["depth_ctrl__head_color"] = h5v4_frame_seq(
        raw.astype(np.float32), [0.05, 10.0], [1.0, 40.0], None, "far",
        norm_range=[0.05, 2.6])
    data["prompt"] = np.array(build_prompt(i))
    data.pop("rgb_gen", None)
    out = os.path.join(DST, os.path.basename(f))
    np.savez_compressed(out + ".tmp.npz", **data)
    os.replace(out + ".tmp.npz", out)
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{len(files)}", flush=True)
print(f"AB_PREP_DONE {len(files)} chunks in {DST}", flush=True)
