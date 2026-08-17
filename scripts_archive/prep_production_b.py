#!/usr/bin/env python
"""Migrate the full production dataset to the B recipe:
- first 200 chunks: adopt the already-rendered ab_test copies (B head + varied prompt + rgb_gen)
- rest: renormalize head to B from stored raw, assign varied prompt, strip rgb_gen"""
import glob, os, shutil, sys
import numpy as np
sys.path.insert(0, "/svl/u/ravenh/fast_roborender")
from renormalize_depth import h5v4_frame_seq
from videogen_prompts import build_prompt

PROD = "/svl/u/ravenh/fast_roborender/videogen_dataset"
AB = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
files = sorted(glob.glob(f"{PROD}/chunk_*.npz"))
adopted = renormed = 0
for i, f in enumerate(files):
    nm = os.path.basename(f)
    abf = os.path.join(AB, nm)
    if os.path.exists(abf) and "rgb_gen" in np.load(abf).files:
        shutil.copy2(abf, f)          # B head + varied prompt + render, complete
        adopted += 1
        continue
    with np.load(f) as z:
        data = {k: z[k].copy() for k in z.files}
    raw = data["depth_raw__head_color"].astype(np.float32)
    data["depth_ctrl__head_color"] = h5v4_frame_seq(
        raw, [0.05, 10], [1, 40], None, "far", norm_range=[0.05, 2.6])
    data["prompt"] = np.array(build_prompt(i))
    data.pop("rgb_gen", None)
    np.savez_compressed(f + ".tmp.npz", **data); os.replace(f + ".tmp.npz", f)
    renormed += 1
    if renormed % 200 == 0:
        print(f"renormed {renormed}", flush=True)
print(f"PROD_B_PREP_DONE adopted={adopted} renormed={renormed}", flush=True)
