#!/usr/bin/env python
"""Build matched A/B comparison videos: A = production render (old head depth,
fixed prompt) vs B = same chunk re-rendered (sharp head depth, varied prompt).
Layout per frame: [A rgb_gen | B rgb_gen] upscaled, with labels."""
import glob, os
import numpy as np, cv2

A_DIR = "/svl/u/ravenh/fast_roborender/videogen_dataset"
B_DIR = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
OUT = "/tmp/ab_compare"
os.makedirs(OUT, exist_ok=True)

b_files = sorted(glob.glob(f"{B_DIR}/chunk_*.npz"))
# pick 6 spread across the set
picks = [b_files[i] for i in [0, len(b_files)//5, 2*len(b_files)//5,
                              3*len(b_files)//5, 4*len(b_files)//5, len(b_files)-1]]
H, W = 216, 384
made = 0
for bf in picks:
    name = os.path.basename(bf)
    af = os.path.join(A_DIR, name)
    if not os.path.exists(af):
        continue
    with np.load(bf) as z:
        if "rgb_gen" not in z.files: continue
        b_rgb = z["rgb_gen"]; prompt = str(z["prompt"])
    with np.load(af) as z:
        if "rgb_gen" not in z.files: continue
        a_rgb = z["rgb_gen"]
    T = min(len(a_rgb), len(b_rgb))
    out = os.path.join(OUT, name.replace(".npz", "_AB.mp4"))
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W*2, H+22))
    for t in range(T):
        a = cv2.resize(a_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(b_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(a, "A: old depth, fixed prompt", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        cv2.putText(b, "B: sharp depth, varied prompt", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        row = np.concatenate([a, b], axis=1)
        bar = np.zeros((22, W*2, 3), np.uint8)
        cv2.putText(bar, prompt[:130], (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
        wr.write(cv2.cvtColor(np.concatenate([row, bar], axis=0), cv2.COLOR_RGB2BGR))
    wr.release()
    made += 1
    print("wrote", out)
print(f"AB_COMPARE_DONE {made}")
