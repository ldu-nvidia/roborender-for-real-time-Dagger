#!/usr/bin/env python
"""3-way near-object comparison: B (sharp depth) vs KNEE vs OLD depth,
all with identical varied prompts. Videos + closest-frame stills."""
import glob, os, numpy as np, cv2
import imageio.v3 as iio
B  = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_test_dataset"
K  = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_knee_dataset"
O  = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_oldd_dataset"
OUT = "/tmp/threeway"; os.makedirs(OUT, exist_ok=True)
H, W = 216, 384
stills = []
for nm in sorted(os.path.basename(f) for f in glob.glob(f"{K}/chunk_*.npz")):
    paths = {c: os.path.join(d, nm) for c, d in (("B",B),("KNEE",K),("OLD",O))}
    data = {}
    ok = True
    for c, p in paths.items():
        if not os.path.exists(p): ok = False; break
        z = np.load(p)
        if "rgb_gen" not in z.files: ok = False; break
        data[c] = z["rgb_gen"]
        if c == "KNEE":
            raw = z["depth_raw__head_color"].astype(np.float32)
    if not ok: continue
    T = min(len(v) for v in data.values())
    # video
    wr = cv2.VideoWriter(os.path.join(OUT, nm.replace(".npz", "_3way.mp4")),
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (W*3, H))
    for t in range(T):
        cols = []
        for c in ("B", "KNEE", "OLD"):
            img = cv2.resize(data[c][t], (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, c, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cols.append(img)
        wr.write(cv2.cvtColor(np.concatenate(cols, axis=1), cv2.COLOR_RGB2BGR))
    wr.release()
    # closest-frame still
    dmin = np.where(raw > 0.01, raw, np.inf).min(axis=(1,2))
    t = int(dmin.argmin())
    row = []
    for c in ("B", "KNEE", "OLD"):
        img = cv2.resize(data[c][t], (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{c} t{t} d={dmin[t]:.2f}m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        row.append(img)
    stills.append(np.concatenate(row, axis=1))
    print("built", nm)
iio.imwrite(os.path.join(OUT, "closest_frames.png"), np.concatenate(stills, axis=0))
print("THREEWAY_DONE")
