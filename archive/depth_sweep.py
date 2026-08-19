import numpy as np, sys, glob
sys.path.insert(0, "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender")
from renormalize_depth import h5v4_frame_seq

files = sorted(glob.glob("/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/videogen_dataset/chunk_*.npz"))[:40]
raws = []
for f in files:
    r = np.load(f)["depth_raw__head_color"].astype(np.float32)
    if np.nanmin(np.where(r > 0.01, r, np.inf)) < 0.18:
        raws.append(r)
    if len(raws) >= 3:
        break
print(f"{len(raws)} near-object chunks for sweep")
raw = np.concatenate(raws)

def metrics(c, r):
    c = c.astype(float)
    near = (r > 0.01) & (r < 0.3)
    mid = (r >= 0.35) & (r <= 1.3)
    np_, mp_ = c[near], c[mid]
    return ((np_ > 250).mean(),
            np.percentile(np_, 90) - np.percentile(np_, 10),
            np.percentile(mp_, 90) - np.percentile(mp_, 10))

print(f"{'params':<30} {'nearSat':>8} {'nearSpread':>11} {'midSpread':>10}")
cands = [
    ((3.0, 100, None), "OLD"),
    ((2.6, 40, None), "B-bad"),
    ((2.6, 100, (8.0, 140.0)), "KNEE-8/140"),
    ((2.6, 100, (10.0, 150.0)), "KNEE-10/150"),
    ((2.4, 100, (8.0, 130.0)), "KNEE-8/130-r2.4"),
]
for (hi_r, ihi, knee), tag in cands:
    c = h5v4_frame_seq(raw, [0.05, 10], [1, ihi], None, "far",
                       norm_range=[0.05, hi_r], knee=knee)
    sat, ns, ms = metrics(c, raw)
    print(f"{tag:<16} range[0.05,{hi_r}] ihi={ihi:<4} {sat:>8.3f} {ns:>11.0f} {ms:>10.0f}")
