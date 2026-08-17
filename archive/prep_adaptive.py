import glob, os, numpy as np, sys
sys.path.insert(0, "/svl/u/ravenh/fast_roborender")
from renormalize_depth import h5v4_frame_seq
B = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
DST = "/svl/u/ravenh/fast_roborender/ab_adaptive_dataset"
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(f"{B}/chunk_*.npz"))
# pick 8: aim for 4 close-heavy + 4 far-only so both branches are exercised
close_picks, far_picks = [], []
for f in files:
    with np.load(f) as z:
        raw = z["depth_raw__head_color"].astype(np.float32)
    dmin = np.where(raw > 0.01, raw, np.inf).min(axis=(1, 2))
    nclose = (dmin < 0.18).sum()
    (close_picks if nclose > 5 else far_picks).append((f, raw, nclose))
    if len(close_picks) >= 4 and len(far_picks) >= 4:
        break
for f, raw, nclose in close_picks[:4] + far_picks[:4]:
    nm = os.path.basename(f)
    with np.load(f) as z:
        data = {k: z[k].copy() for k in z.files}
    if nclose > 5:  # knee branch (close-heavy chunk)
        ctrl = h5v4_frame_seq(raw, [0.05, 10], [1, 100], None, "far",
                              norm_range=[0.05, 2.6], knee=[8.0, 140.0])
        branch = "knee"
    else:           # B branch
        ctrl = h5v4_frame_seq(raw, [0.05, 10], [1, 40], None, "far",
                              norm_range=[0.05, 2.6])
        branch = "B"
    data["depth_ctrl__head_color"] = ctrl
    data["adaptive_branch"] = np.array(branch)
    data.pop("rgb_gen", None)
    out = os.path.join(DST, nm)
    np.savez_compressed(out + ".tmp.npz", **data); os.replace(out + ".tmp.npz", out)
    print(nm, "->", branch, f"({nclose} close frames)")
print("ADAPTIVE_PREP_DONE")
