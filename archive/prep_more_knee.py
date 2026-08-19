import glob, os, numpy as np, sys
sys.path.insert(0, "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender")
from renormalize_depth import h5v4_frame_seq
B = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_test_dataset"
DST = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_knee_dataset"
files = sorted(glob.glob(f"{B}/chunk_*.npz"))
picks = [files[i] for i in range(5, 200, 16)][:12]
n = 0
for src in picks:
    nm = os.path.basename(src)
    if os.path.exists(os.path.join(DST, nm)):
        continue
    with np.load(src) as z:
        data = {k: z[k].copy() for k in z.files}
    raw = data["depth_raw__head_color"].astype(np.float32)
    data["depth_ctrl__head_color"] = h5v4_frame_seq(
        raw, [0.05, 10], [1, 100], None, "far",
        norm_range=[0.05, 2.6], knee=[8.0, 140.0])
    data.pop("rgb_gen", None)
    out = os.path.join(DST, nm)
    np.savez_compressed(out + ".tmp.npz", **data); os.replace(out + ".tmp.npz", out)
    n += 1
print(f"MORE_KNEE_PREPPED {n}")
