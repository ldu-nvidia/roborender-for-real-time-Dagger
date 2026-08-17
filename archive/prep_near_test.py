import glob, os, numpy as np, sys
sys.path.insert(0, "/svl/u/ravenh/fast_roborender")
from renormalize_depth import h5v4_frame_seq
B = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
# near-object chunks found earlier
NAMES = ["chunk_16222830_0000001.npz","chunk_16222830_0000002.npz",
         "chunk_16222830_0000006.npz","chunk_16222830_0000008.npz"]
CONDS = {
    "knee": dict(norm_range=[0.05,2.6], inv=[1,100], knee=[8.0,140.0]),
    "oldd": dict(norm_range=[0.05,3.0], inv=[1,100], knee=None),
}
for cond, p in CONDS.items():
    dst = f"/svl/u/ravenh/fast_roborender/ab_{cond}_dataset"
    os.makedirs(dst, exist_ok=True)
    for nm in NAMES:
        src = os.path.join(B, nm)
        with np.load(src) as z:
            data = {k: z[k].copy() for k in z.files}
        raw = data["depth_raw__head_color"].astype(np.float32)
        data["depth_ctrl__head_color"] = h5v4_frame_seq(
            raw, [0.05,10], p["inv"], None, "far",
            norm_range=p["norm_range"], knee=p["knee"])
        data.pop("rgb_gen", None)
        out = os.path.join(dst, nm)
        np.savez_compressed(out+".tmp.npz", **data); os.replace(out+".tmp.npz", out)
    print(f"{cond}: {len(NAMES)} chunks prepped -> {dst}")
print("NEAR_PREP_DONE")
