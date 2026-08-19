import glob, os, numpy as np, cv2
B = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_test_dataset"
A = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/ab_adaptive_dataset"
OUT = "/tmp/adaptive_cmp"; os.makedirs(OUT, exist_ok=True)
H, W = 216, 384
for af in sorted(glob.glob(f"{A}/chunk_*.npz")):
    nm = os.path.basename(af)
    za = np.load(af); zb = np.load(os.path.join(B, nm))
    if "rgb_gen" not in za.files or "rgb_gen" not in zb.files: continue
    branch = str(za["adaptive_branch"]) if "adaptive_branch" in za.files else "?"
    a_rgb, b_rgb = za["rgb_gen"], zb["rgb_gen"]
    identical = np.array_equal(a_rgb, b_rgb)
    T = min(len(a_rgb), len(b_rgb))
    wr = cv2.VideoWriter(os.path.join(OUT, nm.replace(".npz", f"_adaptive-{branch}.mp4")),
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (W*2, H))
    for t in range(T):
        b = cv2.resize(b_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        a = cv2.resize(a_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(b, "pure B", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        cv2.putText(a, f"ADAPTIVE ({branch} branch)", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        wr.write(cv2.cvtColor(np.concatenate([b, a], axis=1), cv2.COLOR_RGB2BGR))
    wr.release()
    print(f"{nm}: branch={branch} identical_to_B={identical}")
print("ADAPTIVE_CMP_DONE")
