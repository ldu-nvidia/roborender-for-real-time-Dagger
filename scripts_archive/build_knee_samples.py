import glob, os, numpy as np, cv2
B = "/svl/u/ravenh/fast_roborender/ab_test_dataset"
K = "/svl/u/ravenh/fast_roborender/ab_knee_dataset"
OUT = "/tmp/knee_samples"; os.makedirs(OUT, exist_ok=True)
H, W = 216, 384
made = 0
for kf in sorted(glob.glob(f"{K}/chunk_*.npz")):
    nm = os.path.basename(kf)
    bf = os.path.join(B, nm)
    zk = np.load(kf)
    if "rgb_gen" not in zk.files or not os.path.exists(bf): continue
    zb = np.load(bf)
    if "rgb_gen" not in zb.files: continue
    k_rgb, b_rgb = zk["rgb_gen"], zb["rgb_gen"]
    prompt = str(zk["prompt"])
    T = min(len(k_rgb), len(b_rgb))
    wr = cv2.VideoWriter(os.path.join(OUT, nm.replace(".npz", "_BvK.mp4")),
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (W*2, H+22))
    for t in range(T):
        b = cv2.resize(b_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        k = cv2.resize(k_rgb[t], (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(b, "B (sharp, near-crush)", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)
        cv2.putText(k, "KNEE (final recipe)", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)
        row = np.concatenate([b, k], axis=1)
        bar = np.zeros((22, W*2, 3), np.uint8)
        cv2.putText(bar, prompt[:130], (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
        wr.write(cv2.cvtColor(np.concatenate([row, bar], axis=0), cv2.COLOR_RGB2BGR))
    wr.release(); made += 1
print(f"KNEE_SAMPLES_DONE {made}")
