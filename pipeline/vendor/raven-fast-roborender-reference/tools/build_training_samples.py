import numpy as np, glob, random, os, cv2, imageio

POOL = "/svl/u/ravenh/fast_roborender/videogen_dataset"
OUT  = "/svl/u/ravenh/fast_roborender/training_samples_b"
os.makedirs(OUT, exist_ok=True)
VIEWS = ["hand_left", "head_color", "hand_right"]
H, W = 240, 416

def gray3(a):
    return np.stack([a]*3, -1)

fs = sorted(glob.glob(os.path.join(POOL, "chunk_*.npz")))
random.seed(42)
picks = random.sample(fs, 6)
for f in picks:
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f).replace(".npz", "")
    gen = d["rgb_gen"]  # (81,108,192,3)
    T = gen.shape[0]
    frames = []
    for t in range(T):
        row1 = np.concatenate([gray3(d[f"depth_ctrl__{v}"][t]) for v in VIEWS], 1)
        row2 = np.concatenate([d[f"rgb_robot__{v}"][t] for v in VIEWS], 1)
        g = cv2.resize(gen[t], (W*2, int(W*2*gen.shape[1]/gen.shape[2])), interpolation=cv2.INTER_NEAREST)
        pad_h = g.shape[0]
        row3 = np.zeros((pad_h, W*3, 3), np.uint8)
        x0 = (W*3 - g.shape[1])//2
        row3[:, x0:x0+g.shape[1]] = g
        fr = np.concatenate([row1, row2, row3], 0)
        cv2.putText(fr, "depth_ctrl (B) L|head|R", (6, 18), 0, 0.5, (255,255,0), 1)
        cv2.putText(fr, "rgb_robot cond", (6, H+18), 0, 0.5, (255,255,0), 1)
        cv2.putText(fr, "rgb_gen (BC training input, head)", (6, 2*H+18), 0, 0.5, (0,255,0), 1)
        frames.append(fr)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(OUT, name + "_sample.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for fr in frames: vw.write(fr[:, :, ::-1])
    vw.release()
    with open(os.path.join(OUT, name + "_prompt.txt"), "w") as fh:
        fh.write(str(d["prompt"]) + "\n")
    print("wrote", name, flush=True)
print("DONE", flush=True)
