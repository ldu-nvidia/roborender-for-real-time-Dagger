#!/usr/bin/env python3
"""Build side-by-side videos + quality metrics: Image3F vs video model.

Per chunk/variant: for each view, a row of [depth cond | image3f gen];
head row also appends the video-model rgb_gen (upscaled). Also computes
PSNR/SSIM (head, vs video-model rgb_gen at 108x192) and temporal
consistency (mean abs frame diff).
"""
import argparse, json, os
from pathlib import Path

import cv2
import numpy as np

VIEWS = ["head_color", "hand_left", "hand_right"]
W, H = 416, 240


def gray3(a):
    return np.stack([a] * 3, -1)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return float(m.mean())


def temporal(frames):
    f = frames.astype(np.float32)
    return float(np.mean(np.abs(f[1:] - f[:-1])))


def write_mp4(path, frames, fps=15):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr[:, :, ::-1])
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_npz", nargs="+", required=True, help="*_gen_<variant>.npz from image3f_eval.py")
    root = Path(os.environ.get("ROBORENDER_ROOT", Path(__file__).resolve().parents[1]))
    ap.add_argument("--pool", default=str(root / "videogen_dataset"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for gp in args.gen_npz:
        gp = Path(gp)
        stem = gp.stem  # chunk_..._gen_droid
        chunk_name = stem.split("_gen_")[0]
        variant = stem.split("_gen_")[1]
        g = np.load(gp, allow_pickle=True)
        src = np.load(Path(args.pool) / f"{chunk_name}.npz", allow_pickle=True)
        vid_gen = src["rgb_gen"]  # (81,108,192,3) head view, video model
        T = g["gen__head_color"].shape[0]

        rows_all = []
        for t in range(T):
            rows = []
            for v in VIEWS:
                cells = [gray3(g[f"depth__{v}"][t]), g[f"gen__{v}"][t]]
                if v == "head_color":
                    vg = cv2.resize(vid_gen[min(t, len(vid_gen) - 1)], (W, H), interpolation=cv2.INTER_LINEAR)
                    cells.append(vg)
                else:
                    cells.append(np.zeros((H, W, 3), np.uint8))
                rows.append(np.concatenate(cells, 1))
            fr = np.concatenate(rows, 0)
            for i, lbl in enumerate(["head: depth | IMAGE3F | video-model",
                                     "hand_left: depth | IMAGE3F",
                                     "hand_right: depth | IMAGE3F"]):
                cv2.putText(fr, lbl, (6, i * H + 18), 0, 0.55, (0, 255, 255), 1)
            rows_all.append(fr)
        write_mp4(out / f"{chunk_name}_{variant}_cmp.mp4", rows_all)

        head = g["gen__head_color"]
        head_small = np.stack([cv2.resize(f, (vid_gen.shape[2], vid_gen.shape[1]),
                                          interpolation=cv2.INTER_AREA) for f in head])
        n = min(len(head_small), len(vid_gen))
        metrics[f"{chunk_name}_{variant}"] = {
            "psnr_vs_videomodel_head": float(np.mean([psnr(head_small[t], vid_gen[t]) for t in range(1, n)])),
            "ssim_vs_videomodel_head": float(np.mean([ssim(head_small[t], vid_gen[t]) for t in range(1, n)])),
            "temporal_madiff_image3f_head": temporal(head),
            "temporal_madiff_videomodel_head": temporal(vid_gen),
            "prompt": str(g["prompt"])[:200],
        }
        print("done", chunk_name, variant, flush=True)

    with open(out / "quality_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    print("CMP_DONE", flush=True)


if __name__ == "__main__":
    main()
