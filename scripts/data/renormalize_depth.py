#!/usr/bin/env python
"""Re-normalize depth_ctrl from saved raw depth WITHOUT recollecting.

Reads depth_raw__{view} (float16 metric) from chunks, applies the h5_v4
recipe with the given params + invalid-pixel fill, overwrites depth_ctrl__{view},
and strips rgb_gen so the renderer regenerates. Requires chunks collected with
videogen.save_raw_depth: true.

Usage:
  python scripts/data/renormalize_depth.py --dataset_dir DIR \
      --view_params '{"head_color":{"raw_clip":[0,10],"inv_clip":[1,100],"raw_pct":[1,99]},
                      "hand_left":{"raw_clip":[0,1.5],"inv_clip":[0.7,100],"raw_pct":[1,99]},
                      "hand_right":{"raw_clip":[0,1.5],"inv_clip":[0.7,100],"raw_pct":[1,99]}}' \
      --invalid_fill far   # 'far' = dark-gray plateau, 'black' = 0 (training default)
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def h5v4_frame_seq(d, raw_clip, inv_clip, raw_pct, invalid_fill, norm_range=None, knee=None):
    # d: [T, H, W] metric depth, one video (=one chunk here)
    rlo, rhi = raw_clip
    ilo, ihi = inv_clip
    finite = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape, dtype=np.uint8)
    if not finite.any():
        return out
    blo, bhi = rlo, rhi
    if raw_pct is not None:
        valid = d[finite]
        m = (valid >= rlo) & (valid <= rhi)
        if m.any():
            valid = valid[m]
        blo = float(np.percentile(valid, raw_pct[0]))
        bhi = float(np.percentile(valid, raw_pct[1]))
        if bhi <= blo:
            bhi = blo + 1e-3
    dc = np.where(finite, np.clip(d, blo, bhi), bhi)
    valid = dc[finite]
    if norm_range is not None:
        # fixed normalization bounds: a close object no longer anchors the
        # range to white (keeps object shape). far bound still bounds bg.
        lo, hi = float(norm_range[0]), float(norm_range[1])
    else:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        out[:] = 128
        out[~finite] = 0 if invalid_fill == "black" else out[~finite]
        return out
    dn = np.clip((dc - lo) / (hi - lo), 0.0, 1.0)
    inv = np.clip(1.0 / (dn + 1e-3), ilo, ihi)
    if knee is not None:
        # two-segment log mapping: mid/far [ilo..inv_knee] -> [0..b_knee]
        # (steeper mid contrast), near [inv_knee..ihi] -> [b_knee..255]
        # (full near gradation, no flat plateau on close objects).
        inv_knee, b_knee = float(knee[0]), float(knee[1])
        lowseg = np.clip(np.log(np.clip(inv, ilo, inv_knee) / ilo)
                         / np.log(inv_knee / ilo), 0.0, 1.0) * b_knee
        hiseg = np.clip(np.log(np.clip(inv, inv_knee, ihi) / inv_knee)
                        / np.log(ihi / inv_knee), 0.0, 1.0) * (255.0 - b_knee)
        out = np.where(inv <= inv_knee, lowseg, b_knee + hiseg).astype(np.uint8)
    else:
        x = np.log(inv)
        p_lo, p_hi = np.log(ilo), np.log(ihi)
        norm = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        out = (norm * 255.0).astype(np.uint8)
    if invalid_fill == "black":
        out[~finite] = 0
    # 'far': leave invalid pixels at their far-clip normalized value (already computed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--view_params", required=True)
    ap.add_argument("--invalid_fill", choices=["far", "black"], default="far")
    args = ap.parse_args()
    vp = json.loads(args.view_params)

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "chunk_*.npz")))
    n = 0
    for f in files:
        try:
            with np.load(f) as z:
                data = {k: z[k].copy() for k in z.files}
        except Exception as e:
            print(f"skip {f}: {e}"); continue
        raw_keys = [k for k in data if k.startswith("depth_raw__")]
        # native quantized raw (depth_rawq__): dequantize to meters, prefer it
        for qk in [k for k in data if k.startswith("depth_rawq__")]:
            view = qk.split("__")[1]
            shift, dmin, dmax = 3.5, 0.0, 10.0
            ln, lx = np.log(dmin + shift), np.log(dmax + shift)
            d = np.exp(data[qk].astype(np.float32) / 16383.0 * (lx - ln) + ln) - shift
            data[f"depth_raw__{view}"] = d.astype(np.float16)
            if f"depth_raw__{view}" not in raw_keys:
                raw_keys.append(f"depth_raw__{view}")
        if not raw_keys:
            continue
        for rk in raw_keys:
            view = rk.split("__")[1]
            if view not in vp:
                continue
            p = vp[view]
            d = data[rk].astype(np.float32)  # [T,H,W]
            data[f"depth_ctrl__{view}"] = h5v4_frame_seq(
                d, p["raw_clip"], p["inv_clip"], p.get("raw_pct"), args.invalid_fill,
                norm_range=p.get("norm_range"), knee=p.get("knee"))
        data.pop("rgb_gen", None)  # force re-render
        tmp = f + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, f)
        n += 1
        if n % 20 == 0:
            print(f"renormalized {n}", flush=True)
    print(f"RENORM_DONE {n} chunks", flush=True)


if __name__ == "__main__":
    main()
