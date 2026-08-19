#!/usr/bin/env python3
"""Multi-mode Image3F eval: runs several prev-conditioning modes in ONE session.

Modes (comma list via --prev_modes, zero always runs last):
  fixed = anchor frame 0 (from --init_prev_dir) as previous_rgb for EVERY frame
          (the finetune's fixed-lag-256 contract for t<256)
  chain = output frame 0 = anchor frame; frames 1..T-1 conditioned on own prev output
  black = black image as previous_rgb every frame
  zero  = zeroed reference latents + CLIP tokens (training dropout condition)

Output tags: {tag_prefix}{mode}, e.g. ft_fixed -> chunk_*_gen_ft_fixed.npz
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

STAGE_ROOT = Path(os.environ.get(
    "ROBORENDER_STAGE_ROOT",
    "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale",
))
ROBORENDER_ROOT = Path(
    os.environ.get("ROBORENDER_ROOT", Path(__file__).resolve().parents[1])
)
BUNDLE = Path(os.environ.get(
    "IMAGE3F_ROOT", STAGE_ROOT / "assets/roborender_image_model_accelerated"
))
sys.path.insert(0, str(BUNDLE))

from runtime import Image3FRequest, Image3FSession  # noqa: E402

VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
W, H = 416, 240


def h5v4(d, raw_clip, inv_clip, raw_pct=None, invalid_fill="far"):
    d = d.astype(np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    d[bad] = raw_clip[1] if invalid_fill == "far" else raw_clip[0]
    if raw_pct is not None:
        lo, hi = np.percentile(d, raw_pct)
        d = np.clip(d, lo, hi)
    d = np.clip(d, raw_clip[0], raw_clip[1])
    dn = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    inv = 1.0 / (dn + 1e-3)
    inv = np.clip(inv, inv_clip[0], inv_clip[1])
    inv = np.log(inv)
    inv = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
    return (inv * 255).astype(np.uint8)


DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0)),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
}


def to_pil(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a).convert("RGB").resize((W, H), Image.BILINEAR)


def apply_zero_patch():
    import torch
    from diffsynth.pipelines import wan_video as _wv

    def _zero_ref(self, pipe, reference_image_views, reference_image, height, width, num_views):
        per_view_h = height // num_views
        lat = torch.zeros(1, 16, 1, (per_view_h // 8) * num_views, width // 8,
                          dtype=pipe.torch_dtype, device=pipe.device)
        clip = None
        if pipe.image_encoder is not None:
            clip = torch.zeros(1, 257 * num_views, 1280,
                               dtype=pipe.torch_dtype, device=pipe.device)
        return {"reference_latents": lat, "clip_feature": clip}

    _wv.WanVideoUnit_FunReference_MultiView.process = _zero_ref
    print("patched reference unit -> zeroed conditioning", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev_modes", default="fixed,chain,zero")
    ap.add_argument("--tag_prefix", default="ft_")
    ap.add_argument("--depth_variant", choices=["droid", "bctrl"], default="droid")
    ap.add_argument("--init_prev_dir", default=str(ROBORENDER_ROOT / "teacher_fullres"))
    ap.add_argument("--model_root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_frames", type=int, default=81)
    ap.add_argument("--anchor_index", type=int, default=0, help="teacher frame index used as fixed reference")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.prev_modes.split(",") if m.strip()]
    modes.sort(key=lambda m: m == "zero")  # zero last (its patch is global)

    print("loading Image3FSession ...", flush=True)
    skw = dict(memory_mode="auto", vae_mode="untiled")
    if args.model_root:
        skw["model_root"] = args.model_root
    session = Image3FSession(**skw)

    for mode in modes:
        if mode == "zero":
            apply_zero_patch()
        tag = f"{args.tag_prefix}{mode}"
        all_times = []
        for cpath in args.chunks:
            d = np.load(cpath, allow_pickle=True)
            name = Path(cpath).stem
            prompt = str(d["prompt"])
            T = min(int(d["depth_ctrl__head_color"].shape[0]), args.max_frames)

            depth_seq = {}
            for _, view in VIEW_MAP:
                if args.depth_variant == "droid":
                    depth_seq[view] = h5v4(d[f"depth_raw__{view}"].astype(np.float32)[:T],
                                           **DROID_PARAMS[view])
                else:
                    depth_seq[view] = d[f"depth_ctrl__{view}"][:T]

            prev = {v: Image.new("RGB", (W, H)) for _, v in VIEW_MAP}
            gen_frames = {v: [] for _, v in VIEW_MAP}
            t_start = 0
            if mode in ("fixed", "chain"):
                anchor = np.load(Path(args.init_prev_dir) / f"{name}_teacher3v.npz",
                                 allow_pickle=True)
                for _, v in VIEW_MAP:
                    prev[v] = Image.fromarray(anchor[f"gen__{v}"][args.anchor_index])
                if mode == "chain":
                    for _, v in VIEW_MAP:
                        gen_frames[v].append(anchor[f"gen__{v}"][args.anchor_index])
                    t_start = 1
                print(f"[{tag}/{name}] anchored on teacher frame {args.anchor_index}", flush=True)

            times = []
            for t in range(t_start, T):
                req = Image3FRequest(
                    prompt=prompt,
                    depth=tuple(to_pil(depth_seq[v][t]) for _, v in VIEW_MAP),
                    mask=tuple(to_pil(d[f"rgb_robot__{v}"][t]) for _, v in VIEW_MAP),
                    previous_rgb=tuple(prev[v] for _, v in VIEW_MAP),
                    seed=args.seed,
                )
                t0 = time.perf_counter()
                imgs, timing = session.generate(req, output_type="rgb")
                wall = (time.perf_counter() - t0) * 1000
                stacked = np.asarray(imgs[0])
                for vi, (_, view) in enumerate(VIEW_MAP):
                    fr = stacked[vi * H:(vi + 1) * H]
                    gen_frames[view].append(fr)
                    if mode == "chain":
                        prev[view] = Image.fromarray(fr)
                ms = timing.get("pipeline_ms", wall) if isinstance(timing, dict) else wall
                times.append(float(ms))
                if t % 20 == 0:
                    print(f"[{tag}/{name}] frame {t}/{T} {ms:.1f}ms", flush=True)

            np.savez_compressed(out / f"{name}_gen_{tag}.npz",
                                **{f"gen__{v}": np.stack(gen_frames[v]) for _, v in VIEW_MAP},
                                **{f"depth__{v}": depth_seq[v] for _, v in VIEW_MAP},
                                prompt=prompt)
            warm = times[1:]
            all_times += warm
            print(f"[{tag}/{name}] warm mean {np.mean(warm):.1f}ms p90 {np.percentile(warm,90):.1f}ms",
                  flush=True)
        print(json.dumps({"mode": mode, "tag": tag,
                          "mean_ms": float(np.mean(all_times)),
                          "p90_ms": float(np.percentile(all_times, 90))}), flush=True)
        print(f"MODE_DONE:{mode}", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
