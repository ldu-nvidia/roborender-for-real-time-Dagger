"""Close the last confound: TeaCache on/off crossed with schedule, at 5 steps."""
import time, json
from pathlib import Path
import numpy as _np

W, H = 416, 240
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
DATA = Path("/svl/u/ravenh/fast_roborender/videogen_dataset")
TEACH = Path("/svl/u/ravenh/fast_roborender/teacher_fullres")
OUT = Path("/svl/u/ravenh/fast_roborender/step_ab_out")
CHUNKS = ["chunk_16222830_0000051", "chunk_16222830_0000501"]
FRAMES = [10, 40, 70]
PROD_SCHED = [1.0, 0.7539682388305664, 0.6122448446350098,
              0.44871795177459717, 0.24193546175956726, 0.0]
DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0), raw_pct=None),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
}


def h5v4(d, raw_clip, inv_clip, raw_pct=None):
    d = d.astype(np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    lo, hi = raw_clip
    if raw_pct is not None:
        v = d[~bad]
        if v.size:
            lo = max(lo, float(np.percentile(v, raw_pct[0])))
            hi = min(hi, float(np.percentile(v, raw_pct[1])))
    d[bad] = hi
    d = np.clip(d, lo, hi)
    lg = np.log(np.clip(1.0 / (d + 1e-3), inv_clip[0], inv_clip[1]))
    lg = (lg - lg.min()) / max(lg.max() - lg.min(), 1e-6)
    return (lg * 255).astype(np.uint8)


def to_pil(a):
    return Image.fromarray((np.stack([a] * 3, -1) if a.ndim == 2 else a).astype(np.uint8))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def drift(a, b):
    return float(np.abs(a.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)).mean())


CONFIGS = [("prod5_tea", "prod", True), ("prod5_notea", "prod", False),
           ("lin5_tea", "linear", True), ("lin5_notea", "linear", False)]
torch = __import__("torch")
rows = []

for cname in CHUNKS:
    d = np.load(DATA / f"{cname}.npz", allow_pickle=True)
    anchor = np.load(TEACH / f"{cname}_teacher3v.npz", allow_pickle=True)
    prompt = str(d["prompt"])
    prev = tuple(Image.fromarray(anchor[f"gen__{v}"][0]) for _, v in VIEW_MAP)
    depth_seq = {v: h5v4(d[f"depth_raw__{v}"].astype(np.float32), **DROID_PARAMS[v])
                 for _, v in VIEW_MAP}
    for t in FRAMES:
        depth = tuple(to_pil(depth_seq[v][t]) for _, v in VIEW_MAP)
        mask = tuple(to_pil(d[f"rgb_robot__{v}"][t]) for _, v in VIEW_MAP)
        ref = {v: anchor[f"gen__{v}"][t] for _, v in VIEW_MAP}
        for name, sched_kind, tea in CONFIGS:
            t0 = time.perf_counter()
            out = session.pipe(
                prompt=prompt, negative_prompt="",
                control_video_views=[[i] for i in depth],
                control_video_2_views=[[i] for i in mask],
                reference_image_views=list(prev),
                height=H * 3, width=W, num_frames=1, num_views=3,
                seed=1234, cfg_scale=session.cfg_scale, num_inference_steps=5,
                timestep_schedule=(list(PROD_SCHED) if sched_kind == "prod" else None),
                tea_cache_l1_thresh=(session.tea_cache_threshold if tea else None),
                tea_cache_model_id=("Image3FJoint-5step" if tea else None),
                enable_static_cache=True, enable_prompt_cache=True,
                return_latents=False, tiled=False, progress_bar_cmd=lambda v: v,
            )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            stacked = np.asarray(out[0])
            for vi, (_, view) in enumerate(VIEW_MAP):
                fr = stacked[vi * H:(vi + 1) * H]
                rows.append(dict(cfg=name, ms=ms, psnr=psnr(fr, ref[view]),
                                 drift=drift(fr, ref[view])))
            print(f"{cname} t={t} {name} {ms:.0f}ms", flush=True)

summary = {n: dict(mean_ms=float(np.mean([r["ms"] for r in rows if r["cfg"] == n])),
                   psnr=float(np.mean([r["psnr"] for r in rows if r["cfg"] == n])),
                   drift=float(np.mean([r["drift"] for r in rows if r["cfg"] == n])))
           for n, _, _ in CONFIGS}
(OUT / "step_tea_summary.json").write_text(json.dumps(summary, indent=1))
RESULT["summary"] = summary
print(json.dumps(summary, indent=1), flush=True)
