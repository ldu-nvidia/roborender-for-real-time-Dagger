"""5 vs 20 vs 50 denoising-step A/B for Image3F, FT-FIXED conditioning.

Runs inside the warm server process (session/np/Image/Image3FRequest/RESULT in scope).
5-step = production path (deployment timestep_schedule + TeaCache 0.3).
20/50-step = scheduler default schedule, TeaCache disabled (calibration is 5-step only).
Reference for PSNR/SSIM = 50-step teacher video model frames.
"""
import time, json
from pathlib import Path

W, H = 416, 240
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
DATA = Path("/svl/u/ravenh/fast_roborender/videogen_dataset")
TEACH = Path("/svl/u/ravenh/fast_roborender/teacher_fullres")
OUT = Path("/svl/u/ravenh/fast_roborender/step_ab_out")
OUT.mkdir(exist_ok=True)

CHUNKS = ["chunk_16222830_0000051", "chunk_16222830_0000501"]
FRAMES = [10, 40, 70]
STEPS = [5, 20, 50]

DROID_PARAMS = {
    "head_color": dict(raw_clip=(0.0, 10.0), inv_clip=(1.0, 20.0), raw_pct=None),
    "hand_left": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
    "hand_right": dict(raw_clip=(0.0, 5.0), inv_clip=(1.0, 100.0), raw_pct=(1, 99)),
}


def h5v4(d, raw_clip, inv_clip, raw_pct=None, invalid_fill="far"):
    d = d.astype(np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    lo, hi = raw_clip
    if raw_pct is not None:
        v = d[~bad]
        if v.size:
            lo = max(lo, float(np.percentile(v, raw_pct[0])))
            hi = min(hi, float(np.percentile(v, raw_pct[1])))
    d[bad] = hi if invalid_fill == "far" else lo
    d = np.clip(d, lo, hi)
    inv = 1.0 / (d + 1e-3)
    inv = np.clip(inv, inv_clip[0], inv_clip[1])
    lg = np.log(inv)
    lg = (lg - lg.min()) / max(lg.max() - lg.min(), 1e-6)
    return (lg * 255).astype(np.uint8)


def to_pil(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a.astype(np.uint8))


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    # global-window SSIM on grayscale, adequate for A/B ranking
    a = a.astype(np.float64).mean(-1); b = b.astype(np.float64).mean(-1)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


rows = []
panels = {}
sched_info = {}

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
        req = Image3FRequest(prompt=prompt, depth=depth, mask=mask,
                             previous_rgb=prev, seed=1234)
        ref = {v: anchor[f"gen__{v}"][t] for _, v in VIEW_MAP}

        for ns in STEPS:
            trace = {}
            torch = __import__("torch")
            if ns == 5:
                t0 = time.perf_counter()
                imgs, timing = session.generate(req, output_type="rgb")
                ms = (time.perf_counter() - t0) * 1000
                stacked = np.asarray(imgs[0])
            else:
                t0 = time.perf_counter()
                out = session.pipe(
                    prompt=prompt, negative_prompt="",
                    control_video_views=[[i] for i in depth],
                    control_video_2_views=[[i] for i in mask],
                    reference_image_views=list(prev),
                    height=H * 3, width=W, num_frames=1, num_views=3,
                    seed=1234, cfg_scale=session.cfg_scale,
                    num_inference_steps=ns, timestep_schedule=None,
                    tea_cache_l1_thresh=None, tea_cache_model_id=None,
                    enable_static_cache=True, enable_prompt_cache=True,
                    benchmark_trace=trace, return_latents=False,
                    tiled=False, progress_bar_cmd=lambda v: v,
                )
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                stacked = np.asarray(out[0])
                sched_info[ns] = trace.get("actual_sigmas")

            per = {}
            for vi, (_, view) in enumerate(VIEW_MAP):
                fr = stacked[vi * H:(vi + 1) * H]
                per[view] = fr
                rows.append(dict(chunk=cname, frame=t, steps=ns, view=view,
                                 ms=ms, psnr=psnr(fr, ref[view]),
                                 ssim=ssim(fr, ref[view])))
            panels[(cname, t, ns)] = np.concatenate([per[v] for _, v in VIEW_MAP], 0)
            print(f"{cname} t={t} steps={ns} {ms:.0f}ms", flush=True)

        # panel: rows = views, cols = teacher | 5 | 20 | 50
        cols = [np.concatenate([ref[v] for _, v in VIEW_MAP], 0)] + \
               [panels[(cname, t, s)] for s in STEPS]
        Image.fromarray(np.concatenate(cols, 1)).save(
            OUT / f"{cname}_t{t:03d}_teacher_5_20_50.png")

summary = {}
for ns in STEPS:
    sel = [r for r in rows if r["steps"] == ns]
    warm = [r["ms"] for r in sel if not (r["chunk"] == CHUNKS[0] and r["frame"] == FRAMES[0])]
    summary[ns] = dict(
        mean_ms=float(np.mean(warm)) if warm else float(np.mean([r["ms"] for r in sel])),
        psnr=float(np.mean([r["psnr"] for r in sel])),
        ssim=float(np.mean([r["ssim"] for r in sel])),
        psnr_by_view={v: float(np.mean([r["psnr"] for r in sel if r["view"] == v]))
                      for _, v in VIEW_MAP},
    )
(OUT / "step_ab_rows.json").write_text(json.dumps(rows, indent=1))
(OUT / "step_ab_summary.json").write_text(json.dumps(summary, indent=1))
RESULT["summary"] = summary
RESULT["sigmas"] = {k: (v[:6] if v else None) for k, v in sched_info.items()}
print(json.dumps(summary, indent=1), flush=True)
