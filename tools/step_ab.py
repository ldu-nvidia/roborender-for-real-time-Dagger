"""5 vs 20 vs 50 denoising-step A/B for Image3F, FT-FIXED conditioning.

Runs inside the warm server process (session/np/Image/Image3FRequest/RESULT in scope).
5-step = production path (deployment timestep_schedule + TeaCache 0.3).
20/50-step = scheduler default schedule, TeaCache disabled (calibration is 5-step only).
Reference for PSNR/SSIM = 50-step teacher video model frames.
"""
import time, json
from pathlib import Path

import os as _os
import sys as _sys
_ROOT = Path(_os.environ.get(
    "ROBORENDER_ROOT",
    "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender",
))
if str(_ROOT / "tools") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "tools"))
from depth_recipe import load_view_params as _load_view_params, h5v4_control as _h5v4_ctrl

# Per-view params come from the production config, so these renders match training.
# The previous hand-copied DROID_PARAMS inverted RAW depth and min-maxed per frame,
# while production normalizes with fixed norm_range first -- a 17/255 mean difference
# on head_color, worst where a close object anchored the range to white.
DROID_PARAMS = _load_view_params()


def h5v4(d, **_ignored):
    """[T,H,W] metric depth -> uint8 control, exactly as the trainer does it."""
    raise RuntimeError("call h5v4_view(d, view) so the production params are used")


def h5v4_view(d, view):
    return _h5v4_ctrl(d.astype("float32"), DROID_PARAMS[view])


W, H = 416, 240
VIEW_MAP = [("ext1", "head_color"), ("ext2", "hand_left"), ("wrist", "hand_right")]
DATA = _ROOT / "videogen_dataset"
TEACH = _ROOT / "teacher_fullres"
OUT = _ROOT / "step_ab_out"
OUT.mkdir(exist_ok=True)

CHUNKS = ["chunk_16222830_0000051", "chunk_16222830_0000501"]
FRAMES = [10, 40, 70]
STEPS = [5, 20, 50]





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

    depth_seq = {v: h5v4_view(d[f"depth_raw__{v}"].astype(np.float32), v)
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
