"""Ablation separating step-count from schedule shape / static cache.

Configs (all FT-FIXED, seed 1234, TeaCache off except the production baseline):
  prod5        : 5 steps, deployment schedule, TeaCache 0.3      (baseline, session.generate)
  lin5         : 5 steps, scheduler-default (linear) schedule    -> isolates schedule at 5 steps
  prodshape50  : 50 steps, deployment trajectory interpolated to 50 -> isolates step count
  lin50        : 50 steps, default linear schedule               -> repeat of first A/B
  lin50_nosc   : 50 steps, default linear, static cache OFF      -> isolates static cache
"""
import time, json
from pathlib import Path
import numpy as _np

import sys as _sys
if "/svl/u/ravenh/fast_roborender/tools" not in _sys.path:
    _sys.path.insert(0, "/svl/u/ravenh/fast_roborender/tools")
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
DATA = Path("/svl/u/ravenh/fast_roborender/videogen_dataset")
TEACH = Path("/svl/u/ravenh/fast_roborender/teacher_fullres")
OUT = Path("/svl/u/ravenh/fast_roborender/step_ab_out")
OUT.mkdir(exist_ok=True)

CHUNKS = ["chunk_16222830_0000051", "chunk_16222830_0000501"]
FRAMES = [10, 40, 70]
PROD_SCHED = [1.0, 0.7539682388305664, 0.6122448446350098,
              0.44871795177459717, 0.24193546175956726, 0.0]



def interp_sched(base, n):
    """Resample a monotone sigma trajectory to n+1 points, ending exactly at 0."""
    x = _np.linspace(0.0, 1.0, len(base))
    xi = _np.linspace(0.0, 1.0, n + 1)
    s = _np.interp(xi, x, _np.asarray(base, dtype=_np.float64))
    s[-1] = 0.0
    for i in range(1, len(s)):           # enforce strict non-increase
        s[i] = min(s[i], s[i - 1])
    return [float(v) for v in s]




def to_pil(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a.astype(np.uint8))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = a.astype(np.float64).mean(-1); b = b.astype(np.float64).mean(-1)
    mu_a, mu_b, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def mean_rgb(x):
    return [float(v) for v in x.reshape(-1, 3).mean(0)]


CONFIGS = [
    ("prod5",       dict(steps=5,  sched="prod",   static=True,  tea=True)),
    ("lin5",        dict(steps=5,  sched="linear", static=True,  tea=False)),
    ("prodshape50", dict(steps=50, sched="prod",   static=True,  tea=False)),
    ("lin50",       dict(steps=50, sched="linear", static=True,  tea=False)),
    ("lin50_nosc",  dict(steps=50, sched="linear", static=False, tea=False)),
]

torch = __import__("torch")
rows, panels = [], {}

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
        ref = {v: anchor[f"gen__{v}"][t] for _, v in VIEW_MAP}

        for name, cfg in CONFIGS:
            if name == "prod5":
                req = Image3FRequest(prompt=prompt, depth=depth, mask=mask,
                                     previous_rgb=prev, seed=1234)
                t0 = time.perf_counter()
                imgs, _ = session.generate(req, output_type="rgb")
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                stacked = np.asarray(imgs[0])
            else:
                sched = (interp_sched(PROD_SCHED, cfg["steps"])
                         if cfg["sched"] == "prod" else None)
                t0 = time.perf_counter()
                out = session.pipe(
                    prompt=prompt, negative_prompt="",
                    control_video_views=[[i] for i in depth],
                    control_video_2_views=[[i] for i in mask],
                    reference_image_views=list(prev),
                    height=H * 3, width=W, num_frames=1, num_views=3,
                    seed=1234, cfg_scale=session.cfg_scale,
                    num_inference_steps=cfg["steps"], timestep_schedule=sched,
                    tea_cache_l1_thresh=(session.tea_cache_threshold if cfg["tea"] else None),
                    tea_cache_model_id=("Image3FJoint-5step" if cfg["tea"] else None),
                    enable_static_cache=cfg["static"], enable_prompt_cache=True,
                    return_latents=False, tiled=False,
                    progress_bar_cmd=lambda v: v,
                )
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                stacked = np.asarray(out[0])

            per = {}
            for vi, (_, view) in enumerate(VIEW_MAP):
                fr = stacked[vi * H:(vi + 1) * H]
                per[view] = fr
                rows.append(dict(chunk=cname, frame=t, cfg=name, view=view, ms=ms,
                                 psnr=psnr(fr, ref[view]), ssim=ssim(fr, ref[view]),
                                 rgb=mean_rgb(fr), rgb_ref=mean_rgb(ref[view])))
            panels[(cname, t, name)] = np.concatenate([per[v] for _, v in VIEW_MAP], 0)
            print(f"{cname} t={t} {name} {ms:.0f}ms", flush=True)

        cols = [np.concatenate([ref[v] for _, v in VIEW_MAP], 0)] + \
               [panels[(cname, t, n)] for n, _ in CONFIGS]
        Image.fromarray(np.concatenate(cols, 1)).save(
            OUT / f"abl_{cname}_t{t:03d}_teacher_{'_'.join(n for n, _ in CONFIGS)}.png")

summary = {}
for name, _ in CONFIGS:
    sel = [r for r in rows if r["cfg"] == name]
    drift = float(np.mean([np.abs(np.array(r["rgb"]) - np.array(r["rgb_ref"])).mean()
                           for r in sel]))
    summary[name] = dict(mean_ms=float(np.mean([r["ms"] for r in sel])),
                         psnr=float(np.mean([r["psnr"] for r in sel])),
                         ssim=float(np.mean([r["ssim"] for r in sel])),
                         mean_abs_rgb_drift=drift)
(OUT / "step_abl_rows.json").write_text(json.dumps(rows, indent=1))
(OUT / "step_abl_summary.json").write_text(json.dumps(summary, indent=1))
RESULT["summary"] = summary
print(json.dumps(summary, indent=1), flush=True)
