"""Close the last confound: TeaCache on/off crossed with schedule, at 5 steps."""
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
CHUNKS = ["chunk_16222830_0000051", "chunk_16222830_0000501"]
FRAMES = [10, 40, 70]
PROD_SCHED = [1.0, 0.7539682388305664, 0.6122448446350098,
              0.44871795177459717, 0.24193546175956726, 0.0]




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
    depth_seq = {v: h5v4_view(d[f"depth_raw__{v}"].astype(np.float32), v)
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
