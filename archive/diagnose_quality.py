"""Isolate why GR00T renders look foggy: mask absence vs view duplication.
Sends the known-good drawer fixture through the live server under 3 conditions."""
import sys

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, "/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real")
from gr00t.rl.utils.videogen_client import VideoGenRenderClient  # noqa: E402

FIX = "/svl/u/ravenh/fast_roborender/roborender-acceleration/assets/fixtures/input/episode_00000000"
OUT = "/svl/u/ravenh/fast_roborender/quality_diag"
PROMPT = "A robot arm opens a drawer."

with open("/svl/u/ravenh/fast_roborender/raven_server_endpoint.txt") as f:
    host, port = f.read().split()
client = VideoGenRenderClient(host=host, port=int(port))


def load(path, n=81):
    frames = np.stack([f for f in iio.imiter(path)][:n])
    return frames


import os
os.makedirs(OUT, exist_ok=True)

ext1_d = load(f"{FIX}/ext1/depth/original_depth.mp4")
ext1_m = load(f"{FIX}/ext1/mask/rgb_robot.mp4")
wrist_d = load(f"{FIX}/wrist/depth/original_depth.mp4")
wrist_m = load(f"{FIX}/wrist/mask/rgb_robot.mp4")

conds = {
    "a_twoview_mask": ({"ext1": ext1_d, "wrist": wrist_d},
                       {"ext1": ext1_m, "wrist": wrist_m}),
    "b_twoview_nomask": ({"ext1": ext1_d, "wrist": wrist_d}, None),
    "c_dupview_nomask": ({"ext1": ext1_d, "wrist": ext1_d}, None),
}
for name, (depths, masks) in conds.items():
    rgb = client.render_chunk(depths, mask_per_view=masks, prompt=PROMPT, seed=42)
    ext1 = rgb["ext1"]
    iio.imwrite(f"{OUT}/{name}_f40.png", ext1[40])
    iio.imwrite(f"{OUT}/{name}.mp4", ext1, fps=15)
    print(f"{name}: mean={ext1.mean():.1f} std={ext1.std():.1f} "
          f"sat={np.abs(ext1.astype(int) - ext1.mean(axis=-1, keepdims=True)).mean():.2f}",
          flush=True)

# reference: the validated raw-sample output frame
ref = load("/svl/u/ravenh/fast_roborender/roborender-acceleration/outputs/raw_drawer_sample/generated_ext1.mp4")
iio.imwrite(f"{OUT}/ref_rawsample_f40.png", ref[40])
print(f"ref_rawsample: mean={ref.mean():.1f} std={ref.std():.1f} "
      f"sat={np.abs(ref.astype(int) - ref.mean(axis=-1, keepdims=True)).mean():.2f}")
print("DIAG_DONE")
