"""Standalone smoke test for the raven render server: sends one synthetic
81-frame depth chunk (moving gradient) and checks the response shape."""
import sys
import time

import numpy as np

sys.path.insert(0, "/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real")
from gr00t.rl.utils.videogen_client import VideoGenRenderClient  # noqa: E402

T, H, W = 81, 108, 192
yy = np.linspace(0, 1, H)[None, :, None]
tt = np.linspace(0, 1, T)[:, None, None]
depth = ((yy * 0.7 + tt * 0.3) * 255).astype(np.uint8) * np.ones((T, H, W), np.uint8)

client = VideoGenRenderClient(port=18913)
t0 = time.time()
rgb = client.render_chunk({"ext1": depth, "wrist": depth},
                          prompt="A robot arm in a room.", seed=0)
dt = time.time() - t0
for name, arr in rgb.items():
    assert arr.shape == (81, 240, 416, 3), f"{name}: {arr.shape}"
    assert arr.dtype == np.uint8
    print(f"{name}: {arr.shape} {arr.dtype} mean={arr.mean():.1f}")
print(f"CLIENT_TEST_PASS in {dt:.1f}s")
