#!/usr/bin/env python3
"""Prove depth_recipe.h5v4_control is byte-identical to the trainer's _h5v4_control,
and quantify how far the old hand-copied recipe drifted.
"""
import os, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1])).resolve()
GR00T_ROOT = Path(
    os.environ.get("GR00T_ROOT", ROOT / "GR00T-VisualSim2Real")
).resolve()
sys.path.insert(0, str(ROOT / "tools"))
from depth_recipe import h5v4_control, load_view_params  # noqa: E402

TRAINER = GR00T_ROOT / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
if not TRAINER.is_file():
    raise SystemExit(
        f"trainer not found at {TRAINER}; apply transfer/0001..0003 or set GR00T_ROOT"
    )

# --- lift the trainer's method into a stub so we compare against the real code ---------
src = TRAINER.read_text(encoding="utf-8")
body = src.split('    def _h5v4_control(self, d: np.ndarray, view: str | None) -> np.ndarray:')[1]
body = body.split('\n    def ')[0]
ns = {'np': np}
exec('def _h5v4_control(self, d, view):' + body, ns)


class Stub:
    def __init__(self, per_view, fill='far'):
        self.vg_h5v4_per_view = per_view
        # the trainer's real built-in default (distill_trainer_videogen.py:74)
        self.vg_h5v4_default = {"raw_clip": [0.0, 10.0], "inv_clip": [1.0, 20.0]}
        self.vg_invalid_fill = fill
    _h5v4_control = ns['_h5v4_control']


P = load_view_params()
print('production params:')
for v, p in P.items():
    print(f'  {v:11s} {p}')

rng = np.random.RandomState(0)
stub = Stub(P)
ok = True
for view in P:
    for trial in range(3):
        d = rng.uniform(0.0, 8.0, size=(4, 2, 24, 32)).astype(np.float32)
        d[d < 0.4] = 0.0                       # invalid pixels
        d[0, 0, :3, :3] = np.nan               # non-finite
        if trial == 2:
            d[:, 1] = 0.0                      # a fully invalid env
        mine = h5v4_control(d, P[view])
        theirs = stub._h5v4_control(d, view)
        same = np.array_equal(mine, theirs)
        ok &= same
        print(f'  {view:11s} trial{trial}: identical={same} '
              f'maxdiff={int(np.abs(mine.astype(int) - theirs.astype(int)).max())}')

# --- how far off was the old recipe? --------------------------------------------------
def old_recipe(d, raw_clip, inv_clip, raw_pct=None):
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


print('\nold hand-copied recipe vs production, head_color:')
d = rng.uniform(0.05, 6.0, size=(8, 1, 48, 64)).astype(np.float32)
prod = h5v4_control(d, P['head_color'])[:, 0]
old = old_recipe(d[:, 0], (0.0, 10.0), (1.0, 20.0))
diff = np.abs(prod.astype(int) - old.astype(int))
print(f'  mean abs diff {diff.mean():.1f}/255   max {diff.max()}   '
      f'pixels differing >32: {100.0 * (diff > 32).mean():.1f}%')
print(f'  production mean {prod.mean():.1f} std {prod.std():.1f}; '
      f'old mean {old.mean():.1f} std {old.std():.1f}')

print('\nRESULT:', 'module matches production exactly' if ok else 'MISMATCH')
sys.exit(0 if ok else 1)
