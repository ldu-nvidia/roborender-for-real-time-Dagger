"""Single source of truth for the h5_v4 depth control recipe used offline.

Mirrors TRLDistillTrainerVideoGen._h5v4_control exactly, and loads the per-view parameters
from the production experiment config, so analysis scripts cannot drift from training.

Why this exists: the eval/A-B scripts each carried a hand-copied DROID_PARAMS dict that
inverted RAW metric depth and min-maxed each frame, while production normalizes first with
fixed bounds (norm_range) and maps log-inverse with fixed log bounds. A close object
therefore anchored the normalization to white in the offline renders -- visible artifacts
that do not occur in training.

    from depth_recipe import load_view_params, h5v4_control
    P = load_view_params()                       # from wsdpt_image_dagger.yaml
    ctrl = h5v4_control(depth_TBHW, P["head_color"])
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1])).resolve()
GR00T_ROOT = Path(
    os.environ.get("GR00T_ROOT", ROOT / "GR00T-VisualSim2Real")
).resolve()
DEFAULT_CONFIG = str(
    GR00T_ROOT / "gr00t/rl/config/exp/loco_manip/wsdpt_image_dagger.yaml"
)
FALLBACK = {"raw_clip": [0.0, 10.0], "inv_clip": [1.0, 100.0], "raw_pct": [1, 99]}


def load_view_params(config_path: str = DEFAULT_CONFIG) -> dict:
    """Per-view h5_v4 params straight from the production config."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    vg = cfg["algo"]["config"]["videogen"]
    params = dict(vg.get("depth_h5v4_per_view") or {})
    if not params:
        raise RuntimeError(f"no depth_h5v4_per_view in {config_path}")
    return params


def h5v4_control(d: np.ndarray, params: dict, invalid_fill: str = "far") -> np.ndarray:
    """[T,B,H,W] or [T,H,W] metric depth -> uint8 control image, same shape.

    Verbatim port of _h5v4_control: clip (optionally to per-video percentiles), normalize
    to [0,1] with fixed norm_range when given, invert, clip in inverse space, then map log
    inverse through FIXED log bounds from inv_clip.
    """
    squeeze = d.ndim == 3
    if squeeze:
        d = d[:, None]

    rlo, rhi = (float(v) for v in params.get("raw_clip", FALLBACK["raw_clip"]))
    ilo, ihi = (float(v) for v in params.get("inv_clip", FALLBACK["inv_clip"]))
    raw_pct = params.get("raw_pct")
    norm_range = params.get("norm_range")

    out = np.zeros(d.shape, dtype=np.uint8)
    for b in range(d.shape[1]):
        db = d[:, b]
        finite = np.isfinite(db) & (db > 0)
        if not finite.any():
            continue
        blo, bhi = rlo, rhi
        if raw_pct is not None:
            valid_raw = db[finite]
            mask = (valid_raw >= rlo) & (valid_raw <= rhi)
            if mask.any():
                valid_raw = valid_raw[mask]
            blo = float(np.percentile(valid_raw, float(raw_pct[0])))
            bhi = float(np.percentile(valid_raw, float(raw_pct[1])))
            if bhi <= blo:
                bhi = blo + 1e-3
        dc = np.where(finite, np.clip(db, blo, bhi), bhi)
        valid = dc[finite]
        if norm_range is not None:
            lo, hi = float(norm_range[0]), float(norm_range[1])
        else:
            lo, hi = float(valid.min()), float(valid.max())
        if hi <= lo:
            ob = np.full(db.shape, 128, dtype=np.uint8)
            ob[~finite] = 0
            out[:, b] = ob
            continue
        dn = np.clip((dc - lo) / (hi - lo), 0.0, 1.0)
        inv = np.clip(1.0 / (dn + 1e-3), ilo, ihi)
        x = np.log(inv)
        p_lo, p_hi = np.log(ilo), np.log(ihi)
        norm = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        ob = (norm * 255.0).astype(np.uint8)
        if invalid_fill == "black":
            ob[~finite] = 0
        out[:, b] = ob

    return out[:, 0] if squeeze else out
