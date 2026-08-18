#!/usr/bin/env python3
"""Idempotent source transform for Raven's private Image3F DAgger trainer.

The private trainer is not vendored here, so the transform is deliberately
strict: every known source anchor must match exactly once or installation
stops without writing a partially patched file.
"""

from __future__ import annotations

import re


MARKER = "FAST_ROBORENDER_SCALING_V1"


class PatchError(RuntimeError):
    pass


def _sub_once(pattern: str, replacement: str, source: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise PatchError(f"could not locate exactly one {label} block (matched {count})")
    return updated


def patch_source(source: str) -> str:
    """Return trainer source with explicit distributed stream metadata."""
    if MARKER in source:
        return source
    for required in ("_image_dagger_relabel", "_id_env_clients", "ThreadPoolExecutor"):
        if required not in source:
            raise PatchError(f"trainer lacks required Raven marker {required!r}")

    source = _sub_once(
        r"^(?P<i>\s*)self\._id_gen_ms\s*=\s*\[\]\s*$",
        (
            r"\g<i>self._id_gen_ms = []\n"
            r"\g<i># FAST_ROBORENDER_SCALING_V1: explicit ordered stream metadata\n"
            r'\g<i>self.vg_reference_lag = int(vg.get("reference_lag", 0))\n'
            r'\g<i>self._id_rank = int(getattr(self.accelerator, "process_index", 0))\n'
            r"\g<i>self._id_episode_ids = None\n"
            r"\g<i>self._id_episode_steps = None"
        ),
        source,
        "Image3F initialization",
    )

    source = _sub_once(
        (
            r"^(?P<i>\s*)self\._id_prompts\s*=\s*\[None\]\s*\*\s*B\s*\n"
            r"(?P=i)(?=for b in range\(B\):)"
        ),
        (
            r"\g<i>self._id_prompts = [None] * B\n"
            r"\g<i>self._id_episode_ids = np.zeros(B, dtype=np.int64)\n"
            r"\g<i>self._id_episode_steps = np.zeros(B, dtype=np.int64)\n"
            r"\g<i>"
        ),
        source,
        "per-environment episode initialization",
    )

    source = _sub_once(
        (
            r"^(?P<i>\s*)if cur_dones is not None:\s*\n"
            r"(?P<body>(?:^(?P<j>\s+).+\n){1,8}?)"
            r"^(?P<k>\s*)self\._id_sample_anchor\(b\)\s*$"
        ),
        (
            r"\g<i>if cur_dones is not None:\n"
            r"\g<body>"
            r"\g<k>self._id_sample_anchor(b)\n"
            r"\g<k>self._id_episode_ids[b] += 1\n"
            r"\g<k>self._id_episode_steps[b] = 0"
        ),
        source,
        "done/reset handling",
    )

    source = _sub_once(
        (
            r"return self\._id_env_clients\[b\]\.render_frame\(\s*\n"
            r"(?P<i>\s*)depth_d, mask_d, prev_d, "
            r"prompt=self\._id_prompts\[b\], seed=self\.vg_seed\)"
        ),
        (
            r"return self._id_env_clients[b].render_frame(\n"
            r"\g<i>depth_d, mask_d, prev_d, prompt=self._id_prompts[b],\n"
            r"\g<i>seed=self.vg_seed,\n"
            r'''\g<i>stream_id=(f"{os.environ.get('IMAGE3F_RUN_ID', 'run')}:"\n'''
            r'''\g<i>           f"rank{self._id_rank}:env{self._id_rank * B + b}"),\n'''
            r"\g<i>episode_id=int(self._id_episode_ids[b]),\n"
            r"\g<i>step=int(self._id_episode_steps[b]),\n"
            r"\g<i>reference_lag=self.vg_reference_lag)"
        ),
        source,
        "parallel render call",
    )

    source = _sub_once(
        r"^(?P<i>\s*)gens\s*=\s*list\(pool\.map\(_render, range\(B\)\)\)\s*$",
        (
            r"\g<i>gens = list(pool.map(_render, range(B)))\n"
            r"\g<i># Advance only after every environment response succeeded.\n"
            r"\g<i>self._id_episode_steps += 1"
        ),
        source,
        "parallel render completion",
    )
    return source
