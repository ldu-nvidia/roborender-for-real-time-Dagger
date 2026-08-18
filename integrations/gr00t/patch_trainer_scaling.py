#!/usr/bin/env python3
"""Idempotent source transform for Raven's private Image3F DAgger trainer.

The private trainer is not vendored here, so the transform is deliberately
strict: every known source anchor must match exactly once or installation
stops without writing a partially patched file.
"""

from __future__ import annotations

import re


MARKER = "FAST_ROBORENDER_SCALING_V1"
RUNTIME_GATES_MARKER = "FAST_ROBORENDER_PRODUCTION_GATES_V2"
LEGACY_RUNTIME_GATES_MARKER = "FAST_ROBORENDER_PRODUCTION_GATES_V1"
TELEMETRY_MARKER = "FAST_ROBORENDER_DAGGER_TELEMETRY_V1"
LEGACY_REQUEST_MARKER = "reqs = []"
DICT_REQUEST_MARKER = "reqs = {}"
SCALING_REQUIRED_LITERALS = (
    "FAST_ROBORENDER_SCALING_V1: explicit ordered stream metadata",
    'self.vg_reference_lag = int(vg.get("reference_lag", 0))',
    'self._id_rank = int(getattr(self.accelerator, "process_index", 0))',
    "self._id_episode_ids = np.zeros(B, dtype=np.int64)",
    "self._id_episode_steps = np.zeros(B, dtype=np.int64)",
    "self._id_episode_ids[b] += 1",
    "self._id_episode_steps[b] = 0",
    "stream_id=(f\"{os.environ.get('IMAGE3F_RUN_ID', 'run')}:\"",
    "episode_id=int(self._id_episode_ids[b])",
    "step=int(self._id_episode_steps[b])",
    "reference_lag=self.vg_reference_lag,",
    "self._id_episode_steps += 1",
)


class PatchError(RuntimeError):
    pass


def _sub_once(pattern: str, replacement: str, source: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise PatchError(f"could not locate exactly one {label} block (matched {count})")
    return updated


def _require_literals_once(source: str, literals: tuple[str, ...], label: str) -> None:
    for literal in literals:
        count = source.count(literal)
        if count != 1:
            raise PatchError(
                f"invalid {label}: expected exactly one {literal!r} (found {count})"
            )


def _request_batch_shape(source: str) -> str:
    """Identify one exact supported Raven parallel-render implementation."""
    legacy_literals = (
        LEGACY_REQUEST_MARKER,
        "gens = list(pool.map(_render, range(B)))",
    )
    dictionary_literals = (
        DICT_REQUEST_MARKER,
        "gens = {}",
        "for b, g in zip(gen_envs, pool.map(_render, gen_envs)):",
        "gens[b] = g",
    )
    legacy = all(source.count(literal) == 1 for literal in legacy_literals)
    dictionary = all(source.count(literal) == 1 for literal in dictionary_literals)
    if legacy == dictionary:
        raise PatchError(
            "trainer must contain exactly one supported Raven request/generation "
            "batch shape (legacy list or generated-environment dictionary)"
        )
    literals = legacy_literals if legacy else dictionary_literals
    _require_literals_once(source, literals, f"{'legacy list' if legacy else 'dictionary'} batch")
    return "legacy" if legacy else "dictionary"


def _remove_legacy_runtime_gates(source: str) -> str:
    """Remove the exact V1 block before installing the stricter V2 gates."""
    if LEGACY_RUNTIME_GATES_MARKER not in source:
        return source
    if source.count(LEGACY_RUNTIME_GATES_MARKER) != 1:
        raise PatchError("invalid legacy production runtime gates marker count")
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if LEGACY_RUNTIME_GATES_MARKER in line]
    ends = [
        i
        for i, line in enumerate(lines)
        if i >= starts[0]
        and "production gate failed: entropy coefficient mismatch" in line
    ]
    if len(ends) != 1 or ends[0] < starts[0]:
        raise PatchError("could not locate exact legacy production runtime gates block")
    return "".join(lines[: starts[0]] + lines[ends[0] + 1 :])


def _patch_runtime_gates(source: str) -> str:
    """Add fail-fast checks for the controls pinned by the production launcher."""
    required_literals = (
        RUNTIME_GATES_MARKER,
        "IMAGE_DAGGER_EXPECT_PRIMARY_VIEW",
        "production gate failed: primary view mismatch",
        "IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE",
        "production gate failed: rollout action source mismatch",
        "IMAGE_DAGGER_EXPECT_BETA_SCHEDULE",
        "production gate failed: beta schedule mismatch",
        "IMAGE_DAGGER_EXPECT_BETA_START",
        "production gate failed: beta start mismatch",
        "IMAGE_DAGGER_EXPECT_BETA_END",
        "production gate failed: beta end mismatch",
        "IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS",
        "production gate failed: teacher warmup mismatch",
        "IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER",
        "production gate failed: behavior source mismatch",
        "IMAGE_DAGGER_EXPECT_TEACHER_RATIO",
        "production gate failed: teacher ratio mismatch",
        "IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS",
        "production gate failed: learning epochs mismatch",
        "IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF",
        "production gate failed: value loss coefficient mismatch",
        "IMAGE_DAGGER_EXPECT_ENTROPY_COEF",
        "production gate failed: entropy coefficient mismatch",
    )
    if RUNTIME_GATES_MARKER in source:
        _require_literals_once(source, required_literals, "production runtime gates")
        return source
    source = _remove_legacy_runtime_gates(source)
    source = _sub_once(
        r"^(?P<i>\s*)self\._id_episode_steps\s*=\s*None\s*$",
        (
            r"\g<i>self._id_episode_steps = None\n"
            r"\g<i># FAST_ROBORENDER_PRODUCTION_GATES_V2\n"
            r"\g<i>_gate_primary = os.environ.get('IMAGE_DAGGER_EXPECT_PRIMARY_VIEW')\n"
            r"\g<i>if _gate_primary and str(self.vg_primary_view) != _gate_primary:\n"
            r"\g<i>    raise RuntimeError('production gate failed: primary view mismatch')\n"
            r"\g<i>_gate_rollout_source = os.environ.get('IMAGE_DAGGER_EXPECT_ROLLOUT_ACTION_SOURCE')\n"
            r"\g<i>if _gate_rollout_source is not None and str(getattr(self, 'vg_rollout_action_source', None)) != _gate_rollout_source:\n"
            r"\g<i>    raise RuntimeError('production gate failed: rollout action source mismatch')\n"
            r"\g<i>_gate_beta_schedule = os.environ.get('IMAGE_DAGGER_EXPECT_BETA_SCHEDULE')\n"
            r"\g<i>if _gate_beta_schedule is not None and str(getattr(self, 'vg_beta_schedule', None)) != _gate_beta_schedule:\n"
            r"\g<i>    raise RuntimeError('production gate failed: beta schedule mismatch')\n"
            r"\g<i>_gate_beta_start = os.environ.get('IMAGE_DAGGER_EXPECT_BETA_START')\n"
            r"\g<i>if _gate_beta_start is not None and abs(float(getattr(self, 'vg_beta_start', -1.0)) - float(_gate_beta_start)) > 1e-9:\n"
            r"\g<i>    raise RuntimeError('production gate failed: beta start mismatch')\n"
            r"\g<i>_gate_beta_end = os.environ.get('IMAGE_DAGGER_EXPECT_BETA_END')\n"
            r"\g<i>if _gate_beta_end is not None and abs(float(getattr(self, 'vg_beta_end', -1.0)) - float(_gate_beta_end)) > 1e-9:\n"
            r"\g<i>    raise RuntimeError('production gate failed: beta end mismatch')\n"
            r"\g<i>_gate_warmup = os.environ.get('IMAGE_DAGGER_EXPECT_TEACHER_WARMUP_STEPS')\n"
            r"\g<i>if _gate_warmup is not None and int(self.config.get('rollout_with_teacher_num_steps', -1)) != int(_gate_warmup):\n"
            r"\g<i>    raise RuntimeError('production gate failed: teacher warmup mismatch')\n"
            r"\g<i>_gate_enforce = os.environ.get('IMAGE_DAGGER_EXPECT_ENFORCE_TEACHER')\n"
            r"\g<i>if _gate_enforce is not None:\n"
            r"\g<i>    _gate_enforce_bool = _gate_enforce.lower() == 'true'\n"
            r"\g<i>    if bool(self.config.get('enforce_teacher_rollout', False)) != _gate_enforce_bool:\n"
            r"\g<i>        raise RuntimeError('production gate failed: behavior source mismatch')\n"
            r"\g<i>_gate_ratio = os.environ.get('IMAGE_DAGGER_EXPECT_TEACHER_RATIO')\n"
            r"\g<i>if _gate_ratio is not None and abs(float(self.config.get('ratio_teacher_rollout', -1.0)) - float(_gate_ratio)) > 1e-9:\n"
            r"\g<i>    raise RuntimeError('production gate failed: teacher ratio mismatch')\n"
            r"\g<i>_gate_epochs = os.environ.get('IMAGE_DAGGER_EXPECT_LEARNING_EPOCHS')\n"
            r"\g<i>if _gate_epochs is not None and int(self.config.get('num_learning_epochs', -1)) != int(_gate_epochs):\n"
            r"\g<i>    raise RuntimeError('production gate failed: learning epochs mismatch')\n"
            r"\g<i>_gate_value_loss = os.environ.get('IMAGE_DAGGER_EXPECT_VALUE_LOSS_COEF')\n"
            r"\g<i>if _gate_value_loss is not None and abs(float(self.config.get('value_loss_coef', -1.0)) - float(_gate_value_loss)) > 1e-12:\n"
            r"\g<i>    raise RuntimeError('production gate failed: value loss coefficient mismatch')\n"
            r"\g<i>_gate_entropy = os.environ.get('IMAGE_DAGGER_EXPECT_ENTROPY_COEF')\n"
            r"\g<i>if _gate_entropy is not None and abs(float(self.config.get('entropy_coef', -1.0)) - float(_gate_entropy)) > 1e-12:\n"
            r"\g<i>    raise RuntimeError('production gate failed: entropy coefficient mismatch')"
        ),
        source,
        "production runtime gate insertion point",
    )
    _require_literals_once(source, required_literals, "production runtime gates")
    return source


def _patch_telemetry(source: str) -> str:
    """Add rank-zero first-environment scalar and media production."""
    required_literals = (
        TELEMETRY_MARKER,
        "DaggerTelemetry.from_environment(rank=self._id_rank)",
        "trainer_step=int(self._id_telemetry_step))",
        "self._id_telemetry.record(",
        "depth=reqs[0][0], mask=reqs[0][1]",
        "client_anchor=reqs[0][2], generated=gens[0]",
        "not isinstance(gens, dict) or 0 in gens",
        "_id_metadata['parallel_batch_size'] = len(gens)",
        "IMAGE_DAGGER_TELEMETRY_REQUIRED",
        "self._id_telemetry_step += 1",
    )
    if TELEMETRY_MARKER in source:
        _require_literals_once(source, required_literals, "DAgger telemetry integration")
        return source
    _request_batch_shape(source)
    source = _sub_once(
        r"^(?P<i>\s*)self\._id_episode_steps\s*=\s*None\s*$",
        (
            r"\g<i>self._id_episode_steps = None\n"
            r"\g<i># FAST_ROBORENDER_DAGGER_TELEMETRY_V1\n"
            r"\g<i>from gr00t.rl.utils.dagger_telemetry import DaggerTelemetry\n"
            r"\g<i>self._id_telemetry_step = 0\n"
            r"\g<i>self._id_telemetry = DaggerTelemetry.from_environment(rank=self._id_rank)"
        ),
        source,
        "DAgger telemetry initialization point",
    )
    source = _sub_once(
        r"^(?P<i>\s*)reference_lag=self\.vg_reference_lag\)\s*$",
        (
            r"\g<i>reference_lag=self.vg_reference_lag,\n"
            r"\g<i>trainer_step=int(self._id_telemetry_step))"
        ),
        source,
        "trainer-step protocol metadata",
    )
    source = _sub_once(
        (
            r"^(?P<i>\s*)# Advance only after every environment response succeeded\.\s*\n"
            r"(?P=i)self\._id_episode_steps \+= 1\s*$"
        ),
        (
            r"\g<i>if self._id_telemetry is not None and (\n"
            r"\g<i>        not isinstance(gens, dict) or 0 in gens):\n"
            r"\g<i>    _id_metadata = dict(getattr(self._id_env_clients[0], 'last_metadata', {}))\n"
            r"\g<i>    _id_metadata['parallel_batch_wall_ms'] = (time.perf_counter() - t0) * 1000.0\n"
            r"\g<i>    _id_metadata['parallel_batch_size'] = len(gens)\n"
            r"\g<i>    try:\n"
            r"\g<i>        self._id_telemetry.record(\n"
            r"\g<i>            trainer_step=int(self._id_telemetry_step),\n"
            r"\g<i>            depth=reqs[0][0], mask=reqs[0][1],\n"
            r"\g<i>            client_anchor=reqs[0][2], generated=gens[0],\n"
            r"\g<i>            client_metadata=_id_metadata)\n"
            r"\g<i>    except Exception:\n"
            r"\g<i>        if os.environ.get('IMAGE_DAGGER_TELEMETRY_REQUIRED', '0').lower() in {'1', 'true', 'yes'}:\n"
            r"\g<i>            raise\n"
            r"\g<i>        logger.exception('image_dagger telemetry disabled after write failure')\n"
            r"\g<i>        self._id_telemetry = None\n"
            r"\g<i># Advance only after every environment response succeeded.\n"
            r"\g<i>self._id_episode_steps += 1\n"
            r"\g<i>self._id_telemetry_step += 1"
        ),
        source,
        "parallel render telemetry completion",
    )
    _require_literals_once(source, required_literals, "DAgger telemetry integration")
    return source


def patch_source(source: str) -> str:
    """Return trainer source with explicit distributed stream metadata."""
    if MARKER not in source:
        for required in (
            "_image_dagger_relabel",
            "_id_env_clients",
            "ThreadPoolExecutor",
        ):
            if required not in source:
                raise PatchError(f"trainer lacks required Raven marker {required!r}")
        batch_shape = _request_batch_shape(source)

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

        if batch_shape == "legacy":
            source = _sub_once(
                r"^(?P<i>\s*)gens\s*=\s*list\(pool\.map\(_render, range\(B\)\)\)\s*$",
                (
                    r"\g<i>gens = list(pool.map(_render, range(B)))\n"
                    r"\g<i># Advance only after every environment response succeeded.\n"
                    r"\g<i>self._id_episode_steps += 1"
                ),
                source,
                "legacy parallel render completion",
            )
        else:
            source = _sub_once(
                (
                    r"^(?P<i>\s*)gens\s*=\s*\{\}\s*\n"
                    r"(?P=i)if gen_envs:\s*\n"
                    r"(?P=i)    with ThreadPoolExecutor\(max_workers=len\(gen_envs\)\) as pool:\s*\n"
                    r"(?P=i)        for b, g in zip\(gen_envs, pool\.map\(_render, gen_envs\)\):\s*\n"
                    r"(?P=i)            gens\[b\] = g\s*$"
                ),
                (
                    r"\g<i>gens = {}\n"
                    r"\g<i>if gen_envs:\n"
                    r"\g<i>    with ThreadPoolExecutor(max_workers=len(gen_envs)) as pool:\n"
                    r"\g<i>        for b, g in zip(gen_envs, pool.map(_render, gen_envs)):\n"
                    r"\g<i>            gens[b] = g\n"
                    r"\g<i># Advance only after every environment response succeeded.\n"
                    r"\g<i>self._id_episode_steps += 1"
                ),
                source,
                "dictionary parallel render completion",
            )
    source = _patch_telemetry(source)
    _require_literals_once(source, SCALING_REQUIRED_LITERALS, "scaling integration")
    return _patch_runtime_gates(source)
