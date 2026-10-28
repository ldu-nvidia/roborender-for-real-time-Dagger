"""Explicit online Image3F observation ablation; no policy or physics ownership.

The caller admits the immutable renderer/anchor manifest and source lifecycle.
This adapter changes only vision_obs, using current native controls before each
student action. Image3F5 is NOT the offline BC teacher50 rendering distribution.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

VIEWS = ("head_color", "hand_left", "hand_right")
MODE = "image3f_3view"
ADAPTER_SHA256 = "c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588"


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.name,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_observation_transform(trainer, *, arm=None):
    """Student success must render; an explicit teacher diagnostic may use RGB."""
    transform = getattr(trainer, "_success_observation_transform", None)
    arm = os.environ.get("DIAGNOSTIC_ARM") if arm is None else arm
    required = arm in {"bc", "dagger"} or getattr(trainer, "_require_rendered_success_observations", False)
    if transform is None:
        if required:
            raise RuntimeError("student success requires rendered three-view observations; no raytraced fallback")
        return None
    if not callable(transform) or not callable(getattr(transform, "receipt", None)):
        raise RuntimeError("generated observation callback requires its runtime receipt")
    expected = getattr(trainer, "_rendered_success_evaluation_binding", None)
    if required and expected is None and getattr(trainer, "_rendered_paired_baseline", None) is None:
        raise RuntimeError("student success rendering requires an admitted configuration or paired diagnostic")
    if expected is not None:
        read_binding = getattr(transform, 'evaluation_binding_receipt', None)
        binding = read_binding() if callable(read_binding) else transform.receipt().get('evaluation_binding')
        if binding != expected:
            raise RuntimeError("rendered success callback is not bound to this checkpoint/panel/cohort")
    return transform


def apply_observation_transform(trainer, obs, dones, step):
    """Replace only RGB, failing before the student action if rendering is absent."""
    transform = require_observation_transform(trainer)
    if transform is None:
        return obs
    if not isinstance(obs, dict) or "vision_obs" not in obs:
        raise RuntimeError("generated observation callback requires native vision_obs")
    original_values = dict(obs)
    versions = {key: value._version for key, value in obs.items() if isinstance(value, torch.Tensor)}
    original = obs["vision_obs"]
    result = transform(obs, dones, step)
    if (set(obs) != set(original_values)
            or any(obs[key] is not value for key, value in original_values.items())
            or not isinstance(result, dict) or set(result) != set(original_values)
            or any(result[key] is not value for key, value in original_values.items() if key != "vision_obs")
            or any(original_values[key]._version != version for key, version in versions.items())):
        raise RuntimeError("observation ablation mutated native state or input tensors")
    generated = result["vision_obs"]
    if (not isinstance(generated, torch.Tensor) or generated is original
            or generated.shape != original.shape or generated.dtype != original.dtype
            or generated.device != original.device or not torch.isfinite(generated).all()):
        raise RuntimeError("observation ablation returned incompatible/nonfinite policy RGB")
    return result


def observation_receipt(trainer, vector_steps):
    transform = require_observation_transform(trainer)
    if transform is None:
        return {"observation_mode": "raytraced_3view"}
    receipt = transform.receipt()
    if (receipt.get("observation_mode") != MODE
            or receipt.get("vector_steps_rendered") != vector_steps
            or receipt.get("render_requests_completed") != vector_steps * int(trainer.env.num_envs)
            or receipt.get("raytraced_fallbacks") != 0 or receipt.get("failed") is not False):
        raise RuntimeError("generated success result lacks complete all-lane rendering receipts")
    return {"observation_mode": MODE, "rendered_observation_receipt": receipt}


class GeneratedSuccessObservations:
    """One bounded cohort with independent per-lane transport and anchor RNG.

    trainer is the resident VideoGen-derived trainer, not the evaluator facade;
    its exact h5_v4/resize/normalization helpers are reused without reimplementing
    the scientific preprocessing. client_factory(lane) must create one distinct
    hardened _Image3FLoopbackClient per lane through an authenticated tunnel.
    """
    def __init__(self, trainer, *, client_factory, anchor_files, anchor_seed,
                 renderer_identity, preview_directory=None, telemetry_path=None,
                 evaluation_binding=None, endpoint_pool_config=None, endpoint_pool_physical_gpu=None):
        self.trainer = trainer
        self.count = int(trainer.env.num_envs)
        cameras = trainer.env.config.simulator.config.cameras
        if (not 1 <= self.count <= 40 or tuple(cameras.policy_views) != VIEWS
                or list(trainer.vg_control_res) != [240, 416]
                or trainer.vg_depth_mode != "h5_v4"
                or bool(trainer.env.config.domain_rand.image_augmentation.enabled)
                or list(cameras.image_mean) != [.485, .456, .406]
                or list(cameras.image_std) != [.229, .224, .225]
                or tuple(trainer.camera_resolution) != (324, 192, 3)):
            raise ValueError("generated evaluation requires exact three-view control/policy geometry and no augmentation")
        for name in ("_h5v4_control", "_resize_seq", "_normalize_like_env"):
            if not callable(getattr(trainer, name, None)):
                raise ValueError("missing native renderer conditioning helper: " + name)
        identity = deepcopy(renderer_identity)
        if (identity.get("renderer") != "Image3F" or identity.get("adapter_sha256") != ADAPTER_SHA256
                or not _sha(identity.get("engine_sha256")) or identity.get("steps") != 5
                or identity.get("cfg") != 1.0 or identity.get("teacache_threshold") != .3
                or identity.get("seed") != 42 or identity.get("server_rng_isolated") is not True):
            raise ValueError("exact admitted Image3F5 identity and isolated diffusion RNG required")
        _json(identity)
        if type(anchor_seed) is not int or not 0 <= anchor_seed < 2 ** 32:
            raise ValueError("anchor seed must be an independent uint32 seed")
        if not anchor_files or any(set(row) != {"path", "sha256"}
                or not _sha(row["sha256"]) or not Path(row["path"]).is_absolute()
                for row in anchor_files):
            raise ValueError("hash-pinned absolute anchor inventory required")
        self.anchor_files = tuple(deepcopy(anchor_files))
        if len({row["path"] for row in self.anchor_files}) != len(self.anchor_files):
            raise ValueError("duplicate anchor inventory entries")
        self.anchor_seed, self.renderer_identity = anchor_seed, identity
        self.evaluation_binding = deepcopy(evaluation_binding)
        if self.evaluation_binding is not None:
            binding = self.evaluation_binding
            if (not isinstance(binding, dict) or set(binding) != {"schema", "checkpoint_sha256",
                    "panel_sha256", "config_sha256", "cohort_index", "motion_assignment_sha256"}
                    or binding["schema"] != "roborender.rendered-student-success-cohort-binding.v1"
                    or type(binding["cohort_index"]) is not int or binding["cohort_index"] < 0
                    or any(not _sha(binding[key]) for key in ("checkpoint_sha256", "panel_sha256",
                                                             "config_sha256", "motion_assignment_sha256"))):
                raise ValueError("rendered student success requires an exact evaluation binding")
        self.rng = np.random.RandomState(anchor_seed)
        self.clients = tuple(client_factory(lane) for lane in range(self.count))
        if len({id(client) for client in self.clients}) != self.count:
            raise ValueError("one exclusive renderer stream per concurrent lane required")
        # The production framed client exposes its lock-protected _close(),
        # not a public close(). Bind the real lifecycle API before any work;
        # public close() is also accepted for explicit client wrappers.
        self._client_closers = []
        for client in self.clients:
            closer = getattr(client, "_close", None)
            if not callable(closer):
                closer = getattr(client, "close", None)
            if not callable(getattr(client, "render", None)) or not callable(closer):
                raise ValueError("renderer client requires render() and a callable _close()/close()")
            self._client_closers.append(closer)
        self.endpoint_pool = None
        if endpoint_pool_config is not None:
            try:
                if endpoint_pool_config.get('identity') != identity:
                    raise ValueError('endpoint pool renderer identity differs')
                from pipeline.deployment.rtx_actor.rendered_endpoint_pool import RenderedEndpointPool
                self.endpoint_pool = RenderedEndpointPool(self.clients, endpoint_pool_config,
                    cohort_index=evaluation_binding.get('cohort_index') if isinstance(evaluation_binding, dict) else None,
                    physical_gpu=endpoint_pool_physical_gpu)
            except Exception:
                for closer in self._client_closers:
                    closer()
                raise
        self.pool = None if self.endpoint_pool else ThreadPoolExecutor(
            max_workers=self.count, thread_name_prefix="success-image3f")
        self.phase_timings = []
        self.anchors = [None] * self.count
        self.anchor_receipts = []
        self.steps = self.requests = 0
        self.elapsed = 0.0
        self.failed = False
        self.closed = False
        self.first_inputs = []
        self.trace_digest = hashlib.sha256()
        self.preview = Path(preview_directory) if preview_directory else None
        self.telemetry = Path(telemetry_path) if telemetry_path else None
        if self.preview:
            self.preview.mkdir(parents=True, exist_ok=False)
        if self.telemetry:
            if self.telemetry.exists():
                raise ValueError("observation telemetry must be fresh")
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)

    def _anchor(self, lane, step):
        row = self.anchor_files[int(self.rng.randint(len(self.anchor_files)))]
        path = Path(row["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 << 20:
            raise RuntimeError("selected anchor is not a bounded regular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("selected anchor bytes differ from admitted inventory")
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"].tolist() if "frames" in data.files else None
            index = frames.index(0) if frames is not None else 0
            images = {view: data["gen__" + view][index].copy() for view in VIEWS}
            if "prompt" not in data.files:
                raise RuntimeError("selected anchor lacks its matching prompt")
            prompt = str(data["prompt"].item())
        if not prompt or any(value.shape != (240, 416, 3) or value.dtype != np.uint8
                             for value in images.values()):
            raise RuntimeError("selected anchor lacks frame-zero RGB240x416x3 triplet/prompt")
        self.anchors[lane] = (images, prompt)
        self.anchor_receipts.append({"lane": lane, "selected_at_vector_step": step,
            "file_sha256": row["sha256"], "frame_index": 0,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})

    def __call__(self, obs, dones, step):
        if self.failed or self.closed or step != self.steps:
            raise RuntimeError("generated observation callback cannot retry/replay a partial step")
        try:
            return self._transform(obs, dones, step)
        except Exception:
            self.failed = True
            raise

    def _transform(self, obs, dones, step):
        started = time.perf_counter()
        vision = obs["vision_obs"]
        if (not isinstance(vision, torch.Tensor) or vision.shape[0] != self.count
                or vision.numel() != self.count * 324 * 192 * 3 or not vision.is_floating_point()):
            raise RuntimeError("native vision_obs has wrong three-view shape/type")
        if not isinstance(dones, torch.Tensor) or dones.numel() != self.count:
            raise RuntimeError("native done vector differs from admitted lane cardinality")
        done = dones.detach().reshape(-1).cpu().numpy()
        if not np.isfinite(done).all() or not np.isin(done, [0, 1]).all():
            raise RuntimeError("invalid native done flags")
        for lane in range(self.count):
            if self.anchors[lane] is None or bool(done[lane]):
                self._anchor(lane, step)
        views = self.trainer.env.simulator.get_conditioning_views()
        if not isinstance(views, dict) or not set(VIEWS) <= set(views):
            raise RuntimeError("current native conditioning views unavailable; no raytraced fallback")
        cpu_views = {}
        if self.endpoint_pool:
            # One D2H transfer per modality/view, rather than 6*B blocking
            # transfers. Exact scientific helpers remain per-lane and keep
            # their original order/numerics, including h5_v4 percentiles.
            for view in VIEWS:
                entry = views[view]
                raw_depth, raw_robot = entry.get('depth'), entry.get('robot_rgb')
                if (not isinstance(raw_depth, torch.Tensor) or raw_depth.shape[0] != self.count
                        or raw_depth.ndim != 4 or raw_depth.shape[-1] != 1
                        or not isinstance(raw_robot, torch.Tensor)
                        or raw_robot.shape != (*raw_depth.shape[:3], 3) or raw_robot.dtype != torch.uint8):
                    raise RuntimeError('current depth/robot control shape or dtype differs')
                cpu_views[view] = (raw_depth.detach().to('cpu', torch.float32).numpy(),
                                   raw_robot.detach().cpu().numpy())
        requests = []
        for lane in range(self.count):
            depth, mask = {}, {}
            for view in VIEWS:
                entry = views[view]
                raw_depth, raw_robot = entry.get("depth"), entry.get("robot_rgb")
                if (not isinstance(raw_depth, torch.Tensor) or raw_depth.shape[0] != self.count
                        or raw_depth.ndim != 4 or raw_depth.shape[-1] != 1
                        or not isinstance(raw_robot, torch.Tensor)
                        or raw_robot.shape != (*raw_depth.shape[:3], 3) or raw_robot.dtype != torch.uint8):
                    raise RuntimeError("current depth/robot control shape or dtype differs")
                metric = (cpu_views[view][0][lane, ..., 0] if self.endpoint_pool
                          else raw_depth[lane, ..., 0].detach().to("cpu", torch.float32).numpy())
                if not np.isfinite(metric).all():
                    raise RuntimeError("nonfinite native depth")
                d = self.trainer._h5v4_control(metric[None, None], view)
                d = self.trainer._resize_seq(d, [240, 416])[0, 0]
                robot = cpu_views[view][1][lane] if self.endpoint_pool else raw_robot[lane].detach().cpu().numpy()
                m = self.trainer._resize_seq(robot[None, None], [240, 416])[0, 0]
                if d.shape != (240, 416) or m.shape != (240, 416, 3) or d.dtype != np.uint8 or m.dtype != np.uint8:
                    raise RuntimeError("native h5_v4/robot helper returned invalid model controls")
                depth[view], mask[view] = np.ascontiguousarray(d), np.ascontiguousarray(m)
            anchor, prompt = self.anchors[lane]
            requests.append({"depth": depth, "mask": mask, "previous": anchor, "prompt": prompt, "seed": 42})

        def render(lane):
            return self.clients[lane].render(**requests[lane])

        preprocessed = time.perf_counter()
        results = (self.endpoint_pool.render(requests, step) if self.endpoint_pool
                   else list(self.pool.map(render, range(self.count))))
        rendered = time.perf_counter()
        rgb_lanes, records = [], []
        for lane, result in enumerate(results):
            generated = result.views
            if (set(generated) != set(VIEWS) or any(value.shape != (240, 416, 3) or value.dtype != np.uint8
                    for value in generated.values())):
                raise RuntimeError("renderer returned invalid three-view RGB; no fallback")
            # Same native online convention: tile at full control resolution,
            # then bilinear resize to the policy's full324x192 stack.
            rgb = np.concatenate([generated[view] for view in VIEWS], axis=0)
            rgb01 = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1)[None] / 255.
            rgb01 = F.interpolate(rgb01, size=(324, 192), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            rgb_lanes.append(rgb01)
            record = {"lane": lane, "inputs": {kind: {v: _digest(requests[lane][kind][v]) for v in VIEWS}
                       for kind in ("depth", "mask", "previous")},
                      "generated": {v: _digest(generated[v]) for v in VIEWS}}
            records.append(record)
            if self.preview and step < 2:
                from PIL import Image
                Image.fromarray(np.concatenate([generated[v] for v in VIEWS], axis=0)).resize((192, 324)).save(
                    self.preview / f"step-{step:03d}-lane-{lane:02d}.png")
        normalized = self.trainer._normalize_like_env(torch.stack(rgb_lanes))
        if normalized.shape != (self.count, 324, 192, 3) or not torch.isfinite(normalized).all():
            raise RuntimeError("native normalization returned incompatible policy RGB")
        replacement = normalized.to(device=vision.device, dtype=vision.dtype).reshape(vision.shape)
        self.trace_digest.update(_json({"step": step, "lanes": records}).encode())
        elapsed = time.perf_counter() - started
        self.steps += 1
        self.requests += self.count
        self.elapsed += elapsed
        if self.endpoint_pool:
            self.phase_timings.append(dict(step=step, preprocessing_seconds=preprocessed-started,
                render_wait_seconds=rendered-preprocessed, postprocessing_seconds=time.perf_counter()-rendered))
        if step < 2:
            self.first_inputs.append({"step": step, "lanes": records})
        if self.telemetry:
            with self.telemetry.open("a") as stream:
                row = {"step": step, "lanes_rendered": self.count,
                    "seconds": elapsed, "trace_sha256": self.trace_digest.hexdigest()}
                if self.endpoint_pool:
                    row.update(phase_timing=self.phase_timings[-1],
                        endpoints=self.endpoint_pool.steps[-1]['endpoints'])
                stream.write(_json(row) + "\n")
        return {**obs, "vision_obs": replacement}

    def evaluation_binding_receipt(self):
        # The per-action admission guard must not copy O(B*steps) timing
        # evidence at every action. Full evidence is still returned at result
        # publication; this small immutable copy is sufficient for the guard.
        return deepcopy(self.evaluation_binding)

    def receipt(self):
        value = {"schema": "roborender.rendered-success-observation-receipt.v1", "observation_mode": MODE,
            "renderer_identity": deepcopy(self.renderer_identity), "view_order": list(VIEWS),
            "policy_view_shape": [108, 192, 3], "control_view_shape": [240, 416],
            "vector_steps_rendered": self.steps, "render_requests_completed": self.requests,
            "raytraced_fallbacks": 0, "failed": self.failed,
            "anchor_seed": self.anchor_seed, "anchor_policy": "fixed_frame0_per_episode",
            "anchor_selections": deepcopy(self.anchor_receipts), "normalizations_per_vector_step": 1,
            "controls_from_current_simulator_state": True, "simulator_steps_added": 0,
            "optimizer_steps": 0, "offline_BC_teacher50_equivalent": False,
            "elapsed_seconds": self.elapsed, "trace_sha256": self.trace_digest.hexdigest(),
            "bounded_first_inputs": deepcopy(self.first_inputs)}
        if self.evaluation_binding is not None:
            value["evaluation_binding"] = deepcopy(self.evaluation_binding)
        transports=[client.transport_receipt() for client in self.clients
                    if callable(getattr(client,'transport_receipt',None))]
        if any(row.get('schema') is not None for row in transports):
            value['renderer_transport_receipts']=deepcopy(transports)
        if self.endpoint_pool:
            value['renderer_endpoint_pool'] = self.endpoint_pool.receipt()
            value['phase_timing'] = dict(schema='roborender.rendered-success-timing.v1',
                vector_steps_completed=self.steps, steps=deepcopy(self.phase_timings),
                **{key: sum(row[key] for row in self.phase_timings) for key in (
                    'preprocessing_seconds', 'render_wait_seconds', 'postprocessing_seconds')})
        return value

    def close(self):
        if not self.closed:
            if self.endpoint_pool:
                self.endpoint_pool.close()
            else:
                self.pool.shutdown(wait=True)
            for closer in self._client_closers:
                closer()
            self.closed = True


class GeneratedPooledSuccessObservations(GeneratedSuccessObservations):
    """New benchmark episode namespaces; old fixed-panel adapter is unchanged.

Each request is stateless scientifically: its previous image is the fixed
frame-zero episode anchor, not the previous output. Transport request counters
stay monotonic for exactly-once cached replies; they are not episode history.
"""
    def __init__(self, *args, benchmark_sha256, **kwargs):
        if not _sha(benchmark_sha256):
            raise ValueError('pooled render sessions require their benchmark identity')
        self.benchmark_sha256 = benchmark_sha256
        super().__init__(*args, **kwargs)
        self.episode_indices = [-1] * self.count

    def _anchor(self, lane, step):
        super()._anchor(lane, step)
        self.episode_indices[lane] += 1
        self.anchor_receipts[-1].update(episode_index=self.episode_indices[lane],
            episode_session_id=f'{self.benchmark_sha256}:lane{lane}:episode{self.episode_indices[lane]}')

    def receipt(self):
        value = super().receipt()
        value['pooled_episode_sessions'] = dict(benchmark_sha256=self.benchmark_sha256,
            current_episode_indices=list(self.episode_indices),
            natural_done_refreshes_anchor=True, generated_frame_history_carried=False,
            transport_request_sequence='monotonic_not_scientific_episode_state')
        return value
