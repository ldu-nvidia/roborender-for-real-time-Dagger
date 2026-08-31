#!/usr/bin/env python3
"""Rank-zero media/scalar producer for online Image3F DAgger rollouts.

The trainer integration installs this module beside ``image3f_frame_client``
in Raven's external GR00T tree.  It deliberately has no W&B dependency:
artifacts are written atomically beneath ``TRAIN_RUN_DIR/telemetry`` and the
rank-zero W&B sidecar uploads only complete files.

Each record contains the first environment's synchronized camera triplet:
depth, robot mask, the client-supplied RGB anchor, and generated RGB. The
server may replace that anchor with generated RGB at the configured lag; a
numeric source signal makes that distinction observable without returning a
second image payload. Short videos contain the generated triplet over time.
FFmpeg is probed explicitly; when it is
missing or encoding fails, the exact clip frames are retained as a compressed
NumPy bundle with a JSON reason instead of silently dropping telemetry.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LABEL_HEIGHT = 20
GRID_GAP = 2


@dataclass(frozen=True)
class VideoCapability:
    available: bool
    backend: str
    executable: str | None
    reason: str | None


def probe_video_capability(ffmpeg_path: str | None = None) -> VideoCapability:
    """Return the explicit MP4 capability used by this producer.

    ``ffmpeg_path=""`` intentionally disables encoding, which is useful for
    dependency-restricted deployments and deterministic fallback tests.
    """

    if ffmpeg_path == "":
        return VideoCapability(False, "none", None, "FFmpeg explicitly disabled")
    candidate = ffmpeg_path or os.environ.get("IMAGE_DAGGER_FFMPEG") or shutil.which("ffmpeg")
    if not candidate:
        return VideoCapability(False, "none", None, "FFmpeg executable not found")
    resolved = str(Path(candidate).expanduser().resolve())
    if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        return VideoCapability(False, "none", resolved, "FFmpeg is not executable")
    return VideoCapability(True, "ffmpeg", resolved, None)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_write(path, payload)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + kind
        + payload
        + struct.pack("!I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"PNG image must be HxWx3, got {rgb.shape}")
    height, width = rgb.shape[:2]
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"unsupported telemetry image shape {image.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0.0)
    if image.size and float(image.min()) >= 0.0 and float(image.max()) <= 1.0:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8))


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    source_h, source_w = image.shape[:2]
    if source_h < 1 or source_w < 1:
        raise ValueError("telemetry image cannot have an empty dimension")
    rows = np.minimum((np.arange(height) * source_h // height), source_h - 1)
    columns = np.minimum((np.arange(width) * source_w // width), source_w - 1)
    return np.ascontiguousarray(image[rows[:, None], columns[None, :]])


def _draw_label(image: np.ndarray, text: str) -> None:
    """Draw text when Pillow exists; the labeled header remains without it."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    canvas = Image.fromarray(image)
    ImageDraw.Draw(canvas).text((4, 3), text, fill=(255, 255, 255))
    image[:] = np.asarray(canvas)


def build_contact_sheet(
    depth: Mapping[str, np.ndarray],
    mask: Mapping[str, np.ndarray],
    client_anchor: Mapping[str, np.ndarray],
    generated: Mapping[str, np.ndarray],
    *,
    view_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Build a labeled 4-by-3 RGB sheet without rescaling source resolution."""

    views = tuple(view_order or generated.keys())
    if len(views) != 3 or len(set(views)) != 3:
        raise ValueError(f"DAgger telemetry requires exactly three distinct views, got {views}")
    modalities = (
        ("depth", depth),
        ("mask", mask),
        ("client_anchor", client_anchor),
        ("generated", generated),
    )
    for modality, images in modalities:
        missing = [view for view in views if view not in images]
        if missing:
            raise ValueError(f"{modality} is missing telemetry views: {missing}")

    base = _to_uint8_rgb(generated[views[0]])
    height, width = base.shape[:2]
    cell_height = LABEL_HEIGHT + height
    sheet_height = len(modalities) * cell_height + (len(modalities) - 1) * GRID_GAP
    sheet_width = len(views) * width + (len(views) - 1) * GRID_GAP
    sheet = np.full((sheet_height, sheet_width, 3), 20, dtype=np.uint8)
    for row, (modality, images) in enumerate(modalities):
        y = row * (cell_height + GRID_GAP)
        for column, view in enumerate(views):
            x = column * (width + GRID_GAP)
            tile = _resize_nearest(_to_uint8_rgb(images[view]), height, width)
            sheet[y + LABEL_HEIGHT : y + cell_height, x : x + width] = tile
            header = sheet[y : y + LABEL_HEIGHT, x : x + width]
            header[:] = (11, 35, 50)
            _draw_label(header, f"{modality} / {view}")
    return sheet


def build_generated_triplet(
    generated: Mapping[str, np.ndarray], *, view_order: Sequence[str]
) -> np.ndarray:
    """Build one labeled 3-view RGB frame for the temporal MP4."""

    views = tuple(view_order)
    base = _to_uint8_rgb(generated[views[0]])
    height, width = base.shape[:2]
    frame = np.full((height + LABEL_HEIGHT, 3 * width + 2 * GRID_GAP, 3), 20, np.uint8)
    for column, view in enumerate(views):
        x = column * (width + GRID_GAP)
        frame[LABEL_HEIGHT:, x : x + width] = _resize_nearest(
            _to_uint8_rgb(generated[view]), height, width
        )
        header = frame[:LABEL_HEIGHT, x : x + width]
        header[:] = (11, 35, 50)
        _draw_label(header, f"generated / {view}")
    return frame


def _flatten_finite_metrics(
    values: Mapping[str, object], *, prefix: str = "renderer"
) -> dict[str, int | float]:
    clean: dict[str, int | float] = {}

    def visit(path: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(f"{path}/{key}" if path else str(key), nested)
        elif isinstance(value, (bool, np.bool_)):
            clean[path] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if math.isfinite(numeric):
                clean[path] = int(value) if isinstance(value, (int, np.integer)) else numeric

    visit(prefix, values)
    return clean


def _reference_source_metrics(metadata: Mapping[str, object]) -> dict[str, int]:
    """Encode the server's string source as stable numeric W&B signals."""

    source = str(metadata.get("reference_source", "unknown"))
    categories = {
        # Keep the historical aggregate for existing dashboards while exposing
        # the two semantically distinct pre-lag sources.
        "anchor": source in {"anchor", "bootstrap_anchor"},
        "bootstrap_anchor": source == "bootstrap_anchor",
        "generated_frame0": source == "generated_episode_frame0",
        "generated_lag": source.startswith("generated_t-"),
        "idempotent_replay": source == "idempotent_replay",
        "client": source == "client",
    }
    output = {
        f"renderer/reference_source_{name}": int(enabled)
        for name, enabled in categories.items()
    }
    output["renderer/reference_source_unknown"] = int(not any(categories.values()))
    if categories["generated_lag"]:
        try:
            output["renderer/reference_lag_frames"] = int(source.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            output["renderer/reference_source_unknown"] = 1
    return output


class DaggerTelemetry:
    """Atomic first-environment DAgger observability producer."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        scalar_every: int = 10,
        image_every: int = 300,
        clip_every: int = 300,
        clip_frames: int = 32,
        clip_fps: int = 10,
        rollout_steps: int = 300,
        ffmpeg_path: str | None = None,
        require_video: bool = False,
    ):
        if (
            scalar_every < 1
            or image_every < 1
            or clip_every < 0
            or clip_frames < 2
            or clip_fps < 1
            or rollout_steps < 1
        ):
            raise ValueError("invalid DAgger telemetry interval or clip configuration")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.root = self.run_dir / "telemetry"
        self.image_dir = self.root / "images"
        self.video_dir = self.root / "videos"
        self.scalar_dir = self.root / "scalars"
        self.fallback_dir = self.root / "video_fallback"
        for directory in (self.image_dir, self.video_dir, self.scalar_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.scalar_every = int(scalar_every)
        self.image_every = int(image_every)
        self.clip_every = int(clip_every)
        self.clip_frames = int(clip_frames)
        self.clip_fps = int(clip_fps)
        self.rollout_steps = int(rollout_steps)
        self.video_capability = probe_video_capability(ffmpeg_path)
        self.require_video = bool(require_video)
        if self.require_video and not self.video_capability.available:
            raise RuntimeError(
                "required DAgger MP4 telemetry is unavailable: "
                f"{self.video_capability.reason or 'unknown video backend error'}"
            )
        self._frames: deque[np.ndarray] = deque(maxlen=self.clip_frames)
        _atomic_json(self.root / "video_capability.json", asdict(self.video_capability))

    @classmethod
    def from_environment(cls, *, rank: int) -> "DaggerTelemetry | None":
        """Construct only on rank zero when the production run dir is set.

        ``TRAIN_RUN_DIR`` is the production contract. ``RUN_DIR`` remains a
        compatibility fallback for direct/manual trainer launches.  When the
        launcher also exports ``ROBORENDER_TELEMETRY_DIR``, its value must
        agree exactly with ``<run-dir>/telemetry`` so a typo cannot silently
        split local artifacts from the W&B watcher.
        """

        if int(rank) != 0:
            return None
        run_dir = os.environ.get("TRAIN_RUN_DIR") or os.environ.get("RUN_DIR")
        if not run_dir:
            return None
        expected_telemetry = Path(run_dir).expanduser().resolve() / "telemetry"
        configured_telemetry = os.environ.get("ROBORENDER_TELEMETRY_DIR")
        if configured_telemetry and Path(configured_telemetry).expanduser().resolve() != expected_telemetry:
            raise ValueError(
                "ROBORENDER_TELEMETRY_DIR must equal TRAIN_RUN_DIR/telemetry"
            )
        return cls(
            run_dir,
            scalar_every=int(os.environ.get("IMAGE_DAGGER_TELEMETRY_SCALAR_EVERY", "10")),
            image_every=int(os.environ.get("IMAGE_DAGGER_TELEMETRY_IMAGE_EVERY", "300")),
            clip_every=int(os.environ.get("IMAGE_DAGGER_TELEMETRY_CLIP_EVERY", "300")),
            clip_frames=int(os.environ.get("IMAGE_DAGGER_TELEMETRY_CLIP_FRAMES", "32")),
            clip_fps=int(os.environ.get("IMAGE_DAGGER_TELEMETRY_CLIP_FPS", "10")),
            rollout_steps=int(os.environ.get("ROLLOUT_STEPS", "300")),
            ffmpeg_path=os.environ.get("IMAGE_DAGGER_FFMPEG"),
            require_video=os.environ.get(
                "IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO", "0"
            ).strip().lower()
            in {"1", "true", "yes"},
        )

    def _write_mp4(self, path: Path, frames: Sequence[np.ndarray]) -> None:
        capability = self.video_capability
        if not capability.available or not capability.executable:
            raise RuntimeError(capability.reason or "MP4 encoder unavailable")
        height = max(frame.shape[0] for frame in frames)
        width = max(frame.shape[1] for frame in frames)
        # yuv420p requires even dimensions. Padding preserves every source pixel.
        height += height % 2
        width += width % 2
        normalized = []
        for frame in frames:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[: frame.shape[0], : frame.shape[1]] = frame
            normalized.append(canvas)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent
        )
        os.close(descriptor)
        try:
            command = [
                capability.executable,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(self.clip_fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                temporary_name,
            ]
            subprocess.run(
                command,
                input=b"".join(frame.tobytes() for frame in normalized),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _write_video_or_fallback(self, trainer_step: int) -> Path:
        frames = tuple(self._frames)
        destination = self.video_dir / f"step_{trainer_step:09d}_generated_triplet.mp4"
        try:
            self._write_mp4(destination, frames)
            return destination
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if self.require_video:
                raise RuntimeError(
                    "required DAgger MP4 telemetry encoding failed"
                ) from exc
            self.fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback = self.fallback_dir / f"step_{trainer_step:09d}_generated_triplet.npz"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{fallback.name}.", dir=fallback.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez_compressed(handle, frames=np.stack(frames), fps=self.clip_fps)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, fallback)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            _atomic_json(
                fallback.with_suffix(".json"),
                {
                    "step": trainer_step,
                    "fallback": fallback.name,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "capability": asdict(self.video_capability),
                },
            )
            return fallback

    def record(
        self,
        *,
        trainer_step: int,
        depth: Mapping[str, np.ndarray],
        mask: Mapping[str, np.ndarray],
        client_anchor: Mapping[str, np.ndarray],
        generated: Mapping[str, np.ndarray],
        client_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, Path]:
        """Atomically record one first-environment render and its timing data."""

        if isinstance(trainer_step, bool) or int(trainer_step) < 0:
            raise ValueError("trainer_step must be a non-negative integer")
        trainer_step = int(trainer_step)
        views = tuple(generated.keys())
        # Build every frame so the periodic clip remains temporally contiguous.
        triplet = build_generated_triplet(generated, view_order=views)
        self._frames.append(triplet)
        artifacts: dict[str, Path] = {}

        if trainer_step % self.image_every == 0:
            sheet = build_contact_sheet(
                depth, mask, client_anchor, generated, view_order=views
            )
            image_path = self.image_dir / f"step_{trainer_step:09d}_conditions_generated.png"
            _atomic_write(image_path, _png_bytes(sheet))
            artifacts["image"] = image_path

        if trainer_step % self.scalar_every == 0:
            metadata = client_metadata or {}
            metrics = _flatten_finite_metrics(metadata)
            metrics.update(_reference_source_metrics(metadata))
            metrics.update(
                {
                    "telemetry/view_count": len(views),
                    "telemetry/generated_height": int(triplet.shape[0] - LABEL_HEIGHT),
                    "telemetry/generated_triplet_width": int(triplet.shape[1]),
                    "telemetry/video_encoder_available": int(self.video_capability.available),
                    "telemetry/recorded_unix_s": float(time.time()),
                    "rendered_frame_count": trainer_step,
                    "rollout_frame_step": trainer_step % self.rollout_steps,
                    "policy_update_index": trainer_step // self.rollout_steps,
                }
            )
            scalar_path = self.scalar_dir / f"step_{trainer_step:09d}.json"
            _atomic_json(scalar_path, {"step": trainer_step, "metrics": metrics})
            artifacts["scalar"] = scalar_path

        if (
            self.clip_every > 0
            and (trainer_step + 1) % self.clip_every == 0
            and len(self._frames) >= 2
        ):
            artifacts["video"] = self._write_video_or_fallback(trainer_step)
        return artifacts
