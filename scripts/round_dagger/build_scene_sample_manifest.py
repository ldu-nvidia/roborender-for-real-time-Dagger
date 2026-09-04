#!/usr/bin/env python3
"""Package a bounded, provenance-pinned sample of actor render telemetry.

The source is one explicitly selected actor-rank telemetry directory containing
an ``images/`` child.  The manifest is published last and is the only
completion marker.  No source image is modified or followed through a symlink.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import struct
import sys
import tempfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)


SCHEMA = "fast_roborender.round_dagger.scene_samples.v1"
MANIFEST_NAME = "manifest.json"
EXPECTED_WIDTH = 1252
EXPECTED_HEIGHT = 1046
MAX_SAMPLES = 4
MAX_PNG_BYTES = 25 * 1024 * 1024
VIEW_LABELS = ("head_color", "hand_left", "hand_right")
ROW_LABELS = ("depth", "mask", "client_anchor", "generated")
SAMPLING_POLICY = "uniform_index_endpoints_v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PNG_RE = re.compile(r"step_([0-9]{9})_conditions_generated\.png\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SceneSampleError(RuntimeError):
    """Scene telemetry is ambiguous, malformed, or unsafe to package."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_chain(path: Path) -> None:
    path = _absolute(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise SceneSampleError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise SceneSampleError(f"path ancestor is not a directory: {current}")


def _directory(path: str | os.PathLike[str], label: str) -> Path:
    resolved = _absolute(path)
    _check_chain(resolved)
    try:
        info = os.lstat(resolved)
    except OSError as exc:
        raise SceneSampleError(f"{label} is missing") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise SceneSampleError(f"{label} must be a real directory")
    return resolved


def _png_identity(path: Path) -> dict[str, int | str]:
    """Hash and inspect a regular RGB PNG through one no-follow descriptor."""

    _check_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SceneSampleError(f"cannot open source PNG {path.name}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_PNG_BYTES:
            raise SceneSampleError(f"source PNG {path.name} has an unsafe size or type")
        digest = hashlib.sha256()
        header = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            if len(header) < 33:
                header = (header + chunk)[:33]
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if len(header) < 33 or header[:8] != _PNG_SIGNATURE:
        raise SceneSampleError(f"source image {path.name} is not a complete PNG")
    length = struct.unpack("!I", header[8:12])[0]
    if length != 13 or header[12:16] != b"IHDR":
        raise SceneSampleError(f"source image {path.name} lacks canonical IHDR")
    ihdr = header[16:29]
    if zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF != struct.unpack("!I", header[29:33])[0]:
        raise SceneSampleError(f"source image {path.name} has an invalid IHDR CRC")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        "!IIBBBBB", ihdr
    )
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise SceneSampleError(
            f"source image {path.name} dimensions are {width}x{height}, expected "
            f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise SceneSampleError(f"source image {path.name} is not canonical RGB8 PNG")
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": info.st_size,
        "width": width,
        "height": height,
    }


def _candidates(images: Path, *, rollout_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps: set[int] = set()
    for path in sorted(images.iterdir(), key=lambda item: item.name):
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise SceneSampleError("cannot inspect actor telemetry image") from exc
        if not stat.S_ISREG(info.st_mode):
            raise SceneSampleError("actor image directory contains a non-regular entry")
        match = _PNG_RE.fullmatch(path.name)
        if match is None:
            raise SceneSampleError(f"ambiguous PNG inventory entry: {path.name}")
        step = int(match.group(1))
        if step in steps or step >= rollout_steps:
            raise SceneSampleError("actor image rollout steps are duplicate or out of range")
        steps.add(step)
        rows.append({"source": path, "rollout_step": step, **_png_identity(path)})
    rows.sort(key=lambda row: row["rollout_step"])
    if not rows or rows[0]["rollout_step"] != 0:
        raise SceneSampleError("actor telemetry must contain the step-zero contact sheet")
    return rows


def _sample_indices(count: int) -> tuple[int, ...]:
    selected = min(count, MAX_SAMPLES)
    if selected == 1:
        return (0,)
    # Nearest-integer endpoint spacing, expressed with integer arithmetic.
    denominator = selected - 1
    return tuple(
        (index * (count - 1) + denominator // 2) // denominator
        for index in range(selected)
    )


def _copy_png(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    if os.path.lexists(destination):
        observed = _png_identity(destination)
        if observed != {key: expected[key] for key in observed}:
            raise SceneSampleError("existing destination sample differs from source")
        if stat.S_IMODE(os.lstat(destination).st_mode) != 0o444:
            raise SceneSampleError("existing destination sample is writable")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, source_flags)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fchmod(output.fileno(), 0o444)
                os.fsync(output.fileno())
        finally:
            os.close(source_fd)
        if digest.hexdigest() != expected["sha256"] or _png_identity(temporary) != {
            key: expected[key] for key in ("sha256", "size_bytes", "width", "height")
        }:
            raise SceneSampleError("source PNG changed during immutable copy")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            competing = _png_identity(destination)
            if competing != {
                key: expected[key]
                for key in ("sha256", "size_bytes", "width", "height")
            }:
                raise SceneSampleError("competing destination sample differs")
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    final = _png_identity(destination)
    if final != {
        key: expected[key]
        for key in ("sha256", "size_bytes", "width", "height")
    } or stat.S_IMODE(os.lstat(destination).st_mode) != 0o444:
        raise SceneSampleError("published destination sample identity differs")
    try:
        source_sha256 = sha256_file(source)
    except ArtifactError as exc:
        raise SceneSampleError("source PNG became unsafe after immutable copy") from exc
    if source_sha256 != expected["sha256"]:
        raise SceneSampleError("source PNG changed after immutable copy")


def package_scene_samples(
    *,
    actor_telemetry_dir: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    campaign_id: str,
    actor_rank: int,
    rollout_steps: int,
) -> dict[str, Any]:
    """Publish a bounded scene sample set and return its immutable manifest."""

    if not isinstance(campaign_id, str) or _TOKEN_RE.fullmatch(campaign_id) is None:
        raise SceneSampleError("campaign_id is not a safe token")
    if type(actor_rank) is not int or actor_rank < 0:
        raise SceneSampleError("actor_rank must be a non-negative integer")
    if type(rollout_steps) is not int or rollout_steps < 1:
        raise SceneSampleError("rollout_steps must be a positive integer")
    telemetry = _directory(actor_telemetry_dir, "actor telemetry directory")
    images = _directory(telemetry / "images", "actor telemetry image directory")
    output = _absolute(destination)
    if output.name != "scene_samples":
        raise SceneSampleError("destination directory must be named scene_samples")
    _check_chain(output)
    if telemetry == output or telemetry in output.parents or output in telemetry.parents:
        raise SceneSampleError("source telemetry and destination must be disjoint")
    if output.exists():
        output = _directory(output, "scene sample destination")
    else:
        output.mkdir(parents=True, exist_ok=False)
        output = _directory(output, "scene sample destination")

    candidates = _candidates(images, rollout_steps=rollout_steps)
    source_inventory = [
        {
            "path": row["source"].name,
            "rollout_step": row["rollout_step"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "width": row["width"],
            "height": row["height"],
        }
        for row in candidates
    ]
    selected = [candidates[index] for index in _sample_indices(len(candidates))]
    samples: list[dict[str, Any]] = []
    for row in selected:
        target = output / row["source"].name
        _copy_png(row["source"], target, row)
        samples.append(
            {
                "path": target.name,
                "rollout_step": row["rollout_step"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "width": row["width"],
                "height": row["height"],
            }
        )
    expected_names = {item["path"] for item in samples} | {MANIFEST_NAME}
    unexpected = {item.name for item in output.iterdir()} - expected_names
    if unexpected:
        raise SceneSampleError(f"scene sample destination contains unexpected entries: {sorted(unexpected)}")

    core = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "actor_rank": actor_rank,
        "rollout_steps": rollout_steps,
        "view_labels": list(VIEW_LABELS),
        "row_labels": list(ROW_LABELS),
        "sampling_policy": {
            "name": SAMPLING_POLICY,
            "max_samples": MAX_SAMPLES,
            "source_order": "ascending_rollout_step",
            "requires_step_zero": True,
        },
        "source": {
            "actor_telemetry_dir": str(telemetry),
            "candidate_count": len(candidates),
            "inventory_sha256": canonical_json_sha256(source_inventory),
        },
        "sampled_rollout_steps": [item["rollout_step"] for item in samples],
        "samples": samples,
    }
    manifest = {**core, "sample_set_sha256": canonical_json_sha256(core)}
    try:
        publish_immutable_json(output / MANIFEST_NAME, manifest)
    except ArtifactError as exc:
        raise SceneSampleError(f"cannot publish immutable scene sample manifest: {exc}") from exc
    try:
        observed = load_strict_json(output / MANIFEST_NAME)
    except ArtifactError as exc:
        raise SceneSampleError("published scene sample manifest is unsafe") from exc
    if observed != manifest:
        raise SceneSampleError("published scene sample manifest differs")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-telemetry-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--actor-rank", required=True, type=int)
    parser.add_argument("--rollout-steps", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = package_scene_samples(
        actor_telemetry_dir=args.actor_telemetry_dir,
        destination=args.destination,
        campaign_id=args.campaign_id,
        actor_rank=args.actor_rank,
        rollout_steps=args.rollout_steps,
    )
    print(manifest["sample_set_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SceneSampleError as exc:
        print(f"scene sample packaging failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
