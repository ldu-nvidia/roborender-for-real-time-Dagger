#!/usr/bin/env python3
"""Issue one non-evidence Image3F request before a runtime-adapter A/B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from round_dagger.artifacts import publish_immutable_json, sha256_file


SCHEMA = "fast_roborender.round_dagger.image3f_warmup.v1"
VIEWS = ("head_color", "hand_left", "hand_right")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    sha256_file(path)
    return path


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(",".join(str(int(item)) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def warm_renderer(
    *,
    endpoint: Path,
    anchor: Path,
    output: Path,
    renderer_sha256: str,
    stream_id: str,
) -> dict[str, object]:
    endpoint = _regular_file(endpoint, "endpoint descriptor")
    anchor = _regular_file(anchor, "anchor")
    if not _SHA256_RE.fullmatch(renderer_sha256):
        raise ValueError("renderer identity must be a lowercase SHA-256 digest")
    if not stream_id or any(character.isspace() for character in stream_id):
        raise ValueError("warm-up stream ID must be a non-empty token")
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("warm-up output must be an absolute non-symlink path")

    from gr00t.rl.utils.image3f_frame_client import Image3FFrameClient

    with np.load(anchor, allow_pickle=False) as archive:
        expected = {f"gen__{view}" for view in VIEWS}
        if not expected.issubset(archive.files) or "prompt" not in archive.files:
            raise ValueError("anchor does not contain the required three RGB views and prompt")
        previous = {
            view: np.ascontiguousarray(archive[f"gen__{view}"][0]) for view in VIEWS
        }
        prompt = str(archive["prompt"].item())
    for view, image in previous.items():
        if image.dtype != np.uint8 or image.shape != (240, 416, 3):
            raise ValueError(f"anchor view {view} has unexpected shape or dtype")

    depth = {view: np.zeros((240, 416), dtype=np.uint8) for view in VIEWS}
    mask = {view: np.zeros((240, 416, 3), dtype=np.uint8) for view in VIEWS}
    fields = endpoint.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[0] != "127.0.0.1":
        raise ValueError("endpoint descriptor must contain a local host and port")
    try:
        port = int(fields[1])
    except ValueError as exc:
        raise ValueError("endpoint descriptor port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("endpoint descriptor port must be within 1..65535")

    previous_expected = os.environ.get("ROUND_DAGGER_RENDERER_SHA256")
    os.environ["ROUND_DAGGER_RENDERER_SHA256"] = renderer_sha256
    client = None
    try:
        client = Image3FFrameClient(fields[0], port, stream_slot=0)
        generated = client.render_frame(
            depth,
            mask,
            previous,
            prompt=prompt,
            seed=42,
            stream_id=stream_id,
            episode_id="warmup",
            step=0,
            reference_lag=256,
            trainer_step=0,
        )
        metadata = dict(client.last_metadata)
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            if previous_expected is None:
                os.environ.pop("ROUND_DAGGER_RENDERER_SHA256", None)
            else:
                os.environ["ROUND_DAGGER_RENDERER_SHA256"] = previous_expected

    if metadata.get("replayed", False):
        raise RuntimeError("warm-up request was unexpectedly served from replay")
    try:
        inference_ms = float(metadata.get("inference_ms", 0.0))
        client_wall_ms = float(metadata["client_wall_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("warm-up renderer timing metadata is malformed") from exc
    if not np.isfinite(inference_ms) or inference_ms <= 0.0:
        raise RuntimeError("warm-up did not execute real renderer inference")
    if not np.isfinite(client_wall_ms) or client_wall_ms <= 0.0:
        raise RuntimeError("warm-up renderer client timing is invalid")
    if tuple(generated) != VIEWS:
        raise RuntimeError("warm-up output view order differs from the three-view contract")
    for view, image in generated.items():
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (
            240,
            416,
            3,
        ):
            raise RuntimeError(
                f"warm-up output view {view} has unexpected shape or dtype"
            )
    document: dict[str, object] = {
        "schema": SCHEMA,
        "endpoint": str(endpoint),
        "endpoint_sha256": sha256_file(endpoint),
        "anchor": str(anchor),
        "anchor_sha256": sha256_file(anchor),
        "renderer_sha256": renderer_sha256,
        "stream_id": stream_id,
        "seed": 42,
        "reference_lag": 256,
        "replayed": False,
        "inference_ms": inference_ms,
        "client_wall_ms": client_wall_ms,
        "reference_source": metadata.get("reference_source"),
        "generated_sha256": {
            view: _array_sha256(generated[view]) for view in VIEWS
        },
    }
    publish_immutable_json(output, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--renderer-sha256", required=True)
    parser.add_argument("--stream-id", required=True)
    arguments = parser.parse_args()
    result = warm_renderer(
        endpoint=arguments.endpoint,
        anchor=arguments.anchor,
        output=arguments.output,
        renderer_sha256=arguments.renderer_sha256,
        stream_id=arguments.stream_id,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
