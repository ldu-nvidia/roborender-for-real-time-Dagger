"""Hash-bound machine bindings; no policy, renderer, or rollout settings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket

LEGACY = {
    "schema": "roborender.rtx.host-deployment.v1",
    "hostname": "8f68286-lcedt",
    "physical_gpu_index": 1,
    "physical_gpu_uuid": "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9",
    "image": "roborender-round-actor:isaaclab2.3.0-checkpoint-compat-20260901",
    "image_id": "sha256:fcaa595df8f175763ac326187a97dc8bbddca55d6bcc3531cdfecd6be1705a43",
    "raven_root": "/home/ldu/BC_render/rtx_BC_transfer",
    "anchor_root": "/mnt/data1/ldu/research/fast_roborender/runtime/hsg_stage_transfer/anchor_pool_2048",
    "cache_root": "/home/ldu/BC_render/BC_collection_data/persistent-cache/roborender-round-actor-isaaclab2.3.0-checkpoint-compat-20260901",
}


def load_host_deployment(*, path=None, digest=None, hostname=None):
    path = path or os.environ.get("DAGGER_HOST_DEPLOYMENT_FILE")
    digest = digest or os.environ.get("DAGGER_HOST_DEPLOYMENT_SHA256")
    if not path:
        if digest:
            raise ValueError("host deployment hash has no file")
        return dict(LEGACY)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("explicit host deployment requires its SHA256")
    source = Path(path)
    if not source.is_absolute() or source.is_symlink():
        raise ValueError("host deployment must be an absolute regular non-symlink file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("host deployment SHA256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(LEGACY):
        raise ValueError("host deployment fields differ from schema")
    if value["schema"] != LEGACY["schema"]:
        raise ValueError("unknown host deployment schema")
    if type(value["physical_gpu_index"]) is not int or value["physical_gpu_index"] != 1:
        raise ValueError("GPU0 must never be probed or used; only physical GPU1 is allowed")
    if value["hostname"] != (hostname or socket.gethostname()):
        raise ValueError("host deployment belongs to another machine")
    if not isinstance(value["physical_gpu_uuid"], str) or not re.fullmatch(
        r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["physical_gpu_uuid"]
    ):
        raise ValueError("invalid GPU UUID")
    if not isinstance(value["image_id"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_id"]):
        raise ValueError("image must be pinned by immutable ID")
    if not isinstance(value["image"], str) or not re.fullmatch(r"[a-zA-Z0-9._:/@-]+", value["image"]):
        raise ValueError("invalid image reference")
    for field in ("raven_root", "anchor_root", "cache_root"):
        item = value[field]
        if not isinstance(item, str) or not item.startswith("/") or any(c in item for c in "\n\r\x00,") or ".." in Path(item).parts:
            raise ValueError(f"invalid absolute {field}")
    # Scientific source identities in assets.manifest.json retain this root.
    if value["raven_root"] != LEGACY["raven_root"]:
        raise ValueError("Raven root relocation requires a separate source-manifest migration")
    return value


def shell_bindings():
    value = load_host_deployment()
    names = {
        "physical_gpu_uuid": "EXPECTED_PHYSICAL_GPU_UUID",
        "image": "IMAGE", "image_id": "EXPECTED_IMAGE_ID",
        "raven_root": "RAVEN", "anchor_root": "ANCHOR_ROOT", "cache_root": "CACHE_ROOT",
    }
    return "\n".join(f"{name}={shlex.quote(value[key])}" for key, name in names.items())


if __name__ == "__main__":
    print(shell_bindings())
