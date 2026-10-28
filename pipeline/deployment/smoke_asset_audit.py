"""Bounded CPU asset admission; no full-dataset payload scan or GPU allocation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(root, role):
    rows = []
    def pin(path, size, digest, *, hash_bytes=True):
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError("asset missing/size mismatch: " + str(path))
        if hash_bytes and sha(path) != digest:
            raise RuntimeError("asset digest mismatch: " + str(path))
        rows.append({"path": str(path), "bytes": size, "sha256": digest,
                     "check": "sha256" if hash_bytes else "size_only_runtime_load_pending"})
    if role == "beta":
        manifest = json.loads((root / "pipeline/dagger/rtx/config/assets.manifest.json").read_bytes())
        for row in manifest["artifacts"]:
            pin(Path(row["path"]), row["bytes"], row["sha256"])
        anchors = Path("/home/ldu/roborender_assets/anchors/anchor_pool_2048")
        if not anchors.is_dir() or not any(anchors.iterdir()):
            raise RuntimeError("Beta anchor pool missing")
        return {"role": role, "passed": True, "artifacts": rows,
                "anchor_pool_present": True, "gpu_initializations": 0}
    manifest = json.loads((root / "checkpoints/manifest.json").read_bytes())
    for row in manifest["artifacts"]:
        if row["status"].startswith("active"):
            pin(root / "checkpoints" / row["relative_path"], row["bytes"], row["sha256"])
    registry = json.loads((root / "renderer/manifests/registry.v1.json").read_bytes())
    for row in registry["artifacts"]:
        if "relative_path" in row:
            pin(root / "renderer" / row["relative_path"], row["bytes"], row["sha256"],
                hash_bytes=row["bytes"] < 1 << 30)
        for entry in row.get("files", []):
            pin(root / "renderer" / row["relative_root"] / entry["path"], entry["bytes"], entry["sha256"], hash_bytes=False)
    from pipeline.deployment.h200.qualify_bc_resume import CHECKPOINT, CHECKPOINT_SHA
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("BC parent checkpoint mismatch")
    import torch
    parent = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {"policy_state_dict", "optimizer_state_dict", "global_step", "epoch", "policy_spec"}
    if not required <= set(parent) or (parent["epoch"], parent["global_step"]) != (41, 4100):
        raise RuntimeError("BC handoff metadata missing/different")
    groups, streams, ids = {}, {}, {}
    for split in ("train", "validation", "test"):
        data = json.loads((root / "data/BC/splits_autoregressive_v1" / (split + ".json")).read_bytes())
        groups[split] = len(data["items"])
        streams[split] = {r["stream_id"] for r in data["items"]}
        ids[split] = {r["chunk_id"] for r in data["items"]}
        for row in data["items"]:
            if not (root / "data/BC" / row["rendered_npz"]).is_file():
                raise RuntimeError("indexed rendered chunk missing: " + row["chunk_id"])
    if groups != {"train": 1600, "validation": 200, "test": 200}:
        raise RuntimeError("BC split counts differ")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if streams[left] & streams[right] or ids[left] & ids[right]:
            raise RuntimeError("BC split leakage")
    return {"role": role, "passed": True, "artifacts": rows, "split_counts": groups,
            "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_epoch": 41, "checkpoint_step": 4100,
            "payload_scan": "none; manifest/path checks only", "gpu_initializations": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--role", choices=["cpu", "beta"], required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.root, args.role)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
