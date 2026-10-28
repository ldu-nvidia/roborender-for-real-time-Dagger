"""One streaming registry identity pass; no CUDA, model load or dataset reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def metadata_bytes(path):
    path = Path(path)
    if path.stat().st_size > 32 << 20:
        raise ValueError("metadata exceeds bounded 32 MiB limit")
    with path.open("rb") as stream:
        raw = stream.read((32 << 20) + 1)
    if len(raw) > 32 << 20:
        raise ValueError("metadata exceeded bounded limit while reading")
    return raw


def hash_regular(path, *, expected_sha256=None, expected_bytes=None, block_bytes=8 << 20):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("asset must resolve to a regular file")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError("registry size differs: " + str(path))
        digest = hashlib.sha256()
        while block := handle.read(block_bytes):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if file_identity(before) != file_identity(after) or file_identity(after) != file_identity(path.stat()):
        raise ValueError("asset changed during identity verification: " + str(path))
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError("registry SHA256 differs: " + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=after.st_size,
                file_identity=file_identity(after))


def produce(*, registry_path, renderer_root, source_files, prompt_path, output_dir,
            include_online_assets=False, reuse_receipt=None, reuse_receipt_sha256=None, report=lambda event: None):
    """source_files maps stable logical names to actual source file paths."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("fresh identity receipt directory required")
    registry_raw = metadata_bytes(registry_path)
    registry = json.loads(registry_raw)
    if registry.get("schema") != "roborender.renderer_registry.v1":
        raise ValueError("unsupported renderer registry")
    artifacts = {row["id"]: row for row in registry["artifacts"]}
    foundation = artifacts["wan21_fun_control_1p3b_foundation"]
    adapter = artifacts["roborender_offline_bc_video_teacher_50step_lora_step05294"]
    root = Path(renderer_root).resolve(strict=True)
    prior = {}
    if reuse_receipt is not None:
        raw = metadata_bytes(reuse_receipt)
        previous = json.loads(raw)
        if (sha(raw) != reuse_receipt_sha256 or previous.get("schema") != "roborender.bc-render-assets.v1"
                or previous.get("hashes_verified") is not True
                or previous.get("registry_raw_sha256") != sha(registry_raw)):
            raise ValueError("prior asset receipt is not independently hash-bound")
        prior = {r["role"]: r for r in previous["files"] + previous.get("additional_online_assets", [])}
    specs = [("foundation/" + row["path"], root / foundation["relative_root"] / row["path"], row)
             for row in foundation["files"]]
    specs.append(("adapter", root / adapter["relative_path"], adapter))
    additional = []
    if include_online_assets:
        for name in ("roborender_online_dagger_image3f_5step_lora_step05000",
                     "roborender_online_dagger_image3f_sm90_bf16_3view"):
            row = artifacts[name]
            additional.append((name, root / row.get("physical_relative_path", row["relative_path"]), row))
    records, extra = [], []
    for role, path, expected in specs + additional:
        started = time.monotonic()
        report(dict(phase="asset_hash_start", role=role, expected_bytes=expected["bytes"]))
        old = prior.get(role)
        resolved = path.resolve(strict=True)
        if (old and old["path"] == str(resolved) and old["sha256"] == expected["sha256"]
                and old["bytes"] == expected["bytes"] and old["file_identity"] == file_identity(resolved.stat())):
            record = old
            reused = True
        else:
            record = dict(role=role, **hash_regular(path, expected_sha256=expected["sha256"], expected_bytes=expected["bytes"]))
            reused = False
        (records if role in {item[0] for item in specs} else extra).append(record)
        report(dict(phase="asset_hash_complete", role=role, bytes=record["bytes"], reused_unchanged_identity=reused,
                    seconds=time.monotonic()-started))
    source_records = []
    if not source_files:
        raise ValueError("explicit renderer source files required")
    for relative, path in sorted(source_files.items()):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source identity names must be relative")
        if Path(path).stat().st_size > (64 << 20 if relative.startswith("tokenizer/") else 8 << 20):
            raise ValueError("source identity file exceeds bounded metadata/code size")
        record = hash_regular(path)
        source_records.append(dict(relative_path=relative, sha256=record["sha256"], bytes=record["bytes"]))
        records.append(dict(role="source/" + relative, **record))
    source_manifest = dict(schema="roborender.renderer-source-identity.v1", files=source_records)
    source_raw = canonical(source_manifest)
    prompt = hash_regular(prompt_path)
    records.append(dict(role="prompt", **prompt))
    models = dict(adapter=adapter["sha256"], foundation_manifest=sha(canonical(foundation)),
                  renderer_source=sha(source_raw), prompt=prompt["sha256"])
    receipt = dict(schema="roborender.bc-render-assets.v1", hashes_verified=True,
        registry_raw_sha256=sha(registry_raw), source_manifest_raw_sha256=sha(source_raw),
        model_sha256=models, prompt_bytes=prompt["bytes"], files=records,
        additional_online_assets=extra, no_cuda_or_model_load=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "renderer-source.json").write_bytes(source_raw)
    receipt_raw = canonical(receipt)
    (output_dir / "render-assets.json").write_bytes(receipt_raw)
    bindings = dict(renderer_registry={"path": str(Path(registry_path).resolve()), "raw_sha256": sha(registry_raw)},
        renderer_source_manifest={"path": str((output_dir / "renderer-source.json").resolve()), "raw_sha256": sha(source_raw)},
        render_asset_identity={"path": str((output_dir / "render-assets.json").resolve()), "raw_sha256": sha(receipt_raw)},
        render_model_sha256=models)
    (output_dir / "protocol-bindings.json").write_bytes(canonical(bindings))
    report(dict(phase="asset_identity_complete", bindings=bindings,
                total_large_asset_bytes=sum(r["bytes"] for r in records if not r["role"].startswith("source/")) + sum(r["bytes"] for r in extra)))
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-online-assets", action="store_true")
    parser.add_argument("--reuse-asset-receipt", type=Path)
    parser.add_argument("--reuse-asset-receipt-sha256")
    args = parser.parse_args()
    files = {}
    for base, label in ((args.renderer_root / "offline_teacher/source/diffsynth", "diffsynth"),
                        (args.renderer_root / "offline_teacher/source/sim2realgen", "sim2realgen"),
                        (args.code_root / "pipeline/bc_rendering", "pipeline/bc_rendering")):
        base = base.resolve(strict=True)
        for path in sorted(base.rglob("*.py")):
            files[label + "/" + str(path.relative_to(base))] = path
    tokenizer = (args.renderer_root / "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl").resolve(strict=True)
    for path in sorted(tokenizer.rglob("*")):
        if path.is_file():
            files["tokenizer/" + str(path.relative_to(tokenizer))] = path
    produce(registry_path=args.registry, renderer_root=args.renderer_root, source_files=files,
        prompt_path=args.code_root / "pipeline/vendor/raven-fast-roborender-reference/render_prompt.txt",
        output_dir=args.output_dir, include_online_assets=args.include_online_assets,
        reuse_receipt=args.reuse_asset_receipt, reuse_receipt_sha256=args.reuse_asset_receipt_sha256,
        report=lambda event: print(json.dumps(event, sort_keys=True), flush=True))


if __name__ == "__main__":
    main()
