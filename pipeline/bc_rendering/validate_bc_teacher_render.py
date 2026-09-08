#!/usr/bin/env python3
"""Validate complete offline-BC teacher renders against sealed Isaac chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data/BC/isaac"
RENDERED_ROOT = PROJECT_ROOT / "data/BC/rendered"
POOL_MANIFEST = PROJECT_ROOT / "data/BC/manifest.json"
POOL_SEAL = PROJECT_ROOT / "data/BC/SEALED.json"
VIEWS = ("head_color", "hand_left", "hand_right")
RGB_KEYS = ("rgb_gen", *(f"rgb_gen__{view}" for view in VIEWS))
EXACT_PAIRED_KEYS = {
    "gt_actions",
    "dones",
    "prompt",
    "env_id",
    "seq",
    "obs_actor_obs",
    "obs_teacher_obs",
    "obs_gt_obj_pos",
    "obs_critic_obs",
    "obs_homie_obs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_expected() -> list[dict]:
    manifest = json.loads(POOL_MANIFEST.read_text(encoding="utf-8"))
    seal = json.loads(POOL_SEAL.read_text(encoding="utf-8"))
    items = manifest.get("chunks", [])
    if (
        manifest.get("status") != "sealed"
        or int(manifest.get("chunk_count", -1)) != len(items)
        or seal.get("verdict") != "PASS"
        or int(seal.get("isaac_chunks", -1)) != len(items)
        or int(seal.get("rendered_chunks", -1)) != len(items)
    ):
        raise ValueError("unified BC source manifest/seal contract failed")
    return items


def validate_chunk(source_npz: Path, output_npz: Path) -> list[str]:
    errors: list[str] = []
    if not source_npz.is_file() or not output_npz.is_file():
        errors.append("source or rendered NPZ is missing")
        return errors

    try:
        # NPZ is a ZIP of one NPY member per key. Matching CRC and uncompressed
        # size proves that the large Isaac depth/control members survived the
        # render archive rewrite byte-for-byte, without materializing two
        # 65-MB tensors plus a comparison array in the validator cgroup.
        with zipfile.ZipFile(source_npz) as source_zip, zipfile.ZipFile(
            output_npz
        ) as rendered_zip:
            source_members = {item.filename: item for item in source_zip.infolist()}
            rendered_members = {
                item.filename: item for item in rendered_zip.infolist()
            }
            if len(source_members) != len(source_zip.infolist()):
                errors.append("source NPZ contains duplicate ZIP members")
            if len(rendered_members) != len(rendered_zip.infolist()):
                errors.append("rendered NPZ contains duplicate ZIP members")

        with np.load(source_npz, allow_pickle=False) as source, np.load(
            output_npz, allow_pickle=False
        ) as rendered:
            source_keys = set(source.files)
            rendered_keys = set(rendered.files)
            if rendered_keys != source_keys | set(RGB_KEYS):
                errors.append("rendered NPZ key set is not source keys plus four RGB keys")
            paired_keys = sorted(
                key
                for key in source_keys
                if key.startswith("obs_")
                or key in {"gt_actions", "dones", "prompt", "env_id", "seq"}
            )
            for key in paired_keys:
                source_member = source_members.get(f"{key}.npy")
                rendered_member = rendered_members.get(f"{key}.npy")
                if (
                    source_member is None
                    or rendered_member is None
                    or source_member.CRC != rendered_member.CRC
                    or source_member.file_size != rendered_member.file_size
                ):
                    errors.append(f"paired teacher/source value changed: {key}")
                elif key in EXACT_PAIRED_KEYS and not np.array_equal(
                    source[key], rendered[key]
                ):
                    errors.append(f"exact paired teacher/source value changed: {key}")
            if source["gt_actions"].shape != (81, 26):
                errors.append(f"gt_actions shape is {source['gt_actions'].shape}")
            if source["obs_actor_obs"].shape != (81, 113):
                errors.append(f"obs_actor_obs shape is {source['obs_actor_obs'].shape}")
            if not np.isfinite(source["gt_actions"]).all():
                errors.append("gt_actions contains non-finite values")
            for key in paired_keys:
                if key not in EXACT_PAIRED_KEYS:
                    continue
                value = source[key]
                if value.ndim > 0 and value.shape[0] not in {1, 81}:
                    errors.append(f"unsynchronized paired row count: {key}={value.shape}")
            expected_rgb_shape = (81, 108, 192, 3)
            for key in RGB_KEYS:
                if key not in rendered:
                    continue
                value = rendered[key]
                if value.shape != expected_rgb_shape or value.dtype != np.uint8:
                    errors.append(f"{key} is {value.shape}/{value.dtype}")
            if "rgb_gen__head_color" in rendered and "rgb_gen" in rendered:
                if not np.array_equal(rendered["rgb_gen"], rendered["rgb_gen__head_color"]):
                    errors.append("rgb_gen is not the canonical head_color view")
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        errors.append(f"NPZ validation failed: {type(error).__name__}: {error}")

    return errors


def validate_pool() -> dict:
    items = load_expected()
    expected_names = {Path(item["isaac_npz"]).name for item in items}
    actual_names = {path.name for path in RENDERED_ROOT.glob("*.npz")}
    failures: list[dict] = []
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    for item in items:
        name = Path(item["isaac_npz"]).name
        if name in missing:
            continue
        errors = validate_chunk(SOURCE_ROOT / name, RENDERED_ROOT / name)
        if errors:
            failures.append({"chunk": name, "errors": errors})
    return {
        "pool": "BC",
        "expected_chunks": len(expected_names),
        "rendered_chunks": len(actual_names),
        "missing_chunks": missing,
        "unexpected_chunks": unexpected,
        "failed_chunks": failures,
        "verdict": "PASS" if not missing and not unexpected and not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate_pool()
    report = {
        "schema": "roborender.bc_teacher_render_validation.v1",
        "validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pool": result,
        "verdict": result["verdict"],
    }
    write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
