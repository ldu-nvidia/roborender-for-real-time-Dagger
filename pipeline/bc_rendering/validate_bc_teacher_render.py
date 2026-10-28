#!/usr/bin/env python3
"""Validate complete offline-BC teacher renders against sealed Isaac chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .render_recipes import CURRENT_RECIPE, RECIPE_NAMES, context_metadata, recipe_config, collection_provenance
except ImportError:
    from render_recipes import CURRENT_RECIPE, RECIPE_NAMES, context_metadata, recipe_config, collection_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data/BC/isaac"
RENDERED_ROOT = PROJECT_ROOT / "data/BC/rendered_independent_v2"
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


def load_expected(
    pool_manifest: Path = POOL_MANIFEST,
    pool_seal: Path = POOL_SEAL,
) -> list[dict]:
    manifest = json.loads(pool_manifest.read_text(encoding="utf-8"))
    if manifest.get('schema')=='roborender.bc-pool.continuous.v1':
        try:
            from .continuous_source import load_expected as read_continuous
        except ImportError:
            from continuous_source import load_expected as read_continuous
        return read_continuous(pool_manifest,pool_seal)
    seal = json.loads(pool_seal.read_text(encoding="utf-8"))
    items = manifest.get("chunks", [])
    if (
        manifest.get("status") != "sealed"
        or int(manifest.get("chunk_count", -1)) != len(items)
        or seal.get("verdict") != "PASS"
        or int(seal.get("isaac_chunks", -1)) != len(items)
    ):
        raise ValueError("unified BC source manifest/seal contract failed")
    if seal.get("manifest_sha256") and sha256_file(pool_manifest) != seal["manifest_sha256"]:
        raise ValueError("source pool manifest differs from its sealed hash")
    return items


def npy_header(archive: zipfile.ZipFile, key: str) -> tuple[tuple, bool, np.dtype]:
    """Inspect a bounded NPY header without materializing its RGB payload."""
    info = archive.getinfo(key + ".npy")
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version: {version}")
        if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid/object NPY header")
        if info.file_size - stream.tell() != math.prod(shape) * dtype.itemsize:
            raise ValueError("NPY header shape/dtype differs from its declared member byte length")
        return shape, fortran, dtype


def select_deep_sample(items: list[dict], count: int, *, seed: int = 42) -> list[str]:
    """Deterministic coverage-first sample: roles, then motion modes, then fill."""
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("deep RGB sample requires a positive count and integer seed")
    names = [Path(row["isaac_npz"]).name for row in items]
    if len(set(names)) != len(names):
        raise ValueError("render validation source names must be unique")
    ordered = sorted(items, key=lambda row: hashlib.sha256(json.dumps(
        [seed, row.get("chunk_id"), row["isaac_npz"], row["source_sha256"]],
        sort_keys=True, separators=(",", ":")).encode()).digest())
    roles, modes, selected = set(), set(), []
    while ordered and len(selected) < count:
        # A missing provenance field does not invent an "unknown" coverage mode.
        best = max(range(len(ordered)), key=lambda i: (
            int(bool(ordered[i].get("split")) and ordered[i]["split"] not in roles),
            int(bool(ordered[i].get("motion_id")) and ordered[i]["motion_id"] not in modes)))
        row = ordered.pop(best)
        if row.get("split"):
            roles.add(row["split"])
        if row.get("motion_id"):
            modes.add(row["motion_id"])
        selected.append(Path(row["isaac_npz"]).name)
    return selected


def validate_chunk(
    source_npz: Path,
    output_npz: Path,
    *,
    chain_context: dict | None = None,
    deep_rgb: bool = True,
) -> list[str]:
    errors: list[str] = []
    if not source_npz.is_file() or not output_npz.is_file():
        errors.append("source or rendered NPZ is missing")
        return errors

    try:
        # NPZ is a ZIP of one NPY member per key. Matching CRC and uncompressed
        # size is a useful accidental-corruption check, NOT cryptographic
        # identity. validate_pool hashes every full output archive separately.
        # Avoid materializing large Isaac depth/control arrays here.
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
            for key in RGB_KEYS:
                if key + ".npy" not in rendered_members:
                    continue
                shape, fortran, dtype = npy_header(rendered_zip, key)
                if shape != (81, 108, 192, 3) or dtype != np.uint8 or fortran:
                    errors.append(f"{key} NPY header is {shape}/{dtype}/fortran={fortran}")
            head, alias = (rendered_members.get(k + ".npy") for k in ("rgb_gen__head_color", "rgb_gen"))
            if head and alias and (head.CRC != alias.CRC or head.file_size != alias.file_size):
                errors.append("rgb_gen/head_color alias ZIP checks differ")

        with np.load(source_npz, allow_pickle=False) as source, np.load(
            output_npz, allow_pickle=False
        ) as rendered:
            source_keys = set(source.files)
            rendered_keys = set(rendered.files)
            expected_keys = source_keys | set(RGB_KEYS)
            if chain_context is not None:
                expected_metadata = context_metadata(chain_context)
                expected_keys |= set(expected_metadata)
            if rendered_keys != expected_keys:
                errors.append(
                    "rendered NPZ key set differs from the expected source/RGB/chain keys"
                )
            paired_keys = sorted(source_keys)
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
                if key not in rendered.files or not deep_rgb:
                    continue
                value = rendered[key]
                if value.shape != expected_rgb_shape or value.dtype != np.uint8:
                    errors.append(f"{key} is {value.shape}/{value.dtype}")
            if deep_rgb and "rgb_gen__head_color" in rendered.files and "rgb_gen" in rendered.files:
                if not np.array_equal(rendered["rgb_gen"], rendered["rgb_gen__head_color"]):
                    errors.append("rgb_gen is not the canonical head_color view")
            if chain_context is not None and set(expected_metadata) <= rendered_keys:
                actual = {
                    key: rendered[key].item() for key in expected_metadata
                }
                expected = expected_metadata
                if actual != expected:
                    errors.append(
                        f"render chain provenance mismatch: {actual} != {expected}"
                    )
    except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        errors.append(f"NPZ validation failed: {type(error).__name__}: {error}")

    return errors


def expected_chain_contexts(
    items: list[dict],
    *,
    source_root: Path = SOURCE_ROOT,
    rendered_root: Path = RENDERED_ROOT,
    recipe: str = CURRENT_RECIPE,
) -> dict[str, dict]:
    try:
        from .render_bc_teacher_8gpu import build_render_chains, raven_stream_tag
    except ImportError:
        from render_bc_teacher_8gpu import build_render_chains, raven_stream_tag

    jobs = []
    for item in items:
        name = Path(item["isaac_npz"]).name
        path = source_root / name
        with np.load(path, allow_pickle=False) as payload:
            env_id = int(payload["env_id"].item())
            has_done = (
                bool(payload["dones"].any()) if "dones" in payload.files else False
            )
        jobs.append(
            {
                "stream_tag": raven_stream_tag(item),
                "source_run_tag": item.get("source_run_tag", raven_stream_tag(item)),
                "source_batch": int(item.get("source_batch", -1)),
                "rank": int(item.get("rank", 0)),
                "env_id": env_id,
                "sequence_id": int(item["sequence_id"]),
                "source_npz": str(path),
                "source_payload_sha256": item["source_sha256"],
                "output_npz": str(rendered_root / name),
                "global_env_id": int(item.get("global_env_id", env_id)),
                "episode_id": item.get("episode_id", item["sequence_id"]),
                "collection_provenance": collection_provenance(item),
                "frame_count": int(item.get("frames", 81)),
                "has_done": has_done,
            }
        )
    return {
        Path(job["source_npz"]).name: job
        for chain in build_render_chains(jobs, recipe=recipe)
        for job in chain["jobs"]
    }


def validate_pool(
    *,
    source_root: Path = SOURCE_ROOT,
    rendered_root: Path = RENDERED_ROOT,
    recipe: str = CURRENT_RECIPE,
    pool_manifest: Path | None = None,
    pool_seal: Path | None = None,
    rgb_validation: str = "full",
    deep_rgb_sample_count: int = 32,
    sample_seed: int = 42,
) -> dict:
    if rgb_validation not in {"full", "stratified_sample"}:
        raise ValueError("unknown RGB validation mode")
    settings = recipe_config(recipe)
    items = load_expected(pool_manifest or source_root.parent / "manifest.json",
                          pool_seal or source_root.parent / "SEALED.json")
    chain_contexts = expected_chain_contexts(
        items, source_root=source_root, rendered_root=rendered_root, recipe=recipe,
    )
    expected_names = {Path(item["isaac_npz"]).name for item in items}
    if len(expected_names) != len(items):
        raise ValueError("render validation source names must be unique")
    sampled_names = sorted(expected_names) if rgb_validation == "full" else select_deep_sample(
        items, deep_rgb_sample_count, seed=sample_seed)
    sampled = set(sampled_names)
    actual_names = {path.name for path in rendered_root.glob("*.npz")}
    failures: list[dict] = []
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    identities = {}
    for item in items:
        name = Path(item["isaac_npz"]).name
        if name in missing:
            continue
        output = rendered_root / name
        before = output.stat()
        archive_sha256 = sha256_file(output)
        errors = validate_chunk(
            source_root / name,
            output,
            chain_context=chain_contexts[name],
            deep_rgb=name in sampled,
        )
        after = output.stat()
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in identity_fields):
            errors.append("rendered archive changed during identity/validation check")
        identities[name] = {"sha256": archive_sha256, "bytes": before.st_size}
        if errors:
            failures.append({"chunk": name, "errors": errors})
    selected_rows = [item for item in items if Path(item["isaac_npz"]).name in sampled]
    coverage = {field: {"available": sorted({r[field] for r in items if r.get(field)}),
                        "deep_sampled": sorted({r[field] for r in selected_rows if r.get(field)})}
                for field in ("split", "motion_id")}
    return {
        "pool": "BC",
        **settings,
        "expected_chunks": len(expected_names),
        "rendered_chunks": len(actual_names),
        "missing_chunks": missing,
        "unexpected_chunks": unexpected,
        "failed_chunks": failures,
        "rendered_identities": identities,
        "rgb_validation": {"mode": rgb_validation, "sample_seed": sample_seed,
            "requested_deep_count": len(items) if rgb_validation == "full" else deep_rgb_sample_count,
            "deep_count": len(sampled), "deep_sampled_files": sampled_names,
            "deep_sampled_chunk_ids": [r.get("chunk_id", Path(r["isaac_npz"]).name) for r in selected_rows],
            "coverage": coverage,
            "all_chunk_guarantees": ["full_archive_sha256", "metadata_and_recipe",
                "RGB_NPY_shape_dtype_and_declared_bytes", "exact_small_teacher_state_pairs",
                "ZIP_CRC_size_checks_are_not_cryptographic_proof"],
            "deep_sample_guarantees": ["full_RGB_payload_decode", "exact_head_view_alias_equality"],
            "unsampled_limitation": "full RGB decompression/alias element equality not asserted for unsampled chunks"},
        "verdict": "PASS" if not missing and not unexpected and not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--rendered-root", type=Path, default=RENDERED_ROOT)
    parser.add_argument("--recipe", choices=RECIPE_NAMES, default=CURRENT_RECIPE)
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--pool-seal", type=Path)
    parser.add_argument("--rgb-validation", choices=("full", "stratified_sample"), default="full")
    parser.add_argument("--deep-rgb-sample-count", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()
    result = validate_pool(
        source_root=args.source_root.resolve(),
        rendered_root=args.rendered_root.resolve(),
        recipe=args.recipe,
        pool_manifest=args.pool_manifest,
        pool_seal=args.pool_seal,
        rgb_validation=args.rgb_validation, deep_rgb_sample_count=args.deep_rgb_sample_count,
        sample_seed=args.sample_seed,
    )
    report = {
        "schema": "roborender.bc_teacher_render_validation.v2",
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
