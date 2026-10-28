#!/usr/bin/env python3
"""Publish immutable indexes for a validated, explicitly versioned BC render."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .render_recipes import CURRENT_RECIPE, RECIPE_NAMES, INDEPENDENT_RECIPE, recipe_config
except ImportError:
    from render_recipes import CURRENT_RECIPE, RECIPE_NAMES, INDEPENDENT_RECIPE, recipe_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BC_ROOT = PROJECT_ROOT / "data/BC"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable activation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def activate(
    *,
    dataset_root: Path,
    source_root: Path | None = None,
    source_manifest_path: Path | None = None,
    rendered_root: Path,
    manifest_path: Path,
    split_dir: Path | None = None,
    manifest_only: bool = False,
    recipe: str = CURRENT_RECIPE,
    pool_seal: Path | None = None,
    rgb_validation: str = "full",
    deep_rgb_sample_count: int = 32,
    sample_seed: int = 42,
) -> dict:
    from pipeline.bc_rendering.validate_bc_teacher_render import validate_pool
    if not isinstance(manifest_only, bool):
        raise ValueError("manifest_only must be a boolean")
    if manifest_only and split_dir is not None:
        raise ValueError("manifest-only activation must not specify a split directory")
    dataset_root = dataset_root.resolve()
    source_root = (source_root or dataset_root / "isaac").resolve()
    source_manifest_path = (source_manifest_path or dataset_root / "manifest.json").resolve()
    rendered_root = rendered_root.resolve()
    manifest_path = manifest_path.resolve()
    split_dir = None if manifest_only else (split_dir or dataset_root / "splits_independent_v2").resolve()
    settings = recipe_config(recipe)
    if manifest_path.exists() or (split_dir is not None and split_dir.exists()):
        raise FileExistsError(
            "refusing to replace an existing corrected manifest or split directory"
        )
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validation = validate_pool(
        source_root=source_root, rendered_root=rendered_root, recipe=recipe,
        pool_manifest=source_manifest_path, pool_seal=pool_seal,
        rgb_validation=rgb_validation, deep_rgb_sample_count=deep_rgb_sample_count,
        sample_seed=sample_seed,
    )
    if validation["verdict"] != "PASS":
        raise ValueError("selected render pool failed validation")

    rendered_relative_root = Path(os.path.relpath(rendered_root, dataset_root))
    rows = []
    inherited = 0
    chain_ids: set[str] = set()
    for original in source["chunks"]:
        row = copy.deepcopy(original)
        name = Path(row["isaac_npz"]).name
        source_path = source_root / name
        rendered_path = rendered_root / name
        with np.load(rendered_path, allow_pickle=False) as payload:
            contract = str(payload["render_contract"].item())
            chain_id = str(payload["render_chain_id"].item())
            chain_index = int(payload["render_chain_index"].item())
            chain_length = int(payload["render_chain_length"].item())
            inherits = bool(payload["render_inherits_previous_reference"].item())
            if recipe == INDEPENDENT_RECIPE:
                if str(payload["render_recipe"].item()) != recipe:
                    raise ValueError(f"{name}: independent render recipe mismatch")
                row["render_source_stream_id"] = str(payload["render_source_stream_id"].item())
        if contract != settings["render_contract"]:
            raise ValueError(f"{name}: invalid render contract {contract!r}")
        if inherits != (chain_index > 0):
            raise ValueError(f"{name}: inconsistent inherited-reference flag")
        if not 0 <= chain_index < chain_length <= settings["max_chain_chunks"]:
            raise ValueError(f"{name}: invalid chain position")
        inherited += int(inherits)
        chain_ids.add(chain_id)
        row.update(
            {
                "isaac_npz": os.path.relpath(source_path, dataset_root),
                "rendered_npz": str(rendered_relative_root / name),
                "rendered_sha256": validation["rendered_identities"][name]["sha256"],
                "rendered_size_bytes": validation["rendered_identities"][name]["bytes"],
                "render_contract": contract,
                "render_chain_id": chain_id,
                "render_chain_index": chain_index,
                "render_chain_length": chain_length,
                "render_inherits_previous_reference": inherits,
            }
        )
        rows.append(row)

    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    activated = copy.deepcopy(source)
    activated.update(
        {
            "schema": "roborender.bc_pool.rendered.v2",
            **settings,
            "created_at_utc": created,
            "parent_manifest": os.path.relpath(source_manifest_path, manifest_path.parent),
            "parent_manifest_sha256": sha256_file(source_manifest_path),
            "rendered_pool": str(rendered_root),
            "render_chain_count": len(chain_ids),
            "chunks_inheriting_previous_reference": inherited,
            "validation": validation,
            "chunks": rows,
        }
    )
    atomic_new_json(manifest_path, activated)
    if manifest_only:
        # Pure rendered populations (for example all historical training or
        # fresh validation-only data) are partitioned by a separate admission
        # contract. Never invent an 80/10/10 split during render activation.
        return {
            "schema": "roborender.bc_render_activation.v2",
            **settings,
            "created_at_utc": created,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_only": True,
            "splits": None,
            "splits_sha256": None,
            "chunk_count": len(rows),
            "chain_count": len(chain_ids),
            "inherited_reference_chunks": inherited,
            "training_admitted": False,
            "verdict": "PASS",
        }
    from pipeline.bc_training.create_bc_splits import create_splits

    create_splits(
        dataset_root,
        source_manifest_path=manifest_path,
        split_dir=split_dir,
    )
    return {
        "schema": "roborender.bc_render_activation.v1",
        **settings,
        "created_at_utc": created,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": str(split_dir / "manifest.json"),
        "splits_sha256": sha256_file(split_dir / "manifest.json"),
        "chunk_count": len(rows),
        "chain_count": len(chain_ids),
        "inherited_reference_chunks": inherited,
        "verdict": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=BC_ROOT)
    parser.add_argument(
        "--source-root", type=Path, help="defaults to DATASET_ROOT/isaac"
    )
    parser.add_argument(
        "--source-manifest", type=Path, help="defaults to DATASET_ROOT/manifest.json"
    )
    parser.add_argument(
        "--rendered-root", type=Path, help="defaults to DATASET_ROOT/rendered_independent_v2"
    )
    parser.add_argument(
        "--manifest", type=Path, help="defaults to DATASET_ROOT/manifests/independent_v2.json"
    )
    partition = parser.add_mutually_exclusive_group()
    partition.add_argument(
        "--split-dir", type=Path, help="defaults to DATASET_ROOT/splits_independent_v2"
    )
    partition.add_argument(
        "--manifest-only", action="store_true",
        help="publish only the validated rendered population; separate data admission must assign training/validation roles",
    )
    parser.add_argument("--recipe", choices=RECIPE_NAMES, default=CURRENT_RECIPE)
    parser.add_argument("--pool-seal", type=Path)
    parser.add_argument("--rgb-validation", choices=("full", "stratified_sample"), default="full")
    parser.add_argument("--deep-rgb-sample-count", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.recipe != INDEPENDENT_RECIPE and not all(
        (args.rendered_root, args.manifest, args.split_dir or args.manifest_only)
    ):
        parser.error("historical activation requires explicit --rendered-root, --manifest and either --split-dir or --manifest-only")
    result = activate(
        dataset_root=args.dataset_root,
        source_root=args.source_root,
        source_manifest_path=args.source_manifest,
        rendered_root=args.rendered_root or args.dataset_root / "rendered_independent_v2",
        manifest_path=args.manifest or args.dataset_root / "manifests/independent_v2.json",
        split_dir=None if args.manifest_only else args.split_dir or args.dataset_root / "splits_independent_v2",
        manifest_only=args.manifest_only,
        recipe=args.recipe, pool_seal=args.pool_seal,
        rgb_validation=args.rgb_validation, deep_rgb_sample_count=args.deep_rgb_sample_count,
        sample_seed=args.sample_seed,
    )
    atomic_new_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
