"""Rendered population publication must not invent training/validation roles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pipeline.bc_rendering.activate_corrected_render_pool import activate, main
from pipeline.bc_rendering.render_recipes import CURRENT_RECIPE, recipe_config


def fixture(tmp_path: Path, split: str | None):
    source = tmp_path / "isaac"
    rendered = tmp_path / "rendered"
    source.mkdir()
    rendered.mkdir()
    rows = []
    identities = {}
    for i in range(2):
        name = f"chunk-{i}.npz"
        row = {"chunk_id": f"chunk-{i}", "isaac_npz": f"isaac/{name}",
               "source_run_tag": "historical-unknown" if split is None else "fresh-validation",
               "rank": 0, "local_env_id": i, "native_episode_id": None}
        if split is not None:
            row["split"] = split
        rows.append(row)
        path = rendered / name
        np.savez_compressed(
            path, render_contract=np.asarray(recipe_config(CURRENT_RECIPE)["render_contract"]),
            render_recipe=np.asarray(CURRENT_RECIPE), render_chain_id=np.asarray(f"chain-{i}"),
            render_chain_index=np.asarray(0), render_chain_length=np.asarray(1),
            render_inherits_previous_reference=np.asarray(False),
            render_source_stream_id=np.asarray(f"stream-{i}"),
        )
        identities[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size}
    document = {"schema": "roborender.bc_pool.native-cohorts.v1", "chunk_count": 2,
                "frames_per_chunk": 81, "chunks": rows}
    if split is not None:
        document.update(partition_counts={split: 2}, collection_purpose="fresh_validation_only",
                        sampling_plan_schema="roborender.bc-fresh-validation-plan.v1")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(json.dumps(document))
    return dict(dataset_root=tmp_path, source_root=source, source_manifest_path=source_manifest,
                rendered_root=rendered, manifest_path=tmp_path / "activated.json"), document, identities


@pytest.mark.parametrize("split", [None, "validation"])
def test_manifest_only_preserves_population_and_never_partitions(tmp_path, split):
    kwargs, source, identities = fixture(tmp_path, split)
    # An unrelated historical split directory is neither reused nor overwritten.
    existing = tmp_path / "splits_independent_v2"
    existing.mkdir()
    (existing / "marker").write_text("old roles are not this population")
    with (
        patch("pipeline.bc_rendering.validate_bc_teacher_render.validate_pool",
              return_value={"verdict": "PASS", "rendered_identities": identities}) as validator,
        patch("pipeline.bc_training.create_bc_splits.create_splits",
              side_effect=AssertionError("population activation must not partition")),
    ):
        result = activate(**kwargs, manifest_only=True)
    assert validator.call_count == 1
    assert result["schema"] == "roborender.bc_render_activation.v2"
    assert result["manifest_only"] is True
    assert result["training_admitted"] is False
    assert result["splits"] is None and result["splits_sha256"] is None
    activated = json.loads(kwargs["manifest_path"].read_text())
    assert activated["parent_manifest_sha256"] == hashlib.sha256(kwargs["source_manifest_path"].read_bytes()).hexdigest()
    assert activated.get("partition_counts") == source.get("partition_counts")
    assert [r.get("split") for r in activated["chunks"]] == [split, split]
    assert all(r["native_episode_id"] is None for r in activated["chunks"])
    assert sorted(p.name for p in existing.iterdir()) == ["marker"]
    assert (existing / "marker").read_text() == "old roles are not this population"
    with pytest.raises(FileExistsError):
        activate(**kwargs, manifest_only=True)


def test_manifest_only_still_requires_render_validation(tmp_path):
    kwargs, _, _ = fixture(tmp_path, "validation")
    with patch("pipeline.bc_rendering.validate_bc_teacher_render.validate_pool",
               return_value={"verdict": "FAIL"}):
        with pytest.raises(ValueError, match="failed validation"):
            activate(**kwargs, manifest_only=True)
    assert not kwargs["manifest_path"].exists()


@pytest.mark.parametrize("value", [1, "true", None])
def test_manifest_only_rejects_nonboolean(tmp_path, value):
    kwargs, _, _ = fixture(tmp_path, None)
    with pytest.raises(ValueError, match="boolean"):
        activate(**kwargs, manifest_only=value)
    assert not kwargs["manifest_path"].exists()


def test_manifest_only_rejects_conflicting_split_output(tmp_path):
    kwargs, _, _ = fixture(tmp_path, "validation")
    with pytest.raises(ValueError, match="must not specify"):
        activate(**kwargs, manifest_only=True, split_dir=tmp_path / "misleading-splits")
    assert not kwargs["manifest_path"].exists()


def test_cli_manifest_only_is_explicit(tmp_path):
    with (
        patch("sys.argv", ["activate", "--dataset-root", str(tmp_path), "--manifest-only",
                           "--report", str(tmp_path / "report.json")]),
        patch("pipeline.bc_rendering.activate_corrected_render_pool.activate", return_value={}) as called,
        patch("pipeline.bc_rendering.activate_corrected_render_pool.atomic_new_json"),
        patch("builtins.print"),
    ):
        main()
    assert called.call_args.kwargs["manifest_only"] is True
    assert called.call_args.kwargs["split_dir"] is None


def test_cli_rejects_manifest_only_and_split_dir(tmp_path):
    with patch("sys.argv", ["activate", "--manifest-only", "--split-dir", str(tmp_path / "splits"),
                            "--report", str(tmp_path / "report.json")]):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()
