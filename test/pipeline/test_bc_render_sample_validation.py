import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from pipeline.bc_rendering.render_bc_teacher_worker import write_raven_rendered_chunk
from pipeline.bc_rendering.validate_bc_teacher_render import (
    expected_chain_contexts, select_deep_sample, validate_chunk, validate_pool,
)


def pool(tmp_path):
    source, rendered = tmp_path / "isaac", tmp_path / "rendered"
    source.mkdir()
    rendered.mkdir()
    items = []
    for index, (role, motion) in enumerate((r, m) for r in ("train", "validation", "test") for m in ("m0", "m1")):
        path = source / f"chunk_native_{index:07d}.npz"
        np.savez_compressed(path, gt_actions=np.full((81, 26), index, np.float32),
            obs_actor_obs=np.zeros((81, 113), np.float32), dones=np.zeros(81, np.bool_),
            prompt=np.asarray("task"), env_id=np.asarray(0), seq=np.asarray(0))
        items.append({"chunk_id": f"bc-{index:04d}", "isaac_npz": "isaac/" + path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_size_bytes": path.stat().st_size,
            "source_run_tag": f"native-{role}-{index}", "rank": 0, "local_env_id": 0,
            "global_env_id": index, "sequence_id": 0, "motion_id": motion, "motion_sha256": "a" * 64,
            "simulator_seed": index + 1000, "episode_id": f"episode-{index}", "split": role,
            "request_sha256": "b" * 64, "sampling_plan_sha256": "c" * 64,
            "teacher_only": True, "selected_for_split": True, "overcollection": False})
    manifest, seal = tmp_path / "manifest.json", tmp_path / "SEALED.json"
    manifest.write_text(json.dumps({"status": "sealed", "chunk_count": len(items), "chunks": items}))
    seal.write_text(json.dumps({"verdict": "PASS", "isaac_chunks": len(items), "rendered_chunks": 0,
                              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    contexts = expected_chain_contexts(items, source_root=source, rendered_root=rendered)
    images = {view: [Image.new("RGB", (1, 1), (20, 30, 40)) for _ in range(81)]
              for view in ("head_color", "hand_left", "hand_right")}
    for row in items:
        name = Path(row["isaac_npz"]).name
        write_raven_rendered_chunk(source / name, rendered / name, images, chain_context=contexts[name])
    return items, source, rendered, contexts


def rewrite_npz(path, mutation):
    with np.load(path, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    mutation(fields)
    np.savez_compressed(path, **fields)


def test_stratified_sample_is_deterministic_and_covers_roles_and_modes(tmp_path):
    items, _, _, _ = pool(tmp_path)
    names = select_deep_sample(items, 3, seed=25)
    assert names == select_deep_sample(list(reversed(items)), 3, seed=25)
    selected = [r for r in items if Path(r["isaac_npz"]).name in names]
    assert {r["split"] for r in selected} == {"train", "validation", "test"}
    assert {r["motion_id"] for r in selected} == {"m0", "m1"}
    assert len(select_deep_sample(items, 100)) == 6


def test_sampled_pool_hashes_every_output_once_and_does_not_decode_other_rgb(tmp_path, monkeypatch):
    import pipeline.bc_rendering.validate_bc_teacher_render as module
    items, source, rendered, _ = pool(tmp_path)
    sample = set(select_deep_sample(items, 3))
    original_hash, original_get = module.sha256_file, np.lib.npyio.NpzFile.__getitem__
    hashes, decoded = [], []

    def counted_hash(path):
        hashes.append(Path(path).name)
        return original_hash(path)

    def counted_get(self, key):
        if key.startswith("rgb_gen"):
            decoded.append((Path(self.zip.filename).name, key))
        return original_get(self, key)

    monkeypatch.setattr(module, "sha256_file", counted_hash)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted_get)
    result = validate_pool(source_root=source, rendered_root=rendered,
        rgb_validation="stratified_sample", deep_rgb_sample_count=3)
    assert result["verdict"] == "PASS"
    assert {name for name, _ in decoded} == sample
    assert len(result["rendered_identities"]) == 6
    for row in items:
        name = Path(row["isaac_npz"]).name
        assert hashes.count(name) == 1
        assert result["rendered_identities"][name]["sha256"] == original_hash(rendered / name)
    audit = result["rgb_validation"]
    assert audit["deep_count"] == 3 and set(audit["deep_sampled_files"]) == sample
    assert audit["coverage"]["split"]["deep_sampled"] == ["test", "train", "validation"]


def test_unsampled_corrupt_rgb_header_and_small_teacher_values_are_rejected(tmp_path):
    items, source, rendered, contexts = pool(tmp_path)
    sample = set(select_deep_sample(items, 1))
    name = next(Path(r["isaac_npz"]).name for r in items if Path(r["isaac_npz"]).name not in sample)
    rewrite_npz(rendered / name, lambda f: f.update(rgb_gen__hand_left=np.zeros((81, 108, 191, 3), np.uint8)))
    result = validate_pool(source_root=source, rendered_root=rendered,
        rgb_validation="stratified_sample", deep_rgb_sample_count=1)
    assert result["verdict"] == "FAIL"
    assert any("NPY header" in e for r in result["failed_chunks"] for e in r["errors"])
    other = next(Path(r["isaac_npz"]).name for r in items if Path(r["isaac_npz"]).name not in sample | {name})
    rewrite_npz(rendered / other, lambda f: f.update(gt_actions=np.full((81, 26), 999, np.float32)))
    errors = validate_chunk(source / other, rendered / other, chain_context=contexts[other], deep_rgb=False)
    assert any("teacher/source value changed: gt_actions" in e for e in errors)


def test_bad_npy_header_and_all_selected_sample_matches_full_checks(tmp_path):
    _, source, rendered, contexts = pool(tmp_path)
    full = validate_pool(source_root=source, rendered_root=rendered)
    sampled = validate_pool(source_root=source, rendered_root=rendered,
        rgb_validation="stratified_sample", deep_rgb_sample_count=100)
    assert full["verdict"] == sampled["verdict"] == "PASS"
    assert full["rendered_identities"] == sampled["rendered_identities"]
    assert sampled["rgb_validation"]["deep_count"] == 6
    path = next(rendered.glob("*.npz"))
    with zipfile.ZipFile(path) as archive:
        members = {entry.filename: archive.read(entry) for entry in archive.infolist()}
    members["rgb_gen__hand_right.npy"] = b"not-a-valid-npy-header"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    errors = validate_chunk(source / path.name, path, chain_context=contexts[path.name], deep_rgb=False)
    assert any("NPZ validation failed" in error for error in errors)


@pytest.mark.parametrize("count", [0, -1, True])
def test_sample_requires_positive_explicit_count(count):
    with pytest.raises(ValueError):
        select_deep_sample([], count)
