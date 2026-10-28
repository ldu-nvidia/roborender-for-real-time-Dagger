import copy
import pytest

from pipeline.bc_training.data_admission import audit_partitions, INDEPENDENT_RECIPE


def fixture():
    rows, docs = [], {}
    for rank, split in enumerate(("train", "validation", "test")):
        row = dict(chunk_id=split, isaac_npz=split + ".src.npz", rendered_npz=split + ".rgb.npz",
                   source_run_tag="run", source_batch=1, rank=rank, local_env_id=0,
                   source_sha256=str(rank) * 64, motion_id="same-population-mode", simulator_seed=rank,
                   episode_id="episode-" + split, motion_sha256="a" * 64,
                   request_sha256=str(rank) * 64, sampling_plan_sha256="b" * 64,
                   teacher_only=True, split=split, selected_for_split=True, overcollection=False,
                   render_contract=INDEPENDENT_RECIPE, render_chain_id=split,
                   render_chain_length=1, render_chain_index=0, render_inherits_previous_reference=False)
        rows.append(row)
        docs[split] = dict(split=split, chunk_count=1, items=[{k: row[k] for k in
                          ("chunk_id", "isaac_npz", "rendered_npz")}])
    return dict(frames_per_chunk=81, chunks=rows), docs


def test_independent_population_matched_metadata():
    source, docs = fixture()
    result = audit_partitions(source, docs, require_independent_render=True)
    assert not result["production_admitted"] and result["motion_support_matches"]
    assert result["purpose"] == "metadata_only"
    assert not result["test_payloads_opened"]


def test_shared_generated_reference_is_leakage_even_with_disjoint_streams():
    source, docs = fixture()
    source["chunks"][1]["render_chain_id"] = source["chunks"][0]["render_chain_id"]
    with pytest.raises(ValueError, match="render_chain"):
        audit_partitions(source, docs)


@pytest.mark.parametrize("field,value", [("render_chain_length", 6),
    ("render_inherits_previous_reference", True), ("render_contract", "raven_three_view_autoregressive_v1")])
def test_legacy_render_not_admitted(field, value):
    source, docs = fixture()
    source["chunks"][0][field] = value
    with pytest.raises(ValueError, match="independent Raven"):
        audit_partitions(source, docs, require_independent_render=True)


def test_missing_motion_identity_is_not_invented_from_lane_id():
    source, docs = fixture()
    for row in source["chunks"]:
        del row["motion_id"]
    result = audit_partitions(source, docs, require_independent_render=True)
    assert result["metadata_independence_passed"]
    assert not result["production_admitted"]
    assert result["missing_provenance"]["motion_id"] == 3


def test_uncovered_motion_mode_is_not_distribution_match():
    source, docs = fixture()
    source["chunks"][2]["motion_id"] = "unseen-mode"
    assert not audit_partitions(source, docs, require_independent_render=True)["production_admitted"]


def test_matching_motion_names_with_different_bytes_are_rejected():
    source, docs = fixture()
    source["chunks"][2]["motion_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different asset bytes"):
        audit_partitions(source, docs, require_independent_render=True)


def test_duplicate_source_bytes_rejected():
    source, docs = fixture()
    source["chunks"][1]["source_sha256"] = source["chunks"][0]["source_sha256"]
    with pytest.raises(ValueError, match="source_sha256"):
        audit_partitions(source, docs)


def test_noncovering_or_repeated_index_rejected():
    source, docs = fixture()
    docs["test"]["items"][0] = copy.deepcopy(docs["train"]["items"][0])
    with pytest.raises(ValueError, match="repeated chunk"):
        audit_partitions(source, docs)


def test_reused_episode_or_motion_seed_rejected():
    for key in ("episode_id", "simulator_seed"):
        source, docs = fixture()
        source["chunks"][1][key] = source["chunks"][0][key]
        with pytest.raises(ValueError, match="cross-split dependency"):
            audit_partitions(source, docs)


def test_no_production_admission_from_unattested_native_metadata():
    source, docs = fixture()
    del source["chunks"][0]["request_sha256"]
    result = audit_partitions(source, docs, require_independent_render=True)
    assert not result["production_admitted"]
    assert result["missing_provenance"]["request_sha256"] == 1


def test_preassigned_collection_roles_survive_index_creation(tmp_path):
    import json
    from pipeline.bc_training.create_bc_splits import create_splits
    template, _ = fixture()
    rows = []
    for index, role in enumerate(["train"] * 8 + ["validation", "test"]):
        row = copy.deepcopy(next(x for x in template["chunks"] if x["split"] == role))
        row.update(chunk_id=f"chunk-{index:03d}", local_env_id=index,
                   isaac_npz=f"source-{index}.npz", rendered_npz=f"rendered-{index}.npz",
                   render_chain_id=f"chain-{index}", source_sha256=f"{index:064x}")
        rows.append(row)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"frames_per_chunk": 81, "chunks": rows}))
    result = create_splits(tmp_path)
    assert "collection-time" in result["algorithm"]
    for role in ("train", "validation", "test"):
        actual = json.loads((tmp_path / "splits" / (role + ".json")).read_text())
        assert [x["chunk_id"] for x in actual["items"]] == [x["chunk_id"] for x in rows if x["split"] == role]


@pytest.mark.parametrize("change", ["missing", "quota", "plan", "count"])
def test_preassigned_roles_cannot_be_partially_or_silently_replaced(change):
    from pipeline.bc_training.create_bc_splits import _preassigned_membership
    source, _ = fixture()
    rows = source["chunks"]
    if change == "missing": del rows[0]["split"]
    if change == "quota": rows[0]["selected_for_split"] = False
    if change == "plan": rows[0]["sampling_plan_sha256"] = "c" * 64
    requested = {"train": 1, "validation": 1, "test": 2 if change == "count" else 1}
    with pytest.raises(ValueError):
        _preassigned_membership(rows, requested)
