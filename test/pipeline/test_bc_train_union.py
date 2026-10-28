"""CPU-only additive train-union contracts; no GPU or real payload decoding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.bc_training import data_admission
from pipeline.bc_training import train_bc_h200 as trainer
from pipeline.bc_training import train_union


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def base(tmp_path, monkeypatch):
    root = tmp_path / "frozen-data"
    source = root / "manifests" / "independent_v2.json"
    split_dir = root / "splits"
    panel = write_json(tmp_path / "historical-panel.json", {
        "schema": "cpu-only-historical-panel", "episodes": 295,
        "observation_mode": "raytraced_3view", "simulator_seeds": [1001, 1002]})
    test = {"episode_count": 295, "offline_payload_count": 0,
        "purpose": "test_only_not_checkpoint_selection", "executed": False,
        "observation_mode": "raytraced_3view",
        "panel": {"path": str(panel), "raw_sha256": sha(panel)},
        "simulator_seeds": [1001, 1002]}
    roles = {"schema": "roborender.bc-artifact-roles.v1",
        "offline_roles": ["train", "validation"],
        "offline_chunk_counts": {"train": 16, "validation": 16},
        "offline_test_chunks": 0, "closed_loop_test": test}
    rows, documents = [], {}
    for rank, role in enumerate(("train", "validation")):
        items = []
        for lane in range(16):
            cid = f"chunk-{role}-{lane:02d}"
            raw = root / "isaac" / f"{cid}.npz"
            rendered = root / "rendered" / f"{cid}.npz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            rendered.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(("CPU fixture raw " + cid).encode())
            rendered.write_bytes(("CPU fixture rendered " + cid).encode())
            item = {"chunk_id": cid, "isaac_npz": str(raw.relative_to(root)),
                "rendered_npz": str(rendered.relative_to(root)),
                "stream_id": f"stream-{role}-{lane}"}
            items.append(item)
            rows.append({**item, "split": role, "frames": 81,
                "source_sha256": sha(raw), "source_size_bytes": raw.stat().st_size,
                "rendered_sha256": sha(rendered), "rendered_size_bytes": rendered.stat().st_size,
                "teacher_only": True, "selected_for_split": True, "overcollection": False,
                "motion_id": f"motion-{lane}", "motion_sha256": f"{lane + 1:064x}",
                "simulator_seed": 40 + rank, "source_run_tag": "cpu-only-" + role,
                "rank": rank, "local_env_id": lane, "episode_id": f"{role}-episode-{lane}",
                "render_contract": "raven_three_view_independent_v2",
                "render_chain_length": 1, "render_chain_index": 0,
                "render_inherits_previous_reference": False})
        documents[role] = {"schema": "roborender.bc_split.v1", "split": role,
            "chunk_count": len(items), "source_manifest": "../manifests/independent_v2.json",
            "items": items}
    manifest = {"schema": "roborender.bc_pool.rendered.v2", "chunk_count": 32,
        "frames_per_chunk": 81, "chunks": rows, "artifact_roles": roles,
        "render_recipe": "raven_3view_50step_independent_v2",
        "render_contract": "raven_three_view_independent_v2",
        "partition_counts": {"train": 16, "validation": 16}}
    write_json(source, manifest)
    for role, doc in documents.items():
        doc["source_manifest_sha256"] = sha(source)
        write_json(split_dir / f"{role}.json", doc)
    reference = write_json(split_dir / "success_test.json", {
        "schema": "roborender.bc-closed-loop-test-reference.v1", **test})
    source_pool = write_json(root / "manifest.json", {
        "schema": "cpu-only-native-pool", "chunk_count": 32,
        "chunks": [{key: value for key, value in row.items()
                    if not key.startswith("render")} for row in rows]})
    protocol = write_json(tmp_path / "production-protocol.json", {
        "schema": "roborender.bc-production-protocol.v4",
        "source_pool": {"path": str(source_pool), "raw_sha256": sha(source_pool)}})
    report = {"schema": "roborender.bc-data-admission.v3", "purpose": "production",
        "run_name": "cpu-only-base", "production_admitted": True,
        "qualification_admitted": False, "metadata_independence_passed": True,
        "independent_render_required": True, "missing_provenance": {},
        "motion_support_matches": True, "test_payloads_opened": False,
        "source_manifest": str(source), "source_manifest_sha256": sha(source),
        "artifact_roles": roles,
        "protocol": {"path": str(protocol), "raw_sha256": sha(protocol)},
        "split_files": {role: {"path": str(split_dir / f"{role}.json"),
            "sha256": sha(split_dir / f"{role}.json")} for role in documents},
        "splits": {role: {"chunks": 16, "frames": 1296} for role in documents}}
    report_path = write_json(split_dir / "data-admission.json", report)
    calls = []

    def validate_original(path, train, validation, **kwargs):
        # Stand in only for the expensive historical native/render receipt
        # audit. Preserve its real raw metadata binding checks in this fixture.
        assert Path(path) == report_path
        actual = json.loads(report_path.read_text())
        assert Path(train) == Path(actual["split_files"]["train"]["path"])
        assert Path(validation) == Path(actual["split_files"]["validation"]["path"])
        if sha(actual["source_manifest"]) != actual["source_manifest_sha256"]:
            raise ValueError("base source manifest changed")
        for binding in actual["split_files"].values():
            if sha(binding["path"]) != binding["sha256"]:
                raise ValueError("base split changed")
        bound_panel = actual["artifact_roles"]["closed_loop_test"]["panel"]
        if sha(bound_panel["path"]) != bound_panel["raw_sha256"]:
            raise ValueError("base test panel changed")
        protocol_ref = actual["protocol"]
        if sha(protocol_ref["path"]) != protocol_ref["raw_sha256"]:
            raise ValueError("base protocol changed")
        pool_ref = json.loads(Path(protocol_ref["path"]).read_text())["source_pool"]
        if sha(pool_ref["path"]) != pool_ref["raw_sha256"]:
            raise ValueError("base native pool changed")
        calls.append((Path(path), Path(train), Path(validation)))
        return actual

    monkeypatch.setattr(data_admission, "validate_training_admission", validate_original)
    preserved = [source, report_path, reference, panel, source_pool, protocol,
                 split_dir / "train.json", split_dir / "validation.json"]
    return {"root": root, "source": source, "manifest": manifest,
        "documents": documents, "report": report, "report_path": report_path,
        "panel": panel, "reference": reference, "calls": calls,
        "source_pool": source_pool, "protocol": protocol,
        "preserved": {p: p.read_bytes() for p in preserved},
        "output": tmp_path / "new-artifact-root" / "merged2880-fixture"}


def build(base):
    return train_union.build_union(base["report_path"], base["root"], base["output"])


def validate(base, *, root=None):
    output = base["output"]
    return train_union.validate_training_admission(output / "data-admission.json",
        output / "train.json", output / "validation-retired.json",
        base["root"] if root is None else root)


def test_union_adds_all_former_validation_without_modifying_frozen_inputs(base):
    report = build(base)
    assert report["schema"] == "roborender.bc-train-union-admission.v1"
    assert validate(base) == report
    train = json.loads((base["output"] / "train.json").read_text())
    retired = json.loads((base["output"] / "validation-retired.json").read_text())
    assert train["split"] == "train" and train["chunk_count"] == 32
    originals = {x["chunk_id"]: x for doc in base["documents"].values() for x in doc["items"]}
    assert {x["chunk_id"] for x in train["items"]} == set(originals)
    assert len({x["rendered_npz"] for x in train["items"]}) == 32
    for row in train["items"]:
        assert all(row[key] == value for key, value in originals[row["chunk_id"]].items())
    assert retired["split"] == "validation" and retired["retired"] is True
    assert retired["chunk_count"] == 0 and retired["items"] == []
    assert base["calls"], "the original native/rendered admission must still be checked"
    assert all(path.read_bytes() == raw for path, raw in base["preserved"].items())


def test_union_publication_is_immutable_and_idempotent(base):
    first = build(base)
    saved = {p: p.read_bytes() for p in base["output"].iterdir() if p.is_file()}
    assert build(base) == first
    assert all(path.read_bytes() == raw for path, raw in saved.items())


@pytest.mark.parametrize("target", ["train.json", "validation-retired.json", "data-admission.json"])
def test_existing_conflicting_union_artifact_is_never_overwritten(base, target):
    base["output"].mkdir(parents=True)
    path = base["output"] / target
    raw = b'{"preexisting_user_artifact":true}\n'
    path.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        build(base)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["role", "offline_test", "test_index", "not_production"])
def test_no_test_role_or_unadmitted_pool_can_be_promoted(base, fault):
    report = deepcopy(base["report"])
    if fault == "role":
        report["artifact_roles"]["offline_roles"] = ["train", "validation", "test"]
    elif fault == "offline_test":
        report["artifact_roles"]["offline_test_chunks"] = 1
    elif fault == "test_index":
        report["split_files"]["test"] = deepcopy(report["split_files"]["validation"])
    else:
        report["production_admitted"] = False
    write_json(base["report_path"], report)
    with pytest.raises(ValueError):
        build(base)


@pytest.mark.parametrize("field", ["chunk_id", "rendered_npz", "isaac_npz"])
def test_duplicate_union_identity_or_payload_path_is_rejected(base, field):
    train = base["documents"]["train"]
    validation = deepcopy(base["documents"]["validation"])
    validation["items"][0][field] = train["items"][0][field]
    path = Path(base["report"]["split_files"]["validation"]["path"])
    write_json(path, validation)
    report = deepcopy(base["report"])
    report["split_files"]["validation"]["sha256"] = sha(path)
    write_json(base["report_path"], report)
    with pytest.raises(ValueError):
        build(base)


@pytest.mark.parametrize("target", ["source", "train", "validation", "panel", "report",
                                   "protocol", "source_pool"])
def test_original_metadata_cannot_change_after_union_admission(base, target):
    build(base)
    if target in {"train", "validation"}:
        path = Path(base["report"]["split_files"][target]["path"])
    else:
        path = base["report_path" if target == "report" else target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        validate(base)


@pytest.mark.parametrize("target", ["train.json", "validation-retired.json"])
def test_merged_indexes_cannot_change_after_admission(base, target):
    build(base)
    path = base["output"] / target
    doc = json.loads(path.read_text())
    if target == "train.json":
        doc["items"][0]["rendered_npz"] = doc["items"][1]["rendered_npz"]
    else:
        doc["retired"] = False
    write_json(path, doc)
    with pytest.raises(ValueError):
        validate(base)


def test_dataset_root_cannot_be_rebound(base, tmp_path):
    build(base)
    with pytest.raises(ValueError):
        validate(base, root=tmp_path)


def test_initial_root_cannot_substitute_same_size_unverified_payloads(base, tmp_path):
    replacement = tmp_path / "same-sized-but-not-admitted-data"
    for doc in base["documents"].values():
        for row in doc["items"]:
            for field in ("isaac_npz", "rendered_npz"):
                target = replacement / row[field]
                target.parent.mkdir(parents=True, exist_ok=True)
                original = base["root"] / row[field]
                target.write_bytes(b"x" * original.stat().st_size)
    with pytest.raises(ValueError):
        train_union.build_union(base["report_path"], replacement, base["output"])


@pytest.mark.parametrize("field", ["isaac_npz", "rendered_npz"])
def test_missing_raw_or_rendered_payload_is_rejected(base, field):
    build(base)
    (base["root"] / base["documents"]["validation"]["items"][0][field]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(base)


@pytest.mark.parametrize("field", ["isaac_npz", "rendered_npz"])
def test_payload_symlink_escape_is_rejected(base, tmp_path, field):
    row = base["documents"]["validation"]["items"][0]
    payload = base["root"] / row[field]
    external = tmp_path / "outside-the-frozen-dataset.npz"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(external)
    with pytest.raises(ValueError):
        build(base)


def test_merged_dataset_can_only_enter_fixed_final_training(base):
    build(base)
    args = trainer.parse_args(["--max-epochs", "2", "--run-name", "cpu-only-union",
        "--dataset-root", str(base["root"]), "--train-split", str(base["output"] / "train.json"),
        "--validation-split", str(base["output"] / "validation-retired.json"),
        "--data-admission-json", str(base["output"] / "data-admission.json")])
    admitted = trainer.load_data_admission(args)
    assert admitted["schema"] == train_union.SCHEMA
    args.training_protocol = "legacy_validation"
    with pytest.raises(ValueError):
        trainer.load_data_admission(args)


def test_merged_data_never_silently_rebinds_bc116_resume(base):
    build(base)
    expected = {"schema": train_union.SCHEMA,
        "report_sha256": sha(base["output"] / "data-admission.json")}
    checkpoint = {"data_admission": {"report_sha256": sha(base["report_path"])}}
    with pytest.raises(ValueError):
        trainer.validate_resume_data_admission(checkpoint, expected)
