from copy import deepcopy
import hashlib
import json

import pytest

from pipeline.deployment.infrastructure_overlay import changed_control_paths, verify_overlay
from pipeline.deployment.source_snapshot import canonical


def manifest(rows):
    rows=sorted(rows,key=lambda r:r["path"])
    return {"schema":"roborender.source-snapshot.v1","files":rows,"file_count":len(rows),
            "source_sha256":hashlib.sha256(canonical(rows)).hexdigest()}


def row(path,sha="a"):
    return {"path":path,"sha256":sha*64,"bytes":1,"executable":False}


def test_control_release_never_changes_native_or_learner():
    native=manifest([row("pipeline/deployment/beta_site.py"),row("pipeline/dagger/services/raven_learner_worker.py")])
    management=manifest([row("pipeline/deployment/beta_site.py","b"),row("pipeline/dagger/services/raven_learner_worker.py")])
    assert changed_control_paths(native,management)==["pipeline/deployment/beta_site.py"]
    for path in ("pipeline/dagger/services/raven_learner_worker.py","pipeline/dagger/rtx/code/src/dagger_rtx/resident_service.py",
                 "renderer/contracts/CONDITIONING_CONTRACT.md"):
        bad=manifest([*management["files"],row(path,"c")])
        with pytest.raises(RuntimeError):changed_control_paths(native,bad)
    with pytest.raises(RuntimeError):changed_control_paths(native,manifest([management["files"][0]]))


def test_overlay_pin_and_full_current_tree_are_checked(tmp_path,monkeypatch):
    native=manifest([row("pipeline/deployment/beta_site.py")])
    management=manifest([row("pipeline/deployment/beta_site.py","b")])
    p=tmp_path/"manifest.json";p.write_bytes(canonical(management))
    admission={"schema":"roborender.control-only-release.v1","native_source_sha256":native["source_sha256"],
        "management_manifest":str(p),"management_source_sha256":management["source_sha256"],
        "changed_paths":["pipeline/deployment/beta_site.py"]}
    q=tmp_path/"overlay.json";q.write_bytes(canonical(admission));sha=hashlib.sha256(q.read_bytes()).hexdigest()
    calls=[]
    monkeypatch.setattr("pipeline.deployment.infrastructure_overlay.verify",lambda root,m:calls.append((root,m)))
    result=verify_overlay(tmp_path,native,q,sha)
    assert result["native_source_sha256"]==native["source_sha256"] and calls==[(tmp_path,management)]
    with pytest.raises(RuntimeError,match="changed"):verify_overlay(tmp_path,native,q,"f"*64)
    monkeypatch.setattr("pipeline.deployment.infrastructure_overlay.verify",lambda *a:(_ for _ in ()).throw(ValueError("drift")))
    with pytest.raises(ValueError,match="drift"):verify_overlay(tmp_path,native,q,sha)


def test_reviewed_metadata_delta_is_exact_not_blanket_learner_permission():
    from pipeline.deployment.infrastructure_overlay import REVIEWED_METADATA_DELTAS
    for path, (old_sha, new_sha) in REVIEWED_METADATA_DELTAS.items():
        before = [] if old_sha is None else [{**row(path), "sha256": old_sha}]
        after = [{**row(path), "sha256": new_sha}]
        assert changed_control_paths(manifest(before), manifest(after)) == [path]
        with pytest.raises(RuntimeError):
            changed_control_paths(manifest(before), manifest([{**after[0], "sha256": "f" * 64}]))
        with pytest.raises(RuntimeError):
            changed_control_paths(manifest(before), manifest([{**after[0], "executable": True}]))


def test_reader_overlay_is_not_a_general_ledger_exemption():
    from pipeline.deployment.infrastructure_overlay import CONTROL_ONLY, REVIEWED_METADATA_DELTAS
    path = "pipeline/dagger/services/ledger_owner.py"
    assert path not in CONTROL_ONLY
    old, new = REVIEWED_METADATA_DELTAS[path]
    assert old == "0b8935354244bc4572e868f20ebd1ff7aaa2d61ffee6befea90837e5f1aa8a34"
    assert new == "6092451ec8aafec378e411f5c148807d30e13e37874cc01cacf45aaa4dcbc10b"
    with pytest.raises(RuntimeError):
        changed_control_paths(manifest([{**row(path), "sha256": "e" * 64}]),
                              manifest([{**row(path), "sha256": new}]))


def test_explicit_control_release_namespaces_operation_without_native_reset(tmp_path):
    from pipeline.dagger.beta_campaign import Campaign
    # Exercise only pure context creation; no service or simulator is created.
    obj = object.__new__(Campaign)
    obj.root = tmp_path
    obj.first = {"student": {"initial_checkpoint_sha256": "a" * 64,
                             "initial_checkpoint": "/policy.pt"}}
    obj.state = {"run_id": "test", "round_id": 1, "config": {}}
    original = obj.operation_context()
    path = tmp_path / "overlay.json"; path.write_bytes(b"{}")
    release = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    obj.state["control_only_release"] = release
    assert obj.operation_context() == {**original, "control_only_release": release}
    assert "infrastructure_attempts" not in obj.state
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="binding changed"):
        obj.operation_context()
