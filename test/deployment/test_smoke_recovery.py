import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment import beta_e2e_smoke as module


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    old = tmp_path / "runtime/old"
    new = tmp_path / "runtime/new"
    old.mkdir(parents=True)
    new.mkdir()
    run = "dagger-beta-smoke-old"
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    save(old / "status.json", {"schema": "roborender.beta-e2e-smoke.v1", "run_id": run,
         "phase": "FAILED", "production_promoted": False, "native_replay_allowed": False, "source_sha256": "a"*64})
    candidate = tmp_path / "runtime/migrations/alpha-to-beta-20261008/h200" / run / "train/fresh-bc-disposable-candidate.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"checkpoint")
    save(old / "bc-training.json", {"result": "PASS", "optimizer_steps": 1, "checkpoint_sha256": module.BC_SHA,
         "sample_chunks": 16, "frames_per_chunk": 81, "production_promoted": False,
         "candidate_sha256": module.file_sha256(candidate)})
    save(old / "bc-collection.json", {"vector_steps": 243, "optimizer_steps": 0})
    motion = tmp_path / "pipeline/dagger/rtx/config/fixed_panel_motion_inventory.v1.json"
    save(motion, {})
    panel = module.make_panel(module.file_sha256(motion), smoke=True)
    save(old / "panel.json", panel)
    manifest = tmp_path / "manifest.json"
    save(manifest, {"source_sha256": "b"*64})
    monkeypatch.setattr(module, "verify", Mock())
    monkeypatch.setattr(module, "make_first_round_config", Mock(return_value={}))
    pre_run = "dagger-beta-smoke-prebc-"+hashlib.sha256(run.encode()).hexdigest()[:16]
    eid = "success-beta-panel-"+hashlib.sha256(pre_run.encode()).hexdigest()[:16]+"-r1"
    comparison = {"run_id": eid, "complete": True, "arms": {"bc": {}, "dagger": {}}}
    remote = SimpleNamespace(source=Path("/frozen"), json=Mock(return_value={"ok": True}),
        process=Mock(return_value={"state": "COMPLETE", "exit_code": 0}),
        verify_containers_stopped=Mock(return_value=True), read=Mock(return_value=comparison))
    monkeypatch.setattr(module, "Site", Mock(return_value=remote))
    validate = Mock()
    monkeypatch.setattr(module, "validate_comparison", validate)
    smoke = module.Smoke.__new__(module.Smoke)
    smoke.role = "beta"
    smoke.root, smoke.run = new, "dagger-beta-smoke-recovered"
    smoke.site = {"source_manifest": str(manifest), "rtx_python": "/python",
                  "rtx_source_manifest": "/manifest", "rtx_repo": "/home/ldu/roborender"}
    smoke.phase = Mock()
    smoke.allocation = Mock(return_value={"present": False})
    smoke.finish_bc_and_run_dagger = Mock()
    return smoke, old, remote, validate


def test_recovery_reads_only_completed_evaluation_and_preserves_prior_state(tmp_path, monkeypatch):
    smoke, old, remote, validate = fixture(tmp_path, monkeypatch)
    original = (old / "status.json").read_bytes()
    smoke.recover_completed_bc(old)
    remote.process.assert_called_once_with("evaluation", "status", host="beta")
    smoke.allocation.assert_called_once_with(remote, "status")
    smoke.finish_bc_and_run_dagger.assert_called_once()
    validate.assert_called_once()
    receipt = json.loads((smoke.root / "recovery.json").read_bytes())
    assert receipt["native_work_repeated"] is False and receipt["production_promoted"] is False
    assert (old / "status.json").read_bytes() == original


@pytest.mark.parametrize("state", ["RUNNING", "UNCERTAIN", "FAILED", "ABSENT"])
def test_recovery_never_launches_incomplete_native_evaluation(tmp_path, monkeypatch, state):
    smoke, old, remote, _ = fixture(tmp_path, monkeypatch)
    remote.process.return_value = {"state": state}
    with pytest.raises(RuntimeError, match="not complete"):
        smoke.recover_completed_bc(old)
    remote.process.assert_called_once_with("evaluation", "status", host="beta")
    smoke.finish_bc_and_run_dagger.assert_not_called()


@pytest.mark.parametrize("bad", ["campaign", "checkpoint", "panel", "allocation", "comparison"])
def test_recovery_rejects_ambiguous_or_changed_evidence(tmp_path, monkeypatch, bad):
    smoke, old, remote, _ = fixture(tmp_path, monkeypatch)
    if bad == "campaign": (old / "campaign").mkdir()
    if bad == "checkpoint":
        value = json.loads((old / "bc-training.json").read_bytes())
        value["candidate_sha256"] = "0"*64
        (old / "bc-training.json").write_text(json.dumps(value))
    if bad == "panel": (old / "panel.json").write_text("{}")
    if bad == "allocation": smoke.allocation.return_value = {"present": True}
    if bad == "comparison": remote.read.return_value["complete"] = False
    with pytest.raises(RuntimeError): smoke.recover_completed_bc(old)
    smoke.finish_bc_and_run_dagger.assert_not_called()
