import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment import beta_e2e_smoke as m


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    old = tmp_path / "runtime/old"
    new = tmp_path / "runtime/new"
    old.mkdir(parents=True)
    new.mkdir()
    prior = "dagger-leo-smoke-old"
    pre = "dagger-leo-smoke-prebc-named-20261013"
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    save(old / "status.json", {"schema": "roborender.leo-e2e-smoke.v1", "run_id": prior,
        "phase": "FAILED", "production_promoted": False, "native_replay_allowed": False,
        "source_sha256": "a"*64})
    work = tmp_path / "runtime/migrations/beta-to-leo-20261013/h200" / prior
    candidate = work / "train/fresh-bc-disposable-candidate.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"checkpoint")
    rendered = work / "render/rendered.npz"
    rendered.parent.mkdir(parents=True)
    rendered.write_bytes(b"rendered")
    training = {"result": "PASS", "optimizer_steps": 1, "checkpoint_sha256": m.BC_SHA,
        "sample_chunks": 16, "frames_per_chunk": 81, "production_promoted": False,
        "candidate_sha256": m.file_sha256(candidate), "source_sha256": m.file_sha256(rendered),
        "train_mse": .1, "grad_norm": .2}
    save(work / "train/train-result.json", training)
    save(old / "bc-training.json", training)
    save(work / "render/render-result.json", {"result": "PASS", "all_source_arrays_preserved": True,
        "production_promoted": False, "output_sha256": m.file_sha256(rendered)})
    save(old / "bc-collection.json", {"result": "PASS", "native_teacher_only": True, "num_envs": 1,
        "vector_steps": 243, "teacher_action_equalities": 243, "optimizer_steps": 0})
    for label in ("bc-render", "bc-train"):
        save(old / f"prebc/round-1/processes/{label}/result.json",
             {"state": "COMPLETE", "exit_code": 0, "owner": pre})
    save(old / "prebc/round-1/processes/bc-train.json", {"owner": pre})
    motion = tmp_path / "pipeline/dagger/rtx/config/fixed_panel_motion_inventory.v1.json"
    save(motion, {})
    panel = m.make_panel(m.file_sha256(motion), smoke=True)
    save(old / "panel.json", panel)
    manifest = tmp_path / "manifest.json"
    save(manifest, {"source_sha256": "b"*64})
    comparison = new / "comparison.json"
    save(comparison, {"arms": {"bc": {}, "dagger": {}}})
    remote = SimpleNamespace(source=Path("/frozen"),
        json=Mock(side_effect=[{"present": False}, {"ok": True}]),
        process=Mock(return_value={"state": "ABSENT"}), verify_containers_stopped=Mock())
    current = SimpleNamespace(evaluate=Mock(return_value={"comparison_path": str(comparison)}))
    monkeypatch.setattr(m, "Site", Mock(side_effect=[remote, current]))
    monkeypatch.setattr(m, "verify", Mock())
    monkeypatch.setattr(m, "site_host", lambda _: "leo-rtx-6k")
    monkeypatch.setattr(m, "make_first_round_config", Mock(return_value={}))
    monkeypatch.setattr(m, "validate_comparison", Mock())
    smoke = m.Smoke.__new__(m.Smoke)
    smoke.role, smoke.run, smoke.root = "leo", "dagger-leo-smoke-new", new
    smoke.site = {"source_manifest": str(manifest), "broker_python": "/python", "rtx_python": "/python",
                  "rtx_source_manifest": "/manifest"}
    save(tmp_path / "pipeline/deployment/h200/beta_pod.py", {})
    smoke.phase = Mock()
    smoke.begin_bc_logging = Mock(return_value=Mock())
    smoke.finish_bc_and_run_dagger = Mock()
    return smoke, old, remote, current, candidate, work


def test_completed_training_recovery_does_not_repeat_native_or_optimizer_work(tmp_path, monkeypatch):
    smoke, old, remote, current, _, _ = fixture(tmp_path, monkeypatch)
    original = (old / "status.json").read_bytes()
    smoke.resume_after_bc_training(old)
    remote.process.assert_called_once_with("evaluation", "status", host="beta")
    assert remote.json.call_args_list[0].args[0][2] == "status"
    current.evaluate.assert_called_once()
    smoke.finish_bc_and_run_dagger.assert_called_once()
    receipt = json.loads((smoke.root / "recovery.json").read_bytes())
    assert receipt["native_work_repeated"] is False and receipt["bc_optimizer_updates_repeated"] == 0
    assert (old / "status.json").read_bytes() == original


@pytest.mark.parametrize("bad", ["candidate", "render", "training", "collection", "process", "panel", "campaign", "host"])
def test_post_training_recovery_rejects_changed_or_uncertain_evidence(tmp_path, monkeypatch, bad):
    smoke, old, remote, current, candidate, work = fixture(tmp_path, monkeypatch)
    if bad == "candidate": candidate.write_bytes(b"changed")
    if bad == "render": (work / "render/rendered.npz").write_bytes(b"changed")
    if bad == "training": (work / "train/train-result.json").write_text("{}")
    if bad == "collection": (old / "bc-collection.json").write_text("{}")
    if bad == "process": (old / "prebc/round-1/processes/bc-train/result.json").write_text('{"state":"RUNNING"}')
    if bad == "panel": (old / "panel.json").write_text("{}")
    if bad == "campaign": (old / "campaign").mkdir()
    if bad == "host": smoke.role = "beta"
    with pytest.raises((RuntimeError, ValueError)): smoke.resume_after_bc_training(old)
    current.evaluate.assert_not_called()


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETE", "FAILED", "UNCERTAIN"])
def test_post_training_recovery_never_restarts_existing_evaluation(tmp_path, monkeypatch, state):
    smoke, old, remote, current, _, _ = fixture(tmp_path, monkeypatch)
    remote.process.return_value = {"state": state}
    with pytest.raises(RuntimeError, match="native replay"): smoke.resume_after_bc_training(old)
    current.evaluate.assert_not_called()


def test_post_training_recovery_requires_released_h200(tmp_path, monkeypatch):
    smoke, old, remote, current, _, _ = fixture(tmp_path, monkeypatch)
    remote.json.side_effect = [{"present": True}]
    with pytest.raises(RuntimeError, match="released"): smoke.resume_after_bc_training(old)
    current.evaluate.assert_not_called()
