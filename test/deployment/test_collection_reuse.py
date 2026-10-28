import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.deployment import beta_e2e_smoke as m


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    old = tmp_path / "runtime/old"
    old.mkdir(parents=True)
    new = tmp_path / "runtime/new"
    new.mkdir()
    run = "dagger-leo-smoke-prior"
    (old / "status.json").write_text(json.dumps({"schema": "roborender.leo-e2e-smoke.v1",
        "phase": "FAILED", "run_id": run, "production_promoted": False,
        "native_replay_allowed": False, "source_sha256": "a"*64}))
    row = {"path": "/dagger-runtime/bc_isaac/one.npz", "bytes": 4}
    receipt = {"result": "PASS", "native_teacher_only": True, "num_envs": 1,
               "vector_steps": 243, "teacher_action_equalities": 243, "optimizer_steps": 0, "chunks": [row]}
    (old / "bc-collection.json").write_text(json.dumps(receipt))
    local = tmp_path / "runtime/migrations/beta-to-leo-20261013/h200" / run / "fresh-isaac/one.npz"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"data")
    site = SimpleNamespace(read=Mock(side_effect=[{"state": "COMPLETE", "exit_code": 0}, receipt]),
        json=Mock(return_value={"passed": True, "output": "/home/ldu/roborender/runtime/leo-native-qualification/old"}),
        command=Mock(return_value=(hashlib.sha256(b"data").hexdigest()+" file").encode()))
    smoke = m.Smoke.__new__(m.Smoke)
    smoke.role, smoke.root = "leo", new
    smoke.site = {"rtx_repo": "/home/ldu/roborender"}
    return smoke, old, site, local


def test_reuse_verifies_remote_completion_receipt_hash_and_preserves_old_run(tmp_path, monkeypatch):
    smoke, old, site, local = fixture(tmp_path, monkeypatch)
    original = (old / "status.json").read_bytes()
    receipt, paths = smoke.reuse_collected_bc(old, site)
    assert paths == {"one.npz": local} and receipt["vector_steps"] == 243
    assert (old / "status.json").read_bytes() == original
    assert json.loads((smoke.root / "collection-reuse.json").read_bytes())["native_work_repeated"] is False
    assert site.command.call_args.args[0][0] == "sha256sum"


@pytest.mark.parametrize("bad", ["unfinished", "hash", "learning", "campaign", "not-native"])
def test_reuse_rejects_uncertain_changed_or_learning_output(tmp_path, monkeypatch, bad):
    smoke, old, site, local = fixture(tmp_path, monkeypatch)
    if bad == "unfinished": site.read.side_effect = [{"state": "RUNNING"}]
    if bad == "hash": site.command.return_value = b"bad file"
    if bad == "learning": (old / "bc-training.json").write_text("{}")
    if bad == "campaign": (old / "campaign").mkdir()
    if bad == "not-native":
        receipt = json.loads((old / "bc-collection.json").read_bytes())
        receipt["native_teacher_only"] = False
        (old / "bc-collection.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError): smoke.reuse_collected_bc(old, site)
    assert not (smoke.root / "collection-reuse.json").exists()
