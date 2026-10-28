import hashlib
from types import SimpleNamespace

import pytest

from pipeline.dagger.services.runtime_policy_contract import attest_policy_runtime, MODULES, RAVEN_COMMIT


def test_attests_actual_imported_source(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("# test policy source\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = attest_policy_runtime(resolve_module=lambda name: SimpleNamespace(__file__=str(path)),
                                   expected={"test.policy": digest})
    assert result["verified"] and result["raven_commit"] == RAVEN_COMMIT
    assert result["modules"][0]["path"] == str(path)


def test_changed_source_is_not_a_checkpoint_compatibility_exception(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("# changed forward\n")
    with pytest.raises(RuntimeError, match="differs from pinned Raven"):
        attest_policy_runtime(resolve_module=lambda name: SimpleNamespace(__file__=str(path)),
                              expected={"test.policy": "a" * 64})


def test_missing_source_rejected():
    with pytest.raises(RuntimeError, match="no attestable"):
        attest_policy_runtime(resolve_module=lambda name: SimpleNamespace(), expected={"test.policy": "a" * 64})


def test_pinned_dependency_set_is_explicit():
    assert len(MODULES) == 4
    assert all(len(sha) == 64 for sha in MODULES.values())
    assert "gr00t.rl.agents.modules.modules" in MODULES
