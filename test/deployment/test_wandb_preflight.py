from types import SimpleNamespace

import pytest

from pipeline.deployment.wandb_preflight import (admit_logging, authenticate,
                                                read_key, smoke, verify_remote_summary)


def test_absent_key_fails_without_interactive_login(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("NETRC", raising=False)
    with pytest.raises(ValueError, match="absent"):
        read_key()


def test_explicit_netrc_reuses_private_file_without_copy(tmp_path, monkeypatch):
    path = tmp_path / "wandb.netrc"
    path.write_text("machine api.wandb.ai login user password " + "b" * 40 + "\n")
    path.chmod(0o600)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(path))
    assert read_key() == "b" * 40
    assert list(tmp_path.iterdir()) == [path]
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        read_key()


def test_netrc_parse_errors_do_not_expose_credential(tmp_path, monkeypatch):
    path = tmp_path / "wandb.netrc"
    path.write_text("machine api.wandb.ai SECRET_INVALID_TOKEN")
    path.chmod(0o600)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(path))
    with pytest.raises(ValueError, match="malformed") as error:
        read_key()
    assert "SECRET_INVALID_TOKEN" not in str(error.value)


def test_netrc_does_not_accept_an_unrelated_host(tmp_path, monkeypatch):
    path = tmp_path / "wandb.netrc"
    path.write_text("machine example.com login user password " + "b" * 40)
    path.chmod(0o600)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(path))
    with pytest.raises(ValueError, match="absent"):
        read_key()


def test_explicit_credential_file_is_scoped_and_private(tmp_path):
    path = tmp_path / "credential"
    path.write_text("a" * 40)
    path.chmod(0o600)
    assert read_key(path) == "a" * 40
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        read_key(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_key(link)


def test_authentication_is_not_labeled_project_write_permission():
    def factory(**kw):
        assert kw["overrides"]["base_url"] == "https://api.wandb.ai"
        assert kw["overrides"]["project"] == "dagger"
        return SimpleNamespace(viewer=SimpleNamespace(username="fixture"))
    _, result = authenticate("a" * 40, "test-entity", api_factory=factory)
    assert result["authenticated"] and not result["project_write_verified"]
    assert "a" * 40 not in str(result)


def test_sdk_error_cannot_leak_secret():
    def factory(**kw):
        raise RuntimeError(kw["api_key"])
    with pytest.raises(RuntimeError) as caught:
        authenticate("secret-must-not-escape", "test-entity", api_factory=factory)
    assert "secret-must-not-escape" not in str(caught.value)


def test_offline_diagnostic_never_authenticates_or_claims_online(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    api, evidence = admit_logging("offline", None,
                                  api_factory=lambda **_: pytest.fail("offline cannot authenticate"))
    assert api is None and evidence["mode"] == "offline"
    assert not evidence["authenticated"] and not evidence["project_write_verified"]


def test_online_requires_entity_and_authentication(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "a" * 40)
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
    monkeypatch.setenv("WANDB_ENTITY", "restore-test-env")
    with pytest.raises(ValueError, match="explicit W&B entity"):
        admit_logging("online", None)
    _, evidence = admit_logging("online", "test-entity", api_factory=lambda **_: SimpleNamespace(
        viewer=SimpleNamespace(username="fixture")))
    assert evidence["authenticated"] and not evidence["project_write_verified"]


def test_remote_readback_is_bounded_and_requires_exact_metrics():
    calls, waits = [], []
    class Api:
        def flush(self):
            pass
        def run(self, path):
            calls.append(path)
            return SimpleNamespace(summary={"completed_updates": 2, "train/iteration_mse": 9.0})
    with pytest.raises(RuntimeError, match="not verified"):
        verify_remote_summary(Api(), "entity", "run", {
            "completed_updates": 3, "train/iteration_mse": 0.5}, sleep=waits.append)
    assert len(calls) == 5 and waits == [2, 2, 2, 2]


def test_remote_readback_retries_transient_cloud_visibility():
    calls, waits = [], []
    class Api:
        def flush(self):
            pass
        def run(self, path):
            calls.append(path)
            return SimpleNamespace(summary={} if len(calls) == 1 else {"completed_updates": 3})
    verify_remote_summary(Api(), "entity", "run", {"completed_updates": 3}, sleep=waits.append)
    assert len(calls) == 2 and waits == [2]


@pytest.mark.parametrize("nested", [True, False])
def test_remote_readback_accepts_wandb_protocol_encoding(nested):
    protocol = {"sha256": "exact-hash", "reset": {"steps": 0}, "sizes": [2, 2, 1]}
    summary = ({"test/protocol": protocol} if nested else {
        "test/protocol.sha256": "exact-hash", "test/protocol.reset.steps": 0,
        "test/protocol.sizes": {"_type": "large-array", "value": [2, 2, 1]}})
    api = SimpleNamespace(flush=lambda: None, run=lambda _: SimpleNamespace(summary=summary))
    verify_remote_summary(api, "entity", "run", {"test/protocol": protocol})


@pytest.mark.parametrize("change", ["missing", "wrong_reset", "wrong_array", "untagged"])
def test_remote_readback_rejects_changed_flattened_protocol(change):
    summary = {"test/protocol.sha256": "exact-hash", "test/protocol.reset.steps": 0,
               "test/protocol.sizes": {"_type": "large-array", "value": [2, 2, 1]}}
    if change == "missing": del summary["test/protocol.reset.steps"]
    elif change == "wrong_reset": summary["test/protocol.reset.steps"] = 1
    elif change == "wrong_array": summary["test/protocol.sizes"]["value"] = [2, 2, 2]
    else: del summary["test/protocol.sizes"]["_type"]
    api = SimpleNamespace(flush=lambda: None, run=lambda _: SimpleNamespace(summary=summary))
    with pytest.raises(RuntimeError, match="not verified"):
        verify_remote_summary(api, "entity", "run", {"test/protocol": {
            "sha256": "exact-hash", "reset": {"steps": 0}, "sizes": [2, 2, 1]}}, sleep=lambda _: None)


def test_smoke_requires_fresh_scoped_identity(tmp_path):
    with pytest.raises(ValueError, match="fresh"):
        smoke("a" * 40, "test-entity", "production-run", tmp_path / "new")
    with pytest.raises(ValueError, match="already exists"):
        smoke("a" * 40, "test-entity", "smoke-dagger-admission-fixture", tmp_path)


def test_smoke_reads_back_real_remote_fields(tmp_path, monkeypatch):
    summary = {}
    class Run:
        url = "https://wandb.ai/test-entity/dagger/runs/smoke-dagger-admission-fixture"
        def __init__(self):
            self.summary = summary
        def log(self, values, step):
            assert step == 0 and not any("mse" in name for name in values)
            summary.update(values)
        def finish(self, **kw):
            pass
    class Api:
        def __init__(self, **kw):
            self.viewer = SimpleNamespace(username="fixture")
        def flush(self):
            pass
        def run(self, path):
            assert path == "test-entity/dagger/smoke-dagger-admission-fixture"
            return SimpleNamespace(summary=summary)
    def init(**kw):
        assert kw["mode"] == "online" and kw["resume"] == "never"
        assert not kw["config"]["training_performed"]
        return Run()
    monkeypatch.setenv("WANDB_API_KEY", "restore-test-env")
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.wandb.ai")
    module = SimpleNamespace(Api=Api, init=init, Settings=lambda **kw: kw)
    result = smoke("a" * 40, "test-entity", "smoke-dagger-admission-fixture",
                   tmp_path / "new", wandb_module=module)
    assert result["project_write_verified"]
