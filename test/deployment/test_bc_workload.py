from test.deployment.source_paths import DEPLOYMENT_ROOT, REPO_ROOT
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.deployment.h200 import bc_workload as module


def site():
    return json.loads((DEPLOYMENT_ROOT / "profiles/leo_site.direct.example.json").read_bytes())


def instance(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module.BCWorkload(site(), "dagger-leo-bc-unit", gpus=1)


@pytest.mark.parametrize("owner,count", [("dagger-leo-other", 1), ("dagger-beta-bc-unit", 1),
                                        ("dagger-leo-bc-unit", 8), ("dagger-leo-bc-unit", 2)])
def test_rejects_ambiguous_owner_or_unapproved_shape(tmp_path, monkeypatch, owner, count):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        module.BCWorkload(site(), owner, gpus=count)


def test_smallest_bc_allocation_reuses_existing_secure_broker(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.json = Mock(return_value={"created": True})
    # Source file is a real public script, not a credential or copied token.
    monkeypatch.setattr(module, "ROOT", REPO_ROOT)
    workload.allocation("create", ["ssh-ed25519 public1", "ssh-ed25519 public2"])
    args, kwargs = workload.json.call_args
    assert args[0][args[0].index("--gpus") + 1] == "1"
    assert args[0][args[0].index("--run-id") + 1] == workload.run
    assert args[1] == "broker" and b"cosmos-solution-architect-xcwtffiq" in kwargs["data"]


def test_owned_workload_and_release_even_after_failure(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.acquire, workload.release, workload.process = Mock(), Mock(), Mock()
    workload.await_process = Mock(side_effect=RuntimeError("render failed"))
    with pytest.raises(RuntimeError, match="render failed"):
        workload.execute(["/bin/false"], ttl=300)
    workload.acquire.assert_called_once()
    workload.release.assert_called_once()
    assert workload.process.call_args.kwargs["host"] == "gpu"
    assert workload.process.call_args.kwargs["ttl"] == 300


@pytest.mark.parametrize("env,ttl", [({"WANDB_API_KEY": "not-a-real-key"}, 300), ({}, 1)])
def test_invalid_workload_rejected_before_allocation(tmp_path, monkeypatch, env, ttl):
    workload = instance(tmp_path, monkeypatch)
    workload.acquire = Mock()
    with pytest.raises(ValueError):
        workload.execute(["/bin/true"], env=env, ttl=ttl)
    workload.acquire.assert_not_called()


def test_unknown_gpu_client_blocks_release(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.gpu_ssh.touch()
    workload.allocation = Mock(return_value={"present": True})
    workload.stop_process = Mock()
    workload.command = Mock(return_value=b"1234\n")
    with pytest.raises(RuntimeError, match="GPU clients remain"):
        workload.release()
    assert workload.allocation.call_args_list[0].args == ("status",)
    assert workload.allocation.call_count == 1


def test_repeated_device_samples_before_exact_owner_release(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.gpu_ssh.touch()
    workload.allocation = Mock(side_effect=[{"present": True}, {"present": True},
                                           {"delete_requested": True}, {"present": False}])
    workload.stop_process = Mock()
    workload.command = Mock(return_value=b"")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = workload.release(confirm_release=workload.pod)
    assert workload.command.call_count == 3
    assert [c.args[0] for c in workload.allocation.call_args_list] == ["status", "status", "delete", "status"]
    assert workload.allocation.call_args_list[2].kwargs == {"confirm_release": workload.pod}
    assert result["h200_allocation_released"] is True
    assert (workload.round / "released.json").is_file()


def test_default_cleanup_stops_only_owned_workload_and_retains_pod(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.gpu_ssh.touch()
    workload.allocation = Mock(return_value={"present": True})
    workload.stop_process = Mock()
    workload.command = Mock(return_value=b"")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = workload.release()
    workload.stop_process.assert_called_once_with("bc-workload", "gpu")
    assert workload.command.call_count == 3
    assert all(c.args == ("status",) for c in workload.allocation.call_args_list)
    assert result["h200_allocation_retained"] is True
    assert result["h200_allocation_released"] is False
    assert not (workload.round / "released.json").exists()
    receipt = json.loads((workload.round / "retained.json").read_bytes())
    assert receipt["released"] is False and receipt["release_confirmation"] is None


def test_wrong_release_confirmation_rejected_before_scoped_stop(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.allocation, workload.stop_process = Mock(), Mock()
    with pytest.raises(ValueError, match="exact owned pod"):
        workload.release(confirm_release="another-pod")
    workload.allocation.assert_not_called()
    workload.stop_process.assert_not_called()


def test_release_confirmation_is_forwarded_to_streamed_broker(tmp_path, monkeypatch):
    workload = instance(tmp_path, monkeypatch)
    workload.json = Mock(return_value={"delete_requested": True})
    monkeypatch.setattr(module, "ROOT", REPO_ROOT)
    workload.allocation("delete", confirm_release=workload.pod)
    argv = workload.json.call_args.args[0]
    assert argv[argv.index("--confirm-release") + 1] == workload.pod
