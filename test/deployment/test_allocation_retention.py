"""CPU-only coverage for the user-directed retain-until-explicit-release policy."""
import hashlib
import json
from unittest.mock import Mock

import pytest

from pipeline.deployment import beta_lease_watchdog as watchdog
from pipeline.deployment import beta_site as module
from pipeline.deployment import resident_campaign_site as resident_module
from pipeline.deployment.resident_campaign_site import ResidentSite


def instance(tmp_path):
    site = module.Site.__new__(module.Site)
    site.run, site.rid, site.role = "dagger-leo-retention-test", 1, "leo"
    site.pod = "owned-exact-pod"
    site.root = tmp_path
    site.round = tmp_path / "round-1"
    site.round.mkdir()
    site.gpu_ssh = site.round / "h200-ssh.conf"
    site.gpus = [0, 1]
    site.allocation = Mock(return_value={"present": True})
    site.verify_actors_stopped = Mock()
    site.verify_containers_stopped = Mock()
    site.stop_process = Mock()
    site.command = Mock(return_value=b"")
    return site


@pytest.mark.parametrize("phase", ["release", "abort"])
def test_cleanup_retains_allocation_without_false_release_proof(tmp_path, monkeypatch, phase):
    site = instance(tmp_path)
    site.gpu_ssh.touch()
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = getattr(site, phase)()
    assert result["h200_allocation_retained"] is True
    assert result["h200_allocation_released"] is False
    assert all(call.args == ("status",) for call in site.allocation.call_args_list)
    assert (site.round / "retained.json").exists()
    assert not (site.round / "released.json").exists()
    site.verify_actors_stopped.assert_called_once()
    assert any(call.args == ("gpu-services", "gpu") for call in site.stop_process.call_args_list)


@pytest.mark.parametrize("response", [{}, {"present": None}, {"present": 0}, {"present": "false"}, None])
def test_unknown_presence_never_publishes_release_or_retention(tmp_path, response):
    site = instance(tmp_path)
    site.allocation.return_value = response
    with pytest.raises(RuntimeError, match="presence is unconfirmed"):
        site.finish_allocation_cleanup()
    assert list(site.round.iterdir()) == []


def test_absent_owner_is_observed_not_deleted(tmp_path):
    site = instance(tmp_path)
    site.allocation.return_value = {"present": False}
    result = site.finish_allocation_cleanup()
    assert result["h200_allocation_released"] is True
    site.allocation.assert_called_once_with("status")
    receipt = json.loads((site.round / "released.json").read_bytes())
    assert receipt["retained"] is False and receipt["release_confirmation"] is None


def test_explicit_release_requires_ack_and_observed_absence(tmp_path, monkeypatch):
    site = instance(tmp_path)
    site.allocation.side_effect = [{"present": True}, {"delete_requested": True},
                                   {"present": True}, {"present": False}]
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = site.finish_allocation_cleanup(confirm_release=site.pod)
    assert result["h200_allocation_released"] is True
    assert site.allocation.call_args_list[1].kwargs == {"confirm_release": site.pod}
    assert (site.round / "released.json").exists()
    assert not (site.round / "retained.json").exists()


@pytest.mark.parametrize("reply", [{"delete_requested": False, "present": True}, {}])
def test_rejected_explicit_delete_does_not_publish_success(tmp_path, reply):
    site = instance(tmp_path)
    site.allocation.side_effect = [{"present": True}, reply]
    with pytest.raises(RuntimeError, match="release was not accepted"):
        site.finish_allocation_cleanup(confirm_release=site.pod)
    assert list(site.round.iterdir()) == []


def test_unknown_post_delete_status_does_not_publish_release(tmp_path):
    site = instance(tmp_path)
    site.allocation.side_effect = [{"present": True}, {"delete_requested": True}, {}]
    with pytest.raises(RuntimeError, match="presence is unconfirmed"):
        site.finish_allocation_cleanup(confirm_release=site.pod)
    assert list(site.round.iterdir()) == []


def test_delete_timeout_does_not_publish_release(tmp_path, monkeypatch):
    site = instance(tmp_path)
    site.allocation.side_effect = lambda action, **kw: {"delete_requested": True} if action == "delete" else {"present": True}
    clock = iter([0., 91.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="release unconfirmed"):
        site.finish_allocation_cleanup(confirm_release=site.pod)
    assert list(site.round.iterdir()) == []


@pytest.mark.parametrize("phase", ["release", "abort"])
def test_wrong_confirmation_blocks_cleanup_before_mutations(tmp_path, phase):
    site = instance(tmp_path)
    with pytest.raises(ValueError, match="exact owned pod"):
        getattr(site, phase)(confirm_release="wrong-pod")
    site.stop_process.assert_not_called()
    site.allocation.assert_not_called()


def test_resident_abort_checks_confirmation_before_native_abort():
    site = ResidentSite.__new__(ResidentSite)
    site.pod = "owned-exact-pod"
    site.pool_action = Mock()
    with pytest.raises(ValueError, match="exact owned pod"):
        site.abort(confirm_release="wrong-pod")
    site.pool_action.assert_not_called()


def test_retained_cleanup_ends_watchdog_without_repeated_abort(tmp_path, monkeypatch):
    site = instance(tmp_path)
    (site.round / "retained.json").write_text(json.dumps({"retained": True}))
    site.abort = Mock()
    config, request = tmp_path / "site.json", tmp_path / "request.json"
    config.write_text("{}")
    request.write_text("{}")
    monkeypatch.setattr(watchdog, "Site", Mock(return_value=site))
    monkeypatch.setattr("sys.argv", ["watchdog", "--site", str(config), "--site-sha256",
                                    hashlib.sha256(config.read_bytes()).hexdigest(), "--request", str(request)])
    watchdog.main()
    site.abort.assert_not_called()


@pytest.mark.parametrize("entrypoint,resident", [(module, False), (resident_module, True)])
def test_cli_does_not_forge_release_marker_from_successful_retention(tmp_path, monkeypatch, entrypoint, resident):
    site = instance(tmp_path)
    site.release = Mock(return_value={"h200_allocation_released": False,
                                      "h200_allocation_retained": True})
    config, request, receipt = (tmp_path / name for name in ("site.json", "request.json", "receipt.json"))
    config.write_text("{}")
    request.write_text(json.dumps({"phase": "release", "operation_id": "unit", "run_id": site.run, "round_id": 1}))
    monkeypatch.setattr(entrypoint, "ResidentSite" if resident else "Site", Mock(return_value=site))
    argv = ["site", "--site", str(config), "--site-sha256", hashlib.sha256(config.read_bytes()).hexdigest(),
            "--request", str(request), "--receipt", str(receipt)]
    if resident:
        argv += ["--resident-pool", str(tmp_path / "pool.json"), "--resident-pool-sha256", "0" * 64]
    monkeypatch.setattr("sys.argv", argv)
    entrypoint.main()
    assert json.loads(receipt.read_bytes())["h200_allocation_released"] is False
    assert not (site.round / "released.json").exists()
    site.release.assert_called_once_with()
