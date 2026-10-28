import json
import os

from dagger_rtx.container_health import check


def write(root, name, row, age=0):
    path = root / name
    path.write_text(json.dumps({"run_id": "health-test", **row}))
    os.utime(path, (1000 - age, 1000 - age))
    return path


def progress(root, **values):
    return write(root, "progress.json", {
        "schema": "roborender.dagger.rtx.phase-boundary.v1", "phase": "rtx_capture_teacher",
        "event": "end", "status": "complete", **values})


def test_initialization_is_not_reported_ready(tmp_path):
    assert not check(tmp_path, "health-test", now=1000)["ok"]


def test_native_phase_is_healthy_without_kit_log(tmp_path):
    progress(tmp_path)
    assert check(tmp_path, "health-test", now=1000)["ok"]
    assert not check(tmp_path, "wrong-run", now=1000)["ok"]
    assert not check(tmp_path, "health-test", now=1121)["ok"]


def test_failure_overrides_fresh_progress(tmp_path):
    progress(tmp_path)
    write(tmp_path, "qualification-state.json", {"phase": "CRASH_UNCERTAIN"})
    assert check(tmp_path, "health-test", now=1000)["reason"] == "native_driver_failed_closed"


def test_native_error_is_unhealthy(tmp_path):
    progress(tmp_path, event="error")
    assert not check(tmp_path, "health-test", now=1000)["ok"]


def test_checkpoint_wait_has_a_bounded_separate_deadline(tmp_path):
    progress(tmp_path)
    write(tmp_path, "qualification-state.json", {"phase": "WAITING_FOR_CHECKPOINT"})
    assert check(tmp_path, "health-test", now=1599)["ok"]
    assert not check(tmp_path, "health-test", now=1601)["ok"]


def test_production_identity_in_spec(tmp_path):
    path = tmp_path / "production-driver-state.json"
    path.write_text(json.dumps({"spec": {"run_id": "health-test"},
                                "phase": "WAITING_FOR_CHECKPOINT_ADVANCE"}))
    os.utime(path, (1000, 1000))
    assert check(tmp_path, "health-test", now=1000)["ok"]


def test_rejects_symlinks_oversize_and_future_snapshots(tmp_path):
    path = progress(tmp_path)
    os.utime(path, (1010, 1010))
    assert not check(tmp_path, "health-test", now=1000)["ok"]
    path.write_text(" " * 65537)
    assert not check(tmp_path, "health-test", now=1000)["ok"]
    path.unlink()
    path.symlink_to(tmp_path / "absent")
    assert not check(tmp_path, "health-test", now=1000)["ok"]
