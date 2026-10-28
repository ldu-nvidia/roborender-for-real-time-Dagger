import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.bc_training.offline_pipeline import execute, plan


def spec(tmp_path, **kwargs):
    return plan(dataset_root=tmp_path / "data", asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime", run_name="smoke-bc-qualified", **kwargs)


def pool(job, *, train=16, total=20):
    root = Path(job["dataset_root"])
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "roborender.bc_pool.native-cohorts.v1",
        "status": "sealed", "chunk_count": total, "partition_counts": {"train": train}}))
    (root / "SEALED.json").write_text('{"verdict":"PASS"}')


def test_plan_keeps_original_render_training_and_no_test_execution(tmp_path):
    job = spec(tmp_path, qualification=True, min_epochs=1, max_epochs=3)
    assert len(job["stages"]) == 4
    assert job["env"]["CHUNK_BATCH_SIZE"] == "16"
    assert job["env"]["REQUIRE_PRODUCTION_DATA_ADMISSION"] == "1"
    assert job["env"]["ALLOW_QUALIFICATION_DATA"] == "1"
    assert job["test_policy_executed"] is False
    assert "--require-independent-render" in job["stages"][2]["argv"]
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("kwargs", [dict(gpus=(0, 1)), dict(gpus=(1,)), dict(min_epochs=0),
    dict(min_epochs=1), dict(patience=1), dict(min_epochs=11, max_epochs=10)])
def test_plan_rejects_unqualified_resources_and_stopping(tmp_path, kwargs):
    with pytest.raises(ValueError):
        spec(tmp_path, **kwargs)


def test_small_pool_requires_explicit_smoke_and_failure_retained(tmp_path):
    with pytest.raises(ValueError, match="pinned full-population"):
        spec(tmp_path)
    job = spec(tmp_path, qualification=True)
    pool(job)
    calls = []
    def runner(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1)
    with pytest.raises(Exception):
        execute(job, runner=runner)
    progress = json.loads((tmp_path / "runtime/progress.json").read_text())
    assert progress["status"] == "failed"
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        execute(job, runner=runner)
    assert len(calls) == 1


def test_stages_order_and_progress(tmp_path):
    job = spec(tmp_path, qualification=True)
    pool(job)
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["WANDB_MODE"] == "online"
        return SimpleNamespace(returncode=0)
    result = execute(job, runner=runner)
    assert calls == [stage["argv"] for stage in job["stages"]]
    assert result["status"] == "complete" and result["test_policy_executed"] is False
    assert len(list((tmp_path / "runtime").glob("stage-*.json"))) == 4


def test_production_automatically_binds_genuine_worker_receipt_after_activation(tmp_path):
    job = spec(tmp_path, production_protocol=tmp_path / "protocol.json", production_protocol_sha256="a" * 64)
    assert [s["name"] for s in job["stages"]] == ["render_50step_teacher", "activate_disjoint_splits",
        "bind_render_execution", "metadata_data_admission", "bc_train_and_validation_mse"]
    assert "--execution-identity" in job["stages"][0]["argv"]
    assert "RESOLVE_ACTUAL_RENDER_EXECUTION_SHA256" in job["stages"][3]["argv"]
    assert job["env"]["ALLOW_QUALIFICATION_DATA"] == "0"
