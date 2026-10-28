"""Actual worker reporting paths on CPU with fake inference/W&B, no service."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.dagger import heldout_mse_worker as worker
from pipeline.dagger.services import heldout_mse


def fixture_request(iteration=0, mode="online"):
    return {
        "schema": "roborender.dagger-heldout-worker.v1",
        "checkpoint_path": "/cpu-fixture/checkpoint.pt", "checkpoint_sha256": "a" * 64,
        "manifest_path": "/cpu-fixture/validation.json", "manifest_sha256": "b" * 64,
        "training_provenance": {"round_id": 2}, "iteration_id": iteration,
        "logging": {"wandb_mode": mode, "wandb_project": "cpu-fixture",
                    "wandb_run_name": "cpu-fixture", "wandb_run_id": "cpu-fixture"},
    }


def fixture_result():
    return {"schema": heldout_mse.RESULT_SCHEMA, "purpose": "validation", "round_id": 2,
        "validation_manifest_sha256": "b" * 64, "checkpoint_sha256": "a" * 64,
        "mse": .5, "per_action_mse": [.25, .75], "frames": 4, "elements": 8,
        "chunk_count": 2, "inference_only": True, "selection_metric": "action_mse",
        "test_payloads_opened": False}


@pytest.mark.parametrize("iteration,role", [(0, "round_initial_baseline"), (3, "learner_candidate")])
def test_completion_binds_exact_receipt_and_distinguishes_baseline(iteration, role):
    request, result = fixture_request(iteration), fixture_result()
    receipt = {"request_sha256": worker.digest(request), "result": result}
    before = deepcopy((request, receipt))
    summary = worker.completed_validation_summary(request, receipt)
    assert summary == {
        "validation_status": "completed_heldout_action_mse", "validation_iteration_id": iteration,
        "validation_checkpoint_role": role, "validation_checkpoint_sha256": "a" * 64,
        "validation_manifest_sha256": "b" * 64,
        "validation_receipt_canonical_sha256": worker.digest(receipt),
        "validation_request_sha256": worker.digest(request),
    }
    assert (request, receipt) == before
    assert not any(key.startswith("test") or key == "selected_checkpoint_sha256" for key in summary)


@pytest.mark.parametrize("change", [
    {"purpose": "test"}, {"checkpoint_sha256": "c" * 64},
    {"validation_manifest_sha256": "c" * 64}, {"round_id": 3},
    {"inference_only": False}, {"test_payloads_opened": True}, {"selection_metric": "success_rate"},
])
def test_foreign_or_non_validation_result_cannot_claim_complete(change):
    request = fixture_request()
    receipt = {"request_sha256": worker.digest(request), "result": {**fixture_result(), **change}}
    with pytest.raises(ValueError, match="exact inference receipt"):
        worker.completed_validation_summary(request, receipt)


@pytest.mark.parametrize("iteration", [-1, True, "0"])
def test_invalid_iteration_cannot_be_labeled_as_baseline(iteration):
    request = fixture_request(iteration)
    with pytest.raises(ValueError, match="exact inference receipt"):
        worker.completed_validation_summary(request, {"request_sha256": worker.digest(request), "result": fixture_result()})


def test_changed_request_is_not_bound_by_old_receipt():
    request = fixture_request()
    receipt = {"request_sha256": worker.digest(request), "result": fixture_result()}
    request["iteration_id"] = 1
    with pytest.raises(ValueError, match="exact inference receipt"):
        worker.completed_validation_summary(request, receipt)


def prepare_main(monkeypatch, tmp_path, *, iteration=0, mode="online"):
    request, result = fixture_request(iteration, mode), fixture_result()
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["worker", "--request", str(request_path), "--result", str(result_path), "--device", "cpu"])
    events = []
    infer = Mock(side_effect=lambda *args, **kwargs: (events.append("inference"), deepcopy(result))[1])
    monkeypatch.setattr(heldout_mse, "evaluate_checkpoint", infer)

    class Summary(dict):
        def update(self, value):
            events.append("summary")
            super().update(value)

    summary = Summary(validation_status="pending_heldout_action_mse", test_status="not_started",
                      selected_checkpoint_sha256="d" * 64)
    run = SimpleNamespace(summary=summary, define_metric=Mock(),
        log=Mock(side_effect=lambda metrics: events.append("numeric_log")),
        finish=Mock(side_effect=lambda: events.append("finish")))
    api = SimpleNamespace(init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, "wandb", api)
    return request, result, result_path, events, infer, run, api


@pytest.mark.parametrize("iteration", [0, 2])
def test_actual_main_marks_mse_complete_after_numeric_log_only(monkeypatch, tmp_path, iteration):
    request, result, path, events, infer, run, api = prepare_main(monkeypatch, tmp_path, iteration=iteration)
    worker.main()
    receipt = json.loads(path.read_bytes())
    assert receipt == {"request_sha256": worker.digest(request), "result": result}
    assert events == ["inference", "numeric_log", "summary", "finish"]
    assert run.log.call_args.args[0] == {
        "validation/iteration": iteration, "validation/mse": .5, "validation/frames": 4,
        "validation/mse_action_00": .25, "validation/mse_action_01": .75,
    }
    assert run.summary["validation_receipt_canonical_sha256"] == worker.digest(receipt)
    assert run.summary["validation_status"] == "completed_heldout_action_mse"
    assert run.summary["validation_checkpoint_role"] == ("round_initial_baseline" if iteration == 0 else "learner_candidate")
    assert run.summary["test_status"] == "not_started"
    assert run.summary["selected_checkpoint_sha256"] == "d" * 64
    assert str(infer.call_args.kwargs["device"]) == "cpu"
    api.init.assert_called_once()


def test_inference_failure_does_not_publish_completion(monkeypatch, tmp_path):
    _, _, path, events, infer, run, api = prepare_main(monkeypatch, tmp_path)
    infer.side_effect = RuntimeError("fixture inference failure")
    with pytest.raises(RuntimeError, match="inference failure"):
        worker.main()
    api.init.assert_not_called()
    assert not path.exists() and events == []
    assert run.summary["validation_status"] == "pending_heldout_action_mse"


def test_numeric_log_failure_does_not_publish_completion(monkeypatch, tmp_path):
    _, _, path, events, _, run, _ = prepare_main(monkeypatch, tmp_path)
    run.log.side_effect = RuntimeError("fixture logger failure")
    with pytest.raises(RuntimeError, match="logger failure"):
        worker.main()
    assert not path.exists() and events == ["inference", "finish"]
    assert run.summary["validation_status"] == "pending_heldout_action_mse"
    run.finish.assert_called_once()


def test_logging_disabled_preserves_exact_receipt_without_wandb(monkeypatch, tmp_path):
    request, result, path, events, _, _, api = prepare_main(monkeypatch, tmp_path, mode="disabled")
    worker.main()
    assert json.loads(path.read_bytes()) == {"request_sha256": worker.digest(request), "result": result}
    api.init.assert_not_called()
    assert events == ["inference"]
