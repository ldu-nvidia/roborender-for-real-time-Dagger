"""Reporting regression tests; no GPU, network, or selection mutation."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pipeline.dagger.services.mse_status import (
    candidate_summary_status, completed_mse_summary, next_round_display_fields,
)


def test_explicit_pending_labels_preserve_legacy():
    assert candidate_summary_status("heldout_mse_final_success") == {
        "validation_status": "pending_heldout_action_mse", "test_status": "not_started"}
    assert candidate_summary_status() == {
        "validation_status": "pending_closed_loop_success_evaluation"}


@pytest.mark.parametrize("value", ["", None, "unknown"])
def test_unknown_protocol_rejected(value):
    with pytest.raises(ValueError):
        candidate_summary_status(value)


def test_completed_status_is_independent_of_success_value():
    values = [completed_mse_summary({"purpose": "test", "status": "complete", "completed": 2,
        "checkpoint_sha256": "a" * 64, "goal_reached_rate": rate}) for rate in (0., 1.)]
    assert values[0] == values[1]
    assert values[0]["validation_status"] == "completed_heldout_action_mse"


@pytest.mark.parametrize("change", [{"purpose": "validation"}, {"status": "running"},
                                    {"completed": 0}, {"completed": True}])
def test_pending_or_non_test_receipt_cannot_claim_complete(change):
    with pytest.raises(ValueError):
        completed_mse_summary({"purpose": "test", "status": "complete", "completed": 2,
                              "checkpoint_sha256": "a" * 64, **change})


def test_next_round_retains_ownership_but_not_prior_round_display():
    values = next_round_display_fields()
    assert "service_binding" not in values
    assert values["heldout_population_reused"] is False
    assert all(value is None for key, value in values.items() if key != "heldout_population_reused")


def test_actual_final_logger_overwrites_and_verifies_stale_pending_status():
    from pipeline.dagger.mse_campaign import online_mse_loggers
    run = SimpleNamespace(summary={"validation_status": "pending_closed_loop_success_evaluation"},
                          url="https://example.invalid/cpu-test", log=Mock(), finish=Mock())
    wandb = SimpleNamespace(init=Mock(return_value=run))
    report = {"purpose": "test", "status": "complete", "completed": 2,
              "checkpoint_sha256": "a" * 64, "goal_reached_rate": 0., "successes": 0}
    with patch("pipeline.dagger.beta_campaign.online_loggers", return_value=(Mock(), Mock())), \
            patch("pipeline.deployment.wandb_preflight.admit_logging", return_value=(Mock(), {"entity": "cpu-fixture"})), \
            patch("pipeline.deployment.wandb_preflight.verify_remote_summary") as verify, \
            patch.dict(sys.modules, {"wandb": wandb}):
        _, log = online_mse_loggers("cpu-fixture")
        result = log({"logging": {"wandb_run_id": "cpu-test"},
                      "validation": {"observation_mode": "generated"}},
                     {"iteration_id": 1, "train_mse": .7}, report)
    assert result["online_metrics_verified"] is True
    assert run.summary["validation_status"] == "completed_heldout_action_mse"
    assert run.summary["test_status"] == "complete"
    assert verify.call_args.args[-1]["validation_status"] == "completed_heldout_action_mse"
    assert run.log.call_args.args[0]["test/observation_mode"] == "generated"
    assert run.summary["test/observation_mode"] == "generated"
    assert verify.call_args.args[-1]["test/observation_mode"] == "generated"
    run.finish.assert_called_once()


def test_learner_uses_protocol_specific_pending_status():
    import inspect
    from pipeline.dagger.services import raven_learner_worker
    source = inspect.getsource(raven_learner_worker)
    assert "**candidate_summary_status(metric_protocol)" in source
    assert '"validation_status": "pending_closed_loop_success_evaluation"' not in source
