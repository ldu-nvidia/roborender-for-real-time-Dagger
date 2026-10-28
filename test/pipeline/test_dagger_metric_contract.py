from __future__ import annotations

import pytest

from pipeline.dagger.services.metric_contract import (
    LEGACY, MSE_FINAL_TEST, execution_budget, metric_config,
    pending_candidate_metrics,
)


def test_legacy_labels_remain_success_selection():
    assert metric_config() == {
        "metric_protocol": LEGACY,
        "validation_metric": "raven_goal_reached_rate",
        "validation_execution": "external_closed_loop_fixed_panel",
    }
    labels = pending_candidate_metrics()
    assert labels["selection_rule"] == "maximum_fixed_panel_raven_goal_reached_rate"
    assert labels["validation_status"] == "pending_closed_loop_evaluation"


def test_mse_labels_separate_selection_from_independent_test():
    labels = metric_config(MSE_FINAL_TEST)
    assert labels["training_metric"] == labels["validation_metric"] == "action_mse"
    assert labels["checkpoint_selection_metric"] == "heldout_action_mse"
    assert labels["test_metric"] == "raven_goal_reached_rate"
    assert labels["test_outcomes_used_for_selection"] is False
    pending = pending_candidate_metrics(MSE_FINAL_TEST)
    assert pending["selection_rule"] == "minimum_fixed_heldout_action_mse"
    assert pending["validation_status"] == "pending_heldout_action_mse"
    assert pending["validation_mse"] is None


@pytest.mark.parametrize("value", [None, "mse", "test_selected", True])
def test_unknown_metric_protocol_fails(value):
    with pytest.raises(ValueError, match="unknown metric protocol"):
        metric_config(value)


def test_actual_budget_is_not_outer_schema_ceiling():
    assert execution_budget(3, {"max_iterations": 1}) == 1
    assert execution_budget(50) == 50


@pytest.mark.parametrize("outer,maximum", [(0, 1), (3, 4), (3, True), (3, 0), (True, 1)])
def test_invalid_budget_rejected(outer, maximum):
    with pytest.raises(ValueError):
        execution_budget(outer, {"max_iterations": maximum})


def test_checkpoint_labels_do_not_change_policy_or_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    from pipeline.dagger.services.raven_learner_worker import save_iteration_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)
    base = {"validation_mse": 0.123, "validation_per_action_mse": [0.123]}
    request = {"parameters": {"metric_protocol": MSE_FINAL_TEST},
               "input_checkpoint_sha256": "a" * 64, "run_id": "metric-test",
               "round_id": 1, "job_id": "job", "train_chunk_ids": ["chunk"]}
    path = tmp_path / "candidate.pt"
    before = {name: value.clone() for name, value in model.state_dict().items()}
    save_iteration_checkpoint(path, base, model, optimizer, scheduler,
                              iteration_id=1, global_step=1, train_mse=0.2, request=request)
    output = torch.load(path, map_location="cpu", weights_only=False)
    assert base["validation_mse"] == 0.123
    assert output["validation_mse"] is None
    assert output["metric_protocol"] == MSE_FINAL_TEST
    assert output["test_outcomes_used_for_selection"] is False
    assert output["optimizer_state_dict"] == optimizer.state_dict()
    for name, value in before.items():
        assert torch.equal(output["policy_state_dict"][name], value)
        assert torch.equal(model.state_dict()[name], value)
