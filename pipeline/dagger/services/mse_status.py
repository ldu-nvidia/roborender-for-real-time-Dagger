"""Display-only MSE selection/test lifecycle labels; never select a model."""

from .metric_contract import LEGACY, MSE_FINAL_TEST, FIXED100_FINAL_TEST


def candidate_summary_status(protocol=LEGACY):
    if protocol == FIXED100_FINAL_TEST:
        return {'validation_status': 'disabled', 'test_status': 'not_started'}
    if protocol == MSE_FINAL_TEST:
        return {"validation_status": "pending_heldout_action_mse", "test_status": "not_started"}
    if protocol == LEGACY:
        return {"validation_status": "pending_closed_loop_success_evaluation"}
    raise ValueError("unknown metric protocol")


def completed_mse_summary(test_report, *, fixed100=False):
    """Require the completed test receipt, without using its success value."""
    if (test_report.get("purpose") != "test" or test_report.get("status") != "complete"
            or type(test_report.get("completed")) is not int or test_report["completed"] < 1):
        raise ValueError("actual completed independent test required")
    return {"validation_status": 'disabled' if fixed100 else "completed_heldout_action_mse", "test_status": "complete",
            "selected_checkpoint_sha256": test_report["checkpoint_sha256"],
            "checkpoint_selection_metric": 'final_update_100' if fixed100 else "heldout_action_mse"}


def next_round_display_fields():
    # PREPARE still needs service_binding to retire the preceding CPU owner.
    # Clear only prior-round results/display metadata, after copying history.
    return {"config_path": None, "final_checkpoint": None, "final_learner": None,
            "evaluation_receipt": None, "evaluation_report": None, "logging_receipt": None,
            "mse_completion": None, "round_logging": None, "progress": None,
            "release_receipt": None, "heldout_collection_receipt": None,
            "heldout_population_reused": False, "collection_deadline": None,
            "mse_worker_command": None}
