"""Reporting-only names for the two explicitly supported selection protocols.

The controller, not these labels, performs selection. Defaults preserve legacy
success-selected jobs; held-out-MSE jobs must opt in explicitly. No tensors,
optimizer operations, data selection, or test outcomes are handled here.
"""

LEGACY = "legacy_success_selection"
MSE_FINAL_TEST = "heldout_mse_final_success"
FIXED100_FINAL_TEST = 'fixed100_final_success'


def metric_config(protocol=LEGACY):
    if protocol == FIXED100_FINAL_TEST:
        return {'metric_protocol': protocol, 'training_metric': 'action_mse',
                'validation_execution': 'disabled', 'checkpoint_selection_metric': 'final_update_100',
                'test_metric': 'raven_goal_reached_rate',
                'test_execution': 'independent_rendered_fixed295_after_update100',
                'test_outcomes_used_for_selection': False}
    if protocol == LEGACY:
        return {
            "metric_protocol": LEGACY,
            "validation_metric": "raven_goal_reached_rate",
            "validation_execution": "external_closed_loop_fixed_panel",
        }
    if protocol == MSE_FINAL_TEST:
        return {
            "metric_protocol": MSE_FINAL_TEST,
            "training_metric": "action_mse",
            "validation_metric": "action_mse",
            "validation_execution": "fixed_heldout_inference_only",
            "checkpoint_selection_metric": "heldout_action_mse",
            "test_metric": "raven_goal_reached_rate",
            "test_execution": "independent_native_closed_loop_after_selection",
            "test_outcomes_used_for_selection": False,
        }
    raise ValueError("unknown metric protocol")


def pending_candidate_metrics(protocol=LEGACY):
    labels = metric_config(protocol)
    if protocol == FIXED100_FINAL_TEST:
        return {**labels, 'validation_status': 'disabled', 'selection_rule': 'final_update_100',
                'validation_mse': None, 'validation_per_action_mse': None}
    if protocol == LEGACY:
        return {
            "metric_protocol": protocol,
            "validation_metric": labels["validation_metric"],
            "validation_status": "pending_closed_loop_evaluation",
            "selection_rule": "maximum_fixed_panel_raven_goal_reached_rate",
        }
    return {
        **labels,
        "validation_status": "pending_heldout_action_mse",
        "selection_rule": "minimum_fixed_heldout_action_mse",
        # A copied BC parent can contain a validation value. That value is not
        # the new DAgger candidate's score; its external receipt arrives later.
        "validation_mse": None,
        "validation_per_action_mse": None,
    }


def execution_budget(config_window_count, mse_protocol=None):
    if type(config_window_count) is not int or config_window_count < 1:
        raise ValueError("invalid outer collection ceiling")
    if mse_protocol is None:
        return config_window_count
    maximum = mse_protocol.get("max_iterations")
    if type(maximum) is not int or not 1 <= maximum <= config_window_count:
        raise ValueError("MSE budget must lie within the admitted collection ceiling")
    return maximum
