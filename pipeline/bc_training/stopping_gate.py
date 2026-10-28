"""Explicit fixed-final BC protocol and frozen legacy fitting contracts."""
import math
from itertools import count


def fixed_final_contract(max_epochs, *, budget_explicit, epoch_target=None, eval_target=None):
    """New runs need a user-supplied budget, never an inferred patience/cap."""
    if not budget_explicit or type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("fixed_final BC requires an explicit positive --max-epochs budget")
    if epoch_target is not None or eval_target is not None:
        raise ValueError("fixed_final BC disables validation and MSE stopping gates")
    return {
        "schema": "roborender.bc-fixed-final.v1",
        "validation_enabled": False,
        "training_eval_enabled": False,
        "stop_rule": "explicit_absolute_epoch_budget",
        "max_epochs": max_epochs,
        "checkpoint_selection": "last_completed_epoch",
        "test_data_used_for_selection": False,
    }


def validate_fixed_final_resume(checkpoint, requested):
    if checkpoint.get("training_protocol") != requested:
        raise ValueError("strict BC resume requires the exact fixed_final protocol/budget; historical validation checkpoints are not implicitly converted")
    if checkpoint.get("validation_enabled") is not False:
        raise ValueError("fixed_final checkpoint must attest validation disabled")
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError("invalid fixed_final checkpoint epoch/global step")
    if epoch >= requested["max_epochs"]:
        raise ValueError("fixed_final resume checkpoint already reached the requested budget")
    if not isinstance(checkpoint.get("rng_state"), dict):
        raise ValueError("fixed_final resume requires exact checkpoint RNG state")


def training_target_met(target: float | None, measured_mse: float | None) -> bool:
    if target is None:
        return True
    if not math.isfinite(target) or target <= 0:
        raise ValueError("training MSE target must be finite and positive")
    if measured_mse is None or not math.isfinite(measured_mse) or measured_mse < 0:
        raise ValueError("training MSE must be measured, finite and nonnegative")
    return measured_mse < target


def should_stop_for_plateau(plateau: bool, target: float | None, measured_mse: float | None) -> bool:
    met = training_target_met(target, measured_mse)
    return bool(plateau and met)


def fitting_stop_contract(epoch_target=None, eval_target=None):
    if epoch_target is not None and eval_target is not None:
        raise ValueError("choose epoch-average training MSE OR frozen eval-mode MSE")
    if epoch_target is not None:
        training_target_met(epoch_target, 0.0)
        return {
            "schema": "roborender.bc-training-fit-stop.v2",
            "training_epoch_mse_below": epoch_target,
            "measurement": "full_epoch_training_mode_element_weighted_uniform_action_mse",
            "stop_rule": "validation_plateau_and_training_target",
            "checkpoint_selection": "minimum_complete_validation_mse",
        }
    if eval_target is not None:
        training_target_met(eval_target, 0.0)
    return {
        "schema": "roborender.bc-training-fit-stop.v1",
        "training_eval_mse_below": eval_target,
        "measurement": "entire_training_split_frozen_eval_mode_uniform_action_mse",
        "stop_rule": "validation_plateau_and_training_target_or_epoch_cap",
        "checkpoint_selection": "minimum_complete_validation_mse",
    }


def validate_fitting_stop_resume(previous, requested):
    # An explicit opt-in from an older disabled gate is allowed. An enabled
    # target/measurement cannot silently change on subsequent strict resume.
    if previous is None or previous == requested:
        return
    disabled = fitting_stop_contract()
    if previous != disabled:
        raise ValueError("resume training-MSE stopping contract differs from the checkpoint")


def epoch_sequence(start_epoch, max_epochs):
    """None is genuinely uncapped, not a large replacement ceiling."""
    if type(start_epoch) is not int or start_epoch < 0:
        raise ValueError("invalid starting epoch")
    if max_epochs is None:
        return count(start_epoch + 1)
    if type(max_epochs) is not int or max_epochs <= start_epoch:
        raise ValueError("maximum epoch must exceed starting epoch")
    return range(start_epoch + 1, max_epochs + 1)
