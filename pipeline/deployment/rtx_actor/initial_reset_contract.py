"""Reset-all bookkeeping only; no physics, target, policy or RNG operations.

Called by the hash-pinned Raven overlay, never by ordinary reset_envs_idx.
Evidence establishes call order, not numeric PhysX state equivalence.
"""
from copy import deepcopy

RESET_CONTRACT = "roborender.initial-reset-completion.v1"


def begin_initial_reset(env):
    if getattr(env, "_rr_initial_reset_active", False):
        raise RuntimeError("initial reset incomplete; do not replay uncertain initialization")
    env._rr_initial_reset_receipt = None
    env._rr_initial_reset_generation = getattr(env, "_rr_initial_reset_generation", 0) + 1
    env._rr_initial_reset_active = True


def _pending(env):
    import torch
    value = env.need_to_refresh_envs
    if (type(env.num_envs) is not int or env.num_envs < 1
            or not isinstance(value, torch.Tensor) or value.dtype != torch.bool
            or tuple(value.shape) != (env.num_envs,)):
        raise RuntimeError("invalid initial reset pending buffer")
    return value


def _receipt(env):
    return {"schema": RESET_CONTRACT, "generation": env._rr_initial_reset_generation,
            "num_envs": env.num_envs,
            "boundary": "after_reset_all_writes_and_observations",
            "cleared_lanes": env.num_envs, "clear_calls": 1,
            "native_steps_added": 0, "ordinary_done_resets_modified": False}


def complete_initial_reset(env):
    """Consume only flags whose writes reset_all has already completed."""
    if not getattr(env, "_rr_initial_reset_active", False):
        raise RuntimeError("initial reset completion outside reset_all or applied twice")
    pending = _pending(env)
    if not pending.all().item():
        raise RuntimeError("initial reset flags are not all pending after reset_all writes")
    pending.zero_()
    env._rr_initial_reset_receipt = _receipt(env)
    env._rr_initial_reset_active = False
    return deepcopy(env._rr_initial_reset_receipt)


def require_completed_initial_reset(env):
    """Admission at initialization, not every step (ordinary resets may pend)."""
    receipt = getattr(env, "_rr_initial_reset_receipt", None)
    generation = getattr(env, "_rr_initial_reset_generation", None)
    if (getattr(env, "_rr_initial_reset_active", False)
            or type(generation) is not int or generation < 1
            or not isinstance(receipt, dict) or receipt != _receipt(env)
            or _pending(env).any().item()):
        raise RuntimeError("missing or inconsistent corrected initial reset receipt; rematerialize runtime")
    return deepcopy(receipt)
