"""Small owned cold-curriculum snapshot, never a simulator/crash checkpoint."""
from __future__ import annotations

import ast
from collections import deque
import inspect
import textwrap


def _copy_small(value):
    # Duck typing keeps import/CPU tests independent of CUDA/Isaac.
    if hasattr(value, "detach") and hasattr(value, "numel"):
        if value.numel() > 65536:
            raise RuntimeError("phase reset attempted to snapshot a large observation buffer")
        return value.detach().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, deque) and value.maxlen is not None and value.maxlen <= 1024:
        return deque((_copy_small(v) for v in value), maxlen=value.maxlen)
    if isinstance(value, (list, tuple)) and len(value) <= 256:
        return type(value)(_copy_small(v) for v in value)
    if isinstance(value, dict) and len(value) <= 256:
        return {k: _copy_small(v) for k, v in value.items()}
    raise RuntimeError("unknown mutable curriculum/reset state type")


class _ColdEventState:
    """Restore reset eligibility when the independent phase clock is restored.

    EventManager.reset() does not clear these counters and samples interval
    timers. Neither is suitable here: this boundary must consume no RNG and
    must not alter ordinary within-episode reset behavior.
    """
    _FIELDS = ("_reset_term_last_triggered_step_id", "_reset_term_last_triggered_once")

    def __init__(self, manager):
        self.manager = manager
        names = manager._mode_term_names
        # Startup effects and interval/class state need their own lifecycle
        # protocol. Do not claim cold equivalence by copying only two counters.
        if any(names.get(mode) for mode in ("startup", "prestartup", "interval")):
            raise RuntimeError("cold phase has unsupported event lifecycle modes")
        if any(manager._mode_class_term_cfgs.values()):
            raise RuntimeError("cold phase has unsupported class event state")
        self.names = tuple(names.get("reset", ()))
        self.values = {}
        for field in self._FIELDS:
            values = getattr(manager, field)
            if not isinstance(values, list) or len(values) != len(self.names):
                raise RuntimeError("cold phase event counter inventory changed")
            if any(not hasattr(value, "copy_") or value.ndim != 1 for value in values):
                raise RuntimeError("cold phase event counters must be lane tensors")
            self.values[field] = [_copy_small(value) for value in values]

    def restore(self, manager):
        if manager is not self.manager or tuple(manager._mode_term_names.get("reset", ())) != self.names:
            raise RuntimeError("cold phase event manager identity/inventory changed")
        # Validate all destinations before mutating any counter.
        for field, saved in self.values.items():
            current = getattr(manager, field)
            if not isinstance(current, list) or len(current) != len(saved):
                raise RuntimeError("cold phase event counter inventory changed")
            for target, source in zip(current, saved):
                if (not hasattr(target, "copy_") or target.shape != source.shape
                        or target.dtype != source.dtype or target.device != source.device):
                    raise RuntimeError("cold phase event counter shape/dtype/device changed")
        for field, saved in self.values.items():
            for target, source in zip(getattr(manager, field), saved):
                target.copy_(source)


class ColdPhaseState:
    def __init__(self, env):
        names = {name for name in vars(env) if any(word in name for word in
                 ("curriculum", "penalty_scale", "termination_level", "average_episode"))}
        # These small native task metrics feed curriculum/logging at reset.
        # They must not inherit outcomes from a preceding independent phase.
        names.update(name for name in (
            # Raven constructs a fresh task for a new train/test phase. Its
            # vector clock controls history cadence and reset events; warm
            # reuse must not inherit the preceding phase's elapsed steps.
            "common_step_counter",
            "grasp_goal_reached_deque", "walk_stand_goal_reached_deque",
            "average_grasp_goal_reached_rate", "average_walk_stand_goal_reached_rate",
            "skip_initial_goal_tracking", "num_total_task", "num_total_success",
            "last_episode_length_buf", "last_max_stage_buf", "current_max_stage_buf",
            "last_completed_task_buf", "current_completed_task_buf",
            "last_last_stage_completed_task_buf", "current_last_stage_completed_task_buf",
        ) if name in vars(env))
        # Include native method references rather than maintaining a hand-picked
        # camera/Homie curriculum list. Only direct env-owned data is captured.
        for name, method in inspect.getmembers(env, predicate=inspect.ismethod):
            if "curriculum" not in name or name.startswith("__"):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            targets = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets.append(node.target)
            for target in targets:
                while isinstance(target, ast.Subscript):
                    target = target.value
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and target.attr in vars(env)):
                    names.add(target.attr)
        names.discard("config")
        names.discard("simulator")
        names.discard("log_dict")
        self.values = {name: _copy_small(getattr(env, name)) for name in sorted(names)}
        # This is Raven's scalar curriculum/metric metadata, not physics state,
        # camera arrays, a serialized env_state_dict, or an external checkpoint.
        self.logs = _copy_small(getattr(env, "log_dict", {}))
        simulator = getattr(env, "simulator", None)
        self.events = _ColdEventState(simulator.event_manager) if simulator is not None else None

    def restore(self, env):
        if self.events is not None:
            self.events.restore(env.simulator.event_manager)
        for name, saved in self.values.items():
            current = getattr(env, name)
            if hasattr(saved, "detach"):
                if not hasattr(current, "copy_") or current.shape != saved.shape or current.dtype != saved.dtype:
                    raise RuntimeError("cold phase curriculum shape/dtype changed")
                current.copy_(saved)
            else:
                setattr(env, name, _copy_small(saved))
        env.log_dict.clear()
        env.log_dict.update(_copy_small(self.logs))
        return sorted(self.values)
