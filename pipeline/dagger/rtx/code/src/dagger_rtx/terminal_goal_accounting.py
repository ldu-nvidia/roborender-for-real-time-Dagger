"""Preserve Raven's episode goal across same-step automatic reset.

The predicate, termination conditions and simulator step remain native. Only
the outgoing transition's success flag is corrected. Put this mixin before
WalkStandPlaceGraspTurnHomie in the environment's MRO; keep the pinned Raven
checkout immutable. No CPU synchronization or physical-state mutation occurs.
"""
from __future__ import annotations


GOAL_ACCOUNTING_VERSION = "roborender.raven.pre_reset_goal.v1"


class TerminalGoalAccountingMixin:
    goal_accounting_version = GOAL_ACCOUNTING_VERSION

    def _check_termination(self):
        result = super()._check_termination()
        # Native StagedTaskBase has now evaluated and latched the goal. Both
        # buffers are cleared on reset, so retain an independent tensor before
        # that happens. OR preserves Raven's ever-reached episode semantics.
        self._transition_goal_before_reset = (
            self.grasp_goal_reached_buf | self.current_last_stage_completed_task_buf
        ).clone()
        return result

    def _post_physics_step(self):
        # Fail closed if a future native implementation skips the hook. Never
        # reuse the preceding transition's snapshot after a reset or exception.
        self._transition_goal_before_reset = None
        result = super()._post_physics_step()
        goal = self._transition_goal_before_reset
        if goal is None or tuple(goal.shape) != (self.num_envs,):
            raise RuntimeError("native pre-reset goal capture was not executed")
        # Keep the native post-reset episode buffers cleared: restoring them
        # would incorrectly credit the *next* episode with this one's success.
        self.extras["grasp_goal_reached"] = goal
        return result
