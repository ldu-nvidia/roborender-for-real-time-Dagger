"""Raven-faithful closed-loop success validation primitives.

Raven's evaluator latches ``infos["grasp_goal_reached"]`` across an episode and
counts the latched Boolean once when that episode terminates.  Reward and action
MSE are diagnostics; neither defines success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


RAVEN_SUCCESS_METRIC = "raven_goal_reached_rate"


@dataclass
class EpisodeGoalLatch:
    """Accumulate Raven's per-step goal signal for one live episode."""

    goal_reached: bool = False

    def update(self, grasp_goal_reached: bool) -> None:
        if not isinstance(grasp_goal_reached, bool):
            raise TypeError("grasp_goal_reached must be boolean")
        self.goal_reached = self.goal_reached or grasp_goal_reached

    def finish(self) -> bool:
        result = self.goal_reached
        self.goal_reached = False
        return result


def wilson_95(successes: int, total: int) -> list[float]:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    rate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def success_summary(outcomes: Iterable[bool]) -> dict[str, Any]:
    values = list(outcomes)
    if not values or any(not isinstance(value, bool) for value in values):
        raise ValueError("outcomes must be a non-empty Boolean sequence")
    successes = sum(values)
    total = len(values)
    return {
        "metric": RAVEN_SUCCESS_METRIC,
        "successes": successes,
        "failures": total - successes,
        "completed_episodes": total,
        "goal_reached_rate": successes / total,
        "wilson_95_interval": wilson_95(successes, total),
    }


def expand_validation_panel(panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the sealed 64-case x four-seed panel into 256 episodes."""

    if panel.get("schema") != "fast_roborender.round_dagger.validation_panel_registry.v1":
        raise ValueError("unexpected validation panel schema")
    seeds = panel.get("paired_seeds")
    case_ids = panel.get("case_ids")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("paired_seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired_seeds must be unique")
    if not isinstance(case_ids, list) or not case_ids or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError("case_ids must be a non-empty string list")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_ids must be unique")
    episodes = [
        {
            "scenario_id": f"seed{seed}.{case_id}",
            "seed": seed,
            "case_id": case_id,
            "motion_group_id": case_id.rsplit("__reset_", 1)[0],
            "reset_index": int(case_id.rsplit("__reset_", 1)[1]),
        }
        for seed in seeds
        for case_id in case_ids
    ]
    if len(episodes) != panel.get("case_count"):
        raise ValueError("case_count does not equal paired_seeds x case_ids")
    return episodes
