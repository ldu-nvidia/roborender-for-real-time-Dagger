#!/usr/bin/env python3
"""Reject an evaluation baseline that is not Raven's three-view offline BC config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    algo = config["algo"]["config"]
    cameras = config["simulator"]["config"]["cameras"]
    videogen = algo["videogen"]
    require(
        config["trainer"]["_target_"]
        == "gr00t.rl.trl.trainer.distill_trainer_videogen.TRLDistillTrainerVideoGen",
        "baseline trainer differs from Raven",
    )
    require(config["policy_view_count"] == 3, "baseline is not three-view")
    require(
        cameras["policy_views"] == ["head_color", "hand_left", "hand_right"],
        "baseline policy view order differs",
    )
    require(cameras["camera_resolutions"] == [108, 192],
            "baseline policy resolution differs")
    require(algo["num_steps_per_env"] == 81, "baseline rollout length differs")
    require(algo["num_mini_batches"] == 1, "baseline minibatch count differs")
    require(algo["value_loss_coef"] == 0.0, "baseline value loss differs")
    require(algo["entropy_coef"] == 0.0, "baseline entropy loss differs")
    require(videogen["mode"] == "offline", "baseline is not offline BC")
    require(videogen["rollout_action_source"] == "teacher",
            "baseline action source differs")
    print("raven-three-view-offline-bc-baseline-config-valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
