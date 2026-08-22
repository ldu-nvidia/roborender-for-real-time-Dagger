#!/usr/bin/env python3
"""Create and read back a tiny scalar/image/video W&B team validation run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from scripts.scaling.wandb_preflight import _load_wandb, query_identity
except ModuleNotFoundError:  # Direct execution from scripts/scaling.
    from wandb_preflight import _load_wandb, query_identity


REPORT_SCHEMA = "fast_roborender.wandb_online_smoke.v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-email", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument(
        "--authorization-scope", required=True, choices=("team",)
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser


def _media() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:64, :96]
    image = np.stack(
        ((x * 255 // 95), (y * 255 // 63), np.full_like(x, 96)), axis=-1
    ).astype(np.uint8)
    frames = []
    for offset in range(4):
        frame = np.roll(image, shift=offset * 8, axis=1)
        frames.append(np.transpose(frame, (2, 0, 1)))
    return image, np.stack(frames, axis=0)


def main() -> int:
    args = _parser().parse_args()
    wandb = _load_wandb()
    identity = query_identity(
        wandb,
        expected_email=args.expected_email,
        requested_entity=args.entity,
        authorization_scope=args.authorization_scope,
    )
    image, video = _media()
    run = wandb.init(
        entity=identity.entity,
        project=args.project,
        id=args.run_id,
        name=f"gb200-online-smoke-{args.run_id}",
        resume="allow",
        config={"validation": "gb200_wandb_scalar_image_video"},
    )
    run.log(
        {
            "trainer_step": 0,
            "validation/scalar": 1.0,
            "validation/image": wandb.Image(image, caption="GB200 W&B smoke"),
            "validation/video": wandb.Video(video, fps=2, format="mp4"),
        }
    )
    run.finish()

    path = f"{identity.entity}/{args.project}/{args.run_id}"
    deadline = time.monotonic() + args.timeout_s
    found: dict[str, object] = {}
    last_error = ""
    required = {"validation/scalar", "validation/image", "validation/video"}
    while time.monotonic() < deadline:
        try:
            remote = wandb.Api(timeout=30).run(path)
            for row in remote.scan_history(
                keys=["validation/scalar", "validation/image", "validation/video"],
                min_step=0,
                max_step=10,
                page_size=10,
            ):
                found.update({key: row[key] for key in required if key in row})
            if required.issubset(found):
                break
        except Exception as exc:  # Third-party text is not written to the report.
            last_error = type(exc).__name__
        time.sleep(5)
    status = "PASS" if required.issubset(found) else "FAIL"
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": args.project,
        "run_id": args.run_id,
        "run_path": path,
        "observed_keys": sorted(found),
        "last_api_error_type": last_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
