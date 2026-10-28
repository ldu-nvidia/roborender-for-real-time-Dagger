#!/usr/bin/env python3
"""Explicitly launched inference worker; no allocation, simulator or optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dagger.mse_controller import atomic_json, digest


def completed_validation_summary(request, receipt):
    """Bind display completion to this inference, not checkpoint selection/test."""
    result = receipt["result"]
    iteration = request["iteration_id"]
    if (set(receipt) != {"request_sha256", "result"}
            or receipt["request_sha256"] != digest(request)
            or type(iteration) is not int or iteration < 0
            or result.get("schema") != "roborender.dagger-heldout-mse-result.v1"
            or result.get("purpose") != "validation"
            or result.get("selection_metric") != "action_mse"
            or result.get("inference_only") is not True
            or result.get("test_payloads_opened") is not False
            or result.get("checkpoint_sha256") != request["checkpoint_sha256"]
            or result.get("validation_manifest_sha256") != request["manifest_sha256"]
            or result.get("round_id") != request["training_provenance"]["round_id"]):
        raise ValueError("completed validation display requires the exact inference receipt")
    return {
        "validation_status": "completed_heldout_action_mse",
        "validation_iteration_id": iteration,
        "validation_checkpoint_role": "round_initial_baseline" if iteration == 0 else "learner_candidate",
        "validation_checkpoint_sha256": request["checkpoint_sha256"],
        "validation_manifest_sha256": request["manifest_sha256"],
        # Canonical semantic digest, not the whitespace-dependent JSON file hash.
        "validation_receipt_canonical_sha256": digest(receipt),
        "validation_request_sha256": receipt["request_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    if (set(request) != {"schema", "checkpoint_path", "checkpoint_sha256", "manifest_path",
                         "manifest_sha256", "training_provenance", "iteration_id", "logging"}
            or request["schema"] != "roborender.dagger-heldout-worker.v1"):
        raise ValueError("invalid held-out worker request")
    import torch
    from pipeline.dagger.services.heldout_mse import evaluate_checkpoint
    result = evaluate_checkpoint(
        Path(request["checkpoint_path"]), request["checkpoint_sha256"],
        Path(request["manifest_path"]), request["manifest_sha256"],
        request["training_provenance"], device=torch.device(args.device),
    )
    receipt = {"request_sha256": digest(request), "result": result}
    summary = completed_validation_summary(request, receipt)
    logging = request["logging"]
    if logging.get("wandb_mode") not in {"online", "offline", "disabled"}:
        raise ValueError("held-out worker W&B mode must be explicit")
    if logging["wandb_mode"] != "disabled":
        import wandb
        run = wandb.init(project=logging["wandb_project"], name=logging["wandb_run_name"],
                         entity=logging.get("wandb_entity"),
                         id=logging.get("wandb_run_id"), resume="allow", mode=logging["wandb_mode"])
        try:
            run.define_metric("validation/iteration")
            run.define_metric("validation/*", step_metric="validation/iteration")
            run.log({"validation/iteration": request["iteration_id"],
                     "validation/mse": result["mse"], "validation/frames": result["frames"],
                     **{f"validation/mse_action_{index:02d}": value
                        for index, value in enumerate(result["per_action_mse"])}})
            run.summary.update(summary)
        finally:
            run.finish()
    atomic_json(args.result, receipt)


if __name__ == "__main__":
    main()
