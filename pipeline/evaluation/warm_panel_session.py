"""One native evaluator, two frozen policies, and the same complete panel.

Checkpoint changes occur only between complete panels. Native reset/state
pairing remains the responsibility of the existing panel evaluator.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

try:
    from .beta_small_batch_panel import canonical, validate_comparison, validate_panel
except ImportError:  # The native container mounts this directory on PYTHONPATH.
    from beta_small_batch_panel import canonical, validate_comparison, validate_panel

SCHEMA = "roborender.warm-paired-panel.v1"
ARMS = ("bc", "dagger")
CHECKPOINT_PATHS = {
    "bc": "/diagnostic-assets/best_validation.pt",
    "dagger": "/diagnostic-assets/warm-candidate.pt",
}


def validate_request(request, panel):
    validate_panel(panel)
    if set(request) != {"schema", "panel_sha256", "cohort_limit", "checkpoint_sha256"}:
        raise ValueError("unexpected warm-evaluation request fields")
    if (request["schema"] != SCHEMA
            or request["panel_sha256"] != hashlib.sha256(canonical(panel)).hexdigest()
            or type(request["cohort_limit"]) is not int
            or request["cohort_limit"] != len(panel["cohorts"])):
        raise ValueError("warm evaluation requires the exact complete declared panel")
    hashes = request["checkpoint_sha256"]
    if (not isinstance(hashes, dict) or set(hashes) != set(ARMS)
            or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                   for h in hashes.values())):
        raise ValueError("two pinned checkpoint hashes are required")
    return request


def read_request(path, expected_sha256, panel):
    raw = Path(path).read_bytes()
    if len(raw) > 16384 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("warm request identity/size differs")
    request = json.loads(raw)
    if raw != canonical(request):
        raise ValueError("warm request must use canonical JSON")
    return validate_request(request, panel)


def run_pair(request, panel, root, evaluate, *, clock=time.monotonic, report=lambda row: None):
    """Evaluate sequentially through one caller-owned native environment.

    `evaluate` must load the specified policy, execute the existing complete
    panel reset protocol and compare candidate initial states with `reference`.
    A callback failure propagates; no candidate or simulator action is retried.
    """
    validate_request(request, panel)
    root = Path(root)
    results, timings = {}, []
    for arm in ARMS:
        output = root / arm
        if output.exists():
            raise RuntimeError("warm panel output already exists; never replay an uncertain evaluation")
        reference = root / "bc" if arm == "dagger" else None
        started = clock()
        report({"phase": "warm_checkpoint_start", "arm": arm,
                "checkpoint_sha256": request["checkpoint_sha256"][arm]})
        result = evaluate(arm=arm, checkpoint=CHECKPOINT_PATHS[arm],
                          checkpoint_sha256=request["checkpoint_sha256"][arm],
                          output=output, reference=reference)
        if (result.get("checkpoint_sha256") != request["checkpoint_sha256"][arm]
                or result.get("complete") is not True or result.get("optimizer_steps") != 0
                or result.get("policy_unchanged") is not True):
            raise RuntimeError("warm policy identity/completion/inference-only receipt differs")
        results[arm] = result
        timing = {"arm": arm, "seconds": clock() - started,
                  "checkpoint_sha256": result["checkpoint_sha256"],
                  "completed_episodes": result["completed_episodes"], "successes": result["successes"]}
        timings.append(timing)
        report({"phase": "warm_checkpoint_complete", **timing})
    audit = validate_comparison(panel, results, request["checkpoint_sha256"])
    return results, {"schema": "roborender.warm-evaluator-session.v1",
                     "complete": True, "checkpoint_load_order": list(ARMS),
                     "panel_sha256": request["panel_sha256"], "optimizer_steps": 0,
                     "timings": timings, "receipt_audit": audit}
