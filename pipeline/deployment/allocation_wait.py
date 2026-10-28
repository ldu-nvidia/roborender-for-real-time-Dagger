"""Bounded scheduler admission and boot gates; never starts GPU/native work."""
from __future__ import annotations

import time
from pathlib import Path

QUEUE_TIMEOUT_SECONDS = 3600
BOOT_TIMEOUT_SECONDS = 900
# Admission + boot + genuine renderer initialization + bounded control overhead.
PREPARE_TIMEOUT_SECONDS = 7200


def wait_for_host_key(inspect, key_file, report, *,
                      queue_timeout=QUEUE_TIMEOUT_SECONDS,
                      boot_timeout=BOOT_TIMEOUT_SECONDS, poll_seconds=15,
                      clock=time.monotonic, sleep=time.sleep):
    """Require an attested node assignment as well as the boot key.

    Queue time does not consume the boot budget. A lost/replaced assignment
    fails closed; it is not an invitation to replay or restart a simulator.
    Cloud/auth errors propagate rather than being mislabeled as queue time.
    """
    if (type(queue_timeout) is not int or not 1 <= queue_timeout <= 7200
            or type(boot_timeout) is not int or not 1 <= boot_timeout <= 3600
            or not 0 < poll_seconds <= 60):
        raise ValueError("allocation waits require bounded budgets")
    began, assigned_at, assignment = clock(), None, None
    while True:
        value = inspect()
        now = clock()
        if value.get("present") is not True:
            raise RuntimeError("owned allocation disappeared before readiness")
        replicas = value.get("replicas", [])
        if len(replicas) > 1:
            raise RuntimeError("unexpected allocation replica count")
        row = replicas[0] if replicas else {}
        assigned = bool(row.get("node_id"))
        if assigned:
            current = (row["replica_id"], row["node_id"])
            if assignment is not None and current != assignment:
                raise RuntimeError("allocation assignment changed before readiness")
            if assignment is None:
                if now - began >= queue_timeout:
                    raise TimeoutError("reserved allocation queue deadline exceeded")
                assignment, assigned_at = current, now
        elif assignment is not None:
            raise RuntimeError("allocation lost its assigned node before readiness")
        phase = "H200_BOOTING" if assigned else "H200_QUEUED"
        elapsed = now - (assigned_at if assigned else began)
        budget = boot_timeout if assigned else queue_timeout
        ready = assigned and Path(key_file).is_file()
        timed_out = elapsed >= budget
        evidence = {"phase": phase + "_TIMEOUT" if timed_out else "H200_HOST_KEY_READY" if ready else phase,
                    "pod": value["name"], "node_assigned": assigned,
                    "replica_id": row.get("replica_id"), "node_id": row.get("node_id"),
                    "scheduler_reason": row.get("reason", "AwaitingReplica"),
                    "elapsed_seconds": now - began, "phase_elapsed_seconds": elapsed,
                    "phase_budget_seconds": budget, "gpu_services_ready": False}
        report(evidence)
        if timed_out:
            raise TimeoutError("reserved allocation " + ("boot" if assigned else "queue") + " deadline exceeded")
        if ready:
            return evidence
        sleep(min(poll_seconds, budget - elapsed))
