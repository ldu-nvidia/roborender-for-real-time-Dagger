"""Historical B5 paired beta evaluation, retained for compatibility.

This entrypoint does not implement accepted295 production success. Production
BC/DAgger success uses the corrected resident evaluate_explicit_panel operation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import time

from run_paired_student_success import build_command, sha
from run_success_diagnostic import gpu_gate
from qualify_graphics import cleanup_owned, HOSTNAME
from qualify_native_isaac6 import IMAGE
from run_throughput_pair import memory_kib, owned_memory_sample, memory_safety, native_completion


def execute_container(name, output, cmd, cancelled, wall_budget, *, completion_receipt, completion_field):
    """Monitor one owned native lifetime, including its checkpoint switches."""
    initial = memory_kib()
    started = time.monotonic()
    process = None
    try:
        with (output / "console.log").open("wb") as log, (output / "resources.jsonl").open("w") as resources:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"phase": "panel_container_launched", "container": name}), flush=True)
            while process.poll() is None:
                if native_completion(process, completion_receipt,
                        completion_field=completion_field, completion_value=True) is not None:
                    break
                if cancelled.is_set():
                    raise RuntimeError("paired evaluator cancelled after peer failure")
                mem = memory_kib()
                container = owned_memory_sample(name, process, completion_receipt,
                    completion_field=completion_field, completion_value=True)
                resources.write(json.dumps({"elapsed": time.monotonic() - started,
                    "mem_available_kib": mem["MemAvailable"], "swap_free_kib": mem["SwapFree"],
                    "container": container,
                    "memory_safety": memory_safety(initial, mem, [container])}) + "\n")
                resources.flush()
                if time.monotonic() - started > wall_budget:
                    raise RuntimeError("bounded panel wall timeout")
                time.sleep(5)
    finally:
        cleanup_owned(name)
        if process is not None:
            process.wait(timeout=30)
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/chown",
            "--mount", f"type=bind,src={output},dst=/evidence", IMAGE,
            "-R", f"{os.getuid()}:{os.getgid()}", "/evidence"], check=True, timeout=30)
    if process.returncode != 0:
        raise RuntimeError("panel native container failed")


def build_warm_command(name, run, overlay, source_root, panel, request, request_sha,
                       checkpoints, motions, physical_gpu, deployment_host):
    baseline, baseline_sha = checkpoints["bc"]
    candidate, _ = checkpoints["dagger"]
    cmd = build_command(name, run, overlay, source_root / "pipeline/dagger/rtx/code",
                        source_root / "pipeline/deployment/rtx_actor", baseline, baseline_sha,
                        "bc", 1103, motions, physical_gpu=physical_gpu, memory_gib=40,
                        compact_host_heap=True, deployment_host=deployment_host)
    cmd = ["trainer._target_=isaac6_panel_evaluator.WarmPanelEvaluator"
           if arg.startswith("trainer._target_=") else arg for arg in cmd]
    cmd = ["env._target_=isaac6_panel_evaluator.PanelResetEnvironment"
           if arg.startswith("env._target_=") else arg for arg in cmd]
    cmd = [arg.replace("PYTHONPATH=", "PYTHONPATH=/evaluation-source:")
           if arg.startswith("PYTHONPATH=") else arg for arg in cmd]
    # Docker rejects --no-healthcheck together with explicit health options.
    # Those options belong to the training journal, not this native evaluator.
    health_options = {"--health-cmd", "--health-interval", "--health-timeout",
                      "--health-start-period", "--health-retries"}
    iterator = iter(cmd)
    cleaned = []
    for arg in iterator:
        if arg in health_options:
            next(iterator)
        else:
            cleaned.append(arg)
    cmd = cleaned
    specification = json.loads(request.read_bytes())
    extra = ["--mount", f"type=bind,src={panel.resolve()},dst=/panel.json,readonly",
             "--mount", f"type=bind,src={source_root / 'pipeline/evaluation'},dst=/evaluation-source,readonly",
             "--mount", f"type=bind,src={candidate},dst=/diagnostic-assets/warm-candidate.pt,readonly",
             "--mount", f"type=bind,src={request},dst=/warm-request.json,readonly",
             "--env", f"WARM_PANEL_REQUEST_SHA256={request_sha}",
             "--env", f"BETA_PANEL_SHA256={specification['panel_sha256']}",
             "--env", f"BETA_PANEL_COHORT_LIMIT={specification['cohort_limit']}",
             # The image's default health command expects a DAgger journal.
             # This runner monitors native evaluation progress/resources instead.
             "--no-healthcheck"]
    cmd[cmd.index(IMAGE):cmd.index(IMAGE)] = extra
    return cmd


def main():
    def interrupted(signum, frame):
        # Let the enclosing executor set cancellation and join the workers;
        # each worker then stops only its own labelled Docker container.
        raise InterruptedError("owned success evaluator stop requested")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deployment-host", choices=[HOSTNAME, "leo-rtx-6k"], default=HOSTNAME)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-sha256", required=True)
    parser.add_argument("--dagger-checkpoint", type=Path, required=True)
    parser.add_argument("--dagger-sha256", required=True)
    parser.add_argument("--cohort-limit", type=int, choices=[2, 52], default=52)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parallel-arms", action="store_true",
                      help="BC/candidate on separate devices; 21 GiB caps and cold-heap profile")
    mode.add_argument("--warm-arms", action="store_true",
                      help="One native B5 process; hot-load candidate after the complete reference panel")
    parser.parse_args()
    raise RuntimeError(
        "Legacy ray-traced student evaluation is retired. Use the corrected "
        "resident evaluate_explicit_panel operation with the rendered fixed295/smoke "
        "panel and hash-pinned rendered_observation_config. Historical artifacts remain readable."
    )


if __name__ == "__main__":
    main()
