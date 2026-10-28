"""Historical five-motion BC/DAgger migration smoke on beta GPU1.

Retained for compatibility with the legacy paired beta evaluator. Its corrected
Raven goal accounting does not implement the accepted295 student reset protocol.
Production success uses bc_success_dispatch or the coverage campaign's resident
evaluate_explicit_panel operation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from qualify_graphics import HOSTNAME
from qualify_native_isaac6 import IMAGE, command
from run_success_diagnostic import gpu_gate


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(name, output, overlay, code, source, checkpoint, checkpoint_sha,
                  arm, seed, motions, reference=None, *, physical_gpu=1,
                  memory_gib=40, compact_host_heap=False, deployment_host=HOSTNAME):
    cmd = command(name, output, overlay, code, 5, seed=seed, memory_gib=memory_gib,
                  physical_gpu=physical_gpu, compact_host_heap=compact_host_heap, deployment_host=deployment_host)
    overrides = {
        "trainer._target_": "isaac6_success_diagnostic.SuccessDiagnosticTrainer",
        "algo.config.init_at_random_ep_len": "false",
        "++algo.config.raven_exact_raytraced_augmentation": "false",
        "algo.config.videogen.raytraced_mix_prob": "1.0",
    }
    cmd = [key + "=" + overrides[key] if key in overrides else value
           for value in cmd for key in [value.split("=", 1)[0]]]
    for index, value in enumerate(cmd):
        if value.startswith("PYTHONPATH="):
            cmd[index] = value.replace("PYTHONPATH=", "PYTHONPATH=/diagnostic-source:")
    mounts = [
        "--mount", f"type=bind,src={source},dst=/diagnostic-source,readonly",
        "--mount", f"type=bind,src={checkpoint},dst=/diagnostic-assets/best_validation.pt,readonly",
        "--env", f"DIAGNOSTIC_ARM={arm}", "--env", f"DIAGNOSTIC_SEED={seed}",
        "--env", f"DIAGNOSTIC_CHECKPOINT_SHA256={checkpoint_sha}",
        "--env", "DIAGNOSTIC_REFERENCE_DIR=/paired-reference" if reference else "DIAGNOSTIC_REFERENCE_DIR=",
    ]
    if reference:
        mounts += ["--mount", f"type=bind,src={reference},dst=/paired-reference,readonly"]
    cmd[cmd.index(IMAGE):cmd.index(IMAGE)] = mounts
    cmd += [
        "simulator._target_=isaac6_success_diagnostic.IsaacSim",
        "env._target_=isaac6_success_diagnostic.SuccessAccountingEnvironment",
        "++env.config.reset_from_dataset.use_motion_file_dir=false",
        "++env.config.reset_from_dataset.motion_files=[" + ",".join(motions) + "]",
        "++env.config.reset_from_dataset.motion_assignment_offset=0",
        "domain_rand.image_augmentation.enabled=false",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-sha256", required=True)
    parser.add_argument("--dagger-checkpoint", type=Path, required=True)
    parser.add_argument("--dagger-sha256", required=True)
    parser.add_argument("--seed", type=int, default=1103)
    parser.parse_args()
    raise RuntimeError(
        "Legacy ray-traced student evaluation is retired. Use the corrected "
        "resident evaluate_explicit_panel operation with the rendered fixed295/smoke "
        "panel and hash-pinned rendered_observation_config. Historical artifacts remain readable."
    )


if __name__ == "__main__":
    main()
