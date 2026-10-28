"""Prepare the October 24 additive storage catalog; never move old payloads.

Emits a migration plan for explicit review/application by storage_layout. Old
locators below are this dated migration's inputs, not new launcher defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborender.infrastructure.paths import load_registry

BC116 = "2ec87a3c7b76aaa8fbb57d4fd50d02335dbfed75c164f5d741bea0067a33d7a9"
TEACHERS = {
    "manipulation_teacher": ("robot_policy/g1_privileged_action_teacher_wsdpt_step044500.pt", "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38"),
    "locomotion_stand": ("locomotion/g1_locomotion_stand_controller.pt", "ceb976d2745bdaa99e51ba7141e441dc7a140638ff8e2b0efaf50efd20290054"),
    "locomotion_walk": ("locomotion/g1_locomotion_walk_controller.pt", "2efa9d39c6ace247d6d68b43452f591cd5b22fb2f5e58dcfd3b6c5d1b30df77e"),
}


def plan(registry, host):
    loc = lambda key: registry.resolve(host, key)
    if host not in ("cosmos", "leo"):
        raise ValueError("migration is explicitly bound to cosmos or leo")
    roots = [loc("artifact_root")]
    if host == "leo":
        roots.append(loc("source_root").parent)
    dirs = {loc(k) for k in ("artifact_root", "release_root", "run_root", "cache_root", "python_envs", "policy_models", "student_checkpoints", "datasets", "catalog")}
    dirs.update(loc("student_checkpoints") / x for x in ("behavior_cloning", "dagger"))
    dirs.add(loc("policy_models") / "visual_student_bootstrap")
    links = []
    copies = []

    def link(destination, target):
        links.append({"path": str(destination), "target": str(target)})
        dirs.add(destination.parent)

    def copy(destination, target, digest):
        copies.append({"path": str(destination), "target": str(target), "sha256": digest})
        dirs.add(destination.parent)

    if host == "cosmos":
        old = Path("/workspace/project/roborender")
        renderer = old / "renderer"
        data = old / "data/BC/clean_single_20261020"
        teacher = old / "checkpoints/teachers"
        checkpoint = old / "checkpoints/students/BC/prod-10-22-BC-resume-best16-fit0015/validation_checkpoints/epoch_0116_step_018560.pt"
        for key, target in {
            "render_foundation": renderer / "models/wan21_fun_control_1p3b_foundation",
            "offline_bc_adapter": renderer / "offline_teacher/checkpoints/roborender_offline_bc_video_teacher_50step_lora_step05294.safetensors",
            "online_policy_adapter": renderer / "online_image3f/checkpoints/roborender_online_dagger_image3f_5step_lora_step05000.safetensors",
            "online_policy_engine_sm90": renderer / "online_image3f/runtime/assets/generated/engines/sm90_028945bead2e2305/image3f_dit_block_stack_reference_bf16.engine",
            "render_anchors": renderer / "anchors/roborender_online_anchor_pool_teacher50_2048",
            "bc_clean_demonstrations": data / "isaac",
            "bc_clean_rendered": data / "rendered_independent_v2",
            "bc_clean_manifests": data / "manifests",
            "bc_clean_splits": data / "splits",
        }.items():
            link(loc(key), target)
        link(loc("python_envs") / "image3f", renderer / "envs/image3f-py313-cu130-trt10133")
        link(loc("python_envs") / "cpu_tests", old / "runtime/test-venv-system")
    else:
        old = Path("/home/ldu/roborender")
        bundle = Path("/home/ldu/BC_render/rtx_BC_transfer")
        raven = bundle / "pipeline/GR00T-VisualSim2Real"
        teacher = bundle / "checkpoints/teachers"
        checkpoint = old / ("runtime/standing-isaac/standing-isaac-leo-prod-bc116-image3f295-recovery-20261024-v1/assets/" + BC116 + ".pt")
        dirs.update(loc(k) for k in ("simulation_assets", "container_images", "archived_runs"))
        for key, target in {
            "raven_runtime": raven,
            "robots": raven / "gr00t/rl/data/robots",
            "objects": raven / "gr00t/rl/data/objects",
            "motions": raven / "gr00t/rl/data/motions",
            "render_anchors": Path("/home/ldu/roborender_assets/anchors/anchor_pool_2048"),
            "isaac_image_archive": Path("/home/ldu/roborender_migration_beta_to_leo_20261013/isaac6-7aa2c5aa.tar.zst"),
        }.items():
            link(loc(key), target)
        link(loc("python_envs") / "actor_control", old / ".venv")
        link(loc("python_envs") / "cloud_access", Path("/home/ldu/.local/share/roborender-cloud/venv-compatible"))
    link(loc("legacy_runs"), old / "runtime")
    link(loc("legacy_releases"), old / "runtime/source-releases")
    for key, (relative, digest) in TEACHERS.items():
        copy(loc(key), teacher / relative, digest)
    copy(loc("bc_epoch116"), checkpoint, BC116)
    # Enumerate parents explicitly; the application cannot create unlisted roots.
    for directory in list(dirs):
        for root in roots:
            if directory.is_relative_to(root):
                dirs.update(p for p in directory.parents if p == root or p.is_relative_to(root))
    return {"schema": "roborender.storage-catalog.v1", "roots": sorted(map(str, roots)),
            "directories": sorted(map(str, dirs - set(roots))), "links": links, "copy_files": copies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", choices=("cosmos", "leo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (json.dumps(plan(load_registry(args.registry), args.host), indent=2, sort_keys=True) + "\n").encode()
    # Plans are evidence, not mutable operator settings.
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"plan": str(args.output), "host": args.host, "executed": False}))


if __name__ == "__main__":
    main()
