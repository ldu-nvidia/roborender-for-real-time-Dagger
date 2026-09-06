# Local runtime provenance

The qualified Isaac actor runtime is physically stored once at
`runtime/isaac_actor/qualified_h200_v1` (approximately 30 GiB). It was moved on
2026-09-06 from the historical agent-run path
`/workspace/codex/roborender-agent/runs/h200_rollout_qualification_20260903T201136Z`
on the same filesystem, preserving all inodes and bytes. That historical path
is now a compatibility symlink so immutable manifests containing the old
absolute location remain readable; active launchers use only the canonical
project path.

Runtime identity is verified from its checked inventories for Isaac Sim,
IsaacLab, actor Python packages, GR00T source, CUDA/GPU inventory, and the
headless Kit configuration. The directory is a local dependency, not a dataset
or model checkpoint. It must not be copied into Git or a container image.
