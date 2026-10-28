"""Explicit hardware identity for the dedicated Leo workstation.

This does not relax Beta's GPU0 prohibition, change physics, or admit a
production campaign. Leo's GPU2 is a display A400 and must not run the actors.
"""
HOSTNAME = "leo-rtx-6k"
GPU_UUIDS = {
    0: "GPU-4d58bc2b-43c8-6ff8-e7fd-2348575f0cff",
    1: "GPU-b92877a7-a5e0-9a83-3bc0-d8ede481f35d",
}
FORBIDDEN_GPUS = (2,)
BOUNDED_MODES = frozenset({"capture-only", "bc-collect", "isolated-one-step", "single-window", "capacity-probe"})


def checked_gpu(physical_gpu, mode, *, campaign=False):
    if type(physical_gpu) is not int or physical_gpu not in GPU_UUIDS:
        raise ValueError("Leo requires a pinned RTX PRO6000 GPU0/GPU1; display GPU2 is forbidden")
    if mode not in BOUNDED_MODES and not (campaign and mode == "beta-production"):
        raise ValueError("Leo migration binding is bounded; production requires separate campaign qualification")
    return GPU_UUIDS[physical_gpu]
