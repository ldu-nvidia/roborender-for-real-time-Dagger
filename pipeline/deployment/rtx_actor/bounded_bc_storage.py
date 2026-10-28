"""Opt-in BC memory placement, not a new rollout or episode algorithm.

Keep the original 81-row raw window on the actor GPU, then convert one lane
at a time on CPU. Normalization still precedes episode reassembly. This is a
different writer implementation and requires its own same-input and live
capacity evidence; it must never claim the original writer ran unchanged.
"""
from __future__ import annotations

import os
import stat

PROTOCOL = "roborender.bc-gpu-window-lane-writer.v1"
ROWS = 81
VIEWS = {"head_color", "hand_left", "hand_right"}


def recording_masks(owner):
    ledger = getattr(owner, "_bc_ledger", None)
    if ledger is None or getattr(ledger, "drain", None) is None:
        return None
    masks = ledger.window_recording_masks
    if (len(masks) != ROWS or any(len(row) != owner.env.num_envs
            or any(type(value) is not bool for value in row) for row in masks)):
        raise RuntimeError("bounded drain writer requires a complete actual native recording mask")
    return masks


def release_payload_cache(path):
    """Persist one closed owned payload, then advise away only its file cache.

    This neither deletes data nor changes the memory budget. DONTNEED is an
    advisory request, so subsequent admission still uses actual cgroup usage.
    Never use global drop_caches or a broad directory traversal here.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise RuntimeError("completed regular BC payload required for cache release")
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return dict(bytes=info.st_size, device=info.st_dev, inode=info.st_ino,
                    advisory_only=True, fsync_completed=True)
    finally:
        os.close(fd)


def raw_window_bytes(batch, height=480, width=832):
    """Primary plus three conditioning streams, exact float32/uint8 storage."""
    if type(batch) is not int or not 1 <= batch <= 50:
        raise ValueError("one bounded B1..50 population is required")
    return ROWS * batch * height * width * 4 * (4 + 3)


def archive_latest_capture(owner, device):
    """Replace only the newest original CPU captures with owned CUDA copies.

    Called after the original policy_step; no capture/model/RNG/physics calls
    are repeated. Copying is synchronous so original camera storage cannot be
    reused before its data is owned. CPU is allowed here for differential tests;
    the native guard independently requires the exact actor CUDA device.
    """
    import torch

    n = len(owner._depth_chunk)
    if not 1 <= n <= ROWS or len(owner._robot_rgb_chunk) != n or len(owner._views_chunk) != n:
        raise RuntimeError("original capture must append exactly one complete frame")
    views = owner._views_chunk[-1]
    if set(views) != VIEWS or any(v.get("rgb") is not None for v in views.values()):
        raise RuntimeError("bounded capture requires precisely three non-debug views")

    def move(value, dtype):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device.type != "cpu":
            raise RuntimeError("original CPU camera dtype/device changed")
        return value.detach().to(device=device, copy=True, non_blocking=False)

    owner._depth_chunk[-1] = move(owner._depth_chunk[-1], torch.float32)
    owner._robot_rgb_chunk[-1] = move(owner._robot_rgb_chunk[-1], torch.uint8)
    owner._views_chunk[-1] = {name: dict(depth=move(value["depth"], torch.float32),
        robot_rgb=move(value["robot_rgb"], torch.uint8), rgb=None) for name, value in views.items()}


def validate_device_buffers(owner, device):
    """Require every raw tensor on the admitted device, without implicit casts."""
    import torch

    device = torch.device(device)
    for depth, rgb, views in zip(owner._depth_chunk, owner._robot_rgb_chunk, owner._views_chunk):
        pairs = [(depth, torch.float32), (rgb, torch.uint8)] + [
            pair for view in views.values() for pair in
            ((view["depth"], torch.float32), (view["robot_rgb"], torch.uint8))]
        if any(not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype
               for value, dtype in pairs):
            raise RuntimeError("bounded full window is not on its exact admitted device/dtype")


def tail_memory_budget(owner):
    """Copied-tail high-water mark for this accepted window's lane-major writer.

    The current tails are already charged to memory.current. Account separately
    for their possible growth, including the transient 81st row before emission.
    This is tensor/array payload accounting, not a bound on Python, NumPy or NPZ
    allocator overhead; the guard reserves additional conversion/writer scratch.
    """
    b = int(owner.env.num_envs)
    height, width = owner._depth_chunk[0].shape[1:3]
    control_h, control_w = owner.vg_control_res
    # Per view: uint16 native depth; resized float16 depth, one-channel uint8
    # control and three-channel uint8 robot RGB. Control is NOT RGB here.
    frame_bytes = 3 * (height * width * 2 + control_h * control_w * (2 + 1 + 3))
    for key in [key for key in owner.algo_obs_dim_dict if key != "vision_obs"] + ["gt_actions"]:
        value = getattr(owner.storage, key)
        frame_bytes += value[0, 0].numel() * value.element_size()
    tails = owner._ep_buf if owner._ep_buf is not None else [{} for _ in range(b)]
    if len(tails) != b:
        raise RuntimeError("bounded tail population changed")
    counts, existing = [], 0
    for tail in tails:
        count = len(tail.get("gt", []))
        values = list(tail.get("frames", {}).values()) + list(tail.get("obs", {}).values())
        values.append(tail.get("gt", []))
        if count >= ROWS or any(len(rows) != count for rows in values):
            raise RuntimeError("bounded tail lengths changed")
        rows = [value for group in values for value in group]
        if any(value.base is not None or not value.flags.owndata for value in rows):
            raise RuntimeError("bounded tails must own their payload, not alias an older window")
        size = sum(value.nbytes for value in rows)
        if size != count * frame_bytes:
            raise RuntimeError("bounded tail schema/dtype differs from the accepted window")
        counts.append(count)
        existing += size
    current = peak = existing
    dones = owner.storage.dones.cpu().numpy()
    masks = recording_masks(owner)
    if dones.shape != (ROWS, b, 1):
        raise RuntimeError("bounded tail budget requires all accepted done rows")
    for lane, count in enumerate(counts):
        for t in range(ROWS):
            if masks is not None and not masks[t][lane]:
                if count:
                    raise RuntimeError("drained lane retained recorded rows")
                continue
            count += 1
            current += frame_bytes
            peak = max(peak, current)
            if count == ROWS:
                current -= count * frame_bytes
                count = 0
            if bool(dones[t, lane, 0]):
                current -= count * frame_bytes
                count = 0
    return dict(current_tail_bytes=existing, peak_tail_bytes=peak,
                additional_tail_bytes=peak - existing, final_tail_bytes=current,
                bytes_per_tail_row=frame_bytes)


def _lane_arrays(owner, lane):
    """Original numerical operations, with B=1 because each lane is independent.

    h5_v4's percentiles/min/max still see the same entire original window, NOT
    the eventual within-episode clip. Keep insertion order for NPZ byte parity.
    """
    import cv2
    import numpy as np
    import torch

    def stack(rows):
        return torch.stack([row[lane:lane + 1] for row in rows], dim=0).cpu()

    # Original primary conversion has no saved fields in three-view collection.
    # Retain it nevertheless, without retaining its discarded full-B arrays.
    owner._depth_to_control_video(stack(owner._depth_chunk))
    arrays = {}
    for name in owner._views_chunk[0]:
        depth = stack([frame[name]["depth"] for frame in owner._views_chunk])
        native = depth.squeeze(-1).numpy().astype(np.float32)
        shift, dmin, dmax = 3.5, 0.0, 10.0
        lo, hi = np.log(dmin + shift), np.log(dmax + shift)
        arrays[f"depth_rawq__{name}"] = np.clip(((np.log(np.clip(native, 0, dmax) + shift) - lo)
            / (hi - lo) * 16383).round(), 0, 16383).astype(np.uint16)
        raw = depth.squeeze(-1).numpy().astype(np.float16)
        h, w = owner.vg_control_res
        raw = np.stack([np.stack([cv2.resize(raw[t, b], (w, h), interpolation=cv2.INTER_NEAREST)
            for b in range(raw.shape[1])]) for t in range(raw.shape[0])])
        arrays[f"depth_raw__{name}"] = raw
        control = owner._depth_to_control_video(depth, view=name)
        arrays[f"depth_ctrl__{name}"] = owner._resize_seq(control, owner.vg_control_res)
        rgb = stack([frame[name]["robot_rgb"] for frame in owner._views_chunk]).numpy()
        arrays[f"rgb_robot__{name}"] = owner._resize_seq(rgb, owner.vg_control_res)
    return arrays


def _write_lane(owner, lane):
    """One lane's scratch lifetime ends before the next lane is converted."""
    import numpy as np

    arrays = _lane_arrays(owner, lane)
    obs = {"obs_" + key: getattr(owner.storage, key)[:, lane:lane + 1].cpu().numpy()
           for key in owner.algo_obs_dim_dict if key != "vision_obs"}
    gt = owner.storage.gt_actions[:, lane:lane + 1].cpu().numpy()
    dones = owner.storage.dones[:, lane:lane + 1].cpu().numpy()
    masks = recording_masks(owner)
    for t in range(ROWS):
        buf = owner._ep_buf[lane]
        if masks is not None and not masks[t][lane]:
            if buf.get("gt"):
                raise RuntimeError("cannot drop an unfinished recorded episode")
            if bool(dones[t, 0, 0]):
                owner._ep_seq_base[lane] += 2
            continue
        if not buf:
            buf.update(frames={key: [] for key in arrays}, gt=[], obs={key: [] for key in obs})
        if list(buf["frames"]) != list(arrays) or list(buf["obs"]) != list(obs):
            raise RuntimeError("saved tail schema changed between windows")
        for key, value in arrays.items():
            # A tail must not pin the full prior window's backing arrays.
            buf["frames"][key].append(value[t, 0].copy())
        buf["gt"].append(gt[t, 0].copy())
        for key, value in obs.items():
            buf["obs"][key].append(value[t, 0].copy())
        if len(buf["gt"]) == ROWS:
            path = os.path.join(owner.vg_collect_dir,
                f"chunk_{owner._vg_run_tag}_{owner._vg_saved_chunks:07d}.npz")
            values = dict(gt_actions=np.stack(buf["gt"]).astype(np.float32),
                dones=np.zeros(ROWS, dtype=bool), prompt=np.array(owner.vg_prompt),
                env_id=np.array(lane), seq=np.array(owner._ep_seq_base[lane]),
                **{key: np.stack(value) for key, value in buf["frames"].items()},
                **{key: np.stack(value).astype(np.float32) for key, value in buf["obs"].items()})
            disk = getattr(owner, "_bc_disk_guard", None)
            bound = disk.admit_archive(path, values) if disk is not None else None
            np.savez_compressed(path, **values)
            # Release the stacked archive before retaining the next episode row.
            del values
            if disk is not None:
                disk.verify_archive(path, bound)
            written = getattr(owner, "_bc_payload_written", None)
            if written is not None:
                written(path)
            owner._vg_saved_chunks += 1
            owner._ep_seq_base[lane] += 1
            buf["frames"], buf["gt"], buf["obs"] = {key: [] for key in arrays}, [], {key: [] for key in obs}
        if bool(dones[t, 0, 0]):
            owner._ep_buf[lane] = {}
            owner._ep_seq_base[lane] += 2


def write_window(owner):
    """Emit the same NPZ bytes and tail/sequence state with bounded CPU scratch.

    This does not touch RNG, env.step/reset, teacher/student, or optimizer. A
    failure consumes this simulator lifetime exactly as the original writer;
    this function implements no resume, retry, tail flush, or quota.
    """
    b = int(owner.env.num_envs)
    if (not 1 <= b <= 50 or owner.vg_mode != "collect" or not owner.vg_save_raw_depth
            or not owner.vg_save_raw_depth_native or owner.vg_save_sim_rgb
            or not owner.vg_control_res
            or any(len(getattr(owner, key)) != ROWS for key in
                   ("_depth_chunk", "_robot_rgb_chunk", "_views_chunk"))):
        raise RuntimeError("bounded writer requires complete original three-view81-row windows")
    order = list(owner._views_chunk[0])
    if set(order) != VIEWS or any(list(frame) != order for frame in owner._views_chunk):
        raise RuntimeError("conditioning view insertion order changed inside a window")
    if owner._ep_buf is None:
        owner._ep_buf, owner._ep_seq_base = [{} for _ in range(b)], [0] * b
    if len(owner._ep_buf) != b or len(owner._ep_seq_base) != b:
        raise RuntimeError("episode population changed between windows")
    for lane in range(b):
        _write_lane(owner, lane)
        boundary = getattr(owner, "_bc_writer_lane_boundary", None)
        if boundary is not None:
            boundary(lane)
    owner._depth_chunk, owner._robot_rgb_chunk, owner._views_chunk = [], [], []
