"""Raven 94d7985 collection writer, extracted without algorithm changes.

Window-based depth conversion precedes within-episode chunk reassembly.
Append -> emit81 -> terminal-tail discard; never reset Isaac or select a quota.
The default CPU-buffer recipe is retained; bounded GPU storage is separately
qualified. Differential tests execute the pinned original writer on same inputs.
"""
import logging
import os

import numpy as np
import torch

VIDEO_MODEL_CHUNK = 81
logger = logging.getLogger(__name__)


def write_window(self):
    depth = torch.stack(self._depth_chunk, dim=0)  # [T, B, H, W, 1]
    self._depth_chunk = []
    control = self._depth_to_control_video(depth)  # [T, B, H, W]
    num_envs = control.shape[1]

    # Robot-only RGB (textured robot on black) — the video model's
    # control_video_2 / rgb_robot conditioning.
    robot_rgb = None
    if self._robot_rgb_chunk and len(self._robot_rgb_chunk) == control.shape[0]:
        robot_rgb = torch.stack(self._robot_rgb_chunk, dim=0).numpy()  # [T,B,H,W,3] u8
    self._robot_rgb_chunk = []

    # Multi-view conditioning: per-view depth control + robot RGB
    # (keys depth_ctrl__{view} / rgb_robot__{view}).
    view_arrays = {}
    if self._views_chunk and len(self._views_chunk) == control.shape[0]:
        for vname in self._views_chunk[0]:
            vdepth = torch.stack([sv[vname]["depth"] for sv in self._views_chunk], dim=0)
            if self.vg_save_raw_depth_native:
                dn_ = vdepth.squeeze(-1).numpy().astype(np.float32)
                shift, dmin, dmax = 3.5, 0.0, 10.0
                ln, lx = np.log(dmin + shift), np.log(dmax + shift)
                q = np.clip(((np.log(np.clip(dn_, 0, dmax) + shift) - ln)
                             / (lx - ln) * 16383).round(), 0, 16383).astype(np.uint16)
                view_arrays[f"depth_rawq__{vname}"] = q
            if self.vg_save_raw_depth:
                raw = vdepth.squeeze(-1).numpy().astype(np.float16)
                if self.vg_control_res:
                    import cv2
                    h, w = self.vg_control_res
                    raw = np.stack([np.stack([
                        cv2.resize(raw[t, b], (w, h), interpolation=cv2.INTER_NEAREST)
                        for b in range(raw.shape[1])]) for t in range(raw.shape[0])])
                view_arrays[f"depth_raw__{vname}"] = raw
            ctrl = self._depth_to_control_video(vdepth, view=vname)
            if self.vg_control_res:
                ctrl = self._resize_seq(ctrl, self.vg_control_res)
            view_arrays[f"depth_ctrl__{vname}"] = ctrl
            if all(sv[vname]["robot_rgb"] is not None for sv in self._views_chunk):
                rr = torch.stack(
                    [sv[vname]["robot_rgb"] for sv in self._views_chunk], dim=0).numpy()
                if self.vg_control_res:
                    rr = self._resize_seq(rr, self.vg_control_res)
                view_arrays[f"rgb_robot__{vname}"] = rr
            if all(sv[vname].get("rgb") is not None for sv in self._views_chunk):
                rs = torch.stack(
                    [sv[vname]["rgb"] for sv in self._views_chunk], dim=0).numpy()
                if self.vg_control_res:
                    rs = self._resize_seq(rs, self.vg_control_res)
                view_arrays[f"rgb_sim__{vname}"] = rs
    self._views_chunk = []

    # Save every registered obs key except vision (vision is what the
    # offline renderer synthesizes) so offline policy forwards see the
    # same obs dict as online ones (actor_obs, homie_obs, ...).
    obs_arrays = {}
    for key in self.algo_obs_dim_dict:
        if key == "vision_obs":
            continue
        obs_arrays[f"obs_{key}"] = getattr(self.storage, key).cpu().numpy()
    gt_actions = self.storage.gt_actions.cpu().numpy()    # [T, B, A]
    dones = self.storage.dones.cpu().numpy()               # [T, B, 1]

    num_steps = control.shape[0]
    if self._ep_buf is None:
        self._ep_buf = [{} for _ in range(num_envs)]
        self._ep_seq_base = [0] * num_envs

    def _flush(b):
        self._ep_buf[b] = {}

    def _append(b, t):
        buf = self._ep_buf[b]
        if not buf:
            buf["frames"] = {k: [] for k in view_arrays} if view_arrays else {"depth_ctrl": []}
            buf["gt"] = []
            buf["obs"] = {k: [] for k in obs_arrays}
            if not view_arrays and robot_rgb is not None:
                buf["frames"]["rgb_robot"] = []
        if view_arrays:
            for k, v in view_arrays.items():
                buf["frames"][k].append(v[t, b])
        else:
            buf["frames"]["depth_ctrl"].append(control[t, b])
            if robot_rgb is not None:
                buf["frames"]["rgb_robot"].append(robot_rgb[t, b])
        buf["gt"].append(gt_actions[t, b])
        for k, v in obs_arrays.items():
            buf["obs"][k].append(v[t, b])

    def _emit(b):
        buf = self._ep_buf[b]
        idx = self._vg_saved_chunks
        path = os.path.join(
            self.vg_collect_dir, f"chunk_{self._vg_run_tag}_{idx:07d}.npz")
        np.savez_compressed(
            path,
            gt_actions=np.stack(buf["gt"]).astype(np.float32),
            dones=np.zeros(VIDEO_MODEL_CHUNK, dtype=bool),
            prompt=np.array(self.vg_prompt),
            env_id=np.array(b),
            seq=np.array(self._ep_seq_base[b]),
            **{k: np.stack(v) for k, v in buf["frames"].items()},
            **{k: np.stack(v).astype(np.float32) for k, v in buf["obs"].items()},
        )
        self._vg_saved_chunks += 1
        self._ep_seq_base[b] += 1
        buf["frames"] = {k: [] for k in buf["frames"]}
        buf["gt"] = []
        buf["obs"] = {k: [] for k in buf["obs"]}

    for b in range(num_envs):
        for t in range(num_steps):
            _append(b, t)
            if len(self._ep_buf[b]["gt"]) == VIDEO_MODEL_CHUNK:
                _emit(b)
            if bool(dones[t, b, 0]):
                # discard the partial tail of the ended episode and break
                # chaining: bump seq so the renderer never chains across
                # episode boundaries.
                _flush(b)
                self._ep_seq_base[b] += 2
    if self._vg_saved_chunks % (10 * num_envs) < num_envs:
        logger.info("videogen collect: %d chunks saved to %s",
                    self._vg_saved_chunks, self.vg_collect_dir)
