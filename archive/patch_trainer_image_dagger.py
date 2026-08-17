#!/usr/bin/env python3
"""Adds image_dagger mode to distill_trainer_videogen.py (idempotent)."""
p = "/svl/u/ravenh/fast_roborender/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
s = open(p).read()
if "_image_dagger_relabel" in s:
    print("already patched")
    raise SystemExit

s = s.replace('assert self.vg_mode in ("online", "collect", "offline")',
              'assert self.vg_mode in ("online", "collect", "offline", "image_dagger")')

s = s.replace('''        if self.vg_mode == "collect":
            assert self.vg_collect_dir, "videogen.collect_dir required in collect mode"''',
'''        if self.vg_enabled and self.vg_mode == "image_dagger":
            from gr00t.rl.utils.image3f_frame_client import Image3FFrameClient

            ep_files = vg.get("image3f_endpoint_files",
                              ["/svl/u/ravenh/fast_roborender/image3f_server_endpoint.txt"])
            self._id_clients = [Image3FFrameClient.from_endpoint_file(f) for f in ep_files]
            self.vg_anchor_pool_dir = vg.get(
                "anchor_pool_dir", "/svl/u/ravenh/fast_roborender/teacher_fullres")
            self.vg_anchor_index = int(vg.get("anchor_index", 0))
            self._id_anchor_files = sorted(
                glob.glob(os.path.join(self.vg_anchor_pool_dir, "*_teacher3v.npz")))
            assert self._id_anchor_files, f"no anchors in {self.vg_anchor_pool_dir}"
            self._id_anchors = None      # per-env dict view->HxWx3
            self._id_prompts = None      # per-env str
            self._id_rng = np.random.RandomState(self.vg_seed)
            self._id_gen_ms = []

        if self.vg_mode == "collect":
            assert self.vg_collect_dir, "videogen.collect_dir required in collect mode"''')

s = s.replace('''            self._depth_chunk.append(self._get_depth_frame())''',
'''            if self.vg_mode == "image_dagger":
                obs_dict = self._image_dagger_relabel(obs_dict, cur_dones)

            self._depth_chunk.append(self._get_depth_frame())''')

s = s.replace('''    def _rollout_step(self, model, obs_dict):
        if self.vg_mode == "offline":''',
'''    def _id_sample_anchor(self, env_i: int):
        f = self._id_anchor_files[self._id_rng.randint(len(self._id_anchor_files))]
        with np.load(f, allow_pickle=True) as z:
            anchor = {v: z[f"gen__{v}"][self.vg_anchor_index].copy()
                      for v in self.vg_view_names}
            prompt = str(z["prompt"]) if "prompt" in z.files else self.vg_prompt
        self._id_anchors[env_i] = anchor
        self._id_prompts[env_i] = prompt

    def _image_dagger_relabel(self, obs_dict, cur_dones):
        """Per-frame 3-view generated obs via the warm Image3F server.

        Swaps vision_obs for every env BEFORE the policy acts, so both action
        selection and storage see generated RGB (true per-step DAgger).
        Anchor (fixed video-model reference triplet, frame vg_anchor_index)
        is sampled per episode from the teacher pool, resampled on done.
        """
        if not hasattr(self.env.simulator, "get_conditioning_views"):
            return obs_dict
        views = self.env.simulator.get_conditioning_views()
        if not views:
            return obs_dict
        any_v = next(iter(views.values()))
        B = int(any_v["depth"].shape[0])
        if self._id_anchors is None:
            self._id_anchors = [None] * B
            self._id_prompts = [None] * B
            for b in range(B):
                self._id_sample_anchor(b)
        if cur_dones is not None:
            done_idx = torch.nonzero(cur_dones.reshape(-1)).reshape(-1).tolist()
            for b in done_idx:
                if b < B:
                    self._id_sample_anchor(b)

        cam_h, cam_w = int(self.camera_resolution[0]), int(self.camera_resolution[1])
        t0 = time.perf_counter()
        vision = obs_dict["vision_obs"]
        new_vision = vision.clone().reshape(B, cam_h, cam_w, 3)
        for b in range(B):
            depth_d, mask_d, prev_d = {}, {}, {}
            for vname in self.vg_view_names:
                d = views[vname]["depth"][b, ..., :1].detach().to("cpu", torch.float32)
                ctrl = self._h5v4_control(d.numpy()[None, ..., 0], vname)[0]
                if self.vg_control_res:
                    ctrl = self._resize_seq(ctrl[None], self.vg_control_res)[0]
                depth_d[vname] = ctrl.astype(np.uint8)
                rr = views[vname]["robot_rgb"]
                m = rr[b].detach().to("cpu").numpy() if rr is not None else \\
                    np.zeros((ctrl.shape[0], ctrl.shape[1], 3), np.uint8)
                if self.vg_control_res and m.shape[:2] != tuple(self.vg_control_res):
                    m = self._resize_seq(m[None], self.vg_control_res)[0]
                mask_d[vname] = m.astype(np.uint8)
                prev_d[vname] = self._id_anchors[b][vname]
            client = self._id_clients[b % len(self._id_clients)]
            gen = client.render_frame(depth_d, mask_d, prev_d,
                                      prompt=self._id_prompts[b], seed=self.vg_seed)
            rgb = gen[self.vg_primary_view]  # 240x416x3 u8
            rgb_t = torch.from_numpy(rgb.copy()).float() / 255.0
            rgb_t = rgb_t.permute(2, 0, 1)[None]
            rgb_t = F.interpolate(rgb_t, size=(cam_h, cam_w),
                                  mode="bilinear", align_corners=False)
            rgb_t = rgb_t[0].permute(1, 2, 0)
            new_vision[b] = self._normalize_like_env(rgb_t[None])[0].to(
                device=new_vision.device, dtype=new_vision.dtype)
        self._id_gen_ms.append((time.perf_counter() - t0) * 1000)
        if len(self._id_gen_ms) % 50 == 1:
            logger.info("image_dagger: %d envs relabeled in %.0fms (avg over %d steps: %.0fms)",
                        B, self._id_gen_ms[-1], len(self._id_gen_ms),
                        float(np.mean(self._id_gen_ms)))
        out = dict(obs_dict)
        out["vision_obs"] = new_vision.reshape(vision.shape)
        return out

    def _rollout_step(self, model, obs_dict):
        if self.vg_mode == "offline":''')

open(p, "w").write(s)
print("trainer patched")
