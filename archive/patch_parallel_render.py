#!/usr/bin/env python3
"""Parallelize image_dagger render calls: one client per env, threaded requests."""
p = "/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
s = open(p).read()
if "_id_env_clients" in s:
    print("already patched")
    raise SystemExit

s = s.replace('''            ep_files = vg.get("image3f_endpoint_files",
                              ["/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/image3f_server_endpoint.txt"])
            self._id_clients = [Image3FFrameClient.from_endpoint_file(f) for f in ep_files]''',
'''            self._id_endpoint_files = vg.get(
                "image3f_endpoint_files",
                ["/lustre/fs11/portfolios/llmservice/projects/llmservice_fm_vision/ldu_roborender_scale/src/fast_roborender/image3f_server_endpoint.txt"])
            self._id_env_clients = None  # one client per env, built lazily''')

s = s.replace('''        cam_h, cam_w = int(self.camera_resolution[0]), int(self.camera_resolution[1])
        t0 = time.perf_counter()
        vision = obs_dict["vision_obs"]
        new_vision = vision.clone().reshape(B, cam_h, cam_w, 3)
        for b in range(B):
            depth_d, mask_d, prev_d = {}, {}, {}
            for vname in self.vg_view_names:
                d = views[vname]["depth"][b, ..., :1].detach().to("cpu", torch.float32)
                ctrl = self._h5v4_control(d.numpy()[None, None, ..., 0], vname)  # [1,1,H,W]
                if self.vg_control_res:
                    ctrl = self._resize_seq(ctrl, self.vg_control_res)
                depth_d[vname] = ctrl[0, 0].astype(np.uint8)
                rr = views[vname]["robot_rgb"]
                m = rr[b].detach().to("cpu").numpy() if rr is not None else \\
                    np.zeros((*depth_d[vname].shape, 3), np.uint8)
                if self.vg_control_res and m.shape[:2] != tuple(self.vg_control_res):
                    m = self._resize_seq(m[None, None], self.vg_control_res)[0, 0]
                mask_d[vname] = m.astype(np.uint8)
                prev_d[vname] = self._id_anchors[b][vname]
            client = self._id_clients[b % len(self._id_clients)]
            gen = client.render_frame(depth_d, mask_d, prev_d,
                                      prompt=self._id_prompts[b], seed=self.vg_seed)
            rgb = gen[self.vg_primary_view]  # 240x416x3 u8
            dump_dir = os.environ.get("IMAGE_DAGGER_DUMP_DIR")
            if dump_dir and len(self._id_gen_ms) < 15:
                from PIL import Image as _PILImage
                os.makedirs(dump_dir, exist_ok=True)
                step_i = len(self._id_gen_ms)
                panel = np.concatenate(
                    [np.stack([depth_d[v]] * 3, -1) for v in self.vg_view_names]
                    + [gen[v] for v in self.vg_view_names], axis=1)
                _PILImage.fromarray(panel).save(
                    os.path.join(dump_dir, f"step{step_i:03d}_env{b}.png"))
            rgb_t = torch.from_numpy(rgb.copy()).float() / 255.0
            rgb_t = rgb_t.permute(2, 0, 1)[None]
            rgb_t = F.interpolate(rgb_t, size=(cam_h, cam_w),
                                  mode="bilinear", align_corners=False)
            rgb_t = rgb_t[0].permute(1, 2, 0)
            new_vision[b] = self._normalize_like_env(rgb_t[None])[0].to(
                device=new_vision.device, dtype=new_vision.dtype)''',
'''        cam_h, cam_w = int(self.camera_resolution[0]), int(self.camera_resolution[1])
        t0 = time.perf_counter()
        vision = obs_dict["vision_obs"]
        new_vision = vision.clone().reshape(B, cam_h, cam_w, 3)
        if self._id_env_clients is None or len(self._id_env_clients) != B:
            from gr00t.rl.utils.image3f_frame_client import Image3FFrameClient
            nf = len(self._id_endpoint_files)
            self._id_env_clients = [
                Image3FFrameClient.from_endpoint_file(self._id_endpoint_files[b % nf])
                for b in range(B)
            ]
        reqs = []
        for b in range(B):
            depth_d, mask_d, prev_d = {}, {}, {}
            for vname in self.vg_view_names:
                d = views[vname]["depth"][b, ..., :1].detach().to("cpu", torch.float32)
                ctrl = self._h5v4_control(d.numpy()[None, None, ..., 0], vname)  # [1,1,H,W]
                if self.vg_control_res:
                    ctrl = self._resize_seq(ctrl, self.vg_control_res)
                depth_d[vname] = ctrl[0, 0].astype(np.uint8)
                rr = views[vname]["robot_rgb"]
                m = rr[b].detach().to("cpu").numpy() if rr is not None else \\
                    np.zeros((*depth_d[vname].shape, 3), np.uint8)
                if self.vg_control_res and m.shape[:2] != tuple(self.vg_control_res):
                    m = self._resize_seq(m[None, None], self.vg_control_res)[0, 0]
                mask_d[vname] = m.astype(np.uint8)
                prev_d[vname] = self._id_anchors[b][vname]
            reqs.append((depth_d, mask_d, prev_d))

        from concurrent.futures import ThreadPoolExecutor

        def _render(b):
            depth_d, mask_d, prev_d = reqs[b]
            return self._id_env_clients[b].render_frame(
                depth_d, mask_d, prev_d, prompt=self._id_prompts[b], seed=self.vg_seed)

        with ThreadPoolExecutor(max_workers=B) as pool:
            gens = list(pool.map(_render, range(B)))

        dump_dir = os.environ.get("IMAGE_DAGGER_DUMP_DIR")
        for b in range(B):
            gen = gens[b]
            rgb = gen[self.vg_primary_view]  # 240x416x3 u8
            if dump_dir and len(self._id_gen_ms) < 15:
                from PIL import Image as _PILImage
                os.makedirs(dump_dir, exist_ok=True)
                step_i = len(self._id_gen_ms)
                panel = np.concatenate(
                    [np.stack([reqs[b][0][v]] * 3, -1) for v in self.vg_view_names]
                    + [gen[v] for v in self.vg_view_names], axis=1)
                _PILImage.fromarray(panel).save(
                    os.path.join(dump_dir, f"step{step_i:03d}_env{b}.png"))
            rgb_t = torch.from_numpy(rgb.copy()).float() / 255.0
            rgb_t = rgb_t.permute(2, 0, 1)[None]
            rgb_t = F.interpolate(rgb_t, size=(cam_h, cam_w),
                                  mode="bilinear", align_corners=False)
            rgb_t = rgb_t[0].permute(1, 2, 0)
            new_vision[b] = self._normalize_like_env(rgb_t[None])[0].to(
                device=new_vision.device, dtype=new_vision.dtype)''')

open(p, "w").write(s)
print("parallel render patched")
