#!/usr/bin/env python
"""Offline renderer v2: adds video-model-generated RGB to collected chunks.

v2 sends ALL of the video model's training-time conditions:
  * depth control video (depth_ctrl)
  * robot RGB mask (rgb_robot: textured robot on black) when present
  * language prompt (stored per chunk at collection time)
  * previous-frame conditioning: consecutive chunks of the same env are
    grouped into one request (up to --max_chain chunks); the server's batch
    session chains 81-frame clips autoregressively with prev-frame reference.
    Groups are split at episode boundaries (chunks containing a done) and at
    seq discontinuities.

Chunks that already have rgb_gen are skipped (resumable). Survives render
server preemption/migration by re-reading the endpoint file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.environ.get(
    "ROBORENDER_ROOT", os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
sys.path.insert(0, os.path.join(ROOT, "GR00T-VisualSim2Real"))

CHUNK_T = 81


def shard_for(tag: str, env_id: int, nshards: int) -> int:
    """Stable shard for an entire collector/env stream, not an individual chunk."""
    shard_key = f"{tag}:{env_id}"
    return int.from_bytes(
        hashlib.blake2b(shard_key.encode(), digest_size=8).digest(), "big"
    ) % nshards


def main():
    from gr00t.rl.utils.videogen_client import VideoGenRenderClient

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument(
        "--endpoint_file", default=os.path.join(ROOT, "raven_server_endpoint.txt")
    )
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--student_res", type=int, nargs=2, default=[108, 192],
                    metavar=("H", "W"))
    ap.add_argument("--max_chain", type=int, default=6,
                    help="max chunks chained into one request (memory bound)")
    ap.add_argument("--prompt_override", default=None,
                    help="use this prompt instead of the one stored per chunk")
    ap.add_argument("--views", default="hand_left,hand_right,head_color",
                    help="comma-separated view names. Per-view chunk data "
                         "(depth_ctrl__{name}) is used when present; otherwise "
                         "the single ego view is duplicated into every slot")
    ap.add_argument("--primary_view", default="head_color",
                    help="view whose generated RGB becomes rgb_gen (student obs)")
    ap.add_argument("--preview_n", type=int, default=4)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="this renderer's slot")
    ap.add_argument("--nshards", type=int, default=1, help="total render shards")
    args = ap.parse_args()
    if args.nshards < 1:
        ap.error("--nshards must be positive")
    if not 0 <= args.shard < args.nshards:
        ap.error(f"--shard must be in [0, {args.nshards}), got {args.shard}")
    view_names = [v.strip() for v in args.views.split(",") if v.strip()]
    primary_view = args.primary_view if args.primary_view in view_names else view_names[0]

    def make_client():
        if args.host and args.port:
            host, port = args.host, args.port
        else:
            with open(args.endpoint_file) as f:
                host, port_s = f.read().split()
                port = int(port_s)
        print(f"render server: {host}:{port}", flush=True)
        return VideoGenRenderClient(host=host, port=port,
                                    connect_retries=6, connect_retry_interval_s=10)

    # tolerate the server still booting (or endpoint file not yet written)
    while True:
        try:
            client = make_client()
            break
        except Exception as exc:
            print(f"waiting for render server ({exc})", flush=True)
            time.sleep(30)

    def render_with_retry(views, masks, prompt, seed):
        nonlocal client
        while True:
            try:
                return client.render_chunk(views, mask_per_view=masks,
                                           prompt=prompt, seed=seed)
            except (ConnectionError, OSError, RuntimeError) as exc:
                print(f"render failed ({exc}); waiting for server...", flush=True)
                client.close()
                time.sleep(30)
                try:
                    client = make_client()
                except Exception:
                    pass

    st_h, st_w = args.student_res
    done_total = 0
    t_start = time.time()

    while True:
      try:
          # ---- scan and group -------------------------------------------------
          todo_meta = []
          all_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "chunk_*.npz")))
          for path in all_paths:
              try:
                  with np.load(path, allow_pickle=False) as z:
                      # Resume: a chunk is done only when it has the primary rgb_gen AND
                      # a per-view rgb_gen__<view> for every requested view. Chunks
                      # rendered before per-view support are therefore re-rendered.
                      have = ("rgb_gen" in z.files) and all(
                          f"rgb_gen__{v}" in z.files for v in view_names)
                      if have:
                          continue
                      meta = {
                          "path": path,
                          "env_id": int(z["env_id"]) if "env_id" in z.files else -1,
                          "seq": int(z["seq"]) if "seq" in z.files else -1,
                          "has_done": bool(z["dones"].any()) if "dones" in z.files else False,
                          "tag": os.path.basename(path).split("_")[1],
                      }
              except Exception as exc:
                  print(f"skip unreadable {path}: {exc}", flush=True)
                  continue
              # Keep every consecutive chunk for one collector/env on the same
              # renderer so --max_chain preserves autoregressive conditioning.
              # The key remains stable as --follow discovers new files.
              slot = shard_for(meta["tag"], meta["env_id"], args.nshards)
              if slot != args.shard:
                  continue
              todo_meta.append(meta)

          if not todo_meta:
              if not args.follow:
                  break
              time.sleep(30)
              continue

          by_env = defaultdict(list)
          for m in todo_meta:
              by_env[(m["tag"], m["env_id"])].append(m)

          groups = []
          for key, metas in by_env.items():
              metas.sort(key=lambda m: m["seq"])
              cur = []
              prev_seq = None
              for m in metas:
                  discontinuous = prev_seq is not None and m["seq"] != prev_seq + 1
                  if cur and (discontinuous or len(cur) >= args.max_chain):
                      groups.append(cur)
                      cur = []
                  cur.append(m)
                  prev_seq = m["seq"]
                  if m["has_done"]:  # episode ended inside this chunk
                      groups.append(cur)
                      cur = []
                      prev_seq = None
              if cur:
                  groups.append(cur)

          # ---- render groups --------------------------------------------------
          for group in groups:
              datas = []
              try:
                  for m in group:
                      with np.load(m["path"], allow_pickle=False) as z:
                          datas.append({k: z[k].copy() for k in z.files})
              except (FileNotFoundError, OSError, ValueError) as exc:
                  # files can vanish or be mid-write between scan and load
                  print(f"skip group ({exc})", flush=True)
                  continue

              depths, masks = {}, {}
              for name in view_names:
                  dkey = f"depth_ctrl__{name}" if f"depth_ctrl__{name}" in datas[0] else "depth_ctrl"
                  mkey = f"rgb_robot__{name}" if f"rgb_robot__{name}" in datas[0] else "rgb_robot"
                  depths[name] = np.concatenate([d[dkey] for d in datas])
                  if all(mkey in d for d in datas):
                      masks[name] = np.concatenate([d[mkey] for d in datas])
              masks = masks if len(masks) == len(view_names) else None
              prompt = args.prompt_override or str(datas[0].get("prompt", ""))

              resp = render_with_retry(depths, masks, prompt, 42)
              rgb = resp[primary_view]

              def _downsample(seg):
                  return np.stack([
                      np.asarray(Image.fromarray(f).resize((st_w, st_h), Image.BILINEAR))
                      for f in seg
                  ]).astype(np.uint8)

              for i, (m, data) in enumerate(zip(group, datas)):
                  seg = rgb[i * CHUNK_T:(i + 1) * CHUNK_T]
                  # rgb_gen stays the primary view so single-view runs are unaffected;
                  # rgb_gen__<view> is added for every view so a 3-view policy can tile.
                  data["rgb_gen"] = _downsample(seg)
                  for vname in view_names:
                      vrgb = resp.get(vname)
                      if vrgb is None:
                          continue
                      data[f"rgb_gen__{vname}"] = _downsample(
                          vrgb[i * CHUNK_T:(i + 1) * CHUNK_T])
                  tmp = m["path"] + ".tmp.npz"
                  np.savez_compressed(tmp, **data)
                  os.replace(tmp, m["path"])
                  done_total += 1

                  if done_total <= args.preview_n:
                      try:
                          import cv2
                          prev = m["path"].replace(".npz", "_preview.mp4")
                          # one row per view: [depth control | that view's generation]
                          t0, t1 = i * CHUNK_T, (i + 1) * CHUNK_T
                          per_view = []
                          for name in view_names:
                              ctrl_v = depths[name][t0:t1]
                              gen_v = resp[name][t0:t1]
                              per_view.append((ctrl_v, gen_v))
                          gh, gw = per_view[0][1].shape[1], per_view[0][1].shape[2]
                          wr = cv2.VideoWriter(prev, cv2.VideoWriter_fourcc(*"mp4v"),
                                               15, (gw * 2, gh * len(view_names)))
                          for j in range(t1 - t0):
                              rows = []
                              for ctrl_v, gen_v in per_view:
                                  c = cv2.resize(ctrl_v[j], (gw, gh),
                                                 interpolation=cv2.INTER_NEAREST)
                                  rows.append(np.concatenate(
                                      [np.repeat(c[..., None], 3, -1), gen_v[j]], axis=1))
                              wr.write(cv2.cvtColor(np.concatenate(rows, axis=0),
                                                    cv2.COLOR_RGB2BGR))
                          wr.release()
                          print(f"preview {prev} (rows: {','.join(view_names)})", flush=True)
                      except Exception as exc:
                          print(f"preview failed: {exc}", flush=True)

              if done_total % 20 < len(group):
                  rate = done_total / (time.time() - t_start)
                  print(f"rendered {done_total} chunks ({rate*3600:.0f}/h)", flush=True)

          if not args.follow:
              # one full pass done; loop once more to catch stragglers, then exit
              break
      except Exception as exc:
        # NFS hiccups (stale handles, vanished files) must not kill the job
        print(f"pass failed ({type(exc).__name__}: {exc}); retrying in 30s", flush=True)
        time.sleep(30)

    print(f"OFFLINE_RENDER_DONE total={done_total}", flush=True)


if __name__ == "__main__":
    main()
