"""Explicit full-resolution CPU writer probe; not Isaac/capacity certification."""
import argparse
import ctypes
import gc
import json
from pathlib import Path
import tempfile
import time

import torch

from pipeline.deployment.rtx_actor import bounded_bc_storage as storage
from test.pipeline.test_bc_native_same_input import owner


def rss():
    return next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmRSS:'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', type=int, default=4)
    parser.add_argument('--trim', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='bc-writer-memory-') as directory:
        value = owner(Path(directory))
        value.env.num_envs = 1
        value.vg_control_res = [240, 416]
        depth = torch.linspace(.01, 3, 480*832).reshape(1, 480, 832, 1)
        rgb = torch.arange(480*832*3, dtype=torch.int32).reshape(1, 480, 832, 3).to(torch.uint8)
        value._depth_chunk = [depth] * 81
        value._robot_rgb_chunk = [rgb] * 81
        value._views_chunk = [{view: dict(depth=depth, robot_rgb=rgb, rgb=None)
                              for view in ['hand_left', 'hand_right', 'head_color']}] * 81
        for lane in range(args.lanes):
            before = rss()
            start = time.monotonic()
            arrays = storage._lane_arrays(value, 0)
            payload = sum(a.nbytes for a in arrays.values())
            during = rss()
            del arrays
            gc.collect()
            after = rss()
            if args.trim:
                ctypes.CDLL(None).malloc_trim(0)
            print(json.dumps(dict(lane=lane, seconds=time.monotonic()-start,
                payload_bytes=payload, rss_before_kib=before, rss_arrays_kib=during,
                rss_deleted_kib=after, rss_final_kib=rss())), flush=True)


if __name__ == '__main__':
    main()
