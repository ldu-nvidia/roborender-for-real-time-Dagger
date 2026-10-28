"""CPU-only bounded-writer comparison against actual, hash-bound native NPZs.

Reuses the captured-input comparator's complete provenance and byte checks.
Only the writer under test changes; this never replays physics or claims CUDA
capacity. The original capture and comparison are immutable.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
from types import SimpleNamespace


def compare(runtime, reference_path, output, *, legacy_order_proof):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only visibility is required')
    import numpy as np
    import torch
    from pipeline.deployment.rtx_actor import bounded_bc_storage as storage

    path = Path(__file__).with_name('bc_native_same_input.py')
    spec = importlib.util.spec_from_file_location('bounded_captured_reference', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    writer_hash = original.sha(storage.__file__)
    factory, publish = original.reference_writer, original.publish
    before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())

    def bounded_factory(source):
        class Bounded(factory(source)):
            def _save_collected_chunks(self):
                self.env = SimpleNamespace(num_envs=int(self._depth_chunk[0].shape[0]))
                self.vg_mode, self.vg_save_sim_rgb = 'collect', False
                storage.write_window(self)
                gc.collect()
        return Bounded

    def publish_comparison(destination, value):
        if Path(destination) != Path(output) / 'comparison.json':
            raise ValueError('unexpected comparator write')
        after_np = np.random.get_state()
        if (not torch.equal(before[0], torch.get_rng_state())
                or before[1][0] != after_np[0] or before[1][2:] != after_np[2:]
                or not np.array_equal(before[1][1], after_np[1])
                or before[2] != random.getstate()
                or original.sha(storage.__file__) != writer_hash):
            raise AssertionError('writer source or RNG changed during CPU comparison')
        value.update(schema='roborender.bc-bounded-captured-input.v1',
            comparison_scope='bounded_writer_against_original_native_captured_inputs_and_outputs',
            writer_protocol=storage.PROTOCOL, writer_sha256=writer_hash,
            comparator_wrapper_sha256=original.sha(__file__),
            native_payload_writer_unchanged=False, cpu_only=True,
            cuda_transfer_tested=False, current_native_trajectory_tested=False,
            b32_capacity_certified=False, rng_unchanged=True,
            peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        return publish(destination, value)

    original.reference_writer = bounded_factory
    original.publish = publish_comparison
    return original.compare(runtime, reference_path, output,
                            legacy_order_proof=legacy_order_proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--reference-writer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-order-proof', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.runtime, args.reference_writer, args.output,
                             legacy_order_proof=args.legacy_order_proof), sort_keys=True))
