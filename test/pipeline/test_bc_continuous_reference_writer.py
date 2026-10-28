"""Execute the pinned Raven writer against the continuous provenance ledger.

CPU-only differential test: real NPZ writing, terminal boundary cases and
cross-window episode buffers; no simulator or fabricated native qualification.
"""
from test.deployment.source_paths import REPO_ROOT
import ast
import hashlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    CHUNK_ROWS, NATIVE_COLLECTOR_SHA256, SCHEMA_CONTINUOUS, ChunkProvenanceLedger,
)


def reference_writer():
    reference = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985"
    path = reference / "gr00t/rl/trl/trainer/distill_trainer_videogen.py"
    if not path.is_file():
        pytest.skip("requires materialized pinned Raven collection reference")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == NATIVE_COLLECTOR_SHA256
    cls = next(n for n in ast.parse(raw).body
               if isinstance(n, ast.ClassDef) and n.name == "TRLDistillTrainerVideoGen")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_save_collected_chunks")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(torch=torch, np=np, os=os, VIDEO_MODEL_CHUNK=CHUNK_ROWS,
                     logger=logging.getLogger(__name__))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


def test_actual_raven_writer_matches_every_continuous_emission(tmp_path):
    windows, lanes = 8, 3
    inventory = [dict(motion_id=f"m{i}", path=f"/motions/m{i}.npz",
                      sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(5)]
    request = dict(schema=SCHEMA_CONTINUOUS, run_id="bc-reference-differential",
        cohort_id="continuous", split="train", process_seed=0, rank=0,
        global_env_offset=0, motions=inventory[:lanes], motion_inventory=inventory,
        max_windows=windows, output_dir=str(tmp_path), sampling_plan_sha256="a" * 64)
    ledger = ChunkProvenanceLedger(request)
    for lane in range(lanes):
        ledger.task_reset(lane, sample_idx=lane, sample_idx_in_seq=0, rng_sha256="b" * 64)
        ledger.episode_reset(lane)
    owner = SimpleNamespace(vg_collect_dir=str(tmp_path), _vg_run_tag="reference",
        vg_prompt="fixture", _vg_saved_chunks=0, _ep_buf=None, _ep_seq_base=None,
        vg_save_raw_depth_native=False, vg_save_raw_depth=False, vg_control_res=None,
        algo_obs_dim_dict={"actor_obs":113, "vision_obs":3},
        _depth_to_control_video=lambda x, **_: x.squeeze(-1).numpy().astype(np.uint8))
    write = reference_writer()
    # Row80 discarded; row81 emitted before done; row82 leaves a one-row tail.
    # Lane2 also continues well beyond the formerly problematic162-step mark.
    ends = ({80, 161, 243, 413}, {81, 163, 333, 600}, {401})
    for window in range(windows):
        start = window * CHUNK_ROWS
        dones = np.zeros((CHUNK_ROWS, lanes, 1), dtype=bool)
        labels = np.empty((CHUNK_ROWS, lanes, 26), dtype=np.float32)
        observations = np.empty((CHUNK_ROWS, lanes, 113), dtype=np.float32)
        for t in range(CHUNK_ROWS):
            step = start + t
            ledger.begin_transition([v["episode_step_index"] for v in ledger.current])
            flags = [(step + 1) in e for e in ends]
            for lane, done in enumerate(flags):
                labels[t, lane] = step
                observations[t, lane] = step + lane * 10000
                if done:
                    ledger.task_reset(lane, sample_idx=lane, sample_idx_in_seq=0, rng_sha256="c" * 64)
                    ledger.episode_reset(lane)
            ledger.finish_transition(flags)
            dones[t, :, 0] = flags
        owner._depth_chunk = [torch.full((lanes, 2, 2, 1), float(start+t)) for t in range(CHUNK_ROWS)]
        owner._robot_rgb_chunk = []
        owner._views_chunk = [{view: dict(depth=owner._depth_chunk[t],
            robot_rgb=torch.zeros((lanes, 2, 2, 3), dtype=torch.uint8))
            for view in ("head_color", "hand_left", "hand_right")} for t in range(CHUNK_ROWS)]
        owner.storage = SimpleNamespace(gt_actions=torch.from_numpy(labels),
            actor_obs=torch.from_numpy(observations), dones=torch.from_numpy(dones))
        before = owner._vg_saved_chunks
        write(owner)
        pending = ledger.pending()
        assert owner._vg_saved_chunks - before == len(pending)
        for index, expected in enumerate(pending, before):
            with np.load(tmp_path / f"chunk_reference_{index:07d}.npz", allow_pickle=False) as data:
                lane = expected["local_env_id"]
                row_ids = np.array([r["vector_step_index"] for r in expected["rows"]])
                assert int(data["env_id"]) == lane
                assert int(data["seq"]) == expected["seq"]
                np.testing.assert_array_equal(data["gt_actions"][:, 0], row_ids)
                np.testing.assert_array_equal(data["obs_actor_obs"][:, 0], row_ids + lane * 10000)
                assert not data["dones"].any()  # Raven serialization; sidecar keeps true flags.
                assert expected["selected_for_split"] is True
                assert len({r["episode_id"] for r in expected["rows"]}) == 1
            ledger.acknowledge(expected)
        assert [len(b.get("gt", [])) for b in owner._ep_buf] == [len(b) for b in ledger.buffers]
        assert owner._ep_seq_base == ledger.sequences
    assert ledger.vector_steps == windows * CHUNK_ROWS
    assert owner._vg_saved_chunks == sum(ledger.emitted_counts)
    assert ledger.reset_counts == [1+len(e) for e in ends]
    assert sum(len(b) for b in ledger.buffers) > 0  # Finite budget never flushes/pads partials.
    assert not ledger.pending()
