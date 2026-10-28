"""Standalone collection tests: CPU evidence only, never native capacity."""
import ast
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.bc_collection import collector
from pipeline.bc_collection.chunk_writer import write_window
from pipeline.bc_collection.continuous_plan import build_plan, build_request
from test.pipeline.test_bc_continuous_reference_writer import reference_writer
from test.pipeline.test_bc_collection_functional_parity import (
    _classes, _owner, Environment, StagedBoundary,
    test_actual_original_train_loop_and_adapter_have_identical_complete_payloads as compare_actual,
)


def test_default_plan_is_32_lane_standalone_without_quota():
    rows = [dict(motion_id=f'm{i}', path=f'/motions/{i}.npz', sha256=f'{i+1:064x}') for i in range(40)]
    plan = build_plan(rows, run_id='fresh', max_windows=100, output_dir='/run/raw')
    request = build_request(plan)
    assert plan['num_envs'] == collector.DEFAULT_NUM_ENVS == 32
    assert request['collection_loop_protocol'] == collector.PROTOCOL
    assert len(request['motion_inventory']) == 40
    assert 'chunks_per_env' not in request
    assert plan['selection_uses_outcomes'] is False
    from pipeline.deployment.rtx_actor.isaac6_bc_production import validate_collection_request
    request.pop('collection_rng_protocol')
    with pytest.raises(ValueError, match='RNG protocol'):
        validate_collection_request(request)


def test_32_lanes_repeated_episodes_and_more_than_nine_chunks_match_raven(tmp_path):
    original = reference_writer()
    owners = []
    for name in ('original', 'standalone'):
        directory = tmp_path/name
        directory.mkdir()
        owners.append(SimpleNamespace(vg_collect_dir=str(directory), _vg_run_tag='same',
            vg_prompt='CPU differential', _vg_saved_chunks=0, _ep_buf=None, _ep_seq_base=None,
            vg_save_raw_depth_native=False, vg_save_raw_depth=False, vg_control_res=None,
            algo_obs_dim_dict={'actor_obs': 113, 'vision_obs': 3},
            _depth_to_control_video=lambda x, **_: x.squeeze(-1).numpy().astype(np.uint8)))
    # Includes termination exactly at81, short tails, and episodes well past
    # 162steps/ninechunks. Lane0 repeats14times; lane31 spans1000steps.
    periods = [81, 82, 163, 400] * 8
    periods[-1] = 1000
    for window in range(14):
        steps = np.arange(window*81+1, (window+1)*81+1)
        labels = np.broadcast_to(steps[:, None, None], (81, 32, 26)).astype(np.float32)
        obs = np.broadcast_to(steps[:, None, None], (81, 32, 113)).astype(np.float32)
        dones = (steps[:, None] % np.array(periods)[None, :] == 0)[..., None]
        for owner, writer in zip(owners, (original, write_window)):
            owner._depth_chunk = [torch.full((32, 1, 1, 1), float(step % 255)) for step in steps]
            owner._robot_rgb_chunk = []
            owner._views_chunk = []
            owner.storage = SimpleNamespace(gt_actions=torch.from_numpy(labels),
                actor_obs=torch.from_numpy(obs), dones=torch.from_numpy(dones))
            writer(owner)
        assert owners[0]._vg_saved_chunks == owners[1]._vg_saved_chunks
        assert owners[0]._ep_seq_base == owners[1]._ep_seq_base
    left = sorted((tmp_path/'original').glob('*.npz'))
    right = sorted((tmp_path/'standalone').glob('*.npz'))
    per_lane = [0]*32
    for a, b in zip(left, right):
        assert a.name == b.name and a.read_bytes() == b.read_bytes()
        with np.load(b, allow_pickle=False) as clip:
            lane = int(clip['env_id'])
            per_lane[lane] += 1
            rows = clip['gt_actions'][:, 0].astype(int)
            assert np.all(np.diff(rows) == 1)
            assert len(set((rows-1)//periods[lane])) == 1
    assert per_lane[0] == 14
    assert per_lane[-1] > 9
    assert len(owners[1]._ep_buf[-1]['gt']) == 53  # No forced final flush.


def test_no_general_trainer_or_learning_callbacks_in_collection_loop():
    from pathlib import Path
    tree = ast.parse(Path(collector.__file__).read_text())
    forbidden = {'train', 'backward', '_forward_model', '_compute_loss', 'evaluate',
                 '_save_checkpoint', 'on_step_end', 'reinit_sim', '_get_rollout_data'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # model.train() toggles mode only; it is not the inherited trainer.
            if node.func.attr == 'train':
                assert ast.unparse(node.func.value) == 'owner.model'
            else:
                assert node.func.attr not in forbidden


def test_failed_step_propagates_without_reset_or_retry(tmp_path, monkeypatch):
    reference, _ = _classes(monkeypatch)
    request = dict(run_id='failure', cohort_id='single', rank=0,
        max_windows=4, output_dir=str(tmp_path), motion_inventory=[
        dict(path=f'/motions/{i}') for i in range(5)], motions=[dict(path=f'/motions/{i}') for i in range(3)])
    owner = _owner(reference, request, tmp_path/'unused.pt', None)
    obs = owner.env.reset_all()
    calls = []
    def fail(state):
        calls.append(state)
        raise RuntimeError('native state uncertain')
    owner.env.step = fail
    with torch.no_grad(), pytest.raises(RuntimeError, match='native state uncertain'):
        collector.collect_window(owner, owner.model, obs)
    assert len(calls) == 1 and owner.env.reset_calls == 1
    assert owner._in_train_rollout is False and owner._vg_saved_chunks == 0


def test_inclusive_last_motion_sample_is_valid(tmp_path, monkeypatch):
    reset = Environment._reset_task_states
    def last_sample(self, ids):
        reset(self, ids)
        self.sample_idx += 1  # Fixture ranges are [lane,lane+1], both valid.
    monkeypatch.setattr(Environment, '_reset_task_states', last_sample)
    compare_actual(tmp_path, monkeypatch, windows=4, resize=None,
                   incidental_epoch=True, standalone=True)


@pytest.mark.parametrize('field,value', [
    ('dataloader', object()), ('schedule_dict', {}), ('value_model', object()),
    ('train_with_evaluating_env', True), ('is_deepspeed_enabled', True),
    ('num_steps_per_env', 162),
])
def test_unsupported_runtime_fails_before_reset(tmp_path, monkeypatch, field, value):
    reference, _ = _classes(monkeypatch)
    staged = ModuleType('gr00t.rl.envs.base_task.staged_task_base')
    staged.StagedTaskBase = StagedBoundary
    monkeypatch.setitem(sys.modules, staged.__name__, staged)
    request = dict(run_id='invalid', cohort_id='single', rank=0,
        max_windows=4, output_dir=str(tmp_path), motion_inventory=[
            dict(path=f'/motions/{i}') for i in range(5)], motions=[dict(path=f'/motions/{i}') for i in range(3)])
    owner = _owner(reference, request, tmp_path/'unused.pt', None)
    setattr(owner, field, value)
    with pytest.raises(ValueError, match='standalone collection'):
        collector.collect(owner, 4)
    assert owner.env.reset_calls == owner.env.steps == 0
