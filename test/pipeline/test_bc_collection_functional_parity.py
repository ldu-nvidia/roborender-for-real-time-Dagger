"""CPU A/B of pinned Raven's real train/rollout/writer and our BC adapter.

Physics, teacher inference, storage and framework services are doubles. The
entire pinned train loop, rollout, collection policy wrapper, depth conversion
and writer execute unmodified. The incidental-epoch case also executes the
original discarded Normal sample and real AdamW on a tiny deterministic model.
This is CPU boundary/RNG evidence, not native physics qualification.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.deployment.rtx_actor import initial_reset_contract
from pipeline.deployment.rtx_actor import isaac6_bc_production as bc
from test.deployment.source_paths import REPO_ROOT


REFERENCE = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer"
PINS = {
    "ppo_trainer.py": "f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61",
    "distill_trainer.py": "dc386dcf8b2be0c151403fd0e8841129c939d99dc5e9e7850e92829b3ad50ca2",
    "distill_trainer_videogen.py": "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
}
ENDS = ((80, 243, 405), (81, 163, 325), (82, 400))  # One-based terminal steps.
WIDTHS = {"actor_obs": 113, "teacher_obs": 226, "homie_obs": 83,
          "gt_obj_pos": 3, "vision_obs": 6}


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _methods(filename, class_name, names):
    path = REFERENCE / filename
    if not path.is_file():
        pytest.skip(f"pinned Raven reference unavailable: {path}")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINS[filename]
    cls = next(n for n in ast.parse(raw).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == set(names)
    return methods


def _classes(monkeypatch):
    class PolicyBoundary:
        def policy_step(self, policy, obs, cur_dones=None):
            assert not torch.is_grad_enabled() and not self.model.training
            self.env.events.append((self.env.steps, "policy"))
            labels = obs["actor_obs"][:, :26].clone() + torch.arange(26) / 8
            # Deliberately different student actions: the original collector
            # must replace ALL26 values, including the synthetic Homie tail.
            return {"actions": labels - 999, "gt_actions": labels}

    nodes = []
    for name, base, filename, original, methods in (
        ("OriginalPPO", "PolicyBoundary", "ppo_trainer.py", "TRLPPOTrainer",
         {"train", "_rollout_step", "_train_mode", "_train_rollout_mode"}),
        ("OriginalDistill", "OriginalPPO", "distill_trainer.py", "TRLDistillTrainer",
         {"_rollout_step"}),
        ("OriginalVideo", "OriginalDistill", "distill_trainer_videogen.py",
         "TRLDistillTrainerVideoGen", {"policy_step", "_rollout_step", "_get_depth_frame",
          "_depth_to_control_video", "_h5v4_control", "_resize_seq", "_save_collected_chunks"}),
    ):
        nodes.append(ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())],
            keywords=[], decorator_list=[], body=_methods(filename, original, methods)))
    # inspect.getsourcefile in the real adapter must resolve the actual pinned
    # reference bytes, not a changed pin or a mock hash function.
    module = ModuleType("_bc_functional_pinned_reference")
    module.__file__ = str(REFERENCE / "distill_trainer_videogen.py")
    module.__dict__.update(PolicyBoundary=PolicyBoundary, torch=torch, np=np, os=os,
        time=time, math=math, gc=gc, VIDEO_MODEL_CHUNK=81, logger=logging.getLogger(__name__),
        unwrap_model_for_generation=lambda model, *_a, **_k: nullcontext(model),
        process_ep_infos=lambda *_a: {}, wandb_run_exists=lambda: False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "hash_pinned_bc_reference_methods", "exec"), module.__dict__)
    source = Path(bc.__file__)
    adapter = next(n for n in ast.walk(ast.parse(source.read_text()))
                   if isinstance(n, ast.ClassDef) and n.name == "Isaac6BCProductionTrainer")
    namespace = {**vars(bc), "inspect": inspect, "np": np, "torch": torch,
                 "TRLDistillTrainerVideoGen": module.OriginalVideo}
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), str(source), "exec"), namespace)
    return module.OriginalVideo, namespace["Isaac6BCProductionTrainer"]


class Storage:
    def __init__(self):
        self.clears = 0

    def clear(self):
        self.clears += 1
        self.rows = {}
        self.step = 0

    def update_key(self, key, value):
        self.rows.setdefault(key, []).append(value.clone())

    def increment_step(self):
        self.step += 1

    def __getattr__(self, key):
        return torch.stack(self.rows[key])


class StagedBoundary:
    """Marker substituted only for the original loop's isinstance dependency."""


class Environment(StagedBoundary):
    def __init__(self, request):
        self.num_envs, self.device, self.steps = 3, "cpu", 0
        self.events, self.actions, self.first_clocks = [], [], None
        self.reset_calls, self.lane_resets = 0, [0] * 3
        self.episode_length_buf = torch.zeros(3, dtype=torch.int64)
        self.time_in_stage_buf = torch.zeros(3, dtype=torch.int64)
        self.actual_time_in_stage_buf = torch.zeros(3, dtype=torch.int64)
        self.max_episode_length, self.max_stage_time = 400, [137]
        self.stage_buf = torch.zeros(3, dtype=torch.int64)
        self.need_to_refresh_envs = torch.zeros(3, dtype=torch.bool)
        self.config = AttrDict(reset_from_dataset=AttrDict(), task=AttrDict(
            front_object="front", hold_object="hold", tray_object="tray"))
        self.reset_dataset = SimpleNamespace(
            motion_files=[m["path"] for m in request["motion_inventory"]],
            sample_idx_based_on_demo_clip_number=[(i, i + 1) for i in range(5)])
        self.simulator = SimpleNamespace(
            bc_motion_binding_sha256=hashlib.sha256(bc.canonical(request["motions"])).hexdigest(),
            robot_root_states=torch.zeros(3, 13), dof_pos=torch.zeros(3, 26),
            dof_vel=torch.zeros(3, 26), get_task_root_state=lambda _: torch.zeros(3, 13),
            get_depth_hi_image=lambda: self.capture("depth", 0),
            get_robot_rgb_image=lambda: self.capture("robot", 0),
            get_conditioning_views=lambda: {
                view: {"depth": self.capture("view_depth", i),
                       "robot_rgb": self.capture("view_robot", i), "rgb": None}
                for i, view in enumerate(bc.VIEWS)})

    def observe(self):
        self.obs_buf_dict = {key: self.steps + torch.arange(3)[:, None] * 1000
            + torch.arange(width)[None, :] / 128 for key, width in WIDTHS.items()}
        return {key: value.clone() for key, value in self.obs_buf_dict.items()}

    def capture(self, kind, view):
        self.events.append((self.steps, kind))
        grid = torch.arange(24).reshape(1, 4, 6, 1)
        if "robot" in kind:
            return ((grid + self.steps + view * 23 + torch.arange(3)[:, None, None, None])
                    .expand(3, 4, 6, 3) % 256).to(torch.uint8)
        depth = (grid / 25 + self.steps / 1000 + view / 8
                 + torch.arange(3)[:, None, None, None] / 16).float()
        depth[:, 0, 0] = 0  # Actual h5_v4 invalid-depth branch, no converter stub.
        return depth

    def _reset_task_states(self, ids):
        self.sample_idx = ids.clone()
        self.sample_idx_in_seq = torch.zeros(len(ids), dtype=torch.int64)

    def reset_envs_idx(self, ids):
        self._reset_task_states(ids)
        self.episode_length_buf[ids] = 0
        for lane in ids.tolist():
            self.lane_resets[lane] += 1

    def reset_all(self):
        self.reset_calls += 1
        initial_reset_contract.begin_initial_reset(self)
        self.reset_envs_idx(torch.arange(3))
        self.need_to_refresh_envs.fill_(True)
        initial_reset_contract.complete_initial_reset(self)
        return self.observe()

    def set_is_evaluating(self, *_a, **_k):
        pass

    def step(self, state):
        self.events.append((self.steps, "step"))
        assert torch.equal(state["actions"], state["gt_actions"])
        if self.first_clocks is None:
            self.first_clocks = tuple(t.clone() for t in (self.episode_length_buf,
                self.time_in_stage_buf, self.actual_time_in_stage_buf))
        self.actions.append(state["actions"].clone())
        self.steps += 1
        self.episode_length_buf += 1
        done = torch.tensor([self.steps in ends for ends in ENDS])
        ids = torch.nonzero(done).flatten()
        if len(ids):
            self.reset_envs_idx(ids)
        return self.observe(), torch.zeros(3), done, {
            "episode": {}, "time_outs": torch.zeros(3, dtype=torch.bool)}


def _owner(cls, request, model_file, resize):
    owner = cls()
    owner.env = Environment(request)
    owner.storage = Storage()
    owner.vg_collect_dir = request["output_dir"]
    owner._vg_run_tag = bc.source_run_tag(request)
    owner._vg_saved_chunks, owner._ep_buf, owner._ep_seq_base = 0, None, None
    owner.vg_mode, owner.vg_rollout_action_source = "collect", "teacher"
    owner.vg_enabled, owner.vg_save_sim_rgb = True, False
    owner.vg_save_raw_depth_native = owner.vg_save_raw_depth = True
    owner.vg_control_res, owner.vg_prompt = resize, "CPU fixture, not native evidence"
    owner.vg_depth_mode, owner.vg_invalid_fill = "h5_v4", "far"
    owner.vg_h5v4_default = dict(raw_clip=[0.05, 10.0], inv_clip=[1.0, 40.0])
    owner.vg_h5v4_per_view = {
        "head_color": dict(raw_clip=[0.05, 10.0], inv_clip=[1., 40.], norm_range=[.05, 2.6]),
        **{view: dict(raw_clip=[0., 1.5], inv_clip=[.7, 100.], raw_pct=[1, 99])
           for view in ("hand_left", "hand_right")}}
    owner.algo_obs_dim_dict = WIDTHS
    owner.num_steps_per_env = 81
    owner.config = AttrDict(init_at_random_ep_len=True,
        network_load_dict=SimpleNamespace(teacher_actor=SimpleNamespace(path=str(model_file))),
        homie_walk_model_path=str(model_file), homie_stand_model_path=str(model_file))
    owner.args = SimpleNamespace(num_ppo_epochs=0, num_total_batches=request["max_windows"],
        total_episodes=request["max_windows"] * 3, logging_steps=None, eval_steps=None,
        save_steps=None, batch_size=3, ds3_gather_for_generation=False, output_dir=request["output_dir"])
    owner.accelerator = SimpleNamespace(device="cpu", num_processes=1,
        print=lambda *_a: None, unwrap_model=lambda model: model, gather_for_metrics=lambda t: t)
    owner.model = torch.nn.Module()
    owner.lifecycle = []
    def policy(name):
        return SimpleNamespace(is_recurrent=False, init_rollout=lambda: owner.lifecycle.append(name + ":begin"),
                               clear_rollout=lambda: owner.lifecycle.append(name + ":end"))
    owner.model.policy = owner.policy_model = policy("student")
    owner.ref_model = policy("teacher")
    owner.model.value_model = owner.value_model = None
    owner.model.set_mode = owner.model.transform_eval = owner.model.transform_train = lambda *_a: None
    owner.optimizer_calls = []
    owner.optimizer = SimpleNamespace(step=lambda: owner.optimizer_calls.append("step"))
    owner.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [0.0002], step=lambda: None)
    owner.state = SimpleNamespace(episode=0, global_step=0, rewbuffer=[], lenbuffer=[],
                                 tot_timesteps=0, tot_time=0)
    owner.control = SimpleNamespace(should_training_stop=False, should_save=False)
    owner.callback_handler = SimpleNamespace(**{name: lambda _a, _s, control: control
        for name in ("on_train_begin", "on_step_end", "on_train_end")})
    owner.dataloader, owner.train_dataset_len = None, 3
    owner.is_deepspeed_enabled = owner.train_with_evaluating_env = False
    owner.schedule_dict, owner.scheduled_params_dict = None, {}
    owner._register_stats_buffer = lambda: None
    owner._dagger_beta, owner._advance_dagger_beta = lambda: 1., lambda: None
    owner._get_rollout_data = lambda **_k: {}
    owner._get_train_metrics = lambda: {}
    owner._process_env_step = lambda *_a: None
    owner.episode_env_tensors = SimpleNamespace(mean_and_clear=lambda: {})
    owner.ep_infos, owner.logs = [], []
    owner.log = owner.logs.append
    owner.cur_reward_sum, owner.cur_episode_length = torch.zeros(3), torch.zeros(3)
    return owner


def _expected_chunks(total_steps):
    """Closed-form episode partitions: independent of adapter ledger logic."""
    chunks, discarded, tails = [], [], []
    for lane, ends in enumerate(ENDS):
        start = seq = dropped = 0
        for end in [e for e in ends if e <= total_steps] + [total_steps]:
            count, tail = divmod(end - start, 81)
            for chunk in range(count):
                rows = np.arange(start + chunk * 81, start + (chunk + 1) * 81)
                chunks.append(((int(rows[-1]) // 81, lane, int(rows[-1])), lane, seq + chunk, rows))
            seq += count
            if end in ends:
                dropped += tail
                seq += 2
            start = end
        discarded.append(dropped)
        tails.append(tail)
    return sorted(chunks, key=lambda item: item[0]), discarded, tails


@pytest.mark.parametrize("standalone,incidental_epoch", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("windows,resize", [(4, None), (8, [2, 3])])
def test_actual_original_train_loop_and_adapter_have_identical_complete_payloads(
        tmp_path, monkeypatch, windows, resize, incidental_epoch, standalone):
    # This dependency must really be available for the resize case; no fake
    # conversion or silent skip can turn this into a preprocessing certificate.
    if resize:
        import cv2
        assert callable(cv2.resize)
    reference_cls, adapter_cls = _classes(monkeypatch)
    staged = ModuleType("gr00t.rl.envs.base_task.staged_task_base")
    staged.StagedTaskBase = StagedBoundary
    monkeypatch.setitem(sys.modules, staged.__name__, staged)
    monkeypatch.setitem(sys.modules, "initial_reset_contract", initial_reset_contract)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _device: torch.get_rng_state())
    inventory = [dict(motion_id=f"m{i}", path=f"/motions/m{i}.npz",
                      sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(5)]
    raw = dict(schema=bc.SCHEMA_CONTINUOUS, run_id="bc-functional-cpu", cohort_id="continuous",
        split="train", process_seed=0, rank=0, global_env_offset=0, motions=inventory[:3],
        motion_inventory=inventory, max_windows=windows, output_dir=str(tmp_path / "adapter"),
        sampling_plan_sha256="a" * 64)
    if incidental_epoch:
        raw['collection_rng_protocol'] = bc.bc_collection_rng.PROTOCOL
    if standalone:
        raw['collection_loop_protocol'] = bc.COLLECTOR_PROTOCOL
    request_file = tmp_path / "request.json"
    request_file.write_bytes(bc.canonical(raw) + b"\n")
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST", str(request_file))
    monkeypatch.setenv("ROBORENDER_BC_COLLECTION_REQUEST_SHA256", bc.file_sha256(request_file))
    model_file = tmp_path / "cpu-identity-fixture.pt"
    model_file.write_bytes(b"Not teacher weights; no native model is loaded")
    reference_request = {**raw, "output_dir": str(tmp_path / "reference")}
    Path(reference_request["output_dir"]).mkdir()
    reference = _owner(reference_cls, reference_request, model_file, resize)
    adapter = _owner(adapter_cls, raw, model_file, resize)
    if incidental_epoch:
        from test.pipeline.test_bc_collection_rng import TinyPolicy, original_action_mean
        original_forward = original_action_mean()
        for owner in (reference, adapter):
            policy = TinyPolicy(3)
            policy.is_recurrent = False
            policy.init_rollout = lambda owner=owner: owner.lifecycle.append('student:begin')
            policy.clear_rollout = lambda owner=owner: owner.lifecycle.append('student:end')
            owner.model.policy = owner.policy_model = policy
            owner.args.local_batch_size = owner.args.local_mini_batch_size = 3
            owner.args.per_device_train_batch_size = 3
            owner.ppo_shuffle_every_epoch = True
            owner.empty_cache_every_n_ppo_epoch = 0
            owner.accelerator.accumulate = lambda _: nullcontext()
            owner.accelerator.backward = lambda loss: loss.backward()
            owner._get_mb_rollout_data = lambda *_: {}
            owner._forward_model = lambda model, _: original_forward(model.policy, {})
            owner._compute_loss = lambda result, _: {'loss': result['action_mean'].square().mean()}
            owner._gradient_clipping = lambda: None
            owner._update_stats_buffer = lambda *_: None
            owner.optimizer = torch.optim.AdamW(policy.parameters(), lr=.0002)
        reference.args.num_ppo_epochs = 1
        # Only the native architecture gate is replaced for this declared tiny
        # CPU double. Separate tests pin/check the real architecture and source.
        def tiny_admission(owner):
            assert isinstance(owner.policy_model, TinyPolicy)
            receipt = bc.bc_collection_rng.expected_receipt(3, 0)
            receipt.pop('completed_windows')
            return receipt
        monkeypatch.setattr(bc.bc_collection_rng, 'validate_native', tiny_admission)
    # Match filenames as well as bytes; changing an output directory is not a
    # scientific difference, and source_run_tag intentionally binds requests.
    reference._vg_run_tag = bc.source_run_tag(raw)
    rng = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        expected_episode = torch.randint(400, (3,))
        expected_stage = torch.randint(137, (3,))
        torch.manual_seed(0)
        reference.train()
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        report = adapter.train()
        assert torch.equal(torch.get_rng_state(), reference_rng)
    finally:
        torch.set_rng_state(rng)
    assert reference.env.events == adapter.env.events
    assert reference.env.reset_calls == adapter.env.reset_calls == 1
    assert reference.env.lane_resets == adapter.env.lane_resets == [
        1 + sum(end <= windows * 81 for end in ends) for ends in ENDS]
    assert reference.lifecycle == adapter.lifecycle == [
        "teacher:begin", "student:begin", "student:end", "teacher:end"] * windows
    for owner in (reference, adapter):
        assert owner.env.steps == windows * 81 and owner.storage.clears == windows
        assert owner.state.global_step == windows and not owner.optimizer_calls
        for actual, expected in zip(owner.env.first_clocks, (expected_episode, expected_stage, expected_stage)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for step in range(windows * 81):
            events = [event for when, event in owner.env.events if when == step]
            assert events == ["depth", "robot", "view_depth", "view_robot",
                              "view_depth", "view_robot", "view_depth", "view_robot", "policy", "step"]
        # Includes the rows discarded by the writer: correct saved chunks alone
        # would not catch altered behavior actions in short terminal tails.
        wanted_actions = (torch.arange(windows * 81)[:, None, None]
            + torch.arange(3)[None, :, None] * 1000
            + torch.arange(26)[None, None, :] * (1 / 128 + 1 / 8))
        torch.testing.assert_close(torch.stack(owner.env.actions), wanted_actions, rtol=0, atol=0)
    for key, actual in zip(("episode_length", "time_in_stage", "actual_time_in_stage"),
                          adapter.env.first_clocks):
        assert report["initial_state_attestation"]["arrays"][key] == bc.array_attestation(actual.numpy())
    expected, discarded, tails = _expected_chunks(windows * 81)
    files = sorted(Path(raw["output_dir"]).glob("chunk_*.npz"))
    original_files = sorted(Path(reference_request["output_dir"]).glob("chunk_*.npz"))
    records = list(bc.iter_continuous_chunks(report))
    assert len(files) == len(original_files) == len(records) == len(expected)
    for path, original, record, (_, lane, seq, rows) in zip(files, original_files, records, expected):
        assert path.name == original.name
        assert path.read_bytes() == original.read_bytes()
        with np.load(path, allow_pickle=False) as data, np.load(original, allow_pickle=False) as before:
            assert data.files == before.files
            assert set(data.files) == {"gt_actions", "dones", "prompt", "env_id", "seq"} | {
                f"obs_{key}" for key in WIDTHS if key != "vision_obs"} | {
                f"{kind}__{view}" for view in bc.VIEWS
                for kind in ("depth_rawq", "depth_raw", "depth_ctrl", "rgb_robot")}
            for key in data.files:
                assert data[key].dtype == before[key].dtype
                np.testing.assert_array_equal(data[key], before[key])
            assert int(data["env_id"]) == lane and int(data["seq"]) == seq
            for key, width in WIDTHS.items():
                if key != "vision_obs":
                    wanted = rows[:, None] + lane * 1000 + np.arange(width)[None, :] / 128
                    np.testing.assert_array_equal(data[f"obs_{key}"], wanted.astype(np.float32))
            labels = rows[:, None] + lane * 1000 + np.arange(26)[None, :] * (1 / 128 + 1 / 8)
            np.testing.assert_array_equal(data["gt_actions"], labels.astype(np.float32))
            assert not data["dones"].any()  # Original serialization, even terminal-at81.
        sidecar = json.loads(path.with_suffix(".provenance.json").read_text())
        assert [r["vector_step_index"] for r in sidecar["rows"]] == rows.tolist()
        assert [r["actual_done"] for r in sidecar["rows"]] == [int(row) + 1 in ENDS[lane] for row in rows]
        assert record["selected_for_split"] and not record["overcollection"]
    assert report["discarded_reset_tail_rows_per_lane"] == discarded
    assert report["remaining_unsealed_tail_rows_per_lane"] == tails
    assert report["emitted_chunks"] == report["selected_chunks"] == len(expected)
    assert report["overcollection_chunks"] == report["optimizer_steps"] == 0
    if incidental_epoch:
        assert report['collection_rng'] == bc.bc_collection_rng.expected_receipt(3, windows)
        assert len(reference.optimizer.state) > 0 and len(adapter.optimizer.state) == 0
    assert report["accepted_windows"] == windows and report["initial_reset_all_calls"] == 1
    assert report["status"] == "complete"
    assert [len(b.get("gt", [])) for b in reference._ep_buf] == tails
    assert [len(b.get("gt", [])) for b in adapter._ep_buf] == tails
