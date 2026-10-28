"""CPU worker checkpoint integration, not native/CUDA or policy qualification.

The tiny stochastic policy has the real 78-trainable-slot bookkeeping count,
but is deliberately not Raven's network. All checkpoint/evidence files are
synthetic fixtures beneath pytest's temporary directory.
"""
from copy import deepcopy
import ast
import inspect
from pathlib import Path
import random
import textwrap

import numpy as np
import pytest
import torch

from pipeline.dagger.services import bc32_diagnostic_parent as diagnostic
from pipeline.dagger.services import diagnostic_reentry as chain
from pipeline.dagger.services import learner_rng as rng
from pipeline.dagger.services import raven_learner_worker as worker
from test.pipeline.test_bc32_diagnostic_parent import bc_parent
from test.pipeline.test_diagnostic_reentry import initial
from test.pipeline.test_learner_rng import SeventyEightSlotPolicy


@pytest.fixture(autouse=True)
def cpu_rng_isolation(monkeypatch):
    # Never inspect or initialize a device, even on a GPU-equipped test host.
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)

    def set_cpu_only(states):
        assert states == []

    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', set_cpu_only)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        torch.set_num_threads(threads)


class WorkerStudent(SeventyEightSlotPolicy):
    def __init__(self):
        super().__init__()
        self.mlp_module[-1] = torch.nn.Linear(3, 11)
        self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
        self.is_recurrent = False
        self.max_rollout_history = 1
        self.running_mean_std = self.state_history_module = None
        self.clamp_noise_std = True
        self.max_noise_std = .001

    def init_rollout(self):
        pass

    def clear_rollout(self):
        pass

    def rollout_with_obj_pred(self, value):
        return self.action_mean_with_obj_pred(value)

    def action_mean_with_obj_pred(self, value):
        images = value['vision_obs']
        features = self.vision_module(images.reshape(-1, 3))
        # Exercise stochastic forward and activation-recomputation RNG handling.
        # This synthetic randomness is not a claim about Raven's action policy.
        features = features + .01 * torch.randn_like(features)
        hidden = features
        for layer in self.mlp_module[:-1]:
            hidden = hidden + .01 * torch.tanh(layer(hidden))
        return dict(
            action_mean=self.mlp_module[-1](hidden).reshape(*images.shape[:2], 11),
            predicted_obj_pos=self.obj_pred_mlp(features).reshape(*images.shape[:2], 3),
        )


def optimizer_for(policy):
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=.0002, weight_decay=0.)


def update(policy, optimizer, scheduler=None):
    # Every accepted update genuinely advances all three CPU global RNGs.
    offset = random.random() + float(np.random.rand())
    images = torch.linspace(-1, 1, 32 * 81 * 3).reshape(32, 81, 1, 1, 3) + offset
    actor = torch.zeros(32, 81, 1)
    target = torch.full((32, 81, 11), .2)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss = worker.backward_global_mse_in_environment_microbatches(
        policy, images, actor, target, microbatch_environments=32,
        full_environment_batch_per_timestep=True, activation_checkpointing=True,
        zero_auxiliary_object_loss=True,
    )
    parameters = [p for p in policy.parameters() if p.requires_grad]
    assert len(parameters) == 78
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
    assert any(bool(torch.count_nonzero(p.grad)) for p in parameters)
    torch.nn.utils.clip_grad_norm_(parameters, 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss


def assert_tensor_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def snapshot(policy, optimizer, scheduler, train_generator, loss):
    return dict(policy=deepcopy(policy.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                scheduler=deepcopy(scheduler.state_dict()),
                rng=rng.capture_rng(train_generator), loss=loss)


def test_four_real_worker_publications_resume_all_78_slots_and_rng(tmp_path):
    # Reuse the existing admitted metadata fixture; replace its tensors/state
    # with two actual updates of this 78-slot CPU policy. No live asset is read.
    base, _, _ = bc_parent()
    random.seed(21)
    np.random.seed(22)
    torch.random.default_generator.manual_seed(23)
    policy = WorkerStudent()
    optimizer = optimizer_for(policy)
    for _ in range(2):
        update(policy, optimizer)
    train_generator = torch.Generator(device='cpu').manual_seed(24).get_state()
    base.update(policy_state_dict=deepcopy(policy.state_dict()),
                optimizer_state_dict=deepcopy(optimizer.state_dict()),
                rng_state=rng.capture_rng(train_generator))
    initial_bn_count = int(policy.vision_module.num_batches_tracked)
    assert initial_bn_count == 2  # Recompute must not increment twice.
    assert len(optimizer.state) == 78
    path = tmp_path / 'fixture-bc-parent.pt'
    worker.atomic_torch_save(base, path)

    admission, job = initial()
    worker_sha = worker.file_sha256(Path(worker.__file__))
    admission['execution']['worker_sha256'] = worker_sha
    parent_sha = worker.file_sha256(path)
    old = admission['scope_admission']
    old['scope']['parent_checkpoint_sha256'] = parent_sha
    old['segment']['input_checkpoint_sha256'] = parent_sha
    old['scope']['segment_sha256'] = diagnostic.digest(old['segment'])
    old['scope_sha256'] = diagnostic.digest(old['scope'])
    job['parameters'][diagnostic.PARAMETER] = old

    # Reference remains in one process with no model reconstruction or test
    # phase between updates. Expected snapshots include all optimizer tensors.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    # Match the worker's one-time BC -> DAgger setup: it creates the scheduler
    # before loading BC's optimizer group (which has no scheduler initial_lr).
    optimizer.load_state_dict(deepcopy(base['optimizer_state_dict']))
    expected = []
    for _ in range(4):
        loss = update(policy, optimizer, scheduler)
        expected.append(snapshot(policy, optimizer, scheduler, train_generator, loss))

    previous_rng = rng.state_sha256(base['rng_state'])
    for iteration, reference in enumerate(expected, 1):
        base = torch.load(path, map_location='cpu', weights_only=False)
        job['input_checkpoint_sha256'] = worker.file_sha256(path)
        job['parameters']['iteration_id'] = iteration
        job['job_id'] = f'cpu-fixture-update-{iteration}'
        worker.validate_mse_parent_checkpoint(base, requested_learning_rate=.0002,
                                              diagnostic_request=job)
        resumed = WorkerStudent()
        resumed.load_state_dict(base['policy_state_dict'], strict=True)
        resumed.obj_pred_mlp.requires_grad_(False)  # Actual build_policy convention.
        worker.configure_diagnostic_optimizer_policy(resumed, base)
        resumed_optimizer = optimizer_for(resumed)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.)
        resumed_optimizer.load_state_dict(base['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in base:
            resumed_scheduler.load_state_dict(base['lr_scheduler_state_dict'])

        # Stand in for unrelated constructor/data/test/logging work, all before
        # restoration. The worker must not inherit these incidental draws.
        for _ in range(iteration + 2):
            random.random()
            np.random.rand(3)
            torch.rand(7)
        context = rng.prepare_restore(base, job, worker_sha)
        assert context['input_rng_sha256'] == previous_rng
        assert context['parent_global_step'] == iteration + 1
        assert context['parent_checkpoint_sha256'] == job['input_checkpoint_sha256']
        loss = update(resumed, resumed_optimizer, resumed_scheduler)
        assert loss == reference['loss']
        post_update = rng.capture_rng(train_generator)
        assert rng.state_sha256(post_update) == rng.state_sha256(reference['rng'])
        assert post_update['python'] != base['rng_state']['python']
        assert post_update['numpy'][2] != base['rng_state']['numpy'][2]
        assert not torch.equal(post_update['torch_cpu'], base['rng_state']['torch_cpu'])

        path = tmp_path / f'candidate-{iteration}.pt'
        receipt = worker.save_iteration_checkpoint(
            path, base, resumed, resumed_optimizer, resumed_scheduler,
            iteration_id=iteration, global_step=iteration + 2, train_mse=loss,
            request=job, rng_context=context,
        )
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['schema'] == diagnostic.CHECKPOINT_SCHEMA
        assert saved['production_checkpoint'] is saved['production_promoted'] is False
        assert saved[chain.PARAMETER] == admission
        assert saved['dagger_iteration_id'] == iteration
        assert saved['global_step'] == iteration + 2
        assert saved['parent_checkpoint_sha256'] == job['input_checkpoint_sha256']
        assert saved['learner_rng_receipt'] == receipt
        assert receipt['parent_checkpoint_sha256'] == job['input_checkpoint_sha256']
        assert receipt['global_step'] == iteration + 2
        assert receipt['worker_file_sha256'] == worker_sha
        assert receipt['input_rng_sha256'] == previous_rng
        previous_rng = receipt['output_rng_sha256']
        assert previous_rng == rng.state_sha256(saved['rng_state']) == rng.state_sha256(post_update)
        assert previous_rng == rng.state_sha256(rng.capture_rng(train_generator))
        assert torch.equal(saved['rng_state']['train_generator'], train_generator)
        assert receipt['train_generator_usage'] == 'preserved_unused_by_dagger'
        assert_tensor_state_equal(saved['policy_state_dict'], reference['policy'])
        assert int(saved['policy_state_dict']['vision_module.num_batches_tracked']) == initial_bn_count + iteration
        actual_optimizer = saved['optimizer_state_dict']
        assert actual_optimizer['param_groups'] == reference['optimizer']['param_groups']
        assert len(actual_optimizer['state']) == 78
        for ident, state in actual_optimizer['state'].items():
            assert_tensor_state_equal(state, reference['optimizer']['state'][ident])
            assert int(state['step']) == iteration + 2
        for parameter in resumed.obj_pred_mlp.parameters():
            for field in ('exp_avg', 'exp_avg_sq'):
                assert not torch.count_nonzero(resumed_optimizer.state[parameter][field])
        assert saved['lr_scheduler_state_dict'] == reference['scheduler']
        assert not path.with_suffix('.pt.part').exists()

    # Publication cannot quietly allow an unbounded fifth diagnostic update.
    job['parameters']['iteration_id'] = 5
    with pytest.raises(ValueError, match='four bound'):
        worker.validate_mse_parent_checkpoint(saved, diagnostic_request=job)


def call_lines(function, dotted_name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return sorted(node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted_name)


def only_call(function, dotted_name):
    lines = call_lines(function, dotted_name)
    assert len(lines) == 1, (dotted_name, lines)
    return lines[0]


def test_worker_orders_restore_after_setup_and_capture_before_logging():
    # Guard a few semantic call boundaries rather than file text/line numbers.
    restore = only_call(worker.main, 'learner_rng.prepare_restore')
    for setup in ('build_policy', 'optimizer.load_state_dict',
                  'scheduler.load_state_dict', 'load_batch', 'wandb.init'):
        assert only_call(worker.main, setup) < restore, setup
    assert restore < only_call(worker.main, 'backward_global_mse_in_environment_microbatches')
    save = only_call(worker.main, 'save_iteration_checkpoint')
    assert only_call(worker.main, 'optimizer.step') < save
    assert only_call(worker.main, 'scheduler.step') < save
    assert save < only_call(worker.main, 'wandb_run.log')
    assert only_call(worker.save_iteration_checkpoint, 'learner_rng.publish_rng') < only_call(
        worker.save_iteration_checkpoint, 'atomic_torch_save')
