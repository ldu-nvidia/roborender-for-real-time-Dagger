"""Independent CPU producer-to-consumer checks; no native/GPU qualification.

These use real fixed-final BC checkpoint publication, replacing only the
feature network and dataset with small CPU fixtures. They must not relabel a
diagnostic checkpoint as production or silently drop auxiliary AdamW slots.
"""
from contextlib import ExitStack
from copy import deepcopy
from unittest import mock

import pytest
import torch

from pipeline.bc_training import train_bc_h200 as bc
from pipeline.bc_training.qualify_raven_bc_lifecycle import assert_equal
from pipeline.dagger.services import raven_learner_worker as learner
from test.pipeline.test_dagger_parent_optimizer_admission import parent as legacy_parent


class SmallRaven(torch.nn.Module):
    """Use the real model's module names and BN/zero-auxiliary lifecycle."""
    is_recurrent = False
    max_rollout_history = 1
    running_mean_std = state_history_module = None
    use_temporal_attention = False
    clamp_noise_std = True
    max_noise_std = .001

    def __init__(self):
        super().__init__()
        self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
        self.vision_module = torch.nn.Sequential(torch.nn.SyncBatchNorm(3, momentum=.001))
        self.mlp_module = torch.nn.Sequential(torch.nn.Linear(4, 11))
        self.obj_pred_mlp = torch.nn.Sequential(torch.nn.Linear(3, 3))
        self.obj_pred_mlp.requires_grad_(False)

    def init_rollout(self):self.obs_dict_buffer = {}
    def clear_rollout(self):self.obs_dict_buffer = {}

    def forward(self, observation):
        vision, actor = observation['vision_obs'], observation['actor_obs']
        images = vision.reshape(-1, *vision.shape[-3:]).permute(0, 3, 1, 2)
        features = self.vision_module(images).mean((2, 3)).reshape(*vision.shape[:-3], 3)
        return dict(actions=self.mlp_module(torch.cat((features, actor[..., :1]), dim=-1)),
                    obj_pred=self.obj_pred_mlp(features))

    def action_mean_with_obj_pred(self, observation):
        output = self(observation)
        torch.distributions.Normal(output['actions'], output['actions'] * 0 + self.std).sample()
        return dict(action_mean=output['actions'], predicted_obj_pos=output['obj_pred'])

    def rollout_with_obj_pred(self, observation):return self.action_mean_with_obj_pred(observation)


@pytest.fixture(scope='module', autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module', params=((16, 32), (32, 64), (16, 18), (32, 34)))
def published_replay_parent(request, tmp_path_factory):
    size, count = request.param;root = tmp_path_factory.mktemp(f'actual-cpu-bc{size}-n{count}')
    for name in ('bootstrap.pt', 'policy.json', 'train.json'):(root / name).write_text('{}')
    paths = [root / f'chunk-{i:03d}.npz' for i in reversed(range(count))]
    chunks = [dict(vision=torch.full((81, 6, 2, 3), 60 + i % 30, dtype=torch.uint8),
        actor=torch.ones(81, 113), target=torch.full((81, 11), i / count),
        object_target=torch.full((81, 3), .2)) for i in range(count)]
    spec = dict(input_contract=dict(actor_obs_dim=113, rgb_mean=[0, 0, 0], rgb_std=[1, 1, 1]),
                output_contract=dict(policy_action_dim=11))
    args = bc.parse_args(['--max-epochs', '1', '--chunk-batch-size', str(size), '--workers', '0',
        '--wandb-mode', 'disabled', '--dataset-root', str(root), '--train-split', str(root / 'train.json'),
        '--validation-split', str(root / 'absent-validation'), '--data-admission-json', str(root / 'absent-admission'),
        '--diagnostic-without-production-admission', '--bootstrap', str(root / 'bootstrap.pt'),
        '--policy-spec', str(root / 'policy.json'), '--output-root', str(root / 'outputs'), '--run-name', f'smoke-bc{size}'])
    torch.manual_seed(123);model = SmallRaven()
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(bc, 'load_split', return_value=({}, paths)))
        stack.enter_context(mock.patch.object(bc, 'build_policy', return_value=(model, spec)))
        stack.enter_context(mock.patch.object(bc, 'BCChunks', return_value=chunks))
        stack.enter_context(mock.patch.object(bc, 'validate', side_effect=AssertionError('held-out pass forbidden')))
        stack.enter_context(mock.patch.object(torch.cuda, 'get_device_name', return_value='CPU fixture'))
        stack.enter_context(mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
        stack.enter_context(mock.patch.object(torch.cuda, 'set_rng_state_all'))
        stack.enter_context(mock.patch('builtins.print'))
        bc.run_fixed_final(args, dict(status='CPU fixture, not production data admission'),
            bc.fixed_final_contract(1, budget_explicit=True), device=torch.device('cpu'))
    output = args.output_root / args.run_name
    assert (output / 'training-complete.json').is_file()
    value = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    return size, value


def test_real_fixed_final_replay_output_is_accepted_without_diagnostic_capability(published_replay_parent):
    size, checkpoint = published_replay_parent
    before = deepcopy(checkpoint)
    assert checkpoint['schema'] == 'roborender.bc_student_checkpoint.v4'
    assert checkpoint['selection'] == 'last_completed_epoch'
    assert checkpoint['training_batch_contract']['policy_forward_population'] == size * 81
    assert checkpoint['global_step'] == 2
    assert len(checkpoint['optimizer_state_dict']['state']) == 6
    assert checkpoint['optimizer_contract']['trainable_parameters'] == [
        'vision_module.0.weight', 'vision_module.0.bias', 'mlp_module.0.weight', 'mlp_module.0.bias',
        'obj_pred_mlp.0.weight', 'obj_pred_mlp.0.bias']
    learner.validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=2e-4)
    assert_equal(before, checkpoint)


@pytest.mark.parametrize('mutation', [
    lambda value:value['optimizer_contract']['trainable_parameters'].pop(),
    lambda value:value['optimizer_contract']['trainable_parameters'].reverse(),
    lambda value:value['optimizer_state_dict']['param_groups'][0]['params'].pop(),
    lambda value:value['optimizer_state_dict']['state'][5]['exp_avg'].fill_(1.),
    lambda value:value['training_batch_contract'].update(student_lifecycle='direct_supervised_v1'),
    lambda value:value['training_batch_contract'].update(chunks_per_update=8),
    lambda value:value['training_batch_contract'].update(object_head_trainable_with_zero_gradients=False),
    lambda value:value.update(selection='minimum_validation_mse'),
    lambda value:value['training_cursor'].update(next_index=999),
    lambda value:value['training_sampling_contract'].update(cursor_commit='prefetched_batches'),
    lambda value:value['training_sampling_contract'].update(mode='experimental_shuffle'),
    lambda value:value['training_batch_contract'].update(drop_last=True),
])
def test_production_replay_identity_cannot_be_relabelled_or_drop_state(published_replay_parent, mutation):
    _, value = published_replay_parent;value = deepcopy(value);mutation(value)
    with pytest.raises(ValueError):learner.validate_mse_parent_checkpoint(value, requested_learning_rate=2e-4)


def test_historical_direct_bc_parent_remains_its_original_optimizer_identity():
    value, model, optimizer = legacy_parent()
    before = deepcopy(value)
    learner.validate_mse_parent_checkpoint(value, requested_learning_rate=2e-4)
    assert_equal(before, value)
    assert len(optimizer.state) == 4
    assert all(not parameter.requires_grad for parameter in model.obj_pred_mlp.parameters())


@pytest.mark.parametrize('checkpointing', [False, True])
def test_replay_optimizer_survives_updates_and_round_boundary_exactly(
        published_replay_parent, tmp_path, checkpointing):
    _, initial = published_replay_parent
    current = deepcopy(initial)
    reference = SmallRaven();reference.load_state_dict(current['policy_state_dict'], strict=True)
    learner.configure_replay_optimizer_policy(reference, current)
    reference_optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad],
                                           lr=2e-4, weight_decay=0.)
    reference_scheduler = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lambda _: 1.)
    # CPU load_state_dict may reuse state-tensor storage; isolate both copies
    # exactly as two independently deserialized checkpoint files would be.
    reference_optimizer.load_state_dict(deepcopy(current['optimizer_state_dict']))
    initial_auxiliary = deepcopy(reference.obj_pred_mlp.state_dict())
    original_bn = int(reference.vision_module[0].num_batches_tracked)
    lineage = None

    def update(policy, optimizer, scheduler):
        policy.train();optimizer.zero_grad(set_to_none=True)
        loss = learner.backward_global_mse_in_environment_microbatches(policy,
            torch.linspace(0., 1., 32 * 81 * 6 * 2 * 3).reshape(32, 81, 6, 2, 3),
            torch.ones(32, 81, 113), torch.full((32, 81, 11), .3),
            microbatch_environments=32, full_environment_batch_per_timestep=True,
            activation_checkpointing=checkpointing, zero_auxiliary_object_loss=True)
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        return loss

    # A production round boundary requires100 REAL accepted optimizer updates;
    # never relabel the two-update BC or a partial DAgger state as update100.
    sequence = [(1, iteration) for iteration in range(1, 101)] + [(2, 1)]
    for offset, (round_id, iteration_id) in enumerate(sequence, 1):
        learner.validate_mse_parent_checkpoint(current, requested_learning_rate=2e-4)
        assert current['training_cursor'] == initial['training_cursor']
        assert learner.uses_replay_optimizer(current)
        resumed = SmallRaven();resumed.load_state_dict(current['policy_state_dict'], strict=True)
        learner.configure_replay_optimizer_policy(resumed, current)
        optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        optimizer.load_state_dict(deepcopy(current['optimizer_state_dict']))
        if 'lr_scheduler_state_dict' in current:scheduler.load_state_dict(current['lr_scheduler_state_dict'])
        assert_equal(current['optimizer_state_dict'], optimizer.state_dict())
        rng = torch.get_rng_state()
        mse = update(reference, reference_optimizer, reference_scheduler)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        assert update(resumed, optimizer, scheduler) == mse
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_equal(reference.state_dict(), resumed.state_dict())
        assert_equal(reference_optimizer.state_dict(), optimizer.state_dict())
        assert int(resumed.vision_module[0].num_batches_tracked) == original_bn + offset
        assert_equal(initial_auxiliary, resumed.obj_pred_mlp.state_dict())
        assert all(int(state['step']) == 2 + offset for state in optimizer.state.values())
        for parameter in resumed.obj_pred_mlp.parameters():
            assert parameter.grad is not None and not torch.count_nonzero(parameter.grad)
            assert not torch.count_nonzero(optimizer.state[parameter]['exp_avg'])
            assert not torch.count_nonzero(optimizer.state[parameter]['exp_avg_sq'])
        job = dict(run_id='ordinary-replay-lineage', round_id=round_id, job_id=f'r{round_id}-i{iteration_id}',
            input_checkpoint_sha256='a' * 64 if offset == 1 else f'{offset:064x}',
            train_chunk_ids=[f'r{round_id}-i{iteration_id}-chunk-{i}' for i in range(32)],
            parameters=dict(metric_protocol='fixed100_final_success', iteration_id=iteration_id,
                raven_exact_full_batch=True, microbatch_environments=32))
        learner.validate_replay_request(current, job)
        output = tmp_path / f'candidate-{offset}.pt'
        learner.save_iteration_checkpoint(output, current, resumed, optimizer, scheduler,
            iteration_id=iteration_id, global_step=2 + offset, train_mse=mse, request=job)
        current = torch.load(output, map_location='cpu', weights_only=False)
        assert current['schema'] == learner.REPLAY_DAGGER_SCHEMA
        assert 'diagnostic_parent_admission' not in current
        assert learner.uses_replay_optimizer(current)
        assert current.get('replay_optimizer_lineage')
        if lineage is None:lineage = deepcopy(current['replay_optimizer_lineage'])
        else:assert current['replay_optimizer_lineage'] == lineage
        learner.validate_mse_parent_checkpoint(current, requested_learning_rate=2e-4)
        assert current['training_cursor'] == initial['training_cursor']
        bad = deepcopy(current)
        bad['replay_optimizer_lineage']['training_cursor']['next_index'] = 999
        with pytest.raises(ValueError):
            learner.validate_mse_parent_checkpoint(bad, requested_learning_rate=2e-4)
