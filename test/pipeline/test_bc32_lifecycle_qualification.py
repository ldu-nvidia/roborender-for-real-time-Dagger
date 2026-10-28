"""Full-population CPU checks, never H200/native qualification."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from pipeline.bc_training import train_bc_h200 as bc
from pipeline.bc_training import qualify_raven_bc_lifecycle as base
from pipeline.bc_training import qualify_raven_bc32_lifecycle as smoke
from test.pipeline.test_bc_lifecycle_qualification import TinyStudent


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def batches():
    return [dict(vision=torch.full((32, 81, 6, 2, 3), 70 + i, dtype=torch.uint8), actor=torch.ones(32, 81, 113),
        target=torch.full((32, 81, 11), .2), object_target=torch.full((32, 81, 3), .1)) for i in range(2)]


def test_two_actual_cpu_updates_use_2592_bn_population_not_two_b16_microbatches():
    torch.manual_seed(42); policy = TinyStudent(); bc.configure_raven_offline_student(policy)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
    populations = []
    handle = policy.bn.register_forward_pre_hook(lambda module, args: populations.append(len(args[0])))
    result = base.run_updates(policy, optimizer, batches(), torch.device('cpu'),
        torch.zeros(1, 1, 1, 1, 3), torch.ones(1, 1, 1, 1, 3), chunk_batch_size=32)
    handle.remove()
    #81 no-grad B32 forwards + one full32*81 forward + its checkpoint replay.
    assert populations == ([32] * 81 + [2592, 2592]) * 2
    assert result['batch_norm_counter_deltas'] == {'bn': 164}
    assert len(result['updates']) == 2 and result['changed_parameter_names']
    assert all(int(state['step']) == 2 for state in optimizer.state.values())


def test_shape_and_unsupported_batch_fail_before_forward():
    policy = TinyStudent(); bc.configure_raven_offline_student(policy)
    batch = batches()[0]; tensors = [batch[k].float() for k in ('vision', 'actor', 'target', 'object_target')]
    with mock.patch.object(policy, 'init_rollout', side_effect=AssertionError('must fail before any forward')):
        with pytest.raises(ValueError, match='B16'): bc.backward_raven_offline_bc_mse(policy, *tensors)
        for invalid in (1, 8, 31, 64, True):
            with pytest.raises(ValueError): bc.backward_raven_offline_bc_mse(policy, *tensors, chunk_batch_size=invalid)


def test_production_defaults32_and_preserves_explicit16_historical_identity():
    assert bc.parse_args([]).chunk_batch_size == 32
    assert bc.parse_args(['--training-protocol', 'legacy_validation']).chunk_batch_size == 16
    b16 = bc.parse_args(['--chunk-batch-size', '16'])
    assert bc.student_lifecycle_batch_fields(b16.student_lifecycle, b16.chunk_batch_size)['schema'] == bc.RAVEN_REPLAY_BATCHING_CONTRACT
    args = bc.parse_args(['--chunk-batch-size', '32', '--max-epochs', '2'])
    assert args.student_lifecycle == bc.REPLAY_STUDENT_LIFECYCLE
    assert bc.validate_training_batch_population(2880, 32) == 90
    assert bc.student_lifecycle_batch_fields(args.student_lifecycle, 32)['schema'] == bc.RAVEN_REPLAY_B32_BATCHING_CONTRACT
    for arguments in (['--chunk-batch-size', '8'], ['--chunk-batch-size', '32', '--training-protocol', 'legacy_validation'],
                      ['--chunk-batch-size', '32', '--student-lifecycle', bc.DIRECT_STUDENT_LIFECYCLE]):
        with pytest.raises(ValueError): bc.parse_args(arguments)
    with pytest.raises(ValueError): bc.validate_training_batch_population(2879, 32)


def test_balanced_sample_contains_both_merged_roles_without_replacement(tmp_path):
    paths, items, roles = [], [], {}
    for role in ('train', 'validation'):
        for i in reversed(range(40)):
            path = tmp_path / f'{role}-{i:03d}.npz'; path.write_bytes(b'CPU fixture only')
            paths.append(path); items.append(dict(chunk_id=path.stem, rendered_npz=path.name)); roles[path.stem] = role
    selected = smoke.selected_rows(dict(items=items), paths, tmp_path, roles)
    assert len(selected) == 64 and len({r['chunk_id'] for r in selected}) == 64
    for update in range(2):
        subset = selected[32 * update:32 * (update + 1)]
        assert [r['original_role'] for r in subset] == ['train'] * 16 + ['validation'] * 16
        assert subset[0]['chunk_id'] == f'train-{update * 16:03d}'
        assert subset[16]['chunk_id'] == f'validation-{update * 16:03d}'


def test_initial_state_proof_rejects_random_substitute_and_verifies_all_buffers(tmp_path, monkeypatch):
    torch.manual_seed(42); policy = TinyStudent(); path = tmp_path / 'initial.pt'
    torch.save(dict(policy_state_dict=policy.state_dict()), path)
    monkeypatch.setattr(smoke, 'BOOTSTRAP_SHA', bc.sha256_file(path))
    proof = smoke.verify_initial_policy(policy, path)
    assert proof['initial_state_exact_saved_checkpoint'] and proof['initial_state_tensor_count'] == len(policy.state_dict())
    with torch.no_grad(): policy.bn.running_mean[0] += 1
    with pytest.raises(RuntimeError, match='reload mismatch'): smoke.verify_initial_policy(policy, path)


def test_default_dryrun_never_executes_or_contacts_cuda(monkeypatch):
    assert not smoke.parse_args([]).execute and smoke.parse_args([]).min_free_gpu_gib == 120
    monkeypatch.setattr(smoke, 'plan', lambda _: dict(execute=False))
    monkeypatch.setattr(smoke, 'execute', lambda *a: pytest.fail('execution forbidden'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA forbidden'))
    smoke.main([])


def test_actual_memory_guard_caps_only_this_process_and_keeps16gib():
    identity = dict(gpu_free_bytes_before=121 << 30, gpu_total_bytes=141 << 30,
                    host_available_bytes_before=64 << 30, cgroup_headroom_bytes_before=32 << 30)
    policy = smoke.memory_admission(identity)
    assert policy['allocator_cap_bytes'] == 105 << 30
    assert policy['actual_device_headroom_floor_bytes'] == 16 << 30
    assert not policy['retry_allowed'] and not policy['microbatch_fallback_allowed'] and not policy['renderer_eviction_allowed']
    for field, value in [('gpu_free_bytes_before', 119 << 30), ('host_available_bytes_before', 15 << 30),
                         ('cgroup_headroom_bytes_before', 7 << 30)]:
        with pytest.raises(RuntimeError): smoke.memory_admission(dict(identity, **{field: value}))


def test_production_default_b32_sorted_complete_epoch_resume_and_no_validation(tmp_path):
    for name in ('bootstrap.pt', 'policy.json', 'train.json'): (tmp_path / name).write_text('{}')
    paths = [tmp_path / f'chunk-{i:03d}.npz' for i in reversed(range(64))]
    chunks = [dict(vision=torch.full((81, 6, 2, 3), 70 + i % 20, dtype=torch.uint8),
        actor=torch.ones(81, 113), target=torch.full((81, 11), i / 64), object_target=torch.full((81, 3), i / 128)) for i in range(64)]
    spec = dict(input_contract=dict(actor_obs_dim=113, rgb_mean=[0, 0, 0], rgb_std=[1, 1, 1]), output_contract=dict(policy_action_dim=11))
    def run(name, resume=None, size=None):
        batch_args = [] if size is None else ['--chunk-batch-size', str(size)]
        args = bc.parse_args(['--max-epochs', '2', *batch_args, '--workers', '0', '--wandb-mode', 'disabled',
            '--dataset-root', str(tmp_path), '--train-split', str(tmp_path / 'train.json'),
            '--validation-split', str(tmp_path / 'missing-validation'), '--data-admission-json', str(tmp_path / 'missing-admission'),
            '--diagnostic-without-production-admission', '--bootstrap', str(tmp_path / 'bootstrap.pt'),
            '--policy-spec', str(tmp_path / 'policy.json'), '--output-root', str(tmp_path / 'outputs'), '--run-name', name])
        args.resume_checkpoint = resume; torch.manual_seed(123); policy = TinyStudent()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(bc, 'load_split', return_value=({}, paths)))
            stack.enter_context(mock.patch.object(bc, 'build_policy', return_value=(policy, spec)))
            dataset = stack.enter_context(mock.patch.object(bc, 'BCChunks', return_value=chunks))
            stack.enter_context(mock.patch.object(bc, 'validate', side_effect=AssertionError('heldout forbidden')))
            stack.enter_context(mock.patch.object(torch.cuda, 'get_device_name', return_value='CPU synthetic'))
            stack.enter_context(mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
            stack.enter_context(mock.patch.object(torch.cuda, 'set_rng_state_all'))
            bc.run_fixed_final(args, dict(status='fixture'), bc.fixed_final_contract(2, budget_explicit=True), device=torch.device('cpu'))
            assert dataset.call_args.args[0] == sorted(paths)
        return tmp_path / 'outputs' / name
    output = run('smoke-b32-complete')
    resumed = run('smoke-b32-resume', output / 'epoch_checkpoints/epoch_0001_step_000002.pt')
    a = torch.load(output / 'last.pt', weights_only=False); b = torch.load(resumed / 'last.pt', weights_only=False)
    assert a['training_batch_contract']['schema'] == bc.RAVEN_REPLAY_B32_BATCHING_CONTRACT
    assert a['training_batch_contract']['policy_forward_population'] == 2592 and a['global_step'] == 4
    base.assert_equal(a['policy_state_dict'], b['policy_state_dict']); base.assert_equal(a['optimizer_state_dict'], b['optimizer_state_dict'])
    base.assert_equal(a['rng_state'], b['rng_state'])
    with pytest.raises(ValueError, match='training_batch_contract'):
        run('smoke-b16-incompatible', output / 'epoch_checkpoints/epoch_0001_step_000002.pt', size=16)
