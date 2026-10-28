"""Pinned Raven differential state tests; no Isaac or native certification."""
from test.deployment.source_paths import REPO_ROOT
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.dagger.rtx.code.src.dagger_rtx import training_curriculum as module

ROOT = REPO_ROOT
REFERENCE = ROOT / 'runtime/raven_gr00t_collection_94d7985/gr00t/rl/envs'
HASHES = {
    'base_task/base_task.py': 'f1dd624a57d7f3e8ba1a5ebb2af1977512a78615a3202b537ba2e2e58bcdab7d',
    'legged_base_task/legged_robot_base.py': '78d0a736125d09c9ebe650e871e87338bdcb463941dca949d4a347edb70dde5c',
    'base_task/staged_task_base.py': 'c67503aad0a9ebf9c66c6dcc629f5ea0872b58451fe3d3a098b83faf1d949e26',
}


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def original_class():
    classes = []
    for path, name, base, methods in [
        ('base_task/base_task.py', 'BaseTask', None, {'get_env_state_dict', 'load_env_state_dict'}),
        ('legged_base_task/legged_robot_base.py', 'LeggedRobotBase', 'BaseTask', {'get_env_state_dict'}),
        ('base_task/staged_task_base.py', 'StagedTaskBase', 'LeggedRobotBase',
         {'_reset_tasks_callback', '_update_termination_curriculum'}),
    ]:
        raw = (REFERENCE / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == HASHES[path]
        cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == name)
        body = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in body} == methods
        for node in body:
            node.decorator_list = []
        if base is None:
            body.extend(ast.parse('def _reset_tasks_callback(self, env_ids): pass').body)
        classes.append(ast.ClassDef(name=name, bases=[ast.Name(id=base, ctx=ast.Load())] if base else [],
                                    keywords=[], body=body, decorator_list=[]))
    scope = dict(torch=torch, np=np, logger=SimpleNamespace(warning=lambda *_: None),
                 to_torch=lambda v, device: torch.tensor(v, device=device))
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
                 '<hash-pinned-raven-curriculum>', 'exec'), scope)
    return scope['StagedTaskBase']


def env(count=16):
    result = original_class()()
    result.config = Config(termination_curriculum=Config(min_level=.1, max_level=1.,
        level_down_ave_goal_reached_rate=.6, level_up_ave_goal_reached_rate=.96, degree=-.00008))
    result.num_envs = count
    result.device = torch.device('cpu')
    result.max_episode_length = 400
    result.is_evaluating = False
    result.termination_level = torch.tensor(1.)
    result.average_episode_length = 0.0
    result.log_dict = dict(termination_level=result.termination_level, average_goal_reached=0.,
                          average_stage_reached=0., average_last_stage_goal_reached=0.)
    for name in ('last_max_stage_buf', 'current_max_stage_buf'):
        setattr(result, name, torch.zeros(count, dtype=torch.long))
    for name in ('last_completed_task_buf', 'current_completed_task_buf',
                 'last_last_stage_completed_task_buf', 'current_last_stage_completed_task_buf'):
        setattr(result, name, torch.zeros(count, dtype=torch.bool))
    result.physics = torch.arange(count, dtype=torch.float32)
    return result


def trained():
    result = env()
    result.termination_level = torch.tensor(.72)
    result.average_episode_length = torch.tensor(102.75)
    result.log_dict.update(termination_level=result.termination_level,
                          average_episode_length=result.average_episode_length)
    return result


def binding(gpu=0, production=False):
    return dict(run_id='training-previous', round_id=1, actor_id=f'rtx-g{gpu}', physical_gpu=gpu,
        gpu_uuid=f'GPU-{gpu}', native_pid=100+gpu, worker_instance_id=str(gpu+1)*32,
        source_sha256='c'*64, command_id=f'training-previous-g{gpu}', num_envs=16,
        completed_updates=100 if production else 4,
        boundary='production_update100' if production else 'diagnostic_update4',
        final_checkpoint_sha256='d'*64, motion_assignment_sha256='e'*64, config_sha256='f'*64,
        diagnostic_scope_sha256=None if production else 'a'*64)


def exported(production=False):
    return module.export_training_curriculum(trained(), binding(production=production), phase='training')


def owned_predecessor(production=False):
    payload = exported(production)
    receipt = dict(complete=True, training_curriculum=payload, training_curriculum_sha256=module.digest(payload))
    wrapped = dict(training_curriculum=payload, training_curriculum_sha256=module.digest(payload),
                   native_terminal_receipt_sha256=module.digest(receipt))
    b = payload['binding']
    owner = {key: b[key] for key in ('native_pid','worker_instance_id','gpu_uuid','source_sha256')}
    owner['commands'] = {b['command_id']: dict(status='COMPLETE', receipt=receipt,
        command=dict(operation='collect_production_segment' if production else 'probe_continuous_training'))}
    return module.build_restore_request(wrapped), owner, payload


def test_original_getter_and_loader_match_through_effective_first_reset():
    prior = trained()
    # Prove export uses the getter's saved log value, not a newer direct attr.
    prior.termination_level = torch.tensor(.91)
    payload = module.export_training_curriculum(prior, binding(), phase='training')
    assert payload['fields']['termination_level']['value'] == prior.log_dict['termination_level'].item()
    original, current = env(), env()
    physical = current.physics.clone()
    rng = torch.get_rng_state().clone()
    original.load_env_state_dict(prior.get_env_state_dict())
    initialized = module.restore_training_curriculum(current, payload,
        expected_binding=binding(), phase='training')
    for name in payload['fields']:
        assert torch.equal(getattr(original, name), getattr(current, name))
        assert getattr(original, name).dtype == getattr(current, name).dtype
    # Original load leaves log_dict cold until the initial reset callback.
    assert original.log_dict['termination_level'].item() == current.log_dict['termination_level'].item() == 1.
    for target in (original, current):
        target._reset_tasks_callback(torch.arange(16))
    assert torch.equal(original.termination_level, current.termination_level)
    assert current.termination_level.item() > payload['fields']['termination_level']['value']
    assert torch.equal(original.log_dict['termination_level'], current.log_dict['termination_level'])
    assert torch.equal(current.physics, physical) and torch.equal(torch.get_rng_state(), rng)
    assert not current.last_completed_task_buf.any()
    assert initialized['physical_or_rng_state_restored'] is False
    assert set(payload['fields']) == {'termination_level', 'average_episode_length'}


def test_conditional_original_scalar_intersection_and_loader_types():
    prior = trained()
    prior.config.update(rewards=Config(reward_min_penalty_scale=.1, reward_max_penalty_scale=1.,
        reward_limit=Config(reward_limits_curriculum=Config(
            soft_dof_pos_min_limit=.1, soft_dof_pos_max_limit=1., soft_dof_vel_min_limit=.1,
            soft_dof_vel_max_limit=1., soft_torque_min_limit=.1, soft_torque_max_limit=1.))),
        obs=Config(noise_value_min=0., noise_value_max=1.),
        reset_from_dataset=Config(reset_from_teleop_curriculum=True))
    for flag in ('use_reward_penalty_curriculum','use_reward_limits_dof_pos_curriculum',
                 'use_reward_limits_dof_vel_curriculum','use_reward_limits_torque_curriculum','add_noise_currculum'):
        setattr(prior, flag, True)
    for name in module.ALLOWLIST - {'termination_level','average_episode_length'}:
        value = torch.tensor([.5]) if name == 'reset_from_teleop_curri_ratio' else torch.tensor(.5)
        setattr(prior, name, value if name != 'reset_from_teleop_curri_ratio' else .5)
        prior.log_dict[name] = value
    payload = module.export_training_curriculum(prior, binding(), phase='training')
    assert set(payload['fields']) == module.ALLOWLIST
    reference, current = deepcopy(prior), deepcopy(prior)
    reference.load_env_state_dict(prior.get_env_state_dict())
    module.restore_training_curriculum(current, payload, expected_binding=binding(), phase='training')
    for name in module.ALLOWLIST:
        assert torch.equal(getattr(reference, name), getattr(current, name))
    # The original mismatched Homie log key is ignored, never mapped to a new attribute.
    prior.termination_large_homie_linvel_command_threshold = torch.tensor(.7)
    prior.log_dict['large_homie_linvel_command_threshold'] = torch.tensor(.7)
    assert 'large_homie_linvel_command_threshold' not in module.export_training_curriculum(
        prior, binding(), phase='training')['fields']


@pytest.mark.parametrize('fault', ['unknown', 'missing', 'nan', 'infinity', 'bool', 'vector', 'dtype', 'low', 'high',
                                  'test', 'owner', 'checkpoint', 'global', 'bounds'])
def test_restore_rejects_before_any_native_mutation(fault):
    value, target = exported(), env()
    row = value['fields']['termination_level']
    if fault == 'unknown': value['fields']['physics'] = row
    elif fault == 'missing': del value['fields']['termination_level']
    elif fault == 'nan': row['value'] = float('nan')
    elif fault == 'infinity': row['value'] = float('inf')
    elif fault == 'bool': row['value'] = True
    elif fault == 'vector': row['shape'] = [16]
    elif fault == 'dtype': row['dtype'] = 'float64'
    elif fault == 'low': row['value'] = 0.
    elif fault == 'high': row['value'] = 2.
    elif fault == 'test': value['phase'] = 'test'
    elif fault == 'owner': value['binding']['native_pid'] += 1
    elif fault == 'checkpoint': value['binding']['final_checkpoint_sha256'] = 'b'*64
    elif fault == 'global': value['population'] = 'global_b32'
    elif fault == 'bounds': value['bounds']['termination_level'][0] = 0.
    original = deepcopy(vars(target))
    with pytest.raises(ValueError):
        module.restore_training_curriculum(target, value, expected_binding=binding(), phase='training')
    assert torch.equal(target.termination_level, original['termination_level'])
    assert torch.equal(target.physics, original['physics'])
    assert target.average_episode_length == 0.


def test_evaluation_cannot_export_or_restore_and_unknown_saved_attribute_fails():
    value = exported()
    for phase in ('test', 'evaluation'):
        with pytest.raises(ValueError): module.restore_training_curriculum(env(), value, expected_binding=binding(), phase=phase)
        with pytest.raises(ValueError): module.export_training_curriculum(trained(), binding(), phase=phase)
    target = trained(); target.is_evaluating = True
    with pytest.raises(ValueError): module.export_training_curriculum(target, binding(), phase='training')
    target = trained(); target.unreviewed_curriculum = 1.; target.log_dict['unreviewed_curriculum'] = 1.
    with pytest.raises(ValueError, match='intersection'): module.export_training_curriculum(target, binding(), phase='training')


def test_bootstrap_requires_actual_cold_configuration_not_a_prior_phase():
    initialized = module.bootstrap_training_curriculum(env(), phase='training')
    assert initialized['before_initial_reset']['termination_level']['value'] == 1.
    with pytest.raises(ValueError, match='cold configuration'):
        module.bootstrap_training_curriculum(trained(), phase='training')
    with pytest.raises(ValueError): module.bootstrap_training_curriculum(env(), phase='test')


@pytest.mark.parametrize('production', [False, True])
def test_owned_terminal_checkpoint_and_export_are_bound_without_digest_cycle(production):
    request, owner, payload = owned_predecessor(production)
    assert module.resolve_predecessor(request, owner, parent_checkpoint_sha256='d'*64,
        physical_gpu=0, source_sha256='c'*64) == payload
    assert 'terminal_receipt_sha256' not in payload['binding']
    assert module.resolve_predecessor(module.bootstrap_request(), owner,
        parent_checkpoint_sha256='b'*64, physical_gpu=0, source_sha256='c'*64) is None


@pytest.mark.parametrize('fault', ['owner','source','checkpoint','actor','terminal','test','partial','export'])
def test_cannot_import_test_state_or_another_native_owner(fault):
    request, owner, _ = owned_predecessor()
    options = dict(parent_checkpoint_sha256='d'*64, physical_gpu=0, source_sha256='c'*64)
    record = next(iter(owner['commands'].values()))
    if fault == 'owner': owner['native_pid'] += 1
    elif fault == 'source': options['source_sha256'] = 'a'*64
    elif fault == 'checkpoint': options['parent_checkpoint_sha256'] = 'a'*64
    elif fault == 'actor': options['physical_gpu'] = 1
    elif fault == 'terminal': request['predecessor']['native_terminal_receipt_sha256'] = 'a'*64
    elif fault == 'test': record['command']['operation'] = 'evaluate_scaled_rendered_panel'
    elif fault == 'partial': record['status'] = 'RUNNING'
    elif fault == 'export': record['receipt']['training_curriculum']['fields']['termination_level']['value'] = .5
    with pytest.raises(ValueError): module.resolve_predecessor(request, owner, **options)


def test_initial_reset_observer_records_existing_call_without_added_steps_or_rng():
    target = env()
    value = exported()
    calls = []
    target.reset_all = lambda: (calls.append('reset'), target._reset_tasks_callback(torch.arange(16)), {'obs': 'same'})[-1]
    receipt = module.restore_training_curriculum(target, value, expected_binding=binding(), phase='training')
    previous = target.reset_all
    with module.observe_initial_reset(target, receipt):
        assert target.reset_all() == {'obs': 'same'}
    assert target.reset_all is previous and calls == ['reset']
    assert receipt['initial_reset_calls'] == 1
    assert receipt['after_initial_reset']['termination_level']['value'] > receipt['before_initial_reset']['termination_level']['value']


def test_per_actor_curriculum_is_not_global_b32_even_with_same_outcomes():
    first, second, global_env = env(16), env(16), env(32)
    for target in (first, second, global_env): target.termination_level.fill_(.8)
    first.log_dict['average_goal_reached'] = torch.tensor(15 / 16)
    second.log_dict['average_goal_reached'] = torch.tensor(1.)
    global_env.log_dict['average_goal_reached'] = torch.tensor(31 / 32)
    for target in (first, second, global_env): target._update_termination_curriculum()
    assert first.termination_level.item() == torch.tensor(.8).item()
    assert second.termination_level.item() == global_env.termination_level.item()
    assert ((first.termination_level + second.termination_level) / 2).item() != global_env.termination_level.item()


def test_original_checkpoint_load_precedes_train_reset_and_random_clocks():
    source = ROOT / 'runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/ppo_trainer.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61'
    tree = ast.parse(source.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TRLPPOTrainer')
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    train = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    assert 'self.load_checkpoint(checkpoint)' in ast.unparse(constructor)
    body = ast.unparse(train)
    assert body.index('self.env.reset_all()') < body.index('torch.randint_like')
    assert 'self.config.init_at_random_ep_len = False' in (ROOT /
        'pipeline/deployment/rtx_actor/isaac6_motion_evaluator.py').read_text()
