"""CPU-only optimizer identity tests; no native or H200 qualification claim."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pipeline.bc_training import train_bc_h200 as bc
from pipeline.bc_training.qualify_raven_bc32_lifecycle import BOOTSTRAP_SHA, SPEC_SHA
from pipeline.dagger.services import bc32_diagnostic_parent as contract
from pipeline.dagger.services import raven_learner_worker as worker
from pipeline.dagger.rtx.code.src.dagger_rtx.continuous_diagnostic import SCHEMA, ACK
from pipeline.dagger.rtx.code.src.dagger_rtx.contracts import canonical_json


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
        self.vision_module = torch.nn.Linear(2, 4)
        self.mlp_module = torch.nn.Linear(4, 11)
        self.obj_pred_mlp = torch.nn.Linear(4, 3)
        self.is_recurrent = False
        self.max_rollout_history = 1
        self.running_mean_std = self.state_history_module = None
        self.clamp_noise_std = True
        self.max_noise_std = .001
    def init_rollout(self): pass
    def clear_rollout(self): pass
    def rollout_with_obj_pred(self, value): return self.action_mean_with_obj_pred(value)
    def action_mean_with_obj_pred(self, value):
        features = self.vision_module(value['vision_obs'].squeeze(2).squeeze(2))
        return dict(action_mean=self.mlp_module(features), predicted_obj_pos=self.obj_pred_mlp(features))


def admission():
    config = dict(source_sha256='c'*64, student=dict(initial_checkpoint_sha256='a'*64))
    segment = dict(schema='roborender.dagger-continuous-round.v2', run_id='diagnostic', round_id=1,
        window_count=100, start_iteration_index=0, qualification_after_updates=4,
        coverage_plan_sha256='d'*64, mse_protocol_file_sha256='e'*64, input_checkpoint_sha256='a'*64)
    scope = dict(schema=SCHEMA, purpose='diagnostic_only', run_id='diagnostic', round_id=1,
        source_sha256='c'*64, config_sha256=contract.digest(config), segment_sha256=contract.digest(segment),
        coverage_plan_sha256='d'*64, mse_protocol_file_sha256='e'*64, parent_checkpoint_sha256='a'*64,
        optimizer_updates=4, environments_per_actor=16, full_batch_environments=32, rollout_steps=81,
        checkpoint_selection='none_diagnostic_candidate_only',
        terminal_action='finish_diagnostic_after_genuine_prefix_qualification', acknowledgement=ACK,
        owners=[dict(physical_gpu=i, native_pid=100+i, worker_instance_id=str(i+1)*32,
            command_id=f'diagnostic-g{i}', gpu_uuid=f'GPU-{i}') for i in (0, 1)])
    return dict(schema=contract.ADMISSION_SCHEMA, scope=scope, scope_sha256=contract.digest(scope), segment=segment), config


def request(iteration=1, digest='a'*64):
    return dict(run_id='diagnostic', round_id=1, job_id=f'diagnostic-{iteration}',
        input_checkpoint_sha256=digest, train_chunk_ids=[f'chunk-{i}' for i in range(32)],
        parameters=dict(iteration_id=iteration, metric_protocol='fixed100_final_success',
            raven_exact_full_batch=True, microbatch_environments=32,
            **{contract.PARAMETER: admission()[0]}))


def train(model, optimizer, *, checkpointing=False):
    optimizer.zero_grad(set_to_none=True)
    loss = worker.backward_global_mse_in_environment_microbatches(model,
        torch.ones(32, 81, 1, 1, 2), torch.ones(32, 81, 1), torch.ones(32, 81, 11),
        microbatch_environments=32, full_environment_batch_per_timestep=True,
        activation_checkpointing=checkpointing, zero_auxiliary_object_loss=True)
    optimizer.step()
    return loss


def bc_parent():
    torch.manual_seed(9)
    model = Student()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0002, weight_decay=0.)
    for _ in range(2): train(model, optimizer)
    value = dict(schema=contract.BC_SCHEMA, epoch=0, global_step=2,
        qualification_optimizer_steps=2, policy_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), policy_spec={},
        training_protocol=dict(schema=contract.BC_SCHEMA, optimizer_updates=2),
        training_batch_contract=dict(bc.student_lifecycle_batch_fields(bc.REPLAY_STUDENT_LIFECYCLE,32),
            chunks_per_update=32, frames_per_chunk=81, policy_forward_population=2592,
            microbatching=False, gradient_accumulation=False, training_chunk_count=64, updates_per_epoch=None),
        selection='diagnostic_only_not_completed_epoch', validation_enabled=False, completed_epoch=False,
        production_checkpoint=False, production_promoted=False, bootstrap_sha256=BOOTSTRAP_SHA,
        policy_spec_sha256=SPEC_SHA, source_identity={'pipeline/bc_training/qualify_raven_bc32_lifecycle.py':'b'*64},
        initial_student_proof=dict(initial_state_exact_saved_checkpoint=True, initial_checkpoint_sha256=BOOTSTRAP_SHA,
            initial_state_sha256='b'*64, initial_state_tensor_count=7),
        rng_state={key:None for key in ('python','numpy','torch_cpu','torch_cuda','train_generator')})
    return deepcopy(value), model, optimizer


@pytest.mark.parametrize('checkpointing',[False,True])
def test_exact_aux_inclusive_adamw_restoration_and_four_bounded_successors(tmp_path, checkpointing):
    parent, uninterrupted, reference_opt = bc_parent()
    original_aux = deepcopy(uninterrupted.obj_pred_mlp.state_dict())
    reference_sched = torch.optim.lr_scheduler.LambdaLR(reference_opt,lambda _:1.)
    current = parent
    for iteration in range(1,5):
        job = request(iteration, 'a'*64 if iteration == 1 else 'f'*64)
        worker.validate_mse_parent_checkpoint(current, requested_learning_rate=.0002, diagnostic_request=job)
        resumed = Student(); resumed.obj_pred_mlp.requires_grad_(False)
        resumed.load_state_dict(current['policy_state_dict'])
        worker.configure_diagnostic_optimizer_policy(resumed,current)
        optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.0002,weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        optimizer.load_state_dict(current['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in current: scheduler.load_state_dict(current['lr_scheduler_state_dict'])
        assert len(optimizer.state) == len(reference_opt.state) == 6
        before = deepcopy(optimizer.state_dict())
        for ident in before['state']:
            for key in before['state'][ident]:
                assert torch.equal(before['state'][ident][key], current['optimizer_state_dict']['state'][ident][key])
        mse = train(resumed, optimizer, checkpointing=checkpointing); scheduler.step()
        train(uninterrupted,reference_opt,checkpointing=checkpointing);reference_sched.step()
        assert all(torch.equal(v,uninterrupted.state_dict()[k]) for k,v in resumed.state_dict().items())
        assert all(torch.equal(v, original_aux[k]) for k,v in resumed.obj_pred_mlp.state_dict().items())
        for parameter in resumed.obj_pred_mlp.parameters():
            assert int(optimizer.state[parameter]['step']) == iteration+2
            assert not torch.count_nonzero(optimizer.state[parameter]['exp_avg'])
            assert not torch.count_nonzero(optimizer.state[parameter]['exp_avg_sq'])
        path=tmp_path/f'candidate-{iteration}.pt'
        worker.save_iteration_checkpoint(path,current,resumed,optimizer,scheduler,
            iteration_id=iteration,global_step=iteration+2,train_mse=mse,request=job)
        current=torch.load(path,weights_only=False)
        assert current['schema']==contract.CHECKPOINT_SCHEMA
        with pytest.raises(ValueError): worker.validate_mse_parent_checkpoint(current)
    with pytest.raises(ValueError,match='four bound'):
        worker.validate_mse_parent_checkpoint(current,diagnostic_request=request(5,'f'*64))


@pytest.mark.parametrize('mutate',[
    lambda c:c.update(schema='roborender.bc_student_checkpoint.v3'),
    lambda c:c.update(global_step=1), lambda c:c.update(epoch=1),
    lambda c:c.update(production_promoted=True), lambda c:c.update(validation_enabled=True),
    lambda c:c.update(qualification_optimizer_steps=3),
    lambda c:c['training_batch_contract'].update(chunks_per_update=16),
    lambda c:c.update(bootstrap_sha256='0'*64), lambda c:c.update(source_identity={}),
    lambda c:c.update(rng_state={}), lambda c:c.update(lr_scheduler_state_dict={}),
    lambda c:c['optimizer_state_dict']['param_groups'][0]['params'].pop(),
    lambda c:c['optimizer_state_dict']['state'].pop(5),
    lambda c:c['optimizer_state_dict']['state'][5].update(step=torch.tensor(1.)),
    lambda c:c['optimizer_state_dict']['state'][5]['exp_avg'].fill_(1.),
    lambda c:c['optimizer_state_dict']['state'][0]['exp_avg'].fill_(float('nan')),
])
def test_rejects_silent_migration_or_relabelled_parent(mutate):
    value,_,_=bc_parent();mutate(value)
    with pytest.raises(ValueError):worker.validate_mse_parent_checkpoint(value,diagnostic_request=request())


@pytest.mark.parametrize('mutate',[
    lambda j:j.update(round_id=2),lambda j:j.update(run_id='another'),
    lambda j:j.update(input_checkpoint_sha256='f'*64),
    lambda j:j['parameters'].update(iteration_id=5),
    lambda j:j['parameters'].update(microbatch_environments=16),
    lambda j:j['parameters'].update(replay_manifest={}),
    lambda j:j['parameters'].update(metric_protocol='legacy'),
    lambda j:j['parameters'][contract.PARAMETER].update(scope_sha256='f'*64),
    lambda j:j['parameters'][contract.PARAMETER]['scope'].update(optimizer_updates=100),
])
def test_request_scope_cannot_expand_or_change_experiment(mutate):
    job=request();mutate(job)
    with pytest.raises(ValueError):contract.validate_admission(job['parameters'][contract.PARAMETER],job)


def test_controller_requires_actual_immutable_scope_intent_and_config(tmp_path):
    value,config=admission();scope=tmp_path/'diagnostic-scope.json'
    scope.write_bytes(canonical_json(value['scope']))
    intent=dict(schema='roborender.continuous-diagnostic-intent.v1', segment=value['segment'],
        diagnostic_scope=dict(path=str(scope),raw_sha256=value['scope_sha256']))
    (tmp_path/'intent.json').write_text(json.dumps(intent))
    assert contract.controller_admission(scope,config,request())==value
    config['source_sha256']='f'*64
    with pytest.raises(ValueError):contract.controller_admission(scope,config,request())


def test_bc32_requires_explicit_scope_not_default_parent_admission():
    value,_,_=bc_parent()
    with pytest.raises(ValueError,match='supported complete'):
        worker.validate_mse_parent_checkpoint(value)


def test_trainable_order_cannot_silently_reorder_optimizer():
    value,_,_=bc_parent();model=Student()
    value['policy_state_dict']=dict(reversed(list(value['policy_state_dict'].items())))
    with pytest.raises(ValueError,match='trainable order'):
        worker.configure_diagnostic_optimizer_policy(model,value)


def test_learner_service_rejects_fifth_job_before_ledger_or_process(tmp_path):
    from types import SimpleNamespace
    from pipeline.dagger.services.server import Handler
    from pipeline.dagger.services.contracts import ContractError
    handler=object.__new__(Handler)
    handler.server=SimpleNamespace(worker=Path(__file__))  # No store: any ledger access fails this test.
    handler._read_json=lambda:request(5,'f'*64)
    with pytest.raises(ContractError,match='four bound'):
        handler._learner('POST','/v1/jobs')
