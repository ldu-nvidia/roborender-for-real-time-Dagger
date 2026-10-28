"""Pinned original randperm differential + real CPU checkpoint publication.

This is not a live Isaac/H200 trajectory or production qualification test.
"""
import ast
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from pipeline.dagger.services import learner_order as order, learner_rng as rng
from pipeline.dagger.services import raven_learner_worker as worker
from test.pipeline.test_dagger_coupled_functional_parity import pinned_tree
from test.pipeline.test_learner_rng_worker_integration import (
    cpu_rng_isolation, WorkerStudent, optimizer_for,
)


def request(count=32):
    # Lexicographic chunk-hash order deliberately differs from lane order.
    return dict(run_id='cpu-fixture', round_id=1, job_id='cpu-update',
        input_checkpoint_sha256='a'*64,
        train_chunk_ids=[f'chunk-{count-i:03d}' for i in range(count)],
        parameters={order.PARAMETER: order.PROTOCOL, rng.PARAMETER: rng.PROTOCOL,
            'raven_exact_full_batch': True, 'num_mini_batches': 1,
            'microbatch_environments': count, 'iteration_id': 1})


def original_indices(count):
    # Execute the actual nonrecurrent assignment, not a retyped oracle.
    tree = pinned_tree('trainer/ppo_trainer.py')
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == 'b_inds' for x in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'randperm']
    assert len(assignments) == 1
    namespace = dict(torch=torch, args=SimpleNamespace(local_batch_size=count), device=torch.device('cpu'))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<pinned Raven randperm>', 'exec'), namespace)
    return namespace['b_inds']


@pytest.mark.parametrize('count', [2, 16, 32])
def test_exact_original_permutation_rng_and_chronological_time(count):
    req = request(count)
    tensors = tuple(torch.arange(count*81*size).reshape(count,81,size).float() for size in (3,5,11))
    torch.manual_seed(1729)
    expected = original_indices(count)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(1729)
    actual, receipt = order.permute_batch(req, *tensors)
    assert receipt['environment_indices'] == expected.tolist()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert expected.tolist() != list(range(count))
    for result, original in zip(actual, tensors):
        assert torch.equal(result, original[expected])
    order.validate_receipt(receipt, req)


def test_checkpoint_records_actual_order_and_prevents_downgrade(tmp_path):
    req = request()
    policy = WorkerStudent()
    optimizer = optimizer_for(policy)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    base = dict(schema='roborender.bc32-lifecycle-qualification.v1', global_step=2,
        rng_state=rng.capture_rng(torch.Generator().manual_seed(3).get_state()))
    context = rng.prepare_restore(base, req, worker.file_sha256(worker.Path(worker.__file__)))
    batch = (torch.arange(32*81*3).reshape(32,81,1,1,3).float()/1000,
             torch.zeros(32,81,1), torch.full((32,81,11), .2))
    permuted, receipt = order.permute_batch(req, *batch)
    loss = worker.backward_global_mse_in_environment_microbatches(policy,*permuted,
        microbatch_environments=32, full_environment_batch_per_timestep=True,
        zero_auxiliary_object_loss=True)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    optimizer.step(); scheduler.step()
    path = tmp_path/'candidate.pt'
    worker.save_iteration_checkpoint(path,base,policy,optimizer,scheduler,
        iteration_id=1,global_step=3,train_mse=loss,request=req,rng_context=context,order_receipt=receipt)
    saved = torch.load(path, weights_only=False)
    assert saved['learner_order_receipt'] == receipt
    assert saved['dagger_train_chunk_ids'] == req['train_chunk_ids']
    order.validate_parent(saved,req)
    with pytest.raises(ValueError,match='downgrade'):
        order.validate_parent(saved,dict(parameters={}))
    changed = deepcopy(req); changed['train_chunk_ids'].reverse()
    with pytest.raises(ValueError,match='differs'):
        order.validate_receipt(receipt,changed)


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(learner_order_protocol='typo'),
    lambda p:p.pop(rng.PARAMETER),
    lambda p:p.update(replay_manifest={}),
    lambda p:p.update(microbatch_environments=4),
    lambda p:p.update(num_mini_batches=2),
    lambda p:p.update(raven_exact_full_batch=False),
])
def test_protocol_rejects_unmatched_training_contract(mutation):
    req=request(); mutation(req['parameters'])
    with pytest.raises(ValueError):order.enabled(req)


def test_no_silent_upgrade_of_old_dagger_parent():
    with pytest.raises(ValueError,match='fresh BC parent'):
        order.validate_parent(dict(schema='roborender.dagger_student_checkpoint.v3'),request())


def test_legacy_requests_do_not_consume_an_extra_random_draw():
    tensors=(torch.ones(2,81,3),)*3
    before=torch.get_rng_state().clone()
    actual,receipt=order.permute_batch(dict(parameters={}),*tensors)
    assert actual==tensors and receipt is None
    assert torch.equal(torch.get_rng_state(),before)
