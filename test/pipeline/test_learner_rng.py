"""CPU-only RNG/handoff checks; not GPU or native phase certification."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from pipeline.dagger.services import learner_rng as rng


@pytest.fixture(autouse=True)
def preserve_process_rng():
    before = rng.capture_rng(torch.Generator().manual_seed(7).get_state())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    random.setstate(before['python']); np.random.set_state(before['numpy'])
    torch.set_rng_state(before['torch_cpu']); torch.cuda.set_rng_state_all(before['torch_cuda'])
    torch.set_num_threads(threads)


def parent():
    random.seed(5); np.random.seed(6); torch.manual_seed(7)
    return dict(schema='roborender.bc32-lifecycle-qualification.v1', global_step=2,
                rng_state=rng.capture_rng(torch.Generator().manual_seed(8).get_state()))


def request():
    return dict(input_checkpoint_sha256='a'*64, parameters={rng.PARAMETER:rng.PROTOCOL})


def draw():
    return random.random(), np.random.rand(3), torch.rand(5)


def assert_draw(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1],right[1])
    assert torch.equal(left[2],right[2])


def test_hash_is_deterministic_and_does_not_advance_global_rng():
    base=parent(); before=draw()
    rng.prepare_restore(base,request(),'b'*64)
    digest=rng.state_sha256(base['rng_state'])
    assert digest==rng.state_sha256(deepcopy(base['rng_state']))
    assert_draw(before,draw())


def test_restore_ignores_incidental_constructor_or_test_phase_randomness():
    base=parent(); expected=draw()
    for _ in range(13): draw()
    context=rng.prepare_restore(base,request(),'b'*64)
    assert_draw(expected,draw())
    state,receipt=rng.publish_rng(base,request(),context,3,'b'*64)
    assert receipt['input_rng_sha256'] != receipt['output_rng_sha256']
    assert torch.equal(state['train_generator'],base['rng_state']['train_generator'])
    following=dict(schema='explicit-successor',global_step=3,rng_state=state,
                   learner_rng_receipt=receipt,parent_checkpoint_sha256='a'*64)
    next_draw=draw()
    for _ in range(9): draw()
    rng.prepare_restore(following,request(),'b'*64)
    assert_draw(next_draw,draw())


@pytest.mark.parametrize('mutation',[
    lambda x:x['rng_state'].pop('python'),
    lambda x:x['rng_state'].update(torch_cpu=torch.ones(3)),
    lambda x:x['rng_state'].update(train_generator=torch.ones(3,dtype=torch.uint8)),
    lambda x:x['rng_state'].update(torch_cuda=[torch.zeros(32,dtype=torch.uint8)]),
    lambda x:x.update(schema='roborender.dagger_student_checkpoint.v3'),
    lambda x:x.update(global_step=True),
])
def test_incomplete_legacy_or_different_device_state_is_not_exact_resume(mutation):
    base=parent();mutation(base)
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        rng.prepare_restore(base,request(),'b'*64)


def test_unknown_protocol_and_silent_downgrade_rejected():
    base=parent();req=request();req['parameters'][rng.PARAMETER]='typo'
    with pytest.raises(ValueError):rng.prepare_restore(base,req,'b'*64)
    req=request();context=rng.prepare_restore(base,req,'b'*64)
    state,receipt=rng.publish_rng(base,req,context,3,'b'*64)
    base.update(global_step=3,rng_state=state,learner_rng_receipt=receipt,parent_checkpoint_sha256='a'*64)
    with pytest.raises(ValueError,match='downgrade'):
        rng.prepare_restore(base,dict(parameters={}), 'b'*64)
    base['rng_state']['torch_cpu'][0]^=1
    with pytest.raises(ValueError,match='actual state'):
        rng.prepare_restore(base,request(),'b'*64)


@pytest.mark.parametrize('field,value',[
    ('worker_file_sha256','c'*64),('parent_global_step',9),('input_rng_sha256','d'*64),
    ('parent_checkpoint_sha256','e'*64),
])
def test_publication_rejects_changed_restored_context(field,value):
    base=parent();req=request();context=rng.prepare_restore(base,req,'b'*64)
    context[field]=value
    with pytest.raises(ValueError):rng.publish_rng(base,req,context,3,'b'*64)


def test_legacy_jobs_keep_original_rng_behavior_without_false_resume_claim():
    base=parent();req=dict(parameters={})
    assert rng.prepare_restore(base,req,'b'*64) is None
    assert rng.publish_rng(base,req,None,3,'b'*64) is None


class SeventyEightSlotPolicy(torch.nn.Module):
    """Tiny CPU network with the real78-slot/zero-auxiliary bookkeeping count."""
    def __init__(self):
        super().__init__()
        self.vision_module=torch.nn.BatchNorm1d(3)
        self.mlp_module=torch.nn.Sequential(*[torch.nn.Linear(3,3) for _ in range(34)])
        self.obj_pred_mlp=torch.nn.Sequential(*[torch.nn.Linear(3,3) for _ in range(4)])

    def forward(self,x):
        features=self.vision_module(x)
        return self.mlp_module(features),self.obj_pred_mlp(features)


def test_full32x81_78_slot_optimizer_bn_and_rng_match_checkpoint_return():
    base=parent();policy=SeventyEightSlotPolicy()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=2e-4,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1)
    assert len(list(policy.parameters()))==78
    def step(model,opt,sch):
        py, npv, tv=draw()
        x=torch.linspace(-1,1,32*81*3).reshape(32*81,3)+py+float(npv[0])+tv[0]
        model.train();opt.zero_grad(set_to_none=True)
        output,aux=model(x)
        loss=(output-.2).square().mean()+aux.square().mean()*0
        loss.backward();opt.step();sch.step()
        return loss.detach().clone()
    # Two genuine fixture BC updates initialize every optimizer slot.
    for _ in range(2):step(policy,optimizer,scheduler)
    base['rng_state']=rng.capture_rng(base['rng_state']['train_generator'])
    saved_policy=deepcopy(policy.state_dict());saved_optimizer=deepcopy(optimizer.state_dict())
    saved_scheduler=deepcopy(scheduler.state_dict())
    expected_loss=step(policy,optimizer,scheduler)
    expected_state=rng.capture_rng(base['rng_state']['train_generator'])
    # Model construction and simulated testing consume randomness; restoration
    # must occur after those incidental operations, not before construction.
    restored=SeventyEightSlotPolicy();restored.load_state_dict(saved_policy)
    resumed_optimizer=torch.optim.AdamW(restored.parameters(),lr=2e-4,weight_decay=0)
    resumed_scheduler=torch.optim.lr_scheduler.LambdaLR(resumed_optimizer,lambda _:1)
    resumed_optimizer.load_state_dict(saved_optimizer);resumed_scheduler.load_state_dict(saved_scheduler)
    for _ in range(11):draw()
    context=rng.prepare_restore(base,request(),'b'*64)
    assert torch.equal(step(restored,resumed_optimizer,resumed_scheduler),expected_loss)
    state,receipt=rng.publish_rng(base,request(),context,3,'b'*64)
    assert rng.state_sha256(state)==rng.state_sha256(expected_state)
    for name,value in policy.state_dict().items():assert torch.equal(value,restored.state_dict()[name])
    assert len(resumed_optimizer.state)==78
    for left,right in zip(optimizer.state.values(),resumed_optimizer.state.values()):
        for name in left:assert torch.equal(left[name],right[name])
    assert all(int(s['step'])==3 for s in resumed_optimizer.state.values())
    for p in restored.obj_pred_mlp.parameters():
        assert not torch.count_nonzero(resumed_optimizer.state[p]['exp_avg'])
    assert receipt['global_step']==3
