"""Native vector selection, retry safety and pinned policy-step comparison.

CPU/tiny-policy/transport fixtures only; no live physics qualification claim.
"""
import json

import pytest
import torch

from dagger_rtx import behavior_selection as behavior
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.contracts import ContractError
from dagger_rtx.http_transport import TransientTransportError
from test.rtx.test_lean_batch_v3 import _make_owner, _sharded_client
from test.pipeline.test_dagger_coupled_functional_parity import (
    cpu_only_rng, original_classes, reference_runner, tiny_policy, BATCH,
)


@pytest.mark.parametrize('beta',[0.,.25,.5,.75,1.])
def test_native_draws_match_unmodified_original_policy_step_and_rng(beta):
    classes=original_classes(torch)  # REAL PyTorch; no selection-tape override.
    policy=tiny_policy().eval()
    policy.init_rollout()
    reference,wrapper=reference_runner(classes,policy)
    reference._dagger_beta=lambda:beta
    obs=reference.env.reset_all()
    torch.manual_seed(2718)
    before=torch.get_rng_state().clone()
    with torch.no_grad():
        expected=reference.policy_step(policy,obs,cur_dones=torch.zeros(BATCH,dtype=torch.bool))
    after=torch.get_rng_state().clone()
    torch.set_rng_state(before)
    receipt=behavior.sample_vector(BATCH,beta,'cpu')
    assert torch.equal(after,torch.get_rng_state())
    mask=torch.tensor(receipt['draws'])<beta
    # The mean is the student action (the Normal sample is discarded).
    student=torch.cat((policy.action_mean, expected['gt_actions'][:,11:]),dim=-1)
    actual=torch.where(mask[:,None],expected['gt_actions'],student)
    torch.testing.assert_close(actual,expected['actions'],rtol=0,atol=0)


def test_b16_four_shards_lost_response_never_redraws_or_resteps(tmp_path,tiny_profile):
    owner,backend,keys=_make_owner(tmp_path,tiny_profile,num_envs=16)
    client,remotes=_sharded_client(tmp_path/'client',tiny_profile,count=4)
    remotes[1].fail_prepare_after_effect_once=True
    calls=[]
    def sample(count,beta):
        calls.append((count,beta))
        return behavior.sample_vector(count,beta,'cpu')
    orchestrator=BatchStepOrchestrator(owner,client,tmp_path/'orchestrator',num_envs=16,
                                      behavior_sampler=sample)
    try:
        with pytest.raises(TransientTransportError):orchestrator.run_batch(keys)
        assert backend.batch_step_calls==0
        rng_after=torch.get_rng_state().clone()
        result=orchestrator.run_batch(keys)
        assert result['phase']=='COMPLETE' and backend.batch_step_calls==1
        assert calls==[(16,.75)]
        assert torch.equal(rng_after,torch.get_rng_state())
        assert [row['selection_draw'] for row in result['selections']]==result['behavior_selection_receipt']['draws']
        assert orchestrator.run_batch(keys)==result
        assert backend.batch_step_calls==1 and len(calls)==1
    finally:
        client.close()


def test_uncertain_rng_sample_is_not_replayed(tmp_path,tiny_profile):
    owner,backend,keys=_make_owner(tmp_path,tiny_profile)
    client,_=_sharded_client(tmp_path/'client',tiny_profile)
    def crash(count,beta):
        behavior.sample_vector(count,beta,'cpu')
        raise RuntimeError('simulated failure before receipt publication')
    orchestrator=BatchStepOrchestrator(owner,client,tmp_path/'orchestrator',num_envs=2,behavior_sampler=crash)
    try:
        with pytest.raises(RuntimeError,match='simulated failure'):orchestrator.run_batch(keys)
        before=torch.get_rng_state().clone()
        with pytest.raises(ContractError,match='uncertain'):orchestrator.run_batch(keys)
        assert torch.equal(before,torch.get_rng_state()) and backend.batch_step_calls==0
        assert json.loads(next((tmp_path/'orchestrator').rglob('batch-step*.json')).read_text())['phase']=='LOCAL_PREPARED'
    finally:
        client.close()
