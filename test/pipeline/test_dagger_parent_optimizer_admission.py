from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from pipeline.dagger.services import raven_learner_worker as worker
from pipeline.dagger.services.metric_contract import LEGACY, MSE_FINAL_TEST


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.std = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.vision_module = torch.nn.Linear(2, 3)
        self.mlp_module = torch.nn.Linear(3, 1)
        self.obj_pred_mlp = torch.nn.Linear(3, 2)
        self.obj_pred_mlp.requires_grad_(False)

    def forward(self, x):
        return self.mlp_module(self.vision_module(x))


def parent(dagger=False):
    torch.manual_seed(41)
    model = SmallPolicy()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
        scheduler.step()
    result = {
        "schema": "roborender.dagger_student_checkpoint.v3" if dagger else "roborender.bc_student_checkpoint.v3",
        "global_step": 2, "policy_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "policy_spec": {},
    }
    if dagger:
        result["lr_scheduler_state_dict"] = scheduler.state_dict()
    return deepcopy(result), model, optimizer


@pytest.mark.parametrize("dagger", [False, True])
def test_complete_bc_or_dagger_parent_is_admitted_without_mutation(dagger):
    checkpoint, model, optimizer = parent(dagger)
    before = deepcopy(checkpoint)
    worker.validate_mse_parent_checkpoint(checkpoint)
    assert set(checkpoint) == set(before)
    for name, tensor in checkpoint["policy_state_dict"].items():
        assert torch.equal(tensor, before["policy_state_dict"][name])
    for ident, state in checkpoint["optimizer_state_dict"]["state"].items():
        assert all(torch.equal(value, before["optimizer_state_dict"]["state"][ident][key]) for key, value in state.items())
    restored = SmallPolicy()
    restored.load_state_dict(checkpoint["policy_state_dict"])
    resumed = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=2e-4, weight_decay=0)
    resumed.load_state_dict(checkpoint["optimizer_state_dict"])
    for policy, opt in ((model, optimizer), (restored, resumed)):
        opt.zero_grad(set_to_none=True)
        policy(torch.ones(2, 2)).square().mean().backward()
        opt.step()
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("optimizer_state_dict"),
    lambda c: c.pop("global_step"),
    lambda c: c.update(global_step=True),
    lambda c: c.update(global_step=0),
    lambda c: c.update(global_step=3),
    lambda c: c["optimizer_state_dict"].update(state={}),
    lambda c: c["optimizer_state_dict"]["state"].pop(0),
    lambda c: c["optimizer_state_dict"]["state"].update({99: {}}),
    lambda c: c["optimizer_state_dict"]["param_groups"][0].update(params=[0, 1, 2, 2]),
    lambda c: c["optimizer_state_dict"]["state"][0].pop("exp_avg_sq"),
    lambda c: c["optimizer_state_dict"]["state"][0].update(exp_avg=torch.zeros(1)),
    lambda c: c["optimizer_state_dict"]["state"][0].update(exp_avg=torch.zeros(3, 2, dtype=torch.float64)),
    lambda c: c["optimizer_state_dict"]["state"][0].update(step=torch.tensor(float("nan"))),
    lambda c: c["optimizer_state_dict"]["state"][0].update(step=torch.ones(2)),
    lambda c: c["optimizer_state_dict"]["state"][0].update(step=torch.tensor(1.0)),
    lambda c: c["optimizer_state_dict"]["param_groups"][0].update(weight_decay=0.01),
    lambda c: c["optimizer_state_dict"]["param_groups"][0].update(betas=None),
    lambda c: c["optimizer_state_dict"]["param_groups"][0].update(lr=float("nan")),
    lambda c: c["policy_state_dict"].update({0: torch.ones(1)}),
])
def test_missing_incomplete_or_mismatched_adamw_cannot_silently_restart(mutation):
    checkpoint, _, _ = parent()
    mutation(checkpoint)
    with pytest.raises(ValueError):
        worker.validate_mse_parent_checkpoint(checkpoint)


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("lr_scheduler_state_dict"),
    lambda c: c.update(lr_scheduler_state_dict={}),
    lambda c: c["lr_scheduler_state_dict"].update(last_epoch=0),
    lambda c: c["lr_scheduler_state_dict"].update(_step_count=100),
    lambda c: c["lr_scheduler_state_dict"].update(base_lrs=[]),
    lambda c: c["lr_scheduler_state_dict"].update(_last_lr=[float("nan")]),
    lambda c: c["lr_scheduler_state_dict"].update(base_lrs=[1e-3]),
    lambda c: c["lr_scheduler_state_dict"].update(_last_lr=[1e-3]),
    lambda c: c["optimizer_state_dict"]["param_groups"][0].update(initial_lr=1e-3),
    lambda c: c["lr_scheduler_state_dict"].update(lr_lambdas=[{}]),
])
def test_dagger_parent_requires_consistent_scheduler_but_bc_does_not(mutation):
    checkpoint, _, _ = parent(True)
    mutation(checkpoint)
    with pytest.raises(ValueError):
        worker.validate_mse_parent_checkpoint(checkpoint)


@pytest.mark.parametrize("value", [{}, [], {"last_epoch": 3}])
def test_bc_parent_cannot_smuggle_unvalidated_scheduler_state(value):
    checkpoint, _, _ = parent()
    checkpoint["lr_scheduler_state_dict"] = value
    with pytest.raises(ValueError, match="BC-v3"):
        worker.validate_mse_parent_checkpoint(checkpoint)


@pytest.mark.parametrize("dagger", [False, True])
def test_requested_learning_rate_must_match_exact_continuation(dagger):
    checkpoint, _, _ = parent(dagger)
    worker.validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=2e-4)
    with pytest.raises(ValueError, match="requested learning_rate"):
        worker.validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=1e-4)


def fixed_final_parent():
    import random
    import numpy as np
    from pipeline.bc_training.stopping_gate import fixed_final_contract
    checkpoint, _, _ = parent()
    checkpoint.update(schema='roborender.bc_student_checkpoint.v4', epoch=2,
        training_protocol=fixed_final_contract(2, budget_explicit=True),
        validation_enabled=False, selection='last_completed_epoch',
        training_batch_contract=dict(updates_per_epoch=1, chunks_per_update=16,
            frames_per_chunk=81, policy_forward_population=1296),
        rng_state=dict(python=random.getstate(), numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(), torch_cuda=[],
            train_generator=torch.Generator().get_state()))
    return checkpoint


def test_fixed_final_bc_v4_parent_preserves_adamw_admission():
    checkpoint=fixed_final_parent()
    worker.validate_mse_parent_checkpoint(checkpoint, requested_learning_rate=2e-4)
    checkpoint['optimizer_state_dict']['state'][0]['step']=torch.tensor(1.)
    with pytest.raises(ValueError, match='AdamW step'):
        worker.validate_mse_parent_checkpoint(checkpoint)


@pytest.mark.parametrize('mutation', [
    lambda c: c.update(epoch=1),
    lambda c: c.update(validation_enabled=True),
    lambda c: c.update(selection='minimum_validation_mse'),
    lambda c: c['training_protocol'].update(max_epochs=3),
    lambda c: c['training_protocol'].update(validation_enabled=True),
    lambda c: c['training_protocol'].update(training_eval_enabled=0),
    lambda c: c['training_batch_contract'].update(chunks_per_update=4),
    lambda c: c['training_batch_contract'].update(updates_per_epoch=2),
    lambda c: c.update(rng_state={}),
    lambda c: c.update(lr_scheduler_state_dict={}),
])
def test_fixed_final_parent_rejects_partial_or_relabelled_checkpoint(mutation):
    checkpoint=fixed_final_parent();mutation(checkpoint)
    with pytest.raises(ValueError):
        worker.validate_mse_parent_checkpoint(checkpoint)


@pytest.mark.parametrize("protocol,trained,learning_rate,expected_error", [
    (MSE_FINAL_TEST, False, 2e-4, "admission"),
    (LEGACY, False, 2e-4, "cuda"),
    (MSE_FINAL_TEST, True, 2e-4, "cuda"),
    (MSE_FINAL_TEST, True, 1e-4, "requested learning_rate"),
])
def test_main_checks_mse_parent_before_cuda_or_model_and_keeps_legacy_path(
        tmp_path, monkeypatch, protocol, trained, learning_rate, expected_error):
    import json
    digest = "a" * 64
    root = tmp_path / "state"
    (root / "checkpoints").mkdir(parents=True)
    checkpoint = parent()[0] if trained else {"policy_spec": {}, "policy_state_dict": {}}
    torch.save(checkpoint, root / "checkpoints" / f"{digest}.pt")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"run_id": "admission", "round_id": 1,
        "input_checkpoint_sha256": digest, "train_chunk_ids": ["chunk"],
        "parameters": {"metric_protocol": protocol, "iteration_id": 1, "learning_rate": learning_rate}}))
    monkeypatch.setattr(worker, "parse_args", lambda: SimpleNamespace(job=job, state_root=root, result=tmp_path / "result.json"))
    monkeypatch.setattr(worker, "file_sha256", lambda path: digest)
    class ReachedCuda(Exception):
        pass
    def cuda_probe():
        raise ReachedCuda()
    monkeypatch.setattr(torch.cuda, "is_available", cuda_probe)
    monkeypatch.setattr(worker, "build_policy", lambda *args: pytest.fail("model must not be constructed"))
    with pytest.raises(ReachedCuda if expected_error == "cuda" else ValueError,
                       match="requested learning_rate" if expected_error == "requested learning_rate" else None):
        worker.main()
