from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.dagger.services.raven_learner_worker import (
    IndexSafeMaxPool2d,
    backward_global_mse_in_environment_microbatches,
    install_index_safe_max_pool2d,
)


class RecordingPolicy(torch.nn.Module):
    def __init__(self, actor_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(actor_dim, action_dim, bias=False)
        self.input_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(self, observation):
        vision = observation["vision_obs"]
        actor = observation["actor_obs"]
        self.input_shapes.append((tuple(vision.shape), tuple(actor.shape)))
        return {"actions": self.linear(actor)}


def test_raven_exact_batch_is_one_flattened_bxt_policy_forward():
    torch.manual_seed(7)
    batch, timesteps, actor_dim, action_dim = 4, 3, 5, 2
    vision = torch.randn(batch, timesteps, 6, 7, 3)
    actor = torch.randn(batch, timesteps, actor_dim)
    target = torch.randn(batch, timesteps, action_dim)

    policy = RecordingPolicy(actor_dim, action_dim)
    reference = RecordingPolicy(actor_dim, action_dim)
    reference.load_state_dict(policy.state_dict())

    actual_mse = backward_global_mse_in_environment_microbatches(
        policy,
        vision,
        actor,
        target,
        microbatch_environments=batch,
        full_environment_batch_per_timestep=True,
    )
    expected_loss = torch.nn.functional.mse_loss(
        reference.linear(actor), target, reduction="mean"
    )
    expected_loss.backward()

    assert actual_mse == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    expected_shape = (
        (batch, timesteps, 6, 7, 3), (batch, timesteps, actor_dim)
    )
    # Non-reentrant activation checkpointing evaluates the policy once for the
    # forward result and once during backward recomputation.  Crucially, both
    # calls see the complete BxT population; neither is a temporal microbatch.
    assert policy.input_shapes == [expected_shape, expected_shape]
    assert torch.allclose(policy.linear.weight.grad, reference.linear.weight.grad)


def test_full_environment_batch_rejects_environment_microbatching():
    policy = RecordingPolicy(actor_dim=3, action_dim=2)
    with pytest.raises(ValueError, match="complete environment batch"):
        backward_global_mse_in_environment_microbatches(
            policy,
            torch.randn(4, 2, 3, 3, 3),
            torch.randn(4, 2, 3),
            torch.randn(4, 2, 2),
            microbatch_environments=2,
            full_environment_batch_per_timestep=True,
        )


@pytest.mark.parametrize("recompute,full_batch", [("false", True), (False, False)])
def test_recomputation_opt_out_rejects_invalid_or_microbatched_mode(recompute, full_batch):
    policy = RecordingPolicy(actor_dim=3, action_dim=2)
    with pytest.raises(ValueError):
        backward_global_mse_in_environment_microbatches(
            policy,
            torch.randn(4, 2, 3, 3, 3),
            torch.randn(4, 2, 3),
            torch.randn(4, 2, 2),
            microbatch_environments=4,
            full_environment_batch_per_timestep=full_batch,
            activation_checkpointing=recompute,
        )
    assert policy.input_shapes == []


class BatchNormPolicy(torch.nn.Module):
    """Small VisionActorObjPred-shaped policy for exact BN comparisons."""

    def __init__(self, actor_dim: int, action_dim: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.bn = torch.nn.SyncBatchNorm(4, momentum=0.001)
        self.head = torch.nn.Linear(4 + actor_dim, action_dim)
        self.forward_calls = 0

    def forward(self, observation):
        self.forward_calls += 1
        vision = observation["vision_obs"]
        actor = observation["actor_obs"]
        batch, timesteps, height, width, channels = vision.shape
        image = vision.reshape(batch * timesteps, height, width, channels)
        image = image.permute(0, 3, 1, 2)
        features = self.bn(self.conv(image)).mean(dim=(2, 3))
        features = features.reshape(batch, timesteps, -1)
        return {"actions": self.head(torch.cat((actor, features), dim=-1))}


@pytest.mark.parametrize("recompute", [True, False])
def test_raven_exact_batch_matches_full_sync_batch_norm_state_and_gradients(recompute):
    torch.manual_seed(19)
    batch, timesteps, actor_dim, action_dim = 4, 5, 3, 2
    vision = torch.randn(batch, timesteps, 7, 6, 3)
    actor = torch.randn(batch, timesteps, actor_dim)
    target = torch.randn(batch, timesteps, action_dim)

    actual = BatchNormPolicy(actor_dim, action_dim)
    reference = BatchNormPolicy(actor_dim, action_dim)
    reference.load_state_dict(actual.state_dict())
    actual.train()
    reference.train()

    actual_mse = backward_global_mse_in_environment_microbatches(
        actual,
        vision,
        actor,
        target,
        microbatch_environments=batch,
        full_environment_batch_per_timestep=True,
        activation_checkpointing=recompute,
    )
    expected = torch.nn.functional.mse_loss(
        reference({"vision_obs": vision, "actor_obs": actor})["actions"], target
    )
    expected.backward()

    assert actual_mse == pytest.approx(float(expected.detach()), rel=1e-6)
    assert actual.forward_calls == (2 if recompute else 1)
    assert reference.forward_calls == 1
    assert torch.equal(actual.bn.num_batches_tracked, reference.bn.num_batches_tracked)
    assert torch.allclose(actual.bn.running_mean, reference.bn.running_mean)
    assert torch.allclose(actual.bn.running_var, reference.bn.running_var)
    for actual_parameter, expected_parameter in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        assert torch.allclose(actual_parameter.grad, expected_parameter.grad)


def test_index_safe_max_pool_matches_unsplit_output_indices_and_gradient():
    torch.manual_seed(23)
    value = torch.randn(7, 3, 8, 9, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    ordinary = torch.nn.MaxPool2d(3, stride=2, padding=1, return_indices=True)
    safe = IndexSafeMaxPool2d(ordinary, max_input_elements=3 * 8 * 9 * 2)

    actual, actual_indices = safe(value)
    expected, expected_indices = ordinary(reference_value)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_indices, expected_indices)

    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    assert torch.equal(value.grad, reference_value.grad)


def test_install_index_safe_pool_replaces_nested_pool_only_once():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.ReLU()),
    )
    assert install_index_safe_max_pool2d(model, max_input_elements=100) == 1
    assert isinstance(model[1][0], IndexSafeMaxPool2d)
    assert install_index_safe_max_pool2d(model, max_input_elements=100) == 0
