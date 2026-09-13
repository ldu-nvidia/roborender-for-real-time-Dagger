from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.dagger.services.raven_learner_worker import (
    backward_global_mse_in_environment_microbatches,
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


def test_full_environment_batch_is_sliced_only_over_time():
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

    assert actual_mse == pytest.approx(float(expected_loss), rel=1e-6)
    assert len(policy.input_shapes) == timesteps
    assert policy.input_shapes == [
        ((batch, 1, 6, 7, 3), (batch, 1, actor_dim))
    ] * timesteps
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
