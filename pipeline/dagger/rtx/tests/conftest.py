from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.contracts import CONTRACT_VERSION, ROUND_BETAS, RavenShapeProfile, StepKey, encode_array
from dagger_rtx.fake import DeterministicFakeRaven, fake_static_provenance
from dagger_rtx.owner import IsaacTransitionOwner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_profile() -> RavenShapeProfile:
    return RavenShapeProfile(
        actor_obs=4,
        teacher_obs=5,
        critic_obs=5,
        homie_obs=6,
        depth_h=2,
        depth_w=3,
        control_h=2,
        control_w=3,
        policy_h=2,
        policy_w=3,
    )


def make_key(
    rollout_step: int = 0,
    *,
    round_id: int = 1,
    episode_id: str = "episode-a",
    episode_step: int | None = None,
    rollout_window_id: str = "window-0",
) -> StepKey:
    return StepKey(
        run_id="test-run",
        round_id=round_id,
        actor_id="actor-0",
        rollout_window_id=rollout_window_id,
        rollout_step_index=rollout_step,
        episode_id=episode_id,
        episode_step_index=rollout_step if episode_step is None else episode_step,
    )


def make_owner(tmp_path: Path, tiny_profile: RavenShapeProfile, **backend_kwargs):
    backend = DeterministicFakeRaven(tiny_profile, **backend_kwargs)
    owner = IsaacTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    owner.begin_window(
        {
            "contract_version": CONTRACT_VERSION,
            "request_id": "begin-window-0",
            "step_key": make_key().as_dict(),
        }
    )
    return owner, backend


def prepare(owner: IsaacTransitionOwner, key: StepKey):
    return owner.prepare_step(
        {
            "contract_version": CONTRACT_VERSION,
            "request_id": f"prepare-{key.rollout_step_index}",
            "step_key": key.as_dict(),
            "beta": ROUND_BETAS[key.round_id],
        }
    )


def commit_teacher(owner: IsaacTransitionOwner, prepared):
    return owner.commit_step(
        {
            "contract_version": CONTRACT_VERSION,
            "step_key": prepared["step_key"],
            "prepare_receipt": prepared["prepare_receipt"],
            "selection_id": f"select-{prepared['step_key']['rollout_step_index']}",
            "selected_source": "teacher",
            "beta": prepared["metadata"]["beta"],
            "selected_action": prepared["arrays"]["teacher_action"],
        }
    )
