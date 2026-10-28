from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from dagger_rtx.contracts import CONTRACT_VERSION, ROUND_BETAS, ContractError, StepKey, encode_array
from dagger_rtx.fake import DeterministicFakeRaven, fake_static_provenance
from dagger_rtx.owner import CrashUncertain, IsaacTransitionOwner, OwnerConflict

from test.rtx.conftest import PROJECT_ROOT, commit_teacher, make_key, make_owner, prepare


def test_prepare_and_commit_retries_never_double_step(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    first = prepare(owner, make_key())
    again = prepare(owner, make_key())
    assert again == first
    assert backend.capture_calls == 1
    committed = commit_teacher(owner, first)
    retried = commit_teacher(owner, first)
    assert retried == committed
    assert backend.step_calls == 1


def test_changed_prepare_or_commit_retry_is_rejected(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    first = prepare(owner, make_key())
    changed = {
        "contract_version": CONTRACT_VERSION,
        "request_id": "changed",
        "step_key": make_key().as_dict(),
        "beta": ROUND_BETAS[1],
    }
    with pytest.raises(OwnerConflict):
        owner.prepare_step(changed)
    commit_teacher(owner, first)
    action = np.arange(26, dtype=np.float32)
    with pytest.raises(OwnerConflict):
        owner.commit_step(
            {
                "contract_version": CONTRACT_VERSION,
                "step_key": first["step_key"],
                "prepare_receipt": first["prepare_receipt"],
                "selection_id": "different-selection",
                "selected_source": "student",
                "beta": ROUND_BETAS[1],
                "selected_action": encode_array(action),
            }
        )
    assert backend.step_calls == 1


def test_backend_failure_is_permanently_crash_uncertain(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile, raise_at_calls={0})
    first = prepare(owner, make_key())
    with pytest.raises(CrashUncertain):
        commit_teacher(owner, first)
    assert backend.step_calls == 1
    restarted = IsaacTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    assert restarted.status()["phase"] == "CRASH_UNCERTAIN"
    with pytest.raises(CrashUncertain):
        commit_teacher(restarted, first)
    assert backend.step_calls == 1


def test_done_reset_inside_window_still_seals_exactly_81(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile, done_at_calls={2})
    episode_id = "episode-a"
    episode_step = 0
    for rollout_step in range(81):
        if rollout_step == 3:
            episode_id = "episode-b"
            episode_step = 0
        key = make_key(rollout_step, episode_id=episode_id, episode_step=episode_step)
        committed = commit_teacher(owner, prepare(owner, key))
        episode_step += 1
        if rollout_step < 80:
            assert not committed["sealed"]
    assert committed["sealed"]
    assert backend.step_calls == 81
    assert backend.bind_calls == 2
    seal = committed["seal"]
    assert seal["row_count"] == 81
    assert [b["episode_id"] for b in seal["provenance"]["episode_bindings"]] == ["episode-a", "episode-b"]
    schema = json.loads((PROJECT_ROOT / "config/render_student_boundary.v1.schema.json").read_text())
    Draft202012Validator({"$ref": "#/$defs/sealedWindow", "$defs": schema["$defs"]}).validate(seal)
    npz_path = tmp_path / "staging/test-run/D1/actor-0/window-0/sealed.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["episode_ids"].shape == (81,)
        assert list(data["episode_ids"][:4]) == ["episode-a", "episode-a", "episode-a", "episode-b"]
        assert bool(data["dones"][2])
        assert int(data["episode_step_indices"][3]) == 0
        assert np.array_equal(data["rollout_step_indices"], np.arange(81))
        assert np.array_equal(data["round_ids"], np.ones(81, dtype=np.int64))
        assert np.array_equal(data["gt_actions"][:, 11:], data["homie_actions"])
        assert data["gt_actions"].shape == (81, 26)
        assert data["behavior_actions"].shape == (81, 26)
    # A durable SEALED record is not enough to restore the still-live Raven
    # episode/RNG state needed to continue into the next 81-step window.
    restart_runtime = tmp_path / "restart-runtime"
    shutil.copytree(tmp_path / "runtime", restart_runtime)
    restarted = IsaacTransitionOwner(
        DeterministicFakeRaven(tiny_profile),
        restart_runtime,
        tmp_path / "restart-staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    assert restarted.status()["phase"] == "CRASH_UNCERTAIN"
    with pytest.raises(CrashUncertain):
        restarted.begin_window(
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": "unsafe-restart-window",
                "step_key": make_key(
                    0, episode_id="episode-b", episode_step=78, rollout_window_id="unsafe-window"
                ).as_dict(),
            }
        )
    next_window = make_key(
        0,
        episode_id="episode-b",
        episode_step=78,
        rollout_window_id="window-1",
    )
    owner.begin_window(
        {
            "contract_version": CONTRACT_VERSION,
            "request_id": "begin-window-1",
            "step_key": next_window.as_dict(),
        }
    )
    prepare(owner, next_window)
    # The anchor/source assignment must not be resampled at an update-window boundary.
    assert backend.bind_calls == 2


def test_restart_from_durable_commit_intent_never_calls_backend(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    prepared = prepare(owner, make_key())
    owner.state["phase"] = "COMMITTING"
    owner.state["commit_intent"] = {"step_key": make_key().as_dict(), "simulated": True}
    owner._persist()
    restarted = IsaacTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    assert restarted.status()["phase"] == "CRASH_UNCERTAIN"
    with pytest.raises(CrashUncertain):
        commit_teacher(restarted, prepared)
    assert backend.step_calls == 0


def test_owner_pins_run_actor_and_monotonic_rounds(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    owner.state["phase"] = "SEALED"
    owner.state["needs_episode_bind"] = True
    owner._persist()
    for bad in (
        make_key(0, round_id=3, episode_id="episode-b", episode_step=0, rollout_window_id="skip-D2"),
        StepKey("test-run", 2, "other-actor", "wrong-actor", 0, "episode-b", 0),
    ):
        with pytest.raises(OwnerConflict):
            owner.begin_window(
                {
                    "contract_version": CONTRACT_VERSION,
                    "request_id": f"bad-{bad.rollout_window_id}",
                    "step_key": bad.as_dict(),
                }
            )
    good = make_key(
        0, round_id=2, episode_id="episode-b", episode_step=0, rollout_window_id="window-D2"
    )
    owner.begin_window(
        {
            "contract_version": CONTRACT_VERSION,
            "request_id": "begin-D2",
            "step_key": good.as_dict(),
        }
    )
    assert owner.status()["session"]["highest_round_id"] == 2
    assert backend.bind_calls == 2


def test_generated_episode_forbids_unused_raytraced_policy_tiles(tmp_path, tiny_profile):
    owner, backend = make_owner(tmp_path, tiny_profile)
    owner.state["active_episode"]["assignment"]["use_raytraced"] = False
    owner.state["active_episode"]["assignment"]["observation_source"] = "generated_controls"
    backend.use_raytraced = False
    original = backend.capture_current

    def contaminated(*, key):
        capture = original(key=key)
        capture.arrays["raytraced_rgb__head_color"] = np.zeros(
            (tiny_profile.policy_h, tiny_profile.policy_w, 3), dtype=np.float32
        )
        return capture

    backend.capture_current = contaminated
    with pytest.raises(ContractError, match="must not carry"):
        prepare(owner, make_key())


@pytest.mark.parametrize("phase", ["READY", "PREPARED"])
def test_restart_of_active_native_state_fails_closed_without_restore_attestation(
    tmp_path, tiny_profile, phase
):
    owner, backend = make_owner(tmp_path, tiny_profile)
    prepared = None
    if phase == "PREPARED":
        prepared = prepare(owner, make_key())
    assert owner.status()["phase"] == phase
    restarted = IsaacTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
    )
    assert restarted.status()["phase"] == "CRASH_UNCERTAIN"
    if prepared is not None:
        with pytest.raises(CrashUncertain):
            commit_teacher(restarted, prepared)
    assert backend.step_calls == 0
