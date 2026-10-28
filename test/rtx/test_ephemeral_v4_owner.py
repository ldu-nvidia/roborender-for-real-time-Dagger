from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.chunks import CHUNK_LENGTH, COMPACT_AUDIT_ROW_FIELDS
from dagger_rtx.contracts import (
    CONTRACT_VERSION,
    ROUND_BETAS,
    StepKey,
    encode_array,
)
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.owner import CrashUncertain


def _key(lane: int, step: int) -> StepKey:
    return StepKey(
        run_id="v4-ephemeral-test",
        round_id=1,
        actor_id=f"actor-lane-{lane:03d}",
        rollout_window_id="window-000000",
        rollout_step_index=step,
        episode_id=f"episode-lane-{lane:03d}",
        episode_step_index=step,
    )


def _owner(
    tmp_path: Path,
    tiny_profile,
    *,
    ephemeral: bool,
    num_envs: int = 2,
) -> tuple[IsaacBatchTransitionOwner, DeterministicFakeBatchRaven]:
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=num_envs)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "runtime",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        ephemeral_hot_path=ephemeral,
    )
    return owner, backend


def _begin(owner: IsaacBatchTransitionOwner, *, num_envs: int) -> None:
    owner.begin_windows(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"begin-{lane}",
                "step_key": _key(lane, 0).as_dict(),
            }
            for lane in range(num_envs)
        ]
    )


def _step(
    owner: IsaacBatchTransitionOwner,
    step: int,
    *,
    num_envs: int,
) -> list[dict]:
    prepared = owner.prepare_batch_lean(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"prepare-{lane}-{step}",
                "step_key": _key(lane, step).as_dict(),
                "beta": ROUND_BETAS[1],
            }
            for lane in range(num_envs)
        ]
    )
    # Exercise the in-memory idempotent prepare response before committing.
    retried = owner.prepare_batch_lean(
        [
            {
                "contract_version": CONTRACT_VERSION,
                "request_id": f"prepare-{lane}-{step}",
                "step_key": _key(lane, step).as_dict(),
                "beta": ROUND_BETAS[1],
            }
            for lane in range(num_envs)
        ]
    )
    assert [row["prepare_receipt"] for row in retried] == [
        row["prepare_receipt"] for row in prepared
    ]
    requests = []
    for lane, row in enumerate(prepared):
        teacher = np.asarray(row["arrays"]["teacher_action"], dtype=np.float32)
        requests.append(
            {
                "contract_version": CONTRACT_VERSION,
                "step_key": row["step_key"],
                "prepare_receipt": row["prepare_receipt"],
                "selection_id": f"selection-{lane}-{step}",
                "selection_receipt": f"{lane * CHUNK_LENGTH + step + 1:064x}",
                "selected_source": "teacher",
                "beta": ROUND_BETAS[1],
                "selected_action": encode_array(teacher),
            }
        )
    return owner.commit_batch(requests)


def test_v4_ephemeral_owner_has_no_hot_per_lane_files_and_writes_final_seal(
    tmp_path: Path, tiny_profile
) -> None:
    owner, backend = _owner(tmp_path, tiny_profile, ephemeral=True, num_envs=2)
    _begin(owner, num_envs=2)

    marker = tmp_path / "runtime" / "v4_ephemeral_window_marker.json"
    assert json.loads(marker.read_text())["status"] == "active"
    assert not (tmp_path / "runtime" / "batch_owner_state.json").exists()
    assert not (tmp_path / "runtime" / "batch_owner_events.jsonl").exists()

    commits = []
    for step in range(CHUNK_LENGTH):
        commits = _step(owner, step, num_envs=2)
        assert backend.batch_step_calls == step + 1
        assert not list((tmp_path / "runtime").glob("receipts/**/*.json"))
        assert not list((tmp_path / "staging").glob("**/rows/step-*.npz"))

    assert all(row["sealed"] is True for row in commits)
    sealed = sorted((tmp_path / "staging").glob("**/sealed.npz"))
    manifests = sorted((tmp_path / "staging").glob("**/manifest.json"))
    assert len(sealed) == len(manifests) == 2
    for path in sealed:
        with np.load(path, allow_pickle=False) as audit:
            assert set(COMPACT_AUDIT_ROW_FIELDS) <= set(audit.files)
            assert audit["rollout_step_indices"].tolist() == list(range(CHUNK_LENGTH))
    owner.confirm_ephemeral_window_finalized(
        [{"step_key": _key(lane, CHUNK_LENGTH - 1).as_dict()} for lane in range(2)]
    )
    assert json.loads(marker.read_text())["status"] == "remote_finalized"


def test_v4_ephemeral_marker_makes_process_restart_fail_closed(
    tmp_path: Path, tiny_profile
) -> None:
    owner, _backend = _owner(tmp_path, tiny_profile, ephemeral=True, num_envs=1)
    _begin(owner, num_envs=1)
    _step(owner, 0, num_envs=1)

    restarted, _new_backend = _owner(
        tmp_path, tiny_profile, ephemeral=True, num_envs=1
    )
    assert restarted.status()["phase"] == "CRASH_UNCERTAIN"
    with pytest.raises(CrashUncertain, match="replay is forbidden"):
        restarted.prepare_batch_lean([])


def test_default_owner_keeps_existing_durable_receipt_and_row_path(
    tmp_path: Path, tiny_profile
) -> None:
    owner, backend = _owner(tmp_path, tiny_profile, ephemeral=False, num_envs=1)
    _begin(owner, num_envs=1)
    commit = _step(owner, 0, num_envs=1)

    assert backend.batch_step_calls == 1
    assert commit[0]["sealed"] is False
    assert (tmp_path / "runtime" / "batch_owner_state.json").is_file()
    assert (tmp_path / "runtime" / "batch_owner_events.jsonl").is_file()
    assert len(list((tmp_path / "runtime").glob("receipts/**/*.json"))) == 1
    rows = list((tmp_path / "staging").glob("**/rows/step-000.npz"))
    assert len(rows) == 1
