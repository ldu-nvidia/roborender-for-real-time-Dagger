from __future__ import annotations

from pathlib import Path

import numpy as np

import dagger_rtx.chunks as chunks_module
from dagger_rtx.backend import TransitionResult
from dagger_rtx.chunks import (
    CHUNK_LENGTH,
    COMPACT_AUDIT_ROW_FIELDS,
    CurrentRoundChunk,
)
from dagger_rtx.contracts import StepKey
from dagger_rtx.fake import DeterministicFakeRaven, fake_static_provenance


def _key(index: int) -> StepKey:
    return StepKey(
        run_id="compact-audit-test",
        round_id=1,
        actor_id="actor-lane-000",
        rollout_window_id="window-000000",
        rollout_step_index=index,
        episode_id="episode-000000",
        episode_step_index=index,
    )


def _audit(index: int, *, include_selection_receipt: bool = True) -> dict[str, str]:
    values = {
        "prepare_receipt": f"{index + 1:064x}",
        "capture_digest": f"{index + 101:064x}",
        "selection_id": f"selection-{index:03d}",
        "selection_receipt": f"{index + 201:064x}",
        "commit_digest": f"{index + 301:064x}",
        "commit_receipt": f"{index + 401:064x}",
    }
    if not include_selection_receipt:
        values.pop("selection_receipt")
    return values


def test_compact_append_never_persists_observation_or_control_arrays(
    tmp_path: Path, tiny_profile
) -> None:
    backend = DeterministicFakeRaven(tiny_profile)
    key = _key(0)
    backend.bind_episode(key=key)
    capture = backend.capture_current(key=key)
    chunk = CurrentRoundChunk(
        tmp_path,
        run_id=key.run_id,
        round_id=key.round_id,
        actor_id=key.actor_id,
        rollout_window_id=key.rollout_window_id,
    )
    chunk.append(
        key,
        capture,
        capture.arrays["teacher_action"],
        "teacher",
        TransitionResult(1.25, False, False, {"test": True}),
        _audit(0),
    )

    row_path = chunk.rows_dir / "step-000.npz"
    with np.load(row_path, allow_pickle=False) as row:
        assert set(row.files) == set(COMPACT_AUDIT_ROW_FIELDS)
        assert str(row["selection_receipts"]) == _audit(0)["selection_receipt"]
        assert np.array_equal(row["gt_actions"][11:], row["homie_actions"])
        assert np.array_equal(row["behavior_actions"], row["gt_actions"])
        assert not any(
            name.startswith(("obs_", "depth_", "rgb_", "anchor_", "raytraced_"))
            for name in row.files
        )


def test_legacy_bulk_rows_are_sealed_through_selective_compact_projection(
    tmp_path: Path, monkeypatch
) -> None:
    """Legacy recovery may inspect ZIP names but must never inflate bulk members."""

    key0 = _key(0)
    chunk = CurrentRoundChunk(
        tmp_path,
        run_id=key0.run_id,
        round_id=key0.round_id,
        actor_id=key0.actor_id,
        rollout_window_id=key0.rollout_window_id,
    )
    teacher = np.arange(26, dtype=np.float32)
    homie = teacher[11:].copy()
    for index in range(CHUNK_LENGTH):
        audit = _audit(index, include_selection_receipt=False)
        arrays = {
            "gt_actions": teacher,
            "homie_actions": homie,
            "behavior_actions": teacher,
            "behavior_source": np.asarray(0, dtype=np.uint8),
            "dones": np.asarray(False, dtype=np.bool_),
            "timeouts": np.asarray(False, dtype=np.bool_),
            "rewards": np.asarray(index / 10.0, dtype=np.float32),
            "episode_ids": np.asarray(key0.episode_id),
            "episode_step_indices": np.asarray(index, dtype=np.int64),
            "rollout_step_indices": np.asarray(index, dtype=np.int64),
            "round_ids": np.asarray(1, dtype=np.int64),
            "env_id_per_step": np.asarray(7, dtype=np.int64),
            "seq_per_step": np.asarray(9, dtype=np.int64),
            "observation_source": np.asarray("raytraced"),
            "prompt_per_step": np.asarray("compact audit prompt"),
            "prepare_receipts": np.asarray(audit["prepare_receipt"]),
            "capture_digests": np.asarray(audit["capture_digest"]),
            "selection_ids": np.asarray(audit["selection_id"]),
            "commit_digests": np.asarray(audit["commit_digest"]),
            "commit_receipts": np.asarray(audit["commit_receipt"]),
            # Representative legacy-only members.  The guard wrapper below
            # proves seal() never indexes/decompresses them.
            "obs_actor_obs": np.full((113,), index, dtype=np.float32),
            "depth_raw__head_color": np.full((32, 48), index, dtype=np.float16),
            "rgb_robot__head_color": np.full((32, 48, 3), index, dtype=np.uint8),
        }
        chunk._write_npz_atomic(chunk.rows_dir / f"step-{index:03d}.npz", arrays)

    real_load = np.load

    class SelectiveNpz:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.files = wrapped.files

        def __getitem__(self, name):
            if name.startswith(("obs_", "depth_", "rgb_", "anchor_", "raytraced_")):
                raise AssertionError(f"bulk legacy member was inflated: {name}")
            return self._wrapped[name]

        def close(self):
            self._wrapped.close()

    monkeypatch.setattr(
        chunks_module.np,
        "load",
        lambda path, **kwargs: SelectiveNpz(real_load(path, **kwargs)),
    )
    provenance = {
        **fake_static_provenance(),
        "episode_bindings": [
            {
                "episode_id": key0.episode_id,
                "first_rollout_step_index": 0,
                "use_raytraced": True,
                "observation_source": "raytraced",
                "anchor_sha256": "a" * 64,
            }
        ],
    }
    manifest = chunk.seal(provenance=provenance)

    assert manifest["row_count"] == CHUNK_LENGTH
    assert manifest["artifact_role"] == "rtx_identity_outcome_audit"
    with real_load(chunk.sealed_path, allow_pickle=False) as sealed:
        assert "selection_receipts" not in sealed.files
        assert not any(
            name.startswith(("obs_", "depth_", "rgb_", "anchor_", "raytraced_"))
            for name in sealed.files
        )
        assert np.array_equal(sealed["rollout_step_indices"], np.arange(CHUNK_LENGTH))
        assert np.array_equal(sealed["gt_actions"][:, 11:], sealed["homie_actions"])
