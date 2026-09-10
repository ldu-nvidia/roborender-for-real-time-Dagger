from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dagger_rtx.contracts import ContractError, canonical_json
from dagger_rtx.production_driver import (
    H200ContinuationActivityAttestation,
    PRODUCTION_STATE_SCHEMA,
    ProductionContinuationSegment,
    ProductionRestartUnsafe,
    verify_continuation_history,
    verify_episode_namespace_capacity,
    verify_h200_activity_evidence_file,
    verify_predecessor_state_file,
)


RUN_ID = "continuation-safety-run"
CONFIG_SHA = hashlib.sha256(b"config").hexdigest()
CHECKPOINT_SHA = hashlib.sha256(b"checkpoint").hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _activity(start: int, *, prepares: int = 0):
    return H200ContinuationActivityAttestation(
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        round_id=1,
        iteration_id=start + 1,
        rollout_window_id=f"window-{start:06d}",
        prepare_count=prepares,
        commit_count=0,
        chunk_count=0,
        actor_barrier_count=0,
        observed_at_utc="2026-09-10T09:30:00Z",
    )


def _segment(
    predecessor: Path,
    predecessor_raw: bytes,
    evidence: Path,
    evidence_raw: bytes,
    *,
    start: int = 1,
    base: int = 20_000_000,
    prior: tuple[str, ...] = (),
    activity: H200ContinuationActivityAttestation | None = None,
):
    activity = activity or _activity(start)
    return ProductionContinuationSegment(
        segment_id=f"segment-{start:06d}",
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        start_global_window=start,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        simulator_seed=43_000 + start,
        episode_serial_base=base,
        episode_serial_limit_exclusive=base + 100_000,
        reset_reason="crash_uncertain",
        predecessor_state_path=str(predecessor.resolve()),
        predecessor_state_sha256=_sha(predecessor_raw),
        h200_activity_attestation=activity,
        h200_activity_evidence_path=str(evidence.resolve()),
        h200_activity_evidence_sha256=_sha(evidence_raw),
        prior_continuation_manifest_sha256s=prior,
    )


def _predecessor_raw(start: int = 1) -> bytes:
    return canonical_json(
        {
            "schema": PRODUCTION_STATE_SCHEMA,
            "spec": {"run_id": RUN_ID, "config_sha256": CONFIG_SHA},
            "phase": "CRASH_UNCERTAIN",
            "next_window_index": start,
        }
    )


def test_unplanned_segment_binds_real_predecessor_and_zero_h200_evidence(tmp_path):
    predecessor = tmp_path / "production-driver-state.json"
    predecessor_raw = _predecessor_raw()
    predecessor.write_bytes(predecessor_raw)
    activity = _activity(1)
    evidence = tmp_path / "h200-activity.json"
    evidence_raw = canonical_json(activity.as_dict())
    evidence.write_bytes(evidence_raw)
    segment = _segment(
        predecessor, predecessor_raw, evidence, evidence_raw, activity=activity
    )

    assert verify_predecessor_state_file(segment, predecessor)["phase"] == "CRASH_UNCERTAIN"
    assert verify_h200_activity_evidence_file(segment, evidence) == activity
    roundtrip = ProductionContinuationSegment.from_mapping(segment.as_dict())
    assert roundtrip == segment

    alias = tmp_path / "copied-state.json"
    alias.write_bytes(predecessor_raw)
    with pytest.raises(ProductionRestartUnsafe, match="origin path"):
        verify_predecessor_state_file(segment, alias)


def test_unplanned_segment_rejects_nonzero_or_missing_h200_activity(tmp_path):
    predecessor = tmp_path / "production-driver-state.json"
    predecessor_raw = _predecessor_raw()
    predecessor.write_bytes(predecessor_raw)
    evidence = tmp_path / "h200-activity.json"
    nonzero = _activity(1, prepares=1)
    evidence_raw = canonical_json(nonzero.as_dict())
    evidence.write_bytes(evidence_raw)

    with pytest.raises(ContractError, match="nonzero H200"):
        _segment(
            predecessor,
            predecessor_raw,
            evidence,
            evidence_raw,
            activity=nonzero,
        )

    with pytest.raises(ContractError, match="requires zero-H200"):
        ProductionContinuationSegment(
            segment_id="missing-activity",
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            start_global_window=1,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            simulator_seed=43001,
            episode_serial_base=20_000_000,
            episode_serial_limit_exclusive=20_100_000,
            reset_reason="crash_uncertain",
            predecessor_state_path=str(predecessor.resolve()),
            predecessor_state_sha256=_sha(predecessor_raw),
            h200_activity_attestation=None,
            h200_activity_evidence_path=None,
            h200_activity_evidence_sha256=None,
            prior_continuation_manifest_sha256s=(),
        )


def test_episode_namespace_is_checked_against_exact_prior_manifest_set(tmp_path):
    history = tmp_path / "continuations"
    history.mkdir()
    predecessor = tmp_path / "production-driver-state.json"
    predecessor_raw = _predecessor_raw()
    predecessor.write_bytes(predecessor_raw)
    activity = _activity(1)
    evidence = tmp_path / "h200-activity.json"
    evidence_raw = canonical_json(activity.as_dict())
    evidence.write_bytes(evidence_raw)
    legacy = {
        "schema": "roborender.dagger.rtx.production-segment.v1",
        "run_id": RUN_ID,
        "episode_serial_base": 20_000_000,
    }
    legacy_raw = canonical_json(legacy)
    (history / "prior.json").write_bytes(legacy_raw)
    prior = (_sha(legacy_raw),)

    overlapping = _segment(
        predecessor,
        predecessor_raw,
        evidence,
        evidence_raw,
        base=20_050_000,
        prior=prior,
    )
    current = history / "current.json"
    current.write_bytes(canonical_json(overlapping.as_dict()))
    with pytest.raises(ProductionRestartUnsafe, match="overlaps"):
        verify_continuation_history(overlapping, current, history)

    nonoverlap = _segment(
        predecessor,
        predecessor_raw,
        evidence,
        evidence_raw,
        base=20_100_000,
        prior=prior,
    )
    current.write_bytes(canonical_json(nonoverlap.as_dict()))
    assert verify_continuation_history(nonoverlap, current, history) == prior

    incomplete = _segment(
        predecessor,
        predecessor_raw,
        evidence,
        evidence_raw,
        base=20_200_000,
        prior=(),
    )
    current.write_bytes(canonical_json(incomplete.as_dict()))
    with pytest.raises(ProductionRestartUnsafe, match="SHA-256 set"):
        verify_continuation_history(incomplete, current, history)


def test_episode_namespace_capacity_accounts_for_every_possible_reset(tmp_path):
    predecessor = tmp_path / "production-driver-state.json"
    predecessor_raw = _predecessor_raw()
    predecessor.write_bytes(predecessor_raw)
    activity = _activity(1)
    evidence = tmp_path / "h200-activity.json"
    evidence_raw = canonical_json(activity.as_dict())
    evidence.write_bytes(evidence_raw)
    segment = _segment(predecessor, predecessor_raw, evidence, evidence_raw)

    verify_episode_namespace_capacity(segment, range(28))
    with pytest.raises(ProductionRestartUnsafe, match="worst-case"):
        verify_episode_namespace_capacity(segment, [99_999])
