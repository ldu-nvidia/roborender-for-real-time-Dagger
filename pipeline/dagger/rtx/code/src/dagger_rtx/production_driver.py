"""Fail-closed multi-window owner for the native RTX DAgger process.

This module deliberately does not know the coordinator's HTTP representation.
The H200 control adapter translates its authenticated wire response into the
small, typed gates below.  Keeping that interface injectable lets the state
machine be tested without Isaac or a live H200 while preserving a hard rule:
only the already-initialized :class:`BatchWindowRunner` may advance Raven.

The driver is *process persistent*, not process restartable.  Raven/Isaac does
not expose an attested simulator, camera, recurrent-model, and RNG snapshot.
Consequently an interrupted process can be diagnosed from the durable state,
but must never recreate or replay a window.  This is the only safe recovery
semantics until such a native snapshot contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence
import uuid

from .chunks import CHUNK_LENGTH
from .contracts import (
    ROUND_BETAS,
    ContractError,
    canonical_json,
    require_identifier,
    require_sha256,
    sha256_bytes,
)
from .durable import atomic_write_json


PRODUCTION_STATE_SCHEMA = "roborender.dagger.rtx.production-driver.v1"
PRODUCTION_SEGMENT_SCHEMA = "roborender.dagger.rtx.production-segment.v2"
H200_ACTIVITY_ATTESTATION_SCHEMA = (
    "roborender.dagger.rtx.continuation-h200-activity.v1"
)
LEGACY_PRODUCTION_SEGMENT_SCHEMA = "roborender.dagger.rtx.production-segment.v1"
EPISODE_NAMESPACE_MIN_WIDTH = 100_000
PRODUCTION_WIRE_CONTRACT = "roborender.dagger.rtx.v4"
PRODUCTION_WIRE_CONTRACTS = frozenset(
    {
        "roborender.dagger.rtx.v3",
        PRODUCTION_WIRE_CONTRACT,
    }
)
PRODUCTION_WINDOW_COUNT = 40
PRODUCTION_VALIDATION_INTERVAL = 40
PRODUCTION_BATCH_SIZE = 28
PRODUCTION_ROUND_ID = 1
PRODUCTION_BETA = 0.75
RTX_PRO_6000_GPU1_UUID = "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9"


class ProductionDriverError(RuntimeError):
    """A production gate or durable state violates the frozen contract."""


class ProductionRestartUnsafe(ProductionDriverError):
    """A new process cannot prove the native Raven state needed to continue."""


def _require_absolute_path(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ContractError(f"{name} must be a non-empty single-line path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{name} must be an absolute normalized path")
    return value


@dataclass(frozen=True)
class H200ContinuationActivityAttestation:
    """Authenticated H200 ledger snapshot for one not-yet-started iteration.

    The RTX coordinator status endpoint exposes actor barriers but not the
    render/student prepare, commit, or learner-chunk ledgers.  An unplanned
    fresh simulator segment therefore requires a separately supplied,
    hash-bound H200 ledger snapshot proving that all three counts are zero.
    """

    run_id: str
    config_sha256: str
    round_id: int
    iteration_id: int
    rollout_window_id: str
    prepare_count: int
    commit_count: int
    chunk_count: int
    actor_barrier_count: int
    observed_at_utc: str
    source: str = "authenticated_h200_ledger_snapshot"
    schema: str = H200_ACTIVITY_ATTESTATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != H200_ACTIVITY_ATTESTATION_SCHEMA:
            raise ContractError("H200 continuation activity schema mismatch")
        require_identifier(self.run_id, "activity.run_id")
        require_sha256(self.config_sha256, "activity.config_sha256")
        if self.round_id != PRODUCTION_ROUND_ID:
            raise ContractError("activity round_id must equal the production round")
        if (
            isinstance(self.iteration_id, bool)
            or not isinstance(self.iteration_id, int)
            or self.iteration_id < 2
        ):
            raise ContractError("activity iteration_id must be at least 2")
        require_identifier(self.rollout_window_id, "activity.rollout_window_id")
        for name in (
            "prepare_count",
            "commit_count",
            "chunk_count",
            "actor_barrier_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"activity {name} must be a non-negative integer")
        if self.source != "authenticated_h200_ledger_snapshot":
            raise ContractError("activity source is not an authenticated H200 ledger snapshot")
        if (
            not isinstance(self.observed_at_utc, str)
            or len(self.observed_at_utc) < 20
            or not self.observed_at_utc.endswith("Z")
        ):
            raise ContractError("activity observed_at_utc must be an explicit UTC timestamp")

    @property
    def is_zero(self) -> bool:
        return all(
            getattr(self, name) == 0
            for name in (
                "prepare_count",
                "commit_count",
                "chunk_count",
                "actor_barrier_count",
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "round_id": self.round_id,
            "iteration_id": self.iteration_id,
            "rollout_window_id": self.rollout_window_id,
            "prepare_count": self.prepare_count,
            "commit_count": self.commit_count,
            "chunk_count": self.chunk_count,
            "actor_barrier_count": self.actor_barrier_count,
            "observed_at_utc": self.observed_at_utc,
            "source": self.source,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "H200ContinuationActivityAttestation":
        if not isinstance(value, Mapping):
            raise ContractError("H200 continuation activity must be an object")
        exact = {
            "schema",
            "run_id",
            "config_sha256",
            "round_id",
            "iteration_id",
            "rollout_window_id",
            "prepare_count",
            "commit_count",
            "chunk_count",
            "actor_barrier_count",
            "observed_at_utc",
            "source",
        }
        if set(value) != exact:
            raise ContractError("H200 continuation activity keys mismatch")
        return cls(**dict(value))


@dataclass(frozen=True)
class ProductionContinuationSegment:
    """Immutable authorization for one fresh simulator continuation segment.

    A segment never claims to restore Isaac, camera history, recurrent state, or
    RNG state.  It explicitly records that those states start fresh while the
    H200 run, iteration, checkpoint, actor routes, and global window numbering
    continue.  Each segment stops at the next 40-update validation boundary.
    The driver additionally clamps that boundary to the immutable run budget,
    so a post-validation continuation for a 50-window run covers only windows
    40..49 rather than silently extending to window 79.
    """

    segment_id: str
    run_id: str
    config_sha256: str
    start_global_window: int
    expected_checkpoint_sha256: str
    simulator_seed: int
    episode_serial_base: int
    episode_serial_limit_exclusive: int
    reset_reason: str
    predecessor_state_path: str
    predecessor_state_sha256: str
    h200_activity_attestation: H200ContinuationActivityAttestation | None
    h200_activity_evidence_path: str | None
    h200_activity_evidence_sha256: str | None
    prior_continuation_manifest_sha256s: tuple[str, ...]
    schema: str = PRODUCTION_SEGMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_SEGMENT_SCHEMA:
            raise ContractError("production continuation segment schema mismatch")
        require_identifier(self.segment_id, "segment_id", 160)
        require_identifier(self.run_id, "run_id")
        require_sha256(self.config_sha256, "config_sha256")
        require_sha256(
            self.expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
        _require_absolute_path(self.predecessor_state_path, "predecessor_state_path")
        require_sha256(self.predecessor_state_sha256, "predecessor_state_sha256")
        if (
            isinstance(self.start_global_window, bool)
            or not isinstance(self.start_global_window, int)
            or self.start_global_window < 1
        ):
            raise ContractError("continuation start_global_window must be positive")
        if (
            isinstance(self.simulator_seed, bool)
            or not isinstance(self.simulator_seed, int)
            or self.simulator_seed < 0
        ):
            raise ContractError("continuation simulator_seed must be non-negative")
        if (
            isinstance(self.episode_serial_base, bool)
            or not isinstance(self.episode_serial_base, int)
            or self.episode_serial_base < 1_000_000
        ):
            raise ContractError(
                "continuation episode_serial_base must reserve a fresh namespace"
            )
        if (
            isinstance(self.episode_serial_limit_exclusive, bool)
            or not isinstance(self.episode_serial_limit_exclusive, int)
            or self.episode_serial_limit_exclusive - self.episode_serial_base
            < EPISODE_NAMESPACE_MIN_WIDTH
        ):
            raise ContractError(
                "continuation episode namespace must reserve at least "
                f"{EPISODE_NAMESPACE_MIN_WIDTH} serials"
            )
        if self.reset_reason not in {
            "crash_uncertain",
            "post_validation",
            "operator_authorized_fresh_segment",
        }:
            raise ContractError("unsupported continuation reset_reason")
        if not isinstance(self.prior_continuation_manifest_sha256s, tuple):
            raise ContractError("prior continuation manifest SHA-256s must be a tuple")
        for digest in self.prior_continuation_manifest_sha256s:
            require_sha256(digest, "prior_continuation_manifest_sha256")
        if tuple(sorted(set(self.prior_continuation_manifest_sha256s))) != (
            self.prior_continuation_manifest_sha256s
        ):
            raise ContractError(
                "prior continuation manifest SHA-256s must be sorted and unique"
            )

        unplanned = self.reset_reason in {
            "crash_uncertain",
            "operator_authorized_fresh_segment",
        }
        activity = self.h200_activity_attestation
        if activity is not None and not isinstance(
            activity, H200ContinuationActivityAttestation
        ):
            raise ContractError("h200_activity_attestation has the wrong type")
        if unplanned and activity is None:
            raise ContractError(
                "unplanned continuation requires zero-H200-activity attestation"
            )
        if activity is not None:
            if (
                activity.run_id != self.run_id
                or activity.config_sha256 != self.config_sha256
                or activity.round_id != PRODUCTION_ROUND_ID
                or activity.iteration_id != self.start_global_window + 1
                or activity.rollout_window_id
                != f"window-{self.start_global_window:06d}"
            ):
                raise ContractError("H200 activity attestation segment identity mismatch")
            if unplanned and not activity.is_zero:
                raise ContractError(
                    "unplanned continuation has nonzero H200 prepare/commit/chunk activity"
                )
            if self.h200_activity_evidence_path is None:
                raise ContractError("H200 activity attestation lacks an evidence path")
            _require_absolute_path(
                self.h200_activity_evidence_path, "h200_activity_evidence_path"
            )
            require_sha256(
                self.h200_activity_evidence_sha256,
                "h200_activity_evidence_sha256",
            )
        elif (
            self.h200_activity_evidence_path is not None
            or self.h200_activity_evidence_sha256 is not None
        ):
            raise ContractError("H200 activity evidence requires an embedded attestation")

    @property
    def end_global_window_exclusive(self) -> int:
        interval = PRODUCTION_VALIDATION_INTERVAL
        return ((self.start_global_window // interval) + 1) * interval

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "segment_id": self.segment_id,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "start_global_window": self.start_global_window,
            "end_global_window_exclusive": self.end_global_window_exclusive,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "simulator_seed": self.simulator_seed,
            "episode_serial_base": self.episode_serial_base,
            "episode_serial_limit_exclusive": self.episode_serial_limit_exclusive,
            "reset_reason": self.reset_reason,
            "predecessor_state_path": self.predecessor_state_path,
            "predecessor_state_sha256": self.predecessor_state_sha256,
            "h200_activity_attestation": (
                self.h200_activity_attestation.as_dict()
                if self.h200_activity_attestation is not None
                else None
            ),
            "h200_activity_evidence_path": self.h200_activity_evidence_path,
            "h200_activity_evidence_sha256": self.h200_activity_evidence_sha256,
            "prior_continuation_manifest_sha256s": list(
                self.prior_continuation_manifest_sha256s
            ),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProductionContinuationSegment":
        if not isinstance(value, Mapping):
            raise ContractError("production continuation segment must be an object")
        exact = {
            "schema",
            "segment_id",
            "run_id",
            "config_sha256",
            "start_global_window",
            "end_global_window_exclusive",
            "expected_checkpoint_sha256",
            "simulator_seed",
            "episode_serial_base",
            "episode_serial_limit_exclusive",
            "reset_reason",
            "predecessor_state_path",
            "predecessor_state_sha256",
            "h200_activity_attestation",
            "h200_activity_evidence_path",
            "h200_activity_evidence_sha256",
            "prior_continuation_manifest_sha256s",
        }
        if set(value) != exact:
            raise ContractError("production continuation segment keys mismatch")
        result = cls(
            schema=value["schema"],
            segment_id=value["segment_id"],
            run_id=value["run_id"],
            config_sha256=value["config_sha256"],
            start_global_window=value["start_global_window"],
            expected_checkpoint_sha256=value["expected_checkpoint_sha256"],
            simulator_seed=value["simulator_seed"],
            episode_serial_base=value["episode_serial_base"],
            episode_serial_limit_exclusive=value[
                "episode_serial_limit_exclusive"
            ],
            reset_reason=value["reset_reason"],
            predecessor_state_path=value["predecessor_state_path"],
            predecessor_state_sha256=value["predecessor_state_sha256"],
            h200_activity_attestation=(
                H200ContinuationActivityAttestation.from_mapping(
                    value["h200_activity_attestation"]
                )
                if value["h200_activity_attestation"] is not None
                else None
            ),
            h200_activity_evidence_path=value["h200_activity_evidence_path"],
            h200_activity_evidence_sha256=value[
                "h200_activity_evidence_sha256"
            ],
            prior_continuation_manifest_sha256s=tuple(
                value["prior_continuation_manifest_sha256s"]
            ),
        )
        if value["end_global_window_exclusive"] != result.end_global_window_exclusive:
            raise ContractError("continuation validation-boundary endpoint mismatch")
        return result


def _read_regular_file(path: Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ProductionRestartUnsafe(f"{label} must be an existing regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProductionRestartUnsafe(f"could not read {label}: {path}") from exc


def _is_late_checkpoint_advance_predecessor(
    state: Mapping[str, Any],
    segment: ProductionContinuationSegment,
) -> bool:
    """Recognize one fully sealed window whose remote learner recovered later.

    The persistent owner normally advances ``next_window_index`` only after
    ``wait_for_window_advance`` returns.  If H200 reports ``learner_failed`` at
    that point, the local state truthfully remains one cursor behind even
    though every transition, final commit, compact seal, and actor barrier is
    already durable.  H200 may subsequently repair and retry that *same*
    learner job without replaying Isaac.  A fresh simulator continuation is
    safe only for this exact checkpoint-boundary shape; all mid-window and
    partially sealed states continue to fail closed.

    The continuation's separately hash-bound zero-activity evidence and live
    coordinator attestation prove that the next iteration is empty and bound
    to ``expected_checkpoint_sha256``.  This predicate proves the local half:
    the preceding window was complete before the remote-only failure.
    """

    start = segment.start_global_window
    previous_window = start - 1
    spec = state.get("spec")
    pending_window = state.get("pending_window")
    pending_barriers = state.get("pending_barriers")
    completed = state.get("completed_windows")
    if not all(
        isinstance(value, Mapping)
        for value in (spec, pending_window, pending_barriers)
    ) or not isinstance(completed, list):
        return False
    actor_ids = spec.get("actor_ids")
    if not isinstance(actor_ids, list) or len(actor_ids) != PRODUCTION_BATCH_SIZE:
        return False
    if len(set(actor_ids)) != PRODUCTION_BATCH_SIZE or any(
        not isinstance(actor_id, str) for actor_id in actor_ids
    ):
        return False
    state_start = state.get("start_window_index")
    if (
        isinstance(state_start, bool)
        or not isinstance(state_start, int)
        or state_start < 0
        or state_start > previous_window
    ):
        return False
    completed_count = previous_window - state_start
    if len(completed) != completed_count or any(
        not isinstance(row, Mapping)
        or row.get("window_index") != state_start + offset
        for offset, row in enumerate(completed)
    ):
        return False

    rollout_window_id = f"window-{previous_window:06d}"
    input_checkpoint = state.get("current_input_checkpoint_sha256")
    seals = pending_window.get("seal_sha256")
    expected_chunks = [
        f"d{PRODUCTION_ROUND_ID}-{actor_id}-{rollout_window_id}"
        for actor_id in actor_ids
    ]
    return (
        state.get("phase") == "CRASH_UNCERTAIN"
        and state.get("next_window_index") == previous_window
        and state.get("current_iteration_id") == start
        and state.get("running_window_index") == previous_window
        and state.get("running_rollout_window_id") == rollout_window_id
        and state.get("window_checkpoint_sha256") == input_checkpoint
        and state.get("error_type") == "ProductionDriverError"
        and state.get("error_message") == "H200 learner failed closed"
        and state.get("uncertain_reason")
        == (
            f"checkpoint advance after window {previous_window} raised "
            "ProductionDriverError; simulator/recurrent/RNG state cannot be "
            "reconstructed or replayed"
        )
        and pending_window.get("window_index") == previous_window
        and pending_window.get("rollout_window_id") == rollout_window_id
        and pending_window.get("transition_batches") == CHUNK_LENGTH
        and pending_window.get("physical_transition_rows")
        == CHUNK_LENGTH * PRODUCTION_BATCH_SIZE
        and pending_window.get("sealed_lane_count") == PRODUCTION_BATCH_SIZE
        and pending_window.get("remote_acknowledgement_count")
        == PRODUCTION_BATCH_SIZE
        and isinstance(seals, list)
        and len(seals) == PRODUCTION_BATCH_SIZE
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in seals
        )
        and pending_barriers.get("iteration_id") == start
        and pending_barriers.get("checkpoint_sha256") == input_checkpoint
        and pending_barriers.get("actor_ids") == actor_ids
        and pending_barriers.get("chunk_ids") == expected_chunks
        and pending_barriers.get("arrived") == PRODUCTION_BATCH_SIZE
        and pending_barriers.get("expected") == PRODUCTION_BATCH_SIZE
        and pending_barriers.get("ready_for_learning") is True
    )


def verify_predecessor_state_file(
    segment: ProductionContinuationSegment,
    supplied_path: Path,
    *,
    origin_path: str | None = None,
) -> dict[str, Any]:
    """Verify that the segment's predecessor hash names actual state bytes.

    ``origin_path`` is used inside the container, where the host file is mounted
    at a narrow read-only path.  On the host it defaults to the supplied path's
    resolved identity and must exactly equal the path frozen in the manifest.
    """

    supplied_path = Path(supplied_path)
    actual_origin = origin_path or str(supplied_path.resolve(strict=True))
    if actual_origin != segment.predecessor_state_path:
        raise ProductionRestartUnsafe("predecessor-state origin path mismatch")
    raw = _read_regular_file(supplied_path, "predecessor production state")
    if sha256_bytes(raw) != segment.predecessor_state_sha256:
        raise ProductionRestartUnsafe("predecessor production state SHA-256 mismatch")
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProductionRestartUnsafe("predecessor production state is invalid JSON") from exc
    spec = state.get("spec")
    cursor_matches = state.get("next_window_index") == segment.start_global_window
    late_checkpoint_advance = _is_late_checkpoint_advance_predecessor(
        state, segment
    )
    if (
        state.get("schema") != PRODUCTION_STATE_SCHEMA
        or not isinstance(spec, Mapping)
        or spec.get("run_id") != segment.run_id
        or spec.get("config_sha256") != segment.config_sha256
        or (not cursor_matches and not late_checkpoint_advance)
    ):
        raise ProductionRestartUnsafe("predecessor production state identity mismatch")
    expected_phases = (
        {"CRASH_UNCERTAIN"}
        if segment.reset_reason == "crash_uncertain"
        else {"COMPLETE", "CRASH_UNCERTAIN"}
        if segment.reset_reason == "operator_authorized_fresh_segment"
        else {"COMPLETE"}
    )
    if state.get("phase") not in expected_phases:
        raise ProductionRestartUnsafe(
            "predecessor production state phase is incompatible with reset reason"
        )
    return state


def verify_h200_activity_evidence_file(
    segment: ProductionContinuationSegment,
    supplied_path: Path | None,
    *,
    origin_path: str | None = None,
) -> H200ContinuationActivityAttestation | None:
    """Verify the exact H200 ledger snapshot embedded in a segment manifest."""

    expected = segment.h200_activity_attestation
    if expected is None:
        if supplied_path is not None:
            raise ProductionRestartUnsafe("unexpected H200 activity evidence file")
        return None
    if supplied_path is None:
        raise ProductionRestartUnsafe("H200 activity evidence file was not supplied")
    supplied_path = Path(supplied_path)
    actual_origin = origin_path or str(supplied_path.resolve(strict=True))
    if actual_origin != segment.h200_activity_evidence_path:
        raise ProductionRestartUnsafe("H200 activity evidence origin path mismatch")
    raw = _read_regular_file(supplied_path, "H200 activity evidence")
    if sha256_bytes(raw) != segment.h200_activity_evidence_sha256:
        raise ProductionRestartUnsafe("H200 activity evidence SHA-256 mismatch")
    try:
        observed = H200ContinuationActivityAttestation.from_mapping(json.loads(raw))
    except (json.JSONDecodeError, ContractError) as exc:
        raise ProductionRestartUnsafe("H200 activity evidence is invalid") from exc
    if observed != expected:
        raise ProductionRestartUnsafe("H200 activity evidence does not match the manifest")
    return observed


def _legacy_episode_namespace(value: Mapping[str, Any]) -> tuple[int, int]:
    base = value.get("episode_serial_base")
    if isinstance(base, bool) or not isinstance(base, int) or base < 0:
        raise ProductionRestartUnsafe("prior continuation has an invalid episode namespace")
    limit = value.get("episode_serial_limit_exclusive")
    if limit is None and value.get("schema") == LEGACY_PRODUCTION_SEGMENT_SCHEMA:
        # v1 had only a base.  Conservatively reserve the same 100k slot used by
        # its manifest generator, so no future manifest can overlap it.
        limit = base + EPISODE_NAMESPACE_MIN_WIDTH
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= base:
        raise ProductionRestartUnsafe("prior continuation has an invalid episode limit")
    return base, limit


def verify_continuation_history(
    segment: ProductionContinuationSegment,
    current_manifest_path: Path,
    history_dir: Path,
) -> tuple[str, ...]:
    """Prove the new episode namespace is disjoint from every prior segment.

    The manifest freezes the exact sorted SHA set it was checked against.  A
    later manifest appearing in the directory therefore fails closed rather
    than silently invalidating the collision audit.
    """

    current_manifest_path = Path(current_manifest_path)
    history_dir = Path(history_dir)
    if current_manifest_path.is_symlink() or not current_manifest_path.is_file():
        raise ProductionRestartUnsafe("current continuation manifest is not a regular file")
    try:
        current_real = current_manifest_path.resolve(strict=True)
        history_real = history_dir.resolve(strict=True)
    except OSError as exc:
        raise ProductionRestartUnsafe("continuation history path cannot be resolved") from exc
    if current_real.parent != history_real:
        raise ProductionRestartUnsafe("continuation manifest is outside its history directory")

    prior_digests: list[str] = []
    prior_ranges: list[tuple[int, int, Path]] = []
    for path in sorted(history_real.glob("*.json")):
        if path.resolve(strict=True) == current_real:
            continue
        if path.is_symlink() or not path.is_file():
            raise ProductionRestartUnsafe("continuation history contains a non-regular file")
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProductionRestartUnsafe(
                f"continuation history JSON is invalid: {path.name}"
            ) from exc
        if not isinstance(value, Mapping) or value.get("run_id") != segment.run_id:
            continue
        if value.get("schema") not in {
            PRODUCTION_SEGMENT_SCHEMA,
            LEGACY_PRODUCTION_SEGMENT_SCHEMA,
        }:
            continue
        prior_digests.append(sha256_bytes(raw))
        start, limit = _legacy_episode_namespace(value)
        prior_ranges.append((start, limit, path))

    actual = tuple(sorted(prior_digests))
    if actual != segment.prior_continuation_manifest_sha256s:
        raise ProductionRestartUnsafe(
            "continuation history SHA-256 set differs from the frozen manifest"
        )
    new_start = segment.episode_serial_base
    new_limit = segment.episode_serial_limit_exclusive
    for prior_start, prior_limit, path in prior_ranges:
        if new_start < prior_limit and prior_start < new_limit:
            raise ProductionRestartUnsafe(
                f"continuation episode namespace overlaps prior manifest {path.name}"
            )
    return actual


def verify_episode_namespace_capacity(
    segment: ProductionContinuationSegment,
    initial_episode_serials: Iterable[int],
) -> None:
    """Prove every possible per-lane serial in this segment stays reserved."""

    serials = tuple(initial_episode_serials)
    if not serials or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in serials
    ):
        raise ProductionRestartUnsafe("native initial episode serials are invalid")
    segment_steps = (
        segment.end_global_window_exclusive - segment.start_global_window
    ) * CHUNK_LENGTH
    required_limit = (
        segment.episode_serial_base + max(serials) + segment_steps + 1
    )
    if required_limit > segment.episode_serial_limit_exclusive:
        raise ProductionRestartUnsafe(
            "continuation episode namespace cannot contain worst-case native resets"
        )


def resolve_production_simulator_seed(
    configured_seed: Any,
    launched_seed: Any,
    continuation: ProductionContinuationSegment | None,
    window_index: int,
) -> int:
    """Bind a native segment to either the run seed or its fresh continuation seed."""

    for value, name in (
        (configured_seed, "configured simulator seed"),
        (launched_seed, "launched simulator seed"),
        (window_index, "window index"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"{name} must be a non-negative integer")
    if continuation is None:
        if window_index != 0 or launched_seed != configured_seed:
            raise ContractError("initial production simulator seed mismatch")
        return configured_seed
    if continuation.start_global_window != window_index:
        raise ContractError("continuation simulator seed window mismatch")
    if launched_seed != continuation.simulator_seed:
        raise ContractError("continuation simulator seed mismatch")
    return continuation.simulator_seed


@dataclass(frozen=True)
class ProductionRunSpec:
    """Immutable identity and cardinality of one production training budget."""

    run_id: str
    config_sha256: str
    actor_prefix: str
    actor_ids: tuple[str, ...]
    initial_checkpoint_sha256: str
    round_id: int = PRODUCTION_ROUND_ID
    beta: float = PRODUCTION_BETA
    num_envs: int = PRODUCTION_BATCH_SIZE
    total_windows: int = PRODUCTION_WINDOW_COUNT
    rollout_steps: int = CHUNK_LENGTH
    wire_contract_version: str = PRODUCTION_WIRE_CONTRACT
    physical_gpu_uuid: str = RTX_PRO_6000_GPU1_UUID

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        require_identifier(self.actor_prefix, "actor_prefix")
        require_sha256(self.config_sha256, "config_sha256")
        require_sha256(self.initial_checkpoint_sha256, "initial_checkpoint_sha256")
        if self.round_id != PRODUCTION_ROUND_ID or self.beta != ROUND_BETAS[1]:
            raise ContractError("production is frozen to round_id=1 and beta=0.75")
        if self.num_envs != PRODUCTION_BATCH_SIZE:
            raise ContractError("production is frozen to exactly B=28")
        if (
            isinstance(self.total_windows, bool)
            or not isinstance(self.total_windows, int)
            or self.total_windows < PRODUCTION_VALIDATION_INTERVAL
        ):
            raise ContractError(
                "production total_windows must be an integer at least as large "
                "as the 40-iteration validation interval"
            )
        if self.rollout_steps != CHUNK_LENGTH:
            raise ContractError("each production window must contain exactly 81 steps")
        if self.wire_contract_version not in PRODUCTION_WIRE_CONTRACTS:
            raise ContractError("production requires a served lean v3 or v4 boundary")
        if self.physical_gpu_uuid != RTX_PRO_6000_GPU1_UUID:
            raise ContractError("production is pinned to the dedicated physical RTX GPU1 UUID")
        expected = tuple(
            f"{self.actor_prefix}-lane-{lane:03d}" for lane in range(self.num_envs)
        )
        if tuple(self.actor_ids) != expected:
            raise ContractError("actor_ids must exactly equal actor_prefix-lane-000..027")
        for actor_id in self.actor_ids:
            require_identifier(actor_id, "actor_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "actor_prefix": self.actor_prefix,
            "actor_ids": list(self.actor_ids),
            "initial_checkpoint_sha256": self.initial_checkpoint_sha256,
            "round_id": self.round_id,
            "beta": self.beta,
            "num_envs": self.num_envs,
            "total_windows": self.total_windows,
            "rollout_steps": self.rollout_steps,
            "wire_contract_version": self.wire_contract_version,
            "physical_gpu_uuid": self.physical_gpu_uuid,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()))

    @property
    def validation_interval_windows(self) -> int:
        """Canonical Raven fixed-panel cadence, independent of run budget."""

        return PRODUCTION_VALIDATION_INTERVAL


@dataclass(frozen=True)
class InitialProductionGate:
    """H200 attestation immediately before the first native window."""

    run_id: str
    config_sha256: str
    round_id: int
    active_iteration_id: int
    active_checkpoint_sha256: str
    actor_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class WindowAdvanceGate:
    """H200 proof that one iteration trained and the next gate is active.

    ``output_checkpoint_sha256`` is the completed iteration's immutable
    candidate. For windows 0..38 it is also the next iteration's input. At
    window 39 the controller intentionally keeps the run's active checkpoint
    unchanged and moves to ``awaiting_validation``; this field still binds
    iteration 40's candidate.
    """

    run_id: str
    config_sha256: str
    round_id: int
    completed_window_index: int
    completed_iteration_id: int
    next_iteration_id: int | None
    output_checkpoint_sha256: str
    barrier_count: int
    actor_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class BarrierGate:
    """H200 proof that all actor barriers for one window were accepted."""

    run_id: str
    round_id: int
    completed_window_index: int
    iteration_id: int
    checkpoint_sha256: str
    actor_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    arrived: int
    expected: int
    ready_for_learning: bool


class ProductionControl(Protocol):
    """Authenticated H200 controller adapter used by the production driver.

    Implementations must keep polling internally across ordinary service
    unavailability.  Returning means the requested gate is durably satisfied;
    raising is terminal and causes the RTX driver to become CRASH_UNCERTAIN.
    """

    def attest_initial(self, spec: ProductionRunSpec) -> InitialProductionGate:
        ...

    def attest_continuation(
        self,
        spec: ProductionRunSpec,
        segment: ProductionContinuationSegment,
    ) -> InitialProductionGate:
        ...

    def post_window_barriers(
        self,
        spec: ProductionRunSpec,
        *,
        completed_window_index: int,
        input_checkpoint_sha256: str,
    ) -> BarrierGate:
        ...

    def wait_for_window_advance(
        self,
        spec: ProductionRunSpec,
        *,
        completed_window_index: int,
        previous_checkpoint_sha256: str,
    ) -> WindowAdvanceGate:
        ...


class ProductionMultiWindowDriver:
    """Run one persistent Raven segment to validation or budget boundary."""

    def __init__(
        self,
        runner: Any,
        control: ProductionControl,
        runtime_root: Path,
        spec: ProductionRunSpec,
        *,
        continuation: ProductionContinuationSegment | None = None,
        stop_at_learner_barrier: bool = False,
        clock_ns=time.time_ns,
        process_instance_id: str | None = None,
    ) -> None:
        self.runner = runner
        self.control = control
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.runtime_root / "production-driver-state.json"
        self.spec = spec
        self.continuation = continuation
        self.stop_at_learner_barrier = stop_at_learner_barrier
        if not isinstance(stop_at_learner_barrier, bool):
            raise ContractError("stop_at_learner_barrier must be bool")
        if stop_at_learner_barrier and continuation is not None:
            raise ContractError(
                "one-window learner-barrier smoke must start from a fresh H200 run"
            )
        if continuation is not None and (
            continuation.run_id != spec.run_id
            or continuation.config_sha256 != spec.config_sha256
        ):
            raise ContractError("continuation does not bind the production run/config")
        self.start_window_index = (
            continuation.start_global_window if continuation is not None else 0
        )
        self.end_window_index_exclusive = (
            min(continuation.end_global_window_exclusive, spec.total_windows)
            if continuation is not None
            else min(spec.validation_interval_windows, spec.total_windows)
        )
        if self.start_window_index >= self.end_window_index_exclusive:
            raise ContractError(
                "production segment starts at or beyond the immutable window budget"
            )
        self.clock_ns = clock_ns
        self.process_instance_id = process_instance_id or (
            f"pid-{os.getpid()}-{uuid.uuid4().hex}"
        )
        require_identifier(self.process_instance_id, "process_instance_id", 160)
        self._completed_reopen = False
        self._validate_runner()
        if self.state_path.exists():
            self.state = self._load_existing()
            if self.state["phase"] == "COMPLETE":
                self._completed_reopen = True
            else:
                prior_phase = self.state["phase"]
                self.state["phase"] = "CRASH_UNCERTAIN"
                self.state["uncertain_reason"] = (
                    "production driver reopened in a new process from phase "
                    f"{prior_phase}; no attested Raven/Isaac/recurrent/RNG restore exists"
                )
                self.state["reopened_process_instance_id"] = self.process_instance_id
                self.state["updated_time_ns"] = int(self.clock_ns())
                self._persist()
                raise ProductionRestartUnsafe(self.state["uncertain_reason"])
        else:
            now = int(self.clock_ns())
            self.state = {
                "schema": PRODUCTION_STATE_SCHEMA,
                "spec": self.spec.as_dict(),
                "spec_sha256": self.spec.digest,
                "process_instance_id": self.process_instance_id,
                "phase": "CREATED",
                "next_window_index": self.start_window_index,
                "current_iteration_id": None,
                "current_input_checkpoint_sha256": None,
                "final_candidate_checkpoint_sha256": None,
                "completed_windows": [],
                "execution_segment": (
                    self.continuation.as_dict()
                    if self.continuation is not None
                    else None
                ),
                "execution_segment_sha256": (
                    self.continuation.digest
                    if self.continuation is not None
                    else None
                ),
                "stop_at_learner_barrier": self.stop_at_learner_barrier,
                "start_window_index": self.start_window_index,
                "end_window_index_exclusive": self.end_window_index_exclusive,
                "simulator_initialization_count": 1,
                "rollout_lifecycle_initialization_count": 0,
                "uncertain_reason": None,
                "created_time_ns": now,
                "updated_time_ns": now,
            }
            self._persist()

    def _validate_runner(self) -> None:
        if int(getattr(self.runner, "num_envs", -1)) != self.spec.num_envs:
            raise ContractError("production runner batch size does not match spec")
        if getattr(self.runner, "run_id", None) != self.spec.run_id:
            raise ContractError("production runner run_id does not match spec")
        if int(getattr(self.runner, "round_id", -1)) != self.spec.round_id:
            raise ContractError("production runner round_id does not match spec")
        if getattr(self.runner, "actor_prefix", None) != self.spec.actor_prefix:
            raise ContractError("production runner actor_prefix does not match spec")
        lifecycle = getattr(self.runner, "lifecycle", None)
        if lifecycle is None or not callable(getattr(lifecycle, "begin", None)) or not callable(
            getattr(lifecycle, "finish", None)
        ):
            raise ContractError("production runner must expose one persistent rollout lifecycle")

    def _load_existing(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductionRestartUnsafe("production driver state is unreadable") from exc
        if value.get("schema") != PRODUCTION_STATE_SCHEMA:
            raise ProductionRestartUnsafe("production driver state schema mismatch")
        if value.get("spec_sha256") != self.spec.digest or value.get("spec") != self.spec.as_dict():
            raise ProductionRestartUnsafe("production driver state belongs to another spec")
        expected_segment = (
            self.continuation.as_dict() if self.continuation is not None else None
        )
        expected_segment_sha = (
            self.continuation.digest if self.continuation is not None else None
        )
        if (
            value.get("execution_segment") != expected_segment
            or value.get("execution_segment_sha256") != expected_segment_sha
            or value.get("start_window_index") != self.start_window_index
            or value.get("end_window_index_exclusive")
            != self.end_window_index_exclusive
            or value.get("stop_at_learner_barrier")
            != self.stop_at_learner_barrier
        ):
            raise ProductionRestartUnsafe(
                "production driver state belongs to another execution segment"
            )
        return value

    def _persist(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _transition(self, phase: str, **updates: Any) -> None:
        self.state.update(updates)
        self.state["phase"] = phase
        self.state["updated_time_ns"] = int(self.clock_ns())
        self._persist()

    def _fail_closed(self, where: str, exc: BaseException) -> None:
        self._transition(
            "CRASH_UNCERTAIN",
            uncertain_reason=(
                f"{where} raised {type(exc).__name__}; simulator/recurrent/RNG state "
                "cannot be reconstructed or replayed"
            ),
            error_type=type(exc).__name__,
            error_message=str(exc)[:512],
        )

    def _validate_initial_gate(self, gate: InitialProductionGate) -> None:
        expected_iteration = self.start_window_index + 1
        expected_checkpoint = (
            self.continuation.expected_checkpoint_sha256
            if self.continuation is not None
            else self.spec.initial_checkpoint_sha256
        )
        if not isinstance(gate, InitialProductionGate):
            raise ProductionDriverError("control returned an invalid initial gate type")
        if (
            gate.run_id != self.spec.run_id
            or gate.config_sha256 != self.spec.config_sha256
            or gate.round_id != self.spec.round_id
            or gate.active_iteration_id != expected_iteration
            or tuple(gate.actor_ids) != self.spec.actor_ids
            or gate.status != "collecting"
        ):
            raise ProductionDriverError("H200 initial production identity mismatch")
        if gate.active_checkpoint_sha256 != expected_checkpoint:
            raise ProductionDriverError("H200 initial checkpoint mismatch")

    def _expected_chunk_ids(self, completed_window_index: int) -> tuple[str, ...]:
        return tuple(
            f"d{self.spec.round_id}-{actor_id}-window-{completed_window_index:06d}"
            for actor_id in self.spec.actor_ids
        )

    def _validate_barrier_gate(
        self,
        gate: BarrierGate,
        *,
        completed_window_index: int,
        input_checkpoint_sha256: str,
    ) -> None:
        if not isinstance(gate, BarrierGate):
            raise ProductionDriverError("control returned an invalid barrier gate type")
        if (
            gate.run_id != self.spec.run_id
            or gate.round_id != self.spec.round_id
            or gate.completed_window_index != completed_window_index
            or gate.iteration_id != completed_window_index + 1
            or gate.checkpoint_sha256 != input_checkpoint_sha256
            or tuple(gate.actor_ids) != self.spec.actor_ids
            or tuple(gate.chunk_ids) != self._expected_chunk_ids(completed_window_index)
            or gate.arrived != self.spec.num_envs
            or gate.expected != self.spec.num_envs
            or gate.ready_for_learning is not True
        ):
            raise ProductionDriverError("H200 actor-barrier identity mismatch")

    def _validate_advance_gate(
        self,
        gate: WindowAdvanceGate,
        *,
        completed_window_index: int,
        previous_checkpoint_sha256: str,
    ) -> None:
        if not isinstance(gate, WindowAdvanceGate):
            raise ProductionDriverError("control returned an invalid advance gate type")
        expected_completed_iteration = completed_window_index + 1
        final_window = completed_window_index + 1 == self.end_window_index_exclusive
        validation_boundary = (
            expected_completed_iteration % self.spec.validation_interval_windows == 0
        )
        budget_boundary = expected_completed_iteration == self.spec.total_windows
        if final_window and validation_boundary:
            expected_next_iteration = None
            valid_statuses = {"awaiting_validation", "validating"}
        elif final_window and budget_boundary:
            # The learner has advanced the exact final requested window.  The
            # coordinator may expose an empty next collecting iteration, but
            # this immutable RTX owner must not prepare it.
            expected_next_iteration = expected_completed_iteration + 1
            valid_statuses = {"collecting"}
        elif final_window:
            raise ProductionDriverError(
                "production segment endpoint is neither validation nor budget"
            )
        else:
            expected_next_iteration = expected_completed_iteration + 1
            valid_statuses = {"collecting"}
        if (
            gate.run_id != self.spec.run_id
            or gate.config_sha256 != self.spec.config_sha256
            or gate.round_id != self.spec.round_id
            or gate.completed_window_index != completed_window_index
            or gate.completed_iteration_id != expected_completed_iteration
            or gate.next_iteration_id != expected_next_iteration
            or gate.barrier_count != self.spec.num_envs
            or tuple(gate.actor_ids) != self.spec.actor_ids
            or gate.status not in valid_statuses
        ):
            raise ProductionDriverError("H200 window/checkpoint advance identity mismatch")
        require_sha256(gate.output_checkpoint_sha256, "output_checkpoint_sha256")
        if gate.output_checkpoint_sha256 == previous_checkpoint_sha256:
            raise ProductionDriverError("H200 did not advance to a new immutable checkpoint")

    def _validate_window_result(self, window_index: int, result: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise ProductionDriverError("window runner returned a non-object")
        if (
            result.get("batch_size") != self.spec.num_envs
            or result.get("window_index") != window_index
            or result.get("transition_batches") != CHUNK_LENGTH
            or result.get("physical_transition_rows") != CHUNK_LENGTH * self.spec.num_envs
        ):
            raise ProductionDriverError("window runner returned the wrong cardinality")
        final = result.get("final")
        if not isinstance(final, Mapping) or final.get("phase") != "COMPLETE":
            raise ProductionDriverError("window runner lacks a COMPLETE final batch")
        step_keys = final.get("step_keys")
        commits = final.get("local_commits")
        acknowledgements = final.get("remote_acknowledgements")
        if not isinstance(step_keys, Sequence) or len(step_keys) != self.spec.num_envs:
            raise ProductionDriverError("final window lacks B step identities")
        if not isinstance(commits, Sequence) or len(commits) != self.spec.num_envs:
            raise ProductionDriverError("final window lacks B local commits")
        if not isinstance(acknowledgements, Sequence) or len(acknowledgements) != self.spec.num_envs:
            raise ProductionDriverError("final window lacks B remote acknowledgements")
        expected_window_id = f"window-{window_index:06d}"
        for lane, raw in enumerate(step_keys):
            if not isinstance(raw, Mapping):
                raise ProductionDriverError("final step identity is malformed")
            if (
                raw.get("run_id") != self.spec.run_id
                or raw.get("round_id") != self.spec.round_id
                or raw.get("actor_id") != self.spec.actor_ids[lane]
                or raw.get("rollout_window_id") != expected_window_id
                or raw.get("rollout_step_index") != CHUNK_LENGTH - 1
            ):
                raise ProductionDriverError("final step identity is not the expected window/lane")
        if any(not isinstance(item, Mapping) or item.get("sealed") is not True for item in commits):
            raise ProductionDriverError("every lane must be locally sealed at index 80")
        seal_sha256 = []
        for item in commits:
            seal = item.get("seal")
            if not isinstance(seal, Mapping):
                raise ProductionDriverError("sealed lane lacks its audit manifest")
            seal_sha256.append(require_sha256(seal.get("sha256"), "seal.sha256"))
        summary = {
            "window_index": window_index,
            "rollout_window_id": expected_window_id,
            "transition_batches": CHUNK_LENGTH,
            "physical_transition_rows": CHUNK_LENGTH * self.spec.num_envs,
            "sealed_lane_count": self.spec.num_envs,
            "remote_acknowledgement_count": self.spec.num_envs,
            "seal_sha256": seal_sha256,
            "result_sha256": sha256_bytes(canonical_json(result)),
        }
        return summary

    def status(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.state))

    def run(self) -> dict[str, Any]:
        """Run this fresh segment to its next validation boundary."""

        if self._completed_reopen:
            return self.status()
        if self.state["phase"] != "CREATED":
            raise ProductionRestartUnsafe(
                f"production driver cannot run from durable phase {self.state['phase']}"
            )
        try:
            self._transition("INITIAL_ATTESTING")
            initial = (
                self.control.attest_continuation(self.spec, self.continuation)
                if self.continuation is not None
                else self.control.attest_initial(self.spec)
            )
            self._validate_initial_gate(initial)
        except BaseException as exc:
            self._fail_closed("initial H200 attestation", exc)
            raise
        active_checkpoint = initial.active_checkpoint_sha256
        self._transition(
            "READY",
            current_iteration_id=self.start_window_index + 1,
            current_input_checkpoint_sha256=active_checkpoint,
        )

        lifecycle_started = False
        primary_exc: BaseException | None = None
        try:
            # One recurrent/history lifecycle spans this segment only.  At a
            # validation boundary Isaac closes so the fixed-panel evaluator can
            # own GPU1; an attested fresh segment resumes afterward. Episode
            # resets still flow through Raven's _process_env_step.
            lifecycle_started = True
            self.runner.lifecycle.begin()
            self._transition("READY", rollout_lifecycle_initialization_count=1)

            for window_index in range(
                self.start_window_index, self.end_window_index_exclusive
            ):
                if self.state["next_window_index"] != window_index:
                    exc = ProductionDriverError("durable next-window cursor is not sequential")
                    self._fail_closed("pre-window cursor validation", exc)
                    raise exc
                telemetry = getattr(self.runner, "phase_telemetry", None)
                if telemetry is not None and hasattr(telemetry, "set_vector_step_offset"):
                    telemetry.set_vector_step_offset(window_index * CHUNK_LENGTH)
                self._transition(
                    "WINDOW_RUNNING",
                    running_window_index=window_index,
                    running_rollout_window_id=f"window-{window_index:06d}",
                    window_checkpoint_sha256=active_checkpoint,
                )
                try:
                    result = self.runner.run_window(window_index, manage_lifecycle=False)
                    window_summary = self._validate_window_result(window_index, result)
                except BaseException as exc:
                    self._fail_closed(f"window {window_index}", exc)
                    raise

                self._transition(
                    "POSTING_ACTOR_BARRIERS",
                    pending_window=window_summary,
                )
                try:
                    barrier_gate = self.control.post_window_barriers(
                        self.spec,
                        completed_window_index=window_index,
                        input_checkpoint_sha256=active_checkpoint,
                    )
                    self._validate_barrier_gate(
                        barrier_gate,
                        completed_window_index=window_index,
                        input_checkpoint_sha256=active_checkpoint,
                    )
                except BaseException as exc:
                    self._fail_closed(f"actor barriers after window {window_index}", exc)
                    raise

                self._transition(
                    "WAITING_FOR_CHECKPOINT_ADVANCE",
                    pending_barriers={
                        "iteration_id": barrier_gate.iteration_id,
                        "checkpoint_sha256": barrier_gate.checkpoint_sha256,
                        "actor_ids": list(barrier_gate.actor_ids),
                        "chunk_ids": list(barrier_gate.chunk_ids),
                        "arrived": barrier_gate.arrived,
                        "expected": barrier_gate.expected,
                        "ready_for_learning": barrier_gate.ready_for_learning,
                    },
                )
                if self.stop_at_learner_barrier:
                    completed = [*self.state["completed_windows"]]
                    completed.append(
                        {
                            **window_summary,
                            "input_checkpoint_sha256": self.state[
                                "window_checkpoint_sha256"
                            ],
                            "output_checkpoint_sha256": None,
                            "completed_iteration_id": barrier_gate.iteration_id,
                            "next_iteration_id": None,
                            "barrier_count": barrier_gate.arrived,
                            "barrier_chunk_ids": list(barrier_gate.chunk_ids),
                        }
                    )
                    self._transition(
                        "READY_FOR_LEARNING",
                        next_window_index=window_index + 1,
                        current_iteration_id=barrier_gate.iteration_id,
                        current_input_checkpoint_sha256=active_checkpoint,
                        completed_windows=completed,
                        pending_window=None,
                        running_window_index=None,
                        running_rollout_window_id=None,
                        window_checkpoint_sha256=None,
                    )
                    return self.status()
                try:
                    gate = self.control.wait_for_window_advance(
                        self.spec,
                        completed_window_index=window_index,
                        previous_checkpoint_sha256=active_checkpoint,
                    )
                    self._validate_advance_gate(
                        gate,
                        completed_window_index=window_index,
                        previous_checkpoint_sha256=active_checkpoint,
                    )
                except BaseException as exc:
                    self._fail_closed(f"checkpoint advance after window {window_index}", exc)
                    raise

                output_checkpoint = gate.output_checkpoint_sha256
                completed = [*self.state["completed_windows"]]
                completed.append(
                    {
                        **window_summary,
                        "input_checkpoint_sha256": self.state["window_checkpoint_sha256"],
                        "output_checkpoint_sha256": output_checkpoint,
                        "completed_iteration_id": gate.completed_iteration_id,
                        "next_iteration_id": gate.next_iteration_id,
                        "barrier_count": gate.barrier_count,
                        "barrier_chunk_ids": list(barrier_gate.chunk_ids),
                    }
                )
                final_window = window_index + 1 == self.end_window_index_exclusive
                if not final_window or gate.next_iteration_id is not None:
                    active_checkpoint = output_checkpoint
                self._transition(
                    "COMPLETE" if final_window else "READY",
                    next_window_index=window_index + 1,
                    current_iteration_id=(
                        gate.next_iteration_id
                        if gate.next_iteration_id is not None
                        else gate.completed_iteration_id
                    ),
                    current_input_checkpoint_sha256=active_checkpoint,
                    final_candidate_checkpoint_sha256=(
                        output_checkpoint if final_window else None
                    ),
                    completed_windows=completed,
                    pending_window=None,
                    pending_barriers=None,
                    running_window_index=None,
                    running_rollout_window_id=None,
                    window_checkpoint_sha256=None,
                )
        except BaseException as exc:
            primary_exc = exc
            if self.state.get("phase") != "CRASH_UNCERTAIN":
                self._fail_closed("persistent rollout lifecycle", exc)
            raise
        finally:
            if lifecycle_started:
                try:
                    self.runner.lifecycle.finish()
                except BaseException as finish_exc:
                    if primary_exc is None:
                        self._fail_closed("persistent rollout lifecycle cleanup", finish_exc)
                        raise
                    self.state["cleanup_error_type"] = type(finish_exc).__name__
                    self.state["cleanup_error_message"] = str(finish_exc)[:512]
                    self.state["updated_time_ns"] = int(self.clock_ns())
                    self._persist()

        if (
            self.state["phase"] != "COMPLETE"
            or self.state["next_window_index"] != self.end_window_index_exclusive
            or len(self.state["completed_windows"])
            != self.end_window_index_exclusive - self.start_window_index
        ):
            raise ProductionDriverError("production driver did not terminate at exact cardinality")
        return self.status()
