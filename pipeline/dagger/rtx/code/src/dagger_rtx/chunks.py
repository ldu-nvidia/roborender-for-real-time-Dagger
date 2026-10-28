from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .backend import Capture, TransitionResult
from .contracts import (
    CONTRACT_VERSION,
    RAVEN_VIEW_ORDER,
    StepKey,
    require_identifier,
    require_sha256,
)
from .durable import atomic_write_json, fsync_directory

CHUNK_LENGTH = 81

# RTX is an audit/cross-check owner, not the learner-chunk owner.  These are
# the only arrays needed to prove step identity, outcomes, action selection,
# and the teacher/Homie construction invariant.  In particular, camera,
# depth, control, and policy-observation tensors are deliberately absent: the
# H200 render/student ledger already persists the authoritative learner chunk.
COMPACT_AUDIT_ROW_FIELDS = (
    "gt_actions",
    "homie_actions",
    "behavior_actions",
    "behavior_source",
    "dones",
    "timeouts",
    "rewards",
    "episode_ids",
    "episode_step_indices",
    "rollout_step_indices",
    "round_ids",
    "env_id_per_step",
    "seq_per_step",
    "observation_source",
    "prompt_per_step",
    "prepare_receipts",
    "capture_digests",
    "selection_ids",
    "selection_receipts",
    "commit_digests",
    "commit_receipts",
)

_REQUIRED_LEGACY_AUDIT_ROW_FIELDS = frozenset(COMPACT_AUDIT_ROW_FIELDS) - {
    # Rows produced before the compact-audit revision did not retain this
    # H200 receipt locally.  Selective sealing remains compatible with those
    # completed rows without reading their bulky observation tensors.
    "selection_receipts"
}


class ChunkError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CurrentRoundChunk:
    """One 81-transition rollout window, potentially spanning episodes."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        round_id: int,
        actor_id: str,
        rollout_window_id: str,
    ):
        self.run_id = require_identifier(run_id, "run_id")
        if round_id not in (1, 2, 3, 4):
            raise ChunkError("round_id must be in [1,4]")
        self.round_id = round_id
        self.actor_id = require_identifier(actor_id, "actor_id")
        self.rollout_window_id = require_identifier(rollout_window_id, "rollout_window_id")
        self.root = root / self.run_id / f"D{round_id}" / self.actor_id / self.rollout_window_id
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.rows_dir = self.root / "rows"
        self.rows_dir.mkdir(exist_ok=True, mode=0o700)
        self.sealed_path = self.root / "sealed.npz"

    @property
    def count(self) -> int:
        return len(list(self.rows_dir.glob("step-*.npz")))

    def append(
        self,
        key: StepKey,
        capture: Capture,
        selected_action: np.ndarray,
        selected_source: str,
        result: TransitionResult,
        audit: Mapping[str, str],
    ) -> None:
        if (
            key.run_id != self.run_id
            or key.round_id != self.round_id
            or key.actor_id != self.actor_id
            or key.rollout_window_id != self.rollout_window_id
        ):
            raise ChunkError("step key does not belong to this current-round rollout window")
        if key.rollout_step_index != self.count:
            raise ChunkError(f"window expected step {self.count}, got {key.rollout_step_index}")
        if self.sealed_path.exists():
            raise ChunkError("rollout window is already sealed")
        row = self.rows_dir / f"step-{key.rollout_step_index:03d}.npz"
        arrays = self._compact_row_arrays(
            key,
            capture,
            selected_action,
            selected_source,
            result,
            audit,
        )
        self._write_npz_atomic(row, arrays)

    @staticmethod
    def _compact_row_arrays(
        key: StepKey,
        capture: Capture,
        selected_action: np.ndarray,
        selected_source: str,
        result: TransitionResult,
        audit: Mapping[str, str],
    ) -> dict[str, np.ndarray]:
        source_code = 0 if selected_source == "teacher" else 1
        expected_audit = {
            "prepare_receipt",
            "capture_digest",
            "selection_id",
            "selection_receipt",
            "commit_digest",
            "commit_receipt",
        }
        legacy_audit = expected_audit - {"selection_receipt"}
        if frozenset(audit) not in {frozenset(expected_audit), frozenset(legacy_audit)}:
            raise ChunkError(
                "audit keys must be the compact receipt set (selection_receipt "
                "may be absent only for a legacy scalar caller)"
            )
        for name in set(audit) - {"selection_id"}:
            require_sha256(audit[name], name)
        require_identifier(audit["selection_id"], "selection_id")
        arrays: dict[str, np.ndarray] = {
            "gt_actions": capture.arrays["teacher_action"],
            "homie_actions": capture.arrays["homie_action"],
            "behavior_actions": selected_action,
            "behavior_source": np.asarray(source_code, dtype=np.uint8),
            "dones": np.asarray(result.done, dtype=np.bool_),
            "timeouts": np.asarray(result.timeout, dtype=np.bool_),
            "rewards": np.asarray(result.reward, dtype=np.float32),
            "episode_ids": np.asarray(key.episode_id),
            "episode_step_indices": np.asarray(key.episode_step_index, dtype=np.int64),
            "rollout_step_indices": np.asarray(key.rollout_step_index, dtype=np.int64),
            "round_ids": np.asarray(key.round_id, dtype=np.int64),
            "env_id_per_step": np.asarray(capture.metadata["env_id"], dtype=np.int64),
            "seq_per_step": np.asarray(capture.metadata["seq"], dtype=np.int64),
            "observation_source": np.asarray(capture.metadata["observation_source"]),
            "prompt_per_step": np.asarray(capture.metadata["prompt"]),
            "prepare_receipts": np.asarray(audit["prepare_receipt"]),
            "capture_digests": np.asarray(audit["capture_digest"]),
            "selection_ids": np.asarray(audit["selection_id"]),
            "commit_digests": np.asarray(audit["commit_digest"]),
            "commit_receipts": np.asarray(audit["commit_receipt"]),
            "selection_receipts": np.asarray(audit.get("selection_receipt", "")),
        }
        if set(arrays) != set(COMPACT_AUDIT_ROW_FIELDS):
            raise AssertionError("compact RTX audit row field drift")
        return arrays

    @staticmethod
    def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp, 0o600)
            os.replace(temp, path)
            fsync_directory(path.parent)
        finally:
            if temp.exists():
                temp.unlink()

    def seal(self, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
        if self.count != CHUNK_LENGTH:
            raise ChunkError(f"refusing to seal {self.count}; exactly {CHUNK_LENGTH} committed transitions required")
        if self.sealed_path.exists():
            manifest = json.loads((self.root / "manifest.json").read_text())
            if sha256_file(self.sealed_path) != manifest["sha256"]:
                raise ChunkError("existing seal hash mismatch")
            return manifest
        rows = sorted(self.rows_dir.glob("step-*.npz"))
        loaded = [np.load(path, allow_pickle=False) for path in rows]
        try:
            # Read only the compact audit projection.  Legacy rows may contain
            # multi-megabyte camera/depth/control tensors; merely inspecting
            # NpzFile.files reads ZIP metadata, while indexing a selected name
            # decompresses only that member.  Thus recovery can seal legacy
            # rows without rereading, stacking, or recompressing full captures.
            file_sets = [set(row.files) for row in loaded]
            if any(not _REQUIRED_LEGACY_AUDIT_ROW_FIELDS <= names for names in file_sets):
                raise ChunkError("row is missing a required compact audit field")
            has_selection_receipts = ["selection_receipts" in names for names in file_sets]
            if any(has_selection_receipts) and not all(has_selection_receipts):
                raise ChunkError("row field mismatch: mixed selection-receipt generations")
            names = list(COMPACT_AUDIT_ROW_FIELDS)
            if not all(has_selection_receipts):
                names.remove("selection_receipts")
            stacked = {name: np.stack([row[name] for row in loaded], axis=0) for name in names}
        finally:
            for row in loaded:
                row.close()
        return self._seal_stacked(stacked, provenance=provenance)

    def _seal_stacked(
        self,
        stacked: dict[str, np.ndarray],
        *,
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_rollout = np.arange(CHUNK_LENGTH, dtype=np.int64)
        if not np.array_equal(stacked["rollout_step_indices"], expected_rollout):
            raise ChunkError("sealed rollout_step_indices must be exactly 0..80")
        if not bool(np.all(stacked["round_ids"] == self.round_id)):
            raise ChunkError("sealed audit contains a row from a different DAgger round")
        if not np.array_equal(stacked["gt_actions"][:, 11:], stacked["homie_actions"]):
            raise ChunkError("sealed teacher actions do not preserve the exact Homie tail")
        teacher_rows = stacked["behavior_source"] == 0
        student_rows = stacked["behavior_source"] == 1
        if not bool(np.all(teacher_rows | student_rows)):
            raise ChunkError("sealed behavior_source contains an invalid code")
        if not np.array_equal(
            stacked["behavior_actions"][teacher_rows], stacked["gt_actions"][teacher_rows]
        ):
            raise ChunkError("teacher-selected behavior does not equal teacher action")
        if not np.array_equal(
            stacked["behavior_actions"][student_rows, 11:],
            stacked["homie_actions"][student_rows],
        ):
            raise ChunkError("student-selected behavior does not preserve the Homie tail")
        self._validate_episode_continuity(stacked)
        prompts = stacked["prompt_per_step"]
        stacked["prompt"] = np.asarray(str(prompts[0])) if bool(np.all(prompts == prompts[0])) else np.asarray("")
        stacked["round_id"] = np.asarray(self.round_id, dtype=np.int64)
        stacked["rollout_window_id"] = np.asarray(self.rollout_window_id)
        temp = self.root / ".sealed.npz.tmp"
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **stacked)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.sealed_path)
        os.chmod(self.sealed_path, 0o400)
        fsync_directory(self.root)
        digest = sha256_file(self.sealed_path)
        manifest = {
            "schema": "roborender.raven.dagger-window.v1",
            "contract_version": CONTRACT_VERSION,
            "run_id": self.run_id,
            "round_id": self.round_id,
            "actor_id": self.actor_id,
            "rollout_window_id": self.rollout_window_id,
            "row_count": CHUNK_LENGTH,
            "artifact_role": "rtx_identity_outcome_audit",
            "learner_complete": False,
            "contains_policy_rgb": False,
            "authoritative_learner_chunk_owner": "h200_render_student_ledger",
            "episode_ids_are_per_step": True,
            "actual_dones_preserved": True,
            "no_previous_round_replay": True,
            "view_order": list(RAVEN_VIEW_ORDER),
            "per_step_identity_fields": [
                "episode_ids",
                "episode_step_indices",
                "rollout_step_indices",
                "round_ids",
                "env_id_per_step",
                "seq_per_step",
                "observation_source",
                "dones",
                "timeouts",
                "rewards",
                "behavior_source",
                "prepare_receipts",
                "capture_digests",
                "selection_ids",
                "commit_digests",
                "commit_receipts",
            ],
            "episode_ids": [str(value) for value in stacked["episode_ids"].tolist()],
            "episode_step_indices": [int(value) for value in stacked["episode_step_indices"].tolist()],
            "round_ids": [int(value) for value in stacked["round_ids"].tolist()],
            "env_ids": [int(value) for value in stacked["env_id_per_step"].tolist()],
            "seqs": [int(value) for value in stacked["seq_per_step"].tolist()],
            "observation_sources": [str(value) for value in stacked["observation_source"].tolist()],
            "dones": [bool(value) for value in stacked["dones"].tolist()],
            "timeouts": [bool(value) for value in stacked["timeouts"].tolist()],
            "rewards": [float(value) for value in stacked["rewards"].tolist()],
            "behavior_sources": [
                "teacher" if int(value) == 0 else "student"
                for value in stacked["behavior_source"].tolist()
            ],
            "prepare_receipts": [str(value) for value in stacked["prepare_receipts"].tolist()],
            "capture_digests": [str(value) for value in stacked["capture_digests"].tolist()],
            "selection_ids": [str(value) for value in stacked["selection_ids"].tolist()],
            "commit_digests": [str(value) for value in stacked["commit_digests"].tolist()],
            "commit_receipts": [str(value) for value in stacked["commit_receipts"].tolist()],
            "path": self.sealed_path.name,
            "bytes": self.sealed_path.stat().st_size,
            "sha256": digest,
            "provenance": dict(provenance),
        }
        atomic_write_json(self.root / "manifest.json", manifest, mode=0o400)
        return manifest

    @staticmethod
    def _validate_episode_continuity(stacked: Mapping[str, np.ndarray]) -> None:
        episode_ids = stacked["episode_ids"]
        episode_steps = stacked["episode_step_indices"]
        dones = stacked["dones"]
        timeouts = stacked["timeouts"]
        sources = stacked["observation_source"]
        for index in range(1, CHUNK_LENGTH):
            if episode_ids[index] == episode_ids[index - 1]:
                if int(episode_steps[index]) != int(episode_steps[index - 1]) + 1:
                    raise ChunkError("episode step index is not contiguous")
                if bool(dones[index - 1]) or bool(timeouts[index - 1]):
                    raise ChunkError("episode continued after a done/timeout transition")
                if sources[index] != sources[index - 1]:
                    raise ChunkError("observation source changed within an episode")
            else:
                if not (bool(dones[index - 1]) or bool(timeouts[index - 1])):
                    raise ChunkError("episode changed without a preceding done/timeout")
                if int(episode_steps[index]) != 0:
                    raise ChunkError("new episode did not reset episode_step_indices to zero")


class MemoryCurrentRoundChunk(CurrentRoundChunk):
    """Compact audit window buffered entirely in RAM until its final seal.

    This class is intentionally used only by the opt-in v4 ephemeral owner.
    It retains exactly :data:`COMPACT_AUDIT_ROW_FIELDS` and never creates a
    per-step ``rows/step-*.npz`` file.  Final seal validation and artifact
    construction are shared with :class:`CurrentRoundChunk`, so the durable
    artifact has the same schema and invariants as the default path.
    """

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        round_id: int,
        actor_id: str,
        rollout_window_id: str,
    ) -> None:
        super().__init__(
            root,
            run_id=run_id,
            round_id=round_id,
            actor_id=actor_id,
            rollout_window_id=rollout_window_id,
        )
        self._memory_rows: list[dict[str, np.ndarray]] = []

    @property
    def count(self) -> int:
        return len(self._memory_rows)

    def append(
        self,
        key: StepKey,
        capture: Capture,
        selected_action: np.ndarray,
        selected_source: str,
        result: TransitionResult,
        audit: Mapping[str, str],
    ) -> None:
        if (
            key.run_id != self.run_id
            or key.round_id != self.round_id
            or key.actor_id != self.actor_id
            or key.rollout_window_id != self.rollout_window_id
        ):
            raise ChunkError("step key does not belong to this current-round rollout window")
        if key.rollout_step_index != self.count:
            raise ChunkError(f"window expected step {self.count}, got {key.rollout_step_index}")
        if self.sealed_path.exists():
            raise ChunkError("rollout window is already sealed")
        arrays = self._compact_row_arrays(
            key,
            capture,
            selected_action,
            selected_source,
            result,
            audit,
        )
        # Captures and actions are owned by the native backend and may reuse
        # their storage on the next vector step.  Copy only the compact audit
        # projection so the in-memory record is immutable.
        self._memory_rows.append(
            {name: np.array(value, copy=True) for name, value in arrays.items()}
        )

    def seal(self, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
        if self.count != CHUNK_LENGTH:
            raise ChunkError(
                f"refusing to seal {self.count}; exactly {CHUNK_LENGTH} committed transitions required"
            )
        if self.sealed_path.exists():
            manifest = json.loads((self.root / "manifest.json").read_text())
            if sha256_file(self.sealed_path) != manifest["sha256"]:
                raise ChunkError("existing seal hash mismatch")
            return manifest
        if any(set(row) != set(COMPACT_AUDIT_ROW_FIELDS) for row in self._memory_rows):
            raise ChunkError("in-memory row field mismatch")
        stacked = {
            name: np.stack([row[name] for row in self._memory_rows], axis=0)
            for name in COMPACT_AUDIT_ROW_FIELDS
        }
        manifest = self._seal_stacked(stacked, provenance=provenance)
        self._memory_rows.clear()
        return manifest
