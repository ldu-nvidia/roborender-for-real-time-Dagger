from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from .boundary import (
    H200_COMMIT_PATH,
    H200_PREPARE_PATH,
    build_h200_commit,
    build_h200_prepare,
    build_owner_commit,
    validate_h200_commit_ack,
    validate_h200_selection,
)
from .contracts import CONTRACT_VERSION, ROUND_BETAS, StepKey, canonical_json, sha256_bytes
from .durable import atomic_write_json
from .http_transport import JsonTransport


class OwnerClient(Protocol):
    def prepare_step(self, request: Mapping[str, Any]) -> dict[str, Any]: ...
    def commit_step(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


class StepOrchestrator:
    """Durable retry state machine around local owner and H200 boundary."""

    def __init__(self, owner: OwnerClient, h200: JsonTransport, runtime_root: Path):
        self.owner = owner
        self.h200 = h200
        self.runtime_root = runtime_root
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _idem(prefix: str, key: StepKey) -> str:
        return f"{prefix}-{sha256_bytes(canonical_json(key.as_dict()))[:48]}"

    def run_step(self, key: StepKey) -> dict[str, Any]:
        state_path = (
            self.runtime_root
            / key.run_id
            / f"D{key.round_id}"
            / key.actor_id
            / key.rollout_window_id
            / f"step-{key.rollout_step_index:03d}.json"
        )
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["step_key"] != key.as_dict():
                raise RuntimeError("orchestrator receipt path collision")
        else:
            state = {"contract_version": CONTRACT_VERSION, "phase": "INIT", "step_key": key.as_dict()}
            atomic_write_json(state_path, state)
        if state["phase"] == "COMPLETE":
            return dict(state)
        prepare_request = {
            "contract_version": CONTRACT_VERSION,
            "request_id": self._idem("rtx-prepare", key),
            "step_key": key.as_dict(),
            "beta": ROUND_BETAS[key.round_id],
        }
        local_prepare = None
        if state["phase"] in ("INIT", "LOCAL_PREPARED"):
            local_prepare = self.owner.prepare_step(prepare_request)
        if state["phase"] == "INIT" and local_prepare is not None:
            state.update(
                {
                    "phase": "LOCAL_PREPARED",
                    "prepare_receipt": local_prepare["prepare_receipt"],
                    "capture_digest": local_prepare["capture_digest"],
                }
            )
            atomic_write_json(state_path, state)
        if state["phase"] == "LOCAL_PREPARED":
            assert local_prepare is not None
            remote_request = build_h200_prepare(local_prepare, self._idem("h200-prepare", key))
            raw_selection = self.h200.post(H200_PREPARE_PATH, remote_request, max_response_bytes=1 << 20)
            selection = validate_h200_selection(
                raw_selection, local_prepare, expected_idempotency_key=remote_request["idempotency_key"]
            )
            selection.pop("_selected_action_array", None)
            state.update({"phase": "H200_SELECTED", "selection": selection})
            atomic_write_json(state_path, state)
        selection = state["selection"]
        if state["phase"] == "H200_SELECTED":
            local_commit = self.owner.commit_step(build_owner_commit(selection))
            state.update({"phase": "LOCAL_COMMITTED", "local_commit": local_commit})
            atomic_write_json(state_path, state)
        if state["phase"] == "LOCAL_COMMITTED":
            remote_commit = build_h200_commit(
                selection, state["local_commit"], self._idem("h200-commit", key)
            )
            ack = self.h200.post(H200_COMMIT_PATH, remote_commit, max_response_bytes=1 << 20)
            validate_h200_commit_ack(ack, remote_commit)
            state.update({"phase": "COMPLETE", "remote_ack": ack})
            atomic_write_json(state_path, state)
        return dict(state)
