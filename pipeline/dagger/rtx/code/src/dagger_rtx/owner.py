from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .backend import Capture, RavenBackend, TransitionResult
from .chunks import CHUNK_LENGTH, CurrentRoundChunk
from .contracts import (
    CONTRACT_VERSION,
    CAPTURE_METADATA_KEYS,
    MODEL_VIEW_MAP,
    PINNED_RAVEN_COMMIT,
    PINNED_RAVEN_ROOT,
    RAVEN_CAPTURE_VIEWS,
    RAVEN_VIEW_ORDER,
    ArrayRule,
    ContractError,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    decode_array,
    encode_array,
    encode_arrays,
    payload_digest,
    require_identifier,
    require_sha256,
    sha256_bytes,
    validate_action,
    anchor_rgb_digest,
    validate_array,
    validate_beta,
)
from .durable import DurableJournal, atomic_write_json


class OwnerConflict(RuntimeError):
    pass


class CrashUncertain(RuntimeError):
    pass


class IsaacTransitionOwner:
    """Sole owner of irreversible Raven environment transitions.

    The durable COMMITTING state is written before `step_once`. A restart that
    observes it enters CRASH_UNCERTAIN and never guesses whether env.step ran.
    """

    def __init__(
        self,
        backend: RavenBackend,
        runtime_root: Path,
        staging_root: Path,
        *,
        shape_profile: RavenShapeProfile | None = None,
        static_provenance: Mapping[str, Any] | None = None,
        clock_ns=time.time_ns,
    ):
        self.backend = backend
        self.runtime_root = runtime_root
        self.staging_root = staging_root
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.runtime_root / "owner_state.json"
        self.journal = DurableJournal(self.runtime_root / "owner_events.jsonl")
        self.profile = shape_profile or RavenShapeProfile()
        self.static_provenance = dict(static_provenance or {})
        required_provenance = {
            "raven_root",
            "raven_commit",
            "raven_videogen_trainer_sha256",
            "raven_teacher_homie_trainer_sha256",
            "isaac_capture_sha256",
            "image_dagger_config_sha256",
            "teacher_checkpoint_sha256",
            "homie_walk_sha256",
            "homie_stand_sha256",
        }
        if set(self.static_provenance) != required_provenance:
            raise ContractError(f"static provenance keys must be exactly {sorted(required_provenance)}")
        if self.static_provenance["raven_root"] != PINNED_RAVEN_ROOT:
            raise ContractError("static provenance raven_root is not pinned BC_render")
        if self.static_provenance["raven_commit"] != PINNED_RAVEN_COMMIT:
            raise ContractError("static provenance raven_commit mismatch")
        for name in required_provenance - {"raven_root", "raven_commit"}:
            require_sha256(self.static_provenance[name], name)
        self.clock_ns = clock_ns
        self._lock = threading.RLock()
        self._captures: dict[str, Capture] = {}
        had_durable_state = self.state_path.exists()
        self.state = self._load_or_new()
        # The simulator, policy recurrent state, episode RNG, and Raven caches
        # are process-local. Without a backend-specific restore attestation, no
        # active phase is safe to resume merely from the service JSON.
        unsafe_restart_phases = {"READY", "PREPARED", "COMMITTING", "BINDING_EPISODE", "SEALED"}
        if had_durable_state and self.state["phase"] in unsafe_restart_phases:
            prior = self.state["phase"]
            self.state["phase"] = "CRASH_UNCERTAIN"
            self.state["uncertain_reason"] = (
                f"restart observed active {prior} state without a native Raven simulator restore attestation"
            )
            self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"])
            self._persist()

    def _load_or_new(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("contract_version") != CONTRACT_VERSION:
                raise RuntimeError("owner state contract version mismatch")
            return state
        return {
            "contract_version": CONTRACT_VERSION,
            "phase": "EMPTY",
            "window": None,
            "active_episode": None,
            "episode_bindings": [],
            "next_rollout_step": 0,
            "last_prepare": None,
            "last_commit": None,
            "sealed": None,
            "session": None,
        }

    def _persist(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _event(self, kind: str, detail: Any, key: StepKey | None = None) -> None:
        self.journal.append(
            {
                "contract_version": CONTRACT_VERSION,
                "time_ns": int(self.clock_ns()),
                "event": kind,
                "step_key": key.as_dict() if key else None,
                "detail": detail,
            }
        )

    def _assert_operable(self) -> None:
        if self.state["phase"] == "CRASH_UNCERTAIN":
            raise CrashUncertain(self.state.get("uncertain_reason", "environment transition is uncertain"))

    def begin_window(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._assert_operable()
            self._check_version(request)
            self._require_exact_keys(request, {"contract_version", "request_id", "step_key"}, "begin_window")
            key = StepKey.from_mapping(request.get("step_key"))
            if key.rollout_step_index != 0:
                raise ContractError("a rollout window must begin at rollout_step_index zero")
            request_id = require_identifier(request.get("request_id"), "request_id")
            digest = sha256_bytes(canonical_json({"step_key": key.as_dict(), "request_id": request_id}))
            existing = self.state.get("window")
            if existing and existing["begin_digest"] == digest:
                return dict(existing["response"])
            prior_phase = self.state["phase"]
            if prior_phase not in ("EMPTY", "SEALED"):
                raise OwnerConflict("a rollout window is already active")
            next_session = self._validated_session(key)
            if prior_phase == "EMPTY":
                if key.episode_step_index != 0:
                    raise ContractError("the first window must bind an episode at episode_step_index zero")
                binding = self._bind_episode_durably(key)
            elif self.state.get("needs_episode_bind"):
                active = self.state["active_episode"]
                if key.episode_id == active["episode_id"] or key.episode_step_index != 0:
                    raise OwnerConflict("a sealed window ending in done/timeout must start a new episode")
                binding = self._bind_episode_durably(key)
            else:
                active = self.state["active_episode"]
                if (
                    key.episode_id != active["episode_id"]
                    or key.episode_step_index != active["next_episode_step"]
                ):
                    raise OwnerConflict("new window must continue the active episode without resampling")
                binding = dict(active["assignment"])
            response = {
                "contract_version": CONTRACT_VERSION,
                "step_key": key.as_dict(),
                "request_id": request_id,
                "rollout_window_id": key.rollout_window_id,
                "episode_assignment": binding,
            }
            self.state.update(
                {
                    "phase": "READY",
                    "window": {"key": key.as_dict(), "begin_digest": digest, "response": response},
                    "active_episode": {
                        "episode_id": key.episode_id,
                        "next_episode_step": key.episode_step_index,
                        "assignment": binding,
                    },
                    "episode_bindings": [self._manifest_binding(key, binding)],
                    "next_rollout_step": 0,
                    "last_prepare": None,
                    "last_commit": None,
                    "sealed": None,
                    "needs_episode_bind": False,
                    "session": next_session,
                }
            )
            self._event("WINDOW_BEGIN", response, key)
            self._persist()
            return response

    def _bind_episode_durably(self, key: StepKey) -> dict[str, Any]:
        self.state["phase"] = "BINDING_EPISODE"
        self.state["binding_intent"] = key.as_dict()
        self._persist()
        self._event("EPISODE_BIND_INTENT", {}, key)
        try:
            native = dict(self.backend.bind_episode(key=key))
            use_raytraced = native.get("use_raytraced")
            if not isinstance(use_raytraced, bool):
                raise ContractError("pinned Raven hook must return boolean use_raytraced")
            anchor_sha = require_sha256(native.get("anchor_sha256"), "native anchor_sha256")
            if native.get("raven_source") != PINNED_RAVEN_ROOT:
                raise ContractError("episode hook is not bound to the pinned BC_render root")
            if native.get("raven_commit") != PINNED_RAVEN_COMMIT:
                raise ContractError("episode hook is not bound to Raven commit 94d7985")
            result = {
                "use_raytraced": use_raytraced,
                "observation_source": "raytraced" if use_raytraced else "generated_controls",
                "anchor_sha256": anchor_sha,
                "assignment_source": "pinned_raven_id_sample_anchor",
                "raven_source": native.get("raven_source"),
                "raven_commit": native.get("raven_commit"),
            }
            self._event("EPISODE_BOUND", result, key)
            return result
        except BaseException as exc:
            self.state["phase"] = "CRASH_UNCERTAIN"
            self.state["uncertain_reason"] = f"native episode assignment raised {type(exc).__name__}; RNG consumption is uncertain"
            self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"], key)
            self._persist()
            raise CrashUncertain(self.state["uncertain_reason"]) from exc

    def _bind_reset_episode_if_needed(self, key: StepKey) -> None:
        if not self.state.get("needs_episode_bind"):
            return
        active = self.state["active_episode"]
        if key.episode_id == active["episode_id"] or key.episode_step_index != 0:
            raise OwnerConflict("after done/timeout, next step must bind a new episode at episode_step_index zero")
        binding = self._bind_episode_durably(key)
        self.state["active_episode"] = {
            "episode_id": key.episode_id,
            "next_episode_step": 0,
            "assignment": binding,
        }
        self.state["episode_bindings"].append(self._manifest_binding(key, binding))
        self.state["needs_episode_bind"] = False
        self.state["binding_intent"] = None
        self.state["phase"] = "READY"
        self._persist()

    def _validate_capture(self, capture: Capture, key: StepKey) -> None:
        if capture.key != key:
            raise ContractError("capture key mismatch")
        rules = self.profile.capture_rules()
        missing = sorted(set(rules) - set(capture.arrays))
        if missing:
            raise ContractError(f"capture is missing required arrays: {missing}")
        optional_arrays = {f"raytraced_rgb__{view}" for view in RAVEN_CAPTURE_VIEWS}
        unknown = sorted(set(capture.arrays) - set(rules) - optional_arrays)
        if unknown:
            raise ContractError(f"capture has unknown arrays: {unknown}")
        for name, rule in rules.items():
            validate_array(name, capture.arrays[name], rule)
        if not np.array_equal(capture.arrays["teacher_action"][11:], capture.arrays["homie_action"]):
            raise ContractError("teacher_action[11:] must exactly equal the RTX Homie action")
        expected_anchor = self.state["active_episode"]["assignment"]["anchor_sha256"]
        if anchor_rgb_digest(capture.arrays) != expected_anchor:
            raise ContractError("fixed anchor RGB bytes do not match the episode anchor_sha256")
        if tuple(capture.metadata.get("view_order", ())) != RAVEN_VIEW_ORDER:
            raise ContractError(f"online wire view_order must be {RAVEN_VIEW_ORDER}")
        if set(capture.metadata) != CAPTURE_METADATA_KEYS:
            raise ContractError(f"capture metadata keys must be exactly {sorted(CAPTURE_METADATA_KEYS)}")
        if capture.metadata.get("model_view_map") != MODEL_VIEW_MAP:
            raise ContractError("capture model_view_map does not match the online Image3F contract")
        if capture.metadata.get("depth_units") != "meter":
            raise ContractError("metric depth units must be meter")
        if capture.metadata.get("depth_control_encoding") != "raven_h5_v4":
            raise ContractError("depth control must come from pinned Raven h5_v4")
        if capture.metadata.get("robot_control_encoding") != "textured_robot_rgb_on_black":
            raise ContractError("robot control must be textured robot RGB on black")
        expected_source = self.state["active_episode"]["assignment"]["observation_source"]
        if capture.metadata.get("observation_source") != expected_source:
            raise ContractError("capture changed the fixed per-episode observation source")
        ray_names = [f"raytraced_rgb__{view}" for view in RAVEN_CAPTURE_VIEWS]
        if expected_source == "raytraced":
            for name in ray_names:
                validate_array(
                    name,
                    capture.arrays.get(name),
                    ArrayRule((self.profile.policy_h, self.profile.policy_w, 3), "float32", True),
                )
                values = capture.arrays[name]
                if bool((values < 0).any()) or bool((values > 1).any()):
                    raise ContractError(f"{name} must preserve native Raven float32 values in [0,1]")
        elif any(name in capture.arrays for name in ray_names):
            raise ContractError("generated-controls episodes must not carry unused raytraced policy RGB")
        if not isinstance(capture.metadata.get("prompt"), str) or not capture.metadata["prompt"]:
            raise ContractError("capture prompt is required")
        for name in ("env_id", "seq"):
            value = capture.metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"capture {name} must be an integer")

    def prepare_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._assert_operable()
            self._check_version(request)
            self._require_exact_keys(
                request, {"contract_version", "request_id", "step_key", "beta"}, "prepare_step"
            )
            key = StepKey.from_mapping(request.get("step_key"))
            request_id = require_identifier(request.get("request_id"), "request_id")
            beta = validate_beta(key.round_id, request.get("beta"))
            request_digest = sha256_bytes(
                canonical_json({"step_key": key.as_dict(), "request_id": request_id, "beta": beta})
            )
            last = self.state.get("last_prepare")
            if last and last["step_key"] == key.as_dict():
                if last["request_digest"] != request_digest:
                    raise OwnerConflict("prepare retry changed request identity or beta")
                return json.loads((self.runtime_root / last["response_path"]).read_text(encoding="utf-8"))
            if self.state["phase"] != "READY":
                raise OwnerConflict(f"owner phase {self.state['phase']} cannot prepare")
            self._assert_window_key(key)
            self._bind_reset_episode_if_needed(key)
            self._assert_episode_key(key)
            capture = self.backend.capture_current(key=key)
            self._validate_capture(capture, key)
            metadata = dict(capture.metadata)
            metadata.update(
                {
                    "contract_version": CONTRACT_VERSION,
                    "step_key": key.as_dict(),
                    "beta": beta,
                    "episode_assignment": self.state["active_episode"]["assignment"],
                }
            )
            capture_digest = payload_digest(metadata, capture.arrays)
            receipt_id = sha256_bytes(canonical_json({"request_digest": request_digest, "capture_digest": capture_digest}))
            response = {
                "contract_version": CONTRACT_VERSION,
                "step_key": key.as_dict(),
                "request_id": request_id,
                "prepare_receipt": receipt_id,
                "capture_digest": capture_digest,
                "metadata": metadata,
                "arrays": encode_arrays(capture.arrays),
            }
            response_rel = (
                Path("receipts")
                / key.run_id
                / f"D{key.round_id}"
                / key.actor_id
                / key.rollout_window_id
                / f"prepare-{key.rollout_step_index:03d}.json"
            )
            atomic_write_json(self.runtime_root / response_rel, response)
            self._captures[key.token] = capture
            self.state["phase"] = "PREPARED"
            self.state["last_prepare"] = {
                "step_key": key.as_dict(),
                "request_digest": request_digest,
                "prepare_receipt": receipt_id,
                "capture_digest": capture_digest,
                "response_path": str(response_rel),
            }
            self._event("STEP_PREPARED", {"request_id": request_id, "prepare_receipt": receipt_id}, key)
            self._persist()
            return response

    def commit_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._assert_operable()
            self._check_version(request)
            self._require_exact_keys(
                request,
                {
                    "contract_version", "step_key", "prepare_receipt", "selection_id",
                    "selected_source", "beta", "selected_action",
                },
                "commit_step",
            )
            key = StepKey.from_mapping(request.get("step_key"))
            action = decode_array(request.get("selected_action"))
            validate_action(action)
            selection_id = require_identifier(request.get("selection_id"), "selection_id")
            source = request.get("selected_source")
            if source not in ("teacher", "student"):
                raise ContractError("selected_source must be teacher or student")
            beta = validate_beta(key.round_id, request.get("beta"))
            commit_body = {
                "step_key": key.as_dict(),
                "prepare_receipt": request.get("prepare_receipt"),
                "selection_id": selection_id,
                "selected_source": source,
                "beta": beta,
                "selected_action": encode_array(action),
            }
            commit_digest = sha256_bytes(canonical_json(commit_body))
            last_commit = self.state.get("last_commit")
            if last_commit and last_commit["step_key"] == key.as_dict():
                if last_commit["commit_digest"] != commit_digest:
                    raise OwnerConflict("commit retry changed action or selection")
                return dict(last_commit["response"])
            prepared = self.state.get("last_prepare")
            if not prepared or prepared["step_key"] != key.as_dict():
                raise OwnerConflict("commit has no matching prepared step")
            if request.get("prepare_receipt") != prepared["prepare_receipt"]:
                raise OwnerConflict("prepare receipt mismatch")
            if self.state["phase"] != "PREPARED":
                raise OwnerConflict(f"owner phase {self.state['phase']} cannot commit")
            capture = self._capture_for(key)
            homie = capture.arrays["homie_action"]
            teacher = capture.arrays["teacher_action"]
            if source == "teacher" and not np.array_equal(action, teacher):
                raise ContractError("teacher-selected behavior must exactly equal teacher_action")
            if source == "student" and not np.array_equal(action[11:], homie):
                raise ContractError("student-selected behavior tail must exactly equal RTX homie_action")
            self.state["phase"] = "COMMITTING"
            self.state["commit_intent"] = {
                "step_key": key.as_dict(), "commit_digest": commit_digest, "selection_id": selection_id
            }
            self._persist()
            self._event("COMMIT_INTENT", self.state["commit_intent"], key)
            try:
                result = self.backend.step_once(action)
                self._validate_result(result)
                result_dict = self._result_dict(result)
                commit_receipt = sha256_bytes(
                    canonical_json({"commit_digest": commit_digest, "result": result_dict})
                )
                chunk = self._chunk_for(key)
                chunk.append(
                    key,
                    capture,
                    action,
                    source,
                    result,
                    audit={
                        "prepare_receipt": prepared["prepare_receipt"],
                        "capture_digest": prepared["capture_digest"],
                        "selection_id": selection_id,
                        "commit_digest": commit_digest,
                        "commit_receipt": commit_receipt,
                    },
                )
                response = {
                    "contract_version": CONTRACT_VERSION,
                    "step_key": key.as_dict(),
                    "commit_receipt": commit_receipt,
                    "result": result_dict,
                    "committed_steps": key.rollout_step_index + 1,
                    "sealed": False,
                }
                if key.rollout_step_index + 1 == CHUNK_LENGTH:
                    seal = chunk.seal(
                        provenance={
                            **self.static_provenance,
                            "episode_bindings": self.state["episode_bindings"],
                        }
                    )
                    response.update({"sealed": True, "seal": seal})
                    next_phase = "SEALED"
                    self.state["sealed"] = seal
                else:
                    next_phase = "READY"
                ended = result.done or result.timeout
                self.state.update(
                    {
                        "phase": next_phase,
                        "next_rollout_step": key.rollout_step_index + 1,
                        "needs_episode_bind": bool(ended),
                        "commit_intent": None,
                        "last_commit": {
                            "step_key": key.as_dict(), "commit_digest": commit_digest, "response": response
                        },
                    }
                )
                if not ended:
                    self.state["active_episode"]["next_episode_step"] = key.episode_step_index + 1
                self._event("STEP_COMMITTED", response, key)
                self._persist()
                self._captures.pop(key.token, None)
                return response
            except BaseException as exc:
                if isinstance(exc, CrashUncertain):
                    raise
                self.state["phase"] = "CRASH_UNCERTAIN"
                self.state["uncertain_reason"] = (
                    f"post-intent operation raised {type(exc).__name__}; env.step may have completed"
                )
                self._event("CRASH_UNCERTAIN", self.state["uncertain_reason"], key)
                self._persist()
                raise CrashUncertain(self.state["uncertain_reason"]) from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def _capture_for(self, key: StepKey) -> Capture:
        capture = self._captures.get(key.token)
        if capture is not None:
            return capture
        prepared = self.state["last_prepare"]
        response = json.loads((self.runtime_root / prepared["response_path"]).read_text(encoding="utf-8"))
        arrays = {name: decode_array(value) for name, value in response["arrays"].items()}
        return Capture(key=key, metadata=dict(response["metadata"]), arrays=arrays)

    def _chunk_for(self, key: StepKey) -> CurrentRoundChunk:
        return CurrentRoundChunk(
            self.staging_root,
            run_id=key.run_id,
            round_id=key.round_id,
            actor_id=key.actor_id,
            rollout_window_id=key.rollout_window_id,
        )

    def _validated_session(self, key: StepKey) -> dict[str, Any]:
        """Keep one durable actor/run lineage and monotonic D1..D4 progression."""
        session = self.state.get("session")
        if session is None:
            return {
                "run_id": key.run_id,
                "actor_id": key.actor_id,
                "highest_round_id": key.round_id,
            }
        if key.run_id != session["run_id"] or key.actor_id != session["actor_id"]:
            raise OwnerConflict("RTX transition owner is pinned to one run_id and actor_id")
        prior = int(session["highest_round_id"])
        if key.round_id < prior or key.round_id > prior + 1:
            raise OwnerConflict("rounds must be monotonic and advance by at most one")
        return {
            "run_id": key.run_id,
            "actor_id": key.actor_id,
            "highest_round_id": max(prior, key.round_id),
        }

    def _assert_window_key(self, key: StepKey) -> None:
        if not self.state.get("window"):
            raise OwnerConflict("no rollout window has begun")
        root = StepKey.from_mapping(self.state["window"]["key"])
        lhs = (key.run_id, key.round_id, key.actor_id, key.rollout_window_id)
        rhs = (root.run_id, root.round_id, root.actor_id, root.rollout_window_id)
        if lhs != rhs:
            raise OwnerConflict("step belongs to a different run/round/actor/window")
        if key.rollout_step_index != self.state["next_rollout_step"]:
            raise OwnerConflict(f"expected rollout step {self.state['next_rollout_step']}, got {key.rollout_step_index}")

    def _assert_episode_key(self, key: StepKey) -> None:
        active = self.state["active_episode"]
        if key.episode_id != active["episode_id"]:
            raise OwnerConflict("episode changed without a preceding done/timeout")
        if key.episode_step_index != active["next_episode_step"]:
            raise OwnerConflict(
                f"expected episode step {active['next_episode_step']}, got {key.episode_step_index}"
            )

    @staticmethod
    def _validate_result(result: TransitionResult) -> None:
        if not isinstance(result, TransitionResult):
            raise ContractError("backend returned an invalid transition result")
        if not np.isfinite(result.reward):
            raise ContractError("reward is not finite")
        if not isinstance(result.done, bool) or not isinstance(result.timeout, bool):
            raise ContractError("done and timeout must be bool")

    @staticmethod
    def _result_dict(result: TransitionResult) -> dict[str, Any]:
        return {"reward": float(result.reward), "done": result.done, "timeout": result.timeout, "metadata": result.metadata}

    @staticmethod
    def _require_exact_keys(request: Mapping[str, Any], expected: set[str], route: str) -> None:
        if not isinstance(request, Mapping) or set(request) != expected:
            raise ContractError(f"{route} keys must be exactly {sorted(expected)}")

    @staticmethod
    def _manifest_binding(key: StepKey, binding: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "episode_id": key.episode_id,
            "first_rollout_step_index": key.rollout_step_index,
            "use_raytraced": binding["use_raytraced"],
            "observation_source": binding["observation_source"],
            "anchor_sha256": binding["anchor_sha256"],
        }

    @staticmethod
    def _check_version(request: Mapping[str, Any]) -> None:
        if not isinstance(request, Mapping) or request.get("contract_version") != CONTRACT_VERSION:
            raise ContractError(f"contract_version must be {CONTRACT_VERSION}")
